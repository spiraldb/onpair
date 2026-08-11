// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! SIMD prefilter for substring search.
//!
//! Answering `LIKE '%pattern%'` exactly means checking every row — e.g. stepping
//! the token-level KMP automaton of [`contains`](super::contains()) over its
//! codes. This module trims that per-row work down to the rows that *can* match.
//! It compiles a **sound probe cover** from the pattern — dictionary token ids
//! and id ranges chosen so that *any* row containing the pattern holds at least
//! one probe token — then scans the flat code stream to collect a **superset** of
//! the matching rows ([`prefilter_candidates`]).
//!
//! The prefilter stops there: it hands back the candidate rows and leaves the
//! exact check to the caller. Verifying only those survivors with `contains` or
//! another exact substring test recovers the precise answer, since a sound cover
//! drops no true match.
//!
//! # Soundness
//! Every occurrence of the pattern in an encoded row falls into one of two cases,
//! and the cover covers both:
//!
//! * **One token contains the whole pattern.** Its id is added unconditionally
//!   (only reachable when `pattern.len() <= MAX_TOKEN_SIZE`).
//! * **The occurrence crosses at least one token boundary.** Then it begins at
//!   some feasible first-token alignment `k`, after which greedy parsing of
//!   `pattern[k..]` is deterministic. Every such layout is one path through the
//!   alignment DAG, and the cover is a cut of that DAG — so whichever layout the
//!   occurrence takes, it runs into a probe.
//!
//! # Shape
//! * `frequency` — the per-column selectivity index the compiler reads, the one
//!   part of this module the caller builds and owns ([`TokenFrequencyIndex`]).
//! * `graph` — pattern to alignment DAG: every layout of the pattern across
//!   token boundaries, as one graph whose cuts are exactly the sound covers.
//! * `mincut` — the cheapest such cut, by max-flow over the split DAG.
//! * `plan` — the two of them end to end: pattern in, cover out, minus the ids
//!   the code stream never uses.
//! * `cover` — the cover itself, in both the shapes the scan wants.
//! * `scan` — the vector kernels, and the refusal that keeps a wide cover off a
//!   slow path.

mod cover;
mod frequency;
mod graph;
mod mincut;
mod plan;
mod scan;

#[cfg(test)]
mod tests;

pub use frequency::{TokenFrequencyIndex, TokenFrequencyIndexError, build_token_frequency_index};

use crate::core::dictionary::{CompactDictionaryView, DictionaryView};
use crate::core::offset::Offset;
use crate::core::types::Token;

/// Reason the SIMD substring prefilter refused to run.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum PrefilterError {
    /// The crate has no SIMD prefilter kernel for this target architecture.
    UnsupportedArchitecture,
    /// The compiled probe cover requires too many comparisons for the SIMD path.
    ProbeCoverTooWide,
    /// The frequency index does not describe the supplied dictionary/code stream.
    IndexMismatch,
}

impl std::fmt::Display for PrefilterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::UnsupportedArchitecture => {
                "the substring prefilter has no SIMD kernel for this architecture"
            }
            Self::ProbeCoverTooWide => "the substring prefilter probe cover is too wide for SIMD",
            Self::IndexMismatch => {
                "the token frequency index does not match the dictionary/code stream"
            }
        })
    }
}

impl std::error::Error for PrefilterError {}

/// Append the ascending rows that may contain `pattern`.
///
/// The result is a **sound superset**. Verify the survivors with an exact check,
/// such as a [`ContainsTable`](super::ContainsTable) passed to
/// [`contains`](super::contains()), to recover the precise answer.
///
/// `codes` is the row-concatenated code stream, `row_offsets` contains its `R + 1`
/// row delimiters, and `frequencies` must have been built for this `codes` stream
/// and the supplied dictionary. An empty pattern appends every row without
/// requiring SIMD support.
///
/// The function does not silently fall back to a full scalar scan. If the target
/// has no SIMD implementation, the probe cover is too expensive for SIMD, or the
/// index shape does not match the inputs, it returns an error without modifying
/// `out`.
///
/// A pattern whose cover names only tokens absent from `codes` is the exception:
/// no row can match, so it appends nothing and succeeds on any target, SIMD or
/// not.
///
/// # Precondition
/// `dict` is conformant: **sorted** (strict bytewise-lexicographic order) and
/// **complete** (all 256 single-byte tokens present). These properties are
/// guaranteed for any dictionary trained by [`Parser::train`](crate::Parser::train)
/// or passed through [`CompactDictionary::validate`](crate::CompactDictionary::validate).
/// Planning binary-searches the dictionary (see
/// [`prefix_range`](super::prefix_range)), so an unsorted dictionary yields a
/// cover that can miss matching tokens — the result stops being a superset. See
/// [`crate::search`] for the general search precondition.
///
/// # Errors
/// Returns [`PrefilterError::UnsupportedArchitecture`] when no SIMD kernel is
/// available, [`PrefilterError::ProbeCoverTooWide`] when the cover exceeds the
/// SIMD comparison budget, or [`PrefilterError::IndexMismatch`] when the index's
/// token count or total frequency does not match the inputs.
pub fn prefilter_candidates<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pattern: &[u8],
    dict: CompactDictionaryView<'_>,
    frequencies: &TokenFrequencyIndex,
    out: &mut Vec<usize>,
) -> Result<(), PrefilterError> {
    let n = row_offsets.len().saturating_sub(1);
    if pattern.is_empty() {
        out.extend(0..n);
        return Ok(());
    }
    if frequencies.num_tokens() != dict.num_tokens()
        || frequencies.total_frequency() as usize != codes.len()
    {
        return Err(PrefilterError::IndexMismatch);
    }

    let pf = plan::plan(dict, pattern, frequencies);
    scan::scan(codes, row_offsets, &pf, out)
}
