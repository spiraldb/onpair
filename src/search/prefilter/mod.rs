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
//! * `plan` — the two of them end to end: pattern in, normalized cover out,
//!   minus the ids the code stream never uses.
//! * `cover` — the cover itself, in both the shapes the scan wants.
//! * `scan` — the vector kernels. Profitability stays outside execution.

mod cover;
mod frequency;
mod graph;
mod mincut;
mod plan;
mod scan;

#[cfg(test)]
mod tests;

pub use cover::ProbeCover;
pub use frequency::{TokenFrequencyIndex, TokenFrequencyIndexError, build_token_frequency_index};

use crate::core::dictionary::CompactDictionaryView;
use crate::core::offset::Offset;
use crate::core::types::Token;

/// Reason SIMD prefilter execution could not proceed.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum PrefilterError {
    /// The crate has no SIMD prefilter kernel for this target architecture.
    UnsupportedArchitecture,
}

impl std::fmt::Display for PrefilterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::UnsupportedArchitecture => {
                "the substring prefilter has no SIMD kernel for this architecture"
            }
        })
    }
}

impl std::error::Error for PrefilterError {}

/// The normalized probe cover selected for a pattern and its frequency.
///
/// Points and ranges are disjoint, so each covered token occurrence is counted
/// exactly once.
#[derive(Debug, Clone)]
pub struct PrefilterAnalysis {
    probe_cover: ProbeCover,
    covered_frequency: u32,
    total_frequency: u32,
}

impl PrefilterAnalysis {
    /// The normalized checks the SIMD prefilter can execute.
    pub fn probe_cover(&self) -> &ProbeCover {
        &self.probe_cover
    }

    /// Number of code positions whose token is covered by the probes.
    pub fn covered_frequency(&self) -> u32 {
        self.covered_frequency
    }

    /// Fraction of code positions whose token is covered by the probes.
    ///
    /// Returns `0.0` when the indexed code stream is empty.
    pub fn covered_fraction(&self) -> f64 {
        if self.total_frequency == 0 {
            0.0
        } else {
            f64::from(self.covered_frequency) / f64::from(self.total_frequency)
        }
    }

    /// Whether sparse code hits should use binary search to find their rows.
    fn use_sparse_row_mapping(&self) -> bool {
        const MAX_COVERED_FRACTION: f64 = 0.0001;
        self.covered_fraction() < MAX_COVERED_FRACTION
    }
}

/// Return whether the default empirical policy expects prefiltering to beat a
/// bulk-decode fallback for this pattern.
///
/// The policy uses only information available after [`analyze_prefilter`]:
/// pattern length, the number of SIMD comparisons in the normalized cover, and
/// the fraction of code positions covered. A point costs one comparison and an
/// inclusive range costs two. It selects the prefilter when all of these hold:
///
/// * `pattern_len >= 4`;
/// * at most 16 SIMD comparisons; and
/// * covered-code fraction strictly below 1%.
///
/// Empty covers are accepted for patterns of at least four bytes: they prove
/// that no encoded row can match and execute no SIMD scan. Patterns shorter
/// than four bytes are conservatively rejected even when a particular absent
/// pattern would produce an empty cover.
///
/// This is a performance hint, not a correctness requirement. It was
/// calibrated on selective `contains` workloads through 5% row selectivity on
/// AArch64, against both OnPair bulk decode and bulk decode followed by a
/// full-haystack `memmem`. Callers with materially different columns, batch
/// sizes, or architectures may choose their own policy. This function does not
/// execute or bypass the prefilter itself.
pub fn prefilter_is_likely_profitable(pattern_len: usize, analysis: &PrefilterAnalysis) -> bool {
    const MIN_PATTERN_LEN: usize = 4;
    const MAX_SIMD_COMPARISONS: usize = 16;
    const MAX_COVERED_FRACTION: f64 = 0.01;

    let cover = analysis.probe_cover();
    let comparisons = cover
        .points()
        .len()
        .saturating_add(cover.ranges().len().saturating_mul(2));
    pattern_len >= MIN_PATTERN_LEN
        && comparisons <= MAX_SIMD_COMPARISONS
        && analysis.covered_fraction() < MAX_COVERED_FRACTION
}

/// Analyze `pattern` and return its normalized minimum-cut probe cover.
///
/// This function constructs the checks and reports their frequency; the caller
/// decides whether executing them is profitable.
///
/// # Precondition
/// `dict` is conformant: sorted, complete, and unique. These properties are
/// guaranteed for a dictionary trained by [`Parser::train`](crate::Parser::train)
/// or passed through [`CompactDictionary::validate`](crate::CompactDictionary::validate).
/// `frequencies` must describe the dictionary and code stream that the returned
/// cover may later scan.
///
/// # Panics
/// Panics when `pattern` is empty. The empty pattern matches every row and
/// should bypass prefilter analysis.
pub fn analyze_prefilter(
    pattern: &[u8],
    dict: CompactDictionaryView<'_>,
    frequencies: &TokenFrequencyIndex,
) -> PrefilterAnalysis {
    assert!(
        !pattern.is_empty(),
        "the empty pattern matches every row and needs no prefilter"
    );
    let probe_cover = plan::plan(dict, pattern, frequencies);
    let covered_frequency = probe_cover
        .points
        .iter()
        .map(|&token| frequencies.frequency(token))
        .chain(
            probe_cover
                .ranges
                .iter()
                .map(|&range| frequencies.range_frequency(range)),
        )
        .sum();
    PrefilterAnalysis {
        probe_cover,
        covered_frequency,
        total_frequency: frequencies.total_frequency(),
    }
}

/// Execute `analysis` and append the ascending rows that contain a covered code.
///
/// The result is a **sound superset**. Verify the survivors with an exact check,
/// such as a [`ContainsTable`](super::ContainsTable) passed to
/// [`contains`](super::contains()), to recover the precise answer.
///
/// This function only executes the analyzed cover; the caller decides whether
/// scanning it is profitable.
///
/// The function does not silently fall back to a full scalar scan. If the target
/// has no SIMD implementation, it returns an error without modifying `out`.
///
/// An empty cover appends nothing and succeeds without SIMD support.
///
/// # Precondition
/// `row_offsets` are valid delimiters for `codes`, and every code lies in the
/// token domain of the analyzed cover. A validated [`Column`](crate::Column)
/// and an analysis built for that column satisfy these properties.
///
/// # Errors
/// Returns [`PrefilterError::UnsupportedArchitecture`] when no SIMD kernel is
/// available for a non-empty cover.
pub fn prefilter_candidates<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    analysis: &PrefilterAnalysis,
    out: &mut Vec<usize>,
) -> Result<(), PrefilterError> {
    scan::scan(
        codes,
        row_offsets,
        analysis.probe_cover(),
        analysis.use_sparse_row_mapping(),
        out,
    )
}
