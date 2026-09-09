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
//! An empty pattern matches every row, including rows without codes. Its
//! analysis needs no probes, and execution appends all rows without scanning.
//!
//! # Soundness
//! Every occurrence of a non-empty pattern in an encoded row falls into one of
//! two cases, and the cover covers both:
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
//! * [`TokenFrequencyIndex`] — the reusable per-column selectivity index the
//!   compiler reads and the caller owns.
//! * `compile` — pattern to normalized cover, preserving every selected id
//!   regardless of its advisory frequency.
//! * `graph` — pattern to alignment DAG: every layout of the pattern across
//!   token boundaries, as one graph whose cuts are exactly the sound covers.
//! * `mincut` — the cheapest such cut, by max-flow over the split DAG.
//! * `cover` — the cover itself, in both the shapes the scan wants.
//! * `scan` — the vector kernels. Profitability stays outside execution.

mod compile;
mod cover;
mod graph;
mod mincut;
mod scan;

#[cfg(test)]
mod tests;

pub use cover::ProbeCover;

use crate::core::dictionary::CompactDictionaryView;
use crate::core::offset::Offset;
use crate::core::types::Token;
use crate::search::index::{TokenFrequencyIndex, TokenFrequencyIndexStorage};

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

/// A pattern's prefilter checks and their frequency, or an all-rows result.
///
/// Points and ranges are disjoint, so each covered token occurrence is counted
/// exactly once. An empty pattern matches all rows and needs no probes.
#[derive(Debug, Clone)]
pub struct PrefilterAnalysis {
    probe_cover: ProbeCover,
    covered_frequency: u32,
    total_frequency: u32,
    /// Empty patterns admit even rows without codes, independently of the cover.
    matches_all: bool,
}

impl PrefilterAnalysis {
    /// The normalized checks the SIMD prefilter can execute.
    ///
    /// For an empty pattern this cover is empty, but [`prefilter_candidates`]
    /// still returns every row using the analysis's all-rows flag.
    pub fn probe_cover(&self) -> &ProbeCover {
        &self.probe_cover
    }

    /// Number of code positions whose token is covered by the probes.
    /// Returns zero for an empty pattern, which needs no probes.
    pub fn covered_frequency(&self) -> u32 {
        self.covered_frequency
    }

    /// Fraction of code positions whose token is covered by the probes.
    ///
    /// Returns `0.0` when the indexed code stream is empty or the pattern is
    /// empty and needs no probes.
    pub fn covered_fraction(&self) -> f64 {
        if self.total_frequency == 0 {
            0.0
        } else {
            f64::from(self.covered_frequency) / f64::from(self.total_frequency)
        }
    }

    /// Number of code positions represented by the frequency index.
    pub fn total_frequency(&self) -> u32 {
        self.total_frequency
    }

    /// SIMD comparisons each vector of the code stream pays for this cover: one
    /// per point, two per inclusive range. Zero for an empty pattern.
    pub fn comparison_cost(&self) -> usize {
        let cover = self.probe_cover();
        cover
            .points()
            .len()
            .saturating_add(cover.ranges().len().saturating_mul(2))
    }

    /// Expected share of `row_count` rows the scan will admit for verification.
    ///
    /// Verification is charged per row, so the estimate is covered codes per
    /// row: exact when no row holds two covered codes, an over-estimate
    /// otherwise. Returns `0.0` for an empty region and never exceeds `1.0`.
    /// An empty pattern admits every row, so its fraction is `1.0` for any
    /// non-empty region, even one consisting entirely of empty rows.
    pub fn expected_candidate_row_fraction(&self, row_count: usize) -> f64 {
        if row_count == 0 {
            return 0.0;
        }
        if self.matches_all {
            return 1.0;
        }
        (f64::from(self.covered_frequency) / row_count as f64).min(1.0)
    }
}

/// Widest cover the specialized SIMD kernels serve; wider covers fall through to
/// a generic loop costing roughly 1.8x more per comparison.
const MAX_SIMD_COMPARISONS: usize = 16;

/// Largest share of rows the default policy sends to exact verification, which
/// costs 2.1x to 6.5x per row what bulk decoding does.
const MAX_CANDIDATE_ROW_FRACTION: f64 = 0.10;

/// Return whether the default empirical policy expects prefiltering to beat a
/// bulk-decode fallback over a region of `row_count` rows.
///
/// The policy prices the two costs a scan pays, both known after
/// [`analyze_prefilter`]: its
/// [`comparison_cost`](PrefilterAnalysis::comparison_cost) per code, and the
/// [`expected_candidate_row_fraction`](PrefilterAnalysis::expected_candidate_row_fraction)
/// it sends to per-row verification. For a non-empty pattern, an empty cover
/// passes both: it proves no encoded row can match, so it scans nothing and
/// admits nothing.
/// An empty pattern also passes: its all-rows answer is exact and requires
/// neither a scan nor verification.
///
/// This is a performance hint, not a correctness requirement, and it neither
/// executes nor bypasses the prefilter. The thresholds were calibrated on
/// AArch64 over 2877 `contains` queries against a bulk-decode-plus-`memmem`
/// fallback, where they admit no query the fallback would have won. Callers
/// with materially different columns or architectures may choose their own
/// policy.
pub fn prefilter_is_likely_profitable(analysis: &PrefilterAnalysis, row_count: usize) -> bool {
    analysis.matches_all
        || (analysis.comparison_cost() <= MAX_SIMD_COMPARISONS
            && analysis.expected_candidate_row_fraction(row_count) < MAX_CANDIDATE_ROW_FRACTION)
}

/// Analyze `pattern`, selecting a normalized minimum-cut probe cover when needed.
///
/// This function constructs the checks and reports their frequency; the caller
/// decides whether executing them is profitable. An empty pattern produces an
/// analysis that admits every row without compiling probes.
///
/// # Precondition
/// `dict` is conformant: sorted, complete, and unique. These properties are
/// guaranteed for a dictionary trained by [`Parser::train`](crate::Parser::train)
/// or passed through [`CompactDictionary::validate`](crate::CompactDictionary::validate).
/// `frequencies` must use `dict`'s token domain and the scanned code count.
/// Values are advisory weights: they affect the plan and profitability, but
/// never remove members from the resulting cover.
pub fn analyze_prefilter<S: TokenFrequencyIndexStorage>(
    pattern: &[u8],
    dict: CompactDictionaryView<'_>,
    frequencies: &TokenFrequencyIndex<S>,
) -> PrefilterAnalysis {
    if pattern.is_empty() {
        return PrefilterAnalysis {
            probe_cover: ProbeCover::from_membership(Vec::new()),
            covered_frequency: 0,
            total_frequency: frequencies.total_frequency(),
            matches_all: true,
        };
    }
    let frequencies_view = frequencies.as_view();
    let probe_cover = compile::compile_cover(dict, pattern, frequencies_view);
    let covered_frequency = probe_cover
        .points
        .iter()
        .map(|&token| frequencies_view.frequency(token))
        .chain(
            probe_cover
                .ranges
                .iter()
                .map(|&range| frequencies_view.range_frequency(range)),
        )
        .sum();
    PrefilterAnalysis {
        probe_cover,
        covered_frequency,
        total_frequency: frequencies.total_frequency(),
        matches_all: false,
    }
}

/// Execute `analysis` and append its candidate rows in ascending order.
///
/// The result is a **sound superset**. Verify the survivors with an exact check,
/// such as a [`ContainsTable`](super::ContainsTable) passed to
/// [`contains`](super::contains()), to recover the precise answer.
///
/// For a non-empty pattern, candidates are rows containing a covered code; the
/// caller decides whether scanning the cover is profitable. An empty pattern
/// appends every row, including empty rows, without scanning or needing SIMD.
///
/// The function does not silently fall back to a full scalar scan. If the target
/// has no SIMD implementation and a scan is needed, it returns an error without
/// modifying `out`.
///
/// For a non-empty pattern, an empty cover appends nothing and succeeds without
/// SIMD support.
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
    scan::candidates(codes, row_offsets, analysis, out)
}
