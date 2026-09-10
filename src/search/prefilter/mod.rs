// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! SIMD prefilter for substring search.
//!
//! Answering `LIKE '%pattern%'` exactly means checking every row — e.g. stepping
//! the token-level KMP automaton of [`contains`](super::contains()) over its
//! codes. This module trims that per-row work down to the rows that *can* match.
//! It compiles a **sound probe cover** from the pattern — dictionary token ids
//! and id ranges chosen so that *any* row containing the pattern holds at least
//! one probe token — then scans the flat code stream for the rows holding one
//! ([`prefilter_candidates`]).
//!
//! Every hit is then checked against the alignment graph in the compressed
//! domain (`walk`), so the rows [`prefilter_candidates`] returns are exactly
//! the rows containing the pattern and no caller-side verification is needed.
//!
//! An empty pattern matches every row, including rows without codes. Its
//! analysis needs no probes, and execution appends all rows without scanning.
//!
//! # Soundness
//! Every occurrence of a non-empty pattern in an encoded row falls into one of two cases,
//! and the cover covers both:
//!
//! * **One token contains the whole pattern.** Its id is added unconditionally
//!   (only reachable when `pattern.len() <= MAX_TOKEN_SIZE`), as the range of
//!   tokens the pattern is a prefix of or as the set holding it further in.
//! * **The occurrence crosses at least one token boundary.** Then it begins at
//!   some feasible first-token alignment `k`, after which greedy parsing of
//!   `pattern[k..]` is deterministic. Every such layout is one path through the
//!   alignment DAG, and the cover is a cut of that DAG — so whichever layout the
//!   occurrence takes, it runs into a probe.
//!
//! # Shape
//! * [`TokenFrequencyIndex`] — the reusable per-column selectivity index the
//!   compiler reads and the caller owns.
//! * `graph` — pattern to alignment DAG: every layout of the pattern across
//!   token boundaries, as one graph whose cuts are exactly the sound covers.
//! * `mincut` — the cheapest such cut, by max-flow over the split DAG, under
//!   whatever edge weights it is handed.
//! * `plan` — the three end to end: pattern in, the cover the model prices
//!   lowest out, preserving every selected id regardless of its advisory
//!   frequency.
//! * `cover` — the cover itself, in both the shapes the scan wants.
//! * `walk` — the graph flattened for checking a hit in codes, forward to the
//!   sink and back to the source.
//! * `scan` — the vector kernels and, in `policy`, the fitted cost model
//!   that picks them and prices a cover. Profitability stays outside execution.

mod cover;
mod graph;
mod mincut;
mod plan;
mod scan;

#[cfg(test)]
mod tests;

pub use cover::ProbeCover;

use crate::core::dictionary::CompactDictionaryView;
use crate::core::offset::Offset;
use crate::core::types::Token;
use crate::search::index::{TokenFrequencyIndex, TokenFrequencyIndexStorage};

/// The normalized probe cover selected for a pattern and its frequency, or an
/// all-rows result.
///
/// Points and ranges are disjoint, so each covered token occurrence is counted
/// exactly once. An empty pattern matches all rows and needs no probes.
#[derive(Debug, Clone)]
pub struct PrefilterAnalysis {
    probe_cover: ProbeCover,
    covered_frequency: u32,
    total_frequency: u32,
    scan_ns: f64,
    walk: scan::Walk,
    /// Empty patterns admit even rows without codes, independently of the cover.
    matches_all: bool,
}

impl PrefilterAnalysis {
    /// The normalized checks the SIMD prefilter can execute.
    ///
    /// For an empty pattern this cover is empty, but [`prefilter_candidates`]
    /// still returns every row.
    pub fn probe_cover(&self) -> &ProbeCover {
        &self.probe_cover
    }

    /// Expected nanoseconds to scan the cover over the analyzed stream and
    /// verify the rows it admits, from the fitted kernel model. The number
    /// the cover was chosen by, so alternatives compare against it directly.
    /// Zero for an empty pattern, which scans nothing.
    pub fn expected_scan_ns(&self) -> f64 {
        self.scan_ns
    }

    /// Number of code positions whose token is covered by the probes.
    /// Zero for an empty pattern, which needs no probes.
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
/// admits nothing. An empty pattern also passes: its all-rows answer is exact
/// and requires neither a scan nor verification.
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

/// Analyze `pattern` and return the sound probe cover the scan cost model
/// prices lowest over a stream of `row_count` rows.
///
/// This function constructs the checks and reports their frequency and cost;
/// the caller decides whether executing them is profitable. An empty pattern
/// produces an analysis that admits every row without compiling probes.
///
/// # Precondition
/// `dict` is conformant: sorted, complete, and unique. These properties are
/// guaranteed for a dictionary trained by [`Parser::train`](crate::Parser::train)
/// or passed through [`CompactDictionary::validate`](crate::CompactDictionary::validate).
/// `frequencies` must use `dict`'s token domain and the scanned code count.
/// Values are advisory weights: they affect the plan and profitability, but
/// never remove members from the resulting cover.
///
/// # Panics
/// Panics when `pattern` is longer than [`MAX_PATTERN_LEN`], which is what the
/// alignment walk's node ids hold.
pub fn analyze_prefilter<S: TokenFrequencyIndexStorage>(
    pattern: &[u8],
    dict: CompactDictionaryView<'_>,
    frequencies: &TokenFrequencyIndex<S>,
    row_count: usize,
) -> PrefilterAnalysis {
    if pattern.is_empty() {
        return PrefilterAnalysis {
            probe_cover: ProbeCover::from_runs(Vec::new()),
            covered_frequency: 0,
            total_frequency: frequencies.total_frequency(),
            scan_ns: 0.0,
            walk: scan::Walk::default(),
            matches_all: true,
        };
    }
    assert!(
        pattern.len() <= MAX_PATTERN_LEN,
        "pattern of {} bytes exceeds the prefilter's {MAX_PATTERN_LEN}",
        pattern.len()
    );
    let planned = plan::plan(dict, pattern, frequencies.as_view(), row_count);
    PrefilterAnalysis {
        probe_cover: planned.cover,
        covered_frequency: planned.covered,
        total_frequency: frequencies.total_frequency(),
        scan_ns: planned.scan_ns,
        walk: planned.walk,
        matches_all: false,
    }
}

/// Execute `analysis` and append the ascending rows containing the pattern.
///
/// The rows are exact, not a superset: every hit on the cover is verified
/// against the alignment graph in the compressed domain, so no caller-side
/// check such as [`contains`](super::contains()) or a
/// [`BytesVerifier`](super::BytesVerifier) is needed behind this.
///
/// This function only executes the analyzed cover; the caller decides whether
/// scanning it is profitable. For a non-empty pattern, an empty cover appends
/// nothing. An empty pattern appends every row, including empty rows, without
/// scanning.
///
/// # Precondition
/// `row_offsets` are valid delimiters for `codes`, every code lies in the
/// token domain of the analyzed cover, and `dict` is the dictionary the
/// analysis was built over. A validated [`Column`](crate::Column) and an
/// analysis built for that column satisfy these properties.
pub fn prefilter_candidates<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    dict: CompactDictionaryView<'_>,
    analysis: &PrefilterAnalysis,
    out: &mut Vec<usize>,
) {
    if analysis.matches_all {
        out.extend(0..row_offsets.len().saturating_sub(1));
        return;
    }
    let input = scan::ScanInput::full(codes, row_offsets, analysis.probe_cover());
    let plan = scan::plan(input, analysis);
    scan::execute(plan, input, &analysis.walk, dict, out);
}

/// The longest pattern [`analyze_prefilter`] takes: one node per needle offset
/// plus the sink, all of them inside the walk's `u16` ids.
pub const MAX_PATTERN_LEN: usize = u16::MAX as usize;
