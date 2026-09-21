// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Substring query preparation, planning, scanning and verification.
//!
//! [`ContainsDfa`] prepares a token-level KMP automaton for [`row_contains`].
//! [`ContainsScan`] compiles a **sound probe cover** from the pattern — dictionary token ids
//! and id ranges chosen so that *any* row containing the pattern holds at least
//! one probe token — then scans the flat code stream for the rows holding one
//! ([`ContainsScan::scan`]).
//!
//! Every hit is then checked against the alignment graph in the compressed
//! domain (`walk`), so the rows [`ContainsScan::scan`] returns are exactly
//! the rows containing the pattern and no caller-side verification is needed.
//!
//! An empty pattern matches every row, including rows without codes. Its
//! preparation needs no probes, and execution appends all rows without scanning.
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
//! # Responsibilities
//! [`ContainsScan`] coordinates graph construction, cost-based cover selection and walk
//! compilation. `alignment` owns the graph and cut solver; `plan` prices and
//! selects covers and matcher configurations using explicit facts and capabilities.
//! `scan` produces hits and resolves rows, calling the exact verifier in
//! `verify::walk`. Profitability remains a caller decision.

mod alignment;
mod error;
mod plan;
mod scan;
mod verify;

#[cfg(test)]
mod tests;

pub use alignment::cover::ProbeCover;
pub use error::ContainsError;
pub use verify::{ContainsDfa, row_contains};

use alignment::graph::AlignmentGraph;
use verify::walk::Walk;

use crate::core::dictionary::{CompactDictionaryView, DictionaryView};
use crate::core::offset::Offset;
use crate::core::types::Token;
use crate::core::validate::InvalidFrequencyIndex;
use crate::search::index::{TokenFrequencyIndex, TokenFrequencyIndexStorage};

/// Immutable prepared substring scan for one pattern and dictionary.
///
/// Holds the probe cover, compiled walker and token frequencies. Each
/// [`ContainsScan::scan`] call owns its execution state, so preparation can be reused
/// across scans. The dictionary is supplied separately at execution.
///
/// Points and ranges are disjoint, so each covered token occurrence is counted
/// exactly once. An empty pattern matches all rows and needs no probes.
#[derive(Debug, Clone)]
pub struct ContainsScan {
    probe_cover: ProbeCover,
    covered_frequency: u32,
    total_frequency: u32,
    walk: Walk,
    /// Empty patterns admit even rows without codes, independently of the cover.
    matches_all: bool,
}

impl ContainsScan {
    /// Maximum pattern length, bounded by the compiled walker's `u16` node IDs.
    pub const MAX_PATTERN_LEN: usize = u16::MAX as usize;

    /// Prepare an exact substring scan for `pattern` using the dictionary and index.
    /// Select the sampled probe cover with the lowest ranking score and compile
    /// the alignment walker used to verify its hits.
    ///
    /// This constructor prepares the checks and reports their frequency;
    /// the caller decides whether executing them is profitable. An empty pattern
    /// produces a scan that admits every row without compiling probes.
    ///
    /// # Precondition
    /// `dict` is conformant: sorted, complete, and unique. These properties are
    /// guaranteed for a dictionary trained by [`Parser::train`](crate::Parser::train)
    /// or passed through [`CompactDictionary::validate`](crate::CompactDictionary::validate).
    /// `frequencies` must be associated with this dictionary's token IDs and the
    /// code stream used for planning. Build or validate it at the input boundary
    /// using `dict.num_tokens()` and that stream's codes or code count.
    ///
    /// Preparation only checks that the already validated index has
    /// `dict.num_tokens() + 1` cumulative entries, so dictionary token IDs can
    /// safely index it. This is a constant-time size check; it does not establish
    /// dictionary identity or verify the frequencies against the stream.
    /// Values are advisory weights: they affect the plan and profitability, but
    /// never remove members from the resulting cover.
    ///
    /// # Errors
    /// Returns [`ContainsError`] if the pattern exceeds [`Self::MAX_PATTERN_LEN`],
    /// the index size does not match the dictionary, or greedy parsing encounters
    /// a missing dictionary token.
    pub fn new<S: TokenFrequencyIndexStorage>(
        pattern: &[u8],
        dict: CompactDictionaryView<'_>,
        frequencies: &TokenFrequencyIndex<S>,
    ) -> Result<Self, ContainsError> {
        if pattern.len() > Self::MAX_PATTERN_LEN {
            return Err(ContainsError::PatternTooLong {
                length: pattern.len(),
                max: Self::MAX_PATTERN_LEN,
            });
        }
        if frequencies.num_tokens() != dict.num_tokens() {
            return Err(ContainsError::InvalidData(
                InvalidFrequencyIndex::BadLength.into(),
            ));
        }
        if pattern.is_empty() {
            return Ok(Self {
                probe_cover: ProbeCover::from_runs(Vec::new()),
                covered_frequency: 0,
                total_frequency: frequencies.total_frequency(),
                walk: Walk::default(),
                matches_all: true,
            });
        }
        let graph = AlignmentGraph::new(dict, pattern, frequencies.as_view())?;
        let plan::SelectedCover {
            cover,
            covered_frequency,
        } = plan::select_cover(&graph, frequencies.as_view(), scan::detect_target_caps());
        Ok(Self {
            probe_cover: cover,
            covered_frequency,
            total_frequency: frequencies.total_frequency(),
            walk: Walk::from_graph(&graph, pattern),
            matches_all: false,
        })
    }

    /// Execute this scan and append the ascending rows containing the pattern.
    ///
    /// Every hit on the cover is checked against the alignment graph in the
    /// compressed domain. Emitted rows contain a complete occurrence of the pattern.
    ///
    /// This method only executes the prepared cover; the caller decides whether
    /// scanning it is profitable. For a non-empty pattern, an empty cover appends
    /// nothing. An empty pattern appends every row, including empty rows, without
    /// scanning. Execution state is local to this call, so the same preparation
    /// can be reused or shared between concurrent calls.
    ///
    /// # Precondition
    /// `row_offsets` are valid delimiters for `codes`, every code lies in the
    /// token domain of the prepared cover, and `dict` is the dictionary this
    /// scan was built over. Rows must be greedily tokenized with that same
    /// conformant dictionary, as produced by [`Column::compress`](crate::Column::compress).
    /// Validating a dictionary or externally supplied column buffers alone does not
    /// establish greedy tokenization. Each scan appends each matching row once in
    /// ascending order, preserving all prior contents of `out`.
    pub fn scan<O: Offset>(
        &self,
        codes: &[Token],
        row_offsets: &[O],
        dict: CompactDictionaryView<'_>,
        out: &mut Vec<usize>,
    ) {
        if self.matches_all {
            out.extend(0..row_offsets.len().saturating_sub(1));
            return;
        }
        if codes.is_empty() || row_offsets.len() < 2 {
            return;
        }
        let config = plan::select_matcher_config(
            scan::detect_target_caps(),
            &self.probe_cover,
            plan::probe_density(
                self.covered_frequency as usize,
                self.total_frequency as usize,
                codes.len(),
            ),
        );
        scan::matches(
            config,
            scan::ScanInput::new(codes, row_offsets, &self.probe_cover),
            dict,
            &self.walk,
            out,
        );
    }

    /// The normalized checks the SIMD prefilter can execute.
    ///
    /// For an empty pattern this cover is empty, but [`ContainsScan::scan`]
    /// still returns every row.
    pub fn probe_cover(&self) -> &ProbeCover {
        &self.probe_cover
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
