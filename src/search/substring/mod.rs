// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Substring query preparation, planning, scanning and verification.
//!
//! [`ContainsDfa`] prepares a token-level KMP automaton for [`row_contains`].
//! [`ContainsScan`] compiles a **sound probe cover** from the pattern — dictionary
//! token ids and id ranges chosen so that *any* row containing the pattern holds at
//! least one probe token — then scans the flat code stream for the rows holding one
//! ([`ContainsScan::scan`]). Every hit is then checked against the alignment graph
//! in the compressed domain (`walk`).
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
//! selects covers and their matchers using explicit facts and capabilities.
//! `scan` produces hits, resolves rows and verifies each hit with the exact walker
//! from `verify::walk`.

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
use scan::{MatcherKind, ScanInput};
use verify::walk::Walk;

use crate::core::dictionary::{CompactDictionaryView, DictionaryView};
use crate::core::offset::Offset;
use crate::core::types::Token;
use crate::core::validate::InvalidFrequencyIndex;
use crate::search::index::{TokenFrequencyIndex, TokenFrequencyIndexStorage};

/// A prepared substring query, ready to scan any chunk encoded with its dictionary.
///
/// Holds the probe cover, its matcher, the compiled walker and the probe density
/// of the stream used for planning. Each [`ContainsScan::scan`] call owns its
/// execution state, so one preparation can scan many chunks and be shared between
/// concurrent calls. The dictionary is supplied separately at execution, and
/// coverage statistics take the frequency index of the chunk they describe.
/// An empty pattern matches all rows and needs no probes.
#[derive(Debug, Clone)]
pub struct ContainsScan {
    probe_cover: ProbeCover,
    matcher: MatcherKind,
    walk: Walk,
    probe_density: f64,
    matches_all: bool,
}

impl ContainsScan {
    /// Maximum pattern length, bounded by the compiled walker's `u16` node IDs.
    pub const MAX_PATTERN_LEN: usize = u16::MAX as usize;

    /// Prepare an exact substring scan for `pattern` using the dictionary and index.
    /// Select the sampled probe cover with the lowest relative cost and compile
    /// the alignment walker used to verify its hits.
    ///
    /// This constructor prepares the checks; the caller decides whether executing
    /// them is profitable, using the coverage statistics for the chunk at hand.
    /// An empty pattern produces a scan that admits every row without compiling probes.
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
                matcher: MatcherKind::Table,
                walk: Walk::default(),
                probe_density: 0.0,
                matches_all: true,
            });
        }
        let isa = scan::detect_isa();
        let graph = AlignmentGraph::new(dict, pattern, frequencies.as_view())?;
        let (cover, covered_frequency) = plan::select_cover(&graph, frequencies.as_view(), isa);
        Ok(Self {
            matcher: plan::select_matcher(isa, &cover),
            probe_cover: cover,
            walk: Walk::from_graph(&graph, pattern),
            probe_density: fraction(covered_frequency, frequencies.total_frequency()),
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
        if codes.is_empty() || row_offsets.len() < 2 || self.probe_cover.is_empty() {
            return;
        }
        scan::run(
            self.matcher,
            plan::skip_empty_packing(self.probe_density, codes.len()),
            ScanInput::new(codes, row_offsets, dict, &self.probe_cover, &self.walk),
            out,
        );
    }

    /// Whether repeated verification work warrants falling back to another search.
    ///
    /// Returns `true` when `sum(frequency(token) * maximum work(token))` over
    /// probe tokens exceeds the encoded token count. Work counts uncertain lookups,
    /// alignment trials and forward/backward walk steps. Tokens containing the
    /// whole pattern are accepted directly and contribute no uncertain work.
    ///
    /// For example, a row repeating `(a × 255 + b)` cannot contain `a × 256`,
    /// yet frequent `a` tokens can trigger many long, unsuccessful walks. A caller
    /// can use this method to choose decompression followed by a byte search.
    /// [`Self::scan`] never calls this method or falls back automatically.
    ///
    /// The bound assumes every probe occurrence needs all its uncertain checks;
    /// early failures and matches can make actual work much smaller. It excludes
    /// constant hit handling, scanning, row resolution and preparation, so it is
    /// not a latency prediction or a guarantee of a speedup over another search.
    /// Empty patterns, empty covers and empty streams return `false`.
    ///
    /// The bound is computed only on request, using temporary arrays proportional
    /// to the pattern length and visiting the uncertain covered tokens' roles.
    ///
    /// # Precondition
    /// `frequencies` must use this scan's dictionary token IDs and count exact
    /// occurrences in the code stream to be scanned, even if planning used a
    /// different stream. This correspondence is not checked here.
    pub fn should_fallback<S: TokenFrequencyIndexStorage>(
        &self,
        frequencies: &TokenFrequencyIndex<S>,
    ) -> bool {
        self.walk
            .verification_work_bound(&self.probe_cover, frequencies.as_view())
            > u64::from(frequencies.total_frequency())
    }

    /// The normalized checks the SIMD prefilter can execute.
    ///
    /// For an empty pattern this cover is empty, but [`ContainsScan::scan`]
    /// still returns every row.
    pub fn probe_cover(&self) -> &ProbeCover {
        &self.probe_cover
    }

    /// Number of code positions in the indexed stream whose token is covered by
    /// the probes. Points and ranges are disjoint, so each occurrence counts once.
    /// Zero for an empty pattern, which needs no probes.
    ///
    /// # Panics
    /// Panics if a probe token lies outside `frequencies`, which must use this
    /// scan's dictionary token IDs.
    pub fn covered_frequency<S: TokenFrequencyIndexStorage>(
        &self,
        frequencies: &TokenFrequencyIndex<S>,
    ) -> u32 {
        self.probe_cover.frequency(frequencies.as_view())
    }

    /// Fraction of code positions in the indexed stream whose token is covered by
    /// the probes. Returns `0.0` when that stream is empty or the pattern is empty
    /// and needs no probes.
    ///
    /// # Panics
    /// Panics under the same condition as [`Self::covered_frequency`].
    pub fn covered_fraction<S: TokenFrequencyIndexStorage>(
        &self,
        frequencies: &TokenFrequencyIndex<S>,
    ) -> f64 {
        fraction(
            self.covered_frequency(frequencies),
            frequencies.total_frequency(),
        )
    }

    /// Upper bound on the fraction of a chunk's rows containing a probe token.
    ///
    /// Computes `min(1, covered_frequency / row_count)`. Each candidate row holds
    /// at least one covered code, but multiple covered codes may share a row,
    /// making the bound loose. A row may require several verification attempts,
    /// so this fraction does not measure verifier calls.
    ///
    /// `frequencies` must count exact occurrences in the chunk whose rows are
    /// counted by `row_count`. Returns `0.0` for zero rows and never exceeds `1.0`.
    /// An empty pattern admits every row, so its fraction is `1.0` for any
    /// non-empty chunk, even one consisting entirely of empty rows.
    ///
    /// # Panics
    /// Panics under the same condition as [`Self::covered_frequency`].
    pub fn expected_candidate_row_fraction<S: TokenFrequencyIndexStorage>(
        &self,
        frequencies: &TokenFrequencyIndex<S>,
        row_count: usize,
    ) -> f64 {
        if row_count == 0 {
            return 0.0;
        }
        if self.matches_all {
            return 1.0;
        }
        (f64::from(self.covered_frequency(frequencies)) / row_count as f64).min(1.0)
    }
}

/// `covered / total`, or `0.0` for an empty stream.
fn fraction(covered: u32, total: u32) -> f64 {
    if total == 0 {
        0.0
    } else {
        f64::from(covered) / f64::from(total)
    }
}
