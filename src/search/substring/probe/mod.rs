// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ContainsScan`]: substring search by scanning for the codes of a cut.
//!
//! # How it works
//!
//! Rows are tokenized greedily: each token is the longest dictionary entry that
//! prefixes the remaining bytes. An occurrence of the pattern may begin inside a
//! token, but from its first token boundary on, greedy tokenization leaves little
//! choice. Each later token either ends inside the pattern, and is then the longest
//! token that prefixes the rest of the pattern, or finishes it, and then starts with
//! whatever remains.
//!
//! All possible encodings therefore fit in a small DAG, the *alignment graph*. Its
//! nodes are the pattern offsets `0..=n`, and every occurrence in a row is a path
//! from `0` to `n` whose edges are the occurrence's tokens:
//!
//! * an *alignment edge* `0 -> k` for a first token that begins before the
//!   occurrence and ends with its first `k` bytes, or `0 -> n` if it contains the
//!   whole occurrence;
//! * a *greedy edge* for each token that begins and ends inside the pattern;
//! * a *range edge* to `n` for a last token that starts with the rest of the pattern.
//!
//! Preparation picks a *cut* of this graph: a set of edges that every path crosses.
//! The codes on its edges form the *filter*, represented as a [`ProbeCover`]: every
//! row that contains the pattern contains one of them. Any cut is correct, and the
//! planner looks for a cheap one: rare codes mean few hits to verify, and few
//! distinct codes mean a faster matcher.
//!
//! The scan then processes the codes block by block:
//!
//! 1. a matcher marks every code in the filter, a *hit*;
//! 2. each hit is mapped to its row and verified by walking the alignment graph
//!    forward and backward through the neighboring codes.
//!
//! A row is emitted on its first verified hit, and its remaining hits are skipped.
//!
//! # Modules
//!
//! * `alignment`: the alignment graph, its minimum cut, and the resulting filter.
//! * `plan`: the cost model that chooses the cut, its matcher and the mask packing.
//! * `scan`: the block loop, the SIMD matchers and row resolution.
//! * `verify`: the hit verifier.

mod alignment;
mod plan;
mod scan;
mod verify;

pub use alignment::cover::ProbeCover;

use alignment::graph::AlignmentGraph;
use scan::{MatcherKind, ScanInput};
use verify::HitVerifier;

use crate::core::dictionary::{CompactDictionaryView, DictionaryView};
use crate::core::offset::Offset;
use crate::core::types::Token;
use crate::core::validate::{InvalidColumn, InvalidFrequencyIndex, panic_malformed};
use crate::search::index::{TokenFrequencyIndex, TokenFrequencyIndexStorage};
use crate::search::substring::ContainsError;

/// A substring query prepared for one dictionary.
///
/// Preparation chooses the filter and compiles the hit verifier once. Each
/// [`scan`](Self::scan) keeps its state local, so one query can scan any number of
/// chunks encoded with the same dictionary, concurrently if needed. The pattern
/// must not be empty: every row contains the empty pattern, so callers answer it
/// without a scan.
#[derive(Debug, Clone)]
pub struct ContainsScan<'d> {
    dict: CompactDictionaryView<'d>,
    probe_cover: ProbeCover,
    matcher: MatcherKind,
    verifier: HitVerifier,
    probe_density: f64,
}

impl<'d> ContainsScan<'d> {
    /// Longest supported pattern, in bytes. Pattern offsets are stored as `u16`.
    pub const MAX_PATTERN_LEN: usize = u16::MAX as usize;

    /// Prepares a scan for `pattern`.
    ///
    /// `frequencies` counts how often each token occurs in a representative code
    /// stream. It only guides the choice of the filter: inaccurate counts can make scans
    /// slower, never incorrect.
    ///
    /// # Errors
    /// Returns [`ContainsError::EmptyPattern`] if `pattern` is empty,
    /// [`ContainsError::PatternTooLong`] if it is longer than
    /// [`Self::MAX_PATTERN_LEN`], and [`ContainsError::InvalidData`] if `frequencies`
    /// does not have one entry per dictionary token or `dict` cannot tokenize
    /// `pattern`.
    ///
    /// # Precondition
    /// `dict` is conformant (sorted, unique and complete), as guaranteed by
    /// [`Parser::train`](crate::Parser::train) and
    /// [`CompactDictionary::validate`](crate::CompactDictionary::validate), and
    /// `frequencies` is indexed by its token IDs. Only the index length is checked.
    pub fn new<S: TokenFrequencyIndexStorage>(
        pattern: &[u8],
        dict: CompactDictionaryView<'d>,
        frequencies: &TokenFrequencyIndex<S>,
    ) -> Result<Self, ContainsError> {
        if pattern.is_empty() {
            return Err(ContainsError::EmptyPattern);
        }
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
        let isa = scan::detect_isa();
        let graph = AlignmentGraph::new(dict, pattern, frequencies.as_view())?;
        let (cover, matcher, covered_frequency) =
            plan::select_cover(&graph, frequencies.as_view(), isa);
        // Following greedy edges from the source always reaches the sink, so every
        // cut, and with it the filter, is nonempty.
        debug_assert!(!cover.is_empty(), "empty filter for {pattern:?}");
        Ok(Self {
            dict,
            matcher,
            probe_cover: cover,
            verifier: HitVerifier::from_graph(&graph, pattern),
            probe_density: fraction(covered_frequency, frequencies.total_frequency()),
        })
    }

    /// Appends to `out` the index of every row that contains the pattern, in
    /// ascending order.
    ///
    /// Row `i` is `codes[row_offsets[i]..row_offsets[i + 1]]`. Codes outside
    /// `row_offsets[0]..row_offsets[last]` belong to no row and are never read, so a
    /// view sliced from a larger column can pass its whole code buffer.
    ///
    /// # Panics
    /// Panics with [`InvalidColumn::BadRowOffsets`] if the first offset exceeds the
    /// last, or the last exceeds `codes.len()`.
    ///
    /// # Precondition
    /// `row_offsets` is non-decreasing, and the rows were greedily tokenized with the
    /// dictionary this scan was prepared with, as by
    /// [`Column::compress`](crate::Column::compress). Validating the buffers does not
    /// establish greedy tokenization.
    pub fn scan<O: Offset>(&self, codes: &[Token], row_offsets: &[O], out: &mut Vec<usize>) {
        let (Some(first), Some(last)) = (row_offsets.first(), row_offsets.last()) else {
            return;
        };
        let (first, last) = (first.to_usize(), last.to_usize());
        if first > last || last > codes.len() {
            panic_malformed(InvalidColumn::BadRowOffsets);
        }
        // Without codes, no row can contain the pattern.
        if first == last {
            return;
        }
        scan::run(
            self.matcher,
            plan::skip_empty_packing(self.probe_density, last - first),
            ScanInput::new(
                &codes[..last],
                row_offsets,
                self.dict,
                &self.probe_cover,
                &self.verifier,
            ),
            out,
        );
    }

    /// The tokens this scan searches for; never empty.
    pub fn probe_cover(&self) -> &ProbeCover {
        &self.probe_cover
    }

    /// Number of codes in the stream indexed by `frequencies` that hit the filter.
    ///
    /// # Panics
    /// Panics if a filter token is out of range for `frequencies`, which must be
    /// indexed by this scan's dictionary.
    pub fn covered_frequency<S: TokenFrequencyIndexStorage>(
        &self,
        frequencies: &TokenFrequencyIndex<S>,
    ) -> u32 {
        self.probe_cover.frequency(frequencies.as_view())
    }

    /// [`covered_frequency`](Self::covered_frequency) as a fraction of all indexed
    /// codes: the expected hit density. `0.0` for an empty stream.
    ///
    /// # Panics
    /// Panics under the same condition as [`covered_frequency`](Self::covered_frequency).
    pub fn covered_fraction<S: TokenFrequencyIndexStorage>(
        &self,
        frequencies: &TokenFrequencyIndex<S>,
    ) -> f64 {
        fraction(
            self.covered_frequency(frequencies),
            frequencies.total_frequency(),
        )
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

#[cfg(test)]
mod tests;
