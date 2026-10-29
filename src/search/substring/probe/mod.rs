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

use std::fmt;

use alignment::graph::AlignmentGraph;
use scan::{Isa, MatcherKind, ScanInput};
use verify::HitVerifier;

use crate::core::dictionary::{CompactDictionaryView, DictionaryView};
use crate::core::offset::Offset;
use crate::core::types::Token;
use crate::core::validate::{InvalidColumn, panic_malformed};
use crate::search::index::TokenFrequencyIndexView;
use crate::search::substring::ContainsError;

/// A substring query prepared for one dictionary.
///
/// Preparation chooses the filter and compiles the hit verifier once. Each
/// [`scan`](Self::scan) keeps its state local, so one query can scan any number of
/// chunks encoded with the same dictionary, concurrently if needed. The empty
/// pattern matches every row, without a scan.
///
/// # Examples
///
/// ```
/// use onpair::search::ContainsScan;
/// use onpair::search::index::TokenFrequencyIndex;
/// use onpair::{Column, DEFAULT_CONFIG, DictionaryView};
///
/// let bytes = b"catdogbirdcatfish";
/// let offsets = [0u32, 3, 6, 10, 13, 17];
/// let column = Column::compress(bytes, &offsets, DEFAULT_CONFIG)?;
/// let view = column.view();
/// let frequencies = TokenFrequencyIndex::build(view.codes, view.dict.num_tokens())?;
///
/// // Prepare once per pattern, then scan any slice of rows.
/// let scan = ContainsScan::new(b"cat", view.dict, frequencies.as_view())?;
/// let mut rows = Vec::new();
/// scan.scan(view.codes, view.row_offsets, &mut rows);
/// assert_eq!(rows, [0, 3]);
///
/// // Row indices count from the first offset passed in.
/// rows.clear();
/// scan.scan(view.codes, &view.row_offsets[2..], &mut rows);
/// assert_eq!(rows, [1]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone)]
pub struct ContainsScan<'d> {
    dict: CompactDictionaryView<'d>,
    query: Query,
}

/// What a scan does for its pattern.
#[derive(Clone)]
enum Query {
    /// The empty pattern, which every row contains.
    Everything,
    /// Search for the filter's codes and verify each hit.
    Scan {
        probe_cover: ProbeCover,
        isa: Isa,
        matcher: MatcherKind,
        verifier: HitVerifier,
        probe_density: f64,
    },
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
    /// Returns [`ContainsError::PatternTooLong`] if `pattern` is longer than
    /// [`Self::MAX_PATTERN_LEN`], [`ContainsError::FrequencyIndexMismatch`] if
    /// `frequencies` counts a different number of tokens than `dict` has, and
    /// [`ContainsError::MissingByteToken`] if `dict` lacks a single-byte token that
    /// `pattern` needs.
    ///
    /// # Precondition
    /// `dict` is conformant (sorted, unique and complete), as guaranteed by
    /// [`Parser::train`](crate::Parser::train) and
    /// [`CompactDictionary::validate`](crate::CompactDictionary::validate), and
    /// `frequencies` is indexed by its token IDs. Only the index length is checked.
    /// Otherwise the rows its scans return are unspecified.
    pub fn new(
        pattern: &[u8],
        dict: CompactDictionaryView<'d>,
        frequencies: TokenFrequencyIndexView<'_>,
    ) -> Result<Self, ContainsError> {
        if pattern.len() > Self::MAX_PATTERN_LEN {
            return Err(ContainsError::PatternTooLong {
                length: pattern.len(),
                max: Self::MAX_PATTERN_LEN,
            });
        }
        check_frequencies(dict, frequencies)?;
        if pattern.is_empty() {
            return Ok(Self {
                dict,
                query: Query::Everything,
            });
        }
        let isa = scan::detect_isa();
        let graph = AlignmentGraph::new(dict, pattern, frequencies)?;
        let plan = plan::select(&graph, frequencies, isa);
        // Following greedy edges from the source always reaches the sink, so every
        // cut, and with it the filter, is nonempty.
        debug_assert!(!plan.cover.is_empty(), "empty filter for {pattern:?}");
        Ok(Self {
            dict,
            query: Query::Scan {
                probe_cover: plan.cover,
                isa,
                matcher: plan.matcher,
                verifier: HitVerifier::from_graph(&graph, pattern),
                probe_density: fraction(plan.hits, frequencies.total_frequency()),
            },
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
    /// establish greedy tokenization. Otherwise the rows returned are unspecified,
    /// and decreasing offsets may panic with [`InvalidColumn::BadRowOffsets`]. A code
    /// outside the dictionary never matches.
    pub fn scan<O: Offset>(&self, codes: &[Token], row_offsets: &[O], out: &mut Vec<usize>) {
        let (Some(first), Some(last)) = (row_offsets.first(), row_offsets.last()) else {
            return;
        };
        let (first, last) = (first.to_usize(), last.to_usize());
        if first > last || last > codes.len() {
            panic_malformed(InvalidColumn::BadRowOffsets);
        }
        let Query::Scan {
            probe_cover,
            isa,
            matcher,
            verifier,
            probe_density,
        } = &self.query
        else {
            // Every row contains the empty pattern, even a row without codes.
            out.extend(0..row_offsets.len() - 1);
            return;
        };
        // Without codes, no row can contain the pattern.
        if first == last {
            return;
        }
        scan::run(
            *isa,
            *matcher,
            plan::skip_empty_packing(*probe_density, last - first),
            ScanInput::new(
                &codes[..last],
                row_offsets,
                self.dict,
                probe_cover,
                verifier,
            ),
            out,
        );
    }

    /// The tokens this scan searches for, never empty, or `None` for the empty
    /// pattern, which needs no search.
    pub fn probe_cover(&self) -> Option<&ProbeCover> {
        match &self.query {
            Query::Everything => None,
            Query::Scan { probe_cover, .. } => Some(probe_cover),
        }
    }

    /// How many codes of the stream counted by `frequencies` this scan would check
    /// as hits: the occurrences of the filter's tokens, zero for the empty pattern.
    /// Divided by [`total_frequency`](TokenFrequencyIndexView::total_frequency), it
    /// is the expected hit density.
    ///
    /// # Errors
    /// Returns [`ContainsError::FrequencyIndexMismatch`] if `frequencies` counts a
    /// different number of tokens than this scan's dictionary has.
    pub fn expected_hits(
        &self,
        frequencies: TokenFrequencyIndexView<'_>,
    ) -> Result<u32, ContainsError> {
        check_frequencies(self.dict, frequencies)?;
        Ok(self
            .probe_cover()
            .map_or(0, |cover| cover.frequency(frequencies)))
    }
}

/// Shows the plan: the filter, the instruction set, the matcher and the hit
/// density it was planned for. The dictionary and the verifier's tables are left
/// out.
impl fmt::Debug for ContainsScan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("ContainsScan");
        debug.field("probe_cover", &self.probe_cover());
        if let Query::Scan {
            isa,
            matcher,
            probe_density,
            ..
        } = &self.query
        {
            debug
                .field("isa", isa)
                .field("matcher", matcher)
                .field("probe_density", probe_density);
        }
        debug.finish_non_exhaustive()
    }
}

/// Rejects a frequency index that counts a different number of tokens than `dict`
/// has.
fn check_frequencies(
    dict: CompactDictionaryView<'_>,
    frequencies: TokenFrequencyIndexView<'_>,
) -> Result<(), ContainsError> {
    let (index_tokens, dict_tokens) = (frequencies.num_tokens(), dict.num_tokens());
    if index_tokens == dict_tokens {
        Ok(())
    } else {
        Err(ContainsError::FrequencyIndexMismatch {
            index_tokens,
            dict_tokens,
        })
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
