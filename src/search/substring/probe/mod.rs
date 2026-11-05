// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ContainsScan`]: substring search by scanning for a cover of the pattern.
//!
//! # Alignment graph
//!
//! The encoder tokenizes greedily: each token is the longest dictionary entry that
//! prefixes the remaining bytes. An occurrence of an `n`-byte pattern may begin
//! inside a token, but from its first token boundary on, each token either ends
//! inside the pattern, and is then the longest token prefixing the rest of the
//! pattern, or reaches past its end, and then starts with the rest of the pattern.
//!
//! Every encoding of an occurrence is therefore a path from `0` to `n` in a small
//! DAG over the pattern offsets, the *alignment graph*:
//!
//! * *alignment edges* `0 -> k`: the tokens an occurrence begins inside, which end
//!   with its first `k` bytes, and `0 -> n`: the tokens that contain it whole;
//! * at most one *greedy edge* per node: the longest token that prefixes the rest
//!   of the pattern and ends before `n`;
//! * at most one *range edge* `i -> n` per node: the tokens that start with the
//!   rest of the pattern, a contiguous ID range in the sorted dictionary.
//!
//! The graph has at most `2n + MAX_TOKEN_SIZE` edges. Finding the alignment edges
//! takes one pass over the dictionary's bytes, and every other edge one binary
//! search. Without the source and the sink, the greedy edges form a forest.
//!
//! # Cover
//!
//! A *cut* is a set of edges that every source-to-sink path crosses. The tokens on
//! its edges form the *cover* ([`ProbeCover`]): every occurrence is encoded with at
//! least one of them. Any cut is correct, so the choice only affects speed: each
//! covered code in the stream is a *hit* to locate and verify, and each point or
//! range adds matcher work on every code. `plan` prices both with a cost model per
//! instruction set, solves a minimum cut on the forest in linear time for a few
//! edge weightings, and keeps the cheapest cover with its matcher. Alignment edges
//! with too many tokens to list are never cut.
//!
//! # Scan
//!
//! The scan reads the codes in blocks of 4096. A matcher sets one mask bit per hit:
//! `Table` looks codes up in a 64 KiB table, `EqOr` compares every point, and
//! `Nibble` tests eight points at once with shuffle tables. The vector matchers
//! also check ranges, and run on NEON, AVX2 or AVX-512, chosen at runtime. The
//! resolver maps each hit to its row, and the verifier walks the row's codes
//! forward and backward from the hit along the alignment graph, for each cut edge
//! the hit token is on, until both walks spell the pattern for one of them: every
//! occurrence crosses the cut at a hit. A row is appended on its first verified
//! hit; its later hits are skipped.
//!
//! The cover and the walks both rely on greedy tokenization. Patterns made of one
//! repeated byte are the slow case on data rich in that byte: every token of their
//! cover is a run of it, so a scan verifies many hits.
//!
//! # Modules
//!
//! * `alignment`: the alignment graph and its minimum cut.
//! * `cover`: the cover of a cut, a `ProbeCover`.
//! * `plan`: the cost model that chooses the cover, its matcher and the mask packing.
//! * `scan`: the block loop, the matchers and row resolution.
//! * `verify`: the hit verifier.

mod alignment;
mod cover;
mod plan;
mod scan;
mod verify;

pub use cover::ProbeCover;

use std::fmt;
use std::sync::Arc;

use alignment::graph::AlignmentGraph;
use scan::{Isa, Matcher, MatcherKind, ScanInput};
use verify::HitVerifier;

use crate::core::dictionary::{CompactDictionaryView, DictionaryView};
use crate::core::offset::Offset;
use crate::core::types::Token;
use crate::core::validate::{InvalidColumn, panic_malformed};
use crate::search::index::TokenFrequencyIndexView;
use crate::search::substring::ContainsError;

/// Finds the rows of a code stream that contain a pattern, without decoding them.
///
/// Among the tokens that encode an occurrence of the pattern in a greedily tokenized
/// row, at least one belongs to a small set, the pattern's [cover](ProbeCover).
/// Preparation chooses a cover whose tokens are rare in a
/// [`TokenFrequencyIndexView`] and cheap to search for. [`scan`](Self::scan) finds
/// the cover's codes in the stream and confirms each of these *hits* by matching the
/// codes around it against the pattern, so the rows it returns are exact. It uses
/// SIMD where the CPU has it (NEON, AVX2 or AVX-512), detected at runtime, so no
/// build flags are needed; other CPUs use a portable fallback.
///
/// Prepare once per pattern and dictionary, then scan any number of chunks encoded
/// with that dictionary, concurrently if needed. Clones share the prepared search,
/// so cloning is cheap. The empty pattern matches every row, without a scan.
///
/// # Performance
///
/// Preparation reads the whole dictionary once, so reuse a prepared scan across
/// chunks. A scan makes one pass over the codes to find the hits, then verifies
/// each hit. The second part grows with the hit density, the fraction of codes
/// that are hits, which [`expected_hits`](Self::expected_hits) estimates before
/// scanning. Verifying a hit costs roughly as much as matching a hundred codes, so
/// from a hit density of about 1% on, verification takes most of the time. Patterns
/// made of one repeated byte, such as `aaaa`, are the slow case on data rich in that
/// byte: every token they can be found by is a run of it, so hits are many.
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
    /// Search for the cover's codes and verify each hit. Clones share it.
    Scan(Arc<Prepared>),
}

/// The search and verification prepared for a nonempty pattern.
struct Prepared {
    probe_cover: ProbeCover,
    isa: Isa,
    kind: MatcherKind,
    /// The `kind` matcher of `isa`, built for `probe_cover`.
    matcher: Box<dyn Matcher>,
    verifier: HitVerifier,
    hit_density: f64,
}

impl<'d> ContainsScan<'d> {
    /// Longest supported pattern, in bytes. Pattern offsets are stored as `u16`.
    pub const MAX_PATTERN_LEN: usize = u16::MAX as usize;

    /// Prepares a scan for `pattern`.
    ///
    /// `frequencies` counts how often each token occurs in a representative code
    /// stream. It only guides the choice of the cover: inaccurate counts can make scans
    /// slower, never incorrect.
    ///
    /// # Errors
    /// Returns [`ContainsError::PatternTooLong`] if `pattern` is longer than
    /// [`Self::MAX_PATTERN_LEN`], [`ContainsError::FrequencyIndexMismatch`] if
    /// `frequencies` counts a different number of tokens than `dict` has, and
    /// [`ContainsError::MissingByteToken`] if no single-byte token is found for a byte
    /// of `pattern`, as on an incomplete or unsorted `dict`.
    ///
    /// # Precondition
    /// `dict` is conformant (sorted, unique and complete), as guaranteed by
    /// [`Parser::train`](crate::Parser::train) and
    /// [`CompactDictionary::validate`](crate::CompactDictionary::validate).
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
        let graph = AlignmentGraph::new(dict, pattern)?;
        // Planning and verification share one list of edges, which the cut indexes.
        let edges: Vec<_> = graph.edges().collect();
        let plan = plan::select(&edges, graph.sink(), frequencies, isa);
        // The greedy path from the source, closed by a range edge, has only cuttable
        // edges, so every cut, and with it the cover, is nonempty.
        debug_assert!(!plan.cover.is_empty(), "empty cover for {pattern:?}");
        Ok(Self {
            dict,
            query: Query::Scan(Arc::new(Prepared {
                matcher: scan::prepare(isa, plan.matcher, &plan.cover),
                probe_cover: plan.cover,
                isa,
                kind: plan.matcher,
                verifier: HitVerifier::new(&graph, &edges, &plan.cut, pattern),
                hit_density: fraction(plan.hits, frequencies.total_frequency()),
            })),
        })
    }

    /// Appends to `out` the index of every row that contains the pattern, in
    /// ascending order.
    ///
    /// Row `i` is `codes[row_offsets[i]..row_offsets[i + 1]]`. Codes outside
    /// `row_offsets[0]..row_offsets[last]` belong to no row, and with non-decreasing
    /// offsets they are never read, so a view sliced from a larger column can pass
    /// its whole code buffer.
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
        let Query::Scan(prepared) = &self.query else {
            // Every row contains the empty pattern, even a row without codes.
            out.extend(0..row_offsets.len() - 1);
            return;
        };
        // Without codes, no row can contain the pattern.
        if first == last {
            return;
        }
        scan::run(
            &*prepared.matcher,
            plan::skip_empty_packing(prepared.hit_density, last - first),
            ScanInput {
                codes: &codes[..last],
                row_offsets,
                dict: self.dict,
                verifier: &prepared.verifier,
            },
            out,
        );
    }

    /// The tokens this scan searches for, never empty, or `None` for the empty
    /// pattern, which needs no search.
    pub fn probe_cover(&self) -> Option<&ProbeCover> {
        match &self.query {
            Query::Everything => None,
            Query::Scan(prepared) => Some(&prepared.probe_cover),
        }
    }

    /// How many codes of the stream counted by `frequencies` this scan would check
    /// as hits: the occurrences of the cover's tokens, zero for the empty pattern.
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

/// Shows the plan: the size of the cover, the instruction set, the matcher and the
/// hit density it was planned for, or nothing for the empty pattern. The cover
/// itself is in [`probe_cover`](Self::probe_cover), since it can run to thousands of
/// characters. The dictionary and the verifier's tables are left out.
impl fmt::Debug for ContainsScan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("ContainsScan");
        if let Query::Scan(prepared) = &self.query {
            debug
                .field("cover_points", &prepared.probe_cover.points().len())
                .field("cover_ranges", &prepared.probe_cover.ranges().len())
                .field("isa", &prepared.isa)
                .field("matcher", &prepared.kind)
                .field("hit_density", &prepared.hit_density);
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
