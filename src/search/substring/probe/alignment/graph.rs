// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The alignment graph: every way a needle can be encoded in a greedily tokenized row.
//!
//! Nodes are needle offsets, from the source `0` to the sink `n = needle.len()`. An
//! edge `from -> to` is a token covering `needle[from..to]`. There are three kinds:
//!
//! * **Alignment edges** `0 -> k` (`Set`, `UnenumeratedSet`): tokens an occurrence
//!   begins inside, which end with `needle[..k]`, or contain the whole needle if
//!   `k = n`.
//! * **Greedy edges** (`Single`): the longest token that prefixes `needle[from..]`
//!   and ends before the sink. Greedy tokenization lets no other token end inside
//!   the needle there, so each node has at most one.
//! * **Range edges** `from -> n` (`Range`): the tokens that start with
//!   `needle[from..]`, and so finish the needle. They are contiguous in the sorted
//!   dictionary.
//!
//! Every occurrence in a row is a source-to-sink path. The planner cuts all paths
//! to choose the filter, and the verifier walks the same graph to confirm a hit.

use super::starts::MatchStarts;
use crate::core::dictionary::{CompactDictionaryView, DictionaryView};
use crate::core::types::{MAX_TOKEN_SIZE, Token, TokenRange};
use crate::core::validate::InvalidColumn;
use crate::search::index::TokenFrequencyIndexView;
use crate::search::lookup::narrow;
use crate::search::substring::ContainsError;

/// The tokens an edge accepts.
#[derive(Clone, Debug)]
pub(in crate::search::substring::probe) enum EdgeKind {
    /// One token: a greedy edge.
    Single(Token),
    /// A contiguous range of tokens: a range edge.
    Range(TokenRange),
    /// An explicit list of tokens: an alignment edge.
    Set(Box<[Token]>),
    /// An alignment edge with too many tokens to list: those ending with
    /// `needle[..edge.to]`. Its tokens cannot be in a filter, so a cut never
    /// selects it.
    UnenumeratedSet,
}

/// A token step from one needle offset to a later one.
#[derive(Clone, Debug)]
pub(in crate::search::substring::probe) struct Edge {
    /// Needle offset before this token.
    pub(in crate::search::substring::probe) from: u16,
    /// Needle offset reached after this token.
    pub(in crate::search::substring::probe) to: u16,
    kind: EdgeKind,
    /// Occurrences of the accepted tokens in the indexed stream; zero for
    /// unenumerated sets.
    frequency: u32,
}

impl Edge {
    /// Whether the edge's tokens are listed, and so can be in a filter.
    pub(in crate::search::substring::probe) fn cuttable(&self) -> bool {
        !matches!(self.kind, EdgeKind::UnenumeratedSet)
    }

    /// Representation of the tokens accepted by this edge.
    pub(in crate::search::substring::probe) fn kind(&self) -> &EdgeKind {
        &self.kind
    }

    /// Occurrences of the accepted tokens in the indexed stream.
    pub(in crate::search::substring::probe) fn frequency(&self) -> u32 {
        self.frequency
    }

    /// Number of point probes this edge adds to a cover, before normalization.
    pub(in crate::search::substring::probe) fn point_count(&self) -> usize {
        match &self.kind {
            EdgeKind::Single(_) => 1,
            EdgeKind::Set(ids) => ids.len(),
            EdgeKind::Range(_) | EdgeKind::UnenumeratedSet => 0,
        }
    }

    /// Number of range probes this edge adds to a cover, before normalization.
    pub(in crate::search::substring::probe) fn range_count(&self) -> usize {
        usize::from(matches!(self.kind, EdgeKind::Range(_)))
    }
}

/// The alignment graph of one needle, as a flat edge list.
///
/// A needle of `n` bytes has `n + 1` nodes and at most `2n + MAX_TOKEN_SIZE` edges.
/// Each internal node has at most one greedy edge, so removing the source
/// and sink leaves a forest. The cut solver relies on this.
pub(in crate::search::substring::probe) struct AlignmentGraph {
    /// Edges in construction order, which is also the order of cut output.
    pub(in crate::search::substring::probe) edges: Vec<Edge>,
    /// The offset after the whole needle.
    sink: u16,
}

impl AlignmentGraph {
    /// Builds the graph of `needle` and weights each edge by `frequencies`.
    ///
    /// The caller checks that the needle is nonempty and at most
    /// `ContainsScan::MAX_PATTERN_LEN` bytes, that the dictionary is conformant,
    /// and that `frequencies` is indexed by its token IDs.
    ///
    /// # Errors
    /// Returns [`ContainsError::InvalidData`] with
    /// [`InvalidColumn::IncompleteAlphabet`] if no token prefixes some suffix of the
    /// needle.
    pub(in crate::search::substring::probe) fn new(
        dict: CompactDictionaryView<'_>,
        needle: &[u8],
        frequencies: TokenFrequencyIndexView<'_>,
    ) -> Result<Self, ContainsError> {
        let mut builder = GraphBuilder {
            dict,
            needle,
            frequencies,
            edges: Vec::new(),
            built: vec![false; needle.len()],
        };
        let starts = MatchStarts::new(dict, needle);

        // Start with matches beginning at a token boundary.
        builder.build_path(0)?;

        // Build each remaining path before its alignment edge to preserve the
        // construction order used by cut output and verification.
        for length in 1..needle.len().min(MAX_TOKEN_SIZE) {
            let Some(kind) = starts.overlap(length) else {
                continue;
            };
            builder.build_path(length)?;
            builder.add_edge(0, length, kind);
        }

        // Whole matches inside a token go directly to the sink. Tokens starting
        // with the needle are already covered by the range from the source.
        if !starts.contained.is_empty() {
            builder.add_edge(0, needle.len(), EdgeKind::Set(starts.contained.into()));
        }

        Ok(Self {
            edges: builder.edges,
            sink: needle.len() as u16,
        })
    }

    /// Number of nodes, including offsets that no edge reaches.
    pub(in crate::search::substring::probe) fn node_count(&self) -> usize {
        usize::from(self.sink()) + 1
    }

    /// The offset after the whole needle. The source is always 0.
    pub(in crate::search::substring::probe) fn sink(&self) -> u16 {
        self.sink
    }
}

/// Graph construction state; each reachable offset is built once.
struct GraphBuilder<'d, 'n, 'f> {
    dict: CompactDictionaryView<'d>,
    needle: &'n [u8],
    frequencies: TokenFrequencyIndexView<'f>,
    edges: Vec<Edge>,
    /// Offsets whose outgoing edges have been emitted.
    built: Vec<bool>,
}

impl GraphBuilder<'_, '_, '_> {
    /// Occurrences of the edge's tokens. A set lists distinct tokens, so the sum
    /// cannot exceed the stream length.
    fn frequency_of(&self, kind: &EdgeKind) -> u32 {
        match kind {
            EdgeKind::Single(id) => self.frequencies.frequency(*id),
            EdgeKind::Range(range) => self.frequencies.range_frequency(*range),
            EdgeKind::Set(ids) => ids.iter().map(|&id| self.frequencies.frequency(id)).sum(),
            EdgeKind::UnenumeratedSet => 0,
        }
    }

    /// Append an edge with the frequency of its accepted tokens.
    /// Offsets are at most the needle length, which fits `u16`.
    fn add_edge(&mut self, from: usize, to: usize, kind: EdgeKind) {
        let frequency = self.frequency_of(&kind);
        self.edges.push(Edge {
            from: from as u16,
            to: to as u16,
            kind,
            frequency,
        });
    }

    /// Adds the greedy path from `start`, stopping at the first offset already built.
    /// The loop is iterative, so stack use does not grow with the needle.
    fn build_path(&mut self, start: usize) -> Result<(), ContainsError> {
        let n = self.needle.len();
        let mut offset = start;
        while !self.built[offset] {
            self.built[offset] = true;
            let next = self.build_position(offset)?;
            if next >= n {
                break;
            }
            offset = next;
        }
        Ok(())
    }

    /// Adds the edges leaving `offset`, and returns the offset after the longest token
    /// that prefixes the rest of the needle. Fails if there is no such token.
    fn build_position(&mut self, offset: usize) -> Result<usize, ContainsError> {
        let remaining = &self.needle[offset..];
        let num_tokens = self.dict.num_tokens();
        if num_tokens == 0 {
            return Err(ContainsError::InvalidData(
                InvalidColumn::IncompleteAlphabet,
            ));
        }

        // Narrow by successive bytes, remembering the longest exact token.
        let mut range = TokenRange {
            begin: 0,
            last: (num_tokens - 1) as Token,
        };
        let mut longest = None;
        for (k, &byte) in remaining.iter().take(MAX_TOKEN_SIZE).enumerate() {
            range = narrow(self.dict, range, k, byte);
            if range.is_empty() {
                break;
            }
            // An exact prefix, if present, sorts first in this range.
            if self.dict.token_len(range.begin) == k + 1 {
                longest = Some((range.begin, k + 1));
            }
        }

        // If the whole remainder fits, this range finishes the match.
        if remaining.len() <= MAX_TOKEN_SIZE && !range.is_empty() {
            self.add_edge(offset, self.needle.len(), EdgeKind::Range(range));
        }

        let Some((token, token_length)) = longest else {
            return Err(ContainsError::InvalidData(
                InvalidColumn::IncompleteAlphabet,
            ));
        };

        let next_offset = offset + token_length;
        if next_offset < self.needle.len() {
            self.add_edge(offset, next_offset, EdgeKind::Single(token));
        }
        // An exact match of the remainder is already in the range above.
        Ok(next_offset)
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::core::dictionary::{CompactDictionary, Dictionary, pad_raw};

    /// Test graph; callers supply edges satisfying the alignment topology.
    pub(in crate::search::substring::probe::alignment) fn synthetic_graph(
        node_count: usize,
        edges: Vec<Edge>,
    ) -> AlignmentGraph {
        AlignmentGraph {
            edges,
            sink: (node_count - 1) as u16,
        }
    }

    /// Test edge; `None` makes it uncuttable.
    pub(in crate::search::substring::probe::alignment) fn synthetic_edge(
        from: u16,
        to: u16,
        frequency: Option<u32>,
    ) -> Edge {
        Edge {
            from,
            to,
            kind: frequency.map_or(EdgeKind::UnenumeratedSet, |_| EdgeKind::Single(Token::MAX)),
            frequency: frequency.unwrap_or(0),
        }
    }

    fn dictionary(tokens: &[&[u8]]) -> CompactDictionary {
        let mut tokens = tokens.to_vec();
        tokens.sort_unstable();
        tokens.dedup();
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for token in tokens {
            bytes.extend_from_slice(token);
            offsets.push(bytes.len() as u32);
        }
        pad_raw(&mut bytes, &offsets);
        CompactDictionary::from_raw(bytes, offsets)
    }

    /// Check one position against token bytes, including edge order and weights.
    fn check_position(dict: CompactDictionaryView<'_>, needle: &[u8], offset: usize) {
        let remaining = &needle[offset..];
        let ids = (0..dict.num_tokens()).map(|id| id as Token);
        let next = ids
            .clone()
            .filter(|&id| remaining.starts_with(dict.token(id)))
            .max_by_key(|&id| dict.token_len(id));
        let terminal: Vec<_> = ids
            .filter(|&id| dict.token(id).starts_with(remaining))
            .collect();

        // One occurrence per token makes every edge's weight its token count.
        let cumulative: Vec<u32> = (0..=dict.num_tokens() as u32).collect();
        let frequencies = TokenFrequencyIndexView::validate_safety(
            &cumulative,
            dict.num_tokens(),
            dict.num_tokens(),
        )
        .unwrap();
        let mut builder = GraphBuilder {
            dict,
            needle,
            frequencies,
            edges: Vec::new(),
            built: vec![false; needle.len()],
        };
        let expected_next =
            next.map(|id| offset + dict.token_len(id))
                .ok_or(ContainsError::InvalidData(
                    InvalidColumn::IncompleteAlphabet,
                ));
        assert_eq!(
            builder.build_position(offset),
            expected_next,
            "{needle:?}, offset={offset}"
        );

        let mut expected = Vec::new();
        if !terminal.is_empty() {
            expected.push((offset as u16, needle.len() as u16, terminal));
        }
        if let Some(id) = next.filter(|&id| dict.token_len(id) < remaining.len()) {
            expected.push((
                offset as u16,
                (offset + dict.token_len(id)) as u16,
                vec![id],
            ));
        }
        let actual: Vec<_> = builder
            .edges
            .iter()
            .map(|edge| {
                let ids: Vec<Token> = match edge.kind {
                    EdgeKind::Single(id) => vec![id],
                    EdgeKind::Range(range) => (range.begin..=range.last).collect(),
                    _ => panic!("unexpected edge: {edge:?}"),
                };
                assert_eq!(edge.frequency, ids.len() as u32);
                (edge.from, edge.to, ids)
            })
            .collect();
        assert_eq!(actual, expected, "{needle:?}, offset={offset}");
    }

    #[test]
    fn greedy_and_terminal_edges() {
        let dict = dictionary(&[b"app", b"appapple", b"applepie"]);
        check_position(dict.as_view(), b"apple", 0); // Both a range edge and a greedy edge.
        check_position(dict.as_view(), b"appappapple", 0); // Greedy step after narrowing runs out.
        check_position(dict.as_view(), b"app", 0); // Exact token: no duplicate greedy edge.
        check_position(dict.as_view(), b"xxxapple", 3); // Edges use absolute needle offsets.
    }

    #[test]
    fn token_length_boundary() {
        let longest = [b'a'; MAX_TOKEN_SIZE];
        let tokens: Vec<_> = (1..=MAX_TOKEN_SIZE).map(|len| &longest[..len]).collect();
        let dict = dictionary(&tokens);
        for len in [MAX_TOKEN_SIZE, MAX_TOKEN_SIZE + 1, 255] {
            check_position(dict.as_view(), &vec![b'a'; len], 0);
        }
    }

    #[test]
    fn missing_token_prefix_is_an_error() {
        check_position(dictionary(&[]).as_view(), b"apple", 0);
        let dict = dictionary(&[b"applepie"]);
        check_position(dict.as_view(), b"apple", 0); // A range edge but no greedy edge.
        check_position(dict.as_view(), b"absent", 0); // Neither kind of match.
    }

    #[test]
    fn token_prefixes_and_extensions() {
        let mut tokens: Vec<Vec<u8>> = (1..=MAX_TOKEN_SIZE).map(|len| vec![b'a'; len]).collect();
        tokens.extend([
            b"app".to_vec(),
            b"appapple".to_vec(),
            b"applepie".to_vec(),
            vec![0, 255],
        ]);
        for complete in [false, true] {
            if complete {
                tokens.extend((0..=255).map(|byte| vec![byte]));
            }
            let refs: Vec<_> = tokens.iter().map(Vec::as_slice).collect();
            let dict = dictionary(&refs);
            for token in &tokens {
                for len in 1..=token.len() {
                    check_position(dict.as_view(), &token[..len], 0);
                }
                let mut longer = token.clone();
                longer.push(0);
                check_position(dict.as_view(), &longer, 0);
            }
        }
    }
}
