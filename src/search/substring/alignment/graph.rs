// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Token alignments for one needle and dictionary.
//!
//! Nodes are byte offsets into the needle, from source 0 to sink `needle.len()`.
//! Each edge consumes one token; its kind describes which token IDs it accepts.
//! An occurrence in a greedily tokenized row follows a source-to-sink path.
//!
//! The planner cuts every path to choose scan probes. The verifier compiles
//! the same graph to check matches around a probe hit.

use super::starts::MatchStarts;
use crate::core::dictionary::{CompactDictionaryView, DictionaryView};
use crate::core::types::{MAX_TOKEN_SIZE, Token, TokenRange};
use crate::core::validate::InvalidColumn;
use crate::search::index::TokenFrequencyIndexView;
use crate::search::lookup::narrow;
use crate::search::substring::ContainsError;

/// How the tokens accepted by an edge are represented.
#[derive(Clone, Debug)]
pub(in crate::search::substring) enum EdgeKind {
    /// One token ID.
    Single(Token),
    /// A contiguous range of token IDs.
    Range(TokenRange),
    /// An explicit set of token IDs.
    Set(Box<[Token]>),
    /// Tokens whose suffix matches `needle[..edge.to]`, without enumerating
    /// their IDs. Only used for partial source edges; a cut cannot select them.
    UnenumeratedSet,
}

/// A token step between two needle offsets.
#[derive(Clone, Debug)]
pub(in crate::search::substring) struct Edge {
    /// Needle offset before this token.
    pub(in crate::search::substring) from: u32,
    /// Needle offset reached after this token.
    pub(in crate::search::substring) to: u32,
    kind: EdgeKind,
    /// Indexed frequency of the accepted tokens; zero for unenumerated sets.
    frequency: u32,
}

impl Edge {
    /// Whether the edge's tokens can be selected as scan probes.
    pub(in crate::search::substring) fn cuttable(&self) -> bool {
        !matches!(self.kind, EdgeKind::UnenumeratedSet)
    }

    /// Representation of the tokens accepted by this edge.
    pub(in crate::search::substring) fn kind(&self) -> &EdgeKind {
        &self.kind
    }

    /// Indexed frequency of the accepted tokens.
    pub(in crate::search::substring) fn frequency(&self) -> u32 {
        self.frequency
    }

    /// Number of point probes before cover normalization.
    pub(in crate::search::substring) fn point_count(&self) -> u32 {
        match &self.kind {
            EdgeKind::Single(_) => 1,
            EdgeKind::Set(ids) => ids.len() as u32,
            EdgeKind::Range(_) | EdgeKind::UnenumeratedSet => 0,
        }
    }

    /// Number of range probes before cover normalization.
    pub(in crate::search::substring) fn range_count(&self) -> u32 {
        u32::from(matches!(self.kind, EdgeKind::Range(_)))
    }
}

/// Possible token alignments, stored as a flat edge list.
///
/// A needle of `n` bytes has `n + 1` logical nodes and at most
/// `2n + MAX_TOKEN_SIZE` edges. Explicit token lists have separate storage.
pub(in crate::search::substring) struct AlignmentGraph {
    /// Parse steps in construction order, which also breaks minimum-cut ties.
    pub(in crate::search::substring) edges: Vec<Edge>,
    needle_len: usize,
}

impl AlignmentGraph {
    /// Build the graph and attach indexed token frequencies.
    ///
    /// Requires a nonempty needle fitting the walker's `u16` node IDs, a
    /// conformant dictionary, and an index using that dictionary's token IDs.
    /// Query preparation checks the needle length and index size.
    pub(in crate::search::substring) fn new(
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

        // Build each remaining path before its source edge to preserve the
        // edge order used to break minimum-cut ties.
        for length in 1..needle.len().min(MAX_TOKEN_SIZE) {
            let Some(kind) = starts.overlap(length) else {
                continue;
            };
            builder.build_path(length)?;
            builder.add_edge(0, length as u32, kind);
        }

        // Whole matches inside a token go directly to the sink. Tokens starting
        // with the needle are already covered by the range from the source.
        if !starts.contained.is_empty() {
            builder.add_edge(
                0,
                needle.len() as u32,
                EdgeKind::Set(starts.contained.into()),
            );
        }

        Ok(Self {
            edges: builder.edges,
            needle_len: needle.len(),
        })
    }

    /// Logical node count, including offsets no edge reaches.
    pub(in crate::search::substring) fn node_count(&self) -> usize {
        self.sink() as usize + 1
    }

    /// Offset after the whole needle has matched. The source is always 0.
    pub(in crate::search::substring) fn sink(&self) -> u32 {
        self.needle_len as u32
    }
}

/// Temporary state for building each reached needle offset once.
struct GraphBuilder<'d, 'n, 'f> {
    dict: CompactDictionaryView<'d>,
    needle: &'n [u8],
    frequencies: TokenFrequencyIndexView<'f>,
    edges: Vec<Edge>,
    /// Offsets whose outgoing edges have been emitted.
    built: Vec<bool>,
}

impl GraphBuilder<'_, '_, '_> {
    /// Term frequency of the edge's tokens. An explicit set has distinct IDs,
    /// so its sum cannot exceed the code stream's length.
    fn frequency_of(&self, kind: &EdgeKind) -> u32 {
        match kind {
            EdgeKind::Single(id) => self.frequencies.frequency(*id),
            EdgeKind::Range(range) => self.frequencies.range_frequency(*range),
            EdgeKind::Set(ids) => ids.iter().map(|&id| self.frequencies.frequency(id)).sum(),
            EdgeKind::UnenumeratedSet => 0,
        }
    }

    /// Append an edge with the frequency of its accepted tokens.
    fn add_edge(&mut self, from: u32, to: u32, kind: EdgeKind) {
        let frequency = self.frequency_of(&kind);
        self.edges.push(Edge {
            from,
            to,
            kind,
            frequency,
        });
    }

    /// Add the greedy path from `start`, reusing positions already built.
    /// Iteration keeps call-stack usage independent of needle length.
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

    /// Add outgoing edges and return the next greedy offset.
    /// Fails if no dictionary token is a prefix of the remaining needle.
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
        let from = offset as u32;
        if remaining.len() <= MAX_TOKEN_SIZE && !range.is_empty() {
            self.add_edge(from, self.needle.len() as u32, EdgeKind::Range(range));
        }

        let Some((token, token_length)) = longest else {
            return Err(ContainsError::InvalidData(
                InvalidColumn::IncompleteAlphabet,
            ));
        };

        let next_offset = offset + token_length;
        if next_offset < self.needle.len() {
            self.add_edge(from, next_offset as u32, EdgeKind::Single(token));
        }
        // An exact match of the remainder is already in the range above.
        Ok(next_offset)
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::core::dictionary::{CompactDictionary, Dictionary, pad_raw};

    /// Test edge; `None` makes it uncuttable.
    pub(in crate::search::substring::alignment) fn synthetic_edge(
        from: u32,
        to: u32,
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
            expected.push((offset as u32, needle.len() as u32, terminal));
        }
        if let Some(id) = next.filter(|&id| dict.token_len(id) < remaining.len()) {
            expected.push((
                offset as u32,
                (offset + dict.token_len(id)) as u32,
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
        check_position(dict.as_view(), b"apple", 0); // Both a terminal range and a greedy step.
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
        check_position(dict.as_view(), b"apple", 0); // Terminal match but no greedy token.
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
