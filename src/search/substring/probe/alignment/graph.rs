// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The alignment graph: every way a needle can be encoded in a greedily tokenized row.
//!
//! Nodes are needle offsets, from the source `0` to the sink `n = needle.len()`. An
//! edge `from -> to` is a token covering `needle[from..to]`. There are three kinds:
//!
//! * **Alignment edges** `0 -> k`: tokens an occurrence begins inside, which end with
//!   `needle[..k]`, or contain the whole needle if `k = n`.
//! * **Greedy edges**: the longest token that prefixes `needle[from..]` and ends
//!   before the sink. Greedy tokenization lets no other token end inside the needle
//!   there, so each node has at most one.
//! * **Range edges** `from -> n`: the tokens that start with `needle[from..]`, and so
//!   finish the needle. They are contiguous in the sorted dictionary.
//!
//! The graph is stored in that shape: each node's greedy and range edge, and the
//! alignment edges by length. Without the source and the sink, the greedy edges
//! form a forest, which the cut solver relies on.
//!
//! Every occurrence in a row is a source-to-sink path. The planner cuts all paths
//! to choose the filter, and the verifier walks the same graph to confirm a hit.

use super::starts::MatchStarts;
use crate::core::dictionary::CompactDictionaryView;
use crate::core::types::{MAX_TOKEN_SIZE, Token, TokenRange};
use crate::search::lookup::longest_prefix;
use crate::search::substring::ContainsError;

/// The alignment graph of one needle.
pub(crate) struct AlignmentGraph {
    /// The edges leaving each offset before the sink. Offsets no path reaches have
    /// none.
    pub(crate) nodes: Vec<Node>,
    /// The alignment edge `0 -> k` of each overlap length `k`, below both the needle
    /// length and `MAX_TOKEN_SIZE`. Index 0 is unused.
    pub(crate) overlaps: Vec<Option<Overlap>>,
    /// The tokens that contain the whole needle but do not start with it: the
    /// alignment edge `0 -> n`, absent if empty. Tokens starting with the needle are
    /// on the source's range edge.
    pub(crate) contained: Box<[Token]>,
}

/// The edges leaving one offset.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Node {
    /// The greedy edge: its token and the offset after it.
    pub(crate) greedy: Option<(Token, u16)>,
    /// The range edge: the tokens that start with the rest of the needle.
    pub(crate) range: Option<TokenRange>,
}

/// The alignment edge of one overlap length.
#[derive(Clone, Debug)]
pub(crate) enum Overlap {
    /// The tokens that end with the overlapped needle prefix.
    Set(Box<[Token]>),
    /// Too many tokens to list. They cannot be in a filter, so a cut never selects
    /// the edge.
    UnenumeratedSet,
}

/// One edge of the graph, as listed by [`AlignmentGraph::edges`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct Edge<'g> {
    /// Needle offset before this token.
    pub(crate) from: u16,
    /// Needle offset reached after this token.
    pub(crate) to: u16,
    pub(crate) kind: EdgeKind<'g>,
}

/// The tokens an edge accepts.
#[derive(Clone, Copy, Debug)]
pub(crate) enum EdgeKind<'g> {
    /// One token: a greedy edge.
    Single(Token),
    /// A contiguous range of tokens: a range edge.
    Range(TokenRange),
    /// An explicit list of tokens: an alignment edge.
    Set(&'g [Token]),
    /// An alignment edge with too many tokens to list: those ending with
    /// `needle[..edge.to]`.
    UnenumeratedSet,
}

impl Edge<'_> {
    /// Whether the edge's tokens are listed, and so can be in a filter.
    pub(crate) fn cuttable(&self) -> bool {
        !matches!(self.kind, EdgeKind::UnenumeratedSet)
    }
}

impl AlignmentGraph {
    /// Builds the graph of `needle`.
    ///
    /// The caller checks that the needle is nonempty and at most
    /// `ContainsScan::MAX_PATTERN_LEN` bytes, and that the dictionary is conformant.
    ///
    /// # Errors
    /// Returns [`ContainsError::MissingByteToken`] if no token prefixes some suffix of
    /// the needle, which happens only if the dictionary lacks a single-byte token.
    pub(crate) fn new(
        dict: CompactDictionaryView<'_>,
        needle: &[u8],
    ) -> Result<Self, ContainsError> {
        let mut builder = GraphBuilder {
            dict,
            needle,
            nodes: vec![Node::default(); needle.len()],
            built: vec![false; needle.len()],
        };
        let starts = MatchStarts::new(dict, needle);

        // Paths from matches beginning at a token boundary, then from each overlap.
        builder.build_path(0)?;
        let overlaps: Vec<_> = (0..needle.len().min(MAX_TOKEN_SIZE))
            .map(|length| starts.overlap(length))
            .collect();
        for (length, overlap) in overlaps.iter().enumerate() {
            if overlap.is_some() {
                builder.build_path(length)?;
            }
        }

        Ok(Self {
            nodes: builder.nodes,
            overlaps,
            contained: starts.contained.into(),
        })
    }

    /// The offset after the whole needle. The source is always 0.
    pub(crate) fn sink(&self) -> u16 {
        // At most `ContainsScan::MAX_PATTERN_LEN`, which fits `u16`.
        self.nodes.len() as u16
    }

    /// Every edge, in a fixed order: the greedy and range edges of each offset in
    /// turn, then the alignment edges by length.
    pub(crate) fn edges(&self) -> impl Iterator<Item = Edge<'_>> {
        let sink = self.sink();
        // Offsets fit `u16`, as `sink` does.
        let node_edges = self.nodes.iter().enumerate().flat_map(move |(from, node)| {
            let from = from as u16;
            let greedy = node.greedy.map(|(token, to)| Edge {
                from,
                to,
                kind: EdgeKind::Single(token),
            });
            let range = node.range.map(|range| Edge {
                from,
                to: sink,
                kind: EdgeKind::Range(range),
            });
            greedy.into_iter().chain(range)
        });
        let overlaps = self
            .overlaps
            .iter()
            .enumerate()
            .filter_map(|(to, overlap)| {
                let kind = match overlap.as_ref()? {
                    Overlap::Set(tokens) => EdgeKind::Set(tokens),
                    Overlap::UnenumeratedSet => EdgeKind::UnenumeratedSet,
                };
                Some(Edge {
                    from: 0,
                    to: to as u16,
                    kind,
                })
            });
        let contained = (!self.contained.is_empty()).then_some(Edge {
            from: 0,
            to: sink,
            kind: EdgeKind::Set(&self.contained),
        });
        node_edges.chain(overlaps).chain(contained)
    }
}

/// Graph construction state; each reachable offset is built once.
struct GraphBuilder<'d, 'n> {
    dict: CompactDictionaryView<'d>,
    needle: &'n [u8],
    nodes: Vec<Node>,
    /// Offsets whose outgoing edges have been set.
    built: Vec<bool>,
}

impl GraphBuilder<'_, '_> {
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

    /// Sets the edges leaving `offset`, and returns the offset after the longest token
    /// that prefixes the rest of the needle. Fails if there is no such token.
    fn build_position(&mut self, offset: usize) -> Result<usize, ContainsError> {
        let remaining = &self.needle[offset..];
        let (longest, starting_with) = longest_prefix(self.dict, remaining);
        let node = &mut self.nodes[offset];
        // The tokens that start with the whole remainder finish the match.
        if !starting_with.is_empty() {
            node.range = Some(starting_with);
        }

        let Some((token, token_length)) = longest else {
            return Err(ContainsError::MissingByteToken { byte: remaining[0] });
        };

        let next_offset = offset + token_length;
        // An exact match of the remainder is already in the range above. Offsets are
        // at most the needle length, which fits `u16`.
        if next_offset < self.needle.len() {
            node.greedy = Some((token, next_offset as u16));
        }
        Ok(next_offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dictionary::{CompactDictionary, Dictionary, DictionaryView, pad_raw};

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

    /// Check one position's edges against token bytes.
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

        let mut builder = GraphBuilder {
            dict,
            needle,
            nodes: vec![Node::default(); needle.len()],
            built: vec![false; needle.len()],
        };
        let expected_next = next
            .map(|id| offset + dict.token_len(id))
            .ok_or(ContainsError::MissingByteToken { byte: remaining[0] });
        assert_eq!(
            builder.build_position(offset),
            expected_next,
            "{needle:?}, offset={offset}"
        );

        let node = builder.nodes[offset];
        let greedy = next
            .filter(|&id| dict.token_len(id) < remaining.len())
            .map(|id| (id, (offset + dict.token_len(id)) as u16));
        assert_eq!(node.greedy, greedy, "{needle:?}, offset={offset}");
        let range: Vec<Token> = node
            .range
            .map_or(Vec::new(), |range| (range.begin..=range.last).collect());
        assert_eq!(range, terminal, "{needle:?}, offset={offset}");
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
