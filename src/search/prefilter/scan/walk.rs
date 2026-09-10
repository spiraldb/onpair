// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Exact verification of a prefilter hit in the compressed domain.
//!
//! A hit is a covered token at a code index. The alignment graph lists the
//! parse steps that token can be, and each step fixes the node on either side
//! of it, which fixes the codes the encoder produced around it. The forward
//! walk takes the greedy step out of each node until it reaches a terminal
//! range. The backward walk takes the step into each node until it reaches
//! the source, or an occurrence that starts inside the token entering that
//! node. An occurrence exists iff some edge of the hit token passes both
//! walks. No byte is decoded.
//!
//! Neither walk branches. A node has one greedy step and at most one terminal
//! range, and they are disjoint. Backward, the preceding code is looked up in
//! the token table and at most one of its edges enters the current node. Only
//! a set the planner did not enumerate needs the node's needle prefix,
//! compared against the token's tail from the dictionary.
//!
//! This is sound for the same reason the cover is. The encoder's parse of any
//! occurrence is one source-to-sink path of the graph.

use crate::core::dictionary::CompactDictionaryView;
use crate::core::types::{Token, TokenRange};
use crate::search::prefilter::graph::{AlignmentGraph, ProbeSet};

const SOURCE: u32 = 0;

/// A node of the graph: the needle offset a parse has reached.
#[derive(Debug, Clone, Default)]
struct Node {
    /// The one greedy step out: its token and the node it lands on.
    greedy_step: Option<(Token, u32)>,
    /// The tokens that finish the needle from here, one step into the sink.
    terminal_range: Option<TokenRange>,
    /// The needle prefix this node stands for, set only where the planner
    /// gave up enumerating the tokens that enter here: the entry test is then
    /// whether the token's tail is that prefix. An enumerated set needs no
    /// prefix, its members being edges of the token table like any other.
    entry: Option<NeedlePrefix>,
}

/// `needle[..len]` as a little-endian integer, matched against the last
/// `len` bytes of a token's dictionary window.
#[derive(Debug, Clone, Copy)]
struct NeedlePrefix {
    len: usize,
    bytes: u128,
    mask: u128,
}

impl NeedlePrefix {
    fn new(prefix: &[u8]) -> Self {
        let mut bytes = [0u8; 16];
        bytes[..prefix.len()].copy_from_slice(prefix);
        Self {
            len: prefix.len(),
            bytes: u128::from_le_bytes(bytes),
            mask: u128::MAX >> (8 * (16 - prefix.len())),
        }
    }

    #[inline]
    fn is_tail_of(&self, dict: CompactDictionaryView<'_>, token: Token) -> bool {
        let (token_len, window) = dict.token_window(token);
        token_len >= self.len
            && (u128::from_le_bytes(window) >> (8 * (token_len - self.len))) & self.mask
                == self.bytes
    }
}

/// An edge of the alignment graph as the walk reads it: the node the parse
/// is at before the token and the node it is at after.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Edge {
    from: u16,
    to: u16,
}

impl Edge {
    fn new(from: u32, to: u32) -> Self {
        debug_assert!(from < to, "an edge advances the needle offset");
        debug_assert!(
            to <= u32::from(u16::MAX),
            "needle outgrew the edge's node ids"
        );
        Self {
            from: from as u16,
            to: to as u16,
        }
    }
}

/// The edges every token lies on, four bytes per token: the edge itself, or
/// `NONE`, or `LISTED` for a token on more than one edge, which `listed`
/// holds. Every edge has `from < to`, so neither sentinel is a real one.
#[derive(Debug, Clone, Default)]
struct EdgesByToken {
    first_token: Token,
    // one lookup buffer where we can check if a token is on a single edge,
    edge_of_token: Vec<Edge>,
    // if we have more than one edge per token
    edges_of_tokens: Vec<(Token, Edge)>,
}

const NONE: Edge = Edge { from: 0, to: 0 };
const LISTED: Edge = Edge {
    from: u16::MAX,
    to: u16::MAX,
};

impl EdgesByToken {
    fn new(mut token_edges: Vec<(Token, Edge)>) -> Self {
        token_edges.sort_unstable();
        token_edges.dedup();
        let (Some(&(first_token, _)), Some(&(last_token, _))) =
            (token_edges.first(), token_edges.last())
        else {
            return Self::default();
        };
        let mut edges = Self {
            first_token,
            edge_of_token: vec![NONE; usize::from(last_token - first_token) + 1],
            edges_of_tokens: Vec::new(),
        };
        for same_token in token_edges.chunk_by(|a, b| a.0 == b.0) {
            let (token, edge) = same_token[0];
            edges.edge_of_token[usize::from(token - first_token)] = if same_token.len() == 1 {
                edge
            } else {
                edges.edges_of_tokens.extend_from_slice(same_token);
                LISTED
            };
        }
        edges
    }

    #[inline]
    fn of(&self, token: Token) -> impl Iterator<Item = Edge> + '_ {
        let at = usize::from(token.wrapping_sub(self.first_token));
        let (single, listed) = match self.edge_of_token.get(at).copied().unwrap_or(NONE) {
            NONE => (None, &self.edges_of_tokens[..0]),
            LISTED => (None, self.listed_for(token)),
            edge => (Some(edge), &self.edges_of_tokens[..0]),
        };
        single
            .into_iter()
            .chain(listed.iter().map(|&(_, edge)| edge))
    }

    fn listed_for(&self, token: Token) -> &[(Token, Edge)] {
        let first = self.edges_of_tokens.partition_point(|&(t, _)| t < token);
        let count = self.edges_of_tokens[first..]
            .iter()
            .take_while(|&&(t, _)| t == token)
            .count();
        &self.edges_of_tokens[first..first + count]
    }
}

/// The alignment graph flattened for walking from a hit.
#[derive(Debug, Clone)]
pub(in crate::search::prefilter) struct Walk {
    nodes: Vec<Node>,
    edges: EdgesByToken,
}

impl Walk {
    /// Flattens `graph`. Every graph the planner builds has a walk:
    /// `analyze_prefilter` caps the needle so its node ids fit an [`Edge`].
    pub(in crate::search::prefilter) fn from_graph(graph: &AlignmentGraph, needle: &[u8]) -> Self {
        let mut nodes = vec![Node::default(); graph.nodes.count()];
        let mut token_edges: Vec<(Token, Edge)> = Vec::new();
        for edge in &graph.edges {
            let (from, to) = (edge.from, edge.to);
            match edge.probe() {
                ProbeSet::Point(token) => {
                    let token = *token;
                    let node = &mut nodes[from as usize];
                    debug_assert!(
                        node.greedy_step.is_none(),
                        "two greedy steps out of one node"
                    );
                    node.greedy_step = Some((token, to));
                    token_edges.push((token, Edge::new(from, to)));
                }
                ProbeSet::Range(range) => {
                    nodes[from as usize].terminal_range = Some(*range);
                    token_edges.extend(
                        (range.begin..=range.last).map(|token| (token, Edge::new(from, to))),
                    );
                }
                // An enumerated set holds every token that can enter `to`,
                // so its members in the token table answer the entry test on
                // their own and the node needs no prefix.
                ProbeSet::Set(ids) => {
                    token_edges.extend(ids.iter().map(|&token| (token, Edge::new(from, to))));
                }
                ProbeSet::SetTooBig => {
                    nodes[to as usize].entry = Some(NeedlePrefix::new(&needle[..to as usize]));
                }
            }
        }
        let edges = EdgesByToken::new(token_edges);
        Self { nodes, edges }
    }

    fn sink(&self) -> u32 {
        (self.nodes.len() - 1) as u32
    }

    /// Whether an occurrence of the needle inside `codes[row_start..row_end]`
    /// has the covered token at `hit_code_index` as one of its parse steps.
    #[inline]
    pub(super) fn check(
        &self,
        dict: CompactDictionaryView<'_>,
        codes: &[Token],
        row_start: usize,
        row_end: usize,
        hit_code_index: usize,
    ) -> bool {
        let hit = codes[hit_code_index];
        for edge in self.edges.of(hit) {
            let (from, to) = (u32::from(edge.from), u32::from(edge.to));
            if (to == self.sink() || self.forward(to, hit_code_index + 1, codes, row_end))
                && self.backward(from, hit_code_index, codes, row_start, dict)
            {
                return true;
            }
        }
        false
    }

    /// The greedy path from `node`, reading `codes[code_index..row_end]`,
    /// reaches the sink.
    fn forward(
        &self,
        mut node: u32,
        mut code_index: usize,
        codes: &[Token],
        row_end: usize,
    ) -> bool {
        while code_index < row_end {
            let code = codes[code_index];
            let at = &self.nodes[node as usize];
            if at.terminal_range.is_some_and(|range| range.contains(code)) {
                return true;
            }
            match at.greedy_step {
                Some((step_token, next_node)) if step_token == code => {
                    node = next_node;
                    code_index += 1;
                }
                _ => return false,
            }
        }
        false
    }

    /// Some path from the source reaches `node` with `codes[code_end - 1]`
    /// as its last step, reading no further back than `row_start`.
    fn backward(
        &self,
        mut node: u32,
        mut code_end: usize,
        codes: &[Token],
        row_start: usize,
        dict: CompactDictionaryView<'_>,
    ) -> bool {
        // A loop, not a search: at most one edge of the token enters `node`,
        // a greedy step's origin being `to` less the token's length, and a
        // token that steps into `node` is shorter than the prefix `node`
        // stands for, so it cannot also enter there. Terminal ranges all
        // enter the sink, which `node` never is.
        while node != SOURCE {
            if code_end <= row_start {
                return false;
            }
            let code = codes[code_end - 1];
            let step = self.edges.of(code).find(|edge| u32::from(edge.to) == node);
            let Some(step) = step else {
                // Only an occurrence beginning inside `code` is left, decided
                // off the dictionary where the planner never enumerated the
                // tokens entering `node`.
                let Some(prefix) = self.nodes[node as usize].entry else {
                    return false;
                };
                return prefix.is_tail_of(dict, code);
            };
            node = u32::from(step.from);
            code_end -= 1;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dictionary::{CompactDictionary, Dictionary, DictionaryView, pad_raw};
    use crate::search::index::build_token_frequency_index;
    use crate::search::prefilter::graph::build_alignment_graph;
    use crate::search::tokenize;

    /// A complete dictionary of the single bytes plus `extra`, greedy-encoding
    /// `rows` the way the encoder does.
    fn column(extra: &[&[u8]], rows: &[&[u8]]) -> (CompactDictionary, Vec<Token>, Vec<u32>) {
        let mut tokens: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        tokens.extend(extra.iter().map(|t| t.to_vec()));
        tokens.sort();
        tokens.dedup();
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for token in &tokens {
            bytes.extend_from_slice(token);
            offsets.push(bytes.len() as u32);
        }
        pad_raw(&mut bytes, &offsets);
        let dict = CompactDictionary::from_raw(bytes, offsets);
        let mut codes = Vec::new();
        let mut row_offsets = vec![0u32];
        for row in rows {
            codes.extend(tokenize(row, dict.as_view()));
            row_offsets.push(codes.len() as u32);
        }
        (dict, codes, row_offsets)
    }

    /// Every row's answer from the walk alone, trying every code of the row
    /// as the hit, against byte containment.
    fn assert_walk_decides(extra: &[&[u8]], rows: &[&[u8]], needle: &[u8]) {
        let (dict, codes, row_offsets) = column(extra, rows);
        let dict = dict.as_view();
        let frequencies = build_token_frequency_index(&codes, dict.num_tokens()).unwrap();
        let graph = build_alignment_graph(dict, needle, frequencies.as_view());
        let walk = Walk::from_graph(&graph, needle);
        for (row, text) in rows.iter().enumerate() {
            let (start, end) = (row_offsets[row] as usize, row_offsets[row + 1] as usize);
            let walked = (start..end).any(|hit| walk.check(dict, &codes, start, end, hit));
            let contains = text.windows(needle.len()).any(|window| window == needle);
            assert_eq!(
                walked,
                contains,
                "row {row} {:?}",
                String::from_utf8_lossy(text)
            );
        }
    }

    /// `goo|gl|e` is the alignment-0 parse; `agoo` enters at node 3 with the
    /// needle's head as its tail; `es` ends it inside a longer token; `gooxgle`
    /// and `gogle` hit the same tokens and must fail.
    #[test]
    fn interior_entry_and_exit_layouts() {
        assert_walk_decides(
            &[b"goo", b"gl", b"es", b"agoo", b"oo"],
            &[
                b"google",
                b"xagoogle",
                b"googles",
                b"gooxgle",
                b"gogle",
                b"agoo gl e",
                b"",
            ],
            b"No in ",
        );
    }

    /// The needle further inside one token, and a token the needle is a
    /// prefix of, are both a single step from source to sink.
    #[test]
    fn whole_needle_inside_one_token() {
        assert_walk_decides(
            &[b"xgooglex", b"googlez", b"goo", b"gl"],
            &[b"xgooglex", b"googlez", b"xgoogle", b"goog"],
            b"google",
        );
    }

    /// `ab` is a point step out of the source and the terminal range out of
    /// node 2, so each hit tries both roles; only one of them confirms.
    #[test]
    fn a_token_on_two_edges() {
        assert_walk_decides(
            &[b"ab"],
            &[b"abab", b"xabab", b"ababx", b"abxab", b"ab", b"aabb"],
            b"abab",
        );
    }

    /// An occurrence cannot cross a row boundary: `goo` ending one row and
    /// `gle` starting the next is no match, and the walk stops at the row.
    #[test]
    fn walk_stops_at_the_row() {
        assert_walk_decides(&[b"goo", b"gl"], &[b"goo", b"gle", b"google"], b"google");
    }

    /// A failed trial at one hit must not cost the occurrence starting inside
    /// it: `appappapple` holds `appapple` from its second `app` on, and the
    /// first `app` fails in the needle's first role before the third `app`
    /// confirms in its second. A streaming matcher needs a KMP back-edge
    /// here; the anchored walk tries every hit in every role instead. Three
    /// dictionaries parse the row three ways.
    #[test]
    fn overlapping_occurrence_needs_no_back_edge() {
        let rows: &[&[u8]] = &[
            b"appappapple",
            b"appapple",
            b"appappappapple",
            b"xappappapplex",
            b"appappaple",
            b"appapp",
            b"apple",
        ];
        assert_walk_decides(&[b"app", b"le"], rows, b"appapple");
        assert_walk_decides(&[b"app", b"apple", b"le"], rows, b"appapple");
        assert_walk_decides(&[b"appa", b"pp", b"pple", b"le"], rows, b"appapple");
        assert_walk_decides(
            &[b"aa", b"aab", b"aabaaabaa"],
            &[b"aabaaabaa", b"aabaab", b"aaa", b"baaab"],
            b"aaa",
        );
    }

    /// More than `PROBE_SET_SIZE_LIMIT` tokens end with `g`, so the planner never
    /// enumerates alignment 1's set, and the entry is answered off the
    /// dictionary as every entry is.
    #[test]
    fn entry_through_a_set() {
        let tails: Vec<Vec<u8>> = (b'a'..=b'z').map(|b| vec![b, b'g']).collect();
        let mut extra: Vec<&[u8]> = tails.iter().map(Vec::as_slice).collect();
        extra.extend_from_slice(&[b"oo", b"gl"]);
        assert_walk_decides(
            &extra,
            &[b"zgoogle", b"zgoogl", b"google", b"zg oogle"],
            b"google",
        );
    }
}
