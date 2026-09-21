// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Confirm a substring occurrence around a candidate token hit.
//!
//! Preparation compiles the alignment graph into per-offset continuation
//! checks and a token-to-edges lookup. A hit token can belong to several edges;
//! verification tries each role, checking the remaining codes forward and
//! the preceding codes backward within the same row.
//!
//! Forward checks follow greedy steps until a terminal range finishes the
//! needle. Backward checks use the preceding token's single edge when possible;
//! tokens with several roles use their length to locate the only possible
//! predecessor, then check its expected token. Initial overlaps compare the
//! token suffix with a short needle prefix when no explicit edge is used.
//! The row must use the same greedy tokenization represented by the graph.
//!
//! Each trial follows a path without branching, but different roles and hits
//! can revisit codes. This is an anchored verifier, not a streaming KMP scan.

use crate::core::dictionary::{CompactDictionaryView, DictionaryView};
use crate::core::types::{Token, TokenRange};
use crate::search::substring::alignment::graph::{AlignmentGraph, EdgeKind};

/// Needle offset before any bytes have matched.
const SOURCE: u32 = 0;

/// Continuation checks at one needle offset in the compiled graph.
#[derive(Debug, Clone, Default)]
struct Node {
    /// Token and destination for the next interior step, if one exists.
    greedy_step: Option<(Token, u32)>,
    /// Tokens that finish the needle from this offset.
    terminal_range: Option<TokenRange>,
    /// Needle prefix for an unenumerated source edge ending here.
    /// A preceding token can start the match if its suffix equals this prefix.
    entry: Option<NeedlePrefix>,
}

/// A short needle prefix packed for a fixed-width token-suffix comparison.
#[derive(Debug, Clone, Copy)]
struct NeedlePrefix {
    len: usize,
    bytes: u128,
    mask: u128,
}

impl NeedlePrefix {
    /// Pack a nonempty prefix of at most 16 bytes into a masked integer.
    fn new(prefix: &[u8]) -> Self {
        let mut bytes = [0u8; 16];
        bytes[..prefix.len()].copy_from_slice(prefix);
        Self {
            len: prefix.len(),
            bytes: u128::from_le_bytes(bytes),
            mask: u128::MAX >> (8 * (16 - prefix.len())),
        }
    }

    /// Compare this prefix with the last bytes of a valid dictionary token.
    /// A short token fails before the padded 16-byte dictionary read.
    #[inline]
    fn is_tail_of(&self, dict: CompactDictionaryView<'_>, token: Token) -> bool {
        let token_len = dict.token_len(token);
        if token_len < self.len {
            return false;
        }
        // SAFETY: token_len checked the token ID. The dictionary guarantees
        // MAX_TOKEN_SIZE readable bytes at every valid token's pointer.
        let window = unsafe { dict.token_ptr(token).cast::<u128>().read_unaligned() };
        (u128::from_le(window) >> (8 * (token_len - self.len))) & self.mask == self.bytes
    }
}

/// Needle offsets immediately before and after a token step, stored as `u16`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Edge {
    from: u16,
    to: u16,
}

impl Edge {
    /// Graph edges advance needle offsets, bounded by the public pattern limit.
    fn new(from: u32, to: u32) -> Self {
        Self {
            from: from as u16,
            to: to as u16,
        }
    }
}

/// Enumerated graph edges indexed by token ID.
/// The dense array stores one edge, `NONE`, or `LISTED`. Tokens with several
/// roles keep their edges in a separate sorted list. Real edges advance the
/// needle offset, so equal-endpoint sentinels cannot collide with them.
#[derive(Debug, Clone, Default)]
struct EdgesByToken {
    /// Token ID represented by index zero of the dense array.
    first_token: Token,
    /// One edge or sentinel per ID between the first and last indexed token.
    edge_of_token: Vec<Edge>,
    /// Sorted `(token, edge)` entries for tokens with more than one role.
    edges_of_tokens: Vec<(Token, Edge)>,
}

/// Dense-array sentinel for a token with no enumerated graph edge.
const NONE: Edge = Edge { from: 0, to: 0 };

/// Dense-array sentinel for a token whose edges are in the separate list.
const LISTED: Edge = Edge {
    from: u16::MAX,
    to: u16::MAX,
};

impl EdgesByToken {
    /// Deduplicate token-edge pairs and separate single-edge from multi-edge tokens.
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

    /// Iterate every enumerated graph role for this token, or none if absent.
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

    /// Locate the contiguous edge list for a token with multiple roles.
    fn listed_for(&self, token: Token) -> &[(Token, Edge)] {
        let first = self.edges_of_tokens.partition_point(|&(t, _)| t < token);
        let count = self.edges_of_tokens[first..]
            .iter()
            .take_while(|&&(t, _)| t == token)
            .count();
        &self.edges_of_tokens[first..first + count]
    }
}

/// Immutable verification tables compiled from an alignment graph.
/// The empty default is used for empty patterns and is never executed.
#[derive(Debug, Clone, Default)]
pub(in crate::search::substring) struct Walk {
    /// Forward transitions and unenumerated starts, indexed by needle offset.
    nodes: Vec<Node>,
    /// Roles to try for a candidate token and predecessor edges for backward checks.
    edges: EdgesByToken,
    /// Partial source overlaps, indexed by matched prefix length (at most 15).
    /// Multi-role backward steps use these instead of searching their edge lists.
    overlaps: Vec<Option<NeedlePrefix>>,
}

impl Walk {
    /// Compile the graph for the same nonempty needle used to build it.
    /// Preparation bounds needle offsets to `u16`. Enumerated ranges and sets are
    /// expanded into token-edge pairs; unenumerated starts retain a prefix check.
    pub(in crate::search::substring) fn from_graph(graph: &AlignmentGraph, needle: &[u8]) -> Self {
        let mut nodes = vec![Node::default(); graph.node_count()];
        let mut overlaps = vec![None; needle.len().min(16)];
        let mut token_edges: Vec<(Token, Edge)> = Vec::new();
        for edge in &graph.edges {
            let (from, to) = (edge.from, edge.to);
            let compiled_edge = Edge::new(from, to);
            match edge.kind() {
                EdgeKind::Single(token) => {
                    let token = *token;
                    let node = &mut nodes[from as usize];
                    node.greedy_step = Some((token, to));
                    token_edges.push((token, compiled_edge));
                }
                EdgeKind::Range(range) => {
                    nodes[from as usize].terminal_range = Some(*range);
                    token_edges
                        .extend((range.begin..=range.last).map(|token| (token, compiled_edge)));
                }
                // Enumerated entries cover both partial and whole matches.
                // Retain partial prefixes for multi-role backward steps too.
                EdgeKind::Set(ids) => {
                    if from == SOURCE && to != graph.sink() {
                        overlaps[to as usize] = Some(NeedlePrefix::new(&needle[..to as usize]));
                    }
                    token_edges.extend(ids.iter().map(|&token| (token, compiled_edge)));
                }
                EdgeKind::UnenumeratedSet => {
                    // Unenumerated entry edges run from the source to an
                    // alignment inside the first token, at most 15 bytes in.
                    nodes[to as usize].entry = Some(NeedlePrefix::new(&needle[..to as usize]));
                    overlaps[to as usize] = nodes[to as usize].entry;
                }
            }
        }
        let edges = EdgesByToken::new(token_edges);
        Self {
            nodes,
            edges,
            overlaps,
        }
    }

    /// Needle offset reached when the complete pattern has matched.
    fn sink(&self) -> u32 {
        (self.nodes.len() - 1) as u32
    }

    /// Whether a complete occurrence passes through this hit in this row.
    /// Try each edge associated with the hit token. Callers supply valid row
    /// bounds, a hit inside them, and codes greedily encoded with the prepared
    /// dictionary. Both directions must succeed for the same edge.
    #[inline]
    pub(in crate::search::substring) fn check(
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
                && (from == SOURCE || self.backward(from, hit_code_index, codes, row_start, dict))
            {
                return true;
            }
        }
        false
    }

    /// Follow greedy transitions through following row codes until a terminal match.
    /// Failure or the row boundary ends the trial.
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

    /// Follow preceding row codes back to the source or a matching token suffix.
    /// The search never reads before `row_start`.
    fn backward(
        &self,
        mut node: u32,
        mut code_end: usize,
        codes: &[Token],
        row_start: usize,
        dict: CompactDictionaryView<'_>,
    ) -> bool {
        // For a given token and interior offset, there is at most one incoming
        // edge: a greedy step starts at offset minus token length, while a
        // source overlap requires a token longer than that offset. Terminal
        // edges enter the sink, which this backward traversal never starts at.
        while node != SOURCE {
            if code_end <= row_start {
                return false;
            }
            let code = codes[code_end - 1];
            let at = usize::from(code.wrapping_sub(self.edges.first_token));
            let edge = self.edges.edge_of_token.get(at).copied().unwrap_or(NONE);
            if edge == LISTED {
                let length = dict.token_len(code) as u32;
                if length > node {
                    // The match starts inside this token, so check its suffix.
                    return self
                        .overlaps
                        .get(node as usize)
                        .and_then(Option::as_ref)
                        .is_some_and(|prefix| prefix.is_tail_of(dict, code));
                }
                // Length fixes the predecessor; token identity confirms the step.
                let from = node - length;
                match self.nodes[from as usize].greedy_step {
                    Some((expected, _)) if expected == code => node = from,
                    _ => return false,
                }
            } else if edge != NONE && u32::from(edge.to) == node {
                node = u32::from(edge.from);
            } else {
                // An unenumerated source edge may still match this token's suffix.
                return self.nodes[node as usize]
                    .entry
                    .as_ref()
                    .is_some_and(|prefix| prefix.is_tail_of(dict, code));
            }
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
    use crate::search::substring::alignment::graph::AlignmentGraph;
    use crate::search::tokenize;

    #[test]
    fn needle_prefix_matches_token_suffixes() {
        for token_len in 1..=16 {
            for alignment in 0..16 {
                for last in [false, true] {
                    let token: Vec<u8> = (0..token_len).map(|i| (i * 67) as u8).collect();
                    let mut bytes = vec![0xff; alignment];
                    let mut offsets = vec![0u32];
                    if alignment != 0 {
                        offsets.push(alignment as u32);
                    }
                    let id = (offsets.len() - 1) as Token;
                    bytes.extend_from_slice(&token);
                    offsets.push(bytes.len() as u32);
                    if !last {
                        bytes.extend_from_slice(&[0xa5; 16]);
                        offsets.push(bytes.len() as u32);
                    }
                    // Minimum readable padding, with nonzero bytes to ensure
                    // neither neighboring tokens nor padding affect the match.
                    let last_start = offsets[offsets.len() - 2] as usize;
                    bytes.resize(last_start + 16, 0xa5);
                    let dict = CompactDictionaryView::validate_safety(&bytes, &offsets).unwrap();
                    for prefix_len in 1..=16 {
                        let mut prefix = if prefix_len <= token_len {
                            token[token_len - prefix_len..].to_vec()
                        } else {
                            vec![0; prefix_len]
                        };
                        assert_eq!(
                            NeedlePrefix::new(&prefix).is_tail_of(dict, id),
                            token.ends_with(&prefix),
                        );
                        for i in 0..prefix_len {
                            prefix[i] ^= 0x80;
                            assert_eq!(
                                NeedlePrefix::new(&prefix).is_tail_of(dict, id),
                                token.ends_with(&prefix),
                            );
                            prefix[i] ^= 0x80;
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn needle_prefix_rejects_invalid_token_id() {
        let bytes = [b'a'; 16];
        let offsets = [0, 1];
        let dict = CompactDictionaryView::validate_safety(&bytes, &offsets).unwrap();
        NeedlePrefix::new(b"a").is_tail_of(dict, 1);
    }

    /// Build a complete dictionary and greedily encode the fixture rows.
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

    /// Try every row code as a hit and compare the result with byte containment.
    fn assert_walk_decides(extra: &[&[u8]], rows: &[&[u8]], needle: &[u8]) {
        let (dict, codes, row_offsets) = column(extra, rows);
        let dict = dict.as_view();
        let frequencies = build_token_frequency_index(&codes, dict.num_tokens()).unwrap();
        let graph = AlignmentGraph::new(dict, needle, frequencies.as_view()).unwrap();
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

    /// These fixtures can match only at the last token. Anchoring there forces
    /// verification of every preceding token instead of succeeding at an early hit.
    fn assert_last_hit_decides(extra: &[&[u8]], rows: &[&[u8]], needle: &[u8]) {
        let (dict, codes, row_offsets) = column(extra, rows);
        let dict = dict.as_view();
        let frequencies = build_token_frequency_index(&codes, dict.num_tokens()).unwrap();
        let graph = AlignmentGraph::new(dict, needle, frequencies.as_view()).unwrap();
        let walk = Walk::from_graph(&graph, needle);
        for (row, text) in rows.iter().enumerate() {
            let (start, end) = (row_offsets[row] as usize, row_offsets[row + 1] as usize);
            let walked = start < end && walk.check(dict, &codes, start, end, end - 1);
            let contains = text.windows(needle.len()).any(|window| window == needle);
            assert_eq!(walked, contains, "row {row}: {text:?}");
        }
    }

    /// `ab` has multiple roles. Its length locates the predecessor, but it
    /// must still fail at the offset that expects the equally long token `xy`.
    #[test]
    fn backward_checks_token_identity() {
        assert_last_hit_decides(
            &[b"ab", b"xy"],
            &[b"abxyabz", b"abababz", b"xyxyabz", b"abz", b""],
            b"abxyabz",
        );
    }

    /// `aba` is both an interior step and an enumerated initial overlap.
    /// In `ababaqabaz`, its final `a` starts the occurrence at needle offset 1.
    #[test]
    fn backward_checks_multi_role_initial_overlap() {
        assert_last_hit_decides(
            &[b"aba"],
            &[b"ababaqabaz", b"abaqabaz", b"ababaqabz", b"abaz"],
            b"abaqabaz",
        );
    }

    /// The final `b` anchors a long walk through repeated `a` roles.
    #[test]
    fn backward_handles_long_repeated_prefix() {
        let mut needle = vec![b'a'; 1024];
        needle.push(b'b');
        let mut longer = vec![b'a'; 1040];
        longer.push(b'b');
        assert_last_hit_decides(&[], &[&needle, &longer, &needle[1..], b"b"], &needle);
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
            b"google",
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

    /// More than 16 tokens end with `g`, so the one-byte overlap is unenumerated
    /// and the walker checks the token suffix directly.
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
