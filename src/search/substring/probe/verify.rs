// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Hit verification: does a full occurrence of the needle pass through this code?
//!
//! A hit token can take several roles, i.e. be on several edges of the alignment
//! graph. For each role `from -> to`, the verifier walks the row in both directions:
//!
//! * forward from `to`, matching each next code against the node's greedy edge,
//!   until a code on the node's range edge completes the needle;
//! * backward from `from`, matching each previous code against the edge entering
//!   the node, until it reaches the source or a token whose suffix starts the needle.
//!
//! Greedy tokenization leaves each walk a single path, and walks stop at the row
//! boundaries. The hit is confirmed as soon as one role succeeds in both directions.
//! A token that contains the whole needle is accepted without walking.
//!
//! Highly repetitive needles are the slow case: a token such as a run of one byte
//! can take a role at almost every offset, and each hit may try all of them.

use crate::core::dictionary::{CompactDictionaryView, DictionaryView};
use crate::core::types::{Token, TokenRange};
use crate::search::substring::probe::alignment::graph::{AlignmentGraph, EdgeKind};

/// Needle offset before any bytes have matched.
const SOURCE: u16 = 0;

/// The edges leaving one needle offset.
#[derive(Debug, Clone, Default)]
struct Node {
    /// The greedy edge: its token and target offset.
    greedy_step: Option<(Token, u16)>,
    /// The range edge: the tokens that finish the needle from this offset.
    terminal_range: Option<TokenRange>,
    /// For an unenumerated alignment edge into this offset, the needle prefix that a
    /// preceding token must end with.
    entry: Option<NeedlePrefix>,
}

/// A needle prefix of at most 16 bytes, packed to compare with a token suffix in
/// one 128-bit operation.
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

    /// Whether `token`, a valid token ID, ends with this prefix.
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

/// The endpoints of a graph edge: one role of a token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Edge {
    from: u16,
    to: u16,
}

/// The roles of each token: the graph edges that list it.
///
/// A dense array holds, per token, its only role, `NONE`, or `LISTED`. The roles of
/// a `LISTED` token are in a separate list sorted by token. Real edges have
/// `from < to`, so the sentinels cannot be mistaken for them.
#[derive(Debug, Clone, Default)]
struct EdgesByToken {
    /// The token at index zero of the dense arrays.
    first_token: Token,
    /// One role or sentinel per token from `first_token` to the last token with a role.
    edge_of_token: Vec<Edge>,
    /// Whether each token contains the whole needle; empty if none does.
    complete: Vec<bool>,
    /// Roles of `LISTED` tokens, grouped by token, a complete match first.
    edges_of_tokens: Vec<(Token, Edge)>,
}

/// Sentinel for a token with no role.
const NONE: Edge = Edge { from: 0, to: 0 };

/// Sentinel for a token whose roles are in `edges_of_tokens`.
const LISTED: Edge = Edge {
    from: u16::MAX,
    to: u16::MAX,
};

impl EdgesByToken {
    /// Indexes `token_edges`, deduplicated. `sink` identifies complete matches.
    fn new(mut token_edges: Vec<(Token, Edge)>, sink: u16) -> Self {
        let complete = Edge {
            from: SOURCE,
            to: sink,
        };
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
            complete: Vec::new(),
            edges_of_tokens: Vec::new(),
        };
        for same_token in token_edges.chunk_by_mut(|a, b| a.0 == b.0) {
            let (token, edge) = same_token[0];
            edges.edge_of_token[usize::from(token - first_token)] = if same_token.len() == 1 {
                edge
            } else {
                // A complete occurrence needs no walking; try it first.
                if let Some(at) = same_token.iter().position(|&(_, edge)| edge == complete) {
                    same_token.swap(0, at);
                }
                edges.edges_of_tokens.extend_from_slice(same_token);
                LISTED
            };
            if same_token[0].1 == complete {
                if edges.complete.is_empty() {
                    edges.complete.resize(edges.edge_of_token.len(), false);
                }
                edges.complete[usize::from(token - first_token)] = true;
            }
        }
        edges
    }

    /// Whether `token` contains the whole needle, which needs no walk to confirm.
    #[inline]
    fn contains_whole_needle(&self, token: Token) -> bool {
        let at = usize::from(token.wrapping_sub(self.first_token));
        self.complete.get(at).copied().unwrap_or(false)
    }

    /// The roles of `token`, lazily, a complete match first.
    #[inline]
    fn of(&self, token: Token) -> impl Iterator<Item = Edge> + '_ {
        let at = usize::from(token.wrapping_sub(self.first_token));
        let (single, listed) = match self.edge_of_token.get(at).copied().unwrap_or(NONE) {
            NONE => (None, &self.edges_of_tokens[..0]),
            LISTED => {
                let first = self.edges_of_tokens.partition_point(|&(t, _)| t < token);
                (None, &self.edges_of_tokens[first..])
            }
            edge => (Some(edge), &self.edges_of_tokens[..0]),
        };
        single.into_iter().chain(
            listed
                .iter()
                .take_while(move |&&(t, _)| t == token)
                .map(|&(_, edge)| edge),
        )
    }
}

/// An alignment graph compiled for hit verification.
#[derive(Debug, Clone)]
pub(in crate::search::substring::probe) struct HitVerifier {
    /// Outgoing edges, by needle offset.
    nodes: Vec<Node>,
    /// The roles of each token: the edges to try for a hit, and the edges to follow
    /// when walking backward.
    edges: EdgesByToken,
    /// The needle prefix of each partial alignment edge, by length (at most 15).
    /// Backward walks use it to check an overlap without searching a role list.
    overlaps: Vec<Option<NeedlePrefix>>,
}

impl HitVerifier {
    /// Compiles `graph`, which must have been built for `needle`, nonempty.
    pub(in crate::search::substring::probe) fn from_graph(
        graph: &AlignmentGraph,
        needle: &[u8],
    ) -> Self {
        let mut nodes = vec![Node::default(); graph.node_count()];
        let mut overlaps = vec![None; needle.len().min(16)];
        let mut token_edges: Vec<(Token, Edge)> = Vec::new();
        for edge in &graph.edges {
            let (from, to) = (edge.from, edge.to);
            let compiled_edge = Edge { from, to };
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
                // An alignment edge: partial (an overlap), or complete for tokens
                // that contain the needle. Keep the prefix of a partial one for
                // backward walks.
                EdgeKind::Set(ids) => {
                    if from == SOURCE && to != graph.sink() {
                        overlaps[to as usize] = Some(NeedlePrefix::new(&needle[..to as usize]));
                    }
                    token_edges.extend(ids.iter().map(|&token| (token, compiled_edge)));
                }
                EdgeKind::UnenumeratedSet => {
                    // Always a partial alignment edge, so `to` is at most 15.
                    nodes[to as usize].entry = Some(NeedlePrefix::new(&needle[..to as usize]));
                    overlaps[to as usize] = nodes[to as usize].entry;
                }
            }
        }
        let edges = EdgesByToken::new(token_edges, graph.sink());
        Self {
            nodes,
            edges,
            overlaps,
        }
    }

    /// The offset after the whole needle.
    fn sink(&self) -> u16 {
        (self.nodes.len() - 1) as u16
    }

    /// Whether an occurrence of the needle within `codes[row_start..row_end]` covers
    /// the code at `hit_code_index`.
    ///
    /// The row must lie within `codes` and contain the hit, and be greedily
    /// tokenized with the dictionary the graph was built for.
    #[inline]
    pub(in crate::search::substring::probe) fn check(
        &self,
        dict: CompactDictionaryView<'_>,
        codes: &[Token],
        row_start: usize,
        row_end: usize,
        hit_code_index: usize,
    ) -> bool {
        let hit = codes[hit_code_index];
        if self.edges.contains_whole_needle(hit) {
            return true;
        }
        for Edge { from, to } in self.edges.of(hit) {
            if (to == self.sink() || self.forward(to, hit_code_index + 1, codes, row_end))
                && (from == SOURCE || self.backward(from, hit_code_index, codes, row_start, dict))
            {
                return true;
            }
        }
        false
    }

    /// Walks forward from `node`, starting at `code_index`, until a range-edge token
    /// finishes the needle. Fails on a mismatch or at `row_end`.
    fn forward(
        &self,
        mut node: u16,
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

    /// Walks backward from `node` over the codes before `code_end`, until the source
    /// or a token that ends with the needle prefix. Fails on a mismatch or at
    /// `row_start`.
    fn backward(
        &self,
        mut node: u16,
        mut code_end: usize,
        codes: &[Token],
        row_start: usize,
        dict: CompactDictionaryView<'_>,
    ) -> bool {
        // For a given token and internal offset, there is at most one incoming
        // edge: a greedy edge starts at the offset minus the token length, while an
        // alignment edge requires a token longer than the offset. Range edges
        // enter the sink, where backward walks never start.
        while node != SOURCE {
            if code_end <= row_start {
                return false;
            }
            let code = codes[code_end - 1];
            let at = usize::from(code.wrapping_sub(self.edges.first_token));
            let edge = self.edges.edge_of_token.get(at).copied().unwrap_or(NONE);
            if edge == LISTED {
                let length = dict.token_len(code) as u16;
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
            } else if edge != NONE && edge.to == node {
                node = edge.from;
            } else {
                // An unenumerated alignment edge may still match this token's suffix.
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
pub(in crate::search::substring::probe) mod tests {
    use super::*;
    use crate::core::dictionary::{CompactDictionary, Dictionary, DictionaryView, pad_raw};
    use crate::search::index::build_token_frequency_index;
    use crate::search::substring::probe::alignment::graph::AlignmentGraph;
    use crate::search::tokenize;

    /// A hit verifier that accepts every hit: each token counts as containing the whole
    /// needle, so scan tests can exercise matching and row lookup on their own.
    pub(in crate::search::substring::probe) fn accepting_verifier() -> HitVerifier {
        HitVerifier {
            nodes: vec![Node::default()],
            edges: EdgesByToken {
                first_token: 0,
                edge_of_token: vec![NONE; 1 << 16],
                complete: vec![true; 1 << 16],
                edges_of_tokens: Vec::new(),
            },
            overlaps: Vec::new(),
        }
    }

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
    fn assert_verifier_decides(extra: &[&[u8]], rows: &[&[u8]], needle: &[u8]) {
        let (dict, codes, row_offsets) = column(extra, rows);
        let dict = dict.as_view();
        let frequencies = build_token_frequency_index(&codes, dict.num_tokens()).unwrap();
        let graph = AlignmentGraph::new(dict, needle, frequencies.as_view()).unwrap();
        let verifier = HitVerifier::from_graph(&graph, needle);
        for (row, text) in rows.iter().enumerate() {
            let (start, end) = (row_offsets[row] as usize, row_offsets[row + 1] as usize);
            let verified = (start..end).any(|hit| verifier.check(dict, &codes, start, end, hit));
            let contains = text.windows(needle.len()).any(|window| window == needle);
            assert_eq!(
                verified,
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
        let verifier = HitVerifier::from_graph(&graph, needle);
        for (row, text) in rows.iter().enumerate() {
            let (start, end) = (row_offsets[row] as usize, row_offsets[row + 1] as usize);
            let verified = start < end && verifier.check(dict, &codes, start, end, end - 1);
            let contains = text.windows(needle.len()).any(|window| window == needle);
            assert_eq!(verified, contains, "row {row}: {text:?}");
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

    /// `aba` is both a greedy edge and an enumerated alignment edge.
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
        assert_verifier_decides(
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
        assert_verifier_decides(
            &[b"xgooglex", b"googlez", b"goo", b"gl"],
            &[b"xgooglex", b"googlez", b"xgoogle", b"goog"],
            b"google",
        );
    }

    /// `ab` is the greedy edge out of the source and on the range edge out of
    /// node 2, so each hit tries both roles; only one of them confirms.
    #[test]
    fn a_token_on_two_edges() {
        assert_verifier_decides(
            &[b"ab"],
            &[b"abab", b"xabab", b"ababx", b"abxab", b"ab", b"aabb"],
            b"abab",
        );
    }

    /// An occurrence cannot cross a row boundary: `goo` ending one row and
    /// `gle` starting the next is no match, and the walk stops at the row.
    #[test]
    fn walk_stops_at_the_row() {
        assert_verifier_decides(&[b"goo", b"gl"], &[b"goo", b"gle", b"google"], b"google");
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
        assert_verifier_decides(&[b"app", b"le"], rows, b"appapple");
        assert_verifier_decides(&[b"app", b"apple", b"le"], rows, b"appapple");
        assert_verifier_decides(&[b"appa", b"pp", b"pple", b"le"], rows, b"appapple");
        assert_verifier_decides(
            &[b"aa", b"aab", b"aabaaabaa"],
            &[b"aabaaabaa", b"aabaab", b"aaa", b"baaab"],
            b"aaa",
        );
    }

    /// More than 16 tokens end with `g`, so the one-byte overlap is unenumerated
    /// and the hit verifier checks the token suffix directly.
    #[test]
    fn entry_through_a_set() {
        let tails: Vec<Vec<u8>> = (b'a'..=b'z').map(|b| vec![b, b'g']).collect();
        let mut extra: Vec<&[u8]> = tails.iter().map(Vec::as_slice).collect();
        extra.extend_from_slice(&[b"oo", b"gl"]);
        assert_verifier_decides(
            &extra,
            &[b"zgoogle", b"zgoogl", b"google", b"zg oogle"],
            b"google",
        );
    }
}
