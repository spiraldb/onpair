// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Hit verification: does a full occurrence of the pattern pass through this code?
//!
//! A hit token can be on several edges of the alignment graph, but only the edges
//! of the planner's cut need trying: every occurrence crosses the cut, and its code
//! there is a hit on a cut edge. For each cut edge `from -> to` the hit token is on,
//! the verifier walks the row in both directions:
//!
//! * forward from `to`, matching each next code against the node's greedy edge,
//!   until a code on the node's range edge completes the pattern;
//! * backward from `from`, matching each previous code against the edge entering
//!   the node, until it reaches the source or a token whose suffix starts the pattern.
//!
//! Greedy tokenization leaves each walk a single path, and walks stop at the row
//! boundaries. The hit is confirmed as soon as both walks succeed for one edge.
//! A token that contains the whole pattern is accepted without walking.
//!
//! In a highly repetitive pattern, a token such as a run of one byte can be on an
//! edge at almost every offset, but a hit is tried only on its cut edges, usually
//! few.

use crate::core::dictionary::{CompactDictionaryView, DictionaryView};
use crate::core::types::Token;
use crate::search::substring::probe::alignment::graph::{
    AlignmentGraph, Edge as GraphEdge, EdgeKind, Node, Overlap,
};

/// Pattern offset before any bytes have matched.
const SOURCE: u16 = 0;

/// A pattern prefix of at most 16 bytes, packed to compare with a token suffix in
/// one 128-bit operation.
#[derive(Debug, Clone, Copy)]
struct PatternPrefix {
    len: usize,
    bytes: u128,
    mask: u128,
}

impl PatternPrefix {
    /// Packs a nonempty prefix of at most 16 bytes into a masked integer.
    fn new(prefix: &[u8]) -> Self {
        let mut bytes = [0u8; 16];
        bytes[..prefix.len()].copy_from_slice(prefix);
        Self {
            len: prefix.len(),
            bytes: u128::from_le_bytes(bytes),
            mask: u128::MAX >> (8 * (16 - prefix.len())),
        }
    }

    /// Whether `token` ends with this prefix. A code outside the dictionary does
    /// not: walks read row codes, which may be malformed.
    #[inline]
    fn is_tail_of(&self, dict: CompactDictionaryView<'_>, token: Token) -> bool {
        if usize::from(token) >= dict.num_tokens() {
            return false;
        }
        let token_len = dict.token_len(token);
        if token_len < self.len {
            return false;
        }
        // SAFETY: `token` is a valid token ID, checked above. The dictionary
        // guarantees MAX_TOKEN_SIZE readable bytes at every valid token's pointer.
        let window = unsafe { dict.token_ptr(token).cast::<u128>().read_unaligned() };
        (u128::from_le(window) >> (8 * (token_len - self.len))) & self.mask == self.bytes
    }
}

/// The endpoints of an alignment-graph edge that lists a token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Edge {
    from: u16,
    to: u16,
}

/// The edges each token is on, looked up by token.
///
/// A dense array holds, per token, its only edge, `NONE`, or `LISTED`. The cut
/// edges of a `LISTED` token are in a separate list sorted by token: a hit is tried
/// only on those, while backward walks need only the dense array. Real edges have
/// `from < to`, so the sentinels cannot be mistaken for them.
#[derive(Debug, Default)]
struct TokenEdges {
    /// The token at index zero of the dense arrays.
    first_token: Token,
    /// One edge or sentinel per token from `first_token` to the last token on an edge.
    single: Vec<Edge>,
    /// Whether each token contains the whole pattern; empty if none does. Only
    /// `LISTED` tokens consult it: a single edge from the source to the sink already
    /// passes both walks.
    complete: Vec<bool>,
    /// The cut edges of `LISTED` tokens, sorted by token.
    listed: Vec<(Token, Edge)>,
}

/// Sentinel for a token on no edge.
const NONE: Edge = Edge { from: 0, to: 0 };

/// Sentinel for a token whose edges are in `listed`.
const LISTED: Edge = Edge {
    from: u16::MAX,
    to: u16::MAX,
};

impl TokenEdges {
    /// Indexes the tokens of `edges`, and lists for `LISTED` tokens the edges that
    /// `cut` indexes. `sink` identifies complete matches. Tokens go straight into the
    /// dense arrays, so only the cut edges of `LISTED` tokens are sorted.
    fn new(edges: &[GraphEdge<'_>], cut: &[u32], sink: u16) -> Self {
        let complete = Edge {
            from: SOURCE,
            to: sink,
        };
        let Some((first_token, last_token)) = edges
            .iter()
            .filter_map(token_bounds)
            .reduce(|(lo, hi), (first, last)| (lo.min(first), hi.max(last)))
        else {
            return Self::default();
        };
        let span = usize::from(last_token - first_token) + 1;
        let mut index = Self {
            first_token,
            single: vec![NONE; span],
            complete: Vec::new(),
            listed: Vec::new(),
        };
        for graph_edge in edges {
            let edge = endpoints(graph_edge);
            for_each_token(graph_edge, |token| {
                let at = usize::from(token - first_token);
                if edge == complete {
                    if index.complete.is_empty() {
                        index.complete.resize(span, false);
                    }
                    index.complete[at] = true;
                }
                let slot = &mut index.single[at];
                *slot = match *slot {
                    NONE => edge,
                    held if held == edge => held,
                    _ => LISTED,
                };
            });
        }
        for &at in cut {
            let graph_edge = &edges[at as usize];
            let edge = endpoints(graph_edge);
            for_each_token(graph_edge, |token| {
                if index.single[usize::from(token - first_token)] == LISTED {
                    index.listed.push((token, edge));
                }
            });
        }
        index.listed.sort_unstable();
        index.listed.dedup();
        index
    }

    /// Whether `token` contains the whole pattern, which needs no walk to confirm.
    #[inline]
    fn contains_whole_pattern(&self, token: Token) -> bool {
        let at = usize::from(token.wrapping_sub(self.first_token));
        self.complete.get(at).copied().unwrap_or(false)
    }

    /// The only edge of `token`, `NONE`, or `LISTED`.
    #[inline]
    fn single(&self, token: Token) -> Edge {
        let at = usize::from(token.wrapping_sub(self.first_token));
        self.single.get(at).copied().unwrap_or(NONE)
    }

    /// The cut edges of a `LISTED` token, lazily.
    fn listed(&self, token: Token) -> impl Iterator<Item = Edge> + '_ {
        let first = self.listed.partition_point(|&(t, _)| t < token);
        self.listed[first..]
            .iter()
            .take_while(move |&&(t, _)| t == token)
            .map(|&(_, edge)| edge)
    }
}

/// The endpoints of `edge`.
fn endpoints(edge: &GraphEdge<'_>) -> Edge {
    Edge {
        from: edge.from,
        to: edge.to,
    }
}

/// The lowest and the highest token `edge` lists, or `None` if it lists none.
fn token_bounds(edge: &GraphEdge<'_>) -> Option<(Token, Token)> {
    match edge.kind {
        EdgeKind::Single(token) => Some((token, token)),
        EdgeKind::Range(range) => Some((range.begin, range.last)),
        EdgeKind::Set(tokens) => Some((*tokens.iter().min()?, *tokens.iter().max()?)),
        EdgeKind::UnenumeratedSet => None,
    }
}

/// Calls `f` on each token `edge` lists. An unenumerated set lists none: its tokens
/// are recognized by their suffix, through `overlaps`.
#[inline]
fn for_each_token(edge: &GraphEdge<'_>, mut f: impl FnMut(Token)) {
    match edge.kind {
        EdgeKind::Single(token) => f(token),
        EdgeKind::Range(range) => (range.begin..=range.last).for_each(f),
        EdgeKind::Set(tokens) => tokens.iter().for_each(|&token| f(token)),
        EdgeKind::UnenumeratedSet => {}
    }
}

/// An alignment edge `0 -> k` with `k` below 16, as backward walks check it.
#[derive(Debug, Clone, Copy)]
struct OverlapEdge {
    /// `pattern[..k]`, which its tokens end with.
    prefix: PatternPrefix,
    /// Whether its tokens are listed, and so on the edge `0 -> k` in `edges`.
    listed: bool,
}

/// An alignment graph compiled for hit verification.
#[derive(Debug)]
pub(super) struct HitVerifier {
    /// The greedy and range edge of each pattern offset before the sink: the graph's
    /// own nodes.
    nodes: Vec<Node>,
    /// The edges each token is on: the cut ones to try for a hit, and all of them to
    /// follow when walking backward.
    edges: TokenEdges,
    /// The alignment edge into each overlap length, by length (at most 15). Backward
    /// walks check a token's suffix against it instead of searching an edge list.
    overlaps: Vec<Option<OverlapEdge>>,
}

impl HitVerifier {
    /// Compiles `graph`, the alignment graph of `pattern`, nonempty, whose edges are
    /// `edges`. A hit is tried only on the edges `cut` indexes, which every
    /// occurrence crosses, as the planner's cut does.
    pub(super) fn new(
        graph: &AlignmentGraph,
        edges: &[GraphEdge<'_>],
        cut: &[u32],
        pattern: &[u8],
    ) -> Self {
        let overlaps = graph
            .overlaps
            .iter()
            .enumerate()
            .map(|(length, overlap)| {
                overlap.as_ref().map(|overlap| OverlapEdge {
                    prefix: PatternPrefix::new(&pattern[..length]),
                    listed: matches!(overlap, Overlap::Set(_)),
                })
            })
            .collect();
        Self {
            nodes: graph.nodes.clone(),
            edges: TokenEdges::new(edges, cut, graph.sink()),
            overlaps,
        }
    }

    /// The offset after the whole pattern.
    fn sink(&self) -> u16 {
        // At most `ContainsScan::MAX_PATTERN_LEN`, which fits `u16`.
        self.nodes.len() as u16
    }

    /// Whether an occurrence of the pattern within `codes[row_start..row_end]` covers
    /// the code at `hit_code_index`.
    ///
    /// The row must lie within `codes` and contain the hit, and be greedily
    /// tokenized with the dictionary the graph was built for.
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
        let passes = |Edge { from, to }| {
            (to == self.sink() || self.forward(to, hit_code_index + 1, codes, row_end))
                && (from == SOURCE || self.backward(from, hit_code_index, codes, row_start, dict))
        };
        match self.edges.single(hit) {
            NONE => false,
            LISTED => self.edges.contains_whole_pattern(hit) || self.edges.listed(hit).any(passes),
            edge => passes(edge),
        }
    }

    /// Walks forward from `node`, starting at `code_index`, until a range-edge token
    /// finishes the pattern. Fails on a mismatch or at `row_end`.
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
            if at.range.is_some_and(|range| range.contains(code)) {
                return true;
            }
            match at.greedy {
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
    /// or a token that ends with the pattern prefix. Fails on a mismatch or at
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
            let edge = self.edges.single(code);
            if edge == LISTED {
                let length = dict.token_len(code) as u16;
                if length > node {
                    // The match starts inside this token, so check its suffix.
                    return self
                        .overlap(node)
                        .is_some_and(|overlap| overlap.prefix.is_tail_of(dict, code));
                }
                // Length fixes the predecessor; token identity confirms the step.
                let from = node - length;
                match self.nodes[from as usize].greedy {
                    Some((expected, _)) if expected == code => node = from,
                    _ => return false,
                }
            } else if edge != NONE && edge.to == node {
                node = edge.from;
            } else {
                // No listed edge of this token enters `node`, but an unenumerated
                // alignment edge may: its tokens are not listed.
                return self.overlap(node).is_some_and(|overlap| {
                    !overlap.listed && overlap.prefix.is_tail_of(dict, code)
                });
            }
            code_end -= 1;
        }
        true
    }

    /// The alignment edge into `node`, if `node` is an overlap length that has one.
    #[inline]
    fn overlap(&self, node: u16) -> Option<&OverlapEdge> {
        self.overlaps.get(node as usize)?.as_ref()
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::core::dictionary::{CompactDictionary, Dictionary, pad_raw};
    use crate::search::index::TokenFrequencyIndex;
    use crate::search::substring::probe::alignment::graph::AlignmentGraph;
    use crate::search::substring::probe::plan;
    use crate::search::substring::probe::scan::Isa;
    use crate::search::tokenize;

    /// A hit verifier that accepts every hit: each token's only edge spans the whole
    /// pattern, so scan tests can exercise matching and row lookup on their own.
    pub(crate) fn accepting_verifier() -> HitVerifier {
        let whole = Edge {
            from: SOURCE,
            to: 1,
        };
        HitVerifier {
            nodes: vec![Node::default(); 1],
            edges: TokenEdges {
                first_token: 0,
                single: vec![whole; 1 << 16],
                complete: Vec::new(),
                listed: Vec::new(),
            },
            overlaps: Vec::new(),
        }
    }

    #[test]
    fn pattern_prefix_matches_token_suffixes() {
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
                            PatternPrefix::new(&prefix).is_tail_of(dict, id),
                            token.ends_with(&prefix),
                        );
                        for i in 0..prefix_len {
                            prefix[i] ^= 0x80;
                            assert_eq!(
                                PatternPrefix::new(&prefix).is_tail_of(dict, id),
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
    fn pattern_prefix_rejects_invalid_token_id() {
        let bytes = [b'a'; 16];
        let offsets = [0, 1];
        let dict = CompactDictionaryView::validate_safety(&bytes, &offsets).unwrap();
        assert!(!PatternPrefix::new(b"a").is_tail_of(dict, 1));
    }

    /// Builds a complete dictionary and greedily encodes the fixture rows.
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

    /// Every edge of `edges`, as a cut: each hit is tried on all its edges.
    fn every_edge(edges: &[GraphEdge<'_>]) -> Vec<u32> {
        (0..edges.len() as u32).collect()
    }

    /// Tries every row code as a hit and compares the result with byte containment,
    /// with hits tried on every edge, then on each instruction set's planned cut.
    fn assert_verifier_decides(extra: &[&[u8]], rows: &[&[u8]], pattern: &[u8]) {
        let (dict, codes, row_offsets) = column(extra, rows);
        let dict = dict.as_view();
        let index = TokenFrequencyIndex::build(&codes, dict.num_tokens()).unwrap();
        let graph = AlignmentGraph::new(dict, pattern).unwrap();
        let edges: Vec<_> = graph.edges().collect();
        let planned = [Isa::Scalar, Isa::Neon, Isa::Avx2, Isa::Avx512Bw]
            .map(|isa| plan::select(&edges, graph.sink(), index.as_view(), isa).cut);
        for cut in [every_edge(&edges)].into_iter().chain(planned) {
            let verifier = HitVerifier::new(&graph, &edges, &cut, pattern);
            for (row, text) in rows.iter().enumerate() {
                let (start, end) = (row_offsets[row] as usize, row_offsets[row + 1] as usize);
                let verified =
                    (start..end).any(|hit| verifier.check(dict, &codes, start, end, hit));
                let contains = text.windows(pattern.len()).any(|window| window == pattern);
                assert_eq!(
                    verified,
                    contains,
                    "row {row} {:?}, cut {cut:?}",
                    String::from_utf8_lossy(text)
                );
            }
        }
    }

    /// These fixtures can match only at the last token. Anchoring there forces
    /// verification of every preceding token instead of succeeding at an early hit.
    fn assert_last_hit_decides(extra: &[&[u8]], rows: &[&[u8]], pattern: &[u8]) {
        let (dict, codes, row_offsets) = column(extra, rows);
        let dict = dict.as_view();
        let graph = AlignmentGraph::new(dict, pattern).unwrap();
        let edges: Vec<_> = graph.edges().collect();
        let verifier = HitVerifier::new(&graph, &edges, &every_edge(&edges), pattern);
        for (row, text) in rows.iter().enumerate() {
            let (start, end) = (row_offsets[row] as usize, row_offsets[row + 1] as usize);
            let verified = start < end && verifier.check(dict, &codes, start, end, end - 1);
            let contains = text.windows(pattern.len()).any(|window| window == pattern);
            assert_eq!(verified, contains, "row {row}: {text:?}");
        }
    }

    /// `ab` is on several edges. Its length locates the predecessor, but it
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
    /// In `ababaqabaz`, its final `a` starts the occurrence at pattern offset 1.
    #[test]
    fn backward_checks_multi_edge_initial_overlap() {
        assert_last_hit_decides(
            &[b"aba"],
            &[b"ababaqabaz", b"abaqabaz", b"ababaqabz", b"abaz"],
            b"abaqabaz",
        );
    }

    /// The final `b` anchors a long walk back through `a` tokens, which are on an
    /// edge at almost every offset.
    #[test]
    fn backward_handles_long_repeated_prefix() {
        let mut pattern = vec![b'a'; 1024];
        pattern.push(b'b');
        let mut longer = vec![b'a'; 1040];
        longer.push(b'b');
        assert_last_hit_decides(&[], &[&pattern, &longer, &pattern[1..], b"b"], &pattern);
    }

    /// `goo|gl|e` is the alignment-0 parse; `agoo` enters at node 3 with the
    /// pattern's head as its tail; `es` ends it inside a longer token; `gooxgle`
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

    /// The pattern further inside one token, and a token the pattern is a
    /// prefix of, are both a single step from source to sink.
    #[test]
    fn whole_pattern_inside_one_token() {
        assert_verifier_decides(
            &[b"xgooglex", b"googlez", b"goo", b"gl"],
            &[b"xgooglex", b"googlez", b"xgoogle", b"goog"],
            b"google",
        );
    }

    /// `ab` is the greedy edge out of the source and on the range edge out of
    /// node 2, so each hit tries both edges; only one of them confirms.
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
    /// first `app` fails on the edge of the pattern's first `app` before the third
    /// `app` confirms on the edge of its second. A streaming matcher needs a KMP
    /// back-edge here; the anchored walk tries every hit on every edge instead. Three
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

    /// More than 512 tokens end with `go`, so this overlap of two bytes is unenumerated
    /// as well, and the walk reaches the source through the token's suffix.
    #[test]
    fn entry_through_a_two_byte_set() {
        let tails: Vec<Vec<u8>> = (b'a'..=b'z')
            .flat_map(|first| (b'a'..=b'z').map(move |second| vec![first, second, b'g', b'o']))
            .collect();
        let mut extra: Vec<&[u8]> = tails.iter().map(Vec::as_slice).collect();
        extra.extend_from_slice(&[b"og", b"le"]);
        let (dict, _, _) = column(&extra, &[]);
        let graph = AlignmentGraph::new(dict.as_view(), b"google").unwrap();
        assert!(matches!(graph.overlaps[2], Some(Overlap::UnenumeratedSet)));
        assert_verifier_decides(
            &extra,
            &[b"zzgoogle", b"zzgoogl", b"google", b"zzgo ogle"],
            b"google",
        );
    }
}
