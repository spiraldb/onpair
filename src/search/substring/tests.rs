// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scan contracts, alignment regressions, and planning invariants.
//! Matcher, resolver, walker and min-cut algorithms have their own unit tests.

use super::alignment::cover::ProbeCover;
use super::alignment::graph::{AlignmentGraph, Edge, EdgeKind};
use super::alignment::mincut::MinCut;
use super::alignment::starts::{MAX_ENUMERATED_TOKENS, tests::check_starts};
use super::plan::{cover_cost, select_cover};
use super::scan::{BLOCK, detect_isa};
use super::{ContainsDfa, ContainsError, ContainsScan};
use crate::core::dictionary::{CompactDictionaryView, DictionaryView};
use crate::core::types::{MAX_TOKEN_SIZE, Token, TokenRange};
use crate::core::validate::{InvalidColumn, InvalidFrequencyIndex};
use crate::search::index::{
    TokenFrequencyIndex, TokenFrequencyIndexStorage, build_token_frequency_index,
};
use crate::{Column, ColumnView, DEFAULT_CONFIG, compress};

struct BorrowedFrequencies<'a>(&'a [u32]);

impl TokenFrequencyIndexStorage for BorrowedFrequencies<'_> {
    fn cumulative(&self) -> &[u32] {
        self.0
    }
}

fn compress_rows(rows: &[&[u8]]) -> Column<u32> {
    let mut bytes = Vec::new();
    let mut offsets = vec![0u32];
    for row in rows {
        bytes.extend_from_slice(row);
        offsets.push(bytes.len() as u32);
    }
    compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap()
}

fn byte_contains(haystack: &[u8], needle: &[u8]) -> bool {
    needle.is_empty()
        || haystack
            .windows(needle.len())
            .any(|window| window == needle)
}

fn sink_reachable_avoiding(graph: &AlignmentGraph, blocked: impl Fn(&Edge) -> bool) -> bool {
    let mut adjacency = vec![Vec::new(); graph.node_count()];
    for edge in &graph.edges {
        adjacency[edge.from as usize].push(edge);
    }
    let mut seen = vec![false; graph.node_count()];
    let mut stack = vec![0];
    seen[0] = true;
    while let Some(node) = stack.pop() {
        if node == graph.sink() as usize {
            return true;
        }
        for edge in &adjacency[node] {
            let next = edge.to as usize;
            if !seen[next] && !blocked(edge) {
                seen[next] = true;
                stack.push(next);
            }
        }
    }
    false
}

/// Builder guarantees relied on by the infallible cut solver and walker.
fn check_graph(view: ColumnView<'_, u32>, frequencies: &TokenFrequencyIndex, needle: &[u8]) {
    let graph = AlignmentGraph::new(view.dict, needle, frequencies.as_view()).unwrap();
    assert_eq!(graph.node_count(), needle.len() + 1);
    assert!(graph.edges.len() <= 2 * needle.len() + MAX_TOKEN_SIZE);
    assert!(
        graph
            .edges
            .iter()
            .all(|edge| edge.from < edge.to && edge.to as usize <= needle.len())
    );
    assert!(
        !sink_reachable_avoiding(&graph, Edge::cuttable),
        "a path has no probe: {needle:?}"
    );
    let mut has_successor = vec![false; graph.node_count()];
    for edge in &graph.edges {
        if edge.from != 0 && edge.to != graph.sink() {
            assert!(matches!(edge.kind(), EdgeKind::Single(_)));
            assert!(!std::mem::replace(
                &mut has_successor[edge.from as usize],
                true
            ));
        }
        if !edge.cuttable() {
            assert!(edge.from == 0 && edge.to != graph.sink());
        }
    }
    for edge in graph.edges.iter().filter(|edge| edge.cuttable()) {
        let cover = ProbeCover::from_edge_cut(std::iter::once(edge));
        let matched = view
            .codes
            .iter()
            .filter(|&&code| cover.contains(code))
            .count();
        assert_eq!(edge.frequency() as usize, matched, "{edge:?}: {needle:?}");
    }
}

/// Original bytes are the oracle; repeat execution to check reuse and appending.
fn check(view: ColumnView<'_, u32>, rows: &[&[u8]], patterns: &[&[u8]]) {
    let frequencies = build_token_frequency_index(view.codes, view.dict.num_tokens()).unwrap();
    for &pattern in patterns {
        let expected: Vec<_> = rows
            .iter()
            .enumerate()
            .filter_map(|(row, bytes)| byte_contains(bytes, pattern).then_some(row))
            .collect();
        if !pattern.is_empty() {
            check_starts(view.dict, pattern);
            check_graph(view, &frequencies, pattern);
        }
        let scan = ContainsScan::new(pattern, view.dict, &frequencies).unwrap();
        let covered = view
            .codes
            .iter()
            .filter(|&&code| scan.probe_cover().contains(code))
            .count();
        assert_eq!(scan.covered_frequency() as usize, covered, "{pattern:?}");
        assert_eq!(scan.total_frequency() as usize, view.codes.len());
        let fraction = if view.codes.is_empty() {
            0.0
        } else {
            covered as f64 / view.codes.len() as f64
        };
        assert_eq!(scan.covered_fraction(), fraction, "{pattern:?}");
        let mut actual = vec![usize::MAX];
        let mut appended = actual.clone();
        for _ in 0..2 {
            scan.scan(view.codes, view.row_offsets, view.dict, &mut actual);
            appended.extend_from_slice(&expected);
            assert_eq!(actual, appended, "scan: {pattern:?}");
        }
        assert_eq!(
            view.rows_containing_with_scan(pattern, &frequencies)
                .unwrap(),
            expected,
            "column scan: {pattern:?}"
        );
        if pattern.len() <= ContainsDfa::MAX_PATTERN_LEN {
            assert_eq!(
                view.rows_containing(pattern).unwrap(),
                expected,
                "DFA: {pattern:?}"
            );
        }
    }
}

#[test]
fn pattern_limits_are_independent() {
    let long_row = vec![b'a'; 300];
    let rows: &[&[u8]] = &[&long_row, b"abc", b"aaaaaaaaaa"];
    let column = compress_rows(rows);
    let view = column.view();
    let frequencies = build_token_frequency_index(view.codes, view.dict.num_tokens()).unwrap();
    check(view, rows, &[&long_row[..255], &long_row[..256]]);
    assert_eq!(
        view.rows_containing(&long_row[..256]),
        Err(ContainsError::PatternTooLong {
            length: 256,
            max: ContainsDfa::MAX_PATTERN_LEN,
        })
    );
    let oversized = vec![b'a'; ContainsScan::MAX_PATTERN_LEN + 1];
    assert_eq!(
        view.rows_containing_with_scan(&oversized, &frequencies),
        Err(ContainsError::PatternTooLong {
            length: oversized.len(),
            max: ContainsScan::MAX_PATTERN_LEN,
        })
    );
}

#[test]
fn mismatched_frequency_domain_is_a_preparation_error() {
    let column = compress_rows(&[b"abc"]);
    let view = column.view();
    for size in [view.dict.num_tokens() - 1, view.dict.num_tokens() + 1] {
        let frequencies = build_token_frequency_index::<Token>(&[], size).unwrap();
        for pattern in [b"".as_slice(), b"a"] {
            assert_eq!(
                view.rows_containing_with_scan(pattern, &frequencies),
                Err(ContainsError::InvalidData(
                    InvalidFrequencyIndex::BadLength.into()
                ))
            );
        }
    }
}

#[test]
fn missing_alphabet_does_not_produce_a_partial_cover() {
    // Missing `b` is encountered after the graph has already emitted an edge.
    let mut bytes = vec![b'a'];
    bytes.resize(1 + MAX_TOKEN_SIZE, 0);
    let dict = CompactDictionaryView::validate_safety(&bytes, &[0u32, 1]).unwrap();
    let frequencies = build_token_frequency_index(&[0u16], 1).unwrap();
    assert_eq!(
        ContainsScan::new(b"ab", dict, &frequencies).unwrap_err(),
        ContainsError::InvalidData(InvalidColumn::IncompleteAlphabet)
    );
}

#[test]
fn empty_pattern_and_empty_cover_have_distinct_results() {
    let cases: &[&[&[u8]]] = &[&[b"", b"alpha", b"", b"beta", b""], &[b"", b""], &[]];
    for &rows in cases {
        let column = compress_rows(rows);
        let view = column.view();
        let frequencies = build_token_frequency_index(view.codes, view.dict.num_tokens()).unwrap();
        check(view, rows, &[b"", b"absent"]);
        let scan = ContainsScan::new(b"", view.dict, &frequencies).unwrap();
        assert!(scan.probe_cover().is_empty());
        assert!(!scan.should_fallback(&frequencies));
        assert_eq!(
            scan.expected_candidate_row_fraction(rows.len()),
            if rows.is_empty() { 0.0 } else { 1.0 }
        );
        // An empty cover without the empty-pattern flag must leave output alone.
        let no_matches = ContainsScan {
            matches_all: false,
            ..scan
        };
        assert_eq!(no_matches.expected_candidate_row_fraction(rows.len()), 0.0);
        let mut output = vec![usize::MAX];
        no_matches.scan(view.codes, view.row_offsets, view.dict, &mut output);
        assert_eq!(output, [usize::MAX]);
    }
}

#[test]
fn prepared_scan_handles_empty_and_partial_inputs() {
    let rows: &[&[u8]] = &[b"", b"alpha", b"alphabet", b"beta", b""];
    let column = compress_rows(rows);
    let view = column.view();
    let frequencies = build_token_frequency_index(view.codes, view.dict.num_tokens()).unwrap();
    for pattern in [b"".as_slice(), b"alpha", b"absent"] {
        let scan = ContainsScan::new(pattern, view.dict, &frequencies).unwrap();
        for start in 0..=rows.len() {
            for end in start..=rows.len() {
                let first = view.row_offsets[start];
                let last = view.row_offsets[end];
                let offsets: Vec<u32> = view.row_offsets[start..=end]
                    .iter()
                    .map(|&offset| offset - first)
                    .collect();
                let mut out = vec![usize::MAX];
                scan.scan(
                    &view.codes[first as usize..last as usize],
                    &offsets,
                    view.dict,
                    &mut out,
                );
                let mut expected = vec![usize::MAX];
                expected.extend(
                    rows[start..end]
                        .iter()
                        .enumerate()
                        .filter_map(|(row, bytes)| byte_contains(bytes, pattern).then_some(row)),
                );
                assert_eq!(out, expected, "{pattern:?}: rows {start}..{end}");
            }
        }
    }
}

#[test]
fn scan_matches_bytes_on_boundaries_and_overlaps() {
    let rows: &[&[u8]] = &[
        b"",
        b"hello world",
        b"world peace",
        b"xabcabcy",
        b"aaaaab",
        b"aabaaabaa",
        b"appappapple",
        b"appapple pie",
        b"an appappappapple",
        b"appappaple",
        b"apple",
        b"abababab",
        b"ababx abab",
    ];
    let column = compress_rows(rows);
    check(
        column.view(),
        rows,
        &[
            b"hello",
            b"world",
            b"o w",
            b"bcabca",
            b"aa",
            b"aab",
            b"aabaa",
            b"appapple",
            b"aaa",
            b"abab",
            b"ababab",
            b"apple",
            b"papp",
            b"absent",
        ],
    );
}

#[test]
fn scan_matches_bytes_on_repetitive_corpus() {
    let corpus = crate::test_corpus::user_strings(200);
    let rows: Vec<&[u8]> = corpus.iter().map(|row| row.as_bytes()).collect();
    let column = compress_rows(&rows);
    check(
        column.view(),
        &rows,
        &[
            b"example",
            b"https",
            b"://",
            b".com",
            b"/page",
            b"user",
            b"ftp",
            b"zzz",
            b"w",
            b"https://www.example.com/",
        ],
    );
}

#[test]
fn scan_matches_bytes_on_binary_corpus() {
    let corpus = crate::test_corpus::binary_strings(40, 24, 11);
    let rows: Vec<&[u8]> = corpus.iter().map(Vec::as_slice).collect();
    let column = compress_rows(&rows);
    check(
        column.view(),
        &rows,
        &[b"", b"\x00", b"\xff", b"\x00\x01", &[7], &[200, 201]],
    );
}

#[test]
fn boundary_match_with_a_maximal_first_token_is_covered() {
    let corpus: Vec<_> = (0..40)
        .map(|i| format!("{}{:03}", "abcdefghijklmnop".repeat(4), i).into_bytes())
        .collect();
    let rows: Vec<&[u8]> = corpus.iter().map(Vec::as_slice).collect();
    let column = compress_rows(&rows);
    let dict = column.view().dict;
    let needle = b"klmnopabcdefghijklmnop";
    // Internal entry alignments stop at 15 bytes; this needs the source path.
    assert!((0..dict.num_tokens()).any(|id| dict.token(id as Token) == &needle[..MAX_TOKEN_SIZE]));
    check(column.view(), &rows, &[needle]);
}

#[test]
fn probe_scan_appends_each_row_once_across_blocks_and_tail() {
    let mut codes = vec![1u16; BLOCK + 3];
    codes.resize(2 * BLOCK + 6, 0);
    codes.push(1); // A hit in the padded tail, after a nonmatching row.
    let offsets = [
        0,
        0,
        BLOCK + 3,
        BLOCK + 3,
        2 * BLOCK + 6,
        codes.len(),
        codes.len(),
    ]
    .map(|offset| offset as u32);
    let cover = ProbeCover::from_runs(vec![TokenRange { begin: 1, last: 2 }]);
    let mut rows = vec![usize::MAX];
    super::scan::scan(&codes, &offsets, &cover, BLOCK + 4, &mut rows);
    assert_eq!(rows, [usize::MAX, 1, 4]);
}

#[test]
fn cover_merges_overlapping_and_adjacent_runs() {
    let run = |begin, last| TokenRange { begin, last };
    let cover = ProbeCover::from_runs(vec![
        run(7, 8),
        run(0, 0),
        run(3, 3),
        run(6, 7),
        run(1, 1),
        run(8, 8),
    ]);
    assert_eq!(cover.points(), &[3]);
    assert_eq!(cover.ranges(), &[run(0, 1), run(6, 8)]);
    for code in 0..=9 {
        assert_eq!(cover.contains(code), [0, 1, 3, 6, 7, 8].contains(&code));
    }
    assert!(ProbeCover::from_runs(Vec::new()).is_empty());
}

#[test]
fn borrowed_frequencies_preserve_the_plan_and_results() {
    let column = compress_rows(&[b"alpha beta", b"beta gamma", b"alphabet soup", b"delta"]);
    let view = column.view();
    let owned = build_token_frequency_index(view.codes, view.dict.num_tokens()).unwrap();
    let borrowed = TokenFrequencyIndex::validate(
        BorrowedFrequencies(owned.storage().cumulative()),
        view.codes,
        view.dict.num_tokens(),
    )
    .unwrap();
    let owned_scan = ContainsScan::new(b"alpha", view.dict, &owned).unwrap();
    let borrowed_scan = ContainsScan::new(b"alpha", view.dict, &borrowed).unwrap();
    assert_eq!(
        borrowed_scan.probe_cover().points(),
        owned_scan.probe_cover().points()
    );
    assert_eq!(
        borrowed_scan.probe_cover().ranges(),
        owned_scan.probe_cover().ranges()
    );
    assert_eq!(
        borrowed_scan.covered_frequency(),
        owned_scan.covered_frequency()
    );
    assert_eq!(
        borrowed_scan.covered_fraction(),
        owned_scan.covered_fraction()
    );
    assert_eq!(
        view.rows_containing_with_scan(b"alpha", &borrowed).unwrap(),
        [0, 2]
    );
}

#[test]
fn false_zero_frequencies_cannot_hide_a_true_match() {
    let column = compress_rows(&[b"alpha beta", b"gamma", b"alphabet soup", b"delta"]);
    let view = column.view();
    assert!(view.codes.iter().all(|&code| code != 0));
    // Attribute every occurrence to unused token 0. All real tokens have weight 0.
    let mut cumulative = vec![view.codes.len() as u32; view.dict.num_tokens() + 1];
    cumulative[0] = 0;
    let frequencies = TokenFrequencyIndex::validate_safety(
        BorrowedFrequencies(&cumulative),
        view.dict.num_tokens(),
        view.codes.len(),
    )
    .unwrap();
    let scan = ContainsScan::new(b"alpha", view.dict, &frequencies).unwrap();
    assert!(!scan.probe_cover().is_empty());
    assert_eq!(scan.covered_frequency(), 0);
    let mut rows = Vec::new();
    scan.scan(view.codes, view.row_offsets, view.dict, &mut rows);
    assert_eq!(rows, [0, 2]);
}

#[test]
fn sweep_cost_does_not_exceed_the_frequency_cut() {
    let corpus = crate::test_corpus::user_strings(200);
    let rows: Vec<&[u8]> = corpus.iter().map(|row| row.as_bytes()).collect();
    let column = compress_rows(&rows);
    let view = column.view();
    let frequencies = build_token_frequency_index(view.codes, view.dict.num_tokens()).unwrap();
    let freq = frequencies.as_view();
    let code_count = freq.total_frequency();
    let isa = detect_isa();
    for pattern in [
        b"e".as_slice(),
        b"://",
        b".com/page",
        b"https://www.example.com",
    ] {
        let graph = AlignmentGraph::new(view.dict, pattern, freq).unwrap();
        let mut solver = MinCut::new(&graph);
        let baseline = ProbeCover::from_edge_cut(
            solver
                .solve(|edge| u64::from(edge.frequency()))
                .iter()
                .map(|&at| &graph.edges[at as usize]),
        );
        let baseline_cost = cover_cost(isa, &baseline, baseline.frequency(freq), code_count);
        let (cover, covered) = select_cover(&graph, freq, isa);
        assert_eq!(covered, cover.frequency(freq));
        let cost = cover_cost(isa, &cover, covered, code_count);
        assert!(
            cost <= baseline_cost,
            "{pattern:?}: sweep {cost} against {baseline_cost}"
        );
    }
}

#[test]
fn planner_weight_bound_fits_signed_differences() {
    // Calculate in u128 so a future limit increase fails instead of overflowing.
    let n = ContainsScan::MAX_PATTERN_LEN as u128;
    let frequency = u128::from(u32::MAX);
    let dictionary_size = u128::from(Token::MAX) + 1;
    let edges = 2 * n + MAX_TOKEN_SIZE as u128;
    let comparisons =
        3 * n + (MAX_TOKEN_SIZE - 1) as u128 * MAX_ENUMERATED_TOKENS as u128 + dictionary_size;
    let finite_capacity = edges * frequency + comparisons * (frequency + 1);
    assert!(finite_capacity < 1u128 << 51);
    assert!(finite_capacity < i64::MAX as u128);
}
