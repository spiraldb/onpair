// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end tests of substring search against byte-level containment, plus graph
//! and planning invariants. Matchers, the resolver, the hit verifier and the cut
//! solver have their own unit tests.

use std::panic::{AssertUnwindSafe, catch_unwind};

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use super::alignment::graph::{AlignmentGraph, Edge};
use super::alignment::mincut::MinCut;
use super::alignment::starts::tests::check_starts;
use super::plan::{self, CostModel};
use super::scan::{MatcherKind, detect_isa};
use super::{ContainsScan, ProbeCover, Query};
use crate::core::dictionary::{CompactDictionaryView, DictionaryView};
use crate::core::types::{MAX_TOKEN_SIZE, Token, TokenRange};
use crate::search::index::{TokenFrequencyIndex, TokenFrequencyIndexView};
use crate::search::substring::{ContainsDfa, ContainsError, row_contains};
use crate::{Column, ColumnView, DEFAULT_CONFIG, compress};

fn compress_rows(rows: &[&[u8]]) -> Column<u32> {
    let mut bytes = Vec::new();
    let mut offsets = vec![0u32];
    for row in rows {
        bytes.extend_from_slice(row);
        offsets.push(bytes.len() as u32);
    }
    compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap()
}

fn byte_contains(haystack: &[u8], pattern: &[u8]) -> bool {
    pattern.is_empty()
        || haystack
            .windows(pattern.len())
            .any(|window| window == pattern)
}

fn sink_reachable_avoiding<'g>(
    graph: &'g AlignmentGraph,
    blocked: impl Fn(&Edge<'g>) -> bool,
) -> bool {
    let node_count = usize::from(graph.sink()) + 1;
    let mut adjacency = vec![Vec::new(); node_count];
    for edge in graph.edges() {
        adjacency[edge.from as usize].push(edge);
    }
    let mut seen = vec![false; node_count];
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

/// Builder guarantees relied on by the infallible cut solver and hit verifier, and
/// the edge frequencies the planner weighs cuts with.
fn check_graph(
    view: ColumnView<'_, u32>,
    frequencies: TokenFrequencyIndexView<'_>,
    pattern: &[u8],
) {
    let graph = AlignmentGraph::new(view.dict, pattern).unwrap();
    assert_eq!(usize::from(graph.sink()), pattern.len());
    assert!(graph.edges().all(|edge| edge.from < edge.to));
    assert!(
        !sink_reachable_avoiding(&graph, Edge::cuttable),
        "a path has no probe: {pattern:?}"
    );
    for edge in graph.edges().filter(Edge::cuttable) {
        let cover = ProbeCover::from_edge_cut(std::iter::once(edge));
        let matched = view
            .codes
            .iter()
            .filter(|&&code| cover.contains(code))
            .count();
        assert_eq!(
            plan::edge_frequency(&edge, frequencies) as usize,
            matched,
            "{edge:?}: {pattern:?}"
        );
    }
}

/// Original bytes are the oracle; repeat execution to check reuse and appending.
fn check(view: ColumnView<'_, u32>, rows: &[&[u8]], patterns: &[&[u8]]) {
    let index = TokenFrequencyIndex::build(view.codes, view.dict.num_tokens()).unwrap();
    let frequencies = index.as_view();
    for &pattern in patterns {
        let expected: Vec<_> = rows
            .iter()
            .enumerate()
            .filter_map(|(row, bytes)| byte_contains(bytes, pattern).then_some(row))
            .collect();
        if !pattern.is_empty() {
            check_starts(view.dict, pattern);
            check_graph(view, frequencies, pattern);
        }
        check_scan(view, frequencies, pattern, &expected);
        assert_eq!(
            view.rows_containing_with_scan(pattern, frequencies)
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

/// Checks a prepared scan's expected hits and hit density, and its rows over two
/// runs to check reuse and appending.
fn check_scan(
    view: ColumnView<'_, u32>,
    frequencies: TokenFrequencyIndexView<'_>,
    pattern: &[u8],
    expected: &[usize],
) {
    let scan = ContainsScan::new(pattern, view.dict, frequencies).unwrap();
    let covered = view
        .codes
        .iter()
        .filter(|&&code| scan.probe_cover().is_some_and(|cover| cover.contains(code)))
        .count();
    assert_eq!(
        scan.expected_hits(frequencies).unwrap() as usize,
        covered,
        "{pattern:?}"
    );
    // Packing uses the planning stream's density, which this index describes.
    if let Query::Scan(prepared) = &scan.query {
        let density = if view.codes.is_empty() {
            0.0
        } else {
            covered as f64 / view.codes.len() as f64
        };
        assert_eq!(prepared.hit_density, density, "{pattern:?}");
    }
    let mut actual = vec![usize::MAX];
    let mut appended = actual.clone();
    for _ in 0..2 {
        scan.scan(view.codes, view.row_offsets, &mut actual);
        appended.extend_from_slice(expected);
        assert_eq!(actual, appended, "scan: {pattern:?}");
    }
}

#[test]
fn pattern_limits_are_independent() {
    let long_row = vec![b'a'; 300];
    let rows: &[&[u8]] = &[&long_row, b"abc", b"aaaaaaaaaa"];
    let column = compress_rows(rows);
    let view = column.view();
    let frequencies = TokenFrequencyIndex::build(view.codes, view.dict.num_tokens()).unwrap();
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
        view.rows_containing_with_scan(&oversized, frequencies.as_view()),
        Err(ContainsError::PatternTooLong {
            length: oversized.len(),
            max: ContainsScan::MAX_PATTERN_LEN,
        })
    );
}

/// The longest accepted pattern places the sink at `u16::MAX`, the largest
/// pattern offset that graph edges and the hit verifier can store.
#[test]
fn longest_pattern_reaches_the_largest_offset() {
    let mut rng = StdRng::seed_from_u64(7);
    let pattern: Vec<u8> = (0..ContainsScan::MAX_PATTERN_LEN)
        .map(|_| rng.random_range(b'a'..=b'z'))
        .collect();
    let mut wrong_first = pattern.clone();
    wrong_first[0] = b'0';
    let mut wrong_last = pattern.clone();
    wrong_last[pattern.len() - 1] = b'0';
    let padded = [b"xx".as_slice(), &pattern, b"yy"].concat();
    let rows: &[&[u8]] = &[&padded, &wrong_first, &pattern[1..], &pattern, &wrong_last];
    let column = compress_rows(rows);
    let view = column.view();
    let frequencies = TokenFrequencyIndex::build(view.codes, view.dict.num_tokens()).unwrap();
    let graph = AlignmentGraph::new(view.dict, &pattern).unwrap();
    assert_eq!(graph.sink(), u16::MAX);
    assert_eq!(
        view.rows_containing_with_scan(&pattern, frequencies.as_view()),
        Ok(vec![0, 3])
    );
}

/// Malformed row offsets panic with `BadRowOffsets`: the outer ones always, inner
/// ones once the scan reads their row.
#[test]
fn scan_rejects_malformed_row_offsets() {
    let column = compress_rows(&[b"alpha", b"beta"]);
    let view = column.view();
    let frequencies = TokenFrequencyIndex::build(view.codes, view.dict.num_tokens()).unwrap();
    let len = view.codes.len() as u32;
    let scan = ContainsScan::new(b"alpha", view.dict, frequencies.as_view()).unwrap();
    let everything = ContainsScan::new(b"", view.dict, frequencies.as_view()).unwrap();
    let cases: [&[u32]; 4] = [
        // The first offset after the last, and the last past the code stream.
        &[2, 1],
        &[0, len + 1],
        // An inner offset past the last: the end of the first row, read when the
        // scan starts, and of a row holding a hit.
        &[0, len, len - 1],
        &[0, 0, len, len - 1],
    ];
    let assert_rejects = |scan: &ContainsScan<'_>, offsets: &[u32]| {
        let panic = catch_unwind(AssertUnwindSafe(|| {
            scan.scan(view.codes, offsets, &mut Vec::new());
        }))
        .expect_err("malformed offsets must panic");
        let message = panic.downcast_ref::<String>().unwrap();
        assert!(
            message.contains("row offsets must be non-decreasing"),
            "{offsets:?}: {message}"
        );
    };
    for offsets in cases {
        assert_rejects(&scan, offsets);
    }
    // The empty pattern reads no inner offsets, but checks the outer ones.
    for offsets in &cases[..2] {
        assert_rejects(&everything, offsets);
    }
}

/// Both engines treat a code outside the dictionary as matching nothing, so an
/// occurrence cannot span it.
#[test]
fn codes_outside_the_dictionary_never_match() {
    let column = compress_rows(&[b"alpha"]);
    let view = column.view();
    let frequencies = TokenFrequencyIndex::build(view.codes, view.dict.num_tokens()).unwrap();
    let scan = ContainsScan::new(b"alpha", view.dict, frequencies.as_view()).unwrap();
    let dfa = ContainsDfa::new(b"alpha", view.dict).unwrap();
    let alpha = view.codes;
    for outside in [view.dict.num_tokens() as Token, Token::MAX] {
        for (codes, contains) in [
            (vec![outside], false),
            ([&[outside], alpha, &[outside]].concat(), true),
            ([&alpha[..2], &[outside], &alpha[2..]].concat(), false),
        ] {
            let mut rows = Vec::new();
            scan.scan(&codes, &[0, codes.len() as u32], &mut rows);
            assert_eq!(rows == [0], contains, "scan: {codes:?}");
            assert_eq!(row_contains(&codes, &dfa), contains, "DFA: {codes:?}");
        }
    }
}

/// A frequency index without one entry per dictionary token is rejected when
/// preparing a scan and when counting its expected hits.
#[test]
fn mismatched_frequency_index_is_an_error() {
    let column = compress_rows(&[b"abc"]);
    let view = column.view();
    let frequencies = TokenFrequencyIndex::build(view.codes, view.dict.num_tokens()).unwrap();
    let scan = ContainsScan::new(b"a", view.dict, frequencies.as_view()).unwrap();
    let dict_tokens = view.dict.num_tokens();
    for index_tokens in [dict_tokens - 1, dict_tokens + 1] {
        let mismatched = TokenFrequencyIndex::build(&[], index_tokens).unwrap();
        let mismatched = mismatched.as_view();
        let error = ContainsError::FrequencyIndexMismatch {
            index_tokens,
            dict_tokens,
        };
        assert_eq!(view.rows_containing_with_scan(b"a", mismatched), Err(error));
        assert_eq!(scan.expected_hits(mismatched), Err(error));
    }
}

#[test]
fn missing_alphabet_does_not_produce_a_partial_cover() {
    // Missing `b` is encountered after the graph has already emitted an edge.
    let mut bytes = vec![b'a'];
    bytes.resize(1 + MAX_TOKEN_SIZE, 0);
    let dict = CompactDictionaryView::validate_safety(&bytes, &[0u32, 1]).unwrap();
    let frequencies = TokenFrequencyIndex::build(&[0u16], 1).unwrap();
    assert_eq!(
        ContainsScan::new(b"ab", dict, frequencies.as_view()).unwrap_err(),
        ContainsError::MissingByteToken { byte: b'b' }
    );
}

/// Every row contains the empty pattern, including rows without codes. A
/// mismatched frequency index is still an error, as for any pattern.
#[test]
fn empty_pattern_matches_every_row() {
    let cases: &[&[&[u8]]] = &[&[b"", b"alpha", b"", b"beta", b""], &[b"", b""], &[]];
    for &rows in cases {
        let column = compress_rows(rows);
        let view = column.view();
        check(view, rows, &[b"", b"absent"]);
        let frequencies = TokenFrequencyIndex::build(view.codes, view.dict.num_tokens()).unwrap();
        let scan = ContainsScan::new(b"", view.dict, frequencies.as_view()).unwrap();
        assert!(scan.probe_cover().is_none());
        let unrelated = TokenFrequencyIndex::build(&[], 1).unwrap();
        assert_eq!(
            view.rows_containing_with_scan(b"", unrelated.as_view()),
            Err(ContainsError::FrequencyIndexMismatch {
                index_tokens: 1,
                dict_tokens: view.dict.num_tokens(),
            })
        );
    }
}

/// `Debug` shows the plan, not the dictionary or the verifier's tables.
#[test]
fn debug_shows_the_plan() {
    let column = compress_rows(&[b"alpha beta", b"gamma"]);
    let view = column.view();
    let index = TokenFrequencyIndex::build(view.codes, view.dict.num_tokens()).unwrap();
    let empty = ContainsScan::new(b"", view.dict, index.as_view()).unwrap();
    assert_eq!(
        format!("{empty:?}"),
        "ContainsScan { probe_cover: None, .. }"
    );
    let scan = ContainsScan::new(b"alpha", view.dict, index.as_view()).unwrap();
    let debug = format!("{scan:?}");
    assert!(
        debug.starts_with("ContainsScan { probe_cover: Some(ProbeCover {")
            && debug.contains(", matcher: ")
            && debug.ends_with(", .. }")
            && !debug.contains("bytes"),
        "{debug}"
    );
}

#[test]
fn prepared_scan_handles_empty_and_partial_inputs() {
    let rows: &[&[u8]] = &[b"", b"alpha", b"alphabet", b"beta", b""];
    let column = compress_rows(rows);
    let view = column.view();
    let frequencies = TokenFrequencyIndex::build(view.codes, view.dict.num_tokens()).unwrap();
    for pattern in [b"".as_slice(), b"alpha", b"absent"] {
        let scan = ContainsScan::new(pattern, view.dict, frequencies.as_view()).unwrap();
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
                // A slice that keeps the whole code stream and narrows only the
                // offsets, as Vortex slices, selects the same rows.
                let mut out = vec![usize::MAX];
                scan.scan(view.codes, &view.row_offsets[start..=end], &mut out);
                assert_eq!(out, expected, "{pattern:?}: unrebased rows {start}..{end}");
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
    let pattern = b"klmnopabcdefghijklmnop";
    // Internal entry alignments stop at 15 bytes; this needs the source path.
    assert!((0..dict.num_tokens()).any(|id| dict.token(id as Token) == &pattern[..MAX_TOKEN_SIZE]));
    check(column.view(), &rows, &[pattern]);
}

#[test]
fn cover_merges_overlapping_and_adjacent_runs() {
    let run = TokenRange::new;
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
fn false_zero_frequencies_cannot_hide_a_true_match() {
    let column = compress_rows(&[b"alpha beta", b"gamma", b"alphabet soup", b"delta"]);
    let view = column.view();
    assert!(view.codes.iter().all(|&code| code != 0));
    // Attribute every occurrence to unused token 0. All real tokens have weight 0.
    let mut cumulative = vec![view.codes.len() as u32; view.dict.num_tokens() + 1];
    cumulative[0] = 0;
    let frequencies = TokenFrequencyIndexView::new(&cumulative).unwrap();
    let scan = ContainsScan::new(b"alpha", view.dict, frequencies).unwrap();
    assert!(scan.probe_cover().is_some_and(|cover| !cover.is_empty()));
    assert_eq!(scan.expected_hits(frequencies), Ok(0));
    let mut rows = Vec::new();
    scan.scan(view.codes, view.row_offsets, &mut rows);
    assert_eq!(rows, [0, 2]);
}

#[test]
fn planned_cost_does_not_exceed_the_frequency_cut() {
    let corpus = crate::test_corpus::user_strings(200);
    let rows: Vec<&[u8]> = corpus.iter().map(|row| row.as_bytes()).collect();
    let column = compress_rows(&rows);
    let view = column.view();
    let frequencies = TokenFrequencyIndex::build(view.codes, view.dict.num_tokens()).unwrap();
    let freq = frequencies.as_view();
    let code_count = freq.total_frequency();
    let isa = detect_isa();
    for pattern in [
        b"e".as_slice(),
        b"://",
        b".com/page",
        b"https://www.example.com",
    ] {
        let graph = AlignmentGraph::new(view.dict, pattern).unwrap();
        let edges: Vec<_> = graph.edges().collect();
        let mut solver = MinCut::new(&edges, graph.sink());
        let baseline = ProbeCover::from_edge_cut(
            solver
                .solve(|at| u64::from(plan::edge_frequency(&edges[at], freq)))
                .iter()
                .map(|&at| edges[at as usize]),
        );
        let plan = plan::select(&edges, graph.sink(), freq, isa);
        assert_eq!(plan.hits, plan.cover.frequency(freq));
        let Some(model) = CostModel::of(isa) else {
            assert_eq!(plan.matcher, MatcherKind::Table);
            continue;
        };
        let (_, baseline_cost) = model.cheapest(&baseline, baseline.frequency(freq), code_count);
        let (matcher, cost) = model.cheapest(&plan.cover, plan.hits, code_count);
        assert_eq!(plan.matcher, matcher);
        assert!(
            cost <= baseline_cost,
            "{pattern:?}: planned {cost} against {baseline_cost}"
        );
    }
}
