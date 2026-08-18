// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for the whole prefilter: index construction, cover soundness against a
//! brute-force oracle, analysis metadata, and kernel agreement.

use super::cover::ProbeCover;
use super::frequency::checked_frequency_total;
use super::graph::{AlignmentGraph, build_alignment_graph, contained_tokens};
use super::mincut::minimum_vertex_cut;
use super::plan::plan;
use super::{
    TokenFrequencyIndex, TokenFrequencyIndexError, analyze_prefilter, build_token_frequency_index,
    prefilter_candidates,
};
use crate::core::dictionary::{CompactDictionaryView, DictionaryView};
use crate::core::types::{Token, TokenRange};
use crate::search::{ContainsTable, contains};
use crate::{Column, ColumnView, DEFAULT_CONFIG, compress};

fn candidates(
    view: ColumnView<'_, u32>,
    dict: CompactDictionaryView<'_>,
    frequencies: &TokenFrequencyIndex,
    pattern: &[u8],
) -> Vec<usize> {
    if pattern.is_empty() {
        return (0..view.num_rows()).collect();
    }
    let mut out = Vec::new();
    let analysis = analyze_prefilter(pattern, dict, frequencies);
    prefilter_candidates(view.codes, view.row_offsets, &analysis, &mut out).unwrap();
    out
}

/// The obvious way to find the tokens containing `needle`: compare every window
/// of every token. This is the oracle for the flat-payload `memmem` sweep that
/// replaced it — the sweep is a different algorithm over a different buffer, with
/// match attribution and a resume rule of its own, so it earns a reference
/// implementation rather than only end-to-end soundness checks.
fn contained_tokens_by_scan(dict: CompactDictionaryView<'_>, needle: &[u8]) -> Vec<Token> {
    (0..dict.num_tokens() as Token)
        .filter(|&id| {
            let token = dict.token(id);
            token.len() >= needle.len() && token.windows(needle.len()).any(|w| w == needle)
        })
        .collect()
}

fn compress_rows(rows: &[&[u8]]) -> Column<u32> {
    let mut bytes = Vec::new();
    let mut offsets = vec![0u32];
    for r in rows {
        bytes.extend_from_slice(r);
        offsets.push(bytes.len() as u32);
    }
    compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap()
}

fn byte_contains(hay: &[u8], needle: &[u8]) -> bool {
    needle.is_empty() || hay.windows(needle.len()).any(|w| w == needle)
}

fn decode_row(view: ColumnView<'_, u32>, k: usize) -> Vec<u8> {
    let mut buf =
        vec![std::mem::MaybeUninit::uninit(); view.row_decoded_len(k) + crate::DECODE_PADDING];
    // SAFETY: buffer sized for row `k`; view from a trusted column.
    let w = unsafe { view.decompress_row_into(k, &mut buf) };
    unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), w) }.to_vec()
}

#[test]
fn frequency_index_builds_and_validates() {
    let index = build_token_frequency_index(&[2, 0, 2, 3, 2, 0], 5).unwrap();

    assert_eq!(index.num_tokens(), 5);
    assert_eq!(index.frequency(0), 2);
    assert_eq!(index.frequency(1), 0);
    assert_eq!(index.frequency(2), 3);
    assert_eq!(index.frequency(3), 1);
    assert_eq!(index.frequency(4), 0);
    assert_eq!(index.range_frequency(TokenRange { begin: 1, last: 3 }), 4);
    assert_eq!(index.range_frequency(TokenRange::EMPTY), 0);
    assert_eq!(
        build_token_frequency_index(&[0, 2], 2),
        Err(TokenFrequencyIndexError::CodeOutOfRange)
    );
    assert_eq!(
        build_token_frequency_index(&[], Token::MAX as usize + 2),
        Err(TokenFrequencyIndexError::TooManyTokens)
    );
    #[cfg(target_pointer_width = "64")]
    assert_eq!(
        checked_frequency_total(u32::MAX as usize + 1),
        Err(TokenFrequencyIndexError::FrequencyOverflow)
    );
}

/// Whether the sink is reachable from the source without entering a blocked
/// node.
///
/// Blocking every probe asks the DAG's central invariant — that no layout of
/// the pattern escapes unprobed — and blocking a cut asks whether that cut
/// covers the DAG. Both are properties of the edge set, so this walks `edges`
/// rather than re-deriving anything the builder computed.
fn sink_reachable_avoiding(graph: &AlignmentGraph, blocked: &[bool]) -> bool {
    let mut adjacency = vec![Vec::new(); graph.num_nodes()];
    for &(from, to) in &graph.edges {
        adjacency[from as usize].push(to as usize);
    }
    let mut seen = vec![false; graph.num_nodes()];
    let mut stack = vec![graph.source as usize];
    seen[graph.source as usize] = true;
    while let Some(node) = stack.pop() {
        if node == graph.sink as usize {
            return true;
        }
        for &next in &adjacency[node] {
            if !seen[next] && !blocked[next] {
                seen[next] = true;
                stack.push(next);
            }
        }
    }
    false
}

/// Whether every row that really contains the pattern holds a token one of
/// `selection`'s probes covers — what a cover exists to guarantee. The mandatory
/// [`contained`](AlignmentGraph::contained) tokens join the selection here, since
/// the graph deliberately leaves them out of a cut.
fn covers_every_match(
    view: ColumnView<'_, u32>,
    graph: &AlignmentGraph,
    selection: &[u32],
    want: &[usize],
) -> bool {
    let mut members = graph.membership(selection);
    for &id in &graph.contained {
        members[id as usize] = true;
    }
    want.iter()
        .all(|&row| view.row_codes(row).iter().any(|&c| members[c as usize]))
}

/// The properties the alignment DAG has to have, checked directly on the graph:
/// no path escapes unprobed, every probe's weight is the work it would cost, the
/// probes really do cover every matching row, states at equal offsets are shared
/// rather than duplicated per alignment — and the minimum cut over all of that
/// is both sound and, while the probe set is small enough to enumerate, optimal.
fn check_graph(
    view: ColumnView<'_, u32>,
    frequencies: &TokenFrequencyIndex,
    pat: &[u8],
    want: &[usize],
) {
    let graph = build_alignment_graph(view.dict, pat, frequencies);
    let probes: Vec<u32> = (0..graph.num_nodes() as u32)
        .filter(|&node| graph.is_probe(node))
        .collect();

    let mut blocked = vec![false; graph.num_nodes()];
    for &node in &probes {
        blocked[node as usize] = true;
    }
    assert!(
        !sink_reachable_avoiding(&graph, &blocked),
        "a source-to-sink path carries no probe for {pat:?}"
    );

    // A probe's weight is the objective the cut minimizes, so it has to be the
    // number of codes the probe would actually match. Nothing downstream can
    // notice the cut optimizing a wrong number.
    for &node in &probes {
        let covered = graph.membership(&[node]);
        let matched = view.codes.iter().filter(|&&c| covered[c as usize]).count();
        assert_eq!(
            graph.weight(node) as usize,
            matched,
            "probe {node} misreports its term frequency for {pat:?}"
        );
    }

    // Cutting everything cuttable is the weakest sound cover the graph can
    // produce; if even that misses a matching row, the DAG is incomplete.
    assert!(
        covers_every_match(view, &graph, &probes, want),
        "some row matching {pat:?} holds no probe token at all"
    );

    // Source, sink, up to 16 alignments and 15 first-token sets, then one
    // state, one range probe and one point probe per needle offset. Per
    // alignment chains instead of merged states would be ~16n.
    assert!(
        graph.num_nodes() <= 3 * pat.len() + 33,
        "{} nodes for a {}-byte needle: states are not being merged",
        graph.num_nodes(),
        pat.len()
    );

    // The cut is what the plan will actually scan for, so it has to block the
    // DAG on its own and still catch every matching row.
    let cut = minimum_vertex_cut(&graph);
    blocked.fill(false);
    for &node in &cut {
        blocked[node as usize] = true;
    }
    assert!(
        !sink_reachable_avoiding(&graph, &blocked),
        "the minimum cut leaves a source-to-sink path open for {pat:?}"
    );
    assert!(
        covers_every_match(view, &graph, &cut, want),
        "the minimum cut misses a row matching {pat:?}"
    );

    // Optimality, exhaustively, while the probe set is small enough to
    // enumerate: nothing cheaper than the cut blocks the DAG. This is the claim
    // that separates one cut of the merged graph from the per-alignment local
    // choice it replaces, and max-flow is not the kind of code that fails loudly.
    if probes.len() <= 12 {
        let best: u64 = cut.iter().map(|&node| u64::from(graph.weight(node))).sum();
        for mask in 0u32..(1 << probes.len()) {
            let mut weight = 0u64;
            blocked.fill(false);
            for (bit, &node) in probes.iter().enumerate() {
                if (mask >> bit) & 1 == 1 {
                    weight += u64::from(graph.weight(node));
                    blocked[node as usize] = true;
                }
            }
            if weight < best {
                assert!(
                    sink_reachable_avoiding(&graph, &blocked),
                    "a cover of weight {weight} beats the minimum cut's {best} for {pat:?}"
                );
            }
        }
    }
}

fn check(rows: &[&[u8]], patterns: &[&[u8]]) {
    let col = compress_rows(rows);
    let view = col.view();
    let frequencies = build_token_frequency_index(view.codes, view.dict.num_tokens()).unwrap();
    for &pat in patterns {
        let want: Vec<usize> = (0..view.num_rows())
            .filter(|&k| byte_contains(&decode_row(view, k), pat))
            .collect();
        if !pat.is_empty() {
            check_graph(view, &frequencies, pat, &want);
            assert_eq!(
                contained_tokens(view.dict, pat),
                contained_tokens_by_scan(view.dict, pat),
                "the payload sweep and a per-token scan disagree for {pat:?}"
            );
        }

        let cand = candidates(view, view.dict, &frequencies, pat);
        assert!(want.iter().all(|row| cand.contains(row)), "unsound {pat:?}");
        assert!(
            cand.windows(2).all(|w| w[0] < w[1]),
            "unordered candidates for {pat:?}"
        );

        let table = ContainsTable::new(pat, view.dict);
        let exact: Vec<_> = cand
            .into_iter()
            .filter(|&row| contains(view.row_codes(row), &table))
            .collect();
        assert_eq!(exact, want, "incorrect result for {pat:?}");
    }
}

/// The contained-token sweep searches the flat payload, where a match can span
/// two tokens and must be rejected — but rejecting it must not skip past a real
/// match inside the token it ran into.
///
/// A dictionary that learns `aa` puts it directly after the single-byte `a`
/// (nothing sorts between them), so the payload holds `aaa`: the span at offset 0
/// is rejected, and the match at offset 1 is the only witness that `aa` contains
/// the needle. Resuming after the rejected span would lose the token entirely.
#[test]
fn contained_tokens_survive_a_match_spanning_two_tokens() {
    let rows: Vec<&[u8]> = vec![
        b"aaaaaaaaaaaaaaaa",
        b"aaaaaaaa aaaaaaaa",
        b"xaaaaay",
        b"aaa",
        b"aa",
        b"zzaazz",
    ];
    let col = compress_rows(&rows);
    let view = col.view();
    let dict = view.dict;
    let ntok = dict.num_tokens();

    // Without the adjacency the test would pass vacuously.
    assert!(
        (0..ntok as Token).any(|id| dict.token(id) == b"aa"),
        "corpus did not train the `aa` token this test is about"
    );

    let frequencies = build_token_frequency_index(view.codes, ntok).unwrap();
    for pat in [b"aa".as_slice(), b"aaa", b"aaaa"] {
        let graph = build_alignment_graph(dict, pat, &frequencies);
        assert_eq!(
            graph.contained,
            contained_tokens_by_scan(dict, pat),
            "contained tokens for {pat:?} disagree with a per-token scan"
        );
    }
}

#[test]
fn sound_on_edge_cases() {
    let rows: &[&[u8]] = &[
        b"",
        b"hello world",
        b"world peace",
        b"abcabcabc",
        b"xabcabcy",
        b"aaaaab",
        b"aabaab",
    ];
    check(
        rows,
        &[
            b"", b"hello", b"world", b"o w", b"bca", b"bcabca", b"aa", b"aab", b"aabaa", b"absent",
        ],
    );
}

#[test]
fn matches_brute_force_on_repetitive_corpus() {
    use crate::test_corpus::user_strings;
    let corpus: Vec<Vec<u8>> = user_strings(50)
        .into_iter()
        .map(String::into_bytes)
        .collect();
    let rows: Vec<&[u8]> = corpus.iter().map(Vec::as_slice).collect();
    check(
        &rows,
        &[
            b"example",
            b"https",
            b"://",
            b".com",
            b"/page",
            b"ftp",
            b"zzz",
            b"w",
            b"https://www.example.com/",
        ],
    );
}

#[test]
fn matches_brute_force_on_binary_corpus() {
    use crate::test_corpus::binary_strings;
    let corpus = binary_strings(40, 24, 11);
    let rows: Vec<&[u8]> = corpus.iter().map(Vec::as_slice).collect();
    let patterns: &[&[u8]] = &[b"", b"\x00", b"\xff", b"\x00\x01", &[7u8], &[200u8, 201]];
    check(&rows, patterns);
}

#[test]
fn prefilter_accepts_pattern_over_255_bytes() {
    let long = vec![b'a'; 300];
    let short = vec![b'a'; 10];
    let rows: &[&[u8]] = &[&long, b"abc", &short];
    let col = compress_rows(rows);
    let view = col.view();
    let frequencies = build_token_frequency_index(view.codes, view.dict.num_tokens()).unwrap();
    let pat = vec![b'a'; 256];
    let mut candidates = Vec::new();
    let analysis = analyze_prefilter(&pat, view.dict, &frequencies);
    prefilter_candidates(view.codes, view.row_offsets, &analysis, &mut candidates).unwrap();

    assert!(candidates.contains(&0));
}

#[test]
fn analysis_reports_normalized_cover_frequency() {
    let col = compress_rows(&[b"alpha", b"beta"]);
    let view = col.view();
    let frequencies = build_token_frequency_index(view.codes, view.dict.num_tokens()).unwrap();
    let analysis = analyze_prefilter(b"a", view.dict, &frequencies);
    let cover = analysis.probe_cover();
    let expected: u32 = cover
        .points()
        .iter()
        .map(|&token| frequencies.frequency(token))
        .chain(
            cover
                .ranges()
                .iter()
                .map(|&range| frequencies.range_frequency(range)),
        )
        .sum();

    assert_eq!(analysis.covered_frequency(), expected);
    assert_eq!(
        analysis.covered_fraction(),
        f64::from(expected) / view.codes.len() as f64
    );
}

/// A cut hands over overlapping probe sets — a range and a point naming the
/// same token, two ranges that abut, the mandatory contained tokens unioned on
/// top — so the probes are re-derived from the ids themselves. Anything less is
/// a redundant comparison paid on every vector of the code stream.
#[test]
fn cover_probes_the_maximal_runs_of_its_membership() {
    let table = vec![true, true, false, true, false, false, true, true, true];
    let pf = ProbeCover::from_membership(table.clone(), Some);

    assert_eq!(pf.points, vec![3]);
    assert_eq!(
        pf.ranges,
        vec![
            TokenRange { begin: 0, last: 1 },
            TokenRange { begin: 6, last: 8 },
        ]
    );
    assert_eq!(pf.table, table);
    assert_eq!(
        ProbeCover::from_membership(vec![false; 4], Some).points,
        Vec::<Token>::new()
    );
}

/// What a run gives up, it gives up in both shapes: the lists are what the
/// kernels compare against and the table is what the scalar tail looks up, so an
/// id dropped from one has to leave the other.
#[test]
fn cover_narrows_and_drops_the_runs_it_is_told_to() {
    // Runs 0..=1, 3..=3, 6..=9, 11..=12.
    let table = vec![
        true, true, false, true, false, false, true, true, true, true, false, true, true,
    ];
    let pf = ProbeCover::from_membership(table, |run| match run.begin {
        0 => None,                                   // dropped whole
        6 => Some(TokenRange { begin: 7, last: 8 }), // trimmed both ends
        11 => Some(TokenRange {
            begin: 12,
            last: 12,
        }), // a range becomes a point
        _ => Some(run),
    });

    assert_eq!(pf.points, vec![3, 12]);
    assert_eq!(pf.ranges, vec![TokenRange { begin: 7, last: 8 }]);
    assert_eq!(
        pf.table,
        vec![
            false, false, false, true, false, false, false, true, true, false, false, false, true,
        ]
    );
    assert!(!ProbeCover::from_membership(vec![true; 4], Some).is_empty());
    assert!(ProbeCover::from_membership(vec![true; 4], |_| None).is_empty());
}

/// A token the code stream never uses is held by no row, so probing for it can
/// only cost comparisons. The cut selects such tokens readily — they are free by
/// its objective — so the planner has to take them back out.
#[test]
fn unused_tokens_are_kept_out_of_the_cover() {
    let col = compress_rows(&[
        b"https://example.com/search?q=onpair",
        b"https://example.com/index.html",
        b"https://example.org/search?q=compression",
        b"https://example.org/cart/checkout",
    ]);
    let view = col.view();
    let frequencies = build_token_frequency_index(view.codes, view.dict.num_tokens()).unwrap();

    for pattern in [
        b"search?q=".as_slice(),
        b"example.org".as_slice(),
        b"/cart/checkout".as_slice(),
        b"ex".as_slice(),
    ] {
        let pf = plan(view.dict, pattern, &frequencies);

        let used = |id: Token| frequencies.frequency(id) != 0;

        // A point is one id: an unused one would be a comparison that can never
        // fire. A range is one comparison however long it is, so only its ends
        // are worth trimming — an interior unused id rides along for free.
        for &id in &pf.points {
            assert!(used(id), "point probe {id} never occurs");
        }
        for range in &pf.ranges {
            assert!(used(range.begin), "range {range:?} starts unused");
            assert!(used(range.last), "range {range:?} ends unused");
        }
    }
}

/// Pruning must not change which rows are admitted — the whole argument for it
/// is that it cannot.
#[test]
fn pruning_does_not_change_the_candidates() {
    let rows: Vec<&[u8]> = vec![
        b"alpha beta gamma",
        b"beta gamma delta",
        b"gamma delta epsilon",
        b"delta epsilon alpha",
        b"epsilon alpha beta",
    ];
    let col = compress_rows(&rows);
    let view = col.view();
    let frequencies = build_token_frequency_index(view.codes, view.dict.num_tokens()).unwrap();

    for pattern in [
        b"beta".as_slice(),
        b"gamma delta".as_slice(),
        b"a b".as_slice(),
        b"epsilon alpha".as_slice(),
    ] {
        let got = candidates(view, view.dict, &frequencies, pattern);
        for (k, row) in rows.iter().enumerate() {
            if byte_contains(row, pattern) {
                assert!(got.contains(&k), "{pattern:?} lost row {k}");
            }
        }
    }
}

/// When every id a pattern's cover would name is absent from the code stream, no
/// row can match. That is exact, not an approximation, and it needs no scan —
/// so it is answered even where there is no SIMD kernel to refuse with.
#[test]
fn a_fully_unused_cover_admits_no_rows() {
    let col = compress_rows(&[b"alpha", b"beta", b"gamma"]);
    let view = col.view();
    let frequencies = build_token_frequency_index(view.codes, view.dict.num_tokens()).unwrap();

    let mut out = Vec::new();
    let analysis = analyze_prefilter(b"\xff", view.dict, &frequencies);
    assert_eq!(analysis.covered_frequency(), 0);
    assert_eq!(analysis.covered_fraction(), 0.0);
    assert!(analysis.probe_cover().is_empty());
    assert_eq!(
        prefilter_candidates(view.codes, view.row_offsets, &analysis, &mut out,),
        Ok(())
    );
    assert!(out.is_empty());
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[test]
fn wide_probe_cover_dispatches_soundly() {
    let pf = ProbeCover {
        points: vec![0; 33],
        ranges: Vec::new(),
        table: vec![true],
    };
    let mut candidates = Vec::new();

    super::scan::scan(&[0], &[0u32, 1], &pf, 1, &mut candidates).unwrap();
    assert_eq!(candidates, vec![0]);
}

/// Each hit row is appended exactly once, ascending, however the hits fall
/// across vector blocks. A row longer than every vector width, hits in the
/// scalar tail, and empty rows before, between and after the hits are the cases
/// the sink's `row_end` shortcut has to get right.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[test]
fn each_hit_row_is_appended_once_in_order() {
    const HIT: Token = 1;
    const MISS: Token = 0;

    let rows: Vec<Vec<Token>> = vec![
        Vec::new(),            // empty, before any hit
        vec![HIT; 40],         // spans several blocks at every vector width
        Vec::new(),            // empty, between two hit rows
        vec![MISS; 40],        // no hit
        vec![MISS, MISS, HIT], // hit lands in the scalar tail
        Vec::new(),            // empty, after the last hit
    ];
    let mut codes = Vec::new();
    let mut row_offsets = vec![0u32];
    for row in &rows {
        codes.extend_from_slice(row);
        row_offsets.push(codes.len() as u32);
    }

    let pf = ProbeCover::from_membership(vec![false, true], Some);
    let mut out = Vec::new();
    super::scan::scan(&codes, &row_offsets, &pf, 43, &mut out).unwrap();
    assert_eq!(out, vec![1, 4]);

    let mut oracle = Vec::new();
    super::scan::scan_scalar(&codes, &row_offsets, &pf, &mut oracle);
    assert_eq!(out, oracle);
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn assert_kernel_matches_scalar(
    kernel: impl Fn(&[Token], &[u32], &ProbeCover, bool, &mut Vec<usize>),
) {
    use crate::test_corpus::user_strings;
    let corpus: Vec<Vec<u8>> = user_strings(60)
        .into_iter()
        .map(String::into_bytes)
        .collect();
    let rows: Vec<&[u8]> = corpus.iter().map(Vec::as_slice).collect();
    let col = compress_rows(&rows);
    let view = col.view();
    let frequencies = build_token_frequency_index(view.codes, view.dict.num_tokens()).unwrap();
    let patterns: &[&[u8]] = &[
        b"e",
        b"://",
        b"example",
        b".com/page",
        b"https://www.example.com",
        b"zzz",
    ];
    for &pat in patterns {
        let pf = plan(view.dict, pat, &frequencies);
        let mut scalar = Vec::new();
        let mut simd = Vec::new();
        super::scan::scan_scalar(view.codes, view.row_offsets, &pf, &mut scalar);
        kernel(view.codes, view.row_offsets, &pf, false, &mut simd);
        assert_eq!(scalar, simd, "kernel disagrees with scalar for {pat:?}");
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_matches_scalar() {
    assert_kernel_matches_scalar(super::scan::scan_neon::<u32>);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn x86_kernels_match_scalar() {
    assert_kernel_matches_scalar(super::scan::scan_sse2::<u32>);
    if std::is_x86_feature_detected!("avx2") {
        assert_kernel_matches_scalar(|codes, ro, pf, sparse_row_mapping, cand| unsafe {
            super::scan::scan_avx2(codes, ro, pf, sparse_row_mapping, cand)
        });
    }
    if std::is_x86_feature_detected!("avx512bw") {
        assert_kernel_matches_scalar(|codes, ro, pf, sparse_row_mapping, cand| unsafe {
            super::scan::scan_avx512(codes, ro, pf, sparse_row_mapping, cand)
        });
    }
}

#[test]
fn signed_bias_range_matches_unsigned() {
    fn in_range_biased(c: u16, lo: u16, hi: u16) -> bool {
        const BIAS: u16 = 0x8000;
        let cb = (c ^ BIAS) as i16;
        let lob = (lo ^ BIAS) as i16;
        let hib = (hi ^ BIAS) as i16;
        !(lob > cb || cb > hib)
    }
    let bounds: &[(u16, u16)] = &[
        (0, 0),
        (0, u16::MAX),
        (0x7FFF, 0x8000),
        (0x8000, 0xFFFF),
        (0x00FF, 0xFF00),
        (1234, 1234),
        (40000, 50000),
    ];
    for &(lo, hi) in bounds {
        for c in 0..=u16::MAX {
            assert_eq!(
                in_range_biased(c, lo, hi),
                lo <= c && c <= hi,
                "c={c} lo={lo} hi={hi}"
            );
        }
    }
}
