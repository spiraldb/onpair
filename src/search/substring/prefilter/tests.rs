// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end soundness, graph invariants, and SIMD/scalar agreement.

use super::cover::ProbeCover;
use super::graph::{
    AlignmentGraph, Edge, PROBE_SET_SIZE_LIMIT, PROBE_SET_SIZE_LIMIT_K1, alignment_candidates,
    build_alignment_graph,
};
use super::mincut::min_cut;
use super::plan::{cheapest_cover, cover_frequency};
use super::scan::{Region, scan_ns};
use super::{analyze_prefilter, prefilter_candidates, prefilter_is_likely_profitable};
use crate::core::dictionary::{CompactDictionaryView, DictionaryView};
use crate::core::types::{MAX_TOKEN_SIZE, Token, TokenRange};
use crate::search::index::{
    TokenFrequencyIndex, TokenFrequencyIndexStorage, build_token_frequency_index,
};
use crate::search::{ContainsTable, contains};
use crate::{Column, ColumnView, DEFAULT_CONFIG, compress};

fn candidates<S: TokenFrequencyIndexStorage>(
    view: ColumnView<'_, u32>,
    dict: CompactDictionaryView<'_>,
    frequencies: &TokenFrequencyIndex<S>,
    pattern: &[u8],
) -> Vec<usize> {
    let mut out = Vec::new();
    let analysis = analyze_prefilter(pattern, dict, frequencies, view.num_rows());
    prefilter_candidates(view.codes, view.row_offsets, dict, &analysis, &mut out);
    out
}

struct BorrowedFrequencies<'a>(&'a [u32]);

impl TokenFrequencyIndexStorage for BorrowedFrequencies<'_> {
    fn cumulative(&self) -> &[u32] {
        self.0
    }
}

/// The obvious way to find the tokens containing `needle` without starting with
/// it: compare every window of every token. This is the oracle for the anchored
/// payload sweep that replaced it — the sweep is a different algorithm over a different buffer, with
/// match attribution and a resume rule of its own, so it earns a reference
/// implementation rather than only end-to-end soundness checks.
fn contained_tokens_by_scan(dict: CompactDictionaryView<'_>, needle: &[u8]) -> Vec<Token> {
    (0..dict.num_tokens() as Token)
        .filter(|&id| {
            let token = dict.token(id);
            !token.starts_with(needle)
                && token.len() >= needle.len()
                && token.windows(needle.len()).any(|w| w == needle)
        })
        .collect()
}

/// The obvious way to find the first-token sets: try every alignment against
/// every token. This is the oracle for the anchored sweep, which visits only
/// the payload positions where `needle[..2]` occurs and so never looks at most
/// tokens at all.
fn first_token_sets_by_scan(
    dict: CompactDictionaryView<'_>,
    needle: &[u8],
) -> (
    [usize; MAX_TOKEN_SIZE],
    [Token; MAX_TOKEN_SIZE * PROBE_SET_SIZE_LIMIT],
) {
    let mut count = [0usize; MAX_TOKEN_SIZE];
    let mut ids = [0 as Token; MAX_TOKEN_SIZE * PROBE_SET_SIZE_LIMIT];
    for id in 0..dict.num_tokens() as Token {
        let token = dict.token(id);
        for k in 1..needle.len().min(MAX_TOKEN_SIZE) {
            if k < token.len() && token[token.len() - k..] == needle[..k] {
                // Alignment 1 stores only what its walk saw before it stopped.
                let cap = if k == 1 {
                    PROBE_SET_SIZE_LIMIT_K1 + 1
                } else {
                    PROBE_SET_SIZE_LIMIT
                };
                if count[k] < cap {
                    ids[k * PROBE_SET_SIZE_LIMIT + count[k]] = id;
                }
                // Saturating, like the pass: the planner reads only empty or
                // too big.
                count[k] = (count[k] + 1).min(PROBE_SET_SIZE_LIMIT + 1);
            }
        }
    }
    if count[1] > PROBE_SET_SIZE_LIMIT_K1 {
        count[1] = PROBE_SET_SIZE_LIMIT + 1;
    }
    (count, ids)
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

/// What a frequency-weighted cut pays for `edge`.
fn probe_weight(edge: &Edge) -> u64 {
    u64::from(edge.frequency())
}

/// Whether the sink is reachable from the source without taking an edge
/// `blocked` rejects.
///
/// Blocking every probe asks the DAG's central invariant — that no layout of
/// the pattern escapes unprobed — and blocking a cut asks whether that cut
/// covers the DAG. Both are properties of the edge set, so this walks `edges`
/// rather than re-deriving anything the builder computed.
fn sink_reachable_avoiding(graph: &AlignmentGraph, blocked: impl Fn(&Edge) -> bool) -> bool {
    let mut adjacency = vec![Vec::new(); graph.nodes.count()];
    for edge in &graph.edges {
        adjacency[edge.from as usize].push(edge);
    }
    let mut seen = vec![false; graph.nodes.count()];
    let mut stack = vec![graph.nodes.source() as usize];
    seen[graph.nodes.source() as usize] = true;
    while let Some(node) = stack.pop() {
        if node == graph.nodes.sink() as usize {
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

/// Whether `edge` is one of `selection`'s edges. Both borrow the same graph, so
/// identity is the address — which is what a selection of references means.
fn selected(selection: &[&Edge], edge: &Edge) -> bool {
    selection.iter().any(|&picked| std::ptr::eq(picked, edge))
}

/// Whether every row that really contains the pattern holds a token one of
/// `selection`'s probes covers — what a cover exists to guarantee.
fn covers_every_match(view: ColumnView<'_, u32>, selection: &[&Edge], want: &[usize]) -> bool {
    let cover = ProbeCover::from_edge_cut(selection);
    want.iter()
        .all(|&row| view.row_codes(row).iter().any(|&c| cover.contains(c)))
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
    let graph = build_alignment_graph(view.dict, pat, frequencies.as_view());
    let probes: Vec<&Edge> = graph.edges.iter().filter(|edge| edge.cuttable()).collect();

    assert!(
        !sink_reachable_avoiding(&graph, Edge::cuttable),
        "a source-to-sink path carries no probe for {pat:?}"
    );

    // A probe's weight is the objective the cut minimizes, so it has to be the
    // number of codes the probe would actually match. Nothing downstream can
    // notice the cut optimizing a wrong number.
    for &edge in &probes {
        let covered = ProbeCover::from_edge_cut(&[edge]);
        let matched = view.codes.iter().filter(|&&c| covered.contains(c)).count();
        assert_eq!(
            probe_weight(edge) as usize,
            matched,
            "probe {edge:?} misreports its term frequency for {pat:?}"
        );
    }

    // Cutting everything cuttable is the weakest sound cover the graph can
    // produce; if even that misses a matching row, the DAG is incomplete.
    assert!(
        covers_every_match(view, &probes, want),
        "some row matching {pat:?} holds no probe token at all"
    );

    // Node ids are needle offsets, so merged states are structural rather than
    // checkable — but per alignment chains would still show up as duplicated
    // steps. Two edges per offset, plus one per alignment, is the bound that
    // holds only while every alignment shares one chain.
    assert!(
        graph.edges.len() <= 2 * pat.len() + 16,
        "{} edges for a {}-byte needle: states are not being merged",
        graph.edges.len(),
        pat.len()
    );

    // A cut has to block the DAG on its own and still catch every matching
    // row, whatever it was weighted by.
    let cut = min_cut(&graph.edges, graph.nodes, probe_weight);
    assert!(
        !sink_reachable_avoiding(&graph, |edge| selected(&cut, edge)),
        "the minimum cut leaves a source-to-sink path open for {pat:?}"
    );
    assert!(
        covers_every_match(view, &cut, want),
        "the minimum cut misses a row matching {pat:?}"
    );

    // Optimality, exhaustively, while the probe set is small enough to
    // enumerate: nothing cheaper than the cut blocks the DAG. This is the claim
    // that separates one cut of the merged graph from the per-alignment local
    // choice it replaces, and max-flow is not the kind of code that fails loudly.
    if probes.len() <= 12 {
        let best: u64 = cut.iter().copied().map(probe_weight).sum();
        for mask in 0u32..(1 << probes.len()) {
            let subset: Vec<&Edge> = probes
                .iter()
                .enumerate()
                .filter(|(bit, _)| (mask >> bit) & 1 == 1)
                .map(|(_, &edge)| edge)
                .collect();
            let weight: u64 = subset.iter().copied().map(probe_weight).sum();
            if weight < best {
                assert!(
                    sink_reachable_avoiding(&graph, |edge| selected(&subset, edge)),
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
            let candidates = alignment_candidates(view.dict, pat);
            assert_eq!(
                candidates.contained,
                contained_tokens_by_scan(view.dict, pat),
                "the payload sweep and a per-token scan disagree for {pat:?}"
            );
            assert_eq!(
                (candidates.first.count, candidates.first.ids),
                first_token_sets_by_scan(view.dict, pat),
                "the anchored sweep and a per-token scan disagree for {pat:?}"
            );
        }

        let mut cand = Vec::new();
        if pat.is_empty() {
            cand.extend(0..view.num_rows());
        } else {
            let analysis = analyze_prefilter(pat, view.dict, &frequencies, view.num_rows());
            prefilter_candidates(
                view.codes,
                view.row_offsets,
                view.dict,
                &analysis,
                &mut cand,
            );
        }
        assert_eq!(cand, want, "incorrect result for {pat:?}");

        let table = ContainsTable::new(pat, view.dict);
        let by_kmp: Vec<_> = (0..view.num_rows())
            .filter(|&row| contains(view.row_codes(row), &table))
            .collect();
        assert_eq!(by_kmp, want, "the KMP oracle disagrees for {pat:?}");
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

    for pat in [b"aa".as_slice(), b"aaa", b"aaaa"] {
        assert_eq!(
            alignment_candidates(dict, pat).contained,
            contained_tokens_by_scan(dict, pat),
            "contained tokens for {pat:?} disagree with a per-token scan"
        );
    }
}

/// A needle whose greedy first token is a full [`MAX_TOKEN_SIZE`], with more
/// needle left over.
///
/// Alignment `k` usually doubles as the boundary-aligned layout: `first_count[k]`
/// counts tokens that *equal* `needle[..k]`, not only ones ending with it, so a
/// first token of length `L` is a member of alignment `L`'s set. That stand-in
/// runs out at `L == MAX_TOKEN_SIZE`, which `ks_ending_in` never enumerates — so
/// this is the one shape where the source's own chain is the sole representation
/// of a match starting at a token boundary.
#[test]
fn boundary_layout_with_a_maximal_first_token_is_covered() {
    let rows: Vec<Vec<u8>> = (0..40)
        .map(|i| {
            let mut row = b"abcdefghijklmnop".repeat(4);
            row.extend_from_slice(format!("{i:03}").as_bytes());
            row
        })
        .collect();
    let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
    let needle = b"klmnopabcdefghijklmnop";

    // Without a maximal first token the test would pass vacuously.
    let col = compress_rows(&refs);
    let dict = col.view().dict;
    assert!(
        (0..dict.num_tokens() as Token).any(|id| dict.token(id) == &needle[..MAX_TOKEN_SIZE]),
        "corpus did not train the {MAX_TOKEN_SIZE}-byte first token this test is about"
    );

    check(&refs, &[needle.as_slice()]);
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

/// Empty rows have no token that a probe scan can find. The all-rows answer must
/// include them and preserve the candidate API's append semantics.
#[test]
fn empty_pattern_appends_all_rows() {
    let cases: &[&[&[u8]]] = &[&[b"", b"alpha", b"", b"beta", b""], &[b"", b""], &[]];
    for &rows in cases {
        let col = compress_rows(rows);
        let view = col.view();
        let frequencies = build_token_frequency_index(view.codes, view.dict.num_tokens()).unwrap();
        let analysis = analyze_prefilter(b"", view.dict, &frequencies, view.num_rows());
        assert!(analysis.probe_cover().is_empty());
        assert_eq!(analysis.comparison_cost(), 0);
        assert_eq!(analysis.covered_frequency(), 0);
        assert_eq!(analysis.covered_fraction(), 0.0);
        assert_eq!(analysis.expected_scan_ns(), 0.0);
        assert_eq!(analysis.total_frequency(), view.codes.len() as u32);
        assert_eq!(
            analysis.expected_candidate_row_fraction(view.num_rows()),
            if rows.is_empty() { 0.0 } else { 1.0 }
        );
        assert!(prefilter_is_likely_profitable(&analysis, view.num_rows()));

        let expected: Vec<_> = (0..rows.len()).collect();
        let mut got = vec![usize::MAX];
        prefilter_candidates(view.codes, view.row_offsets, view.dict, &analysis, &mut got);
        assert_eq!(got[0], usize::MAX);
        assert_eq!(got[1..], expected);
        assert_eq!(
            view.rows_containing_prefiltered(b"", &frequencies),
            expected
        );
    }
}

/// A non-empty pattern whose cover came out empty proves no row matches, and
/// must not be confused with the all-rows answer.
#[test]
fn empty_probe_cover_still_appends_nothing() {
    let analysis = super::PrefilterAnalysis {
        probe_cover: ProbeCover::new(Vec::new(), Vec::new()),
        covered_frequency: 0,
        total_frequency: 1,
        scan_ns: 0.0,
        walk: super::scan::Walk::default(),
        matches_all: false,
    };
    assert_eq!(analysis.expected_candidate_row_fraction(3), 0.0);
    assert!(prefilter_is_likely_profitable(&analysis, 3));

    let dict = crate::compress(b"a", &[0u32, 1], DEFAULT_CONFIG).unwrap();
    let mut rows = vec![usize::MAX];
    prefilter_candidates(
        &[0],
        &[0u32, 0, 1, 1],
        dict.view().dict,
        &analysis,
        &mut rows,
    );
    assert_eq!(rows, vec![usize::MAX]);
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

/// The same overlap through a trained dictionary and the whole scan: the
/// walk's result is checked against byte containment whenever it ran.
#[test]
fn overlapping_occurrences_through_the_scan() {
    let rows: &[&[u8]] = &[
        b"appappapple",
        b"appapple pie",
        b"an appappappapple",
        b"appappaple",
        b"apple",
        b"aabaaabaa",
        b"aaaa",
        b"abababab",
        b"ababx abab",
    ];
    check(
        rows,
        &[b"appapple", b"aaa", b"abab", b"ababab", b"apple", b"papp"],
    );
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
    let analysis = analyze_prefilter(&pat, view.dict, &frequencies, view.num_rows());
    prefilter_candidates(
        view.codes,
        view.row_offsets,
        view.dict,
        &analysis,
        &mut candidates,
    );

    assert_eq!(candidates, vec![0]);
}

#[test]
fn analysis_reports_normalized_cover_frequency() {
    let col = compress_rows(&[b"alpha", b"beta"]);
    let view = col.view();
    let frequencies = build_token_frequency_index(view.codes, view.dict.num_tokens()).unwrap();
    let analysis = analyze_prefilter(b"a", view.dict, &frequencies, view.num_rows());
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

#[test]
fn external_storage_matches_owned_prefilter_analysis_and_results() {
    let col = compress_rows(&[b"alpha beta", b"beta gamma", b"alphabet soup", b"delta"]);
    let view = col.view();
    let owned = build_token_frequency_index(view.codes, view.dict.num_tokens()).unwrap();
    let cumulative = owned.storage().cumulative().to_vec();
    let external = TokenFrequencyIndex::validate(
        BorrowedFrequencies(&cumulative),
        view.codes,
        view.dict.num_tokens(),
    )
    .unwrap();

    let pattern = b"alpha";
    let owned_analysis = analyze_prefilter(pattern, view.dict, &owned, view.num_rows());
    let external_analysis = analyze_prefilter(pattern, view.dict, &external, view.num_rows());
    assert_eq!(
        external_analysis.probe_cover().points(),
        owned_analysis.probe_cover().points()
    );
    assert_eq!(
        external_analysis.probe_cover().ranges(),
        owned_analysis.probe_cover().ranges()
    );
    assert_eq!(
        external_analysis.covered_frequency(),
        owned_analysis.covered_frequency()
    );
    assert_eq!(
        candidates(view, view.dict, &external, pattern),
        candidates(view, view.dict, &owned, pattern)
    );
    assert_eq!(
        view.rows_containing_prefiltered(pattern, &external),
        view.rows_containing_prefiltered(pattern, &owned)
    );
}

/// A cut hands over overlapping probe runs — a range and a point naming the
/// same token, two ranges that abut, the mandatory contained tokens unioned on
/// top, in no order — so the cover merges them into maximal runs. Anything less
/// is a redundant comparison paid on every vector of the code stream.
#[test]
fn cover_probes_the_maximal_runs_of_its_membership() {
    let run = |begin, last| TokenRange { begin, last };
    let pf = ProbeCover::from_runs(vec![
        run(7, 8),
        run(0, 0),
        run(3, 3),
        run(6, 7),
        run(1, 1),
        run(8, 8),
    ]);

    assert_eq!(pf.points, vec![3]);
    assert_eq!(
        pf.ranges,
        vec![
            TokenRange { begin: 0, last: 1 },
            TokenRange { begin: 6, last: 8 },
        ]
    );
    let members = [true, true, false, true, false, false, true, true, true];
    for (code, &member) in members.iter().enumerate() {
        assert_eq!(pf.contains(code as Token), member, "membership at {code}");
    }
    assert!(ProbeCover::from_runs(Vec::new()).is_empty());
}

/// Even a safety-valid index that falsely reports every actually used token as
/// absent may only influence planning cost; its zeroes cannot delete selected
/// members and make the probe cover miss a real match.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[test]
fn false_zero_frequencies_cannot_hide_a_true_match() {
    let rows: &[&[u8]] = &[b"alpha beta", b"gamma", b"alphabet soup", b"delta"];
    let col = compress_rows(rows);
    let view = col.view();
    assert!(view.codes.iter().all(|&code| code != 0));

    // Attribute the entire code count to token 0, which this ASCII column never
    // uses. Every token that actually occurs therefore has a false zero.
    let mut cumulative = vec![view.codes.len() as u32; view.dict.num_tokens() + 1];
    cumulative[0] = 0;
    let frequencies = TokenFrequencyIndex::validate_safety(
        BorrowedFrequencies(&cumulative),
        view.dict.num_tokens(),
        view.codes.len(),
    )
    .unwrap();
    let pattern = b"alpha";
    let analysis = analyze_prefilter(pattern, view.dict, &frequencies, view.num_rows());
    assert!(!analysis.probe_cover().is_empty());
    assert_eq!(analysis.covered_frequency(), 0);

    let expected: Vec<usize> = rows
        .iter()
        .enumerate()
        .filter_map(|(row, bytes)| byte_contains(bytes, pattern).then_some(row))
        .collect();
    let got = candidates(view, view.dict, &frequencies, pattern);
    assert!(expected.iter().all(|row| got.contains(row)));
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[test]
fn wide_probe_cover_dispatches_soundly() {
    let pf = ProbeCover {
        points: vec![0; 33],
        ranges: Vec::new(),
    };
    let mut candidates = Vec::new();

    super::scan::scan(&[0u16], &[0u32, 1], &pf, 1, &mut candidates);
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

    let pf = ProbeCover::from_runs(vec![TokenRange { begin: 1, last: 2 }]);
    let mut out = Vec::new();
    super::scan::scan(&codes, &row_offsets, &pf, 43, &mut out);
    assert_eq!(out, vec![1, 4]);

    let mut oracle = Vec::new();
    super::scan::scan_scalar(&codes, &row_offsets, &pf, &mut oracle);
    assert_eq!(out, oracle);
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

/// A cover of `points` singleton tokens, ids spaced so none merge.
fn cover_of(points: usize) -> ProbeCover {
    ProbeCover::from_runs(
        (0..points as Token)
            .map(|id| TokenRange {
                begin: 2 * id,
                last: 2 * id,
            })
            .collect(),
    )
}

/// The two things a cut can trade: more probes cost no less at equal
/// coverage, and more coverage costs more at equal shape.
#[test]
fn scan_cost_is_monotone_in_probes_and_coverage() {
    let region = Region {
        code_count: 1 << 24,
        row_count: 1 << 18,
    };
    let costs: Vec<f64> = [1, 3, 8, 16, 24]
        .into_iter()
        .map(|points| scan_ns(&cover_of(points), 1 << 12, region))
        .collect();
    assert!(
        costs.windows(2).all(|pair| pair[0] <= pair[1]),
        "wider covers cost less: {costs:?}"
    );
    let one = cover_of(1);
    let sparse = scan_ns(&one, 1 << 10, region);
    let dense = scan_ns(&one, 1 << 20, region);
    assert!(sparse < dense, "{sparse} for 2^10 hits, {dense} for 2^20");
    assert_eq!(scan_ns(&cover_of(0), 0, region), 0.0);
}

/// The sweep never does worse than the frequency-only cut it starts from,
/// and what it picks is priced as it reports.
#[test]
fn sweep_prices_at_or_below_the_frequency_cut() {
    use crate::test_corpus::user_strings;
    let corpus: Vec<Vec<u8>> = user_strings(200)
        .into_iter()
        .map(String::into_bytes)
        .collect();
    let rows: Vec<&[u8]> = corpus.iter().map(Vec::as_slice).collect();
    let col = compress_rows(&rows);
    let view = col.view();
    let frequencies = build_token_frequency_index(view.codes, view.dict.num_tokens()).unwrap();
    let freq = frequencies.as_view();
    let region = Region {
        code_count: view.codes.len(),
        row_count: view.num_rows(),
    };
    for pat in [
        b"e".as_slice(),
        b"://",
        b".com/page",
        b"https://www.example.com",
    ] {
        let graph = build_alignment_graph(view.dict, pat, freq);
        let by_frequency = min_cut(&graph.edges, graph.nodes, |edge| {
            u64::from(edge.frequency())
        });
        let baseline = ProbeCover::from_edge_cut(&by_frequency);
        let baseline_ns = scan_ns(&baseline, cover_frequency(&baseline, freq), region);

        let (cover, covered, ns) = cheapest_cover(&graph, freq, region);
        assert_eq!(covered, cover_frequency(&cover, freq));
        assert_eq!(ns, scan_ns(&cover, covered, region));
        assert!(
            ns <= baseline_ns,
            "{pat:?}: sweep {ns} against {baseline_ns}"
        );
    }
}

/// The split scan verifies its hits, so on the covers it takes the rows come
/// out exact and the caller has nothing left to check.
#[cfg(target_arch = "aarch64")]
#[test]
fn split_scan_returns_exact_rows() {
    use crate::test_corpus::user_strings;
    let corpus: Vec<Vec<u8>> = user_strings(200)
        .into_iter()
        .map(String::into_bytes)
        .collect();
    let rows: Vec<&[u8]> = corpus.iter().map(Vec::as_slice).collect();
    let col = compress_rows(&rows);
    let view = col.view();
    let frequencies = build_token_frequency_index(view.codes, view.dict.num_tokens()).unwrap();
    for pat in [b"example".as_slice(), b".com", b"://", b"page", b"user"] {
        let analysis = analyze_prefilter(pat, view.dict, &frequencies, view.num_rows());
        let mut got = Vec::new();
        prefilter_candidates(view.codes, view.row_offsets, view.dict, &analysis, &mut got);
        let want: Vec<usize> = (0..view.num_rows())
            .filter(|&k| byte_contains(&decode_row(view, k), pat))
            .collect();
        assert_eq!(got, want, "{pat:?}");
    }
}
