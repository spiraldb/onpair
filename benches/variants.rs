// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//! Head-to-head of the compressed-domain `contains` walk variants, in the
//! study's protocol: ns per candidate row on the **verify** path (rows the
//! prefilter could not rule out), plus whole-column unfiltered scans.
//!
//! Variants:
//!   * `dense_lazy`  — the shipped `TokenDfa` (lazy `(state, code)` table);
//!   * `sparse`      — base + per-state exception lists (D²FA-style);
//!   * `class`       — token→class `u8` map + `C × S` table, one lookup/code;
//!   * `class_pair`  — `C² × S` table, one state-dependent lookup per TWO codes;
//!   * `hyperflex`   — state in a SIMD lane, one `VPERMB` per code (AVX-512 VBMI);
//!   * `hyperflex2`  — `VPERMB` over class-pair rows, one shuffle per two codes;
//!   * `decode_memmem` — decode the candidate row's bytes, `memchr::memmem`.
//!
//! Dataset: ClickBench URL from `ONPAIR_BENCH_PARQUET_DIR` (default
//! `/tmp/clickbench`), first parquet file only, one dictionary. Correctness
//! is cross-checked between all variants before timing.
//!
//! Run with: cargo bench --bench variants
#![allow(
    clippy::cast_possible_truncation,
    clippy::expect_used,
    clippy::missing_panics_doc,
    clippy::unwrap_used
)]

use std::env;
use std::fs::File;
use std::path::PathBuf;
use std::sync::OnceLock;

use arrow_array::cast::AsArray;
use divan::Bencher;
use divan::counter::ItemsCount;
use memchr::memmem;
use onpair::Bits;
use onpair::Column;
use onpair::Config;
use onpair::Threshold;
use onpair::compress;
use onpair::query::ClassSearcher;
use onpair::query::ContainsSearcher;
use onpair::query::SparseSearcher;
use onpair::query::Walk;

/// URL patterns spanning the candidate-density spectrum (the study's gain
/// scales with density: >40% wins 1.25x, <10% loses).
const PATTERNS: &[&str] = &[
    "https",         // ~dense: nearly every row
    ".ru/",          // heavy
    "kinopoisk.ru",  // ~2%
    "yandsearch",    // ~5-10%
    "avtomobil",     // ~0.02%
    "google",        // ~0.007%
    "no-such-xq",    // never
];

struct Corpus {
    col: Column<u64>,
    rows: usize,
}

fn corpus() -> &'static Corpus {
    static C: OnceLock<Corpus> = OnceLock::new();
    C.get_or_init(|| {
        let (bytes, offsets) = load_url();
        let cfg = Config {
            bits: Bits::new(16).unwrap(),
            threshold: Threshold::new(0.5).unwrap(),
            seed: Some(42),
        };
        let col = compress(&bytes, &offsets, cfg).unwrap();
        let rows = offsets.len() - 1;
        eprintln!(
            "[variants bench] {rows} rows, {:.1} MiB raw, {:.1} MiB codes, {} tokens",
            bytes.len() as f64 / (1 << 20) as f64,
            (col.codes.len() * 2) as f64 / (1 << 20) as f64,
            col.dict_offsets.len() - 1,
        );
        Corpus { col, rows }
    })
}

fn load_url() -> (Vec<u8>, Vec<u64>) {
    let dir = PathBuf::from(
        env::var("ONPAIR_BENCH_PARQUET_DIR").unwrap_or_else(|_| "/tmp/clickbench".into()),
    );
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map(|rd| {
            rd.filter_map(|e| {
                let p = e.ok()?.path();
                (p.extension().is_some_and(|x| x == "parquet")).then_some(p)
            })
            .collect()
        })
        .unwrap_or_default();
    paths.sort();
    let path = paths.first().expect("no parquet file in ONPAIR_BENCH_PARQUET_DIR");

    use parquet::arrow::ProjectionMask;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    let file = File::open(path).expect("open parquet");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("parquet reader");
    let idx = builder
        .parquet_schema()
        .columns()
        .iter()
        .position(|c| c.name().eq_ignore_ascii_case("URL"))
        .expect("no URL column");
    let mask = ProjectionMask::leaves(builder.parquet_schema(), [idx]);
    let reader = builder.with_projection(mask).build().expect("parquet read");
    let mut bytes = Vec::new();
    let mut offsets = vec![0u64];
    for batch in reader {
        let batch = batch.expect("parquet batch");
        for s in batch.column(0).as_string::<i32>().iter() {
            bytes.extend_from_slice(s.unwrap_or("").as_bytes());
            offsets.push(bytes.len() as u64);
        }
    }
    (bytes, offsets)
}

/// Everything one pattern needs, compiled once and cross-checked.
struct Prepared {
    dense: ContainsSearcher,
    sparse: SparseSearcher,
    class: ClassSearcher,
    /// Candidate row spans `(a, b)` from the dense searcher's prefilter —
    /// the verify-path work list. Falls back to every row when the pattern
    /// has no selective anchor.
    cand_spans: Vec<(u32, u32)>,
    /// Decoded bytes of each candidate row (for the decode_memmem baseline
    /// the buffer is rebuilt per call; this is just for sizing).
    max_row_bytes: usize,
    matches: usize,
}

fn prepared(pattern: &str) -> &'static Prepared {
    static CACHE: OnceLock<
        std::sync::Mutex<std::collections::HashMap<String, &'static Prepared>>,
    > = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    let mut guard = cache.lock().unwrap();
    guard.entry(pattern.to_string()).or_insert_with(|| {
        let c = corpus();
        let p = pattern.as_bytes();
        let parts = c.col.as_parts();
        let dense = ContainsSearcher::compile(parts, p);
        let sparse = SparseSearcher::compile_dict(&c.col.dict_bytes, &c.col.dict_offsets, p);
        let class = ClassSearcher::compile_dict(&c.col.dict_bytes, &c.col.dict_offsets, p);

        // Cross-check every variant on full scans before timing anything.
        let expect = dense.matching_rows(&c.col.codes, &c.col.code_offsets);
        assert_eq!(
            sparse.matching_rows(&c.col.codes, &c.col.code_offsets),
            expect,
            "sparse disagrees: {pattern}"
        );
        for w in [Walk::Class, Walk::Pair, Walk::Hyperflex, Walk::HyperflexPair] {
            if class.supports(w) {
                assert_eq!(
                    class.matching_rows(&c.col.codes, &c.col.code_offsets, w),
                    expect,
                    "class {w:?} disagrees: {pattern}"
                );
            }
        }

        let cand = dense.candidate_rows(&c.col.codes, &c.col.code_offsets);
        let cand_spans: Vec<(u32, u32)> = cand
            .iter()
            .map(|&r| {
                let a = c.col.code_offsets[r as usize];
                let b = c.col.code_offsets[r as usize + 1];
                (a as u32, b as u32)
            })
            .collect();
        let max_row_bytes = cand_spans
            .iter()
            .map(|&(a, b)| {
                c.col.codes[a as usize..b as usize]
                    .iter()
                    .map(|&t| {
                        (c.col.dict_offsets[t as usize + 1] - c.col.dict_offsets[t as usize])
                            as usize
                    })
                    .sum()
            })
            .max()
            .unwrap_or(0);
        let info = class.info();
        eprintln!(
            "[variants bench] '{pattern}': {} matches, {} candidate rows ({:.2}% of {}), \
             classes {:?} states {:?} trans2 {:?} B",
            expect.len(),
            cand_spans.len(),
            100.0 * cand_spans.len() as f64 / c.rows as f64,
            c.rows,
            info.map(|i| i.nclasses),
            info.map(|i| i.nstates),
            info.map(|i| i.trans2_bytes),
        );
        Box::leak(Box::new(Prepared {
            dense,
            sparse,
            class,
            cand_spans,
            max_row_bytes,
            matches: expect.len(),
        }))
    })
}

// ────────────────────────────────────────────────────────────────────────────
// Verify path: walk exactly the candidate rows. Counter = candidate rows, so
// divan's /iter readings divided by items give ns/row directly.
// ────────────────────────────────────────────────────────────────────────────

fn bench_verify(bencher: Bencher, pattern: &str, f: impl Fn(&Prepared, &[u16], (u32, u32)) -> bool + Sync) {
    let c = corpus();
    let p = prepared(pattern);
    if p.cand_spans.is_empty() {
        return; // nothing to verify (divan reports no samples)
    }
    bencher
        .counter(ItemsCount::new(p.cand_spans.len()))
        .bench(|| {
            let mut n = 0usize;
            for &(a, b) in &p.cand_spans {
                let row = &c.col.codes[a as usize..b as usize];
                n += f(p, divan::black_box(row), (a, b)) as usize;
            }
            assert_eq!(n, p.matches, "verify walk missed matches");
            n
        });
}

#[divan::bench(args = PATTERNS, sample_count = 30)]
fn verify_dense_lazy(bencher: Bencher, pattern: &str) {
    bench_verify(bencher, pattern, |p, row, _| p.dense.row_matches(row));
}

#[divan::bench(args = PATTERNS, sample_count = 30)]
fn verify_sparse(bencher: Bencher, pattern: &str) {
    bench_verify(bencher, pattern, |p, row, _| p.sparse.row_matches(row));
}

#[divan::bench(args = PATTERNS, sample_count = 30)]
fn verify_class(bencher: Bencher, pattern: &str) {
    bench_verify(bencher, pattern, |p, row, _| p.class.row_matches(row, Walk::Class));
}

#[divan::bench(args = PATTERNS, sample_count = 30)]
fn verify_class_pair(bencher: Bencher, pattern: &str) {
    bench_verify(bencher, pattern, |p, row, _| p.class.row_matches(row, Walk::Pair));
}

#[divan::bench(args = PATTERNS, sample_count = 30)]
fn verify_hyperflex(bencher: Bencher, pattern: &str) {
    if !prepared(pattern).class.supports(Walk::Hyperflex) {
        return;
    }
    bench_verify(bencher, pattern, |p, row, _| {
        p.class.row_matches(row, Walk::Hyperflex)
    });
}

#[divan::bench(args = PATTERNS, sample_count = 30)]
fn verify_hyperflex2(bencher: Bencher, pattern: &str) {
    if !prepared(pattern).class.supports(Walk::HyperflexPair) {
        return;
    }
    bench_verify(bencher, pattern, |p, row, _| {
        p.class.row_matches(row, Walk::HyperflexPair)
    });
}

/// Batched verify: all candidate spans in one call, four interleaved state
/// chains hiding each other's load/shuffle latency.
fn bench_verify_ilv4(bencher: Bencher, pattern: &str, walk: Walk) {
    let c = corpus();
    let p = prepared(pattern);
    if p.cand_spans.is_empty() || !p.class.supports(walk) {
        return;
    }
    bencher
        .counter(ItemsCount::new(p.cand_spans.len()))
        .bench(|| {
            let flags =
                p.class
                    .matching_spans_ilv4(divan::black_box(&c.col.codes), &p.cand_spans, walk);
            let n = flags.iter().filter(|&&f| f).count();
            assert_eq!(n, p.matches, "ilv4 verify missed matches");
            n
        });
}

#[divan::bench(args = PATTERNS, sample_count = 30)]
fn verify_class_pair_ilv4(bencher: Bencher, pattern: &str) {
    bench_verify_ilv4(bencher, pattern, Walk::Pair);
}

#[divan::bench(args = PATTERNS, sample_count = 30)]
fn verify_hyperflex2_ilv4(bencher: Bencher, pattern: &str) {
    bench_verify_ilv4(bencher, pattern, Walk::HyperflexPair);
}

/// Whole-column interleaved scans: every row span, four lanes at a time.
fn bench_scan_ilv4(bencher: Bencher, pattern: &str, walk: Walk) {
    let (c, p) = (corpus(), prepared(pattern));
    if !p.class.supports(walk) {
        return;
    }
    let spans: Vec<(u32, u32)> = c
        .col
        .code_offsets
        .windows(2)
        .map(|w| (w[0] as u32, w[1] as u32))
        .collect();
    bencher.counter(ItemsCount::new(c.rows)).bench(|| {
        p.class
            .matching_spans_ilv4(divan::black_box(&c.col.codes), &spans, walk)
            .iter()
            .filter(|&&f| f)
            .count()
    });
}

#[divan::bench(args = PATTERNS, sample_count = 10)]
fn scan_class_pair_ilv4(bencher: Bencher, pattern: &str) {
    bench_scan_ilv4(bencher, pattern, Walk::Pair);
}

#[divan::bench(args = PATTERNS, sample_count = 10)]
fn scan_hyperflex2_ilv4(bencher: Bencher, pattern: &str) {
    bench_scan_ilv4(bencher, pattern, Walk::HyperflexPair);
}

/// Decode the candidate row and `memmem` it — what a system that cannot
/// verify in the compressed domain pays per candidate.
#[divan::bench(args = PATTERNS, sample_count = 30)]
fn verify_decode_memmem(bencher: Bencher, pattern: &str) {
    let c = corpus();
    let p = prepared(pattern);
    if p.cand_spans.is_empty() {
        return;
    }
    let finder = memmem::Finder::new(pattern.as_bytes());
    let dict_bytes = &c.col.dict_bytes;
    let dict_offsets = &c.col.dict_offsets;
    let mut buf: Vec<u8> = Vec::with_capacity(p.max_row_bytes);
    bencher
        .counter(ItemsCount::new(p.cand_spans.len()))
        .bench_local(move || {
            let mut n = 0usize;
            for &(a, b) in &p.cand_spans {
                buf.clear();
                for &t in divan::black_box(&c.col.codes[a as usize..b as usize]) {
                    let lo = dict_offsets[t as usize] as usize;
                    let hi = dict_offsets[t as usize + 1] as usize;
                    buf.extend_from_slice(&dict_bytes[lo..hi]);
                }
                n += finder.find(&buf).is_some() as usize;
            }
            assert_eq!(n, p.matches, "decode verify missed matches");
            n
        });
}

// ────────────────────────────────────────────────────────────────────────────
// Whole-column scans, unfiltered (every row walked) and the full prefiltered
// pipelines. Counter = total rows.
// ────────────────────────────────────────────────────────────────────────────

fn bench_scan(bencher: Bencher, f: impl Fn() -> usize + Sync) {
    let c = corpus();
    bencher.counter(ItemsCount::new(c.rows)).bench(f);
}

#[divan::bench(args = PATTERNS, sample_count = 10)]
fn scan_dense_lazy(bencher: Bencher, pattern: &str) {
    let (c, p) = (corpus(), prepared(pattern));
    bench_scan(bencher, || {
        p.dense
            .matching_rows_unfiltered(divan::black_box(&c.col.codes), &c.col.code_offsets)
            .len()
    });
}

#[divan::bench(args = PATTERNS, sample_count = 10)]
fn scan_sparse(bencher: Bencher, pattern: &str) {
    let (c, p) = (corpus(), prepared(pattern));
    bench_scan(bencher, || {
        p.sparse
            .matching_rows_unfiltered(divan::black_box(&c.col.codes), &c.col.code_offsets)
            .len()
    });
}

#[divan::bench(args = PATTERNS, sample_count = 10)]
fn scan_class(bencher: Bencher, pattern: &str) {
    let (c, p) = (corpus(), prepared(pattern));
    bench_scan(bencher, || {
        p.class
            .matching_rows_unfiltered(divan::black_box(&c.col.codes), &c.col.code_offsets, Walk::Class)
            .len()
    });
}

#[divan::bench(args = PATTERNS, sample_count = 10)]
fn scan_class_pair(bencher: Bencher, pattern: &str) {
    let (c, p) = (corpus(), prepared(pattern));
    bench_scan(bencher, || {
        p.class
            .matching_rows_unfiltered(divan::black_box(&c.col.codes), &c.col.code_offsets, Walk::Pair)
            .len()
    });
}

#[divan::bench(args = PATTERNS, sample_count = 10)]
fn scan_hyperflex(bencher: Bencher, pattern: &str) {
    let (c, p) = (corpus(), prepared(pattern));
    if !p.class.supports(Walk::Hyperflex) {
        return;
    }
    bench_scan(bencher, || {
        p.class
            .matching_rows_unfiltered(divan::black_box(&c.col.codes), &c.col.code_offsets, Walk::Hyperflex)
            .len()
    });
}

#[divan::bench(args = PATTERNS, sample_count = 10)]
fn scan_hyperflex2(bencher: Bencher, pattern: &str) {
    let (c, p) = (corpus(), prepared(pattern));
    if !p.class.supports(Walk::HyperflexPair) {
        return;
    }
    bench_scan(bencher, || {
        p.class
            .matching_rows_unfiltered(divan::black_box(&c.col.codes), &c.col.code_offsets, Walk::HyperflexPair)
            .len()
    });
}

/// Full pipelines: prefilter + verify, end to end.
#[divan::bench(args = PATTERNS, sample_count = 10)]
fn pf_dense(bencher: Bencher, pattern: &str) {
    let (c, p) = (corpus(), prepared(pattern));
    bench_scan(bencher, || {
        p.dense
            .matching_rows(divan::black_box(&c.col.codes), &c.col.code_offsets)
            .len()
    });
}

/// Anchor prefilter + hyperflex2 verify: the composition of the shipped
/// prefilter with the fastest verify walk.
#[divan::bench(args = PATTERNS, sample_count = 10)]
fn pf_hyperflex2(bencher: Bencher, pattern: &str) {
    let (c, p) = (corpus(), prepared(pattern));
    if !p.class.supports(Walk::HyperflexPair) {
        return;
    }
    bench_scan(bencher, || {
        p.dense
            .matching_rows_with(divan::black_box(&c.col.codes), &c.col.code_offsets, |_, row| {
                p.class.row_matches(row, Walk::HyperflexPair)
            })
            .len()
    });
}

#[divan::bench(args = PATTERNS, sample_count = 10)]
fn pf_sparse_combo(bencher: Bencher, pattern: &str) {
    let (c, p) = (corpus(), prepared(pattern));
    bench_scan(bencher, || {
        p.sparse
            .matching_rows(divan::black_box(&c.col.codes), &c.col.code_offsets)
            .len()
    });
}

#[divan::bench(args = PATTERNS, sample_count = 10)]
fn pf_class_pair_combo(bencher: Bencher, pattern: &str) {
    let (c, p) = (corpus(), prepared(pattern));
    bench_scan(bencher, || {
        p.class
            .matching_rows(divan::black_box(&c.col.codes), &c.col.code_offsets, Walk::Pair)
            .len()
    });
}

/// The fully-merged dictionary-only pipeline: the sparse-analysis
/// interesting-run combination prefilter feeding the Hyperflex class-pair
/// walk — every piece of the study composed, no code-stream sampling at
/// compile time.
#[divan::bench(args = PATTERNS, sample_count = 10)]
fn pf_combo_hyperflex2(bencher: Bencher, pattern: &str) {
    let (c, p) = (corpus(), prepared(pattern));
    if !p.class.supports(Walk::HyperflexPair) {
        return;
    }
    bench_scan(bencher, || {
        p.class
            .matching_rows(
                divan::black_box(&c.col.codes),
                &c.col.code_offsets,
                Walk::HyperflexPair,
            )
            .len()
    });
}

fn main() {
    let _ = corpus();
    for p in PATTERNS {
        let _ = prepared(p);
    }
    divan::main();
}
