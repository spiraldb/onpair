// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//! Compressed-domain `contains` over a real ClickBench URL column.
//!
//! Compares, per query, the three ways to answer `URL LIKE '%pattern%'` over
//! an OnPair-compressed column:
//!
//!   * `decompress_memmem` — decompress every row and run `memchr::memmem`
//!     (what a system without compressed-domain search must do);
//!   * `dfa` — the token DFA over codes, no decompression, every row scanned;
//!   * `prefilter_dfa` — SIMD candidate pass over the raw code stream, DFA
//!     only on rows that might match (the production path);
//!   * `prefilter_only` — the candidate pass alone, to expose its raw speed
//!     and pass rate.
//!
//! Data: every `*.parquet` under `ONPAIR_BENCH_PARQUET_DIR` (default
//! `/tmp/clickbench`, e.g. the ClickBench partitioned `hits_*.parquet`),
//! URL column, one OnPair column per file (per-file dictionaries, like row
//! groups). Falls back to a synthetic URL corpus when the directory is
//! missing so the bench always runs.
//!
//! Run with: cargo bench --bench contains
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
use memchr::memmem;
use onpair::Bits;
use onpair::Column;
use onpair::Config;
use onpair::Threshold;
use onpair::compress;
use onpair::decompress;
use onpair::query::ContainsSearcher;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

/// Dictionary code width under test. 16 gives the largest dictionary (64 Ki
/// tokens) — the hardest case for the DFA table and the most realistic for a
/// URL column.
const BITS: u8 = 16;

/// `LIKE '%…%'` patterns. Selectivities measured on the real ClickBench URL
/// column (10 partitions, 10 M rows); mostly low-selectivity, the usual shape
/// of a `contains` predicate.
const QUERIES: &[&str] = &[
    "kinopoisk.ru",      // moderate: 2.05% of rows
    "avtomobil",         // low: 0.0175%
    "google",            // low: 0.0065%
    "no-such-string-xq", // never matches
];

// ─────────────────────────────────────────────────────────────────────────────
// Corpus loading & compression (once, shared by every bench).
// ─────────────────────────────────────────────────────────────────────────────

struct CompressedFile {
    col: Column<u64>,
    rows: usize,
}

struct Corpus {
    files: Vec<CompressedFile>,
    total_bytes: usize,
}

fn corpus() -> &'static Corpus {
    static CORPUS: OnceLock<Corpus> = OnceLock::new();
    CORPUS.get_or_init(|| {
        let files = load_files();
        let cfg = Config {
            bits: Bits::new(BITS).unwrap(),
            threshold: Threshold::new(0.5).unwrap(),
            seed: Some(42),
        };
        let mut out = Vec::new();
        let mut total_bytes = 0usize;
        let mut compressed_bytes = 0usize;
        for (name, bytes, offsets) in files {
            let col = compress(&bytes, &offsets, cfg).unwrap();
            compressed_bytes += col.codes.len() * 2 + col.dict_bytes.len();
            eprintln!(
                "[contains bench] {name}: {} rows, {:.1} MiB raw",
                offsets.len() - 1,
                bytes.len() as f64 / (1024.0 * 1024.0),
            );
            total_bytes += bytes.len();
            out.push(CompressedFile {
                col,
                rows: offsets.len() - 1,
            });
        }
        eprintln!(
            "[contains bench] total {:.1} MiB raw, {:.1} MiB compressed (codes+dict), {} files",
            total_bytes as f64 / (1024.0 * 1024.0),
            compressed_bytes as f64 / (1024.0 * 1024.0),
            out.len()
        );
        let c = Corpus {
            files: out,
            total_bytes,
        };
        validate_and_report(&c);
        c
    })
}

/// Load `(name, packed bytes, offsets)` per parquet file, or one synthetic
/// pseudo-file as a fallback.
fn load_files() -> Vec<(String, Vec<u8>, Vec<u64>)> {
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
    if paths.is_empty() {
        eprintln!(
            "[contains bench] {} has no parquet files; synthetic corpus",
            dir.display()
        );
        let (bytes, offsets) = synthetic_urls(1_000_000);
        return vec![("synthetic".into(), bytes, offsets)];
    }
    paths
        .iter()
        .map(|p| {
            let (bytes, offsets) = read_url_column(p);
            (
                p.file_name().unwrap().to_string_lossy().into_owned(),
                bytes,
                offsets,
            )
        })
        .collect()
}

/// Read the URL column of one parquet file, packed as `(bytes, offsets)`.
fn read_url_column(path: &PathBuf) -> (Vec<u8>, Vec<u64>) {
    let file = File::open(path).expect("open parquet");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("parquet reader");
    let idx = builder
        .parquet_schema()
        .columns()
        .iter()
        .position(|c| c.name().eq_ignore_ascii_case("url"))
        .expect("no URL column in parquet file");
    let mask = ProjectionMask::leaves(builder.parquet_schema(), [idx]);
    let reader = builder.with_projection(mask).build().expect("parquet read");

    let mut bytes = Vec::new();
    let mut offsets = vec![0u64];
    for batch in reader {
        let batch = batch.expect("parquet batch");
        let col = batch.column(0);
        use arrow_schema::DataType::*;
        match col.data_type() {
            Utf8 => {
                for s in col.as_string::<i32>().iter() {
                    bytes.extend_from_slice(s.unwrap_or("").as_bytes());
                    offsets.push(bytes.len() as u64);
                }
            }
            Binary => {
                for s in col.as_binary::<i32>().iter() {
                    bytes.extend_from_slice(s.unwrap_or(b""));
                    offsets.push(bytes.len() as u64);
                }
            }
            other => panic!("unsupported URL column type {other}"),
        }
    }
    (bytes, offsets)
}

/// Fallback corpus mirroring `benches/clickbench.rs`'s URL shape.
fn synthetic_urls(n: usize) -> (Vec<u8>, Vec<u64>) {
    const HOSTS: &[&str] = &[
        "https://www.yandex.ru",
        "https://www.google.com",
        "https://news.ycombinator.com",
        "https://www.example.com",
        "http://m.yandex.ru",
    ];
    const PATHS: &[&str] = &[
        "/",
        "/page",
        "/search?q=",
        "/profile",
        "/clck/jsredir?from=",
    ];
    let mut bytes = Vec::new();
    let mut offsets = vec![0u64];
    let mut x = 0x9E3779B97F4A7C15u64;
    for _ in 0..n {
        x = x.wrapping_add(0x9E3779B97F4A7C15);
        let h = HOSTS[(x as usize) % HOSTS.len()];
        let p = PATHS[((x >> 16) as usize) % PATHS.len()];
        let t = (x >> 48) as u16;
        bytes.extend_from_slice(format!("{h}{p}{t}").as_bytes());
        offsets.push(bytes.len() as u64);
    }
    (bytes, offsets)
}

// ─────────────────────────────────────────────────────────────────────────────
// The three scan strategies.
// ─────────────────────────────────────────────────────────────────────────────

/// Decompress-then-search baseline: materialize the column, recover row byte
/// boundaries from the code stream, `memmem` each row.
fn memmem_rows(file: &CompressedFile, finder: &memmem::Finder<'_>) -> Vec<u64> {
    let parts = file.col.as_parts();
    let bytes = decompress(parts);
    let dict_offsets = parts.dict_offsets;
    let mut out = Vec::new();
    let mut pos = 0usize;
    for (r, w) in file.col.code_offsets.windows(2).enumerate() {
        let row_codes = &parts.codes[w[0] as usize..w[1] as usize];
        let len: usize = row_codes
            .iter()
            .map(|&c| (dict_offsets[c as usize + 1] - dict_offsets[c as usize]) as usize)
            .sum();
        if finder.find(&bytes[pos..pos + len]).is_some() {
            out.push(r as u64);
        }
        pos += len;
    }
    out
}

fn searchers(pattern: &str) -> &'static Vec<ContainsSearcher> {
    static CACHE: OnceLock<
        std::sync::Mutex<std::collections::HashMap<String, &'static Vec<ContainsSearcher>>>,
    > = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    let mut guard = cache.lock().unwrap();
    guard.entry(pattern.to_string()).or_insert_with(|| {
        let s: Vec<ContainsSearcher> = corpus()
            .files
            .iter()
            .map(|f| ContainsSearcher::compile(f.col.as_parts(), pattern.as_bytes()))
            .collect();
        Box::leak(Box::new(s))
    })
}

/// Cross-check the three strategies on every file and print per-query stats.
fn validate_and_report(c: &Corpus) {
    let rows: usize = c.files.iter().map(|f| f.rows).sum();
    for &q in QUERIES {
        let finder = memmem::Finder::new(q.as_bytes());
        let mut matches = 0usize;
        let mut candidates = 0usize;
        let mut info: Option<(&str, f64)> = None;
        for f in &c.files {
            let s = ContainsSearcher::compile(f.col.as_parts(), q.as_bytes());
            let truth = memmem_rows(f, &finder);
            let dfa = s.matching_rows_unfiltered(&f.col.codes, &f.col.code_offsets);
            let pf = s.matching_rows(&f.col.codes, &f.col.code_offsets);
            assert_eq!(dfa, truth, "dfa mismatch: query={q}");
            assert_eq!(pf, truth, "prefiltered mismatch: query={q}");
            matches += truth.len();
            candidates += s.candidate_rows(&f.col.codes, &f.col.code_offsets).len();
            info = info.or_else(|| s.prefilter_info());
        }
        let info = info.map_or_else(
            || "none (dfa only)".to_string(),
            |(strategy, rate)| format!("{strategy}, expected per-code hit rate {rate:.5}"),
        );
        eprintln!(
            "[contains bench] query \"{q}\": {matches} matches ({:.4}% of rows), \
             {candidates} prefilter candidates ({:.4}%), prefilter: {info}",
            100.0 * matches as f64 / rows as f64,
            100.0 * candidates as f64 / rows as f64,
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Benches. Counter = raw (uncompressed) bytes covered by the scan, so results
// read as "logical text scanned per second".
// ─────────────────────────────────────────────────────────────────────────────

#[divan::bench(args = QUERIES, sample_count = 3, sample_size = 1)]
fn decompress_memmem(bencher: Bencher, query: &str) {
    let c = corpus();
    let finder = memmem::Finder::new(query.as_bytes());
    bencher
        .counter(divan::counter::BytesCount::new(c.total_bytes))
        .bench(|| {
            let mut n = 0usize;
            for f in &c.files {
                n += memmem_rows(divan::black_box(f), &finder).len();
            }
            n
        });
}

#[divan::bench(args = QUERIES, sample_count = 5, sample_size = 1)]
fn dfa(bencher: Bencher, query: &str) {
    let c = corpus();
    let s = searchers(query);
    bencher
        .counter(divan::counter::BytesCount::new(c.total_bytes))
        .bench(|| {
            let mut n = 0usize;
            for (f, s) in c.files.iter().zip(s) {
                n += s
                    .matching_rows_unfiltered(divan::black_box(&f.col.codes), &f.col.code_offsets)
                    .len();
            }
            n
        });
}

#[divan::bench(args = QUERIES, sample_count = 10, sample_size = 1)]
fn prefilter_dfa(bencher: Bencher, query: &str) {
    let c = corpus();
    let s = searchers(query);
    bencher
        .counter(divan::counter::BytesCount::new(c.total_bytes))
        .bench(|| {
            let mut n = 0usize;
            for (f, s) in c.files.iter().zip(s) {
                n += s
                    .matching_rows(divan::black_box(&f.col.codes), &f.col.code_offsets)
                    .len();
            }
            n
        });
}

#[divan::bench(args = QUERIES, sample_count = 10, sample_size = 1)]
fn prefilter_only(bencher: Bencher, query: &str) {
    let c = corpus();
    let s = searchers(query);
    bencher
        .counter(divan::counter::BytesCount::new(c.total_bytes))
        .bench(|| {
            let mut n = 0usize;
            for (f, s) in c.files.iter().zip(s) {
                n += s
                    .candidate_rows(divan::black_box(&f.col.codes), &f.col.code_offsets)
                    .len();
            }
            n
        });
}

/// One-off query compile cost (per file dictionary), for context.
#[divan::bench(args = QUERIES, sample_count = 5, sample_size = 1)]
fn compile(bencher: Bencher, query: &str) {
    let c = corpus();
    bencher.bench(|| {
        for f in &c.files {
            divan::black_box(ContainsSearcher::compile(
                divan::black_box(f.col.as_parts()),
                query.as_bytes(),
            ));
        }
    });
}

fn main() {
    let _ = corpus();
    divan::main();
}
