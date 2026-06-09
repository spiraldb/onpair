// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//! Compressed-domain `contains` over real string columns.
//!
//! Compares, per `dataset/pattern` arg, the ways to answer
//! `col LIKE '%pattern%'` over an OnPair-compressed column:
//!
//!   * `decompress_memmem` — decompress every row and run `memchr::memmem`
//!     (what a system without compressed-domain search must do);
//!   * `dfa` — the token DFA over codes, no decompression, every row scanned;
//!   * `prefilter_dfa_*` — SIMD candidate pass over the raw code stream, DFA
//!     only on rows that might match, with the prefilter anchor chosen by:
//!       - `sampled`: a sample of the column's code stream (`compile`),
//!       - `stats`: the stored `CodeStats` frequency table
//!         (`compile_with_stats`),
//!       - `heuristic`: no frequency information at all — a dictionary
//!         length prior (`compile_heuristic`);
//!   * `prefilter_only` / `prefilter_only_deferred` — the candidate pass
//!     alone (deferred = anchor re-chosen per scan, `compile_dict_only`);
//!   * `compile*` — one-off query compile cost per mode.
//!
//! Datasets:
//!   * `url`, `title` — ClickBench URL / Title columns from every
//!     `*.parquet` under `ONPAIR_BENCH_PARQUET_DIR` (default
//!     `/tmp/clickbench`, e.g. the partitioned `hits_*.parquet`), one OnPair
//!     column per file (per-file dictionaries, like row groups). Falls back
//!     to a synthetic corpus when the directory is missing so the bench
//!     always runs.
//!   * `tpch` — TPC-H `l_comment` at SF 1 (generated, capped), in 64 MiB
//!     chunks with per-chunk dictionaries.
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
use onpair::query::CodeStats;
use onpair::query::ContainsSearcher;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tpchgen::generators::LineItemGenerator;
use tpchgen_arrow::LineItemArrow;
use tpchgen_arrow::RecordBatchIterator;

/// Dictionary code width under test. 16 gives the largest dictionary (64 Ki
/// tokens) — the hardest case for the DFA table and the most realistic for
/// text-like columns.
const BITS: u8 = 16;

/// `dataset/pattern` pairs. Selectivities (in comments) measured on the real
/// data; mostly low-selectivity, the usual shape of a `contains` predicate.
const QUERIES: &[&str] = &[
    // ClickBench URL, 10 partitions, 10 M rows.
    "url/kinopoisk.ru",      // 2.05%
    "url/avtomobil",         // 0.0175%
    "url/google",            // 0.0065%
    "url/no-such-string-xq", // never matches
    // ClickBench Title (Cyrillic-heavy; patterns are multi-byte UTF-8).
    "title/погода",   // 1.52%
    "title/Сбербанк", // 0.0249%
    "title/skyrim",   // 0.0011%
    // TPC-H l_comment (English text), SF 1.
    "tpch/slyly final",               // 0.83%
    "tpch/carefully ironic accounts", // 0.0338%
    "tpch/zqx-no-match",              // never matches
];

// ─────────────────────────────────────────────────────────────────────────────
// Corpus loading & compression (once, shared by every bench).
// ─────────────────────────────────────────────────────────────────────────────

struct CompressedFile {
    col: Column<u64>,
    /// Stored per-token frequency summary (1 byte/token), captured at
    /// compression time so queries can compile without reading the codes.
    stats: CodeStats,
}

struct Dataset {
    name: &'static str,
    files: Vec<CompressedFile>,
    total_bytes: usize,
    rows: usize,
}

struct Corpus {
    datasets: Vec<Dataset>,
}

fn corpus() -> &'static Corpus {
    static CORPUS: OnceLock<Corpus> = OnceLock::new();
    CORPUS.get_or_init(|| {
        let cfg = Config {
            bits: Bits::new(BITS).unwrap(),
            threshold: Threshold::new(0.5).unwrap(),
            seed: Some(42),
        };
        let mut datasets = Vec::new();
        for (name, files) in load_datasets() {
            let mut out = Vec::new();
            let mut total_bytes = 0usize;
            let mut compressed_bytes = 0usize;
            let mut rows = 0usize;
            for (bytes, offsets) in files {
                let col = compress(&bytes, &offsets, cfg).unwrap();
                compressed_bytes += col.codes.len() * 2 + col.dict_bytes.len();
                total_bytes += bytes.len();
                rows += offsets.len() - 1;
                let stats = CodeStats::from_codes(col.dict_offsets.len() - 1, &col.codes);
                out.push(CompressedFile { col, stats });
            }
            eprintln!(
                "[contains bench] dataset {name}: {rows} rows, {:.1} MiB raw, \
                 {:.1} MiB compressed (codes+dict), {} chunks",
                total_bytes as f64 / (1024.0 * 1024.0),
                compressed_bytes as f64 / (1024.0 * 1024.0),
                out.len()
            );
            datasets.push(Dataset {
                name,
                files: out,
                total_bytes,
                rows,
            });
        }
        let c = Corpus { datasets };
        validate_and_report(&c);
        c
    })
}

fn dataset(name: &str) -> &'static Dataset {
    corpus()
        .datasets
        .iter()
        .find(|d| d.name == name)
        .expect("unknown dataset in query arg")
}

/// Split a `dataset/pattern` bench arg.
fn parse_query(arg: &str) -> (&str, &str) {
    arg.split_once('/')
        .expect("query arg must be dataset/pattern")
}

type RawFiles = Vec<(Vec<u8>, Vec<u64>)>;

/// Load every dataset's raw `(bytes, offsets)` chunks.
fn load_datasets() -> Vec<(&'static str, RawFiles)> {
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

    let (url, title) = if paths.is_empty() {
        eprintln!(
            "[contains bench] {} has no parquet files; synthetic corpus",
            dir.display()
        );
        (
            vec![synthetic_urls(1_000_000)],
            vec![synthetic_urls(200_000)],
        )
    } else {
        (
            paths.iter().map(|p| read_column(p, "URL")).collect(),
            paths.iter().map(|p| read_column(p, "Title")).collect(),
        )
    };
    vec![
        ("url", url),
        ("title", title),
        ("tpch", tpch_l_comment_chunks(64 << 20, 4)),
    ]
}

/// Read one string/binary column of one parquet file, packed as
/// `(bytes, offsets)`.
fn read_column(path: &PathBuf, name: &str) -> (Vec<u8>, Vec<u64>) {
    let file = File::open(path).expect("open parquet");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("parquet reader");
    let idx = builder
        .parquet_schema()
        .columns()
        .iter()
        .position(|c| c.name().eq_ignore_ascii_case(name))
        .expect("no such column in parquet file");
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
            other => panic!("unsupported column type {other}"),
        }
    }
    (bytes, offsets)
}

/// TPC-H `l_comment` at SF 1, split into chunks of `chunk_bytes` (each chunk
/// gets its own dictionary), at most `max_chunks`.
fn tpch_l_comment_chunks(chunk_bytes: usize, max_chunks: usize) -> RawFiles {
    let it = LineItemArrow::new(LineItemGenerator::new(1.0, 1, 1)).with_batch_size(8192 * 8);
    let schema = it.schema().clone();
    let idx = schema.index_of("l_comment").expect("l_comment column");

    let mut chunks = Vec::new();
    let mut bytes = Vec::new();
    let mut offsets = vec![0u64];
    for batch in it {
        for s in batch.column(idx).as_string_view().iter() {
            bytes.extend_from_slice(s.unwrap_or("").as_bytes());
            offsets.push(bytes.len() as u64);
        }
        if bytes.len() >= chunk_bytes {
            chunks.push((
                std::mem::take(&mut bytes),
                std::mem::replace(&mut offsets, vec![0u64]),
            ));
            if chunks.len() == max_chunks {
                return chunks;
            }
        }
    }
    if offsets.len() > 1 {
        chunks.push((bytes, offsets));
    }
    chunks
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
// Scan strategies + per-mode searcher caches.
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

/// Compile one searcher per dataset chunk under the given anchor-selection
/// mode: `sampled` (code-stream sample), `stats` (stored `CodeStats`),
/// `heuristic` (length prior, no frequency info), `deferred` (chosen per
/// scan).
fn compile_mode(mode: &str, f: &CompressedFile, pattern: &[u8]) -> ContainsSearcher {
    match mode {
        "sampled" => ContainsSearcher::compile(f.col.as_parts(), pattern),
        "stats" => ContainsSearcher::compile_with_stats(
            &f.col.dict_bytes,
            &f.col.dict_offsets,
            pattern,
            &f.stats,
        ),
        "heuristic" => {
            ContainsSearcher::compile_heuristic(&f.col.dict_bytes, &f.col.dict_offsets, pattern)
        }
        "deferred" => {
            ContainsSearcher::compile_dict_only(&f.col.dict_bytes, &f.col.dict_offsets, pattern)
        }
        other => panic!("unknown mode {other}"),
    }
}

fn searchers(mode: &'static str, query_arg: &str) -> &'static Vec<ContainsSearcher> {
    static CACHE: OnceLock<
        std::sync::Mutex<std::collections::HashMap<String, &'static Vec<ContainsSearcher>>>,
    > = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    let mut guard = cache.lock().unwrap();
    guard
        .entry(format!("{mode}:{query_arg}"))
        .or_insert_with(|| {
            let (ds, pattern) = parse_query(query_arg);
            let s: Vec<ContainsSearcher> = dataset(ds)
                .files
                .iter()
                .map(|f| compile_mode(mode, f, pattern.as_bytes()))
                .collect();
            Box::leak(Box::new(s))
        })
}

/// Cross-check every mode against ground truth on every chunk and print the
/// per-query mode comparison (candidate-row rates).
fn validate_and_report(c: &Corpus) {
    if let Some(f) = c.datasets.first().and_then(|d| d.files.first()) {
        eprintln!(
            "[contains bench] stored CodeStats: {} B/chunk vs {} B dict bytes ({:.1}%)",
            f.stats.as_bytes().len(),
            f.col.dict_bytes.len(),
            100.0 * f.stats.as_bytes().len() as f64 / f.col.dict_bytes.len() as f64,
        );
    }
    const MODES: &[&str] = &["sampled", "stats", "heuristic", "deferred"];
    for &arg in QUERIES {
        let (ds, pattern) = parse_query(arg);
        let d = c.datasets.iter().find(|d| d.name == ds).expect("dataset");
        let finder = memmem::Finder::new(pattern.as_bytes());
        let mut matches = 0usize;
        let mut candidates = [0usize; 4];
        for f in &d.files {
            let truth = memmem_rows(f, &finder);
            // The unfiltered DFA is mode-independent; check it once.
            let s = compile_mode("sampled", f, pattern.as_bytes());
            assert_eq!(
                s.matching_rows_unfiltered(&f.col.codes, &f.col.code_offsets),
                truth,
                "dfa mismatch: {arg}"
            );
            for (mi, &mode) in MODES.iter().enumerate() {
                let sm = compile_mode(mode, f, pattern.as_bytes());
                assert_eq!(
                    sm.matching_rows(&f.col.codes, &f.col.code_offsets),
                    truth,
                    "{mode} mismatch: {arg}"
                );
                candidates[mi] += sm.candidate_rows(&f.col.codes, &f.col.code_offsets).len();
            }
            matches += truth.len();
        }
        let pct = |n: usize| 100.0 * n as f64 / d.rows as f64;
        eprintln!(
            "[contains bench] {arg}: {matches} matches ({:.4}%) | candidate rows: \
             sampled {:.4}% / stats {:.4}% / heuristic {:.4}% / deferred {:.4}%",
            pct(matches),
            pct(candidates[0]),
            pct(candidates[1]),
            pct(candidates[2]),
            pct(candidates[3]),
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Benches. Counter = raw (uncompressed) bytes covered by the scan, so results
// read as "logical text scanned per second".
// ─────────────────────────────────────────────────────────────────────────────

fn bench_prefilter_dfa(bencher: Bencher, mode: &'static str, query_arg: &str) {
    let d = dataset(parse_query(query_arg).0);
    let s = searchers(mode, query_arg);
    bencher
        .counter(divan::counter::BytesCount::new(d.total_bytes))
        .bench(|| {
            let mut n = 0usize;
            for (f, s) in d.files.iter().zip(s) {
                n += s
                    .matching_rows(divan::black_box(&f.col.codes), &f.col.code_offsets)
                    .len();
            }
            n
        });
}

#[divan::bench(args = QUERIES, sample_count = 3, sample_size = 1)]
fn decompress_memmem(bencher: Bencher, query_arg: &str) {
    let (ds, pattern) = parse_query(query_arg);
    let d = dataset(ds);
    let finder = memmem::Finder::new(pattern.as_bytes());
    bencher
        .counter(divan::counter::BytesCount::new(d.total_bytes))
        .bench(|| {
            let mut n = 0usize;
            for f in &d.files {
                n += memmem_rows(divan::black_box(f), &finder).len();
            }
            n
        });
}

#[divan::bench(args = QUERIES, sample_count = 5, sample_size = 1)]
fn dfa(bencher: Bencher, query_arg: &str) {
    let d = dataset(parse_query(query_arg).0);
    let s = searchers("sampled", query_arg);
    bencher
        .counter(divan::counter::BytesCount::new(d.total_bytes))
        .bench(|| {
            let mut n = 0usize;
            for (f, s) in d.files.iter().zip(s) {
                n += s
                    .matching_rows_unfiltered(divan::black_box(&f.col.codes), &f.col.code_offsets)
                    .len();
            }
            n
        });
}

#[divan::bench(args = QUERIES, sample_count = 10, sample_size = 1)]
fn prefilter_dfa_sampled(bencher: Bencher, query_arg: &str) {
    bench_prefilter_dfa(bencher, "sampled", query_arg);
}

#[divan::bench(args = QUERIES, sample_count = 10, sample_size = 1)]
fn prefilter_dfa_stats(bencher: Bencher, query_arg: &str) {
    bench_prefilter_dfa(bencher, "stats", query_arg);
}

#[divan::bench(args = QUERIES, sample_count = 10, sample_size = 1)]
fn prefilter_dfa_heuristic(bencher: Bencher, query_arg: &str) {
    bench_prefilter_dfa(bencher, "heuristic", query_arg);
}

#[divan::bench(args = QUERIES, sample_count = 10, sample_size = 1)]
fn prefilter_only(bencher: Bencher, query_arg: &str) {
    let d = dataset(parse_query(query_arg).0);
    let s = searchers("sampled", query_arg);
    bencher
        .counter(divan::counter::BytesCount::new(d.total_bytes))
        .bench(|| {
            let mut n = 0usize;
            for (f, s) in d.files.iter().zip(s) {
                n += s
                    .candidate_rows(divan::black_box(&f.col.codes), &f.col.code_offsets)
                    .len();
            }
            n
        });
}

/// `prefilter_only` with deferred (dictionary-only) searchers: the difference
/// to `prefilter_only` is the per-scan anchor-selection warmup.
#[divan::bench(args = QUERIES, sample_count = 10, sample_size = 1)]
fn prefilter_only_deferred(bencher: Bencher, query_arg: &str) {
    let d = dataset(parse_query(query_arg).0);
    let s = searchers("deferred", query_arg);
    bencher
        .counter(divan::counter::BytesCount::new(d.total_bytes))
        .bench(|| {
            let mut n = 0usize;
            for (f, s) in d.files.iter().zip(s) {
                n += s
                    .candidate_rows(divan::black_box(&f.col.codes), &f.col.code_offsets)
                    .len();
            }
            n
        });
}

/// One-off query compile cost per chunk dictionary, per mode.
fn bench_compile(bencher: Bencher, mode: &'static str, query_arg: &str) {
    let (ds, pattern) = parse_query(query_arg);
    let d = dataset(ds);
    bencher.bench(|| {
        for f in &d.files {
            divan::black_box(compile_mode(mode, divan::black_box(f), pattern.as_bytes()));
        }
    });
}

#[divan::bench(args = QUERIES, sample_count = 5, sample_size = 1)]
fn compile_sampled(bencher: Bencher, query_arg: &str) {
    bench_compile(bencher, "sampled", query_arg);
}

#[divan::bench(args = QUERIES, sample_count = 5, sample_size = 1)]
fn compile_stats(bencher: Bencher, query_arg: &str) {
    bench_compile(bencher, "stats", query_arg);
}

#[divan::bench(args = QUERIES, sample_count = 5, sample_size = 1)]
fn compile_heuristic(bencher: Bencher, query_arg: &str) {
    bench_compile(bencher, "heuristic", query_arg);
}

fn main() {
    let _ = corpus();
    divan::main();
}
