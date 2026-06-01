// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//! Compressed-domain search benchmark: `Pattern::Contains` / `Pattern::Prefix`
//! over a real (or synthetic) string column, never decompressing.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::missing_panics_doc,
    clippy::unwrap_used
)]
//
// A pre-pass scans the corpus to bucket needles by selectivity — `rare`,
// `medium`, `common` — for both modes, so the benchmark reports how throughput
// varies with match density (a `common` needle hits the automaton's early-exit
// on most rows; a `rare` one scans almost every token). The selected needles,
// their measured selectivity, and a brute-force cross-check are printed at
// startup.
//
// Corpus resolution mirrors `clickbench.rs`:
//   1. env `ONPAIR_BENCH_PARQUET` (+ optional `ONPAIR_BENCH_COLUMN`)
//   2. `/tmp/userdata1.parquet`
//   3. a synthetic ClickBench-shaped URL corpus.
// Code width is `ONPAIR_SEARCH_BITS` (default 16).
//
// Run with: cargo bench --bench search

use std::env;
use std::fmt;
use std::fs::File;
use std::path::PathBuf;
use std::sync::OnceLock;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use divan::Bencher;
use divan::counter::BytesCount;
use divan::counter::ItemsCount;
use onpair::Bits;
use onpair::Column;
use onpair::Config;
use onpair::Pattern;
use onpair::Threshold;
use onpair::compress;
use onpair::decompress;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

// ─────────────────────────────────────────────────────────────────────────────
// Corpus loading (shared shape with clickbench.rs).
// ─────────────────────────────────────────────────────────────────────────────

struct Corpus {
    source: String,
    rows: Vec<Vec<u8>>,
    bytes: Vec<u8>,
    offsets: Vec<u64>,
    total_bytes: usize,
}

fn pack(strings: &[Vec<u8>]) -> (Vec<u8>, Vec<u64>) {
    let mut bytes = Vec::with_capacity(strings.iter().map(|s| s.len()).sum());
    let mut offsets = Vec::with_capacity(strings.len() + 1);
    offsets.push(0u64);
    for s in strings {
        bytes.extend_from_slice(s);
        offsets.push(bytes.len() as u64);
    }
    (bytes, offsets)
}

fn corpus() -> &'static Corpus {
    static CORPUS: OnceLock<Corpus> = OnceLock::new();
    CORPUS.get_or_init(|| {
        let (source, rows) = load_corpus();
        let (bytes, offsets) = pack(&rows);
        let total_bytes = bytes.len();
        let c = Corpus {
            source,
            rows,
            bytes,
            offsets,
            total_bytes,
        };
        eprintln!(
            "[onpair search] corpus: {} ({} rows, {:.2} MiB)",
            c.source,
            c.rows.len(),
            c.total_bytes as f64 / (1024.0 * 1024.0)
        );
        c
    })
}

fn load_corpus() -> (String, Vec<Vec<u8>>) {
    if let Ok(path) = env::var("ONPAIR_BENCH_PARQUET")
        && let Some(rows) = read_parquet_strings(&PathBuf::from(&path))
    {
        return (format!("{path} (env)"), rows);
    }
    let fallback = PathBuf::from("/tmp/userdata1.parquet");
    if fallback.exists()
        && let Some(rows) = read_parquet_strings(&fallback)
    {
        return (format!("{} (auto-detected)", fallback.display()), rows);
    }
    let rows = synthetic_clickbench_urls(100_000);
    ("synthetic ClickBench-shaped URL corpus".to_string(), rows)
}

fn read_parquet_strings(path: &PathBuf) -> Option<Vec<Vec<u8>>> {
    let file = File::open(path).ok()?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).ok()?;
    let schema = builder.schema().clone();

    let col_name = env::var("ONPAIR_BENCH_COLUMN").ok();
    let picked = match col_name.as_deref() {
        Some(name) => schema.fields().iter().position(|f| f.name() == name)?,
        None => schema.fields().iter().position(|f| {
            use arrow_schema::DataType::*;
            matches!(
                f.data_type(),
                Utf8 | LargeUtf8 | Utf8View | Binary | LargeBinary | BinaryView
            )
        })?,
    };
    let col_field = schema.fields().get(picked)?.clone();
    eprintln!(
        "[onpair search] reading column #{picked} `{}` ({})",
        col_field.name(),
        col_field.data_type()
    );

    let mut rows: Vec<Vec<u8>> = Vec::new();
    // Optional cap so huge corpora (e.g. FineWeb `text`) fit in memory.
    let max_rows = env::var("ONPAIR_BENCH_MAX_ROWS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(usize::MAX);
    let reader = builder.build().ok()?;
    'outer: for batch in reader.flatten() {
        let arr = batch.column(picked);
        use arrow_schema::DataType::*;
        match arr.data_type() {
            Utf8 => {
                for s in arr.as_string::<i32>().iter() {
                    rows.push(s.unwrap_or("").as_bytes().to_vec());
                }
            }
            LargeUtf8 => {
                for s in arr.as_string::<i64>().iter() {
                    rows.push(s.unwrap_or("").as_bytes().to_vec());
                }
            }
            Utf8View => {
                for s in arr.as_string_view().iter() {
                    rows.push(s.unwrap_or("").as_bytes().to_vec());
                }
            }
            Binary => {
                let a = arr
                    .as_any()
                    .downcast_ref::<arrow_array::BinaryArray>()?;
                for b in a.iter() {
                    rows.push(b.unwrap_or(b"").to_vec());
                }
            }
            LargeBinary => {
                let a = arr
                    .as_any()
                    .downcast_ref::<arrow_array::LargeBinaryArray>()?;
                for b in a.iter() {
                    rows.push(b.unwrap_or(b"").to_vec());
                }
            }
            BinaryView => {
                let a = arr
                    .as_any()
                    .downcast_ref::<arrow_array::BinaryViewArray>()?;
                for b in a.iter() {
                    rows.push(b.unwrap_or(b"").to_vec());
                }
            }
            _ => return None,
        }
        if rows.len() >= max_rows {
            rows.truncate(max_rows);
            break 'outer;
        }
    }
    Some(rows)
}

fn synthetic_clickbench_urls(n: usize) -> Vec<Vec<u8>> {
    const HOSTS: &[&str] = &[
        "https://www.yandex.ru",
        "https://www.google.com",
        "https://news.ycombinator.com",
        "https://www.example.com",
        "https://docs.example.org",
        "https://api.example.net",
        "http://m.yandex.ru",
        "https://maps.example.com",
        "https://shop.example.com",
        "ftp://files.example.com",
    ];
    const PATHS: &[&str] = &[
        "/",
        "/page",
        "/news",
        "/search?q=",
        "/profile",
        "/login",
        "/api/v1/data",
        "/static/asset.png",
        "/blog/post-",
        "/feed.xml",
        "/sitemap.xml",
        "/users/",
        "/admin/dashboard",
        "/categories/electronics",
        "/cart/checkout",
    ];
    const TAILS: &[&str] = &["", "alpha", "beta", "gamma", "delta", "001", "002", "003"];
    let mut out = Vec::with_capacity(n);
    let mut x = 0x9E3779B97F4A7C15u64;
    for _ in 0..n {
        x = x.wrapping_add(0x9E3779B97F4A7C15);
        let h = HOSTS[(x as usize) % HOSTS.len()];
        let p = PATHS[((x >> 16) as usize) % PATHS.len()];
        let t = TAILS[((x >> 32) as usize) % TAILS.len()];
        let m = (x >> 48) as u16;
        out.push(format!("{h}{p}{t}{m}").into_bytes());
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Compressed column (one width, default 16).
// ─────────────────────────────────────────────────────────────────────────────

fn search_bits() -> u8 {
    env::var("ONPAIR_SEARCH_BITS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16)
}

fn column() -> &'static Column<u64> {
    static COL: OnceLock<Column<u64>> = OnceLock::new();
    COL.get_or_init(|| {
        let c = corpus();
        let cfg = Config {
            bits: Bits::new(search_bits()).unwrap(),
            threshold: Threshold::new(0.5).unwrap(),
            seed: Some(42),
        };
        let col = compress(&c.bytes, &c.offsets, cfg).unwrap();
        let dict_b = col.dict_bytes.len() + col.dict_offsets.len() * 4;
        let codes_b = col.codes.len() * 2;
        let offs_b = col.code_offsets.len() * 8;
        let first_b = col.first_codes.as_ref().map_or(0, |f| f.len() * 2);
        let core = dict_b + codes_b + offs_b;
        eprintln!(
            "[onpair search] compressed @ bits={}: {} dict tokens, {} codes",
            col.bits,
            col.dict_offsets.len() - 1,
            col.codes.len(),
        );
        eprintln!(
            "[onpair search] footprint: dict {:.0} KiB + codes {:.0} KiB + code_offsets {:.0} KiB = {:.0} KiB core; \
             first_codes (search index) {:.0} KiB = +{:.2}% over core, +{:.2}% over input",
            dict_b as f64 / 1024.0,
            codes_b as f64 / 1024.0,
            offs_b as f64 / 1024.0,
            core as f64 / 1024.0,
            first_b as f64 / 1024.0,
            100.0 * first_b as f64 / core as f64,
            100.0 * first_b as f64 / c.total_bytes as f64,
        );
        col
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Needle pre-pass: bucket candidates by selectivity.
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Copy, Clone, PartialEq, Eq)]
enum Mode {
    Contains,
    Prefix,
}

struct Needle {
    bucket: &'static str,
    mode: Mode,
    bytes: Vec<u8>,
    selectivity: f64,
}

impl fmt::Display for Needle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // e.g. common:"example"(58.1%)
        write!(
            f,
            "{}:\"{}\"({:.1}%)",
            self.bucket,
            self.bytes.escape_ascii(),
            self.selectivity * 100.0,
        )
    }
}

/// Buckets as (label, target selectivity, inclusive range).
const BUCKETS: &[(&str, f64, f64, f64)] = &[
    ("rare", 0.002, 0.0003, 0.02),
    ("medium", 0.10, 0.03, 0.25),
    ("common", 0.55, 0.40, 1.0),
];

const CAND_LENS: &[usize] = &[3, 5, 8, 12];

/// Count rows in `rows` matching `needle` under `mode`. Brute force.
fn brute_count(rows: &[Vec<u8>], needle: &[u8], mode: Mode) -> usize {
    if needle.is_empty() {
        return rows.len();
    }
    match mode {
        Mode::Prefix => rows.iter().filter(|r| r.starts_with(needle)).count(),
        Mode::Contains => rows
            .iter()
            .filter(|r| r.len() >= needle.len() && r.windows(needle.len()).any(|w| w == needle))
            .count(),
    }
}

/// Pick one representative needle per (bucket, mode) by sampling candidate
/// substrings/prefixes and estimating their selectivity over a row sample.
fn select_needles() -> &'static [Needle] {
    static NEEDLES: OnceLock<Vec<Needle>> = OnceLock::new();
    NEEDLES.get_or_init(|| {
        let rows = &corpus().rows;

        // Explicit override: `ONPAIR_NEEDLES="contains:google,prefix:http://"`.
        // Each spec is `mode:text` (mode = contains|prefix); the bucket label is
        // the literal text so the report names it. Real selectivity is computed
        // over the full corpus. Lets a specific query (e.g. the ClickBench
        // `URL LIKE '%google%'`) be benchmarked directly.
        if let Ok(spec) = env::var("ONPAIR_NEEDLES") {
            let mut out = Vec::new();
            for item in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                let (mode, text) = match item.split_once(':') {
                    Some(("prefix", t)) => (Mode::Prefix, t),
                    Some(("contains", t)) => (Mode::Contains, t),
                    _ => panic!("ONPAIR_NEEDLES item must be `contains:TEXT` or `prefix:TEXT`, got {item:?}"),
                };
                let bytes = text.as_bytes().to_vec();
                let sel = brute_count(rows, &bytes, mode) as f64 / rows.len() as f64;
                out.push(Needle {
                    bucket: Box::leak(text.to_string().into_boxed_str()),
                    mode,
                    bytes,
                    selectivity: sel,
                });
            }
            return out;
        }

        // Deterministic sampler shared across phases.
        let mut x = 0xD1B54A32D192ED03u64;
        let mut next = |bound: usize| -> usize {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((x >> 33) as usize) % bound.max(1)
        };

        // Row sample used for cheap selectivity estimation.
        let est_rows: Vec<Vec<u8>> = {
            let take = rows.len().min(8000);
            (0..take).map(|_| rows[next(rows.len())].clone()).collect()
        };
        let est_n = est_rows.len() as f64;

        let mut out: Vec<Needle> = Vec::new();
        for &mode in &[Mode::Contains, Mode::Prefix] {
            // Generate candidates from random rows × candidate lengths, dedup.
            let mut seen: std::collections::HashSet<Vec<u8>> = std::collections::HashSet::new();
            let mut cands: Vec<Vec<u8>> = Vec::new();
            let target = 700usize;
            let mut tries = 0usize;
            while cands.len() < target && tries < target * 20 {
                tries += 1;
                let row = &rows[next(rows.len())];
                if row.is_empty() {
                    continue;
                }
                let len = CAND_LENS[next(CAND_LENS.len())];
                if row.len() < len {
                    continue;
                }
                let start = match mode {
                    Mode::Prefix => 0,
                    Mode::Contains => next(row.len() - len + 1),
                };
                let cand = row[start..start + len].to_vec();
                if seen.insert(cand.clone()) {
                    cands.push(cand);
                }
            }

            // Estimate selectivity for every candidate, then for each bucket
            // keep the candidate whose selectivity is closest to the target.
            let mut best: Vec<Option<(f64, Vec<u8>)>> = vec![None; BUCKETS.len()];
            for cand in &cands {
                let sel = brute_count(&est_rows, cand, mode) as f64 / est_n;
                for (bi, &(_, tgt, lo, hi)) in BUCKETS.iter().enumerate() {
                    if sel < lo || sel > hi {
                        continue;
                    }
                    let dist = (sel - tgt).abs();
                    let better = best[bi]
                        .as_ref()
                        .is_none_or(|(bdist, _)| dist < *bdist);
                    if better {
                        best[bi] = Some((dist, cand.clone()));
                    }
                }
            }

            for (bi, &(label, ..)) in BUCKETS.iter().enumerate() {
                if let Some((_, bytes)) = best[bi].take() {
                    // Exact selectivity over the full corpus for the report.
                    let sel = brute_count(rows, &bytes, mode) as f64 / rows.len() as f64;
                    out.push(Needle {
                        bucket: label,
                        mode,
                        bytes,
                        selectivity: sel,
                    });
                }
            }
        }
        out
    })
}

fn contains_needles() -> Vec<&'static Needle> {
    select_needles()
        .iter()
        .filter(|n| n.mode == Mode::Contains)
        .collect()
}

fn prefix_needles() -> Vec<&'static Needle> {
    select_needles()
        .iter()
        .filter(|n| n.mode == Mode::Prefix)
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Benches.
// ─────────────────────────────────────────────────────────────────────────────

fn bench_search(bencher: Bencher, needle: &Needle) {
    let parts = column().as_search_parts();
    let c = corpus();
    // Throughput is reported over the bytes the scan must stream and the rows
    // it covers. `Contains` walks the whole code stream; `Prefix` (with the
    // index) streams only the first-code table (2 B/row) in pass 1.
    let bytes_scanned = match needle.mode {
        Mode::Contains => parts.codes.len() * 2,
        Mode::Prefix => c.rows.len() * 2,
    };
    bencher
        .counter(BytesCount::new(bytes_scanned))
        .counter(ItemsCount::new(c.rows.len()))
        .bench_local(|| {
            let pattern = match needle.mode {
                Mode::Contains => Pattern::Contains(&needle.bytes),
                Mode::Prefix => Pattern::Prefix(&needle.bytes),
            };
            // Count via the callback primitive so the timing reflects the scan,
            // not the result-mask allocation.
            let mut matches = 0usize;
            parts.search_callback(pattern, |_| matches += 1);
            divan::black_box(matches)
        });
}

#[divan::bench(args = contains_needles())]
fn contains(bencher: Bencher, needle: &Needle) {
    bench_search(bencher, needle);
}

#[divan::bench(args = prefix_needles())]
fn prefix(bencher: Bencher, needle: &Needle) {
    bench_search(bencher, needle);
}

/// Prefix via `search()` → `RowMask` (the bitmap-merge fast path: pass-1 accept
/// bits are written straight into the mask, no per-row callback). Contrast with
/// `prefix`, which exercises the per-row `search_callback` path.
#[divan::bench(args = prefix_needles())]
fn prefix_mask(bencher: Bencher, needle: &Needle) {
    let parts = column().as_search_parts();
    let c = corpus();
    bencher
        .counter(BytesCount::new(c.rows.len() * 2))
        .counter(ItemsCount::new(c.rows.len()))
        .bench_local(|| {
            let mask = parts.search(Pattern::Prefix(&needle.bytes));
            divan::black_box(popcount(mask.as_words()))
        });
}

/// Count set bits across packed mask words.
fn popcount(words: &[u64]) -> usize {
    words.iter().map(|w| w.count_ones() as usize).sum()
}

/// A/B baseline: identical prefix search but with the first-token index
/// suppressed (`first_codes = None`), forcing the generic per-row scan. The
/// gap to `prefix` is the search index's runtime payoff.
#[divan::bench(args = prefix_needles())]
fn prefix_no_index(bencher: Bencher, needle: &Needle) {
    let mut parts = column().as_search_parts();
    parts.first_codes = None;
    let c = corpus();
    // Same denominator as `prefix` (rows × 2) so the two are directly
    // comparable, though the no-index path actually streams the code stream.
    bencher
        .counter(BytesCount::new(c.rows.len() * 2))
        .counter(ItemsCount::new(c.rows.len()))
        .bench_local(|| {
            let mut matches = 0usize;
            parts.search_callback(Pattern::Prefix(&needle.bytes), |_| matches += 1);
            divan::black_box(matches)
        });
}

// ─────────────────────────────────────────────────────────────────────────────
// Arrow-like baselines: evaluate the predicate over decompressed bytes the way
// an Arrow `StringArray` LIKE kernel does — a (values, offsets) buffer pair with
// `starts_with` (prefix) or `memchr::memmem` (contains, the finder Arrow's
// `contains`/`like` kernels use) — and pack the per-row verdict into a
// `BooleanBuffer` via `collect_bool`, the same 64-bits-per-word packer arrow-rs
// uses to build the `BooleanArray` result. This makes the baseline produce a
// packed bitmask comparable to onpair's `RowMask`, not just a counter.
// `*_arrow` assumes the data is already decompressed in memory; the
// `*_decompress_arrow` pair pays the onpair decompress first, so it is the true
// "decode then scan" alternative to compressed-domain search.
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate `needle` over a decompressed `(bytes, offsets)` buffer
/// Arrow-`StringArray`-style, packing the verdicts into a `BooleanBuffer` with
/// `collect_bool` (the arrow-rs bitmask builder), and return its set-bit count.
fn arrow_mask(bytes: &[u8], offsets: &[u64], needle: &Needle) -> usize {
    let n = offsets.len() - 1;
    let mask = match needle.mode {
        Mode::Prefix => arrow_buffer::BooleanBuffer::collect_bool(n, |r| {
            bytes[offsets[r] as usize..offsets[r + 1] as usize].starts_with(&needle.bytes)
        }),
        Mode::Contains => {
            let finder = memchr::memmem::Finder::new(&needle.bytes);
            arrow_buffer::BooleanBuffer::collect_bool(n, |r| {
                finder
                    .find(&bytes[offsets[r] as usize..offsets[r + 1] as usize])
                    .is_some()
            })
        }
    };
    mask.count_set_bits()
}

#[divan::bench(args = prefix_needles())]
fn prefix_arrow(bencher: Bencher, needle: &Needle) {
    let c = corpus();
    bencher
        .counter(BytesCount::new(c.total_bytes))
        .counter(ItemsCount::new(c.rows.len()))
        .bench_local(|| divan::black_box(arrow_mask(&c.bytes, &c.offsets, needle)));
}

#[divan::bench(args = contains_needles())]
fn contains_arrow(bencher: Bencher, needle: &Needle) {
    let c = corpus();
    bencher
        .counter(BytesCount::new(c.total_bytes))
        .counter(ItemsCount::new(c.rows.len()))
        .bench_local(|| divan::black_box(arrow_mask(&c.bytes, &c.offsets, needle)));
}

#[divan::bench(args = prefix_needles())]
fn prefix_decompress_arrow(bencher: Bencher, needle: &Needle) {
    let col = column();
    let c = corpus();
    bencher
        .counter(BytesCount::new(c.total_bytes))
        .counter(ItemsCount::new(c.rows.len()))
        .bench_local(|| {
            let bytes = decompress(col.as_parts());
            divan::black_box(arrow_mask(&bytes, &c.offsets, needle))
        });
}

#[divan::bench(args = contains_needles())]
fn contains_decompress_arrow(bencher: Bencher, needle: &Needle) {
    let col = column();
    let c = corpus();
    bencher
        .counter(BytesCount::new(c.total_bytes))
        .counter(ItemsCount::new(c.rows.len()))
        .bench_local(|| {
            let bytes = decompress(col.as_parts());
            divan::black_box(arrow_mask(&bytes, &c.offsets, needle))
        });
}

// ─────────────────────────────────────────────────────────────────────────────
// Roofline baselines.
// ─────────────────────────────────────────────────────────────────────────────
// All report throughput against the same logical corpus bytes as the search
// benches, so the GB/s column is directly comparable.
//
//   copy_all_codes      — read + write the whole codes stream (a memcpy of the
//                          compressed payload). The "decode would at least cost
//                          this" reference, and what prefix must beat to win.
//   scan_all_codes       — read every code once (no early exit). The hard floor
//                          for `contains`: it must look at every token of a
//                          non-matching row.
//   first_code_per_row   — read code_offsets + the first code of each row. The
//                          floor for `prefix`, which dies after ~one token.

#[divan::bench]
fn copy_all_codes(bencher: Bencher) {
    let codes = &column().codes;
    let mut dst = vec![0u16; codes.len()];
    bencher
        .counter(BytesCount::new(corpus().total_bytes))
        .bench_local(|| {
            dst.copy_from_slice(codes);
            divan::black_box(&dst);
        });
}

#[divan::bench]
fn scan_all_codes(bencher: Bencher) {
    let codes = &column().codes;
    bencher
        .counter(BytesCount::new(corpus().total_bytes))
        .bench_local(|| {
            let mut acc = 0u64;
            for &c in codes {
                acc = acc.wrapping_add(c as u64);
            }
            divan::black_box(acc)
        });
}

#[divan::bench]
fn first_code_per_row(bencher: Bencher) {
    let col = column();
    bencher
        .counter(BytesCount::new(corpus().total_bytes))
        .counter(ItemsCount::new(corpus().rows.len()))
        .bench_local(|| {
            let mut acc = 0u64;
            for w in col.code_offsets.windows(2) {
                if w[1] > w[0] {
                    acc ^= col.codes[w[0] as usize] as u64;
                }
            }
            divan::black_box(acc)
        });
}

fn main() {
    // Touch corpus, column, and needles so the report prints before divan runs,
    // and cross-check the compressed-domain count against brute force.
    let _ = column();
    let rows = &corpus().rows;
    eprintln!("[onpair search] selected needles (compressed-domain vs brute-force):");
    for n in select_needles() {
        let mode = match n.mode {
            Mode::Contains => "contains",
            Mode::Prefix => "prefix",
        };
        let mask = column().as_search_parts().search(match n.mode {
            Mode::Contains => Pattern::Contains(&n.bytes),
            Mode::Prefix => Pattern::Prefix(&n.bytes),
        });
        let cd = popcount(mask.as_words());
        let bf = brute_count(rows, &n.bytes, n.mode);
        let ok = if cd == bf { "ok" } else { "MISMATCH" };
        eprintln!(
            "  [{ok}] {mode:>8} {:>6} \"{}\" sel={:.3}% cd={cd} bf={bf}",
            n.bucket,
            n.bytes.escape_ascii(),
            n.selectivity * 100.0,
        );
        assert_eq!(cd, bf, "compressed-domain search disagrees with brute force");
    }
    divan::main();
}
