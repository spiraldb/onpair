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
            matches!(f.data_type(), Utf8 | LargeUtf8 | Utf8View)
        })?,
    };
    let col_field = schema.fields().get(picked)?.clone();
    eprintln!(
        "[onpair search] reading column #{picked} `{}` ({})",
        col_field.name(),
        col_field.data_type()
    );

    let mut rows: Vec<Vec<u8>> = Vec::new();
    let reader = builder.build().ok()?;
    for batch in reader.flatten() {
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
            _ => return None,
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
        eprintln!(
            "[onpair search] compressed @ bits={}: {} dict tokens, {} codes",
            col.bits,
            col.dict_offsets.len() - 1,
            col.codes.len(),
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
    bencher
        .counter(BytesCount::new(c.total_bytes))
        .counter(ItemsCount::new(c.rows.len()))
        .bench_local(|| {
            let pattern = match needle.mode {
                Mode::Contains => Pattern::Contains(&needle.bytes),
                Mode::Prefix => Pattern::Prefix(&needle.bytes),
            };
            // Count via the callback primitive so the timing reflects the scan,
            // not the result-mask allocation.
            let mut matches = 0usize;
            parts.search_for_each(pattern, |_| matches += 1);
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

/// Dump the corpus and selected needles as length-prefixed little-endian
/// binary so the C++ harness (`search_bench.cpp`) searches byte-identical
/// inputs. Triggered by `ONPAIR_SEARCH_DUMP=<dir>`.
///
/// `corpus.bin`: `u64 n_rows`, then `n_rows × u32 row_len`, then the
/// concatenated row bytes. `needles.bin`: `u32 count`, then per needle
/// `u8 mode (0=contains,1=prefix)`, `u8 bucket_len` + bucket, `f64 sel`,
/// `u32 len` + needle bytes.
fn dump_for_cpp(dir: &str) {
    use std::io::Write;

    let rows = &corpus().rows;
    let mut cf = std::io::BufWriter::new(File::create(format!("{dir}/corpus.bin")).unwrap());
    cf.write_all(&(rows.len() as u64).to_le_bytes()).unwrap();
    for r in rows {
        cf.write_all(&(r.len() as u32).to_le_bytes()).unwrap();
    }
    for r in rows {
        cf.write_all(r).unwrap();
    }
    cf.flush().unwrap();

    let needles = select_needles();
    let mut nf = std::io::BufWriter::new(File::create(format!("{dir}/needles.bin")).unwrap());
    nf.write_all(&(needles.len() as u32).to_le_bytes()).unwrap();
    for n in needles {
        let mode: u8 = match n.mode {
            Mode::Contains => 0,
            Mode::Prefix => 1,
        };
        nf.write_all(&[mode]).unwrap();
        nf.write_all(&[n.bucket.len() as u8]).unwrap();
        nf.write_all(n.bucket.as_bytes()).unwrap();
        nf.write_all(&n.selectivity.to_le_bytes()).unwrap();
        nf.write_all(&(n.bytes.len() as u32).to_le_bytes()).unwrap();
        nf.write_all(&n.bytes).unwrap();
    }
    nf.flush().unwrap();
    eprintln!(
        "[onpair search] dumped {} rows + {} needles to {dir}",
        rows.len(),
        needles.len()
    );
}

fn main() {
    // Touch corpus, column, and needles so the report prints before divan runs,
    // and cross-check the compressed-domain count against brute force.
    let _ = column();
    let rows = &corpus().rows;
    if let Ok(dir) = env::var("ONPAIR_SEARCH_DUMP") {
        dump_for_cpp(&dir);
    }
    eprintln!("[onpair search] selected needles (compressed-domain vs brute-force):");
    for n in select_needles() {
        let mode = match n.mode {
            Mode::Contains => "contains",
            Mode::Prefix => "prefix",
        };
        let cd = column()
            .as_search_parts()
            .search(match n.mode {
                Mode::Contains => Pattern::Contains(&n.bytes),
                Mode::Prefix => Pattern::Prefix(&n.bytes),
            })
            .count_ones();
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
