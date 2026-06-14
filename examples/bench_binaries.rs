// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//! Compression-quality benchmark of OnPair vs. block-based compressors
//! (Zstandard at several levels, and Snappy) over large executable binaries.
//!
//! For each input file we measure, repeated `N` times each (we report the
//! minimum and the median):
//!   * compression time and the achieved compression ratio (orig / compressed)
//!   * decompression time
//!
//! OnPair is a *columnar* random-access codec: it compresses a collection of
//! values sharing one dictionary. To feed it a flat binary we split the file
//! into fixed-size records ("blocks"). We report:
//!   * `onpair-12` / `onpair-16`        — whole file as a single record (the
//!                                          maximum-ratio configuration)
//!   * `onpair-16-<block>k`             — file split into `block` KiB records,
//!                                          OnPair's actual random-access mode
//!
//! Block compressors (`zstd-*`, `snappy`) compress the whole file in one shot,
//! which is their normal usage.
//!
//! Usage:
//!   cargo run --release --example bench_binaries -- <file1> <file2> ...
//!
//! Env knobs:
//!   ONPAIR_BENCH_ITERS_C  timed compression iterations   (default 5)
//!   ONPAIR_BENCH_ITERS_D  timed decompression iterations (default 10)
//!   ONPAIR_BENCH_BLOCK    OnPair record size in KiB       (default 4)
#![allow(
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::unwrap_used
)]

use std::env;
use std::fs;
use std::time::Instant;

use onpair::Bits;
use onpair::Config;
use onpair::Threshold;
use onpair::compress;
use onpair::decompress;

/// Summary statistics over a set of timed runs (seconds).
struct Stats {
    min: f64,
    median: f64,
}

impl Stats {
    fn from(mut v: Vec<f64>) -> Self {
        v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN timings"));
        let min = v[0];
        let median = v[v.len() / 2];
        Self { min, median }
    }
}

/// Run `f` `iters` times, returning the produced size (bytes) and timing stats.
fn bench<F: FnMut() -> usize>(iters: usize, mut f: F) -> (usize, Stats) {
    let mut size = 0;
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        size = f();
        times.push(t.elapsed().as_secs_f64());
    }
    (size, Stats::from(times))
}

/// Shannon entropy of the byte distribution, in bits/byte (0..=8).
fn entropy(bytes: &[u8]) -> f64 {
    let mut counts = [0u64; 256];
    for &b in bytes {
        counts[b as usize] += 1;
    }
    let n = bytes.len() as f64;
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n;
            -p * p.log2()
        })
        .sum()
}

/// Cumulative byte offsets splitting `len` bytes into `block`-byte records.
fn block_offsets(len: usize, block: usize) -> Vec<u64> {
    let mut offsets = Vec::with_capacity(len / block + 2);
    let mut pos = 0;
    while pos < len {
        offsets.push(pos as u64);
        pos += block;
    }
    offsets.push(len as u64);
    offsets
}

/// On-disk size of an OnPair column, mirroring the accounting used by the
/// `bench_tpch` example: dictionary bytes + dictionary offsets (u32) + codes
/// (u16) + per-row code offsets.
fn onpair_size(col: &onpair::Column<u64>) -> usize {
    let parts = col.as_parts();
    parts.dict_bytes.len()
        + parts.dict_offsets.len() * 4
        + parts.codes.len() * 2
        + std::mem::size_of_val(col.code_offsets.as_slice())
}

fn onpair_cfg(bits: u8) -> Config {
    Config {
        bits: Bits::new(bits).expect("bits in 9..=16"),
        threshold: Threshold::new(0.2).expect("0.2 in range"),
        seed: Some(42),
    }
}

/// Emit one CSV row and verify the roundtrip.
#[allow(clippy::too_many_arguments)]
fn report(
    file: &str,
    orig: usize,
    entropy: f64,
    codec: &str,
    compressed: usize,
    c: &Stats,
    d: &Stats,
    roundtrip_ok: bool,
) {
    let mib = orig as f64 / (1024.0 * 1024.0);
    let ratio = orig as f64 / compressed as f64;
    let c_mbps = mib / c.min;
    let d_mbps = mib / d.min;
    println!(
        "{file},{orig},{entropy:.3},{codec},{compressed},{ratio:.3},\
         {:.2},{:.2},{c_mbps:.1},{:.2},{:.2},{d_mbps:.1},{}",
        c.min * 1e3,
        c.median * 1e3,
        d.min * 1e3,
        d.median * 1e3,
        if roundtrip_ok { "ok" } else { "FAIL" },
    );
    if !roundtrip_ok {
        eprintln!("!! roundtrip FAILED for {file} / {codec}");
    }
}

fn main() {
    let iters_c = env_usize("ONPAIR_BENCH_ITERS_C", 5);
    let iters_d = env_usize("ONPAIR_BENCH_ITERS_D", 10);
    let block_kib = env_usize("ONPAIR_BENCH_BLOCK", 4);
    let block = block_kib * 1024;

    let files: Vec<String> = env::args().skip(1).collect();
    if files.is_empty() {
        eprintln!("usage: bench_binaries <file> [file ...]");
        std::process::exit(2);
    }

    eprintln!("iters: compress={iters_c} decompress={iters_d}  onpair block={block_kib} KiB");
    println!(
        "file,orig_bytes,entropy_bits,codec,compressed_bytes,ratio,\
         compress_min_ms,compress_med_ms,compress_mibps,\
         decompress_min_ms,decompress_med_ms,decompress_mibps,roundtrip"
    );

    for path in &files {
        let bytes = match fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skip {path}: {e}");
                continue;
            }
        };
        let name = path.rsplit('/').next().unwrap_or(path).to_string();
        let orig = bytes.len();
        let ent = entropy(&bytes);
        eprintln!(
            "\n# {name}: {:.2} MiB, entropy {ent:.3} bits/byte",
            orig as f64 / (1024.0 * 1024.0)
        );

        // ---- OnPair, whole file (single record) at two code widths ----
        let whole = vec![0u64, orig as u64];
        for bits in [12u8, 16u8] {
            let cfg = onpair_cfg(bits);
            let (size, cs) = bench(iters_c, || {
                onpair_size(&compress(&bytes, &whole, cfg).expect("onpair compress"))
            });
            let col = compress(&bytes, &whole, cfg).expect("onpair compress");
            let (_, ds) = bench(iters_d, || decompress(col.as_parts()).len());
            let ok = decompress(col.as_parts()) == bytes;
            report(&name, orig, ent, &format!("onpair-{bits}"), size, &cs, &ds, ok);
        }

        // ---- OnPair, blocked (random-access mode), 16-bit ----
        {
            let offsets = block_offsets(orig, block);
            let cfg = onpair_cfg(16);
            let (size, cs) = bench(iters_c, || {
                onpair_size(&compress(&bytes, &offsets, cfg).expect("onpair compress"))
            });
            let col = compress(&bytes, &offsets, cfg).expect("onpair compress");
            let (_, ds) = bench(iters_d, || decompress(col.as_parts()).len());
            let ok = decompress(col.as_parts()) == bytes;
            let codec = format!("onpair-16-{block_kib}k");
            report(&name, orig, ent, &codec, size, &cs, &ds, ok);
        }

        // ---- Zstandard at several levels ----
        for level in [3i32, 19, 22] {
            let (size, cs) = bench(iters_c, || {
                zstd::encode_all(&bytes[..], level).expect("zstd compress").len()
            });
            let comp = zstd::encode_all(&bytes[..], level).expect("zstd compress");
            let (_, ds) = bench(iters_d, || {
                zstd::decode_all(&comp[..]).expect("zstd decompress").len()
            });
            let ok = zstd::decode_all(&comp[..]).expect("zstd decompress") == bytes;
            report(&name, orig, ent, &format!("zstd-{level}"), size, &cs, &ds, ok);
        }

        // ---- Snappy ----
        {
            let (size, cs) = bench(iters_c, || {
                snap::raw::Encoder::new()
                    .compress_vec(&bytes)
                    .expect("snappy compress")
                    .len()
            });
            let comp = snap::raw::Encoder::new()
                .compress_vec(&bytes)
                .expect("snappy compress");
            let (_, ds) = bench(iters_d, || {
                snap::raw::Decoder::new()
                    .decompress_vec(&comp)
                    .expect("snappy decompress")
                    .len()
            });
            let ok = snap::raw::Decoder::new()
                .decompress_vec(&comp)
                .expect("snappy decompress")
                == bytes;
            report(&name, orig, ent, "snappy", size, &cs, &ds, ok);
        }
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}
