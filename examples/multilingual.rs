// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//! OnPair compression analysis over real multilingual corpora.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::unwrap_used
)]
//
// Loads one UTF-8 text file per corpus, splits it into rows on '\n', and reports
// OnPair's compression ratio and dictionary ("symbol table") distribution at
// each requested code width.
//
// Usage:
//   cargo run --release --example multilingual -- <dir> <label=file> [label=file ...]
// e.g.
//   cargo run --release --example multilingual -- /tmp/corpora \
//       "ASCII (English)=en.txt" "Japanese=ja.txt" "Chinese=zh.txt" "Emoji=emoji.txt"

use std::env;
use std::fs;
use std::path::Path;

use onpair::Bits;
use onpair::Config;
use onpair::Threshold;
use onpair::compress;
use onpair::decompress;

const BITS: &[u8] = &[12, 16];

fn main() {
    let mut args = env::args().skip(1);
    let dir = args.next().unwrap_or_else(|| "/tmp/corpora".to_string());
    let manifest: Vec<(String, String)> = {
        let rest: Vec<String> = args.collect();
        if rest.is_empty() {
            vec![
                ("ASCII (English)".into(), "en.txt".into()),
                ("Japanese".into(), "ja.txt".into()),
                ("Chinese".into(), "zh.txt".into()),
                ("Emoji-heavy".into(), "emoji.txt".into()),
            ]
        } else {
            rest.iter()
                .map(|s| {
                    let (l, f) = s.split_once('=').expect("expected label=file");
                    (l.to_string(), f.to_string())
                })
                .collect()
        }
    };

    // Optional common byte budget so dictionary-fill is comparable across
    // corpora of different sizes (set ONPAIR_CAP_BYTES, truncated at a row).
    let cap_bytes = env::var("ONPAIR_CAP_BYTES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());
    if let Some(c) = cap_bytes {
        println!(
            "(corpora capped at {} KiB each, on a row boundary)",
            c / 1024
        );
    }

    for (label, file) in &manifest {
        let path = Path::new(&dir).join(file);
        let text = match fs::read(&path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("skip {label}: cannot read {}: {e}", path.display());
                continue;
            }
        };
        analyze(label, &text, cap_bytes);
    }
}

/// Split `text` into rows on '\n' (newline dropped) and build OnPair inputs.
fn rows(text: &[u8]) -> (Vec<u8>, Vec<u64>) {
    let mut bytes = Vec::with_capacity(text.len());
    let mut offsets = vec![0u64];
    for line in text.split(|&b| b == b'\n') {
        bytes.extend_from_slice(line);
        offsets.push(bytes.len() as u64);
    }
    (bytes, offsets)
}

/// Truncate the row set at the last row boundary that keeps `bytes <= cap`.
fn cap(bytes: Vec<u8>, offsets: Vec<u64>, cap: usize) -> (Vec<u8>, Vec<u64>) {
    if bytes.len() <= cap {
        return (bytes, offsets);
    }
    let r = offsets
        .partition_point(|&o| (o as usize) <= cap)
        .saturating_sub(1);
    let end = offsets[r] as usize;
    (bytes[..end].to_vec(), offsets[..=r].to_vec())
}

fn analyze(label: &str, text: &[u8], cap_bytes: Option<usize>) {
    let (bytes, offsets) = rows(text);
    let (bytes, offsets) = match cap_bytes {
        Some(c) => cap(bytes, offsets, c),
        None => (bytes, offsets),
    };
    let n_rows = offsets.len() - 1;
    let total = bytes.len();
    let n_chars = String::from_utf8_lossy(&bytes).chars().count();

    println!("\n════════════════════════════════════════════════════════════════");
    println!("Corpus: {label}");
    println!(
        "  rows = {n_rows}, bytes = {:.2} KiB, chars = {n_chars}, bytes/char = {:.2}",
        total as f64 / 1024.0,
        total as f64 / n_chars.max(1) as f64
    );
    println!("  UTF-8 byte-width mix (over chars): {}", utf8_mix(&bytes));

    let mut codes16: Vec<u16> = Vec::new();
    for &b in BITS {
        let cfg = Config {
            bits: Bits::new(b).expect("9..=16"),
            threshold: Threshold::new(0.2).expect("in range"),
            seed: Some(42),
        };
        let col = compress(&bytes, &offsets, cfg).expect("compress");
        let parts = col.as_parts();

        // Roundtrip check.
        let decoded = decompress(parts);
        assert!(decoded == bytes, "roundtrip mismatch ({label}, {b}-bit)");

        // Compressed-size breakdown (plain interchange form, §3-4 of the spec).
        let dict_b = parts.dict_bytes.len();
        let dict_off = parts.dict_offsets.len() * 4;
        let codes_b = parts.codes.len() * 2;
        let row_off = col.code_offsets.len() * std::mem::size_of::<u64>();
        let compressed = dict_b + dict_off + codes_b + row_off;

        let n_tokens = parts.dict_offsets.len() - 1;
        let m = parts.codes.len();
        let cap = 1usize << b;
        // Realistic stored size: pack each code to `b` bits instead of plain u16.
        let codes_packed = (m * b as usize).div_ceil(8);

        println!("\n  ── {b}-bit ──");
        println!(
            "  codes-only ratio = {:.3}x (u16 codes) | {:.3}x (bit-packed {b}b codes) | {:.3}x incl. row offsets",
            total as f64 / (dict_b + dict_off + codes_b) as f64,
            total as f64 / (dict_b + dict_off + codes_packed) as f64,
            total as f64 / compressed as f64,
        );
        println!(
            "  size: dict_bytes {} B + dict_offsets {} B + codes {} B + row_offsets {} B = {:.1} KiB",
            dict_b,
            dict_off,
            codes_b,
            row_off,
            compressed as f64 / 1024.0
        );
        println!(
            "  codes M = {m}  (mean bytes/code = {:.2}; this is the compression driver)",
            total as f64 / m.max(1) as f64
        );
        symbol_table_stats(parts.dict_bytes, parts.dict_offsets, n_tokens, cap);
        if b == 16 {
            codes16 = parts.codes.to_vec();
        }
    }

    zstd_compare(total, &bytes, &codes16);
}

/// General-purpose baseline: zstd at medium (9) and high (19) on the *same*
/// bytes OnPair sees, plus zstd applied to OnPair's 16-bit code stream (the
/// "dictionary front-end + entropy back-end" combination).
fn zstd_compare(total: usize, bytes: &[u8], codes16: &[u16]) {
    println!("\n  ── zstd (same input bytes) ──");
    for level in [9, 19] {
        let t = std::time::Instant::now();
        let z = zstd::bulk::compress(bytes, level).expect("zstd");
        let secs = t.elapsed().as_secs_f64();
        let tag = if level == 9 { "medium" } else { "high" };
        println!(
            "  zstd -{level:<2} ({tag}): ratio = {:.3}x  ({:.1} KiB, {:.0} MiB/s)",
            total as f64 / z.len() as f64,
            z.len() as f64 / 1024.0,
            total as f64 / (1024.0 * 1024.0) / secs.max(1e-9),
        );
    }
    // OnPair-16 codes as little-endian bytes, then zstd -19 — a stand-in for
    // storing the code stream with an entropy back-end.
    let mut code_bytes = Vec::with_capacity(codes16.len() * 2);
    for &c in codes16 {
        code_bytes.extend_from_slice(&c.to_le_bytes());
    }
    let z = zstd::bulk::compress(&code_bytes, 19).expect("zstd");
    println!(
        "  OnPair-16 codes + zstd -19: codes-only ratio ≈ {:.3}x  (codes {:.1} KiB → {:.1} KiB)",
        total as f64 / z.len() as f64,
        code_bytes.len() as f64 / 1024.0,
        z.len() as f64 / 1024.0,
    );
}

/// Distribution of the dictionary ("symbol table").
fn symbol_table_stats(dict_bytes: &[u8], dict_offsets: &[u32], n_tokens: usize, cap: usize) {
    // Token byte-length histogram and UTF-8 alignment.
    let mut len_hist = [0usize; 17]; // index 1..=16
    let mut learned_len_sum = 0usize; // tokens longer than 1 byte
    let mut learned = 0usize;
    let mut utf8_aligned = 0usize; // token decodes as whole UTF-8 chars
    let mut cp_hist = [0usize; 6]; // whole-codepoint count: 0(non-utf8),1,2,3,4,5+
    for i in 0..n_tokens {
        let s = dict_offsets[i] as usize;
        let e = dict_offsets[i + 1] as usize;
        let tok = &dict_bytes[s..e];
        let len = tok.len().min(16);
        len_hist[len] += 1;
        if tok.len() > 1 {
            learned += 1;
            learned_len_sum += tok.len();
        }
        match std::str::from_utf8(tok) {
            Ok(st) => {
                utf8_aligned += 1;
                let c = st.chars().count().min(5);
                cp_hist[c] += 1;
            }
            Err(_) => cp_hist[0] += 1,
        }
    }

    println!(
        "  dict tokens N = {n_tokens} / {cap} ({:.0}% full); learned (>1B) = {} (mean {:.2} B)",
        100.0 * n_tokens as f64 / cap as f64,
        learned,
        learned_len_sum as f64 / learned.max(1) as f64
    );
    // Compact byte-length histogram.
    let bucket = |lo: usize, hi: usize| (lo..=hi).map(|l| len_hist[l]).sum::<usize>();
    println!(
        "  token byte-len: 1B={} 2B={} 3B={} 4B={} 5-8B={} 9-16B={}",
        len_hist[1],
        len_hist[2],
        len_hist[3],
        len_hist[4],
        bucket(5, 8),
        bucket(9, 16),
    );
    println!(
        "  UTF-8 char-aligned tokens = {utf8_aligned}/{n_tokens} ({:.0}%); \
         whole-chars/token: 1={} 2={} 3={} 4={} 5+={}  (non-aligned={})",
        100.0 * utf8_aligned as f64 / n_tokens as f64,
        cp_hist[1],
        cp_hist[2],
        cp_hist[3],
        cp_hist[4],
        cp_hist[5],
        cp_hist[0],
    );
}

/// Percent of characters encoded as 1/2/3/4 UTF-8 bytes.
fn utf8_mix(bytes: &[u8]) -> String {
    let s = String::from_utf8_lossy(bytes);
    let mut h = [0usize; 5];
    let mut n = 0usize;
    for c in s.chars() {
        h[c.len_utf8()] += 1;
        n += 1;
    }
    let n = n.max(1) as f64;
    format!(
        "1B={:.1}% 2B={:.1}% 3B={:.1}% 4B={:.1}%",
        100.0 * h[1] as f64 / n,
        100.0 * h[2] as f64 / n,
        100.0 * h[3] as f64 / n,
        100.0 * h[4] as f64 / n,
    )
}
