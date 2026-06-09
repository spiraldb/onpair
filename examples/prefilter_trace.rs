// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//! Scratch (uncommitted): step-by-step trace of the contains-prefilter
//! compile pipeline for one pattern over one real ClickBench file.
//! Mirrors the logic in `src/query/prefilter.rs`.
#![allow(
    clippy::expect_used,
    clippy::needless_range_loop,
    clippy::unwrap_used,
    clippy::use_debug,
    missing_docs
)]

use std::fs::File;

use arrow_array::cast::AsArray;
use onpair::{Bits, Config, Threshold, compress, query::ContainsSearcher};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

const PATTERN: &[u8] = b"google";

fn esc(b: &[u8]) -> String {
    b.iter()
        .map(|&c| {
            if (0x20..0x7f).contains(&c) {
                (c as char).to_string()
            } else {
                format!("\\x{c:02x}")
            }
        })
        .collect()
}

fn main() {
    // ── Load + compress one real file exactly like the bench ────────────────
    let path = "/tmp/clickbench/hits_0.parquet";
    let file = File::open(path).unwrap();
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let idx = builder
        .parquet_schema()
        .columns()
        .iter()
        .position(|c| c.name().eq_ignore_ascii_case("url"))
        .unwrap();
    let mask = ProjectionMask::leaves(builder.parquet_schema(), [idx]);
    let reader = builder.with_projection(mask).build().unwrap();
    let mut bytes = Vec::new();
    let mut offsets = vec![0u64];
    for batch in reader {
        let batch = batch.unwrap();
        for s in batch.column(0).as_binary::<i32>().iter() {
            bytes.extend_from_slice(s.unwrap_or(b""));
            offsets.push(bytes.len() as u64);
        }
    }
    let cfg = Config {
        bits: Bits::new(16).unwrap(),
        threshold: Threshold::new(0.5).unwrap(),
        seed: Some(42),
    };
    let col = compress(&bytes, &offsets, cfg).unwrap();
    let parts = col.as_parts();
    let ntok = parts.dict_offsets.len() - 1;
    let tok = |c: usize| {
        &parts.dict_bytes[parts.dict_offsets[c] as usize..parts.dict_offsets[c + 1] as usize]
    };
    let m = PATTERN.len();
    println!("== input ==");
    println!(
        "file: {path}  rows={}  raw={:.1} MiB  codes={}  dict tokens={ntok}",
        offsets.len() - 1,
        bytes.len() as f64 / (1024.0 * 1024.0),
        col.codes.len()
    );
    println!("pattern: {:?}  (m={m})", esc(PATTERN));

    // ── Step 1: per-anchor candidate sets (same brute force as prefilter) ───
    println!("\n== step 1: anchor candidate sets C_i ==");
    let mut sets: Vec<Vec<bool>> = vec![vec![false; ntok]; m];
    // Remember one witness alignment per (anchor, code) for display.
    let mut witness: Vec<std::collections::HashMap<usize, isize>> = vec![Default::default(); m];
    for c in 0..ntok {
        let t = tok(c);
        let len = t.len() as isize;
        let mi = m as isize;
        for s in (1 - len)..mi {
            let lo = s.max(0);
            let hi = (s + len).min(mi);
            if (lo..hi).all(|j| t[(j - s) as usize] == PATTERN[j as usize]) {
                for i in lo..hi {
                    sets[i as usize][c] = true;
                    witness[i as usize].entry(c).or_insert(s);
                }
            }
        }
    }
    for (i, set) in sets.iter().enumerate() {
        let n = set.iter().filter(|&&b| b).count();
        println!(
            "  anchor i={i} (P[{i}]='{}'): |C_{i}| = {n} of {ntok} tokens",
            PATTERN[i] as char
        );
    }

    // ── Step 2: sample the real code stream, score each anchor ──────────────
    println!("\n== step 2: sampled per-code hit rate (argmin wins) ==");
    let stride = (col.codes.len() / (1 << 16)).max(1);
    let mut best = (usize::MAX, usize::MAX, 0usize); // (hits, popcount, anchor)
    for (i, set) in sets.iter().enumerate() {
        let mut hits = 0usize;
        let mut n = 0usize;
        for &c in col.codes.iter().step_by(stride) {
            hits += set[c as usize] as usize;
            n += 1;
        }
        let pop = set.iter().filter(|&&b| b).count();
        println!(
            "  anchor i={i}: {hits}/{n} sampled codes hit  ({:.4}%)",
            100.0 * hits as f64 / n as f64
        );
        if (hits, pop) < (best.0, best.1) {
            best = (hits, pop, i);
        }
    }
    let bi = best.2;
    println!(
        "  -> chosen anchor: i={bi} (P[{bi}]='{}')",
        PATTERN[bi] as char
    );

    // ── Step 3: candidate codes -> intervals over the sorted dictionary ─────
    println!("\n== step 3: chosen C_{bi} as code intervals (dict is lex-sorted) ==");
    let set = &sets[bi];
    let mut intervals: Vec<(usize, usize)> = Vec::new();
    let mut run: Option<usize> = None;
    for c in 0..=ntok {
        match (c < ntok && set[c], run) {
            (true, None) => run = Some(c),
            (false, Some(s)) => {
                intervals.push((s, c - 1));
                run = None;
            }
            _ => {}
        }
    }
    println!("  {} intervals:", intervals.len());
    for &(lo, hi) in &intervals {
        let n = hi - lo + 1;
        let mut line = format!("  [{lo:>5}, {hi:>5}] ({n:>3} codes): ");
        let show: Vec<usize> = if n <= 4 {
            (lo..=hi).collect()
        } else {
            vec![lo, lo + 1, hi]
        };
        for (k, &c) in show.iter().enumerate() {
            if k == 2 && n > 4 {
                line.push_str("... ");
            }
            let s = witness[bi][&c];
            line.push_str(&format!("\"{}\"(s={s}) ", esc(tok(c))));
        }
        println!("{line}");
    }

    // ── Step 4: scan the code stream + row mapping + DFA verify ─────────────
    println!("\n== step 4: scan ==");
    let cand_codes = col.codes.iter().filter(|&&c| set[c as usize]).count();
    let mut cand_rows: Vec<usize> = Vec::new();
    for (r, w) in col.code_offsets.windows(2).enumerate() {
        if col.codes[w[0] as usize..w[1] as usize]
            .iter()
            .any(|&c| set[c as usize])
        {
            cand_rows.push(r);
        }
    }
    let searcher = ContainsSearcher::compile(parts, PATTERN);
    let matches = searcher.matching_rows(&col.codes, &col.code_offsets);
    println!(
        "  candidate codes: {cand_codes} / {} ({:.4}%)",
        col.codes.len(),
        100.0 * cand_codes as f64 / col.codes.len() as f64
    );
    println!(
        "  candidate rows:  {} / {} ({:.4}%)   -> DFA verifies only these",
        cand_rows.len(),
        offsets.len() - 1,
        100.0 * cand_rows.len() as f64 / (offsets.len() - 1) as f64
    );
    println!(
        "  true matches:    {} ({:.4}%)",
        matches.len(),
        100.0 * matches.len() as f64 / (offsets.len() - 1) as f64
    );
    println!("  sanity: prefilter_info = {:?}", searcher.prefilter_info());

    // Show one true match and one false positive, with the triggering token.
    let row_bytes = |r: usize| &bytes[offsets[r] as usize..offsets[r + 1] as usize];
    let trigger = |r: usize| {
        let w = &col.code_offsets[r..r + 2];
        col.codes[w[0] as usize..w[1] as usize]
            .iter()
            .find(|&&c| set[c as usize])
            .map(|&c| esc(tok(c as usize)))
            .unwrap()
    };
    if let Some(&r) = matches.first() {
        let r = r as usize;
        let row = row_bytes(r);
        let pos = row.windows(m).position(|w| w == PATTERN).unwrap();
        let (a, b) = (pos.saturating_sub(40), (pos + m + 40).min(row.len()));
        println!("\n  example TRUE match row {r} (match at byte {pos}):");
        println!("    ...{}...", esc(&row[a..b]));
        println!("    trigger = candidate token \"{}\"", trigger(r));
    }
    let match_set: std::collections::HashSet<usize> = matches.iter().map(|&r| r as usize).collect();
    if let Some(&r) = cand_rows.iter().find(|r| !match_set.contains(r)) {
        println!("\n  example FALSE POSITIVE row {r} (killed by DFA verify):");
        println!(
            "    url     = \"{}\"",
            esc(&row_bytes(r)[..row_bytes(r).len().min(100)])
        );
        println!("    trigger = candidate token \"{}\"", trigger(r));
    }
}
