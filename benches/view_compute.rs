// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//! OnPairView decode + `build_views` benchmark.
#![allow(
    clippy::cast_possible_truncation,
    clippy::expect_used,
    clippy::missing_panics_doc,
    clippy::unwrap_used
)]
//
// Measures the view-shaped decode path against the flat-decode baseline on a
// self-contained synthetic corpus (no external generators, so it runs fast and
// deterministically):
//
//   * decompress        — flat baseline ([`onpair::decompress`]).
//   * row_byte_offsets  — per-row offset prefix sum alone (no values).
//   * decompress_view   — decode values + per-row offsets.
//   * build_views       — the per-row "make view" kernel over a decoded view;
//                         this is the dominant short-string export cost and the
//                         kernel to optimize.
//
// Run with: cargo bench --bench view_compute

use divan::Bencher;
use divan::counter::BytesCount;
use divan::counter::ItemsCount;
use onpair::onpairview::BinaryView;
use onpair::onpairview::DecodedView;
use onpair::onpairview::build_views_into;
use onpair::onpairview::decompress_row_into;
use onpair::onpairview::decompress_view;
use onpair::onpairview::row_byte_offsets;
use onpair::{Bits, Column, Config, Threshold, compress, decompress};

// (corpus, code-width bits). `url_short` is reference-heavy (> 12 byte rows),
// `words` inline-heavy (≤ 12 byte rows), so `build_views` exercises both arms of
// its split. Only one bit width per corpus: the `build_views` benches operate on
// the decoded view, which is identical across bit widths, so a second width
// would be a pure duplicate there.
const PARAMS: &[(&str, u8)] = &[("url_short", 16), ("words", 16)];

const ROWS: usize = 200_000;

struct Corpus {
    bytes: Vec<u8>,
    offsets: Vec<u32>,
    total_bytes: usize,
    rows: usize,
}

/// Build a deterministic corpus of short strings. `kind` picks the character of
/// the rows; both mix inline (≤ 12 byte) and reference (> 12 byte) rows so
/// `build_views` exercises its inline/reference split.
fn corpus(kind: &str) -> Corpus {
    let mut bytes = Vec::new();
    let mut offsets = vec![0u32];
    for i in 0..ROWS {
        let row = match kind {
            // Free-text-ish words with heavy reuse; many rows ≤ 12 bytes.
            "words" => {
                const W: &[&str] = &[
                    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "and", "then",
                ];
                format!("{} {}", W[i % W.len()], W[(i * 7) % W.len()])
            }
            // Patterned URLs with shared prefixes; mostly > 12 bytes (reference).
            _ => format!("https://ex.com/{}/{}", i % 97, i % 13),
        };
        bytes.extend_from_slice(row.as_bytes());
        offsets.push(bytes.len() as u32);
    }
    let total_bytes = bytes.len();
    Corpus {
        bytes,
        offsets,
        total_bytes,
        rows: ROWS,
    }
}

fn build_column(kind: &'static str, bits: u8) -> (Corpus, Column<u32>) {
    let c = corpus(kind);
    let cfg = Config {
        bits: Bits::new(bits).unwrap(),
        threshold: Threshold::new(0.2).unwrap(),
        seed: Some(42),
    };
    let col = compress(&c.bytes, &c.offsets, cfg).unwrap();
    (c, col)
}

#[divan::bench(args = PARAMS)]
fn decompress_flat(bencher: Bencher, param: (&'static str, u8)) {
    let (kind, bits) = param;
    let (c, col) = build_column(kind, bits);
    bencher
        .counter(BytesCount::new(c.total_bytes))
        .bench(|| divan::black_box(decompress(col.as_parts())));
}

#[divan::bench(args = PARAMS)]
fn row_offsets(bencher: Bencher, param: (&'static str, u8)) {
    let (kind, bits) = param;
    let (c, col) = build_column(kind, bits);
    bencher
        .counter(ItemsCount::new(c.rows))
        .bench(|| divan::black_box(row_byte_offsets(col.as_parts(), &col.code_offsets)));
}

#[divan::bench(args = PARAMS)]
fn decompress_view_full(bencher: Bencher, param: (&'static str, u8)) {
    let (kind, bits) = param;
    let (c, col) = build_column(kind, bits);
    bencher
        .counter(BytesCount::new(c.total_bytes))
        .bench(|| divan::black_box(decompress_view(col.as_parts(), &col.code_offsets)));
}

/// Number of scattered rows fetched by the random-access benches.
const SAMPLE_ROWS: usize = 1000;

/// `SAMPLE_ROWS` row indices spread across the column (a coprime stride so they
/// don't cluster), reused by both random-access arms.
fn scattered_indices(rows: usize) -> Vec<usize> {
    (0..SAMPLE_ROWS).map(|k| (k * 7919) % rows).collect()
}

#[divan::bench(args = PARAMS)]
fn random_rows_single(bencher: Bencher, param: (&'static str, u8)) {
    let (kind, bits) = param;
    let (_, col) = build_column(kind, bits);
    let idx = scattered_indices(col.code_offsets.len() - 1);
    let mut buf = Vec::new();
    // Fetch a sparse set of rows one at a time — O(row) each, no full decode.
    bencher.counter(ItemsCount::new(idx.len())).bench_local(|| {
        for &r in &idx {
            decompress_row_into(col.as_parts(), &col.code_offsets, r, &mut buf);
            divan::black_box(&buf);
        }
    });
}

#[divan::bench(args = PARAMS)]
fn random_rows_via_full(bencher: Bencher, param: (&'static str, u8)) {
    let (kind, bits) = param;
    let (_, col) = build_column(kind, bits);
    let idx = scattered_indices(col.code_offsets.len() - 1);
    // The same sparse fetch, but paying a whole-column decode first — the cost a
    // view encoding avoids for sparse random access.
    bencher.counter(ItemsCount::new(idx.len())).bench(|| {
        let view = decompress_view(col.as_parts(), &col.code_offsets);
        for &r in &idx {
            divan::black_box(view.row(r));
        }
    });
}

#[divan::bench(args = PARAMS)]
fn build_views_only(bencher: Bencher, param: (&'static str, u8)) {
    let (kind, bits) = param;
    let (c, col) = build_column(kind, bits);
    // The decoded view is the input to the make-view kernel; build it once,
    // outside the timed region, so only the per-row descriptor work is measured.
    let view: DecodedView = decompress_view(col.as_parts(), &col.code_offsets);
    // Reuse one output buffer across iterations so the timed region measures the
    // per-row "make view" CPU, not repeated allocation/page-faulting of the
    // descriptor buffer (which otherwise dominates and masks the kernel).
    let mut out: Vec<BinaryView> = Vec::with_capacity(view.len());
    bencher.counter(ItemsCount::new(c.rows)).bench_local(|| {
        build_views_into(divan::black_box(&view), divan::black_box(&mut out));
    });
}

/// Naive per-row reference builder — the pre-optimization shape (zero-init a
/// 16-byte descriptor, then `copy_from_slice`), kept here only to A/B against the
/// `u128` `build_views_into` kernel under identical buffer-reuse conditions.
fn build_views_scalar_into(view: &DecodedView, out: &mut Vec<BinaryView>) {
    out.clear();
    for r in 0..view.len() {
        let bytes = view.row(r);
        let bv = if bytes.len() <= BinaryView::INLINE_LEN {
            BinaryView::inline(bytes)
        } else {
            let prefix: [u8; 4] = bytes[..4].try_into().unwrap();
            BinaryView::reference(bytes.len() as u32, prefix, 0, view.offsets[r])
        };
        out.push(bv);
    }
}

#[divan::bench(args = PARAMS)]
fn build_views_scalar(bencher: Bencher, param: (&'static str, u8)) {
    let (kind, bits) = param;
    let (c, col) = build_column(kind, bits);
    let view: DecodedView = decompress_view(col.as_parts(), &col.code_offsets);
    let mut out: Vec<BinaryView> = Vec::with_capacity(view.len());
    bencher.counter(ItemsCount::new(c.rows)).bench_local(|| {
        build_views_scalar_into(divan::black_box(&view), divan::black_box(&mut out));
    });
}

fn main() {
    divan::main();
}
