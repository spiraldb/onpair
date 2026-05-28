// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![allow(
    clippy::cast_possible_truncation,
    clippy::expect_used,
    clippy::missing_panics_doc,
    clippy::unwrap_used
)]
//
// End-to-end OnPair benchmark on real TPC-H `lineitem.l_comment` data.
// Data is generated in-memory at startup via `tpchgen`/`tpchgen-arrow` and
// cached in a `OnceLock` so the bench groups don't pay the gen cost.
//
// Env:
//   * `ONPAIR_BENCH_MAX_BYTES`    — cap the corpus (default 256 MiB)
//   * `ONPAIR_BENCH_SCALE_FACTOR` — TPC-H scale factor (default 8.0)
//
// Run with: cargo bench --bench tpch
//
// This bench targets the slim public API in PUBLIC_API.md
// (`compress` / `decompress` free fns + `Column::as_parts()`). It will
// compile once those land in `src/`.

use std::env;
use std::sync::OnceLock;

use arrow_array::cast::AsArray;
use divan::Bencher;
use onpair::Column;
use onpair::Config;
use onpair::compress;
use onpair::decompress;
use tpchgen::generators::LineItemGenerator;
use tpchgen_arrow::LineItemArrow;
use tpchgen_arrow::RecordBatchIterator;

const BITS_CONFIGS: &[u32] = &[12, 16];

struct Corpus {
    bytes: Vec<u8>,
    offsets: Vec<u64>,
    total_bytes: usize,
}

fn corpus() -> &'static Corpus {
    static CORPUS: OnceLock<Corpus> = OnceLock::new();
    CORPUS.get_or_init(|| {
        let max_bytes = env::var("ONPAIR_BENCH_MAX_BYTES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(256 << 20);
        let sf = env::var("ONPAIR_BENCH_SCALE_FACTOR")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(8.0);

        let (bytes, offsets) = generate_l_comment(sf, max_bytes);
        let total_bytes = bytes.len();
        eprintln!(
            "[onpair tpch bench] corpus: TPC-H l_comment sf={sf}, \
             {} rows, {:.2} MiB",
            offsets.len() - 1,
            total_bytes as f64 / (1024.0 * 1024.0)
        );
        Corpus {
            bytes,
            offsets,
            total_bytes,
        }
    })
}

/// Generate TPC-H `l_comment` values via `tpchgen-arrow`, stopping once the
/// concatenated string bytes hit `max_bytes`.
fn generate_l_comment(scale_factor: f64, max_bytes: usize) -> (Vec<u8>, Vec<u64>) {
    let batches =
        LineItemArrow::new(LineItemGenerator::new(scale_factor, 1, 1)).with_batch_size(8192 * 8);
    let comment_idx = batches
        .schema()
        .fields()
        .iter()
        .position(|f| f.name() == "l_comment")
        .expect("l_comment column");

    let mut bytes = Vec::with_capacity(max_bytes.min(1 << 28));
    let mut offsets: Vec<u64> = vec![0];
    'outer: for batch in batches {
        let col = batch.column(comment_idx).as_string_view();
        for v in col.iter() {
            let s = v.unwrap_or("").as_bytes();
            bytes.extend_from_slice(s);
            offsets.push(bytes.len() as u64);
            if bytes.len() >= max_bytes {
                break 'outer;
            }
        }
    }
    (bytes, offsets)
}

fn compress_column(bits: u32) -> Column<u64> {
    let c = corpus();
    let cfg = Config {
        bits,
        threshold: 0.2,
        seed: 42,
    };
    compress(&c.bytes, &c.offsets, cfg).unwrap()
}

// ─────────────────────────────────────────────────────────────────────────────
// Benches.
// ─────────────────────────────────────────────────────────────────────────────

#[divan::bench(args = BITS_CONFIGS)]
fn train_and_compress(bencher: Bencher, bits: u32) {
    let c = corpus();
    bencher
        .counter(divan::counter::BytesCount::new(c.total_bytes))
        .bench(|| {
            let cfg = Config {
                bits,
                threshold: 0.2,
                seed: 42,
            };
            compress(
                divan::black_box(&c.bytes),
                divan::black_box(&c.offsets),
                cfg,
            )
            .unwrap()
        });
}

#[divan::bench(args = BITS_CONFIGS)]
fn decompress_all(bencher: Bencher, bits: u32) {
    let c = corpus();
    let col = compress_column(bits);
    bencher
        .counter(divan::counter::BytesCount::new(c.total_bytes))
        .bench(|| divan::black_box(decompress(col.as_parts())));
}

fn main() {
    // Touch the corpus so the source line prints before divan begins.
    let _ = corpus();
    divan::main();
}
