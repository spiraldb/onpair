use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use onpair::{Column, Config, DEFAULT_CONFIG, compress};

struct Args {
    input: PathBuf,
    bits: u32,
    iters: u32,
    warmup: u32,
    decompress: bool,
    verify: bool,
}

fn parse_args() -> Result<Args> {
    let mut input: Option<PathBuf> = None;
    let mut bits: u32 = 12;
    let mut iters: u32 = 5;
    let mut warmup: u32 = 1;
    let mut decompress = false;
    let mut verify = false;

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--bits" => bits = it.next().context("--bits N")?.parse()?,
            "--iters" => iters = it.next().context("--iters N")?.parse()?,
            "--warmup" => warmup = it.next().context("--warmup N")?.parse()?,
            "--decompress" => decompress = true,
            "--verify" => verify = true,
            s if !s.starts_with("--") => input = Some(PathBuf::from(s)),
            other => return Err(anyhow!("unknown arg: {other}")),
        }
    }
    let input = input.ok_or_else(|| anyhow!("missing input path"))?;
    Ok(Args { input, bits, iters, warmup, decompress, verify })
}

/// Build LF-stripped payload + row offsets. Trailing LF terminates the last
/// row rather than starting an empty one. Embedded LFs aren't supported by
/// this format — the python extractor warns and drops them.
fn build_payload_and_offsets(src: &[u8]) -> (Vec<u8>, Vec<u32>) {
    let mut payload = Vec::with_capacity(src.len());
    let mut offsets = Vec::with_capacity(src.len() / 32 + 2);
    offsets.push(0u32);
    let mut row_start = 0usize;
    for (i, &b) in src.iter().enumerate() {
        if b == b'\n' {
            payload.extend_from_slice(&src[row_start..i]);
            offsets.push(payload.len() as u32);
            row_start = i + 1;
        }
    }
    if row_start < src.len() {
        payload.extend_from_slice(&src[row_start..]);
        offsets.push(payload.len() as u32);
    }
    (payload, offsets)
}

fn cfg_for(bits: u32) -> Config {
    Config { bits, ..DEFAULT_CONFIG }
}

fn decompress_row_into(col: &Column<u32>, row: usize, out: &mut Vec<u8>) {
    let parts = col.as_parts();
    let begin = parts.code_boundaries[row] as usize;
    let end = parts.code_boundaries[row + 1] as usize;
    out.clear();
    for &c in &parts.codes[begin..end] {
        let s = parts.dict_offsets[c as usize] as usize;
        let e = parts.dict_offsets[c as usize + 1] as usize;
        out.extend_from_slice(&parts.dict_bytes[s..e]);
    }
}

fn dict_size(col: &Column<u32>) -> usize {
    col.dict_offsets.len().saturating_sub(1)
}

fn dict_bytes(col: &Column<u32>) -> usize {
    col.dict_bytes.len()
}

/// Bit-packed size of the code stream — what a `bits`-wide store would use.
/// The Rust port stores codes as plain `u16`s, so this is smaller than the
/// in-memory representation. Reported for cross-impl comparability with
/// onpair_cpp's `StoreView::bytes_used()` (which bit-packs).
fn codes_bytes(col: &Column<u32>) -> usize {
    let total_bits = col.codes.len() * col.bits as usize;
    total_bits.div_ceil(8)
}

/// dict + bit-packed codes only — excludes the per-row boundary index
/// (the "Store" side-table). Matches what onpair_cpp reports after stripping
/// `StoreView::boundaries` from the view's `bytes_used()`.
fn compressed_bytes(col: &Column<u32>) -> usize {
    dict_bytes(col) + col.dict_offsets.len() * std::mem::size_of::<u32>() + codes_bytes(col)
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let bytes = fs::read(&args.input)
        .with_context(|| format!("read {}", args.input.display()))?;
    let input_bytes = bytes.len();
    let (payload, offsets) = build_payload_and_offsets(&bytes);
    let num_rows = offsets.len().saturating_sub(1);
    let cfg = cfg_for(args.bits);

    let mut compress_ns: Vec<u128> = Vec::with_capacity(args.iters as usize);
    for _ in 0..args.warmup {
        let _ = compress(&payload, &offsets, cfg)?;
    }
    let mut last: Option<Column<u32>> = None;
    for _ in 0..args.iters {
        let t0 = Instant::now();
        let col = compress(&payload, &offsets, cfg)?;
        compress_ns.push(t0.elapsed().as_nanos());
        last = Some(col);
    }
    let col = last.ok_or_else(|| anyhow!("--iters must be >= 1"))?;

    let mut decompress_ns: Vec<u128> = Vec::new();
    if args.decompress {
        let mut scratch: Vec<u8> = Vec::with_capacity(1024);
        for _ in 0..args.warmup {
            for i in 0..num_rows {
                decompress_row_into(&col, i, &mut scratch);
            }
        }
        for _ in 0..args.iters {
            let t0 = Instant::now();
            for i in 0..num_rows {
                decompress_row_into(&col, i, &mut scratch);
            }
            decompress_ns.push(t0.elapsed().as_nanos());
        }
    }

    if args.verify {
        let mut scratch: Vec<u8> = Vec::with_capacity(1024);
        for i in 0..num_rows {
            decompress_row_into(&col, i, &mut scratch);
            let start = offsets[i] as usize;
            let end = offsets[i + 1] as usize;
            if scratch.as_slice() != &payload[start..end] {
                eprintln!("verify failed at row {i}");
                std::process::exit(2);
            }
        }
    }

    let out = serde_json::json!({
        "impl": "rust",
        "bits": args.bits,
        "num_rows": num_rows,
        "input_bytes": input_bytes,
        "dict_size": dict_size(&col),
        "dict_bytes": dict_bytes(&col),
        "codes_bytes": codes_bytes(&col),
        "compressed_bytes": compressed_bytes(&col),
        "compress_ns": compress_ns,
        "decompress_ns": decompress_ns,
    });
    println!("{out}");
    Ok(())
}
