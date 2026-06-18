use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use onpair::{Bits, Column, Config, DEFAULT_CONFIG, compress, decode_into};

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

fn cfg_for(bits: u32) -> Result<Config> {
    let bits = Bits::new(bits as u8).map_err(|_| anyhow!("--bits must be in 9..=16"))?;
    Ok(Config { bits, seed: Some(42), ..DEFAULT_CONFIG })
}

fn dict_size(col: &Column<u32>) -> usize {
    col.dict.num_tokens()
}

fn dict_bytes(col: &Column<u32>) -> usize {
    col.dict.logical_len()
}

/// Bit-packed size of the code stream — what a `bits`-wide store would use.
/// The Rust port stores codes as plain `u16`s, so this is smaller than the
/// in-memory representation. Reported for cross-impl comparability with
/// onpair_cpp's `StoreView::bytes_used()` (which bit-packs at `bit_width`).
fn codes_bytes(col: &Column<u32>, bits: u32) -> usize {
    let total_bits = col.codes.len() * bits as usize;
    total_bits.div_ceil(8)
}

/// dict + bit-packed codes only — excludes the per-row boundary index
/// (the "Store" side-table). Matches what onpair_cpp reports after stripping
/// `StoreView::boundaries` from the view's `bytes_used()`.
fn compressed_bytes(col: &Column<u32>, bits: u32) -> usize {
    dict_bytes(col) + col.dict.offsets.len() * std::mem::size_of::<u32>() + codes_bytes(col, bits)
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let bytes = fs::read(&args.input).with_context(|| format!("read {}", args.input.display()))?;
    let input_bytes = bytes.len();
    let (payload, offsets) = build_payload_and_offsets(&bytes);
    let num_rows = offsets.len().saturating_sub(1);
    let cfg = cfg_for(args.bits)?;

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
    let view = col.view();
    let mut decompress_ns: Vec<u128> = Vec::new();
    if args.decompress {
        // Worst-case decoded length is `codes.len() * MAX_TOKEN_SIZE`, always
        // >= the exact decoded length the kernel writes.
        let mut scratch: Vec<u8> = Vec::with_capacity(view.codes.len() * 16);
        for _ in 0..args.warmup {
            // SAFETY: column codes are in range, dictionary read-padded, scratch
            // sized to the worst-case decoded length.
            let len = unsafe { decode_into(view.codes, view.dict, scratch.spare_capacity_mut()) };
            unsafe { scratch.set_len(len) };
            black_box(scratch.as_slice());
            scratch.clear();
        }
        for _ in 0..args.iters {
            let t0 = Instant::now();
            // SAFETY: as above.
            let len = unsafe { decode_into(view.codes, view.dict, scratch.spare_capacity_mut()) };
            unsafe { scratch.set_len(len) };
            black_box(scratch.as_slice());
            scratch.clear();
            decompress_ns.push(t0.elapsed().as_nanos());
        }
    }

    if args.verify {
        let decoded = view.decompress();
        if decoded.as_slice() != payload.as_slice() {
            eprintln!("verify failed");
            std::process::exit(2);
        }
    }

    let out = serde_json::json!({
        "impl": "rust",
        "bits": args.bits,
        "num_rows": num_rows,
        "input_bytes": input_bytes,
        "dict_size": dict_size(&col),
        "dict_bytes": dict_bytes(&col),
        "codes_bytes": codes_bytes(&col, args.bits),
        "compressed_bytes": compressed_bytes(&col, args.bits),
        "compress_ns": compress_ns,
        "decompress_ns": decompress_ns,
    });
    println!("{out}");
    Ok(())
}
