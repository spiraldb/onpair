use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use vortex_onpair_rs::Column;

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

fn main() -> Result<()> {
    let args = parse_args()?;
    let bytes = fs::read(&args.input)
        .with_context(|| format!("read {}", args.input.display()))?;
    let input_bytes = bytes.len();
    let (payload, offsets) = build_payload_and_offsets(&bytes);
    let num_rows = offsets.len().saturating_sub(1);

    let mut compress_ns: Vec<u128> = Vec::with_capacity(args.iters as usize);
    for _ in 0..args.warmup {
        let _ = Column::compress(args.bits, &payload, &offsets);
    }
    let mut last: Option<Column> = None;
    for _ in 0..args.iters {
        let t0 = Instant::now();
        let col = Column::compress(args.bits, &payload, &offsets);
        compress_ns.push(t0.elapsed().as_nanos());
        last = Some(col);
    }
    let col = last.ok_or_else(|| anyhow!("--iters must be >= 1"))?;

    let mut decompress_ns: Vec<u128> = Vec::new();
    if args.decompress {
        let mut scratch: Vec<u8> = Vec::with_capacity(1024);
        for _ in 0..args.warmup {
            for i in 0..num_rows {
                col.decompress_row(i, &mut scratch);
            }
        }
        for _ in 0..args.iters {
            let t0 = Instant::now();
            for i in 0..num_rows {
                col.decompress_row(i, &mut scratch);
            }
            decompress_ns.push(t0.elapsed().as_nanos());
        }
    }

    if args.verify {
        let mut scratch: Vec<u8> = Vec::with_capacity(1024);
        for i in 0..num_rows {
            col.decompress_row(i, &mut scratch);
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
        "dict_size": col.dict_size(),
        "dict_bytes": col.dict_bytes(),
        "codes_bytes": col.codes_bytes(),
        "compressed_bytes": col.compressed_bytes(),
        "compress_ns": compress_ns,
        "decompress_ns": decompress_ns,
    });
    println!("{out}");
    Ok(())
}
