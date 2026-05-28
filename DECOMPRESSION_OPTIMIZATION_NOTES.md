# Decompression Optimization Notes

Date: 2026-05-28

## Goal

Bring Rust decompression closer to the C++ decoder. The target was at least
3 GB/s for full-column decompression, with the main focus on avoiding
row-at-a-time overhead and matching the C++ padded-copy strategy.

## Baseline

The original Rust benchmark decoded row by row into a `Vec<u8>` using
`extend_from_slice` for each token. That path paid for repeated row-boundary
iteration, slice construction, and `Vec` growth checks.

Observed baseline on `/private/tmp/onpair-profile-corpus.txt`:

- Input bytes: 21,968,191
- Rows: 300,000
- Original row loop: about 16.5 ms, roughly 1.3 GB/s

## Main Findings

The C++ decoder is fast because it treats token decode as a tight copy loop:

- Dictionary tokens are bounded at 16 bytes.
- Dictionary bytes are padded after the logical dictionary end.
- The hot loop can copy 16 bytes from every token start, even for shorter
  tokens, then advance output by the token's real length.
- The caller either provides output padding, or the decoder must avoid
  over-copying when it is near the end of the output buffer.

The Rust row loop was not measuring that style of decoder.

For repeatable measurements, the Rust benchmark now uses `Config {
seed: Some(42), .. }`. Public `Config.seed` is `Option<u64>`: `None` means
non-deterministic training, and `Some(seed)` gives a reproducible dictionary.

## Implemented Rust Changes

### Decode all at once

`decompress_into(parts, out: &mut [MaybeUninit<u8>]) -> usize` decodes the whole
column into a caller-provided flat output buffer. It ignores row materialization
because the caller already owns row offsets.

This avoids:

- Allocating per row.
- Clearing and reusing row scratch buffers.
- Repeated row-boundary dispatch in the benchmark.

### Exact caller buffer

The public `decompress_into` API accepts an exact-size output buffer. It does
not require the caller to add output padding.

The checked padded path is split into two phases:

- A fast prefix loop uses padded 16-byte copies while the output cursor is at
  least `MAX_TOKEN_SIZE` bytes away from the end of the caller buffer.
- Once the cursor reaches the final output tail, the decoder breaks to an exact
  copy loop for the remaining tokens.

That removed the earlier decoded-length prepass from `decompress_into`.

The exact-size unchecked path can use an even simpler split: all but the final
`MAX_TOKEN_SIZE` codes use 16-byte padded copies, and the final
`MAX_TOKEN_SIZE` codes use exact copies. Because dictionary tokens are non-empty,
every prefix token has at least `MAX_TOKEN_SIZE` logical bytes after it, so the
wide output write stays inside an exact-size decoded buffer.

### Dictionary padding

`Parser::parse` now pads `Column.dict_bytes` by
`DECOMPRESS_BUFFER_PADDING = MAX_TOKEN_SIZE - 1`.

`dict_offsets.last()` remains the logical dictionary byte length. This matters
for reporting compressed size and for serialization: stored dictionary bytes
should not count the decoder padding unless the storage format explicitly wants
to persist it.

This padding is for dictionary reads, not output writes. The default output
paths now exact-copy their tail and do not require output padding.

### Decode table

`decode_entries(parts)` builds a `Vec<DecodeEntry>`, one entry per dictionary
token. Each entry packs:

- token byte offset
- token byte length

This is different from `dict_offsets`.

`dict_offsets` is a cumulative offsets table. To decode one token from it, the
hot loop needs two loads and a subtract:

```text
start = dict_offsets[code]
end   = dict_offsets[code + 1]
len   = end - start
```

`DecodeEntry` makes that one load:

```text
entry = decode_entries[code]
start = entry.offset
len   = entry.len
```

The table is small compared to the code stream and removes work from the inner
loop.

### Default API path

The fast path is wired into the default whole-column Rust decode APIs:

- `compress(...)` produces padded dictionary bytes.
- `Column::as_parts()` exposes those bytes.
- `decompress_into(...)` detects the padding and uses the checked padded table
  path.
- `decompress(...)` also uses the padded table path after allocating its own
  output buffer.

Caveat: `decompress_row_into(...)` is still the exact checked row path. Also,
an external persisted read path must re-add dictionary padding after
`dict_offsets.last()` or `decompress_into(...)` will fall back to the safe exact
copy path.

The decompression implementation lives in `src/decompress.rs`; `src/lib.rs`
only wires modules, re-exports the public API, and keeps the `compress(...)`
convenience function.

## Hardware Specifics

The 16-byte copy has an AArch64-specific implementation behind:

```rust
#[cfg(target_arch = "aarch64")]
```

That path uses NEON load/store intrinsics. Non-AArch64 targets use a portable
fallback with two unaligned `u64` loads/stores. There is no runtime hardware
feature check.

## Benchmark Results

Command shape:

```sh
RUSTC_WRAPPER= cargo run \
  --manifest-path benchmarks/onpair-bench/rust-bench/Cargo.toml \
  --release -- /private/tmp/onpair-profile-corpus.txt \
  --bits 12 --iters 12 --warmup 4 --decompress --verify
```

Final public `decompress_into` path, exact output capacity, fixed seed:

| bits | best | median | average |
| --- | ---: | ---: | ---: |
| 12 | 1.527 ms, 14.39 GB/s | 1.549 ms, 14.18 GB/s | 1.571 ms, 13.98 GB/s |
| 16 | 1.806 ms, 12.16 GB/s | 1.927 ms, 11.40 GB/s | 1.926 ms, 11.41 GB/s |

The earlier unsafe table-only harness also showed the same order of magnitude,
around 14.7 GB/s for the 12-bit case. The public exact-buffer path is now close
enough that API safety checks are not the dominant issue.

## Flamegraphs

Generated flamegraphs:

- `/private/tmp/onpair-decode-default.svg`
- `/private/tmp/onpair-decode-bulk.svg`
- `/private/tmp/onpair-decode-padded.svg`
- `/private/tmp/onpair-decode-padded-entries.svg`

The final public-default flamegraph centers on:

```text
onpair::decompress_into_checked_padded_with_entries
```

Earlier flamegraphs showed the cost of:

- Row-at-a-time decode.
- Exact small copies for every token.
- Re-reading `dict_offsets[code]` and `dict_offsets[code + 1]`.
- A decoded-length prepass before public `decompress_into`.

## Validation

Commands run successfully:

```sh
cargo test
cargo test --manifest-path benchmarks/onpair-bench/rust-bench/Cargo.toml
cargo clippy --all-targets --all-features
git diff --check
```

The benchmark uses `--verify`, so decoded bytes were checked against the input
payload for the measured runs.

## Remaining Work

- Add or update any Rust persisted read path so it restores dictionary decoder
  padding after reading logical dictionary bytes.
- Consider a row-level padded-table path if random row decompression needs the
  same treatment as whole-column decode.
- Decide whether `DecodeEntry` should be cached on `Column` rather than rebuilt
  per `decompress_into` call for repeated decode of the same column.
- If x86 numbers matter, run the same profile there and inspect the portable
  two-`u64` fallback codegen.
