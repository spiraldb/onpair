# OnPairView — handover

State of the `onpair::onpairview` module (the view-shaped sibling of the flat
OnPair decode). Read this first.

## Context / scope correction

This is the **standalone `spiraldb/onpair` crate**, not the `vortex` monorepo.
There is no `vortex-onpair` / `vortex-array` / `varbinview` / `vortex-file` here,
and no Arrow `VarBinView` machinery to plug into. OnPairView was therefore built
**net-new against this crate's own decode path**, keeping the *idea* of the
original task (a per-row, random-access "view" output and a `build_views` "make
view" kernel) but expressed in this crate's types.

## What exists

`src/onpairview/mod.rs` (public, namespaced under `onpair::onpairview`):

- `DecodedView { values: Vec<u8>, offsets: Vec<u32> }` — a decoded column in
  view shape (Arrow VarBin / ListView layout): flat bytes + `R+1` per-row
  offsets. `.len()`, `.is_empty()`, `.row(r)`.
- `BinaryView` — a 16-byte **Arrow-compatible** StringView/BinaryView descriptor
  (`#[repr(transparent)] [u8;16]`, little-endian). `INLINE_LEN = 12`. Values
  ≤ 12 bytes are stored inline; longer values reference a buffer.
  `inline`/`reference` constructors; `len`/`is_inline`/`inline_bytes`/
  `reference_location`/`to_le_bytes`/`resolve` accessors. A `&[BinaryView]`
  reinterprets byte-for-byte as an Arrow view buffer.
- `decompress_view(parts, code_offsets)` — decode `values` (== `decompress`) plus
  per-row `offsets`, recovering row boundaries from the compressor's
  `code_offsets` (a token may straddle a row boundary, so rows can't be recovered
  from codes alone). Builds the fat table once and shares it: the per-row offset
  prefix sum reads its compact `u8` `lens` (`row_offsets_from_lens`, which also
  yields the total length and validates the code range), then an *unchecked*
  decode (`fat::decode_loop`) reuses the same table. Validates the dictionary
  once up front. Tight allocation (no over-copy capacity).
- `row_byte_offsets(parts, code_offsets)` — the per-row offset prefix sum alone
  (no values).
- `build_views(view)` / `build_views_into(view, &mut out)` — one `BinaryView`
  per row; `_into` reuses a caller buffer (real export loops want this).
- `Column::decompress_view(&self)` convenience method (`src/column.rs`).

Tests: `src/onpairview/tests.rs` (10 tests, incl. an oracle cross-check of the
optimized `build_views` against a naive scalar builder across the fast/tail
boundary and tiny (<16 byte) buffers, and an explicit Arrow-layout check).

Bench: `benches/view_compute.rs` (divan), registered in `Cargo.toml`.

## The optimized `build_views` kernel

The per-row "make view" pass is the dominant short-string export cost (the
"export ceiling" from the original task). `build_views_into` assembles each
16-byte descriptor as a single `u128` (one unaligned 16-byte load + mask/shift/or
+ one store) and writes through a raw pointer, skipping per-`push` checks and the
old per-row zero-init + `copy_from_slice`. Rows in the final 16 bytes of `values`
fall to a scalar tail (a 16-byte over-read would run off the buffer); buffers
shorter than 16 bytes are entirely scalar. See `make_view_u128`.

### Measured (buffer reuse; A/B vs the `build_views_scalar` arm)

Per-row time (median). Absolute numbers drift with container load, so the robust
statistic is the **ratio** — same harness, only the called fn differs. Stable to
~1 % across runs:

| corpus                     | scalar    | u128 kernel | speedup |
|----------------------------|-----------|-------------|---------|
| url_short (reference-heavy)| ~570 µs   | ~445 µs     | 1.28×   |
| words (inline-heavy)       | ~1303 µs  | ~637 µs     | 2.05×   |

Inline-heavy wins most: the scalar inline path zero-inits 16 bytes then does a
variable-length `copy_from_slice` (memcpy) per row; the u128 path is one load +
mask/shift/or + one store, no zero-init, no memcpy. The `build_views_scalar`
bench arm is kept as the regression guard that proves this stays won.

## Measurement gotchas (important)

- **Container is noisy and has no perf isolation.** Per-`(param)` absolute
  numbers swing 2–3× by run order; trust the **scalar-vs-u128 ratio** (same
  harness, only the called fn differs) over absolute times, and re-run filtered
  and in isolation (`cargo bench --bench view_compute build_views`) with a larger
  `--sample-size`. (`build_views` runs on the decoded view, which is independent
  of the code-width bits — that is why the param matrix carries only one bit
  width per corpus; a second would be a pure duplicate for these benches.)
- **Always measure `build_views` with buffer reuse** (`build_views_into` + a
  pre-allocated `out`). Allocating the ~3 MB descriptor `Vec` per iteration is
  page-fault-bound and masks the kernel; the first allocator user in a process
  pays cold-page cost, so a non-reusing bench looks ~3–5× slower in isolation
  than after other benches have warmed the allocator.

## Verify

```
RUSTC_WRAPPER= cargo build
RUSTC_WRAPPER= cargo test --lib                 # 102 pass
RUSTC_WRAPPER= cargo clippy --all-targets       # clean
RUSTC_WRAPPER= cargo +nightly fmt --check       # clean
RUSTC_WRAPPER= cargo bench --bench view_compute build_views_only \
    -- --sample-count 20 --sample-size 50
```

(`RUSTC_WRAPPER=` only needed if sccache errors in the sandbox.)

## Measured negative results — do NOT re-try

- **`decompress_view`: fuse offsets into the decode (true single pass).** Decode
  row-by-row into a buffer over-allocated by 16 bytes (so every token over-copies
  with no exact-tail branch) and capture each row's offset as the write cursor —
  one pass instead of two. Measured only +3 % (words) … +8 % (url_short), and it
  over-allocates `values` to `codes.len() * 16` (up to ~4× the real size). The
  tight fat-shared two-pass below beat it on speed *and* memory, so the fused
  loop was removed. (The short-row case loses the gain to per-row loop overhead.)
- **`build_views`: carry `start = previous end` to read each offset once.** The
  two per-row offset reads (`offsets[r]`, `offsets[r+1]`) look redundant, but
  carrying `end` forward serializes the loop (a loop-carried dependency through
  `start`) and measured *slower*: url_short 354→447 µs, words 496→820 µs. The
  independent loads have no cross-iteration dependency, so the out-of-order
  engine keeps many iterations in flight. The kernel is near store-bandwidth
  bound; removing the "redundant" load loses ILP. Reverted.

## Done this round (perf)

- **`decompress_view`: fat-shared two-pass (tight).** Build the fat token table
  once; the offset prefix sum reads its compact `u8` `lens` (1 byte/code, ≤64 KB
  table) instead of `dict_offsets` (2×`u32`/code, ≤256 KB), and the decode reuses
  the same table. Controlled single-binary A/B vs the previous two-pass
  (`decompress_view_prev_2pass` bench arm), median over 3 runs: **~6 % faster
  (url_short), ~20 % faster (words)** — the `lens` table is far more cache-
  resident at bits=16 where `words` has many codes/byte. Tight allocation.
- **`decompress_view`: 3 passes → 2.** It used to call `crate::decompress`
  (which itself walks all codes once for `decompressed_len` to size the buffer,
  then again to decode with per-code bounds checks) and *then* `row_byte_offsets`
  — three full passes computing the same length sums twice. Now: validate the
  dictionary once, `row_byte_offsets` (one pass; gives offsets + total + code
  range validation), then an *unchecked* decode (one pass, no per-code branch).
  View overhead vs the flat decode dropped from ~35 % (url_short) / ~90 %
  (words) to ~5 % / ~30 %.

## Not done / next ideas (unmeasured — measure before keeping)

- **`decompress_view_into` / random single-row access.** Today `decompress_view`
  always materializes the whole `values` buffer + offsets. A single-row decode
  (`row r` only, via `code_offsets[r]..code_offsets[r+1]`) would make true random
  access O(row), not O(column) — arguably the headline reason a "view" exists.
- **Fuse offset computation into the decode loop.** `decompress_view` currently
  does two passes (decode, then `row_byte_offsets`). Recording the write cursor
  at each row boundary inside the decode could fuse them — but the decode loop is
  carefully tuned (16-byte over-copy, split fast/tail region), so interleaving
  row-boundary work risks regressing it. Measure against `decompress_flat`.
- **Branchless inline/reference select in `make_view_u128`.** The split is a
  data-dependent branch (well-predicted within a column). A `select`-style
  computation of both arms was *not* tried; only worth it if a mixed-length
  corpus shows branch-miss cost.
- **No `serde`/on-disk format.** The original task's "register in vortex-file +
  serde round-trip" has no analog here (no vortex-file, crate has no serde dep).
  `BinaryView::to_le_bytes` already gives a stable Arrow-compatible wire form;
  the Arrow-layout test covers it.
