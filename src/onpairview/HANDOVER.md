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
  from codes alone). Two passes: `row_byte_offsets` (yields the offsets, the total
  length, and validates the code range by indexing the dictionary), then a reuse
  of the public `decompress_into_unchecked`. Validates the dictionary once up
  front via the shared `decompress::assert_valid_dictionary`.
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
RUSTC_WRAPPER= cargo test --lib                 # 101 pass
RUSTC_WRAPPER= cargo clippy --all-targets       # clean
RUSTC_WRAPPER= cargo +nightly fmt --check       # clean
RUSTC_WRAPPER= cargo bench --bench view_compute build_views_only \
    -- --sample-count 20 --sample-size 50
```

(`RUSTC_WRAPPER=` only needed if sccache errors in the sandbox.)

## Measured negative results — do NOT re-try

- **`decompress_view`: share one fat table, sum offsets from its `u8` `lens`.**
  Build the fat table first, prefix-sum offsets from its compact `lens` (≤64 KB)
  instead of `dict_offsets` (≤256 KB), then decode reusing the table. *Looked*
  like a 6 %/20 % win in one low-load run, but more samples showed it ~2–5 %
  **slower** under load: building the table first lets the offset pass (streaming
  ~1 MB of codes) evict `fat.data` from cache, so the decode reloads it cold. The
  current `decompress_into_unchecked`-based order (offsets, *then* build+decode
  back-to-back) keeps `fat.data` hot for the decode. Reverted — within noise, and
  the simpler version reuses an existing fn (no `FatTable::lens`, no fat plumbing
  in `onpairview`). **Lesson:** for these decode benches the container noise
  envelope is ±10 %; don't trust a single run.
- **`decompress_view`: fuse offsets into the decode (true single pass).** Decode
  row-by-row into a buffer over-allocated by 16 bytes (so every token over-copies
  with no exact-tail branch) and capture each row's offset as the write cursor —
  one pass instead of two. Measured only +3 % (words) … +8 % (url_short) in a
  favourable run, and it over-allocates `values` to `codes.len() * 16` (up to ~4×
  the real size). Not worth the memory; removed. (Short rows lose the gain to
  per-row loop overhead.)
- **`build_views`: carry `start = previous end` to read each offset once.** The
  two per-row offset reads (`offsets[r]`, `offsets[r+1]`) look redundant, but
  carrying `end` forward serializes the loop (a loop-carried dependency through
  `start`) and measured *slower*: url_short 354→447 µs, words 496→820 µs. The
  independent loads have no cross-iteration dependency, so the out-of-order
  engine keeps many iterations in flight. The kernel is near store-bandwidth
  bound; removing the "redundant" load loses ILP. Reverted.

## Done this round (perf)

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
