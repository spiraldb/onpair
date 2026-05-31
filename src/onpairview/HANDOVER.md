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
  from codes alone).
- `row_byte_offsets(parts, code_offsets)` — the per-row offset prefix sum alone
  (no values).
- `build_views(view)` / `build_views_into(view, &mut out)` — one `BinaryView`
  per row; `_into` reuses a caller buffer (real export loops want this).
- `Column::decompress_view(&self)` convenience method (`src/column.rs`).

Tests: `src/onpairview/tests.rs` (13 tests, incl. an oracle cross-check of the
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

### Measured (this container, noisy; `--sample-count 20 --sample-size 50`, buffer reuse)

A/B vs the naive scalar builder (`build_views_scalar` arm in the bench), per-row
throughput, median:

| corpus                    | scalar       | u128 kernel  | speedup |
|---------------------------|--------------|--------------|---------|
| url_short (reference-heavy)| 378 Mitem/s | 565 Mitem/s  | 1.49×   |
| words (inline-heavy)       | 151 Mitem/s | 403 Mitem/s  | 2.67×   |

Inline-heavy wins most (scalar zero-init + variable memcpy per row → one masked
load+store). No regression anywhere.

## Measurement gotchas (important)

- **Container is noisy and has no perf isolation.** Per-`(param)` absolute
  numbers swing 2–3× by run order. Sanity check: `build_views` runs on the
  *decoded view*, which is **identical** for `("url_short",12)` and
  `("url_short",16)` (bits don't affect decoded bytes) — those two arms must
  report the same time. If they don't, it's warmup/ordering noise; re-run the
  bench **filtered and in isolation** (`cargo bench --bench view_compute
  build_views_only`) with a larger `--sample-size`.
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
