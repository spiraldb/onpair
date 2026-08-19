# Auto-vectorized prefilter lab

Standalone benchmark of prefilter scan kernels: the shipped intrinsic kernels
against portable-Rust equivalents that rely on the compiler's auto-vectorizer.
Vendored from the onpair_like iteration branch; runs against synthetic columns
out of the box, and against real compressed columns via the dumps described
below. No dependencies.

## Findings so far (4-core shared Xeon, Skylake-X era AVX-512)

Real clickbench-url-1m covers, geomean ns/code, best of 3, outputs verified
against a scalar oracle:

| build flags                        | av_shape G=256 | av_shp64 G=64 | original intrinsics | best intrinsics (r6g_256) |
|------------------------------------|---------------:|--------------:|--------------------:|--------------------------:|
| default (SSE2 baseline)            | 0.542          | 0.712         | 0.500               | 0.451                      |
| `-C target-cpu=x86-64-v4`          | 0.469          | 0.456         | 0.466               | 0.429                      |
| v4 + `-C target-feature=-prefer-256-bit` | 0.464    | **0.425**     | 0.457               | 0.396                      |

- The auto-vectorizer handles the shape-monomorphized superblock fold at every
  level: `pcmpeqw/por` (SSE2), `vpcmpeqw ymm` (AVX2), `vpcmpeqw k, zmm` +
  `korw` mask reductions (AVX-512).
- LLVM emits 256-bit vectors even at `x86-64-v4` (`prefer-256-bit` tuning);
  subtract that feature to get zmm — that is when the portable kernel passes
  the hand-written original.
- Monomorphization is the enabler: a dynamic `for probe in probes` loop never
  vectorizes (2–9 ns/code); const-generic `(P, R)` probe loops unroll and the
  code loop becomes a clean OR-reduction.
- Autovec cannot do runtime CPU dispatch — the shipped intrinsic kernels keep
  that advantage for a portable binary.

## Layout

- `src/main.rs` — `point-scan-lab`: single-point scan algorithms
  (one-pass/two-pass/SWAR/AVX2/AVX-512), synth + real modes.
- `src/bin/cover_lab.rs` — `cover_lab`: general point/range covers. Kernels:
  `rows_tbl` (row-centric table), `av_c4t`/`av_shape`/`av_shp64` (portable,
  autovec), `ORIGINAL` (shipped AVX-512 generic loop), `r2g_sub` (sub+cmple
  ranges), `r6g_64/256/512` (superblock, retained masks). Set
  `LAB_ALL_SCALARS=1` for the exploratory scalar variants.
- `synth_results.txt`, `real_results.txt` — archived runs from the 4-core VM.

## Benchmarks to run on a bigger machine

Build the matrix once (from this directory):

```sh
cargo build --release --target-dir target-v2
RUSTFLAGS='-C target-cpu=x86-64-v3' cargo build --release --target-dir target-v3
RUSTFLAGS='-C target-cpu=x86-64-v4' cargo build --release --target-dir target-v4
RUSTFLAGS='-C target-cpu=x86-64-v4 -C target-feature=-prefer-256-bit' \
    cargo build --release --target-dir target-v4z
RUSTFLAGS='-C target-cpu=native' cargo build --release --target-dir target-native
```

1. **Synthetic sweeps** (no data needed, ~5 min each):
   `./target-<v>/release/point-scan-lab synth` — single-point algorithms over
   density × row length × layout; `./target-<v>/release/cover_lab synth` —
   cover shapes (P,R) × density.

2. **Real-column kernel shootout** (needs dumps): in the onpair_like repo,
   branch `claude/prefilter-checking-optimize-bflts6`, build
   `scratchpad/prefilter_census` (path-dep on this checkout) and run it with a
   dataset's `payload.bin`/`offsets.u32` plus a `queries.jsonl`; the fourth
   argument dumps `codes.bin`, `row_offsets.u32`, `covers.tsv`. Then:
   `./target-<v>/release/cover_lab real codes.bin row_offsets.u32 covers.tsv`.
   Compare the same build's columns; compare builds only via their own runs.

3. **Library A/B, end to end** (onpair_like repo, same branch):
   `cargo build --release -p lb-harness --no-default-features --features cand-onpair-spiral`
   then `./target/release/bench run specs/onpair-spiral-prefilter.toml -o results/<label>`
   and `python3 scratchpad/summarize_prefilter_run.py results/<before> results/<after>`.
   The committed baseline of the ORIGINAL kernel is `results/onpair-spiral-prefilter-baseline`
   on that branch's machine; regenerate a baseline on the new machine by
   checking this repo out at `5927cce` first.

4. **Three-dataset workload** (onpair_like, `experiments/optimize_prefilter`):
   `./experiments/optimize_prefilter/generate.sh` (needs the datasets ingested
   per `datasets/prepare.py`), then run the generated
   `benchmark-onpair-only.toml` for prefilter-only cells, or `benchmark.sh`
   for the full cross-candidate comparison.

5. **Unit tests / correctness**: `cargo test --release` in this repo — the
   prefilter suite checks every kernel against a scalar oracle and the bail
   path for sound-superset behavior.

Interesting axes on a bigger machine: Ice Lake/Sapphire Rapids (no 512-bit
frequency penalty, 2× mask ports) should widen the zmm-vs-ymm and
superblock-vs-original gaps; higher memory bandwidth raises the ceiling the
sparse scans are currently pinned to (~20 GB/s here).
