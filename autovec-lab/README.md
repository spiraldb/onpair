# Auto-vectorized prefilter lab

This standalone crate compares portable auto-vectorized cover scans with the
explicit AVX-512 kernels. It is a code-generation lab, not a second production
implementation.

The important result is that const-generic cover shapes let LLVM unroll the
predicate and generate the same useful ZMM compare/mask-reduction sequence as
intrinsics. A normal native build may nevertheless choose YMM vectors due to
the target's 256-bit-width preference. Production therefore keeps the explicit
intrinsic kernel for deterministic ZMM width and runtime feature dispatch.

See [`docs/avx512-sparse-prefilter.md`](../docs/avx512-sparse-prefilter.md) for
the shipped hierarchy, end-to-end results, and reproduction commands.

## Programs

- `point-scan-lab`: synthetic and dumped-column single-point experiments.
- `cover_lab`: scalar oracle, portable const-generic candidates, original
  AVX-512 baseline, and retained-mask block-size comparisons. Set `LAB_ALGO` to
  one displayed name, `LAB_REPS` to change repetitions, or
  `LAB_ALL_SCALARS=1` to include exploratory scalar kernels.
- `mem_bw`: one-core AVX-512 streaming-read ceiling for a raw `u16` code file.

## Build matrix

```sh
cargo build --release --target-dir target-v2
RUSTFLAGS='-C target-cpu=x86-64-v3' cargo build --release --target-dir target-v3
RUSTFLAGS='-C target-cpu=x86-64-v4' cargo build --release --target-dir target-v4
RUSTFLAGS='-C target-cpu=x86-64-v4 -C target-feature=-prefer-256-bit' \
  cargo build --release --target-dir target-v4z
RUSTFLAGS='-C target-cpu=native' cargo build --release --target-dir target-native
```

Run `target-*/release/cover_lab synth`, or use `cover_lab real codes.bin
row_offsets.u32 covers.tsv` for an existing dump. Every measured output is
checked against the scalar row-table oracle.
