# Sparse AVX-512 prefilter hierarchy

This change adds a low-selectivity AVX-512 path that separates cheap rejection
from exact row materialization. It is intended for Ice Lake and Sapphire Rapids,
where using 512-bit vectors does not carry the large frequency transition cost
seen on older AVX-512 processors.

The current production choice is the explicit-intrinsic 512-code hierarchy.
See [the full 99,997,497-row ClickBench rerun](clickbench-full-prefilter-results.md)
for the original AVX2/AVX-512, autovec, intrinsic, block-size, assembly, and
counter comparison that validates this choice.

## Design

The policy selects this path only when the cover occupies less than 0.1% of the
token dictionary. Common cover shapes are const-generic specializations; other
shapes use the same algorithm with dynamic probe arrays.

For every 512 compressed `u16` codes, stage 1 executes sixteen 32-lane AVX-512
predicate evaluations and OR-reduces their masks to one bit. Sixty-four block
bits are packed into a `u64`, so the temporary summary costs one byte per 4,096
codes (one byte per 8 KiB of compressed input).

Stage 2 visits only live blocks and repeats the vector predicate to recover the
exact code-position masks. `RowSink` maps those masks to rows. Stage 2 is exact
with respect to the token cover: the hierarchy introduces no additional rows
beyond those emitted by the previous prefilter. The subsequent KMP or `memmem`
verification is still required because a token cover is a necessary condition
for a substring match, not the substring match itself.

The hot predicate uses:

- one `vpcmpeqw` mask comparison per point;
- one `vpsubw` plus unsigned `vpcmpuw` per inclusive range;
- masked compare chaining for fixed shapes, leaving a final complement rather
  than serially moving every mask through a general-purpose register.

The regular AVX-512 retained-mask path remains for denser covers, where a second
scan would cost more than retaining and consuming masks immediately. AVX2
remains the runtime fallback for machines without AVX-512BW. Wide covers retain
the membership-table paths, and one-point/one-range dense covers have small
fixed AVX-512 leaves.

## Why explicit intrinsics are shipped

The portable const-generic loop and the explicit intrinsic loop produced the
same useful ZMM predicate sequence when LLVM was forced to use 512-bit vectors.
With normal `target-cpu=native` tuning, however, LLVM preferred YMM vectors on
the test host because AVX-512VL makes that legal and the target advertises a
256-bit preference. Auto-vectorization also cannot provide the library's
runtime feature boundary.

Consequently production uses one intrinsic implementation: it guarantees ZMM
width, keeps runtime dispatch, and avoids maintaining two equivalent kernels.
The auto-vectorized form stays in `autovec-lab` as a compiler/code-generation
experiment, not as a second production implementation.

## Block-size result

The end-to-end benchmark swept 128, 256, 512, 1,024, 2,048, and 4,096 codes per
summary bit. A fixed size of 512 was the most robust compromise across the
tested corpora. Smaller blocks reduce stage-2 false positives but increase
summary reduction and branch overhead. Larger blocks approach a full second
scan as spatially clustered hits make most large blocks live.

Global token frequency alone did not predict the winning size on ClickBench:
equal-frequency needles had different spatial clustering. An adaptive policy
would need sampled block occupancy or stored spatial statistics, so this patch
does not add a misleading frequency-only selector.

## End-to-end results

Measurements used one pinned core, release builds, best of five runs, mined
low-selectivity two-point/two-range covers, exact stage 2, and final KMP
verification. Throughput is decoded (uncompressed string) bytes divided by
end-to-end time.

| Corpus | Rows | Decoded bytes | Original AVX2 | Original AVX-512 | 512 hierarchy |
|---|---:|---:|---:|---:|---:|
| ClickBench URL | 6,000,000 | 523,587,550 | 13.84 GB/s | 25.65 GB/s | 41.55 GB/s |
| Sentiment140 text | 1,600,000 | 118,683,057 | 13.51 GB/s | 23.30 GB/s | 30.90 GB/s |
| News headlines | 1,226,258 | 50,464,366 | 21.68 GB/s | 31.80 GB/s | 40.70 GB/s |
| Amazon book titles | 158,864 | 8,238,381 | 20.97 GB/s | 31.86 GB/s | 42.38 GB/s |
| Amazon book reviews | 893,196 | 402,331,408 | 9.54 GB/s | 19.27 GB/s | 26.57 GB/s |

Across 77 mined needles, the hierarchy won 77/77. Its geometric-mean speedup
was 2.367x over the original AVX2 path and 1.390x over the original AVX-512
path. The AVX-512 speedup range was 1.065x to 2.041x. These are workload and
machine measurements, not API guarantees.

For the one-million-row ClickBench `google` query, the 19,026,058-byte code
buffer was summarized in 0.642 ms, or 29.62 GB/s of compressed input. The
independent streaming-load ceiling was 30.35 GB/s, so stage 1 reached 97.6% of
the measured one-core read bandwidth. Exact live-block refinement cost another
0.121 ms; final KMP brought the hierarchy to 0.778 ms, versus 1.190 ms for the
original AVX-512 scan plus KMP. This ratio is the useful bandwidth comparison:
decoded GB/s can exceed physical memory bandwidth because OnPair scans the
smaller compressed code stream.

## Reproducing the measurements

The data files are deliberately ignored. Prepare a UTF-8 Parquet string column
and a compressed dump:

```sh
ONPAIR_BENCH_PARQUET=/path/to/data.parquet \
ONPAIR_BENCH_COLUMN=URL \
ONPAIR_PF_DUMP=/path/to/dump \
cargo run --release --example prefilter_e2e -- prepare
```

Run the original AVX2/AVX-512 and all hierarchy sizes on one core:

```sh
ONPAIR_BENCH_PARQUET=/path/to/data.parquet \
ONPAIR_BENCH_COLUMN=URL \
ONPAIR_PF_DUMP=/path/to/dump \
ONPAIR_MINE_NEEDLES=1 ONPAIR_MINE_DENSE=1 \
ONPAIR_P2R2_ONLY=1 ONPAIR_PF_LOW_SELECTIVITY=1 \
ONPAIR_PF_ORIGINAL_ALL=1 ONPAIR_HIER_ALL=1 ONPAIR_HIER_ONLY=1 \
ONPAIR_BENCH_REPS=5 \
taskset -c 0 cargo run --release --example prefilter_e2e -- run
```

The harness reports stage 1, exact refinement, KMP, `memmem`, candidate rows,
live blocks, runtime, and decoded GB/s. The copies of the original AVX2 and
AVX-512 kernels in this harness are benchmark baselines only.

Measure the one-core code-buffer read ceiling independently:

```sh
taskset -c 0 cargo run --release \
  --manifest-path autovec-lab/Cargo.toml --bin mem_bw -- /path/to/codes.u16
```

Inspect generated code with `cargo asm` or `objdump -Cd` on a release binary.
Use identical target flags when comparing auto-vectorized and intrinsic forms;
in particular, forced ZMM autovec requires disabling LLVM's 256-bit preference
on targets that expose it.
