# Prefilter algorithms, lessons, and benchmark guide

This is the reproducibility guide for the compressed substring-search
experiments in `examples/prefilter_e2e.rs`. It fixes the algorithm names,
describes what each stage reads and emits, records the conclusions so far, and
provides the benchmark commands used to compare the implementations.

## Data and common terms

The input column consists of:

- a contiguous `u16` token-code stream;
- a token dictionary (`dict.bytes` and `dict.offsets.u32`);
- a monotonic `rows.u64` array mapping rows to code ranges.

The benchmark dump contains `codes.u16`, `dict.bytes`, `dict.offsets.u32`, and
`rows.u64`. It never rewrites query inputs during a timed run. `prepare` creates
this representation from Parquet; `prepare-lines` creates it from newline
delimited values.

For a needle, the alignment DAG selects a cut. Tokens crossing that cut are
represented by exact token IDs (points) and inclusive token-ID ranges. A point
costs one comparison and a range costs two. A token matching this cover is only
a necessary condition for a substring match; KMP or `memmem` is always the
final exact check.

A segment is a consecutive group of complete rows containing at most 2 or 4
MiB of compressed `u16` codes, except when one row itself exceeds the limit.
Each segment is visited once per full-column pass. All stages for that segment
run while its few MiB are cache-resident.

## Exact baselines

### Direct compressed KMP

```text
for each row:
    run token-aware KMP over the row's compressed codes
    emit the row if the needle matches
```

This has no prefilter, candidates, or decompression buffer. It is the usual
winner when the cover is dense or expensive: a prefilter would reread much of
the column and still reject little.

### Decode plus memmem

```text
for each candidate row or localized window:
    decode its tokens into bytes
    run memmem on those bytes
```

`memmem` only receives selected rows or windows in prefiltered pipelines. It
does not decompress the entire column unless it is used as the direct
baseline. On the measured high-selectivity ClickBench suite, direct compressed
KMP was generally faster because avoiding decompression outweighed `memmem`'s
byte-search advantage.

## Named prefilter pipelines

### 1. Scan Finding Index

This is the original one-pass AVX2/AVX-512 design, including the old AVX-512
variant with a conditional jump after each vector mask.

```text
for every vector of codes:
    compare every lane with all cover points and ranges
    reduce lane predicates to an exact position mask
    if the mask is nonzero:
        map hit positions to row indices and deduplicate them
run exact KMP or decode+memmem over every candidate row
```

It performs one compressed scan and constructs exact candidate-row indices
immediately. Its output is compact, so it uses less temporary memory than a
hierarchical representation. The per-vector mask test can skip output work,
but the comparison and mask-reduction work has already happened. Once that
work dominates, removing the branch changes less than branch-focused intuition
suggests.

### 2. Superblock

This is the one-bit coarse summary. The runner default is 256 codes and the
cross-corpus production candidate is 512; the benchmark also supports 32, 64,
128, and 256. The hierarchical and Mid-cut segmented prototypes are currently
fixed at 512 codes.

```text
for every superblock:
    OR-reduce all cover predicates across its codes
    store one bit: did any code match the cover?
map every live superblock to every row it touches
run exact KMP or decode+memmem over those rows
```

It stores one bit per superblock, not one bit per value. The scan avoids
constructing exact masks and row indices, so it can approach the memory
bandwidth floor. Its weakness is false-positive amplification: one cover hit
makes all rows touched by that block candidates.

### 3. Superblock Hierarchical

This adds exact cover refinement to Superblock.

```text
stage 1: create one live bit per 512-code superblock
stage 2: rescan only live blocks with specialized point/range kernels
         recover exact cover-hit rows
stage 3: run exact KMP or decode+memmem over those rows
```

Stage 2 is exact for the cover, but the cover itself is not an exact substring
test. It must still run stage 3. This pipeline trades fewer final rows for a
second scan, mask construction, sparse-position-to-row localization, and
candidate output. On the whole-column low-selectivity ClickBench suite, that
refinement cost was large enough that coarse Superblock often won.

### 4. Superblock Hierarchical plus length bound

Before exact verification this rejects rows that cannot contain the needle:

```text
tight: row_codes >= ceil(needle_bytes / MAX_TOKEN_SIZE)
loose: row_codes >= floor(needle_bytes / MAX_TOKEN_SIZE)
```

Both are constant-time per candidate and preserve exactness. In the measured
30-needle ClickBench suite neither removed a candidate, while the extra pass
cost roughly 49--52 ms in aggregate. Keep these variants as experimental
comparators rather than dispatcher choices unless a corpus has many short
rows and long needles.

### 5. Superblock Hierarchical Mid-cut

This is the newest localized hierarchy and the preferred sparse-cover
prefilter.

```text
stage 1: create one live bit per 512-code superblock
stage 2: rescan live blocks and emit exact matching code positions
         retain the selected DAG cut's exact left/right token radius
stage 3: map positions monotonically to rows
         merge overlapping cut-bounded windows
         run KMP or decode+memmem only inside those windows
```

The mid-cut condition is stronger than merely knowing that a row contains a
covered token: an exact match must cross a cover hit at the selected alignment
cut. Therefore only the bounded code window around that position needs the
exact scan. This reduced exact KMP work materially on ClickBench. It still has
a localization/output floor, so coarse Superblock or Scan Finding Index can
win for extremely small hit sets.

## Kernels and assembly conclusions

The hot loops specialize common `(point_count, range_count)` shapes at compile
time. Both hierarchy passes use point/range-specific kernels rather than a
generic dynamic predicate loop. Available stage-1 kernels are:

- `intrinsics`: explicit AVX-512 summary reduction;
- `intrinsics-avx2`: explicit AVX2 P2R2 summary reduction;
- `intrinsics-branch`: AVX-512 with the per-vector mask branch;
- `autovec`: safe scalar source shaped for compiler vectorization;
- `autovec-miss`: auto-vectorized source with explicit target features;
- `scalar`: generic scalar reference/source variant.

The best generated code uses vector compares for points, two compares for each
inclusive range, OR reductions, and one summary-bit store per block. AVX-512 is
useful because it handles more `u16` lanes per instruction and has native mask
operations. `-C target-cpu=native` alone does not guarantee that a small helper
will be widened to 512 bits: the compiler balances vector width, reduction
shape, register pressure, and its cost model. Explicit target features and
intrinsics are retained for the best AVX-512 path; the auto-vectorized variants
remain as assembly and performance controls.

The branch was not the only cost in Scan Finding Index. Predicate compares,
mask formation, position-to-row mapping, deduplication, and exact verification
remain. Likewise, Superblock's scan can be bandwidth-like while its complete
pipeline is not: live-block expansion or refinement and exact checking make up
the rest of the runtime.

## Static dispatcher

The dispatcher uses only facts known after query analysis. It never observes
exact match counts, candidate counts, or measured runtime.

```text
if covered_fraction >= 0.06:
    Direct compressed KMP
else:
    Fused Superblock Mid-cut
    (compare-chain probe if point_count + 2 * range_count <= 16,
     cover-bitmap probe otherwise)
```

This is the post-fused-kernel model; the earlier cost-banded model and its
validation table below are retained as history. See
[`prefilter-optimization-experiments.md`](prefilter-optimization-experiments.md)
for the kernels and the 204-case retune (regret 0.07%).

The original exact-row-selectivity gate remains only as a benchmark comparator.
It is not a usable static model because it depends on scanning the data first.

Validation at both 2 and 4 MiB segment sizes:

| Corpus | Cases | Exact oracle choice | Correct family |
|---|---:|---:|---:|
| ClickBench calibration | 26 | 24 (92.31%) | 25 (96.15%) |
| Sentiment140 holdout | 96 | 96 (100%) | 96 (100%) |
| News-headline holdout | 96 | 94 (97.92%) | 95 (98.96%) |
| **Combined** | **218** | **214 (98.17%)** | **216 (99.08%)** |

ClickBench predicted/oracle runtime correlation was 0.99897, with 0.46% total
regret. The model deliberately stays small rather than fitting isolated
sub-millisecond or few-percent crossover misses.

## Main lessons

1. Cover frequency and comparison count both matter. Runtime is not linear in
   equality/range comparisons alone because loads, vector reduction, output
   construction, cache residency, and final exact work overlap or dominate.
2. Row selectivity alone is insufficient. Two covers with similar frequency
   can touch very different numbers of blocks, rows, or mid-cut windows.
3. Segmenting at 2 or 4 MiB changes the crossover. A second pass over a live
   segment can hit cache, unlike independent whole-column passes.
4. Superblock is closest to the bandwidth floor because it emits one bit per
   block. The complete pipeline cannot equal raw memory bandwidth unless
   candidate construction and exact checking are negligible.
5. Exact refinement has real output cost. It recreates lane masks, enumerates
   positions, maps them to rows, and deduplicates or merges windows.
6. Mid-cut wins when it avoids enough exact code scans to pay for localization.
   Direct KMP wins when the cover is dense/expensive. Scan Finding Index owns
   the middle region in the static model.
7. On the full low-selectivity ClickBench run, coarse Superblock beat Scan
   Finding Index in most queries, while Mid-cut had the best aggregate result
   once localization was introduced. Per-segment dispatch is better than one
   globally fixed prefilter.

Detailed historical numbers and stage breakdowns are in
[`clickbench-full-prefilter-results.md`](clickbench-full-prefilter-results.md).

## Build and prepare data

The executable requires an AVX-512BW-capable x86-64 host.

```bash
cargo build --release --example prefilter_e2e
```

Prepare the ClickBench URL column from the fresh single-file Parquet input:

```bash
ONPAIR_BENCH_PARQUET=.benchmark-data/clickbench-fresh-20260822.parquet \
ONPAIR_BENCH_COLUMN=URL \
ONPAIR_PF_DUMP=.benchmark-data/onpair-clickbench-fresh-full \
target/release/examples/prefilter_e2e prepare
```

Prepare newline-delimited strings (for example, extracted headlines):

```bash
ONPAIR_BENCH_LINES=.benchmark-data/headlines.txt \
ONPAIR_PF_DUMP=.benchmark-data/onpair-news-headlines \
target/release/examples/prefilter_e2e prepare-lines
```

Existing benchmark dumps used for the reported model are:

- `.benchmark-data/onpair-clickbench-fresh-full`
- `.benchmark-data/onpair-stringbench-sentiment140`
- `.benchmark-data/onpair-news-headlines`

## Required segmented regression runs

Pin to one otherwise-idle core. Report best-of-three initially and best-of-five
for final numbers. Segmented output reports complete KMP-pipeline runtimes,
oracle selection, dispatcher selection, and dispatcher regret. The current
segmented path does not time decode+`memmem` or split each pipeline into stages;
the unsegmented runner below reports those stage and verifier comparisons.

### ClickBench, 2 MiB and 4 MiB

```bash
ONPAIR_PF_DUMP=.benchmark-data/onpair-clickbench-fresh-full \
ONPAIR_QUERY_FILE=benches/queries/clickbench-high-selectivity.txt \
ONPAIR_SEGMENT_BYTES=2097152 \
ONPAIR_BENCH_REPS=3 \
taskset -c 0 target/release/examples/prefilter_e2e run
```

Repeat with `ONPAIR_SEGMENT_BYTES=4194304`.

To inspect only static features and the model decision without timing scans:

```bash
ONPAIR_PF_DUMP=.benchmark-data/onpair-clickbench-fresh-full \
ONPAIR_QUERY_FILE=benches/queries/clickbench-high-selectivity.txt \
ONPAIR_SEGMENT_BYTES=2097152 \
ONPAIR_FEATURES_ONLY=1 \
target/release/examples/prefilter_e2e run
```

### Mine and run Sentiment140 holdout queries

```bash
ONPAIR_PF_DUMP=.benchmark-data/onpair-stringbench-sentiment140 \
ONPAIR_BENCH_PARQUET=.benchmark-data/stringbench/sentiment140-train.parquet \
ONPAIR_BENCH_COLUMN=text \
ONPAIR_MINE_NEEDLES=1 \
ONPAIR_MINE_DENSE=1 \
ONPAIR_WRITE_QUERIES=.benchmark-data/sentiment140-model-queries.txt \
ONPAIR_FEATURES_ONLY=1 \
target/release/examples/prefilter_e2e run

ONPAIR_PF_DUMP=.benchmark-data/onpair-stringbench-sentiment140 \
ONPAIR_QUERY_FILE=.benchmark-data/sentiment140-model-queries.txt \
ONPAIR_SEGMENT_BYTES=2097152 \
ONPAIR_BENCH_REPS=3 \
taskset -c 0 target/release/examples/prefilter_e2e run
```

Repeat the timed command with `ONPAIR_SEGMENT_BYTES=4194304`.

### Mine and run news-headline holdout queries

```bash
ONPAIR_PF_DUMP=.benchmark-data/onpair-news-headlines \
ONPAIR_BENCH_PARQUET=.benchmark-data/news-headlines/headlines.csv \
ONPAIR_MINE_NEEDLES=1 \
ONPAIR_MINE_DENSE=1 \
ONPAIR_WRITE_QUERIES=.benchmark-data/news-model-queries.txt \
ONPAIR_FEATURES_ONLY=1 \
target/release/examples/prefilter_e2e run

ONPAIR_PF_DUMP=.benchmark-data/onpair-news-headlines \
ONPAIR_QUERY_FILE=.benchmark-data/news-model-queries.txt \
ONPAIR_SEGMENT_BYTES=2097152 \
ONPAIR_BENCH_REPS=3 \
taskset -c 0 target/release/examples/prefilter_e2e run
```

Repeat the timed command with `ONPAIR_SEGMENT_BYTES=4194304`.

## Focused diagnostics

### Isolate one complete pipeline

Set `ONPAIR_SEGMENT_ALGORITHM` to one of `scan`, `superblock`,
`hierarchical`, `hierarchical-length`, `midcut`, `dispatch`, `full-kmp`, or
`gated`:

```bash
ONPAIR_PF_DUMP=.benchmark-data/onpair-clickbench-fresh-full \
ONPAIR_ONLY_QUERY=google \
ONPAIR_SEGMENT_BYTES=2097152 \
ONPAIR_SEGMENT_ALGORITHM=midcut \
ONPAIR_BENCH_REPS=5 \
taskset -c 0 target/release/examples/prefilter_e2e run
```

The per-query oracle is meaningful only when all algorithms are measured. An
isolated run is for stage profiling, not model-accuracy scoring.

### Sweep superblock size and stage-1 kernel

```bash
ONPAIR_PF_DUMP=.benchmark-data/onpair-clickbench-fresh-full \
ONPAIR_PERF_QUERY=google \
ONPAIR_PF_BLOCK_CODES=512 \
ONPAIR_PF_KERNEL=intrinsics \
ONPAIR_PERF_REPS=100 \
taskset -c 0 target/release/examples/prefilter_e2e perf-summary-google
```

Repeat `ONPAIR_PF_BLOCK_CODES` for `32`, `64`, `128`, `256`, and `512` and
`ONPAIR_PF_KERNEL` for `intrinsics`, `intrinsics-avx2`,
`intrinsics-branch`, `autovec`, `autovec-miss`, and `scalar`. Keep the query,
CPU, repetitions, and input dump fixed across the matrix.

### Original branchy scan versus new scan

```bash
ONPAIR_PF_DUMP=.benchmark-data/onpair-clickbench-fresh-full \
ONPAIR_PERF_REPS=100 \
taskset -c 0 target/release/examples/prefilter_e2e perf-original-google

ONPAIR_PF_DUMP=.benchmark-data/onpair-clickbench-fresh-full \
ONPAIR_PERF_REPS=100 \
taskset -c 0 target/release/examples/prefilter_e2e perf-new-google
```

### Raw memory-bandwidth reference

```bash
taskset -c 0 cargo run --release \
  --manifest-path autovec-lab/Cargo.toml \
  --bin mem_bw -- \
  .benchmark-data/onpair-clickbench-fresh-full/codes.u16
```

Compare stage-1 code GB/s with this read-bandwidth floor. Also report decoded
GB/s, but do not confuse it with physical bytes loaded: decoded GB/s uses the
logical uncompressed string size as its numerator.

### Hardware counters

```bash
perf stat -r 5 \
  -e cycles,instructions,branches,branch-misses,cache-references,cache-misses \
  taskset -c 0 env \
  ONPAIR_PF_DUMP=.benchmark-data/onpair-clickbench-fresh-full \
  ONPAIR_ONLY_QUERY=google \
  ONPAIR_SEGMENT_BYTES=2097152 \
  ONPAIR_SEGMENT_ALGORITHM=midcut \
  ONPAIR_BENCH_REPS=5 \
  target/release/examples/prefilter_e2e run
```

Run the same command for `scan`, `superblock`, `hierarchical`, and `full-kmp`.
Use cycles and instructions to explain compute differences; branches and
branch misses test the old-jump hypothesis; cache counters help distinguish a
predicate/output bottleneck from a bandwidth floor.

### Inspect generated assembly

```bash
cargo rustc --release --example prefilter_e2e -- --emit=asm
objdump -Cd target/release/examples/prefilter_e2e > /tmp/prefilter_e2e.asm
```

Find the specialized summary/refinement symbols and check vector width,
point/range compare count, predicate ORs, mask extraction, summary stores, and
loop branches. Compare explicit AVX-512, AVX2, auto-vectorized, and branchy
variants rather than inferring instruction choice from source alone.

### Whole-column stage and verifier breakdown

Omit `ONPAIR_SEGMENT_BYTES` to run the detailed path that reports coarse scan,
refinement/localization, exact KMP, and decode+`memmem` separately:

```bash
ONPAIR_PF_DUMP=.benchmark-data/onpair-clickbench-fresh-full \
ONPAIR_QUERY_FILE=benches/queries/clickbench-high-selectivity.txt \
ONPAIR_SUMMARY_CODES=512 \
ONPAIR_BENCH_REPS=3 \
taskset -c 0 target/release/examples/prefilter_e2e run
```

Use this result for stage costs and KMP-versus-`memmem` comparisons. Use the
segmented result for the 2/4 MiB dispatcher oracle; they answer different cache
questions and should not be combined into one timing total.

## What to report for every change

For both 2 and 4 MiB segments, record:

- query bytes, point/range counts, cover frequency, and exact row matches;
- compressed codes and logical decoded bytes scanned;
- stage 1, refinement/localization, and exact-verification milliseconds;
- end-to-end KMP and decode+`memmem` milliseconds;
- code GB/s and decoded GB/s;
- candidate blocks, positions, windows, rows, and exact codes scanned;
- oracle algorithm, dispatcher algorithm, absolute runtime, and percentage
  regret;
- percentage runtime change versus Scan Finding Index, Superblock,
  Superblock Hierarchical, Mid-cut, and direct KMP.

Do not change model thresholds from the same queries used to claim holdout
accuracy. Mine a fresh corpus or reserve queries before tuning, and rerun the
complete matrix after changes to predicate kernels, output construction,
localization, block size, or dispatcher thresholds.
