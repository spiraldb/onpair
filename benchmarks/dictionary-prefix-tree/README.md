# Shared prefix paths and adaptive full-key rank

Parsing is faster on both prose and URLs. This replaces duplicated prefix ancestry
with eight-byte records and shared paths, retains exact 128-bit predecessor keys,
and adds an eight-way tree behind the two-byte prefix table. It changes neither
public APIs nor the encoded dictionary/codes/offsets format.

The headline uses the median of three independent process medians per case for
`parse` (both budgets; all three TPC-H columns). The other operations use the full
suite's run median. Percentages are **elapsed-time changes**, so negative is faster.

| Input | parse, three runs | train_and_compress | decompress_all |
| --- | ---: | ---: | ---: |
| TPC-H | -18.95% | -11.04% | +0.38% |
| Synthetic URLs | -19.77% | -10.81% | -0.07% |
| FineWeb prose | -19.97% | -9.83% | -2.12% |
| Book reviews | -28.76% | -15.13% | +2.25% |
| Actual URLs | -20.96% | -9.63% | +0.62% |

The repeated parses confirm the large-dictionary prose gain; individual runs vary.
Decoder code is unchanged. Its approximately ±2% variation is not an optimization
claim. The real/held-out reused-buffer kernels improve 9.6–31.1% (geometric mean
18.0%); their absolute throughput is about 222–758 MB/s. The 1 GB/s target is not
achieved for general strings on one core. The separate 256-single-byte dictionary
case reaches 6.0–10.6 GB/s with reused buffers because matching reduces to byte
widening; it is not representative of prose or URL compression.

## Machine and measurement

Intel Xeon 6975P-C, 32 vCPUs, x86-64, 64-byte cache lines; AVX-512F, AVX2 and
POPCNT available. Rust 1.91.0, optimized bench profile, no RUSTFLAGS. Every timed
run is single-threaded and pinned to CPU 0; builds run separately on CPU 8.
All public/real cases selected AVX-512F + POPCNT, flat sample rank with six lanes.
The adaptive eight-way tree uses four lanes on suitable controlled distributions.
The scalar fallback keeps four lanes on x86-64 and six on aarch64.

The existing `tpch` and `clickbench` benchmark bodies run `parse`, `decompress_all`
and `train_and_compress`; the only benchmark addition is ClickBench `parse`.
Full runs use `--bench --sample-count 25 --min-time 0.4 --max-time 3 --color never`.
Each stage has one full-suite process per input; baseline/final have two extra
parse-only processes in reversed order. Training cases can hit the time cap.
These are run medians, not confidence intervals. Public parse includes allocation
and dictionary cloning; training is outside its timer. `train_and_compress`
includes training. `decompress_all` allocates its decoded output.

TPC-H: SF=1, `ONPAIR_BENCH_MAX_BYTES=8388608`, approximately 8 MiB o_comment and
l_comment, 6.25 MiB p_name. Synthetic URLs: 100,000 rows, 3.98 MiB. Natural inputs
are identical whole-row 8 MiB Parquet `text` exports at every stage: FineWeb
8,393,783 bytes/2,461 rows, reviews 8,388,915/8,037, actual URLs
8,388,672/109,369. [Input hashes](inputs.csv) include raw bytes and u64 offsets;
[executable/Parquet hashes](sha256.json) identify frozen measurements.

Public benchmarks retain threshold **0.5 for every input**, including prose.
The separate frozen-dictionary kernels use threshold **0.2 for prose**, 0.5 for
URLs. Therefore their dictionary sizes and timings are different configurations.
[An additional replay](kernel-public-dictionaries.csv) checks exact code/offset
equality and timings with the public benchmarks' 0.5 dictionaries too.

Kernel runs exclude training, allocation and dictionary cloning. All implementations
receive identical frozen dictionaries, compare every emitted code/row offset to the
original scalar implementation before timing, and rotate backend order for three
repetitions: at least 100 ms per real case and 30 ms per controlled case.

## Structure and dispatch

Each aligned 128-byte leaf has eight high and eight low u64 keys. Full unsigned
comparisons rank all eight keys without losing suffix information. Flat sample
search stops its binary steps at eight candidates and finishes with rank as well.
The eight-way forest uses the same full keys as separators, contiguous children,
and a valid-separator mask. Prefix roots with equal candidate block ranges share
one tree. A prefix root is a two-byte trie step, followed by B-tree descent.

Each token stores `{base: u32, lens: u16, id: u16}`. `base` points to the proper
prefix path interned by its parent token; siblings share that path. The existing
LCP/length mask selects a prefix rank. A full fit selects `id`; a shorter fit reads
that rank in the shared path. A spare path entry makes the unconditional read
valid even for a full fit. Computed fit masks are reused rather than reconstructed.

At Arrow parse entry, inputs of at least 256 KiB sample 64 uniformly spaced **byte
positions**. If at least 56 have the same estimated tree height, use tree/four
lanes; otherwise use flat rank/six lanes. This samples input prefix popularity,
not corpus names or a tuned average token length. It estimates predictable lane
completion; either choice is exact. ISA detection and strategy selection happen
once per parse call. Generic `Rows` uses flat rank/six lanes after one ISA check;
its batches already copy noncontiguous rows. The small-input path skips sampling.
No environment switches are present in production.

Full match metadata falls from 32 to 8 bytes per token plus shared paths, but the
whole index is not always smaller: the extra roots/forest increase a 4,096-token
FineWeb index from 336,786 to 552,698 indexed bytes. The 44,944-token kernel
FineWeb index shrinks from 2,379,186 to 1,903,396; its 65,536-token public-benchmark
index shrinks from 3,408,786 to 2,559,428. These count vector elements, excluding
headers/spare capacity, not resident or peak allocator bytes. See
[all footprint/strategy records](layout-kernel.csv) and [public dictionaries](layout-public.csv).

## Token lengths, frequencies and distribution checks

Dictionary averages alone do not describe emitted codes. The 4,096-token FineWeb
dictionary averages 4.46 bytes/token, but emitted tokens average 3.55 bytes; the
44,829-token URL dictionary averages 7.64 while emitted tokens average 10.81.
Longer URL tokens remain useful: more input bytes are consumed per search.
Frequency also changes cache reuse and how many independent cold searches lanes
can overlap. The selection rule samples prefix popularity; it does not maintain a
frequency table in the hot loop. [Length/code-frequency histograms](real-distributions.csv)
and [per-case distributions](kernel-summary.csv) retain the actual evidence.

Lane tuning used only FineWeb-head and URL-head, at 12/16-bit budgets. On that
screen, flat ranks at 3/4/5/6/7 lanes had geometric-mean time ratios versus original
of 0.900/0.859/0.850/0.827/0.860; six was selected. Reviews, separate-shard FineWeb,
and next-disjoint-region URLs were validation, not tuning. Held-out dictionaries
are trained only on the head input. TPC-H and the controlled cases are additional
validation. The classifier's seven-eighths height agreement is a conservative
predictability rule, not a fitted corpus-length cutoff.

The 192 controlled cases cover 256/512/4,096/65,536 tokens; short, long and mixed
lengths; spread/clustered prefixes; uniform, 90%-requested-hot, 32-token bursts,
random misses and all-FF input; and 8/64/512-byte rows. Row truncation can make the
actual emitted distribution differ from the requested generator distribution;
the CSV records both dictionary and emitted length histograms and actual top-code
share. Every case improves in the final run; excluding the 12 single-byte-only
cases, geometric-mean time falls 17.1%, with the smallest improvement 2.7%.
See [raw repetitions](controlled.csv) and [case medians](controlled-summary.csv).
A forced scalar-only run on the same Xeon improves 2.4% overall, with individual
cases from 17.8% faster to 3.4% slower. This is not an ARM hardware benchmark;
ARM was cross-compiled, not performance-tuned.

## Why the dense u32 alternative lost

The earlier u32 column is measured separately in #63. It retains full keys and a
tie index, and a rank can be followed by a separate full-key load and correction.
On the diagnostic run, 94.9% of FineWeb12 and 97.0% of TPC-H o_comment16 leaf
loads straddled 64-byte cache lines. Those are allocation-dependent observations,
not an attribution of elapsed time to alignment alone. On o_comment16 the old
algorithm took 16.5 ns/code versus 30.4 for u32, with L1 misses 2.80→6.15 and branch
misses 0.122→0.956 per code. On FineWeb12, correction is only 1.9% but u32 still
loses, so tie corrections are not the only cause. [Original counters](u32-counters.csv)
show the extra work and loads.

The new full-key version's fresh counters are also instructive: o_comment16 drops
16.32→13.53 ns/code and 288→226 instructions/code, **despite** L1 misses rising
2.77→4.93. FineWeb12 drops 15.61→13.49 ns and 279→234 instructions while L1 misses
rise 1.88→2.68. It wins by shortening dependent searches and overlapping six lanes,
not by minimizing every miss counter. [Counter windows](counters.csv) cover 100
warm parses, with setup/warmup excluded through perf's acknowledged enable/disable
control. Cache-misses reads zero on this host and is not evidence of zero DRAM
traffic. Cycles/ns are comparable, with no observed frequency reduction explaining
the old u32 regression.

## Step 0: prefix ranges and ties

Measured and reported before implementation. Weighting counts actual greedy
lookups, including empty prefix ranges and excluding inactive SIMD lanes.
The prefix histogram counts all 65,536 possible prefixes, not just used prefixes.
The original benchmark inputs need the long-token tie path on 61.7–88.0% of
lookups, so suffix comparisons in a u32 index would be common.

This table uses the existing benchmark dictionaries plus the frozen prose/URL
kernel dictionaries (prose threshold 0.2). All cases have 100% of lookups in
ranges of at most 4,624 tokens. `Long tie` means equal bytes 2–5 with a token longer
than six bytes in the equal-prefix run. `Correction` means the last middle-key
candidate has a full key greater than the window.

| Case | Tokens | Lookups <=16 | Lookups <=272 | Long tie | Correction |
| --- | ---: | ---: | ---: | ---: | ---: |
| tpch/o_comment/12 | 4056 | 23.19% | 100.00% | 74.49% | 54.58% |
| tpch/o_comment/16 | 11264 | 8.14% | 90.96% | 82.00% | 63.31% |
| tpch/p_name/12 | 4063 | 11.41% | 100.00% | 80.70% | 51.76% |
| tpch/p_name/16 | 12113 | 0.19% | 79.94% | 88.00% | 67.16% |
| tpch/l_comment/12 | 4057 | 25.64% | 100.00% | 65.65% | 42.64% |
| tpch/l_comment/16 | 14664 | 6.53% | 88.12% | 75.88% | 57.08% |
| synthetic_urls/12 | 4062 | 49.58% | 98.04% | 61.71% | 41.99% |
| synthetic_urls/16 | 10383 | 20.33% | 89.17% | 63.82% | 46.19% |
| real/fineweb-32mib/12 | 4096 | 48.76% | 100.00% | 10.52% | 1.90% |
| heldout/fineweb-32mib/12 | 4096 | 49.06% | 100.00% | 9.89% | 1.84% |
| real/fineweb-32mib/16 | 44944 | 10.62% | 67.44% | 47.31% | 27.64% |
| heldout/fineweb-32mib/16 | 44944 | 11.26% | 68.18% | 44.37% | 26.26% |
| real/paper-book-reviews-32mib/12 | 4096 | 50.90% | 100.00% | 11.98% | 2.90% |
| real/paper-book-reviews-32mib/16 | 44336 | 9.74% | 69.49% | 48.78% | 28.46% |
| real/clickbench-32mib/12 | 4096 | 70.84% | 100.00% | 36.93% | 13.50% |
| heldout/clickbench-32mib/12 | 4096 | 71.17% | 100.00% | 36.17% | 13.29% |
| real/clickbench-32mib/16 | 44829 | 9.79% | 85.62% | 78.66% | 55.50% |
| heldout/clickbench-32mib/16 | 44829 | 11.93% | 87.14% | 68.91% | 50.05% |

Counts weighted by number of prefixes (held-out rows use the same dictionary and
therefore the same histogram as their training row):

| Dictionary | Empty | 1–16 | 17–272 | 273–4624 | >4624 |
| --- | ---: | ---: | ---: | ---: | ---: |
| tpch/o_comment/12 | 65008 | 464 | 64 | 0 | 0 |
| tpch/o_comment/16 | 64971 | 427 | 135 | 3 | 0 |
| tpch/p_name/12 | 65123 | 343 | 70 | 0 | 0 |
| tpch/p_name/16 | 65117 | 336 | 76 | 7 | 0 |
| tpch/l_comment/12 | 65006 | 462 | 68 | 0 | 0 |
| tpch/l_comment/16 | 64963 | 403 | 165 | 5 | 0 |
| synthetic_urls/12 | 65098 | 361 | 76 | 1 | 0 |
| synthetic_urls/16 | 65094 | 324 | 113 | 5 | 0 |
| real/fineweb-32mib/12 | 64629 | 835 | 72 | 0 | 0 |
| real/fineweb-32mib/16 | 63527 | 1635 | 345 | 29 | 0 |
| real/paper-book-reviews-32mib/12 | 64613 | 857 | 66 | 0 | 0 |
| real/paper-book-reviews-32mib/16 | 63650 | 1505 | 356 | 25 | 0 |
| real/clickbench-32mib/12 | 64314 | 1188 | 34 | 0 | 0 |
| real/clickbench-32mib/16 | 62270 | 2709 | 551 | 6 | 0 |

[Exact histograms](ranges.csv) include both prefix counts and lookup counts for
every observed range size. [Range summary](range-summary.csv) also records the
plain equal-middle rate, multi-entry equal-run rate, maximum run, and weighted mean.

## Per-commit public benchmarks

The following tables use the original full-suite run at each stage, so the final
parse entries differ slightly from the three-run headline. All cells are
geometric means of per-case elapsed-time ratios versus baseline, in percent.
[Absolute medians and immediately preceding-stage deltas](public.csv),
[all summary deltas](public-summary.csv), and [independent parse repeats](parse-repeats.csv)
are included. The byte-only commit has no applicable dictionary in these public
inputs; its effect is measured by the controlled single-byte cases. The tree row
includes that parse-entry fast path. Book-review training varies by a few percent
between intermediate runs; no individual stage is claimed to win every case.

| Stage | Commit |
| --- | --- |
| Baseline | 791c648 |
| Columns | 96f595d |
| + leaf rank | 45bbebb |
| + sample rank | 865c108 |
| + shared paths | 76ab58c |
| + adaptive tree / byte-only | 05735dd |
| + six flat lanes | 5331e13 |

### TPC-H

| Stage | parse | decompress_all | train_and_compress |
| --- | ---: | ---: | ---: |
| Baseline | +0.00% | +0.00% | +0.00% |
| Columns | -6.34% | +0.17% | -2.95% |
| + leaf rank | -5.50% | +0.04% | -2.53% |
| + sample rank | -4.82% | +0.60% | -2.46% |
| + shared paths | -9.35% | +0.73% | -6.22% |
| + adaptive tree / byte-only | -9.32% | -0.08% | -5.57% |
| + six flat lanes | -18.78% | +0.38% | -11.04% |

### Synthetic URLs

| Stage | parse | decompress_all | train_and_compress |
| --- | ---: | ---: | ---: |
| Baseline | +0.00% | +0.00% | +0.00% |
| Columns | -4.28% | +0.27% | -3.83% |
| + leaf rank | -6.68% | +1.77% | -4.69% |
| + sample rank | -5.66% | +0.54% | -3.99% |
| + shared paths | -9.60% | +0.04% | -6.22% |
| + adaptive tree / byte-only | -10.13% | +0.24% | -6.48% |
| + six flat lanes | -17.57% | -0.07% | -10.81% |

### FineWeb prose

| Stage | parse | decompress_all | train_and_compress |
| --- | ---: | ---: | ---: |
| Baseline | +0.00% | +0.00% | +0.00% |
| Columns | -6.25% | -1.75% | -3.28% |
| + leaf rank | -9.06% | -0.82% | -3.90% |
| + sample rank | -12.26% | -1.00% | -5.54% |
| + shared paths | -16.47% | -0.86% | -8.54% |
| + adaptive tree / byte-only | -17.99% | -1.42% | -6.27% |
| + six flat lanes | -20.64% | -2.12% | -9.83% |

### Book reviews

| Stage | parse | decompress_all | train_and_compress |
| --- | ---: | ---: | ---: |
| Baseline | +0.00% | +0.00% | +0.00% |
| Columns | -14.08% | +0.89% | -6.71% |
| + leaf rank | -16.73% | -0.20% | -7.49% |
| + sample rank | -17.69% | +0.66% | -7.95% |
| + shared paths | -23.91% | +0.82% | -12.09% |
| + adaptive tree / byte-only | -21.50% | -0.09% | -9.47% |
| + six flat lanes | -32.37% | +2.25% | -15.13% |

### Actual URLs

| Stage | parse | decompress_all | train_and_compress |
| --- | ---: | ---: | ---: |
| Baseline | +0.00% | +0.00% | +0.00% |
| Columns | -8.37% | -0.08% | -1.39% |
| + leaf rank | -9.45% | +0.77% | -3.65% |
| + sample rank | -10.57% | +1.18% | -4.27% |
| + shared paths | -14.17% | +0.62% | -7.26% |
| + adaptive tree / byte-only | -12.94% | +0.25% | -7.22% |
| + six flat lanes | -20.18% | +0.62% | -9.63% |

The separate SIMD steps are not uniformly faster in isolation: TPC-H improves
from the column layout but gives back about 1.6% with leaf/sample rank before
shared metadata and lane tuning. The retained combination improves all real and
controlled cases. The tree contributes mainly to distribution robustness; all
public input samples choose flat rank, and its construction adds some training
cost. That cost is included above.

## Exactness and edge cases

Every search starts at a block whose first key is <= the input key. Full-key
comparisons choose the last valid separator <= the query; the chosen contiguous
child contains its last predecessor. Inducting down the tree and then ranking the
leaf gives exactly the same q as scalar search, including equal zero-padded keys.
Only separators belonging to a node are counted: unused MAX slots must be masked,
especially when the window itself is MAX. The flat sample search can read beyond
its candidate range because excluded/later-prefix samples compare greater; seven
trailing samples are backed by real padding blocks and FF prefix paths, so MAX
windows also remain valid. No out-of-range memory read or non-existent token id is
introduced. The final LCP, window length bound and prefix length mask are unchanged.
Shared paths store exactly the proper-prefix ids that the old record repeated.

For comparison, the earlier u32 design needs its end mask because adjacent range
entries can have arbitrary middle words. A rank of zero means all tokens inside
the range are greater, so q is s-1. Equal middle words do not order the suffix, so
a full-key correction over the equal run is necessary. This implementation keeps
full keys and therefore does not require that correction branch.

All existing tests pass unchanged. New Hegel tests compare scalar, tree and flat
SIMD paths with brute force across random/prefix-rich dictionaries. Cases cover
embedded zeros and equal padded keys, all token lengths through 16, long shared
six-byte prefixes, 0/1/16/17/272/273/4624/4625 range boundaries, 65,536-token
dictionaries, short input tails, all-FF/MAX padding, and nonzero offsets/empty rows
with preexisting output. Scalar is always exercised; unavailable AVX-512 tests are
skipped after feature detection.

Checks: `cargo test --locked` (158 library tests + 2 doctests), release library
suite, `cargo fmt --all -- --check`, `cargo clippy --locked --all-targets --all-features`,
and `cargo check --locked --lib --target aarch64-unknown-linux-gnu`. No ARM timing
claims or unmeasured AVX2/NEON rank variants are included.

## Reproducing public measurements

Build the listed commits in separate persistent worktrees using Rust 1.91.0 and
`cargo bench --locked --no-run`; copy each produced tpch/clickbench executable
into a stage directory as `tpch` and `clickbench`. Avoid sharing mutable executables
between builds. Export the exact natural rows as the three `*-8mib.parquet` files
with a `text` column and verify `inputs.csv`/`sha256.json`.

`python3 measure.py --data-dir DATA --output-dir RESULTS --stage baseline=BASE_BINS
--stage final=FINAL_BINS --cpu 0` runs the existing three operations against every
input on one core. `--filter parse` repeats only parsing. The original bench
loader reports the chosen corpus; preserve that output to catch a different input.
Raw frozen benchmark binaries and the distribution probe are retained in the
persistent workspace separately from this source PR.
