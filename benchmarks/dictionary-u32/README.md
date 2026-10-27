# Dense u32 dictionary matcher experiment

The u32 matcher preserves results but does not improve these text and URL workloads.
This is a measured prototype, not a recommendation to replace the existing matcher.
The requested design was measured as written before adding a cheaper exact full-key
correction guard. The final four-lane version still regresses every real text/URL
configuration measured here. Three lanes won the small tuning screen but regressed
broader validation, so the final commit retains four.

## Machine and method

Intel Xeon 6975P-C, x86-64, 32 vCPUs, AVX-512F, AVX2 and POPCNT; Rust 1.91.0.
Optimized bench profile, no RUSTFLAGS. All timed cases use CPU 0 on the same machine.
The scalar baseline has four lanes. U32 production measurements use AVX-512F +
POPCNT; the runtime ISA selection happens once per parse call. Four SIMD lanes are
used except in the three-lane screening/validation row. Construction, dictionary
cloning, training and buffer allocation are included or excluded as described below.

The unmodified existing TPC-H and ClickBench benchmark bodies run `parse`,
`decompress_all` and `train_and_compress` for 12/16-bit dictionary budgets, seed 42.
The only benchmark addition is ClickBench `parse`. Each stage uses one Divan run,
`--bench --sample-count 25 --min-time 0.4 --max-time 3`, with at least 25 parse
samples (the 3-second cap can reduce samples for expensive training cases).
Tables use run medians; they are not confidence intervals. Percentages are elapsed
**time** changes, not throughput changes: positive means slower.

TPC-H is generated at SF=1 with `ONPAIR_BENCH_MAX_BYTES=8388608`: l_comment and
o_comment are approximately 8 MiB, p_name is 6.25 MiB. Synthetic URLs are the
existing deterministic fallback (100,000 rows, 3.98 MiB). For real inputs, the same
ClickBench benchmark reads a Parquet `text` column through ONPAIR_BENCH_PARQUET and
ONPAIR_BENCH_COLUMN. These runs keep its existing threshold 0.5, including prose.
The first whole rows reaching 8 MiB are identical at every stage:

| Input | Raw bytes | Rows |
| --- | ---: | ---: |
| FineWeb prose | 8,393,783 | 2,461 |
| Book-review prose | 8,388,915 | 8,037 |
| ClickBench actual URLs | 8,388,672 | 109,369 |

The additional frozen-dictionary **parse kernel** runs use reusable buffers and
exclude public API allocation/dictionary cloning and training. They use threshold
0.2 for prose and 0.5 for URLs, so the kernel and public-benchmark dictionaries are
intentionally distinct configurations. Each kernel case rotates the baseline/new
order for three repetitions, at least 100 ms each on real inputs and 30 ms on
controlled inputs. Every emitted code and row offset is compared exactly to the
original scalar parser before timing. The 192 controlled cases also decode back
to the original bytes. Dictionaries are frozen across implementations.

Held-out FineWeb comes from a separate local shard; held-out URLs come from the
next disjoint 8 MiB, with the dictionary trained only on the first region. Neither
held-out data nor book reviews were used to choose lane count. The lane screen
used only FineWeb-head and URL-head at both budgets. No average-length/frequency
classifier is added: the data does not support one that improves real inputs.

## Step 0: prefix ranges and ties

Measured and reported before implementation. Weighting counts actual greedy
lookups, including empty prefix ranges and excluding inactive SIMD lanes.
The prefix histogram counts all 65,536 possible prefixes, not just used prefixes.
The stop condition for almost-all-tiny or almost-all-huge ranges did not hold.
The premise that full-key ties would be rare did not hold either: the original
benchmark inputs need the long-token tie path on 61.7–88.0% of lookups. The user
explicitly authorized benchmarking the design after those measurements.

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

All values below are geometric means of the per-case elapsed-time ratios versus
the scalar baseline. Each suite has both dictionary budgets; TPC-H has all three
columns. `public.csv` includes absolute median milliseconds and both cumulative
and immediately preceding-commit deltas for every case.

| Stage | Commit | Meaning |
| --- | --- | --- |
| Baseline | 0ac4807 | develop matcher plus ClickBench parse bench |
| Layout | 07e2d46 | full-key vector, starts/mid/tie columns; old scalar lookup |
| SIMD leaves | fcb4700 | masked AVX-512/AVX2 rank, coarse binary leaf search |
| Sample levels | 0235f20 | range-local separator levels |
| Fixed strides | e1de090 | remove variable division from level descent |
| Correction guard | 1680402 | reuse candidate key; search ties only if full key exceeds window |
| Three lanes | bd2572c | screened lane count, subsequently rejected |
| Final | c6ce322 | retain four lanes; runtime identical to correction-guard row |

### TPC-H
| Stage | parse | decompress_all | train_and_compress |
| --- | ---: | ---: | ---: |
| Layout | +4.18% | +0.02% | +3.31% |
| + SIMD leaves | +121.90% | -0.17% | +69.95% |
| + Sample levels | +136.61% | +0.01% | +76.99% |
| + Fixed strides | +125.74% | -0.26% | +71.20% |
| + Correction guard (final) | +79.89% | -0.21% | +45.99% |
| Three lanes (rejected) | +95.58% | +0.27% | +68.30% |
### Synthetic URLs
| Stage | parse | decompress_all | train_and_compress |
| --- | ---: | ---: | ---: |
| Layout | +2.57% | -1.34% | +0.51% |
| + SIMD leaves | +110.95% | -0.13% | +46.53% |
| + Sample levels | +125.26% | -0.50% | +54.52% |
| + Fixed strides | +118.11% | -0.56% | +50.71% |
| + Correction guard (final) | +72.64% | +0.08% | +30.58% |
| Three lanes (rejected) | +81.81% | -0.39% | +35.19% |
### FineWeb prose
| Stage | parse | decompress_all | train_and_compress |
| --- | ---: | ---: | ---: |
| Layout | -5.87% | -1.13% | -0.70% |
| + SIMD leaves | +29.57% | +1.97% | +16.82% |
| + Sample levels | +44.39% | -0.88% | +23.77% |
| + Fixed strides | +32.53% | +0.11% | +18.72% |
| + Correction guard (final) | +11.97% | +0.63% | +11.95% |
| Three lanes (rejected) | +15.77% | +1.77% | +11.27% |
### Book-review prose
| Stage | parse | decompress_all | train_and_compress |
| --- | ---: | ---: | ---: |
| Layout | -0.18% | +0.58% | -3.61% |
| + SIMD leaves | +44.35% | +0.73% | +12.84% |
| + Sample levels | +60.86% | +0.62% | +18.68% |
| + Fixed strides | +45.54% | +1.73% | +14.38% |
| + Correction guard (final) | +20.51% | +2.02% | +2.98% |
| Three lanes (rejected) | +27.65% | -0.66% | +6.47% |
### Actual URLs
| Stage | parse | decompress_all | train_and_compress |
| --- | ---: | ---: | ---: |
| Layout | -1.54% | -0.61% | -1.52% |
| + SIMD leaves | +71.17% | -0.19% | +26.99% |
| + Sample levels | +83.03% | -0.17% | +33.06% |
| + Fixed strides | +76.84% | -0.76% | +28.69% |
| + Correction guard (final) | +41.34% | +0.05% | +17.46% |
| Three lanes (rejected) | +39.69% | -0.15% | +15.93% |

These data do not justify enabling the u32 matcher by default. In particular,
adding sample levels did not pay back their metadata, masks and descent overhead
on these distributions. Avoiding unnecessary full-key searches helps substantially
but does not turn the final result into a win.

## Frozen-kernel validation and distributions

The following final four-lane measurements are medians of the three repetitions.
The original baseline runs alongside the new parser in every case. Dictionary
length is unweighted; matched length is weighted by actual emitted codes.
Frequency summaries are in kernel-real.csv. [distributions.csv](distributions.csv)
contains exact dictionary-length and emitted-length histograms, plus logarithmic
code-frequency bins (including unused dictionary codes). [inputs.csv](inputs.csv)
records byte/offset hashes of the local prose and URL heads; the data is not bundled.

| Case | Mean dict length | Mean matched length | Original ns/code | U32 ns/code | Time change |
| --- | ---: | ---: | ---: | ---: | ---: |
| real/fineweb-32mib/12 | 4.456 | 3.553 | 15.546 | 18.139 | +16.68% |
| heldout/fineweb-32mib/12 | 4.456 | 3.495 | 15.999 | 18.685 | +16.79% |
| real/fineweb-32mib/16 | 7.639 | 5.499 | 27.633 | 35.551 | +28.65% |
| heldout/fineweb-32mib/16 | 7.639 | 5.229 | 28.690 | 35.806 | +24.80% |
| real/paper-book-reviews-32mib/12 | 4.476 | 3.655 | 13.938 | 16.520 | +18.52% |
| real/paper-book-reviews-32mib/16 | 7.513 | 5.661 | 25.502 | 34.198 | +34.10% |
| real/clickbench-32mib/12 | 5.217 | 6.104 | 13.413 | 17.877 | +33.28% |
| heldout/clickbench-32mib/12 | 5.217 | 6.024 | 13.483 | 17.997 | +33.48% |
| real/clickbench-32mib/16 | 7.645 | 10.807 | 21.647 | 33.089 | +52.86% |
| heldout/clickbench-32mib/16 | 7.645 | 9.562 | 21.948 | 33.743 | +53.74% |

[kernel-real.csv](kernel-real.csv) contains every real-input kernel stage and code
frequency summaries. The controlled matrix covers 256, 512, 4,096 and 65,536-token
dictionaries; short, long and mixed token lengths; spread and clustered prefixes;
uniform, 90%-hot and 32-code-burst frequencies; random misses; all-0xFF input; and
8-, 64- and 512-byte rows. Among the 192 controlled cases, the final four-lane
prototype improves 28 by more than 5% and regresses 89 by more than 5%. The
as-written SIMD-leaf stage improves 10 and regresses 156. Gains are concentrated
in large dictionaries of short tokens and do not generalize to the measured prose
or URL inputs.

## Layout and exactness

`starts` is u32 because the final entry must represent 65,536 tokens. The histogram
has at most three ranks on the benchmark dictionaries; a packed u32 side table
stores a two-bit depth and an offset into range-local contiguous sample levels.
Nodes have sixteen separator middle keys and seventeen children; leaf capacities
are 16, 272, 4,624 and 78,608 tokens. The final level is the dictionary-order `mid`
column, padded by fifteen entries. Full u128 keys and PrefixChains stay separate.
An equal-six-byte-run start table bounds full-key correction. The two u32
per-prefix tables add about 512 KiB independent of dictionary size, before the
per-token middle/tie columns and separator arrays. This prototype also retains
the old scalar search metadata so both implementations can be tested directly.

Within a fixed two-byte prefix, unsigned order of bytes 2–5 is monotonic. At each
sample node, counting separators <= w selects the furthest child whose first key
can be <= w. Descending this way ends in the leaf containing the final mid <= w.
The final masked rank therefore gives the last candidate ignoring bytes 6–15.

1. Entries after a prefix range can belong to a different prefix and their middle
   keys need not be greater. They must be masked. Padding by MAX alone is also
   insufficient when w is u32::MAX. Sample separators past the range are masked
   too. An empty final range clamps its zero-masked load to the last real token,
   so fifteen pads are sufficient without reading a sixteenth pad.
2. If the final count is zero at offset zero, no token in the prefix can precede
   the window, so q = s - 1. The complete single-byte dictionary guarantees this
   predecessor. Prefix zero instead contains key zero, so its count cannot be
   zero. A nonzero child offset has a first mid <= w and hence count >= 1.
3. A middle key equal to w does not determine full-key order. The last equal-mid
   token can exceed the window in bytes 6–15. Full keys in that contiguous run are
   sorted, so the existing branch-free `advance_if_le` binary search finds the
   exact predecessor, including the token before the run. If the candidate's full
   key is already <= the window, it is already the last eligible token and no
   correction is needed. Equal zero-padded keys retain the last equal token.

The existing LCP, length mask and PrefixChain lookup then produce the same longest
prefix. The old scalar assembly path and its all-0xFF padded chains remain; the
u32 path masks padding and selects real tokens, including u128::MAX windows.
Bytes beyond input continue to read as zero. No public API or encoded format changes.

## Validation and limits

- 152 tests plus two doctests pass. Existing tests are unchanged, including the
  Hegel and brute-force match tests. A new 400-case property compares scalar,
  scalar-middle, AVX2 and AVX-512 results to brute force. ISA checks skip SIMD
  cleanly on unsupported machines while always running scalar comparisons.
- Cases cover zero-containing tokens and equal keys such as `a`/`a\0`, sixteen-byte
  tokens, long equal-six-byte runs, ranges of 0, 1, 16, 17, 272, 273, 4,624,
  4,625 and 65,280 entries, input tails, empty final ranges and all-0xFF windows.
- Formatting, all-target/all-feature Clippy, private-item rustdoc and the
  aarch64 library cross-check pass with Rust 1.91.0.
- ARM hardware was unavailable. NEON rank is kept test-only for cross-path testing
  on ARM; production ARM retains the original scalar search and lane count. No
  ARM performance improvement is claimed.


## AVX2 comparison and reproducing public benches

A forced four-lane AVX2 specialization was measured with Rust 1.91.0 on the same
Xeon, inputs, frozen dictionaries and three-repetition kernel protocol. It is
slower than both scalar and AVX-512 on the real-input cases: FineWeb-head is
+39.9%/+44.8% versus scalar for 12/16 bits, reviews +42.5%/+48.1%, and actual
URLs +58.8%/+83.5%. [avx2.csv](avx2.csv) includes held-out cases and TPC-H too.
This does not predict AVX2 performance on a different microarchitecture. NEON
intrinsics were type-checked for aarch64 but were not executed or benchmarked.

For public benches, check out the desired commit and use the same machine,
compiler, CPU affinity and input file for every stage:

```sh
ONPAIR_BENCH_SCALE_FACTOR=1 ONPAIR_BENCH_MAX_BYTES=8388608 \
  taskset -c 0 cargo bench --locked --bench tpch -- \
  --sample-count 25 --min-time 0.4 --max-time 3 --color never

ONPAIR_BENCH_PARQUET=corpus-8mib.parquet ONPAIR_BENCH_COLUMN=text \
  taskset -c 0 cargo bench --locked --bench clickbench -- \
  --sample-count 25 --min-time 0.4 --max-time 3 --color never
```

Omit ONPAIR_BENCH_PARQUET to use the existing synthetic URL fallback. The
Parquet inputs were losslessly exported from the local length-prefixed corpora;
raw byte and u64 row-offset hashes above make their exact input identity explicit.
The additional kernel and statistics harness is retained with the local raw
results; the committed tests are self-contained and require no external corpus.
