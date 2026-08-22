# Full ClickBench prefilter comparison

## Named-pipeline rerun after the upstream merge

On 2026-08-22, `origin/feat/search-prefilter` at `5936ffc` was merged into the
benchmark branch and the named pipelines were rerun over all 99,997,497 rows.
The corpus contained 49 needles; the table covers all 30 whose cover frequency
is below 0.1%, the range in which the upstream profitability policy can choose
a prefilter. Each number is the sum of per-query best-of-five measurements on
CPU 0. Lower is better.

| Pipeline | KMP total ms | vs Scan Finding Index | memmem + decompress total ms | vs Scan Finding Index |
|---|---:|---:|---:|---:|
| Scan Finding Index | 15,500.873 | — | 15,757.893 | — |
| **Superblock** | **11,427.434** | **-26.28%** | **12,455.945** | **-20.95%** |
| Superblock Hierarchical | 13,335.665 | -13.97% | 13,564.275 | -13.92% |
| Hierarchical + loose length bound | 13,384.226 | -13.65% | 13,612.836 | -13.61% |
| Hierarchical + tight length bound | 13,387.495 | -13.63% | 13,616.105 | -13.59% |

The names mean:

- **Scan Finding Index**: scan all codes, turn exact hit masks directly into row
  indices, then run the exact verifier.
- **Superblock**: store one bit per 512 codes and run the exact verifier on all
  rows touched by a live block.
- **Superblock Hierarchical**: store one bit per 512 codes, rescan only live
  blocks to recover exact cover-hit rows, then run the exact verifier.

The tight constant-time length bound is
`row_codes >= ceil(needle_bytes / MAX_TOKEN_SIZE)`; the loose bound uses floor
instead of ceil. Neither removed one candidate for any of these needles. Their
additional candidate pass cost 51.829 ms and 48.561 ms respectively across the
suite, so they are not useful here. The adjusted totals above use the same exact
verification cost as the plain hierarchy; separately rerunning the verifier
after the length pass warms its input and would misleadingly make the bound
look faster.

In uncompressed URL space, the average prefilter-stage rates were 18.63 GB/s for
Scan Finding Index, 47.41 GB/s for the Superblock summary, and 21.90 GB/s for
the complete hierarchical summary plus refinement. Superblock won 28 of 30 KMP
queries and 21 of 30 `memmem` queries. KMP was faster in aggregate for every
pipeline; decoding plus `memmem` was 9.00% slower for Superblock because its
coarse candidate lists are much larger.

## Result

**Current production winner: the explicit-intrinsic AVX-512 hierarchy with one
summary bit per 512 compressed codes, followed by an exact rescan of live
blocks.** On the full fresh ClickBench URL column it reduced mean end-to-end KMP
runtime by 51.14% versus the original AVX2 scan and 28.46% versus the original
AVX-512 scan.

An oracle that chooses the best block size independently for every needle gains
only another 0.85% in aggregate. A fixed 256-code block was 0.07% faster than
512 on this corpus alone, which is below the stability needed to replace 512 as
the cross-corpus production choice.

## Input and method

The input is the canonical single-file [ClickBench compatibility
dataset](https://datasets.clickhouse.com/hits_compatible/hits.parquet), fetched
on 2026-08-22. The server reports a 2022-06-25 last-modified date and ETag
`6b028bb94eecf0ff4e6cde62a0f8fa48-829`.

- Parquet: 14,779,976,446 bytes; SHA-256
  `a390f6cb782f6aaef278c72fc1dd86c4f30bc843ebab3c159e9bd4d45ddb079f`.
- URL rows: 99,997,497; decoded URL bytes: 9,038,838,007.
- OnPair: 1,114,050,580 `u16` codes (2,228,101,160 bytes), 65,536 tokens.
- Code SHA-256:
  `ce97dc5d31c215d505b7cc49fa68443aa915d95687939252866460e18245d050`.
- Row-offset SHA-256:
  `64955f272a8ad96ca3322a3ebf07d4b99f288357dbbc2ff91264b48d75d633c3`.

The Parquet binary URL values were consumed directly. There was no transcoding,
row sampling, or value rewrite. Compression used 16 dictionary bits, threshold
0.5, and seed 42. Measurements ran on CPU 0 of an Intel Xeon 6975P-C. Primary
AVX-512/original results are best of five; additional hierarchy variants are
best of three. The 17 needles are mined P2R2 covers below 0.1% code frequency.

`KMP` and `memmem` are both exact final checks. Stage 2 is also exact with
respect to the cover; it regenerates position masks only inside live blocks.

## Aggregate end-to-end runtime

The table reports mean runtime across the same 17 needles. “Adaptive” is an
offline oracle selecting the best measured block size per needle, not a shipped
runtime policy.

| Pipeline | Mean KMP ms | Runtime reduction vs original AVX2 | vs original AVX-512 |
|---|---:|---:|---:|
| Original AVX2, one pass | 831.627 | — | — |
| Original AVX-512, one pass | 568.001 | 31.70% | — |
| AVX2 autovec hierarchy, adaptive | 524.974 | 36.87% | 7.58% |
| AVX-512 autovec hierarchy, adaptive | 541.256 | 34.92% | 4.71% |
| AVX2 intrinsic hierarchy, fixed 128 | 481.574 | 42.09% | 15.22% |
| **AVX-512 intrinsic hierarchy, fixed 512 (production)** | **406.342** | **51.14%** | **28.46%** |
| AVX-512 intrinsic hierarchy, fixed 256 | 406.051 | 51.17% | 28.51% |
| AVX-512 intrinsic hierarchy, adaptive oracle | 402.887 | 51.55% | 29.07% |

With final `memmem`, original AVX2 averaged 840.261 ms, original AVX-512
576.700 ms, and the fixed-512 hierarchy 414.446 ms: reductions of 50.68% and
28.13%, respectively. KMP was slightly faster overall for this selective
candidate set.

## AVX-512 intrinsic block-size sweep

| Codes per bit | Summary storage | Coarse ms | Exact refine ms | End-to-end KMP ms | Reduction vs AVX2 | vs AVX-512 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1.0375 MiB | 213.102 | 223.465 | 475.553 | 42.82% | 16.28% |
| 256 | 0.5188 MiB | 141.399 | 225.737 | 406.051 | 51.17% | 28.51% |
| **512** | **0.2594 MiB** | **136.104** | **231.357** | **406.342** | **51.14%** | **28.46%** |
| 1,024 | 0.1297 MiB | 130.461 | 250.535 | 420.133 | 49.48% | 26.03% |
| 2,048 | 0.0648 MiB | 140.747 | 276.759 | 456.649 | 45.09% | 19.60% |
| 4,096 | 0.0324 MiB | 135.150 | 302.407 | 476.455 | 42.71% | 16.12% |

The coarse scan approaches a floor near 130 ms. Larger blocks do not make that
floor materially lower, while they make more codes eligible for the exact
second scan. The adaptive winner distribution was 256 for seven needles, 512
for five, and 1,024 for five. Frequency alone cannot distinguish their spatial
clustering.

## Summary-only kernel matrix

This isolates the pre-hierarchy operation: scan every code and retain one bit
per block, without exact refinement or substring verification. Times are means
of five scans of the 2.228 GB code buffer for the same P2R2 cover.

| Kernel | 32 codes | 64 codes | 128 codes | 256 codes | 512 codes |
|---|---:|---:|---:|---:|---:|
| AVX2 intrinsics | 324.05 ms | 229.81 ms | **208.67 ms** | 314.89 ms | 312.72 ms |
| AVX2 autovec (`x86-64-v3`) | 310.42 ms | 240.01 ms | 225.02 ms | **179.89 ms** | 290.13 ms |
| AVX-512 autovec (forced ZMM) | **155.13 ms** | 235.45 ms | 221.61 ms | 216.83 ms | 207.82 ms |
| AVX-512 intrinsics | 192.24 ms | 240.11 ms | 215.55 ms | 141.48 ms | **137.36 ms** |

The best useful wide-block result is explicit AVX-512 at 512 codes: 16.22 GB/s
of compressed input. Independent AVX-512 loads reached 17.12 GB/s, so the
summary kernel achieves 94.7% of the one-core memory-bandwidth ceiling.

Autovec is highly sensitive to loop shape and block constant. Forced-ZMM
autovec wins at 32 codes, but explicit intrinsics are 34.8% faster at 256 and
33.9% faster at 512. AVX2 autovec beats the first AVX2 intrinsic draft at 256,
but the AVX2 intrinsic 128-code specialization gives the best complete AVX2
pipeline because its exact mask extraction is much cheaper.

## Per-needle fixed-512 KMP runtime

| Needle | Original AVX2 ms | Original AVX-512 ms | Hierarchy ms | Reduction vs AVX2 | vs AVX-512 |
|---|---:|---:|---:|---:|---:|
| `//novo` | 866.101 | 656.286 | 486.103 | 43.87% | 25.93% |
| `7/view_t` | 1,359.063 | 1,093.941 | 881.917 | 35.11% | 19.38% |
| `ail.ru/sanat` | 698.051 | 402.129 | 273.391 | 60.84% | 32.01% |
| `cars/rep` | 1,358.198 | 1,090.651 | 896.815 | 33.97% | 17.77% |
| `conomics` | 656.580 | 353.645 | 199.668 | 69.59% | 43.54% |
| `facebook` | 782.254 | 541.503 | 387.636 | 50.45% | 28.41% |
| `http://antic` | 619.835 | 349.855 | 191.588 | 69.09% | 45.24% |
| `http://forum/sho` | 910.993 | 683.470 | 542.511 | 40.45% | 20.62% |
| `http://gde24` | 609.823 | 356.731 | 188.575 | 69.08% | 47.14% |
| `http://lazar` | 662.501 | 408.925 | 245.662 | 62.92% | 39.93% |
| `l.ru/a-folders/#` | 871.722 | 643.989 | 497.136 | 42.97% | 22.80% |
| `nopoisk.ru/anime` | 788.918 | 499.793 | 357.402 | 54.70% | 28.49% |
| `onda.votpusk` | 640.915 | 361.506 | 201.947 | 68.49% | 44.14% |
| `pravanga` | 658.069 | 362.626 | 204.851 | 68.87% | 43.51% |
| `s.kz` | 710.859 | 463.815 | 322.979 | 54.56% | 30.36% |
| `u/thread.php` | 673.679 | 381.438 | 231.358 | 65.66% | 39.35% |
| `w/vaca` | 1,270.095 | 1,005.712 | 798.270 | 37.15% | 20.63% |

The fixed-512 hierarchy won all 17 comparisons against both original kernels.

## Assembly and counters

The x86-64-v3 portable loop contains YMM `vpcmpeqw`, `vpsubw`, `vpminuw`, and
`vpor` reductions. The forced-v4 loop contains ZMM compares plus mask operations
(`vpcmpeqw`, `vpcmpleuw`, `kortestd`, `kmovd`). The intrinsic loop is the
shortest stable ZMM form: two `vpcmpneqw`, two `vpcmpnleuw`, two `vpsubw`, mask
combination, and a final test for each vector group. This confirms that the
register width is correct in both AVX-512 builds; the remaining autovec loss is
reduction/loop code shape, not an accidental YMM selection.

For ten full `conomics` scans, the original AVX-512 mode took 3,486.2 ms and the
production path 1,952.3 ms (44.0% lower scan runtime). Whole-process `perf stat`
reported 27.55B to 11.63B instructions, 6.23B to 2.36B branches, and 20.60M to
1.20M branch misses. These counter ratios include dump loading and frequency
index construction, so they are conservative; the timed scan values do not.

### Google P3R2 audit

The full URL-column `google` query normalizes to three points and two ranges.
It previously missed the fixed sparse-shape matrix and entered the dynamic
AVX-512 kernel. Adding P3R2 to that matrix reduced the production exact
prefilter from 498.858 ms to 306.079 ms (38.65%) and prefilter plus KMP from
523.072 ms to 330.971 ms (36.73%).

The fixed prescan assembly uses one ZMM load, three chained mask point-miss
comparisons, two `vpsubw`, and two unsigned mask range comparisons per 32
codes. Sixteen vectors are unrolled per 512-code summary bit and finish with
one `kortestd`; there is no per-vector `kmov` in this pass. Exact refinement
uses the same intrinsic predicate and transfers only final live masks to scalar
registers. Row materialization and block-to-row mapping are scalar because they
walk variable-density masks and row offsets. Token KMP is also scalar: its next
state depends on the preceding state, so it cannot be auto-vectorized.

A diagnostic localized-window refinement reduced KMP input from 5,129,154 to
1,229,374 codes (76.03%). Its best 512-code hierarchy was 304.995 ms, narrowly
behind the direct one-bit-superblock-to-KMP path at 300.703 ms. The exact live
block rescan, not KMP input volume, is the remaining crossover cost.

## Conclusion

- Keep one production implementation: explicit AVX-512 intrinsics, 512 codes
  per summary bit, exact live-block rescan.
- Keep the original dense AVX-512 retained-mask path and AVX2 fallback; they
  cover different density/ISA cases and are not obsolete duplicates.
- Keep autovec in the benchmark lab only. It can generate the desired register
  width, but it does not deliver the best complete hierarchy and changes
  materially with block constants and target flags.
- Do not ship the current AVX2 hierarchy draft. Its 128-code form is useful as
  an experimental comparison, but production AVX-512 is 16.3% faster on
  aggregate and the wider AVX2 specializations have poor reduction code shape.
