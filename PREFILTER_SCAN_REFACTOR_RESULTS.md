# Prefilter scan template refactor results

Date: 2026-08-28

Ratios below are candidate/reference on the 3,354 guard-true queries; values
below 1 are faster. Every timed comparison validated the complete inventory of
3,804 paired queries, including successful status, truth hashes, and identical
cover facts.

## Repositories and host

- Source: `spiraldb/onpair`, branch `codex/prefilter-scan-refactor`. The required
  starting commit `45edc44ac2279d55ec1b4728538a1777ba4b022f` is an ancestor of
  this result.
- Harness: sibling checkout `onpair_like` at
  `8c938ab8ac984427cee4900c34224855126c5ef2`. The existing local edit to
  `datasets/prepare.py` was preserved. The untracked ABI-v7 comparator and
  ignored result directories were also preserved.
- Timed host: Linux x86-64, AMD EPYC 9R05, 12 cores, AVX2 and AVX-512BW,
  rustc 1.91.0. Core 0 pinning was effective; the governor was not reported.
- Untouched isolated repeat, run 1 to run 2: geo `0.9943`, total `0.9912`,
  p95 `1.0862`. This is the measured noise reference for the campaign.

## Phase decisions

| Phase/change | geo | total | p95 | Decision and reason |
|---|---:|---:|---:|---|
| 0: expand the scalar-oracle matrix | — | — | — | **Accept.** All 18 SSE2 shapes, both offset widths and row mappings, boundary lengths, unsigned range edges, empty rows, and long rows passed against the scalar oracle before kernel changes. |
| 1: AVX-512 template | 0.9991 | 1.0011 | 1.1108 | **Accept.** Aggregate timing was neutral versus isolated Phase 0; leaf inventory, hoisted broadcasts, miss-polarity accumulation, group gate, and hot-loop instruction mix passed the assembly gate. |
| 2a: AVX2 template | 1.0058 | 1.0019 | 1.0803 | **Accept.** Neutral aggregate result; both group variants and the early all-zero gate were preserved. None of the six dominant shapes regressed by more than 5%. |
| 2b: SSE2 template | 1.0178 | 1.0172 | 1.1369 | **Accept.** Below the 3% aggregate allowance for source consolidation; all 18 shapes and emitted hot-loop structure were preserved. None of the six dominant shapes regressed by more than 5%. |
| 3: NEON template | — | — | — | **Accept provisionally.** The expanded oracle tests passed and cross-compiled AArch64 leaf bodies matched the pre-port code. Native AArch64 timing remains outstanding, as allowed by the plan when no AArch64 host is available. Stored-lane compaction remains unchanged. |
| 4: flattened policy and dispatch | 0.9986 | 0.9928 | 1.0909 | **Accept.** Full tests passed. All 38 emitted x86 scan symbols had matching sizes/instruction totals, and all 12 inspected NEON leaf bodies had matching sizes. No indirect trait calls, bounds checks, or new hot-loop spills appeared. |

The Phase 1 timing uses the sequenced candidate against isolated Phase 0 run 1.
Phase 2 uses forced-ISA baselines. Phase 4 uses the accepted sequenced Phase 1
run as its reference.

## Phase 5 decisions

Where two triples appear, they are `AVX-512; forced AVX2` in that order.

| Experiment | geo | total | p95 | Decision and reason |
|---|---:|---:|---:|---|
| Wide fixed-shape cutoff: cost 4 | 0.9969; 1.0066 | 1.0067; 1.0081 | 1.1526; 1.0984 | **Accept.** Completes the cost-4 set with `(4,0)` and `(0,2)`; aggregate movement is within the acceptance band. |
| Wide fixed-shape cutoff: cost 6 | 0.9855; 0.9858 | 0.9843; 0.9869 | 1.0973; 1.0371 | **Accept.** About 1.4–1.5% geometric improvement on both paths versus cost 4. |
| Wide fixed-shape cutoff: cost 8 | 0.9774; 0.9882 | 0.9805; 0.9918 | 1.0794; 1.0355 | **Accept.** Further improvement on both paths; fixed shapes now cover the complete cost-at-most-8 set. |
| Wide fixed-shape cutoff: cost 10 | 1.0367 | 1.0213 | 1.2325 | **Revert.** A fresh cost-8 repeat differed from the first by only geo `1.0015`, but cost 10 then lost 3.67% geometrically. Costs 9 and 10 themselves improved, so the regression is attributed to wider code-layout pressure on unchanged shapes. |
| Remove AVX2 `GROUP = 8` | 1.0075 | 1.0097 | 1.0833 | **Accept.** Within noise/allowance, all dominant shapes stayed within 5%, and instantiated AVX2 fixed leaves fell from 48 to 24. |
| Remove dead NEON `(2,1)` pairing clause | — | — | — | **Accept.** The condition was unreachable. In a cross-compiled AArch64 harness, every one of the 12 instantiated NEON leaf bodies matched its baseline byte-for-byte by size and hash. Native AArch64 timing remains outstanding. |
| AVX2 saturated-subtraction range idiom | 0.9945 | 0.9880 | 1.0751 | **Accept.** Range-heavy strata improved and LLVM emitted `vpsubw`/`vpminuw`/`vpcmpeqw`; the early `vptest` gate remains. |
| NEON output reservation | — | — | — | **Not run; unchanged.** It requires a native AArch64 performance measurement, which was unavailable. Reservation therefore remains zero on AArch64 release builds. |
| NEON bitmask compaction | — | — | — | **Not run; unchanged.** It requires a native AArch64 performance measurement. Stored lanes plus `mark_block` remain in place. |
| Sparse coverage cutoff: `1/1,000` | 1.0301 | 1.1143 | 1.3829 | **Revert.** The newly switched `[0.0001,0.001)` coverage band lost geo `1.1694`, total `1.4013`, p95 `2.5927`. |
| Sparse coverage cutoff: `1/100,000` | 1.0253 | 0.9882 | 1.4189 | **Revert.** Despite aggregate total time, geometric mean and tail latency regressed; the newly switched `[0.00001,0.0001)` band had geo `1.0781`, p95 `1.4917`. Keep `1/10,000`. |
| Binary-search code gap: 64 | 1.0227 | 1.0255 | 1.1887 | **Revert.** No simplification benefit and the sparse `[0.00001,0.0001)` band lost geo `1.0416`, total `1.0539`. |
| Binary-search code gap: 256 | 0.9971 | 1.0052 | 1.1066 | **Accept.** Within the measured noise floor, including both sparse coverage bands. The final gap is 256 rather than 128. |

Rejected cutoffs and gaps were reverted before the next experiment, so the
final source contains only accepted policy changes.

## Final combined result

The final accepted tree versus the sequenced Phase 1 AVX-512 template run is:

| Set | geo | total | p95 | p99 |
|---|---:|---:|---:|---:|
| All guard-true queries | **0.9670** | **0.9716** | 1.0946 | 1.2979 |
| amazon-title | 0.9914 | 0.9849 | — | — |
| clickbench-url-1m | 0.9456 | 0.9398 | — | — |
| dbpedia-abstract | 0.9731 | 0.9724 | — | — |

The six dominant shapes all pass their 5% guardrail: P1 `0.9931`, P2 `0.9942`,
P3 `0.9848`, R1 `1.0082`, P1R1 `0.9900`, and P2R1 `0.9984` by geometric mean.
The final wide fixed-shape policy specializes every comparison-cost-at-most-8
shape, AVX2 always uses group one, sparse row mapping keeps the `1/10,000`
coverage cutoff, and row-offset binary search begins at a 256-code gap.

## Artifacts and verification

Benchmark data is under `../onpair_like/results/optimize_prefilter/iterations/`.
The primary final run is `phase5-binary-gap256-avx512`; it contains
`results.jsonl`, `manifest.json`, `bench.nm`, `bench.asm`, and both its immediate
and end-to-end comparison reports. Other experiment directories retain their
results and, where captured, assembly. Phase 2 AVX2 assembly is in
`phase2-avx2-template-assembly`. The AArch64 dead-clause before/after captures
are in `phase5-neon-dead-clause-aarch64`.

Final verification covers release formatting, the full release test suite, an
offline locked AArch64 `cargo check`, required-base ancestry, and a clean source
worktree. Native AArch64 benchmark confirmation is the only platform-specific
item still outstanding.
