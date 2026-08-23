# Prefilter optimization experiments — session log

Continuation of [`prefilter-algorithms-and-benchmarks.md`](prefilter-algorithms-and-benchmarks.md)
and [`clickbench-full-prefilter-results.md`](clickbench-full-prefilter-results.md).
All work on `examples/prefilter_e2e.rs`; Xeon 6975P-C (Granite Rapids, AVX-512BW),
benchmarks pinned to CPU 0, correctness asserted per segment against an
independent KMP ground truth. Segment numbers are `midcut_ms` at 2 MiB unless
noted; ClickBench kernel sweeps use a 1/5-column subset (`ONPAIR_PF_MAX_BLOCKS=800000`).

## Summary of shipped wins

| change | mechanism | representative effect |
|---|---|---|
| 82-shape const dispatch | every cover with `P+2R <= 16` gets a const-generic miss-chain kernel instead of the dynamic `Vec` loop | `https://` (P7R1) coarse 5.77 → 15.05 GB/s |
| mid-cut window merge | sweep-merge overlapping per-row verify windows | dense-cover verify blowup removed (`rodukty%` −18% window codes); mid-cut wins dense covers |
| cover-bitmap kernel | 8 KiB L1-resident membership bitmap, 2 `vpgatherdd` per vector; flat cost in cover size; auto-dispatch at `P+2R > 16` | `php` (P105R52) 3,853 → ~500 ms; `lanet.ru` 1,196 → ~380 |
| fused coarse+refine | one memory pass; no summary array, no live-block re-read; branchless `vpcompressd` emission | −6 % to −87 % across all queries |
| galloping `interpolated_row` | interpolation guess → gallop to bracket → local binary search | `google` whole-column localization −35 % |
| sub4 accumulation | four *miss-domain* `kand` accumulators in the k-file (one bit / 128 codes); re-probe only the live quarter | best-or-tied on every shape; P7R1 −17 % vs 1-bit gate |
| refined static dispatcher | `cost<=8` dense override; `cost<=56 && frac<0.3%` ultra-sparse override | 218-case regret 0.375 % → 0.007 % (pre-fused-kernel surface) |

## Winning-kernel design laws (each measured)

1. **Minimize miss-path work.** The 512-code gate branch itself is nearly free
   (per-vector-branch variant costs only 0–2.3 %); what killed the original
   scan was retaining masks on the 97 %-dead path. A per-vector emission
   branch in the fused loop cut streaming 18.7 → 15.5 GB/s.
2. **Memoize only what is free to keep and immediate to use.** The superblock
   summary array (memo with a *cold* revisit) lost to re-probing L1-hot data.
   Retention wins only inside the register file: bitmap probes retain their 4
   masks (recompute = gathers, expensive); chain probes retain 4 sub-bits in
   k-registers (`sub4`) because the k-file has exactly 6 free entries.
3. **Flat-cost membership beats linear-cost compares past `P+R ≈ 20`**
   (measured crossover: chain wins to cost 22; bitmap from 35; rule `P+2R > 16`).
4. **Segmentation refunds re-reads, not first reads.** 2 MiB morsels: mid-cut
   74 ms warm vs 136 ms cold on `google` (1.8×); 2 MiB (L2) beats 4 MiB (L3)
   72 vs 101 ms.

## Dead ends (kept out of the tree; reasons matter)

| experiment | result | why it failed |
|---|---|---|
| KMP state-0 zero-run skip (8-wide `base` OR) | +8–10 % | tokens ending in a needle prefix are dense in real dictionaries; zero-runs too short |
| 4-row interleaved KMP (`contains4`) | +13–27 % | rows average ~11 codes; OoO already overlaps adjacent rows across the retain loop; lane bookkeeping costs more than the ILP recovered |
| software prefetch in refine / fused loop | ±0–3 % (noise) | hardware streamer saturates the sequential stream; d=2048 helps only the standalone summarize kernel (+5 % DRAM) and hurts past 4 KiB (d=3072 → −21 %) |
| `superblock_candidates` cursor-local binary search | +4–13 % | the full-array `partition_point` search spine stays cache-hot; a moving base touches fresh lines per block |
| standalone `vpcompressd` refine pass | ±1 % | 0.31 hits/vector — the `trailing_zeros` loop was never the bottleneck; compress pays only inside the fused kernel |
| GPR AND-reduction vs `kandd` (summarize) | 62.12 vs 62.09 GB/s | codegen-equivalent |
| gate group 128 / 256 / 1024 | 512 best (256 −4 % only on P7R1) | smaller groups pay more gate tests; 1024 doubles re-probe waste and saves an already-amortized branch |
| sub8 (8 accumulators / 512) | first −13 %, then ≈ sub4 −1–3 % after SROA fix | 8 accumulators overflow the 8-entry k-file → `kmov` + GPR per vector; a variable-index array also stack-spilled until rewritten with constant indices |
| sub4@1024 / sub8@1024 | +8–19 % | a perfect sub8@1024 ≡ two sub4@512 groups minus one ~free branch; merging can only add k-file pressure |
| two-level bitmap probe (`vpermw` hi-byte live set → masked gathers) | +8–10 % on all bitmap covers | cover tokens scatter across the sorted ID space: most codes' high bytes are live, so level 1 rejects too few lanes to pay its ~5 ops/vector |

## Key single-run reproductions

- `google` (ClickBench Q20) over all 99,997,497 rows: features P3R2 cost 7
  frac 0.0100 %; whole-column mid-cut 213.8 ms (docs' P3R2 audit: 213.74);
  segmented 2 MiB 72–74 ms warm / ~128 cold (fused).
- Memory reference: `mem_bw` ceiling 17.07 GB/s; summarize miss-chain 16.4
  (95 %), +prefetch d=2048 → 17.26 (~101 %); L2-resident 62 GB/s; bitmap flat
  11.7 GB/s cache-resident, 5.3 DRAM.
- Small-chunk winners after bitmap+fused (all-slots, full column, 2 MiB):
  `php` mid-cut 497 vs full-KMP 1,682; `html` 955 vs 1,659; `news` 1,209 vs
  1,873; `http://` full-KMP 519 (frac 8.8 % — the surviving no-prefilter band).

## Cross-corpus oracle picture (pre-retune, two-pass kernels)

218 tuning cases (ClickBench 2+4 MiB, Sentiment140, news) + zero-shot
amazon-titles (94) and NASA access logs (96): mid-cut wins ~95 % of
prefilter-worthy cases on every corpus; refined dispatcher regret ≤ 0.5 %
everywhere (amazon 0.35 %, NASA 0.51 %). The fused/bitmap kernels then
flipped the former full-KMP band (`php`/`html`/`news`), obsoleting the
`cost >= 200` rule and the 3 % density gate.

## Final retune and cleanup (landed)

Rerunning the 2 MiB oracle matrix on the final kernels (204 cases across
ClickBench, Sentiment140, news, NASA logs, amazon-titles) collapsed the
dispatch model to a single density gate:

```text
if covered_fraction >= 6%: full compressed KMP
else:                      fused mid-cut (chain if P+2R <= 16, else bitmap)
```

Regret 0.07 %, one sub-ms miss. Scan-finding-index won zero cases and left
the model (it remains a measured comparator slot); the old model reads 85.8 %
regret against the new kernels. Mid-cut won 11/13 ClickBench queries — the
densest mid-cut win is 5.4 % covered codes, the sparsest full-KMP win 8.8 %
(`http://`, `.ru`), so the gate sits between them.

The tree then kept only winners: `scan_hit_positions_chain` (the sub4
k-file-accumulator fused scan), `scan_hit_positions_bitmap`, the auto
chain/bitmap dispatch, window-merged `localized_kmp`/`localized_memmem`, and
the galloping `interpolated_row`. All experiment env switches
(`ONPAIR_MIDCUT_FUSED`, `ONPAIR_FUSED_SHAPE`, `ONPAIR_MIDCUT_BITMAP`,
`ONPAIR_BITMAP_L1`, `ONPAIR_PF_DIST`) and the losing kernels (gate-size
variants, sub8 family, 1024 family, prefetch/GPR summarize variants,
two-level bitmap, two-pass bitmap mid-cut) were removed; the two-pass chain
hierarchy survives only in the named whole-column comparators. Example build
time fell 96 s -> 59 s.

## Verification discipline

Every timed pipeline is asserted equal to an independently computed row set
per segment; the KMP-vs-memmem verifier comparison, kernel matrices, and
dispatcher accuracies are reproducible via the commands in the main guide.
Two measurement pitfalls worth remembering: neighboring stages warm the
segment (all-slots mid-cut reads 1.8× faster than isolated), and best-of
timing across binaries must pin the same core with no concurrent runs.
