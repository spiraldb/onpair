# list-onpair: OnPair for dictionary-encoded list columns

OnPair compresses **string** columns: an input is `(concatenated bytes, per-row
offsets)` and the algorithm builds an incremental-BPE dictionary of byte
sequences, then encodes each row by greedy longest-prefix match into fixed-width
codes that support O(1) random access.

A **dictionary-encoded list column** has the *exact same shape* —
`(concatenated element-ids, per-row offsets)` — only the alphabet is the set of
distinct elements (e.g. distinct stack frames) instead of the 256 byte values.
So the OnPair technique ports directly: this experiment generalizes it to an
**integer element alphabet** and measures the compression ratio on four *real*
datasets.

```
list of strings           dict-encode elements        OnPair over ints
["a","b","c"]  ──────────▶  [0,1,2]  (ListView<int>)  ──────────▶  codes + token dict
```

## What is "list data that is pseudo-regular like text"?

Ranked by how much OnPair-style *contiguous-subsequence merging* should help
(i.e. how text-like / sequential the lists are):

1. **Call stacks / stack traces** (folded flamegraph stacks) — strongest: deep
   lists with long shared prefixes (stack bottoms) and shared frame runs.
2. **File paths** split on `/` — directory-component sequences, shared prefixes.
3. **Clickstreams / session page sequences** — sequential navigation (not tested
   here; same shape as stacks).
4. Set-like (weak: order is arbitrary, lists are shallow): **tags**, **metric
   label sets**, **graph adjacency lists**, market baskets, movie genres.

## Datasets (all real)

| dataset  | source | rows | elements | distinct | avg len |
|----------|--------|-----:|---------:|---------:|--------:|
| `stacks` | Brendan Gregg real `perf` capture → folded stacks (FlameGraph) | 710 | 19 538 | 531 | 27.5 |
| `paths`  | real file tree (`vortex` + `onpair` repos), split on `/` | 2 785 | 14 626 | 1 740 | 5.3 |
| `tags`   | real Stack Exchange data dump (`3dprinting`, `Posts.xml` tags) | 5 735 | 14 518 | 496 | 2.5 |
| `graph`  | real SNAP `wiki-Vote` graph → neighbour list per node | 6 110 | 103 689 | 2 381 | 17.0 |

Every dataset has **> 256 distinct elements**, so the codes do *not* fit in
`u8`: the "just reuse the byte compressor on `u8` codes" shortcut does not apply
to real data, which is why this builds a generalized integer OnPair
(`src/intonpair.rs`).

## Build & run

```bash
python3 prep/build_datasets.py     # downloads real data into ./data (cached)
cargo run --release                # trains, encodes, verifies round-trip, reports
```

## Method

`src/intonpair.rs` ports the byte trainer/encoder to `u32` elements: base tokens
= distinct elements, frequent adjacent token pairs merged up to
`MAX_TOKEN_ELEMS = 16` elements, dynamic-threshold controller ported verbatim
(counting elements, not bytes). After encoding we **prune to the tokens actually
used** and renumber so the code width is minimal.

Sizes are the minimal **bit-packed** representation (apples-to-apples):

* `dict+listview` (the task's step 1, no OnPair): `num_elems × base_bits` values
  + bit-packed row offsets.
* `onpair-int` (step 2): `num_codes × code_bits` codes + token-dict
  (element payload at `base_bits` + offsets) + bit-packed row offsets.
* The shared element→string dictionary is reported separately — every method
  needs it identically.
* `zstd` (level 19) on the raw text and on the fixed-width int stream is shown
  as a *general-compressor* reference (no random access).

## Results

Structural size = code stream + token/row offsets, excluding the shared string
dict. "OnPair Δ" is how much smaller the OnPair int stream is vs the plain
ListView int stream — **this is the value OnPair adds on top of dict-encoding.**

| dataset  | avg_len | codes vs elems | code width | listview (B) | onpair (B) | **OnPair Δ** | zstd(ints) |
|----------|--------:|---------------:|-----------:|-------------:|-----------:|-------------:|-----------:|
| `stacks` | 27.5 | 2.84× fewer | 10b (=base) | 25 757 | 12 712 | **−50.6%** ✅ | 2 908 |
| `paths`  | 5.3  | 1.49× fewer | 11b (=base) | 24 987 | 24 751 | **−0.9%** ~ | 6 643 |
| `graph`  | 17.0 | 1.37× fewer | 13b (+1)    | 168 520 | 173 871 | **+3.2%** ❌ | 130 999 |
| `tags`   | 2.5  | 1.27× fewer | 11b (+2)    | 26 371 | 29 077 | **+10.3%** ❌ | 16 040 |

Full ratios vs raw text (including the shared string dict):

| dataset  | raw | dict+listview | **onpair-int** | zstd(raw) | zstd(ints) |
|----------|----:|--------------:|---------------:|----------:|-----------:|
| `stacks` | 1.00× | 15.9× | **23.4×** | 72.7× | 221.6× |
| `paths`  | 1.00× | 2.56× | **2.57×** | 8.6× | 18.9× |
| `graph`  | 1.00× | 2.73× | **2.65×** | 3.9× | 3.8× |
| `tags`   | 1.00× | 4.69× | **4.33×** | 6.0× | 9.4× |

## Scaling: a store of 1000 perf runs

A single capture undersells OnPair, because its value is *repeated contiguous
subsequences* and one profile has few. The realistic case is an observability
store of many profiles. `prep/build_perf_runs.py` models **1000 perf runs** —
~70% runs of the same "java service" (the *similar* cohort), ~20% `wrk`
load-generator, ~10% other workloads (the *different* cohort). No frames are
invented: every stack is a real stack from the real capture, drawn per-run
(weighted by real sample counts); only the run *population* is constructed.

```bash
python3 prep/build_perf_runs.py     # -> data/perf_runs.lst (+ .runs sidecar)
cargo run --release perf_runs
```

Result — 1000 runs, 203 078 stacks, 5.24 M frames (still 531 distinct frames):

| representation | bytes | ratio vs raw |
|----------------|------:|-------------:|
| raw text | 177.5 MB | 1.0× |
| dict+listview | 7.15 MB | 24.8× |
| **onpair-int** | **1.02 MB** | **173.3×** |
| zstd(raw text) | 0.96 MB | 184.7× |
| zstd(int stream) | 0.49 MB | 360.9× |

* OnPair now saves **85.8%** of the int stream (vs 50.6% at single-run scale):
  with similar runs repeated, merged tokens grow to **9.16 frames** on average
  and fold 5.24 M frames into 415 k codes (**12.6× fewer**), code width still
  10 bits. onpair-int (173×) catches up to `zstd(raw)` (185×) **while keeping
  O(1) random access**; plain zstd does not.
* **Share the dictionary across runs.** Storing all 1000 runs in one
  shared-dictionary column is **48% smaller** than compressing each run
  independently (1.01 MB vs 1.96 MB): a global dictionary captures cross-run
  repeated stacks and amortizes the token-dict overhead.

Caveat: because all runs resample one real capture, the frame alphabet stays at
531. A store of 1000 *genuinely* distinct workloads would have a larger alphabet
(wider base codes, bigger token dict), so treat 173× as the *similar-cohort*
regime — which is exactly the common observability case (one service profiled
repeatedly).

## Real heterogeneous corpus (gathered public profiles)

`prep/gather_real_profiles.py` collects **real** folded-stack files shipped as
test fixtures by public profiling projects (no synthesis): `perf`, dtrace,
`ghcprof` (Haskell), VS profiler, Intel VTune, macOS `sample`, `xctrace`,
async-profiler (JVM), and the vertx perf capture — from `jonhoo/inferno`,
`jlfwong/speedscope`, and `brendangregg/FlameGraph`. After content-dedup this is
**22 distinct real runs, 3 508 stacks, 61 305 frames, 2 031 distinct frames**,
all concatenated into one shared column.

```bash
python3 prep/gather_real_profiles.py    # clones the repos, normalizes fixtures
cargo run --release perf_corpus
```

| representation | bytes | ratio vs raw |
|----------------|------:|-------------:|
| raw text | 2 147 437 | 1.0× |
| dict+listview | 163 117 | 13.2× |
| **onpair-int** | **138 404** | **15.5×** |
| zstd(raw text) | 31 636 | 67.9× |
| zstd(int stream) | 11 332 | 189.5× |

OnPair saves **27.1%** of the int stream here — between the single-run (50.6%)
and the same-service 1000-run (85.8%) cases, because some runs are near-duplicate
re-profiles (the vertx/dcpu/dtrace groups) but most are different tools entirely.

**The decisive finding: share the dictionary only for *similar* runs.**

| corpus | runs | shared-dict vs per-run |
|--------|-----:|-----------------------:|
| same-service resample (`perf_runs`) | 1000 | sharing **saves 48%** ✅ |
| heterogeneous public profiles (`perf_corpus`) | 22 | sharing is **66% worse** ❌ |

For the heterogeneous corpus a *single* global dictionary must span all 2 031
frames, so every run — even a 11-frame VTune capture — pays an 11-bit base /
12-bit code width. Compressed independently, each run uses only its own small
frame alphabet and gets a much narrower code. The crossover is governed by how
much the runs' **frame namespaces overlap**: high overlap (one service profiled
repeatedly) → share; low overlap (a mixed store of different tools/languages) →
compress per run (or cluster by namespace, then share within a cluster).

## Whole-stack dedup (when an entire trace equals another)

A cheaper idea than frame-level OnPair: give each **distinct whole stack** one id
and store the column as one code per row plus a stack table (the `stack table +
sample→stack id` layout real profile stores use). It only helps to the extent
that *entire* traces repeat.

| corpus | distinct / rows | listview | wholestack-dict | onpair | **wholestack+onpair** |
|--------|----------------:|---------:|----------------:|-------:|----------------------:|
| `perf_runs` (resampled) | 710 / 203 078 | 24.8× | **603×** | 173× | **641×** |
| `perf_corpus` (real, mixed) | 2 773 / 3 508 | 13.2× | 14.9× | 15.5× | **17.1×** |

- When rows repeat heavily (resampled service: only 710 distinct traces),
  whole-stack dedup alone beats frame-OnPair by ~3.5× — duplication is *row*-level,
  so dedup the row.
- When most traces are unique (mixed real profiles: 79% distinct), dedup barely
  beats plain listview and frame-OnPair still edges it.
- **Combining them wins in both cases**: dedup removes repeated rows, then OnPair
  squeezes the shared prefixes out of the (smaller) stack table. This is the
  representation to ship.

## Is it good? When, and why not

**It works where the data is genuinely sequential and deep — stack traces.**
On `stacks`, OnPair more than halves the int stream (−50.6%) and lifts the
random-access ratio from 15.9× to 23.4×. Deep lists (27 frames) with repeated
contiguous frame runs are exactly what incremental BPE captures: 127 merged
tokens average 2.74 frames each and fold 19.5k frames into 6.9k codes, all while
the code width stays at the base 10 bits.

**It's neutral-to-negative on set-like / shallow lists — and here's why:**

1. **The code-width cliff.** OnPair adds merged tokens, so the dictionary can
   outgrow the base alphabet and push `code_bits` up. That increase is a tax on
   *every* code. `tags` goes 9b → 11b: a +22% per-code cost applied to all
   11.4k codes, which the modest 1.27× count reduction can't repay. Net +10.3%.
2. **Weak contiguous repetition.** Tags and graph adjacency are *sets* — order
   is arbitrary (sorted), and pair co-occurrence is sparse, so few merges get
   reused enough to pay for their dictionary entry. Adjacency neighbour-lists are
   largely unique, so there is almost nothing to merge.
3. **Shallow lists.** With 2.5 (tags) or 5.3 (paths) elements per row there is
   little room for multi-element tokens; the merge machinery's overhead isn't
   amortized.

**And even where OnPair wins on ratio, a general compressor wins bigger** —
zstd beats OnPair everywhere (stacks: 221× vs 23×) because it adds entropy
coding and *long-range* matching (whole duplicate stacks across rows), neither of
which OnPair does. The reason to use OnPair-on-ints anyway is the same reason the
byte version exists: **O(1) random access to any row** at a fixed code width,
which zstd's stream sacrifices. So the honest summary is:

> Generalized OnPair is a good *random-access-preserving* second stage on top of
> dictionary-encoded **sequential** list columns (stack traces, clickstreams,
> deep paths). On set-like or shallow lists it should be skipped — the encoder
> should fall back to plain fixed-width ListView codes whenever the trained
> dictionary would widen the code beyond `base_bits` without a commensurate drop
> in code count.

### Improvement ideas (not yet implemented)

* **Width-guard:** reject merges that would push `code_bits` past `base_bits`
  unless they reduce total code bits — turns the losses above into no-ops.
* **Entropy-code the codes** (the codes are far from uniform) to claw back the
  gap to zstd while keeping per-row decodability.
* **Sort each set-list canonically** (already done for graph/tags) and consider a
  delta/RLE path for monotone numeric lists (graph adjacency) instead of BPE.
* For sampled observability data, **expand stacks by sample count** — then most
  rows are exact duplicates and row-level dedup dominates (favouring a different
  layout again).
