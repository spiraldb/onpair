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

## Crash / exception stack traces (bug-report datasets)

The corpora above are *profiling* stacks. A second, larger family is *crash*
stack traces from bug-report deduplication research. `prep/convert_crash_traces.py`
ingests two public sources (streamed, byte-capped at 60 MB/dataset):

* **AERI** (Eclipse) — per-problem JSON, frame = `class.method`
  ([download.eclipse.org](https://download.eclipse.org/scava/aeri_stacktraces/))
* **JetBrains EMSE** — Ubuntu(campbell)/Eclipse/NetBeans/Gnome, frame = `function`
  ([Zenodo 5746044](https://doi.org/10.5281/zenodo.5746044))

```bash
python3 prep/convert_crash_traces.py    # needs the archives in data/cache
cargo run --release crash_eclipse crash_netbeans crash_gnome crash_ubuntu crash_aeri crash_corpus
```

Structural sizes (no string dict), OnPair vs the two baselines:

| dataset | lang | traces | distinct frames | onpair saves vs listview | onpair vs wholestack-dedup |
|---------|------|-------:|----------------:|-------------------------:|---------------------------:|
| crash_ubuntu | C/C++ | 14 k | 21 k | 39.1% | **+27.3%** |
| crash_eclipse | Java | 30 k | 58 k | 42.4% | **+33.8%** |
| crash_netbeans | Java | 40 k | 30 k | 51.4% | **+18.5%** |
| crash_gnome | C/C++ | 181 k | 52 k | 44.2% | −1.7% (dedup wins) |
| crash_aeri | Java | 22 k | 52 k | 55.7% | **+44.8%** |

Findings:

- **OnPair beats whole-stack dedup on crash traces** (the reverse of resampled
  profiles): bug reports rarely repeat an *entire* trace, but they heavily share
  call-stack *prefixes* (framework/runtime entry paths), which is exactly what
  frame-level merging captures. Gnome is the one exception — it has the most
  exact-duplicate traces (only 28% distinct).
- **The frame-string dictionary dominates** (55–77% of the OnPair+strdict total).
  Crash frames are long fully-qualified names with 21 k–58 k distinct values, so
  the *vocabulary*, not the list structure, is the real cost — unlike profiling
  data (531–2 031 distinct frames). Compressing the string dictionary matters
  more than the structural encoding here.
- **`wholestack+onpair` is again the best non-zstd representation everywhere**
  (e.g. netbeans 22.5× vs raw, gnome 15.3×), but `zstd(int stream)` still wins on
  pure ratio (84×–273×) because it entropy-codes the long-tailed code stream.
- **Heterogeneous combined corpus** (`crash_corpus`, 5 projects as 5 runs):
  sharing one dictionary is **39% worse** than per-run — same lesson as the mixed
  profiling corpus, because Java and C/C++ frame namespaces are disjoint.

## The trick on a `list(bool)` column (mp4 decode masks)

A real Vortex file (`*_index_listbool.vortex`, an mp4 index) carries a deeply
nested boolean column: `col_v0_tracks[].frames_by_video[].closure_local_decode_mask_le`,
a `list(bool)` of per-frame decode-dependency masks. `vx tree array` shows it is
**164 KB — 35% of the whole file** — and that Vortex stored the mask bits
**completely uncompressed** (`vortex.bool`, a raw 148 KB bit buffer + 16 KB
bitpacked offsets). `vortex-tui/examples/extract_mask.rs` dumps the 5 786 masks
to `maskbool.lst` (one mask per row, bool elements); run with a larger
dictionary so OnPair can build multi-bit tokens on the 2-symbol alphabet:

```bash
DICT_CAP=65536 cargo run --release maskbool
```

| representation | structural bytes | vs raw |
|----------------|-----------------:|-------:|
| listview (≈ what Vortex stores) | 163 349 | 14.5× |
| listview unique-only (dedup) | 102 031 | 23.2× |
| onpair | 59 394 | 39.8× |
| onpair + unique-only | 42 612 | 55.5× |
| zstd medium (L3) | 19 527 | 121× |
| zstd high (L19) | 16 459 | 144× |

The OnPair trick **does** transfer to `list(bool)`: 163 KB → 59 KB (**2.75×**),
or 43 KB (**3.8×**) with whole-mask dedup, while keeping O(1) row access. OnPair
learns the run-patterns (26 merged tokens, avg 13 bits/token) — essentially
discovering RLE.

**But the real story is the data shape.** Every one of the 5 786 masks is a pure
`1^k 0^m` run (100% triangular; popcount 1–272). So a mask is fully described by
a single integer (its popcount — the length already lives in the list offsets).
That collapses the 148 KB bit buffer to **5 786 popcounts ≈ 6.5 KB (23×)**, and
the whole column to **~21.7 KB (7.5×)** — beating OnPair and approaching zstd,
with full random access. The actionable finding for Vortex: this column should
hit an RLE / run-end (or constant-run) encoding; today it falls through to raw
`vortex.bool`. OnPair is a decent general fallback (no special-casing, keeps
random access) but a run-aware encoding is strictly better here.

**Packing bools into bytes helps a lot.** With a 2-symbol alphabet, a 16-element
OnPair token spans only 16 bits, so a ~200-bit run needs ~13 tokens. Packing
8 bools into a little-endian `u8` (`prep/pack_mask_bytes.py` → `maskbyte.lst`)
gives a 9-symbol alphabet (`0,1,3,7,15,31,63,127,255` — the partial bytes of a
`1^k0^m` run) and lets a token span 16 bytes = 128 bits:

| representation (structural, no strdict) | bool elements | **u8 elements** |
|-----------------------------------------|--------------:|----------------:|
| listview (all rows) | 163 349 | 87 100 |
| listview unique-only (dedup) | 102 031 | 58 379 |
| onpair | 59 394 | **24 831** |
| onpair + unique-only | 42 612 | **23 100** |
| zstd high (L19) | 16 459 | **5 207** |

Byte-packing alone halves every method (dict-encoding 9 byte values to 4 bits is
0.5 bits/bool vs the raw 1 bit/bool), and byte-OnPair is **2.4× smaller than
bit-OnPair** (24.8 KB vs 59.4 KB) because each token now covers 8× more bits.
Even so, the `1^k0^m` structure means zstd (5.2 KB) and a popcount encoding
(~6.5 KB values) remain the true optimum; OnPair on bytes is the best
random-access structural option short of a run-aware encoding.

## Multi-round OnPair (Re-Pair-style)

OnPair is one online BPE pass with a hard `MAX_TOKEN_ELEMS = 16` ceiling, so no
token spans more than 16 elements. The recursive generalization is **Re-Pair**
(Larsson–Moffat): iterate pairing until no pair repeats, producing a grammar.
`multiround()` approximates it by feeding each round's pruned code stream back in
as the next round's alphabet, so a round-`r` token expands to up to `16^r` base
elements. Stored cost = final codes + one grammar dictionary per round + row
offsets (run with `cargo run --release <dataset>`):

| dataset | round 1 | round 2 | round 3 | best | +dedup | zstd L19 |
|---------|--------:|--------:|--------:|-----:|-------:|---------:|
| maskbool (bits) | 59 394 | **28 198** | 28 661 | round 2 (**2.1×**) | 42 612 | 16 459 |
| maskbyte | **26 283** | 26 417 | 28 180 | round 1 (saturated) | 24 761 | 5 207 |
| perf_runs | 1 009 822 | **723 043** | 724 820 | round 2 (1.4×) | 262 337 | 492 012 |
| crash_gnome | 3 614 285 | **3 250 787** | 3 497 603 | round 2 (1.1×) | 2 268 705 | 834 250 |

Findings:

- **A second round helps most exactly where round 1 was ceiling-limited.** The
  bit-level masks halve (59 KB → 28 KB): round 1 tokens cap at 16 bits, round 2
  merges them to ~256-bit tokens that capture whole runs. On `perf_runs` round 2
  drops the stream to 203 078 codes = **one code per row** — the grammar learned
  whole repeated stacks.
- **The sweet spot is 2 rounds (occasionally 3).** Beyond that the code stream
  barely shrinks while each round's grammar dictionary accumulates, so the total
  climbs again — the classic Re-Pair tradeoff (you must store the grammar).
- **Byte-packing already does what an extra round would.** `maskbyte` saturates
  at round 1 because a 16-*byte* token is 128 bits — the second round has little
  left to merge.
- It confirms the ceiling theory but **does not overtake whole-mask dedup or zstd
  here**: for the `1^k0^m` masks a run-aware/popcount encoding (~6.5 KB) is still
  the right answer. Multi-round OnPair is the better *general* structural codec
  (no special-casing, keeps random access) and a free win on ceiling-bound data.

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
