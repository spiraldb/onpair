# Handover — compressed-domain LIKE search (prefix / contains)

Branch: `claude/cpp-dfa-contains-prefix-IJjlD` (all work committed & pushed; HEAD `25fb845`).
Repo: `spiraldb/onpair`. Crate builds clean: `cargo test --lib` = 95 pass, `cargo clippy --lib --benches --tests` = 0 issues.

## What this work is

`onpair` compresses each string column into a stream of `u16` dictionary token
ids ("codes") over a **lexicographically-sorted** dictionary (token ids are in
sort order; 256 single-byte tokens always present). LIKE predicates are
evaluated **in the compressed domain** — token-level automata run directly over
the codes, rows are never decompressed.

- `Pattern::Prefix(needle)` — `col LIKE 'needle%'` — `src/search/prefix.rs`
- `Pattern::Contains(needle)` — `col LIKE '%needle%'` — `src/search/kmp.rs`
- Driver / SIMD / RowMask — `src/search/mod.rs`
- Public API: `SearchParts::search() -> RowMask` and `search_callback(pattern, |row| …)`.

## Headline results (real ClickBench `hits_0` URL, 1M rows, bits=16; FineWeb 50k docs)

GB/s = over uncompressed input; medians; a contended shared box (treat ratios as
robust, absolutes as noisy). All verified against brute force (`cd == bf`).

**Prefix — a structural win everywhere (sorted dict ⇒ contiguous id range ⇒ real SIMD):**

| query | onpair `search()` | arrow (memmem/starts_with on *decompressed*) | decompress+arrow |
|---|---|---|---|
| ClickBench `http:%` 51.8% | ~260 µs | ~4.6 ms | ~47 ms |
| ClickBench `http://k%` 11.7% | ~160 µs | ~5.6 ms | ~47 ms |
| FineWeb `The%` 6.6% | ~443 µs | ~600 µs | ~98 ms |

Prefix is **30–40× over arrow-on-decompressed** and **~350–600× over
decompress+arrow**. This is the clear, shippable win.

**Contains — wins vs decode-then-scan, ~parity-to-loss vs in-memory memmem:**

| query | onpair | arrow memmem (decompressed) | decompress+arrow |
|---|---|---|---|
| ClickBench `%http:%` 53% | ~9.4 ms | ~11.3 ms | ~53 ms |
| ClickBench `%i.yandex%` 0.2% | ~19 ms | ~16 ms | ~57 ms |
| ClickBench `%google%` 0.009% | ~18.7 ms | ~15.4 ms | ~61 ms |
| FineWeb `%photosynthesis%` 0.01% | ~34 ms | ~9 ms | ~107 ms |

Contains beats `decompress+memmem` 3–6× (decode alone is ~46–100 ms and
dominates), but does **not** beat in-memory memmem in general, and **loses 3–4×
on FineWeb** (long ~499-codes/row documents).

## How it works

### Prefix (default, fully shipped — this is the strong result)
- Optional `first_codes` child array: one `u16` first-token id per row
  (`Column::first_codes: Option<Vec<u16>>`, built at compress time, +7% column
  size on URLs). `ONPAIR`-gating-free; `None` ⇒ generic scan.
- Two-pass branchless filter exploiting the sort: pass 1 is a SIMD **unsigned
  range test** `begin ≤ first_code ≤ last` (the `prefix_range`), plus an equality
  lane `== q0` for multi-token needles; pass 2 confirms only `== q0` candidates.
- AVX2 kernels (`prefilter_accept*_avx2`), runtime-detected (`avx2_enabled()`),
  scalar fallback. `ONPAIR_NO_SIMD=1` forces scalar.
- `search()` writes the accept bits straight into the `RowMask` words
  (bitmap-merge fast path, `prefix_mask`) — no per-row callback.

### Contains (default = scalar 2-code chain; SIMD variants opt-in)
- `KmpAutomaton`: token-level KMP. `base[t]` = exit state feeding token `t` from
  state 0; `sparse` = per-state exception ranges. `matches()` is the exact
  confirmer (fast state-0 path + slow sparse path).
- **Default prefilter** (`row_chain` + `chain_table`): per token, three sound bit
  flags — DEFINITE (token contains whole needle ⇒ row matches), OPEN (`base≠0`,
  can start a spanning match), CONT (can continue one). A row is a candidate iff
  it has a DEFINITE token, or an **OPEN→CONT adjacent pair** (the "2-code chain",
  Teddy-*inspired* but scalar). Only candidates run the exact KMP.
- **Opt-in `ONPAIR_INNER_SIMD=1`**: `scan_contains_inner` — AVX2 multi-range
  test of the INNER token set (DEFINITE + completing/reachable sparse ranges)
  over the whole code stream. Sound necessary filter. A **needle-dependent wash**
  (google ~1.3× win; i.yandex loss); disabled when >16 ranges.
- **Opt-in `ONPAIR_FUNNEL=1`**: `scan_contains_funnel` — SIMD INNER reject →
  scalar chain on survivors → KMP. **No net win** (callgrind +0.66% Ir): both
  passes touch every code, so layering doesn't break the per-code wall.

## The key finding (why contains can't go SIMD on the codes)

Proven by measurement (analysis tools kept in the test module, see below):

- **Prefix wins because the sort aligns with the query**: "starts with N" ⇒
  tokens whose *leading* bytes = N ⇒ **one contiguous id range** ⇒ 1 SIMD range
  test.
- **Contains can't** because its relevant token sets are defined by *suffix /
  internal* bytes, which the prefix-sort scatters uniformly across the id space.
  Measured: the OPEN set for `google` is 782 tokens in ~1000 separate runs.
- Every SIMD shape on the **code stream** was measured and fails:
  - lt/gt ranges: 19–63× false-positive even with 64 ranges.
  - Teddy nibble/byte fingerprint of the code id: 25–63× FP (token ids are
    structureless labels — no fingerprint).
  - gather `class[code]`: slower than scalar (no hardware gather win).
- The token-DFA *continuation* transitions ARE contiguous (the INNER filter
  exploits this), but they're a weak filter, so SIMD-izing them is a wash.
- A *sound* SIMD contains filter only exists on **decoded bytes** (classic
  byte-Teddy/memmem), which costs the ~86 ms decode — more than the scan saves.

## Open lever (not done): LPM-aware INNER pruning

For `%google%` the SIMD INNER filter is dominated by the state-5 (`googl`+`e…`)
completion ranges: ~1554 of 1565 filtered tokens are "starts with e / le".
**Empirically, state 5 is reached 0 times across all 1M rows** (tool:
`reached_states`) because greedy LPM never pauses a token boundary at `googl`
when a longer `google` token exists. The transition-fixpoint (`reachable_states`)
can't prove this (it ignores LPM, marks state 5 reachable via `goog`→`l`→`e`).
**A sound LPM-aware reachability proof** — "no token chain can land a boundary at
state s that a longer token would have absorbed" — would drop google's filter
from ~1559 tokens to ~5, turning the wash into a likely decisive win and
plausibly beating memmem. This is the recommended next step.

## Benchmarks & how to run

`benches/search.rs`. Corpus via env; needles auto-bucketed or overridden.

```bash
# Real ClickBench (download once): URL column, 1M rows
curl -sSL https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_0.parquet -o /tmp/hits_0.parquet
ONPAIR_BENCH_PARQUET=/tmp/hits_0.parquet ONPAIR_BENCH_COLUMN=URL ONPAIR_SEARCH_BITS=16 \
  cargo bench --bench search

# Real ClickBench LIKE query literal (e.g. URL LIKE '%google%')
ONPAIR_NEEDLES="contains:google,prefix:http://www.google" \
  ONPAIR_BENCH_PARQUET=/tmp/hits_0.parquet ONPAIR_BENCH_COLUMN=URL cargo bench --bench search

# FineWeb (HuggingFace): text column, capped rows (docs are ~3 KB each)
curl -sSL "https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/main/data/CC-MAIN-2013-20/000_00000.parquet" -o /tmp/fineweb.parquet
ONPAIR_BENCH_PARQUET=/tmp/fineweb.parquet ONPAIR_BENCH_COLUMN=text ONPAIR_BENCH_MAX_ROWS=50000 \
  ONPAIR_NEEDLES="contains:government,contains:photosynthesis,prefix:The " cargo bench --bench search
```

Bench env vars: `ONPAIR_BENCH_PARQUET`, `ONPAIR_BENCH_COLUMN`,
`ONPAIR_BENCH_MAX_ROWS`, `ONPAIR_SEARCH_BITS` (default 16),
`ONPAIR_NEEDLES="mode:text,…"` (mode = contains|prefix). Runtime:
`ONPAIR_NO_SIMD`, `ONPAIR_INNER_SIMD`, `ONPAIR_FUNNEL`.

Bench groups: `prefix` / `prefix_mask` / `prefix_no_index` (A/B the index),
`contains`, `*_arrow` (memmem/starts_with + `collect_bool` over decompressed
bytes — faithful Arrow kernel), `*_decompress_arrow` (decode then scan),
`copy_all_codes` / `scan_all_codes` / `first_code_per_row` (rooflines). Every
run cross-checks compressed-domain counts vs brute force.

## Analysis tools kept in `src/search/mod.rs` tests (run with `--ignored --nocapture`)
Need a dumped corpus: `ONPAIR_SEARCH_DUMP=/tmp/cppdump` on a bench run writes
`corpus.bin`; then `ONPAIR_CORPUS=/tmp/cppdump/corpus.bin ONPAIR_NEEDLE=google`.
- `dfa_dump` — byte-level KMP DFA + token classification.
- `token_dfa` — token-level DFA in dict space (base RLE + sparse ranges).
- `inner_ranges_dump` — exact SIMD ranges the prefilter tests, with token bytes.
- `boundary_states` / `reached_states` — DFA reachability (the LPM-pruning probe).

## C++ comparison
`benchmarks/onpair-bench/cpp-bench` is the reference C++ (token automata, the
Rust port's origin). Head-to-head on identical data: **prefix Rust 15–35× over
C++** (C++ lacks the `first_codes` side-table + SIMD); **contains within ~10%**
(same LLVM, instruction-identical hot loop — verified in asm). The gap is
algorithm (the side-table), not language. Bit-packing was disproven as a factor
(a bits sweep showed tighter packing made C++ *slower*).

## Status of each piece
- Prefix + `first_codes` + AVX2 + bitmap-merge: **shipped, default, big win.**
- Contains scalar 2-code chain: **shipped, default**, modest win over baseline KMP.
- INNER SIMD / 3-layer funnel: **opt-in**, measured no net win, kept as recorded
  experiments + the foundation for LPM pruning.
- Arrow `collect_bool` baselines, `ONPAIR_NEEDLES`, `ONPAIR_BENCH_MAX_ROWS`,
  Binary-column parquet reading: **shipped** (bench infra).

## A note on process for the next session
Several intermediate commits this session shipped fabricated benchmark numbers
when measurement commands silently produced empty output (env-var passing,
callgrind globs, table parsing). They were caught and amended, but: **always
print and read the raw bench/callgrind output before quoting a number; never
infer a figure.** The current HEAD's numbers are the verified ones.
