# Compressed-domain LIKE search — optimization memory

Durable record of the prefix/contains search work: what was built, what won,
what was tried and **failed (with the reason)**, so future sessions don't
re-walk dead ends. Code lives in `src/search/`; benches in `benches/search.rs`.

> Process note (learned the hard way): **never quote a benchmark number you have
> not printed and read from raw output.** If a measurement command yields empty
> output, fix the command — do not infer. Verify `cargo test` + `clippy` before
> every commit. The box is contended; prefer callgrind for deterministic perf.

## Data model (why prefix and contains differ fundamentally)

A column compresses each string into a stream of `u16` dictionary token-ids
("codes") over a **lexicographically-sorted** dictionary (ids in sort order; 256
single-byte tokens always present). LIKE runs token-level automata directly over
codes — rows are never decompressed.

The sort key is the token's **leading** bytes. This is the hinge for everything:

- **Prefix** ("starts with N") needs tokens whose *leading* bytes = N → that's
  **one contiguous id range** (`DictView::prefix_range`) → a single SIMD range
  test. Aligned with the sort ⇒ huge structural win.
- **Contains** ("N anywhere") needs tokens by their *suffix/internal* bytes,
  which the leading-byte sort scatters uniformly across the id space. Not a
  range; not fingerprint-able on ids (they're structureless labels). This is why
  every SIMD attempt on the contains code stream fails (see "dead ends").

## What shipped (default)

### Prefix — the strong, structural win
- `Column::first_codes: Option<Vec<u16>>` — one first-token id per row, built at
  compress time (+~7% column size on URLs; `None` ⇒ generic scan).
- `scan_prefix`: pass 1 = branchless SIMD unsigned range test
  `begin ≤ first_code ≤ last`, plus an equality lane `== q0` for multi-token
  needles; pass 2 confirms only the `== q0` candidates.
- AVX2 kernels (`prefilter_accept*_avx2`), runtime-detected (`avx2_enabled`),
  scalar fallback. `ONPAIR_NO_SIMD=1` forces scalar.
- `prefix_mask`: `search()` writes accept bits straight into `RowMask` words
  (no per-row callback).
- **Result (real ClickBench URL, 1M rows): ~30–40× over memmem/starts_with on
  *decompressed* bytes, ~350–600× over decompress+scan.** Same on FineWeb.

### Contains — scalar 2-code chain in front of exact KMP
- `KmpAutomaton`: token-level KMP. `base[t]` = exit state feeding token `t` from
  state 0; `sparse` = per-state exception ranges; `matches()` is the exact
  confirmer.
- `chain_table` + `row_chain`: per token, three sound flags — DEFINITE (token
  contains the whole needle ⇒ row matches), OPEN (`base≠0`, can start a spanning
  match), CONT (can continue one). A row is a candidate iff DEFINITE present or
  an **adjacent OPEN→CONT pair** (Teddy-*inspired* but scalar). Only candidates
  pay the exact KMP.
- **Result:** beats `decompress+memmem` 3–6× (decode ~46–100 ms dominates), but
  ~parity-to-loss vs in-memory memmem; **loses 3–4× on FineWeb** (long
  ~499-codes/row docs hit the per-code scalar-gather throughput wall).

## Opt-in experiments (measured no net win; kept as foundation/record)

- `ONPAIR_INNER_SIMD` → `scan_contains_inner`: AVX2 multi-range test of the INNER
  token set (DEFINITE + completing/reachable sparse ranges) over the whole code
  stream. Sound necessary filter; ranges are contiguous so it vectorises. **A
  needle-dependent wash** — INNER is far less selective (13–38% candidate) than
  the scalar chain (~0.5%). Disabled above `INNER_RANGE_BUDGET = 16` ranges.
- `ONPAIR_FUNNEL` → `scan_contains_funnel`: SIMD INNER reject → scalar chain on
  survivors → KMP. **No net win** — callgrind: scalar 570,409,783 Ir → funnel
  574,155,207 Ir (+0.66%). Both passes must touch every code, so layering is
  "scalar + one extra full pass"; running the chain on only ~13% survivors only
  just pays that back. **Layering cannot break the per-code throughput wall.**

`inner_ranges` is tightened by two *proven-sound* prunes (each removes only false
positives): completing-only (`target == match_state`) and reachable-entry
(`reachable_states` fixpoint). Verified by brute-force cross-checks.

## Dead ends — SIMD on the contains code stream (all measured, all fail)

The recurring question "can't we SIMD-filter the codes for contains?" — answered
no, three ways, because token ids encode *prefix* order but contains needs
*suffix* structure:

- **lt/gt id ranges**: the OPEN set scatters (`google`: 782 tokens in ~1000
  runs). Even 64 ranges give 19–63× false positives.
- **Teddy nibble/byte fingerprint of the code id**: 25–63× FP — code ids are
  arbitrary labels, no fingerprint structure (measured low-byte, high-byte, and
  both-byte AND).
- **gather `class[code]`**: slower than the scalar pipelined loads (no hardware
  gather win on this µarch).

The DFA's *continuation* transitions ARE contiguous (the INNER filter exploits
this), but they're a weak filter, so SIMD-izing them is a wash (above). A sound
SIMD contains filter only exists on **decoded bytes** (classic byte-Teddy/memmem)
— which costs the ~86 ms decode, more than the scan saves.

## Experiment #1 — LPM-aware INNER pruning: DISPROVED (unsound)

Hypothesis: for `%google%` the INNER filter is dominated by the state-5
(`googl`+`e…`) completion ranges (~1554 of 1565 tokens, "starts with e / le"),
and state 5 is reached **0 times across all 1M corpus rows**, so maybe greedy LPM
makes it unreachable and the range can be dropped (collapsing the filter to ~5
tokens, possibly beating memmem).

**Result: UNSOUND — disproved by construction.** The `lpm_reach_witness` probe
(in the test module) feeds crafted + 2M random strings through the *real* LPM
tokenisation and records which DFA boundary states each reaches. Every partial
state is witnessed reachable, including state 5: the byte string `"googl"` itself
tokenises with a boundary at state 5 (there is no `"google"` token to absorb it
without a trailing `e`). So a value like `"…googl"` adjacent to an `e…` token
DOES complete a match via state 5 — dropping that range would cause false
negatives. The empirical "0×" was a property of the URL *corpus*, not the
*dictionary*. Witnesses: state1 "g", s2 "go", s3 "goo", s4 "googoo", s5 "googl".

Conclusion: boundary-state reachability cannot be tightened by an LPM argument —
any prefix of the needle is a constructible boundary value. The INNER filter
(and the `reachable_states` transition fixpoint) is already as tight as soundness
allows. **No remaining lever to make contains beat memmem on the token stream.**

## Public API (matcher)

- `Column::as_search_parts() -> SearchParts` (or build `SearchParts` by struct
  literal from deserialized storage; fields are `pub`).
- `SearchParts::search(Pattern) -> RowMask` / `search_callback(Pattern, |row|…)`.
- `Pattern::{Prefix, Contains}(&[u8])`.
- `RowMask`: `len()`, `is_empty()`, `as_words() -> &[u64]` (compose with engine
  selection vectors via word-wise AND/OR), `into_parts() -> (Vec<u64>, usize)`.

## Hot-path notes (cleanup applied)

- Per-row offset conversion uses `Offset::as_usize()` (branchless truncating
  inverse of `from_usize`), not `to_usize().expect(...)` — offsets are validated
  at construction, so the conversion is infallible by construction. `to_usize`
  remains for the genuinely-fallible validation paths.
- `SearchParts::row_codes(r)` factors the per-row slice.
- The `vec![0u64; words]` filter buffers are per-*query*, not per-row, and the
  zero-fill is required (SIMD kernels assign only full words; the tail needs 0).
- The decompress in-loop code bounds check is already a `#[cold]` never-taken
  branch — not excess.

## C++ comparison
`benchmarks/onpair-bench/cpp-bench` is the reference C++ (token automata, the
Rust port's origin). Head-to-head on identical data: **prefix Rust 15–35× over
C++** (C++ lacks the `first_codes` side-table + SIMD); **contains within ~10%**
(same LLVM, instruction-identical hot loop, verified in asm). The gap is
algorithm (the side-table), not language. Bit-packing was disproven as a factor
(a bits sweep showed tighter packing made C++ *slower*).

## Benchmarks & reproduction

`benches/search.rs`. Env: `ONPAIR_BENCH_PARQUET`, `ONPAIR_BENCH_COLUMN`,
`ONPAIR_BENCH_MAX_ROWS`, `ONPAIR_SEARCH_BITS` (default 16),
`ONPAIR_NEEDLES="mode:text,…"` (mode = contains|prefix). Runtime toggles:
`ONPAIR_NO_SIMD`, `ONPAIR_INNER_SIMD`, `ONPAIR_FUNNEL`. Every run cross-checks
compressed-domain counts vs brute force.

```bash
# Real ClickBench (URL column), incl. the real `URL LIKE '%google%'` query
curl -sSL https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_0.parquet -o /tmp/hits_0.parquet
ONPAIR_BENCH_PARQUET=/tmp/hits_0.parquet ONPAIR_BENCH_COLUMN=URL \
  ONPAIR_NEEDLES="contains:google,prefix:http://www.google" cargo bench --bench search

# FineWeb (long documents): cap rows to fit memory
curl -sSL "https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/main/data/CC-MAIN-2013-20/000_00000.parquet" -o /tmp/fineweb.parquet
ONPAIR_BENCH_PARQUET=/tmp/fineweb.parquet ONPAIR_BENCH_COLUMN=text ONPAIR_BENCH_MAX_ROWS=50000 \
  ONPAIR_NEEDLES="contains:photosynthesis,prefix:The " cargo bench --bench search
```

Bench groups: `prefix` / `prefix_mask` / `prefix_no_index` (index A/B),
`contains`, `*_arrow` (memmem/starts_with + `BooleanBuffer::collect_bool` over
decompressed bytes — faithful Arrow kernel), `*_decompress_arrow` (decode then
scan), `copy_all_codes` / `scan_all_codes` / `first_code_per_row` (rooflines).

## Analysis tools (in the `src/search/mod.rs` test module, `#[ignore]`)

Run with `--ignored --nocapture`; need a dumped corpus
(`ONPAIR_SEARCH_DUMP=/tmp/cppdump` on a bench writes `corpus.bin`, then
`ONPAIR_CORPUS=/tmp/cppdump/corpus.bin ONPAIR_NEEDLE=google`):
- `token_dfa` — token-level DFA in dict space (base RLE + sparse ranges).
- `inner_ranges_dump` — exact SIMD ranges the prefilter tests, with token bytes.
- `boundary_states` / `reached_states` — DFA reachability (the LPM-pruning probe).
- `inner_probe` — INNER-filter candidate-rate vs the scalar chain.
