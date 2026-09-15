The substring integration is implemented on `feat/search-prefilter`, starting from PR `1cd8b99c3681223e189a2761be5570212d5440b4` and retaining refactor `78f53c696475eb2c67ad1c3737791a714feb06c8` as a merge ancestor. PR provides the graph, cut sweep, matcher/resolver pipeline and exact walker. Refactor contributes explicit planning facts, configured kernel choices, clearer preparation/execution boundaries, and fixed one-point NEON preparation.

The feature branch contains a merge baseline followed by separate layout, planning, API and specialization commits. No remote history was rewritten and nothing was pushed.

**Public behavior.** `prefilter_matches` appends exact matching row indices. `prefilter_candidates` remains an alias with the PR signature and exact semantics. `BytesVerifier` is removed, including its export and examples: this is an explicit API removal. `rows_containing_prefiltered` retains PR's Vec result. Standalone KMP remains available independently.

The graph path requires greedy row tokenization with the same conformant dictionary. Graph needles remain limited to 65,535 bytes; KMP remains limited to 255. The current walk's quadratic repetitive-needle case is unchanged. Direct predecessors, local/merged-window KMP and DFA work are deferred.

**Layout and ownership.**

```text
src/search/substring/
  mod.rs                   public facade
  query.rs                 graph/cover/walk preparation and exact orchestration
  alignment/               graph.rs, mincut.rs, cover.rs
  plan.rs                  cover selection
  plan/                    facts.rs, cost.rs, select.rs
  scan/                    dispatch.rs, matcher/, resolver/, bench/
  verify/                  walk.rs, kmp.rs
  tests.rs                 public pipeline
```

`CoverShape`, `AnalysisFacts`, `RegionFacts`, `ScanFacts` and `TargetCaps` retain the refactor's explicit naming. `ScanPlan` and `select_scan_plan` describe the combined matcher/resolver choice. `Kernel` variants carry target-specific configuration. Query compilation lives in `query.rs`; actual cost-based selection lives in `plan/`, so the old `plan::plan` and competing `compile.rs` pipeline are absent.

**Validation.** Rust 1.91.0 with locked dependencies: 197 unit tests and three doc tests pass; four external-corpus calibration tests remain ignored. Formatting, all-target/all-feature build, Clippy, private documentation and native tests pass. Cross-checks cover x86_64-apple-darwin default AVX2 selection and an explicit AVX-512 build. This machine supplies native ARM64/NEON evidence, not native x86 execution.

The independent directed audit passed 98,301 analyses and 46,580,500 row comparisons. The deeper audit passed 263,106 graphs, 483,967,001 edge checks, 215,092,307 hit checks and 373,468 forced exact scans. It exercised 14,136 fixed one-point scans over both resolvers, both packing settings and both offset widths. The known non-greedy counterexample remains reproducible and is outside the newly documented precondition.

8,832 native synthetic planning cases retain PR's matcher family, resolver, effective packing choice and modeled scan cost (the one-point implementation is normalized to its EqOr family). The graph and walker algorithms remain PR's, apart from relocation/visibility and the corrected google test. The later dictionary-access simplification below preserves the walk's suffix semantics.

**Dictionary-access review.** Removed `CompactDictionaryView::token_window`; the walker uses the existing checked `DictionaryView::token_len` and `token_ptr`, followed by an unaligned 16-byte read and the original little-endian shift/mask comparison. ARM64 and x86-64 assembly retains the same fixed-width comparison with fewer bounds checks. Twelve focused native ARM64 timing cases show no regression (comparison time ratios 0.800–0.819, short-token rejection 0.604–0.636). These timings measure the suffix predicate, not complete queries. After this change, 199 unit tests pass in both debug and release builds, including 77,824 new suffix-oracle comparisons and invalid-ID rejection; formatting and all-target/all-feature Clippy pass. The earlier larger audits and query timings above/below remain evidence for their original snapshots. Full assembly, sources and raw timings are in the surrounding workspace at `comparison/token-window/`.

**Performance.** Eight sequential paired runs cover 64 URL/log scenarios across u32/u64 offsets, dictionary widths 9/12, tiny/large regions, absent/rare/common/dense queries, and both reused analysis and complete preparation-plus-scan. Original byte oracles, encoded-input fingerprints and complete cover fingerprints match in every case. On this ARM machine, large absent one-point scans improve about 4.6x and the rare one-point scan about 4.3x. Median merged/PR ratios are 0.994 for prepared scans and 0.997 for full queries; the slowest full-query ratio is 1.040 in these runs. A preparation-only outlier (2.17 versus 1.97 microseconds) did not reproduce in four further alternating paired runs; their mean merged/PR preparation ratio was 0.973. These synthetic warm-cache observations are not a universal performance guarantee or an x86 result.

One-point NEON uses one fixed broadcast and PR's original mask packing, driver and exact resolver. Other fixed shapes were not imported. Original EqOr coefficients conservatively price the specialization so cover choices remain unchanged. Broader retuning, allocation policies and caching remain separate work.

**Platform exception.** M2b permits deferring optional ISA activation without native qualification. Detection and pure costing now agree on compiled/available targets, and dispatch checks them. Generic x86 builds retain the established AVX2-or-table path; AVX-512 remains a build-selected implementation. Automatic AVX-512 multiversion activation and SSE2 additions are deferred, rather than claiming a detector compiles or validates an absent kernel.

**Decision ledger.** All 67 original decision IDs have a final disposition below; deferred entries are intentional follow-ups.

| ID | Implemented disposition |
|---|---|
| A01 | Implemented: exact prefilter_matches plus the PR prefilter_candidates alias; no public candidate-only API. |
| A02 | Retained PR analysis row-count argument for cover costing. |
| A03 | Retained dictionary argument, generic offsets and unit return for exact scans. |
| A04 | Retained PR rows_containing_prefiltered returning exact Vec<usize> results. |
| A05 | Retained PR removal of the decoded convenience method; orchestration lives in query.rs. |
| A06 | Retained graph maximum 65,535 and standalone KMP maximum 255; no new cap. |
| A07 | Retained compiled walk, coverage data and scan estimate in PrefilterAnalysis. |
| A08 | Retained PR generic frequency builder and storage-backed index. |
| A09 | Removed PR token_window after assembly/timing review; existing token_len/token_ptr preserve the fixed-width suffix check. Retained refactor token_payload boundary documentation. |
| G01 | Retained PR offset-node, edge-probe graph. |
| G02 | Retained PR source-to-sink contained-token and prefix edges. |
| G03 | Retained PR forward greedy-chain construction and memoization. |
| G04 | Retained PR incremental dictionary narrowing and malformed-alphabet handling. |
| G05 | Retained PR dictionary candidate discovery, including overlapping payload matches. |
| G06 | Retained PR internal-entry semantics excluding boundary-start duplication. |
| G07 | Retained PR entry limits, saturation and alignment-one early termination. |
| G08 | Retained PR boxed entry sets; arena conversion deferred. |
| G09 | Retained PR direct weighted edge cut; refactor vertex splitting excluded. |
| G10 | Retained PR reusable MinCut topology and u64 capacity sweep. |
| G11 | Retained PR normalized points/ranges; mandatory cover membership table excluded. |
| P01 | Retained PR lambda sweep and pricing of distinct covers. |
| P02 | Retained PR model structure and 8 ns/hit term; known pathological underestimation documented. |
| P03 | Retained original coefficients and extrapolation provenance; one-point specialization uses conservative EqOr pricing. |
| P04 | Implemented explicit TargetCaps and pure selection/costing, separated from feature detection. |
| P05 | Implemented concrete ScanPlan and configured Kernel variants; removed Option<Facts> planning alias. |
| P06 | Retained PR modeled linear/galloping resolver choice; refactor sink thresholds excluded. |
| P07 | Implemented explicit empty execution plus fresh region facts and saturated advisory-count projection. |
| P08 | Retained caller-controlled legacy profitability hint; clarified its limited historical calibration. |
| K01 | Retained PR equality, range, nibble and portable table families. |
| K02 | Retained nibble semantics and target-specific two/three-batch limits in planner admission. |
| K03 | Implemented explicit capability/dispatch boundary and runtime guards for compiled targets. Automatic AVX-512 multiversion activation deferred under M2b's native-qualification exception; both existing build configurations cross-check. |
| K04 | Retained portable table, including x86 without AVX2; SSE2 addition deferred. |
| K05 | Accepted only one-point/no-range NEON specialization after measurement; other shape imports deferred. |
| K06 | Adapted fixed one-point broadcast preparation to the PR mask seam. Other bounded/mixed NEON schedules deferred. |
| K07 | Retained PR AVX2 interval idiom; refactor saturated-subtraction alternative deferred. |
| K08 | Retained PR AVX-512 scheduling; refactor miss-domain/grouped variants deferred. |
| K09 | Retained PR 4,096-code driver; excluded duplicate template/scanner. |
| K10 | Retained PR padded tail and padding-bit clearing; tail redesign deferred. |
| K11 | Retained PR mask representation, including for the accepted specialization. |
| K12 | Retained PR skip-packing policy; AVX-512 selection does not enable a useless skip. |
| K13 | Accepted fixed broadcast without allocation for one-point NEON; other setup and table behavior retained. Caches deferred. |
| K14 | Retained PR output allocation behavior; refactor reservation policy excluded. |
| K15 | Retained PR target-feature and inlining boundaries; new NEON producer uses the existing feature frame. Unqualified x86 activation deferred. |
| V01 | Retained PR LinearSeek/GallopSeek and shared cursor. |
| V02 | Retained failed-hit retry and successful-row completion rules. |
| V03 | Retained PR forward-then-backward exact walk. |
| V04 | Retained existing multi-role lookup and its known quadratic case; predecessor/KMP redesign deferred. |
| V05 | Retained u128 entry-prefix checks and dictionary windows. |
| V06 | Removed BytesVerifier explicitly; relocated standalone KMP without changing its implementation. |
| V07 | Retained exact production Check and test-only Superset; generic verifier interface deferred. |
| T01 | Retained PR graph/solver tests and independent dictionary candidate oracles. |
| T02 | Retained matcher membership oracles; added the fixed one-point producer to them. |
| T03 | Retained PR independent resolver tests for boundaries, density and both offset widths. |
| T04 | Retained exact walker assertions; fixed google needle; ran directed and per-edge/per-hit audits on merged code. |
| T05 | Covered accepted one-point specialization through independent masks and 14,136 forced scans over both resolvers, packing modes and offset widths. Unimported x86-shape tests excluded with their kernels. |
| T06 | Adapted pure capability/admission/empty/subregion properties; excluded assertions freezing dropped sink/reservation thresholds. |
| T07 | Retained exposed KMP and walk tests, empty/storage cases and >255-byte coverage; removed tests exclusive to deleted BytesVerifier. Added exact compatibility-alias test. |
| T08 | Retained cover-cost/sweep/split-scan tests; validated native planner parity and independent paired performance separately. |
| B01 | Retained all PR calibration utilities; updated imports, names, actual-target labels and generated coefficient snippets. External corpus sweeps remain ignored. |
| B02 | Retained PR dev dependencies for calibration; no runtime dependency changes. |
| B03 | Retained PR Cargo.lock byte-for-byte. |
| B04 | Retained PR profiling profile. |
| B05 | Retained PR bench-output ignore policy. |
| B06 | Retained PR/feature-branch publication workflow; refactor develop workflow excluded. |
| L01 | Implemented query/alignment/plan/scan/verify layout and the explicit late-refactor naming map. |
| L02 | Updated exact API, greedy encoding, length/dispatch contracts and benchmark documentation; decoded API removal recorded. |
| L03 | Retained PR lib.rs; unrelated equality/prefix/encoding/storage implementations unchanged. |

**Original differing paths.** All 49 paths are accounted for, including additions Git merged without a textual conflict. Destinations below are relative to this repository; abbreviated substring paths refer to the tree above.

| # | Original path | Destination / disposition |
|---|---|---|
| 01 | `.github/workflows/publish.yml` | `.github/workflows/publish.yml`: disposition recorded by its decision IDs above. |
| 02 | `.gitignore` | `.gitignore`: disposition recorded by its decision IDs above. |
| 03 | `Cargo.lock` | `Cargo.lock`: disposition recorded by its decision IDs above. |
| 04 | `Cargo.toml` | `Cargo.toml`: disposition recorded by its decision IDs above. |
| 05 | `src/column/mod.rs` | `src/column/mod.rs`: disposition recorded by its decision IDs above. |
| 06 | `src/core/dictionary/compact.rs` | `src/core/dictionary/compact.rs`: disposition recorded by its decision IDs above. |
| 07 | `src/lib.rs` | `src/lib.rs`: disposition recorded by its decision IDs above. |
| 08 | `src/search/index/frequency.rs` | `src/search/index/frequency.rs`: disposition recorded by its decision IDs above. |
| 09 | `src/search/mod.rs` | `src/search/mod.rs`: disposition recorded by its decision IDs above. |
| 10 | `src/search/substring/mod.rs` | `src/search/substring/mod.rs`: thin facade and compatibility alias. |
| 11 | `src/search/substring/prefilter/compile.rs` | `query.rs` preparation boundary and `plan.rs` selection; refactor single-cut pipeline excluded. |
| 12 | `src/search/substring/prefilter/cover.rs` | `src/search/substring/alignment/cover.rs`: PR implementation. |
| 13 | `src/search/substring/prefilter/graph.rs` | `src/search/substring/alignment/graph.rs`: PR implementation. |
| 14 | `src/search/substring/prefilter/mincut.rs` | `src/search/substring/alignment/mincut.rs`: PR implementation. |
| 15 | `src/search/substring/prefilter/mod.rs` | `src/search/substring/query.rs`: PR behavior with explicit orchestration. |
| 16 | `src/search/substring/prefilter/plan.rs` | `query.rs` graph/walk construction and `plan.rs` PR cut sweep. |
| 17 | `src/search/substring/prefilter/scan/README.md` | `src/search/substring/scan/README.md`: retained PR responsibility, with required imports/names adapted. |
| 18 | `src/search/substring/prefilter/scan/aarch64/mod.rs` | `scan/dispatch.rs`: adapted capability boundary; alternate scanner excluded. |
| 19 | `src/search/substring/prefilter/scan/aarch64/neon.rs` | `scan/matcher/eq_or.rs`: adapted one-point preparation; other schedules deferred. |
| 20 | `src/search/substring/prefilter/scan/bench/loader.rs` | `src/search/substring/scan/bench/loader.rs`: retained PR responsibility, with required imports/names adapted. |
| 21 | `src/search/substring/prefilter/scan/bench/matcher_fit.rs` | `src/search/substring/scan/bench/matcher_fit.rs`: retained PR responsibility, with required imports/names adapted. |
| 22 | `src/search/substring/prefilter/scan/bench/mod.rs` | `src/search/substring/scan/bench/mod.rs`: retained PR responsibility, with required imports/names adapted. |
| 23 | `src/search/substring/prefilter/scan/bench/resolver_fit.rs` | `src/search/substring/scan/bench/resolver_fit.rs`: retained PR responsibility, with required imports/names adapted. |
| 24 | `src/search/substring/prefilter/scan/bench/utils.rs` | `src/search/substring/scan/bench/utils.rs`: retained PR responsibility, with required imports/names adapted. |
| 25 | `src/search/substring/prefilter/scan/dispatch.rs` | `src/search/substring/scan/dispatch.rs`: retained PR responsibility, with required imports/names adapted. |
| 26 | `src/search/substring/prefilter/scan/matcher/eq_or.rs` | `src/search/substring/scan/matcher/eq_or.rs`: retained PR responsibility, with required imports/names adapted. |
| 27 | `src/search/substring/prefilter/scan/matcher/mod.rs` | `src/search/substring/scan/matcher/mod.rs`: retained PR responsibility, with required imports/names adapted. |
| 28 | `src/search/substring/prefilter/scan/matcher/nibble_n8.rs` | `src/search/substring/scan/matcher/nibble_n8.rs`: retained PR responsibility, with required imports/names adapted. |
| 29 | `src/search/substring/prefilter/scan/matcher/range.rs` | `src/search/substring/scan/matcher/range.rs`: retained PR responsibility, with required imports/names adapted. |
| 30 | `src/search/substring/prefilter/scan/matcher/shared.rs` | `src/search/substring/scan/matcher/shared.rs`: retained PR responsibility, with required imports/names adapted. |
| 31 | `src/search/substring/prefilter/scan/matcher/table.rs` | `src/search/substring/scan/matcher/table.rs`: retained PR responsibility, with required imports/names adapted. |
| 32 | `src/search/substring/prefilter/scan/matcher/tests.rs` | `src/search/substring/scan/matcher/tests.rs`: retained PR responsibility, with required imports/names adapted. |
| 33 | `src/search/substring/prefilter/scan/mod.rs` | `src/search/substring/scan/mod.rs`: retained PR responsibility, with required imports/names adapted. |
| 34 | `src/search/substring/prefilter/scan/policy.rs` | `plan/{facts,cost,select}.rs` and `scan/dispatch.rs`; one coherent policy. |
| 35 | `src/search/substring/prefilter/scan/policy/mod.rs` | `plan/{facts,cost,select}.rs` and `scan/dispatch.rs`; one coherent policy. |
| 36 | `src/search/substring/prefilter/scan/resolver/gallop_seek.rs` | `src/search/substring/scan/resolver/gallop_seek.rs`: retained PR responsibility, with required imports/names adapted. |
| 37 | `src/search/substring/prefilter/scan/resolver/linear_seek.rs` | `src/search/substring/scan/resolver/linear_seek.rs`: retained PR responsibility, with required imports/names adapted. |
| 38 | `src/search/substring/prefilter/scan/resolver/mod.rs` | `src/search/substring/scan/resolver/mod.rs`: retained PR responsibility, with required imports/names adapted. |
| 39 | `src/search/substring/prefilter/scan/resolver/shared.rs` | `src/search/substring/scan/resolver/shared.rs`: retained PR responsibility, with required imports/names adapted. |
| 40 | `src/search/substring/prefilter/scan/resolver/tests.rs` | `src/search/substring/scan/resolver/tests.rs`: retained PR responsibility, with required imports/names adapted. |
| 41 | `src/search/substring/prefilter/scan/sink.rs` | Excluded candidate sink; PR `scan/resolver/` owns exact row completion. |
| 42 | `src/search/substring/prefilter/scan/template.rs` | Excluded competing driver; PR shared mask/resolver seam retained. |
| 43 | `src/search/substring/prefilter/scan/walk.rs` | `verify/walk.rs`: PR verifier; corrected test needle; suffix check uses existing dictionary accessors with boundary tests. |
| 44 | `src/search/substring/prefilter/scan/x86/avx2.rs` | Deferred additional x86 implementations; existing PR kernels retained. Explicit dispatch facts adopted separately. |
| 45 | `src/search/substring/prefilter/scan/x86/avx512.rs` | Deferred additional x86 implementations; existing PR kernels retained. Explicit dispatch facts adopted separately. |
| 46 | `src/search/substring/prefilter/scan/x86/mod.rs` | Deferred additional x86 implementations; existing PR kernels retained. Explicit dispatch facts adopted separately. |
| 47 | `src/search/substring/prefilter/scan/x86/sse2.rs` | Deferred additional x86 implementations; existing PR kernels retained. Explicit dispatch facts adopted separately. |
| 48 | `src/search/substring/prefilter/tests.rs` | `substring/tests.rs` plus responsibility-specific tests; retained properties mapped above. |
| 49 | `src/search/substring/verify.rs` | Removed: BytesVerifier and obsolete exports/examples. |

The independent harnesses and raw logs live in the surrounding comparison workspace: `comparison/integration/`, `comparison/merged-audit/` and `comparison/merged-directed/`. Their source snapshots record file hashes; hooks only expose private boundaries and add the independent audit entry. Full auditing preceded only comment and test-only calibration-snippet cleanup; production tokens were compared afterward. The earlier PR/refactor comparison and fault-injection evidence remain historical controls.
