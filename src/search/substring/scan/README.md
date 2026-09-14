# Substring scanning

The query layer compiles an alignment graph, selects a probe cover and builds
an exact walker. Scanning executes a concrete plan over one region:

```text
codes -> matcher -> hit mask -> resolver -> graph verifier -> exact rows
```

`matcher/` produces a bit per covered code. `resolver/` maps hits to rows using
linear or galloping search. Every hit is checked by `../verify/walk.rs`. A
failed check advances to the next hit, even within the same row. Only a
successful check completes a row; the scan appends it once and skips its
remaining hits. Existing output contents are preserved.

`../plan/facts.rs` describes cover shape, indexed statistics, current region
and target capabilities. `../plan/select.rs` chooses the matcher, resolver and
packing strategy without reading code values. `../plan/cost.rs` retains the
PR calibration coefficients. Cover costing and execution use the same target
capabilities, supplied by `dispatch.rs`; subregion scans derive fresh facts.

The native AArch64 path uses NEON. x86 builds compile AVX2, or AVX-512BW when
selected through build flags, and check runtime availability before dispatch.
Automatic AVX-512 selection in a generic x86 binary awaits native qualification;
detecting a feature does not make an uncompiled implementation available.
The table matcher remains a portable implementation. AVX2 nibble matching is
limited to two batches; NEON and AVX-512 allow three, with eight points per batch.

The graph prefilter requires greedy encoding with the same conformant dictionary.
It supports needles through 65,535 bytes; standalone KMP supports 255 bytes.
Empty needles match all rows, including empty rows. Advisory frequencies affect
planning but never remove required probes. The existing walker can still take
quadratic time on repetitive long needles; changing it is follow-up work.

The retained calibration utilities require the original external datasets.
Run `cargo test --release --lib scan::bench -- --ignored --nocapture` and place
fitted constants in `../plan/cost.rs`. Labels identify the target actually
available to the run. AVX2 nibble coefficients remain extrapolated, and the
constant walk cost does not describe its known pathological case. Correctness
oracles and controlled performance measurements remain separate checks.

For one point and no ranges on NEON, the selected equality matcher uses a fixed
broadcast instead of allocated probe vectors. Its packing and resolver remain
the PR implementation. The existing equality coefficients are retained as a
conservative estimate, so this specialization does not alter cover selection.
The integration measurements cover the exact pipeline as well as reused analysis;
this is not a general adoption of the refactor's architecture-specific scanners.
