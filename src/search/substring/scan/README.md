# scan

Finds the rows a probe may occur in, over a stream of codes. Two stages with a
fixed seam of bits between them, so any matcher can be paired with any resolver.

    codes -> matcher -> bit mask -> resolver -> candidate rows

The probe is K token codes and R code ranges, the shape the planner's cover has.
The matcher (`matcher/`) sets one bit per matching code, 105 GB/s at K=1 down to
10 GB/s for the scalar `table` past K=24. The resolver (`resolver/`) turns set
bits into ascending rows, emitting a row on its first hit, and runs each hit
through the alignment walk in `walk.rs` so the rows are exact. `policy` picks
both kernels from cost models fitted to the sweeps in `bench/`. Refit with
`cargo test --release --lib scan::bench -- --ignored --nocapture` and paste the
printed constants into `policy`.
