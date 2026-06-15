#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Scale the single real perf capture up to ~1000 perf runs.

We only have one real `perf` capture, but the realistic observability scenario
is a *store of ~1000 profiles*: many runs of the same service (similar) plus
some runs of other workloads (different). We model that WITHOUT inventing any
frames: every stack emitted is a real stack from the real capture; only the
*run population* is constructed.

Construction:
  * Group the real folded stacks by root frame -> workload "families"
    (java, wrk, swapper, ...).
  * Define run cohorts that draw from those families with different mixes:
      - "java-service" (similar cohort): mostly java stacks  -> high overlap
      - "wrk-loadgen"  : mostly wrk stacks
      - "mixed"        (different cohort): swapper/perf/other + a little java
  * Each run bootstrap-resamples R stacks (with replacement) from its cohort's
    stack pool, weighted by the real per-stack sample counts. Each sampled
    stack becomes one row.

Output: data/perf_runs.lst  (one stack per line, frames TAB-separated, runs
concatenated in order) and data/perf_runs.runs (one integer row-count per run).
"""
import argparse
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
FOLDED = HERE / "data" / "cache" / "folded.txt"


def load_families():
    fams = defaultdict(list)  # root -> list[(frames, weight)]
    for line in open(FOLDED):
        line = line.strip()
        if not line:
            continue
        stack, cnt = line.rsplit(" ", 1)
        frames = stack.split(";")
        fams[frames[0]].append((frames, int(cnt)))
    return fams


def cohort_pool(fams, primary, others, mix):
    """Weighted pool: `mix` fraction of weight to `others`, rest to primary."""
    pool = []
    prim = fams.get(primary, [])
    pw = sum(w for _, w in prim) or 1
    for frames, w in prim:
        pool.append((frames, (1.0 - mix) * w / pw))
    ow = sum(sum(w for _, w in fams.get(o, [])) for o in others) or 1
    for o in others:
        for frames, w in fams.get(o, []):
            pool.append((frames, mix * w / ow))
    stacks = [f for f, _ in pool]
    weights = [w for _, w in pool]
    return stacks, weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-samples", type=int, default=80)
    ap.add_argument("--max-samples", type=int, default=320)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    fams = load_families()
    fam_names = list(fams)

    # cohort: (probability, primary family, other families, foreign-mix fraction)
    cohorts = [
        (0.70, "java", ["swapper", "gmain"], 0.05),    # similar: java service
        (0.20, "wrk", ["swapper"], 0.05),              # wrk load generator
        (0.10, None, fam_names, 1.0),                  # different: everything
    ]

    out = HERE / "data" / "perf_runs.lst"
    runs_meta = HERE / "data" / "perf_runs.runs"
    total_rows = 0
    total_elems = 0
    cohort_counts = defaultdict(int)
    with open(out, "w") as f, open(runs_meta, "w") as fm:
        for _ in range(args.runs):
            x = rng.random()
            acc = 0.0
            for prob, primary, others, mix in cohorts:
                acc += prob
                if x <= acc:
                    break
            if primary is None:
                # "different" run: pick one random family as primary each time
                primary = rng.choice(fam_names)
                others = [n for n in fam_names if n != primary]
                mix = rng.choice([0.0, 0.3, 0.6])
            cohort_counts[primary] += 1
            stacks, weights = cohort_pool(fams, primary, others, mix)
            r = rng.randint(args.min_samples, args.max_samples)
            chosen = rng.choices(stacks, weights=weights, k=r)
            for frames in chosen:
                f.write("\t".join(frames))
                f.write("\n")
                total_elems += len(frames)
            fm.write(f"{r}\n")
            total_rows += r
    print(f"wrote {out.name}: {args.runs} runs, {total_rows} rows (stacks), {total_elems} elements")
    print("cohort primary-family counts:", dict(cohort_counts))


if __name__ == "__main__":
    main()
