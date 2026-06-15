#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Gather a corpus of *real* public profiling runs (no synthesis).

Each real profile is a "folded stacks" file (`frame;frame;... count`) produced
by some real profiler. We collect them from public repos that ship real
profiler output as test fixtures across many tools/languages:

  * jonhoo/inferno  - perf, dtrace, ghcprof (Haskell), vsprof, vtune, sample
                      (macOS), xctrace, async-profiler (JVM), vertx perf capture
  * jlfwong/speedscope - stackcollapse sample profiles
  * brendangregg/FlameGraph - example perf capture

Every file is treated as one profiling *run*. All runs are concatenated into a
single shared-dictionary column (one stack per row); a `.runs` sidecar records
rows-per-run and a `.sources` sidecar records each run's origin. This is the
heterogeneous "store of many different profiles" case (contrast with the
same-workload resampling in build_perf_runs.py).

Usage: python3 gather_real_profiles.py
"""
import hashlib
import os
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
CACHE = HERE / "data" / "cache"
REPOS = {
    "inferno": "https://github.com/jonhoo/inferno",
    "speedscope": "https://github.com/jlfwong/speedscope",
}
COUNT_RE = re.compile(r"^(.+?)\s+\d+(?:\.\d+)?$")
# Clearly-synthetic fixtures (not real profiler output).
DENY = {"alternating.txt", "recursion.txt"}


def clone(name: str, url: str) -> Path:
    dest = CACHE / "repos" / name
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  cloning {url}")
    subprocess.run(["git", "clone", "--depth", "1", "-q", url, str(dest)], check=True)
    return dest


def is_folded(path: Path):
    """Return stack list if `path` looks like real folded stacks, else None."""
    if path.name in DENY:
        return None
    try:
        lines = [l.rstrip("\n") for l in open(path, encoding="utf-8", errors="ignore") if l.strip()]
    except OSError:
        return None
    if len(lines) < 10:
        return None
    stacks = []
    good = depth = 0
    for l in lines:
        m = COUNT_RE.match(l)
        if not m:
            continue
        good += 1
        stack = m.group(1)
        frames = stack.split(";")
        depth += len(frames)
        stacks.append(frames)
    if good < 0.8 * len(lines) or not stacks:
        return None
    if depth / len(stacks) < 2.0:  # require real call-stack depth
        return None
    return stacks


def main() -> None:
    roots = [clone(n, u) for n, u in REPOS.items()]
    # FlameGraph capture we already collapsed (optional).
    extra = [CACHE / "folded.txt"]

    seen: set[str] = set()
    runs: list[tuple[str, list[list[str]]]] = []
    candidates: list[Path] = []
    for root in roots:
        for dp, _, fs in os.walk(root):
            if ".git" in dp:
                continue
            for f in fs:
                if f.endswith((".txt", ".folded", ".collapsed")):
                    candidates.append(Path(dp) / f)
    candidates += [p for p in extra if p.exists()]

    for p in sorted(candidates):
        stacks = is_folded(p)
        if stacks is None:
            continue
        h = hashlib.md5("\n".join("\t".join(s) for s in stacks).encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        rel = str(p).replace(str(CACHE / "repos") + "/", "").replace(str(CACHE) + "/", "")
        runs.append((rel, stacks))

    out = HERE / "data" / "perf_corpus.lst"
    runs_meta = HERE / "data" / "perf_corpus.runs"
    src_meta = HERE / "data" / "perf_corpus.sources"
    total_rows = total_elems = 0
    with open(out, "w") as f, open(runs_meta, "w") as fm, open(src_meta, "w") as fs:
        for name, stacks in runs:
            for frames in stacks:
                f.write("\t".join(frames))
                f.write("\n")
                total_elems += len(frames)
            fm.write(f"{len(stacks)}\n")
            fs.write(f"{name}\n")
            total_rows += len(stacks)
    print(f"wrote {out.name}: {len(runs)} real runs, {total_rows} stacks, {total_elems} frames")
    print("sources:")
    for name, stacks in runs:
        print(f"  {len(stacks):4d} stacks  {name}")


if __name__ == "__main__":
    main()
