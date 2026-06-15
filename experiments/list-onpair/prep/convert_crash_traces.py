#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Convert public *crash / exception* stack-trace datasets to our `.lst` format.

Unlike the profiling corpora (folded sampled stacks), these are stack traces
from bug-report / crash-deduplication research. Each report carries one or more
exception stack traces; we emit one row per trace, tab-separated frames.

Sources (downloaded to data/cache, see gather note in README):
  * AERI (Eclipse)  - per-problem JSON files; frame = `class.method` (cN.mN)
  * JetBrains EMSE  - Ubuntu(campbell)/Eclipse/NetBeans/Gnome JSON arrays;
                      frame = `function`. (Zenodo 5746044)

These are large (gnome alone is 3.5 GB), so we stream with ijson and cap rows
per dataset for tractability; the cap and the seen/taken counts are reported.
"""
import glob
import json
import os
from pathlib import Path

import ijson

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"
CACHE = DATA / "cache"
CAP = 300_000          # max traces per standalone dataset
COMBINED_CAP = 80_000  # max traces per dataset inside the combined corpus
MAX_BYTES = 60_000_000          # byte budget per standalone dataset
COMBINED_MAX_BYTES = 15_000_000  # byte budget per project inside the combined corpus
MIN_FRAMES = 2         # a trace needs real depth to be interesting


def clean(fr: str) -> str:
    return fr.replace("\t", " ").replace("\n", " ").strip()


def emit(rows, fh):
    """Write one trace per row; return number written."""
    n = 0
    for frames in rows:
        frames = [clean(f) for f in frames if f]
        frames = [f for f in frames if f]
        if len(frames) >= MIN_FRAMES:
            fh.write("\t".join(frames))
            fh.write("\n")
            n += 1
    return n


def jetbrains_traces(path: str):
    """Yield each trace (list of frame strings) from a JetBrains EMSE file."""
    with open(path, "rb") as f:
        for rec in ijson.items(f, "item"):
            st = rec.get("stacktrace")
            traces = st if isinstance(st, list) else [st] if isinstance(st, dict) else []
            for tr in traces:
                if isinstance(tr, dict):
                    yield [fr.get("function") for fr in tr.get("frames", [])]


def aeri_traces(files):
    """Yield each trace from the AERI per-problem JSON files."""
    for p in files:
        try:
            d = json.load(open(p))
        except (OSError, ValueError):
            continue
        for tr in d.get("stacktraces", []) or []:
            if isinstance(tr, list):
                yield [f"{fr.get('cN','')}.{fr.get('mN','')}" for fr in tr if isinstance(fr, dict)]


def convert(name: str, traces, cap: int, max_bytes: int) -> int:
    out = DATA / f"{name}.lst"
    written = seen = 0
    with open(out, "w") as fh:
        for tr in traces:
            seen += 1
            if emit([tr], fh):
                written += 1
            if written >= cap or fh.tell() >= max_bytes:
                break
    print(f"  {name:16} {written:7d} traces, {out.stat().st_size//1_000_000}MB "
          f"({seen} scanned) -> {out.name}")
    return written


def main() -> None:
    jb = CACHE / "jb" / "EMSE_data"
    aeri = sorted(glob.glob(str(CACHE / "aeri" / "output_problems" / "*.json")))
    sources = {
        "crash_eclipse": lambda: jetbrains_traces(str(jb / "eclipse_2018" / "eclipse_stacktraces.json")),
        "crash_netbeans": lambda: jetbrains_traces(str(jb / "netbeans_2016" / "netbeans_stacktraces.json")),
        "crash_gnome": lambda: jetbrains_traces(str(jb / "gnome_2011" / "gnome_stacktraces.json")),
        "crash_ubuntu": lambda: jetbrains_traces(str(jb / "campbell_dataset" / "campbell_stacktraces.json")),
        "crash_aeri": lambda: aeri_traces(aeri),
    }
    print("standalone datasets:")
    for name, gen in sources.items():
        convert(name, gen(), CAP, MAX_BYTES)

    # Combined heterogeneous corpus: Java (eclipse, netbeans, aeri) + C/C++
    # (gnome, ubuntu), each project a "run" for the shared-vs-per-run analysis.
    print("combined corpus (crash_corpus):")
    out = DATA / "crash_corpus.lst"
    runs = DATA / "crash_corpus.runs"
    srcs = DATA / "crash_corpus.sources"
    with open(out, "w") as fh, open(runs, "w") as fr, open(srcs, "w") as fs:
        for name, gen in sources.items():
            n = 0
            start = fh.tell()
            for tr in gen():
                if emit([tr], fh):
                    n += 1
                if n >= COMBINED_CAP or fh.tell() - start >= COMBINED_MAX_BYTES:
                    break
            fr.write(f"{n}\n")
            fs.write(f"{name}\n")
            print(f"  {name:16} {n:7d} traces")


if __name__ == "__main__":
    main()
