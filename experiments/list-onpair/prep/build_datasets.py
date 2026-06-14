#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Build uniform list-of-token datasets from *real* public sources.

Each output `.lst` file holds one list per line; elements are separated by a
TAB. Elements never contain tabs or newlines. These are the inputs to the
generalized integer-OnPair experiment (see ../src/main.rs):

  stacks.lst  call stacks  (Brendan Gregg real perf capture -> folded stacks)
  paths.lst   file paths   (real file tree, split on '/')
  tags.lst    SO-style tags(real Stack Exchange data dump, Posts.xml Tags)
  graph.lst   adjacency    (real SNAP wiki-Vote graph, neighbours per node)

Usage: python3 build_datasets.py [--out DIR]
Downloads are cached under DIR/cache.
"""
import argparse
import gzip
import html
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

PERF_URL = "https://raw.githubusercontent.com/brendangregg/FlameGraph/master/example-perf-stacks.txt.gz"
COLLAPSE_URL = "https://raw.githubusercontent.com/brendangregg/FlameGraph/master/stackcollapse-perf.pl"
SNAP_URL = "https://snap.stanford.edu/data/wiki-Vote.txt.gz"
SE_URL = "https://archive.org/download/stackexchange/3dprinting.stackexchange.com.7z"


def fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "list-onpair/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest


def write_lst(path: Path, lists) -> None:
    n = 0
    elems = 0
    with open(path, "w") as f:
        for lst in lists:
            if not lst:
                continue
            f.write("\t".join(lst))
            f.write("\n")
            n += 1
            elems += len(lst)
    print(f"  wrote {path.name}: {n} lists, {elems} elements")


def build_stacks(cache: Path, out: Path) -> None:
    print("stacks (real perf capture -> folded call stacks)")
    perf_gz = fetch(PERF_URL, cache / "perf.txt.gz")
    perf = cache / "perf.txt"
    if not perf.exists():
        perf.write_bytes(gzip.decompress(perf_gz.read_bytes()))
    collapse = fetch(COLLAPSE_URL, cache / "stackcollapse-perf.pl")
    folded = cache / "folded.txt"
    if not folded.exists():
        with open(folded, "w") as f:
            subprocess.run(["perl", str(collapse), str(perf)], stdout=f, check=True)
    lists = []
    for line in open(folded):
        line = line.strip()
        if not line:
            continue
        stack = line.rsplit(" ", 1)[0]  # drop trailing sample count
        lists.append(stack.split(";"))
    write_lst(out / "stacks.lst", lists)


def build_paths(out: Path, roots) -> None:
    print("paths (real file tree, split on '/')")
    lists = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and ".git/" not in str(p):
                rel = p.relative_to(root.parent)
                lists.append(str(rel).split("/"))
    write_lst(out / "paths.lst", lists)


def build_tags(cache: Path, out: Path) -> None:
    print("tags (real Stack Exchange data dump -> per-question tags)")
    se = fetch(SE_URL, cache / "se.7z")
    posts = cache / "Posts.xml"
    if not posts.exists():
        import py7zr  # noqa: imported lazily; only this dataset needs it

        with py7zr.SevenZipFile(se, "r") as z:
            z.extract(path=cache, targets=["Posts.xml"])
    tag_re = re.compile(r'Tags="([^"]*)"')
    lists = []
    for line in open(posts, encoding="utf-8"):
        m = tag_re.search(line)
        if not m:
            continue
        raw = html.unescape(m.group(1))
        # Two encodings appear across dumps: "<a><b>" and "a|b|".
        if raw.startswith("<"):
            tags = re.findall(r"<([^>]+)>", raw)
        else:
            tags = [t for t in raw.split("|") if t]
        if tags:
            lists.append(tags)
    write_lst(out / "tags.lst", lists)


def build_graph(cache: Path, out: Path) -> None:
    print("graph (real SNAP wiki-Vote -> adjacency list per node)")
    snap_gz = fetch(SNAP_URL, cache / "wiki-Vote.txt.gz")
    txt = cache / "wiki-Vote.txt"
    if not txt.exists():
        txt.write_bytes(gzip.decompress(snap_gz.read_bytes()))
    adj: dict[str, list[str]] = {}
    for line in open(txt):
        if line.startswith("#"):
            continue
        a, b = line.split()
        adj.setdefault(a, []).append(b)
    # Sort neighbours numerically: a realistic on-disk adjacency layout.
    lists = [sorted(v, key=int) for _, v in sorted(adj.items(), key=lambda kv: int(kv[0]))]
    write_lst(out / "graph.lst", lists)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument(
        "--path-roots",
        nargs="*",
        default=[str(Path(__file__).resolve().parents[4] / "vortex"),
                 str(Path(__file__).resolve().parents[4] / "onpair")],
    )
    args = ap.parse_args()
    out = Path(args.out)
    cache = out / "cache"
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    build_stacks(cache, out)
    build_paths(out, args.path_roots)
    build_tags(cache, out)
    build_graph(cache, out)
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
