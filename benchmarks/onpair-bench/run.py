"""OnPair benchmark orchestrator.

Discovers corpora (.txt directly; .parquet → one .txt per string column under
corpora/.cache/), builds the rust + cpp benchmark binaries on staleness, then
sweeps (corpus × bits 9..=16), printing a markdown summary to stdout and
writing the raw timings to results/<UTC>.json.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPORA = HERE / "corpora"
CACHE = CORPORA / ".cache"
RESULTS = HERE / "results"
RUST_DIR = HERE / "rust-bench"
CPP_DIR = HERE / "cpp-bench"
RUST_BIN = RUST_DIR / "target" / "release" / "rust_bench"
CPP_BUILD = CPP_DIR / "build"
CPP_BIN = CPP_BUILD / "cpp_bench"


# --- corpus discovery + parquet extraction ----------------------------------


def _is_newer(src: Path, dst: Path) -> bool:
    return not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime


def _extract_parquet(parquet: Path) -> list[Path]:
    """One .txt per string column, cached by mtime. Warns on embedded LF."""
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
        import pyarrow as pa  # type: ignore[import-not-found]
    except ImportError:
        print(f"warn: pyarrow not installed, skipping {parquet.name}", file=sys.stderr)
        return []

    CACHE.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    table = pq.read_table(parquet)
    for col_name in table.column_names:
        col = table.column(col_name)
        if not pa.types.is_string(col.type) and not (
            hasattr(pa.types, "is_string_view") and pa.types.is_string_view(col.type)
        ):
            continue
        out = CACHE / f"{parquet.stem}__{col_name}.txt"
        if not _is_newer(parquet, out):
            outputs.append(out)
            continue
        dropped = 0
        with out.open("wb") as fh:
            for chunk in col.chunks:
                for v in chunk.to_pylist():
                    if v is None:
                        continue
                    b = v.encode("utf-8") if isinstance(v, str) else bytes(v)
                    if b"\n" in b:
                        dropped += 1
                        continue
                    fh.write(b)
                    fh.write(b"\n")
        if dropped:
            print(
                f"warn: {parquet.name}:{col_name}: dropped {dropped} rows containing LF",
                file=sys.stderr,
            )
        outputs.append(out)
    return outputs


def discover_corpora(extra: list[Path]) -> list[Path]:
    found: list[Path] = []
    if CORPORA.exists():
        for p in sorted(CORPORA.iterdir()):
            if p.is_dir():
                continue
            if p.suffix == ".txt":
                found.append(p)
            elif p.suffix == ".parquet":
                found.extend(_extract_parquet(p))
    for p in extra:
        p = p.resolve()
        if p.suffix == ".parquet":
            # Stash into cache, treat as parquet.
            CACHE.mkdir(parents=True, exist_ok=True)
            link = CACHE / p.name
            if not link.exists():
                link.symlink_to(p)
            found.extend(_extract_parquet(link))
        else:
            found.append(p)
    return found


# --- build steps ------------------------------------------------------------


def _files_under(*roots: Path) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file():
                out.append(p)
    return out


def _max_mtime(paths: list[Path]) -> float:
    return max((p.stat().st_mtime for p in paths), default=0.0)


def build_rust(force: bool) -> None:
    src = _files_under(RUST_DIR / "src", RUST_DIR / "mock")
    src.append(RUST_DIR / "Cargo.toml")
    src.append(RUST_DIR / "mock" / "Cargo.toml")
    if not force and RUST_BIN.exists() and RUST_BIN.stat().st_mtime > _max_mtime(src):
        return
    print("building rust-bench...", file=sys.stderr)
    subprocess.run(
        ["cargo", "build", "--release", "--manifest-path", str(RUST_DIR / "Cargo.toml")],
        check=True,
    )


def build_cpp(force: bool) -> None:
    src = _files_under(CPP_DIR / "mock", CPP_DIR)
    src = [p for p in src if "build" not in p.parts]
    if not force and CPP_BIN.exists() and CPP_BIN.stat().st_mtime > _max_mtime(src):
        return
    print("building cpp-bench...", file=sys.stderr)
    CPP_BUILD.mkdir(parents=True, exist_ok=True)
    gen = "Ninja" if shutil.which("ninja") else "Unix Makefiles"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(CPP_DIR),
            "-B",
            str(CPP_BUILD),
            "-G",
            gen,
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(CPP_BUILD), "--config", "Release"], check=True)


# --- sweep ------------------------------------------------------------------


@dataclass
class Run:
    impl: str
    corpus: str
    bits: int
    payload: dict


def _run_one(binary: Path, impl: str, corpus: Path, bits: int, iters: int, warmup: int,
             decompress: bool, verify: bool) -> Run:
    cmd = [
        str(binary),
        str(corpus),
        "--bits", str(bits),
        "--iters", str(iters),
        "--warmup", str(warmup),
    ]
    if decompress:
        cmd.append("--decompress")
    if verify:
        cmd.append("--verify")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"{impl} bench failed on {corpus.name} bits={bits} rc={proc.returncode}")
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    return Run(impl=impl, corpus=corpus.name, bits=bits, payload=payload)


def _fmt_ns(ns_list: list[int], input_bytes: int) -> str:
    if not ns_list:
        return "-"
    med = statistics.median(ns_list)
    if med <= 0:
        return "-"
    mb_per_s = (input_bytes / (med / 1e9)) / (1024 * 1024)
    return f"{mb_per_s:>7.1f} MB/s"


def _ratio(input_bytes: int, compressed_bytes: int) -> str:
    if compressed_bytes <= 0:
        return "-"
    return f"{input_bytes / compressed_bytes:>5.2f}x"


def render_table(runs: list[Run]) -> str:
    header = "| impl | corpus | bits | rows | ratio | compress | decompress |"
    sep = "|------|--------|------|------|-------|----------|------------|"
    lines = [header, sep]
    for r in sorted(runs, key=lambda x: (x.corpus, x.bits, x.impl)):
        p = r.payload
        lines.append(
            f"| {r.impl} | {r.corpus} | {r.bits} | {p['num_rows']} | "
            f"{_ratio(p['input_bytes'], p['compressed_bytes'])} | "
            f"{_fmt_ns(p['compress_ns'], p['input_bytes'])} | "
            f"{_fmt_ns(p['decompress_ns'], p['input_bytes'])} |"
        )
    return "\n".join(lines)


# --- main -------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus", nargs="*", type=Path, help="extra corpus paths (.txt or .parquet)")
    ap.add_argument("--bits", type=int, nargs="+", default=list(range(9, 17)))
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--no-decompress", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--rust-only", action="store_true")
    ap.add_argument("--cpp-only", action="store_true")
    ap.add_argument("--force-build", action="store_true")
    args = ap.parse_args()

    corpora = discover_corpora(args.corpus)
    if not corpora:
        print(
            f"no corpora found. drop .txt or .parquet into {CORPORA}/ or pass paths on cli",
            file=sys.stderr,
        )
        return 1

    if not args.cpp_only:
        build_rust(args.force_build)
    if not args.rust_only:
        build_cpp(args.force_build)

    runs: list[Run] = []
    for corpus in corpora:
        for bits in args.bits:
            if not args.cpp_only:
                runs.append(_run_one(
                    RUST_BIN, "rust", corpus, bits, args.iters, args.warmup,
                    decompress=not args.no_decompress, verify=not args.no_verify,
                ))
            if not args.rust_only:
                runs.append(_run_one(
                    CPP_BIN, "cpp", corpus, bits, args.iters, args.warmup,
                    decompress=not args.no_decompress, verify=not args.no_verify,
                ))

    table = render_table(runs)
    print(table)

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS / f"{stamp}.json"
    out.write_text(json.dumps([r.__dict__ for r in runs], indent=2))
    print(f"\nresults: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
