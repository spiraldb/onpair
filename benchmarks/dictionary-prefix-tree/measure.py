#!/usr/bin/env python3
"""Run frozen public benchmark binaries against the recorded input files."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", action="append", required=True, metavar="NAME=BIN_DIR")
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--filter", default="")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    inputs = ["fineweb-32mib", "paper-book-reviews-32mib", "clickbench-32mib"]
    files = {name: (args.data_dir / f"{name}-8mib.parquet").resolve() for name in inputs}
    for path in files.values():
        if not path.is_file():
            parser.error(f"missing corpus: {path}")
    env = dict(os.environ, ONPAIR_BENCH_MAX_BYTES="8388608", ONPAIR_BENCH_SCALE_FACTOR="1")
    env.pop("ONPAIR_BENCH_PARQUET", None)
    env.pop("ONPAIR_BENCH_COLUMN", None)
    flags = ["--bench", "--sample-count", "25", "--min-time", "0.4", "--max-time", "3", "--color", "never"]
    if args.filter:
        flags.append(args.filter)
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files.values()}
    for stage in args.stage:
        name, directory = stage.split("=", 1)
        binaries = {suite: (Path(directory) / suite).resolve() for suite in ["tpch", "clickbench"]}
        output = args.output_dir / name
        output.mkdir(parents=True, exist_ok=True)
        for suite, binary in binaries.items():
            hashes[str(binary)] = hashlib.sha256(binary.read_bytes()).hexdigest()
            cases = ["tpch"] if suite == "tpch" else ["synthetic-urls", *inputs]
            for case in cases:
                case_env = dict(env)
                if case in files:
                    case_env.update(ONPAIR_BENCH_PARQUET=str(files[case]), ONPAIR_BENCH_COLUMN="text")
                log = output / f"{case}.txt"
                with log.open("w") as stream:
                    subprocess.run(["taskset", "-c", str(args.cpu), str(binary), *flags], cwd=repo, env=case_env, stdout=stream, stderr=subprocess.STDOUT, check=True)
                text = log.read_text()
                if case == "synthetic-urls" and "synthetic ClickBench-shaped URL corpus" not in text:
                    raise RuntimeError(f"unexpected synthetic corpus; inspect {log}")
                if case in files and str(files[case]) not in text:
                    raise RuntimeError(f"unexpected Parquet corpus; inspect {log}")
                print(name, case, "complete", flush=True)
    (args.output_dir / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")


if __name__ == "__main__":
    main()
