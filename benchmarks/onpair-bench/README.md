# onpair-bench

Cross-impl benchmark harness for OnPair. Builds the Rust and C++ benchmark
binaries, runs them across a corpus sweep × `bits ∈ 9..=16`, and prints a
markdown summary plus per-iteration raw timings under `results/`.

Initialise the C++ submodule before building:

```bash
git submodule update --init --recursive
```

## Input format

LF-delimited bytes. Each binary scans for `\n`, builds `payload: Vec<u8>` and
`offsets: Vec<u32>` in memory, and hands them to the trainer. A trailing `\n`
terminates the last row rather than starting an empty one. Rows containing an
embedded `\n` aren't representable — the parquet extractor warns and drops
them.

## Usage

The bench is a uv workspace member of the repo-root `pyproject.toml`. Sync
once from the repo root, then drive it with `uv run`:

```bash
# from /Users/joeisaacs/git/spiraldb/onpair (one-time):
uv sync

# drop a corpus in:
cp /some/strings.txt benchmarks/onpair-bench/corpora/
# or a parquet (each Utf8/Utf8View column becomes one .txt under .cache/)
cp /some/strings.parquet benchmarks/onpair-bench/corpora/

uv run onpair-bench
uv run onpair-bench --bits 12 14 16 --iters 10
uv run onpair-bench --rust-only --no-decompress
uv run onpair-bench extra1.txt extra2.parquet
```

## Managed datasets (TPC-H, ClickBench, OnPair paper)

`corpus.py` fetches reproducible reference datasets into
`corpora/datasets/<name>/`. Each dataset lands once — a `.done` marker in
its directory short-circuits re-downloads. Run benchmarks against them via
`run.py --dataset NAME` (repeatable) or `run.py --all-datasets`; the
existing parquet → per-column extractor fans every string column out into
its own row in the result table, so rs vs cpp can be compared column by
column.

```bash
# inspect registry + completion state
uv run onpair-bench-corpus list

# fetch one or many
uv run onpair-bench-corpus fetch tpch-sf1 clickbench amazon-books-titles

# fetch every registered dataset
uv run onpair-bench-corpus fetch-all

# remove a specific dataset / wipe everything
uv run onpair-bench-corpus clean tpch-sf1
uv run onpair-bench-corpus clean-all

# print absolute path (handy for scripting)
uv run onpair-bench-corpus path tpch-sf1

# benchmark against a managed dataset (auto-fetches if missing)
uv run onpair-bench --dataset tpch-sf1 --dataset clickbench --bits 12 14 16
uv run onpair-bench --all-datasets --rust-only
```

| name                  | source                                                            | needs           |
|-----------------------|-------------------------------------------------------------------|-----------------|
| `tpch-sf0.1`          | generated via duckdb tpch extension                               | `duckdb`        |
| `tpch-sf1`            | generated via duckdb tpch extension                               | `duckdb`        |
| `clickbench`          | `datasets.clickhouse.com/hits_compatible/athena/hits.parquet`     | network         |
| `amazon-books-titles` | `McAuley-Lab/Amazon-Reviews-2023 :: raw_meta_Books.title`         | `datasets` (HF) |
| `amazon-books-reviews`| `McAuley-Lab/Amazon-Reviews-2023 :: raw_review_Books.text` (≤500MB) | `datasets` (HF) |
| `news-headlines`      | `rajistics/million-headlines :: headline_text`                    | `datasets` (HF) |
| `sentiment140-tweets` | `stanfordnlp/sentiment140 :: text`                                | `datasets` (HF) |

Install the optional fetcher deps with one of (run from the repo root so
the extras land in the shared workspace venv):

```bash
uv sync --extra tpch    # duckdb only
uv sync --extra paper   # HuggingFace datasets only
uv sync --extra full    # both
```

## Compressed-domain search comparison

`benches/search.rs` (Rust, divan) and `cpp-bench/search_bench.cpp` (C++)
benchmark the same `Contains` / `Prefix` searches over the same corpus and
needles. The Rust bench's pre-pass buckets needles by selectivity (rare /
medium / common) and, when `ONPAIR_SEARCH_DUMP=<dir>` is set, dumps
`corpus.bin` + `needles.bin` so the C++ harness searches byte-identical
inputs. Both count matches via a callback and cross-check against brute force.

```bash
# Rust side (+ dump shared inputs). Defaults to a synthetic URL corpus;
# point ONPAIR_BENCH_PARQUET at a parquet file for real data.
mkdir -p /tmp/onpair_dump
ONPAIR_SEARCH_DUMP=/tmp/onpair_dump cargo bench --bench search

# C++ side, on the dumped inputs (needs the submodule + Boost.Unordered ≥ 1.81):
cmake -S benchmarks/onpair-bench/cpp-bench -B benchmarks/onpair-bench/cpp-bench/build \
      -DCMAKE_BUILD_TYPE=Release
cmake --build benchmarks/onpair-bench/cpp-bench/build --target search_bench -j
benchmarks/onpair-bench/cpp-bench/build/search_bench /tmp/onpair_dump --bits 16
```

## Implementations

- **Rust**: `rust-bench` is a separate workspace whose `Cargo.toml` carries a
  path dep on the workspace-root `onpair` crate (`../..`). The bench shells
  out to `onpair::compress` / decodes rows directly off `Column::as_parts()`.
- **C++**: `cpp-bench` links the upstream
  [`gargiulofrancesco/onpair_cpp`](https://github.com/gargiulofrancesco/onpair_cpp)
  vendored under `cpp-bench/onpair_cpp/` as a git submodule. It exports the
  CMake target `onpair` and the `<onpair/api.h>` umbrella header.
