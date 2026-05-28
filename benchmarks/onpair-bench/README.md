# onpair-bench

Cross-impl benchmark harness for OnPair. Builds the Rust and C++ benchmark
binaries, runs them across a corpus sweep × `bits ∈ 9..=16`, and prints a
markdown summary plus per-iteration raw timings under `results/`.

## Layout

```
onpair-bench/
├── run.py                  # orchestrator
├── pyproject.toml
├── corpora/                # gitignored; drop .txt or .parquet here
│   └── .cache/             # parquet → .txt extracts (one per Utf8/Utf8View col)
├── results/                # gitignored; <UTC>.json per run
├── rust-bench/
│   ├── Cargo.toml
│   ├── src/main.rs
│   └── mock/               # passthrough mock; swap to registry dep when real
└── cpp-bench/
    ├── CMakeLists.txt
    ├── main.cpp
    └── mock/               # passthrough mock; swap add_subdirectory to use real
```

## Input format

LF-delimited bytes. Each binary scans for `\n`, builds `payload: Vec<u8>` and
`offsets: Vec<u32>` in memory, and hands them to the trainer. A trailing `\n`
terminates the last row rather than starting an empty one. Rows containing an
embedded `\n` aren't representable — the parquet extractor warns and drops
them.

## Usage

```bash
# drop a corpus in:
cp /some/strings.txt corpora/
# or a parquet (each Utf8/Utf8View column becomes one .txt under .cache/)
cp /some/strings.parquet corpora/

python run.py
python run.py --bits 12 14 16 --iters 10
python run.py --rust-only --no-decompress
python run.py extra1.txt extra2.parquet
```

## Mock vs real impls

Both `rust-bench/mock/` and `cpp-bench/mock/` are no-op passthroughs that
mirror the expected API surface, so the harness builds and `--verify` passes
end-to-end without any upstream code.

To wire up the real impls:

- **Rust**: in `rust-bench/Cargo.toml`, change
  `vortex-onpair-rs = { path = "mock" }` to a registry dep
  (`vortex-onpair-rs = "X.Y"`). The expected API is the `Column` type in
  `mock/src/lib.rs`.
- **C++**: in `cpp-bench/CMakeLists.txt`, replace `add_subdirectory(mock)`
  with `add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../onpair-sys/cmake onpair-sys)`.
  Upstream is expected to expose a CMake target named `onpair` and the header
  `<onpair/api.h>` matching `cpp-bench/mock/include/onpair/api.h`.
