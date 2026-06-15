# OnPair vs. block compressors on executable binaries

Reproduces the compression-quality comparison of **OnPair** against the
block-based compressors **Zstandard** (levels 3 / 19 / 22) and **Snappy**, using
large stripped/native executables (>5 MiB) as the corpus.

## Corpus

Five executables spanning four toolchains were copied from the host
(`*.bin`, git-ignored — recreate with the commands below):

| file               | source                            | toolchain | size (MiB) |
|--------------------|-----------------------------------|-----------|-----------:|
| `rg.bin`           | `/usr/bin/rg`                     | Rust      |       5.01 |
| `python3.12.bin`   | `/usr/bin/python3.12`             | C         |       7.65 |
| `cmake.bin`        | `/usr/bin/cmake`                  | C++       |      11.25 |
| `clang-tidy.bin`   | `/usr/lib/llvm-18/bin/clang-tidy` | C++/LLVM  |      26.08 |
| `golangci-lint.bin`| `/usr/local/bin/golangci-lint`    | Go        |      36.36 |
| `rg-arm64.bin`     | vendored ripgrep `arm64-linux/rg` | Rust/ARM64|       5.00 |

`rg-arm64.bin` is the aarch64 build of the same program as `rg.bin`; it is run
separately (`results_arm64.csv`) for the x86-64-vs-aarch64 comparison in §5 of
`RESULTS.md`.

```sh
cp /usr/bin/rg                      rg.bin
cp /usr/bin/python3.12             python3.12.bin
cp /usr/bin/cmake                  cmake.bin
cp /usr/lib/llvm-18/bin/clang-tidy clang-tidy.bin
cp /usr/local/bin/golangci-lint    golangci-lint.bin
```

## Run

```sh
cargo build --release --example bench_binaries
ONPAIR_BENCH_ITERS_C=3 ONPAIR_BENCH_ITERS_D=10 \
  ./target/release/examples/bench_binaries benchmarks/binaries_data/*.bin \
  > benchmarks/binaries_data/results.csv
```

Each compression run is timed 3× and each decompression run 10×; the harness
reports both the **minimum** and the **median** time per phase. All roundtrips
were verified byte-for-byte (`roundtrip=ok`).

## Codecs

* `onpair-12` / `onpair-16` — whole file as a single OnPair record (max-ratio mode).
* `onpair-16-4k` — file split into 4 KiB records; OnPair's real random-access mode.
* `zstd-3` / `zstd-19` / `zstd-22` — whole-file Zstandard.
* `snappy` — whole-file Snappy.

OnPair sizes use OnPair's native accounting: dictionary bytes + dictionary
offsets (u32) + codes (u16) + per-row code offsets. Codes are not bit-packed to
the configured width, so a packed implementation could shave a little off the
OnPair numbers; it would not change the conclusions.

See [`RESULTS.md`](RESULTS.md) for the analysis and `results.csv` for raw data.
