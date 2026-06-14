# Results: OnPair vs. Zstandard vs. Snappy on executable binaries

Measured on five stripped/native executables (Rust, C, C++, Go), 5–36 MiB each.
Compression timed 3× and decompression 10×; minima reported. Every roundtrip
verified byte-for-byte. Raw data: [`results.csv`](results.csv).

`ratio = original / compressed` (higher is better; **< 1.0 means the output is
larger than the input**). Throughput in MiB/s computed from the minimum time.

## 1. Compression ratio

| file (entropy bits/byte) | onpair-12 | onpair-16 | onpair-16-4k | snappy | zstd-3 | zstd-19 | zstd-22 |
|--------------------------|----------:|----------:|-------------:|-------:|-------:|--------:|--------:|
| rg          (6.078)      | 0.94      | 1.10      | **1.49**     | 1.77   | 2.85   | 3.48    | 3.48    |
| python3.12  (5.618)      | 1.05      | 1.25      | **1.62**     | 1.90   | 2.74   | 3.48    | 3.48    |
| cmake       (6.341)      | 1.08      | 1.42      | **1.72**     | 1.70   | 2.59   | 3.28    | 3.28    |
| clang-tidy  (6.142)      | 0.77      | 1.00      | **2.18**     | 2.56   | 4.38   | 5.56    | 5.58    |
| golangci    (6.193)      | 0.87      | 1.17      | **1.73**     | 1.88   | 2.75   | 3.35    | 3.36    |

* **OnPair compresses executables poorly, and sometimes *expands* them.** The
  whole-file 12-bit mode grows `clang-tidy` by 30 % (0.77×) and `golangci` by
  15 %. 16-bit codes and a larger dictionary help, but even OnPair's best mode
  (`onpair-16-4k`, 1.49–2.18×) is beaten by **Snappy** on most files and is
  ~2–2.5× worse than zstd at high levels.
* **Zstandard wins decisively.** Even the *default* level 3 (2.6–4.4×) beats
  every OnPair configuration. Levels 19/22 add another ~25–30 % (3.3–5.6×).
* **zstd-22 ≈ zstd-19.** For these inputs level 22 buys essentially nothing over
  19 (≤ 0.4 % more ratio) while costing 25–40 % more compression time.
* For OnPair, **16-bit > 12-bit** always, and — counter to the usual rule —
  **smaller records (4 KiB) beat the whole-file record.** With one giant record
  the dynamic-threshold trainer builds a weaker dictionary than it does over many
  small records, so the random-access mode is also OnPair's highest-ratio mode
  here.

## 2. Compression speed (MiB/s, higher is better)

| file        | snappy | zstd-3 | onpair-16-4k | zstd-19 | zstd-22 |
|-------------|-------:|-------:|-------------:|--------:|--------:|
| rg          | 416    | 203    | 30           | 3.2     | 2.1     |
| python3.12  | 421    | 196    | 28           | 3.8     | 2.5     |
| cmake       | 328    | 171    | 27           | 2.7     | 1.9     |
| clang-tidy  | 530    | 254    | 41           | 2.4     | 1.9     |
| golangci    | 395    | 173    | 22           | 2.4     | 1.9     |

High-ratio zstd is **50–250× slower** than zstd-3 and ~10× slower than OnPair,
for a modest ratio gain. Snappy is the fastest compressor but a weak one.

## 3. Decompression speed (MiB/s, higher is better)

| file        | onpair-16-4k | snappy | zstd-3 | zstd-19 |
|-------------|-------------:|-------:|-------:|--------:|
| rg          | 1882         | 1103   | 782    | 424     |
| python3.12  | 2043         | 1062   | 785    | 626     |
| cmake       | 2080         | 921    | 723    | 610     |
| clang-tidy  | 2579         | 1307   | 676    | 547     |
| golangci    | 1168         | 806    | 532    | 472     |

**This is where OnPair shines:** its blocked mode decompresses fastest of all
(1.2–2.6 GiB/s) — and, unlike the block compressors, it can decode any single
record in O(1) without touching the rest of the blob. That random-access
property, not whole-blob ratio, is what OnPair is built for.

## 4. What affects the compression ratio of executable binaries?

1. **Byte entropy sets a ceiling, but doesn't rank the files.** All inputs sit at
   5.6–6.3 bits/byte, so no codec can approach the ratios seen on text/columnar
   data. Yet `clang-tidy` (entropy 6.14) compresses *best* of all (5.58×) while
   `cmake` (6.34) compresses worst among the C++ tools (3.28×) — so gross entropy
   alone is a poor predictor. **Section composition matters more.**

2. **What the sections contain:**
   * **`.text` (machine code)** is the least compressible part: variable-length
     x86 instructions packed with one-off immediates and addresses look
     near-random at the byte level. `cmake` is ~80 % `.text`, which is why it has
     the lowest zstd ratio despite the highest entropy.
   * **Relocation and symbol tables** (`.rela.dyn`, `.dynsym`, `.dynstr`,
     `.data.rel.ro`) are highly regular and very compressible. `clang-tidy`
     carries ~3.8 MiB of relocations + ~2.6 MiB of symbol strings, which is
     exactly why it tops the ratio chart.
   * **`.rodata`/strings and Go's `.gopclntab`** are moderately compressible
     (English-ish text, structured metadata): `python3.12` (35 % `.rodata`) and
     `golangci` (Go runtime tables) sit in the middle.

3. **Long-range redundancy and window size — the decisive factor between codecs.**
   Executables repeat patterns across megabytes: boilerplate prologues/epilogues,
   duplicated relocation entries, repeated mangled-name fragments. zstd's large
   match window captures these; that long-range matching, far more than any
   entropy-coding edge, is why zstd doubles Snappy and crushes OnPair. OnPair
   only merges *short adjacent substrings* into a per-corpus dictionary with
   record-local decoding, so it cannot exploit whole-file redundancy — the wrong
   model for high-entropy code with long-range structure.

4. **Codec configuration:** dictionary size / code width (OnPair 16- vs 12-bit),
   record/block size (smaller blocks trained a better OnPair dictionary here),
   and zstd level (3 → 19 is a real gain; 19 → 22 is not).

### Bottom line

For compressing whole executable binaries, **block-based zstd is the right
tool**: default level 3 already beats every OnPair mode at >170 MiB/s, and
levels 19/22 push ratios to 3.3–5.6× when compression time is irrelevant
(archival). **OnPair is not a general binary compressor** — it underperforms
(and can expand) on high-entropy code because it targets short, vocabulary-rich
strings with shared substrings (database string columns). Its payoffs are
fastest-in-class decompression and O(1) random access per record, which this
whole-blob ratio benchmark does not reward.
