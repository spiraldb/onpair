<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# OnPair on multilingual text: compression ratio and symbol-table distribution

This note measures OnPair on **real** text corpora chosen for their distinct
Unicode byte-width distributions — ASCII, CJK (Japanese, Chinese), and
emoji-heavy social posts — at code widths **12** and **16** bits, and inspects
the resulting dictionary (the "symbol table").

Each corpus is **≥10 MB** so that the 16-bit dictionary (up to 65,536 tokens)
can actually fill; a smaller corpus starves the dictionary and makes the
12-vs-16-bit comparison meaningless (see *Why ≥10 MB* below).

## Corpora

All data is real (no synthetic strings). Each corpus is split into rows on
`\n`. Fetch with [`examples/fetch_multilingual_corpora.sh`](../examples/fetch_multilingual_corpora.sh).

| Corpus | Source | size | rows | bytes/char | UTF-8 char width 1B/2B/3B/4B |
|---|---|--:|--:|--:|---|
| ASCII (English) | Project Gutenberg — 11 novels | 11.2 MB | 220k | 1.02 | **99.1** / 0 / 0.8 / 0 |
| Japanese | Aozora Bunko — 377 works | 11.0 MB | 39k | 2.97 | 1.6 / 0 / **98.4** / 0 |
| Chinese | Project Gutenberg — 9 works | 11.0 MB | 96k | 2.91 | 4.3 / 0 / **95.7** / 0 |
| Emoji-heavy | `enryu43/twitter100m_tweets`, real tweets with ≥2 emoji | 12.8 MB | 71k | 1.61 | 70.4 / 0.7 / 26.1 / **2.9** |

The emoji corpus is real tweets (URLs stripped) filtered to those containing ≥2
emoji; 2.9% of its characters are 4-byte emoji and 26% are 3-byte (CJK from
international tweets) — the richest Unicode mix of the four.

## Run

```bash
examples/fetch_multilingual_corpora.sh /tmp/corpora     # builds four ≥10 MB corpora
cargo run --release --example multilingual -- /tmp/corpora \
    "ASCII (English)=en.txt" "Japanese=ja.txt" "Chinese=zh.txt" "Emoji-heavy=emoji.txt"
```

## Compression ratio

Two ratios are reported. **codes-only** = `orig / (dict_bytes + dict_offsets +
codes)` is the cleanest cross-language number. **incl. offsets** additionally
charges the `u64`-per-row offset layer, which dominates for corpora with many
tiny rows (English has 220k rows, Chinese 96k) — that is interchange-form
overhead, not an OnPair property.

| Corpus | 12-bit (codes-only / incl) | 16-bit (codes-only / incl) | mean bytes/code 12→16 | 16-bit dict fill |
|---|---|---|--:|--:|
| ASCII (English) | 1.79x / 1.40x | **2.47x** / 1.78x | 3.61 → 5.72 | 79% |
| Japanese | 2.04x / 1.93x | **2.73x** / 2.54x | 4.11 → 6.41 | 76% |
| Chinese | 1.59x / 1.43x | **2.07x** / 1.81x | 3.19 → 4.69 | 91% |
| Emoji-heavy | 1.38x / 1.30x | **2.06x** / 1.89x | 2.76 → 4.64 | 100% |

## Symbol table (dictionary) at 16-bit

| Corpus | tokens N | dict fill | UTF-8 char-aligned | mean token len | dominant token shape |
|---|--:|--:|--:|--:|---|
| ASCII (English) | 52,014 | 79% | **100%** | 7.81 B | long words/phrases (43.6k tokens span 5+ chars) |
| Japanese | 49,799 | 76% | 70% | 7.97 B | long kana/kanji runs; 15k tokens **straddle** char boundaries |
| Chinese | 59,339 | 91% | 81% | 6.71 B | two-character compound words (24.5k two-char tokens) |
| Emoji-heavy | 65,431 | **100%** | 84% | 6.62 B | broad length spread; only corpus to fill the dictionary |

## Findings

1. **At ≥10 MB per window, 16-bit beats 12-bit for every language** — by +30%
   to +50% on the codes-only ratio. With enough data the larger dictionary
   fills (76–100%) and the mean bytes replaced per emitted code jumps (e.g.
   English 3.6→5.7, Japanese 4.1→6.4). This is the opposite of what undersized
   corpora suggest, where 16-bit looks useless — see below.

2. **Ratio ranking at 16-bit: Japanese > English > Chinese ≈ Emoji.** Japanese
   compresses best (2.73x): long, highly-repeated kana/kanji runs. The
   **emoji** corpus is the most interesting — *worst* at 12-bit (1.38x) but it
   catches Chinese at 16-bit (2.06x) because it is the only corpus to fill the
   *entire* 65,536-token dictionary: real social text has an enormous tail of
   distinct short repeated sequences (hashtags, mentions, emoji clusters,
   multilingual fragments) that a bigger dictionary can finally capture.

3. **The symbol table differs sharply by script.** OnPair merges adjacent
   *bytes*, so tokens need not respect UTF-8 character boundaries:
   - **English** — every token is valid UTF-8 (100% aligned) and most span
     several characters: it learns whole words and phrases.
   - **Japanese** — only ~70% of tokens are char-aligned; ~30% (15k tokens)
     cut *across* the 3-byte kana/kanji boundary, because frequent byte n-grams
     sit inside and between characters. Alignment is irrelevant to OnPair —
     only byte-substring frequency matters — and Japanese still wins on ratio.
   - **Chinese** — fills the dictionary with two-character compound words, the
     natural unit of meaning.
   - **Emoji-heavy** — flattest length distribution and full dictionary.

## Why ≥10 MB

Earlier runs on 0.5–2.5 MB corpora filled only 7–26% of the 16-bit dictionary,
so 16-bit barely moved the ratio and was sometimes *worse* than 12-bit (the
wider `dict_bytes`/`dict_offsets` were not repaid). That was dictionary
**starvation**, not a property of the languages. A fair 12-vs-16-bit comparison
needs a window large enough to populate the larger code space — hence ≥10 MB
here, at which point all four corpora fill ≥76% of the 16-bit dictionary and
16-bit wins across the board.

## Caveat

The emoji corpus is sampled from a fixed tweet dump, so it is reproducible up to
the dataset's row order. The `multilingual` example revalidates a full decode
roundtrip for every corpus and code width.
