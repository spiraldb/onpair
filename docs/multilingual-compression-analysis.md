<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# OnPair on multilingual text: compression ratio and symbol-table distribution

This note measures OnPair on **real** text corpora chosen for their distinct
Unicode byte-width distributions — ASCII, CJK (Japanese, Chinese), and
emoji-heavy social posts — at code widths **12** and **16** bits, and inspects
the resulting dictionary (the "symbol table").

## Corpora

All data is real (no synthetic strings). Each corpus is split into rows on
`\n`. Fetch with [`examples/fetch_multilingual_corpora.sh`](../examples/fetch_multilingual_corpora.sh).

| Corpus | Source | bytes/char | UTF-8 char width (1B / 2B / 3B / 4B) |
|---|---|---|---|
| ASCII (English) | Gutenberg — *Pride and Prejudice* | 1.01 | **99.4%** / 0 / 0.6% / 0 |
| Japanese | Aozora — 夏目漱石「こころ」 | 3.00 | 0 / 0 / **100%** / 0 |
| Chinese | Gutenberg — *紅樓夢* | 2.95 | 2.6% / 0 / **97.4%** / 0 |
| Emoji-heavy | live Mastodon public timelines | 1.13 | 93.5% / 1.3% / 3.7% / **1.5%** |

The emoji corpus is real social text that *uses* emoji (URLs stripped); emoji
density is ~1.6% of characters, but those are the 4-byte codepoints — 10k+ of
them across ~2,100 posts.

## Run

```bash
examples/fetch_multilingual_corpora.sh /tmp/corpora
cargo run --release --example multilingual -- /tmp/corpora \
    "ASCII (English)=en.txt" "Japanese=ja.txt" "Chinese=zh.txt" "Emoji-heavy=emoji.txt"
# size-normalised (removes the dictionary-fill confound):
ONPAIR_CAP_BYTES=512000 cargo run --release --example multilingual -- /tmp/corpora ...
```

## Compression ratio

Two ratios are reported. **codes-only** = `orig / (dict_bytes + dict_offsets +
codes)` is the cleanest cross-language number. **incl. offsets** additionally
charges the `u64`-per-row offset layer, which dominates for corpora with many
tiny rows (English/Chinese have 14k–28k rows) — it is interchange-form overhead,
not an OnPair property.

### Native sizes

| Corpus | size | 12-bit (codes-only / incl) | 16-bit (codes-only / incl) | mean bytes/code 12→16 |
|---|--:|---|---|---|
| ASCII (English) | 706 KiB | 1.91x / 1.46x | 1.93x / 1.48x | 4.22 → 4.63 |
| Japanese | 542 KiB | **2.07x** / 1.98x | 2.04x / 1.95x | 4.87 → 4.96 |
| Chinese | 2526 KiB | 1.93x / 1.65x | **2.12x** / 1.79x | 3.95 → 4.91 |
| Emoji-heavy | 733 KiB | 1.26x / 1.22x | 1.34x / 1.30x | 2.65 → 3.13 |

### Size-normalised (every corpus capped to 500 KiB)

| Corpus | 12-bit codes-only | 16-bit codes-only | 16-bit dict fill |
|---|--:|--:|--:|
| ASCII (English) | 1.80x | 1.82x | 8% |
| Japanese | 2.03x | 2.01x | 7% |
| Chinese | 1.59x | 1.57x | 8% |
| Emoji-heavy | 1.23x | 1.27x | 11% |

## Findings

1. **Ratio ranking is Japanese > English > Chinese > Emoji**, driven by *mean
   bytes per code* — how many source bytes each emitted code replaces. Literary
   Japanese/English are highly redundant (~4.2–4.9 B/code); real emoji-laden
   social text is high-entropy and low-redundancy (~2.7 B/code), so it
   compresses worst regardless of code width.

2. **12 vs 16 bit is a data-volume question, not a language one.** A 16-bit
   dictionary only helps once a corpus has enough repeated substrings to fill
   *past* 4,096 tokens. Only the 2.5 MB Chinese corpus does (16-bit dict 26%
   full, 16.7k tokens) — and it is the one corpus where 16-bit clearly wins
   (1.93x → 2.12x). At a fixed 500 KiB budget **every** corpus fills only 7–11%
   of the 16-bit space, so 16-bit barely moves the ratio, and for Japanese and
   Chinese it is *slightly worse* than 12-bit because the wider `dict_bytes` /
   `dict_offsets` are not repaid. Emoji is the exception that still gains a
   little at 16-bit (more distinct short tokens to capture).

3. **The symbol table looks very different per script.** OnPair builds tokens by
   merging adjacent bytes, so tokens need not respect UTF-8 character
   boundaries:

   | Corpus (16-bit) | tokens | UTF-8 char-aligned | dominant token shape |
   |---|--:|--:|---|
   | ASCII (English) | 6,367 | **98%** | long word/phrase tokens (mean 5.9 B, many 5+ chars) |
   | Japanese | 4,773 | **40%** | majority *straddle* the 3-byte kana/kanji boundary |
   | Chinese | 16,729 | 71% | many 2-character words (5,449 two-char tokens) |
   | Emoji-heavy | 9,651 | 88% | short 2–4 B tokens, flat length spread |

   - **English** learns whole words/phrases — almost every token is valid UTF-8
     and most span several characters.
   - **Japanese** is the striking case: ~55–60% of learned tokens cut *across*
     codepoint boundaries (2-byte and 4-byte tokens that split 3-byte kana),
     because frequent byte n-grams sit inside and between characters. It still
     achieves the best ratio — boundary alignment is irrelevant to OnPair, only
     byte-substring frequency matters.
   - **Chinese** is where the extra 16-bit headroom pays off: it fills the
     dictionary with two-character compound words, the natural unit of meaning.
   - **Emoji-heavy** has the flattest token-length distribution and the lowest
     mean token length, reflecting low cross-post redundancy.

## Caveat

The emoji corpus is sampled live, so its exact ratios vary per run (±a few %);
the qualitative results (worst ratio, small 16-bit gain, short flat tokens) are
stable. The `multilingual` example revalidates a full decode roundtrip for every
corpus and code width.
