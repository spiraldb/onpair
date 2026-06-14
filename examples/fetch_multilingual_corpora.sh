#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
#
# Fetch real, public-domain / live text corpora with distinct Unicode character
# distributions, for the `multilingual` OnPair analysis example.
#
#   ASCII (English) : Project Gutenberg  — Pride and Prejudice (#1342)
#   Chinese         : Project Gutenberg  — 紅樓夢 / Dream of the Red Chamber (#24264)
#   Japanese        : Aozora Bunko       — 夏目漱石「こころ」(card 148), Shift-JIS → UTF-8
#   Emoji-heavy     : live public Mastodon timelines — real posts containing emoji
#
# The first three are deterministic. The emoji corpus is sampled live from
# federated Mastodon public timelines, so its exact contents vary per run.
#
# Usage:  examples/fetch_multilingual_corpora.sh [OUTDIR]   (default: /tmp/corpora)
set -euo pipefail
OUT="${1:-/tmp/corpora}"
mkdir -p "$OUT"
UA="onpair-research-corpus"

strip_gutenberg() {  # stdin -> stdout: drop PG header/footer boilerplate
  python3 - "$@" <<'PY'
import sys, re
t = sys.stdin.buffer.read().decode("utf-8", "replace")
s = re.search(r'\*\*\* START OF TH[EI]S? PROJECT GUTENBERG EBOOK.*?\*\*\*', t, re.S)
e = re.search(r'\*\*\* END OF TH[EI]S? PROJECT GUTENBERG EBOOK', t, re.S)
sys.stdout.write(t[(s.end() if s else 0):(e.start() if e else len(t))].strip() + "\n")
PY
}

echo "[1/4] ASCII (English) — Pride and Prejudice"
curl -sSL "https://www.gutenberg.org/cache/epub/1342/pg1342.txt" | strip_gutenberg > "$OUT/en.txt"

echo "[2/4] Chinese — Dream of the Red Chamber"
curl -sSL "https://www.gutenberg.org/cache/epub/24264/pg24264.txt" | strip_gutenberg > "$OUT/zh.txt"

echo "[3/4] Japanese — Kokoro (Aozora, Shift-JIS → UTF-8)"
tmp="$(mktemp -d)"
curl -sSL "https://www.aozora.gr.jp/cards/000148/files/773_ruby_5968.zip" -o "$tmp/k.zip"
unzip -o -q "$tmp/k.zip" -d "$tmp"
iconv -f SHIFT_JIS -t UTF-8 "$(ls "$tmp"/*.txt | head -1)" | python3 - <<'PY' > "$OUT/ja.txt"
import sys, re
t = sys.stdin.read()
parts = re.split(r'-{20,}\r?\n', t)          # drop the Aozora header note (between two rules)
body = parts[2] if len(parts) >= 3 else t
sys.stdout.write(re.split(r'底本：', body)[0].strip() + "\n")  # drop the bibliographic footer
PY
rm -rf "$tmp"

echo "[4/4] Emoji-heavy — live Mastodon public timelines (~750 KiB of emoji-bearing posts)"
EMOJI_OUT="$OUT/emoji.txt" python3 - <<'PY'
import json, re, html, os, urllib.request, time, sys
UA = {"User-Agent": "onpair-research-corpus"}
def get(u):
    with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=25) as r:
        return json.load(r)
def clean(h):
    h = re.sub(r'<br\s*/?>', '\n', h); h = re.sub(r'</p>', '\n', h)
    t = html.unescape(re.sub(r'<[^>]+>', ' ', h))
    return re.sub(r'[ \t]+', ' ', re.sub(r'https?://\S+', '', t)).strip()
def emoji(s):  # rough emoji / pictograph count
    return sum(1 for c in s if ord(c) > 0x1F000 or 0x2600 <= ord(c) <= 0x27BF or 0x2190 <= ord(c) <= 0x21FF)
instances = ["mastodon.world","mstdn.social","mas.to","fosstodon.org","mastodon.online",
             "techhub.social","mastodon.gamedev.place","infosec.exchange","hachyderm.io",
             "mstdn.jp","pawoo.net","social.vivaldi.net","universeodon.com","sfba.social"]
seen, kept, total, TARGET = set(), [], 0, 750_000
for inst in instances:
    for mode in ("&local=true", ""):
        max_id, pages = None, 0
        while total < TARGET and pages < 40:
            url = f"https://{inst}/api/v1/timelines/public?limit=40{mode}" + (f"&max_id={max_id}" if max_id else "")
            try: posts = get(url)
            except Exception: break
            if not isinstance(posts, list) or not posts: break
            pages += 1; max_id = posts[-1]["id"]
            for p in posts:
                txt = clean(p.get("content", "")); key = txt[:60]
                if key in seen: continue
                seen.add(key)
                if emoji(txt) >= 2 and len(txt) >= 10:
                    kept.append(txt.replace("\n", " ")); total += len(txt.encode())
            time.sleep(0.1)
        if total >= TARGET: break
    if total >= TARGET: break
open(os.environ["EMOJI_OUT"], "w", encoding="utf-8").write("\n".join(kept) + "\n")
print(f"  emoji corpus: {len(kept)} posts, {total} bytes", file=sys.stderr)
PY

echo "Done. Corpora in $OUT:"
ls -la "$OUT"/{en,ja,zh,emoji}.txt
