#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
#
# Build four real text corpora (~11 MB each) with distinct Unicode character
# distributions, for the `multilingual` OnPair analysis example. Each corpus is
# kept large enough that the 16-bit dictionary fills, so the 12-vs-16-bit
# comparison is meaningful.
#
#   ASCII (English) : Project Gutenberg novels (canonical UTF-8 cache text)
#   Chinese         : Project Gutenberg works in Chinese
#   Japanese        : Aozora Bunko works (Shift-JIS -> UTF-8, ruby/markup stripped)
#   Emoji-heavy     : enryu43/twitter100m_tweets — real tweets containing >=2 emoji
#
# Requires: python3, and the `duckdb` Python package (auto-installed if missing).
#
# Usage:  examples/fetch_multilingual_corpora.sh [OUTDIR] [TARGET_BYTES]
#         (defaults: /tmp/corpora, 11000000)
set -euo pipefail
OUT="${1:-/tmp/corpora}"
TARGET="${2:-11000000}"
mkdir -p "$OUT"
export OUT TARGET

python3 - <<'PY'
import json, re, sys, os, io, csv, time, zipfile, urllib.request
OUT, TARGET = os.environ["OUT"], int(os.environ["TARGET"])
UA = {"User-Agent": "onpair-research-corpus"}
def get(u, t=60): return urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=t).read()
def strip_g(t):
    s = re.search(r'\*\*\* START OF TH[EI]S? PROJECT GUTENBERG EBOOK.*?\*\*\*', t, re.S)
    e = re.search(r'\*\*\* END OF TH[EI]S? PROJECT GUTENBERG EBOOK', t, re.S)
    return t[(s.end() if s else 0):(e.start() if e else len(t))].strip()

def gutenberg(lang, outpath):
    total = n = 0
    out = open(outpath, "w", encoding="utf-8")
    page = f"https://gutendex.com/books?languages={lang}"
    while page and total < TARGET:
        d = json.loads(get(page, 40))
        for b in d["results"]:
            if total >= TARGET: break
            # canonical UTF-8 cache URL is reliable; gutendex format URLs often 404
            cands = [f"https://www.gutenberg.org/cache/epub/{b['id']}/pg{b['id']}.txt"]
            cands += [v for k, v in b.get("formats", {}).items()
                      if k.startswith("text/plain") and not v.endswith(".zip")]
            raw = None
            for u in cands:
                try: raw = get(u).decode("utf-8", "replace"); break
                except Exception: continue
            if not raw: continue
            body = strip_g(raw)
            if len(body) < 2000: continue
            out.write(body + "\n"); total += len(body.encode()); n += 1
            time.sleep(0.25)
        page = d.get("next")
    out.close()
    print(f"[{lang}] {n} works, {total/1e6:.2f} MB -> {outpath}", file=sys.stderr)

def aozora(outpath):
    idx = get("https://www.aozora.gr.jp/index_pages/list_person_all_extended_utf8.zip")
    zf = zipfile.ZipFile(io.BytesIO(idx))
    name = next(n for n in zf.namelist() if n.endswith(".csv"))
    reader = csv.DictReader(io.StringIO(zf.read(name).decode("utf-8-sig")))
    urlcol = next(c for c in reader.fieldnames if "テキストファイルURL" in c)
    def clean(t):
        t = re.sub(r'《[^》]*》', '', t)          # ruby readings
        t = t.replace('｜', '')                  # ruby start marker
        t = re.sub(r'［＃[^］]*］', '', t)         # editorial annotations
        parts = re.split(r'-{20,}\r?\n', t)       # drop Aozora header note
        if len(parts) >= 3: t = parts[2]
        return re.split(r'底本：', t)[0].strip()  # drop bibliographic footer
    total = n = 0
    out = open(outpath, "w", encoding="utf-8")
    for r in reader:
        if total >= TARGET: break
        url = (r.get(urlcol) or "").strip()
        if "aozora.gr.jp" not in url: continue
        try:
            data = get(url)
            if url.endswith(".zip"):
                z = zipfile.ZipFile(io.BytesIO(data))
                tn = [f for f in z.namelist() if f.lower().endswith(".txt")]
                if not tn: continue
                txt = z.read(tn[0]).decode("shift_jis", "replace")
            else:
                txt = data.decode("shift_jis", "replace")
        except Exception:
            continue
        body = clean(txt)
        if len(body) < 1000: continue
        out.write(body + "\n"); total += len(body.encode()); n += 1
        time.sleep(0.2)
    out.close()
    print(f"[ja] {n} works, {total/1e6:.2f} MB -> {outpath}", file=sys.stderr)

def emoji(outpath):
    try:
        import duckdb
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "duckdb"])
        import duckdb
    base = "https://huggingface.co/datasets/enryu43/twitter100m_tweets/resolve/refs%2Fconvert%2Fparquet/default/train"
    con = duckdb.connect(); con.execute("INSTALL httpfs; LOAD httpfs;")
    emoji_re = r'[\x{1F000}-\x{1FAFF}\x{2600}-\x{27BF}\x{2B00}-\x{2BFF}\x{2190}-\x{21FF}]'
    url_re, ws = re.compile(r'https?://\S+'), re.compile(r'[ \t]+')
    def ne(s): return sum(1 for c in s if ord(c) > 0x1F000 or 0x2600 <= ord(c) <= 0x27BF or 0x2190 <= ord(c) <= 0x21FF)
    seen, total = set(), 0
    out = open(outpath, "w", encoding="utf-8")
    for shard in range(3):
        if total >= TARGET: break
        url = f"{base}/{shard:04d}.parquet"
        cur = con.execute(f"SELECT tweet FROM read_parquet('{url}') WHERE regexp_matches(tweet, '{emoji_re}') LIMIT 200000")
        while total < TARGET:
            batch = cur.fetchmany(5000)
            if not batch: break
            for (t,) in batch:
                t = ws.sub(' ', url_re.sub('', t).replace('\n', ' ')).strip()
                k = t[:60]
                if not t or k in seen or ne(t) < 2: continue
                seen.add(k); out.write(t + "\n"); total += len(t.encode())
    out.close()
    print(f"[emoji] {len(seen)} tweets, {total/1e6:.2f} MB -> {outpath}", file=sys.stderr)

print("[1/4] ASCII (English) — Project Gutenberg", file=sys.stderr); gutenberg("en", f"{OUT}/en.txt")
print("[2/4] Chinese — Project Gutenberg", file=sys.stderr);        gutenberg("zh", f"{OUT}/zh.txt")
print("[3/4] Japanese — Aozora Bunko", file=sys.stderr);            aozora(f"{OUT}/ja.txt")
print("[4/4] Emoji-heavy — twitter100m", file=sys.stderr);          emoji(f"{OUT}/emoji.txt")
PY

echo "Done. Corpora in $OUT:"
ls -la "$OUT"/{en,ja,zh,emoji}.txt
