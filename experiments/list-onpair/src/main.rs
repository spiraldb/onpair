// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Experiment: apply the OnPair technique to dictionary-encoded *list* columns
// (lists of ints) instead of strings, on four real datasets. See README.md.
//
//   raw            original text (elements joined, one row per line)
//   dict+listview  step 1 given by the task: dict-encode elements, store the
//                  flat int values bit-packed + row offsets  (the ListView)
//   onpair-int     step 2: run integer-OnPair over that int stream
//   zstd(raw)      general compressor on raw text, for reference
//   zstd(ints)     general compressor on the fixed-width int stream
//
// All structural sizes are the minimal bit-packed representation so the
// comparison is apples-to-apples. The shared element->string dictionary is
// reported separately (every method needs it identically).

mod intonpair;

use std::path::PathBuf;

use hashbrown::HashMap;

/// Dictionary capacity (base + merge tokens). Defaults to 4x the alphabet
/// rounded up, capped at 2^20. Override with DICT_CAP=<n> — needed for tiny
/// alphabets (e.g. boolean masks) where 4x the alphabet is far too few tokens
/// for OnPair to capture multi-element patterns.
fn capacity_for(num_distinct: usize) -> usize {
    if let Ok(v) = std::env::var("DICT_CAP") {
        if let Ok(n) = v.parse::<usize>() {
            return n.max(num_distinct);
        }
    }
    (num_distinct.next_power_of_two() * 4).min(1 << 20)
}

fn bits_for(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        (usize::BITS - (n - 1).leading_zeros()) as usize
    }
}

fn packed_bytes(count: usize, bits: usize) -> usize {
    (count * bits).div_ceil(8)
}

fn zstd_len(data: &[u8]) -> usize {
    zstd::encode_all(data, 19).expect("zstd").len()
}

fn zstd_at(data: &[u8], level: i32) -> usize {
    zstd::encode_all(data, level).expect("zstd").len()
}

struct Dataset {
    name: String,
    rows: Vec<Vec<u32>>,        // each row: element ids
    strings: Vec<String>,       // element id -> original string
    flat: Vec<u32>,
    offsets: Vec<u32>,
    raw_bytes: usize,           // original textual size (elements + separators)
}

fn load(path: &PathBuf) -> Dataset {
    let text = std::fs::read_to_string(path).expect("read dataset");
    let mut dict: HashMap<String, u32> = HashMap::new();
    let mut strings: Vec<String> = Vec::new();
    let mut rows: Vec<Vec<u32>> = Vec::new();
    let mut raw_bytes = 0usize;
    for line in text.lines() {
        if line.is_empty() {
            continue;
        }
        let mut row = Vec::new();
        let mut first = true;
        for elem in line.split('\t') {
            if !first {
                raw_bytes += 1; // separator
            }
            first = false;
            raw_bytes += elem.len();
            let id = *dict.entry(elem.to_string()).or_insert_with(|| {
                strings.push(elem.to_string());
                (strings.len() - 1) as u32
            });
            row.push(id);
        }
        rows.push(row);
    }
    let mut flat = Vec::new();
    let mut offsets = vec![0u32];
    for r in &rows {
        flat.extend_from_slice(r);
        offsets.push(flat.len() as u32);
    }
    let name = path.file_stem().unwrap().to_string_lossy().to_string();
    Dataset {
        name,
        rows,
        strings,
        flat,
        offsets,
        raw_bytes,
    }
}

fn run(ds: &Dataset) {
    let num_rows = ds.rows.len();
    let num_elems = ds.flat.len();
    let num_distinct = ds.strings.len();
    let base_bits = bits_for(num_distinct);

    // Shared element -> string dictionary (identical for every method).
    let string_payload: usize = ds.strings.iter().map(|s| s.len()).sum();
    let string_off = packed_bytes(num_distinct + 1, bits_for(string_payload + 1));
    let string_dict = string_payload + string_off;

    // ── Baseline: dict-encode + ListView (the task's step 1, no OnPair) ──
    let lv_values = packed_bytes(num_elems, base_bits);
    let lv_offsets = packed_bytes(num_rows + 1, bits_for(num_elems + 1));
    let listview = lv_values + lv_offsets;

    // ── Integer-OnPair (step 2) ──
    // Capacity: base alphabet + headroom for merges, capped to keep codes
    // narrow. 4x the alphabet (one extra bit) is plenty on these corpora.
    let capacity = capacity_for(num_distinct);
    let parser = intonpair::train(&ds.flat, &ds.offsets, num_distinct as u32, capacity, 0.5);
    let (codes, code_offsets) = parser.encode(&ds.flat, &ds.offsets);

    // verify round-trip
    verify(ds, &parser, &codes, &code_offsets);

    // Prune to tokens the encoder actually emitted, renumber densely, and let
    // the code width shrink to fit: a merge that never gets used should cost
    // neither a dictionary entry nor a wider code.
    let mut used: Vec<bool> = vec![false; parser.dict.num_tokens()];
    for &c in &codes {
        used[c as usize] = true;
    }
    let mut remap: Vec<u32> = vec![0; used.len()];
    let mut dict_elems = 0usize;
    let mut num_tokens = 0usize;
    let mut num_merged = 0usize;
    for (id, &u) in used.iter().enumerate() {
        if u {
            remap[id] = num_tokens as u32;
            dict_elems += parser.dict.token(id as u32).len();
            num_tokens += 1;
            if id >= num_distinct {
                num_merged += 1;
            }
        }
    }
    if std::env::var("DUMP_DICT").is_ok() {
        dump_dict(ds, &parser, &codes, num_distinct);
    }

    let codes: Vec<u32> = codes.iter().map(|&c| remap[c as usize]).collect();
    let code_bits = bits_for(num_tokens);
    let off_bits = bits_for(dict_elems + 1);

    let op_codes = packed_bytes(codes.len(), code_bits);
    let op_tokdict = packed_bytes(dict_elems, base_bits) + packed_bytes(num_tokens + 1, off_bits);
    let op_rowoff = packed_bytes(num_rows + 1, bits_for(codes.len() + 1));
    let onpair = op_codes + op_tokdict + op_rowoff;

    // ── Whole-stack dictionary (row-level dedup) ──
    // Many rows are byte-identical stack traces, so give each *distinct whole
    // trace* one id: the column becomes one code per row plus a stack table
    // stored as a ListView of frame ids. This is the "stack table + sample ->
    // stack id" layout used by real profile stores.
    let mut stack_ids: HashMap<&[u32], u32> = HashMap::new();
    let mut uniq_stacks_elems = 0usize;
    for r in &ds.rows {
        if stack_ids.insert(r.as_slice(), 0).is_none() {
            uniq_stacks_elems += r.len();
        }
    }
    let num_uniq_stacks = stack_ids.len();
    let ws_codes = packed_bytes(num_rows, bits_for(num_uniq_stacks));
    let ws_table_vals = packed_bytes(uniq_stacks_elems, base_bits);
    let ws_table_off = packed_bytes(num_uniq_stacks + 1, bits_for(uniq_stacks_elems + 1));
    let wholestack = ws_codes + ws_table_vals + ws_table_off;

    // Combination: dedup whole stacks, then OnPair the (smaller) stack table.
    let mut seen: HashMap<&[u32], ()> = HashMap::new();
    let mut uflat: Vec<u32> = Vec::new();
    let mut uoff: Vec<u32> = vec![0];
    for r in &ds.rows {
        if seen.insert(r.as_slice(), ()).is_none() {
            uflat.extend_from_slice(r);
            uoff.push(uflat.len() as u32);
        }
    }
    let ws_onpair_table = onpair_structural(&uflat, &uoff, num_distinct);
    let wholestack_onpair = ws_codes + ws_onpair_table;

    // ── zstd references ──
    let raw_text = std::fs::read(format!(
        "{}/data/{}.lst",
        env!("CARGO_MANIFEST_DIR"),
        ds.name
    ))
    .unwrap_or_default();
    let zstd_raw = if raw_text.is_empty() { 0 } else { zstd_len(&raw_text) };
    let int_le: Vec<u8> = ds.flat.iter().flat_map(|&c| c.to_le_bytes()).collect();
    let zstd_ints = zstd_len(&int_le);

    // ── report ──
    println!("\n=== {} ===", ds.name);
    println!(
        "rows={num_rows}  elements={num_elems}  distinct={num_distinct} ({base_bits}b)  \
         avg_len={:.1}",
        num_elems as f64 / num_rows as f64
    );
    println!(
        "onpair: tokens={num_tokens} (+{num_merged} merged, {code_bits}b codes)  \
         codes={} ({:.2}x fewer than elements)  avg_token_elems={:.2}",
        codes.len(),
        num_elems as f64 / codes.len() as f64,
        dict_elems as f64 / num_tokens as f64,
    );
    println!("\n  representation                bytes     ratio_vs_raw");
    let raw = ds.raw_bytes;
    let line = |label: &str, structural: usize, include_strdict: bool| {
        let total = structural + if include_strdict { string_dict } else { 0 };
        println!(
            "  {label:28} {total:8}      {:6.2}x",
            raw as f64 / total as f64
        );
    };
    println!("  {:28} {raw:8}      {:6.2}x", "raw text", 1.0);
    line("dict+listview (+strdict)", listview, true);
    line("wholestack-dict (+strdict)", wholestack, true);
    line("wholestack+onpair (+strdict)", wholestack_onpair, true);
    line("onpair-int    (+strdict)", onpair, true);
    println!("  {:28} {:8}      {:6.2}x", "zstd(raw text)", zstd_raw, raw as f64 / zstd_raw.max(1) as f64);
    println!("  {:28} {:8}      {:6.2}x", "zstd(int stream)", zstd_ints, raw as f64 / zstd_ints.max(1) as f64);
    println!(
        "\n  structural only (no strdict):  listview={listview}  onpair={onpair}  \
         -> onpair saves {:.1}% of the int stream",
        100.0 * (1.0 - onpair as f64 / listview as f64)
    );
    println!(
        "    string dict (shared) = {string_dict} bytes  ({:.0}% of onpair+strdict total)",
        100.0 * string_dict as f64 / (onpair + string_dict) as f64
    );
    println!(
        "    wholestack-dict: {num_uniq_stacks}/{num_rows} distinct stacks  structural={wholestack} \
         (codes={ws_codes} + table={})  onpair {} it by {:.1}%",
        ws_table_vals + ws_table_off,
        if onpair < wholestack { "beats" } else { "loses to" },
        100.0 * (1.0 - onpair.min(wholestack) as f64 / onpair.max(wholestack) as f64),
    );
    // ── focused comparison (structural bytes, no shared string dict) ──
    let zmed = zstd_at(&int_le, 3);
    let zhigh = zstd_at(&int_le, 19);
    println!("\n  focused comparison (structural bytes, no string dict):");
    let cmp = |label: &str, bytes: usize| {
        println!(
            "    {label:30} {bytes:9}   {:7.2}x",
            ds.raw_bytes as f64 / bytes as f64
        );
    };
    cmp("listview (all rows)", listview);
    cmp("listview unique-only (dedup)", wholestack);
    cmp("onpair", onpair);
    cmp("onpair + unique-only", wholestack_onpair);
    cmp("zstd medium (L3, int stream)", zmed);
    cmp("zstd high   (L19, int stream)", zhigh);

    multiround(ds, 6);

    run_sharing_analysis(ds);
}

/// Train + encode + prune one (sub)column; return its minimal structural size
/// (codes + token dict + row offsets), excluding the shared string dict.
fn onpair_structural(flat: &[u32], offsets: &[u32], num_distinct: usize) -> usize {
    let num_rows = offsets.len() - 1;
    let base_bits = bits_for(num_distinct);
    let capacity = capacity_for(num_distinct);
    let parser = intonpair::train(flat, offsets, num_distinct as u32, capacity, 0.5);
    let (codes, _) = parser.encode(flat, offsets);
    let mut used = vec![false; parser.dict.num_tokens()];
    for &c in &codes {
        used[c as usize] = true;
    }
    let (mut num_tokens, mut dict_elems) = (0usize, 0usize);
    for (id, &u) in used.iter().enumerate() {
        if u {
            dict_elems += parser.dict.token(id as u32).len();
            num_tokens += 1;
        }
    }
    let code_bits = bits_for(num_tokens);
    let off_bits = bits_for(dict_elems + 1);
    packed_bytes(codes.len(), code_bits)
        + packed_bytes(dict_elems, base_bits)
        + packed_bytes(num_tokens + 1, off_bits)
        + packed_bytes(num_rows + 1, bits_for(codes.len() + 1))
}

/// If a `<name>.runs` sidecar (one row-count per run) exists, compare storing
/// all runs in one shared-dictionary column vs. compressing each run alone.
fn run_sharing_analysis(ds: &Dataset) {
    let sidecar = format!("{}/data/{}.runs", env!("CARGO_MANIFEST_DIR"), ds.name);
    let Ok(text) = std::fs::read_to_string(&sidecar) else {
        return;
    };
    let run_sizes: Vec<usize> = text.lines().filter_map(|l| l.trim().parse().ok()).collect();
    if run_sizes.is_empty() {
        return;
    }
    let num_distinct = ds.strings.len();

    let shared = onpair_structural(&ds.flat, &ds.offsets, num_distinct);

    // Per-run independent: each run trains its own dictionary over the same
    // global frame alphabet, then we sum the structural bytes.
    let mut per_run = 0usize;
    let mut row = 0usize;
    for &rs in &run_sizes {
        let s = ds.offsets[row] as usize;
        let e = ds.offsets[row + rs] as usize;
        let flat = &ds.flat[s..e];
        let local: Vec<u32> = ds.offsets[row..=row + rs]
            .iter()
            .map(|&o| o - ds.offsets[row])
            .collect();
        per_run += onpair_structural(flat, &local, num_distinct);
        row += rs;
    }

    println!(
        "\n  runs={}  shared-dict column = {shared} B   per-run independent = {per_run} B   \
         -> sharing saves {:.1}%",
        run_sizes.len(),
        100.0 * (1.0 - shared as f64 / per_run as f64)
    );
}

/// Print a human-readable view of the trained dictionary: how it splits into
/// base vs merged tokens, the merged-length histogram, and the most-used merged
/// tokens decoded back to their frame strings. `codes` is the pre-prune,
/// pre-remap code stream so token ids still index `parser.dict` directly.
fn dump_dict(ds: &Dataset, parser: &intonpair::Parser, codes: &[u32], num_distinct: usize) {
    let ntok = parser.dict.num_tokens();
    let mut freq = vec![0u32; ntok];
    for &c in codes {
        freq[c as usize] += 1;
    }
    // length histogram + usage over *used merged* tokens only.
    let mut hist: std::collections::BTreeMap<usize, usize> = Default::default();
    let mut merged: Vec<(u32, usize, u32)> = Vec::new(); // (freq, len, id)
    let (mut used_base, mut used_merged) = (0usize, 0usize);
    for id in 0..ntok {
        if freq[id] == 0 {
            continue;
        }
        if id < num_distinct {
            used_base += 1;
        } else {
            used_merged += 1;
            let len = parser.dict.token(id as u32).len();
            *hist.entry(len).or_default() += 1;
            merged.push((freq[id], len, id as u32));
        }
    }
    let decode = |id: u32| -> String {
        parser
            .dict
            .token(id)
            .iter()
            .map(|&e| ds.strings[e as usize].as_str())
            .collect::<Vec<_>>()
            .join(" ; ")
    };
    println!("\n  --- onpair-int dictionary for {} ---", ds.name);
    println!(
        "  base tokens (1 frame each): {num_distinct} total, {used_base} used by the encoder",
    );
    println!("  merged tokens used: {used_merged}");
    print!("  merged length histogram (frames -> #tokens):");
    for (len, n) in &hist {
        print!(" {len}:{n}");
    }
    println!();
    merged.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));
    println!("  top merged tokens (uses x length: frames):");
    for (f, len, id) in merged.iter().take(12) {
        let s = decode(*id);
        let s = if s.len() > 140 { format!("{}…", &s[..140]) } else { s };
        println!("    {f:5} x {len:2}: {s}");
    }
}

/// Multi-round OnPair (a practical approximation of Re-Pair): feed each round's
/// pruned code stream back in as the next round's element alphabet, so a round-r
/// token expands to up to 16^r base elements — lifting the per-token ceiling.
///
/// Stored artifacts for R rounds: the final code stream + one grammar dictionary
/// per round (token -> previous-level ids) + the row offsets. We prune+renumber
/// each round so widths stay minimal, and verify each round round-trips.
fn multiround(ds: &Dataset, max_rounds: usize) {
    let num_rows = ds.rows.len();
    let num_distinct = ds.strings.len();

    let mut flat = ds.flat.clone();
    let mut offsets = ds.offsets.clone();
    let mut nd = num_distinct; // alphabet size of the current input
    let mut prev_bits = bits_for(num_distinct); // bits per dict element at this level

    // Per-round record for the tree view.
    struct Round {
        codes_len: usize,
        ntokens: usize,
        code_bits: usize,
        dict_elems: usize,
        elem_bits: usize, // bits per grammar-dict element (previous level width)
        dict_bytes: usize,
        codes_bytes: usize,
        rowoff: usize,
    }
    let mut rounds: Vec<Round> = Vec::new();
    let mut dict_bytes_total = 0usize;

    println!("\n  multi-round onpair (Re-Pair-style), structural bytes (no strdict):");
    println!("    round   codes   tokens   dict+codes+rowoff");
    let mut best = usize::MAX;
    let mut best_round = 1usize;
    for round in 1..=max_rounds {
        let capacity = capacity_for(nd);
        let parser = intonpair::train(&flat, &offsets, nd as u32, capacity, 0.5);
        let (codes, code_offsets) = parser.encode(&flat, &offsets);

        // Prune to used tokens and renumber densely.
        let ntok_all = parser.dict.num_tokens();
        let mut used = vec![false; ntok_all];
        for &c in &codes {
            used[c as usize] = true;
        }
        let mut remap = vec![0u32; ntok_all];
        let (mut ntokens, mut dict_elems) = (0usize, 0usize);
        let (mut delems, mut doff) = (Vec::new(), vec![0u32]);
        for (id, &u) in used.iter().enumerate() {
            if u {
                remap[id] = ntokens as u32;
                let tok = parser.dict.token(id as u32);
                delems.extend_from_slice(tok); // ids in the *previous* level's space
                dict_elems += tok.len();
                doff.push(dict_elems as u32);
                ntokens += 1;
            }
        }
        let codes: Vec<u32> = codes.iter().map(|&c| remap[c as usize]).collect();

        // Verify this round reconstructs its input stream.
        let mut decoded = Vec::with_capacity(flat.len());
        for &c in &codes {
            let b = doff[c as usize] as usize;
            let e = doff[c as usize + 1] as usize;
            decoded.extend_from_slice(&delems[b..e]);
        }
        assert_eq!(decoded, flat, "multiround round {round} round-trip failed in {}", ds.name);

        // Size: this round's grammar dict (elems at prev_bits) + dict offsets.
        let dict_bytes = packed_bytes(dict_elems, prev_bits)
            + packed_bytes(ntokens + 1, bits_for(dict_elems + 1));
        dict_bytes_total += dict_bytes;
        let code_bits = bits_for(ntokens);
        let codes_bytes = packed_bytes(codes.len(), code_bits);
        let rowoff = packed_bytes(num_rows + 1, bits_for(codes.len() + 1));
        let total = codes_bytes + dict_bytes_total + rowoff;
        let mark = if total < best {
            best = total;
            best_round = round;
            " *"
        } else {
            ""
        };
        println!("    {round:5}  {:6}  {:6}   {total:9}{mark}", codes.len(), ntokens);
        rounds.push(Round {
            codes_len: codes.len(),
            ntokens,
            code_bits,
            dict_elems,
            elem_bits: prev_bits,
            dict_bytes,
            codes_bytes,
            rowoff,
        });

        if codes.len() == flat.len() {
            break; // no merges happened
        }
        flat = codes;
        offsets = code_offsets;
        nd = ntokens;
        prev_bits = code_bits;
    }

    // Array-tree view of the best configuration (the grammar = dicts 1..=best).
    let r = &rounds[best_round - 1];
    let raw_bits = packed_bytes(ds.flat.len(), bits_for(num_distinct.max(2)));
    println!(
        "\n  array tree — multi-round onpair, best = round {best_round}  ({best} B, {:.2}x vs {raw_bits} B raw bits)",
        raw_bits as f64 / best as f64
    );
    println!(
        "    onpair-grammar(list, rows={num_rows})                       {best} B (100.0%)",
    );
    let pct = |b: usize| 100.0 * b as f64 / best as f64;
    println!(
        "    ├─ codes        bitpacked(u{:<2} len={})   {} B ({:.1}%)",
        r.code_bits, r.codes_len, r.codes_bytes, pct(r.codes_bytes)
    );
    println!(
        "    ├─ row_offsets  bitpacked(len={})            {} B ({:.1}%)",
        num_rows + 1,
        r.rowoff,
        pct(r.rowoff)
    );
    let gram: usize = rounds[..best_round].iter().map(|x| x.dict_bytes).sum();
    println!("    └─ grammar ({best_round} dict layer(s))                  {gram} B ({:.1}%)", pct(gram));
    for lvl in (1..=best_round).rev() {
        let d = &rounds[lvl - 1];
        let target = if lvl == 1 {
            "base bool".to_string()
        } else {
            format!("L{} tokens", lvl - 1)
        };
        println!(
            "       ├─ dict_{lvl}  token→{target:<10}  {} tokens, {} elems (u{}) {} B ({:.1}%)",
            d.ntokens, d.dict_elems, d.elem_bits, d.dict_bytes, pct(d.dict_bytes)
        );
    }
}

fn verify(ds: &Dataset, parser: &intonpair::Parser, codes: &[u32], code_offsets: &[u32]) {
    for r in 0..ds.rows.len() {
        let mut decoded = Vec::new();
        for &c in &codes[code_offsets[r] as usize..code_offsets[r + 1] as usize] {
            decoded.extend_from_slice(parser.dict.token(c));
        }
        assert_eq!(decoded, ds.rows[r], "round-trip mismatch row {r} in {}", ds.name);
    }
}

fn main() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data");
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("data dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "lst").unwrap_or(false))
        .filter(|p| {
            args.is_empty()
                || args
                    .iter()
                    .any(|a| p.file_stem().map(|s| s == a.as_str()).unwrap_or(false))
        })
        .collect();
    files.sort();
    if files.is_empty() {
        eprintln!("no .lst files in {dir:?}; run prep/build_datasets.py first");
        std::process::exit(1);
    }
    println!("Generalized integer-OnPair on dictionary-encoded list columns");
    for f in &files {
        let ds = load(f);
        run(&ds);
    }
}
