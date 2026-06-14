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
    let capacity = (num_distinct.next_power_of_two() * 4).min(1 << 20);
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
    let codes: Vec<u32> = codes.iter().map(|&c| remap[c as usize]).collect();
    let code_bits = bits_for(num_tokens);
    let off_bits = bits_for(dict_elems + 1);

    let op_codes = packed_bytes(codes.len(), code_bits);
    let op_tokdict = packed_bytes(dict_elems, base_bits) + packed_bytes(num_tokens + 1, off_bits);
    let op_rowoff = packed_bytes(num_rows + 1, bits_for(codes.len() + 1));
    let onpair = op_codes + op_tokdict + op_rowoff;

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
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("data dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "lst").unwrap_or(false))
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
