// SPDX-License-Identifier: Apache-2.0
//! Trace the substring prefilter over a tiny column.
//!
//! Run with: `cargo run --example inspect_prefilter`

use onpair::search::index::build_token_frequency_index;
use onpair::search::{
    BytesVerifier, analyze_prefilter, prefilter_candidates, prefilter_is_likely_profitable,
};
use onpair::{Column, DECODE_PADDING, DEFAULT_CONFIG, DictionaryView};
use std::mem::MaybeUninit;

/// Ten rows. Deliberately sharing substrings, which is what OnPair compresses.
const ROWS: &[&str] = &[
    "https://www.example.com/page/1",
    "https://www.example.com/page/2",
    "https://www.example.com/data/a",
    "https://docs.example.com/spec",
    "https://api.example.net/v1/users",
    "ftp://files.example.com/x",
    "https://www.test.org/page/1",
    "mailto:alice@example.com",
    "https://www.example.com/page/3",
    "postgres://db.internal:5432/main",
];

const PATTERNS: &[&str] = &["page", "example.com", "z", "https", "/v1/", "docs"];

fn show(bytes: &[u8]) -> String {
    format!("\"{}\"", String::from_utf8_lossy(bytes))
}

/// Render a row list without `Debug`, which the package's lints deny.
fn list(rows: &[usize]) -> String {
    let mut out = String::from("[");
    for (i, row) in rows.iter().enumerate() {
        if i != 0 {
            out.push_str(", ");
        }
        out.push_str(&row.to_string());
    }
    out.push(']');
    out
}

/// Grow the corpus by cycling the base rows with a varying tail, so the trainer
/// has enough repetition to learn multi-byte tokens.
fn synthesize(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| format!("{}/{}", ROWS[i % ROWS.len()], i))
        .collect()
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(0);
    let owned = synthesize(n);
    let rows: Vec<&str> = if n == 0 {
        ROWS.to_vec()
    } else {
        owned.iter().map(String::as_str).collect()
    };

    // Pack the rows into the Arrow (bytes, offsets) pair OnPair takes.
    let mut bytes = Vec::new();
    let mut offsets = vec![0u32];
    for row in &rows {
        bytes.extend_from_slice(row.as_bytes());
        offsets.push(bytes.len() as u32);
    }

    let col = Column::compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap();
    let view = col.view();
    let num_tokens = view.dict.num_tokens();

    println!("== column ==");
    println!("rows            {}", view.num_rows());
    println!("raw bytes       {}", bytes.len());
    println!("codes           {}", view.codes.len());
    println!("dictionary      {num_tokens} tokens");
    println!("code bits       {}", view.dict.code_bits());

    // The reusable selectivity index the planner weights its cut by.
    let freqs = build_token_frequency_index(view.codes, num_tokens).unwrap();

    println!("\n== how row 0 was parsed ==");
    print!("row 0 codes ->");
    for &code in view.row_codes(0) {
        print!(" [{}]", show(view.dict.token(code)));
    }
    println!();

    for pattern in PATTERNS {
        let pat = pattern.as_bytes();
        println!("\n== pattern \"{pattern}\" ==");

        // Phase 1: build the alignment DAG, cut it, normalize the cut into a cover.
        let analysis = analyze_prefilter(pat, view.dict, &freqs);
        let cover = analysis.probe_cover();

        print!("probe points  ");
        if cover.points().is_empty() {
            println!("(none)");
        } else {
            for &p in cover.points() {
                print!("{p}:{} ", show(view.dict.token(p)));
            }
            println!();
        }
        print!("probe ranges  ");
        if cover.ranges().is_empty() {
            println!("(none)");
        } else {
            for r in cover.ranges() {
                print!(
                    "{}..={} ({}..={}) ",
                    r.begin,
                    r.last,
                    show(view.dict.token(r.begin)),
                    show(view.dict.token(r.last))
                );
            }
            println!();
        }

        println!(
            "cut cost      {} covered codes of {} ({:.2}% of the stream)",
            analysis.covered_frequency(),
            analysis.total_frequency(),
            analysis.covered_fraction() * 100.0
        );
        println!(
            "scan cost     {} SIMD comparisons per vector",
            analysis.comparison_cost()
        );
        println!(
            "est. verify   {:.1}% of rows",
            analysis.expected_candidate_row_fraction(view.num_rows()) * 100.0
        );
        println!(
            "profitable?   {}",
            prefilter_is_likely_profitable(&analysis, view.num_rows())
        );

        // Phase 2: scan the code stream for the cover. Sound superset.
        let mut candidates = Vec::new();
        prefilter_candidates(view.codes, view.row_offsets, &analysis, &mut candidates).unwrap();

        // Phase 3: exact check on the survivors only.
        let mut verified = candidates.clone();
        BytesVerifier::new(pat).retain(view, &mut verified);

        // Ground truth, by decoding everything and searching the bytes.
        let truth: Vec<usize> = (0..view.num_rows())
            .filter(|&k| {
                let mut buf = vec![MaybeUninit::uninit(); view.row_decoded_len(k) + DECODE_PADDING];
                // SAFETY: buffer sized for row `k` from a column valid by construction.
                let n = unsafe { view.decompress_row_into(k, &mut buf) };
                let row = unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), n) };
                row.windows(pat.len()).any(|w| w == pat)
            })
            .collect();

        if view.num_rows() <= 20 {
            println!("candidates    {}", list(&candidates));
            println!("verified      {}", list(&verified));
            println!("truth         {}", list(&truth));
        }
        println!(
            "admitted      {} rows of {} ({:.1}%)",
            candidates.len(),
            view.num_rows(),
            candidates.len() as f64 / view.num_rows() as f64 * 100.0
        );
        println!("matched       {} rows", verified.len());
        assert_eq!(verified, truth, "prefilter disagreed with brute force");
        assert!(
            truth.iter().all(|r| candidates.contains(r)),
            "unsound: a true match was not admitted"
        );
        let waste = candidates.len() - verified.len();
        println!("false admits  {waste}");
    }

    println!("\nall patterns agreed with brute force.");
}
