// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fsst::Compressor;

use crate::fsst::transcode_fsst_to_onpair;
use crate::{Dictionary, DictionaryView};

const CORPUS: [&str; 10] = [
    "https://example.com/api/v1/users",
    "https://example.com/api/v1/users/42",
    "https://example.com/api/v1/orders",
    "https://example.com/api/v1/orders/42/items",
    "https://example.com/api/v2/users",
    "https://example.com/api/v2/orders",
    "https://example.org/api/v1/users",
    "https://example.org/api/v1/orders",
    "https://cdn.example.com/static/app.js",
    "https://cdn.example.com/static/app.css",
];

const PATTERNS: [&[u8]; 4] = [b"test,", b"/static/", b"/api/v2/", b"zzz"];

#[test]
fn test_fsst_prefilter() {
    use crate::search::index::build_token_frequency_index;
    use crate::search::{analyze_prefilter, prefilter_candidates};

    let lines: Vec<&[u8]> = CORPUS.iter().map(|s| s.as_bytes()).collect();
    let lines = (0..10).flat_map(|_| lines.clone()).collect();
    let compressor = Compressor::train(&lines);
    let compressed = compressor.compress_bulk(&lines);
    let mut row_offsets = vec![0u32];
    for line in &compressed {
        row_offsets.push(row_offsets[row_offsets.len() - 1] + line.len() as u32);
    }
    let codes: Vec<u8> = compressed.concat();

    let n_esp = codes.iter().filter(|&&b| b == fsst::ESCAPE_CODE).count();
    println!(
        "raw compressed stream has {} bytes, {} escape bytes",
        codes.len(),
        n_esp
    );

    let (dictionary, tokens) = transcode_fsst_to_onpair(&compressor, &codes).unwrap();
    let dictionary_view = dictionary.as_view();
    println!(
        "dictionary has {} tokens; token stream has {} codes",
        dictionary.num_tokens(),
        tokens.len()
    );
    // The escape code is a token id in its own right, so the frequency index has
    // to span it even when the dictionary itself is smaller.
    const ESCAPE: crate::Token = fsst::ESCAPE_CODE as crate::Token;
    let escape_token = Some(ESCAPE);
    let frequency_array_size = dictionary.num_tokens().max(ESCAPE as usize + 1);
    let token_frequencies = build_token_frequency_index(&tokens, frequency_array_size).unwrap();

    // The transcoded stream is u8-coded over the same token ids the cover names.
    for pattern in PATTERNS {
        let shown = String::from_utf8_lossy(pattern);
        let analysis =
            analyze_prefilter(pattern, dictionary_view, &token_frequencies, escape_token);
        let cover = analysis.probe_cover();
        assert!(!cover.points().is_empty() || !cover.ranges().is_empty());

        println!("\n{shown}:");
        println!("  points:");
        for &id in cover.points() {
            // check if the token is the escape token
            if let Some(escape_id) = escape_token
                && id == escape_id
            {
                println!("    {id:>4} = \"<ESCAPE>\"");
                continue;
            }
            println!(
                "    {id:>4} = \"{}\"",
                String::from_utf8_lossy(dictionary_view.token(id))
            );
        }
        println!("  ranges:");
        for r in cover.ranges() {
            println!(
                "    {:>4}..={:<4} = \"{}\" ..= \"{}\"",
                r.begin,
                r.last,
                String::from_utf8_lossy(dictionary_view.token(r.begin)),
                String::from_utf8_lossy(dictionary_view.token(r.last))
            );
        }
        println!(
            "  covered {}/{} ({:.1}%)",
            analysis.covered_frequency(),
            analysis.total_frequency(),
            analysis.covered_fraction() * 100.0
        );

        let mut candidates = Vec::new();
        prefilter_candidates(&tokens, &row_offsets, &analysis, &mut candidates).unwrap();
        let want =
            (0..lines.len()).filter(|&row| lines[row].windows(pattern.len()).any(|w| w == pattern));
        assert!(
            want.clone().all(|row| candidates.contains(&row)),
            "unsound for {shown}"
        );
        assert!(candidates.windows(2).all(|w| w[0] < w[1]));
        println!(
            "  {} candidates for {} matches",
            candidates.len(),
            want.count()
        );
    }

    let mut ranked: Vec<(crate::Token, u32)> = (0..dictionary_view.num_tokens() as crate::Token)
        .map(|id| (id, token_frequencies.frequency(id)))
        .collect();
    ranked.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
}
