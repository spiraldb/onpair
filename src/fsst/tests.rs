// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fsst::Compressor;

use crate::fsst::transcode_onpair;
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

#[test]
fn test_fsst_prefilter() {
    use crate::search::analyze_prefilter;
    use crate::search::index::build_token_frequency_index;


    let lines: Vec<&[u8]> = CORPUS.iter().map(|s| s.as_bytes()).collect();
    let lines = (0..10).flat_map(|_| lines.clone()).collect();
    let compressor = Compressor::train(&lines);
    let codes: Vec<u8> = compressor.compress_bulk(&lines).concat();

    let n_esp = codes.iter().filter(|&&b| b == fsst::ESCAPE_CODE).count();
    println!(
        "raw compressed stream has {} bytes, {} escape bytes",
        codes.len(),
        n_esp
    );

    let (dictionary, tokens) = transcode_onpair(&compressor, &codes).unwrap();
    let dictionary_view = dictionary.as_view();
    println!(
        "dictionary has {} tokens; token stream has {} codes",
        dictionary.num_tokens(),
        tokens.len()
    );
    let escape_token = Some(fsst::ESCAPE_CODE as u16);
    let frequency_array_size = dictionary.num_tokens().max(escape_token.unwrap_or(0) as usize + 1);
    let token_frequencies = build_token_frequency_index(&tokens, frequency_array_size).unwrap();

    let pat = b"test,";
    let analysis = analyze_prefilter(pat, dictionary_view, &token_frequencies, escape_token);
    let cover = analysis.probe_cover();

    // let pat = b"static";
    // let analysis = analyze_prefilter(pat, dictionary_view, &token_frequencies);
    // let cover = analysis.probe_cover();

    println!("  points:");
    for &id in cover.points() {
        // check if the token is the escape token
        if let Some(escape_id) = escape_token {
            if id == escape_id {
                println!("    {id:>4} = \"<ESCAPE>\"");
                continue;
            }
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

    let mut ranked: Vec<(crate::Token, u32)> = (0..dictionary_view.num_tokens() as crate::Token)
        .map(|id| (id, token_frequencies.frequency(id)))
        .collect();
    ranked.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    assert!(!cover.points().is_empty() || !cover.ranges().is_empty());
}
