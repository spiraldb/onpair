// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Shared test corpus generators. These replicate the helpers in the C++
// `tests/helpers/corpus.h` so the unit tests in `trainer`, `parser`,
// `decoder`, and `search` all draw from the same data and can be compared
// head-to-head against the upstream test suite.

use rand::RngExt;
use rand::SeedableRng;

use crate::core::dictionary::{CompactDictionary, CompactDictionaryView, DictionaryView};
use crate::core::types::Token;
use crate::encoding::lpm::LongestPrefixMatcher;

/// Bytes for property tests. Few distinct bytes make shared prefixes and
/// nested tokens common; `0` and `0xFF` are the zero padding of keys and the
/// pad key.
pub(crate) const PROPERTY_ALPHABET: [u8; 5] = [b'a', b'b', b'c', 0, 0xFF];

/// Sorted, complete dictionary: every single byte plus `extra`.
pub(crate) fn complete_dictionary(extra: &[&[u8]]) -> CompactDictionary {
    let mut tokens: Vec<Vec<u8>> = (0u16..=255).map(|i| vec![i as u8]).collect();
    tokens.extend(extra.iter().map(|t| t.to_vec()));
    tokens.sort();
    tokens.dedup();

    let mut bytes = Vec::new();
    let mut offsets = vec![0u32];
    for t in &tokens {
        bytes.extend_from_slice(t);
        offsets.push(bytes.len() as u32);
    }
    CompactDictionary::from_raw(bytes, offsets)
}

/// The matcher the trainer uses, holding the tokens of `dict`, and the bytes
/// of each of its token ids. Its ids differ from the dictionary ids.
pub(crate) fn trainer_matcher(
    dict: CompactDictionaryView<'_>,
) -> (LongestPrefixMatcher, Vec<Vec<u8>>) {
    let mut matcher = LongestPrefixMatcher::new();
    let mut tokens: Vec<Vec<u8>> = (0u16..=255).map(|b| vec![b as u8]).collect();
    for id in 0..dict.num_tokens() {
        let token = dict.token(id as Token);
        if token.len() > 1 {
            let new_id = matcher.insert(token);
            debug_assert_eq!(usize::from(new_id), tokens.len());
            tokens.push(token.to_vec());
        }
    }
    (matcher, tokens)
}

/// Arrow-style flat representation of a list of byte strings.
pub(crate) struct Raw {
    pub data: Vec<u8>,
    pub offsets: Vec<u32>,
}

pub(crate) fn make_raw<S: AsRef<[u8]>>(strings: &[S]) -> Raw {
    let mut data = Vec::new();
    let mut offsets = Vec::with_capacity(strings.len() + 1);
    offsets.push(0u32);
    for s in strings {
        data.extend_from_slice(s.as_ref());
        offsets.push(data.len() as u32);
    }
    Raw { data, offsets }
}

/// URL-shaped repetitive corpus — easy BPE merge targets.
pub(crate) fn user_strings(n: usize) -> Vec<String> {
    const BASES: &[&str] = &[
        "https://www.example.com/page",
        "https://www.example.com/data",
        "https://www.test.org/page",
        "ftp://files.example.com/x",
        "https://docs.example.com/spec",
        "https://api.example.net/v1",
    ];
    (0..n).map(|i| BASES[i % BASES.len()].to_string()).collect()
}

/// All copies of the same single-character string of length `len`.
pub(crate) fn homogeneous_strings(n: usize, len: usize, ch: u8) -> Vec<Vec<u8>> {
    (0..n).map(|_| vec![ch; len]).collect()
}

/// "abab..." period-2 strings of length `len`.
pub(crate) fn alternating_strings(n: usize, len: usize) -> Vec<Vec<u8>> {
    (0..n)
        .map(|_| {
            (0..len)
                .map(|i| if i.is_multiple_of(2) { b'a' } else { b'b' })
                .collect()
        })
        .collect()
}

/// Random ascii lowercase strings, length 1..=max_len.
pub(crate) fn random_ascii_strings(n: usize, max_len: usize, seed: u64) -> Vec<Vec<u8>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let l = rng.random_range(1..=max_len);
            (0..l).map(|_| rng.random_range(b'a'..=b'z')).collect()
        })
        .collect()
}

/// Random bytes over the full 0..=255 range.
pub(crate) fn binary_strings(n: usize, max_len: usize, seed: u64) -> Vec<Vec<u8>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let l = rng.random_range(1..=max_len);
            (0..l).map(|_| rng.random_range(0..=255u32) as u8).collect()
        })
        .collect()
}

/// `n` fixed-length strings of length `len`, content rotating through the
/// lowercase alphabet so adjacent strings differ.
pub(crate) fn fixed_length_strings(n: usize, len: usize) -> Vec<Vec<u8>> {
    (0..n)
        .map(|i| (0..len).map(|j| b'a' + ((i + j) as u8 % 26)).collect())
        .collect()
}

/// Random strings with length 0..=max_len — exercises empty + max-len paths.
pub(crate) fn mixed_length_strings(n: usize, max_len: usize, seed: u64) -> Vec<Vec<u8>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let l = rng.random_range(0..=max_len);
            (0..l).map(|_| rng.random_range(b'a'..=b'z')).collect()
        })
        .collect()
}
