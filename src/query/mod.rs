// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compressed-domain `contains` queries.
//!
//! [`ContainsSearcher`] finds the rows of a compressed column whose
//! decompressed bytes contain a byte pattern, without decompressing. Two
//! cooperating pieces:
//!
//! * a **DFA over codes** ([`dfa`]): a KMP byte automaton composed with every
//!   dictionary token, so scanning a row is one transition-table load per
//!   code and matches may straddle token boundaries; and
//! * a **prefilter** ([`prefilter`]): a SIMD pass over the raw code stream
//!   that proves most rows *definitely* don't match (no false negatives), so
//!   the DFA only runs on the few candidate rows. Selective patterns — the
//!   common case for `contains` — make this the hot path.
//!
//! ```ignore
//! use onpair::{compress, query::ContainsSearcher, DEFAULT_CONFIG};
//!
//! let col = compress(&bytes, &offsets, DEFAULT_CONFIG)?;
//! let searcher = ContainsSearcher::compile(col.as_parts(), b"google");
//! let rows = searcher.matching_rows(&col.codes, &col.code_offsets);
//! ```

mod dfa;
mod prefilter;

use crate::Offset;
use crate::Parts;

use dfa::TokenDfa;
use prefilter::Prefilter;

pub use dfa::MAX_PATTERN_LEN;

/// A `contains` query compiled against one column's dictionary (and tuned on
/// its code stream). Reusable across [`matching_rows`] calls for any code
/// stream produced with the same dictionary.
///
/// [`matching_rows`]: ContainsSearcher::matching_rows
pub struct ContainsSearcher {
    /// `None` for the empty pattern, which matches every row.
    dfa: Option<TokenDfa>,
    /// `None` when no anchor was selective enough to pay for a prefilter
    /// pass; rows then go straight to the DFA.
    prefilter: Option<Prefilter>,
}

impl ContainsSearcher {
    /// Compile a searcher for `pattern` against the column viewed by `parts`.
    ///
    /// The dictionary drives the automaton; `parts.codes` is only *sampled*
    /// to pick the most selective prefilter anchor, so the searcher remains
    /// exact for any code stream over the same dictionary.
    ///
    /// ## Panics
    ///
    /// Panics if `parts` fails [`Parts::validate`] or if
    /// `pattern.len() > MAX_PATTERN_LEN`.
    pub fn compile(parts: Parts<'_>, pattern: &[u8]) -> Self {
        if let Err(e) = parts.validate() {
            panic!("onpair: {e}");
        }
        if pattern.is_empty() {
            return Self {
                dfa: None,
                prefilter: None,
            };
        }
        let dfa = TokenDfa::build(pattern, parts.dict_bytes, parts.dict_offsets);
        let prefilter =
            Prefilter::build(pattern, parts.dict_bytes, parts.dict_offsets, parts.codes);
        Self {
            dfa: Some(dfa),
            prefilter,
        }
    }

    /// Indices of the rows whose decompressed bytes contain the pattern.
    ///
    /// `code_offsets` delimits rows exactly as [`crate::Column::code_offsets`]
    /// does: row `r` is `codes[code_offsets[r]..code_offsets[r + 1]]`.
    ///
    /// ## Panics
    ///
    /// Panics if `code_offsets` is malformed (non-monotonic or out of bounds
    /// for `codes`) or if a code is out of range for the dictionary this
    /// searcher was compiled against.
    pub fn matching_rows<O: Offset>(&self, codes: &[u16], code_offsets: &[O]) -> Vec<u64> {
        match (&self.dfa, &self.prefilter) {
            (None, _) => (0..code_offsets.len().saturating_sub(1) as u64).collect(),
            (Some(dfa), None) => Self::scan_rows(codes, code_offsets, |row| dfa.row_matches(row)),
            (Some(dfa), Some(pf)) => {
                let mut mask = vec![0u64; codes.len().div_ceil(64)];
                pf.candidate_mask(codes, &mut mask);
                let mut out = Vec::new();
                Self::for_each_row(codes, code_offsets, |r, a, b, row| {
                    if prefilter::any_bit_in_range(&mask, a, b) && dfa.row_matches(row) {
                        out.push(r);
                    }
                });
                out
            }
        }
    }

    /// Like [`matching_rows`](Self::matching_rows) but with the prefilter
    /// disabled: every row is scanned by the DFA. The exact baseline the
    /// prefilter is measured against.
    pub fn matching_rows_unfiltered<O: Offset>(
        &self,
        codes: &[u16],
        code_offsets: &[O],
    ) -> Vec<u64> {
        match &self.dfa {
            None => (0..code_offsets.len().saturating_sub(1) as u64).collect(),
            Some(dfa) => Self::scan_rows(codes, code_offsets, |row| dfa.row_matches(row)),
        }
    }

    /// Rows the prefilter cannot rule out (a superset of the matching rows).
    /// Exposed for measurement: candidate count / row count is the prefilter's
    /// false-positive-inclusive pass rate. Without a prefilter, every row is a
    /// candidate.
    pub fn candidate_rows<O: Offset>(&self, codes: &[u16], code_offsets: &[O]) -> Vec<u64> {
        match &self.prefilter {
            None => (0..code_offsets.len().saturating_sub(1) as u64).collect(),
            Some(pf) => {
                let mut mask = vec![0u64; codes.len().div_ceil(64)];
                pf.candidate_mask(codes, &mut mask);
                let mut out = Vec::new();
                Self::for_each_row(codes, code_offsets, |r, a, b, _| {
                    if prefilter::any_bit_in_range(&mask, a, b) {
                        out.push(r);
                    }
                });
                out
            }
        }
    }

    /// Prefilter diagnostics: `(strategy, expected per-code hit rate)`, or
    /// `None` when the searcher runs unfiltered.
    pub fn prefilter_info(&self) -> Option<(&'static str, f64)> {
        self.prefilter
            .as_ref()
            .map(|pf| (pf.strategy(), pf.expected_hit_rate()))
    }

    /// Iterate rows, handing each `(row, code_start, code_end, row_codes)` to
    /// `f`. Panics on malformed offsets.
    #[inline]
    fn for_each_row<O: Offset>(
        codes: &[u16],
        code_offsets: &[O],
        mut f: impl FnMut(u64, usize, usize, &[u16]),
    ) {
        for (r, w) in code_offsets.windows(2).enumerate() {
            let a = w[0].to_usize().expect("row offset overflows usize");
            let b = w[1].to_usize().expect("row offset overflows usize");
            assert!(a <= b && b <= codes.len(), "malformed code_offsets");
            f(r as u64, a, b, &codes[a..b]);
        }
    }

    /// Collect the rows for which `matches` returns true.
    #[inline]
    fn scan_rows<O: Offset>(
        codes: &[u16],
        code_offsets: &[O],
        mut matches: impl FnMut(&[u16]) -> bool,
    ) -> Vec<u64> {
        let mut out = Vec::new();
        Self::for_each_row(codes, code_offsets, |r, _, _, row| {
            if matches(row) {
                out.push(r);
            }
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Bits, Config, DEFAULT_CONFIG, compress};

    fn pack(rows: &[&[u8]]) -> (Vec<u8>, Vec<u32>) {
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for r in rows {
            bytes.extend_from_slice(r);
            offsets.push(bytes.len() as u32);
        }
        (bytes, offsets)
    }

    /// Ground truth: decompressed per-row `contains`.
    fn naive(rows: &[&[u8]], pattern: &[u8]) -> Vec<u64> {
        rows.iter()
            .enumerate()
            .filter(|(_, r)| pattern.is_empty() || r.windows(pattern.len()).any(|w| w == pattern))
            .map(|(i, _)| i as u64)
            .collect()
    }

    fn corpus() -> Vec<Vec<u8>> {
        // URL-shaped rows with repetition so training builds long tokens.
        let hosts = ["https://www.google.com", "https://yandex.ru", "ftp://x.org"];
        let paths = ["/search?q=", "/maps/place/", "/", "/login", "/img.png"];
        (0..4000usize)
            .map(|i| {
                format!(
                    "{}{}{}",
                    hosts[i % hosts.len()],
                    paths[(i / 3) % paths.len()],
                    i % 97
                )
                .into_bytes()
            })
            .collect()
    }

    #[test]
    fn agrees_with_naive_contains() {
        let rows_owned = corpus();
        let rows: Vec<&[u8]> = rows_owned.iter().map(|r| r.as_slice()).collect();
        let (bytes, offsets) = pack(&rows);

        for bits in [9u8, 12, 16] {
            let cfg = Config {
                bits: Bits::new(bits).unwrap(),
                ..DEFAULT_CONFIG
            };
            let col = compress(&bytes, &offsets, cfg).unwrap();
            for pattern in [
                &b"google"[..],
                b"maps/place",
                b"yandex.ru/login",
                b"q=",
                b"zzz-no-match",
                b"g",
                b"https://www.google.com/search?q=1", // full row prefix
                b"7",
            ] {
                let s = ContainsSearcher::compile(col.as_parts(), pattern);
                let expect = naive(&rows, pattern);
                let got = s.matching_rows(&col.codes, &col.code_offsets);
                assert_eq!(got, expect, "bits={bits} pattern={:?}", pattern);
                let unfiltered = s.matching_rows_unfiltered(&col.codes, &col.code_offsets);
                assert_eq!(unfiltered, expect, "unfiltered bits={bits}");

                // Candidates must be a superset of matches (no false negatives).
                let cand = s.candidate_rows(&col.codes, &col.code_offsets);
                let mut it = cand.iter().copied();
                assert!(
                    expect.iter().all(|&m| it.any(|c| c == m)),
                    "prefilter dropped a matching row: bits={bits} pattern={:?}",
                    pattern
                );
            }
        }
    }

    /// Randomized cross-check over a tiny alphabet: heavy token merging makes
    /// most matches straddle token boundaries, the hard case for both the DFA
    /// and the prefilter's anchor reasoning.
    #[test]
    fn randomized_small_alphabet_agrees_with_naive() {
        let mut x = 0x243F6A8885A308D3u64;
        let mut rng = move || {
            x = x.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z ^ (z >> 31)
        };
        let rows_owned: Vec<Vec<u8>> = (0..2000)
            .map(|_| {
                let len = (rng() % 40) as usize;
                (0..len).map(|_| b'a' + (rng() % 4) as u8).collect()
            })
            .collect();
        let rows: Vec<&[u8]> = rows_owned.iter().map(|r| r.as_slice()).collect();
        let (bytes, offsets) = pack(&rows);
        let col = compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap();

        for trial in 0..40 {
            // Half the patterns are real substrings (guaranteed matches),
            // half are random (mostly misses).
            let pattern: Vec<u8> = if trial % 2 == 0 {
                let r = &rows_owned[(rng() as usize) % rows_owned.len()];
                if r.is_empty() {
                    continue;
                }
                let s = (rng() as usize) % r.len();
                let e = s + 1 + (rng() as usize) % (r.len() - s);
                r[s..e].to_vec()
            } else {
                let len = 1 + (rng() % 12) as usize;
                (0..len).map(|_| b'a' + (rng() % 4) as u8).collect()
            };
            let s = ContainsSearcher::compile(col.as_parts(), &pattern);
            assert_eq!(
                s.matching_rows(&col.codes, &col.code_offsets),
                naive(&rows, &pattern),
                "pattern={:?}",
                String::from_utf8_lossy(&pattern)
            );
        }
    }

    #[test]
    fn empty_pattern_matches_all_rows() {
        let rows: Vec<&[u8]> = vec![b"abc", b"", b"xyz"];
        let (bytes, offsets) = pack(&rows);
        let col = compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap();
        let s = ContainsSearcher::compile(col.as_parts(), b"");
        assert_eq!(
            s.matching_rows(&col.codes, &col.code_offsets),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn empty_rows_and_pattern_longer_than_row() {
        let rows: Vec<&[u8]> = vec![b"", b"ab", b"abab", b""];
        let (bytes, offsets) = pack(&rows);
        let col = compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap();
        let s = ContainsSearcher::compile(col.as_parts(), b"abab");
        assert_eq!(s.matching_rows(&col.codes, &col.code_offsets), vec![2]);
    }

    #[test]
    fn match_spanning_many_tokens() {
        // A pattern far longer than MAX_TOKEN_SIZE must still be found.
        let row = b"begin-0123456789abcdefghij-ABCDEFGHIJKLMNOPQRSTUVWXYZ-end".to_vec();
        let rows: Vec<&[u8]> = vec![&row, b"other row"];
        let (bytes, offsets) = pack(&rows);
        let col = compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap();
        let s = ContainsSearcher::compile(
            col.as_parts(),
            b"0123456789abcdefghij-ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        );
        assert_eq!(s.matching_rows(&col.codes, &col.code_offsets), vec![0]);
    }
}
