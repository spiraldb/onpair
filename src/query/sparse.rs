// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sparse compression of the token-DFA state space, and the scan shapes it
//! unlocks.
//!
//! The dense [`TokenDfa`](super::dfa::TokenDfa) table is `(m + 1) × ntokens`
//! entries. Almost all of it is redundant: composing the byte automaton over
//! a token from state `s` gives the same result as composing from state `0`
//! unless the token's first byte extends some pattern prefix alive at `s` —
//! the walks converge immediately and stay converged. So the state space
//! compresses to:
//!
//!   * `base[c] = δ*(0, c)` — one byte per token (the only dense array); and
//!   * per-state **exception lists** — the few `(code, dest)` pairs with
//!     `δ*(s, c) ≠ base[c]`, found by walking only the tokens whose first
//!     byte transitions differently from `s` than from `0`.
//!
//! Two consequences shape the scan:
//!
//!   * A **boring** code (`base[c] == 0`, no exceptions) maps *every* state
//!     to `0`: the automaton forgets everything. Only runs of consecutive
//!     **interesting** codes can carry a match, and those runs are short —
//!     a match spanning `t` tokens is `t` consecutive interesting codes.
//!   * Any occurrence of an `m`-byte pattern needs at least
//!     `ceil(m / max_interesting_token_len)` consecutive interesting codes.
//!     Broadcasting the interesting set into a positional mask `M` with the
//!     prefilter's SIMD interval scan and AND-ing shifted copies
//!     (`M & M>>1 & M>>2`, capped at 3) is therefore a *combination*
//!     prefilter over merged 2- and 3-token windows — a strictly stronger
//!     necessary condition than any single-anchor candidate set, and it
//!     needs no frequency information to compile.

use super::dfa::MAX_PATTERN_LEN;
use super::dfa::byte_automaton;
use super::prefilter::Filter;
use super::prefilter::any_bit_in_range;
use crate::Offset;
use crate::Parts;

/// Cap on the shifted-mask combination depth: windows longer than 3 tokens
/// add compares for rapidly diminishing selectivity.
const MAX_COMBO: usize = 3;

/// A `contains` query over the sparse token-DFA representation. Compiled from
/// the dictionary alone — no code-stream sampling, no stored frequency
/// statistics — and reusable across code streams produced with the same
/// dictionary.
pub struct SparseSearcher {
    /// `None` for the empty pattern, which matches every row.
    inner: Option<Inner>,
}

struct Inner {
    /// `δ*(0, c)` per token: the dense but tiny (one byte per token) core of
    /// the compressed state space.
    base: Vec<u8>,
    /// `exceptions[s]`, sorted by code: the `(code, dest)` pairs with
    /// `δ*(s, c) ≠ base[c]`. Indices `0` and `m` are empty (state `0` is the
    /// definition of `base`; the accept state is absorbing and never walked).
    exceptions: Vec<Vec<(u16, u8)>>,
    /// SIMD-scannable membership test for the interesting-code set
    /// (`base[c] != 0` or any exception mentions `c`).
    filter: Filter,
    /// Minimum run of consecutive interesting codes a match requires,
    /// `1..=MAX_COMBO`.
    combo: usize,
    /// `pattern.len()`, the absorbing accept state.
    accept: u8,
    /// `None` when no token is interesting: the pattern cannot occur in any
    /// code stream over this dictionary.
    ntokens: usize,
}

/// Compile-time shape of a [`SparseSearcher`], for diagnostics and
/// measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseInfo {
    /// Bytes resident for the transition structure: `base` plus the
    /// exception lists (the dense equivalent is `(m + 1) × ntokens`).
    pub transition_bytes: usize,
    /// Total `(state, code)` exception pairs.
    pub exceptions: usize,
    /// Interesting codes (candidate-set population).
    pub interesting: usize,
    /// Combination depth of the shifted-mask prefilter (consecutive
    /// interesting codes a match must contain).
    pub combo: usize,
}

impl SparseSearcher {
    /// Compile a searcher for `pattern` against the column viewed by `parts`.
    /// Only the dictionary is read; cost is `O(dict)` plus one short walk per
    /// (state, differing-first-byte-token) pair.
    ///
    /// ## Panics
    ///
    /// Panics if the dictionary fails [`Parts::validate_dictionary`] or if
    /// `pattern.len() > MAX_PATTERN_LEN`.
    pub fn compile(parts: Parts<'_>, pattern: &[u8]) -> Self {
        if let Err(e) = parts.validate_dictionary() {
            panic!("onpair: {e}");
        }
        Self::compile_dict(parts.dict_bytes, parts.dict_offsets, pattern)
    }

    /// [`compile`](Self::compile) from the dictionary arrays alone (already
    /// validated).
    pub fn compile_dict(dict_bytes: &[u8], dict_offsets: &[u32], pattern: &[u8]) -> Self {
        if pattern.is_empty() {
            return Self { inner: None };
        }
        let m = pattern.len();
        assert!(m <= MAX_PATTERN_LEN, "pattern length out of range");
        let ntokens = dict_offsets.len().saturating_sub(1);
        let delta = byte_automaton(pattern);
        let accept = m as u8;

        let token = |c: usize| &dict_bytes[dict_offsets[c] as usize..dict_offsets[c + 1] as usize];
        // Feed one token through the byte automaton from state `s`.
        let walk = |s: usize, tok: &[u8]| -> u8 {
            let mut st = s;
            for &b in tok {
                st = delta[st * 256 + b as usize] as usize;
                if st == m {
                    break; // absorbing
                }
            }
            st as u8
        };

        let mut base = vec![0u8; ntokens];
        let mut max_len = vec![0u32; 256]; // longest token per first byte
        let mut by_first: Vec<Vec<u16>> = vec![Vec::new(); 256];
        for c in 0..ntokens {
            let tok = token(c);
            base[c] = walk(0, tok);
            by_first[tok[0] as usize].push(c as u16);
            max_len[tok[0] as usize] = max_len[tok[0] as usize].max(tok.len() as u32);
        }

        // Exceptions: δ*(s, c) can differ from base[c] only when the token's
        // first byte transitions differently from `s` than from `0` (equal
        // first steps make the remaining walks identical).
        let mut exceptions: Vec<Vec<(u16, u8)>> = vec![Vec::new(); m + 1];
        for (s, exc) in exceptions.iter_mut().enumerate().take(m).skip(1) {
            for b in 0..256 {
                if delta[s * 256 + b] == delta[b] {
                    continue;
                }
                for &c in &by_first[b] {
                    let dest = walk(s, token(c as usize));
                    if dest != base[c as usize] {
                        exc.push((c, dest));
                    }
                }
            }
            exc.sort_unstable();
        }

        // Interesting set + the longest interesting token (for the run
        // lower bound).
        let words = ntokens.div_ceil(64).max(1);
        let mut set = vec![0u64; words];
        let mut lmax = 0usize;
        let mut interesting = |c: usize| {
            set[c / 64] |= 1u64 << (c % 64);
            lmax = lmax.max(token(c).len());
        };
        for (c, &b) in base.iter().enumerate() {
            if b != 0 {
                interesting(c);
            }
        }
        for exc in &exceptions {
            for &(c, _) in exc {
                interesting(c as usize);
            }
        }

        // Any occurrence is covered by consecutive interesting codes whose
        // overlaps with the pattern sum to m and are each at most lmax long,
        // so it spans at least ceil(m / lmax) of them.
        let combo = if lmax == 0 {
            1 // no interesting token: nothing can ever match
        } else {
            m.div_ceil(lmax).clamp(1, MAX_COMBO)
        };

        Self {
            inner: Some(Inner {
                base,
                exceptions,
                filter: Filter::from_bitmap(&set, ntokens),
                combo,
                accept,
                ntokens,
            }),
        }
    }

    /// Indices of the rows whose decompressed bytes contain the pattern:
    /// SIMD interesting-mask over the code stream, shifted-mask combination
    /// prefilter, then a sparse DFA walk over each surviving row.
    ///
    /// ## Panics
    ///
    /// Panics if `code_offsets` is malformed or a code is out of range for
    /// the dictionary this searcher was compiled against.
    pub fn matching_rows<O: Offset>(&self, codes: &[u16], code_offsets: &[O]) -> Vec<u64> {
        let Some(inner) = &self.inner else {
            return (0..code_offsets.len().saturating_sub(1) as u64).collect();
        };
        let combined = inner.combined_mask(codes);
        let mut out = Vec::new();
        let combo = inner.combo;
        // Density-adaptive extraction, as `ClassSearcher::matching_rows`.
        let dense = combined.iter().map(|w| w.count_ones() as usize).sum::<usize>()
            > codes.len() / 8;
        if dense {
            inner.for_each_candidate(codes, code_offsets, &combined, |r, a, b| {
                if inner.walk_row(&codes[a..b]) {
                    out.push(r);
                }
            });
        } else {
            super::for_each_candidate_row(codes, code_offsets, &combined, |r, row| {
                let a = code_offsets[r as usize].to_usize().expect("offset");
                if row.len() >= combo
                    && any_bit_in_range(&combined, a, a + row.len() - combo + 1)
                    && inner.walk_row(row)
                {
                    out.push(r);
                }
            });
        }
        out
    }

    /// Like [`matching_rows`](Self::matching_rows) with the prefilter
    /// disabled: the sparse walk visits every row. The exact baseline the
    /// combination prefilter is measured against, and the direct peer of the
    /// dense table's unfiltered scan.
    pub fn matching_rows_unfiltered<O: Offset>(
        &self,
        codes: &[u16],
        code_offsets: &[O],
    ) -> Vec<u64> {
        let Some(inner) = &self.inner else {
            return (0..code_offsets.len().saturating_sub(1) as u64).collect();
        };
        let mut out = Vec::new();
        for (r, w) in code_offsets.windows(2).enumerate() {
            let (a, b) = span(w, codes.len());
            if inner.walk_row(&codes[a..b]) {
                out.push(r as u64);
            }
        }
        out
    }

    /// Rows the combination prefilter cannot rule out (a superset of the
    /// matching rows), for false-positive measurement.
    ///
    /// ## Panics
    ///
    /// As [`matching_rows`](Self::matching_rows).
    pub fn candidate_rows<O: Offset>(&self, codes: &[u16], code_offsets: &[O]) -> Vec<u64> {
        let Some(inner) = &self.inner else {
            return (0..code_offsets.len().saturating_sub(1) as u64).collect();
        };
        let combined = inner.combined_mask(codes);
        let mut out = Vec::new();
        inner.for_each_candidate(codes, code_offsets, &combined, |r, _, _| out.push(r));
        out
    }

    /// Does one row's code slice contain the pattern? The per-row verify
    /// entry point, exposed so the verify path can be measured in isolation.
    /// The empty pattern matches every row.
    #[inline]
    pub fn row_matches(&self, codes: &[u16]) -> bool {
        match &self.inner {
            None => true,
            Some(inner) => inner.walk_row(codes),
        }
    }

    /// Compile-time shape, or `None` for the empty pattern.
    pub fn info(&self) -> Option<SparseInfo> {
        self.inner.as_ref().map(|inner| SparseInfo {
            transition_bytes: inner.base.len()
                + inner
                    .exceptions
                    .iter()
                    .map(|e| e.len() * size_of::<(u16, u8)>())
                    .sum::<usize>(),
            exceptions: inner.exceptions.iter().map(Vec::len).sum(),
            interesting: {
                let mut set = vec![false; inner.ntokens];
                for (c, &b) in inner.base.iter().enumerate() {
                    if b != 0 {
                        set[c] = true;
                    }
                }
                for exc in &inner.exceptions {
                    for &(c, _) in exc {
                        set[c as usize] = true;
                    }
                }
                set.iter().filter(|&&x| x).count()
            },
            combo: inner.combo,
        })
    }
}

impl Inner {
    /// One automaton step. State `0` reads `base` directly; other states
    /// check their (rare, usually empty) exception list first.
    #[inline]
    fn step(&self, s: usize, c: u16) -> usize {
        if s == 0 {
            return self.base[c as usize] as usize;
        }
        let exc = &self.exceptions[s];
        if exc.is_empty() {
            return self.base[c as usize] as usize;
        }
        match exc.binary_search_by_key(&c, |&(code, _)| code) {
            Ok(i) => exc[i].1 as usize,
            Err(_) => self.base[c as usize] as usize,
        }
    }

    /// Does one row's code slice reach the accept state?
    #[inline]
    fn walk_row(&self, codes: &[u16]) -> bool {
        let accept = self.accept as usize;
        let mut s = 0usize;
        for &c in codes {
            s = self.step(s, c);
            if s == accept {
                return true;
            }
        }
        false
    }

    /// Positional interesting-mask AND-ed with its shifts: bit `i` set means
    /// `codes[i..i + combo]` are all interesting — the merged-window
    /// combination the prefilter tests rows against.
    fn combined_mask(&self, codes: &[u16]) -> Vec<u64> {
        let words = codes.len().div_ceil(64);
        let mut mask = vec![0u64; words];
        self.filter.candidate_mask(codes, &mut mask);
        if words == 0 {
            return mask;
        }
        // Codes past `codes.len()` in the last word are garbage from the
        // scanner's perspective; clear them so shifts cannot smear them in.
        let tail = codes.len() % 64;
        if tail != 0 {
            mask[words - 1] &= !0u64 >> (64 - tail);
        }
        let mut combined = mask.clone();
        for shift in 1..self.combo {
            for w in 0..words {
                let lo = mask[w] >> shift;
                let hi = if w + 1 < words {
                    mask[w + 1] << (64 - shift)
                } else {
                    0
                };
                combined[w] &= lo | hi;
            }
        }
        combined
    }

    /// Hand `(row, code_start, code_end)` to `f` for every row whose span
    /// contains a fully-in-row combination window.
    #[inline]
    fn for_each_candidate<O: Offset>(
        &self,
        codes: &[u16],
        code_offsets: &[O],
        combined: &[u64],
        mut f: impl FnMut(u64, usize, usize),
    ) {
        for (r, w) in code_offsets.windows(2).enumerate() {
            let (a, b) = span(w, codes.len());
            // A window of `combo` codes must fit inside the row.
            if b - a >= self.combo && any_bit_in_range(combined, a, b - self.combo + 1) {
                f(r as u64, a, b);
            }
        }
    }
}

/// Decode one `code_offsets` window into a checked `[a, b)` span.
#[inline]
fn span<O: Offset>(w: &[O], len: usize) -> (usize, usize) {
    let a = w[0].to_usize().expect("row offset overflows usize");
    let b = w[1].to_usize().expect("row offset overflows usize");
    assert!(a <= b && b <= len, "malformed code_offsets");
    (a, b)
}

#[cfg(test)]
mod tests {
    use super::super::ContainsSearcher;
    use super::*;

    /// Hand-built dictionary: tokens laid out back to back, plus the decoder
    /// padding `validate_dictionary` requires (walks are offset-bounded and
    /// never read it).
    fn dict(tokens: &[&[u8]]) -> (Vec<u8>, Vec<u32>) {
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for t in tokens {
            bytes.extend_from_slice(t);
            offsets.push(bytes.len() as u32);
        }
        let last = offsets[offsets.len().saturating_sub(2)] as usize;
        bytes.resize(last + crate::MAX_TOKEN_SIZE, 0);
        (bytes, offsets)
    }

    fn rows(s: &SparseSearcher, codes: &[u16], offsets: &[u32]) -> Vec<u64> {
        let filtered = s.matching_rows(codes, offsets);
        let unfiltered = s.matching_rows_unfiltered(codes, offsets);
        assert_eq!(filtered, unfiltered, "prefilter changed the answer");
        let candidates = s.candidate_rows(codes, offsets);
        for r in &filtered {
            assert!(candidates.contains(r), "candidate set dropped a match");
        }
        filtered
    }

    #[test]
    fn matches_within_and_across_tokens() {
        let tokens: &[&[u8]] = &[b"ab", b"cd", b"abc", b"x"];
        let (bytes, offsets) = dict(tokens);
        let s = SparseSearcher::compile_dict(&bytes, &offsets, b"bcd");
        assert_eq!(rows(&s, &[0, 1, 1, 0, 3, 0, 1], &[0, 2, 4, 7]), vec![0, 2]);
        assert_eq!(rows(&s, &[2, 1, 2, 3, 0, 1], &[0, 2, 6]), vec![1]);
        assert!(rows(&s, &[2], &[0, 1]).is_empty());
    }

    #[test]
    fn overlapping_prefix_suffix() {
        let tokens: &[&[u8]] = &[b"ab", b"a", b"b"];
        let (bytes, offsets) = dict(tokens);
        let s = SparseSearcher::compile_dict(&bytes, &offsets, b"aba");
        assert_eq!(
            rows(&s, &[0, 1, 1, 2, 1, 0, 2, 0, 0, 1], &[0, 2, 5, 7, 10]),
            vec![0, 1, 3]
        );
    }

    #[test]
    fn empty_pattern_matches_everything() {
        let (bytes, offsets) = dict(&[b"ab"]);
        let s = SparseSearcher::compile_dict(&bytes, &offsets, b"");
        assert_eq!(s.matching_rows(&[0, 0], &[0u32, 1, 2]), vec![0, 1]);
        assert!(s.info().is_none());
    }

    #[test]
    fn impossible_pattern_matches_nothing() {
        let (bytes, offsets) = dict(&[b"ab", b"cd"]);
        let s = SparseSearcher::compile_dict(&bytes, &offsets, b"zz");
        assert!(rows(&s, &[0, 1, 0, 1], &[0, 2, 4]).is_empty());
        assert_eq!(s.info().expect("compiled").interesting, 0);
    }

    /// Deterministic pseudo-random cross-check against the dense
    /// [`ContainsSearcher`] pipeline over many dictionaries, patterns, and
    /// code streams.
    #[test]
    fn cross_check_dense() {
        let mut x = 0x9E3779B97F4A7C15u64;
        let mut rng = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let alphabet = b"abcx";
        for _ in 0..200 {
            let ntokens = 3 + (rng() % 6) as usize;
            let tokens: Vec<Vec<u8>> = (0..ntokens)
                .map(|_| {
                    let len = 1 + (rng() % 4) as usize;
                    (0..len)
                        .map(|_| alphabet[(rng() % alphabet.len() as u64) as usize])
                        .collect()
                })
                .collect();
            let refs: Vec<&[u8]> = tokens.iter().map(Vec::as_slice).collect();
            let (bytes, offsets) = dict(&refs);

            let plen = 1 + (rng() % 6) as usize;
            let pattern: Vec<u8> = (0..plen)
                .map(|_| alphabet[(rng() % alphabet.len() as u64) as usize])
                .collect();

            let mut codes = Vec::new();
            let mut code_offsets = vec![0u32];
            for _ in 0..20 {
                for _ in 0..(rng() % 8) {
                    codes.push((rng() % ntokens as u64) as u16);
                }
                code_offsets.push(codes.len() as u32);
            }

            let sparse = SparseSearcher::compile_dict(&bytes, &offsets, &pattern);
            let dense = ContainsSearcher::compile_heuristic(&bytes, &offsets, &pattern);
            let expect = dense.matching_rows_unfiltered(&codes, &code_offsets);
            assert_eq!(
                rows(&sparse, &codes, &code_offsets),
                expect,
                "pattern {:?} tokens {:?}",
                String::from_utf8_lossy(&pattern),
                tokens
                    .iter()
                    .map(|t| String::from_utf8_lossy(t).into_owned())
                    .collect::<Vec<_>>(),
            );
        }
    }
}
