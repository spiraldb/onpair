// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compressed-domain prefix / substring search.
//!
//! Rust port of the token-level search automata in the reference C++
//! implementation (`include/onpair/search/automata/*`). The central idea: a
//! column's bytes are encoded as a stream of dictionary token ids, so instead
//! of decompressing each row and running a byte matcher, we run a small
//! deterministic automaton **directly over the token ids**. Every input byte
//! becomes part of one token, so a `T`-token row costs `T` automaton steps
//! regardless of how many bytes it decodes to — and matches early-exit.
//!
//! Two predicates are supported, expressed as [`Pattern`]:
//!   * [`Pattern::Prefix`] — `col LIKE 'needle%'`, via [`prefix::PrefixAutomaton`].
//!   * [`Pattern::Contains`] — `col LIKE '%needle%'`, via [`kmp::KmpAutomaton`].
//!
//! Both automata are built once per query against the (sorted) dictionary and
//! then driven over every row. Construction relies on two dictionary
//! properties guaranteed by [`crate::Parser::train`]: the token ids are in
//! lexicographic order, and the 256 single-byte tokens are always present.

mod kmp;
mod prefix;
mod tokenize;

use crate::column::Column;
use crate::offset::Offset;
use crate::types::{MAX_TOKEN_SIZE, Token};

use kmp::KmpAutomaton;
use prefix::PrefixAutomaton;

/// A search predicate evaluated against every row of a compressed column,
/// without decompressing it. Borrows the needle bytes for the duration of the
/// search.
#[derive(Copy, Clone, Debug)]
pub enum Pattern<'a> {
    /// Matches rows whose decoded bytes begin with the needle
    /// (SQL `col LIKE 'needle%'`).
    Prefix(&'a [u8]),
    /// Matches rows whose decoded bytes contain the needle anywhere
    /// (SQL `col LIKE '%needle%'`).
    Contains(&'a [u8]),
}

// ─────────────────────────────────────────────────────────────────────────────
// TokenRange — closed range of token ids [begin, last]; begin > last is empty.
// ─────────────────────────────────────────────────────────────────────────────

/// Closed range of token ids `[begin, last]`. The default-constructed
/// `{ begin: 1, last: 0 }` is the canonical empty range.
#[derive(Copy, Clone, Debug)]
pub(crate) struct TokenRange {
    pub(crate) begin: Token,
    pub(crate) last: Token,
}

impl TokenRange {
    /// Canonical empty range (`begin > last`).
    pub(crate) const EMPTY: Self = Self { begin: 1, last: 0 };

    #[inline]
    pub(crate) fn empty(self) -> bool {
        self.begin > self.last
    }

    #[inline]
    pub(crate) fn contains(self, t: Token) -> bool {
        t >= self.begin && t <= self.last
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DictView — borrowed, read-only view over a column's sorted dictionary.
// ─────────────────────────────────────────────────────────────────────────────

/// Borrowed view over the `(bytes, offsets)` of a sorted dictionary. Mirrors
/// the C++ `DictionaryView`: O(1) token access plus O(log n) prefix-range
/// lookups via binary search over the sorted token ids.
#[derive(Copy, Clone)]
pub(crate) struct DictView<'a> {
    bytes: &'a [u8],
    offsets: &'a [u32],
}

impl<'a> DictView<'a> {
    #[inline]
    fn num_tokens(self) -> usize {
        self.offsets.len() - 1
    }

    #[inline]
    fn token_size(self, id: Token) -> usize {
        (self.offsets[id as usize + 1] - self.offsets[id as usize]) as usize
    }

    #[inline]
    fn data(self, id: Token) -> &'a [u8] {
        let s = self.offsets[id as usize] as usize;
        let e = self.offsets[id as usize + 1] as usize;
        &self.bytes[s..e]
    }

    /// First token id in `[start, num_tokens)` whose bytes are `>= target`
    /// under the dictionary's sort order (shorter token sorts before a longer
    /// one sharing its prefix). Direct port of the C++ `lower_bound` lambda.
    fn lower_bound(self, target: &[u8], start: u32) -> u32 {
        let n = self.num_tokens() as u32;
        let (mut lo, mut hi) = (start, n);
        while lo < hi {
            let mid = lo + ((hi - lo) >> 1);
            let tok = self.data(mid as Token);
            let mlen = tok.len();
            let clen = mlen.min(target.len());
            let cmp = tok[..clen].cmp(&target[..clen]);
            // token[mid] < target iff cmp < 0, or equal-prefix and token shorter.
            if cmp.is_lt() || (cmp.is_eq() && mlen < target.len()) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// `[lo, hi]` token-id range whose byte sequences share `prefix`, or the
    /// empty range if none do. Port of `DictionaryView::prefix_range`.
    fn prefix_range(self, prefix: &[u8]) -> TokenRange {
        // A prefix longer than any token can never match.
        if prefix.len() > MAX_TOKEN_SIZE {
            return TokenRange::EMPTY;
        }
        let n = self.num_tokens() as u32;

        let lo = self.lower_bound(prefix, 0);

        // Next lexicographic prefix: increment the last non-0xFF byte after
        // trimming trailing 0xFF bytes. If all bytes are 0xFF the prefix has no
        // successor, so the range runs to the end of the dictionary.
        let mut buf = [0u8; MAX_TOKEN_SIZE];
        let mut ulen = prefix.len();
        let mut overflow = true;
        while ulen > 0 {
            if prefix[ulen - 1] < 0xFF {
                buf[..ulen].copy_from_slice(&prefix[..ulen]);
                buf[ulen - 1] += 1;
                overflow = false;
                break;
            }
            ulen -= 1;
        }

        // hi >= lo always, so the second search starts from lo, not 0.
        let hi = if overflow {
            n
        } else {
            self.lower_bound(&buf[..ulen], lo)
        };

        if lo < hi {
            TokenRange {
                begin: lo as Token,
                last: (hi - 1) as Token,
            }
        } else {
            TokenRange::EMPTY
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Automaton driver.
// ─────────────────────────────────────────────────────────────────────────────

/// Any type that can be driven token-by-token to detect a match within one
/// row. Mirrors the C++ `TokenAutomaton` + `DeadDetectable` concepts: the
/// driver feeds tokens until the row ends or [`is_dead`](Self::is_dead) reports
/// the verdict can no longer change, then reads [`is_accepted`](Self::is_accepted).
pub(crate) trait TokenAutomaton {
    /// Rewind to the start state for a fresh row.
    fn reset(&mut self);
    /// Consume one token.
    fn step(&mut self, t: Token);
    /// Final verdict (only meaningful once the row is exhausted or dead).
    fn is_accepted(&self) -> bool;
    /// True once further tokens cannot change the verdict.
    fn is_dead(&self) -> bool;
}

/// Drive `aut` over one row's tokens, early-exiting on death.
#[inline]
fn drive(aut: &mut impl TokenAutomaton, codes: &[Token]) -> bool {
    aut.reset();
    for &t in codes {
        aut.step(t);
        if aut.is_dead() {
            break;
        }
    }
    aut.is_accepted()
}

/// Drive `aut` over every row delimited by `code_offsets`, invoking `on_match`
/// with the row index of each accepting row.
#[inline]
fn scan<O: Offset>(
    aut: &mut impl TokenAutomaton,
    codes: &[Token],
    code_offsets: &[O],
    mut on_match: impl FnMut(usize),
) {
    for r in 0..code_offsets.len() - 1 {
        let s = code_offsets[r].to_usize().expect("valid code offsets");
        let e = code_offsets[r + 1].to_usize().expect("valid code offsets");
        if drive(aut, &codes[s..e]) {
            on_match(r);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry points.
// ─────────────────────────────────────────────────────────────────────────────

impl<O: Offset> Column<O> {
    /// Evaluate `pattern` against every row, invoking `on_match` with the
    /// 0-based index of each matching row, in order. The match is computed in
    /// the compressed domain — rows are never decompressed.
    pub fn search_for_each(&self, pattern: Pattern<'_>, on_match: impl FnMut(usize)) {
        let dict = DictView {
            bytes: &self.dict_bytes,
            offsets: &self.dict_offsets,
        };
        match pattern {
            Pattern::Contains(needle) => {
                let mut aut = KmpAutomaton::new(needle, dict);
                scan(&mut aut, &self.codes, &self.code_offsets, on_match);
            }
            Pattern::Prefix(needle) => {
                let mut aut = PrefixAutomaton::new(needle, dict);
                scan(&mut aut, &self.codes, &self.code_offsets, on_match);
            }
        }
    }

    /// Evaluate `pattern` against every row and collect the indices of the
    /// matching rows. Convenience wrapper over [`search_for_each`].
    ///
    /// [`search_for_each`]: Self::search_for_each
    pub fn search(&self, pattern: Pattern<'_>) -> Vec<usize> {
        let mut out = Vec::new();
        self.search_for_each(pattern, |r| out.push(r));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Bits, Config, Threshold, compress};

    /// Pack rows into the Arrow `(bytes, offsets)` pair `compress` expects.
    fn pack(rows: &[&[u8]]) -> (Vec<u8>, Vec<u32>) {
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for r in rows {
            bytes.extend_from_slice(r);
            offsets.push(bytes.len() as u32);
        }
        (bytes, offsets)
    }

    fn cfg() -> Config {
        Config {
            bits: Bits::new(12).unwrap(),
            threshold: Threshold::new(0.5).unwrap(),
            seed: Some(42),
        }
    }

    fn naive_contains(row: &[u8], needle: &[u8]) -> bool {
        needle.is_empty() || row.windows(needle.len()).any(|w| w == needle)
    }

    fn assert_matches(rows: &[&[u8]], pattern: Pattern<'_>, expect: impl Fn(&[u8]) -> bool) {
        let (bytes, offsets) = pack(rows);
        let col = compress(&bytes, &offsets, cfg()).unwrap();
        let got = col.search(pattern);
        let want: Vec<usize> = rows
            .iter()
            .enumerate()
            .filter_map(|(i, r)| expect(r).then_some(i))
            .collect();
        assert_eq!(got, want, "pattern {pattern:?}");
    }

    /// A corpus with heavy prefix sharing and repeated substrings so the
    /// trainer emits multi-byte tokens (exercising the sparse KMP transitions
    /// and prefix-divergence intervals rather than only single-byte tokens).
    fn url_corpus() -> Vec<Vec<u8>> {
        let hosts = ["https://www.example.com", "https://api.example.org", "ftp://x.example.net"];
        let paths = ["/index.html", "/search?q=onpair", "/a/b/c", "", "/login"];
        let mut out = Vec::new();
        let mut x = 0x1234_5678u64;
        for _ in 0..2000 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            let h = hosts[(x >> 33) as usize % hosts.len()];
            let p = paths[(x >> 17) as usize % paths.len()];
            out.push(format!("{h}{p}{}", x % 100).into_bytes());
        }
        out
    }

    #[test]
    fn contains_matches_naive_across_needles() {
        let owned = url_corpus();
        let rows: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();
        for needle in [
            b"example".as_slice(),
            b"https://".as_slice(),
            b"search?q=onpair".as_slice(),
            b"/a/b/c".as_slice(),
            b"zzz-not-present".as_slice(),
            b"e".as_slice(),
            b"".as_slice(),
        ] {
            assert_matches(&rows, Pattern::Contains(needle), |r| naive_contains(r, needle));
        }
    }

    #[test]
    fn prefix_matches_naive_across_needles() {
        let owned = url_corpus();
        let rows: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();
        for needle in [
            b"https://".as_slice(),
            b"https://www.example.com".as_slice(),
            b"ftp://".as_slice(),
            b"https://api.example.org/login".as_slice(),
            b"nope".as_slice(),
            b"".as_slice(),
        ] {
            assert_matches(&rows, Pattern::Prefix(needle), |r| r.starts_with(needle));
        }
    }

    #[test]
    fn single_byte_needles() {
        let rows: &[&[u8]] = &[b"abc", b"xyz", b"a", b"", b"cba"];
        for b in [b"a".as_slice(), b"z".as_slice(), b"q".as_slice()] {
            assert_matches(rows, Pattern::Contains(b), |r| naive_contains(r, b));
            assert_matches(rows, Pattern::Prefix(b), |r| r.starts_with(b));
        }
    }

    #[test]
    fn needle_longer_than_any_token() {
        // A 20-byte needle exceeds MAX_TOKEN_SIZE; prefix_range short-circuits.
        let rows: &[&[u8]] = &[b"this is a fairly long row of text", b"short"];
        let needle = b"fairly long row of t"; // 20 bytes
        assert_matches(rows, Pattern::Contains(needle), |r| naive_contains(r, needle));
        let pneedle = b"this is a fairly lon"; // 20 bytes
        assert_matches(rows, Pattern::Prefix(pneedle), |r| r.starts_with(pneedle));
    }
}
