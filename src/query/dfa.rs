// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Substring DFA lifted from bytes to dictionary codes.
//!
//! A classic KMP matching automaton over the pattern's bytes is precomputed;
//! its composition with each dictionary token — the `(state, code)` table the
//! scan reads — is filled **lazily**, one entry on first touch. A `LIKE`-style
//! scan compiles a searcher per row group, and behind the prefilter the DFA
//! only ever visits the (few) candidate rows, so eagerly composing all
//! `(m + 1) × ntokens` entries (megabytes of token walks) would dominate the
//! whole query; lazy composition makes compile O(pattern) while the scan pays
//! one short token walk per *distinct* `(state, code)` pair it actually
//! reaches. Scanning a row stays one table load per code, and a match that
//! straddles token boundaries is found because the automaton state carries
//! the partial match across codes.

use std::sync::atomic::AtomicU8;
use std::sync::atomic::Ordering::Relaxed;

/// Matching automaton over dictionary codes. State `s` means "the longest
/// prefix of the pattern that is a suffix of the bytes read so far has length
/// `s`"; the accept state (`s == pattern.len()`) is absorbing.
pub(crate) struct TokenDfa {
    /// `(m + 1) * 256` byte-level transitions, state-major.
    delta: Vec<u8>,
    /// `(m + 1) * ntokens` composed transitions, state-major
    /// (`table[s * ntokens + c]`), filled on demand; [`UNFILLED`] marks
    /// untouched entries. Concurrent fills race benignly: every thread
    /// computes the same value.
    table: Vec<AtomicU8>,
    /// Owned copy of the dictionary, for on-demand composition. Keeps the
    /// searcher free of borrowed lifetimes and usable across code streams.
    dict_bytes: Vec<u8>,
    dict_offsets: Vec<u32>,
    ntokens: usize,
    accept: u8,
}

/// Sentinel for a not-yet-composed table entry; never a valid state because
/// states span `0..=pattern.len() <= MAX_PATTERN_LEN < 0xFF`.
const UNFILLED: u8 = 0xFF;

/// Largest supported pattern length: states are stored as `u8`, one per
/// matched prefix length plus the empty prefix, and `0xFF` is reserved as the
/// lazy-fill sentinel.
pub const MAX_PATTERN_LEN: usize = u8::MAX as usize - 1;

/// KMP matching automaton over the pattern's bytes, state-major:
/// `delta[s * 256 + b]` is the length of the longest pattern prefix that
/// suffixes (matched prefix of length `s`) + `b`. The accept state
/// (`s == pattern.len()`) is absorbing.
///
/// Requires `1 <= pattern.len() <= MAX_PATTERN_LEN`.
pub(super) fn byte_automaton(pattern: &[u8]) -> Vec<u8> {
    let m = pattern.len();
    assert!(
        (1..=MAX_PATTERN_LEN).contains(&m),
        "pattern length out of range"
    );
    let mut delta = vec![0u8; (m + 1) * 256];
    delta[pattern[0] as usize] = 1;
    let mut fail = 0usize; // automaton state after reading pattern[1..s]
    for s in 1..m {
        let (head, row) = delta.split_at_mut(s * 256);
        row[..256].copy_from_slice(&head[fail * 256..fail * 256 + 256]);
        row[pattern[s] as usize] = (s + 1) as u8;
        fail = head[fail * 256 + pattern[s] as usize] as usize;
    }
    // Accept state is absorbing: a row matches if any prefix of it does.
    for b in 0..256 {
        delta[m * 256 + b] = m as u8;
    }
    delta
}

impl TokenDfa {
    /// Build the byte automaton for `pattern` over the dictionary described
    /// by `dict_bytes` / `dict_offsets`; the token-level table fills lazily
    /// during scans.
    ///
    /// Requires `1 <= pattern.len() <= MAX_PATTERN_LEN` and a validated
    /// dictionary (see [`crate::Parts::validate_dictionary`]).
    pub(crate) fn build(pattern: &[u8], dict_bytes: &[u8], dict_offsets: &[u32]) -> Self {
        let m = pattern.len();
        let ntokens = dict_offsets.len().saturating_sub(1);
        let delta = byte_automaton(pattern);

        let table = (0..(m + 1) * ntokens)
            .map(|_| AtomicU8::new(UNFILLED))
            .collect();

        Self {
            delta,
            table,
            dict_bytes: dict_bytes.to_vec(),
            dict_offsets: dict_offsets.to_vec(),
            ntokens,
            accept: m as u8,
        }
    }

    /// Compose the byte automaton over token `c`'s bytes starting from state
    /// `s`. Cold: each `(state, code)` pair pays this at most once.
    #[cold]
    fn compose(&self, s: usize, c: usize) -> u8 {
        let tok =
            &self.dict_bytes[self.dict_offsets[c] as usize..self.dict_offsets[c + 1] as usize];
        let m = self.accept as usize;
        let mut st = s;
        for &b in tok {
            st = self.delta[st * 256 + b as usize] as usize;
            if st == m {
                break; // absorbing
            }
        }
        st as u8
    }

    /// Run the automaton over one row's codes. Returns `true` as soon as the
    /// accept state is reached.
    ///
    /// Panics (via slice indexing) if a code is out of range for the
    /// dictionary the automaton was built from.
    #[inline]
    pub(crate) fn row_matches(&self, codes: &[u16]) -> bool {
        let mut s = 0usize;
        for &c in codes {
            let idx = s * self.ntokens + c as usize;
            let mut t = self.table[idx].load(Relaxed);
            if t == UNFILLED {
                t = self.compose(s, c as usize);
                self.table[idx].store(t, Relaxed);
            }
            s = t as usize;
            if s == self.accept as usize {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-built dictionary: tokens are byte strings laid out back to back.
    fn dict(tokens: &[&[u8]]) -> (Vec<u8>, Vec<u32>) {
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for t in tokens {
            bytes.extend_from_slice(t);
            offsets.push(bytes.len() as u32);
        }
        (bytes, offsets)
    }

    #[test]
    fn matches_within_and_across_tokens() {
        let tokens: &[&[u8]] = &[b"ab", b"cd", b"abc", b"x"];
        let (bytes, offsets) = dict(tokens);

        // "bcd" never appears inside a single token; only across 0→1 ("ab"+"cd").
        let dfa = TokenDfa::build(b"bcd", &bytes, &offsets);
        assert!(dfa.row_matches(&[0, 1])); // "abcd" contains "bcd"
        assert!(!dfa.row_matches(&[1, 0])); // "cdab"
        assert!(dfa.row_matches(&[3, 0, 1])); // "xabcd"
        assert!(!dfa.row_matches(&[2])); // "abc"
        assert!(!dfa.row_matches(&[2, 1])); // "abccd"
        assert!(dfa.row_matches(&[2, 3, 0, 1])); // "abcxabcd"
    }

    #[test]
    fn within_token_match() {
        let tokens: &[&[u8]] = &[b"hello world", b"z"];
        let (bytes, offsets) = dict(tokens);
        let dfa = TokenDfa::build(b"lo wo", &bytes, &offsets);
        assert!(dfa.row_matches(&[0]));
        assert!(!dfa.row_matches(&[1]));
    }

    #[test]
    fn overlapping_prefix_suffix() {
        // Pattern with a border ("aba") — exercises the failure function.
        let tokens: &[&[u8]] = &[b"ab", b"a", b"b"];
        let (bytes, offsets) = dict(tokens);
        let dfa = TokenDfa::build(b"aba", &bytes, &offsets);
        assert!(dfa.row_matches(&[0, 1])); // "ab"+"a" = "aba"
        assert!(dfa.row_matches(&[1, 2, 1])); // "a"+"b"+"a"
        assert!(!dfa.row_matches(&[0, 2])); // "abb"
        assert!(dfa.row_matches(&[0, 0, 1])); // "ababa"
    }
}
