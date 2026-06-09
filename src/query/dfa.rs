// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Substring DFA lifted from bytes to dictionary codes.
//!
//! A classic KMP matching automaton over the pattern's bytes is precomputed,
//! then composed with every dictionary token to yield a single transition
//! table indexed by `(state, code)`. Scanning a row is then one table load per
//! code — the row's bytes are never materialized, and a match that straddles
//! token boundaries is found because the automaton state carries the partial
//! match across codes.

/// Matching automaton over dictionary codes. State `s` means "the longest
/// prefix of the pattern that is a suffix of the bytes read so far has length
/// `s`"; the accept state (`s == pattern.len()`) is absorbing.
pub(crate) struct TokenDfa {
    /// `(m + 1) * ntokens` transitions, state-major: `table[s * ntokens + c]`.
    table: Vec<u8>,
    ntokens: usize,
    accept: u8,
}

/// Largest supported pattern length: states are stored as `u8` and one state
/// per matched prefix length is needed, plus the empty prefix.
pub const MAX_PATTERN_LEN: usize = u8::MAX as usize;

impl TokenDfa {
    /// Build the `(state, code)` transition table for `pattern` over the
    /// dictionary described by `dict_bytes` / `dict_offsets`.
    ///
    /// Requires `1 <= pattern.len() <= MAX_PATTERN_LEN` and a validated
    /// dictionary (see [`crate::Parts::validate_dictionary`]).
    pub(crate) fn build(pattern: &[u8], dict_bytes: &[u8], dict_offsets: &[u32]) -> Self {
        let m = pattern.len();
        assert!(
            (1..=MAX_PATTERN_LEN).contains(&m),
            "pattern length out of range"
        );
        let ntokens = dict_offsets.len().saturating_sub(1);

        // KMP matching automaton over bytes: delta[s][b] = longest prefix of
        // the pattern that suffixes (matched prefix of length s) + b.
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

        // Compose with each token: T[s][c] = delta* (s, token_bytes(c)).
        let mut table = vec![0u8; (m + 1) * ntokens];
        for s in 0..=m {
            let out = &mut table[s * ntokens..(s + 1) * ntokens];
            for (c, out_c) in out.iter_mut().enumerate() {
                let tok = &dict_bytes[dict_offsets[c] as usize..dict_offsets[c + 1] as usize];
                let mut st = s;
                for &b in tok {
                    st = delta[st * 256 + b as usize] as usize;
                    if st == m {
                        break; // absorbing
                    }
                }
                *out_c = st as u8;
            }
        }

        Self {
            table,
            ntokens,
            accept: m as u8,
        }
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
            s = self.table[s * self.ntokens + c as usize] as usize;
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
