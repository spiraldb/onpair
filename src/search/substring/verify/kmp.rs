// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Search one encoded row using token transitions derived from byte-level KMP.
//!
//! Preparation simulates each token's bytes and stores the resulting transitions
//! in an immutable `ContainsDfa`. `row_contains` then advances one token at a
//! time using local state, without reading the dictionary or decoding the row.
//!
//! The table stores a base transition per token, computed from state zero.
//! Other states store only the token ranges whose transitions differ from that
//! base. A lookup searches those ranges before falling back to the base entry.
//!
//! This path searches the entire row independently of the bulk scan's cover
//! and graph walker. KMP failure links preserve overlapping partial matches;
//! they are folded into token transitions during preparation.

use crate::core::dictionary::DictionaryView;
use crate::core::types::{Token, TokenRange};
use crate::search::lookup::prefix_range;
use crate::search::substring::ContainsError;

/// Number of leading pattern bytes matched, including the accepting length.
/// The `u8` representation limits patterns to 255 bytes.
type State = u8;

/// Immutable token-transition DFA for one pattern and dictionary.
/// Execution state lives in `row_contains`, so the prepared tables can be
/// reused across rows and shared between concurrent calls.
#[derive(Debug, Clone)]
pub struct ContainsDfa {
    /// Accepting state equal to the pattern length; zero for an empty pattern.
    accept: State,
    /// `base[token]` = KMP state after running `token`'s bytes from state 0.
    base: Vec<State>,
    /// Per-state exceptions: for entry state `s`, the tokens in
    /// `sparse[offsets[s]..offsets[s + 1]]` transition to their `target` instead
    /// of `base[token]`. Ranges within a state are ascending and disjoint.
    sparse: Vec<SparseTransition>,
    /// Offsets delimiting each non-accepting state's exception slice.
    offsets: Vec<u32>,
}

/// Token range that overrides the base transition for one entry state.
#[derive(Debug, Clone, Copy)]
struct SparseTransition {
    range: TokenRange,
    target: State,
}

impl ContainsDfa {
    /// Maximum pattern length, bounded by the DFA's `u8` state IDs.
    pub const MAX_PATTERN_LEN: usize = State::MAX as usize;

    /// Compile token transitions for `pattern` using a sorted dictionary.
    /// An empty pattern prepares a DFA that accepts every row, including empty rows.
    ///
    /// # Errors
    /// Returns [`ContainsError::PatternTooLong`] if the pattern exceeds
    /// [`Self::MAX_PATTERN_LEN`].
    pub fn new<V: DictionaryView>(pattern: &[u8], dict: V) -> Result<Self, ContainsError> {
        if pattern.len() > Self::MAX_PATTERN_LEN {
            return Err(ContainsError::PatternTooLong {
                length: pattern.len(),
                max: Self::MAX_PATTERN_LEN,
            });
        }
        let m = pattern.len();
        let num_tokens = dict.num_tokens();

        // Empty pattern: accept state 0, every token a no-op transition.
        if m == 0 {
            return Ok(Self {
                accept: 0,
                base: vec![0; num_tokens],
                sparse: Vec::new(),
                offsets: vec![0, 0],
            });
        }

        let mut build = Build {
            dict,
            p: pattern,
            m,
            fail: kmp_failure(pattern),
            base: vec![0; num_tokens],
            sparse: Vec::new(),
            offsets: vec![0u32; m + 1],
            range_start: 0,
        };
        build.base_pass();
        build.sparse_pass();

        Ok(Self {
            accept: m as State,
            base: build.base,
            sparse: build.sparse,
            offsets: build.offsets,
        })
    }

    /// Apply a token transition, checking this state's exceptions before the base.
    /// Requires a valid dictionary token ID and `state < accept`; `row_contains`
    /// stops at acceptance before another transition is needed.
    #[inline]
    fn next(&self, state: State, token: Token) -> State {
        if state > 0 {
            let lo = self.offsets[state as usize] as usize;
            let hi = self.offsets[state as usize + 1] as usize;
            // Sorted disjoint ranges let the search stop after passing the token.
            for tr in &self.sparse[lo..hi] {
                if token < tr.range.begin {
                    break;
                }
                if token <= tr.range.last {
                    return tr.target;
                }
            }
        }
        self.base[token as usize]
    }
}

/// Whether the encoded row contains the prepared pattern.
/// State starts at zero for each call. Token transitions account for matches
/// inside tokens and across token boundaries; the first acceptance ends the scan.
/// An empty pattern matches every row.
///
/// # Precondition
/// Every code is a valid token ID from the dictionary used to prepare `dfa`.
pub fn row_contains(codes: &[Token], dfa: &ContainsDfa) -> bool {
    let mut state: State = 0;
    if state == dfa.accept {
        return true; // empty pattern is a substring of everything
    }
    for &code in codes {
        state = dfa.next(state, code);
        if state == dfa.accept {
            return true;
        }
    }
    false
}

/// Byte-level KMP failure table: `fail[i]` is the length of the longest proper
/// prefix of `pattern[..=i]` that is also a suffix.
fn kmp_failure(pattern: &[u8]) -> Vec<State> {
    let m = pattern.len();
    let mut fail = vec![0 as State; m];
    let (mut i, mut len) = (1usize, 0usize);
    while i < m {
        if pattern[i] == pattern[len] {
            len += 1;
            fail[i] = len as State;
            i += 1;
        } else if len > 0 {
            len = fail[len - 1] as usize;
        } else {
            fail[i] = 0;
            i += 1;
        }
    }
    fail
}

/// Temporary KMP simulation state used to build base and exceptional transitions.
struct Build<'a, V> {
    dict: V,
    /// Pattern bytes.
    p: &'a [u8],
    /// Pattern length and accepting state.
    m: usize,
    /// Byte-level failure links for falling back to shorter matched prefixes.
    fail: Vec<State>,
    base: Vec<State>,
    sparse: Vec<SparseTransition>,
    offsets: Vec<u32>,
    /// First exception for the current entry state, bounding adjacent-range merging.
    range_start: usize,
}

impl<V: DictionaryView> Build<'_, V> {
    /// Simulate byte-level KMP from state `s`, retaining acceptance once reached.
    fn step_bytes(&self, mut s: State, data: &[u8]) -> State {
        for &b in data {
            if s as usize == self.m {
                return self.m as State;
            }
            while s > 0 && self.p[s as usize] != b {
                s = self.fail[s as usize - 1];
            }
            if self.p[s as usize] == b {
                s += 1;
            }
        }
        s
    }

    /// Compute the state reached from zero after each complete dictionary token.
    fn base_pass(&mut self) {
        let p0 = self.p[0];
        for t in 0..self.base.len() {
            // A token without the pattern's first byte cannot leave state 0.
            let s = {
                let tok = self.dict.token(t as Token);
                if tok.contains(&p0) {
                    self.step_bytes(0, tok)
                } else {
                    0
                }
            };
            self.base[t] = s;
        }
    }

    /// Append `(range, target)`, extending the previous range if it is adjacent
    /// and shares the target. Only merges within the current entry state.
    fn emit(&mut self, range: TokenRange, target: State) {
        if self.sparse.len() > self.range_start
            && let Some(last) = self.sparse.last_mut()
            && last.target == target
            && last.range.last as usize + 1 == range.begin as usize
        {
            last.range.last = range.last;
            return;
        }
        self.sparse.push(SparseTransition { range, target });
    }

    /// Record exceptions to the base table for every nonzero entry state.
    /// Only token prefixes that can advance that state or one of its failure
    /// states need traversal.
    fn sparse_pass(&mut self) {
        let mut relevant: Vec<u8> = Vec::new();
        for j in 1..self.m {
            self.range_start = self.sparse.len();
            self.offsets[j] = self.range_start as u32;

            // Only bytes p[s] along the failure chain j → fail[j-1] → … → 0 can
            // make state j transition differently from state 0; skip the rest.
            relevant.clear();
            let mut s = j as State;
            while s > 0 {
                relevant.push(self.p[s as usize]);
                s = self.fail[s as usize - 1];
            }
            relevant.sort_unstable();
            relevant.dedup();

            for &byte in &relevant {
                let range = prefix_range(self.dict, &[byte]);
                if range.is_empty() {
                    continue;
                }
                let kj = self.step_bytes(j as State, &[byte]);
                let k0 = self.step_bytes(0, &[byte]);
                self.traverse(range, 1, kj, k0);
            }
        }
        self.offsets[self.m] = self.sparse.len() as u32;
    }

    /// Visit tokens sharing a prefix and compare two evolving KMP states.
    /// `kmp_j` starts from the current entry state; `kmp_0` starts from zero.
    /// Equal states need no exception and stop traversal of that subtree.
    fn traverse(&mut self, tr: TokenRange, depth: usize, kmp_j: State, kmp_0: State) {
        if kmp_j == kmp_0 || tr.is_empty() {
            return;
        }
        let (begin, last) = (tr.begin as usize, tr.last as usize);

        // Acceptance survives the remaining token bytes. Record only tokens
        // whose complete base transition does not already accept.
        if kmp_j as usize == self.m {
            let exit = self.m as State;
            let mut i = begin;
            while i <= last {
                if self.base[i] != exit {
                    let start = i;
                    while i <= last && self.base[i] != exit {
                        i += 1;
                    }
                    self.emit(
                        TokenRange {
                            begin: start as Token,
                            last: (i - 1) as Token,
                        },
                        exit,
                    );
                } else {
                    i += 1;
                }
            }
            return;
        }

        // Tokens of length == depth end here; they all exit at kmp_j.
        let mut cur = begin;
        while cur <= last && self.dict.token_len(cur as Token) == depth {
            cur += 1;
        }
        if cur > begin {
            self.emit(
                TokenRange {
                    begin: begin as Token,
                    last: (cur - 1) as Token,
                },
                kmp_j,
            );
        }
        if cur > last {
            return;
        }

        // Partition the remaining (longer) tokens by their byte at `depth` and
        // recurse into each subtree.
        while cur <= last {
            let c = self.dict.token(cur as Token)[depth];
            let mut sub_hi = cur;
            while sub_hi < last && self.dict.token((sub_hi + 1) as Token)[depth] == c {
                sub_hi += 1;
            }
            let kj = self.step_bytes(kmp_j, &[c]);
            let k0 = self.step_bytes(kmp_0, &[c]);
            self.traverse(
                TokenRange {
                    begin: cur as Token,
                    last: sub_hi as Token,
                },
                depth + 1,
                kj,
                k0,
            );
            cur = sub_hi + 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dictionary::Dictionary;
    use crate::{Column, DEFAULT_CONFIG, compress};

    /// Compress byte rows for the DFA fixtures.
    fn compress_rows(rows: &[&[u8]]) -> Column<u32> {
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for r in rows {
            bytes.extend_from_slice(r);
            offsets.push(bytes.len() as u32);
        }
        compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap()
    }

    /// Independent byte-window oracle, including empty-pattern matches.
    fn byte_contains(hay: &[u8], needle: &[u8]) -> bool {
        needle.is_empty() || hay.windows(needle.len()).any(|w| w == needle)
    }

    /// Decode one fixture row into bytes for the containment oracle.
    fn decode_row(view: crate::ColumnView<'_, u32>, k: usize) -> Vec<u8> {
        let mut buf =
            vec![std::mem::MaybeUninit::uninit(); view.row_decoded_len(k) + crate::DECODE_PADDING];
        // SAFETY: buffer sized for row `k`; view from a trusted column.
        let w = unsafe { view.decompress_row_into(k, &mut buf) };
        unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), w) }.to_vec()
    }

    /// Compare row search with byte containment using compact and wide dictionaries.
    fn check(rows: &[&[u8]], patterns: &[&[u8]]) {
        let col = compress_rows(rows);
        let view = col.view();
        let wide = view.dict.to_wide();
        for &pat in patterns {
            let want: Vec<usize> = (0..view.num_rows())
                .filter(|&k| byte_contains(&decode_row(view, k), pat))
                .collect();

            for dfa in [
                ContainsDfa::new(pat, view.dict).unwrap(),
                ContainsDfa::new(pat, wide.as_view()).unwrap(),
            ] {
                let got: Vec<usize> = (0..view.num_rows())
                    .filter(|&k| row_contains(view.row_codes(k), &dfa))
                    .collect();
                assert_eq!(got, want, "pattern {pat:?}");
            }
        }
    }

    #[test]
    fn empty_pattern_matches_all_rows() {
        let rows: &[&[u8]] = &[b"a", b"", b"abc"];
        check(rows, &[b""]);
    }

    #[test]
    fn single_and_multi_token_substrings() {
        let rows: &[&[u8]] = &[b"hello world", b"world peace", b"helloworld", b"hell"];
        check(
            rows,
            &[
                b"hello",
                b"world",
                b"o w",
                b"llowo",
                b"xyz",
                b"hello world",
                b"hello world!",
            ],
        );
    }

    #[test]
    fn substrings_spanning_token_boundaries() {
        // Repetitive corpus → multi-byte tokens; patterns chosen to straddle them.
        let rows: &[&[u8]] = &[b"abcabcabc", b"xabcabcy", b"ababab", b"cab"];
        check(
            rows,
            &[b"abc", b"bca", b"cab", b"bcabca", b"abcabcabc", b"ba"],
        );
    }

    #[test]
    fn repeating_pattern_exercises_failure_links() {
        // Patterns with internal repetition stress the KMP failure function and
        // the sparse cross-token transitions.
        let rows: &[&[u8]] = &[b"aaaaab", b"aabaab", b"ababab", b"aaa"];
        check(rows, &[b"aa", b"aaa", b"aab", b"abab", b"aaaa", b"aabaa"]);
    }

    #[test]
    fn matches_brute_force_on_repetitive_corpus() {
        use crate::test_corpus::user_strings;
        let corpus: Vec<Vec<u8>> = user_strings(50)
            .into_iter()
            .map(String::into_bytes)
            .collect();
        let rows: Vec<&[u8]> = corpus.iter().map(Vec::as_slice).collect();
        check(
            &rows,
            &[
                b"example", b"https", b"://", b".com", b"/page", b"ftp", b"zzz", b"w",
            ],
        );
    }

    #[test]
    fn matches_brute_force_on_binary_corpus() {
        use crate::test_corpus::binary_strings;
        let corpus = binary_strings(40, 24, 11);
        let rows: Vec<&[u8]> = corpus.iter().map(Vec::as_slice).collect();
        let patterns: &[&[u8]] = &[b"", b"\x00", b"\xff", b"\x00\x01", &[7u8], &[200u8, 201]];
        check(&rows, patterns);
    }
}
