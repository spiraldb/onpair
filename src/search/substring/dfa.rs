// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-row search with a KMP automaton lifted from bytes to tokens.
//!
//! The states are the lengths of the matched pattern prefix, as in byte-level KMP.
//! Preparation feeds each token's bytes through the byte automaton to get the
//! token's transition, so `row_contains` then takes one step per code, without
//! reading the dictionary.
//!
//! For most tokens the transition does not depend on the state, so each token
//! stores its transition from state 0, and every other state stores only the token
//! ranges that differ.

use crate::core::dictionary::DictionaryView;
use crate::core::types::{Token, TokenRange};
use crate::search::lookup::prefix_range;
use crate::search::substring::ContainsError;

/// Length of the matched pattern prefix; the pattern length accepts. Being a `u8`,
/// it limits patterns to 255 bytes.
type State = u8;

/// A pattern compiled into a token-level automaton, to test rows one at a time
/// with [`row_contains`].
///
/// It needs no frequency index and supports patterns of up to
/// [`MAX_PATTERN_LEN`](Self::MAX_PATTERN_LEN) bytes. Unlike
/// [`ContainsScan`](crate::search::ContainsScan), it is exact for rows of any
/// tokenization, so it also serves codes the encoder did not produce; over a whole
/// code stream of encoder output, `ContainsScan` is usually faster. The automaton is
/// immutable, so it can be shared between threads.
///
/// # Examples
///
/// ```
/// use onpair::search::{ContainsDfa, row_contains};
/// use onpair::{Column, DEFAULT_CONFIG};
///
/// let bytes = b"catdogbirdcatfish";
/// let offsets = [0u32, 3, 6, 10, 13, 17];
/// let column = Column::compress(bytes, &offsets, DEFAULT_CONFIG)?;
/// let view = column.view();
///
/// // Compile once per pattern, then test rows one at a time.
/// let dfa = ContainsDfa::new(b"at", view.dict)?;
/// let rows: Vec<usize> = (0..view.num_rows())
///     .filter(|&row| row_contains(view.row_codes(row), &dfa))
///     .collect();
/// assert_eq!(rows, [0, 3]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct ContainsDfa {
    /// The accepting state: the pattern length.
    accept: State,
    /// `base[token]` is the state reached by reading `token` from state 0.
    base: Vec<State>,
    /// Exceptions to `base`: from state `s`, the tokens of each range in
    /// `sparse[offsets[s]..offsets[s + 1]]` go to its `target` instead. The ranges
    /// of a state are ascending and disjoint.
    sparse: Vec<SparseTransition>,
    /// Bounds of each state's slice of `sparse`.
    offsets: Vec<u32>,
}

/// A token range whose transition from one state differs from `base`.
#[derive(Debug, Clone, Copy)]
struct SparseTransition {
    range: TokenRange,
    target: State,
}

impl ContainsDfa {
    /// Maximum pattern length, bounded by the DFA's `u8` state IDs.
    pub const MAX_PATTERN_LEN: usize = State::MAX as usize;

    /// Compiles `pattern` for `dict`. The empty pattern matches every row.
    ///
    /// # Errors
    /// Returns [`ContainsError::PatternTooLong`] if `pattern` exceeds
    /// [`Self::MAX_PATTERN_LEN`].
    ///
    /// # Precondition
    /// `dict` is sorted, as guaranteed by [`Parser::train`](crate::Parser::train) and
    /// [`CompactDictionary::validate`](crate::CompactDictionary::validate). Otherwise
    /// the automaton's answers are unspecified, and compiling it may panic.
    pub fn new<V: DictionaryView>(pattern: &[u8], dict: V) -> Result<Self, ContainsError> {
        if pattern.len() > Self::MAX_PATTERN_LEN {
            return Err(ContainsError::PatternTooLong {
                length: pattern.len(),
                max: Self::MAX_PATTERN_LEN,
            });
        }
        // The start state accepts the empty pattern, so no transition is ever taken.
        if pattern.is_empty() {
            return Ok(Self {
                accept: 0,
                base: Vec::new(),
                sparse: Vec::new(),
                offsets: vec![0],
            });
        }
        let accept = pattern.len();
        let num_tokens = dict.num_tokens();
        let mut build = Build {
            dict,
            pattern,
            accept,
            fail: kmp_failure(pattern),
            base: vec![0; num_tokens],
            sparse: Vec::new(),
            offsets: vec![0u32; accept + 1],
            range_start: 0,
        };
        build.base_pass();
        build.sparse_pass();

        Ok(Self {
            accept: accept as State,
            base: build.base,
            sparse: build.sparse,
            offsets: build.offsets,
        })
    }

    /// The state after reading `token` from `state`, which must not be accepting.
    /// A code outside the dictionary leaves no pattern prefix matched.
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
        self.base.get(usize::from(token)).copied().unwrap_or(0)
    }
}

/// Whether the row encoded by `codes` contains the pattern compiled into `dfa`.
///
/// # Precondition
/// `codes` were encoded with the dictionary `dfa` was compiled for. A code outside
/// that dictionary never matches.
pub fn row_contains(codes: &[Token], dfa: &ContainsDfa) -> bool {
    // Testing before each step accepts the empty pattern, whose start state
    // accepts, in every row.
    let mut state: State = 0;
    for &code in codes {
        if state == dfa.accept {
            return true;
        }
        state = dfa.next(state, code);
    }
    state == dfa.accept
}

/// The byte-level KMP failure table: `fail[i]` is the length of the longest proper
/// prefix of `pattern[..=i]` that is also its suffix.
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

/// Construction state of a `ContainsDfa`.
struct Build<'a, V> {
    dict: V,
    pattern: &'a [u8],
    /// The accepting state: the pattern length.
    accept: usize,
    /// Byte-level failure links for falling back to shorter matched prefixes.
    fail: Vec<State>,
    base: Vec<State>,
    sparse: Vec<SparseTransition>,
    offsets: Vec<u32>,
    /// First exception for the current entry state, bounding adjacent-range merging.
    range_start: usize,
}

impl<V: DictionaryView> Build<'_, V> {
    /// Runs the byte automaton over `data` from state `s`. Acceptance is final.
    fn step_bytes(&self, mut s: State, data: &[u8]) -> State {
        for &b in data {
            if s as usize == self.accept {
                return self.accept as State;
            }
            while s > 0 && self.pattern[s as usize] != b {
                s = self.fail[s as usize - 1];
            }
            if self.pattern[s as usize] == b {
                s += 1;
            }
        }
        s
    }

    /// Fills `base`: the state each token leads to from state 0.
    fn base_pass(&mut self) {
        let first = self.pattern[0];
        for t in 0..self.base.len() {
            let token = self.dict.token(t as Token);
            // A token without the pattern's first byte cannot leave state 0.
            self.base[t] = if token.contains(&first) {
                self.step_bytes(0, token)
            } else {
                0
            };
        }
    }

    /// Appends an exception for the current state, merging it into the previous
    /// one when adjacent with the same target.
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

    /// Fills `sparse` with the exceptions of every nonzero, non-accepting state.
    fn sparse_pass(&mut self) {
        let mut relevant: Vec<u8> = Vec::new();
        for j in 1..self.accept {
            self.range_start = self.sparse.len();
            self.offsets[j] = self.range_start as u32;

            // Only bytes p[s] along the failure chain j → fail[j-1] → … → 0 can
            // make state j transition differently from state 0; skip the rest.
            relevant.clear();
            let mut s = j as State;
            while s > 0 {
                relevant.push(self.pattern[s as usize]);
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
        self.offsets[self.accept] = self.sparse.len() as u32;
    }

    /// Records the exceptions among the tokens `tr`, which share their first `depth`
    /// bytes. Those bytes lead to `kmp_j` from the current state and to `kmp_0` from
    /// state 0. Once the two agree, the whole subtree agrees and is skipped.
    fn traverse(&mut self, tr: TokenRange, depth: usize, kmp_j: State, kmp_0: State) {
        if kmp_j == kmp_0 || tr.is_empty() {
            return;
        }
        let (begin, last) = (tr.begin as usize, tr.last as usize);

        // Acceptance survives the remaining token bytes. Record only tokens
        // whose complete base transition does not already accept.
        if kmp_j as usize == self.accept {
            let exit = self.accept as State;
            let mut i = begin;
            while i <= last {
                if self.base[i] != exit {
                    let start = i;
                    while i <= last && self.base[i] != exit {
                        i += 1;
                    }
                    self.emit(TokenRange::new(start as Token, (i - 1) as Token), exit);
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
            self.emit(TokenRange::new(begin as Token, (cur - 1) as Token), kmp_j);
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
                TokenRange::new(cur as Token, sub_hi as Token),
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
    use crate::test_corpus::{byte_contains, compress_rows, decode_row};

    /// Compares row search with byte containment using compact and wide dictionaries.
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
    fn empty_pattern_matches_every_row() {
        let column = compress_rows(&[b"a", b"", b"abc"]);
        let view = column.view();
        let dfa = ContainsDfa::new(b"", view.dict).unwrap();
        assert!((0..view.num_rows()).all(|row| row_contains(view.row_codes(row), &dfa)));
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
        // the sparse cross-token transitions. The last row matches the last pattern
        // only through a failure link that took a fallback to compute.
        let rows: &[&[u8]] = &[b"aaaaab", b"aabaab", b"ababab", b"aaa", b"aabaaabaaaa"];
        check(
            rows,
            &[
                b"aa", b"aaa", b"aab", b"abab", b"aaaa", b"aabaa", b"aabaaaa",
            ],
        );
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
        let patterns: &[&[u8]] = &[b"\x00", b"\xff", b"\x00\x01", &[7u8], &[200u8, 201]];
        check(&rows, patterns);
    }

    /// Every string of `a` and `b` of at most `max` bytes.
    fn ab_strings(max: u32) -> Vec<Vec<u8>> {
        (0..=max)
            .flat_map(|len| {
                (0..1u32 << len)
                    .map(move |bits| (0..len).map(|at| b'a' + (bits >> at & 1) as u8).collect())
            })
            .collect()
    }

    /// Every way to split `row` into tokens of `dict`.
    fn tokenizations<V: DictionaryView>(dict: V, row: &[u8]) -> Vec<Vec<Token>> {
        if row.is_empty() {
            return vec![Vec::new()];
        }
        (0..dict.num_tokens())
            .map(|token| token as Token)
            .filter(|&token| row.starts_with(dict.token(token)))
            .flat_map(|token| {
                tokenizations(dict, &row[dict.token_len(token)..])
                    .into_iter()
                    .map(move |rest| [vec![token], rest].concat())
            })
            .collect()
    }

    /// The automaton is exact for any tokenization: every split of a row into tokens
    /// gets the answer of the row's bytes. The encoder's rows cannot show this, because
    /// an automaton without the transitions they never take still passes on them.
    #[test]
    fn matches_bytes_on_any_tokenization() {
        use crate::test_corpus::complete_dictionary;
        let dict = complete_dictionary(&[b"aa", b"ab", b"aba", b"abab", b"ba", b"bab"]);
        let view = dict.as_view();
        let rows: Vec<(Vec<u8>, Vec<Vec<Token>>)> = ab_strings(7)
            .into_iter()
            .map(|row| {
                let splits = tokenizations(view, &row);
                (row, splits)
            })
            .collect();
        for pattern in ab_strings(5) {
            let dfa = ContainsDfa::new(&pattern, view).unwrap();
            for (row, splits) in &rows {
                let want = byte_contains(row, &pattern);
                for codes in splits {
                    assert_eq!(row_contains(codes, &dfa), want, "{row:?}, {codes:?}");
                }
            }
        }
    }
}
