// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Port of `include/onpair/search/automata/kmp_automaton.h`.

use super::{DictView, TokenAutomaton, TokenRange};
use crate::types::Token;

/// KMP state. A byte-level KMP over a pattern of length `m` has states
/// `0..=m`; `m` is the absorbing match state. Mirrors the C++ `uint8_t` so the
/// per-token `base` table stays one byte wide (it dominates cache footprint at
/// up to 64K tokens). Patterns are therefore capped at 255 bytes.
type State = u8;

/// Tokens in `[range.begin, range.last]` transition the KMP from a given entry
/// state to `target` (overriding the entry-state-0 base transition).
#[derive(Copy, Clone)]
struct SparseTransition {
    range: TokenRange,
    target: State,
}

/// Token-level KMP automaton for substring search (`col LIKE '%pattern%'`).
///
/// Each token id transitions the KMP as if its bytes were fed one by one. The
/// transition table is stored in two tiers:
///   * `base[t]` — the exit state when entering token `t` from state 0 (the
///     common case once the automaton has not yet partially matched);
///   * `sparse` — for each non-zero entry state, the few token ranges whose
///     exit state differs from `base[t]`, grouped by entry state via `offsets`.
///
/// The automaton is dead-detectable: once the match state is reached the
/// verdict can no longer change, so scanning of the row stops.
pub(crate) struct KmpAutomaton {
    match_state: State,
    state: State,
    /// `base[token]` = KMP exit state after consuming the token from state 0.
    base: Vec<State>,
    /// Flattened sparse transitions grouped by entry state: the transitions for
    /// entry state `s` live at `sparse[offsets[s]..offsets[s + 1]]`.
    sparse: Vec<SparseTransition>,
    offsets: Vec<u32>,
}

/// Consume `data` from KMP state `s`, absorbing once the match state `m` is
/// reached. Direct port of the C++ `step_bytes` lambda.
#[inline]
fn step_bytes(p: &[u8], fail: &[State], m: usize, mut s: State, data: &[u8]) -> State {
    for &b in data {
        if s as usize == m {
            return m as State;
        }
        while s > 0 && p[s as usize] != b {
            s = fail[s as usize - 1];
        }
        if p[s as usize] == b {
            s += 1;
        }
    }
    s
}

impl KmpAutomaton {
    pub(crate) fn new(pattern: &[u8], dict: DictView<'_>) -> Self {
        let m = pattern.len();
        assert!(
            m <= State::MAX as usize,
            "onpair: contains needle exceeds 255 bytes"
        );
        let num_tokens = dict.num_tokens();
        let match_state = m as State;

        if m == 0 {
            return Self {
                match_state: 0,
                state: 0,
                base: vec![0; num_tokens],
                sparse: Vec::new(),
                offsets: vec![0, 0],
            };
        }

        let p = pattern;

        // ── 1. KMP failure table ────────────────────────────────────────────
        let mut fail = vec![0 as State; m];
        {
            let mut i = 1usize;
            let mut len = 0 as State;
            while i < m {
                if p[i] == p[len as usize] {
                    len += 1;
                    fail[i] = len;
                    i += 1;
                } else if len > 0 {
                    len = fail[len as usize - 1];
                } else {
                    fail[i] = 0;
                    i += 1;
                }
            }
        }

        // ── 2. Base pass ────────────────────────────────────────────────────
        let mut base = vec![0 as State; num_tokens];
        let p0 = p[0];
        for t in 0..num_tokens {
            let tok = dict.data(t as Token);
            base[t] = if tok.contains(&p0) {
                step_bytes(p, &fail, m, 0, tok)
            } else {
                0
            };
        }

        // ── 3. Sparse pass — dual-KMP trie traversal ────────────────────────
        let mut offsets = vec![0u32; m + 1];
        let mut pass = SparsePass {
            dict,
            p,
            fail: &fail,
            base: &base,
            m,
            sparse: Vec::new(),
            range_start: 0,
        };

        let mut relevant: Vec<u8> = Vec::with_capacity(m);
        for j in 1..m {
            pass.range_start = pass.sparse.len();
            offsets[j] = pass.range_start as u32;

            // Only the bytes p[s] along the failure chain j → fail[j-1] → … → 0
            // can make state j diverge from state 0; gather and dedup them.
            relevant.clear();
            let mut s = j as State;
            while s > 0 {
                relevant.push(p[s as usize]);
                s = fail[s as usize - 1];
            }
            relevant.sort_unstable();
            relevant.dedup();

            for &byte in &relevant {
                let range = dict.prefix_range(&[byte]);
                if range.empty() {
                    continue;
                }
                let kmp_j = step_bytes(p, &fail, m, j as State, &[byte]);
                let kmp_0 = step_bytes(p, &fail, m, 0, &[byte]);
                pass.traverse(range, 1, kmp_j, kmp_0);
            }
        }
        offsets[m] = pass.sparse.len() as u32;
        // Move the sparse table out, ending the `&base` borrow held by `pass`
        // so `base` itself can be moved into the returned automaton.
        let sparse = pass.sparse;

        Self {
            match_state,
            state: 0,
            base,
            sparse,
            offsets,
        }
    }
}

/// Scratch state for the sparse-transition trie traversal. Kept in a struct so
/// the recursion (bounded by `MAX_TOKEN_SIZE` depth) can be a method.
struct SparsePass<'a> {
    dict: DictView<'a>,
    p: &'a [u8],
    fail: &'a [State],
    base: &'a [State],
    m: usize,
    sparse: Vec<SparseTransition>,
    range_start: usize,
}

impl SparsePass<'_> {
    /// Extend the last transition of the current group or push a new one.
    /// Tokens are visited in ascending order, so adjacent same-target ranges
    /// merge on the fly.
    fn emit(&mut self, range: TokenRange, target: State) {
        if self.sparse.len() > self.range_start {
            let last = self.sparse.last_mut().expect("len checked above");
            if last.target == target && last.range.last as u32 + 1 == range.begin as u32 {
                last.range.last = range.last;
                return;
            }
        }
        self.sparse.push(SparseTransition { range, target });
    }

    /// Traverse the implicit trie of the sorted dictionary over `tr`, tracking
    /// the KMP state evolved from entry state `kmp_j` and from state 0
    /// (`kmp_0`) in parallel. Where they agree the subtree yields nothing and
    /// is pruned. Direct port of the recursive C++ `traverse` lambda.
    fn traverse(&mut self, tr: TokenRange, depth: usize, kmp_j: State, kmp_0: State) {
        if kmp_j == kmp_0 || tr.empty() {
            return;
        }

        // Full match: override tokens whose base exit differs from m.
        if kmp_j as usize == self.m {
            let exit = self.m as State;
            let last = tr.last as usize;
            let mut i = tr.begin as usize;
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

        // Leaf tokens (length == depth) are fully consumed and share exit kmp_j.
        let last = tr.last as usize;
        let mut cur = tr.begin as usize;
        while cur <= last && self.dict.token_size(cur as Token) == depth {
            cur += 1;
        }
        if cur > tr.begin as usize {
            self.emit(
                TokenRange {
                    begin: tr.begin,
                    last: (cur - 1) as Token,
                },
                kmp_j,
            );
        }
        if cur > last {
            return;
        }

        // Recurse into subtrees partitioned by the byte at `depth`.
        while cur <= last {
            let c = self.dict.data(cur as Token)[depth];
            let mut sub_hi = cur;
            while sub_hi < last && self.dict.data((sub_hi + 1) as Token)[depth] == c {
                sub_hi += 1;
            }
            let nj = step_bytes(self.p, self.fail, self.m, kmp_j, &[c]);
            let n0 = step_bytes(self.p, self.fail, self.m, kmp_0, &[c]);
            self.traverse(
                TokenRange {
                    begin: cur as Token,
                    last: sub_hi as Token,
                },
                depth + 1,
                nj,
                n0,
            );
            cur = sub_hi + 1;
        }
    }
}

impl TokenAutomaton for KmpAutomaton {
    #[inline]
    fn reset(&mut self) {
        self.state = 0;
    }

    #[inline]
    fn step(&mut self, t: Token) {
        if self.is_dead() {
            return;
        }
        if self.state > 0 {
            let lo = self.offsets[self.state as usize] as usize;
            let hi = self.offsets[self.state as usize + 1] as usize;
            for tr in &self.sparse[lo..hi] {
                if t < tr.range.begin {
                    break;
                }
                if t <= tr.range.last {
                    self.state = tr.target;
                    return;
                }
            }
        }
        self.state = self.base[t as usize];
    }

    #[inline]
    fn is_accepted(&self) -> bool {
        self.state == self.match_state
    }

    #[inline]
    fn is_dead(&self) -> bool {
        self.state == self.match_state
    }
}
