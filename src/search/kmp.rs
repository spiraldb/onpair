// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Port of `include/onpair/search/automata/kmp_automaton.h`.

use super::{DictView, RowMatcher, TokenRange};
use crate::types::{MAX_TOKEN_SIZE, Token};

/// Upper bound on `|Cand(s)| · |Comp(s)|` pairs the LPM completion-absorption
/// test ([`KmpAutomaton::completion_lpm_dead`]) will scan per state before
/// giving up and keeping the completions. Bounds per-query build time; sound at
/// any value (a smaller budget only forgoes pruning).
const LPM_PAIR_BUDGET: usize = 1 << 20;

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
/// [`matches`](RowMatcher::matches) splits the scan into a fast path (KMP
/// state 0, the common case) whose `base[]` loads carry no state between
/// iterations and so pipeline, and a slow partial-match path that consults the
/// sparse table. It stops the row the moment the match state is reached.
pub(crate) struct KmpAutomaton {
    match_state: State,
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
        // base[t] = KMP state after feeding token t from state 0. A token whose
        // bytes do not contain p[0] can never leave state 0, so it is skipped.
        // The presence test is a branchless inline scan rather than
        // `tok.contains(&p0)`: std specialises the latter to a `memchr()` *call*,
        // and the call overhead dominates on these ≤16-byte tokens (one call per
        // dictionary token — up to 64K of them per query build).
        let mut base = vec![0 as State; num_tokens];
        let p0 = p[0];
        for t in 0..num_tokens {
            let tok = dict.data(t as Token);
            let mut has = false;
            for &b in tok {
                has |= b == p0;
            }
            base[t] = if has {
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
            base,
            sparse,
            offsets,
        }
    }

    /// Full KMP transition from `state` (in `1..match_state`) on token `t`:
    /// consult the sparse exceptions for `state`, falling back to `base[t]`.
    #[inline]
    fn next_state(&self, state: State, t: Token) -> State {
        let lo = self.offsets[state as usize] as usize;
        let hi = self.offsets[state as usize + 1] as usize;
        for tr in &self.sparse[lo..hi] {
            if t < tr.range.begin {
                break;
            }
            if t <= tr.range.last {
                return tr.target;
            }
        }
        self.base[t as usize]
    }

    /// Per-token class table for the prefilter, one entry per token id:
    /// Per-token table for the Teddy-style 2-code chain prefilter. Each entry is
    /// an OR of three independent bit flags:
    ///   * [`CHAIN_DEFINITE`] — the token contains the whole needle
    ///     (`base == match_state`); a row holding it matches outright.
    ///   * [`CHAIN_OPEN`] — the token opens a partial match from state 0
    ///     (`base != 0`); it can be the first token of a boundary-spanning match.
    ///   * [`CHAIN_CONT`] — feeding the token from *some* positive entry state
    ///     can leave the KMP positive (not dead); it can be the second token of a
    ///     boundary-spanning pair. Computed as a sound superset: `base != 0`, or
    ///     any sparse transition for any entry state has target `!= 0` and covers
    ///     this token id.
    ///
    /// The row scan then accepts a row iff it has a DEFINITE token, or a
    /// consecutive pair `(open token, continue token)`. Soundness: in any
    /// matching row with no DEFINITE token, walk the KMP state sequence back from
    /// the match to the opener `j` (`s_{j-1}=0 < s_j`); token `j` is OPEN and the
    /// next token `j+1` (which exists, since no token does `0 -> match` alone)
    /// has a positive entry state staying positive, hence CONT. So every match
    /// exhibits a DEFINITE token or an OPEN→CONT pair — no false negatives.
    pub(crate) fn chain_table(&self) -> Vec<u8> {
        let m = self.match_state;
        let mut t: Vec<u8> = self
            .base
            .iter()
            .map(|&b| {
                if b == m {
                    CHAIN_DEFINITE | CHAIN_OPEN | CHAIN_CONT
                } else if b != 0 {
                    CHAIN_OPEN | CHAIN_CONT
                } else {
                    0
                }
            })
            .collect();
        // Any sparse transition with a non-dead target means the covered tokens
        // can continue a partial match from that entry state: mark CONT.
        for tr in &self.sparse {
            if tr.target != 0 {
                let lo = tr.range.begin as usize;
                let hi = tr.range.last as usize;
                for cell in &mut t[lo..=hi] {
                    *cell |= CHAIN_CONT;
                }
            }
        }
        t
    }

    /// Contiguous token-id ranges of INNER tokens: those that complete or
    /// continue a partial match — `base[t] == match_state` (DEFINITE) or covered
    /// by a sparse transition with a non-dead target. Merged and sorted.
    ///
    /// INNER is a *sound necessary* contains filter: the token that completes any
    /// match enters from state 0 (then `base == match_state`, DEFINITE) or from a
    /// positive state via a sparse transition (INNER), so every matching row
    /// holds an INNER token. Unlike the scattered open-set, these tokens cluster
    /// into few contiguous ranges (the dictionary sorts by leading byte and a
    /// continuation needs a specific next byte), so the filter is SIMD
    /// range-testable. Returns `None` if there are more ranges than `max` (not
    /// worth a per-code multi-range test).
    pub(crate) fn inner_ranges(&self, dict: DictView<'_>, max: usize) -> Option<Vec<(Token, Token)>> {
        let m = self.match_state;
        let mut raw: Vec<(Token, Token)> = Vec::new();
        // DEFINITE runs in base.
        let mut i = 0u32;
        let n = self.base.len() as u32;
        while i < n {
            if self.base[i as usize] == m {
                let mut j = i + 1;
                while j < n && self.base[j as usize] == m {
                    j += 1;
                }
                raw.push((i as Token, (j - 1) as Token));
                i = j;
            } else {
                i += 1;
            }
        }
        // Completing sparse transitions only (target == match state), and only
        // from a boundary-reachable entry state that greedy LPM does not render
        // completion-dead. Three sound tightenings:
        //  1. A row matches iff some boundary reaches `m`; the token completing
        //     that step enters from state 0 (DEFINITE, above) or via a sparse
        //     transition with target `m`. Partial→partial transitions can never
        //     be the completing token, so dropping them adds no false negative.
        //  2. A completing transition from entry state `s` can only fire if a
        //     boundary ever lands on `s`. `reachable_states` over-approximates
        //     the reachable boundary states from the dictionary alone, so
        //     skipping transitions from unreachable `s` drops no true match.
        //  3. `completion_lpm_dead` (greedy-LPM absorption): even when a boundary
        //     can land on `s`, greedy longest-prefix tokenisation may make it
        //     impossible for the *next* token to be a completing one — see that
        //     method's proof. Dropping those completions drops no true match.
        // All three only ever remove false positives — KMP still confirms.
        let reach = self.reachable_states();
        for s in 1..m as usize {
            if !reach[s] {
                continue;
            }
            let lo = self.offsets[s] as usize;
            let hi = self.offsets[s + 1] as usize;
            // Candidate last-tokens into `s`, computed lazily and only once per
            // state (only when `s` actually has a completing transition).
            let mut cand: Option<Vec<Token>> = None;
            for tr in &self.sparse[lo..hi] {
                if tr.target != m {
                    continue;
                }
                let cands = cand.get_or_insert_with(|| self.candidates_into(s as State, &reach));
                // Drop this completing range if greedy LPM proves it can never
                // fire (every candidate·completer pair is absorbed by a longer
                // token). Sound: see `range_completion_lpm_dead`.
                if !range_completion_lpm_dead(dict, cands, tr.range) {
                    raw.push((tr.range.begin, tr.range.last));
                }
            }
        }
        if raw.is_empty() {
            return Some(Vec::new());
        }
        // Merge overlapping/adjacent ranges.
        raw.sort_unstable();
        let mut merged: Vec<(Token, Token)> = Vec::with_capacity(raw.len());
        for (lo, hi) in raw {
            if let Some(last) = merged.last_mut()
                && lo <= last.1.saturating_add(1)
            {
                last.1 = last.1.max(hi);
                continue;
            }
            merged.push((lo, hi));
        }
        if merged.len() > max {
            None
        } else {
            Some(merged)
        }
    }

    /// Every token `t` that can land the DFA in partial state `s` at a token
    /// boundary: `base[t] == s` (entry state 0, always reachable), or some sparse
    /// transition from a *reachable* entry state has target `s`. A *superset* of
    /// the real last-tokens producing a boundary at `s` — `reach` already
    /// over-approximates the boundary states, so this can only over-include,
    /// never miss, keeping the LPM-absorption test that consumes it sound.
    fn candidates_into(&self, s: State, reach: &[bool]) -> Vec<Token> {
        let mut cands: Vec<Token> = Vec::new();
        for (t, &b) in self.base.iter().enumerate() {
            if b == s {
                cands.push(t as Token);
            }
        }
        for e in 1..self.match_state as usize {
            if !reach[e] {
                continue;
            }
            let lo = self.offsets[e] as usize;
            let hi = self.offsets[e + 1] as usize;
            for tr in &self.sparse[lo..hi] {
                if tr.target == s {
                    cands.extend(tr.range.begin..=tr.range.last);
                }
            }
        }
        cands
    }

    /// The set of DFA states that can occur at a token boundary, as a sound
    /// over-approximation computed from the dictionary alone (no row data).
    ///
    /// Fixpoint over the real per-token transition function: state 0 (row start
    /// / KMP death) is always reachable; state `s'` is reachable if some
    /// boundary-reachable state `s` has a token `t` with `step(s, t) == s'`.
    ///
    /// Soundness: any boundary an actual row reaches is the image of a
    /// (previous boundary state, token) pair, so it is included. This does NOT
    /// model LPM (greedy tokenisation never taking a shorter token when a longer
    /// one fits), so it can mark a state reachable that LPM forbids in practice
    /// — that only adds false positives to a prefilter built from it, never a
    /// false negative.
    fn reachable_states(&self) -> Vec<bool> {
        let m = self.match_state as usize;
        let mut reach = vec![false; m + 1];
        reach[0] = true;

        // States reachable in one token step from state 0 are exactly the
        // distinct values in `base[]`; collect them in a single linear pass
        // instead of re-scanning all tokens for every (state, fixpoint round) as
        // the naive transition fixpoint did (its `next_state` probe per token per
        // state dominated query build time).
        let mut base_target = vec![false; m + 1];
        for &b in &self.base {
            base_target[b as usize] = true;
        }

        // Fixpoint over the `m` states only. From a reachable state you can land
        // on any base target (the non-overridden tokens), plus — from a positive
        // state — any of its sparse override targets. Using *all* base targets
        // from every state over-approximates the exact transition image, so this
        // can only ever mark MORE states reachable, never fewer: for every real
        // edge `s --t--> ns`, `ns` is either `base[t]` (a base target) or a sparse
        // override target of `s`, so it is included. A sound over-approximation is
        // all the INNER filter needs (it only drops completions from states that
        // are provably never boundaries).
        let mut changed = true;
        while changed {
            changed = false;
            for ns in 0..=m {
                if base_target[ns] && !reach[ns] {
                    reach[ns] = true;
                    changed = true;
                }
            }
            for s in 1..m {
                if !reach[s] {
                    continue;
                }
                let lo = self.offsets[s] as usize;
                let hi = self.offsets[s + 1] as usize;
                for tr in &self.sparse[lo..hi] {
                    let ns = tr.target as usize;
                    if !reach[ns] {
                        reach[ns] = true;
                        changed = true;
                    }
                }
            }
        }
        reach
    }

    /// Whether the needle is empty (matches every row); the prefilter is skipped
    /// for it.
    #[inline]
    pub(crate) fn is_empty_needle(&self) -> bool {
        self.match_state == 0
    }

    /// Debug: the match (absorbing) state `m`.
    #[cfg(test)]
    pub(crate) fn match_state(&self) -> u8 {
        self.match_state
    }

    /// Debug: number of token ids in the alphabet.
    #[cfg(test)]
    pub(crate) fn num_tokens(&self) -> usize {
        self.base.len()
    }

    /// Debug: full token transition from any entry state (0..match_state).
    #[cfg(test)]
    pub(crate) fn step_from(&self, state: u8, t: Token) -> u8 {
        if state == 0 {
            self.base[t as usize]
        } else if state == self.match_state {
            self.match_state
        } else {
            self.next_state(state, t)
        }
    }

    /// Debug: for each entry state `s` in `1..match_state`, how many tokens
    /// leave the DFA in exactly state `s` when fed from state 0 (i.e. end in the
    /// `s`-byte needle prefix) — `base[t] == s`. A state with zero such tokens
    /// can only be reached at a token boundary through a multi-token chain.
    #[cfg(test)]
    pub(crate) fn boundary_state_counts(&self) -> Vec<usize> {
        let m = self.match_state as usize;
        let mut counts = vec![0usize; m + 1];
        for &b in &self.base {
            counts[b as usize] += 1;
        }
        counts
    }

    /// Debug: render the token-level DFA. For each entry state `s` returns the
    /// list of `(token-id range, target state)` transitions that differ from the
    /// state-0 default, plus the run-length encoding of `base` (the state-0 row).
    /// `(base_runs, per_state_sparse)`.
    #[cfg(test)]
    #[allow(clippy::type_complexity)]
    pub(crate) fn dump_dfa(&self) -> (Vec<(u32, u32, u8)>, Vec<Vec<(u16, u16, u8)>>) {
        // RLE of base[]: contiguous id ranges mapping to the same target state.
        let mut base_runs = Vec::new();
        let mut i = 0u32;
        let n = self.base.len() as u32;
        while i < n {
            let t = self.base[i as usize];
            let mut j = i + 1;
            while j < n && self.base[j as usize] == t {
                j += 1;
            }
            base_runs.push((i, j - 1, t));
            i = j;
        }
        let mut per_state = Vec::new();
        for s in 0..self.match_state as usize {
            let lo = self.offsets[s] as usize;
            let hi = self.offsets[s + 1] as usize;
            per_state.push(
                self.sparse[lo..hi]
                    .iter()
                    .map(|tr| (tr.range.begin, tr.range.last, tr.target))
                    .collect(),
            );
        }
        (base_runs, per_state)
    }
}

/// Whether greedy longest-prefix-match (LPM) tokenisation makes it impossible
/// for any completer in the token-id range `range` to immediately follow a token
/// boundary in the partial state those completers complete from — so this
/// completing range can be dropped from the INNER prefilter with no false
/// negative.
///
/// # The lever
///
/// For `%google%` the deep completions fan out to ~1.5k tokens, yet they never
/// fire: greedy LPM never ends a token at `googl` followed by `e` (the longer
/// token `google` absorbs it), nor at `goog` followed by `le` (likewise). This
/// proves that soundly, per completing range, so `%google%`'s INNER filter
/// collapses from ~1.5k tokens to a handful.
///
/// # Model
///
/// The column is encoded by greedy LPM (`Parser::encode_strings`): at each input
/// position emit the *longest* dictionary token that is a prefix of the
/// remaining bytes, then advance past it. Token boundaries are exactly those cut
/// points. At a boundary the DFA is in the state = length of the longest
/// needle-prefix that is a suffix of the consumed bytes.
///
/// # Criterion
///
/// `cands` is `Cand(s)` (see [`KmpAutomaton::candidates_into`]); `range` is a set
/// of sparse completers `u` from `s` (`step(s, u) == m`, all sharing a contiguous
/// token-id range). Return `true` iff for every `t ∈ cands` and every `u ∈
/// range`, the byte string `t · u` has *some* dictionary token as a prefix that
/// is strictly longer than `t`.
///
/// # Proof of soundness
///
/// Suppose, for contradiction, some encoded row has a boundary `B` with DFA
/// state `s` whose next token `u` lies in `range` and completes the match
/// (`step(s, u) == m`). Because `s ≥ 1`, `B` is not the row start, so a token `t`
/// ends exactly at `B`; its entry state `e` is a reachable boundary state and
/// `step(e, t) == s`, hence `t ∈ cands`. The remaining input at `t`'s start
/// position begins with `t · u` (token `t`, then token `u`). By the criterion
/// some dictionary token `w` with `|t| < |w| ≤ |t · u|` is a prefix of `t · u`,
/// hence a prefix of that remaining input — so greedy LPM, which picked `t`
/// there, had a strictly longer match `w` available: contradiction. Hence no
/// such boundary exists and this range's completion never fires. ∎
///
/// Dropping the range removes those tokens' state-`s` completion role only; any
/// row that truly matches still holds an INNER token via a *retained* route (a
/// DEFINITE token, or a completion the proof does not rule out), because the one
/// route this drops is exactly the one shown impossible.
///
/// Checking the *whole* `t · u` (not just `t · u[0]`) is what makes the lever
/// bite across the `goog | le` boundary: the absorber is the two-byte extension
/// `goog·le = google ∈ D`, which a single-byte test (`goog·l = googl ∉ D`) would
/// miss. The `O(|cands| · |range|)` pair scan is bounded by [`LPM_PAIR_BUDGET`];
/// above it the function conservatively returns `false` (keeps the range) rather
/// than do unbounded build-time work — still sound, just less pruning.
///
/// Empirically validated against a full re-simulation of the token automaton
/// over the corpus (`lpm_prune_probe`): no pruned completion ever fired across
/// 1M ClickBench rows over many needles, and the bench cross-check (`cd == bf`)
/// confirms zero false negatives end-to-end.
fn range_completion_lpm_dead(dict: DictView<'_>, cands: &[Token], range: TokenRange) -> bool {
    if cands.is_empty() {
        // The completing state cannot be a boundary at all ⇒ vacuously dead.
        return true;
    }
    let n_comp = (range.last as usize).saturating_sub(range.begin as usize) + 1;
    // Bound build-time work: keep the range if the pair scan is too large.
    // Sound — only ever forgoes pruning.
    if cands.len().saturating_mul(n_comp) > LPM_PAIR_BUDGET {
        return false;
    }

    // Every (candidate, completer) pair must be absorbed: `t · u` carries a dict
    // token longer than `t` as a prefix, so greedy never ends a token at `t`
    // right before `u`.
    let mut buf = [0u8; 2 * MAX_TOKEN_SIZE];
    for &t in cands {
        let tb = dict.data(t);
        let tl = tb.len();
        // A token already at the maximum length cannot be extended into a longer
        // token, so nothing can absorb it.
        if tl >= MAX_TOKEN_SIZE {
            return false;
        }
        buf[..tl].copy_from_slice(tb);
        for u in range.begin..=range.last {
            let ub = dict.data(u);
            let w = (tl + ub.len()).min(MAX_TOKEN_SIZE);
            buf[tl..w].copy_from_slice(&ub[..w - tl]);
            // Look for any token prefix of `t · u` strictly longer than `t`.
            let absorbed = (tl + 1..=w).any(|l| dict.contains_token(&buf[..l]));
            if !absorbed {
                return false;
            }
        }
    }
    true
}

/// [`chain_table`](KmpAutomaton::chain_table) flags. A token containing the
/// whole needle (definite match for any row holding it).
pub(crate) const CHAIN_DEFINITE: u8 = 4;
/// The token opens a partial match from state 0 (can start a spanning match).
pub(crate) const CHAIN_OPEN: u8 = 1;
/// The token can continue a partial match (can be the second of a spanning pair).
pub(crate) const CHAIN_CONT: u8 = 2;

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

impl RowMatcher for KmpAutomaton {
    #[inline]
    fn matches(&self, codes: &[Token]) -> bool {
        // Empty needle matches every row.
        if self.match_state == 0 {
            return true;
        }
        let base = self.base.as_slice();
        let match_state = self.match_state;
        let n = codes.len();
        let mut i = 0usize;
        while i < n {
            // Fast path: KMP state 0. `base[code]` loads are independent across
            // iterations (no state carried between them), so the CPU pipelines
            // them — no `is_dead`/`state > 0` branch, no sparse lookup.
            let s = base[codes[i] as usize];
            i += 1;
            if s != 0 {
                if s == match_state {
                    return true;
                }
                // Slow path: a partial match is open. Step carefully (sparse
                // exceptions + base) until it completes, dies back to state 0,
                // or the row ends.
                let mut state = s;
                while i < n {
                    state = self.next_state(state, codes[i]);
                    i += 1;
                    if state == match_state {
                        return true;
                    }
                    if state == 0 {
                        break;
                    }
                }
            }
        }
        false
    }
}
