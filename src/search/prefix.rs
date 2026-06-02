// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Port of `include/onpair/search/automata/prefix_automaton.h`.

use super::tokenize::tokenize;
use super::{DictView, RowMatcher, TokenRange};
use crate::types::Token;

/// Token-level matcher for prefix search (`col LIKE 'prefix%'`).
///
/// The needle is tokenised once. Walking a row's tokens against the query
/// sequence:
///   * exact match → advance;
///   * mismatch at position `i` → match iff the token falls inside the
///     precomputed valid-divergence interval for `i` (the row's token still
///     has the remaining needle bytes as a prefix), else no match;
///   * all query tokens consumed → match (the rest of the row is irrelevant).
///
/// The decision is final at the first non-advancing token, so most rows are
/// settled in one step.
pub(crate) struct PrefixAutomaton {
    query_tokens: Vec<Token>,
    intervals: Vec<TokenRange>,
}

impl PrefixAutomaton {
    pub(crate) fn new(prefix: &[u8], dv: DictView<'_>) -> Self {
        let query_tokens = tokenize(prefix, dv);
        let q_len = query_tokens.len();
        let mut intervals = vec![TokenRange::EMPTY; q_len];

        // For each query position, the divergence interval is the set of tokens
        // that begin with the not-yet-consumed needle suffix.
        let mut current_pos = 0usize;
        for i in 0..q_len {
            intervals[i] = dv.prefix_range(&prefix[current_pos..]);
            current_pos += dv.token_size(query_tokens[i]);
        }

        Self {
            query_tokens,
            intervals,
        }
    }

    /// Whether the query tokenised to nothing (the empty prefix, which matches
    /// every row). The prefilter path is skipped for it.
    #[inline]
    pub(crate) fn is_empty_query(&self) -> bool {
        self.query_tokens.is_empty()
    }

    /// First-token prefilter parameters. Precondition: the query is non-empty.
    ///
    /// A row matches only if its first token either begins with the whole
    /// needle (`first_code ∈ intervals[0] = [begin, last]`) or equals the query
    /// head `q0` (with the remaining query tokens still to be checked). Because
    /// the dictionary is lexicographically sorted and `q0` is a prefix of the
    /// needle, `q0 <= begin`, so the two id sets are disjoint for a multi-token
    /// query — the [`Prefilter`] reports them separately:
    ///
    ///   * the **accept** range `[begin, last]` — a single unsigned range check
    ///     `(fc - alo) <= awidth` — is a definite match (the first token alone
    ///     begins with the needle), so it needs no row check;
    ///   * the **verify** point `q0` flags the rare case where the needle is
    ///     split at `q0`, which a full row check then settles.
    ///
    /// A single-token query *is* the whole needle, so `q0 == begin` and the
    /// accept range is necessary and sufficient; [`Prefilter::vpoint`] is then
    /// disabled. The `u16::MAX` empty-row sentinel exceeds `last` (when the
    /// dictionary is not saturated) and equals neither, so empties drop out.
    #[inline]
    pub(crate) fn prefilter(&self) -> Prefilter {
        let q0 = self.query_tokens[0];
        let iv = self.intervals[0];
        // Empty accept range → match nothing: `alo` above any u16 makes
        // `(fc - alo)` wrap past `awidth = 0` for every real first code.
        let (alo, awidth) = if iv.empty() {
            (u32::MAX, 0)
        } else {
            (iv.begin as u32, (iv.last - iv.begin) as u32)
        };
        // Single-token query is exact; disable the verify point (no u16 first
        // code can equal u32::MAX).
        let vpoint = if self.query_tokens.len() == 1 {
            u32::MAX
        } else {
            q0 as u32
        };
        Prefilter {
            alo,
            awidth,
            vpoint,
        }
    }
}

/// First-token prefilter parameters; see [`PrefixAutomaton::prefilter`].
pub(crate) struct Prefilter {
    /// Accept range lower bound. `u32::MAX` makes the range match nothing.
    pub alo: u32,
    /// Accept range width: `first_code` accepts iff `(first_code - alo) <= awidth`.
    pub awidth: u32,
    /// Verify point: `first_code == vpoint` needs a full row check. `u32::MAX`
    /// (a value no `u16` first code can take) disables verification.
    pub vpoint: u32,
}

impl Prefilter {
    /// Whether any first code can route to a full row check.
    #[inline]
    pub(crate) fn needs_verify(&self) -> bool {
        self.vpoint != u32::MAX
    }
}

impl RowMatcher for PrefixAutomaton {
    #[inline]
    fn matches(&self, codes: &[Token]) -> bool {
        // Empty prefix matches every row.
        if self.query_tokens.is_empty() {
            return true;
        }
        let mut pos = 0usize;
        for &t in codes {
            if t != self.query_tokens[pos] {
                // First divergence: matches iff the token still carries the
                // remaining needle bytes as a prefix.
                return self.intervals[pos].contains(t);
            }
            pos += 1;
            if pos == self.query_tokens.len() {
                return true;
            }
        }
        // Row ended with every token matched but the prefix not exhausted.
        false
    }
}
