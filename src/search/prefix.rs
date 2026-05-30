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

/// Verdict of the first-token prefilter, before any full row check.
pub(crate) enum Decision {
    /// The row definitely matches (decided from its first token alone).
    Accept,
    /// The row definitely does not match.
    Reject,
    /// Ambiguous — run the full [`RowMatcher::matches`] on the row.
    Verify,
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

    /// Decide a row from its first token id alone where possible.
    ///
    /// Precondition: the query is non-empty and `first_code` is either a real
    /// token id or the empty-row sentinel `u16::MAX` (which routes to
    /// [`Decision::Verify`]).
    #[inline]
    pub(crate) fn first_token_decision(&self, first_code: Token) -> Decision {
        let q0 = self.query_tokens[0];
        if first_code == q0 {
            // First token equals the query head. A single-token query is the
            // whole needle, so the row starts with it; otherwise the remaining
            // query tokens still have to be checked.
            if self.query_tokens.len() == 1 {
                Decision::Accept
            } else {
                Decision::Verify
            }
        } else if first_code != u16::MAX && self.intervals[0].contains(first_code) {
            // First token diverges but still carries the whole needle as a
            // prefix → the row starts with the needle.
            Decision::Accept
        } else if first_code == u16::MAX {
            // Empty row (sentinel): let the full check settle it.
            Decision::Verify
        } else {
            Decision::Reject
        }
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
