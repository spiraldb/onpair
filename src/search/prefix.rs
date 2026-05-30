// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Port of `include/onpair/search/automata/prefix_automaton.h`.

use super::tokenize::tokenize;
use super::{DictView, TokenAutomaton, TokenRange};
use crate::types::Token;

#[derive(Copy, Clone, PartialEq, Eq)]
enum Status {
    Matching,
    Accepted,
    Rejected,
}

/// Token-level automaton for prefix search (`col LIKE 'prefix%'`).
///
/// The needle is tokenised once. Each incoming token is compared to the next
/// expected query token:
///   * exact match → advance;
///   * mismatch → accept iff the token falls inside the precomputed
///     valid-divergence interval for that position (the row's token still has
///     the remaining needle bytes as a prefix), else reject;
///   * all query tokens consumed → accept (the rest of the row is irrelevant).
///
/// The verdict is final the moment a divergence decision is made or the query
/// is exhausted, so the automaton is dead-detectable.
pub(crate) struct PrefixAutomaton {
    query_tokens: Vec<Token>,
    intervals: Vec<TokenRange>,
    pos: usize,
    status: Status,
}

impl PrefixAutomaton {
    pub(crate) fn new(prefix: &[u8], dv: DictView<'_>) -> Self {
        let query_tokens = tokenize(prefix, dv);
        let q_len = query_tokens.len();
        let mut intervals = vec![TokenRange::EMPTY; q_len];

        let status = if q_len == 0 {
            Status::Accepted
        } else {
            // For each query position, the divergence interval is the set of
            // tokens that begin with the not-yet-consumed needle suffix.
            let mut current_pos = 0usize;
            for i in 0..q_len {
                intervals[i] = dv.prefix_range(&prefix[current_pos..]);
                current_pos += dv.token_size(query_tokens[i]);
            }
            Status::Matching
        };

        Self {
            query_tokens,
            intervals,
            pos: 0,
            status,
        }
    }
}

impl TokenAutomaton for PrefixAutomaton {
    #[inline]
    fn reset(&mut self) {
        self.pos = 0;
        self.status = if self.query_tokens.is_empty() {
            Status::Accepted
        } else {
            Status::Matching
        };
    }

    #[inline]
    fn step(&mut self, t: Token) {
        if self.is_dead() {
            return;
        }
        if t != self.query_tokens[self.pos] {
            self.status = if self.intervals[self.pos].contains(t) {
                Status::Accepted
            } else {
                Status::Rejected
            };
            return;
        }
        self.pos += 1;
        if self.pos == self.query_tokens.len() {
            self.status = Status::Accepted;
        }
    }

    #[inline]
    fn is_accepted(&self) -> bool {
        self.status == Status::Accepted
    }

    #[inline]
    fn is_dead(&self) -> bool {
        self.status != Status::Matching
    }
}
