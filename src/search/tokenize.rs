// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Port of `include/onpair/search/detail/tokenize.h`.

use super::DictView;
use crate::types::{MAX_TOKEN_SIZE, Token};

/// Greedy longest-match tokenisation of `text` against the sorted dictionary,
/// matching the encoder's own segmentation. Used to turn a query needle into
/// the token sequence the automata reason about.
///
/// Precondition: the dictionary is sorted and contains the 256 single-byte
/// base tokens (guaranteed after [`crate::Parser::train`]).
pub(crate) fn tokenize(text: &[u8], dv: DictView<'_>) -> Vec<Token> {
    let mut tokens = Vec::with_capacity(text.len());

    let num_tokens = dv.num_tokens();
    let tlen = |t: Token| -> usize { dv.token_size(t) };
    let byte_at = |t: Token, k: usize| -> u8 { dv.data(t)[k] };

    let mut pos = 0usize;
    while pos < text.len() {
        let remaining = text.len() - pos;
        let max_len = remaining.min(MAX_TOKEN_SIZE);

        let mut best: Token = 0;
        let mut range = (0u32, (num_tokens - 1) as u32); // [begin, last]

        for k in 0..max_len {
            let target = text[pos + k];

            // Lower bound: first token in range with byte[k] >= target.
            // Tokens shorter than k+1 sort before any that has the byte.
            let (mut lo, mut hi) = (range.0, range.1);
            while lo < hi {
                let mid = lo + ((hi - lo) >> 1);
                if tlen(mid as Token) <= k || byte_at(mid as Token, k) < target {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            if tlen(lo as Token) <= k || byte_at(lo as Token, k) != target {
                break;
            }

            let first = lo;

            // Upper bound: first token with byte[k] > target, stepped back to
            // the last with byte[k] == target. `lo` already holds `first`.
            hi = range.1;
            while lo < hi {
                let mid = lo + ((hi - lo) >> 1);
                if tlen(mid as Token) <= k || byte_at(mid as Token, k) <= target {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            let last = if tlen(lo as Token) > k && byte_at(lo as Token, k) > target {
                lo - 1
            } else {
                lo
            };

            // The shortest token in range sorts first; if its length is exactly
            // k+1 it is an exact match of the consumed bytes.
            if tlen(first as Token) == k + 1 {
                best = first as Token;
            }

            range = (first, last);
        }

        tokens.push(best);
        pos += tlen(best);
    }

    tokens
}
