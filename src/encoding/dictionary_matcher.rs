// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Frozen longest-prefix matcher for encoding. [`DictionaryMatcher`] is built
//! from the final, sorted dictionary and matches several inputs at once without
//! data-dependent branches.

use std::hint::select_unpredictable;

use crate::core::dictionary::{CompactDictionaryView, DictionaryView};
use crate::core::types::{MAX_TOKEN_SIZE, Token};

/// Number of distinct 2-byte prefixes.
const PREFIX_SLOTS: usize = 1 << 16;

/// Frozen longest-prefix matcher over a sorted, complete dictionary. Token ids
/// are dictionary indices. It matches several inputs at once without
/// data-dependent branches, so independent inputs overlap in the CPU.
///
/// Each token is held as a 16-byte big-endian key, zero-padded past its length.
/// Lexicographic token order is then also key order. For input `w`, let `q` be
/// the last token with `key(q) <= key(w)`. Every token that is a prefix of `w`
/// sorts between that token and `w`, so it is also a prefix of `q`. Thus the
/// longest match is `q` or the longest prefix of `q` that is not longer than
/// the common prefix of `q` and `w`. Each token stores its prefix tokens in one
/// record, so that last step is a mask and one load. The single-byte token
/// of `w[0]` sorts at or before `q`, so `q` always exists and shares the first
/// byte of `w`.
///
/// Keys are stored in blocks of [`BLOCK_KEYS`], one cache line each, and the
/// first key of every block is copied into a small sample array that stays in
/// cache. A direct table on the first two bytes gives the range of tokens to
/// search for `q`. A binary search over the samples of that range finds the
/// block of `q`, and three more steps inside that block find `q`. So a search
/// reads one block that may miss the cache.
/// All inputs of one call search with the same number of steps, so the loop has
/// no branch that depends on the data; see [`advance_if_le`].
#[derive(Debug, Clone)]
pub(crate) struct DictionaryMatcher {
    /// Token keys in dictionary order, then `u128::MAX` up to a whole block.
    blocks: Vec<KeyBlock>,
    /// First key of each block.
    samples: Vec<u128>,
    /// Per 2-byte prefix `p`: the block of the last token before the tokens
    /// whose keys start with `p` (block 0 if there is none). The block of the
    /// last token with prefix `p` is then `first_blocks[p + 1]`.
    first_blocks: Vec<u16>,
    /// Prefix tokens of each token, in dictionary order.
    chains: Vec<PrefixChain>,
    /// Token of each single byte: the first prefix token of every chain.
    singles: [Token; 256],
}

/// Keys per block: one 128-byte cache line.
const BLOCK_KEYS: usize = 8;

/// One cache line of keys.
#[derive(Copy, Clone, Debug)]
#[repr(align(128))]
struct KeyBlock([u128; BLOCK_KEYS]);

/// The tokens that are prefixes of one token `t`, including `t`, in one
/// 32-byte record, so that the match needs one load after the search.
#[derive(Copy, Clone, Debug, Default)]
#[repr(align(32))]
struct PrefixChain {
    /// Bit `k` is set when the first `k + 1` bytes of `t` are a token.
    lens: u16,
    /// The prefix tokens after the single-byte one, shortest first: one per
    /// set bit of `lens` above bit 0.
    ids: [Token; MAX_TOKEN_SIZE - 1],
}

impl DictionaryMatcher {
    /// Build a matcher from a dictionary: token at index `i` receives id `i`.
    ///
    /// Precondition: the dictionary is strictly sorted and contains every
    /// single-byte token.
    pub(crate) fn from_dictionary(dict: CompactDictionaryView<'_>) -> Self {
        let n = dict.num_tokens();
        let mut keys = Vec::with_capacity(n + 1);
        let mut chains = Vec::with_capacity(n);
        let mut singles = [0 as Token; 256];

        // Prefix tokens of the current token, shortest first. A token that is a
        // prefix of a later token stays on the stack while every token between
        // them extends it.
        let mut chain: Vec<Token> = Vec::with_capacity(MAX_TOKEN_SIZE);

        for i in 0..n {
            let id = i as Token;
            let token = dict.token(id);
            debug_assert!(!token.is_empty() && token.len() <= MAX_TOKEN_SIZE);
            debug_assert!(
                i == 0 || dict.token(id - 1) < token,
                "dictionary is not sorted"
            );

            let mut buf = [0u8; MAX_TOKEN_SIZE];
            buf[..token.len()].copy_from_slice(token);
            keys.push(u128::from_be_bytes(buf));

            while let Some(&top) = chain.last() {
                if token.starts_with(dict.token(top)) {
                    break;
                }
                chain.pop();
            }
            chain.push(id);
            debug_assert_eq!(dict.token_len(chain[0]), 1, "dictionary is incomplete");

            if token.len() == 1 {
                singles[token[0] as usize] = id;
            }

            let mut record = PrefixChain::default();
            for &prefix in &chain {
                record.lens |= 1 << (dict.token_len(prefix) - 1);
            }
            record.ids[..chain.len() - 1].copy_from_slice(&chain[1..]);
            chains.push(record);
        }

        // `starts[p]` is the first token whose 2-byte prefix is `p` or larger.
        let mut starts = vec![0usize; PREFIX_SLOTS + 1];
        for &key in &keys[..n] {
            starts[(key >> 112) as usize + 1] += 1;
        }
        for p in 0..PREFIX_SLOTS {
            starts[p + 1] += starts[p];
        }

        // At most 2^16 tokens, so at most 2^13 blocks: a block fits a `u16`.
        let first_blocks = starts
            .iter()
            .map(|&start| ((start.max(1) - 1) / BLOCK_KEYS) as u16)
            .collect();

        // At least one `u128::MAX` pads the last block, so a search past the
        // last token stops there.
        keys.resize((n / BLOCK_KEYS + 1) * BLOCK_KEYS, u128::MAX);
        let blocks: Vec<KeyBlock> = keys
            .chunks_exact(BLOCK_KEYS)
            .map(|chunk| KeyBlock(chunk.try_into().unwrap()))
            .collect();
        let samples = blocks.iter().map(|block| block.0[0]).collect();

        Self {
            blocks,
            samples,
            first_blocks,
            chains,
            singles,
        }
    }

    /// Longest token whose bytes are a prefix of `data`, with that prefix's
    /// length.
    ///
    /// Precondition: `!data.is_empty()`.
    #[cfg(test)]
    pub(crate) fn find_longest_match(&self, data: &[u8]) -> (Token, usize) {
        self.find_all(data, &[0], &[data.len().min(MAX_TOKEN_SIZE)])[0]
    }

    /// For each `j`, the longest token that is a prefix of the `win[j]` bytes at
    /// `bytes[at[j]..]`, with its length.
    ///
    /// Precondition: `1 <= win[j] <= MAX_TOKEN_SIZE`. Bytes past the end of
    /// `bytes` read as zero, so a window that runs past the end gives a result
    /// that is meaningful only up to the end.
    #[inline(always)]
    pub(crate) fn find_all<const K: usize>(
        &self,
        bytes: &[u8],
        at: &[usize; K],
        win: &[usize; K],
    ) -> [(Token, usize); K] {
        let key: [u128; K] = std::array::from_fn(|j| window_key(bytes, at[j], win[j]));

        // Candidate blocks: from the block of the last token before the
        // input's 2-byte prefix to the block of the last token with that
        // prefix. Every key before the prefix is smaller than the input key, so
        // the first candidate block starts with a key `<=` it. With no token
        // before the prefix, the first block is block 0, which starts with the
        // byte 0, whose key `0` is `<=` every input key.
        let mut base = [0usize; K];
        let mut size = [0usize; K];
        for j in 0..K {
            let prefix = (key[j] >> 112) as usize;
            base[j] = self.first_blocks[prefix] as usize;
            size[j] = self.first_blocks[prefix + 1] as usize + 1 - base[j];
        }

        // Find the last candidate block whose first key is `<=` the input key.
        // The searches advance one step at a time across all inputs, so their
        // loads sit next to each other and overlap.
        let largest = size.iter().copied().max().unwrap_or(0);
        let steps = usize::BITS - largest.saturating_sub(1).leading_zeros();
        for _ in 0..steps {
            for j in 0..K {
                let half = size[j] / 2;
                let mid = base[j] + half;
                base[j] = advance_if_le(self.samples[mid], key[j], base[j], half);
                size[j] -= half;
            }
        }

        // Keys are sorted, the block starts with a key `<=` the input key, and
        // the next block starts with a larger one. So `q` is the last key of
        // the block that is `<=` the input key. The steps read one cache line.
        let mut slot = [0usize; K];
        let mut half = BLOCK_KEYS / 2;
        while half > 0 {
            for j in 0..K {
                let block = &self.blocks[base[j]].0;
                slot[j] = advance_if_le(block[slot[j] + half], key[j], slot[j], half);
            }
            half /= 2;
        }

        std::array::from_fn(|j| {
            let q = base[j] * BLOCK_KEYS + slot[j];
            let found = self.blocks[base[j]].0[slot[j]];

            let common = ((found ^ key[j]).leading_zeros() / 8) as usize;
            let max_len = common.min(win[j]);

            // `max_len >= 1` and bit 0 is always set, so `fits` is never zero.
            let chain = &self.chains[q];
            let fits = u32::from(chain.lens) & ((1u32 << max_len) - 1);
            let len = (u32::BITS - fits.leading_zeros()) as usize;
            let rank = fits.count_ones() as usize - 1;

            // Rank 0 is the single-byte token of the first input byte.
            let longer = chain.ids[rank.saturating_sub(1)];
            let single = self.singles[(key[j] >> 120) as usize];
            (select_unpredictable(rank == 0, single, longer), len)
        })
    }
}

/// `at + step` if `a <= b`, else `at`, without a branch: one step of a binary
/// search whose outcome is unpredictable.
#[inline(always)]
fn advance_if_le(a: u128, b: u128, at: usize, step: usize) -> usize {
    // On x86-64 and on generic aarch64 (Graviton), LLVM turns a select on this
    // comparison into a branch, even through `select_unpredictable`. The borrow
    // bit of `b - a`, computed with plain arithmetic, stays branch-free.
    #[cfg(not(all(target_arch = "aarch64", target_vendor = "apple")))]
    {
        let diff = b.wrapping_sub(a);
        let borrow = ((!b & a) | (!(b ^ a) & diff)) >> 127;
        at + step * (1 - borrow as usize)
    }

    // On Apple Silicon the select becomes a conditional select, which costs
    // fewer instructions than the arithmetic.
    #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
    {
        select_unpredictable(a <= b, at + step, at)
    }
}

/// Big-endian key of the `win` bytes at `bytes[at..]`, zero past `win` and
/// past the end of `bytes`.
#[inline(always)]
fn window_key(bytes: &[u8], at: usize, win: usize) -> u128 {
    let raw = if at + MAX_TOKEN_SIZE <= bytes.len() {
        u128::from_be_bytes(bytes[at..at + MAX_TOKEN_SIZE].try_into().unwrap())
    } else {
        let mut buf = [0u8; MAX_TOKEN_SIZE];
        let tail = &bytes[at.min(bytes.len())..];
        let n = tail.len().min(MAX_TOKEN_SIZE);
        buf[..n].copy_from_slice(&tail[..n]);
        u128::from_be_bytes(buf)
    };
    raw & (u128::MAX << (8 * (MAX_TOKEN_SIZE - win)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dictionary::{CompactDictionary, Dictionary};

    /// Sorted, complete dictionary: every single byte plus `extra`.
    fn make_test_dictionary(extra: &[&[u8]]) -> CompactDictionary {
        let mut tokens: Vec<Vec<u8>> = (0u16..=255).map(|i| vec![i as u8]).collect();
        tokens.extend(extra.iter().map(|t| t.to_vec()));
        tokens.sort();
        tokens.dedup();

        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for t in &tokens {
            bytes.extend_from_slice(t);
            offsets.push(bytes.len() as u32);
        }
        CompactDictionary::from_raw(bytes, offsets)
    }

    fn id_of(dict: &CompactDictionary, token: &[u8]) -> Token {
        let view = dict.as_view();
        (0..view.num_tokens())
            .find(|&i| view.token(i as Token) == token)
            .expect("token not in dictionary") as Token
    }

    /// Longest dictionary token that is a prefix of `data`, by linear scan.
    fn brute_force_match(dict: &CompactDictionary, data: &[u8]) -> (Token, usize) {
        let view = dict.as_view();
        (0..view.num_tokens())
            .map(|i| (i as Token, view.token(i as Token)))
            .filter(|(_, t)| data.starts_with(t))
            .map(|(i, t)| (i, t.len()))
            .max_by_key(|&(_, len)| len)
            .expect("dictionary is complete")
    }

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn dictionary_matcher_holds_every_token() {
        let d = make_test_dictionary(&[b"ab", b"abcde"]);
        let m = DictionaryMatcher::from_dictionary(d.as_view());
        let view = d.as_view();
        assert_eq!(view.num_tokens(), 258);
        for i in 0..view.num_tokens() {
            let token = view.token(i as Token);
            assert_eq!(m.find_longest_match(token), (i as Token, token.len()));
        }
    }

    #[test]
    fn dictionary_matcher_single_bytes() {
        let d = make_test_dictionary(&[b"ab", b"a\0"]);
        let m = DictionaryMatcher::from_dictionary(d.as_view());
        for i in 0u16..=255 {
            let b = [i as u8];
            assert_eq!(m.find_longest_match(&b), (id_of(&d, &b), 1), "byte {i}");
        }
    }

    #[test]
    fn dictionary_matcher_multi_byte_token() {
        let d = make_test_dictionary(&[b"ab", b"abcde"]);
        let m = DictionaryMatcher::from_dictionary(d.as_view());
        assert_eq!(m.find_longest_match(b"abcde"), (id_of(&d, b"abcde"), 5));
        assert_eq!(m.find_longest_match(b"abc"), (id_of(&d, b"ab"), 2));
        assert_eq!(m.find_longest_match(b"abcdX"), (id_of(&d, b"ab"), 2));
        assert_eq!(m.find_longest_match(b"az"), (id_of(&d, b"a"), 1));
    }

    #[test]
    fn dictionary_matcher_long_token() {
        let d = make_test_dictionary(&[b"ABCDEFGHI", b"ABCDEFGHIJKLMNOP"]);
        let m = DictionaryMatcher::from_dictionary(d.as_view());
        assert_eq!(
            m.find_longest_match(b"ABCDEFGHIX"),
            (id_of(&d, b"ABCDEFGHI"), 9)
        );
        assert_eq!(
            m.find_longest_match(b"ABCDEFGHIJKLMNOPQ"),
            (id_of(&d, b"ABCDEFGHIJKLMNOP"), 16)
        );
    }

    #[test]
    fn dictionary_matcher_trailing_zero_bytes() {
        // Zero-padded keys tie for these tokens; the length bound must
        // separate them.
        let d = make_test_dictionary(&[b"ab", b"ab\0", b"ab\0\0x"]);
        let m = DictionaryMatcher::from_dictionary(d.as_view());
        assert_eq!(m.find_longest_match(b"ab"), (id_of(&d, b"ab"), 2));
        assert_eq!(m.find_longest_match(b"ab\0"), (id_of(&d, b"ab\0"), 3));
        assert_eq!(m.find_longest_match(b"ab\0\0"), (id_of(&d, b"ab\0"), 3));
        assert_eq!(m.find_longest_match(b"ab\0\0x"), (id_of(&d, b"ab\0\0x"), 5));
        assert_eq!(m.find_longest_match(b"a\0"), (id_of(&d, b"a"), 1));
    }

    #[test]
    fn dictionary_matcher_agrees_with_brute_force() {
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};

        // A small alphabet makes shared prefixes and nested tokens common.
        let mut rng = StdRng::seed_from_u64(7);
        let alphabet = [b'a', b'b', b'c', 0u8, 0xFF];
        let random_bytes = |rng: &mut StdRng, len: usize| -> Vec<u8> {
            (0..len)
                .map(|_| alphabet[rng.random_range(0..alphabet.len())])
                .collect()
        };

        let extra: Vec<Vec<u8>> = (0..2000)
            .map(|_| {
                let len = rng.random_range(2..=MAX_TOKEN_SIZE);
                random_bytes(&mut rng, len)
            })
            .collect();
        let extra: Vec<&[u8]> = extra.iter().map(Vec::as_slice).collect();
        let d = make_test_dictionary(&extra);
        let m = DictionaryMatcher::from_dictionary(d.as_view());

        for _ in 0..20_000 {
            let len = rng.random_range(1..=MAX_TOKEN_SIZE + 4);
            let data = random_bytes(&mut rng, len);
            assert_eq!(
                m.find_longest_match(&data),
                brute_force_match(&d, &data),
                "input {data:?}"
            );
        }
    }
}
