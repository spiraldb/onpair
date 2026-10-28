// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Frozen longest-prefix matcher for encoding. [`DictionaryMatcher`] is built
//! from the final, sorted dictionary and matches several inputs at once without
//! data-dependent branches.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm512_cmpeq_epu64_mask, _mm512_cmple_epu64_mask, _mm512_cmplt_epu64_mask, _mm512_loadu_si512,
    _mm512_set1_epi64,
};
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
/// the common prefix of `q` and `w`. Each token stores its prefix lengths and a shared path of proper prefix
/// tokens, so the last step is a mask, a rank, and a path lookup. The single-byte token
/// of `w[0]` sorts at or before `q`, so `q` always exists and shares the first
/// byte of `w`.
///
/// Keys are stored in 128-byte blocks of [`BLOCK_KEYS`], and the
/// first key of every block is copied into a small sample array that stays in
/// cache. A direct table on the first two bytes gives the range of tokens to
/// search for `q`. A binary search over the samples of that range finds the
/// block of `q`. AVX-512F finishes the sample search and searches that block
/// with eight-wide rank counts; the scalar path uses binary search throughout.
/// A search reads one block that may miss the cache.
/// All inputs of one call search with the same number of steps, so the loop has
/// no branch that depends on the data; see [`advance_if_le`].
#[derive(Debug, Clone)]
pub(crate) struct DictionaryMatcher {
    /// Token keys in dictionary order, then `u128::MAX` up to a whole block.
    blocks: Vec<KeyBlock>,
    /// First key of each block, in columns, including seven trailing pad blocks.
    samples_hi: Vec<u64>,
    samples_lo: Vec<u64>,
    /// Per 2-byte prefix `p`: the block of the last token before the tokens
    /// whose keys start with `p` (block 0 if there is none). The block of the
    /// last token with prefix `p` is then `first_blocks[p + 1]`.
    first_blocks: Vec<u16>,
    /// Prefix tokens of each token, in dictionary order, then of each pad.
    chains: Vec<PrefixChain>,
    /// Proper prefix paths, shared by tokens with the same parent.
    prefix_ids: Vec<Token>,
    single_byte_only: bool,
}

/// Keys per 128-byte block.
const BLOCK_KEYS: usize = 8;

/// One aligned block of key columns.
#[derive(Copy, Clone, Debug)]
#[repr(align(128))]
struct KeyBlock {
    hi: [u64; BLOCK_KEYS],
    lo: [u64; BLOCK_KEYS],
}

impl KeyBlock {
    #[inline(always)]
    fn key(&self, slot: usize) -> u128 {
        (u128::from(self.hi[slot]) << 64) | u128::from(self.lo[slot])
    }
}

/// A token's prefix lengths and its shared path of proper prefix tokens.
/// Siblings share the path indexed by `base`; `id` holds the token itself.
#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
struct PrefixChain {
    base: u32,
    lens: u16,
    id: Token,
}
struct PrefixPool {
    bases: Vec<u32>,
    ids: Vec<Token>,
}
impl PrefixChain {
    fn new(dict: CompactDictionaryView<'_>, chain: &[Token], pool: &mut PrefixPool) -> Self {
        let parent = chain[chain.len().saturating_sub(2)];
        let base = &mut pool.bases[parent as usize];
        if *base == u32::MAX {
            // A dictionary has at most 2^16 tokens, each with at most 16
            // ancestors, so the path offset fits in u32.
            *base = pool.ids.len() as u32;
            let count = chain.len().saturating_sub(1).max(1);
            pool.ids.extend_from_slice(&chain[..count]);
            // The full-fit rank reads this spare entry before selecting `id`.
            // Singles also share their one-entry path with their children.
            pool.ids.push(0);
        }
        let lens = chain
            .iter()
            .fold(0u16, |mask, &id| mask | (1 << (dict.token_len(id) - 1)));
        Self {
            base: *base,
            lens,
            id: *chain.last().unwrap(),
        }
    }
    #[inline(always)]
    fn code(&self, pool: &[Token], fits: u32) -> Token {
        let rank = fits.count_ones() as usize - 1;
        let proper = pool[self.base as usize + rank];
        select_unpredictable(fits == self.lens as u32, self.id, proper)
    }
}

impl DictionaryMatcher {
    /// Build a matcher from a dictionary: token at index `i` receives id `i`.
    ///
    /// Precondition: the dictionary is strictly sorted and contains every
    /// single-byte token.
    pub(crate) fn from_dictionary(dict: CompactDictionaryView<'_>) -> Self {
        let n = dict.num_tokens();
        let mut prefix_pool = PrefixPool {
            bases: vec![u32::MAX; n],
            ids: Vec::new(),
        };
        let mut keys = Vec::with_capacity(n + 1);
        let mut chains = Vec::with_capacity(n);

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

            chains.push(PrefixChain::new(dict, &chain, &mut prefix_pool));
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

        // A window of 16 `0xFF` bytes has the key `u128::MAX` too, so the
        // search can stop on a pad. Its prefix tokens are the runs of `0xFF`.
        // The last token starts with `0xFF`, so the stack holds all of them.
        chain.retain(|&id| dict.token(id).iter().all(|&b| b == 0xFF));
        chains.resize(keys.len(), PrefixChain::new(dict, &chain, &mut prefix_pool));

        // An eight-wide sample load can include seven entries after the last
        // candidate. For key == u128::MAX those entries compare <= too: there
        // is no larger sentinel. Back them with real pad blocks and the same
        // 0xFF prefix chain, so any rank still selects a valid predecessor.
        keys.resize(keys.len() + (BLOCK_KEYS - 1) * BLOCK_KEYS, u128::MAX);
        chains.resize(keys.len(), PrefixChain::new(dict, &chain, &mut prefix_pool));

        let blocks: Vec<KeyBlock> = keys
            .chunks_exact(BLOCK_KEYS)
            .map(|chunk| KeyBlock {
                hi: std::array::from_fn(|i| (chunk[i] >> 64) as u64),
                lo: std::array::from_fn(|i| chunk[i] as u64),
            })
            .collect();
        let samples_hi = blocks.iter().map(|block| block.hi[0]).collect();
        let samples_lo = blocks.iter().map(|block| block.lo[0]).collect();

        Self {
            blocks,
            samples_hi,
            samples_lo,
            first_blocks,
            chains,
            prefix_ids: prefix_pool.ids,
            single_byte_only: n == 256,
        }
    }

    /// A complete dictionary with exactly 256 tokens assigns each byte its value.
    pub(crate) fn is_single_byte_only(&self) -> bool {
        self.single_byte_only
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
        // SAFETY: the scalar specialization requires no target features.
        unsafe { self.find_all_with::<K, false>(bytes, at, win) }
    }

    /// AVX-512 specialization; the caller must detect AVX-512F and POPCNT first.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,popcnt")]
    #[inline]
    pub(crate) unsafe fn find_all_avx512<const K: usize>(
        &self,
        bytes: &[u8],
        at: &[usize; K],
        win: &[usize; K],
    ) -> [(Token, usize); K] {
        // SAFETY: this function enables AVX-512F and POPCNT.
        unsafe { self.find_all_with::<K, true>(bytes, at, win) }
    }

    /// The SIMD specialization requires AVX-512F and POPCNT on x86-64.
    #[inline(always)]
    unsafe fn find_all_with<const K: usize, const SIMD: bool>(
        &self,
        bytes: &[u8],
        at: &[usize; K],
        win: &[usize; K],
    ) -> [(Token, usize); K] {
        // The key holds all 16 bytes at `at`, also past the window. Every token
        // that is a prefix of the window is a prefix of the whole key, so it is
        // also a prefix of `q`; the length bound below drops the longer ones.
        let key: [u128; K] = std::array::from_fn(|j| window_key(bytes, at[j]));

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
        let mut steps = usize::BITS - largest.saturating_sub(1).leading_zeros();
        if SIMD {
            steps = steps.saturating_sub(3);
        }
        for _ in 0..steps {
            for j in 0..K {
                let half = size[j] / 2;
                let mid = base[j] + half;
                let sample =
                    (u128::from(self.samples_hi[mid]) << 64) | u128::from(self.samples_lo[mid]);
                base[j] = advance_if_le(sample, key[j], base[j], half);
                size[j] -= half;
            }
        }

        if SIMD {
            #[cfg(target_arch = "x86_64")]
            for j in 0..K {
                // Outside the remaining range, samples are greater than the
                // key: either an earlier comparison excluded them, or their
                // two-byte prefix is larger. MAX pads are valid blocks too.
                // SAFETY: base is a candidate block, seven trailing samples
                // make eight elements readable, its first key is <= key[j],
                // and the caller guarantees AVX-512F and POPCNT.
                base[j] += unsafe {
                    rank8(
                        self.samples_hi.as_ptr().add(base[j]),
                        self.samples_lo.as_ptr().add(base[j]),
                        key[j],
                    )
                };
            }
        }

        // Keys are sorted, the block starts with a key `<=` the input key, and
        // the next block starts with a larger one. So `q` is the last key of
        // the block that is `<=` the input key. Only MAX padding blocks can
        // follow it with equal keys; they carry the same prefix chain.
        let mut slot = [0usize; K];
        if SIMD {
            #[cfg(target_arch = "x86_64")]
            for j in 0..K {
                let block = &self.blocks[base[j]];
                // SAFETY: both columns contain eight keys, the first is <=
                // key[j], and the caller guarantees AVX-512F and POPCNT.
                slot[j] = unsafe { rank8(block.hi.as_ptr(), block.lo.as_ptr(), key[j]) };
            }
        } else {
            let mut half = BLOCK_KEYS / 2;
            while half > 0 {
                for j in 0..K {
                    let block = &self.blocks[base[j]];
                    slot[j] = advance_if_le(block.key(slot[j] + half), key[j], slot[j], half);
                }
                half /= 2;
            }
        }

        let mut q = [0usize; K];
        let mut fits_out = [0u32; K];
        let len: [usize; K] = std::array::from_fn(|j| {
            q[j] = base[j] * BLOCK_KEYS + slot[j];
            let found = self.blocks[base[j]].key(slot[j]);

            let common = ((found ^ key[j]).leading_zeros() / 8) as usize;
            let max_len = common.min(win[j]);

            // `max_len >= 1` and bit 0 is always set, so `fits` is never zero.
            let fits = u32::from(self.chains[q[j]].lens) & ((1u32 << max_len) - 1);
            fits_out[j] = fits;
            (u32::BITS - fits.leading_zeros()) as usize
        });

        // The prefix token of length `len`; a length of one byte is the
        // single-byte token of the first input byte. The token loads come after
        // all lengths: the next positions then do not wait behind them, which
        // is measurably faster on Intel and costs little elsewhere.
        std::array::from_fn(|j| {
            (
                self.chains[q[j]].code(&self.prefix_ids, fits_out[j]),
                len[j],
            )
        })
    }
}

/// Last of eight sorted keys <= `key`. Both pointers must address eight u64s,
/// and the first key must be <= `key`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,popcnt")]
#[inline]
unsafe fn rank8(hi: *const u64, lo: *const u64, key: u128) -> usize {
    // SAFETY: the caller guarantees eight readable elements in each column.
    let (hi, lo) = unsafe { (_mm512_loadu_si512(hi.cast()), _mm512_loadu_si512(lo.cast())) };
    let key_hi = _mm512_set1_epi64((key >> 64) as i64);
    let key_lo = _mm512_set1_epi64(key as i64);
    let le = _mm512_cmplt_epu64_mask(hi, key_hi)
        | (_mm512_cmpeq_epu64_mask(hi, key_hi) & _mm512_cmple_epu64_mask(lo, key_lo));
    le.count_ones() as usize - 1
}

/// `at + step` if `a <= b`, else `at`, without a branch: one step of a binary
/// search whose outcome is unpredictable.
#[inline(always)]
fn advance_if_le(a: u128, b: u128, at: usize, step: usize) -> usize {
    // On x86-64 and on generic aarch64 (Graviton), LLVM turns a select on this
    // comparison into a branch, even through `select_unpredictable`, and the
    // same arithmetic without a select costs about ten instructions. The
    // assembly is the compare that the select would be: a 128-bit subtraction
    // whose borrow picks the result.
    #[cfg(target_arch = "x86_64")]
    {
        let mut next = at + step;
        // SAFETY: only register arithmetic and flags.
        unsafe {
            std::arch::asm!(
                "cmp {b_lo}, {a_lo}",
                "sbb {b_hi}, {a_hi}",
                "cmovb {next}, {at}",
                b_lo = in(reg) b as u64,
                a_lo = in(reg) a as u64,
                b_hi = inout(reg) (b >> 64) as u64 => _,
                a_hi = in(reg) (a >> 64) as u64,
                at = in(reg) at,
                next = inout(reg) next,
                options(pure, nomem, nostack),
            );
        }
        next
    }

    #[cfg(all(target_arch = "aarch64", not(target_vendor = "apple")))]
    {
        let mut next = at + step;
        // SAFETY: only register arithmetic and flags.
        unsafe {
            std::arch::asm!(
                "cmp {b_lo}, {a_lo}",
                "sbcs xzr, {b_hi}, {a_hi}",
                "csel {next}, {next}, {at}, hs",
                b_lo = in(reg) b as u64,
                a_lo = in(reg) a as u64,
                b_hi = in(reg) (b >> 64) as u64,
                a_hi = in(reg) (a >> 64) as u64,
                at = in(reg) at,
                next = inout(reg) next,
                options(pure, nomem, nostack),
            );
        }
        next
    }

    // On Apple Silicon the select becomes a conditional select already.
    #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
    {
        select_unpredictable(a <= b, at + step, at)
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        select_unpredictable(a <= b, at + step, at)
    }
}

/// Big-endian key of the 16 bytes at `bytes[at..]`, zero past the end of
/// `bytes`.
#[inline(always)]
fn window_key(bytes: &[u8], at: usize) -> u128 {
    if at + MAX_TOKEN_SIZE <= bytes.len() {
        u128::from_be_bytes(bytes[at..at + MAX_TOKEN_SIZE].try_into().unwrap())
    } else {
        let mut buf = [0u8; MAX_TOKEN_SIZE];
        let tail = &bytes[at.min(bytes.len())..];
        let n = tail.len().min(MAX_TOKEN_SIZE);
        buf[..n].copy_from_slice(&tail[..n]);
        u128::from_be_bytes(buf)
    }
}

#[cfg(test)]
mod tests {
    use hegel::TestCase;
    use hegel::generators as gs;

    use super::*;
    use crate::core::dictionary::{CompactDictionary, Dictionary};
    use crate::test_corpus::{
        PROPERTY_ALPHABET, complete_dictionary as make_test_dictionary, trainer_matcher,
    };

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

    fn check_paths<const K: usize>(
        dict: &CompactDictionary,
        matcher: &DictionaryMatcher,
        bytes: &[u8],
        at: &[usize; K],
        win: &[usize; K],
    ) {
        let scalar = matcher.find_all(bytes, at, win);
        for j in 0..K {
            assert_eq!(
                scalar[j],
                brute_force_match(dict, &bytes[at[j]..at[j] + win[j]]),
                "lane {j}, at {at:?}, win {win:?}"
            );
        }
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("popcnt")
        {
            // SAFETY: AVX-512F and POPCNT were detected above.
            let simd = unsafe { matcher.find_all_avx512(bytes, at, win) };
            assert_eq!(simd, scalar);
        }
    }

    #[hegel::test(test_cases = 300)]
    fn scalar_and_simd_agree(tc: TestCase) {
        let bytes = || gs::vecs(gs::sampled_from(&PROPERTY_ALPHABET[..]));
        let mut extra: Vec<Vec<u8>> =
            tc.draw(gs::vecs(bytes().min_size(2).max_size(MAX_TOKEN_SIZE)).max_size(2000));
        // Concentrate tokens under one prefix to exercise long sample searches.
        if tc.draw(gs::booleans()) {
            for token in &mut extra {
                token[..2].copy_from_slice(b"ab");
            }
        }
        extra.extend([
            b"a\0".to_vec(),
            b"a\0\0".to_vec(),
            vec![0; MAX_TOKEN_SIZE],
            vec![0xFF; MAX_TOKEN_SIZE],
        ]);
        let extra: Vec<&[u8]> = extra.iter().map(Vec::as_slice).collect();
        let dict = make_test_dictionary(&extra);
        let matcher = DictionaryMatcher::from_dictionary(dict.as_view());

        let mut data: Vec<u8> = tc.draw(bytes().min_size(1).max_size(80));
        data.extend_from_slice(b"a\0\0");
        data.extend_from_slice(&[0xFF; MAX_TOKEN_SIZE + 1]);
        let random_at = tc.draw(gs::integers::<usize>().max_value(data.len() - 1));
        let at = [
            0,
            random_at,
            data.len() - 17,
            data.len() - 16,
            data.len() - 2,
            data.len() - 1,
        ];
        let win = std::array::from_fn(|j| (data.len() - at[j]).min(MAX_TOKEN_SIZE));
        check_paths(&dict, &matcher, &data, &at, &win);

        // Every token, including tied zero-padded keys and 16-byte tokens,
        // must be found with both full and shorter length bounds.
        let view = dict.as_view();
        let id = tc.draw(gs::integers::<usize>().max_value(view.num_tokens() - 1)) as Token;
        let token = view.token(id);
        let bound = tc.draw(gs::integers::<usize>().min_value(1).max_value(token.len()));
        check_paths(&dict, &matcher, token, &[0, 0], &[token.len(), bound]);
    }

    #[test]
    fn scalar_and_simd_match_tied_keys_and_ff_pads() {
        // Cover every final-block occupancy, including an entirely padded block.
        for extra_len in 0..BLOCK_KEYS * 2 {
            let mut extra: Vec<Vec<u8>> = (2..=MAX_TOKEN_SIZE).map(|len| vec![0xFF; len]).collect();
            extra.extend((0..extra_len).map(|i| vec![b'a', i as u8]));
            let extra: Vec<&[u8]> = extra.iter().map(Vec::as_slice).collect();
            let dict = make_test_dictionary(&extra);
            let matcher = DictionaryMatcher::from_dictionary(dict.as_view());
            let data = [0xFF; MAX_TOKEN_SIZE + 4];
            let at = std::array::from_fn::<_, { MAX_TOKEN_SIZE + 4 }, _>(|i| i);
            let win = std::array::from_fn(|i| (data.len() - i).min(MAX_TOKEN_SIZE));
            check_paths(&dict, &matcher, &data, &at, &win);
            check_paths(&dict, &matcher, b"a\0\0", &[0, 0, 0], &[1, 2, 3]);
        }
    }

    #[test]
    fn sample_padding_preserves_short_ff_chain() {
        for extra_len in 0..BLOCK_KEYS {
            let mut extra = vec![vec![0xFF; 2], vec![0xFF, 0xFF, 0xFF, b'a']];
            extra.extend((0..extra_len).map(|i| vec![b'a', i as u8]));
            let extra: Vec<&[u8]> = extra.iter().map(Vec::as_slice).collect();
            let dict = make_test_dictionary(&extra);
            let matcher = DictionaryMatcher::from_dictionary(dict.as_view());
            let data = [0xFF; MAX_TOKEN_SIZE];
            check_paths(&dict, &matcher, &data, &[0, 0, 0], &[1, 2, MAX_TOKEN_SIZE]);

            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("popcnt")
            {
                let last = matcher.blocks.len() - BLOCK_KEYS;
                // SAFETY: seven samples follow `last`, AVX-512F and POPCNT were detected,
                // and every key is <= u128::MAX.
                let rank = unsafe {
                    rank8(
                        matcher.samples_hi.as_ptr().add(last),
                        matcher.samples_lo.as_ptr().add(last),
                        u128::MAX,
                    )
                };
                assert_eq!(rank, BLOCK_KEYS - 1);
            }
        }
    }

    #[test]
    fn sample_rank_handles_full_dictionary_and_large_prefix_range() {
        let extra: Vec<Vec<u8>> = (0..65_280u16)
            .map(|i| vec![b'a', b'b', (i >> 8) as u8, i as u8])
            .collect();
        let extra: Vec<&[u8]> = extra.iter().map(Vec::as_slice).collect();
        let dict = make_test_dictionary(&extra);
        assert_eq!(dict.num_tokens(), 1 << 16);
        let matcher = DictionaryMatcher::from_dictionary(dict.as_view());
        assert_eq!(std::mem::size_of::<KeyBlock>(), 128);
        assert_eq!(std::mem::align_of::<KeyBlock>(), 128);

        for suffix in [
            0, 1, 7, 8, 9, 63, 64, 65, 255, 256, 32_767, 65_279, 65_535u16,
        ] {
            let data = [b'a', b'b', (suffix >> 8) as u8, suffix as u8];
            check_paths(&dict, &matcher, &data, &[0, 0, 1, 2, 3], &[4, 2, 3, 2, 1]);
        }
        check_paths(
            &dict,
            &matcher,
            &[0xFF; MAX_TOKEN_SIZE],
            &[0],
            &[MAX_TOKEN_SIZE],
        );
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
    fn dictionary_matcher_all_ff_window() {
        // A window of 16 `0xFF` bytes has the same key as the pads after the
        // last token. The token counts leave the last block part empty.
        for extra in [
            &[b"ab".as_slice()][..],
            &[b"ab", b"\xFF\xFF", b"\xFF\xFF\xFFa"],
        ] {
            let d = make_test_dictionary(extra);
            assert_ne!(d.num_tokens() % BLOCK_KEYS, 0);
            let m = DictionaryMatcher::from_dictionary(d.as_view());
            for len in 1..=MAX_TOKEN_SIZE + 4 {
                let data = vec![0xFF; len];
                assert_eq!(
                    m.find_longest_match(&data),
                    brute_force_match(&d, &data),
                    "{len} bytes"
                );
            }
        }
    }

    #[test]
    fn dictionary_matcher_ignores_bytes_past_window() {
        // The key holds the bytes after a short window too. They extend
        // tokens that are longer than the window, and that must not change
        // the match.
        let d = make_test_dictionary(&[b"ab", b"abc", b"abcd", b"b\xFF", b"\xFF\xFF"]);
        let m = DictionaryMatcher::from_dictionary(d.as_view());
        for (data, win) in [
            (&b"abcd"[..], 1),
            (b"abcd", 2),
            (b"abcd", 3),
            (b"abcdx", 4),
            (b"b\xFF\xFF", 1),
            (b"\xFF\xFF\xFF", 1),
        ] {
            let window = &data[..win];
            assert_eq!(
                m.find_all(data, &[0], &[win])[0],
                brute_force_match(&d, window),
                "{window:?}"
            );
        }
    }

    #[hegel::test(test_cases = 500)]
    fn dictionary_matcher_agrees_with_trainer_matcher(tc: TestCase) {
        let bytes = || gs::vecs(gs::sampled_from(&PROPERTY_ALPHABET[..]));
        let extra: Vec<Vec<u8>> =
            tc.draw(gs::vecs(bytes().min_size(2).max_size(MAX_TOKEN_SIZE)).max_size(300));
        // Runs of one byte reach the pad key: 16 bytes of `0xFF`.
        let data: Vec<u8> = if tc.draw(gs::booleans()) {
            let byte = tc.draw(gs::sampled_from(&PROPERTY_ALPHABET[..]));
            let len = tc.draw(
                gs::integers::<usize>()
                    .min_value(1)
                    .max_value(MAX_TOKEN_SIZE + 4),
            );
            vec![byte; len]
        } else {
            tc.draw(bytes().min_size(1).max_size(MAX_TOKEN_SIZE + 4))
        };
        let win = tc.draw(
            gs::integers::<usize>()
                .min_value(1)
                .max_value(data.len().min(MAX_TOKEN_SIZE)),
        );

        let extra: Vec<&[u8]> = extra.iter().map(Vec::as_slice).collect();
        let d = make_test_dictionary(&extra);
        let view = d.as_view();
        let m = DictionaryMatcher::from_dictionary(view);
        let (trainer, trainer_tokens) = trainer_matcher(view);

        // The window is the first `win` bytes; the bytes after it are in
        // `data` too, so the key holds them.
        let window = &data[..win];
        let (token, len) = m.find_all(&data, &[0], &[win])[0];
        assert_eq!((token, len), brute_force_match(&d, window));

        let (expected, expected_len) = trainer.find_longest_match(window);
        assert_eq!(len, expected_len);
        assert_eq!(view.token(token), trainer_tokens[usize::from(expected)]);
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

#[cfg(test)]
#[path = "dictionary_matcher_rank_tests.rs"]
mod rank_tests;
