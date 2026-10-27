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
    keys: Vec<u128>,
    /// Dictionary indices delimiting every two-byte prefix, including 65536.
    starts: Vec<u32>,
    /// Bytes 2..6 as big-endian u32s, followed by fifteen readable pads.
    mid: Vec<u32>,
    /// First token in the equal-six-byte run, or MAX when no token exceeds six bytes.
    tie_starts: Vec<u32>,
    /// Per prefix: sample offset shifted left two bits, then the level count.
    sample_info: Vec<u32>,
    /// Range-local sample levels, highest first; each node has 16 separators.
    level_samples: Vec<u32>,
    /// First key of each block.
    samples: Vec<u128>,
    /// Per 2-byte prefix `p`: the block of the last token before the tokens
    /// whose keys start with `p` (block 0 if there is none). The block of the
    /// last token with prefix `p` is then `first_blocks[p + 1]`.
    first_blocks: Vec<u16>,
    /// Prefix tokens of each token, in dictionary order, then of each pad.
    chains: Vec<PrefixChain>,
    /// Token of each single byte: the first prefix token of every chain.
    singles: [Token; 256],
}

/// Keys per block: one 128-byte cache line.
const BLOCK_KEYS: usize = 8;
const SAMPLE_STRIDES: [usize; 3] = [16, 272, 4624];

/// The prefix tokens of one token `t` by length bound, in one 32-byte record,
/// so that the match needs one load after the search.
#[derive(Copy, Clone, Debug, Default)]
#[repr(align(32))]
struct PrefixChain {
    /// Bit `k` is set when the first `k + 1` bytes of `t` are a token.
    lens: u16,
    /// Entry `k` is the longest token that is a prefix of `t` and not longer
    /// than `k + 2` bytes. A bound of one byte gives the single-byte token.
    ids: [Token; MAX_TOKEN_SIZE - 1],
}

impl PrefixChain {
    /// Record of `chain`: prefix tokens of one token, shortest first, starting
    /// with a single byte.
    fn new(dict: CompactDictionaryView<'_>, chain: &[Token]) -> Self {
        let mut record = Self::default();
        for &prefix in chain {
            let len = dict.token_len(prefix);
            record.lens |= 1 << (len - 1);
            record.ids[len.max(2) - 2..].fill(prefix);
        }
        record
    }
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

            chains.push(PrefixChain::new(dict, &chain));
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
        chains.resize(keys.len(), PrefixChain::new(dict, &chain));

        let samples = keys
            .chunks_exact(BLOCK_KEYS)
            .map(|chunk| chunk[0])
            .collect();
        let mut mid: Vec<u32> = keys[..n].iter().map(|&key| (key >> 80) as u32).collect();
        mid.resize(n + 15, u32::MAX);
        let mut tie_starts = vec![u32::MAX; n];
        let mut first = 0;
        while first < n {
            let mut end = first + 1;
            while end < n && keys[end] >> 80 == keys[first] >> 80 {
                end += 1;
            }
            if (first..end).any(|i| dict.token_len(i as Token) > 6) {
                tie_starts[first..end].fill(first as u32);
            }
            first = end;
        }
        let starts: Vec<u32> = starts.into_iter().map(|i| i as u32).collect();
        // Sixteen separators select seventeen children. Final leaves hold
        // sixteen tokens, giving capacities 16, 272, 4624, and 78608.
        let mut sample_info = vec![0u32; PREFIX_SLOTS];
        let mut level_samples = vec![u32::MAX; 16];
        for p in 0..PREFIX_SLOTS {
            let start = starts[p] as usize;
            let n = starts[p + 1] as usize - start;
            let mut capacity = 16;
            let mut levels = 0;
            while capacity < n {
                capacity *= 17;
                levels += 1;
            }
            if levels == 0 {
                continue;
            }
            sample_info[p] = ((level_samples.len() as u32) << 2) | levels as u32;
            for level in (0..levels).rev() {
                let stride = SAMPLE_STRIDES[level];
                let capacity = stride * 17;
                for node in 0..n.div_ceil(capacity) {
                    for separator in 1..=16 {
                        let index = node * capacity + separator * stride;
                        level_samples.push(if index < n {
                            mid[start + index]
                        } else {
                            u32::MAX
                        });
                    }
                }
            }
        }
        level_samples.resize(level_samples.len() + 15, u32::MAX);

        Self {
            keys,
            starts,
            mid,
            tie_starts,
            sample_info,
            level_samples,
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
                let block = &self.keys[base[j] * BLOCK_KEYS..][..BLOCK_KEYS];
                slot[j] = advance_if_le(block[slot[j] + half], key[j], slot[j], half);
            }
            half /= 2;
        }

        let q = std::array::from_fn(|j| base[j] * BLOCK_KEYS + slot[j]);
        self.finish(&key, &q, win)
    }

    #[inline(always)]
    fn finish<const K: usize>(
        &self,
        key: &[u128; K],
        q: &[usize; K],
        win: &[usize; K],
    ) -> [(Token, usize); K] {
        let len: [usize; K] = std::array::from_fn(|j| {
            let found = self.keys[q[j]];

            let common = ((found ^ key[j]).leading_zeros() / 8) as usize;
            let max_len = common.min(win[j]);

            // `max_len >= 1` and bit 0 is always set, so `fits` is never zero.
            let fits = u32::from(self.chains[q[j]].lens) & ((1u32 << max_len) - 1);
            (u32::BITS - fits.leading_zeros()) as usize
        });

        // The prefix token of length `len`; a length of one byte is the
        // single-byte token of the first input byte. The token loads come after
        // all lengths: the next positions then do not wait behind them, which
        // is measurably faster on Intel and costs little elsewhere.
        std::array::from_fn(|j| {
            let longer = self.chains[q[j]].ids[len[j].max(2) - 2];
            let single = self.singles[(key[j] >> 120) as usize];
            (select_unpredictable(len[j] == 1, single, longer), len[j])
        })
    }
    /// Scalar middle-key variant, also used to validate the SIMD variants.
    #[inline(always)]
    pub(crate) fn find_all_middle<const K: usize>(
        &self,
        bytes: &[u8],
        at: &[usize; K],
        win: &[usize; K],
    ) -> [(Token, usize); K] {
        // SAFETY: mode one requires no target features.
        unsafe { self.find_all_rank::<K, 1>(bytes, at, win) }
    }

    /// The caller must detect AVX-512F and POPCNT.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,popcnt")]
    #[inline]
    pub(crate) unsafe fn find_all_avx512<const K: usize>(
        &self,
        bytes: &[u8],
        at: &[usize; K],
        win: &[usize; K],
    ) -> [(Token, usize); K] {
        // SAFETY: this function enables both required features.
        unsafe { self.find_all_rank::<K, 3>(bytes, at, win) }
    }

    /// The caller must detect AVX2 and POPCNT.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,popcnt")]
    #[inline]
    pub(crate) unsafe fn find_all_avx2<const K: usize>(
        &self,
        bytes: &[u8],
        at: &[usize; K],
        win: &[usize; K],
    ) -> [(Token, usize); K] {
        // SAFETY: this function enables both required features.
        unsafe { self.find_all_rank::<K, 2>(bytes, at, win) }
    }

    /// The caller must detect NEON. Kept available for ARM measurement.
    #[cfg(all(test, target_arch = "aarch64"))]
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn find_all_neon<const K: usize>(
        &self,
        bytes: &[u8],
        at: &[usize; K],
        win: &[usize; K],
    ) -> [(Token, usize); K] {
        // SAFETY: this function enables NEON.
        unsafe { self.find_all_rank::<K, 4>(bytes, at, win) }
    }

    /// Modes two and three require their respective SIMD ISA and POPCNT.
    #[inline(always)]
    unsafe fn find_all_rank<const K: usize, const MODE: u8>(
        &self,
        bytes: &[u8],
        at: &[usize; K],
        win: &[usize; K],
    ) -> [(Token, usize); K] {
        let key: [u128; K] = std::array::from_fn(|j| window_key(bytes, at[j]));
        let start: [usize; K] =
            std::array::from_fn(|j| self.starts[(key[j] >> 112) as usize] as usize);
        let n: [usize; K] =
            std::array::from_fn(|j| self.starts[(key[j] >> 112) as usize + 1] as usize - start[j]);
        let w: [u32; K] = std::array::from_fn(|j| (key[j] >> 80) as u32);
        let last = self.starts[PREFIX_SLOTS] as usize - 1;
        let info: [u32; K] = std::array::from_fn(|j| self.sample_info[(key[j] >> 112) as usize]);
        let depth: [usize; K] = std::array::from_fn(|j| (info[j] & 3) as usize);
        let roots: [usize; K] = std::array::from_fn(|j| (info[j] >> 2) as usize);
        let mut offset = [0usize; K];
        let mut node = [0usize; K];
        let largest = depth.iter().copied().max().unwrap_or(0);
        if largest > 2 {
            // SAFETY: target features and complete sample nodes are provided.
            unsafe {
                self.descend::<K, MODE, 4624, 2>(&n, &w, &depth, &roots, &mut offset, &mut node)
            };
        }
        if largest > 1 {
            // SAFETY: target features and complete sample nodes are provided.
            unsafe {
                self.descend::<K, MODE, 272, 1>(&n, &w, &depth, &roots, &mut offset, &mut node)
            };
        }
        if largest > 0 {
            // SAFETY: target features and complete sample nodes are provided.
            unsafe {
                self.descend::<K, MODE, 16, 0>(&n, &w, &depth, &roots, &mut offset, &mut node)
            };
        }
        let mut q = [0usize; K];
        for j in 0..K {
            let offset = offset[j];
            let valid = ((1u32 << n[j].saturating_sub(offset).min(16)) - 1) as u16;
            // Empty ranges can start at the end of the array. Clamp their
            // zero-masked load so fifteen trailing pads are sufficient.
            let load_at = (start[j] + offset).min(last);
            // SAFETY: load_at is at most the last token, followed by fifteen
            // readable pads; the caller guarantees the selected ISA.
            let count = unsafe { rank16::<MODE>(self.mid.as_ptr().add(load_at), w[j], valid) };
            // Prefix zero contains the key zero, so its count cannot be zero.
            q[j] = start[j] + offset + count - 1;
            if count > 0 && self.mid[q[j]] == w[j] && self.tie_starts[q[j]] != u32::MAX {
                let mut base = (self.tie_starts[q[j]] as usize).saturating_sub(1);
                let mut size = q[j] + 1 - base;
                let steps = usize::BITS - size.saturating_sub(1).leading_zeros();
                for _ in 0..steps {
                    let half = size / 2;
                    base = advance_if_le(self.keys[base + half], key[j], base, half);
                    size -= half;
                }
                q[j] = base;
            }
        }
        self.finish(&key, &q, win)
    }
    /// A fixed stride lets the compiler replace integer division with constants.
    #[inline(always)]
    unsafe fn descend<const K: usize, const MODE: u8, const STRIDE: usize, const LEVEL: usize>(
        &self,
        n: &[usize; K],
        w: &[u32; K],
        depth: &[usize; K],
        roots: &[usize; K],
        offset: &mut [usize; K],
        node: &mut [usize; K],
    ) {
        for j in 0..K {
            let active = usize::from(depth[j] > LEVEL);
            let preceding = if LEVEL == 0 {
                16 * usize::from(depth[j] > 1)
                    + 16 * n[j].div_ceil(4624) * usize::from(depth[j] > 2)
            } else if LEVEL == 1 {
                16 * usize::from(depth[j] > 2)
            } else {
                0
            };
            let load_at = select_unpredictable(active != 0, roots[j] + preceding + node[j] * 16, 0);
            let valid =
                ((1u32 << (n[j].saturating_sub(offset[j] + 1) / STRIDE).min(16)) - 1) as u16;
            let valid = valid & 0u16.wrapping_sub(active as u16);
            // SAFETY: active nodes contain sixteen entries; inactive lanes
            // use the zero-masked sentinel. The caller guarantees the ISA.
            let count =
                unsafe { rank16::<MODE>(self.level_samples.as_ptr().add(load_at), w[j], valid) };
            offset[j] += count * STRIDE;
            node[j] = (node[j] * 17 + count) * active;
        }
    }
}

/// Both the pointer and the ISA are guaranteed by the matcher specialization.
#[inline(always)]
unsafe fn rank16<const MODE: u8>(ptr: *const u32, w: u32, valid: u16) -> usize {
    #[cfg(target_arch = "x86_64")]
    if MODE == 3 {
        // SAFETY: inherited pointer and AVX-512F/POPCNT preconditions.
        return unsafe { rank16_avx512(ptr, w, valid) };
    }
    #[cfg(target_arch = "x86_64")]
    if MODE == 2 {
        // SAFETY: inherited pointer and AVX2/POPCNT preconditions.
        return unsafe { rank16_avx2(ptr, w, valid) };
    }
    #[cfg(all(test, target_arch = "aarch64"))]
    if MODE == 4 {
        // SAFETY: inherited pointer and NEON preconditions.
        return unsafe { rank16_neon(ptr, w, valid) };
    }
    let mut count = 0;
    for i in 0..16 {
        // SAFETY: the caller provides sixteen readable u32 entries.
        count += usize::from(unsafe { *ptr.add(i) } <= w && valid & (1 << i) != 0);
    }
    count
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,popcnt")]
#[inline]
unsafe fn rank16_avx512(ptr: *const u32, w: u32, valid: u16) -> usize {
    // SAFETY: the caller provides sixteen readable u32 entries.
    let keys = unsafe { std::arch::x86_64::_mm512_loadu_si512(ptr.cast()) };
    let w = std::arch::x86_64::_mm512_set1_epi32(w as i32);
    std::arch::x86_64::_mm512_mask_cmple_epu32_mask(valid, keys, w).count_ones() as usize
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,popcnt")]
#[inline]
unsafe fn rank16_avx2(ptr: *const u32, w: u32, valid: u16) -> usize {
    let bias = std::arch::x86_64::_mm256_set1_epi32(i32::MIN);
    let w = std::arch::x86_64::_mm256_set1_epi32((w ^ 0x8000_0000) as i32);
    // SAFETY: the caller provides sixteen readable u32 entries.
    let (a, b) = unsafe {
        (
            std::arch::x86_64::_mm256_loadu_si256(ptr.cast()),
            std::arch::x86_64::_mm256_loadu_si256(ptr.add(8).cast()),
        )
    };
    let a = std::arch::x86_64::_mm256_xor_si256(a, bias);
    let b = std::arch::x86_64::_mm256_xor_si256(b, bias);
    let a = std::arch::x86_64::_mm256_cmpgt_epi32(a, w);
    let b = std::arch::x86_64::_mm256_cmpgt_epi32(b, w);
    let gt = std::arch::x86_64::_mm256_movemask_ps(std::arch::x86_64::_mm256_castsi256_ps(a))
        as u16
        | ((std::arch::x86_64::_mm256_movemask_ps(std::arch::x86_64::_mm256_castsi256_ps(b))
            as u16)
            << 8);
    ((!gt) & valid).count_ones() as usize
}

#[cfg(all(test, target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn rank16_neon(ptr: *const u32, w: u32, valid: u16) -> usize {
    let mut count = 0;
    let w = std::arch::aarch64::vdupq_n_u32(w);
    for group in 0..4 {
        // SAFETY: the caller provides sixteen readable entries.
        let keys = unsafe { std::arch::aarch64::vld1q_u32(ptr.add(group * 4)) };
        let mask: [u32; 4] = std::array::from_fn(|i| u32::from((valid >> (group * 4 + i)) & 1));
        // SAFETY: mask has four readable entries.
        let mask = unsafe { std::arch::aarch64::vld1q_u32(mask.as_ptr()) };
        let le = std::arch::aarch64::vshrq_n_u32::<31>(std::arch::aarch64::vcleq_u32(keys, w));
        count += std::arch::aarch64::vaddvq_u32(std::arch::aarch64::vandq_u32(le, mask)) as usize;
    }
    count
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
#[path = "dictionary_matcher_tests.rs"]
mod rank_tests;
