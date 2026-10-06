// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Longest-prefix matcher: maps byte sequences (`1..=MAX_TOKEN_SIZE` bytes) to
//! token ids and answers "what is the longest dictionary token that is a prefix
//! of this input?".
//!
//! Two-tier storage:
//!   * **short map** — tokens of length `1..=8` keyed by their bytes packed into
//!     a `u64` plus the length.
//!   * **long map** — tokens of length `9..=16` bucketed by their 8-byte prefix.
//!     Each bucket holds the `(suffix, length, token)` triples sharing that
//!     prefix and is searched for the longest matching suffix. A bucket starts
//!     as a vector and switches to length-grouped binary search once it grows.
//!
//! Matching loads up to 16 bytes once, probes the long-token bucket, then checks
//! short tokens from longest to shortest.
//!
//! [`LongestPrefixMatcher`] accepts inserts and drives training. Encoding uses
//! [`DictionaryMatcher`], a frozen form built from the final, sorted
//! dictionary. It matches several inputs at once without data-dependent
//! branches.

use std::hint::select_unpredictable;

use hashbrown::HashMap;

use crate::core::dictionary::{CompactDictionaryView, DictionaryView};
use crate::core::types::{MAX_TOKEN_SIZE, Token};

/// Tokens of this length or shorter live in the short map; longer tokens are
/// bucketed by their first `BUCKET_PREFIX_LEN` bytes.
const BUCKET_PREFIX_LEN: usize = 8;

const MAX_SUFFIX_LEN: usize = MAX_TOKEN_SIZE - BUCKET_PREFIX_LEN;
const PROMOTE_THRESHOLD: usize = 48;

#[inline]
fn load_window(data: &[u8]) -> (u64, u64) {
    let n = data.len();
    if n >= MAX_TOKEN_SIZE {
        return (
            u64::from_le_bytes(data[..8].try_into().unwrap()),
            u64::from_le_bytes(data[8..16].try_into().unwrap()),
        );
    }
    if n >= 8 {
        let lo = u64::from_le_bytes(data[..8].try_into().unwrap());
        let hi = if n > 8 {
            u64::from_le_bytes(data[n - 8..].try_into().unwrap()) >> ((MAX_TOKEN_SIZE - n) * 8)
        } else {
            0
        };
        return (lo, hi);
    }
    let lo = if n >= 4 {
        u32::from_le_bytes(data[..4].try_into().unwrap()) as u64
            | (u32::from_le_bytes(data[n - 4..].try_into().unwrap()) as u64) << ((n - 4) * 8)
    } else if n >= 2 {
        u16::from_le_bytes(data[..2].try_into().unwrap()) as u64
            | (u16::from_le_bytes(data[n - 2..].try_into().unwrap()) as u64) << ((n - 2) * 8)
    } else {
        data[0] as u64
    };
    (lo, 0)
}

/// Pack the low `min(len, data.len(), 8)` bytes of `data` into a little-endian
/// `u64`; higher bytes read as zero. The full-8-byte case is a single load.
#[inline]
fn load_le_u64(data: &[u8], len: usize) -> u64 {
    if len >= BUCKET_PREFIX_LEN && data.len() >= BUCKET_PREFIX_LEN {
        return u64::from_le_bytes(data[..BUCKET_PREFIX_LEN].try_into().unwrap());
    }
    let mut buf = [0u8; 8];
    let n = len.min(data.len());
    buf[..n].copy_from_slice(&data[..n]);
    u64::from_le_bytes(buf)
}

/// Mask of the low `len * 8` bits in a `u64`.
#[inline]
fn mask_u64(len: usize) -> u64 {
    if len >= 8 {
        u64::MAX
    } else {
        (1u64 << (len * 8)) - 1
    }
}

/// One long-token entry within a bucket: the suffix bytes after the shared
/// 8-byte prefix (`slen` of them, packed little-endian and masked to that
/// length) and the token id.
#[derive(Copy, Clone, Debug)]
struct LongEntry {
    suffix: u64,
    slen: u8,
    token: Token,
}

/// Long tokens with the same 8-byte prefix.
#[derive(Debug, Clone)]
enum Bucket {
    Linear(Vec<LongEntry>),
    Grouped(Box<GroupedBucket>),
}

#[inline]
fn search_linear(entries: &[LongEntry], val: u64, max_slen: usize) -> Option<(Token, usize)> {
    for e in entries {
        let elen = e.slen as usize;
        // Matching low bytes = trailing-zero bytes of the XOR.
        if elen <= max_slen && ((val ^ e.suffix).trailing_zeros() >> 3) as usize >= elen {
            return Some((e.token, elen));
        }
    }
    None
}

/// Entries grouped by suffix length and sorted by suffix within each group.
#[derive(Debug, Clone)]
struct GroupedBucket {
    entries: Vec<LongEntry>,
    ends: [u32; MAX_SUFFIX_LEN + 2],
    present: u16,
}

impl GroupedBucket {
    fn build(entries: &[LongEntry]) -> Self {
        let mut sorted = entries.to_vec();
        sorted.sort_unstable_by(|a, b| b.slen.cmp(&a.slen).then(a.suffix.cmp(&b.suffix)));

        let mut ends = [0u32; MAX_SUFFIX_LEN + 2];
        let mut present = 0u16;
        let mut counts = [0u32; MAX_SUFFIX_LEN + 1];
        for e in &sorted {
            counts[e.slen as usize] += 1;
            present |= 1u16 << e.slen;
        }
        let mut acc = 0u32;
        for slen in (1..=MAX_SUFFIX_LEN).rev() {
            acc += counts[slen];
            ends[slen] = acc;
        }
        Self {
            entries: sorted,
            ends,
            present,
        }
    }

    fn insert(&mut self, entry: LongEntry) {
        let slen = entry.slen as usize;
        let start = self.ends[slen + 1] as usize;
        let end = self.ends[slen] as usize;
        let pos = start + self.entries[start..end].partition_point(|e| e.suffix < entry.suffix);
        self.entries.insert(pos, entry);
        for e in &mut self.ends[1..=slen] {
            *e += 1;
        }
        self.present |= 1u16 << slen;
    }

    #[inline]
    fn find(&self, val: u64, max_slen: usize) -> Option<(Token, usize)> {
        let mut lens = self.present & ((1u16 << (max_slen + 1)) - 1);
        while lens != 0 {
            let slen = (u16::BITS - 1 - lens.leading_zeros()) as usize;
            lens &= !(1u16 << slen);

            let group = &self.entries[self.ends[slen + 1] as usize..self.ends[slen] as usize];
            let target = val & mask_u64(slen);
            if let Ok(i) = group.binary_search_by_key(&target, |e| e.suffix) {
                return Some((group[i].token, slen));
            }
        }
        None
    }
}

/// Maps byte sequences (`1..=MAX_TOKEN_SIZE` bytes) to [`Token`] ids. Always
/// holds the 256 single-byte tokens after construction, so
/// [`find_longest_match`](Self::find_longest_match) is total.
#[derive(Default, Debug, Clone)]
pub(crate) struct LongestPrefixMatcher {
    /// Length `1..=8` tokens keyed by (low-`len`-byte u64, length).
    short_map: HashMap<(u64, u8), Token>,
    /// Length `9..=16` tokens bucketed by their 8-byte prefix.
    long_map: HashMap<u64, Bucket>,
    /// Longest short-map token length present (`1..=8`).
    max_short_len: u8,
    /// Next id to assign. `u32` so the full 16-bit token space (65 536 entries)
    /// is representable without overflow.
    next_id: u32,
}

impl LongestPrefixMatcher {
    /// Pre-inserts the 256 single-byte tokens with ids `0..=255`.
    pub(crate) fn new() -> Self {
        let mut short_map = HashMap::with_capacity(256);
        for i in 0u16..=255 {
            short_map.insert((i as u64, 1u8), i);
        }
        Self {
            short_map,
            long_map: HashMap::new(),
            max_short_len: 1,
            next_id: 256,
        }
    }

    /// Reserve training-time maps for the configured dictionary budget.
    pub(crate) fn reserve(&mut self, token_capacity: usize) {
        self.short_map
            .reserve(token_capacity.saturating_sub(self.short_map.len()));
        let long_capacity = (token_capacity / 4).max(16);
        self.long_map
            .reserve(long_capacity.saturating_sub(self.long_map.len()));
    }

    /// Insert `data` and assign it the next available token id.
    ///
    /// Precondition: `1 <= data.len() <= MAX_TOKEN_SIZE` and `size() < 65_536`.
    pub(crate) fn insert(&mut self, data: &[u8]) -> Token {
        let id = self.next_id as Token;
        self.next_id += 1;
        self.insert_internal(data, id);
        id
    }

    #[inline]
    fn insert_internal(&mut self, data: &[u8], id: Token) {
        debug_assert!(!data.is_empty() && data.len() <= MAX_TOKEN_SIZE);
        let len = data.len();
        if len <= BUCKET_PREFIX_LEN {
            let key = load_le_u64(data, len);
            self.short_map.insert((key, len as u8), id);
            self.max_short_len = self.max_short_len.max(len as u8);
            return;
        }

        let prefix = load_le_u64(data, BUCKET_PREFIX_LEN);
        let slen = len - BUCKET_PREFIX_LEN;
        let suffix = load_le_u64(&data[BUCKET_PREFIX_LEN..], slen);
        let entry = LongEntry {
            suffix,
            slen: slen as u8,
            token: id,
        };
        let bucket = self
            .long_map
            .entry(prefix)
            .or_insert_with(|| Bucket::Linear(Vec::new()));
        match bucket {
            Bucket::Linear(entries) => {
                let pos = entries.partition_point(|e| e.slen > entry.slen);
                entries.insert(pos, entry);
                if entries.len() > PROMOTE_THRESHOLD {
                    *bucket = Bucket::Grouped(Box::new(GroupedBucket::build(entries)));
                }
            }
            Bucket::Grouped(grouped) => grouped.insert(entry),
        }
    }

    /// Longest token whose bytes are a prefix of `data`, with that prefix's
    /// length.
    ///
    /// Precondition: `!data.is_empty()` and the matcher contains every
    /// single-byte token (always true after [`new`](Self::new)).
    #[inline]
    pub(crate) fn find_longest_match(&self, data: &[u8]) -> (Token, usize) {
        let (lo64, hi64) = load_window(data);
        let win = data.len().min(MAX_TOKEN_SIZE);

        if win > BUCKET_PREFIX_LEN
            && !self.long_map.is_empty()
            && let Some(bucket) = self.long_map.get(&lo64)
        {
            let max_slen = win - BUCKET_PREFIX_LEN;
            let hit = match bucket {
                Bucket::Linear(entries) => {
                    search_linear(entries, hi64 & mask_u64(max_slen), max_slen)
                }
                Bucket::Grouped(grouped) => grouped.find(hi64, max_slen),
            };
            if let Some((t, slen)) = hit {
                return (t, BUCKET_PREFIX_LEN + slen);
            }
        }

        let short_max = win.min(self.max_short_len as usize);
        for len in (1..=short_max).rev() {
            if let Some(&t) = self.short_map.get(&(lo64 & mask_u64(len), len as u8)) {
                return (t, len);
            }
        }
        unreachable!("LPM precondition: every single-byte token must be present")
    }

    /// Number of tokens currently in the matcher.
    #[inline]
    pub(crate) fn size(&self) -> usize {
        self.next_id as usize
    }
}

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
    // On x86-64, LLVM turns a select on this comparison into a branch, even
    // through `select_unpredictable`. The borrow bit of `b - a`, computed with
    // plain arithmetic, stays branch-free.
    #[cfg(target_arch = "x86_64")]
    {
        let diff = b.wrapping_sub(a);
        let borrow = ((!b & a) | (!(b ^ a) & diff)) >> 127;
        at + step * (1 - borrow as usize)
    }

    // Elsewhere the select becomes a conditional select, which costs fewer
    // instructions than the arithmetic.
    #[cfg(not(target_arch = "x86_64"))]
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

    fn insert_str(lpm: &mut LongestPrefixMatcher, s: &str) -> Token {
        lpm.insert(s.as_bytes())
    }

    fn find_str(lpm: &LongestPrefixMatcher, s: &str) -> (Token, usize) {
        lpm.find_longest_match(s.as_bytes())
    }

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
    fn default_constructor_size_is_256() {
        assert_eq!(LongestPrefixMatcher::new().size(), 256);
    }

    #[test]
    fn all_single_bytes_found_after_construction() {
        let lpm = LongestPrefixMatcher::new();
        for i in 0u16..=255 {
            let b = [i as u8];
            let (tok, len) = lpm.find_longest_match(&b);
            assert_eq!(tok, i, "wrong token for byte {i}");
            assert_eq!(len, 1, "wrong length for byte {i}");
        }
    }

    // ── Insert ───────────────────────────────────────────────────────────────

    #[test]
    fn first_insert_returns_id_256() {
        let mut lpm = LongestPrefixMatcher::new();
        assert_eq!(insert_str(&mut lpm, "ab"), 256);
    }

    #[test]
    fn subsequent_inserts_increment_id() {
        let mut lpm = LongestPrefixMatcher::new();
        assert_eq!(insert_str(&mut lpm, "ab"), 256);
        assert_eq!(insert_str(&mut lpm, "cd"), 257);
        assert_eq!(insert_str(&mut lpm, "ef"), 258);
    }

    #[test]
    fn exactly_eight_bytes_short_store() {
        let mut lpm = LongestPrefixMatcher::new();
        let id = insert_str(&mut lpm, "12345678");
        let (tok, len) = find_str(&lpm, "12345678");
        assert_eq!((tok, len), (id, 8));
    }

    #[test]
    fn exactly_nine_bytes_long_store() {
        let mut lpm = LongestPrefixMatcher::new();
        let id = insert_str(&mut lpm, "123456789");
        let (tok, len) = find_str(&lpm, "123456789X");
        assert_eq!((tok, len), (id, 9));
    }

    #[test]
    fn max_token_size_insert_and_find() {
        let mut lpm = LongestPrefixMatcher::new();
        let pat = "0123456789abcdef";
        assert_eq!(pat.len(), MAX_TOKEN_SIZE);
        let id = lpm.insert(pat.as_bytes());
        let (tok, len) = lpm.find_longest_match(pat.as_bytes());
        assert_eq!((tok, len), (id, MAX_TOKEN_SIZE));
    }

    // ── find_longest_match ───────────────────────────────────────────────────

    #[test]
    fn longest_match_wins_over_shorter() {
        let mut lpm = LongestPrefixMatcher::new();
        insert_str(&mut lpm, "abc");
        let long_id = insert_str(&mut lpm, "abcdefghi");
        let (tok, len) = find_str(&lpm, "abcdefghi");
        assert_eq!((tok, len), (long_id, 9));
    }

    #[test]
    fn falls_back_to_shorter_if_long_not_present() {
        let mut lpm = LongestPrefixMatcher::new();
        let short_id = insert_str(&mut lpm, "abc");
        let (tok, len) = find_str(&lpm, "abcdef");
        assert_eq!((tok, len), (short_id, 3));
    }

    #[test]
    fn falls_back_to_single_byte() {
        let mut lpm = LongestPrefixMatcher::new();
        insert_str(&mut lpm, "XY");
        let (tok, len) = find_str(&lpm, "XZ");
        assert_eq!((tok, len), (b'X' as Token, 1));
    }

    #[test]
    fn nine_byte_beats_eight_byte() {
        let mut lpm = LongestPrefixMatcher::new();
        insert_str(&mut lpm, "ABCDEFGH");
        let id9 = insert_str(&mut lpm, "ABCDEFGHI");
        let (tok, len) = find_str(&lpm, "ABCDEFGHIJ");
        assert_eq!((tok, len), (id9, 9));
    }

    #[test]
    fn multiple_tokens_same_long_prefix() {
        let mut lpm = LongestPrefixMatcher::new();
        let id1 = insert_str(&mut lpm, "ABCDEFGHX");
        let id2 = insert_str(&mut lpm, "ABCDEFGHYZ");
        assert_eq!(find_str(&lpm, "ABCDEFGHX__"), (id1, 9));
        assert_eq!(find_str(&lpm, "ABCDEFGHYZ_"), (id2, 10));
    }

    #[test]
    fn binary_all_zeros_long_sequence() {
        let mut lpm = LongestPrefixMatcher::new();
        let data = [0u8; 10];
        let id = lpm.insert(&data);
        assert_eq!(lpm.find_longest_match(&data), (id, 10));
    }

    // ── trie promotion (>128 entries in one bucket) ───────────────────────────

    #[test]
    fn all_tokens_findable_with_shared_long_prefix() {
        let mut lpm = LongestPrefixMatcher::new();
        let prefix = vec![b'X'; 8];
        let mut inserted = Vec::with_capacity(130);
        for i in 0..130u32 {
            let mut buf = prefix.clone();
            buf.push(i as u8);
            inserted.push(lpm.insert(&buf));
        }
        for i in 0..130u32 {
            let mut buf = prefix.clone();
            buf.push(i as u8);
            buf.push(0xFF);
            let (tok, len) = lpm.find_longest_match(&buf);
            assert_eq!((tok, len), (inserted[i as usize], 9), "token index {i}");
        }
    }

    #[test]
    fn deep_trie_multi_level_suffix() {
        let mut lpm = LongestPrefixMatcher::new();
        let prefix = vec![b'Z'; 8];
        let mut inserted = Vec::with_capacity(130);
        for i in 0..130u32 {
            let mut buf = prefix.clone();
            buf.push(0x00);
            buf.push(i as u8);
            inserted.push(lpm.insert(&buf));
        }
        for i in 0..130u32 {
            let mut buf = prefix.clone();
            buf.push(0x00);
            buf.push(i as u8);
            buf.push(0xFF);
            let (tok, len) = lpm.find_longest_match(&buf);
            assert_eq!((tok, len), (inserted[i as usize], 10), "token index {i}");
        }
    }

    // ── DictionaryMatcher ────────────────────────────────────────────────────

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
