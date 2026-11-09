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
//! A table on the first two bytes records which token lengths start with those
//! bytes. Matching loads up to 16 bytes once, reads that table, probes the
//! long-token bucket only when a long token can match, then checks only the short
//! token lengths that the table holds, from longest to shortest. Single bytes
//! need no probe: their token id is the byte.
//!
//! [`LongestPrefixMatcher`] accepts inserts and drives training. Encoding uses
//! [`DictionaryMatcher`](crate::encoding::dictionary_matcher::DictionaryMatcher),
//! a frozen form built from the final, sorted dictionary.

use hashbrown::HashMap;

use crate::core::types::{MAX_TOKEN_SIZE, Token};

/// Tokens of this length or shorter live in the short map; longer tokens are
/// bucketed by their first `BUCKET_PREFIX_LEN` bytes.
const BUCKET_PREFIX_LEN: usize = 8;

const MAX_SUFFIX_LEN: usize = MAX_TOKEN_SIZE - BUCKET_PREFIX_LEN;
const PROMOTE_THRESHOLD: usize = 48;

/// Number of distinct 2-byte prefixes.
const PREFIX_SLOTS: usize = 1 << 16;

/// Mask of the token lengths, as bits `len - 1`, that the long map holds.
const LONG_LENS: u32 = !((1 << BUCKET_PREFIX_LEN) - 1);

/// Mask of the token lengths, as bits `len - 1`, that the 4-byte filter covers.
const QUAD_LENS: u32 = !0b111;

/// Bits of the 4-byte filter index.
const QUAD_BITS: u32 = 16;

/// Odd multiplier for the multiplicative hashes.
const HASH_MUL: u64 = 0x9e37_79b9_7f4a_7c15;

/// Filter slot of the first four bytes of a window.
#[inline(always)]
fn quad_slot(lo64: u64) -> usize {
    ((lo64 as u32 as u64).wrapping_mul(HASH_MUL) >> (64 - QUAD_BITS)) as usize
}

/// One slot of [`ShortTable`]. A length of zero marks an empty slot.
#[derive(Copy, Clone, Debug, Default)]
#[repr(align(16))]
struct ShortSlot {
    key: u64,
    len: u8,
    token: Token,
}

/// Open-addressing table of the tokens of `2..=BUCKET_PREFIX_LEN` bytes, keyed by
/// their bytes as a little-endian `u64` and their length. Linear probing, at most
/// half full. Tokens are never removed.
#[derive(Clone, Debug)]
struct ShortTable {
    slots: Vec<ShortSlot>,
    shift: u32,
    len: usize,
}

impl ShortTable {
    fn new() -> Self {
        Self::with_slots(64)
    }

    fn with_slots(slots: usize) -> Self {
        debug_assert!(slots.is_power_of_two());
        Self {
            slots: vec![ShortSlot::default(); slots],
            shift: 64 - slots.trailing_zeros(),
            len: 0,
        }
    }

    #[inline(always)]
    fn home(&self, key: u64, len: usize) -> usize {
        ((key ^ ((len as u64) << 60)).wrapping_mul(HASH_MUL) >> self.shift) as usize
    }

    /// Make room for `additional` more tokens without growing.
    fn reserve(&mut self, additional: usize) {
        let need = ((self.len + additional) * 2).next_power_of_two();
        if need > self.slots.len() {
            self.grow(need);
        }
    }

    fn grow(&mut self, slots: usize) {
        let old = std::mem::replace(self, Self::with_slots(slots));
        for slot in old.slots.into_iter().filter(|s| s.len != 0) {
            self.insert(slot.key, slot.len as usize, slot.token);
        }
    }

    fn insert(&mut self, key: u64, len: usize, token: Token) {
        if (self.len + 1) * 2 > self.slots.len() {
            self.grow(self.slots.len() * 2);
        }

        let mask = self.slots.len() - 1;
        let mut i = self.home(key, len);
        loop {
            let slot = &mut self.slots[i];
            if slot.len == 0 {
                *slot = ShortSlot {
                    key,
                    len: len as u8,
                    token,
                };
                self.len += 1;
                return;
            }
            if slot.key == key && slot.len as usize == len {
                slot.token = token;
                return;
            }
            i = (i + 1) & mask;
        }
    }

    #[inline(always)]
    fn get(&self, key: u64, len: usize) -> Option<Token> {
        let mask = self.slots.len() - 1;
        let mut i = self.home(key, len);
        loop {
            let slot = &self.slots[i];
            if slot.key == key && slot.len as usize == len {
                return Some(slot.token);
            }
            if slot.len == 0 {
                return None;
            }
            i = (i + 1) & mask;
        }
    }
}

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
#[derive(Debug, Clone)]
pub(crate) struct LongestPrefixMatcher {
    /// Length `2..=8` tokens keyed by (low-`len`-byte u64, length). The id of a
    /// single-byte token is its byte, so the table does not hold them.
    short_map: ShortTable,
    /// Length `9..=16` tokens bucketed by their 8-byte prefix.
    long_map: HashMap<u64, Bucket>,
    /// Per 2-byte prefix, as a little-endian `u16`: bit `len - 1` is set when a
    /// token of length `len >= 2` starts with that prefix.
    prefix_lens: Box<[u16]>,
    /// Per [`quad_slot`] of the first four bytes: bit `len - 1` is set when a
    /// token of length `len >= 4` starts with four bytes of that slot. Slots
    /// can be shared, so a set bit is only a candidate.
    quad_lens: Box<[u16]>,
    /// Next id to assign. `u32` so the full 16-bit token space (65 536 entries)
    /// is representable without overflow.
    next_id: u32,
}

impl LongestPrefixMatcher {
    /// Holds the 256 single-byte tokens with ids `0..=255`.
    pub(crate) fn new() -> Self {
        Self {
            short_map: ShortTable::new(),
            long_map: HashMap::new(),
            prefix_lens: vec![0u16; PREFIX_SLOTS].into_boxed_slice(),
            quad_lens: vec![0u16; 1 << QUAD_BITS].into_boxed_slice(),
            next_id: 256,
        }
    }

    /// Reserve training-time maps for the configured dictionary budget.
    pub(crate) fn reserve(&mut self, token_capacity: usize) {
        self.short_map
            .reserve(token_capacity.saturating_sub(256 + self.short_map.len));
        let long_capacity = (token_capacity / 4).max(16);
        self.long_map
            .reserve(long_capacity.saturating_sub(self.long_map.len()));
    }

    /// Insert `data` and assign it the next available token id.
    ///
    /// Precondition: `2 <= data.len() <= MAX_TOKEN_SIZE`, `data` is not in the
    /// matcher yet, and `size() < 65_536`.
    pub(crate) fn insert(&mut self, data: &[u8]) -> Token {
        let id = self.next_id as Token;
        self.next_id += 1;
        self.insert_internal(data, id);
        id
    }

    #[inline]
    fn insert_internal(&mut self, data: &[u8], id: Token) {
        debug_assert!(data.len() >= 2 && data.len() <= MAX_TOKEN_SIZE);
        let len = data.len();
        let prefix = u16::from_le_bytes([data[0], data[1]]);
        self.prefix_lens[prefix as usize] |= 1 << (len - 1);
        if len >= 4 {
            let quad = u32::from_le_bytes(data[..4].try_into().unwrap());
            self.quad_lens[quad_slot(u64::from(quad))] |= 1 << (len - 1);
        }

        if len <= BUCKET_PREFIX_LEN {
            let key = load_le_u64(data, len);
            self.short_map.insert(key, len, id);
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
        let single = (data[0] as Token, 1);

        // Token lengths that start with the first two bytes and fit the window.
        // A window of one byte keeps no bit, since the table has no bit 0.
        let prefix = lo64 as u16 as usize;
        let mut lens = u32::from(self.prefix_lens[prefix]) & ((1u32 << win) - 1);
        if lens & QUAD_LENS != 0 {
            lens &= u32::from(self.quad_lens[quad_slot(lo64)]) | !QUAD_LENS;
        }
        if lens == 0 {
            return single;
        }

        if lens & LONG_LENS != 0
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

        lens &= !LONG_LENS;
        while lens != 0 {
            let len = (u32::BITS - lens.leading_zeros()) as usize;
            lens &= !(1 << (len - 1));
            if let Some(t) = self.short_map.get(lo64 & mask_u64(len), len) {
                return (t, len);
            }
        }
        single
    }

    /// Number of tokens currently in the matcher.
    #[inline]
    pub(crate) fn size(&self) -> usize {
        self.next_id as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn insert_str(lpm: &mut LongestPrefixMatcher, s: &str) -> Token {
        lpm.insert(s.as_bytes())
    }

    fn find_str(lpm: &LongestPrefixMatcher, s: &str) -> (Token, usize) {
        lpm.find_longest_match(s.as_bytes())
    }

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
}
