// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fixed-cost longest-prefix matcher for the encode loop.
//!
//! [`LongestPrefixMatcher`](crate::encoding::lpm::LongestPrefixMatcher) does a
//! variable amount of work per lookup: a hash probe, then a bucket scan or trie
//! walk, then up to eight more probes, each behind a data-dependent branch. The
//! encode loop mispredicts several of those per step, and a mispredict is what
//! stops independent row cursors from being interleaved profitably.
//!
//! `FlatMatcher` answers the same question with exactly four probes of one hash
//! table, driven by a binary search over token lengths (Waldvogel et al.,
//! "Scalable High Speed IP Routing Lookups", SIGCOMM '97). Lengths `2..=16` form
//! a balanced tree rooted at 9; a probe that hits moves to the longer child, a
//! miss to the shorter one. To make that sound, the table holds *markers* as
//! well as tokens: a token of length `L` leaves a marker at every shorter length
//! the search visits on its way to `L`, so a miss at some length rules out every
//! longer length beneath it. Each entry carries the longest real token that is
//! a prefix of its own bytes, so when the search ends on a marker the answer is
//! already in hand. Length 1 is never probed: every single byte is a token, and
//! it is the fallback.
//!
//! Each probe checks one 8-byte slot in each of two candidate buckets (cuckoo
//! placement), so a lookup is a fixed sequence of loads, compares and selects
//! with no data-dependent control flow, and short enough that several row
//! cursors can run it in lockstep. Entries are keyed by a 44-bit fingerprint of
//! their bytes and length rather than the bytes themselves, so the final answer
//! is verified against the token's bytes; a fingerprint collision is rare
//! (about one probe in ten trillion), and the caller falls back to the exact
//! matcher when it happens.

use crate::core::dictionary::{CompactDictionaryView, DictionaryView};
use crate::core::types::{MAX_TOKEN_SIZE, Token};
use crate::encoding::lpm::LongestPrefixMatcher;

/// Root of the binary search over lengths `2..=16`.
const ROOT_LEN: usize = 9;
/// Step of each level of the search after the root probe.
const STEPS: [usize; 4] = [4, 2, 1, 0];
/// Slots per bucket: 1 or 2. Two choices of one slot keep a probe at two
/// loads and compares; the price is a lower load factor.
const BUCKET_SLOTS: usize = 1;
/// Target load factor when sizing the table.
const LOAD_FACTOR: f64 = if BUCKET_SLOTS == 1 { 0.45 } else { 0.75 };
/// Cuckoo displacement budget before the table is grown.
const MAX_KICKS: usize = 512;

const FP1: u64 = 0xD6E8_FEB8_6659_FD93;
const FP2: u64 = 0xA0B4_28DB_4A8F_6E05;

/// One table slot, packed into 8 bytes so a probe is one load per slot
/// followed by register arithmetic only:
///
/// ```text
/// bits   0..44   key: fingerprint of the entry's bytes and their length
/// bits  44..60   bmp: longest real token that is a prefix of the entry's bytes
/// bits  60..64   bmp_len - 1: length of `bmp`
/// ```
///
/// Zero is the empty slot. An entry's key is never zero: the fingerprint has
/// its lowest bit forced on.
type Slot = u64;
const KEY_BITS: u32 = 44;
const SLOT_PAYLOAD: u32 = 44;
/// The key field: what a probe compares.
const SLOT_CMP_MASK: u64 = (1u64 << KEY_BITS) - 1;

#[inline(always)]
fn pack_slot(key: u64, bmp: Token, bmp_len: u8) -> Slot {
    debug_assert!(key >> KEY_BITS == 0 && key != 0 && (1..=16).contains(&bmp_len));
    key | ((bmp as u64) << SLOT_PAYLOAD) | (((bmp_len - 1) as u64) << 60)
}

/// A payload: `bmp | (bmp_len - 1) << 16`.
#[inline(always)]
fn payload_token(p: u32) -> Token {
    p as Token
}
#[inline(always)]
fn payload_len(p: u32) -> usize {
    (p >> 16) as usize + 1
}

/// Mask of the low `n` bytes of a `u128`, `n` in `0..=16`.
#[cfg(test)]
fn mask128(n: usize) -> u128 {
    MASK128[n.min(16)]
}

/// `MASK128[n]` is the mask of the low `n` bytes of a `u128`.
#[cfg(test)]
static MASK128: [u128; 17] = {
    let mut t = [0u128; 17];
    let mut n = 1;
    while n <= 16 {
        t[n] = if n == 16 {
            u128::MAX
        } else {
            (1u128 << (n * 8)) - 1
        };
        n += 1;
    }
    t
};

/// `KEY_MASKS[l]` masks the two window halves for a probe of length `l`: the
/// first 8 bytes, and the bytes beyond 8 (all zero for `l <= 8`).
pub(crate) static KEY_MASKS: [[u64; 2]; 17] = {
    let mut t = [[0u64; 2]; 17];
    let mut l: usize = 0;
    while l <= 16 {
        let lo = if l >= 8 { 8 } else { l };
        let hi = l.saturating_sub(8);
        t[l] = [
            if lo == 0 {
                0
            } else {
                u64::MAX >> ((8 - lo) * 8)
            },
            if hi == 0 {
                0
            } else {
                u64::MAX >> ((8 - hi) * 8)
            },
        ];
        l += 1;
    }
    t
};

/// Key for the first `l` bytes of the window `(lo, hi)`, `l` in `2..=16`: a
/// 44-bit fingerprint of those bytes and of `l`, so the same bytes probed at
/// two lengths (a token and its zero-byte extension) get distinct keys and no
/// separate length compare is needed. Never zero, so it cannot match an empty
/// slot.
///
/// A folded 64x64-bit multiply (one `mul` on x86) mixes every bit of both
/// words into every output bit; a plain 64-bit multiply would let the high
/// bytes of `hi` influence only the high bits of the product, and keys
/// differing only in their last bytes would collide.
#[inline(always)]
fn key_for(lo: u64, hi: u64, l: usize) -> u64 {
    // SAFETY: `l <= 16` by contract; the table has 17 entries.
    let [mlo, mhi] = unsafe { *KEY_MASKS.get_unchecked(l) };
    let p = ((lo & mlo) ^ FP1 ^ (l as u64)) as u128 * ((hi & mhi) ^ FP2) as u128;
    let x = (p as u64) ^ ((p >> 64) as u64);
    (x >> (64 - KEY_BITS)) | 1
}

/// The two candidate bucket indices for a key in a table of `mask + 1`
/// buckets, `shift = 44 - log2(mask + 1)`. The key is already a fingerprint,
/// so its top bits index the first bucket directly and a slice of its low bits
/// displaces the second (cuckoo-filter style); neither costs a multiply.
#[inline(always)]
fn buckets(key: u64, shift: u32, mask: usize) -> (usize, usize) {
    let b1 = (key >> shift) as usize;
    let b2 = b1 ^ ((key as usize & mask) | 1);
    (b1, b2)
}

/// Verification word of a token: its bytes, zero-extended, with the length
/// stamped in the top byte (which the bytes cannot occupy unless the token is
/// 16 bytes long, in which case the full 16 bytes identify it). Comparing the
/// input window against this word checks both bytes and length at once.
#[inline(always)]
fn verify_word(bytes: u128, len: usize) -> u128 {
    bytes | (((len & 15) as u128) << 120)
}

/// Load the first `min(len, 16)` bytes of `data` into a little-endian `u128`.
#[inline]
pub(crate) fn load_le_u128(data: &[u8]) -> u128 {
    let mut buf = [0u8; 16];
    let n = data.len().min(MAX_TOKEN_SIZE);
    buf[..n].copy_from_slice(&data[..n]);
    u128::from_le_bytes(buf)
}

/// In-flight state of one length search: the length to probe next and the
/// best payload so far (`bmp | (bmp_len - 1) << 16`).
#[derive(Copy, Clone, Debug)]
pub(crate) struct Search {
    l: usize,
    best: u32,
}

/// Fixed-cost matcher built from a complete dictionary. See the module docs.
#[derive(Debug, Clone)]
pub(crate) struct FlatMatcher {
    slots: Vec<Slot>,
    /// `key >> shift` is a bucket index: `shift = 44 - log2(buckets)`.
    shift: u32,
    /// `buckets - 1`.
    mask: usize,
    /// Id of each single-byte token.
    byte_tok: Vec<Token>,
    /// [`verify_word`] of every token.
    tok_verify: Vec<u128>,
}

impl FlatMatcher {
    /// Build from a dictionary that contains every single-byte token, using
    /// the exact matcher over the same dictionary (with the same token ids) to
    /// find each marker's best real prefix.
    pub(crate) fn from_dictionary(
        dict: CompactDictionaryView<'_>,
        lpm: &LongestPrefixMatcher,
    ) -> Self {
        let n = dict.num_tokens();
        let mut byte_tok = vec![0 as Token; 256];
        let mut tok_verify = Vec::with_capacity(n);
        for i in 0..n {
            let b = dict.token(i as Token);
            debug_assert!(!b.is_empty() && b.len() <= MAX_TOKEN_SIZE);
            tok_verify.push(verify_word(load_le_u128(b), b.len()));
            if b.len() == 1 {
                byte_tok[b[0] as usize] = i as Token;
            }
        }

        // Real tokens first, then markers along each token's search path; a
        // marker never displaces a real token at the same key. Everything goes
        // straight into the table; if placement fails the table doubles and
        // the pass restarts.
        // Markers add about half again as many entries as there are tokens.
        let buckets = ((n + n / 2) as f64 / (BUCKET_SLOTS as f64 * LOAD_FACTOR)).ceil() as usize;
        let mut buckets = buckets.next_power_of_two().max(2);
        let slots = loop {
            debug_assert!(buckets.trailing_zeros() < KEY_BITS);
            let shift = KEY_BITS - buckets.trailing_zeros();
            let mask = buckets - 1;
            let mut table = Table {
                slots: vec![0 as Slot; buckets * BUCKET_SLOTS],
                shift,
                mask,
                rng: 0x2545_F491_4F6C_DD1Du64,
            };
            let placed = (|| {
                for i in 0..n {
                    let b = dict.token(i as Token);
                    let l = b.len();
                    if l < 2 {
                        continue;
                    }
                    let w = load_le_u128(b);
                    let key = key_for(w as u64, (w >> 64) as u64, l);
                    if !table.insert(pack_slot(key, i as Token, l as u8)) {
                        return false;
                    }
                }
                for i in 0..n {
                    let b = dict.token(i as Token);
                    let l = b.len();
                    if l < 2 {
                        continue;
                    }
                    let w = load_le_u128(b);
                    let (lo, hi) = (w as u64, (w >> 64) as u64);
                    let mut m = ROOT_LEN;
                    for step in STEPS {
                        if m == l {
                            break;
                        }
                        if l > m {
                            // The search must hit here to keep looking for `l`.
                            // The marker's answer is the longest real token that
                            // is a prefix of its bytes, which the exact matcher
                            // finds.
                            let key = key_for(lo, hi, m);
                            if !table.contains(key) {
                                let (t, tl) = lpm.find_longest_match(&b[..m]);
                                if !table.insert(pack_slot(key, t, tl as u8)) {
                                    return false;
                                }
                            }
                            m += step;
                        } else {
                            m -= step;
                        }
                    }
                    debug_assert_eq!(m, l);
                }
                true
            })();
            if placed {
                break table.slots;
            }
            buckets *= 2;
        };
        let shift = KEY_BITS - buckets.trailing_zeros();
        Self {
            slots,
            shift,
            mask: buckets - 1,
            byte_tok,
            tok_verify,
        }
    }

    /// Probe length `l` against the window `(lo, hi)`. Returns the hit flag
    /// and the hit entry's payload (`bmp | (bmp_len - 1) << 16`; zero on a
    /// miss). Lengths beyond `avail` never hit. No data-dependent control
    /// flow: one 8-byte load per slot, then register arithmetic.
    #[inline(always)]
    fn probe(&self, lo: u64, hi: u64, l: usize, avail: usize) -> (bool, u32) {
        let key = key_for(lo, hi, l);
        let (b1, b2) = buckets(key, self.shift, self.mask);
        let (b1, b2) = (b1 * BUCKET_SLOTS, b2 * BUCKET_SLOTS);
        let want = key;
        // SAFETY: `hash >> shift < buckets`, so both bucket bases plus
        // `BUCKET_SLOTS - 1` index within `slots`.
        let (s0, s2) = unsafe { (*self.slots.get_unchecked(b1), *self.slots.get_unchecked(b2)) };
        let e0 = (s0 & SLOT_CMP_MASK) == want;
        let e2 = (s2 & SLOT_CMP_MASK) == want;
        // At most one slot holds `(key, l)`, so the payloads simply combine.
        let mut p = std::hint::select_unpredictable(e0, (s0 >> SLOT_PAYLOAD) as u32, 0)
            | std::hint::select_unpredictable(e2, (s2 >> SLOT_PAYLOAD) as u32, 0);
        let mut hit = e0 | e2;
        if BUCKET_SLOTS == 2 {
            // SAFETY: as above.
            let (s1, s3) = unsafe {
                (
                    *self.slots.get_unchecked(b1 + 1),
                    *self.slots.get_unchecked(b2 + 1),
                )
            };
            let e1 = (s1 & SLOT_CMP_MASK) == want;
            let e3 = (s3 & SLOT_CMP_MASK) == want;
            p |= std::hint::select_unpredictable(e1, (s1 >> SLOT_PAYLOAD) as u32, 0)
                | std::hint::select_unpredictable(e3, (s3 >> SLOT_PAYLOAD) as u32, 0);
            hit |= e1 | e3;
        }
        (hit & (l <= avail), p)
    }

    /// Start a search over the window `(lo, hi)` (the next `min(avail, 16)`
    /// input bytes, zero-extended): the fallback answer is the single-byte
    /// token, and the search begins at the root length.
    #[inline(always)]
    pub(crate) fn begin(&self, lo: u64) -> Search {
        // SAFETY: `lo as u8` is in `0..256`, the table's length.
        let byte = unsafe { *self.byte_tok.get_unchecked(lo as u8 as usize) };
        Search {
            l: ROOT_LEN,
            best: byte as u32,
        }
    }

    /// One level of the length search: probe the current length, keep the
    /// hit's payload, and move to the longer child on a hit or the shorter one
    /// on a miss. `LEVEL` is `0..4`.
    #[inline(always)]
    pub(crate) fn level<const LEVEL: usize>(&self, s: &mut Search, lo: u64, hi: u64, avail: usize) {
        let (hit, p) = self.probe(lo, hi, s.l, avail);
        s.best = std::hint::select_unpredictable(hit, p, s.best);
        s.l = std::hint::select_unpredictable(hit, s.l + STEPS[LEVEL], s.l - STEPS[LEVEL]);
    }

    /// Finish a search after its four levels: the token, its length, and
    /// whether the answer verified against the token's bytes. `avail >= 1`.
    ///
    /// When `verified` is false the caller must fall back to an exact matcher;
    /// this only happens on a fingerprint collision.
    #[inline(always)]
    pub(crate) fn finish(&self, s: Search, lo: u64, hi: u64) -> (Token, usize, bool) {
        let tok = payload_token(s.best);
        let len = payload_len(s.best);
        // SAFETY: `tok` came from the table, whose entries are token ids `< n`.
        let expect = unsafe { *self.tok_verify.get_unchecked(tok as usize) };
        // SAFETY: `len <= 16`.
        let [mlo, mhi] = unsafe { *KEY_MASKS.get_unchecked(len) };
        let w = ((lo & mlo) as u128) | (((hi & mhi) as u128) << 64);
        (tok, len, expect == verify_word(w, len))
    }

    /// Whole search on one window; see [`begin`](Self::begin),
    /// [`level`](Self::level) and [`finish`](Self::finish).
    #[cfg(test)]
    pub(crate) fn find(&self, lo: u64, hi: u64, avail: usize) -> (Token, usize, bool) {
        let mut s = self.begin(lo);
        self.level::<0>(&mut s, lo, hi, avail);
        self.level::<1>(&mut s, lo, hi, avail);
        self.level::<2>(&mut s, lo, hi, avail);
        self.level::<3>(&mut s, lo, hi, avail);
        self.finish(s, lo, hi)
    }

    /// Exact longest match on a slice: [`find`](Self::find) with a safe load and
    /// the verification result folded into an `Option`.
    #[cfg(test)]
    pub(crate) fn find_slice(&self, data: &[u8]) -> Option<(Token, usize)> {
        let avail = data.len();
        let w = load_le_u128(data) & mask128(avail.min(MAX_TOKEN_SIZE));
        let (tok, len, verified) = self.find(w as u64, (w >> 64) as u64, avail);
        verified.then_some((tok, len))
    }
}

/// The slot table while it is being filled.
struct Table {
    slots: Vec<Slot>,
    shift: u32,
    mask: usize,
    rng: u64,
}

impl Table {
    /// Whether `key` is already placed.
    fn contains(&self, key: u64) -> bool {
        let (b1, b2) = buckets(key, self.shift, self.mask);
        (0..BUCKET_SLOTS).any(|j| {
            self.slots[b1 * BUCKET_SLOTS + j] & SLOT_CMP_MASK == key
                || self.slots[b2 * BUCKET_SLOTS + j] & SLOT_CMP_MASK == key
        })
    }

    /// Cuckoo-insert `slot`, displacing existing entries as needed. Returns
    /// false if the displacement budget runs out.
    fn insert(&mut self, mut slot: Slot) -> bool {
        for _ in 0..MAX_KICKS {
            let (b1, b2) = buckets(slot & SLOT_CMP_MASK, self.shift, self.mask);
            let (b1, b2) = (b1 * BUCKET_SLOTS, b2 * BUCKET_SLOTS);
            for idx in [b1, b2] {
                for j in 0..BUCKET_SLOTS {
                    if self.slots[idx + j] == 0 {
                        self.slots[idx + j] = slot;
                        return true;
                    }
                }
            }
            // xorshift64
            self.rng ^= self.rng << 13;
            self.rng ^= self.rng >> 7;
            self.rng ^= self.rng << 17;
            let r = (self.rng % (2 * BUCKET_SLOTS as u64)) as usize;
            let victim = [b1, b2][r / BUCKET_SLOTS] + (r - (r / BUCKET_SLOTS) * BUCKET_SLOTS);
            std::mem::swap(&mut slot, &mut self.slots[victim]);
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dictionary::{CompactDictionary, Dictionary};

    fn make_dict(extra: &[&[u8]]) -> CompactDictionary {
        let mut toks: Vec<Vec<u8>> = (0u16..=255).map(|i| vec![i as u8]).collect();
        for &s in extra {
            toks.push(s.to_vec());
        }
        toks.sort();
        toks.dedup();
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for t in &toks {
            bytes.extend_from_slice(t);
            offsets.push(bytes.len() as u32);
        }
        CompactDictionary::from_raw(bytes, offsets)
    }

    fn check_agrees(dict: &CompactDictionary, inputs: &[&[u8]]) {
        let lpm = LongestPrefixMatcher::from_dictionary(dict.as_view());
        let flat = FlatMatcher::from_dictionary(dict.as_view(), &lpm);
        for &s in inputs {
            for start in 0..s.len() {
                let d = &s[start..];
                let want = lpm.find_longest_match(d);
                let got = flat.find_slice(d).expect("verified");
                assert_eq!(got, want, "input {:?}", String::from_utf8_lossy(d));
            }
        }
    }

    #[test]
    fn base_dictionary_matches_single_bytes() {
        let dict = make_dict(&[]);
        check_agrees(
            &dict,
            &[b"hello world", b"\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"],
        );
    }

    #[test]
    fn all_lengths_and_markers() {
        // Tokens at every length 2..=16 sharing prefixes, plus gaps so the
        // search must fall back to marker bmps.
        let long = b"abcdefghijklmnop";
        let mut extra: Vec<&[u8]> = (2..=16).map(|l| &long[..l]).collect();
        extra.push(b"abcdefghijklmnoX");
        extra.push(b"abcXefghijk");
        extra.push(b"zzzzzzzzzzzz");
        extra.push(b"the ");
        extra.push(b"the carefully ");
        let dict = make_dict(&extra);
        check_agrees(
            &dict,
            &[
                b"abcdefghijklmnopqrs",
                b"abcdefghijklmnoXYZ",
                b"abcXefghijklm",
                b"abcXefghi",
                b"zzzzzzzzzzzzzzzzzzzz",
                b"zzzzzzz",
                b"the carefully the car",
                b"thx",
                b"\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
                b"abcdefgh",
                b"abcdefghi",
            ],
        );
    }

    #[test]
    fn matches_lpm_on_trained_dictionaries() {
        use crate::encoding::config::{FixedThreshold, ThresholdSpec, TrainingConfig};
        use crate::encoding::trainer::train;
        use crate::test_corpus::{make_raw, mixed_length_strings, user_strings};
        let mut corpora: Vec<Vec<Vec<u8>>> = Vec::new();
        corpora.push(
            user_strings(200)
                .into_iter()
                .map(String::into_bytes)
                .collect(),
        );
        corpora.push(mixed_length_strings(300, 120, 7));
        for (seed, strings) in corpora.iter().enumerate() {
            let raw = make_raw(strings);
            for bits in [9u8, 12, 16] {
                let cfg = TrainingConfig {
                    max_dict_bits: bits,
                    threshold: ThresholdSpec::Fixed(FixedThreshold { value: 2 }),
                    seed: Some(seed as u64 + 1),
                };
                let result = train(&raw.data, &raw.offsets, &cfg);
                let inputs: Vec<&[u8]> = strings.iter().map(|s| s.as_slice()).collect();
                check_agrees(&result.dict, &inputs);
            }
        }
    }
}
