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
//! cursors can run it in lockstep. Entries longer than 5 bytes are keyed by a 40-bit
//! fingerprint rather than their bytes, so the final answer is verified against
//! the token's bytes; a fingerprint collision is rare (about one in a trillion
//! probes), and the caller falls back to the exact matcher when it happens.

use crate::core::dictionary::{CompactDictionaryView, DictionaryView};
use crate::core::types::{MAX_TOKEN_SIZE, Token};
use crate::encoding::hash::{Map, map_with_capacity};

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

const M1: u64 = 0xC2B2_AE3D_27D4_EB4F;
const M2: u64 = 0x1656_67B1_9E37_79F9;
const FP1: u64 = 0xD6E8_FEB8_6659_FD93;
const FP2: u64 = 0xA0B4_28DB_4A8F_6E05;

/// One table slot, packed into 8 bytes so a probe is one load per slot
/// followed by register arithmetic only:
///
/// ```text
/// bits   0..40   key: exact bytes (`qlen <= 5`) or fingerprint (`qlen >= 6`)
/// bits  40..44   qlen - 1: length of the entry's bytes, the length it is probed at
/// bits  44..60   bmp: longest real token that is a prefix of the entry's bytes
/// bits  60..64   bmp_len - 1: length of `bmp`
/// ```
///
/// Zero is the empty slot; every entry has `qlen >= 2`, so its `qlen - 1` field
/// is non-zero.
type Slot = u64;
const KEY_BITS: u32 = 40;
const SLOT_PAYLOAD: u32 = 44;
/// The key and qlen fields: what a probe compares.
const SLOT_CMP_MASK: u64 = (1u64 << SLOT_PAYLOAD) - 1;
/// Longest key stored exactly; longer keys are fingerprinted.
const EXACT_LEN: usize = 5;

#[inline(always)]
fn pack_slot(key: u64, bmp: Token, bmp_len: u8, qlen: u8) -> Slot {
    debug_assert!(key >> KEY_BITS == 0 && (2..=16).contains(&qlen) && (1..=16).contains(&bmp_len));
    key | (((qlen - 1) as u64) << KEY_BITS)
        | ((bmp as u64) << SLOT_PAYLOAD)
        | (((bmp_len - 1) as u64) << 60)
}

/// What a probe of `(key, l)` must equal after masking with [`SLOT_CMP_MASK`].
#[inline(always)]
fn slot_want(key: u64, l: usize) -> u64 {
    key | (((l - 1) as u64) << KEY_BITS)
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
#[inline(always)]
pub(crate) fn mask128(n: usize) -> u128 {
    MASK128[n.min(16)]
}

/// `MASK128[n]` is the mask of the low `n` bytes of a `u128`.
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

/// Select `a` if `cond` else `b`. The condition is a hit/miss on a hash
/// probe, which is close to a coin flip, so tell the compiler not to turn this
/// into a branch.
#[inline(always)]
fn sel(cond: bool, a: u64, b: u64) -> u64 {
    std::hint::select_unpredictable(cond, a, b)
}

/// 40-bit fingerprint of a masked window: `lo` is its first 8 bytes, `hi` the
/// rest, zero-extended. A folded 64x64-bit multiply (one `mul` on x86) mixes
/// every bit of both words into every output bit; a plain 64-bit multiply
/// would let the high bytes of `hi` influence only the high bits of the
/// product, and keys differing only in their last bytes would collide.
#[inline(always)]
fn fingerprint(lo: u64, hi: u64) -> u64 {
    let p = ((lo ^ FP1) as u128).wrapping_mul((hi ^ FP2) as u128);
    let x = (p as u64) ^ ((p >> 64) as u64);
    x >> (64 - KEY_BITS)
}

/// Table key for the first `l` bytes of the window `(lo, hi)`, `l` in `2..=16`:
/// the bytes themselves when they fit in the key, else their fingerprint.
#[inline(always)]
fn key_for(lo: u64, hi: u64, l: usize) -> u64 {
    // SAFETY: `l <= 16` by contract; the table has 17 entries.
    let [mlo, mhi] = unsafe { *KEY_MASKS.get_unchecked(l) };
    let lo = lo & mlo;
    sel(l > EXACT_LEN, fingerprint(lo, hi & mhi), lo)
}

/// The two candidate buckets for a key, as hashes to be shifted down. Keys
/// probed at different lengths may share a bucket; the qlen field tells them
/// apart.
#[inline(always)]
fn hashes(key: u64) -> (u64, u64) {
    let x = key ^ (key >> 20);
    (x.wrapping_mul(M1), x.wrapping_mul(M2))
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
    /// `hash >> shift` is a bucket index.
    shift: u32,
    /// Id of each single-byte token.
    byte_tok: Vec<Token>,
    /// [`verify_word`] of every token.
    tok_verify: Vec<u128>,
}

impl FlatMatcher {
    /// Build from a dictionary that contains every single-byte token.
    pub(crate) fn from_dictionary(dict: CompactDictionaryView<'_>) -> Self {
        let n = dict.num_tokens();
        let mut byte_tok = vec![0 as Token; 256];
        let mut tok_bytes = Vec::with_capacity(n);
        let mut tok_len = Vec::with_capacity(n);
        let mut tok_verify = Vec::with_capacity(n);
        // Exact set of real tokens, for computing each marker's best prefix.
        let mut real: Map<(u128, u8), Token> = map_with_capacity(n);
        for i in 0..n {
            let id = i as Token;
            let b = dict.token(id);
            debug_assert!(!b.is_empty() && b.len() <= MAX_TOKEN_SIZE);
            let w = load_le_u128(b);
            tok_bytes.push(w);
            tok_len.push(b.len() as u8);
            tok_verify.push(verify_word(w, b.len()));
            real.insert((w, b.len() as u8), id);
            if b.len() == 1 {
                byte_tok[b[0] as usize] = id;
            }
        }

        // Entries keyed by (table key, probe length): real tokens first, then
        // markers along each token's search path, which never displace a real
        // token at the same position.
        let mut entries: Map<(u64, u8), (Token, u8)> = map_with_capacity(n * 2);
        for i in 0..n {
            let l = tok_len[i] as usize;
            if l < 2 {
                continue;
            }
            let w = tok_bytes[i];
            let (lo, hi) = (w as u64, (w >> 64) as u64);
            entries.insert((key_for(lo, hi, l), l as u8), (i as Token, l as u8));
        }
        for i in 0..n {
            let l = tok_len[i] as usize;
            if l < 2 {
                continue;
            }
            let w = tok_bytes[i];
            let (lo, hi) = (w as u64, (w >> 64) as u64);
            let mut m = ROOT_LEN;
            for step in STEPS {
                if m == l {
                    break;
                }
                if l > m {
                    // The search must hit here to keep looking for `l`.
                    entries
                        .entry((key_for(lo, hi, m), m as u8))
                        .or_insert_with(|| best_real_prefix(&real, w, m, &byte_tok));
                    m += step;
                } else {
                    m -= step;
                }
            }
            debug_assert_eq!(m, l);
        }

        // Place the entries; grow and retry if cuckoo insertion fails.
        let mut buckets = ((entries.len() as f64 / (BUCKET_SLOTS as f64 * LOAD_FACTOR)).ceil()
            as usize)
            .next_power_of_two()
            // At least two buckets keeps `shift < 64`.
            .max(2);
        let slots = loop {
            let shift = 64 - buckets.trailing_zeros();
            let mut slots = vec![0 as Slot; buckets * BUCKET_SLOTS];
            let mut rng = 0x2545_F491_4F6C_DD1Du64;
            let ok = entries.iter().all(|(&(key, qlen), &(bmp, bmp_len))| {
                insert(
                    &mut slots,
                    shift,
                    pack_slot(key, bmp, bmp_len, qlen),
                    &mut rng,
                )
            });
            if ok {
                break slots;
            }
            buckets *= 2;
        };
        let shift = 64 - buckets.trailing_zeros();
        Self {
            slots,
            shift,
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
        let (h1, h2) = hashes(key);
        let b1 = (h1 >> self.shift) as usize * BUCKET_SLOTS;
        let b2 = (h2 >> self.shift) as usize * BUCKET_SLOTS;
        let want = slot_want(key, l);
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

/// Longest real token that is a prefix of the first `m` bytes of `w`.
fn best_real_prefix(
    real: &Map<(u128, u8), Token>,
    w: u128,
    m: usize,
    byte_tok: &[Token],
) -> (Token, u8) {
    for k in (2..m).rev() {
        if let Some(&t) = real.get(&(w & mask128(k), k as u8)) {
            return (t, k as u8);
        }
    }
    (byte_tok[w as u8 as usize], 1)
}

/// Cuckoo-insert `slot`, displacing existing entries as needed. Returns false
/// if the displacement budget runs out.
fn insert(slots: &mut [Slot], shift: u32, mut slot: Slot, rng: &mut u64) -> bool {
    for _ in 0..MAX_KICKS {
        let (h1, h2) = hashes(slot & ((1 << KEY_BITS) - 1));
        let b1 = (h1 >> shift) as usize * BUCKET_SLOTS;
        let b2 = (h2 >> shift) as usize * BUCKET_SLOTS;
        for idx in [b1, b2] {
            for j in 0..BUCKET_SLOTS {
                if slots[idx + j] == 0 {
                    slots[idx + j] = slot;
                    return true;
                }
            }
        }
        // xorshift64
        *rng ^= *rng << 13;
        *rng ^= *rng >> 7;
        *rng ^= *rng << 17;
        let r = (*rng % (2 * BUCKET_SLOTS as u64)) as usize;
        let victim = [b1, b2][r / BUCKET_SLOTS] + (r - (r / BUCKET_SLOTS) * BUCKET_SLOTS);
        std::mem::swap(&mut slot, &mut slots[victim]);
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dictionary::{CompactDictionary, Dictionary};
    use crate::encoding::lpm::LongestPrefixMatcher;

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
        let flat = FlatMatcher::from_dictionary(dict.as_view());
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
