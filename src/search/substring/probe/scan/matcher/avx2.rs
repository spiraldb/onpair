// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! AVX2 kernels.
//!
//! 64 codes fill four 256-bit vectors of `u16` lanes. Comparisons yield
//! 0xFFFF/0x0000 lanes, which `narrow` packs to one byte per code, restoring code
//! order across 128-bit lanes; `movemask` extracts one bit per byte.

use std::arch::x86_64::*;

use super::{Block, Kernels, Mask, Matcher, PER_BATCH};
use crate::core::types::{Token, TokenRange};
use crate::search::substring::ProbeCover;
use crate::search::substring::probe::scan::Isa;

/// Membership results for 64 codes: one 0xFF/0x00 byte per code.
type Hits = [__m256i; 2];

/// 64 token codes split across this target's vector registers.
type Vectors = [__m256i; 4];

/// Range start and width (`last - begin`), broadcast into vector lanes.
type Held = (__m256i, __m256i);

/// A token broadcast to every lane.
type Broadcast = __m256i;

/// A 16-byte shuffle row, broadcast into both 128-bit lanes since `vpshufb`
/// indexes within lanes.
type Table = __m256i;

/// Load 64 codes without requiring aligned pointers.
/// The caller must provide 64 readable tokens and the required CPU features.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load(at: *const Token) -> Vectors {
    std::array::from_fn(|i| unsafe { _mm256_loadu_si256(at.add(16 * i).cast()) })
}

/// Fill all mask words by comparing 128 codes per iteration.
/// When skipping is enabled, empty pairs bypass packing and are zeroed.
/// `false` guarantees an empty mask; without skipping, the return is always
/// `true` even if every packed bit is zero.
#[inline]
#[target_feature(enable = "avx2")]
fn words<const SKIP_MOVEMASK_IF_NO_MATCH: bool>(
    codes: &Block,
    bits: &mut Mask,
    hits: impl Fn(Vectors) -> Hits,
) -> bool {
    let mut written = false;
    for (pair, bits) in bits.chunks_exact_mut(2).enumerate() {
        let at = codes[pair * 128..].as_ptr();
        // SAFETY: a block holds 128 codes for every pair of mask words.
        let (a, b) = unsafe { (hits(load(at)), hits(load(at.add(64)))) };
        if SKIP_MOVEMASK_IF_NO_MATCH && !any(a, b) {
            bits.fill(0);
            continue;
        }
        written = true;
        // SAFETY: the chunk is the two words the store writes.
        unsafe { movemask(bits, a, b) };
    }
    written
}

/// Union two sets of hit results without changing code order.
#[inline]
#[target_feature(enable = "avx2")]
fn or(hit: Hits, with: Hits) -> Hits {
    std::array::from_fn(|at| _mm256_or_si256(hit[at], with[at]))
}

/// Whether either 64-code group has a hit.
#[inline]
#[target_feature(enable = "avx2")]
fn any(a: Hits, b: Hits) -> bool {
    let or = _mm256_or_si256(or(a, b)[0], or(a, b)[1]);
    _mm256_testz_si256(or, or) == 0
}

/// Pack 16-bit comparison lanes into bytes, then restore original code order.
/// The permutation corrects the 128-bit lane ordering of `packs_epi16`.
#[inline]
#[target_feature(enable = "avx2")]
fn narrow(hit: [__m256i; 4]) -> Hits {
    let byte = |lo, hi| _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi16(lo, hi));
    [byte(hit[0], hit[1]), byte(hit[2], hit[3])]
}

/// Extract one bit per hit byte into two output words.
/// The caller supplies space for both words and enables AVX2.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn movemask(bits: &mut [u64], a: Hits, b: Hits) {
    let word = |hit: Hits| {
        let lo = _mm256_movemask_epi8(hit[0]) as u32;
        let hi = _mm256_movemask_epi8(hit[1]) as u32;
        u64::from(lo) | u64::from(hi) << 32
    };
    bits[0] = word(a);
    bits[1] = word(b);
}

fn hold(range: TokenRange) -> Held {
    // SAFETY: avx2, checked by `detect_isa` before planning.
    unsafe {
        (
            _mm256_set1_epi16(range.begin as i16),
            _mm256_set1_epi16((range.last - range.begin) as i16),
        )
    }
}

/// No unsigned u16 compare below AVX-512: `min(x, width) == x` is `x <= width`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn inside((lo, width): Held, codes: Vectors) -> Hits {
    let mut hit = codes;
    for hit in &mut hit {
        let inside = _mm256_sub_epi16(*hit, lo);
        *hit = _mm256_cmpeq_epi16(_mm256_min_epu16(inside, width), inside);
    }
    narrow(hit)
}

/// OR range membership into existing hit lanes.
/// The caller must have enabled the compiled vector instruction set.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn check_ranges(mut hit: Hits, held: &[Held], codes: Vectors) -> Hits {
    for &held in held {
        // SAFETY: the caller provides the instruction set required by these helpers.
        hit = unsafe { or(hit, inside(held, codes)) };
    }
    hit
}

/// Fill a block mask from prepared ranges; an empty slice clears the mask.
/// The target-feature boundary keeps range comparisons inside the vector loop.
#[target_feature(enable = "avx2")]
fn ranges_mask<const SKIP_MOVEMASK_IF_NO_MATCH: bool>(
    held: &[Held],
    codes: &Block,
    bits: &mut Mask,
) -> bool {
    let Some((&first, rest)) = held.split_first() else {
        bits.fill(0);
        return false;
    };
    words::<SKIP_MOVEMASK_IF_NO_MATCH>(
        codes,
        bits,
        #[inline(always)]
        |codes| unsafe { check_ranges(inside(first, codes), rest, codes) },
    )
}

fn broadcast(code: Token) -> Broadcast {
    // SAFETY: avx2, checked by `detect_isa` before planning.
    unsafe { _mm256_set1_epi16(code as i16) }
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn eq_or_hits(first: &Broadcast, rest: &[Broadcast], codes: Vectors) -> Hits {
    let mut hit = codes;
    for hit in &mut hit {
        *hit = _mm256_cmpeq_epi16(*hit, *first);
    }
    for token in rest {
        for (hit, &codes) in hit.iter_mut().zip(&codes) {
            *hit = _mm256_or_si256(*hit, _mm256_cmpeq_epi16(codes, *token));
        }
    }
    narrow(hit)
}

/// Broadcast point probes and prepared ranges reused across scan blocks.
pub(in crate::search::substring::probe::scan) struct EqOr<
    const SKIP_MOVEMASK_IF_NO_MATCH: bool,
    const HAS_POINTS: bool = true,
> {
    /// One broadcast vector per point in the cover.
    tokens: Vec<Broadcast>,
    /// Range starts and widths, already broadcast for vector comparisons.
    ranges: Vec<Held>,
}

impl<const SKIP_MOVEMASK_IF_NO_MATCH: bool, const HAS_POINTS: bool> Matcher
    for EqOr<SKIP_MOVEMASK_IF_NO_MATCH, HAS_POINTS>
{
    fn new(cover: &ProbeCover) -> Self {
        debug_assert!(HAS_POINTS || cover.points().is_empty());
        Self {
            tokens: if HAS_POINTS {
                cover.points().iter().copied().map(broadcast).collect()
            } else {
                Vec::new()
            },
            ranges: cover.ranges().iter().copied().map(hold).collect(),
        }
    }

    fn check(&self, codes: &Block, bits: &mut Mask) -> bool {
        // SAFETY: planning selects this kernel only after CPU feature detection.
        unsafe {
            if HAS_POINTS {
                eq_or_mask::<SKIP_MOVEMASK_IF_NO_MATCH>(&self.tokens, &self.ranges, codes, bits)
            } else {
                ranges_mask::<SKIP_MOVEMASK_IF_NO_MATCH>(&self.ranges, codes, bits)
            }
        }
    }
}

/// Fill the block mask with point and range matches.
/// The target-feature boundary lets the comparison closure inline with its caller.
#[target_feature(enable = "avx2")]
fn eq_or_mask<const SKIP_MOVEMASK_IF_NO_MATCH: bool>(
    tokens: &[Broadcast],
    ranges: &[Held],
    codes: &Block,
    bits: &mut Mask,
) -> bool {
    let Some((first, rest)) = tokens.split_first() else {
        return ranges_mask::<SKIP_MOVEMASK_IF_NO_MATCH>(ranges, codes, bits);
    };
    words::<SKIP_MOVEMASK_IF_NO_MATCH>(
        codes,
        bits,
        #[inline(always)]
        |codes| unsafe { check_ranges(eq_or_hits(first, rest, codes), ranges, codes) },
    )
}

/// Four lookup tables ordered from the least to the most significant nibble.
struct Batch([Table; 4]);

/// Build tables for at most eight point IDs.
/// Point `k` contributes bit `1 << k` at each of its four nibble positions.
fn batch(tokens: &[Token]) -> Batch {
    let mut rows = [[0u8; 16]; 4];
    for (k, &token) in tokens.iter().enumerate() {
        for (shift, row) in rows.iter_mut().enumerate() {
            row[usize::from(token >> (4 * shift) & 0x0f)] |= 1 << k;
        }
    }
    Batch(rows.map(table))
}

fn table(row: [u8; 16]) -> Table {
    // SAFETY: avx2, checked by `detect_isa`; the row is the load's 16 bytes.
    unsafe { _mm256_broadcastsi128_si256(_mm_loadu_si128(row.as_ptr().cast())) }
}

/// Pack low and high bytes separately and restore their original code order.
/// Convert surviving point bits to 0xFF lanes for the later movemask.
#[target_feature(enable = "avx2")]
unsafe fn probe<const BATCHES: usize>(batches: &[Batch; BATCHES], codes: Vectors) -> Hits {
    let byte = _mm256_set1_epi16(0x00ff);
    let nibble = _mm256_set1_epi8(0x0f);
    let zero = _mm256_setzero_si256();
    std::array::from_fn(|at| {
        let (lo, hi) = (codes[2 * at], codes[2 * at + 1]);
        let order = |packed| _mm256_permute4x64_epi64::<0xD8>(packed);
        let low_byte = order(_mm256_packus_epi16(
            _mm256_and_si256(lo, byte),
            _mm256_and_si256(hi, byte),
        ));
        let high_byte = order(_mm256_packus_epi16(
            _mm256_srli_epi16::<8>(lo),
            _mm256_srli_epi16::<8>(hi),
        ));
        let nibbles = [
            _mm256_and_si256(low_byte, nibble),
            _mm256_and_si256(_mm256_srli_epi16::<4>(low_byte), nibble),
            _mm256_and_si256(high_byte, nibble),
            _mm256_and_si256(_mm256_srli_epi16::<4>(high_byte), nibble),
        ];
        let pair = |batch: &Batch| {
            (
                _mm256_and_si256(
                    _mm256_shuffle_epi8(batch.0[0], nibbles[0]),
                    _mm256_shuffle_epi8(batch.0[1], nibbles[1]),
                ),
                _mm256_and_si256(
                    _mm256_shuffle_epi8(batch.0[2], nibbles[2]),
                    _mm256_shuffle_epi8(batch.0[3], nibbles[3]),
                ),
            )
        };
        let (low, high) = pair(&batches[0]);
        let mut hit = _mm256_and_si256(low, high);
        for batch in &batches[1..] {
            let (low, high) = pair(batch);
            hit = _mm256_or_si256(hit, _mm256_and_si256(low, high));
        }
        _mm256_cmpeq_epi8(_mm256_cmpeq_epi8(hit, zero), zero)
    })
}

/// Prepared point batches and ranges for repeated block checks.
/// `BATCHES` must be positive and large enough to hold every point in the cover;
/// planning and dispatch choose the supported specialization.
pub(in crate::search::substring::probe::scan) struct NibbleN8<
    const BATCHES: usize,
    const SKIP_MOVEMASK_IF_NO_MATCH: bool,
> {
    /// Up to eight point probes per batch; unused entries contribute no bits.
    batches: [Batch; BATCHES],
    /// Prepared inclusive ranges, added after point matching.
    ranges: Vec<Held>,
}

impl<const BATCHES: usize, const SKIP_MOVEMASK_IF_NO_MATCH: bool> Matcher
    for NibbleN8<BATCHES, SKIP_MOVEMASK_IF_NO_MATCH>
{
    fn new(cover: &ProbeCover) -> Self {
        let codes = cover.points();
        Self {
            batches: std::array::from_fn(|at| {
                let from = (at * PER_BATCH).min(codes.len());
                let to = (from + PER_BATCH).min(codes.len());
                batch(&codes[from..to])
            }),
            ranges: cover.ranges().iter().copied().map(hold).collect(),
        }
    }

    fn check(&self, codes: &Block, bits: &mut Mask) -> bool {
        // SAFETY: planning checks CPU features and the point count; dispatch
        // supplies the matching nonzero batch specialization.
        unsafe {
            nibble_mask::<BATCHES, SKIP_MOVEMASK_IF_NO_MATCH>(
                &self.batches,
                &self.ranges,
                codes,
                bits,
            )
        }
    }
}

/// Fill a block mask with nibble-table and range matches.
/// The target-feature boundary keeps the probe closure inside the vector loop.
#[target_feature(enable = "avx2")]
fn nibble_mask<const BATCHES: usize, const SKIP_MOVEMASK_IF_NO_MATCH: bool>(
    batches: &[Batch; BATCHES],
    ranges: &[Held],
    codes: &Block,
    bits: &mut Mask,
) -> bool {
    words::<SKIP_MOVEMASK_IF_NO_MATCH>(
        codes,
        bits,
        #[inline(always)]
        |codes| {
            // SAFETY: the instruction set, inherited from this function.
            unsafe { check_ranges(probe(batches, codes), ranges, codes) }
        },
    )
}

/// The AVX2 matchers.
pub(in crate::search::substring::probe::scan) struct Avx2;

impl Kernels for Avx2 {
    const ISA: Isa = Isa::Avx2;
    type EqOr<const SKIP_MOVEMASK_IF_NO_MATCH: bool, const HAS_POINTS: bool> =
        EqOr<SKIP_MOVEMASK_IF_NO_MATCH, HAS_POINTS>;
    type NibbleN8<const BATCHES: usize, const SKIP_MOVEMASK_IF_NO_MATCH: bool> =
        NibbleN8<BATCHES, SKIP_MOVEMASK_IF_NO_MATCH>;
}
