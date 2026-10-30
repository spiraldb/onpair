// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! AVX-512BW kernels.
//!
//! 64 codes fill two 512-bit vectors of `u16` lanes. Comparisons land in mask
//! registers, so the hits of 64 codes are already one `u64` mask word: there is
//! nothing to narrow or pack.

use std::arch::x86_64::*;

use super::{Block, Kernels, Mask, Matcher, nibble_rows};
use crate::core::types::{Token, TokenRange};
use crate::search::substring::ProbeCover;
use crate::search::substring::probe::scan::Isa;

/// Membership results for 64 codes: one mask bit per code.
type Hits = u64;

/// 64 token codes split across this target's vector registers.
type Vectors = [__m512i; 2];

/// Range start and width (`last - begin`), broadcast into vector lanes.
type Held = (__m512i, __m512i);

/// A token broadcast to every lane.
type Broadcast = __m512i;

/// A 16-byte shuffle row, broadcast into every 128-bit lane since `vpshufb`
/// indexes within lanes.
type Table = __m512i;

/// Load 64 codes without requiring aligned pointers.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn load(codes: &[Token; 64]) -> Vectors {
    // SAFETY: load `i` reads 32 of the 64 codes, starting at code 32 * i.
    std::array::from_fn(|i| unsafe { _mm512_loadu_si512(codes.as_ptr().add(32 * i).cast()) })
}

/// Fill all mask words by comparing 128 codes per iteration.
/// When skipping is enabled, empty pairs bypass packing and are zeroed.
/// `false` guarantees an empty mask; without skipping, the return is always
/// `true` even if every packed bit is zero.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn words<const SKIP_MOVEMASK_IF_NO_MATCH: bool>(
    codes: &Block,
    bits: &mut Mask,
    hits: impl Fn(Vectors) -> Hits,
) -> bool {
    let mut written = false;
    // Two 64-code groups per pair of mask words.
    let groups = codes.as_chunks::<64>().0.as_chunks::<2>().0;
    for ([lo, hi], bits) in groups.iter().zip(bits.as_chunks_mut::<2>().0) {
        let (a, b) = (hits(load(lo)), hits(load(hi)));
        if SKIP_MOVEMASK_IF_NO_MATCH && !any(a, b) {
            *bits = [0; 2];
            continue;
        }
        written = true;
        movemask(bits, a, b);
    }
    written
}

/// Union two sets of hit results without changing code order.
#[inline]
fn or(hit: Hits, with: Hits) -> Hits {
    hit | with
}

/// Whether either 64-code group has a hit.
#[inline]
fn any(a: Hits, b: Hits) -> bool {
    a | b != 0
}

/// Combine two 32-code masks, placing earlier codes in the low bits.
#[inline]
fn join(lo: u32, hi: u32) -> Hits {
    u64::from(lo) | u64::from(hi) << 32
}

/// Store two already-packed AVX-512 masks as the two mask words.
#[inline]
fn movemask(bits: &mut [u64; 2], a: Hits, b: Hits) {
    *bits = [a, b];
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn hold(range: TokenRange) -> Held {
    (
        _mm512_set1_epi16(range.begin as i16),
        _mm512_set1_epi16((range.last - range.begin) as i16),
    )
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn inside((lo, width): Held, codes: Vectors) -> Hits {
    let inside = |codes| _mm512_cmple_epu16_mask(_mm512_sub_epi16(codes, lo), width);
    join(inside(codes[0]), inside(codes[1]))
}

/// OR range membership into existing hit lanes.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn check_ranges(mut hit: Hits, held: &[Held], codes: Vectors) -> Hits {
    for &held in held {
        hit = or(hit, inside(held, codes));
    }
    hit
}

/// Fill a block mask from prepared ranges; an empty slice clears the mask.
/// The target-feature boundary keeps range comparisons inside the vector loop.
#[target_feature(enable = "avx512f,avx512bw")]
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
        |codes| check_ranges(inside(first, codes), rest, codes),
    )
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn broadcast(code: Token) -> Broadcast {
    _mm512_set1_epi16(code as i16)
}

/// Compares land in mask registers, so nothing to narrow.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn eq_or_hits(first: &Broadcast, rest: &[Broadcast], codes: Vectors) -> Hits {
    let mut hit = join(
        _mm512_cmpeq_epi16_mask(codes[0], *first),
        _mm512_cmpeq_epi16_mask(codes[1], *first),
    );
    for token in rest {
        hit |= join(
            _mm512_cmpeq_epi16_mask(codes[0], *token),
            _mm512_cmpeq_epi16_mask(codes[1], *token),
        );
    }
    hit
}

/// Broadcast point probes and prepared ranges reused across scan blocks.
pub(crate) struct EqOr {
    /// One broadcast vector per point in the cover; none for a cover of ranges only.
    tokens: Vec<Broadcast>,
    /// Range starts and widths, already broadcast for vector comparisons.
    ranges: Vec<Held>,
}

impl Matcher for EqOr {
    unsafe fn new(cover: &ProbeCover) -> Self {
        // SAFETY: the caller guarantees AVX-512, which `broadcast` and `hold` need.
        unsafe {
            Self {
                tokens: cover
                    .points()
                    .iter()
                    .map(|&point| broadcast(point))
                    .collect(),
                ranges: cover.ranges().iter().map(|&range| hold(range)).collect(),
            }
        }
    }

    fn check(&self, codes: &Block, bits: &mut Mask, skip_empty: bool) -> bool {
        let (tokens, ranges) = (&self.tokens, &self.ranges);
        // SAFETY: `self` was built by `new`, whose caller guaranteed AVX-512.
        unsafe {
            if skip_empty {
                eq_or_mask::<true>(tokens, ranges, codes, bits)
            } else {
                eq_or_mask::<false>(tokens, ranges, codes, bits)
            }
        }
    }
}

/// Fill the block mask with point and range matches.
/// The target-feature boundary lets the comparison closure inline with its caller.
#[target_feature(enable = "avx512f,avx512bw")]
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
        |codes| check_ranges(eq_or_hits(first, rest, codes), ranges, codes),
    )
}

/// Four lookup tables ordered from the least to the most significant nibble.
struct Batch([Table; 4]);

/// The tables of batch `at` of `points`.
#[target_feature(enable = "avx512f,avx512bw")]
fn batch(points: &[Token], at: usize) -> Batch {
    Batch(nibble_rows(points, at).map(|row| table(row)))
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn table(row: [u8; 16]) -> Table {
    // SAFETY: the load reads the row's 16 bytes.
    unsafe { _mm512_broadcast_i32x4(_mm_loadu_si128(row.as_ptr().cast())) }
}

/// Pack low and high bytes separately and restore their original code order.
/// AVX-512 tests surviving point bits directly into the output mask.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn probe<const BATCHES: usize>(batches: &[Batch; BATCHES], codes: Vectors) -> Hits {
    let byte = _mm512_set1_epi16(0x00ff);
    let order = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    let low_byte = _mm512_permutexvar_epi64(
        order,
        _mm512_packus_epi16(
            _mm512_and_si512(codes[0], byte),
            _mm512_and_si512(codes[1], byte),
        ),
    );
    let high_byte = _mm512_permutexvar_epi64(
        order,
        _mm512_packus_epi16(
            _mm512_srli_epi16::<8>(codes[0]),
            _mm512_srli_epi16::<8>(codes[1]),
        ),
    );
    let nibble = _mm512_set1_epi8(0x0f);
    let nibbles = [
        _mm512_and_si512(low_byte, nibble),
        _mm512_and_si512(_mm512_srli_epi16::<4>(low_byte), nibble),
        _mm512_and_si512(high_byte, nibble),
        _mm512_and_si512(_mm512_srli_epi16::<4>(high_byte), nibble),
    ];
    let pair = |batch: &Batch| {
        (
            _mm512_and_si512(
                _mm512_shuffle_epi8(batch.0[0], nibbles[0]),
                _mm512_shuffle_epi8(batch.0[1], nibbles[1]),
            ),
            _mm512_and_si512(
                _mm512_shuffle_epi8(batch.0[2], nibbles[2]),
                _mm512_shuffle_epi8(batch.0[3], nibbles[3]),
            ),
        )
    };
    let (low, high) = pair(&batches[0]);
    if BATCHES == 1 {
        return _mm512_test_epi8_mask(low, high);
    }
    let mut hit = _mm512_and_si512(low, high);
    for batch in &batches[1..] {
        let (low, high) = pair(batch);
        hit = _mm512_or_si512(hit, _mm512_and_si512(low, high));
    }
    _mm512_test_epi8_mask(hit, hit)
}

/// Prepared point batches and ranges for repeated block checks.
/// `BATCHES` must be positive and large enough to hold every point in the cover;
/// planning and dispatch choose the supported specialization.
pub(crate) struct NibbleN8<const BATCHES: usize> {
    /// Up to eight point probes per batch; unused entries contribute no bits.
    batches: [Batch; BATCHES],
    /// Prepared inclusive ranges, added after point matching.
    ranges: Vec<Held>,
}

impl<const BATCHES: usize> Matcher for NibbleN8<BATCHES> {
    unsafe fn new(cover: &ProbeCover) -> Self {
        let points = cover.points();
        // SAFETY: the caller guarantees AVX-512, which `batch` and `hold` need.
        unsafe {
            Self {
                batches: std::array::from_fn(|at| batch(points, at)),
                ranges: cover.ranges().iter().map(|&range| hold(range)).collect(),
            }
        }
    }

    fn check(&self, codes: &Block, bits: &mut Mask, skip_empty: bool) -> bool {
        let (batches, ranges) = (&self.batches, &self.ranges);
        // SAFETY: `self` was built by `new`, whose caller guaranteed AVX-512.
        unsafe {
            if skip_empty {
                nibble_mask::<BATCHES, true>(batches, ranges, codes, bits)
            } else {
                nibble_mask::<BATCHES, false>(batches, ranges, codes, bits)
            }
        }
    }
}

/// Fill a block mask with nibble-table and range matches.
/// The target-feature boundary keeps the probe closure inside the vector loop.
#[target_feature(enable = "avx512f,avx512bw")]
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
        |codes| check_ranges(probe(batches, codes), ranges, codes),
    )
}

/// The AVX-512BW matchers.
pub(crate) struct Avx512;

impl Kernels for Avx512 {
    const ISA: Isa = Isa::Avx512Bw;
    type EqOr = EqOr;
    type NibbleN8<const BATCHES: usize> = NibbleN8<BATCHES>;
}
