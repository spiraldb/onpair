// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Match point probes with four nibble tables per batch of up to eight IDs.
//!
//! Each token ID is a `u16`, split into four 4-bit nibbles. A point owns one
//! bit in each table entry selected by its nibbles. Looking up the input code's
//! four nibbles and ANDing their entries leaves a bit only if the same point
//! matches all four positions. This is exact token membership.
//!
//! Batches are ORed together, then range matches are added. The planner limits
//! batch counts to bound register use. `shared::words` handles block loading
//! and mask packing, as it does for the other vector matchers.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::range::{Held, check_ranges, hold};
use super::shared::{Hits, Vectors, words};
use super::{Block, Mask, Matcher};
use crate::core::types::Token;
use crate::search::substring::ProbeCover;

use super::PER_BATCH;

/// A 16-byte shuffle row, broadcast into every 128-bit lane on x86 since
/// `vpshufb` indexes within lanes.
#[cfg(target_arch = "aarch64")]
type Table = uint8x16_t;
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512bw")))]
type Table = __m256i;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
type Table = __m512i;

/// Four lookup tables ordered from the least to the most significant nibble.
pub(in crate::search::substring::scan) struct Batch([Table; 4]);

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

#[cfg(target_arch = "aarch64")]
fn table(row: [u8; 16]) -> Table {
    // SAFETY: neon is baseline on aarch64; the row is the load's 16 bytes.
    unsafe { vld1q_u8(row.as_ptr()) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn probe<const BATCHES: usize>(batches: &[Batch; BATCHES], codes: Vectors) -> Hits {
    let nibble = vdupq_n_u8(0x0f);
    std::array::from_fn(|at| {
        let lo = vreinterpretq_u8_u16(codes[2 * at]);
        let hi = vreinterpretq_u8_u16(codes[2 * at + 1]);
        let (low_byte, high_byte) = (vuzp1q_u8(lo, hi), vuzp2q_u8(lo, hi));
        let nibbles = [
            vandq_u8(low_byte, nibble),
            vshrq_n_u8::<4>(low_byte),
            vandq_u8(high_byte, nibble),
            vshrq_n_u8::<4>(high_byte),
        ];
        let pair = |batch: &Batch| {
            (
                vandq_u8(
                    vqtbl1q_u8(batch.0[0], nibbles[0]),
                    vqtbl1q_u8(batch.0[1], nibbles[1]),
                ),
                vandq_u8(
                    vqtbl1q_u8(batch.0[2], nibbles[2]),
                    vqtbl1q_u8(batch.0[3], nibbles[3]),
                ),
            )
        };
        let (low, high) = pair(&batches[0]);
        if BATCHES == 1 {
            return vtstq_u8(low, high);
        }
        let mut hit = vandq_u8(low, high);
        for batch in &batches[1..] {
            let (low, high) = pair(batch);
            hit = vorrq_u8(hit, vandq_u8(low, high));
        }
        vtstq_u8(hit, hit)
    })
}

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512bw")))]
fn table(row: [u8; 16]) -> Table {
    // SAFETY: avx2, checked by `detect_isa`; the row is the load's 16 bytes.
    unsafe { _mm256_broadcastsi128_si256(_mm_loadu_si128(row.as_ptr().cast())) }
}

/// Pack low and high bytes separately and restore their original code order.
/// Convert surviving point bits to 0xFF lanes for the later movemask.
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512bw")))]
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

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
fn table(row: [u8; 16]) -> Table {
    // SAFETY: the row is the load's 16 bytes.
    unsafe { _mm512_broadcast_i32x4(_mm_loadu_si128(row.as_ptr().cast())) }
}

/// Pack low and high bytes separately and restore their original code order.
/// AVX-512 tests surviving point bits directly into the output mask.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn probe<const BATCHES: usize>(batches: &[Batch; BATCHES], codes: Vectors) -> Hits {
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
pub(in crate::search::substring::scan) struct NibbleN8<
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
            mask::<BATCHES, SKIP_MOVEMASK_IF_NO_MATCH>(&self.batches, &self.ranges, codes, bits)
        }
    }
}

/// Fill a block mask with nibble-table and range matches.
/// The target-feature boundary keeps the probe closure inside the vector loop.
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
fn mask<const BATCHES: usize, const SKIP_MOVEMASK_IF_NO_MATCH: bool>(
    batches: &[Batch; BATCHES],
    ranges: &[Held],
    codes: &Block,
    bits: &mut Mask,
) -> bool {
    words::<SKIP_MOVEMASK_IF_NO_MATCH>(codes, bits, |codes| {
        // SAFETY: the instruction set, inherited from this function.
        unsafe { check_ranges(probe(batches, codes), ranges, codes) }
    })
}
