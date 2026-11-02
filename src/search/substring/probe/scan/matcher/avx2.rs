// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! AVX2 kernels.
//!
//! 64 codes fill four 256-bit vectors of `u16` lanes. Comparisons yield
//! 0xFFFF/0x0000 lanes, which `narrow` packs to one byte per code, restoring code
//! order across 128-bit lanes; `movemask` extracts one bit per byte.
//!
//! The items are described in the parent module; only what is specific to AVX2
//! is documented here.

use std::arch::x86_64::*;

use super::{Block, Kernels, Mask, Matcher, nibble_rows};
use crate::core::types::{Token, TokenRange};
use crate::search::substring::ProbeCover;
use crate::search::substring::probe::scan::Isa;

/// Membership results for 64 codes: one 0xFF/0x00 byte per code.
type Hits = [__m256i; 2];

type Vectors = [__m256i; 4];

type Held = (__m256i, __m256i);

type Broadcast = __m256i;

/// A 16-byte shuffle row, broadcast into both 128-bit lanes since `vpshufb`
/// indexes within lanes.
type NibbleTable = __m256i;

#[inline]
#[target_feature(enable = "avx2")]
fn load(codes: &[Token; 64]) -> Vectors {
    // SAFETY: load `i` reads 16 of the 64 codes, starting at code 16 * i.
    std::array::from_fn(|i| unsafe { _mm256_loadu_si256(codes.as_ptr().add(16 * i).cast()) })
}

#[inline]
#[target_feature(enable = "avx2")]
fn words<const SKIP_EMPTY: bool>(
    codes: &Block,
    bits: &mut Mask,
    hits: impl Fn(Vectors) -> Hits,
) -> bool {
    let mut written = false;
    // Two 64-code groups per pair of mask words.
    let groups = codes.as_chunks::<64>().0.as_chunks::<2>().0;
    for ([lo, hi], bits) in groups.iter().zip(bits.as_chunks_mut::<2>().0) {
        let (a, b) = (hits(load(lo)), hits(load(hi)));
        if SKIP_EMPTY && !any(a, b) {
            *bits = [0; 2];
            continue;
        }
        written = true;
        movemask(bits, a, b);
    }
    written
}

#[inline]
#[target_feature(enable = "avx2")]
fn or(hit: Hits, with: Hits) -> Hits {
    std::array::from_fn(|at| _mm256_or_si256(hit[at], with[at]))
}

#[inline]
#[target_feature(enable = "avx2")]
fn any(a: Hits, b: Hits) -> bool {
    let or = _mm256_or_si256(or(a, b)[0], or(a, b)[1]);
    _mm256_testz_si256(or, or) == 0
}

/// Packs 16-bit comparison lanes into bytes, then restores code order:
/// `packs_epi16` interleaves its inputs' 128-bit lanes, and the permutation
/// undoes it.
#[inline]
#[target_feature(enable = "avx2")]
fn narrow(hit: [__m256i; 4]) -> Hits {
    let byte = |lo, hi| _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi16(lo, hi));
    [byte(hit[0], hit[1]), byte(hit[2], hit[3])]
}

/// Extracts one bit per hit byte into the two mask words.
#[inline]
#[target_feature(enable = "avx2")]
fn movemask(bits: &mut [u64; 2], a: Hits, b: Hits) {
    let word = |hit: Hits| {
        let lo = _mm256_movemask_epi8(hit[0]) as u32;
        let hi = _mm256_movemask_epi8(hit[1]) as u32;
        u64::from(lo) | u64::from(hi) << 32
    };
    bits[0] = word(a);
    bits[1] = word(b);
}

#[inline]
#[target_feature(enable = "avx2")]
fn hold(range: TokenRange) -> Held {
    (
        _mm256_set1_epi16(range.begin as i16),
        _mm256_set1_epi16((range.last - range.begin) as i16),
    )
}

/// No unsigned u16 compare below AVX-512: `min(x, width) == x` is `x <= width`.
#[inline]
#[target_feature(enable = "avx2")]
fn inside((lo, width): Held, codes: Vectors) -> Hits {
    let mut hit = codes;
    for hit in &mut hit {
        let inside = _mm256_sub_epi16(*hit, lo);
        *hit = _mm256_cmpeq_epi16(_mm256_min_epu16(inside, width), inside);
    }
    narrow(hit)
}

#[inline]
#[target_feature(enable = "avx2")]
fn check_ranges(mut hit: Hits, held: &[Held], codes: Vectors) -> Hits {
    for &held in held {
        hit = or(hit, inside(held, codes));
    }
    hit
}

#[target_feature(enable = "avx2")]
fn ranges_mask<const SKIP_EMPTY: bool>(held: &[Held], codes: &Block, bits: &mut Mask) -> bool {
    let Some((&first, rest)) = held.split_first() else {
        bits.fill(0);
        return false;
    };
    words::<SKIP_EMPTY>(
        codes,
        bits,
        #[inline(always)]
        |codes| check_ranges(inside(first, codes), rest, codes),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn broadcast(code: Token) -> Broadcast {
    _mm256_set1_epi16(code as i16)
}

#[target_feature(enable = "avx2")]
#[inline]
fn eq_or_hits(first: &Broadcast, rest: &[Broadcast], codes: Vectors) -> Hits {
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

pub(crate) struct EqOr {
    points: Vec<Broadcast>,
    ranges: Vec<Held>,
}

impl Matcher for EqOr {
    unsafe fn new(cover: &ProbeCover) -> Self {
        // SAFETY: the caller guarantees AVX2, which `broadcast` and `hold` need.
        unsafe {
            Self {
                points: cover
                    .points()
                    .iter()
                    .map(|&point| broadcast(point))
                    .collect(),
                ranges: cover.ranges().iter().map(|&range| hold(range)).collect(),
            }
        }
    }

    fn check(&self, codes: &Block, bits: &mut Mask, skip_empty: bool) -> bool {
        let (points, ranges) = (&self.points, &self.ranges);
        // SAFETY: `self` was built by `new`, whose caller guaranteed AVX2.
        unsafe {
            if skip_empty {
                eq_or_mask::<true>(points, ranges, codes, bits)
            } else {
                eq_or_mask::<false>(points, ranges, codes, bits)
            }
        }
    }
}

#[target_feature(enable = "avx2")]
fn eq_or_mask<const SKIP_EMPTY: bool>(
    points: &[Broadcast],
    ranges: &[Held],
    codes: &Block,
    bits: &mut Mask,
) -> bool {
    let Some((first, rest)) = points.split_first() else {
        return ranges_mask::<SKIP_EMPTY>(ranges, codes, bits);
    };
    words::<SKIP_EMPTY>(
        codes,
        bits,
        #[inline(always)]
        |codes| check_ranges(eq_or_hits(first, rest, codes), ranges, codes),
    )
}

struct Batch([NibbleTable; 4]);

#[target_feature(enable = "avx2")]
fn batch(points: &[Token], at: usize) -> Batch {
    Batch(nibble_rows(points, at).map(|row| nibble_table(row)))
}

#[inline]
#[target_feature(enable = "avx2")]
fn nibble_table(row: [u8; 16]) -> NibbleTable {
    // SAFETY: the load reads the row's 16 bytes.
    unsafe { _mm256_broadcastsi128_si256(_mm_loadu_si128(row.as_ptr().cast())) }
}

/// `packus` splits each 32 codes into their low and high bytes, and
/// `permute4x64` restores their code order across the 128-bit lanes. Comparing
/// twice with zero turns a byte with a point bit left into a 0xFF hit.
#[inline]
#[target_feature(enable = "avx2")]
fn nibble_hits<const BATCHES: usize>(batches: &[Batch; BATCHES], codes: Vectors) -> Hits {
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

pub(crate) struct Nibble<const BATCHES: usize> {
    batches: [Batch; BATCHES],
    ranges: Vec<Held>,
}

impl<const BATCHES: usize> Matcher for Nibble<BATCHES> {
    unsafe fn new(cover: &ProbeCover) -> Self {
        let points = cover.points();
        // SAFETY: the caller guarantees AVX2, which `batch` and `hold` need.
        unsafe {
            Self {
                batches: std::array::from_fn(|at| batch(points, at)),
                ranges: cover.ranges().iter().map(|&range| hold(range)).collect(),
            }
        }
    }

    fn check(&self, codes: &Block, bits: &mut Mask, skip_empty: bool) -> bool {
        let (batches, ranges) = (&self.batches, &self.ranges);
        // SAFETY: `self` was built by `new`, whose caller guaranteed AVX2.
        unsafe {
            if skip_empty {
                nibble_mask::<BATCHES, true>(batches, ranges, codes, bits)
            } else {
                nibble_mask::<BATCHES, false>(batches, ranges, codes, bits)
            }
        }
    }
}

#[target_feature(enable = "avx2")]
fn nibble_mask<const BATCHES: usize, const SKIP_EMPTY: bool>(
    batches: &[Batch; BATCHES],
    ranges: &[Held],
    codes: &Block,
    bits: &mut Mask,
) -> bool {
    words::<SKIP_EMPTY>(
        codes,
        bits,
        #[inline(always)]
        |codes| check_ranges(nibble_hits(batches, codes), ranges, codes),
    )
}

/// The AVX2 matchers.
pub(crate) struct Avx2;

impl Kernels for Avx2 {
    const ISA: Isa = Isa::Avx2;
    type EqOr = EqOr;
    type Nibble<const BATCHES: usize> = Nibble<BATCHES>;
}
