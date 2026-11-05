// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! NEON kernels.
//!
//! 64 codes fill eight 128-bit vectors of `u16` lanes. Comparisons yield
//! 0xFFFF/0x0000 lanes, which `narrow` reduces to one byte per code; `movemask`
//! packs 128 such bytes into two mask words with lane bits and pairwise adds.
//!
//! The items are described in the parent module; only what is specific to NEON
//! is documented here.

use std::arch::aarch64::*;

use super::{Block, Kernels, Mask, Matcher, nibble_rows};
use crate::core::types::{Token, TokenRange};
use crate::search::substring::ProbeCover;
use crate::search::substring::probe::scan::Isa;

/// Membership results for 64 codes: one 0xFF/0x00 byte per code.
type Hits = [uint8x16_t; 4];

type Vectors = [uint16x8_t; 8];

type Held = (uint16x8_t, uint16x8_t);

type Broadcast = uint16x8_t;

/// A 16-byte shuffle row.
type NibbleTable = uint8x16_t;

#[inline]
#[target_feature(enable = "neon")]
fn load(codes: &[Token; 64]) -> Vectors {
    // SAFETY: load `i` reads eight of the 64 codes, starting at code eight * i.
    std::array::from_fn(|i| unsafe { vld1q_u16(codes.as_ptr().add(8 * i)) })
}

#[inline]
#[target_feature(enable = "neon")]
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
#[target_feature(enable = "neon")]
fn or(hit: Hits, with: Hits) -> Hits {
    std::array::from_fn(|at| vorrq_u8(hit[at], with[at]))
}

#[inline]
#[target_feature(enable = "neon")]
fn any(a: Hits, b: Hits) -> bool {
    let a = vorrq_u8(vorrq_u8(a[0], a[1]), vorrq_u8(a[2], a[3]));
    let b = vorrq_u8(vorrq_u8(b[0], b[1]), vorrq_u8(b[2], b[3]));
    vmaxvq_u8(vorrq_u8(a, b)) != 0
}

/// Keeps the low byte of each 0xFFFF/0x0000 comparison lane, in code order.
#[inline]
#[target_feature(enable = "neon")]
fn narrow(hit: [uint16x8_t; 8]) -> Hits {
    let byte = |lo: uint16x8_t, hi: uint16x8_t| {
        vuzp1q_u8(vreinterpretq_u8_u16(lo), vreinterpretq_u8_u16(hi))
    };
    [
        byte(hit[0], hit[1]),
        byte(hit[2], hit[3]),
        byte(hit[4], hit[5]),
        byte(hit[6], hit[7]),
    ]
}

/// Packs 128 hit bytes into two mask words with lane bits and pairwise adds.
#[inline]
#[target_feature(enable = "neon")]
fn movemask(bits: &mut [u64; 2], a: Hits, b: Hits) {
    const LANE_BIT: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    // SAFETY: LANE_BIT is 16 bytes.
    let lane_bit = unsafe { vld1q_u8(LANE_BIT.as_ptr()) };
    let half = |hit: Hits| {
        let lo = vpaddq_u8(vandq_u8(hit[0], lane_bit), vandq_u8(hit[1], lane_bit));
        let hi = vpaddq_u8(vandq_u8(hit[2], lane_bit), vandq_u8(hit[3], lane_bit));
        vpaddq_u8(lo, hi)
    };
    let lanes = vpaddq_u8(half(a), half(b));
    // SAFETY: `bits` is the two words, 16 bytes, that the store writes.
    unsafe { vst1q_u64(bits.as_mut_ptr(), vreinterpretq_u64_u8(lanes)) };
}

#[inline]
#[target_feature(enable = "neon")]
fn hold(range: TokenRange) -> Held {
    (
        vdupq_n_u16(range.begin),
        vdupq_n_u16(range.last - range.begin),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn inside((lo, width): Held, codes: Vectors) -> Hits {
    let mut hit = codes;
    for hit in &mut hit {
        *hit = vcleq_u16(vsubq_u16(*hit, lo), width);
    }
    narrow(hit)
}

#[inline]
#[target_feature(enable = "neon")]
fn check_ranges(mut hit: Hits, held: &[Held], codes: Vectors) -> Hits {
    for &held in held {
        hit = or(hit, inside(held, codes));
    }
    hit
}

#[target_feature(enable = "neon")]
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
#[target_feature(enable = "neon")]
fn broadcast(code: Token) -> Broadcast {
    vdupq_n_u16(code)
}

#[target_feature(enable = "neon")]
#[inline]
fn eq_or_hits(first: &Broadcast, rest: &[Broadcast], codes: Vectors) -> Hits {
    let mut hit = codes;
    for hit in &mut hit {
        *hit = vceqq_u16(*hit, *first);
    }
    for token in rest {
        for (hit, &codes) in hit.iter_mut().zip(&codes) {
            *hit = vorrq_u16(*hit, vceqq_u16(codes, *token));
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
        // SAFETY: the caller guarantees NEON, which `broadcast` and `hold` need.
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
        // SAFETY: `self` was built by `new`, whose caller guaranteed NEON.
        unsafe {
            if skip_empty {
                eq_or_mask::<true>(points, ranges, codes, bits)
            } else {
                eq_or_mask::<false>(points, ranges, codes, bits)
            }
        }
    }
}

#[target_feature(enable = "neon")]
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

#[target_feature(enable = "neon")]
fn batch(points: &[Token], at: usize) -> Batch {
    Batch(nibble_rows(points, at).map(|row| nibble_table(row)))
}

#[inline]
#[target_feature(enable = "neon")]
fn nibble_table(row: [u8; 16]) -> NibbleTable {
    // SAFETY: the load reads the row's 16 bytes.
    unsafe { vld1q_u8(row.as_ptr()) }
}

/// `vuzp1q`/`vuzp2q` split each 16 codes into their low and high bytes, in code
/// order, and `vtstq` turns a byte with a point bit left into a 0xFF hit.
#[inline]
#[target_feature(enable = "neon")]
fn nibble_hits<const BATCHES: usize>(batches: &[Batch; BATCHES], codes: Vectors) -> Hits {
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

pub(crate) struct Nibble<const BATCHES: usize> {
    batches: [Batch; BATCHES],
    ranges: Vec<Held>,
}

impl<const BATCHES: usize> Matcher for Nibble<BATCHES> {
    unsafe fn new(cover: &ProbeCover) -> Self {
        let points = cover.points();
        // SAFETY: the caller guarantees NEON, which `batch` and `hold` need.
        unsafe {
            Self {
                batches: std::array::from_fn(|at| batch(points, at)),
                ranges: cover.ranges().iter().map(|&range| hold(range)).collect(),
            }
        }
    }

    fn check(&self, codes: &Block, bits: &mut Mask, skip_empty: bool) -> bool {
        let (batches, ranges) = (&self.batches, &self.ranges);
        // SAFETY: `self` was built by `new`, whose caller guaranteed NEON.
        unsafe {
            if skip_empty {
                nibble_mask::<BATCHES, true>(batches, ranges, codes, bits)
            } else {
                nibble_mask::<BATCHES, false>(batches, ranges, codes, bits)
            }
        }
    }
}

#[target_feature(enable = "neon")]
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

/// The NEON matchers.
pub(crate) struct Neon;

impl Kernels for Neon {
    const ISA: Isa = Isa::Neon;
    type EqOr = EqOr;
    type Nibble<const BATCHES: usize> = Nibble<BATCHES>;
}
