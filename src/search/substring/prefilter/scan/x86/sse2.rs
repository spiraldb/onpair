// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Baseline x86-64 SSE2 prefilter kernels.

use super::super::sink::{LaneMask, RowSink};
use super::super::template::{DYN, Isa, scan_dynamic, scan_fixed};
use crate::core::offset::Offset;
use crate::core::types::{Token, TokenRange};
use crate::search::substring::prefilter::cover::ProbeCover;

use core::arch::x86_64::{
    __m128i, _mm_cmpeq_epi16, _mm_loadu_si128, _mm_movemask_epi8, _mm_or_si128, _mm_packs_epi16,
    _mm_set1_epi16, _mm_setzero_si128, _mm_sub_epi16, _mm_subs_epu16,
};

#[inline(always)]
fn fixed_mask32<const POINTS: usize, const RANGES: usize>(
    ptr: *const Token,
    base: usize,
    points: &[__m128i],
    ranges: &[(__m128i, __m128i)],
) -> u32 {
    debug_assert_eq!(points.len(), POINTS);
    debug_assert_eq!(ranges.len(), RANGES);
    // SAFETY: the fixed-shape prologue creates exactly these probe counts.
    let points = unsafe { points.get_unchecked(..POINTS) };
    // SAFETY: the fixed-shape prologue creates exactly these probe counts.
    let ranges = unsafe { ranges.get_unchecked(..RANGES) };

    let zero = unsafe { _mm_setzero_si128() };
    let matching = |offset: usize| unsafe {
        let value = _mm_loadu_si128(ptr.add(base + offset).cast::<__m128i>());
        let mut mask = zero;
        for &point in points {
            mask = _mm_or_si128(mask, _mm_cmpeq_epi16(value, point));
        }
        for &(begin, span) in ranges {
            let excess = _mm_subs_epu16(_mm_sub_epi16(value, begin), span);
            mask = _mm_or_si128(mask, _mm_cmpeq_epi16(excess, zero));
        }
        mask
    };
    let m0 = matching(0);
    let m1 = matching(8);
    let m2 = matching(16);
    let m3 = matching(24);
    let lo = unsafe { _mm_movemask_epi8(_mm_packs_epi16(m0, m1)) as u16 };
    let hi = unsafe { _mm_movemask_epi8(_mm_packs_epi16(m2, m3)) as u16 };
    u32::from(lo) | (u32::from(hi) << 16)
}

/// Preserve the arbitrary-shape fallback's probe ordering and 32-code early
/// gate while sharing the outer 64-code walk.
#[inline(always)]
fn dynamic_mask32(ptr: *const Token, points: &[__m128i], ranges: &[(__m128i, __m128i)]) -> u32 {
    unsafe {
        let zero = _mm_setzero_si128();
        let values = [
            _mm_loadu_si128(ptr.cast()),
            _mm_loadu_si128(ptr.add(8).cast()),
            _mm_loadu_si128(ptr.add(16).cast()),
            _mm_loadu_si128(ptr.add(24).cast()),
        ];
        let mut masks = [zero; 4];
        for &point in points {
            for (mask, &value) in masks.iter_mut().zip(&values) {
                *mask = _mm_or_si128(*mask, _mm_cmpeq_epi16(value, point));
            }
        }
        for &(begin, span) in ranges {
            for (mask, &value) in masks.iter_mut().zip(&values) {
                let excess = _mm_subs_epu16(_mm_sub_epi16(value, begin), span);
                *mask = _mm_or_si128(*mask, _mm_cmpeq_epi16(excess, zero));
            }
        }
        let any = _mm_or_si128(
            _mm_or_si128(masks[0], masks[1]),
            _mm_or_si128(masks[2], masks[3]),
        );
        if _mm_movemask_epi8(any) == 0 {
            return 0;
        }
        let lo = _mm_movemask_epi8(_mm_packs_epi16(masks[0], masks[1])) as u16;
        let hi = _mm_movemask_epi8(_mm_packs_epi16(masks[2], masks[3])) as u16;
        u32::from(lo) | (u32::from(hi) << 16)
    }
}

struct Sse2;

impl Isa for Sse2 {
    const BLOCK: usize = 64;

    type Point = __m128i;
    type Range = (__m128i, __m128i);
    type Hits = u64;
    const NO_HITS: Self::Hits = 0;

    #[inline(always)]
    fn point(token: Token) -> Self::Point {
        unsafe { _mm_set1_epi16(token as i16) }
    }

    #[inline(always)]
    fn range(range: TokenRange) -> Self::Range {
        unsafe {
            (
                _mm_set1_epi16(range.begin as i16),
                _mm_set1_epi16(range.last.wrapping_sub(range.begin) as i16),
            )
        }
    }

    #[inline(always)]
    unsafe fn block<const POINTS: usize, const RANGES: usize>(
        codes: *const Token,
        points: &[Self::Point],
        ranges: &[Self::Range],
    ) -> Self::Hits {
        let (lo, hi) = if POINTS == DYN {
            debug_assert_eq!(RANGES, DYN);
            (
                dynamic_mask32(codes, points, ranges),
                dynamic_mask32(unsafe { codes.add(32) }, points, ranges),
            )
        } else {
            (
                fixed_mask32::<POINTS, RANGES>(codes, 0, points, ranges),
                fixed_mask32::<POINTS, RANGES>(codes, 32, points, ranges),
            )
        };
        u64::from(lo) | (u64::from(hi) << 32)
    }

    #[inline(always)]
    fn any(hits: Self::Hits) -> bool {
        hits != Self::NO_HITS
    }

    #[inline(always)]
    fn emit<O: Offset>(base: usize, hits: Self::Hits, sink: &mut RowSink<'_, O>) {
        sink.mark_mask(base, LaneMask::from_bits(hits));
    }
}

/// Eager compact-mask SSE2 leaf for the selected small cover shapes.
#[target_feature(enable = "sse2")]
#[inline(never)]
pub(in crate::search::substring::prefilter::scan) fn scan_sse2_fixed<
    O: Offset,
    const POINTS: usize,
    const RANGES: usize,
>(
    codes: &[Token],
    row_offsets: &[O],
    cover: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    unsafe {
        scan_fixed::<Sse2, O, POINTS, RANGES, 1>(codes, row_offsets, cover, sparse_row_mapping, out)
    }
}

/// SSE2 fallback for arbitrary cover shapes, processing eight lanes at a time.
pub(in crate::search::substring::prefilter::scan) fn scan_sse2_generic<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    cover: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    // SAFETY: SSE2 is part of the x86-64 baseline ISA.
    unsafe { scan_dynamic::<Sse2, O>(codes, row_offsets, cover, sparse_row_mapping, out) };
}

/// Direct generic-kernel entry retained for architecture correctness tests.
#[cfg(test)]
pub(in crate::search::substring::prefilter) fn scan_sse2<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    cover: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    scan_sse2_generic(codes, row_offsets, cover, sparse_row_mapping, out);
}
