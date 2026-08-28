// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Baseline x86-64 SSE2 prefilter kernels.

use super::super::sink::{LaneMask, RowSink, scan_tail};
use crate::core::offset::Offset;
use crate::core::types::{Token, TokenRange};
use crate::search::prefilter::cover::ProbeCover;

#[inline(always)]
fn fixed_mask32<const POINTS: usize, const RANGES: usize>(
    ptr: *const Token,
    base: usize,
    points: &[core::arch::x86_64::__m128i; POINTS],
    ranges: &[(core::arch::x86_64::__m128i, core::arch::x86_64::__m128i); RANGES],
) -> u32 {
    use core::arch::x86_64::{
        __m128i, _mm_cmpeq_epi16, _mm_loadu_si128, _mm_movemask_epi8, _mm_or_si128,
        _mm_packs_epi16, _mm_setzero_si128, _mm_sub_epi16, _mm_subs_epu16,
    };

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

/// Eager compact-mask SSE2 leaf for the selected small cover shapes.
#[target_feature(enable = "sse2")]
#[inline(never)]
pub(in crate::search::prefilter::scan) fn scan_sse2_fixed<
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
    use core::arch::x86_64::{__m128i, _mm_set1_epi16};

    debug_assert_eq!((cover.points.len(), cover.ranges.len()), (POINTS, RANGES));
    let points: [__m128i; POINTS] =
        std::array::from_fn(|index| _mm_set1_epi16(cover.points[index] as i16));
    let ranges: [(__m128i, __m128i); RANGES] = std::array::from_fn(|index| {
        let TokenRange { begin, last } = cover.ranges[index];
        (
            _mm_set1_epi16(begin as i16),
            _mm_set1_epi16(last.wrapping_sub(begin) as i16),
        )
    });

    let mut sink = RowSink::new(row_offsets, out, sparse_row_mapping);
    let ptr = codes.as_ptr();
    let mut base = 0;
    while base + 64 <= codes.len() {
        let lo = u64::from(fixed_mask32(ptr, base, &points, &ranges));
        let hi = u64::from(fixed_mask32(ptr, base + 32, &points, &ranges));
        let mask = lo | (hi << 32);
        if mask != 0 {
            sink.mark_mask(base, LaneMask::from_bits(mask));
        }
        base += 64;
    }
    while base + 32 <= codes.len() {
        let mask = u64::from(fixed_mask32(ptr, base, &points, &ranges));
        if mask != 0 {
            sink.mark_mask(base, LaneMask::from_bits(mask));
        }
        base += 32;
    }
    scan_tail(codes, cover, base, &mut sink);
}

/// Shared four-vector walk retained by the arbitrary-shape fallback.
#[inline(always)]
fn scan_four_vectors<O: Offset>(
    codes: &[Token],
    sink: &mut RowSink<'_, O>,
    mut matching_masks: impl FnMut(
        core::arch::x86_64::__m128i,
        core::arch::x86_64::__m128i,
        core::arch::x86_64::__m128i,
        core::arch::x86_64::__m128i,
    ) -> [core::arch::x86_64::__m128i; 4],
) -> usize {
    use core::arch::x86_64::{
        __m128i, _mm_loadu_si128, _mm_movemask_epi8, _mm_or_si128, _mm_packs_epi16,
    };

    let ptr = codes.as_ptr();
    let mut base = 0;
    while base + 32 <= codes.len() {
        let [m0, m1, m2, m3] = unsafe {
            matching_masks(
                _mm_loadu_si128(ptr.add(base).cast::<__m128i>()),
                _mm_loadu_si128(ptr.add(base + 8).cast::<__m128i>()),
                _mm_loadu_si128(ptr.add(base + 16).cast::<__m128i>()),
                _mm_loadu_si128(ptr.add(base + 24).cast::<__m128i>()),
            )
        };
        unsafe {
            let any = _mm_or_si128(_mm_or_si128(m0, m1), _mm_or_si128(m2, m3));
            if _mm_movemask_epi8(any) != 0 {
                let lo = _mm_movemask_epi8(_mm_packs_epi16(m0, m1)) as u16;
                let hi = _mm_movemask_epi8(_mm_packs_epi16(m2, m3)) as u16;
                sink.mark_mask(
                    base,
                    LaneMask::from_bits(u64::from(lo) | (u64::from(hi) << 16)),
                );
            }
        }
        base += 32;
    }
    base
}

/// SSE2 fallback for arbitrary cover shapes, processing eight lanes at a time.
pub(in crate::search::prefilter::scan) fn scan_sse2_generic<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    cover: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    use core::arch::x86_64::{
        __m128i, _mm_cmpeq_epi16, _mm_or_si128, _mm_set1_epi16, _mm_setzero_si128, _mm_sub_epi16,
        _mm_subs_epu16,
    };

    let zero = unsafe { _mm_setzero_si128() };
    let points: Vec<__m128i> = cover
        .points
        .iter()
        .map(|&point| unsafe { _mm_set1_epi16(point as i16) })
        .collect();
    let ranges: Vec<(__m128i, __m128i)> = cover
        .ranges
        .iter()
        .map(|range| unsafe {
            (
                _mm_set1_epi16(range.begin as i16),
                _mm_set1_epi16((range.last - range.begin) as i16),
            )
        })
        .collect();

    let mut sink = RowSink::new(row_offsets, out, sparse_row_mapping);
    let base = scan_four_vectors(codes, &mut sink, |v0, v1, v2, v3| unsafe {
        let values = [v0, v1, v2, v3];
        let mut masks = [zero; 4];
        for &point in &points {
            for (mask, &value) in masks.iter_mut().zip(&values) {
                *mask = _mm_or_si128(*mask, _mm_cmpeq_epi16(value, point));
            }
        }
        for &(begin, span) in &ranges {
            for (mask, &value) in masks.iter_mut().zip(&values) {
                let excess = _mm_subs_epu16(_mm_sub_epi16(value, begin), span);
                *mask = _mm_or_si128(*mask, _mm_cmpeq_epi16(excess, zero));
            }
        }
        masks
    });
    scan_tail(codes, cover, base, &mut sink);
}

/// Direct generic-kernel entry retained for architecture correctness tests.
#[cfg(test)]
pub(in crate::search::prefilter) fn scan_sse2<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    cover: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    scan_sse2_generic(codes, row_offsets, cover, sparse_row_mapping, out);
}
