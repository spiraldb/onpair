// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! AVX2 prefilter kernels.

use super::super::sink::{LaneMask, RowSink, scan_tail};
use crate::core::offset::Offset;
use crate::core::types::{Token, TokenRange};
use crate::search::prefilter::cover::ProbeCover;

use core::arch::x86_64::{
    __m256i, _mm256_andnot_si256, _mm256_cmpeq_epi16, _mm256_cmpgt_epi16, _mm256_loadu_si256,
    _mm256_movemask_epi8, _mm256_or_si256, _mm256_packs_epi16, _mm256_permute4x64_epi64,
    _mm256_set1_epi16, _mm256_setzero_si256, _mm256_sub_epi16, _mm256_testz_si256,
    _mm256_xor_si256,
};

const BLOCK_CODES: usize = 64;

#[target_feature(enable = "avx2")]
#[inline]
fn compact_masks(m0: __m256i, m1: __m256i, m2: __m256i, m3: __m256i) -> u64 {
    let packed01 = _mm256_permute4x64_epi64::<0xd8>(_mm256_packs_epi16(m0, m1));
    let packed23 = _mm256_permute4x64_epi64::<0xd8>(_mm256_packs_epi16(m2, m3));
    u64::from(_mm256_movemask_epi8(packed01) as u32)
        | (u64::from(_mm256_movemask_epi8(packed23) as u32) << 32)
}

/// Produce the exact hit lanes for four consecutive AVX2 vectors.
///
/// # Safety
/// The caller must execute with AVX2 enabled and make `codes..codes + 64`
/// readable. The fixed scan leaf establishes both conditions.
#[allow(
    unsafe_op_in_unsafe_fn,
    reason = "the AVX2 caller and 64-code bound satisfy every intrinsic and load"
)]
#[inline(always)]
unsafe fn fixed_mask64<const POINTS: usize, const RANGES: usize>(
    codes: *const Token,
    points: &[__m256i; POINTS],
    ranges: &[(__m256i, __m256i); RANGES],
) -> u64 {
    let zero = _mm256_setzero_si256();
    let ones = _mm256_set1_epi16(-1);
    let bias = _mm256_set1_epi16(i16::MIN);
    let mut masks = [zero; 4];
    for (lane, mask) in masks.iter_mut().enumerate() {
        let code = unsafe { _mm256_loadu_si256(codes.add(lane * 16).cast()) };
        for &point in points {
            *mask = _mm256_or_si256(*mask, _mm256_cmpeq_epi16(code, point));
        }
        for &(begin, span_biased) in ranges {
            let delta = _mm256_sub_epi16(code, begin);
            let outside = _mm256_cmpgt_epi16(_mm256_xor_si256(delta, bias), span_biased);
            *mask = _mm256_or_si256(*mask, _mm256_andnot_si256(outside, ones));
        }
    }
    let any = _mm256_or_si256(
        _mm256_or_si256(masks[0], masks[1]),
        _mm256_or_si256(masks[2], masks[3]),
    );
    if _mm256_testz_si256(any, any) != 0 {
        return 0;
    }
    compact_masks(masks[0], masks[1], masks[2], masks[3])
}

#[inline(always)]
fn consume_retained<O: Offset, const BLOCKS: usize>(
    base: usize,
    masks: &[u64; BLOCKS],
    sink: &mut RowSink<'_, O>,
) {
    for (block, &mask) in masks.iter().enumerate() {
        if mask != 0 {
            sink.mark_mask(base + block * BLOCK_CODES, LaneMask::from_bits(mask));
        }
    }
}

/// Const-shape AVX2 leaf for one or eight retained 64-code blocks at a time.
#[target_feature(enable = "avx2")]
#[inline(never)]
pub(in crate::search::prefilter::scan) unsafe fn scan_avx2_fixed<
    O: Offset,
    const POINTS: usize,
    const RANGES: usize,
    const BLOCKS: usize,
>(
    codes: &[Token],
    row_offsets: &[O],
    cover: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    debug_assert_eq!((cover.points.len(), cover.ranges.len()), (POINTS, RANGES));
    debug_assert!(matches!(BLOCKS, 1 | 8));

    let points: [__m256i; POINTS] =
        std::array::from_fn(|i| _mm256_set1_epi16(cover.points[i] as i16));
    let ranges: [(__m256i, __m256i); RANGES] = std::array::from_fn(|i| {
        let TokenRange { begin, last } = cover.ranges[i];
        let span = last.wrapping_sub(begin);
        (
            _mm256_set1_epi16(begin as i16),
            _mm256_set1_epi16((span ^ 0x8000) as i16),
        )
    });

    let group_codes = BLOCKS * BLOCK_CODES;
    let full_groups = codes.len() / group_codes * group_codes;
    let mut sink = RowSink::new(row_offsets, out, sparse_row_mapping);
    let mut base = 0;
    while base < full_groups {
        let mut retained = [0u64; BLOCKS];
        let mut any = 0;
        for (block, mask) in retained.iter_mut().enumerate() {
            *mask = unsafe {
                fixed_mask64::<POINTS, RANGES>(
                    codes.as_ptr().add(base + block * BLOCK_CODES),
                    &points,
                    &ranges,
                )
            };
            any |= *mask;
        }
        if any != 0 {
            consume_retained(base, &retained, &mut sink);
        }
        base += group_codes;
    }
    while base + BLOCK_CODES <= codes.len() {
        let mask =
            unsafe { fixed_mask64::<POINTS, RANGES>(codes.as_ptr().add(base), &points, &ranges) };
        if mask != 0 {
            sink.mark_mask(base, LaneMask::from_bits(mask));
        }
        base += BLOCK_CODES;
    }
    scan_tail(codes, cover, base, &mut sink);
}

/// Shared four-vector walk retained by the arbitrary-shape fallback.
#[target_feature(enable = "avx2")]
#[inline]
fn scan_four_vectors<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    cover: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
    mut matching_masks: impl FnMut(__m256i, __m256i, __m256i, __m256i) -> [__m256i; 4],
) {
    let total = codes.len();
    let ptr = codes.as_ptr();
    let mut sink = RowSink::new(row_offsets, out, sparse_row_mapping);
    let mut base = 0;
    while base + BLOCK_CODES <= total {
        let [m0, m1, m2, m3] = unsafe {
            matching_masks(
                _mm256_loadu_si256(ptr.add(base).cast()),
                _mm256_loadu_si256(ptr.add(base + 16).cast()),
                _mm256_loadu_si256(ptr.add(base + 32).cast()),
                _mm256_loadu_si256(ptr.add(base + 48).cast()),
            )
        };
        let any = _mm256_or_si256(_mm256_or_si256(m0, m1), _mm256_or_si256(m2, m3));
        if _mm256_movemask_epi8(any) != 0 {
            sink.mark_mask(base, LaneMask::from_bits(compact_masks(m0, m1, m2, m3)));
        }
        base += BLOCK_CODES;
    }
    scan_tail(codes, cover, base, &mut sink);
}

/// AVX2 fallback for arbitrary cover shapes.
#[target_feature(enable = "avx2")]
pub(in crate::search::prefilter::scan) fn scan_avx2_generic<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    cover: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    let zero = _mm256_setzero_si256();
    let ones = _mm256_set1_epi16(-1);
    let bias = _mm256_set1_epi16(i16::MIN);
    let points: Vec<__m256i> = cover
        .points
        .iter()
        .map(|&point| _mm256_set1_epi16(point as i16))
        .collect();
    let ranges: Vec<(__m256i, __m256i)> = cover
        .ranges
        .iter()
        .map(|range| {
            (
                _mm256_xor_si256(_mm256_set1_epi16(range.begin as i16), bias),
                _mm256_xor_si256(_mm256_set1_epi16(range.last as i16), bias),
            )
        })
        .collect();

    scan_four_vectors(
        codes,
        row_offsets,
        cover,
        sparse_row_mapping,
        out,
        |v0, v1, v2, v3| {
            let values = [v0, v1, v2, v3];
            let biased = values.map(|value| _mm256_xor_si256(value, bias));
            let mut masks = [zero; 4];
            for &point in &points {
                for (mask, &value) in masks.iter_mut().zip(&values) {
                    *mask = _mm256_or_si256(*mask, _mm256_cmpeq_epi16(value, point));
                }
            }
            for &(lo, hi) in &ranges {
                for (mask, &value) in masks.iter_mut().zip(&biased) {
                    let outside = _mm256_or_si256(
                        _mm256_cmpgt_epi16(lo, value),
                        _mm256_cmpgt_epi16(value, hi),
                    );
                    *mask = _mm256_or_si256(*mask, _mm256_andnot_si256(outside, ones));
                }
            }
            masks
        },
    );
}

/// Direct generic-kernel entry retained for architecture correctness tests.
#[cfg(test)]
#[target_feature(enable = "avx2")]
pub(in crate::search::prefilter) fn scan_avx2<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    cover: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    scan_avx2_generic(codes, row_offsets, cover, sparse_row_mapping, out);
}
