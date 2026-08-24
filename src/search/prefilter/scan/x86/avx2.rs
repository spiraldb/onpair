// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! AVX2 prefilter kernels.

use super::super::sink::{LaneMask, RowSink, mark_block, scan_tail};
use crate::core::offset::Offset;
use crate::core::types::{Token, TokenRange};
use crate::search::prefilter::cover::ProbeCover;

/// Compact four AVX2 all-ones/all-zeroes `u16` masks into one lane bitset.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn compact_avx2_masks(
    m0: core::arch::x86_64::__m256i,
    m1: core::arch::x86_64::__m256i,
    m2: core::arch::x86_64::__m256i,
    m3: core::arch::x86_64::__m256i,
) -> u64 {
    use core::arch::x86_64::{_mm256_movemask_epi8, _mm256_packs_epi16, _mm256_permute4x64_epi64};

    let lanes01 = _mm256_movemask_epi8(_mm256_permute4x64_epi64::<0xd8>(_mm256_packs_epi16(m0, m1)))
        as u32 as u64;
    let lanes23 = _mm256_movemask_epi8(_mm256_permute4x64_epi64::<0xd8>(_mm256_packs_epi16(m2, m3)))
        as u32 as u64;
    lanes01 | (lanes23 << 32)
}

/// Compact two AVX2 all-ones/all-zeroes `u16` masks into 32 lane bits.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
fn compact_two_avx2_masks(m0: core::arch::x86_64::__m256i, m1: core::arch::x86_64::__m256i) -> u32 {
    use core::arch::x86_64::{_mm256_movemask_epi8, _mm256_packs_epi16, _mm256_permute4x64_epi64};

    _mm256_movemask_epi8(_mm256_permute4x64_epi64::<0xd8>(_mm256_packs_epi16(m0, m1))) as u32
}

/// Walk four AVX2 vectors per iteration and materialize only blocks with hits.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(
    clippy::too_many_arguments,
    reason = "the hot SIMD loop keeps scan state explicit so LLVM can optimize it away"
)]
fn scan_avx2_four_vectors<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    compact_hits: bool,
    use_vptest: bool,
    out: &mut Vec<usize>,
    mut matching_masks: impl FnMut(
        core::arch::x86_64::__m256i,
        core::arch::x86_64::__m256i,
        core::arch::x86_64::__m256i,
        core::arch::x86_64::__m256i,
    ) -> (
        core::arch::x86_64::__m256i,
        core::arch::x86_64::__m256i,
        core::arch::x86_64::__m256i,
        core::arch::x86_64::__m256i,
    ),
) {
    use core::arch::x86_64::{
        __m256i, _mm256_loadu_si256, _mm256_movemask_epi8, _mm256_or_si256, _mm256_storeu_si256,
        _mm256_testz_si256,
    };

    let total = codes.len();
    let base = codes.as_ptr();
    let mut sink = RowSink::new(row_offsets, out, sparse_row_mapping);
    let mut i = 0usize;
    while i + 64 <= total {
        let (v0, v1, v2, v3) = unsafe {
            (
                _mm256_loadu_si256(base.add(i).cast::<__m256i>()),
                _mm256_loadu_si256(base.add(i + 16).cast::<__m256i>()),
                _mm256_loadu_si256(base.add(i + 32).cast::<__m256i>()),
                _mm256_loadu_si256(base.add(i + 48).cast::<__m256i>()),
            )
        };
        let (m0, m1, m2, m3) = matching_masks(v0, v1, v2, v3);
        let any = _mm256_or_si256(_mm256_or_si256(m0, m1), _mm256_or_si256(m2, m3));
        let has_hits = if use_vptest {
            _mm256_testz_si256(any, any) == 0
        } else {
            _mm256_movemask_epi8(any) != 0
        };
        if has_hits {
            if compact_hits {
                sink.mark_mask(i, LaneMask::from_bits(compact_avx2_masks(m0, m1, m2, m3)));
            } else {
                let mut hits = [[0u16; 16]; 4];
                unsafe {
                    _mm256_storeu_si256(hits[0].as_mut_ptr().cast::<__m256i>(), m0);
                    _mm256_storeu_si256(hits[1].as_mut_ptr().cast::<__m256i>(), m1);
                    _mm256_storeu_si256(hits[2].as_mut_ptr().cast::<__m256i>(), m2);
                    _mm256_storeu_si256(hits[3].as_mut_ptr().cast::<__m256i>(), m3);
                }
                mark_block(i, &hits[0], &mut sink);
                mark_block(i + 16, &hits[1], &mut sink);
                mark_block(i + 32, &hits[2], &mut sink);
                mark_block(i + 48, &hits[3], &mut sink);
            }
        }
        i += 64;
    }
    scan_tail(codes, pf, i, &mut sink);
}

/// Four-vector AVX2 kernel for the most compact possible cover.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline(never)]
pub(in crate::search::prefilter::scan) fn scan_avx2_one_point<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    compact_hits: bool,
    out: &mut Vec<usize>,
) {
    use core::arch::x86_64::{_mm256_cmpeq_epi16, _mm256_set1_epi16};

    debug_assert_eq!(pf.points.len(), 1);
    debug_assert!(pf.ranges.is_empty());
    let point = _mm256_set1_epi16(pf.points[0] as i16);
    scan_avx2_four_vectors(
        codes,
        row_offsets,
        pf,
        sparse_row_mapping,
        compact_hits,
        false,
        out,
        |v0, v1, v2, v3| {
            (
                _mm256_cmpeq_epi16(v0, point),
                _mm256_cmpeq_epi16(v1, point),
                _mm256_cmpeq_epi16(v2, point),
                _mm256_cmpeq_epi16(v3, point),
            )
        },
    );
}

/// Four-vector AVX2 kernel for one inclusive token range.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline(never)]
pub(in crate::search::prefilter::scan) fn scan_avx2_one_range<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    use core::arch::x86_64::{
        __m256i, _mm256_andnot_si256, _mm256_cmpgt_epi16, _mm256_or_si256, _mm256_set1_epi16,
        _mm256_xor_si256,
    };

    debug_assert!(pf.points.is_empty());
    debug_assert_eq!(pf.ranges.len(), 1);
    let range = pf.ranges[0];
    let bias = _mm256_set1_epi16(i16::MIN);
    let ones = _mm256_set1_epi16(-1);
    let lo = _mm256_xor_si256(_mm256_set1_epi16(range.begin as i16), bias);
    let hi = _mm256_xor_si256(_mm256_set1_epi16(range.last as i16), bias);
    let in_range = |v: __m256i| {
        let code = _mm256_xor_si256(v, bias);
        let outside = _mm256_or_si256(_mm256_cmpgt_epi16(lo, code), _mm256_cmpgt_epi16(code, hi));
        _mm256_andnot_si256(outside, ones)
    };
    scan_avx2_four_vectors(
        codes,
        row_offsets,
        pf,
        sparse_row_mapping,
        true,
        true,
        out,
        |v0, v1, v2, v3| (in_range(v0), in_range(v1), in_range(v2), in_range(v3)),
    );
}

/// Four-vector AVX2 kernel for a fixed small point/range cover.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline(never)]
pub(in crate::search::prefilter::scan) fn scan_avx2_fixed<
    O: Offset,
    const POINTS: usize,
    const RANGES: usize,
>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    use core::arch::x86_64::{
        _mm256_andnot_si256, _mm256_cmpeq_epi16, _mm256_cmpgt_epi16, _mm256_or_si256,
        _mm256_set1_epi16, _mm256_setzero_si256, _mm256_xor_si256,
    };

    debug_assert_eq!(pf.points.len(), POINTS);
    debug_assert_eq!(pf.ranges.len(), RANGES);
    let zero = _mm256_setzero_si256();
    let bias = _mm256_set1_epi16(i16::MIN);
    let ones = _mm256_set1_epi16(-1);
    let mut points = [zero; POINTS];
    for (dst, &point) in points.iter_mut().zip(&pf.points) {
        *dst = _mm256_set1_epi16(point as i16);
    }
    let mut ranges = [(zero, zero); RANGES];
    for (dst, &TokenRange { begin, last }) in ranges.iter_mut().zip(&pf.ranges) {
        *dst = (
            _mm256_xor_si256(_mm256_set1_epi16(begin as i16), bias),
            _mm256_xor_si256(_mm256_set1_epi16(last as i16), bias),
        );
    }

    scan_avx2_four_vectors(
        codes,
        row_offsets,
        pf,
        sparse_row_mapping,
        true,
        matches!((POINTS, RANGES), (3, 0) | (1, 1) | (2, 1) | (3, 1)),
        out,
        |v0, v1, v2, v3| {
            let mut acc0 = zero;
            let mut acc1 = zero;
            let mut acc2 = zero;
            let mut acc3 = zero;
            for &point in &points {
                acc0 = _mm256_or_si256(acc0, _mm256_cmpeq_epi16(v0, point));
                acc1 = _mm256_or_si256(acc1, _mm256_cmpeq_epi16(v1, point));
                acc2 = _mm256_or_si256(acc2, _mm256_cmpeq_epi16(v2, point));
                acc3 = _mm256_or_si256(acc3, _mm256_cmpeq_epi16(v3, point));
            }
            for &(lo, hi) in &ranges {
                let code0 = _mm256_xor_si256(v0, bias);
                let code1 = _mm256_xor_si256(v1, bias);
                let code2 = _mm256_xor_si256(v2, bias);
                let code3 = _mm256_xor_si256(v3, bias);
                let outside0 =
                    _mm256_or_si256(_mm256_cmpgt_epi16(lo, code0), _mm256_cmpgt_epi16(code0, hi));
                let outside1 =
                    _mm256_or_si256(_mm256_cmpgt_epi16(lo, code1), _mm256_cmpgt_epi16(code1, hi));
                let outside2 =
                    _mm256_or_si256(_mm256_cmpgt_epi16(lo, code2), _mm256_cmpgt_epi16(code2, hi));
                let outside3 =
                    _mm256_or_si256(_mm256_cmpgt_epi16(lo, code3), _mm256_cmpgt_epi16(code3, hi));
                acc0 = _mm256_or_si256(acc0, _mm256_andnot_si256(outside0, ones));
                acc1 = _mm256_or_si256(acc1, _mm256_andnot_si256(outside1, ones));
                acc2 = _mm256_or_si256(acc2, _mm256_andnot_si256(outside2, ones));
                acc3 = _mm256_or_si256(acc3, _mm256_andnot_si256(outside3, ones));
            }
            (acc0, acc1, acc2, acc3)
        },
    );
}

/// AVX2 fallback for arbitrary cover shapes.
///
/// Every point and range is broadcast once before the code-stream loop. Each
/// vector then pays exactly the comparison cost reported by the analysis: one
/// equality per point and two ordered comparisons per range.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(in crate::search::prefilter::scan) fn scan_avx2_generic<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    use core::arch::x86_64::{
        __m256i, _mm256_andnot_si256, _mm256_cmpeq_epi16, _mm256_cmpgt_epi16, _mm256_or_si256,
        _mm256_set1_epi16, _mm256_setzero_si256, _mm256_xor_si256,
    };

    let zero = _mm256_setzero_si256();
    let ones = _mm256_set1_epi16(-1);
    let bias = _mm256_set1_epi16(i16::MIN);
    let point_probes: Vec<__m256i> = pf
        .points
        .iter()
        .map(|&point| _mm256_set1_epi16(point as i16))
        .collect();
    let range_probes: Vec<(__m256i, __m256i)> = pf
        .ranges
        .iter()
        .map(|range| {
            (
                _mm256_xor_si256(_mm256_set1_epi16(range.begin as i16), bias),
                _mm256_xor_si256(_mm256_set1_epi16(range.last as i16), bias),
            )
        })
        .collect();

    scan_avx2_four_vectors(
        codes,
        row_offsets,
        pf,
        sparse_row_mapping,
        true,
        false,
        out,
        |v0, v1, v2, v3| {
            let cb0 = _mm256_xor_si256(v0, bias);
            let cb1 = _mm256_xor_si256(v1, bias);
            let cb2 = _mm256_xor_si256(v2, bias);
            let cb3 = _mm256_xor_si256(v3, bias);
            let mut m0 = zero;
            let mut m1 = zero;
            let mut m2 = zero;
            let mut m3 = zero;
            for &point in &point_probes {
                m0 = _mm256_or_si256(m0, _mm256_cmpeq_epi16(v0, point));
                m1 = _mm256_or_si256(m1, _mm256_cmpeq_epi16(v1, point));
                m2 = _mm256_or_si256(m2, _mm256_cmpeq_epi16(v2, point));
                m3 = _mm256_or_si256(m3, _mm256_cmpeq_epi16(v3, point));
            }
            for &(lo, hi) in &range_probes {
                let outside0 =
                    _mm256_or_si256(_mm256_cmpgt_epi16(lo, cb0), _mm256_cmpgt_epi16(cb0, hi));
                let outside1 =
                    _mm256_or_si256(_mm256_cmpgt_epi16(lo, cb1), _mm256_cmpgt_epi16(cb1, hi));
                let outside2 =
                    _mm256_or_si256(_mm256_cmpgt_epi16(lo, cb2), _mm256_cmpgt_epi16(cb2, hi));
                let outside3 =
                    _mm256_or_si256(_mm256_cmpgt_epi16(lo, cb3), _mm256_cmpgt_epi16(cb3, hi));
                m0 = _mm256_or_si256(m0, _mm256_andnot_si256(outside0, ones));
                m1 = _mm256_or_si256(m1, _mm256_andnot_si256(outside1, ones));
                m2 = _mm256_or_si256(m2, _mm256_andnot_si256(outside2, ones));
                m3 = _mm256_or_si256(m3, _mm256_andnot_si256(outside3, ones));
            }
            (m0, m1, m2, m3)
        },
    );
}

/// AVX2 fast path for covers costing at most sixteen comparisons.
///
/// Probe broadcasts are prepared once, then two code vectors are evaluated
/// together. The common no-hit path pays one combined movemask per 32 codes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(in crate::search::prefilter::scan) fn scan_avx2_few<O: Offset, const COMPACT_HITS: bool>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    use core::arch::x86_64::{
        __m256i, _mm256_andnot_si256, _mm256_cmpeq_epi16, _mm256_cmpgt_epi16, _mm256_loadu_si256,
        _mm256_movemask_epi8, _mm256_or_si256, _mm256_set1_epi16, _mm256_setzero_si256,
        _mm256_storeu_si256, _mm256_xor_si256,
    };

    debug_assert!(pf.points.len() + 2 * pf.ranges.len() <= 16);
    let total = codes.len();
    let base = codes.as_ptr();
    let mut sink = RowSink::new(row_offsets, out, sparse_row_mapping);
    let (zero, ones, bias) = (
        _mm256_setzero_si256(),
        _mm256_set1_epi16(-1),
        _mm256_set1_epi16(i16::MIN),
    );
    let mut points = [zero; 16];
    for (dst, &point) in points.iter_mut().zip(&pf.points) {
        *dst = _mm256_set1_epi16(point as i16);
    }
    let points = &points[..pf.points.len()];
    let mut ranges = [(zero, zero); 8];
    for (dst, &TokenRange { begin, last }) in ranges.iter_mut().zip(&pf.ranges) {
        *dst = (
            _mm256_xor_si256(_mm256_set1_epi16(begin as i16), bias),
            _mm256_xor_si256(_mm256_set1_epi16(last as i16), bias),
        );
    }
    let ranges = &ranges[..pf.ranges.len()];

    let mut i = 0usize;
    while i + 32 <= total {
        let (acc0, acc1) = unsafe {
            let v0 = _mm256_loadu_si256(base.add(i).cast::<__m256i>());
            let v1 = _mm256_loadu_si256(base.add(i + 16).cast::<__m256i>());
            let cb0 = _mm256_xor_si256(v0, bias);
            let cb1 = _mm256_xor_si256(v1, bias);
            let mut acc0 = zero;
            let mut acc1 = zero;
            for &point in points {
                acc0 = _mm256_or_si256(acc0, _mm256_cmpeq_epi16(v0, point));
                acc1 = _mm256_or_si256(acc1, _mm256_cmpeq_epi16(v1, point));
            }
            for &(lo, hi) in ranges {
                let below0 = _mm256_cmpgt_epi16(lo, cb0);
                let above0 = _mm256_cmpgt_epi16(cb0, hi);
                let below1 = _mm256_cmpgt_epi16(lo, cb1);
                let above1 = _mm256_cmpgt_epi16(cb1, hi);
                acc0 = _mm256_or_si256(
                    acc0,
                    _mm256_andnot_si256(_mm256_or_si256(below0, above0), ones),
                );
                acc1 = _mm256_or_si256(
                    acc1,
                    _mm256_andnot_si256(_mm256_or_si256(below1, above1), ones),
                );
            }
            (acc0, acc1)
        };
        let any = _mm256_movemask_epi8(_mm256_or_si256(acc0, acc1));
        if any != 0 {
            if COMPACT_HITS {
                sink.mark_mask(
                    i,
                    LaneMask::from_bits(u64::from(compact_two_avx2_masks(acc0, acc1))),
                );
            } else {
                let mut m0 = [0u16; 16];
                let mut m1 = [0u16; 16];
                unsafe {
                    _mm256_storeu_si256(m0.as_mut_ptr().cast::<__m256i>(), acc0);
                    _mm256_storeu_si256(m1.as_mut_ptr().cast::<__m256i>(), acc1);
                }
                mark_block(i, &m0, &mut sink);
                mark_block(i + 16, &m1, &mut sink);
            }
        }
        i += 32;
    }
    scan_tail(codes, pf, i, &mut sink);
}

/// Direct generic-kernel entry retained for architecture correctness tests.
#[cfg(all(target_arch = "x86_64", test))]
#[target_feature(enable = "avx2")]
pub(in crate::search::prefilter) fn scan_avx2<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    scan_avx2_generic(codes, row_offsets, pf, sparse_row_mapping, out);
}
