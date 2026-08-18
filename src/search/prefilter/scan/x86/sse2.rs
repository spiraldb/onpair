// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Baseline x86-64 SSE2 prefilter kernels.

use super::super::sink::{LaneMask, RowSink, scan_tail};
#[cfg(test)]
use super::super::table;
use crate::core::offset::Offset;
use crate::core::types::Token;
use crate::search::prefilter::cover::ProbeCover;

/// Shared four-vector SSE2 walk. Comparison masks are packed to one bit per
/// code only on hit blocks, keeping the common no-hit path compact.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn scan_sse2_four_vectors<O: Offset>(
    codes: &[Token],
    sink: &mut RowSink<'_, O>,
    always_pack: bool,
    mut matching_masks: impl FnMut(
        core::arch::x86_64::__m128i,
        core::arch::x86_64::__m128i,
        core::arch::x86_64::__m128i,
        core::arch::x86_64::__m128i,
    ) -> (
        core::arch::x86_64::__m128i,
        core::arch::x86_64::__m128i,
        core::arch::x86_64::__m128i,
        core::arch::x86_64::__m128i,
    ),
) -> usize {
    use core::arch::x86_64::{
        __m128i, _mm_loadu_si128, _mm_movemask_epi8, _mm_or_si128, _mm_packs_epi16,
    };

    let total = codes.len();
    let base = codes.as_ptr();
    let mut i = 0usize;
    while i + 32 <= total {
        // SAFETY: `i + 32 <= total`; SSE2 is an x86-64 baseline feature.
        let (v0, v1, v2, v3) = unsafe {
            (
                _mm_loadu_si128(base.add(i).cast::<__m128i>()),
                _mm_loadu_si128(base.add(i + 8).cast::<__m128i>()),
                _mm_loadu_si128(base.add(i + 16).cast::<__m128i>()),
                _mm_loadu_si128(base.add(i + 24).cast::<__m128i>()),
            )
        };
        let (m0, m1, m2, m3) = matching_masks(v0, v1, v2, v3);
        // SAFETY: all operations require only baseline SSE2.
        unsafe {
            let any = _mm_or_si128(_mm_or_si128(m0, m1), _mm_or_si128(m2, m3));
            if always_pack || _mm_movemask_epi8(any) != 0 {
                let lanes01 = _mm_movemask_epi8(_mm_packs_epi16(m0, m1)) as u16;
                let lanes23 = _mm_movemask_epi8(_mm_packs_epi16(m2, m3)) as u16;
                let lanes = lanes01 as u64 | ((lanes23 as u64) << 16);
                if lanes != 0 {
                    sink.mark_mask(i, LaneMask::from_bits(lanes));
                }
            }
        }
        i += 32;
    }
    i
}

#[cfg(target_arch = "x86_64")]
pub(in crate::search::prefilter::scan) fn scan_sse2_one_point<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    use core::arch::x86_64::{_mm_cmpeq_epi16, _mm_set1_epi16};

    debug_assert_eq!(pf.points.len(), 1);
    debug_assert!(pf.ranges.is_empty());
    // SAFETY: SSE2 is an x86-64 baseline feature.
    let point = unsafe { _mm_set1_epi16(pf.points[0] as i16) };
    let mut sink = RowSink::new(row_offsets, out, sparse_row_mapping);
    let i = scan_sse2_four_vectors(codes, &mut sink, false, |v0, v1, v2, v3| unsafe {
        (
            _mm_cmpeq_epi16(v0, point),
            _mm_cmpeq_epi16(v1, point),
            _mm_cmpeq_epi16(v2, point),
            _mm_cmpeq_epi16(v3, point),
        )
    });
    scan_tail(codes, pf, i, &mut sink);
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn sse2_fixed_mask<const POINTS: usize, const RANGES: usize>(
    v: core::arch::x86_64::__m128i,
    point_probes: &[core::arch::x86_64::__m128i; POINTS],
    range_los: &[core::arch::x86_64::__m128i; RANGES],
    range_spans: &[core::arch::x86_64::__m128i; RANGES],
    zero: core::arch::x86_64::__m128i,
) -> core::arch::x86_64::__m128i {
    use core::arch::x86_64::{_mm_cmpeq_epi16, _mm_or_si128, _mm_sub_epi16, _mm_subs_epu16};

    // SAFETY: all operations require only baseline SSE2.
    unsafe {
        let mut acc = zero;
        for &point in point_probes {
            acc = _mm_or_si128(acc, _mm_cmpeq_epi16(v, point));
        }
        for range in 0..RANGES {
            let delta = _mm_sub_epi16(v, range_los[range]);
            let excess = _mm_subs_epu16(delta, range_spans[range]);
            acc = _mm_or_si128(acc, _mm_cmpeq_epi16(excess, zero));
        }
        acc
    }
}

#[cfg(target_arch = "x86_64")]
pub(in crate::search::prefilter::scan) fn scan_sse2_fixed<
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
    use core::arch::x86_64::{_mm_set1_epi16, _mm_setzero_si128};

    debug_assert_eq!(pf.points.len(), POINTS);
    debug_assert_eq!(pf.ranges.len(), RANGES);
    // SAFETY: SSE2 is an x86-64 baseline feature.
    let zero = unsafe { _mm_setzero_si128() };
    let mut point_probes = [zero; POINTS];
    let mut range_los = [zero; RANGES];
    let mut range_spans = [zero; RANGES];
    for (probe, &point) in point_probes.iter_mut().zip(&pf.points) {
        *probe = unsafe { _mm_set1_epi16(point as i16) };
    }
    for ((lo, span), range) in range_los.iter_mut().zip(&mut range_spans).zip(&pf.ranges) {
        *lo = unsafe { _mm_set1_epi16(range.begin as i16) };
        *span = unsafe { _mm_set1_epi16((range.last - range.begin) as i16) };
    }

    let mut sink = RowSink::new(row_offsets, out, sparse_row_mapping);
    let i = scan_sse2_four_vectors(
        codes,
        &mut sink,
        POINTS == 0 && RANGES == 1,
        |v0, v1, v2, v3| {
            (
                sse2_fixed_mask(v0, &point_probes, &range_los, &range_spans, zero),
                sse2_fixed_mask(v1, &point_probes, &range_los, &range_spans, zero),
                sse2_fixed_mask(v2, &point_probes, &range_los, &range_spans, zero),
                sse2_fixed_mask(v3, &point_probes, &range_los, &range_spans, zero),
            )
        },
    );
    scan_tail(codes, pf, i, &mut sink);
}

#[cfg(all(target_arch = "x86_64", test))]
pub(in crate::search::prefilter) fn scan_sse2<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    match (pf.points.len(), pf.ranges.len()) {
        (1, 0) => scan_sse2_one_point(codes, row_offsets, pf, sparse_row_mapping, out),
        (0, 1) => scan_sse2_fixed::<O, 0, 1>(codes, row_offsets, pf, sparse_row_mapping, out),
        (2, 0) => scan_sse2_fixed::<O, 2, 0>(codes, row_offsets, pf, sparse_row_mapping, out),
        (3, 0) => scan_sse2_fixed::<O, 3, 0>(codes, row_offsets, pf, sparse_row_mapping, out),
        (1, 1) => scan_sse2_fixed::<O, 1, 1>(codes, row_offsets, pf, sparse_row_mapping, out),
        (4, 0) => scan_sse2_fixed::<O, 4, 0>(codes, row_offsets, pf, sparse_row_mapping, out),
        (2, 1) => scan_sse2_fixed::<O, 2, 1>(codes, row_offsets, pf, sparse_row_mapping, out),
        (5, 0) => scan_sse2_fixed::<O, 5, 0>(codes, row_offsets, pf, sparse_row_mapping, out),
        (0, 2) => scan_sse2_fixed::<O, 0, 2>(codes, row_offsets, pf, sparse_row_mapping, out),
        (1, 2) => scan_sse2_fixed::<O, 1, 2>(codes, row_offsets, pf, sparse_row_mapping, out),
        (3, 1) => scan_sse2_fixed::<O, 3, 1>(codes, row_offsets, pf, sparse_row_mapping, out),
        (6, 0) => scan_sse2_fixed::<O, 6, 0>(codes, row_offsets, pf, sparse_row_mapping, out),
        (4, 1) => scan_sse2_fixed::<O, 4, 1>(codes, row_offsets, pf, sparse_row_mapping, out),
        (2, 2) => scan_sse2_fixed::<O, 2, 2>(codes, row_offsets, pf, sparse_row_mapping, out),
        (0, 3) => scan_sse2_fixed::<O, 0, 3>(codes, row_offsets, pf, sparse_row_mapping, out),
        (3, 2) => scan_sse2_fixed::<O, 3, 2>(codes, row_offsets, pf, sparse_row_mapping, out),
        (1, 3) => scan_sse2_fixed::<O, 1, 3>(codes, row_offsets, pf, sparse_row_mapping, out),
        (4, 2) => scan_sse2_fixed::<O, 4, 2>(codes, row_offsets, pf, sparse_row_mapping, out),
        (points, ranges) if points + 2 * ranges >= 17 => {
            table::scan_codes(codes, row_offsets, pf, sparse_row_mapping, out)
        }
        _ => scan_sse2_generic(codes, row_offsets, pf, sparse_row_mapping, out),
    }
}

/// SSE2 fallback for arbitrary cover shapes, processing eight lanes at a time.
#[cfg(target_arch = "x86_64")]
pub(in crate::search::prefilter::scan) fn scan_sse2_generic<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    use core::arch::x86_64::{
        __m128i, _mm_cmpeq_epi16, _mm_or_si128, _mm_set1_epi16, _mm_setzero_si128, _mm_sub_epi16,
        _mm_subs_epu16,
    };

    // Hoist every broadcast out of the code-stream loop. The dynamic fallback
    // handles arbitrary cover widths; common small shapes use fixed kernels.
    let zero = unsafe { _mm_setzero_si128() };
    let point_probes: Vec<__m128i> = pf
        .points
        .iter()
        .map(|&point| unsafe { _mm_set1_epi16(point as i16) })
        .collect();
    let range_probes: Vec<(__m128i, __m128i)> = pf
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
    let i = scan_sse2_four_vectors(codes, &mut sink, false, |v0, v1, v2, v3| unsafe {
        let mut m0 = zero;
        let mut m1 = zero;
        let mut m2 = zero;
        let mut m3 = zero;
        for &point in &point_probes {
            m0 = _mm_or_si128(m0, _mm_cmpeq_epi16(v0, point));
            m1 = _mm_or_si128(m1, _mm_cmpeq_epi16(v1, point));
            m2 = _mm_or_si128(m2, _mm_cmpeq_epi16(v2, point));
            m3 = _mm_or_si128(m3, _mm_cmpeq_epi16(v3, point));
        }
        for &(lo, span) in &range_probes {
            let e0 = _mm_subs_epu16(_mm_sub_epi16(v0, lo), span);
            let e1 = _mm_subs_epu16(_mm_sub_epi16(v1, lo), span);
            let e2 = _mm_subs_epu16(_mm_sub_epi16(v2, lo), span);
            let e3 = _mm_subs_epu16(_mm_sub_epi16(v3, lo), span);
            m0 = _mm_or_si128(m0, _mm_cmpeq_epi16(e0, zero));
            m1 = _mm_or_si128(m1, _mm_cmpeq_epi16(e1, zero));
            m2 = _mm_or_si128(m2, _mm_cmpeq_epi16(e2, zero));
            m3 = _mm_or_si128(m3, _mm_cmpeq_epi16(e3, zero));
        }
        (m0, m1, m2, m3)
    });
    scan_tail(codes, pf, i, &mut sink);
}
