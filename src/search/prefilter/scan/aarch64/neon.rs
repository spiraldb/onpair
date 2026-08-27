// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! AArch64 NEON prefilter kernels.

use super::super::sink::{RowSink, mark_block};
use super::super::template::{DYN, Isa, scan_dynamic, scan_fixed, walk};
#[cfg(test)]
use super::super::{CoverShape, ScanInput, policy};
#[cfg(test)]
use super::execute as execute_neon;
use crate::core::offset::Offset;
use crate::core::types::{Token, TokenRange};
use crate::search::prefilter::cover::ProbeCover;

use core::arch::aarch64::{
    uint16x8_t, vceqq_u16, vcleq_u16, vdupq_n_u16, vld1q_u16, vmaxvq_u16, vorrq_u16, vst1q_u16,
    vsubq_u16,
};

type NeonRange = (uint16x8_t, uint16x8_t);
type NeonPairHits = (uint16x8_t, uint16x8_t);

#[inline(always)]
fn neon_point(token: Token) -> uint16x8_t {
    unsafe { vdupq_n_u16(token) }
}

#[inline(always)]
fn neon_range(range: TokenRange) -> NeonRange {
    unsafe {
        (
            vdupq_n_u16(range.begin),
            vdupq_n_u16(range.last.wrapping_sub(range.begin)),
        )
    }
}

#[inline(always)]
fn pair_block<const POINTS: usize, const RANGES: usize>(
    codes: *const Token,
    points: &[uint16x8_t],
    ranges: &[NeonRange],
) -> NeonPairHits {
    let points = if POINTS == DYN {
        points
    } else {
        debug_assert_eq!(points.len(), POINTS);
        unsafe { points.get_unchecked(..POINTS) }
    };
    let ranges = if RANGES == DYN {
        ranges
    } else {
        debug_assert_eq!(ranges.len(), RANGES);
        unsafe { ranges.get_unchecked(..RANGES) }
    };
    unsafe {
        let value0 = vld1q_u16(codes);
        let value1 = vld1q_u16(codes.add(8));
        let zero = vdupq_n_u16(0);
        let mut hits0 = zero;
        let mut hits1 = zero;
        for &point in points {
            hits0 = vorrq_u16(hits0, vceqq_u16(value0, point));
            hits1 = vorrq_u16(hits1, vceqq_u16(value1, point));
        }
        for &(begin, span) in ranges {
            hits0 = vorrq_u16(hits0, vcleq_u16(vsubq_u16(value0, begin), span));
            hits1 = vorrq_u16(hits1, vcleq_u16(vsubq_u16(value1, begin), span));
        }
        (hits0, hits1)
    }
}

#[inline(always)]
fn single_block<const POINTS: usize, const RANGES: usize>(
    codes: *const Token,
    points: &[uint16x8_t],
    ranges: &[NeonRange],
) -> uint16x8_t {
    let points = if POINTS == DYN {
        points
    } else {
        debug_assert_eq!(points.len(), POINTS);
        unsafe { points.get_unchecked(..POINTS) }
    };
    let ranges = if RANGES == DYN {
        ranges
    } else {
        debug_assert_eq!(ranges.len(), RANGES);
        unsafe { ranges.get_unchecked(..RANGES) }
    };
    unsafe {
        let value = vld1q_u16(codes);
        let mut hits = vdupq_n_u16(0);
        for &point in points {
            hits = vorrq_u16(hits, vceqq_u16(value, point));
        }
        for &(begin, span) in ranges {
            hits = vorrq_u16(hits, vcleq_u16(vsubq_u16(value, begin), span));
        }
        hits
    }
}

#[inline(always)]
fn pair_any((hits0, hits1): NeonPairHits) -> bool {
    unsafe { vmaxvq_u16(vorrq_u16(hits0, hits1)) != 0 }
}

#[inline(never)]
fn pair_emit<O: Offset>(base: usize, (hits0, hits1): NeonPairHits, sink: &mut RowSink<'_, O>) {
    let mut lanes0 = [0u16; 8];
    let mut lanes1 = [0u16; 8];
    unsafe {
        vst1q_u16(lanes0.as_mut_ptr(), hits0);
        vst1q_u16(lanes1.as_mut_ptr(), hits1);
    }
    mark_block(base, &lanes0, sink);
    mark_block(base + 8, &lanes1, sink);
}

#[inline(never)]
fn single_emit<O: Offset>(base: usize, hits: uint16x8_t, sink: &mut RowSink<'_, O>) {
    let mut lanes = [0u16; 8];
    unsafe { vst1q_u16(lanes.as_mut_ptr(), hits) };
    mark_block(base, &lanes, sink);
}

struct NeonPair;

impl Isa for NeonPair {
    const BLOCK: usize = 16;
    const WALK_REMAINDER: bool = false;

    type Point = uint16x8_t;
    type Range = NeonRange;
    type Hits = NeonPairHits;
    const NO_HITS: Self::Hits = unsafe { core::mem::zeroed() };

    #[inline(always)]
    fn point(token: Token) -> Self::Point {
        neon_point(token)
    }

    #[inline(always)]
    fn range(range: TokenRange) -> Self::Range {
        neon_range(range)
    }

    #[inline(always)]
    unsafe fn block<const POINTS: usize, const RANGES: usize>(
        codes: *const Token,
        points: &[Self::Point],
        ranges: &[Self::Range],
    ) -> Self::Hits {
        pair_block::<POINTS, RANGES>(codes, points, ranges)
    }

    #[inline(always)]
    fn any(hits: Self::Hits) -> bool {
        pair_any(hits)
    }

    #[inline(always)]
    fn emit<O: Offset>(base: usize, hits: Self::Hits, sink: &mut RowSink<'_, O>) {
        pair_emit(base, hits, sink);
    }
}

struct NeonSingle;

impl Isa for NeonSingle {
    const BLOCK: usize = 8;
    const WALK_REMAINDER: bool = false;

    type Point = uint16x8_t;
    type Range = NeonRange;
    type Hits = uint16x8_t;
    const NO_HITS: Self::Hits = unsafe { core::mem::zeroed() };

    #[inline(always)]
    fn point(token: Token) -> Self::Point {
        neon_point(token)
    }

    #[inline(always)]
    fn range(range: TokenRange) -> Self::Range {
        neon_range(range)
    }

    #[inline(always)]
    unsafe fn block<const POINTS: usize, const RANGES: usize>(
        codes: *const Token,
        points: &[Self::Point],
        ranges: &[Self::Range],
    ) -> Self::Hits {
        single_block::<POINTS, RANGES>(codes, points, ranges)
    }

    #[inline(always)]
    fn any(hits: Self::Hits) -> bool {
        unsafe { vmaxvq_u16(hits) != 0 }
    }

    #[inline(always)]
    fn emit<O: Offset>(base: usize, hits: Self::Hits, sink: &mut RowSink<'_, O>) {
        single_emit(base, hits, sink);
    }
}

struct NeonOnePointTwoRanges;

impl Isa for NeonOnePointTwoRanges {
    const BLOCK: usize = 16;
    const WALK_REMAINDER: bool = false;

    type Point = uint16x8_t;
    type Range = NeonRange;
    type Hits = NeonPairHits;
    const NO_HITS: Self::Hits = unsafe { core::mem::zeroed() };

    #[inline(always)]
    fn point(token: Token) -> Self::Point {
        neon_point(token)
    }

    #[inline(always)]
    fn range(range: TokenRange) -> Self::Range {
        neon_range(range)
    }

    #[inline(always)]
    unsafe fn block<const POINTS: usize, const RANGES: usize>(
        codes: *const Token,
        points: &[Self::Point],
        ranges: &[Self::Range],
    ) -> Self::Hits {
        debug_assert_eq!((POINTS, RANGES), (1, 2));
        let point = unsafe { *points.get_unchecked(0) };
        let (begin0, span0) = unsafe { *ranges.get_unchecked(0) };
        let (begin1, span1) = unsafe { *ranges.get_unchecked(1) };
        unsafe {
            let value0 = vld1q_u16(codes);
            let value1 = vld1q_u16(codes.add(8));
            let range00 = vcleq_u16(vsubq_u16(value0, begin0), span0);
            let range01 = vcleq_u16(vsubq_u16(value0, begin1), span1);
            let range10 = vcleq_u16(vsubq_u16(value1, begin0), span0);
            let range11 = vcleq_u16(vsubq_u16(value1, begin1), span1);
            (
                vorrq_u16(vceqq_u16(value0, point), vorrq_u16(range00, range01)),
                vorrq_u16(vceqq_u16(value1, point), vorrq_u16(range10, range11)),
            )
        }
    }

    #[inline(always)]
    fn any(hits: Self::Hits) -> bool {
        pair_any(hits)
    }

    #[inline(always)]
    fn emit<O: Offset>(base: usize, hits: Self::Hits, sink: &mut RowSink<'_, O>) {
        pair_emit(base, hits, sink);
    }
}

/// Point-only covers with 1–8 probes use a two-vector schedule. Exact one-,
/// two-, and three-point shapes are unrolled; 4–8 share the compact dynamic
/// instantiation.
#[inline(never)]
fn scan_neon_few_points<O: Offset, const N: usize>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    let probe_count = if N == 0 {
        pf.points.len()
    } else {
        debug_assert_eq!(pf.points.len(), N);
        N
    };
    let zero = neon_point(0);
    let mut probes = [zero; 8];
    for index in 0..probe_count {
        probes[index] = neon_point(unsafe { *pf.points.get_unchecked(index) });
    }
    let ranges: [NeonRange; 0] = [];
    unsafe {
        if N == 0 {
            walk::<NeonPair, O, DYN, 0, 1>(
                codes,
                row_offsets,
                pf,
                &probes[..probe_count],
                &ranges,
                sparse_row_mapping,
                out,
            )
        } else {
            walk::<NeonPair, O, N, 0, 1>(
                codes,
                row_offsets,
                pf,
                &probes[..probe_count],
                &ranges,
                sparse_row_mapping,
                out,
            )
        }
    }
}

/// Point-only covers at the default policy's upper boundary (9–16 checks).
#[inline(never)]
fn scan_neon_many_points<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    debug_assert!((9..=16).contains(&pf.points.len()));
    let zero = neon_point(0);
    let mut probes = [zero; 16];
    for (probe, &point) in probes.iter_mut().zip(&pf.points) {
        *probe = neon_point(point);
    }
    let ranges: [NeonRange; 0] = [];
    unsafe {
        walk::<NeonPair, O, DYN, 0, 1>(
            codes,
            row_offsets,
            pf,
            &probes[..pf.points.len()],
            &ranges,
            sparse_row_mapping,
            out,
        )
    }
}

#[inline(never)]
pub(in crate::search::prefilter::scan) fn scan_neon_one_range<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    unsafe { scan_fixed::<NeonPair, O, 0, 1, 1>(codes, row_offsets, pf, sparse_row_mapping, out) }
}

#[inline(never)]
pub(in crate::search::prefilter::scan) fn scan_neon_fixed_mixed<
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
    unsafe {
        scan_fixed::<NeonPair, O, POINTS, RANGES, 1>(
            codes,
            row_offsets,
            pf,
            sparse_row_mapping,
            out,
        )
    }
}

/// Preserve the explicit reduction tree used by this uncommon mixed shape.
#[inline(never)]
pub(in crate::search::prefilter::scan) fn scan_neon_one_point_two_ranges<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    unsafe {
        scan_fixed::<NeonOnePointTwoRanges, O, 1, 2, 1>(
            codes,
            row_offsets,
            pf,
            sparse_row_mapping,
            out,
        )
    }
}

/// Compact fallback for small mixed covers. Broadcasts remain in fixed-size
/// stack arrays while the compare loops retain runtime probe counts.
#[inline(never)]
pub(in crate::search::prefilter::scan) fn scan_neon_few_mixed<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    debug_assert!(!pf.ranges.is_empty());
    debug_assert!(pf.points.len() + 2 * pf.ranges.len() <= 16);
    let zero = neon_point(0);
    let mut points = [zero; 16];
    for (point, &token) in points.iter_mut().zip(&pf.points) {
        *point = neon_point(token);
    }
    let mut ranges = [(zero, zero); 8];
    for (range, &source) in ranges.iter_mut().zip(&pf.ranges) {
        *range = neon_range(source);
    }
    unsafe {
        walk::<NeonPair, O, DYN, DYN, 1>(
            codes,
            row_offsets,
            pf,
            &points[..pf.points.len()],
            &ranges[..pf.ranges.len()],
            sparse_row_mapping,
            out,
        )
    }
}

#[inline]
pub(in crate::search::prefilter::scan) fn scan_neon_points<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    match pf.points.len() {
        1 => scan_neon_few_points::<O, 1>(codes, row_offsets, pf, sparse_row_mapping, out),
        2 => scan_neon_few_points::<O, 2>(codes, row_offsets, pf, sparse_row_mapping, out),
        3 => scan_neon_few_points::<O, 3>(codes, row_offsets, pf, sparse_row_mapping, out),
        4..=8 => scan_neon_few_points::<O, 0>(codes, row_offsets, pf, sparse_row_mapping, out),
        _ => unreachable!("invalid NEON point-family shape"),
    }
}

#[inline]
pub(in crate::search::prefilter::scan) fn scan_neon_points_many<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    scan_neon_many_points(codes, row_offsets, pf, sparse_row_mapping, out);
}

/// Arbitrary covers retain the existing one- or two-vector policy while both
/// schedules share the template walk and prepared probe vectors.
pub(in crate::search::prefilter::scan) fn scan_neon_generic<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    use_two_vectors: bool,
    out: &mut Vec<usize>,
) {
    unsafe {
        if use_two_vectors {
            scan_dynamic::<NeonPair, O>(codes, row_offsets, pf, sparse_row_mapping, out)
        } else {
            scan_dynamic::<NeonSingle, O>(codes, row_offsets, pf, sparse_row_mapping, out)
        }
    }
}

/// Direct kernel entry retained for architecture correctness tests.
#[cfg(test)]
pub(in crate::search::prefilter) fn scan_neon<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    let shape = CoverShape {
        points: pf.points.len(),
        ranges: pf.ranges.len(),
    };
    execute_neon(
        policy::select_neon(shape),
        ScanInput::full(codes, row_offsets, pf),
        sparse_row_mapping,
        out,
    );
}
