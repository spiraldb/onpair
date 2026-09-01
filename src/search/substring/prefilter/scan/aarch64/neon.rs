// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! AArch64 NEON prefilter kernels.

use super::super::policy::FixedShape;
use super::super::sink::{RowSink, mark_block};
use super::super::template::{DYN, Isa, scan_dynamic, scan_fixed as fixed, walk};
#[cfg(test)]
use super::super::{
    AnalysisFacts, CoverShape, RegionFacts, ScanFacts, ScanInput,
    policy::{self, TargetCaps},
};
#[cfg(test)]
use super::execute as execute_neon;
use crate::core::offset::Offset;
use crate::core::types::{Token, TokenRange};
use crate::search::substring::prefilter::cover::ProbeCover;

use core::arch::aarch64::{
    uint16x8_t, vceqq_u16, vcleq_u16, vdupq_n_u16, vld1q_u16, vmaxvq_u16, vorrq_u16, vst1q_u16,
    vsubq_u16,
};

type NeonRange = (uint16x8_t, uint16x8_t);

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
fn neon_block<const VECTORS: usize, const POINTS: usize, const RANGES: usize>(
    codes: *const Token,
    points: &[uint16x8_t],
    ranges: &[NeonRange],
) -> [uint16x8_t; VECTORS] {
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
        let values: [uint16x8_t; VECTORS] =
            std::array::from_fn(|vector| vld1q_u16(codes.add(vector * 8)));
        let zero = vdupq_n_u16(0);

        // Preserve the faster reduction tree for this mixed shape.
        // The condition is resolved when the const-generic leaf is compiled.
        if VECTORS == 2 && POINTS == 1 && RANGES == 2 {
            let point = *points.get_unchecked(0);
            let (begin0, span0) = *ranges.get_unchecked(0);
            let (begin1, span1) = *ranges.get_unchecked(1);
            let range00 = vcleq_u16(vsubq_u16(values[0], begin0), span0);
            let range01 = vcleq_u16(vsubq_u16(values[0], begin1), span1);
            let range10 = vcleq_u16(vsubq_u16(values[1], begin0), span0);
            let range11 = vcleq_u16(vsubq_u16(values[1], begin1), span1);
            let mut hits = [zero; VECTORS];
            *hits.get_unchecked_mut(0) =
                vorrq_u16(vceqq_u16(values[0], point), vorrq_u16(range00, range01));
            *hits.get_unchecked_mut(1) =
                vorrq_u16(vceqq_u16(values[1], point), vorrq_u16(range10, range11));
            return hits;
        }

        let mut hits = [zero; VECTORS];
        for &point in points {
            for (hits, &value) in hits.iter_mut().zip(&values) {
                *hits = vorrq_u16(*hits, vceqq_u16(value, point));
            }
        }
        for &(begin, span) in ranges {
            for (hits, &value) in hits.iter_mut().zip(&values) {
                *hits = vorrq_u16(*hits, vcleq_u16(vsubq_u16(value, begin), span));
            }
        }
        hits
    }
}

#[inline(always)]
fn neon_any<const VECTORS: usize>(hits: [uint16x8_t; VECTORS]) -> bool {
    unsafe {
        let mut any = vdupq_n_u16(0);
        for hits in hits {
            any = vorrq_u16(any, hits);
        }
        vmaxvq_u16(any) != 0
    }
}

#[inline(never)]
fn neon_emit<O: Offset, const VECTORS: usize>(
    base: usize,
    hits: [uint16x8_t; VECTORS],
    sink: &mut RowSink<'_, O>,
) {
    for (vector, hits) in hits.into_iter().enumerate() {
        let mut lanes = [0u16; 8];
        unsafe { vst1q_u16(lanes.as_mut_ptr(), hits) };
        mark_block(base + vector * 8, &lanes, sink);
    }
}

struct Neon<const VECTORS: usize>;

impl<const VECTORS: usize> Isa for Neon<VECTORS> {
    const BLOCK: usize = VECTORS * 8;
    const WALK_REMAINDER: bool = false;

    type Point = uint16x8_t;
    type Range = NeonRange;
    type Hits = [uint16x8_t; VECTORS];
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
        neon_block::<VECTORS, POINTS, RANGES>(codes, points, ranges)
    }

    #[inline(always)]
    fn any(hits: Self::Hits) -> bool {
        neon_any(hits)
    }

    #[inline(always)]
    fn emit<O: Offset>(base: usize, hits: Self::Hits, sink: &mut RowSink<'_, O>) {
        neon_emit(base, hits, sink);
    }
}

/// Prepare a bounded run-time cover, then enter the paired-vector walk.
#[inline(never)]
fn scan_bounded<
    O: Offset,
    const POINT_CAPACITY: usize,
    const RANGE_CAPACITY: usize,
    const RANGES: usize,
>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    debug_assert!(pf.points.len() <= POINT_CAPACITY);
    debug_assert!(pf.ranges.len() <= RANGE_CAPACITY);
    debug_assert!(RANGES == DYN || pf.ranges.len() == RANGES);
    let zero = neon_point(0);
    let mut points = [zero; POINT_CAPACITY];
    for (point, &token) in points.iter_mut().zip(&pf.points) {
        *point = neon_point(token);
    }
    let mut ranges = [(zero, zero); RANGE_CAPACITY];
    for (range, &source) in ranges.iter_mut().zip(&pf.ranges) {
        *range = neon_range(source);
    }
    unsafe {
        walk::<Neon<2>, O, DYN, RANGES, 1>(
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

#[inline(never)]
fn scan_fixed<O: Offset, const POINTS: usize, const RANGES: usize>(
    codes: &[Token],
    offsets: &[O],
    cover: &ProbeCover,
    sparse: bool,
    out: &mut Vec<usize>,
) {
    unsafe { fixed::<Neon<2>, O, POINTS, RANGES, 1>(codes, offsets, cover, sparse, out) }
}

/// Execute one of the small-cover schedules selected by NEON policy.
#[inline]
pub(super) fn scan_neon_fixed<O: Offset>(
    shape: FixedShape,
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    match (shape.points, shape.ranges) {
        (1, 0) => scan_fixed::<O, 1, 0>(codes, row_offsets, pf, sparse_row_mapping, out),
        (2, 0) => scan_fixed::<O, 2, 0>(codes, row_offsets, pf, sparse_row_mapping, out),
        (3, 0) => scan_fixed::<O, 3, 0>(codes, row_offsets, pf, sparse_row_mapping, out),
        (4..=8, 0) => scan_bounded::<O, 8, 0, 0>(codes, row_offsets, pf, sparse_row_mapping, out),
        (9..=16, 0) => scan_bounded::<O, 16, 0, 0>(codes, row_offsets, pf, sparse_row_mapping, out),
        (0, 1) => scan_fixed::<O, 0, 1>(codes, row_offsets, pf, sparse_row_mapping, out),
        (1, 1) => scan_fixed::<O, 1, 1>(codes, row_offsets, pf, sparse_row_mapping, out),
        (2, 1) => scan_fixed::<O, 2, 1>(codes, row_offsets, pf, sparse_row_mapping, out),
        (1, 2) => scan_fixed::<O, 1, 2>(codes, row_offsets, pf, sparse_row_mapping, out),
        (points, ranges) if ranges != 0 && points + 2 * ranges <= 16 => {
            scan_bounded::<O, 16, 8, DYN>(codes, row_offsets, pf, sparse_row_mapping, out)
        }
        _ => unreachable!("invalid fixed NEON shape"),
    }
}

/// Arbitrary covers retain the existing one- or two-vector policy while both
/// schedules share the template walk and prepared probe vectors.
pub(super) fn scan_neon_generic<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    use_two_vectors: bool,
    out: &mut Vec<usize>,
) {
    unsafe {
        if use_two_vectors {
            scan_dynamic::<Neon<2>, O>(codes, row_offsets, pf, sparse_row_mapping, out)
        } else {
            scan_dynamic::<Neon<1>, O>(codes, row_offsets, pf, sparse_row_mapping, out)
        }
    }
}

/// Direct kernel entry retained for architecture correctness tests.
#[cfg(test)]
pub(in crate::search::substring::prefilter) fn scan_neon<O: Offset>(
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
    let plan = policy::select_kernel(
        TargetCaps::Aarch64Neon,
        ScanFacts {
            analysis: AnalysisFacts {
                shape,
                covered_codes: 1,
                indexed_codes: 1,
            },
            region: RegionFacts {
                code_count: codes.len(),
                row_count: row_offsets.len().saturating_sub(1),
            },
        },
    );
    execute_neon(
        plan.shape,
        plan.group,
        ScanInput::full(codes, row_offsets, pf),
        sparse_row_mapping,
        out,
    );
}
