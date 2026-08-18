// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! AArch64 NEON prefilter kernels.

use super::super::sink::{RowSink, mark_block, scan_tail};
#[cfg(test)]
use super::super::{CoverShape, ScanInput, policy};
#[cfg(test)]
use super::execute as execute_neon;
use crate::core::offset::Offset;
use crate::core::types::{Token, TokenRange};
use crate::search::prefilter::cover::ProbeCover;

/// Shared two-vector walk. `matching_masks` is always inlined, leaving each
/// caller with a shape-specific compare loop and no indirect calls.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn scan_neon_two_vectors<O: Offset>(
    codes: &[Token],
    sink: &mut RowSink<'_, O>,
    mut matching_masks: impl FnMut(
        core::arch::aarch64::uint16x8_t,
        core::arch::aarch64::uint16x8_t,
    ) -> (
        core::arch::aarch64::uint16x8_t,
        core::arch::aarch64::uint16x8_t,
    ),
) -> usize {
    use core::arch::aarch64::{vld1q_u16, vmaxvq_u16, vorrq_u16, vst1q_u16};

    let total = codes.len();
    let base = codes.as_ptr();
    let mut i = 0usize;
    while i + 16 <= total {
        // SAFETY: `i + 16 <= total`; both vector loads are in bounds.
        let (acc0, acc1) = unsafe { (vld1q_u16(base.add(i)), vld1q_u16(base.add(i + 8))) };
        let (acc0, acc1) = matching_masks(acc0, acc1);
        let hits = unsafe {
            if vmaxvq_u16(vorrq_u16(acc0, acc1)) == 0 {
                None
            } else {
                let mut m0 = [0u16; 8];
                let mut m1 = [0u16; 8];
                vst1q_u16(m0.as_mut_ptr(), acc0);
                vst1q_u16(m1.as_mut_ptr(), acc1);
                Some((m0, m1))
            }
        };
        if let Some((m0, m1)) = hits {
            mark_block(i, &m0, sink);
            mark_block(i + 8, &m1, sink);
        }
        i += 16;
    }
    i
}

/// Point-only covers with 1–8 probes use a two-vector schedule. Exact one-,
/// two-, and three-point shapes are unrolled; 4–8 share the compact dynamic
/// instantiation. Keeping this out of line protects the generic kernel's code
/// generation.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn scan_neon_few_points<O: Offset, const N: usize>(
    codes: &[Token],
    points: &[Token],
    sink: &mut RowSink<'_, O>,
) -> usize {
    use core::arch::aarch64::{vceqq_u16, vdupq_n_u16, vorrq_u16};

    let probe_count = if N == 0 {
        points.len()
    } else {
        debug_assert_eq!(points.len(), N);
        N
    };
    let mut probes = [unsafe { vdupq_n_u16(0) }; 8];
    for j in 0..probe_count {
        probes[j] = unsafe { vdupq_n_u16(*points.get_unchecked(j)) };
    }
    let probes = &probes[..probe_count];
    scan_neon_two_vectors(codes, sink, |v0, v1| unsafe {
        let mut acc0 = vdupq_n_u16(0);
        let mut acc1 = vdupq_n_u16(0);
        for &probe in probes {
            acc0 = vorrq_u16(acc0, vceqq_u16(v0, probe));
            acc1 = vorrq_u16(acc1, vceqq_u16(v1, probe));
        }
        (acc0, acc1)
    })
}

/// Point-only covers at the default policy's upper boundary (9–16 checks).
/// Keep this separate from the common 1–8 kernel: doubling its probe array can
/// change stack layout and register allocation even for the smaller shapes.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn scan_neon_many_points<O: Offset>(
    codes: &[Token],
    points: &[Token],
    sink: &mut RowSink<'_, O>,
) -> usize {
    use core::arch::aarch64::{vceqq_u16, vdupq_n_u16, vorrq_u16};

    debug_assert!((9..=16).contains(&points.len()));
    let zero = unsafe { vdupq_n_u16(0) };
    let mut probes = [zero; 16];
    for (dst, &point) in probes.iter_mut().zip(points) {
        *dst = unsafe { vdupq_n_u16(point) };
    }
    let probes = &probes[..points.len()];
    scan_neon_two_vectors(codes, sink, |v0, v1| unsafe {
        let mut acc0 = zero;
        let mut acc1 = zero;
        for &probe in probes {
            acc0 = vorrq_u16(acc0, vceqq_u16(v0, probe));
            acc1 = vorrq_u16(acc1, vceqq_u16(v1, probe));
        }
        (acc0, acc1)
    })
}

/// A single range has enough independent work to amortize two vector loads, but
/// lives out of line so adding this shape cannot perturb the point hot paths.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
pub(in crate::search::prefilter::scan) fn scan_neon_one_range<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    use core::arch::aarch64::{vcleq_u16, vdupq_n_u16, vsubq_u16};

    let mut sink = RowSink::new(row_offsets, out, sparse_row_mapping);
    let range = pf.ranges[0];
    let lo = unsafe { vdupq_n_u16(range.begin) };
    let span = unsafe { vdupq_n_u16(range.last - range.begin) };
    let i = scan_neon_two_vectors(codes, &mut sink, |v0, v1| unsafe {
        // Unsigned wrapping subtraction maps [begin, last] to [0, span];
        // values outside the range wrap or exceed span.
        (
            vcleq_u16(vsubq_u16(v0, lo), span),
            vcleq_u16(vsubq_u16(v1, lo), span),
        )
    });
    scan_tail(codes, pf, i, &mut sink);
}

/// Exact mixed-cover shapes use const generics so LLVM can unroll every
/// compare while the source remains one reusable kernel. Broadcasts are built
/// once per query, outside the two-vector code loop.
#[cfg(target_arch = "aarch64")]
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
    use core::arch::aarch64::{
        uint16x8_t, vceqq_u16, vcleq_u16, vdupq_n_u16, vorrq_u16, vsubq_u16,
    };

    debug_assert_eq!(pf.points.len(), POINTS);
    debug_assert_eq!(pf.ranges.len(), RANGES);
    let mut sink = RowSink::new(row_offsets, out, sparse_row_mapping);
    let zero = unsafe { vdupq_n_u16(0) };
    let mut points = [zero; POINTS];
    for j in 0..POINTS {
        points[j] = unsafe { vdupq_n_u16(*pf.points.get_unchecked(j)) };
    }
    let mut ranges = [(zero, zero); RANGES];
    for j in 0..RANGES {
        let TokenRange { begin, last } = unsafe { *pf.ranges.get_unchecked(j) };
        ranges[j] = unsafe { (vdupq_n_u16(begin), vdupq_n_u16(last - begin)) };
    }
    let i = scan_neon_two_vectors(codes, &mut sink, |v0, v1| unsafe {
        let mut acc0: uint16x8_t = zero;
        let mut acc1: uint16x8_t = zero;
        for &point in &points {
            acc0 = vorrq_u16(acc0, vceqq_u16(v0, point));
            acc1 = vorrq_u16(acc1, vceqq_u16(v1, point));
        }
        for &(lo, span) in &ranges {
            acc0 = vorrq_u16(acc0, vcleq_u16(vsubq_u16(v0, lo), span));
            acc1 = vorrq_u16(acc1, vcleq_u16(vsubq_u16(v1, lo), span));
        }
        (acc0, acc1)
    });
    scan_tail(codes, pf, i, &mut sink);
}

/// One point plus two ranges needs an explicit reduction tree: the generic
/// const loop serializes its three ORs on current LLVM and loses throughput.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
pub(in crate::search::prefilter::scan) fn scan_neon_one_point_two_ranges<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    use core::arch::aarch64::{vceqq_u16, vcleq_u16, vdupq_n_u16, vorrq_u16, vsubq_u16};

    let mut sink = RowSink::new(row_offsets, out, sparse_row_mapping);
    let point = unsafe { vdupq_n_u16(pf.points[0]) };
    let range0 = pf.ranges[0];
    let range1 = pf.ranges[1];
    let lo0 = unsafe { vdupq_n_u16(range0.begin) };
    let span0 = unsafe { vdupq_n_u16(range0.last - range0.begin) };
    let lo1 = unsafe { vdupq_n_u16(range1.begin) };
    let span1 = unsafe { vdupq_n_u16(range1.last - range1.begin) };
    let i = scan_neon_two_vectors(codes, &mut sink, |v0, v1| unsafe {
        let range00 = vcleq_u16(vsubq_u16(v0, lo0), span0);
        let range01 = vcleq_u16(vsubq_u16(v0, lo1), span1);
        let range10 = vcleq_u16(vsubq_u16(v1, lo0), span0);
        let range11 = vcleq_u16(vsubq_u16(v1, lo1), span1);
        (
            vorrq_u16(vceqq_u16(v0, point), vorrq_u16(range00, range01)),
            vorrq_u16(vceqq_u16(v1, point), vorrq_u16(range10, range11)),
        )
    });
    scan_tail(codes, pf, i, &mut sink);
}

/// Compact fallback for small mixed covers that are too uncommon to justify a
/// separate unrolled kernel. It still hoists every broadcast and widens the
/// walk to two vectors; only the compare/OR loops remain dynamic.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
pub(in crate::search::prefilter::scan) fn scan_neon_few_mixed<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    use core::arch::aarch64::{
        uint16x8_t, vceqq_u16, vcleq_u16, vdupq_n_u16, vorrq_u16, vsubq_u16,
    };

    debug_assert!(!pf.ranges.is_empty());
    debug_assert!(pf.points.len() + 2 * pf.ranges.len() <= 16);
    let mut sink = RowSink::new(row_offsets, out, sparse_row_mapping);
    let zero = unsafe { vdupq_n_u16(0) };
    let mut points = [zero; 16];
    for (dst, &point) in points.iter_mut().zip(&pf.points) {
        *dst = unsafe { vdupq_n_u16(point) };
    }
    let points = &points[..pf.points.len()];
    let mut ranges = [(zero, zero); 8];
    for (dst, &TokenRange { begin, last }) in ranges.iter_mut().zip(&pf.ranges) {
        *dst = unsafe { (vdupq_n_u16(begin), vdupq_n_u16(last - begin)) };
    }
    let ranges = &ranges[..pf.ranges.len()];
    let i = scan_neon_two_vectors(codes, &mut sink, |v0, v1| unsafe {
        let mut acc0: uint16x8_t = zero;
        let mut acc1: uint16x8_t = zero;
        for &point in points {
            acc0 = vorrq_u16(acc0, vceqq_u16(v0, point));
            acc1 = vorrq_u16(acc1, vceqq_u16(v1, point));
        }
        for &(lo, span) in ranges {
            acc0 = vorrq_u16(acc0, vcleq_u16(vsubq_u16(v0, lo), span));
            acc1 = vorrq_u16(acc1, vcleq_u16(vsubq_u16(v1, lo), span));
        }
        (acc0, acc1)
    });
    scan_tail(codes, pf, i, &mut sink);
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub(in crate::search::prefilter::scan) fn scan_neon_points<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    let mut sink = RowSink::new(row_offsets, out, sparse_row_mapping);
    // Match directly on the slice length. Besides validating the policy plan,
    // this preserves the 4..=8 range fact that LLVM uses to bound the dynamic
    // probe loop in the compact shared specialization.
    let i = match pf.points.len() {
        1 => scan_neon_few_points::<O, 1>(codes, &pf.points, &mut sink),
        2 => scan_neon_few_points::<O, 2>(codes, &pf.points, &mut sink),
        3 => scan_neon_few_points::<O, 3>(codes, &pf.points, &mut sink),
        4..=8 => scan_neon_few_points::<O, 0>(codes, &pf.points, &mut sink),
        _ => unreachable!("invalid NEON point-family shape"),
    };
    scan_tail(codes, pf, i, &mut sink);
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub(in crate::search::prefilter::scan) fn scan_neon_points_many<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    let mut sink = RowSink::new(row_offsets, out, sparse_row_mapping);
    let i = scan_neon_many_points(codes, &pf.points, &mut sink);
    scan_tail(codes, pf, i, &mut sink);
}

/// NEON: eight `u16` lanes with native unsigned range comparisons.
#[cfg(target_arch = "aarch64")]
pub(in crate::search::prefilter::scan) fn scan_neon_generic<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    use_two_vectors: bool,
    out: &mut Vec<usize>,
) {
    use core::arch::aarch64::{
        vceqq_u16, vcleq_u16, vdupq_n_u16, vld1q_u16, vmaxvq_u16, vorrq_u16, vst1q_u16, vsubq_u16,
    };

    let total = codes.len();
    let base = codes.as_ptr();
    let mut sink = RowSink::new(row_offsets, out, sparse_row_mapping);
    let mut i = 0usize;
    if use_two_vectors {
        while i + 16 <= total {
            // Two independent vectors share each probe broadcast and expose two
            // compare/OR dependency chains to the out-of-order core. Combining
            // their masks before `vmaxvq` halves horizontal reductions on the
            // common no-hit path; both masks are stored only when either hits.
            let hits = unsafe {
                let v0 = vld1q_u16(base.add(i));
                let v1 = vld1q_u16(base.add(i + 8));
                let mut acc0 = vdupq_n_u16(0);
                let mut acc1 = vdupq_n_u16(0);
                for &p in &pf.points {
                    let probe = vdupq_n_u16(p);
                    acc0 = vorrq_u16(acc0, vceqq_u16(v0, probe));
                    acc1 = vorrq_u16(acc1, vceqq_u16(v1, probe));
                }
                for &TokenRange { begin, last } in &pf.ranges {
                    let lo = vdupq_n_u16(begin);
                    let span = vdupq_n_u16(last - begin);
                    acc0 = vorrq_u16(acc0, vcleq_u16(vsubq_u16(v0, lo), span));
                    acc1 = vorrq_u16(acc1, vcleq_u16(vsubq_u16(v1, lo), span));
                }
                if vmaxvq_u16(vorrq_u16(acc0, acc1)) == 0 {
                    None
                } else {
                    let mut m0 = [0u16; 8];
                    let mut m1 = [0u16; 8];
                    vst1q_u16(m0.as_mut_ptr(), acc0);
                    vst1q_u16(m1.as_mut_ptr(), acc1);
                    Some((m0, m1))
                }
            };
            if let Some((m0, m1)) = hits {
                mark_block(i, &m0, &mut sink);
                mark_block(i + 8, &m1, &mut sink);
            }
            i += 16;
        }
    }
    while i + 8 <= total {
        // SAFETY: `i + 8 <= total`; the load and stack store are in bounds.
        let hits = unsafe {
            let v = vld1q_u16(base.add(i));
            let mut acc = vdupq_n_u16(0);
            for &p in &pf.points {
                acc = vorrq_u16(acc, vceqq_u16(v, vdupq_n_u16(p)));
            }
            for &TokenRange { begin, last } in &pf.ranges {
                let lo = vdupq_n_u16(begin);
                let span = vdupq_n_u16(last - begin);
                acc = vorrq_u16(acc, vcleq_u16(vsubq_u16(v, lo), span));
            }
            if vmaxvq_u16(acc) == 0 {
                None
            } else {
                let mut m = [0u16; 8];
                vst1q_u16(m.as_mut_ptr(), acc);
                Some(m)
            }
        };
        if let Some(m) = hits {
            mark_block(i, &m, &mut sink);
        }
        i += 8;
    }
    scan_tail(codes, pf, i, &mut sink);
}

/// Direct kernel entry retained for architecture correctness tests.
#[cfg(all(target_arch = "aarch64", test))]
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
