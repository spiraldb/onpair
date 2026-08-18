// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scans of the code stream against a compiled cover.
//!
//! The vector kernels walk the flat code stream a vector at a time, OR together
//! one comparison per point and two per range, and append the rows the surviving
//! lanes fall in. On x86, wide dense covers may instead use an exact row-centric
//! membership-table scan with one early exit per row.
//!
//! There is no silent full-column fallback when SIMD is unavailable. The x86
//! row-centric path is an explicit density and row-length policy; other covers
//! use the detected vector kernel. The scalar routine under `cfg(test)` is the
//! common correctness oracle.

use super::PrefilterError;
use super::cover::ProbeCover;
use crate::core::offset::Offset;
use crate::core::types::{Token, TokenRange};

/// Dispatch to the measured kernel for the cover shape and detected ISA.
pub(super) fn scan<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    covered_frequency: usize,
    out: &mut Vec<usize>,
) -> Result<(), PrefilterError> {
    // Nothing to compare against, so nothing can match. Answered here rather
    // than by a kernel, both because scanning for no probes is wasted work and
    // because the answer is exact on any target — a cover this narrow must not be
    // turned away for want of SIMD.
    if pf.is_empty() {
        return Ok(());
    }

    // The analysis frequency counts covered code positions exactly. Keep that
    // full signal at the scan boundary, then derive the shared row-mapping
    // policy here so each architecture can make its own additional decisions.
    const SPARSE_ROW_MAPPING_DENOMINATOR: u128 = 10_000;
    let sparse_row_mapping = codes.is_empty()
        || covered_frequency as u128 * SPARSE_ROW_MAPPING_DENOMINATOR < codes.len() as u128;
    #[cfg(target_arch = "aarch64")]
    {
        match (pf.points.len(), pf.ranges.len()) {
            (0, 1) => scan_neon_one_range(codes, row_offsets, pf, sparse_row_mapping, out),
            (1, 1) => {
                scan_neon_fixed_mixed::<O, 1, 1>(codes, row_offsets, pf, sparse_row_mapping, out)
            }
            (2, 1) => {
                scan_neon_fixed_mixed::<O, 2, 1>(codes, row_offsets, pf, sparse_row_mapping, out)
            }
            (1, 2) => {
                scan_neon_one_point_two_ranges(codes, row_offsets, pf, sparse_row_mapping, out)
            }
            (points, ranges) if ranges != 0 && points + 2 * ranges <= 16 => {
                scan_neon_few_mixed(codes, row_offsets, pf, sparse_row_mapping, out)
            }
            _ => scan_neon(codes, row_offsets, pf, sparse_row_mapping, out),
        }
        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    {
        let row_count = row_offsets.len().saturating_sub(1);
        out.reserve(row_count.min(covered_frequency));
        let use_row_table =
            row_count != 0 && codes.len() / row_count >= 32 && covered_frequency >= row_count;
        let compact_one_point = covered_frequency as u128 * 10_000 >= codes.len() as u128 * 7;
        let compact_few_hits = covered_frequency as u128 * 2_000 >= codes.len() as u128;
        let use_nibble_six =
            pf.table.len() <= 1 << 12 && covered_frequency as u128 * 100 < codes.len() as u128;
        let max_compare_cost = if pf.table.len() <= 1 << 12 { 10 } else { 13 };
        let comparison_cost = pf.points.len() + 2 * pf.ranges.len();
        // Wide covers favor table lookup. Dense covers favor walking each row
        // directly, with a lower density crossover when rows contain enough
        // codes to amortize row mapping.
        let use_sse2_row_table = (use_row_table && comparison_cost > max_compare_cost)
            || (comparison_cost >= 17
                && (covered_frequency as u128 * 20 >= codes.len() as u128
                    || (row_count != 0
                        && codes.len() / row_count >= 8
                        && covered_frequency as u128 * 100 >= codes.len() as u128 * 3)));
        if std::is_x86_feature_detected!("avx512bw") {
            // SAFETY: AVX-512BW was detected and implies AVX-512F.
            unsafe { scan_avx512(codes, row_offsets, pf, sparse_row_mapping, out) };
        } else if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 was detected above.
            unsafe {
                match (pf.points.len(), pf.ranges.len()) {
                    (points, ranges) if use_row_table && points + 2 * ranges > max_compare_cost => {
                        scan_rows_table(codes, row_offsets, pf, out)
                    }
                    (9..=16, 0) if pf.table.len() <= (1 << 12) => {
                        scan_avx2_nibble_points(codes, row_offsets, pf, sparse_row_mapping, out)
                    }
                    (10..=16, 0) => {
                        scan_avx2_nibble_points(codes, row_offsets, pf, sparse_row_mapping, out)
                    }
                    (1, 0) => scan_avx2_one_point(
                        codes,
                        row_offsets,
                        pf,
                        sparse_row_mapping,
                        compact_one_point,
                        out,
                    ),
                    (0, 1) => scan_avx2_one_range(codes, row_offsets, pf, sparse_row_mapping, out),
                    (2, 0) => {
                        scan_avx2_fixed::<O, 2, 0>(codes, row_offsets, pf, sparse_row_mapping, out)
                    }
                    (3, 0) => {
                        scan_avx2_fixed::<O, 3, 0>(codes, row_offsets, pf, sparse_row_mapping, out)
                    }
                    (1, 1) => {
                        scan_avx2_fixed::<O, 1, 1>(codes, row_offsets, pf, sparse_row_mapping, out)
                    }
                    (2, 1) => {
                        scan_avx2_fixed::<O, 2, 1>(codes, row_offsets, pf, sparse_row_mapping, out)
                    }
                    (1, 2) => {
                        scan_avx2_fixed::<O, 1, 2>(codes, row_offsets, pf, sparse_row_mapping, out)
                    }
                    (4, 0) => {
                        scan_avx2_fixed::<O, 4, 0>(codes, row_offsets, pf, sparse_row_mapping, out)
                    }
                    (5, 0) => {
                        scan_avx2_fixed::<O, 5, 0>(codes, row_offsets, pf, sparse_row_mapping, out)
                    }
                    (6, 0) if use_nibble_six => {
                        scan_avx2_nibble_points(codes, row_offsets, pf, sparse_row_mapping, out)
                    }
                    (6, 0) => {
                        scan_avx2_fixed::<O, 6, 0>(codes, row_offsets, pf, sparse_row_mapping, out)
                    }
                    (0, 2) => {
                        scan_avx2_fixed::<O, 0, 2>(codes, row_offsets, pf, sparse_row_mapping, out)
                    }
                    (3, 1) => {
                        scan_avx2_fixed::<O, 3, 1>(codes, row_offsets, pf, sparse_row_mapping, out)
                    }
                    (4, 1) => {
                        scan_avx2_fixed::<O, 4, 1>(codes, row_offsets, pf, sparse_row_mapping, out)
                    }
                    (3, 2) => {
                        scan_avx2_fixed::<O, 3, 2>(codes, row_offsets, pf, sparse_row_mapping, out)
                    }
                    (2, 2) => {
                        scan_avx2_fixed::<O, 2, 2>(codes, row_offsets, pf, sparse_row_mapping, out)
                    }
                    (0, 3) => {
                        scan_avx2_fixed::<O, 0, 3>(codes, row_offsets, pf, sparse_row_mapping, out)
                    }
                    (1, 3) => {
                        scan_avx2_fixed::<O, 1, 3>(codes, row_offsets, pf, sparse_row_mapping, out)
                    }
                    (2, 3) => {
                        scan_avx2_fixed::<O, 2, 3>(codes, row_offsets, pf, sparse_row_mapping, out)
                    }
                    (5, 1) => {
                        scan_avx2_fixed::<O, 5, 1>(codes, row_offsets, pf, sparse_row_mapping, out)
                    }
                    (7, 0) => {
                        scan_avx2_nibble_points(codes, row_offsets, pf, sparse_row_mapping, out)
                    }
                    (4, 2) => {
                        scan_avx2_fixed::<O, 4, 2>(codes, row_offsets, pf, sparse_row_mapping, out)
                    }
                    (6, 1) => {
                        scan_avx2_fixed::<O, 6, 1>(codes, row_offsets, pf, sparse_row_mapping, out)
                    }
                    (8, 0) => {
                        scan_avx2_nibble_points(codes, row_offsets, pf, sparse_row_mapping, out)
                    }
                    (3, 3) => {
                        scan_avx2_fixed::<O, 3, 3>(codes, row_offsets, pf, sparse_row_mapping, out)
                    }
                    (5, 2) => {
                        scan_avx2_fixed::<O, 5, 2>(codes, row_offsets, pf, sparse_row_mapping, out)
                    }
                    (7, 1) => {
                        scan_avx2_fixed::<O, 7, 1>(codes, row_offsets, pf, sparse_row_mapping, out)
                    }
                    (6, 2) => {
                        scan_avx2_fixed::<O, 6, 2>(codes, row_offsets, pf, sparse_row_mapping, out)
                    }
                    (8, 1) => {
                        scan_avx2_fixed::<O, 8, 1>(codes, row_offsets, pf, sparse_row_mapping, out)
                    }
                    (points, ranges) if points + 2 * ranges <= max_compare_cost => {
                        if compact_few_hits {
                            scan_avx2_few::<O, true>(
                                codes,
                                row_offsets,
                                pf,
                                sparse_row_mapping,
                                out,
                            );
                        } else {
                            scan_avx2_few::<O, false>(
                                codes,
                                row_offsets,
                                pf,
                                sparse_row_mapping,
                                out,
                            );
                        }
                    }
                    _ => scan_avx2_gather(codes, row_offsets, pf, sparse_row_mapping, out),
                }
            }
        } else if use_sse2_row_table {
            scan_sse2_rows_table(codes, row_offsets, pf, out);
        } else {
            scan_sse2(codes, row_offsets, pf, sparse_row_mapping, out);
        }
        Ok(())
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = (
            codes,
            row_offsets,
            &pf.points,
            &pf.ranges,
            &pf.table,
            covered_frequency,
            sparse_row_mapping,
            out,
        );
        Err(PrefilterError::UnsupportedArchitecture)
    }
}

/// Test oracle for the SIMD implementations.
#[cfg(test)]
pub(super) fn scan_scalar<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    out: &mut Vec<usize>,
) {
    for row in 0..row_offsets.len().saturating_sub(1) {
        let a = row_offsets[row].to_usize();
        let b = row_offsets[row + 1].to_usize();
        if codes[a..b].iter().any(|&code| pf.table[code as usize]) {
            out.push(row);
        }
    }
}

/// Turns monotonically increasing code indices into ascending, deduplicated row
/// ids.
///
/// Every kernel visits code indices in increasing order, so the owning row only
/// moves forward and a row is finished the moment the scan leaves it. Candidates
/// can therefore be appended as they are discovered, rather than marked in a
/// per-row bitmap that has to be allocated, zeroed, and drained — work
/// proportional to the rows the prefilter *rejects*, which is the case it exists
/// for.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
struct RowSink<'a, O> {
    row_offsets: &'a [O],
    out: &'a mut Vec<usize>,
    /// Row owning the most recent hit.
    row: usize,
    /// End of `row`, or zero before the first hit. A hit below this belongs to a
    /// row that has already been appended.
    row_end: usize,
    binary_search_sparse_gaps: bool,
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
impl<'a, O: Offset> RowSink<'a, O> {
    #[inline]
    fn new(row_offsets: &'a [O], out: &'a mut Vec<usize>, binary_search_sparse_gaps: bool) -> Self {
        Self {
            row_offsets,
            out,
            row: 0,
            row_end: 0,
            binary_search_sparse_gaps,
        }
    }

    /// Record a hit at `code_index`, appending its row unless already appended.
    #[inline]
    fn hit(&mut self, code_index: usize) {
        if code_index < self.row_end {
            return;
        }
        // A sparse cover can jump across hundreds of thousands of rows between
        // hits. Walking every intervening offset makes candidate materialization
        // O(rows), even when the SIMD scan found only a handful of codes. Use a
        // lower-bound search for large code-space gaps; keep the linear cursor
        // for nearby hits, where its predictable sequential loads are cheaper.
        const BINARY_SEARCH_CODE_GAP: usize = 128;
        if self.binary_search_sparse_gaps
            && code_index.saturating_sub(self.row_end) >= BINARY_SEARCH_CODE_GAP
        {
            let suffix = &self.row_offsets[self.row + 1..];
            self.row += suffix.partition_point(|offset| offset.to_usize() <= code_index);
        } else {
            // Empty rows end at or before `code_index`, so this skips them too.
            while self.row + 1 < self.row_offsets.len()
                && self.row_offsets[self.row + 1].to_usize() <= code_index
            {
                self.row += 1;
            }
        }
        // `code_index` is a valid code index, so it lies below the last row
        // offset and the loop above always stops with `row + 1` in bounds.
        self.out.push(self.row);
        self.row_end = self.row_offsets[self.row + 1].to_usize();
    }

    // Record the hit rows named by a compact, ascending lane mask.
    #[inline]
    fn mark_mask(&mut self, base: usize, mut lanes: u64) {
        loop {
            // A previous hit may have emitted a row extending into this block.
            let consumed = self.row_end.saturating_sub(base);
            if consumed >= u64::BITS as usize {
                return;
            }
            lanes &= u64::MAX << consumed;
            if lanes == 0 {
                return;
            }
            self.hit(base + lanes.trailing_zeros() as usize);
        }
    }
}

/// Map a SIMD block's non-zero hit lanes to rows.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline]
fn mark_block<O: Offset>(base: usize, hits: &[u16], sink: &mut RowSink<'_, O>) {
    for (j, &h) in hits.iter().enumerate() {
        if h != 0 {
            sink.hit(base + j);
        }
    }
}

/// Scan the final partial SIMD block.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline]
fn scan_tail<O: Offset>(codes: &[Token], pf: &ProbeCover, from: usize, sink: &mut RowSink<'_, O>) {
    for (off, &c) in codes[from..].iter().enumerate() {
        if pf.table[c as usize] {
            sink.hit(from + off);
        }
    }
}

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
fn scan_neon_one_range<O: Offset>(
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
fn scan_neon_fixed_mixed<O: Offset, const POINTS: usize, const RANGES: usize>(
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
fn scan_neon_one_point_two_ranges<O: Offset>(
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
fn scan_neon_few_mixed<O: Offset>(
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

/// NEON: eight `u16` lanes with native unsigned range comparisons.
#[cfg(target_arch = "aarch64")]
pub(super) fn scan_neon<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    use core::arch::aarch64::{
        vceqq_u16, vcleq_u16, vdupq_n_u16, vld1q_u16, vmaxvq_u16, vorrq_u16, vst1q_u16, vsubq_u16,
    };

    let total = codes.len();
    let base = codes.as_ptr();
    let mut sink = RowSink::new(row_offsets, out, sparse_row_mapping);
    let mut i = 0usize;
    let use_two_vectors = (pf.points.len() == 1 && !pf.ranges.is_empty())
        || (pf.points.len() == 2 && pf.ranges.len() == 1);
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
    if pf.ranges.is_empty() && (1..=16).contains(&pf.points.len()) {
        i = match pf.points.len() {
            1 => scan_neon_few_points::<O, 1>(codes, &pf.points, &mut sink),
            2 => scan_neon_few_points::<O, 2>(codes, &pf.points, &mut sink),
            3 => scan_neon_few_points::<O, 3>(codes, &pf.points, &mut sink),
            4..=8 => scan_neon_few_points::<O, 0>(codes, &pf.points, &mut sink),
            9..=16 => scan_neon_many_points(codes, &pf.points, &mut sink),
            _ => unreachable!(),
        };
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
                    sink.mark_mask(i, lanes);
                }
            }
        }
        i += 32;
    }
    i
}

#[cfg(target_arch = "x86_64")]
fn scan_sse2_one_point<O: Offset>(
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
fn scan_sse2_fixed<O: Offset, const POINTS: usize, const RANGES: usize>(
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

#[cfg(target_arch = "x86_64")]
fn scan_sse2_rows_table<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    out: &mut Vec<usize>,
) {
    for row in 0..row_offsets.len().saturating_sub(1) {
        let begin = row_offsets[row].to_usize();
        let end = row_offsets[row + 1].to_usize();
        if codes[begin..end].iter().any(|&code| {
            // SAFETY: compressed codes are dictionary token ids, and the cover
            // table has exactly one entry per dictionary token.
            unsafe { *pf.table.get_unchecked(code as usize) }
        }) {
            out.push(row);
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn scan_sse2_codes_table<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    let mut sink = RowSink::new(row_offsets, out, sparse_row_mapping);
    for (code_index, &code) in codes.iter().enumerate() {
        // SAFETY: compressed codes are dictionary token ids, and the cover
        // table has exactly one entry per dictionary token.
        if unsafe { *pf.table.get_unchecked(code as usize) } {
            sink.hit(code_index);
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(super) fn scan_sse2<O: Offset>(
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
            scan_sse2_codes_table(codes, row_offsets, pf, sparse_row_mapping, out)
        }
        _ => scan_sse2_generic(codes, row_offsets, pf, sparse_row_mapping, out),
    }
}

/// SSE2: eight lanes; XOR with `0x8000` maps unsigned range order to signed.
#[cfg(target_arch = "x86_64")]
fn scan_sse2_generic<O: Offset>(
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

/// Compact four AVX2 vectors whose low byte in each `u16` lane is non-zero on a
/// hit. The high bytes are intentionally ignored: the nibble lookup uses them as
/// scratch output positions.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
fn compact_avx2_byte_hits(
    h0: core::arch::x86_64::__m256i,
    h1: core::arch::x86_64::__m256i,
    h2: core::arch::x86_64::__m256i,
    h3: core::arch::x86_64::__m256i,
    low_byte_mask: core::arch::x86_64::__m256i,
) -> u64 {
    use core::arch::x86_64::{
        _mm256_and_si256, _mm256_cmpeq_epi8, _mm256_movemask_epi8, _mm256_packus_epi16,
        _mm256_permute4x64_epi64, _mm256_setzero_si256,
    };

    let packed01 = _mm256_permute4x64_epi64::<0xd8>(_mm256_packus_epi16(
        _mm256_and_si256(h0, low_byte_mask),
        _mm256_and_si256(h1, low_byte_mask),
    ));
    let packed23 = _mm256_permute4x64_epi64::<0xd8>(_mm256_packus_epi16(
        _mm256_and_si256(h2, low_byte_mask),
        _mm256_and_si256(h3, low_byte_mask),
    ));
    let zero = _mm256_setzero_si256();
    let lanes01 = !(_mm256_movemask_epi8(_mm256_cmpeq_epi8(packed01, zero)) as u32) as u64;
    let lanes23 = !(_mm256_movemask_epi8(_mm256_cmpeq_epi8(packed23, zero)) as u32) as u64;
    lanes01 | (lanes23 << 32)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn scan_avx2_nibble_points_impl<O: Offset, const NIBBLES: usize>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    use core::arch::x86_64::{
        __m128i, __m256i, _mm_loadu_si128, _mm256_and_si256, _mm256_broadcastsi128_si256,
        _mm256_loadu_si256, _mm256_or_si256, _mm256_set1_epi16, _mm256_shuffle_epi8,
        _mm256_srli_epi16, _mm256_testz_si256,
    };

    debug_assert!(NIBBLES == 3 || NIBBLES == 4);
    let mut lookup = [[0u8; 16]; 4];
    for (j, &point) in pf.points.iter().enumerate() {
        let point_bit = 1u8 << j;
        lookup[0][(point & 0x000f) as usize] |= point_bit;
        lookup[1][((point >> 4) & 0x000f) as usize] |= point_bit;
        lookup[2][((point >> 8) & 0x000f) as usize] |= point_bit;
        lookup[3][((point >> 12) & 0x000f) as usize] |= point_bit;
    }

    let broadcast_lookup = |table: &[u8; 16]| unsafe {
        _mm256_broadcastsi128_si256(_mm_loadu_si128(table.as_ptr().cast::<__m128i>()))
    };
    let table0 = broadcast_lookup(&lookup[0]);
    let table1 = broadcast_lookup(&lookup[1]);
    let table2 = broadcast_lookup(&lookup[2]);
    let table3 = broadcast_lookup(&lookup[3]);
    let nibble_mask = _mm256_set1_epi16(0x000f);
    let low_byte_mask = _mm256_set1_epi16(0x00ff);
    let membership = |values: __m256i| {
        let index0 = _mm256_and_si256(values, nibble_mask);
        let index1 = _mm256_and_si256(_mm256_srli_epi16::<4>(values), nibble_mask);
        let index2 = if NIBBLES == 3 {
            // Every valid value is below 4096, so this byte is already a nibble.
            _mm256_srli_epi16::<8>(values)
        } else {
            _mm256_and_si256(_mm256_srli_epi16::<8>(values), nibble_mask)
        };
        let mut hits = _mm256_and_si256(
            _mm256_and_si256(
                _mm256_shuffle_epi8(table0, index0),
                _mm256_shuffle_epi8(table1, index1),
            ),
            _mm256_shuffle_epi8(table2, index2),
        );
        if NIBBLES == 4 {
            let index3 = _mm256_srli_epi16::<12>(values);
            hits = _mm256_and_si256(hits, _mm256_shuffle_epi8(table3, index3));
        }
        hits
    };

    let total = codes.len();
    let base = codes.as_ptr();
    let mut sink = RowSink::new(row_offsets, out, sparse_row_mapping);
    let mut i = 0usize;
    while i + 64 <= total {
        let h0 = membership(unsafe { _mm256_loadu_si256(base.add(i).cast::<__m256i>()) });
        let h1 = membership(unsafe { _mm256_loadu_si256(base.add(i + 16).cast::<__m256i>()) });
        let h2 = membership(unsafe { _mm256_loadu_si256(base.add(i + 32).cast::<__m256i>()) });
        let h3 = membership(unsafe { _mm256_loadu_si256(base.add(i + 48).cast::<__m256i>()) });
        let any = _mm256_or_si256(_mm256_or_si256(h0, h1), _mm256_or_si256(h2, h3));
        if _mm256_testz_si256(any, low_byte_mask) == 0 {
            sink.mark_mask(i, compact_avx2_byte_hits(h0, h1, h2, h3, low_byte_mask));
        }
        i += 64;
    }
    scan_tail(codes, pf, i, &mut sink);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn scan_avx2_nibble_points_two_banks_impl<O: Offset, const NIBBLES: usize>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    use core::arch::x86_64::{
        __m128i, __m256i, _mm_loadu_si128, _mm256_and_si256, _mm256_broadcastsi128_si256,
        _mm256_loadu_si256, _mm256_or_si256, _mm256_set1_epi16, _mm256_shuffle_epi8,
        _mm256_srli_epi16, _mm256_testz_si256,
    };

    debug_assert!(NIBBLES == 3 || NIBBLES == 4);
    let mut lookup = [[[0u8; 16]; 4]; 2];
    for (j, &point) in pf.points.iter().enumerate() {
        let bank = j / 8;
        let point_bit = 1u8 << (j % 8);
        lookup[bank][0][(point & 0x000f) as usize] |= point_bit;
        lookup[bank][1][((point >> 4) & 0x000f) as usize] |= point_bit;
        lookup[bank][2][((point >> 8) & 0x000f) as usize] |= point_bit;
        lookup[bank][3][((point >> 12) & 0x000f) as usize] |= point_bit;
    }

    let broadcast_lookup = |table: &[u8; 16]| unsafe {
        _mm256_broadcastsi128_si256(_mm_loadu_si128(table.as_ptr().cast::<__m128i>()))
    };
    let bank0_table0 = broadcast_lookup(&lookup[0][0]);
    let bank0_table1 = broadcast_lookup(&lookup[0][1]);
    let bank0_table2 = broadcast_lookup(&lookup[0][2]);
    let bank0_table3 = broadcast_lookup(&lookup[0][3]);
    let bank1_table0 = broadcast_lookup(&lookup[1][0]);
    let bank1_table1 = broadcast_lookup(&lookup[1][1]);
    let bank1_table2 = broadcast_lookup(&lookup[1][2]);
    let bank1_table3 = broadcast_lookup(&lookup[1][3]);
    let nibble_mask = _mm256_set1_epi16(0x000f);
    let low_byte_mask = _mm256_set1_epi16(0x00ff);
    let membership = |values: __m256i| {
        let index0 = _mm256_and_si256(values, nibble_mask);
        let index1 = _mm256_and_si256(_mm256_srli_epi16::<4>(values), nibble_mask);
        let index2 = if NIBBLES == 3 {
            _mm256_srli_epi16::<8>(values)
        } else {
            _mm256_and_si256(_mm256_srli_epi16::<8>(values), nibble_mask)
        };

        let mut bank0_hits = _mm256_and_si256(
            _mm256_and_si256(
                _mm256_shuffle_epi8(bank0_table0, index0),
                _mm256_shuffle_epi8(bank0_table1, index1),
            ),
            _mm256_shuffle_epi8(bank0_table2, index2),
        );
        let mut bank1_hits = _mm256_and_si256(
            _mm256_and_si256(
                _mm256_shuffle_epi8(bank1_table0, index0),
                _mm256_shuffle_epi8(bank1_table1, index1),
            ),
            _mm256_shuffle_epi8(bank1_table2, index2),
        );
        if NIBBLES == 4 {
            let index3 = _mm256_srli_epi16::<12>(values);
            bank0_hits = _mm256_and_si256(bank0_hits, _mm256_shuffle_epi8(bank0_table3, index3));
            bank1_hits = _mm256_and_si256(bank1_hits, _mm256_shuffle_epi8(bank1_table3, index3));
        }
        _mm256_or_si256(bank0_hits, bank1_hits)
    };

    let total = codes.len();
    let base = codes.as_ptr();
    let mut sink = RowSink::new(row_offsets, out, sparse_row_mapping);
    let mut i = 0usize;
    while i + 64 <= total {
        let h0 = membership(unsafe { _mm256_loadu_si256(base.add(i).cast::<__m256i>()) });
        let h1 = membership(unsafe { _mm256_loadu_si256(base.add(i + 16).cast::<__m256i>()) });
        let h2 = membership(unsafe { _mm256_loadu_si256(base.add(i + 32).cast::<__m256i>()) });
        let h3 = membership(unsafe { _mm256_loadu_si256(base.add(i + 48).cast::<__m256i>()) });
        let any = _mm256_or_si256(_mm256_or_si256(h0, h1), _mm256_or_si256(h2, h3));
        if _mm256_testz_si256(any, low_byte_mask) == 0 {
            sink.mark_mask(i, compact_avx2_byte_hits(h0, h1, h2, h3, low_byte_mask));
        }
        i += 64;
    }
    scan_tail(codes, pf, i, &mut sink);
}

/// Exact AVX2 `vpshufb` membership scan for a point-only cover of at most
/// sixteen token ids. Covers above eight points use two independent lookup
/// banks whose per-lane membership results are unioned.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(super) fn scan_avx2_nibble_points<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    assert!((1..=16).contains(&pf.points.len()));
    assert!(pf.ranges.is_empty());
    if pf.points.len() <= 8 {
        if pf.table.len() <= 1 << 12 {
            scan_avx2_nibble_points_impl::<O, 3>(codes, row_offsets, pf, sparse_row_mapping, out);
        } else {
            scan_avx2_nibble_points_impl::<O, 4>(codes, row_offsets, pf, sparse_row_mapping, out);
        }
    } else if pf.table.len() <= 1 << 12 {
        scan_avx2_nibble_points_two_banks_impl::<O, 3>(
            codes,
            row_offsets,
            pf,
            sparse_row_mapping,
            out,
        );
    } else {
        scan_avx2_nibble_points_two_banks_impl::<O, 4>(
            codes,
            row_offsets,
            pf,
            sparse_row_mapping,
            out,
        );
    }
}

/// Walk four AVX2 vectors per iteration and materialize only blocks with hits.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
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
                sink.mark_mask(i, compact_avx2_masks(m0, m1, m2, m3));
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
pub(super) fn scan_avx2_one_point<O: Offset>(
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
pub(super) fn scan_avx2_one_range<O: Offset>(
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
pub(super) fn scan_avx2_fixed<O: Offset, const POINTS: usize, const RANGES: usize>(
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

/// Exact row-centric membership scan with one early exit per row.
#[cfg(target_arch = "x86_64")]
pub(super) fn scan_rows_table<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    out: &mut Vec<usize>,
) {
    for row in 0..row_offsets.len().saturating_sub(1) {
        let begin = row_offsets[row].to_usize();
        let end = row_offsets[row + 1].to_usize();
        if codes[begin..end]
            .iter()
            .any(|&code| pf.table[code as usize])
        {
            out.push(row);
        }
    }
}

/// Pack the normalized point/range cover without revisiting every dictionary id.
#[cfg(target_arch = "x86_64")]
pub(super) fn pack_membership(pf: &ProbeCover) -> Vec<u32> {
    const WORD_BITS: usize = u32::BITS as usize;
    let mut table = vec![0u32; pf.table.len().div_ceil(WORD_BITS)];
    for &point in &pf.points {
        let id = point as usize;
        table[id / WORD_BITS] |= 1 << (id % WORD_BITS);
    }
    for &TokenRange { begin, last } in &pf.ranges {
        let begin = begin as usize;
        let last = last as usize;
        let begin_word = begin / WORD_BITS;
        let last_word = last / WORD_BITS;
        let first_mask = u32::MAX << (begin % WORD_BITS);
        let last_mask = u32::MAX >> (WORD_BITS - 1 - last % WORD_BITS);
        if begin_word == last_word {
            table[begin_word] |= first_mask & last_mask;
        } else {
            table[begin_word] |= first_mask;
            table[begin_word + 1..last_word].fill(u32::MAX);
            table[last_word] |= last_mask;
        }
    }
    table
}

/// Pack membership by walking the complete dictionary-sized table.
///
/// This is cheaper than expanding many normalized ranges for very wide 12-bit
/// covers, where the source table is only 4 KiB and remains cache-resident.
#[cfg(target_arch = "x86_64")]
pub(super) fn pack_membership_dense(pf: &ProbeCover) -> Vec<u32> {
    const WORD_BITS: usize = u32::BITS as usize;
    let mut packed = vec![0u32; pf.table.len().div_ceil(WORD_BITS)];
    for (id, selected) in pf.table.iter().copied().enumerate() {
        packed[id / WORD_BITS] |= u32::from(selected) << (id % WORD_BITS);
    }
    packed
}

/// Exact AVX2 gather from a compact one-bit-per-token membership table.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(super) fn scan_avx2_gather<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    use core::arch::x86_64::{
        __m128i, __m256i, _mm_loadu_si128, _mm256_and_si256, _mm256_castsi256_ps,
        _mm256_cvtepu16_epi32, _mm256_i32gather_epi32, _mm256_movemask_ps, _mm256_set1_epi32,
        _mm256_slli_epi32, _mm256_srli_epi32, _mm256_srlv_epi32,
    };

    let comparison_cost = pf.points.len() + 2 * pf.ranges.len();
    let table = if pf.table.len() <= 1 << 12 && comparison_cost > 128 {
        pack_membership_dense(pf)
    } else {
        pack_membership(pf)
    };

    let total = codes.len();
    let base = codes.as_ptr();
    let table_base = table.as_ptr().cast::<i32>();
    let low_bit = _mm256_set1_epi32(1);
    let bit_mask = _mm256_set1_epi32(31);
    let gather_mask = |indices: __m256i| unsafe {
        let word_indices = _mm256_srli_epi32::<5>(indices);
        let words = _mm256_i32gather_epi32::<4>(table_base, word_indices);
        let bit_indices = _mm256_and_si256(indices, bit_mask);
        let bits = _mm256_and_si256(_mm256_srlv_epi32(words, bit_indices), low_bit);
        _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_slli_epi32::<31>(bits))) as u32
    };

    let mut sink = RowSink::new(row_offsets, out, sparse_row_mapping);
    let mut i = 0usize;
    while i + 32 <= total {
        let (m0, m1, m2, m3) = unsafe {
            (
                gather_mask(_mm256_cvtepu16_epi32(_mm_loadu_si128(
                    base.add(i).cast::<__m128i>(),
                ))),
                gather_mask(_mm256_cvtepu16_epi32(_mm_loadu_si128(
                    base.add(i + 8).cast::<__m128i>(),
                ))),
                gather_mask(_mm256_cvtepu16_epi32(_mm_loadu_si128(
                    base.add(i + 16).cast::<__m128i>(),
                ))),
                gather_mask(_mm256_cvtepu16_epi32(_mm_loadu_si128(
                    base.add(i + 24).cast::<__m128i>(),
                ))),
            )
        };
        let lanes =
            u64::from(m0) | (u64::from(m1) << 8) | (u64::from(m2) << 16) | (u64::from(m3) << 24);
        if lanes != 0 {
            sink.mark_mask(i, lanes);
        }
        i += 32;
    }
    scan_tail(codes, pf, i, &mut sink);
}

/// AVX2 fast path for covers costing at most sixteen comparisons.
///
/// Probe broadcasts are prepared once, then two code vectors are evaluated
/// together. The common no-hit path pays one combined movemask per 32 codes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(super) fn scan_avx2_few<O: Offset, const COMPACT_HITS: bool>(
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
                sink.mark_mask(i, u64::from(compact_two_avx2_masks(acc0, acc1)));
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

/// AVX2: sixteen lanes with the same sign-biased range comparison as SSE2.
#[cfg(all(target_arch = "x86_64", test))]
#[target_feature(enable = "avx2")]
pub(super) fn scan_avx2<O: Offset>(
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

    let total = codes.len();
    let base = codes.as_ptr();
    let mut sink = RowSink::new(row_offsets, out, sparse_row_mapping);
    let mut i = 0usize;
    while i + 16 <= total {
        // SAFETY: `i + 16 <= total`; the caller established AVX2.
        let hits = unsafe {
            let v = _mm256_loadu_si256(base.add(i).cast::<__m256i>());
            let bias = _mm256_set1_epi16(i16::MIN);
            let cb = _mm256_xor_si256(v, bias);
            let ones = _mm256_set1_epi16(-1);
            let mut acc = _mm256_setzero_si256();
            for &p in &pf.points {
                acc = _mm256_or_si256(acc, _mm256_cmpeq_epi16(v, _mm256_set1_epi16(p as i16)));
            }
            for &TokenRange { begin, last } in &pf.ranges {
                let lob = _mm256_xor_si256(_mm256_set1_epi16(begin as i16), bias);
                let hib = _mm256_xor_si256(_mm256_set1_epi16(last as i16), bias);
                let below = _mm256_cmpgt_epi16(lob, cb);
                let above = _mm256_cmpgt_epi16(cb, hib);
                let out = _mm256_or_si256(below, above);
                acc = _mm256_or_si256(acc, _mm256_andnot_si256(out, ones));
            }
            if _mm256_movemask_epi8(acc) == 0 {
                None
            } else {
                let mut m = [0u16; 16];
                _mm256_storeu_si256(m.as_mut_ptr().cast::<__m256i>(), acc);
                Some(m)
            }
        };
        if let Some(m) = hits {
            mark_block(i, &m, &mut sink);
        }
        i += 16;
    }
    scan_tail(codes, pf, i, &mut sink);
}

/// AVX-512BW: 32 lanes with native unsigned comparisons and mask output.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub(super) fn scan_avx512<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    use core::arch::x86_64::{
        _mm512_cmpeq_epu16_mask, _mm512_cmpge_epu16_mask, _mm512_cmple_epu16_mask,
        _mm512_loadu_si512, _mm512_set1_epi16,
    };

    let total = codes.len();
    let base = codes.as_ptr();
    let mut sink = RowSink::new(row_offsets, out, sparse_row_mapping);
    let mut i = 0usize;
    while i + 32 <= total {
        // SAFETY: `i + 32 <= total`; the caller established AVX-512F/BW.
        let mut m = unsafe {
            let v = _mm512_loadu_si512(base.add(i).cast());
            let mut acc: u32 = 0;
            for &p in &pf.points {
                acc |= _mm512_cmpeq_epu16_mask(v, _mm512_set1_epi16(p as i16));
            }
            for &TokenRange { begin, last } in &pf.ranges {
                let ge = _mm512_cmpge_epu16_mask(v, _mm512_set1_epi16(begin as i16));
                let le = _mm512_cmple_epu16_mask(v, _mm512_set1_epi16(last as i16));
                acc |= ge & le;
            }
            acc
        };
        // Lowest set lane first, so code indices stay increasing.
        while m != 0 {
            let j = m.trailing_zeros() as usize;
            sink.hit(i + j);
            m &= m - 1;
        }
        i += 32;
    }
    scan_tail(codes, pf, i, &mut sink);
}
