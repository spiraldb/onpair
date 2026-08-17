// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vectorized scans of the code stream against a compiled cover.
//!
//! Each kernel walks the flat code stream a vector at a time, ORs together one
//! comparison per point and two per range, and appends the rows the surviving
//! lanes fall in. They differ only in vector width and in how the target spells
//! an unsigned 16-bit comparison.
//!
//! There is deliberately **no production scalar fallback**. Every non-empty
//! cover runs through a vector kernel when the target has one, regardless of its
//! width. Whether that work is profitable is a policy decision made before this
//! module is called. The scalar routine below exists only under `cfg(test)`, as
//! the oracle the four kernels are proven against.

use super::PrefilterError;
use super::cover::ProbeCover;
use crate::core::offset::Offset;
use crate::core::types::{Token, TokenRange};

/// Dispatch to the widest available SIMD kernel; never fall back to a scalar scan.
pub(super) fn scan<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) -> Result<(), PrefilterError> {
    // Nothing to compare against, so nothing can match. Answered here rather
    // than by a kernel, both because scanning for no probes is wasted work and
    // because the answer is exact on any target — a cover this narrow must not be
    // turned away for want of SIMD.
    if pf.is_empty() {
        return Ok(());
    }

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
        if std::is_x86_feature_detected!("avx512bw") {
            // SAFETY: AVX-512BW was detected and implies AVX-512F.
            unsafe { scan_avx512(codes, row_offsets, pf, sparse_row_mapping, out) };
        } else if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 was detected above.
            unsafe { scan_avx2(codes, row_offsets, pf, sparse_row_mapping, out) };
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

/// SSE2: eight lanes; XOR with `0x8000` maps unsigned range order to signed.
#[cfg(target_arch = "x86_64")]
pub(super) fn scan_sse2<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    use core::arch::x86_64::{
        __m128i, _mm_andnot_si128, _mm_cmpeq_epi16, _mm_cmpgt_epi16, _mm_loadu_si128,
        _mm_movemask_epi8, _mm_or_si128, _mm_set1_epi16, _mm_setzero_si128, _mm_storeu_si128,
        _mm_xor_si128,
    };

    let total = codes.len();
    let base = codes.as_ptr();
    let mut sink = RowSink::new(row_offsets, out, sparse_row_mapping);
    let mut i = 0usize;
    while i + 8 <= total {
        // SAFETY: `i + 8 <= total`; SSE2 is an x86-64 baseline feature.
        let hits = unsafe {
            let v = _mm_loadu_si128(base.add(i).cast::<__m128i>());
            let bias = _mm_set1_epi16(i16::MIN); // 0x8000: unsigned → signed order
            let cb = _mm_xor_si128(v, bias); // codes in sign-biased space
            let ones = _mm_set1_epi16(-1);
            let mut acc = _mm_setzero_si128();
            for &p in &pf.points {
                acc = _mm_or_si128(acc, _mm_cmpeq_epi16(v, _mm_set1_epi16(p as i16)));
            }
            for &TokenRange { begin, last } in &pf.ranges {
                let lob = _mm_xor_si128(_mm_set1_epi16(begin as i16), bias);
                let hib = _mm_xor_si128(_mm_set1_epi16(last as i16), bias);
                // Out of range = below lo OR above hi; in-range is its complement.
                let below = _mm_cmpgt_epi16(lob, cb);
                let above = _mm_cmpgt_epi16(cb, hib);
                let out = _mm_or_si128(below, above);
                acc = _mm_or_si128(acc, _mm_andnot_si128(out, ones));
            }
            if _mm_movemask_epi8(acc) == 0 {
                None
            } else {
                let mut m = [0u16; 8];
                _mm_storeu_si128(m.as_mut_ptr().cast::<__m128i>(), acc);
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

/// AVX2: sixteen lanes with the same sign-biased range comparison as SSE2.
#[cfg(target_arch = "x86_64")]
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
