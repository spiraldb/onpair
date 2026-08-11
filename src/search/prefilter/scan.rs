// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vectorized scans of the code stream against a compiled cover.
//!
//! Each kernel walks the flat code stream a vector at a time, ORs together one
//! comparison per point and two per range, and appends the rows the surviving
//! lanes fall in. They differ only in vector width and in how the target spells
//! an unsigned 16-bit comparison.
//!
//! There is deliberately **no production scalar path**. Past [`SIMD_CAP`] a cover
//! is refused rather than scanned one code at a time, so a query can never
//! silently fall off a vector kernel onto something slower than the exact check
//! it was meant to avoid. The scalar routine below exists only under `cfg(test)`,
//! as the oracle the four kernels are proven against.

use super::PrefilterError;
use super::cover::ProbeCover;
use crate::core::offset::Offset;
use crate::core::types::{Token, TokenRange};

/// Use the vectorized scan only while the cover's probe count stays at or below
/// this. The public API refuses a wider cover instead of silently switching to a
/// scalar algorithm.
///
/// The unit is probes, not machine comparisons: a range issues two compares and
/// an AND where a point issues one compare, so an all-range cover does roughly
/// twice the work of an all-point cover at the same budget.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(super) const SIMD_CAP: usize = 32;

/// Dispatch to the widest available SIMD kernel; never fall back to a scalar scan.
pub(super) fn scan<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    out: &mut Vec<usize>,
) -> Result<(), PrefilterError> {
    // Nothing to compare against, so nothing can match. Answered here rather
    // than by a kernel, both because scanning for no probes is wasted work and
    // because the answer is exact on any target — a cover this narrow must not be
    // turned away for want of SIMD.
    if pf.is_empty() {
        return Ok(());
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    if pf.cmp_cost() > SIMD_CAP {
        return Err(PrefilterError::ProbeCoverTooWide);
    }

    #[cfg(target_arch = "aarch64")]
    {
        scan_neon(codes, row_offsets, pf, out);
        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512bw") {
            // SAFETY: AVX-512BW was detected and implies AVX-512F.
            unsafe { scan_avx512(codes, row_offsets, pf, out) };
        } else if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 was detected above.
            unsafe { scan_avx2(codes, row_offsets, pf, out) };
        } else {
            scan_sse2(codes, row_offsets, pf, out);
        }
        Ok(())
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = (codes, row_offsets, &pf.points, &pf.ranges, &pf.table, out);
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
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
impl<'a, O: Offset> RowSink<'a, O> {
    #[inline]
    fn new(row_offsets: &'a [O], out: &'a mut Vec<usize>) -> Self {
        Self {
            row_offsets,
            out,
            row: 0,
            row_end: 0,
        }
    }

    /// Record a hit at `code_index`, appending its row unless already appended.
    #[inline]
    fn hit(&mut self, code_index: usize) {
        if code_index < self.row_end {
            return;
        }
        // Empty rows end at or before `code_index`, so this skips them too.
        while self.row + 1 < self.row_offsets.len()
            && self.row_offsets[self.row + 1].to_usize() <= code_index
        {
            self.row += 1;
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

/// NEON: eight `u16` lanes with native unsigned range comparisons.
#[cfg(target_arch = "aarch64")]
pub(super) fn scan_neon<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    out: &mut Vec<usize>,
) {
    use core::arch::aarch64::{
        vandq_u16, vceqq_u16, vcgeq_u16, vcleq_u16, vdupq_n_u16, vld1q_u16, vmaxvq_u16, vorrq_u16,
        vst1q_u16,
    };

    let total = codes.len();
    let base = codes.as_ptr();
    let mut sink = RowSink::new(row_offsets, out);
    let mut i = 0usize;
    while i + 8 <= total {
        // SAFETY: `i + 8 <= total`; the load and stack store are in bounds.
        let hits = unsafe {
            let v = vld1q_u16(base.add(i));
            let mut acc = vdupq_n_u16(0);
            for &p in &pf.points {
                acc = vorrq_u16(acc, vceqq_u16(v, vdupq_n_u16(p)));
            }
            for &TokenRange { begin, last } in &pf.ranges {
                let ge = vcgeq_u16(v, vdupq_n_u16(begin));
                let le = vcleq_u16(v, vdupq_n_u16(last));
                acc = vorrq_u16(acc, vandq_u16(ge, le));
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
    out: &mut Vec<usize>,
) {
    use core::arch::x86_64::{
        __m128i, _mm_andnot_si128, _mm_cmpeq_epi16, _mm_cmpgt_epi16, _mm_loadu_si128,
        _mm_movemask_epi8, _mm_or_si128, _mm_set1_epi16, _mm_setzero_si128, _mm_storeu_si128,
        _mm_xor_si128,
    };

    let total = codes.len();
    let base = codes.as_ptr();
    let mut sink = RowSink::new(row_offsets, out);
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
    out: &mut Vec<usize>,
) {
    use core::arch::x86_64::{
        __m256i, _mm256_andnot_si256, _mm256_cmpeq_epi16, _mm256_cmpgt_epi16, _mm256_loadu_si256,
        _mm256_movemask_epi8, _mm256_or_si256, _mm256_set1_epi16, _mm256_setzero_si256,
        _mm256_storeu_si256, _mm256_xor_si256,
    };

    let total = codes.len();
    let base = codes.as_ptr();
    let mut sink = RowSink::new(row_offsets, out);
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
    out: &mut Vec<usize>,
) {
    use core::arch::x86_64::{
        _mm512_cmpeq_epu16_mask, _mm512_cmpge_epu16_mask, _mm512_cmple_epu16_mask,
        _mm512_loadu_si512, _mm512_set1_epi16,
    };

    let total = codes.len();
    let base = codes.as_ptr();
    let mut sink = RowSink::new(row_offsets, out);
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
