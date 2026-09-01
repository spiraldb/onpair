// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! AVX-512BW prefilter execution.

use super::super::sink::{LaneMask, RowSink, scan_tail};
use super::super::template::{DYN, Isa, scan_fixed};
use crate::core::offset::Offset;
use crate::core::types::{Token, TokenRange};
use crate::search::substring::prefilter::cover::ProbeCover;

use core::arch::x86_64::{
    __m512i, _mm512_cmpeq_epu16_mask, _mm512_cmpge_epu16_mask, _mm512_cmple_epu16_mask,
    _mm512_loadu_si512, _mm512_mask_cmpgt_epu16_mask, _mm512_mask_cmpneq_epu16_mask,
    _mm512_set1_epi16, _mm512_sub_epi16,
};

struct Avx512;

#[inline(always)]
unsafe fn mask32<const POINTS: usize, const RANGES: usize>(
    codes: *const Token,
    points: &[__m512i],
    ranges: &[(__m512i, __m512i)],
) -> u32 {
    let points = if POINTS == DYN {
        points
    } else {
        debug_assert_eq!(points.len(), POINTS);
        // SAFETY: the fixed-shape prologue creates exactly POINTS broadcasts.
        unsafe { points.get_unchecked(..POINTS) }
    };
    let ranges = if RANGES == DYN {
        ranges
    } else {
        debug_assert_eq!(ranges.len(), RANGES);
        // SAFETY: the fixed-shape prologue creates exactly RANGES broadcasts.
        unsafe { ranges.get_unchecked(..RANGES) }
    };

    // Keep the accumulator in the miss domain. Each comparison only examines
    // lanes that no earlier point or range accepted.
    // SAFETY: the caller makes 32 codes readable and enables AVX-512F/BW.
    unsafe {
        let value = _mm512_loadu_si512(codes.cast());
        let mut miss = u32::MAX;
        for &point in points {
            miss = _mm512_mask_cmpneq_epu16_mask(miss, value, point);
        }
        for &(begin, span) in ranges {
            miss = _mm512_mask_cmpgt_epu16_mask(miss, _mm512_sub_epi16(value, begin), span);
        }
        !miss
    }
}

/// Combine the two 32-lane masks in one 64-code block.
///
/// Keeping the producer behind an `FnMut` boundary preserves the proven
/// compare/extract schedule: finish the low mask before starting the high
/// mask. Expressing the two calls directly lets LLVM overlap them through
/// separate mask registers, which regresses the dominant P1 shape on Zen 5.
#[inline(always)]
unsafe fn mask64(mut mask_at: impl FnMut(usize) -> u32) -> u64 {
    let lo = u64::from(mask_at(0));
    let hi = u64::from(mask_at(32));
    lo | (hi << 32)
}

impl Isa for Avx512 {
    const BLOCK: usize = 64;

    type Point = __m512i;
    type Range = (__m512i, __m512i);
    type Hits = u64;
    const NO_HITS: Self::Hits = 0;

    #[inline(always)]
    fn point(token: Token) -> Self::Point {
        // SAFETY: every caller is in the AVX-512 target-feature leaf.
        unsafe { _mm512_set1_epi16(token as i16) }
    }

    #[inline(always)]
    fn range(range: TokenRange) -> Self::Range {
        // SAFETY: every caller is in the AVX-512 target-feature leaf.
        unsafe {
            (
                _mm512_set1_epi16(range.begin as i16),
                _mm512_set1_epi16(range.last.wrapping_sub(range.begin) as i16),
            )
        }
    }

    #[inline(always)]
    unsafe fn block<const POINTS: usize, const RANGES: usize>(
        codes: *const Token,
        points: &[Self::Point],
        ranges: &[Self::Range],
    ) -> Self::Hits {
        // SAFETY: the walk makes 64 codes readable and the target-feature leaf
        // enables AVX-512F/BW.
        unsafe { mask64(|offset| mask32::<POINTS, RANGES>(codes.add(offset), points, ranges)) }
    }

    #[inline(always)]
    fn any(hits: Self::Hits) -> bool {
        hits != Self::NO_HITS
    }

    #[inline(never)]
    fn emit<O: Offset>(base: usize, hits: Self::Hits, sink: &mut RowSink<'_, O>) {
        sink.mark_mask(base, LaneMask::from_bits(hits));
    }
}

/// Proven AVX-512BW fallback for arbitrary cover shapes.
///
/// This intentionally keeps the compact, direct-probe loop. Preparing retained
/// broadcast vectors for arbitrary shapes was measured and rejected; only the
/// selected fixed shapes use the shared retained-mask template.
#[target_feature(enable = "avx512f,avx512bw")]
pub(in crate::search::substring::prefilter) fn scan_avx512<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    let total = codes.len();
    let base = codes.as_ptr();
    let mut sink = RowSink::new(row_offsets, out, sparse_row_mapping);
    let mut index = 0usize;
    while index + 32 <= total {
        // SAFETY: `index + 32 <= total`; the caller established AVX-512F/BW.
        let mut hits = unsafe {
            let value = _mm512_loadu_si512(base.add(index).cast());
            let mut mask = 0;
            for &point in &pf.points {
                mask |= _mm512_cmpeq_epu16_mask(value, _mm512_set1_epi16(point as i16));
            }
            for &TokenRange { begin, last } in &pf.ranges {
                let ge = _mm512_cmpge_epu16_mask(value, _mm512_set1_epi16(begin as i16));
                let le = _mm512_cmple_epu16_mask(value, _mm512_set1_epi16(last as i16));
                mask |= ge & le;
            }
            mask
        };
        while hits != 0 {
            let lane = hits.trailing_zeros() as usize;
            sink.hit(index + lane);
            hits &= hits - 1;
        }
        index += 32;
    }
    scan_tail(codes, pf, index, &mut sink);
}

/// Const-shape leaf for the six cover shapes that dominate the workload.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline(never)]
pub(in crate::search::substring::prefilter::scan) fn scan_avx512_fixed<
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
    // SAFETY: this target-feature wrapper establishes the template's ISA
    // precondition.
    unsafe {
        scan_fixed::<Avx512, O, POINTS, RANGES, 8>(
            codes,
            row_offsets,
            cover,
            sparse_row_mapping,
            out,
        )
    };
}
