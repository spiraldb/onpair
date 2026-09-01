// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! AVX2 prefilter kernels.

use super::super::sink::{LaneMask, RowSink};
use super::super::template::{DYN, Isa, scan_dynamic, scan_fixed};
use crate::core::offset::Offset;
use crate::core::types::{Token, TokenRange};
use crate::search::substring::prefilter::cover::ProbeCover;

use core::arch::x86_64::{
    __m256i, _mm256_cmpeq_epi16, _mm256_loadu_si256, _mm256_movemask_epi8, _mm256_or_si256,
    _mm256_packs_epi16, _mm256_permute4x64_epi64, _mm256_set1_epi16, _mm256_setzero_si256,
    _mm256_sub_epi16, _mm256_subs_epu16, _mm256_testz_si256,
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
    points: &[__m256i],
    ranges: &[(__m256i, __m256i)],
) -> u64 {
    debug_assert_eq!(points.len(), POINTS);
    debug_assert_eq!(ranges.len(), RANGES);
    // SAFETY: the fixed-shape prologue creates exactly these probe counts.
    let points = unsafe { points.get_unchecked(..POINTS) };
    // SAFETY: the fixed-shape prologue creates exactly these probe counts.
    let ranges = unsafe { ranges.get_unchecked(..RANGES) };

    let zero = _mm256_setzero_si256();
    let mut masks = [zero; 4];
    for (lane, mask) in masks.iter_mut().enumerate() {
        let code = unsafe { _mm256_loadu_si256(codes.add(lane * 16).cast()) };
        for &point in points {
            *mask = _mm256_or_si256(*mask, _mm256_cmpeq_epi16(code, point));
        }
        for &(begin, span) in ranges {
            let excess = _mm256_subs_epu16(_mm256_sub_epi16(code, begin), span);
            *mask = _mm256_or_si256(*mask, _mm256_cmpeq_epi16(excess, zero));
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

/// Preserve the arbitrary-shape fallback's point/range loop ordering and range
/// encoding while sharing the block gate and compaction vocabulary.
#[allow(
    unsafe_op_in_unsafe_fn,
    reason = "the template caller and 64-code bound satisfy every intrinsic and load"
)]
#[inline(always)]
unsafe fn dynamic_mask64(
    codes: *const Token,
    points: &[__m256i],
    ranges: &[(__m256i, __m256i)],
) -> u64 {
    let zero = _mm256_setzero_si256();
    let values = [
        unsafe { _mm256_loadu_si256(codes.cast()) },
        unsafe { _mm256_loadu_si256(codes.add(16).cast()) },
        unsafe { _mm256_loadu_si256(codes.add(32).cast()) },
        unsafe { _mm256_loadu_si256(codes.add(48).cast()) },
    ];
    let mut masks = [zero; 4];
    for &point in points {
        for (mask, &value) in masks.iter_mut().zip(&values) {
            *mask = _mm256_or_si256(*mask, _mm256_cmpeq_epi16(value, point));
        }
    }
    for &(begin, span) in ranges {
        for (mask, &value) in masks.iter_mut().zip(&values) {
            let excess = _mm256_subs_epu16(_mm256_sub_epi16(value, begin), span);
            *mask = _mm256_or_si256(*mask, _mm256_cmpeq_epi16(excess, zero));
        }
    }
    let any = _mm256_or_si256(
        _mm256_or_si256(masks[0], masks[1]),
        _mm256_or_si256(masks[2], masks[3]),
    );
    if _mm256_movemask_epi8(any) == 0 {
        return 0;
    }
    compact_masks(masks[0], masks[1], masks[2], masks[3])
}

struct Avx2<const DYNAMIC: bool>;

#[inline(never)]
fn emit_dynamic<O: Offset>(base: usize, hits: u64, sink: &mut RowSink<'_, O>) {
    sink.mark_mask(base, LaneMask::from_bits(hits));
}

impl<const DYNAMIC: bool> Isa for Avx2<DYNAMIC> {
    const BLOCK: usize = BLOCK_CODES;

    type Point = __m256i;
    type Range = (__m256i, __m256i);
    type Hits = u64;
    const NO_HITS: Self::Hits = 0;

    #[inline(always)]
    fn point(token: Token) -> Self::Point {
        // SAFETY: every caller is in the AVX2 target-feature leaf.
        unsafe { _mm256_set1_epi16(token as i16) }
    }

    #[inline(always)]
    fn range(range: TokenRange) -> Self::Range {
        // SAFETY: every caller is in the AVX2 target-feature leaf.
        unsafe {
            (
                _mm256_set1_epi16(range.begin as i16),
                _mm256_set1_epi16(range.last.wrapping_sub(range.begin) as i16),
            )
        }
    }

    #[inline(always)]
    unsafe fn block<const POINTS: usize, const RANGES: usize>(
        codes: *const Token,
        points: &[Self::Point],
        ranges: &[Self::Range],
    ) -> Self::Hits {
        if DYNAMIC {
            debug_assert_eq!(POINTS, DYN);
            debug_assert_eq!(RANGES, DYN);
            unsafe { dynamic_mask64(codes, points, ranges) }
        } else {
            debug_assert_ne!(POINTS, DYN);
            unsafe { fixed_mask64::<POINTS, RANGES>(codes, points, ranges) }
        }
    }

    #[inline(always)]
    fn any(hits: Self::Hits) -> bool {
        hits != Self::NO_HITS
    }

    #[inline(always)]
    fn emit<O: Offset>(base: usize, hits: Self::Hits, sink: &mut RowSink<'_, O>) {
        if DYNAMIC {
            emit_dynamic(base, hits, sink);
        } else {
            sink.mark_mask(base, LaneMask::from_bits(hits));
        }
    }
}

/// Const-shape AVX2 leaf for one or eight retained 64-code blocks at a time.
#[target_feature(enable = "avx2")]
#[inline(never)]
pub(in crate::search::substring::prefilter::scan) unsafe fn scan_avx2_fixed<
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
    debug_assert!(matches!(BLOCKS, 1 | 8));
    unsafe {
        scan_fixed::<Avx2<false>, O, POINTS, RANGES, BLOCKS>(
            codes,
            row_offsets,
            cover,
            sparse_row_mapping,
            out,
        )
    }
}

/// AVX2 fallback for arbitrary cover shapes.
#[target_feature(enable = "avx2")]
pub(in crate::search::substring::prefilter::scan) fn scan_avx2_generic<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    cover: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    // SAFETY: this target-feature leaf establishes the template's ISA
    // precondition.
    unsafe { scan_dynamic::<Avx2<true>, O>(codes, row_offsets, cover, sparse_row_mapping, out) };
}

/// Direct generic-kernel entry retained for architecture correctness tests.
#[cfg(test)]
#[target_feature(enable = "avx2")]
pub(in crate::search::substring::prefilter) fn scan_avx2<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    cover: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    scan_avx2_generic(codes, row_offsets, cover, sparse_row_mapping, out);
}
