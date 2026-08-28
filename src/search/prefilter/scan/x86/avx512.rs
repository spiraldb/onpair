// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! AVX-512BW prefilter execution.

use super::super::sink::{LaneMask, RowSink, scan_tail};
use crate::core::offset::Offset;
use crate::core::types::{Token, TokenRange};
use crate::search::prefilter::cover::ProbeCover;

const SUPERBLOCK: usize = 512;

#[inline(never)]
fn consume_mask<O: Offset>(base: usize, mask: u64, sink: &mut RowSink<'_, O>) {
    sink.mark_mask(base, LaneMask::from_bits(mask));
}

/// Shared retained-mask control for the fixed-shape leaves.
#[inline(always)]
fn scan_masks<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    cover: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
    mut mask_at: impl FnMut(usize) -> u32,
) {
    let total = codes.len();
    let mut sink = RowSink::new(row_offsets, out, sparse_row_mapping);
    let mut base = 0;

    while base + SUPERBLOCK <= total {
        let mut masks = [0u64; SUPERBLOCK / 64];
        let mut any = 0u64;
        for (block, mask) in masks.iter_mut().enumerate() {
            let lo = u64::from(mask_at(base + block * 64));
            let hi = u64::from(mask_at(base + block * 64 + 32));
            *mask = lo | (hi << 32);
            any |= *mask;
        }
        if any != 0 {
            for (block, &mask) in masks.iter().enumerate() {
                if mask != 0 {
                    consume_mask(base + block * 64, mask, &mut sink);
                }
            }
        }
        base += SUPERBLOCK;
    }

    while base + 32 <= total {
        let mask = u64::from(mask_at(base));
        if mask != 0 {
            consume_mask(base, mask, &mut sink);
        }
        base += 32;
    }
    scan_tail(codes, cover, base, &mut sink);
}

/// AVX-512BW: 32 lanes with native unsigned comparisons and mask output.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub(in crate::search::prefilter) fn scan_avx512<O: Offset>(
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

/// Const-shape leaf for the six cover shapes that dominate the workload.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline(never)]
pub(in crate::search::prefilter::scan) fn scan_avx512_fixed<
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
    use core::arch::x86_64::{
        __m512i, _mm512_loadu_si512, _mm512_mask_cmpgt_epu16_mask, _mm512_mask_cmpneq_epu16_mask,
        _mm512_set1_epi16, _mm512_sub_epi16,
    };

    debug_assert_eq!((cover.points.len(), cover.ranges.len()), (POINTS, RANGES));
    let points: [__m512i; POINTS] =
        std::array::from_fn(|index| _mm512_set1_epi16(cover.points[index] as i16));
    let ranges: [(__m512i, __m512i); RANGES] = std::array::from_fn(|index| {
        let TokenRange { begin, last } = cover.ranges[index];
        (
            _mm512_set1_epi16(begin as i16),
            _mm512_set1_epi16(last.wrapping_sub(begin) as i16),
        )
    });
    let ptr = codes.as_ptr();

    scan_masks(
        codes,
        row_offsets,
        cover,
        sparse_row_mapping,
        out,
        |index| unsafe {
            let value = _mm512_loadu_si512(ptr.add(index).cast());
            let mut miss = u32::MAX;
            for &point in &points {
                miss = _mm512_mask_cmpneq_epu16_mask(miss, value, point);
            }
            for &(begin, span) in &ranges {
                let delta = _mm512_sub_epi16(value, begin);
                miss = _mm512_mask_cmpgt_epu16_mask(miss, delta, span);
            }
            !miss
        },
    );
}
