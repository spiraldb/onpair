// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! AVX-512BW prefilter execution.

use super::super::sink::{LaneMask, RowSink, scan_tail};
use crate::core::offset::Offset;
use crate::core::types::{Token, TokenRange};
use crate::search::prefilter::cover::ProbeCover;

/// AVX-512BW: 32 lanes with native unsigned comparisons and mask output.
///
/// The scan walks superblocks of 512 codes: sixteen vector compares whose
/// 32-lane hit masks are collapsed pairwise into eight 64-lane bitsets and
/// OR-ed into a single gate, so the common all-miss superblock costs one
/// branch instead of sixteen. A live superblock emits rows from the bitsets
/// it already holds — positions are a free by-product of the mask compare,
/// nothing is rescanned. Each inclusive range costs one subtract plus one
/// unsigned compare (`code - begin <=ᵤ last - begin`), halving the
/// mask-port pressure of the compare-pair form.
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
        __m512i, _mm512_cmpeq_epu16_mask, _mm512_cmple_epu16_mask, _mm512_loadu_si512,
        _mm512_set1_epi16, _mm512_sub_epi16,
    };

    /// Codes per superblock: one gate branch per this many codes.
    const SUPERBLOCK: usize = 512;

    let total = codes.len();
    let base = codes.as_ptr();
    let mut sink = RowSink::new(row_offsets, out, sparse_row_mapping);

    // Broadcast every probe once, so the per-vector loop issues plain loads
    // instead of `vpbroadcastw` µops that would contend with the compares for
    // the mask port.
    let points: Vec<__m512i> = pf
        .points
        .iter()
        .map(|&p| _mm512_set1_epi16(p as i16))
        .collect();
    let ranges: Vec<(__m512i, __m512i)> = pf
        .ranges
        .iter()
        .map(|&TokenRange { begin, last }| {
            (
                _mm512_set1_epi16(begin as i16),
                _mm512_set1_epi16(last.wrapping_sub(begin) as i16),
            )
        })
        .collect();

    // The closure inherits the enclosing target features (RFC 2396).
    // SAFETY (for both loops below): every `mask_at(i)` has `i + 32 <= total`,
    // and the caller established AVX-512F/BW.
    let mask_at = |i: usize| -> u32 {
        unsafe {
            let v = _mm512_loadu_si512(base.add(i).cast());
            let mut m: u32 = 0;
            for &point in &points {
                m |= _mm512_cmpeq_epu16_mask(v, point);
            }
            for &(lo, span) in &ranges {
                let delta = _mm512_sub_epi16(v, lo);
                m |= _mm512_cmple_epu16_mask(delta, span);
            }
            m
        }
    };

    let mut i = 0usize;
    while i + SUPERBLOCK <= total {
        let mut lanes = [0u64; SUPERBLOCK / 64];
        let mut any = 0u64;
        for (k, slot) in lanes.iter_mut().enumerate() {
            let lo = mask_at(i + k * 64) as u64;
            let hi = mask_at(i + k * 64 + 32) as u64;
            let pair = lo | (hi << 32);
            *slot = pair;
            any |= pair;
        }
        if any != 0 {
            for (k, &pair) in lanes.iter().enumerate() {
                if pair != 0 {
                    sink.mark_mask(i + k * 64, LaneMask::from_bits(pair));
                }
            }
        }
        i += SUPERBLOCK;
    }
    // Whole vectors after the last full superblock.
    while i + 32 <= total {
        let mut m = mask_at(i);
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
