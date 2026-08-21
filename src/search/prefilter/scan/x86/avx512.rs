// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! AVX-512BW prefilter execution.

use super::super::policy::{BAIL_MIN_SEEN_DIVISOR, BAIL_RATIO_DEN, BAIL_RATIO_NUM};
use super::super::sink::{LaneMask, RowSink, scan_tail};
use crate::core::offset::Offset;
use crate::core::types::{Token, TokenRange};
use crate::search::prefilter::cover::ProbeCover;

const BLOCK_CODES: usize = 512;

/// First collapse each 512-code block into one summary bit, then regenerate
/// exact masks only for live blocks. The second pass is exact: rows are emitted
/// only when one of their codes belongs to the probe cover.
///
/// Keeping summary construction separate from sparse row mapping lets the
/// first pass stream at the code-buffer bandwidth ceiling. At one bit per 512
/// codes, its temporary storage is one byte per 4096 input codes.
#[inline(always)]
fn scan_block_masks<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    out: &mut Vec<usize>,
    mut mask_at: impl FnMut(usize) -> u32,
) {
    const STACK_SUMMARY_WORDS: usize = 4096;
    let full_blocks = codes.len() / BLOCK_CODES;
    let summary_words = codes
        .len()
        .div_ceil(BLOCK_CODES)
        .div_ceil(u64::BITS as usize);
    let mut stack_summary = [0u64; STACK_SUMMARY_WORDS];
    let mut heap_summary = Vec::new();
    let summary: &mut [u64] = if summary_words <= STACK_SUMMARY_WORDS {
        &mut stack_summary[..summary_words]
    } else {
        heap_summary.resize(summary_words, 0);
        &mut heap_summary
    };
    let mut block = 0usize;
    for slot in summary.iter_mut() {
        let blocks_in_word = (full_blocks - block).min(64);
        let mut word = 0u64;
        for bit in 0..blocks_in_word {
            let base = block * BLOCK_CODES;
            let mut any = 0u32;
            for offset in (0..BLOCK_CODES).step_by(32) {
                any |= mask_at(base + offset);
            }
            word |= u64::from(any != 0) << bit;
            block += 1;
        }
        *slot = word;
    }
    let tail = full_blocks * BLOCK_CODES;
    if tail != codes.len() && codes[tail..].iter().any(|&code| pf.table[code as usize]) {
        if full_blocks.is_multiple_of(64) {
            summary[full_blocks / 64] = 1;
        } else {
            summary[full_blocks / 64] |= 1 << (full_blocks % 64);
        }
    }

    let mut sink = RowSink::new(row_offsets, out, true);
    for (word_index, &bits) in summary.iter().enumerate() {
        let mut live = bits;
        while live != 0 {
            let live_block = word_index * 64 + live.trailing_zeros() as usize;
            let begin = live_block * BLOCK_CODES;
            let end = (begin + BLOCK_CODES).min(codes.len());
            let mut index = begin;
            while index + 64 <= end {
                let lanes = u64::from(mask_at(index)) | (u64::from(mask_at(index + 32)) << 32);
                if lanes != 0 {
                    sink.mark_mask(index, LaneMask::from_bits(lanes));
                }
                index += 64;
            }
            while index + 32 <= end {
                let lanes = mask_at(index);
                if lanes != 0 {
                    sink.mark_mask(index, LaneMask::from_bits(u64::from(lanes)));
                }
                index += 32;
            }
            scan_tail(&codes[..end], pf, index, &mut sink);
            live &= live - 1;
        }
    }
}

/// Sparse one-bit-per-superblock gate for a dynamic cover shape.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub(in crate::search::prefilter) fn scan_avx512_sparse<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    out: &mut Vec<usize>,
) {
    use core::arch::x86_64::{
        __m512i, _mm512_cmpeq_epu16_mask, _mm512_cmple_epu16_mask, _mm512_loadu_si512,
        _mm512_set1_epi16, _mm512_sub_epi16,
    };

    let points: Vec<__m512i> = pf
        .points
        .iter()
        .map(|&point| _mm512_set1_epi16(point as i16))
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
    let base = codes.as_ptr();
    scan_block_masks(codes, row_offsets, pf, out, |offset| unsafe {
        let value = _mm512_loadu_si512(base.add(offset).cast());
        let mut any = 0u32;
        for &point in &points {
            any |= _mm512_cmpeq_epu16_mask(value, point);
        }
        for &(lo, span) in &ranges {
            any |= _mm512_cmple_epu16_mask(_mm512_sub_epi16(value, lo), span);
        }
        any
    });
}

/// Fixed-shape sparse superblock gate. Const lengths let LLVM keep broadcasts
/// in registers and combine masks with `kord`/`kortest` instead of `kmov` after
/// every comparison.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub(in crate::search::prefilter) fn scan_avx512_sparse_fixed<
    O: Offset,
    const POINTS: usize,
    const RANGES: usize,
>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    out: &mut Vec<usize>,
) {
    use core::arch::x86_64::{
        __m512i, _mm512_loadu_si512, _mm512_mask_cmpgt_epu16_mask, _mm512_mask_cmpneq_epu16_mask,
        _mm512_set1_epi16, _mm512_sub_epi16,
    };

    debug_assert_eq!((pf.points.len(), pf.ranges.len()), (POINTS, RANGES));
    let points: [__m512i; POINTS] = std::array::from_fn(|i| _mm512_set1_epi16(pf.points[i] as i16));
    let ranges: [(__m512i, __m512i); RANGES] = std::array::from_fn(|i| {
        let TokenRange { begin, last } = pf.ranges[i];
        (
            _mm512_set1_epi16(begin as i16),
            _mm512_set1_epi16(last.wrapping_sub(begin) as i16),
        )
    });
    let base = codes.as_ptr();
    scan_block_masks(codes, row_offsets, pf, out, |offset| unsafe {
        let value = _mm512_loadu_si512(base.add(offset).cast());
        // Keep the complement: lanes remain set only while every predicate
        // misses. This exposes the independent range subtracts to LLVM and
        // shortens the generated mask-reduction dependency chain.
        let mut miss = u32::MAX;
        for &point in &points {
            miss = _mm512_mask_cmpneq_epu16_mask(miss, value, point);
        }
        for &(lo, span) in &ranges {
            miss = _mm512_mask_cmpgt_epu16_mask(miss, _mm512_sub_epi16(value, lo), span);
        }
        !miss
    });
}

/// Execute the common retained-mask and row-materialization loop. Callers
/// specialize only the operation that produces one 32-lane mask.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn scan_masks<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    bail: bool,
    out: &mut Vec<usize>,
    mut mask_at: impl FnMut(usize) -> u32,
) {
    const SUPERBLOCK: usize = 256;

    let total = codes.len();
    let mut sink = RowSink::new(row_offsets, out, sparse_row_mapping);
    let rows = row_offsets.len().saturating_sub(1);
    let min_seen = (rows / BAIL_MIN_SEEN_DIVISOR).max(1);
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
            if bail {
                let seen = sink.rows_decided();
                if seen >= min_seen && sink.appended() * BAIL_RATIO_DEN >= seen * BAIL_RATIO_NUM {
                    sink.append_remaining();
                    return;
                }
            }
        }
        i += SUPERBLOCK;
    }
    while i + 32 <= total {
        let mut mask = mask_at(i);
        while mask != 0 {
            let lane = mask.trailing_zeros() as usize;
            sink.hit(i + lane);
            mask &= mask - 1;
        }
        i += 32;
    }
    scan_tail(codes, pf, i, &mut sink);
}

/// AVX-512BW: 32 lanes with native unsigned comparisons and mask output.
///
/// The scan walks superblocks of 256 codes: eight vector compares whose
/// 32-lane hit masks are collapsed pairwise into four 64-lane bitsets and
/// OR-ed into a single gate, so the common all-miss superblock costs one
/// branch instead of eight. A live superblock emits rows from the bitsets
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
    bail: bool,
    out: &mut Vec<usize>,
) {
    use core::arch::x86_64::{
        __m512i, _mm512_cmpeq_epu16_mask, _mm512_cmple_epu16_mask, _mm512_loadu_si512,
        _mm512_set1_epi16, _mm512_sub_epi16,
    };

    let base = codes.as_ptr();

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
    scan_masks(
        codes,
        row_offsets,
        pf,
        sparse_row_mapping,
        bail,
        out,
        |i| unsafe {
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
        },
    );
}

/// Fixed one-point leaf: the hot loop has no dynamic probe-loop branch.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub(in crate::search::prefilter) fn scan_avx512_one_point<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    bail: bool,
    out: &mut Vec<usize>,
) {
    use core::arch::x86_64::{_mm512_cmpeq_epu16_mask, _mm512_loadu_si512, _mm512_set1_epi16};

    debug_assert_eq!((pf.points.len(), pf.ranges.len()), (1, 0));
    let base = codes.as_ptr();
    let point = _mm512_set1_epi16(pf.points[0] as i16);
    scan_masks(codes, row_offsets, pf, sparse_row_mapping, bail, out, |i| {
        // SAFETY: scan_masks requests only complete 32-code vectors.
        unsafe { _mm512_cmpeq_epu16_mask(_mm512_loadu_si512(base.add(i).cast()), point) }
    });
}

/// Fixed one-range leaf: subtract plus unsigned compare, with no probe loop.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub(in crate::search::prefilter) fn scan_avx512_one_range<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    bail: bool,
    out: &mut Vec<usize>,
) {
    use core::arch::x86_64::{
        _mm512_cmple_epu16_mask, _mm512_loadu_si512, _mm512_set1_epi16, _mm512_sub_epi16,
    };

    debug_assert_eq!((pf.points.len(), pf.ranges.len()), (0, 1));
    let base = codes.as_ptr();
    let TokenRange { begin, last } = pf.ranges[0];
    let lo = _mm512_set1_epi16(begin as i16);
    let span = _mm512_set1_epi16(last.wrapping_sub(begin) as i16);
    scan_masks(codes, row_offsets, pf, sparse_row_mapping, bail, out, |i| {
        // SAFETY: scan_masks requests only complete 32-code vectors.
        unsafe {
            let value = _mm512_loadu_si512(base.add(i).cast());
            _mm512_cmple_epu16_mask(_mm512_sub_epi16(value, lo), span)
        }
    });
}
