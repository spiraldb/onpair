// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! AVX2 prefilter kernels.

use super::super::sink::{LaneMask, RowSink, mark_block, scan_tail};
use super::super::table;
use crate::core::offset::Offset;
use crate::core::types::{Token, TokenRange};
use crate::search::prefilter::cover::ProbeCover;

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
            sink.mark_mask(
                i,
                LaneMask::from_bits(compact_avx2_byte_hits(h0, h1, h2, h3, low_byte_mask)),
            );
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
            sink.mark_mask(
                i,
                LaneMask::from_bits(compact_avx2_byte_hits(h0, h1, h2, h3, low_byte_mask)),
            );
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
pub(in crate::search::prefilter::scan) fn scan_avx2_nibble_points<O: Offset>(
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
                sink.mark_mask(i, LaneMask::from_bits(compact_avx2_masks(m0, m1, m2, m3)));
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
pub(in crate::search::prefilter::scan) fn scan_avx2_one_point<O: Offset>(
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
pub(in crate::search::prefilter::scan) fn scan_avx2_one_range<O: Offset>(
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
pub(in crate::search::prefilter::scan) fn scan_avx2_fixed<
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

/// Exact AVX2 gather from a compact one-bit-per-token membership table.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(in crate::search::prefilter::scan) fn scan_avx2_gather<O: Offset>(
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
        table::pack_membership_dense(pf)
    } else {
        table::pack_membership(pf)
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
            sink.mark_mask(i, LaneMask::from_bits(lanes));
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
pub(in crate::search::prefilter::scan) fn scan_avx2_few<O: Offset, const COMPACT_HITS: bool>(
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
                sink.mark_mask(
                    i,
                    LaneMask::from_bits(u64::from(compact_two_avx2_masks(acc0, acc1))),
                );
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
pub(in crate::search::prefilter) fn scan_avx2<O: Offset>(
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
