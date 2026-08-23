// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![allow(clippy::expect_used, clippy::unwrap_used)]

//! ClickBench runner comparing full exact scans with prefilter-only and
//! prefilter-plus-verification pipelines.

use std::collections::HashSet;
use std::fs;
use std::fs::File;
use std::hint::black_box;
use std::mem::{MaybeUninit, size_of, size_of_val};
use std::path::{Path, PathBuf};
use std::time::Instant;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use memchr::memmem::Finder;
use onpair::search::index::build_token_frequency_index;
use onpair::search::{
    BytesVerifier, ContainsTable, ProbeCover, analyze_prefilter, contains, prefilter_candidates,
};
use onpair::{
    Column, CompactDictionary, CompactDictionaryView, Config, DECODE_PADDING, DictionaryView,
    MAX_TOKEN_SIZE, MaxDictBits, OwnedDictionaryStorage, Threshold, TokenRange, compress,
    decode_into,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

const DEFAULT_PARQUET: &str = ".benchmark-data/clickbench-url-1m.parquet";
const DEFAULT_DUMP: &str = ".benchmark-data/onpair-clickbench-16";
const DEFAULT_REPS: usize = 9;
const DEFAULT_SUMMARY_CODES: usize = 256;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn summarize_full_blocks<const BLOCK_CODES: usize>(
    codes: &[u16],
    summary: &mut [u64],
    mut mask_at: impl FnMut(usize) -> u32,
) -> usize {
    let full_blocks = codes.len() / BLOCK_CODES;
    let mut block = 0usize;
    for slot in summary.iter_mut() {
        let blocks_in_word = (full_blocks - block).min(64);
        let mut word = 0u64;
        for bit in 0..blocks_in_word {
            let mut any = 0u32;
            for vector in 0..BLOCK_CODES / 32 {
                any |= mask_at(block * BLOCK_CODES + vector * 32);
            }
            word |= u64::from(any != 0) << bit;
            block += 1;
        }
        *slot = word;
    }
    full_blocks
}

#[cold]
#[inline(never)]
fn set_live_summary_bit(word: &mut u64, bit: usize) {
    *word |= 1u64 << bit;
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn summarize_full_blocks_sparse_branch<const BLOCK_CODES: usize>(
    codes: &[u16],
    summary: &mut [u64],
    mut mask_at: impl FnMut(usize) -> u32,
) -> usize {
    let full_blocks = codes.len() / BLOCK_CODES;
    let mut block = 0usize;
    for slot in summary.iter_mut() {
        let blocks_in_word = (full_blocks - block).min(64);
        let mut word = 0u64;
        for bit in 0..blocks_in_word {
            let mut any = 0u32;
            for vector in 0..BLOCK_CODES / 32 {
                any |= mask_at(block * BLOCK_CODES + vector * 32);
            }
            if any != 0 {
                set_live_summary_bit(&mut word, bit);
            }
            block += 1;
        }
        *slot = word;
    }
    full_blocks
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn summarize_tail<const BLOCK_CODES: usize>(
    codes: &[u16],
    cover: &ProbeCover,
    summary: &mut [u64],
    full_blocks: usize,
) {
    if full_blocks * BLOCK_CODES != codes.len() {
        let live = codes[full_blocks * BLOCK_CODES..].iter().any(|&code| {
            cover.points().contains(&code)
                || cover.ranges().iter().any(|range| range.contains(code))
        });
        summary[full_blocks / 64] |= u64::from(live) << (full_blocks % 64);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
fn summarize_superblocks<const BLOCK_CODES: usize>(
    codes: &[u16],
    cover: &ProbeCover,
    summary: &mut [u64],
) {
    use core::arch::x86_64::{
        __m512i, _mm512_cmpeq_epu16_mask, _mm512_cmple_epu16_mask, _mm512_loadu_si512,
        _mm512_set1_epi16, _mm512_sub_epi16,
    };

    let points: Vec<__m512i> = cover
        .points()
        .iter()
        .map(|&point| _mm512_set1_epi16(point as i16))
        .collect();
    let ranges: Vec<(__m512i, __m512i)> = cover
        .ranges()
        .iter()
        .map(|&TokenRange { begin, last }| {
            (
                _mm512_set1_epi16(begin as i16),
                _mm512_set1_epi16(last.wrapping_sub(begin) as i16),
            )
        })
        .collect();
    let base = codes.as_ptr();
    let full_blocks = summarize_full_blocks::<BLOCK_CODES>(codes, summary, |offset| unsafe {
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
    summarize_tail::<BLOCK_CODES>(codes, cover, summary, full_blocks);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
fn summarize_superblocks_fixed<
    const BLOCK_CODES: usize,
    const POINTS: usize,
    const RANGES: usize,
>(
    codes: &[u16],
    cover: &ProbeCover,
    summary: &mut [u64],
) {
    use core::arch::x86_64::{
        __m512i, _mm512_cmpeq_epu16_mask, _mm512_cmple_epu16_mask, _mm512_loadu_si512,
        _mm512_set1_epi16, _mm512_sub_epi16,
    };

    debug_assert_eq!(
        (cover.points().len(), cover.ranges().len()),
        (POINTS, RANGES)
    );
    let points: [__m512i; POINTS] =
        std::array::from_fn(|i| _mm512_set1_epi16(cover.points()[i] as i16));
    let ranges: [(__m512i, __m512i); RANGES] = std::array::from_fn(|i| {
        let TokenRange { begin, last } = cover.ranges()[i];
        (
            _mm512_set1_epi16(begin as i16),
            _mm512_set1_epi16(last.wrapping_sub(begin) as i16),
        )
    });
    let base = codes.as_ptr();
    let full_blocks = summarize_full_blocks::<BLOCK_CODES>(codes, summary, |offset| unsafe {
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
    summarize_tail::<BLOCK_CODES>(codes, cover, summary, full_blocks);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
fn summarize_superblocks_fixed_miss<
    const BLOCK_CODES: usize,
    const POINTS: usize,
    const RANGES: usize,
>(
    codes: &[u16],
    cover: &ProbeCover,
    summary: &mut [u64],
) {
    use core::arch::x86_64::{
        __m512i, _mm512_loadu_si512, _mm512_mask_cmpgt_epu16_mask, _mm512_mask_cmpneq_epu16_mask,
        _mm512_set1_epi16, _mm512_sub_epi16,
    };

    let points: [__m512i; POINTS] =
        std::array::from_fn(|i| _mm512_set1_epi16(cover.points()[i] as i16));
    let ranges: [(__m512i, __m512i); RANGES] = std::array::from_fn(|i| {
        let TokenRange { begin, last } = cover.ranges()[i];
        (
            _mm512_set1_epi16(begin as i16),
            _mm512_set1_epi16(last.wrapping_sub(begin) as i16),
        )
    });
    let base = codes.as_ptr();
    let full_blocks = summarize_full_blocks::<BLOCK_CODES>(codes, summary, |offset| unsafe {
        let value = _mm512_loadu_si512(base.add(offset).cast());
        let mut miss = u32::MAX;
        for &point in &points {
            miss = _mm512_mask_cmpneq_epu16_mask(miss, value, point);
        }
        for &(lo, span) in &ranges {
            miss = _mm512_mask_cmpgt_epu16_mask(miss, _mm512_sub_epi16(value, lo), span);
        }
        !miss
    });
    summarize_tail::<BLOCK_CODES>(codes, cover, summary, full_blocks);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
fn summarize_superblocks_fixed_miss_branch<
    const BLOCK_CODES: usize,
    const POINTS: usize,
    const RANGES: usize,
>(
    codes: &[u16],
    cover: &ProbeCover,
    summary: &mut [u64],
) {
    use core::arch::x86_64::{
        __m512i, _mm512_loadu_si512, _mm512_mask_cmpgt_epu16_mask, _mm512_mask_cmpneq_epu16_mask,
        _mm512_set1_epi16, _mm512_sub_epi16,
    };

    let points: [__m512i; POINTS] =
        std::array::from_fn(|i| _mm512_set1_epi16(cover.points()[i] as i16));
    let ranges: [(__m512i, __m512i); RANGES] = std::array::from_fn(|i| {
        let TokenRange { begin, last } = cover.ranges()[i];
        (
            _mm512_set1_epi16(begin as i16),
            _mm512_set1_epi16(last.wrapping_sub(begin) as i16),
        )
    });
    let base = codes.as_ptr();
    let full_blocks =
        summarize_full_blocks_sparse_branch::<BLOCK_CODES>(codes, summary, |offset| unsafe {
            let value = _mm512_loadu_si512(base.add(offset).cast());
            let mut miss = u32::MAX;
            for &point in &points {
                miss = _mm512_mask_cmpneq_epu16_mask(miss, value, point);
            }
            for &(lo, span) in &ranges {
                miss = _mm512_mask_cmpgt_epu16_mask(miss, _mm512_sub_epi16(value, lo), span);
            }
            !miss
        });
    summarize_tail::<BLOCK_CODES>(codes, cover, summary, full_blocks);
}

#[inline(never)]
fn summarize_superblocks_scalar<const BLOCK_CODES: usize>(
    codes: &[u16],
    cover: &ProbeCover,
    summary: &mut [u64],
) {
    for (word_index, slot) in summary.iter_mut().enumerate() {
        let mut word = 0u64;
        for bit in 0..64 {
            let begin = (word_index * 64 + bit) * BLOCK_CODES;
            if begin >= codes.len() {
                break;
            }
            let end = (begin + BLOCK_CODES).min(codes.len());
            if codes[begin..end].iter().any(|&code| {
                cover.points().contains(&code)
                    || cover.ranges().iter().any(|range| range.contains(code))
            }) {
                word |= 1u64 << bit;
            }
        }
        *slot = word;
    }
}

#[inline(never)]
fn summarize_superblocks_autovec<
    const BLOCK_CODES: usize,
    const POINTS: usize,
    const RANGES: usize,
>(
    codes: &[u16],
    cover: &ProbeCover,
    summary: &mut [u64],
) {
    let points: [u16; POINTS] = std::array::from_fn(|i| cover.points()[i]);
    let lows: [u16; RANGES] = std::array::from_fn(|i| cover.ranges()[i].begin);
    let spans: [u16; RANGES] = std::array::from_fn(|i| {
        let range = cover.ranges()[i];
        range.last.wrapping_sub(range.begin)
    });
    let full_blocks = codes.len() / BLOCK_CODES;
    let mut block = 0usize;
    for slot in summary.iter_mut() {
        let mut word = 0u64;
        let blocks_in_word = (full_blocks - block).min(64);
        for bit in 0..blocks_in_word {
            let begin = block * BLOCK_CODES;
            let any = codes[begin..begin + BLOCK_CODES]
                .iter()
                .fold(false, |mut any, &code| {
                    for &point in &points {
                        any |= code == point;
                    }
                    for i in 0..RANGES {
                        any |= code.wrapping_sub(lows[i]) <= spans[i];
                    }
                    any
                });
            word |= u64::from(any) << bit;
            block += 1;
        }
        *slot = word;
    }
    let tail = full_blocks * BLOCK_CODES;
    if tail != codes.len() {
        let any = codes[tail..].iter().any(|&code| {
            points.contains(&code) || (0..RANGES).any(|i| code.wrapping_sub(lows[i]) <= spans[i])
        });
        summary[full_blocks / 64] |= u64::from(any) << (full_blocks % 64);
    }
}

#[inline(never)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
fn summarize_superblocks_autovec_miss<
    const BLOCK_CODES: usize,
    const POINTS: usize,
    const RANGES: usize,
>(
    codes: &[u16],
    cover: &ProbeCover,
    summary: &mut [u64],
) {
    let points: [u16; POINTS] = std::array::from_fn(|i| cover.points()[i]);
    let lows: [u16; RANGES] = std::array::from_fn(|i| cover.ranges()[i].begin);
    let spans: [u16; RANGES] = std::array::from_fn(|i| {
        let range = cover.ranges()[i];
        range.last.wrapping_sub(range.begin)
    });
    let full_blocks = codes.len() / BLOCK_CODES;
    let mut block = 0usize;
    for slot in summary.iter_mut() {
        let mut word = 0u64;
        let blocks_in_word = (full_blocks - block).min(64);
        for bit in 0..blocks_in_word {
            let begin = block * BLOCK_CODES;
            let all_miss =
                codes[begin..begin + BLOCK_CODES]
                    .iter()
                    .fold(true, |mut miss, &code| {
                        for &point in &points {
                            miss &= code != point;
                        }
                        for i in 0..RANGES {
                            miss &= code.wrapping_sub(lows[i]) > spans[i];
                        }
                        miss
                    });
            word |= u64::from(!all_miss) << bit;
            block += 1;
        }
        *slot = word;
    }
    let tail = full_blocks * BLOCK_CODES;
    if tail != codes.len() {
        let any = codes[tail..].iter().any(|&code| {
            points.contains(&code) || (0..RANGES).any(|i| code.wrapping_sub(lows[i]) <= spans[i])
        });
        summary[full_blocks / 64] |= u64::from(any) << (full_blocks % 64);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn summarize_superblocks_avx2_p2r2<const BLOCK_CODES: usize>(
    codes: &[u16],
    cover: &ProbeCover,
    summary: &mut [u64],
) {
    use core::arch::x86_64::{
        __m256i, _mm256_andnot_si256, _mm256_cmpeq_epi16, _mm256_cmpgt_epi16, _mm256_loadu_si256,
        _mm256_or_si256, _mm256_set1_epi16, _mm256_setzero_si256, _mm256_testz_si256,
        _mm256_xor_si256,
    };

    debug_assert_eq!((cover.points().len(), cover.ranges().len()), (2, 2));
    debug_assert!(BLOCK_CODES.is_multiple_of(16));
    let zero = _mm256_setzero_si256();
    let ones = _mm256_set1_epi16(-1);
    let bias = _mm256_set1_epi16(i16::MIN);
    let points: [__m256i; 2] = std::array::from_fn(|i| _mm256_set1_epi16(cover.points()[i] as i16));
    let ranges: [(__m256i, __m256i); 2] = std::array::from_fn(|i| {
        let range = cover.ranges()[i];
        (
            _mm256_xor_si256(_mm256_set1_epi16(range.begin as i16), bias),
            _mm256_xor_si256(_mm256_set1_epi16(range.last as i16), bias),
        )
    });
    let base = codes.as_ptr();
    let full_blocks = codes.len() / BLOCK_CODES;
    let mut block = 0usize;
    for slot in summary.iter_mut() {
        let mut word = 0u64;
        for bit in 0..(full_blocks - block).min(64) {
            let mut any = zero;
            for offset in (0..BLOCK_CODES).step_by(16) {
                // SAFETY: the loop visits complete vectors inside a full block.
                let value =
                    unsafe { _mm256_loadu_si256(base.add(block * BLOCK_CODES + offset).cast()) };
                let mut hits = zero;
                for &point in &points {
                    hits = _mm256_or_si256(hits, _mm256_cmpeq_epi16(value, point));
                }
                let biased = _mm256_xor_si256(value, bias);
                for &(lo, hi) in &ranges {
                    let outside = _mm256_or_si256(
                        _mm256_cmpgt_epi16(lo, biased),
                        _mm256_cmpgt_epi16(biased, hi),
                    );
                    hits = _mm256_or_si256(hits, _mm256_andnot_si256(outside, ones));
                }
                any = _mm256_or_si256(any, hits);
            }
            word |= u64::from(_mm256_testz_si256(any, any) == 0) << bit;
            block += 1;
        }
        *slot = word;
    }
    summarize_tail::<BLOCK_CODES>(codes, cover, summary, full_blocks);
}

fn superblock_candidates<const BLOCK_CODES: usize>(
    summary: &[u64],
    row_offsets: &[u64],
    out: &mut Vec<usize>,
) {
    out.clear();
    let rows = row_offsets.len().saturating_sub(1);
    let mut next_row = 0usize;
    for (word_index, &word) in summary.iter().enumerate() {
        let mut live = word;
        while live != 0 {
            let block = word_index * 64 + live.trailing_zeros() as usize;
            let begin = block * BLOCK_CODES;
            let end = ((block + 1) * BLOCK_CODES).min(row_offsets[rows] as usize);
            let first = row_offsets.partition_point(|&offset| offset as usize <= begin) - 1;
            let last = row_offsets.partition_point(|&offset| (offset as usize) < end);
            let first = first.max(next_row);
            if first < last {
                out.extend(first..last);
                next_row = last;
            }
            live &= live - 1;
        }
    }
}

struct OriginalRowSink<'a> {
    row_offsets: &'a [u64],
    out: &'a mut Vec<usize>,
    row: usize,
    row_end: usize,
}

impl OriginalRowSink<'_> {
    #[inline]
    fn hit(&mut self, code_index: usize) {
        if code_index < self.row_end {
            return;
        }
        if code_index.saturating_sub(self.row_end) >= 128 {
            let suffix = &self.row_offsets[self.row + 1..];
            self.row += suffix.partition_point(|&offset| offset as usize <= code_index);
        } else {
            while self.row + 1 < self.row_offsets.len()
                && self.row_offsets[self.row + 1] as usize <= code_index
            {
                self.row += 1;
            }
        }
        self.out.push(self.row);
        self.row_end = self.row_offsets[self.row + 1] as usize;
    }

    #[inline]
    fn mark_mask(&mut self, base: usize, mut lanes: u64) {
        loop {
            let consumed = self.row_end.saturating_sub(base);
            if consumed >= 64 {
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

/// Original AVX2 four-vector exact-mask loop: branch once per 64 codes, then
/// map retained hit lanes to rows.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn original_avx2_candidates(
    codes: &[u16],
    row_offsets: &[u64],
    cover: &ProbeCover,
    out: &mut Vec<usize>,
) {
    use core::arch::x86_64::{
        __m256i, _mm256_andnot_si256, _mm256_cmpeq_epi16, _mm256_cmpgt_epi16, _mm256_loadu_si256,
        _mm256_movemask_epi8, _mm256_or_si256, _mm256_packs_epi16, _mm256_permute4x64_epi64,
        _mm256_set1_epi16, _mm256_setzero_si256, _mm256_xor_si256,
    };

    out.clear();
    let zero = _mm256_setzero_si256();
    let bias = _mm256_set1_epi16(i16::MIN);
    let ones = _mm256_set1_epi16(-1);
    let points: Vec<__m256i> = cover
        .points()
        .iter()
        .map(|&point| _mm256_set1_epi16(point as i16))
        .collect();
    let ranges: Vec<(__m256i, __m256i)> = cover
        .ranges()
        .iter()
        .map(|&TokenRange { begin, last }| {
            (
                _mm256_xor_si256(_mm256_set1_epi16(begin as i16), bias),
                _mm256_xor_si256(_mm256_set1_epi16(last as i16), bias),
            )
        })
        .collect();
    let matching_mask = |value: __m256i| {
        let mut hits = zero;
        for &point in &points {
            hits = _mm256_or_si256(hits, _mm256_cmpeq_epi16(value, point));
        }
        let code = _mm256_xor_si256(value, bias);
        for &(lo, hi) in &ranges {
            let outside =
                _mm256_or_si256(_mm256_cmpgt_epi16(lo, code), _mm256_cmpgt_epi16(code, hi));
            hits = _mm256_or_si256(hits, _mm256_andnot_si256(outside, ones));
        }
        hits
    };
    let base = codes.as_ptr();
    let mut sink = OriginalRowSink {
        row_offsets,
        out,
        row: 0,
        row_end: 0,
    };
    let mut index = 0usize;
    while index + 64 <= codes.len() {
        let masks: [__m256i; 4] = std::array::from_fn(|chunk| unsafe {
            matching_mask(_mm256_loadu_si256(base.add(index + chunk * 16).cast()))
        });
        let lanes01 = _mm256_movemask_epi8(_mm256_permute4x64_epi64::<0xd8>(_mm256_packs_epi16(
            masks[0], masks[1],
        ))) as u32 as u64;
        let lanes23 = _mm256_movemask_epi8(_mm256_permute4x64_epi64::<0xd8>(_mm256_packs_epi16(
            masks[2], masks[3],
        ))) as u32 as u64;
        let lanes = lanes01 | (lanes23 << 32);
        if lanes != 0 {
            sink.mark_mask(index, lanes);
        }
        index += 64;
    }
    for (tail, &code) in codes[index..].iter().enumerate() {
        if cover.points().contains(&code) || cover.ranges().iter().any(|range| range.contains(code))
        {
            sink.hit(index + tail);
        }
    }
}

/// Exact copy of the original branch kernel's 512-code retained-mask loop,
/// specialized only to the benchmark's `u64` row offsets and with bailout
/// omitted (it is disabled for selective covers).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
fn original_avx512_candidates_dynamic(
    codes: &[u16],
    row_offsets: &[u64],
    cover: &ProbeCover,
    out: &mut Vec<usize>,
) {
    use core::arch::x86_64::{
        __m512i, _mm512_cmpeq_epu16_mask, _mm512_cmple_epu16_mask, _mm512_loadu_si512,
        _mm512_set1_epi16, _mm512_sub_epi16,
    };

    const SUPERBLOCK: usize = 512;
    out.clear();
    let points: Vec<__m512i> = cover
        .points()
        .iter()
        .map(|&point| _mm512_set1_epi16(point as i16))
        .collect();
    let ranges: Vec<(__m512i, __m512i)> = cover
        .ranges()
        .iter()
        .map(|&TokenRange { begin, last }| {
            (
                _mm512_set1_epi16(begin as i16),
                _mm512_set1_epi16(last.wrapping_sub(begin) as i16),
            )
        })
        .collect();
    let base = codes.as_ptr();
    let mask_at = |index: usize| unsafe {
        let value = _mm512_loadu_si512(base.add(index).cast());
        let mut mask = 0u32;
        for &point in &points {
            mask |= _mm512_cmpeq_epu16_mask(value, point);
        }
        for &(lo, span) in &ranges {
            mask |= _mm512_cmple_epu16_mask(_mm512_sub_epi16(value, lo), span);
        }
        mask
    };
    let mut sink = OriginalRowSink {
        row_offsets,
        out,
        row: 0,
        row_end: 0,
    };
    let mut index = 0usize;
    while index + SUPERBLOCK <= codes.len() {
        let mut lanes = [0u64; 8];
        let mut any = 0u64;
        for (chunk, slot) in lanes.iter_mut().enumerate() {
            let lo = u64::from(mask_at(index + chunk * 64));
            let hi = u64::from(mask_at(index + chunk * 64 + 32));
            *slot = lo | (hi << 32);
            any |= *slot;
        }
        if any != 0 {
            for (chunk, mask) in lanes.into_iter().enumerate() {
                if mask != 0 {
                    sink.mark_mask(index + chunk * 64, mask);
                }
            }
        }
        index += SUPERBLOCK;
    }
    while index + 32 <= codes.len() {
        let mut mask = mask_at(index);
        while mask != 0 {
            sink.hit(index + mask.trailing_zeros() as usize);
            mask &= mask - 1;
        }
        index += 32;
    }
    for (tail, &code) in codes[index..].iter().enumerate() {
        if cover.points().contains(&code) || cover.ranges().iter().any(|range| range.contains(code))
        {
            sink.hit(index + tail);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
fn original_avx512_candidates_fixed<const POINTS: usize, const RANGES: usize>(
    codes: &[u16],
    row_offsets: &[u64],
    cover: &ProbeCover,
    out: &mut Vec<usize>,
) {
    use core::arch::x86_64::{
        __m512i, _mm512_loadu_si512, _mm512_mask_cmpgt_epu16_mask, _mm512_mask_cmpneq_epu16_mask,
        _mm512_set1_epi16, _mm512_sub_epi16,
    };

    const SUPERBLOCK: usize = 512;
    out.clear();
    let points: [__m512i; POINTS] =
        std::array::from_fn(|i| _mm512_set1_epi16(cover.points()[i] as i16));
    let ranges: [(__m512i, __m512i); RANGES] = std::array::from_fn(|i| {
        let range = cover.ranges()[i];
        (
            _mm512_set1_epi16(range.begin as i16),
            _mm512_set1_epi16(range.last.wrapping_sub(range.begin) as i16),
        )
    });
    let base = codes.as_ptr();
    let mask_at = |index: usize| unsafe {
        let value = _mm512_loadu_si512(base.add(index).cast());
        let mut miss = u32::MAX;
        for &point in &points {
            miss = _mm512_mask_cmpneq_epu16_mask(miss, value, point);
        }
        for &(lo, span) in &ranges {
            miss = _mm512_mask_cmpgt_epu16_mask(miss, _mm512_sub_epi16(value, lo), span);
        }
        !miss
    };
    let mut sink = OriginalRowSink {
        row_offsets,
        out,
        row: 0,
        row_end: 0,
    };
    let mut index = 0usize;
    while index + SUPERBLOCK <= codes.len() {
        let mut lanes = [0u64; 8];
        let mut any = 0u64;
        for (chunk, slot) in lanes.iter_mut().enumerate() {
            let lo = u64::from(mask_at(index + chunk * 64));
            let hi = u64::from(mask_at(index + chunk * 64 + 32));
            *slot = lo | (hi << 32);
            any |= *slot;
        }
        if any != 0 {
            for (chunk, mask) in lanes.into_iter().enumerate() {
                if mask != 0 {
                    sink.mark_mask(index + chunk * 64, mask);
                }
            }
        }
        index += SUPERBLOCK;
    }
    while index + 32 <= codes.len() {
        let mut mask = mask_at(index);
        while mask != 0 {
            sink.hit(index + mask.trailing_zeros() as usize);
            mask &= mask - 1;
        }
        index += 32;
    }
    for (tail, &code) in codes[index..].iter().enumerate() {
        if cover.points().contains(&code) || cover.ranges().iter().any(|range| range.contains(code))
        {
            sink.hit(index + tail);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
fn original_avx512_candidates(
    codes: &[u16],
    row_offsets: &[u64],
    cover: &ProbeCover,
    out: &mut Vec<usize>,
) {
    macro_rules! fixed {
        ($p:literal, $r:literal) => {
            original_avx512_candidates_fixed::<$p, $r>(codes, row_offsets, cover, out)
        };
    }
    match (cover.points().len(), cover.ranges().len()) {
        (0, 1) => fixed!(0, 1),
        (1, 1) => fixed!(1, 1),
        (1, 2) => fixed!(1, 2),
        (1, 3) => fixed!(1, 3),
        (2, 0) => fixed!(2, 0),
        (2, 2) => fixed!(2, 2),
        (3, 1) => fixed!(3, 1),
        (3, 2) => fixed!(3, 2),
        (4, 0) => fixed!(4, 0),
        (10, 2) => fixed!(10, 2),
        (12, 5) => fixed!(12, 5),
        (25, 5) => fixed!(25, 5),
        _ => original_avx512_candidates_dynamic(codes, row_offsets, cover, out),
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
fn refine_live_blocks_fixed<const BLOCK_CODES: usize, const POINTS: usize, const RANGES: usize>(
    codes: &[u16],
    row_offsets: &[u64],
    cover: &ProbeCover,
    summary: &[u64],
    out: &mut Vec<usize>,
) {
    use core::arch::x86_64::{
        __m512i, _mm512_loadu_si512, _mm512_mask_cmpgt_epu16_mask, _mm512_mask_cmpneq_epu16_mask,
        _mm512_set1_epi16, _mm512_sub_epi16,
    };

    debug_assert_eq!(
        (cover.points().len(), cover.ranges().len()),
        (POINTS, RANGES)
    );
    out.clear();
    let points: [__m512i; POINTS] =
        std::array::from_fn(|i| _mm512_set1_epi16(cover.points()[i] as i16));
    let ranges: [(__m512i, __m512i); RANGES] = std::array::from_fn(|i| {
        let range = cover.ranges()[i];
        (
            _mm512_set1_epi16(range.begin as i16),
            _mm512_set1_epi16(range.last.wrapping_sub(range.begin) as i16),
        )
    });
    let base_ptr = codes.as_ptr();
    let mut sink = OriginalRowSink {
        row_offsets,
        out,
        row: 0,
        row_end: 0,
    };
    for (word_index, &word) in summary.iter().enumerate() {
        let mut live = word;
        while live != 0 {
            let block = word_index * 64 + live.trailing_zeros() as usize;
            let begin = block * BLOCK_CODES;
            let end = (begin + BLOCK_CODES).min(codes.len());
            let mut index = begin;
            while index + 32 <= end {
                let value = unsafe { _mm512_loadu_si512(base_ptr.add(index).cast()) };
                let mut miss = u32::MAX;
                for &point in &points {
                    miss = _mm512_mask_cmpneq_epu16_mask(miss, value, point);
                }
                for &(lo, span) in &ranges {
                    miss = _mm512_mask_cmpgt_epu16_mask(miss, _mm512_sub_epi16(value, lo), span);
                }
                let hits = !miss;
                if hits != 0 {
                    sink.mark_mask(index, u64::from(hits));
                }
                index += 32;
            }
            for (tail, &code) in codes[index..end].iter().enumerate() {
                if cover.points().contains(&code)
                    || cover.ranges().iter().any(|range| range.contains(code))
                {
                    sink.hit(index + tail);
                }
            }
            live &= live - 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
fn refine_live_block_positions_fixed<
    const BLOCK_CODES: usize,
    const POINTS: usize,
    const RANGES: usize,
>(
    codes: &[u16],
    cover: &ProbeCover,
    summary: &[u64],
    out: &mut Vec<usize>,
) {
    use core::arch::x86_64::{
        __m512i, _mm512_loadu_si512, _mm512_mask_cmpgt_epu16_mask, _mm512_mask_cmpneq_epu16_mask,
        _mm512_set1_epi16, _mm512_sub_epi16,
    };

    out.clear();
    let points: [__m512i; POINTS] =
        std::array::from_fn(|i| _mm512_set1_epi16(cover.points()[i] as i16));
    let ranges: [(__m512i, __m512i); RANGES] = std::array::from_fn(|i| {
        let range = cover.ranges()[i];
        (
            _mm512_set1_epi16(range.begin as i16),
            _mm512_set1_epi16(range.last.wrapping_sub(range.begin) as i16),
        )
    });
    let base_ptr = codes.as_ptr();
    for (word_index, &word) in summary.iter().enumerate() {
        let mut live = word;
        while live != 0 {
            let block = word_index * 64 + live.trailing_zeros() as usize;
            let begin = block * BLOCK_CODES;
            let end = (begin + BLOCK_CODES).min(codes.len());
            let mut index = begin;
            while index + 32 <= end {
                let value = unsafe { _mm512_loadu_si512(base_ptr.add(index).cast()) };
                let mut miss = u32::MAX;
                for &point in &points {
                    miss = _mm512_mask_cmpneq_epu16_mask(miss, value, point);
                }
                for &(lo, span) in &ranges {
                    miss = _mm512_mask_cmpgt_epu16_mask(miss, _mm512_sub_epi16(value, lo), span);
                }
                let mut hits = !miss;
                while hits != 0 {
                    out.push(index + hits.trailing_zeros() as usize);
                    hits &= hits - 1;
                }
                index += 32;
            }
            for (tail, &code) in codes[index..end].iter().enumerate() {
                if cover.points().contains(&code)
                    || cover.ranges().iter().any(|range| range.contains(code))
                {
                    out.push(index + tail);
                }
            }
            live &= live - 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
fn refine_live_block_positions<const BLOCK_CODES: usize>(
    codes: &[u16],
    cover: &ProbeCover,
    summary: &[u64],
    out: &mut Vec<usize>,
) {
    use core::arch::x86_64::{
        __m512i, _mm512_cmpeq_epu16_mask, _mm512_cmple_epu16_mask, _mm512_loadu_si512,
        _mm512_set1_epi16, _mm512_sub_epi16,
    };

    out.clear();
    let points: Vec<__m512i> = cover
        .points()
        .iter()
        .map(|&point| _mm512_set1_epi16(point as i16))
        .collect();
    let ranges: Vec<(__m512i, __m512i)> = cover
        .ranges()
        .iter()
        .map(|range| {
            (
                _mm512_set1_epi16(range.begin as i16),
                _mm512_set1_epi16(range.last.wrapping_sub(range.begin) as i16),
            )
        })
        .collect();
    let base_ptr = codes.as_ptr();
    for (word_index, &word) in summary.iter().enumerate() {
        let mut live = word;
        while live != 0 {
            let block = word_index * 64 + live.trailing_zeros() as usize;
            let begin = block * BLOCK_CODES;
            let end = (begin + BLOCK_CODES).min(codes.len());
            let mut index = begin;
            while index + 32 <= end {
                let value = unsafe { _mm512_loadu_si512(base_ptr.add(index).cast()) };
                let mut hits = 0u32;
                for &point in &points {
                    hits |= _mm512_cmpeq_epu16_mask(value, point);
                }
                for &(lo, span) in &ranges {
                    hits |= _mm512_cmple_epu16_mask(_mm512_sub_epi16(value, lo), span);
                }
                while hits != 0 {
                    out.push(index + hits.trailing_zeros() as usize);
                    hits &= hits - 1;
                }
                index += 32;
            }
            for (tail, &code) in codes[index..end].iter().enumerate() {
                if cover.points().contains(&code)
                    || cover.ranges().iter().any(|range| range.contains(code))
                {
                    out.push(index + tail);
                }
            }
            live &= live - 1;
        }
    }
}

fn localized_kmp(
    codes: &[u16],
    row_offsets: &[u64],
    positions: &[usize],
    windows: &[(usize, usize)],
    table: &ContainsTable,
    out: &mut Vec<usize>,
) -> usize {
    out.clear();
    let mut scanned_codes = 0usize;
    let mut row = 0usize;
    let mut position_index = 0usize;
    while position_index < positions.len() {
        let position = positions[position_index];
        row = interpolated_row(position, row, row_offsets);
        let row_end = row_offsets[row + 1] as usize;
        let mut matched = false;
        while position_index < positions.len() && positions[position_index] < row_end {
            let hit = positions[position_index];
            if !matched {
                let (before, after) = windows[codes[hit] as usize];
                let begin = (row_offsets[row] as usize).max(hit.saturating_sub(before));
                let end = row_end.min(hit.saturating_add(after).saturating_add(1));
                scanned_codes += end - begin;
                matched = contains(&codes[begin..end], table);
            }
            position_index += 1;
        }
        if matched {
            out.push(row);
        }
    }
    scanned_codes
}

#[allow(clippy::too_many_arguments)]
fn localized_memmem(
    codes: &[u16],
    row_offsets: &[u64],
    dict: CompactDictionaryView<'_>,
    positions: &[usize],
    windows: &[(usize, usize)],
    finder: &Finder<'_>,
    scratch: &mut Vec<MaybeUninit<u8>>,
    out: &mut Vec<usize>,
) -> usize {
    out.clear();
    let mut scanned_codes = 0usize;
    let mut row = 0usize;
    let mut position_index = 0usize;
    while position_index < positions.len() {
        let position = positions[position_index];
        row = interpolated_row(position, row, row_offsets);
        let row_end = row_offsets[row + 1] as usize;
        let mut matched = false;
        while position_index < positions.len() && positions[position_index] < row_end {
            let hit = positions[position_index];
            if !matched {
                let (before, after) = windows[codes[hit] as usize];
                let begin = (row_offsets[row] as usize).max(hit.saturating_sub(before));
                let end = row_end.min(hit.saturating_add(after).saturating_add(1));
                let window = &codes[begin..end];
                scanned_codes += window.len();
                let need = window.len() * MAX_TOKEN_SIZE + DECODE_PADDING;
                if scratch.len() < need {
                    scratch.resize(need, MaybeUninit::uninit());
                }
                let written = unsafe { decode_into(window, dict, scratch) };
                let bytes =
                    unsafe { std::slice::from_raw_parts(scratch.as_ptr().cast::<u8>(), written) };
                matched = finder.find(bytes).is_some();
            }
            position_index += 1;
        }
        if matched {
            out.push(row);
        }
    }
    scanned_codes
}

#[inline]
fn interpolated_row(position: usize, floor: usize, row_offsets: &[u64]) -> usize {
    if position < row_offsets[floor + 1] as usize {
        return floor;
    }
    let rows = row_offsets.len() - 1;
    let base_code = row_offsets[floor] as usize;
    let remaining_codes = row_offsets[rows] as usize - base_code;
    let remaining_rows = rows - floor;
    let delta = position - base_code;
    let mut row = floor + delta.saturating_mul(remaining_rows) / remaining_codes.max(1);
    row = row.min(rows - 1);
    while row > floor && row_offsets[row] as usize > position {
        row -= 1;
    }
    while row + 1 < row_offsets.len() && row_offsets[row + 1] as usize <= position {
        row += 1;
    }
    row
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
fn refine_live_blocks<const BLOCK_CODES: usize>(
    codes: &[u16],
    row_offsets: &[u64],
    cover: &ProbeCover,
    summary: &[u64],
    out: &mut Vec<usize>,
) {
    use core::arch::x86_64::{
        __m512i, _mm512_cmpeq_epu16_mask, _mm512_cmple_epu16_mask, _mm512_loadu_si512,
        _mm512_set1_epi16, _mm512_sub_epi16,
    };

    out.clear();
    let points: Vec<__m512i> = cover
        .points()
        .iter()
        .map(|&point| _mm512_set1_epi16(point as i16))
        .collect();
    let ranges: Vec<(__m512i, __m512i)> = cover
        .ranges()
        .iter()
        .map(|range| {
            (
                _mm512_set1_epi16(range.begin as i16),
                _mm512_set1_epi16(range.last.wrapping_sub(range.begin) as i16),
            )
        })
        .collect();
    let base_ptr = codes.as_ptr();
    let mut sink = OriginalRowSink {
        row_offsets,
        out,
        row: 0,
        row_end: 0,
    };
    for (word_index, &word) in summary.iter().enumerate() {
        let mut live = word;
        while live != 0 {
            let block = word_index * 64 + live.trailing_zeros() as usize;
            let begin = block * BLOCK_CODES;
            let end = (begin + BLOCK_CODES).min(codes.len());
            let mut index = begin;
            while index + 32 <= end {
                let value = unsafe { _mm512_loadu_si512(base_ptr.add(index).cast()) };
                let mut hits = 0u32;
                for &point in &points {
                    hits |= _mm512_cmpeq_epu16_mask(value, point);
                }
                for &(lo, span) in &ranges {
                    hits |= _mm512_cmple_epu16_mask(_mm512_sub_epi16(value, lo), span);
                }
                if hits != 0 {
                    sink.mark_mask(index, u64::from(hits));
                }
                index += 32;
            }
            for (tail, &code) in codes[index..end].iter().enumerate() {
                if cover.points().contains(&code)
                    || cover.ranges().iter().any(|range| range.contains(code))
                {
                    sink.hit(index + tail);
                }
            }
            live &= live - 1;
        }
    }
}

fn refine_live_blocks_autovec_p2r2<const BLOCK_CODES: usize>(
    codes: &[u16],
    row_offsets: &[u64],
    cover: &ProbeCover,
    summary: &[u64],
    out: &mut Vec<usize>,
) {
    debug_assert_eq!((cover.points().len(), cover.ranges().len()), (2, 2));
    out.clear();
    let points: [u16; 2] = std::array::from_fn(|i| cover.points()[i]);
    let lows: [u16; 2] = std::array::from_fn(|i| cover.ranges()[i].begin);
    let spans: [u16; 2] = std::array::from_fn(|i| {
        let range = cover.ranges()[i];
        range.last.wrapping_sub(range.begin)
    });
    let mut sink = OriginalRowSink {
        row_offsets,
        out,
        row: 0,
        row_end: 0,
    };
    for (word_index, &word) in summary.iter().enumerate() {
        let mut live = word;
        while live != 0 {
            let block = word_index * 64 + live.trailing_zeros() as usize;
            let begin = block * BLOCK_CODES;
            let end = (begin + BLOCK_CODES).min(codes.len());
            let mut index = begin;
            while index + 64 <= end {
                let mut lanes = 0u64;
                for lane in 0..64 {
                    let code = codes[index + lane];
                    let hit = points.contains(&code)
                        || (0..2).any(|range| code.wrapping_sub(lows[range]) <= spans[range]);
                    lanes |= u64::from(hit) << lane;
                }
                if lanes != 0 {
                    sink.mark_mask(index, lanes);
                }
                index += 64;
            }
            for (tail, &code) in codes[index..end].iter().enumerate() {
                if points.contains(&code)
                    || (0..2).any(|range| code.wrapping_sub(lows[range]) <= spans[range])
                {
                    sink.hit(index + tail);
                }
            }
            live &= live - 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn refine_live_blocks_avx2_p2r2<const BLOCK_CODES: usize>(
    codes: &[u16],
    row_offsets: &[u64],
    cover: &ProbeCover,
    summary: &[u64],
    out: &mut Vec<usize>,
) {
    use core::arch::x86_64::{
        __m256i, _mm256_andnot_si256, _mm256_cmpeq_epi16, _mm256_cmpgt_epi16, _mm256_loadu_si256,
        _mm256_movemask_epi8, _mm256_or_si256, _mm256_packs_epi16, _mm256_permute4x64_epi64,
        _mm256_set1_epi16, _mm256_setzero_si256, _mm256_xor_si256,
    };

    debug_assert_eq!((cover.points().len(), cover.ranges().len()), (2, 2));
    out.clear();
    let zero = _mm256_setzero_si256();
    let ones = _mm256_set1_epi16(-1);
    let bias = _mm256_set1_epi16(i16::MIN);
    let points: [__m256i; 2] = std::array::from_fn(|i| _mm256_set1_epi16(cover.points()[i] as i16));
    let ranges: [(__m256i, __m256i); 2] = std::array::from_fn(|i| {
        let range = cover.ranges()[i];
        (
            _mm256_xor_si256(_mm256_set1_epi16(range.begin as i16), bias),
            _mm256_xor_si256(_mm256_set1_epi16(range.last as i16), bias),
        )
    });
    let matching_mask = |value: __m256i| {
        let mut hits = zero;
        for &point in &points {
            hits = _mm256_or_si256(hits, _mm256_cmpeq_epi16(value, point));
        }
        let biased = _mm256_xor_si256(value, bias);
        for &(lo, hi) in &ranges {
            let outside = _mm256_or_si256(
                _mm256_cmpgt_epi16(lo, biased),
                _mm256_cmpgt_epi16(biased, hi),
            );
            hits = _mm256_or_si256(hits, _mm256_andnot_si256(outside, ones));
        }
        hits
    };
    let base = codes.as_ptr();
    let mut sink = OriginalRowSink {
        row_offsets,
        out,
        row: 0,
        row_end: 0,
    };
    for (word_index, &word) in summary.iter().enumerate() {
        let mut live = word;
        while live != 0 {
            let block = word_index * 64 + live.trailing_zeros() as usize;
            let begin = block * BLOCK_CODES;
            let end = (begin + BLOCK_CODES).min(codes.len());
            let mut index = begin;
            while index + 64 <= end {
                let masks: [__m256i; 4] = std::array::from_fn(|chunk| {
                    // SAFETY: this loop runs only when all 64 codes are in the
                    // selected live block.
                    matching_mask(unsafe {
                        _mm256_loadu_si256(base.add(index + chunk * 16).cast())
                    })
                });
                let lanes01 = _mm256_movemask_epi8(_mm256_permute4x64_epi64::<0xd8>(
                    _mm256_packs_epi16(masks[0], masks[1]),
                )) as u32 as u64;
                let lanes23 = _mm256_movemask_epi8(_mm256_permute4x64_epi64::<0xd8>(
                    _mm256_packs_epi16(masks[2], masks[3]),
                )) as u32 as u64;
                let lanes = lanes01 | (lanes23 << 32);
                if lanes != 0 {
                    sink.mark_mask(index, lanes);
                }
                index += 64;
            }
            for (tail, &code) in codes[index..end].iter().enumerate() {
                if cover.points().contains(&code)
                    || cover.ranges().iter().any(|range| range.contains(code))
                {
                    sink.hit(index + tail);
                }
            }
            live &= live - 1;
        }
    }
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "run".to_string());
    let parquet = PathBuf::from(
        std::env::var("ONPAIR_BENCH_PARQUET").unwrap_or_else(|_| DEFAULT_PARQUET.to_string()),
    );
    let dump =
        PathBuf::from(std::env::var("ONPAIR_PF_DUMP").unwrap_or_else(|_| DEFAULT_DUMP.to_string()));
    match mode.as_str() {
        "prepare" => prepare(&parquet, &dump),
        "prepare-lines" => prepare_lines(
            Path::new(
                &std::env::var("ONPAIR_BENCH_LINES").expect("ONPAIR_BENCH_LINES is required"),
            ),
            &dump,
        ),
        "run" => run(&parquet, &dump),
        "perf-original-google" => perf_google(&dump, true),
        "perf-new-google" => perf_google(&dump, false),
        "perf-summary-google" => perf_summary_google(&dump),
        _ => panic!(
            "usage: prefilter_e2e [prepare|prepare-lines|run|perf-original-google|perf-new-google|perf-summary-google]"
        ),
    }
}

fn perf_summary_google(dump: &Path) {
    assert!(std::is_x86_feature_detected!("avx512bw"));
    let column = load_column(dump);
    let view = column.view();
    let frequencies = build_token_frequency_index(view.codes, view.dict.num_tokens()).unwrap();
    let query = std::env::var("ONPAIR_PERF_QUERY").unwrap_or_else(|_| "google".to_string());
    let analysis = analyze_prefilter(query.as_bytes(), view.dict, &frequencies);
    assert_eq!(
        (
            analysis.probe_cover().points().len(),
            analysis.probe_cover().ranges().len()
        ),
        (2, 2)
    );
    let reps = std::env::var("ONPAIR_PERF_REPS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1_000);
    let block_codes = std::env::var("ONPAIR_PF_BLOCK_CODES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(64);
    let blocks = view.codes.len().div_ceil(block_codes);
    let mut summary = vec![0u64; blocks.div_ceil(64)];
    let kernel = std::env::var("ONPAIR_PF_KERNEL").unwrap_or_else(|_| "intrinsics".into());
    let start = Instant::now();
    for _ in 0..reps {
        macro_rules! run_block {
            ($block:literal) => {
                match kernel.as_str() {
                    "intrinsics" => unsafe {
                        summarize_superblocks_fixed_miss::<$block, 2, 2>(
                            black_box(view.codes),
                            black_box(analysis.probe_cover()),
                            black_box(&mut summary),
                        )
                    },
                    "intrinsics-avx2" => unsafe {
                        summarize_superblocks_avx2_p2r2::<$block>(
                            black_box(view.codes),
                            black_box(analysis.probe_cover()),
                            black_box(&mut summary),
                        )
                    },
                    "intrinsics-branch" => unsafe {
                        summarize_superblocks_fixed_miss_branch::<$block, 2, 2>(
                            black_box(view.codes),
                            black_box(analysis.probe_cover()),
                            black_box(&mut summary),
                        )
                    },
                    "scalar" => summarize_superblocks_scalar::<$block>(
                        black_box(view.codes),
                        black_box(analysis.probe_cover()),
                        black_box(&mut summary),
                    ),
                    "autovec" => summarize_superblocks_autovec::<$block, 2, 2>(
                        black_box(view.codes),
                        black_box(analysis.probe_cover()),
                        black_box(&mut summary),
                    ),
                    "autovec-miss" => unsafe {
                        summarize_superblocks_autovec_miss::<$block, 2, 2>(
                            black_box(view.codes),
                            black_box(analysis.probe_cover()),
                            black_box(&mut summary),
                        )
                    },
                    other => panic!("unknown ONPAIR_PF_KERNEL={other}"),
                }
            };
        }
        match block_codes {
            32 => run_block!(32),
            64 => run_block!(64),
            128 => run_block!(128),
            256 => run_block!(256),
            512 => run_block!(512),
            other => panic!("unsupported ONPAIR_PF_BLOCK_CODES={other}"),
        }
        black_box(&summary);
    }
    let elapsed = start.elapsed();
    let mut candidates = Vec::new();
    match block_codes {
        32 => superblock_candidates::<32>(&summary, view.row_offsets, &mut candidates),
        64 => superblock_candidates::<64>(&summary, view.row_offsets, &mut candidates),
        128 => superblock_candidates::<128>(&summary, view.row_offsets, &mut candidates),
        256 => superblock_candidates::<256>(&summary, view.row_offsets, &mut candidates),
        512 => superblock_candidates::<512>(&summary, view.row_offsets, &mut candidates),
        _ => unreachable!(),
    }
    eprintln!(
        "query={} kernel={} block_codes={} reps={} live_blocks={} candidates={} elapsed_ms={:.3}",
        query,
        kernel,
        block_codes,
        reps,
        summary
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum::<usize>(),
        candidates.len(),
        elapsed.as_secs_f64() * 1_000.0,
    );
}

fn perf_google(dump: &Path, original: bool) {
    assert!(std::is_x86_feature_detected!("avx512bw"));
    let column = load_column(dump);
    let view = column.view();
    let frequencies = build_token_frequency_index(view.codes, view.dict.num_tokens()).unwrap();
    let query = std::env::var("ONPAIR_PERF_QUERY").unwrap_or_else(|_| "google".to_string());
    let analysis = analyze_prefilter(query.as_bytes(), view.dict, &frequencies);
    let reps = std::env::var("ONPAIR_PERF_REPS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1_000);
    let mut out = Vec::new();
    let start = Instant::now();
    for _ in 0..reps {
        if original {
            unsafe {
                original_avx512_candidates(
                    black_box(view.codes),
                    black_box(view.row_offsets),
                    black_box(analysis.probe_cover()),
                    black_box(&mut out),
                )
            };
        } else {
            out.clear();
            prefilter_candidates(
                black_box(view.codes),
                black_box(view.row_offsets),
                black_box(&analysis),
                black_box(&mut out),
            )
            .unwrap();
        }
        black_box(&out);
    }
    eprintln!(
        "query={} kernel={} reps={} candidates={} elapsed_ms={:.3}",
        query,
        if original { "original" } else { "new" },
        reps,
        out.len(),
        start.elapsed().as_secs_f64() * 1_000.0,
    );
}

fn prepare(parquet: &Path, dump: &Path) {
    let rows = read_urls(parquet);
    prepare_rows(rows, dump);
}

fn prepare_lines(lines: &Path, dump: &Path) {
    let bytes = fs::read(lines).expect("read line corpus");
    let rows = bytes
        .split(|&byte| byte == b'\n')
        .filter(|row| !row.is_empty())
        .map(<[u8]>::to_vec)
        .collect();
    prepare_rows(rows, dump);
}

fn prepare_rows(rows: Vec<Vec<u8>>, dump: &Path) {
    let row_count = rows.len();
    let (bytes, offsets) = pack(&rows);
    eprintln!(
        "preparing {} rows, {:.2} MiB decoded",
        row_count,
        bytes.len() as f64 / 1_048_576.0
    );
    drop(rows);
    let cfg = Config {
        max_dict_bits: MaxDictBits::new(16).unwrap(),
        threshold: Threshold::new(0.5).unwrap(),
        seed: Some(42),
    };
    let column: Column<u64> = compress(&bytes, &offsets, cfg).unwrap();
    fs::create_dir_all(dump).unwrap();
    fs::write(dump.join("dict.bytes"), column.dict.bytes()).unwrap();
    write_u32(&dump.join("dict.offsets.u32"), column.dict.offsets());
    write_u16(&dump.join("codes.u16"), &column.codes);
    write_u64(&dump.join("rows.u64"), &column.row_offsets);
    eprintln!(
        "dumped {} codes, {} rows, {} tokens to {}",
        column.codes.len(),
        column.row_offsets.len() - 1,
        column.dict.offsets().len() - 1,
        dump.display()
    );
}

fn run(parquet: &Path, dump: &Path) {
    assert!(std::is_x86_feature_detected!("avx512bw"));
    let summary_codes = std::env::var("ONPAIR_SUMMARY_CODES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_SUMMARY_CODES);
    assert!(matches!(summary_codes, 32 | 64 | 128 | 256 | 512));
    macro_rules! summarize_block {
        ($block:literal, $codes:expr, $cover:expr, $summary:expr) => {
            match ($cover.points().len(), $cover.ranges().len()) {
                (0, 1) => {
                    summarize_superblocks_fixed_miss::<$block, 0, 1>($codes, $cover, $summary)
                }
                (1, 1) => {
                    summarize_superblocks_fixed_miss::<$block, 1, 1>($codes, $cover, $summary)
                }
                (1, 2) => {
                    summarize_superblocks_fixed_miss::<$block, 1, 2>($codes, $cover, $summary)
                }
                (1, 3) => {
                    summarize_superblocks_fixed_miss::<$block, 1, 3>($codes, $cover, $summary)
                }
                (2, 0) => {
                    summarize_superblocks_fixed_miss::<$block, 2, 0>($codes, $cover, $summary)
                }
                (2, 2) => {
                    summarize_superblocks_fixed_miss::<$block, 2, 2>($codes, $cover, $summary)
                }
                (3, 2) => {
                    summarize_superblocks_fixed_miss::<$block, 3, 2>($codes, $cover, $summary)
                }
                (4, 0) => {
                    summarize_superblocks_fixed_miss::<$block, 4, 0>($codes, $cover, $summary)
                }
                (10, 2) => {
                    summarize_superblocks_fixed_miss::<$block, 10, 2>($codes, $cover, $summary)
                }
                (12, 5) => {
                    summarize_superblocks_fixed_miss::<$block, 12, 5>($codes, $cover, $summary)
                }
                (25, 5) => {
                    summarize_superblocks_fixed_miss::<$block, 25, 5>($codes, $cover, $summary)
                }
                (3, 1) => {
                    summarize_superblocks_fixed_miss::<$block, 3, 1>($codes, $cover, $summary)
                }
                (4, 2) => summarize_superblocks_fixed::<$block, 4, 2>($codes, $cover, $summary),
                (5, 0) => summarize_superblocks_fixed::<$block, 5, 0>($codes, $cover, $summary),
                (10, 3) => summarize_superblocks_fixed::<$block, 10, 3>($codes, $cover, $summary),
                _ => summarize_superblocks::<$block>($codes, $cover, $summary),
            }
        };
    }
    macro_rules! summarize {
        ($codes:expr, $cover:expr, $summary:expr) => {
            match summary_codes {
                32 => summarize_block!(32, $codes, $cover, $summary),
                64 => summarize_block!(64, $codes, $cover, $summary),
                128 => summarize_block!(128, $codes, $cover, $summary),
                256 => summarize_block!(256, $codes, $cover, $summary),
                512 => summarize_block!(512, $codes, $cover, $summary),
                _ => unreachable!(),
            }
        };
    }
    macro_rules! block_candidates {
        ($summary:expr, $offsets:expr, $out:expr) => {
            match summary_codes {
                32 => superblock_candidates::<32>($summary, $offsets, $out),
                64 => superblock_candidates::<64>($summary, $offsets, $out),
                128 => superblock_candidates::<128>($summary, $offsets, $out),
                256 => superblock_candidates::<256>($summary, $offsets, $out),
                512 => superblock_candidates::<512>($summary, $offsets, $out),
                _ => unreachable!(),
            }
        };
    }
    macro_rules! refine_rows_block {
        ($block:literal, $codes:expr, $offsets:expr, $cover:expr, $summary:expr, $out:expr) => {
            match ($cover.points().len(), $cover.ranges().len()) {
                (0, 1) => refine_live_blocks_fixed::<$block, 0, 1>(
                    $codes, $offsets, $cover, $summary, $out,
                ),
                (1, 1) => refine_live_blocks_fixed::<$block, 1, 1>(
                    $codes, $offsets, $cover, $summary, $out,
                ),
                (1, 2) => refine_live_blocks_fixed::<$block, 1, 2>(
                    $codes, $offsets, $cover, $summary, $out,
                ),
                (1, 3) => refine_live_blocks_fixed::<$block, 1, 3>(
                    $codes, $offsets, $cover, $summary, $out,
                ),
                (2, 0) => refine_live_blocks_fixed::<$block, 2, 0>(
                    $codes, $offsets, $cover, $summary, $out,
                ),
                (2, 2) => refine_live_blocks_fixed::<$block, 2, 2>(
                    $codes, $offsets, $cover, $summary, $out,
                ),
                (3, 1) => refine_live_blocks_fixed::<$block, 3, 1>(
                    $codes, $offsets, $cover, $summary, $out,
                ),
                (3, 2) => refine_live_blocks_fixed::<$block, 3, 2>(
                    $codes, $offsets, $cover, $summary, $out,
                ),
                (4, 0) => refine_live_blocks_fixed::<$block, 4, 0>(
                    $codes, $offsets, $cover, $summary, $out,
                ),
                (10, 2) => refine_live_blocks_fixed::<$block, 10, 2>(
                    $codes, $offsets, $cover, $summary, $out,
                ),
                (12, 5) => refine_live_blocks_fixed::<$block, 12, 5>(
                    $codes, $offsets, $cover, $summary, $out,
                ),
                (25, 5) => refine_live_blocks_fixed::<$block, 25, 5>(
                    $codes, $offsets, $cover, $summary, $out,
                ),
                _ => refine_live_blocks::<$block>($codes, $offsets, $cover, $summary, $out),
            }
        };
    }
    let column = load_column(dump);
    let view = column.view();
    let frequencies = build_token_frequency_index(view.codes, view.dict.num_tokens()).unwrap();
    let reps = std::env::var("ONPAIR_BENCH_REPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_REPS);
    let max_codes = std::env::var("ONPAIR_PF_MAX_BLOCKS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .map(|blocks| blocks.saturating_mul(256))
        .unwrap_or(view.codes.len())
        .min(view.codes.len());
    let scan_rows = view
        .row_offsets
        .partition_point(|offset| (*offset as usize) <= max_codes)
        .saturating_sub(1);
    let scan_code_count = view.row_offsets[scan_rows] as usize;
    let scan_codes = &view.codes[..scan_code_count];
    let scan_offsets = &view.row_offsets[..=scan_rows];
    let segment_bytes = std::env::var("ONPAIR_SEGMENT_BYTES")
        .ok()
        .map(|value| value.parse::<usize>().expect("valid ONPAIR_SEGMENT_BYTES"));
    let segment_algorithm = std::env::var("ONPAIR_SEGMENT_ALGORITHM").ok();
    let segment_rows = segment_bytes.map(|bytes| {
        assert!(bytes >= 2, "segments must hold at least one u16 code");
        let max_codes = bytes / size_of::<u16>();
        let mut ranges = Vec::new();
        let mut row_begin = 0usize;
        while row_begin < scan_rows {
            let code_begin = scan_offsets[row_begin] as usize;
            let code_limit = code_begin.saturating_add(max_codes);
            let mut row_end = scan_offsets.partition_point(|&offset| offset as usize <= code_limit);
            row_end = row_end.saturating_sub(1).min(scan_rows);
            if row_end == row_begin {
                row_end += 1;
            }
            ranges.push((row_begin, row_end));
            row_begin = row_end;
        }
        ranges
    });
    let low_selectivity_only = std::env::var_os("ONPAIR_PF_LOW_SELECTIVITY").is_some();
    let measure_original = std::env::var_os("ONPAIR_PF_ORIGINAL_GOOGLE").is_some();
    let measure_original_all = std::env::var_os("ONPAIR_PF_ORIGINAL_ALL").is_some();
    let named_only = std::env::var_os("ONPAIR_NAMED_ONLY").is_some();
    let dispatch_only = std::env::var_os("ONPAIR_DISPATCH_ONLY").is_some();
    let features_only = std::env::var_os("ONPAIR_FEATURES_ONLY").is_some();
    let hierarchy_kernel =
        std::env::var("ONPAIR_HIER_KERNEL").unwrap_or_else(|_| "intrinsics-avx512".to_string());
    let fixed_query = std::env::var("ONPAIR_ONLY_QUERY").ok();
    let query_file = std::env::var("ONPAIR_QUERY_FILE").ok().map(PathBuf::from);
    let rows = if query_file.is_some()
        || (fixed_query.is_some() && std::env::var_os("ONPAIR_HIER_ONLY").is_some())
    {
        vec![Vec::new()]
    } else {
        read_urls(parquet)
    };
    let decoded_bytes = view.decoded_len();
    let mut queries = if let Some(path) = &query_file {
        fs::read(path)
            .expect("read query file")
            .split(|&byte| byte == b'\n')
            .filter(|query| !query.is_empty())
            .map(<[u8]>::to_vec)
            .collect()
    } else {
        queries(&rows)
    };
    if std::env::var_os("ONPAIR_MINE_NEEDLES").is_some() {
        let mut candidates = HashSet::new();
        let stride = (rows.len() / 2_048).max(1);
        for (ordinal, row) in rows.iter().step_by(stride).enumerate() {
            for &len in &[4usize, 6, 8, 12, 16] {
                if row.len() < len {
                    continue;
                }
                let span = row.len() - len + 1;
                for start in [0, (ordinal.wrapping_mul(37)) % span, span - 1] {
                    let needle = &row[start..start + len];
                    if needle.iter().all(|byte| byte.is_ascii_graphic()) {
                        candidates.insert(needle.to_vec());
                    }
                }
            }
        }
        let mut eligible: Vec<(f64, Vec<u8>)> = candidates
            .into_iter()
            .filter_map(|needle| {
                let analysis = analyze_prefilter(&needle, view.dict, &frequencies);
                let cover = analysis.probe_cover();
                ((cover.points().len(), cover.ranges().len()) == (2, 2))
                    .then_some((analysis.covered_fraction(), needle))
            })
            .collect();
        eligible.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut mined = Vec::new();
        let targets: Vec<f64> = if std::env::var_os("ONPAIR_MINE_DENSE").is_some() {
            let lo = 1e-6f64.ln();
            let hi = 2e-3f64.ln();
            (0..32)
                .map(|index| (lo + (hi - lo) * index as f64 / 31.0).exp())
                .collect()
        } else {
            vec![1e-7f64, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
        };
        for target in targets {
            if let Some((fraction, needle)) = eligible.iter().min_by(|a, b| {
                (a.0.max(f64::MIN_POSITIVE).ln() - target.ln())
                    .abs()
                    .total_cmp(&(b.0.max(f64::MIN_POSITIVE).ln() - target.ln()).abs())
            }) && !mined.iter().any(|value: &Vec<u8>| value == needle)
            {
                eprintln!("mined\t{}\tcover_frac={fraction:.8}", label(needle));
                mined.push(needle.clone());
            }
        }
        queries.extend(mined);
        queries.sort();
        queries.dedup();
    }
    if let Some(query) = fixed_query {
        queries.clear();
        queries.push(query.into_bytes());
    } else if std::env::var_os("ONPAIR_ONLY_GOOGLE").is_some() {
        queries.retain(|query| query == b"google");
    }
    if let Some(path) = std::env::var("ONPAIR_WRITE_QUERIES")
        .ok()
        .map(PathBuf::from)
    {
        let capacity = queries.iter().map(Vec::len).sum::<usize>() + queries.len();
        let mut output = Vec::with_capacity(capacity);
        for query in &queries {
            output.extend_from_slice(query);
            output.push(b'\n');
        }
        let bytes = output.len();
        fs::write(&path, output).expect("write query file");
        eprintln!(
            "wrote {} queries / {bytes} bytes to {}",
            queries.len(),
            path.display()
        );
    }
    eprintln!(
        "running {} queries over {} codes / {} rows, best-of-{reps}",
        queries.len(),
        scan_codes.len(),
        scan_rows,
    );
    println!(
        "query\tbytes\tpoints\tranges\tcover_frac\tcandidates\tblock_candidates\tmatches\toriginal_kmp_ms\toriginal_memmem_ms\tstage1_ms\tstage1_kmp_ms\tstage1_memmem_ms\tblock1_ms\tblock1_kmp_ms\tblock1_memmem_ms"
    );
    // Full KMP/memmem, exact-mask stage 1/KMP/memmem, block-bit stage 1/KMP/memmem.
    let mut batch_ns = [0.0; 8];
    let mut measured = 0usize;
    let mut selectivity_ns = [[0.0; 8]; 4];
    let mut selectivity_queries = [0usize; 4];
    for query in queries {
        let analysis = analyze_prefilter(&query, view.dict, &frequencies);
        let cover = analysis.probe_cover();
        if cover.points().is_empty() && cover.ranges().is_empty() {
            continue;
        }
        if std::env::var_os("ONPAIR_P2R2_ONLY").is_some()
            && (cover.points().len(), cover.ranges().len()) != (2, 2)
        {
            continue;
        }
        if low_selectivity_only && analysis.covered_fraction() >= 0.001 {
            continue;
        }
        if features_only {
            let model = static_dispatch(
                analysis.covered_fraction(),
                cover.points().len(),
                cover.ranges().len(),
            );
            println!(
                "FEATURES_{}\tneedle_bytes={}\tpoints={}\tranges={}\tcomparison_cost={}\tcover_fraction={:.8}\tmodel_algorithm={}",
                label(&query),
                query.len(),
                cover.points().len(),
                cover.ranges().len(),
                analysis.comparison_cost(),
                analysis.covered_fraction(),
                model.name(),
            );
            continue;
        }
        let kmp = ContainsTable::new(&query, view.dict);

        if let (Some(segment_bytes), Some(segment_rows)) = (segment_bytes, &segment_rows) {
            let mut windows = vec![(query.len(), query.len()); view.dict.num_tokens()];
            for window in analysis.probe_windows() {
                windows[window.token() as usize] = (window.before_codes(), window.after_codes());
            }
            let selected = static_dispatch(
                analysis.covered_fraction(),
                cover.points().len(),
                cover.ranges().len(),
            );
            let exact_match_count: usize = segment_rows
                .iter()
                .map(|&(row_begin, row_end)| {
                    (row_begin..row_end)
                        .filter(|&row| contains(view.row_codes(row), &kmp))
                        .count()
                })
                .sum();
            let row_match_fraction = exact_match_count as f64 / scan_rows as f64;
            let gated_algorithm = if row_match_fraction < 0.001 {
                selected.name()
            } else {
                "full-kmp"
            };
            let mut run_samples: [Vec<u128>; 8] = std::array::from_fn(|_| Vec::with_capacity(reps));
            let mut max_segment_code_bytes = 0usize;

            for _ in 0..reps {
                let mut run_totals = [0u128; 8];
                for &(row_begin, row_end) in segment_rows {
                    let code_begin = scan_offsets[row_begin] as usize;
                    let code_end = scan_offsets[row_end] as usize;
                    let codes = &scan_codes[code_begin..code_end];
                    max_segment_code_bytes = max_segment_code_bytes.max(size_of_val(codes));
                    let offsets: Vec<u64> = scan_offsets[row_begin..=row_end]
                        .iter()
                        .map(|&offset| offset - code_begin as u64)
                        .collect();
                    let expected: Vec<usize> = (0..row_end - row_begin)
                        .filter(|&row| {
                            contains(
                                &codes[offsets[row] as usize..offsets[row + 1] as usize],
                                &kmp,
                            )
                        })
                        .collect();
                    let mut super_summary =
                        vec![0u64; codes.len().div_ceil(summary_codes).div_ceil(64)];
                    let mut hierarchy_summary = vec![0u64; codes.len().div_ceil(512).div_ceil(64)];
                    let mut rows = Vec::new();
                    let mut positions = Vec::new();

                    macro_rules! measure_segment {
                    ($slot:literal, $name:literal, $body:block) => {{
                        if segment_algorithm
                            .as_deref()
                            .is_none_or(|selected| selected == $name)
                        {
                            let start = Instant::now();
                            $body
                            let elapsed = start.elapsed().as_nanos();
                            assert_eq!(rows, expected);
                            run_totals[$slot] += elapsed;
                        }
                    }};
                }

                    measure_segment!(0, "scan", {
                        unsafe {
                            original_avx512_candidates(codes, &offsets, cover, &mut rows);
                        }
                        rows.retain(|&row| {
                            contains(
                                &codes[offsets[row] as usize..offsets[row + 1] as usize],
                                &kmp,
                            )
                        });
                    });
                    measure_segment!(1, "superblock", {
                        unsafe { summarize!(codes, cover, &mut super_summary) };
                        block_candidates!(&super_summary, &offsets, &mut rows);
                        rows.retain(|&row| {
                            contains(
                                &codes[offsets[row] as usize..offsets[row + 1] as usize],
                                &kmp,
                            )
                        });
                    });
                    measure_segment!(2, "hierarchical", {
                        unsafe {
                            summarize_block!(512, codes, cover, &mut hierarchy_summary);
                            refine_rows_block!(
                                512,
                                codes,
                                &offsets,
                                cover,
                                &hierarchy_summary,
                                &mut rows
                            );
                        }
                        rows.retain(|&row| {
                            contains(
                                &codes[offsets[row] as usize..offsets[row + 1] as usize],
                                &kmp,
                            )
                        });
                    });
                    measure_segment!(3, "hierarchical-length", {
                        unsafe {
                            summarize_block!(512, codes, cover, &mut hierarchy_summary);
                            refine_rows_block!(
                                512,
                                codes,
                                &offsets,
                                cover,
                                &hierarchy_summary,
                                &mut rows
                            );
                        }
                        let minimum_codes = query.len() / MAX_TOKEN_SIZE;
                        rows.retain(|&row| {
                            (offsets[row + 1] - offsets[row]) as usize >= minimum_codes
                        });
                        rows.retain(|&row| {
                            contains(
                                &codes[offsets[row] as usize..offsets[row + 1] as usize],
                                &kmp,
                            )
                        });
                    });
                    measure_segment!(4, "midcut", {
                        unsafe {
                            summarize_block!(512, codes, cover, &mut hierarchy_summary);
                            refine_live_block_positions::<512>(
                                codes,
                                cover,
                                &hierarchy_summary,
                                &mut positions,
                            );
                        }
                        localized_kmp(codes, &offsets, &positions, &windows, &kmp, &mut rows);
                    });
                    measure_segment!(5, "dispatch", {
                        match selected {
                            StaticDispatch::DirectKmp => {
                                rows.clear();
                                rows.extend((0..offsets.len() - 1).filter(|&row| {
                                    contains(
                                        &codes[offsets[row] as usize..offsets[row + 1] as usize],
                                        &kmp,
                                    )
                                }));
                            }
                            StaticDispatch::ScanFindingIndex => unsafe {
                                original_avx512_candidates(codes, &offsets, cover, &mut rows);
                                rows.retain(|&row| {
                                    contains(
                                        &codes[offsets[row] as usize..offsets[row + 1] as usize],
                                        &kmp,
                                    )
                                });
                            },
                            StaticDispatch::HierarchicalMidcut => unsafe {
                                summarize_block!(512, codes, cover, &mut hierarchy_summary);
                                refine_live_block_positions::<512>(
                                    codes,
                                    cover,
                                    &hierarchy_summary,
                                    &mut positions,
                                );
                                localized_kmp(
                                    codes, &offsets, &positions, &windows, &kmp, &mut rows,
                                );
                            },
                        }
                    });
                    measure_segment!(6, "full-kmp", {
                        rows.clear();
                        rows.extend((0..offsets.len() - 1).filter(|&row| {
                            contains(
                                &codes[offsets[row] as usize..offsets[row + 1] as usize],
                                &kmp,
                            )
                        }));
                    });
                    measure_segment!(7, "gated", {
                        if row_match_fraction >= 0.001 {
                            rows.clear();
                            rows.extend((0..offsets.len() - 1).filter(|&row| {
                                contains(
                                    &codes[offsets[row] as usize..offsets[row + 1] as usize],
                                    &kmp,
                                )
                            }));
                        } else {
                            match selected {
                                StaticDispatch::DirectKmp => {
                                    rows.clear();
                                    rows.extend((0..offsets.len() - 1).filter(|&row| {
                                        contains(
                                            &codes
                                                [offsets[row] as usize..offsets[row + 1] as usize],
                                            &kmp,
                                        )
                                    }));
                                }
                                StaticDispatch::ScanFindingIndex => unsafe {
                                    original_avx512_candidates(codes, &offsets, cover, &mut rows);
                                    rows.retain(|&row| {
                                        contains(
                                            &codes
                                                [offsets[row] as usize..offsets[row + 1] as usize],
                                            &kmp,
                                        )
                                    });
                                },
                                StaticDispatch::HierarchicalMidcut => unsafe {
                                    summarize_block!(512, codes, cover, &mut hierarchy_summary);
                                    refine_live_block_positions::<512>(
                                        codes,
                                        cover,
                                        &hierarchy_summary,
                                        &mut positions,
                                    );
                                    localized_kmp(
                                        codes, &offsets, &positions, &windows, &kmp, &mut rows,
                                    );
                                },
                            }
                        }
                    });
                }
                for slot in 0..run_samples.len() {
                    run_samples[slot].push(run_totals[slot]);
                }
            }
            let totals: [u128; 8] = std::array::from_fn(|slot| {
                *run_samples[slot].iter().min().expect("at least one run")
            });
            let oracle = [
                (totals[0], "scan-finding-index"),
                (totals[1], "superblock"),
                (totals[2], "superblock-hierarchical"),
                (totals[3], "superblock-hierarchical-length"),
                (totals[4], "superblock-hierarchical-midcut"),
                (totals[6], "full-kmp"),
            ]
            .into_iter()
            .min_by_key(|&(elapsed, _)| elapsed)
            .unwrap();
            let model_slot = match selected {
                StaticDispatch::DirectKmp => 6,
                StaticDispatch::ScanFindingIndex => 0,
                StaticDispatch::HierarchicalMidcut => 4,
            };
            let model_ns = totals[model_slot];
            let model_regret_pct = if oracle.0 == 0 {
                0.0
            } else {
                (model_ns - oracle.0) as f64 * 100.0 / oracle.0 as f64
            };

            println!(
                "SEGMENTED_{}_{}\tsegments={}\tmax_code_bytes={}\tsuperblock_codes={}\trow_matches={}\trow_match_fraction={:.8}\tscan_finding_index_ms={:.6}\tsuperblock_ms={:.6}\thierarchical_ms={:.6}\thierarchical_loose_length_ms={:.6}\tmidcut_ms={:.6}\tdispatch_algorithm={}\tdispatch_ms={:.6}\tfull_kmp_ms={:.6}\tgated_algorithm={}\tgated_ms={:.6}\toracle_algorithm={}\toracle_ms={:.6}\tmodel_ms={:.6}\tmodel_regret_pct={:.4}",
                segment_bytes,
                label(&query),
                segment_rows.len(),
                max_segment_code_bytes,
                summary_codes,
                exact_match_count,
                row_match_fraction,
                totals[0] as f64 / 1_000_000.0,
                totals[1] as f64 / 1_000_000.0,
                totals[2] as f64 / 1_000_000.0,
                totals[3] as f64 / 1_000_000.0,
                totals[4] as f64 / 1_000_000.0,
                selected.name(),
                totals[5] as f64 / 1_000_000.0,
                totals[6] as f64 / 1_000_000.0,
                gated_algorithm,
                totals[7] as f64 / 1_000_000.0,
                oracle.1,
                oracle.0 as f64 / 1_000_000.0,
                model_ns as f64 / 1_000_000.0,
                model_regret_pct,
            );
            continue;
        }

        let mut verifier = BytesVerifier::new(&query);
        let mut out = Vec::new();
        prefilter_candidates(scan_codes, scan_offsets, &analysis, &mut out).unwrap();
        let candidates = out.len();
        let mut kmp_out = out.clone();
        kmp_out.retain(|&row| contains(view.row_codes(row), &kmp));
        verifier.retain(view, &mut out);
        let matches = out.len();
        assert_eq!(kmp_out, out);

        let blocks = scan_codes.len().div_ceil(summary_codes);
        let mut summary = vec![0u64; blocks.div_ceil(64)];
        // SAFETY: AVX-512BW was detected at entry.
        unsafe { summarize!(scan_codes, cover, &mut summary) };
        let mut block_out = Vec::new();
        block_candidates!(&summary, scan_offsets, &mut block_out);
        let block_candidates = block_out.len();
        let mut block_kmp = block_out.clone();
        block_kmp.retain(|&row| contains(view.row_codes(row), &kmp));
        verifier.retain(view, &mut block_out);
        assert_eq!(block_kmp, out);
        assert_eq!(block_out, out);

        if ((measure_original && query == b"google") || measure_original_all || named_only)
            && !dispatch_only
        {
            macro_rules! bench_original {
                ($name:literal, $scan:ident) => {{
                    let mut original = Vec::new();
                    let mut samples = Vec::with_capacity(reps);
                    for _ in 0..reps {
                        let start = Instant::now();
                        unsafe { $scan(scan_codes, scan_offsets, cover, &mut original) };
                        let stage_ns = start.elapsed().as_nanos();
                        let original_candidates = original.len();
                        // The production policy intentionally returns every row when a
                        // cover is too dense to be profitable.  The named scan still
                        // materializes only exact cover-hit rows, so candidate lists
                        // need not be identical for high-selectivity LIKE workloads.
                        // Correctness is checked after exact verification below.

                        let mut kmp_rows = original.clone();
                        let start = Instant::now();
                        kmp_rows.retain(|&row| contains(view.row_codes(row), &kmp));
                        let kmp_ns = start.elapsed().as_nanos();

                        let mut memmem_rows = original.clone();
                        let start = Instant::now();
                        verifier.retain(view, &mut memmem_rows);
                        let memmem_ns = start.elapsed().as_nanos();

                        assert_eq!(kmp_rows, out);
                        assert_eq!(memmem_rows, out);
                        samples.push((stage_ns, kmp_ns, memmem_ns, original_candidates));
                    }
                    let stage_ns = samples.iter().map(|sample| sample.0).min().unwrap();
                    let kmp_ns = samples.iter().map(|sample| sample.1).min().unwrap();
                    let memmem_ns = samples.iter().map(|sample| sample.2).min().unwrap();
                    let original_candidates = samples[0].3;
                    println!(
                        "ORIGINAL_{}_{}\tdecoded_bytes={decoded_bytes}\tcandidates={original_candidates}\tstage_ms={:.6}\tprecise_kmp_ms={:.6}\te2e_kmp_ms={:.6}\tprecise_memmem_ms={:.6}\te2e_memmem_ms={:.6}",
                        $name,
                        label(&query),
                        stage_ns as f64 / 1_000_000.0,
                        kmp_ns as f64 / 1_000_000.0,
                        (stage_ns + kmp_ns) as f64 / 1_000_000.0,
                        memmem_ns as f64 / 1_000_000.0,
                        (stage_ns + memmem_ns) as f64 / 1_000_000.0,
                    );
                }};
            }
            if measure_original_all && !named_only {
                bench_original!("AVX2", original_avx2_candidates);
            }
            bench_original!("AVX512", original_avx512_candidates);
        }

        if ((std::env::var_os("ONPAIR_HIER512").is_some()
            || std::env::var_os("ONPAIR_HIER_ALL").is_some())
            || named_only)
            && !dispatch_only
        {
            let mut hierarchy_results = Vec::new();
            macro_rules! bench_hierarchy {
                ($block:literal) => {{
                    let blocks = scan_codes.len().div_ceil($block);
                    let mut hierarchy_summary = vec![0u64; blocks.div_ceil(64)];
                    let mut refined = Vec::new();
                    let mut samples = Vec::with_capacity(reps);
                    for _ in 0..reps {
                        let start = Instant::now();
                        match hierarchy_kernel.as_str() {
                            "intrinsics-avx512" => unsafe {
                                summarize_block!(
                                    $block,
                                    black_box(scan_codes),
                                    black_box(cover),
                                    black_box(&mut hierarchy_summary)
                                )
                            },
                            "intrinsics-avx2" => unsafe {
                                summarize_superblocks_avx2_p2r2::<$block>(
                                    black_box(scan_codes),
                                    black_box(cover),
                                    black_box(&mut hierarchy_summary),
                                )
                            },
                            "autovec" => summarize_superblocks_autovec::<$block, 2, 2>(
                                black_box(scan_codes),
                                black_box(cover),
                                black_box(&mut hierarchy_summary),
                            ),
                            other => panic!("unknown ONPAIR_HIER_KERNEL={other}"),
                        }
                        let coarse_ns = start.elapsed().as_nanos();
                        let start = Instant::now();
                        match hierarchy_kernel.as_str() {
                            "intrinsics-avx512" => unsafe {
                                refine_rows_block!(
                                    $block,
                                    black_box(scan_codes),
                                    black_box(scan_offsets),
                                    black_box(cover),
                                    black_box(&hierarchy_summary),
                                    black_box(&mut refined)
                                )
                            },
                            "intrinsics-avx2" => unsafe {
                                refine_live_blocks_avx2_p2r2::<$block>(
                                    black_box(scan_codes),
                                    black_box(scan_offsets),
                                    black_box(cover),
                                    black_box(&hierarchy_summary),
                                    black_box(&mut refined),
                                )
                            },
                            "autovec" => refine_live_blocks_autovec_p2r2::<$block>(
                                black_box(scan_codes),
                                black_box(scan_offsets),
                                black_box(cover),
                                black_box(&hierarchy_summary),
                                black_box(&mut refined),
                            ),
                            _ => unreachable!(),
                        }
                        let refine_ns = start.elapsed().as_nanos();
                        let refined_candidates = refined.len();

                        let mut kmp_rows = refined.clone();
                        let start = Instant::now();
                        kmp_rows.retain(|&row| {
                            contains(black_box(view.row_codes(row)), black_box(&kmp))
                        });
                        let kmp_ns = start.elapsed().as_nanos();

                        let mut memmem_rows = refined.clone();
                        let start = Instant::now();
                        verifier.retain(black_box(view), black_box(&mut memmem_rows));
                        let memmem_ns = start.elapsed().as_nanos();

                        assert_eq!(kmp_rows, out);
                        assert_eq!(memmem_rows, out);
                        let minimum_codes = query.len().div_ceil(MAX_TOKEN_SIZE);
                        let mut length_rows = refined.clone();
                        let start = Instant::now();
                        length_rows.retain(|&row| {
                            (scan_offsets[row + 1] - scan_offsets[row]) as usize >= minimum_codes
                        });
                        let length_ns = start.elapsed().as_nanos();
                        let length_candidates = length_rows.len();
                        let mut length_memmem_rows = length_rows.clone();
                        let start = Instant::now();
                        length_rows.retain(|&row| {
                            contains(black_box(view.row_codes(row)), black_box(&kmp))
                        });
                        let length_kmp_ns = start.elapsed().as_nanos();
                        let start = Instant::now();
                        verifier.retain(black_box(view), black_box(&mut length_memmem_rows));
                        let length_memmem_ns = start.elapsed().as_nanos();
                        assert_eq!(length_rows, out);
                        assert_eq!(length_memmem_rows, out);
                        let loose_minimum_codes = query.len() / MAX_TOKEN_SIZE;
                        let mut loose_rows = refined.clone();
                        let start = Instant::now();
                        loose_rows.retain(|&row| {
                            (scan_offsets[row + 1] - scan_offsets[row]) as usize
                                >= loose_minimum_codes
                        });
                        let loose_ns = start.elapsed().as_nanos();
                        let loose_candidates = loose_rows.len();
                        let mut loose_memmem_rows = loose_rows.clone();
                        let start = Instant::now();
                        loose_rows.retain(|&row| {
                            contains(black_box(view.row_codes(row)), black_box(&kmp))
                        });
                        let loose_kmp_ns = start.elapsed().as_nanos();
                        let start = Instant::now();
                        verifier.retain(black_box(view), black_box(&mut loose_memmem_rows));
                        let loose_memmem_ns = start.elapsed().as_nanos();
                        assert_eq!(loose_rows, out);
                        assert_eq!(loose_memmem_rows, out);
                        samples.push((
                            coarse_ns,
                            refine_ns,
                            kmp_ns,
                            memmem_ns,
                            refined_candidates,
                            length_ns,
                            length_kmp_ns,
                            length_memmem_ns,
                            length_candidates,
                            loose_ns,
                            loose_kmp_ns,
                            loose_memmem_ns,
                            loose_candidates,
                        ));
                    }
                    let coarse_ns = samples.iter().map(|sample| sample.0).min().unwrap();
                    let refine_ns = samples.iter().map(|sample| sample.1).min().unwrap();
                    let kmp_ns = samples.iter().map(|sample| sample.2).min().unwrap();
                    let memmem_ns = samples.iter().map(|sample| sample.3).min().unwrap();
                    let refined_candidates = samples[0].4;
                    let length_ns = samples.iter().map(|sample| sample.5).min().unwrap();
                    let length_kmp_ns = samples.iter().map(|sample| sample.6).min().unwrap();
                    let length_memmem_ns = samples.iter().map(|sample| sample.7).min().unwrap();
                    let length_candidates = samples[0].8;
                    let loose_ns = samples.iter().map(|sample| sample.9).min().unwrap();
                    let loose_kmp_ns = samples.iter().map(|sample| sample.10).min().unwrap();
                    let loose_memmem_ns = samples.iter().map(|sample| sample.11).min().unwrap();
                    let loose_candidates = samples[0].12;
                    let live_blocks: usize = hierarchy_summary
                        .iter()
                        .map(|word| word.count_ones() as usize)
                        .sum();
                    println!(
                        "HIER{}_{}_{}\tlive_blocks={live_blocks}\tcandidates={refined_candidates}\tcoarse_ms={:.6}\trefine_ms={:.6}\tprecise_kmp_ms={:.6}\te2e_kmp_ms={:.6}\tprecise_memmem_ms={:.6}\te2e_memmem_ms={:.6}",
                        $block,
                        hierarchy_kernel,
                        label(&query),
                        coarse_ns as f64 / 1_000_000.0,
                        refine_ns as f64 / 1_000_000.0,
                        kmp_ns as f64 / 1_000_000.0,
                        (coarse_ns + refine_ns + kmp_ns) as f64 / 1_000_000.0,
                        memmem_ns as f64 / 1_000_000.0,
                        (coarse_ns + refine_ns + memmem_ns) as f64 / 1_000_000.0,
                    );
                    println!(
                        "HIER_LEN_LOOSE{}_{}_{}\tcandidates={loose_candidates}\tlength_ms={:.6}\tprecise_kmp_ms={:.6}\te2e_kmp_ms={:.6}\tprecise_memmem_ms={:.6}\te2e_memmem_ms={:.6}",
                        $block,
                        hierarchy_kernel,
                        label(&query),
                        loose_ns as f64 / 1_000_000.0,
                        loose_kmp_ns as f64 / 1_000_000.0,
                        (coarse_ns + refine_ns + loose_ns + loose_kmp_ns) as f64 / 1_000_000.0,
                        loose_memmem_ns as f64 / 1_000_000.0,
                        (coarse_ns + refine_ns + loose_ns + loose_memmem_ns) as f64 / 1_000_000.0,
                    );
                    println!(
                        "HIER_LEN{}_{}_{}\tcandidates={length_candidates}\tlength_ms={:.6}\tprecise_kmp_ms={:.6}\te2e_kmp_ms={:.6}\tprecise_memmem_ms={:.6}\te2e_memmem_ms={:.6}",
                        $block,
                        hierarchy_kernel,
                        label(&query),
                        length_ns as f64 / 1_000_000.0,
                        length_kmp_ns as f64 / 1_000_000.0,
                        (coarse_ns + refine_ns + length_ns + length_kmp_ns) as f64 / 1_000_000.0,
                        length_memmem_ns as f64 / 1_000_000.0,
                        (coarse_ns + refine_ns + length_ns + length_memmem_ns) as f64 / 1_000_000.0,
                    );
                    if named_only {
                        let mut windows = vec![(query.len(), query.len()); view.dict.num_tokens()];
                        for window in analysis.probe_windows() {
                            windows[window.token() as usize] =
                                (window.before_codes(), window.after_codes());
                        }
                        let finder = Finder::new(&query);
                        let mut positions = Vec::new();
                        let mut localized_rows = Vec::new();
                        let mut localized_memmem_rows = Vec::new();
                        let mut scratch = Vec::new();
                        let mut localized_samples = Vec::with_capacity(reps);
                        for _ in 0..reps {
                            let start = Instant::now();
                            unsafe {
                                match (cover.points().len(), cover.ranges().len()) {
                                    (0, 1) => refine_live_block_positions_fixed::<$block, 0, 1>(
                                        black_box(scan_codes), black_box(cover), black_box(&hierarchy_summary), black_box(&mut positions)),
                                    (1, 1) => refine_live_block_positions_fixed::<$block, 1, 1>(
                                        black_box(scan_codes), black_box(cover), black_box(&hierarchy_summary), black_box(&mut positions)),
                                    (1, 2) => refine_live_block_positions_fixed::<$block, 1, 2>(
                                        black_box(scan_codes), black_box(cover), black_box(&hierarchy_summary), black_box(&mut positions)),
                                    (1, 3) => refine_live_block_positions_fixed::<$block, 1, 3>(
                                        black_box(scan_codes), black_box(cover), black_box(&hierarchy_summary), black_box(&mut positions)),
                                    (2, 0) => refine_live_block_positions_fixed::<$block, 2, 0>(
                                        black_box(scan_codes), black_box(cover), black_box(&hierarchy_summary), black_box(&mut positions)),
                                    (2, 2) => refine_live_block_positions_fixed::<$block, 2, 2>(
                                        black_box(scan_codes),
                                        black_box(cover),
                                        black_box(&hierarchy_summary),
                                        black_box(&mut positions),
                                    ),
                                    (3, 2) => refine_live_block_positions_fixed::<$block, 3, 2>(
                                        black_box(scan_codes),
                                        black_box(cover),
                                        black_box(&hierarchy_summary),
                                        black_box(&mut positions),
                                    ),
                                    (3, 1) => refine_live_block_positions_fixed::<$block, 3, 1>(
                                        black_box(scan_codes), black_box(cover), black_box(&hierarchy_summary), black_box(&mut positions)),
                                    (4, 0) => refine_live_block_positions_fixed::<$block, 4, 0>(
                                        black_box(scan_codes), black_box(cover), black_box(&hierarchy_summary), black_box(&mut positions)),
                                    (10, 2) => refine_live_block_positions_fixed::<$block, 10, 2>(
                                        black_box(scan_codes), black_box(cover), black_box(&hierarchy_summary), black_box(&mut positions)),
                                    (12, 5) => refine_live_block_positions_fixed::<$block, 12, 5>(
                                        black_box(scan_codes), black_box(cover), black_box(&hierarchy_summary), black_box(&mut positions)),
                                    (25, 5) => refine_live_block_positions_fixed::<$block, 25, 5>(
                                        black_box(scan_codes), black_box(cover), black_box(&hierarchy_summary), black_box(&mut positions)),
                                    _ => refine_live_block_positions::<$block>(
                                        black_box(scan_codes),
                                        black_box(cover),
                                        black_box(&hierarchy_summary),
                                        black_box(&mut positions),
                                    ),
                                }
                            };
                            let position_ns = start.elapsed().as_nanos();

                            let start = Instant::now();
                            let kmp_codes = localized_kmp(
                                black_box(scan_codes),
                                black_box(scan_offsets),
                                black_box(&positions),
                                black_box(&windows),
                                black_box(&kmp),
                                black_box(&mut localized_rows),
                            );
                            let localized_kmp_ns = start.elapsed().as_nanos();

                            let start = Instant::now();
                            let memmem_codes = localized_memmem(
                                black_box(scan_codes),
                                black_box(scan_offsets),
                                black_box(view.dict),
                                black_box(&positions),
                                black_box(&windows),
                                black_box(&finder),
                                black_box(&mut scratch),
                                black_box(&mut localized_memmem_rows),
                            );
                            let localized_memmem_ns = start.elapsed().as_nanos();
                            assert_eq!(localized_rows, out);
                            assert_eq!(localized_memmem_rows, out);
                            localized_samples.push((
                                position_ns,
                                localized_kmp_ns,
                                localized_memmem_ns,
                                kmp_codes,
                                memmem_codes,
                                positions.len(),
                            ));
                        }
                        let position_ns = localized_samples
                            .iter()
                            .map(|sample| sample.0)
                            .min()
                            .unwrap();
                        let localized_kmp_ns = localized_samples
                            .iter()
                            .map(|sample| sample.1)
                            .min()
                            .unwrap();
                        let localized_memmem_ns = localized_samples
                            .iter()
                            .map(|sample| sample.2)
                            .min()
                            .unwrap();
                        println!(
                            "HIER_MIDCUT{}_{}\thit_positions={}\tkmp_window_codes={}\tmemmem_window_codes={}\tcoarse_ms={:.6}\tposition_refine_ms={:.6}\tlocalized_kmp_ms={:.6}\te2e_kmp_ms={:.6}\tlocalized_memmem_ms={:.6}\te2e_memmem_ms={:.6}",
                            $block,
                            label(&query),
                            localized_samples[0].5,
                            localized_samples[0].3,
                            localized_samples[0].4,
                            coarse_ns as f64 / 1_000_000.0,
                            position_ns as f64 / 1_000_000.0,
                            localized_kmp_ns as f64 / 1_000_000.0,
                            (coarse_ns + position_ns + localized_kmp_ns) as f64 / 1_000_000.0,
                            localized_memmem_ns as f64 / 1_000_000.0,
                            (coarse_ns + position_ns + localized_memmem_ns) as f64 / 1_000_000.0,
                        );
                    }
                    hierarchy_results.push(($block, coarse_ns + refine_ns + kmp_ns));
                }};
            }
            if std::env::var_os("ONPAIR_HIER_ALL").is_some() {
                bench_hierarchy!(128);
                bench_hierarchy!(256);
                bench_hierarchy!(512);
                bench_hierarchy!(1024);
                bench_hierarchy!(2048);
                bench_hierarchy!(4096);
            } else {
                bench_hierarchy!(512);
            }
            if std::env::var_os("ONPAIR_HIER_ALL").is_some() {
                let measured = hierarchy_results
                    .iter()
                    .min_by_key(|result| result.1)
                    .unwrap();
                let measured_block = measured.0;
                let frequency = analysis.covered_fraction();
                let predicted_block = if frequency < 0.000_025 {
                    1024
                } else if frequency < 0.000_150 {
                    512
                } else {
                    256
                };
                let predicted_ns = hierarchy_results
                    .iter()
                    .find(|result| result.0 == predicted_block)
                    .unwrap()
                    .1;
                println!(
                    "HIER_PREDICT_{}\tcover_frac={frequency:.8}\tpredicted_block={predicted_block}\tmeasured_block={measured_block}\tpredicted_e2e_ms={:.6}\tmeasured_e2e_ms={:.6}",
                    label(&query),
                    predicted_ns as f64 / 1_000_000.0,
                    measured.1 as f64 / 1_000_000.0,
                );
            }
        }

        if std::env::var_os("ONPAIR_HIER_ONLY").is_some() {
            continue;
        }

        if named_only {
            if !dispatch_only {
                let mut named_samples = Vec::with_capacity(reps);
                for _ in 0..reps {
                    let start = Instant::now();
                    unsafe {
                        summarize!(
                            black_box(scan_codes),
                            black_box(cover),
                            black_box(&mut summary)
                        )
                    };
                    let summary_ns = start.elapsed().as_nanos();

                    let start = Instant::now();
                    block_candidates!(&summary, scan_offsets, &mut block_out);
                    let materialize_ns = start.elapsed().as_nanos();

                    let mut kmp_rows = block_out.clone();
                    let start = Instant::now();
                    kmp_rows
                        .retain(|&row| contains(black_box(view.row_codes(row)), black_box(&kmp)));
                    let kmp_ns = start.elapsed().as_nanos();
                    assert_eq!(kmp_rows, out);

                    let mut memmem_rows = block_out.clone();
                    let start = Instant::now();
                    verifier.retain(black_box(view), black_box(&mut memmem_rows));
                    let memmem_ns = start.elapsed().as_nanos();
                    assert_eq!(memmem_rows, out);
                    named_samples.push((summary_ns, materialize_ns, kmp_ns, memmem_ns));
                }
                let summary_ns = named_samples.iter().map(|sample| sample.0).min().unwrap();
                let materialize_ns = named_samples.iter().map(|sample| sample.1).min().unwrap();
                let kmp_ns = named_samples.iter().map(|sample| sample.2).min().unwrap();
                let memmem_ns = named_samples.iter().map(|sample| sample.3).min().unwrap();
                println!(
                    "SUPERBLOCK{}_{}\tblock_candidates={block_candidates}\tstage_ms={:.6}\tmaterialize_rows_ms={:.6}\tprecise_kmp_ms={:.6}\te2e_kmp_ms={:.6}\tprecise_memmem_ms={:.6}\te2e_memmem_ms={:.6}",
                    summary_codes,
                    label(&query),
                    summary_ns as f64 / 1_000_000.0,
                    materialize_ns as f64 / 1_000_000.0,
                    kmp_ns as f64 / 1_000_000.0,
                    (summary_ns + materialize_ns + kmp_ns) as f64 / 1_000_000.0,
                    memmem_ns as f64 / 1_000_000.0,
                    (summary_ns + materialize_ns + memmem_ns) as f64 / 1_000_000.0,
                );
            }

            // Measure the complete statically-dispatched KMP pipeline.  The
            // decision uses only the already-computed query cover frequency;
            // it never observes candidates or matches from a preliminary scan.
            let mut dispatch_summary = vec![0u64; scan_codes.len().div_ceil(512).div_ceil(64)];
            let mut dispatch_rows = Vec::new();
            let mut dispatch_positions = Vec::new();
            let dispatch_windows = {
                let mut windows = vec![(query.len(), query.len()); view.dict.num_tokens()];
                for window in analysis.probe_windows() {
                    windows[window.token() as usize] =
                        (window.before_codes(), window.after_codes());
                }
                windows
            };
            let mut dispatch_samples = Vec::with_capacity(reps);
            for _ in 0..reps {
                let start = Instant::now();
                let selected = static_dispatch(
                    black_box(analysis.covered_fraction()),
                    black_box(cover.points().len()),
                    black_box(cover.ranges().len()),
                );
                match selected {
                    StaticDispatch::DirectKmp => {
                        dispatch_rows.clear();
                        dispatch_rows.extend((0..scan_rows).filter(|&row| {
                            contains(black_box(view.row_codes(row)), black_box(&kmp))
                        }));
                    }
                    StaticDispatch::ScanFindingIndex => unsafe {
                        original_avx512_candidates(
                            black_box(scan_codes),
                            black_box(scan_offsets),
                            black_box(cover),
                            black_box(&mut dispatch_rows),
                        );
                        dispatch_rows.retain(|&row| {
                            contains(black_box(view.row_codes(row)), black_box(&kmp))
                        });
                    },
                    StaticDispatch::HierarchicalMidcut => unsafe {
                        summarize_block!(
                            512,
                            black_box(scan_codes),
                            black_box(cover),
                            black_box(&mut dispatch_summary)
                        );
                        refine_live_block_positions::<512>(
                            black_box(scan_codes),
                            black_box(cover),
                            black_box(&dispatch_summary),
                            black_box(&mut dispatch_positions),
                        );
                        localized_kmp(
                            black_box(scan_codes),
                            black_box(scan_offsets),
                            black_box(&dispatch_positions),
                            black_box(&dispatch_windows),
                            black_box(&kmp),
                            black_box(&mut dispatch_rows),
                        );
                    },
                }
                let elapsed_ns = start.elapsed().as_nanos();
                assert_eq!(dispatch_rows, out);
                dispatch_samples.push(elapsed_ns);
            }
            let selected = static_dispatch(
                analysis.covered_fraction(),
                cover.points().len(),
                cover.ranges().len(),
            );
            let dispatch_ns = *dispatch_samples.iter().min().unwrap();
            println!(
                "STATIC_DISPATCH_{}\talgorithm={}\tcover_frac={:.8}\te2e_kmp_ms={:.6}",
                label(&query),
                selected.name(),
                analysis.covered_fraction(),
                dispatch_ns as f64 / 1_000_000.0,
            );
            continue;
        }

        let mut full_kmp = Vec::new();
        let mut full_memmem = Vec::new();
        let mut samples: [Vec<u128>; 8] = std::array::from_fn(|_| Vec::with_capacity(reps));
        for _ in 0..reps {
            full_kmp.clear();
            let start = Instant::now();
            full_kmp.extend(
                (0..scan_rows)
                    .filter(|&row| contains(black_box(view.row_codes(row)), black_box(&kmp))),
            );
            samples[0].push(start.elapsed().as_nanos());

            full_memmem.clear();
            let start = Instant::now();
            for row in 0..scan_rows {
                if verifier.contains_row(black_box(view), row) {
                    full_memmem.push(row);
                }
            }
            samples[1].push(start.elapsed().as_nanos());

            out.clear();
            let start = Instant::now();
            prefilter_candidates(
                black_box(scan_codes),
                black_box(scan_offsets),
                black_box(&analysis),
                black_box(&mut out),
            )
            .unwrap();
            samples[2].push(start.elapsed().as_nanos());
            let start = Instant::now();
            verifier.retain(black_box(view), black_box(&mut out));
            samples[4].push(start.elapsed().as_nanos());

            out.clear();
            prefilter_candidates(scan_codes, scan_offsets, &analysis, &mut out).unwrap();
            let start = Instant::now();
            out.retain(|&row| contains(black_box(view.row_codes(row)), black_box(&kmp)));
            samples[3].push(start.elapsed().as_nanos());

            let start = Instant::now();
            // SAFETY: AVX-512BW was detected at entry.
            unsafe {
                summarize!(
                    black_box(scan_codes),
                    black_box(cover),
                    black_box(&mut summary)
                )
            };
            samples[5].push(start.elapsed().as_nanos());

            let start = Instant::now();
            block_candidates!(&summary, scan_offsets, &mut block_out);
            block_out.retain(|&row| contains(black_box(view.row_codes(row)), black_box(&kmp)));
            samples[6].push(start.elapsed().as_nanos());

            let start = Instant::now();
            block_candidates!(&summary, scan_offsets, &mut block_out);
            verifier.retain(black_box(view), black_box(&mut block_out));
            samples[7].push(start.elapsed().as_nanos());

            assert_eq!(black_box(&full_kmp), black_box(&full_memmem));
            assert_eq!(black_box(&full_kmp), black_box(&out));
            assert_eq!(black_box(&full_kmp), black_box(&block_out));
        }
        let best_ns = samples.map(|mut values| {
            values.sort_unstable();
            values[0] as f64
        });
        for (total, best) in batch_ns.iter_mut().zip(best_ns) {
            *total += best;
        }
        measured += 1;
        let band = selectivity_band(analysis.covered_fraction());
        for (total, best) in selectivity_ns[band].iter_mut().zip(best_ns) {
            *total += best;
        }
        selectivity_queries[band] += 1;
        println!(
            "{}\t{}\t{}\t{}\t{:.8}\t{}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}",
            label(&query),
            query.len(),
            cover.points().len(),
            cover.ranges().len(),
            analysis.covered_fraction(),
            candidates,
            block_candidates,
            matches,
            best_ns[0] / 1_000_000.0,
            best_ns[1] / 1_000_000.0,
            best_ns[2] / 1_000_000.0,
            (best_ns[2] + best_ns[3]) / 1_000_000.0,
            (best_ns[2] + best_ns[4]) / 1_000_000.0,
            best_ns[5] / 1_000_000.0,
            (best_ns[5] + best_ns[6]) / 1_000_000.0,
            (best_ns[5] + best_ns[7]) / 1_000_000.0,
        );
    }
    for (band, label) in [
        "tiny <0.01%",
        "small 0.01-0.1%",
        "medium 0.1-1%",
        "large >=1%",
    ]
    .into_iter()
    .enumerate()
    {
        let queries = selectivity_queries[band];
        let ns = selectivity_ns[band];
        println!(
            "SELECTIVITY {label}\t{queries}\t-\t-\t-\t-\t-\t-\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            ns[0] / 1_000_000.0,
            ns[1] / 1_000_000.0,
            ns[2] / 1_000_000.0,
            (ns[2] + ns[3]) / 1_000_000.0,
            (ns[2] + ns[4]) / 1_000_000.0,
            ns[5] / 1_000_000.0,
            (ns[5] + ns[6]) / 1_000_000.0,
            (ns[5] + ns[7]) / 1_000_000.0,
        );
    }
    println!(
        "BATCH\t{measured}\t-\t-\t-\t-\t-\t-\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
        batch_ns[0] / 1_000_000.0,
        batch_ns[1] / 1_000_000.0,
        batch_ns[2] / 1_000_000.0,
        (batch_ns[2] + batch_ns[3]) / 1_000_000.0,
        (batch_ns[2] + batch_ns[4]) / 1_000_000.0,
        batch_ns[5] / 1_000_000.0,
        (batch_ns[5] + batch_ns[6]) / 1_000_000.0,
        (batch_ns[5] + batch_ns[7]) / 1_000_000.0,
    );
}

fn selectivity_band(fraction: f64) -> usize {
    if fraction < 0.0001 {
        0
    } else if fraction < 0.001 {
        1
    } else if fraction < 0.01 {
        2
    } else {
        3
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StaticDispatch {
    DirectKmp,
    ScanFindingIndex,
    HierarchicalMidcut,
}

impl StaticDispatch {
    fn name(self) -> &'static str {
        match self {
            Self::DirectKmp => "full-kmp",
            Self::ScanFindingIndex => "scan-finding-index",
            Self::HierarchicalMidcut => "superblock-hierarchical-midcut",
        }
    }
}

/// Choose the complete search pipeline using query analysis only.  The model
/// deliberately avoids measured row/candidate counts, so dispatch does not
/// require a data scan.
fn static_dispatch(
    covered_fraction: f64,
    point_count: usize,
    range_count: usize,
) -> StaticDispatch {
    // A point is one SIMD comparison; an inclusive range is subtraction plus
    // comparison, so price it as two. This matches the analysis API's cost.
    let comparison_cost = point_count.saturating_add(range_count.saturating_mul(2));
    if comparison_cost >= 200 {
        StaticDispatch::DirectKmp
    } else if covered_fraction >= 0.03 {
        if comparison_cost >= 64 {
            StaticDispatch::DirectKmp
        } else {
            StaticDispatch::ScanFindingIndex
        }
    } else if comparison_cost <= 24 {
        StaticDispatch::HierarchicalMidcut
    } else {
        StaticDispatch::ScanFindingIndex
    }
}

#[cfg(test)]
mod dispatch_model_tests {
    use super::{StaticDispatch, static_dispatch};

    #[test]
    fn dense_or_complex_covers_use_direct_kmp() {
        assert_eq!(static_dispatch(0.03, 64, 0), StaticDispatch::DirectKmp);
        assert_eq!(static_dispatch(0.001, 100, 50), StaticDispatch::DirectKmp);
        assert_eq!(
            static_dispatch(0.03, 20, 0),
            StaticDispatch::ScanFindingIndex
        );
    }

    #[test]
    fn tiny_cuts_localize_and_moderate_cuts_scan() {
        assert_eq!(
            static_dispatch(0.001, 20, 1),
            StaticDispatch::HierarchicalMidcut
        );
        assert_eq!(
            static_dispatch(0.001, 23, 1),
            StaticDispatch::ScanFindingIndex
        );
    }
}

fn queries(rows: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let mut queries: Vec<Vec<u8>> = [
        "google",
        "yandex",
        "facebook",
        "wikipedia",
        "https://",
        "http://",
        ".com",
        ".ru",
        "/search",
        "utm_source",
        "jpg",
        "news",
        "click",
        "2014",
        "index",
        "html",
        "php",
        "youtube",
    ]
    .into_iter()
    .map(|value| value.as_bytes().to_vec())
    .collect();
    for (ordinal, &row_index) in [17usize, 1009, 10_007, 100_003, 500_009, 900_001]
        .iter()
        .enumerate()
    {
        let row = &rows[row_index % rows.len()];
        for &len in &[8usize, 16] {
            if row.len() >= len {
                let span = row.len() - len + 1;
                let start = (row_index.wrapping_mul(37) + ordinal * 11) % span;
                queries.push(row[start..start + len].to_vec());
            }
        }
    }
    queries.sort();
    queries.dedup();
    queries
}

fn label(query: &[u8]) -> String {
    String::from_utf8_lossy(query)
        .replace('\t', "\\t")
        .replace('\n', "\\n")
}

fn read_urls(path: &Path) -> Vec<Vec<u8>> {
    if path.extension().is_some_and(|extension| extension == "csv") {
        let bytes = fs::read(path).expect("read CSV input");
        return bytes
            .split(|&byte| byte == b'\n')
            .skip(1)
            .filter_map(|line| {
                let comma = line.iter().position(|&byte| byte == b',')?;
                let value = &line[comma + 1..];
                (!value.is_empty()).then(|| value.to_vec())
            })
            .collect();
    }
    let mut paths = if path.is_dir() {
        let mut paths: Vec<_> = fs::read_dir(path)
            .expect("read parquet directory")
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "parquet"))
            .collect();
        paths.sort();
        paths
    } else {
        vec![path.to_path_buf()]
    };
    assert!(!paths.is_empty(), "no parquet inputs at {}", path.display());
    let column = std::env::var("ONPAIR_BENCH_COLUMN").unwrap_or_else(|_| "URL".to_string());
    let mut rows = Vec::new();
    for path in paths.drain(..) {
        let file = File::open(&path).expect("open parquet");
        let builder =
            ParquetRecordBatchReaderBuilder::try_new(file).expect("read parquet metadata");
        let schema = builder.schema().clone();
        let picked = schema
            .fields()
            .iter()
            .position(|field| field.name() == &column)
            .unwrap_or_else(|| panic!("column {column:?} missing from {}", path.display()));
        for batch in builder.build().unwrap().flatten() {
            let array = batch.column(picked);
            match array.data_type() {
                arrow_schema::DataType::Utf8 => rows.extend(
                    array
                        .as_string::<i32>()
                        .iter()
                        .map(|value| value.unwrap_or("").as_bytes().to_vec()),
                ),
                arrow_schema::DataType::LargeUtf8 => rows.extend(
                    array
                        .as_string::<i64>()
                        .iter()
                        .map(|value| value.unwrap_or("").as_bytes().to_vec()),
                ),
                arrow_schema::DataType::Binary => rows.extend(
                    array
                        .as_binary::<i32>()
                        .iter()
                        .map(|value| value.unwrap_or_default().to_vec()),
                ),
                arrow_schema::DataType::LargeBinary => rows.extend(
                    array
                        .as_binary::<i64>()
                        .iter()
                        .map(|value| value.unwrap_or_default().to_vec()),
                ),
                data_type => panic!(
                    "column {column:?} in {} has unsupported type {data_type:?}",
                    path.display()
                ),
            }
        }
    }
    rows
}

fn pack(rows: &[Vec<u8>]) -> (Vec<u8>, Vec<u64>) {
    let mut bytes = Vec::with_capacity(rows.iter().map(Vec::len).sum());
    let mut offsets = Vec::with_capacity(rows.len() + 1);
    offsets.push(0);
    for row in rows {
        bytes.extend_from_slice(row);
        offsets.push(bytes.len() as u64);
    }
    (bytes, offsets)
}

fn load_column(path: &Path) -> Column<u64> {
    let dict = CompactDictionary::validate(OwnedDictionaryStorage::new(
        fs::read(path.join("dict.bytes")).unwrap(),
        read_u32(&path.join("dict.offsets.u32")),
    ))
    .unwrap();
    Column {
        dict,
        codes: read_u16(&path.join("codes.u16")),
        row_offsets: read_u64(&path.join("rows.u64")),
    }
}

fn write_u16(path: &Path, values: &[u16]) {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for &value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes).unwrap();
}

fn write_u32(path: &Path, values: &[u32]) {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for &value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes).unwrap();
}

fn write_u64(path: &Path, values: &[u64]) {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for &value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes).unwrap();
}

fn read_u16(path: &Path) -> Vec<u16> {
    fs::read(path)
        .unwrap()
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn read_u32(path: &Path) -> Vec<u32> {
    fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn read_u64(path: &Path) -> Vec<u64> {
    fs::read(path)
        .unwrap()
        .chunks_exact(8)
        .map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}
