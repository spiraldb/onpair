// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Greedy parsing with the lanes in AVX-512 or AVX2 vectors.
//!
//! The scalar lanes of [`lanes`](crate::encoding::lanes) spend most of their
//! time on instructions: each lane runs its own search. Here eight lanes share
//! one AVX-512 vector, or four lanes one AVX2 vector, and every search step is
//! a gather and a compare for all of them. Several vectors advance together,
//! so their gathers overlap.
//!
//! A search finds `q`, the last token whose key is `<=` the window key, by a
//! binary search over the token keys of the window's 2-byte prefix range. The
//! rest follows [`DictionaryMatcher`]: the common prefix of `q` and the window
//! bounds the length, and the prefix chain of `q` gives the token.
//!
//! Rows rarely end, so row completion is a scalar step that runs only when a
//! lane reaches the end of its row.

use std::arch::x86_64::{
    __cpuid, __m128i, __m256i, __m512i, __mmask8, _mm256_add_epi8, _mm256_add_epi64,
    _mm256_and_si256, _mm256_andnot_si256, _mm256_blendv_epi8, _mm256_castpd_si256,
    _mm256_castsi256_pd, _mm256_cmpeq_epi8, _mm256_cmpeq_epi64, _mm256_cmpgt_epi64,
    _mm256_cvtepu32_epi64, _mm256_i64gather_epi32, _mm256_i64gather_epi64, _mm256_loadu_si256,
    _mm256_movemask_pd, _mm256_or_si256, _mm256_sad_epu8, _mm256_set_epi64x, _mm256_set1_epi8,
    _mm256_set1_epi64x, _mm256_setr_epi8, _mm256_setzero_si256, _mm256_shuffle_epi8,
    _mm256_sllv_epi64, _mm256_srli_epi16, _mm256_srli_epi64, _mm256_storeu_si256, _mm256_sub_epi64,
    _mm256_sub_pd, _mm256_xor_si256, _mm512_add_epi64, _mm512_and_si512, _mm512_cmpeq_epu64_mask,
    _mm512_cmpge_epu64_mask, _mm512_cmpgt_epu64_mask, _mm512_cmple_epu64_mask,
    _mm512_cmplt_epu64_mask, _mm512_cmpneq_epu64_mask, _mm512_cvtepi64_epi32,
    _mm512_cvtepu32_epi64, _mm512_loadu_si512, _mm512_lzcnt_epi64, _mm512_mask_add_epi64,
    _mm512_mask_blend_epi64, _mm512_mask_i64gather_epi32, _mm512_mask_i64gather_epi64,
    _mm512_mask_i64scatter_epi32, _mm512_mask_sub_epi64, _mm512_min_epu64, _mm512_popcnt_epi64,
    _mm512_set1_epi64, _mm512_setzero_si512, _mm512_shuffle_epi8, _mm512_sllv_epi64,
    _mm512_srli_epi64, _mm512_storeu_si512, _mm512_sub_epi64, _mm512_xor_si512,
};

use std::sync::OnceLock;

use fearless_simd::Level;
use fearless_simd::x86::{Avx2, Avx512};

use crate::core::offset::Offset;
use crate::core::types::{MAX_TOKEN_SIZE, Token};
use crate::encoding::dictionary_matcher::{DictionaryMatcher, FlatMatcher};
use crate::encoding::lanes::OffsetLike;

/// Lanes per AVX-512 vector.
const WIDTH: usize = 8;

/// AVX-512 vectors that advance together.
const GROUPS: usize = 4;

/// All lanes of a batch.
const LANES: usize = WIDTH * GROUPS;

/// Lanes per AVX2 vector.
const WIDTH_AVX2: usize = 4;

/// AVX2 vectors that advance together.
const GROUPS_AVX2: usize = LANES / WIDTH_AVX2;

/// A vector parser kernel, with the token that proves its target features.
#[derive(Clone, Copy)]
pub(crate) enum Kernel {
    /// Eight lanes per AVX-512 vector.
    Avx512(Avx512),

    /// Four lanes per AVX2 vector.
    Avx2(Avx2),
}

impl Kernel {
    /// The kernel for this CPU, if the vector parser runs on it.
    ///
    /// The parser needs fast gathers. On AMD the scalar lanes are faster or
    /// equal: Zen 3 by about a third with AVX2, Zen 4 by about a third with
    /// AVX-512, and Zen 5 is equal. Intel CPUs are faster with it: Haswell by
    /// 5% to 15% with AVX2, and Ice Lake to Granite Rapids by 30% to 55% with
    /// AVX-512. Tests run it on every CPU with the features, so that its
    /// results stay checked.
    pub(crate) fn detect() -> Option<Self> {
        static INTEL: OnceLock<bool> = OnceLock::new();
        if !(cfg!(test) || *INTEL.get_or_init(is_intel)) {
            return None;
        }

        let level = Level::new();
        level
            .as_avx512()
            .map(Self::Avx512)
            .or_else(|| level.as_avx2().map(Self::Avx2))
    }
}

/// Whether the CPU vendor is Intel.
fn is_intel() -> bool {
    let leaf = __cpuid(0);
    [leaf.ebx, leaf.edx, leaf.ecx]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .eq(*b"GenuineIntel")
}

/// Reusable buffers of the vector parser.
#[derive(Default)]
pub(crate) struct VectorBuffers {
    /// Codes of all lanes, each lane in its own region.
    codes: Vec<Token>,
    /// Per row of the batch: the code count of its lane at the row's end.
    row_ends: Vec<usize>,
}

/// Scalar state of all lanes.
struct Lanes {
    at: [u64; LANES],
    row_end: [u64; LANES],
    count: [u64; LANES],
    /// First code slot of each lane in [`VectorBuffers::codes`].
    region: [u64; LANES],
    row: [usize; LANES],
    last_row: [usize; LANES],
    /// Bit `j` is set while lane `j` has rows to parse.
    active: u32,
}

/// Encode one batch of contiguous rows with all vector lanes.
///
/// # Safety
/// The caller guarantees a valid Arrow pair, and that at least
/// `MAX_TOKEN_SIZE - 1` bytes of `bytes` follow the end of the last row.
pub(crate) unsafe fn parse_batch<I: OffsetLike, O: Offset>(
    kernel: Kernel,
    matcher: &DictionaryMatcher,
    bytes: &[u8],
    offsets: &[I],
    buffers: &mut VectorBuffers,
    codes: &mut Vec<Token>,
    row_offsets: &mut Vec<O>,
) {
    let n = offsets.len() - 1;
    let first = offsets[0].to_usize();
    let total = offsets[n].to_usize() - first;
    debug_assert!(offsets[n].to_usize() + MAX_TOKEN_SIZE - 1 <= bytes.len());

    // Lane `j` parses rows `bounds[j]..bounds[j + 1]`, split by byte count.
    let mut bounds = [0usize; LANES + 1];
    for (j, bound) in bounds.iter_mut().enumerate().skip(1) {
        let target = first + total * j / LANES;
        *bound = offsets.partition_point(|o| o.to_usize() < target).min(n);
    }
    bounds[LANES] = n;

    // One code per byte at most, and a spare slot: a code is stored as 32 bits.
    buffers.codes.resize(total + 2 * LANES, 0);
    buffers.row_ends.resize(n, 0);

    let mut lanes = Lanes {
        at: [0; LANES],
        row_end: [0; LANES],
        count: [0; LANES],
        region: [0; LANES],
        row: [0; LANES],
        last_row: [0; LANES],
        active: 0,
    };
    for j in 0..LANES {
        let (r0, r1) = (bounds[j], bounds[j + 1]);
        let start = offsets[r0].to_usize();
        lanes.at[j] = start as u64;
        lanes.region[j] = (start - first + 2 * j) as u64;
        lanes.row[j] = r0;
        lanes.last_row[j] = r1;
        if r0 < r1 {
            lanes.row_end[j] = offsets[r0 + 1].to_usize() as u64;
            lanes.active |= 1 << j;
        } else {
            lanes.row_end[j] = start as u64;
        }
    }

    // Rows that start empty end before the first step.
    finish_rows(&mut lanes, offsets, &mut buffers.row_ends, u32::MAX);

    // SAFETY: the caller guarantees the readable bytes. The AVX-512 token is
    // the Ice Lake level, which includes BW, CD, VPOPCNTDQ and POPCNT; the
    // AVX2 token includes POPCNT.
    match kernel {
        Kernel::Avx512(_) => unsafe { run(matcher.flat(), bytes, offsets, buffers, &mut lanes) },
        Kernel::Avx2(_) => unsafe { run_avx2(matcher.flat(), bytes, offsets, buffers, &mut lanes) },
    }

    let total_codes: u64 = lanes.count.iter().sum();
    codes.reserve(total_codes as usize);
    for j in 0..LANES {
        let base = codes.len();
        let region = lanes.region[j] as usize;
        codes.extend_from_slice(&buffers.codes[region..region + lanes.count[j] as usize]);
        row_offsets.extend(
            buffers.row_ends[bounds[j]..bounds[j + 1]]
                .iter()
                .map(|&end| O::from_usize(base + end)),
        );
    }
}

/// End the current row of every lane in `done` that is active and has reached
/// the end of its row, and skip the empty rows after it. A lane past its last
/// row turns inactive.
fn finish_rows<I: OffsetLike>(lanes: &mut Lanes, offsets: &[I], row_ends: &mut [usize], done: u32) {
    for j in 0..LANES {
        let bit = 1 << j;
        if done & bit == 0 || lanes.active & bit == 0 {
            continue;
        }
        while lanes.row[j] < lanes.last_row[j] && lanes.at[j] == lanes.row_end[j] {
            row_ends[lanes.row[j]] = lanes.count[j] as usize;
            lanes.row[j] += 1;
            if lanes.row[j] < lanes.last_row[j] {
                lanes.row_end[j] = offsets[lanes.row[j] + 1].to_usize() as u64;
            }
        }
        if lanes.row[j] == lanes.last_row[j] {
            lanes.active &= !bit;
        }
    }
}

/// Byte order reversal within each 64-bit element.
#[inline(always)]
fn bswap_control() -> __m512i {
    let pattern: [u8; 64] = std::array::from_fn(|i| ((i & !7) + 7 - (i & 7)) as u8);
    // SAFETY: the array has 64 readable bytes.
    unsafe { _mm512_loadu_si512(pattern.as_ptr().cast()) }
}

/// Advance all lanes until every lane has parsed its rows.
#[target_feature(enable = "avx512f,avx512bw,avx512cd,avx512vpopcntdq,popcnt")]
unsafe fn run<I: OffsetLike>(
    flat: FlatMatcher,
    bytes: &[u8],
    offsets: &[I],
    buffers: &mut VectorBuffers,
    lanes: &mut Lanes,
) {
    let zero = _mm512_setzero_si512();
    let one = _mm512_set1_epi64(1);
    let low16 = _mm512_set1_epi64(0xFFFF);
    let low32 = _mm512_set1_epi64(0xFFFF_FFFF);
    let max_len = _mm512_set1_epi64(MAX_TOKEN_SIZE as i64);
    let bswap = bswap_control();
    let input = bytes.as_ptr().cast::<i64>();
    let out = buffers.codes.as_mut_ptr().cast::<i32>();

    // SAFETY (all loads below): each array holds 8 lanes.
    let load = |a: &[u64; LANES], g: usize| unsafe {
        _mm512_loadu_si512(a.as_ptr().add(g * WIDTH).cast())
    };
    let mut at: [__m512i; GROUPS] = std::array::from_fn(|g| load(&lanes.at, g));
    let mut row_end: [__m512i; GROUPS] = std::array::from_fn(|g| load(&lanes.row_end, g));
    let mut count: [__m512i; GROUPS] = std::array::from_fn(|g| load(&lanes.count, g));
    let region: [__m512i; GROUPS] = std::array::from_fn(|g| load(&lanes.region, g));

    let mut qh = [zero; GROUPS];
    let mut ql = [zero; GROUPS];
    let mut base = [zero; GROUPS];
    let mut size = [zero; GROUPS];

    while lanes.active != 0 {
        let active: [__mmask8; GROUPS] =
            std::array::from_fn(|g| (lanes.active >> (g * WIDTH)) as __mmask8);

        // Window keys and their 2-byte prefix ranges. Bytes past the window
        // belong to the key: see `DictionaryMatcher::find_all`.
        for g in 0..GROUPS {
            let m = active[g];
            // SAFETY: an active lane reads 16 bytes at a position before its
            // row end, which the caller guarantees readable.
            let (w0, w1) = unsafe {
                (
                    _mm512_mask_i64gather_epi64::<1>(zero, m, at[g], input),
                    _mm512_mask_i64gather_epi64::<1>(
                        zero,
                        m,
                        _mm512_add_epi64(at[g], _mm512_set1_epi64(8)),
                        input,
                    ),
                )
            };
            qh[g] = _mm512_shuffle_epi8(w0, bswap);
            ql[g] = _mm512_shuffle_epi8(w1, bswap);

            // SAFETY: a prefix indexes the bounds, and three pads follow.
            let range = unsafe {
                _mm512_mask_i64gather_epi64::<2>(
                    zero,
                    m,
                    _mm512_srli_epi64::<48>(qh[g]),
                    flat.token_bounds,
                )
            };
            base[g] = _mm512_and_si512(range, low16);
            let last = _mm512_and_si512(_mm512_srli_epi64::<16>(range), low16);
            size[g] = _mm512_add_epi64(_mm512_sub_epi64(last, base[g]), one);
        }

        // Binary search for the last key `<=` the window key. The first key
        // of each range is `<=` it; every lane steps until its range is one.
        loop {
            let mut busy = false;
            for g in 0..GROUPS {
                let m = active[g] & _mm512_cmpgt_epu64_mask(size[g], one);
                busy |= m != 0;
                let half = _mm512_srli_epi64::<1>(size[g]);
                let mid = _mm512_add_epi64(base[g], half);
                // SAFETY: `mid` is a token of the range.
                let (kh, kl) = unsafe {
                    (
                        _mm512_mask_i64gather_epi64::<8>(zero, m, mid, flat.keys_hi),
                        _mm512_mask_i64gather_epi64::<8>(zero, m, mid, flat.keys_lo),
                    )
                };
                let le = _mm512_cmplt_epu64_mask(kh, qh[g])
                    | (_mm512_cmpeq_epu64_mask(kh, qh[g]) & _mm512_cmple_epu64_mask(kl, ql[g]));
                base[g] = _mm512_mask_blend_epi64(m & le, base[g], mid);
                size[g] = _mm512_mask_sub_epi64(size[g], m, size[g], half);
            }
            if !busy {
                break;
            }
        }

        let mut done = 0u32;
        for g in 0..GROUPS {
            let m = active[g];
            let q = base[g];
            // SAFETY: `q` is a token.
            let (fh, fl, chain) = unsafe {
                (
                    _mm512_mask_i64gather_epi64::<8>(zero, m, q, flat.keys_hi),
                    _mm512_mask_i64gather_epi64::<8>(zero, m, q, flat.keys_lo),
                    _mm512_mask_i64gather_epi64::<8>(zero, m, q, flat.chains),
                )
            };

            // Common prefix bytes of `q` and the window key, up to 16.
            let xh = _mm512_xor_si512(fh, qh[g]);
            let xl = _mm512_xor_si512(fl, ql[g]);
            let bits = _mm512_mask_blend_epi64(
                _mm512_cmpneq_epu64_mask(xh, zero),
                _mm512_add_epi64(_mm512_lzcnt_epi64(xl), _mm512_set1_epi64(64)),
                _mm512_lzcnt_epi64(xh),
            );
            let common = _mm512_srli_epi64::<3>(bits);
            let win = _mm512_min_epu64(_mm512_sub_epi64(row_end[g], at[g]), max_len);
            let bound = _mm512_min_epu64(common, win);

            // `bound >= 1` and bit 0 is always set, so `fits` is never zero.
            let lens = _mm512_and_si512(_mm512_srli_epi64::<32>(chain), low16);
            let mask = _mm512_sub_epi64(_mm512_sllv_epi64(one, bound), one);
            let fits = _mm512_and_si512(lens, mask);
            let len = _mm512_sub_epi64(_mm512_set1_epi64(64), _mm512_lzcnt_epi64(fits));

            // The token: `q` itself when every prefix length fits, else the
            // prefix path entry at the rank of the longest fitting length.
            let rank = _mm512_sub_epi64(_mm512_popcnt_epi64(fits), one);
            let path = _mm512_add_epi64(_mm512_and_si512(chain, low32), rank);
            // SAFETY: the path entry exists, and a spare entry follows the last.
            let proper: __m256i = unsafe {
                _mm512_mask_i64gather_epi32::<2>(_mm256_setzero_si256(), m, path, flat.prefix_ids)
            };
            let proper = _mm512_and_si512(_mm512_cvtepu32_epi64(proper), low16);
            let id = _mm512_srli_epi64::<48>(chain);
            let code = _mm512_mask_blend_epi64(_mm512_cmpeq_epu64_mask(fits, lens), proper, id);

            // SAFETY: each lane writes 32 bits at its next code slot, inside
            // its region; the upper half is zero and the next code replaces it.
            unsafe {
                _mm512_mask_i64scatter_epi32::<2>(
                    out,
                    m,
                    _mm512_add_epi64(region[g], count[g]),
                    _mm512_cvtepi64_epi32(code),
                );
            }
            count[g] = _mm512_mask_add_epi64(count[g], m, count[g], one);
            at[g] = _mm512_mask_add_epi64(at[g], m, at[g], len);
            done |= u32::from(m & _mm512_cmpge_epu64_mask(at[g], row_end[g])) << (g * WIDTH);
        }

        if done != 0 {
            for g in 0..GROUPS {
                // SAFETY: each array holds 8 lanes per group.
                unsafe {
                    _mm512_storeu_si512(lanes.at.as_mut_ptr().add(g * WIDTH).cast(), at[g]);
                    _mm512_storeu_si512(lanes.count.as_mut_ptr().add(g * WIDTH).cast(), count[g]);
                }
            }
            finish_rows(lanes, offsets, &mut buffers.row_ends, done);
            for g in 0..GROUPS {
                row_end[g] = load(&lanes.row_end, g);
            }
        }
    }

    for g in 0..GROUPS {
        // SAFETY: each array holds 8 lanes per group.
        unsafe { _mm512_storeu_si512(lanes.count.as_mut_ptr().add(g * WIDTH).cast(), count[g]) };
    }
}

/// Lane masks of AVX2 group `g`: all bits set in each active lane.
#[target_feature(enable = "avx2")]
#[inline]
fn active_avx2(active: u32, g: usize) -> __m256i {
    let bits = _mm256_set_epi64x(8, 4, 2, 1);
    let lanes = _mm256_set1_epi64x(i64::from((active >> (g * WIDTH_AVX2)) & 0xF));
    _mm256_cmpeq_epi64(_mm256_and_si256(lanes, bits), bits)
}

/// Equal leading bytes of two keys in memory order (first byte lowest), up
/// to 8 per 64-bit lane.
#[target_feature(enable = "avx2")]
#[inline]
fn common_bytes_avx2(a: __m256i, b: __m256i) -> __m256i {
    let differ = _mm256_xor_si256(_mm256_cmpeq_epi8(a, b), _mm256_set1_epi8(-1));
    // The lowest bit of the first differing byte, then all bits below it.
    let first = _mm256_and_si256(differ, _mm256_sub_epi64(_mm256_setzero_si256(), differ));
    let below = _mm256_sub_epi64(first, _mm256_set1_epi64x(1));
    _mm256_sad_epu8(
        _mm256_and_si256(below, _mm256_set1_epi8(1)),
        _mm256_setzero_si256(),
    )
}

/// Set bits in each 64-bit lane of values below `2^16`.
#[target_feature(enable = "avx2")]
#[inline]
fn popcnt_avx2(x: __m256i) -> __m256i {
    let table = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3,
        3, 4,
    );
    let nibble = _mm256_set1_epi8(0x0F);
    let low = _mm256_shuffle_epi8(table, _mm256_and_si256(x, nibble));
    let high = _mm256_shuffle_epi8(table, _mm256_and_si256(_mm256_srli_epi16::<4>(x), nibble));
    _mm256_sad_epu8(_mm256_add_epi8(low, high), _mm256_setzero_si256())
}

/// Bit length of each 64-bit lane of values in `1..2^52`.
#[target_feature(enable = "avx2")]
#[inline]
fn bit_len_avx2(x: __m256i) -> __m256i {
    // `2^52 + x` as a double, minus `2^52`, is `x` exactly; its exponent
    // field is `1023 + floor(log2(x))`.
    let magic = _mm256_set1_epi64x(0x4330_0000_0000_0000);
    let value = _mm256_sub_pd(
        _mm256_castsi256_pd(_mm256_or_si256(x, magic)),
        _mm256_castsi256_pd(magic),
    );
    let exponent = _mm256_srli_epi64::<52>(_mm256_castpd_si256(value));
    _mm256_sub_epi64(exponent, _mm256_set1_epi64x(1022))
}

/// Advance all lanes until every lane has parsed its rows, with AVX2.
///
/// The lanes of inactive lanes keep valid positions, so all gathers read in
/// bounds and need no masks; only the state updates use the lane masks.
#[target_feature(enable = "avx2,popcnt")]
unsafe fn run_avx2<I: OffsetLike>(
    flat: FlatMatcher,
    bytes: &[u8],
    offsets: &[I],
    buffers: &mut VectorBuffers,
    lanes: &mut Lanes,
) {
    let one = _mm256_set1_epi64x(1);
    let eight = _mm256_set1_epi64x(8);
    let sign = _mm256_set1_epi64x(i64::MIN);
    let low16 = _mm256_set1_epi64x(0xFFFF);
    let low32 = _mm256_set1_epi64x(0xFFFF_FFFF);
    let bswap = _mm256_setr_epi8(
        7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13,
        12, 11, 10, 9, 8,
    );
    let input = bytes.as_ptr().cast::<i64>();
    let out = buffers.codes.as_mut_ptr();

    // SAFETY (all loads and stores of lane state): each array holds 4 lanes
    // per group.
    let load = |a: &[u64; LANES], g: usize| unsafe {
        _mm256_loadu_si256(a.as_ptr().add(g * WIDTH_AVX2).cast())
    };
    let zero = _mm256_setzero_si256();
    let mut at: [__m256i; GROUPS_AVX2] = std::array::from_fn(|g| load(&lanes.at, g));
    let mut row_end: [__m256i; GROUPS_AVX2] = std::array::from_fn(|g| load(&lanes.row_end, g));
    let mut count: [__m256i; GROUPS_AVX2] = std::array::from_fn(|g| load(&lanes.count, g));
    let slot: [__m256i; GROUPS_AVX2] = std::array::from_fn(|g| load(&lanes.region, g));

    let mut w0 = [zero; GROUPS_AVX2];
    let mut w1 = [zero; GROUPS_AVX2];
    let mut qh = [zero; GROUPS_AVX2];
    let mut ql = [zero; GROUPS_AVX2];
    let mut base = [zero; GROUPS_AVX2];
    let mut size = [zero; GROUPS_AVX2];
    let mut codes = [0u64; LANES];

    while lanes.active != 0 {
        let active: [__m256i; GROUPS_AVX2] = std::array::from_fn(|g| active_avx2(lanes.active, g));

        // Window keys and their 2-byte prefix ranges. An inactive lane
        // searches with a range of one token.
        for g in 0..GROUPS_AVX2 {
            // SAFETY: a lane position is at most its last row end, and the
            // caller guarantees 16 readable bytes there.
            unsafe {
                w0[g] = _mm256_i64gather_epi64::<1>(input, at[g]);
                w1[g] = _mm256_i64gather_epi64::<1>(input, _mm256_add_epi64(at[g], eight));
            }
            qh[g] = _mm256_shuffle_epi8(w0[g], bswap);
            ql[g] = _mm256_shuffle_epi8(w1[g], bswap);

            // SAFETY: a prefix indexes the bounds, and three pads follow.
            let range = unsafe {
                _mm256_i64gather_epi64::<2>(flat.token_bounds, _mm256_srli_epi64::<48>(qh[g]))
            };
            base[g] = _mm256_and_si256(range, low16);
            let last = _mm256_and_si256(_mm256_srli_epi64::<16>(range), low16);
            let span = _mm256_add_epi64(_mm256_sub_epi64(last, base[g]), one);
            size[g] = _mm256_blendv_epi8(one, span, active[g]);
        }

        // Binary search for the last key `<=` the window key. A lane with a
        // range of one compares the first key, which is `<=` the window key,
        // so it keeps its result.
        let qhs: [__m256i; GROUPS_AVX2] = std::array::from_fn(|g| _mm256_xor_si256(qh[g], sign));
        let qls: [__m256i; GROUPS_AVX2] = std::array::from_fn(|g| _mm256_xor_si256(ql[g], sign));
        loop {
            let mut busy = 0;
            for g in 0..GROUPS_AVX2 {
                busy |= _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpgt_epi64(size[g], one)));
                let half = _mm256_srli_epi64::<1>(size[g]);
                let mid = _mm256_add_epi64(base[g], half);
                // SAFETY: `mid` is a token of the range.
                let (kh, kl) = unsafe {
                    (
                        _mm256_i64gather_epi64::<8>(flat.keys_hi, mid),
                        _mm256_i64gather_epi64::<8>(flat.keys_lo, mid),
                    )
                };
                let lt = _mm256_cmpgt_epi64(qhs[g], _mm256_xor_si256(kh, sign));
                let eq = _mm256_cmpeq_epi64(kh, qh[g]);
                let gt = _mm256_cmpgt_epi64(_mm256_xor_si256(kl, sign), qls[g]);
                let le = _mm256_or_si256(lt, _mm256_andnot_si256(gt, eq));
                base[g] = _mm256_blendv_epi8(base[g], mid, le);
                size[g] = _mm256_sub_epi64(size[g], half);
            }
            if busy == 0 {
                break;
            }
        }

        let mut done = 0u32;
        for g in 0..GROUPS_AVX2 {
            let q = base[g];
            // SAFETY: `q` is a token.
            let (fh, fl, chain) = unsafe {
                (
                    _mm256_i64gather_epi64::<8>(flat.keys_hi, q),
                    _mm256_i64gather_epi64::<8>(flat.keys_lo, q),
                    _mm256_i64gather_epi64::<8>(flat.chains, q),
                )
            };

            // Common prefix bytes of `q` and the window, up to 16.
            let high = common_bytes_avx2(_mm256_shuffle_epi8(fh, bswap), w0[g]);
            let low = common_bytes_avx2(_mm256_shuffle_epi8(fl, bswap), w1[g]);
            let common =
                _mm256_add_epi64(high, _mm256_and_si256(low, _mm256_cmpeq_epi64(high, eight)));
            let win = _mm256_sub_epi64(row_end[g], at[g]);
            let bound = _mm256_blendv_epi8(common, win, _mm256_cmpgt_epi64(common, win));

            // Bit 0 of `lens` is always set. An inactive lane can have a zero
            // bound, so `fits` keeps bit 0 to stay in bounds below.
            let lens = _mm256_and_si256(_mm256_srli_epi64::<32>(chain), low16);
            let mask = _mm256_sub_epi64(_mm256_sllv_epi64(one, bound), one);
            let fits = _mm256_or_si256(_mm256_and_si256(lens, mask), one);
            let len = bit_len_avx2(fits);

            // The token: `q` itself when every prefix length fits, else the
            // prefix path entry at the rank of the longest fitting length.
            let rank = _mm256_sub_epi64(popcnt_avx2(fits), one);
            let path = _mm256_add_epi64(_mm256_and_si256(chain, low32), rank);
            // SAFETY: the path entry exists, and a spare entry follows the last.
            let proper: __m128i = unsafe { _mm256_i64gather_epi32::<2>(flat.prefix_ids, path) };
            let proper = _mm256_and_si256(_mm256_cvtepu32_epi64(proper), low16);
            let id = _mm256_srli_epi64::<48>(chain);
            let code = _mm256_blendv_epi8(proper, id, _mm256_cmpeq_epi64(fits, lens));

            // Each lane writes its code at its next slot. An inactive lane
            // writes past its codes, inside its region, and the write is lost.
            // The code is a `Token`, so a direct store writes only its slot.
            let target = _mm256_add_epi64(slot[g], count[g]);
            let mut targets = [0u64; WIDTH_AVX2];
            // SAFETY: both arrays hold 4 lanes from this offset.
            unsafe {
                _mm256_storeu_si256(codes.as_mut_ptr().add(g * WIDTH_AVX2).cast(), code);
                _mm256_storeu_si256(targets.as_mut_ptr().cast(), target);
            }
            for (k, &target) in targets.iter().enumerate() {
                // SAFETY: the slot is inside the lane's region of the buffer.
                unsafe { *out.add(target as usize) = codes[g * WIDTH_AVX2 + k] as Token };
            }

            count[g] = _mm256_sub_epi64(count[g], active[g]);
            at[g] = _mm256_add_epi64(at[g], _mm256_and_si256(len, active[g]));
            let ended = _mm256_andnot_si256(_mm256_cmpgt_epi64(row_end[g], at[g]), active[g]);
            done |= (_mm256_movemask_pd(_mm256_castsi256_pd(ended)) as u32) << (g * WIDTH_AVX2);
        }

        if done != 0 {
            for g in 0..GROUPS_AVX2 {
                // SAFETY: each array holds 4 lanes per group.
                unsafe {
                    _mm256_storeu_si256(lanes.at.as_mut_ptr().add(g * WIDTH_AVX2).cast(), at[g]);
                    _mm256_storeu_si256(
                        lanes.count.as_mut_ptr().add(g * WIDTH_AVX2).cast(),
                        count[g],
                    );
                }
            }
            finish_rows(lanes, offsets, &mut buffers.row_ends, done);
            for g in 0..GROUPS_AVX2 {
                row_end[g] = load(&lanes.row_end, g);
            }
        }
    }

    for g in 0..GROUPS_AVX2 {
        // SAFETY: each array holds 4 lanes per group.
        unsafe {
            _mm256_storeu_si256(
                lanes.count.as_mut_ptr().add(g * WIDTH_AVX2).cast(),
                count[g],
            );
        }
    }
}
