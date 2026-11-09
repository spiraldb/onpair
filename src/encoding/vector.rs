// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Greedy parsing with the lanes in AVX-512 vectors.
//!
//! The scalar lanes of [`lanes`](crate::encoding::lanes) spend most of their
//! time on instructions: each lane runs its own search. Here eight lanes share
//! one vector, and every search step is a gather and a compare for all eight.
//! Several vectors advance together, so their gathers overlap.
//!
//! A search finds `q`, the last token whose key is `<=` the window key, by a
//! binary search over the token keys of the window's 2-byte prefix range. The
//! rest follows [`DictionaryMatcher`]: the common prefix of `q` and the window
//! bounds the length, and the prefix chain of `q` gives the token.
//!
//! Rows rarely end, so row completion is a scalar step that runs only when a
//! lane reaches the end of its row.

use std::arch::x86_64::{
    __cpuid, __m256i, __m512i, __mmask8, _mm256_setzero_si256, _mm512_add_epi64, _mm512_and_si512,
    _mm512_cmpeq_epu64_mask, _mm512_cmpge_epu64_mask, _mm512_cmpgt_epu64_mask,
    _mm512_cmple_epu64_mask, _mm512_cmplt_epu64_mask, _mm512_cmpneq_epu64_mask,
    _mm512_cvtepi64_epi32, _mm512_cvtepu32_epi64, _mm512_loadu_si512, _mm512_lzcnt_epi64,
    _mm512_mask_add_epi64, _mm512_mask_blend_epi64, _mm512_mask_i64gather_epi32,
    _mm512_mask_i64gather_epi64, _mm512_mask_i64scatter_epi32, _mm512_mask_sub_epi64,
    _mm512_min_epu64, _mm512_popcnt_epi64, _mm512_set1_epi64, _mm512_setzero_si512,
    _mm512_shuffle_epi8, _mm512_sllv_epi64, _mm512_srli_epi64, _mm512_storeu_si512,
    _mm512_sub_epi64, _mm512_xor_si512,
};

use std::sync::OnceLock;

use fearless_simd::Level;
use fearless_simd::x86::Avx512;

use crate::core::offset::Offset;
use crate::core::types::{MAX_TOKEN_SIZE, Token};
use crate::encoding::dictionary_matcher::{DictionaryMatcher, FlatMatcher};
use crate::encoding::lanes::OffsetLike;

/// Lanes per vector.
const WIDTH: usize = 8;

/// Vectors that advance together.
const GROUPS: usize = 4;

/// All lanes of a batch.
const LANES: usize = WIDTH * GROUPS;

/// A vector parser kernel, with the token that proves its target features.
#[derive(Clone, Copy)]
pub(crate) enum Kernel {
    /// Eight lanes per AVX-512 vector.
    Avx512(Avx512),
}

impl Kernel {
    /// The kernel for this CPU, if the vector parser runs on it.
    ///
    /// The parser needs fast gathers. AMD Zen 4 gathers are slow, and there
    /// the scalar lanes are faster by about a third; on Zen 5 the two are
    /// equal. Intel CPUs from Ice Lake on are faster with the vector parser.
    /// Tests run it on every CPU with the features, so that its results stay
    /// checked.
    pub(crate) fn detect() -> Option<Self> {
        static INTEL: OnceLock<bool> = OnceLock::new();
        if !(cfg!(test) || *INTEL.get_or_init(is_intel)) {
            return None;
        }

        Level::new().as_avx512().map(Self::Avx512)
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
    active: [__mmask8; GROUPS],
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
    match kernel {
        // SAFETY: the token proves AVX-512 (Ice Lake level, which includes
        // BW, CD, VPOPCNTDQ and POPCNT), and the caller guarantees the rest.
        Kernel::Avx512(_) => unsafe {
            parse_batch_avx512(matcher, bytes, offsets, buffers, codes, row_offsets)
        },
    }
}

/// [`parse_batch`] with AVX-512.
///
/// # Safety
/// The CPU supports the target features, and the [`parse_batch`] contract
/// holds.
#[target_feature(enable = "avx512f,avx512bw,avx512cd,avx512vpopcntdq,popcnt")]
unsafe fn parse_batch_avx512<I: OffsetLike, O: Offset>(
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
        active: [0; GROUPS],
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
            lanes.active[j / WIDTH] |= 1 << (j % WIDTH);
        } else {
            lanes.row_end[j] = start as u64;
        }
    }

    // Rows that start empty end before the first step.
    finish_rows(&mut lanes, offsets, &mut buffers.row_ends, u32::MAX);

    // SAFETY: the caller guarantees the features and the readable bytes.
    unsafe { run(matcher.flat(), bytes, offsets, buffers, &mut lanes) };

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
        let (g, bit) = (j / WIDTH, 1 << (j % WIDTH));
        if done & (1 << j) == 0 || lanes.active[g] & bit == 0 {
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
            lanes.active[g] &= !bit;
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

    while lanes.active.iter().any(|&m| m != 0) {
        let active = lanes.active;

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
