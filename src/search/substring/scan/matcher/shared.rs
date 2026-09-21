// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vector loading and mask packing shared by the membership kernels.
//!
//! `Vectors` holds 64 token codes. A kernel turns those codes into `Hits`:
//! 0xFF/0x00 byte lanes on NEON and AVX2, or a `u64` mask on AVX-512.
//! Narrowing comparison lanes preserves one result per original code.
//!
//! `words` processes two groups of 64 codes at a time and writes their mask
//! words in code order. With skipping enabled, an empty group pair is cleared
//! without packing. The caller has already selected an available instruction set.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::{Block, Mask};
use crate::core::types::Token;

/// Membership results for 64 codes, in the representation used by this target.
#[cfg(target_arch = "aarch64")]
pub(super) type Hits = [uint8x16_t; 4];
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512bw")))]
pub(super) type Hits = [__m256i; 2];
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
pub(super) type Hits = u64;

/// 64 token codes split across this target's vector registers.
#[cfg(target_arch = "aarch64")]
pub(in crate::search::substring::scan) type Vectors = [uint16x8_t; 8];
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512bw")))]
pub(in crate::search::substring::scan) type Vectors = [__m256i; 4];
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
pub(in crate::search::substring::scan) type Vectors = [__m512i; 2];

/// Load 64 codes without requiring aligned pointers.
/// The caller must provide 64 readable tokens and the required CPU features.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn load(at: *const Token) -> Vectors {
    std::array::from_fn(|i| unsafe { vld1q_u16(at.add(8 * i)) })
}

/// Load 64 codes without requiring aligned pointers.
/// The caller must provide 64 readable tokens and the required CPU features.
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512bw")))]
#[target_feature(enable = "avx2")]
unsafe fn load(at: *const Token) -> Vectors {
    std::array::from_fn(|i| unsafe { _mm256_loadu_si256(at.add(16 * i).cast()) })
}

/// Load 64 codes without requiring aligned pointers.
/// The caller must provide 64 readable tokens and the required CPU features.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn load(at: *const Token) -> Vectors {
    std::array::from_fn(|i| unsafe { _mm512_loadu_si512(at.add(32 * i).cast()) })
}

/// Fill all mask words by comparing 128 codes per iteration.
/// When skipping is enabled, empty pairs bypass packing and are zeroed.
/// `false` guarantees an empty mask; without skipping, the return is always
/// `true` even if every packed bit is zero.
#[inline]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
pub(super) fn words<const SKIP_MOVEMASK_IF_NO_MATCH: bool>(
    codes: &Block,
    bits: &mut Mask,
    hits: impl Fn(Vectors) -> Hits,
) -> bool {
    let mut written = false;
    for (pair, bits) in bits.chunks_exact_mut(2).enumerate() {
        let at = codes[pair * 128..].as_ptr();
        // SAFETY: a block holds 128 codes for every pair of mask words.
        let (a, b) = unsafe { (hits(load(at)), hits(load(at.add(64)))) };
        if SKIP_MOVEMASK_IF_NO_MATCH && !any(a, b) {
            bits.fill(0);
            continue;
        }
        written = true;
        // SAFETY: the chunk is the two words the store writes.
        unsafe { movemask(bits, a, b) };
    }
    written
}

/// Union two sets of hit results without changing code order.
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(super) fn or(hit: Hits, with: Hits) -> Hits {
    std::array::from_fn(|at| vorrq_u8(hit[at], with[at]))
}

/// Whether either 64-code group has a hit.
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
fn any(a: Hits, b: Hits) -> bool {
    let a = vorrq_u8(vorrq_u8(a[0], a[1]), vorrq_u8(a[2], a[3]));
    let b = vorrq_u8(vorrq_u8(b[0], b[1]), vorrq_u8(b[2], b[3]));
    vmaxvq_u8(vorrq_u8(a, b)) != 0
}

/// Keep the low byte of each 0xFFFF/0x0000 comparison lane in code order.
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(super) fn narrow(hit: [uint16x8_t; 8]) -> Hits {
    let byte = |lo: uint16x8_t, hi: uint16x8_t| {
        vuzp1q_u8(vreinterpretq_u8_u16(lo), vreinterpretq_u8_u16(hi))
    };
    [
        byte(hit[0], hit[1]),
        byte(hit[2], hit[3]),
        byte(hit[4], hit[5]),
        byte(hit[6], hit[7]),
    ]
}

/// Pack 128 hit bytes into two words using lane bits and pairwise adds.
/// The caller supplies space for both words and enables NEON.
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn movemask(bits: &mut [u64], a: Hits, b: Hits) {
    const LANE_BIT: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    // SAFETY: LANE_BIT is 16 bytes.
    let lane_bit = unsafe { vld1q_u8(LANE_BIT.as_ptr()) };
    let half = |hit: Hits| {
        let lo = vpaddq_u8(vandq_u8(hit[0], lane_bit), vandq_u8(hit[1], lane_bit));
        let hi = vpaddq_u8(vandq_u8(hit[2], lane_bit), vandq_u8(hit[3], lane_bit));
        vpaddq_u8(lo, hi)
    };
    let lanes = vpaddq_u8(half(a), half(b));
    // SAFETY: the caller's chunk is the two words this writes.
    unsafe { vst1q_u64(bits.as_mut_ptr(), vreinterpretq_u64_u8(lanes)) };
}

/// Union two sets of hit results without changing code order.
#[inline]
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512bw")))]
#[target_feature(enable = "avx2")]
pub(super) fn or(hit: Hits, with: Hits) -> Hits {
    std::array::from_fn(|at| _mm256_or_si256(hit[at], with[at]))
}

/// Whether either 64-code group has a hit.
#[inline]
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512bw")))]
#[target_feature(enable = "avx2")]
fn any(a: Hits, b: Hits) -> bool {
    let or = _mm256_or_si256(or(a, b)[0], or(a, b)[1]);
    _mm256_testz_si256(or, or) == 0
}

/// Pack 16-bit comparison lanes into bytes, then restore original code order.
/// The permutation corrects the 128-bit lane ordering of `packs_epi16`.
#[inline]
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512bw")))]
#[target_feature(enable = "avx2")]
pub(super) fn narrow(hit: [__m256i; 4]) -> Hits {
    let byte = |lo, hi| _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi16(lo, hi));
    [byte(hit[0], hit[1]), byte(hit[2], hit[3])]
}

/// Extract one bit per hit byte into two output words.
/// The caller supplies space for both words and enables AVX2.
#[inline]
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512bw")))]
#[target_feature(enable = "avx2")]
unsafe fn movemask(bits: &mut [u64], a: Hits, b: Hits) {
    let word = |hit: Hits| {
        let lo = _mm256_movemask_epi8(hit[0]) as u32;
        let hi = _mm256_movemask_epi8(hit[1]) as u32;
        u64::from(lo) | u64::from(hi) << 32
    };
    bits[0] = word(a);
    bits[1] = word(b);
}

/// Union two sets of hit results without changing code order.
#[inline]
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
pub(super) fn or(hit: Hits, with: Hits) -> Hits {
    hit | with
}

/// Whether either 64-code group has a hit.
#[inline]
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
fn any(a: Hits, b: Hits) -> bool {
    a | b != 0
}

/// Combine two 32-code masks, placing earlier codes in the low bits.
#[inline]
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
pub(super) fn join(lo: u32, hi: u32) -> Hits {
    u64::from(lo) | u64::from(hi) << 32
}

/// Store two already-packed AVX-512 masks in the caller's two-word slice.
#[inline]
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
unsafe fn movemask(bits: &mut [u64], a: Hits, b: Hits) {
    bits[0] = a;
    bits[1] = b;
}
