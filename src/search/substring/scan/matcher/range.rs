// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vector membership checks for inclusive ranges of token IDs.
//!
//! For a valid range `[begin, last]`, unsigned wrapping subtraction turns
//! membership into `code.wrapping_sub(begin) <= last - begin`. Values below
//! `begin` wrap above the permitted width and are rejected.
//!
//! `Range` handles covers containing only ranges. Equality and nibble matchers
//! reuse `check_ranges` to add range hits to their point results. All paths
//! produce the same exact membership mask.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
use super::shared::join;
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", not(target_feature = "avx512bw"))
))]
use super::shared::narrow;
use super::shared::{Hits, Vectors, or, words};
use super::{Block, Mask, Matcher};
use crate::core::types::TokenRange;
use crate::search::substring::ProbeCover;

/// Range start and width (`last - begin`), broadcast into vector lanes.
#[cfg(target_arch = "aarch64")]
pub(in crate::search::substring::scan) type Held = (uint16x8_t, uint16x8_t);
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512bw")))]
pub(in crate::search::substring::scan) type Held = (__m256i, __m256i);
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
pub(in crate::search::substring::scan) type Held = (__m512i, __m512i);

#[cfg(target_arch = "aarch64")]
pub(in crate::search::substring::scan) fn hold(range: TokenRange) -> Held {
    // SAFETY: neon is baseline on aarch64.
    unsafe {
        (
            vdupq_n_u16(range.begin),
            vdupq_n_u16(range.last - range.begin),
        )
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(in crate::search::substring::scan) unsafe fn inside((lo, width): Held, codes: Vectors) -> Hits {
    let mut hit = codes;
    for hit in &mut hit {
        *hit = vcleq_u16(vsubq_u16(*hit, lo), width);
    }
    narrow(hit)
}

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512bw")))]
pub(in crate::search::substring::scan) fn hold(range: TokenRange) -> Held {
    // SAFETY: avx2, checked by `detect_target_caps` before planning.
    unsafe {
        (
            _mm256_set1_epi16(range.begin as i16),
            _mm256_set1_epi16((range.last - range.begin) as i16),
        )
    }
}

/// No unsigned u16 compare below AVX-512: `min(x, width) == x` is `x <= width`.
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512bw")))]
#[target_feature(enable = "avx2")]
pub(in crate::search::substring::scan) unsafe fn inside((lo, width): Held, codes: Vectors) -> Hits {
    let mut hit = codes;
    for hit in &mut hit {
        let inside = _mm256_sub_epi16(*hit, lo);
        *hit = _mm256_cmpeq_epi16(_mm256_min_epu16(inside, width), inside);
    }
    narrow(hit)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
pub(in crate::search::substring::scan) fn hold(range: TokenRange) -> Held {
    // SAFETY: avx512bw, enabled by the build and checked by `detect_target_caps`.
    unsafe {
        (
            _mm512_set1_epi16(range.begin as i16),
            _mm512_set1_epi16((range.last - range.begin) as i16),
        )
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
#[target_feature(enable = "avx512f,avx512bw")]
pub(in crate::search::substring::scan) unsafe fn inside((lo, width): Held, codes: Vectors) -> Hits {
    let inside = |codes| _mm512_cmple_epu16_mask(_mm512_sub_epi16(codes, lo), width);
    join(inside(codes[0]), inside(codes[1]))
}

/// OR range membership into existing hit lanes.
/// The caller must have enabled the compiled vector instruction set.
#[inline]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
pub(super) unsafe fn check_ranges(mut hit: Hits, held: &[Held], codes: Vectors) -> Hits {
    for &held in held {
        // SAFETY: the caller provides the instruction set required by these helpers.
        hit = unsafe { or(hit, inside(held, codes)) };
    }
    hit
}

/// Prepared matcher for a cover containing ranges and no individual points.
pub(in crate::search::substring::scan) struct Range<const SKIP_MOVEMASK_IF_NO_MATCH: bool>(
    Vec<Held>,
);

impl<const SKIP_MOVEMASK_IF_NO_MATCH: bool> Matcher for Range<SKIP_MOVEMASK_IF_NO_MATCH> {
    fn new(cover: &ProbeCover) -> Self {
        Self(cover.ranges().iter().copied().map(hold).collect())
    }

    fn check(&self, codes: &Block, bits: &mut Mask) -> bool {
        // SAFETY: planning selects this kernel only after CPU feature detection.
        unsafe { mask::<SKIP_MOVEMASK_IF_NO_MATCH>(&self.0, codes, bits) }
    }
}

/// Fill a block mask from prepared ranges; an empty slice clears the mask.
/// The target-feature boundary keeps range comparisons inside the vector loop.
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
pub(super) fn mask<const SKIP_MOVEMASK_IF_NO_MATCH: bool>(
    held: &[Held],
    codes: &Block,
    bits: &mut Mask,
) -> bool {
    let Some((&first, rest)) = held.split_first() else {
        bits.fill(0);
        return false;
    };
    words::<SKIP_MOVEMASK_IF_NO_MATCH>(codes, bits, |codes| unsafe {
        check_ranges(inside(first, codes), rest, codes)
    })
}
