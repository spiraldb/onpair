// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `code - begin <= last - begin` unsigned, so the wrap rejects codes below
//! `begin`; ORed over R ranges, folded into every kernel and run alone here.

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
use crate::search::prefilter::ProbeCover;

/// `begin` and `last - begin`, each broadcast.
#[cfg(target_arch = "aarch64")]
pub(in crate::search::prefilter::scan) type Held = (uint16x8_t, uint16x8_t);
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512bw")))]
pub(in crate::search::prefilter::scan) type Held = (__m256i, __m256i);
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
pub(in crate::search::prefilter::scan) type Held = (__m512i, __m512i);

#[cfg(target_arch = "aarch64")]
pub(in crate::search::prefilter::scan) fn hold(range: TokenRange) -> Held {
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
pub(in crate::search::prefilter::scan) unsafe fn inside((lo, width): Held, codes: Vectors) -> Hits {
    narrow(codes.map(|codes| vcleq_u16(vsubq_u16(codes, lo), width)))
}

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512bw")))]
pub(in crate::search::prefilter::scan) fn hold(range: TokenRange) -> Held {
    // SAFETY: avx2, checked by `available` before planning.
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
pub(in crate::search::prefilter::scan) unsafe fn inside((lo, width): Held, codes: Vectors) -> Hits {
    narrow(codes.map(|codes| {
        let inside = _mm256_sub_epi16(codes, lo);
        _mm256_cmpeq_epi16(_mm256_min_epu16(inside, width), inside)
    }))
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
pub(in crate::search::prefilter::scan) fn hold(range: TokenRange) -> Held {
    // SAFETY: avx512bw, enabled by the build and checked by `available`.
    unsafe {
        (
            _mm512_set1_epi16(range.begin as i16),
            _mm512_set1_epi16((range.last - range.begin) as i16),
        )
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
#[target_feature(enable = "avx512f,avx512bw")]
pub(in crate::search::prefilter::scan) unsafe fn inside((lo, width): Held, codes: Vectors) -> Hits {
    let inside = |codes| _mm512_cmple_epu16_mask(_mm512_sub_epi16(codes, lo), width);
    join(inside(codes[0]), inside(codes[1]))
}

#[inline]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
pub(super) unsafe fn check_ranges(mut hit: Hits, held: &[Held], codes: Vectors) -> Hits {
    for &held in held {
        // SAFETY: the set, inherited from this function.
        hit = unsafe { or(hit, inside(held, codes)) };
    }
    hit
}

/// For a cover with no tokens.
pub(in crate::search::prefilter::scan) struct Range<const SKIP_MOVEMASK_IF_NO_MATCH: bool>(
    Vec<Held>,
);

impl<const SKIP_MOVEMASK_IF_NO_MATCH: bool> Matcher for Range<SKIP_MOVEMASK_IF_NO_MATCH> {
    fn new(cover: &ProbeCover) -> Self {
        Self(cover.ranges().iter().copied().map(hold).collect())
    }

    fn check(&self, codes: &Block, bits: &mut Mask) -> bool {
        // SAFETY: `policy::takes` answered for the set.
        unsafe { mask::<SKIP_MOVEMASK_IF_NO_MATCH>(&self.0, codes, bits) }
    }
}

/// Outside `check` so the closure inherits the target features and inlines.
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
fn mask<const SKIP_MOVEMASK_IF_NO_MATCH: bool>(
    held: &[Held],
    codes: &Block,
    bits: &mut Mask,
) -> bool {
    let (&first, rest) = held.split_first().expect("a range, per policy::takes");
    words::<SKIP_MOVEMASK_IF_NO_MATCH>(codes, bits, |codes| unsafe {
        check_ranges(inside(first, codes), rest, codes)
    })
}
