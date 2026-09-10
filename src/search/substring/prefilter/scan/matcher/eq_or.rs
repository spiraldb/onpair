// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! One compare per token per vector, ORed, plus the ranges. L = 1, any K, any R.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::range::{Held, check_ranges, hold};
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
use super::shared::join;
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", not(target_feature = "avx512bw"))
))]
use super::shared::narrow;
use super::shared::{Hits, Vectors, words};
use super::{Block, Mask, Matcher};
use crate::core::types::Token;
use crate::search::substring::prefilter::ProbeCover;

/// A token broadcast to every lane.
#[cfg(target_arch = "aarch64")]
type Broadcast = uint16x8_t;
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512bw")))]
type Broadcast = __m256i;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
type Broadcast = __m512i;

#[cfg(target_arch = "aarch64")]
fn broadcast(code: Token) -> Broadcast {
    // SAFETY: neon is baseline on aarch64.
    unsafe { vdupq_n_u16(code) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn hits(tokens: &[Broadcast], codes: Vectors) -> Hits {
    let (first, rest) = tokens.split_first().unwrap();
    let mut hit = codes.map(|codes| vceqq_u16(codes, *first));
    for token in rest {
        for (hit, &codes) in hit.iter_mut().zip(&codes) {
            *hit = vorrq_u16(*hit, vceqq_u16(codes, *token));
        }
    }
    narrow(hit)
}

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512bw")))]
fn broadcast(code: Token) -> Broadcast {
    // SAFETY: avx2, checked by `available` before planning.
    unsafe { _mm256_set1_epi16(code as i16) }
}

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512bw")))]
#[target_feature(enable = "avx2")]
unsafe fn hits(tokens: &[Broadcast], codes: Vectors) -> Hits {
    let (first, rest) = tokens.split_first().unwrap();
    let mut hit = codes.map(|codes| _mm256_cmpeq_epi16(codes, *first));
    for token in rest {
        for (hit, &codes) in hit.iter_mut().zip(&codes) {
            *hit = _mm256_or_si256(*hit, _mm256_cmpeq_epi16(codes, *token));
        }
    }
    narrow(hit)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
fn broadcast(code: Token) -> Broadcast {
    // SAFETY: avx512bw, enabled by the build and checked by `available`.
    unsafe { _mm512_set1_epi16(code as i16) }
}

/// Compares land in mask registers, so nothing to narrow.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn hits(tokens: &[Broadcast], codes: Vectors) -> Hits {
    let (first, rest) = tokens.split_first().unwrap();
    let mut hit = join(
        _mm512_cmpeq_epi16_mask(codes[0], *first),
        _mm512_cmpeq_epi16_mask(codes[1], *first),
    );
    for token in rest {
        hit |= join(
            _mm512_cmpeq_epi16_mask(codes[0], *token),
            _mm512_cmpeq_epi16_mask(codes[1], *token),
        );
    }
    hit
}

pub(in crate::search::substring::prefilter::scan) struct EqOr<const SKIP_MOVEMASK_IF_NO_MATCH: bool>
{
    tokens: Vec<Broadcast>,
    ranges: Vec<Held>,
}

impl<const SKIP_MOVEMASK_IF_NO_MATCH: bool> Matcher for EqOr<SKIP_MOVEMASK_IF_NO_MATCH> {
    fn new(cover: &ProbeCover) -> Self {
        Self {
            tokens: cover.points().iter().copied().map(broadcast).collect(),
            ranges: cover.ranges().iter().copied().map(hold).collect(),
        }
    }

    fn check(&self, codes: &Block, bits: &mut Mask) -> bool {
        // SAFETY: `policy::takes` answered for the set.
        unsafe { mask::<SKIP_MOVEMASK_IF_NO_MATCH>(&self.tokens, &self.ranges, codes, bits) }
    }
}

/// Outside `check` so the closure inherits the target features and inlines.
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
fn mask<const SKIP_MOVEMASK_IF_NO_MATCH: bool>(
    tokens: &[Broadcast],
    ranges: &[Held],
    codes: &Block,
    bits: &mut Mask,
) -> bool {
    words::<SKIP_MOVEMASK_IF_NO_MATCH>(codes, bits, |codes| unsafe {
        check_ranges(hits(tokens, codes), ranges, codes)
    })
}
