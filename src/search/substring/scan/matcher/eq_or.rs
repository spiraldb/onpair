// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Match point probes with vector equality comparisons.
//!
//! Each point is broadcast across lanes, compared with the input codes, and
//! ORed into the hit lanes. Inclusive-range results are then ORed into the
//! same mask. Work grows with the number of points and ranges in the cover.
//!
//! The planner selects this kernel for covers with points. An empty point
//! slice falls back to range matching. `shared::words` loads blocks and packs
//! the hit lanes, optionally skipping the pack for groups with no hits.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::range::{self, Held, check_ranges, hold};
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
use crate::search::substring::ProbeCover;

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
#[inline]
unsafe fn hits(first: &Broadcast, rest: &[Broadcast], codes: Vectors) -> Hits {
    let mut hit = codes;
    for hit in &mut hit {
        *hit = vceqq_u16(*hit, *first);
    }
    for token in rest {
        for (hit, &codes) in hit.iter_mut().zip(&codes) {
            *hit = vorrq_u16(*hit, vceqq_u16(codes, *token));
        }
    }
    narrow(hit)
}

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512bw")))]
fn broadcast(code: Token) -> Broadcast {
    // SAFETY: avx2, checked by `detect_isa` before planning.
    unsafe { _mm256_set1_epi16(code as i16) }
}

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512bw")))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hits(first: &Broadcast, rest: &[Broadcast], codes: Vectors) -> Hits {
    let mut hit = codes;
    for hit in &mut hit {
        *hit = _mm256_cmpeq_epi16(*hit, *first);
    }
    for token in rest {
        for (hit, &codes) in hit.iter_mut().zip(&codes) {
            *hit = _mm256_or_si256(*hit, _mm256_cmpeq_epi16(codes, *token));
        }
    }
    narrow(hit)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
fn broadcast(code: Token) -> Broadcast {
    // SAFETY: avx512bw, enabled by the build and checked by `detect_isa`.
    unsafe { _mm512_set1_epi16(code as i16) }
}

/// Compares land in mask registers, so nothing to narrow.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
unsafe fn hits(first: &Broadcast, rest: &[Broadcast], codes: Vectors) -> Hits {
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

/// Broadcast point probes and prepared ranges reused across scan blocks.
pub(in crate::search::substring::scan) struct EqOr<const SKIP_MOVEMASK_IF_NO_MATCH: bool> {
    /// One broadcast vector per point in the cover.
    tokens: Vec<Broadcast>,
    /// Range starts and widths, already broadcast for vector comparisons.
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
        // SAFETY: planning selects this kernel only after CPU feature detection.
        unsafe { mask::<SKIP_MOVEMASK_IF_NO_MATCH>(&self.tokens, &self.ranges, codes, bits) }
    }
}

/// Fill the block mask with point and range matches.
/// The target-feature boundary lets the comparison closure inline with its caller.
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
fn mask<const SKIP_MOVEMASK_IF_NO_MATCH: bool>(
    tokens: &[Broadcast],
    ranges: &[Held],
    codes: &Block,
    bits: &mut Mask,
) -> bool {
    let Some((first, rest)) = tokens.split_first() else {
        return range::mask::<SKIP_MOVEMASK_IF_NO_MATCH>(ranges, codes, bits);
    };
    words::<SKIP_MOVEMASK_IF_NO_MATCH>(
        codes,
        bits,
        #[inline(always)]
        |codes| unsafe { check_ranges(hits(first, rest, codes), ranges, codes) },
    )
}
