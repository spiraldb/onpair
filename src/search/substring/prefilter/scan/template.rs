// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared stream walk for target-specific prefilter scan vocabularies.

use super::sink::{RowSink, scan_tail};
use crate::core::offset::Offset;
use crate::core::types::{Token, TokenRange};
use crate::search::substring::prefilter::cover::ProbeCover;

/// Probe count known only at run time.
pub(super) const DYN: usize = usize::MAX;

/// The short instruction sequences needed by the shared scan walk.
pub(super) trait Isa {
    /// Codes examined by one call to `block`.
    const BLOCK: usize;

    /// Whether the original walk consumed complete blocks left after its
    /// grouped loop. NEON's group-one leaves went directly to the scalar tail;
    /// x86 keeps its existing remainder-block loops.
    const WALK_REMAINDER: bool = true;

    /// Hoisted point and range broadcasts.
    type Point: Copy;
    type Range: Copy;

    /// Retained live lanes for one block.
    type Hits: Copy;
    const NO_HITS: Self::Hits;

    fn point(token: Token) -> Self::Point;
    fn range(range: TokenRange) -> Self::Range;

    /// Evaluate one complete block.
    ///
    /// # Safety
    ///
    /// `codes..codes + BLOCK` must be readable and the current target-feature
    /// frame must enable this implementation's ISA.
    unsafe fn block<const POINTS: usize, const RANGES: usize>(
        codes: *const Token,
        points: &[Self::Point],
        ranges: &[Self::Range],
    ) -> Self::Hits;

    fn any(hits: Self::Hits) -> bool;

    fn emit<O: Offset>(base: usize, hits: Self::Hits, sink: &mut RowSink<'_, O>);
}

/// Walk every complete block in order, then scan the scalar residue.
#[inline(always)]
pub(super) unsafe fn walk<
    I: Isa,
    O: Offset,
    const POINTS: usize,
    const RANGES: usize,
    const GROUP: usize,
>(
    codes: &[Token],
    row_offsets: &[O],
    cover: &ProbeCover,
    points: &[I::Point],
    ranges: &[I::Range],
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    debug_assert_ne!(GROUP, 0);
    let mut sink = RowSink::new(row_offsets, out, sparse_row_mapping);
    let stride = GROUP * I::BLOCK;
    let mut base = 0;

    while base + stride <= codes.len() {
        let mut retained = [I::NO_HITS; GROUP];
        let mut live = false;
        for (block, hits) in retained.iter_mut().enumerate() {
            // SAFETY: the group bound keeps this complete block in `codes`;
            // the target-feature wrapper establishes the ISA precondition.
            *hits = unsafe {
                I::block::<POINTS, RANGES>(
                    codes.as_ptr().add(base + block * I::BLOCK),
                    points,
                    ranges,
                )
            };
            live |= I::any(*hits);
        }
        if live {
            for (block, &hits) in retained.iter().enumerate() {
                if I::any(hits) {
                    I::emit(base + block * I::BLOCK, hits, &mut sink);
                }
            }
        }
        base += stride;
    }

    if I::WALK_REMAINDER {
        while base + I::BLOCK <= codes.len() {
            // SAFETY: the loop bound keeps this complete block in `codes`; the
            // target-feature wrapper establishes the ISA precondition.
            let hits =
                unsafe { I::block::<POINTS, RANGES>(codes.as_ptr().add(base), points, ranges) };
            if I::any(hits) {
                I::emit(base, hits, &mut sink);
            }
            base += I::BLOCK;
        }
    }

    scan_tail(codes, cover, base, &mut sink);
}

/// Prepare a fixed cover shape once, then enter the shared walk.
#[inline(always)]
pub(super) unsafe fn scan_fixed<
    I: Isa,
    O: Offset,
    const POINTS: usize,
    const RANGES: usize,
    const GROUP: usize,
>(
    codes: &[Token],
    row_offsets: &[O],
    cover: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    debug_assert_eq!((cover.points.len(), cover.ranges.len()), (POINTS, RANGES));
    let points: [I::Point; POINTS] = std::array::from_fn(|index| I::point(cover.points[index]));
    let ranges: [I::Range; RANGES] = std::array::from_fn(|index| I::range(cover.ranges[index]));
    // SAFETY: the target-feature leaf establishes the ISA precondition.
    unsafe {
        walk::<I, O, POINTS, RANGES, GROUP>(
            codes,
            row_offsets,
            cover,
            &points,
            &ranges,
            sparse_row_mapping,
            out,
        )
    };
}

/// Prepare an arbitrary cover shape once, then enter the shared walk.
#[inline(always)]
#[allow(
    dead_code,
    reason = "introduced with the template; the AVX2 and SSE2 ports consume it in phase 2"
)]
pub(super) unsafe fn scan_dynamic<I: Isa, O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    cover: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    let points = cover
        .points
        .iter()
        .copied()
        .map(I::point)
        .collect::<Vec<_>>();
    let ranges = cover
        .ranges
        .iter()
        .copied()
        .map(I::range)
        .collect::<Vec<_>>();
    // SAFETY: the target-feature leaf establishes the ISA precondition.
    unsafe {
        walk::<I, O, DYN, DYN, 1>(
            codes,
            row_offsets,
            cover,
            &points,
            &ranges,
            sparse_row_mapping,
            out,
        )
    };
}
