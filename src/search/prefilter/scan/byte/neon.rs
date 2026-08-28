// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! AArch64 NEON byte kernel: sixteen `u8` lanes per vector.

use core::arch::aarch64::{
    uint8x16_t, vceqq_u8, vcleq_u8, vdupq_n_u8, vget_lane_u64, vld1q_u8, vmaxvq_u32, vorrq_u8,
    vreinterpret_u64_u8, vreinterpretq_u16_u8, vreinterpretq_u32_u8, vshrn_n_u16, vsubq_u8,
};

use super::super::ScanInput;
use super::super::sink::RowSink;
use crate::core::offset::Offset;
use crate::search::prefilter::cover::ProbeCover;

/// Widest point-only cover the compact dynamic instantiation serves.
const MAX_FEW_POINTS: usize = 8;

/// The cover's probes narrowed to the byte domain, ranges as `(begin, span)`.
///
/// A byte stream cannot name an id above `u8::MAX`, so probes beyond it match
/// nothing and are dropped; a range straddling the boundary keeps its low part.
fn byte_probes(cover: &ProbeCover) -> (Vec<u8>, Vec<(u8, u8)>) {
    const LAST: u16 = u8::MAX as u16;
    let points = cover
        .points
        .iter()
        .filter(|&&point| point <= LAST)
        .map(|&point| point as u8)
        .collect();
    let ranges = cover
        .ranges
        .iter()
        .filter(|range| range.begin <= LAST)
        .map(|range| {
            (
                range.begin as u8,
                (range.last.min(LAST) - range.begin) as u8,
            )
        })
        .collect();
    (points, ranges)
}

/// NEON has no movemask. Narrowing the sixteen byte lanes to nibbles packs a
/// block into one register-resident word, which doubles as the hit test and as
/// the mask the sink walks — no horizontal reduction and no stack round-trip.
#[inline(always)]
fn nibble_mask(acc: uint8x16_t) -> u64 {
    // SAFETY: reinterpretation and a narrowing shift, both total over any input.
    unsafe {
        vget_lane_u64::<0>(vreinterpret_u64_u8(vshrn_n_u16::<4>(vreinterpretq_u16_u8(
            acc,
        ))))
    }
}

/// Extract and consume a hit pair's lanes.
///
/// Out of line so the narrowing shifts and their vector-to-scalar transfers stay
/// off the miss path — left inline they are hoisted above the hit test, and the
/// miss path is where a selective cover spends all of its time.
#[inline(never)]
fn mark_pair<O: Offset>(
    base: usize,
    acc0: uint8x16_t,
    acc1: uint8x16_t,
    sink: &mut RowSink<'_, O>,
) {
    let m0 = nibble_mask(acc0);
    if m0 != 0 {
        sink.mark_nibbles(base, m0);
    }
    let m1 = nibble_mask(acc1);
    if m1 != 0 {
        sink.mark_nibbles(base + 16, m1);
    }
}

/// Shared two-vector walk. `matching_masks` is always inlined, leaving each
/// caller with a shape-specific compare loop and no indirect calls.
///
/// Two independent accumulators combine in one `vorrq`, so the hit test never
/// serializes behind a running accumulator, and `GATED` shapes reach the lane
/// extraction only when the pair actually hit.
#[inline(always)]
fn scan_two_vectors<O: Offset, const GATED: bool>(
    codes: &[u8],
    sink: &mut RowSink<'_, O>,
    mut matching_masks: impl FnMut(uint8x16_t, uint8x16_t) -> (uint8x16_t, uint8x16_t),
) -> usize {
    let total = codes.len();
    let base = codes.as_ptr();
    let mut i = 0usize;
    while i + 32 <= total {
        // SAFETY: `i + 32 <= total`; both vector loads are in bounds.
        let (acc0, acc1) = unsafe {
            let (v0, v1) = (vld1q_u8(base.add(i)), vld1q_u8(base.add(i + 16)));
            matching_masks(v0, v1)
        };
        // Lanes are all-zero or all-ones, so a four-lane `u32` reduction answers
        // the same question as a sixteen-lane `u8` one, two tree levels cheaper.
        // A cover dense enough to hit most pairs pays that for nothing, so those
        // shapes go straight to extraction instead.
        let hit = !GATED || unsafe { vmaxvq_u32(vreinterpretq_u32_u8(vorrq_u8(acc0, acc1))) } != 0;
        if hit {
            mark_pair(i, acc0, acc1, sink);
        }
        i += 32;
    }
    i
}

/// Point-only covers. `N` non-zero unrolls the compare chain outright; `N == 0`
/// is the compact dynamic instantiation the wider shapes share. Kept out of line
/// so each specialization gets its own register allocation.
#[inline(never)]
fn scan_points<O: Offset, const N: usize>(
    codes: &[u8],
    points: &[u8],
    sink: &mut RowSink<'_, O>,
) -> usize {
    let probe_count = if N == 0 {
        points.len()
    } else {
        debug_assert_eq!(points.len(), N);
        N
    };
    // SAFETY: broadcasts are total; `probe_count <= MAX_FEW_POINTS` by dispatch.
    let mut probes = [unsafe { vdupq_n_u8(0) }; MAX_FEW_POINTS];
    for (probe, &point) in probes.iter_mut().zip(points) {
        *probe = unsafe { vdupq_n_u8(point) };
    }
    let probes = &probes[..probe_count];
    scan_two_vectors::<O, true>(codes, sink, |v0, v1| unsafe {
        let mut acc0 = vdupq_n_u8(0);
        let mut acc1 = vdupq_n_u8(0);
        for &probe in probes {
            acc0 = vorrq_u8(acc0, vceqq_u8(v0, probe));
            acc1 = vorrq_u8(acc1, vceqq_u8(v1, probe));
        }
        (acc0, acc1)
    })
}

/// Arbitrary-shape fallback: every broadcast is still hoisted out of the walk,
/// only the compare/OR loops stay dynamic.
#[inline(never)]
fn scan_generic<O: Offset>(
    codes: &[u8],
    points: &[u8],
    ranges: &[(u8, u8)],
    sink: &mut RowSink<'_, O>,
) -> usize {
    // SAFETY: broadcasts are total over any byte.
    let points: Vec<uint8x16_t> = points.iter().map(|&p| unsafe { vdupq_n_u8(p) }).collect();
    let ranges: Vec<(uint8x16_t, uint8x16_t)> = ranges
        .iter()
        .map(|&(begin, span)| unsafe { (vdupq_n_u8(begin), vdupq_n_u8(span)) })
        .collect();
    scan_two_vectors::<O, false>(codes, sink, |v0, v1| unsafe {
        let mut acc0 = vdupq_n_u8(0);
        let mut acc1 = vdupq_n_u8(0);
        for &probe in &points {
            acc0 = vorrq_u8(acc0, vceqq_u8(v0, probe));
            acc1 = vorrq_u8(acc1, vceqq_u8(v1, probe));
        }
        for &(lo, span) in &ranges {
            acc0 = vorrq_u8(acc0, vcleq_u8(vsubq_u8(v0, lo), span));
            acc1 = vorrq_u8(acc1, vcleq_u8(vsubq_u8(v1, lo), span));
        }
        (acc0, acc1)
    })
}

/// Walk the flat code stream a vector at a time, one comparison per point and
/// two per range, and append the rows the surviving lanes fall in.
pub(super) fn scan<O: Offset>(
    input: ScanInput<'_, O, u8>,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    let ScanInput {
        codes,
        row_offsets,
        cover,
    } = input;
    let (points, ranges) = byte_probes(cover);
    let mut sink = RowSink::new(row_offsets, out, sparse_row_mapping);
    let consumed = match (points.len(), ranges.len()) {
        (1, 0) => scan_points::<O, 1>(codes, &points, &mut sink),
        (2, 0) => scan_points::<O, 2>(codes, &points, &mut sink),
        (3, 0) => scan_points::<O, 3>(codes, &points, &mut sink),
        (4, 0) => scan_points::<O, 4>(codes, &points, &mut sink),
        (5..=MAX_FEW_POINTS, 0) => scan_points::<O, 0>(codes, &points, &mut sink),
        _ => scan_generic(codes, &points, &ranges, &mut sink),
    };
    for (offset, &code) in codes[consumed..].iter().enumerate() {
        if cover.table[code as usize] {
            sink.hit(consumed + offset);
        }
    }
}
