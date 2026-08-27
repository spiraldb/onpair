// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Candidate-row materialization shared by stream-centric kernels.

use super::super::cover::ProbeCover;
use crate::core::offset::Offset;
use crate::core::types::Token;

/// A compact set of hit lanes in ascending bit order.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub(super) struct LaneMask(u64);

#[cfg(target_arch = "x86_64")]
impl LaneMask {
    #[inline]
    pub(super) const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }
}

/// Turns monotonically increasing code indices into ascending, deduplicated row
/// ids.
///
/// Every kernel visits code indices in increasing order, so the owning row only
/// moves forward and a row is finished the moment the scan leaves it. Candidates
/// can therefore be appended as they are discovered, rather than marked in a
/// per-row bitmap that has to be allocated, zeroed, and drained — work
/// proportional to the rows the prefilter rejects.
pub(super) struct RowSink<'a, O> {
    row_offsets: &'a [O],
    out: &'a mut Vec<usize>,
    /// Row owning the most recent hit.
    row: usize,
    /// End of `row`, or zero before the first hit. A hit below this belongs to a
    /// row that has already been appended.
    row_end: usize,
    binary_search_sparse_gaps: bool,
}

impl<'a, O: Offset> RowSink<'a, O> {
    #[inline]
    pub(super) fn new(
        row_offsets: &'a [O],
        out: &'a mut Vec<usize>,
        binary_search_sparse_gaps: bool,
    ) -> Self {
        Self {
            row_offsets,
            out,
            row: 0,
            row_end: 0,
            binary_search_sparse_gaps,
        }
    }

    /// Record a hit at `code_index`, appending its row unless already appended.
    #[inline]
    pub(super) fn hit(&mut self, code_index: usize) {
        if code_index < self.row_end {
            return;
        }
        // A sparse cover can jump across hundreds of thousands of rows between
        // hits. Walking every intervening offset makes candidate materialization
        // O(rows), even when the SIMD scan found only a handful of codes. Use a
        // lower-bound search for large code-space gaps; keep the linear cursor
        // for nearby hits, where its predictable sequential loads are cheaper.
        const BINARY_SEARCH_CODE_GAP: usize = 128;
        if self.binary_search_sparse_gaps
            && code_index.saturating_sub(self.row_end) >= BINARY_SEARCH_CODE_GAP
        {
            let suffix = &self.row_offsets[self.row + 1..];
            self.row += suffix.partition_point(|offset| offset.to_usize() <= code_index);
        } else {
            // Empty rows end at or before `code_index`, so this skips them too.
            while self.row + 1 < self.row_offsets.len()
                && self.row_offsets[self.row + 1].to_usize() <= code_index
            {
                self.row += 1;
            }
        }
        // `code_index` is a valid code index, so it lies below the last row
        // offset and the loop above always stops with `row + 1` in bounds.
        self.out.push(self.row);
        self.row_end = self.row_offsets[self.row + 1].to_usize();
    }

    /// Record the hit rows named by a compact, ascending lane mask.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    pub(super) fn mark_mask(&mut self, base: usize, mask: LaneMask) {
        let mut lanes = mask.0;
        loop {
            // A previous hit may have emitted a row extending into this block.
            let consumed = self.row_end.saturating_sub(base);
            if consumed >= u64::BITS as usize {
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

/// Map a SIMD block's non-zero hit lanes to rows.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) fn mark_block<O: Offset>(base: usize, hits: &[u16], sink: &mut RowSink<'_, O>) {
    for (lane, &hit) in hits.iter().enumerate() {
        if hit != 0 {
            sink.hit(base + lane);
        }
    }
}

/// Scan the final partial SIMD block.
#[inline]
pub(super) fn scan_tail<O: Offset>(
    codes: &[Token],
    cover: &ProbeCover,
    from: usize,
    sink: &mut RowSink<'_, O>,
) {
    for (offset, &code) in codes[from..].iter().enumerate() {
        if cover.table[code as usize] {
            sink.hit(from + offset);
        }
    }
}
