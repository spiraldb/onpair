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
        // Gallop from the cursor, then binary search the bracket that overshoots:
        // the walk costs log(gap) rather than log(rows), so a sparse cover pays
        // for its own gap and not for the column behind it.
        if self.binary_search_sparse_gaps {
            let base = self.row;
            let mut step = 1;
            while base + step < self.row_offsets.len()
                && self.row_offsets[base + step].to_usize() <= code_index
            {
                step *= 2;
            }
            let lo = base + step / 2;
            let hi = (base + step).min(self.row_offsets.len());
            let bracket = &self.row_offsets[lo + 1..hi];
            self.row = lo + bracket.partition_point(|offset| offset.to_usize() <= code_index);
        } else {
            // Empty rows end at or before `code_index`, so this skips them too.
            while self.row + 1 < self.row_offsets.len()
                && self.row_offsets[self.row + 1].to_usize() <= code_index
            {
                self.row += 1;
            }
        }
        // `code_index` lies below the last row offset, so `row + 1` stays in bounds.
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

    /// Record the hit rows named by a NEON block's nibble mask: four bits per
    /// lane, all set when the lane hit.
    #[cfg(target_arch = "aarch64")]
    #[inline]
    pub(super) fn mark_nibbles(&mut self, base: usize, mask: u64) {
        const LANES: usize = 16;
        let mut lanes = mask;
        loop {
            // A previous hit may have emitted a row extending into this block.
            let consumed = self.row_end.saturating_sub(base);
            if consumed >= LANES {
                return;
            }
            lanes &= u64::MAX << (4 * consumed);
            if lanes == 0 {
                return;
            }
            self.hit(base + lanes.trailing_zeros() as usize / 4);
        }
    }
}

/// Map a SIMD block's non-zero hit lanes to rows.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) fn mark_block<O: Offset, T: Copy + Default + PartialEq>(
    base: usize,
    hits: &[T],
    sink: &mut RowSink<'_, O>,
) {
    let zero = T::default();
    for (lane, &hit) in hits.iter().enumerate() {
        if hit != zero {
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
