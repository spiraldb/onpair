// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The loop both resolvers run; they differ only in how a hit's row is found.

use super::{Check, Mask};
use crate::core::offset::Offset;

/// A row is emitted on its first hit that passes the check and the bit
/// search resumes at the row's end, so its further hits are never read. A
/// hit that fails moves the search on by one bit.
pub(super) struct Cursor<'a, O> {
    row_offsets: &'a [O],
    row: usize,
    /// One past the last code of `row`, in stream indices. Zero before the
    /// first hit.
    row_end: usize,
    /// `row_end` of the last row emitted; a bit below it is a repeat of a row
    /// already out. Zero before the first emitted row.
    resolved_code: usize,
}

impl<'a, O: Offset> Cursor<'a, O> {
    pub(super) fn new(row_offsets: &'a [O]) -> Self {
        Self {
            row_offsets,
            row: 0,
            row_end: 0,
            resolved_code: 0,
        }
    }

    /// `find(row_offsets, from, code)` is the row holding `code`, from a
    /// cursor at or before it.
    #[inline]
    pub(super) fn rows(
        &mut self,
        bits: &Mask,
        first_code: usize,
        check: Check<'_>,
        out: &mut Vec<usize>,
        find: impl Fn(&[O], usize, usize) -> usize,
    ) {
        let mut at = self.resolved_code.saturating_sub(first_code);
        while let Some(hit) = next_set(bits, at) {
            let code = first_code + hit;
            if code >= self.row_end {
                self.row = find(self.row_offsets, self.row, code);
                self.row_end = self.row_offsets[self.row + 1].to_usize();
            }
            if check.passes(code, self.row_offsets[self.row].to_usize(), self.row_end) {
                out.push(self.row);
                self.resolved_code = self.row_end;
                at = self.row_end - first_code;
            } else {
                at = hit + 1;
            }
        }
    }
}

/// The set bit at or after `from`, `None` if the block holds none past it.
#[inline]
fn next_set(bits: &Mask, from: usize) -> Option<usize> {
    let mut word = from / 64;
    let mut set = bits.get(word)? & (u64::MAX << (from % 64));
    while set == 0 {
        word += 1;
        set = *bits.get(word)?;
    }
    Some(word * 64 + set.trailing_zeros() as usize)
}
