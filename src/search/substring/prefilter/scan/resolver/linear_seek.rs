// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The row walked to, one row at a time.

use super::shared::Cursor;
use super::{Check, Mask, Resolver};
use crate::core::offset::Offset;

/// One step per row crossed: what a dense mask over long rows wants.
pub(in crate::search::substring::prefilter::scan) struct LinearSeek<'a, O>(Cursor<'a, O>);

impl<'a, O: Offset> Resolver<'a> for LinearSeek<'a, O> {
    type Offset = O;

    fn new(row_offsets: &'a [O]) -> Self {
        Self(Cursor::new(row_offsets))
    }

    fn rows(&mut self, bits: &Mask, first_code: usize, check: Check<'_>, out: &mut Vec<usize>) {
        self.0.rows(bits, first_code, check, out, walk);
    }
}

/// The row holding `code`, from a cursor at or before it.
#[inline]
fn walk<O: Offset>(row_offsets: &[O], from: usize, code: usize) -> usize {
    let mut row = from;
    while row_offsets[row + 1].to_usize() <= code {
        row += 1;
    }
    row
}
