// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The row galloped to, not walked.

use super::shared::Cursor;
use super::{Check, Mask, Resolver};
use crate::core::offset::Offset;

/// The row of a hit found by doubling the step from the cursor until the
/// offsets overshoot, then binary searching that bracket: log(rows crossed)
/// instead of one step each. What a sparse mask over short rows wants.
pub(in crate::search::prefilter::scan) struct GallopSeek<'a, O>(Cursor<'a, O>);

impl<'a, O: Offset> Resolver<'a> for GallopSeek<'a, O> {
    type Offset = O;

    fn new(row_offsets: &'a [O]) -> Self {
        Self(Cursor::new(row_offsets))
    }

    fn rows(&mut self, bits: &Mask, first_code: usize, check: Check<'_>, out: &mut Vec<usize>) {
        self.0.rows(bits, first_code, check, out, gallop);
    }
}

/// The row holding `code`, from a cursor at or before it.
#[inline]
fn gallop<O: Offset>(row_offsets: &[O], from: usize, code: usize) -> usize {
    let mut step = 1;
    while from + step < row_offsets.len() && row_offsets[from + step].to_usize() <= code {
        step *= 2;
    }
    let lo = from + step / 2;
    let hi = (from + step).min(row_offsets.len());
    lo + row_offsets[lo + 1..hi].partition_point(|&offset| offset.to_usize() <= code)
}
