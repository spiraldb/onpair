// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Stage two: bit mask in, candidate rows out.

mod gallop_seek;
mod linear_seek;
mod shared;
#[cfg(test)]
mod tests;

use super::{Check, Mask};
use crate::core::offset::Offset;

pub(in crate::search::substring::prefilter::scan) use gallop_seek::GallopSeek;
pub(in crate::search::substring::prefilter::scan) use linear_seek::LinearSeek;

/// Appends the rows the set bits fall in, ascending and without repeats, one
/// block of mask at a time. Blocks arrive in stream order, only the non-empty
/// ones arrive at all, and a row can span several, so the resolver keeps
/// whatever state it needs across calls. `first_code` is the stream index of
/// bit 0. The row layer covers every code a bit can be set for, so a set bit
/// always has a row. A row goes out on its first bit that `check` passes,
/// and its further bits are never read.
pub(in crate::search::substring::prefilter::scan) trait Resolver<'a>:
    Sized
{
    /// The width the row layer is stored at, which the resolver reads in
    /// place: no copy of the layer, at either width.
    type Offset: Offset;

    fn new(row_offsets: &'a [Self::Offset]) -> Self;

    fn rows(&mut self, bits: &Mask, first_code: usize, check: Check<'_>, out: &mut Vec<usize>);
}

/// The rows a mask names, worked out in one pass from the layer rather than
/// walked: what a resolver has to arrive at, without being one. The bench and
/// the tests check against this, so a resolver is never checked against a
/// sibling that could be wrong the same way. Nothing outside them asks.
#[cfg(test)]
pub(in crate::search::substring::prefilter::scan) fn expected_rows<O: Offset>(
    mask: &[u64],
    row_offsets: &[O],
) -> Vec<usize> {
    let mut rows = Vec::new();
    for (word, &set) in mask.iter().enumerate() {
        let mut set = set;
        while set != 0 {
            let code = word * 64 + set.trailing_zeros() as usize;
            set &= set - 1;
            // The first offset past the code, less one, is the row holding it;
            // an empty row shares its start and loses the tie, as it should.
            let row = row_offsets.partition_point(|&offset| offset.to_usize() <= code) - 1;
            if rows.last() != Some(&row) {
                rows.push(row);
            }
        }
    }
    rows
}
