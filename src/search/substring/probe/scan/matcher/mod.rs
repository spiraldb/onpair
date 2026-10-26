// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Membership kernels: mark the codes of a block that belong to the filter.
//!
//! Every kernel computes the same mask, with bit `i` set iff code `i` is covered:
//! `table` with a scalar lookup table, `eq_or` with one vector comparison per point,
//! and `nibble_n8` with shuffle tables for up to eight points at once. `range` adds
//! the cover's ranges to the vector kernels, and `shared` holds their loads and
//! mask packing.

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
mod eq_or;
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
mod nibble_n8;
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
mod range;
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
mod shared;
mod table;

use super::{Block, Mask};
use crate::search::substring::ProbeCover;

/// Number of points in one batch of nibble tables.
pub(in crate::search::substring::probe) const PER_BATCH: usize = 8;
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(in crate::search::substring::probe::scan) use eq_or::EqOr;
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(in crate::search::substring::probe::scan) use nibble_n8::NibbleN8;
pub(in crate::search::substring::probe::scan) use table::Table;

/// A cover membership test, prepared once and run on every block.
pub(in crate::search::substring::probe::scan) trait Matcher:
    Sized
{
    /// Prepares `cover`. A vector matcher may only be built for a cover and CPU
    /// that planning found eligible.
    fn new(cover: &ProbeCover) -> Self;

    /// Overwrites `bits` with the block's mask. Returns `false` only if the mask is
    /// empty, so the caller can skip row lookup; `true` does not imply a hit.
    fn check(&self, codes: &Block, bits: &mut Mask) -> bool;
}

#[cfg(test)]
mod tests;
