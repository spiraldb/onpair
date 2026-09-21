// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Test token membership in a probe cover, one block at a time.
//!
//! Each matcher writes the same exact mask: bit `i` is set when code `i`
//! belongs to a point or range in the cover. This establishes a candidate
//! token, not a complete substring match; row resolution invokes the walker.
//!
//! `eq_or` compares each point, `nibble_n8` groups points into shuffle tables,
//! `range` checks inclusive intervals, and `table` uses scalar lookups.
//! `shared` handles vector loads and packing their results into mask words.

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
mod eq_or;
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
mod nibble_n8;
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
mod range;
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
mod shared;
mod table;
#[cfg(test)]
mod tests;

use super::{Block, Mask};
use crate::search::substring::ProbeCover;

/// Point probes represented by one batch of nibble tables.
pub(in crate::search::substring) const PER_BATCH: usize = 8;
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(in crate::search::substring::scan) use eq_or::EqOr;
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(in crate::search::substring::scan) use nibble_n8::NibbleN8;
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(in crate::search::substring::scan) use range::Range;
pub(in crate::search::substring::scan) use table::Table;

/// Prepared membership checks with one output bit per input token code.
pub(in crate::search::substring::scan) trait Matcher:
    Sized
{
    /// Prepare the selected cover for repeated block checks.
    /// The caller must establish the supported shape and CPU features through
    /// planning and capability detection before constructing a vector matcher.
    fn new(cover: &ProbeCover) -> Self;

    /// Overwrite every mask word for this block.
    /// `false` guarantees that all bits are zero, allowing row lookup to be
    /// skipped. `true` allows an empty mask when the matcher omits that test.
    fn check(&self, codes: &Block, bits: &mut Mask) -> bool;
}
