// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Stage one: codes in, bit mask out.

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

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(in crate::search::substring::scan) use eq_or::EqOr;
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(in crate::search::substring) use nibble_n8::{MAX_BATCHES, PER_BATCH};
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(in crate::search::substring::scan) use range::Range;
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(in crate::search::substring) use shared::available as simd_available;
pub(in crate::search::substring::scan) use table::Table;

/// Bit `i` of `bits` is set iff the cover admits `codes[i]`. See `README.md`.
pub(in crate::search::substring::scan) trait Matcher:
    Sized
{
    /// Callers check `policy::takes` first.
    fn new(cover: &ProbeCover) -> Self;

    /// Fills `bits` for one block. `false` promises an empty mask, so stage two
    /// is skipped; `true` promises nothing.
    fn check(&self, codes: &Block, bits: &mut Mask) -> bool;
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(in crate::search::substring::scan) use nibble_n8::NibbleN8;
