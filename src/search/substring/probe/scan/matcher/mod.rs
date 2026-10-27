// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Membership kernels: mark the codes of a block that belong to the filter.
//!
//! Every kernel computes the same mask, with bit `i` set iff code `i` is covered.
//! `table` holds the portable kernel, a scalar lookup table. The vector kernels have
//! one file per instruction set, `neon`, `avx2` and `avx512`, each with the same
//! items:
//!
//! - `EqOr` broadcasts each point to all lanes, compares it with the codes, and ORs
//!   the results together with the range hits. Work grows linearly with the number
//!   of points and ranges. Covers without points get a specialization that only
//!   checks ranges.
//! - `NibbleN8` tests up to eight points at once with shuffle tables. A token ID is
//!   a `u16`, so four 4-bit nibbles. Each batch has one 16-entry table per nibble
//!   position, and point `k` sets bit `k` in the entry selected by its nibble at
//!   each position. Looking up a code's four nibbles and ANDing the entries leaves
//!   bit `k` iff the code equals point `k`. Batches are ORed together, then range
//!   hits are added. Planning caps the batch count per instruction set.
//! - `check_ranges` adds the cover's ranges to either kernel's point hits, and
//!   `ranges_mask` handles covers with ranges only. A code is in `[begin, last]` iff
//!   `code.wrapping_sub(begin) <= last - begin`: codes below `begin` wrap around to
//!   large values, so one unsigned comparison suffices.
//! - A kernel maps `Vectors`, 64 codes, to `Hits`. `words` runs it over a block,
//!   128 codes at a time, and packs the hits into mask words; when skipping is
//!   enabled, groups without hits are zeroed without packing.
//!
//! AArch64 builds contain the `neon` kernels, which every AArch64 CPU runs. x86_64
//! builds contain both the `avx2` and the `avx512` kernels, each compiled for its
//! own instruction set whatever the build's target features, and `dispatch` runs
//! the best one the CPU supports. Each file names its matchers in a `Kernels` impl
//! (`Neon`, `Avx2`, `Avx512`), so dispatch and the tests are written once for all.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "aarch64")]
mod neon;
mod table;

use super::{Block, Mask};
use crate::search::substring::ProbeCover;

/// Number of points in one batch of nibble tables.
pub(in crate::search::substring::probe) const PER_BATCH: usize = 8;
#[cfg(target_arch = "x86_64")]
pub(in crate::search::substring::probe::scan) use avx2::Avx2;
#[cfg(target_arch = "x86_64")]
pub(in crate::search::substring::probe::scan) use avx512::Avx512;
#[cfg(target_arch = "aarch64")]
pub(in crate::search::substring::probe::scan) use neon::Neon;
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

/// The vector matchers of one instruction set.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(in crate::search::substring::probe::scan) trait Kernels {
    /// The instruction set the matchers run on.
    const ISA: super::Isa;
    /// One vector comparison per point, plus range checks.
    type EqOr<const SKIP_MOVEMASK_IF_NO_MATCH: bool, const HAS_POINTS: bool>: Matcher;
    /// Nibble shuffle tables, eight points per batch, plus range checks.
    type NibbleN8<const BATCHES: usize, const SKIP_MOVEMASK_IF_NO_MATCH: bool>: Matcher;
}

#[cfg(test)]
mod tests;
