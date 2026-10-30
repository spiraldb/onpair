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
//!   of points and ranges; a cover without points checks only its ranges.
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
//!   enabled, groups without hits are zeroed without packing. The closures passed
//!   to `words` are `#[inline(always)]`, since `words` calls each twice per 128
//!   codes and the compiler would otherwise keep it out of line.
//!
//! AArch64 builds contain the `neon` kernels. x86_64 builds contain both the
//! `avx2` and the `avx512` kernels, each compiled for its own instruction set
//! whatever the build's target features. `dispatch` builds the best kernels the
//! CPU supports, once per query, and checks that support first. Each file names its
//! matchers in a `Kernels` impl (`Neon`, `Avx2`, `Avx512`), so dispatch is written
//! once for all.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "aarch64")]
mod neon;
mod table;

use super::{Block, Mask};
use crate::core::types::Token;
use crate::search::substring::ProbeCover;

/// Number of points in one batch of nibble tables.
pub(crate) const PER_BATCH: usize = 8;
#[cfg(target_arch = "x86_64")]
pub(super) use avx2::Avx2;
#[cfg(target_arch = "x86_64")]
pub(super) use avx512::Avx512;
#[cfg(target_arch = "aarch64")]
pub(super) use neon::Neon;
pub(super) use table::Table;

/// A cover membership test, prepared once per query and run on every block.
///
/// A vector matcher runs the instructions of its instruction set. Building one is
/// the unsafe step: once built, it may be checked anywhere.
pub(crate) trait Matcher: Send + Sync {
    /// Prepares `cover`, which must be eligible for this matcher.
    ///
    /// # Safety
    /// The CPU supports the matcher's instruction set. The table needs none.
    unsafe fn new(cover: &ProbeCover) -> Self
    where
        Self: Sized;

    /// Overwrites `bits` with the block's mask. With `skip_empty`, a vector matcher
    /// tests each 128 codes for a hit and zeroes their mask words without packing
    /// if there is none. Returns `false` only if the mask is empty, so the caller
    /// can skip row lookup; `true` does not imply a hit.
    fn check(&self, codes: &Block, bits: &mut Mask, skip_empty: bool) -> bool;
}

/// The vector matchers of one instruction set.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(super) trait Kernels {
    /// The instruction set the matchers run on.
    const ISA: super::Isa;
    /// One vector comparison per point, plus range checks.
    type EqOr: Matcher + 'static;
    /// Nibble shuffle tables, eight points per batch, plus range checks.
    type NibbleN8<const BATCHES: usize>: Matcher + 'static;
}

/// The tables of `NibbleN8`'s batch `at`, which holds the points from
/// `at * PER_BATCH` on: one 16-entry row per nibble position, from the least to the
/// most significant. Point `k` of the batch sets bit `k` in the entry its nibble
/// selects. A batch past the last point is all zero, and matches no code.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn nibble_rows(points: &[Token], at: usize) -> [[u8; 16]; 4] {
    let from = (at * PER_BATCH).min(points.len());
    let to = (from + PER_BATCH).min(points.len());
    let mut rows = [[0u8; 16]; 4];
    for (k, &point) in points[from..to].iter().enumerate() {
        for (shift, row) in rows.iter_mut().enumerate() {
            row[usize::from(point >> (4 * shift) & 0x0f)] |= 1 << k;
        }
    }
    rows
}

#[cfg(test)]
mod tests;
