// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Membership kernels, which mark the codes of a block that belong to the cover.
//!
//! Every kernel computes the same mask, with bit `i` set iff code `i` is covered.
//! `table` holds the portable kernel, a scalar lookup table. The vector kernels have
//! one file per instruction set, `neon`, `avx2` and `avx512`. The files define the
//! same items, described here once; each documents only what is specific to its
//! instruction set.
//!
//! - `EqOr` compares the codes with each point, broadcast to all lanes by
//!   `broadcast`, and ORs the results (`eq_or_hits`) together with the range hits.
//!   Work grows linearly with the number of points and ranges; a cover without
//!   points checks only its ranges.
//! - `Nibble` tests up to eight points at once with shuffle tables. A token ID is
//!   a `u16`, so four 4-bit nibbles. Each batch has one 16-entry table per nibble
//!   position, and point `k` sets bit `k` in the entry selected by its nibble at
//!   each position. Looking up a code's four nibbles and ANDing the entries leaves
//!   bit `k` iff the code equals point `k`. `batch` loads the rows `nibble_rows`
//!   builds into one `Table` each, and `probe` looks a group's nibbles up in every
//!   batch. Batches are ORed together, then range hits are added. `BATCHES` must
//!   hold every point of the cover: planning caps it per instruction set, and
//!   `dispatch` picks the specialization.
//! - `hold` broadcasts a range's start and width, and `inside` tests a group of
//!   codes against them. A code is in `[begin, last]` iff
//!   `code.wrapping_sub(begin) <= last - begin`: codes below `begin` wrap around to
//!   large values, so one unsigned comparison suffices. `check_ranges` adds the
//!   cover's ranges to either kernel's point hits, and `ranges_mask` handles covers
//!   with ranges only, clearing the mask if there are none.
//! - A kernel maps `Vectors`, a group of 64 codes read by `load`, to `Hits`, which
//!   `or` combines. `words` runs it over a block, a pair of groups at a time, and
//!   `movemask` packs each group's hits into a mask word; when skipping is enabled,
//!   a pair without hits (`any`) is zeroed without packing. `words` returns `false`
//!   only if the mask is empty, and without skipping always returns `true`.
//! - `eq_or_mask`, `nibble_mask` and `ranges_mask` are the `#[target_feature]`
//!   functions each matcher's `check` calls. The kernel and `words` inline into
//!   them, so the loop runs with the instruction set enabled. The closures passed
//!   to `words` are `#[inline(always)]`, since `words` calls each twice per pair
//!   and the compiler would otherwise keep it out of line.
//!
//! Each vector file is compiled for its own instruction set, whatever the build's
//! target features, and names its matchers in a `Kernels` impl (`Neon`, `Avx2`,
//! `Avx512`), so `dispatch`, which picks one, is written once for all.

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
    /// tests each pair of 64-code groups for a hit and zeroes their mask words
    /// without packing if there is none. On AVX-512 the hits already are mask words,
    /// so skipping saves no packing there and only lets `check` report an empty
    /// block. Returns `false` only if the mask is empty, so the caller can skip row
    /// lookup; `true` does not imply a hit.
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
    type Nibble<const BATCHES: usize>: Matcher + 'static;
}

/// The tables of `Nibble`'s batch `at`, which holds the points from
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
