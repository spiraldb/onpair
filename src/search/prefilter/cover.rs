// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The compiled probe cover: what the scan compares codes against.
//!
//! A cover carries the same membership set in two shapes, because the two scan
//! paths want different ones. The vector kernels compare each code against every
//! point (equality) and every range (unsigned `>= lo && <= hi`), so they want the
//! point and range lists; the scalar tail wants one lookup per code, so it wants
//! the membership table. The table is what the planner hands over, and the lists
//! are read back off it.

use crate::core::types::{Token, TokenRange};

/// A sound probe cover over dictionary token ids.
///
/// Sound means every row containing the pattern holds at least one covered
/// token, so a scan for these ids drops no true match. Nothing here enforces
/// that — it is established by whoever selects the ids.
#[derive(Debug, Clone)]
pub(super) struct ProbeCover {
    pub(super) points: Vec<Token>,
    pub(super) ranges: Vec<TokenRange>,
    pub(super) table: Vec<bool>,
}

impl ProbeCover {
    /// Describe the ids `table` marks with as few probes as they can be
    /// described: every maximal run of set ids becomes one [`TokenRange`], or a
    /// point when the run is a single id.
    ///
    /// The ids arrive as *overlapping* sets — a cut's range probe and point
    /// probe can name the same token, two ranges can abut, and the mandatory
    /// contained tokens are unioned in on top — so probing for them as selected
    /// would have the kernels compare against ids they already cover. Reading
    /// runs back off the table makes [`cmp_cost`](Self::cmp_cost) minimal for
    /// the set, and a comparison saved here is one saved per vector of the
    /// entire code stream.
    pub(super) fn from_membership(table: Vec<bool>) -> Self {
        let mut points = Vec::new();
        let mut ranges = Vec::new();
        let mut id = 0;
        while id < table.len() {
            if !table[id] {
                id += 1;
                continue;
            }
            let begin = id;
            while id < table.len() && table[id] {
                id += 1;
            }
            let last = id - 1;
            if last == begin {
                points.push(begin as Token);
            } else {
                ranges.push(TokenRange {
                    begin: begin as Token,
                    last: last as Token,
                });
            }
        }

        Self {
            points,
            ranges,
            table,
        }
    }

    /// Per-vector comparison budget: one compare per point or range.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[inline]
    pub(super) fn cmp_cost(&self) -> usize {
        self.points.len() + self.ranges.len()
    }
}
