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
pub struct ProbeCover {
    pub(super) points: Vec<Token>,
    pub(super) ranges: Vec<TokenRange>,
    pub(super) table: Vec<bool>,
}

impl ProbeCover {
    /// Equality probes issued for every SIMD vector.
    pub fn points(&self) -> &[Token] {
        &self.points
    }

    /// Inclusive range probes issued for every SIMD vector.
    pub fn ranges(&self) -> &[TokenRange] {
        &self.ranges
    }

    /// Describe the ids `table` marks with as few probes as they can be
    /// described: every maximal run of set ids becomes one [`TokenRange`], or a
    /// point when the run is a single id.
    ///
    /// The ids arrive as *overlapping* sets — a cut's range probe and point
    /// probe can name the same token, two ranges can abut, and the mandatory
    /// contained tokens are unioned in on top — so probing for them as selected
    /// would have the kernels compare against ids they already cover. Reading
    /// runs back off the table minimizes the cover's checks, and a comparison
    /// saved here is one saved per vector of the entire code stream.
    ///
    /// `probe_for` gets each run and answers with the sub-run to actually probe
    /// for, or `None` to drop it; whatever it gives up is cleared from the table
    /// too, so the two shapes keep describing the same ids. Deciding per run,
    /// rather than per id, is what keeps a decision from costing more than it
    /// saves: runs are the unit the kernels are charged for, so dropping or
    /// narrowing one can only help, where dropping an id from the middle of one
    /// would split it and add a comparison.
    pub(super) fn from_membership(
        mut table: Vec<bool>,
        probe_for: impl Fn(TokenRange) -> Option<TokenRange>,
    ) -> Self {
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
            let run = TokenRange {
                begin: begin as Token,
                last: last as Token,
            };

            let Some(probe) = probe_for(run) else {
                table[begin..=last].fill(false);
                continue;
            };
            debug_assert!(
                !probe.is_empty() && probe.begin >= run.begin && probe.last <= run.last,
                "probe_for widened a run"
            );
            table[begin..probe.begin as usize].fill(false);
            table[probe.last as usize + 1..last + 1].fill(false);
            if probe.last == probe.begin {
                points.push(probe.begin);
            } else {
                ranges.push(probe);
            }
        }

        Self {
            points,
            ranges,
            table,
        }
    }

    /// Whether the cover names no id at all.
    ///
    /// Reachable only through pruning: the planner drops ids the code stream
    /// never uses, and every id a pattern's cover would have named can be one of
    /// those. It is the strongest answer the prefilter can give — no row holds a
    /// covered code, so no row matches, and no scan is needed to find that out.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty() && self.ranges.is_empty()
    }
}
