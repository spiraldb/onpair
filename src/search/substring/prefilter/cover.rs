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

    /// Turn each maximal run in `table` into a range, or a point for a singleton.
    /// This removes overlapping probes while preserving membership exactly;
    /// advisory frequencies do not participate.
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
            let probe = TokenRange {
                begin: begin as Token,
                last: last as Token,
            };
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

    /// Whether the cover names no token id.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty() && self.ranges.is_empty()
    }
}
