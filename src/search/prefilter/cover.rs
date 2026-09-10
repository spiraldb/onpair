// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The compiled probe cover: what the scan compares codes against.
//!
//! A cover carries the token ids in the two shapes the kernels compare against:
//! points, tested for equality, and inclusive ranges, tested as unsigned
//! `>= lo && <= hi`.

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

    /// Whether the cover names no token id.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty() && self.ranges.is_empty()
    }

    /// Merge runs that overlap or abut, then file single-id runs as points and
    /// the rest as ranges. Input may be in any order. The planner reaches this
    /// through [`from_edge_cut`](Self::from_edge_cut), defined beside the graph.
    pub(super) fn from_runs(mut runs: Vec<TokenRange>) -> Self {
        runs.sort_unstable_by_key(|run| run.begin);
        let mut merged: Vec<TokenRange> = Vec::with_capacity(runs.len());
        for run in runs {
            match merged.last_mut() {
                Some(open) if run.begin <= open.last.saturating_add(1) => {
                    open.last = open.last.max(run.last)
                }
                _ => merged.push(run),
            }
        }
        let (points, ranges): (Vec<_>, Vec<_>) =
            merged.into_iter().partition(|run| run.begin == run.last);
        let points = points.into_iter().map(|run| run.begin).collect();
        Self { points, ranges }
    }

    /// Points and ranges as given. Ranges must be disjoint.
    #[cfg(test)]
    pub(super) fn new(points: Vec<Token>, ranges: Vec<TokenRange>) -> Self {
        Self { points, ranges }
    }

    /// Whether the cover names `code`. The test oracles ask; the kernels
    /// never do, they probe the whole vector.
    #[cfg(test)]
    pub fn contains(&self, code: Token) -> bool {
        self.points.contains(&code) || self.ranges.iter().any(|range| range.contains(code))
    }
}
