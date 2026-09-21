// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Token IDs selected as probes for substring scanning.
//!
//! Selected graph edges contribute single IDs, ranges, or explicit sets.
//! Normalization combines them into sorted points and inclusive ranges,
//! merging overlaps and adjacent IDs without changing the covered tokens.

use super::graph::{Edge, EdgeKind};
use crate::core::types::{Token, TokenRange};

/// Token IDs that the scanner searches for.
///
/// The planner chooses a cover that intersects every source-to-sink path
/// in the alignment graph. Every matching row must therefore contain a
/// covered token, but a probe hit still needs verification.
#[derive(Debug, Clone)]
pub struct ProbeCover {
    /// Individual token IDs.
    pub(in crate::search::substring) points: Vec<Token>,
    /// Inclusive ranges of token IDs.
    pub(in crate::search::substring) ranges: Vec<TokenRange>,
}

impl ProbeCover {
    /// Token IDs tested for equality.
    pub fn points(&self) -> &[Token] {
        &self.points
    }

    /// Inclusive ranges of token IDs.
    pub fn ranges(&self) -> &[TokenRange] {
        &self.ranges
    }

    /// Number of individual point probes.
    pub fn n_points(&self) -> usize {
        self.points.len()
    }

    /// Number of range probes, regardless of how many token IDs each covers.
    pub fn n_ranges(&self) -> usize {
        self.ranges.len()
    }

    /// Whether the cover contains no token IDs.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty() && self.ranges.is_empty()
    }

    /// Normalize inclusive ranges into points and ranges.
    ///
    /// Input may be unsorted and contain duplicates. Overlapping and adjacent
    /// ranges are merged; ranges containing one ID become points.
    /// For example, `[10, 20]` and `[15, 30]` become `[10, 30]`.
    pub(in crate::search::substring) fn from_runs(mut runs: Vec<TokenRange>) -> Self {
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

    /// Build a normalized cover from selected graph edges.
    ///
    /// Requires a cut that intersects every source-to-sink path and contains
    /// no unenumerated sets. The cut solver establishes these conditions.
    pub(in crate::search::substring) fn from_edge_cut<'a>(
        cut: impl ExactSizeIterator<Item = &'a Edge>,
    ) -> Self {
        let point = |id: Token| TokenRange {
            begin: id,
            last: id,
        };
        let mut runs = Vec::with_capacity(cut.len());
        for edge in cut {
            match edge.kind() {
                EdgeKind::Single(id) => runs.push(point(*id)),
                EdgeKind::Range(range) => runs.push(*range),
                EdgeKind::Set(ids) => runs.extend(ids.iter().map(|&id| point(id))),
                // Unenumerated sets have no IDs to collect and cannot be cut.
                EdgeKind::UnenumeratedSet => {}
            }
        }
        Self::from_runs(runs)
    }

    /// Build a test cover without normalization. Ranges must be disjoint.
    #[cfg(test)]
    pub(in crate::search::substring) fn new(points: Vec<Token>, ranges: Vec<TokenRange>) -> Self {
        Self { points, ranges }
    }

    /// Check membership of one token ID for test oracles.
    #[cfg(test)]
    pub fn contains(&self, code: Token) -> bool {
        self.points.contains(&code) || self.ranges.iter().any(|range| range.contains(code))
    }
}
