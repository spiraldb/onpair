// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The filter of a cut, as a `ProbeCover`: the tokens a scan searches for.
//!
//! Cut edges contribute single tokens, ranges and lists. Normalization merges them
//! into sorted, disjoint points and inclusive ranges, without changing the covered
//! set.

use super::graph::{Edge, EdgeKind};
use crate::core::types::{Token, TokenRange};
use crate::search::index::TokenFrequencyIndexView;

/// The filter of a [`ContainsScan`](crate::search::ContainsScan): the tokens it
/// searches for, taken from a cut of the pattern's alignment graph.
///
/// Every row that contains the pattern contains at least one covered token, so rows
/// without one are skipped. Rows with a hit are only candidates: the scan verifies
/// each hit.
#[derive(Debug, Clone)]
pub struct ProbeCover {
    /// Individual token IDs.
    points: Vec<Token>,
    /// Inclusive ranges of token IDs.
    ranges: Vec<TokenRange>,
}

impl ProbeCover {
    /// Individual covered token IDs, in ascending order.
    pub fn points(&self) -> &[Token] {
        &self.points
    }

    /// Inclusive ranges of covered token IDs, in ascending order and disjoint from
    /// the points.
    pub fn ranges(&self) -> &[TokenRange] {
        &self.ranges
    }

    /// Number of individual point probes.
    pub fn n_points(&self) -> usize {
        self.points.len()
    }

    /// Number of range probes, however many token IDs each covers.
    pub fn n_ranges(&self) -> usize {
        self.ranges.len()
    }

    /// Whether the cover contains no token IDs.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty() && self.ranges.is_empty()
    }

    /// Occurrences of covered tokens in the indexed stream. Points and ranges are
    /// disjoint, so each occurrence counts once.
    pub(in crate::search::substring::probe) fn frequency(
        &self,
        frequencies: TokenFrequencyIndexView<'_>,
    ) -> u32 {
        let points: u32 = self
            .points()
            .iter()
            .map(|&t| frequencies.frequency(t))
            .sum();
        let ranges: u32 = self
            .ranges()
            .iter()
            .map(|&r| frequencies.range_frequency(r))
            .sum();
        points + ranges
    }

    /// Normalizes inclusive ranges, given in any order and possibly overlapping.
    /// Overlapping and adjacent ranges merge, so `[10, 20]` and `[15, 30]` become
    /// `[10, 30]`; single-ID ranges become points.
    pub(in crate::search::substring::probe) fn from_runs(mut runs: Vec<TokenRange>) -> Self {
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

    /// Builds the cover of a cut. The cover is only sound if the cut crosses every
    /// source-to-sink path, as the cut solver guarantees.
    pub(in crate::search::substring::probe) fn from_edge_cut<'a>(
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
                // Never cut: there are no listed tokens to probe.
                EdgeKind::UnenumeratedSet => {}
            }
        }
        Self::from_runs(runs)
    }

    /// Build a test cover without normalization. Ranges must be disjoint.
    #[cfg(test)]
    pub(in crate::search::substring::probe) fn new(
        points: Vec<Token>,
        ranges: Vec<TokenRange>,
    ) -> Self {
        Self { points, ranges }
    }

    /// Check membership of one token ID for test oracles.
    #[cfg(test)]
    pub fn contains(&self, code: Token) -> bool {
        self.points.contains(&code) || self.ranges.iter().any(|range| range.contains(code))
    }
}
