// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Choose probe covers during preparation and matcher configurations before scanning.
//!
//! Minimum cuts use additive edge weights: frequency plus a penalty for the
//! number of comparisons. Each sampled cut is normalized and ranked by its
//! lowest eligible matcher score plus a fixed penalty per covered occurrence.
//! Execution chooses mask packing separately for the selected cover.
//!
//! `facts` describes the inputs. `select` chooses eligible matchers and mask packing
//! using the formulas in `cost`. Neither performs CPU detection or scanning.
//! Frequencies guide these choices but never remove tokens from a cover.

mod cost;
mod facts;
mod select;

use super::ProbeCover;
use super::alignment::graph::{AlignmentGraph, Edge};
use super::alignment::mincut::MinCut;
use crate::search::index::TokenFrequencyIndexView;
pub(super) use facts::{AnalysisFacts, CoverShape, Isa, RegionFacts, ScanFacts, TargetCaps};
#[cfg(test)]
pub(super) use select::supports_matcher;
pub(super) use select::{score_cover, select_matcher_config};

/// Selected probes and their indexed token occurrence count.
pub(super) struct SelectedCover {
    pub cover: ProbeCover,
    pub covered_frequency: u32,
}

/// Matcher families considered during selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MatcherKind {
    Table,
    EqOr,
    Range,
    NibbleN8K,
}

/// Vector matcher configuration; nibble matching also needs a batch count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum VectorMatcher {
    EqOr,
    Range,
    NibbleN8 { batches: usize },
}

/// Matcher implementation and mask-packing policy for one scan.
/// Dispatch prepares the concrete matcher from this configuration and the cover.
/// Vector variants may skip packing empty groups.
/// `Empty` requires no scan; `Table` uses scalar membership lookups.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MatcherConfig {
    Empty,
    Table,
    Neon { matcher: VectorMatcher, skip: bool },
    Avx2 { matcher: VectorMatcher, skip: bool },
    Avx512Bw { matcher: VectorMatcher, skip: bool },
}

/// Select the sampled cover with the lowest ranking score.
///
/// First evaluate a large comparison penalty, then try lambda = 0, 1, 4, ...
/// up to the indexed code count. Stop early when a cut matches the first one,
/// and avoid repricing consecutive identical cuts. This samples alternatives;
/// it does not enumerate every cut or guarantee the lowest possible score.
pub(super) fn select_cover(
    graph: &AlignmentGraph,
    frequencies: TokenFrequencyIndexView<'_>,
    region: RegionFacts,
    caps: TargetCaps,
) -> SelectedCover {
    let ceiling = u64::from(frequencies.total_frequency());
    let mut solver = MinCut::new(graph);
    let build_cover = |cut: &[u32]| {
        let cover = ProbeCover::from_edge_cut(cut.iter().map(|&at| &graph.edges[at as usize]));
        let covered_frequency = cover_frequency(&cover, frequencies);
        SelectedCover {
            cover,
            covered_frequency,
        }
    };
    let rank = |candidate: &SelectedCover| {
        score_cover(caps, &candidate.cover, candidate.covered_frequency, region)
    };

    // Evaluate the comparison-heavy end before sampling lower penalties.
    let high_penalty_cut = solver.solve(edge_weight(ceiling + 1)).to_vec();
    let mut best = build_cover(&high_penalty_cut);
    let mut best_score = rank(&best);

    let mut previous_cut: Vec<u32> = Vec::new();
    let mut lambda = 0u64;
    while lambda <= ceiling {
        let cut = solver.solve(edge_weight(lambda));
        if cut == high_penalty_cut {
            break;
        }
        if cut != previous_cut {
            previous_cut = cut.to_vec();
            let candidate = build_cover(&previous_cut);
            let candidate_score = rank(&candidate);
            if candidate_score < best_score {
                best = candidate;
                best_score = candidate_score;
            }
        }
        lambda = (lambda * 4).max(1);
    }
    best
}

/// Additive cut weight: indexed frequency plus a comparison-count penalty.
/// This proxy generates cuts; complete covers are scored after normalization.
fn edge_weight(lambda: u64) -> impl Fn(&Edge) -> u64 {
    // With n <= u16::MAX and F <= u32::MAX, there are at most 2n + 16 edges.
    // Their comparison counts sum to at most 3n + 15*512 + 65536: one point
    // and range per offset, up to 15 overlap sets, and one contained-token set.
    // For lambda <= F + 1, total finite capacity is therefore below 2^51.
    move |edge| {
        let comparisons = edge.point_count() + 2 * edge.range_count();
        u64::from(edge.frequency()) + lambda * u64::from(comparisons)
    }
}

/// Sum the indexed frequencies of the normalized cover's tokens.
/// Disjoint points and ranges ensure each token is counted once.
pub(super) fn cover_frequency(cover: &ProbeCover, frequencies: TokenFrequencyIndexView<'_>) -> u32 {
    let points: u32 = cover
        .points()
        .iter()
        .map(|&t| frequencies.frequency(t))
        .sum();
    let ranges: u32 = cover
        .ranges()
        .iter()
        .map(|&r| frequencies.range_frequency(r))
        .sum();
    points + ranges
}
