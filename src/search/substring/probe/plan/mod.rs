// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Planning: choose the cut and its filter, the matcher, and the mask packing per scan.
//!
//! Every cut of the alignment graph gives a correct filter, so planning only affects
//! speed. A filter costs matcher work for every scanned code, which grows with its
//! number of codes, plus verification work for every hit, which grows with their
//! frequency. A minimum cut can only minimize an additive edge weight, so
//! `select_cover` samples cuts for the weight `frequency + λ · comparisons` over a
//! range of `λ`, and ranks the resulting filters with the full cost model in `cost`.
//!
//! The matcher is chosen once per query, in `select`. Mask packing is chosen per
//! scan, from the expected hit density.

mod cost;
mod select;

use super::ProbeCover;
use super::alignment::graph::{AlignmentGraph, Edge};
use super::alignment::mincut::MinCut;
use super::scan::{Isa, MatcherKind};
use crate::search::index::TokenFrequencyIndexView;
#[cfg(test)]
pub(super) use select::is_eligible;
pub(super) use select::{cover_cost, select_matcher, skip_empty_packing};

/// Returns the cheapest sampled cover, the matcher chosen for it, and its
/// frequency in the indexed stream.
///
/// The first cut uses a `λ` above the code count, which minimizes comparisons.
/// Then `λ = 0, 1, 4, 16, ...` up to the code count, stopping once a cut repeats
/// the first. This samples the trade-off; it does not guarantee the cheapest cover.
pub(super) fn select_cover(
    graph: &AlignmentGraph,
    frequencies: TokenFrequencyIndexView<'_>,
    isa: Isa,
) -> (ProbeCover, MatcherKind, u32) {
    let code_count = frequencies.total_frequency();
    let mut solver = MinCut::new(graph);
    let evaluate_cut = |cut: &[u32]| {
        let cover = ProbeCover::from_edge_cut(cut.iter().map(|&at| &graph.edges[at as usize]));
        let matcher = select_matcher(isa, &cover);
        let frequency = cover.frequency(frequencies);
        let cost = cover_cost(isa, &cover, matcher, frequency, code_count);
        (cover, matcher, frequency, cost)
    };

    // Evaluate the comparison-heavy end before sampling lower penalties.
    let high_penalty_cut = solver
        .solve(edge_weight(u64::from(code_count) + 1))
        .to_vec();
    let (mut best_cover, mut best_matcher, mut best_frequency, mut best_cost) =
        evaluate_cut(&high_penalty_cut);

    let mut previous_cut: Vec<u32> = Vec::new();
    let mut lambda = 0u64;
    while lambda <= u64::from(code_count) {
        let cut = solver.solve(edge_weight(lambda));
        if cut == high_penalty_cut {
            break;
        }
        if cut != previous_cut {
            previous_cut = cut.to_vec();
            let (cover, matcher, frequency, cost) = evaluate_cut(&previous_cut);
            if cost < best_cost {
                best_cover = cover;
                best_matcher = matcher;
                best_frequency = frequency;
                best_cost = cost;
            }
        }
        lambda = (lambda * 4).max(1);
    }
    (best_cover, best_matcher, best_frequency)
}

/// Cut weight of an edge: its frequency plus `lambda` per comparison it adds.
/// Only a proxy for generating cuts; covers are ranked by `cover_cost`.
fn edge_weight(lambda: u64) -> impl Fn(&Edge) -> u64 {
    // Overflow bound for the cut solver. With n <= u16::MAX and F <= u32::MAX,
    // there are at most 2n + 16 edges, and their comparisons sum to at most
    // 3n + 15 * 512 + 65536: a point and a range per offset, up to 15 overlap
    // lists, and one contained-token list. For lambda <= F + 1, the total weight
    // stays below 2^51.
    move |edge| {
        let comparisons = edge.point_count() + 2 * edge.range_count();
        u64::from(edge.frequency()) + lambda * comparisons as u64
    }
}
