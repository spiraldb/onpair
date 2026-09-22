// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Choose probe covers during preparation and matcher configurations before scanning.
//!
//! Minimum cuts use additive edge weights: frequency plus a penalty for the
//! number of comparisons. Each sampled cut is normalized and ranked by its
//! lowest eligible matcher cost plus a fixed penalty per covered occurrence.
//! NEON shares its integer matcher costs between both decisions and charges
//! 4096 per covered occurrence. Execution chooses mask packing separately.
//!
//! `scan` defines the instruction sets and matcher configurations. `select`
//! chooses eligible matchers using the formulas in `cost` and chooses mask
//! packing separately. Neither performs CPU detection or scanning.
//! Frequencies guide these choices but never remove tokens from a cover.

mod cost;
mod select;

use super::ProbeCover;
use super::alignment::graph::{AlignmentGraph, Edge};
use super::alignment::mincut::MinCut;
use super::scan::Isa;
use crate::search::index::TokenFrequencyIndexView;
#[cfg(test)]
pub(super) use select::is_eligible;
pub(super) use select::{cover_cost, probe_density, select_matcher_config};

/// Select the sampled cover with the lowest relative cost.
/// Return the cover and its indexed token occurrence count.
///
/// First evaluate a large comparison penalty, then try lambda = 0, 1, 4, ...
/// up to the indexed code count. Stop early when a cut matches the first one,
/// and avoid repricing consecutive identical cuts. This samples alternatives;
/// it does not enumerate every cut or guarantee the lowest possible cost.
pub(super) fn select_cover(
    graph: &AlignmentGraph,
    frequencies: TokenFrequencyIndexView<'_>,
    isa: Isa,
) -> (ProbeCover, u32) {
    let code_count = frequencies.total_frequency();
    let ceiling = u64::from(code_count);
    let mut solver = MinCut::new(graph);
    let evaluate_cut = |cut: &[u32]| {
        let cover = ProbeCover::from_edge_cut(cut.iter().map(|&at| &graph.edges[at as usize]));
        let frequency = cover.frequency(frequencies);
        let cost = cover_cost(isa, &cover, frequency, code_count);
        (cover, frequency, cost)
    };

    // Evaluate the comparison-heavy end before sampling lower penalties.
    let high_penalty_cut = solver.solve(edge_weight(ceiling + 1)).to_vec();
    let (mut best_cover, mut best_frequency, mut best_cost) = evaluate_cut(&high_penalty_cut);

    let mut previous_cut: Vec<u32> = Vec::new();
    let mut lambda = 0u64;
    while lambda <= ceiling {
        let cut = solver.solve(edge_weight(lambda));
        if cut == high_penalty_cut {
            break;
        }
        if cut != previous_cut {
            previous_cut = cut.to_vec();
            let (cover, frequency, cost) = evaluate_cut(&previous_cut);
            if cost < best_cost {
                best_cover = cover;
                best_frequency = frequency;
                best_cost = cost;
            }
        }
        lambda = (lambda * 4).max(1);
    }
    (best_cover, best_frequency)
}

/// Additive cut weight: indexed frequency plus a comparison-count penalty.
/// This proxy generates cuts; complete covers are evaluated after normalization.
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
