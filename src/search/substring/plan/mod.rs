// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Choose probe covers during preparation and matcher configurations before scanning.
//!
//! Minimum cuts use additive edge weights: frequency plus a penalty for the
//! number of comparisons. Each sampled cut is normalized and ranked by its
//! lowest eligible matcher score plus a fixed penalty per covered occurrence.
//! Execution chooses mask packing separately for the selected cover.
//!
//! The types below describe planning inputs and results. `select` chooses eligible
//! matchers using the formulas in `score` and chooses mask packing separately.
//! Neither performs CPU detection or scanning.
//! Frequencies guide these choices but never remove tokens from a cover.

mod score;
mod select;

use super::ProbeCover;
use super::alignment::graph::{AlignmentGraph, Edge};
use super::alignment::mincut::MinCut;
use crate::search::index::TokenFrequencyIndexView;
#[cfg(test)]
pub(super) use select::supports_matcher;
pub(super) use select::{probe_density, score_cover, select_matcher_config};

/// Instruction-set family used for kernel selection and ranking weights.
/// Production callers use the family returned by `scan::detect_isa`;
/// planning tests can supply any family without executing its kernels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)] // Some variants are only constructed on other build targets.
pub(super) enum Isa {
    Scalar,
    Neon,
    Avx2,
    Avx512Bw,
}

/// Algorithm used to test token membership in the probe cover.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MatcherKind {
    Table,
    EqOr,
    Range,
    NibbleN8,
}

/// Selected algorithm and its mask-packing policy.
/// Dispatch derives nibble batches from the cover and specializes packing once,
/// before scanning. The instruction set determines eligibility during selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct MatcherConfig {
    pub(super) kind: MatcherKind,
    /// Skip packing empty vector groups; always false for the scalar table.
    pub(super) skip_empty_packing: bool,
}

/// Select the sampled cover with the lowest ranking score.
/// Return the cover and its indexed token occurrence count.
///
/// First evaluate a large comparison penalty, then try lambda = 0, 1, 4, ...
/// up to the indexed code count. Stop early when a cut matches the first one,
/// and avoid repricing consecutive identical cuts. This samples alternatives;
/// it does not enumerate every cut or guarantee the lowest possible score.
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
        let score = score_cover(isa, &cover, frequency, code_count);
        (cover, frequency, score)
    };

    // Evaluate the comparison-heavy end before sampling lower penalties.
    let high_penalty_cut = solver.solve(edge_weight(ceiling + 1)).to_vec();
    let (mut best_cover, mut best_frequency, mut best_score) = evaluate_cut(&high_penalty_cut);

    let mut previous_cut: Vec<u32> = Vec::new();
    let mut lambda = 0u64;
    while lambda <= ceiling {
        let cut = solver.solve(edge_weight(lambda));
        if cut == high_penalty_cut {
            break;
        }
        if cut != previous_cut {
            previous_cut = cut.to_vec();
            let (cover, frequency, score) = evaluate_cut(&previous_cut);
            if score < best_score {
                best_cover = cover;
                best_frequency = frequency;
                best_score = score;
            }
        }
        lambda = (lambda * 4).max(1);
    }
    (best_cover, best_frequency)
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
