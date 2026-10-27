// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Planning: choose the cut and its filter, the matcher, and the mask packing per scan.
//!
//! Every cut of the alignment graph gives a correct filter, so planning only affects
//! speed. A filter costs matcher work for every scanned code, which grows with its
//! number of codes, plus verification work for every hit, which grows with their
//! frequency. For a fixed matcher both terms add up over the cut's edges, so a
//! minimum cut finds that matcher's cheapest filter, up to how the filter merges
//! the edges' shared tokens and overlapping ranges. `select_cover` solves a few such
//! cuts and keeps the cheapest filter under the full cost model in `cost`.
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
use cost::Slopes;
#[cfg(test)]
pub(super) use select::is_eligible;
pub(super) use select::{cover_cost, select_matcher, skip_empty_packing};

/// Returns the cheapest cover, the matcher chosen for it, and its frequency in
/// the indexed stream.
///
/// The first cut minimizes frequency alone. That is the table's cheapest filter,
/// since its speed does not depend on the filter, and the only cut on scalar
/// targets, where only the table runs. Vector targets also try the cuts of
/// `CutWeights::priced`. Cuts often coincide, so a repeated cut is not evaluated
/// again: building a cover can cost more than solving for it, as when one edge
/// lists every token containing a byte.
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

    let first = solver.solve(|edge| u64::from(edge.frequency())).to_vec();
    let mut best = evaluate_cut(&first);
    let mut seen = vec![first];
    if let Some(slopes) = cost::slopes(isa) {
        for weights in CutWeights::priced(slopes, code_count) {
            let cut = solver.solve(|edge| weights.of(edge));
            if seen.iter().any(|earlier| earlier == cut) {
                continue;
            }
            let priced = evaluate_cut(cut);
            seen.push(cut.to_vec());
            if priced.3 < best.3 {
                best = priced;
            }
        }
    }
    let (cover, matcher, frequency, _) = best;
    (cover, matcher, frequency)
}

/// Additive cut weights: an edge weighs `hit` per hit, plus `point` and `range`
/// per point and range it adds to the filter.
#[derive(Clone, Copy, Debug)]
struct CutWeights {
    hit: f64,
    point: f64,
    range: f64,
}

impl CutWeights {
    /// The cuts tried after the frequency cut, over `code_count` scanned codes:
    /// - the fewest probes, ties broken by frequency: a probe outweighs any one
    ///   edge's hits. Merging can make this filter cheaper than the priced ones;
    /// - EqOr's cost per point and per range;
    /// - Nibble's, with each batch's cost spread over its eight points. This only
    ///   approximates Nibble's steps per batch and its limit on batches.
    fn priced(slopes: Slopes, code_count: u32) -> [Self; 3] {
        let codes = f64::from(code_count);
        let cut = |hit, scale: f64, point: f64| Self {
            hit,
            point: scale * point,
            range: scale * slopes.range,
        };
        [
            cut(1.0, codes + 1.0, slopes.eq_or_point),
            cut(slopes.candidate, codes, slopes.eq_or_point),
            cut(slopes.candidate, codes, slopes.nibble_point),
        ]
    }

    /// `edge`'s weight, rounded down.
    fn of(self, edge: &Edge) -> u64 {
        let hits = self.hit * f64::from(edge.frequency());
        let scan = self.point * edge.point_count() as f64 + self.range * edge.range_count() as f64;
        (hits + scan) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{MAX_TOKEN_SIZE, Token};
    use crate::search::substring::ContainsScan;
    use crate::search::substring::probe::alignment::starts::MAX_ENUMERATED_TOKENS;

    /// The largest alignment graph at the largest code count stays within the cut
    /// solver's i64 for every priced cut. Computed in u128, so a future limit
    /// increase fails here instead of overflowing.
    #[test]
    fn cut_weights_fit_the_solver() {
        let n = ContainsScan::MAX_PATTERN_LEN as u128;
        let codes = u128::from(u32::MAX);
        // At most 2n + 16 edges, each with frequency at most the code count.
        let edges = 2 * n + MAX_TOKEN_SIZE as u128;
        // A point per offset, up to 15 overlap lists and one contained-token list;
        // a range per offset.
        let points = n
            + (MAX_TOKEN_SIZE - 1) as u128 * MAX_ENUMERATED_TOKENS as u128
            + u128::from(Token::MAX)
            + 1;
        let ranges = n;
        for isa in [Isa::Neon, Isa::Avx2, Isa::Avx512Bw] {
            let slopes = cost::slopes(isa).unwrap();
            for weights in CutWeights::priced(slopes, u32::MAX) {
                let total = weights.hit.ceil() as u128 * edges * codes
                    + weights.point.ceil() as u128 * points
                    + weights.range.ceil() as u128 * ranges;
                assert!(total < i64::MAX as u128, "{isa:?} {weights:?}: {total}");
            }
        }
    }
}
