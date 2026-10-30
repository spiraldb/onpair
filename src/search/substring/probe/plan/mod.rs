// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Planning, which chooses the cut and its cover, the matcher, and the mask packing.
//!
//! Every cut of the alignment graph gives a correct cover, so planning only affects
//! speed. A cover costs matcher work for every scanned code, which grows with its
//! number of codes, plus verification work for every hit, which grows with their
//! frequency. For EqOr both terms add up over the cut's edges, for Nibble
//! approximately, and for the table only the hits vary, so a minimum cut finds
//! that matcher's cheapest cover, up to how the cover merges the edges' shared
//! tokens and overlapping ranges. `select` solves a few such cuts
//! and keeps the cheapest cover, with its matcher, under the cost model in `cost`.
//!
//! Mask packing is chosen per scan, from the expected hit density.

mod cost;

use super::ProbeCover;
use super::alignment::graph::{AlignmentGraph, Edge, EdgeKind};
use super::alignment::mincut::MinCut;
use super::scan::{Isa, MatcherKind, PER_BATCH};
use crate::search::index::TokenFrequencyIndexView;
pub(super) use cost::{CostModel, max_nibble_batches};

/// Hit density below which vector matchers test each pair of 64-code groups for a
/// hit before packing it: 0.025%, about one expected hit per 4096-code block. Clustered
/// hits can make skipping pay off above it too.
const SKIP_PACKING_DENSITY_THRESHOLD: f64 = 2.5e-4;

/// A cover and the matcher chosen for it.
pub(super) struct Plan {
    pub(super) cover: ProbeCover,
    pub(super) matcher: MatcherKind,
    /// Occurrences of the cover's tokens in the indexed stream.
    pub(super) hits: u32,
}

/// The cheapest cover of `graph`, with its matcher.
///
/// The first cut minimizes frequency alone. That is the table's cheapest cover,
/// since its speed does not depend on the cover, and the only cut on scalar
/// targets, where only the table runs. Vector targets also try the cuts of
/// `CutWeights::priced`. Cuts often coincide, so a repeated cut is not evaluated
/// again: building a cover can cost more than solving for it, as when one edge
/// lists every token containing a byte.
pub(super) fn select(
    graph: &AlignmentGraph,
    frequencies: TokenFrequencyIndexView<'_>,
    isa: Isa,
) -> Plan {
    let codes = frequencies.total_frequency();
    let model = CostModel::of(isa);
    let edges: Vec<_> = graph.edges().collect();
    // Counted once per edge, since an edge can list thousands of tokens.
    let edge_hits: Vec<_> = edges
        .iter()
        .map(|edge| edge_frequency(edge, frequencies))
        .collect();
    let mut solver = MinCut::new(&edges, graph.sink());
    let evaluate = |cut: &[u32]| {
        let cover = ProbeCover::from_edge_cut(cut.iter().map(|&at| edges[at as usize]));
        let hits = cover.frequency(frequencies);
        let (matcher, cost) = match model {
            Some(model) => model.cheapest(&cover, hits, codes),
            None => (MatcherKind::Table, 0.0),
        };
        (
            Plan {
                cover,
                matcher,
                hits,
            },
            cost,
        )
    };

    let first = solver.solve(|at| u64::from(edge_hits[at])).to_vec();
    let (mut best, mut best_cost) = evaluate(&first);
    let Some(model) = model else {
        return best;
    };
    let mut seen = vec![first];
    for weights in CutWeights::priced(model, codes) {
        let cut = solver.solve(|at| weights.of(&edges[at], edge_hits[at]));
        if seen.iter().any(|earlier| earlier == cut) {
            continue;
        }
        let (plan, cost) = evaluate(cut);
        seen.push(cut.to_vec());
        if cost < best_cost {
            (best, best_cost) = (plan, cost);
        }
    }
    best
}

/// Occurrences of `edge`'s tokens in the indexed stream. A set lists distinct
/// tokens, so the sum cannot exceed the stream length.
pub(super) fn edge_frequency(edge: &Edge<'_>, frequencies: TokenFrequencyIndexView<'_>) -> u32 {
    match edge.kind {
        EdgeKind::Single(token) => frequencies.frequency(token),
        EdgeKind::Range(range) => frequencies.range_frequency(range),
        EdgeKind::Set(tokens) => tokens
            .iter()
            .map(|&token| frequencies.frequency(token))
            .sum(),
        EdgeKind::UnenumeratedSet => 0,
    }
}

/// Whether `matcher` can run `cover` on `isa`, an instruction set this CPU supports.
#[cfg(test)]
pub(super) fn is_eligible(isa: Isa, matcher: MatcherKind, cover: &ProbeCover) -> bool {
    CostModel::of(isa).map_or(matcher == MatcherKind::Table, |model| {
        model.can_run(matcher, cover)
    })
}

/// Whether vector matchers should skip packing pairs of groups without hits in this
/// scan.
///
/// True when hits are sparse (`hit_density`, the covered fraction of the indexed
/// stream, is below the threshold) or when the scan's `code_count` codes are
/// expected to hold less than one hit. The table matcher ignores this policy.
pub(super) fn skip_empty_packing(hit_density: f64, code_count: usize) -> bool {
    hit_density < SKIP_PACKING_DENSITY_THRESHOLD || hit_density * (code_count as f64) < 1.0
}

/// Additive cut weights: an edge weighs `hit` per hit, plus `point` and `range`
/// per point and range it adds to the cover.
#[derive(Clone, Copy, Debug)]
struct CutWeights {
    hit: f64,
    point: f64,
    range: f64,
}

impl CutWeights {
    /// The cuts tried after the frequency cut, over `code_count` scanned codes:
    /// - the fewest probes, ties broken by frequency: a point outweighs any one
    ///   edge's hits, and a range weighs as many points as it costs in EqOr.
    ///   Merging can make this cover cheaper than the priced ones;
    /// - EqOr's cost per point and per range;
    /// - Nibble's, with each batch's cost spread over its eight points. This only
    ///   approximates Nibble's steps per batch and its limit on batches.
    fn priced(model: &CostModel, code_count: u32) -> [Self; 3] {
        let codes = f64::from(code_count);
        let probe = codes + 1.0;
        let cut = |point: f64| Self {
            hit: model.hit,
            point: codes * point,
            range: codes * model.range,
        };
        [
            Self {
                hit: 1.0,
                point: probe,
                range: probe * model.range / model.point,
            },
            cut(model.point),
            cut(model.nibble_batch / PER_BATCH as f64),
        ]
    }

    /// The weight of `edge`, whose tokens occur `frequency` times, rounded down.
    /// Points and ranges are counted before the cover merges them.
    fn of(self, edge: &Edge<'_>, frequency: u32) -> u64 {
        let (points, ranges) = match edge.kind {
            EdgeKind::Single(_) => (1, 0),
            EdgeKind::Range(_) => (0, 1),
            EdgeKind::Set(tokens) => (tokens.len(), 0),
            EdgeKind::UnenumeratedSet => (0, 0),
        };
        let hits = self.hit * f64::from(frequency);
        let scan = self.point * points as f64 + self.range * ranges as f64;
        (hits + scan) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{MAX_TOKEN_SIZE, Token, TokenRange};
    use crate::search::substring::ContainsScan;
    use crate::search::substring::probe::alignment::starts::MAX_ENUMERATED_TOKENS;

    /// Builds separated points and ranges with the requested probe counts.
    fn cover(points: usize, ranges: usize) -> ProbeCover {
        ProbeCover::new(
            (0..points).map(|i| (2 * i) as Token).collect(),
            (0..ranges)
                .map(|i| TokenRange::new((128 + 4 * i) as Token, (129 + 4 * i) as Token))
                .collect(),
        )
    }

    #[test]
    fn nibble_admission_respects_batch_caps() {
        for (isa, limit) in [
            (Isa::Scalar, 0),
            (Isa::Neon, 24),
            (Isa::Avx2, 32),
            (Isa::Avx512Bw, 32),
        ] {
            for points in 0..=40 {
                assert_eq!(
                    is_eligible(isa, MatcherKind::Nibble, &cover(points, 0)),
                    points > 0 && points <= limit
                );
            }
        }
    }

    #[test]
    fn sparse_probes_or_few_projected_hits_skip_packing() {
        for (density, code_count, skip) in [
            // The density threshold is strict.
            (0.00025_f64.next_down(), 1_000_000, true),
            (0.00025, 1_000_000, false),
            (0.00025_f64.next_up(), 1_000_000, false),
            // Partial scans use the same density but project fewer hits.
            (1.0 / 400.0, 399, true),
            (1.0 / 400.0, 400, false),
            (1.0 / 400.0, 401, false),
        ] {
            assert_eq!(
                skip_empty_packing(density, code_count),
                skip,
                "density {density}, {code_count} codes"
            );
        }
    }

    #[test]
    fn selected_matchers_are_eligible() {
        for isa in [Isa::Neon, Isa::Avx2, Isa::Avx512Bw] {
            let model = CostModel::of(isa).unwrap();
            for points in 0..=32 {
                for ranges in 0..=4 {
                    let cover = cover(points, ranges);
                    if !cover.is_empty() {
                        let (matcher, _) = model.cheapest(&cover, 0, 1);
                        assert!(model.can_run(matcher, &cover), "{isa:?} {matcher:?}");
                    }
                }
            }
        }
    }

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
            let model = CostModel::of(isa).unwrap();
            for weights in CutWeights::priced(model, u32::MAX) {
                let total = weights.hit.ceil() as u128 * edges * codes
                    + weights.point.ceil() as u128 * points
                    + weights.range.ceil() as u128 * ranges;
                assert!(total < i64::MAX as u128, "{isa:?} {weights:?}: {total}");
            }
        }
    }
}
