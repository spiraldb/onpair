// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pattern to probe cover: build the alignment DAG, cut it, price the cuts.
//!
//! A minimum cut is linear in its edge weights and the scan's cost is not: a
//! kernel takes eight tokens for the price of one, and a range costs the
//! bitmap nothing. So the cut runs over `frequency + λ · comparisons` for a
//! sweep of λ, which walks from the most selective cut to the narrowest, and
//! each distinct cut is priced by [`scan_ns`] as the cover it becomes. The
//! sweep sees the cuts on the lower hull of (frequency, comparisons); a flat
//! kernel can make one off the hull cheaper, which is accepted.
//!
//! Frequencies weight the cut but never alter its selected membership:
//! safety-valid stored weights may contain false zeroes, so pruning by
//! frequency would be unsound. Profitability remains a caller decision.

pub(super) mod cost;
pub(super) mod facts;
pub(super) mod select;

use self::cost::scan_ns;
use self::facts::{RegionFacts, TargetCaps};
use super::ProbeCover;
use super::alignment::graph::{AlignmentGraph, Edge};
use super::alignment::mincut::MinCut;
use crate::search::index::TokenFrequencyIndexView;

/// The cut whose cover [`scan_ns`] prices lowest, with what it covers and
/// costs.
///
/// λ trades a cut's frequency against its comparisons, and the cut it selects
/// is piecewise constant in λ: it changes only where two cuts swap places, and
/// in practice there are one or two such pieces. So the narrowest cut is priced
/// first — no λ can go past it, since comparisons only fall as λ rises — and the
/// ladder from zero stops as soon as it arrives there, rather than stepping to a
/// ceiling set by the stream length.
pub(super) fn cheapest_cover(
    graph: &AlignmentGraph,
    frequencies: TokenFrequencyIndexView<'_>,
    region: RegionFacts,
    caps: TargetCaps,
) -> (ProbeCover, u32, f64) {
    let ceiling = u64::from(frequencies.total_frequency());
    // With n <= u16::MAX and F <= u32::MAX, there are at most 2n + 16 edges.
    // Their comparison counts sum to at most 3n + 15*512 + 65536: one point
    // and range per offset, up to 15 entry sets, and one contained-token set.
    // For lambda <= F + 1, total finite capacity is therefore below 2^51.
    let by = |lambda: u64| {
        move |edge: &Edge| {
            let comparisons = edge.point_count() + 2 * edge.range_count();
            u64::from(edge.frequency()) + lambda * u64::from(comparisons)
        }
    };
    let mut solver = MinCut::new(graph);
    let price = |cut: &[u32]| {
        let edges: Vec<&Edge> = cut.iter().map(|&at| &graph.edges[at as usize]).collect();
        let cover = ProbeCover::from_edge_cut(&edges);
        let covered = cover_frequency(&cover, frequencies);
        let ns = scan_ns(caps, &cover, covered, region);
        (cover, covered, ns)
    };

    let narrowest = solver.solve(by(ceiling + 1)).to_vec();
    let mut best = price(&narrowest);

    let mut last: Vec<u32> = Vec::new();
    let mut lambda = 0u64;
    while lambda <= ceiling {
        let cut = solver.solve(by(lambda));
        if cut == narrowest {
            break;
        }
        if cut != last {
            last = cut.to_vec();
            let candidate = price(&last);
            if candidate.2 < best.2 {
                best = candidate;
            }
        }
        lambda = (lambda * 4).max(1);
    }
    best
}

/// Codes `cover` matches in the indexed stream. Points and ranges are
/// disjoint, so each covered code is counted once.
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
