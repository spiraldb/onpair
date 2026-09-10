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

use super::cover::ProbeCover;
use super::graph::{AlignmentGraph, Edge, build_alignment_graph};
use super::mincut::MinCut;
use super::scan::Walk;
use super::scan::{Region, scan_ns};
use crate::core::dictionary::CompactDictionaryView;
use crate::search::index::TokenFrequencyIndexView;

/// What [`plan`] settles on for one pattern.
pub(super) struct Planned {
    pub(super) cover: ProbeCover,
    /// Codes the cover matches in the indexed stream.
    pub(super) covered: u32,
    /// [`scan_ns`] of the cover.
    pub(super) scan_ns: f64,
    /// The exact check a hit on the cover admits.
    pub(super) walk: Walk,
}

/// Compile a sound probe cover for `pattern` over `dict`, cheapest by the
/// scan cost model over a stream of `row_count` rows.
pub(super) fn plan(
    dict: CompactDictionaryView<'_>,
    pattern: &[u8],
    frequencies: TokenFrequencyIndexView<'_>,
    row_count: usize,
) -> Planned {
    let graph = build_alignment_graph(dict, pattern, frequencies);
    let region = Region {
        code_count: frequencies.total_frequency() as usize,
        row_count,
    };
    let (cover, covered, scan_ns) = cheapest_cover(&graph, frequencies, region);

    Planned {
        cover,
        covered,
        scan_ns,
        walk: Walk::from_graph(&graph, pattern),
    }
}

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
    region: Region,
) -> (ProbeCover, u32, f64) {
    let ceiling = u64::from(frequencies.total_frequency());
    let by = |lambda: u64| {
        move |edge: &Edge| {
            let (points, ranges) = edge.shape();
            u64::from(edge.frequency()) + lambda * u64::from(points + 2 * ranges)
        }
    };
    let mut solver = MinCut::new(&graph.edges, graph.nodes);
    let mut best: Option<(ProbeCover, u32, f64)> = None;
    let price = |cut: &[u32], best: &mut Option<(ProbeCover, u32, f64)>| {
        let edges: Vec<&Edge> = cut.iter().map(|&at| &graph.edges[at as usize]).collect();
        let cover = ProbeCover::from_edge_cut(&edges);
        let covered = cover_frequency(&cover, frequencies);
        let ns = scan_ns(&cover, covered, region);
        if best.as_ref().is_none_or(|(_, _, best_ns)| ns < *best_ns) {
            *best = Some((cover, covered, ns));
        }
    };

    let narrowest = solver.solve(&graph.edges, by(ceiling + 1)).to_vec();
    price(&narrowest, &mut best);

    let mut last: Vec<u32> = Vec::new();
    let mut lambda = 0u64;
    while lambda <= ceiling {
        let cut = solver.solve(&graph.edges, by(lambda));
        if cut == narrowest {
            break;
        }
        if cut != last {
            last = cut.to_vec();
            price(&last, &mut best);
        }
        lambda = (lambda * 4).max(1);
    }
    best.expect("the sweep prices at least the narrowest cut")
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
