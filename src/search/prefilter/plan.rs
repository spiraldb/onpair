// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pattern to probe cover: build the alignment DAG, cut it, scan for the cut.
//!
//! A minimum cut chooses across every pattern alignment at once. Frequencies
//! weight the cut but never alter its selected membership: safety-valid stored
//! weights may contain false zeroes, so pruning by frequency would be unsound.
//! Profitability remains a caller decision.

use super::cover::ProbeCover;
use super::graph::build_alignment_graph;
use super::mincut::min_cut;
use crate::core::dictionary::CompactDictionaryView;
use crate::search::index::TokenFrequencyIndexView;

/// Compile a sound probe cover for `pattern` over `dict`, cheapest by term
/// frequency according to the advisory `frequencies` weights.
pub(super) fn plan(
    dict: CompactDictionaryView<'_>,
    pattern: &[u8],
    frequencies: TokenFrequencyIndexView<'_>,
) -> ProbeCover {
    let graph = build_alignment_graph(dict, pattern, frequencies);
    let cut = min_cut(&graph.edges, graph.nodes);
    ProbeCover::from_membership(graph.membership(&cut))
}
