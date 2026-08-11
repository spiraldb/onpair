// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pattern to probe cover: build the alignment DAG, cut it, scan for the cut.
//!
//! One decision, taken over every layout of the pattern at once. The alternative
//! — walking each alignment and picking the least frequent sound probe for each
//! of its accepting terminals — is cheaper to compute and reliably worse to run.
//! It pays twice for alignments that converge on a shared suffix, when one probe
//! at the join would block both; and it weighs a first-token set against a
//! single terminal, when choosing that set would have obviated all of them. A
//! cut of the merged graph has neither blind spot, because it is not making a
//! sequence of local choices at all.
//!
//! What the cut is deliberately not told about is the SIMD comparison budget:
//! its objective is frequency weight, and the budget counts probes. The two
//! mostly agree — merging runs pulls the count down, and a cheap cover is
//! usually a narrow one — but where they don't, the scan refuses with
//! [`ProbeCoverTooWide`](super::PrefilterError::ProbeCoverTooWide) rather than
//! have the plan pick a heavier cut to fit. A pattern whose *cheapest* cover is
//! that wide is one prefiltering has little to offer, and saying so is worth
//! more to the caller than scanning for it anyway.

use super::cover::ProbeCover;
use super::frequency::TokenFrequencyIndex;
use super::graph::build_alignment_graph;
use super::mincut::minimum_vertex_cut;
use crate::core::dictionary::DictionaryView;

/// Compile a sound probe cover for `pattern` over `dict`, cheapest by term
/// frequency in the code stream `frequencies` was built from.
pub(super) fn plan<V: DictionaryView>(
    dict: V,
    pattern: &[u8],
    frequencies: &TokenFrequencyIndex,
) -> ProbeCover {
    let graph = build_alignment_graph(dict, pattern, frequencies);
    let mut members = graph.membership(&minimum_vertex_cut(&graph));
    // Mandatory, and outside the graph by construction: a token containing the
    // whole pattern matches without crossing a boundary, so no path stands for
    // it and no cut could have selected it.
    for &id in &graph.contained {
        members[id as usize] = true;
    }
    ProbeCover::from_membership(members)
}
