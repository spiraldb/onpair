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
//! One thing the cut is told nothing about either, and is corrected for
//! afterwards: whether the tokens it picked ever occur. A zero-weight probe is
//! free by its objective, so it is selected readily — and a dictionary offers
//! plenty, since all 256 single-byte tokens are present whether or not the byte
//! appears, and the trainer leaves behind pairs the final parse abandoned.
//! Probing for one is pure waste: no row holds a token the code stream never
//! uses, so covering it cannot admit a row, and dropping it cannot lose one.
//!
//! Correcting it afterwards, rather than as the cut is read, is what keeps the
//! correction from backfiring. The cut's probes overlap and abut, so an unused
//! one can sit between two used ones — removing it there would split one range
//! probe into two and *raise* the comparison count. The decision therefore
//! belongs to whoever sees the ids merged, one maximal run at a time, which is
//! [`ProbeCover::from_membership`] (see [`live_span`]).
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
use crate::core::dictionary::CompactDictionaryView;
use crate::core::types::{Token, TokenRange};

/// Compile a sound probe cover for `pattern` over `dict`, cheapest by term
/// frequency in the code stream `frequencies` was built from.
pub(super) fn plan(
    dict: CompactDictionaryView<'_>,
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
    // Probe only for what the code stream actually uses. Sound for the same
    // reason it is worth doing: a row is admitted iff it holds a covered code,
    // and no row holds a token that occurs nowhere, so such a token's membership
    // changes no row's verdict either way.
    ProbeCover::from_membership(members, |run| live_span(run, frequencies))
}

/// The part of `run` worth probing for: `run` with the ids `frequencies` counts
/// no occurrences of trimmed off either end, or `None` when that leaves nothing.
///
/// Trimming stops at the ends because that is where it pays. An entirely unused
/// run costs a comparison for something that can never fire, and a run trimmed
/// down to a single id is probed as a point rather than a range — one comparison
/// instead of two and an AND. An unused id *between* two used ones is a different
/// matter: dropping it would split the run and add a comparison to every vector
/// of the scan, so it is left in, where a range carries it for free.
fn live_span(run: TokenRange, frequencies: &TokenFrequencyIndex) -> Option<TokenRange> {
    let unused = |id: usize| frequencies.frequency(id as Token) == 0;
    let mut begin = run.begin as usize;
    let mut last = run.last as usize;

    while begin <= last && unused(begin) {
        begin += 1;
    }
    if begin > last {
        return None;
    }
    // `begin` is in use, so this stops at or before it.
    while unused(last) {
        last -= 1;
    }

    Some(TokenRange {
        begin: begin as Token,
        last: last as Token,
    })
}
