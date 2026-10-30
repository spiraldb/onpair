// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The alignment graph of a pattern, and its minimum cut.
//!
//! `graph` builds the graph, using `starts` to find the tokens an occurrence can
//! begin inside. `mincut` finds the cheapest cut for given edge weights, whose
//! edges become the cover, a `ProbeCover`.

pub(super) mod graph;
pub(super) mod mincut;
pub(super) mod starts;
