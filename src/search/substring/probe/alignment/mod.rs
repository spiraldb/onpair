// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The alignment graph of a pattern, and the filter given by a cut of it.
//!
//! `graph` builds the graph, using `starts` to find the tokens an occurrence can
//! begin inside. `mincut` finds the cheapest cut for given edge weights, and
//! `cover` turns the cut edges into the filter, a `ProbeCover`.

pub(super) mod cover;
pub(super) mod graph;
pub(super) mod mincut;
pub(super) mod starts;
