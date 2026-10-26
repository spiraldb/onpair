// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The alignment graph of a pattern, and the filter given by a cut of it.
//!
//! `graph` builds the graph, using `starts` to find the tokens an occurrence can
//! begin inside. `mincut` finds the cheapest cut for given edge weights, and
//! `cover` turns the cut edges into the filter, a `ProbeCover`.

pub(in crate::search::substring::probe) mod cover;
pub(in crate::search::substring::probe) mod graph;
pub(in crate::search::substring::probe) mod mincut;
pub(in crate::search::substring::probe) mod starts;
