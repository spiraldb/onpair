// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Substring search and its prefilter.
//!
//! Exact matching is available in two forms: [`contains()`] stays in the
//! compressed domain and steps a prepared token-level KMP table, while
//! [`BytesVerifier`] decodes selected rows and searches their contiguous bytes.
//! [`prefilter_candidates`] scans the code stream for the rows holding a probe
//! and verifies each hit against the alignment graph, so it answers the query
//! on its own for a selective pattern.

mod contains;
mod prefilter;
mod verify;

pub use contains::{ContainsTable, contains};
pub use prefilter::{
    MAX_PATTERN_LEN, PrefilterAnalysis, ProbeCover, analyze_prefilter, prefilter_candidates,
    prefilter_is_likely_profitable,
};
pub use verify::BytesVerifier;
