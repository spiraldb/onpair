// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Substring search and its optional candidate prefilter.
//!
//! Exact matching is available in two forms: [`contains()`] stays in the
//! compressed domain and steps a prepared token-level KMP table, while
//! [`BytesVerifier`] decodes selected rows and searches their contiguous bytes.
//! The [`prefilter_candidates`] path can narrow either verifier to a sound
//! candidate superset before that exact check.

mod contains;
mod prefilter;
mod verify;

pub use contains::{ContainsTable, contains};
pub use prefilter::{
    PrefilterAnalysis, PrefilterError, ProbeCover, analyze_prefilter, prefilter_candidates,
    prefilter_is_likely_profitable,
};
pub use verify::BytesVerifier;
