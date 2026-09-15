// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Substring query preparation, planning, scanning and verification.

mod alignment;
mod plan;
mod query;
mod scan;
mod verify;

#[cfg(test)]
mod tests;

pub use alignment::cover::ProbeCover;
pub use query::{
    MAX_PATTERN_LEN, PrefilterAnalysis, analyze_prefilter, prefilter_is_likely_profitable,
    prefilter_matches,
};
pub use verify::{ContainsTable, contains};
