// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Substring query preparation, planning, scanning and verification.

mod alignment;
mod error;
mod plan;
mod query;
mod scan;
mod verify;

#[cfg(test)]
mod tests;

pub use alignment::cover::ProbeCover;
pub use error::ContainsError;
pub use query::ContainsScan;
pub use verify::{ContainsDfa, row_contains};
