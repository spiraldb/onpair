// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Substring search over compressed rows, without decoding them.
//!
//! There are two implementations:
//!
//! * [`ContainsScan`] (`probe`) finds every row of a column that contains a pattern.
//!   It scans the code stream for the codes of a cut of the pattern's alignment
//!   graph, with SIMD where available, and verifies each hit in place.
//! * [`ContainsDfa`] with [`row_contains`] (`dfa`) tests one row at a time with a
//!   token-level KMP automaton. It needs no frequency index and works with any
//!   dictionary view, but supports only patterns of up to 255 bytes.

mod dfa;
mod error;
mod probe;

pub use dfa::{ContainsDfa, row_contains};
pub use error::ContainsError;
pub use probe::{ContainsScan, ProbeCover};
