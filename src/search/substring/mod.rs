// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Substring search over compressed rows, without decoding them.
//!
//! An occurrence of the pattern can begin and end inside tokens, so the codes that
//! encode it depend on the bytes around it. `probe` handles this for a whole code
//! stream: [`ContainsScan`] searches the codes for the tokens of a [`ProbeCover`],
//! which every occurrence is encoded with, and verifies each hit on the codes
//! around it. `dfa` handles it one row at a time: [`ContainsDfa`] lifts the
//! pattern's KMP automaton from bytes to tokens. The public [`search`](crate::search)
//! docs say when to use which. Both match the empty pattern against every row, as
//! [`str::contains`] does.

mod dfa;
mod error;
mod probe;

pub use dfa::{ContainsDfa, row_contains};
pub use error::ContainsError;
pub use probe::{ContainsScan, ProbeCover};

use std::panic::{RefUnwindSafe, UnwindSafe};

// Both document that one prepared query can be shared between threads. Both also
// stay usable after a caught panic, as their state never changes once prepared.
const _: () = {
    const fn shareable<T: Send + Sync + UnwindSafe + RefUnwindSafe>() {}
    shareable::<ContainsDfa>();
    shareable::<ContainsScan<'static>>();
};
