// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Substring search over compressed rows, without decoding them.
//!
//! An occurrence of the pattern can begin and end inside tokens, so the codes that
//! encode it depend on the bytes around it. Two implementations handle this:
//!
//! * [`ContainsScan`] (`probe`) finds every row of a code stream that contains the
//!   pattern. It searches the codes for the tokens of a [`ProbeCover`], chosen so
//!   that every occurrence is encoded with at least one of them, with SIMD where
//!   available, and verifies each hit on the codes around it. It requires greedily
//!   tokenized rows, uses a token frequency index to prefer rare tokens, and accepts
//!   patterns of up to 65535 bytes.
//! * [`ContainsDfa`] (`dfa`) lifts the pattern's KMP automaton from bytes to tokens,
//!   and [`row_contains`] runs it over one row at a time, one step per code. It is
//!   correct for any tokenization and needs only a sorted dictionary and no
//!   frequency index, but accepts patterns of at most 255 bytes.
//!
//! Both are exact, and both match the empty pattern against every row, as
//! [`str::contains`] does.

mod dfa;
mod error;
mod probe;

pub use dfa::{ContainsDfa, row_contains};
pub use error::ContainsError;
pub use probe::{ContainsScan, ProbeCover};

// Both document that one prepared query can be shared between threads.
const _: () = {
    const fn send_sync<T: Send + Sync>() {}
    send_sync::<ContainsDfa>();
    send_sync::<ContainsScan<'static>>();
};
