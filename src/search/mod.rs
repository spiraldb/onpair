// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Equality, prefix and substring search over compressed codes, without decoding rows.
//!
//! Prepare queries once and reuse them across rows or scans:
//!
//! * Equality: [`tokenize`](tokenize()) the needle, then call [`equals`](equals()).
//! * Prefix: build a [`PrefixQuery`], then call [`starts_with`].
//! * Substring per row: build a [`ContainsDfa`], then call [`row_contains`]. It
//!   takes one step per code, is exact for rows of any tokenization, needs no
//!   frequency index, and accepts patterns of up to 255 bytes.
//! * Substring across rows: build a [`ContainsScan`], then call [`ContainsScan::scan`]
//!   to append the indices of the matching rows. It searches the whole code stream
//!   for a few rare tokens, with SIMD where available, and is usually faster. It
//!   plans with a reusable [`index::TokenFrequencyIndex`], requires greedily
//!   tokenized rows, and accepts patterns of up to 65535 bytes.
//!
//! Search requires a dictionary with unique tokens in bytewise-lexicographic order
//! and all 256 single-byte tokens. Training or full dictionary validation establishes
//! these properties; [`crate::DictionaryView`] alone guarantees only structural safety.
//!
//! Equality, prefix search and [`ContainsScan`] also require rows greedily tokenized
//! with the same dictionary, as [`Parser`](crate::Parser) encodes them. Validating
//! external buffers does not establish greedy tokenization.

mod equals;
pub mod index;
mod lookup;
mod prefix;
mod substring;
mod tokenize;

pub use equals::equals;
pub use lookup::prefix_range;
pub use prefix::{PrefixQuery, starts_with};
pub use substring::{ContainsDfa, ContainsError, ContainsScan, ProbeCover, row_contains};
pub use tokenize::tokenize;
