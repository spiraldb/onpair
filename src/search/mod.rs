// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Equality, prefix and substring search over compressed codes, without decoding rows.
//!
//! Prepare queries once and reuse them across rows or scans:
//!
//! * Equality: [`tokenize`](tokenize()) the needle, then call [`equals`](equals()).
//! * Prefix: build a [`PrefixQuery`], then call [`starts_with`].
//! * Substring per row: build a [`ContainsDfa`], then call [`row_contains`].
//! * Substring across rows: build a [`ContainsScan`], then call [`ContainsScan::scan`]
//!   to append exact matching row indices. Preparation uses a reusable
//!   [`index::TokenFrequencyIndex`].
//!
//! Search requires a dictionary with unique tokens in bytewise-lexicographic order
//! and all 256 single-byte tokens. Training or full dictionary validation establishes
//! these properties; [`crate::DictionaryView`] alone guarantees only structural safety.
//!
//! Equality, prefix search and [`ContainsScan`] also require rows greedily tokenized
//! with the same dictionary, as produced by the encoder. Validating external buffers
//! does not establish greedy tokenization.

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
