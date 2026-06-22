// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![cfg(target_endian = "little")]
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::int_plus_one,
    clippy::manual_slice_size_calculation,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::panic,
    clippy::unwrap_used
)]

//! OnPair: dictionary-based short-string compression for fast random access.
//!
//! Rust port of the algorithm described in
//! [arXiv:2508.02280](https://arxiv.org/abs/2508.02280). OnPair replaces
//! recurring substrings ("tokens") with fixed-width integer codes into a
//! dictionary; decoding is a gather-copy, so individual rows decode in
//! isolation at no extra cost.
//!
//! # Layout
//! A compressed [`Column`] is a [`CompactDictionary`] (token bytes + offsets), a
//! code stream (one [`Token`] per emitted token), and a row layer (offsets into
//! the code stream). Borrow it as a [`ColumnView`] — or build a view directly
//! from buffers deserialized from storage — and decode with [`decode_into`] over
//! the compact dictionary or a reusable [`WideDictionary`].
//!
//! # Examples
//! ```
//! use onpair::{Column, DEFAULT_CONFIG};
//!
//! // Compress an Arrow (bytes, offsets) value pair.
//! let bytes = b"catdogcat";
//! let offsets: [u32; 4] = [0, 3, 6, 9];
//! let col = Column::compress(bytes, &offsets, DEFAULT_CONFIG).unwrap();
//!
//! // Bulk decode, or random-access a single row.
//! assert_eq!(col.view().decompress(), b"catdogcat");
//! assert_eq!(col.view().decompress_row(1), b"dog");
//! ```
//!
//! The trained encoder is also available directly via [`Parser`], to reuse one
//! dictionary across several corpora.

mod column;
mod core;
mod decoding;
mod encoding;
pub mod search;

#[cfg(test)]
mod test_corpus;

pub use crate::column::{Column, ColumnView};
pub use crate::core::dictionary::{
    CompactDictionary, CompactDictionaryView, Dictionary, DictionaryView, WideDictionary,
    WideDictionaryView,
};
pub use crate::core::offset::Offset;
pub use crate::core::types::{MAX_TOKEN_SIZE, Token, TokenRange};
pub use crate::decoding::{decode_into, decode_to_vec, decoded_len};
pub use crate::encoding::config::{Bits, Config, DEFAULT_CONFIG, Error, Threshold};
pub use crate::encoding::parser::Parser;

/// Compress an Arrow `(bytes, offsets)` value pair end-to-end. Equivalent to
/// `Parser::train(..)?.parse(..)`, but validates the offsets once instead of in
/// both the train and parse steps. `offsets` has `n + 1` entries.
///
/// # Errors
/// [`Error::InvalidArg`] if `offsets` is empty or its last entry exceeds
/// `bytes.len()`.
pub fn compress<O: Offset>(bytes: &[u8], offsets: &[O], cfg: Config) -> Result<Column<O>, Error> {
    encoding::parser::validate_offsets(bytes, offsets)?;
    let parser = Parser::train_unchecked(bytes, offsets, cfg);
    Ok(parser.parse_unchecked(bytes, offsets))
}
