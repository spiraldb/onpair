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
//! from buffers deserialized from storage — and decode into a caller-owned buffer:
//! the whole column with [`ColumnView::decompress_into`], one row with
//! [`ColumnView::decompress_row_into`], or [`decompress_into`] over a reusable
//! [`WideDictionary`]. The caller owns buffer sizing: OnPair does not store the
//! original lengths, so keep them with the column. A buffer of the original length
//! is enough, and for a row [`MAX_TOKEN_SIZE`] bytes per code is the fastest.
//!
//! # Search
//! [`search`] answers equality, prefix and substring queries on the codes, without
//! decoding rows. On a column view, [`ColumnView::rows_equal_to`],
//! [`ColumnView::rows_starting_with`], [`ColumnView::rows_containing`] and
//! [`ColumnView::rows_containing_with_scan`] return the rows that match.
//!
//! # Examples
//! ```
//! use onpair::{Column, DEFAULT_CONFIG};
//!
//! // Compress an Arrow (bytes, offsets) value pair.
//! let bytes = b"catdogcat";
//! let offsets: [u32; 4] = [0, 3, 6, 9];
//! let col = Column::compress(bytes, &offsets, DEFAULT_CONFIG).unwrap();
//! let view = col.view();
//!
//! // Decompression restores the original values, so their lengths size the buffers.
//! // OnPair does not store these lengths: keep them with the column.
//! let column_len = bytes.len();
//! let row_len = (offsets[2] - offsets[1]) as usize;
//!
//! // Random-access one row, decoded straight from the compact dictionary.
//! let mut row = Vec::with_capacity(row_len);
//! let decompressed = view.decompress_row_into(1, row.spare_capacity_mut()).unwrap();
//! assert_eq!(decompressed, b"dog");
//!
//! // Decompress the whole column.
//! let mut all = Vec::with_capacity(column_len);
//! let decompressed = view.decompress_into(all.spare_capacity_mut()).unwrap();
//! assert_eq!(decompressed, b"catdogcat");
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
    CompactDictionary, CompactDictionaryView, Dictionary, DictionaryStorage, DictionaryView,
    OwnedDictionaryStorage, WideDictionary, WideDictionaryView, code_bits_for_num_tokens,
};
pub use crate::core::offset::Offset;
pub use crate::core::types::{MAX_TOKEN_SIZE, Token, TokenRange};
pub use crate::core::validate::InvalidColumn;
pub use crate::decoding::{
    DECOMPRESS_PADDING, OutputTooSmall, decompress_into, decompress_into_unchecked,
};
pub use crate::encoding::config::{Config, DEFAULT_CONFIG, Error, MaxDictBits, Threshold};
pub use crate::encoding::parser::Parser;
pub use crate::encoding::rows::Rows;

/// Compress an Arrow `(bytes, offsets)` value pair end-to-end. Equivalent to
/// `Parser::train(..)?.parse(..)`, but validates the offsets once instead of in
/// both the train and parse steps. Rows are encoded greedily, as [`Parser`] does.
/// `offsets` has `n + 1` entries. Its first entry may be non-zero; bytes outside
/// `offsets[0]..offsets[n]` are ignored.
///
/// # Errors
/// [`Error::InvalidArg`] if `offsets` is empty or its last entry exceeds
/// `bytes.len()`.
pub fn compress<O: Offset>(bytes: &[u8], offsets: &[O], cfg: Config) -> Result<Column<O>, Error> {
    encoding::parser::validate_offsets(bytes, offsets)?;
    let parser = Parser::train_unchecked(bytes, offsets, cfg);
    Ok(parser.parse_unchecked(bytes, offsets))
}

/// Compress a [`Rows`] input without first copying it into a contiguous buffer.
/// Rows are encoded greedily, as [`Parser`] does.
pub fn compress_rows<R: Rows + ?Sized, O: Offset>(rows: &R, cfg: Config) -> Column<O> {
    Parser::train_rows(rows, cfg).parse_rows(rows)
}
