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
//! # Trying OnPair
//! [`Column`] compresses Arrow `(bytes, offsets)` values and owns the result, and its
//! [`view`](Column::view) decompresses and searches them.
//!
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
//! # Storing OnPair data
//! A storage engine keeps the parts of a column and calls the free functions on them.
//! [`Column`] and [`ColumnView`] are conveniences over the same calls.
//!
//! **What to store.** [`Column::into_raw`] gives the parts: a [`CompactDictionary`]
//! (token bytes and offsets), a code stream (one [`Token`] per token of the input),
//! and the row offsets into the code stream. The
//! [interchange format](https://github.com/spiraldb/onpair/blob/develop/docs/interchange-format.md)
//! specifies them.
//!
//! OnPair does not store two things an engine may want beside them: each row's
//! decompressed length, which sizes output buffers, and a
//! [`TokenFrequencyIndex`](search::index::TokenFrequencyIndex), which
//! [`ContainsScan`](search::ContainsScan) plans with.
//!
//! **Reading it back.** Buffers from storage can be corrupt.
//! [`CompactDictionaryView::validate_safety`] checks what memory safety needs, and
//! [`CompactDictionaryView::validate`] also checks what correct search needs. Without
//! the second, search can give wrong answers, but stays memory safe.
//!
//! **Decompressing.** Build a [`WideDictionary`] once for whole columns and keep it.
//! Single rows go straight to the compact view, because building the wide form costs
//! time proportional to the dictionary. The output is `&mut [MaybeUninit<u8>]`, so a
//! `Vec`'s [`spare_capacity_mut`](Vec::spare_capacity_mut) works without zeroing it.
//!
//! | You have | Call | Buffer |
//! |---|---|---|
//! | A whole column | [`decompress_into`] | Its decompressed length |
//! | A row | [`decompress_into`] | [`MAX_TOKEN_SIZE`] bytes per code |
//! | A row, and a length you trust | [`decompress_into_unchecked`] | That length plus [`DECOMPRESS_PADDING`] |
//!
//! The last row is the fastest for short rows, and offers nothing for whole columns. A
//! length read from storage is not one to trust: if it were corrupt, the unchecked call
//! would write out of bounds. Use the safe call and compare the length it returns.
//!
//! ```
//! use onpair::{Column, CompactDictionaryView, DEFAULT_CONFIG, Dictionary};
//!
//! let bytes = b"catdogcat";
//! let offsets: [u32; 4] = [0, 3, 6, 9];
//! let (dict, codes, _) = Column::compress(bytes, &offsets, DEFAULT_CONFIG)?.into_raw();
//!
//! // Validate the dictionary read back from storage, and build the wide form once.
//! let view = CompactDictionaryView::validate(dict.bytes(), dict.offsets())?;
//! let wide = view.to_wide();
//!
//! // Decompress into a buffer's spare capacity, sized from the stored length.
//! let stored_len = bytes.len();
//! let mut out = Vec::with_capacity(stored_len);
//! let n = onpair::decompress_into(&codes, wide.as_view(), out.spare_capacity_mut())?.len();
//! assert_eq!(n, stored_len);
//! // SAFETY: `decompress_into` initialized the first `n` bytes.
//! unsafe { out.set_len(n) };
//! assert_eq!(out, bytes);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Search
//! [`search`] answers equality, prefix and substring queries on the codes, without
//! decoding rows. On a column view, [`ColumnView::rows_equal_to`],
//! [`ColumnView::rows_starting_with`], [`ColumnView::rows_containing`] and
//! [`ColumnView::rows_containing_with_scan`] return the rows that match. To search
//! many chunks encoded with one dictionary, prepare a
//! [`ContainsScan`](search::ContainsScan) once and scan each of them.
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
