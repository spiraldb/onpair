// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
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

//! OnPair: short-strings compression for fast random access.
//!
//! Rust port of the algorithm described in
//! [arXiv:2508.02280](https://arxiv.org/abs/2508.02280).
//!
//! ```ignore
//! use onpair::{compress, decompress_into, DEFAULT_CONFIG};
//!
//! let col = compress(&bytes, &offsets, DEFAULT_CONFIG)?;
//! let mut decoded = Vec::with_capacity(bytes.len());
//! let len = decompress_into(col.as_parts(), decoded.spare_capacity_mut());
//! unsafe { decoded.set_len(len) };
//! ```
//!
//! The trained encoder is also available directly:
//!
//! ```ignore
//! use onpair::{Parser, DEFAULT_CONFIG};
//!
//! let parser = Parser::train(&sample_bytes, &sample_offsets, DEFAULT_CONFIG)?;
//! let col_a = parser.parse(&corpus_a_bytes, &corpus_a_offsets)?;
//! let col_b = parser.parse(&corpus_b_bytes, &corpus_b_offsets)?;
//! ```

mod column;
mod config;
mod decompress;
mod dict;
mod hash;
mod lpm;
mod offset;
mod parser;
mod trainer;
mod types;

#[cfg(test)]
mod test_corpus;

pub use column::Column;
pub use column::Parts;
pub use config::Config;
pub use config::DEFAULT_CONFIG;
pub use config::Error;
pub use decompress::InvalidParts;
pub use decompress::decompress;
pub use decompress::decompress_into;
pub use decompress::decompress_into_unchecked;
pub use decompress::decompress_row_into;
pub use decompress::decompressed_len;
pub use decompress::decompressed_row_len;
pub use dict::Dictionary;
pub use offset::Offset;
pub use parser::Parser;
pub use types::MAX_TOKEN_SIZE;

/// Compress `bytes` / `offsets` end-to-end. Equivalent to
/// `Parser::train(..)?.parse(..)`.
pub fn compress<O: Offset>(bytes: &[u8], offsets: &[O], cfg: Config) -> Result<Column<O>, Error> {
    Parser::train(bytes, offsets, cfg)?.parse(bytes, offsets)
}
