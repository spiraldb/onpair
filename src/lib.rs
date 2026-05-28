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
//! use onpair::{compress, decompress, DEFAULT_CONFIG};
//!
//! let col = compress(&bytes, &offsets, DEFAULT_CONFIG)?;
//! let decoded = decompress(col.as_parts());
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
pub use dict::Dictionary;
pub use offset::Offset;
pub use parser::Parser;

/// Compress `bytes` / `offsets` end-to-end. Equivalent to
/// `Parser::train(..)?.parse(..)`.
pub fn compress<O: Offset>(bytes: &[u8], offsets: &[O], cfg: Config) -> Result<Column<O>, Error> {
    Parser::train(bytes, offsets, cfg)?.parse(bytes, offsets)
}

/// Decode every row in a [`Parts`] view into one flat byte buffer in input
/// order. The caller already owns the row offsets (they passed them to
/// [`compress`] or used them to build the `Parts`), so they are not returned.
///
/// Does not validate the `Parts` invariants documented in the crate-root
/// PUBLIC_API: a malformed `Parts` will panic or produce out-of-bounds reads.
pub fn decompress<O: Offset>(parts: Parts<'_, O>) -> Vec<u8> {
    let num_rows = parts.code_boundaries.len().saturating_sub(1);
    let mut out: Vec<u8> = Vec::with_capacity(parts.codes.len() * 2);
    for row in 0..num_rows {
        let begin = parts.code_boundaries[row]
            .to_usize()
            .expect("code boundary fits usize");
        let end = parts.code_boundaries[row + 1]
            .to_usize()
            .expect("code boundary fits usize");
        for &c in &parts.codes[begin..end] {
            let s = parts.dict_offsets[c as usize] as usize;
            let e = parts.dict_offsets[c as usize + 1] as usize;
            out.extend_from_slice(&parts.dict_bytes[s..e]);
        }
    }
    out
}
