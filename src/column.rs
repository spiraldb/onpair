// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Owned `Column<O>` and borrowed `Parts<'a, O>`. Codes are stored as plain
// `u16` (no bit packing); per-row boundaries mirror the input offset width.

use crate::offset::Offset;

/// Owned compressed column produced by [`crate::compress`] /
/// [`crate::Parser::parse`].
#[derive(Debug, Clone)]
pub struct Column<O: Offset> {
    /// Dictionary bytes, with trailing decoder padding: the buffer extends
    /// [`crate::MAX_TOKEN_SIZE`] bytes past the highest token offset so the
    /// decoder's fixed-width read of any token is in bounds (see
    /// [`Parts::validate_dictionary`]). [`crate::Parser::parse`] emits it.
    pub dict_bytes: Vec<u8>,
    pub dict_offsets: Vec<u32>,
    pub bits: u32,
    pub codes: Vec<u16>,
    pub code_boundaries: Vec<O>,
}

/// Borrowed view of the same data, consumed by [`crate::decompress`] and
/// [`crate::decompress_into`].
/// Downstream consumers deserializing from storage build this via struct
/// literal — there is no constructor.
#[derive(Copy, Clone, Debug)]
pub struct Parts<'a, O: Offset> {
    pub dict_bytes: &'a [u8],
    pub dict_offsets: &'a [u32],
    pub bits: u32,
    pub codes: &'a [u16],
    pub code_boundaries: &'a [O],
}

impl<O: Offset> Column<O> {
    /// Zero-copy view over this column's arrays. Pass directly to
    /// [`crate::decompress`] or [`crate::decompress_into`].
    #[inline]
    pub fn as_parts(&self) -> Parts<'_, O> {
        Parts {
            dict_bytes: &self.dict_bytes,
            dict_offsets: &self.dict_offsets,
            bits: self.bits,
            codes: &self.codes,
            code_boundaries: &self.code_boundaries,
        }
    }
}
