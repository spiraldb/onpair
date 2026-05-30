// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Owned `Column<O>` and borrowed `Parts<'a>`. Codes are stored as plain `u16`
// (no bit packing). `Column` additionally carries per-row `code_offsets`
// (mirroring the input offset width); see the field docs.

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
    /// `dict_offsets[i]..dict_offsets[i + 1]` is token `i`'s byte range in
    /// [`dict_bytes`](Self::dict_bytes); `dict_offsets.len() == num_tokens + 1`
    /// and `dict_offsets[0] == 0`.
    pub dict_offsets: Vec<u32>,
    /// Code width chosen at training time, in `9..=16`. Consumers may use it to
    /// store [`codes`](Self::codes) more compactly than `u16`.
    pub bits: u32,
    /// One `u16` per encoded token, in row-concatenated order; each indexes a
    /// token via [`dict_offsets`](Self::dict_offsets).
    pub codes: Vec<u16>,
    /// `R + 1` offsets into `codes` delimiting the `R` input rows: row `r`'s
    /// codes are `codes[code_offsets[r]..code_offsets[r + 1]]`. The compressor
    /// emits these because a token may span a row boundary, so the row
    /// structure cannot be recovered from the codes alone.
    pub code_offsets: Vec<O>,
    /// Per-row first token id (`R` entries): `first_codes[r] == codes` of the
    /// first token of row `r`, or [`u16::MAX`] for an empty row. A contiguous
    /// side-table that lets prefix search prefilter rows with a single linear
    /// scan instead of a scattered `codes[code_offsets[r]]` gather per row —
    /// see [`crate::SearchParts::search`]. Costs 2 bytes per row.
    pub first_codes: Vec<u16>,
}

/// Borrowed view of the data the decoder needs, consumed by
/// [`crate::decompress`] and [`crate::decompress_into`].
/// Downstream consumers deserializing from storage build this via struct
/// literal — there is no constructor.
#[derive(Copy, Clone, Debug)]
pub struct Parts<'a> {
    /// Dictionary bytes, with the trailing decoder padding required by
    /// [`validate_dictionary`](Self::validate_dictionary). Mirrors
    /// [`Column::dict_bytes`].
    pub dict_bytes: &'a [u8],
    /// Token byte ranges into [`dict_bytes`](Self::dict_bytes); mirrors
    /// [`Column::dict_offsets`].
    pub dict_offsets: &'a [u32],
    /// Code width chosen at training time, in `9..=16`; mirrors
    /// [`Column::bits`].
    pub bits: u32,
    /// Encoded tokens indexing [`dict_offsets`](Self::dict_offsets); mirrors
    /// [`Column::codes`].
    pub codes: &'a [u16],
}

impl<O: Offset> Column<O> {
    /// Zero-copy view over this column's decode arrays. Pass directly to
    /// [`crate::decompress`] or [`crate::decompress_into`]. `code_offsets` is
    /// compressor metadata and is not part of the view.
    #[inline]
    pub fn as_parts(&self) -> Parts<'_> {
        Parts {
            dict_bytes: &self.dict_bytes,
            dict_offsets: &self.dict_offsets,
            bits: self.bits,
            codes: &self.codes,
        }
    }
}
