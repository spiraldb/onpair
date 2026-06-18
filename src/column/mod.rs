// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The compressed column: decodable data (dictionary + code stream) plus the
//! row layer that delimits the original strings within the stream.
//!
//! The code stream is plain [`Token`] data — there is no separate wrapper type.
//! A bulk-only consumer ignores `row_offsets`; the decode kernels never read it.

use crate::core::dictionary::{CompactDictionary, CompactDictionaryView, Dictionary, WideDictionary};
use crate::core::offset::Offset;
use crate::core::types::Token;
use crate::decoding;

/// Owned compressed column, produced by [`Column::compress`] /
/// [`Parser::parse`](crate::Parser::parse). Self-contained: it carries its own
/// dictionary, so it decodes without reference to the training corpus.
#[derive(Debug, Clone)]
pub struct Column<O: Offset> {
    /// Token dictionary, read-padded.
    pub dict: CompactDictionary,
    /// Code stream: one [`Token`] per emitted token, in row-concatenated order.
    /// Every code is `< dict.num_tokens()`.
    pub codes: Vec<Token>,
    /// Row layer: `R + 1` offsets into `codes` delimiting the `R` rows. Row `k`
    /// is `codes[row_offsets[k]..row_offsets[k + 1]]`. `row_offsets[0] == 0`,
    /// non-decreasing, and `row_offsets[R] == codes.len()`.
    pub row_offsets: Vec<O>,
}

impl<O: Offset> Column<O> {
    /// Compress an Arrow `(bytes, offsets)` value pair end-to-end (train a
    /// dictionary, then encode). `offsets` has `n + 1` entries; string `i` is
    /// `bytes[offsets[i]..offsets[i + 1]]`.
    ///
    /// # Errors
    /// [`Error::InvalidArg`](crate::Error::InvalidArg) if `offsets` is empty or
    /// its last entry exceeds `bytes.len()`.
    ///
    /// # Examples
    /// ```
    /// use onpair::{Column, DEFAULT_CONFIG};
    ///
    /// let bytes = b"catdogcat";
    /// let offsets: [u32; 4] = [0, 3, 6, 9];
    /// let col = Column::compress(bytes, &offsets, DEFAULT_CONFIG).unwrap();
    ///
    /// assert_eq!(col.view().decompress(), b"catdogcat");
    /// assert_eq!(col.view().decompress_row(1), b"dog");
    /// ```
    pub fn compress(bytes: &[u8], offsets: &[O], cfg: crate::Config) -> Result<Self, crate::Error> {
        crate::compress(bytes, offsets, cfg)
    }

    /// Borrow as a [`ColumnView`].
    #[inline]
    pub fn view(&self) -> ColumnView<'_, O> {
        ColumnView {
            dict: self.dict.as_view(),
            codes: &self.codes,
            row_offsets: &self.row_offsets,
        }
    }
}

/// Borrowed, `Copy` view over a compressed column — obtained from a [`Column`]
/// or built directly from buffers deserialized from storage.
#[derive(Copy, Clone, Debug)]
pub struct ColumnView<'a, O: Offset> {
    /// The token dictionary.
    pub dict: CompactDictionaryView<'a>,
    /// The code stream (see [`Column::codes`]).
    pub codes: &'a [Token],
    /// The row layer (see [`Column::row_offsets`]).
    pub row_offsets: &'a [O],
}

impl<'a, O: Offset> ColumnView<'a, O> {
    /// Number of rows.
    #[inline]
    pub fn num_rows(&self) -> usize {
        self.row_offsets.len().saturating_sub(1)
    }

    /// The codes for row `k`. Precondition: `k < num_rows()`.
    #[inline]
    pub fn row_codes(&self, k: usize) -> &'a [Token] {
        let a = self.row_offsets[k].to_usize();
        let b = self.row_offsets[k + 1].to_usize();
        &self.codes[a..b]
    }

    /// Exact decoded byte length of the whole column.
    #[inline]
    pub fn decoded_len(&self) -> usize {
        decoding::decoded_len(self.codes, self.dict)
    }

    /// Build a reusable [`WideDictionary`] for this column's dictionary. Amortize
    /// it across many decodes (`decode_into`/`decode_to_vec` over its view) when
    /// doing repeated random access; for a single bulk decode prefer
    /// [`Self::decompress`].
    #[inline]
    pub fn wide_dict(&self) -> WideDictionary {
        self.dict.to_wide()
    }

    /// Decode the whole column into a fresh `Vec` (bulk path, via the wide form).
    pub fn decompress(&self) -> Vec<u8> {
        // A column upholds the invariants: every code is in range.
        decoding::decode_to_vec(self.codes, self.wide_dict().as_view())
    }

    /// Decode row `k` into a fresh `Vec` (random-access path, compact dictionary —
    /// no wide-table build). Precondition: `k < num_rows()`.
    pub fn decompress_row(&self, k: usize) -> Vec<u8> {
        decoding::decode_to_vec(self.row_codes(k), self.dict)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Bits, Config, DEFAULT_CONFIG, compress};

    fn pack(rows: &[&[u8]]) -> (Vec<u8>, Vec<u32>) {
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for r in rows {
            bytes.extend_from_slice(r);
            offsets.push(bytes.len() as u32);
        }
        (bytes, offsets)
    }

    #[test]
    fn roundtrip_bulk_and_per_row() {
        let rows: &[&[u8]] = &[b"alpha", b"", b"beta beta", b"gamma"];
        let (bytes, offsets) = pack(rows);
        let col = compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap();
        let view = col.view();

        assert_eq!(view.decompress(), bytes);
        assert_eq!(view.num_rows(), rows.len());
        for (k, row) in rows.iter().enumerate() {
            assert_eq!(view.decompress_row(k), *row, "row {k}");
        }
    }

    #[test]
    fn roundtrip_across_bit_widths() {
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for i in 0..5000u32 {
            let row = format!("row-{i:04}-https://example.com/path/{}", i % 37);
            bytes.extend_from_slice(row.as_bytes());
            offsets.push(bytes.len() as u32);
        }
        for bits in 9..=16u8 {
            let cfg = Config { bits: Bits::new(bits).unwrap(), ..DEFAULT_CONFIG };
            let col = compress(&bytes, &offsets, cfg).unwrap();
            assert_eq!(col.view().decompress(), bytes, "bits={bits}");
        }
    }

    #[test]
    fn code_bits_is_within_capacity() {
        let (bytes, offsets) = pack(&[b"hello world", b"hello there", b"world peace"]);
        let cfg = Config { bits: Bits::new(12).unwrap(), ..DEFAULT_CONFIG };
        let col = compress(&bytes, &offsets, cfg).unwrap();
        // Minimal packing width never exceeds the configured capacity.
        assert!(col.dict.code_bits() <= 12);
    }
}
