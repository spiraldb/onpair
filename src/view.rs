// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! ListView-style ("FSST view") compressed column.
//!
//! [`Column`] is an Arrow *List* layout: the shared decode buffers (the
//! dictionary — [`dict_bytes`](Column::dict_bytes),
//! [`dict_offsets`](Column::dict_offsets), [`bits`](Column::bits) — and the
//! flat [`codes`](Column::codes) stream) are delimited by one monotonic
//! [`code_offsets`](Column::code_offsets) array of `R + 1` entries, where row
//! `r` is `codes[code_offsets[r]..code_offsets[r + 1]]`.
//!
//! [`FsstView`] / [`FsstViewColumn`] are the Arrow *ListView* layout of the
//! same data: the single `R + 1` offset array is replaced by *two* `R`-entry
//! arrays — `code_offsets` (each row's start) and `code_sizes` (each row's code
//! count) — so row `r` is `codes[code_offsets[r]..code_offsets[r] +
//! code_sizes[r]]`. Everything else (the dictionary and the `codes` stream) is
//! byte-for-byte the same, and a single row still decodes through the existing
//! [`crate::decompress`] path via [`FsstView::row_parts`].
//!
//! # Why a view buys fast `filter` / `take` / `slice`
//!
//! In a List layout the offsets must stay monotonic and contiguous, so
//! reordering or selecting rows can't just permute offsets — it has to *gather*
//! every selected code into a fresh compact `codes` buffer and rebuild the
//! offsets, which is `O(total selected codes)` of data movement over the large
//! stream.
//!
//! A ListView lifts the monotonic/contiguous constraint: rows may appear in any
//! order, overlap, or leave gaps. So [`take`](FsstView::take),
//! [`filter`](FsstView::filter), and [`slice`](FsstView::slice) only permute or
//! sub-slice the `O(R)` `(offset, size)` metadata and keep the original `codes`
//! and dictionary buffers **shared and untouched** — `O(1)` in the code stream.
//! Each surviving row still points into the original `codes` via its
//! `(offset, size)` pair, and `codes[off..off + size]` is a contiguous slice
//! handed straight to the decoder, so nothing has to be re-encoded.

use std::borrow::Cow;

use crate::Column;
use crate::Offset;
use crate::Parts;

/// Owned ListView-style compressed column: a [`Column`] whose per-row
/// delimiters are an `(offset, size)` pair per row instead of one monotonic
/// offset array. Construct one from a [`Column`] with [`FsstViewColumn::from`]
/// / [`FsstViewColumn::from_column`], then borrow it as an [`FsstView`] with
/// [`as_view`](FsstViewColumn::as_view) to slice / take / filter.
#[derive(Debug, Clone)]
pub struct FsstViewColumn<O: Offset> {
    /// Dictionary bytes, with the trailing decoder padding required by
    /// [`Parts::validate_dictionary`]. Mirrors [`Column::dict_bytes`].
    pub dict_bytes: Vec<u8>,
    /// Token byte ranges into [`dict_bytes`](Self::dict_bytes). Mirrors
    /// [`Column::dict_offsets`].
    pub dict_offsets: Vec<u32>,
    /// Code width chosen at training time, in `9..=16`. Mirrors
    /// [`Column::bits`].
    pub bits: u32,
    /// The shared, row-concatenated code stream. Mirrors [`Column::codes`].
    /// Unlike a List layout this is never rebuilt by `take` / `filter` /
    /// `slice`; the per-row metadata addresses into it.
    pub codes: Vec<u16>,
    /// `code_offsets[r]` is the index into [`codes`](Self::codes) where row
    /// `r`'s codes begin. Length `R` (the row count). Need not be monotonic.
    pub code_offsets: Vec<O>,
    /// `code_sizes[r]` is the number of codes in row `r`, so row `r` is
    /// `codes[code_offsets[r]..code_offsets[r] + code_sizes[r]]`. Length `R`.
    pub code_sizes: Vec<O>,
}

impl<O: Offset> FsstViewColumn<O> {
    /// Convert a List-layout [`Column`] into the equivalent ListView column.
    ///
    /// The dictionary and `codes` buffers are moved through unchanged; the
    /// `R + 1` monotonic `code_offsets` become `R` starts plus `R` sizes
    /// (`sizes[r] = offsets[r + 1] - offsets[r]`). `O(R)` work, no code-stream
    /// movement.
    pub fn from_column(col: Column<O>) -> Self {
        let Column {
            dict_bytes,
            dict_offsets,
            bits,
            codes,
            mut code_offsets,
        } = col;
        let rows = code_offsets.len().saturating_sub(1);
        let code_sizes: Vec<O> = (0..rows)
            .map(|r| {
                let s = code_offsets[r].to_usize().expect("offset fits usize");
                let e = code_offsets[r + 1].to_usize().expect("offset fits usize");
                O::from_usize(e - s)
            })
            .collect();
        // Drop the trailing terminator offset so `code_offsets` is one start
        // per row, matching `code_sizes`.
        code_offsets.truncate(rows);
        Self {
            dict_bytes,
            dict_offsets,
            bits,
            codes,
            code_offsets,
            code_sizes,
        }
    }

    /// Number of rows in the column.
    #[inline]
    pub fn len(&self) -> usize {
        self.code_offsets.len()
    }

    /// Whether the column has no rows.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.code_offsets.is_empty()
    }

    /// Borrow this owned column as an [`FsstView`]. Every buffer is borrowed
    /// (no copy), so the returned view's [`slice`](FsstView::slice) is itself
    /// zero-copy.
    #[inline]
    pub fn as_view(&self) -> FsstView<'_, O> {
        FsstView {
            dict_bytes: &self.dict_bytes,
            dict_offsets: &self.dict_offsets,
            bits: self.bits,
            codes: &self.codes,
            code_offsets: Cow::Borrowed(&self.code_offsets),
            code_sizes: Cow::Borrowed(&self.code_sizes),
        }
    }
}

impl<O: Offset> From<Column<O>> for FsstViewColumn<O> {
    #[inline]
    fn from(col: Column<O>) -> Self {
        Self::from_column(col)
    }
}

/// Borrowed ListView-style column — the workhorse for `slice` / `take` /
/// `filter`.
///
/// The large shared buffers (dictionary + `codes`) are always borrowed; only
/// the `O(R)` per-row `(offset, size)` metadata is owned-or-borrowed via
/// [`Cow`], so a freshly built selection borrows the same code stream it was
/// built from. Obtain one from [`FsstViewColumn::as_view`] or
/// [`FsstView::from_column`].
#[derive(Debug, Clone)]
pub struct FsstView<'a, O: Offset> {
    /// Dictionary bytes, padded for the decoder. Borrowed; mirrors
    /// [`Parts::dict_bytes`].
    pub dict_bytes: &'a [u8],
    /// Token byte ranges into [`dict_bytes`](Self::dict_bytes). Borrowed;
    /// mirrors [`Parts::dict_offsets`].
    pub dict_offsets: &'a [u32],
    /// Code width, in `9..=16`. Mirrors [`Parts::bits`].
    pub bits: u32,
    /// The shared code stream that every row's `(offset, size)` addresses into.
    /// Borrowed and never rebuilt by the view operations.
    pub codes: &'a [u16],
    /// Per-row start index into [`codes`](Self::codes); length `R`. Need not be
    /// monotonic.
    pub code_offsets: Cow<'a, [O]>,
    /// Per-row code count; length `R`. Row `r` is
    /// `codes[code_offsets[r]..code_offsets[r] + code_sizes[r]]`.
    pub code_sizes: Cow<'a, [O]>,
}

impl<'a, O: Offset> FsstView<'a, O> {
    /// Borrow a List-layout [`Column`] as a ListView without consuming it. The
    /// dictionary and `codes` are borrowed; `code_offsets` is borrowed as the
    /// per-row starts and the `R` sizes are computed (`O(R)`, no code-stream
    /// movement).
    pub fn from_column(col: &'a Column<O>) -> Self {
        let rows = col.code_offsets.len().saturating_sub(1);
        let code_sizes: Vec<O> = col
            .code_offsets
            .windows(2)
            .map(|w| {
                let s = w[0].to_usize().expect("offset fits usize");
                let e = w[1].to_usize().expect("offset fits usize");
                O::from_usize(e - s)
            })
            .collect();
        Self {
            dict_bytes: &col.dict_bytes,
            dict_offsets: &col.dict_offsets,
            bits: col.bits,
            codes: &col.codes,
            code_offsets: Cow::Borrowed(&col.code_offsets[..rows]),
            code_sizes: Cow::Owned(code_sizes),
        }
    }

    /// Number of rows in the view.
    #[inline]
    pub fn len(&self) -> usize {
        self.code_offsets.len()
    }

    /// Whether the view has no rows.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.code_offsets.is_empty()
    }

    /// A [`Parts`] view for decoding a single row `r` against the shared
    /// dictionary. The returned `Parts.codes` is the contiguous sub-slice
    /// `codes[off..off + size]` — no copy. Decode it with [`crate::decompress`]
    /// or [`crate::decompress_into`].
    ///
    /// ## Panics
    ///
    /// Panics if `r` is out of range or the row's `(offset, size)` runs past
    /// the code stream.
    #[inline]
    pub fn row_parts(&self, r: usize) -> Parts<'a> {
        let off = self.code_offsets[r].to_usize().expect("offset fits usize");
        let size = self.code_sizes[r].to_usize().expect("size fits usize");
        Parts {
            dict_bytes: self.dict_bytes,
            dict_offsets: self.dict_offsets,
            bits: self.bits,
            // `self.codes` is `&'a [u16]`, so this sub-slice is also `'a`: the
            // returned `Parts` outlives `&self`, pointing into the shared
            // stream with no copy.
            codes: &self.codes[off..off + size],
        }
    }

    /// Decode row `r` to its original bytes. Convenience over
    /// [`row_parts`](Self::row_parts) + [`crate::decompress`].
    #[inline]
    pub fn decompress_row(&self, r: usize) -> Vec<u8> {
        crate::decompress(self.row_parts(r))
    }

    /// A `len`-row window starting at row `start`. Sub-slices the `O(len)`
    /// metadata only — the shared `codes` / dictionary buffers are untouched
    /// (`O(1)` in the code stream), and the sub-slice is zero-copy when the
    /// source metadata is borrowed.
    ///
    /// ## Panics
    ///
    /// Panics if `start + len > self.len()`.
    pub fn slice(&self, start: usize, len: usize) -> FsstView<'a, O> {
        assert!(start + len <= self.len(), "slice out of bounds");
        FsstView {
            dict_bytes: self.dict_bytes,
            dict_offsets: self.dict_offsets,
            bits: self.bits,
            codes: self.codes,
            code_offsets: slice_cow(&self.code_offsets, start, len),
            code_sizes: slice_cow(&self.code_sizes, start, len),
        }
    }

    /// Gather the rows named by `indices` into a new view. Rows may be
    /// reordered, repeated, or dropped. Only the `O(indices.len())`
    /// `(offset, size)` metadata is built; the shared `codes` / dictionary
    /// buffers are borrowed unchanged — no code-stream movement. (A List-layout
    /// take would have to gather every selected code into a fresh buffer.)
    ///
    /// ## Panics
    ///
    /// Panics if any index is out of range.
    pub fn take(&self, indices: &[usize]) -> FsstView<'a, O> {
        let code_offsets = indices.iter().map(|&i| self.code_offsets[i]).collect();
        let code_sizes = indices.iter().map(|&i| self.code_sizes[i]).collect();
        FsstView {
            dict_bytes: self.dict_bytes,
            dict_offsets: self.dict_offsets,
            bits: self.bits,
            codes: self.codes,
            code_offsets: Cow::Owned(code_offsets),
            code_sizes: Cow::Owned(code_sizes),
        }
    }

    /// Keep the rows whose `mask` entry is `true`, in order. Only the `O(R)`
    /// `(offset, size)` metadata is built; the shared `codes` / dictionary
    /// buffers are borrowed unchanged — no code-stream movement.
    ///
    /// ## Panics
    ///
    /// Panics if `mask.len() != self.len()`.
    pub fn filter(&self, mask: &[bool]) -> FsstView<'a, O> {
        assert_eq!(mask.len(), self.len(), "mask length must equal row count");
        let kept = mask.iter().filter(|&&keep| keep).count();
        let mut code_offsets = Vec::with_capacity(kept);
        let mut code_sizes = Vec::with_capacity(kept);
        for (i, &keep) in mask.iter().enumerate() {
            if keep {
                code_offsets.push(self.code_offsets[i]);
                code_sizes.push(self.code_sizes[i]);
            }
        }
        FsstView {
            dict_bytes: self.dict_bytes,
            dict_offsets: self.dict_offsets,
            bits: self.bits,
            codes: self.codes,
            code_offsets: Cow::Owned(code_offsets),
            code_sizes: Cow::Owned(code_sizes),
        }
    }
}

/// Sub-slice a `Cow<[O]>` preserving the source lifetime: zero-copy when the
/// source is borrowed, an `O(len)` clone of just the window when it is owned.
/// Either way the large shared buffers are never touched.
#[inline]
fn slice_cow<'a, O: Offset>(c: &Cow<'a, [O]>, start: usize, len: usize) -> Cow<'a, [O]> {
    match c {
        Cow::Borrowed(s) => Cow::Borrowed(&s[start..start + len]),
        Cow::Owned(v) => Cow::Owned(v[start..start + len].to_vec()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DEFAULT_CONFIG, compress};

    /// Build a column from `rows` and return it alongside the flat bytes.
    fn column(rows: &[&[u8]]) -> Column<u32> {
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for row in rows {
            bytes.extend_from_slice(row);
            offsets.push(bytes.len() as u32);
        }
        compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap()
    }

    #[test]
    fn view_round_trips_every_row() {
        let rows: &[&[u8]] = &[b"alpha", b"", b"beta beta", b"gamma", b"alpha"];
        let col = column(rows);
        let view = FsstViewColumn::from_column(col);
        assert_eq!(view.len(), rows.len());
        let view = view.as_view();
        for (r, expected) in rows.iter().enumerate() {
            assert_eq!(view.decompress_row(r), *expected, "row {r}");
        }
    }

    #[test]
    fn from_column_matches_list_layout() {
        // The view's sizes must reconstruct the List layout's offset deltas.
        let col = column(&[b"one", b"two!", b"three?!"]);
        let borrowed = FsstView::from_column(&col);
        for r in 0..borrowed.len() {
            let start = borrowed.code_offsets[r].to_usize().unwrap();
            let size = borrowed.code_sizes[r].to_usize().unwrap();
            assert_eq!(start, col.code_offsets[r] as usize);
            assert_eq!(start + size, col.code_offsets[r + 1] as usize);
        }
    }

    #[test]
    fn slice_keeps_rows_and_shares_codes() {
        let rows: &[&[u8]] = &[b"r0", b"r1xx", b"r2yyyy", b"r3", b"r4zzz"];
        let owned = FsstViewColumn::from_column(column(rows));
        let view = owned.as_view();
        let codes_ptr = view.codes.as_ptr();

        let mid = view.slice(1, 3);
        assert_eq!(mid.len(), 3);
        // The slice points at the SAME codes buffer — no rebuild.
        assert_eq!(mid.codes.as_ptr(), codes_ptr);
        for (i, r) in (1..4).enumerate() {
            assert_eq!(mid.decompress_row(i), rows[r]);
        }
    }

    #[test]
    fn take_reorders_and_repeats_without_moving_codes() {
        let rows: &[&[u8]] = &[b"apple", b"banana", b"cherry"];
        let owned = FsstViewColumn::from_column(column(rows));
        let view = owned.as_view();
        let codes_ptr = view.codes.as_ptr();

        let taken = view.take(&[2, 0, 2, 1]);
        assert_eq!(taken.codes.as_ptr(), codes_ptr);
        let want: &[&[u8]] = &[b"cherry", b"apple", b"cherry", b"banana"];
        for (i, w) in want.iter().enumerate() {
            assert_eq!(taken.decompress_row(i), *w, "taken row {i}");
        }
    }

    #[test]
    fn filter_selects_without_moving_codes() {
        let rows: &[&[u8]] = &[b"keep0", b"drop", b"keep2", b"drop", b"keep4"];
        let owned = FsstViewColumn::from_column(column(rows));
        let view = owned.as_view();
        let codes_ptr = view.codes.as_ptr();

        let kept = view.filter(&[true, false, true, false, true]);
        assert_eq!(kept.codes.as_ptr(), codes_ptr);
        assert_eq!(kept.len(), 3);
        let want: &[&[u8]] = &[b"keep0", b"keep2", b"keep4"];
        for (i, w) in want.iter().enumerate() {
            assert_eq!(kept.decompress_row(i), *w);
        }
    }

    #[test]
    fn ops_compose() {
        let rows: &[&[u8]] = &[b"a", b"bb", b"ccc", b"dddd", b"eeeee", b"ffffff"];
        let owned = FsstViewColumn::from_column(column(rows));
        // slice -> take -> filter, all sharing one codes buffer.
        let view = owned.as_view().slice(1, 4); // rows b,c,d,e
        let view = view.take(&[3, 0, 2]); // e, b, d
        let view = view.filter(&[true, false, true]); // e, d
        let want: &[&[u8]] = &[b"eeeee", b"dddd"];
        assert_eq!(view.len(), want.len());
        for (i, w) in want.iter().enumerate() {
            assert_eq!(view.decompress_row(i), *w);
        }
    }

    #[test]
    fn empty_column_is_empty_view() {
        let col = column(&[]);
        let owned = FsstViewColumn::from_column(col);
        assert!(owned.is_empty());
        assert!(owned.as_view().is_empty());
    }
}
