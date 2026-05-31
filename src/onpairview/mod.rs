// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! OnPairView: the view-shaped sibling of the flat OnPair decode.
//!
//! [`crate::decompress_into`] decodes every code into one flat byte buffer in
//! row-concatenated order — which loses the row boundaries. OnPair exists for
//! *random access*, so a consumer almost always wants per-row structure back.
//! This module reconstructs it without re-running the encoder:
//!
//! * [`decompress_view`] decodes into a `values` buffer plus `R + 1` per-row
//!   `offsets` (the Arrow VarBin / ListView shape). The row boundaries come from
//!   the compressor's [`Column::code_offsets`], since a token may straddle a row
//!   boundary and the row structure cannot be recovered from the codes alone.
//! * [`build_views`] turns a decoded [`DecodedView`] into one
//!   [`BinaryView`] descriptor per row — Arrow's 16-byte StringView/BinaryView
//!   layout, with values ≤ [`BinaryView::INLINE_LEN`] bytes stored inline and
//!   longer values referencing the `values` buffer. This per-row "make view"
//!   pass is the dominant cost of exporting short strings, so it is the kernel
//!   to optimize (see the bulk split note on [`build_views`]).
//!
//! ```ignore
//! use onpair::{compress, DEFAULT_CONFIG};
//! use onpair::onpairview::build_views;
//!
//! let col = compress(&bytes, &offsets, DEFAULT_CONFIG)?;
//! let view = col.decompress_view();          // values + per-row offsets
//! let descriptors = build_views(&view);      // one BinaryView per row
//! assert_eq!(descriptors.len(), offsets.len() - 1);
//! ```
//!
//! [`Column::code_offsets`]: crate::Column::code_offsets

use crate::column::Parts;
use crate::offset::Offset;

#[cfg(test)]
mod tests;

/// A decoded column in view shape: a flat `values` buffer plus `R + 1` per-row
/// `offsets` into it (Arrow VarBin / ListView layout).
///
/// Row `r` occupies `values[offsets[r] as usize..offsets[r + 1] as usize]`.
/// `offsets[0] == 0` and `offsets[R] == values.len()`. Produced by
/// [`decompress_view`] / [`crate::Column::decompress_view`] and consumed by
/// [`build_views`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedView {
    /// Row bytes, concatenated in input order. The backing buffer for any
    /// non-inline [`BinaryView`] built from this view.
    pub values: Vec<u8>,
    /// `R + 1` byte offsets delimiting the `R` rows in [`values`](Self::values):
    /// row `r` is `values[offsets[r]..offsets[r + 1]]`.
    pub offsets: Vec<u32>,
}

impl DecodedView {
    /// Number of rows (`offsets.len() - 1`).
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Whether the view has no rows.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The bytes of row `r`.
    ///
    /// ## Panics
    ///
    /// Panics if `r` is out of range or the offsets are malformed.
    #[inline]
    #[must_use]
    pub fn row(&self, r: usize) -> &[u8] {
        let s = self.offsets[r] as usize;
        let e = self.offsets[r + 1] as usize;
        &self.values[s..e]
    }
}

/// An Arrow-compatible 16-byte binary view descriptor (the StringView /
/// BinaryView element layout).
///
/// The bytes are laid out exactly as Arrow's view buffer, little-endian:
///
/// * `[0..4]`   — `length: u32`.
/// * If `length <= ` [`INLINE_LEN`](Self::INLINE_LEN): `[4..4 + length]` holds
///   the value bytes and the remainder is zero (a fully self-contained,
///   buffer-free view).
/// * Otherwise: `[4..8]` is the 4-byte `prefix` (the value's first 4 bytes),
///   `[8..12]` a `buffer_index: u32`, and `[12..16]` an `offset: u32` into that
///   buffer.
///
/// Construct with [`inline`](Self::inline) / [`reference`](Self::reference) (or
/// [`build_views`] in bulk) and read back with the accessors. Because the byte
/// layout matches Arrow, a `&[BinaryView]` can be reinterpreted as an Arrow view
/// buffer without a copy.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct BinaryView([u8; 16]);

impl BinaryView {
    /// The largest value length, in bytes, that is stored inline. Values up to
    /// this length carry their bytes in the descriptor itself and need no
    /// backing buffer; longer values are stored by reference.
    pub const INLINE_LEN: usize = 12;

    /// Build an inline view holding `bytes` directly.
    ///
    /// ## Panics
    ///
    /// Panics if `bytes.len() > ` [`INLINE_LEN`](Self::INLINE_LEN).
    #[inline]
    #[must_use]
    pub fn inline(bytes: &[u8]) -> Self {
        assert!(
            bytes.len() <= Self::INLINE_LEN,
            "inline view exceeds INLINE_LEN"
        );
        let mut raw = [0u8; 16];
        raw[0..4].copy_from_slice(&(bytes.len() as u32).to_le_bytes());
        raw[4..4 + bytes.len()].copy_from_slice(bytes);
        Self(raw)
    }

    /// Build a reference view for a value of `len` bytes whose first four bytes
    /// are `prefix`, stored at `offset` in buffer `buffer_index`.
    ///
    /// ## Panics
    ///
    /// Panics if `len <= ` [`INLINE_LEN`](Self::INLINE_LEN) (such values must be
    /// [inline](Self::inline)).
    #[inline]
    #[must_use]
    pub fn reference(len: u32, prefix: [u8; 4], buffer_index: u32, offset: u32) -> Self {
        assert!(
            len as usize > Self::INLINE_LEN,
            "reference view must exceed INLINE_LEN"
        );
        let mut raw = [0u8; 16];
        raw[0..4].copy_from_slice(&len.to_le_bytes());
        raw[4..8].copy_from_slice(&prefix);
        raw[8..12].copy_from_slice(&buffer_index.to_le_bytes());
        raw[12..16].copy_from_slice(&offset.to_le_bytes());
        Self(raw)
    }

    /// The value's length in bytes.
    #[inline]
    #[must_use]
    pub fn len(&self) -> u32 {
        u32::from_le_bytes([self.0[0], self.0[1], self.0[2], self.0[3]])
    }

    /// Whether this view is empty (length 0).
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether the value bytes are stored inline (rather than by reference).
    #[inline]
    #[must_use]
    pub fn is_inline(&self) -> bool {
        self.len() as usize <= Self::INLINE_LEN
    }

    /// The inline value bytes, or `None` for a reference view.
    #[inline]
    #[must_use]
    pub fn inline_bytes(&self) -> Option<&[u8]> {
        self.is_inline()
            .then(|| &self.0[4..4 + self.len() as usize])
    }

    /// The `(buffer_index, offset)` of a reference view, or `None` if inline.
    #[inline]
    #[must_use]
    pub fn reference_location(&self) -> Option<(u32, u32)> {
        (!self.is_inline()).then(|| {
            let buffer = u32::from_le_bytes([self.0[8], self.0[9], self.0[10], self.0[11]]);
            let offset = u32::from_le_bytes([self.0[12], self.0[13], self.0[14], self.0[15]]);
            (buffer, offset)
        })
    }

    /// The raw 16-byte Arrow descriptor.
    #[inline]
    #[must_use]
    pub fn to_le_bytes(self) -> [u8; 16] {
        self.0
    }

    /// Resolve the value bytes against the backing `buffers`: an inline view
    /// returns its own bytes; a reference view indexes `buffers`.
    ///
    /// ## Panics
    ///
    /// Panics if a reference view names a missing buffer or an out-of-range
    /// offset.
    #[inline]
    #[must_use]
    pub fn resolve<'a>(&'a self, buffers: &'a [&'a [u8]]) -> &'a [u8] {
        match self.reference_location() {
            None => &self.0[4..4 + self.len() as usize],
            Some((buffer, offset)) => {
                let s = offset as usize;
                &buffers[buffer as usize][s..s + self.len() as usize]
            }
        }
    }
}

impl std::fmt::Debug for BinaryView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("BinaryView");
        d.field("len", &self.len());
        if let Some(bytes) = self.inline_bytes() {
            d.field("inline", &bytes);
        } else if let Some((buffer, offset)) = self.reference_location() {
            d.field("buffer", &buffer).field("offset", &offset);
        }
        d.finish()
    }
}

/// Decode `parts` into [view shape](DecodedView): a flat `values` buffer plus
/// `R + 1` per-row byte `offsets`, where the `R` rows are delimited by
/// `code_offsets` (the compressor's [`Column::code_offsets`]).
///
/// `code_offsets` has `R + 1` entries; row `r`'s codes are
/// `parts.codes[code_offsets[r]..code_offsets[r + 1]]`. The decoded `values` are
/// identical to [`crate::decompress`]; the added work is the per-row offset
/// prefix sum over the codes' token lengths.
///
/// Two passes: [`row_byte_offsets`] yields both the per-row offsets and the total
/// decoded length (so there is no separate length pass) and validates the code
/// range as it indexes the dictionary, so the decode then runs unchecked. The
/// dictionary itself is validated once up front for the decode's over-read.
///
/// ## Panics
///
/// Panics if `parts` is malformed (see [`Parts::validate`]) or if `code_offsets`
/// is empty, not non-decreasing, or its last entry is not `parts.codes.len()`.
///
/// [`Column::code_offsets`]: crate::Column::code_offsets
#[must_use]
pub fn decompress_view<O: Offset>(parts: Parts<'_>, code_offsets: &[O]) -> DecodedView {
    // The unchecked decode over-reads the dictionary (fixed 16-byte token reads),
    // so validate the padding invariant once, off the hot loop.
    crate::decompress::assert_valid_dictionary(parts);
    let offsets = row_byte_offsets(parts, code_offsets);
    let total = offsets.last().copied().unwrap_or(0) as usize;
    let mut values: Vec<u8> = Vec::with_capacity(total);
    // SAFETY: the dictionary is validated (padding ⇒ the over-copy decode's
    // fixed-width reads are in bounds); `row_byte_offsets` panicked unless every
    // code indexes the dictionary; and `total` is the exact decoded length, so
    // the store cannot overrun `values`.
    let n = unsafe { crate::decompress_into_unchecked(parts, values.spare_capacity_mut()) };
    debug_assert_eq!(n, total, "decoded length must equal the row offsets total");
    // SAFETY: `decompress_into_unchecked` initialized exactly `n` leading bytes.
    unsafe { values.set_len(n) };
    DecodedView { values, offsets }
}

/// Compute the `R + 1` per-row byte offsets for `code_offsets` over `parts`,
/// without materializing the decoded bytes.
///
/// Row `r`'s decoded byte length is the sum of its codes' token lengths; the
/// returned offsets are the running prefix sum, so `offsets[r + 1] - offsets[r]`
/// is row `r`'s byte length and `offsets[R]` the total decoded length.
///
/// ## Panics
///
/// Panics on a malformed `code_offsets` (see [`decompress_view`]) or an
/// out-of-range code.
#[must_use]
pub fn row_byte_offsets<O: Offset>(parts: Parts<'_>, code_offsets: &[O]) -> Vec<u32> {
    assert!(!code_offsets.is_empty(), "code_offsets must have R + 1 ≥ 1");
    let rows = code_offsets.len() - 1;
    let mut offsets = Vec::with_capacity(rows + 1);
    offsets.push(0u32);

    let mut acc = 0u32;
    let mut ci = 0usize;
    for r in 0..rows {
        let end = code_offsets[r + 1]
            .to_usize()
            .expect("code offset exceeds usize");
        assert!(
            end >= ci && end <= parts.codes.len(),
            "code_offsets must be non-decreasing and within codes"
        );
        while ci < end {
            let c = parts.codes[ci] as usize;
            acc += parts.dict_offsets[c + 1] - parts.dict_offsets[c];
            ci += 1;
        }
        offsets.push(acc);
    }
    offsets
}

/// Build one Arrow [`BinaryView`] descriptor per row of `view`.
///
/// Rows up to [`BinaryView::INLINE_LEN`] bytes are stored inline; longer rows
/// reference buffer `0` (i.e. `view.values`) at the row's offset. The returned
/// `Vec` reinterprets, byte-for-byte, as an Arrow view buffer over the single
/// data buffer `view.values`.
///
/// This per-row "make view" pass is the dominant cost of exporting short
/// strings: every row touches a 16-byte descriptor and the work splits on the
/// inline-vs-reference branch. It is the kernel a future bulk/SIMD pass should
/// target — compute every row length up front, then build descriptors with a
/// branch-reduced inline/reference split.
///
/// ## Panics
///
/// Panics if `view.offsets` is malformed (not `R + 1` non-decreasing entries) or
/// any row exceeds `u32::MAX` bytes.
#[must_use]
pub fn build_views(view: &DecodedView) -> Vec<BinaryView> {
    let mut out = Vec::new();
    build_views_into(view, &mut out);
    out
}

/// Like [`build_views`], but writes the descriptors into a caller-owned `out`,
/// reusing its allocation. `out` is cleared first and ends with exactly one
/// [`BinaryView`] per row.
///
/// Export loops that view many columns should keep one `out` buffer and call
/// this per column: the per-row work then dominates instead of repeatedly
/// allocating (and page-faulting) a fresh descriptor buffer.
///
/// ## Panics
///
/// Same conditions as [`build_views`].
pub fn build_views_into(view: &DecodedView, out: &mut Vec<BinaryView>) {
    let rows = view.len();
    let offsets = &view.offsets;
    let values = &view.values;
    out.clear();
    out.reserve(rows);

    // Fast region: a row whose value starts at least 16 bytes before the end of
    // `values` can be loaded with one unaligned 16-byte read, so its descriptor
    // is assembled as a single `u128` and written with one store — no per-row
    // zero-init or `copy_from_slice`. `make_view_u128` keeps the inline/reference
    // split (a within-column-predictable branch) but reduces each arm to a
    // mask-shift-or. Rows in the final 16 bytes fall to the scalar tail, where a
    // 16-byte over-read could run off the buffer.
    // `start <= safe_end` ⇔ `start + 16 <= values.len()` (16 readable bytes).
    // `None` when the whole buffer is shorter than the over-read window, so every
    // row takes the scalar tail.
    let safe_end = values.len().checked_sub(BinaryView::INLINE_LEN + 4);
    let vbase = values.as_ptr();
    let dst = out.as_mut_ptr();
    let mut r = 0usize;
    // `start` and `end` are read independently per row rather than carried
    // (`start = previous end`): the independent loads have no loop-carried
    // dependency, so the out-of-order engine keeps many iterations in flight.
    // Carrying `end` forward to save one load measured *slower* (it serializes
    // the loop) — measure before "deduplicating" this read.
    while r < rows {
        let start = offsets[r] as usize;
        if safe_end.is_none_or(|se| start > se) {
            break;
        }
        let end = offsets[r + 1] as usize;
        debug_assert!(end >= start, "row offsets must be non-decreasing");
        let len = (end - start) as u32;
        // SAFETY: `start <= safe_end` ⇒ `start + 16 <= values.len()`, so the
        // 16-byte read is in bounds; `r < rows <= capacity`, so `dst.add(r)` is a
        // valid, owned slot whose `set_len` below makes it initialized.
        unsafe {
            let chunk = vbase.add(start).cast::<u128>().read_unaligned();
            let raw = make_view_u128(u128::from_le(chunk), len, start as u32);
            dst.add(r).write(BinaryView(raw.to_le_bytes()));
        }
        r += 1;
    }

    // Scalar tail: the trailing rows whose value lies within the last 16 bytes
    // (and the whole input when `values` is shorter than the over-read window).
    for r in r..rows {
        let start = offsets[r] as usize;
        let end = offsets[r + 1] as usize;
        assert!(end >= start, "row offsets must be non-decreasing");
        let len = (end - start) as u32;
        let bv = if len as usize <= BinaryView::INLINE_LEN {
            BinaryView::inline(&values[start..end])
        } else {
            let prefix: [u8; 4] = values[start..start + 4]
                .try_into()
                .expect("len > INLINE_LEN guarantees ≥ 4 bytes");
            BinaryView::reference(len, prefix, 0, start as u32)
        };
        // SAFETY: `r < rows <= capacity`; the slot is owned and `set_len` below
        // makes the whole `[0, rows)` range initialized.
        unsafe { dst.add(r).write(bv) };
    }

    // SAFETY: `reserve(rows)` gave capacity ≥ rows and every slot in `[0, rows)`
    // was written by exactly one of the two loops above.
    unsafe { out.set_len(rows) };
}

/// Assemble the little-endian `u128` of a [`BinaryView`] descriptor from a
/// 16-byte little-endian load `chunk` starting at the value's first byte.
///
/// `chunk`'s byte `k` is value byte `k`. The Arrow descriptor places the length
/// in bits `[0, 32)` and value/reference data from bit 32 up, so each arm is a
/// mask-shift-or over `chunk` (see [`BinaryView`] for the byte layout). Inline
/// values keep their low `len` bytes; references keep the 4-byte prefix and add
/// the (buffer 0) offset.
#[inline]
fn make_view_u128(chunk: u128, len: u32, offset: u32) -> u128 {
    if len as usize <= BinaryView::INLINE_LEN {
        // Keep the low `len` value bytes at descriptor byte 4 (bit 32). `len <=
        // 12` ⇒ the shift `8 * len <= 96` never overflows; `len == 0` masks to 0.
        let keep = (1u128 << (8 * len)) - 1;
        (len as u128) | ((chunk & keep) << 32)
    } else {
        // [len][prefix = first 4 value bytes][buffer = 0][offset].
        let prefix = chunk & 0xFFFF_FFFF;
        (len as u128) | (prefix << 32) | ((offset as u128) << 96)
    }
}
