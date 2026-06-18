// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The wide (decode-optimized) dictionary representation.
//!
//! [`WideDictionary`] stores `num_tokens` rows of [`MAX_TOKEN_SIZE`] bytes — row
//! `id` holds token `id` (zero-padded) — plus a per-token length. Trades space
//! for a **load-free** token address: token `id`'s bytes are at `data + id*16`,
//! with no `code → offset → bytes` indirection, so a decode is one independent
//! load.
//!
//! Build it from the compact form with [`CompactDictionaryView::to_wide`];
//! recover the compact form with [`WideDictionaryView::to_compact`]. Both
//! representations implement [`DictionaryView`] (via their views), so the decode
//! kernels treat them uniformly.

use super::{CompactDictionary, Dictionary, DictionaryView};
use crate::core::types::{MAX_TOKEN_SIZE, Token};

/// Owned wide dictionary: `num_tokens` rows of [`MAX_TOKEN_SIZE`] bytes plus
/// per-token lengths.
///
/// Fields are public (like [`CompactDictionary`]) so a consumer can construct one
/// from buffers deserialized from storage. Invariants, a precondition of
/// decoding: `data.len() == num_tokens * MAX_TOKEN_SIZE`, `lens.len() == num_tokens`,
/// each `lens[id]` in `1..=MAX_TOKEN_SIZE`, and row `id`'s first `lens[id]` bytes
/// are token `id`.
#[derive(Default, Debug, Clone)]
pub struct WideDictionary {
    /// `num_tokens * MAX_TOKEN_SIZE` bytes. The widest decode access is the
    /// 16-byte over-store of the last row, which ends exactly at the buffer end
    /// (load width == row stride) — no trailing padding needed.
    pub data: Vec<u8>,
    /// `num_tokens` true token lengths.
    pub lens: Vec<u8>,
}

impl WideDictionary {
    /// Number of tokens.
    #[inline]
    pub fn num_tokens(&self) -> usize {
        self.lens.len()
    }

    /// Rebuild the [`CompactDictionary`] form (see
    /// [`WideDictionaryView::to_compact`]). Borrow as a view first with
    /// [`Dictionary::as_view`].
    #[inline]
    pub fn to_compact(&self) -> CompactDictionary {
        self.as_view().to_compact()
    }
}

impl Dictionary for WideDictionary {
    type View<'a> = WideDictionaryView<'a>;
    #[inline]
    fn as_view(&self) -> WideDictionaryView<'_> {
        WideDictionaryView { data: &self.data, lens: &self.lens }
    }
}

/// Borrowed, `Copy` view over a wide dictionary's buffers.
#[derive(Copy, Clone, Debug)]
pub struct WideDictionaryView<'a> {
    /// `num_tokens * MAX_TOKEN_SIZE` row-major token bytes.
    pub data: &'a [u8],
    /// `num_tokens` true token lengths.
    pub lens: &'a [u8],
}

impl<'a> WideDictionaryView<'a> {
    /// Number of tokens.
    #[inline]
    pub fn num_tokens(&self) -> usize {
        self.lens.len()
    }

    /// Rebuild the [`CompactDictionary`] form (read-padded, decodable).
    ///
    /// Safe: copies only `lens[id]` exact bytes per row, never over-reads.
    pub fn to_compact(&self) -> CompactDictionary {
        let n = self.num_tokens();
        let mut bytes = Vec::new();
        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0u32);
        for id in 0..n {
            let len = self.lens[id] as usize;
            let row = id * MAX_TOKEN_SIZE;
            bytes.extend_from_slice(&self.data[row..row + len]);
            offsets.push(bytes.len() as u32);
        }
        let mut d = CompactDictionary { bytes, offsets };
        d.pad_for_decoder(); // restore the read-padding invariant
        d
    }
}

impl DictionaryView for WideDictionaryView<'_> {
    #[inline]
    fn token(&self, id: Token) -> &[u8] {
        let row = id as usize * MAX_TOKEN_SIZE;
        &self.data[row..row + self.lens[id as usize] as usize]
    }

    #[inline]
    fn token_len(&self, id: Token) -> usize {
        self.lens[id as usize] as usize
    }

    #[inline]
    fn decoded_len(&self, codes: &[Token]) -> usize {
        codes.iter().map(|&id| self.lens[id as usize] as usize).sum()
    }

    #[inline]
    unsafe fn token_ptr(&self, id: Token) -> *const u8 {
        // SAFETY: id < num_tokens ⇒ row id is within data; the last row ends
        // exactly at data.len() (= n*16), so 16 bytes are readable. No load.
        unsafe { self.data.as_ptr().add(id as usize * MAX_TOKEN_SIZE) }
    }

    #[inline]
    unsafe fn token_len_unchecked(&self, id: Token) -> usize {
        // SAFETY: id < num_tokens ⇒ lens[id] is in bounds.
        unsafe { *self.lens.get_unchecked(id as usize) as usize }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn padded_compact(tokens: &[&[u8]]) -> CompactDictionary {
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for t in tokens {
            bytes.extend_from_slice(t);
            offsets.push(bytes.len() as u32);
        }
        let mut d = CompactDictionary { bytes, offsets };
        d.pad_for_decoder();
        d
    }

    #[test]
    fn num_tokens_counts_rows() {
        let wide = padded_compact(&[b"a", b"bc", b"def"]).to_wide();
        assert_eq!(wide.num_tokens(), 3);
        assert_eq!(wide.as_view().num_tokens(), 3);
    }

    #[test]
    fn to_compact_round_trips_all_length_buckets() {
        // Token lengths spanning every copy bucket: 1, 3, 5, 11, 15, 16.
        let t = [vec![b'a'; 1], vec![b'b'; 3], vec![b'c'; 5], vec![b'd'; 11], vec![b'e'; 15], vec![b'f'; 16]];
        let tokens: Vec<&[u8]> = t.iter().map(Vec::as_slice).collect();
        let compact = padded_compact(&tokens);
        let back = compact.to_wide().to_compact();
        assert_eq!(back.offsets, compact.offsets);
        assert_eq!(&back.bytes[..back.logical_len()], &compact.bytes[..compact.logical_len()]);
    }
}
