// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The compact dictionary: Arrow binary `bytes` + `offsets`.
//!
//! Layout is Arrow binary — a flat `bytes` buffer plus an `offsets` index of
//! length `num_tokens + 1`; token `i` is `bytes[offsets[i]..offsets[i + 1]]`.
//!
//! # Invariants
//! Upheld by the trainer; a precondition of every accessor and of decoding.
//! - `offsets[0] == 0` and `offsets.len() == num_tokens + 1`.
//! - **Strictly increasing** offsets — every token is non-empty, with length in
//!   `1..=MAX_TOKEN_SIZE`.
//! - **Sorted** — tokens are in strictly ascending bytewise-lexicographic order.
//! - **Complete** — all 256 single-byte tokens are present, so any byte string
//!   is encodable.
//! - **Unique** — no two tokens are equal.
//! - **Read-padded** — `bytes` is readable for [`MAX_TOKEN_SIZE`] bytes past the
//!   highest token offset (call [`CompactDictionary::pad_for_decoder`] once after
//!   filling). `offsets.last()` is the logical length; `bytes.len()` may exceed
//!   it by the padding.

use super::{Dictionary, DictionaryView, WideDictionary};
use crate::core::types::{MAX_TOKEN_SIZE, Token, code_bits_for};

/// Owned compact dictionary.
///
/// Fields are public — this is a data type a consumer may construct directly
/// from buffers it deserialized from storage — and must satisfy the invariants
/// described in this module's documentation.
#[derive(Default, Debug, Clone)]
pub struct CompactDictionary {
    /// Concatenated token bytes, followed by read-padding.
    pub bytes: Vec<u8>,
    /// `num_tokens + 1` offsets delimiting the tokens within `bytes`.
    pub offsets: Vec<u32>,
}

impl CompactDictionary {
    /// Number of tokens.
    #[inline]
    pub fn num_tokens(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Logical byte length — token bytes only, excluding read-padding.
    #[inline]
    pub fn logical_len(&self) -> usize {
        self.offsets.last().copied().unwrap_or(0) as usize
    }

    /// Minimum bits per code needed to address this dictionary,
    /// `ceil(log2(num_tokens))`. A consumer that bit-packs the code stream packs
    /// each code in this many bits. `8..=16` for a conformant dictionary.
    #[inline]
    pub fn code_bits(&self) -> u8 {
        code_bits_for(self.num_tokens())
    }

    /// Append `MAX_TOKEN_SIZE - len(last token)` zero bytes so the decoder's
    /// fixed-width read from any token offset stays in bounds. Idempotent: a
    /// no-op once the padding is present or when the last token is already
    /// `MAX_TOKEN_SIZE` wide.
    pub fn pad_for_decoder(&mut self) {
        if self.offsets.len() < 2 {
            return;
        }
        let last_token_start = self.offsets[self.offsets.len() - 2] as usize;
        let required = last_token_start + MAX_TOKEN_SIZE;
        if self.bytes.len() < required {
            self.bytes.resize(required, 0);
        }
    }

    /// Materialize the [`WideDictionary`] form (see
    /// [`CompactDictionaryView::to_wide`]). Borrow as a view first with
    /// [`Dictionary::as_view`].
    #[inline]
    pub fn to_wide(&self) -> WideDictionary {
        self.as_view().to_wide()
    }
}

impl Dictionary for CompactDictionary {
    type View<'a> = CompactDictionaryView<'a>;
    #[inline]
    fn as_view(&self) -> CompactDictionaryView<'_> {
        CompactDictionaryView { bytes: &self.bytes, offsets: &self.offsets }
    }
}

/// Borrowed, `Copy` view over a compact dictionary's buffers.
///
/// Borrows the raw slices rather than an owned [`CompactDictionary`], so a
/// consumer can build a view directly from buffers deserialized from storage.
/// The slices must satisfy the same invariants (see this module's documentation).
#[derive(Copy, Clone, Debug)]
pub struct CompactDictionaryView<'a> {
    /// Read-padded token bytes.
    pub bytes: &'a [u8],
    /// `num_tokens + 1` offsets into `bytes`.
    pub offsets: &'a [u32],
}

impl<'a> CompactDictionaryView<'a> {
    /// Number of tokens.
    #[inline]
    pub fn num_tokens(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Minimum bits per code needed to address this dictionary,
    /// `ceil(log2(num_tokens))`. See [`CompactDictionary::code_bits`].
    #[inline]
    pub fn code_bits(&self) -> u8 {
        code_bits_for(self.num_tokens())
    }

    /// Materialize the [`WideDictionary`] form.
    ///
    /// A conformant dictionary is read-padded, so the fixed 16-byte copy from
    /// each token offset is in bounds (only `lens[id]` bytes of each row are the
    /// token's own; the rest is overwritten by neighbours or padding and never
    /// read by decode). The copy is bounds-checked, so a non-padded (malformed)
    /// view panics rather than risking UB.
    pub fn to_wide(&self) -> WideDictionary {
        let n = self.num_tokens();
        let mut data = vec![0u8; n * MAX_TOKEN_SIZE];
        let mut lens = vec![0u8; n];
        for id in 0..n {
            let off = self.offsets[id] as usize;
            let len = self.offsets[id + 1] as usize - off;
            lens[id] = len as u8;
            let row = id * MAX_TOKEN_SIZE;
            // Read-padding ⇒ 16 bytes readable from every token offset; lowers
            // to a single SIMD move plus a (well-predicted) bounds check.
            data[row..row + MAX_TOKEN_SIZE].copy_from_slice(&self.bytes[off..off + MAX_TOKEN_SIZE]);
        }
        WideDictionary { data, lens }
    }
}

impl DictionaryView for CompactDictionaryView<'_> {
    #[inline]
    fn token(&self, id: Token) -> &[u8] {
        let begin = self.offsets[id as usize] as usize;
        let end = self.offsets[id as usize + 1] as usize;
        &self.bytes[begin..end]
    }

    #[inline]
    fn token_len(&self, id: Token) -> usize {
        (self.offsets[id as usize + 1] - self.offsets[id as usize]) as usize
    }

    #[inline]
    fn decoded_len(&self, codes: &[Token]) -> usize {
        codes
            .iter()
            .map(|&id| (self.offsets[id as usize + 1] - self.offsets[id as usize]) as usize)
            .sum()
    }

    #[inline]
    unsafe fn token_ptr(&self, id: Token) -> *const u8 {
        // SAFETY: id < num_tokens ⇒ offsets[id] is in bounds; the read-padding
        // invariant guarantees MAX_TOKEN_SIZE readable bytes at the offset.
        unsafe { self.bytes.as_ptr().add(*self.offsets.get_unchecked(id as usize) as usize) }
    }

    #[inline]
    unsafe fn token_len_unchecked(&self, id: Token) -> usize {
        // SAFETY: id < num_tokens ⇒ offsets[id] and offsets[id + 1] are in bounds.
        unsafe {
            (*self.offsets.get_unchecked(id as usize + 1) - *self.offsets.get_unchecked(id as usize)) as usize
        }
    }
}

impl<'a> From<&'a CompactDictionary> for CompactDictionaryView<'a> {
    #[inline]
    fn from(d: &'a CompactDictionary) -> Self {
        d.as_view()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dict(offsets: Vec<u32>, bytes: &[u8]) -> CompactDictionary {
        CompactDictionary { bytes: bytes.to_vec(), offsets }
    }

    /// A read-padded compact dictionary built from tokens.
    fn padded(tokens: &[&[u8]]) -> CompactDictionary {
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
    fn num_tokens_zero_when_offsets_empty() {
        assert_eq!(CompactDictionary::default().num_tokens(), 0);
    }

    #[test]
    fn num_tokens_is_offsets_len_minus_one() {
        assert_eq!(dict(vec![0, 3, 5, 8], b"").num_tokens(), 3);
    }

    #[test]
    fn token_returns_correct_slice() {
        let d = dict(vec![0, 1, 3, 6], b"abcdef");
        let v = d.as_view();
        assert_eq!(v.token(0), b"a");
        assert_eq!(v.token(1), b"bc");
        assert_eq!(v.token(2), b"def");
        assert_eq!(v.token_len(2), 3);
    }

    #[test]
    fn code_bits_is_ceil_log2_num_tokens() {
        assert_eq!(dict(vec![0; 257], b"").code_bits(), 8); // 256 tokens -> 8 bits
        assert_eq!(dict(vec![0; 258], b"").code_bits(), 9); // 257 tokens -> 9 bits
        assert_eq!(dict(vec![0; 513], b"").code_bits(), 9); // 512 tokens -> 9 bits
        assert_eq!(dict(vec![0; 514], b"").code_bits(), 10); // 513 tokens -> 10 bits
    }

    #[test]
    fn pad_for_decoder_extends_to_max_token_read() {
        // Last token "bc" is 2 bytes, so padding = MAX_TOKEN_SIZE - 2.
        let mut d = dict(vec![0, 1, 3], b"abc");
        d.pad_for_decoder();
        assert_eq!(d.logical_len(), 3);
        assert_eq!(d.bytes.len(), 1 + MAX_TOKEN_SIZE); // offset(last)=1, +16
    }

    #[test]
    fn pad_for_decoder_is_idempotent() {
        let mut d = dict(vec![0, 1, 3], b"abc");
        d.pad_for_decoder();
        let len = d.bytes.len();
        d.pad_for_decoder();
        assert_eq!(d.bytes.len(), len);
    }

    #[test]
    fn pad_for_decoder_tops_up_insufficient_trailing_bytes() {
        // bytes already exceed logical_len (3) but lack room for a full
        // MAX_TOKEN_SIZE read from the last token's start (offset 1).
        let mut d = dict(vec![0, 1, 3], &[b'a', b'b', b'c', 0]);
        d.pad_for_decoder();
        assert_eq!(d.bytes.len(), 1 + MAX_TOKEN_SIZE);
    }

    #[test]
    fn pad_for_decoder_noop_for_full_width_last_token() {
        let mut d = dict(vec![0, MAX_TOKEN_SIZE as u32], &[b'z'; MAX_TOKEN_SIZE]);
        d.pad_for_decoder();
        assert_eq!(d.bytes.len(), MAX_TOKEN_SIZE);
    }

    #[test]
    fn to_wide_rows_and_lens_match_tokens() {
        let tokens: &[&[u8]] = &[b"a", b"bc", b"def", b"ghij"];
        let d = padded(tokens);
        let wide = d.to_wide();
        assert_eq!(wide.num_tokens(), tokens.len());
        for (id, tok) in tokens.iter().enumerate() {
            assert_eq!(wide.lens[id] as usize, tok.len());
            assert_eq!(&wide.data[id * MAX_TOKEN_SIZE..id * MAX_TOKEN_SIZE + tok.len()], *tok);
        }
    }

    #[test]
    fn to_wide_then_to_compact_round_trips_logical_content() {
        let tokens: &[&[u8]] = &[b"a", b"bc", b"def", b"ghij", &[b'z'; MAX_TOKEN_SIZE]];
        let d = padded(tokens);
        let back = d.to_wide().to_compact();
        assert_eq!(back.offsets, d.offsets);
        assert_eq!(&back.bytes[..back.logical_len()], &d.bytes[..d.logical_len()]);
    }
}
