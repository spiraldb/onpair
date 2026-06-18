// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The decode-side dictionary: the token table a code stream indexes into.
//!
//! Layout is Arrow binary — a flat `bytes` buffer plus an `offsets` index of
//! length `num_tokens + 1`; token `i` is `bytes[offsets[i]..offsets[i + 1]]`.
//!
//! # Invariants
//! Upheld by the trainer; a precondition of every accessor and of decoding.
//! - `offsets[0] == 0` and `offsets.len() == num_tokens + 1`.
//! - **Strictly increasing** offsets — every token is non-empty, with length in
//!   `1..=MAX_TOKEN_SIZE`.
//! - **Sorted** — tokens are in strictly ascending bytewise-lexicographic
//!   order.
//! - **Complete** — all 256 single-byte tokens are present, so any byte string
//!   is encodable.
//! - **Unique** — no two tokens are equal.
//! - **Read-padded** — `bytes` is readable for [`MAX_TOKEN_SIZE`] bytes past the
//!   highest token offset (call [`Dictionary::pad_for_decoder`] once after
//!   filling). `offsets.last()` is the logical length; `bytes.len()` may exceed
//!   it by the padding.

use crate::core::types::{MAX_TOKEN_SIZE, Token};

/// Minimum bits per code needed to address `num_tokens` distinct tokens:
/// `ceil(log2(num_tokens))`.
#[inline]
fn code_bits_for(num_tokens: usize) -> u32 {
    debug_assert!(num_tokens >= 1, "log2(0) is undefined; num_tokens must be >= 1");
    if num_tokens <= 1 {
        1
    } else {
        (num_tokens as u32 - 1).ilog2() + 1
    }
}

/// Owned decode-side dictionary.
///
/// Fields are public — this is a data type a consumer may construct directly
/// from buffers it deserialized from storage — and must satisfy the invariants
/// described in this module's documentation.
#[derive(Default, Debug, Clone)]
pub struct Dictionary {
    /// Concatenated token bytes, followed by read-padding.
    pub bytes: Vec<u8>,
    /// `num_tokens + 1` offsets delimiting the tokens within `bytes`.
    pub offsets: Vec<u32>,
}

impl Dictionary {
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
    pub fn code_bits(&self) -> u32 {
        code_bits_for(self.num_tokens())
    }

    /// Bytes of token `id`. Precondition: `id < num_tokens()`.
    #[inline]
    pub fn token(&self, id: Token) -> &[u8] {
        let begin = self.offsets[id as usize] as usize;
        let end = self.offsets[id as usize + 1] as usize;
        &self.bytes[begin..end]
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

    /// Borrow as a [`DictionaryView`].
    #[inline]
    pub fn as_view(&self) -> DictionaryView<'_> {
        DictionaryView { bytes: &self.bytes, offsets: &self.offsets }
    }
}

/// Borrowed, `Copy` view over a dictionary's buffers.
///
/// Borrows the raw slices rather than an owned [`Dictionary`], so a consumer can
/// build a view directly from buffers deserialized from storage. The slices
/// must satisfy the same invariants (see this module's documentation).
#[derive(Copy, Clone, Debug)]
pub struct DictionaryView<'a> {
    /// Read-padded token bytes.
    pub bytes: &'a [u8],
    /// `num_tokens + 1` offsets into `bytes`.
    pub offsets: &'a [u32],
}

impl<'a> DictionaryView<'a> {
    /// Number of tokens.
    #[inline]
    pub fn num_tokens(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Byte length of token `id`. Precondition: `id < num_tokens()`.
    #[inline]
    pub fn token_len(&self, id: Token) -> usize {
        (self.offsets[id as usize + 1] - self.offsets[id as usize]) as usize
    }

    /// Bytes of token `id`. Precondition: `id < num_tokens()`.
    #[inline]
    pub fn token(&self, id: Token) -> &'a [u8] {
        let begin = self.offsets[id as usize] as usize;
        let end = self.offsets[id as usize + 1] as usize;
        &self.bytes[begin..end]
    }

    /// Minimum bits per code needed to address this dictionary,
    /// `ceil(log2(num_tokens))`. See [`Dictionary::code_bits`].
    #[inline]
    pub fn code_bits(&self) -> u32 {
        code_bits_for(self.num_tokens())
    }
}

impl<'a> From<&'a Dictionary> for DictionaryView<'a> {
    #[inline]
    fn from(d: &'a Dictionary) -> Self {
        d.as_view()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dict(offsets: Vec<u32>, bytes: &[u8]) -> Dictionary {
        Dictionary { bytes: bytes.to_vec(), offsets }
    }

    #[test]
    fn num_tokens_zero_when_offsets_empty() {
        assert_eq!(Dictionary::default().num_tokens(), 0);
    }

    #[test]
    fn num_tokens_is_offsets_len_minus_one() {
        assert_eq!(dict(vec![0, 3, 5, 8], b"").num_tokens(), 3);
    }

    #[test]
    fn token_returns_correct_slice() {
        let d = dict(vec![0, 1, 3, 6], b"abcdef");
        assert_eq!(d.token(0), b"a");
        assert_eq!(d.token(1), b"bc");
        assert_eq!(d.token(2), b"def");
        assert_eq!(d.as_view().token(2), b"def");
        assert_eq!(d.as_view().token_len(2), 3);
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
}
