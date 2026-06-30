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
//!   highest token offset (applied by [`pad_raw`] at construction).
//!   `offsets.last()` is the logical length; `bytes.len()` may exceed it by the
//!   padding.

use super::{Dictionary, DictionaryView, WideDictionary};
use crate::core::types::{MAX_TOKEN_SIZE, Token, code_bits_for};

/// Append `MAX_TOKEN_SIZE - len(last token)` zero bytes to `bytes` so the decoder's
/// fixed-width read from any token offset stays in bounds — the read-padding
/// invariant. Applied once, on the raw buffers, just before sealing a
/// [`CompactDictionary`]. Idempotent: a no-op once the padding is present or when
/// the last token is already `MAX_TOKEN_SIZE` wide.
pub(crate) fn pad_raw(bytes: &mut Vec<u8>, offsets: &[u32]) {
    if offsets.len() < 2 {
        return;
    }
    let last_token_start = offsets[offsets.len() - 2] as usize;
    let required = last_token_start + MAX_TOKEN_SIZE;
    if bytes.len() < required {
        bytes.resize(required, 0);
    }
}

/// Owned compact dictionary — **trusted**: holding one is a proof that its
/// buffers satisfy the invariants in this module's documentation.
///
/// The fields are private, so a value can only be obtained through a door that
/// establishes that proof: the trainer, or
/// [`UntrustedDictionary::validate`](crate::UntrustedDictionary::validate) /
/// [`trust_unchecked`](crate::UntrustedDictionary::trust_unchecked) applied to
/// deserialized buffers. Read the buffers back with [`bytes`](Self::bytes) /
/// [`offsets`](Self::offsets) (e.g. to serialize).
#[derive(Default, Debug, Clone)]
pub struct CompactDictionary {
    /// Concatenated token bytes, followed by read-padding.
    bytes: Vec<u8>,
    /// `num_tokens + 1` offsets delimiting the tokens within `bytes`.
    offsets: Vec<u32>,
}

impl CompactDictionary {
    /// Number of tokens.
    #[inline]
    pub fn num_tokens(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// The token bytes, including trailing read-padding (the serialized
    /// `dict_bytes`; see `docs/interchange-format.md`).
    #[inline]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The `num_tokens + 1` token offsets (the serialized `dict_offsets`).
    #[inline]
    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    /// Seal raw buffers into a trusted dictionary. The crate-internal trust mint:
    /// the caller (trainer, or the [`validate`](crate::UntrustedDictionary::validate)
    /// door) guarantees the invariants.
    #[inline]
    pub(crate) fn from_raw(bytes: Vec<u8>, offsets: Vec<u32>) -> Self {
        Self { bytes, offsets }
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
        CompactDictionaryView {
            bytes: &self.bytes,
            offsets: &self.offsets,
        }
    }
}

/// Borrowed, `Copy`, **trusted** view over a compact dictionary's buffers.
///
/// Like [`CompactDictionary`] its fields are private: a value can only be obtained
/// from a trusted owned dictionary ([`Dictionary::as_view`]) or by validating an
/// [`UntrustedDictionaryView`](crate::UntrustedDictionaryView).
#[derive(Copy, Clone, Debug)]
pub struct CompactDictionaryView<'a> {
    /// Read-padded token bytes.
    bytes: &'a [u8],
    /// `num_tokens + 1` offsets into `bytes`.
    offsets: &'a [u32],
}

impl<'a> CompactDictionaryView<'a> {
    /// Seal raw borrowed buffers into a trusted view (crate-internal trust mint;
    /// the caller guarantees the invariants).
    #[inline]
    pub(crate) fn from_raw(bytes: &'a [u8], offsets: &'a [u32]) -> Self {
        Self { bytes, offsets }
    }

    /// Minimum bits per code needed to address this dictionary,
    /// `ceil(log2(num_tokens))`. See [`CompactDictionary::code_bits`].
    #[inline]
    pub fn code_bits(&self) -> u8 {
        code_bits_for(self.num_tokens())
    }

    /// Materialize the [`WideDictionary`] form: every token laid out in its own
    /// fixed [`MAX_TOKEN_SIZE`]-byte row, so a decode reaches a token at
    /// `code * MAX_TOKEN_SIZE` with no `code → offset → bytes` indirection. Worth
    /// building once to amortize over a bulk or repeated decode; see
    /// [`WideDictionary`] for the space/speed trade-off.
    ///
    /// The source is a **trusted** [`CompactDictionaryView`], so this never
    /// validates and never fails — the wide form is valid by construction. Two
    /// trusted invariants carry the build:
    ///
    /// * read-padding lets each row be filled with one fixed 16-byte copy from the
    ///   token's offset — an over-read past the token into neighbouring or padding
    ///   bytes (harmless: decode only ever reads a row's first `lens[id]` bytes),
    ///   kept in bounds by the padding;
    /// * the `≤ MAX_TOKEN_SIZE` length bound makes `lens[id] = len as u8` exact,
    ///   not a silent truncation.
    ///
    /// `O(num_tokens)`, dominated by the row copy.
    pub fn to_wide(&self) -> WideDictionary {
        let n = self.num_tokens();
        let mut data = vec![0u8; n * MAX_TOKEN_SIZE];
        let mut lens = vec![0u8; n];
        let src = self.bytes.as_ptr();
        let dst = data.as_mut_ptr();
        for id in 0..n {
            // SAFETY: a trusted view has `offsets.len() == n + 1`, so `id` and
            // `id + 1` index it in bounds. Offsets are strictly increasing with
            // every token length in `1..=MAX_TOKEN_SIZE`, so `end - off` neither
            // wraps nor overflows the `u8` length.
            let (off, end) = unsafe {
                (
                    *self.offsets.get_unchecked(id) as usize,
                    *self.offsets.get_unchecked(id + 1) as usize,
                )
            };
            // SAFETY: `id < n == lens.len()`.
            unsafe { *lens.get_unchecked_mut(id) = (end - off) as u8 };
            // SAFETY: dst — `(id + 1) * MAX_TOKEN_SIZE <= n * MAX_TOKEN_SIZE == data.len()`.
            // src — the read-padding invariant guarantees `MAX_TOKEN_SIZE` readable
            // bytes at `off`; `src` (borrowed dictionary) and the freshly-allocated
            // `dst` are distinct allocations, so the copy cannot overlap.
            unsafe {
                std::ptr::copy_nonoverlapping(src.add(off), dst.add(id * MAX_TOKEN_SIZE), MAX_TOKEN_SIZE);
            }
        }
        WideDictionary::from_raw(data, lens)
    }
}

impl DictionaryView for CompactDictionaryView<'_> {
    #[inline]
    fn num_tokens(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

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
    unsafe fn token_ptr(&self, id: Token) -> *const u8 {
        // SAFETY: id < num_tokens ⇒ offsets[id] is in bounds; the read-padding
        // invariant guarantees MAX_TOKEN_SIZE readable bytes at the offset.
        unsafe {
            self.bytes
                .as_ptr()
                .add(*self.offsets.get_unchecked(id as usize) as usize)
        }
    }

    #[inline]
    unsafe fn token_len_unchecked(&self, id: Token) -> usize {
        // SAFETY: id < num_tokens ⇒ offsets[id] and offsets[id + 1] are in bounds.
        unsafe {
            (*self.offsets.get_unchecked(id as usize + 1)
                - *self.offsets.get_unchecked(id as usize)) as usize
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
        CompactDictionary::from_raw(bytes.to_vec(), offsets)
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
    fn pad_raw_extends_to_max_token_read() {
        // Last token "bc" is 2 bytes; padding fills to offset(last) + MAX_TOKEN_SIZE.
        let mut bytes = b"abc".to_vec();
        pad_raw(&mut bytes, &[0, 1, 3]);
        assert_eq!(bytes.len(), 1 + MAX_TOKEN_SIZE); // offset(last)=1, +16
    }

    #[test]
    fn pad_raw_is_idempotent() {
        let mut bytes = b"abc".to_vec();
        let offsets = [0u32, 1, 3];
        pad_raw(&mut bytes, &offsets);
        let len = bytes.len();
        pad_raw(&mut bytes, &offsets);
        assert_eq!(bytes.len(), len);
    }

    #[test]
    fn pad_raw_tops_up_insufficient_trailing_bytes() {
        // bytes already exceed logical_len (3) but lack room for a full
        // MAX_TOKEN_SIZE read from the last token's start (offset 1).
        let mut bytes = vec![b'a', b'b', b'c', 0];
        pad_raw(&mut bytes, &[0, 1, 3]);
        assert_eq!(bytes.len(), 1 + MAX_TOKEN_SIZE);
    }

    #[test]
    fn pad_raw_noop_for_full_width_last_token() {
        let mut bytes = vec![b'z'; MAX_TOKEN_SIZE];
        pad_raw(&mut bytes, &[0, MAX_TOKEN_SIZE as u32]);
        assert_eq!(bytes.len(), MAX_TOKEN_SIZE);
    }
}
