// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Token decode.
//!
//! Decoding is a gather-copy: each code names a dictionary token; the output is
//! those tokens concatenated. [`decode_into`] copies a fixed 16 bytes per token
//! (one `copy16` store) whatever the token's true length — no per-token length
//! branch. The cost is two paddings to keep that over-read in bounds: the
//! dictionary's read-padding backs the source over-read, and the output buffer's
//! [`DECODE_PADDING`] backs the destination over-store. It is generic over the
//! dictionary representation ([`DictionaryView`]) — a load-free
//! [`WideDictionaryView`](crate::core::dictionary::WideDictionaryView) or a
//! [`CompactDictionaryView`](crate::core::dictionary::CompactDictionaryView).
//! Codes are bounds-checked in the loop; [`ColumnView`](crate::ColumnView) wraps
//! it as the buffer-sized [`decompress_into`](crate::ColumnView::decompress_into).

use std::mem::MaybeUninit;

use crate::core::dictionary::DictionaryView;
use crate::core::types::{MAX_TOKEN_SIZE, Token};
use crate::core::validate::{InvalidColumn, panic_malformed};

mod copy;

/// Trailing bytes a [`decode_into`] output buffer needs **beyond** the decoded
/// length. The decoder over-stores a fixed [`MAX_TOKEN_SIZE`]-byte chunk for the
/// final token — up to `MAX_TOKEN_SIZE - 1` bytes past the logical end — so an
/// output buffer is sized as `decoded_len + DECODE_PADDING`.
pub const DECODE_PADDING: usize = MAX_TOKEN_SIZE;

/// Exact decoded byte length of `codes` against `dict` (the sum of token
/// lengths) — sizes a decode buffer.
///
/// Bounds-checks each code, so a malformed code stream panics with
/// [`InvalidColumn::CodeOutOfRange`] rather than reading out of bounds. The sum
/// is only meaningful for a structurally valid dictionary.
#[inline]
pub fn decoded_len<V: DictionaryView>(codes: &[Token], dict: V) -> usize {
    let n = dict.num_tokens();
    let mut sum = 0usize;
    for &c in codes {
        if (c as usize) >= n {
            panic_malformed(InvalidColumn::CodeOutOfRange);
        }
        // SAFETY: c < num_tokens.
        let len = unsafe { dict.token_len_unchecked(c) };
        sum = sum
            .checked_add(len)
            .unwrap_or_else(|| panic_malformed(InvalidColumn::DecodedLenOverflow));
    }
    sum
}

/// Decode `codes` against an **already-validated** `dict` into `out`, returning
/// the bytes written. It over-reads a fixed 16 bytes per token, trusting the
/// dictionary's offsets/lengths (validated up front). Each code is bounds-checked
/// in the loop (a near-free, predicted-not-taken branch); an out-of-range code
/// panics with [`InvalidColumn::CodeOutOfRange`].
///
/// `dict`'s validity is a *type* invariant — only a trusted [`DictionaryView`]
/// (sealed; obtained through `validate`/`to_wide`) can be passed — so it is not a
/// precondition here.
///
/// # Safety
/// `out.len() >= decoded_len(codes, dict) + DECODE_PADDING`.
pub unsafe fn decode_into<V: DictionaryView>(
    codes: &[Token],
    dict: V,
    out: &mut [MaybeUninit<u8>],
) -> usize {
    let ntok = dict.num_tokens();
    let dst = out.as_mut_ptr().cast::<u8>();
    let mut w = 0usize;
    for &code in codes {
        if code as usize >= ntok {
            panic_malformed(InvalidColumn::CodeOutOfRange);
        }
        // SAFETY: code in range ⇒ token_ptr is readable for 16 bytes (the dict is
        // read-padded); the trailing DECODE_PADDING keeps the last token's
        // over-store within `out`.
        unsafe {
            let src = dict.token_ptr(code);
            let len = dict.token_len_unchecked(code);
            copy::copy16(src, dst.add(w));
            w += len;
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dictionary::{CompactDictionaryView, Dictionary};

    /// Build read-padded compact `(bytes, offsets)` from tokens.
    fn padded(tokens: &[&[u8]]) -> (Vec<u8>, Vec<u32>) {
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for t in tokens {
            bytes.extend_from_slice(t);
            offsets.push(bytes.len() as u32);
        }
        bytes.resize(bytes.len() + MAX_TOKEN_SIZE, 0); // worst-case read padding
        (bytes, offsets)
    }

    fn expected(tokens: &[&[u8]], codes: &[Token]) -> Vec<u8> {
        codes
            .iter()
            .flat_map(|&c| tokens[c as usize].iter().copied())
            .collect()
    }

    /// Decode into a padded buffer and return the initialized bytes.
    fn vec_decode<V: DictionaryView>(codes: &[Token], dict: V) -> Vec<u8> {
        let n = decoded_len(codes, dict);
        let mut out = Vec::with_capacity(n + DECODE_PADDING);
        // SAFETY: read-padded dict, in-range codes, buffer + DECODE_PADDING.
        let w = unsafe { decode_into(codes, dict, out.spare_capacity_mut()) };
        unsafe { out.set_len(w) };
        out
    }

    /// Decode through both representations and compare. Cheap enough to run under
    /// Miri, which then proves the over-copy has no UB.
    fn check(tokens: &[&[u8]], codes: &[Token]) {
        let (bytes, offsets) = padded(tokens);
        let view = CompactDictionaryView::from_raw(&bytes, &offsets);
        let want = expected(tokens, codes);

        assert_eq!(decoded_len(codes, view), want.len());
        assert_eq!(vec_decode(codes, view), want, "compact");

        let wide = view.to_wide();
        assert_eq!(vec_decode(codes, wide.as_view()), want, "wide");
    }

    #[test]
    fn decodes_mixed_length_tokens() {
        let tokens: &[&[u8]] = &[b"a", b"bc", b"def", b"ghij"];
        let codes: Vec<Token> = (0..40).map(|i| (i % 4) as Token).collect();
        check(tokens, &codes);
    }

    #[test]
    fn decodes_full_width_last_token() {
        let full = vec![b'z'; MAX_TOKEN_SIZE];
        let tokens: &[&[u8]] = &[b"x", &full];
        let codes: Vec<Token> = (0..40).map(|i| (i % 2) as Token).collect();
        check(tokens, &codes);
    }

    #[test]
    fn decodes_short_final_token() {
        // Ends in a 1-byte token: the last over-store must land in the
        // DECODE_PADDING room. Miri checks the bound.
        let tokens: &[&[u8]] = &[b"a", b"bcde"];
        let mut codes: Vec<Token> = (0..40).map(|i| (i % 2) as Token).collect();
        *codes.last_mut().unwrap() = 0;
        check(tokens, &codes);
    }

    #[test]
    fn decodes_all_tail_length_buckets() {
        let t = [
            vec![b'a'; 1],
            vec![b'b'; 3],
            vec![b'c'; 5],
            vec![b'd'; 11],
            vec![b'e'; 15],
            vec![b'f'; 16],
        ];
        let tokens: Vec<&[u8]> = t.iter().map(Vec::as_slice).collect();
        let codes: Vec<Token> = (0..40).map(|i| (i % t.len()) as Token).collect();
        check(&tokens, &codes);
    }

    #[test]
    fn decodes_empty_code_stream() {
        check(&[b"a", b"b"], &[]);
    }

    #[test]
    fn single_token_decode() {
        let tokens: &[&[u8]] = &[b"hello", b"world"];
        check(tokens, &[0]);
        check(tokens, &[1]);
    }

    #[test]
    #[should_panic(expected = "code index out of range")]
    fn decode_into_panics_on_out_of_range_code() {
        let (bytes, offsets) = padded(&[b"a", b"b"]);
        let view = CompactDictionaryView::from_raw(&bytes, &offsets);
        let mut out = vec![MaybeUninit::uninit(); 64];
        // SAFETY: read-padded dict + generous buffer; the bad code must panic.
        unsafe { decode_into(&[0, 5], view, &mut out) };
    }

    #[test]
    #[should_panic(expected = "code index out of range")]
    fn decoded_len_panics_on_out_of_range_code() {
        let (bytes, offsets) = padded(&[b"a", b"b"]);
        let view = CompactDictionaryView::from_raw(&bytes, &offsets);
        let _ = decoded_len(&[0, 5], view);
    }
}
