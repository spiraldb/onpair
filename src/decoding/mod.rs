// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Token decode.
//!
//! Decoding is a gather-copy: each code names a dictionary token; the output is
//! those tokens concatenated. The kernels are generic over the dictionary
//! representation ([`DictionaryView`]):
//!
//! * [`CompactDictionaryView`](crate::core::dictionary::CompactDictionaryView) —
//!   a `code → offset → bytes` lookup (two dependent loads). Best for random
//!   access / one-shot; no precomputation.
//! * [`WideDictionaryView`](crate::core::dictionary::WideDictionaryView) —
//!   a load-free `data + code*MAX_TOKEN_SIZE` address. Built once (via
//!   `to_wide`), then best for bulk decode or repeated random access.
//!
//! To decode one row pass that row's code slice
//! ([`ColumnView::row_codes`](crate::ColumnView::row_codes)); to bulk-decode
//! pass the whole stream.

use std::mem::MaybeUninit;

use crate::core::dictionary::DictionaryView;
use crate::core::types::{MAX_TOKEN_SIZE, Token};

mod copy;

/// Exact decoded byte length of `codes` against `dict` (the sum of token
/// lengths).
#[inline]
pub fn decoded_len<V: DictionaryView>(codes: &[Token], dict: V) -> usize {
    dict.decoded_len(codes)
}

/// Decode `codes` against `dict` into `out`, returning the number of bytes
/// written (`== decoded_len(codes, dict)`).
///
/// All but the final [`MAX_TOKEN_SIZE`] tokens are written with a fixed 16-byte
/// over-store (the fast path); the tail tokens are copied at their true length,
/// so the last byte written lands exactly at `decoded_len`. This lets `out` be
/// sized to the exact decoded length, with no trailing slack.
///
/// # Safety
/// The caller must guarantee all of:
/// - every `code` is `< dict.num_tokens()`;
/// - `dict` upholds [`DictionaryView::token_ptr`]'s 16-byte-readable contract
///   (for a compact view: read-padded);
/// - `out.len() >= decoded_len(codes, dict)`.
pub unsafe fn decode_into<V: DictionaryView>(codes: &[Token], dict: V, out: &mut [MaybeUninit<u8>]) -> usize {
    let dst = out.as_mut_ptr().cast::<u8>();
    let mut w = 0usize;
    // The fast path over-stores [`MAX_TOKEN_SIZE`] bytes per token; that stays in
    // bounds only while at least `MAX_TOKEN_SIZE` tokens still follow, since they
    // span >= `MAX_TOKEN_SIZE` bytes and absorb the over-store. The final
    // <= `MAX_TOKEN_SIZE` tokens are copied at their true length instead.
    let (fast, tail) = codes.split_at(codes.len().saturating_sub(MAX_TOKEN_SIZE));
    for &code in fast {
        // SAFETY: code in range ⇒ token_ptr/token_len_unchecked are valid and the
        // pointer is readable for 16 bytes; >= MAX_TOKEN_SIZE tokens remain ⇒
        // w + 16 <= decoded_len <= out.len(). `len` is read right after `ptr` so
        // the compact path folds its shared `offsets[code]` load; the wide copy
        // does not depend on `len`, so it is not serialized behind that load.
        unsafe {
            let src = dict.token_ptr(code);
            let len = dict.token_len_unchecked(code);
            copy::copy16(src, dst.add(w));
            w += len;
        }
    }
    for &code in tail {
        // SAFETY: code in range; the exact-length copy reads `len <= MAX_TOKEN_SIZE`
        // bytes and writes within `[w, decoded_len) ⊆ out`.
        unsafe {
            let src = dict.token_ptr(code);
            let len = dict.token_len_unchecked(code);
            copy::copy_token_bytes(src, dst.add(w), len);
            w += len;
        }
    }
    w
}

/// Decode `codes` against `dict` into a freshly allocated `Vec`, sized exactly.
///
/// Safe convenience: it assumes `dict` and `codes` uphold their invariants
/// (true for any [`ColumnView`](crate::ColumnView) obtained from a
/// [`Column`](crate::Column)). Feeding it a hand-built, malformed dictionary or
/// out-of-range code is undefined behavior.
pub fn decode_to_vec<V: DictionaryView>(codes: &[Token], dict: V) -> Vec<u8> {
    let n = dict.decoded_len(codes);
    let mut out = Vec::with_capacity(n);
    // SAFETY: buffer sized to the exact decoded length; trust assumption as above.
    let w = unsafe { decode_into(codes, dict, out.spare_capacity_mut()) };
    // SAFETY: the kernel initialized exactly `w` leading bytes.
    unsafe { out.set_len(w) };
    out
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
        codes.iter().flat_map(|&c| tokens[c as usize].iter().copied()).collect()
    }

    /// Decode through both representations and compare. Cheap enough to run under
    /// Miri, which then proves the `unsafe` over-copy has no UB.
    fn check(tokens: &[&[u8]], codes: &[Token]) {
        let (bytes, offsets) = padded(tokens);
        let view = CompactDictionaryView { bytes: &bytes, offsets: &offsets };
        let want = expected(tokens, codes);

        assert_eq!(decoded_len(codes, view), want.len());
        assert_eq!(decode_to_vec(codes, view), want, "compact");

        let wide = view.to_wide();
        assert_eq!(decoded_len(codes, wide.as_view()), want.len());
        assert_eq!(decode_to_vec(codes, wide.as_view()), want, "wide");
    }

    #[test]
    fn decodes_mixed_length_tokens() {
        let tokens: &[&[u8]] = &[b"a", b"bc", b"def", b"ghij"];
        // 40 codes (> MAX_TOKEN_SIZE) so the over-copy region is exercised.
        let codes: Vec<Token> = (0..40).map(|i| (i % 4) as Token).collect();
        check(tokens, &codes);
    }

    #[test]
    fn decodes_full_width_last_token() {
        // A full-width last token: the wide read ends exactly at the logical end.
        let full = vec![b'z'; MAX_TOKEN_SIZE];
        let tokens: &[&[u8]] = &[b"x", &full];
        let codes: Vec<Token> = (0..40).map(|i| (i % 2) as Token).collect();
        check(tokens, &codes);
    }

    #[test]
    fn decodes_short_final_token() {
        // > MAX_TOKEN_SIZE codes ending in a 1-byte token: the fast/exact split
        // must route the short tail through the exact copy so nothing over-stores
        // past the exact-sized buffer (Miri checks the bound).
        let tokens: &[&[u8]] = &[b"a", b"bcde"];
        let mut codes: Vec<Token> = (0..40).map(|i| (i % 2) as Token).collect();
        *codes.last_mut().unwrap() = 0; // final token is the 1-byte "a"
        check(tokens, &codes);
    }

    #[test]
    fn decodes_all_tail_length_buckets() {
        // Tokens spanning every `copy_token_bytes` size bucket (1, 2|3, 4..=7,
        // 8..=15, 16); > MAX_TOKEN_SIZE codes cycle through them so each bucket
        // lands in the exact-copy tail. Miri then vets the overlapping writes.
        let t = [vec![b'a'; 1], vec![b'b'; 3], vec![b'c'; 5], vec![b'd'; 11], vec![b'e'; 15], vec![b'f'; 16]];
        let tokens: Vec<&[u8]> = t.iter().map(Vec::as_slice).collect();
        let codes: Vec<Token> = (0..40).map(|i| (i % t.len()) as Token).collect();
        check(&tokens, &codes);
    }

    #[test]
    fn decodes_empty_code_stream() {
        let tokens: &[&[u8]] = &[b"a", b"b"];
        check(tokens, &[]);
    }

    #[test]
    fn single_token_decode() {
        let tokens: &[&[u8]] = &[b"hello", b"world"];
        check(tokens, &[0]);
        check(tokens, &[1]);
    }
}
