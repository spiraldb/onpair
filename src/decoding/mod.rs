// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Token decode.
//!
//! Decoding is a gather-copy: each code names a dictionary token; the output is
//! those tokens concatenated. Two kernels share the same vocabulary — both take
//! a `&[Token]` and decode it into a caller buffer:
//!
//! * [`decode_into`] — direct; reads the dictionary offsets per code (two
//!   dependent loads). No precomputation; best for random access / one-shot.
//! * [`FatTable`] — materializes a `num_tokens × MAX_TOKEN_SIZE` byte-strided
//!   copy of the dictionary once, after which a decode is a single
//!   `table[code]` load. Best for bulk decode or repeated random access.
//!
//! To decode one row pass that row's code slice
//! ([`ColumnView::row_codes`](crate::ColumnView::row_codes)); to bulk-decode
//! pass the whole stream.

use std::mem::MaybeUninit;

use crate::core::dictionary::DictionaryView;
use crate::core::types::{MAX_TOKEN_SIZE, Token};

mod fat;
mod scalar;

pub use fat::FatTable;

/// Exact decoded byte length of `codes` against `dict` (the sum of token
/// lengths).
#[inline]
pub fn decoded_len(codes: &[Token], dict: DictionaryView<'_>) -> usize {
    codes.iter().map(|&c| dict.token_len(c)).sum()
}

/// Decode `codes` against `dict` into `out`, returning the number of bytes
/// written (`== decoded_len(codes, dict)`). Direct path; no table build.
///
/// All but the final [`MAX_TOKEN_SIZE`] tokens are written with a fixed 16-byte
/// over-store (the fast path); the tail tokens are copied at their true length,
/// so the last byte written lands exactly at `decoded_len`. This lets `out` be
/// sized to the exact decoded length, with no trailing slack.
///
/// # Safety
/// The caller must guarantee all of:
/// - every `code` is `< dict.num_tokens()`;
/// - `dict` is read-padded (the fast path reads [`MAX_TOKEN_SIZE`] bytes per
///   token);
/// - `out.len() >= decoded_len(codes, dict)`.
pub unsafe fn decode_into(codes: &[Token], dict: DictionaryView<'_>, out: &mut [MaybeUninit<u8>]) -> usize {
    let bytes = dict.bytes.as_ptr();
    let offsets = dict.offsets.as_ptr();
    let dst = out.as_mut_ptr().cast::<u8>();
    let mut w = 0usize;
    // The fast path over-stores [`MAX_TOKEN_SIZE`] bytes per token; that stays in
    // bounds only while at least `MAX_TOKEN_SIZE` tokens still follow, since they
    // span >= `MAX_TOKEN_SIZE` bytes and absorb the over-store. The final
    // <= `MAX_TOKEN_SIZE` tokens are copied at their true length instead.
    let (fast, tail) = codes.split_at(codes.len().saturating_sub(MAX_TOKEN_SIZE));
    for &code in fast {
        let c = code as usize;
        // SAFETY: c < num_tokens ⇒ offsets[c] and offsets[c + 1] are in bounds
        // and the token starts within `bytes`; read-padding ⇒ 16 readable bytes
        // there; >= MAX_TOKEN_SIZE tokens remain ⇒ w + 16 <= decoded_len <= out.len().
        unsafe {
            let off = *offsets.add(c) as usize;
            let len = (*offsets.add(c + 1) as usize) - off;
            scalar::copy16(bytes.add(off), dst.add(w));
            w += len;
        }
    }
    for &code in tail {
        let c = code as usize;
        // SAFETY: c < num_tokens ⇒ offsets[c] and offsets[c + 1] are in bounds;
        // the exact-length copy reads `len <= MAX_TOKEN_SIZE` bytes (no read-padding
        // needed) and writes within `[w, decoded_len) ⊆ out`.
        unsafe {
            let off = *offsets.add(c) as usize;
            let len = (*offsets.add(c + 1) as usize) - off;
            scalar::copy_token_bytes(bytes.add(off), dst.add(w), len);
            w += len;
        }
    }
    w
}

/// Decode `codes` against `dict` into a freshly allocated `Vec` (direct path),
/// sized exactly.
///
/// Safe convenience: it assumes `dict` and `codes` uphold their invariants
/// (true for any [`ColumnView`](crate::ColumnView) obtained from a
/// [`Column`](crate::Column)). Feeding it a hand-built, malformed dictionary or
/// out-of-range code is undefined behavior.
pub fn decode_to_vec(codes: &[Token], dict: DictionaryView<'_>) -> Vec<u8> {
    let n = decoded_len(codes, dict);
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

    /// Build a read-padded dictionary `(bytes, offsets)` from tokens.
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

    /// Decode through both kernels and compare. Cheap enough to run under Miri,
    /// which then proves the `unsafe` over-copy has no UB.
    fn check(tokens: &[&[u8]], codes: &[Token]) {
        let (bytes, offsets) = padded(tokens);
        let dict = DictionaryView { bytes: &bytes, offsets: &offsets };
        let want = expected(tokens, codes);

        assert_eq!(decoded_len(codes, dict), want.len());
        assert_eq!(decode_to_vec(codes, dict), want, "direct kernel");

        // SAFETY: padded dict, in-range codes.
        let table = unsafe { FatTable::build(dict) };
        let mut out = Vec::<u8>::with_capacity(want.len());
        // SAFETY: buffer sized to the exact decoded length; codes in range.
        let w = unsafe { table.decode_into(codes, out.spare_capacity_mut()) };
        unsafe { out.set_len(w) };
        assert_eq!(out, want, "fat table");

        assert_eq!(table.decoded_len(codes), want.len());
        assert_eq!(table.decode_to_vec(codes), want, "fat table to_vec");
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
