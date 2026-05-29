// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Token decoder.
//!
//! The decode hot path is one token copy per code: look the code's byte range
//! up in the dictionary and copy it to the output cursor. Tokens are capped at
//! [`MAX_TOKEN_SIZE`] bytes, which lets the padded fast paths over-copy a
//! fixed 16-byte chunk per token and advance the cursor by the token's true
//! length.
//!
//! Per-token copy is a single 16-byte value copy ([`scalar::copy16`]), which
//! lowers to one native 128-bit store on every target (`movups` on x86, `str q`
//! on AArch64). An exhaustive study (AVX-512 `vpcompressb` compaction, masked
//! stores, non-temporal/streaming stores, software prefetch, 4-bit nibble length
//! arrays, 32-byte rows, length co-located with the token) found **none** beat
//! this scalar copy — they tied or lost — so there is a single scalar backend
//! and no architecture-specific SIMD modules.
//!
//! Tokens are materialized into a 16-byte-strided "fat" table ([`fat`]) so a
//! decode load addresses `data + code*16` directly; the copy is always scalar.
//! Both the table build and the decode read a fixed [`MAX_TOKEN_SIZE`] bytes
//! from each token offset, so the dictionary must extend `MAX_TOKEN_SIZE` past
//! its highest offset (a format invariant; see [`MAX_TOKEN_SIZE`] and
//! [`Parts::validate_dictionary`]).
//!
//! (A half-size "entries" layout for cache-constrained hosts was removed; see
//! the `entries-layout` TODO below and commit e036a4c.)

use std::mem::MaybeUninit;

use crate::column::Parts;
use crate::offset::Offset;
use crate::types::MAX_TOKEN_SIZE;

mod scalar;

pub(crate) mod fat;

/// Return the exact decoded byte length of one row.
///
/// ## Panics
///
/// Panics if `row` is out of bounds or if `parts` violates the invariants
/// documented by the public API.
pub fn decompressed_row_len<O: Offset>(parts: Parts<'_, O>, row: usize) -> usize {
    let (begin, end) = row_code_range(parts, row);
    parts.codes[begin..end]
        .iter()
        .map(|&code| code_len(parts, code))
        .sum()
}

/// Return the exact decoded byte length of all rows in input order.
///
/// ## Panics
///
/// Panics if `parts` violates the invariants documented by the public API.
pub fn decompressed_len<O: Offset>(parts: Parts<'_, O>) -> usize {
    parts.codes.iter().map(|&code| code_len(parts, code)).sum()
}

/// Decode one row into a caller-provided output buffer.
///
/// Returns the number of initialized bytes in `out`.
///
/// ## Panics
///
/// Panics if `row` is out of bounds, if `out` is too small, or if `parts`
/// violates the invariants documented by the public API.
pub fn decompress_row_into<O: Offset>(
    parts: Parts<'_, O>,
    row: usize,
    out: &mut [MaybeUninit<u8>],
) -> usize {
    let (begin, end) = row_code_range(parts, row);
    let out_ptr = out.as_mut_ptr().cast::<u8>();
    let out_len = out.len();
    let mut written = 0;
    for &code in &parts.codes[begin..end] {
        let (s, e) = code_byte_range(parts, code);
        let src = parts
            .dict_bytes
            .get(s..e)
            .expect("dictionary offset range fits dictionary bytes");
        let len = src.len();
        assert!(
            len <= out_len.saturating_sub(written),
            "output buffer too small for decompressed bytes"
        );
        // SAFETY: the assertion above guarantees `out_ptr.add(written)..+len` is
        // within the caller-provided output buffer, and the dictionary range is
        // derived from the `Parts` dictionary offset table.
        unsafe {
            scalar::copy_token_bytes(src.as_ptr(), out_ptr.add(written), len);
        }
        written += len;
    }
    written
}

/// Decode every row in a [`Parts`] view into one caller-provided flat byte
/// buffer in input order.
///
/// Returns the number of initialized bytes in `out`. The caller already owns
/// the row offsets (they passed them to [`crate::compress`] or used them to
/// build the `Parts`), so they are not returned.
///
/// ## Panics
///
/// Panics if `out` is too small or if `parts` violates the invariants
/// documented by the public API.
pub fn decompress_into<O: Offset>(parts: Parts<'_, O>, out: &mut [MaybeUninit<u8>]) -> usize {
    // The checked decode is just the unchecked decode behind a one-time output
    // bounds check — no duplicate loop. The unchecked path is safe iff
    // `out.len() >= decompressed_len(parts)`. Validate that with a cheap
    // sufficient test first: every token is <= MAX_TOKEN_SIZE, so
    // `codes.len() * MAX_TOKEN_SIZE` upper-bounds the output and a generously
    // sized buffer clears it in O(1); only a tight buffer pays the exact sum.
    let big_enough = out.len() >= parts.codes.len().saturating_mul(MAX_TOKEN_SIZE)
        || out.len() >= decompressed_len(parts);
    assert!(big_enough, "output buffer too small for decompressed bytes");
    // SAFETY: `out` is at least the decoded length, so the output store cannot
    // overrun. `CHECK = true` bounds-checks each code in-loop (a cold,
    // predicted-never-taken branch that measures within noise of the unchecked
    // loop) and `decode_fat` validates the dictionary (offsets, token sizes, and
    // the required decoder padding) up front, so a malformed `Parts` panics
    // rather than reading out of bounds — making this sound for any `Parts`.
    unsafe { decode_fat::<true, O>(parts, out) }
}

/// Out-of-line panic for an out-of-range code. `#[cold]` + `#[inline(never)]` so
/// the in-loop guard is laid out as a never-taken forward branch and the hot
/// loop stays straight-line (a `cmp; jbe <cold>` ahead of the copy).
#[cold]
#[inline(never)]
pub(crate) fn code_out_of_range() -> ! {
    panic!("onpair: code index out of range for dictionary")
}

/// Why a [`Parts`] is not safe to decode — returned by [`Parts::validate`] and
/// [`Parts::validate_dictionary`].
///
/// A `Parts` is built by downstream consumers via struct literal from
/// deserialized storage (there is no validating constructor), so its arrays may
/// be corrupt. These are the violations that would make a decoder read or write
/// out of bounds.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum InvalidParts {
    /// Dictionary offsets are not strictly increasing (a token is empty or the
    /// offsets decrease).
    NonIncreasingOffsets,
    /// A dictionary token is longer than [`MAX_TOKEN_SIZE`].
    TokenTooLarge,
    /// `dict_bytes` does not extend [`MAX_TOKEN_SIZE`] bytes past the highest
    /// token offset, so the decoder's fixed-width read would run off the end.
    MissingDecoderPadding,
    /// A code does not index the dictionary (`code >= dict_tokens`).
    CodeOutOfRange,
}

impl std::fmt::Display for InvalidParts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::NonIncreasingOffsets => {
                "dictionary offsets must be increasing (non-empty tokens)"
            }
            Self::TokenTooLarge => "dictionary token exceeds MAX_TOKEN_SIZE",
            Self::MissingDecoderPadding => "dict_bytes lacks the required trailing decoder padding",
            Self::CodeOutOfRange => "code index out of range for dictionary",
        })
    }
}

impl std::error::Error for InvalidParts {}

impl<O: Offset> Parts<'_, O> {
    /// Validate the dictionary metadata every decoder relies on for memory
    /// safety, in `O(dict_tokens)` — independent of the code stream.
    ///
    /// Establishes, for every token, that offsets are strictly increasing
    /// (non-empty, non-decreasing) and no token exceeds [`MAX_TOKEN_SIZE`], and
    /// that `dict_bytes` extends [`MAX_TOKEN_SIZE`] bytes past the *highest*
    /// token offset (the last token's). Together these let the over-copy decode
    /// store a fixed 16 bytes per token without running past the decoded length,
    /// keep every copy within one token, make `codes.len() * MAX_TOKEN_SIZE` a
    /// true upper bound on the decoded length, and make the fat-table build's
    /// fixed [`MAX_TOKEN_SIZE`]-byte read of every token offset in bounds.
    ///
    /// The trailing-bytes requirement is variable: the last token needs
    /// `MAX_TOKEN_SIZE - len(last token)` bytes past the logical end — 0 when the
    /// last token is full-width, up to `MAX_TOKEN_SIZE - 1` when it is a single
    /// byte. This check accepts the exact minimum, so a minimally-padded column
    /// validates, not just a worst-case-padded one.
    ///
    /// The trailing padding is part of the on-disk format spec — [`compress`]
    /// emits it ([`Column::dict_bytes`]) — so a conformant column always
    /// validates; the check exists to reject corrupt or hand-built `Parts`.
    ///
    /// The safe decoders ([`decompress`], [`decompress_into`]) call this once
    /// per decode — it is off the `O(codes)` hot loop, so it does not affect
    /// throughput. Validate a deserialized `Parts` here once and the dictionary
    /// is known good thereafter.
    ///
    /// [`compress`]: crate::compress
    /// [`Column::dict_bytes`]: crate::Column::dict_bytes
    pub fn validate_dictionary(&self) -> Result<(), InvalidParts> {
        // `s` of the final window is the highest (last) token's offset.
        let mut last_offset = None;
        for w in self.dict_offsets.windows(2) {
            let (s, e) = (w[0], w[1]);
            if s >= e {
                return Err(InvalidParts::NonIncreasingOffsets);
            }
            if (e - s) as usize > MAX_TOKEN_SIZE {
                return Err(InvalidParts::TokenTooLarge);
            }
            last_offset = Some(s as usize);
        }
        // The decoder reads a fixed MAX_TOKEN_SIZE bytes from each token offset;
        // the highest is the last token's, so `dict_bytes` must extend
        // MAX_TOKEN_SIZE past it — i.e. MAX_TOKEN_SIZE - len(last) trailing bytes
        // past the logical end (0 when the last token is full-width). Empty
        // dictionaries decode nothing, so need no padding.
        if let Some(off) = last_offset
            && off + MAX_TOKEN_SIZE > self.dict_bytes.len()
        {
            return Err(InvalidParts::MissingDecoderPadding);
        }
        Ok(())
    }

    /// Fully validate this `Parts` for decoding: the dictionary
    /// ([`validate_dictionary`](Self::validate_dictionary)) plus every code in
    /// `[0, dict_tokens)`. `O(dict_tokens + codes)`.
    ///
    /// After `Ok(())`, the `decompress_into_unchecked*` family is memory-safe
    /// for an output buffer of at least [`decompressed_len`] bytes — validate a
    /// deserialized `Parts` once, then decode repeatedly on the unchecked fast
    /// path without re-checking.
    pub fn validate(&self) -> Result<(), InvalidParts> {
        self.validate_dictionary()?;
        let ntok = self.dict_offsets.len().saturating_sub(1);
        if self.codes.iter().any(|&c| c as usize >= ntok) {
            return Err(InvalidParts::CodeOutOfRange);
        }
        Ok(())
    }
}

/// Panic helper for the safe decoders: assert the dictionary is valid before
/// running the (otherwise unchecked) decode loop.
#[inline]
fn assert_valid_dictionary<O: Offset>(parts: Parts<'_, O>) {
    if let Err(e) = parts.validate_dictionary() {
        panic!("onpair: {e}");
    }
}

// TODO(entries-layout): reintroduce the half-size "entries" table layout for
// hosts where the fat table spills cache.
//
// The decoder used to choose per call between the fat table (`dict_tokens * 16`,
// direct `code*16` addressing) and an "entries" table (`DecodeEntry` = packed
// offset+len, `dict_tokens * 8`) via `plan()` keyed on the per-core L2 size, on
// the theory that the half-size entries table stays cache-resident when the fat
// table would not. It was dropped because it never won in measurement: the max
// fat table is `2^16 * 16 = 1 MiB`, so on any host with L2 >= 1 MiB the fat
// table can never spill L2, and the entries layout's extra dependent load
// (`code -> entry -> dict[offset]`) lost to fat's single load by +6%..+34%
// across TPC-H comment columns at bits 12/16. It is only *theoretically* useful
// on a sub-1 MiB-L2 CPU with a near-maximal bits=16 dictionary — unverified, and
// not worth the second decode loop + table + L2 probe until such a host matters.
//
// To bring it back, see commit e036a4c ("decode: fat-table layout with scalar
// copy and L2-indexed fallback"), which contains the full implementation:
// `Layout`/`plan()`/`cpu::l2_cache_bytes()`, the `DecodeEntry` table and
// `decode_entries()`, and the `padded_unchecked_loop` over it. Re-add `plan()`
// here and dispatch in `decode_fat` below.

#[inline]
fn row_code_range<O: Offset>(parts: Parts<'_, O>, row: usize) -> (usize, usize) {
    let begin = parts.code_boundaries[row]
        .to_usize()
        .expect("code boundary fits usize");
    let end = parts.code_boundaries[row + 1]
        .to_usize()
        .expect("code boundary fits usize");
    (begin, end)
}

#[inline]
fn code_byte_range<O: Offset>(parts: Parts<'_, O>, code: u16) -> (usize, usize) {
    let s = parts.dict_offsets[code as usize] as usize;
    let e = parts.dict_offsets[code as usize + 1] as usize;
    assert!(e >= s, "dictionary offsets must be nondecreasing");
    (s, e)
}

#[inline]
fn code_len<O: Offset>(parts: Parts<'_, O>, code: u16) -> usize {
    let (s, e) = code_byte_range(parts, code);
    e - s
}

/// Decode the padded fast path: build the fat table and over-copy each code.
///
/// `CHECK` is forwarded to the inner loop: `true` bounds-checks each code (the
/// safe entry points use it), `false` is the bare unchecked decode.
///
/// ## Safety
///
/// `parts` must satisfy the decode contract — in particular `parts.dict_bytes`
/// must extend [`MAX_TOKEN_SIZE`] bytes past the highest token offset (see
/// [`Parts::validate_dictionary`]) so the fat-table build's fixed-width read is
/// in bounds — and `out` must be at least the fully decoded length. With
/// `CHECK == false`, every code must also be a valid token index; with
/// `CHECK == true` that is enforced.
#[inline]
unsafe fn decode_fat<const CHECK: bool, O: Offset>(
    parts: Parts<'_, O>,
    out: &mut [MaybeUninit<u8>],
) -> usize {
    if CHECK {
        // O(dict_tokens), off the hot loop: makes the fat-table build and every
        // per-token access below in-bounds for an arbitrary `Parts`.
        assert_valid_dictionary(parts);
    }
    // SAFETY: the dictionary is padded (checked when CHECK, caller-promised
    // otherwise), so the fat build's over-read is in bounds; the decode loop
    // upholds the over-copy contract.
    unsafe { fat::decode_loop::<CHECK>(parts.codes, &fat::build(parts), out) }
}

/// Decode every code in a [`Parts`] view into one caller-provided flat byte
/// buffer without per-token bounds checks.
///
/// Returns the number of initialized bytes in `out`.
///
/// ## Safety
///
/// The caller must ensure that `out` is large enough for the fully decoded byte
/// stream and that `parts` satisfies the public API invariants — including
/// `dict_bytes` extending [`MAX_TOKEN_SIZE`] bytes past the highest token offset
/// (the over-copy decode reads them). [`Parts::validate`] checks all of these.
pub unsafe fn decompress_into_unchecked<O: Offset>(
    parts: Parts<'_, O>,
    out: &mut [MaybeUninit<u8>],
) -> usize {
    // SAFETY: forwarded under this function's safety contract.
    unsafe { decode_fat::<false, O>(parts, out) }
}

/// Decode every row in a [`Parts`] view into one flat byte buffer in input
/// order. The caller already owns the row offsets (they passed them to
/// [`crate::compress`] or used them to build the `Parts`), so they are not
/// returned.
///
/// A malformed `Parts` panics rather than reading or writing out of bounds (the
/// dictionary is validated once up front and each code is bounds-checked in the
/// decode loop). See [`Parts::validate`].
pub fn decompress<O: Offset>(parts: Parts<'_, O>) -> Vec<u8> {
    let decoded_len = decompressed_len(parts);
    let mut out: Vec<u8> = Vec::with_capacity(decoded_len);
    // SAFETY: the vector was allocated with the exact decoded length, and
    // `decode_fat` with `CHECK = true` validates the dictionary (incl. the
    // required decoder padding) and bounds-checks every code.
    let len = unsafe { decode_fat::<true, O>(parts, out.spare_capacity_mut()) };
    // SAFETY: the decoder returns exactly the number of logical bytes it
    // initialized in `out.spare_capacity_mut()`.
    unsafe { out.set_len(len) };
    out
}

#[cfg(test)]
mod tests {
    use crate::{Config, DEFAULT_CONFIG, Parts, compress};

    use super::*;

    #[test]
    fn decompress_into_uses_caller_buffer() {
        let rows: &[&[u8]] = &[b"alpha", b"", b"beta beta", b"gamma"];
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for row in rows {
            bytes.extend_from_slice(row);
            offsets.push(bytes.len() as u32);
        }

        let col = compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap();
        assert_eq!(
            col.as_parts().validate_dictionary(),
            Ok(()),
            "compressed columns include the required decoder padding"
        );
        let mut decoded = Vec::with_capacity(bytes.len());

        let len = decompress_into(col.as_parts(), decoded.spare_capacity_mut());
        // SAFETY: `len` bytes have been initialized by `decompress_into`.
        unsafe { decoded.set_len(len) };

        assert_eq!(decoded, bytes);
    }

    /// A valid, hand-built padded `Parts` with enough codes (> `MAX_TOKEN_SIZE`)
    /// to drive the 16-byte over-copy fast region as well as the exact tail.
    /// `dict_bytes` carries the worst-case `MAX_TOKEN_SIZE - 1` trailing padding.
    fn valid_padded(tokens: &[&[u8]], code_seq: &[u16]) -> (Vec<u8>, Vec<u32>, Vec<u16>, Vec<u32>) {
        let mut dict = Vec::new();
        let mut offsets = vec![0u32];
        for t in tokens {
            dict.extend_from_slice(t);
            offsets.push(dict.len() as u32);
        }
        dict.resize(dict.len() + MAX_TOKEN_SIZE - 1, 0);
        let codes = code_seq.to_vec();
        let boundaries = vec![0u32, codes.len() as u32];
        (dict, offsets, codes, boundaries)
    }

    fn parts<'a>(
        dict: &'a [u8],
        offsets: &'a [u32],
        codes: &'a [u16],
        boundaries: &'a [u32],
    ) -> Parts<'a, u32> {
        Parts {
            dict_bytes: dict,
            dict_offsets: offsets,
            bits: 3,
            codes,
            code_boundaries: boundaries,
        }
    }

    /// Happy path with a hand-built `Parts` (no compressor): decode through the
    /// over-copy fast region + exact tail and check the bytes. Cheap enough to
    /// run under Miri, which then proves the decode loop's `unsafe` has no UB.
    #[test]
    fn decode_valid_padded_roundtrip() {
        let tokens: &[&[u8]] = &[b"a", b"bc", b"def", b"ghij"];
        // 40 codes (> MAX_TOKEN_SIZE) so split = 24 fast-region + 16 exact tail.
        let seq: Vec<u16> = (0..40).map(|i| (i % 4) as u16).collect();
        let (dict, offsets, codes, bounds) = valid_padded(tokens, &seq);
        let p = parts(&dict, &offsets, &codes, &bounds);

        let expected: Vec<u8> = seq
            .iter()
            .flat_map(|&c| tokens[c as usize].iter().copied())
            .collect();

        // Safe Vec API (CHECK = true).
        assert_eq!(decompress(p), expected);

        // Safe into-buffer API (CHECK = true), generous buffer → O(1) check.
        let mut out: Vec<MaybeUninit<u8>> = (0..codes.len() * MAX_TOKEN_SIZE)
            .map(|_| MaybeUninit::uninit())
            .collect();
        let n = decompress_into(p, &mut out);
        let decoded: Vec<u8> = out[..n]
            .iter()
            .map(|b| unsafe { b.assume_init() })
            .collect();
        assert_eq!(decoded, expected);

        // Tight buffer → exact-length path through the same checked loop.
        let mut tight: Vec<MaybeUninit<u8>> =
            (0..expected.len()).map(|_| MaybeUninit::uninit()).collect();
        let n = decompress_into(p, &mut tight);
        assert_eq!(n, expected.len());

        // Explicit unchecked path (CHECK = false) must match too.
        let mut ue: Vec<MaybeUninit<u8>> = (0..codes.len() * MAX_TOKEN_SIZE)
            .map(|_| MaybeUninit::uninit())
            .collect();
        // SAFETY: valid padded parts, buffer oversized.
        let n = unsafe { decompress_into_unchecked(p, &mut ue) };
        let decoded: Vec<u8> = ue[..n].iter().map(|b| unsafe { b.assume_init() }).collect();
        assert_eq!(decoded, expected);
    }

    /// Every dictionary-corruption hazard the checked path must turn into a clean
    /// panic instead of UB. Run with a generous output buffer so the O(1) buffer
    /// check short-circuits and the dictionary validation is what fires.
    fn assert_decode_panics(dict: &[u8], offsets: &[u32], codes: &[u16]) {
        let bounds = vec![0u32, codes.len() as u32];
        let p = parts(dict, offsets, codes, &bounds);
        let mut out: Vec<MaybeUninit<u8>> = (0..codes.len() * MAX_TOKEN_SIZE + 16)
            .map(|_| MaybeUninit::uninit())
            .collect();
        decompress_into(p, &mut out);
    }

    #[test]
    #[should_panic(expected = "offsets must be increasing")]
    fn checked_panics_on_non_monotonic_offsets() {
        // offsets decrease: token 1 has e < s.
        let mut dict = b"ab".to_vec();
        dict.resize(2 + MAX_TOKEN_SIZE - 1, 0);
        assert_decode_panics(&dict, &[0, 2, 1], &[0, 1]);
    }

    #[test]
    #[should_panic(expected = "offsets must be increasing")]
    fn checked_panics_on_zero_length_token() {
        // token 1 is empty (s == e): breaks the over-copy "≥ 1 byte ahead" rule.
        let mut dict = b"ab".to_vec();
        dict.resize(2 + MAX_TOKEN_SIZE - 1, 0);
        assert_decode_panics(&dict, &[0, 1, 1, 2], &[0, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "exceeds MAX_TOKEN_SIZE")]
    fn checked_panics_on_oversize_token() {
        // token 0 is 20 bytes (> MAX_TOKEN_SIZE) → over-copy could outrun `out`.
        let mut dict = vec![b'x'; 21];
        dict.resize(21 + MAX_TOKEN_SIZE - 1, 0);
        assert_decode_panics(&dict, &[0, 20, 21], &[0, 1]);
    }

    #[test]
    #[should_panic(expected = "decoder padding")]
    fn checked_panics_on_missing_padding() {
        // 6-byte dict with no trailing padding: the over-read would run off the
        // end, so validation must reject it.
        assert_decode_panics(b"abcdef", &[0, 4, 6], &[0, 1]);
    }

    #[test]
    fn parts_validate_classifies_corruption() {
        // Valid padded parts → Ok for both the dictionary and full checks.
        let tokens: &[&[u8]] = &[b"a", b"bc", b"def"];
        let (dict, offsets, codes, bounds) = valid_padded(tokens, &[0, 1, 2, 0]);
        let p = parts(&dict, &offsets, &codes, &bounds);
        assert_eq!(p.validate_dictionary(), Ok(()));
        assert_eq!(p.validate(), Ok(()));

        let pad = |dict: &mut Vec<u8>| dict.resize(dict.len() + MAX_TOKEN_SIZE - 1, 0);

        // Non-increasing offsets.
        let mut d = b"ab".to_vec();
        pad(&mut d);
        assert_eq!(
            parts(&d, &[0, 2, 1], &[0], &[0, 1]).validate_dictionary(),
            Err(InvalidParts::NonIncreasingOffsets)
        );

        // Oversize token.
        let mut d = vec![b'x'; 21];
        pad(&mut d);
        assert_eq!(
            parts(&d, &[0, 20, 21], &[0], &[0, 1]).validate_dictionary(),
            Err(InvalidParts::TokenTooLarge)
        );

        // Dictionary lacks the required trailing decoder padding.
        assert_eq!(
            parts(b"abcdef", &[0, 4, 6], &[0], &[0, 1]).validate_dictionary(),
            Err(InvalidParts::MissingDecoderPadding)
        );

        // Dictionary is fine but a code is out of range: only the full check
        // catches it (the dictionary check is independent of the code stream).
        let mut d = b"ab".to_vec();
        pad(&mut d);
        let p = parts(&d, &[0, 1, 2], &[0, 5], &[0, 2]);
        assert_eq!(p.validate_dictionary(), Ok(()));
        assert_eq!(p.validate(), Err(InvalidParts::CodeOutOfRange));
    }

    #[test]
    fn decode_zero_padding_full_width_last_token() {
        // Tightest case for the fat build's fixed-width over-read: the last token
        // is a full MAX_TOKEN_SIZE bytes, so the dictionary carries ZERO trailing
        // padding and the 16-byte read of that token ends exactly at the buffer
        // end. > MAX_TOKEN_SIZE codes drive the over-copy fast region. (Miri
        // confirms the boundary read is in bounds.)
        let mut dict = vec![b'x']; // token 0: 1 byte
        dict.extend(std::iter::repeat_n(b'y', MAX_TOKEN_SIZE)); // token 1: full width
        let offsets = [0u32, 1, 1 + MAX_TOKEN_SIZE as u32];
        let seq: Vec<u16> = (0..40).map(|i| (i % 2) as u16).collect();
        let bounds = vec![0u32, seq.len() as u32];
        let p = parts(&dict, &offsets, &seq, &bounds);
        assert_eq!(p.validate_dictionary(), Ok(()));

        let expected: Vec<u8> = seq
            .iter()
            .flat_map(|&c| {
                dict[offsets[c as usize] as usize..offsets[c as usize + 1] as usize].to_vec()
            })
            .collect();
        assert_eq!(decompress(p), expected);
    }

    #[test]
    fn validate_padding_requirement_is_variable() {
        // A full-width (MAX_TOKEN_SIZE) last token needs ZERO trailing padding:
        // the fixed MAX_TOKEN_SIZE read from its offset ends exactly at the
        // logical end.
        let full = vec![b'z'; MAX_TOKEN_SIZE];
        let offs = [0u32, MAX_TOKEN_SIZE as u32];
        assert_eq!(
            parts(&full, &offs, &[0], &[0, 1]).validate_dictionary(),
            Ok(())
        );
        // One byte short → the fixed read would run off the end.
        let short = vec![b'z'; MAX_TOKEN_SIZE - 1];
        assert_eq!(
            parts(&short, &offs, &[0], &[0, 1]).validate_dictionary(),
            Err(InvalidParts::MissingDecoderPadding)
        );

        // The other extreme: a 1-byte last token ("b" at offset 1) needs the
        // full MAX_TOKEN_SIZE - 1 padding — reading MAX_TOKEN_SIZE from offset 1
        // needs 1 + MAX_TOKEN_SIZE bytes (= 2 logical + MAX_TOKEN_SIZE - 1).
        let mut buf = b"ab".to_vec();
        buf.resize(1 + MAX_TOKEN_SIZE, 0);
        assert_eq!(
            parts(&buf, &[0, 1, 2], &[0, 1], &[0, 2]).validate_dictionary(),
            Ok(())
        );
        buf.pop(); // one byte short
        assert_eq!(
            parts(&buf, &[0, 1, 2], &[0, 1], &[0, 2]).validate_dictionary(),
            Err(InvalidParts::MissingDecoderPadding)
        );
    }

    #[test]
    #[should_panic(expected = "code index out of range")]
    fn decompress_into_panics_on_out_of_range_code() {
        // A code past the dictionary would read out of bounds; the in-loop guard
        // (`CHECK = true`) must turn that into a clean panic instead of UB. Pad
        // the dictionary so the padded fast path is taken, and size `out`
        // generously so `decompress_into`'s O(1) buffer check short-circuits
        // before `decompressed_len` would index the bad code itself.
        let mut dict = b"ab".to_vec();
        dict.resize(2 + MAX_TOKEN_SIZE - 1, 0);
        let offsets = [0u32, 1, 2];
        let boundaries = [0u32, 2];
        let codes = [0u16, 5]; // 5 is out of range (dict has 2 tokens)
        let parts = Parts {
            dict_bytes: &dict,
            dict_offsets: &offsets,
            bits: 3,
            codes: &codes,
            code_boundaries: &boundaries,
        };
        assert_eq!(parts.validate_dictionary(), Ok(()));

        let mut out: Vec<MaybeUninit<u8>> = (0..codes.len() * MAX_TOKEN_SIZE)
            .map(|_| MaybeUninit::uninit())
            .collect();
        decompress_into(parts, &mut out);
    }

    #[test]
    #[should_panic(expected = "decoder padding")]
    fn decompress_panics_on_unpadded_parts() {
        // The trailing decoder padding is part of the format spec; an unpadded
        // dictionary is rejected rather than silently decoded.
        let offsets = [0u32, 1, 2];
        let boundaries = [0u32, 2];
        let codes = [0u16, 1];
        let parts = Parts {
            dict_bytes: b"ab",
            dict_offsets: &offsets,
            bits: 1,
            codes: &codes,
            code_boundaries: &boundaries,
        };
        decompress(parts);
    }

    #[test]
    fn decompress_row_into_uses_caller_buffer() {
        let rows: &[&[u8]] = &[b"short", b"longer-row", b"", b"tail"];
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for row in rows {
            bytes.extend_from_slice(row);
            offsets.push(bytes.len() as u32);
        }

        let col = compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap();
        for (row, expected) in rows.iter().enumerate() {
            let mut decoded = Vec::with_capacity(expected.len());
            let len = decompress_row_into(col.as_parts(), row, decoded.spare_capacity_mut());
            // SAFETY: `len` bytes have been initialized by `decompress_row_into`.
            unsafe { decoded.set_len(len) };
            assert_eq!(decoded, *expected);
        }
    }

    /// Exercise the full decode width sweep against a corpus large enough to
    /// drive the batched AVX-512 prefix, the scalar 16-byte remainder, and the
    /// exact tail in a single call.
    #[test]
    fn decompress_matches_input_across_widths() {
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for i in 0..5000u32 {
            let row = format!("row-{i:04}-https://example.com/path/{}", i % 37);
            bytes.extend_from_slice(row.as_bytes());
            offsets.push(bytes.len() as u32);
        }
        for bits in 9..=16u32 {
            let cfg = Config {
                bits,
                ..DEFAULT_CONFIG
            };
            let col = compress(&bytes, &offsets, cfg).unwrap();
            assert_eq!(
                decompress(col.as_parts()),
                bytes,
                "decompress @ bits={bits}"
            );

            let mut decoded = Vec::with_capacity(bytes.len());
            let len = decompress_into(col.as_parts(), decoded.spare_capacity_mut());
            // SAFETY: `len` bytes initialized by `decompress_into`.
            unsafe { decoded.set_len(len) };
            assert_eq!(decoded, bytes, "decompress_into @ bits={bits}");
        }
    }
}
