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
//! [`plan`] chooses between two table layouts ([`fat`] vs `entries`) per call
//! from the dictionary size vs L2; the copy is always scalar.

use std::mem::MaybeUninit;

use crate::column::Parts;
use crate::offset::Offset;
use crate::types::MAX_TOKEN_SIZE;

mod scalar;

pub(crate) mod fat;

/// Extra bytes required after the logical dictionary bytes when using
/// fixed-width dictionary reads.
pub const DECOMPRESS_BUFFER_PADDING: usize = MAX_TOKEN_SIZE - 1;

/// Precomputed decode metadata for one dictionary token.
#[derive(Copy, Clone, Debug)]
pub struct DecodeEntry(u64);

impl DecodeEntry {
    #[inline]
    fn new(offset: u32, len: u32) -> Self {
        Self(((len as u64) << 32) | offset as u64)
    }

    #[inline]
    pub(crate) fn offset(self) -> usize {
        self.0 as u32 as usize
    }

    #[inline]
    pub(crate) fn len(self) -> usize {
        (self.0 >> 32) as usize
    }
}

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

#[inline]
fn dict_has_decoder_padding<O: Offset>(parts: Parts<'_, O>) -> bool {
    let Some(&logical_len) = parts.dict_offsets.last() else {
        return false;
    };
    (logical_len as usize)
        .checked_add(DECOMPRESS_BUFFER_PADDING)
        .is_some_and(|padded_len| parts.dict_bytes.len() >= padded_len)
}

#[inline]
fn write_code<O: Offset>(
    parts: Parts<'_, O>,
    code: u16,
    out_ptr: *mut u8,
    out_len: usize,
    written: &mut usize,
) {
    let (s, e) = code_byte_range(parts, code);
    let src = parts
        .dict_bytes
        .get(s..e)
        .expect("dictionary offset range fits dictionary bytes");
    let len = src.len();
    assert!(
        len <= out_len.saturating_sub(*written),
        "output buffer too small for decompressed bytes"
    );

    // SAFETY: the assertion above guarantees `out_ptr.add(*written)..+len`
    // is within the caller-provided output buffer, and the dictionary range is
    // derived from the `Parts` dictionary offset table.
    unsafe {
        scalar::copy_token_bytes(src.as_ptr(), out_ptr.add(*written), len);
    }
    *written += len;
}

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

/// Build a per-token decode table for repeated fast decompression.
///
/// ## Panics
///
/// Panics if `parts` violates the dictionary offset invariants documented by
/// the public API.
pub fn decode_entries<O: Offset>(parts: Parts<'_, O>) -> Vec<DecodeEntry> {
    let len = parts.dict_offsets.len().saturating_sub(1);
    (0..len)
        .map(|i| {
            let s = parts.dict_offsets[i];
            let e = parts.dict_offsets[i + 1];
            assert!(e > s, "dictionary tokens must be nonempty");
            DecodeEntry::new(s, e - s)
        })
        .collect()
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
    let mut written = 0;
    for &code in &parts.codes[begin..end] {
        write_code(parts, code, out_ptr, out.len(), &mut written);
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
    // loop) and `decode_padded_unchecked` validates the dictionary up front, so
    // a malformed `Parts` panics rather than reading out of bounds — making this
    // sound for any `Parts`. Padding selects the over-copy fast path vs the exact
    // path.
    unsafe {
        if dict_has_decoder_padding(parts) {
            decode_padded_unchecked::<true, O>(parts, out)
        } else {
            unpadded_loop::<true, O>(parts, out)
        }
    }
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
    /// The last dictionary offset runs past the end of `dict_bytes`.
    OffsetsExceedBytes,
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
            Self::OffsetsExceedBytes => "dictionary offsets exceed dictionary bytes",
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
    /// (non-empty, non-decreasing), no token exceeds [`MAX_TOKEN_SIZE`], and the
    /// last offset lies within `dict_bytes`. Together these let the over-copy
    /// fast path store a fixed 16 bytes per token without running past the
    /// decoded length, keep every `copy16` / `copy_token_bytes` within one
    /// token, and make `codes.len() * MAX_TOKEN_SIZE` a true upper bound on the
    /// decoded length. (The 16-byte *over-read* additionally needs 15 bytes of
    /// trailing dictionary padding, which the padded decoders require
    /// separately.)
    ///
    /// The safe decoders ([`decompress`], [`decompress_into`]) call this once
    /// per decode — it is off the `O(codes)` hot loop, so it does not affect
    /// throughput. Validate a deserialized `Parts` here once and the dictionary
    /// is known good thereafter.
    pub fn validate_dictionary(&self) -> Result<(), InvalidParts> {
        for w in self.dict_offsets.windows(2) {
            let (s, e) = (w[0], w[1]);
            if s >= e {
                return Err(InvalidParts::NonIncreasingOffsets);
            }
            if (e - s) as usize > MAX_TOKEN_SIZE {
                return Err(InvalidParts::TokenTooLarge);
            }
        }
        match self.dict_offsets.last() {
            Some(&last) if last as usize > self.dict_bytes.len() => {
                Err(InvalidParts::OffsetsExceedBytes)
            }
            _ => Ok(()),
        }
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

/// Decode the padded fast path over the entries table.
///
/// Mirrors `decompress_into_unchecked_padded_with_entries`: the
/// final [`MAX_TOKEN_SIZE`] codes are copied exactly so the output buffer
/// needs no trailing padding; everything before is over-copied 16 bytes at a
/// time.
///
/// When `CHECK` is `true`, each code is bounds-checked against `entries` with a
/// cold, predicted-never-taken branch so a malformed `Parts` panics instead of
/// reading out of bounds. When `false`, the guard compiles out — byte-identical
/// to a bare unchecked loop.
///
/// ## Safety
///
/// `entries` must be built from `parts`, `parts.dict_bytes` must have 16-byte
/// read padding past every token offset, and `out` must be at least the fully
/// decoded byte length. With `CHECK == false`, every code must also index
/// `entries`; with `CHECK == true` that is enforced.
#[inline]
unsafe fn padded_unchecked_loop<const CHECK: bool, O: Offset>(
    parts: Parts<'_, O>,
    entries: &[DecodeEntry],
    out: &mut [MaybeUninit<u8>],
) -> usize {
    let entries_ptr = entries.as_ptr();
    let ntok = entries.len();
    let dict = parts.dict_bytes.as_ptr();
    let out_ptr = out.as_mut_ptr().cast::<u8>();
    let n = parts.codes.len();
    let split = n.saturating_sub(MAX_TOKEN_SIZE);

    let mut written = 0usize;
    let mut i = 0usize;
    while i < split {
        let c = parts.codes[i] as usize;
        if CHECK && c >= ntok {
            code_out_of_range();
        }
        // SAFETY: `c < ntok` (checked above when CHECK, caller-promised
        // otherwise); ≥ MAX_TOKEN_SIZE codes remain after `i`, guaranteeing ≥ 16
        // trailing output bytes for the over-store; dict read padding per
        // contract.
        unsafe {
            let entry = *entries_ptr.add(c);
            scalar::copy16(dict.add(entry.offset()), out_ptr.add(written));
            written += entry.len();
        }
        i += 1;
    }

    for &code in &parts.codes[split..] {
        let c = code as usize;
        if CHECK && c >= ntok {
            code_out_of_range();
        }
        // SAFETY: as above; exact copy of the token's true length within the
        // decoded len.
        unsafe {
            let entry = *entries_ptr.add(c);
            scalar::copy_token_bytes(dict.add(entry.offset()), out_ptr.add(written), entry.len());
            written += entry.len();
        }
    }
    written
}

/// Token table layout chosen per decode call by [`plan`].
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Layout {
    /// `DecodeEntry` (offset + len), `dict_tokens * 8` bytes. One extra dependent
    /// load per token, but stays cache-resident when the fat table would not.
    Entries,
    /// 16-byte-strided rows ([`fat`]), `dict_tokens * 16` bytes. Direct
    /// `code * 16` addressing, no dependent load.
    Fat,
}

/// Pick the decode table layout from the actual dictionary size vs the host L2.
///
/// [`Layout::Fat`] when its `dict_tokens * 16` table fits L2 (direct addressing
/// beats the entries dependent load); otherwise [`Layout::Entries`], which is
/// half the size and stays resident. Keyed on the *actual* trained dictionary,
/// not the `2^bits` capacity, since the trainer usually fills only a fraction.
fn plan(dict_tokens: usize) -> Layout {
    if dict_tokens * 16 <= crate::cpu::l2_cache_bytes() {
        Layout::Fat
    } else {
        Layout::Entries
    }
}

/// Trained dictionary token count for a column (one fat row per token).
#[inline]
fn dict_tokens<O: Offset>(parts: Parts<'_, O>) -> usize {
    parts.dict_offsets.len().saturating_sub(1)
}

/// Layout-dispatched decode of the padded fast path: pick the table by
/// dictionary size vs L2, materialize it, decode (scalar copy).
///
/// `CHECK` is forwarded to the inner loop: `true` bounds-checks each code (the
/// safe entry points use it), `false` is the bare unchecked decode.
///
/// ## Safety
///
/// `parts` must satisfy the padded-decode contract and `out` must be at least
/// the fully decoded length. With `CHECK == false`, every code must also be a
/// valid token index; with `CHECK == true` that is enforced.
#[inline]
unsafe fn decode_padded_unchecked<const CHECK: bool, O: Offset>(
    parts: Parts<'_, O>,
    out: &mut [MaybeUninit<u8>],
) -> usize {
    if CHECK {
        // O(dict_tokens), off the hot loop: makes every per-token table and
        // dictionary access below in-bounds for an arbitrary `Parts`.
        assert_valid_dictionary(parts);
    }
    // SAFETY: tables are built from `parts`; the loops uphold the padded-decode
    // contract.
    unsafe {
        match plan(dict_tokens(parts)) {
            Layout::Fat => fat::decode_loop::<CHECK>(parts.codes, &fat::build(parts), out),
            Layout::Entries => {
                padded_unchecked_loop::<CHECK, O>(parts, &decode_entries(parts), out)
            }
        }
    }
}

/// Exact (non-over-copying) decode of every code into `out`, reading token byte
/// ranges straight from `dict_offsets`. Used when the dictionary lacks the
/// trailing read padding the over-copy fast paths require.
///
/// When `CHECK` is `true`, each code is bounds-checked against the dictionary
/// with a cold, predicted-never-taken branch (so `*offsets.add(i + 1)` stays in
/// bounds); when `false` the guard compiles out — byte-identical to a bare
/// unchecked loop.
///
/// ## Safety
///
/// `out` must be at least the fully decoded length and `parts` must satisfy the
/// public API invariants. With `CHECK == false`, every code must also be a valid
/// token index (`< dict_tokens`); with `CHECK == true` that is enforced.
#[inline]
unsafe fn unpadded_loop<const CHECK: bool, O: Offset>(
    parts: Parts<'_, O>,
    out: &mut [MaybeUninit<u8>],
) -> usize {
    if CHECK {
        // O(dict_tokens), off the hot loop: makes every offset/dict access below
        // in-bounds for an arbitrary `Parts`.
        assert_valid_dictionary(parts);
    }
    let offsets = parts.dict_offsets.as_ptr();
    let ntok = dict_tokens(parts);
    let dict = parts.dict_bytes.as_ptr();
    let out_ptr = out.as_mut_ptr().cast::<u8>();
    let mut written = 0;
    for &code in parts.codes {
        let i = code as usize;
        if CHECK && i >= ntok {
            code_out_of_range();
        }
        // SAFETY: `i < ntok` (checked above when CHECK, caller-promised
        // otherwise) ⇒ `offsets.add(i)` and `offsets.add(i + 1)` are in bounds;
        // output length guaranteed by this function's safety contract.
        unsafe {
            let s = *offsets.add(i) as usize;
            let e = *offsets.add(i + 1) as usize;
            let len = e - s;
            scalar::copy_token_bytes(dict.add(s), out_ptr.add(written), len);
            written += len;
        }
    }
    written
}

/// Decode every code in a [`Parts`] view into one caller-provided flat byte
/// buffer without per-token bounds checks.
///
/// Returns the number of initialized bytes in `out`.
///
/// ## Safety
///
/// The caller must ensure that `out` is large enough for the fully decoded
/// byte stream and that `parts` satisfies the public API invariants.
pub unsafe fn decompress_into_unchecked<O: Offset>(
    parts: Parts<'_, O>,
    out: &mut [MaybeUninit<u8>],
) -> usize {
    // SAFETY: forwarded under this function's safety contract.
    unsafe { unpadded_loop::<false, O>(parts, out) }
}

/// Decode every code in a [`Parts`] view using fixed-width token over-copies.
///
/// This mirrors the C++ fast path for the fast prefix: each prefix token copies
/// 16 bytes and advances the output cursor by the token's true length. The
/// final `MAX_TOKEN_SIZE` codes are copied exactly, so the output buffer does
/// not need trailing padding.
///
/// ## Safety
///
/// The caller must ensure that:
///
/// - `out` is at least the fully decoded byte length.
/// - `parts.dict_bytes` has enough trailing padding that reading 16 bytes from
///   every token offset is valid.
/// - `parts` satisfies the public API invariants.
pub unsafe fn decompress_into_unchecked_padded<O: Offset>(
    parts: Parts<'_, O>,
    out: &mut [MaybeUninit<u8>],
) -> usize {
    let offsets = parts.dict_offsets.as_ptr();
    let dict = parts.dict_bytes.as_ptr();
    let out_ptr = out.as_mut_ptr().cast::<u8>();
    let mut written = 0;

    let (fast_codes, exact_codes) = parts
        .codes
        .split_at(parts.codes.len().saturating_sub(MAX_TOKEN_SIZE));

    for &code in fast_codes {
        let i = code as usize;
        // SAFETY: guaranteed by this function's safety contract.
        unsafe {
            let s = *offsets.add(i) as usize;
            let e = *offsets.add(i + 1) as usize;
            scalar::copy16(dict.add(s), out_ptr.add(written));
            written += e - s;
        }
    }

    for &code in exact_codes {
        let i = code as usize;
        // SAFETY: guaranteed by this function's safety contract.
        unsafe {
            let s = *offsets.add(i) as usize;
            let e = *offsets.add(i + 1) as usize;
            let len = e - s;
            scalar::copy_token_bytes(dict.add(s), out_ptr.add(written), len);
            written += len;
        }
    }

    written
}

/// Decode every code using fixed-width over-copies and precomputed
/// [`DecodeEntry`] metadata.
///
/// ## Safety
///
/// The caller must ensure that:
///
/// - `entries` was built from the same dictionary metadata as `parts`.
/// - `out` is at least the fully decoded byte length.
/// - `parts.dict_bytes` has enough trailing padding that reading 16 bytes from
///   every token offset is valid.
/// - `parts` satisfies the public API invariants.
pub unsafe fn decompress_into_unchecked_padded_with_entries<O: Offset>(
    parts: Parts<'_, O>,
    entries: &[DecodeEntry],
    out: &mut [MaybeUninit<u8>],
) -> usize {
    // SAFETY: forwarded under this function's safety contract; `CHECK = false`
    // makes this byte-identical to a bare unchecked decode.
    unsafe { padded_unchecked_loop::<false, O>(parts, entries, out) }
}

/// Decode every row in a [`Parts`] view into one flat byte buffer in input
/// order. The caller already owns the row offsets (they passed them to
/// [`crate::compress`] or used them to build the `Parts`), so they are not
/// returned.
///
/// An out-of-range code panics rather than reading out of bounds (the decode
/// loop bounds-checks each code against the dictionary). Other `Parts` invariant
/// violations documented in the crate-root PUBLIC_API are not separately
/// validated.
pub fn decompress<O: Offset>(parts: Parts<'_, O>) -> Vec<u8> {
    let decoded_len = decompressed_len(parts);
    let mut out: Vec<u8> = Vec::with_capacity(decoded_len);
    let len = if dict_has_decoder_padding(parts) {
        // SAFETY: the vector was allocated with the exact decoded length, and
        // `dict_has_decoder_padding` guarantees dictionary read padding;
        // `CHECK = true` bounds-checks each code so an out-of-range code panics
        // rather than reading out of bounds.
        unsafe { decode_padded_unchecked::<true, O>(parts, out.spare_capacity_mut()) }
    } else {
        // SAFETY: the vector was allocated with at least the exact decoded
        // length; `CHECK = true` bounds-checks each code.
        unsafe { unpadded_loop::<true, O>(parts, out.spare_capacity_mut()) }
    };
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
        assert!(
            dict_has_decoder_padding(col.as_parts()),
            "compressed columns include decoder padding"
        );
        let mut decoded = Vec::with_capacity(bytes.len());

        let len = decompress_into(col.as_parts(), decoded.spare_capacity_mut());
        // SAFETY: `len` bytes have been initialized by `decompress_into`.
        unsafe { decoded.set_len(len) };

        assert_eq!(decoded, bytes);
    }

    /// A valid, hand-built padded `Parts` with enough codes (> `MAX_TOKEN_SIZE`)
    /// to drive the 16-byte over-copy fast region as well as the exact tail.
    /// `dict_bytes` carries the trailing `DECOMPRESS_BUFFER_PADDING`.
    fn valid_padded(tokens: &[&[u8]], code_seq: &[u16]) -> (Vec<u8>, Vec<u32>, Vec<u16>, Vec<u32>) {
        let mut dict = Vec::new();
        let mut offsets = vec![0u32];
        for t in tokens {
            dict.extend_from_slice(t);
            offsets.push(dict.len() as u32);
        }
        dict.resize(dict.len() + DECOMPRESS_BUFFER_PADDING, 0);
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

        // Explicit unchecked entries path (CHECK = false) must match too.
        let entries = decode_entries(p);
        let mut ue: Vec<MaybeUninit<u8>> = (0..codes.len() * MAX_TOKEN_SIZE)
            .map(|_| MaybeUninit::uninit())
            .collect();
        // SAFETY: valid padded parts, entries built from it, buffer oversized.
        let n = unsafe { decompress_into_unchecked_padded_with_entries(p, &entries, &mut ue) };
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
        dict.resize(2 + DECOMPRESS_BUFFER_PADDING, 0);
        assert_decode_panics(&dict, &[0, 2, 1], &[0, 1]);
    }

    #[test]
    #[should_panic(expected = "offsets must be increasing")]
    fn checked_panics_on_zero_length_token() {
        // token 1 is empty (s == e): breaks the over-copy "≥ 1 byte ahead" rule.
        let mut dict = b"ab".to_vec();
        dict.resize(2 + DECOMPRESS_BUFFER_PADDING, 0);
        assert_decode_panics(&dict, &[0, 1, 1, 2], &[0, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "exceeds MAX_TOKEN_SIZE")]
    fn checked_panics_on_oversize_token() {
        // token 0 is 20 bytes (> MAX_TOKEN_SIZE) → over-copy could outrun `out`.
        let mut dict = vec![b'x'; 21];
        dict.resize(21 + DECOMPRESS_BUFFER_PADDING, 0);
        assert_decode_panics(&dict, &[0, 20, 21], &[0, 1]);
    }

    #[test]
    #[should_panic(expected = "offsets exceed dictionary bytes")]
    fn checked_panics_on_offset_past_dict_bytes() {
        // Last offset (8) runs past the 6-byte dictionary (no padding → exact
        // path), so a token read would be out of bounds.
        assert_decode_panics(b"abcdef", &[0, 4, 8], &[0, 1]);
    }

    #[test]
    fn parts_validate_classifies_corruption() {
        // Valid padded parts → Ok for both the dictionary and full checks.
        let tokens: &[&[u8]] = &[b"a", b"bc", b"def"];
        let (dict, offsets, codes, bounds) = valid_padded(tokens, &[0, 1, 2, 0]);
        let p = parts(&dict, &offsets, &codes, &bounds);
        assert_eq!(p.validate_dictionary(), Ok(()));
        assert_eq!(p.validate(), Ok(()));

        let pad = |dict: &mut Vec<u8>| dict.resize(dict.len() + DECOMPRESS_BUFFER_PADDING, 0);

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

        // Offsets past dict bytes.
        assert_eq!(
            parts(b"abcdef", &[0, 4, 8], &[0], &[0, 1]).validate_dictionary(),
            Err(InvalidParts::OffsetsExceedBytes)
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
    #[should_panic(expected = "code index out of range")]
    fn decompress_into_panics_on_out_of_range_code() {
        // A code past the dictionary would read out of bounds; the in-loop guard
        // (`CHECK = true`) must turn that into a clean panic instead of UB. Pad
        // the dictionary so the padded fast path is taken, and size `out`
        // generously so `decompress_into`'s O(1) buffer check short-circuits
        // before `decompressed_len` would index the bad code itself.
        let mut dict = b"ab".to_vec();
        dict.resize(2 + DECOMPRESS_BUFFER_PADDING, 0);
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
        assert!(dict_has_decoder_padding(parts));

        let mut out: Vec<MaybeUninit<u8>> = (0..codes.len() * MAX_TOKEN_SIZE)
            .map(|_| MaybeUninit::uninit())
            .collect();
        decompress_into(parts, &mut out);
    }

    #[test]
    fn decompress_falls_back_for_unpadded_parts() {
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

        assert!(!dict_has_decoder_padding(parts));
        assert_eq!(decompress(parts), b"ab");
    }

    #[test]
    fn plan_layout_tracks_actual_dict_size() {
        let l2 = crate::cpu::l2_cache_bytes();
        // A tiny dictionary's fat table fits any L2 → Fat.
        assert_eq!(plan(1), Layout::Fat, "tiny dict → fat");
        // A dictionary whose fat table (tokens × 16) exceeds L2 → Entries, keyed
        // on the actual token count, not the `2^bits` capacity.
        assert_eq!(
            plan(l2 / 16 + 4096),
            Layout::Entries,
            "dict whose fat table exceeds L2 → entries"
        );
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
