// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;

use crate::column::Parts;
use crate::offset::Offset;
use crate::types;

/// Extra bytes required after the logical dictionary bytes when using
/// fixed-width dictionary reads.
pub const DECOMPRESS_BUFFER_PADDING: usize = types::MAX_TOKEN_SIZE - 1;

/// Precomputed decode metadata for one dictionary token.
#[derive(Copy, Clone, Debug)]
pub struct DecodeEntry(u64);

impl DecodeEntry {
    #[inline]
    fn new(offset: u32, len: u32) -> Self {
        Self(((len as u64) << 32) | offset as u64)
    }

    #[inline]
    fn offset(self) -> usize {
        self.0 as u32 as usize
    }

    #[inline]
    fn len(self) -> usize {
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

#[inline(always)]
unsafe fn copy_16_token_bytes(src: *const u8, dst: *mut u8) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        std::arch::aarch64::vst1q_u8(dst, std::arch::aarch64::vld1q_u8(src));
    }

    #[cfg(not(target_arch = "aarch64"))]
    unsafe {
        dst.cast::<u64>()
            .write_unaligned(src.cast::<u64>().read_unaligned());
        dst.add(8)
            .cast::<u64>()
            .write_unaligned(src.add(8).cast::<u64>().read_unaligned());
    }
}

#[inline(always)]
unsafe fn copy_token_bytes(src: *const u8, dst: *mut u8, len: usize) {
    // Tokens are capped at 16 bytes. For non-power-of-two lengths, copy the
    // first and last chunk of the next lower power of two; the overlapping
    // middle bytes are written twice but never outside `[dst, dst + len)`.
    unsafe {
        match len {
            0 => {}
            1 => dst.write(src.read()),
            2 | 3 => {
                dst.cast::<u16>()
                    .write_unaligned(src.cast::<u16>().read_unaligned());
                dst.add(len - 2)
                    .cast::<u16>()
                    .write_unaligned(src.add(len - 2).cast::<u16>().read_unaligned());
            }
            4..=7 => {
                dst.cast::<u32>()
                    .write_unaligned(src.cast::<u32>().read_unaligned());
                dst.add(len - 4)
                    .cast::<u32>()
                    .write_unaligned(src.add(len - 4).cast::<u32>().read_unaligned());
            }
            8..=15 => {
                dst.cast::<u64>()
                    .write_unaligned(src.cast::<u64>().read_unaligned());
                dst.add(len - 8)
                    .cast::<u64>()
                    .write_unaligned(src.add(len - 8).cast::<u64>().read_unaligned());
            }
            16 => copy_16_token_bytes(src, dst),
            _ => std::ptr::copy_nonoverlapping(src, dst, len),
        }
    }
}

#[inline(always)]
unsafe fn copy_padded_token_bytes(src: *const u8, dst: *mut u8) {
    // SAFETY: guaranteed by the caller. This intentionally over-copies one
    // maximum-width token.
    unsafe {
        copy_16_token_bytes(src, dst);
    }
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
        copy_token_bytes(src.as_ptr(), out_ptr.add(*written), len);
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

#[inline]
fn decompress_into_checked<O: Offset>(parts: Parts<'_, O>, out: &mut [MaybeUninit<u8>]) -> usize {
    let out_ptr = out.as_mut_ptr().cast::<u8>();
    let mut written = 0;
    for &code in parts.codes {
        write_code(parts, code, out_ptr, out.len(), &mut written);
    }
    written
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
    if dict_has_decoder_padding(parts) {
        let entries = decode_entries(parts);
        // SAFETY: `decode_entries` was built from `parts`,
        // `dict_has_decoder_padding` guarantees dictionary read padding, and
        // output capacity is checked before each token write.
        return unsafe { decompress_into_checked_padded_with_entries(parts, &entries, out) };
    }

    decompress_into_checked(parts, out)
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
    let offsets = parts.dict_offsets.as_ptr();
    let dict = parts.dict_bytes.as_ptr();
    let out_ptr = out.as_mut_ptr().cast::<u8>();
    let mut written = 0;
    for &code in parts.codes {
        let i = code as usize;
        // SAFETY: guaranteed by this function's safety contract.
        unsafe {
            let s = *offsets.add(i) as usize;
            let e = *offsets.add(i + 1) as usize;
            let len = e - s;
            copy_token_bytes(dict.add(s), out_ptr.add(written), len);
            written += len;
        }
    }
    written
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
        .split_at(parts.codes.len().saturating_sub(types::MAX_TOKEN_SIZE));

    for &code in fast_codes {
        let i = code as usize;
        // SAFETY: guaranteed by this function's safety contract.
        unsafe {
            let s = *offsets.add(i) as usize;
            let e = *offsets.add(i + 1) as usize;
            copy_padded_token_bytes(dict.add(s), out_ptr.add(written));
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
            copy_token_bytes(dict.add(s), out_ptr.add(written), len);
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
    let entries = entries.as_ptr();
    let dict = parts.dict_bytes.as_ptr();
    let out_ptr = out.as_mut_ptr().cast::<u8>();
    let mut written = 0;

    let (fast_codes, exact_codes) = parts
        .codes
        .split_at(parts.codes.len().saturating_sub(types::MAX_TOKEN_SIZE));

    for &code in fast_codes {
        // SAFETY: guaranteed by this function's safety contract.
        unsafe {
            let entry = *entries.add(code as usize);
            copy_padded_token_bytes(dict.add(entry.offset()), out_ptr.add(written));
            written += entry.len();
        }
    }

    for &code in exact_codes {
        // SAFETY: guaranteed by this function's safety contract.
        unsafe {
            let entry = *entries.add(code as usize);
            copy_token_bytes(dict.add(entry.offset()), out_ptr.add(written), entry.len());
            written += entry.len();
        }
    }

    written
}

unsafe fn decompress_into_checked_padded_with_entries<O: Offset>(
    parts: Parts<'_, O>,
    entries: &[DecodeEntry],
    out: &mut [MaybeUninit<u8>],
) -> usize {
    let entries = entries.as_ptr();
    let dict = parts.dict_bytes.as_ptr();
    let out_ptr = out.as_mut_ptr().cast::<u8>();
    let out_len = out.len();
    let mut written = 0;
    let mut code_index = 0;
    let fast_end = out_len.saturating_sub(types::MAX_TOKEN_SIZE - 1);

    while code_index < parts.codes.len() && written < fast_end {
        let code = parts.codes[code_index];
        // SAFETY: guaranteed by this function's safety contract.
        unsafe {
            let entry = *entries.add(code as usize);
            copy_padded_token_bytes(dict.add(entry.offset()), out_ptr.add(written));
            written += entry.len();
        }
        code_index += 1;
    }

    for &code in &parts.codes[code_index..] {
        // SAFETY: guaranteed by this function's safety contract.
        unsafe {
            let entry = *entries.add(code as usize);
            assert!(
                written <= out_len,
                "output buffer too small for decompressed bytes"
            );
            let remaining = out_len - written;
            assert!(
                entry.len() <= remaining,
                "output buffer too small for decompressed bytes"
            );
            copy_token_bytes(dict.add(entry.offset()), out_ptr.add(written), entry.len());
            written += entry.len();
        }
    }
    written
}

/// Decode every row in a [`Parts`] view into one flat byte buffer in input
/// order. The caller already owns the row offsets (they passed them to
/// [`crate::compress`] or used them to build the `Parts`), so they are not
/// returned.
///
/// Does not validate the `Parts` invariants documented in the crate-root
/// PUBLIC_API: a malformed `Parts` will panic or produce out-of-bounds reads.
pub fn decompress<O: Offset>(parts: Parts<'_, O>) -> Vec<u8> {
    let decoded_len = decompressed_len(parts);
    let mut out: Vec<u8> = Vec::with_capacity(decoded_len);
    let len = if dict_has_decoder_padding(parts) {
        let entries = decode_entries(parts);
        // SAFETY: the vector was allocated with the exact decoded length, and
        // `dict_has_decoder_padding` guarantees dictionary read padding.
        unsafe {
            decompress_into_unchecked_padded_with_entries(parts, &entries, out.spare_capacity_mut())
        }
    } else {
        // SAFETY: the vector was allocated with at least the exact decoded
        // length.
        unsafe { decompress_into_unchecked(parts, out.spare_capacity_mut()) }
    };
    // SAFETY: the decoder returns exactly the number of logical bytes it
    // initialized in `out.spare_capacity_mut()`.
    unsafe { out.set_len(len) };
    out
}

#[cfg(test)]
mod tests {
    use crate::{DEFAULT_CONFIG, Parts, compress};

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
}
