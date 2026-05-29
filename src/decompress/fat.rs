// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! "Fat" token table layout.
//!
//! Each token is materialized into a 16-byte-strided row, so a decode load
//! addresses `data + code * 16` straight from the code — replacing the
//! `code → entry → dict[offset]` dependent-load chain of the
//! [`super::DecodeEntry`] layout with a single independent load. Costs
//! `dict_tokens * 16` bytes of table; whether that pays is a cache-residency
//! question the [`super::plan`] index decides per host.
//!
//! Loop structure: a 16-byte over-copy fast region ([`super::scalar::copy16`])
//! plus an exact, length-aware tail.

use std::mem::MaybeUninit;

use super::scalar;
use crate::column::Parts;
use crate::types::MAX_TOKEN_SIZE;

/// 16-byte-strided token table: row `code` holds the token bytes (zero-padded to
/// 16) and `lens[code]` the true length.
pub(crate) struct FatTable {
    data: Vec<u8>,
    lens: Vec<u8>,
}

/// Materialize the [`FatTable`] for a column. Built once per decode call.
///
/// Each token is over-copied into its 16-byte row with a single branchless
/// [`super::scalar::copy16`] — the same fixed [`MAX_TOKEN_SIZE`]-byte read the
/// decode loop uses, which is why the dictionary must carry trailing padding.
/// The row's bytes past the token's true length hold neighbouring dictionary
/// bytes; that is harmless because decode advances the output cursor by the true
/// length (`lens[code]`) and the over-written tail is reclaimed by the next
/// token (or by the exact decode tail).
///
/// ## Safety
///
/// `parts.dict_bytes` must extend [`MAX_TOKEN_SIZE`] bytes past the highest
/// token offset (so the fixed-width read from every offset is in bounds — at
/// most `MAX_TOKEN_SIZE - 1` bytes past the logical end), and
/// `parts.dict_offsets` must be non-decreasing with tokens ≤ [`MAX_TOKEN_SIZE`]
/// (i.e. [`Parts::validate_dictionary`] holds).
pub(crate) unsafe fn build(parts: Parts<'_>) -> FatTable {
    let dict_size = parts.dict_offsets.len().saturating_sub(1);
    // One extra row of slack so the last row's 16-byte over-store stays in
    // bounds even though the token's true length may be < 16.
    let mut data = vec![0u8; dict_size * 16 + 16];
    let mut lens = vec![0u8; dict_size];
    let src = parts.dict_bytes.as_ptr();
    let dst = data.as_mut_ptr();
    for i in 0..dict_size {
        let s = parts.dict_offsets[i] as usize;
        let e = parts.dict_offsets[i + 1] as usize;
        lens[i] = (e - s) as u8;
        // SAFETY: dict padding (contract) gives 16 readable bytes from `s`; row
        // `i` is 16 writable bytes within `data` (`dict_size*16 + 16`).
        unsafe { scalar::copy16(src.add(s), dst.add(i * 16)) };
    }
    FatTable { data, lens }
}

/// Decode `codes` into `out`.
///
/// When `CHECK` is `true`, each code is bounds-checked against the dictionary
/// (`dict_tokens`) with a cold, predicted-never-taken branch so a malformed
/// `Parts` panics instead of reading out of bounds. The loop stays count-bound
/// and the copy stays a single store, so the guard is free in practice —
/// measured within noise of, and at small dictionaries faster than, the
/// unchecked loop (code-layout effects dominate the guard's cost). When `false`,
/// the guard compiles out — byte-identical to a bare unchecked loop.
///
/// ## Safety
///
/// `fat` must be built from the same column as `codes`, and `out` must be at
/// least the fully decoded length. With `CHECK == false`, every code must also
/// be a valid token index (`< dict_tokens`); with `CHECK == true` that is
/// enforced.
#[inline]
pub(crate) unsafe fn decode_loop<const CHECK: bool>(
    codes: &[u16],
    fat: &FatTable,
    out: &mut [MaybeUninit<u8>],
) -> usize {
    let data = fat.data.as_ptr();
    let lens = fat.lens.as_ptr();
    let ntok = fat.lens.len(); // == dict_tokens
    let op = out.as_mut_ptr().cast::<u8>();
    let n = codes.len();
    let split = n.saturating_sub(MAX_TOKEN_SIZE);

    let mut w = 0usize;
    let mut i = 0usize;
    while i < split {
        let c = codes[i] as usize;
        if CHECK && c >= ntok {
            super::code_out_of_range();
        }
        // SAFETY: `c < ntok` (checked above when CHECK, caller-promised
        // otherwise) ⇒ the fat row and length reads are in bounds; ≥
        // MAX_TOKEN_SIZE codes remain after `i` ⇒ ≥ 16 trailing output bytes for
        // the over-store; fat row holds 16 readable bytes.
        unsafe {
            scalar::copy16(data.add(c * 16), op.add(w));
            w += *lens.add(c) as usize;
        }
        i += 1;
    }

    for &code in &codes[split..] {
        let c = code as usize;
        if CHECK && c >= ntok {
            super::code_out_of_range();
        }
        // SAFETY: as above; exact copy of the token's true length within the
        // decoded len.
        unsafe {
            let len = *lens.add(c) as usize;
            scalar::copy_token_bytes(data.add(c * 16), op.add(w), len);
            w += len;
        }
    }
    w
}
