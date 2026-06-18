// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The fat-table decode kernel.
//!
//! [`FatTable`] materializes a `num_tokens × MAX_TOKEN_SIZE` byte-strided copy
//! of a dictionary once, turning each subsequent decode into a single
//! `table[code]` load with no `code → offset → bytes` indirection. Best for
//! bulk decode or repeated random access; for a one-shot decode prefer the
//! direct [`decode_into`](super::decode_into).

use std::mem::MaybeUninit;

use super::scalar;
use crate::core::dictionary::DictionaryView;
use crate::core::types::{MAX_TOKEN_SIZE, Token};

/// A `num_tokens × MAX_TOKEN_SIZE` byte-strided copy of a dictionary: row `c`
/// holds token `c`'s bytes (zero-padded to 16) and `lens[c]` its true length.
///
/// Built once from a dictionary, then reused to decode any number of code
/// slices: a decode addresses `data + c * 16` directly — one independent load,
/// with no `code → offset → bytes` indirection.
pub struct FatTable {
    /// `num_tokens` rows of 16 bytes (`num_tokens * 16` total). The widest
    /// access is the 16-byte over-store of the last row; since the load width
    /// equals the row stride, it ends exactly at the buffer's end — no trailing
    /// padding needed.
    data: Vec<u8>,
    /// `num_tokens` true token lengths.
    lens: Vec<u8>,
}

impl FatTable {
    /// Materialize the table for `dict`.
    ///
    /// # Safety
    /// `dict` must be read-padded — the build reads [`MAX_TOKEN_SIZE`] bytes from
    /// each token offset (the same wide read the decode performs).
    pub unsafe fn build(dict: DictionaryView<'_>) -> Self {
        let n = dict.num_tokens();
        let mut data = vec![0u8; n * 16];
        let mut lens = vec![0u8; n];
        let src = dict.bytes.as_ptr();
        let dpos = data.as_mut_ptr();
        for i in 0..n {
            let off = dict.offsets[i] as usize;
            let len = dict.offsets[i + 1] as usize - off;
            lens[i] = len as u8;
            // SAFETY: read-padding ⇒ 16 readable bytes from `off`; row `i` is 16
            // writable bytes within `data` (i < n, so i*16 + 16 <= n*16).
            unsafe { scalar::copy16(src.add(off), dpos.add(i * 16)) };
        }
        Self { data, lens }
    }

    /// Number of tokens in the table.
    #[inline]
    pub fn num_tokens(&self) -> usize {
        self.lens.len()
    }

    /// Exact decoded byte length of `codes` against this table (the sum of token
    /// lengths). Sizes a [`decode_into`](Self::decode_into) buffer from the table
    /// alone, with no need to re-consult the dictionary.
    ///
    /// Precondition: every code is `< self.num_tokens()`.
    #[inline]
    pub fn decoded_len(&self, codes: &[Token]) -> usize {
        codes.iter().map(|&c| self.lens[c as usize] as usize).sum()
    }

    /// Decode `codes` into `out`, returning bytes written. As with
    /// [`decode_into`](super::decode_into), the final [`MAX_TOKEN_SIZE`] tokens
    /// are copied at their true length, so `out` may be sized to the exact
    /// decoded length.
    ///
    /// # Safety
    /// - every `code` is `< self.num_tokens()`;
    /// - `out.len() >= decoded_len`, the sum of the decoded token lengths.
    pub unsafe fn decode_into(&self, codes: &[Token], out: &mut [MaybeUninit<u8>]) -> usize {
        let data = self.data.as_ptr();
        let lens = self.lens.as_ptr();
        let dst = out.as_mut_ptr().cast::<u8>();
        let mut w = 0usize;
        // See `decode_into`: over-store while >= MAX_TOKEN_SIZE tokens follow,
        // then copy the short tail at its true length.
        let (fast, tail) = codes.split_at(codes.len().saturating_sub(MAX_TOKEN_SIZE));
        for &code in fast {
            let c = code as usize;
            // SAFETY: c < num_tokens ⇒ row c (16 bytes) and lens[c] are in
            // bounds; >= MAX_TOKEN_SIZE tokens remain ⇒ the over-store lands
            // within `[w, decoded_len) ⊆ out`.
            unsafe {
                scalar::copy16(data.add(c * 16), dst.add(w));
                w += *lens.add(c) as usize;
            }
        }
        for &code in tail {
            let c = code as usize;
            // SAFETY: c < num_tokens ⇒ row c and lens[c] are in bounds; the
            // exact copy of `len <= MAX_TOKEN_SIZE` bytes writes within
            // `[w, decoded_len) ⊆ out`.
            unsafe {
                let len = *lens.add(c) as usize;
                scalar::copy_token_bytes(data.add(c * 16), dst.add(w), len);
                w += len;
            }
        }
        w
    }

    /// Decode `codes` into a freshly allocated `Vec`, sized exactly. Bulk
    /// counterpart to the direct [`decode_to_vec`](super::decode_to_vec).
    ///
    /// Assumes every code is `< self.num_tokens()` — true for any code stream
    /// from the [`Column`](crate::Column) the table was built for; an
    /// out-of-range code panics in the length pass.
    pub fn decode_to_vec(&self, codes: &[Token]) -> Vec<u8> {
        let n = self.decoded_len(codes);
        let mut out = Vec::with_capacity(n);
        // SAFETY: buffer sized to the exact decoded length; codes assumed in range.
        let w = unsafe { self.decode_into(codes, out.spare_capacity_mut()) };
        // SAFETY: the kernel initialized exactly `w` leading bytes.
        unsafe { out.set_len(w) };
        out
    }
}
