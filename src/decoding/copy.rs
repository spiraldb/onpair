// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The per-token copy primitive.
//!
//! Decode throughput is dominated by a fixed 16-byte over-copy per token. A
//! single 16-byte value copy ([`copy16`]) lowers to one native 128-bit store on
//! every target (`movups` on x86 at the SSE2 baseline, `str q` on AArch64), so a
//! hand-written SIMD intrinsic buys nothing over it.

/// Over-copy a full 16-byte token chunk from `src` to `dst`.
///
/// Callers advance the output cursor by the token's true length; the extra bytes
/// are overwritten by the next token, and the trailing
/// [`DECODE_PADDING`](super::DECODE_PADDING) room absorbs the last token's
/// over-store.
///
/// # Safety
/// `src` must be readable for 16 bytes and `dst` writable for 16 bytes.
#[inline(always)]
pub(crate) unsafe fn copy16(src: *const u8, dst: *mut u8) {
    // SAFETY: caller guarantees 16 readable/writable bytes. A 16-byte value copy
    // lowers to a single 128-bit `movups` (x86) / `str q` (AArch64).
    unsafe {
        dst.cast::<[u8; 16]>()
            .write_unaligned(src.cast::<[u8; 16]>().read_unaligned());
    }
}
