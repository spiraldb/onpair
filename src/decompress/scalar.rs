// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Portable scalar decode primitives — the only per-token copy backend.
//!
//! Decode throughput is dominated by a 16-byte over-copy per token. A single
//! 16-byte value copy ([`copy16`]) lowers to one native 128-bit store on every
//! target (`movups` on x86 at the SSE2 baseline, `str q` on AArch64), so a
//! hand-written SIMD intrinsic buys nothing over it — measured equal to an
//! explicit `_mm_storeu_si128`, and faster than every wider-SIMD scheme tried.
//! Hence there is one scalar backend, no architecture-specific copy modules.

/// Over-copy a full 16-byte token chunk from `src` to `dst`.
///
/// Callers advance the output cursor by the token's true length; the extra
/// bytes are overwritten by the next token (or live in trailing output slack).
///
/// ## Safety
///
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

/// Copy exactly `len` bytes (`len <= MAX_TOKEN_SIZE`) from `src` to `dst`.
///
/// For non-power-of-two lengths this copies the first and last chunk of the
/// next lower power of two; the overlapping middle bytes are written twice but
/// never outside `[dst, dst + len)`, so no source or destination padding is
/// required.
///
/// ## Safety
///
/// `src` and `dst` must be valid for `len` bytes.
#[inline(always)]
pub(crate) unsafe fn copy_token_bytes(src: *const u8, dst: *mut u8, len: usize) {
    // SAFETY: every arm reads and writes only within `[ptr, ptr + len)`.
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
            16 => copy16(src, dst),
            _ => std::ptr::copy_nonoverlapping(src, dst, len),
        }
    }
}
