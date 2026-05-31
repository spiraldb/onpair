// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod sealed {
    pub trait Sealed {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
}

/// Width of a row offset. Sealed to `u32` and `u64` to match Arrow binary /
/// large-binary layouts.
pub trait Offset: sealed::Sealed + Copy + Clone + Default + std::fmt::Debug + 'static {
    /// Convert to `usize`, returning `None` if the value does not fit.
    fn to_usize(self) -> Option<usize>;
    /// Construct from a `usize`, truncating if it does not fit.
    fn from_usize(n: usize) -> Self;
    /// Convert to `usize` by truncation — the exact inverse of
    /// [`from_usize`](Self::from_usize). For offsets that were validated at
    /// construction (fit in `usize`, ≤ buffer length) this is lossless, and
    /// unlike [`to_usize`](Self::to_usize) it is branchless: no fallible check,
    /// no panic path. Use it on hot per-row paths over already-validated
    /// offsets.
    fn as_usize(self) -> usize;
}

impl Offset for u32 {
    #[inline]
    fn to_usize(self) -> Option<usize> {
        Some(self as usize)
    }
    #[inline]
    fn from_usize(n: usize) -> Self {
        n as u32
    }
    #[inline]
    fn as_usize(self) -> usize {
        self as usize
    }
}

impl Offset for u64 {
    #[inline]
    fn to_usize(self) -> Option<usize> {
        usize::try_from(self).ok()
    }
    #[inline]
    fn from_usize(n: usize) -> Self {
        n as u64
    }
    #[inline]
    fn as_usize(self) -> usize {
        self as usize
    }
}
