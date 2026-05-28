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
    fn to_usize(self) -> Option<usize>;
    fn from_usize(n: usize) -> Self;
    fn zero() -> Self;
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
    fn zero() -> Self {
        0
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
    fn zero() -> Self {
        0
    }
}
