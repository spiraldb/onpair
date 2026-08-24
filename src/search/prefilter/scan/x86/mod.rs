// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! x86-64 scan execution and target-feature boundaries.

use super::ScanInput;
use super::policy::{
    Avx2Kernel, FixedShape, HitMaterialization, Sse2Kernel, with_avx2_fixed_shapes,
    with_sse2_fixed_shapes,
};
use crate::core::offset::Offset;
use crate::core::types::Token;
use crate::search::prefilter::cover::ProbeCover;

mod avx2;
mod avx512;
mod sse2;

#[cfg(test)]
pub(in crate::search::prefilter) use avx2::scan_avx2;
pub(super) use avx2::{
    scan_avx2_few, scan_avx2_fixed, scan_avx2_generic, scan_avx2_one_point, scan_avx2_one_range,
};
pub(in crate::search::prefilter) use avx512::scan_avx512;
#[cfg(test)]
pub(in crate::search::prefilter) use sse2::scan_sse2;
pub(super) use sse2::{scan_sse2_fixed, scan_sse2_generic, scan_sse2_one_point};

#[inline]
pub(super) fn execute_sse2<O: Offset>(
    kernel: Sse2Kernel,
    input: ScanInput<'_, O>,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    let ScanInput {
        codes,
        row_offsets,
        cover,
        ..
    } = input;
    match kernel {
        Sse2Kernel::OnePoint => {
            scan_sse2_one_point(codes, row_offsets, cover, sparse_row_mapping, out)
        }
        Sse2Kernel::Fixed(shape) => {
            execute_sse2_fixed(shape, codes, row_offsets, cover, sparse_row_mapping, out)
        }
        Sse2Kernel::Generic => {
            scan_sse2_generic(codes, row_offsets, cover, sparse_row_mapping, out)
        }
    }
}

#[inline]
fn execute_sse2_fixed<O: Offset>(
    shape: FixedShape,
    codes: &[Token],
    row_offsets: &[O],
    cover: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    macro_rules! call {
        ($points:literal, $ranges:literal) => {
            scan_sse2_fixed::<O, $points, $ranges>(
                codes,
                row_offsets,
                cover,
                sparse_row_mapping,
                out,
            )
        };
    }
    macro_rules! execute_shape {
        ($(($points:literal, $ranges:literal),)+) => {
            match (shape.points, shape.ranges) {
                $(($points, $ranges) => call!($points, $ranges),)+
                _ => unreachable!("invalid SSE2 fixed shape"),
            }
        };
    }
    with_sse2_fixed_shapes!(execute_shape)
}

pub(super) unsafe fn execute_avx2<O: Offset>(
    kernel: Avx2Kernel,
    input: ScanInput<'_, O>,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    let ScanInput {
        codes,
        row_offsets,
        cover,
        ..
    } = input;
    // Keep the executor itself baseline-compatible to make the runtime feature
    // boundary explicit. The large AVX2 leaf kernels are also `#[inline(never)]`:
    // ThinLTO otherwise folds their shape specializations into one
    // instruction-cache-heavy dispatcher.
    unsafe {
        match kernel {
            Avx2Kernel::OnePoint { hits } => scan_avx2_one_point(
                codes,
                row_offsets,
                cover,
                sparse_row_mapping,
                hits == HitMaterialization::CompactMask,
                out,
            ),
            Avx2Kernel::OneRange => {
                scan_avx2_one_range(codes, row_offsets, cover, sparse_row_mapping, out)
            }
            Avx2Kernel::Fixed(shape) => {
                execute_avx2_fixed(shape, codes, row_offsets, cover, sparse_row_mapping, out)
            }
            Avx2Kernel::Few {
                hits: HitMaterialization::CompactMask,
            } => scan_avx2_few::<O, true>(codes, row_offsets, cover, sparse_row_mapping, out),
            Avx2Kernel::Few {
                hits: HitMaterialization::StoredLanes,
            } => scan_avx2_few::<O, false>(codes, row_offsets, cover, sparse_row_mapping, out),
            Avx2Kernel::Generic => {
                scan_avx2_generic(codes, row_offsets, cover, sparse_row_mapping, out)
            }
        }
    }
}

unsafe fn execute_avx2_fixed<O: Offset>(
    shape: FixedShape,
    codes: &[Token],
    row_offsets: &[O],
    cover: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    macro_rules! call {
        ($points:literal, $ranges:literal) => {
            scan_avx2_fixed::<O, $points, $ranges>(
                codes,
                row_offsets,
                cover,
                sparse_row_mapping,
                out,
            )
        };
    }
    macro_rules! execute_shape {
        ($(($points:literal, $ranges:literal),)+) => {
            match (shape.points, shape.ranges) {
                $(($points, $ranges) => call!($points, $ranges),)+
                _ => unreachable!("invalid AVX2 fixed shape"),
            }
        };
    }
    unsafe { with_avx2_fixed_shapes!(execute_shape) }
}
