// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! AArch64 scan execution.

use super::ScanInput;
use super::policy::{FixedShape, NeonKernel};
use crate::core::offset::Offset;

mod neon;

#[cfg(test)]
pub(in crate::search::prefilter) use neon::scan_neon;
pub(super) use neon::{
    scan_neon_few_mixed, scan_neon_fixed_mixed, scan_neon_generic, scan_neon_one_point_two_ranges,
    scan_neon_one_range, scan_neon_points, scan_neon_points_many,
};

#[inline]
pub(super) fn execute<O: Offset>(
    kernel: NeonKernel,
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
        NeonKernel::FewPoints { points: 1..=8 } => {
            scan_neon_points(codes, row_offsets, cover, sparse_row_mapping, out)
        }
        NeonKernel::ManyPoints => {
            scan_neon_points_many(codes, row_offsets, cover, sparse_row_mapping, out)
        }
        NeonKernel::OneRange => {
            scan_neon_one_range(codes, row_offsets, cover, sparse_row_mapping, out)
        }
        NeonKernel::Fixed(FixedShape {
            points: 1,
            ranges: 1,
        }) => scan_neon_fixed_mixed::<O, 1, 1>(codes, row_offsets, cover, sparse_row_mapping, out),
        NeonKernel::Fixed(FixedShape {
            points: 2,
            ranges: 1,
        }) => scan_neon_fixed_mixed::<O, 2, 1>(codes, row_offsets, cover, sparse_row_mapping, out),
        NeonKernel::OnePointTwoRanges => {
            scan_neon_one_point_two_ranges(codes, row_offsets, cover, sparse_row_mapping, out)
        }
        NeonKernel::FewMixed => {
            scan_neon_few_mixed(codes, row_offsets, cover, sparse_row_mapping, out)
        }
        NeonKernel::Generic { two_vectors } => scan_neon_generic(
            codes,
            row_offsets,
            cover,
            sparse_row_mapping,
            two_vectors,
            out,
        ),
        _ => unreachable!("invalid NEON kernel shape"),
    }
}
