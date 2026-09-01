// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! AArch64 scan execution.

use super::ScanInput;
use super::policy::FixedShape;
use crate::core::offset::Offset;

mod neon;

#[cfg(test)]
pub(in crate::search::substring::prefilter) use neon::scan_neon;

#[inline]
pub(super) fn execute<O: Offset>(
    shape: Option<FixedShape>,
    group: u8,
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
    match shape {
        Some(shape) => {
            neon::scan_neon_fixed(shape, codes, row_offsets, cover, sparse_row_mapping, out)
        }
        None => neon::scan_neon_generic(
            codes,
            row_offsets,
            cover,
            sparse_row_mapping,
            group == 2,
            out,
        ),
    }
}
