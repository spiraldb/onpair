// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scans of the code stream against a compiled cover.
//!
//! The vector kernels walk the flat code stream a vector at a time, OR together
//! one comparison per point and two per range, and append the rows the surviving
//! lanes fall in. On x86, wide dense covers may instead use an exact row-centric
//! membership-table scan with one early exit per row.
//!
//! There is no silent full-column fallback when SIMD is unavailable. The x86
//! row-centric path is an explicit density and row-length policy; other covers
//! use the detected vector kernel. The scalar routine under `cfg(test)` is the
//! common correctness oracle.

#[cfg(target_arch = "aarch64")]
mod aarch64;
mod policy;
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
mod sink;
#[cfg(target_arch = "x86_64")]
mod table;
#[cfg(target_arch = "x86_64")]
mod x86;

#[cfg(all(target_arch = "aarch64", test))]
pub(super) use self::aarch64::scan_neon;
#[cfg(target_arch = "x86_64")]
use self::policy::Avx512Kernel;
use self::policy::{
    AnalysisFacts, CoverShape, Kernel, KernelPlan, RegionFacts, ScanFacts, detect_target_caps,
    select_kernel,
};
#[cfg(all(target_arch = "x86_64", test))]
pub(super) use self::x86::{scan_avx2, scan_avx512, scan_sse2};
use super::cover::ProbeCover;
use super::{PrefilterAnalysis, PrefilterError};
use crate::core::offset::Offset;
use crate::core::types::Token;

/// Borrowed buffers for one scan region.
#[derive(Clone, Copy)]
pub(super) struct ScanInput<'a, O> {
    pub(super) codes: &'a [Token],
    pub(super) row_offsets: &'a [O],
    pub(super) cover: &'a ProbeCover,
}

impl<'a, O> ScanInput<'a, O> {
    pub(super) const fn full(
        codes: &'a [Token],
        row_offsets: &'a [O],
        cover: &'a ProbeCover,
    ) -> Self {
        Self {
            codes,
            row_offsets,
            cover,
        }
    }
}

/// Derive an ephemeral kernel plan without inspecting code values.
#[inline]
pub(super) fn plan<O: Offset>(input: ScanInput<'_, O>, analysis: &PrefilterAnalysis) -> KernelPlan {
    let facts = ScanFacts {
        analysis: AnalysisFacts {
            shape: CoverShape {
                points: input.cover.points.len(),
                ranges: input.cover.ranges.len(),
            },
            table_len: input.cover.table.len(),
            covered_codes: analysis.covered_frequency() as usize,
            indexed_codes: analysis.total_frequency() as usize,
        },
        region: RegionFacts {
            code_count: input.codes.len(),
            row_count: input.row_offsets.len().saturating_sub(1),
        },
    };
    select_kernel(detect_target_caps(), facts)
}

#[inline]
pub(super) const fn reserve(plan: KernelPlan) -> usize {
    plan.reserve
}

/// Compatibility entry for tests that exercise dispatch with a synthetic
/// cover, without constructing a complete analysis.
#[cfg(test)]
pub(super) fn scan<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    cover: &ProbeCover,
    covered_frequency: usize,
    out: &mut Vec<usize>,
) -> Result<(), PrefilterError> {
    let input = ScanInput::full(codes, row_offsets, cover);
    let facts = ScanFacts {
        analysis: AnalysisFacts {
            shape: CoverShape {
                points: cover.points.len(),
                ranges: cover.ranges.len(),
            },
            table_len: cover.table.len(),
            covered_codes: covered_frequency,
            indexed_codes: codes.len(),
        },
        region: RegionFacts {
            code_count: codes.len(),
            row_count: row_offsets.len().saturating_sub(1),
        },
    };
    let plan = select_kernel(detect_target_caps(), facts);
    out.reserve(plan.reserve);
    execute(plan, input, out)
}

/// Execute a previously selected plan. This is the first stage that inspects
/// code values.
#[inline]
pub(super) fn execute<O: Offset>(
    plan: KernelPlan,
    input: ScanInput<'_, O>,
    out: &mut Vec<usize>,
) -> Result<(), PrefilterError> {
    let sparse_row_mapping = plan.row_mapping.uses_sparse_gaps();
    match plan.kernel {
        Kernel::Empty => Ok(()),
        #[cfg(any(not(any(target_arch = "aarch64", target_arch = "x86_64")), test))]
        Kernel::Unsupported => Err(PrefilterError::UnsupportedArchitecture),
        #[cfg(target_arch = "aarch64")]
        Kernel::Neon(kernel) => {
            aarch64::execute(kernel, input, sparse_row_mapping, out);
            Ok(())
        }
        #[cfg(target_arch = "x86_64")]
        Kernel::RowsTable => {
            table::scan_rows(input.codes, input.row_offsets, input.cover, out);
            Ok(())
        }
        #[cfg(target_arch = "x86_64")]
        Kernel::Sse2(kernel) => {
            x86::execute_sse2(kernel, input, sparse_row_mapping, out);
            Ok(())
        }
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx2(kernel) => {
            // SAFETY: the plan is created only after runtime AVX2 detection.
            unsafe { x86::execute_avx2(kernel, input, sparse_row_mapping, out) };
            Ok(())
        }
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx512(Avx512Kernel::Generic) => {
            // SAFETY: the plan is created only after runtime AVX-512BW
            // detection, which implies AVX-512F.
            unsafe {
                x86::scan_avx512(
                    input.codes,
                    input.row_offsets,
                    input.cover,
                    sparse_row_mapping,
                    out,
                )
            };
            Ok(())
        }
        #[cfg(test)]
        _ => unreachable!("kernel plan does not match the compiling target"),
    }
}

/// Test oracle for the SIMD implementations.
#[cfg(test)]
pub(super) fn scan_scalar<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    out: &mut Vec<usize>,
) {
    for row in 0..row_offsets.len().saturating_sub(1) {
        let a = row_offsets[row].to_usize();
        let b = row_offsets[row + 1].to_usize();
        if codes[a..b].iter().any(|&code| pf.table[code as usize]) {
            out.push(row);
        }
    }
}
