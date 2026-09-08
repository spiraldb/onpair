// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pure, host-independent scan-kernel selection.

/// Invoke a local macro with every wide-x86 shape of comparison cost at most
/// eight. Policy and execution expand this list, keeping selection and dispatch
/// aligned.
#[cfg(any(target_arch = "x86_64", test))]
macro_rules! with_x86_fixed_shapes {
    ($apply:ident) => {
        $apply! {
            (1, 0),
            (2, 0),
            (3, 0),
            (0, 1),
            (1, 1),
            (2, 1),
            (4, 0),
            (0, 2),
            (5, 0),
            (3, 1),
            (1, 2),
            (6, 0),
            (4, 1),
            (2, 2),
            (0, 3),
            (7, 0),
            (5, 1),
            (3, 2),
            (1, 3),
            (8, 0),
            (6, 1),
            (4, 2),
            (2, 3),
            (0, 4),
        }
    };
}

/// SSE2's narrower vectors benefit from fixed probes on this original subset.
#[cfg(any(target_arch = "x86_64", test))]
macro_rules! with_sse2_fixed_shapes {
    ($apply:ident) => {
        $apply! {
            (1, 0),
            (0, 1),
            (2, 0),
            (3, 0),
            (1, 1),
            (4, 0),
            (2, 1),
            (5, 0),
            (0, 2),
            (1, 2),
            (3, 1),
            (6, 0),
            (4, 1),
            (2, 2),
            (0, 3),
            (3, 2),
            (1, 3),
            (4, 2),
        }
    };
}

#[cfg(any(target_arch = "x86_64", test))]
pub(super) use {with_sse2_fixed_shapes, with_x86_fixed_shapes};

/// Shape of a normalized probe cover.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CoverShape {
    pub(super) points: usize,
    pub(super) ranges: usize,
}

impl CoverShape {
    #[inline]
    pub(super) const fn is_empty(self) -> bool {
        self.points == 0 && self.ranges == 0
    }
}

/// Stable facts produced by prefilter analysis.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct AnalysisFacts {
    pub(super) shape: CoverShape,
    pub(super) covered_codes: usize,
    pub(super) indexed_codes: usize,
}

/// Facts about the particular region being scanned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RegionFacts {
    pub(super) code_count: usize,
    pub(super) row_count: usize,
}

/// All data used to choose a kernel. No code values are inspected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ScanFacts {
    pub(super) analysis: AnalysisFacts,
    pub(super) region: RegionFacts,
}

impl ScanFacts {
    /// Expected covered codes in this region.
    ///
    /// This is exact for the indexed population and a policy-only projection
    /// for a subset. It never affects correctness.
    #[cfg_attr(
        not(any(target_arch = "x86_64", test)),
        allow(
            dead_code,
            reason = "AArch64 reservation remains disabled pending its Phase 5 measurement"
        )
    )]
    #[inline]
    pub(super) fn expected_covered_codes(self) -> usize {
        if self.analysis.indexed_codes == self.region.code_count {
            return self.analysis.covered_codes;
        }
        if self.analysis.indexed_codes == 0 {
            return 0;
        }
        let projected = (self.analysis.covered_codes as u128)
            .saturating_mul(self.region.code_count as u128)
            / self.analysis.indexed_codes as u128;
        usize::try_from(projected).unwrap_or(usize::MAX)
    }

    #[inline]
    pub(super) fn covered_below(self, ratio: Ratio) -> bool {
        ratio.above_fraction(self.analysis.covered_codes, self.analysis.indexed_codes)
    }
}

/// An overflow-safe ratio used by scan policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Ratio {
    numerator: u64,
    denominator: u64,
}

impl Ratio {
    pub(super) const fn new(numerator: u64, denominator: u64) -> Self {
        assert!(denominator != 0);
        Self {
            numerator,
            denominator,
        }
    }

    #[inline]
    fn above_fraction(self, numerator: usize, denominator: usize) -> bool {
        (numerator as u128).saturating_mul(self.denominator as u128)
            < (denominator as u128).saturating_mul(self.numerator as u128)
    }
}

const SPARSE_ROW_MAPPING_MAX_COVERAGE: Ratio = Ratio::new(1, 10_000);

/// SIMD capabilities relevant to prefilter execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TargetCaps {
    #[cfg(any(target_arch = "aarch64", test))]
    Aarch64Neon,
    #[cfg(any(target_arch = "x86_64", test))]
    X86_64 { avx2: bool, avx512bw: bool },
    #[cfg(any(not(any(target_arch = "aarch64", target_arch = "x86_64")), test))]
    Unsupported,
}

/// Detect capabilities once per public scan invocation.
#[inline]
pub(super) fn detect_target_caps() -> TargetCaps {
    #[cfg(target_arch = "aarch64")]
    {
        TargetCaps::Aarch64Neon
    }
    #[cfg(target_arch = "x86_64")]
    {
        TargetCaps::X86_64 {
            avx2: std::is_x86_feature_detected!("avx2"),
            avx512bw: std::is_x86_feature_detected!("avx512bw"),
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        TargetCaps::Unsupported
    }
}

/// Candidate-row materialization policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RowMapping {
    Linear,
    AdaptiveSparse,
}

impl RowMapping {
    #[inline]
    pub(super) const fn uses_sparse_gaps(self) -> bool {
        matches!(self, Self::AdaptiveSparse)
    }
}

/// A const-generic point/range specialization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct FixedShape {
    pub(super) points: usize,
    pub(super) ranges: usize,
}

impl FixedShape {
    const fn new(points: usize, ranges: usize) -> Self {
        Self { points, ranges }
    }
}

/// Executable kernel selected for one scan region.
///
/// Each target carries only configuration meaningful to that target, so an
/// execution plan cannot contain a mismatched ISA, shape, and group size.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Kernel {
    Empty,
    #[cfg(any(target_arch = "aarch64", test))]
    Neon {
        fixed: Option<FixedShape>,
        two_vectors: bool,
    },
    #[cfg(any(target_arch = "x86_64", test))]
    Sse2 {
        fixed: Option<FixedShape>,
    },
    #[cfg(any(target_arch = "x86_64", test))]
    Avx2 {
        fixed: Option<FixedShape>,
    },
    #[cfg(any(target_arch = "x86_64", test))]
    Avx512 {
        fixed: Option<FixedShape>,
    },
    #[cfg(any(not(any(target_arch = "aarch64", target_arch = "x86_64")), test))]
    Unsupported,
}

/// Complete, ephemeral execution plan for one scan region.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct KernelPlan {
    pub(super) kernel: Kernel,
    pub(super) row_mapping: RowMapping,
    pub(super) reserve: usize,
}

/// Select a kernel without inspecting code values or executing SIMD.
#[inline]
pub(super) fn select_kernel(caps: TargetCaps, facts: ScanFacts) -> KernelPlan {
    if facts.analysis.shape.is_empty() {
        return KernelPlan {
            kernel: Kernel::Empty,
            row_mapping: RowMapping::Linear,
            reserve: 0,
        };
    }

    let row_mapping =
        if facts.region.code_count == 0 || facts.covered_below(SPARSE_ROW_MAPPING_MAX_COVERAGE) {
            RowMapping::AdaptiveSparse
        } else {
            RowMapping::Linear
        };

    match caps {
        #[cfg(any(target_arch = "aarch64", test))]
        TargetCaps::Aarch64Neon => {
            let shape = facts.analysis.shape;
            let specialized = match (shape.points, shape.ranges) {
                (0, 1) | (1, 1) | (2, 1) | (1, 2) => {
                    Some(FixedShape::new(shape.points, shape.ranges))
                }
                (points, ranges) if ranges != 0 && points + 2 * ranges <= 16 => {
                    Some(FixedShape::new(points, ranges))
                }
                (1..=16, 0) => Some(FixedShape::new(shape.points, 0)),
                _ => None,
            };
            let paired_generic = specialized.is_none() && shape.points == 1 && shape.ranges != 0;
            KernelPlan {
                kernel: Kernel::Neon {
                    fixed: specialized,
                    two_vectors: paired_generic,
                },
                row_mapping,
                reserve: 0,
            }
        }
        #[cfg(any(target_arch = "x86_64", test))]
        TargetCaps::X86_64 { avx2, avx512bw } => {
            macro_rules! match_shapes {
                ($(($points:literal, $ranges:literal),)+) => {
                    match (facts.analysis.shape.points, facts.analysis.shape.ranges) {
                        $(($points, $ranges) => Some(FixedShape::new($points, $ranges)),)+
                        _ => None,
                    }
                };
            }

            let kernel = if avx512bw {
                Kernel::Avx512 {
                    fixed: with_x86_fixed_shapes!(match_shapes),
                }
            } else if avx2 {
                Kernel::Avx2 {
                    fixed: with_x86_fixed_shapes!(match_shapes),
                }
            } else {
                Kernel::Sse2 {
                    fixed: with_sse2_fixed_shapes!(match_shapes),
                }
            };
            KernelPlan {
                kernel,
                row_mapping,
                reserve: facts.region.row_count.min(facts.expected_covered_codes()),
            }
        }
        #[cfg(any(not(any(target_arch = "aarch64", target_arch = "x86_64")), test))]
        TargetCaps::Unsupported => KernelPlan {
            kernel: Kernel::Unsupported,
            row_mapping,
            reserve: 0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(points: usize, ranges: usize) -> ScanFacts {
        ScanFacts {
            analysis: AnalysisFacts {
                shape: CoverShape { points, ranges },
                covered_codes: 1,
                indexed_codes: 10_000,
            },
            region: RegionFacts {
                code_count: 10_000,
                row_count: 1_000,
            },
        }
    }

    fn x86(points: usize, ranges: usize, avx2: bool, avx512bw: bool) -> Kernel {
        select_kernel(TargetCaps::X86_64 { avx2, avx512bw }, facts(points, ranges)).kernel
    }

    #[test]
    fn empty_cover_precedes_target_selection() {
        let plan = select_kernel(TargetCaps::Unsupported, facts(0, 0));
        assert_eq!(plan.kernel, Kernel::Empty);
        assert_eq!(plan.reserve, 0);
    }

    #[test]
    fn sparse_mapping_boundary_is_strict() {
        let mut below = facts(1, 0);
        below.analysis.covered_codes = 0;
        let plan = select_kernel(TargetCaps::Aarch64Neon, below);
        assert_eq!(plan.row_mapping, RowMapping::AdaptiveSparse);
        assert_eq!(
            plan.kernel,
            Kernel::Neon {
                fixed: Some(FixedShape::new(1, 0)),
                two_vectors: false,
            }
        );
        assert_eq!(
            select_kernel(TargetCaps::Aarch64Neon, facts(1, 0)).row_mapping,
            RowMapping::Linear
        );
    }

    #[test]
    fn neon_shape_matrix_preserves_schedules() {
        let cases = [
            ((0, 1), Some(FixedShape::new(0, 1)), false),
            ((1, 1), Some(FixedShape::new(1, 1)), false),
            ((2, 1), Some(FixedShape::new(2, 1)), false),
            ((1, 2), Some(FixedShape::new(1, 2)), false),
            ((3, 2), Some(FixedShape::new(3, 2)), false),
            ((3, 0), Some(FixedShape::new(3, 0)), false),
            ((12, 0), Some(FixedShape::new(12, 0)), false),
            ((1, 8), None, true),
            ((17, 0), None, false),
        ];
        for ((points, ranges), fixed, two_vectors) in cases {
            assert_eq!(
                select_kernel(TargetCaps::Aarch64Neon, facts(points, ranges)).kernel,
                Kernel::Neon { fixed, two_vectors }
            );
        }
    }

    #[test]
    fn x86_preserves_original_shape_sets_and_best_available_isa() {
        const WIDE_FIXED: [(usize, usize); 24] = [
            (1, 0),
            (2, 0),
            (3, 0),
            (0, 1),
            (1, 1),
            (2, 1),
            (4, 0),
            (0, 2),
            (5, 0),
            (3, 1),
            (1, 2),
            (6, 0),
            (4, 1),
            (2, 2),
            (0, 3),
            (7, 0),
            (5, 1),
            (3, 2),
            (1, 3),
            (8, 0),
            (6, 1),
            (4, 2),
            (2, 3),
            (0, 4),
        ];
        const SSE2_FIXED: [(usize, usize); 18] = [
            (1, 0),
            (0, 1),
            (2, 0),
            (3, 0),
            (1, 1),
            (4, 0),
            (2, 1),
            (5, 0),
            (0, 2),
            (1, 2),
            (3, 1),
            (6, 0),
            (4, 1),
            (2, 2),
            (0, 3),
            (3, 2),
            (1, 3),
            (4, 2),
        ];

        for (points, ranges) in WIDE_FIXED {
            let wide = Some(FixedShape::new(points, ranges));
            let sse2 = SSE2_FIXED
                .contains(&(points, ranges))
                .then_some(FixedShape::new(points, ranges));
            assert_eq!(
                x86(points, ranges, false, false),
                Kernel::Sse2 { fixed: sse2 }
            );
            assert_eq!(
                x86(points, ranges, true, false),
                Kernel::Avx2 { fixed: wide }
            );
            assert_eq!(
                x86(points, ranges, true, true),
                Kernel::Avx512 { fixed: wide }
            );
        }

        for (points, ranges) in [(9, 0), (7, 1), (5, 2), (3, 3), (1, 4), (12, 0), (17, 0)] {
            assert_eq!(
                x86(points, ranges, false, false),
                Kernel::Sse2 { fixed: None }
            );
            assert_eq!(
                x86(points, ranges, true, false),
                Kernel::Avx2 { fixed: None }
            );
            assert_eq!(
                x86(points, ranges, true, true),
                Kernel::Avx512 { fixed: None }
            );
        }
    }

    #[test]
    fn expected_covered_codes_projects_subset_regions() {
        let input = ScanFacts {
            analysis: AnalysisFacts {
                shape: CoverShape {
                    points: 1,
                    ranges: 0,
                },
                covered_codes: 1_000,
                indexed_codes: 1_000_000,
            },
            region: RegionFacts {
                code_count: 2_000,
                row_count: 100,
            },
        };
        assert_eq!(input.expected_covered_codes(), 2);
        assert_eq!(
            select_kernel(
                TargetCaps::X86_64 {
                    avx2: true,
                    avx512bw: true,
                },
                input,
            )
            .reserve,
            2
        );
    }
}
