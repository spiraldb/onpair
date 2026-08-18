// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pure, host-independent scan-kernel selection.

/// Invoke a local macro with every measured SSE2 fixed shape. Policy and
/// execution both expand this list, so adding or removing a specialization
/// cannot leave the two sides out of sync.
#[cfg(any(target_arch = "x86_64", test))]
macro_rules! with_sse2_fixed_shapes {
    ($apply:ident) => {
        $apply! {
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

/// Invoke a local macro with every measured AVX2 fixed shape.
#[cfg(any(target_arch = "x86_64", test))]
macro_rules! with_avx2_fixed_shapes {
    ($apply:ident) => {
        $apply! {
            (2, 0),
            (3, 0),
            (1, 1),
            (2, 1),
            (1, 2),
            (4, 0),
            (5, 0),
            (6, 0),
            (0, 2),
            (3, 1),
            (4, 1),
            (3, 2),
            (2, 2),
            (0, 3),
            (1, 3),
            (2, 3),
            (5, 1),
            (4, 2),
            (6, 1),
            (3, 3),
            (5, 2),
            (7, 1),
            (6, 2),
            (8, 1),
        }
    };
}

#[cfg(any(target_arch = "x86_64", test))]
pub(super) use {with_avx2_fixed_shapes, with_sse2_fixed_shapes};

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

    #[cfg(any(target_arch = "x86_64", test))]
    #[inline]
    pub(super) const fn comparison_cost(self) -> usize {
        self.points.saturating_add(self.ranges.saturating_mul(2))
    }
}

/// Stable facts produced by prefilter analysis.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct AnalysisFacts {
    pub(super) shape: CoverShape,
    pub(super) table_len: usize,
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
    #[cfg(any(target_arch = "x86_64", test))]
    #[inline]
    pub(super) const fn average_row_len(self) -> usize {
        if self.region.row_count == 0 {
            0
        } else {
            self.region.code_count / self.region.row_count
        }
    }

    /// Expected covered codes in this region.
    ///
    /// This is exact for the indexed population and a policy-only projection
    /// for a subset. It never affects correctness.
    #[cfg(any(target_arch = "x86_64", test))]
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

    #[cfg(any(target_arch = "x86_64", test))]
    #[inline]
    pub(super) fn covered_at_least(self, ratio: Ratio) -> bool {
        ratio.at_most_fraction(self.analysis.covered_codes, self.analysis.indexed_codes)
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

    #[cfg(any(target_arch = "x86_64", test))]
    #[inline]
    fn at_most_fraction(self, numerator: usize, denominator: usize) -> bool {
        (numerator as u128).saturating_mul(self.denominator as u128)
            >= (denominator as u128).saturating_mul(self.numerator as u128)
    }

    #[inline]
    fn above_fraction(self, numerator: usize, denominator: usize) -> bool {
        (numerator as u128).saturating_mul(self.denominator as u128)
            < (denominator as u128).saturating_mul(self.numerator as u128)
    }
}

const SPARSE_ROW_MAPPING_MAX_COVERAGE: Ratio = Ratio::new(1, 10_000);
#[cfg(any(target_arch = "x86_64", test))]
const COMPACT_ONE_POINT_MIN_COVERAGE: Ratio = Ratio::new(7, 10_000);
#[cfg(any(target_arch = "x86_64", test))]
const COMPACT_FEW_HITS_MIN_COVERAGE: Ratio = Ratio::new(1, 2_000);
#[cfg(any(target_arch = "x86_64", test))]
const NIBBLE_SIX_MAX_COVERAGE: Ratio = Ratio::new(1, 100);
#[cfg(any(target_arch = "x86_64", test))]
const WIDE_COVER_ROW_TABLE_MIN_COVERAGE: Ratio = Ratio::new(1, 20);
#[cfg(any(target_arch = "x86_64", test))]
const LONG_ROW_TABLE_MIN_COVERAGE: Ratio = Ratio::new(3, 100);

#[cfg(any(target_arch = "x86_64", test))]
const SMALL_TABLE_LEN: usize = 1 << 12;
#[cfg(any(target_arch = "x86_64", test))]
const SMALL_TABLE_MAX_COMPARE_COST: usize = 10;
#[cfg(any(target_arch = "x86_64", test))]
const LARGE_TABLE_MAX_COMPARE_COST: usize = 13;
#[cfg(any(target_arch = "x86_64", test))]
const WIDE_COVER_COMPARE_COST: usize = 17;
#[cfg(any(target_arch = "x86_64", test))]
const LONG_ROW_CODES: usize = 32;
#[cfg(any(target_arch = "x86_64", test))]
const MEDIUM_ROW_CODES: usize = 8;

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg(any(target_arch = "x86_64", test))]
pub(super) enum HitMaterialization {
    StoredLanes,
    CompactMask,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg(any(target_arch = "aarch64", test))]
pub(super) enum NeonKernel {
    FewPoints { points: usize },
    ManyPoints,
    OneRange,
    Fixed(FixedShape),
    OnePointTwoRanges,
    FewMixed,
    Generic { two_vectors: bool },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg(any(target_arch = "x86_64", test))]
pub(super) enum Sse2Kernel {
    OnePoint,
    Fixed(FixedShape),
    CodesTable,
    Generic,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg(any(target_arch = "x86_64", test))]
pub(super) enum Avx2Kernel {
    OnePoint { hits: HitMaterialization },
    OneRange,
    Fixed(FixedShape),
    NibblePoints,
    Few { hits: HitMaterialization },
    Gather,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg(any(target_arch = "x86_64", test))]
pub(super) enum Avx512Kernel {
    Generic,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Kernel {
    Empty,
    #[cfg(any(target_arch = "x86_64", test))]
    RowsTable,
    #[cfg(any(target_arch = "aarch64", test))]
    Neon(NeonKernel),
    #[cfg(any(target_arch = "x86_64", test))]
    Sse2(Sse2Kernel),
    #[cfg(any(target_arch = "x86_64", test))]
    Avx2(Avx2Kernel),
    #[cfg(any(target_arch = "x86_64", test))]
    Avx512(Avx512Kernel),
    #[cfg(any(not(any(target_arch = "aarch64", target_arch = "x86_64")), test))]
    Unsupported,
}

/// Complete, ephemeral execution plan for one scan region.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::search::prefilter) struct KernelPlan {
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
        TargetCaps::Aarch64Neon => KernelPlan {
            kernel: Kernel::Neon(select_neon(facts.analysis.shape)),
            row_mapping,
            reserve: 0,
        },
        #[cfg(any(target_arch = "x86_64", test))]
        TargetCaps::X86_64 { avx2, avx512bw } => select_x86_64(facts, row_mapping, avx2, avx512bw),
        #[cfg(any(not(any(target_arch = "aarch64", target_arch = "x86_64")), test))]
        TargetCaps::Unsupported => KernelPlan {
            kernel: Kernel::Unsupported,
            row_mapping,
            reserve: 0,
        },
    }
}

#[cfg(any(target_arch = "aarch64", test))]
#[inline]
pub(super) fn select_neon(shape: CoverShape) -> NeonKernel {
    match (shape.points, shape.ranges) {
        (0, 1) => NeonKernel::OneRange,
        (1, 1) | (2, 1) => NeonKernel::Fixed(FixedShape::new(shape.points, shape.ranges)),
        (1, 2) => NeonKernel::OnePointTwoRanges,
        (points, ranges) if ranges != 0 && points + 2 * ranges <= 16 => NeonKernel::FewMixed,
        (1..=8, 0) => NeonKernel::FewPoints {
            points: shape.points,
        },
        (9..=16, 0) => NeonKernel::ManyPoints,
        _ => NeonKernel::Generic {
            two_vectors: (shape.points == 1 && shape.ranges != 0)
                || (shape.points == 2 && shape.ranges == 1),
        },
    }
}

#[cfg(any(target_arch = "x86_64", test))]
#[inline]
fn select_x86_64(
    facts: ScanFacts,
    row_mapping: RowMapping,
    avx2: bool,
    avx512bw: bool,
) -> KernelPlan {
    let reserve = facts.region.row_count.min(facts.expected_covered_codes());
    let kernel = if avx512bw {
        Kernel::Avx512(Avx512Kernel::Generic)
    } else if avx2 {
        select_avx2(facts)
    } else if use_sse2_rows_table(facts) {
        Kernel::RowsTable
    } else {
        Kernel::Sse2(select_sse2(facts.analysis.shape))
    };
    KernelPlan {
        kernel,
        row_mapping,
        reserve,
    }
}

#[cfg(any(target_arch = "x86_64", test))]
fn max_compare_cost(table_len: usize) -> usize {
    if table_len <= SMALL_TABLE_LEN {
        SMALL_TABLE_MAX_COMPARE_COST
    } else {
        LARGE_TABLE_MAX_COMPARE_COST
    }
}

#[cfg(any(target_arch = "x86_64", test))]
fn base_rows_table(facts: ScanFacts) -> bool {
    facts.region.row_count != 0
        && facts.average_row_len() >= LONG_ROW_CODES
        && facts.expected_covered_codes() >= facts.region.row_count
}

#[cfg(any(target_arch = "x86_64", test))]
fn use_sse2_rows_table(facts: ScanFacts) -> bool {
    let cost = facts.analysis.shape.comparison_cost();
    (base_rows_table(facts) && cost > max_compare_cost(facts.analysis.table_len))
        || (cost >= WIDE_COVER_COMPARE_COST
            && (facts.covered_at_least(WIDE_COVER_ROW_TABLE_MIN_COVERAGE)
                || (facts.region.row_count != 0
                    && facts.average_row_len() >= MEDIUM_ROW_CODES
                    && facts.covered_at_least(LONG_ROW_TABLE_MIN_COVERAGE))))
}

#[cfg(any(target_arch = "x86_64", test))]
#[inline]
fn select_sse2(shape: CoverShape) -> Sse2Kernel {
    if shape
        == (CoverShape {
            points: 1,
            ranges: 0,
        })
    {
        return Sse2Kernel::OnePoint;
    }
    if is_sse2_fixed(shape) {
        return Sse2Kernel::Fixed(FixedShape::new(shape.points, shape.ranges));
    }
    if shape.comparison_cost() >= WIDE_COVER_COMPARE_COST {
        Sse2Kernel::CodesTable
    } else {
        Sse2Kernel::Generic
    }
}

#[cfg(any(target_arch = "x86_64", test))]
fn is_sse2_fixed(shape: CoverShape) -> bool {
    macro_rules! matches_shape {
        ($(($points:literal, $ranges:literal),)+) => {
            matches!((shape.points, shape.ranges), $(($points, $ranges))|+)
        };
    }
    with_sse2_fixed_shapes!(matches_shape)
}

#[cfg(any(target_arch = "x86_64", test))]
#[inline]
fn select_avx2(facts: ScanFacts) -> Kernel {
    let shape = facts.analysis.shape;
    if base_rows_table(facts)
        && shape.comparison_cost() > max_compare_cost(facts.analysis.table_len)
    {
        return Kernel::RowsTable;
    }

    let leaf = match (shape.points, shape.ranges) {
        (9..=16, 0) if facts.analysis.table_len <= SMALL_TABLE_LEN => Avx2Kernel::NibblePoints,
        (10..=16, 0) => Avx2Kernel::NibblePoints,
        (1, 0) => Avx2Kernel::OnePoint {
            hits: if facts.covered_at_least(COMPACT_ONE_POINT_MIN_COVERAGE) {
                HitMaterialization::CompactMask
            } else {
                HitMaterialization::StoredLanes
            },
        },
        (0, 1) => Avx2Kernel::OneRange,
        (6, 0)
            if facts.analysis.table_len <= SMALL_TABLE_LEN
                && facts.covered_below(NIBBLE_SIX_MAX_COVERAGE) =>
        {
            Avx2Kernel::NibblePoints
        }
        (7..=8, 0) => Avx2Kernel::NibblePoints,
        _ if is_avx2_fixed(shape) => Avx2Kernel::Fixed(FixedShape::new(shape.points, shape.ranges)),
        _ if shape.comparison_cost() <= max_compare_cost(facts.analysis.table_len) => {
            Avx2Kernel::Few {
                hits: if facts.covered_at_least(COMPACT_FEW_HITS_MIN_COVERAGE) {
                    HitMaterialization::CompactMask
                } else {
                    HitMaterialization::StoredLanes
                },
            }
        }
        _ => Avx2Kernel::Gather,
    };
    Kernel::Avx2(leaf)
}

#[cfg(any(target_arch = "x86_64", test))]
fn is_avx2_fixed(shape: CoverShape) -> bool {
    macro_rules! matches_shape {
        ($(($points:literal, $ranges:literal),)+) => {
            matches!((shape.points, shape.ranges), $(($points, $ranges))|+)
        };
    }
    with_avx2_fixed_shapes!(matches_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(points: usize, ranges: usize) -> ScanFacts {
        ScanFacts {
            analysis: AnalysisFacts {
                shape: CoverShape { points, ranges },
                table_len: 4096,
                covered_codes: 1,
                indexed_codes: 10_000,
            },
            region: RegionFacts {
                code_count: 10_000,
                row_count: 1_000,
            },
        }
    }

    #[test]
    fn empty_cover_precedes_target_selection() {
        let plan = select_kernel(TargetCaps::Unsupported, facts(0, 0));
        assert_eq!(plan.kernel, Kernel::Empty);
    }

    #[test]
    fn sparse_mapping_boundary_is_strict() {
        let mut below = facts(1, 0);
        below.analysis.covered_codes = 0;
        assert_eq!(
            select_kernel(TargetCaps::Aarch64Neon, below).row_mapping,
            RowMapping::AdaptiveSparse
        );
        let at = facts(1, 0);
        assert_eq!(
            select_kernel(TargetCaps::Aarch64Neon, at).row_mapping,
            RowMapping::Linear
        );
    }

    #[test]
    fn neon_shape_matrix_preserves_specializations() {
        let cases = [
            ((0, 1), NeonKernel::OneRange),
            ((1, 1), NeonKernel::Fixed(FixedShape::new(1, 1))),
            ((2, 1), NeonKernel::Fixed(FixedShape::new(2, 1))),
            ((1, 2), NeonKernel::OnePointTwoRanges),
            ((3, 2), NeonKernel::FewMixed),
            ((3, 0), NeonKernel::FewPoints { points: 3 }),
            ((12, 0), NeonKernel::ManyPoints),
            ((1, 8), NeonKernel::Generic { two_vectors: true }),
            ((17, 0), NeonKernel::Generic { two_vectors: false }),
        ];
        for ((points, ranges), expected) in cases {
            assert_eq!(select_neon(CoverShape { points, ranges }), expected);
        }
    }

    #[test]
    fn avx512_keeps_priority_over_other_x86_paths() {
        let mut input = facts(20, 0);
        input.analysis.covered_codes = 8_000;
        input.region.row_count = 10;
        assert_eq!(
            select_kernel(
                TargetCaps::X86_64 {
                    avx2: true,
                    avx512bw: true,
                },
                input,
            )
            .kernel,
            Kernel::Avx512(Avx512Kernel::Generic)
        );
    }

    #[test]
    fn avx2_row_table_boundary_matches_upstream() {
        let mut input = facts(11, 0);
        input.region = RegionFacts {
            code_count: 32_000,
            row_count: 1_000,
        };
        input.analysis.indexed_codes = 32_000;
        input.analysis.covered_codes = 999;
        assert!(matches!(select_avx2(input), Kernel::Avx2(_)));
        input.analysis.covered_codes = 1_000;
        assert_eq!(select_avx2(input), Kernel::RowsTable);
    }

    #[test]
    fn sse2_wide_cover_uses_code_or_row_table_at_density_boundaries() {
        let mut input = facts(17, 0);
        input.region.row_count = 2_000;
        input.analysis.covered_codes = 499;
        assert_eq!(
            select_kernel(
                TargetCaps::X86_64 {
                    avx2: false,
                    avx512bw: false,
                },
                input,
            )
            .kernel,
            Kernel::Sse2(Sse2Kernel::CodesTable)
        );
        input.analysis.covered_codes = 500;
        assert_eq!(
            select_kernel(
                TargetCaps::X86_64 {
                    avx2: false,
                    avx512bw: false,
                },
                input,
            )
            .kernel,
            Kernel::RowsTable
        );
    }

    #[test]
    fn projected_region_count_does_not_reuse_global_absolute_count() {
        let input = ScanFacts {
            analysis: AnalysisFacts {
                shape: CoverShape {
                    points: 1,
                    ranges: 0,
                },
                table_len: 1024,
                covered_codes: 1_000,
                indexed_codes: 1_000_000,
            },
            region: RegionFacts {
                code_count: 2_000,
                row_count: 100,
            },
        };
        assert_eq!(input.expected_covered_codes(), 2);
    }
}
