// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Inputs and concrete choices shared by planning and execution.

use super::super::ProbeCover;

pub(in crate::search::substring) const BLOCK: usize = 4096;
pub(in crate::search::substring) const PER_BATCH: usize = 8;

/// Targets also name coefficient sets in calibration files.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::search::substring) enum Isa {
    Scalar,
    Neon,
    Avx2,
    Avx512Bw,
}

/// A target whose implementation is compiled and available to the caller.
/// Production values come from scan dispatch; tests can supply synthetic targets.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::search::substring) struct TargetCaps {
    pub(in crate::search::substring) isa: Isa,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::search::substring) struct CoverShape {
    pub(in crate::search::substring) points: usize,
    pub(in crate::search::substring) ranges: usize,
}
impl CoverShape {
    pub(in crate::search::substring) fn of(cover: &ProbeCover) -> Self {
        Self {
            points: cover.points().len(),
            ranges: cover.ranges().len(),
        }
    }
    pub(super) fn is_empty(self) -> bool {
        self.points == 0 && self.ranges == 0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::search::substring) struct AnalysisFacts {
    pub(in crate::search::substring) shape: CoverShape,
    pub(in crate::search::substring) covered_codes: usize,
    pub(in crate::search::substring) indexed_codes: usize,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::search::substring) struct RegionFacts {
    pub(in crate::search::substring) code_count: usize,
    pub(in crate::search::substring) row_count: usize,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::search::substring) struct ScanFacts {
    pub(in crate::search::substring) analysis: AnalysisFacts,
    pub(in crate::search::substring) region: RegionFacts,
}
impl ScanFacts {
    /// Preserve the PR projection, including advisory weights and saturation.
    pub(in crate::search::substring) fn expected_covered_codes(self) -> usize {
        let AnalysisFacts {
            covered_codes,
            indexed_codes,
            ..
        } = self.analysis;
        if indexed_codes == self.region.code_count {
            return covered_codes;
        }
        if indexed_codes == 0 {
            return 0;
        }
        let projected = (covered_codes as u128).saturating_mul(self.region.code_count as u128)
            / indexed_codes as u128;
        usize::try_from(projected).unwrap_or(usize::MAX)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::search::substring) enum MatcherKind {
    Table,
    EqOr,
    Range,
    NibbleN8K,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::search::substring) enum ResolverKind {
    LinearSeek,
    GallopSeek,
}

/// Only vector operations occur inside a vector target's configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::search::substring) enum VectorMatcher {
    EqOr,
    Range,
    NibbleN8 { batches: usize },
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::search::substring) enum Kernel {
    Empty,
    Table,
    Neon { matcher: VectorMatcher, skip: bool },
    Avx2 { matcher: VectorMatcher, skip: bool },
    Avx512Bw { matcher: VectorMatcher, skip: bool },
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::search::substring) struct ScanPlan {
    pub(in crate::search::substring) kernel: Kernel,
    pub(in crate::search::substring) resolver: ResolverKind,
}
