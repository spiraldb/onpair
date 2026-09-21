// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cover statistics, region sizes, and capabilities supplied to planning.
//!
//! Analysis facts describe the prepared cover and its frequency index. Region
//! facts describe the buffers being scanned, which may be a smaller region.
//! Target capabilities name a compiled kernel family available on this CPU.
//!
//! Selection combines these inputs into a `MatcherConfig`. Dispatch prepares
//! the corresponding matcher; these facts perform no CPU detection or scanning.

use super::super::ProbeCover;

/// Instruction-set family used for kernel selection and cost coefficients.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)] // Some variants are only constructed on other build targets.
pub(in crate::search::substring) enum Isa {
    Scalar,
    Neon,
    Avx2,
    Avx512Bw,
}

/// A kernel family compiled into this build and supported by the CPU.
/// Dispatch detects production capabilities; planning tests can supply them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::search::substring) struct TargetCaps {
    pub(in crate::search::substring) isa: Isa,
}

/// Number of point probes and ranges after cover normalization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::search::substring) struct CoverShape {
    pub(in crate::search::substring) points: usize,
    pub(in crate::search::substring) ranges: usize,
}
impl CoverShape {
    /// Read the normalized cover shape.
    pub(in crate::search::substring) fn of(cover: &ProbeCover) -> Self {
        Self {
            points: cover.points().len(),
            ranges: cover.ranges().len(),
        }
    }

    /// Whether neither kind of probe is present.
    pub(super) fn is_empty(self) -> bool {
        self.points == 0 && self.ranges == 0
    }
}

/// Cover shape and advisory counts from the preparation frequency index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::search::substring) struct AnalysisFacts {
    pub(in crate::search::substring) shape: CoverShape,
    /// Indexed occurrences of tokens in the cover.
    pub(in crate::search::substring) covered_codes: usize,
    /// Total code count represented by the frequency index.
    pub(in crate::search::substring) indexed_codes: usize,
}

/// Size of the code stream and number of rows being planned or scanned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::search::substring) struct RegionFacts {
    pub(in crate::search::substring) code_count: usize,
    pub(in crate::search::substring) row_count: usize,
}

/// Prepared analysis paired with the current scan region.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::search::substring) struct ScanFacts {
    pub(in crate::search::substring) analysis: AnalysisFacts,
    pub(in crate::search::substring) region: RegionFacts,
}
impl ScanFacts {
    /// Project indexed covered-code counts onto this region by its relative size.
    /// Equal-sized regions retain the original count; an empty index projects to
    /// zero. The result is an estimate used for costs, not a filter on matches.
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
