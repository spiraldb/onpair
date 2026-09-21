// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Select matcher configurations from explicit facts and target capabilities.
//!
//! First restrict matchers to supported cover shapes, then compare their
//! relative scores. Estimated hit density determines whether empty groups skip
//! mask packing. `scan::dispatch` prepares and executes the selected matcher;
//! selection performs no scanning or feature detection.
//!
//! During preparation, `score_cover` ranks candidate covers using the
//! lowest eligible matcher score and a fixed penalty per covered token occurrence.

use super::super::{ProbeCover, scan::PER_BATCH};
use super::cost::{COVER_HIT_PENALTY, matcher_score, should_skip_packing};
use super::{
    CoverShape, Isa, MatcherConfig, MatcherKind, RegionFacts, ScanFacts, TargetCaps, VectorMatcher,
};

/// Whether this matcher supports the cover shape on the supplied target.
/// The caller supplies available capabilities. Nibble batches are limited to
/// two on AVX2 and three on NEON or AVX-512 to bound register use.
pub(in crate::search::substring) fn supports_matcher(
    caps: TargetCaps,
    matcher: MatcherKind,
    shape: CoverShape,
) -> bool {
    let vector = caps.isa != Isa::Scalar;
    match matcher {
        MatcherKind::Table => true,
        MatcherKind::EqOr => vector && shape.points > 0,
        MatcherKind::Range => vector && shape.points == 0 && shape.ranges > 0,
        MatcherKind::NibbleN8K => {
            vector
                && shape.points > 0
                && shape.points.div_ceil(PER_BATCH)
                    <= match caps.isa {
                        Isa::Avx2 => 2,
                        _ => 3,
                    }
        }
    }
}

/// Choose the eligible matcher with the lowest per-code score.
/// The scalar table is always eligible; equal scores retain the earlier choice.
pub(super) fn select_matcher(caps: TargetCaps, shape: CoverShape) -> MatcherKind {
    let mut best = MatcherKind::Table;
    let mut best_score = matcher_score(caps.isa, best, shape);

    for matcher in [
        MatcherKind::EqOr,
        MatcherKind::Range,
        MatcherKind::NibbleN8K,
    ] {
        if !supports_matcher(caps, matcher, shape) {
            continue;
        }

        let score = matcher_score(caps.isa, matcher, shape);
        if score.total_cmp(&best_score).is_lt() {
            best = matcher;
            best_score = score;
        }
    }

    best
}

/// Choose the matcher and empty-group packing policy.
/// An empty cover or region needs no kernel. Advisory hit counts are projected
/// to the current region before estimating packing savings.
pub(in crate::search::substring) fn select_matcher_config(
    caps: TargetCaps,
    facts: ScanFacts,
) -> MatcherConfig {
    if facts.analysis.shape.is_empty()
        || facts.region.row_count == 0
        || facts.region.code_count == 0
    {
        return MatcherConfig::Empty;
    }
    let expected_hits = facts.expected_covered_codes() as f64;
    let kind = select_matcher(caps, facts.analysis.shape);
    let matcher = match kind {
        MatcherKind::Table => return MatcherConfig::Table,
        MatcherKind::EqOr => VectorMatcher::EqOr,
        MatcherKind::Range => VectorMatcher::Range,
        MatcherKind::NibbleN8K => VectorMatcher::NibbleN8 {
            batches: facts.analysis.shape.points.div_ceil(PER_BATCH),
        },
    };
    let skip = should_skip_packing(caps.isa, expected_hits / facts.region.code_count as f64);
    match caps.isa {
        Isa::Neon => MatcherConfig::Neon { matcher, skip },
        Isa::Avx2 => MatcherConfig::Avx2 { matcher, skip },
        Isa::Avx512Bw => MatcherConfig::Avx512Bw { matcher, skip },
        Isa::Scalar => MatcherConfig::Table,
    }
}

/// Heuristic for comparing candidate covers for the same scan. Lower is better.
/// Combines the per-code matcher score with a fixed penalty per covered occurrence.
/// The mask-packing policy is chosen separately at execution.
/// An empty cover or region scores zero; callers handle empty patterns separately.
pub(in crate::search::substring) fn score_cover(
    caps: TargetCaps,
    cover: &ProbeCover,
    covered: u32,
    region: RegionFacts,
) -> f64 {
    if cover.is_empty() || region.code_count == 0 || region.row_count == 0 {
        return 0.0;
    }
    let shape = CoverShape::of(cover);
    let matcher = select_matcher(caps, shape);
    let scan_score = matcher_score(caps.isa, matcher, shape);
    region.code_count as f64 * scan_score + f64::from(covered) * COVER_HIT_PENALTY
}

#[cfg(test)]
mod tests {
    use super::super::AnalysisFacts;
    use super::*;

    /// Sample preparation and region sizes for deterministic selection tests.
    fn facts(points: usize, ranges: usize) -> ScanFacts {
        ScanFacts {
            analysis: AnalysisFacts {
                shape: CoverShape { points, ranges },
                covered_codes: 100,
                indexed_codes: 10000,
            },
            region: RegionFacts {
                code_count: 2000,
                row_count: 200,
            },
        }
    }

    #[test]
    fn empty_regions_and_covers_do_not_require_a_kernel() {
        for isa in [Isa::Scalar, Isa::Neon, Isa::Avx2, Isa::Avx512Bw] {
            for f in [
                facts(0, 0),
                ScanFacts {
                    region: RegionFacts {
                        code_count: 0,
                        row_count: 5,
                    },
                    ..facts(1, 0)
                },
                ScanFacts {
                    region: RegionFacts {
                        code_count: 0,
                        row_count: 0,
                    },
                    ..facts(1, 0)
                },
            ] {
                assert_eq!(
                    select_matcher_config(TargetCaps { isa }, f),
                    MatcherConfig::Empty
                );
            }
        }
    }

    #[test]
    fn target_admission_respects_nibble_register_budgets() {
        for (isa, limit) in [
            (Isa::Scalar, 0),
            (Isa::Neon, 24),
            (Isa::Avx2, 16),
            (Isa::Avx512Bw, 24),
        ] {
            for points in 0..=32 {
                assert_eq!(
                    supports_matcher(
                        TargetCaps { isa },
                        MatcherKind::NibbleN8K,
                        CoverShape { points, ranges: 0 }
                    ),
                    points > 0 && points <= limit
                );
            }
        }
    }

    #[test]
    fn configurations_use_the_requested_target_and_legal_shapes() {
        for isa in [Isa::Scalar, Isa::Neon, Isa::Avx2, Isa::Avx512Bw] {
            for points in 0..=32 {
                for ranges in 0..=4 {
                    let config = select_matcher_config(TargetCaps { isa }, facts(points, ranges));
                    match config {
                        MatcherConfig::Empty => assert_eq!((points, ranges), (0, 0)),
                        MatcherConfig::Table => {}
                        MatcherConfig::Neon { .. } => assert_eq!(isa, Isa::Neon),
                        MatcherConfig::Avx2 { .. } => assert_eq!(isa, Isa::Avx2),
                        MatcherConfig::Avx512Bw { skip, .. } => {
                            assert_eq!(isa, Isa::Avx512Bw);
                            assert!(!skip);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn regional_projection_preserves_advisory_counts() {
        let mut f = facts(1, 0);
        assert_eq!(f.expected_covered_codes(), 20);
        f.region.code_count = 10000;
        assert_eq!(f.expected_covered_codes(), 100);
        f.analysis.indexed_codes = 0;
        assert_eq!(f.expected_covered_codes(), 0);
        f.analysis.indexed_codes = 1;
        f.analysis.covered_codes = 1;
        f.region.code_count = usize::MAX;
        assert_eq!(f.expected_covered_codes(), usize::MAX);
    }
}
