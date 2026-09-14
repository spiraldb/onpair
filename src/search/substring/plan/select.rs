// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pure selection with PR's matcher, resolver and packing policy.

use super::cost::{hit_rows, ns_per_code, seek_ns_per_row, skip_ns_per_code};
use super::facts::*;

pub(in crate::search::substring) fn takes(
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

pub(super) fn select_matcher(caps: TargetCaps, shape: CoverShape) -> MatcherKind {
    [
        MatcherKind::Table,
        MatcherKind::EqOr,
        MatcherKind::Range,
        MatcherKind::NibbleN8K,
    ]
    .into_iter()
    .filter(|&matcher| takes(caps, matcher, shape))
    .min_by(|&a, &b| ns_per_code(caps.isa, a, shape).total_cmp(&ns_per_code(caps.isa, b, shape)))
    .expect("the byte table takes every cover")
}

pub(in crate::search::substring) fn select_scan_plan(
    caps: TargetCaps,
    facts: ScanFacts,
) -> ScanPlan {
    if facts.analysis.shape.is_empty()
        || facts.region.row_count == 0
        || facts.region.code_count == 0
    {
        return ScanPlan {
            kernel: Kernel::Empty,
            resolver: ResolverKind::LinearSeek,
        };
    }
    let rows = facts.region.row_count as f64;
    let expected_hits = facts.expected_covered_codes() as f64;
    let g = rows / hit_rows(expected_hits, rows).max(1.0);
    let resolver = if seek_ns_per_row(ResolverKind::GallopSeek, g)
        < seek_ns_per_row(ResolverKind::LinearSeek, g)
    {
        ResolverKind::GallopSeek
    } else {
        ResolverKind::LinearSeek
    };
    let kind = select_matcher(caps, facts.analysis.shape);
    let matcher = match kind {
        MatcherKind::Table => {
            return ScanPlan {
                kernel: Kernel::Table,
                resolver,
            };
        }
        MatcherKind::EqOr => VectorMatcher::EqOr,
        MatcherKind::Range => VectorMatcher::Range,
        MatcherKind::NibbleN8K => VectorMatcher::NibbleN8 {
            batches: facts.analysis.shape.points.div_ceil(PER_BATCH),
        },
    };
    let skip = skip_ns_per_code(caps.isa, expected_hits / facts.region.code_count as f64) < 0.0;
    let kernel = match caps.isa {
        Isa::Neon => Kernel::Neon { matcher, skip },
        Isa::Avx2 => Kernel::Avx2 { matcher, skip },
        Isa::Avx512Bw => Kernel::Avx512Bw { matcher, skip },
        Isa::Scalar => unreachable!("scalar selection admits only the table"),
    };
    ScanPlan { kernel, resolver }
}

#[cfg(test)]
mod tests {
    use super::*;

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
                    select_scan_plan(TargetCaps { isa }, f).kernel,
                    Kernel::Empty
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
                    takes(
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
    fn plans_use_the_requested_target_and_legal_shapes() {
        for isa in [Isa::Scalar, Isa::Neon, Isa::Avx2, Isa::Avx512Bw] {
            for points in 0..=32 {
                for ranges in 0..=4 {
                    let plan = select_scan_plan(TargetCaps { isa }, facts(points, ranges));
                    match plan.kernel {
                        Kernel::Empty => assert_eq!((points, ranges), (0, 0)),
                        Kernel::Table => {}
                        Kernel::Neon { .. } => assert_eq!(isa, Isa::Neon),
                        Kernel::Avx2 { .. } => assert_eq!(isa, Isa::Avx2),
                        Kernel::Avx512Bw { skip, .. } => {
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
