// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Select matcher configurations from cover shape, hit density, and CPU capabilities.
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
use super::{CoverShape, Isa, MatcherConfig, MatcherKind, TargetCaps, VectorMatcher};

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

/// Estimate hit density after projecting indexed counts onto the current scan.
/// Round the projected hit count down before dividing by the scan's code count.
/// Equal-sized scans retain the indexed count; empty scans or indexes return zero.
/// Counts are advisory and affect packing only, never which tokens can match.
#[inline]
pub(in crate::search::substring) fn probe_density(
    covered_codes: usize,
    indexed_codes: usize,
    code_count: usize,
) -> f64 {
    if code_count == 0 || indexed_codes == 0 {
        return 0.0;
    }
    let expected_hits = if indexed_codes == code_count {
        covered_codes
    } else {
        let projected =
            (covered_codes as u128).saturating_mul(code_count as u128) / indexed_codes as u128;
        usize::try_from(projected).unwrap_or(usize::MAX)
    };
    expected_hits as f64 / code_count as f64
}

/// Choose the matcher and empty-group packing policy for a nonempty scan.
/// An empty cover needs no kernel. The caller supplies the estimated hit density
/// and handles empty code and row buffers before selection.
pub(in crate::search::substring) fn select_matcher_config(
    caps: TargetCaps,
    shape: CoverShape,
    probe_density: f64,
) -> MatcherConfig {
    if shape.is_empty() {
        return MatcherConfig::Empty;
    }
    let kind = select_matcher(caps, shape);
    let matcher = match kind {
        MatcherKind::Table => return MatcherConfig::Table,
        MatcherKind::EqOr => VectorMatcher::EqOr,
        MatcherKind::Range => VectorMatcher::Range,
        MatcherKind::NibbleN8K => VectorMatcher::NibbleN8 {
            batches: shape.points.div_ceil(PER_BATCH),
        },
    };
    let skip = should_skip_packing(caps.isa, probe_density);
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
/// An empty cover or index scores zero; callers handle empty patterns separately.
pub(in crate::search::substring) fn score_cover(
    caps: TargetCaps,
    cover: &ProbeCover,
    covered: u32,
    code_count: u32,
) -> f64 {
    if cover.is_empty() || code_count == 0 {
        return 0.0;
    }
    let shape = CoverShape::of(cover);
    let matcher = select_matcher(caps, shape);
    let scan_score = matcher_score(caps.isa, matcher, shape);
    f64::from(code_count) * scan_score + f64::from(covered) * COVER_HIT_PENALTY
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_covers_do_not_require_a_kernel() {
        for isa in [Isa::Scalar, Isa::Neon, Isa::Avx2, Isa::Avx512Bw] {
            assert_eq!(
                select_matcher_config(
                    TargetCaps { isa },
                    CoverShape {
                        points: 0,
                        ranges: 0
                    },
                    0.0
                ),
                MatcherConfig::Empty
            );
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
                    let config = select_matcher_config(
                        TargetCaps { isa },
                        CoverShape { points, ranges },
                        0.01,
                    );
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
    fn probe_density_preserves_projected_hit_counts() {
        for (covered, indexed, scanned, expected) in [
            (100, 10_000, 2_000, 0.01),
            (100, 10_000, 10_000, 0.01),
            (100, 0, 10_000, 0.0),
            (100, 10_000, 0, 0.0),
            (0, 0, 0, 0.0),
            (1, 1, usize::MAX, 1.0),
            (usize::MAX, 1, usize::MAX, 1.0),
            (200, 100, 100, 2.0),
        ] {
            assert_eq!(probe_density(covered, indexed, scanned), expected);
        }

        // Rounding a partial scan down to zero hits changes its packing decision.
        let whole = probe_density(1, 400, 400);
        let partial = probe_density(1, 400, 399);
        assert_eq!(whole, 1.0 / 400.0);
        assert_eq!(partial, 0.0);
        assert!(!should_skip_packing(Isa::Neon, whole));
        assert!(should_skip_packing(Isa::Neon, partial));
    }
}
