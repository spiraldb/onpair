// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Select matcher configurations from cover shape, hit density, and instruction set.
//!
//! First restrict matchers to supported cover shapes, then compare their
//! relative scores. Estimated hit density determines whether empty groups skip
//! mask packing. `scan::dispatch` prepares and executes the selected matcher;
//! selection performs no scanning or feature detection.
//!
//! During preparation, `score_cover` ranks candidate covers using the
//! lowest eligible matcher score and a fixed penalty per covered token occurrence.

use super::super::ProbeCover;
use super::super::scan::{Isa, MatcherConfig, MatcherKind, PER_BATCH};
use super::score::{COVER_HIT_PENALTY, matcher_score};

/// Codes in the pair of mask words tested together before packing.
const PACK_GROUP: f64 = 128.0;

/// Whether this matcher supports the cover shape on the supplied target.
/// The caller supplies an available instruction set. Nibble batches are limited to
/// two on AVX2 and three on NEON or AVX-512 to bound register use.
pub(in crate::search::substring) fn supports_matcher(
    isa: Isa,
    matcher: MatcherKind,
    cover: &ProbeCover,
) -> bool {
    let vector = isa != Isa::Scalar;
    match matcher {
        MatcherKind::Table => true,
        MatcherKind::EqOr => vector && cover.n_points() > 0,
        MatcherKind::Range => vector && cover.n_points() == 0 && cover.n_ranges() > 0,
        MatcherKind::NibbleN8 => {
            vector
                && cover.n_points() > 0
                && cover.n_points().div_ceil(PER_BATCH)
                    <= match isa {
                        Isa::Avx2 => 2,
                        _ => 3,
                    }
        }
    }
}

/// Choose the eligible matcher with the lowest per-code score.
/// The scalar table is always eligible; equal scores retain the earlier choice.
pub(super) fn select_matcher(isa: Isa, cover: &ProbeCover) -> MatcherKind {
    let mut best = MatcherKind::Table;
    let mut best_score = matcher_score(isa, best, cover);

    for matcher in [MatcherKind::EqOr, MatcherKind::Range, MatcherKind::NibbleN8] {
        if !supports_matcher(isa, matcher, cover) {
            continue;
        }

        let score = matcher_score(isa, matcher, cover);
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

/// Choose the matcher and empty-group packing policy for a nonempty scan and cover.
/// The caller handles empty inputs and supplies the estimated hit density.
pub(in crate::search::substring) fn select_matcher_config(
    isa: Isa,
    cover: &ProbeCover,
    probe_density: f64,
) -> MatcherConfig {
    let kind = select_matcher(isa, cover);
    MatcherConfig {
        kind,
        skip_empty_packing: kind != MatcherKind::Table && should_skip_packing(isa, probe_density),
    }
}

/// Whether testing for empty groups is expected to save packing work.
/// Balances a reduction on every group against packing saved on empty groups.
/// `density` is the estimated fraction of codes covered by the probes.
fn should_skip_packing(isa: Isa, density: f64) -> bool {
    let (reduction, pack) = match isa {
        // AVX-512 already produces a mask: skipping the pack only adds work.
        Isa::Avx512Bw => (0.35, 0.0),
        _ => (0.75, 1.01),
    };
    // Poisson estimate of empty groups. Clustered hits can change the savings.
    let no_match = (-PACK_GROUP * density).exp();
    (reduction - pack * no_match) / PACK_GROUP < 0.0
}

/// Heuristic for comparing candidate covers for the same scan. Lower is better.
/// Combines the per-code matcher score with a fixed penalty per covered occurrence.
/// The mask-packing policy is chosen separately at execution.
/// An empty cover or index scores zero; callers handle empty patterns separately.
pub(in crate::search::substring) fn score_cover(
    isa: Isa,
    cover: &ProbeCover,
    covered: u32,
    code_count: u32,
) -> f64 {
    if cover.is_empty() || code_count == 0 {
        return 0.0;
    }
    let matcher = select_matcher(isa, cover);
    let scan_score = matcher_score(isa, matcher, cover);
    f64::from(code_count) * scan_score + f64::from(covered) * COVER_HIT_PENALTY
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{Token, TokenRange};

    /// Build separated points and ranges with the requested probe counts.
    fn cover(points: usize, ranges: usize) -> ProbeCover {
        ProbeCover::new(
            (0..points).map(|i| (2 * i) as Token).collect(),
            (0..ranges)
                .map(|i| TokenRange {
                    begin: (128 + 4 * i) as Token,
                    last: (129 + 4 * i) as Token,
                })
                .collect(),
        )
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
                    supports_matcher(isa, MatcherKind::NibbleN8, &cover(points, 0)),
                    points > 0 && points <= limit
                );
            }
        }
    }

    #[test]
    fn configurations_use_eligible_matchers_and_packing() {
        for isa in [Isa::Scalar, Isa::Neon, Isa::Avx2, Isa::Avx512Bw] {
            for points in 0..=32 {
                for ranges in 0..=4 {
                    let cover = cover(points, ranges);
                    if cover.is_empty() {
                        continue;
                    }
                    for density in [0.0, 0.01, 1.0] {
                        let config = select_matcher_config(isa, &cover, density);
                        assert!(supports_matcher(isa, config.kind, &cover));
                        if config.kind == MatcherKind::Table || isa == Isa::Avx512Bw {
                            assert!(!config.skip_empty_packing);
                        } else {
                            assert_eq!(config.skip_empty_packing, density == 0.0);
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
