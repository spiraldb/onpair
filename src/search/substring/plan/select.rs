// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Select matcher configurations from cover shape, hit density, and instruction set.
//!
//! First restrict matchers to supported cover shapes, then compare their
//! relative costs. A shared hit-density threshold determines whether empty groups
//! skip mask packing. `scan::dispatch` prepares and executes the selected matcher;
//! selection performs no scanning or feature detection.
//!
//! During preparation, `cover_cost` ranks candidate covers using scan costs
//! balanced against a fixed penalty per covered token occurrence. The same
//! matcher costs choose the kernel that will execute each cover.

use super::super::ProbeCover;
use super::super::scan::{Isa, MatcherConfig, MatcherKind, PER_BATCH};
use super::cost::{candidate_cost, matcher_cost};

/// Probe density below which vector matchers skip packing empty groups.
/// This conservative empirical cutoff (0.0075%) is shared by NEON, AVX2 and
/// AVX-512. Sparse hits also leave whole blocks empty, avoiding row resolution.
/// Clustered hits can make skipping profitable above the cutoff too.
const SKIP_PACKING_DENSITY_THRESHOLD: f64 = 7.5e-5;

/// Whether this matcher is eligible for selection on the supplied target and cover.
/// The caller supplies an available instruction set. Nibble batches are limited to
/// two on AVX2 and three on NEON or AVX-512 to bound register use.
pub(in crate::search::substring) fn is_eligible(
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

/// Choose the eligible matcher with the lowest cost.
/// The scalar table is always eligible; equal costs retain the earlier choice.
fn select_matcher(isa: Isa, cover: &ProbeCover) -> MatcherKind {
    let mut best = MatcherKind::Table;
    let mut best_cost = matcher_cost(isa, best, cover);

    for matcher in [MatcherKind::EqOr, MatcherKind::Range, MatcherKind::NibbleN8] {
        if !is_eligible(isa, matcher, cover) {
            continue;
        }

        let cost = matcher_cost(isa, matcher, cover);
        if cost.total_cmp(&best_cost).is_lt() {
            best = matcher;
            best_cost = cost;
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
        skip_empty_packing: kind != MatcherKind::Table
            && probe_density < SKIP_PACKING_DENSITY_THRESHOLD,
    }
}

/// Rank normalized covers for the same indexed stream. Lower is better.
///
/// `cost = N * min(matcher_cost) + gamma * F`.
///
/// Scanning pays the cost of the selected matcher for every token code. Candidate
/// processing pays the profile's fixed penalty per covered occurrence to approximate
/// row lookup and exact verification. For a fixed nonempty stream, this is equivalent
/// to ranking by `min(matcher_cost) + gamma * F / N`, without a division.
///
/// `N = code_count` is the index's total token count. `F = covered_frequency` counts
/// occurrences of the cover's tokens in that same index, with each token
/// position counted once after cover normalization.
///
/// The cost expresses relative work: verification lengths vary, and successful
/// rows skip later hits. Preparation costs and the separately selected
/// mask-packing policy are excluded.
/// An empty cover or index has zero cost; callers handle empty patterns separately.
pub(in crate::search::substring) fn cover_cost(
    isa: Isa,
    cover: &ProbeCover,
    covered_frequency: u32,
    code_count: u32,
) -> f64 {
    if cover.is_empty() || code_count == 0 {
        return 0.0;
    }
    let matcher = select_matcher(isa, cover);
    let scanning = f64::from(code_count) * matcher_cost(isa, matcher, cover);
    let candidate_processing = candidate_cost(isa, covered_frequency);
    scanning + candidate_processing
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
                    is_eligible(isa, MatcherKind::NibbleN8, &cover(points, 0)),
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
                    for (density, skip) in [
                        (0.0, true),
                        (0.000075_f64.next_down(), true),
                        (0.000075, false),
                        (0.000075_f64.next_up(), false),
                        (0.01, false),
                        (1.0, false),
                    ] {
                        let config = select_matcher_config(isa, &cover, density);
                        assert!(is_eligible(isa, config.kind, &cover));
                        if config.kind == MatcherKind::Table {
                            assert!(!config.skip_empty_packing);
                        } else {
                            assert_eq!(config.skip_empty_packing, skip);
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
        let cover = cover(1, 0);
        assert!(!select_matcher_config(Isa::Neon, &cover, whole).skip_empty_packing);
        assert!(select_matcher_config(Isa::Neon, &cover, partial).skip_empty_packing);
    }
}
