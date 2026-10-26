// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Select matchers from cover shape and ISA, and their mask-packing policy from
//! hit density and scan size.
//!
//! First restrict matchers to supported cover shapes, then compare their
//! relative costs. Sparse probes or small scans skip packing empty mask groups.
//! `scan::dispatch` prepares and executes the selected matcher;
//! selection performs no scanning or feature detection.
//!
//! During preparation, `cover_cost` ranks candidate covers using scan costs
//! balanced against a fixed penalty per covered token occurrence. The same
//! matcher costs choose the kernel that will execute each cover.

use super::super::ProbeCover;
use super::super::scan::{Isa, MatcherKind, PER_BATCH};
use super::cost::{candidate_cost, matcher_cost};

/// Probe density below which vector matchers skip packing empty groups.
/// This empirical cutoff (0.025%) is shared by NEON, AVX2 and
/// AVX-512. Sparse hits also leave whole blocks empty, avoiding row resolution.
/// Clustered hits can make skipping profitable above the cutoff too.
const SKIP_PACKING_DENSITY_THRESHOLD: f64 = 2.5e-4;

/// Whether this matcher is eligible for selection on the supplied target and cover.
/// The caller supplies an available instruction set. Nibble batches are limited to
/// two on AVX2 and three on NEON or AVX-512 to bound register use.
pub(in crate::search::substring::probe) fn is_eligible(
    isa: Isa,
    matcher: MatcherKind,
    cover: &ProbeCover,
) -> bool {
    let vector = isa != Isa::Scalar;
    match matcher {
        MatcherKind::Table => true,
        MatcherKind::EqOr => vector && !cover.is_empty(),
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
pub(in crate::search::substring::probe) fn select_matcher(
    isa: Isa,
    cover: &ProbeCover,
) -> MatcherKind {
    let mut best = MatcherKind::Table;
    let mut best_cost = matcher_cost(isa, best, cover);

    for matcher in [MatcherKind::EqOr, MatcherKind::NibbleN8] {
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

/// Whether vector matchers skip packing empty groups for a whole scan.
///
/// `probe_density` is the covered fraction of the indexed stream; `code_count`
/// is the number of codes in this scan. Skip when probes are sparse or the scan
/// has fewer than one projected hit. The scalar table ignores the policy.
pub(in crate::search::substring::probe) fn skip_empty_packing(
    probe_density: f64,
    code_count: usize,
) -> bool {
    probe_density < SKIP_PACKING_DENSITY_THRESHOLD || probe_density * (code_count as f64) < 1.0
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
pub(in crate::search::substring::probe) fn cover_cost(
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
    fn sparse_probes_or_few_projected_hits_skip_packing() {
        for (density, code_count, skip) in [
            // The density threshold is strict.
            (0.00025_f64.next_down(), 1_000_000, true),
            (0.00025, 1_000_000, false),
            (0.00025_f64.next_up(), 1_000_000, false),
            // Partial scans use the same density but project fewer hits.
            (1.0 / 400.0, 399, true),
            (1.0 / 400.0, 400, false),
            (1.0 / 400.0, 401, false),
        ] {
            assert_eq!(
                skip_empty_packing(density, code_count),
                skip,
                "density {density}, {code_count} codes"
            );
        }
    }

    #[test]
    fn selected_matchers_are_eligible() {
        for isa in [Isa::Scalar, Isa::Neon, Isa::Avx2, Isa::Avx512Bw] {
            for points in 0..=32 {
                for ranges in 0..=4 {
                    let cover = cover(points, ranges);
                    if !cover.is_empty() {
                        assert!(is_eligible(isa, select_matcher(isa, &cover), &cover));
                    }
                }
            }
        }
    }
}
