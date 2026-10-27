// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Matcher selection, the mask-packing policy, and the cost of a whole cover.

use super::super::ProbeCover;
use super::super::scan::{Isa, MatcherKind, PER_BATCH};
use super::cost::{candidate_cost, matcher_cost, max_nibble_batches};

/// Hit density below which vector matchers skip packing empty groups (0.025%).
/// Empirical, and shared by NEON, AVX2 and AVX-512. Clustered hits can make
/// skipping pay off above it too.
const SKIP_PACKING_DENSITY_THRESHOLD: f64 = 2.5e-4;

/// Whether `matcher` can run `cover` on `isa`, an instruction set this CPU supports.
/// Nibble batches are capped per instruction set, by the cost model.
pub(in crate::search::substring::probe) fn is_eligible(
    isa: Isa,
    matcher: MatcherKind,
    cover: &ProbeCover,
) -> bool {
    match matcher {
        MatcherKind::Table => true,
        MatcherKind::EqOr => isa != Isa::Scalar && !cover.is_empty(),
        MatcherKind::NibbleN8 => {
            cover.n_points() > 0 && cover.n_points().div_ceil(PER_BATCH) <= max_nibble_batches(isa)
        }
    }
}

/// The cheapest eligible matcher for `cover`. The table is always eligible, and
/// ties keep the earlier candidate.
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

/// Whether vector matchers should skip packing groups without hits in this scan.
///
/// True when hits are sparse (`probe_density`, the covered fraction of the indexed
/// stream, is below the threshold) or when the scan's `code_count` codes are
/// expected to hold less than one hit. The table matcher ignores this policy.
pub(in crate::search::substring::probe) fn skip_empty_packing(
    probe_density: f64,
    code_count: usize,
) -> bool {
    probe_density < SKIP_PACKING_DENSITY_THRESHOLD || probe_density * (code_count as f64) < 1.0
}

/// Estimated cost of scanning the indexed stream for `cover` with `matcher`. Lower
/// is better.
///
/// `cost = N * matcher_cost + candidate_cost(F)`, where `N = code_count` is the
/// number of indexed codes and `F = covered_frequency` the number of them in the
/// cover. Zero for an empty cover or stream.
pub(in crate::search::substring::probe) fn cover_cost(
    isa: Isa,
    cover: &ProbeCover,
    matcher: MatcherKind,
    covered_frequency: u32,
    code_count: u32,
) -> f64 {
    if cover.is_empty() || code_count == 0 {
        return 0.0;
    }
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
    fn nibble_admission_respects_batch_caps() {
        for (isa, limit) in [
            (Isa::Scalar, 0),
            (Isa::Neon, 24),
            (Isa::Avx2, 32),
            (Isa::Avx512Bw, 24),
        ] {
            for points in 0..=40 {
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
