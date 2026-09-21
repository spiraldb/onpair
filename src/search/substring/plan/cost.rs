// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scores for cover and matcher ranking, plus the mask-packing decision.
//!
//! Cover ranking combines a matcher score with a fixed penalty per probe hit.
//! Matcher weights depend on the instruction set and cover shape. They retain
//! the scale of the original timing measurements and serve as relative weights;
//! the resulting cover score is not a prediction of query latency.
//!
//! Mask packing uses a separate heuristic based on expected hit density.
//! `select` compares eligible matchers and applies the packing decision.
//! Measured and extrapolated weights are identified below.

use super::super::{ProbeCover, scan::PER_BATCH};
use super::{Isa, MatcherKind};

/// Relative matcher score per token code. Lower is preferred.
/// Covers use the same scale when combining this score with the hit penalty.
/// Empty-group packing is chosen separately and does not affect this score.
pub(in crate::search::substring) fn matcher_score(
    isa: Isa,
    matcher: MatcherKind,
    cover: &ProbeCover,
) -> f64 {
    match isa {
        Isa::Neon => neon(matcher, cover),
        Isa::Avx2 => avx2(matcher, cover),
        Isa::Avx512Bw => avx512bw(matcher, cover),
        Isa::Scalar => match matcher {
            // Scalar table weights follow the build architecture even
            // when that CPU cannot use its vector kernels.
            MatcherKind::Table => {
                if cfg!(all(target_arch = "x86_64", target_feature = "avx512bw")) {
                    avx512bw(matcher, cover)
                } else if cfg!(target_arch = "x86_64") {
                    avx2(matcher, cover)
                } else {
                    0.195
                }
            }
            _ => f64::INFINITY,
        },
    }
}

/// NEON weights fitted on Apple M4 Pro using `ch/hits/URL_1m`
/// on 2026-09-08. Point, range, and batch counts describe their respective work.
fn neon(matcher: MatcherKind, cover: &ProbeCover) -> f64 {
    let k = cover.n_points() as f64;
    let r = cover.n_ranges() as f64;
    let batches = cover.n_points().div_ceil(PER_BATCH) as f64;
    match matcher {
        MatcherKind::Table => 0.195,
        MatcherKind::EqOr => 0.0043 + 0.0144 * k + 0.0205 * r,
        MatcherKind::NibbleN8K => 0.0228 + 0.0325 * batches + 0.0213 * r,
        MatcherKind::Range => 0.0043 + 0.0212 * r,
    }
}

/// AVX2 weights fitted on Intel Xeon 6975P-C on 2026-09-08.
/// The `ch/hits/URL_1m` stream was shortened to fit L2 so memory bandwidth
/// did not dominate the kernel fit. The nibble weights are extrapolated.
fn avx2(matcher: MatcherKind, cover: &ProbeCover) -> f64 {
    let k = cover.n_points() as f64;
    let r = cover.n_ranges() as f64;
    let batches = cover.n_points().div_ceil(PER_BATCH) as f64;
    match matcher {
        MatcherKind::Table => 0.201,
        MatcherKind::EqOr => 0.0136 + 0.0105 * k + 0.0239 * r,
        // Unmeasured nibble terms: AVX-512 weights scaled by 2.2 for
        // narrower vectors and extra compares. The range slope is measured.
        MatcherKind::NibbleN8K => 0.0594 + 0.0310 * batches + 0.0225 * r,
        MatcherKind::Range => 0.0084 + 0.0225 * r,
    }
}

/// AVX-512 weights fitted on the same Xeon and L2-sized stream as AVX2.
/// Comparisons produce mask bits directly, avoiding byte-lane mask packing.
fn avx512bw(matcher: MatcherKind, cover: &ProbeCover) -> f64 {
    let k = cover.n_points() as f64;
    let r = cover.n_ranges() as f64;
    let batches = cover.n_points().div_ceil(PER_BATCH) as f64;
    match matcher {
        MatcherKind::Table => 0.204,
        MatcherKind::EqOr => 0.0079 + 0.0113 * k + 0.0127 * r,
        MatcherKind::NibbleN8K => 0.0270 + 0.0141 * batches + 0.0119 * r,
        MatcherKind::Range => 0.0085 + 0.0112 * r,
    }
}

/// Codes in the pair of mask words tested together before packing.
const PACK_GROUP: f64 = 128.0;

/// Whether testing for empty groups is expected to save packing work.
/// Balances a reduction on every group against packing saved on empty groups.
/// `density` is the estimated fraction of codes covered by the probes.
pub(in crate::search::substring) fn should_skip_packing(isa: Isa, density: f64) -> bool {
    let (reduction, pack) = match isa {
        // AVX-512 already produces a mask: skipping the pack only adds work.
        Isa::Avx512Bw => (0.35, 0.0),
        _ => (0.75, 1.01),
    };
    // Poisson estimate of empty groups. Clustered hits can change the savings.
    let no_match = (-PACK_GROUP * density).exp();
    (reduction - pack * no_match) / PACK_GROUP < 0.0
}

/// Ranking penalty per covered token occurrence, on the matcher-score scale.
/// Represents row lookup and verification work with one fixed weight.
/// The tested weight is 2048 times the NEON one-point score of 0.0187.
/// Keep it fixed across targets; it is independent of the cut-generation lambda.
pub(super) const COVER_HIT_PENALTY: f64 = 2048.0 * 0.0187;
