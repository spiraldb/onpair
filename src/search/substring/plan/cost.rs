// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Relative costs for cover and matcher selection.
//!
//! Each instruction set has a coefficient profile, followed by shared formulas
//! for scanning, candidate processing, and empty-group packing. Recalibration
//! replaces the profile values without changing the formulas or selection logic.
//!
//! Costs retain the scale of the original timing measurements and serve as
//! relative weights, not predictions of query latency. `select` compares eligible
//! matchers, combines scanning and candidate costs, and chooses packing separately.

use super::super::ProbeCover;
use super::super::scan::{Isa, MatcherKind, PER_BATCH};

/// Coefficients on a common relative-cost scale.
///
/// Matcher terms describe cost per scanned code. Their base terms are
/// independent of cover shape; they are not one-time setup costs.
struct Coefficients {
    /// Table lookup cost per code.
    table_code: f64,

    /// EqOr: base + point * points + range * ranges.
    eq_or_base: f64,
    eq_or_point: f64,
    eq_or_range: f64,

    /// Range: base + range * ranges.
    range_base: f64,
    range_range: f64,

    /// Nibble: base + batch * ceil(points / PER_BATCH) + range * ranges.
    nibble_base: f64,
    nibble_batch: f64,
    nibble_range: f64,

    /// Effective candidate-processing cost per covered occurrence.
    /// Approximates row resolution and verification: an occurrence need not
    /// cause a walker call, and traversal lengths vary.
    covered_occurrence: f64,

    /// Cost of checking whether a group has any hits.
    empty_check_per_group: f64,
    /// Packing cost avoided when a group has no hits.
    packing_per_group: f64,
}

/// NEON matcher weights fitted on Apple M4 Pro using `ch/hits/URL_1m`
/// on 2026-09-08. The candidate weight is separately tuned: 2048 times the
/// one-point cost of 0.0187, independent of the cut-generation lambda.
/// Packing weights retain the original empty-group calibration.
const NEON: Coefficients = Coefficients {
    table_code: 0.195,

    eq_or_base: 0.0043,
    eq_or_point: 0.0144,
    eq_or_range: 0.0205,

    range_base: 0.0043,
    range_range: 0.0212,

    nibble_base: 0.0228,
    nibble_batch: 0.0325,
    nibble_range: 0.0213,

    covered_occurrence: 2048.0 * 0.0187,

    empty_check_per_group: 0.75,
    packing_per_group: 1.01,
};

/// AVX2 matcher weights fitted on Intel Xeon 6975P-C on 2026-09-08.
/// The `ch/hits/URL_1m` stream was shortened to fit L2 so memory bandwidth
/// did not dominate the kernel fit. Candidate and packing weights retain
/// the shared values used before introducing profiles.
const AVX2: Coefficients = Coefficients {
    table_code: 0.201,

    eq_or_base: 0.0136,
    eq_or_point: 0.0105,
    eq_or_range: 0.0239,

    range_base: 0.0084,
    range_range: 0.0225,

    // Unmeasured base and batch terms: AVX-512 weights scaled by 2.2 for
    // narrower vectors and extra compares. The range slope is measured.
    nibble_base: 0.0594,
    nibble_batch: 0.0310,
    nibble_range: 0.0225,

    covered_occurrence: 2048.0 * 0.0187,

    empty_check_per_group: 0.75,
    packing_per_group: 1.01,
};

/// AVX-512 weights fitted on the same Xeon and L2-sized stream as AVX2.
/// Comparisons produce mask bits directly, avoiding byte-lane mask packing.
/// The candidate weight retains the shared value used before profiles.
const AVX512BW: Coefficients = Coefficients {
    table_code: 0.204,

    eq_or_base: 0.0079,
    eq_or_point: 0.0113,
    eq_or_range: 0.0127,

    range_base: 0.0085,
    range_range: 0.0112,

    nibble_base: 0.0270,
    nibble_batch: 0.0141,
    nibble_range: 0.0119,

    covered_occurrence: 2048.0 * 0.0187,

    empty_check_per_group: 0.35,
    packing_per_group: 0.0,
};

/// Select a profile without detecting CPU features.
/// Scalar table weights follow the build architecture, as in the original fit.
#[inline]
fn coefficients(isa: Isa) -> Coefficients {
    match isa {
        Isa::Neon => NEON,
        Isa::Avx2 => AVX2,
        Isa::Avx512Bw => AVX512BW,
        Isa::Scalar => {
            if cfg!(all(target_arch = "x86_64", target_feature = "avx512bw")) {
                AVX512BW
            } else if cfg!(target_arch = "x86_64") {
                AVX2
            } else {
                NEON
            }
        }
    }
}

/// Relative matcher cost per token code. Lower is preferred.
/// Empty-group packing is chosen separately and does not affect this cost.
pub(in crate::search::substring) fn matcher_cost(
    isa: Isa,
    matcher: MatcherKind,
    cover: &ProbeCover,
) -> f64 {
    if isa == Isa::Scalar && matcher != MatcherKind::Table {
        return f64::INFINITY;
    }

    let costs = coefficients(isa);
    let points = cover.n_points() as f64;
    let ranges = cover.n_ranges() as f64;
    let batches = cover.n_points().div_ceil(PER_BATCH) as f64;
    match matcher {
        MatcherKind::Table => costs.table_code,
        MatcherKind::EqOr => {
            costs.eq_or_base + costs.eq_or_point * points + costs.eq_or_range * ranges
        }
        MatcherKind::Range => costs.range_base + costs.range_range * ranges,
        MatcherKind::NibbleN8 => {
            costs.nibble_base + costs.nibble_batch * batches + costs.nibble_range * ranges
        }
    }
}

/// Approximate row-resolution and verification cost.
/// Frequency counts covered token occurrences, not matching rows or walker calls.
pub(super) fn candidate_cost(isa: Isa, covered_frequency: u32) -> f64 {
    f64::from(covered_frequency) * coefficients(isa).covered_occurrence
}

/// Codes processed together by the scanner's packing loop.
/// This is an implementation constant, not a fitted coefficient.
const PACK_GROUP: f64 = 128.0;

/// Extra cost per code when checking for empty groups before vector mask packing.
///
/// Negative means avoided packing outweighs the empty-group check. The empty-group
/// probability uses a Poisson approximation; clustered hits can change the savings.
/// Scalar tables do not pack masks and do not use this estimate.
pub(super) fn packing_cost_delta(isa: Isa, density: f64) -> f64 {
    let costs = coefficients(isa);
    let empty_probability = (-PACK_GROUP * density).exp();
    (costs.empty_check_per_group - costs.packing_per_group * empty_probability) / PACK_GROUP
}
