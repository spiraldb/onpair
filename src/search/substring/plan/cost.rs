// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Relative costs for cover and matcher selection.
//!
//! Matcher selection and cover ranking share the same scan costs. NEON uses
//! small integer weights; other targets retain their calibrated scan costs.
//! These costs express relative work, not predicted latency.
//!
//! `select::cover_cost` combines scanning and candidate-processing work.
//! NEON charges 4096 per covered token occurrence on the integer cost scale.
//! Mask packing is chosen separately and retains its existing calibration.

use super::super::ProbeCover;
use super::super::scan::{Isa, MatcherKind, PER_BATCH};

/// Relative matcher cost per scanned code, shared with cover ranking.
///
/// NEON uses Table = 22, EqOr = 2p + 3r, Range = 3r, and
/// Nibble = 2 + 5 * ceil(p / 8) + 3r, where p and r count normalized probes.
/// These are calibrated weights, not literal instruction counts. They were
/// fitted on four datasets and checked on eight additional datasets with fixed
/// covers. Other targets retain their original matcher calibration.
pub(super) fn matcher_cost(isa: Isa, matcher: MatcherKind, cover: &ProbeCover) -> f64 {
    let points = cover.n_points() as f64;
    let ranges = cover.n_ranges() as f64;
    let batches = cover.n_points().div_ceil(PER_BATCH) as f64;
    let costs = match isa {
        Isa::Neon => {
            return match matcher {
                MatcherKind::Table => 22.0,
                MatcherKind::EqOr => 2.0 * points + 3.0 * ranges,
                MatcherKind::Range => 3.0 * ranges,
                MatcherKind::NibbleN8 => 2.0 + 5.0 * batches + 3.0 * ranges,
            };
        }
        Isa::Scalar => {
            return if matcher == MatcherKind::Table {
                scalar_table_cost()
            } else {
                f64::INFINITY
            };
        }
        Isa::Avx2 => AVX2,
        Isa::Avx512Bw => AVX512BW,
    };
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

/// The existing x86 matcher calibration.
///
/// Matcher terms describe cost per scanned code. Their base terms are
/// independent of cover shape; they are not one-time setup costs.
struct MatcherCosts {
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
}

/// AVX2 matcher weights fitted on Intel Xeon 6975P-C on 2026-09-08.
/// The `ch/hits/URL_1m` stream was shortened to fit L2 so memory bandwidth
/// did not dominate the kernel fit.
const AVX2: MatcherCosts = MatcherCosts {
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
};

/// AVX-512 weights fitted on the same Xeon and L2-sized stream as AVX2.
/// Comparisons produce mask bits directly, avoiding byte-lane mask packing.
const AVX512BW: MatcherCosts = MatcherCosts {
    table_code: 0.204,

    eq_or_base: 0.0079,
    eq_or_point: 0.0113,
    eq_or_range: 0.0127,

    range_base: 0.0085,
    range_range: 0.0112,

    nibble_base: 0.0270,
    nibble_batch: 0.0141,
    nibble_range: 0.0119,
};

/// Preserve the scalar table's original weight for the build architecture.
#[inline]
fn scalar_table_cost() -> f64 {
    if cfg!(all(target_arch = "x86_64", target_feature = "avx512bw")) {
        AVX512BW.table_code
    } else if cfg!(target_arch = "x86_64") {
        AVX2.table_code
    } else {
        0.195
    }
}

/// Approximate row-resolution and verification cost.
/// Frequency counts covered token occurrences, not matching rows or walker calls.
///
/// NEON's 4096 weight balances candidate work against its integer matcher costs.
/// It was selected to minimize the worst training slowdown on four datasets,
/// then evaluated on eight others. It is independent of the cut-generation lambda;
/// traversal lengths and successful-row skipping are not modeled explicitly.
/// Other targets retain their original candidate weight.
pub(super) fn candidate_cost(isa: Isa, covered_frequency: u32) -> f64 {
    let weight = match isa {
        Isa::Neon => 4096.0,
        _ => 2048.0 * 0.0187,
    };
    f64::from(covered_frequency) * weight
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
    let (empty_check, packing) = match isa {
        Isa::Avx512Bw => (0.35, 0.0),
        Isa::Scalar if cfg!(all(target_arch = "x86_64", target_feature = "avx512bw")) => {
            (0.35, 0.0)
        }
        _ => (0.75, 1.01),
    };
    let empty_probability = (-PACK_GROUP * density).exp();
    (empty_check - packing * empty_probability) / PACK_GROUP
}
