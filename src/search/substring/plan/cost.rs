// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Relative costs for cover and matcher selection.
//!
//! One profile per instruction set prices both matcher work and covered token
//! occurrences. Matcher selection minimizes the matcher term; cover ranking
//! combines that same term with candidate processing. Costs express relative
//! work, not nanoseconds. Mask packing is chosen separately.

use super::super::ProbeCover;
use super::super::scan::{Isa, MatcherKind, PER_BATCH};

/// Rank matchers for a fixed cover. Lower is better.
///
/// With `p` points, `r` ranges, and `b = ceil(p / PER_BATCH)`:
/// - Table: `table`.
/// - EqOr: `point * p + range * r`.
/// - Range: `range * r`.
/// - Nibble: `nibble_base + nibble_batch * b + range * r`.
///
/// The weights are calibrated relative costs, not literal instruction counts.
/// Eligibility and ties are handled by selection, independently of this formula.
pub(super) fn matcher_cost(isa: Isa, matcher: MatcherKind, cover: &ProbeCover) -> f64 {
    let weights = match isa {
        Isa::Neon => NEON_COSTS,
        Isa::Avx2 => AVX2_COSTS,
        Isa::Avx512Bw => AVX512BW_COSTS,
        Isa::Scalar => {
            return if matcher == MatcherKind::Table {
                1.0
            } else {
                f64::INFINITY
            };
        }
    };
    let points = cover.n_points() as f64;
    let ranges = cover.n_ranges() as f64;
    let batches = cover.n_points().div_ceil(PER_BATCH) as f64;
    match matcher {
        MatcherKind::Table => weights.table,
        MatcherKind::EqOr => weights.point * points + weights.range * ranges,
        MatcherKind::Range => weights.range * ranges,
        MatcherKind::NibbleN8 => {
            weights.nibble_base + weights.nibble_batch * batches + weights.range * ranges
        }
    }
}

/// Shared formula, calibrated separately for each instruction set.
struct CostWeights {
    /// Table lookup, independent of cover shape.
    table: f64,
    /// One point comparison in EqOr.
    point: f64,
    /// One inclusive range check in any vector matcher.
    range: f64,
    /// Nibble setup paid for each scanned vector, independent of batch count.
    nibble_base: f64,
    /// One batch of up to eight point probes in Nibble.
    nibble_batch: f64,
    /// Row resolution and verification per covered occurrence, in the same scale.
    candidate: f64,
}

/// Fitted on four datasets and checked on eight others with fixed covers.
/// The candidate weight was evaluated separately on those datasets.
const NEON_COSTS: CostWeights = CostWeights {
    table: 22.0,
    point: 2.0,
    range: 3.0,
    nibble_base: 2.0,
    nibble_batch: 5.0,
    candidate: 4096.0,
};

/// Conservative x86 profiles checked on fixed covers from ClickBench URLs,
/// Amazon titles, and DBpedia abstracts on an AMD EPYC 9R05.
/// The provisional candidate weight rounds the three-dataset fit of 3668.7225.
/// Baseline-relative validation against decompress-then-scan is still pending.
const AVX2_COSTS: CostWeights = CostWeights {
    table: 36.0,
    point: 1.0,
    range: 2.0,
    nibble_base: 5.0,
    nibble_batch: 2.0,
    candidate: 3669.0,
};

/// AVX-512 profile from the same fixed-cover evaluation as AVX2.
/// The provisional candidate weight rounds the three-dataset fit of 3170.2343.
/// Baseline-relative validation against decompress-then-scan is still pending.
const AVX512BW_COSTS: CostWeights = CostWeights {
    table: 31.0,
    point: 1.0,
    range: 2.0,
    nibble_base: 1.0,
    nibble_batch: 1.0,
    candidate: 3170.0,
};

/// Approximate row-resolution and verification cost.
/// Frequency counts covered token occurrences, not matching rows or walker calls.
///
/// Each profile balances this term against its matcher costs. The weight is
/// independent of the cut-generation lambda; traversal lengths and successful-row
/// skipping are not modeled explicitly. Scalar scans only have the table matcher,
/// whose cost is independent of the cover, so their ranking minimizes frequency.
pub(super) fn candidate_cost(isa: Isa, covered_frequency: u32) -> f64 {
    let weight = match isa {
        Isa::Neon => NEON_COSTS.candidate,
        Isa::Avx2 => AVX2_COSTS.candidate,
        Isa::Avx512Bw => AVX512BW_COSTS.candidate,
        Isa::Scalar => 1.0,
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
