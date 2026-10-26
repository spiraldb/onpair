// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cost model for covers and matchers, calibrated per instruction set.
//!
//! Costs are relative work, not time. `matcher_cost` prices scanning one code with
//! a given matcher; `candidate_cost` prices the row lookup and verification of the
//! hits. Matcher selection uses the first; cover ranking uses both.

use super::super::ProbeCover;
use super::super::scan::{Isa, MatcherKind, PER_BATCH};

/// Cost of scanning one code with `matcher`. Lower is better.
///
/// With `p` points, `r` ranges and `b = ceil(p / PER_BATCH)` nibble batches:
/// - Table: `table`.
/// - EqOr: `point * p + range * r`.
/// - Nibble: `nibble_base + nibble_batch * b + range * r`.
///
/// Eligibility is checked separately, by `select::is_eligible`.
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
        MatcherKind::NibbleN8 => {
            weights.nibble_base + weights.nibble_batch * batches + weights.range * ranges
        }
    }
}

/// Weights of the cost model for one instruction set.
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
    /// Row lookup and verification of one hit.
    candidate: f64,
}

/// Fitted on four datasets and checked on eight others with fixed covers. The
/// candidate weight was evaluated separately on the same datasets.
const NEON_COSTS: CostWeights = CostWeights {
    table: 22.0,
    point: 2.0,
    range: 3.0,
    nibble_base: 2.0,
    nibble_batch: 5.0,
    candidate: 4096.0,
};

/// Conservative profile checked on fixed covers from ClickBench URLs, Amazon titles
/// and DBpedia abstracts, on an AMD EPYC 9R05. The candidate weight is provisional:
/// it rounds the three-dataset fit of 3668.7225 and has not yet been validated
/// against decompress-then-scan.
const AVX2_COSTS: CostWeights = CostWeights {
    table: 36.0,
    point: 1.0,
    range: 2.0,
    nibble_base: 5.0,
    nibble_batch: 2.0,
    candidate: 3669.0,
};

/// From the same evaluation as the AVX2 profile. The candidate weight is
/// provisional: it rounds the three-dataset fit of 3170.2343 and has not yet been
/// validated against decompress-then-scan.
const AVX512BW_COSTS: CostWeights = CostWeights {
    table: 31.0,
    point: 1.0,
    range: 2.0,
    nibble_base: 1.0,
    nibble_batch: 1.0,
    candidate: 3170.0,
};

/// Cost of looking up and verifying `covered_frequency` hits.
///
/// A flat cost per hit: walk lengths and the hits skipped after a row matches are
/// not modeled. On scalar targets only the table matcher exists, and its cost does
/// not depend on the cover, so ranking there simply minimizes frequency.
pub(super) fn candidate_cost(isa: Isa, covered_frequency: u32) -> f64 {
    let weight = match isa {
        Isa::Neon => NEON_COSTS.candidate,
        Isa::Avx2 => AVX2_COSTS.candidate,
        Isa::Avx512Bw => AVX512BW_COSTS.candidate,
        Isa::Scalar => 1.0,
    };
    f64::from(covered_frequency) * weight
}
