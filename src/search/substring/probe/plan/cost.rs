// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cost model for covers and matchers, calibrated per instruction set.
//!
//! Costs are in table scans: one is the time the table matcher takes to test one
//! code, so the table costs 1 on every instruction set. Per code, a filter costs
//! `matcher + density * candidate`: `matcher_cost` prices the matcher on the
//! filter, `density` is the fraction of codes the filter covers, and `candidate`
//! prices the row lookup and verification of one hit (`candidate_cost`). Matcher
//! selection compares the first term; cover ranking compares the sum.
//!
//! The weights set where one choice overtakes another. With NEON's, for filters
//! without ranges, EqOr beats the table up to 10 points, Nibble beats EqOr from 4
//! points and the table up to 24, and a filter covering 1% of the codes spends
//! about one table scan per code on its hits. The NEON weights are rounded
//! measurements. The x86 ones steer planning rather than measure it: on AVX2, EqOr
//! with one to four points costs about a quarter of a table scan whatever the
//! count, bound by reading the codes, but weights that said so led planning to
//! filters with more points, which ran slower. Their point weight acts as a
//! preference for fewer points.

use super::super::ProbeCover;
use super::super::scan::{Isa, MatcherKind, PER_BATCH};

/// Cost of scanning one code with `matcher`, in table scans. Lower is better.
///
/// With `p` points, `r` ranges and `b = ceil(p / PER_BATCH)` nibble batches:
/// - Table: 1.
/// - EqOr: `point * p + range * r`.
/// - Nibble: `nibble_base + nibble_batch * b + range * r`.
///
/// Eligibility is checked separately, by `select::is_eligible`.
pub(super) fn matcher_cost(isa: Isa, matcher: MatcherKind, cover: &ProbeCover) -> f64 {
    let Some(weights) = weights(isa) else {
        return if matcher == MatcherKind::Table {
            1.0
        } else {
            f64::INFINITY
        };
    };
    let points = cover.n_points() as f64;
    let ranges = cover.n_ranges() as f64;
    let batches = cover.n_points().div_ceil(PER_BATCH) as f64;
    match matcher {
        MatcherKind::Table => 1.0,
        MatcherKind::EqOr => weights.point * points + weights.range * ranges,
        MatcherKind::NibbleN8 => {
            weights.nibble_base + weights.nibble_batch * batches + weights.range * ranges
        }
    }
}

/// Weights of the cost model for one instruction set, in table scans.
struct CostWeights {
    /// One point comparison in EqOr.
    point: f64,
    /// One inclusive range check in any vector matcher.
    range: f64,
    /// Nibble setup paid for each scanned vector, independent of batch count.
    nibble_base: f64,
    /// One batch of up to eight point probes in Nibble.
    nibble_batch: f64,
    /// Most batches planning gives Nibble. Up to it each batch costs
    /// `nibble_batch`; past it the cost per batch can jump, as on AVX2 after the
    /// fourth.
    max_nibble_batches: usize,
    /// Row lookup and verification of one hit.
    candidate: f64,
}

const NEON_COSTS: CostWeights = CostWeights {
    point: 0.092,
    range: 0.138,
    nibble_base: 0.092,
    nibble_batch: 0.23,
    max_nibble_batches: 3,
    candidate: 94.0,
};

const AVX2_COSTS: CostWeights = CostWeights {
    point: 0.028,
    range: 0.056,
    nibble_base: 0.14,
    nibble_batch: 0.056,
    max_nibble_batches: 4,
    candidate: 103.0,
};

const AVX512BW_COSTS: CostWeights = CostWeights {
    point: 0.033,
    range: 0.066,
    nibble_base: 0.033,
    nibble_batch: 0.033,
    max_nibble_batches: 3,
    candidate: 105.0,
};

/// The weights of `isa`, or `None` on scalar targets, where only the table runs.
fn weights(isa: Isa) -> Option<&'static CostWeights> {
    match isa {
        Isa::Neon => Some(&NEON_COSTS),
        Isa::Avx2 => Some(&AVX2_COSTS),
        Isa::Avx512Bw => Some(&AVX512BW_COSTS),
        Isa::Scalar => None,
    }
}

/// The most batches planning gives Nibble on `isa`; none on scalar targets.
pub(super) fn max_nibble_batches(isa: Isa) -> usize {
    weights(isa).map_or(0, |weights| weights.max_nibble_batches)
}

/// The cost model's slopes: what one more point or range adds to scanning a code,
/// and what one hit costs. They make additive cut weights.
#[derive(Clone, Copy, Debug)]
pub(super) struct Slopes {
    /// One point in EqOr.
    pub(super) eq_or_point: f64,
    /// One point in Nibble: its batch's cost, spread over the batch's points.
    pub(super) nibble_point: f64,
    /// One range in any vector matcher.
    pub(super) range: f64,
    /// Row lookup and verification of one hit.
    pub(super) candidate: f64,
}

/// The slopes of `isa`'s cost model, or `None` on scalar targets, where the
/// table's cost does not depend on the cover.
pub(super) fn slopes(isa: Isa) -> Option<Slopes> {
    let weights = weights(isa)?;
    Some(Slopes {
        eq_or_point: weights.point,
        nibble_point: weights.nibble_batch / PER_BATCH as f64,
        range: weights.range,
        candidate: weights.candidate,
    })
}

/// Cost of looking up and verifying `covered_frequency` hits.
///
/// A flat cost per hit, although sparse hits cost more each than dense ones. Walk
/// lengths and the hits skipped after a row matches are not modeled either. On
/// scalar targets only the table matcher exists, and its cost does not depend on
/// the cover, so ranking there simply minimizes frequency.
pub(super) fn candidate_cost(isa: Isa, covered_frequency: u32) -> f64 {
    let weight = weights(isa).map_or(1.0, |weights| weights.candidate);
    f64::from(covered_frequency) * weight
}
