// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cost model for covers and matchers, with weights per instruction set.
//!
//! Costs are in table scans: one is the time the table matcher takes to test one
//! code, so the table costs 1 on every instruction set. Per code, a cover costs
//! `matcher + density * hit`: `matcher` prices the matcher on the cover, `density`
//! is the hit density, the fraction of codes that are hits, and `hit` prices the
//! row lookup and verification of one hit. Matcher selection compares the first
//! term; cover ranking compares the sum. Scalar targets have no model: only the
//! table runs there, and its cost does not depend on the cover.
//!
//! The weights set where one choice overtakes another. With NEON's, for covers
//! without ranges, EqOr beats the table up to 10 points, Nibble beats EqOr from 4
//! points and the table up to 24, and at a hit density of 1% a cover spends about
//! one table scan per code on its hits. On x86 the point weight is a
//! preference rather than a time: on AVX2, EqOr with up to four points is bound by
//! loading the codes, so its cost barely depends on the point count, and the
//! weight steers planning toward covers with fewer points.

use crate::search::substring::ProbeCover;
use crate::search::substring::probe::scan::{Isa, MatcherKind, PER_BATCH};

/// The cost model of one instruction set, in table scans.
pub(crate) struct CostModel {
    /// One point comparison in EqOr.
    pub(super) point: f64,
    /// One inclusive range check in any vector matcher.
    pub(super) range: f64,
    /// Nibble's cost per code that does not depend on the batch count.
    nibble_base: f64,
    /// One batch of up to eight point probes in Nibble.
    pub(super) nibble_batch: f64,
    /// Most batches planning gives Nibble. Up to it each batch costs
    /// `nibble_batch`; past it the cost per batch can jump, as on AVX2 after the
    /// fourth.
    max_nibble_batches: usize,
    /// Row lookup and verification of one hit. The cost is flat, although sparse
    /// hits cost more each than dense ones; walk lengths and the hits skipped after
    /// a row matches are not modeled either.
    pub(super) hit: f64,
}

const NEON_COSTS: CostModel = CostModel {
    point: 0.092,
    range: 0.138,
    nibble_base: 0.092,
    nibble_batch: 0.23,
    max_nibble_batches: 3,
    hit: 94.0,
};

const AVX2_COSTS: CostModel = CostModel {
    point: 0.028,
    range: 0.056,
    nibble_base: 0.14,
    nibble_batch: 0.056,
    max_nibble_batches: 4,
    hit: 103.0,
};

const AVX512BW_COSTS: CostModel = CostModel {
    point: 0.033,
    range: 0.066,
    nibble_base: 0.033,
    nibble_batch: 0.033,
    max_nibble_batches: 4,
    hit: 105.0,
};

impl CostModel {
    /// The model of `isa`, or `None` on scalar targets, where only the table runs.
    pub(crate) const fn of(isa: Isa) -> Option<&'static Self> {
        match isa {
            Isa::Neon => Some(&NEON_COSTS),
            Isa::Avx2 => Some(&AVX2_COSTS),
            Isa::Avx512Bw => Some(&AVX512BW_COSTS),
            Isa::Scalar => None,
        }
    }

    /// Whether `matcher` can run `cover`. Nibble needs points, at most
    /// `max_nibble_batches` batches of them.
    pub(crate) fn can_run(&self, matcher: MatcherKind, cover: &ProbeCover) -> bool {
        match matcher {
            MatcherKind::Table => true,
            MatcherKind::EqOr => !cover.is_empty(),
            MatcherKind::Nibble => {
                let points = cover.points().len();
                points > 0 && points.div_ceil(PER_BATCH) <= self.max_nibble_batches
            }
        }
    }

    /// The cheapest matcher that can run `cover`, and the cost of scanning `codes`
    /// codes for `cover` with it when `hits` of them are hits: zero for an empty
    /// cover or stream. Ties keep the table, then EqOr.
    pub(crate) fn cheapest(&self, cover: &ProbeCover, hits: u32, codes: u32) -> (MatcherKind, f64) {
        let mut best = MatcherKind::Table;
        let mut best_cost = self.matcher(best, cover);
        for matcher in [MatcherKind::EqOr, MatcherKind::Nibble] {
            if !self.can_run(matcher, cover) {
                continue;
            }
            let cost = self.matcher(matcher, cover);
            if cost.total_cmp(&best_cost).is_lt() {
                best = matcher;
                best_cost = cost;
            }
        }
        if cover.is_empty() || codes == 0 {
            return (best, 0.0);
        }
        let scanning = f64::from(codes) * best_cost;
        let verifying = f64::from(hits) * self.hit;
        (best, scanning + verifying)
    }

    /// Cost of scanning one code with `matcher`. With `p` points, `r` ranges and
    /// `b = ceil(p / PER_BATCH)` Nibble batches:
    /// - Table: 1.
    /// - EqOr: `point * p + range * r`.
    /// - Nibble: `nibble_base + nibble_batch * b + range * r`.
    fn matcher(&self, matcher: MatcherKind, cover: &ProbeCover) -> f64 {
        let points = cover.points().len() as f64;
        let ranges = cover.ranges().len() as f64;
        let batches = cover.points().len().div_ceil(PER_BATCH) as f64;
        match matcher {
            MatcherKind::Table => 1.0,
            MatcherKind::EqOr => self.point * points + self.range * ranges,
            MatcherKind::Nibble => {
                self.nibble_base + self.nibble_batch * batches + self.range * ranges
            }
        }
    }
}

/// The most batches planning gives Nibble on `isa`, and so the most dispatch
/// compiles a Nibble kernel for; none on scalar targets.
pub(crate) const fn max_nibble_batches(isa: Isa) -> usize {
    match CostModel::of(isa) {
        Some(model) => model.max_nibble_batches,
        None => 0,
    }
}
