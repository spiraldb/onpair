// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! PR scan cost model and calibration provenance.
//! Capabilities are inputs shared with execution selection.

use super::super::ProbeCover;
use super::facts::{
    BLOCK, CoverShape, Isa, MatcherKind, PER_BATCH, RegionFacts, ResolverKind, TargetCaps,
};
use super::select::select_matcher;

#[cfg(test)]
pub(in crate::search::substring) const BYTES_PER_CODE: f64 = 2.0;

/// Codes one pass of the pack covers, which is the two mask words
/// [`words`](super::super::scan::matcher) writes at a time and the unit the flag decides
/// over.
const PACK_GROUP: f64 = 128.0;

/// What `SKIP_MOVEMASK_IF_NO_MATCH` adds, in ns per code: the lane OR it pays on
/// every group, less the pack it saves on the groups with no match. Both are
/// fitted to the two ends of the sweep and land on their instruction counts.
/// Negative is worth taking. See `README.md`.
pub(in crate::search::substring) fn skip_ns_per_code(isa: Isa, density: f64) -> f64 {
    let (reduction, pack) = match isa {
        // AVX-512 already produces a mask: skipping the pack only adds work.
        Isa::Avx512Bw => (0.35, 0.0),
        _ => (0.75, 1.01),
    };
    // The share of groups with no match, if the hits fall evenly. They
    // cluster, so the break-even is an order and not a point.
    let no_match = (-PACK_GROUP * density).exp();
    (reduction - pack * no_match) / PACK_GROUP
}

/// What the planned matcher costs per code at this hit density: the kernel
/// [`select_matcher`] picks plus the pack-skipping flag where it pays. The
/// scalar kernels have no pack to skip.
pub(in crate::search::substring) fn stage_one_ns_per_code(
    caps: TargetCaps,
    shape: CoverShape,
    density: f64,
) -> f64 {
    let matcher = select_matcher(caps, shape);
    let skip = match matcher {
        MatcherKind::Table => 0.0,
        _ => skip_ns_per_code(caps.isa, density).min(0.0),
    };
    ns_per_code(caps.isa, matcher, shape) + skip
}

/// Fitted cost for the selected target; no feature detection occurs here.
pub(in crate::search::substring) fn ns_per_code(
    isa: Isa,
    matcher: MatcherKind,
    shape: CoverShape,
) -> f64 {
    match isa {
        Isa::Neon => neon(matcher, shape),
        Isa::Avx2 => avx2(matcher, shape),
        Isa::Avx512Bw => avx512bw(matcher, shape),
        Isa::Scalar => match matcher {
            MatcherKind::Table => 0.195,
            _ => f64::INFINITY,
        },
    }
}

/// Fitted on `ch/hits/URL_1m` on an Apple M4 Pro, 2026-09-08, every kernel
/// within 3.4% of its rows.
fn neon(matcher: MatcherKind, shape: CoverShape) -> f64 {
    let k = shape.points as f64;
    let r = shape.ranges as f64;
    let batches = shape.points.div_ceil(PER_BATCH) as f64;
    match matcher {
        MatcherKind::Table => 0.195,
        MatcherKind::EqOr => 0.0043 + 0.0144 * k + 0.0205 * r,
        MatcherKind::NibbleN8K => 0.0228 + 0.0325 * batches + 0.0213 * r,
        MatcherKind::Range => 0.0043 + 0.0212 * r,
    }
}

/// Fitted on `ch/hits/URL_1m` on an Intel Xeon 6975P-C, 2026-09-08. The
/// stream was cut to fit L2 there, because on a 4 Mcode one that core pins
/// every vector kernel at 31 GB/s, and a fit through capped rows describes
/// the memory system rather than the kernel. So these are what the kernels
/// cost when fed; nothing above 77 GB/s was reachable.
fn avx2(matcher: MatcherKind, shape: CoverShape) -> f64 {
    let k = shape.points as f64;
    let r = shape.ranges as f64;
    let batches = shape.points.div_ceil(PER_BATCH) as f64;
    match matcher {
        MatcherKind::Table => 0.201,
        MatcherKind::EqOr => 0.0136 + 0.0105 * k + 0.0239 * r,
        // NOT MEASURED. The kernel is newer than the sweep, so this row is
        // the AVX-512 row's vector terms at 2.2x: twice for the half-width
        // lanes, and a fifth again for the compare pair AVX2 spends where
        // `vptestmb` writes the mask word for nothing. The range slope is
        // the measured one from the rows above. Re-run `bench::fit` on an
        // AVX2 build and paste its block over this.
        MatcherKind::NibbleN8K => 0.0594 + 0.0310 * batches + 0.0225 * r,
        MatcherKind::Range => 0.0084 + 0.0225 * r,
    }
}

/// Fitted on the same core and stream as [`avx2`], built for AVX-512. The
/// mask register is what these show: a range costs half what it costs NEON
/// and a bitmap batch half again, neither of them paying for a pack.
fn avx512bw(matcher: MatcherKind, shape: CoverShape) -> f64 {
    let k = shape.points as f64;
    let r = shape.ranges as f64;
    let batches = shape.points.div_ceil(PER_BATCH) as f64;
    match matcher {
        MatcherKind::Table => 0.204,
        MatcherKind::EqOr => 0.0079 + 0.0113 * k + 0.0127 * r,
        MatcherKind::NibbleN8K => 0.0270 + 0.0141 * batches + 0.0119 * r,
        MatcherKind::Range => 0.0085 + 0.0112 * r,
    }
}

/// One hit through the alignment walk, false hits and confirmations
/// averaged: the median of 57 needles over three columns at two dictionary
/// widths in the `prefilter_walk_*.csv` runs, each above the subtraction's
/// jitter floor, spread 2.6 to 30 ns.
const WALK_NS_PER_HIT: f64 = 8.0;

/// Stage two: one mask word read in a block that hit, one row `LinearSeek`
/// emits, one row it walks past, one halving of a `GallopSeek` search.
/// Fitted on aarch64 Apple M4 Pro, from novel_resolve_2026-09-08_15-35-11.csv,
/// by `bench::resolver_fit`. The two resolvers sit within 20% of each other
/// below the crossover and the fit within 35% of its rows, which is the
/// machine's noise across a run and not a term the model lacks.
const WORD_NS: f64 = 0.05;
const LINEAR_SEEK_ROW_NS: f64 = 3.98;
const LINEAR_SEEK_CROSS_NS: f64 = 0.41;
const GALLOP_SEEK_STEP_NS: f64 = 3.89;

/// What a resolver pays per emitted row, less the word scan both pay, when
/// the cursor crosses `g` rows to reach it: the walk is linear in `g`, the
/// search logarithmic.
pub(in crate::search::substring) fn seek_ns_per_row(resolver: ResolverKind, g: f64) -> f64 {
    match resolver {
        ResolverKind::LinearSeek => LINEAR_SEEK_ROW_NS + LINEAR_SEEK_CROSS_NS * g,
        ResolverKind::GallopSeek => GALLOP_SEEK_STEP_NS * (1.0 + g).log2(),
    }
}

/// Stage two over `words` mask words in blocks that hit, emitting `emitted`
/// rows with `crossed` rows walked or searched past on the way.
pub(in crate::search::substring) fn stage_two_ns(
    resolver: ResolverKind,
    words: f64,
    emitted: f64,
    crossed: f64,
) -> f64 {
    if emitted <= 0.0 {
        return WORD_NS * words;
    }
    WORD_NS * words + emitted * seek_ns_per_row(resolver, crossed / emitted)
}

/// Rows the mask is expected to name: a row of `x` expected hits holds one
/// with probability `1 - exp(-x)`, which the resolver sweep measured to hold
/// within 20%.
pub(in crate::search::substring) fn hit_rows(expected_hits: f64, rows: f64) -> f64 {
    rows * (1.0 - (-expected_hits / rows).exp())
}

/// Expected nanoseconds to scan `cover` over `region` and walk every hit,
/// given the `covered` codes it matches there: stage one at the kernel the
/// shape gets, stage two at the cheaper resolver, the walk per hit. An empty
/// cover proves no row matches and costs nothing.
pub(in crate::search::substring) fn scan_ns(
    caps: TargetCaps,
    cover: &ProbeCover,
    covered: u32,
    region: RegionFacts,
) -> f64 {
    if cover.is_empty() || region.code_count == 0 || region.row_count == 0 {
        return 0.0;
    }
    let codes = region.code_count as f64;
    let rows = region.row_count as f64;
    let covered = f64::from(covered);
    let shape = CoverShape {
        points: cover.points().len(),
        ranges: cover.ranges().len(),
    };
    let stage_one = codes * stage_one_ns_per_code(caps, shape, covered / codes);

    let emitted = hit_rows(covered, rows);
    let blocks_hit = 1.0 - (-covered / codes * BLOCK as f64).exp();
    let words = codes / 64.0 * blocks_hit;
    let stage_two = [ResolverKind::LinearSeek, ResolverKind::GallopSeek]
        .into_iter()
        .map(|resolver| stage_two_ns(resolver, words, emitted, rows))
        .fold(f64::MAX, f64::min);

    stage_one + stage_two + covered * WALK_NS_PER_HIT
}
