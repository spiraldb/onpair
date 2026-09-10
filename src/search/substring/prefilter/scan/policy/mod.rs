// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Which matcher and which resolver, from the needle set and the region, and
//! what the scan they make costs.
//!
//! Stage one is decided by the cover alone: K and which halves the cover
//! has are what a kernel declines it for.
//! Stage two is decided by the mask it will see, of which only the expected
//! hit count and the row count are known here.
//! [`scan_ns`] adds the two to the alignment walk, so covers of different
//! shape and selectivity compare on one number. Every constant is a fit under
//! `bench::fit` to the sweeps in `bench`.

use super::super::cover::ProbeCover;
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use super::matcher::simd_available;
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use super::matcher::{MAX_BATCHES, PER_BATCH};
use super::{BLOCK, Isa};

/// What the planner reads about the region. No code values are inspected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::search::substring::prefilter) struct Facts {
    /// Codes the mask is expected to set a bit for, over the whole region.
    pub(in crate::search::substring::prefilter) expected_hits: usize,
    /// Codes in the region, which with the hits is the density the
    /// pack-skipping flag turns on.
    pub(in crate::search::substring::prefilter) code_count: usize,
    pub(in crate::search::substring::prefilter) row_count: usize,
}

/// What stage one's cost depends on: the cover's counts, with no code
/// values. A planner holds these before it has a cover.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::search::substring::prefilter) struct Shape {
    /// K.
    pub(in crate::search::substring::prefilter) tokens: usize,
    /// R.
    pub(in crate::search::substring::prefilter) ranges: usize,
}

impl Shape {
    pub(super) fn of(cover: &ProbeCover) -> Self {
        Self {
            tokens: cover.points().len(),
            ranges: cover.ranges().len(),
        }
    }
}

/// Stage one. Every kernel takes the ranges of a cover beside its tokens,
/// so this is the kernel alone. The compare, the range predicate and the
/// batched bitmap run on all three vector sets; the byte table is scalar and
/// runs anywhere.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Match {
    /// One byte per code of the whole code space. Takes every cover, so it
    /// is what a target with no vector kernel runs, and where the ladder ends
    /// on one that has them.
    Table,
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    EqOr,
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    Range,
    /// `ceil(K / 8)` batches of eight tokens, ORed, up to [`MAX_BATCHES`] of
    /// them: three, and two on AVX2, which has the registers for no more.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    NibbleN8K,
}

/// Stage two.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Resolve {
    LinearSeek,
    GallopSeek,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::search::substring::prefilter::scan) struct Plan {
    pub(super) matcher: Match,
    pub(super) resolver: Resolve,
    /// Whether the matcher tests its lanes before packing them, which is a
    /// const the kernel is compiled for rather than a branch it takes.
    pub(super) skip: bool,
}

/// What one code weighs, so that a cost per code reads as a rate per byte.
#[cfg(test)]
pub(super) const BYTES_PER_CODE: f64 = 2.0;

pub(super) fn select(cover: &ProbeCover, facts: Facts) -> Plan {
    Plan {
        matcher: select_matcher(Shape::of(cover)),
        resolver: select_resolver(facts),
        skip: skip_ns_per_code(density(facts)) < 0.0,
    }
}

/// Bits the mask is expected to set per code over the region, which is what
/// the pack-skipping flag turns on.
fn density(facts: Facts) -> f64 {
    match facts.code_count {
        0 => 0.0,
        codes => facts.expected_hits as f64 / codes as f64,
    }
}

/// Codes one pass of the pack covers, which is the two mask words
/// [`words`](super::matcher) writes at a time and the unit the flag decides
/// over.
const PACK_GROUP: f64 = 128.0;

/// What `SKIP_MOVEMASK_IF_NO_MATCH` adds, in ns per code: the lane OR it pays on
/// every group, less the pack it saves on the groups with no match. Both are
/// fitted to the two ends of the sweep and land on their instruction counts.
/// Negative is worth taking. See `README.md`.
fn skip_ns_per_code(density: f64) -> f64 {
    /// The lane OR and its cross-domain read, then the pack and its store,
    /// both ns per group and both fitted to the two ends of the sweep.
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512bw")))]
    const COST: (f64, f64) = (0.75, 1.01);
    /// On AVX-512 the compare has already written the mask word, so there is
    /// no pack to skip and the test is a `ktest`: the flag can only cost.
    /// The 2026-09-08 sweep measured it positive at every selectivity, from
    /// 0.0007 ns per code on a stream nothing hits to 0.030 where the branch
    /// stops predicting, against nothing saved.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    const COST: (f64, f64) = (0.35, 0.0);
    let (reduction, pack) = COST;
    // The share of groups with no match, if the hits fall evenly. They
    // cluster, so the break-even is an order and not a point.
    let no_match = (-PACK_GROUP * density).exp();
    (reduction - pack * no_match) / PACK_GROUP
}

/// The cheapest kernel [`ns_per_code`] models that [`takes`] the cover.
/// See `README.md`.
fn select_matcher(shape: Shape) -> Match {
    const KERNELS: &[Match] = &[
        Match::Table,
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        Match::EqOr,
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        Match::Range,
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        Match::NibbleN8K,
    ];
    KERNELS
        .iter()
        .copied()
        .filter(|&kernel| takes(kernel, shape))
        .min_by(|&a, &b| ns_per_code(a, shape).total_cmp(&ns_per_code(b, shape)))
        .expect("the byte table takes every cover")
}

/// Which covers each kernel holds: its limit on K and which lists it reads.
/// The kernels do not check this themselves.
pub(super) fn takes(kernel: Match, shape: Shape) -> bool {
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    let Shape { tokens, ranges } = shape;
    match kernel {
        Match::Table => true,
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        Match::EqOr => simd_available() && tokens > 0,
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        Match::Range => simd_available() && tokens == 0 && ranges > 0,
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        Match::NibbleN8K => {
            simd_available() && tokens > 0 && tokens.div_ceil(PER_BATCH) <= MAX_BATCHES
        }
    }
}

/// What the planned matcher costs per code at this hit density: the kernel
/// [`select_matcher`] picks plus the pack-skipping flag where it pays. The
/// scalar kernels have no pack to skip.
pub(in crate::search::substring::prefilter) fn stage_one_ns_per_code(
    shape: Shape,
    density: f64,
) -> f64 {
    let matcher = select_matcher(shape);
    let skip = match matcher {
        Match::Table => 0.0,
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        _ => skip_ns_per_code(density).min(0.0),
    };
    ns_per_code(matcher, shape) + skip
}

/// Nanoseconds per code, one function per instruction set, because the same
/// kernel is a different cost on each: a range costs AVX-512 half what it
/// costs NEON, and a bitmap batch half again. Each is `bench::fit` on that
/// set's own sweep, and each is a table, so `README.md` reads as its rows.
/// The dispatch folds away, [`Isa::BUILT`] being a const.
pub(super) fn ns_per_code(matcher: Match, shape: Shape) -> f64 {
    match Isa::BUILT {
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon(matcher, shape),
        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512bw")))]
        Isa::Avx2 => avx2(matcher, shape),
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
        Isa::Avx512Bw => avx512bw(matcher, shape),
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        Isa::Scalar => scalar(matcher, shape),
        _ => unreachable!("a build compiles one set, and it is Isa::BUILT"),
    }
}

/// Fitted on `ch/hits/URL_1m` on an Apple M4 Pro, 2026-09-08, every kernel
/// within 3.4% of its rows.
#[cfg(target_arch = "aarch64")]
fn neon(matcher: Match, shape: Shape) -> f64 {
    let k = shape.tokens as f64;
    let r = shape.ranges as f64;
    let batches = shape.tokens.div_ceil(PER_BATCH) as f64;
    match matcher {
        Match::Table => 0.195,
        Match::EqOr => 0.0043 + 0.0144 * k + 0.0205 * r,
        Match::NibbleN8K => 0.0228 + 0.0325 * batches + 0.0213 * r,
        Match::Range => 0.0043 + 0.0212 * r,
    }
}

/// Fitted on `ch/hits/URL_1m` on an Intel Xeon 6975P-C, 2026-09-08. The
/// stream was cut to fit L2 there, because on a 4 Mcode one that core pins
/// every vector kernel at 31 GB/s, and a fit through capped rows describes
/// the memory system rather than the kernel. So these are what the kernels
/// cost when fed; nothing above 77 GB/s was reachable.
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512bw")))]
fn avx2(matcher: Match, shape: Shape) -> f64 {
    let k = shape.tokens as f64;
    let r = shape.ranges as f64;
    let batches = shape.tokens.div_ceil(PER_BATCH) as f64;
    match matcher {
        Match::Table => 0.201,
        Match::EqOr => 0.0136 + 0.0105 * k + 0.0239 * r,
        // NOT MEASURED. The kernel is newer than the sweep, so this row is
        // the AVX-512 row's vector terms at 2.2x: twice for the half-width
        // lanes, and a fifth again for the compare pair AVX2 spends where
        // `vptestmb` writes the mask word for nothing. The range slope is
        // the measured one from the rows above. Re-run `bench::fit` on an
        // AVX2 build and paste its block over this.
        Match::NibbleN8K => 0.0594 + 0.0310 * batches + 0.0225 * r,
        Match::Range => 0.0084 + 0.0225 * r,
    }
}

/// Fitted on the same core and stream as [`avx2`], built for AVX-512. The
/// mask register is what these show: a range costs half what it costs NEON
/// and a bitmap batch half again, neither of them paying for a pack.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
fn avx512bw(matcher: Match, shape: Shape) -> f64 {
    let k = shape.tokens as f64;
    let r = shape.ranges as f64;
    let batches = shape.tokens.div_ceil(PER_BATCH) as f64;
    match matcher {
        Match::Table => 0.204,
        Match::EqOr => 0.0079 + 0.0113 * k + 0.0127 * r,
        Match::NibbleN8K => 0.0270 + 0.0141 * batches + 0.0119 * r,
        Match::Range => 0.0085 + 0.0112 * r,
    }
}

/// No vector kernels, so the two scalar ones are the whole ladder.
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn scalar(matcher: Match, shape: Shape) -> f64 {
    match matcher {
        Match::Table => 0.195,
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
pub(super) fn seek_ns_per_row(resolver: Resolve, g: f64) -> f64 {
    match resolver {
        Resolve::LinearSeek => LINEAR_SEEK_ROW_NS + LINEAR_SEEK_CROSS_NS * g,
        Resolve::GallopSeek => GALLOP_SEEK_STEP_NS * (1.0 + g).log2(),
    }
}

/// Stage two over `words` mask words in blocks that hit, emitting `emitted`
/// rows with `crossed` rows walked or searched past on the way.
pub(super) fn stage_two_ns(resolver: Resolve, words: f64, emitted: f64, crossed: f64) -> f64 {
    if emitted <= 0.0 {
        return WORD_NS * words;
    }
    WORD_NS * words + emitted * seek_ns_per_row(resolver, crossed / emitted)
}

/// Rows the mask is expected to name: a row of `x` expected hits holds one
/// with probability `1 - exp(-x)`, which the resolver sweep measured to hold
/// within 20%.
fn hit_rows(expected_hits: f64, rows: f64) -> f64 {
    rows * (1.0 - (-expected_hits / rows).exp())
}

/// Hits per row is the whole of it: it fixes both the rows crossed to reach
/// a hit row and the hits found once there, so the model decides on the rows
/// crossed per emitted row alone and the row length enters through it.
fn select_resolver(facts: Facts) -> Resolve {
    let rows = facts.row_count as f64;
    let g = rows / hit_rows(facts.expected_hits as f64, rows).max(1.0);
    if seek_ns_per_row(Resolve::GallopSeek, g) < seek_ns_per_row(Resolve::LinearSeek, g) {
        Resolve::GallopSeek
    } else {
        Resolve::LinearSeek
    }
}

/// The stream a cover would be scanned over.
#[derive(Clone, Copy, Debug)]
pub(in crate::search::substring::prefilter) struct Region {
    pub(in crate::search::substring::prefilter) code_count: usize,
    pub(in crate::search::substring::prefilter) row_count: usize,
}

/// Expected nanoseconds to scan `cover` over `region` and walk every hit,
/// given the `covered` codes it matches there: stage one at the kernel the
/// shape gets, stage two at the cheaper resolver, the walk per hit. An empty
/// cover proves no row matches and costs nothing.
pub(in crate::search::substring::prefilter) fn scan_ns(
    cover: &ProbeCover,
    covered: u32,
    region: Region,
) -> f64 {
    if cover.is_empty() || region.code_count == 0 || region.row_count == 0 {
        return 0.0;
    }
    let codes = region.code_count as f64;
    let rows = region.row_count as f64;
    let covered = f64::from(covered);
    let shape = Shape {
        tokens: cover.points().len(),
        ranges: cover.ranges().len(),
    };
    let stage_one = codes * stage_one_ns_per_code(shape, covered / codes);

    let emitted = hit_rows(covered, rows);
    let blocks_hit = 1.0 - (-covered / codes * BLOCK as f64).exp();
    let words = codes / 64.0 * blocks_hit;
    let stage_two = [Resolve::LinearSeek, Resolve::GallopSeek]
        .into_iter()
        .map(|resolver| stage_two_ns(resolver, words, emitted, rows))
        .fold(f64::MAX, f64::min);

    stage_one + stage_two + covered * WALK_NS_PER_HIT
}
