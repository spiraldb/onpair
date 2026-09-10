// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scans of the code stream against a compiled cover: codes to bit mask,
//! bit mask to rows. See README.md for the contract between the halves.
//!
//! The scan takes any cover with something in it - the points as tokens, the
//! ranges beside them, or either alone - and picks its own two halves from a
//! cost model. Every hit is verified against the alignment walk in the
//! compressed domain, so the rows are exact.
//!
//! Vector kernels are compiled for NEON, AVX2 and AVX-512; a target with
//! none, or an x86 without AVX2, runs the byte table instead, which is
//! scalar and portable. The scalar routine under `cfg(test)` is the
//! correctness oracle.

#[cfg(test)]
mod bench;
mod dispatch;
mod matcher;
mod policy;
mod resolver;
mod walk;

use matcher::Matcher;
pub(in crate::search::prefilter) use policy::{Facts, Region, scan_ns};
use resolver::Resolver;
pub(in crate::search::prefilter) use walk::Walk;

use super::PrefilterAnalysis;
use super::cover::ProbeCover;
use crate::core::dictionary::CompactDictionaryView;
use crate::core::offset::Offset;
use crate::core::types::Token;

/// Borrowed buffers for one scan region.
#[derive(Clone, Copy)]
pub(super) struct ScanInput<'a, O> {
    pub(super) codes: &'a [Token],
    pub(super) row_offsets: &'a [O],
    pub(super) cover: &'a ProbeCover,
}

impl<'a, O> ScanInput<'a, O> {
    pub(super) const fn full(
        codes: &'a [Token],
        row_offsets: &'a [O],
        cover: &'a ProbeCover,
    ) -> Self {
        Self {
            codes,
            row_offsets,
            cover,
        }
    }
}

/// What the scan will be handed, without inspecting code values. `None` where
/// the region admits nothing whatever runs over it: an empty cover proves no
/// row matches, and a region with no rows has nothing to emit.
pub(super) type ScanPlan = Option<Facts>;

/// Derive an ephemeral plan without inspecting code values.
#[inline]
pub(super) fn plan<O: Offset>(input: ScanInput<'_, O>, analysis: &PrefilterAnalysis) -> ScanPlan {
    facts(
        input,
        analysis.covered_frequency() as usize,
        analysis.total_frequency() as usize,
    )
}

/// The region the scan will see and the hits it can expect there. Which
/// kernel runs over it is `policy`'s business, decided once the probe
/// is built.
fn facts<O: Offset>(
    input: ScanInput<'_, O>,
    covered_frequency: usize,
    total_frequency: usize,
) -> ScanPlan {
    let row_count = input.row_offsets.len().saturating_sub(1);
    if input.cover.is_empty() || row_count == 0 {
        return None;
    }
    Some(Facts {
        expected_hits: expected_hits(covered_frequency, total_frequency, input.codes.len()),
        code_count: input.codes.len(),
        row_count,
    })
}

/// Covered codes expected in this region: exact for the indexed population,
/// and a projection of it for a subset. Planning only, never correctness.
fn expected_hits(covered_frequency: usize, total_frequency: usize, code_count: usize) -> usize {
    if total_frequency == code_count {
        return covered_frequency;
    }
    if total_frequency == 0 {
        return 0;
    }
    let projected =
        (covered_frequency as u128).saturating_mul(code_count as u128) / total_frequency as u128;
    usize::try_from(projected).unwrap_or(usize::MAX)
}

/// Compatibility entry for tests that exercise dispatch with a synthetic
/// cover, without constructing a complete analysis.
#[cfg(test)]
pub(super) fn scan<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    cover: &ProbeCover,
    covered_frequency: usize,
    out: &mut Vec<usize>,
) {
    let input = ScanInput::full(codes, row_offsets, cover);
    execute_check(
        facts(input, covered_frequency, codes.len()),
        input,
        Check::Superset,
        out,
    );
}

/// Execute a previously derived plan. This is the first stage that inspects
/// code values. Every hit goes through `walk`, so the rows are exact.
#[inline]
pub(super) fn execute<O: Offset>(
    plan: ScanPlan,
    input: ScanInput<'_, O>,
    walk: &Walk,
    dict: CompactDictionaryView<'_>,
    out: &mut Vec<usize>,
) {
    let codes = input.codes;
    execute_check(plan, input, Check::Walk { walk, dict, codes }, out);
}

/// The plan under whatever stage two asks of a hit.
fn execute_check<O: Offset>(
    plan: ScanPlan,
    input: ScanInput<'_, O>,
    check: Check<'_>,
    out: &mut Vec<usize>,
) {
    let Some(facts) = plan else {
        return;
    };
    dispatch::run(
        policy::select(input.cover, facts),
        input.cover,
        input.codes,
        input.row_offsets,
        check,
        out,
    );
}

/// Test oracle for the split scan.
#[cfg(test)]
pub(super) fn scan_scalar<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    out: &mut Vec<usize>,
) {
    for row in 0..row_offsets.len().saturating_sub(1) {
        let a = row_offsets[row].to_usize();
        let b = row_offsets[row + 1].to_usize();
        if codes[a..b].iter().any(|&code| pf.contains(code)) {
            out.push(row);
        }
    }
}

/// Codes per block, 64 words of mask. Kernels rely on the multiple of 64.
pub(in crate::search::prefilter) const BLOCK: usize = 4096;

/// The instruction set this build's kernels run on. The other thing a cost
/// is per: the same kernel is a different cost on each, since what a range
/// or a pack costs is the set's business. `Scalar` is a target with no
/// vector kernels at all, where only the byte table is compiled.
///
/// Total by design though a build compiles one of them: a fit reads other
/// machines' numbers out of a CSV and has to name their sets.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::search::prefilter) enum Isa {
    Neon,
    Avx2,
    Avx512Bw,
    Scalar,
}

impl Isa {
    /// The one this build compiled. A const, so the dispatch over it folds
    /// away and the other sets' costs are never asked for.
    #[cfg(target_arch = "aarch64")]
    pub(in crate::search::prefilter) const BUILT: Self = Self::Neon;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    pub(in crate::search::prefilter) const BUILT: Self = Self::Avx512Bw;
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512bw")))]
    pub(in crate::search::prefilter) const BUILT: Self = Self::Avx2;
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    pub(in crate::search::prefilter) const BUILT: Self = Self::Scalar;
}

/// One block of codes, exactly what a matcher is handed.
type Block = [Token; BLOCK];
/// One block of mask, bit `i` for code `i` of the block.
type Mask = [u64; BLOCK / 64];

/// What stage two asks of a hit before its row goes out. Every scan the
/// planner drives applies the walk; `Superset` is how the benches time the
/// two stages without the walk's per-hit cost inside them.
#[derive(Clone, Copy)]
enum Check<'a> {
    /// Nothing: every hit's row is a candidate.
    #[cfg(test)]
    Superset,
    /// The alignment walk over the codes: only a hit some occurrence of the
    /// needle parses through has its row emitted.
    Walk {
        walk: &'a Walk,
        dict: CompactDictionaryView<'a>,
        codes: &'a [Token],
    },
}

impl Check<'_> {
    /// Whether the hit at stream index `code`, inside the row
    /// `[row_start, row_end)`, passes.
    #[inline]
    fn passes(&self, code: usize, row_start: usize, row_end: usize) -> bool {
        match *self {
            #[cfg(test)]
            Self::Superset => true,
            Self::Walk { walk, dict, codes } => walk.check(dict, codes, row_start, row_end, code),
        }
    }
}

/// Both stages over a whole stream. Rows come out ascending and unique.
fn both_stages<'a, M: Matcher, R: Resolver<'a>>(
    cover: &ProbeCover,
    codes: &[Token],
    row_offsets: &'a [R::Offset],
    check: Check<'_>,
    out: &mut Vec<usize>,
) {
    let matcher = M::new(cover);
    let mut resolver = R::new(row_offsets);
    let mut bits = [0u64; BLOCK / 64];
    blocks(codes, &mut |block, at, valid| {
        // A block the matcher reports empty costs stage two nothing.
        if matcher.check(block, &mut bits) {
            clear_from(&mut bits, valid);
            resolver.rows(&bits, at, check, out);
        }
    });
}

/// Walks the stream in blocks, straight out of `codes` while a whole [`Block`]
/// remains and through a zero-padded copy at the end. `step(block, at, valid)`
/// gets the stream index of the block and how many of its codes are the
/// stream's; bits past `valid` are the padding's and must go.
fn blocks(codes: &[Token], step: &mut dyn FnMut(&Block, usize, usize)) {
    let mut at = 0;
    while at + BLOCK <= codes.len() {
        step(codes[at..at + BLOCK].try_into().unwrap(), at, BLOCK);
        at += BLOCK;
    }
    if at < codes.len() {
        let rest = codes.len() - at;
        let mut tail = [Token::default(); BLOCK];
        tail[..rest].copy_from_slice(&codes[at..]);
        step(&tail, at, rest);
    }
}

/// Clears the bits from position `valid` on.
fn clear_from(bits: &mut Mask, valid: usize) {
    if valid < BLOCK {
        bits[valid / 64] &= (1u64 << (valid % 64)) - 1;
        bits[valid / 64 + 1..].fill(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::TokenRange;

    /// What the scan is handed, and what it declines.
    #[test]
    fn the_scan_takes_a_cover_it_can_probe() {
        let codes = [0 as Token; 8];
        let rows = [0u32, 4, 8];
        let takes = |points: Vec<Token>, ranges: Vec<TokenRange>| {
            let cover = ProbeCover { points, ranges };
            facts(ScanInput::full(&codes, &rows, &cover), 1, codes.len()).is_some()
        };
        assert!(takes(vec![7], Vec::new()), "a point");
        assert!(
            takes(Vec::new(), vec![TokenRange { begin: 1, last: 9 }]),
            "a range"
        );
        assert!(
            takes(vec![7], vec![TokenRange { begin: 1, last: 9 }]),
            "both"
        );
        assert!(!takes(Vec::new(), Vec::new()), "a cover with nothing in it");

        let cover = ProbeCover {
            points: vec![7],
            ranges: Vec::new(),
        };
        let rowless: &[u32] = &[0];
        assert!(facts(ScanInput::full(&codes, rowless, &cover), 1, 8).is_none());
    }

    /// The facts the scan plans its own halves from.
    #[test]
    fn the_scan_is_handed_the_region_it_will_see() {
        let codes = [0 as Token; 2_000];
        let rows: Vec<u32> = (0..=200).map(|row| row * 10).collect();
        let cover = ProbeCover {
            points: vec![7],
            ranges: Vec::new(),
        };
        assert_eq!(
            facts(ScanInput::full(&codes, &rows, &cover), 500, 10_000),
            Some(Facts {
                // A twentieth of the indexed codes are covered, so a
                // twentieth of the region's.
                expected_hits: 100,
                code_count: 2_000,
                row_count: 200,
            })
        );
    }

    /// The projection at its ends: the whole index, none of it, and a region
    /// wider than the index it was measured over.
    #[test]
    fn the_hit_estimate_is_a_projection_of_the_index() {
        assert_eq!(expected_hits(7, 100, 100), 7);
        assert_eq!(expected_hits(500, 10_000, 2_000), 100);
        assert_eq!(expected_hits(1, 0, 100), 0);
        assert_eq!(expected_hits(1, 1, usize::MAX), usize::MAX);
    }
}
