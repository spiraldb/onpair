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
mod resolver;

use super::PrefilterAnalysis;
use super::ProbeCover;
pub(super) use super::plan::facts::BLOCK;
#[cfg(test)]
use super::plan::facts::Isa;
use super::plan::facts::{AnalysisFacts, CoverShape, Kernel, RegionFacts, ScanFacts, ScanPlan};
use super::plan::select::select_scan_plan;
use super::verify::walk::Walk;
use crate::core::dictionary::CompactDictionaryView;
use crate::core::offset::Offset;
use crate::core::types::Token;
pub(super) use dispatch::detect_target_caps;
use matcher::Matcher;
use resolver::Resolver;

/// Borrowed buffers for one scan region.
#[derive(Clone, Copy)]
struct ScanInput<'a, O> {
    codes: &'a [Token],
    row_offsets: &'a [O],
    cover: &'a ProbeCover,
}

impl<'a, O> ScanInput<'a, O> {
    const fn full(codes: &'a [Token], row_offsets: &'a [O], cover: &'a ProbeCover) -> Self {
        Self {
            codes,
            row_offsets,
            cover,
        }
    }
}

/// Append exact matches while keeping region planning and execution private.
#[inline]
pub(super) fn matches<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    dict: CompactDictionaryView<'_>,
    analysis: &PrefilterAnalysis,
    out: &mut Vec<usize>,
) {
    let input = ScanInput::full(codes, row_offsets, analysis.probe_cover());
    let plan = select_scan_plan(
        detect_target_caps(),
        facts(
            input,
            analysis.covered_frequency() as usize,
            analysis.total_frequency() as usize,
        ),
    );
    execute_check(
        plan,
        input,
        Check::Walk {
            walk: &analysis.walk,
            dict,
            codes,
        },
        out,
    );
}

fn facts<O: Offset>(
    input: ScanInput<'_, O>,
    covered_codes: usize,
    indexed_codes: usize,
) -> ScanFacts {
    ScanFacts {
        analysis: AnalysisFacts {
            shape: CoverShape::of(input.cover),
            covered_codes,
            indexed_codes,
        },
        region: RegionFacts {
            code_count: input.codes.len(),
            row_count: input.row_offsets.len().saturating_sub(1),
        },
    }
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
        select_scan_plan(
            detect_target_caps(),
            facts(input, covered_frequency, codes.len()),
        ),
        input,
        Check::Superset,
        out,
    );
}

/// The plan under whatever stage two asks of a hit.
fn execute_check<O: Offset>(
    plan: ScanPlan,
    input: ScanInput<'_, O>,
    check: Check<'_>,
    out: &mut Vec<usize>,
) {
    if plan.kernel == Kernel::Empty {
        return;
    }
    dispatch::run(
        plan,
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
            select_scan_plan(
                detect_target_caps(),
                facts(ScanInput::full(&codes, &rows, &cover), 1, codes.len()),
            )
            .kernel
                != Kernel::Empty
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
        assert!(
            select_scan_plan(
                detect_target_caps(),
                facts(ScanInput::full(&codes, rowless, &cover), 1, 8)
            )
            .kernel
                == Kernel::Empty
        );
    }

    /// Rows but no codes: the block driver has nothing to hand the matcher,
    /// and the empty rows must come out as no candidates rather than a panic.
    #[test]
    fn rows_without_codes_make_no_candidate() {
        let cover = ProbeCover {
            points: vec![7],
            ranges: Vec::new(),
        };
        let mut out = Vec::new();
        scan(&[], &[0u32, 0, 0, 0], &cover, 0, &mut out);
        assert!(out.is_empty());
        scan(&[], &[0u64, 0, 0, 0], &cover, 0, &mut out);
        assert!(out.is_empty());
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
            ScanFacts {
                analysis: AnalysisFacts {
                    shape: CoverShape {
                        points: 1,
                        ranges: 0
                    },
                    covered_codes: 500,
                    indexed_codes: 10000
                },
                region: RegionFacts {
                    // A twentieth of the indexed codes are covered, so a
                    // twentieth of the region's.
                    code_count: 2_000,
                    row_count: 200,
                }
            }
        );
    }
}
