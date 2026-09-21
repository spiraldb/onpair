// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execute a prepared cover and append exact matching rows.
//!
//! The caller supplies a matcher configuration for the current buffers. The matcher turns
//! blocks of token codes into bit masks; the resolver maps set
//! bits to rows and asks `verify::walk` to check each candidate occurrence.
//! A row is appended on its first confirmed hit, then its later hits are skipped.
//!
//! `dispatch` connects the configuration to a concrete matcher. `matcher` owns token
//! membership checks; `resolver` owns row lookup and duplicate suppression.
//! Empty-pattern handling belongs to `ContainsScan::scan`, before this module.

mod dispatch;
mod matcher;
mod resolver;

use super::ProbeCover;
use super::plan::MatcherConfig;
#[cfg(test)]
use super::plan::{AnalysisFacts, CoverShape, RegionFacts, ScanFacts, select_matcher_config};
use super::verify::walk::Walk;
use crate::core::dictionary::CompactDictionaryView;
use crate::core::offset::Offset;
use crate::core::types::Token;
pub(super) use dispatch::detect_target_caps;
use matcher::Matcher;
pub(super) use matcher::PER_BATCH;
use resolver::Resolver;

/// Token codes processed in one matcher block.
pub(super) const BLOCK: usize = 4096;

/// Borrowed buffers for one scan region.
#[derive(Clone, Copy)]
pub(super) struct ScanInput<'a, O> {
    codes: &'a [Token],
    row_offsets: &'a [O],
    cover: &'a ProbeCover,
}

impl<'a, O> ScanInput<'a, O> {
    /// Borrow the code stream, row boundaries, and prepared probe cover.
    pub(super) const fn new(
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

/// Execute the selected matcher and append exact matching row indices.
/// The caller supplies a cover and walker prepared for the same pattern and
/// dictionary, and a configuration eligible for this cover and the current CPU.
#[inline]
pub(super) fn matches<O: Offset>(
    config: MatcherConfig,
    input: ScanInput<'_, O>,
    dict: CompactDictionaryView<'_>,
    walk: &Walk,
    out: &mut Vec<usize>,
) {
    execute_check(
        config,
        input,
        Check::Walk {
            walk,
            dict,
            codes: input.codes,
        },
        out,
    );
}

/// Combine prepared frequency counts with the actual code and row counts.
#[cfg(test)]
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

/// Scan a synthetic cover for tests, returning candidate rows without verification.
#[cfg(test)]
pub(super) fn scan<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    cover: &ProbeCover,
    covered_frequency: usize,
    out: &mut Vec<usize>,
) {
    let input = ScanInput::new(codes, row_offsets, cover);
    execute_check(
        select_matcher_config(
            detect_target_caps(),
            facts(input, covered_frequency, codes.len()),
        ),
        input,
        Check::Superset,
        out,
    );
}

/// Dispatch a nonempty matcher configuration with the requested hit verification.
fn execute_check<O: Offset>(
    config: MatcherConfig,
    input: ScanInput<'_, O>,
    check: Check<'_>,
    out: &mut Vec<usize>,
) {
    if config == MatcherConfig::Empty {
        return;
    }
    dispatch::run(
        config,
        input.cover,
        input.codes,
        input.row_offsets,
        check,
        out,
    );
}

/// Token codes passed to a matcher in one call.
type Block = [Token; BLOCK];
/// One bit per code in a block; bit `i` describes code `i`.
type Mask = [u64; BLOCK / 64];

/// Verification applied before a candidate row is appended.
/// Production scans use the graph walker. Tests can accept all
/// probe hits to exercise matching and row lookup independently.
#[derive(Clone, Copy)]
enum Check<'a> {
    /// Accept every probe hit without checking the substring.
    #[cfg(test)]
    Superset,
    /// Confirm a complete occurrence through this hit using the alignment graph.
    Walk {
        walk: &'a Walk,
        dict: CompactDictionaryView<'a>,
        codes: &'a [Token],
    },
}

impl Check<'_> {
    /// Check the hit at absolute code index `code` within `[row_start, row_end)`.
    #[inline]
    fn passes(&self, code: usize, row_start: usize, row_end: usize) -> bool {
        match *self {
            #[cfg(test)]
            Self::Superset => true,
            Self::Walk { walk, dict, codes } => walk.check(dict, codes, row_start, row_end, code),
        }
    }
}

/// Match blocks, verify candidate hits, and append each matching row once.
/// Matcher setup, mask storage, and resolver state are reused across blocks.
/// Appended indices are ascending; existing output contents are preserved.
fn both_stages<M: Matcher, O: Offset>(
    cover: &ProbeCover,
    codes: &[Token],
    row_offsets: &[O],
    check: Check<'_>,
    out: &mut Vec<usize>,
) {
    let matcher = M::new(cover);
    let mut resolver = Resolver::new(row_offsets);
    let mut bits = [0u64; BLOCK / 64];
    blocks(codes, &mut |block, at, valid| {
        // A false result guarantees no hits; true may still mean an empty mask.
        if matcher.check(block, &mut bits) {
            clear_from(&mut bits, valid);
            resolver.rows(&bits, at, check, out);
        }
    });
}

/// Visit full blocks directly and copy the final partial block into zero padding.
/// The callback receives the block, its absolute start, and its valid code count.
/// It must discard mask bits beyond that count before resolving rows.
fn blocks(codes: &[Token], step: &mut dyn FnMut(&Block, usize, usize)) {
    let (whole, rest) = codes.as_chunks::<BLOCK>();
    let mut at = 0;
    for block in whole {
        step(block, at, BLOCK);
        at += BLOCK;
    }
    if !rest.is_empty() {
        let mut tail = [Token::default(); BLOCK];
        tail[..rest.len()].copy_from_slice(rest);
        step(&tail, at, rest.len());
    }
}

/// Clear mask bits at or beyond the valid code count, excluding padded tokens.
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

    /// Nonempty covers need a kernel when both rows and codes are present.
    #[test]
    fn the_scan_takes_a_cover_it_can_probe() {
        let codes = [0 as Token; 8];
        let rows = [0u32, 4, 8];
        let takes = |points: Vec<Token>, ranges: Vec<TokenRange>| {
            let cover = ProbeCover { points, ranges };
            select_matcher_config(
                detect_target_caps(),
                facts(ScanInput::new(&codes, &rows, &cover), 1, codes.len()),
            ) != MatcherConfig::Empty
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
            select_matcher_config(
                detect_target_caps(),
                facts(ScanInput::new(&codes, rowless, &cover), 1, 8)
            ) == MatcherConfig::Empty
        );
    }

    /// Empty rows contain no candidate tokens at either offset width.
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

    /// Execution facts retain indexed counts alongside the actual region size.
    #[test]
    fn the_scan_is_handed_the_region_it_will_see() {
        let codes = [0 as Token; 2_000];
        let rows: Vec<u32> = (0..=200).map(|row| row * 10).collect();
        let cover = ProbeCover {
            points: vec![7],
            ranges: Vec::new(),
        };
        assert_eq!(
            facts(ScanInput::new(&codes, &rows, &cover), 500, 10_000),
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
                    // Projection uses these actual sizes, not the index size.
                    code_count: 2_000,
                    row_count: 200,
                }
            }
        );
    }
}
