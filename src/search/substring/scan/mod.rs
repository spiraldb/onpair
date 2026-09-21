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
use super::plan::{CoverShape, probe_density, select_matcher_config};
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

/// Scan a synthetic cover for tests, returning candidate rows without verification.
#[cfg(test)]
pub(super) fn scan<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    cover: &ProbeCover,
    covered_frequency: usize,
    out: &mut Vec<usize>,
) {
    if codes.is_empty() || row_offsets.len() < 2 {
        return;
    }
    let input = ScanInput::new(codes, row_offsets, cover);
    execute_check(
        select_matcher_config(
            detect_target_caps(),
            CoverShape::of(cover),
            probe_density(covered_frequency, codes.len(), codes.len()),
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

    /// Point, range, mixed, and empty covers select the expected candidate rows.
    #[test]
    fn the_scan_takes_a_cover_it_can_probe() {
        let codes = [1, 7, 15];
        let rows = [0u32, 1, 2, 3];
        let range = TokenRange {
            begin: 10,
            last: 20,
        };
        for (points, ranges, expected) in [
            (vec![7], vec![], vec![1]),
            (vec![], vec![range], vec![2]),
            (vec![7], vec![range], vec![1, 2]),
            (vec![], vec![], vec![]),
        ] {
            let mut out = Vec::new();
            scan(&codes, &rows, &ProbeCover { points, ranges }, 1, &mut out);
            assert_eq!(out, expected);
        }
    }

    /// Empty code or row buffers leave existing output untouched at either width.
    #[test]
    fn rows_without_codes_make_no_candidate() {
        let cover = ProbeCover {
            points: vec![7],
            ranges: Vec::new(),
        };
        let mut out = vec![usize::MAX];
        scan(&[], &[0u32, 0, 0, 0], &cover, 0, &mut out);
        scan(&[], &[0u64, 0, 0, 0], &cover, 0, &mut out);
        for offsets in [&[][..], &[0u32][..]] {
            scan(&[7], offsets, &cover, 1, &mut out);
            let wide: Vec<u64> = offsets.iter().copied().map(u64::from).collect();
            scan(&[7], &wide, &cover, 1, &mut out);
        }
        assert_eq!(out, [usize::MAX]);
    }
}
