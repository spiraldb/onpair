// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Row resolution: map hit bits to rows and append the verified ones.
//!
//! Blocks arrive in stream order, so the resolver keeps the current row across
//! blocks. A hit in a later row is located by galloping forward from the next row
//! and binary-searching the last step, in time logarithmic in the rows skipped.
//! Once a row is appended, its remaining hits are skipped, even in later blocks.

use super::{Mask, ScanInput};
use crate::core::dictionary::CompactDictionaryView;
use crate::core::offset::Offset;
use crate::core::types::Token;
use crate::search::substring::probe::verify::HitVerifier;

/// Row lookup and verification state, kept across blocks.
pub(super) struct Resolver<'a, O> {
    /// Boundaries of rows in absolute code indices.
    row_offsets: &'a [O],
    /// Codes, dictionary and hit verifier used to check each hit.
    codes: &'a [Token],
    dict: CompactDictionaryView<'a>,
    verifier: &'a HitVerifier,
    /// Last row located for a hit, initially zero.
    row: usize,
    /// Exclusive end of the current row, initialized from the first row boundary.
    row_end: usize,
    /// End of the last appended row; hits before it are skipped.
    resolved_code: usize,
}

impl<'a, O: Offset> Resolver<'a, O> {
    /// Starts at the first row of `input`.
    pub(super) fn new(input: ScanInput<'a, O>) -> Self {
        Self {
            row_offsets: input.row_offsets,
            codes: input.codes,
            dict: input.dict,
            verifier: input.verifier,
            row: 0,
            row_end: input.row_offsets.get(1).map_or(0, |x| x.to_usize()),
            resolved_code: 0,
        }
    }

    /// Verifies the hits in `bits`, the mask of the block starting at `first_code`,
    /// and appends their rows. Blocks must arrive in increasing code order.
    #[inline]
    pub(super) fn rows(&mut self, bits: &Mask, first_code: usize, out: &mut Vec<usize>) {
        let mut at = self.resolved_code.saturating_sub(first_code);
        while let Some(hit) = next_set(bits, at) {
            let code = first_code + hit;
            if code >= self.row_end {
                self.row = find_row(self.row_offsets, self.row + 1, code);
                self.row_end = self.row_offsets[self.row + 1].to_usize();
            }
            let row_start = self.row_offsets[self.row].to_usize();
            if self
                .verifier
                .check(self.dict, self.codes, row_start, self.row_end, code)
            {
                out.push(self.row);
                self.resolved_code = self.row_end;
                at = self.row_end - first_code;
            } else {
                at = hit + 1;
            }
        }
    }
}

/// The first set bit at or after `from`, if any.
#[inline]
fn next_set(bits: &Mask, from: usize) -> Option<usize> {
    let mut word = from / 64;
    let mut set = bits.get(word)? & (u64::MAX << (from % 64));
    while set == 0 {
        word += 1;
        set = *bits.get(word)?;
    }
    Some(word * 64 + set.trailing_zeros() as usize)
}

/// The row containing `code`, searching from row `from`.
///
/// Requires `offsets[from] <= code < offsets.last()`. Checks row `from` first, then
/// doubles the step until a boundary passes `code`, and binary-searches that last
/// step. Empty rows, with equal offsets, are skipped.
#[inline]
fn find_row<O: Offset>(offsets: &[O], from: usize, code: usize) -> usize {
    if offsets[from + 1].to_usize() > code {
        return from;
    }
    let last = offsets.len() - 1;
    let mut lo = from + 1;
    let mut step = 1;
    loop {
        let hi = (lo + step).min(last);
        if offsets[hi].to_usize() > code {
            return lo + offsets[lo + 1..hi].partition_point(|&x| x.to_usize() <= code);
        }
        lo = hi;
        step *= 2;
    }
}

#[cfg(test)]
mod tests {
    //! Check row resolution against an independent offset-search oracle.
    //!
    //! The same masks are resolved with u32 and u64 row offsets. Fixtures vary
    //! hit density and row length, including empty rows, long gaps, and rows that
    //! span blocks. These tests accept every hit to isolate row resolution.

    use super::{Mask, Resolver, find_row};
    use crate::core::dictionary::Dictionary;
    use crate::core::offset::Offset;
    use crate::core::types::Token;
    use crate::search::substring::ProbeCover;
    use crate::search::substring::probe::scan::tests::byte_dictionary;
    use crate::search::substring::probe::scan::{BLOCK, ScanInput};
    use crate::search::substring::probe::verify::tests::accepting_verifier;

    /// Enough blocks to place a row across an entire intervening block.
    const BLOCKS: usize = 4;
    const CODES: usize = BLOCKS * BLOCK;

    /// Resolve a whole mask in block order, omitting blocks without hits.
    /// The hit verifier accepts every hit, so rows depend only on lookup and deduplication.
    fn resolve<O: Offset>(mask: &[u64], row_offsets: &[O]) -> Vec<usize> {
        let codes = [0 as Token; CODES];
        let cover = ProbeCover::new(Vec::new(), Vec::new());
        let verifier = accepting_verifier();
        let dict = byte_dictionary();
        let input = ScanInput::new(&codes, row_offsets, dict.as_view(), &cover, &verifier);
        let mut out = Vec::new();
        let mut resolver = Resolver::new(input);
        for (block, bits) in mask.chunks_exact(BLOCK / 64).enumerate() {
            let bits: &Mask = bits.try_into().unwrap();
            if bits.iter().any(|&set| set != 0) {
                resolver.rows(bits, block * BLOCK, &mut out);
            }
        }
        out
    }

    /// Independent row-membership oracle for a whole-stream mask.
    /// Binary-search each set bit in the offsets and remove consecutive duplicates;
    /// no exact substring verification is applied in this test helper.
    fn expected_rows<O: Offset>(mask: &[u64], row_offsets: &[O]) -> Vec<usize> {
        let mut rows = Vec::new();
        for (word, &set) in mask.iter().enumerate() {
            let mut set = set;
            while set != 0 {
                let code = word * 64 + set.trailing_zeros() as usize;
                set &= set - 1;
                // Choose the last boundary at or before the code, skipping
                // empty rows that share the same boundary.
                let row = row_offsets.partition_point(|&offset| offset.to_usize() <= code) - 1;
                if rows.last() != Some(&row) {
                    rows.push(row);
                }
            }
        }
        rows
    }

    /// Check basic oracle properties before comparing resolver output.
    fn expected(mask: &[u64], row_offsets: &[u32]) -> Vec<usize> {
        let rows = expected_rows(mask, row_offsets);
        assert!(
            rows.windows(2).all(|pair| pair[0] < pair[1]),
            "the expectation is not ascending and unique: {rows:?}"
        );
        let bits: u32 = mask.iter().map(|set| set.count_ones()).sum();
        assert_eq!(rows.is_empty(), bits == 0, "{bits} hits and no rows");
        rows
    }

    /// Check the resolver with both stored offset widths.
    fn check_resolver(case: &str, mask: &[u64], row_offsets: &[u32]) {
        let expected = expected(mask, row_offsets);
        assert_eq!(resolve(mask, row_offsets), expected, "u32 on {case}");
        // Wide offsets are read directly by the same resolver interface.
        let wide: Vec<u64> = row_offsets
            .iter()
            .map(|&offset| u64::from(offset))
            .collect();
        assert_eq!(resolve(mask, &wide), expected, "u64 on {case}");
    }

    /// Create fixed-size rows, truncating the final row at the stream end.
    fn uniform(codes: usize) -> Vec<u32> {
        let mut offsets: Vec<u32> = (0..CODES as u32).step_by(codes).collect();
        offsets.push(CODES as u32);
        offsets
    }

    /// Create uneven row lengths, including empty rows that lookup must skip.
    fn ragged() -> Vec<u32> {
        let mut offsets = vec![0u32];
        let mut state = 0x2545_F491u32;
        while *offsets.last().unwrap() < CODES as u32 {
            state = state.wrapping_mul(0x0019_660D).wrapping_add(0x3C6E_F35F);
            let len = match state >> 30 {
                0 => 0,
                1 => 1 + state % 8,
                2 => 1 + state % 512,
                _ => 1 + state % 4096,
            };
            offsets.push((offsets.last().unwrap() + len).min(CODES as u32));
        }
        offsets
    }

    /// Allocate a cleared mask for the fixture stream.
    fn empty_mask() -> Vec<u64> {
        vec![0u64; CODES / 64]
    }

    /// Set one hit every `stride` code positions.
    fn strided(stride: usize) -> Vec<u64> {
        let mut mask = empty_mask();
        for code in (0..CODES).step_by(stride) {
            mask[code / 64] |= 1 << (code % 64);
        }
        mask
    }

    /// Row layouts from tiny rows through rows spanning multiple blocks.
    fn layers() -> Vec<(String, Vec<u32>)> {
        let mut layers = vec![
            ("ragged".to_string(), ragged()),
            ("one row".to_string(), vec![0, CODES as u32]),
        ];
        for codes in [1, 3, 8, 64, 65, 1024, BLOCK, BLOCK + 7, 3 * BLOCK] {
            layers.push((format!("{codes} codes per row"), uniform(codes)));
        }
        layers
    }

    #[test]
    fn agree_on_every_density() {
        for (name, row_offsets) in layers() {
            check_resolver(&format!("{name}, empty"), &empty_mask(), &row_offsets);
            check_resolver(
                &format!("{name}, full"),
                &vec![u64::MAX; CODES / 64],
                &row_offsets,
            );
            for stride in [1, 2, 7, 64, 97, 4096, 5000] {
                check_resolver(
                    &format!("{name}, every {stride}th code"),
                    &strided(stride),
                    &row_offsets,
                );
            }
        }
    }

    /// Multiple hits on either side of a block boundary must emit their row once.
    #[test]
    fn agree_on_a_row_hit_in_two_blocks() {
        let row_offsets = vec![0, 100, (BLOCK + 2000) as u32, CODES as u32];
        for pair in [
            [BLOCK - 1, BLOCK],
            [BLOCK - 1, BLOCK + 1999],
            [150, BLOCK + 1999],
        ] {
            let mut mask = empty_mask();
            for code in pair {
                mask[code / 64] |= 1 << (code % 64);
            }
            assert_eq!(
                expected(&mask, &row_offsets),
                vec![1],
                "row 1 not emitted once for {pair:?}"
            );
            check_resolver(&format!("hits at {pair:?}"), &mask, &row_offsets);
        }
    }

    /// The first and last codes must resolve without crossing the offset bounds.
    #[test]
    fn agree_on_the_ends_of_the_stream() {
        for layer in [uniform(1), uniform(64), ragged()] {
            for code in [0, CODES - 1] {
                let mut mask = empty_mask();
                mask[code / 64] |= 1 << (code % 64);
                check_resolver(&format!("hit at {code}"), &mask, &layer);
            }
        }
    }

    /// Sparse hits exercise long row gaps and galloping search boundaries.
    #[test]
    fn agree_on_one_hit_after_many_rows() {
        let row_offsets = uniform(3);
        for code in [1, 2, 3, 4, 8, 4095, 4096, 4097, CODES - 2] {
            let mut mask = empty_mask();
            mask[code / 64] |= 1 << (code % 64);
            let rows = expected(&mask, &row_offsets);
            assert_eq!(rows, vec![code / 3], "lost the hit at {code}");
            check_resolver(&format!("one hit at {code}"), &mask, &row_offsets);
        }
    }

    /// Search respects duplicate boundaries, its starting row, and offsets above u32.
    #[test]
    fn lookup_handles_empty_rows_and_wide_offsets() {
        let cases = [
            vec![0, 0, 0, 1, 1, 2, 2, 2, 10, 10, 100],
            vec![
                0,
                1,
                10,
                u64::from(u32::MAX) - 1,
                u64::from(u32::MAX),
                u64::from(u32::MAX) + 1,
                u64::from(u32::MAX) + 100,
            ],
            (0..4096)
                .map(|i| if i < 2048 { 0 } else { i - 2047 })
                .collect(),
        ];
        for offsets in cases {
            for &boundary in &offsets {
                for code in [boundary.saturating_sub(1), boundary, boundary + 1] {
                    if code >= *offsets.last().unwrap() {
                        continue;
                    }
                    let expected = offsets.partition_point(|&offset| offset <= code) - 1;
                    for from in [0, expected / 2, expected] {
                        assert_eq!(find_row(&offsets, from, code as usize), expected);
                    }
                }
            }
        }
    }
}
