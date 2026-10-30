// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scan execution, which finds the hits block by block and appends the verified rows.
//!
//! The codes are processed in blocks of `BLOCK`. For each block, the matcher writes
//! a bitmask with one bit per code, set if the code is in the cover. The resolver
//! maps each set bit to its row and runs the hit verifier. A row is appended on its
//! first verified hit, and its remaining hits are skipped.
//!
//! `matcher` holds the membership kernels, `resolver` the row lookup, and `dispatch`
//! builds the kernel chosen by `plan`, once per query.

mod dispatch;
mod matcher;
mod resolver;

use super::ProbeCover;
use super::verify::HitVerifier;
use crate::core::dictionary::CompactDictionaryView;
use crate::core::offset::Offset;
use crate::core::types::Token;
pub(super) use dispatch::{detect_isa, prepare};
pub(super) use matcher::{Matcher, PER_BATCH};
use resolver::Resolver;

/// The vector instruction set the kernels and cost weights are chosen for.
/// Scans use `detect_isa`; planning tests may pass any value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)] // Some variants are only constructed on other build targets.
pub(super) enum Isa {
    /// No vector kernels: only the table matcher is available.
    Scalar,
    Neon,
    Avx2,
    Avx512Bw,
}

/// The kernel that tests codes for membership in the cover.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MatcherKind {
    /// Scalar lookup table; supports any cover and any target.
    Table,
    /// One vector equality per point, plus range checks.
    EqOr,
    /// Nibble shuffle tables, eight points per batch, plus range checks.
    Nibble,
}

/// Number of codes per matcher block.
pub(super) const BLOCK: usize = 4096;

/// Everything one scan borrows for row resolution: the encoded region and the
/// prepared hit verifier.
#[derive(Clone, Copy)]
pub(super) struct ScanInput<'a, O> {
    pub(super) codes: &'a [Token],
    pub(super) row_offsets: &'a [O],
    pub(super) dict: CompactDictionaryView<'a>,
    pub(super) verifier: &'a HitVerifier,
}

/// Token codes passed to a matcher in one call.
type Block = [Token; BLOCK];
/// One bit per code in a block; bit `i` describes code `i`.
type Mask = [u64; BLOCK / 64];

/// Runs `matcher` over every block and appends each verified row once, in
/// ascending order. `skip_empty` is passed to every `check`.
pub(super) fn run<O: Offset>(
    matcher: &dyn Matcher,
    skip_empty: bool,
    input: ScanInput<'_, O>,
    out: &mut Vec<usize>,
) {
    let mut resolver = Resolver::new(input);
    let mut bits = [0u64; BLOCK / 64];
    blocks(
        input.codes,
        input.row_offsets[0].to_usize(),
        |block, at, valid| {
            // A false result guarantees no hits; true may still mean an empty mask.
            if matcher.check(block, &mut bits, skip_empty) {
                clear_from(&mut bits, valid);
                resolver.rows(&bits, at, out);
            }
        },
    );
}

/// Calls `step` on each block of codes from `first`, with the block's absolute start
/// and its number of valid codes. The last partial block is zero-padded, so `step`
/// must discard mask bits past the valid count.
fn blocks(codes: &[Token], first: usize, mut step: impl FnMut(&Block, usize, usize)) {
    let (whole, rest) = codes[first..].as_chunks::<BLOCK>();
    let mut at = first;
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

/// Clears the mask bits of padding codes, at index `valid` and beyond.
fn clear_from(bits: &mut Mask, valid: usize) {
    if valid < BLOCK {
        bits[valid / 64] &= (1u64 << (valid % 64)) - 1;
        bits[valid / 64 + 1..].fill(0);
    }
}

#[cfg(test)]
mod tests {
    use super::dispatch::supports;
    use super::*;
    use crate::core::dictionary::{CompactDictionary, Dictionary, pad_raw};
    use crate::core::types::TokenRange;
    use crate::search::substring::probe::plan::is_eligible;
    use crate::search::substring::probe::verify::tests::accepting_verifier;

    /// A dictionary of the 256 single-byte tokens, for scans whose verifier never
    /// reads it.
    pub(super) fn byte_dictionary() -> CompactDictionary {
        let mut bytes: Vec<u8> = (0..=255).collect();
        let offsets: Vec<u32> = (0..=256).collect();
        pad_raw(&mut bytes, &offsets);
        CompactDictionary::from_raw(bytes, offsets)
    }

    /// Runs every kernel eligible on every instruction set this CPU supports, in both
    /// packing modes, with a verifier that accepts every hit, and checks that each
    /// appends exactly `expected`.
    fn check_every_kernel<O: Offset>(
        cover: &ProbeCover,
        codes: &[Token],
        row_offsets: &[O],
        expected: &[usize],
    ) {
        let dict = byte_dictionary();
        let verifier = accepting_verifier();
        let input = ScanInput {
            codes,
            row_offsets,
            dict: dict.as_view(),
            verifier: &verifier,
        };
        let isas = [Isa::Scalar, Isa::Neon, Isa::Avx2, Isa::Avx512Bw];
        for isa in isas.into_iter().filter(|&isa| supports(isa)) {
            for kind in [MatcherKind::Table, MatcherKind::EqOr, MatcherKind::Nibble] {
                if !is_eligible(isa, kind, cover) {
                    continue;
                }
                let matcher = prepare(isa, kind, cover);
                for skip_empty in [false, true] {
                    let mut out = Vec::new();
                    run(&*matcher, skip_empty, input, &mut out);
                    assert_eq!(out, expected, "{isa:?}, {kind:?}, skip_empty={skip_empty}");
                }
            }
        }
    }

    /// Rows that start partway into the codes, as in a sliced view, ignore the hits
    /// before them, while a hit in the zero-padded last block is still found. Checked
    /// for every kernel, at both offset widths.
    #[test]
    fn scan_ignores_codes_before_the_first_row() {
        // Distinct codes, so each planted point occurs exactly once.
        let codes: Vec<Token> = (0..2 * BLOCK + 100)
            .map(|at| (at * 37 % 50021) as Token)
            .collect();
        let (early, late) = (50, 2 * BLOCK + 40);
        let cover = ProbeCover::new(vec![codes[early], codes[late]], Vec::new());
        // Rows of 64 codes from code 0, or from code 100 on.
        for (first, expected) in [
            (0, vec![early / 64, late / 64]),
            (100, vec![(late - 100) / 64]),
        ] {
            let offsets: Vec<u32> = (first as u32..=codes.len() as u32).step_by(64).collect();
            let codes = &codes[..*offsets.last().unwrap() as usize];
            check_every_kernel(&cover, codes, &offsets, &expected);
            let wide: Vec<u64> = offsets.iter().map(|&offset| u64::from(offset)).collect();
            check_every_kernel(&cover, codes, &wide, &expected);
        }
    }

    /// Zero padding never becomes a hit, even when the cover contains token 0.
    #[test]
    fn padding_never_becomes_a_hit() {
        let low = TokenRange::new(0, 3);
        for cover in [
            ProbeCover::new(vec![0], Vec::new()),
            ProbeCover::new(Vec::new(), vec![low]),
        ] {
            for len in [1, 2, 63, 64, 65, BLOCK - 1, BLOCK, BLOCK + 1] {
                // Codes start at 4, so only padding could match the cover.
                let codes: Vec<Token> = (0..len).map(|at| (at % 1000 + 4) as Token).collect();
                check_every_kernel(&cover, &codes, &[0u32, len as u32], &[]);
            }
        }
    }

    /// Every code from `first` is visited once at its absolute index, and a partial
    /// tail is zero-padded to a full block.
    #[test]
    fn blocks_visit_every_code_once() {
        for len in [0, 1, 63, 64, BLOCK - 1, BLOCK, BLOCK + 1, 2 * BLOCK + 100] {
            let codes: Vec<Token> = (1..=len).map(|code| code as Token).collect();
            for first in [0, len / 2] {
                let mut seen = Vec::new();
                blocks(&codes, first, |block, at, valid| {
                    assert!(block[valid..].iter().all(|&code| code == 0), "{len} codes");
                    seen.extend(
                        (at..)
                            .zip(&block[..valid])
                            .map(|(index, &code)| (index, code)),
                    );
                });
                let expected: Vec<_> = codes.iter().copied().enumerate().skip(first).collect();
                assert_eq!(seen, expected, "{len} codes from {first}");
            }
        }
    }

    /// Bits at or past the valid count are cleared, so tail padding never reaches
    /// row resolution; earlier bits are kept.
    #[test]
    fn clear_from_drops_the_padding() {
        for valid in [1, 63, 64, 65, BLOCK - 1, BLOCK] {
            let mut bits = [u64::MAX; BLOCK / 64];
            clear_from(&mut bits, valid);
            for code in 0..BLOCK {
                let set = bits[code / 64] >> (code % 64) & 1 == 1;
                assert_eq!(set, code < valid, "valid {valid}, code {code}");
            }
        }
    }
}
