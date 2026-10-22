// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execute a prepared cover and append exact matching rows.
//!
//! The caller supplies the prepared matcher, its mask-packing policy and the scan
//! input: the encoded region plus the prepared cover and hit verifier. The matcher turns
//! blocks of token codes into bit masks; the resolver maps set bits to rows and
//! verifies each candidate occurrence with the hit verifier. A row is appended on its
//! first verified hit, then its later hits are skipped.
//!
//! `dispatch` connects the matcher and packing policy to a concrete implementation.
//! `matcher` owns token membership checks; `resolver` owns row lookup and duplicate
//! suppression. This module defines the instruction sets and matchers; `plan`
//! selects the matcher during preparation and the packing policy for each scan.
//! Empty patterns and covers are handled by `ContainsScan::scan`, before this module.

mod dispatch;
mod matcher;
mod resolver;

use super::ProbeCover;
use super::verify::hit::HitVerifier;
use crate::core::dictionary::CompactDictionaryView;
use crate::core::offset::Offset;
use crate::core::types::Token;
pub(super) use dispatch::{detect_isa, run};
use matcher::Matcher;
pub(super) use matcher::PER_BATCH;
use resolver::Resolver;

/// Instruction-set family used for kernel selection and ranking weights.
/// Production callers use the family returned by `detect_isa`;
/// planning tests can supply any family without executing its kernels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)] // Some variants are only constructed on other build targets.
pub(super) enum Isa {
    Scalar,
    Neon,
    Avx2,
    Avx512Bw,
}

/// Algorithm used to test token membership in the probe cover.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MatcherKind {
    Table,
    EqOr,
    NibbleN8,
}

/// Token codes processed in one matcher block.
pub(super) const BLOCK: usize = 4096;

/// Everything one scan borrows: the encoded region and the prepared query.
#[derive(Clone, Copy)]
pub(super) struct ScanInput<'a, O> {
    codes: &'a [Token],
    row_offsets: &'a [O],
    dict: CompactDictionaryView<'a>,
    cover: &'a ProbeCover,
    verifier: &'a HitVerifier,
}

impl<'a, O> ScanInput<'a, O> {
    /// Borrow the encoded region and the cover and hit verifier prepared for it.
    pub(super) const fn new(
        codes: &'a [Token],
        row_offsets: &'a [O],
        dict: CompactDictionaryView<'a>,
        cover: &'a ProbeCover,
        verifier: &'a HitVerifier,
    ) -> Self {
        Self {
            codes,
            row_offsets,
            dict,
            cover,
            verifier,
        }
    }
}

/// Token codes passed to a matcher in one call.
type Block = [Token; BLOCK];
/// One bit per code in a block; bit `i` describes code `i`.
type Mask = [u64; BLOCK / 64];

/// Match blocks, verify candidate hits, and append each matching row once.
/// Matcher setup, mask storage, and resolver state are reused across blocks.
/// Appended indices are ascending; existing output contents are preserved.
fn both_stages<M: Matcher, O: Offset>(input: ScanInput<'_, O>, out: &mut Vec<usize>) {
    let matcher = M::new(input.cover);
    let mut resolver = Resolver::new(input);
    let mut bits = [0u64; BLOCK / 64];
    blocks(
        input.codes,
        input.row_offsets[0].to_usize(),
        &mut |block, at, valid| {
            // A false result guarantees no hits; true may still mean an empty mask.
            if matcher.check(block, &mut bits) {
                clear_from(&mut bits, valid);
                resolver.rows(&bits, at, out);
            }
        },
    );
}

/// Visit the codes from `first` in full blocks, and copy the final partial block
/// into zero padding. The callback receives the block, its absolute start, and its
/// valid code count. It must discard mask bits beyond that count before resolving rows.
fn blocks(codes: &[Token], first: usize, step: &mut dyn FnMut(&Block, usize, usize)) {
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

    /// Every code from `first` is visited once at its absolute index, and a partial
    /// tail is zero-padded to a full block.
    #[test]
    fn blocks_visit_every_code_once() {
        for len in [0, 1, 63, 64, BLOCK - 1, BLOCK, BLOCK + 1, 2 * BLOCK + 100] {
            let codes: Vec<Token> = (1..=len).map(|code| code as Token).collect();
            for first in [0, len / 2] {
                let mut seen = Vec::new();
                blocks(&codes, first, &mut |block, at, valid| {
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
