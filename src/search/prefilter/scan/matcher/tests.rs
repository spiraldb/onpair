// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Every matcher against `ProbeCover::contains`.

use super::{Block, Mask, Matcher, Table};
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use super::{EqOr, Range};
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use super::{NibbleN8, PER_BATCH};
use crate::core::types::{Token, TokenRange};
use crate::search::prefilter::ProbeCover;
use crate::search::prefilter::scan::policy::{Match, Shape, takes};
use crate::search::prefilter::scan::{BLOCK, Check, both_stages, resolver};

fn mask<M: Matcher>(cover: &ProbeCover, codes: &Block) -> Mask {
    let mut bits = [0u64; BLOCK / 64];
    // A cleared flag promises an empty mask. The other way round is allowed:
    // a kernel that packs regardless never asked the question.
    let any_hit = M::new(cover).check(codes, &mut bits);
    assert!(
        any_hit || bits.iter().all(|&set| set == 0),
        "a set bit under a cleared empty-block flag"
    );
    bits
}

/// The mask the cover defines: bit `i` iff the probe matches at `codes[i]`.
/// One code at a time, straight off `ProbeCover::contains`, so a kernel is checked
/// against the contract and never against a sibling.
fn expected(cover: &ProbeCover, codes: &Block) -> Mask {
    let mut bits = [0u64; BLOCK / 64];
    for (at, &code) in codes.iter().enumerate() {
        if cover.contains(code) {
            bits[at / 64] |= 1 << (at % 64);
        }
    }
    bits
}

/// Bit for bit with the definition, where the policy hands the kernel the
/// cover at all.
fn agrees<M: Matcher>(kind: Match, name: &str, cover: &ProbeCover, codes: &Block) {
    if !takes(kind, Shape::of(cover)) {
        return;
    }
    assert_eq!(
        mask::<M>(cover, codes),
        expected(cover, codes),
        "{name} differs from the cover it was built from"
    );
}

/// A block whose codes are all distinct: 37 is invertible modulo 50021, so
/// no value repeats over a block and a needle has exactly one hit.
fn block() -> Block {
    let mut codes = [0 as Token; BLOCK];
    for (at, code) in codes.iter_mut().enumerate() {
        *code = (at * 37 % 50021) as Token;
    }
    codes
}

/// Every matcher on one cover and one block.
fn every_matcher(cover: &ProbeCover, codes: &Block) {
    agrees::<Table>(Match::Table, "table", cover, codes);
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    {
        agrees::<EqOr<false>>(Match::EqOr, "eq_or", cover, codes);
        agrees::<EqOr<true>>(Match::EqOr, "eq_or_skip_empty", cover, codes);
        agrees::<Range<false>>(Match::Range, "range", cover, codes);
        agrees::<Range<true>>(Match::Range, "range_skip_empty", cover, codes);
        // At the batch count the dispatch compiles for K.
        let n8k = Match::NibbleN8K;
        match cover.points().len().div_ceil(PER_BATCH) {
            1 => {
                agrees::<NibbleN8<1, false>>(n8k, "nibble_n8", cover, codes);
                agrees::<NibbleN8<1, true>>(n8k, "nibble_n8_skip_empty", cover, codes);
            }
            2 => {
                agrees::<NibbleN8<2, false>>(n8k, "nibble_n16", cover, codes);
                agrees::<NibbleN8<2, true>>(n8k, "nibble_n16_skip_empty", cover, codes);
            }
            3 => {
                agrees::<NibbleN8<3, false>>(n8k, "nibble_n24", cover, codes);
                agrees::<NibbleN8<3, true>>(n8k, "nibble_n24_skip_empty", cover, codes);
            }
            _ => {}
        }
    }
}

/// One cover as it comes, then with one range beside it and with two, so
/// a kernel that stops after the first is caught.
fn every_matcher_ranged(cover: ProbeCover, lo: Token, hi: Token, codes: &Block) {
    every_matcher(&cover, codes);
    every_matcher(
        &ProbeCover::new(
            cover.points().to_vec(),
            vec![TokenRange {
                begin: lo,
                last: hi,
            }],
        ),
        codes,
    );
    let two = [
        TokenRange {
            begin: lo,
            last: hi,
        },
        TokenRange {
            begin: 60000,
            last: 60100,
        },
    ];
    every_matcher(
        &ProbeCover::new(cover.points().to_vec(), two.to_vec()),
        codes,
    );
}

#[test]
fn agree_on_needles_the_block_holds() {
    let codes = block();
    for count in [1, 2, 3, 8, 9, 12, 16, 17, 24, 64] {
        let needles: Vec<Token> = (0..count).map(|k| codes[k * 37]).collect();
        let cover = ProbeCover::new(needles.to_vec(), vec![]);
        assert_eq!(
            expected(&cover, &codes)
                .iter()
                .map(|word| word.count_ones())
                .sum::<u32>(),
            count as u32,
            "wrong hits at K = {count}"
        );
        every_matcher_ranged(cover, 1000, 2000, &codes);
    }
}

#[test]
fn agree_on_needles_the_block_does_not_hold() {
    let mut codes = block();
    codes.fill(7);
    let needles: Vec<Token> = (0..4).map(|k| 60000 + k).collect();
    every_matcher(&ProbeCover::new(needles.to_vec(), vec![]), &codes);
}

/// A hit at the last position of a block, which for L = 1 is the last code
/// the mask covers rather than anything the lookahead reaches.
#[test]
fn agree_on_a_hit_at_the_seam() {
    let mut codes = [7 as Token; BLOCK];
    codes[BLOCK - 1] = 60000;
    let cover = ProbeCover::new(vec![60000], vec![]);
    assert_eq!(
        expected(&cover, &codes)[BLOCK / 64 - 1],
        1 << 63,
        "the definition missed the seam hit"
    );
    every_matcher(&cover, &codes);
}

/// Narrowing keeps the low byte of each lane, so two codes that share one
/// must not be confused for each other.
#[test]
fn agree_on_codes_sharing_a_low_byte() {
    let mut codes = [0x0102 as Token; BLOCK];
    codes[70] = 0x0202;
    let cover = ProbeCover::new(vec![0x0202], vec![]);
    assert_eq!(
        expected(&cover, &codes)[1],
        1 << 6,
        "the definition missed the planted code"
    );
    // A range whose codes share the low byte of every code in the block but
    // are none of them: narrowing its compare has to keep the high byte.
    every_matcher_ranged(cover, 0x0302, 0x0402, &codes);
}

/// A range at both ends of the code space, and over a run the block holds
/// nothing of: it holds nothing at or above 50021.
#[test]
fn agree_on_ranges_of_wide_codes() {
    let codes = block();
    for (lo, hi) in [
        (0 as Token, Token::MAX),
        (0, 0),
        (37, 37),
        (1000, 2000),
        (50000, 50020),
        (50021, Token::MAX),
    ] {
        let ranges = [TokenRange {
            begin: lo,
            last: hi,
        }];
        every_matcher(&ProbeCover::new(vec![], ranges.to_vec()), &codes);
        every_matcher(
            &ProbeCover::new(vec![0 as Token, 30000, 50020], ranges.to_vec()),
            &codes,
        );
    }
    let all = expected(
        &ProbeCover::new(
            vec![],
            vec![TokenRange {
                begin: 0,
                last: Token::MAX,
            }],
        ),
        &codes,
    );
    assert!(
        all.iter().all(|&word| word == !0),
        "a code outside the width"
    );
    let none = expected(
        &ProbeCover::new(
            vec![],
            vec![TokenRange {
                begin: 50021,
                last: Token::MAX,
            }],
        ),
        &codes,
    );
    assert!(
        none.iter().all(|&word| word == 0),
        "a code at or above 50021"
    );
}

/// The codes just outside either end of what the cover admits, with a
/// range setting one end.
#[test]
fn table_rejects_the_codes_around_its_own() {
    let mut codes = [0 as Token; BLOCK];
    for (at, code) in codes.iter_mut().enumerate() {
        *code = [999, 1000, 3000, 3001, 0, Token::MAX][at % 6];
    }
    let ranges = [TokenRange {
        begin: 2500 as Token,
        last: 3000,
    }];
    every_matcher(
        &ProbeCover::new(vec![1000 as Token], ranges.to_vec()),
        &codes,
    );
    every_matcher(&ProbeCover::new(vec![1000 as Token, 3000], vec![]), &codes);
    let bits = mask::<Table>(&ProbeCover::new(vec![1000 as Token, 3000], vec![]), &codes);
    assert_eq!(bits[0] & 0x3f, 0b000110);
}

/// One planted code, one hit and no more, through the kernel every target
/// compiles.
#[test]
fn the_table_masks_one_code() {
    let codes = block();
    let bits = mask::<Table>(&ProbeCover::new(codes[100..101].to_vec(), vec![]), &codes);
    assert_eq!(bits[100 / 64] >> (100 % 64) & 1, 1, "missed the code");
    assert_eq!(
        bits.iter().map(|word| word.count_ones()).sum::<u32>(),
        1,
        "extra hits"
    );
}

/// And the same through the block driver, whose tail is zero-padded and
/// trimmed to the codes the stream holds.
#[test]
fn the_driver_scans_every_block() {
    // Two and a bit blocks, so both the whole-block and the padded tail path
    // run, with the planted hit inside the tail.
    let codes: Vec<Token> = (0..2 * BLOCK + 100)
        .map(|at| (at * 37 % 50021) as Token)
        .collect();
    let row_offsets: Vec<u32> = (0..=codes.len() as u32).step_by(64).collect();
    let at = 2 * BLOCK + 40;
    let needles = codes[at..at + 1].to_vec();
    let mut found = Vec::new();
    both_stages::<Table, resolver::LinearSeek<'_, u32>>(
        &ProbeCover::new(needles.to_vec(), vec![]),
        &codes,
        &row_offsets,
        Check::Superset,
        &mut found,
    );
    assert_eq!(found, vec![at / 64]);
}

/// The tail is padded with zeros, so a zero the padding carries must not
/// reach a resolver, whatever the stream length leaves in the last block.
#[test]
fn padding_makes_no_candidate() {
    for length in [1, 2, 3, 5, BLOCK, BLOCK + 1, BLOCK + 2, BLOCK + 3] {
        // No zero in the stream, so every zero a token matches is padding.
        let codes: Vec<Token> = (0..length).map(|at| (at % 255 + 1) as Token).collect();
        let row_offsets = vec![0, length as u32];
        let mut found = Vec::new();
        both_stages::<Table, resolver::LinearSeek<'_, u32>>(
            &ProbeCover::new(vec![0 as Token], vec![]),
            &codes,
            &row_offsets,
            Check::Superset,
            &mut found,
        );
        assert!(found.is_empty(), "over {length} codes: {found:?}");
    }
}
