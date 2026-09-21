// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Check matcher masks against scalar probe-cover membership.
//!
//! Each eligible kernel receives the same cover and code block. The oracle
//! calls `ProbeCover::contains` per code, independently of vector packing.
//! Cases exercise point counts, ranges, code-width boundaries, and block tails.
//! The block-driver checks also ensure padding cannot create candidate rows.

use super::{Block, Mask, Matcher, Table};
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use super::{EqOr, Range};
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use super::{NibbleN8, PER_BATCH};
use crate::core::types::{Token, TokenRange};
use crate::search::substring::ProbeCover;
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use crate::search::substring::plan::Isa;
use crate::search::substring::plan::MatcherKind;
use crate::search::substring::scan::{BLOCK, Check, both_stages};

/// Run one matcher and verify that a false return guarantees an empty mask.
fn mask<M: Matcher>(cover: &ProbeCover, codes: &Block) -> Mask {
    let mut bits = [0u64; BLOCK / 64];
    // A matcher may return true for an empty mask when it always packs.
    let any_hit = M::new(cover).check(codes, &mut bits);
    assert!(
        any_hit || bits.iter().all(|&set| set == 0),
        "a set bit under a cleared empty-block flag"
    );
    bits
}

/// Build the expected bit mask directly from scalar cover membership.
fn expected(cover: &ProbeCover, codes: &Block) -> Mask {
    let mut bits = [0u64; BLOCK / 64];
    for (at, &code) in codes.iter().enumerate() {
        if cover.contains(code) {
            bits[at / 64] |= 1 << (at % 64);
        }
    }
    bits
}

/// Compare an eligible matcher with the independent mask oracle.
fn agrees<M: Matcher>(kind: MatcherKind, name: &str, cover: &ProbeCover, codes: &Block) {
    if !supports_matcher(detect_target_caps(), kind, cover) {
        return;
    }
    assert_eq!(
        mask::<M>(cover, codes),
        expected(cover, codes),
        "{name} differs from the cover it was built from"
    );
}

/// Create distinct codes so a selected point has exactly one hit per block.
fn block() -> Block {
    let mut codes = [0 as Token; BLOCK];
    for (at, code) in codes.iter_mut().enumerate() {
        *code = (at * 37 % 50021) as Token;
    }
    codes
}

/// Check all kernels and packing variants eligible on this CPU.
fn every_matcher(cover: &ProbeCover, codes: &Block) {
    agrees::<Table>(MatcherKind::Table, "table", cover, codes);
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    {
        agrees::<EqOr<false>>(MatcherKind::EqOr, "eq_or", cover, codes);
        agrees::<EqOr<true>>(MatcherKind::EqOr, "eq_or_skip_empty", cover, codes);
        agrees::<Range<false>>(MatcherKind::Range, "range", cover, codes);
        agrees::<Range<true>>(MatcherKind::Range, "range_skip_empty", cover, codes);
        // Use the same point-to-batch mapping as production dispatch.
        let n8k = MatcherKind::NibbleN8K;
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

/// Check the point set alone, then with one and two ranges.
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
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn eq_or_without_points_checks_ranges() {
    if detect_target_caps().isa == Isa::Scalar {
        return;
    }
    let codes = block();
    for ranges in [
        vec![],
        vec![
            TokenRange {
                begin: 0,
                last: 100,
            },
            TokenRange {
                begin: 1000,
                last: 2000,
            },
        ],
    ] {
        let cover = ProbeCover::new(vec![], ranges);
        let want = expected(&cover, &codes);
        let mut bits = [u64::MAX; BLOCK / 64];
        EqOr::<false>::new(&cover).check(&codes, &mut bits);
        assert_eq!(bits, want);
        bits.fill(u64::MAX);
        let any_hit = EqOr::<true>::new(&cover).check(&codes, &mut bits);
        assert_eq!(bits, want);
        assert!(any_hit || bits.iter().all(|&word| word == 0));
    }
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

/// A final-code hit must set the last bit in the block mask.
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

/// Membership must use the entire u16 code before comparison lanes are narrowed.
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
    // No block code lies in this range, even though both endpoints share
    // the block codes' low byte.
    every_matcher_ranged(cover, 0x0302, 0x0402, &codes);
}

/// Test full-domain, singleton, populated, and absent ranges.
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

/// Values around 0x8000 must compare as unsigned codes, including on AVX2.
#[test]
fn agree_on_the_sign_boundary() {
    let mut codes = [0 as Token; BLOCK];
    for (at, code) in codes.iter_mut().enumerate() {
        *code = [0x7ffe, 0x7fff, 0x8000, 0x8001, 0x8002, 7][at % 6];
    }
    for needles in [
        vec![0x7fff],
        vec![0x8000],
        vec![0x8001],
        vec![0x7fff, 0x8000, 0x8001],
    ] {
        every_matcher(&ProbeCover::new(needles, vec![]), &codes);
    }
    for (lo, hi) in [
        (0x7ffe, 0x8001),
        (0x7fff, 0x7fff),
        (0x8000, 0x8000),
        (0, 0x7fff),
        (0x8000, Token::MAX),
    ] {
        let range = TokenRange {
            begin: lo,
            last: hi,
        };
        every_matcher(&ProbeCover::new(vec![], vec![range]), &codes);
        every_matcher(&ProbeCover::new(vec![7], vec![range]), &codes);
    }
    let across = expected(
        &ProbeCover::new(
            vec![],
            vec![TokenRange {
                begin: 0x7ffe,
                last: 0x8001,
            }],
        ),
        &codes,
    );
    assert_eq!(
        across[0] & 0x3f,
        0b001111,
        "the definition misplaced 0x8000"
    );
}

/// Ranges ending at the maximum token ID must include it without wrapping.
#[test]
fn agree_at_the_top_of_the_code_space() {
    let mut codes = [0 as Token; BLOCK];
    for (at, code) in codes.iter_mut().enumerate() {
        *code = [65527, 65531, 65532, 65533, 65534, Token::MAX, 7][at % 7];
    }
    for needles in [vec![65527], vec![Token::MAX], vec![65527, Token::MAX]] {
        every_matcher(&ProbeCover::new(needles, vec![]), &codes);
    }
    for (lo, hi) in [
        (65532, Token::MAX),
        (Token::MAX, Token::MAX),
        (65528, 65531),
    ] {
        let range = TokenRange {
            begin: lo,
            last: hi,
        };
        every_matcher(&ProbeCover::new(vec![], vec![range]), &codes);
        every_matcher(&ProbeCover::new(vec![65527], vec![range]), &codes);
    }
}

/// Probe endpoints are inclusive; neighboring codes outside them are rejected.
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

/// The portable table kernel must mark exactly the planted code.
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

/// Scan full blocks and a partial tail, using both row-offset widths.
#[test]
fn the_driver_scans_every_block() {
    // Two and a bit blocks, so both the whole-block and the padded tail path
    // run, with the planted hit inside the tail.
    let codes: Vec<Token> = (0..2 * BLOCK + 100)
        .map(|at| (at * 37 % 50021) as Token)
        .collect();
    let row_offsets: Vec<u32> = (0..=codes.len() as u32).step_by(64).collect();
    let at = 2 * BLOCK + 40;
    let cover = ProbeCover::new(codes[at..at + 1].to_vec(), vec![]);
    let mut found = Vec::new();
    both_stages::<Table, u32>(&cover, &codes, &row_offsets, Check::Superset, &mut found);
    assert_eq!(found, vec![at / 64]);

    // The same layer at the wide offset width.
    let wide: Vec<u64> = row_offsets.iter().map(|&o| u64::from(o)).collect();
    found.clear();
    both_stages::<Table, u64>(&cover, &codes, &wide, Check::Superset, &mut found);
    assert_eq!(found, vec![at / 64]);
}

/// Zero padding in a partial block must not reach row resolution as a hit.
#[test]
fn padding_makes_no_candidate() {
    for length in [1, 2, 3, 5, BLOCK, BLOCK + 1, BLOCK + 2, BLOCK + 3] {
        // No zero in the stream, so every zero a token matches is padding.
        let codes: Vec<Token> = (0..length).map(|at| (at % 255 + 1) as Token).collect();
        let row_offsets = vec![0, length as u32];
        let mut found = Vec::new();
        both_stages::<Table, u32>(
            &ProbeCover::new(vec![0 as Token], vec![]),
            &codes,
            &row_offsets,
            Check::Superset,
            &mut found,
        );
        assert!(found.is_empty(), "over {length} codes: {found:?}");
    }
}

use crate::search::substring::plan::supports_matcher;
use crate::search::substring::scan::detect_target_caps;
