// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! x86-64 scan execution and target-feature boundaries.

use super::ScanInput;
use super::policy::{FixedShape, with_sse2_fixed_shapes, with_x86_fixed_shapes};
use crate::core::offset::Offset;
use crate::core::types::Token;
use crate::search::prefilter::cover::ProbeCover;

mod avx2;
mod avx512;
mod sse2;

#[cfg(test)]
pub(in crate::search::prefilter) use avx2::scan_avx2;
#[cfg(test)]
pub(in crate::search::prefilter) use avx512::scan_avx512;
#[cfg(test)]
pub(in crate::search::prefilter) use sse2::scan_sse2;

#[inline]
pub(super) fn execute_sse2<O: Offset>(
    shape: Option<FixedShape>,
    input: ScanInput<'_, O>,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    let ScanInput {
        codes,
        row_offsets,
        cover,
    } = input;
    match shape {
        Some(shape) => {
            execute_sse2_fixed(shape, codes, row_offsets, cover, sparse_row_mapping, out)
        }
        None => sse2::scan_sse2_generic(codes, row_offsets, cover, sparse_row_mapping, out),
    }
}

fn execute_sse2_fixed<O: Offset>(
    shape: FixedShape,
    codes: &[Token],
    row_offsets: &[O],
    cover: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    macro_rules! call {
        ($points:literal, $ranges:literal) => {
            sse2::scan_sse2_fixed::<O, $points, $ranges>(
                codes,
                row_offsets,
                cover,
                sparse_row_mapping,
                out,
            )
        };
    }
    macro_rules! dispatch {
        ($(($points:literal, $ranges:literal),)+) => {
            match (shape.points, shape.ranges) {
                $(($points, $ranges) => call!($points, $ranges),)+
                _ => unreachable!("invalid fixed x86 shape"),
            }
        };
    }
    unsafe { with_sse2_fixed_shapes!(dispatch) }
}

pub(super) unsafe fn execute_avx2<O: Offset>(
    shape: Option<FixedShape>,
    input: ScanInput<'_, O>,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    let ScanInput {
        codes,
        row_offsets,
        cover,
    } = input;
    unsafe {
        match shape {
            Some(shape) => {
                execute_avx2_fixed::<O>(shape, codes, row_offsets, cover, sparse_row_mapping, out)
            }
            None => avx2::scan_avx2_generic(codes, row_offsets, cover, sparse_row_mapping, out),
        }
    }
}

unsafe fn execute_avx2_fixed<O: Offset>(
    shape: FixedShape,
    codes: &[Token],
    row_offsets: &[O],
    cover: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    macro_rules! call {
        ($points:literal, $ranges:literal) => {
            avx2::scan_avx2_fixed::<O, $points, $ranges, 1>(
                codes,
                row_offsets,
                cover,
                sparse_row_mapping,
                out,
            )
        };
    }
    macro_rules! dispatch {
        ($(($points:literal, $ranges:literal),)+) => {
            match (shape.points, shape.ranges) {
                $(($points, $ranges) => call!($points, $ranges),)+
                _ => unreachable!("invalid fixed x86 shape"),
            }
        };
    }
    unsafe { with_x86_fixed_shapes!(dispatch) }
}

pub(super) unsafe fn execute_avx512<O: Offset>(
    shape: Option<FixedShape>,
    input: ScanInput<'_, O>,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    let ScanInput {
        codes,
        row_offsets,
        cover,
    } = input;
    unsafe {
        match shape {
            Some(shape) => {
                execute_avx512_fixed(shape, codes, row_offsets, cover, sparse_row_mapping, out)
            }
            None => avx512::scan_avx512(codes, row_offsets, cover, sparse_row_mapping, out),
        }
    }
}

unsafe fn execute_avx512_fixed<O: Offset>(
    shape: FixedShape,
    codes: &[Token],
    row_offsets: &[O],
    cover: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    macro_rules! call {
        ($points:literal, $ranges:literal) => {
            avx512::scan_avx512_fixed::<O, $points, $ranges>(
                codes,
                row_offsets,
                cover,
                sparse_row_mapping,
                out,
            )
        };
    }
    macro_rules! dispatch {
        ($(($points:literal, $ranges:literal),)+) => {
            match (shape.points, shape.ranges) {
                $(($points, $ranges) => call!($points, $ranges),)+
                _ => unreachable!("invalid fixed x86 shape"),
            }
        };
    }
    unsafe { with_x86_fixed_shapes!(dispatch) }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EDGE_POINTS: [Token; 17] = [
        7, 17, 31, 63, 127, 255, 511, 1_023, 2_047, 4_095, 8_191, 16_383, 24_575, 32_760, 32_772,
        49_151, 65_527,
    ];
    const EDGE_RANGES: [(Token, Token); 3] = [
        (Token::MIN, 3),
        (0x7ffe, 0x8001),
        (Token::MAX - 3, Token::MAX),
    ];

    fn edge_cover(points: usize, ranges: usize) -> ProbeCover {
        assert!(points <= EDGE_POINTS.len());
        assert!(ranges <= EDGE_RANGES.len());
        let mut table = vec![false; usize::from(Token::MAX) + 1];
        for &point in &EDGE_POINTS[..points] {
            table[usize::from(point)] = true;
        }
        for &(begin, last) in &EDGE_RANGES[..ranges] {
            table[usize::from(begin)..=usize::from(last)].fill(true);
        }
        let cover = ProbeCover::from_membership(table);
        assert_eq!((cover.points.len(), cover.ranges.len()), (points, ranges));
        cover
    }

    fn cover<const POINTS: usize, const RANGES: usize>() -> ProbeCover {
        edge_cover(POINTS, RANGES)
    }

    fn input<O: Offset>() -> (Vec<Token>, Vec<O>) {
        const VALUES: [Token; 13] = [
            0, 1, 6, 7, 8, 32_767, 32_768, 32_769, 65_531, 65_532, 65_533, 65_534, 65_535,
        ];
        // Three complete AVX-512 superblocks, one complete AVX2 block, and
        // non-multiple tails for every vector width.
        let codes = (0..1_603)
            .map(|index| VALUES[(index * 7 + index / 11) % VALUES.len()])
            .collect::<Vec<_>>();
        let mut offsets = vec![O::from_usize(0)];
        let mut offset = 0;
        while offset < codes.len() {
            if offsets.len().is_multiple_of(17) {
                offsets.push(O::from_usize(offset));
            }
            let width = (offsets.len() * 11 % 23) + 1;
            offset = (offset + width).min(codes.len());
            offsets.push(O::from_usize(offset));
        }
        (codes, offsets)
    }

    fn assert_cover<O: Offset, const POINTS: usize, const RANGES: usize>(
        codes: &[Token],
        row_offsets: &[O],
        cover: &ProbeCover,
    ) {
        let mut expected = Vec::new();
        super::super::scan_scalar(codes, row_offsets, cover, &mut expected);

        for sparse_row_mapping in [false, true] {
            let mut sse2 = Vec::new();
            unsafe {
                sse2::scan_sse2_fixed::<O, POINTS, RANGES>(
                    codes,
                    row_offsets,
                    cover,
                    sparse_row_mapping,
                    &mut sse2,
                )
            };
            assert_eq!(sse2, expected);

            if std::is_x86_feature_detected!("avx2") {
                for blocks in [1, 8] {
                    let mut avx2 = Vec::new();
                    unsafe {
                        match blocks {
                            1 => avx2::scan_avx2_fixed::<O, POINTS, RANGES, 1>(
                                codes,
                                row_offsets,
                                cover,
                                sparse_row_mapping,
                                &mut avx2,
                            ),
                            8 => avx2::scan_avx2_fixed::<O, POINTS, RANGES, 8>(
                                codes,
                                row_offsets,
                                cover,
                                sparse_row_mapping,
                                &mut avx2,
                            ),
                            _ => unreachable!(),
                        }
                    }
                    assert_eq!(avx2, expected);
                }
            }

            if std::is_x86_feature_detected!("avx512bw") {
                let mut avx512 = Vec::new();
                unsafe {
                    avx512::scan_avx512_fixed::<O, POINTS, RANGES>(
                        codes,
                        row_offsets,
                        cover,
                        sparse_row_mapping,
                        &mut avx512,
                    )
                };
                assert_eq!(avx512, expected);
            }
        }
    }

    fn assert_shape<O: Offset, const POINTS: usize, const RANGES: usize>() {
        let (codes, row_offsets) = input::<O>();
        assert_cover::<O, POINTS, RANGES>(&codes, &row_offsets, &cover::<POINTS, RANGES>());
    }

    fn assert_sse2_shape<O: Offset, const POINTS: usize, const RANGES: usize>() {
        let (codes, row_offsets) = input::<O>();
        let cover = cover::<POINTS, RANGES>();
        let mut expected = Vec::new();
        super::super::scan_scalar(&codes, &row_offsets, &cover, &mut expected);
        for sparse_row_mapping in [false, true] {
            let mut actual = Vec::new();
            unsafe {
                sse2::scan_sse2_fixed::<O, POINTS, RANGES>(
                    &codes,
                    &row_offsets,
                    &cover,
                    sparse_row_mapping,
                    &mut actual,
                )
            };
            assert_eq!(actual, expected);
        }
    }

    fn assert_full_domain_range<O: Offset>() {
        let (codes, row_offsets) = input::<O>();
        let cover = ProbeCover::from_membership(vec![true; usize::from(Token::MAX) + 1]);
        assert!(cover.points.is_empty());
        assert_eq!(
            cover.ranges,
            [crate::core::types::TokenRange {
                begin: Token::MIN,
                last: Token::MAX,
            }]
        );
        assert_cover::<O, 0, 1>(&codes, &row_offsets, &cover);
    }

    fn assert_all_shapes<O: Offset>() {
        assert_shape::<O, 1, 0>();
        assert_shape::<O, 2, 0>();
        assert_shape::<O, 3, 0>();
        assert_shape::<O, 0, 1>();
        assert_shape::<O, 1, 1>();
        assert_shape::<O, 2, 1>();
    }

    fn assert_all_sse2_shapes<O: Offset>() {
        macro_rules! assert_shapes {
            ($(($points:literal, $ranges:literal),)+) => {
                $(assert_sse2_shape::<O, $points, $ranges>();)+
            };
        }
        with_sse2_fixed_shapes!(assert_shapes);
    }

    fn edge_input<O: Offset>(len: usize, cover: &ProbeCover) -> (Vec<Token>, Vec<O>) {
        let miss = cover
            .table
            .iter()
            .position(|&covered| !covered)
            .expect("edge covers leave at least one token uncovered") as Token;
        let mut live = cover.points.clone();
        for range in &cover.ranges {
            live.push(range.begin);
            live.push(range.begin + (range.last - range.begin) / 2);
            live.push(range.last);
        }

        let mut codes = vec![miss; len];
        let mut positions = (0..live.len()).collect::<Vec<_>>();
        positions.extend([
            7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 112, 127, 128, 129, 255, 256, 257, 511,
            512,
        ]);
        positions.sort_unstable();
        positions.dedup();
        for (index, position) in positions
            .into_iter()
            .filter(|&position| position < len)
            .enumerate()
        {
            codes[position] = live[index % live.len()];
        }

        let one = len.min(1);
        let long_row_end = len.min(90);
        let offsets = [0, 0, one, one, long_row_end, long_row_end, len, len]
            .map(O::from_usize)
            .to_vec();
        (codes, offsets)
    }

    fn assert_generic_cover<O: Offset>(codes: &[Token], row_offsets: &[O], cover: &ProbeCover) {
        let mut expected = Vec::new();
        super::super::scan_scalar(codes, row_offsets, cover, &mut expected);
        for sparse_row_mapping in [false, true] {
            let mut sse2 = Vec::new();
            sse2::scan_sse2_generic(codes, row_offsets, cover, sparse_row_mapping, &mut sse2);
            assert_eq!(sse2, expected);

            if std::is_x86_feature_detected!("avx2") {
                let mut avx2 = Vec::new();
                unsafe {
                    avx2::scan_avx2_generic(
                        codes,
                        row_offsets,
                        cover,
                        sparse_row_mapping,
                        &mut avx2,
                    )
                };
                assert_eq!(avx2, expected);
            }

            if std::is_x86_feature_detected!("avx512bw") {
                let mut avx512 = Vec::new();
                unsafe {
                    avx512::scan_avx512(codes, row_offsets, cover, sparse_row_mapping, &mut avx512)
                };
                assert_eq!(avx512, expected);
            }
        }
    }

    fn assert_edge_matrix<O: Offset>() {
        const LENGTHS: [usize; 21] = [
            0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 113, 127, 128, 129, 511, 512, 513,
        ];
        const SHAPES: [(usize, usize); 20] = [
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (8, 0),
            (9, 0),
            (16, 0),
            (17, 0),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (1, 2),
            (2, 2),
            (3, 2),
            (4, 2),
            (0, 3),
            (1, 3),
            (8, 2),
            (17, 3),
        ];
        for (points, ranges) in SHAPES {
            let cover = edge_cover(points, ranges);
            for len in LENGTHS {
                let (codes, row_offsets) = edge_input::<O>(len, &cover);
                assert_generic_cover(&codes, &row_offsets, &cover);
            }
        }
    }

    #[test]
    fn fixed_producers_match_scalar_at_u16_boundaries() {
        assert_all_shapes::<u32>();
        assert_all_shapes::<u64>();
        assert_full_domain_range::<u32>();
        assert_full_domain_range::<u64>();
    }

    #[test]
    fn all_sse2_fixed_shapes_match_scalar() {
        assert_all_sse2_shapes::<u32>();
        assert_all_sse2_shapes::<u64>();
    }

    #[test]
    fn x86_edge_matrix_matches_scalar() {
        assert_edge_matrix::<u32>();
        assert_edge_matrix::<u64>();
    }
}
