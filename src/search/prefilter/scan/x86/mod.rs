// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! x86-64 scan execution and target-feature boundaries.

use super::ScanInput;
use super::policy::{
    Avx2Group, Avx2Kernel, Avx512Kernel, FixedShape, Sse2Kernel, with_sse2_fixed_shapes,
    with_x86_fixed_shapes,
};
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
    kernel: Sse2Kernel,
    input: ScanInput<'_, O>,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    let ScanInput {
        codes,
        row_offsets,
        cover,
    } = input;
    match kernel {
        Sse2Kernel::Fixed(shape) => {
            execute_sse2_fixed(shape, codes, row_offsets, cover, sparse_row_mapping, out)
        }
        Sse2Kernel::Generic => {
            sse2::scan_sse2_generic(codes, row_offsets, cover, sparse_row_mapping, out)
        }
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
    kernel: Avx2Kernel,
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
        match kernel {
            Avx2Kernel::Fixed {
                shape,
                group: Avx2Group::One,
            } => execute_avx2_fixed::<O, 1>(
                shape,
                codes,
                row_offsets,
                cover,
                sparse_row_mapping,
                out,
            ),
            Avx2Kernel::Fixed {
                shape,
                group: Avx2Group::Eight,
            } => execute_avx2_fixed::<O, 8>(
                shape,
                codes,
                row_offsets,
                cover,
                sparse_row_mapping,
                out,
            ),
            Avx2Kernel::Generic => {
                avx2::scan_avx2_generic(codes, row_offsets, cover, sparse_row_mapping, out)
            }
        }
    }
}

unsafe fn execute_avx2_fixed<O: Offset, const BLOCKS: usize>(
    shape: FixedShape,
    codes: &[Token],
    row_offsets: &[O],
    cover: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    macro_rules! call {
        ($points:literal, $ranges:literal) => {
            avx2::scan_avx2_fixed::<O, $points, $ranges, BLOCKS>(
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
    kernel: Avx512Kernel,
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
        match kernel {
            Avx512Kernel::Fixed(shape) => {
                execute_avx512_fixed(shape, codes, row_offsets, cover, sparse_row_mapping, out)
            }
            Avx512Kernel::Generic => {
                avx512::scan_avx512(codes, row_offsets, cover, sparse_row_mapping, out)
            }
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

    fn cover<const POINTS: usize, const RANGES: usize>() -> ProbeCover {
        let mut table = vec![false; usize::from(Token::MAX) + 1];
        match (POINTS, RANGES) {
            (1, 0) => table[usize::from(Token::MAX)] = true,
            (2, 0) => {
                table[7] = true;
                table[usize::from(Token::MAX)] = true;
            }
            (3, 0) => {
                table[7] = true;
                table[32_768] = true;
                table[usize::from(Token::MAX)] = true;
            }
            (0, 1) => table[65_532..=65_535].fill(true),
            (1, 1) => {
                table[7] = true;
                table[65_532..=65_535].fill(true);
            }
            (2, 1) => {
                table[7] = true;
                table[32_768] = true;
                table[65_532..=65_535].fill(true);
            }
            _ => unreachable!(),
        }
        let cover = ProbeCover::from_membership(table);
        assert_eq!((cover.points.len(), cover.ranges.len()), (POINTS, RANGES));
        if RANGES != 0 {
            assert_eq!(
                cover.ranges[0],
                crate::core::types::TokenRange {
                    begin: 65_532,
                    last: Token::MAX,
                }
            );
        }
        cover
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

    #[test]
    fn fixed_producers_match_scalar_at_u16_boundaries() {
        assert_all_shapes::<u32>();
        assert_all_shapes::<u64>();
        assert_full_domain_range::<u32>();
        assert_full_domain_range::<u64>();
    }
}
