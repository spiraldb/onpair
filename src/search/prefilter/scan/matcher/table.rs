// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! One table byte per u16 code, one load per code. L = 1, any K, any R, scalar.

use super::{Block, Mask, Matcher};
use crate::search::prefilter::ProbeCover;

pub(in crate::search::prefilter::scan) struct Table {
    /// 0xFF where the cover admits the code, not 1, so `& 1` reads as one bit.
    admits: Vec<u8>,
}

impl Matcher for Table {
    fn new(cover: &ProbeCover) -> Self {
        let mut admits = vec![0u8; 1 << 16];
        for &code in cover.points() {
            admits[usize::from(code)] = 0xFF;
        }
        for range in cover.ranges() {
            admits[usize::from(range.begin)..=usize::from(range.last)].fill(0xFF);
        }
        Self { admits }
    }

    fn check(&self, codes: &Block, bits: &mut Mask) -> bool {
        let mut any = 0;
        for (word, codes) in bits.iter_mut().zip(codes.chunks_exact(64)) {
            // Eight independent shift-OR chains per word.
            let mut packed = 0u64;
            for (group, codes) in codes.chunks_exact(8).enumerate() {
                let mut byte = 0u64;
                for (bit, &code) in codes.iter().enumerate() {
                    // SAFETY: the table has an entry for every u16.
                    let admits = unsafe { *self.admits.get_unchecked(usize::from(code)) };
                    byte |= u64::from(admits & 1) << bit;
                }
                packed |= byte << (8 * group);
            }
            *word = packed;
            any |= packed;
        }
        any != 0
    }
}
