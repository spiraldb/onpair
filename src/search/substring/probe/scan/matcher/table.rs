// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar membership through a lookup table with one byte per possible code.
//!
//! The portable fallback: it supports every cover and needs no vector instructions.

use super::{Block, Mask, Matcher};
use crate::search::substring::ProbeCover;

/// The membership table of one cover.
pub(crate) struct Table {
    /// 0xFF for covered IDs and zero otherwise; packing extracts the low bit. One
    /// entry per `u16`, so indexing by a code needs no bounds check.
    admits: Box<[u8; 1 << 16]>,
}

impl Matcher for Table {
    unsafe fn new(cover: &ProbeCover) -> Self {
        // Built on the heap, without a 64 KiB array on the stack.
        let mut admits: Box<[u8; 1 << 16]> = vec![0; 1 << 16].try_into().unwrap();
        for &code in cover.points() {
            admits[usize::from(code)] = 0xFF;
        }
        for range in cover.ranges() {
            admits[usize::from(range.begin)..=usize::from(range.last)].fill(0xFF);
        }
        Self { admits }
    }

    /// Always packs: a lookup per code leaves nothing to skip.
    fn check(&self, codes: &Block, bits: &mut Mask, _skip_empty: bool) -> bool {
        let mut any = 0;
        for (word, codes) in bits.iter_mut().zip(codes.chunks_exact(64)) {
            // Eight independent shift-OR chains per word.
            let mut packed = 0u64;
            for (group, codes) in codes.chunks_exact(8).enumerate() {
                let mut byte = 0u64;
                for (bit, &code) in codes.iter().enumerate() {
                    byte |= u64::from(self.admits[usize::from(code)] & 1) << bit;
                }
                packed |= byte << (8 * group);
            }
            *word = packed;
            any |= packed;
        }
        any != 0
    }
}
