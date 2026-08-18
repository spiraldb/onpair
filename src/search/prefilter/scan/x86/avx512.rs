// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! AVX-512BW prefilter execution.

use super::super::sink::{RowSink, scan_tail};
use crate::core::offset::Offset;
use crate::core::types::{Token, TokenRange};
use crate::search::prefilter::cover::ProbeCover;

/// AVX-512BW: 32 lanes with native unsigned comparisons and mask output.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub(in crate::search::prefilter) fn scan_avx512<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    use core::arch::x86_64::{
        _mm512_cmpeq_epu16_mask, _mm512_cmpge_epu16_mask, _mm512_cmple_epu16_mask,
        _mm512_loadu_si512, _mm512_set1_epi16,
    };

    let total = codes.len();
    let base = codes.as_ptr();
    let mut sink = RowSink::new(row_offsets, out, sparse_row_mapping);
    let mut i = 0usize;
    while i + 32 <= total {
        // SAFETY: `i + 32 <= total`; the caller established AVX-512F/BW.
        let mut m = unsafe {
            let v = _mm512_loadu_si512(base.add(i).cast());
            let mut acc: u32 = 0;
            for &p in &pf.points {
                acc |= _mm512_cmpeq_epu16_mask(v, _mm512_set1_epi16(p as i16));
            }
            for &TokenRange { begin, last } in &pf.ranges {
                let ge = _mm512_cmpge_epu16_mask(v, _mm512_set1_epi16(begin as i16));
                let le = _mm512_cmple_epu16_mask(v, _mm512_set1_epi16(last as i16));
                acc |= ge & le;
            }
            acc
        };
        // Lowest set lane first, so code indices stay increasing.
        while m != 0 {
            let j = m.trailing_zeros() as usize;
            sink.hit(i + j);
            m &= m - 1;
        }
        i += 32;
    }
    scan_tail(codes, pf, i, &mut sink);
}
