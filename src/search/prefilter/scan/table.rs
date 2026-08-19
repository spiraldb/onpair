// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Exact membership-table scans and AVX2 table preparation.

use super::super::cover::ProbeCover;
use super::policy::{BAIL_MIN_SEEN_DIVISOR, BAIL_RATIO_DEN, BAIL_RATIO_NUM};
use super::sink::RowSink;
use crate::core::offset::Offset;
use crate::core::types::{Token, TokenRange};

/// Scan row by row and stop at the first covered code in each row.
///
/// With `bail` set, the scan may stop pruning once almost every row it has
/// decided turned out to be a candidate (see the `BAIL_*` policy constants),
/// appending all remaining rows: still a sound superset, exact up to the
/// bail point.
pub(super) fn scan_rows<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    cover: &ProbeCover,
    bail: bool,
    out: &mut Vec<usize>,
) {
    let rows = row_offsets.len().saturating_sub(1);
    let initial = out.len();
    let min_seen = (rows / BAIL_MIN_SEEN_DIVISOR).max(1);
    for row in 0..rows {
        let begin = row_offsets[row].to_usize();
        let end = row_offsets[row + 1].to_usize();
        if codes[begin..end]
            .iter()
            .any(|&code| cover.table[code as usize])
        {
            out.push(row);
        }
        if bail && row & 0x0fff == 0x0fff && row + 1 >= min_seen {
            let appended = out.len() - initial;
            if appended * BAIL_RATIO_DEN >= (row + 1) * BAIL_RATIO_NUM {
                out.extend(row + 1..rows);
                return;
            }
        }
    }
}

/// Scan codes in order and map covered positions back to rows.
pub(super) fn scan_codes<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    cover: &ProbeCover,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    let mut sink = RowSink::new(row_offsets, out, sparse_row_mapping);
    for (code_index, &code) in codes.iter().enumerate() {
        // SAFETY: compressed codes are dictionary token ids, and the cover
        // table has exactly one entry per dictionary token.
        if unsafe { *cover.table.get_unchecked(code as usize) } {
            sink.hit(code_index);
        }
    }
}

/// Pack the normalized point/range cover without revisiting every dictionary id.
pub(super) fn pack_membership(cover: &ProbeCover) -> Vec<u32> {
    const WORD_BITS: usize = u32::BITS as usize;
    let mut table = vec![0u32; cover.table.len().div_ceil(WORD_BITS)];
    for &point in &cover.points {
        let id = point as usize;
        table[id / WORD_BITS] |= 1 << (id % WORD_BITS);
    }
    for &TokenRange { begin, last } in &cover.ranges {
        let begin = begin as usize;
        let last = last as usize;
        let begin_word = begin / WORD_BITS;
        let last_word = last / WORD_BITS;
        let first_mask = u32::MAX << (begin % WORD_BITS);
        let last_mask = u32::MAX >> (WORD_BITS - 1 - last % WORD_BITS);
        if begin_word == last_word {
            table[begin_word] |= first_mask & last_mask;
        } else {
            table[begin_word] |= first_mask;
            table[begin_word + 1..last_word].fill(u32::MAX);
            table[last_word] |= last_mask;
        }
    }
    table
}

/// Pack membership by walking the complete dictionary-sized table.
///
/// This is cheaper than expanding many normalized ranges for very wide 12-bit
/// covers, where the source table is only 4 KiB and remains cache-resident.
pub(super) fn pack_membership_dense(cover: &ProbeCover) -> Vec<u32> {
    const WORD_BITS: usize = u32::BITS as usize;
    let mut packed = vec![0u32; cover.table.len().div_ceil(WORD_BITS)];
    for (id, selected) in cover.table.iter().copied().enumerate() {
        packed[id / WORD_BITS] |= u32::from(selected) << (id % WORD_BITS);
    }
    packed
}
