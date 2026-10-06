// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Interleaved greedy parsing.
//!
//! Greedy parsing is one long dependency chain: each match starts where the
//! previous one ends, so a single row cannot go faster than one match latency
//! at a time. Rows are independent, though. The parser splits each batch of
//! rows into [`LANES`] lanes of about equal size and advances all lanes in one
//! loop with [`DictionaryMatcher::find_all`]. That step has no data-dependent
//! branches, so the CPU overlaps the memory accesses of the lanes. Each lane
//! writes its codes to its own buffer; the buffers are joined in row order
//! after the batch.

use std::hint::select_unpredictable;

use crate::core::offset::Offset;
use crate::core::types::{MAX_TOKEN_SIZE, Token};
use crate::encoding::lpm::DictionaryMatcher;
use crate::encoding::rows::Rows;

/// Number of lanes that advance together. Measured best: 4 on x86-64 (Sapphire
/// Rapids, Zen 4, Zen 5), 6 on Apple Silicon, 5 on Graviton4 (6 is within 3%).
/// Beyond that, lane state spills from registers.
#[cfg(target_arch = "x86_64")]
const LANES: usize = 4;
#[cfg(not(target_arch = "x86_64"))]
const LANES: usize = 6;

/// Approximate input bytes per batch. Large enough that the lanes of a batch
/// end at nearly the same time; small enough that the lane buffers stay in
/// cache.
const BATCH_BYTES: usize = 256 << 10;

/// Reusable per-lane buffers.
#[derive(Default)]
pub(crate) struct LaneBuffers {
    /// Codes of each lane in the current batch.
    codes: [Vec<Token>; LANES],
    /// Per lane: code count at the end of each row, then one slot for the
    /// writes of rounds that end no row.
    row_ends: [Vec<usize>; LANES],
    /// Copy of the rows of one batch, for inputs that are not contiguous.
    bytes: Vec<u8>,
    /// Row offsets into `bytes`.
    offsets: Vec<usize>,
}

/// Encode the rows of an Arrow pair: row `i` is
/// `bytes[offsets[i]..offsets[i + 1]]`. Appends each row's codes to `codes`
/// and each row's end, as an index into `codes`, to `row_offsets`.
///
/// The caller guarantees a valid Arrow pair: non-empty, monotonic offsets,
/// last offset `<= bytes.len()`.
pub(crate) fn parse_arrow<I: Offset, O: Offset>(
    matcher: &DictionaryMatcher,
    bytes: &[u8],
    offsets: &[I],
    buffers: &mut LaneBuffers,
    codes: &mut Vec<Token>,
    row_offsets: &mut Vec<O>,
) {
    parse_arrow_batched(
        matcher,
        bytes,
        offsets,
        BATCH_BYTES,
        buffers,
        codes,
        row_offsets,
    );
}

fn parse_arrow_batched<I: Offset, O: Offset>(
    matcher: &DictionaryMatcher,
    bytes: &[u8],
    offsets: &[I],
    batch_bytes: usize,
    buffers: &mut LaneBuffers,
    codes: &mut Vec<Token>,
    row_offsets: &mut Vec<O>,
) {
    let n = offsets.len() - 1;
    let mut start = 0;
    while start < n {
        // At least one row per batch, then whole rows up to `batch_bytes`.
        let limit = offsets[start].to_usize() + batch_bytes;
        let fit = offsets[start + 1..=n].partition_point(|o| o.to_usize() <= limit);
        let end = start + fit.max(1);
        parse_batch(
            matcher,
            bytes,
            &offsets[start..=end],
            buffers,
            codes,
            row_offsets,
        );
        start = end;
    }
}

/// Encode any [`Rows`] input. Rows are copied into a contiguous, padded
/// buffer one batch at a time, so the lanes can read past a row end.
pub(crate) fn parse_rows<R: Rows + ?Sized, O: Offset>(
    matcher: &DictionaryMatcher,
    rows: &R,
    buffers: &mut LaneBuffers,
    codes: &mut Vec<Token>,
    row_offsets: &mut Vec<O>,
) {
    parse_rows_batched(matcher, rows, BATCH_BYTES, buffers, codes, row_offsets);
}

fn parse_rows_batched<R: Rows + ?Sized, O: Offset>(
    matcher: &DictionaryMatcher,
    rows: &R,
    batch_bytes: usize,
    buffers: &mut LaneBuffers,
    codes: &mut Vec<Token>,
    row_offsets: &mut Vec<O>,
) {
    let mut bytes = std::mem::take(&mut buffers.bytes);
    let mut offsets = std::mem::take(&mut buffers.offsets);

    let n = rows.num_rows();
    let mut i = 0;
    while i < n {
        bytes.clear();
        offsets.clear();
        offsets.push(0);

        // At least one row per batch, then rows until `batch_bytes` is reached.
        while i < n && (offsets.len() == 1 || bytes.len() < batch_bytes) {
            bytes.extend_from_slice(rows.row(i));
            offsets.push(bytes.len());
            i += 1;
        }

        // Padding keeps every window load on the fast path.
        bytes.resize(bytes.len() + MAX_TOKEN_SIZE, 0);
        parse_batch(matcher, &bytes, &offsets, buffers, codes, row_offsets);
    }

    buffers.bytes = bytes;
    buffers.offsets = offsets;
}

/// Encode one batch of contiguous rows with all lanes.
fn parse_batch<I: OffsetLike, O: Offset>(
    matcher: &DictionaryMatcher,
    bytes: &[u8],
    offsets: &[I],
    buffers: &mut LaneBuffers,
    codes: &mut Vec<Token>,
    row_offsets: &mut Vec<O>,
) {
    let n = offsets.len() - 1;
    let first = offsets[0].to_usize();
    let total = offsets[n].to_usize() - first;

    // Lane `j` parses rows `bounds[j]..bounds[j + 1]`, split by byte count.
    let mut bounds = [0usize; LANES + 1];
    for (j, bound) in bounds.iter_mut().enumerate().skip(1) {
        let target = first + total * j / LANES;
        *bound = offsets.partition_point(|o| o.to_usize() < target).min(n);
    }
    bounds[LANES] = n;

    // Per lane: position, end of the current row, current row, end row, and
    // number of codes written.
    let mut at = [0usize; LANES];
    let mut row_end = [0usize; LANES];
    let mut row = [0usize; LANES];
    let mut last_row = [0usize; LANES];
    let mut count = [0usize; LANES];

    for j in 0..LANES {
        let (r0, r1) = (bounds[j], bounds[j + 1]);
        let start = offsets[r0].to_usize();
        at[j] = start;
        row[j] = r0;
        last_row[j] = r1;

        // An empty lane starts finished: no bytes left.
        row_end[j] = if r0 < r1 {
            offsets[r0 + 1].to_usize()
        } else {
            start
        };

        // One code per byte at most, plus one slot for the discarded write of
        // an inactive lane.
        let lane_bytes = offsets[r1].to_usize() - start;
        buffers.codes[j].clear();
        buffers.codes[j].resize(lane_bytes + 1, 0);
        buffers.row_ends[j].clear();
        buffers.row_ends[j].resize(r1 - r0 + 1, 0);
    }

    loop {
        // An inactive lane still runs one search; its result is discarded.
        let win: [usize; LANES] =
            std::array::from_fn(|j| (row_end[j] - at[j]).clamp(1, MAX_TOKEN_SIZE));
        let found = matcher.find_all(bytes, &at, &win);

        let mut busy = false;
        for j in 0..LANES {
            let running = row[j] < last_row[j];
            busy |= running;

            let active = row_end[j] > at[j];
            let (token, len) = found[j];
            buffers.codes[j][count[j]] = token;
            count[j] += active as usize;
            at[j] += select_unpredictable(active, len, 0);

            // A row ends when its bytes are used up. The next row starts where
            // this one ends.
            let done = at[j] == row_end[j] && running;
            let slot =
                select_unpredictable(done, row[j] - bounds[j], buffers.row_ends[j].len() - 1);
            buffers.row_ends[j][slot] = count[j];
            row[j] += done as usize;

            // After its last row a lane keeps `row_end == at`, so it stays
            // inactive.
            let next_end = offsets[(row[j] + 1).min(n)].to_usize();
            let next_end = select_unpredictable(row[j] < last_row[j], next_end, at[j]);
            row_end[j] = select_unpredictable(done, next_end, row_end[j]);
        }

        if !busy {
            break;
        }
    }

    for j in 0..LANES {
        let base = codes.len();
        codes.extend_from_slice(&buffers.codes[j][..count[j]]);

        let rows = bounds[j + 1] - bounds[j];
        row_offsets.extend(
            buffers.row_ends[j][..rows]
                .iter()
                .map(|&end| O::from_usize(base + end)),
        );
    }
}

/// Row offsets of a batch: the caller's Arrow offsets, or `usize` offsets
/// into a batch copy.
trait OffsetLike: Copy {
    fn to_usize(self) -> usize;
}

impl<T: Offset> OffsetLike for T {
    #[inline]
    fn to_usize(self) -> usize {
        Offset::to_usize(self)
    }
}

impl OffsetLike for usize {
    #[inline]
    fn to_usize(self) -> usize {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dictionary::Dictionary;
    use crate::search::tokenize;
    use crate::test_corpus::{
        binary_strings, make_raw, mixed_length_strings, random_ascii_strings, user_strings,
    };
    use crate::{DEFAULT_CONFIG, Parser};

    /// Codes and row offsets from the one-input matcher, row by row.
    fn reference(matcher: &DictionaryMatcher, rows: &[&[u8]]) -> (Vec<Token>, Vec<u64>) {
        let mut codes = Vec::new();
        let mut row_offsets = vec![0u64];
        for row in rows {
            let mut pos = 0;
            while pos < row.len() {
                let (token, len) = matcher.find_longest_match(&row[pos..]);
                codes.push(token);
                pos += len;
            }
            row_offsets.push(codes.len() as u64);
        }
        (codes, row_offsets)
    }

    fn check<S: AsRef<[u8]>, T: AsRef<[u8]>>(train: &[S], input: &[T]) {
        let traw = make_raw(train);
        let parser = Parser::train(&traw.data, &traw.offsets, DEFAULT_CONFIG).unwrap();
        let rows: Vec<&[u8]> = input.iter().map(AsRef::as_ref).collect();
        let (expected_codes, expected_offsets) = reference(&parser.lpm, &rows);

        // The one-input matcher agrees with the dictionary-only tokenizer.
        let flat: Vec<Token> = rows
            .iter()
            .flat_map(|row| tokenize(row, parser.dict.as_view()))
            .collect();
        assert_eq!(flat, expected_codes);

        let raw = make_raw(input);
        for batch in [1, 7, 64, 1 << 20] {
            let mut buffers = LaneBuffers::default();

            let mut codes = Vec::new();
            let mut row_offsets = vec![0u64];
            parse_arrow_batched(
                &parser.lpm,
                &raw.data,
                &raw.offsets,
                batch,
                &mut buffers,
                &mut codes,
                &mut row_offsets,
            );
            assert_eq!(codes, expected_codes, "arrow, batch {batch}");
            assert_eq!(row_offsets, expected_offsets, "arrow, batch {batch}");

            let mut codes = Vec::new();
            let mut row_offsets = vec![0u64];
            parse_rows_batched(
                &parser.lpm,
                rows.as_slice(),
                batch,
                &mut buffers,
                &mut codes,
                &mut row_offsets,
            );
            assert_eq!(codes, expected_codes, "rows, batch {batch}");
            assert_eq!(row_offsets, expected_offsets, "rows, batch {batch}");
        }
    }

    #[test]
    fn lanes_match_reference() {
        let users = user_strings(300);
        check(&users, &users);
        check(&users, &random_ascii_strings(200, 40, 3));
        let mixed = mixed_length_strings(300, 100, 9);
        check(&mixed, &mixed);
        let binary = binary_strings(200, 30, 5);
        check(&binary, &binary);
    }

    #[test]
    fn lanes_handle_empty_rows_and_tiny_inputs() {
        let train = user_strings(100);
        let empty: [&[u8]; 0] = [];
        check(&train, &empty);
        check(&train, &[b"".as_slice()]);
        check(&train, &[b"a".as_slice()]);
        check(
            &train,
            &[b"".as_slice(), b"a", b"", b"", b"hello world", b"x", b""],
        );
    }

    #[test]
    fn lanes_handle_nonzero_first_offset() {
        let traw = make_raw(&user_strings(100));
        let parser = Parser::train(&traw.data, &traw.offsets, DEFAULT_CONFIG).unwrap();
        let bytes = b"skipalphabetaunused";
        let offsets = [4u32, 9, 13];
        let rows: [&[u8]; 2] = [b"alpha", b"beta"];
        let (expected_codes, expected_offsets) = reference(&parser.lpm, &rows);

        let mut codes = Vec::new();
        let mut row_offsets = vec![0u64];
        parse_arrow(
            &parser.lpm,
            bytes,
            &offsets,
            &mut LaneBuffers::default(),
            &mut codes,
            &mut row_offsets,
        );
        assert_eq!((codes, row_offsets), (expected_codes, expected_offsets));
    }
}
