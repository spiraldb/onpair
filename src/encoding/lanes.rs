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
//!
//! Lanes parse only the non-empty rows. An empty row needs no search, and
//! columns of short strings often hold many of them; its end is the end of
//! the last non-empty row before it.

use std::hint::select_unpredictable;

use crate::core::offset::Offset;
use crate::core::types::{MAX_TOKEN_SIZE, Token};
use crate::encoding::dictionary_matcher::DictionaryMatcher;
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

/// Maximum rows per batch. Short or empty rows fill a batch with many rows
/// before [`BATCH_BYTES`]; this bound keeps the per-row buffers in cache.
const BATCH_ROWS: usize = 1 << 14;

/// Reusable per-lane buffers.
#[derive(Default)]
pub(crate) struct LaneBuffers {
    /// Codes of each lane in the current batch.
    codes: [Vec<Token>; LANES],
    /// Per lane: code count at the end of each row, then one spare slot for
    /// the writes after the last row.
    row_ends: [Vec<usize>; LANES],
    /// Byte range of each non-empty row of the current batch.
    spans: Vec<(usize, usize)>,
    /// Entry `k` is the code count at the end of the first `k` non-empty rows
    /// of the current batch, as an index into the output codes.
    span_ends: Vec<usize>,
    /// Copy of the rows of one batch, for inputs that are not contiguous.
    bytes: Vec<u8>,
    /// Per row of the current batch: the number of non-empty rows up to and
    /// including it.
    filled: Vec<u32>,
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
    let mut spans = std::mem::take(&mut buffers.spans);
    let mut span_ends = std::mem::take(&mut buffers.span_ends);

    let n = offsets.len() - 1;
    let mut start = 0;
    while start < n {
        // At least one row per batch, then whole rows up to `batch_bytes` and
        // `BATCH_ROWS`.
        let limit = offsets[start].to_usize() + batch_bytes;
        let last = n.min(start + BATCH_ROWS);
        let fit = offsets[start + 1..=last].partition_point(|o| o.to_usize() <= limit);
        let end = start + fit.max(1);
        let batch = &offsets[start..=end];

        spans.clear();
        for w in batch.windows(2) {
            let (row_start, row_end) = (w[0].to_usize(), w[1].to_usize());
            if row_end > row_start {
                spans.push((row_start, row_end));
            }
        }

        parse_spans(matcher, bytes, &mut spans, buffers, codes, &mut span_ends);

        // A row ends where the last non-empty row up to it ends.
        let mut k = 0;
        row_offsets.extend(batch.windows(2).map(|w| {
            k += (w[1].to_usize() > w[0].to_usize()) as usize;
            O::from_usize(span_ends[k])
        }));

        start = end;
    }

    buffers.spans = spans;
    buffers.span_ends = span_ends;
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
    let mut filled = std::mem::take(&mut buffers.filled);
    let mut spans = std::mem::take(&mut buffers.spans);
    let mut span_ends = std::mem::take(&mut buffers.span_ends);

    let n = rows.num_rows();
    let mut i = 0;
    while i < n {
        bytes.clear();
        filled.clear();
        spans.clear();

        // At least one row per batch, then rows until `batch_bytes` or
        // `BATCH_ROWS` is reached. Empty rows add no bytes and no span.
        let last = n.min(i + BATCH_ROWS);
        while i < last && (filled.is_empty() || bytes.len() < batch_bytes) {
            let row = rows.row(i);
            if !row.is_empty() {
                let start = bytes.len();
                bytes.extend_from_slice(row);
                spans.push((start, bytes.len()));
            }
            filled.push(spans.len() as u32);
            i += 1;
        }

        // Padding keeps every window load on the fast path.
        bytes.resize(bytes.len() + MAX_TOKEN_SIZE, 0);
        parse_spans(matcher, &bytes, &mut spans, buffers, codes, &mut span_ends);

        // A row ends where the last non-empty row up to it ends.
        row_offsets.extend(filled.iter().map(|&k| O::from_usize(span_ends[k as usize])));
    }

    buffers.bytes = bytes;
    buffers.filled = filled;
    buffers.spans = spans;
    buffers.span_ends = span_ends;
}

/// Encode the non-empty rows `spans`, given as byte ranges of `bytes` in
/// order, with all lanes. Appends their codes to `codes`. Sets `span_ends[k]`
/// to the length of `codes` after the first `k` rows.
///
/// Every row must hold at least one byte, and each row must start where the
/// one before it ends.
fn parse_spans(
    matcher: &DictionaryMatcher,
    bytes: &[u8],
    spans: &mut Vec<(usize, usize)>,
    buffers: &mut LaneBuffers,
    codes: &mut Vec<Token>,
    span_ends: &mut Vec<usize>,
) {
    span_ends.clear();
    span_ends.push(codes.len());

    let m = spans.len();
    let (Some(&(first, _)), Some(&(_, rest))) = (spans.first(), spans.last()) else {
        return;
    };

    // Lane `j` parses rows `bounds[j]..bounds[j + 1]`, split by byte count.
    let total = rest - first;
    let mut bounds = [0usize; LANES + 1];
    for (j, bound) in bounds.iter_mut().enumerate().skip(1) {
        let target = first + total * j / LANES;
        *bound = spans.partition_point(|&(start, _)| start < target);
    }
    bounds[LANES] = m;

    // A spare span past the last row: lanes read it after their last row.
    spans.push((rest, rest));

    // Per lane: position, end of the current row, current row, end row, and
    // number of codes written.
    let mut at = [0usize; LANES];
    let mut row_end = [0usize; LANES];
    let mut row = [0usize; LANES];
    let mut last_row = [0usize; LANES];
    let mut count = [0usize; LANES];

    for j in 0..LANES {
        let (r0, r1) = (bounds[j], bounds[j + 1]);
        row[j] = r0;
        last_row[j] = r1;

        // An empty lane starts finished: no bytes left.
        let (start, end) = spans[r0];
        at[j] = start;
        row_end[j] = select_unpredictable(r0 < r1, end, start);

        // One code per byte at most, plus one slot for the discarded write of
        // an inactive lane.
        let lane_bytes = spans[r1].0 - start;
        buffers.codes[j].clear();
        buffers.codes[j].resize(lane_bytes + 1, 0);
        buffers.row_ends[j].clear();
        buffers.row_ends[j].resize(r1 - r0 + 1, 0);
    }

    loop {
        // An inactive lane still runs one search; its result is discarded.
        let mut win = [0usize; LANES];
        for j in 0..LANES {
            win[j] = (row_end[j] - at[j]).clamp(1, MAX_TOKEN_SIZE);
        }
        let found = matcher.find_all(bytes, &at, &win);

        let mut busy = false;
        for j in 0..LANES {
            let running = row[j] < last_row[j];
            busy |= running;

            let (token, len) = found[j];
            buffers.codes[j][count[j]] = token;
            count[j] += running as usize;
            at[j] += select_unpredictable(running, len, 0);

            // The slot of the current row holds its code count so far, so it
            // is final when the row ends. A lane past its last row writes the
            // spare slot.
            buffers.row_ends[j][row[j] - bounds[j]] = count[j];

            // A row ends when its bytes are used up.
            let done = at[j] == row_end[j] && running;
            row[j] += done as usize;

            // The next row starts where this one ends. After its last row a
            // lane keeps `row_end == at`, so it stays inactive.
            let next_end = spans[row[j].min(m)].1;
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
        span_ends.extend(buffers.row_ends[j][..rows].iter().map(|&end| base + end));
    }
}

#[cfg(test)]
mod tests {
    use hegel::TestCase;
    use hegel::generators as gs;

    use super::*;
    use crate::core::dictionary::{Dictionary, DictionaryView};
    use crate::search::tokenize;
    use crate::test_corpus::{
        PROPERTY_ALPHABET, binary_strings, complete_dictionary, make_raw, mixed_length_strings,
        random_ascii_strings, trainer_matcher, user_strings,
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

    #[hegel::test(test_cases = 300)]
    fn lanes_agree_with_trainer_matcher(tc: TestCase) {
        let bytes = || gs::vecs(gs::sampled_from(&PROPERTY_ALPHABET[..]));
        let extra: Vec<Vec<u8>> =
            tc.draw(gs::vecs(bytes().min_size(2).max_size(MAX_TOKEN_SIZE)).max_size(300));
        let mut rows: Vec<Vec<u8>> = tc.draw(gs::vecs(bytes().max_size(40)).max_size(40));
        let batch = tc.draw(gs::integers::<usize>().min_value(1).max_value(128));

        // Runs of one byte reach the pad key: 16 bytes of `0xFF`.
        for row in &mut rows {
            if let Some(&first) = row.first()
                && tc.draw(gs::booleans())
            {
                row.fill(first);
            }
        }

        let extra: Vec<&[u8]> = extra.iter().map(Vec::as_slice).collect();
        let dict = complete_dictionary(&extra);
        let view = dict.as_view();
        let matcher = DictionaryMatcher::from_dictionary(view);
        let (trainer, trainer_tokens) = trainer_matcher(view);

        // Greedy parse row by row with the trainer's matcher, as token bytes.
        let mut expected: Vec<&[u8]> = Vec::new();
        let mut expected_offsets = vec![0u64];
        for row in &rows {
            let mut pos = 0;
            while pos < row.len() {
                let (token, len) = trainer.find_longest_match(&row[pos..]);
                expected.push(&trainer_tokens[usize::from(token)]);
                pos += len;
            }
            expected_offsets.push(expected.len() as u64);
        }

        let raw = make_raw(&rows);
        let slices: Vec<&[u8]> = rows.iter().map(Vec::as_slice).collect();
        let mut buffers = LaneBuffers::default();
        for arrow in [true, false] {
            let mut codes = Vec::new();
            let mut row_offsets = vec![0u64];
            if arrow {
                parse_arrow_batched(
                    &matcher,
                    &raw.data,
                    &raw.offsets,
                    batch,
                    &mut buffers,
                    &mut codes,
                    &mut row_offsets,
                );
            } else {
                parse_rows_batched(
                    &matcher,
                    slices.as_slice(),
                    batch,
                    &mut buffers,
                    &mut codes,
                    &mut row_offsets,
                );
            }

            let tokens: Vec<&[u8]> = codes.iter().map(|&c| view.token(c)).collect();
            assert_eq!(tokens, expected, "arrow {arrow}");
            assert_eq!(row_offsets, expected_offsets, "arrow {arrow}");
            assert_eq!(tokens.concat(), raw.data, "arrow {arrow}");
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
    fn lanes_handle_mostly_empty_rows_across_row_bound() {
        // More rows than `BATCH_ROWS`, so batches end on the row bound.
        let train = user_strings(100);
        let input: Vec<&[u8]> = (0..BATCH_ROWS * 2 + 3)
            .map(|i| match i % 7 {
                0 => b"hello".as_slice(),
                3 => b"x",
                _ => b"",
            })
            .collect();
        check(&train, &input);
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
