// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Interleaved greedy parsing.
//!
//! Greedy parsing is one long dependency chain: each match starts where the
//! previous one ends, so a single row cannot go faster than one match latency
//! at a time. Rows are independent, though. The parser splits each batch of
//! rows into [`LANES`] lanes of about equal size and advances all lanes in one
//! loop with the selected matcher. Independent lanes overlap their memory
//! accesses. The predecessor search remains exact in every variant. Each lane
//! writes its codes to its own buffer; the buffers are joined in row order
//! after the batch.

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

/// Four lanes suit the tree's completion branches; six overlap the flat rank
/// search best on the measured prose and URL tuning inputs.
const TREE_LANES: usize = 4;
const RANK_LANES: usize = 6;
const SCALAR: u8 = 0;
#[cfg(target_arch = "x86_64")]
const TREE: u8 = 1;
#[cfg(target_arch = "x86_64")]
const RANK: u8 = 2;
#[cfg(target_arch = "x86_64")]
const RANK_ROWS: u8 = 3;
const MAX_LANES: usize = if TREE_LANES > RANK_LANES && TREE_LANES > LANES {
    TREE_LANES
} else if RANK_LANES > LANES {
    RANK_LANES
} else {
    LANES
};

/// Reusable per-lane buffers.
#[derive(Default)]
pub(crate) struct LaneBuffers {
    /// Codes of each lane in the current batch.
    codes: [Vec<Token>; MAX_LANES],
    /// Per lane: code count at the end of each row, then one spare slot for
    /// the writes after the last row.
    row_ends: [Vec<usize>; MAX_LANES],
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
    if matcher.is_single_byte_only() {
        let first = offsets[0].to_usize();
        let last = offsets[offsets.len() - 1].to_usize();
        let base = codes.len();
        codes.extend(bytes[first..last].iter().copied().map(Token::from));
        row_offsets.extend(
            offsets[1..]
                .iter()
                .map(|o| O::from_usize(base + o.to_usize() - first)),
        );
        return;
    }
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
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("popcnt")
    {
        let first = offsets[0].to_usize();
        let last = offsets[offsets.len() - 1].to_usize();
        if last - first >= BATCH_BYTES && matcher.prefers_tree(&bytes[first..last]) {
            return parse_arrow_batched_with::<I, O, TREE>(
                matcher,
                bytes,
                offsets,
                batch_bytes,
                buffers,
                codes,
                row_offsets,
            );
        }
        return parse_arrow_batched_with::<I, O, RANK>(
            matcher,
            bytes,
            offsets,
            batch_bytes,
            buffers,
            codes,
            row_offsets,
        );
    }

    parse_arrow_batched_with::<I, O, SCALAR>(
        matcher,
        bytes,
        offsets,
        batch_bytes,
        buffers,
        codes,
        row_offsets,
    );
}

fn parse_arrow_batched_with<I: Offset, O: Offset, const MODE: u8>(
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
    #[cfg(target_arch = "x86_64")]
    let mut parse: BatchParser<I, O> = parse_batch::<I, O, MODE>;
    #[cfg(target_arch = "x86_64")]
    let mut probe = MODE == RANK;
    while start < n {
        // At least one row per batch, then whole rows up to `batch_bytes`.
        let limit = offsets[start].to_usize() + batch_bytes;
        let fit = offsets[start + 1..=n].partition_point(|o| o.to_usize() <= limit);
        let end = start + fit.max(1);
        #[cfg(target_arch = "x86_64")]
        let end = if probe { end.min(start + 64) } else { end };
        #[cfg(target_arch = "x86_64")]
        {
            let before = codes.len();
            parse(
                matcher,
                bytes,
                &offsets[start..=end],
                buffers,
                codes,
                row_offsets,
            );
            if probe {
                parse = rank_batch_parser::<I, O>(codes.len() - before, end - start);
                probe = false;
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        parse_batch::<I, O, MODE>(
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
    if matcher.is_single_byte_only() {
        for i in 0..rows.num_rows() {
            codes.extend(rows.row(i).iter().copied().map(Token::from));
            row_offsets.push(O::from_usize(codes.len()));
        }
        return;
    }
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
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("popcnt")
    {
        return parse_rows_batched_with::<R, O, RANK>(
            matcher,
            rows,
            batch_bytes,
            buffers,
            codes,
            row_offsets,
        );
    }

    parse_rows_batched_with::<R, O, SCALAR>(
        matcher,
        rows,
        batch_bytes,
        buffers,
        codes,
        row_offsets,
    );
}

#[cfg(target_arch = "x86_64")]
type BatchParser<I, O> =
    fn(&DictionaryMatcher, &[u8], &[I], &mut LaneBuffers, &mut Vec<Token>, &mut Vec<O>);

#[cfg(target_arch = "x86_64")]
fn rank_batch_parser<I: OffsetLike, O: Offset>(codes: usize, rows: usize) -> BatchParser<I, O> {
    // Branches pay off when row completions are infrequent. Use measured codes
    // per row, accounting for the input's actual dictionary token frequencies.
    if codes >= 16 * rows {
        parse_batch::<I, O, RANK_ROWS>
    } else {
        parse_batch::<I, O, RANK>
    }
}

fn parse_rows_batched_with<R: Rows + ?Sized, O: Offset, const MODE: u8>(
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
    #[cfg(target_arch = "x86_64")]
    let mut parse: BatchParser<usize, O> = parse_batch::<usize, O, MODE>;
    #[cfg(target_arch = "x86_64")]
    let mut probe = MODE == RANK;
    while i < n {
        bytes.clear();
        offsets.clear();
        offsets.push(0);

        // Parse up to 64 rows with the branch-free loop to measure how often
        // rows end. The probe contributes directly to the output.
        #[cfg(target_arch = "x86_64")]
        let limit = if probe { n.min(i + 64) } else { n };
        #[cfg(not(target_arch = "x86_64"))]
        let limit = n;
        // At least one row per batch, then rows until `batch_bytes` is reached.
        while i < limit && (offsets.len() == 1 || bytes.len() < batch_bytes) {
            bytes.extend_from_slice(rows.row(i));
            offsets.push(bytes.len());
            i += 1;
        }

        // Padding keeps every window load on the fast path.
        bytes.resize(bytes.len() + MAX_TOKEN_SIZE, 0);
        #[cfg(target_arch = "x86_64")]
        {
            let before = codes.len();
            parse(matcher, &bytes, &offsets, buffers, codes, row_offsets);
            if probe {
                parse = rank_batch_parser::<usize, O>(codes.len() - before, offsets.len() - 1);
                probe = false;
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        parse_batch::<usize, O, MODE>(matcher, &bytes, &offsets, buffers, codes, row_offsets);
    }

    buffers.bytes = bytes;
    buffers.offsets = offsets;
}

/// Encode one batch of contiguous rows with all lanes.
fn parse_batch<I: OffsetLike, O: Offset, const MODE: u8>(
    matcher: &DictionaryMatcher,
    bytes: &[u8],
    offsets: &[I],
    buffers: &mut LaneBuffers,
    codes: &mut Vec<Token>,
    row_offsets: &mut Vec<O>,
) {
    #[cfg(target_arch = "x86_64")]
    if MODE == RANK_ROWS {
        // SAFETY: the parse entry point detected AVX-512F and POPCNT.
        return unsafe {
            parse_batch_rank_rows(matcher, bytes, offsets, buffers, codes, row_offsets)
        };
    }
    #[cfg(target_arch = "x86_64")]
    if MODE == RANK {
        // SAFETY: the parse entry point detected AVX-512F and POPCNT.
        return unsafe { parse_batch_rank(matcher, bytes, offsets, buffers, codes, row_offsets) };
    }
    #[cfg(target_arch = "x86_64")]
    if MODE == TREE {
        // SAFETY: the parse entry point detected AVX-512F and POPCNT before choosing SIMD.
        return unsafe { parse_batch_avx512(matcher, bytes, offsets, buffers, codes, row_offsets) };
    }
    parse_batch_with::<I, O, SCALAR, LANES>(matcher, bytes, offsets, buffers, codes, row_offsets);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,popcnt")]
unsafe fn parse_batch_avx512<I: OffsetLike, O: Offset>(
    matcher: &DictionaryMatcher,
    bytes: &[u8],
    offsets: &[I],
    buffers: &mut LaneBuffers,
    codes: &mut Vec<Token>,
    row_offsets: &mut Vec<O>,
) {
    parse_batch_with::<I, O, TREE, TREE_LANES>(
        matcher,
        bytes,
        offsets,
        buffers,
        codes,
        row_offsets,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,popcnt")]
unsafe fn parse_batch_rank<I: OffsetLike, O: Offset>(
    matcher: &DictionaryMatcher,
    bytes: &[u8],
    offsets: &[I],
    buffers: &mut LaneBuffers,
    codes: &mut Vec<Token>,
    row_offsets: &mut Vec<O>,
) {
    parse_batch_with::<I, O, RANK, RANK_LANES>(
        matcher,
        bytes,
        offsets,
        buffers,
        codes,
        row_offsets,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,popcnt")]
unsafe fn parse_batch_rank_rows<I: OffsetLike, O: Offset>(
    matcher: &DictionaryMatcher,
    bytes: &[u8],
    offsets: &[I],
    buffers: &mut LaneBuffers,
    codes: &mut Vec<Token>,
    row_offsets: &mut Vec<O>,
) {
    parse_batch_with::<I, O, RANK_ROWS, RANK_LANES>(
        matcher,
        bytes,
        offsets,
        buffers,
        codes,
        row_offsets,
    );
}

#[inline(always)]
fn parse_batch_with<I: OffsetLike, O: Offset, const MODE: u8, const K: usize>(
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
    let mut bounds = [0usize; MAX_LANES + 1];
    for (j, bound) in bounds.iter_mut().enumerate().take(K + 1).skip(1) {
        let target = first + total * j / K;
        *bound = offsets.partition_point(|o| o.to_usize() < target).min(n);
    }
    bounds[K] = n;

    // Per lane: position, end of the current row, current row, end row, and
    // number of codes written.
    let mut at = [0usize; K];
    let mut row_end = [0usize; K];
    let mut row = [0usize; K];
    let mut last_row = [0usize; K];
    let mut count = [0usize; K];

    for j in 0..K {
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
        // Existing entries are initialized and overwritten before being copied
        // out, so only initialize newly grown storage between batches.
        buffers.codes[j].resize(lane_bytes + 1, 0);
        buffers.row_ends[j].resize(r1 - r0 + 1, 0);
    }

    loop {
        // An inactive lane still runs one search; its result is discarded.
        let win: [usize; K] =
            std::array::from_fn(|j| (row_end[j] - at[j]).clamp(1, MAX_TOKEN_SIZE));
        #[cfg(target_arch = "x86_64")]
        let found = if MODE == RANK || MODE == RANK_ROWS {
            // SAFETY: the batch wrapper enables AVX-512F and POPCNT.
            unsafe { matcher.find_all_rank(bytes, &at, &win) }
        } else if MODE == TREE {
            // SAFETY: the batch wrapper enables AVX-512F and POPCNT.
            unsafe { matcher.find_all_avx512(bytes, &at, &win) }
        } else {
            matcher.find_all(bytes, &at, &win)
        };
        #[cfg(not(target_arch = "x86_64"))]
        let found = matcher.find_all(bytes, &at, &win);

        let mut busy = false;
        for j in 0..K {
            let running = row[j] < last_row[j];
            busy |= running;

            let active = row_end[j] > at[j];
            let (token, len) = found[j];
            buffers.codes[j][count[j]] = token;
            count[j] += active as usize;
            at[j] += select_unpredictable(active, len, 0);

            #[cfg(target_arch = "x86_64")]
            if MODE == RANK_ROWS {
                if at[j] == row_end[j] && running {
                    buffers.row_ends[j][row[j] - bounds[j]] = count[j];
                    row[j] += 1;
                    let next_end = offsets[(row[j] + 1).min(n)].to_usize();
                    row_end[j] = select_unpredictable(row[j] < last_row[j], next_end, at[j]);
                }
                continue;
            }

            // The slot of the current row holds its code count so far, so it
            // is final when the row ends. A lane past its last row writes the
            // spare slot.
            buffers.row_ends[j][row[j] - bounds[j]] = count[j];

            // A row ends when its bytes are used up. The next row starts where
            // this one ends.
            let done = at[j] == row_end[j] && running;
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

    for j in 0..K {
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

    #[hegel::test(test_cases = 100)]
    fn reused_buffers_match_scalar_across_row_lengths(tc: TestCase) {
        let bytes = || gs::vecs(gs::sampled_from(&PROPERTY_ALPHABET[..]));
        let mut extra: Vec<Vec<u8>> =
            tc.draw(gs::vecs(bytes().min_size(2).max_size(MAX_TOKEN_SIZE)).max_size(100));
        extra.extend([vec![0; 16], vec![255; 16], b"a\0".to_vec()]);
        let extra: Vec<&[u8]> = extra.iter().map(Vec::as_slice).collect();
        let dict = complete_dictionary(&extra);
        let matcher = DictionaryMatcher::from_dictionary(dict.as_view());

        // Even 16-byte matches leave at least 32 codes per row in the probe.
        let byte = tc.draw(gs::sampled_from(&PROPERTY_ALPHABET[..]));
        let mut rows = vec![vec![byte; 512]; 64];
        rows.extend(tc.draw(gs::vecs(bytes().max_size(40)).max_size(40)));
        rows.extend([vec![], vec![255; 17], vec![0; 1], vec![]]);
        let rows: Vec<&[u8]> = rows.iter().map(Vec::as_slice).collect();
        let expected = reference(&matcher, &rows);
        let mut buffers = LaneBuffers::default();
        check_reused_buffers::<SCALAR>(&matcher, &rows, &expected, &mut buffers);
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("popcnt")
        {
            check_reused_buffers::<RANK>(&matcher, &rows, &expected, &mut buffers);
            check_reused_buffers::<RANK_ROWS>(&matcher, &rows, &expected, &mut buffers);
        }
    }

    fn check_reused_buffers<const MODE: u8>(
        matcher: &DictionaryMatcher,
        rows: &[&[u8]],
        expected: &(Vec<Token>, Vec<u64>),
        buffers: &mut LaneBuffers,
    ) {
        let mut codes = Vec::new();
        let mut ends = vec![0u64];
        let raw = make_raw(rows);
        parse_arrow_batched_with::<_, u64, MODE>(
            matcher,
            &raw.data,
            &raw.offsets,
            32 << 10,
            buffers,
            &mut codes,
            &mut ends,
        );
        assert_eq!((&codes, &ends), (&expected.0, &expected.1));
        codes.clear();
        ends.clear();
        ends.push(0);
        parse_rows_batched_with::<_, u64, MODE>(
            matcher,
            rows,
            32 << 10,
            buffers,
            &mut codes,
            &mut ends,
        );
        assert_eq!(&(codes, ends), expected);
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
    fn single_byte_dictionary_appends_with_empty_rows_and_nonzero_offset() {
        let dict = complete_dictionary(&[]);
        let matcher = DictionaryMatcher::from_dictionary(dict.as_view());
        let bytes = [99, b'a', 0, 255, 88];
        let offsets = [1u32, 1, 4, 4];
        let rows: [&[u8]; 3] = [&[], &bytes[1..4], &[]];
        for arrow in [false, true] {
            let mut codes = vec![42];
            let mut ends = vec![0u64, 1];
            let mut buffers = LaneBuffers::default();
            if arrow {
                parse_arrow(
                    &matcher,
                    &bytes,
                    &offsets,
                    &mut buffers,
                    &mut codes,
                    &mut ends,
                );
            } else {
                parse_rows(
                    &matcher,
                    rows.as_slice(),
                    &mut buffers,
                    &mut codes,
                    &mut ends,
                );
            }
            assert_eq!(codes, [42, 97, 0, 255]);
            assert_eq!(ends, [0, 1, 1, 4, 4]);
            assert!(buffers.bytes.is_empty());
        }
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
