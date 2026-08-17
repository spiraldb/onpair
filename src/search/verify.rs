// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Exact substring verification in the decoded domain.
//!
//! [`prefilter_candidates`](super::prefilter_candidates) returns a superset, and
//! something has to check it. One option stays compressed:
//! [`contains`](super::contains()) steps a token-level KMP automaton over a row's
//! codes and never decodes. This module is the other one — decode the row, then
//! `memchr::memmem` its bytes.
//!
//! Decoding sounds like the more expensive of the two and measures as the
//! cheaper. A row's decode is a gather-copy of a few 16-byte stores, after which
//! `memmem` runs a vectorized skip loop over contiguous bytes; the KMP walk is a
//! dependent load per token through a table indexed by the dictionary, which is
//! neither vectorizable nor cache-resident once the dictionary is large. It also
//! drops the 255-byte pattern cap, which is an artifact of KMP's `u8` states
//! rather than anything the search itself needs.
//!
//! # Allocation
//! The decode buffer is the only allocation, and it belongs to the verifier, not
//! to a call: it grows to fit the row in front of it and never shrinks, so a
//! verifier reused across queries stops allocating once it has met its largest
//! row. Growth goes through [`Vec::reserve`]'s doubling, which bounds the total
//! copying by the final capacity however the row sizes happen to arrive.
//!
//! Sizing is the free upper bound — [`MAX_TOKEN_SIZE`] per code plus
//! [`DECODE_PADDING`], read straight off the row layer — rather than the exact
//! [`row_decoded_len`](ColumnView::row_decoded_len), which would walk every token
//! of the row to size the buffer and then walk them again to fill it. On a column
//! of short strings the overshoot it would recover is a few kilobytes, once.

use std::mem::MaybeUninit;

use memchr::memmem::Finder;

use crate::ColumnView;
use crate::core::offset::Offset;
use crate::core::types::MAX_TOKEN_SIZE;
use crate::decoding::DECODE_PADDING;

/// A pattern prepared for decoded-domain verification, plus the buffer its rows
/// decode through.
///
/// Build one per pattern and keep it for as long as that pattern is being
/// queried: the buffer is the reusable part, and a fresh verifier per call
/// throws it away.
///
/// ```
/// use onpair::search::{
///     BytesVerifier, analyze_prefilter, build_token_frequency_index, prefilter_candidates,
/// };
/// # use onpair::{Column, DictionaryView, DEFAULT_CONFIG};
/// # let rows: &[&[u8]] = &[b"alpha", b"beta", b"alphabet"];
/// # let mut bytes = Vec::new();
/// # let mut offsets = vec![0u32];
/// # for r in rows { bytes.extend_from_slice(r); offsets.push(bytes.len() as u32); }
/// # let col = Column::compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap();
/// let view = col.view();
/// let freqs = build_token_frequency_index(view.codes, view.dict.num_tokens()).unwrap();
///
/// let mut verifier = BytesVerifier::new(b"pha");
/// let mut rows = Vec::new();
/// let analysis = analyze_prefilter(b"pha", view.dict, &freqs);
/// prefilter_candidates(view.codes, view.row_offsets, &analysis, &mut rows)?;
/// verifier.retain(view, &mut rows);
/// assert_eq!(rows, vec![0, 2]);
/// # Ok::<(), onpair::search::PrefilterError>(())
/// ```
pub struct BytesVerifier<'p> {
    /// The prepared needle. Borrows the pattern rather than copying it.
    finder: Finder<'p>,
    /// Decode scratch, reused across rows and across calls. Its length is the
    /// capacity the rows seen so far demanded, never less.
    buf: Vec<MaybeUninit<u8>>,
}

impl std::fmt::Debug for BytesVerifier<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BytesVerifier")
            .field("pattern_len", &self.finder.needle().len())
            .field("buffer_len", &self.buf.len())
            .finish()
    }
}

impl<'p> BytesVerifier<'p> {
    /// Prepare `pattern`. Borrows it, and allocates no decode buffer until the
    /// first row asks for one.
    ///
    /// Unlike [`ContainsTable`](super::ContainsTable) this has no length limit,
    /// and its cost does not scale with the dictionary.
    pub fn new(pattern: &'p [u8]) -> Self {
        Self {
            finder: Finder::new(pattern),
            buf: Vec::new(),
        }
    }

    /// Drop from `rows` every row that does not contain the pattern, keeping the
    /// rest in place and in order.
    ///
    /// Intended for the candidate list [`prefilter_candidates`](super::prefilter_candidates)
    /// appended to: filtering happens in place, so verification adds no allocation
    /// of its own beyond the decode buffer.
    ///
    /// # Panics
    /// If any entry of `rows` is not a valid row index for `view`, or `view`'s row
    /// layer is malformed ([`InvalidColumn`](crate::InvalidColumn)).
    pub fn retain<O: Offset>(&mut self, view: ColumnView<'_, O>, rows: &mut Vec<usize>) {
        rows.retain(|&row| self.contains_row(view, row));
    }

    /// Whether row `k` of `view` contains the pattern, decoding it into the
    /// reused buffer. Precondition: `k < view.num_rows()`.
    ///
    /// The per-row form of [`retain`](Self::retain), for callers holding their
    /// candidates as something other than a `Vec<usize>`.
    ///
    /// # Panics
    /// With [`InvalidColumn`](crate::InvalidColumn) on a malformed row layer or an
    /// out-of-range code — never UB.
    pub fn contains_row<O: Offset>(&mut self, view: ColumnView<'_, O>, k: usize) -> bool {
        // Every token decodes to at most MAX_TOKEN_SIZE bytes, so this bounds the
        // row's decoded length without consulting the dictionary at all.
        let need = MAX_TOKEN_SIZE * view.row_codes(k).len() + DECODE_PADDING;
        if need > self.buf.len() {
            self.buf.resize(need, MaybeUninit::uninit());
        }

        // SAFETY: `need <= self.buf.len()` bounds this row's decoded length plus
        // DECODE_PADDING for the decoder's final over-store.
        let written = unsafe { view.decompress_row_into(k, &mut self.buf) };
        // SAFETY: `decompress_row_into` initialized exactly the first `written`
        // bytes of the buffer, and `MaybeUninit<u8>` has `u8`'s layout.
        let bytes = unsafe { std::slice::from_raw_parts(self.buf.as_ptr().cast::<u8>(), written) };
        self.finder.find(bytes).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Column, DEFAULT_CONFIG};

    fn column(rows: &[&[u8]]) -> Column<u32> {
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for r in rows {
            bytes.extend_from_slice(r);
            offsets.push(bytes.len() as u32);
        }
        crate::compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap()
    }

    #[test]
    fn retains_exactly_the_containing_rows() {
        let col = column(&[b"alpha", b"beta", b"alphabet", b"", b"pha"]);
        let mut rows: Vec<usize> = (0..5).collect();
        BytesVerifier::new(b"pha").retain(col.view(), &mut rows);
        assert_eq!(rows, vec![0, 2, 4]);

        // An empty pattern is contained in every row, empty rows included.
        let mut rows: Vec<usize> = (0..5).collect();
        BytesVerifier::new(b"").retain(col.view(), &mut rows);
        assert_eq!(rows, vec![0, 1, 2, 3, 4]);
    }

    /// The point of the type: the buffer outlives the call. It has to grow for a
    /// row that needs more than the last one did, and must not shrink back for a
    /// shorter row, or every long/short alternation would reallocate.
    #[test]
    fn buffer_grows_to_the_longest_row_and_stays() {
        let long = vec![b'x'; 4096];
        let col = column(&[b"ab", long.as_slice(), b"ab"]);
        let view = col.view();
        let mut verifier = BytesVerifier::new(b"xxx");

        assert!(!verifier.contains_row(view, 0));
        let after_short = verifier.buf.len();

        assert!(verifier.contains_row(view, 1));
        let after_long = verifier.buf.len();
        assert!(
            after_long > after_short,
            "buffer did not grow for a long row"
        );
        assert!(
            after_long >= view.row_decoded_len(1) + DECODE_PADDING,
            "buffer is too small for the row it just decoded"
        );

        assert!(!verifier.contains_row(view, 2));
        assert_eq!(verifier.buf.len(), after_long, "buffer shrank back");
    }

    /// Rows longer than 255 bytes and patterns longer than 255 bytes both work:
    /// the cap belongs to the KMP table, not to the search.
    #[test]
    fn verifies_patterns_over_the_kmp_state_limit() {
        let mut haystack = vec![b'a'; 300];
        haystack.extend_from_slice(b"tail");
        let col = column(&[haystack.as_slice(), b"aaa"]);

        let pattern = vec![b'a'; 300];
        let mut rows = vec![0, 1];
        BytesVerifier::new(&pattern).retain(col.view(), &mut rows);
        assert_eq!(rows, vec![0]);
    }
}
