// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The tokens an occurrence can begin inside: the alignment edges of the graph.
//!
//! Such a token either ends with a needle prefix of `k` bytes, an *overlap* giving
//! the alignment edge `0 -> k`, or contains the whole needle. One pass over the
//! dictionary's concatenated token bytes finds both.
//!
//! Overlap lists are capped, more tightly for one-byte overlaps, which are common.
//! Past the cap only the overlap's existence is kept: the edge becomes unenumerated,
//! and the verifier compares token suffixes instead.

use memchr::memmem::Finder;

use super::graph::EdgeKind;
use crate::core::dictionary::CompactDictionaryView;
use crate::core::types::{MAX_TOKEN_SIZE, Token};

/// Maximum number of tokens listed for one overlap.
pub(in crate::search::substring::probe) const MAX_ENUMERATED_TOKENS: usize = 512;

/// Maximum number of tokens listed for a one-byte overlap.
const MAX_SINGLE_BYTE_TOKENS: usize = 16;

/// Count marking an overlap whose tokens exceeded the cap.
const UNENUMERATED: usize = MAX_ENUMERATED_TOKENS + 1;

/// The tokens an occurrence of one needle can begin inside.
pub(super) struct MatchStarts {
    /// For each overlap length, the number of listed tokens or `UNENUMERATED`.
    counts: [usize; MAX_TOKEN_SIZE],
    /// Listed tokens, by overlap length.
    ids: [[Token; MAX_ENUMERATED_TOKENS]; MAX_TOKEN_SIZE],
    /// Tokens that contain the whole needle but do not start with it, in ID order.
    /// Tokens starting with the needle are on the range edge from the source instead.
    pub(super) contained: Vec<Token>,
}

impl MatchStarts {
    /// Finds the match starts of a nonempty needle.
    #[inline]
    pub(super) fn new(dict: CompactDictionaryView<'_>, needle: &[u8]) -> Self {
        let mut starts = Self {
            counts: [0; MAX_TOKEN_SIZE],
            ids: [[0; MAX_ENUMERATED_TOKENS]; MAX_TOKEN_SIZE],
            contained: Vec::new(),
        };
        let (payload, offsets) = dict.token_payload();

        if needle.len() > 1 {
            starts.collect_single_byte_overlaps(payload, offsets, needle[0]);
        }
        starts.scan_payload(payload, offsets, needle);
        starts
    }

    /// The alignment edge for an overlap of `length` bytes, or `None` if no token has one.
    pub(super) fn overlap(&self, length: usize) -> Option<EdgeKind> {
        let &count = self.counts.get(length)?;
        if count == 0 {
            None
        } else if count > MAX_ENUMERATED_TOKENS {
            Some(EdgeKind::UnenumeratedSet)
        } else {
            Some(EdgeKind::Set(self.ids[length][..count].into()))
        }
    }

    /// Lists a token, or marks the overlap unenumerated once it exceeds `limit`.
    fn record_overlap(&mut self, overlap_len: usize, token: Token, limit: usize) {
        let count = &mut self.counts[overlap_len];
        if *count < limit {
            self.ids[overlap_len][*count] = token;
            *count += 1;
        } else {
            *count = UNENUMERATED;
        }
    }

    /// Whether the overlap exceeded its cap.
    fn is_unenumerated(&self, overlap_len: usize) -> bool {
        self.counts[overlap_len] == UNENUMERATED
    }

    /// Lists tokens ending with the needle's first byte. One-byte tokens are skipped:
    /// an occurrence starting with one begins at a token boundary.
    fn collect_single_byte_overlaps(&mut self, payload: &[u8], offsets: &[u32], byte: u8) {
        for (token_id, bounds) in offsets.windows(2).enumerate() {
            let start = bounds[0] as usize;
            let end = bounds[1] as usize;
            if end - start > 1 && payload[end - 1] == byte {
                self.record_overlap(1, token_id as Token, MAX_SINGLE_BYTE_TOKENS);
                if self.is_unenumerated(1) {
                    break;
                }
            }
        }
    }

    /// Finds longer overlaps and contained needles in one pass over the payload,
    /// searching for the needle's first two bytes (or its only byte).
    fn scan_payload(&mut self, payload: &[u8], offsets: &[u32], needle: &[u8]) {
        let prefix_len = needle.len().min(2);
        let finder = Finder::new(&needle[..prefix_len]);
        let mut cursor = 0;
        let mut token_id = 0;

        while let Some(relative_hit) = finder.find(&payload[cursor..]) {
            let hit = cursor + relative_hit;
            // The next hit may overlap this one.
            cursor = hit + 1;

            // Hits arrive in order, so the token cursor only moves forward.
            while offsets[token_id + 1] as usize <= hit {
                token_id += 1;
            }
            let token_start = offsets[token_id] as usize;
            let token_end = offsets[token_id + 1] as usize;
            let tail = &payload[hit..token_end];

            // Searching the concatenated payload can cross a token boundary.
            if tail.len() < prefix_len {
                continue;
            }
            if tail.starts_with(needle) {
                if hit != token_start {
                    self.contained.push(token_id as Token);
                }
                // This token's whole matches are accounted for. Only its last
                // needle.len() - 1 bytes can still start a partial overlap.
                cursor = token_end - (needle.len() - 1);
            } else if hit != token_start && tail.len() < needle.len() && needle.starts_with(tail) {
                self.record_overlap(tail.len(), token_id as Token, MAX_ENUMERATED_TOKENS);
            }
        }
    }
}

#[cfg(test)]
pub(in crate::search::substring::probe) mod tests {
    use super::*;
    use crate::core::dictionary::{CompactDictionary, Dictionary, DictionaryView, pad_raw};
    use crate::test_corpus::make_raw;

    fn dictionary(mut tokens: Vec<Vec<u8>>) -> CompactDictionary {
        tokens.sort();
        tokens.dedup();
        let mut raw = make_raw(&tokens);
        pad_raw(&mut raw.data, &raw.offsets);
        CompactDictionary::from_raw(raw.data, raw.offsets)
    }

    /// Compare the collected starts against independent per-token searches.
    pub(in crate::search::substring::probe) fn check_starts(
        dict: CompactDictionaryView<'_>,
        needle: &[u8],
    ) {
        let tokens: Vec<_> = (0..dict.num_tokens())
            .map(|id| (id as Token, dict.token(id as Token)))
            .collect();
        let starts = MatchStarts::new(dict, needle);
        let contained: Vec<_> = tokens
            .iter()
            .filter(|(_, token)| {
                !token.starts_with(needle)
                    && token.windows(needle.len()).any(|window| window == needle)
            })
            .map(|&(id, _)| id)
            .collect();
        assert_eq!(starts.contained, contained, "contained: {needle:?}");

        for length in 1..needle.len().min(MAX_TOKEN_SIZE) {
            let expected: Vec<_> = tokens
                .iter()
                .filter(|(_, token)| token.len() > length && token.ends_with(&needle[..length]))
                .map(|&(id, _)| id)
                .collect();
            let limit = if length == 1 {
                MAX_SINGLE_BYTE_TOKENS
            } else {
                MAX_ENUMERATED_TOKENS
            };
            match starts.overlap(length) {
                None => assert!(expected.is_empty(), "overlap {length}: {needle:?}"),
                Some(EdgeKind::UnenumeratedSet) => {
                    assert!(expected.len() > limit, "overlap {length}: {needle:?}");
                }
                Some(EdgeKind::Set(actual)) => {
                    assert!(!expected.is_empty() && expected.len() <= limit);
                    assert_eq!(&*actual, expected, "overlap {length}: {needle:?}");
                }
                Some(other) => panic!("unexpected overlap edge: {other:?}"),
            }
        }
    }

    #[test]
    fn enumeration_limits_preserve_overlap_presence() {
        for (suffix, limit) in [
            (b"a".as_slice(), MAX_SINGLE_BYTE_TOKENS),
            (b"ab".as_slice(), MAX_ENUMERATED_TOKENS),
        ] {
            for count in [0, limit - 1, limit, limit + 1, limit * 3] {
                let tokens = (0..count)
                    .map(|i| [vec![128 + (i / 256) as u8, i as u8], suffix.to_vec()].concat())
                    .collect();
                check_starts(dictionary(tokens).as_view(), b"abc");
            }
        }
    }

    #[test]
    fn contained_matches_are_uncapped_and_unique() {
        let mut tokens: Vec<_> = (0..=MAX_ENUMERATED_TOKENS)
            .map(|i| [vec![128 + (i / 256) as u8, i as u8], b"ababa".to_vec()].concat())
            .collect();
        // A token starting with the needle is covered by the range edge,
        // even when it contains another occurrence later.
        tokens.push(b"ababa".to_vec());
        let dict = dictionary(tokens);
        check_starts(dict.as_view(), b"aba");
        assert_eq!(
            MatchStarts::new(dict.as_view(), b"aba").contained.len(),
            MAX_ENUMERATED_TOKENS + 1
        );
    }

    #[test]
    fn payload_boundary_hit_does_not_skip_an_overlapping_match() {
        // Payload a|baba has "aba" at offsets 0 and 2. Reject the cross-token
        // hit at 0 and retain the contained hit at 2.
        let mut raw = make_raw(&[b"a".as_slice(), b"baba"]);
        pad_raw(&mut raw.data, &raw.offsets);
        let dict = CompactDictionaryView::validate_safety(&raw.data, &raw.offsets).unwrap();
        assert_eq!(MatchStarts::new(dict, b"aba").contained, [1]);
        for pattern in [b"a".as_slice(), b"ba", b"aba"] {
            check_starts(dict, pattern);
        }
    }
}
