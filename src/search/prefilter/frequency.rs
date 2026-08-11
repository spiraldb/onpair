// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The prefilter's selectivity signal: a prefix-sum index of token frequencies
//! over a code stream.

use crate::core::types::{Token, TokenRange};

/// Prefix-sum index of token frequencies in a code stream.
///
/// Entry `i` stores the number of codes whose token id is less than `i`. The
/// representation is private so callers cannot accidentally pass arbitrary
/// `u32` data to the prefilter; construct an index with
/// [`build_token_frequency_index`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenFrequencyIndex {
    cumulative: Box<[u32]>,
}

impl TokenFrequencyIndex {
    /// Number of token ids covered by this index.
    #[inline]
    pub fn num_tokens(&self) -> usize {
        self.cumulative.len() - 1
    }

    /// Frequency of `token` in the indexed code stream.
    ///
    /// # Panics
    /// Panics if `token` is outside the indexed token-id domain.
    #[inline]
    pub fn frequency(&self, token: Token) -> u32 {
        let token = token as usize;
        assert!(token < self.num_tokens(), "token outside frequency index");
        self.cumulative[token + 1] - self.cumulative[token]
    }

    /// Summed frequency of the inclusive token-id `range`.
    ///
    /// The empty range has frequency zero.
    ///
    /// # Panics
    /// Panics if a non-empty range extends outside the indexed token-id domain.
    #[inline]
    pub fn range_frequency(&self, range: TokenRange) -> u32 {
        if range.is_empty() {
            return 0;
        }
        let begin = range.begin as usize;
        let last = range.last as usize;
        assert!(last < self.num_tokens(), "range outside frequency index");
        self.cumulative[last + 1] - self.cumulative[begin]
    }

    /// Total number of codes represented by the index.
    #[inline]
    pub(super) fn total_frequency(&self) -> u32 {
        self.cumulative[self.cumulative.len() - 1]
    }
}

/// Failure to construct a [`TokenFrequencyIndex`].
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum TokenFrequencyIndexError {
    /// The code stream contains more than `u32::MAX` token occurrences.
    FrequencyOverflow,
    /// `num_tokens` exceeds the token-id domain representable by [`Token`].
    TooManyTokens,
    /// A code is not in `0..num_tokens`.
    CodeOutOfRange,
}

impl std::fmt::Display for TokenFrequencyIndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::FrequencyOverflow => "code stream is too large for a u32 frequency index",
            Self::TooManyTokens => "token count exceeds the Token id domain",
            Self::CodeOutOfRange => "code index is outside the frequency index token domain",
        })
    }
}

impl std::error::Error for TokenFrequencyIndexError {}

/// Build the prefix-sum frequency index for `codes` over `num_tokens` token ids.
///
/// The resulting index contains `num_tokens + 1` `u32` entries and represents
/// code streams of up to `u32::MAX` token occurrences.
///
/// # Errors
/// Returns [`TokenFrequencyIndexError::FrequencyOverflow`] if the code count
/// cannot be represented by `u32`,
/// [`TokenFrequencyIndexError::TooManyTokens`] if `num_tokens` exceeds the
/// [`Token`] id domain, or [`TokenFrequencyIndexError::CodeOutOfRange`] if any
/// code is not less than `num_tokens`.
pub fn build_token_frequency_index(
    codes: &[Token],
    num_tokens: usize,
) -> Result<TokenFrequencyIndex, TokenFrequencyIndexError> {
    checked_frequency_total(codes.len())?;
    if num_tokens > Token::MAX as usize + 1 {
        return Err(TokenFrequencyIndexError::TooManyTokens);
    }

    let mut cumulative = vec![0u32; num_tokens + 1];
    for &code in codes {
        let count = cumulative
            .get_mut(code as usize + 1)
            .ok_or(TokenFrequencyIndexError::CodeOutOfRange)?;
        *count += 1;
    }
    for token in 0..num_tokens {
        cumulative[token + 1] += cumulative[token];
    }
    debug_assert_eq!(cumulative[num_tokens], codes.len() as u32);

    Ok(TokenFrequencyIndex {
        cumulative: cumulative.into_boxed_slice(),
    })
}

#[inline]
pub(super) fn checked_frequency_total(len: usize) -> Result<u32, TokenFrequencyIndexError> {
    u32::try_from(len).map_err(|_| TokenFrequencyIndexError::FrequencyOverflow)
}
