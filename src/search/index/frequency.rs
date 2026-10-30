// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A prefix-sum index of token frequencies over a code stream.

use std::fmt;

use crate::core::types::{Token, TokenRange};

/// Most tokens an index can count: one per [`Token`] value.
const MAX_TOKENS: usize = Token::MAX as usize + 1;

/// How often each token occurs in a code stream, as prefix sums: entry `i` counts
/// the codes below token `i`, and the last entry counts them all.
///
/// [`ContainsScan`](crate::search::ContainsScan) plans with it, preferring covers of
/// rare tokens. The index records counts, not the dictionary or stream they came
/// from, and a scan checks only that it counts as many tokens as the dictionary
/// has. Counts that do not match the stream make scans slower, never wrong.
///
/// Build an index with [`build`](Self::build), store its prefix sums, and borrow
/// them back with [`TokenFrequencyIndexView::new`]:
///
/// ```
/// use onpair::TokenRange;
/// use onpair::search::index::{TokenFrequencyIndex, TokenFrequencyIndexView};
///
/// let index = TokenFrequencyIndex::build(&[2, 0, 2, 3, 2, 0], 5)?;
/// assert_eq!(index.as_view().frequency(2), 3);
///
/// let stored: Vec<u32> = index.into_cumulative();
/// let view = TokenFrequencyIndexView::new(&stored)?;
/// assert_eq!(view.range_frequency(TokenRange::new(1, 3)), 4);
/// assert_eq!(view.total_frequency(), 6);
/// # Ok::<(), onpair::search::index::InvalidFrequencyIndex>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenFrequencyIndex {
    cumulative: Vec<u32>,
}

impl TokenFrequencyIndex {
    /// Counts `codes` over a dictionary of `num_tokens` tokens.
    ///
    /// # Errors
    /// Returns [`InvalidFrequencyIndex::TooManyTokens`] if `num_tokens` exceeds
    /// 2^16, [`InvalidFrequencyIndex::FrequencyOverflow`] if there are more than
    /// `u32::MAX` codes, and [`InvalidFrequencyIndex::CodeOutOfRange`] if a code is
    /// not below `num_tokens`.
    pub fn build(codes: &[Token], num_tokens: usize) -> Result<Self, InvalidFrequencyIndex> {
        if num_tokens > MAX_TOKENS {
            return Err(InvalidFrequencyIndex::TooManyTokens);
        }
        if u32::try_from(codes.len()).is_err() {
            return Err(InvalidFrequencyIndex::FrequencyOverflow);
        }
        let mut cumulative = vec![0u32; num_tokens + 1];
        for &code in codes {
            *cumulative
                .get_mut(usize::from(code) + 1)
                .ok_or(InvalidFrequencyIndex::CodeOutOfRange)? += 1;
        }
        for token in 0..num_tokens {
            cumulative[token + 1] += cumulative[token];
        }
        Ok(Self { cumulative })
    }

    /// Adopts stored prefix sums without copying them.
    ///
    /// # Errors
    /// As [`TokenFrequencyIndexView::new`].
    pub fn from_cumulative(cumulative: Vec<u32>) -> Result<Self, InvalidFrequencyIndex> {
        check_cumulative(&cumulative)?;
        Ok(Self { cumulative })
    }

    /// The prefix sums, for storage, without copying them.
    pub fn into_cumulative(self) -> Vec<u32> {
        self.cumulative
    }

    /// Borrows the index.
    #[inline]
    pub fn as_view(&self) -> TokenFrequencyIndexView<'_> {
        TokenFrequencyIndexView {
            cumulative: &self.cumulative,
        }
    }
}

/// A borrowed [`TokenFrequencyIndex`], for example over stored prefix sums.
///
/// The prefix sums are checked when the view is made, so its queries never
/// underflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenFrequencyIndexView<'a> {
    cumulative: &'a [u32],
}

impl<'a> TokenFrequencyIndexView<'a> {
    /// Borrows stored prefix sums without copying them.
    ///
    /// # Errors
    /// Returns [`InvalidFrequencyIndex::Empty`] if `cumulative` is empty,
    /// [`InvalidFrequencyIndex::TooManyTokens`] if it counts more than 2^16 tokens,
    /// [`InvalidFrequencyIndex::FirstNotZero`] if it does not start at zero, and
    /// [`InvalidFrequencyIndex::Decreasing`] if it decreases.
    pub fn new(cumulative: &'a [u32]) -> Result<Self, InvalidFrequencyIndex> {
        check_cumulative(cumulative)?;
        Ok(Self { cumulative })
    }

    /// The prefix sums: one entry per token, then the total.
    #[inline]
    pub fn cumulative(&self) -> &'a [u32] {
        self.cumulative
    }

    /// Number of tokens counted.
    #[inline]
    pub fn num_tokens(&self) -> usize {
        self.cumulative.len() - 1
    }

    /// How often `token` occurs.
    ///
    /// # Panics
    /// Panics if `token` is not below [`num_tokens`](Self::num_tokens).
    #[inline]
    pub fn frequency(&self, token: Token) -> u32 {
        let token = usize::from(token);
        assert!(token < self.num_tokens(), "token outside frequency index");
        self.cumulative[token + 1] - self.cumulative[token]
    }

    /// How often the tokens of the inclusive `range` occur together; zero for the
    /// empty range.
    ///
    /// # Panics
    /// Panics if a nonempty `range` ends at or past
    /// [`num_tokens`](Self::num_tokens).
    #[inline]
    pub fn range_frequency(&self, range: TokenRange) -> u32 {
        if range.is_empty() {
            return 0;
        }
        let begin = usize::from(range.begin);
        let last = usize::from(range.last);
        assert!(last < self.num_tokens(), "range outside frequency index");
        self.cumulative[last + 1] - self.cumulative[begin]
    }

    /// Number of codes counted.
    #[inline]
    pub fn total_frequency(&self) -> u32 {
        self.cumulative[self.cumulative.len() - 1]
    }
}

/// Why a token frequency index could not be built or borrowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InvalidFrequencyIndex {
    /// The prefix sums are empty: they need one entry more than the tokens they
    /// count.
    Empty,
    /// The index would count more than 2^16 tokens, more than [`Token`] can name.
    TooManyTokens,
    /// The prefix sums do not start at zero.
    FirstNotZero,
    /// The prefix sums decrease.
    Decreasing,
    /// A code is not below the token count.
    CodeOutOfRange,
    /// There are more than `u32::MAX` codes to count.
    FrequencyOverflow,
}

impl fmt::Display for InvalidFrequencyIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Empty => "frequency index is empty",
            Self::TooManyTokens => "frequency index counts more than 2^16 tokens",
            Self::FirstNotZero => "frequency index must start at zero",
            Self::Decreasing => "frequency index must be non-decreasing",
            Self::CodeOutOfRange => "code is outside the frequency index's tokens",
            Self::FrequencyOverflow => "code stream is too long for a u32 frequency index",
        })
    }
}

impl std::error::Error for InvalidFrequencyIndex {}

/// Checks that `cumulative` is a prefix sum over at most [`MAX_TOKENS`] tokens.
fn check_cumulative(cumulative: &[u32]) -> Result<(), InvalidFrequencyIndex> {
    let Some(&first) = cumulative.first() else {
        return Err(InvalidFrequencyIndex::Empty);
    };
    if cumulative.len() > MAX_TOKENS + 1 {
        return Err(InvalidFrequencyIndex::TooManyTokens);
    }
    if first != 0 {
        return Err(InvalidFrequencyIndex::FirstNotZero);
    }
    if cumulative.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(InvalidFrequencyIndex::Decreasing);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_counts_codes_and_rejects_bad_input() {
        let index = TokenFrequencyIndex::build(&[2, 0, 2, 3, 2, 0], 5).unwrap();
        let view = index.as_view();
        assert_eq!(view.num_tokens(), 5);
        assert_eq!(view.total_frequency(), 6);
        assert_eq!(
            (0..5)
                .map(|token| view.frequency(token))
                .collect::<Vec<_>>(),
            [2, 0, 3, 1, 0]
        );
        assert_eq!(view.range_frequency(TokenRange::new(1, 3)), 4);
        assert_eq!(view.range_frequency(TokenRange::EMPTY), 0);

        assert_eq!(
            TokenFrequencyIndex::build(&[0, 2], 2),
            Err(InvalidFrequencyIndex::CodeOutOfRange)
        );
        assert_eq!(
            TokenFrequencyIndex::build(&[], MAX_TOKENS + 1),
            Err(InvalidFrequencyIndex::TooManyTokens)
        );
        assert!(TokenFrequencyIndex::build(&[Token::MAX], MAX_TOKENS).is_ok());
    }

    #[test]
    fn stored_prefix_sums_are_checked_and_never_copied() {
        let index = TokenFrequencyIndex::build(&[0, 0, 2, 2, 2, 3], 4).unwrap();
        let stored = index.clone().into_cumulative();
        let view = TokenFrequencyIndexView::new(&stored).unwrap();
        assert_eq!(view, index.as_view());
        assert_eq!(view.cumulative().as_ptr(), stored.as_ptr());

        let pointer = stored.as_ptr();
        let adopted = TokenFrequencyIndex::from_cumulative(stored).unwrap();
        assert_eq!(adopted, index);
        let returned = adopted.into_cumulative();
        assert_eq!(returned.as_ptr(), pointer);
    }

    #[test]
    fn malformed_prefix_sums_are_rejected() {
        let too_many = vec![0; MAX_TOKENS + 2];
        let cases: &[(&[u32], InvalidFrequencyIndex)] = &[
            (&[], InvalidFrequencyIndex::Empty),
            (&too_many, InvalidFrequencyIndex::TooManyTokens),
            (&[1, 1], InvalidFrequencyIndex::FirstNotZero),
            (&[0, 2, 1], InvalidFrequencyIndex::Decreasing),
        ];
        for &(cumulative, error) in cases {
            assert_eq!(TokenFrequencyIndexView::new(cumulative), Err(error));
            assert_eq!(
                TokenFrequencyIndex::from_cumulative(cumulative.to_vec()),
                Err(error)
            );
        }
        assert!(TokenFrequencyIndexView::new(&[0]).is_ok());
        assert!(TokenFrequencyIndexView::new(&vec![0; MAX_TOKENS + 1]).is_ok());
    }
}
