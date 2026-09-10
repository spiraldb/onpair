// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A prefix-sum index of token frequencies over a code stream.

use crate::core::types::{Token, TokenRange};
use crate::core::validate::{InvalidColumn, InvalidFrequencyIndex};

/// Storage for a token-frequency index's cumulative counts.
///
/// The returned slice must remain stable and immutable while the storage is
/// alive. Storage only supplies bytes; [`TokenFrequencyIndex::validate_safety`]
/// establishes their invariants without copying them.
pub trait TokenFrequencyIndexStorage {
    /// The `num_tokens + 1` prefix sums of token occurrences.
    fn cumulative(&self) -> &[u32];
}

/// The default owned storage used by [`TokenFrequencyIndex`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedTokenFrequencyIndexStorage {
    cumulative: Vec<u32>,
}

impl OwnedTokenFrequencyIndexStorage {
    /// Build owned token-frequency storage from its serialized buffer.
    pub fn new(cumulative: Vec<u32>) -> Self {
        Self { cumulative }
    }

    /// Consume the storage and return its serialized buffer without copying.
    pub fn into_raw(self) -> Vec<u32> {
        self.cumulative
    }
}

impl TokenFrequencyIndexStorage for OwnedTokenFrequencyIndexStorage {
    #[inline]
    fn cumulative(&self) -> &[u32] {
        &self.cumulative
    }
}

/// Prefix-sum index of token frequencies in a code stream.
///
/// Entry `i` stores the number of codes whose token id is less than `i`. A
/// structurally valid representation begins with zero, is nondecreasing, and
/// ends with the indexed code count.
///
/// [`validate_safety`](Self::validate_safety) checks the cumulative layout;
/// [`validate`](Self::validate) also proves exact correspondence with a code
/// stream. Builder-produced indexes are exact. Safety-only indexes are valid
/// advisory prefilter weights but cannot change query correctness.
///
/// Build the usual owned representation with [`build_token_frequency_index`],
/// or validate another storage implementation without copying it:
///
/// ```
/// use onpair::search::index::{
///     TokenFrequencyIndex, TokenFrequencyIndexStorage,
/// };
///
/// struct Borrowed<'a>(&'a [u32]);
/// impl TokenFrequencyIndexStorage for Borrowed<'_> {
///     fn cumulative(&self) -> &[u32] {
///         self.0
///     }
/// }
///
/// let cumulative = [0, 2, 2, 5];
/// let codes = [0, 2, 0, 2, 2];
/// let index = TokenFrequencyIndex::validate(Borrowed(&cumulative), &codes, 3)?;
/// assert_eq!(index.frequency(0), 2);
/// assert_eq!(index.frequency(1), 0);
/// assert_eq!(index.frequency(2), 3);
/// # Ok::<(), onpair::InvalidColumn>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenFrequencyIndex<S = OwnedTokenFrequencyIndexStorage> {
    storage: S,
}

impl<S> TokenFrequencyIndex<S>
where
    S: TokenFrequencyIndexStorage,
{
    /// Validate the cumulative layout for `num_tokens` and `total_codes`,
    /// retaining `storage` without copying. This takes `O(num_tokens)` and does
    /// not prove that individual deltas match a code stream.
    ///
    /// # Errors
    /// Returns [`InvalidColumn`] if the representation is malformed.
    pub fn validate_safety(
        storage: S,
        num_tokens: usize,
        total_codes: usize,
    ) -> Result<Self, InvalidColumn> {
        validate_cumulative_safety(storage.cumulative(), num_tokens, total_codes)?;
        Ok(Self { storage })
    }

    /// Validate the layout and its exact correspondence with `codes` in
    /// `O(codes.len() + num_tokens)` time.
    pub fn validate(storage: S, codes: &[Token], num_tokens: usize) -> Result<Self, InvalidColumn> {
        let index = Self::validate_safety(storage, num_tokens, codes.len())?;
        index.check_correctness(codes)?;
        Ok(index)
    }

    /// Check that this index's cumulative deltas exactly count `codes`.
    pub fn check_correctness(&self, codes: &[Token]) -> Result<(), InvalidColumn> {
        validate_frequency_correctness(self.storage.cumulative(), codes)
    }

    /// Borrow the storage backing this index.
    #[inline]
    pub fn storage(&self) -> &S {
        &self.storage
    }

    /// Consume the index and return its backing storage without copying.
    #[inline]
    pub fn into_storage(self) -> S {
        self.storage
    }

    /// Borrow this index as a storage-independent, zero-copy view.
    #[inline]
    pub fn as_view(&self) -> TokenFrequencyIndexView<'_> {
        TokenFrequencyIndexView {
            cumulative: self.storage.cumulative(),
        }
    }

    /// Number of token ids covered by this index.
    #[inline]
    pub fn num_tokens(&self) -> usize {
        self.as_view().num_tokens()
    }

    /// Frequency of `token` in the indexed code stream.
    ///
    /// # Panics
    /// Panics if `token` is outside the indexed token-id domain.
    #[inline]
    pub fn frequency(&self, token: Token) -> u32 {
        self.as_view().frequency(token)
    }

    /// Summed frequency of the inclusive token-id `range`.
    ///
    /// The empty range has frequency zero.
    ///
    /// # Panics
    /// Panics if a non-empty range extends outside the indexed token-id domain.
    #[inline]
    pub fn range_frequency(&self, range: TokenRange) -> u32 {
        self.as_view().range_frequency(range)
    }

    /// Total number of codes represented by the index.
    #[inline]
    pub(crate) fn total_frequency(&self) -> u32 {
        self.as_view().total_frequency()
    }
}

/// Borrowed, `Copy` view over a validated token-frequency buffer.
#[derive(Copy, Clone, Debug)]
pub struct TokenFrequencyIndexView<'a> {
    cumulative: &'a [u32],
}

impl<'a> TokenFrequencyIndexView<'a> {
    /// Validate a borrowed cumulative buffer without copying it.
    pub fn validate_safety(
        cumulative: &'a [u32],
        num_tokens: usize,
        total_codes: usize,
    ) -> Result<Self, InvalidColumn> {
        validate_cumulative_safety(cumulative, num_tokens, total_codes)?;
        Ok(Self { cumulative })
    }

    /// Validate a borrowed buffer and its exact correspondence with `codes`.
    pub fn validate(
        cumulative: &'a [u32],
        codes: &[Token],
        num_tokens: usize,
    ) -> Result<Self, InvalidColumn> {
        let view = Self::validate_safety(cumulative, num_tokens, codes.len())?;
        view.check_correctness(codes)?;
        Ok(view)
    }

    /// Check that this view's cumulative deltas exactly count `codes`.
    pub fn check_correctness(&self, codes: &[Token]) -> Result<(), InvalidColumn> {
        validate_frequency_correctness(self.cumulative, codes)
    }

    /// The validated cumulative-frequency buffer.
    #[inline]
    pub fn cumulative(&self) -> &'a [u32] {
        self.cumulative
    }

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
    pub(crate) fn total_frequency(&self) -> u32 {
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
/// `codes` may be any code width that fits the [`Token`] id domain — `u8` as
/// well as [`Token`] itself — since the index counts ids rather than bytes.
///
/// ```
/// use onpair::TokenRange;
/// use onpair::search::index::build_token_frequency_index;
///
/// let index = build_token_frequency_index(&[2u16, 0, 2, 3, 2, 0], 5)?;
/// assert_eq!(index.frequency(2), 3);
/// assert_eq!(index.range_frequency(TokenRange { begin: 1, last: 3 }), 4);
/// # Ok::<(), onpair::search::index::TokenFrequencyIndexError>(())
/// ```
///
/// # Errors
/// Returns [`TokenFrequencyIndexError::FrequencyOverflow`] if the code count
/// cannot be represented by `u32`,
/// [`TokenFrequencyIndexError::TooManyTokens`] if `num_tokens` exceeds the
/// [`Token`] id domain, or [`TokenFrequencyIndexError::CodeOutOfRange`] if any
/// code is not less than `num_tokens`.
pub fn build_token_frequency_index<C: Copy + Into<Token>>(
    codes: &[C],
    num_tokens: usize,
) -> Result<TokenFrequencyIndex, TokenFrequencyIndexError> {
    checked_frequency_total(codes.len())?;
    if num_tokens > Token::MAX as usize + 1 {
        return Err(TokenFrequencyIndexError::TooManyTokens);
    }

    let mut cumulative = vec![0u32; num_tokens + 1];
    for &code in codes {
        let count = cumulative
            .get_mut(code.into() as usize + 1)
            .ok_or(TokenFrequencyIndexError::CodeOutOfRange)?;
        *count += 1;
    }
    for token in 0..num_tokens {
        cumulative[token + 1] += cumulative[token];
    }
    debug_assert_eq!(cumulative[num_tokens], codes.len() as u32);

    Ok(TokenFrequencyIndex {
        storage: OwnedTokenFrequencyIndexStorage::new(cumulative),
    })
}

fn validate_cumulative_safety(
    cumulative: &[u32],
    num_tokens: usize,
    total_codes: usize,
) -> Result<(), InvalidColumn> {
    if num_tokens > Token::MAX as usize + 1 {
        return Err(InvalidColumn::CodeOutOfRange);
    }
    let cumulative_len = num_tokens
        .checked_add(1)
        .ok_or(InvalidFrequencyIndex::BadLength)?;
    if cumulative.len() != cumulative_len {
        return Err(InvalidFrequencyIndex::BadLength.into());
    }
    if cumulative[0] != 0 {
        return Err(InvalidFrequencyIndex::FirstCumulativeNotZero.into());
    }
    if cumulative.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(InvalidFrequencyIndex::DecreasingCumulative.into());
    }
    let total_codes =
        u32::try_from(total_codes).map_err(|_| InvalidFrequencyIndex::FrequencyOverflow)?;
    if cumulative[num_tokens] != total_codes {
        return Err(InvalidFrequencyIndex::InvalidTotal.into());
    }
    Ok(())
}

fn validate_frequency_correctness(
    cumulative: &[u32],
    codes: &[Token],
) -> Result<(), InvalidColumn> {
    if u32::try_from(codes.len()).ok() != cumulative.last().copied() {
        return Err(InvalidFrequencyIndex::FrequenciesMismatch.into());
    }

    let num_tokens = cumulative.len() - 1;
    let mut counts = vec![0u32; num_tokens];
    for &code in codes {
        let count = counts
            .get_mut(code as usize)
            .ok_or(InvalidColumn::CodeOutOfRange)?;
        *count += 1;
    }
    if counts
        .iter()
        .zip(cumulative.windows(2))
        .any(|(&count, pair)| count != pair[1] - pair[0])
    {
        return Err(InvalidFrequencyIndex::FrequenciesMismatch.into());
    }
    Ok(())
}

#[inline]
pub(crate) fn checked_frequency_total(len: usize) -> Result<u32, TokenFrequencyIndexError> {
    u32::try_from(len).map_err(|_| TokenFrequencyIndexError::FrequencyOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct ExternalStorage<'a>(&'a [u32]);

    impl TokenFrequencyIndexStorage for ExternalStorage<'_> {
        fn cumulative(&self) -> &[u32] {
            self.0
        }
    }

    fn invalid_index(error: InvalidFrequencyIndex) -> InvalidColumn {
        InvalidColumn::FrequencyIndex(error)
    }

    #[test]
    fn builder_preserves_frequency_behavior_and_errors() {
        let codes = [2, 0, 2, 3, 2, 0];
        let index = build_token_frequency_index(&codes, 5).unwrap();

        assert_eq!(index.num_tokens(), 5);
        assert_eq!(
            (0..5).map(|id| index.frequency(id)).collect::<Vec<_>>(),
            [2, 0, 3, 1, 0]
        );
        assert_eq!(index.range_frequency(TokenRange { begin: 1, last: 3 }), 4);
        assert_eq!(index.range_frequency(TokenRange::EMPTY), 0);
        assert_eq!(index.check_correctness(&codes), Ok(()));
        assert_eq!(
            build_token_frequency_index(&[0u16, 2], 2),
            Err(TokenFrequencyIndexError::CodeOutOfRange)
        );
        assert_eq!(
            build_token_frequency_index::<Token>(&[], Token::MAX as usize + 2),
            Err(TokenFrequencyIndexError::TooManyTokens)
        );
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            checked_frequency_total(u32::MAX as usize + 1),
            Err(TokenFrequencyIndexError::FrequencyOverflow)
        );
    }

    #[test]
    fn safety_and_full_validation_distinguish_inexact_frequencies() {
        let codes = [0, 0, 2, 2];
        let cumulative = [0, 0, 2, 4];
        let index =
            TokenFrequencyIndex::validate_safety(ExternalStorage(&cumulative), 3, codes.len())
                .unwrap();

        assert_eq!(index.frequency(0), 0);
        assert_eq!(
            index.check_correctness(&codes),
            Err(invalid_index(InvalidFrequencyIndex::FrequenciesMismatch))
        );
        assert_eq!(
            TokenFrequencyIndex::validate(ExternalStorage(&cumulative), &codes, 3).map(|_| ()),
            Err(invalid_index(InvalidFrequencyIndex::FrequenciesMismatch))
        );
        assert_eq!(
            TokenFrequencyIndexView::validate(&cumulative, &codes, 3).map(drop),
            Err(invalid_index(InvalidFrequencyIndex::FrequenciesMismatch))
        );
    }

    #[test]
    fn external_storage_is_retained_and_uses_the_same_queries() {
        let codes = [0, 0, 2, 2, 2, 3];
        let cumulative = [0, 2, 2, 5, 6];
        let original = cumulative.as_ptr();
        let owned = build_token_frequency_index(&codes, 4).unwrap();
        let index = TokenFrequencyIndex::validate(ExternalStorage(&cumulative), &codes, 4).unwrap();

        assert_eq!(index.storage().0.as_ptr(), original);
        assert_eq!(index.as_view().cumulative().as_ptr(), original);
        assert_eq!(index.frequency(2), owned.frequency(2));
        assert_eq!(
            index.range_frequency(TokenRange { begin: 1, last: 3 }),
            owned.range_frequency(TokenRange { begin: 1, last: 3 })
        );
        assert_eq!(
            TokenFrequencyIndexView::validate(&cumulative, &codes, 4)
                .unwrap()
                .cumulative()
                .as_ptr(),
            original
        );
        assert_eq!(index.into_storage().0.as_ptr(), original);
    }

    #[test]
    fn rejects_malformed_stored_representations() {
        let validate = |cumulative: &[u32], num_tokens, total_codes| {
            TokenFrequencyIndexView::validate_safety(cumulative, num_tokens, total_codes)
                .map(|_| ())
        };

        let cases: &[(&[u32], usize, usize, InvalidColumn)] = &[
            (
                &[0, 1],
                2,
                1,
                invalid_index(InvalidFrequencyIndex::BadLength),
            ),
            (
                &[1, 1],
                1,
                1,
                invalid_index(InvalidFrequencyIndex::FirstCumulativeNotZero),
            ),
            (
                &[0, 2, 1],
                2,
                1,
                invalid_index(InvalidFrequencyIndex::DecreasingCumulative),
            ),
            (
                &[0, 1, 3],
                2,
                2,
                invalid_index(InvalidFrequencyIndex::InvalidTotal),
            ),
            (
                &[],
                Token::MAX as usize + 2,
                0,
                InvalidColumn::CodeOutOfRange,
            ),
        ];
        for &(cumulative, num_tokens, total_codes, error) in cases {
            assert_eq!(validate(cumulative, num_tokens, total_codes), Err(error));
        }
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            validate(&[0], 0, u32::MAX as usize + 1),
            Err(invalid_index(InvalidFrequencyIndex::FrequencyOverflow))
        );
    }

    #[test]
    fn correctness_rejects_out_of_range_codes_and_wrong_totals() {
        let out_of_range = TokenFrequencyIndexView::validate_safety(&[0, 1, 1], 2, 1).unwrap();
        assert_eq!(
            out_of_range.check_correctness(&[2]),
            Err(InvalidColumn::CodeOutOfRange)
        );

        let wrong_total = TokenFrequencyIndexView::validate_safety(&[0, 2], 1, 2).unwrap();
        assert_eq!(
            wrong_total.check_correctness(&[0]),
            Err(invalid_index(InvalidFrequencyIndex::FrequenciesMismatch))
        );
    }
}
