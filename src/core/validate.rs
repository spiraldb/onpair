// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Validation errors for deserialized column data.
//!
//! `validate_safety` methods establish the invariants required for safe access;
//! full `validate` methods also check semantic conformance. Infallible
//! operations surface these errors through [`panic_malformed`]. Bad arguments
//! to encoding APIs use the separate [`Error`](crate::Error) type.

/// A violation found while validating compressed buffers.
///
/// Safety variants prevent valid access; conformance variants report data that
/// is structurally usable but semantically inconsistent.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum InvalidColumn {
    // ── Safety / addressability ──────────────────────────────────────────────
    /// The dictionary has no token, so it cannot be used by the search tokenizer.
    EmptyDictionary,
    /// The first dictionary offset is not zero.
    FirstOffsetNotZero,
    /// Dictionary offsets decrease (`offsets[i] > offsets[i + 1]`), which would
    /// underflow the unchecked token-length subtraction.
    DecreasingOffsets,
    /// A dictionary token is longer than [`MAX_TOKEN_SIZE`](crate::MAX_TOKEN_SIZE).
    TokenTooLarge,
    /// A token offset has fewer than [`MAX_TOKEN_SIZE`](crate::MAX_TOKEN_SIZE)
    /// readable bytes after it, so the decoder's fixed-width read runs off the end.
    MissingPadding,
    /// A dictionary or frequency index has more than `2^16` entries, or a code
    /// does not index its token domain (`code >= num_tokens`). In either case,
    /// the `u16` token/code type cannot address the requested entry.
    CodeOutOfRange,
    /// Row offsets are not non-decreasing, or the last exceeds the code count.
    BadRowOffsets,
    /// The column's tokens sum to more than `usize::MAX` decoded bytes, so the
    /// decoded-length computation overflows and would under-size the output buffer.
    DecodedLenOverflow,
    /// A dictionary token has zero length (offsets are not strictly increasing),
    /// so the search tokenizer would not make progress.
    EmptyToken,
    // ── Conformance / semantic correctness ──────────────────────────────────
    /// Dictionary tokens are not in strictly ascending bytewise order, so they are
    /// not sorted (binary search breaks) or not unique.
    UnsortedTokens,
    /// The dictionary lacks one or more of the 256 single-byte tokens, so some
    /// inputs are not encodable.
    IncompleteAlphabet,
    // ── Component-specific validation hierarchies ───────────────────────────
    /// A stored token-frequency index is structurally malformed or does not
    /// conform to the associated column. See [`InvalidFrequencyIndex`] for the
    /// precise safety or correctness failure.
    FrequencyIndex(InvalidFrequencyIndex),
}

/// A structural or semantic failure in a stored token-frequency index.
///
/// Exposed through [`InvalidColumn::FrequencyIndex`] because validation depends
/// on a column's token domain and code stream. Builder argument errors use
/// [`TokenFrequencyIndexError`](crate::search::index::TokenFrequencyIndexError).
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum InvalidFrequencyIndex {
    // ── Safety / addressability ──────────────────────────────────────────────
    /// The cumulative buffer does not contain exactly one more entry than its
    /// token domain requires.
    BadLength,
    /// The first cumulative entry is not zero.
    FirstCumulativeNotZero,
    /// Cumulative entries decrease, which would underflow a frequency or
    /// range-frequency subtraction.
    DecreasingCumulative,
    /// The associated code count cannot be represented by the index's `u32`
    /// cumulative entries.
    FrequencyOverflow,
    /// The final cumulative entry does not equal the associated code count.
    InvalidTotal,
    // ── Conformance / semantic correctness ──────────────────────────────────
    /// The cumulative deltas do not exactly count the associated code stream.
    FrequenciesMismatch,
}

impl std::fmt::Display for InvalidColumn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::EmptyDictionary => "dictionary must contain at least one token",
            Self::FirstOffsetNotZero => "dictionary offsets must start at zero",
            Self::DecreasingOffsets => "dictionary offsets must be strictly increasing",
            Self::TokenTooLarge => "dictionary token exceeds MAX_TOKEN_SIZE",
            Self::MissingPadding => "dictionary bytes lack the required trailing decoder padding",
            Self::CodeOutOfRange => "code index out of range for token domain",
            Self::BadRowOffsets => "row offsets must be non-decreasing and within the code stream",
            Self::DecodedLenOverflow => "column decodes to more than usize::MAX bytes",
            Self::EmptyToken => "dictionary contains an empty token",
            Self::UnsortedTokens => "dictionary tokens must be sorted and unique",
            Self::IncompleteAlphabet => "dictionary is missing one or more single-byte tokens",
            Self::FrequencyIndex(error) => return error.fmt(f),
        };
        f.write_str(message)
    }
}

impl std::error::Error for InvalidColumn {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::FrequencyIndex(error) => Some(error),
            _ => None,
        }
    }
}

impl From<InvalidFrequencyIndex> for InvalidColumn {
    fn from(error: InvalidFrequencyIndex) -> Self {
        Self::FrequencyIndex(error)
    }
}

impl std::fmt::Display for InvalidFrequencyIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::BadLength => "frequency index length must equal the token count plus one",
            Self::FirstCumulativeNotZero => "frequency index must start at zero",
            Self::DecreasingCumulative => "frequency index values must be non-decreasing",
            Self::FrequencyOverflow => "code stream is too large for a u32 frequency index",
            Self::InvalidTotal => "frequency index total does not match the associated code count",
            Self::FrequenciesMismatch => {
                "token frequencies do not match the associated code stream"
            }
        })
    }
}

impl std::error::Error for InvalidFrequencyIndex {}

/// Panic for a malformed column/dictionary, message derived from
/// `InvalidColumn`'s `Display`. `#[cold]` + `#[inline(never)]` so a caller's
/// guard is laid out as a never-taken branch.
#[cold]
#[inline(never)]
pub(crate) fn panic_malformed(e: InvalidColumn) -> ! {
    panic!("onpair: {e}")
}
