// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The dictionary: the token table a code stream indexes into.
//!
//! A dictionary is a code-addressable token vocabulary in one of two physical
//! representations of the same data:
//!
//! * [`CompactDictionary`] / [`CompactDictionaryView`] ([`compact`]) — Arrow
//!   binary: a flat `bytes` buffer plus an `offsets` index.
//! * [`WideDictionary`] / [`WideDictionaryView`] ([`wide`]) — a
//!   `num_tokens × MAX_TOKEN_SIZE` byte-strided copy.
//!
//! Both borrowed views implement [`DictionaryView`], the layout-agnostic
//! token-read interface; the owned forms implement [`Dictionary`], which lends
//! the matching view. The compact form is the serialized one; accelerate decoding
//! by materializing the wide form with [`CompactDictionaryView::to_wide`] or
//! [`WideDictionary::validate_safety`] when starting from raw storage.
//!
//! # Invariants and trust
//! The dictionary types above are crate-private in their fields and implement a
//! **sealed** [`DictionaryView`], so external code cannot forge one.
//! A `DictionaryView` guarantees the structural invariants required for bounded
//! token access. Sortedness, uniqueness, and alphabet completeness are separate
//! semantic properties used by search and tokenization.
//!
//! Raw deserialized buffers cross into the trusted forms through
//! [`CompactDictionary::validate_safety`] / [`CompactDictionary::validate`] or
//! directly into the decode-optimized [`WideDictionary`] with
//! [`WideDictionary::validate_safety`]. The `new_unchecked` constructors are
//! unsafe backdoors for callers that already guarantee the structural dictionary
//! invariants.

mod compact;
mod wide;

pub(crate) use compact::pad_raw;
pub use compact::{
    CompactDictionary, CompactDictionaryView, DictionaryStorage, OwnedDictionaryStorage,
    code_bits_for_num_tokens,
};
pub use wide::{WideDictionary, WideDictionaryView};

use crate::core::types::Token;

/// The crate-internal half of [`DictionaryView`].
///
/// Two jobs, and the first follows from the second. The module is `pub(crate)`, so
/// the trait is unreachable and undocumented outside the crate even though it is
/// nominally `pub` — which seals [`DictionaryView`], since that trait cannot be
/// implemented without also implementing this one. And it is where the crate keeps
/// the layout questions it needs to ask a view but does not publish, because a
/// caller holding layout assumptions is exactly what the trust model above rules
/// out. See [`internal::ViewInternal::token_payload`].
///
/// Contrast the `Sealed` marker behind [`Offset`](crate::Offset), which carries
/// nothing and is therefore named for the sealing alone.
pub(crate) mod internal {
    /// What the crate asks a view about its layout, and — by being unreachable
    /// downstream — the seal on [`DictionaryView`](super::DictionaryView).
    pub trait ViewInternal {
        /// The tokens as one buffer in which **every byte belongs to exactly one
        /// token**, plus the `num_tokens + 1` offsets attributing them — or `None`
        /// from a layout that has no such buffer.
        ///
        /// That property, rather than contiguity as such, is what a caller needs:
        /// it makes the buffer a valid haystack, because any match found within it
        /// can be attributed to the token whose offsets bracket it. The strided
        /// layout has to answer `None` precisely because the property fails for
        /// its padding, which holds the bytes of *following* tokens and would
        /// manufacture matches that no token contains.
        ///
        /// `None` is a permanent fact about a layout and not an unimplemented
        /// case, so a caller needs a fallback that does not read a payload at all.
        fn token_payload(&self) -> Option<(&[u8], &[u32])>;
    }
}

/// An owned dictionary that can lend its borrowed [`DictionaryView`].
///
/// Implemented by both representations ([`CompactDictionary`] and
/// [`WideDictionary`]), so generic code can accept any owned dictionary and
/// obtain a view.
pub trait Dictionary {
    /// The borrowed view this dictionary lends.
    type View<'a>: DictionaryView
    where
        Self: 'a;

    /// Borrow as a [`DictionaryView`].
    fn as_view(&self) -> Self::View<'_>;
}

/// A borrowed dictionary's token-read interface, abstracted over the layout
/// ([`CompactDictionaryView`] or [`WideDictionaryView`]).
///
/// **Sealed:** implemented only by this crate's trusted view types, so a value of
/// `V: DictionaryView` has the structural invariants required by the unchecked
/// accessors below. Semantic conformance is not part of this trait. Untrusted
/// buffers reach a view through [`CompactDictionaryView::validate_safety`] or
/// [`CompactDictionaryView::validate`], not by implementing this trait.
pub trait DictionaryView: Copy + internal::ViewInternal {
    /// Number of tokens in the dictionary. The valid token ids are
    /// `0..num_tokens()`.
    fn num_tokens(&self) -> usize;

    /// Bytes of token `id`. Bounds-checked; panics if `id` is out of range.
    fn token(&self, id: Token) -> &[u8];

    /// Byte length of token `id`. Bounds-checked; panics if `id` is out of range.
    fn token_len(&self, id: Token) -> usize;

    /// Raw pointer to token `id`'s bytes.
    ///
    /// # Safety
    /// `id` is a valid code (less than the number of tokens), and the pointer
    /// must be valid for [`MAX_TOKEN_SIZE`](crate::MAX_TOKEN_SIZE) readable bytes
    /// — the fast decode path over-reads a fixed 16-byte chunk regardless of the
    /// true length.
    unsafe fn token_ptr(&self, id: Token) -> *const u8;

    /// Token `id`'s length — the unchecked counterpart of
    /// [`token_len`](Self::token_len).
    ///
    /// # Safety
    /// `id` is a valid code (less than the number of tokens).
    unsafe fn token_len_unchecked(&self, id: Token) -> usize;
}
