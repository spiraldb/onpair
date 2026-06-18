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
//! the matching view. Convert between representations with
//! [`CompactDictionaryView::to_wide`] and [`WideDictionaryView::to_compact`].

mod compact;
mod wide;

pub use compact::{CompactDictionary, CompactDictionaryView};
pub use wide::{WideDictionary, WideDictionaryView};

use crate::core::types::Token;

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
pub trait DictionaryView: Copy {
    /// Bytes of token `id`. Bounds-checked; panics if `id` is out of range.
    fn token(&self, id: Token) -> &[u8];

    /// Byte length of token `id`. Bounds-checked; panics if `id` is out of range.
    fn token_len(&self, id: Token) -> usize;

    /// Total byte length of the tokens `codes` name (the sum of their lengths).
    /// Bounds-checked — panics on an out-of-range code. Sizes the output buffer
    /// for [`decode_to_vec`](crate::decode_to_vec).
    fn decoded_len(&self, codes: &[Token]) -> usize;

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
