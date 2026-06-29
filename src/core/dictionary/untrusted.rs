// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Untrusted dictionary buffers and the doors into the trusted forms.
//!
//! [`UntrustedDictionary`] / [`UntrustedDictionaryView`] hold the raw
//! `(bytes, offsets)` a consumer deserialized from storage. Their fields are
//! public and they serialize freely, but they implement none of the trusted
//! token interface. The two doors across the trust boundary are
//! [`validate`](UntrustedDictionary::validate) (checked) and
//! [`trust_unchecked`](UntrustedDictionary::trust_unchecked) (the `unsafe`
//! backdoor for callers that already know the buffers are conformant).

use super::compact::{CompactDictionary, CompactDictionaryView};
use crate::core::types::MAX_TOKEN_SIZE;
use crate::core::validate::InvalidColumn;

/// Validate raw compact `(bytes, offsets)` into a fully conformant dictionary:
/// both safety (no out-of-bounds decode) and conformance (correct search /
/// tokenize). `O(total token bytes)` — the token bytes are read once.
///
/// Two passes: the first checks the offset-only safety invariants and proves every
/// token's bytes are in bounds; the second reads those bytes to check sortedness
/// and alphabet completeness.
fn validate_compact(bytes: &[u8], offsets: &[u32]) -> Result<(), InvalidColumn> {
    // Pass 1 — safety: strictly-increasing offsets (so the length subtraction can't
    // underflow and no token is empty), bounded length, and the trailing
    // read-padding. `s` of the final window is the highest token offset, so a
    // single padding bound covers every token's fixed-width over-read. After this,
    // every `bytes[offsets[i]..offsets[i + 1]]` slice is in bounds.
    let mut last_offset = 0usize;
    for w in offsets.windows(2) {
        let (s, e) = (w[0], w[1]);
        if e < s {
            return Err(InvalidColumn::NonDecreasingOffsets);
        }
        if e == s {
            return Err(InvalidColumn::EmptyToken);
        }
        if (e - s) as usize > MAX_TOKEN_SIZE {
            return Err(InvalidColumn::TokenTooLarge);
        }
        last_offset = s as usize;
    }
    if offsets.len() >= 2 && last_offset + MAX_TOKEN_SIZE > bytes.len() {
        return Err(InvalidColumn::MissingPadding);
    }

    // Pass 2 — conformance: tokens strictly ascending (hence sorted and unique) and
    // the 256 single-byte tokens all present. Indexing `bytes` is sound after pass 1.
    let mut seen = [false; 256];
    let mut prev: &[u8] = &[];
    for w in offsets.windows(2) {
        let token = &bytes[w[0] as usize..w[1] as usize];
        if prev >= token {
            return Err(InvalidColumn::UnsortedTokens);
        }
        if token.len() == 1 {
            seen[token[0] as usize] = true;
        }
        prev = token;
    }
    if seen.iter().any(|&present| !present) {
        return Err(InvalidColumn::IncompleteAlphabet);
    }
    Ok(())
}

/// Untrusted owned compact dictionary: the raw `(bytes, offsets)` buffers a
/// consumer deserialized from storage, before they are trusted.
///
/// Public fields, freely constructible and serializable. Cross into the trusted
/// [`CompactDictionary`] with [`validate`](Self::validate) or
/// [`trust_unchecked`](Self::trust_unchecked).
#[derive(Default, Debug, Clone)]
pub struct UntrustedDictionary {
    /// Read-padded token bytes (see `docs/interchange-format.md` §3.1).
    pub bytes: Vec<u8>,
    /// `num_tokens + 1` offsets into `bytes`.
    pub offsets: Vec<u32>,
}

impl UntrustedDictionary {
    /// Validate the buffers into a fully conformant trusted [`CompactDictionary`]
    /// (moved, no copy) — both safe to decode and correct to search / tokenize.
    ///
    /// # Errors
    /// [`InvalidColumn`] for any safety violation (decreasing offsets, an over-long
    /// token, missing read-padding) or conformance violation (an empty token,
    /// unsorted/duplicate tokens, or an incomplete single-byte alphabet). Does
    /// **not** pad — a conformant serialized dictionary already carries the
    /// read-padding.
    pub fn validate(self) -> Result<CompactDictionary, InvalidColumn> {
        validate_compact(&self.bytes, &self.offsets)?;
        // SAFETY: the buffers just passed validation.
        Ok(unsafe { self.trust_unchecked() })
    }

    /// Seal into a trusted [`CompactDictionary`] without checking.
    ///
    /// # Safety
    /// Every invariant documented for [`CompactDictionary`] is a precondition: the
    /// resulting value must be indistinguishable from a validated one. They split
    /// into two tiers by how a violation bites, but both are required —
    ///
    /// * **Read-safety** (violating these is UB — an unchecked decoder reads or
    ///   writes out of bounds): `offsets[0] == 0`, `offsets.len() == num_tokens + 1`,
    ///   non-decreasing offsets, every token `≤ MAX_TOKEN_SIZE`, and read-padded
    ///   (`offsets.last() + MAX_TOKEN_SIZE <= bytes.len()`).
    /// * **Conformance** (violating these is not UB, but search / tokenize then
    ///   return wrong answers): strictly-increasing offsets, tokens sorted and
    ///   unique, and the 256-symbol alphabet complete.
    ///
    /// [`validate`](Self::validate) checks both tiers; this skips both and places
    /// the whole obligation on the caller.
    pub unsafe fn trust_unchecked(self) -> CompactDictionary {
        CompactDictionary::from_raw(self.bytes, self.offsets)
    }
}

/// Untrusted borrowed compact dictionary — the zero-copy counterpart of
/// [`UntrustedDictionary`] for buffers borrowed from storage (e.g. mmap).
#[derive(Copy, Clone, Debug)]
pub struct UntrustedDictionaryView<'a> {
    /// Read-padded token bytes.
    pub bytes: &'a [u8],
    /// `num_tokens + 1` offsets into `bytes`.
    pub offsets: &'a [u32],
}

impl<'a> UntrustedDictionaryView<'a> {
    /// Check the safety invariants and, on success, return a trusted
    /// [`CompactDictionaryView`] over the same borrowed slices (no copy).
    ///
    /// # Errors
    /// As [`UntrustedDictionary::validate`]. The borrowed bytes must already be
    /// read-padded — a borrow cannot be extended.
    pub fn validate(self) -> Result<CompactDictionaryView<'a>, InvalidColumn> {
        validate_compact(self.bytes, self.offsets)?;
        // SAFETY: the slices just passed validation.
        Ok(unsafe { self.trust_unchecked() })
    }

    /// Trust the borrowed slices without checking.
    ///
    /// # Safety
    /// As [`UntrustedDictionary::trust_unchecked`].
    pub unsafe fn trust_unchecked(self) -> CompactDictionaryView<'a> {
        CompactDictionaryView::from_raw(self.bytes, self.offsets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dictionary::DictionaryView;

    /// Read-padded `(bytes, offsets)` from an explicit token list — the caller
    /// controls exactly which conformance property is under test.
    fn padded(tokens: &[&[u8]]) -> (Vec<u8>, Vec<u32>) {
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for t in tokens {
            bytes.extend_from_slice(t);
            offsets.push(bytes.len() as u32);
        }
        bytes.resize(bytes.len() + MAX_TOKEN_SIZE, 0); // worst-case read padding
        (bytes, offsets)
    }

    /// A conformant dictionary: all 256 single-byte tokens plus `extra`, sorted and
    /// deduped (so strictly increasing, sorted, unique, complete, and padded).
    fn conformant(extra: &[&[u8]]) -> (Vec<u8>, Vec<u32>) {
        let mut toks: Vec<Vec<u8>> = (0u16..256).map(|b| vec![b as u8]).collect();
        for &t in extra {
            toks.push(t.to_vec());
        }
        toks.sort();
        toks.dedup();
        let refs: Vec<&[u8]> = toks.iter().map(Vec::as_slice).collect();
        padded(&refs)
    }

    /// Map to `Result<(), _>` so we can compare (`CompactDictionary` isn't `Eq`).
    fn check(bytes: Vec<u8>, offsets: Vec<u32>) -> Result<(), InvalidColumn> {
        UntrustedDictionary { bytes, offsets }
            .validate()
            .map(|_| ())
    }

    #[test]
    fn validate_accepts_conformant() {
        let (bytes, offsets) = conformant(&[b"bc", b"def"]);
        assert_eq!(check(bytes, offsets), Ok(()));
    }

    #[test]
    fn validate_classifies_safety_corruption() {
        // Decreasing offsets (would underflow the length subtraction).
        let mut bytes = b"ab".to_vec();
        bytes.resize(2 + MAX_TOKEN_SIZE, 0);
        assert_eq!(
            check(bytes, vec![0, 2, 1]),
            Err(InvalidColumn::NonDecreasingOffsets)
        );

        // Zero-length token (`e == s`).
        assert_eq!(
            check(vec![0u8; MAX_TOKEN_SIZE], vec![0, 0]),
            Err(InvalidColumn::EmptyToken)
        );

        // Token longer than MAX_TOKEN_SIZE.
        assert_eq!(
            check(vec![b'x'; 20 + MAX_TOKEN_SIZE], vec![0, 20]),
            Err(InvalidColumn::TokenTooLarge)
        );

        // Missing the trailing read-padding.
        assert_eq!(
            check(b"abc".to_vec(), vec![0, 1, 3]),
            Err(InvalidColumn::MissingPadding)
        );
    }

    #[test]
    fn validate_classifies_conformance_corruption() {
        // Safe + padded but out of order (also how a duplicate would surface).
        let (bytes, offsets) = padded(&[&[1u8], &[0u8]]);
        assert_eq!(check(bytes, offsets), Err(InvalidColumn::UnsortedTokens));

        // Sorted + safe but missing all but three single-byte tokens.
        let (bytes, offsets) = padded(&[&[0u8], &[1u8], &[2u8]]);
        assert_eq!(
            check(bytes, offsets),
            Err(InvalidColumn::IncompleteAlphabet)
        );
    }

    #[test]
    fn trust_unchecked_matches_validate() {
        let (bytes, offsets) = conformant(&[b"bc"]);
        let checked = UntrustedDictionary {
            bytes: bytes.clone(),
            offsets: offsets.clone(),
        }
        .validate()
        .unwrap();
        // SAFETY: `conformant` produces a conformant dictionary.
        let trusted = unsafe { UntrustedDictionary { bytes, offsets }.trust_unchecked() };
        assert_eq!(checked.bytes(), trusted.bytes());
        assert_eq!(checked.offsets(), trusted.offsets());
    }

    #[test]
    fn untrusted_view_validate_yields_usable_view() {
        let (bytes, offsets) = conformant(&[b"bc"]);
        let view = UntrustedDictionaryView {
            bytes: &bytes,
            offsets: &offsets,
        }
        .validate()
        .unwrap();
        assert_eq!(view.num_tokens(), 257); // 256 single bytes + "bc"
        assert_eq!(view.token(0), &[0u8]); // byte 0 sorts first

        // A non-conformant borrow is rejected.
        let raw: &[u8] = b"abc";
        assert!(
            UntrustedDictionaryView {
                bytes: raw,
                offsets: &[0, 1, 3],
            }
            .validate()
            .is_err()
        );
    }
}
