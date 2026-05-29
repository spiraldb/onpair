// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Decode-side dictionary: Arrow binary layout (flat `bytes` + `offsets` of
// length `num_tokens + 1`). `bits` records the trained code width so consumers
// can choose how to store codes if they want narrower than `u16`. The 256
// single-byte tokens are always present after `train`, which makes every
// input string encodable.

use crate::types::ByteSpan;
use crate::types::Token;

/// Decode-side dictionary. Plain data — no methods required by the public API.
#[derive(Default, Debug, Clone)]
pub struct Dictionary {
    /// Concatenation of token byte sequences.
    pub bytes: Vec<u8>,
    /// `offsets[i]..offsets[i+1]` = byte range of token `i` in `bytes`.
    /// `offsets.len() == num_tokens + 1`, `offsets[0] == 0`.
    pub offsets: Vec<u32>,
    /// Code width used at training time. `9..=16`.
    pub bits: u32,
}

impl Dictionary {
    #[inline]
    pub(crate) fn num_tokens(&self) -> usize {
        if self.offsets.is_empty() {
            0
        } else {
            self.offsets.len() - 1
        }
    }

    #[inline]
    pub(crate) fn span(&self, id: Token) -> ByteSpan {
        ByteSpan {
            begin: self.offsets[id as usize],
            end: self.offsets[id as usize + 1],
        }
    }

    #[inline]
    pub(crate) fn data(&self, id: Token) -> &[u8] {
        let s = self.span(id);
        &self.bytes[s.begin as usize..s.end as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_tokens_zero_when_offsets_empty() {
        let d = Dictionary::default();
        assert_eq!(d.num_tokens(), 0);
    }

    #[test]
    fn num_tokens_is_offsets_len_minus_one() {
        let d = Dictionary {
            bytes: vec![],
            offsets: vec![0, 3, 5, 8],
            bits: 12,
        };
        assert_eq!(d.num_tokens(), 3);
    }

    #[test]
    fn span_returns_correct_range() {
        let d = Dictionary {
            bytes: b"abcdef".to_vec(),
            offsets: vec![0, 1, 3, 6],
            bits: 12,
        };
        assert_eq!(d.span(0), ByteSpan { begin: 0, end: 1 });
        assert_eq!(d.span(1), ByteSpan { begin: 1, end: 3 });
        assert_eq!(d.span(2), ByteSpan { begin: 3, end: 6 });
    }

    #[test]
    fn data_returns_correct_slice() {
        let d = Dictionary {
            bytes: b"abcdef".to_vec(),
            offsets: vec![0, 1, 3, 6],
            bits: 12,
        };
        assert_eq!(d.data(0), b"a");
        assert_eq!(d.data(1), b"bc");
        assert_eq!(d.data(2), b"def");
    }
}
