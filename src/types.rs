// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Crate-private numeric types shared by trainer / LPM / dictionary.

/// Number of bits per code. Legal range: 9..=16 (validated at the public
/// boundary).
pub(crate) type BitWidth = u8;

/// Token identifier within a dictionary. Capped at `2^bits` per column.
pub(crate) type Token = u16;

/// Maximum byte length of any dictionary token.
///
/// Also the decoder's fixed read width: it reads `MAX_TOKEN_SIZE` bytes from
/// each token offset and slices to the token's true length (the branchless
/// "fat read, then advance by `len`" pattern). A dictionary's byte buffer must
/// therefore extend `MAX_TOKEN_SIZE` past its highest token offset so that read
/// never touches unallocated memory — see [`crate::Parts::validate_dictionary`].
pub const MAX_TOKEN_SIZE: usize = 16;

/// Byte range `[begin, end)` inside the dictionary buffer.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct ByteSpan {
    pub(crate) begin: u32,
    pub(crate) end: u32,
}

/// Maximum dictionary size given a bit width.
#[inline]
pub(crate) const fn max_dict_size(bits: BitWidth) -> usize {
    1usize << bits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_dict_size_12_is_4096() {
        assert_eq!(max_dict_size(12), 4096);
    }

    #[test]
    fn max_dict_size_16_is_65536() {
        assert_eq!(max_dict_size(16), 65536);
    }

    #[test]
    fn max_token_size_is_16() {
        assert_eq!(MAX_TOKEN_SIZE, 16);
    }
}
