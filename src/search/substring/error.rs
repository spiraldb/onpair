// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;

/// Error returned when a substring query cannot be prepared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ContainsError {
    /// The pattern is longer than the chosen implementation supports.
    PatternTooLong {
        /// Actual pattern length in bytes.
        length: usize,
        /// Maximum supported length in bytes.
        max: usize,
    },
    /// The frequency index counts a different number of tokens than the dictionary
    /// has.
    FrequencyIndexMismatch {
        /// Tokens counted by the frequency index.
        index_tokens: usize,
        /// Tokens in the dictionary.
        dict_tokens: usize,
    },
    /// The dictionary cannot tokenize the pattern: no single-byte token was found
    /// for `byte`, a byte of the pattern. Conformant dictionaries, such as trained
    /// ones, have all 256, so the dictionary is incomplete or unsorted.
    MissingByteToken {
        /// The pattern byte without a token.
        byte: u8,
    },
}

impl fmt::Display for ContainsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PatternTooLong { length, max } => {
                write!(
                    f,
                    "substring pattern has {length} bytes; this implementation supports at most {max}"
                )
            }
            Self::FrequencyIndexMismatch {
                index_tokens,
                dict_tokens,
            } => write!(
                f,
                "frequency index counts {index_tokens} tokens; the dictionary has {dict_tokens}"
            ),
            Self::MissingByteToken { byte } => {
                write!(
                    f,
                    "found no single-byte token for {byte:#04x}; the dictionary is incomplete or unsorted"
                )
            }
        }
    }
}

impl std::error::Error for ContainsError {}
