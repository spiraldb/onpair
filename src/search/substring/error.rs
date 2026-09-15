// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;

use crate::core::validate::InvalidColumn;

/// A substring query could not be prepared. No matching rows have been produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainsError {
    /// The pattern exceeds the state width of the chosen implementation.
    PatternTooLong {
        /// Actual pattern length in bytes.
        length: usize,
        /// Maximum supported length in bytes.
        max: usize,
    },
    /// Preparation encountered invalid dictionary or frequency-index data.
    InvalidData(InvalidColumn),
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
            Self::InvalidData(error) => write!(f, "invalid substring search data: {error}"),
        }
    }
}

impl std::error::Error for ContainsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidData(error) => Some(error),
            _ => None,
        }
    }
}
