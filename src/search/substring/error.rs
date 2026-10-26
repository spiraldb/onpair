// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;

use crate::core::validate::InvalidColumn;

/// Error returned when a substring query cannot be prepared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainsError {
    /// The pattern is longer than the chosen implementation supports.
    PatternTooLong {
        /// Actual pattern length in bytes.
        length: usize,
        /// Maximum supported length in bytes.
        max: usize,
    },
    /// The dictionary or frequency index is invalid.
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
