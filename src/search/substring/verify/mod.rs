// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Exact substring verification and standalone KMP search.

mod bytes;
mod kmp;
pub(super) mod walk;

pub use bytes::BytesVerifier;
pub use kmp::{ContainsTable, contains};
