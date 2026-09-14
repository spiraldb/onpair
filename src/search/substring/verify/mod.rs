// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Exact substring verification and standalone KMP search.

mod kmp;
pub(super) mod walk;

pub use kmp::{ContainsTable, contains};
