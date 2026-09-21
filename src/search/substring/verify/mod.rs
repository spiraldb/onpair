// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Exact substring checks over encoded rows.
//!
//! `walk` confirms candidate token hits produced by bulk scanning, using the
//! compiled alignment graph. `kmp` prepares an independent token-transition
//! DFA for `row_contains`, which searches a whole row without a probe cover.
//! Both use local execution state and return exact containment results.

mod kmp;
pub(super) mod walk;

pub use kmp::{ContainsDfa, row_contains};
