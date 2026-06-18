// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compressed-domain search — reserved.
//!
//! This module is a placeholder for substring, prefix, and equality search over
//! the code stream without decoding. No automata are implemented yet; the core
//! types already satisfy the contract such operations need:
//!
//! * The dictionary is **sorted, complete, and unique**
//!   ([`CompactDictionary`](crate::CompactDictionary) invariants): sortedness enables binary
//!   search and prefix-range lookup; completeness makes any query string
//!   encodable into codes; uniqueness makes distinct code sequences denote
//!   distinct strings (required for compressed-domain equality).
//! * Codes are a plain `&[`[`Token`](crate::Token)`]`, so scans run directly
//!   over token-id slices — a row to scan is
//!   [`ColumnView::row_codes`](crate::ColumnView::row_codes).
//!
//! When search lands, `ColumnView` grows `contains` / `starts_with` / `equals`
//! and `CompactDictionaryView` grows `prefix_range`, all on top of these guarantees.
