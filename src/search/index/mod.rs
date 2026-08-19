// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reusable indexes over compressed search data.
//!
//! Each index owns its storage contract and validation boundary. Token-frequency
//! indexes support cheap structural validation of deserialized storage and
//! optional full validation against a code stream.

mod frequency;

pub use frequency::{
    OwnedTokenFrequencyIndexStorage, TokenFrequencyIndex, TokenFrequencyIndexError,
    TokenFrequencyIndexStorage, TokenFrequencyIndexView, build_token_frequency_index,
};
