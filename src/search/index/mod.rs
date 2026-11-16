// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Indexes over a code stream that steer search planning.
//!
//! A [`TokenFrequencyIndex`] counts how often each token occurs. It is built once
//! per stream and can be stored next to it: [`TokenFrequencyIndexView`] borrows the
//! stored prefix sums without copying them. Without a stored index, a sample of the
//! stream can be counted instead.

mod frequency;

pub use frequency::{InvalidFrequencyIndex, TokenFrequencyIndex, TokenFrequencyIndexView};
