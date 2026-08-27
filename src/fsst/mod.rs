// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bridge between [`fsst`] symbol tables and onpair dictionaries: [`encode`]
//! compresses with FSST into an onpair [`Column`](crate::Column)
pub mod encode;

pub use crate::fsst::encode::transcode_onpair;

#[cfg(test)]
mod tests;
