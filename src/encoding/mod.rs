// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression: configuration, dictionary training, and string parsing.
//!
//! [`config`] is the public training configuration. [`trainer`] discovers a
//! dictionary from a sample using the incremental [`lpm`] longest-prefix
//! matcher; [`parser`] drives the fixed-cost [`flat`] matcher over the input to
//! produce a column. [`hash`] is the shared hasher.

pub(crate) mod config;
pub(crate) mod flat;
pub(crate) mod hash;
pub(crate) mod lpm;
pub(crate) mod parser;
pub(crate) mod trainer;
