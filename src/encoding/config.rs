// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Public training configuration and the crate-internal form it lowers to.

use crate::core::types::BitWidth;

// ─────────────────────────────────────────────────────────────────────────────
// Public config.
// ─────────────────────────────────────────────────────────────────────────────

/// Code width: the maximum dictionary size is `2^bits`. Validated to `9..=16`
/// at construction, so a [`Bits`] always holds an in-range value.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Bits(u8);

impl Bits {
    /// Construct a [`Bits`], returning [`Error::InvalidArg`] unless `value` is
    /// in `9..=16`.
    pub const fn new(value: u8) -> Result<Self, Error> {
        if 9 <= value && value <= 16 {
            Ok(Self(value))
        } else {
            Err(Error::InvalidArg)
        }
    }

    /// The validated code width, in `9..=16`.
    pub const fn value(self) -> u8 {
        self.0
    }
}

impl TryFrom<u8> for Bits {
    type Error = Error;
    fn try_from(value: u8) -> Result<Self, Error> {
        Self::new(value)
    }
}

/// Dynamic-threshold sample fraction. Validated to `(0.0, 1.0]` at
/// construction, so a [`Threshold`] always holds an in-range value.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Threshold(f64);

impl Threshold {
    /// Construct a [`Threshold`], returning [`Error::InvalidArg`] unless `value`
    /// is in `(0.0, 1.0]`.
    pub const fn new(value: f64) -> Result<Self, Error> {
        if value > 0.0 && value <= 1.0 {
            Ok(Self(value))
        } else {
            Err(Error::InvalidArg)
        }
    }

    /// The validated sample fraction, in `(0.0, 1.0]`.
    pub const fn value(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for Threshold {
    type Error = Error;
    fn try_from(value: f64) -> Result<Self, Error> {
        Self::new(value)
    }
}

/// Training configuration. See [`DEFAULT_CONFIG`] for a reasonable starting
/// point.
#[derive(Copy, Clone, Debug)]
pub struct Config {
    /// Code width; see [`Bits`].
    pub bits: Bits,
    /// Dynamic-threshold sample fraction; see [`Threshold`].
    pub threshold: Threshold,
    /// RNG seed for sampling; `None` means non-deterministic.
    pub seed: Option<u64>,
}

/// Reasonable starting point: 12-bit codes, dynamic threshold sampling 15 %.
pub const DEFAULT_CONFIG: Config = Config {
    bits: match Bits::new(12) {
        Ok(b) => b,
        Err(_) => unreachable!(),
    },
    threshold: match Threshold::new(0.15) {
        Ok(t) => t,
        Err(_) => unreachable!(),
    },
    seed: None,
};

impl Default for Config {
    fn default() -> Self {
        DEFAULT_CONFIG
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Error — single-variant.
// ─────────────────────────────────────────────────────────────────────────────

/// Error returned by the public training and encoding API.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Error {
    /// A configuration value or input buffer was out of range or malformed.
    InvalidArg,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InvalidArg => f.write_str("onpair: invalid argument"),
        }
    }
}

impl std::error::Error for Error {}

// ─────────────────────────────────────────────────────────────────────────────
// Internal training config — crate-private. Kept richer than the public Config
// so unit tests can still drive fixed-threshold training.
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct FixedThreshold {
    pub(crate) value: u8,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct DynamicThreshold {
    pub(crate) sample_fraction: f64,
}

impl Default for DynamicThreshold {
    fn default() -> Self {
        Self { sample_fraction: 0.15 }
    }
}

#[derive(Copy, Clone, Debug)]
#[allow(dead_code)] // `Fixed` is used only in tests
pub(crate) enum ThresholdSpec {
    Fixed(FixedThreshold),
    Dynamic(DynamicThreshold),
}

impl Default for ThresholdSpec {
    fn default() -> Self {
        Self::Dynamic(DynamicThreshold::default())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct TrainingConfig {
    pub(crate) bits: BitWidth,
    pub(crate) threshold: ThresholdSpec,
    pub(crate) seed: Option<u64>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self { bits: 16, threshold: ThresholdSpec::default(), seed: None }
    }
}

impl From<Config> for TrainingConfig {
    fn from(c: Config) -> Self {
        Self {
            bits: c.bits.value(),
            threshold: ThresholdSpec::Dynamic(DynamicThreshold {
                sample_fraction: c.threshold.value(),
            }),
            seed: c.seed,
        }
    }
}
