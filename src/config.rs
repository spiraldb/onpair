// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::types::BitWidth;

// ─────────────────────────────────────────────────────────────────────────────
// Public config — slim, FFI-friendly.
// ─────────────────────────────────────────────────────────────────────────────

/// Training configuration.
///
/// - `bits`: max dict size = `2^bits`. Range `9..=16`.
/// - `threshold`: dynamic-threshold sample fraction. Range `(0.0, 1.0]`.
/// - `seed`: `0` means non-deterministic.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Config {
    pub bits: u32,
    pub threshold: f64,
    pub seed: u64,
}

/// Reasonable starting point: 12-bit codes, dynamic threshold sampling 20 %.
pub const DEFAULT_CONFIG: Config = Config {
    bits: 12,
    threshold: 0.2,
    seed: 0,
};

impl Default for Config {
    fn default() -> Self {
        DEFAULT_CONFIG
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Error — single-variant.
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Error {
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
        Self {
            sample_fraction: 0.2,
        }
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
        Self {
            bits: 16,
            threshold: ThresholdSpec::default(),
            seed: None,
        }
    }
}

impl From<Config> for TrainingConfig {
    fn from(c: Config) -> Self {
        Self {
            bits: c.bits as BitWidth,
            threshold: ThresholdSpec::Dynamic(DynamicThreshold {
                sample_fraction: c.threshold,
            }),
            seed: (c.seed != 0).then_some(c.seed),
        }
    }
}

/// Validate a public [`Config`].
pub(crate) fn validate_config(cfg: Config) -> Result<(), Error> {
    if !(9..=16).contains(&cfg.bits) {
        return Err(Error::InvalidArg);
    }
    if !(cfg.threshold > 0.0 && cfg.threshold <= 1.0) {
        return Err(Error::InvalidArg);
    }
    Ok(())
}
