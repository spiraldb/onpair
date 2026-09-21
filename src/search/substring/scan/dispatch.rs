// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Connect a matcher configuration to its concrete implementation.
//!
//! The configuration chooses the matcher family and empty-group packing policy.
//! Dispatch derives nibble batches from the cover and resolves them once before scanning,
//! so the block loop runs with concrete type and const parameters.
//!
//! AArch64 uses NEON. On x86, this build contains AVX2 kernels unless `avx512bw`
//! is enabled at compile time, in which case it contains the AVX-512 versions.
//! Runtime detection enables the compiled family when supported; otherwise
//! selection uses the scalar table. Other architectures use the table too.

use super::matcher::{self, Matcher};
use super::{Check, Isa, MatcherConfig, both_stages};
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use super::{MatcherKind, PER_BATCH};
use crate::core::offset::Offset;
use crate::core::types::Token;
use crate::search::substring::ProbeCover;

/// Select whether empty groups skip mask packing: `S` enables it, `P` disables it.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn with_skip<O: Offset, S: Matcher, P: Matcher>(
    config: MatcherConfig,
    cover: &ProbeCover,
    codes: &[Token],
    row_offsets: &[O],
    check: Check<'_>,
    out: &mut Vec<usize>,
) {
    if config.skip_empty_packing {
        both_stages::<S, O>(cover, codes, row_offsets, check, out)
    } else {
        both_stages::<P, O>(cover, codes, row_offsets, check, out)
    }
}

/// Execute the selected matcher with the shared row resolver.
/// Planning must supply an eligible cover shape and an available instruction set;
/// this function does not repeat those checks.
pub(super) fn run<O: Offset>(
    config: MatcherConfig,
    cover: &ProbeCover,
    codes: &[Token],
    row_offsets: &[O],
    check: Check<'_>,
    out: &mut Vec<usize>,
) {
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = config;
        both_stages::<matcher::Table, O>(cover, codes, row_offsets, check, out);
    }
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    match config.kind {
        MatcherKind::Table => {
            both_stages::<matcher::Table, O>(cover, codes, row_offsets, check, out)
        }
        MatcherKind::EqOr => with_skip::<O, matcher::EqOr<true>, matcher::EqOr<false>>(
            config,
            cover,
            codes,
            row_offsets,
            check,
            out,
        ),
        MatcherKind::Range => with_skip::<O, matcher::Range<true>, matcher::Range<false>>(
            config,
            cover,
            codes,
            row_offsets,
            check,
            out,
        ),
        MatcherKind::NibbleN8 => match cover.n_points().div_ceil(PER_BATCH) {
            1 => with_skip::<O, matcher::NibbleN8<1, true>, matcher::NibbleN8<1, false>>(
                config,
                cover,
                codes,
                row_offsets,
                check,
                out,
            ),
            2 => with_skip::<O, matcher::NibbleN8<2, true>, matcher::NibbleN8<2, false>>(
                config,
                cover,
                codes,
                row_offsets,
                check,
                out,
            ),
            3 => with_skip::<O, matcher::NibbleN8<3, true>, matcher::NibbleN8<3, false>>(
                config,
                cover,
                codes,
                row_offsets,
                check,
                out,
            ),
            _ => both_stages::<matcher::Table, O>(cover, codes, row_offsets, check, out),
        },
    }
}

/// Return the compiled kernel family supported by this CPU, or the scalar fallback.
/// AVX-512 requires compile-time enablement as well as runtime CPU support.
pub(in crate::search::substring) fn detect_isa() -> Isa {
    #[cfg(target_arch = "aarch64")]
    let isa = Isa::Neon;
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512bw")))]
    let isa = if std::is_x86_feature_detected!("avx2") {
        Isa::Avx2
    } else {
        Isa::Scalar
    };
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    let isa = if std::is_x86_feature_detected!("avx512bw") && std::is_x86_feature_detected!("avx2")
    {
        Isa::Avx512Bw
    } else {
        Isa::Scalar
    };
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let isa = Isa::Scalar;
    isa
}
