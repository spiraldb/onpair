// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Connect a prepared matcher and packing policy to their concrete implementation.
//!
//! Dispatch derives nibble batches from the cover and resolves them, together
//! with the packing policy, once before scanning, so the block loop runs with
//! concrete type and const parameters.
//!
//! AArch64 uses NEON. On x86, this build contains AVX2 kernels unless `avx512bw`
//! is enabled at compile time, in which case it contains the AVX-512 versions.
//! Runtime detection enables the compiled family when supported; otherwise
//! selection uses the scalar table. Other architectures use the table too.

use super::matcher;
use super::{Isa, MatcherKind, ScanInput, both_stages};
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use super::{PER_BATCH, matcher::Matcher};

use crate::core::offset::Offset;

/// Select whether empty groups skip mask packing: `S` enables it, `P` disables it.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn with_skip<O: Offset, S: Matcher, P: Matcher>(
    skip_empty_packing: bool,
    input: ScanInput<'_, O>,
    out: &mut Vec<usize>,
) {
    if skip_empty_packing {
        both_stages::<S, O>(input, out)
    } else {
        both_stages::<P, O>(input, out)
    }
}

/// Execute the prepared matcher and append the rows the input's hit verifier accepts.
/// `skip_empty_packing` applies to vector matchers. Preparation must supply a
/// matcher eligible for the cover and the current CPU; this function does not
/// repeat those checks.
pub(in crate::search::substring) fn run<O: Offset>(
    kind: MatcherKind,
    skip_empty_packing: bool,
    input: ScanInput<'_, O>,
    out: &mut Vec<usize>,
) {
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = (kind, skip_empty_packing);
        both_stages::<matcher::Table, O>(input, out);
    }
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    match kind {
        MatcherKind::Table => both_stages::<matcher::Table, O>(input, out),
        MatcherKind::EqOr if input.cover.n_points() == 0 => {
            with_skip::<O, matcher::EqOr<true, false>, matcher::EqOr<false, false>>(
                skip_empty_packing,
                input,
                out,
            )
        }
        MatcherKind::EqOr => with_skip::<O, matcher::EqOr<true>, matcher::EqOr<false>>(
            skip_empty_packing,
            input,
            out,
        ),
        MatcherKind::NibbleN8 => match input.cover.n_points().div_ceil(PER_BATCH) {
            1 => with_skip::<O, matcher::NibbleN8<1, true>, matcher::NibbleN8<1, false>>(
                skip_empty_packing,
                input,
                out,
            ),
            2 => with_skip::<O, matcher::NibbleN8<2, true>, matcher::NibbleN8<2, false>>(
                skip_empty_packing,
                input,
                out,
            ),
            3 => with_skip::<O, matcher::NibbleN8<3, true>, matcher::NibbleN8<3, false>>(
                skip_empty_packing,
                input,
                out,
            ),
            _ => both_stages::<matcher::Table, O>(input, out),
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
