// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Instantiate the planned kernel, so the block loop runs on concrete types.
//!
//! The matcher kind, the packing policy and the number of nibble batches are
//! resolved once per scan into type and const parameters.
//!
//! AArch64 always has NEON. An x86 build contains the AVX2 kernels, or the AVX-512
//! ones if `avx512bw` is enabled at compile time, and falls back to the table when
//! the CPU lacks them. Other targets only have the table.

use super::matcher;
use super::{Isa, MatcherKind, ScanInput, both_stages};
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use super::{PER_BATCH, matcher::Matcher};

use crate::core::offset::Offset;

/// Runs `S`, the variant that skips packing empty groups, or `P`, the one that
/// always packs.
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

/// Scans `input` with the `kind` kernel and appends the verified rows to `out`.
/// `kind` must be eligible for the cover on this CPU, as planning ensures; this is
/// not checked again.
pub(in crate::search::substring::probe) fn run<O: Offset>(
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

/// The compiled kernel family if this CPU supports it, otherwise `Isa::Scalar`.
pub(in crate::search::substring::probe) fn detect_isa() -> Isa {
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
