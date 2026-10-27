// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Instantiate the planned kernel, so the block loop runs on concrete types.
//!
//! The instruction set, the matcher kind, the packing policy and the number of
//! nibble batches are resolved once per scan into type and const parameters.
//!
//! AArch64 always has NEON. An x86_64 build contains both the AVX2 and the AVX-512
//! kernels and picks the best one the CPU supports at runtime, falling back to the
//! table without either. Other targets only have the table.

use super::matcher;
use super::{Isa, MatcherKind, ScanInput, both_stages};
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use super::{PER_BATCH, matcher::Kernels, matcher::Matcher};

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

/// Scans `input` with the `kind` kernel of `isa` and appends the verified rows to
/// `out`. `isa` must be supported by this CPU and `kind` eligible for the cover on
/// it, as planning ensures; neither is checked again.
pub(in crate::search::substring::probe) fn run<O: Offset>(
    isa: Isa,
    kind: MatcherKind,
    skip_empty_packing: bool,
    input: ScanInput<'_, O>,
    out: &mut Vec<usize>,
) {
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = (kind, skip_empty_packing);
    match isa {
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => run_with::<matcher::Neon, O>(kind, skip_empty_packing, input, out),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => run_with::<matcher::Avx2, O>(kind, skip_empty_packing, input, out),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512Bw => run_with::<matcher::Avx512, O>(kind, skip_empty_packing, input, out),
        _ => both_stages::<matcher::Table, O>(input, out),
    }
}

/// Scans `input` with the `kind` kernel among `K`'s, one instruction set's matchers.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn run_with<K: Kernels, O: Offset>(
    kind: MatcherKind,
    skip_empty_packing: bool,
    input: ScanInput<'_, O>,
    out: &mut Vec<usize>,
) {
    debug_assert!(
        supports(K::ISA),
        "{:?} kernels on a CPU without them",
        K::ISA
    );
    match kind {
        MatcherKind::Table => both_stages::<matcher::Table, O>(input, out),
        MatcherKind::EqOr if input.cover.n_points() == 0 => {
            with_skip::<O, K::EqOr<true, false>, K::EqOr<false, false>>(
                skip_empty_packing,
                input,
                out,
            )
        }
        MatcherKind::EqOr => with_skip::<O, K::EqOr<true, true>, K::EqOr<false, true>>(
            skip_empty_packing,
            input,
            out,
        ),
        MatcherKind::NibbleN8 => match input.cover.n_points().div_ceil(PER_BATCH) {
            1 => with_skip::<O, K::NibbleN8<1, true>, K::NibbleN8<1, false>>(
                skip_empty_packing,
                input,
                out,
            ),
            2 => with_skip::<O, K::NibbleN8<2, true>, K::NibbleN8<2, false>>(
                skip_empty_packing,
                input,
                out,
            ),
            3 => with_skip::<O, K::NibbleN8<3, true>, K::NibbleN8<3, false>>(
                skip_empty_packing,
                input,
                out,
            ),
            _ => both_stages::<matcher::Table, O>(input, out),
        },
    }
}

/// Whether this build has kernels for `isa` and this CPU can run them.
pub(in crate::search::substring::probe) fn supports(isa: Isa) -> bool {
    match isa {
        Isa::Scalar => true,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => true,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => std::is_x86_feature_detected!("avx2"),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512Bw => {
            std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw")
        }
        _ => false,
    }
}

/// The fastest instruction set this CPU supports: AVX-512 before AVX2 on x86_64,
/// NEON on AArch64, and `Isa::Scalar` without any of them.
pub(in crate::search::substring::probe) fn detect_isa() -> Isa {
    [Isa::Avx512Bw, Isa::Avx2, Isa::Neon]
        .into_iter()
        .find(|&isa| supports(isa))
        .unwrap_or(Isa::Scalar)
}
