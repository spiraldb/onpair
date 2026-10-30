// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The planned matcher, built once per query.
//!
//! The instruction set, the matcher kind and the number of Nibble batches are
//! resolved into a concrete matcher behind `dyn Matcher`, which every scan of the
//! query then calls once per block. Whether to skip packing pairs of groups without
//! hits is decided per scan, and passed to the matcher.
//!
//! AArch64 always has NEON. An x86_64 build contains both the AVX2 and the AVX-512
//! kernels and picks the best one the CPU supports at runtime, falling back to the
//! table without either. Other targets only have the table.

use std::sync::Arc;

use super::matcher::{self, Matcher};
use super::{Isa, MatcherKind, ProbeCover};
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use super::{PER_BATCH, matcher::Kernels};
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use crate::search::substring::probe::plan::max_nibble_batches;

/// Builds the `kind` matcher of `isa` for `cover`. `kind` must be eligible for the
/// cover on `isa`, as planning ensures. `Isa::Scalar`, and a vector instruction set
/// this build has no kernels for, get the table.
///
/// # Panics
/// If `isa` is a vector instruction set this build has kernels for, and this CPU
/// does not support it.
pub(crate) fn prepare(isa: Isa, kind: MatcherKind, cover: &ProbeCover) -> Arc<dyn Matcher> {
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = kind;
    match isa {
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => prepare_with::<matcher::Neon>(kind, cover),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => prepare_with::<matcher::Avx2>(kind, cover),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512Bw => prepare_with::<matcher::Avx512>(kind, cover),
        // SAFETY: the table runs on any CPU.
        _ => Arc::new(unsafe { matcher::Table::new(cover) }),
    }
}

/// Builds the `kind` matcher among `K`'s, one instruction set's matchers.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn prepare_with<K: Kernels>(kind: MatcherKind, cover: &ProbeCover) -> Arc<dyn Matcher> {
    // Every matcher built below relies on this check.
    assert!(
        supports(K::ISA),
        "vector kernels on a CPU without their instructions"
    );
    // SAFETY: the CPU supports `K::ISA`, the instruction set of every matcher
    // below except the table, which needs none.
    unsafe {
        match kind {
            MatcherKind::Table => Arc::new(matcher::Table::new(cover)),
            MatcherKind::EqOr => Arc::new(K::EqOr::new(cover)),
            // An arm is compiled only if planning can give Nibble that many batches on
            // this instruction set.
            MatcherKind::Nibble => match cover.points().len().div_ceil(PER_BATCH) {
                1 => Arc::new(K::Nibble::<1>::new(cover)),
                2 if const { max_nibble_batches(K::ISA) >= 2 } => {
                    Arc::new(K::Nibble::<2>::new(cover))
                }
                3 if const { max_nibble_batches(K::ISA) >= 3 } => {
                    Arc::new(K::Nibble::<3>::new(cover))
                }
                4 if const { max_nibble_batches(K::ISA) >= 4 } => {
                    Arc::new(K::Nibble::<4>::new(cover))
                }
                batches => unreachable!("Nibble planned with {batches} batches, above the cap"),
            },
        }
    }
}

// Every batch count planning can give Nibble has an arm in `prepare_with`.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const _: () = assert!(
    max_nibble_batches(Isa::Neon) <= 4
        && max_nibble_batches(Isa::Avx2) <= 4
        && max_nibble_batches(Isa::Avx512Bw) <= 4,
    "a Nibble batch cap exceeds the batch counts `prepare_with` compiles"
);

/// Whether this build has kernels for `isa` and this CPU can run them.
pub(crate) fn supports(isa: Isa) -> bool {
    match isa {
        Isa::Scalar => true,
        // The NEON kernels read the bytes of `u16` lanes in little-endian order.
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => {
            cfg!(target_endian = "little") && std::arch::is_aarch64_feature_detected!("neon")
        }
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
pub(crate) fn detect_isa() -> Isa {
    [Isa::Avx512Bw, Isa::Avx2, Isa::Neon]
        .into_iter()
        .find(|&isa| supports(isa))
        .unwrap_or(Isa::Scalar)
}
