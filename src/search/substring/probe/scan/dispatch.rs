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

use super::ProbeCover;
use super::matcher::{self, Matcher};
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use super::{PER_BATCH, matcher::Kernels};

/// The vector instruction set the kernels and cost weights are chosen for.
/// Scans use `detect_isa`; planning tests may pass any value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Isa {
    /// No vector kernels: only the table matcher is available.
    Scalar,
    Neon,
    Avx2,
    Avx512Bw,
}

/// The kernel that tests codes for membership in the cover.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MatcherKind {
    /// Scalar lookup table; supports any cover and any target.
    Table,
    /// One vector equality per point, plus range checks.
    EqOr,
    /// Nibble shuffle tables, eight points per batch, plus range checks.
    Nibble,
}

/// The most batches planning gives Nibble, and `prepare_with` builds a matcher for
/// each count up to it. Up to it each batch costs the same; past it the cost per
/// batch can jump, as on AVX2 after the fourth.
pub(crate) const MAX_NIBBLE_BATCHES: usize = 4;

/// Builds the `kind` matcher of `isa` for `cover`. `kind` must be eligible for the
/// cover on `isa`, as planning ensures. `Isa::Scalar`, and a vector instruction set
/// this build has no kernels for, get the table.
///
/// # Panics
/// If `isa` is a vector instruction set this build has kernels for, and this CPU
/// does not support it.
pub(crate) fn prepare(isa: Isa, kind: MatcherKind, cover: &ProbeCover) -> Box<dyn Matcher> {
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
        _ => Box::new(unsafe { matcher::Table::new(cover) }),
    }
}

/// Builds the `kind` matcher among `K`'s, one instruction set's matchers.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn prepare_with<K: Kernels>(kind: MatcherKind, cover: &ProbeCover) -> Box<dyn Matcher> {
    // Every matcher built below relies on this check.
    assert!(
        supports(K::ISA),
        "vector kernels on a CPU without their instructions"
    );
    // SAFETY: the CPU supports `K::ISA`, the instruction set of every matcher
    // below except the table, which needs none.
    unsafe {
        match kind {
            MatcherKind::Table => Box::new(matcher::Table::new(cover)),
            MatcherKind::EqOr => Box::new(K::EqOr::new(cover)),
            MatcherKind::Nibble => match cover.points().len().div_ceil(PER_BATCH) {
                1 => Box::new(K::Nibble::<1>::new(cover)),
                2 => Box::new(K::Nibble::<2>::new(cover)),
                3 => Box::new(K::Nibble::<3>::new(cover)),
                4 => Box::new(K::Nibble::<4>::new(cover)),
                batches => unreachable!("Nibble planned with {batches} batches, above the cap"),
            },
        }
    }
}

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
