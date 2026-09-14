// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A plan as the two type parameters [`both_stages`] wants.

use super::super::plan::facts::{Isa, Kernel, ResolverKind, ScanPlan, TargetCaps, VectorMatcher};
use super::matcher::{self, Matcher};
use super::{Check, both_stages, resolver};
use crate::core::offset::Offset;
use crate::core::types::Token;
use crate::search::substring::ProbeCover;

/// The resolver half of the dispatch.
fn with_resolver<O: Offset, M: Matcher>(
    plan: ScanPlan,
    cover: &ProbeCover,
    codes: &[Token],
    row_offsets: &[O],
    check: Check<'_>,
    out: &mut Vec<usize>,
) {
    match plan.resolver {
        ResolverKind::LinearSeek => {
            both_stages::<M, resolver::LinearSeek<'_, O>>(cover, codes, row_offsets, check, out)
        }
        ResolverKind::GallopSeek => {
            both_stages::<M, resolver::GallopSeek<'_, O>>(cover, codes, row_offsets, check, out)
        }
    }
}

/// The flag half: the same kernel compiled with the pack skipped, `S`, and
/// without, `P`.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn with_skip<O: Offset, S: Matcher, P: Matcher>(
    plan: ScanPlan,
    cover: &ProbeCover,
    codes: &[Token],
    row_offsets: &[O],
    check: Check<'_>,
    out: &mut Vec<usize>,
) {
    if matches!(
        plan.kernel,
        Kernel::Neon { skip: true, .. }
            | Kernel::Avx2 { skip: true, .. }
            | Kernel::Avx512Bw { skip: true, .. }
    ) {
        with_resolver::<O, S>(plan, cover, codes, row_offsets, check, out)
    } else {
        with_resolver::<O, P>(plan, cover, codes, row_offsets, check, out)
    }
}

/// The planned kernel pair as the type parameters [`both_stages`] wants.
pub(super) fn run<O: Offset>(
    plan: ScanPlan,
    cover: &ProbeCover,
    codes: &[Token],
    row_offsets: &[O],
    check: Check<'_>,
    out: &mut Vec<usize>,
) {
    let selected = match plan.kernel {
        Kernel::Empty => return,
        Kernel::Table => {
            return with_resolver::<O, matcher::Table>(plan, cover, codes, row_offsets, check, out);
        }
        Kernel::Neon { matcher, .. } => {
            assert_eq!(detect_target_caps().isa, Isa::Neon);
            matcher
        }
        Kernel::Avx2 { matcher, .. } => {
            assert_eq!(detect_target_caps().isa, Isa::Avx2);
            matcher
        }
        Kernel::Avx512Bw { matcher, .. } => {
            assert_eq!(detect_target_caps().isa, Isa::Avx512Bw);
            matcher
        }
    };
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    unreachable!("this target compiles only scalar execution: {selected:?}");
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    match selected {
        #[cfg(target_arch = "aarch64")]
        VectorMatcher::OnePoint => {
            with_skip::<O, matcher::OnePoint<true>, matcher::OnePoint<false>>(
                plan,
                cover,
                codes,
                row_offsets,
                check,
                out,
            )
        }
        #[cfg(target_arch = "x86_64")]
        VectorMatcher::OnePoint => unreachable!("the one-point specialization is NEON-only"),
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        VectorMatcher::EqOr => with_skip::<O, matcher::EqOr<true>, matcher::EqOr<false>>(
            plan,
            cover,
            codes,
            row_offsets,
            check,
            out,
        ),
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        VectorMatcher::Range => with_skip::<O, matcher::Range<true>, matcher::Range<false>>(
            plan,
            cover,
            codes,
            row_offsets,
            check,
            out,
        ),
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        VectorMatcher::NibbleN8 { batches } => match batches {
            1 => with_skip::<O, matcher::NibbleN8<1, true>, matcher::NibbleN8<1, false>>(
                plan,
                cover,
                codes,
                row_offsets,
                check,
                out,
            ),
            2 => with_skip::<O, matcher::NibbleN8<2, true>, matcher::NibbleN8<2, false>>(
                plan,
                cover,
                codes,
                row_offsets,
                check,
                out,
            ),
            3 => with_skip::<O, matcher::NibbleN8<3, true>, matcher::NibbleN8<3, false>>(
                plan,
                cover,
                codes,
                row_offsets,
                check,
                out,
            ),
            _ => unreachable!("the planner caps the batches at MAX_BATCHES"),
        },
    }
}

/// Detect only targets whose kernels this build contains. Optional AVX-512
/// multiversion activation is deferred pending native x86 qualification.
pub(in crate::search::substring) fn detect_target_caps() -> TargetCaps {
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
    TargetCaps { isa }
}
