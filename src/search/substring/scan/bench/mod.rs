// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Timing harness for the split scan, and the fits that turn its rows into
//! the constants in `policy`.
//!
//! ```text
//! cargo test --release --lib scan::bench::matcher_fit::sweep  -- --ignored --nocapture
//! cargo test --release --lib scan::bench::resolver_fit::sweep -- --ignored --nocapture
//! ```
//!
//! [`matcher_fit`] is stage one, a matcher against the probe shapes a cover
//! arrives in. [`resolver_fit`] is stage two, a resolver against one mask per
//! needle set, so what is timed is the mask-to-rows half alone. Each `sweep`
//! writes one timestamped CSV to `bench/output/`, which the marimo dashboards
//! out of tree read, then fits the model to its rows and prints the
//! block to paste into `policy`; each `refit` does the same from the newest
//! CSV, or the one `MASK_CSV` or `RESOLVE_CSV` names, so a constant is a
//! measurement and not a memory.
//!
//! A u16 code weighs two bytes, so `gbs` and `gcodes` part company on the
//! onpair encodings: compare kernels to each other on `gcodes`, and to what
//! the memory system can deliver on `gbs`.

mod loader;
#[cfg(test)]
mod matcher_fit;
#[cfg(test)]
mod resolver_fit;
mod utils;

use std::time::Instant;

use super::matcher::Matcher;
use super::resolver::Resolver;
use super::{BLOCK, Check, Isa, Mask, blocks};
use crate::core::types::Token;

/// Timed passes per measurement; the fastest is the one to keep.
const PASSES: usize = 5;

/// Seconds of the fastest of [`PASSES`] runs, after one untimed. A run the
/// clock cannot resolve is repeated inside the timed region until it can, and
/// the time divided back: the sparsest masks leave stage two a few hits in a
/// megabyte, which is tens of nanoseconds of work and was reported as an
/// infinite rate. A run already past the floor repeats once, as before.
fn best(run: &mut dyn FnMut()) -> f64 {
    /// Timed region short enough that the clock's own step shows up in it.
    const FLOOR: f64 = 1e-6;
    run();
    let mut reps = 1;
    loop {
        let mut best = f64::INFINITY;
        for _ in 0..PASSES {
            let start = Instant::now();
            for _ in 0..reps {
                run();
            }
            best = best.min(start.elapsed().as_secs_f64());
        }
        if best > FLOOR || reps == 1 << 20 {
            return best / f64::from(reps);
        }
        reps *= 2;
    }
}

/// Stage one over a whole stream, block by block, mask thrown away. The buffer
/// is reused, so what is timed is the matcher and the mask write the seam
/// costs, not an allocator. The empty-block flag is dropped rather than acted
/// on, so a kernel that raises it is still timed on all its own work.
fn mask_stream<M: Matcher>(matcher: &M, codes: &[Token], bits: &mut Mask) {
    blocks(codes, &mut |block, _, _| {
        matcher.check(block, bits);
    });
}

/// Stage two over a whole stream, from a mask built once: the blocks a
/// matcher would report non-empty, in order, into a buffer the passes reuse.
/// Clearing that buffer and appending to it is the resolver's own cost and is
/// timed with it.
fn resolve_stream<'a, R: Resolver<'a>>(
    mask: &[u64],
    hit: &[usize],
    row_offsets: &'a [R::Offset],
    out: &mut Vec<usize>,
) {
    out.clear();
    let mut resolver = R::new(row_offsets);
    for &block in hit {
        let bits = &mask[block * (BLOCK / 64)..][..BLOCK / 64];
        resolver.rows(
            bits.try_into().unwrap(),
            block * BLOCK,
            Check::Superset,
            out,
        );
    }
}

/// Machine identity for the CSV: architecture and CPU model, commas stripped.
/// The machine and the set its kernels were built for, which is the key a
/// fit groups rows by: an AVX2 and an AVX-512 build of this crate run
/// different kernels on the same core.
fn machine() -> String {
    let model = cpu_model().unwrap_or_else(|| "unknown cpu".to_string());
    format!(
        "{} {model} {}",
        std::env::consts::ARCH,
        isa_name(Isa::BUILT)
    )
    .replace(',', " ")
    .split_whitespace()
    .collect::<Vec<_>>()
    .join(" ")
}

/// The CPU as the platform will name it. `/proc/cpuinfo` carries a `model
/// name` on x86 but not on aarch64, where it has only the implementer and
/// part number and `lscpu` is what resolves those to a name; macOS has
/// neither and answers through `sysctl`.
fn cpu_model() -> Option<String> {
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    labelled(&cpuinfo, "model name")
        .or_else(|| labelled(&run("lscpu", &[]), "Model name"))
        .or_else(|| trimmed(run("sysctl", &["-n", "machdep.cpu.brand_string"])))
}

/// The value of the first `label: value` line, which is how both
/// `/proc/cpuinfo` and `lscpu` answer.
fn labelled(text: &str, label: &str) -> Option<String> {
    text.lines()
        .filter_map(|line| line.split_once(':'))
        .find(|(name, _)| name.trim() == label)
        .and_then(|(_, value)| trimmed(value.to_string()))
}

fn run(program: &str, arg: &[&str]) -> String {
    std::process::Command::new(program)
        .args(arg)
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .unwrap_or_default()
}

fn trimmed(text: String) -> Option<String> {
    let text = text.trim().to_string();
    (!text.is_empty()).then_some(text)
}

/// What a bench writes in its machine column and a fit reads back.
fn isa_name(isa: Isa) -> &'static str {
    match isa {
        Isa::Neon => "neon",
        Isa::Avx2 => "avx2",
        Isa::Avx512Bw => "avx512bw",
        Isa::Scalar => "scalar",
    }
}

/// The set a machine column names, `None` where it names none.
fn isa_named(name: &str) -> Option<Isa> {
    [Isa::Neon, Isa::Avx2, Isa::Avx512Bw, Isa::Scalar]
        .into_iter()
        .find(|isa| isa_name(*isa) == name)
}

/// The `cfg` a build of this set sits behind, for a fit to print above the
/// function it emits. Must agree with the `cfg`s on the functions in `policy`.
fn isa_cfg(isa: Isa) -> &'static str {
    match isa {
        Isa::Neon => "#[cfg(target_arch = \"aarch64\")]",
        Isa::Avx2 => "#[cfg(all(target_arch = \"x86_64\", not(target_feature = \"avx512bw\")))]",
        Isa::Avx512Bw => "#[cfg(all(target_arch = \"x86_64\", target_feature = \"avx512bw\"))]",
        Isa::Scalar => "#[cfg(not(any(target_arch = \"aarch64\", target_arch = \"x86_64\")))]",
    }
}
