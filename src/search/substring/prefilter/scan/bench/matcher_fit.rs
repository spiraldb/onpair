// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Stage one: every kernel of this build against every probe shape, and the
//! matcher rows of `policy::ns_per_code` fitted to what that measured.
//!
//! A probe is K tokens from the catalog, R ranges spread over the code space,
//! or both, since a cover arrives as points and runs together. R is the axis
//! a range's cost moves with, so it is swept; the codes per range are the
//! control on the claim that a range costs the same however many codes lie
//! between its ends.
//!
//! The fit prints, per machine, the coefficients, the fit's own error against
//! the rows, and the compiled model's error against the same rows. A compiled
//! error near the fit's is a model that still describes the machine; anything
//! larger is a constant to move. The machine column carries the instruction
//! set the kernels were built for, so an AVX2 and an AVX-512 run of one core
//! do not share a group.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use super::super::matcher::{EqOr, Range};
use super::super::matcher::{Matcher, Table};
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use super::super::matcher::{NibbleN8, PER_BATCH};
use super::super::policy::{BYTES_PER_CODE, Match, Shape, ns_per_code, takes};
use super::super::{BLOCK, Check, Isa, both_stages, resolver};
use super::loader::{
    CHECK_CODES, CODES, NeedleSet, SAMPLE, STREAMS, WIDE, code, load_corpus, load_needles, paths,
    row_layer,
};
use super::utils::{error, file_name, line, read_csv, slope, write_csv};
use super::{best, isa_cfg, isa_name, isa_named, machine, mask_stream};
use crate::core::types::Token;
use crate::core::types::TokenRange;
use crate::search::substring::prefilter::ProbeCover;

/// Ranges per probe. R is the axis a range's cost is expected to move with,
/// so it is swept and everything else about a range is a control.
const RANGE_COUNTS: &[usize] = &[1, 2, 3, 4, 6, 8];

/// Codes per range. A range holds its two ends in a register whatever lies
/// between them, so this is the control on that: a width the code space
/// cannot fit R disjoint copies of is skipped.
const RANGE_WIDTHS: &[usize] = &[1, 16, 256, 4096];

/// Token counts a range is measured beside, one from each band
/// `select_matcher` picks a kernel for. The ranges' cost should not move with
/// K, so this is the subsample that checks it rather than a cross product.
const RANGED_TOKEN_COUNTS: &[usize] = &[1, 8, 16];

/// And the one selectivity bin of those, since what a range costs is not the
/// stream's business either.
const RANGED_TARGET: f64 = 0.01;

/// Codes per range there, from the middle of [`RANGE_WIDTHS`].
const RANGED_WIDTH: usize = 16;

/// One measurement. `count` is K and `length` L, both zero where a probe is
/// ranges alone; `ranges` is R and `width` the codes in each, both zero
/// where it is tokens alone.
#[derive(Serialize, Deserialize)]
struct Row {
    stream: String,
    encoding: String,
    codes: usize,
    rows: usize,
    matcher: String,
    length: usize,
    count: usize,
    ranges: usize,
    width: usize,
    /// The selectivity the needle set was built for, which is the density
    /// the pack-skipping flag decides over.
    target: f64,
    achieved: f64,
    prefix_selectivity: f64,
    gbs: f64,
    gcodes: f64,
    machine: String,
}

impl Row {
    /// What the fit works in. A rate is its reciprocal.
    fn ns(&self) -> f64 {
        BYTES_PER_CODE / self.gbs
    }

    fn shape(&self) -> Shape {
        Shape {
            tokens: self.count,
            ranges: self.ranges,
        }
    }

    /// What each kernel's cost is linear in: K for the compare, the batch
    /// count for the bitmap batches, R for the range kernel, nothing for the
    /// table.
    fn regressor(&self) -> Option<(f64, &'static str)> {
        match self.matcher.as_str() {
            "eq_or" => Some((self.count as f64, "K")),
            "nibble_n8k" => Some((self.count.max(1).div_ceil(8) as f64, "B")),
            "range" => Some((self.ranges as f64, "R")),
            "table" => Some((0.0, "")),
            _ => None,
        }
    }
}

/// What one measurement probes for: a token set from the catalog, R
/// ranges spread over the code space, or both. One shape for all three,
/// so every kernel meets every combination through the same call.
#[derive(Debug)]
struct Probe<'a> {
    set: Option<&'a NeedleSet>,
    /// Empty where the probe is tokens alone.
    ranges: Vec<TokenRange>,
    /// Codes per range, zero where there are none.
    width: usize,
}

impl Probe<'_> {
    fn cover(&self) -> ProbeCover {
        let points = self.set.map_or(vec![], |set| set.needles.to_vec());
        ProbeCover::new(points, self.ranges.clone())
    }
}

/// R disjoint ranges of `width` codes, spread evenly over the code space,
/// or `None` where they do not fit in it.
fn spread_ranges(count: usize, width: usize) -> Option<Vec<TokenRange>> {
    let stride = (1usize << Token::BITS) / count;
    if width > stride {
        return None;
    }
    Some(
        (0..count)
            .map(|at| {
                let lo = at * stride;
                TokenRange {
                    begin: code(lo as u32),
                    last: code((lo + width - 1) as u32),
                }
            })
            .collect(),
    )
}

/// Every probe of one encoding: the catalog's token sets, ranges on their
/// own at every R and width that fits, and ranges beside the tokens of
/// [`RANGED_TOKEN_COUNTS`].
fn probes<'a>(sets: &'a [NeedleSet]) -> Vec<Probe<'a>> {
    let mut probes: Vec<Probe<'a>> = sets
        .iter()
        .map(|set| Probe {
            set: Some(set),
            ranges: Vec::new(),
            width: 0,
        })
        .collect();
    for &count in RANGE_COUNTS {
        for &width in RANGE_WIDTHS {
            if let Some(ranges) = spread_ranges(count, width) {
                probes.push(Probe {
                    set: None,
                    ranges,
                    width,
                });
            }
        }
    }
    let beside = sets.iter().filter(|set| {
        RANGED_TOKEN_COUNTS.contains(&set.count) && (set.target - RANGED_TARGET).abs() < 1e-9
    });
    for set in beside {
        for &count in RANGE_COUNTS {
            if let Some(ranges) = spread_ranges(count, RANGED_WIDTH) {
                probes.push(Probe {
                    set: Some(set),
                    ranges,
                    width: RANGED_WIDTH,
                });
            }
        }
    }
    probes
}

/// Seconds for stage one over the codes, and the rows a whole scan finds
/// over the checked prefix and its row layer, or `None` if the kernel
/// declined the probe.
type Run = fn(Match, &Probe<'_>, &[Token], &[Token], &[u32]) -> Option<(f64, Vec<usize>)>;

/// One kernel as the harness sees it, so they can be held in one list and
/// taken in turn on each probe.
struct Kernel {
    name: &'static str,
    kind: Match,
    run: Run,
}

fn kernel<M: Matcher>(kind: Match, name: &'static str) -> Kernel {
    Kernel {
        name,
        kind,
        run: run::<M>,
    }
}

fn run<M: Matcher>(
    kind: Match,
    probe: &Probe<'_>,
    codes: &[Token],
    checked: &[Token],
    check_rows: &[u32],
) -> Option<(f64, Vec<usize>)> {
    let cover = probe.cover();
    if !takes(kind, Shape::of(&cover)) {
        return None;
    }
    let matcher = M::new(&cover);
    let mut bits = [0u64; BLOCK / 64];
    let seconds = best(&mut || mask_stream(&matcher, codes, &mut bits));
    let mut found = Vec::new();
    both_stages::<M, resolver::LinearSeek<'_, u32>>(
        &cover,
        checked,
        check_rows,
        Check::Superset,
        &mut found,
    );
    Some((seconds, found))
}

/// `NibbleN8` at the batch count K needs, `ceil(K / 8)`, so one row
/// shows what a kernel that sizes itself to the cover would run at.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn nibble_n8k<const SKIP: bool>(name: &'static str) -> Kernel {
    fn run_8k<const SKIP: bool>(
        kind: Match,
        probe: &Probe<'_>,
        codes: &[Token],
        checked: &[Token],
        check_rows: &[u32],
    ) -> Option<(f64, Vec<usize>)> {
        match probe.cover().points().len().div_ceil(PER_BATCH) {
            1 => run::<NibbleN8<1, SKIP>>(kind, probe, codes, checked, check_rows),
            2 => run::<NibbleN8<2, SKIP>>(kind, probe, codes, checked, check_rows),
            3 => run::<NibbleN8<3, SKIP>>(kind, probe, codes, checked, check_rows),
            _ => None,
        }
    }
    Kernel {
        name,
        kind: Match::NibbleN8K,
        run: run_8k::<SKIP>,
    }
}

fn kernels() -> Vec<Kernel> {
    #[allow(unused_mut)]
    let mut all = vec![kernel::<Table>(Match::Table, "table")];
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    all.extend([
        kernel::<EqOr<false>>(Match::EqOr, "eq_or"),
        kernel::<EqOr<true>>(Match::EqOr, "eq_or_skip"),
        kernel::<Range<false>>(Match::Range, "range"),
        kernel::<Range<true>>(Match::Range, "range_skip"),
        nibble_n8k::<false>("nibble_n8k"),
        nibble_n8k::<true>("nibble_n8k_skip"),
    ]);
    all
}

/// Every kernel on every probe of one encoding of one stream. Every kernel is
/// timed on one probe before any is timed on the next, so a machine that
/// drifts over a run drifts under all of them equally.
fn measure(stream: &str, encoding: &str, kernels: &[Kernel], machine: &str, out: &mut Vec<Row>) {
    let (corpus_path, needles_path) = paths(stream, encoding);
    let corpus = load_corpus(&corpus_path);
    let codes = &corpus.codes[..CODES.min(corpus.codes.len())];
    // The row layer, cut to the codes being scanned and closed over the
    // row the cut fell inside: a bit set for the last code needs a row,
    // and a range sets one for every code.
    let row_offsets = row_layer(&corpus.row_offsets, codes.len());
    let sets: Vec<NeedleSet> = load_needles(&needles_path)
        .into_iter()
        .filter(|set| set.sample == SAMPLE)
        .collect();
    let probes = probes(&sets);
    println!(
        "\n{stream} {encoding}: {} codes, {} rows, {} probes over {} needle sets",
        codes.len(),
        row_offsets.len() - 1,
        probes.len(),
        sets.len()
    );
    let checked = &codes[..CHECK_CODES.min(codes.len())];
    let check_rows = row_layer(&corpus.row_offsets, checked.len());
    for probe in &probes {
        // Untimed, and the reason the catalog is loaded rather than
        // needles invented: the rows found here track the selectivity
        // the set was built for. Only loosely, since this is a prefix
        // of the stream and a column like `hits` clusters. Worked out from
        // the cover, so a kernel is never checked against a sibling.
        let cover = probe.cover();
        let expected: Vec<usize> = (0..check_rows.len() - 1)
            .filter(|&row| {
                let (from, to) = (check_rows[row] as usize, check_rows[row + 1] as usize);
                checked[from..to].iter().any(|&code| cover.contains(code))
            })
            .collect();
        for kernel in kernels {
            let Some((seconds, found)) =
                (kernel.run)(kernel.kind, probe, codes, checked, &check_rows)
            else {
                continue;
            };
            assert_eq!(
                found, expected,
                "{} differs from the cover on {probe:?}",
                kernel.name
            );
            let (length, count, target, achieved) = match probe.set {
                Some(set) => (1, set.count, set.target, set.achieved),
                None => (0, 0, 0.0, 0.0),
            };
            out.push(Row {
                stream: stream.to_string(),
                encoding: encoding.to_string(),
                codes: codes.len(),
                rows: row_offsets.len() - 1,
                matcher: kernel.name.to_string(),
                length,
                count,
                ranges: probe.ranges.len(),
                width: probe.width,
                target,
                achieved,
                prefix_selectivity: found.len() as f64 / (check_rows.len() - 1) as f64,
                gbs: (codes.len() * size_of::<Token>()) as f64 / seconds / 1e9,
                gcodes: codes.len() as f64 / seconds / 1e9,
                machine: machine.to_string(),
            });
        }
    }
}

/// One machine's coefficients, in the shape [`ns_per_code`] wants them: an
/// intercept and a slope per kernel, and what a range adds to the two
/// kernels that fold them in.
#[derive(Clone, Copy, Default)]
struct Fit {
    eq_or: Option<(f64, f64)>,
    range: Option<(f64, f64)>,
    nibble_n8k: Option<(f64, f64)>,
    table: Option<f64>,
    /// Per range, on top of the compare's own line, and on top of the
    /// bitmap batches'.
    beside: f64,
    beside_n8k: f64,
    /// The fastest any kernel ran, which no model term describes.
    ceiling: f64,
}

/// The variant this build compiled, so the model can be asked what it
/// predicts. `None` where this build has no such kernel.
fn compiled(matcher: &str) -> Option<Match> {
    Some(match matcher {
        "table" => Match::Table,
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        "eq_or" => Match::EqOr,
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        "range" => Match::Range,
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        "nibble_n8k" => Match::NibbleN8K,
        _ => return None,
    })
}

/// The rows of one kernel with no ranges beside its tokens, which is the
/// line the ranges are then fitted on top of.
fn alone<'a>(row: &[&'a Row], kernel: &str) -> Vec<&'a Row> {
    row.iter()
        .filter(|row| row.matcher == kernel && (row.ranges == 0 || kernel == "range"))
        .copied()
        .collect()
}

fn points(row: &[&Row]) -> Vec<(f64, f64)> {
    row.iter()
        .filter_map(|row| row.regressor().map(|(x, _)| (x, row.ns())))
        .collect()
}

/// The fits as `policy::ns_per_code` would have them, to paste under the
/// `cfg` for the build the rows came from. Only the kernels the sweep
/// measured are emitted; the rest keep whatever the model says now.
fn snippet(machine: &str, source: &str, fit: &Fit) {
    // The set is the last word of the machine column, and it decides which
    // function this block is.
    let isa = machine
        .rsplit(' ')
        .next()
        .and_then(isa_named)
        .unwrap_or(Isa::Scalar);
    println!("\n/// Fitted on {machine}, from {source}.");
    println!("{}", isa_cfg(isa));
    println!(
        "fn {}(matcher: Match, shape: Shape) -> f64 {{",
        isa_name(isa)
    );
    println!("    let k = shape.tokens as f64;");
    println!("    let r = shape.ranges as f64;");
    println!("    let batches = shape.tokens.div_ceil(PER_BATCH) as f64;");
    println!("    match matcher {{");
    if let Some(table) = fit.table {
        println!("        Match::Table => {table:.3},");
    }
    // One arm per kernel, which is the shape the model has and the order
    // `README.md` tabulates them in.
    if let Some((a, b)) = fit.eq_or {
        println!(
            "        Match::EqOr => {a:.5} + {b:.5} * k + {:.5} * r,",
            fit.beside
        );
    }
    if let Some((a, b)) = fit.nibble_n8k {
        let batches = match b {
            0.0 => String::new(),
            slope => format!(" + {slope:.5} * batches"),
        };
        println!(
            "        Match::NibbleN8K => {a:.5}{batches} + {:.5} * r,",
            fit.beside_n8k
        );
    }
    if let Some((a, b)) = fit.range {
        println!("        Match::Range => {a:.5} + {b:.5} * r,");
    }
    println!("    }}");
    println!("}}");
    println!(
        "// nothing above {:.1} GB/s was reachable, so clamp with \
         `.max({:.5})` if that binds.",
        BYTES_PER_CODE / fit.ceiling,
        fit.ceiling
    );
}

/// The model fitted to `rows`, per machine, against the model compiled in.
fn fit(rows: &[Row], source: &str) {
    let mut group: BTreeMap<&str, Vec<&Row>> = BTreeMap::new();
    for row in rows {
        group.entry(&row.machine).or_default().push(row);
    }

    let mut fitted: BTreeMap<&str, Fit> = BTreeMap::new();
    for (&machine, row) in &group {
        let codes = row.iter().map(|row| row.codes).max().unwrap_or(0);
        println!("\n{machine}  up to {codes} codes");
        let mut fit = Fit::default();
        println!(
            "  {:<13} {:<26} {:>6} {:>7}",
            "kernel", "fitted ns/code", "fit%", "model%"
        );
        for kernel in ["eq_or", "range", "nibble_n8k", "table"] {
            // Ranges are fitted on top of the kernel without them, so the
            // rows that have both are held back for the line below.
            let alone = alone(row, kernel);
            let point = points(&alone);
            if point.is_empty() {
                continue;
            }
            let axis = alone[0].regressor().unwrap().1;
            let (a, b) = line(&point);
            // A kernel measured at one value of its own axis is flat as far
            // as this run knows.
            let flat = point.iter().all(|&(x, _)| x == point[0].0);
            match kernel {
                "eq_or" => fit.eq_or = Some((a, b)),
                "range" => fit.range = Some((a, b)),
                "nibble_n8k" => fit.nibble_n8k = Some((a, if flat { 0.0 } else { b })),
                _ => fit.table = Some(a),
            }
            let fitted = match axis {
                _ if flat => format!("{a:.5}"),
                axis => format!("{a:.5} + {b:.5}*{axis}"),
            };
            let model = compiled(kernel)
                .map(|kernel| {
                    let by_shape: Vec<(f64, f64)> = alone
                        .iter()
                        .map(|row| (ns_per_code(kernel, row.shape()), row.ns()))
                        .collect();
                    error(&by_shape, |predicted| predicted)
                })
                .unwrap_or(f64::NAN);
            println!(
                "  {kernel:<13} {fitted:<26} {:>6.1} {model:>7.1}",
                error(&point, |x| a + b * x)
            );
        }

        // What a range adds to a token kernel: the residual over that
        // kernel's own line, per range, on the rows carrying both.
        for kernel in ["eq_or", "nibble_n8k"] {
            let alone = points(&alone(row, kernel));
            if alone.is_empty() {
                continue;
            }
            let (a, b) = line(&alone);
            let point: Vec<(f64, f64)> = row
                .iter()
                .filter(|row| row.matcher == kernel && row.ranges > 0 && row.count > 0)
                .filter_map(|row| {
                    let (x, _) = row.regressor()?;
                    Some((row.ranges as f64, row.ns() - (a + b * x)))
                })
                .filter(|&(_, residual)| residual > 0.0)
                .collect();
            if point.len() > 1 {
                let per_range = slope(&point);
                println!("  {kernel} with ranges: {per_range:.5} per range");
                match kernel {
                    "eq_or" => fit.beside = per_range,
                    _ => fit.beside_n8k = per_range,
                }
            }
        }

        // The fastest any kernel ran at the largest stream in the file. No
        // model term describes it, and on a core whose kernels share a
        // ceiling it is the only number that matters: nothing the model
        // predicts above this rate is reachable there.
        fit.ceiling = row
            .iter()
            .filter(|row| row.codes == codes)
            .map(|row| row.ns())
            .fold(f64::MAX, f64::min);
        println!(
            "  ceiling at {codes} codes: {:.2} GB/s, so no ns/code below {:.5}",
            BYTES_PER_CODE / fit.ceiling,
            fit.ceiling
        );

        // The pack-skipping flag, as the two ends the model is fitted to:
        // what it costs where every group matches, and what it saves where
        // none does.
        for kernel in ["eq_or", "nibble_n8k"] {
            let mut delta: BTreeMap<String, (f64, usize)> = BTreeMap::new();
            for skip in row
                .iter()
                .filter(|row| row.matcher == format!("{kernel}_skip"))
            {
                let plain = row.iter().find(|row| {
                    row.matcher == kernel
                        && (row.length, row.count, row.ranges)
                            == (skip.length, skip.count, skip.ranges)
                        && row.target == skip.target
                });
                if let Some(plain) = plain {
                    let cell = delta.entry(format!("{:.5}", skip.target)).or_default();
                    cell.0 += skip.ns() - plain.ns();
                    cell.1 += 1;
                }
            }
            let ends: Vec<String> = delta
                .iter()
                .map(|(target, (sum, count))| format!("{target} {:+.5}", sum / *count as f64))
                .collect();
            if !ends.is_empty() {
                println!(
                    "  {kernel} skip flag, ns/code by selectivity: {}",
                    ends.join("  ")
                );
            }
        }
        fitted.insert(machine, fit);
    }

    for (&machine, fit) in &fitted {
        snippet(machine, source, fit);
    }
}

/// Sweep, write the CSV, fit.
#[test]
#[ignore]
fn sweep() {
    let machine = machine();
    let mut rows = Vec::new();
    for &stream in STREAMS {
        for &encoding in WIDE {
            measure(stream, encoding, &kernels(), &machine, &mut rows);
        }
    }
    let path = write_csv("novel_mask", &rows);
    println!("wrote {}", path.display());
    fit(&rows, &file_name(&path));
}

/// Fit the newest `novel_mask_*.csv`, or the one `MASK_CSV` names.
#[test]
#[ignore]
fn refit() {
    let (path, rows) = read_csv::<Row>("MASK_CSV", "novel_mask_");
    println!("{}\n{} rows", path.display(), rows.len());
    fit(&rows, &file_name(&path));
}
