// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Stage two: the resolvers against one mask per needle set, built once from
//! the cover itself so no matcher runs while the clock does, and the four
//! constants of `policy::stage_two_ns` fitted to what that measured.
//!
//! The sweep's axes are the mask's hit density and the row length, swept
//! past each other because that is what decides which resolver wins. Rates
//! are over the code stream, so the two halves of one scan add up; in the
//! sparsest bin, where a mask holds a handful of bits, a rate says "no work"
//! rather than a speed, and the fit leaves such rows out.
//!
//! What each resolver's time is linear in, and the work it counts:
//!
//! ```text
//! words     mask words in the blocks that hit, read by both
//! emitted   rows the resolver names
//! crossed   rows the cursor passes to reach them; LinearSeek steps each
//! searched  emitted · log2(1 + crossed / emitted); GallopSeek halves each
//! ```
//!
//! The word rate is one constant shared by both resolvers, so the two are
//! fitted together with the other resolver's terms zeroed on each row.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::super::policy::{Resolve, seek_ns_per_row, stage_two_ns};
use super::super::{BLOCK, blocks, clear_from, resolver};
use super::loader::{NeedleSet, SAMPLE, load_corpus, load_needles, paths, row_layer};
use super::utils::{error, file_name, lstsq, read_csv, write_csv};
use super::{best, machine, resolve_stream};
use crate::core::types::Token;
use crate::search::prefilter::ProbeCover;

/// Streams for stage two, picked for their mean row length: about seven codes
/// a row and about eighty. Only one encoding of each, `onpair16`: a wider code
/// reaches stage two through the codes a row holds and nothing else, and
/// [`MERGES`] sweeps that anyway.
const STREAMS: &[&str] = &["imdb/name/name_1m", "ch/hits/URL_1m"];

/// Row layers per stream: every `factor` rows fused into one. The mask and
/// the length distribution stay as they are, so this is the row count on its
/// own, from hundreds of rows to a block down to one row over several.
const MERGES: &[usize] = &[1, 4, 16, 64];

/// Needle counts to draw masks from. K reaches stage two only through the
/// density of the mask it leaves behind, so two of the eight bins span it.
const COUNTS: &[usize] = &[1, 16];

/// Codes per measurement. The resolvers cost per bit and per row, so a
/// shorter prefix than stage one's is just as steady a number, and a mask
/// has to be built for every needle set first.
const CODES: usize = 1 << 20;

/// One measurement of one resolver on one mask over one row layer.
#[derive(Serialize, Deserialize)]
struct Row {
    stream: String,
    codes: usize,
    rows: usize,
    codes_per_row: usize,
    resolver: String,
    length: usize,
    count: usize,
    target: f64,
    achieved: f64,
    /// Set bits per code.
    density: f64,
    /// Share of blocks the matcher would hand over.
    blocks_hit: f64,
    /// Share of rows the mask names.
    selectivity: f64,
    gbs: f64,
    machine: String,
}

impl Row {
    fn resolve(&self) -> Resolve {
        match self.resolver.as_str() {
            "linear_seek" => Resolve::LinearSeek,
            _ => Resolve::GallopSeek,
        }
    }

    /// What one cell of the sweep is: the stream, needle set and row layer.
    fn cell(&self) -> (&str, usize, usize, u64, usize) {
        (
            &self.stream,
            self.length,
            self.count,
            self.target.to_bits(),
            self.codes_per_row,
        )
    }

    /// Nanoseconds over the stream, what the fit works in.
    fn ns(&self) -> f64 {
        self.codes as f64 / self.gbs
    }

    fn words(&self) -> f64 {
        64.0 * self.blocks_hit * self.codes as f64 / BLOCK as f64
    }

    fn emitted(&self) -> f64 {
        self.selectivity * self.rows as f64
    }

    /// The cursor stops at the last hit rather than the last row.
    fn crossed(&self) -> f64 {
        self.rows as f64 * self.emitted() / (self.emitted() + 1.0)
    }

    /// Rows with at least one bit and one row to emit: a mask with neither
    /// is no work, and its rate says nothing about a resolver.
    fn has_work(&self) -> bool {
        self.density * self.codes as f64 >= 1.0 && self.emitted() >= 1.0
    }

    /// The row in the joint model's terms.
    fn terms(&self) -> [f64; 4] {
        let (words, emitted, crossed) = (self.words(), self.emitted(), self.crossed());
        match self.resolve() {
            Resolve::LinearSeek => [words, emitted, crossed, 0.0],
            Resolve::GallopSeek => [words, 0.0, 0.0, emitted * (1.0 + crossed / emitted).log2()],
        }
    }

    fn model(&self) -> f64 {
        stage_two_ns(self.resolve(), self.words(), self.emitted(), self.crossed())
    }
}

/// One bit per code for the whole stream, straight off the cover's own
/// definition of a hit and laid out as the blocks the seam hands over.
fn mask_of(set: &NeedleSet, codes: &[Token]) -> Vec<u64> {
    let cover = ProbeCover::new(set.needles.to_vec(), vec![]);
    let mut mask = Vec::with_capacity(codes.len() / 64);
    let mut bits = [0u64; BLOCK / 64];
    blocks(codes, &mut |block, _, valid| {
        bits.fill(0);
        for (at, &code) in block.iter().enumerate() {
            if cover.contains(code) {
                bits[at / 64] |= 1 << (at % 64);
            }
        }
        clear_from(&mut bits, valid);
        mask.extend_from_slice(&bits);
    });
    mask
}

/// The blocks a matcher would report non-empty. Finding them is stage
/// one's work, so it happens once and outside the timed run.
fn hit_blocks(mask: &[u64]) -> Vec<usize> {
    mask.chunks_exact(BLOCK / 64)
        .enumerate()
        .filter(|(_, bits)| bits.iter().any(|&set| set != 0))
        .map(|(block, _)| block)
        .collect()
}

/// Every `factor` rows of the layer fused into one.
fn merge_rows(row_offsets: &[u32], factor: usize) -> Vec<u32> {
    let mut merged: Vec<u32> = row_offsets.iter().copied().step_by(factor).collect();
    let last = *row_offsets.last().unwrap();
    if *merged.last().unwrap() < last {
        merged.push(last);
    }
    merged
}

/// One resolver over the stream, checked against the rows the mask names
/// before its time counts.
fn resolve<'a, R: resolver::Resolver<'a>>(
    name: &'static str,
    mask: &[u64],
    hit: &[usize],
    row_offsets: &'a [R::Offset],
    expected: &[usize],
) -> (&'static str, f64) {
    let mut out = Vec::new();
    let seconds = best(&mut || resolve_stream::<R>(mask, hit, row_offsets, &mut out));
    assert_eq!(
        out, *expected,
        "{name} differs from the rows the mask names"
    );
    (name, seconds)
}

/// Every resolver on one mask before any is timed on the next, at every row
/// count of [`MERGES`].
fn measure(stream: &str, machine: &str, out: &mut Vec<Row>) {
    let (corpus_path, needles_path) = paths(stream, "onpair16");
    let corpus = load_corpus(&corpus_path);
    // Whole blocks only, so the mask is exactly the blocks it holds.
    let codes = &corpus.codes[..CODES.min(corpus.codes.len()) / BLOCK * BLOCK];
    let layer = row_layer(&corpus.row_offsets, codes.len());
    let sets: Vec<NeedleSet> = load_needles(&needles_path)
        .into_iter()
        .filter(|set| set.sample == SAMPLE && COUNTS.contains(&set.count))
        .collect();
    println!(
        "\n{stream} onpair16: {} codes, {} rows, {} needle sets",
        codes.len(),
        layer.len() - 1,
        sets.len()
    );
    let mut empty = 0;
    for set in &sets {
        let mask = mask_of(set, codes);
        let hit = hit_blocks(&mask);
        let bits: u32 = mask.iter().map(|word| word.count_ones()).sum();
        // No bit in the prefix is no work at all, and no rate: the 0% bin
        // of a catalog is built to hit nothing, and the top bins hold the
        // sets their corpus could not deliver.
        if hit.is_empty() {
            empty += 1;
            continue;
        }
        for &factor in MERGES {
            let row_offsets = merge_rows(&layer, factor);
            let rows = row_offsets.len() - 1;
            let expected = resolver::expected_rows(&mask, &row_offsets);
            for (name, seconds) in [
                resolve::<resolver::LinearSeek<'_, u32>>(
                    "linear_seek",
                    &mask,
                    &hit,
                    &row_offsets,
                    &expected,
                ),
                resolve::<resolver::GallopSeek<'_, u32>>(
                    "gallop_seek",
                    &mask,
                    &hit,
                    &row_offsets,
                    &expected,
                ),
            ] {
                out.push(Row {
                    stream: stream.to_string(),
                    codes: codes.len(),
                    rows,
                    codes_per_row: codes.len() / rows,
                    resolver: name.to_string(),
                    length: 1,
                    count: set.count,
                    target: set.target,
                    achieved: set.achieved,
                    density: f64::from(bits) / codes.len() as f64,
                    blocks_hit: hit.len() as f64 / (mask.len() / (BLOCK / 64)) as f64,
                    selectivity: expected.len() as f64 / rows as f64,
                    gbs: codes.len() as f64 / seconds / 1e9,
                    machine: machine.to_string(),
                });
            }
        }
    }
    if empty > 0 {
        println!("  {empty} of {} sets never hit the prefix", sets.len());
    }
}

/// Rows crossed per emitted row above which the search beats the walk. From
/// g = 2: at g = 1 every row hits and the two constants sit within the fit's
/// error of each other, which is not a crossover.
fn crossover(seek: impl Fn(Resolve, f64) -> f64) -> f64 {
    let mut g = 2.0;
    while g < 1e7 {
        if seek(Resolve::GallopSeek, g) < seek(Resolve::LinearSeek, g) {
            return g;
        }
        g *= 1.001;
    }
    f64::INFINITY
}

/// Mean relative error, in percent, of `predict` over the rows of one
/// resolver.
fn resolver_error(row: &[&Row], resolver: Resolve, predict: impl Fn(&Row) -> f64) -> f64 {
    let point: Vec<(f64, f64)> = row
        .iter()
        .filter(|row| row.resolve() == resolver)
        .map(|row| (predict(row), row.ns()))
        .collect();
    error(&point, |predicted| predicted)
}

/// How the compiled model does against always picking the measured winner,
/// over the cells that measured both resolvers: the share it gets right, and
/// the time given up on average and at worst, in percent.
fn score(row: &[&Row]) -> (f64, f64, f64) {
    let mut cell: BTreeMap<_, Vec<&Row>> = BTreeMap::new();
    for row in row {
        cell.entry(row.cell()).or_default().push(row);
    }
    let (mut right, mut lost, mut worst, mut count) = (0, 0.0, 0.0, 0);
    for pair in cell.into_values().filter(|pair| pair.len() == 2) {
        let picked = pair
            .iter()
            .min_by(|a, b| a.model().total_cmp(&b.model()))
            .unwrap();
        let fastest = pair
            .iter()
            .min_by(|a, b| a.ns().total_cmp(&b.ns()))
            .unwrap();
        let gave_up = picked.ns() / fastest.ns() - 1.0;
        right += usize::from(picked.resolver == fastest.resolver);
        lost += gave_up;
        worst = f64::max(worst, gave_up);
        count += 1;
    }
    let count = count.max(1) as f64;
    (
        100.0 * right as f64 / count,
        100.0 * lost / count,
        100.0 * worst,
    )
}

/// The model fitted to `rows`, per machine, against the model compiled in.
fn fit(rows: &[Row], source: &str) {
    let mut group: BTreeMap<&str, Vec<&Row>> = BTreeMap::new();
    for row in rows.iter().filter(|row| row.has_work()) {
        group.entry(&row.machine).or_default().push(row);
    }
    let mut fitted: BTreeMap<&str, [f64; 4]> = BTreeMap::new();
    for (&machine, row) in &group {
        let point: Vec<([f64; 4], f64)> = row.iter().map(|row| (row.terms(), row.ns())).collect();
        let beta @ [word, per_row, cross, step] = lstsq(&point);
        if beta.iter().any(|&b| b < 0.0) {
            println!(
                "\n{machine}: a term fitted negative, so the fit is borrowing from a collinear one"
            );
        }
        let fit = |row: &Row| {
            row.terms()
                .iter()
                .zip(beta)
                .map(|(x, b)| x * b)
                .sum::<f64>()
        };
        println!("\n{machine}, {} rows with work in them", row.len());
        println!(
            "  {:<12} {:<34} {:>6} {:>7}",
            "resolver", "fitted ns", "fit%", "model%"
        );
        println!("  {:<12} {word:.2} per word", "both");
        for (resolver, name, terms) in [
            (
                Resolve::LinearSeek,
                "linear_seek",
                format!("{per_row:.2} per row + {cross:.2} per crossed"),
            ),
            (
                Resolve::GallopSeek,
                "gallop_seek",
                format!("{step:.2} per halving"),
            ),
        ] {
            println!(
                "  {name:<12} {terms:<34} {:>6.1} {:>7.1}",
                resolver_error(row, resolver, fit),
                resolver_error(row, resolver, Row::model)
            );
        }
        let g_fit = crossover(|resolver, g| match resolver {
            Resolve::LinearSeek => per_row + cross * g,
            Resolve::GallopSeek => step * (1.0 + g).log2(),
        });
        println!(
            "  gallop_seek above g = {g_fit:.0} rows crossed per emitted row, one hit row in {:.3}; the compiled model says {:.0}",
            1.0 / g_fit,
            crossover(seek_ns_per_row)
        );
        let (right, lost, worst) = score(row);
        println!(
            "  the compiled model picks the measured winner in {right:.0}% of cells, giving up {lost:.1}% on average and {worst:.0}% at worst"
        );
        fitted.insert(machine, beta);
    }
    for (&machine, [word, per_row, cross, step]) in &fitted {
        println!("\n/// Fitted on {machine}, from {source}.");
        println!("const WORD_NS: f64 = {word:.2};");
        println!("const LINEAR_SEEK_ROW_NS: f64 = {per_row:.2};");
        println!("const LINEAR_SEEK_CROSS_NS: f64 = {cross:.2};");
        println!("const GALLOP_SEEK_STEP_NS: f64 = {step:.2};");
    }
}

/// Sweep, write the CSV, fit.
#[test]
#[ignore]
fn sweep() {
    let machine = machine();
    let mut rows = Vec::new();
    for &stream in STREAMS {
        measure(stream, &machine, &mut rows);
    }
    let path = write_csv("novel_resolve", &rows);
    println!("wrote {}", path.display());
    fit(&rows, &file_name(&path));
}

/// Fit the newest `novel_resolve_*.csv`, or the one `RESOLVE_CSV` names.
#[test]
#[ignore]
fn refit() {
    let (path, rows) = read_csv::<Row>("RESOLVE_CSV", "novel_resolve_");
    println!("{}\n{} rows", path.display(), rows.len());
    fit(&rows, &file_name(&path));
}
