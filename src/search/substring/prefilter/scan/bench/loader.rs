// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The corpora and needle catalogs, read from `bench/data`, where the
//! out-of-tree examples write them: `bench_corpus` the streams, one file per
//! encoding, `bench_needles` a catalog beside each drawn from that encoding's own
//! alphabet.

use std::path::{Path, PathBuf};

use crate::core::types::Token;

/// Streams to measure, as `<db>/<table>/<column>_<size>`.
pub(super) const STREAMS: &[&str] = &["ch/hits/URL_1m"];

/// Encodings of them, each with a catalog beside it.
pub(super) const WIDE: &[&str] = &["onpair12", "onpair16"];

/// Codes per measurement. A prefix is enough for a throughput number.
pub(super) const CODES: usize = 4 << 20;

/// Codes each kernel's rows are checked against the cover's own over: a
/// sixteenth of the prefix, since the check is a smoke test over real codes
/// and bit-exactness is `matcher::tests`' job.
pub(super) const CHECK_CODES: usize = CODES / 16;

/// Needle samples to measure. The three sets of a bin are interchangeable for
/// timing, so one is enough until a bin looks noisy.
pub(super) const SAMPLE: usize = 0;

/// One code stream, rows delimited as they are in the file.
pub(super) struct Corpus {
    pub(super) codes: Vec<Token>,
    pub(super) row_offsets: Vec<u32>,
}

/// One needle set from a catalog, with the selectivity it was built for and
/// the one it measured.
#[derive(Debug)]
pub(super) struct NeedleSet {
    pub(super) count: usize,
    pub(super) target: f64,
    pub(super) achieved: f64,
    pub(super) sample: usize,
    /// The K codes, the layout `ProbeCover` takes.
    pub(super) needles: Vec<Token>,
}

pub(super) fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("bench/data")
}

/// `<db>/<table>/<column>_<size>` and an encoding to the stream and its
/// catalog.
pub(super) fn paths(stream: &str, encoding: &str) -> (PathBuf, PathBuf) {
    let base = data_dir().join(stream);
    (
        base.with_extension(format!("{encoding}.csv")),
        base.with_extension(format!("{encoding}.needles.csv")),
    )
}

/// One code as the corpus files spell it, decimal and unsigned.
pub(super) fn code(value: u32) -> Token {
    Token::try_from(value).expect("code wider than the stream")
}

/// Parse `1,2,3\n4,5\n`: one line is one row, one field is one code.
pub(super) fn load_corpus(path: &Path) -> Corpus {
    let text = std::fs::read(path).unwrap_or_else(|_| {
        panic!(
            "missing {}; run the out-of-tree bench_corpus example",
            path.display()
        )
    });
    let mut codes = Vec::with_capacity(text.len() / 2);
    let mut row_offsets = vec![0u32];
    let (mut value, mut digits) = (0u32, false);
    for &byte in &text {
        match byte {
            b'0'..=b'9' => {
                value = value * 10 + u32::from(byte - b'0');
                digits = true;
            }
            b',' | b'\n' => {
                if digits {
                    codes.push(code(value));
                }
                if byte == b'\n' {
                    row_offsets.push(codes.len() as u32);
                }
                value = 0;
                digits = false;
            }
            _ => panic!("unexpected byte in {}", path.display()),
        }
    }
    Corpus { codes, row_offsets }
}

/// Parse a needle catalog: `l,k,target,achieved,rows,sample,codes`, needles
/// separated by `|` and their codes by spaces. Only the one-code sets, which
/// are the only shape a kernel takes.
pub(super) fn load_needles(path: &Path) -> Vec<NeedleSet> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|_| {
        panic!(
            "missing {}; run the out-of-tree bench_needles example",
            path.display()
        )
    });
    text.lines()
        .skip(1)
        .filter(|line| line.starts_with("1,"))
        .map(|line| {
            let field: Vec<&str> = line.split(',').collect();
            NeedleSet {
                count: field[1].parse().unwrap(),
                target: field[2].parse().unwrap(),
                achieved: field[3].parse().unwrap(),
                sample: field[5].parse().unwrap(),
                needles: field[6]
                    .split(['|', ' '])
                    .map(|field| code(field.parse().unwrap()))
                    .collect(),
            }
        })
        .collect()
}

/// The corpus row layer cut to the codes being scanned, ending with a row
/// for the value the cut fell inside, so every code a bit can be set for
/// has a row: what stage two is promised.
pub(super) fn row_layer(row_offsets: &[u32], codes: usize) -> Vec<u32> {
    let mut layer: Vec<u32> = row_offsets
        .iter()
        .copied()
        .take_while(|&offset| offset as usize <= codes)
        .collect();
    if *layer.last().unwrap() < codes as u32 {
        layer.push(codes as u32);
    }
    layer
}
