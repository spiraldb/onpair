// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Where a sweep's rows go and come back from, and the least squares every
//! fit runs on them.

use std::path::{Path, PathBuf};

use serde::Serialize;
use serde::de::DeserializeOwned;

pub(super) fn output_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("bench/output")
}

/// Writes `rows` to a timestamped `bench/output/{prefix}_*.csv`, one column
/// per field in declaration order, and returns the path.
pub(super) fn write_csv<R: Serialize>(prefix: &str, rows: &[R]) -> PathBuf {
    let dir = output_dir();
    std::fs::create_dir_all(&dir).unwrap();
    let stamp = std::process::Command::new("date")
        .arg("+%Y-%m-%d_%H-%M-%S")
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|out| out.trim().to_string())
        .filter(|stamp| !stamp.is_empty())
        .unwrap_or_else(|| "unstamped".to_string());
    let path = dir.join(format!("{prefix}_{stamp}.csv"));
    let mut out = csv::Writer::from_path(&path).unwrap();
    for row in rows {
        out.serialize(row).unwrap();
    }
    out.flush().unwrap();
    path
}

/// The CSV `var` names, else the newest `bench/output/{prefix}_*.csv`, read
/// back as rows. Nothing is rebuilt to read another machine's numbers: copy
/// its CSV over and point `var` at it.
pub(super) fn read_csv<R: DeserializeOwned>(var: &str, prefix: &str) -> (PathBuf, Vec<R>) {
    let path = match std::env::var(var) {
        Ok(path) => PathBuf::from(path),
        Err(_) => {
            let mut file: Vec<PathBuf> = std::fs::read_dir(output_dir())
                .unwrap_or_else(|_| panic!("run the {prefix} sweep first"))
                .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                .filter(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.starts_with(prefix) && name.ends_with(".csv"))
                })
                .collect();
            file.sort();
            file.pop()
                .unwrap_or_else(|| panic!("run the {prefix} sweep first"))
        }
    };
    let rows = csv::Reader::from_path(&path)
        .unwrap()
        .deserialize()
        .collect::<Result<Vec<R>, _>>()
        .unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    (path, rows)
}

pub(super) fn file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("a sweep")
        .to_string()
}

/// `a + b·x` minimising the relative error, since a model is read as a rate
/// and an absolute fit would chase the slow rows.
pub(super) fn line(point: &[(f64, f64)]) -> (f64, f64) {
    let (mut sw, mut swx, mut swy, mut swxx, mut swxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for &(x, y) in point {
        let w = 1.0 / (y * y);
        sw += w;
        swx += w * x;
        swy += w * y;
        swxx += w * x * x;
        swxy += w * x * y;
    }
    let spread = sw * swxx - swx * swx;
    if spread.abs() < 1e-12 {
        return (swy / sw, 0.0);
    }
    let slope = (sw * swxy - swx * swy) / spread;
    ((swy - slope * swx) / sw, slope)
}

/// A slope through the origin, for what one term adds to a model already
/// fitted without it.
pub(super) fn slope(point: &[(f64, f64)]) -> f64 {
    let (mut swxy, mut swxx) = (0.0, 0.0);
    for &(x, y) in point {
        let w = 1.0 / (y * y);
        swxy += w * x * y;
        swxx += w * x * x;
    }
    if swxx.abs() < 1e-12 { 0.0 } else { swxy / swxx }
}

/// Least squares of `y ≈ Σ βᵢ·xᵢ` on relative error: every row is scaled by
/// its own `y`, so each measurement counts the same. Normal equations, solved
/// by elimination.
pub(super) fn lstsq<const N: usize>(row: &[([f64; N], f64)]) -> [f64; N] {
    let mut a = [[0.0; N]; N];
    let mut b = [0.0; N];
    for (x, y) in row {
        for i in 0..N {
            b[i] += x[i] / y;
            for j in 0..N {
                a[i][j] += x[i] * x[j] / (y * y);
            }
        }
    }
    for col in 0..N {
        let pivot = (col..N)
            .max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))
            .unwrap();
        a.swap(col, pivot);
        b.swap(col, pivot);
        if a[col][col] == 0.0 {
            continue;
        }
        for i in (0..N).filter(|&i| i != col) {
            let f = a[i][col] / a[col][col];
            for j in 0..N {
                a[i][j] -= f * a[col][j];
            }
            b[i] -= f * b[col];
        }
    }
    std::array::from_fn(|i| if a[i][i] == 0.0 { 0.0 } else { b[i] / a[i][i] })
}

/// Mean relative error of `pred` over the points, in percent.
pub(super) fn error(point: &[(f64, f64)], pred: impl Fn(f64) -> f64) -> f64 {
    if point.is_empty() {
        return f64::NAN;
    }
    let sum: f64 = point.iter().map(|&(x, y)| (pred(x) / y - 1.0).abs()).sum();
    100.0 * sum / point.len() as f64
}
