// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scans of a narrow code stream.

#[cfg(target_arch = "aarch64")]
mod neon;

use super::ScanInput;
use crate::core::offset::Offset;

/// Append the ascending rows holding a covered code.
#[inline]
pub(super) fn scan<O: Offset>(
    input: ScanInput<'_, O, u8>,
    sparse_row_mapping: bool,
    out: &mut Vec<usize>,
) {
    #[cfg(target_arch = "aarch64")]
    {
        neon::scan(input, sparse_row_mapping, out);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = sparse_row_mapping;
        scan_scalar(input, out);
    }
}

/// Row-at-a-time membership walk, and the oracle the vector kernels answer to.
#[cfg(any(not(target_arch = "aarch64"), test))]
pub(super) fn scan_scalar<O: Offset>(input: ScanInput<'_, O, u8>, out: &mut Vec<usize>) {
    let ScanInput {
        codes,
        row_offsets,
        cover,
    } = input;
    for row in 0..row_offsets.len().saturating_sub(1) {
        let a = row_offsets[row].to_usize();
        let b = row_offsets[row + 1].to_usize();
        if codes[a..b].iter().any(|&code| cover.table[code as usize]) {
            out.push(row);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Token;
    use crate::search::prefilter::cover::ProbeCover;

    /// A stream and row layer with uneven rows, empty rows, and a tail that is
    /// not a multiple of any vector width.
    fn corpus() -> (Vec<u8>, Vec<u32>) {
        let codes: Vec<u8> = (0..1_603u32)
            .map(|index| (index * 37 + index / 13) as u8)
            .collect();
        let mut offsets = vec![0u32];
        let mut offset = 0;
        while offset < codes.len() {
            if offsets.len().is_multiple_of(17) {
                offsets.push(offset as u32);
            }
            offset = (offset + (offsets.len() * 11 % 23) + 1).min(codes.len());
            offsets.push(offset as u32);
        }
        (codes, offsets)
    }

    fn cover_of(covered: impl Fn(u8) -> bool) -> ProbeCover {
        let mut table = vec![false; usize::from(Token::MAX) + 1];
        for code in 0..=u8::MAX {
            table[code as usize] = covered(code);
        }
        ProbeCover::from_membership(table)
    }

    /// TEMPORARY measurement harness.
    #[test]
    #[ignore]
    fn measure() {
        use std::time::Instant;
        const ROW_LEN: usize = 12;
        const ROWS: usize = (8 << 20) / ROW_LEN;
        const CODES: usize = ROWS * ROW_LEN;
        /// Rows the prefilter is expected to admit, at the top of its range.
        const TARGET_ROW_HITS: f64 = 0.02;

        // Background codes never collide with a probe, so selectivity is set by
        // how often a probe is sprinkled in rather than by the domain size.
        let background = |i: usize| 100 + (i % 100) as u8;
        let stream = |points: &[u8]| -> Vec<u8> {
            let per_probe = TARGET_ROW_HITS / ROW_LEN as f64 / points.len() as f64;
            let cutoff = (per_probe * points.len() as f64 * f64::from(u32::MAX)) as u32;
            (0..CODES)
                .map(|i| {
                    let hash = ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as u32;
                    if hash < cutoff {
                        points[i % points.len()]
                    } else {
                        background(i)
                    }
                })
                .collect()
        };
        // Exponentially distributed row lengths with the same mean: short rows
        // dominate, long ones are rare. An even column is where an interpolated
        // entry point is exact, so the spread is what it has to survive.
        let skewed = || {
            let mut offsets = vec![0u32];
            let mut offset = 0usize;
            let mut state = 0x243F_6A88_85A3_08D3u64;
            while offset < CODES {
                state = state
                    .wrapping_mul(0x5851_F42D_4C95_7F2D)
                    .wrapping_add(0x1405_7B7E_F767_814F);
                let uniform = ((state >> 32) as f64 + 1.0) / (f64::from(u32::MAX) + 2.0);
                let len = 1 + (-(ROW_LEN as f64 - 1.0) * uniform.ln()) as usize;
                offset = (offset + len).min(CODES);
                offsets.push(offset as u32);
            }
            offsets
        };
        let layouts = [
            (
                "even",
                (0..=ROWS)
                    .map(|r| (r * ROW_LEN) as u32)
                    .collect::<Vec<u32>>(),
            ),
            ("skewed", skewed()),
        ];

        for (layout, row_offsets) in &layouts {
            let rows = row_offsets.len() - 1;
            let longest = row_offsets
                .windows(2)
                .map(|pair| pair[1] - pair[0])
                .max()
                .unwrap_or(0);
            println!(
                "--- {layout} rows: {rows} rows, mean {:.1}, longest {longest}",
                CODES as f64 / rows as f64
            );
            for (name, probes) in [
                ("0 hits", &[][..]),
                ("1 point", &[7][..]),
                ("2 points", &[7, 40][..]),
                ("3 points", &[7, 40, 70][..]),
                ("4 points", &[7, 40, 70, 95][..]),
            ] {
                let cover =
                    cover_of(|code| probes.contains(&code) || (probes.is_empty() && code == 9));
                let codes = stream(if probes.is_empty() { &[9] } else { probes });
                let codes = if probes.is_empty() {
                    (0..CODES).map(background).collect()
                } else {
                    codes
                };
                let wide: Vec<u16> = codes.iter().map(|&code| u16::from(code)).collect();
                let input = ScanInput::full(&codes, row_offsets, &cover);
                let mut out = Vec::with_capacity(rows);
                let mut time = |run: &mut dyn FnMut(&mut Vec<usize>)| {
                    let mut best = f64::MAX;
                    let mut hits = 0;
                    for _ in 0..20 {
                        out.clear();
                        let start = Instant::now();
                        run(&mut out);
                        best = best.min(start.elapsed().as_secs_f64());
                        hits = out.len();
                    }
                    (best, hits)
                };
                let (scalar, a) = time(&mut |out| scan_scalar(input, out));
                let (vector, b) = time(&mut |out| scan(input, false, out));
                let (sparse, d) = time(&mut |out| scan(input, true, out));
                let (u16_neon, c) = time(&mut |out| {
                    super::super::scan_neon(&wide, row_offsets, &cover, false, out)
                });
                assert_eq!((a, a, a), (b, c, d));
                let cmp = cover.points().len() + 2 * cover.ranges().len();
                println!(
                    "{name:>9}: {cmp:>2} cmp/vec {:>5.2}% rows | scalar {:>5.2}GB/s | u8 linear {:>5.2}GB/s | u8 sparse {:>5.2}GB/s | u16 linear {:>5.2}GB/s | sparse/u16 {:>4.2}x",
                    100.0 * a as f64 / rows as f64,
                    CODES as f64 / scalar / 1e9,
                    CODES as f64 / vector / 1e9,
                    CODES as f64 / sparse / 1e9,
                    2.0 * CODES as f64 / u16_neon / 1e9,
                    (CODES as f64 / sparse) / (2.0 * CODES as f64 / u16_neon),
                );
            }
        }
    }

    #[test]
    fn kernel_matches_the_scalar_oracle() {
        let (codes, row_offsets) = corpus();
        let covers = [
            cover_of(|code| code == 7),
            cover_of(|code| code == 0 || code == 255),
            cover_of(|code| matches!(code, 7 | 40 | 200)),
            cover_of(|code| matches!(code, 0 | 7 | 40 | 255)),
            cover_of(|code| matches!(code, 1 | 2 | 4 | 8 | 16 | 32)),
            cover_of(|code| code < 16),
            cover_of(|code| (32..=64).contains(&code)),
            cover_of(|code| code == 9 || (32..=64).contains(&code)),
            cover_of(|code| code % 17 == 0),
            cover_of(|_| true),
            cover_of(|_| false),
        ];
        for cover in &covers {
            let input = ScanInput::full(&codes, &row_offsets, cover);
            let mut expected = Vec::new();
            scan_scalar(input, &mut expected);
            for sparse_row_mapping in [false, true] {
                let mut got = Vec::new();
                scan(input, sparse_row_mapping, &mut got);
                assert_eq!(got, expected);
            }
        }
    }

    /// Hits thousands of rows apart drive the sparse walk deep into its gallop,
    /// where the dense corpus above never leaves the first bracket.
    #[test]
    fn sparse_gaps_match_the_scalar_oracle() {
        for row_len in [1usize, 3, 12, 257] {
            for gap in [1usize, 2, 7, 64, 1_000, 30_011] {
                let rows = 4_096;
                let codes: Vec<u8> = (0..rows * row_len)
                    .map(|index| if index % gap == 0 { 7 } else { 200 })
                    .collect();
                let row_offsets: Vec<u32> = (0..=rows).map(|row| (row * row_len) as u32).collect();
                let cover = cover_of(|code| code == 7);
                let input = ScanInput::full(&codes, &row_offsets, &cover);

                let mut expected = Vec::new();
                scan_scalar(input, &mut expected);
                for sparse_row_mapping in [false, true] {
                    let mut got = Vec::new();
                    scan(input, sparse_row_mapping, &mut got);
                    assert_eq!(got, expected, "row_len {row_len}, gap {gap}");
                }
            }
        }
    }
}
