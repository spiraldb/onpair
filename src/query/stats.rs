// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compact per-code frequency summary.
//!
//! [`ContainsSearcher::compile`](crate::query::ContainsSearcher::compile)
//! picks its prefilter anchor by sampling the column's code stream, which
//! requires the codes to be resident at query-compile time. [`CodeStats`] is
//! the alternative: one byte per dictionary token — about a tenth of the
//! dictionary bytes, a fraction of a percent of the column — captured once
//! when the column is built (the compressor already touches every code) and
//! stored alongside it. Compiling with stats never reads the code stream, so
//! a searcher can be built from the dictionary alone — e.g. to decide
//! whether a row group is worth reading at all.
//!
//! Counts are quantized to their binary magnitude (`0`, or `1 + floor(log2
//! n)`), a within-2× per-code approximation. That is plenty for anchor
//! selection, which chooses between candidate sets whose hit rates differ by
//! orders of magnitude; the choice only affects speed, never correctness.

/// Per-token frequency magnitudes for one code stream: entry `c` is `0` when
/// code `c` never occurs, else `1 + floor(log2(count))`.
pub struct CodeStats {
    magnitudes: Box<[u8]>,
}

impl CodeStats {
    /// Tally `codes` (one pass) and quantize. `ntokens` must match the
    /// dictionary the codes index.
    ///
    /// ## Panics
    ///
    /// Panics if a code is `>= ntokens`.
    pub fn from_codes(ntokens: usize, codes: &[u16]) -> Self {
        let mut counts = vec![0u64; ntokens];
        for &c in codes {
            counts[c as usize] += 1;
        }
        Self::from_counts(&counts)
    }

    /// Quantize exact per-token counts (entry per dictionary token).
    pub fn from_counts(counts: &[u64]) -> Self {
        let magnitudes = counts
            .iter()
            .map(|&n| if n == 0 { 0 } else { 1 + n.ilog2() as u8 })
            .collect();
        Self { magnitudes }
    }

    /// Rebuild from bytes previously taken via [`as_bytes`](Self::as_bytes)
    /// (the serialized form is the raw magnitude array).
    pub fn from_bytes(magnitudes: &[u8]) -> Self {
        Self {
            magnitudes: magnitudes.into(),
        }
    }

    /// The serialized form: one magnitude byte per dictionary token.
    pub fn as_bytes(&self) -> &[u8] {
        &self.magnitudes
    }

    /// Number of dictionary tokens covered.
    pub fn num_tokens(&self) -> usize {
        self.magnitudes.len()
    }

    /// Approximate occurrence count of `code`: `0`, or `2^(magnitude - 1)`
    /// (within 2× of the true count).
    pub(crate) fn approx_count(&self, code: usize) -> u64 {
        match self.magnitudes[code] {
            0 => 0,
            q => 1u64 << (q - 1),
        }
    }

    /// Approximate total code count (sum of per-token approximations).
    pub(crate) fn approx_total(&self) -> u64 {
        (0..self.magnitudes.len())
            .map(|c| self.approx_count(c))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantization_within_2x() {
        let counts = [0u64, 1, 2, 3, 1000, 1 << 40];
        let stats = CodeStats::from_counts(&counts);
        for (c, &n) in counts.iter().enumerate() {
            let approx = stats.approx_count(c);
            if n == 0 {
                assert_eq!(approx, 0);
            } else {
                assert!(approx <= n && n < approx * 2, "count {n} approx {approx}");
            }
        }
    }

    #[test]
    fn from_codes_matches_from_counts_and_roundtrips() {
        let codes = [0u16, 0, 0, 2, 2, 5];
        let stats = CodeStats::from_codes(8, &codes);
        let expect = CodeStats::from_counts(&[3, 0, 2, 0, 0, 1, 0, 0]);
        assert_eq!(stats.as_bytes(), expect.as_bytes());
        assert_eq!(stats.num_tokens(), 8);
        let rt = CodeStats::from_bytes(stats.as_bytes());
        assert_eq!(rt.as_bytes(), stats.as_bytes());
    }
}
