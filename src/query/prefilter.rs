// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Code-domain prefilter: prove a row *definitely* does not contain the
//! pattern without running the DFA, in the spirit of Hyperscan's literal
//! prefilters — but over dictionary codes instead of raw bytes.
//!
//! ## Necessary condition
//!
//! Fix an *anchor* position `i` in the pattern `P`. In any occurrence of `P`
//! inside a row, some dictionary token covers the byte at pattern position
//! `i`. That token, placed at signed offset `s` relative to the occurrence
//! start (`s <= i < s + len`), must agree with `P` on every byte of overlap
//! — bytes of the token outside `P` are unconstrained. The set of codes for
//! which *some* such alignment agrees is computable from the dictionary alone,
//! so a row whose codes contain **no** candidate for anchor `i` cannot match.
//!
//! At query compile time we build this candidate set for every anchor and keep
//! the one that fires least often, estimated by sampling the column's actual
//! code stream. Mid-pattern anchors need several bytes of agreement, so their
//! sets are usually tiny (a one-byte-suffix anchor like "tokens starting with
//! 'e'" never wins the argmin).
//!
//! ## Scan shape
//!
//! Because the dictionary is lexicographically sorted, candidates ("tokens
//! starting with P\[s..\]") cluster into a handful of contiguous code ranges.
//! When few ranges suffice, membership is a branch-free SIMD range check —
//! 16 codes per AVX2 compare, with each `u16` code standing in for ~3-4 bytes
//! of decompressed text. Otherwise we fall back to an 8 KiB (L1-resident)
//! bitmap probed per code.

/// Sorted, inclusive, non-overlapping code intervals.
type Intervals = Vec<(u16, u16)>;

/// How candidate membership is tested during the scan.
#[derive(Clone)]
enum Kind {
    /// Few intervals: SIMD range checks (AVX2 when available).
    Intervals(Intervals),
    /// Arbitrary candidate set: one bit per possible `u16` code (8 KiB).
    Bitmap(Box<[u64; 1024]>),
}

/// Above this many intervals the per-vector compare chain costs more than the
/// bitmap probe, so the scan switches representation.
const MAX_SCAN_INTERVALS: usize = 16;

/// Sample cap for compile-time anchor selection: paid once per query, so
/// spend more for a steadier estimate.
const COMPILE_SAMPLES: usize = 1 << 16;

/// Sample cap for deferred (per-scan) anchor selection: paid on every scan,
/// so keep the warmup well under the scan itself. Anchor selection only
/// separates hit rates that differ by orders of magnitude, so a few thousand
/// samples are statistically ample; near-zero anchors that tie at 0 hits fall
/// back to the set-size tie-break.
const SCAN_SAMPLES: usize = 1 << 13;

/// A prefilter whose anchor has been chosen: a candidate-membership test plus
/// its estimated firing rate.
#[derive(Clone)]
pub(crate) struct Filter {
    kind: Kind,
    /// Estimated fraction of stream codes that are candidates, from whatever
    /// frequency source chose the anchor (diagnostic; drives nothing at
    /// runtime).
    expected_hit_rate: f64,
}

/// Compiled prefilter for one pattern over one dictionary.
pub(crate) enum Prefilter {
    /// Anchor chosen at compile time, from a code-stream sample or stored
    /// [`crate::query::CodeStats`].
    Fixed(Filter),
    /// Dictionary-only compile: every anchor's candidate set is kept and the
    /// rarest is chosen per scan by sampling the code stream being scanned —
    /// no frequency information is needed before the scan.
    Deferred { sets: Vec<Vec<u64>>, ntokens: usize },
}

/// Per-anchor candidate bitmaps over `ntokens` codes.
fn anchor_candidates(pattern: &[u8], dict_bytes: &[u8], dict_offsets: &[u32]) -> Vec<Vec<u64>> {
    let m = pattern.len();
    let ntokens = dict_offsets.len().saturating_sub(1);
    let words = ntokens.div_ceil(64);
    let mut sets = vec![vec![0u64; words]; m];

    // Every agreeing alignment pins one byte: for `s > 0` the token's first
    // byte must equal `pattern[s]`, and for `s <= 0` the token byte at `-s`
    // must equal `pattern[0]`. Enumerating only those alignments (via the
    // pattern positions of each byte, and the token positions of
    // `pattern[0]`) replaces the all-alignments sweep — compile is on the
    // LIKE-scan path, and this loop over a 64 Ki-token dictionary dominated
    // it.
    let mut ppos: Vec<Vec<u8>> = vec![Vec::new(); 256];
    for (s, &b) in pattern.iter().enumerate().skip(1) {
        ppos[b as usize].push(s as u8);
    }
    let mi = m as isize;
    let mark = |sets: &mut Vec<Vec<u64>>, tok: &[u8], c: usize, s: isize| {
        let lo = s.max(0);
        let hi = (s + tok.len() as isize).min(mi);
        debug_assert!(lo < hi);
        let agrees = (lo..hi).all(|j| tok[(j - s) as usize] == pattern[j as usize]);
        if agrees {
            for i in lo..hi {
                sets[i as usize][c / 64] |= 1u64 << (c % 64);
            }
        }
    };
    for c in 0..ntokens {
        let tok = &dict_bytes[dict_offsets[c] as usize..dict_offsets[c + 1] as usize];
        // s <= 0: pattern position 0 falls on token byte -s.
        for j in 0..tok.len() {
            if tok[j] == pattern[0] {
                mark(&mut sets, tok, c, -(j as isize));
            }
        }
        // s in 1..m: token byte 0 falls on pattern position s.
        for &s in &ppos[tok[0] as usize] {
            mark(&mut sets, tok, c, s as isize);
        }
    }
    sets
}

/// Where anchor-frequency estimates come from at compile time.
pub(crate) enum ScoreSource<'a> {
    /// Evenly strided sample of the column's code stream.
    SampledCodes(&'a [u16]),
    /// Stored per-token frequency summary; the code stream is never read.
    Stats(&'a crate::query::CodeStats),
    /// No frequency information at all: a dictionary length prior
    /// (`4^-len`, given `dict_offsets`) standing in for token frequency —
    /// shorter tokens are presumed exponentially more common.
    DictPrior(&'a [u32]),
}

/// Estimated fraction of stream codes hitting each anchor's candidate set.
/// `max_samples` bounds the sample size when `source` is a code stream.
fn anchor_hit_rates(sets: &[Vec<u64>], source: &ScoreSource<'_>, max_samples: usize) -> Vec<f64> {
    match source {
        ScoreSource::SampledCodes(codes) => {
            // Sampling cost is dominated by cache misses on the touched
            // stream locations, not the per-anchor bit tests, so samples are
            // taken as 32-code blocks (one cache line each) at evenly spaced
            // offsets. The sample budget scales down with anchor count and is
            // capped; anchor selection only separates rates that differ by
            // orders of magnitude, so a few thousand samples are ample (and a
            // misrank costs speed, never correctness).
            let m = sets.len().max(1);
            let target = (codes.len() / (16 * m)).clamp(256, max_samples);
            let mut hits = vec![0usize; sets.len()];
            let mut sampled = 0usize;
            let mut tally = |block: &[u16]| {
                for &c in block {
                    let (w, b) = (c as usize / 64, c as usize % 64);
                    for (set, hit) in sets.iter().zip(hits.iter_mut()) {
                        // A code out of range for the dictionary panics here
                        // via the slice index (codes are not pre-validated).
                        *hit += (set[w] >> b & 1) as usize;
                    }
                }
                sampled += block.len();
            };
            if codes.len() <= target {
                tally(codes);
            } else {
                const BLOCK: usize = 32;
                let nblocks = target / BLOCK; // >= 8 (target >= 256)
                let stride = codes.len() / nblocks; // >= BLOCK as len > target
                for b in 0..nblocks {
                    tally(&codes[b * stride..b * stride + BLOCK]);
                }
            }
            hits.iter()
                .map(|&h| h as f64 / sampled.max(1) as f64)
                .collect()
        }
        ScoreSource::Stats(stats) => {
            let total = stats.approx_total().max(1) as f64;
            sets.iter()
                .map(|set| {
                    let hit: u64 = (0..stats.num_tokens())
                        .filter(|&c| set[c / 64] >> (c % 64) & 1 == 1)
                        .map(|c| stats.approx_count(c))
                        .sum();
                    hit as f64 / total
                })
                .collect()
        }
        ScoreSource::DictPrior(dict_offsets) => {
            let ntokens = dict_offsets.len().saturating_sub(1);
            let weight = |c: usize| 0.25f64.powi((dict_offsets[c + 1] - dict_offsets[c]) as i32);
            let total: f64 = (0..ntokens).map(weight).sum::<f64>().max(f64::MIN_POSITIVE);
            sets.iter()
                .map(|set| {
                    (0..ntokens)
                        .filter(|&c| set[c / 64] >> (c % 64) & 1 == 1)
                        .map(weight)
                        .sum::<f64>()
                        / total
                })
                .collect()
        }
    }
}

/// Maximal runs of consecutive set bits, as inclusive code intervals.
/// Word-skipping (candidate sets are sparse), so the common cost is
/// `ntokens / 64` word tests plus one popcount-loop step per candidate.
fn bitmap_to_intervals(set: &[u64], ntokens: usize) -> Intervals {
    let mut out = Intervals::new();
    let mut run: Option<(u16, u16)> = None; // (start, last seen)
    for (w, &word) in set.iter().enumerate() {
        if word == 0 {
            continue;
        }
        let mut x = word;
        while x != 0 {
            let c = w * 64 + x.trailing_zeros() as usize;
            x &= x - 1;
            if c >= ntokens {
                break;
            }
            run = match run {
                Some((start, last)) if c as u16 == last + 1 => Some((start, c as u16)),
                Some((start, last)) => {
                    out.push((start, last));
                    Some((c as u16, c as u16))
                }
                None => Some((c as u16, c as u16)),
            };
        }
    }
    if let Some((start, last)) = run {
        out.push((start, last));
    }
    out
}

/// Pick the rarest anchor by `rates` and package its set as a [`Filter`].
/// Returns `None` when no anchor is selective enough to pay for the extra
/// pass (the caller then runs the DFA unfiltered).
fn select_filter(sets: &[Vec<u64>], rates: &[f64], ntokens: usize) -> Option<Filter> {
    // Argmin by rate, tie-broken toward the smaller candidate set.
    let (best, &expected_hit_rate) = rates.iter().enumerate().min_by(|&(i, ra), &(j, rb)| {
        let pop = |k: usize| sets[k].iter().map(|w| w.count_ones()).sum::<u32>();
        ra.total_cmp(rb).then_with(|| pop(i).cmp(&pop(j)))
    })?;

    // A prefilter that fires on most codes only adds a pass; let the DFA
    // run unassisted. (A row has many codes, so even a small per-code rate
    // means many rows pass; the verify step keeps the result exact either
    // way.)
    if expected_hit_rate > 0.25 {
        return None;
    }

    let set = &sets[best];
    let intervals = bitmap_to_intervals(set, ntokens);
    let kind = if intervals.len() <= MAX_SCAN_INTERVALS {
        Kind::Intervals(intervals)
    } else {
        let mut bm = Box::new([0u64; 1024]);
        bm[..set.len()].copy_from_slice(set);
        Kind::Bitmap(bm)
    };
    Some(Filter {
        kind,
        expected_hit_rate,
    })
}

impl Prefilter {
    /// Build a prefilter for `pattern` with its anchor chosen now, by the
    /// frequency estimates in `source`. `None` when no anchor is selective
    /// enough.
    pub(crate) fn build(
        pattern: &[u8],
        dict_bytes: &[u8],
        dict_offsets: &[u32],
        source: &ScoreSource<'_>,
    ) -> Option<Self> {
        let ntokens = dict_offsets.len().saturating_sub(1);
        if pattern.is_empty() || ntokens == 0 {
            return None;
        }
        let sets = anchor_candidates(pattern, dict_bytes, dict_offsets);
        let rates = anchor_hit_rates(&sets, source, COMPILE_SAMPLES);
        select_filter(&sets, &rates, ntokens).map(Self::Fixed)
    }

    /// Build a prefilter from the dictionary alone, deferring anchor choice
    /// to scan time (each scan samples the codes it is given).
    pub(crate) fn build_deferred(
        pattern: &[u8],
        dict_bytes: &[u8],
        dict_offsets: &[u32],
    ) -> Option<Self> {
        let ntokens = dict_offsets.len().saturating_sub(1);
        if pattern.is_empty() || ntokens == 0 {
            return None;
        }
        let sets = anchor_candidates(pattern, dict_bytes, dict_offsets);
        Some(Self::Deferred { sets, ntokens })
    }

    /// The filter to scan `codes` with: the compile-time choice, or — when
    /// deferred — the rarest anchor on a strided sample of `codes` (costing
    /// `pattern_len` bit-tests per sampled code, negligible next to the
    /// scan). `None` means no anchor is selective enough for this stream and
    /// the caller should run the DFA over every row.
    pub(crate) fn resolve(&self, codes: &[u16]) -> Option<std::borrow::Cow<'_, Filter>> {
        match self {
            Self::Fixed(f) => Some(std::borrow::Cow::Borrowed(f)),
            Self::Deferred { sets, ntokens } => {
                let rates = anchor_hit_rates(sets, &ScoreSource::SampledCodes(codes), SCAN_SAMPLES);
                select_filter(sets, &rates, *ntokens).map(std::borrow::Cow::Owned)
            }
        }
    }

    /// Compile-time diagnostics: `(strategy, expected per-code hit rate)`.
    /// `None` for a deferred prefilter (nothing is known until a scan).
    pub(crate) fn info(&self) -> Option<(&'static str, f64)> {
        match self {
            Self::Fixed(f) => Some((f.strategy(), f.expected_hit_rate)),
            Self::Deferred { .. } => None,
        }
    }
}

impl Filter {
    /// Package an arbitrary candidate bitmap over `ntokens` codes as a
    /// scannable filter, choosing the interval or bitmap representation by
    /// the same rule as anchor selection. The hit-rate estimate is the set's
    /// density (diagnostic only; nothing at runtime reads it).
    pub(crate) fn from_bitmap(set: &[u64], ntokens: usize) -> Self {
        let intervals = bitmap_to_intervals(set, ntokens);
        let kind = if intervals.len() <= MAX_SCAN_INTERVALS {
            Kind::Intervals(intervals)
        } else {
            let mut bm = Box::new([0u64; 1024]);
            bm[..set.len()].copy_from_slice(set);
            Kind::Bitmap(bm)
        };
        let pop: u32 = set.iter().map(|w| w.count_ones()).sum();
        Self {
            kind,
            expected_hit_rate: f64::from(pop) / ntokens.max(1) as f64,
        }
    }

    /// Human-readable scan strategy, for diagnostics.
    pub(crate) fn strategy(&self) -> &'static str {
        match self.kind {
            Kind::Intervals(_) => "simd-intervals",
            Kind::Bitmap(_) => "bitmap",
        }
    }

    /// Set bit `i` of `mask` for every candidate code `codes[i]`. `mask` must
    /// hold at least `codes.len().div_ceil(64)` words; words are overwritten.
    pub(crate) fn candidate_mask(&self, codes: &[u16], mask: &mut [u64]) {
        match &self.kind {
            Kind::Intervals(iv) => scan_intervals(codes, iv, mask),
            Kind::Bitmap(bm) => scan_bitmap(codes, bm, mask),
        }
    }
}

/// Any bit set in `mask[a..b)` (bit indices)? Used to test "row has a
/// candidate code" against the mask produced by `candidate_mask`.
#[inline]
pub(crate) fn any_bit_in_range(mask: &[u64], a: usize, b: usize) -> bool {
    if a >= b {
        return false;
    }
    let (wa, wb) = (a / 64, (b - 1) / 64);
    let lo = !0u64 << (a % 64);
    let hi = !0u64 >> (63 - (b - 1) % 64);
    if wa == wb {
        return mask[wa] & lo & hi != 0;
    }
    if mask[wa] & lo != 0 || mask[wb] & hi != 0 {
        return true;
    }
    mask[wa + 1..wb].iter().any(|&w| w != 0)
}

/// Interval-membership scan, dispatching to the widest SIMD available
/// (AVX-512BW, then AVX2, then scalar). The `ONPAIR_SIMD` env var
/// (`avx512` / `avx2` / `scalar`, read once) caps the dispatch — a
/// diagnostic escape hatch for benchmarking kernels against each other, not
/// a stable interface.
fn scan_intervals(codes: &[u16], intervals: &Intervals, mask: &mut [u64]) {
    #[cfg(target_arch = "x86_64")]
    {
        static OVERRIDE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        let force = OVERRIDE
            .get_or_init(|| std::env::var("ONPAIR_SIMD").ok())
            .as_deref();
        let allow = |name: &str| force.is_none_or(|f| f == name);
        if allow("avx512") && is_x86_feature_detected!("avx512bw") {
            // SAFETY: AVX-512BW presence checked at runtime.
            unsafe { avx512::scan(codes, intervals, mask) };
            return;
        }
        if allow("avx2") && is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 presence checked at runtime.
            unsafe { avx2::scan(codes, intervals, mask) };
            return;
        }
    }
    scan_intervals_scalar(codes, intervals, mask);
}

/// Bitmap-membership scan: one bit test per code against the 8 KiB set,
/// dispatching to an AVX-512 `VPGATHERDD` kernel when available (the same
/// `ONPAIR_SIMD` cap as `scan_intervals` applies).
fn scan_bitmap(codes: &[u16], bm: &[u64; 1024], mask: &mut [u64]) {
    #[cfg(target_arch = "x86_64")]
    {
        static OVERRIDE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        let force = OVERRIDE
            .get_or_init(|| std::env::var("ONPAIR_SIMD").ok())
            .as_deref();
        if force.is_none_or(|f| f == "avx512") && is_x86_feature_detected!("avx512f") {
            // SAFETY: AVX-512F presence checked at runtime.
            unsafe { avx512::scan_bitmap(codes, bm, mask) };
            return;
        }
    }
    scan_bitmap_scalar(codes, bm, mask);
}

fn scan_bitmap_scalar(codes: &[u16], bm: &[u64; 1024], mask: &mut [u64]) {
    for (w, chunk) in codes.chunks(64).enumerate() {
        let mut word = 0u64;
        for (b, &c) in chunk.iter().enumerate() {
            let hit = bm[c as usize / 64] >> (c as usize % 64) & 1;
            word |= hit << b;
        }
        mask[w] = word;
    }
}

fn scan_intervals_scalar(codes: &[u16], intervals: &[(u16, u16)], mask: &mut [u64]) {
    for (w, chunk) in codes.chunks(64).enumerate() {
        let mut word = 0u64;
        for (b, &c) in chunk.iter().enumerate() {
            let hit = intervals.iter().any(|&(lo, hi)| lo <= c && c <= hi);
            word |= (hit as u64) << b;
        }
        mask[w] = word;
    }
}

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use std::arch::x86_64::*;

    /// One AVX-512 iteration handles 64 codes — exactly one output mask
    /// word: two 32-lane `u16` vectors, one unsigned range test per interval
    /// (`vpcmpuw` writes straight to a `k` mask register), OR-accumulated and
    /// stored. No pack/permute/movemask tail like AVX2 needs.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub(super) unsafe fn scan(codes: &[u16], intervals: &[(u16, u16)], mask: &mut [u64]) {
        let bounds: Vec<(__m512i, __m512i)> = intervals
            .iter()
            .map(|&(lo, hi)| (_mm512_set1_epi16(lo as i16), _mm512_set1_epi16(hi as i16)))
            .collect();

        let mut word = 0usize;
        let mut chunks = codes.chunks_exact(64);
        for chunk in &mut chunks {
            // SAFETY: chunk holds 64 u16s = two 512-bit unaligned loads.
            let v0 = unsafe { _mm512_loadu_si512(chunk.as_ptr().cast()) };
            let v1 = unsafe { _mm512_loadu_si512(chunk.as_ptr().add(32).cast()) };
            let mut k0: __mmask32 = 0;
            let mut k1: __mmask32 = 0;
            for &(lo, hi) in &bounds {
                k0 |= _mm512_cmpge_epu16_mask(v0, lo) & _mm512_cmple_epu16_mask(v0, hi);
                k1 |= _mm512_cmpge_epu16_mask(v1, lo) & _mm512_cmple_epu16_mask(v1, hi);
            }
            mask[word] = (k0 as u64) | ((k1 as u64) << 32);
            word += 1;
        }

        // Scalar tail (< 64 codes).
        let rem = chunks.remainder();
        if !rem.is_empty() {
            super::scan_intervals_scalar(rem, intervals, &mut mask[word..]);
        }
    }

    /// Bitmap membership via `VPGATHERDD`: per 16 codes, gather the 16
    /// containing `u32` words of the bitmap (index `code >> 5`), variable-
    /// shift each right by `code & 31`, and test bit 0 straight into a
    /// `k`-mask. Four iterations fill one output word. This is the only
    /// vector route into an *arbitrary* candidate set — the interval kernel
    /// needs the set to cluster.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub(super) unsafe fn scan_bitmap(codes: &[u16], bm: &[u64; 1024], mask: &mut [u64]) {
        let base = bm.as_ptr().cast::<i32>();
        let thirty_one = _mm512_set1_epi32(31);
        let one = _mm512_set1_epi32(1);
        let mut word = 0usize;
        let mut chunks = codes.chunks_exact(64);
        for chunk in &mut chunks {
            let mut acc = 0u64;
            for (q, c16) in chunk.chunks_exact(16).enumerate() {
                // SAFETY: c16 holds 16 u16s = one 256-bit unaligned load.
                let c = unsafe { _mm256_loadu_si256(c16.as_ptr().cast()) };
                let c = _mm512_cvtepu16_epi32(c);
                let widx = _mm512_srli_epi32::<5>(c);
                // SAFETY: widx lanes are < 2048, the u32 length of `bm`.
                let words = unsafe { _mm512_i32gather_epi32::<4>(widx, base.cast()) };
                let sh = _mm512_and_si512(c, thirty_one);
                let bits = _mm512_srlv_epi32(words, sh);
                let hits = _mm512_test_epi32_mask(bits, one);
                acc |= (hits as u64) << (16 * q);
            }
            mask[word] = acc;
            word += 1;
        }
        let rem = chunks.remainder();
        if !rem.is_empty() {
            super::scan_bitmap_scalar(rem, bm, &mut mask[word..]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    /// One AVX2 iteration handles 32 codes: two 16-lane `u16` vectors, an
    /// unsigned range test per interval (`max(x, lo) == x && min(x, hi) == x`),
    /// then pack + movemask to a 32-bit hit mask.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn scan(codes: &[u16], intervals: &[(u16, u16)], mask: &mut [u64]) {
        let bounds: Vec<(__m256i, __m256i)> = intervals
            .iter()
            .map(|&(lo, hi)| (_mm256_set1_epi16(lo as i16), _mm256_set1_epi16(hi as i16)))
            .collect();

        let mut word = 0usize;
        let mut chunks = codes.chunks_exact(64);
        for chunk in &mut chunks {
            let mut acc: u64 = 0;
            for (half, chunk32) in chunk.chunks_exact(32).enumerate() {
                // SAFETY: chunk32 holds 32 u16s = two 256-bit unaligned loads.
                let v0 = unsafe { _mm256_loadu_si256(chunk32.as_ptr().cast()) };
                let v1 = unsafe { _mm256_loadu_si256(chunk32.as_ptr().add(16).cast()) };
                let mut hit0 = _mm256_setzero_si256();
                let mut hit1 = _mm256_setzero_si256();
                for &(lo, hi) in &bounds {
                    hit0 = _mm256_or_si256(hit0, in_range(v0, lo, hi));
                    hit1 = _mm256_or_si256(hit1, in_range(v1, lo, hi));
                }
                // Lanes are 0x0000/0xFFFF; pack to bytes then movemask. packs
                // interleaves 128-bit halves, so permute back into code order.
                let packed = _mm256_packs_epi16(hit0, hit1);
                let ordered = _mm256_permute4x64_epi64(packed, 0b11011000);
                let bits = _mm256_movemask_epi8(ordered) as u32;
                acc |= (bits as u64) << (32 * half);
            }
            mask[word] = acc;
            word += 1;
        }

        // Scalar tail (< 64 codes).
        let rem = chunks.remainder();
        if !rem.is_empty() {
            super::scan_intervals_scalar(rem, intervals, &mut mask[word..]);
        }
    }

    /// Per-lane unsigned `lo <= x && x <= hi`, as 0xFFFF/0x0000 lanes.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn in_range(x: __m256i, lo: __m256i, hi: __m256i) -> __m256i {
        let ge_lo = _mm256_cmpeq_epi16(_mm256_max_epu16(x, lo), x);
        let le_hi = _mm256_cmpeq_epi16(_mm256_min_epu16(x, hi), x);
        _mm256_and_si256(ge_lo, le_hi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_scan_matches_scalar() {
        let intervals: Intervals = vec![(3, 7), (100, 100), (60000, 65535)];
        let codes: Vec<u16> = (0..1000u32)
            .map(|i| (i.wrapping_mul(2654435761) >> 16) as u16)
            .collect();
        let words = codes.len().div_ceil(64);
        let mut simd = vec![0u64; words];
        let mut scalar = vec![0u64; words];
        scan_intervals(&codes, &intervals, &mut simd);
        scan_intervals_scalar(&codes, &intervals, &mut scalar);
        assert_eq!(simd, scalar);
        // Spot-check against the definition.
        for (i, &c) in codes.iter().enumerate() {
            let expect = intervals.iter().any(|&(lo, hi)| lo <= c && c <= hi);
            assert_eq!(scalar[i / 64] >> (i % 64) & 1 == 1, expect, "code {c}");
        }
        // Every available vector kernel must agree with scalar bit-for-bit,
        // including boundary codes (0, u16::MAX) and odd tails.
        #[cfg(target_arch = "x86_64")]
        {
            let mut edge: Vec<u16> = codes.clone();
            edge.extend([0, 1, 2, 3, 7, 8, 99, 100, 101, 59999, 60000, 65534, 65535]);
            let words = edge.len().div_ceil(64);
            let mut expect = vec![0u64; words];
            scan_intervals_scalar(&edge, &intervals, &mut expect);
            if is_x86_feature_detected!("avx2") {
                let mut got = vec![0u64; words];
                // SAFETY: AVX2 presence checked above.
                unsafe { avx2::scan(&edge, &intervals, &mut got) };
                assert_eq!(got, expect, "avx2 kernel diverges from scalar");
            }
            if is_x86_feature_detected!("avx512bw") {
                let mut got = vec![0u64; words];
                // SAFETY: AVX-512BW presence checked above.
                unsafe { avx512::scan(&edge, &intervals, &mut got) };
                assert_eq!(got, expect, "avx512 kernel diverges from scalar");
            }
        }
    }

    #[test]
    fn bitmap_scan_matches_scalar() {
        let mut bm = Box::new([0u64; 1024]);
        for c in [0u16, 1, 31, 32, 63, 64, 100, 999, 32768, 65534, 65535] {
            bm[c as usize / 64] |= 1 << (c % 64);
        }
        let mut codes: Vec<u16> = (0..1500u32)
            .map(|i| (i.wrapping_mul(2654435761) >> 13) as u16)
            .collect();
        codes.extend([0, 1, 31, 32, 63, 64, 100, 999, 32768, 65534, 65535, 2]);
        let words = codes.len().div_ceil(64);
        let mut got = vec![0u64; words];
        let mut expect = vec![0u64; words];
        scan_bitmap(&codes, &bm, &mut got);
        scan_bitmap_scalar(&codes, &bm, &mut expect);
        assert_eq!(got, expect);
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            let mut simd = vec![0u64; words];
            // SAFETY: AVX-512F presence checked above.
            unsafe { avx512::scan_bitmap(&codes, &bm, &mut simd) };
            assert_eq!(simd, expect, "gather kernel diverges from scalar");
        }
    }

    #[test]
    fn any_bit_in_range_word_boundaries() {
        let mut mask = vec![0u64; 3];
        mask[1] = 1 << 5; // bit 69
        assert!(any_bit_in_range(&mask, 0, 192));
        assert!(any_bit_in_range(&mask, 69, 70));
        assert!(any_bit_in_range(&mask, 64, 128));
        assert!(!any_bit_in_range(&mask, 0, 69));
        assert!(!any_bit_in_range(&mask, 70, 192));
        assert!(!any_bit_in_range(&mask, 0, 0));
        assert!(any_bit_in_range(&mask, 63, 70));
    }

    #[test]
    fn anchor_candidates_are_necessary() {
        // Dictionary of all bytes plus a few multi-byte tokens.
        let mut bytes: Vec<u8> = (0u16..256).map(|b| b as u8).collect();
        let mut offsets: Vec<u32> = (0u32..=256).collect();
        for t in [&b"goog"[..], b"le.com", b"oogle", b"xyz"] {
            bytes.extend_from_slice(t);
            offsets.push(bytes.len() as u32);
        }
        let pattern = b"google";
        let sets = anchor_candidates(pattern, &bytes, &offsets);
        assert_eq!(sets.len(), pattern.len());

        let has = |set: &[u64], c: usize| set[c / 64] >> (c % 64) & 1 == 1;
        // Token "goog" (code 256) starts the pattern: covers anchors 0..4.
        for i in 0..4 {
            assert!(has(&sets[i], 256), "anchor {i} must allow \"goog\"");
        }
        // "le.com" (257) aligns at s=4 ("le" agrees, ".com" past the end).
        assert!(has(&sets[4], 257));
        assert!(has(&sets[5], 257));
        // "oogle" (258) covers anchors 1..6 at s=1.
        for i in 1..6 {
            assert!(has(&sets[i], 258), "anchor {i} must allow \"oogle\"");
        }
        // "xyz" (259) overlaps nowhere.
        for set in &sets {
            assert!(!has(set, 259));
        }
        // Single byte 'g' covers anchors 0 and 3; not 1.
        assert!(has(&sets[0], b'g' as usize));
        assert!(has(&sets[3], b'g' as usize));
        assert!(!has(&sets[1], b'g' as usize));
    }
}
