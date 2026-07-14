// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Approximate SIMD prefilter for substring search.
//!
//! Answering `LIKE '%pattern%'` exactly means checking every row — e.g. stepping
//! the token-level KMP automaton of [`contains`](super::contains()) over its
//! codes. This module trims that per-row work down to the rows that *can* match.
//! It compiles a **sound probe cover** ([`ContainsPrefilter`]) from the pattern —
//! dictionary token ids and id ranges chosen so that *any* row containing the
//! pattern holds at least one probe token — then scans the flat code stream
//! (vectorized where available) to collect a **superset** of the matching rows
//! ([`prefilter_candidates`]).
//!
//! The prefilter stops there: it hands back the candidate rows and leaves the
//! exact check to the caller. Verifying only those survivors — with `contains`,
//! a decode-and-`memmem`, or any exact substring test — recovers the precise
//! answer, since a sound cover drops no true match.

use super::prefix_range;
use crate::core::dictionary::DictionaryView;
use crate::core::offset::Offset;
use crate::core::types::{MAX_TOKEN_SIZE, Token, TokenRange};

/// Largest first-token set (tokens ending with `needle[..k]`) still cheap enough
/// to enumerate and use as a probe. Above this the alignment falls back to its
/// interior/final probe.
const SET_CAP: usize = 16;

/// Use the vectorized scan only while the per-vector compare budget
/// (`points + 2·ranges`) stays at or below this; a wider cover is cheaper scanned
/// scalar through the membership table.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const SIMD_CAP: usize = 32;

/// A needle prepared for substring prefiltering: a sound probe cover over the
/// dictionary. Built once against a dictionary with [`ContainsPrefilter::new`],
/// then shared by `&` across the scan; holds no scan state.
#[derive(Debug, Clone)]
pub struct ContainsPrefilter {
    /// Point probes: individual dictionary token ids (sorted, deduped).
    points: Vec<Token>,
    /// Range probes: inclusive `[lo, hi]` token-id ranges (sorted, deduped).
    ranges: Vec<(Token, Token)>,
    /// Membership table, `len == num_tokens`: `table[id]` iff `id` is any probe.
    /// Backs the scalar scan (one lookup per code).
    table: Vec<bool>,
    /// Empty pattern (`LIKE '%%'`): every row is a candidate. Set instead of
    /// compiling a cover, so [`prefilter_candidates`] stays correct standalone.
    match_all: bool,
}

impl ContainsPrefilter {
    /// Compile the sound probe cover for `pattern` against the sorted `dict`,
    /// selecting per terminal the option that flags the fewest expected rows.
    ///
    /// `cum_freq` is the **prefix sum** of per-token term frequency
    /// (`cum_freq.len() == dict.num_tokens() + 1`), computed once per column at
    /// compression time and stored on it — the `cum_token_freq` field of
    /// [`ColumnView`](crate::ColumnView) — so `ContainsPrefilter` holds no
    /// frequency state itself. Storing the prefix sums (not the raw counts) keeps
    /// both point- and range-selectivity O(1) with nothing proportional to
    /// `num_tokens` rebuilt per call.
    ///
    /// `O(pattern · num_tokens)`: it sweeps the dictionary a handful of times
    /// (first-token feasibility + set frequency, set expansion, and the
    /// contained-set search) — the same order as building a
    /// [`ContainsTable`](super::ContainsTable). The payoff is replacing the
    /// *per-row* KMP with a vectorized stream scan.
    pub fn new<V: DictionaryView>(pattern: &[u8], dict: V, cum_freq: &[u64]) -> Self {
        if pattern.is_empty() {
            return Self {
                points: Vec::new(),
                ranges: Vec::new(),
                table: Vec::new(),
                match_all: true,
            };
        }
        compile_probes(dict, pattern, cum_freq)
    }

    /// Per-vector SIMD compare budget: one compare per point, two per range.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[inline]
    fn cmp_cost(&self) -> usize {
        self.points.len() + 2 * self.ranges.len()
    }
}

/// One accepting parse of the needle for a fixed first-token alignment `k`: the
/// first token covers `needle[..k]` as its tail (`k == 0` ⇒ it starts at the
/// needle), then greedy longest-match tokens run to the needle end.
struct Parse {
    /// Bytes of the needle absorbed into the first token's tail.
    k: usize,
    /// Forced interior tokens (exact greedy matches strictly inside the needle).
    interior: Vec<Token>,
    /// Needle offset where each interior token starts (parallel to `interior`).
    interior_pos: Vec<usize>,
    /// Dictionary range for the final (needle-ending) token.
    final_range: TokenRange,
    /// Interior offsets where a token could instead extend *past* the needle end,
    /// giving an alternate terminal (a shorter, possibly more selective, probe).
    early_cuts: Vec<usize>,
}

/// A terminal to cover: `ilen` interior tokens are available as point probes, the
/// range `rng` as a range probe (each scored by its summed term frequency).
struct Terminal {
    /// Number of leading interior tokens usable as point probes for this terminal.
    ilen: usize,
    /// Final/extending token range for this terminal.
    rng: TokenRange,
}

/// Which probe a terminal contributes (the cheapest, i.e. longest-token, option).
enum Choice {
    /// No probe (unreachable: every terminal has a non-empty final/extend range).
    None,
    /// A single interior token id.
    Point(Token),
    /// The terminal's token-id range.
    Range,
    /// The first-token set for alignment `k` (expanded to point probes later).
    Set,
}

/// Greedy longest in-needle token at `s` (= `needle[p..]`), capped at `s.len()`
/// and [`MAX_TOKEN_SIZE`]. Replicates the encoder's longest-prefix match
/// restricted to the needle; returns `(token id, byte length)`.
fn greedy_in_needle<V: DictionaryView>(dict: V, s: &[u8]) -> (Token, usize) {
    debug_assert!(!s.is_empty(), "greedy_in_needle needs a non-empty suffix");
    let r = s.len().min(MAX_TOKEN_SIZE);
    for len in (1..=r).rev() {
        let rng = prefix_range(dict, &s[..len]);
        // The shortest token in a prefix range sorts first, so length exactly
        // `len` at `rng.begin` means it is the exact token `s[..len]`.
        if !rng.is_empty() && dict.token_len(rng.begin) == len {
            return (rng.begin, len);
        }
    }
    // Unreachable for a complete dictionary: the `len == 1` step always matches
    // the single-byte token `[s[0]]`, which sorts first in its prefix range.
    // Resolved by lookup, *not* `s[0] as Token`: in a bytewise-sorted dictionary a
    // token id is a sorted position, not a byte value.
    let rng = prefix_range(dict, &s[..1]);
    (rng.begin, 1)
}

/// Build the accepting [`Parse`] for first-token alignment `k` (`k < needle.len()`).
fn build_parse<V: DictionaryView>(dict: V, needle: &[u8], k: usize) -> Parse {
    let n = needle.len();
    let mut pz = Parse {
        k,
        interior: Vec::new(),
        interior_pos: Vec::new(),
        final_range: TokenRange::EMPTY,
        early_cuts: Vec::new(),
    };
    let mut p = k;
    loop {
        let (tid, len) = greedy_in_needle(dict, &needle[p..]);
        if p + len == n {
            // Greedy reached the needle end: this token is the final one.
            pz.final_range = prefix_range(dict, &needle[p..]);
            break;
        }
        pz.interior.push(tid);
        pz.interior_pos.push(p);
        // Early cut: a token beginning with the whole remaining suffix would run
        // past the needle end here, an alternate (shorter-suffix) terminal.
        if !prefix_range(dict, &needle[p..]).is_empty() {
            pz.early_cuts.push(p);
        }
        p += len;
    }
    pz
}

/// Compile the sound probe cover for a non-empty `np` (see [`ContainsPrefilter::new`]).
///
/// `cum_freq` is the **prefix sum** of per-token term frequency
/// (`cum_freq.len() == num_tokens + 1`, `cum_freq[i] = Σ freq[0..i]`), built once
/// per column and reused across patterns (see [`ContainsPrefilter::new`]). Storing
/// the prefix sums — not
/// the raw counts — means both a point's row-mass (`cum_freq[t+1] − cum_freq[t]`)
/// and a range's (`cum_freq[hi+1] − cum_freq[lo]`) are O(1) with **no per-query
/// precompute**: nothing proportional to `num_tokens` is rebuilt in `new`.
fn compile_probes<V: DictionaryView>(dict: V, np: &[u8], cum_freq: &[u64]) -> ContainsPrefilter {
    let n = np.len();
    let ntok = dict.num_tokens();
    let kmax = n.min(MAX_TOKEN_SIZE);

    // Row-mass estimates, both O(1) from the stored prefix sums.
    let freq = |t: usize| -> f64 { (cum_freq[t + 1] - cum_freq[t]) as f64 };
    let range_rows = |r: TokenRange| -> f64 {
        if r.is_empty() {
            f64::INFINITY // never present in practice; keep it strictly worst
        } else {
            (cum_freq[r.last as usize + 1] - cum_freq[r.begin as usize]) as f64
        }
    };

    // Parse every first-token alignment (needle-only, cheap).
    let mut parses: Vec<Parse> = Vec::with_capacity(kmax);
    for k in 0..kmax {
        parses.push(build_parse(dict, np, k));
    }

    // Size the first-token sets in ONE pass over the dictionary. For each k ≥ 1 we
    // want the tokens ending with needle[..k]: their count (`fcount`), total term
    // frequency (`fsum`), and — while the set is still small enough to be usable —
    // its members (`first_set`, so set-expansion below needs no second scan).
    // `feasible[0]` is always true (the first token can start exactly at the needle).
    // Only `fsum`/`first_set` up to `SET_CAP` matter — a bigger set is never chosen —
    // so we stop growing them past the cap.
    let mut feasible = vec![false; kmax];
    let mut fcount = vec![0usize; kmax];
    let mut fsum = vec![0f64; kmax];
    let mut first_set: Vec<Vec<Token>> = vec![Vec::new(); kmax];
    feasible[0] = true;
    for t in 0..ntok {
        let tok = dict.token(t as Token);
        let tl = tok.len();
        // A token of length tl can end with needle[..k] only for 1 ≤ k ≤ min(tl, kmax-1).
        let khi = kmax.min(tl + 1);
        for k in 1..khi {
            if tok[tl - k..] == np[..k] {
                fcount[k] += 1;
                if fcount[k] <= SET_CAP {
                    fsum[k] += freq(t);
                    first_set[k].push(t as Token);
                }
            }
        }
    }
    for k in 1..kmax {
        feasible[k] = fcount[k] > 0;
    }

    // Per terminal, keep the option flagging the fewest expected rows (summed
    // term frequency); options are tried interior → range → set with a strict
    // improvement to switch, so comparisons act only as a tiebreak.
    let mut points: Vec<Token> = Vec::new();
    let mut ranges: Vec<(Token, Token)> = Vec::new();
    let mut set_mask: u32 = 0;
    for pz in &parses {
        if !feasible[pz.k] {
            continue;
        }
        let mut terminals: Vec<Terminal> = vec![Terminal {
            ilen: pz.interior.len(),
            rng: pz.final_range,
        }];
        for &cut in &pz.early_cuts {
            let mut ilen = 0usize;
            while ilen < pz.interior_pos.len() && pz.interior_pos[ilen] < cut {
                ilen += 1;
            }
            terminals.push(Terminal {
                ilen,
                rng: prefix_range(dict, &np[cut..]),
            });
        }
        for tm in &terminals {
            let mut best = f64::INFINITY;
            let mut choice = Choice::None;
            for i in 0..tm.ilen {
                let t = pz.interior[i];
                let c = freq(t as usize);
                if c < best {
                    best = c;
                    choice = Choice::Point(t);
                }
            }
            let cf = range_rows(tm.rng);
            if cf < best {
                best = cf;
                choice = Choice::Range;
            }
            if pz.k >= 1 && fcount[pz.k] <= SET_CAP {
                let sc = fsum[pz.k];
                if sc < best {
                    choice = Choice::Set;
                }
            }
            match choice {
                Choice::Point(t) => points.push(t),
                Choice::Range => {
                    if !tm.rng.is_empty() {
                        ranges.push((tm.rng.begin, tm.rng.last));
                    }
                }
                Choice::Set => set_mask |= 1u32 << pz.k,
                Choice::None => {}
            }
        }
    }

    // Expand each chosen first-token set into its point probes. The members were
    // already collected in the single pass above (a set is only ever chosen when
    // its fcount ≤ SET_CAP, so `first_set[k]` holds all of them) — no second scan.
    for k in 1..kmax {
        if (set_mask >> k) & 1 == 1 {
            points.extend_from_slice(&first_set[k]);
        }
    }

    // Contained set: every token whose bytes hold the whole needle (the pattern
    // living inside one token). A token is ≤ MAX_TOKEN_SIZE, so this is provably
    // empty for a longer needle — skip the whole dictionary scan in that case.
    if n <= MAX_TOKEN_SIZE {
        for t in 0..ntok {
            let tok = dict.token(t as Token);
            if tok.len() >= n && tok.windows(n).any(|w| w == np) {
                points.push(t as Token);
            }
        }
    }

    points.sort_unstable();
    points.dedup();
    ranges.sort_unstable();
    ranges.dedup();

    let mut table = vec![false; ntok];
    for &p in &points {
        table[p as usize] = true;
    }
    for &(lo, hi) in &ranges {
        for id in lo..=hi {
            table[id as usize] = true;
        }
    }

    ContainsPrefilter {
        points,
        ranges,
        table,
        match_all: false,
    }
}

/// Mark each row holding ≥ 1 probe token in `cand` (one flag per row), dispatching
/// to the widest vectorized kernel this target and CPU offer, else the scalar
/// reference. The cover's per-vector compare budget (`cmp_cost`) gates the SIMD
/// paths: past [`SIMD_CAP`] a wide cover is cheaper scanned scalar through the
/// membership table than as that many vector compares per block.
fn scan<O: Offset>(codes: &[Token], row_offsets: &[O], pf: &ContainsPrefilter, cand: &mut [bool]) {
    // AArch64: NEON is part of the Armv8-A baseline, so no runtime probe is needed.
    #[cfg(target_arch = "aarch64")]
    {
        if pf.cmp_cost() <= SIMD_CAP {
            scan_neon(codes, row_offsets, pf, cand);
            return;
        }
    }
    // x86-64: pick the widest kernel the running CPU advertises — AVX-512BW, then
    // AVX2 — falling back to the SSE2 baseline (guaranteed by the x86-64 target).
    #[cfg(target_arch = "x86_64")]
    {
        if pf.cmp_cost() <= SIMD_CAP {
            if std::is_x86_feature_detected!("avx512bw") {
                // SAFETY: `avx512bw` was just detected (and implies `avx512f` on
                // any real CPU), so `scan_avx512`'s intrinsics are legal here.
                unsafe { scan_avx512(codes, row_offsets, pf, cand) };
                return;
            }
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: `avx2` was just detected at runtime, so the AVX2
                // intrinsics inside `scan_avx2` are legal on this CPU.
                unsafe { scan_avx2(codes, row_offsets, pf, cand) };
                return;
            }
            scan_sse2(codes, row_offsets, pf, cand);
            return;
        }
    }
    scan_scalar(codes, row_offsets, pf, cand);
}

/// Portable scan: walk each row's codes, flag it on the first probe hit. The
/// reference kernel — always compiled, and the path on any non-accelerated target.
fn scan_scalar<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ContainsPrefilter,
    cand: &mut [bool],
) {
    for r in 0..cand.len() {
        let a = row_offsets[r].to_usize();
        let b = row_offsets[r + 1].to_usize();
        for &c in &codes[a..b] {
            if pf.table[c as usize] {
                cand[r] = true;
                break;
            }
        }
    }
}

// The SIMD kernels below share three pieces of scalar bookkeeping — the fiddly,
// easy-to-break part — so each kernel is just its own compare core. All are gated
// to the accelerated targets; a target with no SIMD kernel never references them.

/// Advance the monotone row cursor to the row owning code index `idx`, then flag
/// it. A scan visits code indices in non-decreasing order (low lane → high lane,
/// blocks ascending), so `cur_row` only ever moves forward across a whole scan.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline]
fn flag_row<O: Offset>(
    row_offsets: &[O],
    n: usize,
    cur_row: &mut usize,
    idx: usize,
    cand: &mut [bool],
) {
    while *cur_row + 1 <= n && row_offsets[*cur_row + 1].to_usize() <= idx {
        *cur_row += 1;
    }
    cand[*cur_row] = true;
}

/// Fold one SIMD block's per-lane hit mask into per-row candidates. `hits[j]` is
/// non-zero iff the code at `base + j` is a probe token; `hits.len()` is the
/// kernel's lane count. Lanes are visited low → high, keeping `base + j` monotone.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline]
fn mark_block<O: Offset>(
    row_offsets: &[O],
    base: usize,
    hits: &[u16],
    cur_row: &mut usize,
    cand: &mut [bool],
) {
    let n = cand.len();
    for (j, &h) in hits.iter().enumerate() {
        if h != 0 {
            flag_row(row_offsets, n, cur_row, base + j, cand);
        }
    }
}

/// Scalar membership over the trailing `codes[from..]` — the fewer-than-one-block
/// remainder a SIMD kernel leaves — continuing the same monotone row cursor.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline]
fn scan_tail<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ContainsPrefilter,
    from: usize,
    cur_row: &mut usize,
    cand: &mut [bool],
) {
    let n = cand.len();
    for (off, &c) in codes[from..].iter().enumerate() {
        if pf.table[c as usize] {
            flag_row(row_offsets, n, cur_row, from + off, cand);
        }
    }
}

/// NEON scan: test 8 codes per 128-bit vector against every point (equality) and
/// range (native unsigned `≥ lo ∧ ≤ hi`), then hand the per-lane hit mask to
/// [`mark_block`]. Correct for any cover; the [`scan`] dispatcher only prefers it
/// under [`SIMD_CAP`].
#[cfg(target_arch = "aarch64")]
fn scan_neon<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ContainsPrefilter,
    cand: &mut [bool],
) {
    use core::arch::aarch64::{
        vandq_u16, vceqq_u16, vcgeq_u16, vcleq_u16, vdupq_n_u16, vld1q_u16, vmaxvq_u16, vorrq_u16,
        vst1q_u16,
    };

    let total = codes.len();
    let base = codes.as_ptr();
    let mut cur_row = 0usize;
    let mut i = 0usize;
    while i + 8 <= total {
        // SAFETY: `i + 8 <= total`, so the 8-lane (16-byte) load stays within
        // `codes`. The remaining intrinsics are register-only compares/reduces and
        // a store into the 8-lane stack array `m`.
        let hits = unsafe {
            let v = vld1q_u16(base.add(i));
            let mut acc = vdupq_n_u16(0);
            for &p in &pf.points {
                acc = vorrq_u16(acc, vceqq_u16(v, vdupq_n_u16(p)));
            }
            for &(lo, hi) in &pf.ranges {
                let ge = vcgeq_u16(v, vdupq_n_u16(lo));
                let le = vcleq_u16(v, vdupq_n_u16(hi));
                acc = vorrq_u16(acc, vandq_u16(ge, le));
            }
            if vmaxvq_u16(acc) == 0 {
                None
            } else {
                let mut m = [0u16; 8];
                vst1q_u16(m.as_mut_ptr(), acc);
                Some(m)
            }
        };
        if let Some(m) = hits {
            mark_block(row_offsets, i, &m, &mut cur_row, cand);
        }
        i += 8;
    }
    scan_tail(codes, row_offsets, pf, i, &mut cur_row, cand);
}

/// SSE2 scan: the x86-64 baseline kernel, 8 codes per 128-bit vector. SSE2 has
/// only *signed* 16-bit compares, so ranges are tested in sign-biased space (XOR
/// `0x8000` maps unsigned order onto signed) — [`Token`] ids exceed `0x7FFF` once
/// a dictionary is large, so the bias is load-bearing, not cosmetic. Point probes
/// use exact equality, which is sign-agnostic. Per-lane hits go to [`mark_block`];
/// correct for any cover, gated by [`scan`] under [`SIMD_CAP`].
#[cfg(target_arch = "x86_64")]
fn scan_sse2<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ContainsPrefilter,
    cand: &mut [bool],
) {
    use core::arch::x86_64::{
        __m128i, _mm_andnot_si128, _mm_cmpeq_epi16, _mm_cmpgt_epi16, _mm_loadu_si128,
        _mm_movemask_epi8, _mm_or_si128, _mm_set1_epi16, _mm_setzero_si128, _mm_storeu_si128,
        _mm_xor_si128,
    };

    let total = codes.len();
    let base = codes.as_ptr();
    let mut cur_row = 0usize;
    let mut i = 0usize;
    while i + 8 <= total {
        // SAFETY: `i + 8 <= total`, so the 128-bit load reads 8 in-bounds codes.
        // SSE2 is guaranteed on x86-64; the rest are register-only ops plus one
        // store into the 8-lane stack array `m`.
        let hits = unsafe {
            let v = _mm_loadu_si128(base.add(i).cast::<__m128i>());
            let bias = _mm_set1_epi16(i16::MIN); // 0x8000: unsigned → signed order
            let cb = _mm_xor_si128(v, bias); // codes in sign-biased space
            let ones = _mm_set1_epi16(-1);
            let mut acc = _mm_setzero_si128();
            for &p in &pf.points {
                acc = _mm_or_si128(acc, _mm_cmpeq_epi16(v, _mm_set1_epi16(p as i16)));
            }
            for &(lo, hi) in &pf.ranges {
                let lob = _mm_xor_si128(_mm_set1_epi16(lo as i16), bias);
                let hib = _mm_xor_si128(_mm_set1_epi16(hi as i16), bias);
                // Out of range = below lo OR above hi; in-range is its complement.
                let below = _mm_cmpgt_epi16(lob, cb);
                let above = _mm_cmpgt_epi16(cb, hib);
                let out = _mm_or_si128(below, above);
                acc = _mm_or_si128(acc, _mm_andnot_si128(out, ones));
            }
            if _mm_movemask_epi8(acc) == 0 {
                None
            } else {
                let mut m = [0u16; 8];
                _mm_storeu_si128(m.as_mut_ptr().cast::<__m128i>(), acc);
                Some(m)
            }
        };
        if let Some(m) = hits {
            mark_block(row_offsets, i, &m, &mut cur_row, cand);
        }
        i += 8;
    }
    scan_tail(codes, row_offsets, pf, i, &mut cur_row, cand);
}

/// AVX2 scan: 16 codes per 256-bit vector — the widest kernel here, chosen when
/// the CPU advertises `avx2`. AVX2 likewise lacks unsigned 16-bit compares, so it
/// uses the same sign-biased range test as [`scan_sse2`]; per-lane hits go to
/// [`mark_block`]. Correct for any cover, gated by [`scan`] under [`SIMD_CAP`].
///
/// `#[target_feature(enable = "avx2")]` makes this unsafe to call without the
/// feature; [`scan`] gates the call behind an `is_x86_feature_detected!` probe.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn scan_avx2<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ContainsPrefilter,
    cand: &mut [bool],
) {
    use core::arch::x86_64::{
        __m256i, _mm256_andnot_si256, _mm256_cmpeq_epi16, _mm256_cmpgt_epi16, _mm256_loadu_si256,
        _mm256_movemask_epi8, _mm256_or_si256, _mm256_set1_epi16, _mm256_setzero_si256,
        _mm256_storeu_si256, _mm256_xor_si256,
    };

    let total = codes.len();
    let base = codes.as_ptr();
    let mut cur_row = 0usize;
    let mut i = 0usize;
    while i + 16 <= total {
        // SAFETY: `i + 16 <= total`, so the 256-bit load reads 16 in-bounds codes.
        // The caller established `avx2`; the rest are register-only ops plus one
        // store into the 16-lane stack array `m`.
        let hits = unsafe {
            let v = _mm256_loadu_si256(base.add(i).cast::<__m256i>());
            let bias = _mm256_set1_epi16(i16::MIN);
            let cb = _mm256_xor_si256(v, bias);
            let ones = _mm256_set1_epi16(-1);
            let mut acc = _mm256_setzero_si256();
            for &p in &pf.points {
                acc = _mm256_or_si256(acc, _mm256_cmpeq_epi16(v, _mm256_set1_epi16(p as i16)));
            }
            for &(lo, hi) in &pf.ranges {
                let lob = _mm256_xor_si256(_mm256_set1_epi16(lo as i16), bias);
                let hib = _mm256_xor_si256(_mm256_set1_epi16(hi as i16), bias);
                let below = _mm256_cmpgt_epi16(lob, cb);
                let above = _mm256_cmpgt_epi16(cb, hib);
                let out = _mm256_or_si256(below, above);
                acc = _mm256_or_si256(acc, _mm256_andnot_si256(out, ones));
            }
            if _mm256_movemask_epi8(acc) == 0 {
                None
            } else {
                let mut m = [0u16; 16];
                _mm256_storeu_si256(m.as_mut_ptr().cast::<__m256i>(), acc);
                Some(m)
            }
        };
        if let Some(m) = hits {
            mark_block(row_offsets, i, &m, &mut cur_row, cand);
        }
        i += 16;
    }
    scan_tail(codes, row_offsets, pf, i, &mut cur_row, cand);
}

/// AVX-512BW scan: 32 codes per 512-bit vector — the widest kernel, chosen when
/// the CPU advertises `avx512bw`. AVX-512 has *native* unsigned 16-bit compares,
/// so it drops the sign-bias dance [`scan_sse2`]/[`scan_avx2`] need; and its
/// compares yield a mask register (`__mmask32`, a `u32`), so hits are bit-iterated
/// straight into [`flag_row`] — no lane vector, no store-to-array, no
/// [`mark_block`]. Correct for any cover, gated by [`scan`] under [`SIMD_CAP`].
///
/// `#[target_feature]` makes this unsafe to call without the features; [`scan`]
/// gates the call behind an `is_x86_feature_detected!` probe. `avx512f` is enabled
/// alongside `avx512bw` for the 512-bit load (a base-`f` intrinsic).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
fn scan_avx512<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ContainsPrefilter,
    cand: &mut [bool],
) {
    use core::arch::x86_64::{
        _mm512_cmpeq_epu16_mask, _mm512_cmpge_epu16_mask, _mm512_cmple_epu16_mask,
        _mm512_loadu_si512, _mm512_set1_epi16,
    };

    let n = cand.len();
    let total = codes.len();
    let base = codes.as_ptr();
    let mut cur_row = 0usize;
    let mut i = 0usize;
    while i + 32 <= total {
        // SAFETY: `i + 32 <= total`, so the 512-bit load reads 32 in-bounds codes.
        // The caller established `avx512{f,bw}`; the compares are register-only and
        // return mask registers, so no store or spill occurs here.
        let mut m = unsafe {
            let v = _mm512_loadu_si512(base.add(i).cast());
            let mut acc: u32 = 0; // __mmask32: bit j set ⇒ lane j is a probe hit
            for &p in &pf.points {
                acc |= _mm512_cmpeq_epu16_mask(v, _mm512_set1_epi16(p as i16));
            }
            for &(lo, hi) in &pf.ranges {
                // Native unsigned compares — no bias needed, unlike SSE2/AVX2.
                let ge = _mm512_cmpge_epu16_mask(v, _mm512_set1_epi16(lo as i16));
                let le = _mm512_cmple_epu16_mask(v, _mm512_set1_epi16(hi as i16));
                acc |= ge & le;
            }
            acc
        };
        // Fold the hit mask into rows, low lane → high lane so `i + j` stays
        // monotone for the cursor `flag_row` advances.
        while m != 0 {
            let j = m.trailing_zeros() as usize;
            flag_row(row_offsets, n, &mut cur_row, i + j, cand);
            m &= m - 1;
        }
        i += 32;
    }
    scan_tail(codes, row_offsets, pf, i, &mut cur_row, cand);
}

/// Append (ascending) the rows that hold ≥ 1 probe token of `pf` — a **sound
/// superset** of the rows containing the prepared pattern. The caller verifies
/// the survivors with any exact substring check (the token-KMP
/// [`contains`](super::contains()), a decode-and-`memmem`, …) to recover the
/// precise answer; a sound cover drops no true match.
///
/// `codes` is the row-concatenated code stream and `row_offsets` its `R + 1` row
/// delimiters (see [`ColumnView`](crate::ColumnView)). An empty prepared pattern
/// yields every row.
pub fn prefilter_candidates<O: Offset>(
    codes: &[Token],
    row_offsets: &[O],
    pf: &ContainsPrefilter,
    out: &mut Vec<usize>,
) {
    let n = row_offsets.len().saturating_sub(1);
    if pf.match_all {
        out.extend(0..n);
        return;
    }
    let mut cand = vec![false; n];
    scan(codes, row_offsets, pf, &mut cand);
    for r in 0..n {
        if cand[r] {
            out.push(r);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dictionary::Dictionary;
    use crate::search::{ContainsTable, contains};
    use crate::{Column, DEFAULT_CONFIG, compress};

    /// The "prefilter, then verify" recipe wired to the token-KMP
    /// [`contains`] verifier, over raw buffers so the test can drive either
    /// dictionary representation. This is the exact answer the library expects a
    /// caller to assemble from the prefilter primitives; the survivors could
    /// equally be checked by decode-and-`memmem`.
    fn prefilter_then_contains<V: DictionaryView, O: Offset>(
        codes: &[Token],
        row_offsets: &[O],
        dict: V,
        cum_freq: &[u64],
        pattern: &[u8],
    ) -> Vec<usize> {
        if pattern.is_empty() {
            return (0..row_offsets.len().saturating_sub(1)).collect();
        }
        let table = ContainsTable::new(pattern, dict);
        let pf = ContainsPrefilter::new(pattern, dict, cum_freq);
        let mut cand = Vec::new();
        prefilter_candidates(codes, row_offsets, &pf, &mut cand);
        cand.into_iter()
            .filter(|&r| {
                let a = row_offsets[r].to_usize();
                let b = row_offsets[r + 1].to_usize();
                contains(&codes[a..b], &table)
            })
            .collect()
    }

    fn compress_rows(rows: &[&[u8]]) -> Column<u32> {
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for r in rows {
            bytes.extend_from_slice(r);
            offsets.push(bytes.len() as u32);
        }
        compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap()
    }

    fn byte_contains(hay: &[u8], needle: &[u8]) -> bool {
        needle.is_empty() || hay.windows(needle.len()).any(|w| w == needle)
    }

    /// Decode row `k` to bytes via the into-buffer API, for the oracle.
    fn decode_row(view: crate::ColumnView<'_, u32>, k: usize) -> Vec<u8> {
        let mut buf =
            vec![std::mem::MaybeUninit::uninit(); view.row_decoded_len(k) + crate::DECODE_PADDING];
        // SAFETY: buffer sized for row `k`; view from a trusted column.
        let w = unsafe { view.decompress_row_into(k, &mut buf) };
        unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), w) }.to_vec()
    }

    /// Prefilter-then-verify must equal a brute-force decode-and-substring oracle
    /// and the existing exact `rows_containing`, over both dictionary
    /// representations; and the raw candidate set must be a sound *superset* of the
    /// true matches, ascending and unique.
    fn check(rows: &[&[u8]], patterns: &[&[u8]]) {
        let col = compress_rows(rows);
        let view = col.view();
        let wide = view.wide_dict();
        for &pat in patterns {
            let want: Vec<usize> = (0..view.num_rows())
                .filter(|&k| byte_contains(&decode_row(view, k), pat))
                .collect();

            // Prefilter + token-KMP verify equals the oracle over both dictionary
            // representations, using the column's stored frequencies (identical
            // for either dictionary — same token count)...
            assert_eq!(
                prefilter_then_contains(
                    view.codes,
                    view.row_offsets,
                    view.dict,
                    view.cum_token_freq,
                    pat
                ),
                want,
                "compact {pat:?}"
            );
            assert_eq!(
                prefilter_then_contains(
                    view.codes,
                    view.row_offsets,
                    wide.as_view(),
                    view.cum_token_freq,
                    pat
                ),
                want,
                "wide {pat:?}"
            );
            // ...and so do the column conveniences that assemble the same recipe,
            // with either the token-KMP verify or the decode-and-`memmem` verify.
            assert_eq!(
                view.rows_containing_prefiltered(pat),
                want,
                "column {pat:?}"
            );
            assert_eq!(
                view.rows_containing_prefiltered(pat),
                view.rows_containing(pat)
            );
            assert_eq!(
                view.rows_containing_prefiltered_memmem(pat),
                want,
                "memmem {pat:?}"
            );

            // Soundness: candidates ⊇ true matches, ascending and unique.
            let pf = ContainsPrefilter::new(pat, view.dict, view.cum_token_freq);
            let mut cand = Vec::new();
            prefilter_candidates(view.codes, view.row_offsets, &pf, &mut cand);
            for w in &want {
                assert!(
                    cand.contains(w),
                    "candidate set dropped row {w} for {pat:?}"
                );
            }
            assert!(
                cand.windows(2).all(|w| w[0] < w[1]),
                "candidates must be ascending and unique for {pat:?}"
            );
        }
    }

    #[test]
    fn empty_pattern_matches_all_rows() {
        let rows: &[&[u8]] = &[b"a", b"", b"abc"];
        check(rows, &[b""]);
    }

    #[test]
    fn single_and_multi_token_substrings() {
        let rows: &[&[u8]] = &[b"hello world", b"world peace", b"helloworld", b"hell"];
        check(
            rows,
            &[
                b"hello",
                b"world",
                b"o w",
                b"llowo",
                b"xyz",
                b"hello world",
                b"hello world!",
            ],
        );
    }

    #[test]
    fn substrings_spanning_token_boundaries() {
        let rows: &[&[u8]] = &[b"abcabcabc", b"xabcabcy", b"ababab", b"cab"];
        check(
            rows,
            &[b"abc", b"bca", b"cab", b"bcabca", b"abcabcabc", b"ba"],
        );
    }

    #[test]
    fn repeating_pattern_exercises_early_cuts() {
        let rows: &[&[u8]] = &[b"aaaaab", b"aabaab", b"ababab", b"aaa"];
        check(rows, &[b"aa", b"aaa", b"aab", b"abab", b"aaaa", b"aabaa"]);
    }

    #[test]
    fn matches_brute_force_on_repetitive_corpus() {
        use crate::test_corpus::user_strings;
        let corpus: Vec<Vec<u8>> = user_strings(50)
            .into_iter()
            .map(String::into_bytes)
            .collect();
        let rows: Vec<&[u8]> = corpus.iter().map(Vec::as_slice).collect();
        check(
            &rows,
            &[
                b"example",
                b"https",
                b"://",
                b".com",
                b"/page",
                b"ftp",
                b"zzz",
                b"w",
                b"https://www.example.com/",
            ],
        );
    }

    #[test]
    fn matches_brute_force_on_binary_corpus() {
        use crate::test_corpus::binary_strings;
        let corpus = binary_strings(40, 24, 11);
        let rows: Vec<&[u8]> = corpus.iter().map(Vec::as_slice).collect();
        let patterns: &[&[u8]] = &[b"", b"\x00", b"\xff", b"\x00\x01", &[7u8], &[200u8, 201]];
        check(&rows, patterns);
    }

    #[test]
    #[should_panic(expected = "contains pattern exceeds 255 bytes")]
    fn pattern_over_255_panics_like_contains() {
        let col = compress_rows(&[b"abc"]);
        let big = vec![b'a'; 256];
        let _ = col.view().rows_containing_prefiltered(&big);
    }

    /// The decode-and-`memmem` variant has no 255-byte cap (no [`ContainsTable`]),
    /// so a pattern that would panic the token-KMP path resolves correctly here.
    #[test]
    fn memmem_handles_pattern_over_255_bytes() {
        let long = vec![b'a'; 300];
        let short = vec![b'a'; 10];
        let rows: &[&[u8]] = &[&long, b"abc", &short];
        let col = compress_rows(rows);
        let pat = vec![b'a'; 256];
        // Only the 300-byte row contains 256 consecutive 'a's.
        assert_eq!(col.view().rows_containing_prefiltered_memmem(&pat), vec![0]);
    }

    /// A SIMD `kernel` must flag exactly the rows [`scan_scalar`] does, over a
    /// realistic multi-byte-token corpus and a spread of patterns (single-token,
    /// boundary-spanning, absent). This is the equivalence contract every kernel
    /// shares; the per-ISA tests below just supply their kernel.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    fn assert_kernel_matches_scalar(
        kernel: impl Fn(&[Token], &[u32], &ContainsPrefilter, &mut [bool]),
    ) {
        use crate::test_corpus::user_strings;
        let corpus: Vec<Vec<u8>> = user_strings(60)
            .into_iter()
            .map(String::into_bytes)
            .collect();
        let rows: Vec<&[u8]> = corpus.iter().map(Vec::as_slice).collect();
        let col = compress_rows(&rows);
        let view = col.view();
        let n = view.num_rows();
        let patterns: &[&[u8]] = &[
            b"e",
            b"://",
            b"example",
            b".com/page",
            b"https://www.example.com",
            b"zzz",
        ];
        for &pat in patterns {
            let pf = ContainsPrefilter::new(pat, view.dict, view.cum_token_freq);
            let mut scalar = vec![false; n];
            let mut simd = vec![false; n];
            scan_scalar(view.codes, view.row_offsets, &pf, &mut scalar);
            kernel(view.codes, view.row_offsets, &pf, &mut simd);
            assert_eq!(scalar, simd, "kernel disagrees with scalar for {pat:?}");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar() {
        assert_kernel_matches_scalar(scan_neon::<u32>);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn sse2_matches_scalar() {
        assert_kernel_matches_scalar(scan_sse2::<u32>);
    }

    /// AVX2 must agree with scalar — skipped on CPUs without it (where the
    /// dispatcher would never pick it anyway).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar() {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: the closure runs the AVX2 kernel only under this probe.
            assert_kernel_matches_scalar(|codes, ro, pf, cand| unsafe {
                scan_avx2(codes, ro, pf, cand)
            });
        }
    }

    /// AVX-512 must agree with scalar — skipped on CPUs without AVX-512BW (where
    /// the dispatcher would never pick it anyway).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_matches_scalar() {
        if std::is_x86_feature_detected!("avx512bw") {
            // SAFETY: the closure runs the AVX-512 kernel only under this probe.
            assert_kernel_matches_scalar(|codes, ro, pf, cand| unsafe {
                scan_avx512(codes, ro, pf, cand)
            });
        }
    }

    /// The SSE2/AVX2 range test runs in sign-biased space because those ISAs have
    /// only *signed* 16-bit compares, while [`Token`] ids span the full unsigned
    /// `u16` range. This checks that emulation — one lane, in scalar `i16`,
    /// mirroring the kernels' `xor 0x8000` + signed `cmpgt` + complement — equals
    /// a true unsigned `lo <= c <= hi` for *every* code and a spread of bounds,
    /// including ranges straddling `0x8000` where a naive signed compare breaks.
    /// Architecture-independent: it validates the arithmetic the vector kernels
    /// rely on, so it stands in for their runtime check where they can't execute.
    #[test]
    fn signed_bias_range_matches_unsigned() {
        // One lane of the kernels' range core, in scalar signed space.
        fn in_range_biased(c: u16, lo: u16, hi: u16) -> bool {
            const BIAS: u16 = 0x8000;
            let cb = (c ^ BIAS) as i16;
            let lob = (lo ^ BIAS) as i16;
            let hib = (hi ^ BIAS) as i16;
            let below = lob > cb; // c < lo (unsigned)
            let above = cb > hib; // c > hi (unsigned)
            !(below || above) // in-range = complement of out-of-range
        }
        let bounds: &[(u16, u16)] = &[
            (0, 0),
            (0, u16::MAX),
            (0x7FFF, 0x8000), // straddles the signed sign boundary
            (0x8000, 0xFFFF), // wholly in the high half (negative as i16)
            (0x00FF, 0xFF00),
            (1234, 1234),
            (40000, 50000),
        ];
        for &(lo, hi) in bounds {
            for c in 0..=u16::MAX {
                assert_eq!(
                    in_range_biased(c, lo, hi),
                    lo <= c && c <= hi,
                    "c={c} lo={lo} hi={hi}"
                );
            }
        }
    }
}
