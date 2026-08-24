// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Class-compressed token DFA: sparse-alphabet reduction, multi-token
//! (pair) lookup, and a Hyperflex-style in-register walk on AVX-512 VBMI.
//!
//! The dense token DFA is `(m + 1) × ntokens` but has very few *distinct
//! columns*: most tokens map every state to the same destination (their
//! `δ*(0, ·)` walk), and only tokens whose first byte extends a live pattern
//! prefix differ per state. Grouping tokens by their full transition column
//! gives a **class alphabet** of C classes (measured median ~9, max ~27, out
//! of 65,536 tokens):
//!
//!   * `class_map: token -> class` — a 64 KiB `u8` map, the only dense array;
//!   * `trans: C × S` — the transition table over classes and (compacted,
//!     reachable) states, tens of bytes;
//!   * `trans2: C² × S` — the table over class *pairs*, feasible only because
//!     the alphabet collapsed. One state-dependent load advances two tokens.
//!
//! On top of the same reduction, a Hyperflex-style walk (arXiv 2512.07123)
//! keeps the state in a SIMD lane: each class's transition column is a
//! 64-byte row and a transition is `VPERMB(state, row)` — the row load
//! depends only on the input code, so it pipelines, and the serial chain is
//! one shuffle per step (or per pair, with `trans2` rows).

use std::collections::HashMap;

use super::dfa::MAX_PATTERN_LEN;
use super::dfa::byte_automaton;
use super::prefilter::Filter;
use super::prefilter::any_bit_in_range;
use crate::Offset;

/// Cap on the shifted-mask combination depth (as [`super::sparse`]).
const MAX_COMBO: usize = 3;

/// Cap on the pair-table size; `C² × S` beyond this stops paying for itself
/// (measured median is 288 bytes).
const MAX_TRANS2_BYTES: usize = 4 << 20;

/// How a compiled [`ClassSearcher`] walks a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Walk {
    /// One class-table lookup per code.
    Class,
    /// One pair-table lookup per two codes (requires `trans2`).
    Pair,
    /// Hyperflex: state in a SIMD lane, one `VPERMB` per code
    /// (requires AVX-512 VBMI and at most 64 states).
    Hyperflex,
    /// Hyperflex over class pairs: one `VPERMB` per two codes.
    HyperflexPair,
}

/// Compile-time shape of a [`ClassSearcher`], for diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClassInfo {
    /// Distinct transition columns (alphabet size after reduction).
    pub nclasses: usize,
    /// Reachable, compacted states (accept included).
    pub nstates: usize,
    /// Bytes of the pair table, 0 when not built.
    pub trans2_bytes: usize,
    /// Interesting codes (class != 0 population).
    pub interesting: usize,
    /// Combination depth of the shifted-mask prefilter.
    pub combo: usize,
}

/// A `contains` query over the class-compressed token DFA. Compiled from the
/// dictionary alone; reusable across code streams over the same dictionary.
pub struct ClassSearcher {
    /// `None` for the empty pattern, which matches every row.
    inner: Option<Inner>,
}

struct Inner {
    /// `token -> class`, padded to 64 Ki entries (+3 bytes so 4-byte gathers
    /// at any code stay in bounds). Class 0 is the "boring" column mapping
    /// every state to 0. Only valid when `nclasses <= 256`.
    class_map8: Vec<u8>,
    /// `token -> class` without the u8 cap; the construction map, and the
    /// scalar fallback when `nclasses > 256`.
    class_map16: Vec<u16>,
    nclasses: usize,
    /// Compacted reachable states; state 0 is compact id 0.
    nstates: usize,
    /// `trans[k * nstates + s]`: next compact state. The multiply is
    /// computed from the code alone, off the serial state chain.
    trans: Vec<u8>,
    /// `trans2[(k1 * nclasses + k2) * nstates + s]`, `None` when over cap
    /// or `nclasses > 256`. Compact strides: padding to powers of two was
    /// measured to cost more in cache footprint than the shifts saved.
    trans2: Option<Vec<u8>>,
    /// Hyperflex rows: per class, a 64-byte next-state column (`None` when
    /// `nstates > 64` or `nclasses > 256`).
    hf_rows: Option<Vec<u8>>,
    /// Hyperflex pair rows: per class pair, a 64-byte column.
    hf_rows2: Option<Vec<u8>>,
    /// Compact accept state id.
    accept: u8,
    /// SIMD-scannable membership for the interesting set (class != 0).
    filter: Filter,
    /// Consecutive interesting codes a match requires, `1..=MAX_COMBO`.
    combo: usize,
}

/// Bind `$step` to the row-walk closure for `$walk` and evaluate `$body`
/// once per arm, so row loops are monomorphized per walk instead of
/// dispatching per row.
macro_rules! dispatch_walk {
    ($inner:expr, $walk:expr, $step:ident => $body:expr) => {{
        let inner = $inner;
        match $walk {
            Walk::Class => {
                if inner.class_map8.is_empty() {
                    let $step = |row: &[u16]| inner.walk_row_class16(row);
                    $body
                } else {
                    let $step = |row: &[u16]| inner.walk_row_class(row);
                    $body
                }
            }
            Walk::Pair => {
                let $step = |row: &[u16]| inner.walk_row_pair(row);
                $body
            }
            #[cfg(target_arch = "x86_64")]
            Walk::Hyperflex => {
                // SAFETY: `supports` gates on runtime AVX-512 VBMI detection.
                let $step = |row: &[u16]| unsafe { inner.walk_row_hyperflex(row) };
                $body
            }
            #[cfg(target_arch = "x86_64")]
            Walk::HyperflexPair => {
                // SAFETY: as above.
                let $step = |row: &[u16]| unsafe { inner.walk_row_hyperflex_pair(row) };
                $body
            }
            #[cfg(not(target_arch = "x86_64"))]
            Walk::Hyperflex | Walk::HyperflexPair => {
                panic!("hyperflex walks require x86_64 AVX-512 VBMI")
            }
        }
    }};
}

impl ClassSearcher {
    /// Compile a searcher for `pattern` over the (validated) dictionary.
    ///
    /// ## Panics
    ///
    /// Panics if `pattern.len() > MAX_PATTERN_LEN`.
    pub fn compile_dict(dict_bytes: &[u8], dict_offsets: &[u32], pattern: &[u8]) -> Self {
        if pattern.is_empty() {
            return Self { inner: None };
        }
        let m = pattern.len();
        assert!(m <= MAX_PATTERN_LEN, "pattern length out of range");
        let ntokens = dict_offsets.len().saturating_sub(1);
        let delta = byte_automaton(pattern);

        let token = |c: usize| &dict_bytes[dict_offsets[c] as usize..dict_offsets[c + 1] as usize];
        let walk = |s: usize, tok: &[u8]| -> u8 {
            let mut st = s;
            for &b in tok {
                st = delta[st * 256 + b as usize] as usize;
                if st == m {
                    break; // absorbing
                }
            }
            st as u8
        };

        // δ*(0, c) per token, plus tokens bucketed by first byte (the only
        // tokens whose columns can deviate from base at some state).
        let mut base = vec![0u8; ntokens];
        let mut by_first: Vec<Vec<u16>> = vec![Vec::new(); 256];
        for c in 0..ntokens {
            let tok = token(c);
            base[c] = walk(0, tok);
            by_first[tok[0] as usize].push(c as u16);
        }

        // Per-code exception list (state, dest), states ascending: the
        // entries of the code's column that differ from base.
        let mut exc_by_code: HashMap<u16, Vec<(u8, u8)>> = HashMap::new();
        for s in 1..m {
            for b in 0..256 {
                if delta[s * 256 + b] == delta[b] {
                    continue;
                }
                for &c in &by_first[b] {
                    let dest = walk(s, token(c as usize));
                    if dest != base[c as usize] {
                        exc_by_code.entry(c).or_default().push((s as u8, dest));
                    }
                }
            }
        }

        // Group tokens by column: key = (base, exception list). Class 0 is
        // the boring column (base 0, no exceptions).
        let mut classes: HashMap<(u8, Vec<(u8, u8)>), u16> = HashMap::new();
        classes.insert((0, Vec::new()), 0);
        // Representative column per class, as (base, exceptions).
        let mut columns: Vec<(u8, Vec<(u8, u8)>)> = vec![(0, Vec::new())];
        let mut class_map16 = vec![0u16; ntokens];
        for c in 0..ntokens {
            let mut key = (base[c], exc_by_code.remove(&(c as u16)).unwrap_or_default());
            key.1.sort_unstable();
            let next = columns.len() as u16;
            let id = *classes.entry(key.clone()).or_insert_with(|| {
                columns.push(key);
                next
            });
            class_map16[c] = id;
        }
        let nclasses = columns.len();

        // Reachable-state compaction: token-level states are 0, the accept
        // state, and every base/exception destination.
        let mut seen = vec![false; m + 1];
        seen[0] = true;
        seen[m] = true;
        for &(b, ref exc) in &columns {
            seen[b as usize] = true;
            for &(_, d) in exc {
                seen[d as usize] = true;
            }
        }
        let orig: Vec<usize> = (0..=m).filter(|&s| seen[s]).collect();
        let mut compact = vec![0u8; m + 1];
        for (i, &s) in orig.iter().enumerate() {
            compact[s] = i as u8;
        }
        let nstates = orig.len();
        let accept = compact[m];

        // trans[k * nstates + s]: apply class k's column to orig state s.
        let mut trans = vec![0u8; nclasses * nstates];
        for (k, &(b, ref exc)) in columns.iter().enumerate() {
            for (cs, &s) in orig.iter().enumerate() {
                let dest = if s == m {
                    m as u8 // absorbing accept
                } else {
                    match exc.binary_search_by_key(&(s as u8), |&(st, _)| st) {
                        Ok(i) => exc[i].1,
                        Err(_) => b,
                    }
                };
                trans[k * nstates + cs] = compact[dest as usize];
            }
        }

        // Fixed-size u8 map (+ gather pad) when the alphabet fits.
        let class_map8 = if nclasses <= 256 {
            let mut map = vec![0u8; (1 << 16) + 4];
            for (c, &k) in class_map16.iter().enumerate() {
                map[c] = k as u8;
            }
            map
        } else {
            Vec::new()
        };

        // Pair table: trans2[k1, k2][s] = trans[k2][trans[k1][s]].
        let t2_len = nclasses * nclasses * nstates;
        let trans2 = (nclasses <= 256 && t2_len <= MAX_TRANS2_BYTES).then(|| {
            let mut t2 = vec![0u8; t2_len];
            for k1 in 0..nclasses {
                for k2 in 0..nclasses {
                    let row = &mut t2[(k1 * nclasses + k2) * nstates..][..nstates];
                    for s in 0..nstates {
                        let mid = trans[k1 * nstates + s] as usize;
                        row[s] = trans[k2 * nstates + mid];
                    }
                }
            }
            t2
        });

        // Hyperflex rows: 64-byte next-state columns (VPERMB indexes 64
        // bytes, so up to 64 states).
        let hf_ok = nstates <= 64 && nclasses <= 256;
        let hf_rows = hf_ok.then(|| {
            let mut rows = vec![0u8; nclasses * 64];
            for k in 0..nclasses {
                for s in 0..nstates {
                    rows[k * 64 + s] = trans[k * nstates + s];
                }
            }
            rows
        });
        let hf_rows2 = match (&trans2, hf_ok) {
            (Some(t2), true) => {
                let mut rows = vec![0u8; nclasses * nclasses * 64];
                for kk in 0..nclasses * nclasses {
                    for s in 0..nstates {
                        rows[kk * 64 + s] = t2[kk * nstates + s];
                    }
                }
                Some(rows)
            }
            _ => None,
        };

        // Interesting set (class != 0) for the combination prefilter.
        let words = ntokens.div_ceil(64).max(1);
        let mut set = vec![0u64; words];
        let mut lmax = 0usize;
        for (c, &k) in class_map16.iter().enumerate() {
            if k != 0 {
                set[c / 64] |= 1u64 << (c % 64);
                lmax = lmax.max(token(c).len());
            }
        }
        let combo = if lmax == 0 {
            1
        } else {
            m.div_ceil(lmax).clamp(1, MAX_COMBO)
        };

        Self {
            inner: Some(Inner {
                class_map8,
                class_map16,
                nclasses,
                nstates,
                trans,
                trans2,
                hf_rows,
                hf_rows2,
                accept,
                filter: Filter::from_bitmap(&set, ntokens),
                combo,
            }),
        }
    }

    /// Rows containing the pattern via the **fused state-0 skip**: one SIMD
    /// pass marks the interesting codes (class != 0), and the automaton
    /// walks only the maximal runs of set bits — a boring code maps every
    /// state to 0, so the walk can teleport between runs. The compressed
    /// analog of memmem's rare-byte skip, in-line in the scan instead of a
    /// separate row-granularity prefilter. Exact: matches cannot span rows
    /// (state resets at row boundaries) and cannot begin on a boring code.
    ///
    /// ## Panics
    ///
    /// As [`matching_rows_unfiltered`](Self::matching_rows_unfiltered).
    pub fn matching_rows_skip<O: Offset>(&self, codes: &[u16], code_offsets: &[O]) -> Vec<u64> {
        let Some(inner) = &self.inner else {
            return (0..code_offsets.len().saturating_sub(1) as u64).collect();
        };
        let nrows = code_offsets.len().saturating_sub(1);
        if nrows == 0 {
            return Vec::new();
        }
        let off = |r: usize| -> usize {
            let o = code_offsets[r].to_usize().expect("row offset overflows usize");
            assert!(o <= codes.len(), "malformed code_offsets");
            o
        };
        let ncodes = off(nrows);

        let mut mask = vec![0u64; codes.len().div_ceil(64)];
        inner.filter.candidate_mask(codes, &mut mask);

        // Next set bit at or after `from`, or ncodes when none.
        let next_interesting = |from: usize| -> usize {
            if from >= ncodes {
                return ncodes;
            }
            let mut w = from / 64;
            let mut word = mask[w] & (!0u64 << (from % 64));
            loop {
                if word != 0 {
                    return (w * 64 + word.trailing_zeros() as usize).min(ncodes);
                }
                w += 1;
                if w >= mask.len() {
                    return ncodes;
                }
                word = mask[w];
            }
        };
        let bit = |i: usize| mask[i / 64] >> (i % 64) & 1 == 1;

        let accept = inner.accept as usize;
        let nstates = inner.nstates;
        let mut out = Vec::new();
        let (mut row, mut row_end) = (0usize, off(1));
        let mut p = next_interesting(0);
        while p < ncodes {
            // Walk one run of interesting codes; a boring code (class 0)
            // maps every state to 0 without a table lookup.
            let mut s = 0usize;
            loop {
                if p >= ncodes {
                    return out;
                }
                if p >= row_end {
                    // Gallop the row cursor to the row containing p; matches
                    // cannot span rows.
                    let mut step = 1usize;
                    let mut lo = row + 1;
                    while lo + step <= nrows && off(lo + step) <= p {
                        lo += step;
                        step <<= 1;
                    }
                    let mut hi = (lo + step).min(nrows);
                    while lo + 1 < hi {
                        let mid = (lo + hi) / 2;
                        if off(mid) <= p {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                    row = lo;
                    row_end = off(lo + 1);
                    s = 0;
                }
                if !bit(p) {
                    if s == 0 {
                        break; // run over: skip to the next interesting code
                    }
                    s = 0; // boring code: state dies, no lookup needed
                    p += 1;
                    continue;
                }
                // SAFETY: index invariants as `walk_row_class`; p < ncodes.
                unsafe {
                    let k = *inner.class_map8.get_unchecked(codes[p] as usize) as usize;
                    s = *inner.trans.get_unchecked(k * nstates + s) as usize;
                }
                if s == accept {
                    out.push(row as u64);
                    p = row_end; // nothing more to learn in this row
                    s = 0;
                    continue;
                }
                p += 1;
            }
            p = next_interesting(p + 1);
        }
        out
    }

    /// Which of `spans` (as `[a, b)` code ranges) contain the pattern,
    /// walked **four rows at a time** so the four serial state chains hide
    /// each other's table-load latency. No early exit: accept is absorbing
    /// and exhausted lanes freeze, so walking every span to its end is
    /// exact. Supports [`Walk::Pair`] and [`Walk::HyperflexPair`]; other
    /// walks fall back to the sequential row walk.
    ///
    /// ## Panics
    ///
    /// Panics if a span is out of bounds or a code out of range.
    pub fn matching_spans_ilv4(&self, codes: &[u16], spans: &[(u32, u32)], walk: Walk) -> Vec<bool> {
        let Some(inner) = &self.inner else {
            return vec![true; spans.len()];
        };
        for &(a, b) in spans {
            assert!(a <= b && (b as usize) <= codes.len(), "span out of bounds");
        }
        let mut out = vec![false; spans.len()];
        match walk {
            // In span order: length-bucketing the batches was measured to
            // lose more to broken prefetch over the code stream than lane
            // convergence gained.
            Walk::Pair if inner.trans2.is_some() => {
                let mut i = 0;
                while i + 4 <= spans.len() {
                    let m4 = inner.walk4_pair(
                        codes,
                        [spans[i], spans[i + 1], spans[i + 2], spans[i + 3]],
                    );
                    out[i..i + 4].copy_from_slice(&m4);
                    i += 4;
                }
                for (o, &(a, b)) in out[i..].iter_mut().zip(&spans[i..]) {
                    *o = inner.walk_row_pair(&codes[a as usize..b as usize]);
                }
            }
            #[cfg(target_arch = "x86_64")]
            Walk::HyperflexPair if inner.hf_rows2.is_some() && avx512_vbmi() => {
                let mut i = 0;
                while i + 4 <= spans.len() {
                    // SAFETY: AVX-512 VBMI presence checked above.
                    let m4 = unsafe {
                        inner.walk4_hyperflex_pair(
                            codes,
                            [spans[i], spans[i + 1], spans[i + 2], spans[i + 3]],
                        )
                    };
                    out[i..i + 4].copy_from_slice(&m4);
                    i += 4;
                }
                for (o, &(a, b)) in out[i..].iter_mut().zip(&spans[i..]) {
                    // SAFETY: as above.
                    *o = unsafe { inner.walk_row_hyperflex_pair(&codes[a as usize..b as usize]) };
                }
            }
            _ => {
                for (o, &(a, b)) in out.iter_mut().zip(spans) {
                    *o = inner.walk_row(&codes[a as usize..b as usize], walk);
                }
            }
        }
        out
    }

    /// Does one row's code slice contain the pattern? The per-row verify
    /// entry point, exposed so the verify path can be measured in isolation.
    /// The empty pattern matches every row.
    ///
    /// ## Panics
    ///
    /// As [`matching_rows_unfiltered`](Self::matching_rows_unfiltered).
    #[inline]
    pub fn row_matches(&self, codes: &[u16], walk: Walk) -> bool {
        match &self.inner {
            None => true,
            Some(inner) => inner.walk_row(codes, walk),
        }
    }

    /// Compile-time shape, or `None` for the empty pattern.
    pub fn info(&self) -> Option<ClassInfo> {
        self.inner.as_ref().map(|i| ClassInfo {
            nclasses: i.nclasses,
            nstates: i.nstates,
            trans2_bytes: i.trans2.as_ref().map_or(0, Vec::len),
            interesting: i.class_map16.iter().filter(|&&k| k != 0).count(),
            combo: i.combo,
        })
    }

    /// Is `walk` available for this compiled searcher on this CPU?
    pub fn supports(&self, walk: Walk) -> bool {
        let Some(i) = &self.inner else {
            return true; // empty pattern: every walk trivially agrees
        };
        match walk {
            Walk::Class => true,
            Walk::Pair => i.trans2.is_some(),
            Walk::Hyperflex => i.hf_rows.is_some() && avx512_vbmi(),
            Walk::HyperflexPair => i.hf_rows2.is_some() && avx512_vbmi(),
        }
    }

    /// Rows containing the pattern, every row walked (no prefilter).
    ///
    /// ## Panics
    ///
    /// Panics if `walk` is unsupported (check [`supports`](Self::supports)),
    /// `code_offsets` is malformed, or a code is out of range.
    pub fn matching_rows_unfiltered<O: Offset>(
        &self,
        codes: &[u16],
        code_offsets: &[O],
        walk: Walk,
    ) -> Vec<u64> {
        let Some(inner) = &self.inner else {
            return (0..code_offsets.len().saturating_sub(1) as u64).collect();
        };
        dispatch_walk!(inner, walk, step => {
            let mut out = Vec::new();
            for (r, w) in code_offsets.windows(2).enumerate() {
                let (a, b) = span(w, codes.len());
                if step(&codes[a..b]) {
                    out.push(r as u64);
                }
            }
            out
        })
    }

    /// Rows containing the pattern: SIMD interesting-mask + shifted-mask
    /// combination prefilter, then `walk` over each surviving row.
    ///
    /// ## Panics
    ///
    /// As [`matching_rows_unfiltered`](Self::matching_rows_unfiltered).
    pub fn matching_rows<O: Offset>(
        &self,
        codes: &[u16],
        code_offsets: &[O],
        walk: Walk,
    ) -> Vec<u64> {
        let Some(inner) = &self.inner else {
            return (0..code_offsets.len().saturating_sub(1) as u64).collect();
        };
        let combined = inner.combined_mask(codes);
        let combo = inner.combo;
        // Sparse masks: gallop over set bits, skipping candidate-free rows
        // wholesale. Dense masks (> 1/8 of positions): the per-row range
        // test is cheaper than bit-chasing, so keep the row loop.
        let dense = combined.iter().map(|w| w.count_ones() as usize).sum::<usize>()
            > codes.len() / 8;
        dispatch_walk!(inner, walk, step => {
            let mut out = Vec::new();
            if dense {
                for (r, w) in code_offsets.windows(2).enumerate() {
                    let (a, b) = span(w, codes.len());
                    if b - a >= combo
                        && any_bit_in_range(&combined, a, b - combo + 1)
                        && step(&codes[a..b])
                    {
                        out.push(r as u64);
                    }
                }
            } else {
                // Re-check the row's window precisely: a set bit whose combo
                // window does not fit the row is not a candidate.
                super::for_each_candidate_row(codes, code_offsets, &combined, |r, row| {
                    let a = code_offsets[r as usize].to_usize().expect("offset");
                    if row.len() >= combo
                        && any_bit_in_range(&combined, a, a + row.len() - combo + 1)
                        && step(row)
                    {
                        out.push(r);
                    }
                });
            }
            out
        })
    }
}

impl Inner {
    /// Walk one row with the requested strategy.
    #[inline]
    fn walk_row(&self, codes: &[u16], walk: Walk) -> bool {
        match walk {
            Walk::Class => {
                if self.class_map8.is_empty() {
                    self.walk_row_class16(codes)
                } else {
                    self.walk_row_class(codes)
                }
            }
            Walk::Pair => self.walk_row_pair(codes),
            #[cfg(target_arch = "x86_64")]
            // SAFETY: `supports` gates on runtime AVX-512 VBMI detection.
            Walk::Hyperflex => unsafe { self.walk_row_hyperflex(codes) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: as above.
            Walk::HyperflexPair => unsafe { self.walk_row_hyperflex_pair(codes) },
            #[cfg(not(target_arch = "x86_64"))]
            Walk::Hyperflex | Walk::HyperflexPair => {
                panic!("hyperflex walks require x86_64 AVX-512 VBMI")
            }
        }
    }

    /// One class-table lookup per code.
    ///
    /// Index safety invariants, shared by every walk below: `class_map8` has
    /// 64 Ki+4 entries so any `u16` code is in bounds; its values are
    /// `< nclasses`; `trans` / `trans2` / `hf_rows*` are sized for the full
    /// padded `(class, state)` index space and every stored state is
    /// `< nstates`.
    #[inline]
    fn walk_row_class(&self, codes: &[u16]) -> bool {
        let accept = self.accept as usize;
        let nstates = self.nstates;
        let mut s = 0usize;
        for &c in codes {
            // SAFETY: see the invariants above.
            unsafe {
                let k = *self.class_map8.get_unchecked(c as usize) as usize;
                s = *self.trans.get_unchecked(k * nstates + s) as usize;
            }
            if s == accept {
                return true;
            }
        }
        false
    }

    /// Scalar fallback when the alphabet exceeds 256 classes.
    #[inline]
    fn walk_row_class16(&self, codes: &[u16]) -> bool {
        let accept = self.accept as usize;
        let nstates = self.nstates;
        let mut s = 0usize;
        for &c in codes {
            let k = self.class_map16[c as usize] as usize;
            s = self.trans[k * nstates + s] as usize;
            if s == accept {
                return true;
            }
        }
        false
    }

    /// One pair-table lookup per two codes; the tail code takes a single
    /// step. Exact under early exit because accept is absorbing. Both codes
    /// of a pair arrive in one `u32` load; the two class probes are
    /// state-independent and pipeline, so the serial chain is one table load
    /// per two codes.
    #[inline]
    fn walk_row_pair(&self, codes: &[u16]) -> bool {
        let t2 = self.trans2.as_deref().expect("pair walk requires trans2");
        let accept = self.accept as usize;
        let nstates = self.nstates;
        let nclasses = self.nclasses;
        let map = self.class_map8.as_ptr();
        let mut s = 0usize;
        let n2 = codes.len() & !1;
        let mut i = 0usize;
        while i < n2 {
            // SAFETY: i + 1 < codes.len(); index invariants as
            // `walk_row_class`.
            unsafe {
                let both = codes.as_ptr().add(i).cast::<u32>().read_unaligned();
                let k1 = *map.add((both & 0xFFFF) as usize) as usize;
                let k2 = *map.add((both >> 16) as usize) as usize;
                s = *t2.get_unchecked((k1 * nclasses + k2) * nstates + s) as usize;
            }
            if s == accept {
                return true;
            }
            i += 2;
        }
        if let Some(&c) = codes.get(n2) {
            // SAFETY: index invariants as `walk_row_class`.
            unsafe {
                let k = *map.add(c as usize) as usize;
                s = *self.trans.get_unchecked(k * nstates + s) as usize;
            }
        }
        s == accept
    }

    /// Hyperflex: state in a SIMD lane, one `VPERMB` per code. The row load
    /// depends only on the code (it pipelines); the serial chain is the
    /// shuffle. Accept is tested per step with a mask compare off the chain.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
    unsafe fn walk_row_hyperflex(&self, codes: &[u16]) -> bool {
        use std::arch::x86_64::*;
        let rows = self.hf_rows.as_deref().expect("hyperflex rows").as_ptr();
        let map = self.class_map8.as_ptr();
        let acceptv = _mm512_set1_epi8(self.accept as i8);
        let mut sv = _mm512_setzero_si512();
        for &c in codes {
            // SAFETY: class_map8 is 64 Ki entries; rows hold nclasses * 64
            // bytes and k < nclasses.
            let k = unsafe { *map.add(c as usize) } as usize;
            let row = unsafe { _mm512_loadu_si512(rows.add(k * 64).cast()) };
            sv = _mm512_permutexvar_epi8(sv, row);
            if _mm512_cmpeq_epi8_mask(sv, acceptv) != 0 {
                return true;
            }
        }
        false
    }

    /// Hyperflex over pairs: one `VPERMB` per two codes.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
    unsafe fn walk_row_hyperflex_pair(&self, codes: &[u16]) -> bool {
        use std::arch::x86_64::*;
        let rows2 = self.hf_rows2.as_deref().expect("hyperflex pair rows").as_ptr();
        let map = self.class_map8.as_ptr();
        let nclasses = self.nclasses;
        let acceptv = _mm512_set1_epi8(self.accept as i8);
        let mut sv = _mm512_setzero_si512();
        let n2 = codes.len() & !1;
        let mut i = 0usize;
        while i < n2 {
            // SAFETY: i + 1 < codes.len(); index invariants as
            // `walk_row_class`. Two independent u16 loads measure faster
            // than one fused u32 load here (both class probes issue at
            // once), unlike the scalar pair walk.
            unsafe {
                let c1 = *codes.get_unchecked(i) as usize;
                let c2 = *codes.get_unchecked(i + 1) as usize;
                let k1 = *map.add(c1) as usize;
                let k2 = *map.add(c2) as usize;
                let row = _mm512_loadu_si512(rows2.add((k1 * nclasses + k2) * 64).cast());
                sv = _mm512_permutexvar_epi8(sv, row);
            }
            if _mm512_cmpeq_epi8_mask(sv, acceptv) != 0 {
                return true;
            }
            i += 2;
        }
        let mut s = _mm512_cvtsi512_si32(sv) as u8 as usize;
        if let Some(&c) = codes.get(n2) {
            // SAFETY: single-step fallback through the scalar table.
            unsafe {
                let k = *map.add(c as usize) as usize;
                s = *self.trans.get_unchecked(k * self.nstates + s) as usize;
            }
        }
        s == self.accept as usize
    }

    /// Four interleaved pair walks: the four serial `trans2`-load chains
    /// hide each other's L1 latency across the three load ports. The main
    /// loop runs all four lanes unconditionally for `min(pairs)` rounds
    /// (rows in a chunk have similar lengths, so that is most of the work),
    /// breaking early only when every lane has accepted; each lane then
    /// finishes with the sequential walk. Exact because accept is
    /// absorbing.
    fn walk4_pair(&self, codes: &[u16], spans: [(u32, u32); 4]) -> [bool; 4] {
        let t2 = self.trans2.as_deref().expect("pair walk requires trans2");
        let accept = self.accept as usize;
        let nstates = self.nstates;
        let nclasses = self.nclasses;
        let map = self.class_map8.as_ptr();

        let (mut s0, mut s1, mut s2, mut s3) = (0usize, 0usize, 0usize, 0usize);
        let (a0, a1, a2, a3) = (
            spans[0].0 as usize,
            spans[1].0 as usize,
            spans[2].0 as usize,
            spans[3].0 as usize,
        );
        let rounds = spans
            .iter()
            .map(|&(a, b)| (b as usize - a as usize) / 2)
            .min()
            .unwrap_or(0);
        // SAFETY throughout: each lane reads pairs inside its own span
        // (j < min pairs), and index invariants as `walk_row_class`.
        for j in 0..rounds {
            unsafe {
                let step = |base: usize, s: usize| -> usize {
                    let c1 = *codes.get_unchecked(base + 2 * j) as usize;
                    let c2 = *codes.get_unchecked(base + 2 * j + 1) as usize;
                    let k1 = *map.add(c1) as usize;
                    let k2 = *map.add(c2) as usize;
                    *t2.get_unchecked((k1 * nclasses + k2) * nstates + s) as usize
                };
                s0 = step(a0, s0);
                s1 = step(a1, s1);
                s2 = step(a2, s2);
                s3 = step(a3, s3);
            }
            if s0.min(s1).min(s2).min(s3) == accept {
                return [true; 4]; // accept is the largest compact state
            }
        }
        // Per-lane tails (longer rows, odd trailing code), sequential.
        let mut out = [false; 4];
        for (l, &st) in [s0, s1, s2, s3].iter().enumerate() {
            let (a, b) = (spans[l].0 as usize, spans[l].1 as usize);
            out[l] = st == accept
                || self.finish_pair(&codes[a + 2 * rounds..b], st);
        }
        out
    }

    /// Continue a pair walk from state `s` over the remaining codes.
    #[inline]
    fn finish_pair(&self, codes: &[u16], s: usize) -> bool {
        let t2 = self.trans2.as_deref().expect("pair walk requires trans2");
        let accept = self.accept as usize;
        let nstates = self.nstates;
        let nclasses = self.nclasses;
        let map = self.class_map8.as_ptr();
        let mut s = s;
        let n2 = codes.len() & !1;
        let mut i = 0usize;
        while i < n2 {
            // SAFETY: i + 1 < codes.len(); index invariants as
            // `walk_row_class`.
            unsafe {
                let both = codes.as_ptr().add(i).cast::<u32>().read_unaligned();
                let k1 = *map.add((both & 0xFFFF) as usize) as usize;
                let k2 = *map.add((both >> 16) as usize) as usize;
                s = *t2.get_unchecked((k1 * nclasses + k2) * nstates + s) as usize;
            }
            if s == accept {
                return true;
            }
            i += 2;
        }
        if let Some(&c) = codes.get(n2) {
            // SAFETY: index invariants as `walk_row_class`.
            unsafe {
                let k = *map.add(c as usize) as usize;
                s = *self.trans.get_unchecked(k * nstates + s) as usize;
            }
        }
        s == accept
    }

    /// Four interleaved Hyperflex pair walks: four independent `VPERMB`
    /// chains fill the shuffle pipeline instead of waiting out its latency.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
    unsafe fn walk4_hyperflex_pair(&self, codes: &[u16], spans: [(u32, u32); 4]) -> [bool; 4] {
        use std::arch::x86_64::*;
        let rows2 = self.hf_rows2.as_deref().expect("hyperflex pair rows").as_ptr();
        let accept = self.accept as usize;
        let nclasses = self.nclasses;
        let map = self.class_map8.as_ptr();
        if codes.len() < 2 {
            // SAFETY: caller checked VBMI.
            return spans
                .map(|(a, b)| unsafe { self.walk_row_hyperflex_pair(&codes[a as usize..b as usize]) });
        }

        let mut sv = [_mm512_setzero_si512(); 4];
        let mut pos = spans.map(|(a, _)| a as usize);
        let mut rem = spans.map(|(a, b)| (b as usize - a as usize) / 2);
        let rounds = rem.into_iter().max().unwrap_or(0);
        for _ in 0..rounds {
            for l in 0..4 {
                let active = rem[l] > 0;
                let p = if active { pos[l] } else { 0 };
                // SAFETY: as `walk4_pair`.
                let next = unsafe {
                    let c1 = *codes.get_unchecked(p) as usize;
                    let c2 = *codes.get_unchecked(p + 1) as usize;
                    let k1 = *map.add(c1) as usize;
                    let k2 = *map.add(c2) as usize;
                    let row = _mm512_loadu_si512(rows2.add((k1 * nclasses + k2) * 64).cast());
                    _mm512_permutexvar_epi8(sv[l], row)
                };
                sv[l] = if active { next } else { sv[l] };
                pos[l] += 2 * usize::from(active);
                rem[l] -= usize::from(active);
            }
        }
        let mut out = [false; 4];
        for l in 0..4 {
            let mut s = _mm512_cvtsi512_si32(sv[l]) as u8 as usize;
            let (_, b) = spans[l];
            if pos[l] < b as usize {
                // SAFETY: index invariants as `walk_row_class`.
                unsafe {
                    let k = *map.add(*codes.get_unchecked(pos[l]) as usize) as usize;
                    s = *self.trans.get_unchecked(k * self.nstates + s) as usize;
                }
            }
            out[l] = s == accept;
        }
        out
    }

    /// Interesting-positions mask AND-ed with its shifts (see
    /// [`super::sparse`]).
    fn combined_mask(&self, codes: &[u16]) -> Vec<u64> {
        let words = codes.len().div_ceil(64);
        let mut mask = vec![0u64; words];
        self.filter.candidate_mask(codes, &mut mask);
        if words == 0 {
            return mask;
        }
        let tail = codes.len() % 64;
        if tail != 0 {
            mask[words - 1] &= !0u64 >> (64 - tail);
        }
        let mut combined = mask.clone();
        for shift in 1..self.combo {
            for w in 0..words {
                let lo = mask[w] >> shift;
                let hi = if w + 1 < words {
                    mask[w + 1] << (64 - shift)
                } else {
                    0
                };
                combined[w] &= lo | hi;
            }
        }
        combined
    }
}

/// Runtime gate for the Hyperflex walks.
fn avx512_vbmi() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx512vbmi")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Decode one `code_offsets` window into a checked `[a, b)` span.
#[inline]
fn span<O: Offset>(w: &[O], len: usize) -> (usize, usize) {
    let a = w[0].to_usize().expect("row offset overflows usize");
    let b = w[1].to_usize().expect("row offset overflows usize");
    assert!(a <= b && b <= len, "malformed code_offsets");
    (a, b)
}

#[cfg(test)]
mod tests {
    use super::super::ContainsSearcher;
    use super::*;

    fn dict(tokens: &[&[u8]]) -> (Vec<u8>, Vec<u32>) {
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for t in tokens {
            bytes.extend_from_slice(t);
            offsets.push(bytes.len() as u32);
        }
        let last = offsets[offsets.len().saturating_sub(2)] as usize;
        bytes.resize(last + crate::MAX_TOKEN_SIZE, 0);
        (bytes, offsets)
    }

    fn all_walks(s: &ClassSearcher) -> Vec<Walk> {
        [Walk::Class, Walk::Pair, Walk::Hyperflex, Walk::HyperflexPair]
            .into_iter()
            .filter(|&w| s.supports(w))
            .collect()
    }

    /// Every supported walk — filtered, unfiltered, and 4-way interleaved —
    /// must agree.
    fn rows(s: &ClassSearcher, codes: &[u16], offsets: &[u32]) -> Vec<u64> {
        let expect = s.matching_rows_unfiltered(codes, offsets, Walk::Class);
        let spans: Vec<(u32, u32)> = offsets.windows(2).map(|w| (w[0], w[1])).collect();
        for w in all_walks(s) {
            assert_eq!(
                s.matching_rows_unfiltered(codes, offsets, w),
                expect,
                "unfiltered {w:?} disagrees"
            );
            assert_eq!(
                s.matching_rows(codes, offsets, w),
                expect,
                "filtered {w:?} disagrees"
            );
            let ilv: Vec<u64> = s
                .matching_spans_ilv4(codes, &spans, w)
                .iter()
                .enumerate()
                .filter_map(|(i, &m)| m.then_some(i as u64))
                .collect();
            assert_eq!(ilv, expect, "ilv4 {w:?} disagrees");
        }
        assert_eq!(
            s.matching_rows_skip(codes, offsets),
            expect,
            "fused skip scan disagrees"
        );
        expect
    }

    #[test]
    fn matches_within_and_across_tokens() {
        let tokens: &[&[u8]] = &[b"ab", b"cd", b"abc", b"x"];
        let (bytes, offsets) = dict(tokens);
        let s = ClassSearcher::compile_dict(&bytes, &offsets, b"bcd");
        assert_eq!(rows(&s, &[0, 1, 1, 0, 3, 0, 1], &[0, 2, 4, 7]), vec![0, 2]);
        assert_eq!(rows(&s, &[2, 1, 2, 3, 0, 1], &[0, 2, 6]), vec![1]);
        assert!(rows(&s, &[2], &[0, 1]).is_empty());
    }

    #[test]
    fn overlapping_prefix_suffix() {
        let tokens: &[&[u8]] = &[b"ab", b"a", b"b"];
        let (bytes, offsets) = dict(tokens);
        let s = ClassSearcher::compile_dict(&bytes, &offsets, b"aba");
        assert_eq!(
            rows(&s, &[0, 1, 1, 2, 1, 0, 2, 0, 0, 1], &[0, 2, 5, 7, 10]),
            vec![0, 1, 3]
        );
    }

    #[test]
    fn empty_pattern_matches_everything() {
        let (bytes, offsets) = dict(&[b"ab"]);
        let s = ClassSearcher::compile_dict(&bytes, &offsets, b"");
        assert_eq!(
            s.matching_rows(&[0, 0], &[0u32, 1, 2], Walk::Class),
            vec![0, 1]
        );
        assert!(s.info().is_none());
    }

    #[test]
    fn class_count_is_tiny() {
        let tokens: &[&[u8]] = &[b"ab", b"cd", b"abc", b"x", b"yz", b"qq", b"zz"];
        let (bytes, offsets) = dict(tokens);
        let s = ClassSearcher::compile_dict(&bytes, &offsets, b"bcd");
        let info = s.info().expect("compiled");
        assert!(info.nclasses < tokens.len() + 1, "classes must collapse");
        assert!(info.nstates <= 4, "only reachable states kept");
        assert!(info.trans2_bytes > 0, "pair table fits well under the cap");
    }

    /// Deterministic pseudo-random cross-check against the dense pipeline
    /// over many dictionaries, patterns, and code streams.
    #[test]
    fn cross_check_dense() {
        let mut x = 0x2545F4914F6CDD1Du64;
        let mut rng = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let alphabet = b"abcx";
        for _ in 0..200 {
            let ntokens = 3 + (rng() % 6) as usize;
            let tokens: Vec<Vec<u8>> = (0..ntokens)
                .map(|_| {
                    let len = 1 + (rng() % 4) as usize;
                    (0..len)
                        .map(|_| alphabet[(rng() % alphabet.len() as u64) as usize])
                        .collect()
                })
                .collect();
            let refs: Vec<&[u8]> = tokens.iter().map(Vec::as_slice).collect();
            let (bytes, offsets) = dict(&refs);

            let plen = 1 + (rng() % 6) as usize;
            let pattern: Vec<u8> = (0..plen)
                .map(|_| alphabet[(rng() % alphabet.len() as u64) as usize])
                .collect();

            let mut codes = Vec::new();
            let mut code_offsets = vec![0u32];
            for _ in 0..20 {
                for _ in 0..(rng() % 8) {
                    codes.push((rng() % ntokens as u64) as u16);
                }
                code_offsets.push(codes.len() as u32);
            }

            let class = ClassSearcher::compile_dict(&bytes, &offsets, &pattern);
            let dense = ContainsSearcher::compile_heuristic(&bytes, &offsets, &pattern);
            let expect = dense.matching_rows_unfiltered(&codes, &code_offsets);
            assert_eq!(
                rows(&class, &codes, &code_offsets),
                expect,
                "pattern {:?} tokens {:?}",
                String::from_utf8_lossy(&pattern),
                tokens
                    .iter()
                    .map(|t| String::from_utf8_lossy(t).into_owned())
                    .collect::<Vec<_>>(),
            );
        }
    }
}
