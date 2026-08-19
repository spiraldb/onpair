//! General-cover lab: does the superblock technique win for point/range
//! covers (P points + R inclusive ranges, P+R <= 8)?
//!
//! Contenders:
//!   scalar: a1t rows-table early-exit (reference; = shipped RowsTable),
//!           b1t code-centric branchy table (= shipped CodesTable),
//!           c4p superblock w/ probe-fold pass 1 (code-outer),
//!           c4po superblock w/ probe-outer pass 1,
//!           c4t superblock w/ table-fold pass 1
//!   avx512: r2g shipped generic (2-compare ranges, per-32 extract),
//!           r2gs generic w/ sub+cmple single-compare ranges,
//!           r6g256/r6g512 superblock (sub-trick ranges, retained u64 masks,
//!           one gate per 256/512 codes)
//!
//! Modes:
//!   cover-lab synth
//!   cover-lab real <codes.bin> <row_offsets.u32> <covers.tsv>

#![allow(clippy::needless_range_loop)]

use std::hint::black_box;
use std::time::Instant;

const REPS: usize = 3;

struct Cover {
    points: Vec<u16>,
    ranges: Vec<(u16, u16)>, // inclusive (lo, hi)
    table: Vec<u8>,          // 64K membership
}

impl Cover {
    fn build(points: Vec<u16>, ranges: Vec<(u16, u16)>) -> Cover {
        let mut table = vec![0u8; 1 << 16];
        for &p in &points {
            table[p as usize] = 1;
        }
        for &(lo, hi) in &ranges {
            for v in lo..=hi {
                table[v as usize] = 1;
            }
        }
        Cover { points, ranges, table }
    }
    fn covered_values(&self) -> Vec<u16> {
        (0..=u16::MAX).filter(|&v| self.table[v as usize] != 0).collect()
    }
}

struct Input {
    codes: Vec<u16>,
    off: Vec<u32>,
    sparse: bool,
}

impl Input {
    fn rows(&self) -> usize {
        self.off.len() - 1
    }
}

struct Scratch {
    blockany: Vec<u8>,
}

// ───────────────────────────────────────────────────────────────────── sink

struct Sink<'a> {
    off: &'a [u32],
    out: &'a mut Vec<usize>,
    row: usize,
    row_end: usize,
    sparse: bool,
}

impl<'a> Sink<'a> {
    fn new(off: &'a [u32], out: &'a mut Vec<usize>, sparse: bool) -> Self {
        Sink { off, out, row: 0, row_end: 0, sparse }
    }
    #[inline]
    fn hit(&mut self, i: usize) {
        if i < self.row_end {
            return;
        }
        const GAP: usize = 128;
        if self.sparse && i.saturating_sub(self.row_end) >= GAP {
            let suffix = &self.off[self.row + 1..];
            self.row += suffix.partition_point(|&o| (o as usize) <= i);
        } else {
            while (self.off[self.row + 1] as usize) <= i {
                self.row += 1;
            }
        }
        self.out.push(self.row);
        self.row_end = self.off[self.row + 1] as usize;
    }
    #[inline]
    fn mark_mask(&mut self, base: usize, mut lanes: u64) {
        loop {
            let consumed = self.row_end.saturating_sub(base);
            if consumed >= 64 {
                return;
            }
            lanes &= u64::MAX << consumed;
            if lanes == 0 {
                return;
            }
            self.hit(base + lanes.trailing_zeros() as usize);
        }
    }
}

fn tail_table(cover: &Cover, inp: &Input, from: usize, sink: &mut Sink) {
    for (o, &c) in inp.codes[from..].iter().enumerate() {
        if cover.table[c as usize] != 0 {
            sink.hit(from + o);
        }
    }
}

// ─────────────────────────────────────────────────────────────────── scalar

/// Reference + shipped RowsTable: row-centric early-exit over the table.
fn a1t_rows_table(cover: &Cover, inp: &Input, _s: &mut Scratch, out: &mut Vec<usize>) {
    for r in 0..inp.rows() {
        let (a, b) = (inp.off[r] as usize, inp.off[r + 1] as usize);
        if inp.codes[a..b].iter().any(|&c| cover.table[c as usize] != 0) {
            out.push(r);
        }
    }
}

/// Shipped CodesTable: code-centric branchy table probe into the sink.
fn b1t_codes_table(cover: &Cover, inp: &Input, _s: &mut Scratch, out: &mut Vec<usize>) {
    let mut sink = Sink::new(&inp.off, out, inp.sparse);
    for (i, &c) in inp.codes.iter().enumerate() {
        if cover.table[c as usize] != 0 {
            sink.hit(i);
        }
    }
}

#[inline]
fn hit_probes(c: u16, pts: &[u16], rgs: &[(u16, u16)]) -> bool {
    let mut any = false;
    for &p in pts {
        any |= c == p;
    }
    for &(lo, hi) in rgs {
        any |= c.wrapping_sub(lo) <= hi.wrapping_sub(lo);
    }
    any
}

/// Superblock, pass 1 = probe-fold per code (code-outer, probe-inner).
fn c4p_sb512(cover: &Cover, inp: &Input, s: &mut Scratch, out: &mut Vec<usize>) {
    const G: usize = 512;
    let n = inp.codes.len();
    let full = n / G;
    for sb in 0..full {
        let base = sb * G;
        let any = inp.codes[base..base + G]
            .iter()
            .fold(false, |acc, &c| acc | hit_probes(c, &cover.points, &cover.ranges));
        s.blockany[sb] = any as u8;
    }
    let mut sink = Sink::new(&inp.off, out, inp.sparse);
    for sb in 0..full {
        if s.blockany[sb] != 0 {
            let base = sb * G;
            for j in 0..G {
                if cover.table[inp.codes[base + j] as usize] != 0 {
                    sink.hit(base + j);
                }
            }
        }
    }
    tail_table(cover, inp, full * G, &mut sink);
}

/// Superblock, pass 1 = probe-outer: one autovectorizable sweep per probe.
fn c4po_sb512(cover: &Cover, inp: &Input, s: &mut Scratch, out: &mut Vec<usize>) {
    const G: usize = 512;
    let n = inp.codes.len();
    let full = n / G;
    let mut anyb = [0u8; G];
    for sb in 0..full {
        let base = sb * G;
        let blk = &inp.codes[base..base + G];
        anyb.fill(0);
        for &p in &cover.points {
            for j in 0..G {
                anyb[j] |= (blk[j] == p) as u8;
            }
        }
        for &(lo, hi) in &cover.ranges {
            let span = hi.wrapping_sub(lo);
            for j in 0..G {
                anyb[j] |= (blk[j].wrapping_sub(lo) <= span) as u8;
            }
        }
        s.blockany[sb] = anyb.iter().fold(0u8, |a, &b| a | b);
    }
    let mut sink = Sink::new(&inp.off, out, inp.sparse);
    for sb in 0..full {
        if s.blockany[sb] != 0 {
            let base = sb * G;
            for j in 0..G {
                if cover.table[inp.codes[base + j] as usize] != 0 {
                    sink.hit(base + j);
                }
            }
        }
    }
    tail_table(cover, inp, full * G, &mut sink);
}

/// Superblock, pass 1 = table-fold per code (shape-independent cost).
fn c4t_sb512(cover: &Cover, inp: &Input, s: &mut Scratch, out: &mut Vec<usize>) {
    const G: usize = 512;
    let n = inp.codes.len();
    let full = n / G;
    for sb in 0..full {
        let base = sb * G;
        let any = inp.codes[base..base + G]
            .iter()
            .fold(0u8, |acc, &c| acc | cover.table[c as usize]);
        s.blockany[sb] = any;
    }
    let mut sink = Sink::new(&inp.off, out, inp.sparse);
    for sb in 0..full {
        if s.blockany[sb] != 0 {
            let base = sb * G;
            for j in 0..G {
                if cover.table[inp.codes[base + j] as usize] != 0 {
                    sink.hit(base + j);
                }
            }
        }
    }
    tail_table(cover, inp, full * G, &mut sink);
}

// ────────────────────────────────────────────── autovec (no intrinsics) ─────
// Portable Rust only; vector width comes from the compiler's -C target-cpu.

/// Rescan a live block plus the tail through the membership table.
#[inline]
fn rescan_live_blocks<const G: usize>(
    cover: &Cover,
    inp: &Input,
    s: &Scratch,
    full: usize,
    out: &mut Vec<usize>,
) {
    let mut sink = Sink::new(&inp.off, out, inp.sparse);
    for sb in 0..full {
        if s.blockany[sb] != 0 {
            let base = sb * G;
            for j in 0..G {
                if cover.table[inp.codes[base + j] as usize] != 0 {
                    sink.hit(base + j);
                }
            }
        }
    }
    tail_table(cover, inp, full * G, &mut sink);
}

/// av_c4t: superblock G=256, pass 1 = table-fold (shape-independent).
#[inline(never)]
fn av_c4t256(cover: &Cover, inp: &Input, s: &mut Scratch, out: &mut Vec<usize>) {
    const G: usize = 256;
    let full = inp.codes.len() / G;
    for sb in 0..full {
        let base = sb * G;
        let any = inp.codes[base..base + G]
            .iter()
            .fold(0u8, |acc, &c| acc | cover.table[c as usize]);
        s.blockany[sb] = any;
    }
    rescan_live_blocks::<G>(cover, inp, s, full, out);
}

/// av_shape: superblock of G codes, pass 1 = code-outer probe fold,
/// monomorphized per cover shape so the probe loops fully unroll and the code
/// loop auto-vectorizes as an OR-reduction.
#[inline(never)]
fn av_shape_impl<const P: usize, const R: usize, const G: usize>(
    cover: &Cover,
    inp: &Input,
    s: &mut Scratch,
    out: &mut Vec<usize>,
) {
    let mut pts = [0u16; P];
    pts.copy_from_slice(&cover.points);
    let mut los = [0u16; R];
    let mut spans = [0u16; R];
    for k in 0..R {
        los[k] = cover.ranges[k].0;
        spans[k] = cover.ranges[k].1.wrapping_sub(cover.ranges[k].0);
    }
    let full = inp.codes.len() / G;
    for sb in 0..full {
        let blk = &inp.codes[sb * G..sb * G + G];
        let mut any = false;
        for &c in blk {
            let mut h = false;
            for k in 0..P {
                h |= c == pts[k];
            }
            for k in 0..R {
                h |= c.wrapping_sub(los[k]) <= spans[k];
            }
            any |= h;
        }
        s.blockany[sb] = any as u8;
    }
    if G >= 256 {
        rescan_live_blocks::<G>(cover, inp, s, full, out);
    } else {
        // Fine summaries: stride the summary bytes eight at a time so eight
        // dead blocks cost one u64 compare, and rescan only live G-blocks.
        let mut sink = Sink::new(&inp.off, out, inp.sparse);
        let mut sb = 0usize;
        while sb + 8 <= full {
            let word = u64::from_le_bytes(s.blockany[sb..sb + 8].try_into().unwrap());
            if word != 0 {
                for k in 0..8 {
                    if s.blockany[sb + k] != 0 {
                        let base = (sb + k) * G;
                        for j in 0..G {
                            if cover.table[inp.codes[base + j] as usize] != 0 {
                                sink.hit(base + j);
                            }
                        }
                    }
                }
            }
            sb += 8;
        }
        for b in sb..full {
            if s.blockany[b] != 0 {
                let base = b * G;
                for j in 0..G {
                    if cover.table[inp.codes[base + j] as usize] != 0 {
                        sink.hit(base + j);
                    }
                }
            }
        }
        tail_table(cover, inp, full * G, &mut sink);
    }
}

/// Dispatch to a monomorphized shape when we have one; table-fold otherwise.
fn av_shape_g<const G: usize>(cover: &Cover, inp: &Input, s: &mut Scratch, out: &mut Vec<usize>) {
    macro_rules! arms {
        ($(($p:literal, $r:literal)),+ $(,)?) => {
            match (cover.points.len(), cover.ranges.len()) {
                $(($p, $r) => return av_shape_impl::<$p, $r, G>(cover, inp, s, out),)+
                _ => av_c4t256(cover, inp, s, out),
            }
        };
    }
    arms!(
        (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (8, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2),
        (0, 3), (1, 3), (2, 3), (0, 4), (2, 4), (4, 4),
    )
}

fn av_shape(cover: &Cover, inp: &Input, s: &mut Scratch, out: &mut Vec<usize>) {
    av_shape_g::<256>(cover, inp, s, out)
}

fn av_shape64(cover: &Cover, inp: &Input, s: &mut Scratch, out: &mut Vec<usize>) {
    av_shape_g::<64>(cover, inp, s, out)
}

// ─────────────────────────────────────────────────────────────────── avx512

#[cfg(target_arch = "x86_64")]
mod simd {
    use super::{tail_table, Cover, Input, Scratch, Sink};
    use core::arch::x86_64::*;

    struct Probes {
        points: [__m512i; 64],
        np: usize,
        lo: [__m512i; 32],
        span: [__m512i; 32],
        ge: [__m512i; 32],
        le: [__m512i; 32],
        nr: usize,
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn broadcast(cover: &Cover) -> Probes {
        let zero = _mm512_setzero_si512();
        let mut p = Probes {
            points: [zero; 64],
            np: cover.points.len(),
            lo: [zero; 32],
            span: [zero; 32],
            ge: [zero; 32],
            le: [zero; 32],
            nr: cover.ranges.len(),
        };
        for (k, &pt) in cover.points.iter().enumerate() {
            p.points[k] = _mm512_set1_epi16(pt as i16);
        }
        for (k, &(lo, hi)) in cover.ranges.iter().enumerate() {
            p.lo[k] = _mm512_set1_epi16(lo as i16);
            p.span[k] = _mm512_set1_epi16(hi.wrapping_sub(lo) as i16);
            p.ge[k] = _mm512_set1_epi16(lo as i16);
            p.le[k] = _mm512_set1_epi16(hi as i16);
        }
        p
    }

    /// One vector's hit mask, shipped shape: cmpeq per point, cmpge+cmple+kand
    /// per range.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn mask_2cmp(v: __m512i, p: &Probes) -> u32 {
        let mut acc: u32 = 0;
        for k in 0..p.np {
            acc |= _mm512_cmpeq_epu16_mask(v, p.points[k]);
        }
        for k in 0..p.nr {
            let ge = _mm512_cmpge_epu16_mask(v, p.ge[k]);
            let le = _mm512_cmple_epu16_mask(v, p.le[k]);
            acc |= ge & le;
        }
        acc
    }

    /// One vector's hit mask, sub-trick ranges: (v - lo) <=u span.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn mask_sub(v: __m512i, p: &Probes) -> u32 {
        let mut acc: u32 = 0;
        for k in 0..p.np {
            acc |= _mm512_cmpeq_epu16_mask(v, p.points[k]);
        }
        for k in 0..p.nr {
            let d = _mm512_sub_epi16(v, p.lo[k]);
            acc |= _mm512_cmple_epu16_mask(d, p.span[k]);
        }
        acc
    }

    /// r2g: the shipped generic loop — per-32 mask, immediate per-bit extract.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn r2g<const SUB: bool>(
        cover: &Cover,
        inp: &Input,
        _s: &mut Scratch,
        out: &mut Vec<usize>,
    ) {
        let n = inp.codes.len();
        let base = inp.codes.as_ptr();
        let p = broadcast(cover);
        let mut sink = Sink::new(&inp.off, out, inp.sparse);
        let mut i = 0usize;
        while i + 32 <= n {
            let v = _mm512_loadu_si512(base.add(i).cast());
            let mut m = if SUB { mask_sub(v, &p) } else { mask_2cmp(v, &p) };
            while m != 0 {
                let j = m.trailing_zeros() as usize;
                sink.hit(i + j);
                m &= m - 1;
            }
            i += 32;
        }
        tail_table(cover, inp, i, &mut sink);
    }

    /// r6g: superblock — V vectors' masks collapsed to u64 pairs, one gate per
    /// V*32 codes, extraction from the retained masks. Sub-trick ranges.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn r6g<const V: usize>(
        cover: &Cover,
        inp: &Input,
        _s: &mut Scratch,
        out: &mut Vec<usize>,
    ) {
        debug_assert!(V % 2 == 0);
        let n = inp.codes.len();
        let base = inp.codes.as_ptr();
        let p = broadcast(cover);
        let mut sink = Sink::new(&inp.off, out, inp.sparse);
        let sb = V * 32;
        let mut lanes = [0u64; 16];
        let mut i = 0usize;
        while i + sb <= n {
            let mut any = 0u64;
            for k in 0..V / 2 {
                let m0 = mask_sub(_mm512_loadu_si512(base.add(i + k * 64).cast()), &p);
                let m1 = mask_sub(_mm512_loadu_si512(base.add(i + k * 64 + 32).cast()), &p);
                let pair = (m0 as u64) | ((m1 as u64) << 32);
                lanes[k] = pair;
                any |= pair;
            }
            if any != 0 {
                for k in 0..V / 2 {
                    if lanes[k] != 0 {
                        sink.mark_mask(i + k * 64, lanes[k]);
                    }
                }
            }
            i += sb;
        }
        tail_table(cover, inp, i, &mut sink);
    }
}

// ─────────────────────────────────────────────────────────────────── harness

type ImplFn = fn(&Cover, &Input, &mut Scratch, &mut Vec<usize>);

struct Algo {
    name: &'static str,
    f: ImplFn,
}

fn registry() -> Vec<Algo> {
    // KERNELS_ONLY: the shipped-original compare loop plus the five strongest
    // new scan kernels, no scalar exploratory variants.
    let mut v: Vec<Algo> = vec![
        Algo { name: "rows_tbl", f: a1t_rows_table },
        Algo { name: "av_c4t", f: av_c4t256 },
        Algo { name: "av_shape", f: av_shape },
        Algo { name: "av_shp64", f: av_shape64 },
    ];
    if std::env::var_os("LAB_ALL_SCALARS").is_some() {
        v.push(Algo { name: "b1t_codes", f: b1t_codes_table });
        v.push(Algo { name: "c4p_512", f: c4p_sb512 });
        v.push(Algo { name: "c4po_512", f: c4po_sb512 });
        v.push(Algo { name: "c4t_512", f: c4t_sb512 });
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
            v.push(Algo { name: "ORIGINAL", f: |c, i, s, o| unsafe { simd::r2g::<false>(c, i, s, o) } });
            v.push(Algo { name: "r2g_sub", f: |c, i, s, o| unsafe { simd::r2g::<true>(c, i, s, o) } });
            v.push(Algo { name: "r6g_64", f: |c, i, s, o| unsafe { simd::r6g::<2>(c, i, s, o) } });
            v.push(Algo { name: "r6g_256", f: |c, i, s, o| unsafe { simd::r6g::<8>(c, i, s, o) } });
            v.push(Algo { name: "r6g_512", f: |c, i, s, o| unsafe { simd::r6g::<16>(c, i, s, o) } });
        }
    }
    v
}

fn measure(
    algo: &Algo,
    cover: &Cover,
    inp: &Input,
    s: &mut Scratch,
    expect: &[usize],
) -> Result<f64, String> {
    let mut out: Vec<usize> = Vec::with_capacity(inp.rows() + 64);
    (algo.f)(cover, inp, s, &mut out);
    if out != expect {
        return Err(format!("MISMATCH {} vs {}", out.len(), expect.len()));
    }
    let mut best = f64::INFINITY;
    for _ in 0..REPS {
        out.clear();
        let t = Instant::now();
        (algo.f)(black_box(cover), black_box(inp), s, &mut out);
        best = best.min(t.elapsed().as_secs_f64());
        black_box(&out);
    }
    Ok(best * 1e9 / inp.codes.len() as f64)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// Random disjoint cover of P points and R ranges (width 2..=24).
fn gen_cover(np: usize, nr: usize, rng: &mut Rng) -> Cover {
    let mut table = vec![false; 1 << 16];
    let mut points = Vec::new();
    while points.len() < np {
        let v = (rng.next() >> 16) as u16;
        if !table[v as usize] {
            table[v as usize] = true;
            points.push(v);
        }
    }
    let mut ranges = Vec::new();
    while ranges.len() < nr {
        let w = 2 + (rng.next() % 23) as u16;
        let lo = ((rng.next() >> 16) as u16).min(u16::MAX - w);
        let hi = lo + w;
        if (lo..=hi).all(|v| !table[v as usize]) {
            for v in lo..=hi {
                table[v as usize] = true;
            }
            ranges.push((lo, hi));
        }
    }
    Cover::build(points, ranges)
}

fn gen_input(n: usize, avg_row: usize, density: f64, cover: &Cover, rng: &mut Rng) -> Input {
    let covered = cover.covered_values();
    let mut codes: Vec<u16> = (0..n)
        .map(|_| loop {
            let c = (rng.next() >> 16) as u16;
            if cover.table[c as usize] == 0 {
                break c;
            }
        })
        .collect();
    let k = (n as f64 * density).round() as usize;
    for _ in 0..k {
        let pos = (rng.next() as usize) % n;
        codes[pos] = covered[(rng.next() as usize) % covered.len()];
    }
    let mut off = vec![0u32];
    let mut cum = 0usize;
    while cum < n {
        let len = 1 + (rng.next() as usize) % (2 * avg_row - 1);
        cum = (cum + len).min(n);
        off.push(cum as u32);
    }
    let hits = codes.iter().filter(|&&c| cover.table[c as usize] != 0).count();
    let sparse = (hits as f64) < (n as f64) * 1e-4;
    Input { codes, off, sparse }
}

fn run_synth() {
    let n = 16_000_000usize;
    let shapes: &[(usize, usize)] = &[
        (1, 0), (0, 1), (2, 0), (1, 1), (0, 2),
        (4, 0), (2, 2), (0, 4),
        (8, 0), (6, 2), (4, 4), (2, 6), (0, 8),
    ];
    let densities = [1e-5, 1e-3, 1e-2, 5e-2];
    let algos = registry();
    for &d in &densities {
        println!("\n== synth covers: n=16M, avg_row=11, density={d:.0e} (ns/code, best of {REPS}) ==");
        print!("{:10}", "shape P,R");
        for a in &algos {
            print!("{:>10}", a.name);
        }
        println!();
        for &(np, nr) in shapes {
            let mut rng = Rng(0xc0ffee ^ ((np * 64 + nr) as u64) << 8 | 1);
            let cover = gen_cover(np, nr, &mut rng);
            let inp = gen_input(n, 11, d, &cover, &mut rng);
            let mut expect = Vec::new();
            let mut s0 = Scratch { blockany: vec![] };
            a1t_rows_table(&cover, &inp, &mut s0, &mut expect);
            let mut scratch = Scratch { blockany: vec![0u8; n / 64 + 16] };
            print!("{:10}", format!("({np},{nr})"));
            for algo in &algos {
                match measure(algo, &cover, &inp, &mut scratch, &expect) {
                    Ok(ns) => print!("{ns:>10.3}"),
                    Err(_) => print!("{:>10}", "FAIL"),
                }
            }
            println!();
        }
    }
}

fn run_real(codes_path: &str, off_path: &str, covers_path: &str) {
    let codes: Vec<u16> = std::fs::read(codes_path)
        .unwrap()
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let off: Vec<u32> = std::fs::read(off_path)
        .unwrap()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let algos = registry();
    println!("== real covers, P+R in 1..=8 (ns/code, best of {REPS}) ==");
    print!("{:>6}{:>10}{:>8}", "P,R", "coverage", "rows");
    for a in &algos {
        print!("{:>10}", a.name);
    }
    println!();
    let mut rows_meta: Vec<(usize, usize, f64, Vec<u16>, Vec<(u16, u16)>, String)> = Vec::new();
    for line in std::fs::read_to_string(covers_path).unwrap().lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let coverage: f64 = f[1].parse().unwrap();
        let points: Vec<u16> = if f[2].is_empty() {
            vec![]
        } else {
            f[2].split(',').map(|s| s.parse().unwrap()).collect()
        };
        let ranges: Vec<(u16, u16)> = if f[3].is_empty() {
            vec![]
        } else {
            f[3].split(';')
                .map(|s| {
                    let (a, b) = s.split_once('-').unwrap();
                    (a.parse().unwrap(), b.parse().unwrap())
                })
                .collect()
        };
        let (np, nr) = (points.len(), ranges.len());
        // Kernel shootout scope: any shape up to 64 comparisons.
        if np + nr == 0 || np + 2 * nr > 64 {
            continue;
        }
        rows_meta.push((np, nr, coverage, points, ranges, f[0].to_string()));
    }
    rows_meta.sort_by(|a, b| {
        (a.0 + 2 * a.1, &a.2)
            .partial_cmp(&(b.0 + 2 * b.1, &b.2))
            .unwrap()
    });
    // Cap the run: evenly spaced sample across the (cost, coverage) order.
    const CAP: usize = 48;
    if rows_meta.len() > CAP {
        let step = rows_meta.len() as f64 / CAP as f64;
        rows_meta = (0..CAP)
            .map(|k| rows_meta[(k as f64 * step) as usize].clone())
            .collect();
    }
    for (np, nr, coverage, points, ranges, id) in rows_meta {
        let cover = Cover::build(points, ranges);
        let inp = Input { codes: codes.clone(), off: off.clone(), sparse: coverage < 1e-4 };
        let mut expect = Vec::new();
        let mut s0 = Scratch { blockany: vec![] };
        a1t_rows_table(&cover, &inp, &mut s0, &mut expect);
        let mut scratch = Scratch { blockany: vec![0u8; inp.codes.len() / 64 + 16] };
        print!("{:>6}{coverage:>10.6}{:>8}", format!("{np},{nr}"), expect.len());
        for algo in &algos {
            match measure(algo, &cover, &inp, &mut scratch, &expect) {
                Ok(ns) => print!("{ns:>10.3}"),
                Err(_) => print!("{:>10}", "FAIL"),
            }
        }
        println!("  {id}");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("real") => run_real(&args[2], &args[3], &args[4]),
        _ => run_synth(),
    }
}
