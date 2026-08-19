//! Algorithm lab for the single-point prefilter scan.
//!
//! Contract under test: given `codes: &[u16]`, `row_offsets: &[u32]` and one
//! point `p`, append the ascending, deduplicated indices of rows holding at
//! least one code equal to `p`.
//!
//! Families (see the naming in the report):
//!   a* — one pass, row-centric (row id free; outer loop over rows)
//!   b* — one pass, code-centric (linear over codes; row via cursor)
//!   c* — two passes over hits (materialize, then stitch rows)
//!   sw* — SWAR detection plugged into the c-shapes
//!   r* — SIMD references (r1 avx2 best-known, r2 avx512 as shipped,
//!        r3 avx512 fused-64, r4 avx512 compress-store, r5 avx512 row-centric)
//!
//! Modes:
//!   point-scan-lab synth
//!   point-scan-lab real <codes.bin> <row_offsets.u32> <points.tsv>
//!     points.tsv lines: <token>\t<covered_fraction>\t<label>

#![allow(clippy::needless_range_loop)]

use std::hint::black_box;
use std::time::Instant;

const REPS: usize = 3;

// ─────────────────────────────────────────────────────────────── input/scratch

struct Input {
    codes: Vec<u16>,
    off: Vec<u32>, // row offsets, len = rows + 1, off[0] = 0
    sparse: bool,  // mirror of onpair's AdaptiveSparse row mapping (<1e-4 covered)
}

impl Input {
    fn rows(&self) -> usize {
        self.off.len() - 1
    }
}

struct Scratch {
    bitset: Vec<u64>,
    idx: Vec<u32>,
    rowbits: Vec<u8>,
    blockany: Vec<u8>,
}

impl Scratch {
    fn for_input(inp: &Input) -> Self {
        let n = inp.codes.len();
        Scratch {
            bitset: vec![0u64; n / 64 + 2],
            idx: vec![0u32; n + 64],
            rowbits: vec![0u8; inp.rows() + 1],
            blockany: vec![0u8; n / 128 + 2],
        }
    }
}

// ───────────────────────────────────────────────────────────────────── sink
// Faithful copy of onpair's RowSink: ascending code index -> ascending deduped
// rows; binary search across big gaps when the cover is sparse.

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

    /// onpair's mark_mask: emit rows for a 64-lane hit mask at `base`.
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

/// Branchy scalar tail from `from`, shared by all blocked kernels.
fn tail(inp: &Input, p: u16, from: usize, sink: &mut Sink) {
    for (o, &c) in inp.codes[from..].iter().enumerate() {
        if c == p {
            sink.hit(from + o);
        }
    }
}

// ──────────────────────────────────────────────────────── family A: row-centric

/// a1: per row, branchy early exit on first hit. Also the correctness reference.
fn a1_row_early(inp: &Input, p: u16, _s: &mut Scratch, out: &mut Vec<usize>) {
    for r in 0..inp.rows() {
        let (a, b) = (inp.off[r] as usize, inp.off[r + 1] as usize);
        for i in a..b {
            if inp.codes[i] == p {
                out.push(r);
                break;
            }
        }
    }
}

/// a2: per row, branchless OR-fold + push-over-top (speculative write).
fn a2_row_any_pushover(inp: &Input, p: u16, _s: &mut Scratch, out: &mut Vec<usize>) {
    out.reserve(inp.rows());
    let ptr = out.as_mut_ptr();
    let mut len = 0usize;
    for r in 0..inp.rows() {
        let (a, b) = (inp.off[r] as usize, inp.off[r + 1] as usize);
        let any = inp.codes[a..b].iter().fold(false, |acc, &c| acc | (c == p));
        // SAFETY: capacity reserved to rows() above; len <= r < rows().
        unsafe { *ptr.add(len) = r };
        len += any as usize;
    }
    // SAFETY: len entries initialized above.
    unsafe { out.set_len(len) };
}

/// a3: per row, branchless OR-fold + unconditional row-bitmap write, then a
/// row-sweep collects indices.
fn a3_row_any_bitmap(inp: &Input, p: u16, s: &mut Scratch, out: &mut Vec<usize>) {
    for r in 0..inp.rows() {
        let (a, b) = (inp.off[r] as usize, inp.off[r + 1] as usize);
        let any = inp.codes[a..b].iter().fold(false, |acc, &c| acc | (c == p));
        s.rowbits[r] = any as u8;
    }
    for r in 0..inp.rows() {
        if s.rowbits[r] != 0 {
            out.push(r);
        }
    }
}

/// a4: per row, branchless fold over 16-code chunks with early exit between chunks.
fn a4_row_chunked(inp: &Input, p: u16, _s: &mut Scratch, out: &mut Vec<usize>) {
    for r in 0..inp.rows() {
        let (a, b) = (inp.off[r] as usize, inp.off[r + 1] as usize);
        let row = &inp.codes[a..b];
        let mut found = false;
        for chunk in row.chunks(16) {
            if chunk.iter().fold(false, |acc, &c| acc | (c == p)) {
                found = true;
                break;
            }
        }
        if found {
            out.push(r);
        }
    }
}

// ─────────────────────────────────────────────────────── family B: code-centric

/// b1: linear over codes, branchy hit into the sink (the shipped scalar model).
fn b1_codes_branchy(inp: &Input, p: u16, _s: &mut Scratch, out: &mut Vec<usize>) {
    let mut sink = Sink::new(&inp.off, out, inp.sparse);
    for (i, &c) in inp.codes.iter().enumerate() {
        if c == p {
            sink.hit(i);
        }
    }
}

/// b2: linear over codes, cursor advanced every code, branchless bitmap write;
/// includes the zeroing + collection sweep it needs.
fn b2_codes_cursor_bitmap(inp: &Input, p: u16, s: &mut Scratch, out: &mut Vec<usize>) {
    s.rowbits[..inp.rows()].fill(0);
    let mut row = 0usize;
    let mut next = inp.off[1] as usize;
    for (i, &c) in inp.codes.iter().enumerate() {
        while next <= i {
            row += 1;
            next = inp.off[row + 1] as usize;
        }
        s.rowbits[row] |= (c == p) as u8;
    }
    for r in 0..inp.rows() {
        if s.rowbits[r] != 0 {
            out.push(r);
        }
    }
}

// ───────────────────────────────────────────────────────── family C: two-pass

#[inline]
fn build_word64(codes: &[u16], base: usize, p: u16) -> u64 {
    let mut w = 0u64;
    for j in 0..64 {
        w |= ((codes[base + j] == p) as u64) << j;
    }
    w
}

/// c1: full bitset over all codes, then a second pass extracts hit indices.
fn c1_bitset_full(inp: &Input, p: u16, s: &mut Scratch, out: &mut Vec<usize>) {
    let n = inp.codes.len();
    let full = n / 64;
    for blk in 0..full {
        s.bitset[blk] = build_word64(&inp.codes, blk * 64, p);
    }
    let mut sink = Sink::new(&inp.off, out, inp.sparse);
    for blk in 0..full {
        let w = s.bitset[blk];
        if w != 0 {
            sink.mark_mask(blk * 64, w);
        }
    }
    tail(inp, p, full * 64, &mut sink);
}

/// c2: branchless compact list of matching code indices, then map to rows.
fn c2_codeidx_list(inp: &Input, p: u16, s: &mut Scratch, out: &mut Vec<usize>) {
    let mut k = 0usize;
    for (i, &c) in inp.codes.iter().enumerate() {
        s.idx[k] = i as u32;
        k += (c == p) as usize;
    }
    let mut sink = Sink::new(&inp.off, out, inp.sparse);
    for &i in &s.idx[..k] {
        sink.hit(i as usize);
    }
}

/// c3: block-fused bitset — build each 64-code word, extract immediately.
fn c3_block64_fused(inp: &Input, p: u16, _s: &mut Scratch, out: &mut Vec<usize>) {
    let n = inp.codes.len();
    let full = n / 64;
    let mut sink = Sink::new(&inp.off, out, inp.sparse);
    for blk in 0..full {
        let w = build_word64(&inp.codes, blk * 64, p);
        if w != 0 {
            sink.mark_mask(blk * 64, w);
        }
    }
    tail(inp, p, full * 64, &mut sink);
}

/// c4: hierarchical — pass 1 stores one "any" byte per G-code superblock,
/// pass 2 rescans only the non-empty superblocks.
fn c4_superblock<const G: usize>(inp: &Input, p: u16, s: &mut Scratch, out: &mut Vec<usize>) {
    let n = inp.codes.len();
    let full = n / G;
    for sb in 0..full {
        let base = sb * G;
        let any = inp.codes[base..base + G]
            .iter()
            .fold(false, |acc, &c| acc | (c == p));
        s.blockany[sb] = any as u8;
    }
    let mut sink = Sink::new(&inp.off, out, inp.sparse);
    for sb in 0..full {
        if s.blockany[sb] != 0 {
            let base = sb * G;
            for j in 0..G {
                if inp.codes[base + j] == p {
                    sink.hit(base + j);
                }
            }
        }
    }
    tail(inp, p, full * G, &mut sink);
}

/// c5: full 1-bit-per-code bitset in pass 1 (as c1), but pass 2 skips the
/// bitset 512 codes (8 words) at a time and decodes positions from the bits
/// alone — `codes` is never touched again.
fn c5_bitset_sb512(inp: &Input, p: u16, s: &mut Scratch, out: &mut Vec<usize>) {
    let n = inp.codes.len();
    let full = n / 64;
    for blk in 0..full {
        s.bitset[blk] = build_word64(&inp.codes, blk * 64, p);
    }
    let mut sink = Sink::new(&inp.off, out, inp.sparse);
    let mut blk = 0usize;
    while blk + 8 <= full {
        let w = &s.bitset[blk..blk + 8];
        let any = w[0] | w[1] | w[2] | w[3] | w[4] | w[5] | w[6] | w[7];
        if any != 0 {
            for (o, &word) in w.iter().enumerate() {
                if word != 0 {
                    sink.mark_mask((blk + o) * 64, word);
                }
            }
        }
        blk += 8;
    }
    for b in blk..full {
        if s.bitset[b] != 0 {
            sink.mark_mask(b * 64, s.bitset[b]);
        }
    }
    tail(inp, p, full * 64, &mut sink);
}

// ──────────────────────────────────────────────────────────── SWAR detection

const LANE_LOW: u64 = 0x0001_0001_0001_0001;
const LANE_HIGH: u64 = 0x8000_8000_8000_8000;
const LANE_LOW15: u64 = !LANE_HIGH; // 0x7fff repeated

/// High bit set in each 16-bit lane of `w` that equals `p`. Exact per lane:
/// `(x & 0x7fff) + 0x7fff` sets bit 15 iff the low 15 bits are non-zero and
/// never carries across lanes; OR-ing `x` folds in bit 15 itself.
#[inline]
fn swar_eq(w: u64, pbroad: u64) -> u64 {
    let x = w ^ pbroad;
    let t = ((x & LANE_LOW15) + LANE_LOW15) | x;
    !t & LANE_HIGH
}

/// sw3: c3 with SWAR detection — skip at 4-code word granularity, resolve
/// lanes only in non-empty words.
fn sw3_word_fused(inp: &Input, p: u16, _s: &mut Scratch, out: &mut Vec<usize>) {
    let n = inp.codes.len();
    let full = n / 4;
    let pb = (p as u64) * LANE_LOW;
    let ptr = inp.codes.as_ptr();
    let mut sink = Sink::new(&inp.off, out, inp.sparse);
    for wi in 0..full {
        // SAFETY: wi*4 + 3 < n.
        let w = unsafe { (ptr.add(wi * 4) as *const u64).read_unaligned() };
        let t = swar_eq(w, pb);
        if t != 0 {
            let base = wi * 4;
            let mut m = t;
            while m != 0 {
                let lane = (m.trailing_zeros() / 16) as usize;
                sink.hit(base + lane);
                m &= m - 1;
            }
        }
    }
    tail(inp, p, full * 4, &mut sink);
}

/// sw4: c4 with SWAR detection — OR the SWAR masks across a 512-code
/// superblock for the "any" summary, rescan non-empty superblocks.
fn sw4_superblock512(inp: &Input, p: u16, s: &mut Scratch, out: &mut Vec<usize>) {
    let n = inp.codes.len();
    let full = n / 512;
    let pb = (p as u64) * LANE_LOW;
    let ptr = inp.codes.as_ptr();
    for sb in 0..full {
        let mut acc = 0u64;
        for wi in 0..128 {
            // SAFETY: sb*512 + wi*4 + 3 < n.
            let w = unsafe { (ptr.add(sb * 512 + wi * 4) as *const u64).read_unaligned() };
            acc |= swar_eq(w, pb);
        }
        s.blockany[sb] = (acc != 0) as u8;
    }
    let mut sink = Sink::new(&inp.off, out, inp.sparse);
    for sb in 0..full {
        if s.blockany[sb] != 0 {
            let base = sb * 512;
            for j in 0..512 {
                if inp.codes[base + j] == p {
                    sink.hit(base + j);
                }
            }
        }
    }
    tail(inp, p, full * 512, &mut sink);
}

// ──────────────────────────────────────────────────────────── SIMD references

#[cfg(target_arch = "x86_64")]
mod simd {
    use super::{tail, Input, Scratch, Sink};
    use core::arch::x86_64::*;

    /// r1: the shipped AVX2 one-point kernel's algorithm — 64 codes per
    /// iteration, compacted u64 mask, skip-if-zero, mark_mask extraction.
    #[target_feature(enable = "avx2")]
    pub unsafe fn r1_avx2_block64(inp: &Input, p: u16, _s: &mut Scratch, out: &mut Vec<usize>) {
        let n = inp.codes.len();
        let base = inp.codes.as_ptr();
        let point = _mm256_set1_epi16(p as i16);
        let mut sink = Sink::new(&inp.off, out, inp.sparse);
        let mut i = 0usize;
        while i + 64 <= n {
            let m0 = _mm256_cmpeq_epi16(_mm256_loadu_si256(base.add(i).cast()), point);
            let m1 = _mm256_cmpeq_epi16(_mm256_loadu_si256(base.add(i + 16).cast()), point);
            let m2 = _mm256_cmpeq_epi16(_mm256_loadu_si256(base.add(i + 32).cast()), point);
            let m3 = _mm256_cmpeq_epi16(_mm256_loadu_si256(base.add(i + 48).cast()), point);
            let lanes01 = _mm256_movemask_epi8(_mm256_permute4x64_epi64::<0xd8>(
                _mm256_packs_epi16(m0, m1),
            )) as u32 as u64;
            let lanes23 = _mm256_movemask_epi8(_mm256_permute4x64_epi64::<0xd8>(
                _mm256_packs_epi16(m2, m3),
            )) as u32 as u64;
            let lanes = lanes01 | (lanes23 << 32);
            if lanes != 0 {
                sink.mark_mask(i, lanes);
            }
            i += 64;
        }
        tail(inp, p, i, &mut sink);
    }

    /// r2: the shipped AVX-512 generic loop, restricted to one point — one
    /// 32-lane vector per iteration, per-bit extraction, no fused skip.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn r2_avx512_shipped(inp: &Input, p: u16, _s: &mut Scratch, out: &mut Vec<usize>) {
        let n = inp.codes.len();
        let base = inp.codes.as_ptr();
        let point = _mm512_set1_epi16(p as i16);
        let mut sink = Sink::new(&inp.off, out, inp.sparse);
        let mut i = 0usize;
        while i + 32 <= n {
            let v = _mm512_loadu_si512(base.add(i).cast());
            let mut m = _mm512_cmpeq_epu16_mask(v, point);
            while m != 0 {
                let j = m.trailing_zeros() as usize;
                sink.hit(i + j);
                m &= m - 1;
            }
            i += 32;
        }
        tail(inp, p, i, &mut sink);
    }

    /// r3: AVX-512, two vectors fused to a u64 mask, skip-if-zero, mark_mask.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn r3_avx512_fused64(inp: &Input, p: u16, _s: &mut Scratch, out: &mut Vec<usize>) {
        let n = inp.codes.len();
        let base = inp.codes.as_ptr();
        let point = _mm512_set1_epi16(p as i16);
        let mut sink = Sink::new(&inp.off, out, inp.sparse);
        let mut i = 0usize;
        while i + 64 <= n {
            let m0 = _mm512_cmpeq_epu16_mask(_mm512_loadu_si512(base.add(i).cast()), point);
            let m1 = _mm512_cmpeq_epu16_mask(_mm512_loadu_si512(base.add(i + 32).cast()), point);
            let lanes = (m0 as u64) | ((m1 as u64) << 32);
            if lanes != 0 {
                sink.mark_mask(i, lanes);
            }
            i += 64;
        }
        tail(inp, p, i, &mut sink);
    }

    /// r4: AVX-512 compress — pass 1 compress-stores matching code indices
    /// (cost density-independent, no bit loop), pass 2 maps indices to rows.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn r4_avx512_compress(inp: &Input, p: u16, s: &mut Scratch, out: &mut Vec<usize>) {
        let n = inp.codes.len();
        let base = inp.codes.as_ptr();
        let point = _mm512_set1_epi32(p as i32);
        let step = _mm512_set1_epi32(16);
        let mut cur = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let idxp = s.idx.as_mut_ptr();
        let mut k = 0usize;
        let mut i = 0usize;
        while i + 16 <= n {
            let v = _mm512_cvtepu16_epi32(_mm256_loadu_si256(base.add(i).cast()));
            let m = _mm512_cmpeq_epi32_mask(v, point);
            let packed = _mm512_maskz_compress_epi32(m, cur);
            _mm512_storeu_si512(idxp.add(k).cast(), packed);
            k += m.count_ones() as usize;
            cur = _mm512_add_epi32(cur, step);
            i += 16;
        }
        while i < n {
            s.idx[k] = i as u32;
            k += (*base.add(i) == p) as usize;
            i += 1;
        }
        let mut sink = Sink::new(&inp.off, out, inp.sparse);
        for &ci in &s.idx[..k] {
            sink.hit(ci as usize);
        }
    }

    /// r6: AVX-512 superblock — compare V consecutive vectors, collapse the
    /// k-masks into u64 lane bitsets, OR them all for ONE gate per V*32 codes,
    /// and extract from the retained bitsets on a live superblock. C4's coarse
    /// gate + C5's retained positions; positions are a free by-product of the
    /// compare on AVX-512, so nothing is ever rescanned.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn r6_avx512_superblock<const V: usize>(
        inp: &Input,
        p: u16,
        _s: &mut Scratch,
        out: &mut Vec<usize>,
    ) {
        debug_assert!(V % 2 == 0);
        let n = inp.codes.len();
        let base = inp.codes.as_ptr();
        let point = _mm512_set1_epi16(p as i16);
        let mut sink = Sink::new(&inp.off, out, inp.sparse);
        let sb = V * 32;
        let mut i = 0usize;
        let mut lanes = [0u64; 16];
        while i + sb <= n {
            let mut any = 0u64;
            for k in 0..V / 2 {
                let m0 =
                    _mm512_cmpeq_epu16_mask(_mm512_loadu_si512(base.add(i + k * 64).cast()), point);
                let m1 = _mm512_cmpeq_epu16_mask(
                    _mm512_loadu_si512(base.add(i + k * 64 + 32).cast()),
                    point,
                );
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
        tail(inp, p, i, &mut sink);
    }

    /// r5: AVX-512 row-centric — one masked load + compare + ktest per row,
    /// push-over-top. Only valid when every row is at most 32 codes.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn r5_avx512_rowwise(inp: &Input, p: u16, _s: &mut Scratch, out: &mut Vec<usize>) {
        let point = _mm512_set1_epi16(p as i16);
        let base = inp.codes.as_ptr();
        out.reserve(inp.rows());
        let ptr = out.as_mut_ptr();
        let mut len = 0usize;
        for r in 0..inp.rows() {
            let a = inp.off[r] as usize;
            let l = (inp.off[r + 1] as usize) - a;
            debug_assert!(l < 32);
            let mask: u32 = (1u32 << l) - 1;
            let v = _mm512_maskz_loadu_epi16(mask, base.add(a).cast());
            let m = _mm512_cmpeq_epu16_mask(v, point);
            *ptr.add(len) = r;
            len += (m != 0) as usize;
        }
        out.set_len(len);
    }
}

// ─────────────────────────────────────────────────────────────── impl registry

type ImplFn = fn(&Input, u16, &mut Scratch, &mut Vec<usize>);

struct Algo {
    name: &'static str,
    f: ImplFn,
    /// requires every row < 32 codes
    short_rows_only: bool,
}

fn registry() -> Vec<Algo> {
    let mut v: Vec<Algo> = vec![
        Algo { name: "a1_row_early", f: a1_row_early, short_rows_only: false },
        Algo { name: "a2_row_pushover", f: a2_row_any_pushover, short_rows_only: false },
        Algo { name: "a3_row_bitmap", f: a3_row_any_bitmap, short_rows_only: false },
        Algo { name: "a4_row_chunked", f: a4_row_chunked, short_rows_only: false },
        Algo { name: "b1_codes_branchy", f: b1_codes_branchy, short_rows_only: false },
        Algo { name: "b2_codes_cursor", f: b2_codes_cursor_bitmap, short_rows_only: false },
        Algo { name: "c1_bitset_full", f: c1_bitset_full, short_rows_only: false },
        Algo { name: "c2_codeidx_list", f: c2_codeidx_list, short_rows_only: false },
        Algo { name: "c3_block64_fused", f: c3_block64_fused, short_rows_only: false },
        Algo { name: "c4_sb128", f: c4_superblock::<128>, short_rows_only: false },
        Algo { name: "c4_sb512", f: c4_superblock::<512>, short_rows_only: false },
        Algo { name: "c4_sb2048", f: c4_superblock::<2048>, short_rows_only: false },
        Algo { name: "c4_sb8192", f: c4_superblock::<8192>, short_rows_only: false },
        Algo { name: "c5_bs512", f: c5_bitset_sb512, short_rows_only: false },
        Algo { name: "sw3_word_fused", f: sw3_word_fused, short_rows_only: false },
        Algo { name: "sw4_superblk512", f: sw4_superblock512, short_rows_only: false },
    ];
    #[cfg(target_arch = "x86_64")]
    {
        let avx2 = std::is_x86_feature_detected!("avx2");
        let avx512 = std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw");
        if avx2 {
            v.push(Algo {
                name: "r1_avx2_block64",
                f: |i, p, s, o| unsafe { simd::r1_avx2_block64(i, p, s, o) },
                short_rows_only: false,
            });
        }
        if avx512 {
            v.push(Algo {
                name: "r2_avx512_shipped",
                f: |i, p, s, o| unsafe { simd::r2_avx512_shipped(i, p, s, o) },
                short_rows_only: false,
            });
            v.push(Algo {
                name: "r3_avx512_fused64",
                f: |i, p, s, o| unsafe { simd::r3_avx512_fused64(i, p, s, o) },
                short_rows_only: false,
            });
            v.push(Algo {
                name: "r4_avx512_compress",
                f: |i, p, s, o| unsafe { simd::r4_avx512_compress(i, p, s, o) },
                short_rows_only: false,
            });
            v.push(Algo {
                name: "r5_avx512_rowwise",
                f: |i, p, s, o| unsafe { simd::r5_avx512_rowwise(i, p, s, o) },
                short_rows_only: true,
            });
            v.push(Algo {
                name: "r6_sb256",
                f: |i, p, s, o| unsafe { simd::r6_avx512_superblock::<8>(i, p, s, o) },
                short_rows_only: false,
            });
            v.push(Algo {
                name: "r6_sb512",
                f: |i, p, s, o| unsafe { simd::r6_avx512_superblock::<16>(i, p, s, o) },
                short_rows_only: false,
            });
        }
    }
    v
}

// ──────────────────────────────────────────────────────────────── measurement

/// Best-of-REPS wall time for one impl; verifies output against `expect`.
fn measure(
    algo: &Algo,
    inp: &Input,
    p: u16,
    s: &mut Scratch,
    expect: &[usize],
) -> Result<f64, String> {
    let mut out: Vec<usize> = Vec::with_capacity(inp.rows() + 64);
    // verify once
    (algo.f)(inp, p, s, &mut out);
    if out != expect {
        return Err(format!(
            "MISMATCH got {} rows, want {} rows",
            out.len(),
            expect.len()
        ));
    }
    let mut best = f64::INFINITY;
    for _ in 0..REPS {
        out.clear();
        let t = Instant::now();
        (algo.f)(black_box(inp), black_box(p), s, &mut out);
        let dt = t.elapsed().as_secs_f64();
        black_box(&out);
        best = best.min(dt);
    }
    Ok(best * 1e9 / inp.codes.len() as f64) // ns per code
}

// ─────────────────────────────────────────────────────────────────── datagen

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

const POINT: u16 = 0x2a2a;

fn gen(n: usize, avg_row: usize, density: f64, bursty: bool, seed: u64) -> Input {
    let mut rng = Rng(seed | 1);
    let mut codes: Vec<u16> = (0..n)
        .map(|_| {
            let c = (rng.next() >> 24) as u16;
            if c == POINT { c ^ 1 } else { c }
        })
        .collect();
    let mut off = vec![0u32];
    let mut cum = 0usize;
    while cum < n {
        let len = 1 + (rng.next() as usize) % (2 * avg_row - 1);
        cum = (cum + len).min(n);
        off.push(cum as u32);
    }
    let k = (n as f64 * density).round() as usize;
    if bursty {
        let mut placed = 0usize;
        while placed < k {
            let start = (rng.next() as usize) % n;
            let span = 512.min(n - start);
            let cnt = 32.min(k - placed);
            for _ in 0..cnt {
                codes[start + (rng.next() as usize) % span] = POINT;
            }
            placed += cnt;
        }
    } else {
        for _ in 0..k {
            codes[(rng.next() as usize) % n] = POINT;
        }
    }
    let hits = codes.iter().filter(|&&c| c == POINT).count();
    let sparse = (hits as f64) < (n as f64) * 1e-4;
    Input { codes, off, sparse }
}

// ─────────────────────────────────────────────────────────────────── reports

fn run_synth() {
    let n = 16_000_000usize;
    let densities = [1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 2e-1];
    let algos = registry();
    for &avg_row in &[4usize, 11, 32] {
        for &bursty in &[false, true] {
            println!(
                "\n== synth n={}M codes, avg_row={}, layout={} (ns/code, best of {}) ==",
                n / 1_000_000,
                avg_row,
                if bursty { "bursty" } else { "uniform" },
                REPS
            );
            print!("{:20}", "impl");
            for d in densities {
                print!("{:>10}", format!("{d:.0e}"));
            }
            println!();
            let inputs: Vec<Input> = densities
                .iter()
                .map(|&d| gen(n, avg_row, d, bursty, 0x5eed_0001 + avg_row as u64))
                .collect();
            let expects: Vec<Vec<usize>> = inputs
                .iter()
                .map(|inp| {
                    let mut e = Vec::new();
                    let mut s = Scratch { bitset: vec![], idx: vec![], rowbits: vec![], blockany: vec![] };
                    a1_row_early(inp, POINT, &mut s, &mut e);
                    e
                })
                .collect();
            let mut scratch: Vec<Scratch> = inputs.iter().map(Scratch::for_input).collect();
            for algo in &algos {
                if algo.short_rows_only && avg_row * 2 >= 32 {
                    println!("{:20}{}", algo.name, "       n/a".repeat(densities.len()));
                    continue;
                }
                print!("{:20}", algo.name);
                for (i, inp) in inputs.iter().enumerate() {
                    match measure(algo, inp, POINT, &mut scratch[i], &expects[i]) {
                        Ok(ns) => print!("{ns:>10.3}"),
                        Err(_) => print!("{:>10}", "FAIL"),
                    }
                }
                println!();
            }
        }
    }
}

fn run_real(codes_path: &str, off_path: &str, points_path: &str) {
    let codes_raw = std::fs::read(codes_path).unwrap();
    let codes: Vec<u16> = codes_raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let off_raw = std::fs::read(off_path).unwrap();
    let off: Vec<u32> = off_raw
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let max_row = off.windows(2).map(|w| w[1] - w[0]).max().unwrap();
    println!(
        "real column: {} codes, {} rows, max row len {} codes",
        codes.len(),
        off.len() - 1,
        max_row
    );
    let algos = registry();
    println!(
        "\n== real single-point queries (ns/code, best of {REPS}; rows = queries sorted by coverage) =="
    );
    print!("{:>10}{:>9}{:>9}", "coverage", "hits", "rows");
    for a in &algos {
        let short: String = a.name.chars().take(9).collect();
        print!("{short:>10}");
    }
    println!();
    let mut queries: Vec<(u16, f64, String)> = std::fs::read_to_string(points_path)
        .unwrap()
        .lines()
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            (f[0].parse().unwrap(), f[1].parse().unwrap(), f[2].to_string())
        })
        .collect();
    queries.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    for (point, frac, label) in queries {
        let inp = Input { codes: codes.clone(), off: off.clone(), sparse: frac < 1e-4 };
        let mut expect = Vec::new();
        let mut s0 = Scratch { bitset: vec![], idx: vec![], rowbits: vec![], blockany: vec![] };
        a1_row_early(&inp, point, &mut s0, &mut expect);
        let mut scratch = Scratch::for_input(&inp);
        print!("{frac:>10.6}{:>9}", inp.codes.iter().filter(|&&c| c == point).count());
        print!("{:>9}", expect.len());
        for algo in &algos {
            if algo.short_rows_only && max_row >= 32 {
                print!("{:>10}", "n/a");
                continue;
            }
            match measure(algo, &inp, point, &mut scratch, &expect) {
                Ok(ns) => print!("{ns:>10.3}"),
                Err(_) => print!("{:>10}", "FAIL"),
            }
        }
        println!("  {label}");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("real") => run_real(&args[2], &args[3], &args[4]),
        _ => run_synth(),
    }
}
