// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compressed-domain prefix / substring search.
//!
//! Rust port of the token-level search automata in the reference C++
//! implementation (`include/onpair/search/automata/*`). The central idea: a
//! column's bytes are encoded as a stream of dictionary token ids, so instead
//! of decompressing each row and running a byte matcher, we run a small
//! deterministic automaton **directly over the token ids**. Every input byte
//! becomes part of one token, so a `T`-token row costs `T` automaton steps
//! regardless of how many bytes it decodes to — and matches early-exit.
//!
//! Two predicates are supported, expressed as [`Pattern`]:
//!   * [`Pattern::Prefix`] — `col LIKE 'needle%'`, via [`prefix::PrefixAutomaton`].
//!   * [`Pattern::Contains`] — `col LIKE '%needle%'`, via [`kmp::KmpAutomaton`].
//!
//! Both automata are built once per query against the (sorted) dictionary and
//! then driven over every row. Construction relies on two dictionary
//! properties guaranteed by [`crate::Parser::train`]: the token ids are in
//! lexicographic order, and the 256 single-byte tokens are always present.

mod kmp;
mod prefix;
mod tokenize;

use crate::column::Column;
use crate::offset::Offset;
use crate::types::{MAX_TOKEN_SIZE, Token};

use kmp::{CHAIN_CONT, CHAIN_DEFINITE, CHAIN_OPEN, KmpAutomaton};
use prefix::PrefixAutomaton;

/// A search predicate evaluated against every row of a compressed column,
/// without decompressing it. Borrows the needle bytes for the duration of the
/// search.
#[derive(Copy, Clone, Debug)]
pub enum Pattern<'a> {
    /// Matches rows whose decoded bytes begin with the needle
    /// (SQL `col LIKE 'needle%'`).
    Prefix(&'a [u8]),
    /// Matches rows whose decoded bytes contain the needle anywhere
    /// (SQL `col LIKE '%needle%'`).
    Contains(&'a [u8]),
}

// ─────────────────────────────────────────────────────────────────────────────
// TokenRange — closed range of token ids [begin, last]; begin > last is empty.
// ─────────────────────────────────────────────────────────────────────────────

/// Closed range of token ids `[begin, last]`. The default-constructed
/// `{ begin: 1, last: 0 }` is the canonical empty range.
#[derive(Copy, Clone, Debug)]
pub(crate) struct TokenRange {
    pub(crate) begin: Token,
    pub(crate) last: Token,
}

impl TokenRange {
    /// Canonical empty range (`begin > last`).
    pub(crate) const EMPTY: Self = Self { begin: 1, last: 0 };

    #[inline]
    pub(crate) fn empty(self) -> bool {
        self.begin > self.last
    }

    #[inline]
    pub(crate) fn contains(self, t: Token) -> bool {
        t >= self.begin && t <= self.last
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DictView — borrowed, read-only view over a column's sorted dictionary.
// ─────────────────────────────────────────────────────────────────────────────

/// Borrowed view over the `(bytes, offsets)` of a sorted dictionary. Mirrors
/// the C++ `DictionaryView`: O(1) token access plus O(log n) prefix-range
/// lookups via binary search over the sorted token ids.
#[derive(Copy, Clone)]
pub(crate) struct DictView<'a> {
    bytes: &'a [u8],
    offsets: &'a [u32],
}

impl<'a> DictView<'a> {
    #[inline]
    fn num_tokens(self) -> usize {
        self.offsets.len() - 1
    }

    #[inline]
    fn token_size(self, id: Token) -> usize {
        (self.offsets[id as usize + 1] - self.offsets[id as usize]) as usize
    }

    #[inline]
    fn data(self, id: Token) -> &'a [u8] {
        let s = self.offsets[id as usize] as usize;
        let e = self.offsets[id as usize + 1] as usize;
        &self.bytes[s..e]
    }

    /// First token id in `[start, num_tokens)` whose bytes are `>= target`
    /// under the dictionary's sort order (shorter token sorts before a longer
    /// one sharing its prefix). Direct port of the C++ `lower_bound` lambda.
    fn lower_bound(self, target: &[u8], start: u32) -> u32 {
        let n = self.num_tokens() as u32;
        let (mut lo, mut hi) = (start, n);
        while lo < hi {
            let mid = lo + ((hi - lo) >> 1);
            let tok = self.data(mid as Token);
            let mlen = tok.len();
            let clen = mlen.min(target.len());
            let cmp = tok[..clen].cmp(&target[..clen]);
            // token[mid] < target iff cmp < 0, or equal-prefix and token shorter.
            if cmp.is_lt() || (cmp.is_eq() && mlen < target.len()) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// `[lo, hi]` token-id range whose byte sequences share `prefix`, or the
    /// empty range if none do. Port of `DictionaryView::prefix_range`.
    fn prefix_range(self, prefix: &[u8]) -> TokenRange {
        // A prefix longer than any token can never match.
        if prefix.len() > MAX_TOKEN_SIZE {
            return TokenRange::EMPTY;
        }
        let n = self.num_tokens() as u32;

        let lo = self.lower_bound(prefix, 0);

        // Next lexicographic prefix: increment the last non-0xFF byte after
        // trimming trailing 0xFF bytes. If all bytes are 0xFF the prefix has no
        // successor, so the range runs to the end of the dictionary.
        let mut buf = [0u8; MAX_TOKEN_SIZE];
        let mut ulen = prefix.len();
        let mut overflow = true;
        while ulen > 0 {
            if prefix[ulen - 1] < 0xFF {
                buf[..ulen].copy_from_slice(&prefix[..ulen]);
                buf[ulen - 1] += 1;
                overflow = false;
                break;
            }
            ulen -= 1;
        }

        // hi >= lo always, so the second search starts from lo, not 0.
        let hi = if overflow {
            n
        } else {
            self.lower_bound(&buf[..ulen], lo)
        };

        if lo < hi {
            TokenRange {
                begin: lo as Token,
                last: (hi - 1) as Token,
            }
        } else {
            TokenRange::EMPTY
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Row matcher.
// ─────────────────────────────────────────────────────────────────────────────

/// A compiled query that decides whether one row's token sequence matches.
///
/// Stateless across rows: all per-row state lives in [`matches`](Self::matches)
/// locals, so one matcher is built per query and reused for every row (no
/// reset between rows, and it can be shared by reference).
pub(crate) trait RowMatcher {
    /// Whether the row whose codes are `codes` matches.
    fn matches(&self, codes: &[Token]) -> bool;
}

/// Run `matcher` over every row delimited by `code_offsets`, invoking
/// `on_match` with the index of each matching row.
#[inline]
fn scan<O: Offset>(
    matcher: &impl RowMatcher,
    codes: &[Token],
    code_offsets: &[O],
    mut on_match: impl FnMut(usize),
) {
    for r in 0..code_offsets.len() - 1 {
        let s = code_offsets[r].as_usize();
        let e = code_offsets[r + 1].as_usize();
        if matcher.matches(&codes[s..e]) {
            on_match(r);
        }
    }
}

/// Whether the AVX2 pass-1 kernels should be used: the CPU supports AVX2 and
/// the `ONPAIR_NO_SIMD` benchmarking escape hatch is unset. Resolved once.
#[cfg(target_arch = "x86_64")]
fn avx2_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static STATE: AtomicU8 = AtomicU8::new(u8::MAX); // MAX = not yet resolved
    let cached = STATE.load(Ordering::Relaxed);
    if cached != u8::MAX {
        return cached == 1;
    }
    let on = std::is_x86_feature_detected!("avx2") && std::env::var_os("ONPAIR_NO_SIMD").is_none();
    STATE.store(on as u8, Ordering::Relaxed);
    on
}

/// Whether the AVX-512BW prefix kernel should be used: the CPU supports
/// AVX-512BW and SIMD is not disabled. Measured ~1.2× faster than the AVX2
/// prefix kernel (32 `u16`/vector + mask-register output, no pack/movemask).
/// `ONPAIR_NO_SIMD` disables it; `ONPAIR_NO_AVX512` forces the AVX2 path for A/B.
/// Resolved once.
#[cfg(target_arch = "x86_64")]
fn avx512_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static STATE: AtomicU8 = AtomicU8::new(u8::MAX);
    let cached = STATE.load(Ordering::Relaxed);
    if cached != u8::MAX {
        return cached == 1;
    }
    let on = std::is_x86_feature_detected!("avx512bw")
        && std::env::var_os("ONPAIR_NO_SIMD").is_none()
        && std::env::var_os("ONPAIR_NO_AVX512").is_none();
    STATE.store(on as u8, Ordering::Relaxed);
    on
}

/// Verdict of the Teddy-style 2-code chain row filter; see [`row_chain`].
enum RowChain {
    /// A token contains the whole needle — the row matches outright.
    Definite,
    /// A consecutive open→continue token pair exists — confirm with exact KMP.
    Candidate,
    /// Neither — the row cannot contain the needle.
    Reject,
}

/// Teddy-style 2-code chain filter over one row's codes, using the per-token
/// [`chain_table`](kmp::KmpAutomaton::chain_table). Carries the previous token's
/// `CHAIN_OPEN` bit and looks for a consecutive pair `(open, continue)` — far
/// more selective than "any opener present", since a boundary-spanning match
/// needs an opener token *immediately followed* by a continuation token.
///
/// Returns [`RowChain::Definite`] on the first DEFINITE token (the row matches
/// with no KMP needed), [`RowChain::Candidate`] if an open→continue pair occurs
/// (a spanning match is possible; the exact KMP confirms), else
/// [`RowChain::Reject`]. The early `return` on a definite token keeps LLVM from
/// the slower auto-vectorized gather (verified in the asm); the loop body is a
/// single scattered `chain[code]` load plus a register-carried `prev_open`.
#[inline]
fn row_chain(chain: &[u8], codes: &[Token]) -> RowChain {
    let mut prev_open = false;
    let mut candidate = false;
    for &c in codes {
        let f = chain[c as usize];
        if f & CHAIN_DEFINITE != 0 {
            return RowChain::Definite;
        }
        // Open token immediately followed by a continue token = possible
        // boundary-spanning match.
        candidate |= prev_open && (f & CHAIN_CONT != 0);
        prev_open = f & CHAIN_OPEN != 0;
    }
    if candidate {
        RowChain::Candidate
    } else {
        RowChain::Reject
    }
}

/// Pass-1 accept filter: set bit `r` of `acc` iff `first_codes[r]` lies in the
/// inclusive accept range `[alo, alo + awidth]` (unsigned). Branchless;
/// dispatches to AVX2 when available. Precondition for the SIMD path: the range
/// is non-empty (`alo <= u16::MAX`), which holds for every single-token query.
#[inline]
fn prefilter_accept(first_codes: &[u16], alo: u32, awidth: u32, acc: &mut [u64]) {
    #[cfg(target_arch = "x86_64")]
    if alo <= u16::MAX as u32 {
        if avx512_enabled() {
            // SAFETY: avx512bw just confirmed present.
            unsafe { prefilter_accept_avx512(first_codes, alo as u16, awidth as u16, acc) };
            return;
        }
        if avx2_enabled() {
            // SAFETY: avx2 just confirmed present.
            unsafe { prefilter_accept_avx2(first_codes, alo as u16, awidth as u16, acc) };
            return;
        }
    }
    prefilter_accept_scalar(first_codes, alo, awidth, acc);
}

/// Scalar fully-branchless accept filter: `(fc - alo) <= awidth` lowers to a
/// `sub` + unsigned compare with no branch, accumulated into one bitset word
/// per 64 rows.
#[inline]
fn prefilter_accept_scalar(first_codes: &[u16], alo: u32, awidth: u32, acc: &mut [u64]) {
    for (word, chunk) in acc.iter_mut().zip(first_codes.chunks(64)) {
        let mut w = 0u64;
        for (i, &fc) in chunk.iter().enumerate() {
            w |= u64::from((fc as u32).wrapping_sub(alo) <= awidth) << i;
        }
        *word = w;
    }
}

/// Pass-1 accept + verify filter: as [`prefilter_accept`], but also sets bit
/// `r` of `ver` iff `first_codes[r] == vpoint`. The two predicates are disjoint
/// (`vpoint < alo`), so no row lands in both. Branchless; dispatches to AVX2.
#[inline]
fn prefilter_accept_verify(
    first_codes: &[u16],
    alo: u32,
    awidth: u32,
    vpoint: u32,
    acc: &mut [u64],
    ver: &mut [u64],
) {
    #[cfg(target_arch = "x86_64")]
    if avx2_enabled() {
        // An empty accept range (alo > u16::MAX) is encoded by disabling the
        // accept compare; vpoint is always a real `u16` here (multi-token q0).
        let (alo16, awidth16, aenable) = if alo <= u16::MAX as u32 {
            (alo as u16, awidth as u16, 0xFFFFu16)
        } else {
            (0, 0, 0)
        };
        // SAFETY: avx2 just confirmed present.
        unsafe {
            prefilter_accept_verify_avx2(
                first_codes, alo16, awidth16, aenable, vpoint as u16, acc, ver,
            )
        };
        return;
    }
    prefilter_accept_verify_scalar(first_codes, alo, awidth, vpoint, acc, ver);
}

/// Scalar fully-branchless accept + verify filter.
#[inline]
fn prefilter_accept_verify_scalar(
    first_codes: &[u16],
    alo: u32,
    awidth: u32,
    vpoint: u32,
    acc: &mut [u64],
    ver: &mut [u64],
) {
    for ((accw, verw), chunk) in acc.iter_mut().zip(ver.iter_mut()).zip(first_codes.chunks(64)) {
        let mut a = 0u64;
        let mut v = 0u64;
        for (i, &fc) in chunk.iter().enumerate() {
            let fc = fc as u32;
            a |= u64::from(fc.wrapping_sub(alo) <= awidth) << i;
            v |= u64::from(fc == vpoint) << i;
        }
        *accw = a;
        *verw = v;
    }
}

/// Maximum INNER range count for which the SIMD multi-range contains pass-1 is
/// attempted; above this the per-code range chain outweighs the scalar gather.
const INNER_RANGE_BUDGET: usize = 16;

/// Set bit `i` of `bits` iff `codes[i]` lies in any of the (sorted, merged)
/// INNER `ranges`. Dispatches to AVX2 when available.
fn classify_inner(codes: &[Token], ranges: &[(Token, Token)], bits: &mut [u64]) {
    #[cfg(target_arch = "x86_64")]
    if avx2_enabled() {
        // SAFETY: avx2 confirmed present.
        unsafe { classify_inner_avx2(codes, ranges, bits) };
        return;
    }
    classify_inner_scalar(codes, ranges, bits);
}

/// Scalar reference for [`classify_inner`].
fn classify_inner_scalar(codes: &[Token], ranges: &[(Token, Token)], bits: &mut [u64]) {
    for (i, &c) in codes.iter().enumerate() {
        if ranges.iter().any(|&(lo, hi)| c >= lo && c <= hi) {
            bits[i >> 6] |= 1u64 << (i & 63);
        }
    }
}

/// Whether any bit in `bits[lo..hi]` (bit indices) is set.
#[inline]
fn any_bit_in_range(bits: &[u64], lo: usize, hi: usize) -> bool {
    if lo >= hi {
        return false;
    }
    let (wlo, whi) = (lo >> 6, (hi - 1) >> 6);
    if wlo == whi {
        let mask = (!0u64 << (lo & 63)) & (!0u64 >> (63 - ((hi - 1) & 63)));
        return bits[wlo] & mask != 0;
    }
    if bits[wlo] & (!0u64 << (lo & 63)) != 0 {
        return true;
    }
    if bits[wlo + 1..whi].iter().any(|&w| w != 0) {
        return true;
    }
    bits[whi] & (!0u64 >> (63 - ((hi - 1) & 63))) != 0
}

/// Invoke `f` with the index of every set bit in `words`, in ascending order.
#[inline]
fn for_each_set_bit(words: &[u64], mut f: impl FnMut(usize)) {
    for (w, &word) in words.iter().enumerate() {
        let mut bits = word;
        let base = w * 64;
        while bits != 0 {
            f(base + bits.trailing_zeros() as usize);
            bits &= bits - 1;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AVX2 pass-1 kernels. The range filter over the contiguous first-token table
// is a pure SIMD shape: one `sub` + unsigned compare per lane, 16 u16 rows per
// vector, packed straight into the candidate bitset words.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Reduce 16 `u16` lanes that are each `0xFFFF` (true) or `0x0000` (false) to a
/// 16-bit mask, bit `i` from lane `i`.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn movemask_epu16(v: __m256i) -> u32 {
    // Saturating pack i16->i8 maps 0xFFFF (-1) -> 0xFF and 0 -> 0, preserving
    // lane order across the two 128-bit halves, then one byte movemask.
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256::<1>(v);
    _mm_movemask_epi8(_mm_packs_epi16(lo, hi)) as u32
}

/// Lanewise `(fc - alo) <= awidth`, unsigned, as a `0xFFFF`/`0` mask vector.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn in_range_epu16(v: __m256i, valo: __m256i, vawidth: __m256i) -> __m256i {
    let sub = _mm256_sub_epi16(v, valo);
    // Unsigned `sub <= awidth` == `min_epu16(sub, awidth) == sub`.
    _mm256_cmpeq_epi16(_mm256_min_epu16(sub, vawidth), sub)
}

/// AVX2 accept filter; see [`prefilter_accept`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn prefilter_accept_avx2(first_codes: &[u16], alo: u16, awidth: u16, acc: &mut [u64]) {
    let valo = _mm256_set1_epi16(alo as i16);
    let vawidth = _mm256_set1_epi16(awidth as i16);
    let n = first_codes.len();
    let ptr = first_codes.as_ptr();
    let mut r = 0usize;
    let mut wi = 0usize;
    while r + 64 <= n {
        let mut word = 0u64;
        for k in 0..4 {
            // SAFETY: r + k*16 + 16 <= r + 64 <= n, in bounds; both helpers are
            // avx2, confirmed present.
            let v = unsafe { _mm256_loadu_si256(ptr.add(r + k * 16) as *const __m256i) };
            let m = unsafe { movemask_epu16(in_range_epu16(v, valo, vawidth)) };
            word |= (m as u64) << (k * 16);
        }
        acc[wi] = word;
        wi += 1;
        r += 64;
    }
    if r < n {
        prefilter_accept_scalar(&first_codes[r..], alo as u32, awidth as u32, &mut acc[wi..]);
    }
}

/// AVX-512BW accept filter (experiment #3): 32 `u16` codes per vector, one
/// `vpsubw` + `vpcmpuw` (`cmple_epu16`) yielding a `__mmask32` directly — no
/// pack/movemask reduction. Two masks compose one 64-bit bitset word.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512f")]
unsafe fn prefilter_accept_avx512(first_codes: &[u16], alo: u16, awidth: u16, acc: &mut [u64]) {
    let valo = _mm512_set1_epi16(alo as i16);
    let vawidth = _mm512_set1_epi16(awidth as i16);
    let n = first_codes.len();
    let ptr = first_codes.as_ptr();
    let mut r = 0usize;
    let mut wi = 0usize;
    while r + 64 <= n {
        // SAFETY: r + 32 and r + 64 are <= n.
        let v0 = unsafe { _mm512_loadu_si512(ptr.add(r) as *const __m512i) };
        let v1 = unsafe { _mm512_loadu_si512(ptr.add(r + 32) as *const __m512i) };
        // Unsigned (fc - alo) <= awidth, directly to a 32-bit mask register.
        let m0 = _mm512_cmple_epu16_mask(_mm512_sub_epi16(v0, valo), vawidth);
        let m1 = _mm512_cmple_epu16_mask(_mm512_sub_epi16(v1, valo), vawidth);
        acc[wi] = (m0 as u64) | ((m1 as u64) << 32);
        wi += 1;
        r += 64;
    }
    if r < n {
        prefilter_accept_scalar(&first_codes[r..], alo as u32, awidth as u32, &mut acc[wi..]);
    }
}

/// AVX2 multi-range INNER classifier; see [`classify_inner`]. For each 16-code
/// vector, OR together one `in_range_epu16` per INNER range, pack to a 16-bit
/// mask, and accumulate into the per-code bitset words (64 codes per word).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn classify_inner_avx2(codes: &[Token], ranges: &[(Token, Token)], bits: &mut [u64]) {
    // Preload (lo, width) vectors for each range.
    let zero = _mm256_setzero_si256();
    let mut vlo = [zero; INNER_RANGE_BUDGET];
    let mut vw = [zero; INNER_RANGE_BUDGET];
    for (i, &(lo, hi)) in ranges.iter().enumerate() {
        vlo[i] = _mm256_set1_epi16(lo as i16);
        vw[i] = _mm256_set1_epi16((hi - lo) as i16);
    }
    let nr = ranges.len();
    let n = codes.len();
    let ptr = codes.as_ptr();
    let (mut r, mut wi) = (0usize, 0usize);
    while r + 64 <= n {
        let mut word = 0u64;
        for k in 0..4 {
            // SAFETY: r + k*16 + 16 <= r + 64 <= n.
            let v = unsafe { _mm256_loadu_si256(ptr.add(r + k * 16) as *const __m256i) };
            let mut hit = zero;
            for vrange in vlo.iter().zip(vw.iter()).take(nr) {
                // SAFETY: both helpers are avx2, enabled for this fn.
                hit = _mm256_or_si256(hit, unsafe { in_range_epu16(v, *vrange.0, *vrange.1) });
            }
            // SAFETY: avx2 enabled for this fn.
            word |= (unsafe { movemask_epu16(hit) } as u64) << (k * 16);
        }
        bits[wi] = word;
        wi += 1;
        r += 64;
    }
    if r < n {
        classify_inner_scalar(&codes[r..], ranges, &mut bits[wi..]);
    }
}

/// AVX2 accept + verify filter; see [`prefilter_accept_verify`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn prefilter_accept_verify_avx2(
    first_codes: &[u16],
    alo: u16,
    awidth: u16,
    aenable: u16,
    vpoint: u16,
    acc: &mut [u64],
    ver: &mut [u64],
) {
    let valo = _mm256_set1_epi16(alo as i16);
    let vawidth = _mm256_set1_epi16(awidth as i16);
    let vaenable = _mm256_set1_epi16(aenable as i16);
    let vvpoint = _mm256_set1_epi16(vpoint as i16);
    let n = first_codes.len();
    let ptr = first_codes.as_ptr();
    let mut r = 0usize;
    let mut wi = 0usize;
    while r + 64 <= n {
        let mut accword = 0u64;
        let mut verword = 0u64;
        for k in 0..4 {
            // SAFETY: r + k*16 + 16 <= r + 64 <= n, in bounds; helpers are avx2.
            let v = unsafe { _mm256_loadu_si256(ptr.add(r + k * 16) as *const __m256i) };
            // Accept, masked off when the range is empty (aenable == 0).
            let accl = _mm256_and_si256(unsafe { in_range_epu16(v, valo, vawidth) }, vaenable);
            let verl = _mm256_cmpeq_epi16(v, vvpoint);
            accword |= (unsafe { movemask_epu16(accl) } as u64) << (k * 16);
            verword |= (unsafe { movemask_epu16(verl) } as u64) << (k * 16);
        }
        acc[wi] = accword;
        ver[wi] = verword;
        wi += 1;
        r += 64;
    }
    if r < n {
        // Reproduce the empty-range encoding for the scalar tail: alo = u32::MAX
        // makes `(fc - alo) <= 0` false for every real first code.
        let (talo, tawidth) = if aenable != 0 {
            (alo as u32, awidth as u32)
        } else {
            (u32::MAX, 0)
        };
        prefilter_accept_verify_scalar(
            &first_codes[r..],
            talo,
            tawidth,
            vpoint as u32,
            &mut acc[wi..],
            &mut ver[wi..],
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RowMask — packed result bitset.
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a [`search`](SearchParts::search): a packed bitmap over the
/// column's rows, one bit per row. Bit `i` is set iff row `i` matched.
///
/// The packed `u64` representation composes directly with a query engine's
/// own selection vectors (AND/OR of masks is word-wise), and is compact even
/// when most rows match.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RowMask {
    words: Vec<u64>,
    rows: usize,
}

impl RowMask {
    /// All-zero mask sized for `rows` rows.
    fn zeros(rows: usize) -> Self {
        Self {
            words: vec![0; rows.div_ceil(64)],
            rows,
        }
    }

    #[inline]
    fn set(&mut self, i: usize) {
        self.words[i >> 6] |= 1u64 << (i & 63);
    }

    /// Number of rows the mask covers (set or not). The bitmap has
    /// `len().div_ceil(64)` words; bits at indices `>= len()` in the final word
    /// are zero.
    #[inline]
    pub fn len(&self) -> usize {
        self.rows
    }

    /// Whether the mask covers zero rows.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    /// Borrow the packed bitmap words (bit `i` = row `i`, LSB-first within each
    /// word). Compose directly with a query engine's own selection vectors via
    /// word-wise AND/OR. Length is `len().div_ceil(64)`.
    #[inline]
    pub fn as_words(&self) -> &[u64] {
        &self.words
    }

    /// Consume the mask into its owned `(words, len)` parts: the packed bitmap
    /// and the row count it covers. Inverse shape of the borrowed
    /// [`as_words`](Self::as_words) + [`len`](Self::len).
    #[inline]
    pub fn into_parts(self) -> (Vec<u64>, usize) {
        (self.words, self.rows)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SearchParts — borrowed view of the data search needs.
// ─────────────────────────────────────────────────────────────────────────────

/// Borrowed view of everything compressed-domain search needs: the sorted
/// dictionary plus the per-row code stream. Mirrors [`crate::Parts`] (the
/// decode view) but additionally carries `code_offsets`, the row delimiters a
/// row-wise scan requires.
///
/// Build one cheaply from an owned column with
/// [`Column::as_search_parts`], or by struct literal from data
/// deserialized out of storage.
#[derive(Copy, Clone, Debug)]
pub struct SearchParts<'a, O: Offset> {
    /// Dictionary bytes (sorted token order). Mirrors [`Column::dict_bytes`].
    pub dict_bytes: &'a [u8],
    /// Token byte ranges into `dict_bytes`. Mirrors [`Column::dict_offsets`].
    pub dict_offsets: &'a [u32],
    /// Encoded tokens, row-concatenated. Mirrors [`Column::codes`].
    pub codes: &'a [u16],
    /// `R + 1` offsets into `codes` delimiting the `R` rows: row `r`'s codes
    /// are `codes[code_offsets[r]..code_offsets[r + 1]]`. Mirrors
    /// [`Column::code_offsets`].
    pub code_offsets: &'a [O],
    /// Optional per-row first token id (`R` entries); mirrors
    /// [`Column::first_codes`]. When present, it is used as a contiguous
    /// prefilter for [`Pattern::Prefix`] searches; when `None`, prefix search
    /// falls back to the generic per-row scan.
    pub first_codes: Option<&'a [u16]>,
}

impl<O: Offset> SearchParts<'_, O> {
    #[inline]
    fn dict(&self) -> DictView<'_> {
        DictView {
            bytes: self.dict_bytes,
            offsets: self.dict_offsets,
        }
    }

    /// Number of rows in the view.
    #[inline]
    fn num_rows(&self) -> usize {
        self.code_offsets.len().saturating_sub(1)
    }

    /// Codes of row `r`: `codes[code_offsets[r]..code_offsets[r + 1]]`. Offsets
    /// are validated at construction (monotonic, in bounds), so the conversion
    /// is the branchless [`Offset::as_usize`].
    #[inline]
    fn row_codes(&self, r: usize) -> &[Token] {
        let s = self.code_offsets[r].as_usize();
        let e = self.code_offsets[r + 1].as_usize();
        &self.codes[s..e]
    }

    /// Evaluate `pattern` against every row, invoking `on_match` with the
    /// 0-based index of each matching row, in order. The low-level primitive
    /// [`search`](Self::search) builds its [`RowMask`] on top of.
    pub fn search_callback(&self, pattern: Pattern<'_>, on_match: impl FnMut(usize)) {
        let dict = self.dict();
        match pattern {
            Pattern::Contains(needle) => {
                let aut = KmpAutomaton::new(needle, dict);
                self.scan_contains(&aut, on_match);
            }
            Pattern::Prefix(needle) => {
                let aut = PrefixAutomaton::new(needle, dict);
                self.scan_prefix(&aut, dict.num_tokens(), on_match);
            }
        }
    }

    /// Contains scan in two passes over the whole code stream.
    ///
    /// Unlike prefix (which need only inspect each row's first token), a
    /// substring can begin at any token, so pass 1 must stream every code. Using
    /// the KMP [`class_table`](KmpAutomaton::class_table), each row is reduced to
    /// one of three verdicts by OR-ing its tokens' classes:
    ///   * a [`CLASS_DEFINITE`] token present → the row matches outright (a token
    ///     contains the whole needle); emit without a row check;
    ///   * else a [`CLASS_OPENER`] token present → the row is a candidate; the
    ///     exact KMP confirms it in pass 2;
    ///   * else (all classes zero) → reject, never touching the KMP.
    ///
    /// The dependent-load + branch chain of the KMP fast path is thus paid only
    /// on candidate rows, not on the (dominant at low/medium selectivity)
    /// reject majority. Falls back to the generic per-row scan only for the empty
    /// needle (which matches every row).
    fn scan_contains(&self, aut: &KmpAutomaton, mut on_match: impl FnMut(usize)) {
        let n = self.code_offsets.len() - 1;
        if aut.is_empty_needle() {
            scan(aut, self.codes, self.code_offsets, on_match);
            return;
        }

        // Optional SIMD INNER pass-1: when the INNER token set collapses into a
        // small number of contiguous id ranges, classify the whole code stream
        // with AVX2 range tests (any-INNER per code), reduce to candidate rows,
        // and confirm with the exact KMP. Sound (every match holds an INNER
        // token). Gated behind a small range budget — above it the per-code
        // range chain is longer than the scalar gather it replaces.
        if std::env::var_os("ONPAIR_INNER_SIMD").is_some()
            && let Some(ranges) = aut.inner_ranges(INNER_RANGE_BUDGET)
        {
            if ranges.is_empty() {
                return; // no INNER token ⇒ no match possible.
            }
            self.scan_contains_inner(aut, &ranges, on_match);
            return;
        }

        // Optional 3-layer funnel: a cheap SIMD INNER reject (layer 1) over the
        // whole stream, then the precise scalar adjacency chain (layer 2) only on
        // layer-1 survivors, then exact KMP (layer 3) on chain candidates. Both
        // INNER-presence and the open→cont chain are independently necessary for
        // a match, so ANDing them drops no true match. The point: replace
        // row_chain over ALL codes with classify_inner over all codes (SIMD) +
        // row_chain over only the survivors (13–38%).
        if std::env::var_os("ONPAIR_FUNNEL").is_some()
            && let Some(ranges) = aut.inner_ranges(INNER_RANGE_BUDGET)
        {
            if ranges.is_empty() {
                return;
            }
            self.scan_contains_funnel(aut, &ranges, on_match);
            return;
        }

        let chain = aut.chain_table();
        for r in 0..n {
            let codes = self.row_codes(r);
            match row_chain(&chain, codes) {
                // A DEFINITE token: the row matches outright.
                RowChain::Definite => on_match(r),
                // An open→continue pair exists: a boundary-spanning match is
                // possible; confirm with the exact KMP.
                RowChain::Candidate => {
                    if aut.matches(codes) {
                        on_match(r);
                    }
                }
                // Neither: the row cannot contain the needle.
                RowChain::Reject => {}
            }
        }
    }

    /// SIMD INNER contains scan. Pass 1 marks each code that lies in any INNER
    /// range (AVX2 multi-range test over the whole stream) into a per-code
    /// bitset; pass 2 visits rows holding a marked code and confirms with the
    /// exact KMP. INNER presence is a sound necessary condition for a match.
    fn scan_contains_inner(
        &self,
        aut: &KmpAutomaton,
        ranges: &[(Token, Token)],
        mut on_match: impl FnMut(usize),
    ) {
        let m = self.codes.len();
        let words = m.div_ceil(64);
        let mut inner_bits = vec![0u64; words];
        classify_inner(self.codes, ranges, &mut inner_bits);
        for r in 0..self.code_offsets.len() - 1 {
            let s = self.code_offsets[r].as_usize();
            let e = self.code_offsets[r + 1].as_usize();
            if any_bit_in_range(&inner_bits, s, e) && aut.matches(&self.codes[s..e]) {
                on_match(r);
            }
        }
    }

    /// 3-layer funnel contains scan. Layer 1: SIMD INNER classify over the whole
    /// code stream into a per-code bitset (cheap, but only ~13–38% selective).
    /// Layer 2: for rows surviving layer 1, the precise scalar adjacency chain
    /// (`row_chain`) — run only on survivors, not all rows. Layer 3: exact KMP on
    /// chain candidates. Both INNER-presence and the open→cont chain are
    /// necessary for a match, so ANDing the layers drops no true match.
    fn scan_contains_funnel(
        &self,
        aut: &KmpAutomaton,
        ranges: &[(Token, Token)],
        mut on_match: impl FnMut(usize),
    ) {
        let words = self.codes.len().div_ceil(64);
        let mut inner_bits = vec![0u64; words];
        classify_inner(self.codes, ranges, &mut inner_bits);
        let chain = aut.chain_table();
        for r in 0..self.code_offsets.len() - 1 {
            let s = self.code_offsets[r].as_usize();
            let e = self.code_offsets[r + 1].as_usize();
            // Layer 1: SIMD INNER reject — skip the scalar chain entirely if no
            // INNER code is present.
            if !any_bit_in_range(&inner_bits, s, e) {
                continue;
            }
            let codes = &self.codes[s..e];
            // Layer 2+3: precise chain, then exact KMP.
            match row_chain(&chain, codes) {
                RowChain::Definite => on_match(r),
                RowChain::Candidate => {
                    if aut.matches(codes) {
                        on_match(r);
                    }
                }
                RowChain::Reject => {}
            }
        }
    }

    /// Prefix scan in two passes over the contiguous first-token table.
    ///
    /// Pass 1 is a fully branchless range filter: a row is a candidate iff its
    /// first token lies in the sound superset range `[lo, hi]` returned by
    /// [`PrefixAutomaton::prefilter_range`]. It touches one code per row (the
    /// linear `first_codes`, never the scattered code stream), so it is cheap
    /// even at low selectivity, and is the part that vectorises.
    ///
    /// Pass 2 only visits candidates. For a single-token query the range is
    /// exact, so candidates are emitted directly; otherwise each is confirmed
    /// with a full row check — the only place the scattered codes are read.
    ///
    /// Falls back to the generic per-row scan for the empty query, or when the
    /// dictionary is fully saturated (`num_tokens == 65536`) and the empty-row
    /// sentinel `u16::MAX` could collide with a real token id.
    fn scan_prefix(
        &self,
        aut: &PrefixAutomaton,
        num_tokens: usize,
        mut on_match: impl FnMut(usize),
    ) {
        let n = self.code_offsets.len() - 1;
        // Use the prefilter only with a same-length first-token table and an
        // unsaturated dictionary (so the u16::MAX empty-row sentinel cannot
        // collide with a real id); otherwise scan generically.
        let first_codes = match self.first_codes {
            Some(fc) if fc.len() == n && num_tokens <= u16::MAX as usize => fc,
            _ => {
                scan(aut, self.codes, self.code_offsets, on_match);
                return;
            }
        };
        if aut.is_empty_query() {
            scan(aut, self.codes, self.code_offsets, on_match);
            return;
        }
        let pf = aut.prefilter();
        let words = n.div_ceil(64);

        if !pf.needs_verify() {
            // Single-token query: the accept range is exact. One branchless
            // pass, emit directly — no row ever touches the scattered codes.
            let mut acc = vec![0u64; words];
            prefilter_accept(first_codes, pf.alo, pf.awidth, &mut acc);
            for_each_set_bit(&acc, on_match);
            return;
        }

        // Multi-token query. Pass 1 splits rows into definite accepts (first
        // token begins with the whole needle) and verify candidates (first
        // token equals the query head). Both predicates are branchless.
        let mut acc = vec![0u64; words];
        let mut ver = vec![0u64; words];
        prefilter_accept_verify(first_codes, pf.alo, pf.awidth, pf.vpoint, &mut acc, &mut ver);

        // Definite accepts: emit directly.
        for_each_set_bit(&acc, &mut on_match);
        // Pass 2: confirm only the (usually few) verify candidates — the one
        // place the scattered code stream is read.
        for_each_set_bit(&ver, |r| {
            if aut.matches(self.row_codes(r)) {
                on_match(r);
            }
        });
    }

    /// Prefix scan that writes its result directly as a [`RowMask`] bitset,
    /// skipping the per-row callback. Pass 1's accept predicate already produces
    /// the matching-rows bitmap, so it is written straight into the mask words
    /// (a contiguous SIMD store) instead of being walked bit-by-bit; only the
    /// verify candidates are confirmed and OR'd in individually. This is the
    /// fast path behind [`search`](Self::search) for prefix queries — at high
    /// selectivity it avoids emitting hundreds of thousands of bits one call at
    /// a time.
    ///
    /// Returns `None` when the first-token prefilter is not applicable (empty
    /// query, missing/short index, or saturated dictionary), so the caller can
    /// fall back to the generic callback scan.
    fn prefix_mask(&self, aut: &PrefixAutomaton, num_tokens: usize) -> Option<RowMask> {
        let n = self.code_offsets.len() - 1;
        let first_codes = match self.first_codes {
            Some(fc) if fc.len() == n && num_tokens <= u16::MAX as usize => fc,
            _ => return None,
        };
        if aut.is_empty_query() {
            return None;
        }
        let pf = aut.prefilter();
        let words = n.div_ceil(64);
        let mut acc = vec![0u64; words];

        if pf.needs_verify() {
            // Multi-token: accepts go straight into `acc`; verify candidates are
            // confirmed and OR'd in (they are disjoint from the accept range).
            let mut ver = vec![0u64; words];
            prefilter_accept_verify(first_codes, pf.alo, pf.awidth, pf.vpoint, &mut acc, &mut ver);
            for_each_set_bit(&ver, |r| {
                if aut.matches(self.row_codes(r)) {
                    acc[r >> 6] |= 1u64 << (r & 63);
                }
            });
        } else {
            // Single-token: the accept range is exact — pass 1 is the answer.
            prefilter_accept(first_codes, pf.alo, pf.awidth, &mut acc);
        }
        Some(RowMask {
            words: acc,
            rows: n,
        })
    }

    /// Evaluate `pattern` against every row, returning a [`RowMask`] whose set
    /// bits are the matching row indices. The match is computed in the
    /// compressed domain — rows are never decompressed.
    pub fn search(&self, pattern: Pattern<'_>) -> RowMask {
        // Prefix queries take the bitmap-merge fast path: the prefilter writes
        // the result bits straight into the mask instead of via a per-row
        // callback. Falls through to the generic callback build otherwise.
        if let Pattern::Prefix(needle) = pattern {
            let dict = self.dict();
            let aut = PrefixAutomaton::new(needle, dict);
            if let Some(mask) = self.prefix_mask(&aut, dict.num_tokens()) {
                return mask;
            }
        }
        let mut mask = RowMask::zeros(self.num_rows());
        self.search_callback(pattern, |r| mask.set(r));
        mask
    }
}

impl<O: Offset> Column<O> {
    /// Zero-copy [`SearchParts`] view over this column, for
    /// [`SearchParts::search`]. Parallels [`as_parts`](Column::as_parts), but
    /// includes `code_offsets` (the row delimiters search needs).
    #[inline]
    pub fn as_search_parts(&self) -> SearchParts<'_, O> {
        SearchParts {
            dict_bytes: &self.dict_bytes,
            dict_offsets: &self.dict_offsets,
            codes: &self.codes,
            code_offsets: &self.code_offsets,
            first_codes: self.first_codes.as_deref(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Bits, Config, Threshold, compress};

    /// EXPERIMENT #1 (LPM-aware reachability soundness). For each needle, tries
    /// hard to CONSTRUCT a real byte string whose LPM tokenisation (using the
    /// trained dictionary) lands a token boundary at each partial DFA state — in
    /// particular the "unreachable" deep states. If any partial state is
    /// witnessed reachable by a constructed/random string, dropping its
    /// completion range from the prefilter would be UNSOUND. Proves whether the
    /// empirical "0× in corpus" is a real dictionary-level impossibility.
    /// ONPAIR_NEEDLE=google ONPAIR_CORPUS=/tmp/cppdump/corpus.bin \
    ///   cargo test --lib lpm_reach_witness -- --ignored --nocapture
    #[test]
    #[ignore]
    #[allow(clippy::use_debug)]
    fn lpm_reach_witness() {
        use crate::Parser;
        let needle = std::env::var("ONPAIR_NEEDLE").unwrap_or_else(|_| "google".into());
        let p = needle.as_bytes();
        let m = p.len();
        // Train on the real corpus, then re-tokenise arbitrary probe strings
        // through the SAME dictionary via Parser::parse.
        let col0 = load_corpus_col();
        let parts0 = col0.as_search_parts();
        // Reconstruct training bytes is unnecessary: re-train a Parser on the
        // corpus rows so we can parse() probe strings.
        let raw = std::fs::read(std::env::var("ONPAIR_CORPUS").unwrap()).unwrap();
        let n = u64::from_le_bytes(raw[0..8].try_into().unwrap()) as usize;
        let mut o = 8;
        let mut lens = Vec::with_capacity(n);
        for _ in 0..n {
            lens.push(u32::from_le_bytes(raw[o..o + 4].try_into().unwrap()) as usize);
            o += 4;
        }
        let mut cbytes = Vec::new();
        let mut coffs = vec![0u32];
        for &l in &lens {
            cbytes.extend_from_slice(&raw[o..o + l]);
            o += l;
            coffs.push(cbytes.len() as u32);
        }
        let parser = Parser::train(
            &cbytes,
            &coffs,
            Config { bits: Bits::new(16).unwrap(), threshold: Threshold::new(0.5).unwrap(), seed: Some(42) },
        )
        .unwrap();

        let dict = DictView { bytes: parts0.dict_bytes, offsets: parts0.dict_offsets };
        let aut = KmpAutomaton::new(p, dict);

        // Run a probe string through LPM tokenisation and return the set of
        // boundary states it reaches (excluding 0).
        let probe = |s: &[u8], reached: &mut [bool], witness: &mut Vec<Option<Vec<u8>>>| {
            let pcol = parser.parse(s, &[0u32, s.len() as u32]).unwrap();
            let pp = pcol.as_search_parts();
            let mut st = 0u8;
            for &c in pp.codes {
                st = aut.step_from(st, c);
                if !reached[st as usize] {
                    reached[st as usize] = true;
                    witness[st as usize] = Some(s.to_vec());
                }
                if st as usize == m {
                    break;
                }
            }
        };

        let mut reached = vec![false; m + 1];
        let mut witness: Vec<Option<Vec<u8>>> = vec![None; m + 1];
        // 1. Crafted witnesses: for each partial state s, the s-byte prefix of
        //    the needle, sandwiched between filler designed to force boundaries.
        let fillers: &[&[u8]] = &[b"", b" ", b"/", b"x", b"zz", b"://", b".", b"=", b"?", b"&"];
        for s in 1..m {
            for fa in fillers {
                for fb in fillers {
                    let mut v = Vec::new();
                    v.extend_from_slice(fa);
                    v.extend_from_slice(&p[..s]);
                    v.extend_from_slice(fb);
                    probe(&v, &mut reached, &mut witness);
                    // also doubled-prefix tricks: googgl-style to force a split
                    let mut v2 = Vec::new();
                    v2.extend_from_slice(&p[..s]);
                    v2.extend_from_slice(&p[..s]);
                    v2.extend_from_slice(fb);
                    probe(&v2, &mut reached, &mut witness);
                }
            }
        }
        // 2. Random fuzz around needle bytes.
        let mut x = 0x2545F4914F6CDD1Du64;
        let alpha = {
            let mut a: Vec<u8> = p.to_vec();
            a.extend_from_slice(b" /.:=?&-_0123456789abcdefghijklmnopqrstuvwxyz");
            a.sort_unstable();
            a.dedup();
            a
        };
        for _ in 0..2_000_000u64 {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            let len = 2 + (x as usize % 14);
            let mut v = Vec::with_capacity(len);
            let mut y = x;
            for _ in 0..len {
                y = y.wrapping_mul(6364136223846793005).wrapping_add(1);
                v.push(alpha[(y >> 33) as usize % alpha.len()]);
            }
            probe(&v, &mut reached, &mut witness);
        }

        eprintln!("=== LPM reachability witnesses for {needle:?} (m={m}) ===");
        for s in 1..m {
            let w = witness[s]
                .as_ref()
                .map(|v| String::from_utf8_lossy(v).into_owned())
                .unwrap_or_default();
            eprintln!(
                "  state {s} (prefix {:?}): {}  witness={w:?}",
                std::str::from_utf8(&p[..s]).unwrap_or("?"),
                if reached[s] { "REACHABLE — prune UNSOUND" } else { "no witness found" }
            );
        }
    }

    /// Load the dumped corpus, compress it, and return the owned column.
    #[cfg(test)]
    fn load_corpus_col() -> Column<u32> {
        let raw = std::fs::read(std::env::var("ONPAIR_CORPUS").unwrap()).unwrap();
        let n = u64::from_le_bytes(raw[0..8].try_into().unwrap()) as usize;
        let mut o = 8;
        let mut lens = Vec::with_capacity(n);
        for _ in 0..n {
            lens.push(u32::from_le_bytes(raw[o..o + 4].try_into().unwrap()) as usize);
            o += 4;
        }
        let mut bytes = Vec::new();
        let mut offs = vec![0u32];
        for &l in &lens {
            bytes.extend_from_slice(&raw[o..o + l]);
            o += l;
            offs.push(bytes.len() as u32);
        }
        compress(
            &bytes,
            &offs,
            Config { bits: Bits::new(16).unwrap(), threshold: Threshold::new(0.5).unwrap(), seed: Some(42) },
        )
        .unwrap()
    }

    /// Temporary: how many tokens land in each DFA state via `base[]` — i.e.
    /// which partial-match states are reachable at a token boundary at all.
    /// ONPAIR_NEEDLE=google ONPAIR_CORPUS=/tmp/cppdump/corpus.bin \
    ///   cargo test --lib boundary_states -- --ignored --nocapture
    #[test]
    #[ignore]
    #[allow(clippy::use_debug)]
    fn boundary_states() {
        let needle = std::env::var("ONPAIR_NEEDLE").unwrap_or_else(|_| "google".into());
        let col = load_corpus_col();
        let parts = col.as_search_parts();
        let dict = DictView { bytes: parts.dict_bytes, offsets: parts.dict_offsets };
        let aut = KmpAutomaton::new(needle.as_bytes(), dict);
        let counts = aut.boundary_state_counts();
        eprintln!("=== boundary-reachable states for {needle:?} ===");
        for (s, &c) in counts.iter().enumerate() {
            let what = if s == 0 { "inert" } else if s == needle.len() { "MATCH (definite)" } else { "partial" };
            eprintln!("  state {s} ({what}): {c} tokens end here (base==s)");
        }
    }

    /// Temporary: across ALL rows, which DFA boundary states actually occur?
    /// Re-runs the token automaton over every row recording the set of states
    /// seen at token boundaries — to test whether LPM makes deep partial states
    /// unreachable in practice (so their continuation ranges can be pruned).
    /// ONPAIR_NEEDLE=google ONPAIR_CORPUS=/tmp/cppdump/corpus.bin \
    ///   cargo test --lib reached_states -- --ignored --nocapture
    #[test]
    #[ignore]
    #[allow(clippy::use_debug)]
    fn reached_states() {
        let needle = std::env::var("ONPAIR_NEEDLE").unwrap_or_else(|_| "google".into());
        let col = load_corpus_col();
        let parts = col.as_search_parts();
        let dict = DictView { bytes: parts.dict_bytes, offsets: parts.dict_offsets };
        let aut = KmpAutomaton::new(needle.as_bytes(), dict);
        let m = needle.len();
        // Per-state count of how often a boundary lands there (across all rows).
        let mut seen = vec![0u64; m + 1];
        let codes = parts.codes;
        let co = parts.code_offsets;
        for r in 0..co.len() - 1 {
            let (s0, e0) = (co[r] as usize, co[r + 1] as usize);
            let mut st = 0u8;
            for &c in &codes[s0..e0] {
                st = aut.step_from(st, c);
                seen[st as usize] += 1;
                if st as usize == m {
                    break;
                }
            }
        }
        eprintln!("=== boundary states actually REACHED across all rows, {needle:?} ===");
        for (s, &c) in seen.iter().enumerate() {
            eprintln!("  state {s}: reached {c} times");
        }
    }

    /// Temporary: dump EXACTLY what the SIMD INNER prefilter range-tests for a
    /// needle — the merged INNER id ranges (each one AVX2 `in_range_epu16` test)
    /// with their token byte content.
    /// ONPAIR_NEEDLE=google ONPAIR_CORPUS=/tmp/cppdump/corpus.bin \
    ///   cargo test --lib inner_ranges_dump -- --ignored --nocapture
    #[test]
    #[ignore]
    #[allow(clippy::use_debug)]
    fn inner_ranges_dump() {
        let needle = std::env::var("ONPAIR_NEEDLE").unwrap_or_else(|_| "google".into());
        let raw = std::fs::read(std::env::var("ONPAIR_CORPUS").unwrap()).unwrap();
        let n = u64::from_le_bytes(raw[0..8].try_into().unwrap()) as usize;
        let mut o = 8;
        let mut lens = Vec::with_capacity(n);
        for _ in 0..n {
            lens.push(u32::from_le_bytes(raw[o..o + 4].try_into().unwrap()) as usize);
            o += 4;
        }
        let mut bytes = Vec::new();
        let mut offs = vec![0u32];
        for &l in &lens {
            bytes.extend_from_slice(&raw[o..o + l]);
            o += l;
            offs.push(bytes.len() as u32);
        }
        let col = compress(
            &bytes,
            &offs,
            Config { bits: Bits::new(16).unwrap(), threshold: Threshold::new(0.5).unwrap(), seed: Some(42) },
        )
        .unwrap();
        let parts = col.as_search_parts();
        let dict = DictView { bytes: parts.dict_bytes, offsets: parts.dict_offsets };
        let aut = KmpAutomaton::new(needle.as_bytes(), dict);
        let ranges = aut.inner_ranges(64).expect("within budget");
        let tok = |id: u16| String::from_utf8_lossy(dict.data(id)).into_owned();
        eprintln!("=== SIMD prefilter for {needle:?}: {} range tests ===", ranges.len());
        let mut total = 0usize;
        for (lo, hi) in &ranges {
            let cnt = (hi - lo + 1) as usize;
            total += cnt;
            eprintln!(
                "  ids {lo}..={hi} ({cnt} tok): {:?} .. {:?}",
                tok(*lo),
                tok(*hi)
            );
        }
        eprintln!("a code is a candidate iff it falls in ANY of those {} ranges ({total} token ids)", ranges.len());
    }

    /// Temporary: dump the TOKEN-LEVEL DFA for a needle over the real dict
    /// (alphabet = token ids, not bytes).
    /// ONPAIR_NEEDLE=google ONPAIR_CORPUS=/tmp/cppdump/corpus.bin \
    ///   cargo test --lib token_dfa -- --ignored --nocapture
    #[test]
    #[ignore]
    #[allow(clippy::use_debug)]
    fn token_dfa() {
        let needle = std::env::var("ONPAIR_NEEDLE").unwrap_or_else(|_| "google".into());
        let raw = std::fs::read(std::env::var("ONPAIR_CORPUS").unwrap()).unwrap();
        let n = u64::from_le_bytes(raw[0..8].try_into().unwrap()) as usize;
        let mut o = 8;
        let mut lens = Vec::with_capacity(n);
        for _ in 0..n {
            lens.push(u32::from_le_bytes(raw[o..o + 4].try_into().unwrap()) as usize);
            o += 4;
        }
        let mut bytes = Vec::new();
        let mut offs = vec![0u32];
        for &l in &lens {
            bytes.extend_from_slice(&raw[o..o + l]);
            o += l;
            offs.push(bytes.len() as u32);
        }
        let col = compress(
            &bytes,
            &offs,
            Config { bits: Bits::new(16).unwrap(), threshold: Threshold::new(0.5).unwrap(), seed: Some(42) },
        )
        .unwrap();
        let parts = col.as_search_parts();
        let dict = DictView { bytes: parts.dict_bytes, offsets: parts.dict_offsets };
        let nt = dict.num_tokens();
        let aut = KmpAutomaton::new(needle.as_bytes(), dict);
        let (base_runs, per_state) = aut.dump_dfa();
        let m = needle.len();
        let tokstr = |id: u16| String::from_utf8_lossy(dict.data(id)).into_owned();

        eprintln!("=== TOKEN-LEVEL DFA for {needle:?}  ({nt} tokens = the alphabet, {m}+1 states) ===\n");
        eprintln!("STATE 0 (no partial match) — base[] table, run-length encoded:");
        eprintln!("  {} non-zero runs out of {} total runs:", base_runs.iter().filter(|r| r.2 != 0).count(), base_runs.len());
        for &(lo, hi, t) in base_runs.iter().filter(|r| r.2 != 0) {
            let lbl = if lo == hi { format!("token {lo} {:?}", tokstr(lo as u16)) }
                      else { format!("tokens {lo}..={hi} (e.g. {:?})", tokstr(lo as u16)) };
            eprintln!("    →state {t}: {lbl}");
        }
        for (s, trs) in per_state.iter().enumerate() {
            let s = s + 1;
            if s >= m { continue; }
            eprintln!("\nSTATE {s} (matched {} needle bytes) — {} sparse exceptions over base:", s, trs.len());
            for &(lo, hi, t) in trs.iter().take(12) {
                let lbl = if lo == hi { format!("token {lo} {:?}", tokstr(lo)) }
                          else { format!("tokens {lo}..={hi}") };
                eprintln!("    on {lbl} → state {t}");
            }
            if trs.len() > 12 { eprintln!("    … {} more", trs.len() - 12); }
        }
        let total_sparse: usize = per_state.iter().map(|v| v.len()).sum();
        let nz_base: u32 = base_runs.iter().filter(|r| r.2 != 0).map(|&(lo, hi, _)| hi - lo + 1).sum();
        eprintln!("\nSUMMARY: state-0 alphabet that matters = {nz_base} token ids in {} runs;",
            base_runs.iter().filter(|r| r.2 != 0).count());
        eprintln!("         {total_sparse} sparse exception ranges across the partial-match states.");
    }

    /// Temporary: measure the selectivity of the SIMD-able INNER filter — a row
    /// is a candidate iff it has a DEFINITE token or an INNER token (one covered
    /// by a sparse continuation range, which are contiguous id ranges). This is
    /// a sound necessary filter (the token completing any match is DEFINITE or
    /// INNER), and unlike the open-set it IS range-testable with SIMD. Compares
    /// its candidate rate to the current adjacency chain.
    /// ONPAIR_NEEDLE=google ONPAIR_CORPUS=/tmp/cppdump/corpus.bin \
    ///   cargo test --lib inner_probe -- --ignored --nocapture
    #[test]
    #[ignore]
    #[allow(clippy::use_debug)]
    fn inner_probe() {
        let needle = std::env::var("ONPAIR_NEEDLE").unwrap_or_else(|_| "google".into());
        let raw = std::fs::read(std::env::var("ONPAIR_CORPUS").unwrap()).unwrap();
        let n = u64::from_le_bytes(raw[0..8].try_into().unwrap()) as usize;
        let mut o = 8;
        let mut lens = Vec::with_capacity(n);
        for _ in 0..n {
            lens.push(u32::from_le_bytes(raw[o..o + 4].try_into().unwrap()) as usize);
            o += 4;
        }
        let mut bytes = Vec::new();
        let mut offs = vec![0u32];
        for &l in &lens {
            bytes.extend_from_slice(&raw[o..o + l]);
            o += l;
            offs.push(bytes.len() as u32);
        }
        let col = compress(
            &bytes,
            &offs,
            Config { bits: Bits::new(16).unwrap(), threshold: Threshold::new(0.5).unwrap(), seed: Some(42) },
        )
        .unwrap();
        let parts = col.as_search_parts();
        let dict = DictView { bytes: parts.dict_bytes, offsets: parts.dict_offsets };
        let nt = dict.num_tokens();
        let aut = KmpAutomaton::new(needle.as_bytes(), dict);
        let (base_runs, per_state) = aut.dump_dfa();
        // INNER set: DEFINITE tokens (base==m) + every token in a sparse range
        // whose target != 0. Collect the contiguous INNER ranges (these are what
        // SIMD range-tests check).
        let m = needle.len() as u8;
        let mut inner = vec![false; nt];
        let mut ranges: Vec<(u16, u16)> = Vec::new();
        for &(lo, hi, t) in &base_runs {
            if t == m {
                for i in lo..=hi { inner[i as usize] = true; }
                ranges.push((lo as u16, hi as u16));
            }
        }
        for trs in &per_state {
            for &(lo, hi, t) in trs {
                if t != 0 {
                    for i in lo..=hi { inner[i as usize] = true; }
                    ranges.push((lo, hi));
                }
            }
        }
        ranges.sort_unstable();
        let n_inner: usize = inner.iter().filter(|&&b| b).count();
        // Per-row: candidate iff any INNER token present.
        let codes = parts.codes;
        let co = parts.code_offsets;
        let mut cand_inner = 0usize;
        for r in 0..co.len() - 1 {
            let (s, e) = (co[r] as usize, co[r + 1] as usize);
            if codes[s..e].iter().any(|&c| inner[c as usize]) { cand_inner += 1; }
        }
        let rows = co.len() - 1;
        eprintln!("=== INNER (SIMD-rangeable) filter for {needle:?} ===");
        eprintln!("INNER tokens: {n_inner} in {} contiguous ranges (SIMD: {} lt/gt range tests)",
            ranges.len(), ranges.len());
        eprintln!("ranges: {ranges:?}");
        eprintln!("candidate rows (INNER present): {cand_inner} / {rows} ({:.2}%)",
            100.0 * cand_inner as f64 / rows as f64);
        eprintln!("(for comparison the adjacency chain marked ~0.5% candidate on i.yandex)");
    }

    /// Pack rows into the Arrow `(bytes, offsets)` pair `compress` expects.
    fn pack(rows: &[&[u8]]) -> (Vec<u8>, Vec<u32>) {
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for r in rows {
            bytes.extend_from_slice(r);
            offsets.push(bytes.len() as u32);
        }
        (bytes, offsets)
    }

    fn cfg() -> Config {
        Config {
            bits: Bits::new(12).unwrap(),
            threshold: Threshold::new(0.5).unwrap(),
            seed: Some(42),
        }
    }

    fn naive_contains(row: &[u8], needle: &[u8]) -> bool {
        needle.is_empty() || row.windows(needle.len()).any(|w| w == needle)
    }

    /// Materialise the set-row indices of a mask from its packed words.
    fn mask_ones(mask: &RowMask) -> Vec<usize> {
        let mut out = Vec::new();
        for (w, &word) in mask.as_words().iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                out.push(w * 64 + bits.trailing_zeros() as usize);
                bits &= bits - 1;
            }
        }
        out
    }

    fn assert_matches(rows: &[&[u8]], pattern: Pattern<'_>, expect: impl Fn(&[u8]) -> bool) {
        let (bytes, offsets) = pack(rows);
        let col = compress(&bytes, &offsets, cfg()).unwrap();
        let mask = col.as_search_parts().search(pattern);
        let got = mask_ones(&mask);
        let want: Vec<usize> = rows
            .iter()
            .enumerate()
            .filter_map(|(i, r)| expect(r).then_some(i))
            .collect();
        assert_eq!(got, want, "pattern {pattern:?}");
        assert_eq!(mask.len(), rows.len());
    }

    /// A corpus with heavy prefix sharing and repeated substrings so the
    /// trainer emits multi-byte tokens (exercising the sparse KMP transitions
    /// and prefix-divergence intervals rather than only single-byte tokens).
    fn url_corpus() -> Vec<Vec<u8>> {
        let hosts = ["https://www.example.com", "https://api.example.org", "ftp://x.example.net"];
        let paths = ["/index.html", "/search?q=onpair", "/a/b/c", "", "/login"];
        let mut out = Vec::new();
        let mut x = 0x1234_5678u64;
        for _ in 0..2000 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            let h = hosts[(x >> 33) as usize % hosts.len()];
            let p = paths[(x >> 17) as usize % paths.len()];
            out.push(format!("{h}{p}{}", x % 100).into_bytes());
        }
        out
    }

    #[test]
    fn contains_matches_naive_across_needles() {
        let owned = url_corpus();
        let rows: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();
        for needle in [
            b"example".as_slice(),
            b"https://".as_slice(),
            b"search?q=onpair".as_slice(),
            b"/a/b/c".as_slice(),
            b"zzz-not-present".as_slice(),
            b"e".as_slice(),
            b"".as_slice(),
        ] {
            assert_matches(&rows, Pattern::Contains(needle), |r| naive_contains(r, needle));
        }
    }

    #[test]
    fn prefix_matches_naive_across_needles() {
        let owned = url_corpus();
        let rows: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();
        for needle in [
            b"https://".as_slice(),
            b"https://www.example.com".as_slice(),
            b"ftp://".as_slice(),
            b"https://api.example.org/login".as_slice(),
            b"nope".as_slice(),
            b"".as_slice(),
        ] {
            assert_matches(&rows, Pattern::Prefix(needle), |r| r.starts_with(needle));
        }
    }

    #[test]
    fn single_byte_needles() {
        let rows: &[&[u8]] = &[b"abc", b"xyz", b"a", b"", b"cba"];
        for b in [b"a".as_slice(), b"z".as_slice(), b"q".as_slice()] {
            assert_matches(rows, Pattern::Contains(b), |r| naive_contains(r, b));
            assert_matches(rows, Pattern::Prefix(b), |r| r.starts_with(b));
        }
    }

    #[test]
    fn needle_longer_than_any_token() {
        // A 20-byte needle exceeds MAX_TOKEN_SIZE; prefix_range short-circuits.
        let rows: &[&[u8]] = &[b"this is a fairly long row of text", b"short"];
        let needle = b"fairly long row of t"; // 20 bytes
        assert_matches(rows, Pattern::Contains(needle), |r| naive_contains(r, needle));
        let pneedle = b"this is a fairly lon"; // 20 bytes
        assert_matches(rows, Pattern::Prefix(pneedle), |r| r.starts_with(pneedle));
    }
}
