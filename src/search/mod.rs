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

use kmp::{CLASS_DEFINITE, CLASS_OPENER, KmpAutomaton};
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
        let s = code_offsets[r].to_usize().expect("valid code offsets");
        let e = code_offsets[r + 1].to_usize().expect("valid code offsets");
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

/// Reduce a row's codes to a single class via the per-token `class` table:
/// [`CLASS_DEFINITE`] if any token contains the whole needle, else
/// [`CLASS_OPENER`] if any token can open a match, else `0` (reject). Short-
/// circuits on the first definite token. This is the scalar contains pass 1;
/// the dependent `class[code]` load pipelines across the loop (no carried
/// state), the same shape as the KMP fast path but with a one-byte verdict.
#[inline]
fn row_class(class: &[u8], codes: &[Token]) -> u8 {
    let mut acc = 0u8;
    for &c in codes {
        let cls = class[c as usize];
        if cls == CLASS_DEFINITE {
            return CLASS_DEFINITE;
        }
        acc |= cls;
    }
    acc
}

/// Pass-1 accept filter: set bit `r` of `acc` iff `first_codes[r]` lies in the
/// inclusive accept range `[alo, alo + awidth]` (unsigned). Branchless;
/// dispatches to AVX2 when available. Precondition for the SIMD path: the range
/// is non-empty (`alo <= u16::MAX`), which holds for every single-token query.
#[inline]
fn prefilter_accept(first_codes: &[u16], alo: u32, awidth: u32, acc: &mut [u64]) {
    #[cfg(target_arch = "x86_64")]
    if alo <= u16::MAX as u32 && avx2_enabled() {
        // SAFETY: avx2 just confirmed present.
        unsafe { prefilter_accept_avx2(first_codes, alo as u16, awidth as u16, acc) };
        return;
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

    /// Number of rows the mask covers (set or not).
    #[inline]
    pub fn len(&self) -> usize {
        self.rows
    }

    /// Whether the mask covers zero rows.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    /// Whether row `i` matched. Returns `false` for `i >= len()`.
    #[inline]
    pub fn contains(&self, i: usize) -> bool {
        i < self.rows && (self.words[i >> 6] >> (i & 63)) & 1 == 1
    }

    /// Number of matching rows.
    #[inline]
    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Iterate the indices of matching rows in ascending order.
    pub fn iter_ones(&self) -> impl Iterator<Item = usize> + '_ {
        self.words.iter().enumerate().flat_map(|(w, &word)| {
            BitIndices { word }.map(move |b| w * 64 + b)
        })
    }

    /// The packed bitmap words (LSB-first within each word). Length is
    /// `len().div_ceil(64)`.
    #[inline]
    pub fn as_words(&self) -> &[u64] {
        &self.words
    }
}

/// Iterator over the set-bit positions of a single `u64`, ascending.
struct BitIndices {
    word: u64,
}

impl Iterator for BitIndices {
    type Item = usize;
    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.word == 0 {
            return None;
        }
        let b = self.word.trailing_zeros() as usize;
        self.word &= self.word - 1;
        Some(b)
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

    /// Evaluate `pattern` against every row, invoking `on_match` with the
    /// 0-based index of each matching row, in order. The low-level primitive
    /// [`search`](Self::search) builds its [`RowMask`] on top of.
    pub fn search_callback(&self, pattern: Pattern<'_>, on_match: impl FnMut(usize)) {
        let dict = self.dict();
        match pattern {
            Pattern::Contains(needle) => {
                let aut = KmpAutomaton::new(needle, dict);
                self.scan_contains(&aut, dict.num_tokens(), on_match);
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
    /// reject majority. Falls back to the generic per-row scan for the empty
    /// needle, a saturated dictionary, or a malformed code stream.
    fn scan_contains(
        &self,
        aut: &KmpAutomaton,
        num_tokens: usize,
        mut on_match: impl FnMut(usize),
    ) {
        let n = self.code_offsets.len() - 1;
        if aut.is_empty_needle() || num_tokens > u16::MAX as usize + 1 {
            scan(aut, self.codes, self.code_offsets, on_match);
            return;
        }
        let class = aut.class_table();
        for r in 0..n {
            let s = self.code_offsets[r].to_usize().expect("valid code offsets");
            let e = self.code_offsets[r + 1].to_usize().expect("valid code offsets");
            match row_class(&class, &self.codes[s..e]) {
                CLASS_DEFINITE => on_match(r),
                CLASS_OPENER => {
                    if aut.matches(&self.codes[s..e]) {
                        on_match(r);
                    }
                }
                _ => {}
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
            let s = self.code_offsets[r].to_usize().expect("valid code offsets");
            let e = self.code_offsets[r + 1].to_usize().expect("valid code offsets");
            if aut.matches(&self.codes[s..e]) {
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
                let s = self.code_offsets[r].to_usize().expect("valid code offsets");
                let e = self.code_offsets[r + 1].to_usize().expect("valid code offsets");
                if aut.matches(&self.codes[s..e]) {
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

    fn assert_matches(rows: &[&[u8]], pattern: Pattern<'_>, expect: impl Fn(&[u8]) -> bool) {
        let (bytes, offsets) = pack(rows);
        let col = compress(&bytes, &offsets, cfg()).unwrap();
        let mask = col.as_search_parts().search(pattern);
        let got: Vec<usize> = mask.iter_ones().collect();
        let want: Vec<usize> = rows
            .iter()
            .enumerate()
            .filter_map(|(i, r)| expect(r).then_some(i))
            .collect();
        assert_eq!(got, want, "pattern {pattern:?}");
        assert_eq!(mask.len(), rows.len());
        assert_eq!(mask.count_ones(), want.len());
        // `contains` agrees with the index list.
        for i in 0..rows.len() {
            assert_eq!(mask.contains(i), want.contains(&i));
        }
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
