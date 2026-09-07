// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The trained encoder: pairs a [`CompactDictionary`] with a [`LongestPrefixMatcher`]
//! that drives encoding. Build with [`Parser::train`]; encode with
//! [`Parser::parse`].

use crate::column::Column;
use crate::core::dictionary::{CompactDictionary, Dictionary};
use crate::core::offset::Offset;
use crate::core::types::{MAX_TOKEN_SIZE, Token};
use crate::encoding::config::{Config, Error, TrainingConfig};
use crate::encoding::flat::{FlatMatcher, KEY_MASKS};
use crate::encoding::lpm::LongestPrefixMatcher;
use crate::encoding::trainer::{TrainResult, train};

/// A trained encoder. Holds the [`CompactDictionary`] (cloned into each
/// [`Column`] so columns are self-contained) and a crate-private, encode-side
/// longest-prefix matcher built from it.
#[derive(Debug, Clone)]
pub struct Parser {
    /// The trained dictionary: sorted and read-padded.
    pub dict: CompactDictionary,
    /// Exact incremental matcher; the encode loop's fallback.
    pub(crate) lpm: LongestPrefixMatcher,
    /// Fixed-cost matcher that drives the encode loop.
    pub(crate) flat: FlatMatcher,
}

impl Parser {
    /// Train a dictionary against `bytes` / `offsets` and build the matching
    /// matcher. `offsets` has length `n + 1`.
    ///
    /// # Errors
    /// [`Error::InvalidArg`] if `offsets` is empty or its last entry exceeds
    /// `bytes.len()`.
    pub fn train<O: Offset>(bytes: &[u8], offsets: &[O], cfg: Config) -> Result<Self, Error> {
        validate_offsets(bytes, offsets)?;
        Ok(Self::train_unchecked(bytes, offsets, cfg))
    }

    /// Like [`Parser::train`] but skips offset validation. The caller guarantees
    /// `(bytes, offsets)` is a valid Arrow pair (non-empty, monotonic
    /// non-decreasing offsets, last `<= bytes.len()`).
    pub(crate) fn train_unchecked<O: Offset>(bytes: &[u8], offsets: &[O], cfg: Config) -> Self {
        let internal_cfg: TrainingConfig = cfg.into();
        let TrainResult { dict, lpm } = train(bytes, offsets, &internal_cfg);
        // `train` returns a dictionary that is sorted and read-padded by
        // construction — nothing left to do here.
        let flat = FlatMatcher::from_dictionary(dict.as_view(), &lpm);
        Self { dict, lpm, flat }
    }

    /// Encode `bytes` / `offsets` using this parser. The dictionary is cloned
    /// into the returned [`Column`], so the column is self-contained — the
    /// strings need not be the corpus the parser was trained on.
    ///
    /// # Errors
    /// [`Error::InvalidArg`] if `offsets` is empty or its last entry exceeds
    /// `bytes.len()`.
    pub fn parse<O: Offset>(&self, bytes: &[u8], offsets: &[O]) -> Result<Column<O>, Error> {
        validate_offsets(bytes, offsets)?;
        Ok(self.parse_unchecked(bytes, offsets))
    }

    /// [`Parser::parse`] with the cursor count exposed, so the best value for a
    /// machine can be measured rather than assumed. `K == 0` selects the
    /// single-cursor exact matcher.
    #[doc(hidden)]
    pub fn parse_lanes<O: Offset, const K: usize>(&self, bytes: &[u8], offsets: &[O]) -> Column<O> {
        let (codes, row_offsets) = if K == 0 {
            encode_strings(bytes, offsets, &self.lpm)
        } else {
            encode_strings_lanes::<O, K>(bytes, offsets, &self.flat, &self.lpm)
        };
        Column {
            dict: self.dict.clone(),
            codes,
            row_offsets,
        }
    }

    /// Like [`Parser::parse`] but skips offset validation; same caller
    /// guarantees as [`Parser::train_unchecked`].
    pub(crate) fn parse_unchecked<O: Offset>(&self, bytes: &[u8], offsets: &[O]) -> Column<O> {
        let (codes, row_offsets) =
            encode_strings_lanes::<O, LANES>(bytes, offsets, &self.flat, &self.lpm);
        // `self.dict` is already read-padded, so the cloned column dictionary is
        // too.
        Column {
            dict: self.dict.clone(),
            codes,
            row_offsets,
        }
    }
}

/// Number of independent row cursors the encode loop interleaves. Measured on
/// TPC-H text: throughput climbs from one to eight lanes and falls off past
/// twelve as the lane state outgrows the register file.
const LANES: usize = 8;

/// Largest supported lane count.
const MAX_LANES: usize = 16;

/// Encode every string into a flat code stream plus per-row offsets with the
/// exact matcher and a single cursor. Offset `[i]..[i + 1]` indexes the codes
/// for row `i`.
pub(crate) fn encode_strings<O: Offset>(
    bytes: &[u8],
    offsets: &[O],
    lpm: &LongestPrefixMatcher,
) -> (Vec<Token>, Vec<O>) {
    let n = offsets.len() - 1;
    let mut codes: Vec<Token> = Vec::with_capacity(bytes.len());
    let mut row_offsets: Vec<O> = Vec::with_capacity(n + 1);
    row_offsets.push(O::from_usize(0));
    for i in 0..n {
        let s = offsets[i].to_usize();
        let e = offsets[i + 1].to_usize();
        let mut pos = s;
        while pos < e {
            let (tok, mlen) = lpm.find_longest_match(&bytes[pos..e]);
            codes.push(tok);
            pos += mlen;
        }
        row_offsets.push(O::from_usize(codes.len()));
    }
    (codes, row_offsets)
}

/// One row cursor of the interleaved encode loop.
#[derive(Copy, Clone)]
struct Cursor {
    /// Next input byte.
    pos: usize,
    /// End of the current row.
    row_end: usize,
    /// Index of the current row.
    row: usize,
    /// Next code slot to write.
    out: usize,
}

/// The next `min(avail, 16)` bytes at `pos` as two little-endian words,
/// zero-extended.
///
/// # Safety
/// `pos + 16 <= bytes.len()`.
#[inline(always)]
unsafe fn load_window(bytes: &[u8], pos: usize, avail: usize) -> (u64, u64) {
    // SAFETY: `avail.min(16) <= 16`; the table has 17 entries.
    let [mlo, mhi] = unsafe { *KEY_MASKS.get_unchecked(avail.min(MAX_TOKEN_SIZE)) };
    // SAFETY: the caller guarantees 16 readable bytes at `pos`.
    let (lo, hi) = unsafe {
        let p = bytes.as_ptr().add(pos);
        (
            std::ptr::read_unaligned(p.cast::<u64>()),
            std::ptr::read_unaligned(p.add(8).cast::<u64>()),
        )
    };
    (lo & mlo, hi & mhi)
}

/// Everything the lockstep loop shares between lanes.
struct Encode<'a, O: Offset> {
    bytes: &'a [u8],
    offsets: &'a [O],
    /// Number of rows.
    n: usize,
    flat: &'a FlatMatcher,
    lpm: &'a LongestPrefixMatcher,
    /// Code buffer with room for one code per input byte per lane, plus one.
    codes: *mut Token,
    /// `n + 1` row offsets.
    row_offsets: *mut O,
}

impl<O: Offset> Encode<'_, O> {
    /// Emit `tok` and advance `c` by `len`, moving onto the next row when the
    /// current one ends. Branch-free. An empty row costs one step: nothing is
    /// emitted and the cursor moves on.
    ///
    /// # Safety
    /// `c.row < n` and `codes` has room at `c.out`.
    #[inline(always)]
    unsafe fn advance(&self, c: &mut Cursor, tok: Token, len: usize, avail: usize) {
        let emit = avail != 0;
        let len = std::hint::select_unpredictable(emit, len, 0);
        // SAFETY: the caller guarantees room at `out`; when nothing is emitted
        // the slot is overwritten by the next step.
        unsafe { *self.codes.add(c.out) = tok };
        c.out += emit as usize;
        c.pos += len;
        let ended = c.pos == c.row_end;
        // SAFETY: `row < n`, so `row + 1 <= n` indexes the `n + 1` entries.
        unsafe { *self.row_offsets.add(c.row + 1) = O::from_usize(c.out) };
        c.row += ended as usize;
        c.row_end = self.offsets[(c.row + 1).min(self.n)].to_usize();
    }

    /// Advance every cursor by one token, in lockstep: all lanes load their
    /// windows, then all lanes run search level 0, then level 1, and so on.
    /// Each level's `K` probes are independent, and grouping them keeps their
    /// loads adjacent in program order, so their latencies overlap without
    /// relying on the out-of-order window to reach across whole steps (a step
    /// is a few hundred instructions; the window holds about that many).
    ///
    /// # Safety
    /// For every lane: `pos + 16 <= bytes.len()`, `row < n`, and `codes` has
    /// room at `out`.
    #[inline(always)]
    unsafe fn step_lanes<const K: usize>(&self, cur: &mut [Cursor; K]) {
        debug_assert!(K >= 1 && K <= MAX_LANES);
        let mut avail = [0usize; K];
        let mut lo = [0u64; K];
        let mut hi = [0u64; K];
        let mut s = [self.flat.begin(0); K];
        for i in 0..K {
            avail[i] = cur[i].row_end - cur[i].pos;
            // SAFETY: the caller guarantees 16 readable bytes at `pos`.
            (lo[i], hi[i]) = unsafe { load_window(self.bytes, cur[i].pos, avail[i]) };
            s[i] = self.flat.begin(lo[i]);
        }
        for i in 0..K {
            self.flat.level::<0>(&mut s[i], lo[i], hi[i], avail[i]);
        }
        for i in 0..K {
            self.flat.level::<1>(&mut s[i], lo[i], hi[i], avail[i]);
        }
        for i in 0..K {
            self.flat.level::<2>(&mut s[i], lo[i], hi[i], avail[i]);
        }
        for i in 0..K {
            self.flat.level::<3>(&mut s[i], lo[i], hi[i], avail[i]);
        }
        for i in 0..K {
            let (mut tok, mut len, verified) = self.flat.finish(s[i], lo[i], hi[i]);
            if !verified {
                let (t, l) = self
                    .lpm
                    .find_longest_match(&self.bytes[cur[i].pos..cur[i].row_end]);
                tok = t;
                len = l;
            }
            // SAFETY: as documented on the caller.
            unsafe { self.advance(&mut cur[i], tok, len, avail[i]) };
        }
    }
}

/// [`encode_strings`] over the fixed-cost matcher with `K` interleaved cursors.
///
/// A single cursor is latency-bound: the bytes a step consumes come out of that
/// step's lookup, so the next lookup cannot start until it lands. Rows encode
/// independently, so `K` cursors over disjoint row ranges keep `K` of those
/// chains in flight, and the lookup has no data-dependent branches to flush
/// them. Every row is encoded exactly as a single exact-matcher cursor would
/// encode it.
pub(crate) fn encode_strings_lanes<O: Offset, const K: usize>(
    bytes: &[u8],
    offsets: &[O],
    flat: &FlatMatcher,
    lpm: &LongestPrefixMatcher,
) -> (Vec<Token>, Vec<O>) {
    debug_assert!(K >= 1);
    let n = offsets.len() - 1;
    let base = offsets[0].to_usize();
    let total = offsets[n].to_usize() - base;

    // Rows that end at least 16 bytes before the end of the buffer can be
    // read with one unaligned 16-byte load per step. Offsets are sorted, so
    // those rows are a prefix: `c` is the first row whose end is too close.
    let c = offsets[1..].partition_point(|&e| e.to_usize() + MAX_TOKEN_SIZE <= bytes.len());

    // Every lane gets a disjoint code region sized for its worst case (one
    // code per byte, plus one slot for the empty-row write), so lanes never
    // contend; regions are slid together at the end.
    let mut codes: Vec<Token> = Vec::with_capacity(total + K + 1);
    let mut row_offsets: Vec<O> = vec![O::default(); n + 1];
    let codes_ptr = codes.as_mut_ptr();
    let enc = Encode {
        bytes,
        offsets,
        n,
        flat,
        lpm,
        codes: codes_ptr,
        row_offsets: row_offsets.as_mut_ptr(),
    };

    let mut bounds = [0usize; K];
    for (lane, b) in bounds.iter_mut().enumerate() {
        *b = lane * c / K;
    }
    let mut region_start = [0usize; K];
    for lane in 0..K {
        region_start[lane] = offsets[bounds[lane]].to_usize() - base + lane;
    }
    let tail_start = offsets[c].to_usize() - base + K;

    let mut cur: [Cursor; K] = std::array::from_fn(|lane| Cursor {
        pos: offsets[bounds[lane]].to_usize(),
        row_end: offsets[(bounds[lane] + 1).min(n)].to_usize(),
        row: bounds[lane],
        out: region_start[lane],
    });
    let mut ends = [0usize; K];
    for lane in 0..K {
        ends[lane] = if lane + 1 < K { bounds[lane + 1] } else { c };
    }

    if c >= K {
        loop {
            let mut done = false;
            for lane in 0..K {
                done |= cur[lane].row == ends[lane];
            }
            if done {
                break;
            }
            // SAFETY: rows below `c` leave 16 readable bytes at every
            // position; each lane's region has room for one code per byte.
            unsafe { enc.step_lanes(&mut cur) };
        }
    }
    // Drain the lanes that still have rows once the first one ran out.
    for lane in 0..K {
        let mut one = [cur[lane]];
        while one[0].row < ends[lane] {
            // SAFETY: as above.
            unsafe { enc.step_lanes(&mut one) };
        }
        cur[lane] = one[0];
    }

    // Rows near the end of the buffer: exact matcher, single cursor.
    let mut tail_out = tail_start;
    for i in c..n {
        let s = offsets[i].to_usize();
        let e = offsets[i + 1].to_usize();
        let mut pos = s;
        while pos < e {
            let (tok, mlen) = lpm.find_longest_match(&bytes[pos..e]);
            // SAFETY: the tail region has room for one code per byte.
            unsafe { *codes_ptr.add(tail_out) = tok };
            tail_out += 1;
            pos += mlen;
        }
        row_offsets[i + 1] = O::from_usize(tail_out);
    }

    // Slide each region down onto the end of the previous one and shift its
    // row offsets to match.
    let mut written = 0usize;
    let mut compact = |start: usize, end: usize, rows: std::ops::Range<usize>| {
        let len = end - start;
        if start != written {
            // SAFETY: both ranges lie inside the reserved capacity and the
            // destination starts at or before the source.
            unsafe { std::ptr::copy(codes_ptr.add(start), codes_ptr.add(written), len) };
        }
        for r in rows {
            let v = row_offsets[r + 1].to_usize();
            row_offsets[r + 1] = O::from_usize(v - start + written);
        }
        written += len;
    };
    for lane in 0..K {
        compact(region_start[lane], cur[lane].out, bounds[lane]..ends[lane]);
    }
    compact(tail_start, tail_out, c..n);
    // SAFETY: `written` codes were initialized above.
    unsafe { codes.set_len(written) };
    (codes, row_offsets)
}

/// Validate the `(bytes, offsets)` Arrow pair: `offsets` must be non-empty and
/// monotonic non-decreasing (the Arrow contract, debug-asserted), and its last
/// (maximum) offset must fit and be `<= bytes.len()`. `O(1)` in release.
pub(crate) fn validate_offsets<O: Offset>(bytes: &[u8], offsets: &[O]) -> Result<(), Error> {
    debug_assert!(
        offsets
            .windows(2)
            .all(|w| w[0].to_usize() <= w[1].to_usize()),
        "offsets must be monotonic non-decreasing",
    );
    let last = offsets.last().ok_or(Error::InvalidArg)?;
    if last.to_usize() > bytes.len() {
        return Err(Error::InvalidArg);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dictionary::{
        CompactDictionary, CompactDictionaryView, Dictionary, DictionaryView,
    };
    use crate::encoding::config::{FixedThreshold, ThresholdSpec, TrainingConfig};
    use crate::encoding::trainer::train;
    use crate::test_corpus::{
        alternating_strings as make_alternating_strings, binary_strings as make_binary_strings,
        homogeneous_strings as make_homogeneous_strings, make_raw,
        mixed_length_strings as make_mixed_length_strings,
        random_ascii_strings as make_random_strings, user_strings as make_user_strings,
    };

    fn make_base_dict() -> CompactDictionary {
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for i in 0u16..=255 {
            bytes.push(i as u8);
            offsets.push(bytes.len() as u32);
        }
        CompactDictionary::from_raw(bytes, offsets)
    }

    /// Decode the whole flat code stream against `dict`.
    fn decode_all(codes: &[Token], dict: CompactDictionaryView<'_>) -> Vec<u8> {
        let mut out = Vec::new();
        for &c in codes {
            out.extend_from_slice(dict.token(c));
        }
        out
    }

    /// Decode the codes for row `idx` against `dict`.
    fn decode_row(
        codes: &[Token],
        row_offsets: &[u32],
        dict: CompactDictionaryView<'_>,
        idx: usize,
    ) -> Vec<u8> {
        let begin = row_offsets[idx] as usize;
        let end = row_offsets[idx + 1] as usize;
        let mut out = Vec::new();
        for &c in &codes[begin..end] {
            out.extend_from_slice(dict.token(c));
        }
        out
    }

    /// Every lane count must produce exactly the single-cursor exact encoding.
    fn check_lanes_agree<S: AsRef<[u8]>>(strings: &[S], max_dict_bits: u8, seed: u64) {
        let raw = make_raw(strings);
        let cfg = TrainingConfig {
            max_dict_bits,
            threshold: ThresholdSpec::Fixed(FixedThreshold { value: 2 }),
            seed: Some(seed),
        };
        let TrainResult { dict, lpm } = train(&raw.data, &raw.offsets, &cfg);
        let flat = FlatMatcher::from_dictionary(dict.as_view(), &lpm);
        let want = encode_strings(&raw.data, &raw.offsets, &lpm);
        fn check<const K: usize>(
            raw: &crate::test_corpus::Raw,
            flat: &FlatMatcher,
            lpm: &LongestPrefixMatcher,
            want: &(Vec<Token>, Vec<u32>),
        ) {
            let got = encode_strings_lanes::<u32, K>(&raw.data, &raw.offsets, flat, lpm);
            assert_eq!(got.0, want.0, "codes differ at K={K}");
            assert_eq!(got.1, want.1, "row offsets differ at K={K}");
        }
        check::<1>(&raw, &flat, &lpm, &want);
        check::<2>(&raw, &flat, &lpm, &want);
        check::<3>(&raw, &flat, &lpm, &want);
        check::<4>(&raw, &flat, &lpm, &want);
        check::<8>(&raw, &flat, &lpm, &want);
    }

    #[test]
    fn lanes_agree_with_exact_single_cursor() {
        for &bits in WIDTHS {
            check_lanes_agree(&make_user_strings(120), bits, 42);
            check_lanes_agree(&make_mixed_length_strings(200, 100, 31415), bits, 7);
            check_lanes_agree(&make_binary_strings(80, 30, 777), bits, 3);
        }
        // Empty rows everywhere, including runs and at both ends.
        let ragged: Vec<&[u8]> = vec![
            b"",
            b"",
            b"alpha",
            b"",
            b"beta beta",
            b"g",
            b"",
            b"",
            b"gamma delta epsilon",
            b"",
            b"",
        ];
        check_lanes_agree(&ragged, 12, 1);
        // Fewer rows than lanes, and a single row.
        check_lanes_agree(&[b"one row only".as_slice()], 12, 1);
        check_lanes_agree(&[b"a".as_slice(), b"bb"], 12, 1);
        check_lanes_agree::<&[u8]>(&[], 12, 1);
    }

    fn roundtrip_all<S: AsRef<[u8]>>(strings: &[S], max_dict_bits: u8, seed: u64) -> bool {
        if strings.is_empty() {
            return true;
        }
        let raw = make_raw(strings);
        let cfg = TrainingConfig {
            max_dict_bits,
            threshold: ThresholdSpec::Fixed(FixedThreshold { value: 2 }),
            seed: Some(seed),
        };
        let TrainResult { dict, lpm } = train(&raw.data, &raw.offsets, &cfg);
        let (codes, _) = encode_strings(&raw.data, &raw.offsets, &lpm);
        decode_all(&codes, dict.as_view()) == raw.data
    }

    const WIDTHS: &[u8] = &[9, 10, 11, 12, 13, 14, 15, 16];

    #[test]
    fn zero_strings_produces_no_codes() {
        let lpm = LongestPrefixMatcher::new();
        let (codes, row_offsets) = encode_strings::<u32>(&[], &[0], &lpm);
        assert!(codes.is_empty());
        assert_eq!(row_offsets, vec![0u32]);
    }

    #[test]
    fn single_empty_string_produces_no_codes() {
        let lpm = LongestPrefixMatcher::new();
        let (codes, row_offsets) = encode_strings::<u32>(&[], &[0, 0], &lpm);
        assert!(codes.is_empty());
        assert_eq!(row_offsets, vec![0u32, 0]);
    }

    #[test]
    fn row_offsets_delimit_each_row() {
        let lpm = LongestPrefixMatcher::new();
        let d = make_base_dict();
        let strings: &[&[u8]] = &[b"alpha", b"", b"beta beta", b"gamma"];
        let raw = make_raw(strings);
        let (codes, row_offsets) = encode_strings(&raw.data, &raw.offsets, &lpm);

        assert_eq!(row_offsets.len(), strings.len() + 1);
        assert_eq!(row_offsets[0], 0);
        assert_eq!(*row_offsets.last().unwrap() as usize, codes.len());
        for w in row_offsets.windows(2) {
            assert!(w[1] >= w[0], "row_offsets must be monotonic");
        }
        for (i, s) in strings.iter().enumerate() {
            assert_eq!(decode_row(&codes, &row_offsets, d.as_view(), i), *s);
        }
    }

    #[test]
    fn base_tokens_single_known_string() {
        let lpm = LongestPrefixMatcher::new();
        let d = make_base_dict();
        let raw = make_raw(&["Hello, World!"]);
        let (codes, _) = encode_strings(&raw.data, &raw.offsets, &lpm);
        assert_eq!(decode_all(&codes, d.as_view()), b"Hello, World!");
    }

    #[test]
    fn trained_lpm_produces_multi_byte_tokens() {
        let raw = make_raw(&make_homogeneous_strings(50, 40, b'a'));
        let cfg = TrainingConfig {
            max_dict_bits: 16,
            threshold: ThresholdSpec::Fixed(FixedThreshold { value: 2 }),
            seed: Some(42),
        };
        let TrainResult { dict: _, lpm } = train(&raw.data, &raw.offsets, &cfg);
        let (codes, _) = encode_strings(&raw.data, &raw.offsets, &lpm);
        assert!(
            codes.len() < raw.data.len(),
            "parser did not use multi-byte tokens"
        );
    }

    #[test]
    fn validate_offsets_rejects_empty_and_overflow() {
        assert_eq!(validate_offsets::<u32>(b"abc", &[]), Err(Error::InvalidArg));
        assert_eq!(validate_offsets(b"abc", &[0u32, 4]), Err(Error::InvalidArg));
        assert_eq!(validate_offsets(b"abc", &[0u32, 3]), Ok(()));
    }

    #[test]
    fn roundtrip_user_strings() {
        for &bits in WIDTHS {
            assert!(roundtrip_all(&make_user_strings(50), bits, 42));
        }
    }

    #[test]
    fn roundtrip_random_ascii_strings() {
        for &bits in WIDTHS {
            assert!(roundtrip_all(&make_random_strings(60, 50, 1337), bits, 42));
        }
    }

    #[test]
    fn roundtrip_binary_strings_with_nul_bytes() {
        for &bits in WIDTHS {
            assert!(roundtrip_all(&make_binary_strings(40, 30, 777), bits, 42));
        }
    }

    #[test]
    fn roundtrip_homogeneous_strings() {
        for &bits in WIDTHS {
            assert!(roundtrip_all(
                &make_homogeneous_strings(30, 40, b'a'),
                bits,
                42
            ));
        }
    }

    #[test]
    fn roundtrip_alternating_strings() {
        for &bits in WIDTHS {
            assert!(roundtrip_all(&make_alternating_strings(30, 40), bits, 42));
        }
    }

    #[test]
    fn roundtrip_mixed_length_strings() {
        for &bits in WIDTHS {
            assert!(roundtrip_all(
                &make_mixed_length_strings(80, 100, 31415),
                bits,
                42
            ));
        }
    }
}
