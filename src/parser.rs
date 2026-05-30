// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// `Parser` ties a trained `Dictionary` to a `LongestPrefixMatcher` so encoding
// is fast and self-contained. Codes are emitted as plain `u16` (no bit
// packing); decoders look up dict bytes directly.

use crate::column::Column;
use crate::config::Config;
use crate::config::Error;
use crate::config::TrainingConfig;
use crate::dict::Dictionary;
use crate::lpm::LongestPrefixMatcher;
use crate::offset::Offset;
use crate::trainer::TrainResult;
use crate::trainer::train;
use crate::types::MAX_TOKEN_SIZE;

/// Trained encoder: pairs the decode-side [`Dictionary`] with a crate-private
/// longest-prefix matcher that drives encoding. Build with [`Parser::train`];
/// encode with [`Parser::parse`].
#[derive(Debug, Clone)]
pub struct Parser {
    /// Decode-side dictionary, cloned into each [`Column`] produced by
    /// [`Parser::parse`] so columns are self-contained.
    pub dict: Dictionary,
    pub(crate) lpm: LongestPrefixMatcher,
}

impl Parser {
    /// Train a dictionary against `bytes` / `offsets` and build the matching
    /// LPM. `offsets` has length `n + 1`. Returns [`Error::InvalidArg`] if
    /// `offsets` is empty or its last (maximum) offset cannot be represented in
    /// `usize` or exceeds `bytes.len()` — see [`validate_offsets`]. The `cfg`
    /// is valid by construction ([`Bits`](crate::Bits) /
    /// [`Threshold`](crate::Threshold)).
    pub fn train<O: Offset>(bytes: &[u8], offsets: &[O], cfg: Config) -> Result<Self, Error> {
        validate_offsets(bytes, offsets)?;
        Ok(Self::train_unchecked(bytes, offsets, cfg))
    }

    /// Like [`Parser::train`] but skips offset validation. The caller must
    /// guarantee `(bytes, offsets)` is a valid Arrow-style pair (`offsets`
    /// non-empty and monotonic non-decreasing, every offset ≤ `bytes.len()`
    /// and representable in `usize`); slicing panics otherwise.
    pub(crate) fn train_unchecked<O: Offset>(bytes: &[u8], offsets: &[O], cfg: Config) -> Self {
        let internal_cfg: TrainingConfig = cfg.into();
        let TrainResult { dict, lpm } = train(bytes, offsets, &internal_cfg);
        Self { dict, lpm }
    }

    /// Encode `bytes` / `offsets` using this parser. The dictionary is cloned
    /// into the returned [`Column`] so the column is fully decode-self-
    /// contained — the strings need not be the corpus the parser was trained
    /// on. Returns [`Error::InvalidArg`] on invalid offsets — see
    /// [`validate_offsets`].
    pub fn parse<O: Offset>(&self, bytes: &[u8], offsets: &[O]) -> Result<Column<O>, Error> {
        validate_offsets(bytes, offsets)?;
        Ok(self.parse_unchecked(bytes, offsets))
    }

    /// Like [`Parser::parse`] but skips offset validation. Same caller
    /// guarantees as [`Parser::train_unchecked`].
    pub(crate) fn parse_unchecked<O: Offset>(&self, bytes: &[u8], offsets: &[O]) -> Column<O> {
        let (codes, code_offsets) = encode_strings(bytes, offsets, &self.lpm);
        let mut dict_bytes = self.dict.bytes.clone();
        // Decoder reads a fixed MAX_TOKEN_SIZE bytes from every token offset;
        // pad so that read is in bounds for the last token (worst case: a
        // 1-byte final token needs MAX_TOKEN_SIZE - 1 trailing bytes). See
        // `Parts::validate_dictionary`.
        dict_bytes.resize(dict_bytes.len() + (MAX_TOKEN_SIZE - 1), 0);
        let first_codes = Some(first_codes(&codes, &code_offsets));
        Column {
            dict_bytes,
            dict_offsets: self.dict.offsets.clone(),
            bits: self.dict.bits,
            codes,
            code_offsets,
            first_codes,
        }
    }
}

/// Build the per-row first-token side-table: `first_codes[r]` is the first
/// code of row `r`, or `u16::MAX` for an empty row (a sentinel that never
/// equals a real token id when the dictionary is not fully saturated).
pub(crate) fn first_codes<O: Offset>(codes: &[u16], code_offsets: &[O]) -> Vec<u16> {
    let n = code_offsets.len() - 1;
    let mut out = Vec::with_capacity(n);
    for r in 0..n {
        let s = code_offsets[r].to_usize().expect("valid code offsets");
        let e = code_offsets[r + 1].to_usize().expect("valid code offsets");
        out.push(if s < e { codes[s] } else { u16::MAX });
    }
    out
}

/// Encode every string into a flat `Vec<u16>` of codes plus per-row
/// `code_offsets`. Offset `[i]..[i + 1]` indexes the codes for row `i`. The
/// offsets are compressor metadata — a token may span a row boundary, so the
/// row structure cannot be recovered from the codes alone — and are not needed
/// to decode the column as one flat stream.
pub(crate) fn encode_strings<O: Offset>(
    bytes: &[u8],
    offsets: &[O],
    lpm: &LongestPrefixMatcher,
) -> (Vec<u16>, Vec<O>) {
    let n = offsets.len() - 1;
    let mut codes: Vec<u16> = Vec::with_capacity(bytes.len());
    let mut code_offsets: Vec<O> = Vec::with_capacity(n + 1);
    code_offsets.push(O::from_usize(0));
    for i in 0..n {
        let s = offsets[i].to_usize().expect("validated");
        let e = offsets[i + 1].to_usize().expect("validated");
        let mut pos = s;
        while pos < e {
            let (tok, mlen) = lpm.find_longest_match(&bytes[pos..e]);
            codes.push(tok);
            pos += mlen;
        }
        code_offsets.push(O::from_usize(codes.len()));
    }
    (codes, code_offsets)
}

/// Validate the `(bytes, offsets)` Arrow-style pair. `offsets` must be
/// non-empty and monotonic non-decreasing — the Arrow contract, relied on by
/// every consumer (slices `bytes[offsets[i]..offsets[i + 1]]`) and asserted
/// here only in debug builds. Given monotonicity the maximum offset is the
/// last, so a single bounds check on it suffices: it must fit in `usize` and
/// be ≤ `bytes.len()`. Empty offsets is a hard error. O(1) in release.
pub(crate) fn validate_offsets<O: Offset>(bytes: &[u8], offsets: &[O]) -> Result<(), Error> {
    debug_assert!(
        offsets
            .windows(2)
            .all(|w| w[0].to_usize() <= w[1].to_usize()),
        "offsets must be monotonic non-decreasing",
    );
    let last = offsets.last().ok_or(Error::InvalidArg)?;
    if last.to_usize().ok_or(Error::InvalidArg)? > bytes.len() {
        return Err(Error::InvalidArg);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FixedThreshold;
    use crate::config::ThresholdSpec;
    use crate::config::TrainingConfig;
    use crate::dict::Dictionary;
    use crate::test_corpus::alternating_strings as make_alternating_strings;
    use crate::test_corpus::binary_strings as make_binary_strings;
    use crate::test_corpus::homogeneous_strings as make_homogeneous_strings;
    use crate::test_corpus::make_raw;
    use crate::test_corpus::mixed_length_strings as make_mixed_length_strings;
    use crate::test_corpus::random_ascii_strings as make_random_strings;
    use crate::test_corpus::user_strings as make_user_strings;
    use crate::trainer::TrainResult;
    use crate::trainer::train;
    use crate::types::BitWidth;
    use crate::types::Token;

    fn make_base_dict() -> Dictionary {
        let mut d = Dictionary {
            bits: 16,
            ..Dictionary::default()
        };
        d.offsets.push(0);
        for i in 0u16..=255 {
            d.bytes.push(i as u8);
            d.offsets.push(d.bytes.len() as u32);
        }
        d
    }

    /// Decode the whole flat code stream against `dict`.
    fn decode_all(codes: &[u16], dict: &Dictionary) -> Vec<u8> {
        let mut out = Vec::new();
        for &c in codes {
            out.extend_from_slice(dict.data(c as Token));
        }
        out
    }

    /// Decode the codes for row `idx` (delimited by `code_offsets`) against
    /// `dict`.
    fn decode_row(codes: &[u16], code_offsets: &[u32], dict: &Dictionary, idx: usize) -> Vec<u8> {
        let begin = code_offsets[idx] as usize;
        let end = code_offsets[idx + 1] as usize;
        let mut out = Vec::new();
        for &c in &codes[begin..end] {
            out.extend_from_slice(dict.data(c as Token));
        }
        out
    }

    fn roundtrip_all<S: AsRef<[u8]>>(strings: &[S], bits: BitWidth, seed: u64) -> bool {
        if strings.is_empty() {
            return true;
        }
        let raw = make_raw(strings);
        let cfg = TrainingConfig {
            bits,
            threshold: ThresholdSpec::Fixed(FixedThreshold { value: 2 }),
            seed: Some(seed),
        };
        let TrainResult { dict, lpm } = train(&raw.data, &raw.offsets, &cfg);
        let (codes, _) = encode_strings(&raw.data, &raw.offsets, &lpm);
        // Rows decode as one flat concatenation, so the whole stream must equal
        // the concatenated input bytes.
        decode_all(&codes, &dict) == raw.data
    }

    const WIDTHS: &[BitWidth] = &[9, 10, 11, 12, 13, 14, 15, 16];

    #[test]
    fn zero_strings_produces_no_codes() {
        let lpm = LongestPrefixMatcher::new();
        let (codes, code_offsets) = encode_strings::<u32>(&[], &[0], &lpm);
        assert!(codes.is_empty());
        assert_eq!(code_offsets, vec![0u32]);
    }

    #[test]
    fn single_empty_string_produces_no_codes() {
        let lpm = LongestPrefixMatcher::new();
        let (codes, code_offsets) = encode_strings::<u32>(&[], &[0, 0], &lpm);
        assert!(codes.is_empty());
        assert_eq!(code_offsets, vec![0u32, 0]);
    }

    #[test]
    fn code_offsets_delimit_each_row() {
        // `code_offsets` has R + 1 entries, is monotonic, ends at codes.len(),
        // and each row's slice decodes back to that input row — even when tokens
        // span row boundaries.
        let lpm = LongestPrefixMatcher::new();
        let d = make_base_dict();
        let strings: &[&[u8]] = &[b"alpha", b"", b"beta beta", b"gamma"];
        let raw = make_raw(strings);
        let (codes, code_offsets) = encode_strings(&raw.data, &raw.offsets, &lpm);

        assert_eq!(code_offsets.len(), strings.len() + 1);
        assert_eq!(code_offsets[0], 0);
        assert_eq!(*code_offsets.last().unwrap() as usize, codes.len());
        for w in code_offsets.windows(2) {
            assert!(w[1] >= w[0], "code_offsets must be monotonic");
        }
        for (i, s) in strings.iter().enumerate() {
            assert_eq!(decode_row(&codes, &code_offsets, &d, i), *s);
        }
    }

    #[test]
    fn base_tokens_single_known_string() {
        let lpm = LongestPrefixMatcher::new();
        let d = make_base_dict();
        let expected = "Hello, World!";
        let raw = make_raw(&[expected]);
        let (codes, _) = encode_strings(&raw.data, &raw.offsets, &lpm);
        assert_eq!(decode_all(&codes, &d), expected.as_bytes());
    }

    #[test]
    fn base_tokens_all_single_byte_values() {
        let lpm = LongestPrefixMatcher::new();
        let d = make_base_dict();
        let strings: Vec<Vec<u8>> = (0u16..=255).map(|i| vec![i as u8]).collect();
        let raw = make_raw(&strings);
        let (codes, _) = encode_strings(&raw.data, &raw.offsets, &lpm);
        assert_eq!(decode_all(&codes, &d), raw.data);
    }

    #[test]
    fn trained_lpm_produces_multi_byte_tokens() {
        let strings = make_homogeneous_strings(50, 40, b'a');
        let raw = make_raw(&strings);
        let cfg = TrainingConfig {
            bits: 16,
            threshold: ThresholdSpec::Fixed(FixedThreshold { value: 2 }),
            seed: Some(42),
        };
        let TrainResult { dict: _, lpm } = train(&raw.data, &raw.offsets, &cfg);
        let (codes, _) = encode_strings(&raw.data, &raw.offsets, &lpm);
        // Multi-byte tokens mean fewer codes than input bytes.
        assert!(
            codes.len() < raw.data.len(),
            "parser did not use any multi-byte tokens"
        );
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
