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
use crate::config::validate_config;
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
    pub dict: Dictionary,
    pub(crate) lpm: LongestPrefixMatcher,
}

impl Parser {
    /// Train a dictionary against `bytes` / `offsets` and build the matching
    /// LPM. `offsets` has length `n + 1`. Returns [`Error::InvalidArg`] on
    /// bad `cfg`, on any `offsets[i] > bytes.len()`, on `offsets.is_empty()`,
    /// or if any offset cannot be represented in `usize`.
    pub fn train<O: Offset>(bytes: &[u8], offsets: &[O], cfg: Config) -> Result<Self, Error> {
        validate_config(cfg)?;
        validate_offsets(bytes, offsets)?;

        let internal_cfg: TrainingConfig = cfg.into();
        let TrainResult { dict, lpm } = train(bytes, offsets, &internal_cfg);
        Ok(Self { dict, lpm })
    }

    /// Encode `bytes` / `offsets` using this parser. The dictionary is cloned
    /// into the returned [`Column`] so the column is fully decode-self-
    /// contained — the strings need not be the corpus the parser was trained
    /// on.
    pub fn parse<O: Offset>(&self, bytes: &[u8], offsets: &[O]) -> Result<Column<O>, Error> {
        validate_offsets(bytes, offsets)?;
        let (codes, code_boundaries) = encode_strings(bytes, offsets, &self.lpm);
        let mut dict_bytes = self.dict.bytes.clone();
        // Decoder reads a fixed MAX_TOKEN_SIZE bytes from every token offset;
        // pad so that read is in bounds for the last token (worst case: a
        // 1-byte final token needs MAX_TOKEN_SIZE - 1 trailing bytes). See
        // `Parts::validate_dictionary`.
        dict_bytes.resize(dict_bytes.len() + (MAX_TOKEN_SIZE - 1), 0);
        Ok(Column {
            dict_bytes,
            dict_offsets: self.dict.offsets.clone(),
            bits: self.dict.bits,
            codes,
            code_boundaries,
        })
    }
}

/// Encode every string into a flat `Vec<u16>` of codes plus per-row token
/// boundaries. Output boundary `[i]..[i+1]` indexes into `codes`.
pub(crate) fn encode_strings<O: Offset>(
    bytes: &[u8],
    offsets: &[O],
    lpm: &LongestPrefixMatcher,
) -> (Vec<u16>, Vec<O>) {
    let n = offsets.len() - 1;
    let mut codes: Vec<u16> = Vec::with_capacity(bytes.len());
    let mut boundaries: Vec<O> = Vec::with_capacity(n + 1);
    boundaries.push(O::from_usize(0));
    for i in 0..n {
        let s = offsets[i].to_usize().expect("validated");
        let e = offsets[i + 1].to_usize().expect("validated");
        let mut pos = s;
        while pos < e {
            let (tok, mlen) = lpm.find_longest_match(&bytes[pos..e]);
            codes.push(tok);
            pos += mlen;
        }
        boundaries.push(O::from_usize(codes.len()));
    }
    (codes, boundaries)
}

/// Validate the `(bytes, offsets)` Arrow-style pair. Empty offsets is a hard
/// error; otherwise every offset must fit in `usize` and be ≤ `bytes.len()`.
pub(crate) fn validate_offsets<O: Offset>(bytes: &[u8], offsets: &[O]) -> Result<(), Error> {
    if offsets.is_empty() {
        return Err(Error::InvalidArg);
    }
    for o in offsets {
        let p = o.to_usize().ok_or(Error::InvalidArg)?;
        if p > bytes.len() {
            return Err(Error::InvalidArg);
        }
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

    /// Decode all tokens for row `idx` against `dict`.
    fn decode_tokens(codes: &[u16], boundaries: &[u32], dict: &Dictionary, idx: usize) -> Vec<u8> {
        let begin = boundaries[idx] as usize;
        let end = boundaries[idx + 1] as usize;
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
        let (codes, boundaries) = encode_strings(&raw.data, &raw.offsets, &lpm);
        for i in 0..strings.len() {
            let decoded = decode_tokens(&codes, &boundaries, &dict, i);
            if decoded != strings[i].as_ref() {
                return false;
            }
        }
        true
    }

    const WIDTHS: &[BitWidth] = &[9, 10, 11, 12, 13, 14, 15, 16];

    #[test]
    fn zero_strings_produces_one_boundary() {
        let lpm = LongestPrefixMatcher::new();
        let (codes, boundaries) = encode_strings::<u32>(&[], &[0], &lpm);
        assert_eq!(boundaries, vec![0u32]);
        assert!(codes.is_empty());
    }

    #[test]
    fn single_empty_string_produces_two_zero_boundaries() {
        let lpm = LongestPrefixMatcher::new();
        let (codes, boundaries) = encode_strings::<u32>(&[], &[0, 0], &lpm);
        assert_eq!(boundaries, vec![0u32, 0]);
        assert!(codes.is_empty());
    }

    #[test]
    fn boundary_count_is_n_plus_one() {
        let lpm = LongestPrefixMatcher::new();
        let raw = make_raw(&make_user_strings(20));
        let (_, boundaries) = encode_strings(&raw.data, &raw.offsets, &lpm);
        assert_eq!(boundaries.len(), raw.n + 1);
    }

    #[test]
    fn boundaries_are_monotonic() {
        let lpm = LongestPrefixMatcher::new();
        let raw = make_raw(&make_random_strings(25, 40, 7));
        let (_, boundaries) = encode_strings(&raw.data, &raw.offsets, &lpm);
        for i in 1..boundaries.len() {
            assert!(
                boundaries[i] >= boundaries[i - 1],
                "non-monotonic at index {i}"
            );
        }
    }

    #[test]
    fn last_boundary_equals_total_token_count() {
        let lpm = LongestPrefixMatcher::new();
        let raw = make_raw(&make_random_strings(15, 30, 99));
        let (codes, boundaries) = encode_strings(&raw.data, &raw.offsets, &lpm);
        assert_eq!(*boundaries.last().unwrap() as usize, codes.len());
    }

    #[test]
    fn base_tokens_single_known_string() {
        let lpm = LongestPrefixMatcher::new();
        let d = make_base_dict();
        let expected = "Hello, World!";
        let raw = make_raw(&[expected]);
        let (codes, boundaries) = encode_strings(&raw.data, &raw.offsets, &lpm);
        assert_eq!(
            decode_tokens(&codes, &boundaries, &d, 0),
            expected.as_bytes()
        );
    }

    #[test]
    fn base_tokens_all_single_byte_values() {
        let lpm = LongestPrefixMatcher::new();
        let d = make_base_dict();
        let strings: Vec<Vec<u8>> = (0u16..=255).map(|i| vec![i as u8]).collect();
        let raw = make_raw(&strings);
        let (codes, boundaries) = encode_strings(&raw.data, &raw.offsets, &lpm);
        for (i, s) in strings.iter().enumerate() {
            assert_eq!(
                decode_tokens(&codes, &boundaries, &d, i),
                *s,
                "mismatch for byte value {i}"
            );
        }
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
        let (_, boundaries) = encode_strings(&raw.data, &raw.offsets, &lpm);
        let tokens_0 = boundaries[1] - boundaries[0];
        assert!(tokens_0 < 40, "parser did not use any multi-byte tokens");
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
