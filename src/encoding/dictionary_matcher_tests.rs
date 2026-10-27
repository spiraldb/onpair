use hegel::TestCase;
use hegel::generators as gs;

use super::*;
use crate::core::dictionary::{CompactDictionary, Dictionary, DictionaryView};
use crate::test_corpus::{PROPERTY_ALPHABET, complete_dictionary};

fn check<const K: usize>(
    dict: &CompactDictionary,
    bytes: &[u8],
    at: &[usize; K],
    win: &[usize; K],
) {
    let matcher = DictionaryMatcher::from_dictionary(dict.as_view());
    let scalar = matcher.find_all(bytes, at, win);
    assert_eq!(matcher.find_all_middle(bytes, at, win), scalar);
    for j in 0..K {
        let view = dict.as_view();
        let expected = (0..view.num_tokens())
            .map(|i| (i as Token, view.token(i as Token)))
            .filter(|(_, t)| bytes[at[j]..at[j] + win[j]].starts_with(t))
            .max_by_key(|(_, t)| t.len())
            .map(|(i, t)| (i, t.len()))
            .unwrap();
        assert_eq!(scalar[j], expected);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("popcnt")
        {
            // SAFETY: the required target features were detected.
            assert_eq!(unsafe { matcher.find_all_avx2(bytes, at, win) }, scalar);
        }
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("popcnt")
        {
            // SAFETY: the required target features were detected.
            assert_eq!(unsafe { matcher.find_all_avx512(bytes, at, win) }, scalar);
        }
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        // SAFETY: NEON was detected.
        assert_eq!(unsafe { matcher.find_all_neon(bytes, at, win) }, scalar);
    }
}

#[hegel::test(test_cases = 400)]
fn rank_paths_agree_with_scalar_and_brute_force(tc: TestCase) {
    let bytes = || gs::vecs(gs::sampled_from(&PROPERTY_ALPHABET[..]));
    let mut extra: Vec<Vec<u8>> =
        tc.draw(gs::vecs(bytes().min_size(2).max_size(16)).max_size(2000));
    extra.extend([
        b"a\0".to_vec(),
        b"a\0\0".to_vec(),
        vec![0; 16],
        vec![0xFF; 16],
    ]);
    for id in 0..64u16 {
        let mut token = b"abcdef".to_vec();
        token.extend_from_slice(&id.to_be_bytes());
        token.resize(16, 0xA5);
        extra.push(token);
    }
    let refs: Vec<&[u8]> = extra.iter().map(Vec::as_slice).collect();
    let dict = complete_dictionary(&refs);
    let mut data: Vec<u8> = tc.draw(bytes().min_size(1).max_size(80));
    data.extend_from_slice(b"abcdef\0\x20\xA4\0\0\0\0\0\0\0");
    data.extend_from_slice(&[0xFF; 17]);
    let middle = tc.draw(gs::integers::<usize>().max_value(data.len() - 1));
    let at = [
        0,
        middle,
        data.len() - 33,
        data.len() - 17,
        data.len() - 2,
        data.len() - 1,
    ];
    let win = std::array::from_fn(|j| {
        tc.draw(
            gs::integers::<usize>()
                .min_value(1)
                .max_value((data.len() - at[j]).min(16)),
        )
    });
    check(&dict, &data, &at, &win);
}

#[test]
fn range_level_boundaries_and_long_equal_prefix_runs() {
    for n in [0usize, 1, 16, 17, 272, 273, 4624, 4625, 65_280] {
        for tied in [false, true] {
            let extra: Vec<Vec<u8>> = (0..n)
                .map(|i| {
                    let mut token = if tied {
                        b"abcdef".to_vec()
                    } else {
                        b"ab".to_vec()
                    };
                    token.extend_from_slice(&(i as u32).to_be_bytes());
                    token.resize(16, 0xA5);
                    token
                })
                .collect();
            let refs: Vec<&[u8]> = extra.iter().map(Vec::as_slice).collect();
            let dict = complete_dictionary(&refs);
            for id in [0, n / 2, n.saturating_sub(1), n] {
                let mut data = if tied {
                    b"abcdef".to_vec()
                } else {
                    b"ab".to_vec()
                };
                data.extend_from_slice(&(id as u32).to_be_bytes());
                data.resize(16, 0xA5);
                check(&dict, &data, &[0, 0, 0, 15], &[16, 6, 1, 1]);
            }
        }
    }
}

#[test]
fn empty_last_ranges_zero_keys_and_ff_windows() {
    for extra in [
        vec![],
        vec![
            b"a\0".to_vec(),
            vec![0; 16],
            vec![0xFF; 2],
            vec![0xFF, 0xFF, 0xFF, b'a'],
        ],
        vec![vec![0xFF; 16]],
    ] {
        let refs: Vec<&[u8]> = extra.iter().map(Vec::as_slice).collect();
        let dict = complete_dictionary(&refs);
        for byte in [0, 0xFF] {
            let data = [byte; 20];
            let at = std::array::from_fn::<_, 20, _>(|i| i);
            let win = std::array::from_fn(|i| (20 - i).min(16));
            check(&dict, &data, &at, &win);
        }
    }
}
