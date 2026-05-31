// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use super::*;
use crate::{DEFAULT_CONFIG, compress};

/// Concatenate `rows` into the flat (bytes, offsets) OnPair input shape.
fn corpus(rows: &[&[u8]]) -> (Vec<u8>, Vec<u32>) {
    let mut bytes = Vec::new();
    let mut offsets = vec![0u32];
    for row in rows {
        bytes.extend_from_slice(row);
        offsets.push(bytes.len() as u32);
    }
    (bytes, offsets)
}

#[test]
fn decompress_view_recovers_rows() {
    let rows: &[&[u8]] = &[
        b"alpha",
        b"",
        b"beta beta beta beta",
        b"gamma",
        b"a-much-longer-string-than-twelve-bytes",
    ];
    let (bytes, offsets) = corpus(rows);
    let col = compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap();

    let view = col.decompress_view();
    assert_eq!(view.len(), rows.len());
    assert_eq!(view.values, bytes, "values match a flat decode");
    for (r, expected) in rows.iter().enumerate() {
        assert_eq!(view.row(r), *expected, "row {r}");
    }
    // Offsets are contiguous and cover the whole buffer.
    assert_eq!(view.offsets[0], 0);
    assert_eq!(*view.offsets.last().unwrap() as usize, bytes.len());
}

#[test]
fn decompress_row_matches_whole_column() {
    let rows: &[&[u8]] = &[
        b"alpha",
        b"",
        b"beta beta beta beta",
        b"g",
        b"a-much-longer-string-than-twelve-bytes",
    ];
    let (bytes, offsets) = corpus(rows);
    let col = compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap();

    // Each single-row decode must equal the corresponding row of a full decode,
    // for every row (including the empty one, the first, and the last).
    let view = col.decompress_view();
    let mut buf = Vec::new();
    for r in 0..rows.len() {
        assert_eq!(col.decompress_row(r), rows[r], "row {r} (owned)");
        decompress_row_into(col.as_parts(), &col.code_offsets, r, &mut buf);
        assert_eq!(buf, view.row(r), "row {r} (into, vs view)");
    }
}

#[test]
#[should_panic]
fn decompress_row_out_of_range_panics() {
    let (bytes, offsets) = corpus(&[b"a", b"bc"]);
    let col = compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap();
    let _ = col.decompress_row(2); // only rows 0 and 1 exist
}

#[test]
fn row_byte_offsets_matches_input_offsets() {
    let rows: &[&[u8]] = &[b"one", b"", b"three!!", b"four four four four"];
    let (bytes, offsets) = corpus(rows);
    let col = compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap();

    // Per-row byte offsets recovered from codes must equal the original input
    // offsets (the encode/decode round trip preserves row boundaries).
    let got = row_byte_offsets(col.as_parts(), &col.code_offsets);
    assert_eq!(got, offsets);
}

#[test]
fn build_views_inline_vs_reference_split() {
    // 12 bytes: the largest inline length. 13 bytes: smallest reference.
    let inline12: &[u8] = b"twelve_bytes";
    let ref13: &[u8] = b"thirteen_byte";
    assert_eq!(inline12.len(), BinaryView::INLINE_LEN);
    assert_eq!(ref13.len(), BinaryView::INLINE_LEN + 1);

    let rows: &[&[u8]] = &[b"", b"short", inline12, ref13, b"loooong reference value"];
    let (bytes, offsets) = corpus(rows);
    let col = compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap();
    let view = col.decompress_view();
    let views = build_views(&view);

    assert_eq!(views.len(), rows.len());
    let buffers: &[&[u8]] = &[&view.values];
    for (r, expected) in rows.iter().enumerate() {
        let v = &views[r];
        assert_eq!(v.len() as usize, expected.len(), "row {r} len");
        assert_eq!(v.is_inline(), expected.len() <= BinaryView::INLINE_LEN);
        // Resolving the descriptor recovers the exact bytes either way.
        assert_eq!(v.resolve(buffers), *expected, "row {r} resolve");
    }

    // Spot-check the inline / reference boundary explicitly.
    assert!(views[2].is_inline(), "12-byte row is inline");
    assert!(!views[3].is_inline(), "13-byte row is a reference");
    assert_eq!(views[3].inline_bytes(), None);
    let (buffer, offset) = views[3].reference_location().unwrap();
    assert_eq!(buffer, 0);
    assert_eq!(offset as usize, offsets[3] as usize);
}

#[test]
fn binary_view_arrow_layout() {
    // Inline: [len:u32 LE][bytes][zero pad].
    let v = BinaryView::inline(b"abc");
    let raw = v.to_le_bytes();
    assert_eq!(&raw[0..4], &3u32.to_le_bytes());
    assert_eq!(&raw[4..7], b"abc");
    assert_eq!(&raw[7..16], &[0u8; 9]);
    assert!(v.is_inline());
    assert_eq!(v.inline_bytes(), Some(&b"abc"[..]));

    // Reference: [len:u32][prefix:4][buffer:u32][offset:u32], all LE.
    let v = BinaryView::reference(100, *b"PREF", 2, 4096);
    let raw = v.to_le_bytes();
    assert_eq!(&raw[0..4], &100u32.to_le_bytes());
    assert_eq!(&raw[4..8], b"PREF");
    assert_eq!(&raw[8..12], &2u32.to_le_bytes());
    assert_eq!(&raw[12..16], &4096u32.to_le_bytes());
    assert!(!v.is_inline());
    assert_eq!(v.reference_location(), Some((2, 4096)));
}

#[test]
fn empty_column_is_empty_view() {
    let bytes: Vec<u8> = Vec::new();
    let offsets = vec![0u32];
    let col = compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap();
    let view = col.decompress_view();
    assert!(view.is_empty());
    assert_eq!(view.offsets, vec![0u32]);
    assert!(build_views(&view).is_empty());
}

#[test]
fn all_empty_rows() {
    let rows: &[&[u8]] = &[b"", b"", b""];
    let (bytes, offsets) = corpus(rows);
    let col = compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap();
    let view = col.decompress_view();
    assert_eq!(view.len(), 3);
    assert_eq!(view.values, Vec::<u8>::new());
    let views = build_views(&view);
    assert!(views.iter().all(|v| v.is_empty() && v.is_inline()));
}

/// A naive, obviously-correct `build_views` to cross-check the optimized one.
fn oracle(view: &DecodedView) -> Vec<BinaryView> {
    (0..view.len())
        .map(|r| {
            let bytes = view.row(r);
            if bytes.len() <= BinaryView::INLINE_LEN {
                BinaryView::inline(bytes)
            } else {
                let prefix: [u8; 4] = bytes[..4].try_into().unwrap();
                BinaryView::reference(bytes.len() as u32, prefix, 0, view.offsets[r])
            }
        })
        .collect()
}

#[test]
fn build_views_matches_oracle_across_boundary() {
    // Cover three regimes: a buffer shorter than the 16-byte over-read window
    // (every row hits the scalar tail), a buffer where rows straddle the
    // fast/tail boundary, and an exactly-16-byte buffer.
    for tail in 0..20usize {
        let mut rows: Vec<Vec<u8>> = vec![b"a-reference-length-value-here".to_vec(); 8];
        // Append short rows so the final bytes sit in the tail window.
        for k in 0..tail {
            rows.push(vec![b'x'; k % 5]);
        }
        let slices: Vec<&[u8]> = rows.iter().map(Vec::as_slice).collect();
        let (bytes, offsets) = corpus(&slices);
        let col = compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap();
        let view = col.decompress_view();
        assert_eq!(
            build_views(&view),
            oracle(&view),
            "tail={tail}: optimized build_views must match the scalar oracle"
        );
    }
}

#[test]
fn build_views_tiny_buffer_all_inline() {
    // values.len() < 16 → checked_sub yields None → every row is scalar tail.
    let rows: &[&[u8]] = &[b"ab", b"cde"];
    let (bytes, offsets) = corpus(rows);
    let col = compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap();
    let view = col.decompress_view();
    assert!(view.values.len() < 16);
    assert_eq!(build_views(&view), oracle(&view));
}

#[test]
#[should_panic(expected = "inline view exceeds INLINE_LEN")]
fn inline_rejects_oversized() {
    let _ = BinaryView::inline(b"this is far more than twelve bytes");
}

#[test]
#[should_panic(expected = "reference view must exceed INLINE_LEN")]
fn reference_rejects_inline_length() {
    let _ = BinaryView::reference(4, *b"abcd", 0, 0);
}
