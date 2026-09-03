// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::core::offset::Offset;

/// A random-access sequence of byte rows used for training and encoding.
pub trait Rows {
    /// Number of rows.
    fn num_rows(&self) -> usize;

    /// Total bytes across every row.
    fn total_bytes(&self) -> usize;

    /// Return row `i` as `(window, len)`. The row is `window[..len]`; remaining
    /// bytes are optional lookahead for the matcher.
    fn row(&self, i: usize) -> (&[u8], usize);
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct ArrowRows<'a, O: Offset> {
    bytes: &'a [u8],
    offsets: &'a [O],
}

impl<'a, O: Offset> ArrowRows<'a, O> {
    #[inline]
    pub(crate) fn new(bytes: &'a [u8], offsets: &'a [O]) -> Self {
        Self { bytes, offsets }
    }
}

impl<O: Offset> Rows for ArrowRows<'_, O> {
    #[inline]
    fn num_rows(&self) -> usize {
        self.offsets.len() - 1
    }

    #[inline]
    fn total_bytes(&self) -> usize {
        self.offsets[self.offsets.len() - 1].to_usize()
    }

    #[inline]
    fn row(&self, i: usize) -> (&[u8], usize) {
        let start = self.offsets[i].to_usize();
        let end = self.offsets[i + 1].to_usize();
        (&self.bytes[start..], end - start)
    }
}

impl<T: AsRef<[u8]>> Rows for [T] {
    #[inline]
    fn num_rows(&self) -> usize {
        self.len()
    }

    #[inline]
    fn total_bytes(&self) -> usize {
        self.iter().map(|r| r.as_ref().len()).sum()
    }

    #[inline]
    fn row(&self, i: usize) -> (&[u8], usize) {
        let row = self[i].as_ref();
        (row, row.len())
    }
}

impl<R: Rows + ?Sized> Rows for &R {
    #[inline]
    fn num_rows(&self) -> usize {
        (**self).num_rows()
    }

    #[inline]
    fn total_bytes(&self) -> usize {
        (**self).total_bytes()
    }

    #[inline]
    fn row(&self, i: usize) -> (&[u8], usize) {
        (**self).row(i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrow_rows_include_lookahead() {
        let rows = ArrowRows::new(b"alphabeta", &[0u32, 5, 9]);
        assert_eq!(rows.num_rows(), 2);
        assert_eq!(rows.total_bytes(), 9);
        assert_eq!(rows.row(0), (b"alphabeta".as_slice(), 5));
        assert_eq!(rows.row(1), (b"beta".as_slice(), 4));
    }

    #[test]
    fn slice_rows_have_no_lookahead() {
        let rows: &[&[u8]] = &[b"alpha", b"", b"beta"];
        assert_eq!(rows.num_rows(), 3);
        assert_eq!(rows.total_bytes(), 9);
        for i in 0..rows.num_rows() {
            let (window, len) = rows.row(i);
            assert_eq!(window.len(), len);
        }
    }

    #[test]
    fn owned_rows_are_supported() {
        let rows = [b"one".to_vec(), b"two".to_vec()];
        assert_eq!(rows.as_slice().row(1).0, b"two");
    }
}
