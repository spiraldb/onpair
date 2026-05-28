//! Mock vortex-onpair-rs. Passthrough impl: payload is stored verbatim, offsets
//! identify row boundaries, decompress returns the exact input bytes.
//!
//! API shape is the contract rust-bench codes against; replace with the real
//! crate by switching the path dep in `rust-bench/Cargo.toml` to a registry
//! version that exposes the same `Column` interface.

pub struct Column {
    payload: Vec<u8>,
    offsets: Vec<u32>,
    bits: u32,
}

impl Column {
    pub fn compress(bits: u32, payload: &[u8], offsets: &[u32]) -> Self {
        Self {
            payload: payload.to_vec(),
            offsets: offsets.to_vec(),
            bits,
        }
    }

    pub fn bits(&self) -> u32 {
        self.bits
    }

    pub fn num_rows(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub fn dict_size(&self) -> usize {
        0
    }

    pub fn dict_bytes(&self) -> usize {
        0
    }

    pub fn codes_bytes(&self) -> usize {
        self.payload.len()
    }

    pub fn compressed_bytes(&self) -> usize {
        self.payload.len() + self.offsets.len() * std::mem::size_of::<u32>()
    }

    pub fn decompress_row(&self, idx: usize, out: &mut Vec<u8>) {
        let start = self.offsets[idx] as usize;
        let end = self.offsets[idx + 1] as usize;
        out.clear();
        out.extend_from_slice(&self.payload[start..end]);
    }
}
