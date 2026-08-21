//! Single-core read-bandwidth ceiling for the code buffer.

use std::hint::black_box;
use std::time::Instant;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn read_avx512(values: &[u16]) -> u64 {
    use core::arch::x86_64::{__m512i, _mm512_loadu_si512, _mm512_storeu_si512, _mm512_xor_si512};

    let zero: __m512i = unsafe { core::mem::zeroed() };
    let mut acc = [zero; 8];
    let chunks = values.len() / 256;
    let base = values.as_ptr();
    for chunk in 0..chunks {
        for (lane, slot) in acc.iter_mut().enumerate() {
            let loaded =
                unsafe { _mm512_loadu_si512(base.add(chunk * 256 + lane * 32).cast::<__m512i>()) };
            *slot = _mm512_xor_si512(*slot, loaded);
        }
    }
    let mut folded = acc[0];
    for value in &acc[1..] {
        folded = _mm512_xor_si512(folded, *value);
    }
    let mut words = [0u64; 8];
    unsafe { _mm512_storeu_si512(words.as_mut_ptr().cast::<__m512i>(), folded) };
    let mut checksum = words.into_iter().fold(0, u64::wrapping_add);
    for &value in &values[chunks * 256..] {
        checksum = checksum.wrapping_add(value as u64);
    }
    checksum
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: mem_bw <codes.u16>");
    let bytes = std::fs::read(path).unwrap();
    let values: &[u16] =
        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<u16>(), bytes.len() / 2) };
    assert!(std::is_x86_feature_detected!("avx512f"));
    let reps = 100;
    let mut best = f64::INFINITY;
    let mut sum = 0u64;
    for _ in 0..reps {
        let start = Instant::now();
        sum ^= unsafe { read_avx512(black_box(values)) };
        best = best.min(start.elapsed().as_secs_f64());
    }
    black_box(sum);
    println!(
        "{} bytes, {:.3} ms, {:.2} GB/s",
        bytes.len(),
        best * 1e3,
        bytes.len() as f64 / best / 1e9,
    );
}
