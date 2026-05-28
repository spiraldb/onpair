#![allow(missing_docs)]

use criterion::{Criterion, criterion_group, criterion_main};

fn smoke(c: &mut Criterion) {
    c.bench_function("noop", |b| b.iter(|| 1u64.wrapping_add(1)));
}

criterion_group!(benches, smoke);
criterion_main!(benches);
