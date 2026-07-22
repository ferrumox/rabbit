//! `cargo bench` — scalar vs AVX2 vs AVX-512/VNNI for the IDOT dot-product kernels (the
//! integer path `matmul_qt` uses for every int8 tensor, and int4 at `S>=2`), matching Fase
//! 7's plan: "cargo bench comparando escalar vs AVX2 vs AVX-512/VNNI".
//!
//! Uses the `pub` tier-specific functions (`dot_i8i8_scalar`/`dot_i8i8_avx2`/
//! `dot_i8i8_avx512vnni`, same for `dot_i4i8_*`) directly — the auto-dispatching
//! `dot_i8i8`/`dot_i4i8` always pick the fastest tier this CPU has, which would make a
//! same-machine scalar-vs-AVX2 comparison impossible.
//!
//! `I=4096` matches a plausible real hidden/intermediate dimension (GLM-5.2's `hidden_size`
//! is 12288-ish territory; 4096 sits in the same order of magnitude without skewing the
//! benchmark toward an unrealistically huge single dot product).

use criterion::{Criterion, criterion_group, criterion_main};
use rabbit::kernels::{dot_i4i8_scalar, dot_i8i8_scalar, matmul_i4_scalar};
#[cfg(target_arch = "x86_64")]
use rabbit::kernels::{dot_i4i8_avx2, dot_i4i8_avx512vnni, dot_i8i8_avx2, dot_i8i8_avx512vnni, matmul_i4_avx2, matmul_i4_avx512};
use std::hint::black_box;

const I: usize = 4096;
const O: usize = 4096;

fn xorshift(seed: &mut u32) -> u32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    *seed
}

fn random_i8(n: usize, seed: u32) -> Vec<i8> {
    let mut s = seed;
    (0..n).map(|_| (xorshift(&mut s) % 256) as i8).collect()
}

fn random_bytes(n: usize, seed: u32) -> Vec<u8> {
    let mut s = seed;
    (0..n).map(|_| (xorshift(&mut s) % 256) as u8).collect()
}

fn bench_dot_i8i8(c: &mut Criterion) {
    let w = random_i8(I, 1);
    let x = random_i8(I, 2);
    let mut group = c.benchmark_group("dot_i8i8");

    group.bench_function("scalar", |b| b.iter(|| dot_i8i8_scalar(black_box(&w), black_box(&x), I)));

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        group.bench_function("avx2", |b| b.iter(|| unsafe { dot_i8i8_avx2(black_box(&w), black_box(&x), I) }));
    }

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512bw") {
        group.bench_function("avx512-vnni", |b| b.iter(|| unsafe { dot_i8i8_avx512vnni(black_box(&w), black_box(&x), I) }));
    }

    group.finish();
}

fn bench_dot_i4i8(c: &mut Criterion) {
    let w4 = random_bytes(I.div_ceil(2), 3);
    let x = random_i8(I, 4);
    let mut group = c.benchmark_group("dot_i4i8");

    group.bench_function("scalar", |b| b.iter(|| dot_i4i8_scalar(black_box(&w4), black_box(&x), I)));

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        group.bench_function("avx2", |b| b.iter(|| unsafe { dot_i4i8_avx2(black_box(&w4), black_box(&x), I) }));
    }

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512bw") {
        group.bench_function("avx512-vnni", |b| b.iter(|| unsafe { dot_i4i8_avx512vnni(black_box(&w4), black_box(&x), I) }));
    }

    group.finish();
}

/// `matmul_i4` (float-weight int4 dequant-and-FMA path) is the one float-weight matmul with an
/// AVX-512 tier at all (colibrì's `I4_ACC512`) — see `kernels.rs`'s module doc. Benched at the
/// full `matmul_i4_*` level (not a single dot product like the IDOT benches above) since that
/// tier also parallelizes over `O` rows via `rayon`, which is part of its real-world cost/win,
/// not just the inner dot product.
fn bench_matmul_i4(c: &mut Criterion) {
    let x = random_i8(I, 5).iter().map(|&v| v as f32).collect::<Vec<f32>>();
    let q4 = random_bytes(O * I.div_ceil(2), 6);
    let scale: Vec<f32> = (0..O).map(|o| 1.0 + (o % 7) as f32 * 0.1).collect();
    let mut y = vec![0f32; O];
    let mut group = c.benchmark_group("matmul_i4");

    group.bench_function("scalar", |b| {
        b.iter(|| matmul_i4_scalar(black_box(&mut y), black_box(&x), black_box(&q4), black_box(&scale), 1, I, O))
    });

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        group.bench_function("avx2", |b| {
            b.iter(|| unsafe { matmul_i4_avx2(black_box(&mut y), black_box(&x), black_box(&q4), black_box(&scale), 1, I, O) })
        });
    }

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
        group.bench_function("avx512", |b| {
            b.iter(|| unsafe { matmul_i4_avx512(black_box(&mut y), black_box(&x), black_box(&q4), black_box(&scale), 1, I, O) })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_dot_i8i8, bench_dot_i4i8, bench_matmul_i4);
criterion_main!(benches);
