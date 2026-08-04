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
use rabbit::kernels::{axpy_i4_f32_scalar, dot_i4_f32_scalar, dot_i4i8_scalar, dot_i8i8_scalar, matmul_i4_scalar, matmul_mxfp4_scalar};
#[cfg(target_arch = "x86_64")]
use rabbit::kernels::{
    axpy_f32_avx2, axpy_i4_f32_avx512, dot_f32_avx2, dot_i4_f32_avx512, dot_i4i8_avx2, dot_i4i8_avx512vnni, dot_i8i8_avx2,
    dot_i8i8_avx512vnni, matmul_i4_avx2, matmul_i4_avx512, matmul_mxfp4_avx2, matmul_mxfp4_avx512,
};
use std::hint::black_box;

const I: usize = 4096;
const O: usize = 4096;

/// GLM-5.2's `kv_lora_rank` — the dimension the MLA-absorb helpers (`qt_addrow`/
/// `qt_matvec_rows`'s int4 branch, and the score/value-mix reductions) actually run at in the
/// real decode path, per `rabbit-plan.md`/`attention.rs`'s own `kv_lora <= 512` gate.
const KV_LORA: usize = 512;

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

/// `qt_addrow`'s int4 branch (MLA-absorb `q_nope` absorption) at the real `kv_lora=512` this
/// runs at in `attention.rs`'s absorbed decode path — see that module's doc for why this and
/// `qt_matvec_rows` are fixed O(kv_lora) per call, independent of context length.
fn bench_qt_addrow_i4(c: &mut Criterion) {
    let w = random_bytes(KV_LORA.div_ceil(2), 7);
    let coef = 1.7f32;
    let mut group = c.benchmark_group("qt_addrow_i4");

    group.bench_function("scalar", |b| {
        let mut acc = vec![0f32; KV_LORA];
        b.iter(|| axpy_i4_f32_scalar(black_box(&w), black_box(coef), black_box(&mut acc), KV_LORA))
    });

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
        group.bench_function("avx512", |b| {
            let mut acc = vec![0f32; KV_LORA];
            b.iter(|| unsafe { axpy_i4_f32_avx512(black_box(&w), black_box(coef), black_box(&mut acc), KV_LORA) })
        });
    }

    group.finish();
}

/// `qt_matvec_rows`'s int4 branch (MLA-absorb value projection) at `kv_lora=512`.
fn bench_qt_matvec_rows_i4(c: &mut Criterion) {
    let w = random_bytes(KV_LORA.div_ceil(2), 8);
    let x = random_i8(KV_LORA, 9).iter().map(|&v| v as f32).collect::<Vec<f32>>();
    let mut group = c.benchmark_group("qt_matvec_rows_i4");

    group.bench_function("scalar", |b| b.iter(|| dot_i4_f32_scalar(black_box(&w), black_box(&x), KV_LORA)));

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
        group.bench_function("avx512", |b| {
            b.iter(|| unsafe { dot_i4_f32_avx512(black_box(&w), black_box(&x), KV_LORA) })
        });
    }

    group.finish();
}

/// The MLA-absorb score dot-product (`dot(qabs, Lt)`) and value-mix axpy
/// (`clat[i] += sc[jj]*Lt[i]`) at `kv_lora=512` — both operate on already-dequantized f32
/// latents, so unlike the two benches above these run regardless of the KV weight's quant
/// format. See `kernels.rs`'s `dot_f32_avx2`/`axpy_f32_avx2` docs for the bit-identity discussion.
fn bench_mla_score_and_vmix(c: &mut Criterion) {
    let qabs = random_i8(KV_LORA, 10).iter().map(|&v| v as f32).collect::<Vec<f32>>();
    let lt = random_i8(KV_LORA, 11).iter().map(|&v| v as f32).collect::<Vec<f32>>();
    let coef = 0.3f32;

    let mut group = c.benchmark_group("mla_score_dot");
    group.bench_function("scalar", |b| {
        b.iter(|| black_box(&qabs).iter().zip(black_box(&lt)).map(|(&x, &y)| x * y).sum::<f32>())
    });
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        group.bench_function("avx2", |b| b.iter(|| unsafe { dot_f32_avx2(black_box(&qabs), black_box(&lt), KV_LORA) }));
    }
    group.finish();

    let mut group = c.benchmark_group("mla_vmix_axpy");
    group.bench_function("scalar", |b| {
        let mut clat = vec![0f32; KV_LORA];
        b.iter(|| {
            for i in 0..KV_LORA {
                clat[i] += coef * lt[i];
            }
            black_box(&clat);
        })
    });
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        group.bench_function("avx2", |b| {
            let mut clat = vec![0f32; KV_LORA];
            b.iter(|| unsafe { axpy_f32_avx2(black_box(coef), black_box(&lt), black_box(&mut clat), KV_LORA) })
        });
    }
    group.finish();
}

/// `matmul_mxfp4` (Kimi K3's routed-expert format on real disk) — same tier ladder and
/// dual-accumulator AVX-512F/BW structure as `matmul_i4` (32-element blocks map exactly onto
/// its 2x16-lane chunking), see `kernels.rs`'s module doc. `block_scale` bytes are clamped to a
/// sane E8M0 range (not `0xFF`'s reserved NaN code, not extreme exponents) — real values only
/// ever come from `choose_block_scale`, and arbitrary/extreme bytes would risk skewing the
/// timing with denormal/overflow float handling unrelated to the kernel's actual cost.
fn bench_matmul_mxfp4(c: &mut Criterion) {
    let x = random_i8(I, 12).iter().map(|&v| v as f32).collect::<Vec<f32>>();
    let data = random_bytes(O * I.div_ceil(2), 13);
    let bpr = I.div_ceil(32);
    let block_scale: Vec<u8> = (0..O * bpr).map(|k| 120 + (k % 15) as u8).collect();
    let mut y = vec![0f32; O];
    let mut group = c.benchmark_group("matmul_mxfp4");

    group.bench_function("scalar", |b| {
        b.iter(|| matmul_mxfp4_scalar(black_box(&mut y), black_box(&x), black_box(&data), black_box(&block_scale), 1, I, O))
    });

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        group.bench_function("avx2", |b| {
            b.iter(|| unsafe {
                matmul_mxfp4_avx2(black_box(&mut y), black_box(&x), black_box(&data), black_box(&block_scale), 1, I, O)
            })
        });
    }

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
        group.bench_function("avx512", |b| {
            b.iter(|| unsafe {
                matmul_mxfp4_avx512(black_box(&mut y), black_box(&x), black_box(&data), black_box(&block_scale), 1, I, O)
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_dot_i8i8,
    bench_dot_i4i8,
    bench_matmul_i4,
    bench_matmul_mxfp4,
    bench_qt_addrow_i4,
    bench_qt_matvec_rows_i4,
    bench_mla_score_and_vmix
);
criterion_main!(benches);
