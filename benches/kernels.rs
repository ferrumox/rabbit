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
    dot_i8i8_avx512vnni, matmul_i4_avx2, matmul_i4_avx512, matmul_mxfp4_avx512,
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

/// K3 routed-expert MXFP4 matmul (`matmul_mxfp4`) at the real per-expert dims: gate/up projections
/// are `[moe_inter=3072, moe_hidden=3584]` (i=3584, o=3072), down is `[3584, 3072]` (i=3072,
/// o=3584). Benched at s=1 (decode) and s=8 (a small batch), the before/after instrument for
/// Phase 2's scalar block restructure and Phase 3's AVX-512 tier. In Phase 2 this measures the
/// single scalar tier (`matmul_mxfp4` is scalar); Phase 3 splits it into scalar/avx512 entries.
fn bench_matmul_mxfp4(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_mxfp4");
    for &(name, i, o) in &[("gate_up_i3584_o3072", 3584usize, 3072usize), ("down_i3072_o3584", 3072usize, 3584usize)] {
        let data = random_bytes(o * i.div_ceil(2), 21);
        let scale = random_bytes(o * i.div_ceil(32), 22);
        for &s in &[1usize, 8] {
            let x: Vec<f32> = random_i8(s * i, 23).iter().map(|&v| v as f32).collect();
            let mut y = vec![0f32; s * o];
            group.bench_function(format!("scalar/{name}/s{s}"), |b| {
                b.iter(|| matmul_mxfp4_scalar(black_box(&mut y), black_box(&x), black_box(&data), black_box(&scale), s, i, o))
            });
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                group.bench_function(format!("avx512/{name}/s{s}"), |b| {
                    b.iter(|| unsafe { matmul_mxfp4_avx512(black_box(&mut y), black_box(&x), black_box(&data), black_box(&scale), s, i, o) })
                });
            }
        }
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
