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

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use rayon::prelude::*;
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

/// One real `qt_matvec_rows` call's worth of work (128 sequential `dot_i4_f32_avx512` calls,
/// `vh=128` at K3's real dims) against a REAL-scale `kv_b`-shaped matrix (`h*(qk_nope+vh)=24576`
/// rows × `kv_lora.div_ceil(2)=256` bytes/row ≈ 6.29MiB — comparable to this machine's L3, unlike
/// the plain `qt_matvec_rows_i4` bench above which reuses one 256-byte row 100M+ times). A real
/// `perf record` on the real checkpoint found `dot_i4_f32_avx512` responsible for ~27% of total
/// cycles despite doing only ~0.3% of `matmul_mxfp4`'s total arithmetic — the warm single-row
/// bench above (22.85ns/call) predicts ~6.7ms/token total, wildly below what that cycle share
/// implies. Hypothesis: `kv_b` would fit comfortably in L3 on its own, but the MASSIVE MoE expert
/// traffic interleaved with it (many MB per layer, every layer) keeps evicting it before its next
/// reuse — the same cold-cache story as `matmul_mxfp4_k3_dims_cold`, but for a smaller structure
/// that would otherwise be cache-friendly. `warm` reuses the same 128-row window every iteration
/// (like the plain bench, just batched to 128 calls); `cold` evicts the whole matrix (a 64MiB
/// scratch read) before every timed batch, rotating which 128-row window is read to avoid the
/// eviction pass itself re-warming the same rows.
fn bench_qt_matvec_rows_i4_real_scale(c: &mut Criterion) {
    const ROWS: usize = 24576;
    const ROW_BYTES: usize = KV_LORA.div_ceil(2);
    const VH: usize = 128;
    let kv_b = random_bytes(ROWS * ROW_BYTES, 26);
    let x = random_i8(KV_LORA, 27).iter().map(|&v| v as f32).collect::<Vec<f32>>();
    let evict = random_bytes(64 * 1024 * 1024, 28);

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
        let mut group = c.benchmark_group("qt_matvec_rows_i4_real_scale");
        group.sample_size(30);

        group.bench_function("warm", |b| {
            b.iter(|| {
                let mut acc = 0f32;
                for r in 0..VH {
                    let row = &kv_b[r * ROW_BYTES..(r + 1) * ROW_BYTES];
                    acc += unsafe { dot_i4_f32_avx512(black_box(row), black_box(&x), KV_LORA) };
                }
                black_box(acc)
            })
        });

        let mut offset = 0usize;
        group.bench_function("cold", |b| {
            b.iter_batched(
                || {
                    let sum: u64 = evict.iter().step_by(64).map(|&v| v as u64).sum();
                    black_box(sum);
                    offset = (offset + VH) % (ROWS - VH);
                    offset
                },
                |off| {
                    let mut acc = 0f32;
                    for r in 0..VH {
                        let row = &kv_b[(off + r) * ROW_BYTES..(off + r + 1) * ROW_BYTES];
                        acc += unsafe { dot_i4_f32_avx512(black_box(row), black_box(&x), KV_LORA) };
                    }
                    black_box(acc)
                },
                BatchSize::PerIteration,
            )
        });
        group.finish();
    }
}

/// `matmul_mxfp4` at K3's REAL routed-expert dims (the real checkpoint's `text_config`:
/// `routed_expert_hidden_size=3584`, `moe_intermediate_size=3072` — every routed expert's
/// `w1`/`w3` is `[3072,3584]`, `w2` is `[3072,3584]` too once transposed for the down
/// direction, same MAC count either way), not the generic `I=O=4096` above — lets a real
/// per-token compute-time estimate (`n_calls * this` vs. the measured `expert_matmul_s` phase
/// in `PERFORMANCE_KIMI_K3.md`) be computed directly instead of scaled/guessed.
fn bench_matmul_mxfp4_k3_dims(c: &mut Criterion) {
    const KI: usize = 3584;
    const KO: usize = 3072;
    let x = random_i8(KI, 14).iter().map(|&v| v as f32).collect::<Vec<f32>>();
    let data = random_bytes(KO * KI.div_ceil(2), 15);
    let bpr = KI.div_ceil(32);
    let block_scale: Vec<u8> = (0..KO * bpr).map(|k| 120 + (k % 15) as u8).collect();
    let mut y = vec![0f32; KO];
    let mut group = c.benchmark_group("matmul_mxfp4_k3_dims");

    group.bench_function("scalar", |b| {
        b.iter(|| matmul_mxfp4_scalar(black_box(&mut y), black_box(&x), black_box(&data), black_box(&block_scale), 1, KI, KO))
    });

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        group.bench_function("avx2", |b| {
            b.iter(|| unsafe {
                matmul_mxfp4_avx2(black_box(&mut y), black_box(&x), black_box(&data), black_box(&block_scale), 1, KI, KO)
            })
        });
    }

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
        group.bench_function("avx512", |b| {
            b.iter(|| unsafe {
                matmul_mxfp4_avx512(black_box(&mut y), black_box(&x), black_box(&data), black_box(&block_scale), 1, KI, KO)
            })
        });
    }

    group.finish();
}

/// Same call as `bench_matmul_mxfp4_k3_dims`'s `avx512` case, but with `data`/`block_scale`/`x`/
/// `y` forced COLD in cache before every timed call, via `iter_batched`'s untimed setup reading a
/// 64MiB scratch buffer (bigger than this machine's combined L2+L3, ~36MiB) to evict everything
/// else. The plain bench above reuses the same buffers across 100+ back-to-back iterations —
/// warm in L1/L2 after the first pass — but a REAL expert's bytes just arrived off disk via
/// `io_uring` and are genuinely cold the first (and only) time the matmul touches them per token.
/// Exists to directly test a real, still-open hypothesis from `PERFORMANCE_KIMI_K3.md` ("Where
/// does compute actually go?"): that the measured real-checkpoint compute time
/// (~751µs/call average) is dominated by cold-cache DRAM fill, not FLOPs — if so, this bench
/// should land close to that real number instead of the warm bench's 222µs.
fn bench_matmul_mxfp4_k3_dims_cold(c: &mut Criterion) {
    const KI: usize = 3584;
    const KO: usize = 3072;
    let x = random_i8(KI, 16).iter().map(|&v| v as f32).collect::<Vec<f32>>();
    let data = random_bytes(KO * KI.div_ceil(2), 17);
    let bpr = KI.div_ceil(32);
    let block_scale: Vec<u8> = (0..KO * bpr).map(|k| 120 + (k % 15) as u8).collect();
    let mut y = vec![0f32; KO];
    let evict = random_bytes(64 * 1024 * 1024, 18);

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
        let mut group = c.benchmark_group("matmul_mxfp4_k3_dims_cold");
        group.sample_size(30);
        group.bench_function("avx512", |b| {
            b.iter_batched(
                || {
                    let sum: u64 = evict.iter().step_by(64).map(|&v| v as u64).sum();
                    black_box(sum);
                },
                |_| unsafe {
                    matmul_mxfp4_avx512(black_box(&mut y), black_box(&x), black_box(&data), black_box(&block_scale), 1, KI, KO)
                },
                BatchSize::PerIteration,
            )
        });
        group.finish();
    }
}

/// One cache line (64B) per `_mm_prefetch(_MM_HINT_T0)` — issuing a prefetch for every line of
/// `buf` doesn't wait for any of them to land; it just tells the memory subsystem to start
/// pulling them in while the CPU goes on to do something else, which is the whole point of the
/// bench below.
#[cfg(target_arch = "x86_64")]
unsafe fn prefetch_all(buf: &[u8]) {
    use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
    let mut i = 0;
    while i < buf.len() {
        unsafe { _mm_prefetch(buf.as_ptr().add(i) as *const i8, _MM_HINT_T0) };
        i += 64;
    }
}

/// Tests the ceiling of the "software prefetch" idea from `PERFORMANCE_KIMI_K3.md`'s "Where does
/// compute actually go?": if an expert's `data`/`block_scale` are prefetched as soon as they're
/// known (e.g. right when its `io_uring` read lands) and something else useful runs while the
/// prefetch is in flight (e.g. another expert's matmul — simulated here by `filler`, a real
/// `matmul_mxfp4_avx512` call against an already-warm buffer of the same size), does the TIMED
/// call recover most of the cold penalty measured by the `_cold` bench above (470.7µs), landing
/// back near the warm bench's 222.2µs? If yes, prefetch-ahead is worth building into
/// `expert_cache.rs`'s real dispatch; if the timed call is still close to 470.7µs, memory
/// bandwidth (not latency) is the real limit and prefetching earlier wouldn't help — the
/// `filler` call competes for the same bandwidth as the real hardware would while a prefetch is
/// in flight, so this isn't just an optimistic best case.
fn bench_matmul_mxfp4_k3_dims_prefetched(c: &mut Criterion) {
    const KI: usize = 3584;
    const KO: usize = 3072;
    let x = random_i8(KI, 19).iter().map(|&v| v as f32).collect::<Vec<f32>>();
    let data = random_bytes(KO * KI.div_ceil(2), 20);
    let bpr = KI.div_ceil(32);
    let block_scale: Vec<u8> = (0..KO * bpr).map(|k| 120 + (k % 15) as u8).collect();
    let mut y = vec![0f32; KO];
    let evict = random_bytes(64 * 1024 * 1024, 21);

    // The "other expert" whose compute is meant to overlap with this iteration's prefetch —
    // reused every iteration (deliberately kept warm, like a just-computed neighbor would be
    // partway through the same batch), so its own cost stays constant and doesn't contaminate
    // the measurement of how well prefetch hid the TARGET buffer's cold-cache penalty.
    let filler_x = random_i8(KI, 22).iter().map(|&v| v as f32).collect::<Vec<f32>>();
    let filler_data = random_bytes(KO * KI.div_ceil(2), 23);
    let filler_scale: Vec<u8> = (0..KO * bpr).map(|k| 120 + (k % 15) as u8).collect();
    let mut filler_y = vec![0f32; KO];

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
        let mut group = c.benchmark_group("matmul_mxfp4_k3_dims_prefetched");
        group.sample_size(30);
        group.bench_function("avx512", |b| {
            b.iter_batched(
                || {
                    let sum: u64 = evict.iter().step_by(64).map(|&v| v as u64).sum();
                    black_box(sum);
                    unsafe {
                        prefetch_all(&data);
                        prefetch_all(&block_scale);
                        matmul_mxfp4_avx512(
                            black_box(&mut filler_y),
                            black_box(&filler_x),
                            black_box(&filler_data),
                            black_box(&filler_scale),
                            1,
                            KI,
                            KO,
                        );
                    }
                },
                |_| unsafe {
                    matmul_mxfp4_avx512(black_box(&mut y), black_box(&x), black_box(&data), black_box(&block_scale), 1, KI, KO)
                },
                BatchSize::PerIteration,
            )
        });
        group.finish();
    }
}

/// Isolates "just touch these cold bytes" from "touch them AND dequant+FMA them": same eviction
/// technique and same total byte volume (`data`+`block_scale`, one call's worth) as
/// `bench_matmul_mxfp4_k3_dims_cold`, but the timed routine only sums bytes (`rayon`
/// `par_chunks`, same pool) instead of calling the real kernel. If this lands close to the real
/// kernel's 470.7µs, the cold-read cost itself (not the dequant math) explains almost all of
/// it — meaning the earlier ~5x gap against the 1GiB raw-bandwidth probe
/// (`examples/mem_bandwidth_probe.rs`) is apples-to-oranges (many small ~5.85MB bursts don't
/// reach the same steady-state throughput a single sustained 1GiB sweep does), not a fixable
/// inefficiency. If this is meaningfully FASTER than 470.7µs, the dequant math itself is adding
/// real extra cost beyond the unavoidable cold-read floor, which WOULD be worth optimizing.
fn bench_mxfp4_cold_readonly(c: &mut Criterion) {
    const KI: usize = 3584;
    const KO: usize = 3072;
    let data = random_bytes(KO * KI.div_ceil(2), 24);
    let bpr = KI.div_ceil(32);
    let block_scale: Vec<u8> = (0..KO * bpr).map(|k| 120 + (k % 15) as u8).collect();
    let evict = random_bytes(64 * 1024 * 1024, 25);

    let mut group = c.benchmark_group("mxfp4_cold_readonly");
    group.sample_size(30);
    group.bench_function("rayon_sum", |b| {
        b.iter_batched(
            || {
                let sum: u64 = evict.iter().step_by(64).map(|&v| v as u64).sum();
                black_box(sum);
            },
            |_| {
                let s1: u64 = data.par_chunks(4096).map(|c| c.iter().step_by(64).map(|&v| v as u64).sum::<u64>()).sum();
                let s2: u64 = block_scale.par_chunks(4096).map(|c| c.iter().step_by(64).map(|&v| v as u64).sum::<u64>()).sum();
                black_box(s1 + s2)
            },
            BatchSize::PerIteration,
        )
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_dot_i8i8,
    bench_dot_i4i8,
    bench_matmul_i4,
    bench_matmul_mxfp4,
    bench_matmul_mxfp4_k3_dims,
    bench_matmul_mxfp4_k3_dims_cold,
    bench_matmul_mxfp4_k3_dims_prefetched,
    bench_mxfp4_cold_readonly,
    bench_qt_addrow_i4,
    bench_qt_matvec_rows_i4,
    bench_qt_matvec_rows_i4_real_scale,
    bench_mla_score_and_vmix
);
criterion_main!(benches);
