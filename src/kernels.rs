//! Port of `glm.c`'s matmul family. Fase 3 built the scalar baseline; this phase adds AVX2 and
//! AVX-512/VNNI tiers on top, selected at runtime (`is_x86_feature_detected!`) — same idea as
//! the C's `g_i4s`/`IDOT_KERNEL` picking a kernel by measured hardware, not compile-time. Every
//! `*_scalar` function is the untouched Fase 3 implementation; the public names
//! (`matmul_q`/`matmul_i4`/`matmul_i2`/`dot_i8i8`/`dot_i4i8`) are now dispatchers.
//!
//! Tier ladder: `matmul_q`/`matmul_i2` (float-weight dequant-and-FMA path) get scalar/AVX2 only
//! — colibrì never added an AVX-512 tier for those. `matmul_i4` is the one exception: colibrì's
//! later `I4_ACC512` (`dot_i4f_avx512`) adds an AVX-512F/BW tier specifically for int4, two
//! independent `__m512` FMA chains reduced via a single `_mm512_reduce_add_ps` tree-sum at the
//! end instead of AVX2's running accumulator — same lossless nibble-unpack math, genuinely
//! *less* rounding error than the AVX2/scalar order (colibrì measured 2-6x lower max relative
//! error vs the scalar oracle), but for that same reason NOT bit-identical to them — parity
//! tests for `matmul_i4`'s tiers are within-tolerance, not bit-exact, same as AVX2-vs-scalar
//! already was for this function. `matmul_mxfp4` (K3's routed-expert MXFP4 path, added for the
//! target-box work — `K3_OPTIMIZE_BRIEF.md`) follows the same split: a bit-exact scalar tier
//! (`matmul_mxfp4_scalar`) and an AVX-512F/BW tier (`matmul_mxfp4_avx512` — a `permutexvar` E2M1
//! decode with per-block E8M0 scale-fold and dual-chain FMA) that is within-tolerance, not
//! bit-identical, for the same per-row reassociation reason. `dot_i8i8`/`dot_i4i8` (the integer IDOT path) get
//! scalar/AVX2/AVX-512-VNNI: pure integer accumulation, so unlike either float path there's no
//! reassociation to worry about — every tier must agree bit-for-bit there, which is exactly
//! what this module's parity tests check.
//!
//! `y[S,O] = x[S,I] @ W^T` throughout, `W` given in one of the `QT` formats from `quant.rs`.
//! The IDOT kernels additionally quantize activations to int8 per row (`qrow_i8`, scalar only
//! in the original — never vectorized there, so not here either) so the whole dot product
//! runs in integer arithmetic — the reference implementation measures this at ~2-3x over the float-weight path, at
//! ~0.3% added RMS error per matmul from the activation quantization.

use crate::quant::{QT, QTKind, e2m1_decode, e8m0_decode};
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Every `matmul_*` below parallelizes over output rows (`oi`, 0..O) with rayon — same axis
/// the reference implementation's `#pragma omp parallel for` picks for its C matmul kernels, and the natural one:
/// each `oi` reads the same `x`/activations but a disjoint row of the weight matrix, so there's
/// no cross-row dependency. The catch is `y`'s own layout (`y[si*O+oi]`, row-major by sequence
/// position): a fixed `oi` touches `S` elements strided by `O`, which safe Rust can't split into
/// disjoint mutable chunks across threads. `yt`'s `[O,S]` layout (row-major by output index)
/// makes each `oi`'s slice contiguous instead — exactly what `par_chunks_mut` needs — at the
/// cost of one sequential transpose back into `y` afterward. That transpose is O(S*O); the
/// matmul itself is O(S*O*I), so for any realistic `I` (hidden/intermediate dims in the
/// thousands) the transpose is noise.
///
/// Since Phase 5 v2 (`K3_OPTIMIZE_BRIEF.md`/`NUMA_AMX_BRIEF.md` N3) the fan-out is COARSENED:
/// rows go to the pool in [`matmul_chunk_rows`]-sized blocks via [`par_rows`], not one task per
/// row (see those docs — one-row tasks were measured as a scheduling floor, not a parallelism
/// win). Callers that already hold coarser independent work (K3's expert dispatch) skip the
/// internal fan-out entirely via the serial row-range entry point [`matmul_qt_rows`].
fn transpose_so(y: &mut [f32], yt: &[f32], s: usize, o: usize) {
    for oi in 0..o {
        for si in 0..s {
            y[si * o + oi] = yt[oi * s + si];
        }
    }
}

thread_local! {
    /// The `yt` transpose buffer every `matmul_*` below needs, reused across calls instead of
    /// freshly allocated each time — one of the reference implementation's own tricks (`_Thread_local` scratch
    /// buffers, `glm.c:458`) that a naive Rust port doesn't get for free just from adding
    /// `rayon`. Thread-local, not a shared pool: every call into a `matmul_*` function happens
    /// synchronously on the SAME calling thread (the generation loop never calls `matmul_qt`
    /// reentrantly, and rayon's own worker threads — spawned transiently inside one call — never
    /// call back into `matmul_qt` themselves), so a per-thread cell is enough to eliminate the
    /// realloc without needing any actual pooling/locking machinery.
    static YT_SCRATCH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Runs `f` against a `len`-element scratch buffer reused across calls on this thread — grows
/// on first use (or whenever a bigger call comes along) and never shrinks, so steady-state
/// generation (same shapes every layer/token) settles into zero further allocations after an
/// initial warm-up. Callers get a raw `&mut [f32]`, not the `RefCell` itself, and always start
/// from zero-filled — every existing caller already expected a fresh `vec![0f32; len]`.
///
/// Reentrancy: a `matmul_*` holds this thread-local borrow for the whole duration of its inner
/// `par_chunks_mut` region, during which rayon can work-steal ANOTHER `matmul_*` onto this same
/// thread (K3's Phase-5 across-expert parallelism nests an outer per-expert `par_iter` around the
/// matmuls' inner fan-out — the single-level-parallelism assumption the original `borrow_mut()`
/// encoded no longer holds). The inner (reentrant) call falls back to a one-shot heap buffer
/// instead of panicking on the already-held borrow. The common non-nested path (GLM/Kimi, and
/// K3's own non-matmul callers) still hits the zero-alloc thread-local fast path unchanged.
/// Output rows per rayon task for a matmul over `o` rows. `par_chunks_mut(s)` — one task per
/// output row — is the natural expression but a pathological schedule at batch-1 decode: it hands
/// rayon thousands of tasks worth one dot product each, and the pool spends more time splitting,
/// waking and joining than computing. Coarsening to a few tasks per pool thread keeps every
/// thread fed and work-stealing able to even out a ragged tail, while collapsing the split tree by
/// orders of magnitude. Bit-identity is unaffected: task granularity changes which thread computes
/// a row, never the row's own accumulation order.
///
/// Measured on the 384-thread target box (`PERFORMANCE.md`, Phase 5 v2): with one task per row,
/// K3 decode got *slower* the more threads it was given — 0.094 s/token at 48 threads against
/// 1.02 s/token at 384, i.e. the pool was a liability well before it was a resource. The laptop
/// this file was originally tuned on has 12 cores and never surfaced it.
fn matmul_chunk_rows(o: usize) -> usize {
    o.div_ceil((rayon::current_num_threads() * 4).max(1)).max(1)
}

/// Fans `row(oi, out)` — one output row's `s` values into its own `[S]` slice of the transposed
/// `yt` — across the pool in [`matmul_chunk_rows`]-sized blocks. Every `matmul_*` below shares
/// this rather than calling `par_chunks_mut` itself, so the granularity decision lives in exactly
/// one place.
fn par_rows(yt: &mut [f32], s: usize, o: usize, row: impl Fn(usize, &mut [f32]) + Sync + Send) {
    let cr = matmul_chunk_rows(o);
    yt.par_chunks_mut(s * cr).enumerate().for_each(|(ci, block)| {
        for (k, out) in block.chunks_mut(s).enumerate() {
            row(ci * cr + k, out);
        }
    });
}

fn with_yt_scratch<R>(len: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    YT_SCRATCH.with(|cell| {
        if let Ok(mut buf) = cell.try_borrow_mut() {
            if buf.len() < len {
                buf.resize(len, 0.0);
            } else {
                buf[..len].fill(0.0);
            }
            f(&mut buf[..len])
        } else {
            let mut buf = vec![0.0f32; len];
            f(&mut buf)
        }
    })
}

#[cfg(target_arch = "x86_64")]
fn has_avx2() -> bool {
    is_x86_feature_detected!("avx2")
}
#[cfg(not(target_arch = "x86_64"))]
fn has_avx2() -> bool {
    false
}

/// Gate for the MLA-absorption helpers' (`qt_addrow`/`qt_matvec_rows`) AVX-512 tier — same
/// feature pair `matmul_i4`'s dispatcher checks, factored out since both call sites live outside
/// `mod simd` and need the same two-flag check.
#[cfg(target_arch = "x86_64")]
fn has_avx512_i4() -> bool {
    is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
}
#[cfg(not(target_arch = "x86_64"))]
fn has_avx512_i4() -> bool {
    false
}

/// One output element of [`matmul`] — factored out of the inner loop so [`matmul_qt_rows`] can
/// compute a row block through the SAME code (and therefore the same accumulation order) rather
/// than a re-derived copy of it. Same reason the other `row_dot_*` helpers below exist; see
/// [`matmul_qt_rows`]'s doc for the bit-identity contract they collectively make possible.
#[inline]
fn row_dot_f32(wr: &[f32], xs: &[f32]) -> f32 {
    xs.iter().zip(wr).map(|(a, b)| a * b).sum()
}

/// y[S,O] = x[S,I] @ W^T, W[O,I] f32.
pub fn matmul(y: &mut [f32], x: &[f32], w: &[f32], s: usize, i: usize, o: usize) {
    with_yt_scratch(o * s, |yt| {
        par_rows(yt, s, o, |oi, out| {
            let wr = &w[oi * i..(oi + 1) * i];
            for (si, slot) in out.iter_mut().enumerate() {
                *slot = row_dot_f32(wr, &x[si * i..(si + 1) * i]);
            }
        });
        transpose_so(y, yt, s, o);
    });
}

/// y[S,O] = x[S,I] @ W^T, W int8[O,I] per-row scale (dequant-on-use). Dispatches to
/// AVX2/scalar; see the module doc for why there's no AVX-512 tier here.
pub fn matmul_q(y: &mut [f32], x: &[f32], q: &[i8], scale: &[f32], s: usize, i: usize, o: usize) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        return unsafe { matmul_q_avx2(y, x, q, scale, s, i, o) };
    }
    matmul_q_scalar(y, x, q, scale, s, i, o)
}

fn matmul_q_scalar(y: &mut [f32], x: &[f32], q: &[i8], scale: &[f32], s: usize, i: usize, o: usize) {
    with_yt_scratch(o * s, |yt| {
        par_rows(yt, s, o, |oi, out| {
            let w = &q[oi * i..(oi + 1) * i];
            let sc = scale[oi];
            for (si, slot) in out.iter_mut().enumerate() {
                let xs = &x[si * i..(si + 1) * i];
                let a: f32 = xs.iter().zip(w).map(|(&xv, &wv)| xv * wv as f32).sum();
                *slot = a * sc;
            }
        });
        transpose_so(y, yt, s, o);
    });
}

/// y[S,O] = x[S,I] @ W^T, W int4-packed[O,ceil(I/2)] (2 values/byte) per-row scale. Dispatches
/// AVX-512F/BW (`I4_ACC512`'s dual-accumulator kernel) > AVX2 > scalar — see the module doc for
/// why this is the one float-weight matmul with an AVX-512 tier at all.
pub fn matmul_i4(y: &mut [f32], x: &[f32], q4: &[u8], scale: &[f32], s: usize, i: usize, o: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            return unsafe { matmul_i4_avx512(y, x, q4, scale, s, i, o) };
        }
        if has_avx2() {
            return unsafe { matmul_i4_avx2(y, x, q4, scale, s, i, o) };
        }
    }
    matmul_i4_scalar(y, x, q4, scale, s, i, o)
}

/// The unscaled dot behind [`matmul_i4_scalar`]'s inner loop. NOT interchangeable with
/// [`dot_i4_f32_scalar`] despite computing the same mathematical quantity: this one adds each
/// packed byte's TWO products together before folding them into the running sum
/// (`a += xs[k]*lo + xs[k+1]*hi`), whereas `dot_i4_f32_scalar` — the reference
/// `qt_addrow`/`qt_matvec_rows` are written against — accumulates strictly one element at a time.
/// Different association, different bits. Keep them separate.
#[inline]
fn row_dot_i4_f32_pairs(w: &[u8], xs: &[f32], i: usize) -> f32 {
    let mut a = 0f32;
    let mut ii = 0;
    while ii + 1 < i {
        let byte = w[ii >> 1];
        let lo = (byte & 0xF) as i32 - 8;
        let hi = (byte >> 4) as i32 - 8;
        a += xs[ii] * lo as f32 + xs[ii + 1] * hi as f32;
        ii += 2;
    }
    if ii < i {
        let byte = w[ii >> 1];
        let lo = (byte & 0xF) as i32 - 8;
        a += xs[ii] * lo as f32;
    }
    a
}

/// `pub` so `benches/kernels.rs` can compare tiers directly — same reason `dot_i8i8_scalar` is.
pub fn matmul_i4_scalar(y: &mut [f32], x: &[f32], q4: &[u8], scale: &[f32], s: usize, i: usize, o: usize) {
    let rb = i.div_ceil(2);
    with_yt_scratch(o * s, |yt| {
        par_rows(yt, s, o, |oi, out| {
            let w = &q4[oi * rb..(oi + 1) * rb];
            let sc = scale[oi];
            for (si, slot) in out.iter_mut().enumerate() {
                *slot = row_dot_i4_f32_pairs(w, &x[si * i..(si + 1) * i], i) * sc;
            }
        });
        transpose_so(y, yt, s, o);
    });
}

/// y[S,O] = x[S,I] @ W^T, W int2-packed[O,ceil(I/4)] (4 values/byte) per-row scale.
pub fn matmul_i2(y: &mut [f32], x: &[f32], q2: &[u8], scale: &[f32], s: usize, i: usize, o: usize) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        return unsafe { matmul_i2_avx2(y, x, q2, scale, s, i, o) };
    }
    matmul_i2_scalar(y, x, q2, scale, s, i, o)
}

/// The unscaled dot behind [`matmul_i2_scalar`]'s inner loop — see [`row_dot_f32`] for why the
/// per-row bodies are factored out at all.
#[inline]
fn row_dot_i2_f32_scalar(w: &[u8], xs: &[f32], i: usize) -> f32 {
    let mut a = 0f32;
    for ii in 0..i {
        let byte = w[ii >> 2];
        let sh = (ii & 3) * 2;
        let v = ((byte >> sh) & 3) as i32 - 2;
        a += xs[ii] * v as f32;
    }
    a
}

fn matmul_i2_scalar(y: &mut [f32], x: &[f32], q2: &[u8], scale: &[f32], s: usize, i: usize, o: usize) {
    let rb = i.div_ceil(4);
    with_yt_scratch(o * s, |yt| {
        par_rows(yt, s, o, |oi, out| {
            let w = &q2[oi * rb..(oi + 1) * rb];
            let sc = scale[oi];
            for (si, slot) in out.iter_mut().enumerate() {
                *slot = row_dot_i2_f32_scalar(w, &x[si * i..(si + 1) * i], i) * sc;
            }
        });
        transpose_so(y, yt, s, o);
    });
}

/// y[S,O] = x[S,I] @ W^T, W in OCP-MX FP4 (`QTKind::MxFp4` — see `quant.rs`'s doc): 4-bit E2M1
/// elements, one E8M0 scale per 32-element block along the row (not per-row, unlike every other
/// format here). Dispatches AVX-512F/BW (`matmul_mxfp4_avx512`) > scalar, same ladder as
/// `matmul_i4` — see the module doc for why the AVX-512 tier is within-tolerance, not bit-exact.
pub fn matmul_mxfp4(y: &mut [f32], x: &[f32], data: &[u8], block_scale: &[u8], s: usize, i: usize, o: usize) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
        return unsafe { matmul_mxfp4_avx512(y, x, data, block_scale, s, i, o) };
    }
    matmul_mxfp4_scalar(y, x, data, block_scale, s, i, o)
}

/// Scalar tier of [`matmul_mxfp4`]. `pub` so `benches/kernels.rs` can compare tiers directly, same
/// as `matmul_i4_scalar`. The inner loop iterates the row's **32-element scale blocks**:
/// `e8m0_decode` is evaluated once per block (not once per element as the pre-Phase-2 loop did — it
/// recomputed `2^(byte-127)` for all 32 elements sharing one scale byte), and each packed byte's
/// two nibbles are decoded together, removing the per-element `k & 1` parity branch. The
/// per-element arithmetic `xs[k] * e2m1 * scale` and the accumulation order (ascending `k`) are
/// byte-for-byte the pre-Phase-2 element-at-a-time loop's, so this is **bit-identical** to it —
/// pinned by `matmul_mxfp4_matches_the_pre_block_reference`. Deliberately NOT per-block partial
/// sums (that reassociates); reassociation lives in the AVX-512 tier.
pub fn matmul_mxfp4_scalar(y: &mut [f32], x: &[f32], data: &[u8], block_scale: &[u8], s: usize, i: usize, o: usize) {
    let rb = i.div_ceil(2);
    let bpr = i.div_ceil(32);
    with_yt_scratch(o * s, |yt| {
        par_rows(yt, s, o, |oi, out| {
            let w = &data[oi * rb..(oi + 1) * rb];
            let bs = &block_scale[oi * bpr..(oi + 1) * bpr];
            for (si, slot) in out.iter_mut().enumerate() {
                *slot = row_dot_mxfp4_f32_scalar(w, bs, &x[si * i..(si + 1) * i], i);
            }
        });
        transpose_so(y, yt, s, o);
    });
}

/// The dot behind [`matmul_mxfp4_scalar`]'s inner loop (MXFP4 carries its scale per 32-element
/// block, so unlike the other formats there is nothing left to factor out afterward) — see
/// [`row_dot_f32`] for why the per-row bodies are factored out at all.
#[inline]
fn row_dot_mxfp4_f32_scalar(w: &[u8], bs: &[u8], xs: &[f32], i: usize) -> f32 {
    let mut a = 0f32;
    let mut k = 0;
    for &scale_byte in bs {
        let scale = e8m0_decode(scale_byte);
        let block_end = (k + 32).min(i);
        // Blocks start on even `k` (multiples of 32), so within a block a packed byte
        // holds element k (low nibble) then k+1 (high nibble) — same mapping the old
        // `if k & 1 == 0` branch computed, just unrolled two at a time.
        while k + 1 < block_end {
            let byte = w[k >> 1];
            a += xs[k] * e2m1_decode(byte & 0xF) * scale;
            a += xs[k + 1] * e2m1_decode(byte >> 4) * scale;
            k += 2;
        }
        if k < block_end {
            let byte = w[k >> 1];
            a += xs[k] * e2m1_decode(byte & 0xF) * scale;
            k += 1;
        }
    }
    a
}

/// y[S,O] = x[S,I] @ W^T, W int4-packed[O,ceil(I/2)] (2 values/byte) with a GROUPED scale
/// (`QTKind::I4Grouped` — one `f32` per `group_size`-element run along each row, not one per
/// whole row). Scalar only for now, same "correctness first" precedent as `matmul_mxfp4` (no
/// real grouped-int4 checkpoint existed to benchmark against until this session's Kimi Linear
/// conversion) — the scale must be applied INSIDE the accumulation loop here (it varies within
/// a row), unlike `matmul_i4_scalar`'s single `sc` factored out after the whole dot product.
#[allow(clippy::too_many_arguments)]
pub fn matmul_i4_grouped(y: &mut [f32], x: &[f32], q4: &[u8], scale: &[f32], group_size: usize, s: usize, i: usize, o: usize) {
    let rb = i.div_ceil(2);
    let ngroups = i.div_ceil(group_size);
    with_yt_scratch(o * s, |yt| {
        par_rows(yt, s, o, |oi, out| {
            let w = &q4[oi * rb..(oi + 1) * rb];
            let sc = &scale[oi * ngroups..(oi + 1) * ngroups];
            for (si, slot) in out.iter_mut().enumerate() {
                *slot = row_dot_i4_grouped_scalar(w, sc, &x[si * i..(si + 1) * i], i, group_size);
            }
        });
        transpose_so(y, yt, s, o);
    });
}

/// The dot behind [`matmul_i4_grouped`]'s inner loop (the group scale varies within the row, so
/// like MXFP4 it folds in per element) — see [`row_dot_f32`] for why these are factored out.
#[inline]
fn row_dot_i4_grouped_scalar(w: &[u8], sc: &[f32], xs: &[f32], i: usize, group_size: usize) -> f32 {
    let mut a = 0f32;
    for k in 0..i {
        let byte = w[k >> 1];
        let nibble = if k & 1 == 0 { byte & 0xF } else { byte >> 4 };
        a += xs[k] * (nibble as i32 - 8) as f32 * sc[k / group_size];
    }
    a
}

/// Quantizes one activation row to int8 (absmax/127, Q8_0-style) for the IDOT kernels.
/// Returns the row's scale; `q.len() == x.len()` required. Scalar only — the C never
/// vectorized this either (it's a single absmax-then-round pass, not the hot inner loop).
pub fn qrow_i8(x: &[f32], q: &mut [i8]) -> f32 {
    let amax = x.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let s = (amax / 127.0).max(1e-12);
    let inv = 1.0 / s;
    for (qi, &xi) in q.iter_mut().zip(x) {
        *qi = (xi * inv).round_ties_even() as i32 as i8;
    }
    s
}

/// int8·int8 dot product. Pairs are bounded by `127*127*2 < i32::MAX` up to unrealistic `I`,
/// so plain `i32` accumulation never overflows in practice — true at every tier alike, since
/// they all compute the same sum, just batched differently. Dispatches AVX-512-VNNI > AVX2 >
/// scalar.
pub fn dot_i8i8(w: &[i8], x: &[i8], i: usize) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512bw") {
            return unsafe { dot_i8i8_avx512vnni(w, x, i) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { dot_i8i8_avx2(w, x, i) };
        }
    }
    dot_i8i8_scalar(w, x, i)
}

/// `pub` (not just the `dot_i8i8` dispatcher above) so `benches/kernels.rs` can compare tiers
/// directly — the dispatcher always picks the fastest tier this CPU has, which makes it
/// useless for "how much faster is AVX2 than scalar" measurements.
pub fn dot_i8i8_scalar(w: &[i8], x: &[i8], i: usize) -> i32 {
    let mut sum = 0i32;
    for k in 0..i {
        sum += w[k] as i32 * x[k] as i32;
    }
    sum
}

/// int4(packed)·int8 dot product: unpack each nibble to `[-8,7]` on the fly. Dispatches
/// AVX-512-VNNI > AVX2 > scalar.
pub fn dot_i4i8(w4: &[u8], x: &[i8], i: usize) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512bw") {
            return unsafe { dot_i4i8_avx512vnni(w4, x, i) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { dot_i4i8_avx2(w4, x, i) };
        }
    }
    dot_i4i8_scalar(w4, x, i)
}

/// `pub` for the same benchmarking reason as `dot_i8i8_scalar`.
pub fn dot_i4i8_scalar(w4: &[u8], x: &[i8], i: usize) -> i32 {
    let mut sum = 0i32;
    let mut k = 0;
    while k + 1 < i {
        let b = w4[k >> 1];
        sum += ((b & 0xF) as i32 - 8) * x[k] as i32 + ((b >> 4) as i32 - 8) * x[k + 1] as i32;
        k += 2;
    }
    if k < i {
        let b = w4[k >> 1];
        sum += ((b & 0xF) as i32 - 8) * x[k] as i32;
    }
    sum
}

/// Name of the fastest dot-kernel tier this CPU actually gets at runtime — matches the C's
/// `IDOT_KERNEL` debug string. Diagnostic only; every call site here still picks its tier via
/// `is_x86_feature_detected!` independently (this is not a cached/forced override).
pub fn active_dot_kernel() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512bw") {
            return "avx512-vnni";
        }
        if is_x86_feature_detected!("avx2") {
            return "avx2";
        }
    }
    "scalar"
}

// ---- SIMD tiers (x86_64 only) ----
//
// Every function below is a direct, intrinsic-for-intrinsic port of its `glm.c` counterpart
// (see the `#ifdef __AVX2__` / `#if defined(__AVX512VNNI__) ...` blocks inline in `matmul_q`,
// `matmul_i4`, `matmul_i2`, `dot_i8i8`, `dot_i4i8`). The one structural difference: the C
// inlines the AVX body directly into each matmul's `(o,s)` loop; here the per-row dot product
// is pulled out into a small named helper (`dot_*_avx2`) called from a loop that otherwise
// matches the scalar version's shape — same instructions, same order, just not copy-pasted
// three times.
#[cfg(target_arch = "x86_64")]
mod simd {
    use super::*;

    // register-only shuffles/adds (no memory access) — safe to call from a matching
    // target_feature context without a nested `unsafe` block, unlike the load/store
    // intrinsics below.
    #[target_feature(enable = "avx2")]
    fn hsum256(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm_add_ps(lo, hi);
        let sh = _mm_movehl_ps(lo, lo);
        let lo = _mm_add_ps(lo, sh);
        let sh = _mm_shuffle_ps(lo, lo, 1);
        let lo = _mm_add_ss(lo, sh);
        _mm_cvtss_f32(lo)
    }

    #[target_feature(enable = "avx2")]
    fn hsum256_i32(v: __m256i) -> i32 {
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256(v, 1);
        let lo = _mm_add_epi32(lo, hi);
        let lo = _mm_hadd_epi32(lo, lo);
        let lo = _mm_hadd_epi32(lo, lo);
        _mm_cvtsi128_si32(lo)
    }

    /// int8 weight row (dequantized on the fly) · f32 activation row -> f32.
    #[target_feature(enable = "avx2")]
    unsafe fn dot_q8_f32_avx2(w: &[i8], xs: &[f32]) -> f32 {
        unsafe {
            let n = w.len();
            let mut acc = _mm256_setzero_ps();
            let mut i = 0;
            while i + 8 <= n {
                let wi = _mm256_cvtepi8_epi32(_mm_loadl_epi64(w.as_ptr().add(i) as *const __m128i));
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs.as_ptr().add(i)), _mm256_cvtepi32_ps(wi), acc);
                i += 8;
            }
            let mut a = hsum256(acc);
            while i < n {
                a += xs[i] * w[i] as f32;
                i += 1;
            }
            a
        }
    }

    /// Plain f32 dot product, AVX2: 8-lane fmadd + `hsum256`, scalar tail. Used by the
    /// absorbed-attention decode path's MLA-absorb score reduction (`attention.rs`,
    /// `dot(qabs, Lt)` over `kv_lora` already-dequantized latent elements — NOT a dequant
    /// helper like the other functions here) — ported from colibrì's PR #442 (commit
    /// `d469c54`, upstream v1.1.0). Reassociates vs. the scalar sequential sum (8-wide partial
    /// sums combined via `hsum256` at the end, instead of one running scalar accumulator) — NOT
    /// bit-identical, but colibrì measured the rounding delta at ~1.8e-6 there, softened by the
    /// softmax immediately downstream (never crosses a near-tie threshold in practice).
    ///
    /// # Safety
    /// Caller must have verified `is_x86_feature_detected!("avx2")`. `a` and `b` must each have
    /// length >= `n`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_f32_avx2(a: &[f32], b: &[f32], n: usize) -> f32 {
        unsafe {
            let mut acc = _mm256_setzero_ps();
            let mut i = 0;
            while i + 8 <= n {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i)), _mm256_loadu_ps(b.as_ptr().add(i)), acc);
                i += 8;
            }
            let mut s = hsum256(acc);
            while i < n {
                s += a[i] * b[i];
                i += 1;
            }
            s
        }
    }

    /// `acc[0..n) += coef * b[0..n)`, AVX2: 8-lane fmadd, per-lane writeback, scalar tail. Used
    /// by the absorbed-attention decode path's MLA-absorb value-mix accumulation
    /// (`attention.rs`, `clat[i] += sc[jj] * Lt[i]`) — ported from the same colibrì PR as
    /// `dot_f32_avx2` above. Each `acc[i]` receives exactly one fma with no cross-element
    /// reduction, so this IS bit-identical to the scalar loop (colibrì measured diff 0.0),
    /// unlike `dot_f32_avx2`.
    ///
    /// # Safety
    /// Caller must have verified `is_x86_feature_detected!("avx2")`. `b` and `acc` must each
    /// have length >= `n`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn axpy_f32_avx2(coef: f32, b: &[f32], acc: &mut [f32], n: usize) {
        unsafe {
            let cv = _mm256_set1_ps(coef);
            let mut i = 0;
            while i + 8 <= n {
                let bv = _mm256_loadu_ps(b.as_ptr().add(i));
                let av = _mm256_loadu_ps(acc.as_ptr().add(i));
                _mm256_storeu_ps(acc.as_mut_ptr().add(i), _mm256_fmadd_ps(cv, bv, av));
                i += 8;
            }
            while i < n {
                acc[i] += coef * b[i];
                i += 1;
            }
        }
    }

    /// int4-packed weight row · f32 activation row -> f32. `n` = logical (unpacked) length.
    ///
    /// # Safety
    /// Caller must have verified `is_x86_feature_detected!("avx2")`. `w` must have length >=
    /// `ceil(n/2)`, `xs` length >= `n`.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn dot_i4_f32_avx2(w: &[u8], xs: &[f32], n: usize) -> f32 {
        unsafe {
            let m4 = _mm_set1_epi8(0x0F);
            let b8 = _mm256_set1_epi32(8);
            let mut acc = _mm256_setzero_ps();
            let mut i = 0;
            while i + 16 <= n {
                let by = _mm_loadl_epi64(w.as_ptr().add(i >> 1) as *const __m128i);
                let lo = _mm_and_si128(by, m4);
                let hi = _mm_and_si128(_mm_srli_epi16::<4>(by), m4);
                let nib = _mm_unpacklo_epi8(lo, hi);
                let w0 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib), b8));
                let w1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(nib)), b8));
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs.as_ptr().add(i)), w0, acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs.as_ptr().add(i + 8)), w1, acc);
                i += 16;
            }
            let mut a = hsum256(acc);
            while i + 1 < n {
                let byte = w[i >> 1];
                let lo = (byte & 0xF) as i32 - 8;
                let hi = (byte >> 4) as i32 - 8;
                a += xs[i] * lo as f32 + xs[i + 1] * hi as f32;
                i += 2;
            }
            if i < n {
                let byte = w[i >> 1];
                let lo = (byte & 0xF) as i32 - 8;
                a += xs[i] * lo as f32;
            }
            a
        }
    }

    /// int4-packed weight row · f32 activation row -> f32, AVX-512F/BW (colibrì's
    /// `dot_i4f_avx512`/`I4_ACC512`): 32 weights/iteration across two independent `__m512` FMA
    /// chains, combined via one `_mm512_reduce_add_ps` tree-sum at the very end instead of
    /// accumulating into a single running vector — see the module doc for why this makes it
    /// NOT bit-identical to `dot_i4_f32_avx2`/scalar despite the same lossless nibble-unpack
    /// math. `n` = logical (unpacked) length.
    ///
    /// `pub` (beyond the AVX-512 tier's own `matmul_i4_avx512` use below) so `qt_matvec_rows`
    /// can reuse it for the MLA-absorption value projection (same accumulation-order tradeoff
    /// there as here, both accepted for the same reason — colibrì's own
    /// `I4_ACC512`/`g_i4_acc512` precedent) and so `benches/kernels.rs` can measure it directly.
    ///
    /// # Safety
    /// Caller must have verified `is_x86_feature_detected!("avx512f")` and `"avx512bw"`. `w`
    /// must have length >= `ceil(n/2)`, `xs` length >= `n`.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn dot_i4_f32_avx512(w: &[u8], xs: &[f32], n: usize) -> f32 {
        unsafe {
            let m4 = _mm_set1_epi8(0x0F);
            let b8 = _mm512_set1_epi32(8);
            let mut acc0 = _mm512_setzero_ps();
            let mut acc1 = _mm512_setzero_ps();
            let mut i = 0;
            while i + 32 <= n {
                let by = _mm_loadu_si128(w.as_ptr().add(i >> 1) as *const __m128i);
                let lo = _mm_and_si128(by, m4);
                let hi = _mm_and_si128(_mm_srli_epi16::<4>(by), m4);
                // unpacklo/unpackhi interleave lo_k/hi_k back into sequential element order:
                // n0 = elements [i, i+16), n1 = elements [i+16, i+32) — matches x's two loads.
                let n0 = _mm_unpacklo_epi8(lo, hi);
                let n1 = _mm_unpackhi_epi8(lo, hi);
                let w0 = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(n0), b8));
                let w1 = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(n1), b8));
                acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(xs.as_ptr().add(i)), w0, acc0);
                acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(xs.as_ptr().add(i + 16)), w1, acc1);
                i += 32;
            }
            let mut a = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
            while i + 1 < n {
                let byte = w[i >> 1];
                let lo = (byte & 0xF) as i32 - 8;
                let hi = (byte >> 4) as i32 - 8;
                a += xs[i] * lo as f32 + xs[i + 1] * hi as f32;
                i += 2;
            }
            if i < n {
                let byte = w[i >> 1];
                let lo = (byte & 0xF) as i32 - 8;
                a += xs[i] * lo as f32;
            }
            a
        }
    }

    /// `acc[0..n) += coef * dequant(int4 row)`, AVX-512F/BW — the axpy twin of
    /// `dot_i4_f32_avx512` for the MLA-absorption path's `qt_addrow` (`q_nope` absorption:
    /// `acc` accumulates `W_K^T q_nope` one scaled row at a time). Each `acc[k]` receives
    /// exactly ONE fma (no cross-element accumulation, unlike `dot_i4_f32_avx512`'s tree-sum),
    /// so this IS bit-identical to the scalar loop — ported from colibrì's `axpy_i4f_avx512`
    /// (commit `a66c99a`, upstream v1.1.0), which makes the same claim and backs it with its own
    /// bit-exact CI check.
    ///
    /// # Safety
    /// Caller must have verified `is_x86_feature_detected!("avx512f")` and `"avx512bw"`. `w`
    /// must have length >= `ceil(n/2)`, `acc` length >= `n`.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn axpy_i4_f32_avx512(w: &[u8], coef: f32, acc: &mut [f32], n: usize) {
        unsafe {
            let m4 = _mm_set1_epi8(0x0F);
            let b8 = _mm512_set1_epi32(8);
            let cv = _mm512_set1_ps(coef);
            let mut i = 0;
            while i + 32 <= n {
                let by = _mm_loadu_si128(w.as_ptr().add(i >> 1) as *const __m128i);
                let lo = _mm_and_si128(by, m4);
                let hi = _mm_and_si128(_mm_srli_epi16::<4>(by), m4);
                let n0 = _mm_unpacklo_epi8(lo, hi);
                let n1 = _mm_unpackhi_epi8(lo, hi);
                let w0 = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(n0), b8));
                let w1 = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(n1), b8));
                let a0 = _mm512_loadu_ps(acc.as_ptr().add(i));
                let a1 = _mm512_loadu_ps(acc.as_ptr().add(i + 16));
                _mm512_storeu_ps(acc.as_mut_ptr().add(i), _mm512_fmadd_ps(cv, w0, a0));
                _mm512_storeu_ps(acc.as_mut_ptr().add(i + 16), _mm512_fmadd_ps(cv, w1, a1));
                i += 32;
            }
            while i + 1 < n {
                let byte = w[i >> 1];
                let lo = (byte & 0xF) as i32 - 8;
                let hi = (byte >> 4) as i32 - 8;
                acc[i] += coef * lo as f32;
                acc[i + 1] += coef * hi as f32;
                i += 2;
            }
            if i < n {
                let byte = w[i >> 1];
                let lo = (byte & 0xF) as i32 - 8;
                acc[i] += coef * lo as f32;
            }
        }
    }

    /// int2-packed weight row · f32 activation row -> f32. `n` = logical (unpacked) length.
    ///
    /// # Safety
    /// Caller must have verified `is_x86_feature_detected!("avx2")`. `w` must have length >=
    /// `ceil(n/4)`, `xs` length >= `n`.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn dot_i2_f32_avx2(w: &[u8], xs: &[f32], n: usize) -> f32 {
        unsafe {
            let m2 = _mm_set1_epi8(0x03);
            let b2 = _mm256_set1_epi32(2);
            let mut acc = _mm256_setzero_ps();
            let mut i = 0;
            while i + 16 <= n {
                let by = _mm_loadu_si32(w.as_ptr().add(i >> 2) as *const _);
                let p0 = _mm_and_si128(by, m2);
                let p1 = _mm_and_si128(_mm_srli_epi16::<2>(by), m2);
                let p2 = _mm_and_si128(_mm_srli_epi16::<4>(by), m2);
                let p3 = _mm_and_si128(_mm_srli_epi16::<6>(by), m2);
                let lo = _mm_unpacklo_epi8(p0, p1);
                let hi = _mm_unpacklo_epi8(p2, p3);
                let nib = _mm_unpacklo_epi16(lo, hi);
                let w0 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib), b2));
                let w1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(nib)), b2));
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs.as_ptr().add(i)), w0, acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs.as_ptr().add(i + 8)), w1, acc);
                i += 16;
            }
            let mut a = hsum256(acc);
            while i < n {
                let byte = w[i >> 2];
                let sh = (i & 3) * 2;
                a += xs[i] * (((byte >> sh) & 3) as i32 - 2) as f32;
                i += 1;
            }
            a
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn matmul_q_avx2(y: &mut [f32], x: &[f32], q: &[i8], scale: &[f32], s: usize, i: usize, o: usize) {
        with_yt_scratch(o * s, |yt| {
            par_rows(yt, s, o, |oi, out| {
                let w = &q[oi * i..(oi + 1) * i];
                let sc = scale[oi];
                for (si, slot) in out.iter_mut().enumerate() {
                    let xs = &x[si * i..(si + 1) * i];
                    // Safety: caller of `matmul_q_avx2` already verified AVX2 at the dispatch
                    // site (`matmul_q`); that's a whole-machine capability, not per-thread
                    // state, so it still holds inside this rayon worker closure.
                    *slot = unsafe { dot_q8_f32_avx2(w, xs) } * sc;
                }
            });
            transpose_so(y, yt, s, o);
        });
    }

    /// y[S,O] = x[S,I] @ W^T, int4-packed W, AVX2 tier.
    ///
    /// # Safety
    /// Caller must have verified `is_x86_feature_detected!("avx2")`. `x` must have length >=
    /// `s*i`, `q4` length >= `o*ceil(i/2)`, `scale` length >= `o`, `y` length >= `s*o`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn matmul_i4_avx2(y: &mut [f32], x: &[f32], q4: &[u8], scale: &[f32], s: usize, i: usize, o: usize) {
        let rb = i.div_ceil(2);
        with_yt_scratch(o * s, |yt| {
            par_rows(yt, s, o, |oi, out| {
                let w = &q4[oi * rb..(oi + 1) * rb];
                let sc = scale[oi];
                for (si, slot) in out.iter_mut().enumerate() {
                    let xs = &x[si * i..(si + 1) * i];
                    *slot = unsafe { dot_i4_f32_avx2(w, xs, i) } * sc;
                }
            });
            transpose_so(y, yt, s, o);
        });
    }

    /// y[S,O] = x[S,I] @ W^T, int4-packed W, AVX-512F/BW dual-accumulator tier.
    ///
    /// # Safety
    /// Caller must have verified `is_x86_feature_detected!("avx512f")` and `"avx512bw"`. `x`
    /// must have length >= `s*i`, `q4` length >= `o*ceil(i/2)`, `scale` length >= `o`, `y`
    /// length >= `s*o`.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn matmul_i4_avx512(y: &mut [f32], x: &[f32], q4: &[u8], scale: &[f32], s: usize, i: usize, o: usize) {
        let rb = i.div_ceil(2);
        with_yt_scratch(o * s, |yt| {
            par_rows(yt, s, o, |oi, out| {
                let w = &q4[oi * rb..(oi + 1) * rb];
                let sc = scale[oi];
                for (si, slot) in out.iter_mut().enumerate() {
                    let xs = &x[si * i..(si + 1) * i];
                    *slot = unsafe { dot_i4_f32_avx512(w, xs, i) } * sc;
                }
            });
            transpose_so(y, yt, s, o);
        });
    }

    /// One row of MXFP4 weights · f32 activation row -> f32, AVX-512F/BW. The format is nearly
    /// designed for this: one 32-element E8M0 scale block = 16 packed bytes = two 16-lane f32
    /// vectors, and E2M1 has exactly 16 code points, so the whole sign×magnitude decode table fits
    /// one `zmm` addressed by `_mm512_permutexvar_ps` (which uses each index's low 4 bits — exactly
    /// the nibble). Per full block: unpack the 16 bytes' low/high nibbles into two i32 index
    /// vectors (the same `unpacklo/unpackhi` interleave `dot_i4_f32_avx512` uses so element order
    /// is preserved), gather E2M1 values via `permutexvar`, fold the block's scalar-decoded E8M0
    /// scale in, and FMA into two independent accumulator chains reduced by one `_mm512_reduce_add_ps`
    /// tree-sum at the end. That scale-fold + two-chain reduction reassociates relative to the
    /// scalar `x*e2m1*scale` running sum, so this is **within-tolerance, NOT bit-identical** — the
    /// same accuracy tradeoff `dot_i4_f32_avx512` documents (colibrì measured such reordering as
    /// *lower* max error than scalar, but different bits). `n` = logical row length; the `cols % 32`
    /// tail (a final partial block) is done scalar, matching the scalar decode exactly.
    ///
    /// `pub` so `benches/kernels.rs` can measure it directly, same as the other tier kernels.
    ///
    /// # Safety
    /// Caller must have verified `is_x86_feature_detected!("avx512f")` and `"avx512bw"`. `data`
    /// must have length >= `ceil(n/2)`, `bs` length >= `ceil(n/32)`, `xs` length >= `n`.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn dot_mxfp4_f32_avx512(data: &[u8], bs: &[u8], xs: &[f32], n: usize) -> f32 {
        unsafe {
            // table[nibble] = e2m1_decode(nibble): bit 3 = sign, low 3 bits index the magnitudes.
            let table = _mm512_setr_ps(
                0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
            );
            let m4 = _mm_set1_epi8(0x0F);
            let mut acc0 = _mm512_setzero_ps();
            let mut acc1 = _mm512_setzero_ps();
            let mut k = 0;
            let mut b = 0;
            while k + 32 <= n {
                let scale = _mm512_set1_ps(e8m0_decode(bs[b]));
                let by = _mm_loadu_si128(data.as_ptr().add(k >> 1) as *const __m128i);
                let lo = _mm_and_si128(by, m4);
                let hi = _mm_and_si128(_mm_srli_epi16::<4>(by), m4);
                let n0 = _mm_unpacklo_epi8(lo, hi); // nibbles for elements [k, k+16)
                let n1 = _mm_unpackhi_epi8(lo, hi); // nibbles for elements [k+16, k+32)
                let w0 = _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_cvtepu8_epi32(n0), table), scale);
                let w1 = _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_cvtepu8_epi32(n1), table), scale);
                acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(xs.as_ptr().add(k)), w0, acc0);
                acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(xs.as_ptr().add(k + 16)), w1, acc1);
                k += 32;
                b += 1;
            }
            let mut a = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
            if k < n {
                // Final partial block (< 32 elements): scalar, same decode/order as the scalar tier.
                let scale = e8m0_decode(bs[b]);
                while k < n {
                    let byte = data[k >> 1];
                    let nibble = if k & 1 == 0 { byte & 0xF } else { byte >> 4 };
                    a += xs[k] * e2m1_decode(nibble) * scale;
                    k += 1;
                }
            }
            a
        }
    }

    /// AVX-512F/BW tier of `matmul_mxfp4` — parallelizes over output rows exactly like the scalar
    /// tier and `matmul_i4_avx512`, calling `dot_mxfp4_f32_avx512` per (row, seq) dot. Within-
    /// tolerance vs scalar (per-row reassociation), see `dot_mxfp4_f32_avx512`.
    ///
    /// # Safety
    /// Caller must have verified `is_x86_feature_detected!("avx512f")` and `"avx512bw"`. Slice
    /// lengths as for `matmul_mxfp4`.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn matmul_mxfp4_avx512(y: &mut [f32], x: &[f32], data: &[u8], block_scale: &[u8], s: usize, i: usize, o: usize) {
        let rb = i.div_ceil(2);
        let bpr = i.div_ceil(32);
        with_yt_scratch(o * s, |yt| {
            par_rows(yt, s, o, |oi, out| {
                let w = &data[oi * rb..(oi + 1) * rb];
                let bs = &block_scale[oi * bpr..(oi + 1) * bpr];
                for (si, slot) in out.iter_mut().enumerate() {
                    let xs = &x[si * i..(si + 1) * i];
                    *slot = unsafe { dot_mxfp4_f32_avx512(w, bs, xs, i) };
                }
            });
            transpose_so(y, yt, s, o);
        });
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn matmul_i2_avx2(y: &mut [f32], x: &[f32], q2: &[u8], scale: &[f32], s: usize, i: usize, o: usize) {
        let rb = i.div_ceil(4);
        with_yt_scratch(o * s, |yt| {
            par_rows(yt, s, o, |oi, out| {
                let w = &q2[oi * rb..(oi + 1) * rb];
                let sc = scale[oi];
                for (si, slot) in out.iter_mut().enumerate() {
                    let xs = &x[si * i..(si + 1) * i];
                    *slot = unsafe { dot_i2_f32_avx2(w, xs, i) } * sc;
                }
            });
            transpose_so(y, yt, s, o);
        });
    }

    /// int8·int8 dot, AVX2: the sign trick (|w| unsigned × x·sign(w) signed) — safe because
    /// pairs are bounded by `128*127*2 = 32512 < 32767` up to `I=16384`.
    ///
    /// # Safety
    /// Caller must have verified `is_x86_feature_detected!("avx2")`, and `w`/`x` must each
    /// have length >= `n`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_i8i8_avx2(w: &[i8], x: &[i8], n: usize) -> i32 {
        unsafe {
            let mut acc = _mm256_setzero_si256();
            let ones = _mm256_set1_epi16(1);
            let mut i = 0;
            while i + 32 <= n {
                let wv = _mm256_loadu_si256(w.as_ptr().add(i) as *const __m256i);
                let xv = _mm256_loadu_si256(x.as_ptr().add(i) as *const __m256i);
                let p = _mm256_maddubs_epi16(_mm256_sign_epi8(wv, wv), _mm256_sign_epi8(xv, wv));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, ones));
                i += 32;
            }
            let mut sum = hsum256_i32(acc);
            while i < n {
                sum += w[i] as i32 * x[i] as i32;
                i += 1;
            }
            sum
        }
    }

    /// int8·int8 dot, AVX-512/VNNI: `vpdpbusd` -> s32 directly, 64 bytes/iter, no 16-bit
    /// intermediate. AVX-512 has no `vpsignb`; `|w|` via `abs`, sign folded into `x` with a
    /// mask-negate (`w==0` -> product 0 either way).
    ///
    /// # Safety
    /// Caller must have verified `is_x86_feature_detected!("avx512vnni")` and `"avx512bw"`,
    /// and `w`/`x` must each have length >= `n`.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    pub unsafe fn dot_i8i8_avx512vnni(w: &[i8], x: &[i8], n: usize) -> i32 {
        unsafe {
            let mut acc = _mm512_setzero_si512();
            let mut i = 0;
            while i + 64 <= n {
                let wv = _mm512_loadu_si512(w.as_ptr().add(i) as *const _);
                let xv = _mm512_loadu_si512(x.as_ptr().add(i) as *const _);
                let neg = _mm512_movepi8_mask(wv);
                let xs = _mm512_mask_sub_epi8(xv, neg, _mm512_setzero_si512(), xv);
                acc = _mm512_dpbusd_epi32(acc, _mm512_abs_epi8(wv), xs);
                i += 64;
            }
            let mut sum = _mm512_reduce_add_epi32(acc);
            while i < n {
                sum += w[i] as i32 * x[i] as i32;
                i += 1;
            }
            sum
        }
    }

    /// int4(packed)·int8 dot, AVX2: nibble -> int8 `[-8,7]` on the fly, then the same sign
    /// trick as `dot_i8i8_avx2`.
    ///
    /// # Safety
    /// Caller must have verified `is_x86_feature_detected!("avx2")`, `x` must have length >=
    /// `n`, and `w4` must have length >= `ceil(n/2)`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_i4i8_avx2(w4: &[u8], x: &[i8], n: usize) -> i32 {
        unsafe {
            let m4 = _mm_set1_epi8(0x0F);
            let b8 = _mm256_set1_epi8(8);
            let ones = _mm256_set1_epi16(1);
            let mut acc = _mm256_setzero_si256();
            let mut i = 0;
            while i + 32 <= n {
                let by = _mm_loadu_si128(w4.as_ptr().add(i >> 1) as *const __m128i);
                let lo = _mm_and_si128(by, m4);
                let hi = _mm_and_si128(_mm_srli_epi16::<4>(by), m4);
                let n0 = _mm_unpacklo_epi8(lo, hi);
                let n1 = _mm_unpackhi_epi8(lo, hi);
                let wv = _mm256_sub_epi8(_mm256_set_m128i(n1, n0), b8);
                let xv = _mm256_loadu_si256(x.as_ptr().add(i) as *const __m256i);
                let p = _mm256_maddubs_epi16(_mm256_sign_epi8(wv, wv), _mm256_sign_epi8(xv, wv));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, ones));
                i += 32;
            }
            let mut sum = hsum256_i32(acc);
            while i + 1 < n {
                let b = w4[i >> 1];
                sum += ((b & 0xF) as i32 - 8) * x[i] as i32 + ((b >> 4) as i32 - 8) * x[i + 1] as i32;
                i += 2;
            }
            if i < n {
                let b = w4[i >> 1];
                sum += ((b & 0xF) as i32 - 8) * x[i] as i32;
            }
            sum
        }
    }

    /// int4(packed)·int8 dot, AVX-512/VNNI: 32 bytes = 64 nibbles -> int8 in `[-8,7]`, one
    /// `vpdpbusd` per 64 values. The 256-bit unpack leaves values in per-128-lane order
    /// `[0-15][32-47]/[16-31][48-63]`; dot pairing is order-invariant, so `x`'s 128-bit blocks
    /// are permuted to match instead of re-ordering `w` (one `vpermq` per iter, off the
    /// critical unpack path).
    ///
    /// # Safety
    /// Caller must have verified `is_x86_feature_detected!("avx512vnni")` and `"avx512bw"`,
    /// `x` must have length >= `n`, and `w4` must have length >= `ceil(n/2)`.
    #[target_feature(enable = "avx2,avx512f,avx512bw,avx512vnni")]
    pub unsafe fn dot_i4i8_avx512vnni(w4: &[u8], x: &[i8], n: usize) -> i32 {
        unsafe {
            let m4v = _mm256_set1_epi8(0x0F);
            let b8v = _mm512_set1_epi8(8);
            let xidx = _mm512_setr_epi64(0, 1, 4, 5, 2, 3, 6, 7);
            let mut acc = _mm512_setzero_si512();
            let mut i = 0;
            while i + 64 <= n {
                let by = _mm256_loadu_si256(w4.as_ptr().add(i >> 1) as *const __m256i);
                let lo = _mm256_and_si256(by, m4v);
                let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(by), m4v);
                let z0 = _mm256_unpacklo_epi8(lo, hi);
                let z1 = _mm256_unpackhi_epi8(lo, hi);
                let wv = _mm512_sub_epi8(_mm512_inserti64x4::<1>(_mm512_castsi256_si512(z0), z1), b8v);
                let xv = _mm512_permutexvar_epi64(xidx, _mm512_loadu_si512(x.as_ptr().add(i) as *const _));
                let neg = _mm512_movepi8_mask(wv);
                let xs = _mm512_mask_sub_epi8(xv, neg, _mm512_setzero_si512(), xv);
                acc = _mm512_dpbusd_epi32(acc, _mm512_abs_epi8(wv), xs);
                i += 64;
            }
            let mut sum = _mm512_reduce_add_epi32(acc);
            while i + 1 < n {
                let b = w4[i >> 1];
                sum += ((b & 0xF) as i32 - 8) * x[i] as i32 + ((b >> 4) as i32 - 8) * x[i + 1] as i32;
                i += 2;
            }
            if i < n {
                let b = w4[i >> 1];
                sum += ((b & 0xF) as i32 - 8) * x[i] as i32;
            }
            sum
        }
    }
}
#[cfg(target_arch = "x86_64")]
use simd::*;
// re-exported `pub` (beyond the crate-internal `use simd::*` above) so
// `benches/kernels.rs` can invoke each IDOT tier directly — see the doc comments on
// `dot_i8i8_scalar`/`dot_i4i8_scalar` for why the auto-dispatching `dot_i8i8`/`dot_i4i8`
// functions can't be used for a scalar-vs-AVX2-vs-AVX-512 comparison.
#[cfg(target_arch = "x86_64")]
pub use simd::{
    axpy_f32_avx2, axpy_i4_f32_avx512, dot_f32_avx2, dot_i4_f32_avx512, dot_i4i8_avx2, dot_i4i8_avx512vnni, dot_i8i8_avx2,
    dot_i8i8_avx512vnni, dot_mxfp4_f32_avx512, matmul_i4_avx2, matmul_i4_avx512, matmul_mxfp4_avx512,
};

// mirrors matmul_q_idot/matmul_i4_idot's C signature 1:1; both are private, single-call-site
// helpers for matmul_qt, so a wrapper struct would be indirection with no real caller benefit.
#[allow(clippy::too_many_arguments)]
fn matmul_q_idot(y: &mut [f32], xq: &[i8], sx: &[f32], q: &[i8], scale: &[f32], s: usize, i: usize, o: usize) {
    with_yt_scratch(o * s, |yt| {
        par_rows(yt, s, o, |oi, out| {
            let w = &q[oi * i..(oi + 1) * i];
            let sc = scale[oi];
            for (si, slot) in out.iter_mut().enumerate() {
                let xrow = &xq[si * i..(si + 1) * i];
                *slot = dot_i8i8(w, xrow, i) as f32 * sc * sx[si];
            }
        });
        transpose_so(y, yt, s, o);
    });
}

#[allow(clippy::too_many_arguments)]
fn matmul_i4_idot(y: &mut [f32], xq: &[i8], sx: &[f32], q4: &[u8], scale: &[f32], s: usize, i: usize, o: usize) {
    let rb = i.div_ceil(2);
    with_yt_scratch(o * s, |yt| {
        par_rows(yt, s, o, |oi, out| {
            let w = &q4[oi * rb..(oi + 1) * rb];
            let sc = scale[oi];
            for (si, slot) in out.iter_mut().enumerate() {
                let xrow = &xq[si * i..(si + 1) * i];
                *slot = dot_i4i8(w, xrow, i) as f32 * sc * sx[si];
            }
        });
        transpose_so(y, yt, s, o);
    });
}

/// x86 default from `glm.c`'s `g_i4s`: without ARM SDOT, int4 IDOT only pays off at S>=2 —
/// at S=1 (decode) the float-weight path (`matmul_i4`) wins.
const I4_IDOT_MIN_S: usize = 2;

/// y[S,O] = x[S,I] @ W^T for a `QT` in any format — the dispatcher every layer calls.
/// int8 IDOT is always used (the reference implementation measures 1.4-2.5x over the float-weight path); int4
/// IDOT only kicks in at `S >= I4_IDOT_MIN_S`; int2 has no IDOT path (matches the original,
/// which never added one).
pub fn matmul_qt(y: &mut [f32], x: &[f32], w: &QT, s: usize) {
    match &w.kind {
        QTKind::F32(data) => matmul(y, x, data, s, w.cols, w.rows),
        QTKind::I8 { data, scale } => {
            let (xq, sx) = quantize_activations(x, s, w.cols);
            matmul_q_idot(y, &xq, &sx, data, scale, s, w.cols, w.rows);
        }
        QTKind::I4 { data, scale } => {
            if s >= I4_IDOT_MIN_S {
                let (xq, sx) = quantize_activations(x, s, w.cols);
                matmul_i4_idot(y, &xq, &sx, data, scale, s, w.cols, w.rows);
            } else {
                matmul_i4(y, x, data, scale, s, w.cols, w.rows);
            }
        }
        QTKind::I2 { data, scale } => matmul_i2(y, x, data, scale, s, w.cols, w.rows),
        QTKind::I4Grouped { data, scale, group_size } => matmul_i4_grouped(y, x, data, scale, *group_size, s, w.cols, w.rows),
        QTKind::MxFp4 { data, block_scale } => matmul_mxfp4(y, x, data, block_scale, s, w.cols, w.rows),
    }
}

/// Activations prepared once, ahead of a row-blocked fan-out over one weight matrix.
///
/// [`matmul_qt`] parallelizes over output rows INTERNALLY (`par_chunks_mut`) and quantizes `x`
/// once per call for its IDOT tiers. [`matmul_qt_rows`] inverts that: the caller owns the
/// parallelism and calls it once per row block, so the quantization would otherwise be repeated
/// per block. Preparing it here hoists that work back out to once per (`x`, weight) pair.
///
/// `x` is always retained, so an `acts` prepared against a different-kind weight is still
/// *correct* (the IDOT arms fall back to quantizing inline) — just slower than intended.
pub struct RowActs<'a> {
    x: &'a [f32],
    s: usize,
    /// `Some` exactly when `matmul_qt` would take an int8-IDOT tier for the weight this was
    /// prepared against: `QTKind::I8` always, `QTKind::I4` at `s >= I4_IDOT_MIN_S`.
    q: Option<(Vec<i8>, Vec<f32>)>,
}

impl<'a> RowActs<'a> {
    /// Prepares `x` (`[s, w.cols]`) for row-blocked matmuls against `w`.
    pub fn prepare(x: &'a [f32], w: &QT, s: usize) -> RowActs<'a> {
        let idot = match &w.kind {
            QTKind::I8 { .. } => true,
            QTKind::I4 { .. } => s >= I4_IDOT_MIN_S,
            _ => false,
        };
        let q = idot.then(|| quantize_activations(x, s, w.cols));
        RowActs { x, s, q }
    }
}

/// `yt[j*s + si] = <W[r0+j, :], x[si, :]>` for `j` in `0..nrows` — ONE ROW BLOCK of what
/// [`matmul_qt`] computes, evaluated serially on the calling thread and written in `matmul_qt`'s
/// own internal `[O, S]` transposed layout (its `yt`), which is what makes a row block a
/// contiguous, independently-writable slice.
///
/// The point is inverted parallelism: `matmul_qt` forks internally over output rows and joins,
/// which at batch-1 decode divides thousands of one-element tasks over the pool and spends more
/// time scheduling than computing (`PERFORMANCE.md`, Phase 3 and Phase 5 v2). A caller that
/// already has coarser work to spread — K3's `latent_moe`, which has 16 experts × 3 matmuls of
/// independent rows available at once — calls this instead and forks exactly once itself.
///
/// **Bit-identity contract:** tier selection mirrors `matmul_qt`'s dispatch exactly, and every
/// per-row body is the same `#[inline] row_dot_*` helper (or the same `dot_*` SIMD kernel) the
/// whole-matrix kernels call, in the same per-element order. So assembling every row block and
/// transposing `[O,S]` → `[S,O]` reproduces `matmul_qt`'s output **bit for bit** — pinned across
/// all six `QTKind`s by `matmul_qt_rows_reassembled_is_bit_identical_to_matmul_qt`. (That is a
/// contract about the two *paths* agreeing; where the tier itself is within-tolerance rather than
/// exact against scalar — AVX-512 int4/MXFP4 — both paths are equally within-tolerance, because
/// they run the identical kernel.)
///
/// `acts` should come from [`RowActs::prepare`] against this same `w` and `x`; `yt.len()` must be
/// `nrows * s`, and `r0 + nrows` must not exceed `w.rows`.
pub fn matmul_qt_rows(yt: &mut [f32], acts: &RowActs, w: &QT, r0: usize, nrows: usize) {
    let i = w.cols;
    let s = acts.s;
    let x = acts.x;
    debug_assert_eq!(yt.len(), nrows * s, "matmul_qt_rows: yt must be [nrows, s]");
    debug_assert!(r0 + nrows <= w.rows, "matmul_qt_rows: row block past the end of the weight");
    match &w.kind {
        QTKind::F32(data) => {
            for (j, out) in yt.chunks_mut(s).enumerate() {
                let row = r0 + j;
                let wr = &data[row * i..(row + 1) * i];
                for (si, slot) in out.iter_mut().enumerate() {
                    *slot = row_dot_f32(wr, &x[si * i..(si + 1) * i]);
                }
            }
        }
        QTKind::I8 { data, scale } => {
            let owned;
            let (xq, sx) = match &acts.q {
                Some((xq, sx)) => (xq, sx),
                None => {
                    owned = quantize_activations(x, s, i);
                    (&owned.0, &owned.1)
                }
            };
            for (j, out) in yt.chunks_mut(s).enumerate() {
                let row = r0 + j;
                let wr = &data[row * i..(row + 1) * i];
                let sc = scale[row];
                for (si, slot) in out.iter_mut().enumerate() {
                    *slot = dot_i8i8(wr, &xq[si * i..(si + 1) * i], i) as f32 * sc * sx[si];
                }
            }
        }
        QTKind::I4 { data, scale } => {
            let rb = i.div_ceil(2);
            // Mirrors `matmul_qt`'s own `s >= I4_IDOT_MIN_S` split, then `matmul_i4`'s tier ladder.
            if s >= I4_IDOT_MIN_S {
                let owned;
                let (xq, sx) = match &acts.q {
                    Some((xq, sx)) => (xq, sx),
                    None => {
                        owned = quantize_activations(x, s, i);
                        (&owned.0, &owned.1)
                    }
                };
                for (j, out) in yt.chunks_mut(s).enumerate() {
                    let row = r0 + j;
                    let wr = &data[row * rb..(row + 1) * rb];
                    let sc = scale[row];
                    for (si, slot) in out.iter_mut().enumerate() {
                        *slot = dot_i4i8(wr, &xq[si * i..(si + 1) * i], i) as f32 * sc * sx[si];
                    }
                }
                return;
            }
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                    for (j, out) in yt.chunks_mut(s).enumerate() {
                        let row = r0 + j;
                        let wr = &data[row * rb..(row + 1) * rb];
                        let sc = scale[row];
                        for (si, slot) in out.iter_mut().enumerate() {
                            // Safety: AVX-512F/BW just verified; a whole-machine capability.
                            *slot = unsafe { dot_i4_f32_avx512(wr, &x[si * i..(si + 1) * i], i) } * sc;
                        }
                    }
                    return;
                }
                if has_avx2() {
                    for (j, out) in yt.chunks_mut(s).enumerate() {
                        let row = r0 + j;
                        let wr = &data[row * rb..(row + 1) * rb];
                        let sc = scale[row];
                        for (si, slot) in out.iter_mut().enumerate() {
                            // Safety: AVX2 just verified; a whole-machine capability.
                            *slot = unsafe { dot_i4_f32_avx2(wr, &x[si * i..(si + 1) * i], i) } * sc;
                        }
                    }
                    return;
                }
            }
            for (j, out) in yt.chunks_mut(s).enumerate() {
                let row = r0 + j;
                let wr = &data[row * rb..(row + 1) * rb];
                let sc = scale[row];
                for (si, slot) in out.iter_mut().enumerate() {
                    *slot = row_dot_i4_f32_pairs(wr, &x[si * i..(si + 1) * i], i) * sc;
                }
            }
        }
        QTKind::I2 { data, scale } => {
            let rb = i.div_ceil(4);
            #[cfg(target_arch = "x86_64")]
            if has_avx2() {
                for (j, out) in yt.chunks_mut(s).enumerate() {
                    let row = r0 + j;
                    let wr = &data[row * rb..(row + 1) * rb];
                    let sc = scale[row];
                    for (si, slot) in out.iter_mut().enumerate() {
                        // Safety: AVX2 just verified; a whole-machine capability.
                        *slot = unsafe { dot_i2_f32_avx2(wr, &x[si * i..(si + 1) * i], i) } * sc;
                    }
                }
                return;
            }
            for (j, out) in yt.chunks_mut(s).enumerate() {
                let row = r0 + j;
                let wr = &data[row * rb..(row + 1) * rb];
                let sc = scale[row];
                for (si, slot) in out.iter_mut().enumerate() {
                    *slot = row_dot_i2_f32_scalar(wr, &x[si * i..(si + 1) * i], i) * sc;
                }
            }
        }
        QTKind::I4Grouped { data, scale, group_size } => {
            let rb = i.div_ceil(2);
            let ngroups = i.div_ceil(*group_size);
            for (j, out) in yt.chunks_mut(s).enumerate() {
                let row = r0 + j;
                let wr = &data[row * rb..(row + 1) * rb];
                let sc = &scale[row * ngroups..(row + 1) * ngroups];
                for (si, slot) in out.iter_mut().enumerate() {
                    *slot = row_dot_i4_grouped_scalar(wr, sc, &x[si * i..(si + 1) * i], i, *group_size);
                }
            }
        }
        QTKind::MxFp4 { data, block_scale } => {
            let rb = i.div_ceil(2);
            let bpr = i.div_ceil(32);
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                for (j, out) in yt.chunks_mut(s).enumerate() {
                    let row = r0 + j;
                    let wr = &data[row * rb..(row + 1) * rb];
                    let bs = &block_scale[row * bpr..(row + 1) * bpr];
                    for (si, slot) in out.iter_mut().enumerate() {
                        // Safety: AVX-512F/BW just verified; a whole-machine capability.
                        *slot = unsafe { dot_mxfp4_f32_avx512(wr, bs, &x[si * i..(si + 1) * i], i) };
                    }
                }
                return;
            }
            for (j, out) in yt.chunks_mut(s).enumerate() {
                let row = r0 + j;
                let wr = &data[row * rb..(row + 1) * rb];
                let bs = &block_scale[row * bpr..(row + 1) * bpr];
                for (si, slot) in out.iter_mut().enumerate() {
                    *slot = row_dot_mxfp4_f32_scalar(wr, bs, &x[si * i..(si + 1) * i], i);
                }
            }
        }
    }
}

fn quantize_activations(x: &[f32], s: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
    let mut xq = vec![0i8; s * cols];
    let mut sx = vec![0f32; s];
    for si in 0..s {
        sx[si] = qrow_i8(&x[si * cols..(si + 1) * cols], &mut xq[si * cols..(si + 1) * cols]);
    }
    (xq, sx)
}

/// A dense weight row-sharded across the NUMA domains (Phase N4a, `NUMA_AMX_BRIEF.md`): shard
/// `i`'s rows live in their own [`QT`] whose pages were first-touched inside domain `i`'s pinned
/// pool, so a [`QTSharded::matvec`] fan-out has every domain reading domain-local weight bytes.
/// Row-major `QT` buffers make the split trivial — a contiguous row range of any `QTKind` is a
/// contiguous byte range in each of its buffers.
///
/// s=1 (decode matvec) only, deliberately: at s=1 `matmul_qt`'s internal transposed `yt` layout
/// IS the output layout, so shard `i` writes `y[start_i..start_i+rows_i]` directly and
/// reassembly is free. Every converted call site (lm_head, KDA projections) is structurally
/// s=1; prefill's s>1 matmuls are not converted (see the N4 `PERFORMANCE.md` section for the
/// fan-out-frequency arithmetic that also keeps the latent down/up unconverted).
pub struct QTSharded {
    shards: Vec<QT>,
    /// `starts[i]` = first output row of shard `i`; `starts[n] == rows` (one-past-the-end
    /// sentinel, so `starts[i+1] - starts[i]` is shard `i`'s row count).
    starts: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

impl QTSharded {
    /// Splits `qt`'s rows into one contiguous block per domain of `pools`, COPYING each block
    /// into a fresh `QT` allocated and filled inside that domain's pinned pool — the copy is
    /// the placement mechanism (first touch on a policy-reset pinned worker), exactly N3b's
    /// trick applied to a dense weight. The original allocation is dropped. Load-time cost:
    /// one extra pass over the weight's bytes.
    pub fn shard(qt: QT, pools: &crate::numa::NodePools) -> QTSharded {
        let n = pools.n();
        let (rows, cols) = (qt.rows, qt.cols);
        let per = rows.div_ceil(n);
        let starts: Vec<usize> = (0..=n).map(|i| (i * per).min(rows)).collect();
        let mut shards: Vec<Option<QT>> = (0..n).map(|_| None).collect();
        {
            let slots: Vec<std::sync::Mutex<&mut Option<QT>>> = shards.iter_mut().map(std::sync::Mutex::new).collect();
            let (qt, starts) = (&qt, &starts);
            pools.run_all(|i| {
                **slots[i].lock().unwrap() = Some(qt.copy_rows(starts[i], starts[i + 1] - starts[i]));
            });
        }
        QTSharded { shards: shards.into_iter().map(|s| s.expect("every domain filled its shard")).collect(), starts, rows, cols }
    }

    /// `y[rows] = W @ x[cols]` — [`matmul_qt`] at s=1, as ONE cross-domain fan-out with each
    /// domain computing its own shard's row range (row-block-parallel within its pool) straight
    /// into its disjoint slice of `y`. Bit-identical to `matmul_qt` on the unsharded weight:
    /// every row goes through the same [`matmul_qt_rows`] kernel bodies in the same per-element
    /// order, and shard boundaries change only which thread computes a row (pinned by
    /// `qt_sharded_matvec_is_bit_identical_to_matmul_qt` on this machine's real topology).
    pub fn matvec(&self, y: &mut [f32], x: &[f32], pools: &crate::numa::NodePools) {
        debug_assert_eq!(y.len(), self.rows);
        matvec_sharded_batch(pools, x, &mut [(y, self)]);
    }
}

/// A dense weight that is either today's plain [`QT`] or its NUMA-sharded form — the storage
/// type for the handful of decode-hot dense weights Phase N4 converts (K3's lm_head and KDA
/// q/k/v/o projections). Loaded `Plain` always; a post-load pass (`kimi_k3::model::Model::
/// distribute_dense`) shards them iff `--numa` pools exist, so every non-NUMA configuration
/// carries exactly the storage and code path it had before.
pub enum DenseQT {
    Plain(QT),
    Sharded(QTSharded),
}

impl DenseQT {
    /// s=1 matvec through whichever storage this is. `Sharded` fetches the singleton pools —
    /// sharding only ever happens because the pools exist, and the singleton lives for the
    /// process, so this cannot dangle.
    pub fn matvec(&self, y: &mut [f32], x: &[f32]) {
        match self {
            DenseQT::Plain(qt) => matmul_qt(y, x, qt, 1),
            DenseQT::Sharded(w) => w.matvec(y, x, crate::numa::NodePools::get().expect("sharded weights exist only when the node pools do")),
        }
    }

    /// Consumes a `Plain` into `Sharded` across `pools` (no-op if already sharded).
    pub fn shard(self, pools: &crate::numa::NodePools) -> DenseQT {
        match self {
            DenseQT::Plain(qt) => DenseQT::Sharded(QTSharded::shard(qt, pools)),
            sharded => sharded,
        }
    }

    /// In-place [`DenseQT::shard`] for weights living behind `&mut` in a loaded model (the
    /// momentary placeholder is a zero-row `QT`, never observable — this is single-threaded
    /// load-time code).
    pub fn shard_in_place(&mut self, pools: &crate::numa::NodePools) {
        if matches!(self, DenseQT::Plain(_)) {
            let plain = std::mem::replace(self, DenseQT::Plain(QT::alloc(0, 0, 32, false)));
            *self = plain.shard(pools);
        }
    }

    pub fn rows(&self) -> usize {
        match self {
            DenseQT::Plain(qt) => qt.rows,
            DenseQT::Sharded(w) => w.rows,
        }
    }
}

/// Several sharded matvecs over the SAME `x` in ONE cross-domain fan-out — the batched shape
/// KDA's q/k/v projections want (three weights, one fan-out per layer instead of three; the
/// fan-out itself is the dominant cost at this call frequency, see `PERFORMANCE.md`).
pub fn matvec_sharded_batch(pools: &crate::numa::NodePools, x: &[f32], jobs: &mut [(&mut [f32], &QTSharded)]) {
    // Split every output at its weight's shard boundaries up front, so the fan-out closure only
    // ever sees its own domain's disjoint `&mut` slices (Mutex = `Fn`-boundary plumbing, one
    // uncontended lock per (job, domain), never blocking).
    let mut per_domain: Vec<Vec<(std::sync::Mutex<&mut [f32]>, &QT, usize)>> = (0..pools.n()).map(|_| Vec::new()).collect();
    for (y, w) in jobs.iter_mut() {
        debug_assert_eq!(y.len(), w.rows);
        let mut rest: &mut [f32] = y;
        for i in 0..pools.n() {
            let nr = w.starts[i + 1] - w.starts[i];
            let (mine, tail) = rest.split_at_mut(nr);
            rest = tail;
            per_domain[i].push((std::sync::Mutex::new(mine), &w.shards[i], nr));
        }
    }
    pools.run_all(|i| {
        let tpp = pools.threads_per_pool();
        for (slot, shard, nr) in &per_domain[i] {
            if *nr == 0 {
                continue;
            }
            let mut out = slot.lock().unwrap();
            let acts = RowActs::prepare(x, shard, 1);
            let blk = nr.div_ceil(tpp * 4).max(1);
            out.par_chunks_mut(blk).enumerate().for_each(|(b, seg)| {
                matmul_qt_rows(seg, &acts, shard, b * blk, seg.len());
            });
        }
    });
}

/// Scalar reference for `qt_addrow`'s int4 branch — factored out of the inline loop so
/// `axpy_i4_f32_avx512` can be checked for bit-exactness against this SAME logic (not a
/// differently-associated reference like `QT::row_f32`), and so the AVX-512 dispatch's fallback
/// arm has something to call instead of duplicating the loop body.
pub fn axpy_i4_f32_scalar(w: &[u8], coef: f32, acc: &mut [f32], n: usize) {
    for k in 0..n {
        let byte = w[k >> 1];
        let nibble = if k & 1 == 0 { byte & 0xF } else { byte >> 4 };
        acc[k] += coef * (nibble as i32 - 8) as f32;
    }
}

/// Scalar reference for `qt_matvec_rows`'s int4 branch — same reasoning as
/// `axpy_i4_f32_scalar` above, for the `dot_i4_f32_avx512` dispatch.
pub fn dot_i4_f32_scalar(w: &[u8], x: &[f32], n: usize) -> f32 {
    let mut acc = 0f32;
    for k in 0..n {
        let byte = w[k >> 1];
        let nibble = if k & 1 == 0 { byte & 0xF } else { byte >> 4 };
        acc += (nibble as i32 - 8) as f32 * x[k];
    }
    acc
}

/// `acc[0..I) += coef * W[row,:]` — dequantizes one row of `w` on the fly and scales it by
/// `coef` before accumulating. This is the "weight absorption" building block: MLA's decode
/// path folds `q_nope · k_nope_t = q_nope · (W_K L_t) = (W_K^T q_nope) · L_t`, so instead of
/// reconstructing `k_nope_t` per cached token it accumulates `W_K^T q_nope` once (a sum of
/// `qk_nope` scaled rows) and dot-products that against the raw latents directly. Int4 dispatches
/// AVX-512F/BW (bit-identical axpy, see `axpy_i4_f32_avx512`'s doc) when available, else scalar —
/// the other formats stay scalar-only (colibrì never vectorized this helper beyond int4 either).
pub fn qt_addrow(w: &QT, row: usize, coef: f32, acc: &mut [f32]) {
    let i = w.cols;
    match &w.kind {
        QTKind::F32(data) => {
            let wr = &data[row * i..(row + 1) * i];
            for (a, &wv) in acc.iter_mut().zip(wr) {
                *a += coef * wv;
            }
        }
        QTKind::I8 { data, scale } => {
            let c = coef * scale[row];
            let wr = &data[row * i..(row + 1) * i];
            for (a, &wv) in acc.iter_mut().zip(wr) {
                *a += c * wv as f32;
            }
        }
        QTKind::I4 { data, scale } => {
            let c = coef * scale[row];
            let rb = i.div_ceil(2);
            let wr = &data[row * rb..(row + 1) * rb];
            #[cfg(target_arch = "x86_64")]
            if has_avx512_i4() {
                unsafe { axpy_i4_f32_avx512(wr, c, acc, i) };
                return;
            }
            axpy_i4_f32_scalar(wr, c, acc, i);
        }
        QTKind::I2 { data, scale } => {
            let c = coef * scale[row];
            let rb = i.div_ceil(4);
            let wr = &data[row * rb..(row + 1) * rb];
            for k in 0..i {
                let byte = wr[k >> 2];
                let bits = (byte >> ((k & 3) * 2)) & 3;
                acc[k] += c * (bits as i32 - 2) as f32;
            }
        }
        QTKind::I4Grouped { data, scale, group_size } => {
            let ngroups = i.div_ceil(*group_size);
            let sr = &scale[row * ngroups..(row + 1) * ngroups];
            let rb = i.div_ceil(2);
            let wr = &data[row * rb..(row + 1) * rb];
            for k in 0..i {
                let byte = wr[k >> 1];
                let nibble = if k & 1 == 0 { byte & 0xF } else { byte >> 4 };
                acc[k] += coef * (nibble as i32 - 8) as f32 * sr[k / group_size];
            }
        }
        QTKind::MxFp4 { data, block_scale } => {
            let rb = i.div_ceil(2);
            let bpr = i.div_ceil(32);
            let wr = &data[row * rb..(row + 1) * rb];
            let bsr = &block_scale[row * bpr..(row + 1) * bpr];
            for k in 0..i {
                let byte = wr[k >> 1];
                let nibble = if k & 1 == 0 { byte & 0xF } else { byte >> 4 };
                acc[k] += coef * e2m1_decode(nibble) * e8m0_decode(bsr[k / 32]);
            }
        }
    }
}

/// `y[j] = W[r0+j,:] · x` for `j` in `0..n` — a matvec over a row slice of `w`, dequantizing
/// on the fly. Used by the absorbed decode path to apply `W_V` to the attention-weighted
/// latent sum without ever materializing the full dense `V` matrix.
///
/// Accumulates in `f64` (matching `qt_matvec_rows` in `glm.c`) even though every other kernel
/// here uses `f32` — this is the original's own choice, not a rabbit addition, so it's kept
/// for token-exact parity rather than "corrected" to match the rest of the file.
pub fn qt_matvec_rows(w: &QT, r0: usize, n: usize, x: &[f32], y: &mut [f32]) {
    let i = w.cols;
    for (j, yj) in y.iter_mut().enumerate().take(n) {
        let row = r0 + j;
        let a: f64 = match &w.kind {
            QTKind::F32(data) => {
                let wr = &data[row * i..(row + 1) * i];
                wr.iter().zip(x).map(|(&wv, &xv)| wv as f64 * xv as f64).sum()
            }
            QTKind::I8 { data, scale } => {
                let wr = &data[row * i..(row + 1) * i];
                let acc: f32 = wr.iter().zip(x).map(|(&wv, &xv)| wv as f32 * xv).sum();
                acc as f64 * scale[row] as f64
            }
            QTKind::I4 { data, scale } => {
                let rb = i.div_ceil(2);
                let wr = &data[row * rb..(row + 1) * rb];
                #[cfg(target_arch = "x86_64")]
                let acc = if has_avx512_i4() {
                    unsafe { dot_i4_f32_avx512(wr, x, i) }
                } else {
                    dot_i4_f32_scalar(wr, x, i)
                };
                #[cfg(not(target_arch = "x86_64"))]
                let acc = dot_i4_f32_scalar(wr, x, i);
                acc as f64 * scale[row] as f64
            }
            QTKind::I2 { data, scale } => {
                let rb = i.div_ceil(4);
                let wr = &data[row * rb..(row + 1) * rb];
                let mut acc = 0f32;
                for k in 0..i {
                    let byte = wr[k >> 2];
                    let bits = (byte >> ((k & 3) * 2)) & 3;
                    acc += (bits as i32 - 2) as f32 * x[k];
                }
                acc as f64 * scale[row] as f64
            }
            QTKind::I4Grouped { data, scale, group_size } => {
                let ngroups = i.div_ceil(*group_size);
                let sr = &scale[row * ngroups..(row + 1) * ngroups];
                let rb = i.div_ceil(2);
                let wr = &data[row * rb..(row + 1) * rb];
                let mut acc = 0f64;
                for k in 0..i {
                    let byte = wr[k >> 1];
                    let nibble = if k & 1 == 0 { byte & 0xF } else { byte >> 4 };
                    acc += ((nibble as i32 - 8) as f32 * sr[k / group_size]) as f64 * x[k] as f64;
                }
                acc
            }
            QTKind::MxFp4 { data, block_scale } => {
                let rb = i.div_ceil(2);
                let bpr = i.div_ceil(32);
                let wr = &data[row * rb..(row + 1) * rb];
                let bsr = &block_scale[row * bpr..(row + 1) * bpr];
                let mut acc = 0f64;
                for k in 0..i {
                    let byte = wr[k >> 1];
                    let nibble = if k & 1 == 0 { byte & 0xF } else { byte >> 4 };
                    acc += (e2m1_decode(nibble) * e8m0_decode(bsr[k / 32])) as f64 * x[k] as f64;
                }
                acc
            }
        };
        *yj = a as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(seed: &mut u32) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 17;
        *seed ^= *seed << 5;
        ((*seed as f32 / u32::MAX as f32) - 0.5) * 2.0
    }

    fn random_vec(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n).map(|_| xorshift(&mut s)).collect()
    }

    /// The contract `matmul_qt_rows` exists to make (and that K3's Phase 5 v2 dispatch is built
    /// on): slicing a matmul into arbitrary row blocks and reassembling them reproduces
    /// `matmul_qt`'s output **bit for bit**, for every `QTKind`, at `s = 1` (decode, where the
    /// int4 tier takes its float branch) and `s > 1` (prefill, where it takes int8 IDOT).
    ///
    /// `assert_eq!` on the raw f32s, not a tolerance — the whole point is that the two paths run
    /// the same per-row kernels in the same per-element order, so any drift between them (a tier
    /// that stops mirroring `matmul_qt`'s dispatch, a `row_dot_*` helper edited on one side only)
    /// must fail here rather than quietly change a checkpoint's output.
    ///
    /// Dims are deliberately awkward: `cols` not a multiple of 32 or 64 exercises the SIMD tiers'
    /// scalar tails, `rows` prime means the row blocks below don't divide evenly.
    #[test]
    fn matmul_qt_rows_reassembled_is_bit_identical_to_matmul_qt() {
        let (rows, cols) = (37usize, 100usize);
        let w = random_vec(rows * cols, 7);
        let gs = 32;
        let variants: Vec<(&str, QT)> = vec![
            ("f32", {
                let mut t = QT::alloc(rows, cols, 32, false);
                t.fill(&w);
                t
            }),
            ("i8", {
                let mut t = QT::alloc(rows, cols, 8, false);
                t.fill(&w);
                t
            }),
            ("i4", {
                let mut t = QT::alloc(rows, cols, 4, false);
                t.fill(&w);
                t
            }),
            ("i2", {
                let mut t = QT::alloc(rows, cols, 2, false);
                t.fill(&w);
                t
            }),
            ("i4grouped", {
                let mut t = QT::alloc_grouped(rows, cols, 4, gs);
                t.fill(&w);
                t
            }),
            ("mxfp4", {
                let mut t = QT::alloc_mxfp4(rows, cols);
                t.fill(&w);
                t
            }),
        ];

        for s in [1usize, 3] {
            let x = random_vec(s * cols, 91);
            for (name, t) in &variants {
                let mut expected = vec![0f32; s * rows];
                matmul_qt(&mut expected, &x, t, s);

                // Reassemble from row blocks of a size that leaves a short final block.
                let acts = RowActs::prepare(&x, t, s);
                let mut yt = vec![0f32; rows * s];
                let blk = 8;
                for (b, seg) in yt.chunks_mut(blk * s).enumerate() {
                    matmul_qt_rows(seg, &acts, t, b * blk, seg.len() / s);
                }
                let mut got = vec![0f32; s * rows];
                for (oi, col) in yt.chunks(s).enumerate() {
                    for (si, &v) in col.iter().enumerate() {
                        got[si * rows + oi] = v;
                    }
                }
                assert_eq!(got, expected, "{name} at s={s}");
            }
        }
    }

    /// Phase N4's acceptance gate at the unit level: sharding a weight across real pinned
    /// domain pools and computing the matvec as one cross-domain fan-out reproduces `matmul_qt`
    /// **bit for bit**, for every `QTKind` — including a batched (multi-weight, one fan-out)
    /// call and a row count that doesn't divide evenly across domains. Throwaway
    /// `NodePools::build` pools (never the singleton); SKIPs on single-node machines.
    #[test]
    fn qt_sharded_matvec_is_bit_identical_to_matmul_qt() {
        let Some(topo) = crate::numa::topology() else {
            eprintln!("SKIP: no NUMA topology on this machine");
            return;
        };
        let Some(pools) = crate::numa::NodePools::build(topo.nodes, 8) else {
            eprintln!("SKIP: single NUMA node — nothing to shard across");
            return;
        };
        let (rows, cols) = (37usize, 100usize);
        let w = random_vec(rows * cols, 5);
        let x = random_vec(cols, 55);
        let variants: Vec<QT> = vec![
            {
                let mut t = QT::alloc(rows, cols, 32, false);
                t.fill(&w);
                t
            },
            {
                let mut t = QT::alloc(rows, cols, 8, false);
                t.fill(&w);
                t
            },
            {
                let mut t = QT::alloc(rows, cols, 4, false);
                t.fill(&w);
                t
            },
            {
                let mut t = QT::alloc(rows, cols, 2, false);
                t.fill(&w);
                t
            },
            {
                let mut t = QT::alloc_grouped(rows, cols, 4, 32);
                t.fill(&w);
                t
            },
            {
                let mut t = QT::alloc_mxfp4(rows, cols);
                t.fill(&w);
                t
            },
        ];
        let mut sharded = Vec::new();
        for t in variants {
            let mut expected = vec![0f32; rows];
            matmul_qt(&mut expected, &x, &t, 1);
            let s = QTSharded::shard(t, &pools);
            let mut got = vec![0f32; rows];
            s.matvec(&mut got, &x, &pools);
            let a: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            let b: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
            assert_eq!(a, b, "sharded matvec must be bit-identical to matmul_qt");
            sharded.push((s, expected));
        }
        // Batched: all six weights in ONE fan-out, same bits.
        let mut outs: Vec<Vec<f32>> = sharded.iter().map(|(s, _)| vec![0f32; s.rows]).collect();
        {
            let mut jobs: Vec<(&mut [f32], &QTSharded)> = outs.iter_mut().map(|o| o.as_mut_slice()).zip(sharded.iter().map(|(s, _)| s)).collect();
            matvec_sharded_batch(&pools, &x, &mut jobs);
        }
        for (out, (_, expected)) in outs.iter().zip(&sharded) {
            let a: Vec<u32> = out.iter().map(|v| v.to_bits()).collect();
            let b: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
            assert_eq!(a, b, "batched sharded matvec must be bit-identical too");
        }
    }

    /// `RowActs` keeps `x` even when it prepared an int8 copy, so an `acts` built against the
    /// wrong weight still computes the right answer (it re-quantizes inline) instead of silently
    /// using a mismatched activation — the fallback documented on `RowActs`. Cheap insurance
    /// against a caller that reuses one `acts` across differently-quantized weights.
    #[test]
    fn matmul_qt_rows_is_still_correct_when_acts_were_prepared_for_another_weight_kind() {
        let (rows, cols, s) = (6usize, 64usize, 2usize);
        let w = random_vec(rows * cols, 13);
        let x = random_vec(s * cols, 17);
        let mut i8t = QT::alloc(rows, cols, 8, false);
        i8t.fill(&w);
        let mut f32t = QT::alloc(rows, cols, 32, false);
        f32t.fill(&w);

        let mut expected = vec![0f32; s * rows];
        matmul_qt(&mut expected, &x, &i8t, s);

        // Prepared against the f32 weight (so: no int8 copy), used against the int8 one.
        let mismatched = RowActs::prepare(&x, &f32t, s);
        let mut yt = vec![0f32; rows * s];
        matmul_qt_rows(&mut yt, &mismatched, &i8t, 0, rows);
        for (oi, col) in yt.chunks(s).enumerate() {
            for (si, &v) in col.iter().enumerate() {
                assert_eq!(v, expected[si * rows + oi], "row {oi}, seq {si}");
            }
        }
    }

    #[test]
    fn matmul_matches_hand_computed_dot_products() {
        // x: [S=2, I=3], W: [O=2, I=3]
        let x = [1.0, 2.0, 3.0, -1.0, 0.5, 2.0];
        let w = [1.0, 0.0, 0.0, 0.0, 1.0, 1.0];
        let mut y = [0.0; 4];
        matmul(&mut y, &x, &w, 2, 3, 2);
        assert_eq!(y, [1.0, 5.0, -1.0, 2.5]);
    }

    #[test]
    fn matmul_q_dequantizes_exactly_for_integer_valued_weights() {
        // amax=127=qmax -> scale=1.0 exactly, so these small integers quantize losslessly.
        let w = [10.0, -20.0, 30.0, 127.0];
        let mut t = crate::quant::QT::alloc(1, 4, 8, false);
        t.fill(&w);
        let (data, scale) = match &t.kind {
            QTKind::I8 { data, scale } => (data.clone(), scale.clone()),
            _ => panic!("expected I8"),
        };
        assert_eq!(scale[0], 1.0);
        let x = [1.0, 1.0, 1.0, 1.0];
        let mut y = [0.0; 1];
        matmul_q(&mut y, &x, &data, &scale, 1, 4, 1);
        let expected: f32 = w.iter().sum();
        assert_eq!(y[0], expected);
    }

    #[test]
    fn matmul_i4_matches_manual_nibble_unpack() {
        // 5 columns: 2 full byte-pairs + 1 tail nibble.
        let vals = [3i32, -7, 0, 5, -2];
        let mut bytes = vec![0u8; vals.len().div_ceil(2)];
        for (idx, &v) in vals.iter().enumerate() {
            let nibble = (v + 8) as u8;
            if idx % 2 == 0 {
                bytes[idx / 2] |= nibble;
            } else {
                bytes[idx / 2] |= nibble << 4;
            }
        }
        let scale = [2.0f32];
        let x = [1.0, 1.0, 1.0, 1.0, 1.0];
        let mut y = [0.0; 1];
        matmul_i4(&mut y, &x, &bytes, &scale, 1, 5, 1);
        let expected: f32 = vals.iter().sum::<i32>() as f32 * 2.0;
        assert_eq!(y[0], expected);
    }

    #[test]
    fn matmul_i2_matches_manual_nibble_unpack() {
        // 5 columns: values -2,1,0,-1,1 packed 4/byte, low bits first.
        let vals = [-2i32, 1, 0, -1, 1];
        let mut bytes = [0u8; 2];
        for (idx, &v) in vals.iter().enumerate() {
            bytes[idx / 4] |= ((v + 2) as u8) << ((idx % 4) * 2);
        }
        let scale = [3.0f32];
        let x = [1.0, 1.0, 1.0, 1.0, 1.0];
        let mut y = [0.0; 1];
        matmul_i2(&mut y, &x, &bytes, &scale, 1, 5, 1);
        let expected: f32 = vals.iter().sum::<i32>() as f32 * 3.0;
        assert_eq!(y[0], expected);
    }

    #[test]
    fn matmul_mxfp4_matches_manual_e2m1_decode_across_two_blocks() {
        // 40 columns -> block 0 (0..32) at code 4 (=2.0), block 1 (32..40, the tail block) at
        // code 7|sign (=-6.0) — two different E8M0 scales, one per block, in the SAME row.
        let cols: usize = 40;
        let mut data = vec![0u8; cols.div_ceil(2)];
        for byte in data[0..16].iter_mut() {
            *byte = 4 | (4 << 4); // both nibbles in block 0: code 4
        }
        for byte in data[16..20].iter_mut() {
            *byte = (7 | 0x8) | ((7 | 0x8) << 4); // both nibbles in block 1: code 7|sign
        }
        let block_scale = [127u8, 130u8]; // scale=1.0, scale=8.0
        let x = vec![1.0f32; cols];
        let mut y = [0.0f32; 1];
        matmul_mxfp4(&mut y, &x, &data, &block_scale, 1, cols, 1);
        // block 0: 32 * (2.0 * 1.0) = 64.0; block 1: 8 * (-6.0 * 8.0) = -384.0
        assert_eq!(y[0], 32.0 * 2.0 + 8.0 * -48.0);
    }

    /// Phase 2's block-structured `matmul_mxfp4` must be **bit-identical** to the pre-Phase-2
    /// element-at-a-time loop (`e8m0_decode` per element, `k & 1` nibble branch) — same inputs to
    /// the same decode functions in the same per-element order, only the loop shape changed.
    /// Verified across sizes including `i` not a multiple of 32/64 and odd `o`, at s=1 and s>1,
    /// comparing raw f32 bits (not an epsilon) since the claim is exactness, not closeness.
    #[test]
    fn matmul_mxfp4_matches_the_pre_block_reference() {
        // A straight port of the pre-Phase-2 inner loop, kept here as the exactness oracle.
        fn reference(y: &mut [f32], x: &[f32], data: &[u8], block_scale: &[u8], s: usize, i: usize, o: usize) {
            let rb = i.div_ceil(2);
            let bpr = i.div_ceil(32);
            for oi in 0..o {
                let w = &data[oi * rb..(oi + 1) * rb];
                let bs = &block_scale[oi * bpr..(oi + 1) * bpr];
                for si in 0..s {
                    let xs = &x[si * i..(si + 1) * i];
                    let mut a = 0f32;
                    for k in 0..i {
                        let byte = w[k >> 1];
                        let nibble = if k & 1 == 0 { byte & 0xF } else { byte >> 4 };
                        a += xs[k] * e2m1_decode(nibble) * e8m0_decode(bs[k / 32]);
                    }
                    y[si * o + oi] = a;
                }
            }
        }

        let mut seed = 0x1234_5678u32;
        let rnd = |seed: &mut u32| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 17;
            *seed ^= *seed << 5;
            *seed
        };
        // (s, i, o): i covers multiples of 32, non-multiples, odd, <32 tail; o includes odd rows.
        for &(s, i, o) in &[(1usize, 32usize, 1usize), (1, 40, 3), (8, 96, 5), (1, 33, 7), (4, 3584, 6), (2, 3072, 4), (1, 17, 2), (3, 64, 1)] {
            let rb = i.div_ceil(2);
            let bpr = i.div_ceil(32);
            let data: Vec<u8> = (0..o * rb).map(|_| (rnd(&mut seed) & 0xFF) as u8).collect();
            // E8M0 bytes in [118,136] keep 2^(b-127) a modest finite factor (no inf poisoning).
            let scale: Vec<u8> = (0..o * bpr).map(|_| 118 + (rnd(&mut seed) % 19) as u8).collect();
            let x: Vec<f32> = (0..s * i).map(|_| (rnd(&mut seed) as f32 / u32::MAX as f32 - 0.5) * 2.0).collect();

            let mut y_new = vec![0f32; s * o];
            let mut y_ref = vec![0f32; s * o];
            // Targets the SCALAR tier explicitly: `matmul_mxfp4` dispatches to the AVX-512 tier on
            // capable CPUs, which is within-tolerance (reassociated), not bit-identical.
            matmul_mxfp4_scalar(&mut y_new, &x, &data, &scale, s, i, o);
            reference(&mut y_ref, &x, &data, &scale, s, i, o);
            for (a, b) in y_new.iter().zip(&y_ref) {
                assert_eq!(a.to_bits(), b.to_bits(), "mismatch at (s={s}, i={i}, o={o}): {a} vs {b}");
            }
        }
    }

    /// Phase 3's AVX-512 MXFP4 tier vs the scalar tier — **within-tolerance, not bit-exact**: the
    /// tier folds the E8M0 scale into each block's decoded weights and reduces two FMA chains via a
    /// tree-sum, reassociating the scalar `x*e2m1*scale` running sum (same tradeoff `matmul_i4`'s
    /// AVX-512 tier's tests accept). Random weights/scales/activations across dims including `i` not
    /// a multiple of 32/64 (partial tail block) and odd `o`. Tolerance is relative-with-floor at
    /// 2e-3 — generous next to the sub-1e-4 relative error reassociation actually produces here, but
    /// robust to the magnitudes larger `i` and E8M0 scale bytes create.
    #[test]
    fn matmul_mxfp4_avx512_matches_scalar_within_tolerance() {
        if !(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")) {
            eprintln!("SKIP: no AVX-512F/BW on this CPU");
            return;
        }
        let mut seed = 0x9E37_79B1u32;
        let rnd = |seed: &mut u32| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 17;
            *seed ^= *seed << 5;
            *seed
        };
        for &(s, i, o) in &[(1usize, 32usize, 4usize), (1, 64, 3), (8, 96, 5), (1, 40, 7), (2, 3584, 4), (1, 3072, 6), (1, 33, 2), (4, 128, 1)] {
            let rb = i.div_ceil(2);
            let bpr = i.div_ceil(32);
            let data: Vec<u8> = (0..o * rb).map(|_| (rnd(&mut seed) & 0xFF) as u8).collect();
            let scale: Vec<u8> = (0..o * bpr).map(|_| 122 + (rnd(&mut seed) % 11) as u8).collect(); // [122,132]
            let x: Vec<f32> = (0..s * i).map(|_| (rnd(&mut seed) as f32 / u32::MAX as f32 - 0.5) * 2.0).collect();
            let mut y_scalar = vec![0f32; s * o];
            let mut y_avx = vec![0f32; s * o];
            matmul_mxfp4_scalar(&mut y_scalar, &x, &data, &scale, s, i, o);
            unsafe { matmul_mxfp4_avx512(&mut y_avx, &x, &data, &scale, s, i, o) };
            for (a, b) in y_avx.iter().zip(&y_scalar) {
                let tol = 2e-3 * (1.0 + b.abs());
                assert!((a - b).abs() <= tol, "s={s} i={i} o={o}: {a} vs {b} (tol {tol})");
            }
        }
    }

    #[test]
    fn matmul_i4_grouped_matches_manual_nibble_unpack_across_two_groups() {
        // 6 columns, group_size=3 -> group 0 = cols[0..3] (scale 2.0), group 1 = cols[3..6]
        // (scale 5.0). Values chosen so each group's contribution is easy to hand-verify.
        let vals = [1i32, -1, 2, 3, -2, 0]; // group0: 1,-1,2 ; group1: 3,-2,0
        let mut bytes = vec![0u8; vals.len().div_ceil(2)];
        for (idx, &v) in vals.iter().enumerate() {
            let nibble = (v + 8) as u8;
            if idx % 2 == 0 {
                bytes[idx / 2] |= nibble;
            } else {
                bytes[idx / 2] |= nibble << 4;
            }
        }
        let scale = [2.0f32, 5.0f32];
        let x = [1.0f32; 6];
        let mut y = [0.0f32; 1];
        matmul_i4_grouped(&mut y, &x, &bytes, &scale, 3, 1, 6, 1);
        // group0: (1-1+2)*2.0 = 4.0 ; group1: (3-2+0)*5.0 = 5.0
        assert_eq!(y[0], 4.0 + 5.0);
    }

    #[test]
    fn matmul_qt_i4_grouped_matches_manual_dequant_dot_product() {
        let rows = 5;
        let cols = 70; // group_size=16 -> 4 groups/row (16,16,16,22... wait 70/16=4.375 -> 5 groups), a deliberately non-multiple case
        let gs = 16;
        let s = 2;
        let w = random_vec(rows * cols, 41);
        let x = random_vec(s * cols, 42);
        let mut t = QT::alloc_grouped(rows, cols, 4, gs);
        t.fill(&w);

        let mut expected = vec![0.0f32; s * rows];
        for si in 0..s {
            for oi in 0..rows {
                let wr = t.row_f32(oi);
                let xr = &x[si * cols..(si + 1) * cols];
                expected[si * rows + oi] = wr.iter().zip(xr).map(|(&a, &b)| a * b).sum();
            }
        }
        let mut y = vec![0.0f32; s * rows];
        matmul_qt(&mut y, &x, &t, s);
        for (a, b) in y.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn axpy_i4_f32_avx512_matches_scalar_bit_exact() {
        if !(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")) {
            eprintln!("SKIP: no AVX-512F/BW on this CPU");
            return;
        }
        // include a case past the first 32-wide chunk (33) so the tail-handling path (both the
        // 2-at-a-time scalar tail and the final odd element) is actually exercised.
        for &n in &[1usize, 7, 16, 17, 32, 33, 65] {
            let rows = 1;
            let w = random_vec(rows * n, n as u32 * 3 + 9);
            let mut t = QT::alloc(rows, n, 4, false);
            t.fill(&w);
            let data = match &t.kind {
                QTKind::I4 { data, .. } => data.clone(),
                _ => panic!("expected I4"),
            };
            let coef = 1.7_f32;

            let mut acc_scalar = vec![0.0f32; n];
            axpy_i4_f32_scalar(&data, coef, &mut acc_scalar, n);
            let mut acc_avx512 = vec![0.0f32; n];
            unsafe { axpy_i4_f32_avx512(&data, coef, &mut acc_avx512, n) };
            assert_eq!(acc_scalar, acc_avx512, "n={n}");
        }
    }

    #[test]
    fn qt_addrow_i4_dispatch_matches_scalar_reference_bit_exact() {
        // Exercises qt_addrow's real dispatch (whichever tier this CPU actually picks) against
        // the SAME scalar logic the AVX-512 arm falls back to — proving the bit-identical claim
        // holds end-to-end through the public API, not just for the low-level kernel in
        // isolation (see `axpy_i4_f32_avx512_matches_scalar_bit_exact` above).
        let (rows, cols) = (3, 40);
        let w = random_vec(rows * cols, 51);
        let mut t = QT::alloc(rows, cols, 4, false);
        t.fill(&w);
        let (data, scale) = match &t.kind {
            QTKind::I4 { data, scale } => (data.clone(), scale.clone()),
            _ => panic!("expected I4"),
        };
        let coef = 2.3_f32;
        let row = 1;

        let mut acc = vec![0.0f32; cols];
        qt_addrow(&t, row, coef, &mut acc);

        let rb = cols.div_ceil(2);
        let wr = &data[row * rb..(row + 1) * rb];
        let c = coef * scale[row];
        let mut expected = vec![0.0f32; cols];
        axpy_i4_f32_scalar(wr, c, &mut expected, cols);
        assert_eq!(acc, expected);
    }

    #[test]
    fn qt_matvec_rows_i4_dispatch_matches_scalar_reference_within_tolerance() {
        // Not bit-exact when this CPU picks the AVX-512 tier (dot_i4_f32_avx512's tree-sum
        // reduction reorders vs the scalar loop, same accepted tradeoff as matmul_i4's own
        // AVX-512 tier) — tolerance-based, matching this file's precedent for that kernel.
        let (rows, cols) = (4, 48);
        let w = random_vec(rows * cols, 52);
        let mut t = QT::alloc(rows, cols, 4, false);
        t.fill(&w);
        let x = random_vec(cols, 53);

        let mut y = vec![0.0f32; 2];
        qt_matvec_rows(&t, 1, 2, &x, &mut y);

        let (data, scale) = match &t.kind {
            QTKind::I4 { data, scale } => (data.clone(), scale.clone()),
            _ => panic!("expected I4"),
        };
        let rb = cols.div_ceil(2);
        for (j, &yj) in y.iter().enumerate() {
            let row = 1 + j;
            let wr = &data[row * rb..(row + 1) * rb];
            let dot = dot_i4_f32_scalar(wr, &x, cols);
            let expected = dot as f64 * scale[row] as f64;
            assert!((yj as f64 - expected).abs() < 1e-3, "{yj} vs {expected}");
        }
    }

    #[test]
    fn qt_addrow_i4_grouped_matches_row_f32() {
        let (rows, cols, gs) = (3, 40, 8);
        let w = random_vec(rows * cols, 43);
        let mut t = QT::alloc_grouped(rows, cols, 4, gs);
        t.fill(&w);

        let mut acc = vec![0.0f32; cols];
        qt_addrow(&t, 1, 3.0, &mut acc);
        let expected: Vec<f32> = t.row_f32(1).iter().map(|&v| v * 3.0).collect();
        for (a, b) in acc.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn qt_matvec_rows_i4_grouped_matches_row_f32_dot_product() {
        let (rows, cols, gs) = (4, 24, 8);
        let w = random_vec(rows * cols, 44);
        let mut t = QT::alloc_grouped(rows, cols, 4, gs);
        t.fill(&w);
        let x = random_vec(cols, 45);

        let mut y = vec![0.0f32; 2];
        qt_matvec_rows(&t, 1, 2, &x, &mut y);
        for (j, &yj) in y.iter().enumerate() {
            let expected: f32 = t.row_f32(1 + j).iter().zip(&x).map(|(&a, &b)| a * b).sum();
            assert!((yj - expected).abs() < 1e-2, "{yj} vs {expected}");
        }
    }

    #[test]
    fn matmul_qt_mxfp4_matches_manual_dequant_dot_product() {
        let rows = 5;
        let cols = 96; // 3 full blocks/row
        let s = 2;
        let w = random_vec(rows * cols, 31);
        let x = random_vec(s * cols, 32);
        let mut t = QT::alloc_mxfp4(rows, cols);
        t.fill(&w);

        let mut expected = vec![0.0f32; s * rows];
        for si in 0..s {
            for oi in 0..rows {
                let wr = t.row_f32(oi);
                let xr = &x[si * cols..(si + 1) * cols];
                expected[si * rows + oi] = wr.iter().zip(xr).map(|(&a, &b)| a * b).sum();
            }
        }
        let mut y = vec![0.0f32; s * rows];
        matmul_qt(&mut y, &x, &t, s);
        // Was `assert_eq!` when the MxFp4 path was scalar-only. Phase 3 gave `matmul_mxfp4` an
        // AVX-512 tier that `matmul_qt` dispatches to here; it reassociates the per-row sum
        // (scale-fold + dual-chain tree reduce), so this is now WITHIN-TOLERANCE, exactly like this
        // file's `matmul_qt` int4 test — whose i4 path has had an AVX-512 tier all along.
        for (a, b) in y.iter().zip(&expected) {
            assert!((a - b).abs() <= 1e-3 * (1.0 + b.abs()), "{a} vs {b}");
        }
    }

    #[test]
    fn dot_i8i8_matches_naive_sum() {
        let w: Vec<i8> = (0..40).map(|i| ((i % 7) - 3) as i8).collect();
        let x: Vec<i8> = (0..40).map(|i| ((i % 5) - 2) as i8).collect();
        let expected: i32 = w.iter().zip(&x).map(|(&a, &b)| a as i32 * b as i32).sum();
        assert_eq!(dot_i8i8(&w, &x, 40), expected);
    }

    #[test]
    fn dot_i4i8_matches_naive_sum() {
        let vals: Vec<i32> = (0..41).map(|i| (i % 16) - 8).collect(); // odd length -> tail nibble
        let mut w4 = vec![0u8; vals.len().div_ceil(2)];
        for (idx, &v) in vals.iter().enumerate() {
            let byte = (v + 8) as u8;
            if idx % 2 == 0 {
                w4[idx / 2] |= byte;
            } else {
                w4[idx / 2] |= byte << 4;
            }
        }
        let x: Vec<i8> = (0..vals.len()).map(|i| ((i as i32 % 5) - 2) as i8).collect();
        let expected: i32 = vals.iter().zip(&x).map(|(&a, &b)| a * b as i32).sum();
        assert_eq!(dot_i4i8(&w4, &x, vals.len()), expected);
    }

    #[test]
    fn qrow_i8_round_trips_close_to_original() {
        let x = random_vec(64, 5);
        let mut q = vec![0i8; 64];
        let s = qrow_i8(&x, &mut q);
        for (i, &xi) in x.iter().enumerate() {
            let dq = q[i] as f32 * s;
            assert!((dq - xi).abs() < 0.02, "index {i}: {dq} vs {xi}");
        }
    }

    #[test]
    fn matmul_qt_f32_matches_plain_matmul() {
        let rows = 6;
        let cols = 9;
        let s = 3;
        let w = random_vec(rows * cols, 11);
        let x = random_vec(s * cols, 12);
        let mut t = QT::alloc(rows, cols, 32, false);
        t.fill(&w);

        let mut y_direct = vec![0.0; s * rows];
        matmul(&mut y_direct, &x, &w, s, cols, rows);
        let mut y_qt = vec![0.0; s * rows];
        matmul_qt(&mut y_qt, &x, &t, s);
        assert_eq!(y_direct, y_qt);
    }

    #[test]
    fn matmul_qt_i8_idot_is_close_to_direct_matmul_q() {
        let rows = 4;
        let cols = 16;
        let s = 2;
        let w = random_vec(rows * cols, 21);
        let x = random_vec(s * cols, 22);
        let mut t = QT::alloc(rows, cols, 8, false);
        t.fill(&w);
        let (data, scale) = match &t.kind {
            QTKind::I8 { data, scale } => (data.clone(), scale.clone()),
            _ => panic!("expected I8"),
        };

        let mut y_direct = vec![0.0; s * rows];
        matmul_q(&mut y_direct, &x, &data, &scale, s, cols, rows);
        let mut y_qt = vec![0.0; s * rows];
        matmul_qt(&mut y_qt, &x, &t, s);
        // IDOT additionally quantizes activations to int8 -> small extra error, not exact.
        for (a, b) in y_direct.iter().zip(&y_qt) {
            assert!((a - b).abs() < 0.05, "{a} vs {b}");
        }
    }

    #[test]
    fn matmul_qt_i4_gates_idot_by_seq_len() {
        let rows = 3;
        let cols = 10;
        let w = random_vec(rows * cols, 31);
        let mut t = QT::alloc(rows, cols, 4, false);
        t.fill(&w);
        let (data, scale) = match &t.kind {
            QTKind::I4 { data, scale } => (data.clone(), scale.clone()),
            _ => panic!("expected I4"),
        };

        // S=1: below I4_IDOT_MIN_S, must be bit-identical to the direct float-weight path.
        let x1 = random_vec(cols, 32);
        let mut y1_direct = vec![0.0; rows];
        matmul_i4(&mut y1_direct, &x1, &data, &scale, 1, cols, rows);
        let mut y1_qt = vec![0.0; rows];
        matmul_qt(&mut y1_qt, &x1, &t, 1);
        assert_eq!(y1_direct, y1_qt);
    }

    // ---- SIMD tier parity: scalar vs AVX2 vs AVX-512/VNNI ----
    //
    // dot_i8i8/dot_i4i8 are pure integer accumulation, so every tier must agree bit-for-bit
    // (no floating-point reassociation to blur the comparison) — that's the "cuantización
    // entera es exacta" property Fase 7 exists to preserve. matmul_q/i4/i2 dequantize into
    // f32 and accumulate with FMA, so scalar and AVX2 can differ by a ULP or two from
    // reassociation; those use a small tolerance instead of `assert_eq!`.
    //
    // Lengths are deliberately NOT round multiples of any SIMD width (32/64 for the int
    // kernels, 8/16 for the float ones), so every test also exercises the scalar tail loop.

    fn random_i8(n: usize, seed: u32) -> Vec<i8> {
        let mut s = seed;
        (0..n).map(|_| (xorshift(&mut s) * 127.0) as i8).collect()
    }

    fn random_packed_nibbles(n_logical: usize, seed: u32) -> Vec<u8> {
        let mut s = seed;
        let vals: Vec<i32> = (0..n_logical).map(|_| ((xorshift(&mut s) * 8.0) as i32).clamp(-8, 7)).collect();
        let mut packed = vec![0u8; n_logical.div_ceil(2)];
        for (idx, &v) in vals.iter().enumerate() {
            let nibble = (v + 8) as u8;
            if idx % 2 == 0 {
                packed[idx / 2] |= nibble;
            } else {
                packed[idx / 2] |= nibble << 4;
            }
        }
        packed
    }

    #[test]
    fn dot_i8i8_avx2_matches_scalar_bit_exact() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("SKIP: no AVX2 on this CPU");
            return;
        }
        for &n in &[1usize, 31, 32, 37, 64, 100, 257] {
            let w = random_i8(n, n as u32 * 7 + 1);
            let x = random_i8(n, n as u32 * 13 + 2);
            let expected = dot_i8i8_scalar(&w, &x, n);
            let got = unsafe { dot_i8i8_avx2(&w, &x, n) };
            assert_eq!(got, expected, "n={n}");
        }
    }

    #[test]
    fn dot_i8i8_avx512vnni_matches_scalar_bit_exact() {
        if !(is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512bw")) {
            eprintln!("SKIP: no AVX-512 VNNI on this CPU");
            return;
        }
        for &n in &[1usize, 63, 64, 70, 128, 200, 513] {
            let w = random_i8(n, n as u32 * 7 + 3);
            let x = random_i8(n, n as u32 * 13 + 4);
            let expected = dot_i8i8_scalar(&w, &x, n);
            let got = unsafe { dot_i8i8_avx512vnni(&w, &x, n) };
            assert_eq!(got, expected, "n={n}");
        }
    }

    #[test]
    fn dot_i4i8_avx2_matches_scalar_bit_exact() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("SKIP: no AVX2 on this CPU");
            return;
        }
        for &n in &[1usize, 31, 32, 37, 64, 100, 257] {
            let w4 = random_packed_nibbles(n, n as u32 * 5 + 1);
            let x = random_i8(n, n as u32 * 11 + 2);
            let expected = dot_i4i8_scalar(&w4, &x, n);
            let got = unsafe { dot_i4i8_avx2(&w4, &x, n) };
            assert_eq!(got, expected, "n={n}");
        }
    }

    #[test]
    fn dot_i4i8_avx512vnni_matches_scalar_bit_exact() {
        if !(is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512bw")) {
            eprintln!("SKIP: no AVX-512 VNNI on this CPU");
            return;
        }
        for &n in &[1usize, 63, 64, 70, 128, 200, 513] {
            let w4 = random_packed_nibbles(n, n as u32 * 5 + 5);
            let x = random_i8(n, n as u32 * 11 + 6);
            let expected = dot_i4i8_scalar(&w4, &x, n);
            let got = unsafe { dot_i4i8_avx512vnni(&w4, &x, n) };
            assert_eq!(got, expected, "n={n}");
        }
    }

    #[test]
    fn matmul_q_avx2_matches_scalar_within_tolerance() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("SKIP: no AVX2 on this CPU");
            return;
        }
        let rows = 3;
        for &cols in &[1usize, 7, 8, 15, 33] {
            let w = random_vec(rows * cols, cols as u32 * 3 + 1);
            let mut t = QT::alloc(rows, cols, 8, false);
            t.fill(&w);
            let (data, scale) = match &t.kind {
                QTKind::I8 { data, scale } => (data.clone(), scale.clone()),
                _ => panic!("expected I8"),
            };
            let x = random_vec(cols, cols as u32 * 7 + 2);

            let mut y_scalar = vec![0.0; rows];
            matmul_q_scalar(&mut y_scalar, &x, &data, &scale, 1, cols, rows);
            let mut y_avx2 = vec![0.0; rows];
            unsafe { matmul_q_avx2(&mut y_avx2, &x, &data, &scale, 1, cols, rows) };
            for (a, b) in y_scalar.iter().zip(&y_avx2) {
                assert!((a - b).abs() < 1e-4, "cols={cols}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn matmul_i4_avx2_matches_scalar_within_tolerance() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("SKIP: no AVX2 on this CPU");
            return;
        }
        let rows = 3;
        for &cols in &[1usize, 7, 16, 17, 33] {
            let w = random_vec(rows * cols, cols as u32 * 3 + 3);
            let mut t = QT::alloc(rows, cols, 4, false);
            t.fill(&w);
            let (data, scale) = match &t.kind {
                QTKind::I4 { data, scale } => (data.clone(), scale.clone()),
                _ => panic!("expected I4"),
            };
            let x = random_vec(cols, cols as u32 * 7 + 4);

            let mut y_scalar = vec![0.0; rows];
            matmul_i4_scalar(&mut y_scalar, &x, &data, &scale, 1, cols, rows);
            let mut y_avx2 = vec![0.0; rows];
            unsafe { matmul_i4_avx2(&mut y_avx2, &x, &data, &scale, 1, cols, rows) };
            for (a, b) in y_scalar.iter().zip(&y_avx2) {
                assert!((a - b).abs() < 1e-4, "cols={cols}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn matmul_i4_avx512_matches_scalar_within_tolerance() {
        if !(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")) {
            eprintln!("SKIP: no AVX-512F/BW on this CPU");
            return;
        }
        let rows = 3;
        // include a case past the first 32-wide chunk (33) so the tail-handling path (both the
        // 2-at-a-time scalar tail and the final odd element) is actually exercised.
        for &cols in &[1usize, 7, 16, 17, 32, 33, 65] {
            let w = random_vec(rows * cols, cols as u32 * 3 + 3);
            let mut t = QT::alloc(rows, cols, 4, false);
            t.fill(&w);
            let (data, scale) = match &t.kind {
                QTKind::I4 { data, scale } => (data.clone(), scale.clone()),
                _ => panic!("expected I4"),
            };
            let x = random_vec(cols, cols as u32 * 7 + 4);

            let mut y_scalar = vec![0.0; rows];
            matmul_i4_scalar(&mut y_scalar, &x, &data, &scale, 1, cols, rows);
            let mut y_avx512 = vec![0.0; rows];
            unsafe { matmul_i4_avx512(&mut y_avx512, &x, &data, &scale, 1, cols, rows) };
            for (a, b) in y_scalar.iter().zip(&y_avx512) {
                assert!((a - b).abs() < 1e-4, "cols={cols}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn matmul_i4_avx512_matches_avx2_within_tolerance() {
        if !(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")) {
            eprintln!("SKIP: no AVX-512F/BW on this CPU");
            return;
        }
        if !is_x86_feature_detected!("avx2") {
            eprintln!("SKIP: no AVX2 on this CPU");
            return;
        }
        // Both tiers are non-bit-exact vs scalar (different reassociation order) but should
        // still land close to each other, not just close to scalar independently — same
        // lossless nibble math, just accumulated in a different order.
        let rows = 3;
        for &cols in &[1usize, 7, 16, 17, 32, 33, 65] {
            let w = random_vec(rows * cols, cols as u32 * 3 + 3);
            let mut t = QT::alloc(rows, cols, 4, false);
            t.fill(&w);
            let (data, scale) = match &t.kind {
                QTKind::I4 { data, scale } => (data.clone(), scale.clone()),
                _ => panic!("expected I4"),
            };
            let x = random_vec(cols, cols as u32 * 7 + 4);

            let mut y_avx2 = vec![0.0; rows];
            unsafe { matmul_i4_avx2(&mut y_avx2, &x, &data, &scale, 1, cols, rows) };
            let mut y_avx512 = vec![0.0; rows];
            unsafe { matmul_i4_avx512(&mut y_avx512, &x, &data, &scale, 1, cols, rows) };
            for (a, b) in y_avx2.iter().zip(&y_avx512) {
                assert!((a - b).abs() < 1e-4, "cols={cols}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn matmul_i2_avx2_matches_scalar_within_tolerance() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("SKIP: no AVX2 on this CPU");
            return;
        }
        let rows = 3;
        for &cols in &[1usize, 7, 16, 17, 33] {
            let w = random_vec(rows * cols, cols as u32 * 3 + 5);
            let mut t = QT::alloc(rows, cols, 2, false);
            t.fill(&w);
            let (data, scale) = match &t.kind {
                QTKind::I2 { data, scale } => (data.clone(), scale.clone()),
                _ => panic!("expected I2"),
            };
            let x = random_vec(cols, cols as u32 * 7 + 6);

            let mut y_scalar = vec![0.0; rows];
            matmul_i2_scalar(&mut y_scalar, &x, &data, &scale, 1, cols, rows);
            let mut y_avx2 = vec![0.0; rows];
            unsafe { matmul_i2_avx2(&mut y_avx2, &x, &data, &scale, 1, cols, rows) };
            for (a, b) in y_scalar.iter().zip(&y_avx2) {
                assert!((a - b).abs() < 1e-4, "cols={cols}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn active_dot_kernel_reports_a_known_tier() {
        assert!(["avx512-vnni", "avx2", "scalar"].contains(&active_dot_kernel()));
        // this dev machine (per rabbit-plan.md, Ryzen AI 9 HX 370) has full AVX-512/VNNI.
        if is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512bw") {
            assert_eq!(active_dot_kernel(), "avx512-vnni");
        }
    }
}
