//! Port of `glm.c`'s matmul family. Fase 3 built the scalar baseline; this phase adds AVX2 and
//! AVX-512/VNNI tiers on top, selected at runtime (`is_x86_feature_detected!`) — same idea as
//! the C's `g_i4s`/`IDOT_KERNEL` picking a kernel by measured hardware, not compile-time. Every
//! `*_scalar` function is the untouched Fase 3 implementation; the public names
//! (`matmul_q`/`matmul_i4`/`matmul_i2`/`dot_i8i8`/`dot_i4i8`) are now dispatchers.
//!
//! Tier ladder, matching the C exactly: `matmul_q`/`matmul_i4`/`matmul_i2` (float-weight
//! dequant-and-FMA path) get scalar/AVX2 only — the original never added an AVX-512 tier for
//! them. `dot_i8i8`/`dot_i4i8` (the integer IDOT path) get scalar/AVX2/AVX-512-VNNI: pure
//! integer accumulation, so unlike the float path there's no reassociation to worry about —
//! every tier must agree bit-for-bit, which is exactly what this module's parity tests check.
//!
//! `y[S,O] = x[S,I] @ W^T` throughout, `W` given in one of the `QT` formats from `quant.rs`.
//! The IDOT kernels additionally quantize activations to int8 per row (`qrow_i8`, scalar only
//! in the original — never vectorized there, so not here either) so the whole dot product
//! runs in integer arithmetic — colibri measures this at ~2-3x over the float-weight path, at
//! ~0.3% added RMS error per matmul from the activation quantization.

use crate::quant::{QT, QTKind};
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Every `matmul_*` below parallelizes over output rows (`oi`, 0..O) with rayon — same axis
/// colibri's `#pragma omp parallel for` picks for its C matmul kernels, and the natural one:
/// each `oi` reads the same `x`/activations but a disjoint row of the weight matrix, so there's
/// no cross-row dependency. The catch is `y`'s own layout (`y[si*O+oi]`, row-major by sequence
/// position): a fixed `oi` touches `S` elements strided by `O`, which safe Rust can't split into
/// disjoint mutable chunks across threads. `yt`'s `[O,S]` layout (row-major by output index)
/// makes each `oi`'s slice contiguous instead — exactly what `par_chunks_mut(s)` needs — at the
/// cost of one sequential transpose back into `y` afterward. That transpose is O(S*O); the
/// matmul itself is O(S*O*I), so for any realistic `I` (hidden/intermediate dims in the
/// thousands) the transpose is noise.
fn transpose_so(y: &mut [f32], yt: &[f32], s: usize, o: usize) {
    for oi in 0..o {
        for si in 0..s {
            y[si * o + oi] = yt[oi * s + si];
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn has_avx2() -> bool {
    is_x86_feature_detected!("avx2")
}
#[cfg(not(target_arch = "x86_64"))]
fn has_avx2() -> bool {
    false
}

/// y[S,O] = x[S,I] @ W^T, W[O,I] f32.
pub fn matmul(y: &mut [f32], x: &[f32], w: &[f32], s: usize, i: usize, o: usize) {
    let mut yt = vec![0f32; o * s];
    yt.par_chunks_mut(s).enumerate().for_each(|(oi, row)| {
        let wr = &w[oi * i..(oi + 1) * i];
        for (si, slot) in row.iter_mut().enumerate() {
            let xs = &x[si * i..(si + 1) * i];
            *slot = xs.iter().zip(wr).map(|(a, b)| a * b).sum();
        }
    });
    transpose_so(y, &yt, s, o);
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
    let mut yt = vec![0f32; o * s];
    yt.par_chunks_mut(s).enumerate().for_each(|(oi, row)| {
        let w = &q[oi * i..(oi + 1) * i];
        let sc = scale[oi];
        for (si, slot) in row.iter_mut().enumerate() {
            let xs = &x[si * i..(si + 1) * i];
            let a: f32 = xs.iter().zip(w).map(|(&xv, &wv)| xv * wv as f32).sum();
            *slot = a * sc;
        }
    });
    transpose_so(y, &yt, s, o);
}

/// y[S,O] = x[S,I] @ W^T, W int4-packed[O,ceil(I/2)] (2 values/byte) per-row scale.
pub fn matmul_i4(y: &mut [f32], x: &[f32], q4: &[u8], scale: &[f32], s: usize, i: usize, o: usize) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        return unsafe { matmul_i4_avx2(y, x, q4, scale, s, i, o) };
    }
    matmul_i4_scalar(y, x, q4, scale, s, i, o)
}

fn matmul_i4_scalar(y: &mut [f32], x: &[f32], q4: &[u8], scale: &[f32], s: usize, i: usize, o: usize) {
    let rb = i.div_ceil(2);
    let mut yt = vec![0f32; o * s];
    yt.par_chunks_mut(s).enumerate().for_each(|(oi, row)| {
        let w = &q4[oi * rb..(oi + 1) * rb];
        let sc = scale[oi];
        for (si, slot) in row.iter_mut().enumerate() {
            let xs = &x[si * i..(si + 1) * i];
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
            *slot = a * sc;
        }
    });
    transpose_so(y, &yt, s, o);
}

/// y[S,O] = x[S,I] @ W^T, W int2-packed[O,ceil(I/4)] (4 values/byte) per-row scale.
pub fn matmul_i2(y: &mut [f32], x: &[f32], q2: &[u8], scale: &[f32], s: usize, i: usize, o: usize) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        return unsafe { matmul_i2_avx2(y, x, q2, scale, s, i, o) };
    }
    matmul_i2_scalar(y, x, q2, scale, s, i, o)
}

fn matmul_i2_scalar(y: &mut [f32], x: &[f32], q2: &[u8], scale: &[f32], s: usize, i: usize, o: usize) {
    let rb = i.div_ceil(4);
    let mut yt = vec![0f32; o * s];
    yt.par_chunks_mut(s).enumerate().for_each(|(oi, row)| {
        let w = &q2[oi * rb..(oi + 1) * rb];
        let sc = scale[oi];
        for (si, slot) in row.iter_mut().enumerate() {
            let xs = &x[si * i..(si + 1) * i];
            let mut a = 0f32;
            for ii in 0..i {
                let byte = w[ii >> 2];
                let sh = (ii & 3) * 2;
                let v = ((byte >> sh) & 3) as i32 - 2;
                a += xs[ii] * v as f32;
            }
            *slot = a * sc;
        }
    });
    transpose_so(y, &yt, s, o);
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

    /// int4-packed weight row · f32 activation row -> f32. `n` = logical (unpacked) length.
    #[target_feature(enable = "avx2")]
    unsafe fn dot_i4_f32_avx2(w: &[u8], xs: &[f32], n: usize) -> f32 {
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

    /// int2-packed weight row · f32 activation row -> f32. `n` = logical (unpacked) length.
    #[target_feature(enable = "avx2")]
    unsafe fn dot_i2_f32_avx2(w: &[u8], xs: &[f32], n: usize) -> f32 {
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
        let mut yt = vec![0f32; o * s];
        yt.par_chunks_mut(s).enumerate().for_each(|(oi, row)| {
            let w = &q[oi * i..(oi + 1) * i];
            let sc = scale[oi];
            for (si, slot) in row.iter_mut().enumerate() {
                let xs = &x[si * i..(si + 1) * i];
                // Safety: caller of `matmul_q_avx2` already verified AVX2 at the dispatch
                // site (`matmul_q`); that's a whole-machine capability, not per-thread state,
                // so it still holds inside this rayon worker closure.
                *slot = unsafe { dot_q8_f32_avx2(w, xs) } * sc;
            }
        });
        transpose_so(y, &yt, s, o);
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn matmul_i4_avx2(y: &mut [f32], x: &[f32], q4: &[u8], scale: &[f32], s: usize, i: usize, o: usize) {
        let rb = i.div_ceil(2);
        let mut yt = vec![0f32; o * s];
        yt.par_chunks_mut(s).enumerate().for_each(|(oi, row)| {
            let w = &q4[oi * rb..(oi + 1) * rb];
            let sc = scale[oi];
            for (si, slot) in row.iter_mut().enumerate() {
                let xs = &x[si * i..(si + 1) * i];
                *slot = unsafe { dot_i4_f32_avx2(w, xs, i) } * sc;
            }
        });
        transpose_so(y, &yt, s, o);
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn matmul_i2_avx2(y: &mut [f32], x: &[f32], q2: &[u8], scale: &[f32], s: usize, i: usize, o: usize) {
        let rb = i.div_ceil(4);
        let mut yt = vec![0f32; o * s];
        yt.par_chunks_mut(s).enumerate().for_each(|(oi, row)| {
            let w = &q2[oi * rb..(oi + 1) * rb];
            let sc = scale[oi];
            for (si, slot) in row.iter_mut().enumerate() {
                let xs = &x[si * i..(si + 1) * i];
                *slot = unsafe { dot_i2_f32_avx2(w, xs, i) } * sc;
            }
        });
        transpose_so(y, &yt, s, o);
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
pub use simd::{dot_i4i8_avx2, dot_i4i8_avx512vnni, dot_i8i8_avx2, dot_i8i8_avx512vnni};

// mirrors matmul_q_idot/matmul_i4_idot's C signature 1:1; both are private, single-call-site
// helpers for matmul_qt, so a wrapper struct would be indirection with no real caller benefit.
#[allow(clippy::too_many_arguments)]
fn matmul_q_idot(y: &mut [f32], xq: &[i8], sx: &[f32], q: &[i8], scale: &[f32], s: usize, i: usize, o: usize) {
    let mut yt = vec![0f32; o * s];
    yt.par_chunks_mut(s).enumerate().for_each(|(oi, row)| {
        let w = &q[oi * i..(oi + 1) * i];
        let sc = scale[oi];
        for (si, slot) in row.iter_mut().enumerate() {
            let xrow = &xq[si * i..(si + 1) * i];
            *slot = dot_i8i8(w, xrow, i) as f32 * sc * sx[si];
        }
    });
    transpose_so(y, &yt, s, o);
}

#[allow(clippy::too_many_arguments)]
fn matmul_i4_idot(y: &mut [f32], xq: &[i8], sx: &[f32], q4: &[u8], scale: &[f32], s: usize, i: usize, o: usize) {
    let rb = i.div_ceil(2);
    let mut yt = vec![0f32; o * s];
    yt.par_chunks_mut(s).enumerate().for_each(|(oi, row)| {
        let w = &q4[oi * rb..(oi + 1) * rb];
        let sc = scale[oi];
        for (si, slot) in row.iter_mut().enumerate() {
            let xrow = &xq[si * i..(si + 1) * i];
            *slot = dot_i4i8(w, xrow, i) as f32 * sc * sx[si];
        }
    });
    transpose_so(y, &yt, s, o);
}

/// x86 default from `glm.c`'s `g_i4s`: without ARM SDOT, int4 IDOT only pays off at S>=2 —
/// at S=1 (decode) the float-weight path (`matmul_i4`) wins.
const I4_IDOT_MIN_S: usize = 2;

/// y[S,O] = x[S,I] @ W^T for a `QT` in any format — the dispatcher every layer calls.
/// int8 IDOT is always used (colibri measures 1.4-2.5x over the float-weight path); int4
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

/// `acc[0..I) += coef * W[row,:]` — dequantizes one row of `w` on the fly and scales it by
/// `coef` before accumulating. This is the "weight absorption" building block: MLA's decode
/// path folds `q_nope · k_nope_t = q_nope · (W_K L_t) = (W_K^T q_nope) · L_t`, so instead of
/// reconstructing `k_nope_t` per cached token it accumulates `W_K^T q_nope` once (a sum of
/// `qk_nope` scaled rows) and dot-products that against the raw latents directly.
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
            for k in 0..i {
                let byte = wr[k >> 1];
                let nibble = if k & 1 == 0 { byte & 0xF } else { byte >> 4 };
                acc[k] += c * (nibble as i32 - 8) as f32;
            }
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
                let mut acc = 0f32;
                for k in 0..i {
                    let byte = wr[k >> 1];
                    let nibble = if k & 1 == 0 { byte & 0xF } else { byte >> 4 };
                    acc += (nibble as i32 - 8) as f32 * x[k];
                }
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
