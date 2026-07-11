//! Port of `glm.c`'s matmul family — scalar baseline only (Fase 3 of the plan). AVX2 and
//! AVX-512/VNNI paths are added on top of these same signatures in a later phase, selected at
//! runtime; nothing here needs to change when that happens.
//!
//! `y[S,O] = x[S,I] @ W^T` throughout, `W` given in one of the `QT` formats from `quant.rs`.
//! The IDOT kernels (`dot_i8i8`/`dot_i4i8`) additionally quantize activations to int8 per row
//! (`qrow_i8`) so the whole dot product runs in integer arithmetic — colibri measures this at
//! ~2-3x over the float-weight path, at ~0.3% added RMS error per matmul from the activation
//! quantization.

use crate::quant::{QT, QTKind};

/// y[S,O] = x[S,I] @ W^T, W[O,I] f32.
pub fn matmul(y: &mut [f32], x: &[f32], w: &[f32], s: usize, i: usize, o: usize) {
    for oi in 0..o {
        let wr = &w[oi * i..(oi + 1) * i];
        for si in 0..s {
            let xs = &x[si * i..(si + 1) * i];
            let a: f32 = xs.iter().zip(wr).map(|(a, b)| a * b).sum();
            y[si * o + oi] = a;
        }
    }
}

/// y[S,O] = x[S,I] @ W^T, W int8[O,I] per-row scale (dequant-on-use).
pub fn matmul_q(y: &mut [f32], x: &[f32], q: &[i8], scale: &[f32], s: usize, i: usize, o: usize) {
    for oi in 0..o {
        let w = &q[oi * i..(oi + 1) * i];
        let sc = scale[oi];
        for si in 0..s {
            let xs = &x[si * i..(si + 1) * i];
            let a: f32 = xs.iter().zip(w).map(|(&xv, &wv)| xv * wv as f32).sum();
            y[si * o + oi] = a * sc;
        }
    }
}

/// y[S,O] = x[S,I] @ W^T, W int4-packed[O,ceil(I/2)] (2 values/byte) per-row scale.
pub fn matmul_i4(y: &mut [f32], x: &[f32], q4: &[u8], scale: &[f32], s: usize, i: usize, o: usize) {
    let rb = i.div_ceil(2);
    for oi in 0..o {
        let w = &q4[oi * rb..(oi + 1) * rb];
        let sc = scale[oi];
        for si in 0..s {
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
            y[si * o + oi] = a * sc;
        }
    }
}

/// y[S,O] = x[S,I] @ W^T, W int2-packed[O,ceil(I/4)] (4 values/byte) per-row scale.
pub fn matmul_i2(y: &mut [f32], x: &[f32], q2: &[u8], scale: &[f32], s: usize, i: usize, o: usize) {
    let rb = i.div_ceil(4);
    for oi in 0..o {
        let w = &q2[oi * rb..(oi + 1) * rb];
        let sc = scale[oi];
        for si in 0..s {
            let xs = &x[si * i..(si + 1) * i];
            let mut a = 0f32;
            for ii in 0..i {
                let byte = w[ii >> 2];
                let sh = (ii & 3) * 2;
                let v = ((byte >> sh) & 3) as i32 - 2;
                a += xs[ii] * v as f32;
            }
            y[si * o + oi] = a * sc;
        }
    }
}

/// Quantizes one activation row to int8 (absmax/127, Q8_0-style) for the IDOT kernels.
/// Returns the row's scale; `q.len() == x.len()` required.
pub fn qrow_i8(x: &[f32], q: &mut [i8]) -> f32 {
    let amax = x.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let s = (amax / 127.0).max(1e-12);
    let inv = 1.0 / s;
    for (qi, &xi) in q.iter_mut().zip(x) {
        *qi = (xi * inv).round_ties_even() as i32 as i8;
    }
    s
}

/// int8·int8 dot product, scalar. Pairs are bounded by `127*127*2 < i32::MAX` up to
/// unrealistic `I`, so plain `i32` accumulation never overflows in practice.
pub fn dot_i8i8(w: &[i8], x: &[i8], i: usize) -> i32 {
    let mut sum = 0i32;
    for k in 0..i {
        sum += w[k] as i32 * x[k] as i32;
    }
    sum
}

/// int4(packed)·int8 dot product, scalar: unpack each nibble to `[-8,7]` on the fly.
pub fn dot_i4i8(w4: &[u8], x: &[i8], i: usize) -> i32 {
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

// mirrors matmul_q_idot/matmul_i4_idot's C signature 1:1; both are private, single-call-site
// helpers for matmul_qt, so a wrapper struct would be indirection with no real caller benefit.
#[allow(clippy::too_many_arguments)]
fn matmul_q_idot(y: &mut [f32], xq: &[i8], sx: &[f32], q: &[i8], scale: &[f32], s: usize, i: usize, o: usize) {
    for oi in 0..o {
        let w = &q[oi * i..(oi + 1) * i];
        let sc = scale[oi];
        for si in 0..s {
            let xrow = &xq[si * i..(si + 1) * i];
            y[si * o + oi] = dot_i8i8(w, xrow, i) as f32 * sc * sx[si];
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn matmul_i4_idot(y: &mut [f32], xq: &[i8], sx: &[f32], q4: &[u8], scale: &[f32], s: usize, i: usize, o: usize) {
    let rb = i.div_ceil(2);
    for oi in 0..o {
        let w = &q4[oi * rb..(oi + 1) * rb];
        let sc = scale[oi];
        for si in 0..s {
            let xrow = &xq[si * i..(si + 1) * i];
            y[si * o + oi] = dot_i4i8(w, xrow, i) as f32 * sc * sx[si];
        }
    }
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
}
