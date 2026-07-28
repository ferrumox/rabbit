//! Small elementwise primitives KDA's Q/K/V pipeline needs on top of `short_conv.rs`'s
//! `ShortConvState` (arXiv:2510.26692 §4: `q,k = L2Norm(Swish(ShortConv(Wx)))`,
//! `v = Swish(ShortConv(Wx))`).
//!
//! `swish` duplicates `glm52::moe.rs`'s private `siluf` (`SiLU(x) == Swish(x) == x*sigmoid(x)`,
//! same function, different name in each paper) rather than sharing it — a one-line function
//! isn't worth introducing a cross-architecture dependency for, per this project's stance on not
//! building shared abstractions before a second real need forces the shape (see
//! `rabbit-plan.md`'s Phase 1 notes on the same tradeoff for `Model`/`KvState`).
//!
//! `l2_norm`'s exact formula (`x / max(||x||_2, eps)`, a clamp on the norm itself — NOT
//! `x / sqrt(sum(x^2) + eps)`, a different and easily-mixed-up formula) is confirmed against
//! `ggml_compute_forward_l2_norm_f32` in the vendored llama.cpp reference
//! (`fox/vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp`), the same real implementation
//! `kimi-linear.cpp` calls for Q/K normalization — not guessed from the paper's prose alone.
//!
//! `rmsnorm` and `head_output_gate` cover the output stage: `output = RMSNorm(o) * sigmoid(g2)`
//! per head, before `W_o` — also confirmed against `kimi-linear.cpp`, in particular that the
//! gate is `sigmoid` and not `swish` despite the norm layer's usual default (see
//! `head_output_gate`'s doc comment).

pub fn swish(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Normalizes `x` to unit L2 norm in place, `x_i /= max(||x||_2, eps)` — the `eps` floor (not
/// added inside the sqrt) only matters when `||x||_2` is smaller than `eps`, avoiding a
/// division blow-up on a near-zero vector without perturbing any normally-scaled vector's
/// direction.
pub fn l2_norm(x: &mut [f32], eps: f32) {
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let scale = 1.0 / sum_sq.sqrt().max(eps);
    for v in x.iter_mut() {
        *v *= scale;
    }
}

/// `v_i *= w_i / sqrt(mean(v^2) + eps)` — the same formula as `glm52::attention::rmsnorm`
/// (including its `f64` accumulation, for the same reason: `mean(v^2)` over a wide vector loses
/// precision in `f32`). Deliberately a separate copy rather than a cross-architecture `use` of
/// `glm52`'s version, for the same reason `swish` above duplicates `glm52::moe.rs::siluf`.
pub fn rmsnorm(v: &mut [f32], w: &[f32], eps: f32) {
    let n = v.len() as f64;
    let ms: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum();
    let r = 1.0f32 / (((ms / n) as f32) + eps).sqrt();
    for (vi, &wi) in v.iter_mut().zip(w) {
        *vi *= r * wi;
    }
}

/// KDA's output stage for one head (arXiv:2510.26692 §4, confirmed against llama.cpp's
/// `kimi-linear.cpp`: `output = RMSNorm(o) * sigmoid(g2)`, applied per head before the heads are
/// concatenated and passed through `W_o`). `o` is the KDA recurrence's `head_dim`-wide output for
/// this head (`KdaState::step`'s `o` argument), normalized in place; `o_norm_weight` is shared
/// across all heads (one `head_dim`-wide learned vector, not per-head); `g2` is this head's slice
/// of the low-rank output-gate projection `g_b(g_a(x))`, **pre**-sigmoid. Note this gate is a
/// plain `sigmoid`, not `swish`/`SiLU` — confirmed via an explicit comment in the real reference
/// flagging it as a deliberate deviation from that norm layer's usual default.
pub fn head_output_gate(o: &mut [f32], o_norm_weight: &[f32], g2: &[f32], eps: f32) {
    rmsnorm(o, o_norm_weight, eps);
    for (oi, &gi) in o.iter_mut().zip(g2) {
        *oi *= sigmoid(gi);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swish_matches_x_times_sigmoid_x() {
        for &x in &[-5.0f32, -1.0, -0.001, 0.0, 0.001, 1.0, 5.0] {
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            assert!((swish(x) - x * sigmoid).abs() < 1e-6);
        }
    }

    #[test]
    fn swish_at_zero_is_zero() {
        assert_eq!(swish(0.0), 0.0);
    }

    #[test]
    fn l2_norm_produces_a_unit_vector() {
        let mut v = vec![3.0, 4.0]; // ||v|| = 5
        l2_norm(&mut v, 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn l2_norm_preserves_direction() {
        let mut v = vec![1.0, -2.0, 3.0, -4.0];
        let original = v.clone();
        l2_norm(&mut v, 1e-6);
        // every component's sign must be unchanged, and ratios between components preserved.
        for i in 0..v.len() {
            assert_eq!(v[i].signum(), original[i].signum());
        }
        assert!((v[1] / v[0] - original[1] / original[0]).abs() < 1e-4);
    }

    #[test]
    fn l2_norm_clamps_at_eps_for_a_near_zero_vector() {
        // ||v|| far below eps: scale must be 1/eps (the clamp), not 1/||v|| (which would blow
        // up), matching ggml_compute_forward_l2_norm_f32's `1/max(sqrt(sum), eps)`.
        let mut v = vec![1e-10f32, 0.0];
        let eps = 1e-3f32;
        l2_norm(&mut v, eps);
        assert!((v[0] - 1e-10 / eps).abs() < 1e-12, "got {}", v[0]);
    }

    #[test]
    fn l2_norm_of_exact_zero_vector_does_not_divide_by_zero() {
        let mut v = vec![0.0f32, 0.0, 0.0];
        l2_norm(&mut v, 1e-6);
        for x in v {
            assert!(x.is_finite());
            assert_eq!(x, 0.0);
        }
    }

    #[test]
    fn sigmoid_matches_hand_computed_values() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(100.0) > 0.999999);
        assert!(sigmoid(-100.0) < 0.000001);
    }

    #[test]
    fn rmsnorm_matches_naive_reference() {
        // Independent, unfused reference: v_i -> v_i * w_i / sqrt(mean(v^2) + eps).
        fn naive_rmsnorm(v: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
            let mean_sq = v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32;
            let r = 1.0 / (mean_sq + eps).sqrt();
            v.iter().zip(w).map(|(&x, &wi)| x * r * wi).collect()
        }
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0, 0.5, 2.0, 1.0];
        let expected = naive_rmsnorm(&v, &w, 1e-5);
        rmsnorm(&mut v, &w, 1e-5);
        for (a, b) in v.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn rmsnorm_of_unit_weight_produces_unit_root_mean_square() {
        let mut v = vec![2.0, -6.0, 3.0, 1.0];
        let w = vec![1.0; 4];
        rmsnorm(&mut v, &w, 1e-8);
        let rms = (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
        assert!((rms - 1.0).abs() < 1e-4);
    }

    #[test]
    fn head_output_gate_fully_closed_gate_zeroes_the_output() {
        // g2 very negative -> sigmoid ~= 0 -> gated output must be ~0 regardless of o/o_norm_weight.
        let mut o = vec![5.0, -3.0, 2.0];
        let w = vec![1.0, 1.0, 1.0];
        let g2 = vec![-100.0, -100.0, -100.0];
        head_output_gate(&mut o, &w, &g2, 1e-5);
        for x in o {
            assert!(x.abs() < 1e-8, "expected ~0, got {x}");
        }
    }

    #[test]
    fn head_output_gate_fully_open_gate_reduces_to_plain_rmsnorm() {
        // g2 very positive -> sigmoid ~= 1 -> gated output must equal RMSNorm(o) alone.
        let mut gated = vec![5.0, -3.0, 2.0];
        let mut plain = gated.clone();
        let w = vec![1.0, 0.7, 1.3];
        let g2 = vec![100.0, 100.0, 100.0];
        head_output_gate(&mut gated, &w, &g2, 1e-5);
        rmsnorm(&mut plain, &w, 1e-5);
        for (a, b) in gated.iter().zip(&plain) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn head_output_gate_matches_hand_computed_half_open_gate() {
        // g2 = 0 -> sigmoid(0) = 0.5 exactly -> gated output = RMSNorm(o) * 0.5.
        let mut gated = vec![4.0, -4.0];
        let w = vec![1.0, 1.0];
        let g2 = vec![0.0, 0.0];
        head_output_gate(&mut gated, &w, &g2, 1e-8);
        // RMSNorm([4,-4], w=[1,1]): mean(v^2)=16, r=1/4 -> normed=[1,-1]; * 0.5 -> [0.5,-0.5].
        assert!((gated[0] - 0.5).abs() < 1e-4);
        assert!((gated[1] - (-0.5)).abs() < 1e-4);
    }
}
