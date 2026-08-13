//! Qwen 3.8's normalization, which is NOT the RMSNorm every other family in this crate uses.
//!
//! `Qwen3_5MoeRMSNorm.forward` (read from the real `modeling_qwen3_5_moe.py` this session) is
//!
//! ```text
//! output = x * rsqrt(mean(x^2) + eps) * (1.0 + weight)
//! ```
//!
//! — note the **`1.0 +`**, and that the parameter is initialized to ZEROS rather than ones
//! (`nn.Parameter(torch.zeros(dim))`), i.e. the stored weights are deviations FROM unity, not
//! multipliers. `glm52::attention::rmsnorm` and `kimi_linear::ops::rmsnorm` both compute
//! `norm(x) * weight`, which on a Qwen checkpoint would collapse every normalized activation toward
//! zero (a checkpoint whose weights sit near 0.0 would nearly annihilate the residual stream) —
//! silently, with no shape error anywhere.
//!
//! This applies to `input_layernorm`, `post_attention_layernorm`, the final `model.norm`, and the
//! per-head `q_norm`/`k_norm` inside the attention layers — every `Qwen3_5MoeRMSNorm` instance.
//!
//! It deliberately does NOT apply to Gated DeltaNet's output norm: `Qwen3_5MoeRMSNormGated` is a
//! separate class whose `forward` multiplies by `self.weight` plainly (and initializes it to ones),
//! so `gdn::head_output_gate_silu` keeps using `kimi_linear::ops::rmsnorm`. Two norms, two
//! conventions, in the same model.

/// `x * rsqrt(mean(x^2) + eps) * (1 + w)`, in place — Qwen 3.8's `Qwen3_5MoeRMSNorm`.
pub fn rmsnorm_1p(v: &mut [f32], w: &[f32], eps: f32) {
    debug_assert_eq!(v.len(), w.len());
    let n = v.len() as f64;
    let ms: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum();
    let r = 1.0f32 / (((ms / n) as f32) + eps).sqrt();
    for (vi, &wi) in v.iter_mut().zip(w) {
        *vi *= r * (1.0 + wi);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point: a zero weight is the IDENTITY scale here, where the usual RMSNorm would
    /// zero the vector out.
    #[test]
    fn a_zero_weight_leaves_the_normalized_vector_alone() {
        let w = vec![0.0f32; 4];
        let mut v = vec![2.0f32, 2.0, 2.0, 2.0];
        rmsnorm_1p(&mut v, &w, 1e-6);
        for x in &v {
            assert!((x - 1.0).abs() < 1e-4, "expected the normalized 1.0, got {x}");
        }

        let mut plain = vec![2.0f32, 2.0, 2.0, 2.0];
        crate::glm52::attention::rmsnorm(&mut plain, &w, 1e-6);
        assert!(plain.iter().all(|&x| x == 0.0), "the crate's other RMSNorm zeroes it: {plain:?}");
    }

    #[test]
    fn matches_the_reference_formula_elementwise() {
        let v0 = vec![1.0f32, -2.0, 3.0, -4.0, 0.5, 0.25];
        let w = vec![0.1f32, -0.2, 0.0, 0.5, -0.75, 1.0];
        let eps = 1e-6f32;

        let ms: f32 = v0.iter().map(|x| x * x).sum::<f32>() / v0.len() as f32;
        let want: Vec<f32> = v0.iter().zip(&w).map(|(&x, &wi)| x / (ms + eps).sqrt() * (1.0 + wi)).collect();

        let mut got = v0.clone();
        rmsnorm_1p(&mut got, &w, eps);
        for (i, (g, e)) in got.iter().zip(&want).enumerate() {
            assert!((g - e).abs() < 1e-5, "dim {i}: {g} vs {e}");
        }
    }

    /// `1 + w` vs `w` differ for every weight except exactly 0.5 — pinned so the two norms can't be
    /// swapped by accident and still pass.
    #[test]
    fn differs_from_plain_rmsnorm_for_a_realistic_weight() {
        let w = vec![0.05f32; 8];
        let mut ours = vec![3.0f32; 8];
        let mut plain = vec![3.0f32; 8];
        rmsnorm_1p(&mut ours, &w, 1e-6);
        crate::glm52::attention::rmsnorm(&mut plain, &w, 1e-6);
        assert!((ours[0] - 1.05).abs() < 1e-4, "got {}", ours[0]);
        assert!((plain[0] - 0.05).abs() < 1e-4, "got {}", plain[0]);
    }
}
