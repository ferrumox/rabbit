//! Qwen 3.8's linear-attention layers: **Gated DeltaNet** (69 of the 92 layers). The recurrence
//! itself needs no new code — it is exactly `kimi_linear::kda::KdaState::step`, the delta rule this
//! crate already ships and tests — so this module is the surrounding neural wiring plus the four
//! places GDN genuinely differs from Kimi Delta Attention.
//!
//! Ported from `transformers`' real `Qwen3_5MoeGatedDeltaNet.forward`,
//! `torch_recurrent_gated_delta_rule` and `Qwen3_5MoeRMSNormGated` (`modeling_qwen3_5_moe.py` /
//! `modeling_qwen3_next.py`, read this session).
//!
//! **The recurrence is the same rule, written differently.** The reference computes, per head:
//!
//! ```text
//! S = S * exp(g)                     # scalar decay
//! kv_mem = S^T k                     # (last_recurrent_state * k[:,None]).sum(-2)
//! delta  = (v - kv_mem) * beta
//! S      = S + k (x) delta           # outer product
//! o      = S^T q
//! ```
//!
//! and `KdaState::step`'s `S_t = (I - β k k^T) Diag(α) S_{t-1} + β k v^T`, `o = S_t^T q` expands to
//! `αS + β k (v - (αS)^T k)^T` — the same thing, with the decay applied before `kv_mem` in both.
//! `alpha_from_decay` below is what bridges them (see point 1), and
//! `gdn_step_matches_the_reference_recurrence` checks the equality numerically against a literal
//! transcription of the reference loop.
//!
//! The four real differences from KDA:
//!
//! **1. The decay is SCALAR per head, not per channel.** KDA's whole refinement over GDN is a
//! `d_k`-wide `α`; here `A_log`, `dt_bias` and the projected `a` are all one value per V head, so
//! `α = exp(-exp(A_log) · softplus(a + dt_bias))` is a scalar broadcast over the head's `d_k` rows.
//! `alpha_from_decay` fills a `d_k`-wide buffer with it, and `kimi_linear::kda::decay_gate`
//! (`lower_bound: None` — GDN has no bounded-gate variant) computes the same formula, which
//! `decay_matches_kimis_gate_formula_broadcast` pins.
//!
//! **2. Q and K heads are GQA-shaped.** 16 key heads feed 128 value heads (1:8) via
//! `repeat_interleave`, so V head `h` reads K head `h / 8` — contiguous blocks, same convention as
//! `attention.rs`' `repeat_kv`. `key_head_of` is the one place that mapping lives.
//!
//! **3. One depthwise conv over the CONCATENATED q|k|v, then SiLU, then split.** The reference has a
//! single `conv1d` of `conv_dim = 2 * key_dim + value_dim` channels (20,480 here: 2,048 + 2,048 +
//! 16,384), no bias, `kernel = 4`, and `causal_conv1d` applies `hidden_act` (`"silu"`) to its OUTPUT
//! — so the activation lands after the convolution, not before it, and covers q, k and v alike.
//! `kimi_linear::short_conv::ShortConvState` handles the convolution unchanged (it applies no
//! activation of its own; `kimi_linear::generate` likewise swishes afterwards); `activate_and_split`
//! covers the rest.
//!
//! **4. The output gate uses SiLU, not sigmoid.** `Qwen3_5MoeRMSNormGated` is
//! `rmsnorm(o) * silu(z)`, where `z` is `in_proj_z`'s own per-head slice. This is the exact opposite
//! deviation from `kimi_linear::ops::head_output_gate`, whose doc records that Kimi deliberately
//! uses a plain `sigmoid` there — so the existing helper must NOT be reused, and
//! `head_output_gate_silu` exists to make the difference explicit rather than a one-character edit
//! away from wrong.
//!
//! Also unlike KDA: **q is scaled by `1/sqrt(head_k_dim)` after L2 normalization**
//! (`scale = 1 / query.shape[-1] ** 0.5` in the reference kernel, applied to the query only).
//! L2-normalizing before the GQA repeat is equivalent to after (the reference normalizes inside the
//! kernel, i.e. post-repeat) because normalizing copies of a vector gives copies of its normalization
//! — done once per K head here, not eight times.

use crate::kimi_linear::kda::{KdaState, decay_gate};
use crate::kimi_linear::ops::{l2_norm, rmsnorm, sigmoid, swish};

/// The L2-normalization epsilon the reference kernel passes (`l2norm(..., eps=1e-6)`).
const L2_EPS: f32 = 1e-6;

/// Which K head V head `v_head` reads, under the reference's `repeat_interleave(n_v / n_k, dim=2)`:
/// integer division, so K head `j` serves the contiguous block of V heads `j * n_rep ..
/// (j + 1) * n_rep`. The other plausible reading (`v_head % n_k_heads`, i.e. `repeat` rather than
/// `repeat_interleave`) type-checks and keeps every shape valid while pairing each head with the
/// wrong keys.
pub fn key_head_of(v_head: usize, n_v_heads: usize, n_k_heads: usize) -> usize {
    debug_assert!(n_k_heads > 0 && n_v_heads.is_multiple_of(n_k_heads));
    v_head / (n_v_heads / n_k_heads)
}

/// `alpha = exp(-exp(a_log) · softplus(a + dt_bias))` for one head, broadcast over `alpha`'s
/// `d_k` entries — see this module's doc, point 1. Delegates to `kimi_linear::kda::decay_gate` so
/// there is exactly one implementation of the formula in the crate.
pub fn alpha_from_decay(a_log: f32, a: f32, dt_bias: f32, alpha: &mut [f32]) {
    let mut one = [0f32];
    decay_gate(a_log, &[a], &[dt_bias], None, &mut one);
    alpha.fill(one[0]);
}

/// `beta = sigmoid(b)` for one head (`b.sigmoid()` in the reference) — the delta rule's write
/// strength, a single scalar per head exactly as in KDA.
pub fn beta_from_projection(b: f32) -> f32 {
    sigmoid(b)
}

/// Applies the conv's `hidden_act` (SiLU) in place to the whole `conv_dim`-wide post-convolution row,
/// then splits it into `(q, k, v)` — widths `key_dim`, `key_dim`, `value_dim`, in that order
/// (`torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)`).
pub fn activate_and_split(qkv: &mut [f32], key_dim: usize, value_dim: usize) -> (&[f32], &[f32], &[f32]) {
    assert_eq!(qkv.len(), 2 * key_dim + value_dim, "conv output must be 2 * key_dim + value_dim wide");
    for x in qkv.iter_mut() {
        *x = swish(*x);
    }
    let (q, rest) = qkv.split_at(key_dim);
    let (k, v) = rest.split_at(key_dim);
    (q, k, v)
}

/// L2-normalizes one K head's `q` and `k` in place, then scales `q` by `1 / sqrt(d_k)` — the
/// reference kernel's `use_qk_l2norm_in_kernel` plus its `query = query * scale`. `k` is NOT scaled.
pub fn prepare_qk(q: &mut [f32], k: &mut [f32]) {
    debug_assert_eq!(q.len(), k.len());
    l2_norm(q, L2_EPS);
    l2_norm(k, L2_EPS);
    let scale = 1.0 / (q.len() as f32).sqrt();
    for x in q.iter_mut() {
        *x *= scale;
    }
}

/// `Qwen3_5MoeRMSNormGated` for one head: `rmsnorm(o, norm_weight) * silu(z)`, in place. `norm_weight`
/// is one `head_v_dim`-wide vector shared by every head. **SiLU, not sigmoid** — see this module's
/// doc, point 4.
pub fn head_output_gate_silu(o: &mut [f32], norm_weight: &[f32], z: &[f32], eps: f32) {
    debug_assert_eq!(o.len(), z.len());
    rmsnorm(o, norm_weight, eps);
    for (oi, &zi) in o.iter_mut().zip(z) {
        *oi *= swish(zi);
    }
}

/// One token through a whole GDN layer's recurrence, all V heads: the piece between the input
/// projections (which produce `qkv`, `z`, `a`, `b` — matmuls, so they live in the model loader) and
/// `out_proj`.
///
/// * `qkv` — the post-convolution row, mutated in place by [`activate_and_split`]'s SiLU.
/// * `z`, `out` — `n_v_heads * head_v_dim`.
/// * `a`, `b` — `n_v_heads` each, from `in_proj_a`/`in_proj_b`.
/// * `a_log`, `dt_bias` — `n_v_heads` each, learned.
/// * `norm_weight` — `head_v_dim`, the gated RMSNorm's shared weight.
/// * `states` — one [`KdaState`] per V head, each `[head_k_dim, head_v_dim]`, carried across tokens.
///
/// Serial over heads, like `attention.rs`' own loop: correctness first. Note the shape of the work
/// per token — 128 heads x a 128x128 state — is 2 MB of state per layer touched twice, which is why
/// this (not the 23 attention layers) is the part that will eventually want `rayon`.
#[allow(clippy::too_many_arguments)]
pub fn step_token(
    states: &mut [KdaState],
    qkv: &mut [f32],
    z: &[f32],
    a: &[f32],
    b: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    norm_weight: &[f32],
    eps: f32,
    out: &mut [f32],
) {
    let n_v_heads = states.len();
    assert!(n_v_heads > 0, "a GDN layer has at least one value head");
    let head_k_dim = states[0].d_k();
    let head_v_dim = states[0].d_v();
    assert_eq!(out.len(), n_v_heads * head_v_dim, "out must be n_v_heads * head_v_dim");
    assert_eq!(z.len(), out.len(), "z must match the output width");
    assert_eq!(a.len(), n_v_heads, "a must be one value per V head");
    assert_eq!(b.len(), n_v_heads, "b must be one value per V head");
    assert_eq!(a_log.len(), n_v_heads, "A_log must be one value per V head");
    assert_eq!(dt_bias.len(), n_v_heads, "dt_bias must be one value per V head");
    assert_eq!(norm_weight.len(), head_v_dim, "the gated RMSNorm weight is head_v_dim wide");

    let value_dim = n_v_heads * head_v_dim;
    let key_dim = qkv.len().saturating_sub(value_dim) / 2;
    assert!(key_dim > 0 && key_dim.is_multiple_of(head_k_dim), "conv row doesn't split into whole K heads");
    let n_k_heads = key_dim / head_k_dim;

    let (q_all, k_all, v_all) = activate_and_split(qkv, key_dim, value_dim);

    // Normalize each K head ONCE, not once per V head that shares it (equivalent — see this
    // module's doc — and 8x less work at the real config's 1:8 ratio).
    let mut q_prepared = q_all.to_vec();
    let mut k_prepared = k_all.to_vec();
    for kh in 0..n_k_heads {
        let range = kh * head_k_dim..(kh + 1) * head_k_dim;
        let (q_head, k_head) = (&mut q_prepared[range.clone()], &mut k_prepared[range]);
        prepare_qk(q_head, k_head);
    }

    let mut alpha = vec![0f32; head_k_dim];
    for h in 0..n_v_heads {
        let kh = key_head_of(h, n_v_heads, n_k_heads);
        let k_range = kh * head_k_dim..(kh + 1) * head_k_dim;
        let v_range = h * head_v_dim..(h + 1) * head_v_dim;

        alpha_from_decay(a_log[h], a[h], dt_bias[h], &mut alpha);
        let beta = beta_from_projection(b[h]);
        states[h].step(
            &q_prepared[k_range.clone()],
            &k_prepared[k_range],
            &v_all[v_range.clone()],
            &alpha,
            beta,
            &mut out[v_range.clone()],
        );
        head_output_gate_silu(&mut out[v_range.clone()], norm_weight, &z[v_range], eps);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn softplus(x: f32) -> f32 {
        x.max(0.0) + (-x.abs()).exp().ln_1p()
    }

    fn ramp(n: usize, seed: u32) -> Vec<f32> {
        (0..n).map(|i| (((i as u32 * 37 + seed) % 23) as f32 - 11.0) / 5.0).collect()
    }

    /// A literal transcription of `torch_recurrent_gated_delta_rule`'s loop body, structured like the
    /// Python (decay, `kv_mem`, `delta`, outer-product update, readout) rather than like
    /// `KdaState::step`'s fused passes — so agreement between the two is real evidence, not a
    /// tautology.
    #[allow(clippy::too_many_arguments)] // a transcription of the reference's own parameter list
    fn reference_step(s: &mut [f32], d_k: usize, d_v: usize, q: &[f32], k: &[f32], v: &[f32], g_exp: f32, beta: f32) -> Vec<f32> {
        for x in s.iter_mut() {
            *x *= g_exp;
        }
        let mut kv_mem = vec![0f32; d_v];
        for i in 0..d_k {
            for j in 0..d_v {
                kv_mem[j] += s[i * d_v + j] * k[i];
            }
        }
        let delta: Vec<f32> = (0..d_v).map(|j| (v[j] - kv_mem[j]) * beta).collect();
        for i in 0..d_k {
            for j in 0..d_v {
                s[i * d_v + j] += k[i] * delta[j];
            }
        }
        let mut o = vec![0f32; d_v];
        for i in 0..d_k {
            for j in 0..d_v {
                o[j] += s[i * d_v + j] * q[i];
            }
        }
        o
    }

    /// The claim that lets this port reuse KDA: with a broadcast scalar decay, `KdaState::step` IS
    /// the gated delta rule, over several tokens (so any per-step state drift would show up).
    #[test]
    fn gdn_step_matches_the_reference_recurrence() {
        let (d_k, d_v) = (8usize, 6usize);
        let mut state = KdaState::new(d_k, d_v);
        let mut ref_s = vec![0f32; d_k * d_v];

        for t in 0..5u32 {
            let (mut q, mut k) = (ramp(d_k, t), ramp(d_k, t + 7));
            let v = ramp(d_v, t + 13);
            prepare_qk(&mut q, &mut k);

            let (a_log, a_proj, dt_bias) = (0.3f32, -0.4f32 + t as f32 * 0.2, 0.7f32);
            let mut alpha = vec![0f32; d_k];
            alpha_from_decay(a_log, a_proj, dt_bias, &mut alpha);
            let beta = beta_from_projection(0.25 * t as f32 - 0.5);

            let mut got = vec![0f32; d_v];
            state.step(&q, &k, &v, &alpha, beta, &mut got);
            let want = reference_step(&mut ref_s, d_k, d_v, &q, &k, &v, alpha[0], beta);

            for (j, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!((g - w).abs() < 1e-5, "token {t}, dim {j}: {g} vs {w}");
            }
        }
    }

    /// The decay formula, against an independent transcription — and against
    /// `kimi_linear::kda::decay_gate`, which must agree because it's the same formula with a
    /// per-channel input.
    #[test]
    fn decay_matches_kimis_gate_formula_broadcast() {
        let d_k = 4usize;
        for &(a_log, a, dt) in &[(0.0f32, 0.0f32, 0.0f32), (0.5, -1.0, 0.7), (2.0, 3.0, -0.2), (-1.0, 0.4, 1.5)] {
            let mut alpha = vec![0f32; d_k];
            alpha_from_decay(a_log, a, dt, &mut alpha);
            let want = (-a_log.exp() * softplus(a + dt)).exp();
            assert!((alpha[0] - want).abs() < 1e-6, "a_log={a_log} a={a} dt={dt}: {} vs {want}", alpha[0]);
            assert!(alpha.iter().all(|&x| x == alpha[0]), "the scalar decay must be broadcast, not per channel");
            assert!(alpha[0] > 0.0 && alpha[0] <= 1.0, "a decay gate must land in (0, 1]: {}", alpha[0]);

            let mut per_channel = vec![0f32; d_k];
            decay_gate(a_log, &vec![a; d_k], &vec![dt; d_k], None, &mut per_channel);
            for (i, &pc) in per_channel.iter().enumerate() {
                assert!((pc - alpha[i]).abs() < 1e-6, "channel {i}: kda gate {pc} vs broadcast {}", alpha[i]);
            }
        }
    }

    /// `repeat_interleave`, not `repeat`: V heads 0-7 all read K head 0 at the real 128/16 ratio.
    #[test]
    fn value_heads_map_onto_contiguous_blocks_of_key_heads() {
        for h in 0..128usize {
            assert_eq!(key_head_of(h, 128, 16), h / 8, "V head {h}");
        }
        // the wrong (`repeat`) mapping would send V head 1 to K head 1 instead of K head 0
        assert_eq!(key_head_of(1, 128, 16), 0);
        // degenerate 1:1 case (a hypothetical checkpoint with as many K heads as V heads)
        assert_eq!(key_head_of(5, 8, 8), 5);
    }

    #[test]
    fn activate_and_split_applies_silu_then_cuts_q_k_v_in_order() {
        let (key_dim, value_dim) = (2usize, 3usize);
        let mut qkv = vec![0.0f32, 1.0, -1.0, 2.0, 10.0, -10.0, 0.5];
        assert_eq!(qkv.len(), 2 * key_dim + value_dim);
        let expected: Vec<f32> = qkv.iter().map(|&x| x / (1.0 + (-x).exp())).collect();
        let (q, k, v) = activate_and_split(&mut qkv, key_dim, value_dim);

        assert_eq!(q, &expected[..2], "q takes the first key_dim channels");
        assert_eq!(k, &expected[2..4], "k the next key_dim");
        assert_eq!(v, &expected[4..], "v the remaining value_dim");
        assert!((q[0] - 0.0).abs() < 1e-6, "silu(0) = 0");
        assert!(q[1] > 0.7 && q[1] < 0.74, "silu(1) ~ 0.731, got {}", q[1]);
    }

    #[test]
    fn prepare_qk_normalizes_both_and_scales_only_the_query() {
        let d_k = 16usize;
        let (mut q, mut k) = (vec![3.0f32; d_k], vec![4.0f32; d_k]);
        prepare_qk(&mut q, &mut k);

        let k_norm: f32 = k.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((k_norm - 1.0).abs() < 1e-4, "k must be a unit vector, got norm {k_norm}");
        let q_norm: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        let want = 1.0 / (d_k as f32).sqrt();
        assert!((q_norm - want).abs() < 1e-4, "q must be unit-normalized THEN scaled by 1/sqrt(d_k): {q_norm} vs {want}");
    }

    /// SiLU gating, not sigmoid: at `z = 0` a SiLU gate zeroes the head's output while Kimi's
    /// sigmoid gate would halve it. Checked against `kimi_linear::ops::head_output_gate` directly so
    /// the two can't silently converge.
    #[test]
    fn head_output_gate_uses_silu_not_kimis_sigmoid() {
        let head_v_dim = 4usize;
        let w = vec![1.0f32; head_v_dim];
        let z = vec![0.0f32; head_v_dim];

        let mut ours = vec![2.0f32; head_v_dim];
        head_output_gate_silu(&mut ours, &w, &z, 1e-6);
        assert!(ours.iter().all(|&x| x.abs() < 1e-6), "silu(0) = 0 must zero the output: {ours:?}");

        let mut kimis = vec![2.0f32; head_v_dim];
        crate::kimi_linear::ops::head_output_gate(&mut kimis, &w, &z, 1e-6);
        assert!(kimis.iter().all(|&x| (x - 0.5).abs() < 1e-3), "kimi's sigmoid gate halves it: {kimis:?}");

        // and with a large positive gate both approach the normalized value
        let mut big = vec![2.0f32; head_v_dim];
        head_output_gate_silu(&mut big, &w, &vec![12.0; head_v_dim], 1e-6);
        assert!(big.iter().all(|&x| x > 11.0), "silu(12) ~ 12 scales the normalized 1.0 up: {big:?}");
    }

    /// The GQA wiring inside `step_token`: with 4 V heads over 2 K heads, corrupting K head 1 must
    /// change ONLY V heads 2 and 3.
    #[test]
    fn step_token_routes_each_value_head_to_its_own_key_head() {
        let (n_v, n_k, d_k, d_v) = (4usize, 2usize, 4usize, 3usize);
        let (key_dim, value_dim) = (n_k * d_k, n_v * d_v);
        let base_qkv = {
            let mut v = ramp(2 * key_dim + value_dim, 1);
            // make K head 0 and K head 1 clearly different
            for (i, x) in v[key_dim..key_dim + d_k].iter_mut().enumerate() {
                *x = 0.5 + i as f32 * 0.1;
            }
            v
        };
        let z = vec![2.0f32; value_dim];
        let (a, b) = (vec![0.1f32; n_v], vec![0.2f32; n_v]);
        let (a_log, dt_bias) = (vec![0.0f32; n_v], vec![0.5f32; n_v]);
        let norm_w = vec![1.0f32; d_v];

        let run = |qkv: &[f32]| -> Vec<f32> {
            let mut states: Vec<KdaState> = (0..n_v).map(|_| KdaState::new(d_k, d_v)).collect();
            let mut work = qkv.to_vec();
            let mut out = vec![0f32; value_dim];
            step_token(&mut states, &mut work, &z, &a, &b, &a_log, &dt_bias, &norm_w, 1e-6, &mut out);
            out
        };

        let before = run(&base_qkv);
        let mut corrupted = base_qkv.clone();
        // K head 1 lives at [key_dim + d_k .. key_dim + 2*d_k)
        for x in corrupted[key_dim + d_k..key_dim + 2 * d_k].iter_mut() {
            *x += 3.0;
        }
        let after = run(&corrupted);

        for h in 0..n_v {
            let range = h * d_v..(h + 1) * d_v;
            let changed = before[range.clone()].iter().zip(&after[range]).any(|(x, y)| (x - y).abs() > 1e-6);
            let expect_changed = key_head_of(h, n_v, n_k) == 1;
            assert_eq!(changed, expect_changed, "V head {h} (reads K head {})", key_head_of(h, n_v, n_k));
        }
    }

    /// State really carries across tokens: feeding the same token twice must NOT give the same
    /// output (the first call leaves history behind), while a fresh state reproduces the first.
    #[test]
    fn step_token_carries_recurrent_state_across_tokens() {
        let (n_v, n_k, d_k, d_v) = (2usize, 1usize, 4usize, 3usize);
        let (key_dim, value_dim) = (n_k * d_k, n_v * d_v);
        let qkv = ramp(2 * key_dim + value_dim, 5);
        let z = vec![1.5f32; value_dim];
        let (a, b) = (vec![0.3f32; n_v], vec![0.4f32; n_v]);
        let (a_log, dt_bias) = (vec![0.2f32; n_v], vec![0.1f32; n_v]);
        let norm_w = vec![1.0f32; d_v];

        let mut states: Vec<KdaState> = (0..n_v).map(|_| KdaState::new(d_k, d_v)).collect();
        let mut first = vec![0f32; value_dim];
        step_token(&mut states, &mut qkv.clone(), &z, &a, &b, &a_log, &dt_bias, &norm_w, 1e-6, &mut first);
        let mut second = vec![0f32; value_dim];
        step_token(&mut states, &mut qkv.clone(), &z, &a, &b, &a_log, &dt_bias, &norm_w, 1e-6, &mut second);
        assert_ne!(first, second, "the recurrent state must influence the next token");

        let mut fresh: Vec<KdaState> = (0..n_v).map(|_| KdaState::new(d_k, d_v)).collect();
        let mut again = vec![0f32; value_dim];
        step_token(&mut fresh, &mut qkv.clone(), &z, &a, &b, &a_log, &dt_bias, &norm_w, 1e-6, &mut again);
        assert_eq!(first, again, "a fresh state must reproduce the first token exactly");
    }
}
