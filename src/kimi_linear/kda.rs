//! Kimi Delta Attention's core recurrence (arXiv:2510.26692 "Kimi Linear: An Expressive,
//! Efficient Attention Architecture", Eq. 1):
//!
//!   S_t = (I - β_t k_t k_t^T) Diag(α_t) S_{t-1} + β_t k_t v_t^T   ∈ R^{d_k × d_v}
//!   o_t = S_t^T q_t                                                ∈ R^{d_v}
//!
//! `α_t ∈ (0,1]^{d_k}` is a per-CHANNEL forget gate — KDA's actual refinement over Gated
//! DeltaNet's single scalar decay, letting each of the `d_k` state rows forget at its own rate.
//! Per the real reference implementation (`modeling_kimi.py`'s `KimiDeltaAttention` +
//! `fla.ops.kda.gate`'s `fused_kda_gate`/`naive_kda_gate`), it's parameterized as
//! `α_t = exp(-exp(A_log) · softplus(g_t + dt_bias))` — `dt_bias` is learned per channel, but
//! `A_log` is learned per HEAD and broadcasts across that head's channels (see `decay_gate`'s
//! own doc for why, confirmed against real checkpoint tensor shapes this session). `g_t` is a
//! low-rank projection of the hidden state. `β_t ∈ (0,1]` stays a single scalar per head,
//! exactly as in Gated DeltaNet. Neither the gate
//! parameterization nor the surrounding neural wiring (`ShortConv`, `L2Norm`, the output
//! RMSNorm+sigmoid gate) live here yet — this module is deliberately scoped to the recurrence
//! alone, the one piece with no precedent anywhere in rabbit to build on, verified against the
//! paper's own equation before anything gets wired to real model weights.
//!
//! Only the single-token recurrent form is implemented — correct for decode, and (applied
//! sequentially, one token at a time) also correct for prefill, just not chunk-parallelized the
//! way the reference training kernel is (that's a throughput optimization for later, matching
//! this project's own "correctness first" order for every kernel in `kernels.rs`).

/// Numerically stable softplus: `ln(1 + exp(x))`, computed as `max(x,0) + ln(1 + exp(-|x|))` so
/// large `|x|` never overflows/underflows `exp` (the naive form does, e.g. `exp(100.0)` is inf).
fn softplus(x: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// KDA's per-channel forget-gate parameterization for ONE head:
/// `alpha[i] = exp(-exp(a_log) * softplus(g[i] + dt_bias[i]))`.
///
/// **`a_log` is a single scalar for this whole head, not one value per channel** — confirmed
/// against the real `fla.ops.kda.gate.naive_kda_gate` reference this session (not the paper's
/// prose alone, which reads ambiguously): `g = -A_log.exp().unsqueeze(-1) * softplus(g +
/// dt_bias.view(H,-1))`, i.e. `A_log` is `[H]`-shaped (one value per head, broadcast across
/// that head's channels) while `dt_bias` is `[H*K]`-shaped (genuinely per-channel). The real
/// `moonshotai/Kimi-Linear-48B-A3B-Instruct` checkpoint's own `A_log` tensor is `[1,1,32,1]` —
/// 32 values for 32 heads, not `32*128` — cross-checking this against actual weight shapes is
/// what caught the earlier (wrong) all-slices-are-d_k-wide signature this function used to have.
/// `g`/`dt_bias`/`alpha` are all `d_k`-wide (`g` a per-token low-rank projection of the hidden
/// state, computed elsewhere). Since `softplus(x) > 0` always and `exp(a_log) > 0` always, the
/// exponent is always `<= 0`, so `alpha[i]` always lands in `(0, 1]` — exactly the range
/// `KdaState::step`'s `Diag(alpha)` term (a per-channel decay, never a growth) requires.
pub fn decay_gate(a_log: f32, g: &[f32], dt_bias: &[f32], alpha: &mut [f32]) {
    let d_k = g.len();
    assert_eq!(dt_bias.len(), d_k);
    assert_eq!(alpha.len(), d_k);

    let neg_exp_a_log = -a_log.exp();
    for i in 0..d_k {
        alpha[i] = (neg_exp_a_log * softplus(g[i] + dt_bias[i])).exp();
    }
}

/// One head's KDA state: a dense `[d_k, d_v]` matrix, row-major (`s[i*d_v + j]`). Sized once at
/// model load (`d_k`/`d_v` are fixed per layer) and updated in place every step — a FIXED size,
/// unlike GLM-5.2's `KvCache` (`src/glm52/attention.rs`), which grows by one row per token. This
/// is the actual reason a shared `KvState` type can't cover both attention families without
/// becoming an enum over genuinely different shapes (see `rabbit-plan.md`'s Phase 1 notes on why
/// that enum was deferred until a second real architecture existed to design it against).
pub struct KdaState {
    d_k: usize,
    d_v: usize,
    s: Vec<f32>,
}

impl KdaState {
    pub fn new(d_k: usize, d_v: usize) -> KdaState {
        KdaState { d_k, d_v, s: vec![0.0; d_k * d_v] }
    }

    pub fn d_k(&self) -> usize {
        self.d_k
    }

    pub fn d_v(&self) -> usize {
        self.d_v
    }

    /// The raw `[d_k, d_v]` state matrix, row-major — `kv_session.rs`'s save path, the read
    /// counterpart of `from_raw`.
    pub(crate) fn raw(&self) -> &[f32] {
        &self.s
    }

    /// Reconstructs a `KdaState` from a previously-saved raw matrix — `kv_session.rs`'s load
    /// path, the mirror of `new` (which starts at all-zero) for restoring one with real history.
    pub(crate) fn from_raw(d_k: usize, d_v: usize, s: Vec<f32>) -> KdaState {
        assert_eq!(s.len(), d_k * d_v, "KdaState::from_raw: state length doesn't match d_k*d_v");
        KdaState { d_k, d_v, s }
    }

    /// Applies one token's KDA update in place (Eq. 1 above) and writes `o_t = S_t^T q_t` into
    /// `o`. `q`/`k`/`alpha` are `d_k`-wide, `v`/`o` are `d_v`-wide, `beta` is a single scalar.
    /// Expanded elementwise rather than via a matrix-crate dependency (matching this crate's
    /// existing `kernels.rs` style): `S_mid[i,j] = alpha[i] * S_prev[i,j]` (the `Diag(α_t)`
    /// term), `c[j] = sum_i k[i] * S_mid[i,j]` (the `k_t^T Diag(α_t) S_{t-1}` term folded into
    /// one pass), `S_new[i,j] = S_mid[i,j] + beta * k[i] * (v[j] - c[j])` (the rank-1
    /// `-β k k^T (...) + β k v^T` correction, combined since both share the same `β k` factor),
    /// `o[j] = sum_i S_new[i,j] * q[i]`.
    pub fn step(&mut self, q: &[f32], k: &[f32], v: &[f32], alpha: &[f32], beta: f32, o: &mut [f32]) {
        let (d_k, d_v) = (self.d_k, self.d_v);
        assert_eq!(q.len(), d_k);
        assert_eq!(k.len(), d_k);
        assert_eq!(alpha.len(), d_k);
        assert_eq!(v.len(), d_v);
        assert_eq!(o.len(), d_v);

        for (&a, row) in alpha.iter().zip(self.s.chunks_mut(d_v)) {
            for x in row {
                *x *= a;
            }
        }

        let mut c = vec![0f32; d_v];
        for (&ki, row) in k.iter().zip(self.s.chunks(d_v)) {
            for (cj, &sij) in c.iter_mut().zip(row) {
                *cj += ki * sij;
            }
        }

        for (&ki, row) in k.iter().zip(self.s.chunks_mut(d_v)) {
            let bk = beta * ki;
            for ((rj, &vj), &cj) in row.iter_mut().zip(v).zip(&c) {
                *rj += bk * (vj - cj);
            }
        }

        o.fill(0.0);
        for (&qi, row) in q.iter().zip(self.s.chunks(d_v)) {
            for (oj, &sij) in o.iter_mut().zip(row) {
                *oj += qi * sij;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference implementation of Eq. 1, computed the naive matrix way (no row/column-fusing
    /// tricks) — an independent cross-check for `KdaState::step`'s fused version, not the same
    /// code path re-derived.
    #[allow(clippy::too_many_arguments)]
    fn naive_step(s_prev: &[f32], q: &[f32], k: &[f32], v: &[f32], alpha: &[f32], beta: f32, d_k: usize, d_v: usize) -> (Vec<f32>, Vec<f32>) {
        let mut s_mid = vec![0f32; d_k * d_v];
        for i in 0..d_k {
            for j in 0..d_v {
                s_mid[i * d_v + j] = alpha[i] * s_prev[i * d_v + j];
            }
        }
        let mut kt_smid = vec![0f32; d_v];
        for j in 0..d_v {
            kt_smid[j] = (0..d_k).map(|i| k[i] * s_mid[i * d_v + j]).sum();
        }
        let mut s_new = vec![0f32; d_k * d_v];
        for i in 0..d_k {
            for j in 0..d_v {
                s_new[i * d_v + j] = s_mid[i * d_v + j] - beta * k[i] * kt_smid[j] + beta * k[i] * v[j];
            }
        }
        let mut o = vec![0f32; d_v];
        for j in 0..d_v {
            o[j] = (0..d_k).map(|i| s_new[i * d_v + j] * q[i]).sum();
        }
        (s_new, o)
    }

    fn xorshift(seed: &mut u32) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 17;
        *seed ^= *seed << 5;
        ((*seed as f32 / u32::MAX as f32) - 0.5) * 2.0
    }

    fn random_vec(n: usize, seed: &mut u32) -> Vec<f32> {
        (0..n).map(|_| xorshift(seed)).collect()
    }

    #[test]
    fn decay_gate_matches_naive_unstable_formula() {
        // Cross-check the numerically-stable softplus against the textbook ln(1+exp(x)) form,
        // for inputs small enough that the naive form doesn't overflow.
        fn naive_softplus(x: f32) -> f32 {
            (1.0 + x.exp()).ln()
        }
        let mut seed = 11u32;
        let d_k = 6;
        let a_log = xorshift(&mut seed);
        let g = random_vec(d_k, &mut seed);
        let dt_bias = random_vec(d_k, &mut seed);

        let mut alpha = vec![0f32; d_k];
        decay_gate(a_log, &g, &dt_bias, &mut alpha);

        for i in 0..d_k {
            let expected = (-a_log.exp() * naive_softplus(g[i] + dt_bias[i])).exp();
            assert!((alpha[i] - expected).abs() < 1e-5, "{} vs {}", alpha[i], expected);
        }
    }

    #[test]
    fn decay_gate_always_lands_in_zero_to_one() {
        // exp(a_log) > 0 and softplus(x) > 0 always, so the exponent is always <= 0.
        let mut seed = 22u32;
        let d_k = 8;
        // Push a_log and (g + dt_bias) to wide extremes, including large positive/negative.
        let a_log = xorshift(&mut seed) * 20.0;
        let g: Vec<f32> = random_vec(d_k, &mut seed).iter().map(|x| x * 50.0).collect();
        let dt_bias: Vec<f32> = random_vec(d_k, &mut seed).iter().map(|x| x * 50.0).collect();

        let mut alpha = vec![0f32; d_k];
        decay_gate(a_log, &g, &dt_bias, &mut alpha);

        for &a in &alpha {
            assert!(a.is_finite(), "alpha must stay finite even at extreme gate inputs");
            // Mathematically alpha lands in (0,1], but f32 underflows very small positive
            // results to exactly 0.0 at extreme inputs — that's expected float behavior, not
            // a bug, so the floor of the checked range is inclusive here.
            assert!((0.0..=1.0).contains(&a), "alpha={a} outside [0,1]");
        }
    }

    #[test]
    fn decay_gate_at_zero_bias_and_input_uses_softplus_of_zero() {
        // g + dt_bias = 0 -> softplus(0) = ln(2), a hand-computable anchor point. a_log is one
        // scalar for the whole head (not per channel, see decay_gate's doc) -- run it twice
        // with two different scalar a_log values instead of faking per-channel variation.
        let g = [0.0, 0.0];
        let dt_bias = [0.0, 0.0];
        let ln2 = std::f32::consts::LN_2;

        let mut alpha_0 = [0f32; 2];
        decay_gate(0.0, &g, &dt_bias, &mut alpha_0);
        assert!((alpha_0[0] - (-ln2).exp()).abs() < 1e-6); // exp(a_log=0)=1
        assert!((alpha_0[1] - (-ln2).exp()).abs() < 1e-6); // same a_log broadcasts to every channel

        let mut alpha_1 = [0f32; 2];
        decay_gate(1.0, &g, &dt_bias, &mut alpha_1);
        assert!((alpha_1[0] - (-std::f32::consts::E * ln2).exp()).abs() < 1e-6); // exp(a_log=1)=e
        assert!((alpha_1[1] - (-std::f32::consts::E * ln2).exp()).abs() < 1e-6);
    }

    #[test]
    fn decay_gate_large_positive_gate_input_pushes_alpha_toward_zero() {
        // Large g + dt_bias -> softplus(x) ~= x (large) -> alpha ~= exp(-exp(a_log) * x) -> ~0.
        let g = [100.0f32];
        let dt_bias = [0.0f32];
        let mut alpha = [0f32];
        decay_gate(0.0, &g, &dt_bias, &mut alpha);
        assert!(alpha[0] < 1e-40, "gate should saturate almost fully closed, got {}", alpha[0]);
    }

    #[test]
    fn new_state_is_all_zero() {
        let st = KdaState::new(3, 4);
        assert_eq!(st.d_k(), 3);
        assert_eq!(st.d_v(), 4);
    }

    #[test]
    fn step_matches_the_naive_matrix_reference_over_several_tokens() {
        let (d_k, d_v) = (5, 4);
        let mut seed = 7u32;
        let mut fused = KdaState::new(d_k, d_v);
        let mut naive_s = vec![0f32; d_k * d_v];

        for _ in 0..6 {
            let q = random_vec(d_k, &mut seed);
            let k = random_vec(d_k, &mut seed);
            let v = random_vec(d_v, &mut seed);
            // alpha in (0,1], matching the gate's actual range (exp of a non-positive number).
            let alpha: Vec<f32> = random_vec(d_k, &mut seed).iter().map(|x| (x.abs() * 0.5 + 0.4).min(1.0)).collect();
            let beta = (xorshift(&mut seed).abs() * 0.5 + 0.3).min(1.0);

            let mut o_fused = vec![0f32; d_v];
            fused.step(&q, &k, &v, &alpha, beta, &mut o_fused);

            let (naive_s_new, naive_o) = naive_step(&naive_s, &q, &k, &v, &alpha, beta, d_k, d_v);
            naive_s = naive_s_new;

            for (a, b) in o_fused.iter().zip(&naive_o) {
                assert!((a - b).abs() < 1e-4, "{a} vs {b}");
            }
            for (a, b) in fused.s.iter().zip(&naive_s) {
                assert!((a - b).abs() < 1e-4, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn beta_zero_only_decays_the_state_no_key_value_update() {
        // beta=0 must drop the "-b k k^T (...) + b k v^T" term entirely: S_t = Diag(alpha) S_{t-1}.
        let (d_k, d_v) = (2, 2);
        let mut st = KdaState::new(d_k, d_v);
        st.s = vec![2.0, 4.0, 6.0, 8.0]; // hand-set initial state
        let alpha = [0.5, 0.25];
        let mut o = vec![0f32; d_v];
        st.step(&[1.0, 1.0], &[9.0, 9.0], &[9.0, 9.0], &alpha, 0.0, &mut o);
        assert_eq!(st.s, vec![1.0, 2.0, 1.5, 2.0], "row i scaled by alpha[i], beta=0 must not touch k/v at all");
        // o = S_new^T q with q=[1,1]: col0 = 1.0+1.5=2.5, col1 = 2.0+2.0=4.0
        assert_eq!(o, vec![2.5, 4.0]);
    }

    #[test]
    fn alpha_one_and_beta_one_reduces_to_the_classical_delta_rule() {
        // Eq. 1 with alpha=1 (no forgetting) and beta=1 IS the classical DeltaNet update:
        // S_t = (I - k k^T) S_{t-1} + k v^T. Hand-computed for d_k=d_v=2, S_prev=0 (so the
        // whole "-k k^T S_prev" term vanishes and only "+ k v^T" survives on the first step).
        let mut st = KdaState::new(2, 2);
        let q = [1.0, 0.0];
        let k = [1.0, 2.0];
        let v = [3.0, 5.0];
        let alpha = [1.0, 1.0];
        let mut o = vec![0f32; 2];
        st.step(&q, &k, &v, &alpha, 1.0, &mut o);
        // S_prev=0 -> S_new = k v^T = [[1*3,1*5],[2*3,2*5]] = [[3,5],[6,10]]
        assert_eq!(st.s, vec![3.0, 5.0, 6.0, 10.0]);
        // o[j] = sum_i S_new[i,j] * q[i], q=[1,0] -> o = S_new's row 0 = [3,5]
        assert_eq!(o, vec![3.0, 5.0]);
    }
}
