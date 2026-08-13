//! Qwen 3.8's full-attention layers: plain **GQA** — 64 query heads over 4 KV heads,
//! `head_dim: 256` — with three twists that GLM-5.2's and Kimi's MLA paths have no equivalent of.
//! This module is the math only (pure `f32` slices, no weights, no `QT`); the projections and layer
//! wiring live in the model/generate loaders.
//!
//! Ported from `transformers`' real `Qwen3_5MoeAttention.forward` +
//! `Qwen3_5MoeTextRotaryEmbedding` + `apply_rotary_pos_emb` (`modeling_qwen3_5_moe.py`, read this
//! session — not from the paper or from Qwen3-Next's docs):
//!
//! **1. `q_proj` emits query AND gate, interleaved PER HEAD.** It projects to
//! `2 * n_heads * head_dim`, and the reference splits it as
//! `chunk(q_proj(x).view(*input_shape, -1, head_dim * 2), 2, dim=-1)` — i.e. the row is
//! `[head0_query(256), head0_gate(256), head1_query(256), head1_gate(256), ...]`, NOT
//! `[all queries..., all gates...]`. Reading it as two contiguous halves would silently mix head 32's
//! query into head 0's gate and produce plausible-looking garbage; [`split_query_gate`] pins the
//! real layout, and its test would fail under the wrong one.
//!
//! **2. Partial RoPE with the `rotate_half` convention.** `rope_dim = head_dim *
//! partial_rotary_factor` = `256 * 0.25` = **64**: only the first 64 dims of each head rotate, the
//! remaining 192 pass through untouched (NoPE). And the rotation splits that 64-wide block in
//! HALVES (`rotate_half`: `(-x[32..64], x[0..32])`), unlike `glm52::attention::rope_interleave`'s
//! adjacent-pair reading — same theta, same positions, different permutation, so the existing helper
//! cannot be reused. See [`rope_half`].
//!
//! **mRoPE is a no-op here.** The reference rotary builds three position grids (text/height/width)
//! and interleaves them via `apply_interleaved_mrope`. For text-only input all three grids hold the
//! SAME `position_ids` (the code expands a 2-D `position_ids` into three identical copies), so the
//! interleave rewrites each frequency with an equal value and the result is plain RoPE. Qwen3.8-Max's
//! `rope_parameters` carries no `mrope_section` at all — that field only appears on the multimodal
//! siblings. Nothing to port; stated here because the reference code looks like it matters.
//!
//! **3. A sigmoid output gate.** `attn_output = attn_output * torch.sigmoid(gate)`, applied to the
//! per-head context BEFORE `o_proj` — plain sigmoid gating, elementwise. `config.output_gate_type`
//! is `"swish"` but the attention path applies `sigmoid(gate)`, not `gate * sigmoid(gate)`; the
//! reference is explicit and `qwen38::config` validates the field rather than branching on it. See
//! [`apply_output_gate`].
//!
//! `q_norm`/`k_norm` are RMSNorm over `head_dim` ONLY (per head, not over the whole row — the
//! reference even comments "unlike olmo, only on the head dim!"), applied to the query AFTER the
//! gate is split off and to the key after `k_proj`, and BEFORE RoPE. They are
//! `Qwen3_5MoeRMSNorm` instances, so they scale by `(1 + weight)` and NOT by `weight` like every
//! other family's norms in this crate — [`norm_heads`] goes through
//! [`crate::qwen38::ops::rmsnorm_1p`]; see that module's doc for why using the usual one would
//! quietly collapse the activations instead of failing.

use crate::glm52::attention::softmax;
use crate::kimi_linear::ops::sigmoid;
use crate::qwen38::ops::rmsnorm_1p;

/// One full-attention layer's KV cache: ordinary GQA, `n_kv_heads * head_dim` floats of K and of V
/// per absolute position, appended in order. Nothing compressed (no `kv_lora` latent like MLA), so
/// it's the cheapest cache in this crate: 4 heads x 256 dims x 2 x 4 bytes = **8 KB per token per
/// layer**, and only 23 of the 92 layers have one at all.
pub struct KvCache {
    n_kv_heads: usize,
    head_dim: usize,
    k: Vec<f32>,
    v: Vec<f32>,
}

impl KvCache {
    pub fn new(n_kv_heads: usize, head_dim: usize) -> KvCache {
        KvCache { n_kv_heads, head_dim, k: Vec::new(), v: Vec::new() }
    }

    /// Cached positions.
    pub fn len(&self) -> usize {
        let row = self.n_kv_heads * self.head_dim;
        self.k.len().checked_div(row).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Appends one position's keys and values, each `n_kv_heads * head_dim` wide (head-major).
    pub fn push(&mut self, k_row: &[f32], v_row: &[f32]) {
        let row = self.n_kv_heads * self.head_dim;
        assert_eq!(k_row.len(), row, "k row must be n_kv_heads * head_dim");
        assert_eq!(v_row.len(), row, "v row must be n_kv_heads * head_dim");
        self.k.extend_from_slice(k_row);
        self.v.extend_from_slice(v_row);
    }

    fn head_slice(buf: &[f32], pos: usize, kv_head: usize, n_kv_heads: usize, head_dim: usize) -> &[f32] {
        let off = (pos * n_kv_heads + kv_head) * head_dim;
        &buf[off..off + head_dim]
    }

    pub fn k_at(&self, pos: usize, kv_head: usize) -> &[f32] {
        Self::head_slice(&self.k, pos, kv_head, self.n_kv_heads, self.head_dim)
    }

    pub fn v_at(&self, pos: usize, kv_head: usize) -> &[f32] {
        Self::head_slice(&self.v, pos, kv_head, self.n_kv_heads, self.head_dim)
    }

    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// The K and V rows for positions `from..to`, as flat slices — `kv_session.rs`'s save path,
    /// which appends only the rows a turn actually added.
    pub(crate) fn rows(&self, from: usize, to: usize) -> (&[f32], &[f32]) {
        let row = self.n_kv_heads * self.head_dim;
        (&self.k[from * row..to * row], &self.v[from * row..to * row])
    }

    /// Rebuilds a cache from previously-saved rows — `kv_session.rs`'s load path, the counterpart of
    /// repeated [`KvCache::push`] calls.
    pub(crate) fn from_raw(n_kv_heads: usize, head_dim: usize, k: Vec<f32>, v: Vec<f32>) -> KvCache {
        assert_eq!(k.len(), v.len(), "K and V must cover the same positions");
        assert!(k.len().is_multiple_of(n_kv_heads * head_dim), "saved rows must be whole positions");
        KvCache { n_kv_heads, head_dim, k, v }
    }
}

/// Splits `q_proj`'s `2 * n_heads * head_dim` output into `(query, gate)`, each
/// `n_heads * head_dim`. See this module's doc, point 1: the split is per head, so head `h`'s query
/// is `qg[h * 2 * head_dim ..][..head_dim]` and its gate the `head_dim` right after it.
pub fn split_query_gate(qg: &[f32], head_dim: usize) -> (Vec<f32>, Vec<f32>) {
    assert!(qg.len().is_multiple_of(2 * head_dim), "q_proj output must be a multiple of 2 * head_dim");
    let n_heads = qg.len() / (2 * head_dim);
    let mut query = Vec::with_capacity(n_heads * head_dim);
    let mut gate = Vec::with_capacity(n_heads * head_dim);
    for h in 0..n_heads {
        let base = h * 2 * head_dim;
        query.extend_from_slice(&qg[base..base + head_dim]);
        gate.extend_from_slice(&qg[base + head_dim..base + 2 * head_dim]);
    }
    (query, gate)
}

/// RMSNorm over each `head_dim`-wide head of `x` independently (`q_norm`/`k_norm`). `weight` is one
/// `head_dim`-wide vector shared by every head, as in the reference.
///
/// Uses [`crate::qwen38::ops::rmsnorm_1p`], NOT the crate's usual RMSNorm: `q_norm`/`k_norm` are
/// `Qwen3_5MoeRMSNorm` instances, which scale by `(1 + weight)` — see that module's doc.
pub fn norm_heads(x: &mut [f32], head_dim: usize, weight: &[f32], eps: f32) {
    debug_assert_eq!(weight.len(), head_dim);
    for head in x.chunks_mut(head_dim) {
        rmsnorm_1p(head, weight, eps);
    }
}

/// Qwen's partial RoPE with the `rotate_half` convention, in place, for ONE head's `head_dim`-wide
/// slice: rotates `v[..rope_dim]`, leaves `v[rope_dim..]` alone.
///
/// `q_embed = q_rot * cos + rotate_half(q_rot) * sin` where `cos`/`sin` are `cat(freqs, freqs)` over
/// `half = rope_dim / 2` frequencies, so for `j < half`:
///
/// ```text
/// out[j]        = v[j]        * cos_j - v[j + half] * sin_j
/// out[j + half] = v[j + half] * cos_j + v[j]        * sin_j
/// ```
///
/// with `cos_j = cos(pos * theta^(-2j / rope_dim))`. Deliberately NOT
/// `glm52::attention::rope_interleave`, which reads `(v[2j], v[2j+1])` pairs instead — see this
/// module's doc, point 2.
pub fn rope_half(v: &mut [f32], rope_dim: usize, pos: usize, theta: f32) {
    assert!(rope_dim <= v.len(), "rope_dim must not exceed the head width");
    assert!(rope_dim.is_multiple_of(2), "rope_dim must be even");
    let half = rope_dim / 2;
    for j in 0..half {
        let inv = theta.powf(-2.0 * j as f32 / rope_dim as f32);
        let (sn, cs) = (pos as f32 * inv).sin_cos();
        let a = v[j];
        let b = v[j + half];
        v[j] = a * cs - b * sn;
        v[j + half] = b * cs + a * sn;
    }
}

/// Causal GQA attention for the tokens already in `kv` (the newest position must be the last one
/// pushed): for each of `q`'s heads, scores against every cached position, softmax, then the
/// value-weighted sum. `out` is `n_heads * head_dim` wide, same as `q`.
///
/// Query head `h` reads KV head `h / (n_heads / n_kv_heads)` — integer division, matching
/// `repeat_kv`'s `expand(..., n_rep, ...)` layout, which maps KV head `j` onto the CONTIGUOUS block
/// of query heads `j * n_rep .. (j + 1) * n_rep`. The other plausible mapping (`h % n_kv_heads`)
/// would pair every head with the wrong keys while keeping all the shapes valid.
///
/// Serial over heads for now: correctness first, in the same order every kernel in this crate was
/// built (`kernels.rs`' own SIMD paths came after their scalar versions). The per-head loop is
/// embarrassingly parallel and `glm52::attention` already shows the `rayon` shape to copy when this
/// starts mattering — with 23 such layers against 69 recurrent ones, it won't be the first thing to
/// profile.
pub fn attend(q: &[f32], kv: &KvCache, scale: f32, out: &mut [f32]) {
    let head_dim = kv.head_dim;
    let n_heads = q.len() / head_dim;
    assert!(q.len().is_multiple_of(head_dim), "query width must be a multiple of head_dim");
    assert_eq!(out.len(), q.len(), "out must match the query width");
    assert!(n_heads >= kv.n_kv_heads && n_heads.is_multiple_of(kv.n_kv_heads), "n_heads must be a multiple of n_kv_heads");
    let n_rep = n_heads / kv.n_kv_heads;
    let positions = kv.len();

    let mut scores = vec![0f32; positions];
    for h in 0..n_heads {
        let kv_head = h / n_rep;
        let qh = &q[h * head_dim..(h + 1) * head_dim];
        for (t, score) in scores.iter_mut().enumerate() {
            let k = kv.k_at(t, kv_head);
            *score = qh.iter().zip(k).map(|(&a, &b)| a * b).sum::<f32>() * scale;
        }
        softmax(&mut scores);
        let dst = &mut out[h * head_dim..(h + 1) * head_dim];
        dst.fill(0.0);
        for (t, &w) in scores.iter().enumerate() {
            let v = kv.v_at(t, kv_head);
            for (d, &vi) in dst.iter_mut().zip(v) {
                *d += w * vi;
            }
        }
    }
}

/// `attn_output * sigmoid(gate)`, elementwise over the whole `n_heads * head_dim` context — applied
/// before `o_proj`. See this module's doc, point 3 (sigmoid, not silu, despite
/// `output_gate_type: "swish"`).
pub fn apply_output_gate(ctx: &mut [f32], gate: &[f32]) {
    assert_eq!(ctx.len(), gate.len(), "the gate must be as wide as the attention context");
    for (c, &g) in ctx.iter_mut().zip(gate) {
        *c *= sigmoid(g);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference formula, written independently of `rope_half`'s loop: build the `cat(freqs,
    /// freqs)` cos/sin vectors, then `q * cos + rotate_half(q) * sin` literally.
    fn reference_rope(v: &[f32], rope_dim: usize, pos: usize, theta: f32) -> Vec<f32> {
        let half = rope_dim / 2;
        let freqs: Vec<f32> = (0..half).map(|j| pos as f32 / theta.powf(2.0 * j as f32 / rope_dim as f32)).collect();
        let cos: Vec<f32> = freqs.iter().chain(freqs.iter()).map(|f| f.cos()).collect();
        let sin: Vec<f32> = freqs.iter().chain(freqs.iter()).map(|f| f.sin()).collect();
        let rot: Vec<f32> = v[..rope_dim].to_vec();
        // rotate_half: (-x2, x1)
        let rotated: Vec<f32> =
            rot[half..].iter().map(|x| -x).chain(rot[..half].iter().copied()).collect();
        let mut out: Vec<f32> = (0..rope_dim).map(|i| rot[i] * cos[i] + rotated[i] * sin[i]).collect();
        out.extend_from_slice(&v[rope_dim..]);
        out
    }

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i * 37 % 19) as f32 - 9.0) / 4.0).collect()
    }

    #[test]
    fn rope_half_matches_the_reference_cos_sin_formula() {
        let (head_dim, rope_dim, theta) = (256usize, 64usize, 1e7f32);
        for pos in [0usize, 1, 7, 1000] {
            let v = ramp(head_dim);
            let mut got = v.clone();
            rope_half(&mut got, rope_dim, pos, theta);
            let want = reference_rope(&v, rope_dim, pos, theta);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!((g - w).abs() < 1e-4, "pos {pos}, dim {i}: {g} vs {w}");
            }
        }
    }

    #[test]
    fn rope_half_is_identity_at_position_zero_and_never_touches_the_nope_tail() {
        let v = ramp(256);
        let mut at_zero = v.clone();
        rope_half(&mut at_zero, 64, 0, 1e7);
        assert_eq!(at_zero, v, "cos=1, sin=0 at position 0");

        let mut rotated = v.clone();
        rope_half(&mut rotated, 64, 13, 1e7);
        assert_eq!(&rotated[64..], &v[64..], "dims 64..256 must pass through unrotated");
        assert_ne!(&rotated[..64], &v[..64], "dims 0..64 must actually rotate");
    }

    /// A rotation preserves the norm of the rotated block, and pairs `(j, j + half)` are what get
    /// mixed — a pair-adjacent (GLM-style) implementation would fail the per-pair check even though
    /// it also preserves the total norm.
    #[test]
    fn rope_half_rotates_the_j_and_j_plus_half_pairs() {
        let (rope_dim, half) = (64usize, 32usize);
        let v = ramp(rope_dim);
        let mut got = v.clone();
        rope_half(&mut got, rope_dim, 5, 1e7);
        for j in 0..half {
            let before = v[j] * v[j] + v[j + half] * v[j + half];
            let after = got[j] * got[j] + got[j + half] * got[j + half];
            assert!((before - after).abs() < 1e-3, "pair ({j},{}) must keep its norm: {before} vs {after}", j + half);
        }
    }

    /// The layout trap: query and gate interleave PER HEAD. Under the "two contiguous halves"
    /// reading, head 0's gate would come out as head `n_heads/2`'s query.
    #[test]
    fn split_query_gate_splits_within_each_head_not_across_the_row() {
        let (n_heads, head_dim) = (4usize, 3usize);
        // head h: query = [h*10 + 0,1,2], gate = [h*10 + 5,6,7]
        let mut qg = Vec::new();
        for h in 0..n_heads {
            qg.extend((0..head_dim).map(|d| (h * 10 + d) as f32));
            qg.extend((0..head_dim).map(|d| (h * 10 + 5 + d) as f32));
        }
        let (q, g) = split_query_gate(&qg, head_dim);
        assert_eq!(q, vec![0., 1., 2., 10., 11., 12., 20., 21., 22., 30., 31., 32.]);
        assert_eq!(g, vec![5., 6., 7., 15., 16., 17., 25., 26., 27., 35., 36., 37.]);

        // what the wrong reading would have produced, spelled out so the intent can't drift
        let wrong_first_half = &qg[..n_heads * head_dim];
        assert_ne!(q.as_slice(), wrong_first_half);
    }

    #[test]
    fn norm_heads_normalizes_each_head_independently() {
        let head_dim = 4usize;
        // weight 0.0 is the identity scale under `(1 + w)` — see `qwen38::ops`
        let weight = vec![0.0f32; head_dim];
        // Two heads at wildly different scales must land at the same magnitude after RMSNorm.
        // Both are chosen well above `eps` (an eps-dominated head, e.g. all 1e-3 against
        // eps = 1e-6, normalizes to 0.707 instead of 1.0 — real behavior, but it would make this
        // test about eps rather than about per-head independence).
        let mut x = vec![2.0, 2.0, 2.0, 2.0, 100.0, 100.0, 100.0, 100.0];
        norm_heads(&mut x, head_dim, &weight, 1e-6);
        for i in 0..head_dim {
            assert!((x[i] - x[head_dim + i]).abs() < 1e-3, "head 0 and head 1 must normalize alike: {x:?}");
        }
        assert!((x[0] - 1.0).abs() < 1e-3, "an all-equal head normalizes to 1.0 with unit weight");
    }

    #[test]
    fn attending_over_a_single_cached_position_returns_that_positions_value() {
        let (n_kv_heads, head_dim) = (1usize, 4usize);
        let mut kv = KvCache::new(n_kv_heads, head_dim);
        assert!(kv.is_empty());
        kv.push(&[1.0, 0.0, 0.0, 0.0], &[7.0, 8.0, 9.0, 10.0]);
        assert_eq!(kv.len(), 1);

        let q = vec![0.5, 0.5, 0.5, 0.5];
        let mut out = vec![0f32; 4];
        attend(&q, &kv, 1.0 / (head_dim as f32).sqrt(), &mut out);
        assert_eq!(out, vec![7.0, 8.0, 9.0, 10.0], "softmax over one position is 1.0");
    }

    /// Equal scores over two positions must average their values — pins that the softmax runs over
    /// the WHOLE cache, not just the newest row.
    #[test]
    fn attention_weights_are_a_softmax_over_the_whole_cache() {
        let mut kv = KvCache::new(1, 2);
        kv.push(&[0.0, 0.0], &[0.0, 0.0]); // score 0 regardless of q
        kv.push(&[0.0, 0.0], &[4.0, 8.0]);
        let mut out = vec![0f32; 2];
        attend(&[1.0, 1.0], &kv, 1.0, &mut out);
        assert_eq!(out, vec![2.0, 4.0], "two equal scores -> mean of the two values");
    }

    /// GQA head mapping: with 8 query heads over 2 KV heads, query heads 0-3 must read KV head 0 and
    /// 4-7 KV head 1 (`h / n_rep`). The `h % n_kv_heads` mistake would give 0,1,0,1,... — same
    /// shapes, wrong keys.
    #[test]
    fn query_heads_map_onto_contiguous_blocks_of_kv_heads() {
        let (n_heads, n_kv_heads, head_dim) = (8usize, 2usize, 2usize);
        let mut kv = KvCache::new(n_kv_heads, head_dim);
        // kv head 0 -> value [1,1]; kv head 1 -> value [9,9]
        kv.push(&[1.0, 0.0, 1.0, 0.0], &[1.0, 1.0, 9.0, 9.0]);

        let q = vec![1.0f32; n_heads * head_dim];
        let mut out = vec![0f32; n_heads * head_dim];
        attend(&q, &kv, 1.0, &mut out);

        for h in 0..n_heads {
            let want = if h < 4 { 1.0 } else { 9.0 };
            assert_eq!(&out[h * head_dim..(h + 1) * head_dim], &[want, want], "query head {h}");
        }
    }

    #[test]
    fn output_gate_multiplies_by_sigmoid_not_silu() {
        let mut ctx = vec![2.0f32, 2.0, 2.0];
        let gate = vec![0.0f32, 10.0, -10.0];
        apply_output_gate(&mut ctx, &gate);
        assert!((ctx[0] - 1.0).abs() < 1e-6, "sigmoid(0) = 0.5 -> 2 * 0.5 = 1");
        assert!((ctx[1] - 2.0).abs() < 1e-3, "sigmoid(10) ~ 1");
        assert!(ctx[2].abs() < 1e-3, "sigmoid(-10) ~ 0");
        // silu gating would have given 2 * (0 * 0.5) = 0 for the first element
        assert!(ctx[0] > 0.5, "a silu gate would have zeroed this one");
    }
}
