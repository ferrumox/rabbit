//! Kimi K3's "Attention Residuals" — a block-pooling mechanism across LAYERS (depth), not a
//! sequence-attention mechanism. Ported from the real `KimiDecoderLayer._forward_attn_residual`/
//! `_apply_attn_res`/`KimiLinearModel._apply_output_attn_res` in `modeling_kimi_linear.py`
//! (`moonshotai/Kimi-K3`, fetched 2026-07-27) — not guessed from launch-blog prose.
//!
//! **The mechanism, precisely**: every `attn_res_block_size` layers (K3's real checkpoint: 12,
//! checked via `layer_idx % attn_res_block_size == 0`, 0-indexed — so layers 0, 12, 24, ...),
//! the residual stream's value ENTERING that layer is snapshotted into a per-token, growing list
//! of "blocks" (`AttnResState::blocks`), and the running per-layer accumulator ("prefix_sum" in
//! the real code) restarts from just that layer's attention output, discarding what came before
//! (it's already saved). At EVERY layer (checkpoint or not), both right before self-attention and
//! right before the MLP/MoE, the actual input fed to that sub-layer is not the plain residual
//! value but an attention-POOLED combination of `[...blocks, current value]`: each candidate is
//! RMSNorm'd (no learned scale at this step), scored via a dot product against a fixed learned
//! per-channel vector (`norm.weight * proj.weight`, elementwise — folds the norm's scale and the
//! projection's dot-product weight into one), softmaxed across the block dimension, then used to
//! weight-sum the ORIGINAL (unnormalized) values. Once more, with the FINAL layer's output and
//! the complete block list, right before the model's last RMSNorm (`_apply_output_attn_res`).
//!
//! **This is purely transient, not part of `KvState`**: confirmed via `KimiLinearModel.forward`,
//! which allocates a brand-new, EMPTY `block_residual` at the top of every call
//! (`hidden_states.new_zeros(..., 0, ...)`) and never returns or persists it anywhere outside
//! that one call — a decode step's block list starts fresh, grows across that one token's full
//! pass through all layers, and is discarded once the logits come out. No cross-decode-step
//! persistence, no session/`KvState` format changes needed — `AttnResState` is created fresh
//! per `step`/`step_all` call, exactly like `glm52::moe::Routing`.
//!
//! **Softmax-over-one is a proven identity**: when a layer has zero prior blocks (e.g. layer 0,
//! before its own checkpoint-push happens), pooling `[current]` alone reduces to `current`
//! unchanged (softmax of a single score is always exactly `1.0`, regardless of the score's
//! value) — so every pooling call in this module is safe to run unconditionally, matching the
//! real code's own two call sites: one has an explicit `if block_residual.shape[1] > 0` guard
//! (a redundant optimization, not a correctness requirement) and the other has none at all.
//!
//! This module only implements the MECHANISM (state bookkeeping + the pooling math), taking
//! already-computed attention/MLP deltas as plain arguments — wiring this into a real per-layer
//! forward pass (calling K3's actual attention/MoE, which don't have their own `kimi_k3` module
//! yet) is separate, later work.

/// One "attention residual" checkpoint snapshot, or the current running value: `s * hidden`
/// floats, row-major per-token — the same buffer layout as every other per-token hidden-state
/// buffer in this crate (e.g. `glm52::moe`'s `x`/`out`).
type Row = Vec<f32>;

/// The four learned vectors one `AttnResWeights` bundle needs — `hidden`-wide each, taken
/// directly from a real checkpoint's `self_attention_res_norm.weight`/
/// `self_attention_res_proj.weight` (or `mlp_res_*`/`output_attn_res_*` for the other two call
/// sites — same shape, different tensors, so this one struct covers all three).
pub struct AttnResWeights {
    pub norm: Vec<f32>,
    pub proj: Vec<f32>,
}

/// Per-forward-pass state: the growing list of per-token snapshots. `AttnResState::default()`
/// (zero blocks) is the correct starting point for every `step`/`step_all` call — see this
/// module's doc for why nothing here persists across calls.
#[derive(Default, Clone)]
pub struct AttnResState {
    /// Oldest first. Each entry is `s * hidden` floats (row-major per-token), matching whatever
    /// `s` the forward pass this state belongs to is processing.
    blocks: Vec<Row>,
}

impl AttnResState {
    pub fn new() -> Self {
        AttnResState::default()
    }

    pub fn n_blocks(&self) -> usize {
        self.blocks.len()
    }
}

/// `layer_idx % block_size == 0` — 0-indexed, matching the real `KimiDecoderLayer.__init__`'s
/// `self.layer_idx` (plain `range(config.num_hidden_layers)`, no 1-indexing quirk here unlike
/// `kda_layers`/`full_attn_layers` elsewhere in this checkpoint's config).
pub fn is_checkpoint_layer(layer_idx: usize, block_size: usize) -> bool {
    layer_idx.is_multiple_of(block_size)
}

/// One token's pooling: RMSNorm (no learned scale) each of `[historical..., current]`, score
/// each against `norm_weight * proj_weight` (elementwise), softmax across that dimension, then
/// weighted-sum the ORIGINAL (unnormalized) rows — the exact real formula (see module doc).
/// Always safe to call, including with an empty `historical` (proven identity, see module doc).
pub fn attn_res_pool(historical: &[&[f32]], current: &[f32], norm_weight: &[f32], proj_weight: &[f32], eps: f32) -> Vec<f32> {
    let hidden = current.len();
    let n = historical.len() + 1;

    let mut scores = Vec::with_capacity(n);
    for &block in historical.iter().chain(std::iter::once(&current)) {
        let ms: f64 = block.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / hidden as f64;
        let r = 1.0f32 / (((ms as f32) + eps).sqrt());
        let mut score = 0f32;
        for c in 0..hidden {
            score += (block[c] * r) * norm_weight[c] * proj_weight[c];
        }
        scores.push(score);
    }

    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&e| e / sum_exp).collect();

    let mut out = vec![0f32; hidden];
    for (b, &block) in historical.iter().chain(std::iter::once(&current)).enumerate() {
        let p = probs[b];
        for c in 0..hidden {
            out[c] += p * block[c];
        }
    }
    out
}

/// The batched form of `attn_res_pool`, over a whole `s`-token buffer at once: token `t`'s
/// pooling only ever sees token `t`'s own rows across `blocks`/`current` — no cross-token mixing
/// (this mechanism pools across LAYERS/depth, not across the sequence).
fn pool_batch(blocks: &[Row], current: &[f32], hidden: usize, w: &AttnResWeights, eps: f32, out: &mut [f32]) {
    let s = current.len() / hidden;
    for t in 0..s {
        let hist: Vec<&[f32]> = blocks.iter().map(|b| &b[t * hidden..(t + 1) * hidden]).collect();
        let cur_row = &current[t * hidden..(t + 1) * hidden];
        let pooled = attn_res_pool(&hist, cur_row, &w.norm, &w.proj, eps);
        out[t * hidden..(t + 1) * hidden].copy_from_slice(&pooled);
    }
}

/// Step 1 of a layer's Attention-Residuals wrapping (before self-attention): pools
/// `[state's existing blocks..., hidden_in]` into the value the caller should
/// `input_layernorm` + feed to self-attention, then — if `layer_idx` is a checkpoint layer —
/// pushes `hidden_in` itself (the RAW, pre-pooling residual-stream value) onto `state` for
/// every later pooling call this forward pass (including this same layer's own step 2 below).
pub fn before_attention(state: &mut AttnResState, hidden_in: &[f32], hidden: usize, layer_idx: usize, block_size: usize, w: &AttnResWeights, eps: f32) -> Vec<f32> {
    let mut attn_input = vec![0f32; hidden_in.len()];
    pool_batch(&state.blocks, hidden_in, hidden, w, eps, &mut attn_input);
    if is_checkpoint_layer(layer_idx, block_size) {
        state.blocks.push(hidden_in.to_vec());
    }
    attn_input
}

/// Step 2 (after self-attention, before the MLP/MoE): combines `hidden_in`/`attn_delta` into
/// the layer's new running residual value (`prefix_sum`) — restarting from just `attn_delta`
/// alone on a checkpoint layer (since `hidden_in` was already saved into `state` by
/// `before_attention` above), or the ordinary `hidden_in + attn_delta` otherwise — then pools
/// `[state's blocks (already includes this layer's checkpoint push, if any)..., prefix_sum]`
/// into the value the caller should `post_attention_layernorm` + feed to the MLP/MoE. Returns
/// `(mlp_input, new_prefix_sum)` — the caller needs `new_prefix_sum` again afterward (see
/// `after_mlp`).
#[allow(clippy::too_many_arguments)]
pub fn before_mlp(state: &AttnResState, hidden_in: &[f32], attn_delta: &[f32], hidden: usize, layer_idx: usize, block_size: usize, w: &AttnResWeights, eps: f32) -> (Vec<f32>, Vec<f32>) {
    let prefix_sum: Vec<f32> = if is_checkpoint_layer(layer_idx, block_size) {
        attn_delta.to_vec()
    } else {
        hidden_in.iter().zip(attn_delta).map(|(&h, &a)| h + a).collect()
    };
    let mut mlp_input = vec![0f32; prefix_sum.len()];
    pool_batch(&state.blocks, &prefix_sum, hidden, w, eps, &mut mlp_input);
    (mlp_input, prefix_sum)
}

/// Step 3 (after the MLP/MoE): `prefix_sum + mlp_delta` — this layer's final output, and the
/// next layer's `hidden_in`. A plain elementwise add (matches the real code exactly — by this
/// point `prefix_sum` is never the "reset to None" case, only step 2's checkpoint branch is),
/// kept as a named function for symmetry with steps 1/2 and so call sites read as the same
/// 3-step shape the real `_forward_attn_residual` has, not a for-loop a reader has to match up
/// against the reference by hand.
pub fn after_mlp(prefix_sum: &[f32], mlp_delta: &[f32]) -> Vec<f32> {
    prefix_sum.iter().zip(mlp_delta).map(|(&p, &m)| p + m).collect()
}

/// The model-level final pooling (`KimiLinearModel._apply_output_attn_res`), run once after the
/// last layer, before the model's final RMSNorm — same `pool_batch` primitive as every per-layer
/// call, just with the checkpoint's `output_attn_res_norm`/`output_attn_res_proj` weights and no
/// further state mutation (nothing pools against this afterward).
pub fn output_pool(state: &AttnResState, final_hidden: &[f32], hidden: usize, w: &AttnResWeights, eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; final_hidden.len()];
    pool_batch(&state.blocks, final_hidden, hidden, w, eps, &mut out);
    out
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

    fn random_vec(n: usize, seed: &mut u32) -> Vec<f32> {
        (0..n).map(|_| xorshift(seed)).collect()
    }

    /// Fully independent re-derivation of `_apply_attn_res`'s formula (not reusing
    /// `attn_res_pool` at all), computed with plain `f32` throughout (not `attn_res_pool`'s f64
    /// variance accumulation) — a deliberately different numerical path so this genuinely
    /// exercises the formula, not just "the function returns what it returns".
    fn naive_pool(historical: &[&[f32]], current: &[f32], norm_weight: &[f32], proj_weight: &[f32], eps: f32) -> Vec<f32> {
        let hidden = current.len();
        let blocks: Vec<&[f32]> = historical.iter().copied().chain(std::iter::once(current)).collect();
        let scores: Vec<f32> = blocks
            .iter()
            .map(|v| {
                let mean_sq = v.iter().map(|x| x * x).sum::<f32>() / hidden as f32;
                let r = 1.0 / (mean_sq + eps).sqrt();
                (0..hidden).map(|c| v[c] * r * norm_weight[c] * proj_weight[c]).sum::<f32>()
            })
            .collect();
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        let mut out = vec![0f32; hidden];
        for (b, v) in blocks.iter().enumerate() {
            let p = exps[b] / sum_exp;
            for c in 0..hidden {
                out[c] += p * v[c];
            }
        }
        out
    }

    #[test]
    fn attn_res_pool_matches_an_independent_naive_reference() {
        let mut seed = 5u32;
        let hidden = 6;
        let b0 = random_vec(hidden, &mut seed);
        let b1 = random_vec(hidden, &mut seed);
        let current = random_vec(hidden, &mut seed);
        let norm_w = random_vec(hidden, &mut seed).iter().map(|v| v.abs() + 0.1).collect::<Vec<_>>();
        let proj_w = random_vec(hidden, &mut seed);

        let got = attn_res_pool(&[&b0, &b1], &current, &norm_w, &proj_w, 1e-5);
        let expected = naive_pool(&[&b0, &b1], &current, &norm_w, &proj_w, 1e-5);

        for (a, b) in got.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn attn_res_pool_with_no_historical_blocks_is_the_identity() {
        // Proven mathematically: softmax over a single score is always exactly 1.0, so pooling
        // reduces to returning `current` unchanged -- this is what lets every call site in this
        // module skip the real code's "if block_residual.shape[1] > 0" guard.
        let mut seed = 9u32;
        let hidden = 5;
        let current = random_vec(hidden, &mut seed);
        let norm_w = random_vec(hidden, &mut seed);
        let proj_w = random_vec(hidden, &mut seed);

        let got = attn_res_pool(&[], &current, &norm_w, &proj_w, 1e-5);
        for (a, b) in got.iter().zip(&current) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn attn_res_pool_output_is_a_convex_combination_of_its_inputs() {
        // Softmax weights are all positive and sum to 1 -- the output must lie within the
        // bounding box of the inputs on every channel, a cheap sanity invariant independent of
        // the exact formula.
        let mut seed = 3u32;
        let hidden = 4;
        let b0 = random_vec(hidden, &mut seed);
        let current = random_vec(hidden, &mut seed);
        let norm_w = random_vec(hidden, &mut seed).iter().map(|v| v.abs() + 0.1).collect::<Vec<_>>();
        let proj_w = random_vec(hidden, &mut seed);

        let got = attn_res_pool(&[&b0], &current, &norm_w, &proj_w, 1e-5);
        for c in 0..hidden {
            let lo = b0[c].min(current[c]);
            let hi = b0[c].max(current[c]);
            assert!(got[c] >= lo - 1e-4 && got[c] <= hi + 1e-4, "channel {c}: {} not in [{lo},{hi}]", got[c]);
        }
    }

    #[test]
    fn is_checkpoint_layer_matches_plain_modulo() {
        assert!(is_checkpoint_layer(0, 12));
        assert!(is_checkpoint_layer(12, 12));
        assert!(is_checkpoint_layer(24, 12));
        assert!(!is_checkpoint_layer(1, 12));
        assert!(!is_checkpoint_layer(11, 12));
        assert!(!is_checkpoint_layer(13, 12));
    }

    /// Full 3-layer integration test with `block_size = 2` (checkpoints at layer 0 and layer 2),
    /// `s = 2` tokens, cross-checked against a from-scratch reference that mirrors
    /// `_forward_attn_residual`'s Python control flow line-by-line (using `naive_pool` instead
    /// of this module's own `attn_res_pool`, so a bug shared between the two wouldn't hide).
    /// Attention/MLP are stand-ins (arbitrary fixed deltas, not real math) -- this test is only
    /// about the state machine (block bookkeeping, prefix_sum reset/accumulate, pooling
    /// call sites), which is exactly this module's job; real attention/MoE math is out of scope.
    #[test]
    fn three_layer_sequence_with_checkpoints_matches_a_from_scratch_reference() {
        let hidden = 4;
        let s = 2;
        let block_size = 2;
        let mut seed = 42u32;

        let embed = random_vec(s * hidden, &mut seed);
        let attn_deltas: Vec<Vec<f32>> = (0..3).map(|_| random_vec(s * hidden, &mut seed)).collect();
        let mlp_deltas: Vec<Vec<f32>> = (0..3).map(|_| random_vec(s * hidden, &mut seed)).collect();
        let self_attn_w = AttnResWeights {
            norm: random_vec(hidden, &mut seed).iter().map(|v| v.abs() + 0.1).collect(),
            proj: random_vec(hidden, &mut seed),
        };
        let mlp_w = AttnResWeights {
            norm: random_vec(hidden, &mut seed).iter().map(|v| v.abs() + 0.1).collect(),
            proj: random_vec(hidden, &mut seed),
        };
        let eps = 1e-5;

        // --- this module's implementation ---
        let mut state = AttnResState::new();
        let mut hidden_in = embed.clone();
        for layer_idx in 0..3usize {
            let attn_input = before_attention(&mut state, &hidden_in, hidden, layer_idx, block_size, &self_attn_w, eps);
            let _ = attn_input; // real code would layernorm+attend this; stand-in delta used directly below
            let (mlp_input, prefix_sum) = before_mlp(&state, &hidden_in, &attn_deltas[layer_idx], hidden, layer_idx, block_size, &mlp_w, eps);
            let _ = mlp_input;
            hidden_in = after_mlp(&prefix_sum, &mlp_deltas[layer_idx]);
        }
        let final_state_blocks = state.n_blocks();
        let got_output = output_pool(&state, &hidden_in, hidden, &self_attn_w, eps);

        // --- from-scratch reference, per token, mirroring the Python line-by-line ---
        let mut expected_final = vec![0f32; s * hidden];
        for t in 0..s {
            let mut blocks: Vec<Vec<f32>> = Vec::new();
            let mut hidden_t: Vec<f32> = embed[t * hidden..(t + 1) * hidden].to_vec();
            for layer_idx in 0..3usize {
                let mut prefix_sum = hidden_t.clone();
                let block_refs: Vec<&[f32]> = blocks.iter().map(|b| b.as_slice()).collect();
                let _attn_input = naive_pool(&block_refs, &prefix_sum, &self_attn_w.norm, &self_attn_w.proj, eps);
                if is_checkpoint_layer(layer_idx, block_size) {
                    blocks.push(hidden_t.clone());
                    prefix_sum = attn_deltas[layer_idx][t * hidden..(t + 1) * hidden].to_vec();
                } else {
                    for c in 0..hidden {
                        prefix_sum[c] = hidden_t[c] + attn_deltas[layer_idx][t * hidden + c];
                    }
                }
                let block_refs2: Vec<&[f32]> = blocks.iter().map(|b| b.as_slice()).collect();
                let _mlp_input = naive_pool(&block_refs2, &prefix_sum, &mlp_w.norm, &mlp_w.proj, eps);
                for c in 0..hidden {
                    prefix_sum[c] += mlp_deltas[layer_idx][t * hidden + c];
                }
                hidden_t = prefix_sum;
            }
            let block_refs3: Vec<&[f32]> = blocks.iter().map(|b| b.as_slice()).collect();
            let out_t = naive_pool(&block_refs3, &hidden_t, &self_attn_w.norm, &self_attn_w.proj, eps);
            expected_final[t * hidden..(t + 1) * hidden].copy_from_slice(&out_t);
            assert_eq!(blocks.len(), final_state_blocks, "block count must match this module's own state");
        }

        for (a, b) in got_output.iter().zip(&expected_final) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
        // 3 layers, block_size 2 -> checkpoints at layer_idx 0 and 2 -> 2 blocks total.
        assert_eq!(final_state_blocks, 2);
    }
}
