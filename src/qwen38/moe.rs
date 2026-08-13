//! Qwen 3.8's MoE routing: 512 experts, top-10, plus one shared expert with its own scalar gate.
//! Ported from the real `Qwen3_5MoeTopKRouter` / `Qwen3_5MoeSparseMoeBlock`
//! (`modeling_qwen3_5_moe.py`, read this session).
//!
//! **A different router family from everything else in this crate.** GLM-5.2 and Kimi use
//! `noaux_tc` routing — a SIGMOID score per expert, a learned per-expert bias correction, expert
//! GROUPS ranked by their top-2 sum, and a `routed_scaling_factor` applied afterwards
//! (`glm52::moe::route`). Qwen 3.8 has none of that: plain softmax over all experts, no bias, no
//! groups, no scaling factor. The reference is four lines:
//!
//! ```text
//! router_logits = weight @ x                 # [num_experts], no bias
//! router_probs  = softmax(router_logits)     # over ALL 512 experts, in float32
//! top_value, top_idx = topk(router_probs, top_k)
//! top_value /= top_value.sum()               # renormalized over the selected k
//! ```
//!
//! **The renormalization is unconditional.** `config.norm_topk_prob` exists (and defaults to true —
//! see `qwen38::config`) but this router never reads it: it always divides by the top-k sum. So
//! `Cfg::norm_topk` is recorded config, not a branch, and [`route`] has no flag for it.
//!
//! **Top-k of a softmax is top-k of the logits, and renormalizing gives a softmax over just those.**
//! `p_i / Σ_{j∈topk} p_j = exp(l_i) / Σ_{j∈topk} exp(l_j)`, since softmax is monotone in the logits
//! and its denominator cancels. [`route`] therefore selects on logits and softmaxes the `k` winners —
//! identical output, without building a 512-wide probability vector per token per layer.
//! `route_matches_the_reference_order_of_operations` checks that equality against a literal
//! transcription of the four lines above.
//!
//! **Ties break toward the lower expert id**, matching `torch.topk`'s own documented behavior for
//! equal values, so a checkpoint with duplicated logits routes deterministically here too.

use crate::kimi_linear::ops::sigmoid;

/// One token's routing decision: `(expert_id, weight)` pairs, highest weight first, weights summing
/// to 1. Shaped like one row of `glm52::moe::Routing::choices` so the eventual expert-streaming
/// wiring can consume either family the same way.
pub type Choices = Vec<(usize, f32)>;

/// Selects the top-`topk` experts by router logit and returns them with softmax weights over just
/// those, descending by weight. `topk` is clamped to `logits.len()`.
pub fn route(logits: &[f32], topk: usize) -> Choices {
    let k = topk.min(logits.len());
    if k == 0 {
        return Vec::new();
    }

    // Partial selection by (logit desc, id asc). `k` is 10 against 512 experts, so a full sort of
    // (id, logit) pairs would do ~8x more comparisons than needed; this keeps a running top-k.
    let mut top: Vec<(usize, f32)> = Vec::with_capacity(k);
    for (id, &l) in logits.iter().enumerate() {
        // strictly-greater keeps the FIRST (lowest-id) of equal logits ahead of later ones
        let pos = top.iter().position(|&(_, tl)| l > tl);
        match pos {
            Some(p) if top.len() < k => top.insert(p, (id, l)),
            Some(p) => {
                top.pop();
                top.insert(p, (id, l));
            }
            None if top.len() < k => top.push((id, l)),
            None => {}
        }
    }

    // softmax over the k selected logits, max-subtracted for stability (the reference softmaxes in
    // float32 over all experts and then renormalizes; see this module's doc for why that's the same)
    let max = top[0].1;
    let mut sum = 0f32;
    for (_, l) in top.iter_mut() {
        *l = (*l - max).exp();
        sum += *l;
    }
    for (_, w) in top.iter_mut() {
        *w /= sum;
    }
    top
}

/// The shared expert's gate: `sigmoid(shared_expert_gate @ x)`, a SCALAR per token (the projection is
/// `Linear(hidden, 1)`), multiplying the shared expert's whole output before it's added to the routed
/// experts' sum. GLM's/Kimi's shared experts have no gate at all.
pub fn shared_expert_scale(gate_logit: f32) -> f32 {
    sigmoid(gate_logit)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference's four lines, transcribed literally: full softmax, then top-k, then divide by
    /// the selected sum.
    fn reference_route(logits: &[f32], topk: usize) -> Choices {
        let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
        let total: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / total).collect();

        let mut idx: Vec<usize> = (0..probs.len()).collect();
        // stable sort by prob desc, id asc — torch.topk's tie behavior
        idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap().then(a.cmp(&b)));
        idx.truncate(topk.min(probs.len()));
        let sum: f32 = idx.iter().map(|&i| probs[i]).sum();
        idx.into_iter().map(|i| (i, probs[i] / sum)).collect()
    }

    fn logits(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.max(1);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) % 2000) as f32 / 100.0 - 10.0
            })
            .collect()
    }

    #[test]
    fn route_matches_the_reference_order_of_operations() {
        for seed in 1..6u32 {
            let l = logits(512, seed);
            let got = route(&l, 10);
            let want = reference_route(&l, 10);
            assert_eq!(got.len(), 10);
            for (i, ((gi, gw), (wi, ww))) in got.iter().zip(&want).enumerate() {
                assert_eq!(gi, wi, "seed {seed}, rank {i}: expert id");
                assert!((gw - ww).abs() < 1e-6, "seed {seed}, rank {i}: weight {gw} vs {ww}");
            }
        }
    }

    #[test]
    fn weights_sum_to_one_and_come_back_descending() {
        let l = logits(64, 7);
        let choices = route(&l, 10);
        let sum: f32 = choices.iter().map(|&(_, w)| w).sum();
        assert!((sum - 1.0).abs() < 1e-5, "weights must renormalize to 1, got {sum}");
        for pair in choices.windows(2) {
            assert!(pair[0].1 >= pair[1].1, "weights must be descending: {choices:?}");
        }
    }

    #[test]
    fn selects_exactly_the_largest_logits() {
        // expert 3 biggest, then 7, then 1
        let l = vec![0.0, 2.0, -1.0, 5.0, 0.5, -3.0, 1.0, 3.0];
        let choices = route(&l, 3);
        assert_eq!(choices.iter().map(|&(id, _)| id).collect::<Vec<_>>(), vec![3, 7, 1]);
        assert!(choices[0].1 > choices[1].1 && choices[1].1 > choices[2].1);
    }

    /// Softmax is shift-invariant, so adding a constant to every logit must change neither the
    /// selection nor the weights — a cheap check that no absolute-magnitude assumption crept in.
    #[test]
    fn routing_is_invariant_to_a_constant_logit_shift() {
        let l = logits(128, 11);
        let shifted: Vec<f32> = l.iter().map(|x| x + 17.5).collect();
        let a = route(&l, 10);
        let b = route(&shifted, 10);
        for ((ai, aw), (bi, bw)) in a.iter().zip(&b) {
            assert_eq!(ai, bi);
            assert!((aw - bw).abs() < 1e-6, "{aw} vs {bw}");
        }
    }

    /// Equal logits must resolve toward the LOWER expert id, like `torch.topk`.
    #[test]
    fn ties_break_toward_the_lower_expert_id() {
        let l = vec![1.0, 1.0, 1.0, 1.0];
        assert_eq!(route(&l, 2).iter().map(|&(id, _)| id).collect::<Vec<_>>(), vec![0, 1]);

        let mixed = vec![0.0, 5.0, 5.0, 5.0];
        assert_eq!(route(&mixed, 2).iter().map(|&(id, _)| id).collect::<Vec<_>>(), vec![1, 2]);
    }

    #[test]
    fn topk_at_or_above_the_expert_count_selects_every_expert() {
        let l = vec![3.0, 1.0, 2.0];
        let all = route(&l, 10);
        assert_eq!(all.len(), 3);
        assert_eq!(all.iter().map(|&(id, _)| id).collect::<Vec<_>>(), vec![0, 2, 1]);
        let sum: f32 = all.iter().map(|&(_, w)| w).sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(route(&l, 0).is_empty());
    }

    #[test]
    fn shared_expert_gate_is_a_plain_sigmoid_of_one_logit() {
        assert!((shared_expert_scale(0.0) - 0.5).abs() < 1e-6);
        assert!(shared_expert_scale(10.0) > 0.999);
        assert!(shared_expert_scale(-10.0) < 0.001);
    }
}
