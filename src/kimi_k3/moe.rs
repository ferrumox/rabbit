//! Kimi K3's "Stable LatentMoE" — not a routing algorithm, despite the name: routed experts
//! operate in a narrower space than the model's real hidden size. Confirmed against the real
//! `KimiSparseMoeBlock.forward`/`__init__` in `modeling_kimi_linear.py` (fetched 2026-07-27, not
//! guessed): when `config.routed_expert_hidden_size` is set (K3's real checkpoint: `3584`, half
//! of `hidden_size`'s `7168`), hidden states are down-projected to that width BEFORE being
//! dispatched to experts (each expert instantiated with `hidden_size=moe_hidden_size`, so its
//! on-disk `gate_proj`/`up_proj`/`down_proj` tensors are genuinely shaped for the narrower width,
//! not the model's real hidden size), optionally RMSNorm'd, then up-projected back. The router
//! gate itself scores using the ORIGINAL (pre-down-proj) hidden states — routing and expert
//! compute happen at two different widths, which is why this can't just reuse
//! `glm52::moe::moe()` unchanged (that function assumes a single width serves both).
//!
//! Deliberately does NOT touch `glm52::moe::moe()` (GLM-5.2's and Kimi Linear 48B's shared,
//! perf-tuned dispatch path — `PERFORMANCE.md` chronicles real measured work on it, including the
//! begin_loading/finish_loading_streaming disk-read/shared-expert-compute overlap): this is a
//! separate function reusing that module's already-tested `unique_experts`/`apply_single_expert`
//! (widened to `pub(crate)` for exactly this) rather than risk changing that hot path's behavior.
//! Correctness first, matching this project's own established discipline (see e.g. Phase 2's
//! "scalar kernel first, SIMD tiers after correctness is proven") — this dispatch loop uses the
//! simpler synchronous `ExpertCache::ensure_loaded`/`get`, not the io_uring streaming overlap
//! trick `moe()` uses; that's a real, deliberately deferred optimization opportunity once this is
//! proven correct against the real checkpoint, not an oversight.
//!
//! `shared_experts(identity)` (computed at the model's real hidden size, on the pre-down-proj
//! hidden states) is each caller's own job, same as `modeling_kimi_linear.py`'s
//! `KimiSparseMoeBlock.forward` adds it AFTER this wrapper's up-projected output — not folded in
//! here, since it needs no down/up-projection at all and `glm52::moe.rs`'s existing shared-expert
//! matmul code is exactly what would be duplicated to inline it.

use crate::expert_cache::ExpertCache;
use crate::glm52::config::Cfg;
use crate::glm52::model::{ModelError, MoeWeights};
use crate::glm52::moe::{self, RouteConfig};
use crate::kernels::matmul_qt;
use crate::kimi_linear::ops::rmsnorm;
use crate::quant::QT;
use crate::safetensors::Shards;

/// The three extra tensors K3's latent-MoE wrapper needs, on top of the ordinary `MoeWeights`
/// (router + shared experts, unchanged) and whatever `ExpertCache` streams in per routed expert.
pub struct LatentMoeWeights {
    /// `[moe_hidden, hidden]` — projects the model's real hidden size down before routing.
    pub down_proj: QT,
    /// `[hidden, moe_hidden]` — projects the routed-expert mix back up afterward.
    pub up_proj: QT,
    /// `Some(weight)` (`moe_hidden`-wide) iff `latent_moe_use_norm` — RMSNorm applied to the
    /// summed routed-expert output BEFORE `up_proj`, matching `KimiSparseMoeBlock.forward`'s
    /// `if self.latent_moe_use_norm: y = self.routed_expert_norm(y)` ordering exactly (this is
    /// why the norm can't be folded per-expert into `apply_single_expert`'s accumulation even
    /// though `up_proj` mathematically could be — RMSNorm isn't linear, it must see the full
    /// weighted sum across all of a token's chosen experts, not each one individually).
    pub norm: Option<Vec<f32>>,
}

/// Runs K3's latent-MoE wrapper for one layer: routes using `x` (the model's real hidden size,
/// via `cfg_full`), down-projects, dispatches every routed expert through `cache` (tensors shaped
/// per `cfg_expert`, whose `.hidden` must be `lw`'s `moe_hidden` width — the caller is
/// responsible for that split, typically two `Cfg`s differing only in that one field), optionally
/// RMSNorms the summed result, then up-projects into `out`. `out` is OVERWRITTEN (matches
/// `matmul_qt`'s own convention, and mirrors `glm52::moe::moe()`'s "caller adds the shared
/// expert's contribution separately" shape) — it does NOT include the shared-expert term, see
/// this module's doc.
#[allow(clippy::too_many_arguments)]
pub fn latent_moe(
    cfg_full: &Cfg,
    cfg_expert: &Cfg,
    w: &MoeWeights,
    lw: &LatentMoeWeights,
    cache: &mut ExpertCache,
    shards: &Shards,
    layer: usize,
    ebits: u8,
    route_cfg: &RouteConfig,
    x: &[f32],
    s: usize,
    eps: f32,
    activation: moe::Activation,
    out: &mut [f32],
) -> Result<(), ModelError> {
    let moe_hidden = cfg_expert.hidden as usize;
    let inter = cfg_expert.moe_inter as usize;

    // Routing scores the ORIGINAL (full-width) hidden states -- confirmed against the real
    // `KimiSparseMoeBlock.forward`, which calls `self.gate(hidden_states)` before the down-proj.
    let routing = moe::route_cache_aware(cfg_full, w, x, s, cache, route_cfg);
    for choices in &routing.choices {
        for &(eid, _) in choices {
            cache.record_selection(eid);
        }
    }

    let mut x_latent = vec![0f32; s * moe_hidden];
    matmul_qt(&mut x_latent, x, &lw.down_proj, s);

    let mut routed = vec![0f32; s * moe_hidden];
    let uniq = moe::unique_experts(&routing);
    let chunk_size = cache.capacity().max(1);
    for chunk in uniq.chunks(chunk_size) {
        cache.ensure_loaded(shards, cfg_expert, layer, chunk, ebits)?;
        for &eid in chunk {
            if let Some(slot) = cache.get(eid) {
                moe::apply_single_expert(slot, &routing, &x_latent, moe_hidden, inter, activation, &mut routed);
            }
        }
    }

    if let Some(norm_w) = &lw.norm {
        for row in routed.chunks_mut(moe_hidden) {
            rmsnorm(row, norm_w, eps);
        }
    }

    matmul_qt(out, &routed, &lw.up_proj, s);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glm52::moe::Routing;
    use serde_json::json;
    use std::fs;

    struct TempDir(std::path::PathBuf);
    impl TempDir {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(name);
            fs::create_dir_all(&dir).unwrap();
            TempDir(dir)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).ok();
        }
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

    fn random_qt_f32(rows: usize, cols: usize, seed: &mut u32) -> QT {
        let mut t = QT::alloc(rows, cols, 32, false);
        t.fill(&random_vec(rows * cols, seed));
        t
    }

    fn f32_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// Same fixture shape as `glm52::moe`'s own `build_expert_fixture` test helper, but each
    /// expert's tensors are sized `moe_hidden` wide (not the model's real `hidden`) — matching a
    /// real K3 checkpoint, where the routed-expert tensors genuinely are narrower than the rest
    /// of the model.
    fn build_expert_fixture(dir: &std::path::Path, n_experts: usize, moe_inter: usize, moe_hidden: usize, seed: &mut u32) -> Shards {
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_string(), json!({"format": "rabbit-test"}));
        let mut data = Vec::new();
        for eid in 0..n_experts {
            let gate = random_vec(moe_inter * moe_hidden, seed);
            let up = random_vec(moe_inter * moe_hidden, seed);
            let down = random_vec(moe_hidden * moe_inter, seed);
            for (suf, rows, cols, vals) in [
                ("gate_proj", moe_inter, moe_hidden, &gate),
                ("up_proj", moe_inter, moe_hidden, &up),
                ("down_proj", moe_hidden, moe_inter, &down),
            ] {
                let name = format!("model.layers.0.mlp.experts.{eid}.{suf}.weight");
                let bytes = f32_bytes(vals);
                let start = data.len() as u64;
                data.extend_from_slice(&bytes);
                let end = data.len() as u64;
                header.insert(name, json!({"dtype": "F32", "shape": [rows, cols], "data_offsets": [start, end]}));
            }
        }
        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out_bytes = Vec::new();
        out_bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out_bytes.extend_from_slice(&header_bytes);
        out_bytes.extend_from_slice(&data);
        fs::write(dir.join("model.safetensors"), out_bytes).unwrap();
        Shards::open(dir).unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    fn tiny_cfg(n_experts: i32, topk: i32, moe_inter: i32, hidden: i32, n_shared: i32) -> Cfg {
        Cfg {
            hidden,
            n_layers: 1,
            n_heads: 1,
            n_experts,
            topk,
            moe_inter,
            dense_inter: 1,
            first_dense: 0,
            q_lora: 1,
            kv_lora: 1,
            qk_nope: 1,
            qk_rope: 2,
            qk_head: 3,
            v_head: 1,
            n_shared,
            vocab: 1,
            n_group: 1,
            topk_group: 1,
            norm_topk: false,
            stop_ids: vec![],
            index_topk: 0,
            index_nh: 0,
            index_hd: 0,
            idx_type: vec![false],
            eps: 1e-5,
            theta: 10000.0,
            attn_scale: 1.0,
            routed_scale: 1.0,
            group_size: 0,
        }
    }

    /// End-to-end: down-proj -> dispatch -> up-proj, no norm, cross-checked against a fully
    /// independent naive reference (plain matmuls + the same weighted-sum-over-chosen-experts
    /// logic `apply_single_expert` implements, computed by hand here instead of reused).
    #[test]
    fn latent_moe_matches_a_naive_reference_without_norm() {
        let (hidden, moe_hidden, inter, n_experts, topk, n_shared) = (6usize, 4usize, 3usize, 4usize, 2i32, 1i32);
        let mut seed = 11u32;
        let dir = TempDir::new("rabbit_test_k3_latent_moe_no_norm");
        let shards = build_expert_fixture(&dir.0, n_experts, inter, moe_hidden, &mut seed);

        let cfg_full = tiny_cfg(n_experts as i32, topk, inter as i32, hidden as i32, n_shared);
        let cfg_expert = tiny_cfg(n_experts as i32, topk, inter as i32, moe_hidden as i32, n_shared);
        let w = MoeWeights {
            router: random_vec(n_experts * hidden, &mut seed),
            router_bias: random_vec(n_experts, &mut seed),
            sh_gate: random_qt_f32(inter, hidden, &mut seed),
            sh_up: random_qt_f32(inter, hidden, &mut seed),
            sh_down: random_qt_f32(hidden, inter, &mut seed),
        };
        let down_proj = random_qt_f32(moe_hidden, hidden, &mut seed);
        let up_proj = random_qt_f32(hidden, moe_hidden, &mut seed);
        let lw = LatentMoeWeights { down_proj, up_proj, norm: None };

        let s = 3;
        let x = random_vec(s * hidden, &mut seed);

        let mut cache = ExpertCache::new(n_experts);
        let mut out = vec![0f32; s * hidden];
        latent_moe(&cfg_full, &cfg_expert, &w, &lw, &mut cache, &shards, 0, 32, &RouteConfig::default(), &x, s, 1e-5, moe::Activation::Silu, &mut out).unwrap();

        // Independent reference: re-derive routing with the plain (non-cache-aware) router,
        // which `RouteConfig::default()` (cache_route: false) must agree with bit-for-bit.
        let routing: Routing = moe::route(&cfg_full, &w, &x, s);
        let mut x_latent_ref = vec![0f32; s * moe_hidden];
        matmul_qt(&mut x_latent_ref, &x, &lw.down_proj, s);
        let mut routed_ref = vec![0f32; s * moe_hidden];
        for eid in 0..n_experts {
            let mut cache2 = ExpertCache::new(n_experts);
            cache2.ensure_loaded(&shards, &cfg_expert, 0, &[eid], 32).unwrap();
            let slot = cache2.get(eid).unwrap();
            moe::apply_single_expert(slot, &routing, &x_latent_ref, moe_hidden, inter, moe::Activation::Silu, &mut routed_ref);
        }
        let mut expected = vec![0f32; s * hidden];
        matmul_qt(&mut expected, &routed_ref, &lw.up_proj, s);

        for (a, b) in out.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    /// Same setup as the no-norm test, but with `norm: Some(..)` — proves the norm is genuinely
    /// applied to the SUMMED per-token routed output (post weighted-sum-over-experts) before
    /// `up_proj`, rather than, say, silently skipped or applied per-expert (RMSNorm isn't linear,
    /// so those would give a different, wrong answer).
    #[test]
    fn latent_moe_applies_norm_to_the_summed_output_before_up_proj() {
        let (hidden, moe_hidden, inter, n_experts, topk, n_shared) = (6usize, 4usize, 3usize, 4usize, 2i32, 1i32);
        let mut seed = 23u32;
        let dir = TempDir::new("rabbit_test_k3_latent_moe_with_norm");
        let shards = build_expert_fixture(&dir.0, n_experts, inter, moe_hidden, &mut seed);

        let cfg_full = tiny_cfg(n_experts as i32, topk, inter as i32, hidden as i32, n_shared);
        let cfg_expert = tiny_cfg(n_experts as i32, topk, inter as i32, moe_hidden as i32, n_shared);
        let w = MoeWeights {
            router: random_vec(n_experts * hidden, &mut seed),
            router_bias: random_vec(n_experts, &mut seed),
            sh_gate: random_qt_f32(inter, hidden, &mut seed),
            sh_up: random_qt_f32(inter, hidden, &mut seed),
            sh_down: random_qt_f32(hidden, inter, &mut seed),
        };
        let down_proj = random_qt_f32(moe_hidden, hidden, &mut seed);
        let up_proj = random_qt_f32(hidden, moe_hidden, &mut seed);
        let norm_w: Vec<f32> = random_vec(moe_hidden, &mut seed).iter().map(|v| v.abs() + 0.5).collect();
        let eps = 1e-5;

        let s = 2;
        let x = random_vec(s * hidden, &mut seed);

        let lw = LatentMoeWeights { down_proj, up_proj, norm: Some(norm_w.clone()) };
        let mut cache = ExpertCache::new(n_experts);
        let mut out = vec![0f32; s * hidden];
        latent_moe(&cfg_full, &cfg_expert, &w, &lw, &mut cache, &shards, 0, 32, &RouteConfig::default(), &x, s, eps, moe::Activation::Silu, &mut out).unwrap();

        // Independent reference: routing + per-expert dispatch exactly as in the no-norm test,
        // but explicitly apply (then verify) the norm step before up-projecting.
        let routing: Routing = moe::route(&cfg_full, &w, &x, s);
        let mut x_latent_ref = vec![0f32; s * moe_hidden];
        matmul_qt(&mut x_latent_ref, &x, &lw.down_proj, s);
        let mut routed_ref = vec![0f32; s * moe_hidden];
        for eid in 0..n_experts {
            let mut c = ExpertCache::new(n_experts);
            c.ensure_loaded(&shards, &cfg_expert, 0, &[eid], 32).unwrap();
            let slot = c.get(eid).unwrap();
            moe::apply_single_expert(slot, &routing, &x_latent_ref, moe_hidden, inter, moe::Activation::Silu, &mut routed_ref);
        }
        let unnormed = routed_ref.clone();
        for row in routed_ref.chunks_mut(moe_hidden) {
            rmsnorm(row, &norm_w, eps);
        }
        assert_ne!(routed_ref, unnormed, "rmsnorm must actually change the summed routed output for this to be a real test");

        let mut expected = vec![0f32; s * hidden];
        matmul_qt(&mut expected, &routed_ref, &lw.up_proj, s);
        for (a, b) in out.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }
}
