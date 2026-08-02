//! Kimi Linear's own `layer_forward`/`layers_forward`/`step`, the sibling of
//! `crate::generate`'s GLM-5.2 versions — not a variant of them, since the per-layer state
//! shape genuinely differs (a fixed-size KDA recurrent state + ShortConv FIFOs for KDA layers,
//! vs. GLM's growing `KvCache` for MLA layers), matching `rabbit-plan.md`'s Phase 1 decision to
//! defer a shared `KvState` design until a second real architecture existed to design it
//! against.
//!
//! FFN dispatch (`glm52::moe::{dense_mlp, moe}`) and MLA attention
//! (`glm52::attention::attention()`, via `model::mla_call_args`) are reused completely
//! unchanged — only KDA's per-token math (`kda.rs`/`short_conv.rs`/`ops.rs`, all already
//! unit-tested independently) is new here, just wired together per-layer.
//!
//! Routed-expert loading on a cache miss (`glm52::moe::moe()`'s call into
//! `expert_cache.rs::ExpertCache`) now also works for Kimi: `ExpertCaches::new` below builds
//! every layer's `ExpertCache` via `ExpertCache::for_family(capacity, ExpertNaming::KimiLinear)`,
//! which reads Kimi's real on-disk names (`block_sparse_moe.experts.{eid}.{w1,w2,w3}.weight` —
//! see `kimi_linear::model`'s doc) instead of GLM's `mlp.experts.{eid}.{gate_proj,up_proj,
//! down_proj}.weight`. This module's own tests still use dense-FFN-only fixtures (no reason to
//! duplicate `expert_cache.rs`'s own from-a-real-fixture MoE coverage here).

use crate::expert_cache::{ExpertCache, ExpertNaming};
use crate::generate::{Phases, StepProfile};
use crate::glm52::attention::{self, rmsnorm, OutputGate};
use crate::glm52::model::{Ffn, ModelError};
use crate::glm52::moe;
use crate::kernels::matmul_qt;
use crate::kimi_linear::config::Cfg;
use crate::kimi_linear::kda::{decay_gate, KdaState};
use crate::kimi_linear::model::{self, Attn, KdaWeights, Model};
use crate::kimi_linear::ops::{head_output_gate, l2_norm, sigmoid, swish};
use crate::kimi_linear::short_conv::ShortConvState;
use crate::safetensors::Shards;
use rayon::prelude::*;

/// One KDA layer's recurrent state: `kda_n_heads` independent `KdaState` matrices (KDA's
/// recurrence runs per head, see `kda.rs`) plus 3 `ShortConvState` FIFOs (q/k/v) — each
/// `d_inner`-wide (ALL heads concatenated), since the causal conv runs over the whole projected
/// channel space before the per-head split, exactly matching `modeling_kimi.py`'s
/// `KimiDeltaAttention.forward` (`q_conv1d`/`k_conv1d`/`v_conv1d` operate on the full
/// `projection_k_size`/`projection_size`-wide tensors, reshaped into heads only afterward).
pub struct KdaLayerState {
    heads: Vec<KdaState>,
    q_conv: ShortConvState,
    k_conv: ShortConvState,
    v_conv: ShortConvState,
}

impl KdaLayerState {
    pub fn new(cfg: &Cfg) -> KdaLayerState {
        let head_dim = cfg.kda_head_dim as usize;
        let n_heads = cfg.kda_n_heads as usize;
        let d_inner = head_dim * n_heads;
        let kernel = cfg.short_conv_kernel as usize;
        KdaLayerState {
            heads: (0..n_heads).map(|_| KdaState::new(head_dim, head_dim)).collect(),
            q_conv: ShortConvState::new(d_inner, kernel),
            k_conv: ShortConvState::new(d_inner, kernel),
            v_conv: ShortConvState::new(d_inner, kernel),
        }
    }

    pub(crate) fn heads(&self) -> &[KdaState] {
        &self.heads
    }

    pub(crate) fn q_conv(&self) -> &ShortConvState {
        &self.q_conv
    }

    pub(crate) fn k_conv(&self) -> &ShortConvState {
        &self.k_conv
    }

    pub(crate) fn v_conv(&self) -> &ShortConvState {
        &self.v_conv
    }

    /// Reconstructs a `KdaLayerState` from previously-saved raw heads/conv FIFOs —
    /// `kv_session.rs`'s load path, the mirror of `new` (which starts fresh) for restoring one
    /// with real history.
    pub(crate) fn from_raw(heads: Vec<KdaState>, q_conv: ShortConvState, k_conv: ShortConvState, v_conv: ShortConvState) -> KdaLayerState {
        KdaLayerState { heads, q_conv, k_conv, v_conv }
    }
}

/// One layer's recurrent/cache state: `Kda` (new, this module) or `Mla` (GLM-5.2's `KvCache`,
/// reused as-is). Always paired with the matching `kimi_linear::model::Attn` variant for the
/// same layer — `KvState::new` is the only constructor, and it derives this from the model.
pub enum LayerState {
    Kda(KdaLayerState),
    Mla(attention::KvCache),
}

/// Per-layer state for one generation session, the sibling of `crate::generate::KvState`.
pub struct KvState {
    layers: Vec<LayerState>,
}

impl KvState {
    pub fn new(model: &Model) -> KvState {
        let layers = model
            .layers
            .iter()
            .map(|l| match &l.attn {
                Attn::Kda(_) => LayerState::Kda(KdaLayerState::new(&model.cfg)),
                Attn::Mla(_) => LayerState::Mla(attention::KvCache::new(model.cfg.kv_lora as usize, model.cfg.qk_rope as usize)),
            })
            .collect();
        KvState { layers }
    }

    pub(crate) fn layers(&self) -> &[LayerState] {
        &self.layers
    }

    /// Reconstructs a `KvState` from raw saved per-layer states — `kv_session.rs`'s load path.
    pub(crate) fn from_raw(layers: Vec<LayerState>) -> KvState {
        KvState { layers }
    }
}

/// One `ExpertCache` per MoE layer — the sibling of `crate::generate::ExpertCaches`, kept as
/// its own small type rather than generalizing the GLM one (their only truly shared logic is the
/// constructor pattern; `crate::generate::ExpertCaches` already has several GLM-5.2 call sites
/// depending on its exact API). `warm_start`/`save_usage`/`hit_miss_totals`/`io_wait_nanos_total`
/// mirror `crate::generate::ExpertCaches`'s own implementations line for line — both operate
/// purely through `expert_cache::ExpertCache`'s already-architecture-agnostic API (keyed by
/// plain `(layer, expert_id)` pairs) and the top-level, also-architecture-agnostic
/// `usage_cache` module, so there's no Kimi-specific logic here at all, just the same wiring
/// GLM's version has, needed once `chat.rs`'s `Session`/`--chat`/`--serve` route through this
/// family too.
pub struct ExpertCaches(Vec<Option<ExpertCache>>);

impl ExpertCaches {
    pub fn new(model: &Model, capacity: usize) -> ExpertCaches {
        let v = model
            .layers
            .iter()
            .map(|l| if matches!(l.ffn, Ffn::Moe(_)) { Some(ExpertCache::for_family(capacity, model.cfg.n_experts as usize, ExpertNaming::KimiLinear)) } else { None })
            .collect();
        ExpertCaches(v)
    }

    /// Summed `hits`/`misses`/`load_nanos` across every MoE layer's cache — see
    /// `crate::generate::ExpertCaches::hit_miss_totals`'s doc.
    pub fn hit_miss_totals(&self) -> (u64, u64, u64) {
        self.0.iter().flatten().fold((0, 0, 0), |(h, m, n), c| (h + c.hits, m + c.misses, n + c.load_nanos))
    }

    /// Summed pure `io_uring` disk-wait time — see
    /// `crate::generate::ExpertCaches::io_wait_nanos_total`'s doc.
    pub fn io_wait_nanos_total(&self) -> u64 {
        self.0.iter().flatten().map(|c| c.io_wait_nanos).sum()
    }

    /// Whether any layer's cache loads through an `io_uring` ring. `false` (every MXFP4/K3 run,
    /// whose naming never gets a ring) means `io_wait_nanos_total` is structurally zero, so the
    /// CLI must not present it as "actual disk wait" (Phase 4c) — see `ExpertCache::has_ring`.
    pub fn any_has_ring(&self) -> bool {
        self.0.iter().flatten().any(|c| c.has_ring())
    }

    /// Phase 4b: preload every MoE layer's routed experts up front (see
    /// `expert_cache::preload_layers`). `to_glm_cfg` supplies the expert shapes, same as the
    /// dispatch path's `moe::moe` call.
    pub fn preload(&mut self, model: &Model, shards: &Shards) -> Result<(), ModelError> {
        let cfg = crate::kimi_linear::model::to_glm_cfg(&model.cfg);
        let n_experts = cfg.n_experts as usize;
        Ok(crate::expert_cache::preload_layers(&mut self.0, shards, &cfg, model.ebits, n_experts)?)
    }

    /// Seeds usage counters from `<model_dir>/.rabbit_usage` and marks pin candidates once
    /// confidence crosses the threshold — see `crate::generate::ExpertCaches::warm_start`'s doc
    /// (identical logic, ported verbatim).
    pub fn warm_start(&mut self, model_dir: &std::path::Path, cache_capacity: usize) -> crate::generate::WarmStartStats {
        let loaded = crate::usage_cache::load(&crate::usage_cache::usage_path(model_dir));
        let hist: u64 = loaded.values().sum();

        let mut per_layer: std::collections::HashMap<usize, Vec<(usize, u64)>> = std::collections::HashMap::new();
        for (&(li, eid), &count) in &loaded {
            per_layer.entry(li).or_default().push((eid, count));
        }
        for (li, cache_opt) in self.0.iter_mut().enumerate() {
            if let (Some(cache), Some(entries)) = (cache_opt, per_layer.remove(&li)) {
                cache.seed_usage(entries.into_iter());
            }
        }

        if hist < crate::usage_cache::HIST_THRESHOLD {
            return crate::generate::WarmStartStats { hist, confidence: crate::usage_cache::confidence(hist), pin_candidates: 0 };
        }

        let confidence = crate::usage_cache::confidence(hist);
        let budget = crate::usage_cache::pin_budget(cache_capacity, confidence);
        let mut pin_candidates = 0;
        for cache_opt in self.0.iter_mut() {
            let Some(cache) = cache_opt else { continue };
            let mut top: Vec<(usize, u64)> = cache.usage_counts().collect();
            top.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            top.truncate(budget);
            pin_candidates += top.len();
            cache.mark_pin_candidates(top.into_iter().map(|(eid, _)| eid));
        }
        crate::generate::WarmStartStats { hist, confidence, pin_candidates }
    }

    /// Writes every MoE layer's current usage counters back to `<model_dir>/.rabbit_usage` — see
    /// `crate::generate::ExpertCaches::save_usage`'s doc.
    pub fn save_usage(&self, model_dir: &std::path::Path) -> std::io::Result<()> {
        let path = crate::usage_cache::usage_path(model_dir);
        let entries = self.0.iter().enumerate().flat_map(|(li, c)| {
            c.as_ref().into_iter().flat_map(move |cache| cache.usage_counts().map(move |(eid, count)| (li, eid, count)))
        });
        crate::usage_cache::save(&path, entries)
    }
}

fn embed_tokens(model: &Model, ids: &[usize]) -> Vec<f32> {
    let d = model.cfg.hidden as usize;
    let mut x = vec![0f32; ids.len() * d];
    for (si, &tok) in ids.iter().enumerate() {
        x[si * d..(si + 1) * d].copy_from_slice(&model.embed_row(tok));
    }
    x
}

/// One KDA layer's forward for a single token: `q,k,v = proj(x)`, `ShortConv`, `Swish`,
/// `L2Norm` (q,k only), the decay gate, the recurrence (per head), the output gate, `o_proj`.
/// Mirrors `modeling_kimi.py`'s `KimiDeltaAttention.forward` step for step — every op here was
/// independently unit-tested in `kda.rs`/`short_conv.rs`/`ops.rs` before this function wired
/// them together.
fn kda_step(cfg: &Cfg, w: &KdaWeights, state: &mut KdaLayerState, x: &[f32], out: &mut [f32]) {
    let head_dim = cfg.kda_head_dim as usize;
    let n_heads = cfg.kda_n_heads as usize;
    let d_inner = head_dim * n_heads;
    let eps = cfg.eps;

    let mut q_pre = vec![0f32; d_inner];
    matmul_qt(&mut q_pre, x, &w.q_proj, 1);
    let mut k_pre = vec![0f32; d_inner];
    matmul_qt(&mut k_pre, x, &w.k_proj, 1);
    let mut v_pre = vec![0f32; d_inner];
    matmul_qt(&mut v_pre, x, &w.v_proj, 1);

    let mut q = vec![0f32; d_inner];
    state.q_conv.step(&q_pre, &w.q_conv, &mut q);
    let mut k = vec![0f32; d_inner];
    state.k_conv.step(&k_pre, &w.k_conv, &mut k);
    let mut v = vec![0f32; d_inner];
    state.v_conv.step(&v_pre, &w.v_conv, &mut v);

    for c in q.iter_mut() {
        *c = swish(*c);
    }
    for c in k.iter_mut() {
        *c = swish(*c);
    }
    for c in v.iter_mut() {
        *c = swish(*c);
    }
    // The real reference kernel (fla.ops.kda.{chunk,fused_recurrent}, via naive_recurrent_kda's
    // own default `scale = K ** -0.5`) scales q by 1/sqrt(head_dim) AFTER L2Norm -- not part of
    // Eq. 1's core recurrence (kda.rs::KdaState::step takes q/k/v/alpha/beta as-is, no implicit
    // scale), a real-checkpoint wiring detail confirmed by cross-checking rabbit's own Rust
    // output against a CPU-patched real modeling_kimi.py forward pass this session (caught a
    // ~30% argmax mismatch rate on the oracle before this fix). Scaling BEFORE L2Norm would be
    // pointless (L2Norm would just re-normalize it away), so the order matters.
    let q_scale = 1.0 / (head_dim as f32).sqrt();
    for h in 0..n_heads {
        let sl = h * head_dim..(h + 1) * head_dim;
        l2_norm(&mut q[sl.clone()], eps);
        l2_norm(&mut k[sl.clone()], eps);
        for c in &mut q[sl] {
            *c *= q_scale;
        }
    }

    let mut f_a = vec![0f32; head_dim];
    matmul_qt(&mut f_a, x, &w.f_a_proj, 1);
    let mut g = vec![0f32; d_inner];
    matmul_qt(&mut g, &f_a, &w.f_b_proj, 1);

    let mut beta_pre = vec![0f32; n_heads];
    matmul_qt(&mut beta_pre, x, &w.b_proj, 1);

    let mut g_a = vec![0f32; head_dim];
    matmul_qt(&mut g_a, x, &w.g_a_proj, 1);
    let mut g2 = vec![0f32; d_inner];
    matmul_qt(&mut g2, &g_a, &w.g_b_proj, 1);

    // Heads are independent: each reads shared q/k/v/g/g2/beta_pre and writes only its own
    // disjoint head_dim-wide slice of `o` -- no cross-head reduction anywhere, so parallelizing
    // this axis changes execution ORDER but not any floating-point reassociation within a
    // head's own math (bit-identical to the sequential version). Same reasoning and pattern as
    // `glm52::attention::attention()`'s absorbed-decode path's `ctx_row.par_chunks_mut(vh)`.
    let mut o = vec![0f32; d_inner];
    state.heads.par_iter_mut().zip(o.par_chunks_mut(head_dim)).enumerate().for_each(|(h, (head_state, o_slot))| {
        let sl = h * head_dim..(h + 1) * head_dim;
        let mut alpha = vec![0f32; head_dim];
        decay_gate(w.a_log[h], &g[sl.clone()], &w.dt_bias[sl.clone()], None, &mut alpha);
        let beta = sigmoid(beta_pre[h]);
        head_state.step(&q[sl.clone()], &k[sl.clone()], &v[sl.clone()], &alpha, beta, o_slot);
        head_output_gate(o_slot, &w.o_norm, &g2[sl], eps);
    });

    matmul_qt(out, &o, &w.o_proj, 1);
}

/// One transformer layer: `x += attention(rmsnorm(x, in_ln))`, then `x += ffn(rmsnorm(x,
/// post_ln))` — same residual structure as `crate::generate::layer_forward`, dispatched over
/// `Attn`/`LayerState` (KDA runs one token at a time regardless of `s`, since it's a genuine
/// recurrence; MLA reuses `attention()`'s own batched-token handling).
#[allow(clippy::too_many_arguments)]
fn layer_forward(
    cfg: &Cfg,
    model: &Model,
    li: usize,
    shards: &Shards,
    caches: &mut ExpertCaches,
    x: &mut [f32],
    s: usize,
    pos_base: usize,
    layer_state: &mut LayerState,
    mut phases: Option<&mut Phases>,
) -> Result<(), ModelError> {
    let d = cfg.hidden as usize;
    let layer = &model.layers[li];

    let mut nrm = vec![0f32; s * d];
    for si in 0..s {
        nrm[si * d..(si + 1) * d].copy_from_slice(&x[si * d..(si + 1) * d]);
        rmsnorm(&mut nrm[si * d..(si + 1) * d], &layer.in_ln, cfg.eps);
    }

    let mut tmp = vec![0f32; s * d];
    let attn_t = std::time::Instant::now();
    match (&layer.attn, layer_state) {
        (Attn::Kda(w), LayerState::Kda(state)) => {
            for si in 0..s {
                kda_step(cfg, w, state, &nrm[si * d..(si + 1) * d], &mut tmp[si * d..(si + 1) * d]);
            }
        }
        (Attn::Mla(w), LayerState::Mla(kv)) => {
            let (glm_cfg, dsa, absorb, rope, qproj) = model::mla_call_args(cfg);
            attention::attention(&glm_cfg, w, kv, &nrm, s, pos_base, dsa, absorb, rope, qproj, OutputGate::Off, &mut tmp);
        }
        _ => unreachable!("Attn/LayerState variant mismatch -- KvState::new always pairs them per layer"),
    }
    if let Some(p) = phases.as_deref_mut() {
        p.attention_s += attn_t.elapsed().as_secs_f32();
    }
    for (xi, &ti) in x.iter_mut().zip(&tmp) {
        *xi += ti;
    }

    for si in 0..s {
        nrm[si * d..(si + 1) * d].copy_from_slice(&x[si * d..(si + 1) * d]);
        rmsnorm(&mut nrm[si * d..(si + 1) * d], &layer.post_ln, cfg.eps);
    }
    let glm_cfg = model::to_glm_cfg(cfg);
    match &layer.ffn {
        Ffn::Dense(w) => {
            let t = std::time::Instant::now();
            moe::dense_mlp(w, &nrm, s, cfg.dense_inter as usize, moe::Activation::Silu, &mut tmp);
            if let Some(p) = phases.as_deref_mut() {
                p.expert_matmul_s += t.elapsed().as_secs_f32();
            }
        }
        Ffn::Moe(w) => {
            let cache = caches.0[li].as_mut().expect("MoE layer must have an ExpertCache");
            let wait_before = cache.io_wait_nanos;
            let t = std::time::Instant::now();
            moe::moe(&glm_cfg, w, cache, shards, li, model.ebits, &model.route_cfg, &nrm, s, moe::Activation::Silu, &mut tmp)?;
            let elapsed = t.elapsed().as_secs_f32();
            if let Some(p) = phases {
                let wait_delta = ((cache.io_wait_nanos - wait_before) as f32 / 1e9).max(0.0);
                p.expert_wait_s += wait_delta;
                p.expert_matmul_s += (elapsed - wait_delta).max(0.0);
            }
        }
    }
    for (xi, &ti) in x.iter_mut().zip(&tmp) {
        *xi += ti;
    }

    Ok(())
}

/// Runs every layer in order on new tokens `x[S,hidden]`, updating `x` in place.
pub fn layers_forward(model: &Model, shards: &Shards, caches: &mut ExpertCaches, x: &mut [f32], s: usize, pos_base: usize, kv: &mut KvState) -> Result<(), ModelError> {
    layers_forward_profiled(model, shards, caches, x, s, pos_base, kv, None)
}

/// Like `layers_forward`, but accumulates per-layer phase timing into `phases` when given —
/// mirrors `crate::generate::layers_forward`'s own `Option<&mut Phases>` threading, reusing the
/// SAME `Phases`/`StepProfile` types (plain data, no GLM-specific coupling).
#[allow(clippy::too_many_arguments)]
fn layers_forward_profiled(model: &Model, shards: &Shards, caches: &mut ExpertCaches, x: &mut [f32], s: usize, pos_base: usize, kv: &mut KvState, mut phases: Option<&mut Phases>) -> Result<(), ModelError> {
    let cfg = &model.cfg;
    for li in 0..model.layers.len() {
        layer_forward(cfg, model, li, shards, caches, x, s, pos_base, &mut kv.layers[li], phases.as_deref_mut())?;
    }
    Ok(())
}

/// Decode/prefill step returning logits for only the LAST new position — mirrors
/// `crate::generate::step`.
pub fn step(model: &Model, shards: &Shards, caches: &mut ExpertCaches, kv: &mut KvState, ids: &[usize], pos_base: usize) -> Result<Vec<f32>, ModelError> {
    let s = ids.len();
    let d = model.cfg.hidden as usize;
    let mut x = embed_tokens(model, ids);
    layers_forward(model, shards, caches, &mut x, s, pos_base, kv)?;

    let mut last = x[(s - 1) * d..s * d].to_vec();
    rmsnorm(&mut last, &model.final_norm, model.cfg.eps);
    let mut logit = vec![0f32; model.cfg.vocab as usize];
    matmul_qt(&mut logit, &last, &model.lm_head, 1);
    Ok(logit)
}

/// Like `step`, but also returns a [`StepProfile`] — mirrors `crate::generate::step_profiled`,
/// the HTTP server's `/profile` dashboard's data source via `chat.rs`'s `generate_reply`.
pub fn step_profiled(model: &Model, shards: &Shards, caches: &mut ExpertCaches, kv: &mut KvState, ids: &[usize], pos_base: usize) -> Result<(Vec<f32>, StepProfile), ModelError> {
    let s = ids.len();
    let d = model.cfg.hidden as usize;
    let mut x = embed_tokens(model, ids);
    let mut phases = Phases::default();
    layers_forward_profiled(model, shards, caches, &mut x, s, pos_base, kv, Some(&mut phases))?;

    let mut last = x[(s - 1) * d..s * d].to_vec();
    rmsnorm(&mut last, &model.final_norm, model.cfg.eps);
    let mut logit = vec![0f32; model.cfg.vocab as usize];
    let t = std::time::Instant::now();
    matmul_qt(&mut logit, &last, &model.lm_head, 1);
    let lm_head_s = t.elapsed().as_secs_f32();
    Ok((logit, StepProfile { phases, lm_head_s }))
}

/// Like `step`, but returns logits for EVERY new position `[S,vocab]` — mirrors
/// `crate::generate::step_all`.
pub fn step_all(model: &Model, shards: &Shards, caches: &mut ExpertCaches, kv: &mut KvState, ids: &[usize], pos_base: usize) -> Result<Vec<f32>, ModelError> {
    let s = ids.len();
    let d = model.cfg.hidden as usize;
    let v = model.cfg.vocab as usize;
    let mut x = embed_tokens(model, ids);
    layers_forward(model, shards, caches, &mut x, s, pos_base, kv)?;

    let mut lo = vec![0f32; s * v];
    for si in 0..s {
        let mut row = x[si * d..(si + 1) * d].to_vec();
        rmsnorm(&mut row, &model.final_norm, model.cfg.eps);
        matmul_qt(&mut lo[si * v..(si + 1) * v], &row, &model.lm_head, 1);
    }
    Ok(lo)
}

#[cfg(test)]
mod tests {
    use super::*;
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

    fn f32_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// A tiny 2-layer model (layer 0: KDA, layer 1: MLA), both dense FFN — isolates KDA/MLA
    /// dispatch correctness from MoE routing/expert-loading, which
    /// `kda_layer_with_real_moe_ffn_produces_correct_length_logits` below covers separately.
    fn build_two_layer_fixture(name: &str) -> TempDir {
        let dir = TempDir::new(name);
        let mut seed = 3u32;
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_string(), json!({"format": "rabbit-test"}));
        let mut data = Vec::new();
        let mut add = |header: &mut serde_json::Map<String, serde_json::Value>, name: String, shape: Vec<usize>| {
            let n: usize = shape.iter().product::<usize>().max(1);
            let bytes = f32_bytes(&random_vec(n, &mut seed));
            let start = data.len() as u64;
            data.extend_from_slice(&bytes);
            let end = data.len() as u64;
            header.insert(name, json!({"dtype": "F32", "shape": shape, "data_offsets": [start, end]}));
        };

        let d = 8;
        let h = 2;
        let qk_nope = 3;
        let qk_rope = 2;
        let qh = qk_nope + qk_rope;
        let v_head = 4;
        let kv_lora = 5;
        let vocab = 16;
        let dense_inter = 10;
        let kda_head_dim = 4;
        let kda_n_heads = 2;
        let d_inner = kda_head_dim * kda_n_heads;
        let kernel = 3;

        add(&mut header, "model.embed_tokens.weight".into(), vec![vocab, d]);
        add(&mut header, "lm_head.weight".into(), vec![vocab, d]);
        add(&mut header, "model.norm.weight".into(), vec![d]);

        for i in 0..2usize {
            let p = |s: &str| format!("model.layers.{i}.{s}");
            add(&mut header, p("input_layernorm.weight"), vec![d]);
            add(&mut header, p("post_attention_layernorm.weight"), vec![d]);
            add(&mut header, p("mlp.gate_proj.weight"), vec![dense_inter, d]);
            add(&mut header, p("mlp.up_proj.weight"), vec![dense_inter, d]);
            add(&mut header, p("mlp.down_proj.weight"), vec![d, dense_inter]);

            let ap = |s: &str| format!("model.layers.{i}.self_attn.{s}");
            if i == 0 {
                add(&mut header, ap("q_proj.weight"), vec![d_inner, d]);
                add(&mut header, ap("k_proj.weight"), vec![d_inner, d]);
                add(&mut header, ap("v_proj.weight"), vec![d_inner, d]);
                add(&mut header, ap("q_conv1d.weight"), vec![d_inner, 1, kernel]);
                add(&mut header, ap("k_conv1d.weight"), vec![d_inner, 1, kernel]);
                add(&mut header, ap("v_conv1d.weight"), vec![d_inner, 1, kernel]);
                add(&mut header, ap("f_a_proj.weight"), vec![kda_head_dim, d]);
                add(&mut header, ap("f_b_proj.weight"), vec![d_inner, kda_head_dim]);
                add(&mut header, ap("dt_bias"), vec![d_inner]);
                add(&mut header, ap("A_log"), vec![1, 1, kda_n_heads, 1]);
                add(&mut header, ap("b_proj.weight"), vec![kda_n_heads, d]);
                add(&mut header, ap("g_a_proj.weight"), vec![kda_head_dim, d]);
                add(&mut header, ap("g_b_proj.weight"), vec![d_inner, kda_head_dim]);
                add(&mut header, ap("o_norm.weight"), vec![kda_head_dim]);
                add(&mut header, ap("o_proj.weight"), vec![d, d_inner]);
            } else {
                add(&mut header, ap("q_proj.weight"), vec![h * qh, d]);
                add(&mut header, ap("kv_a_proj_with_mqa.weight"), vec![kv_lora + qk_rope, d]);
                add(&mut header, ap("kv_a_layernorm.weight"), vec![kv_lora]);
                add(&mut header, ap("kv_b_proj.weight"), vec![h * (qk_nope + v_head), kv_lora]);
                add(&mut header, ap("o_proj.weight"), vec![d, h * v_head]);
            }
        }

        let cfg_json = json!({
            "model_type": "kimi_linear",
            "hidden_size": d, "num_hidden_layers": 2, "num_attention_heads": h,
            "first_k_dense_replace": 2, "q_lora_rank": null, "kv_lora_rank": kv_lora,
            "qk_nope_head_dim": qk_nope, "qk_rope_head_dim": qk_rope, "v_head_dim": v_head,
            "num_experts": 1, "num_experts_per_token": 1, "num_shared_experts": 0,
            "num_expert_group": 1, "topk_group": 1, "moe_intermediate_size": 1,
            "intermediate_size": dense_inter, "vocab_size": vocab, "moe_renormalize": true,
            "rms_norm_eps": 1e-5, "routed_scaling_factor": 1.0, "mla_use_nope": true,
            "moe_router_activation_func": "sigmoid", "rope_theta": 10000.0,
            "linear_attn_config": {
                "head_dim": kda_head_dim, "num_heads": kda_n_heads, "short_conv_kernel_size": kernel,
                "kda_layers": [1], "full_attn_layers": [2]
            }
        });
        fs::write(dir.0.join("config.json"), cfg_json.to_string()).unwrap();

        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_bytes);
        out.extend_from_slice(&data);
        fs::write(dir.0.join("model.safetensors"), out).unwrap();

        dir
    }

    /// A single KDA layer with a REAL MoE FFN (routed experts included, Kimi's real
    /// `block_sparse_moe.experts.{eid}.{w1,w2,w3}` naming) — exercises `ExpertNaming::
    /// KimiLinear` end to end through `layers_forward`/`step`, not just `expert_cache.rs`'s own
    /// unit tests.
    fn build_kda_moe_fixture(name: &str) -> TempDir {
        let dir = TempDir::new(name);
        let mut seed = 21u32;
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_string(), json!({"format": "rabbit-test"}));
        let mut data = Vec::new();
        let mut add = |header: &mut serde_json::Map<String, serde_json::Value>, name: String, shape: Vec<usize>| {
            let n: usize = shape.iter().product::<usize>().max(1);
            let bytes = f32_bytes(&random_vec(n, &mut seed));
            let start = data.len() as u64;
            data.extend_from_slice(&bytes);
            let end = data.len() as u64;
            header.insert(name, json!({"dtype": "F32", "shape": shape, "data_offsets": [start, end]}));
        };

        let d = 8;
        let vocab = 16;
        let kda_head_dim = 4;
        let kda_n_heads = 2;
        let d_inner = kda_head_dim * kda_n_heads;
        let kernel = 3;
        let n_experts = 3;
        let topk = 2;
        let moe_inter = 5;
        let n_shared = 1;

        add(&mut header, "model.embed_tokens.weight".into(), vec![vocab, d]);
        add(&mut header, "lm_head.weight".into(), vec![vocab, d]);
        add(&mut header, "model.norm.weight".into(), vec![d]);

        add(&mut header, "model.layers.0.input_layernorm.weight".into(), vec![d]);
        add(&mut header, "model.layers.0.post_attention_layernorm.weight".into(), vec![d]);
        let ap = |s: &str| format!("model.layers.0.self_attn.{s}");
        add(&mut header, ap("q_proj.weight"), vec![d_inner, d]);
        add(&mut header, ap("k_proj.weight"), vec![d_inner, d]);
        add(&mut header, ap("v_proj.weight"), vec![d_inner, d]);
        add(&mut header, ap("q_conv1d.weight"), vec![d_inner, 1, kernel]);
        add(&mut header, ap("k_conv1d.weight"), vec![d_inner, 1, kernel]);
        add(&mut header, ap("v_conv1d.weight"), vec![d_inner, 1, kernel]);
        add(&mut header, ap("f_a_proj.weight"), vec![kda_head_dim, d]);
        add(&mut header, ap("f_b_proj.weight"), vec![d_inner, kda_head_dim]);
        add(&mut header, ap("dt_bias"), vec![d_inner]);
        add(&mut header, ap("A_log"), vec![1, 1, kda_n_heads, 1]);
        add(&mut header, ap("b_proj.weight"), vec![kda_n_heads, d]);
        add(&mut header, ap("g_a_proj.weight"), vec![kda_head_dim, d]);
        add(&mut header, ap("g_b_proj.weight"), vec![d_inner, kda_head_dim]);
        add(&mut header, ap("o_norm.weight"), vec![kda_head_dim]);
        add(&mut header, ap("o_proj.weight"), vec![d, d_inner]);

        let mp = |s: &str| format!("model.layers.0.block_sparse_moe.{s}");
        add(&mut header, mp("gate.weight"), vec![n_experts, d]);
        add(&mut header, mp("gate.e_score_correction_bias"), vec![n_experts]);
        let s_i = moe_inter * n_shared;
        add(&mut header, mp("shared_experts.gate_proj.weight"), vec![s_i, d]);
        add(&mut header, mp("shared_experts.up_proj.weight"), vec![s_i, d]);
        add(&mut header, mp("shared_experts.down_proj.weight"), vec![d, s_i]);
        for eid in 0..n_experts {
            add(&mut header, format!("model.layers.0.block_sparse_moe.experts.{eid}.w1.weight"), vec![moe_inter, d]);
            add(&mut header, format!("model.layers.0.block_sparse_moe.experts.{eid}.w2.weight"), vec![d, moe_inter]);
            add(&mut header, format!("model.layers.0.block_sparse_moe.experts.{eid}.w3.weight"), vec![moe_inter, d]);
        }

        let cfg_json = json!({
            "model_type": "kimi_linear",
            "hidden_size": d, "num_hidden_layers": 1, "num_attention_heads": 2,
            "first_k_dense_replace": 0, "q_lora_rank": null, "kv_lora_rank": 4,
            "qk_nope_head_dim": 2, "qk_rope_head_dim": 2, "v_head_dim": 3,
            "num_experts": n_experts, "num_experts_per_token": topk, "num_shared_experts": n_shared,
            "num_expert_group": 1, "topk_group": 1, "moe_intermediate_size": moe_inter,
            "intermediate_size": 1, "vocab_size": vocab, "moe_renormalize": true,
            "rms_norm_eps": 1e-5, "routed_scaling_factor": 1.0, "mla_use_nope": true,
            "moe_router_activation_func": "sigmoid", "rope_theta": 10000.0,
            "linear_attn_config": {
                "head_dim": kda_head_dim, "num_heads": kda_n_heads, "short_conv_kernel_size": kernel,
                "kda_layers": [1], "full_attn_layers": []
            }
        });
        fs::write(dir.0.join("config.json"), cfg_json.to_string()).unwrap();

        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_bytes);
        out.extend_from_slice(&data);
        fs::write(dir.0.join("model.safetensors"), out).unwrap();

        dir
    }

    #[test]
    fn kda_layer_with_real_moe_ffn_produces_correct_length_logits() {
        let fixture = build_kda_moe_fixture("rabbit_test_kimi_generate_kda_moe");
        let model = Model::load(&fixture.0, 32, 32).unwrap();
        let shards = Shards::open(&fixture.0).unwrap();

        let mut caches = ExpertCaches::new(&model, 8);
        let mut kv = KvState::new(&model);
        let logits = step(&model, &shards, &mut caches, &mut kv, &[0, 1, 2], 0).unwrap();
        assert_eq!(logits.len(), 16); // vocab

        // a decode-style follow-up step must also succeed -- proves the ExpertCache (now
        // holding real loaded experts) keeps working across calls, not just the first one.
        let logits2 = step(&model, &shards, &mut caches, &mut kv, &[3], 3).unwrap();
        assert_eq!(logits2.len(), 16);
    }

    #[test]
    fn layers_forward_matches_manual_per_layer_composition() {
        let fixture = build_two_layer_fixture("rabbit_test_kimi_generate_layers");
        let model = Model::load(&fixture.0, 32, 32).unwrap();
        let shards = Shards::open(&fixture.0).unwrap();
        let s = 3;
        let d = model.cfg.hidden as usize;
        let mut seed = 44u32;
        let x0 = random_vec(s * d, &mut seed);

        let mut caches_a = ExpertCaches::new(&model, 4);
        let mut kv_a = KvState::new(&model);
        let mut x_lf = x0.clone();
        layers_forward(&model, &shards, &mut caches_a, &mut x_lf, s, 0, &mut kv_a).unwrap();

        // independent manual reference: same primitives, re-derived by hand.
        let mut caches_b = ExpertCaches::new(&model, 4);
        let mut kv_b = KvState::new(&model);
        let mut x_manual = x0.clone();
        for li in 0..2 {
            layer_forward(&model.cfg, &model, li, &shards, &mut caches_b, &mut x_manual, s, 0, &mut kv_b.layers[li], None).unwrap();
        }

        for (a, b) in x_lf.iter().zip(&x_manual) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn step_and_step_all_agree_on_the_last_position() {
        let fixture = build_two_layer_fixture("rabbit_test_kimi_generate_step_consistency");
        let model = Model::load(&fixture.0, 32, 32).unwrap();
        let shards = Shards::open(&fixture.0).unwrap();
        let ids = vec![1usize, 2, 3];

        let mut caches_a = ExpertCaches::new(&model, 4);
        let mut kv_a = KvState::new(&model);
        let last_only = step(&model, &shards, &mut caches_a, &mut kv_a, &ids, 0).unwrap();

        let mut caches_b = ExpertCaches::new(&model, 4);
        let mut kv_b = KvState::new(&model);
        let all = step_all(&model, &shards, &mut caches_b, &mut kv_b, &ids, 0).unwrap();
        let v = model.cfg.vocab as usize;
        let last_row = &all[(ids.len() - 1) * v..ids.len() * v];

        assert_eq!(last_only, last_row);
    }

    #[test]
    fn step_profiled_matches_step_and_reports_nonzero_attention_time() {
        let fixture = build_two_layer_fixture("rabbit_test_kimi_generate_step_profiled");
        let model = Model::load(&fixture.0, 32, 32).unwrap();
        let shards = Shards::open(&fixture.0).unwrap();
        let ids = vec![1usize, 2, 3];

        let mut caches_a = ExpertCaches::new(&model, 4);
        let mut kv_a = KvState::new(&model);
        let plain = step(&model, &shards, &mut caches_a, &mut kv_a, &ids, 0).unwrap();

        let mut caches_b = ExpertCaches::new(&model, 4);
        let mut kv_b = KvState::new(&model);
        let (profiled, profile) = step_profiled(&model, &shards, &mut caches_b, &mut kv_b, &ids, 0).unwrap();

        assert_eq!(plain, profiled, "step_profiled must compute the exact same logits as step");
        assert!(profile.phases.attention_s > 0.0, "2 attention layers (1 KDA, 1 MLA) ran; attention_s must be nonzero");
        assert!(profile.phases.expert_matmul_s >= 0.0);
        assert!(profile.phases.expert_wait_s >= 0.0);
        assert!(profile.lm_head_s >= 0.0);
    }

    #[test]
    fn kda_layer_state_carries_across_sequential_single_token_steps() {
        // Decoding one token at a time (s=1 per call, growing pos_base) must produce the exact
        // same final logits as one prefill call covering the same tokens -- proof KdaLayerState
        // is actually being threaded through and accumulating, not reset/ignored between calls.
        let fixture = build_two_layer_fixture("rabbit_test_kimi_generate_kda_state_carries");
        let model = Model::load(&fixture.0, 32, 32).unwrap();
        let shards = Shards::open(&fixture.0).unwrap();
        let ids = vec![1usize, 2, 3];

        let mut caches_prefill = ExpertCaches::new(&model, 4);
        let mut kv_prefill = KvState::new(&model);
        let prefill_logits = step(&model, &shards, &mut caches_prefill, &mut kv_prefill, &ids, 0).unwrap();

        let mut caches_seq = ExpertCaches::new(&model, 4);
        let mut kv_seq = KvState::new(&model);
        let mut logits = Vec::new();
        for (pos, &id) in ids.iter().enumerate() {
            logits = step(&model, &shards, &mut caches_seq, &mut kv_seq, &[id], pos).unwrap();
        }

        assert_eq!(prefill_logits, logits, "sequential single-token decode must match one prefill call bit-for-bit");
    }
}
