//! Kimi K3's own `layer_forward`/`layers_forward`/`step`, the sibling of
//! `kimi_linear::generate.rs` (itself the sibling of `crate::generate`'s GLM-5.2 version) — not a
//! variant: reuses MLA/KDA's per-token math primitives (`kda.rs`/`short_conv.rs`/`ops.rs`,
//! `glm52::attention::attention()`) unchanged, but wires them together differently per this
//! module's own doc in `model.rs` (five real K3-only differences: non-null `q_lora`, `SituAndMul`
//! activation, Stable LatentMoE, Attention Residuals, the two new attention gates).
//!
//! `KdaLayerState`/`LayerState`/`KvState` are a deliberate near-verbatim copy of
//! `kimi_linear::generate.rs`'s own types, not a reuse of them: that module's `KdaLayerState`
//! only exposes READ-ONLY `pub(crate)` accessors (for `kv_session.rs`'s save path) — no mutable
//! access to `heads`/`q_conv`/`k_conv`/`v_conv` exists outside that module, since
//! `kimi_linear::generate::kda_step` (the only place that needs mutation) lives inside it. K3's
//! own `kda_step` below needs that same mutation from a DIFFERENT module, so it needs its own
//! copy of the state shape rather than punching new `pub(crate) mut` holes through a type whose
//! privacy was a deliberate encapsulation choice — matching this codebase's established
//! "structurally-identical-but-distinct sibling types get duplicated, not shared" convention
//! (see e.g. `ops.rs`'s `swish` vs `glm52::moe.rs`'s `siluf`).
//!
//! **Attention Residuals is purely transient** (see `attn_res.rs`'s doc) — `AttnResState` is
//! created fresh in `layers_forward`/`layers_forward_profiled` and threaded through every layer
//! in one forward pass, never touching `KvState`/session persistence.

use crate::expert_cache::{ExpertCache, ExpertNaming};
use crate::generate::{Phases, StepProfile};
use crate::glm52::attention::{self, rmsnorm, OutputGate};
use crate::glm52::moe::{self, Activation};
use crate::kernels::{matmul_qt, matvec_sharded_batch, DenseQT};
use crate::kimi_k3::attn_res::{self, AttnResState};
use crate::kimi_k3::config::Cfg;
use crate::kimi_k3::model::{self, Attn, Ffn, KdaOutputGate, KdaWeights, Model, ModelError};
use crate::kimi_k3::moe as k3_moe;
use crate::kimi_linear::kda::{decay_gate, KdaState};
use crate::kimi_linear::ops::{head_output_gate, l2_norm, sigmoid, swish};
use crate::kimi_linear::short_conv::ShortConvState;
use crate::safetensors::Shards;
use rayon::prelude::*;

/// One KDA layer's recurrent state — see this module's doc for why this is its own copy rather
/// than a reuse of `kimi_linear::generate::KdaLayerState`.
pub struct KdaLayerState {
    heads: Vec<KdaState>,
    q_conv: ShortConvState,
    k_conv: ShortConvState,
    v_conv: ShortConvState,
}

impl KdaLayerState {
    pub fn new(cfg: &Cfg) -> KdaLayerState {
        let head_dim = cfg.base.kda_head_dim as usize;
        let n_heads = cfg.base.kda_n_heads as usize;
        let d_inner = head_dim * n_heads;
        let kernel = cfg.base.short_conv_kernel as usize;
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

pub enum LayerState {
    Kda(KdaLayerState),
    Mla(attention::KvCache),
}

/// Per-layer state for one generation session, the sibling of `kimi_linear::generate::KvState`.
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
                Attn::Mla(_) => LayerState::Mla(attention::KvCache::new(model.cfg.base.kv_lora as usize, model.cfg.base.qk_rope as usize)),
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

/// One `ExpertCache` per MoE layer — the sibling of `kimi_linear::generate::ExpertCaches`
/// (identical wiring, ported verbatim; see that module's doc for why this isn't generalized
/// into one shared type across families).
pub struct ExpertCaches(Vec<Option<ExpertCache>>);

impl ExpertCaches {
    pub fn new(model: &Model, capacity: usize) -> ExpertCaches {
        // Real published checkpoints have their routed experts natively MXFP4-quantized on disk
        // (`cfg.mxfp4_experts`, parsed from `quantization_config`) and need `KimiK3Mxfp4`'s
        // `.weight_packed`/`.weight_scale` naming; every synthetic test fixture in this crate
        // uses plain `.weight` float tensors and needs the original `KimiK3` naming.
        let naming = if model.cfg.mxfp4_experts { ExpertNaming::KimiK3Mxfp4 } else { ExpertNaming::KimiK3 };
        let v = model.layers.iter().map(|l| if matches!(l.ffn, Ffn::Moe(_)) { Some(ExpertCache::for_family(capacity, model.cfg.base.n_experts as usize, naming)) } else { None }).collect();
        ExpertCaches(v)
    }

    pub fn hit_miss_totals(&self) -> (u64, u64, u64) {
        self.0.iter().flatten().fold((0, 0, 0), |(h, m, n), c| (h + c.hits, m + c.misses, n + c.load_nanos))
    }

    pub fn io_wait_nanos_total(&self) -> u64 {
        self.0.iter().flatten().map(|c| c.io_wait_nanos).sum()
    }

    /// Whether any layer's cache loads through an `io_uring` ring. `false` (every MXFP4/K3 run,
    /// whose naming never gets a ring) means `io_wait_nanos_total` is structurally zero, so the
    /// CLI must not present it as "actual disk wait" (Phase 4c) — see `ExpertCache::has_ring`.
    pub fn any_has_ring(&self) -> bool {
        self.0.iter().flatten().any(|c| c.has_ring())
    }

    /// Phase 4b: preload every MoE layer's routed experts up front. K3's experts live at the
    /// LATENT width, so `to_glm_cfg_expert` (not `to_glm_cfg`) supplies the shapes — the same
    /// config `latent_moe`'s own `ensure_loaded` uses. See `expert_cache::preload_layers`.
    pub fn preload(&mut self, model: &Model, shards: &Shards) -> Result<(), ModelError> {
        let cfg = crate::kimi_k3::model::to_glm_cfg_expert(&model.cfg);
        let n_experts = cfg.n_experts as usize;
        Ok(crate::expert_cache::preload_layers(&mut self.0, shards, &cfg, model.ebits, n_experts)?)
    }

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

    pub fn save_usage(&self, model_dir: &std::path::Path) -> std::io::Result<()> {
        let path = crate::usage_cache::usage_path(model_dir);
        let entries = self.0.iter().enumerate().flat_map(|(li, c)| {
            c.as_ref().into_iter().flat_map(move |cache| cache.usage_counts().map(move |(eid, count)| (li, eid, count)))
        });
        crate::usage_cache::save(&path, entries)
    }
}

fn embed_tokens(model: &Model, ids: &[usize]) -> Vec<f32> {
    let d = model.cfg.base.hidden as usize;
    let mut x = vec![0f32; ids.len() * d];
    for (si, &tok) in ids.iter().enumerate() {
        x[si * d..(si + 1) * d].copy_from_slice(&model.embed_row(tok));
    }
    x
}

/// One KDA layer's forward for a single token — same math as `kimi_linear::generate::kda_step`
/// (see that function's doc for the full derivation, unchanged here) with two K3-only
/// differences: `decay_gate` gets `cfg.kda_gate_lower_bound` (the bounded formula, `None` on
/// checkpoints that don't set it — see `kda.rs::decay_gate`'s doc), and the output gate's `g2`
/// input comes from EITHER `w.output_gate`'s low-rank or full-rank projection (the math
/// consuming `g2`, `head_output_gate`, is unchanged either way).
fn kda_step(cfg: &Cfg, w: &KdaWeights, state: &mut KdaLayerState, x: &[f32], out: &mut [f32], mut phases: Option<&mut Phases>) {
    let head_dim = cfg.base.kda_head_dim as usize;
    let n_heads = cfg.base.kda_n_heads as usize;
    let d_inner = head_dim * n_heads;
    let eps = cfg.base.eps;

    // Phase N5a instrumentation (timing only — every operation below is byte-for-byte what it
    // was): the step's cost is bucketed into its projection matmuls vs everything recurrent
    // (convs, activations/norms, the 96-head state update). A handful of `Instant::now()` calls
    // against a multi-ms step; noise. See `Phases::attn_kda_proj_s`'s doc.
    let mut proj_ns = 0u128;
    let mut recur_ns = 0u128;
    let mut t = std::time::Instant::now();

    let mut q_pre = vec![0f32; d_inner];
    let mut k_pre = vec![0f32; d_inner];
    let mut v_pre = vec![0f32; d_inner];
    // N4: sharded q/k/v run as ONE cross-domain fan-out (three weights amortizing one fan-out's
    // wake cost — at 69 KDA layers/token the fan-out count is the price that matters). Plain
    // weights (no --numa) take exactly the pre-N4 three matmuls.
    match (&w.q_proj, &w.k_proj, &w.v_proj, crate::numa::NodePools::get()) {
        (DenseQT::Sharded(q), DenseQT::Sharded(k), DenseQT::Sharded(v), Some(pools)) => {
            matvec_sharded_batch(pools, x, &mut [(&mut q_pre, q), (&mut k_pre, k), (&mut v_pre, v)]);
        }
        _ => {
            w.q_proj.matvec(&mut q_pre, x);
            w.k_proj.matvec(&mut k_pre, x);
            w.v_proj.matvec(&mut v_pre, x);
        }
    }
    proj_ns += t.elapsed().as_nanos();
    t = std::time::Instant::now();

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
    let q_scale = 1.0 / (head_dim as f32).sqrt();
    for h in 0..n_heads {
        let sl = h * head_dim..(h + 1) * head_dim;
        l2_norm(&mut q[sl.clone()], eps);
        l2_norm(&mut k[sl.clone()], eps);
        for c in &mut q[sl] {
            *c *= q_scale;
        }
    }
    recur_ns += t.elapsed().as_nanos();
    t = std::time::Instant::now();

    let mut f_a = vec![0f32; head_dim];
    matmul_qt(&mut f_a, x, &w.f_a_proj, 1);
    let mut g = vec![0f32; d_inner];
    matmul_qt(&mut g, &f_a, &w.f_b_proj, 1);

    let mut beta_pre = vec![0f32; n_heads];
    matmul_qt(&mut beta_pre, x, &w.b_proj, 1);

    let mut g2 = vec![0f32; d_inner];
    match &w.output_gate {
        KdaOutputGate::FullRank { g_proj } => {
            matmul_qt(&mut g2, x, g_proj, 1);
        }
        KdaOutputGate::LowRank { g_a_proj, g_b_proj } => {
            let mut g_a = vec![0f32; head_dim];
            matmul_qt(&mut g_a, x, g_a_proj, 1);
            matmul_qt(&mut g2, &g_a, g_b_proj, 1);
        }
    }
    proj_ns += t.elapsed().as_nanos();
    t = std::time::Instant::now();

    let mut o = vec![0f32; d_inner];
    state.heads.par_iter_mut().zip(o.par_chunks_mut(head_dim)).enumerate().for_each(|(h, (head_state, o_slot))| {
        let sl = h * head_dim..(h + 1) * head_dim;
        let mut alpha = vec![0f32; head_dim];
        decay_gate(w.a_log[h], &g[sl.clone()], &w.dt_bias[sl.clone()], cfg.kda_gate_lower_bound, &mut alpha);
        let beta = sigmoid(beta_pre[h]);
        head_state.step(&q[sl.clone()], &k[sl.clone()], &v[sl.clone()], &alpha, beta, o_slot);
        head_output_gate(o_slot, &w.o_norm, &g2[sl], eps);
    });
    recur_ns += t.elapsed().as_nanos();
    t = std::time::Instant::now();

    w.o_proj.matvec(out, &o);
    proj_ns += t.elapsed().as_nanos();

    if let Some(p) = phases.as_deref_mut() {
        p.attn_kda_proj_s += proj_ns as f32 / 1e9;
        p.attn_kda_recur_s += recur_ns as f32 / 1e9;
    }
}

/// One transformer layer, Attention-Residuals-aware: when `cfg.attn_res_block > 0` (and
/// `attn_res_state` is `Some`), wraps self-attention and the MLP/MoE with `attn_res::
/// before_attention`/`before_mlp`/`after_mlp` exactly as `_forward_attn_residual` does (see
/// `attn_res.rs`'s doc for the full derivation); otherwise degrades to the plain
/// `x += attention(...); x += ffn(...)` residual structure every other architecture in this
/// crate uses.
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
    attn_res_state: &mut Option<AttnResState>,
    mut phases: Option<&mut Phases>,
) -> Result<(), ModelError> {
    let d = cfg.base.hidden as usize;
    let layer = &model.layers[li];
    let eps = cfg.base.eps;
    let block = cfg.attn_res_block as usize;

    let hidden_in = x.to_vec();

    let attn_input = match (&layer.attn_res, attn_res_state.as_mut()) {
        (Some(ar), Some(state)) => attn_res::before_attention(state, &hidden_in, d, li, block, &ar.self_attn, eps),
        _ => hidden_in.clone(),
    };

    let mut nrm = attn_input;
    for si in 0..s {
        rmsnorm(&mut nrm[si * d..(si + 1) * d], &layer.in_ln, eps);
    }

    let mut attn_delta = vec![0f32; s * d];
    let attn_t = std::time::Instant::now();
    match (&layer.attn, layer_state) {
        (Attn::Kda(w), LayerState::Kda(state)) => {
            for si in 0..s {
                kda_step(cfg, w, state, &nrm[si * d..(si + 1) * d], &mut attn_delta[si * d..(si + 1) * d], phases.as_deref_mut());
            }
        }
        (Attn::Mla(w), LayerState::Mla(kv)) => {
            let (glm_cfg, dsa, absorb, rope, qproj) = model::mla_call_args(cfg);
            let output_gate = match &w.output_gate {
                Some(g_proj) => OutputGate::On(g_proj),
                None => OutputGate::Off,
            };
            attention::attention(&glm_cfg, &w.attn, kv, &nrm, s, pos_base, dsa, absorb, rope, qproj, output_gate, &mut attn_delta);
            if let Some(p) = phases.as_deref_mut() {
                p.attn_mla_s += attn_t.elapsed().as_secs_f32();
            }
        }
        _ => unreachable!("Attn/LayerState variant mismatch -- KvState::new always pairs them per layer"),
    }
    if let Some(p) = phases.as_deref_mut() {
        p.attention_s += attn_t.elapsed().as_secs_f32();
    }

    let (mlp_input, prefix_sum) = match (&layer.attn_res, attn_res_state.as_mut()) {
        (Some(ar), Some(state)) => attn_res::before_mlp(state, &hidden_in, &attn_delta, d, li, block, &ar.mlp, eps),
        _ => {
            let ps: Vec<f32> = hidden_in.iter().zip(&attn_delta).map(|(&a, &b)| a + b).collect();
            (ps.clone(), ps)
        }
    };

    let mut nrm2 = mlp_input;
    for si in 0..s {
        rmsnorm(&mut nrm2[si * d..(si + 1) * d], &layer.post_ln, eps);
    }

    let activation = model::ffn_activation(cfg);
    let mut ffn_delta = vec![0f32; s * d];
    match &layer.ffn {
        Ffn::Dense(w) => {
            let t = std::time::Instant::now();
            moe::dense_mlp(w, &nrm2, s, cfg.base.dense_inter as usize, activation, &mut ffn_delta);
            if let Some(p) = phases.as_deref_mut() {
                p.expert_matmul_s += t.elapsed().as_secs_f32();
            }
        }
        Ffn::Moe(w) => {
            let cache = caches.0[li].as_mut().expect("MoE layer must have an ExpertCache");
            let wait_before = cache.io_wait_nanos;
            let t = std::time::Instant::now();
            let glm_cfg_full = model::to_glm_cfg(cfg);
            match &w.latent {
                Some(latent) => {
                    let glm_cfg_expert = model::to_glm_cfg_expert(cfg);
                    let mut routed = vec![0f32; s * d];
                    k3_moe::latent_moe(
                        &glm_cfg_full,
                        &glm_cfg_expert,
                        &w.moe,
                        latent,
                        cache,
                        shards,
                        li,
                        model.ebits,
                        &model.route_cfg,
                        &nrm2,
                        s,
                        eps,
                        activation,
                        &mut routed,
                    )?;
                    // shared_experts(identity), computed at full hidden width -- matches
                    // KimiSparseMoeBlock.forward's `y = y + self.shared_experts(identity)`,
                    // added after the latent-MoE wrapper's up-projected output.
                    let s_i = (cfg.base.moe_inter * cfg.base.n_shared) as usize;
                    let mut sg = vec![0f32; s * s_i];
                    let mut su = vec![0f32; s * s_i];
                    matmul_qt(&mut sg, &nrm2, &w.moe.sh_gate, s);
                    matmul_qt(&mut su, &nrm2, &w.moe.sh_up, s);
                    match activation {
                        Activation::Silu => {
                            for (g, &u) in sg.iter_mut().zip(&su) {
                                *g = *g / (1.0 + (-*g).exp()) * u;
                            }
                        }
                        Activation::Situ { beta, linear_beta } => {
                            for (g, &u) in sg.iter_mut().zip(&su) {
                                let u = match linear_beta {
                                    Some(lb) => lb * (u / lb).tanh(),
                                    None => u,
                                };
                                let situ = beta * (*g / beta).tanh() * (1.0 / (1.0 + (-*g).exp()));
                                *g = situ * u;
                            }
                        }
                    }
                    matmul_qt(&mut ffn_delta, &sg, &w.moe.sh_down, s);
                    for (o, &r) in ffn_delta.iter_mut().zip(&routed) {
                        *o += r;
                    }
                }
                None => {
                    moe::moe(&glm_cfg_full, &w.moe, cache, shards, li, model.ebits, &model.route_cfg, &nrm2, s, activation, &mut ffn_delta)?;
                }
            }
            let elapsed = t.elapsed().as_secs_f32();
            if let Some(p) = phases {
                let wait_delta = ((cache.io_wait_nanos - wait_before) as f32 / 1e9).max(0.0);
                p.expert_wait_s += wait_delta;
                p.expert_matmul_s += (elapsed - wait_delta).max(0.0);
            }
        }
    }

    let final_x = match (&layer.attn_res, attn_res_state.as_mut()) {
        (Some(_), Some(_)) => attn_res::after_mlp(&prefix_sum, &ffn_delta),
        _ => prefix_sum.iter().zip(&ffn_delta).map(|(&a, &b)| a + b).collect(),
    };
    x.copy_from_slice(&final_x);

    Ok(())
}

/// Runs every layer in order on new tokens `x[S,hidden]`, updating `x` in place, then (when
/// `cfg.attn_res_block > 0`) applies the model-level output pooling — mirrors
/// `KimiLinearModel.forward`'s `hidden_states = self._apply_output_attn_res(hidden_states,
/// block_residual)` call right after the layer loop, before the final RMSNorm (applied by
/// `step`/`step_all`, not here — same split as every other architecture's `layers_forward`).
pub fn layers_forward(model: &Model, shards: &Shards, caches: &mut ExpertCaches, x: &mut [f32], s: usize, pos_base: usize, kv: &mut KvState) -> Result<(), ModelError> {
    layers_forward_profiled(model, shards, caches, x, s, pos_base, kv, None)
}

#[allow(clippy::too_many_arguments)]
fn layers_forward_profiled(model: &Model, shards: &Shards, caches: &mut ExpertCaches, x: &mut [f32], s: usize, pos_base: usize, kv: &mut KvState, mut phases: Option<&mut Phases>) -> Result<(), ModelError> {
    let cfg = &model.cfg;
    let mut attn_res_state = if cfg.attn_res_block > 0 { Some(AttnResState::new()) } else { None };
    for li in 0..model.layers.len() {
        layer_forward(cfg, model, li, shards, caches, x, s, pos_base, &mut kv.layers[li], &mut attn_res_state, phases.as_deref_mut())?;
    }
    if let (Some(ar), Some(state)) = (&model.output_attn_res, &attn_res_state) {
        let d = cfg.base.hidden as usize;
        let pooled = attn_res::output_pool(state, x, d, ar, cfg.base.eps);
        x.copy_from_slice(&pooled);
    }
    Ok(())
}

pub fn step(model: &Model, shards: &Shards, caches: &mut ExpertCaches, kv: &mut KvState, ids: &[usize], pos_base: usize) -> Result<Vec<f32>, ModelError> {
    let s = ids.len();
    let d = model.cfg.base.hidden as usize;
    let mut x = embed_tokens(model, ids);
    layers_forward(model, shards, caches, &mut x, s, pos_base, kv)?;

    let mut last = x[(s - 1) * d..s * d].to_vec();
    rmsnorm(&mut last, &model.final_norm, model.cfg.base.eps);
    let mut logit = vec![0f32; model.cfg.base.vocab as usize];
    model.lm_head.matvec(&mut logit, &last);
    Ok(logit)
}

pub fn step_profiled(model: &Model, shards: &Shards, caches: &mut ExpertCaches, kv: &mut KvState, ids: &[usize], pos_base: usize) -> Result<(Vec<f32>, StepProfile), ModelError> {
    let s = ids.len();
    let d = model.cfg.base.hidden as usize;
    let mut x = embed_tokens(model, ids);
    let mut phases = Phases::default();
    layers_forward_profiled(model, shards, caches, &mut x, s, pos_base, kv, Some(&mut phases))?;

    let mut last = x[(s - 1) * d..s * d].to_vec();
    rmsnorm(&mut last, &model.final_norm, model.cfg.base.eps);
    let mut logit = vec![0f32; model.cfg.base.vocab as usize];
    let t = std::time::Instant::now();
    model.lm_head.matvec(&mut logit, &last);
    let lm_head_s = t.elapsed().as_secs_f32();
    Ok((logit, StepProfile { phases, lm_head_s }))
}

pub fn step_all(model: &Model, shards: &Shards, caches: &mut ExpertCaches, kv: &mut KvState, ids: &[usize], pos_base: usize) -> Result<Vec<f32>, ModelError> {
    let s = ids.len();
    let d = model.cfg.base.hidden as usize;
    let v = model.cfg.base.vocab as usize;
    let mut x = embed_tokens(model, ids);
    layers_forward(model, shards, caches, &mut x, s, pos_base, kv)?;

    let mut lo = vec![0f32; s * v];
    for si in 0..s {
        let mut row = x[si * d..(si + 1) * d].to_vec();
        rmsnorm(&mut row, &model.final_norm, model.cfg.base.eps);
        model.lm_head.matvec(&mut lo[si * v..(si + 1) * v], &row);
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

    /// A tiny 2-layer K3-shaped fixture (layer 0: KDA, full-rank output gate, dense FFN; layer
    /// 1: MLA with `q_lora>0` + output gate, latent-MoE FFN) with Attention Residuals turned on
    /// (`attn_res_block_size=1` -- every layer is a checkpoint, the simplest nonzero setting) and
    /// `hidden_act="situ"` -- exercises every K3-only piece through the real `step`/`step_all`/
    /// `layers_forward` entry points, not just `Model::load`'s shapes (`model.rs`'s own tests
    /// already cover that).
    struct TinyFixture {
        dir: TempDir,
        tensors: Vec<(String, Vec<usize>, Vec<u8>)>,
        seed: u32,
    }

    impl TinyFixture {
        fn new(name: &str) -> Self {
            TinyFixture { dir: TempDir::new(&format!("rabbit_test_k3_generate_tiny_{name}")), tensors: Vec::new(), seed: 2 }
        }

        fn add(&mut self, name: &str, shape: Vec<usize>) {
            let n: usize = shape.iter().product::<usize>().max(1);
            let data = random_vec(n, &mut self.seed);
            self.tensors.push((name.to_string(), shape, f32_bytes(&data)));
        }

        fn build(mut self) -> TempDir {
            let d = 8;
            let h = 2;
            let qk_nope = 3;
            let qk_rope = 2;
            let qh = qk_nope + qk_rope;
            let v_head = 4;
            let kv_lora = 5;
            let q_lora = 6;
            let vocab = 16;
            let dense_inter = 10;
            let n_experts = 3;
            let topk = 2;
            let moe_inter = 5;
            let n_shared = 1;
            let kda_head_dim = 4;
            let kda_n_heads = 2;
            let d_inner = kda_head_dim * kda_n_heads;
            let kernel = 3;
            let moe_hidden = 4;

            self.add("language_model.model.embed_tokens.weight", vec![vocab, d]);
            self.add("language_model.lm_head.weight", vec![vocab, d]);
            self.add("language_model.model.norm.weight", vec![d]);
            self.add("language_model.model.output_attn_res_norm.weight", vec![d]);
            self.add("language_model.model.output_attn_res_proj.weight", vec![1, d]);

            for i in 0..2usize {
                let p = |s: &str| format!("language_model.model.layers.{i}.{s}");
                self.add(&p("input_layernorm.weight"), vec![d]);
                self.add(&p("post_attention_layernorm.weight"), vec![d]);
                self.add(&p("self_attention_res_norm.weight"), vec![d]);
                self.add(&p("self_attention_res_proj.weight"), vec![1, d]);
                self.add(&p("mlp_res_norm.weight"), vec![d]);
                self.add(&p("mlp_res_proj.weight"), vec![1, d]);

                let ap = |s: &str| format!("language_model.model.layers.{i}.self_attn.{s}");
                if i == 0 {
                    self.add(&ap("q_proj.weight"), vec![d_inner, d]);
                    self.add(&ap("k_proj.weight"), vec![d_inner, d]);
                    self.add(&ap("v_proj.weight"), vec![d_inner, d]);
                    self.add(&ap("q_conv1d.weight"), vec![d_inner, 1, kernel]);
                    self.add(&ap("k_conv1d.weight"), vec![d_inner, 1, kernel]);
                    self.add(&ap("v_conv1d.weight"), vec![d_inner, 1, kernel]);
                    self.add(&ap("f_a_proj.weight"), vec![kda_head_dim, d]);
                    self.add(&ap("f_b_proj.weight"), vec![d_inner, kda_head_dim]);
                    self.add(&ap("dt_bias"), vec![d_inner]);
                    self.add(&ap("A_log"), vec![1, 1, kda_n_heads, 1]);
                    self.add(&ap("b_proj.weight"), vec![kda_n_heads, d]);
                    self.add(&ap("g_proj.weight"), vec![d_inner, d]); // full-rank output gate
                    self.add(&ap("o_norm.weight"), vec![kda_head_dim]);
                    self.add(&ap("o_proj.weight"), vec![d, d_inner]);

                    self.add(&p("mlp.gate_proj.weight"), vec![dense_inter, d]);
                    self.add(&p("mlp.up_proj.weight"), vec![dense_inter, d]);
                    self.add(&p("mlp.down_proj.weight"), vec![d, dense_inter]);
                } else {
                    self.add(&ap("q_a_proj.weight"), vec![q_lora, d]);
                    self.add(&ap("q_a_layernorm.weight"), vec![q_lora]);
                    self.add(&ap("q_b_proj.weight"), vec![h * qh, q_lora]);
                    self.add(&ap("kv_a_proj_with_mqa.weight"), vec![kv_lora + qk_rope, d]);
                    self.add(&ap("kv_a_layernorm.weight"), vec![kv_lora]);
                    self.add(&ap("kv_b_proj.weight"), vec![h * (qk_nope + v_head), kv_lora]);
                    self.add(&ap("o_proj.weight"), vec![d, h * v_head]);
                    self.add(&ap("g_proj.weight"), vec![h * v_head, d]);

                    let mp = |s: &str| format!("language_model.model.layers.{i}.block_sparse_moe.{s}");
                    self.add(&mp("gate.weight"), vec![n_experts, d]);
                    self.add(&mp("gate.e_score_correction_bias"), vec![n_experts]);
                    let s_i = moe_inter * n_shared;
                    self.add(&mp("shared_experts.gate_proj.weight"), vec![s_i, d]);
                    self.add(&mp("shared_experts.up_proj.weight"), vec![s_i, d]);
                    self.add(&mp("shared_experts.down_proj.weight"), vec![d, s_i]);
                    self.add(&mp("routed_expert_down_proj.weight"), vec![moe_hidden, d]);
                    self.add(&mp("routed_expert_up_proj.weight"), vec![d, moe_hidden]);
                    self.add(&mp("routed_expert_norm.weight"), vec![moe_hidden]);
                    for eid in 0..n_experts {
                        self.add(&format!("language_model.model.layers.1.block_sparse_moe.experts.{eid}.w1.weight"), vec![moe_inter, moe_hidden]);
                        self.add(&format!("language_model.model.layers.1.block_sparse_moe.experts.{eid}.w2.weight"), vec![moe_hidden, moe_inter]);
                        self.add(&format!("language_model.model.layers.1.block_sparse_moe.experts.{eid}.w3.weight"), vec![moe_inter, moe_hidden]);
                    }
                }
            }

            let linear_attn_config = json!({
                "head_dim": kda_head_dim, "num_heads": kda_n_heads, "short_conv_kernel_size": kernel,
                "gate_lower_bound": -5.0, "use_full_rank_gate": true,
                "kda_layers": [1], "full_attn_layers": [2]
            });
            let text_config = json!({
                "model_type": "kimi_linear",
                "hidden_size": d, "num_hidden_layers": 2, "num_attention_heads": h,
                "first_k_dense_replace": 1, "q_lora_rank": q_lora, "kv_lora_rank": kv_lora,
                "qk_nope_head_dim": qk_nope, "qk_rope_head_dim": qk_rope, "v_head_dim": v_head,
                "num_experts": n_experts, "num_experts_per_token": topk, "num_shared_experts": n_shared,
                "num_expert_group": 1, "topk_group": 1, "moe_intermediate_size": moe_inter,
                "intermediate_size": dense_inter, "vocab_size": vocab, "moe_renormalize": true,
                "rms_norm_eps": 1e-5, "routed_scaling_factor": 1.0, "mla_use_nope": true,
                "moe_router_activation_func": "sigmoid", "rope_theta": 10000.0,
                "hidden_act": "situ", "activation_situ_beta": 4.0, "activation_situ_linear_beta": 25.0,
                "routed_expert_hidden_size": moe_hidden, "latent_moe_use_norm": true,
                "attn_res_block_size": 1, "mla_use_output_gate": true,
                "linear_attn_config": linear_attn_config
            });
            let cfg_json = json!({ "model_type": "kimi_k3", "text_config": text_config });
            fs::write(self.dir.0.join("config.json"), cfg_json.to_string()).unwrap();

            let mut header = serde_json::Map::new();
            header.insert("__metadata__".to_string(), json!({"format": "rabbit-test"}));
            let mut data = Vec::new();
            for (name, shape, bytes) in &self.tensors {
                let start = data.len() as u64;
                data.extend_from_slice(bytes);
                let end = data.len() as u64;
                header.insert(name.clone(), json!({"dtype": "F32", "shape": shape, "data_offsets": [start, end]}));
            }
            let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
            let mut out = Vec::new();
            out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
            out.extend_from_slice(&header_bytes);
            out.extend_from_slice(&data);
            fs::write(self.dir.0.join("model.safetensors"), out).unwrap();

            self.dir
        }
    }

    #[test]
    fn step_produces_correct_length_logits() {
        let fixture = TinyFixture::new("step_len").build();
        let model = Model::load(&fixture.0, 32, 32).unwrap();
        let shards = Shards::open(&fixture.0).unwrap();

        let mut caches = ExpertCaches::new(&model, 8);
        let mut kv = KvState::new(&model);
        let logits = step(&model, &shards, &mut caches, &mut kv, &[0, 1, 2], 0).unwrap();
        assert_eq!(logits.len(), 16); // vocab

        // a decode-style follow-up step must also succeed.
        let logits2 = step(&model, &shards, &mut caches, &mut kv, &[3], 3).unwrap();
        assert_eq!(logits2.len(), 16);
    }

    #[test]
    fn step_and_step_all_agree_on_the_last_position() {
        let fixture = TinyFixture::new("step_consistency").build();
        let model = Model::load(&fixture.0, 32, 32).unwrap();
        let shards = Shards::open(&fixture.0).unwrap();
        let ids = vec![1usize, 2, 3];

        let mut caches_a = ExpertCaches::new(&model, 8);
        let mut kv_a = KvState::new(&model);
        let last_only = step(&model, &shards, &mut caches_a, &mut kv_a, &ids, 0).unwrap();

        let mut caches_b = ExpertCaches::new(&model, 8);
        let mut kv_b = KvState::new(&model);
        let all = step_all(&model, &shards, &mut caches_b, &mut kv_b, &ids, 0).unwrap();
        let v = model.cfg.base.vocab as usize;
        let last_row = &all[(ids.len() - 1) * v..ids.len() * v];

        assert_eq!(last_only, last_row);
    }

    #[test]
    fn layers_forward_matches_manual_per_layer_composition() {
        let fixture = TinyFixture::new("layers_forward").build();
        let model = Model::load(&fixture.0, 32, 32).unwrap();
        let shards = Shards::open(&fixture.0).unwrap();
        let s = 3;
        let d = model.cfg.base.hidden as usize;
        let mut seed = 44u32;
        let x0 = random_vec(s * d, &mut seed);

        let mut caches_a = ExpertCaches::new(&model, 8);
        let mut kv_a = KvState::new(&model);
        let mut x_lf = x0.clone();
        layers_forward(&model, &shards, &mut caches_a, &mut x_lf, s, 0, &mut kv_a).unwrap();

        // independent manual reference: same primitives, re-derived by hand (including the
        // model-level Attention-Residuals output pooling `layers_forward` applies internally).
        let mut caches_b = ExpertCaches::new(&model, 8);
        let mut kv_b = KvState::new(&model);
        let mut x_manual = x0.clone();
        let cfg = &model.cfg;
        let mut attn_res_state = if cfg.attn_res_block > 0 { Some(AttnResState::new()) } else { None };
        for li in 0..2 {
            layer_forward(cfg, &model, li, &shards, &mut caches_b, &mut x_manual, s, 0, &mut kv_b.layers[li], &mut attn_res_state, None).unwrap();
        }
        if let (Some(ar), Some(state)) = (&model.output_attn_res, &attn_res_state) {
            let pooled = attn_res::output_pool(state, &x_manual, d, ar, cfg.base.eps);
            x_manual.copy_from_slice(&pooled);
        }

        for (a, b) in x_lf.iter().zip(&x_manual) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn kda_layer_state_carries_across_sequential_single_token_steps() {
        // Decoding one token at a time must match one prefill call bit-for-bit -- proves
        // KdaLayerState is threaded/accumulating correctly AND that Attention Residuals (purely
        // transient, recreated every call) doesn't interfere with that cross-call KDA state.
        let fixture = TinyFixture::new("kda_state_carries").build();
        let model = Model::load(&fixture.0, 32, 32).unwrap();
        let shards = Shards::open(&fixture.0).unwrap();
        let ids = vec![1usize, 2, 3];

        let mut caches_prefill = ExpertCaches::new(&model, 8);
        let mut kv_prefill = KvState::new(&model);
        let prefill_logits = step(&model, &shards, &mut caches_prefill, &mut kv_prefill, &ids, 0).unwrap();

        let mut caches_seq = ExpertCaches::new(&model, 8);
        let mut kv_seq = KvState::new(&model);
        let mut logits = Vec::new();
        for (pos, &id) in ids.iter().enumerate() {
            logits = step(&model, &shards, &mut caches_seq, &mut kv_seq, &[id], pos).unwrap();
        }

        assert_eq!(prefill_logits, logits, "sequential single-token decode must match one prefill call bit-for-bit");
    }

    #[test]
    fn step_profiled_matches_step_and_reports_nonzero_attention_time() {
        let fixture = TinyFixture::new("step_profiled").build();
        let model = Model::load(&fixture.0, 32, 32).unwrap();
        let shards = Shards::open(&fixture.0).unwrap();
        let ids = vec![1usize, 2, 3];

        let mut caches_a = ExpertCaches::new(&model, 8);
        let mut kv_a = KvState::new(&model);
        let plain = step(&model, &shards, &mut caches_a, &mut kv_a, &ids, 0).unwrap();

        let mut caches_b = ExpertCaches::new(&model, 8);
        let mut kv_b = KvState::new(&model);
        let (profiled, profile) = step_profiled(&model, &shards, &mut caches_b, &mut kv_b, &ids, 0).unwrap();

        assert_eq!(plain, profiled, "step_profiled must compute the exact same logits as step");
        assert!(profile.phases.attention_s > 0.0, "2 attention layers (1 KDA, 1 MLA) ran; attention_s must be nonzero");
        assert!(profile.phases.expert_matmul_s >= 0.0);
        assert!(profile.phases.expert_wait_s >= 0.0);
        assert!(profile.lm_head_s >= 0.0);
    }
}
