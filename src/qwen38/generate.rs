//! Qwen 3.8's forward pass: the 92-layer loop, its hybrid per-layer state, and the routed-expert
//! streaming that keeps 1.26 TB of experts on disk.
//!
//! Every op here was unit-tested on its own first (`attention.rs`, `gdn.rs`, `moe.rs`, `ops.rs`) —
//! this module only wires them together, in the order `Qwen3_5MoeDecoderLayer.forward` does:
//!
//! ```text
//! x += mixer(rmsnorm_1p(x, input_layernorm))          # GDN on 69 layers, GQA on 23
//! x += moe(rmsnorm_1p(x, post_attention_layernorm))   # every layer: 10 routed + 1 shared expert
//! ```
//!
//! **Two kinds of per-layer state.** A GQA layer keeps an ordinary growing KV cache (8 KB/token);
//! a GDN layer keeps FIXED-size recurrent state — 128 heads x a 128x128 matrix (8 MB) plus the
//! conv's `kernel - 1` sample FIFO — which never grows with context length. That asymmetry is the
//! architecture's whole point and the reason a 262 k context costs so little here: only 23 layers
//! accumulate anything at all.
//!
//! **Expert streaming is reused wholesale, not reimplemented.** `expert_cache.rs`'s
//! `begin_loading`/`finish_loading_streaming` (`io_uring` batching, per-expert early drain) and
//! `glm52::moe::apply_single_expert` (the gate/up/down matmuls, including the fused-gate_up form)
//! are architecture-agnostic — they take a `Routing` and a GLM-shaped `Cfg` carrier, which
//! `qwen38::model::to_glm_cfg_expert` provides. What is Qwen-specific and lives here: the softmax
//! router (`qwen38::moe::route`), the shared expert's sigmoid gate, and the overlap of the first
//! chunk's disk read with the shared expert's compute (the same trick `glm52::moe::moe` documents:
//! the shared expert is always active, so its matmuls don't depend on which routed experts land).
//!
//! Prefill runs token by token rather than as one batched GEMM: the GDN recurrence is inherently
//! sequential, and `attention::attend` expects its own position to be the newest in the cache. Same
//! shape as `kimi_linear::generate`'s KDA path.

use crate::expert_cache::ExpertCache;
use crate::generate::{Phases, StepProfile};
use crate::glm52::moe::{Activation, Routing, apply_single_expert, unique_experts};
use crate::kernels::{matmul, matmul_qt};
use crate::kimi_linear::kda::KdaState;
use crate::kimi_linear::ops::swish;
use crate::kimi_linear::short_conv::ShortConvState;
use crate::qwen38::attention::{self, apply_output_gate, norm_heads, rope_half, split_query_gate};
use crate::qwen38::config::Cfg;
use crate::qwen38::gdn;
use crate::qwen38::model::{AttnWeights, GdnWeights, Mixer, Model, ModelError, MoeWeights, expert_naming, to_glm_cfg_expert};
use crate::qwen38::moe::{route, shared_expert_scale};
use crate::qwen38::ops::rmsnorm_1p;
use crate::safetensors::Shards;

/// One Gated DeltaNet layer's recurrent state: one [`KdaState`] per V head plus the shared causal
/// conv FIFO. Fixed size — independent of how many tokens have been seen.
pub struct GdnLayerState {
    /// `n_v_heads` states, each `[head_k_dim, head_v_dim]`.
    pub heads: Vec<KdaState>,
    /// One FIFO over all `conv_dim` channels (Qwen convolves q|k|v together, unlike Kimi Linear's
    /// three separate `ShortConv`s).
    pub conv: ShortConvState,
}

impl GdnLayerState {
    pub fn new(cfg: &Cfg) -> GdnLayerState {
        let n_v_heads = cfg.lin_value_heads as usize;
        let (k_dim, v_dim) = (cfg.lin_key_head_dim as usize, cfg.lin_value_head_dim as usize);
        let key_dim = (cfg.lin_key_heads * cfg.lin_key_head_dim) as usize;
        let value_dim = n_v_heads * v_dim;
        GdnLayerState {
            heads: (0..n_v_heads).map(|_| KdaState::new(k_dim, v_dim)).collect(),
            conv: ShortConvState::new(2 * key_dim + value_dim, cfg.conv_kernel as usize),
        }
    }
}

pub enum LayerState {
    Gdn(GdnLayerState),
    Attn(attention::KvCache),
}

pub struct KvState {
    layers: Vec<LayerState>,
}

impl KvState {
    pub fn new(model: &Model) -> KvState {
        let cfg = &model.cfg;
        let layers = model
            .layers
            .iter()
            .map(|l| match &l.mixer {
                Mixer::Gdn(_) => LayerState::Gdn(GdnLayerState::new(cfg)),
                Mixer::Attn(_) => LayerState::Attn(attention::KvCache::new(cfg.n_kv_heads as usize, cfg.head_dim as usize)),
            })
            .collect();
        KvState { layers }
    }

    /// Read-only view of every layer's state — `kv_session.rs`'s save path (not yet written for this
    /// family) and this module's own tests.
    #[allow(dead_code)]
    pub(crate) fn layers(&self) -> &[LayerState] {
        &self.layers
    }

    /// Reconstructs a `KvState` from raw saved per-layer states — `kv_session.rs`'s load path, not
    /// yet written for this family (the mirror of `new`, which starts every layer empty).
    #[allow(dead_code)]
    pub(crate) fn from_raw(layers: Vec<LayerState>) -> KvState {
        KvState { layers }
    }
}

/// One [`ExpertCache`] per layer — every Qwen 3.8 layer is a routed-MoE layer (there is no
/// `first_k_dense_replace`-style dense prefix like GLM's or Kimi's), so unlike the sibling families'
/// versions this holds no `Option`s.
pub struct ExpertCaches(Vec<ExpertCache>);

impl ExpertCaches {
    pub fn new(model: &Model, capacity: usize) -> ExpertCaches {
        Self::new_with_io_batch_mmap(model, capacity, capacity, false)
    }

    pub fn new_with_io_batch(model: &Model, capacity: usize, io_batch_size: usize) -> ExpertCaches {
        Self::new_with_io_batch_mmap(model, capacity, io_batch_size, false)
    }

    /// Full constructor: `io_batch_size` decoupled from `capacity` (see
    /// `ExpertCache::io_batch_size`'s doc) and the opt-in `--mmap-experts` path, which for this
    /// family is live rather than a no-op — Qwen 3.8's real checkpoint IS MXFP4, the only format
    /// that mmap branch handles.
    pub fn new_with_io_batch_mmap(model: &Model, capacity: usize, io_batch_size: usize, mmap_experts: bool) -> ExpertCaches {
        let naming = expert_naming(&model.cfg);
        let v = model
            .layers
            .iter()
            .map(|_| ExpertCache::for_family_with_io_batch_mmap(capacity, io_batch_size, naming, mmap_experts))
            .collect();
        ExpertCaches(v)
    }

    pub fn hit_miss_totals(&self) -> (u64, u64, u64) {
        self.0.iter().fold((0, 0, 0), |(h, m, n), c| (h + c.hits, m + c.misses, n + c.load_nanos))
    }

    pub fn io_wait_nanos_total(&self) -> u64 {
        self.0.iter().map(|c| c.io_wait_nanos).sum()
    }

    /// Seeds every layer's cache from the checkpoint directory's persisted `.rabbit_usage`
    /// histogram — same mechanism, and the same architecture-agnostic `usage_cache` module, as
    /// `crate::generate::ExpertCaches::warm_start`.
    pub fn warm_start(&mut self, model_dir: &std::path::Path, cache_capacity: usize) -> crate::generate::WarmStartStats {
        let loaded = crate::usage_cache::load(&crate::usage_cache::usage_path(model_dir));
        let hist: u64 = loaded.values().sum();

        let mut per_layer: std::collections::HashMap<usize, Vec<(usize, u64)>> = std::collections::HashMap::new();
        for (&(li, eid), &count) in &loaded {
            per_layer.entry(li).or_default().push((eid, count));
        }
        for (li, cache) in self.0.iter_mut().enumerate() {
            if let Some(entries) = per_layer.remove(&li) {
                cache.seed_usage(entries.into_iter());
            }
        }

        if hist < crate::usage_cache::HIST_THRESHOLD {
            return crate::generate::WarmStartStats { hist, confidence: crate::usage_cache::confidence(hist), pin_candidates: 0 };
        }

        let confidence = crate::usage_cache::confidence(hist);
        let budget = crate::usage_cache::pin_budget(cache_capacity, confidence);
        let mut pin_candidates = 0;
        for cache in self.0.iter_mut() {
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
        let entries = self
            .0
            .iter()
            .enumerate()
            .flat_map(|(li, cache)| cache.usage_counts().map(move |(eid, count)| (li, eid, count)));
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

/// One GDN layer, one token: the four input projections, the shared causal conv, then
/// `gdn::step_token` (SiLU + split + per-head recurrence + gated output norm), then `out_proj`.
fn gdn_step(cfg: &Cfg, w: &GdnWeights, state: &mut GdnLayerState, x: &[f32], out: &mut [f32]) {
    let n_v_heads = cfg.lin_value_heads as usize;
    let key_dim = (cfg.lin_key_heads * cfg.lin_key_head_dim) as usize;
    let value_dim = n_v_heads * cfg.lin_value_head_dim as usize;
    let conv_dim = 2 * key_dim + value_dim;

    let mut qkv_pre = vec![0f32; conv_dim];
    matmul_qt(&mut qkv_pre, x, &w.in_proj_qkv, 1);
    let mut qkv = vec![0f32; conv_dim];
    state.conv.step(&qkv_pre, &w.conv, &mut qkv);

    let mut z = vec![0f32; value_dim];
    matmul_qt(&mut z, x, &w.in_proj_z, 1);
    let mut a = vec![0f32; n_v_heads];
    matmul_qt(&mut a, x, &w.in_proj_a, 1);
    let mut b = vec![0f32; n_v_heads];
    matmul_qt(&mut b, x, &w.in_proj_b, 1);

    let mut core = vec![0f32; value_dim];
    gdn::step_token(&mut state.heads, &mut qkv, &z, &a, &b, &w.a_log, &w.dt_bias, &w.norm, cfg.eps, &mut core);

    matmul_qt(out, &core, &w.out_proj, 1);
}

/// One full-attention layer, one token at absolute position `pos`: `q_proj` (query + gate, split per
/// head), `k`/`v` projections, per-head `(1 + w)` RMSNorm on q and k, partial RoPE, append to the KV
/// cache, causal GQA attention over everything cached, the sigmoid output gate, `o_proj`.
fn attn_step(cfg: &Cfg, w: &AttnWeights, kv: &mut attention::KvCache, x: &[f32], pos: usize, out: &mut [f32]) {
    let head_dim = cfg.head_dim as usize;
    let n_heads = cfg.n_heads as usize;
    let kv_width = (cfg.n_kv_heads * cfg.head_dim) as usize;

    let mut qg = vec![0f32; 2 * n_heads * head_dim];
    matmul_qt(&mut qg, x, &w.q_proj, 1);
    let (mut q, gate) = split_query_gate(&qg, head_dim);

    let mut k = vec![0f32; kv_width];
    matmul_qt(&mut k, x, &w.k_proj, 1);
    let mut v = vec![0f32; kv_width];
    matmul_qt(&mut v, x, &w.v_proj, 1);

    norm_heads(&mut q, head_dim, &w.q_norm, cfg.eps);
    norm_heads(&mut k, head_dim, &w.k_norm, cfg.eps);

    let rope_dim = cfg.rope_dim as usize;
    for head in q.chunks_mut(head_dim) {
        rope_half(head, rope_dim, pos, cfg.theta);
    }
    for head in k.chunks_mut(head_dim) {
        rope_half(head, rope_dim, pos, cfg.theta);
    }

    kv.push(&k, &v);
    let mut ctx = vec![0f32; n_heads * head_dim];
    attention::attend(&q, kv, cfg.attn_scale, &mut ctx);
    apply_output_gate(&mut ctx, &gate);

    matmul_qt(out, &ctx, &w.o_proj, 1);
}

/// One layer's MoE block for `s` tokens: softmax routing, streamed routed experts, and the gated
/// shared expert. See this module's doc for what is reused from `glm52::moe` and why.
#[allow(clippy::too_many_arguments)]
fn moe_forward(
    cfg: &Cfg,
    w: &MoeWeights,
    cache: &mut ExpertCache,
    shards: &Shards,
    layer: usize,
    ebits: u8,
    x: &[f32],
    s: usize,
    out: &mut [f32],
) -> Result<(), ModelError> {
    let d = cfg.hidden as usize;
    let n_experts = cfg.n_experts as usize;
    let shared_inter = cfg.shared_inter as usize;
    let glm_cfg = to_glm_cfg_expert(cfg);

    // Router logits stay in f32 (the weights are unquantized — see `MoeWeights::router`).
    let mut logits = vec![0f32; s * n_experts];
    matmul(&mut logits, x, &w.router, s, d, n_experts);
    let routing = Routing {
        choices: (0..s).map(|si| route(&logits[si * n_experts..(si + 1) * n_experts], cfg.topk as usize)).collect(),
    };
    for choices in &routing.choices {
        for &(eid, _) in choices {
            cache.record_selection(eid);
        }
    }
    out[..s * d].fill(0.0);

    let uniq = unique_experts(&routing);
    let chunk_size = cache.io_batch_size().max(1);
    let mut chunks = uniq.chunks(chunk_size);

    // First chunk's reads go in flight BEFORE the shared expert's compute, which is independent of
    // routing — see this module's doc.
    let first_chunk = chunks.next();
    let pending = match first_chunk {
        Some(chunk) => Some(cache.begin_loading(shards, &glm_cfg, layer, chunk, ebits)?),
        None => None,
    };

    // shared expert: SwiGLU at `shared_expert_intermediate_size`, then scaled per token by
    // `sigmoid(shared_expert_gate . x)` — the gate GLM's and Kimi's shared experts don't have.
    let mut sg = vec![0f32; s * shared_inter];
    let mut su = vec![0f32; s * shared_inter];
    matmul_qt(&mut sg, x, &w.sh_gate, s);
    matmul_qt(&mut su, x, &w.sh_up, s);
    // SwiGLU: `silu(gate) * up`, written here because `Activation::apply` is private to
    // `glm52::moe` (only its `Activation` VALUE is exported, for `apply_single_expert`).
    for (g, &u) in sg.iter_mut().zip(&su) {
        *g = swish(*g) * u;
    }
    let mut shared = vec![0f32; s * d];
    matmul_qt(&mut shared, &sg, &w.sh_down, s);
    for si in 0..s {
        let logit: f32 = x[si * d..(si + 1) * d].iter().zip(&w.shared_gate).map(|(&xi, &gi)| xi * gi).sum();
        let scale = shared_expert_scale(logit);
        for v in shared[si * d..(si + 1) * d].iter_mut() {
            *v *= scale;
        }
    }

    if let (Some(chunk), Some(pending)) = (first_chunk, pending) {
        for &eid in chunk {
            if let Some(slot) = cache.get(eid) {
                apply_single_expert(slot, &routing, x, d, cfg.moe_inter as usize, Activation::Silu, out);
            }
        }
        cache.finish_loading_streaming(pending, shards, &glm_cfg, layer, ebits, |slot| {
            apply_single_expert(slot, &routing, x, d, cfg.moe_inter as usize, Activation::Silu, out);
        })?;
    }
    for chunk in chunks {
        let pending = cache.begin_loading(shards, &glm_cfg, layer, chunk, ebits)?;
        for &eid in chunk {
            if let Some(slot) = cache.get(eid) {
                apply_single_expert(slot, &routing, x, d, cfg.moe_inter as usize, Activation::Silu, out);
            }
        }
        cache.finish_loading_streaming(pending, shards, &glm_cfg, layer, ebits, |slot| {
            apply_single_expert(slot, &routing, x, d, cfg.moe_inter as usize, Activation::Silu, out);
        })?;
    }

    // added last, same relative accumulation order as `glm52::moe::moe` keeps
    for (o, &sh) in out[..s * d].iter_mut().zip(&shared) {
        *o += sh;
    }
    Ok(())
}

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
        rmsnorm_1p(&mut nrm[si * d..(si + 1) * d], &layer.in_ln, cfg.eps);
    }

    let mut tmp = vec![0f32; s * d];
    let mixer_t = std::time::Instant::now();
    match (&layer.mixer, layer_state) {
        (Mixer::Gdn(w), LayerState::Gdn(state)) => {
            for si in 0..s {
                gdn_step(cfg, w, state, &nrm[si * d..(si + 1) * d], &mut tmp[si * d..(si + 1) * d]);
            }
        }
        (Mixer::Attn(w), LayerState::Attn(kv)) => {
            for si in 0..s {
                attn_step(cfg, w, kv, &nrm[si * d..(si + 1) * d], pos_base + si, &mut tmp[si * d..(si + 1) * d]);
            }
        }
        _ => unreachable!("Mixer/LayerState mismatch -- KvState::new always pairs them per layer"),
    }
    if let Some(p) = phases.as_deref_mut() {
        p.attention_s += mixer_t.elapsed().as_secs_f32();
    }
    for (xi, &ti) in x.iter_mut().zip(&tmp) {
        *xi += ti;
    }

    for si in 0..s {
        nrm[si * d..(si + 1) * d].copy_from_slice(&x[si * d..(si + 1) * d]);
        rmsnorm_1p(&mut nrm[si * d..(si + 1) * d], &layer.post_ln, cfg.eps);
    }
    let cache = &mut caches.0[li];
    let wait_before = cache.io_wait_nanos;
    let t = std::time::Instant::now();
    moe_forward(cfg, &layer.moe, cache, shards, li, model.ebits, &nrm, s, &mut tmp)?;
    let elapsed = t.elapsed().as_secs_f32();
    if let Some(p) = phases {
        let wait_delta = ((cache.io_wait_nanos - wait_before) as f32 / 1e9).max(0.0);
        p.expert_wait_s += wait_delta;
        p.expert_matmul_s += (elapsed - wait_delta).max(0.0);
    }
    for (xi, &ti) in x.iter_mut().zip(&tmp) {
        *xi += ti;
    }
    Ok(())
}

pub fn layers_forward(
    model: &Model,
    shards: &Shards,
    caches: &mut ExpertCaches,
    x: &mut [f32],
    s: usize,
    pos_base: usize,
    kv: &mut KvState,
) -> Result<(), ModelError> {
    layers_forward_profiled(model, shards, caches, x, s, pos_base, kv, None)
}

#[allow(clippy::too_many_arguments)]
fn layers_forward_profiled(
    model: &Model,
    shards: &Shards,
    caches: &mut ExpertCaches,
    x: &mut [f32],
    s: usize,
    pos_base: usize,
    kv: &mut KvState,
    mut phases: Option<&mut Phases>,
) -> Result<(), ModelError> {
    for li in 0..model.layers.len() {
        layer_forward(&model.cfg, model, li, shards, caches, x, s, pos_base, &mut kv.layers[li], phases.as_deref_mut())?;
    }
    Ok(())
}

/// Forwards `ids` and returns the LAST position's logits — the decode step.
pub fn step(
    model: &Model,
    shards: &Shards,
    caches: &mut ExpertCaches,
    kv: &mut KvState,
    ids: &[usize],
    pos_base: usize,
) -> Result<Vec<f32>, ModelError> {
    Ok(step_profiled(model, shards, caches, kv, ids, pos_base)?.0)
}

/// `step` plus per-phase timings for `chat.rs`'s `/profile` reporting.
pub fn step_profiled(
    model: &Model,
    shards: &Shards,
    caches: &mut ExpertCaches,
    kv: &mut KvState,
    ids: &[usize],
    pos_base: usize,
) -> Result<(Vec<f32>, StepProfile), ModelError> {
    let cfg = &model.cfg;
    let d = cfg.hidden as usize;
    let s = ids.len();
    let mut phases = Phases::default();
    let mut x = embed_tokens(model, ids);
    layers_forward_profiled(model, shards, caches, &mut x, s, pos_base, kv, Some(&mut phases))?;

    let mut last = x[(s - 1) * d..].to_vec();
    rmsnorm_1p(&mut last, &model.final_norm, cfg.eps);
    let t = std::time::Instant::now();
    let mut logits = vec![0f32; cfg.vocab as usize];
    matmul_qt(&mut logits, &last, &model.lm_head, 1);
    let lm_head_s = t.elapsed().as_secs_f32();
    Ok((logits, StepProfile { phases, lm_head_s }))
}

/// Every position's logits (not just the last) — what teacher-forced evaluation needs.
pub fn step_all(
    model: &Model,
    shards: &Shards,
    caches: &mut ExpertCaches,
    kv: &mut KvState,
    ids: &[usize],
    pos_base: usize,
) -> Result<Vec<f32>, ModelError> {
    let cfg = &model.cfg;
    let (d, vocab) = (cfg.hidden as usize, cfg.vocab as usize);
    let s = ids.len();
    let mut x = embed_tokens(model, ids);
    layers_forward(model, shards, caches, &mut x, s, pos_base, kv)?;

    let mut out = vec![0f32; s * vocab];
    for si in 0..s {
        let mut row = x[si * d..(si + 1) * d].to_vec();
        rmsnorm_1p(&mut row, &model.final_norm, cfg.eps);
        matmul_qt(&mut out[si * vocab..(si + 1) * vocab], &row, &model.lm_head, 1);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen38::model::tests::{TempDir, write_tiny_checkpoint};

    fn setup(name: &str) -> (TempDir, Model, Shards) {
        let dir = TempDir::new(name);
        write_tiny_checkpoint(&dir.0);
        // dbits 16 keeps the fixture's exact constants; ebits 4 exercises the real expert path
        let model = Model::load(&dir.0, 16, 4).expect("fixture must load");
        let shards = Shards::open(&dir.0).expect("fixture shards must open");
        (dir, model, shards)
    }

    #[test]
    fn a_forward_pass_produces_finite_logits_for_every_token() {
        let (_dir, model, shards) = setup("rabbit_test_qwen38_gen_forward");
        let mut caches = ExpertCaches::new(&model, 4);
        let mut kv = KvState::new(&model);

        let logits = step(&model, &shards, &mut caches, &mut kv, &[1, 2, 3], 0).unwrap();
        assert_eq!(logits.len(), model.cfg.vocab as usize);
        assert!(logits.iter().all(|x| x.is_finite()), "logits must be finite: {logits:?}");
        assert!(logits.iter().any(|&x| x != 0.0), "and not all zero");
    }

    /// Every layer's state must be the kind its mixer needs, and only the attention layers may grow
    /// with context: after 3 tokens the KV cache holds 3 positions while GDN state stays fixed.
    #[test]
    fn only_attention_layers_grow_with_context() {
        let (_dir, model, shards) = setup("rabbit_test_qwen38_gen_state");
        let mut caches = ExpertCaches::new(&model, 4);
        let mut kv = KvState::new(&model);

        for (li, st) in kv.layers().iter().enumerate() {
            match st {
                LayerState::Gdn(_) => assert!(model.layers[li].mixer.is_gdn(), "layer {li}"),
                LayerState::Attn(c) => {
                    assert!(!model.layers[li].mixer.is_gdn(), "layer {li}");
                    assert!(c.is_empty(), "layer {li} starts empty");
                }
            }
        }

        step(&model, &shards, &mut caches, &mut kv, &[1, 2, 3], 0).unwrap();
        for st in kv.layers() {
            if let LayerState::Attn(c) = st {
                assert_eq!(c.len(), 3, "3 tokens forwarded -> 3 cached positions");
            }
        }
        step(&model, &shards, &mut caches, &mut kv, &[4], 3).unwrap();
        for st in kv.layers() {
            if let LayerState::Attn(c) = st {
                assert_eq!(c.len(), 4);
            }
        }
    }

    /// Prefilling `s` tokens in one call must equal stepping them one at a time — the property that
    /// ties the batched path to the incremental one, and the one that breaks if `pos_base` isn't
    /// threaded into RoPE per token or if the GDN recurrence gets fed out of order.
    #[test]
    fn batched_prefill_matches_token_by_token_stepping() {
        let (_dir, model, shards) = setup("rabbit_test_qwen38_gen_prefill");
        let ids = [1usize, 5, 2, 9];

        let mut caches_a = ExpertCaches::new(&model, 4);
        let mut kv_a = KvState::new(&model);
        let batched = step(&model, &shards, &mut caches_a, &mut kv_a, &ids, 0).unwrap();

        let mut caches_b = ExpertCaches::new(&model, 4);
        let mut kv_b = KvState::new(&model);
        let mut one_at_a_time = Vec::new();
        for (i, &tok) in ids.iter().enumerate() {
            one_at_a_time = step(&model, &shards, &mut caches_b, &mut kv_b, &[tok], i).unwrap();
        }

        for (i, (a, b)) in batched.iter().zip(&one_at_a_time).enumerate() {
            assert!((a - b).abs() < 1e-3, "logit {i}: batched {a} vs incremental {b}");
        }
    }

    /// `step_all` must return each position's own logits, and its last row must match `step`'s single
    /// row for the same input.
    #[test]
    fn step_all_returns_per_position_logits_ending_in_steps_own() {
        let (_dir, model, shards) = setup("rabbit_test_qwen38_gen_all");
        let vocab = model.cfg.vocab as usize;
        let ids = [3usize, 1, 4];

        let mut caches = ExpertCaches::new(&model, 4);
        let mut kv = KvState::new(&model);
        let all = step_all(&model, &shards, &mut caches, &mut kv, &ids, 0).unwrap();
        assert_eq!(all.len(), ids.len() * vocab);

        let mut caches2 = ExpertCaches::new(&model, 4);
        let mut kv2 = KvState::new(&model);
        let last = step(&model, &shards, &mut caches2, &mut kv2, &ids, 0).unwrap();
        for (i, (a, b)) in all[(ids.len() - 1) * vocab..].iter().zip(&last).enumerate() {
            assert!((a - b).abs() < 1e-4, "logit {i}: {a} vs {b}");
        }
    }

    /// The recurrent state must actually influence later tokens: the same token forwarded twice in a
    /// row cannot produce the same logits.
    #[test]
    fn repeating_a_token_gives_different_logits_the_second_time() {
        let (_dir, model, shards) = setup("rabbit_test_qwen38_gen_recurrence");
        let mut caches = ExpertCaches::new(&model, 4);
        let mut kv = KvState::new(&model);

        let first = step(&model, &shards, &mut caches, &mut kv, &[2], 0).unwrap();
        let second = step(&model, &shards, &mut caches, &mut kv, &[2], 1).unwrap();
        assert!(
            first.iter().zip(&second).any(|(a, b)| (a - b).abs() > 1e-6),
            "state (GDN recurrence + KV cache) must change the second forward"
        );
    }

    /// Routed experts really do stream through the cache: a forward pass must register selections and
    /// misses, and a second pass over the same tokens must hit what the first one left resident.
    #[test]
    fn routed_experts_stream_through_the_expert_cache() {
        let (_dir, model, shards) = setup("rabbit_test_qwen38_gen_streaming");
        let n_experts = model.cfg.n_experts as usize;
        let mut caches = ExpertCaches::new(&model, n_experts); // room for every expert
        let mut kv = KvState::new(&model);

        step(&model, &shards, &mut caches, &mut kv, &[1], 0).unwrap();
        let (hits, misses, _) = caches.hit_miss_totals();
        assert!(misses > 0, "the first token must fault experts in from disk");

        step(&model, &shards, &mut caches, &mut kv, &[1], 1).unwrap();
        let (hits2, _, _) = caches.hit_miss_totals();
        assert!(hits2 > hits, "the second pass must hit the now-resident experts");
    }
}
