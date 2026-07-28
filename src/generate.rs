//! Port of `layer_forward`/`layers_forward`/`kv_alloc`/`step`/`step_all` + sampling
//! (`argmax_v`/`dist_build`/`dist_sample`/`pick_tok`) — the autoregressive generation loop.
//!
//! `kv_alloc`'s fixed-capacity preallocation (`max_t`) isn't ported: `KvCache`/`DsaCache`
//! (from `attention.rs`) already grow on demand via `Vec`, so there's nothing to preallocate
//! — `KvState::new` just creates the empty per-layer caches.
//!
//! Out of scope for this whole port stage (per `rabbit-plan.md`'s "Fuera de alcance"): MTP /
//! speculative decoding (`mtp_*`, `ngram_draft`, `spec_decode`'s draft machinery) and the
//! `PILOT`/`SPEC`/`LOOKA` prefetch-and-research instrumentation. What's left of `spec_decode`
//! once you delete all of that is exactly a plain greedy/sampling loop — that's `generate()`
//! here.

use crate::expert_cache::{ExpertCache, ExpertNaming};
use crate::glm52::attention::{self, Absorb, Dsa, DsaCache, KvCache, OutputGate, QProj, Rope, Selection};
use crate::glm52::config::Cfg;
use crate::glm52::model::{Ffn, Model, ModelError};
use crate::glm52::moe;
use crate::kernels::matmul_qt;
use crate::safetensors::Shards;
use crate::usage_cache;
use std::collections::HashMap;
use std::path::Path;

/// Per-layer KV state for one generation session: compressed MLA latents (`KvCache`) for
/// every layer, plus the DSA indexer's key cache (`DsaCache`) for layers that have one.
pub struct KvState {
    layers: Vec<KvCache>,
    dsa: Vec<Option<DsaCache>>,
}

impl KvState {
    pub fn new(model: &Model) -> KvState {
        let kv_lora = model.cfg.kv_lora as usize;
        let qk_rope = model.cfg.qk_rope as usize;
        let index_hd = model.cfg.index_hd as usize;
        let layers = model.layers.iter().map(|_| KvCache::new(kv_lora, qk_rope)).collect();
        let dsa = model.layers.iter().map(|l| l.dsa.as_ref().map(|_| DsaCache::new(index_hd))).collect();
        KvState { layers, dsa }
    }

    pub(crate) fn layers(&self) -> &[KvCache] {
        &self.layers
    }

    pub(crate) fn dsa(&self) -> &[Option<DsaCache>] {
        &self.dsa
    }

    /// Reconstructs a `KvState` from raw saved per-layer caches — `kv_session.rs`'s load path.
    pub(crate) fn from_raw(layers: Vec<KvCache>, dsa: Vec<Option<DsaCache>>) -> KvState {
        KvState { layers, dsa }
    }
}

/// One `ExpertCache` per MoE layer (`None` for dense layers) — owned by the caller so it
/// persists (and keeps warming up) across multiple `generate`/`step` calls.
pub struct ExpertCaches(Vec<Option<ExpertCache>>);

impl ExpertCaches {
    pub fn new(model: &Model, capacity: usize) -> ExpertCaches {
        let naming = if model.fused_gate_up { ExpertNaming::Glm52FusedGateUp } else { ExpertNaming::Glm52 };
        let v = model
            .layers
            .iter()
            .map(|l| if matches!(l.ffn, Ffn::Moe(_)) { Some(ExpertCache::for_family(capacity, naming)) } else { None })
            .collect();
        ExpertCaches(v)
    }

    /// Summed `hits`/`misses`/`load_nanos` across every MoE layer's cache — a coarse signal
    /// for whether generation time is still dominated by disk loads (misses climbing, I/O time
    /// a large share of the step) or has become compute-bound (hits dominating, I/O time small
    /// relative to the step's wall time).
    pub fn hit_miss_totals(&self) -> (u64, u64, u64) {
        self.0.iter().flatten().fold((0, 0, 0), |(h, m, n), c| (h + c.hits, m + c.misses, n + c.load_nanos))
    }

    /// Summed `io_wait_nanos` (pure `io_uring` disk wait, excluding decode/copy work) across
    /// every MoE layer's cache — see `ExpertCache::io_wait_nanos`'s doc and `PERFORMANCE.md`
    /// for why this exists separately from `hit_miss_totals`'s `load_nanos`, which conflates
    /// the two.
    pub fn io_wait_nanos_total(&self) -> u64 {
        self.0.iter().flatten().map(|c| c.io_wait_nanos).sum()
    }

    /// Reads `<model_dir>/.rabbit_usage` (if any), seeds every MoE layer's usage counters from
    /// it, and — if the persisted total selection count (`hist`, summed across ALL layers)
    /// reaches `usage_cache::HIST_THRESHOLD` — marks each layer's top
    /// `usage_cache::pin_budget(cache_capacity, confidence)` historically-hottest experts as
    /// pin CANDIDATES (`ExpertCache::mark_pin_candidates`). Nothing is loaded here: a real
    /// measured comparison against the real checkpoint showed colibrì's eager `pin_load` design
    /// (loading candidates synchronously right here) makes `--prompt`'s one-shot wall-clock
    /// time WORSE, not better — see `expert_cache.rs`'s module doc for the full story. Instead
    /// each candidate gets promoted into the pinned tier lazily, the first time this session
    /// actually loads it. Never returns an error and must never prevent a session from starting
    /// (a missing/corrupt usage file just means nothing gets marked).
    pub fn warm_start(&mut self, model_dir: &Path, cache_capacity: usize) -> WarmStartStats {
        let loaded = usage_cache::load(&usage_cache::usage_path(model_dir));
        let hist: u64 = loaded.values().sum();

        let mut per_layer: HashMap<usize, Vec<(usize, u64)>> = HashMap::new();
        for (&(li, eid), &count) in &loaded {
            per_layer.entry(li).or_default().push((eid, count));
        }
        for (li, cache_opt) in self.0.iter_mut().enumerate() {
            if let (Some(cache), Some(entries)) = (cache_opt, per_layer.remove(&li)) {
                cache.seed_usage(entries.into_iter());
            }
        }

        if hist < usage_cache::HIST_THRESHOLD {
            return WarmStartStats { hist, confidence: usage_cache::confidence(hist), pin_candidates: 0 };
        }

        let confidence = usage_cache::confidence(hist);
        let budget = usage_cache::pin_budget(cache_capacity, confidence);
        let mut pin_candidates = 0;
        for cache_opt in self.0.iter_mut() {
            let Some(cache) = cache_opt else { continue };
            let mut top: Vec<(usize, u64)> = cache.usage_counts().collect();
            top.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            top.truncate(budget);
            pin_candidates += top.len();
            cache.mark_pin_candidates(top.into_iter().map(|(eid, _)| eid));
        }
        WarmStartStats { hist, confidence, pin_candidates }
    }

    /// Writes every MoE layer's current usage counters back to `<model_dir>/.rabbit_usage`,
    /// overwriting the previous snapshot (see `usage_cache::save`'s doc for why a full
    /// overwrite, not an append, is fine here). Call at turn/response boundaries only — never
    /// inside the decode loop.
    pub fn save_usage(&self, model_dir: &Path) -> std::io::Result<()> {
        let path = usage_cache::usage_path(model_dir);
        let entries = self.0.iter().enumerate().flat_map(|(li, c)| {
            c.as_ref().into_iter().flat_map(move |cache| cache.usage_counts().map(move |(eid, count)| (li, eid, count)))
        });
        usage_cache::save(&path, entries)
    }
}

/// Result of `ExpertCaches::warm_start` — how much history existed and how many experts got
/// marked as pin candidates (not yet loaded — see `warm_start`'s doc), for the caller to log a
/// one-line startup summary.
pub struct WarmStartStats {
    pub hist: u64,
    pub confidence: f32,
    pub pin_candidates: usize,
}

fn embed_tokens(model: &Model, ids: &[usize]) -> Vec<f32> {
    let d = model.cfg.hidden as usize;
    let mut x = vec![0f32; ids.len() * d];
    for (si, &tok) in ids.iter().enumerate() {
        x[si * d..(si + 1) * d].copy_from_slice(&model.embed_row(tok));
    }
    x
}

/// Phase-by-phase wall time for one `step`/`step_all` call, accumulated across every layer —
/// feeds the HTTP server's `/profile` dashboard (`chat.rs`'s `TurnProfile`). `expert_wait_s` is
/// the portion of `expert_matmul_s`'s layers' FFN time that was pure `io_uring` stall (see
/// `ExpertCache::io_wait_nanos`'s doc); `expert_matmul_s` is everything else in the FFN dispatch
/// (dense layers' whole time counts here too — there's nothing to wait on there). `lm_head_s`
/// lives on `StepProfile` instead, since the lm_head matmul happens once per step, outside the
/// per-layer loop this struct is threaded through.
#[derive(Default)]
pub struct Phases {
    pub attention_s: f32,
    pub expert_wait_s: f32,
    pub expert_matmul_s: f32,
}

/// A profiled `step_profiled` call's full timing breakdown: [`Phases`] plus the final lm_head
/// matmul, which sits outside the layer loop `Phases` is accumulated through.
pub struct StepProfile {
    pub phases: Phases,
    pub lm_head_s: f32,
}

/// One transformer layer: `x += attention(rmsnorm(x, in_ln))`, then `x += ffn(rmsnorm(x,
/// post_ln))` — dense MLP or MoE depending on `layer.ffn`. `dsa` selects this layer's DSA
/// behavior (already resolved by `layers_forward`, which owns the full-vs-shared threading).
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
    kv_layer: &mut KvCache,
    dsa: Dsa<'_>,
    mut phases: Option<&mut Phases>,
) -> Result<Option<Selection>, ModelError> {
    let d = cfg.hidden as usize;
    let layer = &model.layers[li];

    let mut nrm = vec![0f32; s * d];
    let mut tmp = vec![0f32; s * d];

    for si in 0..s {
        nrm[si * d..(si + 1) * d].copy_from_slice(&x[si * d..(si + 1) * d]);
        attention::rmsnorm(&mut nrm[si * d..(si + 1) * d], &layer.in_ln, cfg.eps);
    }
    let attn_t = std::time::Instant::now();
    let fresh_selection = attention::attention(cfg, &layer.attn, kv_layer, &nrm, s, pos_base, dsa, Absorb::Auto, Rope::Interleaved, QProj::Lora, OutputGate::Off, &mut tmp);
    if let Some(p) = phases.as_deref_mut() {
        p.attention_s += attn_t.elapsed().as_secs_f32();
    }
    for (xi, &ti) in x.iter_mut().zip(&tmp) {
        *xi += ti;
    }

    for si in 0..s {
        nrm[si * d..(si + 1) * d].copy_from_slice(&x[si * d..(si + 1) * d]);
        attention::rmsnorm(&mut nrm[si * d..(si + 1) * d], &layer.post_ln, cfg.eps);
    }
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
            moe::moe(cfg, w, cache, shards, li, model.ebits, &model.route_cfg, &nrm, s, moe::Activation::Silu, &mut tmp)?;
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

    Ok(fresh_selection)
}

/// Runs every layer in order on new tokens `x[S,hidden]` at absolute position `pos_base`,
/// updating `x` in place with each layer's residual output.
///
/// DSA selection (see `attention.rs`'s module doc) is only computed on the very first call
/// for this KV state (`pos_base==0`, matching the C's per-layer `kv_start[layer]==0` gate,
/// which — since every layer is always advanced in lockstep by the same `S` new positions —
/// reduces to a single model-wide condition): a "full" indexer layer computes a fresh
/// `Selection`; a "shared" layer reuses whichever `Selection` the most recent "full" layer
/// produced. Decode calls (`pos_base>0`) always get dense (unrestricted) attention.
#[allow(clippy::too_many_arguments)]
pub fn layers_forward(model: &Model, shards: &Shards, caches: &mut ExpertCaches, x: &mut [f32], s: usize, pos_base: usize, kv: &mut KvState, mut phases: Option<&mut Phases>) -> Result<(), ModelError> {
    let cfg = &model.cfg;
    let is_first_call = pos_base == 0;
    let mut last_selection: Option<Selection> = None;

    for li in 0..model.layers.len() {
        let dsa_mode = if model.has_dsa && is_first_call {
            if let Some(dsa_w) = &model.layers[li].dsa {
                match kv.dsa[li].as_mut() {
                    Some(cache) => Dsa::Compute { weights: dsa_w, cache, force: false },
                    None => Dsa::Off,
                }
            } else if let Some(sel) = &last_selection {
                Dsa::Reuse(sel)
            } else {
                Dsa::Off
            }
        } else {
            Dsa::Off
        };

        let fresh = layer_forward(cfg, model, li, shards, caches, x, s, pos_base, &mut kv.layers[li], dsa_mode, phases.as_deref_mut())?;
        if let Some(sel) = fresh {
            last_selection = Some(sel);
        }
    }
    Ok(())
}

/// Decode/prefill step returning logits for only the LAST new position — the common case
/// (one new token per generation step; a prefix prefill only needs the tail logits to pick
/// the next token).
pub fn step(model: &Model, shards: &Shards, caches: &mut ExpertCaches, kv: &mut KvState, ids: &[usize], pos_base: usize) -> Result<Vec<f32>, ModelError> {
    let s = ids.len();
    let d = model.cfg.hidden as usize;
    let mut x = embed_tokens(model, ids);
    layers_forward(model, shards, caches, &mut x, s, pos_base, kv, None)?;

    let mut last = x[(s - 1) * d..s * d].to_vec();
    attention::rmsnorm(&mut last, &model.final_norm, model.cfg.eps);
    let mut logit = vec![0f32; model.cfg.vocab as usize];
    matmul_qt(&mut logit, &last, &model.lm_head, 1);
    Ok(logit)
}

/// Like `step`, but also returns a [`StepProfile`] breaking down where this one forward pass
/// spent its wall time: attention, expert wait (pure `io_uring` stall), expert matmul, and the
/// final lm_head matmul. The only forward-pass entry point that times anything — used solely by
/// the HTTP server's `/profile` dashboard (via `chat.rs`'s `generate_reply`); every other caller
/// keeps using the untimed `step`/`step_all` above.
pub fn step_profiled(model: &Model, shards: &Shards, caches: &mut ExpertCaches, kv: &mut KvState, ids: &[usize], pos_base: usize) -> Result<(Vec<f32>, StepProfile), ModelError> {
    let s = ids.len();
    let d = model.cfg.hidden as usize;
    let mut x = embed_tokens(model, ids);
    let mut phases = Phases::default();
    layers_forward(model, shards, caches, &mut x, s, pos_base, kv, Some(&mut phases))?;

    let mut last = x[(s - 1) * d..s * d].to_vec();
    attention::rmsnorm(&mut last, &model.final_norm, model.cfg.eps);
    let mut logit = vec![0f32; model.cfg.vocab as usize];
    let t = std::time::Instant::now();
    matmul_qt(&mut logit, &last, &model.lm_head, 1);
    let lm_head_s = t.elapsed().as_secs_f32();
    Ok((logit, StepProfile { phases, lm_head_s }))
}

/// Like `step`, but returns logits for EVERY new position `[S,vocab]` — teacher-forcing
/// validation and (were MTP in scope) draft verification both need the whole range.
pub fn step_all(model: &Model, shards: &Shards, caches: &mut ExpertCaches, kv: &mut KvState, ids: &[usize], pos_base: usize) -> Result<Vec<f32>, ModelError> {
    let s = ids.len();
    let d = model.cfg.hidden as usize;
    let v = model.cfg.vocab as usize;
    let mut x = embed_tokens(model, ids);
    layers_forward(model, shards, caches, &mut x, s, pos_base, kv, None)?;

    let mut lo = vec![0f32; s * v];
    for si in 0..s {
        let mut row = x[si * d..(si + 1) * d].to_vec();
        attention::rmsnorm(&mut row, &model.final_norm, model.cfg.eps);
        matmul_qt(&mut lo[si * v..(si + 1) * v], &row, &model.lm_head, 1);
    }
    Ok(lo)
}

pub fn argmax(logits: &[f32]) -> usize {
    let mut best = 0;
    let mut bv = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best
}

/// `temperature<=0` means greedy (`argmax`); otherwise softmax(logits/temperature) truncated
/// to the top `nucleus` cumulative probability mass (`nucleus<=0` or `>=1` disables nucleus).
pub struct SamplingConfig {
    pub temperature: f32,
    pub nucleus: f32,
}

impl SamplingConfig {
    pub fn greedy() -> SamplingConfig {
        SamplingConfig { temperature: 0.0, nucleus: 0.0 }
    }
}

/// xorshift64 PRNG — same constants and update as `g_rng`/`rndu` in the C, so sampling draws
/// reproduce bit-for-bit given the same seed.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 * (1.0 / 9007199254740992.0)
    }
}

impl Default for Rng {
    fn default() -> Rng {
        Rng(0x9E3779B97F4A7C15)
    }
}

/// Builds `softmax(logits/temperature)`, truncated to nucleus top-p and renormalized.
pub fn build_distribution(logits: &[f32], sampling: &SamplingConfig) -> Vec<f32> {
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let invt = 1.0 / sampling.temperature.max(1e-4);
    let mut p: Vec<f32> = logits.iter().map(|&l| ((l - mx) * invt).exp()).collect();
    let sum: f32 = p.iter().sum();
    for v in p.iter_mut() {
        *v /= sum;
    }

    if sampling.nucleus > 0.0 && sampling.nucleus < 1.0 {
        let mut order: Vec<usize> = (0..p.len()).collect();
        order.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap());
        let mut cum = 0f64;
        let mut keep = p.len();
        for (i, &ix) in order.iter().enumerate() {
            cum += p[ix] as f64;
            if cum >= sampling.nucleus as f64 {
                keep = i + 1;
                break;
            }
        }
        for &ix in &order[keep..] {
            p[ix] = 0.0;
        }
        let s2: f32 = order[..keep].iter().map(|&ix| p[ix]).sum();
        for &ix in &order[..keep] {
            p[ix] /= s2;
        }
    }
    p
}

/// Samples from a distribution built by `build_distribution`. `ban` (if given) is excluded
/// and the remaining mass renormalized on the fly — used by speculative-decoding rejection
/// sampling in the original; unused by `generate()` here since MTP is out of scope, but kept
/// as a parameter since it's intrinsic to `dist_sample`'s own contract, not an MTP add-on.
pub fn sample(dist: &[f32], rng: &mut Rng, ban: Option<usize>) -> usize {
    let banned_mass = ban.map(|b| dist[b] as f64).unwrap_or(0.0);
    let z = (1.0 - banned_mass).max(1e-12);
    let u = rng.next_f64() * z;
    let mut cum = 0f64;
    for (i, &p) in dist.iter().enumerate() {
        if Some(i) == ban {
            continue;
        }
        cum += p as f64;
        if cum >= u {
            return i;
        }
    }
    for i in (0..dist.len()).rev() {
        if Some(i) != ban && dist[i] > 0.0 {
            return i;
        }
    }
    0
}

pub fn pick_token(logits: &[f32], sampling: &SamplingConfig, rng: &mut Rng, ban: Option<usize>) -> usize {
    if sampling.temperature <= 0.0 {
        argmax(logits)
    } else {
        let dist = build_distribution(logits, sampling);
        sample(&dist, rng, ban)
    }
}

/// Greedy/sampling autoregressive generation: forwards `prompt`, then repeatedly picks and
/// forwards one token at a time. Returns only the newly generated tokens (not the prompt).
#[allow(clippy::too_many_arguments)]
pub fn generate(
    model: &Model,
    shards: &Shards,
    caches: &mut ExpertCaches,
    kv: &mut KvState,
    prompt: &[usize],
    n_new: usize,
    sampling: &SamplingConfig,
    rng: &mut Rng,
    stop_ids: &[usize],
) -> Result<Vec<usize>, ModelError> {
    let mut out = Vec::with_capacity(n_new);
    let mut logits = step(model, shards, caches, kv, prompt, 0)?;
    let mut pos = prompt.len();

    while out.len() < n_new {
        let next = pick_token(&logits, sampling, rng, None);
        if stop_ids.contains(&next) {
            break;
        }
        out.push(next);
        if out.len() >= n_new {
            break;
        }
        logits = step(model, shards, caches, kv, &[next], pos)?;
        pos += 1;
    }
    Ok(out)
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

    // ---- sampling primitives: no model needed ----

    #[test]
    fn argmax_finds_the_maximum() {
        assert_eq!(argmax(&[1.0, 5.0, 3.0, -2.0]), 1);
        assert_eq!(argmax(&[9.0]), 0);
    }

    #[test]
    fn pick_token_is_greedy_argmax_at_zero_temperature() {
        let logits = vec![0.1, 0.9, -0.3, 0.5];
        let sampling = SamplingConfig::greedy();
        let mut rng = Rng::new(12345);
        for _ in 0..5 {
            // greedy never touches the RNG, so repeated calls must all agree with argmax.
            assert_eq!(pick_token(&logits, &sampling, &mut rng, None), argmax(&logits));
        }
    }

    #[test]
    fn build_distribution_sums_to_one_and_favors_larger_logits() {
        let logits = vec![1.0, 2.0, 3.0];
        let sampling = SamplingConfig { temperature: 1.0, nucleus: 0.0 };
        let p = build_distribution(&logits, &sampling);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(p[0] < p[1] && p[1] < p[2]);
    }

    #[test]
    fn build_distribution_nucleus_zeroes_out_the_low_probability_tail() {
        // one dominant logit -> top-p=0.5 should keep only that one token.
        let logits = vec![10.0, 0.0, 0.0, 0.0];
        let sampling = SamplingConfig { temperature: 1.0, nucleus: 0.5 };
        let p = build_distribution(&logits, &sampling);
        assert!((p[0] - 1.0).abs() < 1e-4);
        assert_eq!(p[1], 0.0);
        assert_eq!(p[2], 0.0);
        assert_eq!(p[3], 0.0);
    }

    #[test]
    fn sample_excludes_the_banned_token() {
        let dist = vec![0.5, 0.5];
        let mut rng = Rng::new(1);
        for _ in 0..20 {
            assert_eq!(sample(&dist, &mut rng, Some(0)), 1);
        }
    }

    #[test]
    fn rng_is_deterministic_from_seed_and_stays_in_unit_range() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..10 {
            let (va, vb) = (a.next_f64(), b.next_f64());
            assert_eq!(va, vb);
            assert!((0.0..1.0).contains(&va));
        }
        let mut c = Rng::new(43);
        assert_ne!(a.next_f64(), c.next_f64());
    }

    // ---- full-model fixture: 3 layers (dense, MoE+DSA-full, MoE+DSA-shared) ----

    struct Dims {
        hidden: usize,
        heads: usize,
        qk_nope: usize,
        qk_rope: usize,
        v_head: usize,
        q_lora: usize,
        kv_lora: usize,
        vocab: usize,
        dense_inter: usize,
        n_experts: usize,
        topk: usize,
        moe_inter: usize,
        n_shared: usize,
        index_nh: usize,
        index_hd: usize,
        index_topk: usize,
    }

    fn dims() -> Dims {
        Dims {
            hidden: 6,
            heads: 2,
            qk_nope: 2,
            qk_rope: 2,
            v_head: 3,
            q_lora: 4,
            kv_lora: 4,
            vocab: 10,
            dense_inter: 5,
            n_experts: 3,
            topk: 2,
            moe_inter: 4,
            n_shared: 1,
            index_nh: 2,
            index_hd: 2,
            index_topk: 2, // small on purpose: with seq_len=4 the real selection path engages
        }
    }

    /// 3 layers: layer 0 dense, layers 1-2 MoE; DSA indexer weights only on layer 1 ("full"),
    /// layer 2 is "shared" and must reuse layer 1's selection. All F32 (dbits=ebits=32) so
    /// comparisons below can use a tight tolerance without quantization noise.
    fn build_three_layer_fixture(name: &str) -> (TempDir, Dims) {
        let dm = dims();
        let dir = TempDir::new(name);
        let mut seed = 5u32;
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_string(), json!({"format": "rabbit-test"}));
        let mut data = Vec::new();

        let mut add = |header: &mut serde_json::Map<String, serde_json::Value>, name: String, rows: usize, cols: usize| {
            let n = if cols == 0 { rows } else { rows * cols };
            let vals = random_vec(n, &mut seed);
            let shape = if cols == 0 { vec![rows] } else { vec![rows, cols] };
            let bytes = f32_bytes(&vals);
            let start = data.len() as u64;
            data.extend_from_slice(&bytes);
            let end = data.len() as u64;
            header.insert(name, json!({"dtype": "F32", "shape": shape, "data_offsets": [start, end]}));
        };

        let qh = dm.qk_nope + dm.qk_rope;
        add(&mut header, "model.embed_tokens.weight".into(), dm.vocab, dm.hidden);
        add(&mut header, "lm_head.weight".into(), dm.vocab, dm.hidden);
        add(&mut header, "model.norm.weight".into(), dm.hidden, 0);

        for i in 0..3usize {
            let p = |s: &str| format!("model.layers.{i}.{s}");
            add(&mut header, p("input_layernorm.weight"), dm.hidden, 0);
            add(&mut header, p("post_attention_layernorm.weight"), dm.hidden, 0);
            add(&mut header, p("self_attn.q_a_proj.weight"), dm.q_lora, dm.hidden);
            add(&mut header, p("self_attn.q_a_layernorm.weight"), dm.q_lora, 0);
            add(&mut header, p("self_attn.q_b_proj.weight"), dm.heads * qh, dm.q_lora);
            add(&mut header, p("self_attn.kv_a_proj_with_mqa.weight"), dm.kv_lora + dm.qk_rope, dm.hidden);
            add(&mut header, p("self_attn.kv_a_layernorm.weight"), dm.kv_lora, 0);
            add(&mut header, p("self_attn.kv_b_proj.weight"), dm.heads * (dm.qk_nope + dm.v_head), dm.kv_lora);
            add(&mut header, p("self_attn.o_proj.weight"), dm.hidden, dm.heads * dm.v_head);

            if i == 0 {
                add(&mut header, p("mlp.gate_proj.weight"), dm.dense_inter, dm.hidden);
                add(&mut header, p("mlp.up_proj.weight"), dm.dense_inter, dm.hidden);
                add(&mut header, p("mlp.down_proj.weight"), dm.hidden, dm.dense_inter);
            } else {
                add(&mut header, p("mlp.gate.weight"), dm.n_experts, dm.hidden);
                add(&mut header, p("mlp.gate.e_score_correction_bias"), dm.n_experts, 0);
                let s_i = dm.moe_inter * dm.n_shared;
                add(&mut header, p("mlp.shared_experts.gate_proj.weight"), s_i, dm.hidden);
                add(&mut header, p("mlp.shared_experts.up_proj.weight"), s_i, dm.hidden);
                add(&mut header, p("mlp.shared_experts.down_proj.weight"), dm.hidden, s_i);
                for eid in 0..dm.n_experts {
                    add(&mut header, format!("model.layers.{i}.mlp.experts.{eid}.gate_proj.weight"), dm.moe_inter, dm.hidden);
                    add(&mut header, format!("model.layers.{i}.mlp.experts.{eid}.up_proj.weight"), dm.moe_inter, dm.hidden);
                    add(&mut header, format!("model.layers.{i}.mlp.experts.{eid}.down_proj.weight"), dm.hidden, dm.moe_inter);
                }
                if i == 1 {
                    // "full" indexer layer: only this one gets indexer weights.
                    let ip = |s: &str| format!("model.layers.{i}.self_attn.indexer.{s}");
                    add(&mut header, ip("wq_b.weight"), dm.index_nh * dm.index_hd, dm.q_lora);
                    add(&mut header, ip("wk.weight"), dm.index_hd, dm.hidden);
                    add(&mut header, ip("weights_proj.weight"), dm.index_nh, dm.hidden);
                    add(&mut header, ip("k_norm.weight"), dm.index_hd, 0);
                    add(&mut header, ip("k_norm.bias"), dm.index_hd, 0);
                }
            }
        }

        let cfg_json = json!({
            "model_type": "glm_moe_dsa",
            "hidden_size": dm.hidden, "num_hidden_layers": 3, "num_attention_heads": dm.heads,
            "n_routed_experts": dm.n_experts, "num_experts_per_tok": dm.topk,
            "moe_intermediate_size": dm.moe_inter, "intermediate_size": dm.dense_inter,
            "first_k_dense_replace": 1, "q_lora_rank": dm.q_lora, "kv_lora_rank": dm.kv_lora,
            "qk_nope_head_dim": dm.qk_nope, "qk_rope_head_dim": dm.qk_rope, "v_head_dim": dm.v_head,
            "n_shared_experts": dm.n_shared, "vocab_size": dm.vocab, "n_group": 1, "topk_group": 1,
            "index_topk": dm.index_topk, "index_n_heads": dm.index_nh, "index_head_dim": dm.index_hd,
            "indexer_types": ["shared", "full", "shared"],
        });
        fs::write(dir.0.join("config.json"), cfg_json.to_string()).unwrap();

        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_bytes);
        out.extend_from_slice(&data);
        fs::write(dir.0.join("model.safetensors"), out).unwrap();

        (dir, dm)
    }

    #[test]
    fn layers_forward_matches_manual_per_layer_composition_with_dsa_reuse() {
        let (fixture, dm) = build_three_layer_fixture("rabbit_test_generate_layers");
        let model = Model::load(&fixture.0, 32, 32).unwrap();
        assert!(model.has_dsa, "fixture must have a complete DSA indexer for this test to mean anything");
        assert!(model.layers[0].dsa.is_none());
        assert!(model.layers[1].dsa.is_some());
        assert!(model.layers[2].dsa.is_none(), "layer 2 is \"shared\" -> no indexer weights of its own");

        let shards = Shards::open(&fixture.0).unwrap();
        let s = 4; // > index_topk=2, so DSA's real selection path (not the dense no-op) engages
        let mut seed = 77u32;
        let x0 = random_vec(s * dm.hidden, &mut seed);

        // ---- run via layers_forward (the code under test) ----
        let mut caches_a = ExpertCaches::new(&model, 8);
        let mut kv_a = KvState::new(&model);
        let mut x_lf = x0.clone();
        layers_forward(&model, &shards, &mut caches_a, &mut x_lf, s, 0, &mut kv_a, None).unwrap();

        // ---- independent manual reference: same primitives, wiring re-derived by hand ----
        let mut caches_b = ExpertCaches::new(&model, 8);
        let mut kv_b = KvState::new(&model);
        let mut x_manual = x0.clone();
        let d = dm.hidden;
        let mut last_selection: Option<Selection> = None;
        for (li, layer) in model.layers.iter().enumerate() {
            let mut nrm = x_manual.clone();
            for si in 0..s {
                attention::rmsnorm(&mut nrm[si * d..(si + 1) * d], &layer.in_ln, model.cfg.eps);
            }
            let dsa_mode = if let Some(dsa_w) = &layer.dsa {
                Dsa::Compute { weights: dsa_w, cache: kv_b.dsa[li].as_mut().unwrap(), force: false }
            } else if let Some(sel) = &last_selection {
                Dsa::Reuse(sel)
            } else {
                Dsa::Off
            };
            let mut attn_out = vec![0f32; s * d];
            let fresh = attention::attention(&model.cfg, &layer.attn, &mut kv_b.layers[li], &nrm, s, 0, dsa_mode, Absorb::Auto, Rope::Interleaved, QProj::Lora, OutputGate::Off, &mut attn_out);
            if let Some(sel) = fresh {
                last_selection = Some(sel);
            }
            for (xi, &ai) in x_manual.iter_mut().zip(&attn_out) {
                *xi += ai;
            }

            let mut nrm2 = x_manual.clone();
            for si in 0..s {
                attention::rmsnorm(&mut nrm2[si * d..(si + 1) * d], &layer.post_ln, model.cfg.eps);
            }
            let mut ffn_out = vec![0f32; s * d];
            match &layer.ffn {
                Ffn::Dense(w) => moe::dense_mlp(w, &nrm2, s, dm.dense_inter, moe::Activation::Silu, &mut ffn_out),
                Ffn::Moe(w) => {
                    let cache = caches_b.0[li].as_mut().unwrap();
                    moe::moe(&model.cfg, w, cache, &shards, li, model.ebits, &model.route_cfg, &nrm2, s, moe::Activation::Silu, &mut ffn_out).unwrap();
                }
            }
            for (xi, &fi) in x_manual.iter_mut().zip(&ffn_out) {
                *xi += fi;
            }
        }

        for (a, b) in x_lf.iter().zip(&x_manual) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn step_and_step_all_agree_on_the_last_position() {
        let (fixture, dm) = build_three_layer_fixture("rabbit_test_generate_step_consistency");
        let model = Model::load(&fixture.0, 32, 32).unwrap();
        let shards = Shards::open(&fixture.0).unwrap();
        let ids = vec![1usize, 2, 3];

        let mut caches_a = ExpertCaches::new(&model, 8);
        let mut kv_a = KvState::new(&model);
        let last_only = step(&model, &shards, &mut caches_a, &mut kv_a, &ids, 0).unwrap();

        let mut caches_b = ExpertCaches::new(&model, 8);
        let mut kv_b = KvState::new(&model);
        let all = step_all(&model, &shards, &mut caches_b, &mut kv_b, &ids, 0).unwrap();
        let v = dm.vocab;
        let last_row = &all[(ids.len() - 1) * v..ids.len() * v];

        assert_eq!(last_only, last_row);
    }

    #[test]
    fn step_profiled_reports_nonzero_attention_and_expert_matmul_time() {
        let (fixture, _dm) = build_three_layer_fixture("rabbit_test_step_profiled");
        let model = Model::load(&fixture.0, 32, 32).unwrap();
        let shards = Shards::open(&fixture.0).unwrap();
        let ids = vec![1usize, 2, 3];

        let mut caches = ExpertCaches::new(&model, 8);
        let mut kv = KvState::new(&model);
        let (logits, profile) = step_profiled(&model, &shards, &mut caches, &mut kv, &ids, 0).unwrap();

        assert_eq!(logits.len(), model.cfg.vocab as usize);
        assert!(profile.phases.attention_s > 0.0, "3 attention layers ran; attention_s must be nonzero");
        assert!(profile.phases.expert_matmul_s > 0.0, "2 MoE layers ran; expert_matmul_s must be nonzero");
        assert!(profile.phases.expert_wait_s >= 0.0);
        assert!(profile.lm_head_s >= 0.0);
    }

    #[test]
    fn generate_is_deterministic_under_greedy_sampling() {
        let (fixture, _dm) = build_three_layer_fixture("rabbit_test_generate_greedy_det");
        let model = Model::load(&fixture.0, 32, 32).unwrap();
        let shards = Shards::open(&fixture.0).unwrap();
        let prompt = vec![1usize, 2];
        let sampling = SamplingConfig::greedy();

        let mut caches1 = ExpertCaches::new(&model, 8);
        let mut kv1 = KvState::new(&model);
        let mut rng1 = Rng::new(1);
        let out1 = generate(&model, &shards, &mut caches1, &mut kv1, &prompt, 5, &sampling, &mut rng1, &[]).unwrap();

        let mut caches2 = ExpertCaches::new(&model, 8);
        let mut kv2 = KvState::new(&model);
        let mut rng2 = Rng::new(999); // greedy never touches the RNG -> seed must not matter
        let out2 = generate(&model, &shards, &mut caches2, &mut kv2, &prompt, 5, &sampling, &mut rng2, &[]).unwrap();

        assert_eq!(out1.len(), 5);
        assert_eq!(out1, out2);
    }

    #[test]
    fn generate_stops_at_a_stop_id_without_emitting_it() {
        let (fixture, _dm) = build_three_layer_fixture("rabbit_test_generate_stop");
        let model = Model::load(&fixture.0, 32, 32).unwrap();
        let shards = Shards::open(&fixture.0).unwrap();
        let prompt = vec![1usize, 2];
        let sampling = SamplingConfig::greedy();

        let mut caches = ExpertCaches::new(&model, 8);
        let mut kv = KvState::new(&model);
        let mut rng = Rng::new(1);
        let unrestricted = generate(&model, &shards, &mut caches, &mut kv, &prompt, 5, &sampling, &mut rng, &[]).unwrap();
        assert!(!unrestricted.is_empty(), "fixture should deterministically emit at least one token");
        let stop_id = unrestricted[0];

        let mut caches2 = ExpertCaches::new(&model, 8);
        let mut kv2 = KvState::new(&model);
        let mut rng2 = Rng::new(1);
        let stopped = generate(&model, &shards, &mut caches2, &mut kv2, &prompt, 5, &sampling, &mut rng2, &[stop_id]).unwrap();

        assert!(stopped.is_empty(), "must stop before emitting the banned first token");
    }

    // ---- kv_session.rs: save/load round-tripping (reuses the 3-layer dense/MoE+DSA fixture) ----

    use crate::kv_session::LoadError;

    #[test]
    fn kv_session_round_trip_gives_bit_identical_continuation() {
        let (fixture, _dm) = build_three_layer_fixture("rabbit_test_kv_session_roundtrip");
        let model = Model::load(&fixture.0, 32, 32).unwrap();
        let shards = Shards::open(&fixture.0).unwrap();
        let prompt = vec![1usize, 2, 3];
        let next_id = 4usize;

        // continuous path: never touches disk.
        let mut caches_a = ExpertCaches::new(&model, 8);
        let mut kv_a = KvState::new(&model);
        step_all(&model, &shards, &mut caches_a, &mut kv_a, &prompt, 0).unwrap();
        let pos = prompt.len();
        let continuous = step(&model, &shards, &mut caches_a, &mut kv_a, &[next_id], pos).unwrap();

        // save/reload path: build the identical prompt-only state, save it, drop it, cold-load,
        // then continue with the same next token.
        let mut caches_b = ExpertCaches::new(&model, 8);
        let mut kv_b = KvState::new(&model);
        step_all(&model, &shards, &mut caches_b, &mut kv_b, &prompt, 0).unwrap();
        let dir = TempDir::new("rabbit_test_kv_session_roundtrip_file");
        let path = dir.0.join("session.kv");
        kv_b.save(0, pos, &path).unwrap();
        drop(kv_b);

        let (mut kv_c, loaded_pos) = KvState::load(&path, &model).unwrap();
        assert_eq!(loaded_pos, pos);
        let mut caches_c = ExpertCaches::new(&model, 8);
        let reloaded = step(&model, &shards, &mut caches_c, &mut kv_c, &[next_id], loaded_pos).unwrap();

        assert_eq!(continuous, reloaded, "save/load round trip must be bit-identical to uninterrupted continuation");
    }

    #[test]
    fn kv_session_two_partial_saves_equal_one_combined_save() {
        let (fixture, _dm) = build_three_layer_fixture("rabbit_test_kv_session_append");
        let model = Model::load(&fixture.0, 32, 32).unwrap();
        let shards = Shards::open(&fixture.0).unwrap();
        let prompt = vec![1usize, 2, 3, 4, 5];

        // Build the KV state via ONE forward pass -- this test is about kv_session.rs's
        // append format, not about the model's own prefill semantics (splitting the forward
        // pass itself would also split `layers_forward`'s `is_first_call`-gated DSA-indexer
        // computation, which is a real behavior difference, not a persistence bug).
        let mut caches = ExpertCaches::new(&model, 8);
        let mut kv = KvState::new(&model);
        step_all(&model, &shards, &mut caches, &mut kv, &prompt, 0).unwrap();
        let total = prompt.len();

        // path A: one combined save covering the whole prompt.
        let dir_a = TempDir::new("rabbit_test_kv_session_append_a");
        let path_a = dir_a.0.join("session.kv");
        kv.save(0, total, &path_a).unwrap();

        // path B: two partial saves of the SAME state, split mid-conversation.
        let split = 2;
        let dir_b = TempDir::new("rabbit_test_kv_session_append_b");
        let path_b = dir_b.0.join("session.kv");
        kv.save(0, split, &path_b).unwrap();
        kv.save(split, total, &path_b).unwrap();

        let (kv_loaded_a, pos_a) = KvState::load(&path_a, &model).unwrap();
        let (kv_loaded_b, pos_b) = KvState::load(&path_b, &model).unwrap();
        assert_eq!(pos_a, total);
        assert_eq!(pos_a, pos_b);

        for (la, lb) in kv_loaded_a.layers().iter().zip(kv_loaded_b.layers().iter()) {
            assert_eq!(la.l_range(0, pos_a), lb.l_range(0, pos_b));
            assert_eq!(la.r_range(0, pos_a), lb.r_range(0, pos_b));
        }
        for (da, db) in kv_loaded_a.dsa().iter().zip(kv_loaded_b.dsa().iter()) {
            match (da, db) {
                (Some(da), Some(db)) => assert_eq!(da.k_range(0, pos_a), db.k_range(0, pos_b)),
                (None, None) => {}
                _ => panic!("dsa presence disagrees between the two save strategies"),
            }
        }
    }

    #[test]
    fn kv_session_load_errors_on_bad_magic() {
        let (fixture, _dm) = build_three_layer_fixture("rabbit_test_kv_session_bad_magic_model");
        let model = Model::load(&fixture.0, 32, 32).unwrap();

        let dir = TempDir::new("rabbit_test_kv_session_bad_magic_file");
        let path = dir.0.join("session.kv");
        fs::write(&path, b"not a rabbit kv session file at all").unwrap();

        let err = match KvState::load(&path, &model) {
            Err(e) => e,
            Ok(_) => panic!("load must fail on a bad-magic file"),
        };
        assert!(matches!(err, LoadError::BadMagic));
    }

    #[test]
    fn kv_session_load_errors_on_dimension_mismatch() {
        let (fixture, _dm) = build_three_layer_fixture("rabbit_test_kv_session_dim_mismatch_model");
        let model = Model::load(&fixture.0, 32, 32).unwrap();

        // hand-craft a header claiming a different kv_lora than the model actually has,
        // mirroring kv_session.rs's format spec directly -- same approach
        // `build_three_layer_fixture` uses to fabricate a fake safetensors file by hand.
        let dir = TempDir::new("rabbit_test_kv_session_dim_mismatch_file");
        let path = dir.0.join("session.kv");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RBTKVSN1");
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&(model.layers.len() as u32).to_le_bytes());
        for layer in &model.layers {
            let bad_kv_lora = model.cfg.kv_lora as u32 + 1;
            bytes.extend_from_slice(&bad_kv_lora.to_le_bytes());
            bytes.extend_from_slice(&(model.cfg.qk_rope as u32).to_le_bytes());
            if layer.dsa.is_some() {
                bytes.push(1);
                bytes.extend_from_slice(&(model.cfg.index_hd as u32).to_le_bytes());
            } else {
                bytes.push(0);
            }
        }
        fs::write(&path, &bytes).unwrap();

        let err = match KvState::load(&path, &model) {
            Err(e) => e,
            Ok(_) => panic!("load must fail on a dimension mismatch"),
        };
        assert!(matches!(err, LoadError::ConfigMismatch { field: "kv_lora", .. }));
    }

    #[test]
    fn kv_session_load_recovers_from_a_truncated_trailing_record() {
        let (fixture, _dm) = build_three_layer_fixture("rabbit_test_kv_session_truncated_model");
        let model = Model::load(&fixture.0, 32, 32).unwrap();
        let shards = Shards::open(&fixture.0).unwrap();

        let mut caches = ExpertCaches::new(&model, 8);
        let mut kv = KvState::new(&model);
        let dir = TempDir::new("rabbit_test_kv_session_truncated_file");
        let path = dir.0.join("session.kv");

        let first = vec![1usize, 2, 3];
        step_all(&model, &shards, &mut caches, &mut kv, &first, 0).unwrap();
        kv.save(0, first.len(), &path).unwrap();
        let pos_after_first = first.len();

        let second = vec![4usize, 5];
        step_all(&model, &shards, &mut caches, &mut kv, &second, pos_after_first).unwrap();
        kv.save(pos_after_first, pos_after_first + second.len(), &path).unwrap();

        // chop a few bytes off the very end -- guaranteed to land inside the second record's
        // payload (it's the last thing written), never on a clean record boundary.
        let full_len = fs::metadata(&path).unwrap().len();
        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(full_len - 5).unwrap();
        drop(file);

        let (recovered, pos) = KvState::load(&path, &model).unwrap();
        assert_eq!(pos, pos_after_first, "must discard the truncated trailing record, keeping only the complete one");

        for (a, b) in recovered.layers().iter().zip(kv.layers().iter()) {
            assert_eq!(a.l_range(0, pos_after_first), b.l_range(0, pos_after_first));
        }
    }

    // ---- ExpertCaches::warm_start / save_usage (Fase 14's persistent usage cache) ----

    #[test]
    fn warm_start_below_threshold_seeds_counters_but_marks_no_pin_candidates() {
        let (fixture, _dm) = build_three_layer_fixture("rabbit_test_warm_start_below_threshold");
        let model = Model::load(&fixture.0, 32, 32).unwrap();
        let dir = TempDir::new("rabbit_test_warm_start_below_threshold_file");
        let usage_path = crate::usage_cache::usage_path(&dir.0);
        // well under HIST_THRESHOLD (5000)
        crate::usage_cache::save(&usage_path, vec![(1usize, 0usize, 10u64), (1, 1, 5)].into_iter()).unwrap();

        let mut caches = ExpertCaches::new(&model, 8);
        let stats = caches.warm_start(&dir.0, 8);

        assert_eq!(stats.hist, 15);
        assert_eq!(stats.pin_candidates, 0, "must not mark pin candidates below the confidence threshold");
        assert!(!caches.0[1].as_ref().unwrap().is_pinned(0));

        let counts: std::collections::HashMap<usize, u64> = caches.0[1].as_ref().unwrap().usage_counts().collect();
        assert_eq!(counts.get(&0), Some(&10), "counters must still be seeded even without marking candidates");
        assert_eq!(counts.get(&1), Some(&5));
    }

    #[test]
    fn warm_start_above_threshold_marks_candidates_matching_budget_which_promote_on_first_load() {
        let (fixture, _dm) = build_three_layer_fixture("rabbit_test_warm_start_above_threshold");
        let model = Model::load(&fixture.0, 32, 32).unwrap();
        let shards = Shards::open(&fixture.0).unwrap();
        let dir = TempDir::new("rabbit_test_warm_start_above_threshold_file");
        let usage_path = crate::usage_cache::usage_path(&dir.0);
        // layer 1 (MoE + DSA-full): two nonzero entries, totalling 200_000 -> confidence 1.0.
        // layer 2 (MoE + DSA-shared) gets none, so it must end up with zero candidates.
        crate::usage_cache::save(&usage_path, vec![(1usize, 0usize, 150_000u64), (1, 1, 50_000)].into_iter()).unwrap();

        let cache_capacity = 8;
        let mut caches = ExpertCaches::new(&model, cache_capacity);
        let stats = caches.warm_start(&dir.0, cache_capacity);

        assert_eq!(stats.hist, 200_000);
        assert!((stats.confidence - 1.0).abs() < 1e-6);
        // budget = floor(8 * 0.5 * 1.0) = 4, but only 2 nonzero entries exist for layer 1 -> 2 candidates.
        assert_eq!(stats.pin_candidates, 2);

        // lazy: marking candidates must not load anything.
        assert!(!caches.0[1].as_ref().unwrap().is_pinned(0));
        assert_eq!(caches.0[1].as_ref().unwrap().pinned_len(), 0);

        // the first real load of each marked candidate promotes it.
        caches.0[1].as_mut().unwrap().get_or_load(&shards, &model.cfg, 1, 0, model.ebits).unwrap();
        caches.0[1].as_mut().unwrap().get_or_load(&shards, &model.cfg, 1, 1, model.ebits).unwrap();
        let layer1 = caches.0[1].as_ref().unwrap();
        assert!(layer1.is_pinned(0));
        assert!(layer1.is_pinned(1));
        assert_eq!(layer1.pinned_len(), 2);

        let layer2 = caches.0[2].as_ref().unwrap();
        assert_eq!(layer2.pinned_len(), 0, "a layer with no usage history must not get anything pinned");
    }

    #[test]
    fn save_usage_then_warm_start_round_trips_live_counters_across_a_fresh_caches_instance() {
        let (fixture, _dm) = build_three_layer_fixture("rabbit_test_usage_roundtrip");
        let model = Model::load(&fixture.0, 32, 32).unwrap();
        let dir = TempDir::new("rabbit_test_usage_roundtrip_file");

        let mut caches_a = ExpertCaches::new(&model, 8);
        caches_a.0[1].as_mut().unwrap().record_selection(0);
        caches_a.0[1].as_mut().unwrap().record_selection(0);
        caches_a.0[1].as_mut().unwrap().record_selection(2);
        caches_a.save_usage(&dir.0).unwrap();

        let mut caches_b = ExpertCaches::new(&model, 8);
        let stats = caches_b.warm_start(&dir.0, 8);
        assert_eq!(stats.hist, 3);

        let counts: std::collections::HashMap<usize, u64> = caches_b.0[1].as_ref().unwrap().usage_counts().collect();
        assert_eq!(counts.get(&0), Some(&2));
        assert_eq!(counts.get(&2), Some(&1));
    }
}
