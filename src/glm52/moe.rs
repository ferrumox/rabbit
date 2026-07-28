//! Port of `moe()` (glm.c:1163) and `dense_mlp()` (glm.c:1284) — the two alternative FFN
//! sublayers GLM-5.2 switches between per layer (`layer.sparse`).
//!
//! `moe()`'s routing is GLM's `noaux_tc` scheme: a "correction bias" (a learned per-expert
//! offset, unrelated to any auxiliary loss at inference time) steers *which* experts get
//! picked without affecting *how much weight* each gets — selection sorts by
//! `sigmoid(logit) + bias`, but the weight actually applied is the plain `sigmoid(logit)`.
//! Losing track of that split (using the biased value as the weight, or the unbiased value
//! for selection) is the single easiest way to port this wrong while still producing
//! plausible-looking numbers, so `route()` keeps them as two clearly separate values instead
//! of collapsing into one "score" array like the C's `choice`/`logit` naming barely does.
//!
//! Scope cuts vs the original, all CLI-only debug knobs with no effect at their defaults
//! (`g_topk=0`, `g_topp=0`), so skipping them changes nothing about default behavior:
//! - **`TOPK` override** (`g_topk`): would shrink `K` below `cfg.topk` for disk-cost research.
//! - **Adaptive `TOPP`** (`g_topp`): would keep only enough top-k experts to reach a
//!   cumulative-weight threshold instead of always using all `K`.
//! - **`LOOKA`/`SPEC`/`PILOT` instrumentation and cross-layer prefetch bookkeeping**: research
//!   counters and disk-readahead hints with zero effect on the computed output.
//! - **Live re-pin heat** (`eheat`/`REPIN`): still out of scope (see `expert_cache.rs`'s module
//!   doc). The PERSISTENT half (`eusage` -> `ExpertCache::record_selection`, Fase 13) is now
//!   ported — see the top-k loop below.
//!
//! Expert dispatch is also simplified vs the C's block-of-64 batching — see
//! `expert_cache.rs`'s module doc for why that's a pure performance cut.

use crate::expert_cache::{ExpertCache, ExpertSlot, GateUp};
use crate::glm52::config::Cfg;
use crate::glm52::model::{DenseMlpWeights, ModelError, MoeWeights};
use crate::kernels::matmul_qt;
use crate::safetensors::Shards;

fn siluf(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// The 3 dense layers before `first_k_dense_replace`: plain SiLU-gated MLP, no routing.
/// `hidden` isn't a parameter here (unlike the C's `dense_mlp`, which takes but never uses
/// `D`) — the output width is already fixed by `down_proj`'s shape.
pub fn dense_mlp(w: &DenseMlpWeights, x: &[f32], s: usize, i: usize, out: &mut [f32]) {
    let mut g = vec![0f32; s * i];
    let mut u = vec![0f32; s * i];
    matmul_qt(&mut g, x, &w.gate_proj, s);
    matmul_qt(&mut u, x, &w.up_proj, s);
    for k in 0..s * i {
        g[k] = siluf(g[k]) * u[k];
    }
    matmul_qt(out, &g, &w.down_proj, s);
}

/// Per-token chosen experts: `(expert_id, weight)` pairs, always exactly `cfg.topk` long.
pub struct Routing {
    pub choices: Vec<Vec<(usize, f32)>>,
}

/// Opt-in cache-aware MoE routing (colibrì's `CACHE_ROUTE`, arXiv:2412.00099's max-rank
/// selection) — off by default, matching colibrì's own "never default" stance. When enabled,
/// selection no longer always takes the strict top-`k` by `choice`: the true top-`route_j`
/// ranks are still always taken (never dropped, even if uncached), but the remaining slots
/// prefer experts already resident (pin ∪ LRU) among the next-ranked experts up to
/// `route_m`, falling back to true rank order for anything still unfilled. This can only
/// change WHICH lower-ranked expert ids fill the non-sacred slots — it never runs fewer than
/// `k` experts and never touches the top-`route_j` picks, so a cold cache degrades exactly to
/// plain top-`k` routing (every "preferred resident" slot falls through to the same fallback
/// loop `route()` itself uses).
///
/// Scope cuts vs colibrì's flag surface, both left at their "effectively off" default and not
/// implemented here (see `docs/CACHE_ROUTE.md` in the colibrì source for what they'd do):
/// `ROUTE_P` (cumulative-router-mass window instead of a fixed `route_m`) and `ROUTE_ALPHA`
/// (down-weight substituted experts' gate mass before renorm — default `1` is a no-op anyway).
#[derive(Clone, Copy)]
pub struct RouteConfig {
    pub cache_route: bool,
    pub route_j: usize,
    pub route_m: usize,
}

impl Default for RouteConfig {
    fn default() -> RouteConfig {
        // colibrì's own defaults (ROUTE_J=2, ROUTE_M=12) for when cache_route gets turned on;
        // cache_route itself defaults off, matching colibrì's "never default" stance.
        RouteConfig { cache_route: false, route_j: 2, route_m: 12 }
    }
}

/// Router forward (matmul + sigmoid) and the bias-augmented selection score for one token —
/// shared by both `route()`'s plain top-k and `route_cache_aware`'s wider ranking window.
fn router_scores(cfg: &Cfg, w: &MoeWeights, xs: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let d = cfg.hidden as usize;
    let e = cfg.n_experts as usize;
    let mut logit = vec![0f32; e];
    crate::kernels::matmul(&mut logit, xs, &w.router, 1, d, e);
    for v in logit.iter_mut() {
        *v = 1.0 / (1.0 + (-*v).exp()); // sigmoid
    }
    let choice: Vec<f32> = logit.iter().zip(&w.router_bias).map(|(&l, &b)| l + b).collect();
    (logit, choice)
}

/// Ranks the top `m` experts by `choice` (descending; ties keep the lowest expert id, since a
/// later equal `choice` value doesn't beat an already-found `bv`) — shared by plain top-k
/// selection and CACHE_ROUTE's wider ranking window. Each entry's paired value is that
/// expert's own plain `sigmoid(logit)`, not `choice` — matches `route()`'s existing
/// selection-vs-weight split (see the module doc).
fn rank_top(choice: &[f32], logit: &[f32], m: usize) -> Vec<(usize, f32)> {
    let m = m.min(choice.len());
    let mut taken = vec![false; choice.len()];
    let mut ranked = Vec::with_capacity(m);
    for _ in 0..m {
        let mut best = None;
        let mut bv = -1e30f32;
        for (ei, (&taken_e, &choice_e)) in taken.iter().zip(choice).enumerate() {
            if !taken_e && choice_e > bv {
                bv = choice_e;
                best = Some(ei);
            }
        }
        let best = best.expect("m must not exceed n_experts");
        taken[best] = true;
        ranked.push((best, logit[best]));
    }
    ranked
}

/// Same ranking as `rank_top`, but only among experts where `allowed[eid]` is set — the
/// post-group-restriction top-k step in `rank_top_grouped`.
fn rank_top_within(choice: &[f32], logit: &[f32], k: usize, allowed: &[bool]) -> Vec<(usize, f32)> {
    let k = k.min(allowed.iter().filter(|&&a| a).count());
    let mut taken = vec![false; choice.len()];
    let mut ranked = Vec::with_capacity(k);
    for _ in 0..k {
        let mut best = None;
        let mut bv = -1e30f32;
        for (ei, ((&taken_e, &choice_e), &allowed_e)) in taken.iter().zip(choice).zip(allowed).enumerate() {
            if !taken_e && allowed_e && choice_e > bv {
                bv = choice_e;
                best = Some(ei);
            }
        }
        let best = best.expect("k must not exceed the number of allowed experts");
        taken[best] = true;
        ranked.push((best, logit[best]));
    }
    ranked
}

/// Sum of the two largest values in `vals` (a group's "how good are its best experts" score,
/// DeepSeek-V3/GLM/Kimi-Linear style) — a group with fewer than 2 experts falls back to just
/// its single value rather than manufacturing a second one.
fn top2_sum(vals: &[f32]) -> f32 {
    let mut top1 = f32::NEG_INFINITY;
    let mut top2 = f32::NEG_INFINITY;
    for &v in vals {
        if v > top1 {
            top2 = top1;
            top1 = v;
        } else if v > top2 {
            top2 = v;
        }
    }
    if top2.is_finite() { top1 + top2 } else { top1 }
}

/// Grouped top-k: partitions the `n_experts` choices into `n_group` equal-size contiguous
/// groups, scores each group by `top2_sum`, keeps only the `topk_group` highest-scoring groups
/// (ties keep the lower group index), then runs the ordinary top-`k` (see `rank_top`'s own
/// tie-break) restricted to that surviving set. `n_group <= 1` skips all of this and calls
/// `rank_top` directly — not just an optimization: with one group "keep the top group" would
/// trivially keep everything anyway, so this is the exact same selection, computed the cheap
/// way. This is what `config.rs::Cfg::load` validates `n_experts % n_group == 0` for.
fn rank_top_grouped(choice: &[f32], logit: &[f32], k: usize, n_group: usize, topk_group: usize) -> Vec<(usize, f32)> {
    if n_group <= 1 {
        return rank_top(choice, logit, k);
    }
    let group_size = choice.len() / n_group;
    let mut group_scores: Vec<(usize, f32)> = (0..n_group).map(|g| (g, top2_sum(&choice[g * group_size..(g + 1) * group_size]))).collect();
    group_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("router scores are never NaN").then(a.0.cmp(&b.0)));

    let mut allowed = vec![false; choice.len()];
    for &(g, _) in group_scores.iter().take(topk_group) {
        for e in allowed.iter_mut().skip(g * group_size).take(group_size) {
            *e = true;
        }
    }
    rank_top_within(choice, logit, k, &allowed)
}

fn normalize_and_scale(cfg: &Cfg, picked: &mut [(usize, f32)]) {
    if cfg.norm_topk {
        let sum: f32 = picked.iter().map(|&(_, wt)| wt).sum::<f32>() + 1e-20;
        for (_, wt) in picked.iter_mut() {
            *wt /= sum;
        }
    }
    for (_, wt) in picked.iter_mut() {
        *wt *= cfg.routed_scale;
    }
}

/// Router forward + top-k selection for every token in the batch. Selection order is by
/// `sigmoid(logit) + bias` (descending, ties broken by lowest expert id via a strict `>`
/// scan), but each chosen expert's stored weight is its plain `sigmoid(logit)` — see the
/// module doc.
pub fn route(cfg: &Cfg, w: &MoeWeights, x: &[f32], s: usize) -> Routing {
    let d = cfg.hidden as usize;
    let k = cfg.topk as usize;

    let mut choices = Vec::with_capacity(s);
    for si in 0..s {
        let xs = &x[si * d..(si + 1) * d];
        let (logit, choice) = router_scores(cfg, w, xs);
        let mut picked = rank_top_grouped(&choice, &logit, k, cfg.n_group as usize, cfg.topk_group as usize);
        normalize_and_scale(cfg, &mut picked);
        choices.push(picked);
    }
    Routing { choices }
}

/// Same as `route()`, but consults `cache` for CACHE_ROUTE's resident-preferring fill when
/// `route_cfg.cache_route` is set — see `RouteConfig`'s doc. With `cache_route: false` this is
/// bit-identical to `route()` (same per-token computation, just routed through the shared
/// `rank_top`/`normalize_and_scale` helpers).
///
/// **Not grouping-aware**: `cache_route_select` below still ranks over every expert via plain
/// `rank_top`, not `rank_top_grouped` — combining CACHE_ROUTE (opt-in, off by default) with
/// `n_group > 1` isn't implemented or tested yet. Fine for GLM-5.2 (`n_group` always 1 there,
/// where grouped and plain top-k are identical anyway); revisit before enabling CACHE_ROUTE on
/// a real grouped-routing checkpoint.
pub fn route_cache_aware(cfg: &Cfg, w: &MoeWeights, x: &[f32], s: usize, cache: &ExpertCache, route_cfg: &RouteConfig) -> Routing {
    if !route_cfg.cache_route {
        return route(cfg, w, x, s);
    }
    let d = cfg.hidden as usize;
    let k = cfg.topk as usize;

    let mut choices = Vec::with_capacity(s);
    for si in 0..s {
        let xs = &x[si * d..(si + 1) * d];
        let (logit, choice) = router_scores(cfg, w, xs);
        let mut picked = cache_route_select(&choice, &logit, k, cache, route_cfg);
        normalize_and_scale(cfg, &mut picked);
        choices.push(picked);
    }
    Routing { choices }
}

/// CACHE_ROUTE's max-rank selection for one token: keep the true top-`route_j` always; fill
/// remaining slots preferring resident (pin ∪ LRU) experts among the next-ranked ones up to
/// `route_m`; fall back to true rank order for anything still unfilled.
fn cache_route_select(choice: &[f32], logit: &[f32], k: usize, cache: &ExpertCache, route_cfg: &RouteConfig) -> Vec<(usize, f32)> {
    let m = route_cfg.route_m.max(k);
    let ranked = rank_top(choice, logit, m);
    let j = route_cfg.route_j.min(k).min(ranked.len());

    let mut picked: Vec<(usize, f32)> = Vec::with_capacity(k);
    picked.extend_from_slice(&ranked[..j]);

    if picked.len() < k {
        for &(eid, wt) in &ranked[j..] {
            if picked.len() >= k {
                break;
            }
            if cache.get(eid).is_some() {
                picked.push((eid, wt));
            }
        }
    }
    if picked.len() < k {
        for &(eid, wt) in &ranked[j..] {
            if picked.len() >= k {
                break;
            }
            if picked.iter().any(|&(e2, _)| e2 == eid) {
                continue;
            }
            picked.push((eid, wt));
        }
    }
    picked
}

/// Distinct expert ids across the whole batch's routing, in first-occurrence order — loaded
/// (or cache-hit) once each and applied to every row that picked them, instead of once per
/// (token, k-slot) pair.
fn unique_experts(routing: &Routing) -> Vec<usize> {
    let mut seen = std::collections::HashSet::new();
    let mut uniq = Vec::new();
    for choices in &routing.choices {
        for &(eid, _) in choices {
            if seen.insert(eid) {
                uniq.push(eid);
            }
        }
    }
    uniq
}

/// Routed experts (batch-union dispatch through `cache`) + the always-on shared expert.
/// `out[S,hidden]` is overwritten (not accumulated into) — matches `moe()` zeroing `out` at
/// the start of routing.
#[allow(clippy::too_many_arguments)]
pub fn moe(
    cfg: &Cfg,
    w: &MoeWeights,
    cache: &mut ExpertCache,
    shards: &Shards,
    layer: usize,
    ebits: u8,
    route_cfg: &RouteConfig,
    x: &[f32],
    s: usize,
    out: &mut [f32],
) -> Result<(), ModelError> {
    let d = cfg.hidden as usize;
    let i = cfg.moe_inter as usize;
    let s_i = (cfg.moe_inter * cfg.n_shared) as usize;

    // `route_cache_aware` reads (never mutates) `cache` for residency checks — must happen
    // before the mutable `record_selection`/loading calls below borrow it exclusively.
    let routing = route_cache_aware(cfg, w, x, s, cache, route_cfg);
    // Persistent usage histogram (colibrì's `eusage[layer][eid]++`): once per top-k selection,
    // before any cache resolution -- reflects the router's decision, not cache hit/miss.
    for choices in &routing.choices {
        for &(eid, _) in choices {
            cache.record_selection(eid);
        }
    }
    for v in out[..s * d].iter_mut() {
        *v = 0.0;
    }

    // Fase 8: resolve cache misses via `ensure_loaded` — on Linux that's one `io_uring`
    // submission per chunk for every miss expert's 3 tensors, instead of the `get_or_load`
    // loop's `pread`-per-tensor, per-expert. Chunked at `cache.capacity()`: a single
    // `ensure_loaded` call can't keep more than `capacity` experts resident at once (past
    // that, its own LRU eviction would reclaim earlier insertions from the SAME call before
    // this loop gets to read them back), so a batch whose unique-expert count exceeds
    // capacity — a long prompt's prefill with a high `topk`, easily hundreds of unique ids
    // across a real 256-expert layer — must be dispatched in capacity-sized groups instead of
    // one `ensure_loaded(uniq)` call.
    let uniq = unique_experts(&routing);
    let chunk_size = cache.capacity().max(1);
    let mut chunks = uniq.chunks(chunk_size);

    // Overlap the FIRST chunk's disk read with the shared expert's compute: the shared expert
    // is always active regardless of routing, so its matmuls don't depend on which ROUTED
    // experts end up loaded — `begin_loading` submits that chunk's `io_uring` reads without
    // waiting, we compute the shared expert's VALUE while that read is in flight, and only
    // then drain it. This changes nothing about the RESULT (the shared expert's contribution
    // is still added to `out` in the same relative position, after every routed expert's — see
    // below), only when its otherwise-idle wait time gets spent on independent CPU work
    // instead.
    let first_chunk = chunks.next();
    let pending = match first_chunk {
        Some(chunk) => Some(cache.begin_loading(shards, cfg, layer, chunk, ebits)?),
        None => None,
    };

    let mut sg = vec![0f32; s * s_i];
    let mut su = vec![0f32; s * s_i];
    matmul_qt(&mut sg, x, &w.sh_gate, s);
    matmul_qt(&mut su, x, &w.sh_up, s);
    for z in 0..s * s_i {
        sg[z] = siluf(sg[z]) * su[z];
    }
    let mut hh = vec![0f32; s * d];
    matmul_qt(&mut hh, &sg, &w.sh_down, s);

    // Per-expert early drain: rather than waiting for the WHOLE chunk's disk reads to finish
    // before computing ANY routed expert's matmul, `finish_loading_streaming` calls back the
    // moment each individual expert's own reads land, so its matmul can run while the ring is
    // still waiting on the rest of the chunk. Measured on the real checkpoint (see
    // `PERFORMANCE.md`): across genuinely disk-bound rounds, ~33% of a round's total wait time
    // on average still remains even after half its reads have already completed — a real,
    // otherwise-wasted window this fills with useful compute instead of leaving the CPU idle.
    // Hits (already resident, no read to wait on) are handled separately right after
    // `begin_loading` returns, since `finish_loading_streaming`'s callback only ever sees
    // experts THIS call is actually loading.
    if let (Some(chunk), Some(pending)) = (first_chunk, pending) {
        for &eid in chunk {
            if let Some(slot) = cache.get(eid) {
                apply_single_expert(slot, &routing, x, d, i, out);
            }
        }
        cache.finish_loading_streaming(pending, shards, cfg, layer, ebits, |slot| {
            apply_single_expert(slot, &routing, x, d, i, out);
        })?;
    }
    for chunk in chunks {
        dispatch_chunk_streaming(cache, shards, cfg, layer, chunk, ebits, &routing, x, d, i, out)?;
    }

    // shared expert's contribution, added last — same relative accumulation order into `out`
    // as before this overlap existed (every routed expert's contribution, then this one),
    // even though its VALUE (`hh`) was computed earlier above.
    for z in 0..s * d {
        out[z] += hh[z];
    }

    Ok(())
}

/// Computes and accumulates one routed expert's contribution into `out` — the per-expert body
/// `apply_expert_chunk` used to loop over a whole chunk with; split out so it can be called
/// either from that loop (hits, already resident) or from `finish_loading_streaming`'s
/// per-expert callback (misses, as each one's reads land) without waiting for its neighbors.
#[allow(clippy::too_many_arguments)]
fn apply_single_expert(slot: &ExpertSlot, routing: &Routing, x: &[f32], d: usize, i: usize, out: &mut [f32]) {
    let eid = slot.eid;
    let rows: Vec<(usize, f32)> = routing
        .choices
        .iter()
        .enumerate()
        .filter_map(|(si, picks)| picks.iter().find(|&&(e, _)| e == eid).map(|&(_, wt)| (si, wt)))
        .collect();
    if rows.is_empty() {
        return;
    }
    let nr = rows.len();

    let mut xg = vec![0f32; nr * d];
    for (r, &(si, _)) in rows.iter().enumerate() {
        xg[r * d..(r + 1) * d].copy_from_slice(&x[si * d..(si + 1) * d]);
    }

    let mut gg = vec![0f32; nr * i];
    match &slot.gate_up {
        GateUp::Separate { gate, up } => {
            let mut uu = vec![0f32; nr * i];
            matmul_qt(&mut gg, &xg, gate, nr);
            matmul_qt(&mut uu, &xg, up, nr);
            for z in 0..nr * i {
                gg[z] = siluf(gg[z]) * uu[z];
            }
        }
        // One matmul against the fused `[2*i, d]` weight instead of two against `[i, d]` each —
        // `gu`'s row layout matches `matmul`'s `y[S,O]` convention (`O = 2*i`), so each row's
        // first `i` columns are gate's output and the next `i` are up's, a cheap contiguous
        // per-row split (not a raw-byte split) — see `ROADMAP.md`'s "Fuse gate_proj + up_proj"
        // entry for why that distinction is what makes this safe to do post-matmul.
        GateUp::Fused { gate_up } => {
            let mut gu = vec![0f32; nr * 2 * i];
            matmul_qt(&mut gu, &xg, gate_up, nr);
            for r in 0..nr {
                let row = &gu[r * 2 * i..(r + 1) * 2 * i];
                for k in 0..i {
                    gg[r * i + k] = siluf(row[k]) * row[i + k];
                }
            }
        }
    }
    let mut hh = vec![0f32; nr * d];
    matmul_qt(&mut hh, &gg, &slot.down, nr);

    for (r, &(si, wgt)) in rows.iter().enumerate() {
        for dd in 0..d {
            out[si * d + dd] += wgt * hh[r * d + dd];
        }
    }
}

/// Loads `chunk` (hits touched immediately, misses submitted to `io_uring`) and computes every
/// expert's contribution into `out` as soon as it's available — hits right away (already
/// resident, nothing to wait on), misses as each one's own reads land via
/// `finish_loading_streaming`'s callback, rather than waiting for the whole chunk. Used for
/// every chunk except the first (which gets its own inline version in `moe()` so its
/// `begin_loading` call can happen BEFORE the shared expert's compute, not right before this).
#[allow(clippy::too_many_arguments)]
fn dispatch_chunk_streaming(
    cache: &mut ExpertCache,
    shards: &Shards,
    cfg: &Cfg,
    layer: usize,
    chunk: &[usize],
    ebits: u8,
    routing: &Routing,
    x: &[f32],
    d: usize,
    i: usize,
    out: &mut [f32],
) -> Result<(), ModelError> {
    let pending = cache.begin_loading(shards, cfg, layer, chunk, ebits)?;
    for &eid in chunk {
        if let Some(slot) = cache.get(eid) {
            apply_single_expert(slot, routing, x, d, i, out);
        }
    }
    cache.finish_loading_streaming(pending, shards, cfg, layer, ebits, |slot| {
        apply_single_expert(slot, routing, x, d, i, out);
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::QT;
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

    /// All `rows` rows of `t`, concatenated row-major into one flat `Vec<f32>` — the f32
    /// analogue of what `bin/fuse_gate_up.rs` does at the raw-byte level to build a real
    /// `gate_up_proj` tensor from `gate_proj`+`up_proj`.
    fn qt_rows_f32(t: &QT, rows: usize) -> Vec<f32> {
        (0..rows).flat_map(|r| t.row_f32(r)).collect()
    }

    #[test]
    fn apply_single_expert_fused_gate_up_matches_separate_bit_exact() {
        // d=hidden, i=moe_inter -- deliberately small and non-square so a transposition bug
        // in the fused split would show up as a shape mismatch, not just wrong numbers.
        let (d, i) = (6usize, 8usize);
        let mut seed = 7u32;
        let gate = random_qt_f32(i, d, &mut seed);
        let up = random_qt_f32(i, d, &mut seed);
        let down_vals = random_vec(d * i, &mut seed);
        let down_for_separate = { let mut t = QT::alloc(d, i, 32, false); t.fill(&down_vals); t };
        let down_for_fused = { let mut t = QT::alloc(d, i, 32, false); t.fill(&down_vals); t };

        // The fused [2*i, d] tensor: gate's rows first, then up's -- matches
        // `ExpertNaming::Glm52FusedGateUp`'s doc and `bin/fuse_gate_up.rs`'s own layout.
        let mut fused_vals = qt_rows_f32(&gate, i);
        fused_vals.extend(qt_rows_f32(&up, i));
        let mut gate_up = QT::alloc(2 * i, d, 32, false);
        gate_up.fill(&fused_vals);

        let separate_slot = ExpertSlot::new_for_test(0, GateUp::Separate { gate, up }, down_for_separate);
        let fused_slot = ExpertSlot::new_for_test(0, GateUp::Fused { gate_up }, down_for_fused);

        let x = random_vec(3 * d, &mut seed); // 3 tokens, all routed to expert 0
        let routing = Routing { choices: vec![vec![(0, 0.6)], vec![(0, 1.3)], vec![(0, -0.4)]] };

        let mut out_separate = vec![0f32; 3 * d];
        apply_single_expert(&separate_slot, &routing, &x, d, i, &mut out_separate);
        let mut out_fused = vec![0f32; 3 * d];
        apply_single_expert(&fused_slot, &routing, &x, d, i, &mut out_fused);

        // Splitting a matmul's OUTPUT columns per row (what the fused path does) doesn't
        // reassociate any single row's dot product -- same bytes, same accumulation order per
        // row -- so this must be exact, not just within tolerance.
        assert_eq!(out_separate, out_fused);
    }

    fn f32_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn tiny_cfg(n_experts: i32, topk: i32, moe_inter: i32, hidden: i32, n_shared: i32, norm_topk: bool) -> Cfg {
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
            norm_topk,
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

    /// Builds a tiny on-disk fixture with `n_experts` gate/up/down expert tensors (F32, layer
    /// 0) — used by the CACHE_ROUTE tests below, which need a real `ExpertCache`/`Shards` pair
    /// to seed genuine residency via `ensure_loaded` (unlike the plain routing tests above,
    /// which never touch the cache at all).
    fn build_expert_fixture(dir: &std::path::Path, n_experts: usize, moe_inter: usize, hidden: usize, seed: &mut u32) -> Shards {
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_string(), json!({"format": "rabbit-test"}));
        let mut data = Vec::new();
        for eid in 0..n_experts {
            let gate = random_vec(moe_inter * hidden, seed);
            let up = random_vec(moe_inter * hidden, seed);
            let down = random_vec(hidden * moe_inter, seed);
            for (suf, rows, cols, vals) in [
                ("gate_proj", moe_inter, hidden, &gate),
                ("up_proj", moe_inter, hidden, &up),
                ("down_proj", hidden, moe_inter, &down),
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

    #[test]
    fn route_cache_aware_matches_route_when_cache_route_disabled() {
        let mut seed = 7;
        let cfg = tiny_cfg(5, 3, 4, 5, 1, false);
        let w = MoeWeights {
            router: random_vec(5 * 5, &mut seed),
            router_bias: random_vec(5, &mut seed),
            sh_gate: QT::alloc(1, 5, 32, false),
            sh_up: QT::alloc(1, 5, 32, false),
            sh_down: QT::alloc(5, 1, 32, false),
        };
        let x = random_vec(5, &mut seed);
        let plain = route(&cfg, &w, &x, 1);
        let cache = ExpertCache::new(5);
        let aware = route_cache_aware(&cfg, &w, &x, 1, &cache, &RouteConfig::default());
        assert_eq!(plain.choices, aware.choices, "RouteConfig::default() (cache_route: false) must be bit-identical to plain route()");
    }

    #[test]
    fn cache_route_select_always_keeps_true_top_j_regardless_of_residency() {
        // choice[e] == e as f32, so rank order is exactly descending expert id: 4,3,2,1,0.
        let choice = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let logit = choice.clone();
        let cache = ExpertCache::new(5); // nothing resident anywhere
        let route_cfg = RouteConfig { cache_route: true, route_j: 2, route_m: 5 };
        let picked = cache_route_select(&choice, &logit, 3, &cache, &route_cfg);
        let ids: Vec<usize> = picked.iter().map(|&(e, _)| e).collect();
        assert_eq!(ids, vec![4, 3, 2], "cold cache with no residents must fall back to true top-3, same as plain routing");
    }

    #[test]
    fn cache_route_select_prefers_resident_experts_over_higher_ranked_non_resident_ones() {
        let dir = TempDir::new("rabbit_test_cache_route_resident");
        let mut seed = 99;
        let shards = build_expert_fixture(&dir.0, 5, 4, 5, &mut seed);
        let cfg = tiny_cfg(5, 3, 4, 5, 1, false);
        let mut cache = ExpertCache::new(5);
        cache.ensure_loaded(&shards, &cfg, 0, &[1, 0], 32).unwrap(); // seed genuine residency for experts 1 and 0

        let choice = vec![0.0, 1.0, 2.0, 3.0, 4.0]; // rank order: 4,3,2,1,0
        let logit = choice.clone();
        let route_cfg = RouteConfig { cache_route: true, route_j: 1, route_m: 5 };
        let picked = cache_route_select(&choice, &logit, 3, &cache, &route_cfg);
        let ids: Vec<usize> = picked.iter().map(|&(e, _)| e).collect();
        assert_eq!(
            ids,
            vec![4, 1, 0],
            "the 2 non-sacred slots must prefer resident experts 1 and 0 over higher-ranked but uncached 3 and 2"
        );
        // a substituted expert's stored weight must be its OWN plain sigmoid, not the rank it displaced.
        for &(eid, wt) in &picked {
            assert_eq!(wt, logit[eid]);
        }
    }

    #[test]
    fn cache_route_select_fills_only_as_many_resident_slots_as_needed_then_falls_back() {
        let dir = TempDir::new("rabbit_test_cache_route_partial_resident");
        let mut seed = 17;
        let shards = build_expert_fixture(&dir.0, 5, 4, 5, &mut seed);
        let cfg = tiny_cfg(5, 3, 4, 5, 1, false);
        let mut cache = ExpertCache::new(5);
        cache.ensure_loaded(&shards, &cfg, 0, &[0], 32).unwrap(); // only expert 0 is resident

        let choice = vec![0.0, 1.0, 2.0, 3.0, 4.0]; // rank order: 4,3,2,1,0
        let logit = choice.clone();
        let route_cfg = RouteConfig { cache_route: true, route_j: 1, route_m: 5 };
        let picked = cache_route_select(&choice, &logit, 3, &cache, &route_cfg);
        let ids: Vec<usize> = picked.iter().map(|&(e, _)| e).collect();
        // sacred top-1 (4), then resident 0 fills one non-sacred slot, then the last slot falls
        // back to true rank order (next-highest unpicked: 3), skipping non-resident 2 and 1.
        assert_eq!(ids, vec![4, 0, 3]);
    }

    /// Safety invariant this whole feature depends on: a COLD cache (nothing resident, which is
    /// true at the start of every `moe()` call since routing runs before any loading) must
    /// degrade CACHE_ROUTE to bit-identical output vs it being off — every "prefer resident"
    /// slot finds nothing and falls through to the same true-rank-order fallback plain routing
    /// already uses.
    #[test]
    fn moe_output_with_cache_route_enabled_on_a_cold_cache_matches_disabled() {
        let n_experts = 5;
        let moe_inter = 4;
        let hidden = 5;
        let n_shared = 1;
        let mut seed = 123;

        let dir = TempDir::new("rabbit_test_cache_route_cold_moe");
        let shards = build_expert_fixture(&dir.0, n_experts, moe_inter, hidden, &mut seed);

        let cfg = tiny_cfg(n_experts as i32, 3, moe_inter as i32, hidden as i32, n_shared as i32, false);
        let w = MoeWeights {
            router: random_vec(n_experts * hidden, &mut seed),
            router_bias: random_vec(n_experts, &mut seed),
            sh_gate: random_qt_f32(moe_inter * n_shared, hidden, &mut seed),
            sh_up: random_qt_f32(moe_inter * n_shared, hidden, &mut seed),
            sh_down: random_qt_f32(hidden, moe_inter * n_shared, &mut seed),
        };
        let s = 2;
        let x = random_vec(s * hidden, &mut seed);

        let mut cache_off = ExpertCache::new(n_experts);
        let mut out_off = vec![0f32; s * hidden];
        moe(&cfg, &w, &mut cache_off, &shards, 0, 32, &RouteConfig::default(), &x, s, &mut out_off).unwrap();

        let mut cache_on = ExpertCache::new(n_experts);
        let mut out_on = vec![0f32; s * hidden];
        let route_cfg = RouteConfig { cache_route: true, route_j: 2, route_m: 12 };
        moe(&cfg, &w, &mut cache_on, &shards, 0, 32, &route_cfg, &x, s, &mut out_on).unwrap();

        for (a, b) in out_off.iter().zip(&out_on) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn dense_mlp_matches_hand_computed_silu_gate() {
        let mut seed = 1;
        let d = 3;
        let i = 2;
        let w = DenseMlpWeights {
            gate_proj: random_qt_f32(i, d, &mut seed),
            up_proj: random_qt_f32(i, d, &mut seed),
            down_proj: random_qt_f32(d, i, &mut seed),
        };
        let x = random_vec(d, &mut seed);

        let mut out = vec![0f32; d];
        dense_mlp(&w, &x, 1, i, &mut out);

        // hand-reference using the same primitives, computed independently of dense_mlp's body.
        let mut g = vec![0f32; i];
        let mut u = vec![0f32; i];
        matmul_qt(&mut g, &x, &w.gate_proj, 1);
        matmul_qt(&mut u, &x, &w.up_proj, 1);
        let gated: Vec<f32> = g.iter().zip(&u).map(|(&gv, &uv)| siluf(gv) * uv).collect();
        let mut expected = vec![0f32; d];
        matmul_qt(&mut expected, &gated, &w.down_proj, 1);

        assert_eq!(out, expected);
    }

    #[test]
    fn route_selection_uses_biased_score_but_weight_uses_plain_sigmoid() {
        // 2 experts, top-1: expert 0 has the higher raw logit (so higher sigmoid), but
        // expert 1's correction bias flips the SELECTION to expert 1 — and the weight
        // returned must be expert 1's plain sigmoid, not its biased choice score.
        let cfg = tiny_cfg(2, 1, 4, 2, 1, false);
        let router = vec![10.0, 0.0, -10.0, 0.0]; // row-major [E=2, D=2]; expert0 logit dominates on x=[1,0]
        let router_bias = vec![0.0, 100.0]; // expert 1's bias makes it win selection regardless
        let w = MoeWeights {
            router,
            router_bias,
            sh_gate: QT::alloc(1, 2, 32, false),
            sh_up: QT::alloc(1, 2, 32, false),
            sh_down: QT::alloc(2, 1, 32, false),
        };
        let x = vec![1.0, 0.0];
        let routing = route(&cfg, &w, &x, 1);

        assert_eq!(routing.choices[0].len(), 1);
        let (eid, weight) = routing.choices[0][0];
        assert_eq!(eid, 1, "correction bias must steer selection to expert 1");
        let expected_sigmoid_e1 = 1.0 / (1.0 + (10.0f32).exp()); // logit for expert1 on x=[1,0] is -10
        assert!((weight - expected_sigmoid_e1).abs() < 1e-5, "weight must be the UNBIASED sigmoid, not the choice score");
    }

    #[test]
    fn route_norm_topk_normalizes_weights_to_sum_to_routed_scale() {
        let mut seed = 3;
        let mut cfg = tiny_cfg(4, 3, 4, 5, 1, true);
        cfg.routed_scale = 2.0;
        let w = MoeWeights {
            router: random_vec(4 * 5, &mut seed),
            router_bias: random_vec(4, &mut seed),
            sh_gate: QT::alloc(1, 5, 32, false),
            sh_up: QT::alloc(1, 5, 32, false),
            sh_down: QT::alloc(5, 1, 32, false),
        };
        let x = random_vec(5, &mut seed);
        let routing = route(&cfg, &w, &x, 1);
        let sum: f32 = routing.choices[0].iter().map(|&(_, wt)| wt).sum();
        assert!((sum - cfg.routed_scale).abs() < 1e-5, "norm_topk weights must sum to routed_scale, got {sum}");
    }

    // ---- grouped routing (DeepSeek-V3/GLM/Kimi-Linear style: n_group > 1) ----

    #[test]
    fn top2_sum_adds_the_two_largest_values() {
        assert_eq!(top2_sum(&[1.0, 5.0, 3.0, 2.0]), 5.0 + 3.0);
        assert_eq!(top2_sum(&[5.0, 5.0]), 10.0, "ties still both count");
    }

    #[test]
    fn top2_sum_falls_back_to_the_single_value_for_a_one_element_group() {
        assert_eq!(top2_sum(&[7.0]), 7.0);
    }

    #[test]
    fn rank_top_grouped_matches_plain_rank_top_when_n_group_is_one() {
        let choice = [0.5, 3.0, 1.0, 4.0, 2.0];
        let logit = choice;
        assert_eq!(rank_top_grouped(&choice, &logit, 2, 1, 1), rank_top(&choice, &logit, 2));
    }

    #[test]
    fn rank_top_grouped_keeps_only_experts_from_the_top_scoring_groups() {
        // 8 experts, 4 groups of 2. Group scores (top-2 sum, degenerates to the pair's own sum
        // here since each group has exactly 2 members): group0={0,1}=0.1+0.2=0.3,
        // group1={2,3}=9.0+8.0=17.0, group2={4,5}=0.3+0.4=0.7, group3={6,7}=7.0+6.5=13.5.
        // topk_group=2 keeps group1 and group3 (17.0, 13.5) — group0/group2 must be fully
        // excluded even though nothing INSIDE those groups is being compared here, only the
        // group aggregate.
        let choice = [0.1, 0.2, 9.0, 8.0, 0.3, 0.4, 7.0, 6.5];
        let logit = choice;
        let picked = rank_top_grouped(&choice, &logit, 3, 4, 2);
        let ids: Vec<usize> = picked.iter().map(|&(e, _)| e).collect();
        assert_eq!(ids, vec![2, 3, 6], "top-3 must come only from group1 (2,3) and group3 (6,7), never group0/group2");
    }

    #[test]
    fn rank_top_grouped_ties_keep_the_lower_group_index() {
        // 4 groups of 2, all four scoring identically (0.9+0.9=1.8 each) — topk_group=1 must
        // deterministically keep group0, not an arbitrary tied group.
        let choice = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9];
        let logit = choice;
        let picked = rank_top_grouped(&choice, &logit, 2, 4, 1);
        let ids: Vec<usize> = picked.iter().map(|&(e, _)| e).collect();
        assert_eq!(ids, vec![0, 1], "tied groups must resolve to the lowest index, group0");
    }

    #[test]
    fn route_respects_n_group_and_topk_group_end_to_end() {
        let mut seed = 5;
        let mut cfg = tiny_cfg(8, 3, 4, 6, 1, false);
        cfg.n_group = 4;
        cfg.topk_group = 2;
        let w = MoeWeights {
            router: random_vec(8 * 6, &mut seed),
            router_bias: vec![0.0; 8], // no bias -> selection is purely the router's own sigmoid
            sh_gate: QT::alloc(1, 6, 32, false),
            sh_up: QT::alloc(1, 6, 32, false),
            sh_down: QT::alloc(6, 1, 32, false),
        };
        let x = random_vec(6, &mut seed);

        let (logit, choice) = router_scores(&cfg, &w, &x);
        let expected = rank_top_grouped(&choice, &logit, cfg.topk as usize, cfg.n_group as usize, cfg.topk_group as usize);
        let expected_ids: Vec<usize> = expected.iter().map(|&(e, _)| e).collect();

        let routing = route(&cfg, &w, &x, 1);
        let got_ids: Vec<usize> = routing.choices[0].iter().map(|&(e, _)| e).collect();
        assert_eq!(got_ids, expected_ids, "route() must select the same experts rank_top_grouped alone would");
    }

    /// End-to-end: with `topk == n_experts`, routing always selects EVERY expert, so `moe()`'s
    /// output must equal an independently hand-assembled sum over every expert (each run
    /// directly via `matmul_qt`, bypassing the cache/dispatch machinery entirely) plus the
    /// shared expert — this is the "routing + accumulation arithmetic" the plan calls for.
    #[test]
    fn moe_output_matches_independent_full_expert_sum_when_topk_equals_n_experts() {
        let n_experts = 3;
        let moe_inter = 4;
        let hidden = 5;
        let n_shared = 1;
        let mut seed = 11;

        let dir = TempDir::new("rabbit_test_moe_full_sum");
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_string(), json!({"format": "rabbit-test"}));
        let mut data = Vec::new();
        let mut expert_weights = Vec::new(); // (gate_vals, up_vals, down_vals) per expert, for the independent reference
        for eid in 0..n_experts {
            let gate = random_vec(moe_inter * hidden, &mut seed);
            let up = random_vec(moe_inter * hidden, &mut seed);
            let down = random_vec(hidden * moe_inter, &mut seed);
            for (suf, rows, cols, vals) in [
                ("gate_proj", moe_inter, hidden, &gate),
                ("up_proj", moe_inter, hidden, &up),
                ("down_proj", hidden, moe_inter, &down),
            ] {
                let name = format!("model.layers.0.mlp.experts.{eid}.{suf}.weight");
                let bytes = f32_bytes(vals);
                let start = data.len() as u64;
                data.extend_from_slice(&bytes);
                let end = data.len() as u64;
                header.insert(name, json!({"dtype": "F32", "shape": [rows, cols], "data_offsets": [start, end]}));
            }
            expert_weights.push((gate, up, down));
        }
        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out_bytes = Vec::new();
        out_bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out_bytes.extend_from_slice(&header_bytes);
        out_bytes.extend_from_slice(&data);
        fs::write(dir.0.join("model.safetensors"), out_bytes).unwrap();
        let shards = Shards::open(&dir.0).unwrap();

        let cfg = tiny_cfg(n_experts as i32, n_experts as i32, moe_inter as i32, hidden as i32, n_shared as i32, false);
        let w = MoeWeights {
            router: random_vec(n_experts * hidden, &mut seed),
            router_bias: random_vec(n_experts, &mut seed),
            sh_gate: random_qt_f32(moe_inter * n_shared, hidden, &mut seed),
            sh_up: random_qt_f32(moe_inter * n_shared, hidden, &mut seed),
            sh_down: random_qt_f32(hidden, moe_inter * n_shared, &mut seed),
        };
        let s = 2;
        let x = random_vec(s * hidden, &mut seed);

        let mut cache = ExpertCache::new(n_experts);
        let mut out = vec![0f32; s * hidden];
        moe(&cfg, &w, &mut cache, &shards, 0, 32, &RouteConfig::default(), &x, s, &mut out).unwrap();

        // independent reference: route (to get weights) + sum every expert directly, bypassing moe()'s dispatch.
        let routing = route(&cfg, &w, &x, s);
        let mut expected = vec![0f32; s * hidden];
        for si in 0..s {
            let xs = &x[si * hidden..(si + 1) * hidden];
            for &(eid, wgt) in &routing.choices[si] {
                let (gate_vals, up_vals, down_vals) = &expert_weights[eid];
                let mut gate_qt = QT::alloc(moe_inter, hidden, 32, false);
                gate_qt.fill(gate_vals);
                let mut up_qt = QT::alloc(moe_inter, hidden, 32, false);
                up_qt.fill(up_vals);
                let mut down_qt = QT::alloc(hidden, moe_inter, 32, false);
                down_qt.fill(down_vals);

                let mut g = vec![0f32; moe_inter];
                let mut u = vec![0f32; moe_inter];
                matmul_qt(&mut g, xs, &gate_qt, 1);
                matmul_qt(&mut u, xs, &up_qt, 1);
                let gated: Vec<f32> = g.iter().zip(&u).map(|(&gv, &uv)| siluf(gv) * uv).collect();
                let mut d_out = vec![0f32; hidden];
                matmul_qt(&mut d_out, &gated, &down_qt, 1);
                for dd in 0..hidden {
                    expected[si * hidden + dd] += wgt * d_out[dd];
                }
            }
        }
        // + shared expert, same as moe()'s own FASE E.
        let mut sg = vec![0f32; s * moe_inter * n_shared];
        let mut su = vec![0f32; s * moe_inter * n_shared];
        matmul_qt(&mut sg, &x, &w.sh_gate, s);
        matmul_qt(&mut su, &x, &w.sh_up, s);
        for z in 0..s * moe_inter * n_shared {
            sg[z] = siluf(sg[z]) * su[z];
        }
        let mut sh = vec![0f32; s * hidden];
        matmul_qt(&mut sh, &sg, &w.sh_down, s);
        for z in 0..s * hidden {
            expected[z] += sh[z];
        }

        for (a, b) in out.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
        // sanity: every expert really was touched (topk==n_experts), not a vacuous pass.
        assert_eq!(cache.misses, n_experts as u64);
    }

    /// Fase 13's pin tier is purely a bookkeeping/performance mechanism: whether an expert
    /// happens to already be resident via `pin_expert` (rather than getting there through the
    /// ordinary LRU miss path) must never change `moe()`'s numeric output — same invariant this
    /// module already relies on for io_uring vs. sequential loading.
    #[test]
    fn moe_output_is_identical_whether_or_not_an_expert_is_pre_pinned() {
        let n_experts = 3;
        let moe_inter = 4;
        let hidden = 5;
        let n_shared = 1;
        let mut seed = 42;

        let dir = TempDir::new("rabbit_test_moe_pin_invariance");
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_string(), json!({"format": "rabbit-test"}));
        let mut data = Vec::new();
        for eid in 0..n_experts {
            let gate = random_vec(moe_inter * hidden, &mut seed);
            let up = random_vec(moe_inter * hidden, &mut seed);
            let down = random_vec(hidden * moe_inter, &mut seed);
            for (suf, rows, cols, vals) in [
                ("gate_proj", moe_inter, hidden, &gate),
                ("up_proj", moe_inter, hidden, &up),
                ("down_proj", hidden, moe_inter, &down),
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
        fs::write(dir.0.join("model.safetensors"), out_bytes).unwrap();
        let shards = Shards::open(&dir.0).unwrap();

        let cfg = tiny_cfg(n_experts as i32, n_experts as i32, moe_inter as i32, hidden as i32, n_shared as i32, false);
        let w = MoeWeights {
            router: random_vec(n_experts * hidden, &mut seed),
            router_bias: random_vec(n_experts, &mut seed),
            sh_gate: random_qt_f32(moe_inter * n_shared, hidden, &mut seed),
            sh_up: random_qt_f32(moe_inter * n_shared, hidden, &mut seed),
            sh_down: random_qt_f32(hidden, moe_inter * n_shared, &mut seed),
        };
        let s = 2;
        let x = random_vec(s * hidden, &mut seed);

        let mut cache_unpinned = ExpertCache::new(n_experts);
        let mut out_unpinned = vec![0f32; s * hidden];
        moe(&cfg, &w, &mut cache_unpinned, &shards, 0, 32, &RouteConfig::default(), &x, s, &mut out_unpinned).unwrap();

        let mut cache_pinned = ExpertCache::new(n_experts);
        cache_pinned.mark_pin_candidates(std::iter::once(0usize)); // lazy: promotes on moe()'s own first load of it
        let mut out_pinned = vec![0f32; s * hidden];
        moe(&cfg, &w, &mut cache_pinned, &shards, 0, 32, &RouteConfig::default(), &x, s, &mut out_pinned).unwrap();
        assert!(cache_pinned.is_pinned(0), "topk==n_experts guarantees moe() touched expert 0, so it must have been promoted");

        for (a, b) in out_unpinned.iter().zip(&out_pinned) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    /// Regression test: a batch whose unique-expert count exceeds the cache's capacity used
    /// to panic (`cache.get(eid).expect("just ensured loaded above")`) because a single
    /// `ensure_loaded` call filling the cache past capacity evicted its own earlier
    /// insertions — every slot inserted within one call shares the same LRU timestamp, so the
    /// eviction tie-break (lowest `used`, first element wins) reclaimed experts from the SAME
    /// batch before this function's own loop got to read them back. `topk == n_experts` here
    /// guarantees every token activates all 8 experts, so `uniq.len() == 8` against a cache of
    /// capacity 3 — `moe()` must chunk internally and still produce the exact same output as
    /// the uncapped `moe_output_matches_independent_full_expert_sum_when_topk_equals_n_experts`
    /// reference.
    #[test]
    fn moe_output_is_correct_when_batch_unique_experts_exceed_cache_capacity() {
        let n_experts = 8;
        let moe_inter = 4;
        let hidden = 5;
        let n_shared = 1;
        let mut seed = 21;

        let dir = TempDir::new("rabbit_test_moe_cache_smaller_than_batch");
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_string(), json!({"format": "rabbit-test"}));
        let mut data = Vec::new();
        let mut expert_weights = Vec::new();
        for eid in 0..n_experts {
            let gate = random_vec(moe_inter * hidden, &mut seed);
            let up = random_vec(moe_inter * hidden, &mut seed);
            let down = random_vec(hidden * moe_inter, &mut seed);
            for (suf, rows, cols, vals) in [
                ("gate_proj", moe_inter, hidden, &gate),
                ("up_proj", moe_inter, hidden, &up),
                ("down_proj", hidden, moe_inter, &down),
            ] {
                let name = format!("model.layers.0.mlp.experts.{eid}.{suf}.weight");
                let bytes = f32_bytes(vals);
                let start = data.len() as u64;
                data.extend_from_slice(&bytes);
                let end = data.len() as u64;
                header.insert(name, json!({"dtype": "F32", "shape": [rows, cols], "data_offsets": [start, end]}));
            }
            expert_weights.push((gate, up, down));
        }
        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out_bytes = Vec::new();
        out_bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out_bytes.extend_from_slice(&header_bytes);
        out_bytes.extend_from_slice(&data);
        fs::write(dir.0.join("model.safetensors"), out_bytes).unwrap();
        let shards = Shards::open(&dir.0).unwrap();

        let cfg = tiny_cfg(n_experts as i32, n_experts as i32, moe_inter as i32, hidden as i32, n_shared as i32, false);
        let w = MoeWeights {
            router: random_vec(n_experts * hidden, &mut seed),
            router_bias: random_vec(n_experts, &mut seed),
            sh_gate: random_qt_f32(moe_inter * n_shared, hidden, &mut seed),
            sh_up: random_qt_f32(moe_inter * n_shared, hidden, &mut seed),
            sh_down: random_qt_f32(hidden, moe_inter * n_shared, &mut seed),
        };
        let s = 3;
        let x = random_vec(s * hidden, &mut seed);

        // capacity (3) < uniq.len() (8, since topk==n_experts) -> forces moe() to chunk.
        let mut cache = ExpertCache::new(3);
        let mut out = vec![0f32; s * hidden];
        moe(&cfg, &w, &mut cache, &shards, 0, 32, &RouteConfig::default(), &x, s, &mut out).unwrap();

        let routing = route(&cfg, &w, &x, s);
        let mut expected = vec![0f32; s * hidden];
        for si in 0..s {
            let xs = &x[si * hidden..(si + 1) * hidden];
            for &(eid, wgt) in &routing.choices[si] {
                let (gate_vals, up_vals, down_vals) = &expert_weights[eid];
                let mut gate_qt = QT::alloc(moe_inter, hidden, 32, false);
                gate_qt.fill(gate_vals);
                let mut up_qt = QT::alloc(moe_inter, hidden, 32, false);
                up_qt.fill(up_vals);
                let mut down_qt = QT::alloc(hidden, moe_inter, 32, false);
                down_qt.fill(down_vals);

                let mut g = vec![0f32; moe_inter];
                let mut u = vec![0f32; moe_inter];
                matmul_qt(&mut g, xs, &gate_qt, 1);
                matmul_qt(&mut u, xs, &up_qt, 1);
                let gated: Vec<f32> = g.iter().zip(&u).map(|(&gv, &uv)| siluf(gv) * uv).collect();
                let mut d_out = vec![0f32; hidden];
                matmul_qt(&mut d_out, &gated, &down_qt, 1);
                for dd in 0..hidden {
                    expected[si * hidden + dd] += wgt * d_out[dd];
                }
            }
        }
        let mut sg = vec![0f32; s * moe_inter * n_shared];
        let mut su = vec![0f32; s * moe_inter * n_shared];
        matmul_qt(&mut sg, &x, &w.sh_gate, s);
        matmul_qt(&mut su, &x, &w.sh_up, s);
        for z in 0..s * moe_inter * n_shared {
            sg[z] = siluf(sg[z]) * su[z];
        }
        let mut sh = vec![0f32; s * hidden];
        matmul_qt(&mut sh, &sg, &w.sh_down, s);
        for z in 0..s * hidden {
            expected[z] += sh[z];
        }

        for (a, b) in out.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
        // every one of the 8 experts had to be loaded at least once, despite the cache only
        // ever holding 3 at a time.
        assert_eq!(cache.misses, n_experts as u64);
        assert_eq!(cache.len(), 3);
    }
}
