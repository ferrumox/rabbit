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
//! - **Persistent learning cache updates** (`eusage`/`eheat`): out of scope for this stage.
//!
//! Expert dispatch is also simplified vs the C's block-of-64 batching — see
//! `expert_cache.rs`'s module doc for why that's a pure performance cut.

use crate::config::Cfg;
use crate::expert_cache::ExpertCache;
use crate::kernels::matmul_qt;
use crate::model::{DenseMlpWeights, ModelError, MoeWeights};
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

/// Router forward + top-k selection for every token in the batch. Selection order is by
/// `sigmoid(logit) + bias` (descending, ties broken by lowest expert id via a strict `>`
/// scan), but each chosen expert's stored weight is its plain `sigmoid(logit)` — see the
/// module doc.
pub fn route(cfg: &Cfg, w: &MoeWeights, x: &[f32], s: usize) -> Routing {
    let d = cfg.hidden as usize;
    let e = cfg.n_experts as usize;
    let k = cfg.topk as usize;

    let mut choices = Vec::with_capacity(s);
    for si in 0..s {
        let xs = &x[si * d..(si + 1) * d];
        let mut logit = vec![0f32; e];
        crate::kernels::matmul(&mut logit, xs, &w.router, 1, d, e);
        for v in logit.iter_mut() {
            *v = 1.0 / (1.0 + (-*v).exp()); // sigmoid
        }
        let choice: Vec<f32> = logit.iter().zip(&w.router_bias).map(|(&l, &b)| l + b).collect();

        let mut taken = vec![false; e];
        let mut picked: Vec<(usize, f32)> = Vec::with_capacity(k);
        for _ in 0..k {
            let mut best = None;
            let mut bv = -1e30f32;
            for (ei, (&taken_e, &choice_e)) in taken.iter().zip(&choice).enumerate() {
                if !taken_e && choice_e > bv {
                    bv = choice_e;
                    best = Some(ei);
                }
            }
            let best = best.expect("topk must not exceed n_experts");
            taken[best] = true;
            picked.push((best, logit[best]));
        }

        if cfg.norm_topk {
            let sum: f32 = picked.iter().map(|&(_, wt)| wt).sum::<f32>() + 1e-20;
            for (_, wt) in picked.iter_mut() {
                *wt /= sum;
            }
        }
        for (_, wt) in picked.iter_mut() {
            *wt *= cfg.routed_scale;
        }
        choices.push(picked);
    }
    Routing { choices }
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
    x: &[f32],
    s: usize,
    out: &mut [f32],
) -> Result<(), ModelError> {
    let d = cfg.hidden as usize;
    let i = cfg.moe_inter as usize;
    let s_i = (cfg.moe_inter * cfg.n_shared) as usize;

    let routing = route(cfg, w, x, s);
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
    // then call `finish_loading` to wait for it. This changes nothing about the RESULT (the
    // shared expert's contribution is still added to `out` in the same relative position,
    // after every routed expert's — see below), only when its otherwise-idle wait time gets
    // spent on independent CPU work instead. Later chunks (rare — only when a batch's unique
    // expert count exceeds `cache.capacity()`) don't get this treatment: there's no more
    // independent work left to overlap them with once the shared expert is already computed.
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

    if let (Some(chunk), Some(pending)) = (first_chunk, pending) {
        cache.finish_loading(pending, shards, cfg, layer, ebits)?;
        apply_expert_chunk(cache, chunk, &routing, x, d, i, out);
    }
    for chunk in chunks {
        cache.ensure_loaded(shards, cfg, layer, chunk, ebits)?;
        apply_expert_chunk(cache, chunk, &routing, x, d, i, out);
    }

    // shared expert's contribution, added last — same relative accumulation order into `out`
    // as before this overlap existed (every routed expert's contribution, then this one),
    // even though its VALUE (`hh`) was computed earlier above.
    for z in 0..s * d {
        out[z] += hh[z];
    }

    Ok(())
}

/// Computes and accumulates every routed expert in `chunk`'s contribution into `out`. Every
/// id in `chunk` must already be resident in `cache` (via `ensure_loaded`, or
/// `begin_loading`+`finish_loading`) before calling this.
#[allow(clippy::too_many_arguments)]
fn apply_expert_chunk(cache: &ExpertCache, chunk: &[usize], routing: &Routing, x: &[f32], d: usize, i: usize, out: &mut [f32]) {
    for &eid in chunk {
        let rows: Vec<(usize, f32)> = routing
            .choices
            .iter()
            .enumerate()
            .filter_map(|(si, picks)| picks.iter().find(|&&(e, _)| e == eid).map(|&(_, wt)| (si, wt)))
            .collect();
        if rows.is_empty() {
            continue;
        }
        let nr = rows.len();

        let mut xg = vec![0f32; nr * d];
        for (r, &(si, _)) in rows.iter().enumerate() {
            xg[r * d..(r + 1) * d].copy_from_slice(&x[si * d..(si + 1) * d]);
        }

        let slot = cache.get(eid).expect("just ensured loaded above");
        let mut gg = vec![0f32; nr * i];
        let mut uu = vec![0f32; nr * i];
        matmul_qt(&mut gg, &xg, &slot.gate, nr);
        matmul_qt(&mut uu, &xg, &slot.up, nr);
        for z in 0..nr * i {
            gg[z] = siluf(gg[z]) * uu[z];
        }
        let mut hh = vec![0f32; nr * d];
        matmul_qt(&mut hh, &gg, &slot.down, nr);

        for (r, &(si, wgt)) in rows.iter().enumerate() {
            for dd in 0..d {
                out[si * d + dd] += wgt * hh[r * d + dd];
            }
        }
    }
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
        moe(&cfg, &w, &mut cache, &shards, 0, 32, &x, s, &mut out).unwrap();

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
        moe(&cfg, &w, &mut cache, &shards, 0, 32, &x, s, &mut out).unwrap();

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
