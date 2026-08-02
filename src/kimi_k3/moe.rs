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
//! Dispatch shape (Phase 5 v2 + Phase N3): the apply stage runs as (expert × row-block) tasks
//! through the serial `matmul_qt_rows` — two fan-outs per chunk on the global pool, or one
//! cross-node fan-out with each expert computed on its home node's pinned pool under `--numa`
//! (`dispatch_numa`, bit-identical either way — see `ExpertJob`'s and `dispatch_numa`'s docs for
//! the full argument and `PERFORMANCE.md` for the measurements).
//!
//! `shared_experts(identity)` (computed at the model's real hidden size, on the pre-down-proj
//! hidden states) is each caller's own job, same as `modeling_kimi_linear.py`'s
//! `KimiSparseMoeBlock.forward` adds it AFTER this wrapper's up-projected output — not folded in
//! here, since it needs no down/up-projection at all and `glm52::moe.rs`'s existing shared-expert
//! matmul code is exactly what would be duplicated to inline it.

use crate::expert_cache::{ExpertCache, ExpertSlot, GateUp};
use crate::glm52::config::Cfg;
use crate::glm52::model::{ModelError, MoeWeights};
use crate::glm52::moe::{self, RouteConfig};
use crate::kernels::{matmul_qt, matmul_qt_rows, RowActs};
use crate::kimi_linear::ops::rmsnorm;
use crate::quant::QT;
use crate::safetensors::Shards;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

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

    // `routed` is accumulated in TRANSPOSED `[moe_hidden, s]` layout for the duration of the
    // dispatch (see `ExpertJob`'s doc for why), then transposed back to the `[s, moe_hidden]` the
    // norm and up-projection expect.
    let mut routed_t = vec![0f32; moe_hidden * s];
    let uniq = moe::unique_experts(&routing);
    let chunk_size = cache.capacity().max(1);
    let threads = rayon::current_num_threads();
    for chunk in uniq.chunks(chunk_size) {
        cache.ensure_loaded(shards, cfg_expert, layer, chunk, ebits)?;
        // The chunk's resident slots, in chunk order — immutable refs, because `ensure_loaded`
        // already stamped LRU recency in `begin_loading`, so no `&mut` borrow is needed here
        // (contra the brief's K5 note that `get` takes `&mut self`).
        let jobs: Vec<ExpertJob> = chunk
            .iter()
            .filter_map(|&eid| cache.get(eid))
            .filter_map(|slot| ExpertJob::new(slot, &routing, &x_latent, moe_hidden))
            .collect();
        if jobs.is_empty() {
            continue;
        }
        // Phase N3: with `--numa` active (node pools built), each expert's chain runs on its home
        // node's pinned pool — bit-identical to the global-pool path by `dispatch_numa`'s
        // argument. Otherwise, the two-fan-out global-pool dispatch (Phase 5 v2).
        match crate::numa::NodePools::get() {
            Some(pools) => dispatch_numa(pools, &jobs, layer, cfg_full.n_experts as usize, inter, moe_hidden, s, activation, &mut routed_t),
            None => {
                let refs: Vec<&ExpertJob> = jobs.iter().collect();
                let ggt = gate_up_blocked(&refs, inter, activation, threads);
                down_blocked(&mut routed_t, &jobs, &ggt, moe_hidden, s, threads);
            }
        }
    }

    let mut routed = vec![0f32; s * moe_hidden];
    for (dd, col) in routed_t.chunks(s).enumerate() {
        for (si, &v) in col.iter().enumerate() {
            routed[si * moe_hidden + dd] = v;
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

/// Row-block tasks each fan-out below aims to produce per pool thread. Enough that work-stealing
/// can even out a ragged tail, few enough that each task's own overhead (a heap scratch buffer
/// plus a `matmul_qt_rows` dispatch) stays negligible against the thousands of MACs it then does.
const TASKS_PER_THREAD: usize = 4;

/// Rows per task when `total_rows` independent output rows are to be spread across the pool.
fn block_rows(total_rows: usize, threads: usize) -> usize {
    total_rows.div_ceil((threads * TASKS_PER_THREAD).max(1)).max(1)
}

/// One resident expert's share of a dispatch chunk: the token rows routed to it, and those rows'
/// activations gathered contiguously — exactly the `rows`/`xg` pair `apply_single_expert` builds.
///
/// **Phase 5 v2** (`K3_OPTIMIZE_BRIEF.md`): v1 ran one task per expert and let each task's three
/// `matmul_qt` calls fan out internally. That left the inner fork in place — at batch-1 decode
/// every one of those matmuls splits ~3000 single-element tasks across the pool, so the pool
/// spends its time scheduling rather than computing (measured: decode got *slower* the more
/// threads it was given, `PERFORMANCE.md`). v2 inverts it. The chunk's work is flattened into
/// (expert × row-block) tasks and dispatched as exactly **two** fan-outs per chunk — gate/up +
/// activation, then down + accumulate — with each task calling the serial `matmul_qt_rows` on its
/// own row range. No nested rayon at all, and every task does real vector work.
///
/// Both stages write **transposed** (`[rows, s]`, `matmul_qt`'s own internal `yt` layout) rather
/// than `[s, rows]`: that is what makes one task's row block a contiguous, independently
/// borrowable slice, so the split is `par_chunks_mut` in safe Rust with no aliasing tricks. The
/// two transposes back (`[inter, nr]` → `[nr, inter]` per expert, `[moe_hidden, s]` → `[s,
/// moe_hidden]` once per layer) are O(rows) against O(rows × cols) of matmul — noise, and the
/// same trade `transpose_so` already makes inside every kernel.
struct ExpertJob<'a> {
    slot: &'a ExpertSlot,
    /// `(sequence position, routing weight)`, ascending — `moe::expert_rows`' list verbatim, so
    /// the scatter below reproduces `apply_single_expert`'s exactly.
    rows: Vec<(usize, f32)>,
    /// `[nr, moe_hidden]` — this expert's token rows gathered out of the down-projected input.
    xg: Vec<f32>,
}

impl<'a> ExpertJob<'a> {
    /// `None` for an expert no token routed to — the same early-out `apply_single_expert` takes,
    /// keeping such experts out of the accumulation entirely rather than adding zeros.
    fn new(slot: &'a ExpertSlot, routing: &moe::Routing, x_latent: &[f32], d: usize) -> Option<ExpertJob<'a>> {
        let rows = moe::expert_rows(routing, slot.eid);
        if rows.is_empty() {
            return None;
        }
        let mut xg = vec![0f32; rows.len() * d];
        for (r, &(si, _)) in rows.iter().enumerate() {
            xg[r * d..(r + 1) * d].copy_from_slice(&x_latent[si * d..(si + 1) * d]);
        }
        Some(ExpertJob { slot, rows, xg })
    }

    fn nr(&self) -> usize {
        self.rows.len()
    }
}

/// A job's gate/up weights paired with activations prepared for them once (`RowActs::prepare`
/// hoists the int8-IDOT tiers' per-call quantization out of the per-block calls — see its doc).
/// Mirrors `GateUp` so the fan-out below matches on this alone and can't reach an impossible
/// weights/activations pairing.
enum GateUpActs<'a> {
    Separate { gate: &'a QT, up: &'a QT, xg_gate: RowActs<'a>, xg_up: RowActs<'a> },
    Fused { gate_up: &'a QT, xg: RowActs<'a> },
}

impl<'j> GateUpActs<'j> {
    fn prepare(job: &'j ExpertJob<'_>) -> GateUpActs<'j> {
        let nr = job.nr();
        match &job.slot.gate_up {
            GateUp::Separate { gate, up } => GateUpActs::Separate {
                gate,
                up,
                xg_gate: RowActs::prepare(&job.xg, gate, nr),
                xg_up: RowActs::prepare(&job.xg, up, nr),
            },
            GateUp::Fused { gate_up } => GateUpActs::Fused { gate_up, xg: RowActs::prepare(&job.xg, gate_up, nr) },
        }
    }
}

/// One (expert × row-block) task of stage A: rows `[r0, r0 + seg.len()/nr)` of one job's gate
/// and up matmuls plus the activation, written transposed (`[n, nr]`) into `seg`. Factored out
/// of [`gate_up_blocked`] so the NUMA per-node dispatch ([`dispatch_numa`]) runs the byte-exact
/// same task body — two call sites, one arithmetic path, no way to drift.
fn gate_up_block(acts: &GateUpActs, nr: usize, inter: usize, activation: moe::Activation, r0: usize, seg: &mut [f32]) {
    let n = seg.len() / nr;
    match acts {
        GateUpActs::Separate { gate, up, xg_gate, xg_up } => {
            matmul_qt_rows(seg, xg_gate, gate, r0, n);
            let mut uu = vec![0f32; n * nr];
            matmul_qt_rows(&mut uu, xg_up, up, r0, n);
            activation.apply(seg, &uu);
        }
        // The fused weight's row `k` is gate's, row `inter + k` is up's — the same split
        // `apply_single_expert`'s `Fused` arm makes within one wide output row.
        GateUpActs::Fused { gate_up, xg } => {
            let mut g = vec![0f32; n * nr];
            let mut u = vec![0f32; n * nr];
            matmul_qt_rows(&mut g, xg, gate_up, r0, n);
            matmul_qt_rows(&mut u, xg, gate_up, inter + r0, n);
            for (o, (&gv, &uv)) in seg.iter_mut().zip(g.iter().zip(&u)) {
                *o = activation.combine(gv, uv);
            }
        }
    }
}

/// `[rows, nr]`-transposed stage output → the `[nr, rows]` layout `apply_single_expert`'s `gg`
/// buffer has (see `ExpertJob`'s doc for why the stages compute transposed in the first place).
fn untranspose(t: &[f32], nr: usize) -> Vec<f32> {
    let rows = t.len() / nr.max(1);
    let mut out = vec![0f32; t.len()];
    for (k, col) in t.chunks(nr).enumerate() {
        for (r, &v) in col.iter().enumerate() {
            out[r * rows + k] = v;
        }
    }
    out
}

/// Stage A: every job's gate and up matmuls plus the activation, as ONE fan-out over
/// (expert × `inter` row-block) tasks. Returns each expert's `[nr, inter]` activated intermediate
/// — the buffer `apply_single_expert` calls `gg`, in the same layout, element for element.
///
/// Takes `&[&ExpertJob]` rather than `&[ExpertJob]` so [`dispatch_numa`] can hand it one home
/// domain's (non-contiguous) subset of the chunk's jobs and run the SAME flattened fan-out
/// inside that domain's pool — per-expert-sequential dispatch inside a wide pool was measured
/// re-creating the exact micro-task pathology v2 exists to kill (`PERFORMANCE.md`, D1 flip).
fn gate_up_blocked(jobs: &[&ExpertJob], inter: usize, activation: moe::Activation, threads: usize) -> Vec<Vec<f32>> {
    let acts: Vec<GateUpActs> = jobs.iter().map(|j| GateUpActs::prepare(j)).collect();
    let mut ggt: Vec<Vec<f32>> = jobs.iter().map(|j| vec![0f32; j.nr() * inter]).collect();
    let blk = block_rows(inter * jobs.len(), threads);
    jobs.par_iter()
        .zip(acts.par_iter())
        .zip(ggt.par_iter_mut())
        .flat_map(|((job, acts), buf)| {
            let nr = job.nr();
            buf.par_chunks_mut(blk * nr).enumerate().map(move |(b, seg)| (acts, nr, b * blk, seg))
        })
        .for_each(|(acts, nr, r0, seg)| gate_up_block(acts, nr, inter, activation, r0, seg));
    jobs.iter().zip(&ggt).map(|(job, t)| untranspose(t, job.nr())).collect()
}

/// The down-projections of every job's `gg`, as ONE flattened fan-out over (expert ×
/// `moe_hidden` row-block) tasks, each expert's result in its own TRANSPOSED `[moe_hidden, nr]`
/// buffer. Used by [`dispatch_numa`], which scatters these into `routed_t` sequentially
/// afterward; the `--numa`-off path keeps [`down_blocked`] instead, which fuses the down matmul
/// and the accumulation into one fan-out (it can, because it owns ALL experts — a single domain
/// owning a subset cannot accumulate shared output rows without racing other domains).
fn down_transposed(jobs: &[&ExpertJob], ggs: &[Vec<f32>], moe_hidden: usize, threads: usize) -> Vec<Vec<f32>> {
    let acts: Vec<RowActs> = jobs.iter().zip(ggs).map(|(j, gg)| RowActs::prepare(gg, &j.slot.down, j.nr())).collect();
    let mut hhs: Vec<Vec<f32>> = jobs.iter().map(|j| vec![0f32; moe_hidden * j.nr()]).collect();
    let blk = block_rows(moe_hidden * jobs.len(), threads);
    jobs.par_iter()
        .zip(acts.par_iter())
        .zip(hhs.par_iter_mut())
        .flat_map(|((job, acts), buf)| {
            let nr = job.nr();
            buf.par_chunks_mut(blk * nr).enumerate().map(move |(b, seg)| (job, acts, nr, b * blk, seg))
        })
        .for_each(|(job, acts, nr, r0, seg)| matmul_qt_rows(seg, acts, &job.slot.down, r0, seg.len() / nr));
    hhs
}

/// Stage B: the down-projection and its routing-weighted accumulation into `routed_t`
/// (`[moe_hidden, s]`), as ONE fan-out over `moe_hidden` row-blocks.
///
/// Note which axis is parallel: **rows, not experts**. Each task walks every job in chunk order
/// for its own row range, so each output element accumulates the experts' contributions in the
/// order the sequential loop applied them — no per-expert partial buffers and no reduction pass,
/// and the float addition order is unchanged, which is what makes this bit-identical.
fn down_blocked(routed_t: &mut [f32], jobs: &[ExpertJob], ggs: &[Vec<f32>], moe_hidden: usize, s: usize, threads: usize) {
    let acts: Vec<RowActs> = jobs.iter().zip(ggs).map(|(j, gg)| RowActs::prepare(gg, &j.slot.down, j.nr())).collect();
    let nr_max = jobs.iter().map(ExpertJob::nr).max().unwrap_or(0);
    let blk = block_rows(moe_hidden, threads);
    routed_t.par_chunks_mut(blk * s).enumerate().for_each(|(b, seg)| {
        let r0 = b * blk;
        let n = seg.len() / s;
        let mut hh = vec![0f32; n * nr_max];
        for (job, acts) in jobs.iter().zip(&acts) {
            let nr = job.nr();
            matmul_qt_rows(&mut hh[..n * nr], acts, &job.slot.down, r0, n);
            for (r, &(si, wgt)) in job.rows.iter().enumerate() {
                for j in 0..n {
                    seg[j * s + si] += wgt * hh[j * nr + r];
                }
            }
        }
    });
}

/// Phase N3c (`NUMA_AMX_BRIEF.md`): the same two stages as [`gate_up_blocked`]/[`down_blocked`],
/// but each home domain's pinned pool runs them over ITS OWN experts — the ones whose weight
/// pages `expert_cache`'s `numa_homed_load` first-touched onto that domain (`numa::home_node` is
/// the single agreement point). One [`NodePools::run_all`] fan-out per chunk; domains proceed
/// independently (a domain's stage B needs only its own experts' stage-A output), no cross-domain
/// barrier between stages, no work stealing across domains (brief §2: routing skew is per-token
/// noise — log it via [`numa_moe_stats`], don't fight it).
///
/// Within a domain the work is FLATTENED across its experts — the same
/// [`gate_up_blocked`]/[`down_transposed`] fan-outs the global path uses, just scoped to the
/// domain's job subset and its pool width. The first build of this function ran the domain's
/// experts sequentially (each expert's row-blocks split across the whole pool), which re-created
/// v2's micro-task pathology the moment D1 widened the pools to per-socket: `block_rows(3072,
/// 192×4)` is a 4-row task. Measured as a straight regression and rebuilt this way (see
/// `PERFORMANCE.md`, D1 flip).
///
/// **Bit-identity vs the `--numa`-off path** (the N3 acceptance gate): each row's dot runs the
/// same `matmul_qt_rows` kernel over the same elements ([`gate_up_block`] is shared code;
/// [`down_transposed`] computes the same per-element values [`down_blocked`] does, row-blocking
/// never changes a row's own accumulation), and the routing-weighted accumulation into
/// `routed_t` happens HERE, sequentially, walking jobs in chunk order — the same per-element
/// addition order `down_blocked` produces by walking jobs in order inside each row-block task.
/// Only WHICH thread computes each value and WHEN the accumulation happens differ; no float
/// operation is reordered.
#[allow(clippy::too_many_arguments)]
fn dispatch_numa(
    pools: &crate::numa::NodePools,
    jobs: &[ExpertJob],
    layer: usize,
    n_experts: usize,
    inter: usize,
    moe_hidden: usize,
    s: usize,
    activation: moe::Activation,
    routed_t: &mut [f32],
) {
    let tpp = pools.threads_per_pool();
    let mut by_domain: Vec<Vec<usize>> = vec![Vec::new(); pools.n()];
    for (k, job) in jobs.iter().enumerate() {
        by_domain[crate::numa::home_node(layer, n_experts, job.slot.eid, pools.n())].push(k);
    }
    let mut hhs: Vec<Vec<f32>> = vec![Vec::new(); jobs.len()];
    {
        // One slot per job, each written by exactly one domain — the mutexes are uncontended and
        // exist only to hand `&mut` slots across the `Fn` closure boundary in safe Rust.
        let slots: Vec<std::sync::Mutex<&mut Vec<f32>>> = hhs.iter_mut().map(std::sync::Mutex::new).collect();
        let stats = numa_stats(pools.n());
        pools.run_all(|i| {
            if by_domain[i].is_empty() {
                return;
            }
            let t = std::time::Instant::now();
            let mine: Vec<&ExpertJob> = by_domain[i].iter().map(|&k| &jobs[k]).collect();
            let ggs = gate_up_blocked(&mine, inter, activation, tpp);
            for (&k, hh) in by_domain[i].iter().zip(down_transposed(&mine, &ggs, moe_hidden, tpp)) {
                **slots[k].lock().unwrap() = hh;
            }
            stats[i].0.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            stats[i].1.fetch_add(by_domain[i].len() as u64, Ordering::Relaxed);
        });
    }
    for (job, hh_t) in jobs.iter().zip(&hhs) {
        let nr = job.nr();
        for (j, col) in hh_t.chunks(nr).enumerate() {
            for (r, &(si, wgt)) in job.rows.iter().enumerate() {
                routed_t[j * s + si] += wgt * col[r];
            }
        }
    }
}

/// Cumulative per-node MoE dispatch counters since process start (Phase N3d): `(busy_nanos,
/// experts_dispatched)` per node, updated by [`dispatch_numa`], monotonic. This is the data D1
/// (pool granularity) and the no-cross-node-rebalancing decision are judged on — busy-time
/// imbalance across nodes IS the measured cost of routing skew.
static NUMA_MOE_STATS: OnceLock<Vec<(AtomicU64, AtomicU64)>> = OnceLock::new();

fn numa_stats(n_nodes: usize) -> &'static [(AtomicU64, AtomicU64)] {
    NUMA_MOE_STATS.get_or_init(|| (0..n_nodes).map(|_| (AtomicU64::new(0), AtomicU64::new(0))).collect())
}

/// Snapshot of [`NUMA_MOE_STATS`] as `(busy_seconds, experts_dispatched)` per node — `None`
/// until the first NUMA-dispatched MoE layer has run (which also means: always `None` with
/// `--numa` off). Served by `GET /profile`; totals since process start, so a dashboard diffs
/// consecutive polls for rates.
pub fn numa_moe_stats() -> Option<Vec<(f64, u64)>> {
    let stats = NUMA_MOE_STATS.get()?;
    Some(stats.iter().map(|(ns, n)| (ns.load(Ordering::Relaxed) as f64 / 1e9, n.load(Ordering::Relaxed))).collect())
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

    /// Phase 5 v2's acceptance gate: the row-blocked dispatch is **bit-identical** to applying
    /// the experts one whole expert at a time through `apply_single_expert` — which is both what
    /// this function did before v2 and what `glm52::moe` still does.
    ///
    /// Not a tolerance comparison. v2 changes only WHICH thread computes which output rows, never
    /// the arithmetic: each row's dot runs the same kernel over the same elements in the same
    /// order (`matmul_qt_rows`' own contract, pinned in `kernels.rs`), and each output element
    /// still accumulates its experts in chunk order because stage B parallelizes over rows rather
    /// than experts. `s > 1` and `topk > 1` so experts genuinely share token rows — with `s = 1`
    /// the transposed layouts would be trivially identical and the test would prove much less.
    #[test]
    fn latent_moe_row_blocked_dispatch_is_bit_identical_to_applying_experts_one_at_a_time() {
        let (hidden, moe_hidden, inter, n_experts, topk, n_shared) = (6usize, 4usize, 3usize, 5usize, 3i32, 1i32);
        let mut seed = 909u32;
        let dir = TempDir::new("rabbit_test_k3_latent_moe_row_blocked");
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
        let lw = LatentMoeWeights {
            down_proj: random_qt_f32(moe_hidden, hidden, &mut seed),
            up_proj: random_qt_f32(hidden, moe_hidden, &mut seed),
            norm: None,
        };
        let s = 4;
        let x = random_vec(s * hidden, &mut seed);

        let mut cache = ExpertCache::new(n_experts);
        let mut out = vec![0f32; s * hidden];
        latent_moe(&cfg_full, &cfg_expert, &w, &lw, &mut cache, &shards, 0, 32, &RouteConfig::default(), &x, s, 1e-5, moe::Activation::Silu, &mut out).unwrap();

        // The pre-v2 shape, verbatim: one `apply_single_expert` per expert, accumulating into a
        // single `[s, moe_hidden]` buffer in expert-id (= chunk) order.
        let routing: Routing = moe::route(&cfg_full, &w, &x, s);
        let mut x_latent = vec![0f32; s * moe_hidden];
        matmul_qt(&mut x_latent, &x, &lw.down_proj, s);
        let mut routed_ref = vec![0f32; s * moe_hidden];
        let mut ref_cache = ExpertCache::new(n_experts);
        for eid in moe::unique_experts(&routing) {
            ref_cache.ensure_loaded(&shards, &cfg_expert, 0, &[eid], 32).unwrap();
            let slot = ref_cache.get(eid).unwrap();
            moe::apply_single_expert(slot, &routing, &x_latent, moe_hidden, inter, moe::Activation::Silu, &mut routed_ref);
        }
        let mut expected = vec![0f32; s * hidden];
        matmul_qt(&mut expected, &routed_ref, &lw.up_proj, s);

        let got: Vec<u32> = out.iter().map(|v| v.to_bits()).collect();
        let want: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
        assert_eq!(got, want, "row-blocked dispatch must be bit-identical to the per-expert loop");
    }

    /// Phase N3's acceptance gate at the unit level: `dispatch_numa` (each expert's chain on its
    /// home node's pinned pool, sequential chunk-order accumulation afterward) produces
    /// **bit-identical** output to the global-pool two-fan-out dispatch, on the same jobs. Runs
    /// against the machine's real topology via throwaway `NodePools::build` pools (NOT the
    /// process singleton — that would flip every concurrent test onto the NUMA paths); SKIPs on
    /// single-node machines, where there is nothing to pin to.
    ///
    /// `n_experts = 5` against 6 pools on the target box also exercises nodes with no work; the
    /// `s = 4`, `topk = 3` shape makes experts share token rows, same reasoning as the v2 test.
    #[test]
    fn numa_dispatch_is_bit_identical_to_the_global_pool_dispatch() {
        let Some(topo) = crate::numa::topology() else {
            eprintln!("SKIP: no NUMA topology on this machine");
            return;
        };
        let Some(pools) = crate::numa::NodePools::build(topo.nodes, 12) else {
            eprintln!("SKIP: single NUMA node — nothing to pin to");
            return;
        };
        let (hidden, moe_hidden, inter, n_experts, topk, n_shared) = (6usize, 4usize, 3usize, 5usize, 3i32, 1i32);
        let mut seed = 4242u32;
        let dir = TempDir::new("rabbit_test_k3_numa_dispatch");
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
        let s = 4;
        let x = random_vec(s * hidden, &mut seed);

        let routing: Routing = moe::route(&cfg_full, &w, &x, s);
        let mut x_latent = vec![0f32; s * moe_hidden];
        matmul_qt(&mut x_latent, &x, &down_proj, s);
        let mut cache = ExpertCache::new(n_experts);
        cache.ensure_loaded(&shards, &cfg_expert, 0, &(0..n_experts).collect::<Vec<_>>(), 32).unwrap();
        let jobs: Vec<ExpertJob> = (0..n_experts)
            .filter_map(|eid| cache.get(eid))
            .filter_map(|slot| ExpertJob::new(slot, &routing, &x_latent, moe_hidden))
            .collect();
        assert!(!jobs.is_empty());

        let threads = rayon::current_num_threads();
        let mut routed_global = vec![0f32; moe_hidden * s];
        let refs: Vec<&ExpertJob> = jobs.iter().collect();
        let ggt = gate_up_blocked(&refs, inter, moe::Activation::Silu, threads);
        down_blocked(&mut routed_global, &jobs, &ggt, moe_hidden, s, threads);

        let mut routed_numa = vec![0f32; moe_hidden * s];
        dispatch_numa(&pools, &jobs, 0, n_experts, inter, moe_hidden, s, moe::Activation::Silu, &mut routed_numa);

        let a: Vec<u32> = routed_global.iter().map(|v| v.to_bits()).collect();
        let b: Vec<u32> = routed_numa.iter().map(|v| v.to_bits()).collect();
        assert_eq!(a, b, "NUMA dispatch must be bit-identical to the global-pool dispatch");
    }

    /// Phase 5 determinism gate: two identical warm runs of `latent_moe` produce **bit-identical**
    /// output. Both fan-outs write each output element from exactly one task and accumulate the
    /// experts in fixed chunk order (not nondeterministic completion order), so there is no
    /// run-to-run variation — which is what makes a bit-identical acceptance gate valid at all
    /// for a parallelized path.
    #[test]
    fn latent_moe_is_bit_identical_across_two_warm_runs() {
        let (hidden, moe_hidden, inter, n_experts, topk, n_shared) = (6usize, 4usize, 3usize, 4usize, 2i32, 1i32);
        let mut seed = 77u32;
        let dir = TempDir::new("rabbit_test_k3_latent_moe_determinism");
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
        let lw = LatentMoeWeights {
            down_proj: random_qt_f32(moe_hidden, hidden, &mut seed),
            up_proj: random_qt_f32(hidden, moe_hidden, &mut seed),
            norm: None,
        };
        let s = 4;
        let x = random_vec(s * hidden, &mut seed);

        let run = || {
            let mut cache = ExpertCache::new(n_experts);
            let mut out = vec![0f32; s * hidden];
            latent_moe(&cfg_full, &cfg_expert, &w, &lw, &mut cache, &shards, 0, 32, &RouteConfig::default(), &x, s, 1e-5, moe::Activation::Silu, &mut out).unwrap();
            out
        };
        let a: Vec<u32> = run().iter().map(|v| v.to_bits()).collect();
        let b: Vec<u32> = run().iter().map(|v| v.to_bits()).collect();
        assert_eq!(a, b, "latent_moe must be bit-identical across identical warm runs");
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
