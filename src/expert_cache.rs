//! Port of `expert_load`/`ESlot`/the LRU cache slots inside `moe()` — a bounded, per-layer
//! cache of loaded routed-expert weights.
//!
//! Fase 8 replaces the hot path — resolving a whole batch of cache-miss experts —  with one
//! `io_uring` submission (Linux only) instead of `3 * misses.len()` sequential `pread` calls:
//! "un puñado de syscalls en vez de N hilos bloqueados en pread" per the plan. This is a pure
//! I/O-mechanism change, not a logic change — `ensure_loaded` and `get_or_load` decode and
//! quantize the exact same bytes either way, so `moe.rs`'s output is unaffected; see
//! `tests/teacher_forcing.rs`, still 32/32 after this phase.
//!
//! `benches/expert_load.rs` measures this on the dev machine and — honestly — `io_uring`
//! comes out slower there, because the benchmark's fixture ends up page-cache-hot after the
//! first read. `io_uring`'s win is collapsing blocking syscalls that would otherwise wait on
//! real (cold) disk latency; there's nothing to collapse when the data is already in RAM. See
//! that file's doc comment for the full explanation — it's real disk streaming (the actual
//! 750GB/21,504-expert target) this phase is built for, which this dev box's page cache can't
//! reproduce in a quick local benchmark.
//!
//! Scope cuts vs the original, all purely performance (never correctness) concerns, so the
//! output of `moe.rs` is identical with or without them — any expert fetch, cache hit or disk
//! load, returns the mathematically same weights:
//! - No **pin slots** (`m->pin[layer]`): a deployment feature (`pin_load`/`repin_pass`) for
//!   keeping specific hot experts permanently resident on real hardware, orthogonal to LRU.
//! - No **block-of-64 cap** on a single batch: the C bounds its per-round scratch array at 64
//!   unique experts partly because a fixed-size C array needs a compile-time bound; `Vec` (and
//!   one io_uring ring sized to the batch) doesn't need one. A batch this large only happens
//!   with `S`x`topk` distinct experts in one forward, which stays well under 64 for realistic
//!   batch sizes even on the real 21,504-expert model.
//! - No **O_DIRECT**: the reference implementation's own default is buffered reads too
//!   (`g_direct=0` — "su questo host... il buffered liscio e' risultato il migliore");
//!   O_DIRECT needs page-aligned buffers/offsets/lengths, which is real complexity for a
//!   benefit the original's own measurements don't reliably show. The `io_uring` win here
//!   is purely about collapsing N syscalls into ~1, which buffered reads already get.
//! - No **`.qs` pre-quantized fast path**: same cut as `model.rs`'s `qt_load` — out of scope
//!   for this whole port stage (see `rabbit-plan.md`).
//!
//! Fase 14 (this file) DOES now port the persistent learning cache (`.coli_usage`'s `eusage` ->
//! `usage_cache.rs`'s `.rabbit_usage`) and its pin tier (`m->pin[layer]` -> `ExpertCache::pinned`)
//! — see `usage_cache.rs` and `generate.rs`'s `ExpertCaches::warm_start`. Still cut: colibrì's
//! opt-in LIVE re-pin (`REPIN`/`eheat`/`tier.h`'s decay-and-swap), off by default there too, and
//! any VRAM/CUDA promotion (no CUDA backend here).
//!
//! One deliberate DEVIATION from colibrì's own `pin_load`: that function loads its chosen
//! experts EAGERLY, synchronously, at startup. A real measured A/B against the real checkpoint
//! (`--prompt`, single-shot) showed this makes total wall-clock WORSE, not better — every
//! invocation is a fresh process that pays the full eager-load cost with no long session to
//! amortize it over, for experts THIS particular prompt might not even touch. `pin_candidates`
//! (below) is lazy/sticky instead: `warm_start` only marks which ids are worth keeping once
//! loaded; `insert_or_pin` promotes a candidate into `pinned` the first time generation actually
//! needs it. Zero wasted I/O either way, and `--chat`/`--serve`'s long sessions still converge
//! to the same steady-state protection colibrì's eager version gets, just without ever paying
//! for an expert that never got used.

use crate::glm52::config::Cfg;
use crate::glm52::model::{ModelError, qt_load};
use crate::quant::{QT, QTKind};
use crate::safetensors::Shards;
use std::collections::HashMap;

/// Which on-disk tensor-name convention a checkpoint's routed experts use — the one piece of
/// `expert_cache.rs` that was still GLM-hardcoded (see `rabbit-plan.md`'s Phase 1 notes: "Only
/// one hardcoded HF tensor-name template ties it to GLM's on-disk naming — trivially
/// parameterizable"). Everything else here (LRU, pinning, `io_uring` batching, usage tracking)
/// is keyed by plain `(layer, expert_id)` pairs and was already architecture-agnostic.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExpertNaming {
    /// `model.layers.{layer}.mlp.experts.{eid}.{gate_proj,up_proj,down_proj}.weight`.
    Glm52,
    /// `model.layers.{layer}.block_sparse_moe.experts.{eid}.{w1,w2,w3}.weight` — `w1` is the
    /// gate projection, `w2` is down, `w3` is up (confirmed via `KimiBlockSparseMLP`'s own code
    /// comments in the real `modeling_kimi.py`, not guessed; see `kimi_linear::model`'s doc).
    KimiLinear,
}

impl ExpertNaming {
    /// Returns `[gate, up, down]` tensor names, in that fixed order — every caller in this file
    /// (the `io_uring` batch path and the sequential fallback alike) already indexes reads
    /// positionally as gate/up/down (see `complete_batch_streaming`'s `base+0/1/2`), so this is
    /// the one place that ordering has to be gotten right.
    fn tensor_names(&self, layer: usize, eid: usize) -> [String; 3] {
        match self {
            ExpertNaming::Glm52 => {
                let p = |suf: &str| format!("model.layers.{layer}.mlp.experts.{eid}.{suf}.weight");
                [p("gate_proj"), p("up_proj"), p("down_proj")]
            }
            ExpertNaming::KimiLinear => {
                let p = |suf: &str| format!("model.layers.{layer}.block_sparse_moe.experts.{eid}.{suf}.weight");
                [p("w1"), p("w3"), p("w2")]
            }
        }
    }
}

#[cfg(target_os = "linux")]
mod uring_load {
    use super::{Cfg, ExpertNaming, ExpertSlot, ModelError, QT};
    use crate::safetensors::{DType, SafetensorsError, Shards, TensorLocation, dequant_fp8_blockscale};
    use io_uring::{IoUring, opcode, types};

    struct Req {
        rows: usize,
        cols: usize,
        loc: TensorLocation,
        /// `Some(index into Pending::scale_bufs)` when a `{name}.qs` sibling exists: `loc` then
        /// points at already-packed int8/int4/int2 bytes (the reference's pre-quantized
        /// container format, see `model.rs`'s `qt_load`) to wrap as-is via `QT::from_packed`,
        /// not raw f32/bf16/FP8 to decode and requantize. The scale itself is tiny (`rows`
        /// floats) but is submitted in the SAME `io_uring` batch as `loc`'s own read (not a
        /// synchronous pre-read loop before the batch even starts) — measured on the real
        /// checkpoint that the old "it's only a few KB, not worth batching" reasoning was
        /// wrong: with up to `cache_capacity` misses per round each paying one blocking
        /// syscall's worth of latency SEQUENTIALLY before the big reads even began, that prefix
        /// alone accounted for close to the whole gap between a raw-disk throughput probe
        /// (~4.8 GB/s on this box) and what a real generation run actually achieved (~1.15
        /// GB/s) — see `PERFORMANCE.md`.
        packed_scale: Option<usize>,
        /// `Some(index into Pending::scale_bufs)` when `loc.dtype == F8E4M3` (mutually
        /// exclusive with `packed_scale` — a tensor is never both `.qs`-packed and raw FP8):
        /// the `{name}_scale_inv` 128x128 block-scale sidecar, batched the same way
        /// `packed_scale` is, so the raw bytes `io_uring` reads back can be dequantized
        /// correctly instead of falling through to `Shards::decode_f32`'s *unscaled*
        /// per-element FP8 decode.
        fp8_scale: Option<usize>,
    }

    /// A persistent `io_uring` instance, reused across every `submit_batch` call for as long as
    /// the owning `ExpertCache` lives. Creating a ring isn't free (`io_uring_setup` + mmap'ing
    /// the shared SQ/CQ memory) — the first version of this phase created a fresh one per
    /// call and the `cargo bench` numbers came out SLOWER than plain sequential `pread`,
    /// entirely from that per-call setup cost swallowing the syscall-count savings. `capacity`
    /// is fixed at creation (io_uring rings don't resize) — sized to `cache_capacity * 3` (see
    /// `new_ring`), so a batch chunked to `ExpertCache::capacity()` (as `moe.rs` always does)
    /// fits in exactly one submission round; `submit_batch` falls back to a synchronous load
    /// for any batch that doesn't fit, rather than submitting it across multiple rounds.
    pub(super) struct Ring {
        io: IoUring,
        capacity: usize,
    }

    impl Ring {
        fn new(entries: usize) -> std::io::Result<Ring> {
            let capacity = entries.max(1).next_power_of_two();
            Ok(Ring { io: IoUring::new(capacity as u32)?, capacity })
        }
    }

    /// Creates the persistent ring for an `ExpertCache` of the given ADT capacity: 3
    /// tensors/expert, plus up to one scale-sidecar read per tensor (`.qs` packed scale or FP8
    /// `_scale_inv` — mutually exclusive per tensor, so this is a worst-case upper bound, not a
    /// typical one) submitted in the SAME batch instead of a synchronous pre-read — see
    /// `submit_batch`'s doc for why. Returns `None` if `io_uring` isn't usable on this host (the
    /// `io_uring_disabled` sysctl, a seccomp profile blocking the syscalls, ...) — every
    /// `submit_batch` call then falls back to a synchronous load, same as a hard per-call
    /// failure would.
    pub(super) fn new_ring(cache_capacity: usize) -> Option<Ring> {
        Ring::new(cache_capacity * 3 * 2).ok()
    }

    /// A batch of expert reads submitted to `ring` but not yet awaited — `complete_batch` must
    /// be called (on the SAME `ring`) before the data is valid to decode. Holds the owned
    /// buffers the in-flight SQEs point at, so they can't be dropped out from under the kernel
    /// before completion.
    pub(super) struct Pending {
        reqs: Vec<Req>,
        bufs: Vec<Vec<u8>>,
        /// Scale-sidecar buffers (`.qs`/`_scale_inv`), indexed by `Req::packed_scale`/
        /// `Req::fp8_scale` — submitted in the same `io_uring` round as `bufs`, at `user_data`
        /// tags `reqs.len()..reqs.len()+scale_bufs.len()`.
        scale_bufs: Vec<Vec<u8>>,
        eids: Vec<usize>,
    }

    impl Pending {
        /// The expert ids this batch covers — cheap to clone (just `usize`s), used by
        /// `ExpertCache::finish_loading` to retry via `sequential_fallback` if
        /// `complete_batch` reports an I/O error, without needing to hold onto the (already
        /// consumed) `Pending` itself.
        pub(super) fn eids_for_fallback(&self) -> Vec<usize> {
            self.eids.clone()
        }
    }

    /// Submits reads for every tensor in `misses` (3/expert) — plus, in the SAME `io_uring`
    /// round, any `.qs`/FP8 scale sidecar those tensors need — WITHOUT waiting for completion.
    /// Locating a sidecar (`shards.tensor_location`) is a plain in-memory header lookup, not
    /// I/O, so nothing here blocks; only the actual byte reads are deferred to the ring.
    /// Returns `Ok(None)` — not an error — when the ring is unavailable, the batch doesn't fit
    /// in one submission round, or the submission itself fails; every one of these just means
    /// "the caller should fall back to a synchronous load for this batch," same as a hard
    /// `load_batch` failure always meant before this split existed.
    pub(super) fn submit_batch(ring: &mut Option<Ring>, shards: &Shards, cfg: &Cfg, layer: usize, misses: &[usize], naming: ExpertNaming) -> Result<Option<Pending>, ModelError> {
        let Some(r) = ring.as_mut() else { return Ok(None) };

        let i = cfg.moe_inter as usize;
        let d = cfg.hidden as usize;

        let mut reqs = Vec::with_capacity(misses.len() * 3);
        let mut scale_locs: Vec<TensorLocation> = Vec::new();
        for &eid in misses {
            let names = naming.tensor_names(layer, eid);
            for (name, rows, cols) in [(&names[0], i, d), (&names[1], i, d), (&names[2], d, i)] {
                let loc = shards.tensor_location(name).ok_or(ModelError::Safetensors(SafetensorsError::MissingTensor(name.clone())))?;

                let mut packed_scale = None;
                let mut fp8_scale = None;
                if let Some(sloc) = shards.tensor_location(&format!("{name}.qs")) {
                    packed_scale = Some(scale_locs.len());
                    scale_locs.push(sloc);
                } else if loc.dtype == DType::F8E4M3 {
                    let scale_name = format!("{name}_scale_inv");
                    let sloc = shards.tensor_location(&scale_name).ok_or(ModelError::Safetensors(SafetensorsError::MissingTensor(scale_name)))?;
                    fp8_scale = Some(scale_locs.len());
                    scale_locs.push(sloc);
                }
                reqs.push(Req { rows, cols, loc, packed_scale, fp8_scale });
            }
        }

        if reqs.len() + scale_locs.len() > r.capacity {
            return Ok(None);
        }

        let mut bufs: Vec<Vec<u8>> = reqs.iter().map(|req| vec![0u8; req.loc.nbytes as usize]).collect();
        let mut scale_bufs: Vec<Vec<u8>> = scale_locs.iter().map(|loc| vec![0u8; loc.nbytes as usize]).collect();

        {
            let mut sq = r.io.submission();
            for (local, req) in reqs.iter().enumerate() {
                let buf = &mut bufs[local];
                let read_e = opcode::Read::new(types::Fd(req.loc.fd), buf.as_mut_ptr(), buf.len() as u32).offset(req.loc.offset).build().user_data(local as u64);
                // Safety: `buf` stays alive (owned by the returned `Pending`, held by the
                // caller) until `complete_batch` reaps this SQE's completion, satisfying the
                // lifetime requirement; `push` never fails since we just checked the batch
                // fits within the ring's own capacity.
                unsafe {
                    sq.push(&read_e).expect("checked batch fits ring capacity above");
                }
            }
            for (local, loc) in scale_locs.iter().enumerate() {
                let buf = &mut scale_bufs[local];
                let user_data = (reqs.len() + local) as u64;
                let read_e = opcode::Read::new(types::Fd(loc.fd), buf.as_mut_ptr(), buf.len() as u32).offset(loc.offset).build().user_data(user_data);
                // Safety: same as the main-tensor push above.
                unsafe {
                    sq.push(&read_e).expect("checked batch fits ring capacity above");
                }
            }
        }
        if r.io.submit().is_err() {
            return Ok(None);
        }

        Ok(Some(Pending { reqs, bufs, scale_bufs, eids: misses.to_vec() }))
    }

    /// Waits for every read `submit_batch` started, validates each one's byte count, and
    /// decodes the results into `ExpertSlot`s — the other half of `submit_batch`'s split,
    /// called on the same `ring` (`submit_batch` only returns `Some` when the ring exists, so
    /// this never needs its own "ring unavailable" fallback).
    ///
    /// Unlike a single `submit_and_wait(total)` for the whole round, this waits ONE completion
    /// at a time and decodes each expert's `ExpertSlot` — and calls `on_slot` with it — the
    /// MOMENT that expert's own 3 tensors (+ any scale sidecars) have all arrived, rather than
    /// waiting for every other expert in the round too. A real-checkpoint measurement motivated
    /// this: across 297 genuinely disk-bound rounds, on average 33% of a round's total wait time
    /// still remained even after HALF its reads had already completed (see `PERFORMANCE.md`) —
    /// a real, measured overlap window, not a guess. `on_slot` lets the caller (`moe.rs`) start
    /// that expert's matmul immediately, while the ring keeps waiting on the rest, instead of
    /// leaving the CPU idle until the slowest read in the round finishes.
    ///
    /// Returns the loaded slots (in completion order, not `eids` order — callers that insert by
    /// id, like `finish_loading`, don't care) plus, separately, how many nanoseconds were spent
    /// purely waiting on completions — split out from `load_nanos` the same way the previous
    /// non-streaming version was, for the same reason (see `PERFORMANCE.md`).
    pub(super) fn complete_batch_streaming(ring: &mut Ring, pending: Pending, bits: u8, used: u64, mut on_slot: impl FnMut(&ExpertSlot)) -> Result<(Vec<ExpertSlot>, u64), std::io::Error> {
        let Pending { reqs, mut bufs, scale_bufs, eids } = pending;
        let total = reqs.len() + scale_bufs.len();
        let n_experts = eids.len();

        // `scale_owner[s]` = which expert (index into `eids`) scale-sidecar read `s` belongs
        // to; `needed[e]` = how many of expert `e`'s reads (3 main + 0-3 scale) must land
        // before it's fully decodable.
        let mut scale_owner = vec![usize::MAX; scale_bufs.len()];
        let mut needed = vec![3usize; n_experts];
        for (local, req) in reqs.iter().enumerate() {
            let owner = local / 3;
            if let Some(idx) = req.packed_scale {
                scale_owner[idx] = owner;
                needed[owner] += 1;
            }
            if let Some(idx) = req.fp8_scale {
                scale_owner[idx] = owner;
                needed[owner] += 1;
            }
        }
        let mut got = vec![0usize; n_experts];
        let mut scales: Vec<Vec<f32>> = vec![Vec::new(); scale_bufs.len()];
        let mut out = Vec::with_capacity(n_experts);

        let wait_t = std::time::Instant::now();
        let mut completed = 0usize;
        while completed < total {
            ring.io.submit_and_wait(1)?;
            for cqe in ring.io.completion() {
                completed += 1;
                let idx = cqe.user_data() as usize;
                let n = cqe.result();
                let owner;
                if idx < reqs.len() {
                    let req = &reqs[idx];
                    match n {
                        n if n as u64 == req.loc.nbytes => {}
                        n if n >= 0 => {
                            let msg = format!("expert read: short read ({n}/{} bytes)", req.loc.nbytes);
                            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, msg));
                        }
                        n => return Err(std::io::Error::from_raw_os_error(-n)),
                    }
                    owner = idx / 3;
                } else {
                    let sidx = idx - reqs.len();
                    let buf = &scale_bufs[sidx];
                    match n {
                        n if n as u64 == buf.len() as u64 => {}
                        n if n >= 0 => {
                            let msg = format!("expert scale read: short read ({n}/{} bytes)", buf.len());
                            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, msg));
                        }
                        n => return Err(std::io::Error::from_raw_os_error(-n)),
                    }
                    // decode as soon as this specific scale lands (plain byte->f32
                    // reinterpretation, no I/O left to do) rather than waiting for the round.
                    scales[sidx] = Shards::decode_f32(buf, DType::F32);
                    owner = scale_owner[sidx];
                }

                got[owner] += 1;
                if got[owner] == needed[owner] {
                    // this expert's every read has landed — decode and hand it to the caller
                    // NOW, while the ring keeps waiting on everyone else in the round.
                    // `mem::take` (not indexing/cloning) lets `qt_from_raw` take each buffer BY
                    // VALUE (straight into `QT::from_packed`, which already wants an owned
                    // `Vec<u8>`) instead of `raw.to_vec()`-cloning it — same reasoning as the
                    // buffer-copy fix this replaced (see `PERFORMANCE.md`).
                    let base = owner * 3;
                    let gate = qt_from_raw(std::mem::take(&mut bufs[base]), &reqs[base], bits, &scales).map_err(|e| std::io::Error::other(e.to_string()))?;
                    let up = qt_from_raw(std::mem::take(&mut bufs[base + 1]), &reqs[base + 1], bits, &scales).map_err(|e| std::io::Error::other(e.to_string()))?;
                    let down = qt_from_raw(std::mem::take(&mut bufs[base + 2]), &reqs[base + 2], bits, &scales).map_err(|e| std::io::Error::other(e.to_string()))?;
                    let slot = ExpertSlot { eid: eids[owner], gate, up, down, used };
                    on_slot(&slot);
                    out.push(slot);
                }
            }
        }
        let io_wait_nanos = wait_t.elapsed().as_nanos() as u64;
        Ok((out, io_wait_nanos))
    }

    fn qt_from_raw(raw: Vec<u8>, req: &Req, bits: u8, scales: &[Vec<f32>]) -> Result<QT, ModelError> {
        if let Some(idx) = req.packed_scale {
            return QT::from_packed(req.rows, req.cols, bits, raw, scales[idx].clone())
                .map_err(|source| ModelError::PackedFormat { name: "expert tensor (.qs)".to_string(), source });
        }
        let w = if let Some(idx) = req.fp8_scale {
            dequant_fp8_blockscale(&raw, req.rows as u64, req.cols as u64, &scales[idx], "expert tensor (fp8 scale_inv)")
                .map_err(ModelError::Safetensors)?
        } else {
            Shards::decode_f32(&raw, req.loc.dtype)
        };
        let mut t = QT::alloc(req.rows, req.cols, bits, false);
        t.fill(&w);
        Ok(t)
    }
}

pub struct ExpertSlot {
    pub eid: usize,
    pub gate: QT,
    pub up: QT,
    pub down: QT,
    used: u64,
}

/// A batch of experts `ExpertCache::begin_loading` started resolving — opaque to callers other
/// than "pass this to `finish_loading`". See that pair's doc for why this split exists.
pub struct PendingExpertLoad(LoadKind);

enum LoadKind {
    /// Every requested id was already cached — nothing to wait for.
    Nothing,
    /// Loaded synchronously at `begin_loading` time already (non-Linux, or the ring was
    /// unavailable/the batch didn't fit in one round) — `finish_loading` just inserts these.
    Sync(Vec<ExpertSlot>),
    /// Submitted to the `io_uring` ring but not yet awaited.
    #[cfg(target_os = "linux")]
    Async(uring_load::Pending),
}

/// A fixed-capacity, per-layer LRU cache of `ExpertSlot`s. Real usage holds one of these per
/// MoE layer (`layers_forward`), living for the whole generation session — which matters here
/// specifically because the `io_uring` ring (Linux) is created once in `new` and reused for
/// every `ensure_loaded` call over that lifetime; see `uring_load::Ring`'s doc for why.
pub struct ExpertCache {
    capacity: usize,
    slots: Vec<ExpertSlot>,
    /// Checked before `slots` on every lookup, never touched by `insert`'s LRU eviction —
    /// colibrì's `m->pin[layer]`. Structurally separate from `slots` rather than a "never
    /// evict" flag on the same `Vec`, so the existing LRU code in `insert` needs zero changes
    /// to stay correct. Populated LAZILY (see `pin_candidates` below), not by eagerly loading
    /// anything at startup — a real measured comparison against the real checkpoint showed
    /// eager pinning (colibrì's own `pin_load` design) actively hurts `--prompt`'s one-shot
    /// wall-clock time: a fresh process pays the full eager-load cost every invocation with no
    /// session to amortize it over, and the experts it eagerly loads may not even be the ones
    /// THIS particular prompt needs.
    pinned: Vec<ExpertSlot>,
    /// Expert ids `ExpertCaches::warm_start` decided are worth keeping resident forever, based
    /// on persisted history — but not loaded yet. The next time one of these is genuinely
    /// loaded from disk because generation actually needed it (`insert_or_pin`'s miss path),
    /// it's promoted straight into `pinned` instead of the ordinary LRU `slots`. This "lazy" /
    /// "sticky" design never loads anything speculatively (unlike colibrì's eager `pin_load`):
    /// zero wasted I/O if this session doesn't touch a candidate, and the same steady-state
    /// protection-from-eviction benefit once it does. Must be populated (via
    /// `mark_pin_candidates`) before the FIRST load of any given id in a session — marking a
    /// candidate that's already resident in `slots` does not retroactively promote it (only a
    /// fresh miss does), which is fine since `warm_start` always runs before any generation.
    pin_candidates: std::collections::HashSet<usize>,
    /// This layer's cumulative selection count per expert id — colibrì's `eusage[layer][*]`.
    /// Seeded from `.rabbit_usage` at startup (`seed_usage`) and bumped live by `moe.rs`
    /// (`record_selection`) on every router pick, then written back out (`usage_counts`) at
    /// turn/response boundaries. Never decayed, unlike colibrì's separate (and here unported)
    /// `eheat`.
    usage: HashMap<usize, u64>,
    clock: u64,
    pub hits: u64,
    pub misses: u64,
    /// Cumulative wall time spent actually loading missed experts from disk — the synchronous
    /// load in `begin_loading` (non-Linux, or no ring) plus the wait in `finish_loading`
    /// (`io_uring`'s `complete_batch`), not `get_or_load` — that path is only used directly by
    /// tests/benches. A diagnostic counter, not used by any dispatch logic — see
    /// `ExpertCaches::hit_miss_totals` in `generate.rs` for why it exists: telling apart "still
    /// I/O-bound" from "already compute-bound" before investing in either kind of further
    /// optimization.
    pub load_nanos: u64,
    /// The portion of `load_nanos` (Linux `io_uring` path only) spent purely in
    /// `submit_and_wait` — actually waiting on the kernel/disk, not decoding or copying the
    /// results afterward. `load_nanos - io_wait_nanos` is everything else `finish_loading`'s
    /// async branch does (validation, scale decode, the `qt_from_raw` loop's buffer copies) —
    /// distinguishing the two is what found the buffer-copy fix documented in
    /// `PERFORMANCE.md`: about 40% of a real run's reported "disk I/O" time turned out to be
    /// this CPU-side work, not the drive.
    pub io_wait_nanos: u64,
    #[cfg(target_os = "linux")]
    ring: Option<uring_load::Ring>,
    naming: ExpertNaming,
}

impl ExpertCache {
    /// The most unique experts `ensure_loaded` can keep simultaneously resident. Callers that
    /// dispatch a batch of experts (`moe.rs`) must chunk any set larger than this — a single
    /// `ensure_loaded` call filling the cache past capacity evicts its OWN earlier insertions
    /// (every slot inserted within one call shares the same LRU timestamp, so ties break by
    /// insertion order) before the caller gets a chance to read them back via `get`.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn new(capacity: usize) -> ExpertCache {
        Self::for_family(capacity, ExpertNaming::Glm52)
    }

    /// Like `new`, but for a checkpoint whose routed experts use a different on-disk
    /// tensor-name convention (see `ExpertNaming`) — e.g. Kimi Linear's
    /// `block_sparse_moe.experts.{eid}.{w1,w2,w3}` instead of GLM-5.2's
    /// `mlp.experts.{eid}.{gate_proj,up_proj,down_proj}`.
    pub fn for_family(capacity: usize, naming: ExpertNaming) -> ExpertCache {
        ExpertCache {
            capacity,
            slots: Vec::new(),
            pinned: Vec::new(),
            pin_candidates: std::collections::HashSet::new(),
            usage: HashMap::new(),
            clock: 0,
            hits: 0,
            misses: 0,
            load_nanos: 0,
            io_wait_nanos: 0,
            #[cfg(target_os = "linux")]
            ring: uring_load::new_ring(capacity),
            naming,
        }
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Drops every cached slot (next lookup for any id is a miss) without tearing down the
    /// `io_uring` ring — for benchmarking/testing load performance repeatedly against the
    /// same long-lived cache, which is the realistic shape (a fresh `ExpertCache` per
    /// forward pass would pay ring setup cost every call; see `uring_load::Ring`'s doc).
    pub fn clear(&mut self) {
        self.slots.clear();
    }

    /// Returns the slot for `eid`, loading it from disk on a miss. On a miss at capacity,
    /// evicts whichever cached slot was least recently touched (smallest `used` stamp) —
    /// same tie-break as the C's `for z in 1..*nn: if Sl[z].used<Sl[lru].used: lru=z` (first
    /// minimum wins).
    pub fn get_or_load(
        &mut self,
        shards: &Shards,
        cfg: &Cfg,
        layer: usize,
        eid: usize,
        bits: u8,
    ) -> Result<&ExpertSlot, ModelError> {
        self.clock += 1;
        if let Some(pos) = self.pinned.iter().position(|s| s.eid == eid) {
            self.hits += 1;
            return Ok(&self.pinned[pos]);
        }
        if let Some(pos) = self.slots.iter().position(|s| s.eid == eid) {
            self.hits += 1;
            self.slots[pos].used = self.clock;
            return Ok(&self.slots[pos]);
        }

        self.misses += 1;
        let fresh = load_expert(shards, cfg, layer, eid, bits, self.clock, self.naming)?;
        self.insert_or_pin(fresh);
        Ok(self.get(eid).expect("just inserted or pinned above"))
    }

    /// Ensures every expert in `eids` is present in the cache — hits are just touched
    /// (`used = clock`), misses are loaded via one batched `io_uring` submission (Linux) or
    /// sequential `pread`s (elsewhere), then inserted with the same LRU eviction as
    /// `get_or_load`. Look up results afterward with `get` — `moe.rs`'s per-row dispatch loop
    /// wants one `&ExpertSlot` at a time anyway, and returning several `&mut self`-derived
    /// references from one call would fight the borrow checker for no real benefit.
    pub fn ensure_loaded(&mut self, shards: &Shards, cfg: &Cfg, layer: usize, eids: &[usize], bits: u8) -> Result<(), ModelError> {
        let pending = self.begin_loading(shards, cfg, layer, eids, bits)?;
        self.finish_loading(pending, shards, cfg, layer, bits)
    }

    /// Begins resolving every expert in `eids`: hits are touched immediately (same LRU
    /// bookkeeping `ensure_loaded` always did); misses are either fully loaded synchronously
    /// right now (non-Linux, or the `io_uring` ring unavailable/the batch too big for one
    /// submission round) or SUBMITTED to the ring WITHOUT waiting for completion. Splitting
    /// this out from `ensure_loaded` lets a caller (`moe.rs`) do independent work — the shared
    /// expert's matmuls don't depend on which ROUTED experts are loaded — while the disk read
    /// is in flight, instead of blocking on it up front. Must be paired with `finish_loading`
    /// on the same `eids`/`bits` before looking any of them up via `get`.
    pub fn begin_loading(&mut self, shards: &Shards, cfg: &Cfg, layer: usize, eids: &[usize], bits: u8) -> Result<PendingExpertLoad, ModelError> {
        self.clock += 1;
        let mut misses = Vec::new();
        for &eid in eids {
            if self.pinned.iter().any(|s| s.eid == eid) {
                self.hits += 1;
            } else if let Some(pos) = self.slots.iter().position(|s| s.eid == eid) {
                self.hits += 1;
                self.slots[pos].used = self.clock;
            } else if !misses.contains(&eid) {
                misses.push(eid);
            }
        }
        if misses.is_empty() {
            return Ok(PendingExpertLoad(LoadKind::Nothing));
        }
        self.misses += misses.len() as u64;

        // Timed from here, not just around the actual disk reads below: `submit_batch` itself
        // does real synchronous I/O first (reading each miss's `.qs`/FP8 scale sidecar, a
        // handful of small `pread`s per expert before it ever touches the ring) — leaving that
        // untimed would silently undercount `load_nanos` for the async path relative to what
        // `finish_loading` measures for its own wait, making the two look artificially
        // different without any real work having moved anywhere. Caught this by comparing
        // `load_nanos` readings against total wall-clock time, which didn't move the way the
        // I/O counter suggested it should have — see the project memory for the full story.
        let load_t = std::time::Instant::now();

        #[cfg(target_os = "linux")]
        if let Some(pending) = uring_load::submit_batch(&mut self.ring, shards, cfg, layer, &misses, self.naming)? {
            self.load_nanos += load_t.elapsed().as_nanos() as u64;
            return Ok(PendingExpertLoad(LoadKind::Async(pending)));
        }

        let loaded = sequential_fallback(shards, cfg, layer, &misses, bits, self.clock, self.naming)?;
        self.load_nanos += load_t.elapsed().as_nanos() as u64;
        Ok(PendingExpertLoad(LoadKind::Sync(loaded)))
    }

    /// Waits for (if `begin_loading` submitted an async read) and inserts every expert that
    /// call started resolving. `shards`/`cfg`/`layer`/`bits` must match the `begin_loading`
    /// call this `pending` came from — only needed for the rare fallback path (an `io_uring`
    /// completion error retries as a plain synchronous read of the same misses).
    pub fn finish_loading(&mut self, pending: PendingExpertLoad, shards: &Shards, cfg: &Cfg, layer: usize, bits: u8) -> Result<(), ModelError> {
        self.finish_loading_streaming(pending, shards, cfg, layer, bits, |_| {})
    }

    /// Same contract as `finish_loading`, but calls `on_slot` with each expert's `ExpertSlot`
    /// the moment ITS reads land, instead of only after the whole batch completes — see
    /// `uring_load::complete_batch_streaming`'s doc for why that gap is real and worth using.
    /// `on_slot` only ever sees experts THIS call is loading (misses) — anything already
    /// resident (a hit) never appears here; callers that need to handle hits too (`moe.rs`)
    /// should check `cache.get(eid)` for the rest of their batch separately.
    pub fn finish_loading_streaming(&mut self, pending: PendingExpertLoad, shards: &Shards, cfg: &Cfg, layer: usize, bits: u8, mut on_slot: impl FnMut(&ExpertSlot)) -> Result<(), ModelError> {
        let loaded = match pending.0 {
            LoadKind::Nothing => return Ok(()),
            LoadKind::Sync(v) => {
                for slot in &v {
                    on_slot(slot);
                }
                v
            }
            #[cfg(target_os = "linux")]
            LoadKind::Async(p) => {
                let eids_for_fallback = p.eids_for_fallback();
                let load_t = std::time::Instant::now();
                let ring = self.ring.as_mut().expect("Async pending implies the ring existed when begin_loading submitted it");
                let result = uring_load::complete_batch_streaming(ring, p, bits, self.clock, &mut on_slot);
                self.load_nanos += load_t.elapsed().as_nanos() as u64;
                match result {
                    Ok((v, io_wait_nanos)) => {
                        self.io_wait_nanos += io_wait_nanos;
                        v
                    }
                    Err(_) => {
                        let v = sequential_fallback(shards, cfg, layer, &eids_for_fallback, bits, self.clock, self.naming)?;
                        for slot in &v {
                            on_slot(slot);
                        }
                        v
                    }
                }
            }
        };

        for slot in loaded {
            self.insert_or_pin(slot);
        }
        Ok(())
    }

    /// Looks up an already-cached slot without touching LRU state or loading anything —
    /// pairs with `ensure_loaded`, which the caller runs first for a whole batch. Checks the
    /// pinned tier first (never evicted, so a pinned id is always found here even if it would
    /// also — redundantly — appear in `slots` from before it got pinned).
    pub fn get(&self, eid: usize) -> Option<&ExpertSlot> {
        self.pinned.iter().find(|s| s.eid == eid).or_else(|| self.slots.iter().find(|s| s.eid == eid))
    }

    /// Records one router selection of `eid` this layer — colibrì's `eusage[layer][eid]++`,
    /// called from `moe.rs` right after top-k selection, before cache resolution. Saturating,
    /// not wrapping (colibrì's raw `uint32_t++` can theoretically wrap after 4B selections;
    /// `u64` here makes that a non-concern, `saturating_add` is just cheap insurance).
    pub(crate) fn record_selection(&mut self, eid: usize) {
        let c = self.usage.entry(eid).or_insert(0);
        *c = c.saturating_add(1);
    }

    /// This layer's current usage counts, for `ExpertCaches::save_usage` to persist.
    pub(crate) fn usage_counts(&self) -> impl Iterator<Item = (usize, u64)> + '_ {
        self.usage.iter().map(|(&eid, &c)| (eid, c))
    }

    /// Seeds this layer's counters from persisted history at startup — before any live
    /// `record_selection` calls this session, so `usage` ends up holding old+new counts in one
    /// place with no separate delta-tracking needed (mirrors colibrì's `usage_load`, which adds
    /// into the same `eusage` array live counting uses).
    pub(crate) fn seed_usage(&mut self, counts: impl Iterator<Item = (usize, u64)>) {
        for (eid, c) in counts {
            self.usage.insert(eid, c);
        }
    }

    pub fn is_pinned(&self, eid: usize) -> bool {
        self.pinned.iter().any(|s| s.eid == eid)
    }

    pub fn pinned_len(&self) -> usize {
        self.pinned.len()
    }

    /// Marks `eids` as pin candidates — see `pin_candidates`'s field doc for the lazy/sticky
    /// design this enables. Must be called before any of these ids get loaded this session
    /// (`ExpertCaches::warm_start` always runs before generation starts, so this holds in
    /// practice).
    pub(crate) fn mark_pin_candidates(&mut self, eids: impl Iterator<Item = usize>) {
        self.pin_candidates.extend(eids);
    }

    /// Routes a freshly loaded expert to the right tier: if it's a pin candidate, promotes it
    /// straight into `pinned` (mlocked, never evicted) instead of the ordinary LRU `slots` —
    /// the one and only place lazy/sticky promotion actually happens. `used` is irrelevant for
    /// a pinned slot, stamped 0.
    fn insert_or_pin(&mut self, mut slot: ExpertSlot) {
        if self.pin_candidates.remove(&slot.eid) {
            slot.used = 0;
            mlock_best_effort(&slot);
            self.pinned.push(slot);
        } else {
            self.insert(slot);
        }
    }

    fn insert(&mut self, slot: ExpertSlot) {
        if self.slots.len() < self.capacity {
            self.slots.push(slot);
        } else {
            let lru = self
                .slots
                .iter()
                .enumerate()
                .min_by_key(|(_, s)| s.used)
                .map(|(i, _)| i)
                .expect("capacity > 0 implies at least one slot once full");
            self.slots[lru] = slot;
        }
    }
}

/// Used as `begin_loading`'s non-Linux (or ring-unavailable) synchronous path, and as
/// `finish_loading`'s fallback when `uring_load::complete_batch` reports an I/O error.
fn sequential_fallback(shards: &Shards, cfg: &Cfg, layer: usize, misses: &[usize], bits: u8, used: u64, naming: ExpertNaming) -> Result<Vec<ExpertSlot>, ModelError> {
    misses.iter().map(|&eid| load_expert(shards, cfg, layer, eid, bits, used, naming)).collect()
}

fn load_expert(shards: &Shards, cfg: &Cfg, layer: usize, eid: usize, bits: u8, used: u64, naming: ExpertNaming) -> Result<ExpertSlot, ModelError> {
    let i = cfg.moe_inter as usize;
    let d = cfg.hidden as usize;
    let names = naming.tensor_names(layer, eid);
    let gate = qt_load(shards, &names[0], i, d, bits)?;
    let up = qt_load(shards, &names[1], i, d, bits)?;
    let down = qt_load(shards, &names[2], d, i, bits)?;
    Ok(ExpertSlot { eid, gate, up, down, used })
}

/// Best-effort `mlock` on a pinned expert's backing buffers, so the OS can't swap out memory
/// pinning exists specifically to keep resident — same "unsafe libc call, best-effort, never a
/// hard error" precedent already established for `posix_fadvise` in `safetensors.rs`. A failure
/// (commonly `RLIMIT_MEMLOCK` too low for an unprivileged process) is logged once and otherwise
/// ignored: losing the OS-level guarantee just means a pinned expert *could* get swapped under
/// real memory pressure, same risk profile as not pinning it at all — not a correctness issue.
fn mlock_best_effort(slot: &ExpertSlot) {
    for qt in [&slot.gate, &slot.up, &slot.down] {
        let bufs: Vec<(*const u8, usize)> = match &qt.kind {
            QTKind::F32(v) => vec![(v.as_ptr() as *const u8, std::mem::size_of_val(v.as_slice()))],
            QTKind::I8 { data, scale } => {
                vec![(data.as_ptr() as *const u8, std::mem::size_of_val(data.as_slice())), (scale.as_ptr() as *const u8, std::mem::size_of_val(scale.as_slice()))]
            }
            QTKind::I4 { data, scale } | QTKind::I2 { data, scale } => {
                vec![(data.as_ptr(), data.len()), (scale.as_ptr() as *const u8, std::mem::size_of_val(scale.as_slice()))]
            }
            QTKind::MxFp4 { data, block_scale } => {
                vec![(data.as_ptr(), data.len()), (block_scale.as_ptr(), block_scale.len())]
            }
        };
        for (ptr, len) in bufs {
            if len > 0 && unsafe { libc::mlock(ptr as *const libc::c_void, len) } != 0 {
                eprintln!("usage cache: mlock failed for a pinned expert (best-effort, continuing): {}", std::io::Error::last_os_error());
            }
        }
    }
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

    fn f32_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// writes `model.safetensors` with `gate_proj`/`up_proj`/`down_proj` for expert ids
    /// `0..n_experts` at layer 0, shapes `[moe_inter, hidden]`/`[moe_inter, hidden]`/
    /// `[hidden, moe_inter]`, and returns the fixture dir (kept alive by the caller).
    fn build_experts_fixture(name: &str, n_experts: usize, moe_inter: usize, hidden: usize) -> TempDir {
        let dir = TempDir::new(name);
        let mut seed = 1u32;
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_string(), json!({"format": "rabbit-test"}));
        let mut data = Vec::new();
        let mut push = |header: &mut serde_json::Map<String, serde_json::Value>, name: String, rows: usize, cols: usize| {
            let n = rows * cols;
            let vals: Vec<f32> = (0..n).map(|_| xorshift(&mut seed)).collect();
            let bytes = f32_bytes(&vals);
            let start = data.len() as u64;
            data.extend_from_slice(&bytes);
            let end = data.len() as u64;
            header.insert(name, json!({"dtype": "F32", "shape": [rows, cols], "data_offsets": [start, end]}));
        };
        for eid in 0..n_experts {
            push(&mut header, format!("model.layers.0.mlp.experts.{eid}.gate_proj.weight"), moe_inter, hidden);
            push(&mut header, format!("model.layers.0.mlp.experts.{eid}.up_proj.weight"), moe_inter, hidden);
            push(&mut header, format!("model.layers.0.mlp.experts.{eid}.down_proj.weight"), hidden, moe_inter);
        }
        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_bytes);
        out.extend_from_slice(&data);
        fs::write(dir.0.join("model.safetensors"), out).unwrap();
        dir
    }

    /// Like `build_experts_fixture`, but Kimi Linear's real on-disk naming
    /// (`block_sparse_moe.experts.{eid}.{w1,w2,w3}.weight`) and a distinct constant fill per
    /// tensor (`w1`=1.0, `w3`=3.0, `w2`=2.0) instead of random data, so a test can verify each
    /// one lands in the right `ExpertSlot` field (`w1`→`gate`, `w3`→`up`, `w2`→`down`) by VALUE,
    /// not just by shape (`w1`/`w3` share the same `[moe_inter, hidden]` shape, so a shape-only
    /// check couldn't catch the two being swapped).
    fn build_kimi_experts_fixture(name: &str, n_experts: usize, moe_inter: usize, hidden: usize) -> TempDir {
        let dir = TempDir::new(name);
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_string(), json!({"format": "rabbit-test"}));
        let mut data = Vec::new();
        let mut push = |header: &mut serde_json::Map<String, serde_json::Value>, name: String, rows: usize, cols: usize, fill: f32| {
            let bytes = f32_bytes(&vec![fill; rows * cols]);
            let start = data.len() as u64;
            data.extend_from_slice(&bytes);
            let end = data.len() as u64;
            header.insert(name, json!({"dtype": "F32", "shape": [rows, cols], "data_offsets": [start, end]}));
        };
        for eid in 0..n_experts {
            push(&mut header, format!("model.layers.0.block_sparse_moe.experts.{eid}.w1.weight"), moe_inter, hidden, 1.0);
            push(&mut header, format!("model.layers.0.block_sparse_moe.experts.{eid}.w3.weight"), moe_inter, hidden, 3.0);
            push(&mut header, format!("model.layers.0.block_sparse_moe.experts.{eid}.w2.weight"), hidden, moe_inter, 2.0);
        }
        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_bytes);
        out.extend_from_slice(&data);
        fs::write(dir.0.join("model.safetensors"), out).unwrap();
        dir
    }

    fn tiny_cfg(n_experts: i32, moe_inter: i32, hidden: i32) -> Cfg {
        Cfg {
            hidden,
            n_layers: 1,
            n_heads: 1,
            n_experts,
            topk: 1,
            moe_inter,
            dense_inter: 1,
            first_dense: 0,
            q_lora: 1,
            kv_lora: 1,
            qk_nope: 1,
            qk_rope: 2,
            qk_head: 3,
            v_head: 1,
            n_shared: 1,
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
        }
    }

    #[test]
    fn loads_and_caches_experts_with_correct_shapes() {
        let fixture = build_experts_fixture("rabbit_test_ecache_shapes", 3, 4, 6);
        let shards = Shards::open(&fixture.0).unwrap();
        let cfg = tiny_cfg(3, 4, 6);
        let mut cache = ExpertCache::new(8);

        let slot = cache.get_or_load(&shards, &cfg, 0, 1, 32).unwrap();
        assert_eq!(slot.eid, 1);
        assert_eq!(slot.gate.rows, 4);
        assert_eq!(slot.gate.cols, 6);
        assert_eq!(slot.down.rows, 6);
        assert_eq!(slot.down.cols, 4);
        assert_eq!(cache.hits, 0);
        assert_eq!(cache.misses, 1);

        cache.get_or_load(&shards, &cfg, 0, 1, 32).unwrap();
        assert_eq!(cache.hits, 1);
        assert_eq!(cache.misses, 1);
        assert_eq!(cache.len(), 1);
    }

    /// The real point of `ExpertNaming::KimiLinear`: `get_or_load` (the sequential, non-`io_uring`
    /// path, portable to any OS) must read Kimi's real on-disk tensor names
    /// (`block_sparse_moe.experts.{eid}.{w1,w2,w3}.weight`) and land `w1`/`w3`/`w2` in
    /// `gate`/`up`/`down` respectively — checked by VALUE (each tensor has a distinct constant
    /// fill), not just shape, since `w1`/`w3` share a shape and a shape-only check couldn't
    /// catch them being swapped.
    #[test]
    fn kimi_linear_naming_loads_w1_w3_w2_into_gate_up_down() {
        let fixture = build_kimi_experts_fixture("rabbit_test_ecache_kimi_naming", 2, 4, 6);
        let shards = Shards::open(&fixture.0).unwrap();
        let cfg = tiny_cfg(2, 4, 6);
        let mut cache = ExpertCache::for_family(8, ExpertNaming::KimiLinear);

        let slot = cache.get_or_load(&shards, &cfg, 0, 1, 32).unwrap();
        assert_eq!(slot.gate.rows, 4);
        assert_eq!(slot.gate.cols, 6);
        assert_eq!(slot.down.rows, 6);
        assert_eq!(slot.down.cols, 4);
        assert!(qt_values(&slot.gate).iter().all(|&v| (v - 1.0).abs() < 1e-6), "w1 must land in gate");
        assert!(qt_values(&slot.up).iter().all(|&v| (v - 3.0).abs() < 1e-6), "w3 must land in up");
        assert!(qt_values(&slot.down).iter().all(|&v| (v - 2.0).abs() < 1e-6), "w2 must land in down");
    }

    /// `ensure_loaded`'s batched path (the `io_uring` route on Linux, still the sequential
    /// fallback elsewhere) must respect `ExpertNaming` too, not just `get_or_load` — cross-check
    /// against `get_or_load`'s own result on the same Kimi-named fixture, mirroring
    /// `ensure_loaded_matches_sequential_get_or_load_values`'s existing GLM-naming version.
    #[test]
    fn kimi_linear_naming_ensure_loaded_matches_get_or_load() {
        let fixture = build_kimi_experts_fixture("rabbit_test_ecache_kimi_naming_batch", 3, 4, 6);
        let shards = Shards::open(&fixture.0).unwrap();
        let cfg = tiny_cfg(3, 4, 6);

        let mut cache_seq = ExpertCache::for_family(8, ExpertNaming::KimiLinear);
        let eids = [0usize, 1, 2];
        for &eid in &eids {
            cache_seq.get_or_load(&shards, &cfg, 0, eid, 32).unwrap();
        }

        let mut cache_batch = ExpertCache::for_family(8, ExpertNaming::KimiLinear);
        cache_batch.ensure_loaded(&shards, &cfg, 0, &eids, 32).unwrap();

        for &eid in &eids {
            let a = cache_seq.get(eid).unwrap();
            let b = cache_batch.get(eid).unwrap();
            assert_eq!(qt_values(&a.gate), qt_values(&b.gate), "expert {eid} gate");
            assert_eq!(qt_values(&a.up), qt_values(&b.up), "expert {eid} up");
            assert_eq!(qt_values(&a.down), qt_values(&b.down), "expert {eid} down");
        }
    }

    #[test]
    fn evicts_the_least_recently_used_slot_on_overflow() {
        let fixture = build_experts_fixture("rabbit_test_ecache_lru", 4, 4, 6);
        let shards = Shards::open(&fixture.0).unwrap();
        let cfg = tiny_cfg(4, 4, 6);
        let mut cache = ExpertCache::new(2);

        cache.get_or_load(&shards, &cfg, 0, 0, 32).unwrap(); // clock=1, slots=[0]
        cache.get_or_load(&shards, &cfg, 0, 1, 32).unwrap(); // clock=2, slots=[0,1]
        cache.get_or_load(&shards, &cfg, 0, 0, 32).unwrap(); // clock=3, hit -> 0 touched last
        cache.get_or_load(&shards, &cfg, 0, 2, 32).unwrap(); // clock=4, miss, capacity full -> evict 1 (least recently used)

        assert_eq!(cache.misses, 3);
        assert!(cache.slots.iter().any(|s| s.eid == 0));
        assert!(cache.slots.iter().any(|s| s.eid == 2));
        assert!(!cache.slots.iter().any(|s| s.eid == 1), "expert 1 should have been evicted (least recently used)");

        // expert 1 is gone -> re-requesting it is a fresh miss, not a hit.
        let misses_before = cache.misses;
        cache.get_or_load(&shards, &cfg, 0, 1, 32).unwrap();
        assert_eq!(cache.misses, misses_before + 1);
    }

    fn qt_values(t: &QT) -> Vec<f32> {
        (0..t.rows).flat_map(|r| t.row_f32(r)).collect()
    }

    /// The whole point of Fase 8: `ensure_loaded` (batched `io_uring`, Linux) and
    /// `get_or_load` (sequential `pread`) must decode the exact same bytes — same dequantized
    /// weight values, not just matching shapes — for every expert in a batch.
    #[test]
    fn ensure_loaded_matches_sequential_get_or_load_values() {
        let fixture = build_experts_fixture("rabbit_test_ecache_batch_values", 5, 6, 8);
        let shards = Shards::open(&fixture.0).unwrap();
        let cfg = tiny_cfg(5, 6, 8);
        let eids: Vec<usize> = (0..5).collect();

        let mut batch_cache = ExpertCache::new(8);
        batch_cache.ensure_loaded(&shards, &cfg, 0, &eids, 32).unwrap();

        let mut sequential_cache = ExpertCache::new(8);
        for &eid in &eids {
            sequential_cache.get_or_load(&shards, &cfg, 0, eid, 32).unwrap();
        }

        for &eid in &eids {
            let a = batch_cache.get(eid).unwrap();
            let b = sequential_cache.get(eid).unwrap();
            assert_eq!(qt_values(&a.gate), qt_values(&b.gate), "expert {eid} gate_proj");
            assert_eq!(qt_values(&a.up), qt_values(&b.up), "expert {eid} up_proj");
            assert_eq!(qt_values(&a.down), qt_values(&b.down), "expert {eid} down_proj");
        }
    }

    /// `begin_loading` + `finish_loading` (the split that lets `moe.rs` overlap a chunk's disk
    /// read with independent compute) must produce the exact same cache contents, hit/miss
    /// counts, and LRU eviction behavior as `ensure_loaded` — which is now defined as exactly
    /// that pair called back to back, but this pins the contract explicitly rather than relying
    /// on that implementation detail.
    #[test]
    fn begin_then_finish_loading_matches_ensure_loaded() {
        let fixture = build_experts_fixture("rabbit_test_ecache_split_load", 5, 6, 8);
        let shards = Shards::open(&fixture.0).unwrap();
        let cfg = tiny_cfg(5, 6, 8);
        let eids: Vec<usize> = (0..5).collect();

        let mut split_cache = ExpertCache::new(8);
        let pending = split_cache.begin_loading(&shards, &cfg, 0, &eids, 32).unwrap();
        // the whole point: real "independent work" could happen here, between begin and finish.
        split_cache.finish_loading(pending, &shards, &cfg, 0, 32).unwrap();

        let mut plain_cache = ExpertCache::new(8);
        plain_cache.ensure_loaded(&shards, &cfg, 0, &eids, 32).unwrap();

        assert_eq!(split_cache.hits, plain_cache.hits);
        assert_eq!(split_cache.misses, plain_cache.misses);
        assert_eq!(split_cache.len(), plain_cache.len());
        for &eid in &eids {
            let a = split_cache.get(eid).unwrap();
            let b = plain_cache.get(eid).unwrap();
            assert_eq!(qt_values(&a.gate), qt_values(&b.gate), "expert {eid} gate_proj");
            assert_eq!(qt_values(&a.up), qt_values(&b.up), "expert {eid} up_proj");
            assert_eq!(qt_values(&a.down), qt_values(&b.down), "expert {eid} down_proj");
        }

        // a batch that's entirely cache hits must short-circuit through `LoadKind::Nothing`
        // cleanly, not touch disk or misscount.
        let pending2 = split_cache.begin_loading(&shards, &cfg, 0, &eids, 32).unwrap();
        split_cache.finish_loading(pending2, &shards, &cfg, 0, 32).unwrap();
        assert_eq!(split_cache.hits, plain_cache.hits + eids.len() as u64);
        assert_eq!(split_cache.misses, plain_cache.misses);
    }

    #[test]
    fn ensure_loaded_counts_hits_and_misses_and_dedupes_repeated_ids() {
        let fixture = build_experts_fixture("rabbit_test_ecache_batch_counts", 3, 4, 6);
        let shards = Shards::open(&fixture.0).unwrap();
        let cfg = tiny_cfg(3, 4, 6);
        let mut cache = ExpertCache::new(8);

        // duplicate id 0 within the same (empty-cache) batch must still count as ONE miss,
        // not two — moe.rs's own batch-union step already dedupes before calling this, so
        // this is a defensive guarantee on the API itself, not something real callers hit.
        cache.ensure_loaded(&shards, &cfg, 0, &[0, 1, 0], 32).unwrap();
        assert_eq!(cache.misses, 2);
        assert_eq!(cache.hits, 0); // nothing was cached yet when this batch started
        assert_eq!(cache.len(), 2);

        cache.ensure_loaded(&shards, &cfg, 0, &[0, 1, 2], 32).unwrap();
        assert_eq!(cache.misses, 3); // 2 was the only new one
        assert_eq!(cache.hits, 2); // 0 and 1 from the batch above, hit again
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn ensure_loaded_evicts_lru_same_as_get_or_load() {
        let fixture = build_experts_fixture("rabbit_test_ecache_batch_lru", 4, 4, 6);
        let shards = Shards::open(&fixture.0).unwrap();
        let cfg = tiny_cfg(4, 4, 6);
        let mut cache = ExpertCache::new(2);

        cache.ensure_loaded(&shards, &cfg, 0, &[0, 1], 32).unwrap(); // slots=[0,1]
        cache.get(0).unwrap(); // touch is via ensure_loaded's hit path, not this lookup
        cache.ensure_loaded(&shards, &cfg, 0, &[0], 32).unwrap(); // hit -> 0 touched last
        cache.ensure_loaded(&shards, &cfg, 0, &[2], 32).unwrap(); // miss at capacity -> evict 1

        assert!(cache.get(0).is_some());
        assert!(cache.get(2).is_some());
        assert!(cache.get(1).is_none(), "expert 1 should have been evicted (least recently used)");
    }

    /// Both `ensure_loaded` (the `io_uring` batch path on Linux) and `get_or_load`
    /// (sequential, via `qt_load`) must wrap a `.qs`-backed expert tensor as-is, not
    /// requantize it — hand-picked bytes an actual quantization pass would be astronomically
    /// unlikely to reproduce by coincidence, so an exact match is strong evidence of that.
    #[test]
    fn qs_backed_expert_tensors_pass_through_unquantized_on_both_paths() {
        let dir = TempDir::new("rabbit_test_ecache_qs_passthrough");
        let moe_inter = 3;
        let hidden = 5;
        // int8: byte count == moe_inter*hidden for gate/up, hidden*moe_inter for down.
        let gate_bytes: Vec<u8> = (0..moe_inter * hidden).map(|i| (i * 37 + 11) as u8).collect();
        let gate_scale = vec![0.25f32; moe_inter];
        let up_bytes: Vec<u8> = (0..moe_inter * hidden).map(|i| (i * 53 + 5) as u8).collect();
        let up_scale = vec![0.75f32; moe_inter];
        let down_bytes: Vec<u8> = (0..hidden * moe_inter).map(|i| (i * 61 + 3) as u8).collect();
        let down_scale = vec![1.5f32; hidden];

        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_string(), json!({"format": "rabbit-test"}));
        let mut data = Vec::new();
        let mut push = |header: &mut serde_json::Map<String, serde_json::Value>, name: String, dtype: &str, shape: Vec<usize>, bytes: Vec<u8>| {
            let start = data.len() as u64;
            data.extend_from_slice(&bytes);
            let end = data.len() as u64;
            header.insert(name, json!({"dtype": dtype, "shape": shape, "data_offsets": [start, end]}));
        };
        let eid = 0;
        for (suf, rows, cols, bytes, scale) in [
            ("gate_proj", moe_inter, hidden, &gate_bytes, &gate_scale),
            ("up_proj", moe_inter, hidden, &up_bytes, &up_scale),
            ("down_proj", hidden, moe_inter, &down_bytes, &down_scale),
        ] {
            let name = format!("model.layers.0.mlp.experts.{eid}.{suf}.weight");
            push(&mut header, name.clone(), "U8", vec![rows * cols], bytes.clone());
            push(&mut header, format!("{name}.qs"), "F32", vec![rows], f32_bytes(scale));
        }
        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_bytes);
        out.extend_from_slice(&data);
        fs::write(dir.0.join("model.safetensors"), out).unwrap();

        let shards = Shards::open(&dir.0).unwrap();
        let cfg = tiny_cfg(1, moe_inter as i32, hidden as i32);

        let check = |slot: &ExpertSlot| {
            let crate::quant::QTKind::I8 { data, scale } = &slot.gate.kind else { panic!("expected I8 gate") };
            assert_eq!(data, &gate_bytes.iter().map(|&b| b as i8).collect::<Vec<i8>>());
            assert_eq!(scale, &gate_scale);
            let crate::quant::QTKind::I8 { data, scale } = &slot.up.kind else { panic!("expected I8 up") };
            assert_eq!(data, &up_bytes.iter().map(|&b| b as i8).collect::<Vec<i8>>());
            assert_eq!(scale, &up_scale);
            let crate::quant::QTKind::I8 { data, scale } = &slot.down.kind else { panic!("expected I8 down") };
            assert_eq!(data, &down_bytes.iter().map(|&b| b as i8).collect::<Vec<i8>>());
            assert_eq!(scale, &down_scale);
        };

        let mut batch_cache = ExpertCache::new(4);
        batch_cache.ensure_loaded(&shards, &cfg, 0, &[eid], 8).unwrap();
        check(batch_cache.get(eid).unwrap());

        let mut sequential_cache = ExpertCache::new(4);
        sequential_cache.get_or_load(&shards, &cfg, 0, eid, 8).unwrap();
        check(sequential_cache.get(eid).unwrap());
    }

    /// Both `ensure_loaded` (`io_uring`) and `get_or_load` (sequential, via `qt_load` ->
    /// `read_f32`'s automatic FP8 handling) must apply the same FP8 block-scale dequant to a
    /// raw `F8_E4M3` expert tensor and its `{name}_scale_inv` sidecar — this is the regression
    /// test for the gap where `io_uring`'s `qt_from_raw` used to call `Shards::decode_f32`
    /// directly, which decodes FP8 bytes WITHOUT applying any scale at all.
    #[test]
    fn fp8_backed_expert_tensors_dequantize_identically_on_both_paths() {
        let dir = TempDir::new("rabbit_test_ecache_fp8_parity");
        let moe_inter = 3;
        let hidden = 5;
        // 0x38 = 1.0, 0xB8 = -1.0 in e4m3fn (see safetensors.rs's f8e4m3 tests) — every
        // element decodes to +-1.0 before scaling, so a wrong (unscaled) path is trivially
        // distinguishable from a correctly-scaled one.
        let fp8_pattern = |n: usize| -> Vec<u8> { (0..n).map(|i| if i % 2 == 0 { 0x38 } else { 0xB8 }).collect() };
        // every tensor here is well inside one 128x128 block -> exactly one scale value.
        let gate_bytes = fp8_pattern(moe_inter * hidden);
        let gate_scale = vec![2.0f32];
        let up_bytes = fp8_pattern(moe_inter * hidden);
        let up_scale = vec![3.0f32];
        let down_bytes = fp8_pattern(hidden * moe_inter);
        let down_scale = vec![4.0f32];

        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_string(), json!({"format": "rabbit-test"}));
        let mut data = Vec::new();
        let mut push = |header: &mut serde_json::Map<String, serde_json::Value>, name: String, dtype: &str, shape: Vec<usize>, bytes: Vec<u8>| {
            let start = data.len() as u64;
            data.extend_from_slice(&bytes);
            let end = data.len() as u64;
            header.insert(name, json!({"dtype": dtype, "shape": shape, "data_offsets": [start, end]}));
        };
        let eid = 0;
        for (suf, rows, cols, bytes, scale) in [
            ("gate_proj", moe_inter, hidden, &gate_bytes, &gate_scale),
            ("up_proj", moe_inter, hidden, &up_bytes, &up_scale),
            ("down_proj", hidden, moe_inter, &down_bytes, &down_scale),
        ] {
            let name = format!("model.layers.0.mlp.experts.{eid}.{suf}.weight");
            push(&mut header, name.clone(), "F8_E4M3", vec![rows, cols], bytes.clone());
            push(&mut header, format!("{name}_scale_inv"), "F32", vec![1, 1], f32_bytes(scale));
        }
        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_bytes);
        out.extend_from_slice(&data);
        fs::write(dir.0.join("model.safetensors"), out).unwrap();

        let shards = Shards::open(&dir.0).unwrap();
        let cfg = tiny_cfg(1, moe_inter as i32, hidden as i32);

        let mut batch_cache = ExpertCache::new(4);
        batch_cache.ensure_loaded(&shards, &cfg, 0, &[eid], 32).unwrap();
        let batch_slot = batch_cache.get(eid).unwrap();

        let mut sequential_cache = ExpertCache::new(4);
        sequential_cache.get_or_load(&shards, &cfg, 0, eid, 32).unwrap();
        let sequential_slot = sequential_cache.get(eid).unwrap();

        assert_eq!(qt_values(&batch_slot.gate), qt_values(&sequential_slot.gate), "gate_proj");
        assert_eq!(qt_values(&batch_slot.up), qt_values(&sequential_slot.up), "up_proj");
        assert_eq!(qt_values(&batch_slot.down), qt_values(&sequential_slot.down), "down_proj");

        // and the scale was actually applied, not silently skipped -> magnitude ~2.0/3.0/4.0,
        // not ~1.0 (which is what the pre-fix bug's unscaled decode would have produced).
        let gate_vals = qt_values(&batch_slot.gate);
        assert!(gate_vals.iter().any(|&v| v.abs() > 1.5), "gate scale (2.0) doesn't look applied: {gate_vals:?}");
    }

    #[test]
    fn record_selection_and_usage_counts_round_trip() {
        let mut cache = ExpertCache::new(4);
        cache.record_selection(3);
        cache.record_selection(3);
        cache.record_selection(7);
        let counts: std::collections::HashMap<usize, u64> = cache.usage_counts().collect();
        assert_eq!(counts.get(&3), Some(&2));
        assert_eq!(counts.get(&7), Some(&1));
    }

    #[test]
    fn seed_usage_sets_counters_seen_by_usage_counts() {
        let mut cache = ExpertCache::new(4);
        cache.seed_usage(vec![(1usize, 100u64), (2, 200)].into_iter());
        cache.record_selection(2); // live count must land on top of the seeded value, not replace it via a fresh insert
        let counts: std::collections::HashMap<usize, u64> = cache.usage_counts().collect();
        assert_eq!(counts.get(&1), Some(&100));
        assert_eq!(counts.get(&2), Some(&201));
    }

    #[test]
    fn a_pin_candidate_is_not_loaded_until_generation_actually_needs_it() {
        let mut cache = ExpertCache::new(4);

        cache.mark_pin_candidates(std::iter::once(1usize));
        assert!(!cache.is_pinned(1), "marking a candidate must not eagerly load anything");
        assert_eq!(cache.pinned_len(), 0);
        assert_eq!(cache.misses, 0, "no disk I/O should have happened yet");
        assert!(cache.get(1).is_none());
    }

    #[test]
    fn a_pin_candidate_gets_promoted_on_its_first_real_load_and_counted_as_a_hit_after() {
        let fixture = build_experts_fixture("rabbit_test_ecache_pin_promote", 3, 4, 6);
        let shards = Shards::open(&fixture.0).unwrap();
        let cfg = tiny_cfg(3, 4, 6);
        let mut cache = ExpertCache::new(4);

        cache.mark_pin_candidates(std::iter::once(1usize));
        cache.get_or_load(&shards, &cfg, 0, 1, 32).unwrap(); // the FIRST load: a real miss, triggers promotion
        assert_eq!(cache.misses, 1, "the first load of a candidate is still a genuine miss");
        assert!(cache.is_pinned(1));
        assert_eq!(cache.pinned_len(), 1);
        assert_eq!(cache.len(), 0, "a promoted expert must not also land in the ordinary LRU slots");

        let slot = cache.get(1).unwrap();
        assert_eq!(slot.eid, 1);

        // every subsequent lookup must be a hit through the pinned tier, never another miss.
        cache.begin_loading(&shards, &cfg, 0, &[1], 32).unwrap();
        assert_eq!(cache.hits, 1);
        assert_eq!(cache.misses, 1);

        cache.get_or_load(&shards, &cfg, 0, 1, 32).unwrap();
        assert_eq!(cache.hits, 2);
        assert_eq!(cache.misses, 1);
    }

    #[test]
    fn a_promoted_expert_survives_lru_pressure_at_capacity_one() {
        let fixture = build_experts_fixture("rabbit_test_ecache_pin_survives_lru", 4, 4, 6);
        let shards = Shards::open(&fixture.0).unwrap();
        let cfg = tiny_cfg(4, 4, 6);
        let mut cache = ExpertCache::new(1); // capacity 1: every OTHER load evicts the LRU slot

        cache.mark_pin_candidates(std::iter::once(0usize));
        cache.get_or_load(&shards, &cfg, 0, 0, 32).unwrap(); // promotes 0 into the pinned tier
        // hammer the tiny LRU cache with unrelated experts -- must never disturb the pin.
        for eid in [1, 2, 3, 1, 2, 3] {
            cache.get_or_load(&shards, &cfg, 0, eid, 32).unwrap();
        }
        assert!(cache.is_pinned(0));
        assert_eq!(cache.pinned_len(), 1);
        assert!(cache.get(0).is_some(), "pinned expert 0 must still resolve after LRU churn");
    }

    #[test]
    fn marking_a_candidate_after_it_is_already_resident_does_not_retroactively_promote_it() {
        // documents a real, intentional limitation of the lazy design: mark_pin_candidates
        // must run before the FIRST load of a given id (warm_start always does, in practice).
        let fixture = build_experts_fixture("rabbit_test_ecache_pin_too_late", 2, 4, 6);
        let shards = Shards::open(&fixture.0).unwrap();
        let cfg = tiny_cfg(2, 4, 6);
        let mut cache = ExpertCache::new(4);

        cache.get_or_load(&shards, &cfg, 0, 0, 32).unwrap(); // lands in the ordinary LRU slots
        cache.mark_pin_candidates(std::iter::once(0usize)); // too late for THIS residency
        assert!(!cache.is_pinned(0), "a hit never re-triggers insert_or_pin, so no promotion happens here");
        assert!(cache.get(0).is_some(), "it's still resolvable via the ordinary LRU slot though");
    }

    #[test]
    fn marking_the_same_candidate_twice_does_not_duplicate_the_pinned_slot() {
        let fixture = build_experts_fixture("rabbit_test_ecache_pin_dedup", 2, 4, 6);
        let shards = Shards::open(&fixture.0).unwrap();
        let cfg = tiny_cfg(2, 4, 6);
        let mut cache = ExpertCache::new(4);

        cache.mark_pin_candidates(std::iter::once(0usize));
        cache.mark_pin_candidates(std::iter::once(0usize));
        cache.get_or_load(&shards, &cfg, 0, 0, 32).unwrap();
        assert_eq!(cache.pinned_len(), 1);
    }
}
