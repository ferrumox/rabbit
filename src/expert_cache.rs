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
    /// Like `Glm52`, but `gate_proj`/`up_proj` have been pre-fused into one
    /// `gate_up_proj.weight` tensor (`[2*moe_inter, hidden]`, gate's rows first, then up's) by
    /// a converter — see `bin/fuse_gate_up.rs`. Auto-detected per checkpoint at `Model::load`
    /// time (`glm52::model::detect_fused_gate_up`) — a checkpoint is either fully fused or not
    /// at all, never a mix, so this and `Glm52` are mutually exclusive for a given load.
    Glm52FusedGateUp,
    /// `model.layers.{layer}.block_sparse_moe.experts.{eid}.{w1,w2,w3}.weight` — `w1` is the
    /// gate projection, `w2` is down, `w3` is up (confirmed via `KimiBlockSparseMLP`'s own code
    /// comments in the real `modeling_kimi.py`, not guessed; see `kimi_linear::model`'s doc).
    KimiLinear,
    /// Like `KimiLinear`, but under K3's real `language_model.` prefix (K3's actual published
    /// checkpoint is the full multimodal `KimiK3ForConditionalGeneration`, which wraps the text
    /// backbone as `self.language_model = KimiLinearForCausalLM(...)` — confirmed against the
    /// real `model.safetensors.index.json`, fetched 2026-07-27: e.g.
    /// `language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight...`).
    ///
    /// **This naming alone is NOT sufficient to load K3's real checkpoint's routed experts** —
    /// their `w1`/`w2`/`w3` tensors are natively MXFP4-quantized on disk (see `KimiK3Mxfp4`).
    /// This variant is correct for any K3-shaped checkpoint whose experts AREN'T MXFP4 (e.g. a
    /// future rabbit-side int4 re-conversion, or a synthetic test fixture using plain F32
    /// tensors, which is what most of this crate's own `kimi_k3::generate`/`kv_session` tests
    /// do).
    KimiK3,
    /// K3's REAL published checkpoint: same `language_model.`-prefixed tensor tree as `KimiK3`,
    /// but each routed expert's `w1`/`w2`/`w3` is stored as a `{name}.weight_packed` +
    /// `{name}.weight_scale` U8 pair (OCP Microscaling MXFP4 — `compressed-tensors`'
    /// `mxfp4-pack-quantized` format, `group_size: 32`, confirmed against the real
    /// `config.json`'s `quantization_config`) instead of one plain `{name}.weight` float tensor.
    /// `tensor_specs` returns the BASE name (no `.weight` suffix) for this variant —
    /// `qt_load_mxfp4` appends `.weight_packed`/`.weight_scale` itself. Picked automatically by
    /// `kimi_k3::generate::ExpertCaches::new` from `cfg.mxfp4_experts`.
    ///
    /// Rabbit's own `QTKind::MxFp4` byte layout (E2M1 magnitude table, low-nibble-first packing,
    /// E8M0 bias-127 block scale, 32-element blocks) was checked bit-for-bit against
    /// `compressed-tensors`' real `pack_fp4_to_uint8`/`generate_mx_scales` source (not guessed)
    /// and matches exactly, so `qt_load_mxfp4` reads the on-disk bytes straight into
    /// `QTKind::MxFp4` with no transcoding step.
    KimiK3Mxfp4,
}

impl ExpertNaming {
    /// How many `.safetensors` tensors this naming reads per expert — `submit_batch`'s ring
    /// sizing and `complete_batch_streaming`'s per-expert read-count bookkeeping both key off
    /// this instead of a hardcoded `3`.
    fn tensor_count(&self) -> usize {
        match self {
            ExpertNaming::Glm52FusedGateUp => 2,
            ExpertNaming::Glm52 | ExpertNaming::KimiLinear | ExpertNaming::KimiK3 | ExpertNaming::KimiK3Mxfp4 => 3,
        }
    }

    /// `true` iff this naming's weights are MXFP4-packed on disk (`{name}.weight_packed`/
    /// `{name}.weight_scale`) rather than a plain `{name}.weight` tensor. Used by `load_expert`'s
    /// synchronous path to pick `qt_load_mxfp4` over `qt_load`, and by `uring_load::submit_batch`
    /// to read the packed+scale pair instead of one plain tensor (see `Req::mxfp4_scale`'s doc) —
    /// `io_uring`-batched MXFP4 used to be out of scope (every load fell back to the plain
    /// synchronous path) until it was built to close the gap once disk I/O, not compute, became
    /// the dominant cost per token (see `PERFORMANCE_KIMI_K3.md`).
    fn is_mxfp4(&self) -> bool {
        matches!(self, ExpertNaming::KimiK3Mxfp4)
    }

    /// Returns `(tensor_name, rows, cols)` in read order — every caller in this file (the
    /// `io_uring` batch path and the sequential fallback alike) indexes reads positionally in
    /// this same order (see `complete_batch_streaming`'s `base+0/1/...`), so this is the one
    /// place that ordering has to be gotten right.
    fn tensor_specs(&self, layer: usize, eid: usize, moe_inter: usize, hidden: usize) -> Vec<(String, usize, usize)> {
        match self {
            ExpertNaming::Glm52 => {
                let p = |suf: &str| format!("model.layers.{layer}.mlp.experts.{eid}.{suf}.weight");
                vec![(p("gate_proj"), moe_inter, hidden), (p("up_proj"), moe_inter, hidden), (p("down_proj"), hidden, moe_inter)]
            }
            ExpertNaming::Glm52FusedGateUp => {
                let p = |suf: &str| format!("model.layers.{layer}.mlp.experts.{eid}.{suf}.weight");
                vec![(p("gate_up_proj"), 2 * moe_inter, hidden), (p("down_proj"), hidden, moe_inter)]
            }
            ExpertNaming::KimiLinear => {
                let p = |suf: &str| format!("model.layers.{layer}.block_sparse_moe.experts.{eid}.{suf}.weight");
                vec![(p("w1"), moe_inter, hidden), (p("w3"), moe_inter, hidden), (p("w2"), hidden, moe_inter)]
            }
            ExpertNaming::KimiK3 => {
                let p = |suf: &str| format!("language_model.model.layers.{layer}.block_sparse_moe.experts.{eid}.{suf}.weight");
                vec![(p("w1"), moe_inter, hidden), (p("w3"), moe_inter, hidden), (p("w2"), hidden, moe_inter)]
            }
            ExpertNaming::KimiK3Mxfp4 => {
                // No `.weight` suffix — `qt_load_mxfp4` appends `.weight_packed`/`.weight_scale`
                // itself, since this naming has no single `.weight` tensor to point at.
                let p = |suf: &str| format!("language_model.model.layers.{layer}.block_sparse_moe.experts.{eid}.{suf}");
                vec![(p("w1"), moe_inter, hidden), (p("w3"), moe_inter, hidden), (p("w2"), hidden, moe_inter)]
            }
        }
    }
}

#[cfg(target_os = "linux")]
mod uring_load {
    use super::{Cfg, ExpertNaming, ExpertSlot, GateUp, ModelError, QT, mxfp4_from_raw};
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
        /// `Some(index into Pending::scale_bufs)` for `ExpertNaming::KimiK3Mxfp4` (mutually
        /// exclusive with `packed_scale`/`fp8_scale` — MXFP4 is never also `.qs`-packed or raw
        /// FP8): the `{name}.weight_scale` E8M0 sidecar. Unlike the other two scale kinds, this
        /// one is NOT decoded to `f32` on completion — it's raw bytes (one E8M0 exponent byte per
        /// 32-element block), used verbatim by `mxfp4_from_raw`, same as the synchronous
        /// `qt_load_mxfp4` path. Also unlike the other two, `loc` itself isn't a plain float
        /// tensor to decode either — it's `{name}.weight_packed`, also used verbatim.
        mxfp4_scale: Option<usize>,
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

    /// Creates the persistent ring sized for `io_batch_size` experts/round (see
    /// `ExpertCache::io_batch_size`'s doc — NOT `capacity`, since a round can submit more reads
    /// than the resident cache holds onto afterward): `naming.tensor_count()` tensors/expert (3,
    /// or 2 for a fused-gate-up checkpoint), plus up to one scale-sidecar read per tensor (`.qs`
    /// packed scale, FP8 `_scale_inv`, or — always, not just "up to", for `KimiK3Mxfp4` — the
    /// `.weight_scale` E8M0 sidecar; the three are mutually exclusive per tensor, so this sizing
    /// is a worst-case upper bound in the first two cases and exact in the MXFP4 case) submitted
    /// in the SAME batch instead of a synchronous pre-read — see `submit_batch`'s doc for why.
    /// Returns `None` if `io_uring` isn't usable on this host (the `io_uring_disabled` sysctl, a
    /// seccomp profile blocking the syscalls, ...) — every `submit_batch` call then falls back
    /// to a synchronous load, same as a hard per-call failure would.
    pub(super) fn new_ring(io_batch_size: usize, naming: ExpertNaming) -> Option<Ring> {
        Ring::new(io_batch_size * naming.tensor_count() * 2).ok()
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

    /// Submits reads for every tensor in `misses` (`naming.tensor_count()`/expert) — plus, in
    /// the SAME `io_uring` round, any `.qs`/FP8 scale sidecar those tensors need — WITHOUT
    /// waiting for completion. Locating a sidecar (`shards.tensor_location`) is a plain
    /// in-memory header lookup, not I/O, so nothing here blocks; only the actual byte reads are
    /// deferred to the ring. Returns `Ok(None)` — not an error — when the ring is unavailable,
    /// the batch doesn't fit in one submission round, or the submission itself fails; every one
    /// of these just means "the caller should fall back to a synchronous load for this batch,"
    /// same as a hard `load_batch` failure always meant before this split existed.
    pub(super) fn submit_batch(ring: &mut Option<Ring>, shards: &Shards, cfg: &Cfg, layer: usize, misses: &[usize], naming: ExpertNaming) -> Result<Option<Pending>, ModelError> {
        let Some(r) = ring.as_mut() else { return Ok(None) };

        let i = cfg.moe_inter as usize;
        let d = cfg.hidden as usize;

        let mut reqs = Vec::with_capacity(misses.len() * naming.tensor_count());
        let mut scale_locs: Vec<TensorLocation> = Vec::new();
        for &eid in misses {
            let specs = naming.tensor_specs(layer, eid, i, d);
            for (name, rows, cols) in &specs {
                if naming.is_mxfp4() {
                    // `name` is the BASE name here (see `ExpertNaming::tensor_specs`'s doc) —
                    // unlike every other naming, there's no plain `{name}` tensor to look up at
                    // all, and the scale sidecar always exists (not a maybe-present `.qs`/FP8
                    // check like the branches below).
                    let packed_name = format!("{name}.weight_packed");
                    let scale_name = format!("{name}.weight_scale");
                    let loc = shards.tensor_location(&packed_name).ok_or(ModelError::Safetensors(SafetensorsError::MissingTensor(packed_name)))?;
                    let sloc = shards.tensor_location(&scale_name).ok_or(ModelError::Safetensors(SafetensorsError::MissingTensor(scale_name)))?;
                    let mxfp4_scale = Some(scale_locs.len());
                    scale_locs.push(sloc);
                    reqs.push(Req { rows: *rows, cols: *cols, loc, packed_scale: None, fp8_scale: None, mxfp4_scale });
                    continue;
                }

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
                reqs.push(Req { rows: *rows, cols: *cols, loc, packed_scale, fp8_scale, mxfp4_scale: None });
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
    pub(super) fn complete_batch_streaming(
        ring: &mut Ring,
        pending: Pending,
        bits: u8,
        group_size: usize,
        used: u64,
        naming: ExpertNaming,
        mut on_slot: impl FnMut(&ExpertSlot),
    ) -> Result<(Vec<ExpertSlot>, u64), std::io::Error> {
        let tensors = naming.tensor_count();
        let Pending { reqs, mut bufs, mut scale_bufs, eids } = pending;
        let total = reqs.len() + scale_bufs.len();
        let n_experts = eids.len();

        // `scale_owner[s]` = which expert (index into `eids`) scale-sidecar read `s` belongs
        // to; `needed[e]` = how many of expert `e`'s reads (`tensors` main + 0-`tensors` scale)
        // must land before it's fully decodable. `is_mxfp4_scale[s]` marks the ones that must
        // stay raw bytes rather than get decoded to `f32` below (E8M0 exponent bytes, not
        // IEEE-754 floats — see `Req::mxfp4_scale`'s doc).
        let mut scale_owner = vec![usize::MAX; scale_bufs.len()];
        let mut is_mxfp4_scale = vec![false; scale_bufs.len()];
        let mut needed = vec![tensors; n_experts];
        for (local, req) in reqs.iter().enumerate() {
            let owner = local / tensors;
            if let Some(idx) = req.packed_scale {
                scale_owner[idx] = owner;
                needed[owner] += 1;
            }
            if let Some(idx) = req.fp8_scale {
                scale_owner[idx] = owner;
                needed[owner] += 1;
            }
            if let Some(idx) = req.mxfp4_scale {
                scale_owner[idx] = owner;
                needed[owner] += 1;
                is_mxfp4_scale[idx] = true;
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
                    owner = idx / tensors;
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
                    // reinterpretation, no I/O left to do) rather than waiting for the round —
                    // except MXFP4's E8M0 sidecar, which stays raw bytes (see
                    // `Req::mxfp4_scale`'s doc); `qt_from_raw` grabs it straight out of
                    // `scale_bufs` below instead of through `scales`.
                    if !is_mxfp4_scale[sidx] {
                        scales[sidx] = Shards::decode_f32(buf, DType::F32);
                    }
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
                    let base = owner * tensors;
                    let mut take = |k: usize| -> Result<QT, std::io::Error> {
                        let req = &reqs[base + k];
                        let raw = std::mem::take(&mut bufs[base + k]);
                        if let Some(sidx) = req.mxfp4_scale {
                            let block_scale = std::mem::take(&mut scale_bufs[sidx]);
                            return mxfp4_from_raw(raw, block_scale, req.rows, req.cols, "expert tensor (weight_packed)", "expert tensor (weight_scale)")
                                .map_err(|e| std::io::Error::other(e.to_string()));
                        }
                        qt_from_raw(raw, req, bits, group_size, &scales).map_err(|e| std::io::Error::other(e.to_string()))
                    };
                    let slot = if tensors == 2 {
                        let gate_up = take(0)?;
                        let down = take(1)?;
                        ExpertSlot { eid: eids[owner], gate_up: GateUp::Fused { gate_up }, down, used }
                    } else {
                        let gate = take(0)?;
                        let up = take(1)?;
                        let down = take(2)?;
                        ExpertSlot { eid: eids[owner], gate_up: GateUp::Separate { gate, up }, down, used }
                    };
                    on_slot(&slot);
                    out.push(slot);
                }
            }
        }
        let io_wait_nanos = wait_t.elapsed().as_nanos() as u64;
        Ok((out, io_wait_nanos))
    }

    fn qt_from_raw(raw: Vec<u8>, req: &Req, bits: u8, group_size: usize, scales: &[Vec<f32>]) -> Result<QT, ModelError> {
        if let Some(idx) = req.packed_scale {
            return QT::from_packed_grouped(req.rows, req.cols, bits, raw, scales[idx].clone(), group_size)
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

/// The gate/up half of an expert's FFN weights — either the original two separate tensors, or
/// (see `bin/fuse_gate_up.rs`) one pre-fused `[2*moe_inter, hidden]` tensor read and matmul'd in
/// a single shot. `moe.rs`'s `apply_single_expert` matches on this to pick the right compute
/// path; every other consumer of an `ExpertSlot` (the cache/LRU/pinning machinery here) doesn't
/// care which variant it holds.
pub enum GateUp {
    Separate { gate: QT, up: QT },
    Fused { gate_up: QT },
}

pub struct ExpertSlot {
    pub eid: usize,
    pub gate_up: GateUp,
    pub down: QT,
    used: u64,
}

#[cfg(test)]
impl ExpertSlot {
    /// Test-only constructor — `used` stays private in production code (only `expert_cache.rs`
    /// itself sets it, via the LRU/pin bookkeeping), but `moe.rs`'s own tests need to build an
    /// `ExpertSlot` directly to compare `GateUp::Fused` against `GateUp::Separate` for the same
    /// underlying weights, without going through a real (or fixture) `.safetensors` load.
    pub(crate) fn new_for_test(eid: usize, gate_up: GateUp, down: QT) -> ExpertSlot {
        ExpertSlot { eid, gate_up, down, used: 0 }
    }

    /// Test-only convenience accessors for the pre-fusion `Separate` shape — every existing
    /// test fixture builds an unfused checkpoint, so these panic (loudly, not silently wrong)
    /// if ever called against a `Fused` slot, same as indexing past the end of a `Vec` would.
    fn gate(&self) -> &QT {
        match &self.gate_up {
            GateUp::Separate { gate, .. } => gate,
            GateUp::Fused { .. } => panic!("expected GateUp::Separate, got Fused"),
        }
    }
    fn up(&self) -> &QT {
        match &self.gate_up {
            GateUp::Separate { up, .. } => up,
            GateUp::Fused { .. } => panic!("expected GateUp::Separate, got Fused"),
        }
    }
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
    /// How many experts' reads `moe.rs` submits together in ONE `io_uring` round — see
    /// `io_batch_size()`'s doc for why this is a SEPARATE concept from `capacity` (resident
    /// memory) and defaults to matching it.
    io_batch_size: usize,
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

    /// How many experts `moe.rs` should submit together in one `io_uring` round (its own
    /// `chunk_size`) — a DIFFERENT concept from `capacity`, added 2026-07-29 after a real
    /// checkpoint measurement showed the two being permanently coupled (the original design: one
    /// round = one `capacity`-sized chunk) has a real cost once `capacity` gets clamped small for
    /// memory safety (see `PERFORMANCE_KIMI_K3.md`'s auto-clamp story): a small `capacity` also
    /// forces small, less-concurrent `io_uring` rounds, and this project's OWN prior investigation
    /// (`PERFORMANCE.md`'s "Lead 2") found MORE concurrent scattered reads measurably faster on
    /// this drive, not fewer. Decoupling lets a round submit MORE reads than the resident cache
    /// can hold onto afterward — correct because `finish_loading_streaming`'s `on_slot` callback
    /// already consumes each expert's data the moment ITS OWN read lands, strictly BEFORE that
    /// batch's insertion-and-eviction pass runs (see `finish_loading_streaming`'s body) — so a
    /// same-batch eviction only means "this expert won't stay cached long," never "its compute
    /// used stale/missing data." Defaults to `capacity` (today's original coupled behavior)
    /// unless `for_family_with_io_batch` sets it explicitly — see that constructor's doc.
    pub fn io_batch_size(&self) -> usize {
        self.io_batch_size
    }

    pub fn new(capacity: usize) -> ExpertCache {
        Self::for_family(capacity, ExpertNaming::Glm52)
    }

    /// Like `new`, but for a checkpoint whose routed experts use a different on-disk
    /// tensor-name convention (see `ExpertNaming`) — e.g. Kimi Linear's
    /// `block_sparse_moe.experts.{eid}.{w1,w2,w3}` instead of GLM-5.2's
    /// `mlp.experts.{eid}.{gate_proj,up_proj,down_proj}`. `io_batch_size` defaults to `capacity`
    /// (coupled, the original behavior) — use `for_family_with_io_batch` to set it independently.
    pub fn for_family(capacity: usize, naming: ExpertNaming) -> ExpertCache {
        Self::for_family_with_io_batch(capacity, capacity, naming)
    }

    /// Like `for_family`, but `io_batch_size` (see that method's doc) is set independently of
    /// `capacity` instead of defaulting to match it — the real entry point `--io-batch-size`
    /// wires up when explicitly passed; every other caller (tests, examples, `for_family` itself)
    /// keeps using the coupled default. The `io_uring` ring is sized off `io_batch_size` (it must
    /// fit whatever `moe.rs` actually submits per round), NOT `capacity`.
    pub fn for_family_with_io_batch(capacity: usize, io_batch_size: usize, naming: ExpertNaming) -> ExpertCache {
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
            ring: uring_load::new_ring(io_batch_size, naming),
            naming,
            io_batch_size,
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
                let result = uring_load::complete_batch_streaming(ring, p, bits, cfg.group_size as usize, self.clock, self.naming, &mut on_slot);
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
    let gs = cfg.group_size as usize;
    let specs = naming.tensor_specs(layer, eid, i, d);
    let load_one = |name: &str, rows: usize, cols: usize| -> Result<QT, ModelError> {
        if naming.is_mxfp4() { qt_load_mxfp4(shards, name, rows, cols) } else { qt_load(shards, gs, name, rows, cols, bits) }
    };
    let (gate_up, down) = if specs.len() == 2 {
        let gate_up = load_one(&specs[0].0, specs[0].1, specs[0].2)?;
        let down = load_one(&specs[1].0, specs[1].1, specs[1].2)?;
        (GateUp::Fused { gate_up }, down)
    } else {
        let gate = load_one(&specs[0].0, specs[0].1, specs[0].2)?;
        let up = load_one(&specs[1].0, specs[1].1, specs[1].2)?;
        let down = load_one(&specs[2].0, specs[2].1, specs[2].2)?;
        (GateUp::Separate { gate, up }, down)
    };
    Ok(ExpertSlot { eid, gate_up, down, used })
}

/// Fraction of real available RAM (`/proc/meminfo`'s `MemAvailable`) used as the auto
/// `--expert-cache` safety budget. Replaces this fix's original fixed 24GB constant (2026-07-28):
/// a real controlled A/B on the machine that hit the original OOM found the fixed budget was
/// needlessly conservative THERE (auto-capacity landed at 4, ~8.5% hit rate) — but a fixed number
/// picked for one machine is exactly wrong-shaped for this problem: too small wastes real headroom
/// on a bigger machine, too large risks the same OOM/swap on a smaller one. Reading actual
/// available memory generalizes correctly either way. 40% deliberately leaves the other 60% of
/// whatever's free RIGHT NOW (at process startup, before this checkpoint's own base-resident
/// weights, KV cache growth, or anything else on the machine has claimed more) as headroom — see
/// `SAFE_MXFP4_AUTO_CACHE_FALLBACK_BUDGET_BYTES`'s doc for what happens when this can't be read.
const MXFP4_AUTO_CACHE_BUDGET_FRACTION: f64 = 0.4;

/// Fallback safety budget when `/proc/meminfo` can't be read (non-Linux, missing, a sandboxed
/// environment, a malformed `MemAvailable` line, ...) — the ORIGINAL fixed constant from before
/// this was made RAM-aware, kept only as a safety net for that case, not the primary mechanism
/// anymore. Conservative on purpose: better to under-cache on an unusual host than risk the OOM
/// this whole fix exists to prevent.
const SAFE_MXFP4_AUTO_CACHE_FALLBACK_BUDGET_BYTES: u64 = 24_000_000_000;

/// Never clamp the budget below this, however little `MemAvailable` reports — a degenerate
/// near-zero budget would clamp capacity to 0 or 1 even on a merely busy (not actually
/// memory-starved) machine; 4GB keeps a small but real cache alive in that case, letting the
/// existing OOM-safety math (not this floor) be the thing that decides if that's still too much.
const MXFP4_AUTO_CACHE_BUDGET_FLOOR_BYTES: u64 = 4_000_000_000;

/// Parses `MemAvailable`'s value (in kB) out of `/proc/meminfo`'s real text format
/// (`"MemAvailable:   122444808 kB"`, arbitrary internal whitespace) — split out from the file
/// read itself so a unit test can check the parsing against a fixed string, not the real
/// (host-dependent, non-deterministic) file.
fn parse_mem_available_kb(meminfo: &str) -> Option<u64> {
    meminfo.lines().find_map(|l| l.strip_prefix("MemAvailable:")).and_then(|rest| rest.split_whitespace().next()).and_then(|n| n.parse().ok())
}

/// Real available RAM right now, in bytes — `None` if it can't be determined (non-Linux, or any
/// read/parse failure), in which case callers fall back to
/// `SAFE_MXFP4_AUTO_CACHE_FALLBACK_BUDGET_BYTES`.
#[cfg(target_os = "linux")]
fn available_memory_bytes() -> Option<u64> {
    let content = std::fs::read_to_string("/proc/meminfo").ok()?;
    parse_mem_available_kb(&content).map(|kb| kb * 1024)
}
#[cfg(not(target_os = "linux"))]
fn available_memory_bytes() -> Option<u64> {
    None
}

/// One MXFP4 expert's real total resident bytes (`w1`+`w2`+`w3`, packed+scale each) at this
/// checkpoint's real `moe_inter`/`hidden` — shape-only, no disk I/O: builds an empty `QT` of the
/// right shape and reads its own `resident_bytes()`, the exact same math the real loaded tensors
/// would report.
fn mxfp4_expert_bytes(moe_inter: usize, hidden: usize) -> u64 {
    ExpertNaming::KimiK3Mxfp4
        .tensor_specs(0, 0, moe_inter, hidden)
        .iter()
        .map(|(_, rows, cols)| QT::alloc_mxfp4(*rows, *cols).resident_bytes() as u64)
        .sum()
}

/// Clamps `requested` (an AUTO `--expert-cache` default, never an explicit one — see
/// `crate::model::safe_default_expert_cache_capacity`'s doc for why) so that `n_moe_layers *
/// capacity * 1.5 * per_expert_bytes` — the LRU tier's own worst case plus `pin_budget`'s worst
/// case on top, the two tiers this checkpoint's real OOM came from — stays under a RAM-aware
/// safety budget (`MXFP4_AUTO_CACHE_BUDGET_FRACTION` of real `MemAvailable`, see that constant's
/// doc). Prints one `eprintln!` note when it actually had to lower anything, so a
/// slower-than-expected run is never a silent surprise.
pub(crate) fn safe_mxfp4_capacity(requested: usize, n_moe_layers: usize, moe_inter: usize, hidden: usize) -> usize {
    let budget = available_memory_bytes()
        .map(|avail| (avail as f64 * MXFP4_AUTO_CACHE_BUDGET_FRACTION) as u64)
        .unwrap_or(SAFE_MXFP4_AUTO_CACHE_FALLBACK_BUDGET_BYTES)
        .max(MXFP4_AUTO_CACHE_BUDGET_FLOOR_BYTES);
    safe_mxfp4_capacity_with_budget(requested, n_moe_layers, moe_inter, hidden, budget)
}

/// The actual clamp math, `budget_bytes` factored out as a parameter so tests can check it
/// against a fixed, deterministic budget instead of this host's real (and non-deterministic
/// across machines/CI) `MemAvailable`. `safe_mxfp4_capacity` is the real entry point production
/// code calls; this is `pub(crate)` only for `#[cfg(test)]` use.
pub(crate) fn safe_mxfp4_capacity_with_budget(requested: usize, n_moe_layers: usize, moe_inter: usize, hidden: usize, budget_bytes: u64) -> usize {
    if n_moe_layers == 0 {
        return requested;
    }
    let per_expert = mxfp4_expert_bytes(moe_inter, hidden);
    if per_expert == 0 {
        return requested;
    }
    let worst_case_multiplier = n_moe_layers as u64 * per_expert * 3 / 2;
    let max_capacity = (budget_bytes / worst_case_multiplier.max(1)).max(1) as usize;
    let clamped = requested.min(max_capacity);
    if clamped < requested {
        eprintln!(
            "expert cache: auto --expert-cache {requested} would risk ~{:.0}GB peak memory on this checkpoint \
             ({n_moe_layers} MoE layers x ~{:.1}MB/expert, LRU + pinned tiers combined) -- lowered the auto \
             default to {clamped} to stay under a ~{:.0}GB safety budget ({:.0}% of real available RAM, or the \
             fixed fallback if that couldn't be read). Pass --expert-cache explicitly to override this (no \
             clamp applies to an explicit value).",
            (requested as u64 * worst_case_multiplier) as f64 / 1e9,
            per_expert as f64 / 1e6,
            budget_bytes as f64 / 1e9,
            MXFP4_AUTO_CACHE_BUDGET_FRACTION * 100.0,
        );
    }
    clamped
}

/// Reads a routed expert's real on-disk MXFP4 tensors directly into a `QTKind::MxFp4` — no
/// decode/requantize step, since rabbit's own `QTKind::MxFp4` byte layout was confirmed
/// bit-for-bit identical to the real quantizer's (see `ExpertNaming::KimiK3Mxfp4`'s doc): `data`
/// is `{base_name}.weight_packed` verbatim, `block_scale` is `{base_name}.weight_scale` verbatim.
fn qt_load_mxfp4(shards: &Shards, base_name: &str, rows: usize, cols: usize) -> Result<QT, ModelError> {
    let packed_name = format!("{base_name}.weight_packed");
    let scale_name = format!("{base_name}.weight_scale");
    let data = shards.read_raw(&packed_name, false).map_err(ModelError::Safetensors)?;
    let block_scale = shards.read_raw(&scale_name, false).map_err(ModelError::Safetensors)?;
    mxfp4_from_raw(data, block_scale, rows, cols, &packed_name, &scale_name)
}

/// Shared by `qt_load_mxfp4` (the synchronous path) and `uring_load::complete_batch_streaming`
/// (the `io_uring` path) — validates `data`/`block_scale`'s byte lengths against the shapes
/// `weight_packed`/`weight_scale` must have and constructs `QTKind::MxFp4` directly. No float
/// decode step either way: rabbit's own MXFP4 byte layout already matches the real quantizer's
/// (see `ExpertNaming::KimiK3Mxfp4`'s doc), so the raw bytes are used verbatim.
fn mxfp4_from_raw(data: Vec<u8>, block_scale: Vec<u8>, rows: usize, cols: usize, packed_name: &str, scale_name: &str) -> Result<QT, ModelError> {
    let expected_data = rows * cols.div_ceil(2);
    if data.len() != expected_data {
        return Err(ModelError::ShapeMismatch { name: packed_name.to_string(), expected: expected_data, got: data.len() });
    }
    let expected_scale = rows * cols.div_ceil(32);
    if block_scale.len() != expected_scale {
        return Err(ModelError::ShapeMismatch { name: scale_name.to_string(), expected: expected_scale, got: block_scale.len() });
    }

    Ok(QT { rows, cols, bits: 4, kind: QTKind::MxFp4 { data, block_scale } })
}

/// `true` once any `mlock` call this process has made has failed — checked/set by
/// `mlock_best_effort` below. `RLIMIT_MEMLOCK` (the overwhelmingly likely real cause on an
/// unprivileged process, see that function's doc) doesn't change over a process's lifetime, so a
/// single failure means every future attempt will fail identically. Found the hard way
/// (2026-07-28): a real, long session that promotes many experts to `pinned` (which grows over
/// time as `usage_cache`'s persisted history accumulates across runs) was retrying — and
/// re-logging — this same doomed syscall thousands of times, real measured overhead that once
/// erased this same session's `io_uring`-for-MXFP4 win entirely in one A/B (see
/// `PERFORMANCE_KIMI_K3.md`). `Relaxed` ordering: this is a best-effort perf/log-spam guard, not
/// a correctness barrier — a handful of redundant syscalls from a benign race right at the first
/// failure is harmless, unlike `usage`/`pinned`'s own state which is never touched concurrently
/// anyway (this whole struct is `&mut self`-only, no interior mutability elsewhere).
static MLOCK_KNOWN_TO_FAIL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Best-effort `mlock` on a pinned expert's backing buffers, so the OS can't swap out memory
/// pinning exists specifically to keep resident — same "unsafe libc call, best-effort, never a
/// hard error" precedent already established for `posix_fadvise` in `safetensors.rs`. A failure
/// (commonly `RLIMIT_MEMLOCK` too low for an unprivileged process) is logged once and otherwise
/// ignored: losing the OS-level guarantee just means a pinned expert *could* get swapped under
/// real memory pressure, same risk profile as not pinning it at all — not a correctness issue.
/// Once ANY attempt fails, every later call (this expert's remaining buffers, and every future
/// expert) short-circuits immediately via `MLOCK_KNOWN_TO_FAIL` — no more syscalls, no more log
/// lines, for the rest of the process. See that flag's doc for why this is safe to assume.
fn mlock_best_effort(slot: &ExpertSlot) {
    use std::sync::atomic::Ordering;
    if MLOCK_KNOWN_TO_FAIL.load(Ordering::Relaxed) {
        return;
    }
    let gate_up_qts: Vec<&QT> = match &slot.gate_up {
        GateUp::Separate { gate, up } => vec![gate, up],
        GateUp::Fused { gate_up } => vec![gate_up],
    };
    for qt in gate_up_qts.into_iter().chain(std::iter::once(&slot.down)) {
        let bufs: Vec<(*const u8, usize)> = match &qt.kind {
            QTKind::F32(v) => vec![(v.as_ptr() as *const u8, std::mem::size_of_val(v.as_slice()))],
            QTKind::I8 { data, scale } => {
                vec![(data.as_ptr() as *const u8, std::mem::size_of_val(data.as_slice())), (scale.as_ptr() as *const u8, std::mem::size_of_val(scale.as_slice()))]
            }
            QTKind::I4 { data, scale } | QTKind::I2 { data, scale } | QTKind::I4Grouped { data, scale, .. } => {
                vec![(data.as_ptr(), data.len()), (scale.as_ptr() as *const u8, std::mem::size_of_val(scale.as_slice()))]
            }
            QTKind::MxFp4 { data, block_scale } => {
                vec![(data.as_ptr(), data.len()), (block_scale.as_ptr(), block_scale.len())]
            }
        };
        for (ptr, len) in bufs {
            if len > 0 && unsafe { libc::mlock(ptr as *const libc::c_void, len) } != 0 {
                if !MLOCK_KNOWN_TO_FAIL.swap(true, Ordering::Relaxed) {
                    eprintln!(
                        "usage cache: mlock failed for a pinned expert (best-effort, disabling further mlock attempts this process): {}",
                        std::io::Error::last_os_error()
                    );
                }
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;

    // ---- safe_mxfp4_capacity (2026-07-28: a real `--prompt` run got OOM-killed at the flat
    // `64` default against K3's real per-expert scale — see this function's own doc and
    // `PERFORMANCE_KIMI_K3.md`'s "A red herring" section for the real numbers). All tests below
    // use `safe_mxfp4_capacity_with_budget` with a FIXED budget, not the real
    // `safe_mxfp4_capacity` entry point — that one reads this host's live `/proc/meminfo`,
    // which is exactly the non-determinism these tests need to avoid (a CI box or a future dev
    // machine with different real RAM would otherwise get a different, flaky answer).
    const TEST_BUDGET_BYTES: u64 = 24_000_000_000;

    /// At K3's real real dimensions (`moe_intermediate_size=3072`, `hidden_size=7168`,
    /// `num_hidden_layers=93`), the flat `64` default must get clamped to something whose
    /// combined LRU+pinned worst case actually fits the given safety budget — fast (no real
    /// checkpoint, shape-only math), unlike confirming this against the real 1.56TB checkpoint.
    #[test]
    fn safe_mxfp4_capacity_clamps_the_flat_default_at_real_k3_scale() {
        let (moe_inter, hidden, n_moe_layers) = (3072usize, 7168usize, 93usize);
        let clamped = safe_mxfp4_capacity_with_budget(64, n_moe_layers, moe_inter, hidden, TEST_BUDGET_BYTES);
        assert!(clamped < 64, "the flat default must actually get lowered at K3's real scale, got {clamped}");
        assert!(clamped >= 1, "must never clamp to 0 and silently disable caching entirely");

        let per_expert = mxfp4_expert_bytes(moe_inter, hidden);
        let combined_worst_case = n_moe_layers as u64 * clamped as u64 * per_expert * 3 / 2;
        assert!(
            combined_worst_case <= TEST_BUDGET_BYTES,
            "clamped capacity {clamped} still risks {combined_worst_case} bytes, over the {TEST_BUDGET_BYTES} byte budget"
        );
    }

    /// A request that's already safe at this checkpoint's scale (e.g. a small synthetic test
    /// fixture, or a deliberately conservative `--expert-cache`) must pass through unchanged —
    /// this function only ever lowers, never raises, and never interferes when there's nothing
    /// to protect against.
    #[test]
    fn safe_mxfp4_capacity_does_not_touch_an_already_safe_request() {
        assert_eq!(safe_mxfp4_capacity_with_budget(2, 93, 3072, 7168, TEST_BUDGET_BYTES), 2);
        assert_eq!(safe_mxfp4_capacity_with_budget(8, 1, 16, 16, TEST_BUDGET_BYTES), 8, "tiny synthetic shapes: nowhere near the budget");
    }

    /// `n_moe_layers == 0` (a checkpoint with no MoE layers at all, or called before any are
    /// known) must return the request unchanged rather than divide by zero.
    #[test]
    fn safe_mxfp4_capacity_handles_zero_moe_layers() {
        assert_eq!(safe_mxfp4_capacity_with_budget(64, 0, 3072, 7168, TEST_BUDGET_BYTES), 64);
    }

    /// A bigger budget must allow (not force) a bigger capacity — the whole point of making the
    /// budget RAM-aware instead of a fixed constant: a machine with more real available memory
    /// should get more real caching benefit, not the same conservative number regardless.
    #[test]
    fn safe_mxfp4_capacity_scales_up_with_a_bigger_budget() {
        let (moe_inter, hidden, n_moe_layers) = (3072usize, 7168usize, 93usize);
        let small_budget = safe_mxfp4_capacity_with_budget(64, n_moe_layers, moe_inter, hidden, 24_000_000_000);
        let big_budget = safe_mxfp4_capacity_with_budget(64, n_moe_layers, moe_inter, hidden, 48_000_000_000);
        assert!(big_budget > small_budget, "doubling the budget should allow a bigger auto capacity, got {small_budget} vs {big_budget}");
    }

    // ---- parse_mem_available_kb / available_memory_bytes (2026-07-28: replaced the fixed 24GB
    // budget with one based on real `/proc/meminfo` — a fixed number is either wastefully
    // conservative on a bigger machine or, worse, still unsafe on a smaller one) ----

    #[test]
    fn parse_mem_available_kb_reads_the_real_proc_meminfo_format() {
        let fixture = "MemTotal:       129433076 kB\nMemFree:        10000000 kB\nMemAvailable:   122444808 kB\nBuffers:          123456 kB\n";
        assert_eq!(parse_mem_available_kb(fixture), Some(122444808));
    }

    #[test]
    fn parse_mem_available_kb_returns_none_when_the_field_is_missing() {
        let fixture = "MemTotal:       129433076 kB\nMemFree:        10000000 kB\n";
        assert_eq!(parse_mem_available_kb(fixture), None);
    }

    #[test]
    fn parse_mem_available_kb_returns_none_on_a_malformed_value() {
        let fixture = "MemAvailable:   not-a-number kB\n";
        assert_eq!(parse_mem_available_kb(fixture), None);
    }

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

    /// Like `build_experts_fixture`, but writes the FUSED layout `bin/fuse_gate_up.rs` would
    /// produce FROM that exact fixture: `gate_up_proj.weight` (`[2*moe_inter, hidden]`, gate's
    /// rows first then up's — same xorshift seed sequence, so byte-for-byte the same values)
    /// instead of separate `gate_proj`/`up_proj` tensors. `down_proj` is unchanged (fusion never
    /// touches it). Used to prove the `Glm52FusedGateUp` read path lands on the exact same
    /// numbers `Glm52`'s separate read of the pre-fusion fixture would.
    fn build_fused_experts_fixture(name: &str, n_experts: usize, moe_inter: usize, hidden: usize) -> TempDir {
        let dir = TempDir::new(name);
        let mut seed = 1u32;
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_string(), json!({"format": "rabbit-test"}));
        let mut data = Vec::new();
        let mut push = |header: &mut serde_json::Map<String, serde_json::Value>, name: String, rows: usize, cols: usize, vals: Vec<f32>| {
            let bytes = f32_bytes(&vals);
            let start = data.len() as u64;
            data.extend_from_slice(&bytes);
            let end = data.len() as u64;
            header.insert(name, json!({"dtype": "F32", "shape": [rows, cols], "data_offsets": [start, end]}));
        };
        for eid in 0..n_experts {
            // Same order `build_experts_fixture` draws from the same seed: gate, then up, then
            // down — critical for the two fixtures to describe the SAME logical checkpoint.
            let gate_vals: Vec<f32> = (0..moe_inter * hidden).map(|_| xorshift(&mut seed)).collect();
            let up_vals: Vec<f32> = (0..moe_inter * hidden).map(|_| xorshift(&mut seed)).collect();
            let down_vals: Vec<f32> = (0..hidden * moe_inter).map(|_| xorshift(&mut seed)).collect();
            let mut gate_up_vals = gate_vals;
            gate_up_vals.extend(up_vals);
            push(&mut header, format!("model.layers.0.mlp.experts.{eid}.gate_up_proj.weight"), 2 * moe_inter, hidden, gate_up_vals);
            push(&mut header, format!("model.layers.0.mlp.experts.{eid}.down_proj.weight"), hidden, moe_inter, down_vals);
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

    /// Writes one expert's `w1`/`w3`/`w2` as REAL-shaped MXFP4 pairs (`{name}.weight_packed` +
    /// `{name}.weight_scale`, `language_model.`-prefixed like `ExpertNaming::KimiK3Mxfp4`
    /// expects) — `w1` from HAND-PICKED raw bytes (independent of `QT::alloc_mxfp4`/`fill`, so a
    /// bug in that encoder couldn't hide a matching bug here), `w3`/`w2` round-tripped through
    /// `QT::alloc_mxfp4`/`fill` from values chosen to be exactly E2M1-representable with a
    /// forced 6.0 (or -6.0) magnitude in every 32-block, which `choose_block_scale` always maps
    /// to `scale=1.0` (byte 127) — so the round trip is exact, not just "close".
    fn build_k3_mxfp4_experts_fixture(name: &str, moe_inter: usize, hidden: usize) -> TempDir {
        let dir = TempDir::new(name);
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_string(), json!({"format": "rabbit-test"}));
        let mut data = Vec::new();
        let mut push_raw = |header: &mut serde_json::Map<String, serde_json::Value>, name: String, shape: Vec<usize>, bytes: Vec<u8>| {
            let start = data.len() as u64;
            data.extend_from_slice(&bytes);
            let end = data.len() as u64;
            header.insert(name, json!({"dtype": "U8", "shape": shape, "data_offsets": [start, end]}));
        };

        // w1: [moe_inter, hidden], hand-encoded. E2M1_MAGNITUDES = [0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0],
        // so magnitude 1.0 is index 2, not 1. Row 0: codes 2 (=1.0), 0xB (=-1.5), 7 (=6.0), 8
        // (=-0.0), then zero-padded; scale byte 127 (2^0 = 1.0) for every block. Packed
        // low-nibble-first: byte0 = idx0 | (idx1<<4) = 0x02 | 0xB0 = 0xB2; byte1 = idx2 |
        // (idx3<<4) = 0x07 | 0x80 = 0x87.
        assert!(hidden >= 4, "fixture assumes room for the 4 hand-picked w1 codes");
        let mut w1_packed = vec![0u8; moe_inter * hidden.div_ceil(2)];
        w1_packed[0] = 0xB2;
        w1_packed[1] = 0x87;
        let w1_scale = vec![127u8; moe_inter * hidden.div_ceil(32)];
        push_raw(&mut header, "language_model.model.layers.0.block_sparse_moe.experts.0.w1.weight_packed".to_string(), vec![moe_inter, hidden], w1_packed);
        push_raw(&mut header, "language_model.model.layers.0.block_sparse_moe.experts.0.w1.weight_scale".to_string(), vec![moe_inter, hidden.div_ceil(32)], w1_scale);

        // Forces a 6.0 magnitude at every block's FIRST column (local `c % 32 == 0`, not a
        // flattened index) — `choose_block_scale` picks its scale from each row's own
        // 32-contiguous-element blocks, so the forced-max element must land inside every one of
        // those blocks, not just periodically across the whole flattened matrix.
        let mk_round_tripped = |rows: usize, cols: usize, seed: usize| -> (Vec<u8>, Vec<u8>) {
            let mut w = vec![0f32; rows * cols];
            for r in 0..rows {
                for c in 0..cols {
                    let mag = if c % 32 == 0 { 6.0 } else { E2M1_TEST_MAGNITUDES[(r + c + seed) % E2M1_TEST_MAGNITUDES.len()] };
                    let sign = if (r + c + seed).is_multiple_of(2) { 1.0 } else { -1.0 };
                    w[r * cols + c] = sign * mag;
                }
            }
            let mut t = QT::alloc_mxfp4(rows, cols);
            t.fill(&w);
            match t.kind {
                QTKind::MxFp4 { data, block_scale } => (data, block_scale),
                _ => unreachable!(),
            }
        };
        let (w3_packed, w3_scale) = mk_round_tripped(moe_inter, hidden, 3);
        push_raw(&mut header, "language_model.model.layers.0.block_sparse_moe.experts.0.w3.weight_packed".to_string(), vec![moe_inter, hidden], w3_packed);
        push_raw(&mut header, "language_model.model.layers.0.block_sparse_moe.experts.0.w3.weight_scale".to_string(), vec![moe_inter, hidden.div_ceil(32)], w3_scale);
        let (w2_packed, w2_scale) = mk_round_tripped(hidden, moe_inter, 2);
        push_raw(&mut header, "language_model.model.layers.0.block_sparse_moe.experts.0.w2.weight_packed".to_string(), vec![hidden, moe_inter], w2_packed);
        push_raw(&mut header, "language_model.model.layers.0.block_sparse_moe.experts.0.w2.weight_scale".to_string(), vec![hidden, moe_inter.div_ceil(32)], w2_scale);

        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_bytes);
        out.extend_from_slice(&data);
        fs::write(dir.0.join("model.safetensors"), out).unwrap();
        dir
    }

    const E2M1_TEST_MAGNITUDES: [f32; 7] = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

    /// Like `build_k3_mxfp4_experts_fixture`, but `n_experts` of them, all round-tripped through
    /// `QT::alloc_mxfp4`/`fill` (no hand-encoded bytes — that byte-order check is already covered
    /// by `kimi_k3_mxfp4_naming_reads_weight_packed_and_weight_scale_directly`). Used to compare
    /// the `io_uring` batch path against `get_or_load`'s sequential one across a real MULTI-expert
    /// batch, the shape `submit_batch`/`complete_batch_streaming`'s MXFP4 branch actually runs at.
    fn build_k3_mxfp4_multi_experts_fixture(name: &str, n_experts: usize, moe_inter: usize, hidden: usize) -> TempDir {
        let dir = TempDir::new(name);
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_string(), json!({"format": "rabbit-test"}));
        let mut data = Vec::new();
        let mut push_raw = |header: &mut serde_json::Map<String, serde_json::Value>, name: String, shape: Vec<usize>, bytes: Vec<u8>| {
            let start = data.len() as u64;
            data.extend_from_slice(&bytes);
            let end = data.len() as u64;
            header.insert(name, json!({"dtype": "U8", "shape": shape, "data_offsets": [start, end]}));
        };
        let mk = |rows: usize, cols: usize, seed: u32| -> (Vec<u8>, Vec<u8>) {
            let mut s = seed;
            let w: Vec<f32> = (0..rows * cols).map(|_| xorshift(&mut s)).collect();
            let mut t = QT::alloc_mxfp4(rows, cols);
            t.fill(&w);
            match t.kind {
                QTKind::MxFp4 { data, block_scale } => (data, block_scale),
                _ => unreachable!(),
            }
        };
        for eid in 0..n_experts {
            let p = |suf: &str| format!("language_model.model.layers.0.block_sparse_moe.experts.{eid}.{suf}");
            let (w1_packed, w1_scale) = mk(moe_inter, hidden, (eid as u32) * 3 + 1);
            push_raw(&mut header, format!("{}.weight_packed", p("w1")), vec![moe_inter, hidden], w1_packed);
            push_raw(&mut header, format!("{}.weight_scale", p("w1")), vec![moe_inter, hidden.div_ceil(32)], w1_scale);
            let (w3_packed, w3_scale) = mk(moe_inter, hidden, (eid as u32) * 3 + 2);
            push_raw(&mut header, format!("{}.weight_packed", p("w3")), vec![moe_inter, hidden], w3_packed);
            push_raw(&mut header, format!("{}.weight_scale", p("w3")), vec![moe_inter, hidden.div_ceil(32)], w3_scale);
            let (w2_packed, w2_scale) = mk(hidden, moe_inter, (eid as u32) * 3 + 3);
            push_raw(&mut header, format!("{}.weight_packed", p("w2")), vec![hidden, moe_inter], w2_packed);
            push_raw(&mut header, format!("{}.weight_scale", p("w2")), vec![hidden, moe_inter.div_ceil(32)], w2_scale);
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
            group_size: 0,
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
        assert_eq!(slot.gate().rows, 4);
        assert_eq!(slot.gate().cols, 6);
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
        assert_eq!(slot.gate().rows, 4);
        assert_eq!(slot.gate().cols, 6);
        assert_eq!(slot.down.rows, 6);
        assert_eq!(slot.down.cols, 4);
        assert!(qt_values(slot.gate()).iter().all(|&v| (v - 1.0).abs() < 1e-6), "w1 must land in gate");
        assert!(qt_values(slot.up()).iter().all(|&v| (v - 3.0).abs() < 1e-6), "w3 must land in up");
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
            assert_eq!(qt_values(a.gate()), qt_values(b.gate()), "expert {eid} gate");
            assert_eq!(qt_values(a.up()), qt_values(b.up()), "expert {eid} up");
            assert_eq!(qt_values(&a.down), qt_values(&b.down), "expert {eid} down");
        }
    }

    /// `ExpertNaming::KimiK3Mxfp4`'s whole point: read `{name}.weight_packed`/`{name}.weight_scale`
    /// straight off disk into `QTKind::MxFp4`, no `.weight` tensor involved. `w1`'s bytes are
    /// hand-encoded (independent of `QT::alloc_mxfp4`/`fill`), so this also exercises the exact
    /// E2M1/E8M0 decode a real checkpoint's bytes would hit, not just a round trip through
    /// rabbit's own encoder.
    #[test]
    fn kimi_k3_mxfp4_naming_reads_weight_packed_and_weight_scale_directly() {
        let moe_inter = 4;
        let hidden = 40; // >32 so w1/w3 span 2 blocks/row, matching the real checkpoint's shape
        let fixture = build_k3_mxfp4_experts_fixture("rabbit_test_ecache_k3_mxfp4", moe_inter, hidden);
        let shards = Shards::open(&fixture.0).unwrap();
        let cfg = tiny_cfg(1, moe_inter as i32, hidden as i32);
        let mut cache = ExpertCache::for_family(8, ExpertNaming::KimiK3Mxfp4);

        let slot = cache.get_or_load(&shards, &cfg, 0, 0, 4).unwrap();
        assert_eq!(slot.gate().rows, moe_inter);
        assert_eq!(slot.gate().cols, hidden);
        assert_eq!(slot.down.rows, hidden);
        assert_eq!(slot.down.cols, moe_inter);
        assert!(matches!(slot.gate().kind, QTKind::MxFp4 { .. }));
        assert!(matches!(slot.up().kind, QTKind::MxFp4 { .. }));
        assert!(matches!(slot.down.kind, QTKind::MxFp4 { .. }));

        let gate_row0 = slot.gate().row_f32(0);
        assert_eq!(&gate_row0[0..4], &[1.0, -1.5, 6.0, -0.0], "hand-encoded w1 row 0's first 4 codes");
        assert!(gate_row0[4..].iter().all(|&v| v == 0.0), "the rest of w1's fixture bytes are zeroed");

        // w3/w2 round-tripped through rabbit's own encoder with values chosen to be exact.
        let up_vals = qt_values(slot.up());
        let down_vals = qt_values(&slot.down);
        assert_eq!(up_vals.len(), moe_inter * hidden);
        assert_eq!(down_vals.len(), hidden * moe_inter);
        for &v in up_vals.iter().chain(down_vals.iter()) {
            assert!(E2M1_TEST_MAGNITUDES.contains(&v.abs()) || v == 0.0, "value {v} isn't an exact E2M1 magnitude");
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

    /// Same as `fused_gate_up_naming_matches_separate_naming_values` below, but through
    /// `ensure_loaded` — the batched `io_uring` path (Linux) — instead of `get_or_load`'s
    /// always-sequential `load_expert`. `submit_batch`/`complete_batch_streaming`'s ring-sizing
    /// and per-expert read-count bookkeeping both had to learn `tensor_count()` isn't always 3
    /// for this to pass; a shape-only check wouldn't catch a bufs-index-off-by-one there.
    #[test]
    fn ensure_loaded_fused_gate_up_matches_get_or_load_separate_values() {
        let (n_experts, moe_inter, hidden) = (3, 5, 7);
        let sep_fixture = build_experts_fixture("rabbit_test_ecache_fused_uring_sep", n_experts, moe_inter, hidden);
        let fused_fixture = build_fused_experts_fixture("rabbit_test_ecache_fused_uring_fused", n_experts, moe_inter, hidden);
        let sep_shards = Shards::open(&sep_fixture.0).unwrap();
        let fused_shards = Shards::open(&fused_fixture.0).unwrap();
        let cfg = tiny_cfg(n_experts as i32, moe_inter as i32, hidden as i32);
        let eids: Vec<usize> = (0..n_experts).collect();

        let mut sep_cache = ExpertCache::new(8);
        for &eid in &eids {
            sep_cache.get_or_load(&sep_shards, &cfg, 0, eid, 32).unwrap();
        }

        let mut fused_cache = ExpertCache::for_family(8, ExpertNaming::Glm52FusedGateUp);
        fused_cache.ensure_loaded(&fused_shards, &cfg, 0, &eids, 32).unwrap();

        for &eid in &eids {
            let sep_slot = sep_cache.get(eid).unwrap();
            let expected_gate = qt_values(sep_slot.gate());
            let expected_up = qt_values(sep_slot.up());
            let expected_down = qt_values(&sep_slot.down);

            let fused_slot = fused_cache.get(eid).unwrap();
            let GateUp::Fused { gate_up } = &fused_slot.gate_up else { panic!("expected GateUp::Fused") };
            let gate_up_vals = qt_values(gate_up);
            let (got_gate, got_up) = gate_up_vals.split_at(moe_inter * hidden);
            assert_eq!(got_gate, expected_gate.as_slice(), "expert {eid}: gate half (via ensure_loaded/io_uring)");
            assert_eq!(got_up, expected_up.as_slice(), "expert {eid}: up half (via ensure_loaded/io_uring)");
            assert_eq!(qt_values(&fused_slot.down), expected_down, "expert {eid}: down_proj (via ensure_loaded/io_uring)");
        }
    }

    /// `ExpertNaming::Glm52FusedGateUp` (`bin/fuse_gate_up.rs`'s output format) must decode to
    /// the EXACT same weight values as `ExpertNaming::Glm52` reading the pre-fusion checkpoint
    /// it was produced from — this is the whole correctness contract the converter and this
    /// read path both depend on. Checked through `get_or_load` (the sequential fallback).
    #[test]
    fn fused_gate_up_naming_matches_separate_naming_values() {
        let (n_experts, moe_inter, hidden) = (2, 4, 6);
        let sep_fixture = build_experts_fixture("rabbit_test_ecache_fused_vs_sep_sep", n_experts, moe_inter, hidden);
        let fused_fixture = build_fused_experts_fixture("rabbit_test_ecache_fused_vs_sep_fused", n_experts, moe_inter, hidden);
        let sep_shards = Shards::open(&sep_fixture.0).unwrap();
        let fused_shards = Shards::open(&fused_fixture.0).unwrap();
        let cfg = tiny_cfg(n_experts as i32, moe_inter as i32, hidden as i32);

        let mut sep_cache = ExpertCache::new(8);
        let mut fused_cache = ExpertCache::for_family(8, ExpertNaming::Glm52FusedGateUp);

        for eid in 0..n_experts {
            let sep_slot = sep_cache.get_or_load(&sep_shards, &cfg, 0, eid, 32).unwrap();
            let expected_gate = qt_values(sep_slot.gate());
            let expected_up = qt_values(sep_slot.up());
            let expected_down = qt_values(&sep_slot.down);

            let fused_slot = fused_cache.get_or_load(&fused_shards, &cfg, 0, eid, 32).unwrap();
            let GateUp::Fused { gate_up } = &fused_slot.gate_up else { panic!("expected GateUp::Fused") };
            assert_eq!(gate_up.rows, 2 * moe_inter, "expert {eid}: gate_up_proj row count");
            let gate_up_vals = qt_values(gate_up);
            let (got_gate, got_up) = gate_up_vals.split_at(moe_inter * hidden);
            assert_eq!(got_gate, expected_gate.as_slice(), "expert {eid}: gate half of gate_up_proj");
            assert_eq!(got_up, expected_up.as_slice(), "expert {eid}: up half of gate_up_proj");
            assert_eq!(qt_values(&fused_slot.down), expected_down, "expert {eid}: down_proj");
        }
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
            assert_eq!(qt_values(a.gate()), qt_values(b.gate()), "expert {eid} gate_proj");
            assert_eq!(qt_values(a.up()), qt_values(b.up()), "expert {eid} up_proj");
            assert_eq!(qt_values(&a.down), qt_values(&b.down), "expert {eid} down_proj");
        }
    }

    /// Same contract as `ensure_loaded_matches_sequential_get_or_load_values`, but for
    /// `ExpertNaming::KimiK3Mxfp4` — the whole point of this session's `io_uring` work: before
    /// this, `ExpertCache::for_family` never even created a ring for this naming (every load fell
    /// back to sequential), so `ensure_loaded`'s batch path for MXFP4 was previously untested
    /// because it didn't exist. Proves `submit_batch`/`complete_batch_streaming`'s
    /// `{name}.weight_packed`+`{name}.weight_scale` handling decodes identically to
    /// `get_or_load`'s sequential `qt_load_mxfp4` — same real-checkpoint tensor naming, same
    /// `QTKind::MxFp4` values, across a real multi-expert batch (not just the single-expert,
    /// hand-encoded-bytes shape check `kimi_k3_mxfp4_naming_reads_weight_packed_and_weight_scale_
    /// directly` already covers).
    #[test]
    fn ensure_loaded_matches_sequential_get_or_load_values_for_mxfp4() {
        let (n_experts, moe_inter, hidden) = (4, 4, 40); // hidden>32: w1/w3 span 2 blocks/row
        let fixture = build_k3_mxfp4_multi_experts_fixture("rabbit_test_ecache_mxfp4_uring", n_experts, moe_inter, hidden);
        let shards = Shards::open(&fixture.0).unwrap();
        let cfg = tiny_cfg(n_experts as i32, moe_inter as i32, hidden as i32);
        let eids: Vec<usize> = (0..n_experts).collect();

        let mut batch_cache = ExpertCache::for_family(8, ExpertNaming::KimiK3Mxfp4);
        batch_cache.ensure_loaded(&shards, &cfg, 0, &eids, 4).unwrap();

        let mut sequential_cache = ExpertCache::for_family(8, ExpertNaming::KimiK3Mxfp4);
        for &eid in &eids {
            sequential_cache.get_or_load(&shards, &cfg, 0, eid, 4).unwrap();
        }

        for &eid in &eids {
            let a = batch_cache.get(eid).unwrap();
            let b = sequential_cache.get(eid).unwrap();
            assert!(matches!(a.gate().kind, QTKind::MxFp4 { .. }), "expert {eid}: io_uring path should still produce QTKind::MxFp4");
            assert_eq!(qt_values(a.gate()), qt_values(b.gate()), "expert {eid} w1/gate");
            assert_eq!(qt_values(a.up()), qt_values(b.up()), "expert {eid} w3/up");
            assert_eq!(qt_values(&a.down), qt_values(&b.down), "expert {eid} w2/down");
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
            assert_eq!(qt_values(a.gate()), qt_values(b.gate()), "expert {eid} gate_proj");
            assert_eq!(qt_values(a.up()), qt_values(b.up()), "expert {eid} up_proj");
            assert_eq!(qt_values(&a.down), qt_values(&b.down), "expert {eid} down_proj");
        }

        // a batch that's entirely cache hits must short-circuit through `LoadKind::Nothing`
        // cleanly, not touch disk or misscount.
        let pending2 = split_cache.begin_loading(&shards, &cfg, 0, &eids, 32).unwrap();
        split_cache.finish_loading(pending2, &shards, &cfg, 0, 32).unwrap();
        assert_eq!(split_cache.hits, plain_cache.hits + eids.len() as u64);
        assert_eq!(split_cache.misses, plain_cache.misses);
    }

    /// The whole point of decoupling `io_batch_size` from `capacity` (2026-07-29): a batch
    /// bigger than the resident cache must still correctly compute EVERY expert's contribution
    /// via the streaming `on_slot` callback, even though most of them get evicted again almost
    /// immediately afterward (capacity far smaller than the batch). Uses `begin_loading`/
    /// `finish_loading_streaming` directly (the same pair `moe.rs` calls), capturing every
    /// streamed slot's values via `on_slot` BEFORE any eviction — the same way `moe.rs`'s real
    /// callback consumes each expert's data the moment it lands.
    #[test]
    fn io_batch_size_bigger_than_capacity_still_streams_every_experts_correct_values() {
        let (n_experts, moe_inter, hidden) = (10, 4, 6);
        let fixture = build_experts_fixture("rabbit_test_ecache_io_batch_decouple", n_experts, moe_inter, hidden);
        let shards = Shards::open(&fixture.0).unwrap();
        let cfg = tiny_cfg(n_experts as i32, moe_inter as i32, hidden as i32);
        let eids: Vec<usize> = (0..n_experts).collect();

        // capacity=2 (memory-safe, small) but io_batch_size=10 (the whole batch fits one round)
        // -- decoupled on purpose, unlike `ExpertCache::new`'s coupled default.
        let mut cache = ExpertCache::for_family_with_io_batch(2, n_experts, ExpertNaming::Glm52);
        let mut reference = ExpertCache::new(n_experts); // capacity=10: holds all of them, for comparison
        for &eid in &eids {
            reference.get_or_load(&shards, &cfg, 0, eid, 32).unwrap();
        }

        let mut streamed = Vec::new();
        let pending = cache.begin_loading(&shards, &cfg, 0, &eids, 32).unwrap();
        cache
            .finish_loading_streaming(pending, &shards, &cfg, 0, 32, |slot| {
                streamed.push((slot.eid, qt_values(slot.gate()), qt_values(slot.up()), qt_values(&slot.down)));
            })
            .unwrap();

        assert_eq!(streamed.len(), n_experts, "every expert's data must stream through on_slot, regardless of resident capacity");
        for (eid, gate, up, down) in &streamed {
            let want = reference.get(*eid).unwrap();
            assert_eq!(gate, &qt_values(want.gate()), "expert {eid} gate_proj");
            assert_eq!(up, &qt_values(want.up()), "expert {eid} up_proj");
            assert_eq!(down, &qt_values(&want.down), "expert {eid} down_proj");
        }

        // AFTER streaming, the resident cache must still respect `capacity` (2), not
        // `io_batch_size` (10) -- the whole point of keeping the two concepts separate.
        assert!(cache.len() <= 2, "resident cache must stay within capacity, got {}", cache.len());
        assert_eq!(cache.io_batch_size(), n_experts);
        assert_eq!(cache.capacity(), 2);
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
            let crate::quant::QTKind::I8 { data, scale } = &slot.gate().kind else { panic!("expected I8 gate") };
            assert_eq!(data, &gate_bytes.iter().map(|&b| b as i8).collect::<Vec<i8>>());
            assert_eq!(scale, &gate_scale);
            let crate::quant::QTKind::I8 { data, scale } = &slot.up().kind else { panic!("expected I8 up") };
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

        assert_eq!(qt_values(batch_slot.gate()), qt_values(sequential_slot.gate()), "gate_proj");
        assert_eq!(qt_values(batch_slot.up()), qt_values(sequential_slot.up()), "up_proj");
        assert_eq!(qt_values(&batch_slot.down), qt_values(&sequential_slot.down), "down_proj");

        // and the scale was actually applied, not silently skipped -> magnitude ~2.0/3.0/4.0,
        // not ~1.0 (which is what the pre-fix bug's unscaled decode would have produced).
        let gate_vals = qt_values(batch_slot.gate());
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
