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
//! - No **O_DIRECT**: colibri's own default is buffered reads too (`g_direct=0` — "su questo
//!   host... il buffered liscio e' risultato il migliore"); O_DIRECT needs page-aligned
//!   buffers/offsets/lengths, which is real complexity for a benefit the original's own
//!   measurements don't reliably show. The `io_uring` win here is purely about collapsing N
//!   syscalls into ~1, which buffered reads already get.
//! - No **`.qs` pre-quantized fast path**: same cut as `model.rs`'s `qt_load` — out of scope
//!   for this whole port stage (see `rabbit-plan.md`).
//!
//! Fase 13 (this file) DOES now port the persistent learning cache (`.coli_usage`'s `eusage` ->
//! `usage_cache.rs`'s `.rabbit_usage`) and its pin tier (`m->pin[layer]` -> `ExpertCache::pinned`)
//! — see `usage_cache.rs` and `generate.rs`'s `ExpertCaches::warm_start`. Still cut: colibrì's
//! opt-in LIVE re-pin (`REPIN`/`eheat`/`tier.h`'s decay-and-swap), off by default there too, and
//! any VRAM/CUDA promotion (no CUDA backend here).

use crate::config::Cfg;
use crate::model::{ModelError, qt_load};
use crate::quant::{QT, QTKind};
use crate::safetensors::Shards;
use std::collections::HashMap;

#[cfg(target_os = "linux")]
mod uring_load {
    use super::{Cfg, ExpertSlot, ModelError, QT};
    use crate::safetensors::{DType, SafetensorsError, Shards, TensorLocation, dequant_fp8_blockscale};
    use io_uring::{IoUring, opcode, types};

    struct Req {
        rows: usize,
        cols: usize,
        loc: TensorLocation,
        /// `Some(scale)` when a `{name}.qs` sibling exists: `loc` then points at already-
        /// packed int8/int4/int2 bytes (colibri's pre-quantized container, see `model.rs`'s
        /// `qt_load`) to wrap as-is via `QT::from_packed`, not raw f32/bf16/FP8 to decode and
        /// requantize. The scale itself is tiny (`rows` floats) — fetched with a plain
        /// `read_f32` up front rather than folded into the batched `io_uring` reads, which
        /// exist to amortize the *big* per-expert tensors, not a few-KB sidecar.
        packed_scale: Option<Vec<f32>>,
        /// `Some(scale)` when `loc.dtype == F8E4M3` (mutually exclusive with `packed_scale` —
        /// a tensor is never both `.qs`-packed and raw FP8): the `{name}_scale_inv` 128x128
        /// block-scale sidecar, fetched up front the same way `packed_scale` is, so the raw
        /// bytes `io_uring` reads back can be dequantized correctly instead of falling through
        /// to `Shards::decode_f32`'s *unscaled* per-element FP8 decode.
        fp8_scale: Option<Vec<f32>>,
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

    /// Creates the persistent ring for an `ExpertCache` of the given ADT capacity (3
    /// tensors/expert). Returns `None` if `io_uring` isn't usable on this host (the
    /// `io_uring_disabled` sysctl, a seccomp profile blocking the syscalls, ...) — every
    /// `submit_batch` call then falls back to a synchronous load, same as a hard per-call
    /// failure would.
    pub(super) fn new_ring(cache_capacity: usize) -> Option<Ring> {
        Ring::new(cache_capacity * 3).ok()
    }

    /// A batch of expert reads submitted to `ring` but not yet awaited — `complete_batch` must
    /// be called (on the SAME `ring`) before the data is valid to decode. Holds the owned
    /// buffers the in-flight SQEs point at, so they can't be dropped out from under the kernel
    /// before completion.
    pub(super) struct Pending {
        reqs: Vec<Req>,
        bufs: Vec<Vec<u8>>,
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

    /// Submits reads for every tensor in `misses` (3/expert) WITHOUT waiting for completion.
    /// Returns `Ok(None)` — not an error — when the ring is unavailable, the batch doesn't fit
    /// in one submission round, or the submission itself fails; every one of these just means
    /// "the caller should fall back to a synchronous load for this batch," same as a hard
    /// `load_batch` failure always meant before this split existed.
    pub(super) fn submit_batch(ring: &mut Option<Ring>, shards: &Shards, cfg: &Cfg, layer: usize, misses: &[usize]) -> Result<Option<Pending>, ModelError> {
        let Some(r) = ring.as_mut() else { return Ok(None) };

        let i = cfg.moe_inter as usize;
        let d = cfg.hidden as usize;

        let mut reqs = Vec::with_capacity(misses.len() * 3);
        for &eid in misses {
            let p = |suf: &str| format!("model.layers.{layer}.mlp.experts.{eid}.{suf}.weight");
            for (suf, rows, cols) in [("gate_proj", i, d), ("up_proj", i, d), ("down_proj", d, i)] {
                let name = p(suf);
                let qs_name = format!("{name}.qs");
                let packed_scale = if shards.has(&qs_name) { Some(shards.read_f32(&qs_name, false)?) } else { None };
                let loc = shards.tensor_location(&name).ok_or(ModelError::Safetensors(SafetensorsError::MissingTensor(name.clone())))?;
                let fp8_scale = if packed_scale.is_none() && loc.dtype == DType::F8E4M3 {
                    let scale_name = format!("{name}_scale_inv");
                    Some(shards.read_f32(&scale_name, false)?)
                } else {
                    None
                };
                reqs.push(Req { rows, cols, loc, packed_scale, fp8_scale });
            }
        }

        if reqs.len() > r.capacity {
            return Ok(None);
        }

        let mut bufs: Vec<Vec<u8>> = reqs.iter().map(|req| vec![0u8; req.loc.nbytes as usize]).collect();

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
        }
        if r.io.submit().is_err() {
            return Ok(None);
        }

        Ok(Some(Pending { reqs, bufs, eids: misses.to_vec() }))
    }

    /// Waits for every read `submit_batch` started, validates each one's byte count, and
    /// decodes the results into `ExpertSlot`s — the other half of `submit_batch`'s split,
    /// called on the same `ring` (`submit_batch` only returns `Some` when the ring exists, so
    /// this never needs its own "ring unavailable" fallback).
    pub(super) fn complete_batch(ring: &mut Ring, pending: Pending, bits: u8, used: u64) -> Result<Vec<ExpertSlot>, std::io::Error> {
        let Pending { reqs, bufs, eids } = pending;

        ring.io.submit_and_wait(reqs.len())?;

        let mut results = vec![None; reqs.len()];
        for cqe in ring.io.completion() {
            results[cqe.user_data() as usize] = Some(cqe.result());
        }
        for (local, req) in reqs.iter().enumerate() {
            match results[local] {
                Some(n) if n as u64 == req.loc.nbytes => {}
                Some(n) if n >= 0 => {
                    let msg = format!("expert read: short read ({n}/{} bytes)", req.loc.nbytes);
                    return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, msg));
                }
                Some(n) => return Err(std::io::Error::from_raw_os_error(-n)),
                None => return Err(std::io::Error::other("expert read: no completion queue entry")),
            }
        }

        // group the 3 reads per expert back into one ExpertSlot each, in `eids` order. A
        // decode error here (bad `.qs`/FP8 format) is a real data problem, not a transient I/O
        // one, so retrying via `sequential_fallback` (as `finish_loading` does on any error
        // from this function) wouldn't help — same bytes, same format problem — but folding it
        // into the same I/O-error path here is simpler than threading a second error type
        // through, and `finish_loading`'s fallback fails the same way through `qt_load`
        // instead: same end result either way.
        let mut out = Vec::with_capacity(eids.len());
        for (chunk, &eid) in eids.iter().enumerate() {
            let base = chunk * 3;
            let decode = |buf: &[u8], req: &Req| qt_from_raw(buf, req, bits).map_err(|e| std::io::Error::other(e.to_string()));
            let gate = decode(&bufs[base], &reqs[base])?;
            let up = decode(&bufs[base + 1], &reqs[base + 1])?;
            let down = decode(&bufs[base + 2], &reqs[base + 2])?;
            out.push(ExpertSlot { eid, gate, up, down, used });
        }
        Ok(out)
    }

    fn qt_from_raw(raw: &[u8], req: &Req, bits: u8) -> Result<QT, ModelError> {
        if let Some(scale) = &req.packed_scale {
            return QT::from_packed(req.rows, req.cols, bits, raw.to_vec(), scale.clone())
                .map_err(|source| ModelError::PackedFormat { name: "expert tensor (.qs)".to_string(), source });
        }
        let w = if let Some(scale) = &req.fp8_scale {
            dequant_fp8_blockscale(raw, req.rows as u64, req.cols as u64, scale, "expert tensor (fp8 scale_inv)")
                .map_err(ModelError::Safetensors)?
        } else {
            Shards::decode_f32(raw, req.loc.dtype)
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
    /// Eagerly loaded via `pin_expert`, checked before `slots` on every lookup, never touched
    /// by `insert`'s LRU eviction — colibrì's `m->pin[layer]`. Structurally separate from
    /// `slots` rather than a "never evict" flag on the same `Vec`, so the existing LRU code in
    /// `insert` needs zero changes to stay correct.
    pinned: Vec<ExpertSlot>,
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
    #[cfg(target_os = "linux")]
    ring: Option<uring_load::Ring>,
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
        ExpertCache {
            capacity,
            slots: Vec::new(),
            pinned: Vec::new(),
            usage: HashMap::new(),
            clock: 0,
            hits: 0,
            misses: 0,
            load_nanos: 0,
            #[cfg(target_os = "linux")]
            ring: uring_load::new_ring(capacity),
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
        let fresh = load_expert(shards, cfg, layer, eid, bits, self.clock)?;
        self.insert(fresh);
        Ok(self.slots.iter().find(|s| s.eid == eid).unwrap())
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
        if let Some(pending) = uring_load::submit_batch(&mut self.ring, shards, cfg, layer, &misses)? {
            self.load_nanos += load_t.elapsed().as_nanos() as u64;
            return Ok(PendingExpertLoad(LoadKind::Async(pending)));
        }

        let loaded = sequential_fallback(shards, cfg, layer, &misses, bits, self.clock)?;
        self.load_nanos += load_t.elapsed().as_nanos() as u64;
        Ok(PendingExpertLoad(LoadKind::Sync(loaded)))
    }

    /// Waits for (if `begin_loading` submitted an async read) and inserts every expert that
    /// call started resolving. `shards`/`cfg`/`layer`/`bits` must match the `begin_loading`
    /// call this `pending` came from — only needed for the rare fallback path (an `io_uring`
    /// completion error retries as a plain synchronous read of the same misses).
    pub fn finish_loading(&mut self, pending: PendingExpertLoad, shards: &Shards, cfg: &Cfg, layer: usize, bits: u8) -> Result<(), ModelError> {
        let loaded = match pending.0 {
            LoadKind::Nothing => return Ok(()),
            LoadKind::Sync(v) => v,
            #[cfg(target_os = "linux")]
            LoadKind::Async(p) => {
                let eids_for_fallback = p.eids_for_fallback();
                let load_t = std::time::Instant::now();
                let ring = self.ring.as_mut().expect("Async pending implies the ring existed when begin_loading submitted it");
                let result = uring_load::complete_batch(ring, p, bits, self.clock);
                self.load_nanos += load_t.elapsed().as_nanos() as u64;
                match result {
                    Ok(v) => v,
                    Err(_) => sequential_fallback(shards, cfg, layer, &eids_for_fallback, bits, self.clock)?,
                }
            }
        };

        for slot in loaded {
            self.insert(slot);
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

    pub(crate) fn is_pinned(&self, eid: usize) -> bool {
        self.pinned.iter().any(|s| s.eid == eid)
    }

    pub fn pinned_len(&self) -> usize {
        self.pinned.len()
    }

    /// Eagerly, synchronously loads `eid` and adds it to the pinned tier (a no-op if already
    /// pinned) — colibrì's `pin_load`, one expert at a time. Reuses the same synchronous
    /// single-expert primitive `get_or_load`'s miss path already calls, rather than inventing a
    /// new loader. `used` is irrelevant for a pinned slot (never evicted, never touched by
    /// `insert`), stamped 0.
    pub(crate) fn pin_expert(&mut self, shards: &Shards, cfg: &Cfg, layer: usize, eid: usize, bits: u8) -> Result<(), ModelError> {
        if self.is_pinned(eid) {
            return Ok(());
        }
        let slot = load_expert(shards, cfg, layer, eid, bits, 0)?;
        mlock_best_effort(&slot);
        self.pinned.push(slot);
        Ok(())
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
fn sequential_fallback(shards: &Shards, cfg: &Cfg, layer: usize, misses: &[usize], bits: u8, used: u64) -> Result<Vec<ExpertSlot>, ModelError> {
    misses.iter().map(|&eid| load_expert(shards, cfg, layer, eid, bits, used)).collect()
}

fn load_expert(shards: &Shards, cfg: &Cfg, layer: usize, eid: usize, bits: u8, used: u64) -> Result<ExpertSlot, ModelError> {
    let i = cfg.moe_inter as usize;
    let d = cfg.hidden as usize;
    let p = |suf: &str| format!("model.layers.{layer}.mlp.experts.{eid}.{suf}.weight");
    let gate = qt_load(shards, &p("gate_proj"), i, d, bits)?;
    let up = qt_load(shards, &p("up_proj"), i, d, bits)?;
    let down = qt_load(shards, &p("down_proj"), d, i, bits)?;
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
    fn pin_expert_is_returned_by_get_and_counted_as_a_hit_not_a_miss() {
        let fixture = build_experts_fixture("rabbit_test_ecache_pin_hit", 3, 4, 6);
        let shards = Shards::open(&fixture.0).unwrap();
        let cfg = tiny_cfg(3, 4, 6);
        let mut cache = ExpertCache::new(4);

        cache.pin_expert(&shards, &cfg, 0, 1, 32).unwrap();
        assert!(cache.is_pinned(1));
        assert_eq!(cache.pinned_len(), 1);
        assert_eq!(cache.hits, 0, "pin_expert itself is not a cache lookup, must not touch hits/misses");
        assert_eq!(cache.misses, 0);
        assert_eq!(cache.len(), 0, "a pinned expert must not also land in the ordinary LRU slots");

        let slot = cache.get(1).unwrap();
        assert_eq!(slot.eid, 1);

        // begin_loading/get_or_load must treat a pinned id as a hit, never as a miss to load.
        cache.begin_loading(&shards, &cfg, 0, &[1], 32).unwrap();
        assert_eq!(cache.hits, 1);
        assert_eq!(cache.misses, 0);

        cache.get_or_load(&shards, &cfg, 0, 1, 32).unwrap();
        assert_eq!(cache.hits, 2);
        assert_eq!(cache.misses, 0);
    }

    #[test]
    fn pinned_experts_survive_lru_pressure_at_capacity_one() {
        let fixture = build_experts_fixture("rabbit_test_ecache_pin_survives_lru", 4, 4, 6);
        let shards = Shards::open(&fixture.0).unwrap();
        let cfg = tiny_cfg(4, 4, 6);
        let mut cache = ExpertCache::new(1); // capacity 1: every OTHER load evicts the LRU slot

        cache.pin_expert(&shards, &cfg, 0, 0, 32).unwrap();
        // hammer the tiny LRU cache with unrelated experts -- must never disturb the pin.
        for eid in [1, 2, 3, 1, 2, 3] {
            cache.get_or_load(&shards, &cfg, 0, eid, 32).unwrap();
        }
        assert!(cache.is_pinned(0));
        assert_eq!(cache.pinned_len(), 1);
        assert!(cache.get(0).is_some(), "pinned expert 0 must still resolve after LRU churn");
    }

    #[test]
    fn get_prefers_pinned_over_a_stale_lru_slot_with_the_same_id() {
        let fixture = build_experts_fixture("rabbit_test_ecache_pin_precedence", 2, 4, 6);
        let shards = Shards::open(&fixture.0).unwrap();
        let cfg = tiny_cfg(2, 4, 6);
        let mut cache = ExpertCache::new(4);

        // load id 0 the normal way first (lands in `slots`), then also pin it -- `get` must
        // still resolve without ambiguity/panics, and pinned lookup must not regress hit
        // counting for the ordinary path.
        cache.get_or_load(&shards, &cfg, 0, 0, 32).unwrap();
        cache.pin_expert(&shards, &cfg, 0, 0, 32).unwrap();
        assert!(cache.get(0).is_some());
        assert!(cache.is_pinned(0));
    }

    #[test]
    fn pin_expert_is_idempotent() {
        let fixture = build_experts_fixture("rabbit_test_ecache_pin_idempotent", 2, 4, 6);
        let shards = Shards::open(&fixture.0).unwrap();
        let cfg = tiny_cfg(2, 4, 6);
        let mut cache = ExpertCache::new(4);

        cache.pin_expert(&shards, &cfg, 0, 0, 32).unwrap();
        cache.pin_expert(&shards, &cfg, 0, 0, 32).unwrap();
        assert_eq!(cache.pinned_len(), 1, "pinning the same id twice must not duplicate the slot");
    }
}
