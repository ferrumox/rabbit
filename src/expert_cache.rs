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
//! - No **persistent learning cache** (`.coli_usage`/`eusage`/`eheat`): explicitly out of
//!   scope for this stage per the plan.

use crate::config::Cfg;
use crate::model::{ModelError, qt_load};
use crate::quant::QT;
use crate::safetensors::Shards;

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

    /// A persistent `io_uring` instance, reused across every `load_batch` call for as long as
    /// the owning `ExpertCache` lives. Creating a ring isn't free (`io_uring_setup` + mmap'ing
    /// the shared SQ/CQ memory) — the first version of this phase created a fresh one per
    /// call and the `cargo bench` numbers came out SLOWER than plain sequential `pread`,
    /// entirely from that per-call setup cost swallowing the syscall-count savings. `capacity`
    /// is fixed at creation (io_uring rings don't resize); `load_batch` submits in chunks when
    /// a batch is larger than that.
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
    /// `load_batch` call then falls back to sequential `pread`, same as a hard per-call
    /// failure would.
    pub(super) fn new_ring(cache_capacity: usize) -> Option<Ring> {
        Ring::new(cache_capacity * 3).ok()
    }

    /// Loads every expert in `misses` (3 tensors each: gate/up/down) via `io_uring`,
    /// submitted in as few rounds as `ring`'s fixed capacity allows. Falls back to the
    /// sequential loader if `ring` is `None` (see `new_ring`) or a submission fails outright.
    pub(super) fn load_batch(ring: &mut Option<Ring>, shards: &Shards, cfg: &Cfg, layer: usize, misses: &[usize], bits: u8, used: u64) -> Result<Vec<ExpertSlot>, ModelError> {
        let Some(r) = ring.as_mut() else {
            return super::sequential_fallback(shards, cfg, layer, misses, bits, used);
        };

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

        let mut bufs: Vec<Vec<u8>> = reqs.iter().map(|req| vec![0u8; req.loc.nbytes as usize]).collect();

        if submit_chunked(r, &reqs, &mut bufs).is_err() {
            return super::sequential_fallback(shards, cfg, layer, misses, bits, used);
        }

        // group the 3 reads per expert back into one ExpertSlot each, in `misses` order.
        let mut out = Vec::with_capacity(misses.len());
        for (chunk, &eid) in misses.iter().enumerate() {
            let base = chunk * 3;
            let gate = qt_from_raw(&bufs[base], &reqs[base], bits)?;
            let up = qt_from_raw(&bufs[base + 1], &reqs[base + 1], bits)?;
            let down = qt_from_raw(&bufs[base + 2], &reqs[base + 2], bits)?;
            out.push(ExpertSlot { eid, gate, up, down, used });
        }
        Ok(out)
    }

    /// Submits `reqs` on `ring` in chunks no larger than its fixed capacity, fully draining
    /// each chunk's completions (and checking every read's byte count) before submitting the
    /// next — so the ring is always back to empty when this returns, ready for the next call.
    fn submit_chunked(ring: &mut Ring, reqs: &[Req], bufs: &mut [Vec<u8>]) -> Result<(), std::io::Error> {
        for start in (0..reqs.len()).step_by(ring.capacity) {
            let end = (start + ring.capacity).min(reqs.len());
            let chunk = &reqs[start..end];
            let chunk_bufs = &mut bufs[start..end];

            {
                let mut sq = ring.io.submission();
                for (local, req) in chunk.iter().enumerate() {
                    let buf = &mut chunk_bufs[local];
                    let read_e = opcode::Read::new(types::Fd(req.loc.fd), buf.as_mut_ptr(), buf.len() as u32).offset(req.loc.offset).build().user_data(local as u64);
                    // Safety: `buf` stays alive (owned by `bufs`, held by the caller) until
                    // this chunk's completions are fully reaped below, satisfying the SQE's
                    // lifetime requirement; `push` never fails since the chunk never exceeds
                    // the ring's own capacity.
                    unsafe {
                        sq.push(&read_e).expect("chunk sized to ring capacity");
                    }
                }
            }
            ring.io.submit_and_wait(chunk.len())?;

            let mut results = vec![None; chunk.len()];
            for cqe in ring.io.completion() {
                results[cqe.user_data() as usize] = Some(cqe.result());
            }
            for (local, req) in chunk.iter().enumerate() {
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
        }
        Ok(())
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

/// A fixed-capacity, per-layer LRU cache of `ExpertSlot`s. Real usage holds one of these per
/// MoE layer (`layers_forward`), living for the whole generation session — which matters here
/// specifically because the `io_uring` ring (Linux) is created once in `new` and reused for
/// every `ensure_loaded` call over that lifetime; see `uring_load::Ring`'s doc for why.
pub struct ExpertCache {
    capacity: usize,
    slots: Vec<ExpertSlot>,
    clock: u64,
    pub hits: u64,
    pub misses: u64,
    #[cfg(target_os = "linux")]
    ring: Option<uring_load::Ring>,
}

impl ExpertCache {
    pub fn new(capacity: usize) -> ExpertCache {
        ExpertCache {
            capacity,
            slots: Vec::new(),
            clock: 0,
            hits: 0,
            misses: 0,
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
        self.clock += 1;
        let mut misses = Vec::new();
        for &eid in eids {
            if let Some(pos) = self.slots.iter().position(|s| s.eid == eid) {
                self.hits += 1;
                self.slots[pos].used = self.clock;
            } else if !misses.contains(&eid) {
                misses.push(eid);
            }
        }
        if misses.is_empty() {
            return Ok(());
        }
        self.misses += misses.len() as u64;

        #[cfg(target_os = "linux")]
        let loaded = uring_load::load_batch(&mut self.ring, shards, cfg, layer, &misses, bits, self.clock)?;
        #[cfg(not(target_os = "linux"))]
        let loaded = sequential_fallback(shards, cfg, layer, &misses, bits, self.clock)?;

        for slot in loaded {
            self.insert(slot);
        }
        Ok(())
    }

    /// Looks up an already-cached slot without touching LRU state or loading anything —
    /// pairs with `ensure_loaded`, which the caller runs first for a whole batch.
    pub fn get(&self, eid: usize) -> Option<&ExpertSlot> {
        self.slots.iter().find(|s| s.eid == eid)
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

/// Used both as `ensure_loaded`'s non-Linux fallback and as `uring_load::load_batch`'s
/// fallback when the ring itself can't be created.
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
}
