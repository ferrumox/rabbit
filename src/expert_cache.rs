//! Port of `expert_load`/`ESlot`/the LRU cache slots inside `moe()` — a bounded, per-layer
//! cache of loaded routed-expert weights.
//!
//! Scope cuts vs the original, all purely performance (never correctness) concerns, so the
//! output of `moe.rs` is identical with or without them — any expert fetch, cache hit or disk
//! load, returns the mathematically same weights:
//! - No **pin slots** (`m->pin[layer]`): a deployment feature (`pin_load`/`repin_pass`) for
//!   keeping specific hot experts permanently resident on real hardware, orthogonal to LRU.
//! - No **block-of-64 batching / async readahead** (`expert_prefetch`'s `WILLNEED` hint): the
//!   C bounds its per-block scratch array and overlaps disk reads with compute for the
//!   21,504-expert real model; the tiny oracle's 8 experts/layer make this a no-op either way.
//! - No **`.qs` pre-quantized fast path**: same cut as `model.rs`'s `qt_load` — out of scope
//!   for this whole port stage (see `rabbit-plan.md`).
//! - No **persistent learning cache** (`.coli_usage`/`eusage`/`eheat`): explicitly out of
//!   scope for this stage per the plan.

use crate::config::Cfg;
use crate::model::{ModelError, qt_load};
use crate::quant::QT;
use crate::safetensors::Shards;

pub struct ExpertSlot {
    pub eid: usize,
    pub gate: QT,
    pub up: QT,
    pub down: QT,
    used: u64,
}

/// A fixed-capacity, per-layer LRU cache of `ExpertSlot`s. Real usage holds one of these per
/// MoE layer (`layers_forward`, a later phase); Fase 5 only needs the primitive.
pub struct ExpertCache {
    capacity: usize,
    slots: Vec<ExpertSlot>,
    clock: u64,
    pub hits: u64,
    pub misses: u64,
}

impl ExpertCache {
    pub fn new(capacity: usize) -> ExpertCache {
        ExpertCache { capacity, slots: Vec::new(), clock: 0, hits: 0, misses: 0 }
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
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
        if self.slots.len() < self.capacity {
            self.slots.push(fresh);
            Ok(self.slots.last().unwrap())
        } else {
            let lru = self
                .slots
                .iter()
                .enumerate()
                .min_by_key(|(_, s)| s.used)
                .map(|(i, _)| i)
                .expect("capacity > 0 implies at least one slot once full");
            self.slots[lru] = fresh;
            Ok(&self.slots[lru])
        }
    }
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
}
