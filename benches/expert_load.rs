//! `cargo bench` — batched `io_uring` expert loading vs sequential `pread`-per-tensor, matching
//! Fase 8's plan: "un microbenchmark de lecturas/segundo contra la versión pread simple".
//!
//! Writes a real on-disk fixture (temp dir, cleaned up on drop) with `n_experts` routed
//! experts at realistic-ish dimensions, then times resolving a batch of fresh misses (via
//! `ExpertCache::clear`, not a new cache per iteration — see the comment below on why that
//! distinction matters for `io_uring`) either way.
//!
//! **Measured result on this dev machine: `io_uring` is NOT faster here** (~80ms vs ~21ms for
//! `sequential_pread`, 32 experts / 192MB) — and that's an honest, expected result of what
//! this benchmark actually exercises, not a sign the Fase 8 implementation is broken (`tests/
//! teacher_forcing.rs` still gets 32/32 through the exact same `io_uring` path, and
//! `expert_cache.rs`'s tests confirm byte-identical output against the sequential path). The
//! fixture is small enough (192MB) to sit entirely in the OS page cache after the first read,
//! so every iteration after that is a **memory-speed** read, not a disk one. `io_uring`'s
//! actual value proposition — collapsing N blocking syscalls (each potentially waiting on real
//! disk latency) into one submission the kernel services concurrently — has nothing to
//! amortize when there's no blocking to begin with; its own per-SQE bookkeeping overhead then
//! shows up as pure cost with no offsetting win. The scenario Fase 8 targets is the real
//! 744B-parameter model's 21,504 experts genuinely streamed from a ~750GB file that cannot
//! fit in a 25GB RAM budget — reproducing that here would mean either a multi-GB fixture (slow
//! to generate, and Linux's page cache would still absorb it on a 105GB-RAM dev box) or
//! O_DIRECT to force real device I/O (deliberately out of scope for this phase — see the
//! module doc in `expert_cache.rs`). Take this benchmark as a correctness/plumbing check and a
//! template for measuring the real win on constrained hardware, not as evidence either way
//! about production performance.

use criterion::{Criterion, criterion_group, criterion_main};
use rabbit::glm52::config::Cfg;
use rabbit::expert_cache::ExpertCache;
use rabbit::safetensors::Shards;
use std::hint::black_box;
use std::path::PathBuf;

struct TempDir(PathBuf);
impl TempDir {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(name);
        std::fs::create_dir_all(&dir).unwrap();
        TempDir(dir)
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
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

/// Writes `n_experts` at layer 0, each with `gate_proj`/`up_proj`/`down_proj` of shape
/// `[moe_inter,hidden]`/`[moe_inter,hidden]`/`[hidden,moe_inter]`.
fn build_experts_fixture(name: &str, n_experts: usize, moe_inter: usize, hidden: usize) -> TempDir {
    let dir = TempDir::new(name);
    let mut seed = 1u32;
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".to_string(), serde_json::json!({"format": "rabbit-bench"}));
    let mut data = Vec::new();
    let mut push = |header: &mut serde_json::Map<String, serde_json::Value>, name: String, rows: usize, cols: usize| {
        let n = rows * cols;
        let vals: Vec<f32> = (0..n).map(|_| xorshift(&mut seed)).collect();
        let bytes = f32_bytes(&vals);
        let start = data.len() as u64;
        data.extend_from_slice(&bytes);
        let end = data.len() as u64;
        header.insert(name, serde_json::json!({"dtype": "F32", "shape": [rows, cols], "data_offsets": [start, end]}));
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
    std::fs::write(dir.0.join("model.safetensors"), out).unwrap();
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

fn bench_expert_batch_load(c: &mut Criterion) {
    // moe_inter=512, hidden=1024 -> ~2MB/expert, closer to a real expert's disk footprint
    // than a synthetic tiny shape, without needing a multi-GB fixture on disk.
    let n_experts = 32;
    let moe_inter = 512;
    let hidden = 1024;
    let fixture = build_experts_fixture("rabbit_bench_expert_load", n_experts, moe_inter, hidden);
    let shards = Shards::open(&fixture.0).unwrap();
    let cfg = tiny_cfg(n_experts as i32, moe_inter as i32, hidden as i32);
    let eids: Vec<usize> = (0..n_experts).collect();

    let mut group = c.benchmark_group("expert_batch_load");
    group.sample_size(20);

    // Cache (and, on Linux, its io_uring ring) created ONCE, outside the timed loop: a fresh
    // ring per call is exactly the bug this benchmark caught in the first place (ring setup
    // dominating over the syscall-count savings it's supposed to measure) — see
    // `uring_load::Ring`'s doc in `expert_cache.rs`. `clear()` forces a real miss (and disk
    // read) every iteration without tearing the ring down.
    let mut io_uring_cache = ExpertCache::new(n_experts);
    group.bench_function("io_uring_batch", |b| {
        b.iter(|| {
            io_uring_cache.clear();
            io_uring_cache.ensure_loaded(&shards, &cfg, 0, black_box(&eids), 32).unwrap();
            black_box(&io_uring_cache);
        })
    });

    let mut sequential_cache = ExpertCache::new(n_experts);
    group.bench_function("sequential_pread", |b| {
        b.iter(|| {
            sequential_cache.clear();
            for &eid in &eids {
                sequential_cache.get_or_load(&shards, &cfg, 0, black_box(eid), 32).unwrap();
            }
            black_box(&sequential_cache);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_expert_batch_load);
criterion_main!(benches);
