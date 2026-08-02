//! Deterministic decode-only timing harness for a real checkpoint.
//!
//! Free-running greedy generation is NOT reproducible run-to-run on this codebase (confirmed
//! 2026-07-22, same session as the AVX-512/AVX2 MLA-absorb port this harness was built to
//! measure): `moe.rs`'s per-expert early-drain (`dispatch_chunk_streaming`, shipped v0.19.0)
//! applies each cache-miss expert's contribution the MOMENT its disk read completes, so the
//! floating-point summation order in `apply_single_expert`'s `out[...] += wgt * hh[...]` depends
//! on real disk I/O completion timing -- which varies by run, independent of any kernel choice.
//! A tiny numeric difference from that can flip an argmax near-tie, and autoregressive greedy
//! decode then diverges onto a completely different continuation, making "generate N tokens and
//! time it" an apples-to-oranges comparison between two runs (even two runs of the SAME binary).
//!
//! This harness sidesteps that: after a real prefill, it feeds a FIXED, precomputed token
//! sequence as the next input at every decode step (teacher forcing) instead of sampling the
//! model's own prediction. `model::step`/`step_all` (the architecture-dispatching wrappers
//! around each family's own generate module) never call `argmax`/`sample` themselves -- only the
//! CLI's own generation loop does -- so driving them directly here means no argmax is ever
//! evaluated, and the input sequence both binaries see is byte-identical by construction.
//! Real disk I/O, real expert routing, real absorbed-attention math still run exactly as they
//! would in a live decode -- only the SOURCE of the next token id is fixed instead of sampled.
//!
//! Usage:
//!   cargo run --release --example teacher_forced_decode_bench -- --model <dir> \
//!       [--prompt <text>] [--steps N] [--dbits N] [--ebits N] [--expert-cache N] \
//!       [--shard-dirs <dir1,dir2,...>] [--numa]
//!
//! The final line includes a **logits fingerprint**: an FNV-1a hash folded over every decode
//! step's full logits vector (raw f32 bits). Two runs that print the same fingerprint computed
//! bit-identical logits at every step — this is the acceptance instrument for scheduling-only
//! changes (`NUMA_AMX_BRIEF.md` invariant 1: same fingerprint with `--numa` on and off, warm).

use rabbit::model::{self, ExpertCaches, KvState, Model, Tokenizer};
use rabbit::safetensors::Shards;
use std::path::PathBuf;
use std::time::Instant;

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model_dir = PathBuf::from(arg_value(&args, "--model").expect("--model <dir> is required"));
    let prompt = arg_value(&args, "--prompt").unwrap_or_else(|| "Write two sentences describing France.".to_string());
    let steps: usize = arg_value(&args, "--steps").and_then(|s| s.parse().ok()).unwrap_or(30);
    let dbits: u8 = arg_value(&args, "--dbits").and_then(|s| s.parse().ok()).unwrap_or(4);
    let ebits: u8 = arg_value(&args, "--ebits").and_then(|s| s.parse().ok()).unwrap_or(4);
    let cache_capacity: usize = arg_value(&args, "--expert-cache").and_then(|s| s.parse().ok()).unwrap_or(64);
    let mut shard_dirs = vec![model_dir.clone()];
    shard_dirs.extend(arg_value(&args, "--shard-dirs").into_iter().flat_map(|s| s.split(',').map(PathBuf::from).collect::<Vec<_>>()));
    if args.iter().any(|a| a == "--numa") {
        // Node pools sized from the global pool's total (which honors RAYON_NUM_THREADS) unless
        // `--numa-threads N` decouples them — the diagnostic that separates "how many unpinned
        // threads should attention get" from "how many pinned threads should the experts get",
        // which the N3 sweep showed are different questions.
        let total = arg_value(&args, "--numa-threads").and_then(|s| s.parse().ok()).unwrap_or_else(rayon::current_num_threads);
        match rabbit::numa::NodePools::init(total) {
            Ok(p) => eprintln!("--numa: {} pinned node pools x {} threads", p.n(), p.threads_per_pool()),
            Err(why) => eprintln!("--numa: {why}"),
        }
    }

    eprintln!("loading model (dbits={dbits}, ebits={ebits})...");
    let t0 = Instant::now();
    let model = Model::load_multi(&shard_dirs, dbits, ebits)?;
    eprintln!("  model loaded in {:.1}s ({} layers)", t0.elapsed().as_secs_f32(), model.n_layers());
    let tokenizer = Tokenizer::load(&model_dir, &model)?;
    let shards = Shards::open_multi(&shard_dirs)?;
    let mut caches = ExpertCaches::new(&model, cache_capacity);

    let prompt_ids: Vec<usize> = tokenizer.encode(&prompt).into_iter().map(|id| id as usize).collect();
    assert!(!prompt_ids.is_empty(), "prompt encoded to zero tokens");
    eprintln!("prompt: {} tokens", prompt_ids.len());

    let mut kv = KvState::new(&model);
    let t_prefill = Instant::now();
    model::step_all(&model, &shards, &mut caches, &mut kv, &prompt_ids, 0)?;
    eprintln!("prefill done in {:.1}s", t_prefill.elapsed().as_secs_f32());

    // Teacher-forced decode: cycle back through the prompt's own token ids as the "next" input
    // at every step, ignoring whatever the model itself would have predicted. Deterministic by
    // construction -- no argmax/sample call anywhere in this loop -- so the exact same sequence
    // of `step` calls (same ids, same positions) happens on every run, on every binary.
    let t_decode = Instant::now();
    let mut fingerprint: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
    let mut phases = rabbit::generate::Phases::default();
    let mut lm_head_s = 0f32;
    for (i, pos) in (0..steps).zip(prompt_ids.len()..) {
        let id = prompt_ids[i % prompt_ids.len()];
        let (logits, prof) = model::step_profiled(&model, &shards, &mut caches, &mut kv, &[id], pos)?;
        phases.attention_s += prof.phases.attention_s;
        phases.attn_kda_proj_s += prof.phases.attn_kda_proj_s;
        phases.attn_kda_recur_s += prof.phases.attn_kda_recur_s;
        phases.attn_mla_s += prof.phases.attn_mla_s;
        phases.expert_wait_s += prof.phases.expert_wait_s;
        phases.expert_matmul_s += prof.phases.expert_matmul_s;
        lm_head_s += prof.lm_head_s;
        for v in logits {
            for b in v.to_bits().to_le_bytes() {
                fingerprint = (fingerprint ^ b as u64).wrapping_mul(0x100000001b3);
            }
        }
        let (hits, misses, _) = caches.hit_miss_totals();
        eprintln!(
            "  step {}/{} at {:.1}s elapsed (expert cache totals: {hits} hits, {misses} misses)",
            i + 1,
            steps,
            t_decode.elapsed().as_secs_f32()
        );
    }
    let decode_elapsed = t_decode.elapsed().as_secs_f32();
    println!("{steps} teacher-forced decode steps in {decode_elapsed:.2}s ({:.4} tok/s), logits fingerprint {fingerprint:016x}", steps as f32 / decode_elapsed);
    // Per-token phase buckets over the decode steps (N5a instrumentation — the same numbers
    // `/profile` serves in --serve, without needing a server boot to read them).
    let per = |v: f32| v / steps as f32;
    println!(
        "per-token buckets: expert {:.3}s (wait {:.3}s), attention {:.3}s (kda_proj {:.3}s, kda_recur {:.3}s, mla {:.3}s), lm_head {:.3}s",
        per(phases.expert_matmul_s),
        per(phases.expert_wait_s),
        per(phases.attention_s),
        per(phases.attn_kda_proj_s),
        per(phases.attn_kda_recur_s),
        per(phases.attn_mla_s),
        per(lm_head_s)
    );
    Ok(())
}
