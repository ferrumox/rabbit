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
//!       [--shard-dirs <dir1,dir2,...>]

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
    for (i, pos) in (0..steps).zip(prompt_ids.len()..) {
        let id = prompt_ids[i % prompt_ids.len()];
        model::step(&model, &shards, &mut caches, &mut kv, &[id], pos)?;
        let (hits, misses, _) = caches.hit_miss_totals();
        eprintln!(
            "  step {}/{} at {:.1}s elapsed (expert cache totals: {hits} hits, {misses} misses)",
            i + 1,
            steps,
            t_decode.elapsed().as_secs_f32()
        );
    }
    let decode_elapsed = t_decode.elapsed().as_secs_f32();
    println!("{steps} teacher-forced decode steps in {decode_elapsed:.2}s ({:.4} tok/s)", steps as f32 / decode_elapsed);
    Ok(())
}
