//! Real-checkpoint smoke test for Qwen 3.8 — loads a real `qwen3_5_moe_text` checkpoint directory
//! and runs a prefill plus a few greedy decode steps directly against
//! `rabbit::qwen38::{model, generate}`, bypassing `rabbit::chat`/the CLI. The sibling of
//! `k3_smoke.rs`, for the same reason: it works on raw token ids, so it can measure the real
//! checkpoint's disk-streaming/memory/timing behavior without the tokenizer or chat template being
//! involved at all. Coherence is the CLI's job (`--prompt`), not this file's.
//!
//! Reports the two numbers that decide whether this model is usable on this machine: seconds per
//! decode token, and how much of that was pure disk wait (the expert cache's own `io_uring`
//! accounting) rather than compute. On the real 2.4T checkpoint every decode token needs ~24.6 GB
//! of MXFP4 routed experts read from disk, so that split is the whole story.
//!
//! Usage:
//!   cargo run --release --example qwen38_smoke -- --model <dir> [--shard-dirs <d1,d2>] \
//!       [--dbits N] [--ebits N] [--expert-cache N] [--prompt-len N] [--max-tokens N]

use rabbit::qwen38::generate::{self as qgen, ExpertCaches, KvState};
use rabbit::qwen38::model::Model;
use rabbit::safetensors::Shards;
use std::path::PathBuf;
use std::time::Instant;

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned()
}

/// Current process resident set size, in MiB — same one-function helper `k3_smoke.rs`/`kimi_smoke.rs`
/// each carry (duplicated rather than shared: not worth a module for three examples).
fn rss_mib() -> f64 {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else { return 0.0 };
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:")
            && let Some(kb) = rest.trim().strip_suffix(" kB").and_then(|s| s.trim().parse::<f64>().ok())
        {
            return kb / 1024.0;
        }
    }
    0.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model_dir = PathBuf::from(
        arg_value(&args, "--model")
            .expect("usage: --model <dir> [--shard-dirs <d1,d2>] [--dbits N] [--ebits N] [--expert-cache N] [--prompt-len N] [--max-tokens N]"),
    );
    let dbits: u8 = arg_value(&args, "--dbits").map(|s| s.parse().unwrap()).unwrap_or(4);
    let ebits: u8 = arg_value(&args, "--ebits").map(|s| s.parse().unwrap()).unwrap_or(4);
    let prompt_len: usize = arg_value(&args, "--prompt-len").map(|s| s.parse().unwrap()).unwrap_or(8);
    let max_tokens: usize = arg_value(&args, "--max-tokens").map(|s| s.parse().unwrap()).unwrap_or(10);

    // The real checkpoint lives across both drives here, so this example needs the same
    // `--shard-dirs` the CLI has (config.json still comes from `--model` alone).
    let mut dirs = vec![model_dir.clone()];
    if let Some(extra) = arg_value(&args, "--shard-dirs") {
        dirs.extend(extra.split(',').map(PathBuf::from));
    }

    eprintln!("loading {} (dbits={dbits}, ebits={ebits})...", model_dir.display());
    let t0 = Instant::now();
    let model = Model::load_multi(&dirs, dbits, ebits)?;
    let shards = Shards::open_multi(&dirs)?;
    let cfg = &model.cfg;
    let gdn_layers = cfg.is_linear.iter().filter(|&&l| l).count();
    eprintln!(
        "loaded in {:.1}s — {} layers ({gdn_layers} GDN / {} attention), hidden={}, {} experts/layer top-{}, vocab={} (RSS: {:.0} MiB)",
        t0.elapsed().as_secs_f32(),
        cfg.n_layers,
        cfg.n_full_attn_layers(),
        cfg.hidden,
        cfg.n_experts,
        cfg.topk,
        cfg.vocab,
        rss_mib()
    );

    // `--expert-cache` defaults to the same RAM-aware clamp the CLI applies, rather than a flat
    // number that would OOM at this checkpoint's 28 MB-per-expert scale.
    let cache_capacity: usize = match arg_value(&args, "--expert-cache") {
        Some(s) => s.parse().unwrap(),
        None => rabbit::expert_cache::safe_mxfp4_capacity(64, cfg.n_layers as usize, cfg.moe_inter as usize, cfg.hidden as usize),
    };
    eprintln!("expert cache capacity: {cache_capacity} per layer");
    let mut caches = ExpertCaches::new(&model, cache_capacity);
    let mut kv = KvState::new(&model);

    // Arbitrary in-range token ids — no tokenizer here (see this file's doc). A fixed xorshift
    // sequence keeps repeated runs comparable to each other.
    let mut seed = 0x9E3779B9u32;
    let mut next_id = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as usize) % (cfg.vocab as usize)
    };
    let prompt: Vec<usize> = (0..prompt_len).map(|_| next_id()).collect();

    eprintln!("\nprefill: {} tokens...", prompt.len());
    let t1 = Instant::now();
    let mut logits = qgen::step(&model, &shards, &mut caches, &mut kv, &prompt, 0)?;
    let (hits, misses, _) = caches.hit_miss_totals();
    eprintln!(
        "  done in {:.2}s (expert cache: {hits} hits, {misses} misses, {:.1}s disk wait; RSS: {:.0} MiB)",
        t1.elapsed().as_secs_f32(),
        caches.io_wait_nanos_total() as f32 / 1e9,
        rss_mib()
    );

    let mut total_decode = 0f32;
    let mut wait_before = caches.io_wait_nanos_total();
    for (i, pos) in (prompt.len()..prompt.len() + max_tokens).enumerate() {
        let next = rabbit::generate::argmax(&logits);
        let t = Instant::now();
        logits = qgen::step(&model, &shards, &mut caches, &mut kv, &[next], pos)?;
        let step_s = t.elapsed().as_secs_f32();
        total_decode += step_s;
        let wait_now = caches.io_wait_nanos_total();
        eprintln!(
            "  token {}/{max_tokens} (id={next}) in {step_s:.2}s ({:.2}s disk wait)",
            i + 1,
            (wait_now - wait_before) as f32 / 1e9
        );
        wait_before = wait_now;
    }

    let (hits, misses, _) = caches.hit_miss_totals();
    eprintln!(
        "\n{max_tokens} tokens in {total_decode:.1}s ({:.2}s/token, {:.3} tok/s) — cache {hits} hits / {misses} misses, final RSS: {:.0} MiB",
        total_decode / max_tokens.max(1) as f32,
        max_tokens as f32 / total_decode.max(0.001),
        rss_mib()
    );
    Ok(())
}
