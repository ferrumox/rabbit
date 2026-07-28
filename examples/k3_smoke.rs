//! Real-checkpoint smoke test for Kimi K3 — loads a real `kimi_k3` checkpoint directory and runs
//! a prefill + a few greedy decode steps directly against `rabbit::kimi_k3::{model, generate}`,
//! bypassing `rabbit::chat`/the CLI entirely. The sibling of `kimi_smoke.rs`'s Kimi Linear 48B
//! version — same reasoning: works on raw token ids (no tokenizer needed for a structural/
//! performance probe), correctness against Moonshot's own reference implementation is already
//! covered by `tests/teacher_forcing_k3.rs`'s oracle comparison — this example is purely about
//! exercising the REAL 2.8T checkpoint's actual disk-streaming/memory/timing behavior at scale.
//!
//! Usage:
//!   cargo run --release --example k3_smoke -- --model <dir> [--dbits N] [--ebits N] \
//!       [--expert-cache N] [--prompt-len N] [--max-tokens N]

use rabbit::kimi_k3::generate::{self as kgen, ExpertCaches, KvState};
use rabbit::kimi_k3::model::Model;
use rabbit::safetensors::Shards;
use std::path::PathBuf;
use std::time::Instant;

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned()
}

/// Current process resident set size, in MiB — see `kimi_smoke.rs::rss_mib`'s doc (identical,
/// duplicated rather than shared: a one-function example-only helper isn't worth a shared module).
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
    let model_dir = PathBuf::from(arg_value(&args, "--model").expect("usage: --model <dir> [--dbits N] [--ebits N] [--expert-cache N] [--prompt-len N] [--max-tokens N]"));
    let dbits: u8 = arg_value(&args, "--dbits").map(|s| s.parse().unwrap()).unwrap_or(4);
    let ebits: u8 = arg_value(&args, "--ebits").map(|s| s.parse().unwrap()).unwrap_or(4);
    let cache_capacity: usize = arg_value(&args, "--expert-cache").map(|s| s.parse().unwrap()).unwrap_or(64);
    let prompt_len: usize = arg_value(&args, "--prompt-len").map(|s| s.parse().unwrap()).unwrap_or(16);
    let max_tokens: usize = arg_value(&args, "--max-tokens").map(|s| s.parse().unwrap()).unwrap_or(20);

    eprintln!("loading {} (dbits={dbits}, ebits={ebits}, expert-cache={cache_capacity})...", model_dir.display());
    let t0 = Instant::now();
    let model = Model::load(&model_dir, dbits, ebits)?;
    let shards = Shards::open(&model_dir)?;
    eprintln!(
        "loaded in {:.1}s — {} layers, hidden={}, {} experts/layer, vocab={} (RSS: {:.0} MiB)",
        t0.elapsed().as_secs_f32(),
        model.cfg.base.n_layers,
        model.cfg.base.hidden,
        model.cfg.base.n_experts,
        model.cfg.base.vocab,
        rss_mib()
    );

    let mut caches = ExpertCaches::new(&model, cache_capacity);
    let mut kv = KvState::new(&model);

    // Arbitrary in-range token ids -- no tokenizer wired here (see this file's doc), so this is
    // a structural/performance probe, not a coherence check. A fixed xorshift sequence keeps
    // repeated runs comparable to each other.
    let mut seed = 0x9E3779B9u32;
    let mut next_id = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as usize) % (model.cfg.base.vocab as usize)
    };
    let prompt: Vec<usize> = (0..prompt_len).map(|_| next_id()).collect();

    eprintln!("\nprefill: {} tokens...", prompt.len());
    let t1 = Instant::now();
    let mut logits = kgen::step(&model, &shards, &mut caches, &mut kv, &prompt, 0)?;
    eprintln!("  done in {:.2}s (RSS: {:.0} MiB)", t1.elapsed().as_secs_f32(), rss_mib());

    let mut total_decode = 0f32;
    for (i, pos) in (prompt.len()..prompt.len() + max_tokens).enumerate() {
        let next = rabbit::generate::argmax(&logits);
        let t = Instant::now();
        logits = kgen::step(&model, &shards, &mut caches, &mut kv, &[next], pos)?;
        let step_s = t.elapsed().as_secs_f32();
        total_decode += step_s;
        eprintln!("  token {}/{max_tokens} (id={next}) in {step_s:.2}s", i + 1);
    }

    eprintln!(
        "\n{max_tokens} tokens in {total_decode:.1}s ({:.2} tok/s) — final RSS: {:.0} MiB",
        max_tokens as f32 / total_decode.max(0.001),
        rss_mib()
    );
    Ok(())
}
