//! Real-checkpoint smoke test for Kimi Linear — loads a real `kimi_linear` checkpoint directory
//! and runs a prefill + a few greedy decode steps directly against
//! `rabbit::kimi_linear::{model, generate}`, bypassing `rabbit::chat`/the CLI entirely.
//!
//! Why this exists instead of `rabbit --prompt`: `chat.rs` (and therefore `main.rs`'s
//! `--prompt`/`--chat`/`--serve`) is still hardcoded to `glm52::model::Model` and GLM-5.2's own
//! BPE tokenizer — neither the `crate::model` family-dispatch enum nor a Kimi Linear tokenizer
//! port are wired in yet (see `rabbit-plan.md`'s Phase 3 notes). `tests/oracle/make_kimi_oracle.py`
//! already validated correctness against a real (if tiny) instance of Moonshot's own reference
//! implementation — this example is purely about exercising a REAL 48B checkpoint's actual
//! disk-streaming/memory/timing behavior, so it works directly on raw token ids, same as the
//! oracle test does, no tokenizer needed.
//!
//! Usage:
//!   cargo run --release --example kimi_smoke -- --model <dir> [--dbits N] [--ebits N] \
//!       [--expert-cache N] [--prompt-len N] [--max-tokens N]

use rabbit::kimi_linear::generate::{self as kgen, ExpertCaches, KvState};
use rabbit::kimi_linear::model::Model;
use rabbit::safetensors::Shards;
use std::path::PathBuf;
use std::time::Instant;

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned()
}

/// Current process resident set size, in MiB — parsed from `/proc/self/status`'s `VmRSS` line
/// (Linux-only, matching this whole project's Linux-first stance elsewhere, e.g.
/// `expert_cache.rs`'s `io_uring` path). Best-effort: returns 0.0 if unavailable rather than
/// failing the whole smoke test over a diagnostic-only reading.
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
        model.cfg.n_layers,
        model.cfg.hidden,
        model.cfg.n_experts,
        model.cfg.vocab,
        rss_mib()
    );

    let mut caches = ExpertCaches::new(&model, cache_capacity);
    let mut kv = KvState::new(&model);

    // Arbitrary in-range token ids -- no tokenizer available yet (see this file's doc), so this
    // is a structural/performance probe, not a coherence check (that's what the oracle test is
    // for). A fixed xorshift sequence keeps repeated runs comparable to each other.
    let mut seed = 0x9E3779B9u32;
    let mut next_id = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as usize) % (model.cfg.vocab as usize)
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
