//! Empirical quality check: runs the SAME real prompts through two Kimi Linear checkpoints (a
//! "baseline" — meant to be the original unquantized BF16 — and a "candidate" — meant to be a
//! converted/quantized checkpoint) and compares their predictions directly, rather than just
//! estimating per-tensor quantization error. Two signals per prompt:
//!
//!   1. Teacher-forcing argmax agreement over the prompt's own positions: at every position,
//!      do baseline and candidate predict the SAME next token? (cheap — one prefill call each,
//!      one data point per prompt token, independent of whether either prediction is "correct".)
//!   2. Greedy-decode divergence: generate `--gen-tokens` tokens from each checkpoint and
//!      compare the sequences directly — the position of first disagreement (if any), and both
//!      final decoded strings side by side.
//!
//! Usage:
//!   cargo run --release --example kimi_quality_check -- --baseline <dir> --candidate <dir> \
//!       [--baseline-dbits N] [--baseline-ebits N] [--candidate-dbits N] [--candidate-ebits N] \
//!       [--expert-cache N] [--gen-tokens N]

use rabbit::kimi_linear::chat_template::render_turn;
use rabbit::kimi_linear::generate::{self as kgen, ExpertCaches, KvState};
use rabbit::kimi_linear::model::Model;
use rabbit::kimi_linear::tokenizer::Tokenizer;
use rabbit::safetensors::Shards;
use std::path::{Path, PathBuf};
use std::time::Instant;

const PROMPTS: &[&str] = &["What is 2+2?", "What is the capital of France?", "Say hello in one short sentence."];

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned()
}

struct Checkpoint {
    model: Model,
    shards: Shards,
    caches: ExpertCaches,
    tokenizer: Tokenizer,
}

fn load(dir: &Path, dbits: u8, ebits: u8, cache_capacity: usize, label: &str) -> Result<Checkpoint, Box<dyn std::error::Error>> {
    eprintln!("loading {label} ({}): dbits={dbits}, ebits={ebits}...", dir.display());
    let t0 = Instant::now();
    let model = Model::load(dir, dbits, ebits)?;
    let shards = Shards::open(dir)?;
    let caches = ExpertCaches::new(&model, cache_capacity);
    let tokenizer = Tokenizer::load(dir)?;
    eprintln!("  loaded in {:.1}s", t0.elapsed().as_secs_f32());
    Ok(Checkpoint { model, shards, caches, tokenizer })
}

fn argmax(row: &[f32]) -> usize {
    row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap()
}

fn greedy_generate(ck: &mut Checkpoint, prompt_ids: &[usize], max_tokens: usize) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let mut kv = KvState::new(&ck.model);
    let mut ids = prompt_ids.to_vec();
    let mut pos = 0usize;
    let mut generated = Vec::new();
    let stop_ids = ck.model.cfg.stop_ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
    for _ in 0..max_tokens {
        let logits = kgen::step(&ck.model, &ck.shards, &mut ck.caches, &mut kv, &ids, pos)?;
        let next = argmax(&logits);
        pos += ids.len();
        if stop_ids.contains(&next) {
            break;
        }
        generated.push(next);
        ids = vec![next];
    }
    Ok(generated)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let baseline_dir = PathBuf::from(arg_value(&args, "--baseline").expect("usage: --baseline <dir> --candidate <dir> [flags]"));
    let candidate_dir = PathBuf::from(arg_value(&args, "--candidate").expect("usage: --baseline <dir> --candidate <dir> [flags]"));
    let bdbits: u8 = arg_value(&args, "--baseline-dbits").map(|s| s.parse().unwrap()).unwrap_or(16);
    let bebits: u8 = arg_value(&args, "--baseline-ebits").map(|s| s.parse().unwrap()).unwrap_or(16);
    let cdbits: u8 = arg_value(&args, "--candidate-dbits").map(|s| s.parse().unwrap()).unwrap_or(4);
    let cebits: u8 = arg_value(&args, "--candidate-ebits").map(|s| s.parse().unwrap()).unwrap_or(4);
    let cache_capacity: usize = arg_value(&args, "--expert-cache").map(|s| s.parse().unwrap()).unwrap_or(256);
    let gen_tokens: usize = arg_value(&args, "--gen-tokens").map(|s| s.parse().unwrap()).unwrap_or(12);

    let mut baseline = load(&baseline_dir, bdbits, bebits, cache_capacity, "baseline")?;
    let mut candidate = load(&candidate_dir, cdbits, cebits, cache_capacity, "candidate")?;

    let mut total_positions = 0usize;
    let mut agreeing_positions = 0usize;

    for prompt in PROMPTS {
        println!("\n=== prompt: {prompt:?} ===");
        let text = render_turn(prompt, true, false, None);
        let ids: Vec<usize> = baseline.tokenizer.encode(&text).into_iter().map(|id| id as usize).collect();

        let t0 = Instant::now();
        let mut kv_b = KvState::new(&baseline.model);
        let logits_b = kgen::step_all(&baseline.model, &baseline.shards, &mut baseline.caches, &mut kv_b, &ids, 0)?;
        let baseline_t = t0.elapsed().as_secs_f32();

        let t0 = Instant::now();
        let mut kv_c = KvState::new(&candidate.model);
        let logits_c = kgen::step_all(&candidate.model, &candidate.shards, &mut candidate.caches, &mut kv_c, &ids, 0)?;
        let candidate_t = t0.elapsed().as_secs_f32();

        let vocab = baseline.model.cfg.vocab as usize;
        let s = ids.len();
        let mut agree = 0usize;
        for si in 0..s {
            let a = argmax(&logits_b[si * vocab..(si + 1) * vocab]);
            let b = argmax(&logits_c[si * vocab..(si + 1) * vocab]);
            if a == b {
                agree += 1;
            }
        }
        total_positions += s;
        agreeing_positions += agree;
        println!("teacher-forcing argmax agreement: {agree}/{s} positions (baseline {baseline_t:.1}s, candidate {candidate_t:.1}s prefill)");

        let gen_b = greedy_generate(&mut baseline, &ids, gen_tokens)?;
        let gen_c = greedy_generate(&mut candidate, &ids, gen_tokens)?;
        let first_diverge = gen_b.iter().zip(&gen_c).position(|(a, b)| a != b);
        match first_diverge {
            None if gen_b.len() == gen_c.len() => println!("greedy decode: IDENTICAL ({} tokens)", gen_b.len()),
            None => println!("greedy decode: identical up to the shorter length, then one stopped early (baseline {} tokens, candidate {} tokens)", gen_b.len(), gen_c.len()),
            Some(i) => println!("greedy decode: diverges at token {i}"),
        }
        let decode_text = |ck: &Checkpoint, ids: &[usize]| String::from_utf8_lossy(&ck.tokenizer.decode(&ids.iter().map(|&id| id as i32).collect::<Vec<_>>())).to_string();
        println!("  baseline : {:?}", decode_text(&baseline, &gen_b));
        println!("  candidate: {:?}", decode_text(&candidate, &gen_c));
    }

    println!("\n=== overall ===");
    println!(
        "teacher-forcing argmax agreement: {agreeing_positions}/{total_positions} positions ({:.1}%)",
        100.0 * agreeing_positions as f64 / total_positions.max(1) as f64
    );
    // This number reads lower than it means: most prompt-token positions have no single
    // "correct" continuation (teacher-forced mid-sentence, e.g. after "What" or "is"), so
    // baseline/candidate disagreeing there isn't evidence of quantization damage. The greedy-
    // decode comparison above (what each model actually GENERATES) is the signal that matters.

    Ok(())
}
