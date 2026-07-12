//! CLI entrypoint: loads a checkpoint directory (config.json + tokenizer.json + safetensors
//! shards, either plain or `.qs`/FP8-quantized) and greedy/samples a completion for a prompt.

use rabbit::generate::{self, ExpertCaches, KvState, Rng, SamplingConfig};
use rabbit::model::Model;
use rabbit::safetensors::Shards;
use rabbit::tokenizer::Tokenizer;
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "usage: rabbit --model <dir> --prompt <text> [--max-tokens N] [--temperature F] \
[--nucleus F] [--seed N] [--dbits N] [--ebits N] [--expert-cache N]";

struct Args {
    model_dir: PathBuf,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    nucleus: f32,
    seed: u64,
    dbits: u8,
    ebits: u8,
    cache_capacity: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut model_dir = None;
    let mut prompt = None;
    let mut max_tokens = 200usize;
    let mut temperature = 0.0f32;
    let mut nucleus = 0.0f32;
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut dbits = 4u8;
    let mut ebits = 4u8;
    let mut cache_capacity = 64usize;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut next = |flag: &str| args.next().ok_or_else(|| format!("{flag} needs a value"));
        match a.as_str() {
            "--model" => model_dir = Some(PathBuf::from(next("--model")?)),
            "--prompt" => prompt = Some(next("--prompt")?),
            "--max-tokens" => max_tokens = next("--max-tokens")?.parse().map_err(|e| format!("--max-tokens: {e}"))?,
            "--temperature" => temperature = next("--temperature")?.parse().map_err(|e| format!("--temperature: {e}"))?,
            "--nucleus" => nucleus = next("--nucleus")?.parse().map_err(|e| format!("--nucleus: {e}"))?,
            "--seed" => seed = next("--seed")?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--dbits" => dbits = next("--dbits")?.parse().map_err(|e| format!("--dbits: {e}"))?,
            "--ebits" => ebits = next("--ebits")?.parse().map_err(|e| format!("--ebits: {e}"))?,
            "--expert-cache" => cache_capacity = next("--expert-cache")?.parse().map_err(|e| format!("--expert-cache: {e}"))?,
            "-h" | "--help" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown argument: {other}\n\n{USAGE}")),
        }
    }

    Ok(Args {
        model_dir: model_dir.ok_or_else(|| format!("--model is required\n\n{USAGE}"))?,
        prompt: prompt.ok_or_else(|| format!("--prompt is required\n\n{USAGE}"))?,
        max_tokens,
        temperature,
        nucleus,
        seed,
        dbits,
        ebits,
        cache_capacity,
    })
}

fn run(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("loading tokenizer...");
    let tokenizer = Tokenizer::load(&args.model_dir.join("tokenizer.json"))?;

    eprintln!("loading model (dbits={}, ebits={})...", args.dbits, args.ebits);
    let t0 = std::time::Instant::now();
    let model = Model::load(&args.model_dir, args.dbits, args.ebits)?;
    eprintln!(
        "model loaded in {:.1}s ({} layers, has_dsa={})",
        t0.elapsed().as_secs_f32(),
        model.layers.len(),
        model.has_dsa
    );

    let shards = Shards::open(&args.model_dir)?;

    let prompt_ids: Vec<usize> = tokenizer.encode(&args.prompt).into_iter().map(|id| id as usize).collect();
    eprintln!("prompt: {} tokens", prompt_ids.len());

    let mut caches = ExpertCaches::new(&model, args.cache_capacity);
    let mut kv = KvState::new(&model);
    let sampling = SamplingConfig { temperature: args.temperature, nucleus: args.nucleus };
    let mut rng = Rng::new(args.seed);
    let stop_ids: Vec<usize> = model.cfg.stop_ids.iter().map(|&id| id as usize).collect();

    // Inlines `generate::generate`'s own loop (rather than calling it directly) so each step
    // — prefill, then one per new token — reports timing to stderr as it happens. A 744B-param
    // model forwarding on CPU can take a long time per step; silent output until the very end
    // would be indistinguishable from a hang.
    let t1 = std::time::Instant::now();
    eprintln!("prefill ({} tokens)...", prompt_ids.len());
    let mut step_t = std::time::Instant::now();
    let mut logits = generate::step(&model, &shards, &mut caches, &mut kv, &prompt_ids, 0)?;
    let (h, m, mut io_ns) = caches.hit_miss_totals();
    eprintln!(
        "  prefill done in {:.1}s (expert cache: {h} hits, {m} misses, {:.1}s in disk I/O)",
        step_t.elapsed().as_secs_f32(),
        io_ns as f64 / 1e9
    );
    let mut pos = prompt_ids.len();

    let mut out_ids = Vec::with_capacity(args.max_tokens);
    while out_ids.len() < args.max_tokens {
        let next = generate::pick_token(&logits, &sampling, &mut rng, None);
        if stop_ids.contains(&next) {
            break;
        }
        out_ids.push(next);
        if out_ids.len() >= args.max_tokens {
            break;
        }
        let io_ns_before = io_ns;
        step_t = std::time::Instant::now();
        logits = generate::step(&model, &shards, &mut caches, &mut kv, &[next], pos)?;
        let step_elapsed = step_t.elapsed().as_secs_f32();
        let (h, m, io_ns_now) = caches.hit_miss_totals();
        io_ns = io_ns_now;
        eprintln!(
            "  token {}/{} in {:.1}s ({:.1}s in disk I/O this step; expert cache totals: {h} hits, {m} misses)",
            out_ids.len() + 1,
            args.max_tokens,
            step_elapsed,
            (io_ns - io_ns_before) as f64 / 1e9
        );
        pos += 1;
    }
    let elapsed = t1.elapsed().as_secs_f32();

    let out_i32: Vec<i32> = out_ids.iter().map(|&id| id as i32).collect();
    let text = String::from_utf8_lossy(&tokenizer.decode(&out_i32)).into_owned();

    println!("{text}");
    eprintln!("\n{} tokens in {:.1}s ({:.1} tok/s)", out_ids.len(), elapsed, out_ids.len() as f32 / elapsed.max(0.001));
    Ok(())
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("{msg}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = run(&args) {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
