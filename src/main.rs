//! CLI entrypoint: loads a checkpoint directory (config.json + tokenizer.json + safetensors
//! shards, either plain or `.qs`/FP8-quantized) and either completes a single prompt
//! (`--prompt`) or runs an interactive multi-turn chat (`--chat`).

use rabbit::generate::{self, ExpertCaches, KvState, Rng, SamplingConfig};
use rabbit::model::Model;
use rabbit::safetensors::Shards;
use rabbit::tokenizer::Tokenizer;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "usage: rabbit --model <dir> (--prompt <text> | --chat) [--max-tokens N] \
[--temperature F] [--nucleus F] [--seed N] [--dbits N] [--ebits N] [--expert-cache N] [--think]";

struct Args {
    model_dir: PathBuf,
    prompt: Option<String>,
    chat: bool,
    think: bool,
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
    let mut chat = false;
    let mut think = false;
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
            "--chat" => chat = true,
            "--think" => think = true,
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

    if !chat && prompt.is_none() {
        return Err(format!("either --prompt or --chat is required\n\n{USAGE}"));
    }

    Ok(Args {
        model_dir: model_dir.ok_or_else(|| format!("--model is required\n\n{USAGE}"))?,
        prompt,
        chat,
        think,
        max_tokens,
        temperature,
        nucleus,
        seed,
        dbits,
        ebits,
        cache_capacity,
    })
}

struct Session {
    model: Model,
    shards: Shards,
    tokenizer: Tokenizer,
    caches: ExpertCaches,
    sampling: SamplingConfig,
    rng: Rng,
    stop_ids: Vec<usize>,
    max_tokens: usize,
}

fn load_session(args: &Args) -> Result<Session, Box<dyn std::error::Error>> {
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
    let caches = ExpertCaches::new(&model, args.cache_capacity);
    let sampling = SamplingConfig { temperature: args.temperature, nucleus: args.nucleus };
    let rng = Rng::new(args.seed);
    let stop_ids: Vec<usize> = model.cfg.stop_ids.iter().map(|&id| id as usize).collect();

    Ok(Session { model, shards, tokenizer, caches, sampling, rng, stop_ids, max_tokens: args.max_tokens })
}

/// Forwards `turn_ids` (prefill, continuing from `pos_base` positions already in `kv`) then
/// greedy/samples new tokens until a stop id or `max_tokens`, reporting per-step timing to
/// stderr as it happens — a real-model forward can take seconds to tens of seconds per step,
/// so silent output would be indistinguishable from a hang. Returns the decoded reply text and
/// the new total position (`pos_base + turn_ids.len() + generated.len()`), for the caller to
/// pass back in as the next turn's `pos_base`.
#[allow(clippy::too_many_arguments)]
fn generate_reply(sess: &mut Session, kv: &mut KvState, turn_ids: &[usize], pos_base: usize) -> Result<(String, usize), Box<dyn std::error::Error>> {
    eprintln!("prefill ({} tokens)...", turn_ids.len());
    let t1 = std::time::Instant::now();
    let mut step_t = std::time::Instant::now();
    let mut logits = generate::step(&sess.model, &sess.shards, &mut sess.caches, kv, turn_ids, pos_base)?;
    let (h, m, mut io_ns) = sess.caches.hit_miss_totals();
    eprintln!(
        "  prefill done in {:.1}s (expert cache: {h} hits, {m} misses, {:.1}s in disk I/O)",
        step_t.elapsed().as_secs_f32(),
        io_ns as f64 / 1e9
    );
    let mut pos = pos_base + turn_ids.len();

    let mut out_ids = Vec::with_capacity(sess.max_tokens);
    while out_ids.len() < sess.max_tokens {
        let next = generate::pick_token(&logits, &sess.sampling, &mut sess.rng, None);
        if sess.stop_ids.contains(&next) {
            break;
        }
        out_ids.push(next);
        if out_ids.len() >= sess.max_tokens {
            break;
        }
        let io_ns_before = io_ns;
        step_t = std::time::Instant::now();
        logits = generate::step(&sess.model, &sess.shards, &mut sess.caches, kv, &[next], pos)?;
        let step_elapsed = step_t.elapsed().as_secs_f32();
        let (h, m, io_ns_now) = sess.caches.hit_miss_totals();
        io_ns = io_ns_now;
        eprintln!(
            "  token {}/{} in {:.1}s ({:.1}s in disk I/O this step; expert cache totals: {h} hits, {m} misses)",
            out_ids.len() + 1,
            sess.max_tokens,
            step_elapsed,
            (io_ns - io_ns_before) as f64 / 1e9
        );
        pos += 1;
    }
    let elapsed = t1.elapsed().as_secs_f32();

    let out_i32: Vec<i32> = out_ids.iter().map(|&id| id as i32).collect();
    let text = String::from_utf8_lossy(&sess.tokenizer.decode(&out_i32)).into_owned();
    eprintln!("{} tokens in {:.1}s ({:.1} tok/s)", out_ids.len(), elapsed, out_ids.len() as f32 / elapsed.max(0.001));

    Ok((text, pos))
}

fn run_single_shot(args: &Args, prompt: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut sess = load_session(args)?;
    let prompt_ids: Vec<usize> = sess.tokenizer.encode(prompt).into_iter().map(|id| id as usize).collect();
    eprintln!("prompt: {} tokens", prompt_ids.len());

    let mut kv = KvState::new(&sess.model);
    let (text, _pos) = generate_reply(&mut sess, &mut kv, &prompt_ids, 0)?;
    println!("{text}");
    Ok(())
}

/// GLM-5.2's official chat template (no newline after role tags): `[gMASK]<sop>` opens the
/// FIRST turn only (matches colibri's own `first`-gated prefix, the one other implementation
/// already validated against this exact checkpoint — see the conversation this was built
/// from); every turn is `<|user|>{msg}<|assistant|>{think_tag}`, where `<think></think>`
/// disables GLM-5.2's reasoning block (the model babbles and never emits a stop token with the
/// wrong tag here — colibri's own comment flags this exact failure mode) and bare `<think>`
/// (via `--think`) leaves it open for the model to reason before answering.
fn run_chat(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let mut sess = load_session(args)?;
    let think_tag = if args.think { "<think>" } else { "<think></think>" };

    println!("rabbit chat — escribí un mensaje y Enter. :reset reinicia la conversación, :quit sale.\n");

    let mut kv = KvState::new(&sess.model);
    let mut pos = 0usize;
    let mut first = true;
    let stdin = std::io::stdin();

    loop {
        print!("> ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if stdin.read_line(&mut line)? == 0 {
            break; // EOF (e.g. piped input exhausted)
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == ":quit" || line == ":q" {
            break;
        }
        if line == ":reset" {
            // fresh KV state (conversation-specific) but the SAME ExpertCaches (expert
            // weights don't depend on conversation content, no reason to re-load them).
            kv = KvState::new(&sess.model);
            pos = 0;
            first = true;
            println!("(conversación reiniciada)\n");
            continue;
        }

        let turn_text =
            if first { format!("[gMASK]<sop><|user|>{line}<|assistant|>{think_tag}") } else { format!("<|user|>{line}<|assistant|>{think_tag}") };
        first = false;
        let turn_ids: Vec<usize> = sess.tokenizer.encode(&turn_text).into_iter().map(|id| id as usize).collect();

        let (reply, new_pos) = generate_reply(&mut sess, &mut kv, &turn_ids, pos)?;
        pos = new_pos;
        println!("{}\n", reply.trim());
    }
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
    let result = if args.chat { run_chat(&args) } else { run_single_shot(&args, args.prompt.as_deref().unwrap()) };
    if let Err(e) = result {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
