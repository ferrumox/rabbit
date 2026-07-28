//! CLI entrypoint: loads a checkpoint directory (config.json + tokenizer.json + safetensors
//! shards, either plain or `.qs`/FP8-quantized) and either completes a single prompt
//! (`--prompt`), runs an interactive multi-turn chat (`--chat`), or serves an OpenAI-compatible
//! HTTP API (`--serve`). The actual session/generation/template logic lives in `rabbit::chat`
//! (shared with `rabbit::server`); this file is just argument parsing and thin CLI wrappers.

use rabbit::chat::{self, GenEvent, KvState, LoadArgs, Session};
use rabbit::server::{self, ServeConfig};
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "usage: rabbit --model <dir> (--prompt <text> | --chat | --serve) [--max-tokens N] \
[--temperature F] [--nucleus F] [--seed N] [--dbits N] [--ebits N] [--expert-cache N] [--think] \
[--session <path>] [--no-usage-cache] [--cache-route] [--threads N] [--host H] [--port N] [--api-key K]";

struct Args {
    model_dir: PathBuf,
    prompt: Option<String>,
    chat: bool,
    serve: bool,
    think: bool,
    max_tokens: usize,
    temperature: f32,
    nucleus: f32,
    seed: u64,
    dbits: u8,
    ebits: u8,
    cache_capacity: usize,
    session: Option<PathBuf>,
    no_usage_cache: bool,
    cache_route: bool,
    /// `None` = use the default (physical core count — see `main`'s call to
    /// `rayon::ThreadPoolBuilder`). `--threads` overrides it explicitly.
    threads: Option<usize>,
    host: String,
    port: u16,
    api_key: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut model_dir = None;
    let mut prompt = None;
    let mut chat = false;
    let mut serve = false;
    let mut think = false;
    let mut max_tokens = 200usize;
    let mut temperature = 0.0f32;
    let mut nucleus = 0.0f32;
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut dbits = 4u8;
    let mut ebits = 4u8;
    let mut cache_capacity = 64usize;
    let mut session = None;
    let mut no_usage_cache = false;
    let mut cache_route = false;
    let mut threads = None;
    let mut host = "127.0.0.1".to_string();
    let mut port = 8000u16;
    let mut api_key = None;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut next = |flag: &str| args.next().ok_or_else(|| format!("{flag} needs a value"));
        match a.as_str() {
            "--model" => model_dir = Some(PathBuf::from(next("--model")?)),
            "--prompt" => prompt = Some(next("--prompt")?),
            "--chat" => chat = true,
            "--serve" => serve = true,
            "--think" => think = true,
            "--max-tokens" => max_tokens = next("--max-tokens")?.parse().map_err(|e| format!("--max-tokens: {e}"))?,
            "--temperature" => temperature = next("--temperature")?.parse().map_err(|e| format!("--temperature: {e}"))?,
            "--nucleus" => nucleus = next("--nucleus")?.parse().map_err(|e| format!("--nucleus: {e}"))?,
            "--seed" => seed = next("--seed")?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--dbits" => dbits = next("--dbits")?.parse().map_err(|e| format!("--dbits: {e}"))?,
            "--ebits" => ebits = next("--ebits")?.parse().map_err(|e| format!("--ebits: {e}"))?,
            "--expert-cache" => cache_capacity = next("--expert-cache")?.parse().map_err(|e| format!("--expert-cache: {e}"))?,
            "--session" => session = Some(PathBuf::from(next("--session")?)),
            "--no-usage-cache" => no_usage_cache = true,
            "--cache-route" => cache_route = true,
            "--threads" => threads = Some(next("--threads")?.parse().map_err(|e| format!("--threads: {e}"))?),
            "--host" => host = next("--host")?,
            "--port" => port = next("--port")?.parse().map_err(|e| format!("--port: {e}"))?,
            "--api-key" => api_key = Some(next("--api-key")?),
            "-h" | "--help" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown argument: {other}\n\n{USAGE}")),
        }
    }

    if [chat, serve, prompt.is_some()].iter().filter(|&&x| x).count() != 1 {
        return Err(format!("exactly one of --prompt, --chat, or --serve is required\n\n{USAGE}"));
    }
    if session.is_some() && !chat {
        return Err(format!("--session is only valid with --chat\n\n{USAGE}"));
    }

    Ok(Args {
        model_dir: model_dir.ok_or_else(|| format!("--model is required\n\n{USAGE}"))?,
        prompt,
        chat,
        serve,
        think,
        max_tokens,
        temperature,
        nucleus,
        seed,
        dbits,
        ebits,
        cache_capacity,
        session,
        no_usage_cache,
        cache_route,
        threads,
        host,
        port,
        api_key,
    })
}

fn load_args(args: &Args) -> LoadArgs {
    LoadArgs {
        model_dir: args.model_dir.clone(),
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        nucleus: args.nucleus,
        seed: args.seed,
        dbits: args.dbits,
        ebits: args.ebits,
        cache_capacity: args.cache_capacity,
        no_usage_cache: args.no_usage_cache,
        cache_route: args.cache_route,
    }
}

/// Prints a `GenEvent` to stderr exactly as `main.rs` always has — a real-model forward can
/// take seconds to tens of seconds per step, so silent output would be indistinguishable from
/// a hang.
fn print_progress(ev: GenEvent) {
    match ev {
        GenEvent::Prefill { tokens, seconds, hits, misses, io_seconds, io_wait_seconds } => {
            eprintln!("prefill ({tokens} tokens)...");
            eprintln!(
                "  prefill done in {seconds:.1}s (expert cache: {hits} hits, {misses} misses, {io_seconds:.1}s in disk I/O [{io_wait_seconds:.1}s actual disk wait])"
            );
        }
        GenEvent::Token { index, max, seconds, hits, misses, io_seconds, io_wait_seconds, .. } => {
            eprintln!(
                "  token {index}/{max} in {seconds:.1}s ({io_seconds:.1}s in disk I/O [{io_wait_seconds:.1}s actual disk wait] this step; expert cache totals: {hits} hits, {misses} misses)"
            );
        }
    }
}

fn run_single_shot(args: &Args, prompt: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut sess = chat::load_session(&load_args(args))?;
    let prompt_ids: Vec<usize> = sess.tokenizer.encode(prompt).into_iter().map(|id| id as usize).collect();
    eprintln!("prompt: {} tokens", prompt_ids.len());

    let mut kv = KvState::new(&sess.model);
    let t1 = std::time::Instant::now();
    let (text, _pos, n, _profile) = chat::generate_reply(&mut sess, &mut kv, &prompt_ids, 0, print_progress)?;
    let elapsed = t1.elapsed().as_secs_f32();
    println!("{text}");
    eprintln!("\n{n} tokens in {elapsed:.1}s ({:.1} tok/s)", n as f32 / elapsed.max(0.001));

    if sess.usage_cache_enabled
        && let Err(e) = sess.caches.save_usage(&sess.model_dir)
    {
        eprintln!("(no se pudo guardar el usage cache: {e})");
    }
    Ok(())
}

fn run_chat(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let mut sess = chat::load_session(&load_args(args))?;
    let think_tag_note = if args.think { " (think)" } else { "" };

    println!("rabbit chat{think_tag_note} — escribí un mensaje y Enter. :reset reinicia la conversación, :quit sale.\n");

    let (mut kv, mut pos) = match &args.session {
        Some(path) if path.exists() => {
            let (kv, pos) = KvState::load(path, &sess.model)?;
            println!("(sesión retomada de {}: {pos} tokens de contexto)\n", path.display());
            (kv, pos)
        }
        _ => (KvState::new(&sess.model), 0usize),
    };
    let mut first = pos == 0;
    // becomes false if a `:reset` fails to delete the session file — keeps the interactive
    // session alive instead of killing it, but stops appending under a stale position.
    let mut session_persists = true;
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
            if let Some(path) = &args.session
                && path.exists()
                && let Err(e) = std::fs::remove_file(path)
            {
                eprintln!(
                    "(no se pudo borrar el archivo de sesión, se desactiva la persistencia para el resto de esta corrida: {e})"
                );
                session_persists = false;
            }
            println!("(conversación reiniciada)\n");
            continue;
        }

        let turn_text = chat::render_turn(line, first, args.think, None);
        first = false;
        let turn_ids: Vec<usize> = sess.tokenizer.encode(&turn_text).into_iter().map(|id| id as usize).collect();

        let from_pos = pos;
        let t1 = std::time::Instant::now();
        let (reply, new_pos, n, _profile) = chat::generate_reply(&mut sess, &mut kv, &turn_ids, pos, print_progress)?;
        let elapsed = t1.elapsed().as_secs_f32();
        eprintln!("{n} tokens in {elapsed:.1}s ({:.1} tok/s)", n as f32 / elapsed.max(0.001));
        pos = new_pos;

        if session_persists
            && let Some(path) = &args.session
            && let Err(e) = kv.save(from_pos, pos, path)
        {
            eprintln!("(no se pudo guardar la sesión: {e})");
        }

        if sess.usage_cache_enabled
            && let Err(e) = sess.caches.save_usage(&sess.model_dir)
        {
            eprintln!("(no se pudo guardar el usage cache: {e})");
        }

        println!("{}\n", reply.trim());
    }
    Ok(())
}

fn run_serve(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let sess: Session = chat::load_session(&load_args(args))?;
    let cfg = ServeConfig {
        host: args.host.clone(),
        port: args.port,
        api_key: args.api_key.clone(),
        model_id: args.model_dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "rabbit".to_string()),
        think: args.think,
        max_tokens_cap: args.max_tokens,
    };
    server::serve(sess, cfg)
}

/// Sizes the global `rayon` pool to physical cores by default, not `std::thread::available_
/// parallelism()`'s logical count (which includes SMT/hyperthread siblings) — measured on this
/// project's own dev box (12 physical / 24 logical): running the default 24 threads vs 12
/// physical-only gave the SAME wall-clock decode speed but roughly HALF the total CPU-seconds
/// (`user` time), confirming this workload's bottleneck is memory/disk bandwidth, not compute
/// throughput SMT could help with — see `PERFORMANCE.md`. `--threads` overrides explicitly for
/// anyone who wants to test otherwise on different hardware. Only ever called once, before any
/// generation work starts, so a `build_global` failure (already-initialized pool) can't happen
/// in practice; logged rather than panicking just in case.
fn configure_thread_pool(threads: Option<usize>) {
    let n = threads.unwrap_or_else(|| num_cpus::get_physical().max(1));
    if let Err(e) = rayon::ThreadPoolBuilder::new().num_threads(n).build_global() {
        eprintln!("warning: failed to configure {n}-thread rayon pool ({e}), using the default");
    } else {
        eprintln!("using {n} threads ({} physical cores detected)", num_cpus::get_physical());
    }
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("{msg}");
            return ExitCode::FAILURE;
        }
    };
    configure_thread_pool(args.threads);
    let result = if args.serve {
        run_serve(&args)
    } else if args.chat {
        run_chat(&args)
    } else {
        run_single_shot(&args, args.prompt.as_deref().unwrap())
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
