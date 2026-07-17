//! Shared session/generation/template logic for both the CLI's `--chat` mode and the HTTP
//! server (`server.rs`) — extracted from `main.rs` so neither has to duplicate it.

use crate::generate::{self, ExpertCaches, Rng, SamplingConfig};
pub use crate::generate::KvState;
use crate::model::Model;
use crate::safetensors::Shards;
use crate::tokenizer::Tokenizer;
use std::path::PathBuf;

pub struct LoadArgs {
    pub model_dir: PathBuf,
    pub max_tokens: usize,
    pub temperature: f32,
    pub nucleus: f32,
    pub seed: u64,
    pub dbits: u8,
    pub ebits: u8,
    pub cache_capacity: usize,
    pub no_usage_cache: bool,
    /// opt-in CACHE_ROUTE (see `moe::RouteConfig`) — off by default, matching colibrì's own
    /// stance, since it's still unmeasured on rabbit's own architecture.
    pub cache_route: bool,
}

pub struct Session {
    pub model: Model,
    pub shards: Shards,
    pub tokenizer: Tokenizer,
    pub caches: ExpertCaches,
    pub sampling: SamplingConfig,
    pub rng: Rng,
    pub stop_ids: Vec<usize>,
    pub max_tokens: usize,
    pub model_dir: PathBuf,
    pub usage_cache_enabled: bool,
}

pub fn load_session(args: &LoadArgs) -> Result<Session, Box<dyn std::error::Error>> {
    eprintln!("loading tokenizer...");
    let tokenizer = Tokenizer::load(&args.model_dir.join("tokenizer.json"))?;

    eprintln!("loading model (dbits={}, ebits={})...", args.dbits, args.ebits);
    let t0 = std::time::Instant::now();
    let mut model = Model::load(&args.model_dir, args.dbits, args.ebits)?;
    model.route_cfg.cache_route = args.cache_route;
    eprintln!(
        "model loaded in {:.1}s ({} layers, has_dsa={}), cache_route={}",
        t0.elapsed().as_secs_f32(),
        model.layers.len(),
        model.has_dsa,
        model.route_cfg.cache_route
    );

    let shards = Shards::open(&args.model_dir)?;
    let mut caches = ExpertCaches::new(&model, args.cache_capacity);
    let usage_cache_enabled = !args.no_usage_cache;
    if usage_cache_enabled {
        let stats = caches.warm_start(&args.model_dir, args.cache_capacity);
        if stats.pin_candidates > 0 {
            eprintln!(
                "usage cache: {} historical selections, confidence {:.2}, {} pin candidates marked",
                stats.hist, stats.confidence, stats.pin_candidates
            );
        } else if stats.hist > 0 {
            eprintln!("usage cache: {} historical selections (below auto-pin threshold)", stats.hist);
        }
    }
    let sampling = SamplingConfig { temperature: args.temperature, nucleus: args.nucleus };
    let rng = Rng::new(args.seed);
    let stop_ids: Vec<usize> = model.cfg.stop_ids.iter().map(|&id| id as usize).collect();

    Ok(Session {
        model,
        shards,
        tokenizer,
        caches,
        sampling,
        rng,
        stop_ids,
        max_tokens: args.max_tokens,
        model_dir: args.model_dir.clone(),
        usage_cache_enabled,
    })
}

/// GLM-5.2's official chat template (no newline after role tags), reverse-engineered from
/// colibrì's own C source (`glm.c`) since the downloaded checkpoint ships no
/// `chat_template.jinja` — that C implementation is the only validated source of truth for
/// this template, already proven against this exact checkpoint. `[gMASK]<sop>` opens the
/// FIRST turn only; every turn is `<|user|>{msg}<|assistant|>{think_tag}`, where
/// `<think></think>` disables GLM-5.2's reasoning block (the model babbles and never emits a
/// stop token with the wrong tag here — colibrì's own comment flags this exact failure mode)
/// and bare `<think>` leaves it open for the model to reason before answering.
///
/// `system`, if given, renders as `<|system|>{content}` immediately before the user turn —
/// **unvalidated**: neither colibrì's own code nor this project's testing has exercised the
/// `<|system|>` role against the real checkpoint, despite the tokenizer vocabulary defining
/// it. It's a reasonable bet (every GLM/ChatGLM-lineage template convention places system
/// content there), but carries the same "wrong template detail -> model never stops" risk
/// `think_tag` above already has a proven history of — validate any code path using this with
/// a real request against the real checkpoint before trusting it in production.
pub fn render_turn(user_msg: &str, first: bool, think: bool, system: Option<&str>) -> String {
    let think_tag = if think { "<think>" } else { "<think></think>" };
    let mut out = String::new();
    if first {
        out.push_str("[gMASK]<sop>");
        if let Some(sys) = system {
            out.push_str("<|system|>");
            out.push_str(sys);
        }
    }
    out.push_str("<|user|>");
    out.push_str(user_msg);
    out.push_str("<|assistant|>");
    out.push_str(think_tag);
    out
}

pub enum Role {
    System,
    User,
    Assistant,
}

/// Renders a FULL conversation (as an OpenAI-style stateless `messages` array arrives) into
/// one prompt string — the HTTP server's counterpart to `render_turn`'s incremental,
/// KV-continuation-based rendering: the server never keeps a `KvState` across requests (see
/// the Fase 11 plan's "stateless" design decision), so every request re-renders the WHOLE
/// history from scratch. Every message except a trailing user turn is rendered complete
/// (`<|user|>{msg}`/`<|assistant|>{msg}`, no think tag — already-said text needs none); if the
/// conversation doesn't already end on an assistant turn, an open `<|assistant|>{think_tag}` is
/// appended for the model to fill in. Same unvalidated-`<|system|>` caveat as `render_turn`.
pub fn render_messages(messages: &[(Role, String)], think: bool) -> String {
    let think_tag = if think { "<think>" } else { "<think></think>" };
    let mut out = String::from("[gMASK]<sop>");
    for (role, content) in messages {
        let tag = match role {
            Role::System => "<|system|>",
            Role::User => "<|user|>",
            Role::Assistant => "<|assistant|>",
        };
        out.push_str(tag);
        out.push_str(content);
    }
    if !matches!(messages.last(), Some((Role::Assistant, _))) {
        out.push_str("<|assistant|>");
        out.push_str(think_tag);
    }
    out
}

/// One event `generate_reply` reports as it works — enough for a caller to either print rich
/// progress (the CLI's stderr output) or build streaming output (the server's SSE chunks)
/// without `generate_reply` itself needing to know which.
pub enum GenEvent<'a> {
    /// The prefill step (forwarding `turn_ids`) completed.
    Prefill { tokens: usize, seconds: f32, hits: u64, misses: u64, io_seconds: f32 },
    /// One new token was generated (decode step `index` of at most `max`).
    Token { token_id: usize, bytes: &'a [u8], index: usize, max: usize, seconds: f32, hits: u64, misses: u64, io_seconds: f32 },
}

/// Forwards `turn_ids` (prefill, continuing from `pos_base` positions already in `kv`) then
/// greedy/samples new tokens until a stop id or `sess.max_tokens`, reporting a [`GenEvent`] per
/// step — a real-model forward can take seconds to tens of seconds per step, so a caller that
/// discards these entirely (silent output) would be indistinguishable from a hang; the CLI's
/// `--chat`/`--prompt` modes print them to stderr, the HTTP server's streaming mode turns
/// `Token` events into SSE chunks. Returns the full decoded reply text, the new total position
/// (`pos_base + turn_ids.len() + generated.len()`, for the caller to pass back in as the next
/// turn's `pos_base`), and the count of tokens generated.
pub fn generate_reply(
    sess: &mut Session,
    kv: &mut KvState,
    turn_ids: &[usize],
    pos_base: usize,
    mut on_event: impl FnMut(GenEvent),
) -> Result<(String, usize, usize), Box<dyn std::error::Error>> {
    let mut step_t = std::time::Instant::now();
    let mut logits = generate::step(&sess.model, &sess.shards, &mut sess.caches, kv, turn_ids, pos_base)?;
    let (mut hits, mut misses, mut io_ns) = sess.caches.hit_miss_totals();
    on_event(GenEvent::Prefill { tokens: turn_ids.len(), seconds: step_t.elapsed().as_secs_f32(), hits, misses, io_seconds: io_ns as f32 / 1e9 });
    let mut pos = pos_base + turn_ids.len();

    let mut out_ids = Vec::with_capacity(sess.max_tokens);
    while out_ids.len() < sess.max_tokens {
        let next = generate::pick_token(&logits, &sess.sampling, &mut sess.rng, None);
        if sess.stop_ids.contains(&next) {
            break;
        }
        out_ids.push(next);
        if out_ids.len() >= sess.max_tokens {
            let decoded = sess.tokenizer.decode(&[next as i32]);
            on_event(GenEvent::Token { token_id: next, bytes: &decoded, index: out_ids.len(), max: sess.max_tokens, seconds: 0.0, hits, misses, io_seconds: 0.0 });
            break;
        }
        let io_ns_before = io_ns;
        step_t = std::time::Instant::now();
        logits = generate::step(&sess.model, &sess.shards, &mut sess.caches, kv, &[next], pos)?;
        let step_seconds = step_t.elapsed().as_secs_f32();
        (hits, misses, io_ns) = sess.caches.hit_miss_totals();
        let decoded = sess.tokenizer.decode(&[next as i32]);
        on_event(GenEvent::Token {
            token_id: next,
            bytes: &decoded,
            index: out_ids.len(),
            max: sess.max_tokens,
            seconds: step_seconds,
            hits,
            misses,
            io_seconds: (io_ns - io_ns_before) as f32 / 1e9,
        });
        pos += 1;
    }

    let out_i32: Vec<i32> = out_ids.iter().map(|&id| id as i32).collect();
    let text = String::from_utf8_lossy(&sess.tokenizer.decode(&out_i32)).into_owned();

    Ok((text, pos, out_ids.len()))
}
