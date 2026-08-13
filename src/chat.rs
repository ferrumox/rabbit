//! Shared session/generation/template logic for both the CLI's `--chat` mode and the HTTP
//! server (`server.rs`) — extracted from `main.rs` so neither has to duplicate it.

use crate::generate::{self, Rng, SamplingConfig, StepProfile};
use crate::model::{self, ExpertCaches, Model, Tokenizer};
pub use crate::model::KvState;
use crate::safetensors::Shards;
use std::collections::VecDeque;
use std::path::PathBuf;

/// How many recent chat-completion turns' [`TurnProfile`]s `Session::profile` keeps — old
/// entries are dropped as new ones arrive (see `server.rs`'s `/profile` handler). Matches
/// colibrì's own rolling-window size for the equivalent feature.
pub const PROFILE_TURNS: usize = 120;

pub struct LoadArgs {
    pub model_dir: PathBuf,
    pub max_tokens: usize,
    pub temperature: f32,
    pub nucleus: f32,
    pub seed: u64,
    pub dbits: u8,
    pub ebits: u8,
    /// `None` = pick a safe default once the model's real architecture/size is known
    /// (`model::safe_default_expert_cache_capacity`) — `64` for every checkpoint except a real
    /// K3 one, where that flat default risks OOM at K3's real per-expert scale (a real crash
    /// found this, see `expert_cache::safe_mxfp4_capacity`'s doc). `Some(n)` (`--expert-cache n`)
    /// always wins over this, whatever the risk — never silently overridden.
    pub cache_capacity: Option<usize>,
    /// `None` = match the resolved `cache_capacity` (today's original coupled behavior — one
    /// `io_uring` round submits exactly `capacity`-worth of reads). `Some(n)` (`--io-batch-size
    /// n`) decouples them: `moe.rs` submits `n` experts' reads per round regardless of how many
    /// stay resident afterward. Added 2026-07-29 after finding a small, memory-safe `capacity`
    /// (see `cache_capacity`'s own doc) also throttles per-round read concurrency — and this
    /// project's own `PERFORMANCE.md` ("Lead 2") already found MORE concurrent scattered reads
    /// measurably faster on this drive, not fewer. See
    /// `expert_cache::ExpertCache::io_batch_size`'s doc for the full reasoning and why this is
    /// safe to decouple from `capacity` at all.
    pub io_batch_size: Option<usize>,
    pub no_usage_cache: bool,
    /// opt-in CACHE_ROUTE (see `moe::RouteConfig`) — off by default, matching colibrì's own
    /// stance, since it's still unmeasured on rabbit's own architecture.
    pub cache_route: bool,
    /// Extra directories to scan for `.safetensors` shards, alongside `model_dir` — lets a
    /// checkpoint's shards be split across separate drives (e.g. a second NVMe added for
    /// capacity/bandwidth, without an OS-level RAID array — see `Shards::open_multi`'s doc).
    /// `config.json`/tokenizer files still come from `model_dir` alone. Empty by default (every
    /// shard in `model_dir`, matching every version of rabbit before this option existed).
    pub shard_dirs: Vec<PathBuf>,
    /// Opt-in `--mmap-experts` experiment (default `false`) — routes the ordinary LRU expert
    /// cache's miss path through an `mmap`-backed load instead of the default owned-buffer
    /// `pread`/`io_uring` path, for MXFP4 checkpoints only (see
    /// `expert_cache::ExpertCache::begin_loading`'s mmap branch and `PERFORMANCE_KIMI_K3.md` for
    /// the double-caching hypothesis this tests). Never affects the pinned tier or the auto
    /// `--expert-cache` clamp math (see `expert_cache::safe_mxfp4_capacity`'s doc for why that
    /// stays a safe, if conservative, upper bound under this flag).
    pub mmap_experts: bool,
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
    /// Rolling window of the last `PROFILE_TURNS` chat-completion turns' phase timings — fed by
    /// `server.rs`'s HTTP handlers (not by `generate_reply` itself, which just computes and
    /// returns each turn's [`TurnProfile`]) and served read-only at `GET /profile`. Empty when
    /// running via `--chat`/`--prompt` — nothing ever pushes into it outside `server.rs`.
    pub profile: VecDeque<TurnProfile>,
    pub profile_seq: u64,
}

pub fn load_session(args: &LoadArgs) -> Result<Session, Box<dyn std::error::Error>> {
    let mut shard_dirs = Vec::with_capacity(1 + args.shard_dirs.len());
    shard_dirs.push(args.model_dir.clone());
    shard_dirs.extend(args.shard_dirs.iter().cloned());

    eprintln!("loading model (dbits={}, ebits={})...", args.dbits, args.ebits);
    let t0 = std::time::Instant::now();
    let mut model = Model::load_multi(&shard_dirs, args.dbits, args.ebits)?;
    model.set_cache_route(args.cache_route);
    eprintln!("model loaded in {:.1}s ({} layers), cache_route={}", t0.elapsed().as_secs_f32(), model.n_layers(), args.cache_route);

    eprintln!("loading tokenizer...");
    let tokenizer = Tokenizer::load(&args.model_dir, &model)?;

    let shards = Shards::open_multi(&shard_dirs)?;
    let cache_capacity = args.cache_capacity.unwrap_or_else(|| model::safe_default_expert_cache_capacity(&model));
    let io_batch_size = args.io_batch_size.unwrap_or(cache_capacity);
    let mut caches = ExpertCaches::new_with_io_batch_mmap(&model, cache_capacity, io_batch_size, args.mmap_experts);
    let usage_cache_enabled = !args.no_usage_cache;
    if usage_cache_enabled {
        let stats = caches.warm_start(&args.model_dir, cache_capacity);
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
    let stop_ids = model.stop_ids();

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
        profile: VecDeque::new(),
        profile_seq: 0,
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
fn glm52_render_turn(user_msg: &str, first: bool, think: bool, system: Option<&str>) -> String {
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

/// Renders one incremental chat turn (`--chat`'s KV-continuation mode: only the NEW turn needs
/// rendering, prior turns already live in `kv`) into the exact prompt text to encode and
/// forward — dispatches on `model`'s family, since GLM-5.2, Kimi Linear, Kimi K3 and Qwen 3.8 all use
/// genuinely different chat templates (see `glm52_render_turn`'s doc,
/// `kimi_linear::chat_template::render_turn`'s doc, and `kimi_k3::chat_template::render_turn`'s
/// doc for each one's specifics, all read from real reference sources, not guessed).
pub fn render_turn(model: &Model, user_msg: &str, first: bool, think: bool, system: Option<&str>) -> String {
    match model {
        Model::Glm52(_) => glm52_render_turn(user_msg, first, think, system),
        Model::KimiLinear(_) => crate::kimi_linear::chat_template::render_turn(user_msg, first, think, system),
        Model::KimiK3(_) => crate::kimi_k3::chat_template::render_turn(user_msg, first, think, system),
        Model::Qwen38(_) => crate::qwen38::chat_template::render_turn(user_msg, first, think, system),
    }
}

/// Renders a FULL conversation (as an OpenAI-style stateless `messages` array arrives) into
/// one prompt string — the HTTP server's counterpart to `render_turn`'s incremental,
/// KV-continuation-based rendering: the server never keeps a `KvState` across requests (see
/// the Fase 11 plan's "stateless" design decision), so every request re-renders the WHOLE
/// history from scratch. Same per-family dispatch as `render_turn`.
fn glm52_render_messages(messages: &[(Role, String)], think: bool) -> String {
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

pub fn render_messages(model: &Model, messages: &[(Role, String)], think: bool) -> String {
    match model {
        Model::Glm52(_) => glm52_render_messages(messages, think),
        Model::KimiLinear(_) => crate::kimi_linear::chat_template::render_messages(messages, think),
        Model::KimiK3(_) => crate::kimi_k3::chat_template::render_messages(messages, think),
        Model::Qwen38(_) => crate::qwen38::chat_template::render_messages(messages, think),
    }
}

/// A whole chat-completion turn's (prefill + every decode step) phase-timing totals — the
/// `/profile` dashboard's per-entry shape. `hits`/`misses`/`prompt_tokens`/`completion_tokens`
/// mirror the same counters `GenEvent` already reports per step, just summed over the turn;
/// `attention_s`/`expert_wait_s`/`expert_matmul_s`/`lm_head_s` come from `generate::StepProfile`
/// (see its doc for what each phase measures). `forwards` counts real forward passes only — the
/// synthetic zero-cost final `Token` event `generate_reply` emits when `max_tokens` is hit
/// exactly on selection (no further step needed) does not call `step_profiled`, so it doesn't
/// bump this.
#[derive(serde::Serialize, Clone, Default)]
pub struct TurnProfile {
    pub wall_s: f32,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub hits: u64,
    pub misses: u64,
    pub attention_s: f32,
    pub expert_wait_s: f32,
    pub expert_matmul_s: f32,
    pub lm_head_s: f32,
    pub forwards: u64,
}

impl TurnProfile {
    fn accumulate(&mut self, step: &StepProfile) {
        self.attention_s += step.phases.attention_s;
        self.expert_wait_s += step.phases.expert_wait_s;
        self.expert_matmul_s += step.phases.expert_matmul_s;
        self.lm_head_s += step.lm_head_s;
        self.forwards += 1;
    }
}

/// One event `generate_reply` reports as it works — enough for a caller to either print rich
/// progress (the CLI's stderr output) or build streaming output (the server's SSE chunks)
/// without `generate_reply` itself needing to know which.
pub enum GenEvent<'a> {
    /// The prefill step (forwarding `turn_ids`) completed. `io_wait_seconds` is the portion of
    /// `io_seconds` that was pure disk wait (`io_uring`'s `submit_and_wait`) rather than the
    /// decode/copy work that follows it — see `ExpertCache::io_wait_nanos`'s doc.
    Prefill { tokens: usize, seconds: f32, hits: u64, misses: u64, io_seconds: f32, io_wait_seconds: f32 },
    /// One new token was generated (decode step `index` of at most `max`).
    Token {
        token_id: usize,
        bytes: &'a [u8],
        index: usize,
        max: usize,
        seconds: f32,
        hits: u64,
        misses: u64,
        io_seconds: f32,
        io_wait_seconds: f32,
    },
}

/// Forwards `turn_ids` (prefill, continuing from `pos_base` positions already in `kv`) then
/// greedy/samples new tokens until a stop id or `sess.max_tokens`, reporting a [`GenEvent`] per
/// step — a real-model forward can take seconds to tens of seconds per step, so a caller that
/// discards these entirely (silent output) would be indistinguishable from a hang; the CLI's
/// `--chat`/`--prompt` modes print them to stderr, the HTTP server's streaming mode turns
/// `Token` events into SSE chunks. Returns the full decoded reply text, the new total position
/// (`pos_base + turn_ids.len() + generated.len()`, for the caller to pass back in as the next
/// turn's `pos_base`), the count of tokens generated, and this whole turn's [`TurnProfile`]
/// (every caller gets one — the timing overhead is a handful of `Instant::now()` calls per
/// layer, negligible next to a real forward pass; only `server.rs` does anything with it).
pub fn generate_reply(
    sess: &mut Session,
    kv: &mut KvState,
    turn_ids: &[usize],
    pos_base: usize,
    mut on_event: impl FnMut(GenEvent),
) -> Result<(String, usize, usize, TurnProfile), Box<dyn std::error::Error>> {
    let turn_t = std::time::Instant::now();
    let mut profile = TurnProfile { prompt_tokens: turn_ids.len(), ..TurnProfile::default() };

    let mut step_t = std::time::Instant::now();
    let (mut logits, step_profile) = model::step_profiled(&sess.model, &sess.shards, &mut sess.caches, kv, turn_ids, pos_base)?;
    profile.accumulate(&step_profile);
    let (mut hits, mut misses, mut io_ns) = sess.caches.hit_miss_totals();
    let mut io_wait_ns = sess.caches.io_wait_nanos_total();
    on_event(GenEvent::Prefill {
        tokens: turn_ids.len(),
        seconds: step_t.elapsed().as_secs_f32(),
        hits,
        misses,
        io_seconds: io_ns as f32 / 1e9,
        io_wait_seconds: io_wait_ns as f32 / 1e9,
    });
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
            on_event(GenEvent::Token {
                token_id: next,
                bytes: &decoded,
                index: out_ids.len(),
                max: sess.max_tokens,
                seconds: 0.0,
                hits,
                misses,
                io_seconds: 0.0,
                io_wait_seconds: 0.0,
            });
            break;
        }
        let io_ns_before = io_ns;
        let io_wait_ns_before = io_wait_ns;
        step_t = std::time::Instant::now();
        let (next_logits, step_profile) = model::step_profiled(&sess.model, &sess.shards, &mut sess.caches, kv, &[next], pos)?;
        logits = next_logits;
        profile.accumulate(&step_profile);
        let step_seconds = step_t.elapsed().as_secs_f32();
        (hits, misses, io_ns) = sess.caches.hit_miss_totals();
        io_wait_ns = sess.caches.io_wait_nanos_total();
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
            io_wait_seconds: (io_wait_ns - io_wait_ns_before) as f32 / 1e9,
        });
        pos += 1;
    }

    let out_i32: Vec<i32> = out_ids.iter().map(|&id| id as i32).collect();
    let text = String::from_utf8_lossy(&sess.tokenizer.decode(&out_i32)).into_owned();

    profile.wall_s = turn_t.elapsed().as_secs_f32();
    profile.completion_tokens = out_ids.len();
    profile.hits = hits;
    profile.misses = misses;

    Ok((text, pos, out_ids.len(), profile))
}
