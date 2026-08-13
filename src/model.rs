//! Family dispatch: wraps GLM-5.2's and Kimi Linear's `Model`/`KvState`/`ExpertCaches` behind
//! one enum each, so a caller (an HTTP server, a CLI) can load and drive either architecture
//! without knowing in advance which one a checkpoint directory holds.
//!
//! This is the "one new dispatch point" `rabbit-plan.md`'s Phase 1 deferred until a second real
//! architecture existed to design it against (`generate.rs::layer_forward`'s own internals stay
//! untouched — GLM-5.2's whole pipeline is unchanged, still reachable directly via
//! `crate::generate`/`crate::glm52` for anyone who doesn't need multi-architecture dispatch).
//! An enum, not a trait object, for the same reason `rabbit-plan.md`'s Phase 1 notes picked an
//! enum for the architecture split in general: a small, compile-time-known set of families,
//! each with a structurally different `KvState` shape (`glm52::generate::KvState`'s per-layer
//! growing `KvCache`/`DsaCache` vs. `kimi_linear::generate::KvState`'s per-layer fixed-size
//! `KdaLayerState`/`KvCache` mix) that a shared trait would only paper over.

use crate::kimi_linear::config::ConfigError as KimiConfigError;
use crate::kimi_linear::model::ModelError as KimiModelError;
use serde_json::Value;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

pub enum Model {
    Glm52(crate::glm52::model::Model),
    KimiLinear(crate::kimi_linear::model::Model),
    KimiK3(crate::kimi_k3::model::Model),
    Qwen38(crate::qwen38::model::Model),
}

pub enum KvState {
    Glm52(crate::generate::KvState),
    KimiLinear(crate::kimi_linear::generate::KvState),
    KimiK3(crate::kimi_k3::generate::KvState),
    Qwen38(crate::qwen38::generate::KvState),
}

pub enum ExpertCaches {
    Glm52(crate::generate::ExpertCaches),
    KimiLinear(crate::kimi_linear::generate::ExpertCaches),
    KimiK3(crate::kimi_k3::generate::ExpertCaches),
    Qwen38(crate::qwen38::generate::ExpertCaches),
}

#[derive(Debug)]
pub enum ModelError {
    Glm52(crate::glm52::model::ModelError),
    KimiLinear(KimiModelError),
    KimiK3(crate::kimi_k3::model::ModelError),
    Qwen38(crate::qwen38::model::ModelError),
    Io(std::io::Error),
    Json(serde_json::Error),
    /// `config.json`'s `model_type` was missing or matched none of `glm52`'s/`kimi_linear`'s/
    /// `kimi_k3`'s loaders.
    UnknownArchitecture { found: Option<String> },
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::Glm52(e) => write!(f, "{e}"),
            ModelError::KimiLinear(e) => write!(f, "{e}"),
            ModelError::KimiK3(e) => write!(f, "{e}"),
            ModelError::Qwen38(e) => write!(f, "{e}"),
            ModelError::Io(e) => write!(f, "config.json: {e}"),
            ModelError::Json(e) => write!(f, "config.json: {e}"),
            ModelError::UnknownArchitecture { found: Some(m) } => {
                write!(f, "config.json's model_type is {m:?}, which no architecture in rabbit recognizes")
            }
            ModelError::UnknownArchitecture { found: None } => write!(f, "config.json has no model_type field"),
        }
    }
}

impl std::error::Error for ModelError {}

impl From<crate::glm52::model::ModelError> for ModelError {
    fn from(e: crate::glm52::model::ModelError) -> Self {
        ModelError::Glm52(e)
    }
}

impl From<KimiModelError> for ModelError {
    fn from(e: KimiModelError) -> Self {
        ModelError::KimiLinear(e)
    }
}

impl From<crate::kimi_k3::model::ModelError> for ModelError {
    fn from(e: crate::kimi_k3::model::ModelError) -> Self {
        ModelError::KimiK3(e)
    }
}

impl From<crate::qwen38::model::ModelError> for ModelError {
    fn from(e: crate::qwen38::model::ModelError) -> Self {
        ModelError::Qwen38(e)
    }
}

/// `kimi_linear::config::Cfg::load`'s own `UnsupportedArchitecture` (wrong/missing
/// `model_type`) collapses into `ModelError::UnknownArchitecture` here — `read_model_type`
/// below already excludes that case before either family's loader ever runs, so this only
/// fires from a race (the file changing between the peek and the real load) or a
/// `kimi_linear`-specific error that happens to be `UnsupportedArchitecture` for another
/// reason; treating it the same way is the honest answer either way.
impl From<KimiConfigError> for ModelError {
    fn from(e: KimiConfigError) -> Self {
        ModelError::KimiLinear(KimiModelError::Config(e))
    }
}

/// Peeks `config.json`'s `model_type` WITHOUT going through either architecture's own `Cfg`
/// loader (which would reject an unrecognized `model_type` as an error before this dispatch
/// point ever got to try the other family) — same raw-JSON-first approach both `Cfg::load`s
/// already use internally, just stopping after the one field this needs.
fn read_model_type(snap_dir: &Path) -> Result<Option<String>, ModelError> {
    let text = fs::read_to_string(snap_dir.join("config.json")).map_err(ModelError::Io)?;
    let r: Value = serde_json::from_str(&text).map_err(ModelError::Json)?;
    Ok(r.get("model_type").and_then(Value::as_str).map(str::to_string))
}

impl Model {
    pub fn load(snap_dir: &Path, dbits: u8, ebits: u8) -> Result<Model, ModelError> {
        Self::load_multi(std::slice::from_ref(&snap_dir.to_path_buf()), dbits, ebits)
    }

    /// Same as `load`, but reads the checkpoint's `.safetensors` shards from MULTIPLE
    /// directories (`dirs[0]` is still the primary directory — the only one `config.json` is
    /// read from, both here to detect the architecture and inside each family's own loader).
    /// Lets a checkpoint be split across separate drives — see `Shards::open_multi`'s doc.
    pub fn load_multi(dirs: &[PathBuf], dbits: u8, ebits: u8) -> Result<Model, ModelError> {
        match read_model_type(&dirs[0])?.as_deref() {
            Some("glm_moe_dsa") => Ok(Model::Glm52(crate::glm52::model::Model::load_multi(dirs, dbits, ebits)?)),
            Some("kimi_linear") => Ok(Model::KimiLinear(crate::kimi_linear::model::Model::load_multi(dirs, dbits, ebits)?)),
            Some("kimi_k3") => Ok(Model::KimiK3(crate::kimi_k3::model::Model::load_multi(dirs, dbits, ebits)?)),
            // Both spellings: `qwen3_5_moe_text` is Qwen3.8-Max's own flat, text-only form and
            // `qwen3_5_moe` the multimodal wrapper whose text fields nest under `text_config`
            // (Qwen3.6-35B-A3B and friends) -- `qwen38::config::Cfg::from_root` reads either.
            Some("qwen3_5_moe_text") | Some("qwen3_5_moe") => Ok(Model::Qwen38(crate::qwen38::model::Model::load_multi(dirs, dbits, ebits)?)),
            found => Err(ModelError::UnknownArchitecture { found: found.map(String::from) }),
        }
    }

    /// Number of transformer layers — for a caller (e.g. `chat.rs`'s startup log) that wants a
    /// quick summary without matching on the family itself.
    pub fn n_layers(&self) -> usize {
        match self {
            Model::Glm52(m) => m.layers.len(),
            Model::KimiLinear(m) => m.layers.len(),
            Model::KimiK3(m) => m.layers.len(),
            Model::Qwen38(m) => m.layers.len(),
        }
    }

    /// Configured stop/eos token ids, as `usize` — every family's `Cfg::stop_ids` is a `Vec<i32>`
    /// read the same way (`eos_token_id`, single int or array) from `config.json` (K3's own
    /// `Cfg` reads it via `cfg.base`, the embedded `kimi_linear::config::Cfg`).
    pub fn stop_ids(&self) -> Vec<usize> {
        match self {
            Model::Glm52(m) => m.cfg.stop_ids.iter().map(|&id| id as usize).collect(),
            Model::KimiLinear(m) => m.cfg.stop_ids.iter().map(|&id| id as usize).collect(),
            Model::KimiK3(m) => m.cfg.base.stop_ids.iter().map(|&id| id as usize).collect(),
            // Qwen's list is the UNION of config.json's and generation_config.json's -- see
            // `qwen38::config`'s module doc: `<|im_end|>` only appears in the latter.
            Model::Qwen38(m) => m.cfg.stop_ids.iter().map(|&id| id as usize).collect(),
        }
    }

    /// Sets the opt-in CACHE_ROUTE toggle (see `glm52::moe::RouteConfig`) — every family reuses
    /// the exact same `RouteConfig`/`moe()` routing code (Kimi's `Ffn::Moe` layers dispatch
    /// through `glm52::moe::moe()` too, see `kimi_linear::generate`'s doc), so this one setter
    /// covers all three.
    pub fn set_cache_route(&mut self, on: bool) {
        match self {
            Model::Glm52(m) => m.route_cfg.cache_route = on,
            Model::KimiLinear(m) => m.route_cfg.cache_route = on,
            Model::KimiK3(m) => m.route_cfg.cache_route = on,
            // Qwen 3.8 routes through `qwen38::moe::route` (plain softmax top-k), not
            // `glm52::moe::route_cache_aware`, so it has no CACHE_ROUTE knob to set: the flag is
            // accepted and ignored for this family rather than silently pretending to apply.
            Model::Qwen38(_) => {}
        }
    }
}

/// `--chat --session <path>` persistence error — dispatches to each family's own `LoadError`
/// (`kv_session.rs` for GLM-5.2, `kimi_linear`/`kimi_k3`/`qwen38::kv_session` for the others: a
/// genuinely different on-disk format per family — see those modules' docs for why).
#[derive(Debug)]
pub enum KvSessionError {
    Glm52Load(crate::kv_session::LoadError),
    KimiLinearLoad(crate::kimi_linear::kv_session::LoadError),
    KimiK3Load(crate::kimi_k3::kv_session::LoadError),
    Qwen38Load(crate::qwen38::kv_session::LoadError),
    Io(std::io::Error),
}

impl fmt::Display for KvSessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KvSessionError::Glm52Load(e) => write!(f, "{e}"),
            KvSessionError::KimiLinearLoad(e) => write!(f, "{e}"),
            KvSessionError::KimiK3Load(e) => write!(f, "{e}"),
            KvSessionError::Qwen38Load(e) => write!(f, "{e}"),
            KvSessionError::Io(e) => write!(f, "session file: {e}"),
        }
    }
}

impl std::error::Error for KvSessionError {}

impl KvState {
    pub fn new(model: &Model) -> KvState {
        match model {
            Model::Glm52(m) => KvState::Glm52(crate::generate::KvState::new(m)),
            Model::KimiLinear(m) => KvState::KimiLinear(crate::kimi_linear::generate::KvState::new(m)),
            Model::KimiK3(m) => KvState::KimiK3(crate::kimi_k3::generate::KvState::new(m)),
            Model::Qwen38(m) => KvState::Qwen38(crate::qwen38::generate::KvState::new(m)),
        }
    }

    /// Appends one completed turn's new KV rows to `path` — see `kv_session.rs`'s doc for
    /// GLM-5.2, `kimi_linear::kv_session.rs`'s doc for Kimi Linear, `kimi_k3::kv_session.rs`'s
    /// doc for K3 (same two-file MLA-log + KDA-snapshot format, distinct magic bytes, no
    /// Attention-Residuals state to persist since that's purely transient per forward call).
    pub fn save(&self, from_pos: usize, to_pos: usize, path: &Path) -> Result<(), KvSessionError> {
        match self {
            KvState::Glm52(kv) => kv.save(from_pos, to_pos, path).map_err(KvSessionError::Io),
            KvState::KimiLinear(kv) => kv.save(from_pos, to_pos, path).map_err(KvSessionError::Io),
            KvState::KimiK3(kv) => kv.save(from_pos, to_pos, path).map_err(KvSessionError::Io),
            KvState::Qwen38(kv) => kv.save(from_pos, to_pos, path).map_err(KvSessionError::Io),
        }
    }

    /// Loads a previously-saved session file — see `kv_session.rs`'s doc for GLM-5.2,
    /// `kimi_linear::kv_session.rs`'s doc for Kimi Linear, `kimi_k3::kv_session.rs`'s doc for K3.
    pub fn load(path: &Path, model: &Model) -> Result<(KvState, usize), KvSessionError> {
        match model {
            Model::Glm52(m) => {
                let (kv, pos) = crate::generate::KvState::load(path, m).map_err(KvSessionError::Glm52Load)?;
                Ok((KvState::Glm52(kv), pos))
            }
            Model::KimiLinear(m) => {
                let (kv, pos) = crate::kimi_linear::generate::KvState::load(path, m).map_err(KvSessionError::KimiLinearLoad)?;
                Ok((KvState::KimiLinear(kv), pos))
            }
            Model::KimiK3(m) => {
                let (kv, pos) = crate::kimi_k3::generate::KvState::load(path, m).map_err(KvSessionError::KimiK3Load)?;
                Ok((KvState::KimiK3(kv), pos))
            }
            Model::Qwen38(m) => {
                let (kv, pos) = crate::qwen38::generate::KvState::load(path, m).map_err(KvSessionError::Qwen38Load)?;
                Ok((KvState::Qwen38(kv), pos))
            }
        }
    }
}

/// `main.rs`'s AUTO `--expert-cache` default (used only when the flag isn't passed explicitly —
/// see `LoadArgs::cache_capacity`'s doc) — `64` everywhere EXCEPT a real K3 checkpoint (native
/// MXFP4 routed experts), where that flat default risks OOM at this checkpoint's real scale (see
/// `expert_cache::safe_mxfp4_capacity`'s doc for the real numbers and the real crash that found
/// this). An EXPLICIT `--expert-cache N` always wins over this — never silently overridden,
/// whatever the risk, matching this whole flag's existing "the user's own choice" contract.
pub fn safe_default_expert_cache_capacity(model: &Model) -> usize {
    const DEFAULT: usize = 64;
    match model {
        // Every Qwen 3.8 layer is a routed-MoE layer (no dense prefix), and the real checkpoint's
        // experts are MXFP4 at 8192x2048 -- 28 MB per expert, so the flat 64 default would ask for
        // far more RAM than this machine has. Same clamp K3's own crash produced.
        Model::Qwen38(m) if m.cfg.mxfp4_experts => crate::expert_cache::safe_mxfp4_capacity(
            DEFAULT,
            m.layers.len(),
            m.cfg.moe_inter as usize,
            m.cfg.hidden as usize,
        ),
        Model::KimiK3(m) if m.cfg.mxfp4_experts => {
            let n_moe_layers = m.layers.iter().filter(|l| matches!(l.ffn, crate::kimi_k3::model::Ffn::Moe(_))).count();
            crate::expert_cache::safe_mxfp4_capacity(DEFAULT, n_moe_layers, m.cfg.base.moe_inter as usize, m.cfg.base.hidden as usize)
        }
        _ => DEFAULT,
    }
}

impl ExpertCaches {
    pub fn new(model: &Model, capacity: usize) -> ExpertCaches {
        Self::new_with_io_batch(model, capacity, capacity)
    }

    /// Like `new`, but `io_batch_size` (how many experts' reads `moe.rs` submits per `io_uring`
    /// round — see `expert_cache::ExpertCache::io_batch_size`'s doc) is set independently of
    /// `capacity` (resident memory) instead of defaulting to match it. Wired up by `chat.rs`'s
    /// `--io-batch-size` flag.
    pub fn new_with_io_batch(model: &Model, capacity: usize, io_batch_size: usize) -> ExpertCaches {
        match model {
            Model::Glm52(m) => ExpertCaches::Glm52(crate::generate::ExpertCaches::new_with_io_batch(m, capacity, io_batch_size)),
            Model::KimiLinear(m) => ExpertCaches::KimiLinear(crate::kimi_linear::generate::ExpertCaches::new_with_io_batch(m, capacity, io_batch_size)),
            Model::KimiK3(m) => ExpertCaches::KimiK3(crate::kimi_k3::generate::ExpertCaches::new_with_io_batch(m, capacity, io_batch_size)),
            Model::Qwen38(m) => ExpertCaches::Qwen38(crate::qwen38::generate::ExpertCaches::new_with_io_batch(m, capacity, io_batch_size)),
        }
    }

    /// Like `new_with_io_batch`, but also sets the opt-in `--mmap-experts` flag on whichever
    /// family's `ExpertCache`s get built — see each family's own `new_with_io_batch_mmap` doc.
    /// The one call site that matters is `chat.rs::load_session`, which reads the real
    /// `--mmap-experts` CLI flag (`LoadArgs::mmap_experts`) and passes it straight through here.
    pub fn new_with_io_batch_mmap(model: &Model, capacity: usize, io_batch_size: usize, mmap_experts: bool) -> ExpertCaches {
        match model {
            Model::Glm52(m) => ExpertCaches::Glm52(crate::generate::ExpertCaches::new_with_io_batch_mmap(m, capacity, io_batch_size, mmap_experts)),
            Model::KimiLinear(m) => {
                ExpertCaches::KimiLinear(crate::kimi_linear::generate::ExpertCaches::new_with_io_batch_mmap(m, capacity, io_batch_size, mmap_experts))
            }
            Model::KimiK3(m) => ExpertCaches::KimiK3(crate::kimi_k3::generate::ExpertCaches::new_with_io_batch_mmap(m, capacity, io_batch_size, mmap_experts)),
            Model::Qwen38(m) => ExpertCaches::Qwen38(crate::qwen38::generate::ExpertCaches::new_with_io_batch_mmap(m, capacity, io_batch_size, mmap_experts)),
        }
    }

    pub fn hit_miss_totals(&self) -> (u64, u64, u64) {
        match self {
            ExpertCaches::Glm52(c) => c.hit_miss_totals(),
            ExpertCaches::KimiLinear(c) => c.hit_miss_totals(),
            ExpertCaches::KimiK3(c) => c.hit_miss_totals(),
            ExpertCaches::Qwen38(c) => c.hit_miss_totals(),
        }
    }

    pub fn io_wait_nanos_total(&self) -> u64 {
        match self {
            ExpertCaches::Glm52(c) => c.io_wait_nanos_total(),
            ExpertCaches::KimiLinear(c) => c.io_wait_nanos_total(),
            ExpertCaches::KimiK3(c) => c.io_wait_nanos_total(),
            ExpertCaches::Qwen38(c) => c.io_wait_nanos_total(),
        }
    }

    pub fn warm_start(&mut self, model_dir: &Path, cache_capacity: usize) -> crate::generate::WarmStartStats {
        match self {
            ExpertCaches::Glm52(c) => c.warm_start(model_dir, cache_capacity),
            ExpertCaches::KimiLinear(c) => c.warm_start(model_dir, cache_capacity),
            ExpertCaches::KimiK3(c) => c.warm_start(model_dir, cache_capacity),
            ExpertCaches::Qwen38(c) => c.warm_start(model_dir, cache_capacity),
        }
    }

    pub fn save_usage(&self, model_dir: &Path) -> std::io::Result<()> {
        match self {
            ExpertCaches::Glm52(c) => c.save_usage(model_dir),
            ExpertCaches::KimiLinear(c) => c.save_usage(model_dir),
            ExpertCaches::KimiK3(c) => c.save_usage(model_dir),
            ExpertCaches::Qwen38(c) => c.save_usage(model_dir),
        }
    }
}

/// Decode/prefill step returning logits for only the LAST new position — dispatches to
/// `crate::generate::step` or `crate::kimi_linear::generate::step`. Panics (via the `unreachable`
/// arms) only if `model`/`kv`/`caches` come from DIFFERENT `Model::load` calls of different
/// families — `KvState::new`/`ExpertCaches::new` always tag their variant from the same `model`
/// they're built from, so a mismatch here means the caller mixed up two sessions, not a
/// reachable runtime state for any single, correctly-threaded session.
pub fn step(model: &Model, shards: &crate::safetensors::Shards, caches: &mut ExpertCaches, kv: &mut KvState, ids: &[usize], pos_base: usize) -> Result<Vec<f32>, ModelError> {
    match (model, caches, kv) {
        (Model::Glm52(m), ExpertCaches::Glm52(c), KvState::Glm52(k)) => Ok(crate::generate::step(m, shards, c, k, ids, pos_base)?),
        (Model::KimiLinear(m), ExpertCaches::KimiLinear(c), KvState::KimiLinear(k)) => Ok(crate::kimi_linear::generate::step(m, shards, c, k, ids, pos_base)?),
        (Model::KimiK3(m), ExpertCaches::KimiK3(c), KvState::KimiK3(k)) => Ok(crate::kimi_k3::generate::step(m, shards, c, k, ids, pos_base)?),
        (Model::Qwen38(m), ExpertCaches::Qwen38(c), KvState::Qwen38(k)) => Ok(crate::qwen38::generate::step(m, shards, c, k, ids, pos_base)?),
        _ => unreachable!("Model/ExpertCaches/KvState family mismatch -- always construct KvState/ExpertCaches from the same Model"),
    }
}

/// Like `step`, but returns logits for EVERY new position `[S,vocab]` — same dispatch/panic
/// contract as `step`.
pub fn step_all(model: &Model, shards: &crate::safetensors::Shards, caches: &mut ExpertCaches, kv: &mut KvState, ids: &[usize], pos_base: usize) -> Result<Vec<f32>, ModelError> {
    match (model, caches, kv) {
        (Model::Glm52(m), ExpertCaches::Glm52(c), KvState::Glm52(k)) => Ok(crate::generate::step_all(m, shards, c, k, ids, pos_base)?),
        (Model::KimiLinear(m), ExpertCaches::KimiLinear(c), KvState::KimiLinear(k)) => Ok(crate::kimi_linear::generate::step_all(m, shards, c, k, ids, pos_base)?),
        (Model::KimiK3(m), ExpertCaches::KimiK3(c), KvState::KimiK3(k)) => Ok(crate::kimi_k3::generate::step_all(m, shards, c, k, ids, pos_base)?),
        (Model::Qwen38(m), ExpertCaches::Qwen38(c), KvState::Qwen38(k)) => Ok(crate::qwen38::generate::step_all(m, shards, c, k, ids, pos_base)?),
        _ => unreachable!("Model/ExpertCaches/KvState family mismatch -- always construct KvState/ExpertCaches from the same Model"),
    }
}

/// Like `step`, but also returns a [`crate::generate::StepProfile`] — same dispatch/panic
/// contract as `step`. Every family's `step_profiled` shares that one profile type (plain data,
/// no family-specific coupling — see `kimi_linear::generate::step_profiled`'s doc).
pub fn step_profiled(
    model: &Model,
    shards: &crate::safetensors::Shards,
    caches: &mut ExpertCaches,
    kv: &mut KvState,
    ids: &[usize],
    pos_base: usize,
) -> Result<(Vec<f32>, crate::generate::StepProfile), ModelError> {
    match (model, caches, kv) {
        (Model::Glm52(m), ExpertCaches::Glm52(c), KvState::Glm52(k)) => Ok(crate::generate::step_profiled(m, shards, c, k, ids, pos_base)?),
        (Model::KimiLinear(m), ExpertCaches::KimiLinear(c), KvState::KimiLinear(k)) => Ok(crate::kimi_linear::generate::step_profiled(m, shards, c, k, ids, pos_base)?),
        (Model::KimiK3(m), ExpertCaches::KimiK3(c), KvState::KimiK3(k)) => Ok(crate::kimi_k3::generate::step_profiled(m, shards, c, k, ids, pos_base)?),
        (Model::Qwen38(m), ExpertCaches::Qwen38(c), KvState::Qwen38(k)) => Ok(crate::qwen38::generate::step_profiled(m, shards, c, k, ids, pos_base)?),
        _ => unreachable!("Model/ExpertCaches/KvState family mismatch -- always construct KvState/ExpertCaches from the same Model"),
    }
}

pub enum Tokenizer {
    Glm52(Box<crate::tokenizer::Tokenizer>),
    KimiLinear(Box<crate::kimi_linear::tokenizer::Tokenizer>),
    KimiK3(Box<crate::kimi_k3::tokenizer::Tokenizer>),
    /// The SAME `crate::tokenizer::Tokenizer` type GLM-5.2 uses (both are byte-level BPE from a
    /// `tokenizer.json`); a separate variant only because it's loaded through
    /// `qwen38::tokenizer::load`, which verifies the file declares the pre-tokenizer and
    /// `ignore_merges` this port was written against.
    Qwen38(Box<crate::tokenizer::Tokenizer>),
}

#[derive(Debug)]
pub enum TokenizerError {
    Glm52(crate::tokenizer::TokenizerError),
    KimiLinear(crate::kimi_linear::tokenizer::TokenizerError),
    KimiK3(crate::kimi_k3::tokenizer::TokenizerError),
    Qwen38(crate::qwen38::tokenizer::LoadError),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerError::Glm52(e) => write!(f, "{e}"),
            TokenizerError::KimiLinear(e) => write!(f, "{e}"),
            TokenizerError::KimiK3(e) => write!(f, "{e}"),
            TokenizerError::Qwen38(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

impl Tokenizer {
    /// Loads the tokenizer matching `model`'s family: GLM-5.2's `tokenizer.json` (a single
    /// file inside `dir`), or either Kimi family's `tiktoken.model` + `tokenizer_config.json`
    /// (both read directly from `dir` — see `kimi_linear::tokenizer::Tokenizer::load`'s/
    /// `kimi_k3::tokenizer::Tokenizer::load`'s docs; K3's own file FORMAT is identical to Kimi
    /// Linear 48B's, but the reserved-special-token COUNT differs — see `kimi_k3::tokenizer`'s
    /// module doc). Takes `model` rather than re-peeking `config.json`'s `model_type` itself,
    /// since the caller already resolved that via `Model::load` — no reason to read the file
    /// twice.
    pub fn load(dir: &Path, model: &Model) -> Result<Tokenizer, TokenizerError> {
        match model {
            Model::Glm52(_) => Ok(Tokenizer::Glm52(Box::new(
                crate::tokenizer::Tokenizer::load(&dir.join("tokenizer.json")).map_err(TokenizerError::Glm52)?,
            ))),
            Model::KimiLinear(_) => {
                Ok(Tokenizer::KimiLinear(Box::new(crate::kimi_linear::tokenizer::Tokenizer::load(dir).map_err(TokenizerError::KimiLinear)?)))
            }
            Model::KimiK3(_) => Ok(Tokenizer::KimiK3(Box::new(crate::kimi_k3::tokenizer::Tokenizer::load(dir).map_err(TokenizerError::KimiK3)?))),
            Model::Qwen38(_) => Ok(Tokenizer::Qwen38(Box::new(crate::qwen38::tokenizer::load(dir).map_err(TokenizerError::Qwen38)?))),
        }
    }

    pub fn encode(&self, text: &str) -> Vec<i32> {
        match self {
            Tokenizer::Glm52(t) => t.encode(text),
            Tokenizer::KimiLinear(t) => t.encode(text),
            Tokenizer::KimiK3(t) => t.encode(text),
            Tokenizer::Qwen38(t) => t.encode(text),
        }
    }

    pub fn decode(&self, ids: &[i32]) -> Vec<u8> {
        match self {
            Tokenizer::Glm52(t) => t.decode(ids),
            Tokenizer::KimiLinear(t) => t.decode(ids),
            Tokenizer::KimiK3(t) => t.decode(ids),
            Tokenizer::Qwen38(t) => t.decode(ids),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    struct TempDir(std::path::PathBuf);
    impl TempDir {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(name);
            fs::create_dir_all(&dir).unwrap();
            TempDir(dir)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).ok();
        }
    }

    #[test]
    fn load_rejects_an_unrecognized_model_type_without_touching_either_loader() {
        let dir = TempDir::new("rabbit_test_model_dispatch_unknown_arch");
        fs::write(dir.0.join("config.json"), r#"{"model_type": "llama"}"#).unwrap();

        let err = match Model::load(&dir.0, 32, 32) {
            Err(e) => e,
            Ok(_) => panic!("expected an error"),
        };
        assert!(
            matches!(err, ModelError::UnknownArchitecture { found: Some(ref m) } if m == "llama"),
            "expected UnknownArchitecture(\"llama\"), got {err:?}"
        );
    }

    #[test]
    fn load_rejects_a_config_with_no_model_type_at_all() {
        let dir = TempDir::new("rabbit_test_model_dispatch_no_arch");
        fs::write(dir.0.join("config.json"), r#"{"hidden_size": 128}"#).unwrap();

        let err = match Model::load(&dir.0, 32, 32) {
            Err(e) => e,
            Ok(_) => panic!("expected an error"),
        };
        assert!(matches!(err, ModelError::UnknownArchitecture { found: None }), "expected UnknownArchitecture(None), got {err:?}");
    }

    fn xorshift(seed: &mut u32) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 17;
        *seed ^= *seed << 5;
        ((*seed as f32 / u32::MAX as f32) - 0.5) * 2.0
    }

    fn random_vec(n: usize, seed: &mut u32) -> Vec<f32> {
        (0..n).map(|_| xorshift(seed)).collect()
    }

    fn f32_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn write_safetensors(dir: &Path, header: serde_json::Map<String, Value>, data: Vec<u8>) {
        let header_bytes = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_bytes);
        out.extend_from_slice(&data);
        fs::write(dir.join("model.safetensors"), out).unwrap();
    }

    /// A minimal single-layer GLM-5.2 checkpoint (dense FFN, no DSA) -- just enough for
    /// `Model::load` to succeed and `step()` to run end to end.
    fn build_minimal_glm52_fixture(name: &str) -> TempDir {
        let dir = TempDir::new(name);
        let mut seed = 9u32;
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_string(), serde_json::json!({"format": "rabbit-test"}));
        let mut data = Vec::new();
        let mut add = |header: &mut serde_json::Map<String, Value>, name: String, shape: Vec<usize>| {
            let n: usize = shape.iter().product::<usize>().max(1);
            let bytes = f32_bytes(&random_vec(n, &mut seed));
            let start = data.len() as u64;
            data.extend_from_slice(&bytes);
            let end = data.len() as u64;
            header.insert(name, serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, end]}));
        };

        let (d, h, qk_nope, qk_rope, v_head, q_lora, kv_lora, vocab, dense_inter) = (6, 2, 2, 2, 3, 4, 4, 10, 5);
        let qh = qk_nope + qk_rope;
        add(&mut header, "model.embed_tokens.weight".into(), vec![vocab, d]);
        add(&mut header, "lm_head.weight".into(), vec![vocab, d]);
        add(&mut header, "model.norm.weight".into(), vec![d]);
        add(&mut header, "model.layers.0.input_layernorm.weight".into(), vec![d]);
        add(&mut header, "model.layers.0.post_attention_layernorm.weight".into(), vec![d]);
        add(&mut header, "model.layers.0.self_attn.q_a_proj.weight".into(), vec![q_lora, d]);
        add(&mut header, "model.layers.0.self_attn.q_a_layernorm.weight".into(), vec![q_lora]);
        add(&mut header, "model.layers.0.self_attn.q_b_proj.weight".into(), vec![h * qh, q_lora]);
        add(&mut header, "model.layers.0.self_attn.kv_a_proj_with_mqa.weight".into(), vec![kv_lora + qk_rope, d]);
        add(&mut header, "model.layers.0.self_attn.kv_a_layernorm.weight".into(), vec![kv_lora]);
        add(&mut header, "model.layers.0.self_attn.kv_b_proj.weight".into(), vec![h * (qk_nope + v_head), kv_lora]);
        add(&mut header, "model.layers.0.self_attn.o_proj.weight".into(), vec![d, h * v_head]);
        add(&mut header, "model.layers.0.mlp.gate_proj.weight".into(), vec![dense_inter, d]);
        add(&mut header, "model.layers.0.mlp.up_proj.weight".into(), vec![dense_inter, d]);
        add(&mut header, "model.layers.0.mlp.down_proj.weight".into(), vec![d, dense_inter]);

        let cfg_json = serde_json::json!({
            "model_type": "glm_moe_dsa",
            "hidden_size": d, "num_hidden_layers": 1, "num_attention_heads": h,
            "n_routed_experts": 2, "num_experts_per_tok": 1, "moe_intermediate_size": 2,
            "intermediate_size": dense_inter, "first_k_dense_replace": 1, "q_lora_rank": q_lora,
            "kv_lora_rank": kv_lora, "qk_nope_head_dim": qk_nope, "qk_rope_head_dim": qk_rope,
            "v_head_dim": v_head, "n_shared_experts": 1, "vocab_size": vocab, "n_group": 1,
            "topk_group": 1, "index_topk": 0, "index_n_heads": 0, "index_head_dim": 0,
        });
        fs::write(dir.0.join("config.json"), cfg_json.to_string()).unwrap();
        write_safetensors(&dir.0, header, data);
        dir
    }

    /// A minimal single-layer Kimi Linear checkpoint (KDA + dense FFN) -- just enough for
    /// `Model::load` to succeed and `step()` to run end to end.
    fn build_minimal_kimi_linear_fixture(name: &str) -> TempDir {
        let dir = TempDir::new(name);
        let mut seed = 13u32;
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_string(), serde_json::json!({"format": "rabbit-test"}));
        let mut data = Vec::new();
        let mut add = |header: &mut serde_json::Map<String, Value>, name: String, shape: Vec<usize>| {
            let n: usize = shape.iter().product::<usize>().max(1);
            let bytes = f32_bytes(&random_vec(n, &mut seed));
            let start = data.len() as u64;
            data.extend_from_slice(&bytes);
            let end = data.len() as u64;
            header.insert(name, serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, end]}));
        };

        let (d, h, qk_nope, qk_rope, v_head, kv_lora, vocab, dense_inter) = (6, 2, 2, 2, 3, 4, 10, 5);
        let (kda_head_dim, kda_n_heads, kernel) = (4, 2, 3);
        let d_inner = kda_head_dim * kda_n_heads;

        add(&mut header, "model.embed_tokens.weight".into(), vec![vocab, d]);
        add(&mut header, "lm_head.weight".into(), vec![vocab, d]);
        add(&mut header, "model.norm.weight".into(), vec![d]);
        add(&mut header, "model.layers.0.input_layernorm.weight".into(), vec![d]);
        add(&mut header, "model.layers.0.post_attention_layernorm.weight".into(), vec![d]);
        add(&mut header, "model.layers.0.self_attn.q_proj.weight".into(), vec![d_inner, d]);
        add(&mut header, "model.layers.0.self_attn.k_proj.weight".into(), vec![d_inner, d]);
        add(&mut header, "model.layers.0.self_attn.v_proj.weight".into(), vec![d_inner, d]);
        add(&mut header, "model.layers.0.self_attn.q_conv1d.weight".into(), vec![d_inner, 1, kernel]);
        add(&mut header, "model.layers.0.self_attn.k_conv1d.weight".into(), vec![d_inner, 1, kernel]);
        add(&mut header, "model.layers.0.self_attn.v_conv1d.weight".into(), vec![d_inner, 1, kernel]);
        add(&mut header, "model.layers.0.self_attn.f_a_proj.weight".into(), vec![kda_head_dim, d]);
        add(&mut header, "model.layers.0.self_attn.f_b_proj.weight".into(), vec![d_inner, kda_head_dim]);
        add(&mut header, "model.layers.0.self_attn.dt_bias".into(), vec![d_inner]);
        add(&mut header, "model.layers.0.self_attn.A_log".into(), vec![1, 1, kda_n_heads, 1]);
        add(&mut header, "model.layers.0.self_attn.b_proj.weight".into(), vec![kda_n_heads, d]);
        add(&mut header, "model.layers.0.self_attn.g_a_proj.weight".into(), vec![kda_head_dim, d]);
        add(&mut header, "model.layers.0.self_attn.g_b_proj.weight".into(), vec![d_inner, kda_head_dim]);
        add(&mut header, "model.layers.0.self_attn.o_norm.weight".into(), vec![kda_head_dim]);
        add(&mut header, "model.layers.0.self_attn.o_proj.weight".into(), vec![d, d_inner]);
        add(&mut header, "model.layers.0.mlp.gate_proj.weight".into(), vec![dense_inter, d]);
        add(&mut header, "model.layers.0.mlp.up_proj.weight".into(), vec![dense_inter, d]);
        add(&mut header, "model.layers.0.mlp.down_proj.weight".into(), vec![d, dense_inter]);

        let cfg_json = serde_json::json!({
            "model_type": "kimi_linear",
            "hidden_size": d, "num_hidden_layers": 1, "num_attention_heads": h,
            "first_k_dense_replace": 1, "q_lora_rank": null, "kv_lora_rank": kv_lora,
            "qk_nope_head_dim": qk_nope, "qk_rope_head_dim": qk_rope, "v_head_dim": v_head,
            "num_experts": 2, "num_experts_per_token": 1, "num_shared_experts": 0,
            "num_expert_group": 1, "topk_group": 1, "moe_intermediate_size": 2,
            "intermediate_size": dense_inter, "vocab_size": vocab, "moe_renormalize": true,
            "rms_norm_eps": 1e-5, "routed_scaling_factor": 1.0, "mla_use_nope": true,
            "moe_router_activation_func": "sigmoid", "rope_theta": 10000.0,
            "linear_attn_config": {
                "head_dim": kda_head_dim, "num_heads": kda_n_heads, "short_conv_kernel_size": kernel,
                "kda_layers": [1], "full_attn_layers": []
            }
        });
        fs::write(dir.0.join("config.json"), cfg_json.to_string()).unwrap();
        write_safetensors(&dir.0, header, data);
        dir
    }

    /// A minimal single-layer Kimi K3 checkpoint (KDA + dense FFN, no attn-res/latent-MoE/output
    /// gates -- those get their own dedicated coverage in `kimi_k3::model`/`generate`'s own test
    /// modules; this one is only about proving the top-level `Model`/`KvState`/`ExpertCaches`
    /// dispatch enum routes `model_type: "kimi_k3"` correctly end to end).
    fn build_minimal_k3_fixture(name: &str) -> TempDir {
        let dir = TempDir::new(name);
        let mut seed = 17u32;
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_string(), serde_json::json!({"format": "rabbit-test"}));
        let mut data = Vec::new();
        let mut add = |header: &mut serde_json::Map<String, Value>, name: String, shape: Vec<usize>| {
            let n: usize = shape.iter().product::<usize>().max(1);
            let bytes = f32_bytes(&random_vec(n, &mut seed));
            let start = data.len() as u64;
            data.extend_from_slice(&bytes);
            let end = data.len() as u64;
            header.insert(name, serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, end]}));
        };

        let (d, dense_inter, vocab) = (6, 5, 10);
        let (kda_head_dim, kda_n_heads, kernel) = (4, 2, 3);
        let d_inner = kda_head_dim * kda_n_heads;

        add(&mut header, "language_model.model.embed_tokens.weight".into(), vec![vocab, d]);
        add(&mut header, "language_model.lm_head.weight".into(), vec![vocab, d]);
        add(&mut header, "language_model.model.norm.weight".into(), vec![d]);
        add(&mut header, "language_model.model.layers.0.input_layernorm.weight".into(), vec![d]);
        add(&mut header, "language_model.model.layers.0.post_attention_layernorm.weight".into(), vec![d]);
        add(&mut header, "language_model.model.layers.0.self_attn.q_proj.weight".into(), vec![d_inner, d]);
        add(&mut header, "language_model.model.layers.0.self_attn.k_proj.weight".into(), vec![d_inner, d]);
        add(&mut header, "language_model.model.layers.0.self_attn.v_proj.weight".into(), vec![d_inner, d]);
        add(&mut header, "language_model.model.layers.0.self_attn.q_conv1d.weight".into(), vec![d_inner, 1, kernel]);
        add(&mut header, "language_model.model.layers.0.self_attn.k_conv1d.weight".into(), vec![d_inner, 1, kernel]);
        add(&mut header, "language_model.model.layers.0.self_attn.v_conv1d.weight".into(), vec![d_inner, 1, kernel]);
        add(&mut header, "language_model.model.layers.0.self_attn.f_a_proj.weight".into(), vec![kda_head_dim, d]);
        add(&mut header, "language_model.model.layers.0.self_attn.f_b_proj.weight".into(), vec![d_inner, kda_head_dim]);
        add(&mut header, "language_model.model.layers.0.self_attn.dt_bias".into(), vec![d_inner]);
        add(&mut header, "language_model.model.layers.0.self_attn.A_log".into(), vec![1, 1, kda_n_heads, 1]);
        add(&mut header, "language_model.model.layers.0.self_attn.b_proj.weight".into(), vec![kda_n_heads, d]);
        add(&mut header, "language_model.model.layers.0.self_attn.g_a_proj.weight".into(), vec![kda_head_dim, d]);
        add(&mut header, "language_model.model.layers.0.self_attn.g_b_proj.weight".into(), vec![d_inner, kda_head_dim]);
        add(&mut header, "language_model.model.layers.0.self_attn.o_norm.weight".into(), vec![kda_head_dim]);
        add(&mut header, "language_model.model.layers.0.self_attn.o_proj.weight".into(), vec![d, d_inner]);
        add(&mut header, "language_model.model.layers.0.mlp.gate_proj.weight".into(), vec![dense_inter, d]);
        add(&mut header, "language_model.model.layers.0.mlp.up_proj.weight".into(), vec![dense_inter, d]);
        add(&mut header, "language_model.model.layers.0.mlp.down_proj.weight".into(), vec![d, dense_inter]);

        let text_config = serde_json::json!({
            "model_type": "kimi_linear",
            "hidden_size": d, "num_hidden_layers": 1, "num_attention_heads": 2,
            "first_k_dense_replace": 1, "q_lora_rank": null, "kv_lora_rank": 4,
            "qk_nope_head_dim": 2, "qk_rope_head_dim": 2, "v_head_dim": 3,
            "num_experts": 2, "num_experts_per_token": 1, "num_shared_experts": 0,
            "num_expert_group": 1, "topk_group": 1, "moe_intermediate_size": 2,
            "intermediate_size": dense_inter, "vocab_size": vocab, "moe_renormalize": true,
            "rms_norm_eps": 1e-5, "routed_scaling_factor": 1.0, "mla_use_nope": true,
            "moe_router_activation_func": "sigmoid", "rope_theta": 10000.0,
            "linear_attn_config": {
                "head_dim": kda_head_dim, "num_heads": kda_n_heads, "short_conv_kernel_size": kernel,
                "kda_layers": [1], "full_attn_layers": []
            }
        });
        let cfg_json = serde_json::json!({ "model_type": "kimi_k3", "text_config": text_config });
        fs::write(dir.0.join("config.json"), cfg_json.to_string()).unwrap();
        write_safetensors(&dir.0, header, data);
        dir
    }

    /// The Qwen 3.8 arm of every dispatch point at once: `model_type` recognition, `step` through the
    /// shared entry point, `stop_ids` coming from BOTH config files, the MoE-aware `--expert-cache`
    /// clamp, the chat template, and the tokenizer's own validating loader. Reuses
    /// `qwen38::model`'s tiny fixture rather than hand-building a second one.
    #[test]
    fn load_and_step_dispatch_correctly_for_a_qwen38_checkpoint() {
        let fixture = crate::qwen38::model::tests::TempDir::new("rabbit_test_model_dispatch_qwen38");
        crate::qwen38::model::tests::write_tiny_checkpoint(&fixture.0);

        let model = Model::load(&fixture.0, 16, 4).unwrap();
        assert!(matches!(model, Model::Qwen38(_)), "config.json's qwen3_5_moe_text must route here");
        assert_eq!(model.n_layers(), 4);

        let shards = crate::safetensors::Shards::open(&fixture.0).unwrap();
        let mut caches = ExpertCaches::new(&model, 4);
        let mut kv = KvState::new(&model);
        let logits = step(&model, &shards, &mut caches, &mut kv, &[1, 2], 0).unwrap();
        assert_eq!(logits.len(), 10); // vocab
        assert!(logits.iter().all(|x| x.is_finite()));

        // step_all goes through the same dispatch and returns one row per position
        let mut kv2 = KvState::new(&model);
        let all = step_all(&model, &shards, &mut caches, &mut kv2, &[1, 2], 0).unwrap();
        assert_eq!(all.len(), 2 * 10);

        // the fixture's config.json has eos 7 and no generation_config.json
        assert_eq!(model.stop_ids(), vec![7]);

        // ChatML, through the family dispatch
        let rendered = crate::chat::render_turn(&model, "hola", true, true, None);
        assert!(rendered.starts_with("<|im_start|>system\n"), "got {rendered:?}");
        assert!(rendered.ends_with("<|im_start|>assistant\n<think>\n"), "got {rendered:?}");

        // `--session` round-trips through the dispatch too (the format itself is covered in
        // `qwen38::kv_session`'s own tests)
        let session_path = fixture.0.join("session.bin");
        kv.save(0, 2, &session_path).unwrap();
        let (_restored, pos) = KvState::load(&session_path, &model).unwrap();
        assert_eq!(pos, 2);

        // no tokenizer.json in this fixture -> the Qwen-specific loader's error, not a panic
        assert!(matches!(Tokenizer::load(&fixture.0, &model), Err(TokenizerError::Qwen38(_))));
    }

    /// The auto `--expert-cache` clamp must be REACHED for a Qwen checkpoint whose experts are MXFP4,
    /// with the right arguments: every layer counts as a routed-MoE layer (Qwen has no dense prefix).
    /// The clamp itself is RAM-aware (see `expert_cache::safe_mxfp4_capacity`), so a 4-layer fixture
    /// genuinely fits and is NOT clamped — the real 92-layer consequence is pinned separately below,
    /// by calling that helper directly, rather than by pretending the fixture is huge.
    #[test]
    fn expert_cache_default_routes_qwen38_mxfp4_through_the_ram_aware_clamp() {
        let fixture = crate::qwen38::model::tests::TempDir::new("rabbit_test_model_dispatch_qwen38_clamp");
        crate::qwen38::model::tests::write_tiny_checkpoint(&fixture.0);
        let model = Model::load(&fixture.0, 16, 4).unwrap();
        assert_eq!(safe_default_expert_cache_capacity(&model), 64, "a non-MXFP4 checkpoint keeps the flat default");

        let Model::Qwen38(mut m) = model else { panic!("expected the Qwen arm") };
        m.cfg.mxfp4_experts = true;
        m.cfg.hidden = 8192;
        m.cfg.moe_inter = 2048;
        let (layers, moe_inter, hidden) = (m.layers.len(), m.cfg.moe_inter as usize, m.cfg.hidden as usize);
        let got = safe_default_expert_cache_capacity(&Model::Qwen38(m));
        assert_eq!(
            got,
            crate::expert_cache::safe_mxfp4_capacity(64, layers, moe_inter, hidden),
            "the Qwen arm must pass ALL its layers as MoE layers, plus the real expert dimensions"
        );

        // ...and at the real model's 92 layers (92 x 64 x 28 MB = ~165 GB of experts) that helper
        // really does clamp, which is the whole reason this arm exists.
        let real = crate::expert_cache::safe_mxfp4_capacity(64, 92, 2048, 8192);
        assert!(real < 64, "92 MXFP4 layers must clamp below the flat default, got {real}");
        assert!(real >= 1, "...but never to zero");
    }

    #[test]
    fn load_and_step_dispatch_correctly_for_a_real_glm52_checkpoint() {
        let fixture = build_minimal_glm52_fixture("rabbit_test_model_dispatch_glm52_happy_path");
        let model = Model::load(&fixture.0, 32, 32).unwrap();
        assert!(matches!(model, Model::Glm52(_)));

        let shards = crate::safetensors::Shards::open(&fixture.0).unwrap();
        let mut caches = ExpertCaches::new(&model, 4);
        let mut kv = KvState::new(&model);
        let logits = step(&model, &shards, &mut caches, &mut kv, &[0, 1], 0).unwrap();
        assert_eq!(logits.len(), 10); // vocab
    }

    #[test]
    fn load_and_step_dispatch_correctly_for_a_real_kimi_linear_checkpoint() {
        let fixture = build_minimal_kimi_linear_fixture("rabbit_test_model_dispatch_kimi_happy_path");
        let model = Model::load(&fixture.0, 32, 32).unwrap();
        assert!(matches!(model, Model::KimiLinear(_)));

        let shards = crate::safetensors::Shards::open(&fixture.0).unwrap();
        let mut caches = ExpertCaches::new(&model, 4);
        let mut kv = KvState::new(&model);
        let logits = step(&model, &shards, &mut caches, &mut kv, &[0, 1], 0).unwrap();
        assert_eq!(logits.len(), 10); // vocab

        // a second, decode-style step (s=1, pos_base>0) must also work -- the KDA state has to
        // carry across dispatch calls the same way it does calling kimi_linear::generate directly.
        let logits2 = step(&model, &shards, &mut caches, &mut kv, &[2], 2).unwrap();
        assert_eq!(logits2.len(), 10);
    }

    #[test]
    fn load_and_step_dispatch_correctly_for_a_real_kimi_k3_checkpoint() {
        let fixture = build_minimal_k3_fixture("rabbit_test_model_dispatch_k3_happy_path");
        let model = Model::load(&fixture.0, 32, 32).unwrap();
        assert!(matches!(model, Model::KimiK3(_)));
        assert_eq!(model.n_layers(), 1);

        let shards = crate::safetensors::Shards::open(&fixture.0).unwrap();
        let mut caches = ExpertCaches::new(&model, 4);
        let mut kv = KvState::new(&model);
        let logits = step(&model, &shards, &mut caches, &mut kv, &[0, 1], 0).unwrap();
        assert_eq!(logits.len(), 10); // vocab

        // a second, decode-style step must also work through the dispatch enum, same as the
        // Kimi Linear test above.
        let logits2 = step(&model, &shards, &mut caches, &mut kv, &[2], 2).unwrap();
        assert_eq!(logits2.len(), 10);

        // --session works through the dispatch enum too -- save the two steps taken above,
        // reload into a fresh KvState, and confirm a follow-up step continues identically.
        let session_path = fixture.0.join("session.bin");
        kv.save(0, 3, &session_path).unwrap();
        let (mut kv_loaded, loaded_pos) = KvState::load(&session_path, &model).unwrap();
        assert_eq!(loaded_pos, 3);
        let mut caches_loaded = ExpertCaches::new(&model, 4);
        let logits3_original = step(&model, &shards, &mut caches, &mut kv, &[4], 3).unwrap();
        let logits3_loaded = step(&model, &shards, &mut caches_loaded, &mut kv_loaded, &[4], 3).unwrap();
        assert_eq!(logits3_original, logits3_loaded, "resumed K3 session must continue bit-for-bit identically");

        // The tokenizer IS wired -- but this fixture's dir has no tiktoken.model/
        // tokenizer_config.json (only the checkpoint's own tensors/config), so loading must fail
        // with a real (KimiK3-wrapped) IO error, not silently succeed or panic.
        assert!(matches!(Tokenizer::load(&fixture.0, &model), Err(TokenizerError::KimiK3(_))));

        // With the REAL fixture files (tests/fixtures/k3/, if present -- same policy as
        // k3_tokenizer_fixture.rs) copied alongside the checkpoint, the tokenizer must load and
        // dispatch for real.
        let real_fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/k3");
        if real_fixtures.join("tiktoken.model").is_file() {
            fs::copy(real_fixtures.join("tiktoken.model"), fixture.0.join("tiktoken.model")).unwrap();
            fs::copy(real_fixtures.join("tokenizer_config.json"), fixture.0.join("tokenizer_config.json")).unwrap();
            let tok = Tokenizer::load(&fixture.0, &model).expect("real K3 tokenizer fixture must load");
            let ids = tok.encode("hello");
            assert_eq!(tok.decode(&ids), b"hello");
        } else {
            eprintln!(
                "SKIP the real-tokenizer part of load_and_step_dispatch_correctly_for_a_real_kimi_k3_checkpoint: \
                 tests/fixtures/k3 not found — run tools/fetch_k3_tokenizer_fixture.py first."
            );
        }
    }

    #[test]
    fn step_profiled_dispatches_correctly_for_both_families_and_matches_step() {
        let glm_fixture = build_minimal_glm52_fixture("rabbit_test_model_dispatch_step_profiled_glm52");
        let glm_model = Model::load(&glm_fixture.0, 32, 32).unwrap();
        let glm_shards = crate::safetensors::Shards::open(&glm_fixture.0).unwrap();
        let mut glm_caches_a = ExpertCaches::new(&glm_model, 4);
        let mut glm_kv_a = KvState::new(&glm_model);
        let plain = step(&glm_model, &glm_shards, &mut glm_caches_a, &mut glm_kv_a, &[0, 1], 0).unwrap();
        let mut glm_caches_b = ExpertCaches::new(&glm_model, 4);
        let mut glm_kv_b = KvState::new(&glm_model);
        let (profiled, profile) = step_profiled(&glm_model, &glm_shards, &mut glm_caches_b, &mut glm_kv_b, &[0, 1], 0).unwrap();
        assert_eq!(plain, profiled);
        assert!(profile.phases.attention_s > 0.0);

        let kimi_fixture = build_minimal_kimi_linear_fixture("rabbit_test_model_dispatch_step_profiled_kimi");
        let kimi_model = Model::load(&kimi_fixture.0, 32, 32).unwrap();
        let kimi_shards = crate::safetensors::Shards::open(&kimi_fixture.0).unwrap();
        let mut kimi_caches_a = ExpertCaches::new(&kimi_model, 4);
        let mut kimi_kv_a = KvState::new(&kimi_model);
        let plain = step(&kimi_model, &kimi_shards, &mut kimi_caches_a, &mut kimi_kv_a, &[0, 1], 0).unwrap();
        let mut kimi_caches_b = ExpertCaches::new(&kimi_model, 4);
        let mut kimi_kv_b = KvState::new(&kimi_model);
        let (profiled, profile) = step_profiled(&kimi_model, &kimi_shards, &mut kimi_caches_b, &mut kimi_kv_b, &[0, 1], 0).unwrap();
        assert_eq!(plain, profiled);
        assert!(profile.phases.attention_s > 0.0);
    }

    #[test]
    fn model_helpers_report_sane_values_for_both_families() {
        let glm_fixture = build_minimal_glm52_fixture("rabbit_test_model_dispatch_helpers_glm52");
        let mut glm_model = Model::load(&glm_fixture.0, 32, 32).unwrap();
        assert_eq!(glm_model.n_layers(), 1);
        glm_model.set_cache_route(true); // must not panic; no getter to assert against generically

        let kimi_fixture = build_minimal_kimi_linear_fixture("rabbit_test_model_dispatch_helpers_kimi");
        let mut kimi_model = Model::load(&kimi_fixture.0, 32, 32).unwrap();
        assert_eq!(kimi_model.n_layers(), 1);
        kimi_model.set_cache_route(true);

        let k3_fixture = build_minimal_k3_fixture("rabbit_test_model_dispatch_helpers_k3");
        let mut k3_model = Model::load(&k3_fixture.0, 32, 32).unwrap();
        assert_eq!(k3_model.n_layers(), 1);
        k3_model.set_cache_route(true);
    }

    #[test]
    fn load_routes_kimi_linear_model_type_to_the_kimi_loader_not_glm() {
        // A minimal kimi_linear config that's missing required fields must fail INSIDE the
        // kimi_linear loader (a KimiLinear-wrapped error), never silently succeed via glm52's
        // loader or fail with glm52's own error type -- proof the model_type dispatch actually
        // picked the right family before any architecture-specific parsing began.
        let dir = TempDir::new("rabbit_test_model_dispatch_routes_kimi");
        fs::write(dir.0.join("config.json"), r#"{"model_type": "kimi_linear"}"#).unwrap();

        let err = match Model::load(&dir.0, 32, 32) {
            Err(e) => e,
            Ok(_) => panic!("expected an error"),
        };
        assert!(matches!(err, ModelError::KimiLinear(_)), "expected a KimiLinear-family error, got {err:?}");
    }
}
