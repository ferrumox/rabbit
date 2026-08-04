//! Loads a Kimi K3 checkpoint's dense-resident tensors — the sibling of
//! `kimi_linear::model.rs`, not a variant: reuses everything that's structurally identical
//! (`glm52::model::{AttnWeights, DenseMlpWeights, MoeWeights}`, `glm52::attention::attention()`,
//! MLA/KDA math) but K3's real checkpoint differs from Kimi Linear 48B in five real ways this
//! module has to account for, each already built and unit-tested in isolation this session:
//!
//! 1. **`q_lora_rank` is non-null** (K3's real checkpoint: `1536`) — unlike Kimi Linear 48B
//!    (always `null`), so MLA layers use `QProj::Lora` (GLM-5.2's own two-stage `q_b(rmsnorm(
//!    q_a(x)))`), not `QProj::Direct`. `mla_call_args` below picks per-checkpoint from
//!    `cfg.base.q_lora`, not hardcoded either way.
//! 2. **`SituAndMul` activation** (`glm52::moe::Activation::Situ`, new this session) replaces
//!    `Silu` for every FFN sublayer (dense, shared experts, routed experts) when
//!    `cfg.use_situ_activation` — see `ffn_activation` below.
//! 3. **Stable LatentMoE** (`kimi_k3::moe::LatentMoeWeights`) — an extra down/up-proj +
//!    optional norm around the routed-expert block, present per-MoE-layer when
//!    `cfg.routed_expert_hidden > 0`.
//! 4. **Attention Residuals** (`kimi_k3::attn_res::AttnResWeights`) — 4 extra learned
//!    norm+proj pairs per layer (`self_attn`/`mlp`) plus one model-level pair
//!    (`output_attn_res`), present uniformly across every layer (never per-layer optional) when
//!    `cfg.attn_res_block > 0`.
//! 5. **Two new attention gates**: MLA's optional `g_proj` (`glm52::attention::OutputGate`, new
//!    this session) when `cfg.mla_output_gate`; KDA's output gate is EITHER the existing
//!    low-rank `g_a_proj`/`g_b_proj` pair OR a full-rank `g_proj`, chosen by
//!    `cfg.kda_full_rank_gate` (`KdaOutputGate` below) — the math consuming either
//!    (`ops::head_output_gate`) is unchanged, only which weight produces its input differs.
//!
//! Every tensor name/shape below is taken from a REAL forward pass through the actual
//! `moonshotai/Kimi-K3` reference code this session (`tests/oracle/make_k3_oracle.py`'s printed
//! `state_dict`, with every K3-only field turned on at once) — not guessed.

use crate::glm52::attention::{Absorb, Dsa, QProj, Rope};
use crate::glm52::config::Cfg as GlmCfg;
use crate::glm52::model::{AttnWeights, DenseMlpWeights, MoeWeights};
use crate::glm52::moe::{Activation, RouteConfig};
use crate::kimi_k3::attn_res::AttnResWeights;
use crate::kimi_k3::config::{Cfg, ConfigError};
use crate::kimi_k3::moe::LatentMoeWeights;
use crate::quant::{PackedFormatError, QT};
use crate::safetensors::{SafetensorsError, Shards};
use rayon::prelude::*;
use std::fmt;
use std::path::{Path, PathBuf};

/// KDA's output gate projection — see this module's doc point 5. Both variants feed the same
/// `head_dim`-wide-per-head `g2` into `kimi_linear::ops::head_output_gate`, unchanged.
pub enum KdaOutputGate {
    LowRank { g_a_proj: QT, g_b_proj: QT },
    FullRank { g_proj: QT },
}

/// One KDA layer's attention weights — same fields as `kimi_linear::model::KdaWeights` except
/// `output_gate` (see `KdaOutputGate`) replacing that struct's always-low-rank
/// `g_a_proj`/`g_b_proj` pair.
pub struct KdaWeights {
    pub q_proj: QT,
    pub k_proj: QT,
    pub v_proj: QT,
    /// `[d_inner, kernel]` row-major — see `kimi_linear::model::KdaWeights::q_conv`'s doc.
    pub q_conv: Vec<f32>,
    pub k_conv: Vec<f32>,
    pub v_conv: Vec<f32>,
    pub f_a_proj: QT,
    pub f_b_proj: QT,
    pub dt_bias: Vec<f32>,
    /// `[kda_n_heads]` — one scalar per head, see `kda::decay_gate`'s doc.
    pub a_log: Vec<f32>,
    pub b_proj: QT,
    pub output_gate: KdaOutputGate,
    pub o_norm: Vec<f32>,
    pub o_proj: QT,
}

/// One MLA layer's weights: `glm52::model::AttnWeights`, reused unchanged (K3's MLA tensor
/// names/shapes match GLM-5.2's exactly, same as Kimi Linear 48B), plus an optional output-gate
/// projection (`Some` iff `cfg.mla_output_gate`).
pub struct MlaWeights {
    pub attn: AttnWeights,
    pub output_gate: Option<QT>,
}

pub enum Attn {
    Kda(Box<KdaWeights>),
    Mla(Box<MlaWeights>),
}

/// One MoE layer's FFN weights: the ordinary router + shared-expert weights
/// (`glm52::model::MoeWeights`, unchanged), plus the Stable LatentMoE down/up-proj wrapper when
/// `cfg.routed_expert_hidden > 0` (every real K3 MoE layer, but not assumed unconditional here).
pub struct MoeLayerWeights {
    pub moe: MoeWeights,
    pub latent: Option<LatentMoeWeights>,
}

// Load-time-only weights, not a hot-path per-token allocation -- the size skew between variants
// (Dense's 3 inline QTs vs Moe's already-boxed, mostly-empty MoeLayerWeights) doesn't matter here.
#[allow(clippy::large_enum_variant)]
pub enum Ffn {
    Dense(DenseMlpWeights),
    Moe(Box<MoeLayerWeights>),
}

/// One layer's Attention-Residuals weight pair — `self_attn` gates the pooling right before
/// self-attention, `mlp` right before the MLP/MoE (see `attn_res.rs::before_attention`/
/// `before_mlp`). Present on EVERY layer when `cfg.attn_res_block > 0` (a model-wide toggle, not
/// a per-layer one — confirmed via `KimiDecoderLayer.__init__`'s unconditional-per-layer
/// `if self.use_attn_residuals: <create these>`).
pub struct AttnResLayerWeights {
    pub self_attn: AttnResWeights,
    pub mlp: AttnResWeights,
}

pub struct Layer {
    pub in_ln: Vec<f32>,
    pub post_ln: Vec<f32>,
    pub attn: Attn,
    pub ffn: Ffn,
    pub attn_res: Option<AttnResLayerWeights>,
}

pub struct Model {
    pub cfg: Cfg,
    pub embed: QT,
    pub lm_head: QT,
    pub final_norm: Vec<f32>,
    pub layers: Vec<Layer>,
    /// The model-level final Attention-Residuals pooling (`KimiLinearModel._apply_output_attn_res`)
    /// — `Some` iff `cfg.attn_res_block > 0`.
    pub output_attn_res: Option<AttnResWeights>,
    pub ebits: u8,
    pub route_cfg: RouteConfig,
}

#[derive(Debug)]
pub enum ModelError {
    Config(ConfigError),
    Safetensors(SafetensorsError),
    ShapeMismatch { name: String, expected: usize, got: usize },
    PackedFormat { name: String, source: PackedFormatError },
    /// Propagated from `glm52::moe::moe()`/`kimi_k3::moe::latent_moe` (expert-loading I/O
    /// errors, mostly) — generate.rs's own error type, not this loader's, but the two share a
    /// `Result` type throughout `layer_forward` for the same reason `kimi_linear::model::
    /// ModelError` does.
    Glm52(crate::glm52::model::ModelError),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::Config(e) => write!(f, "{e}"),
            ModelError::Safetensors(e) => write!(f, "{e}"),
            ModelError::ShapeMismatch { name, expected, got } => {
                write!(f, "{name}: expected {expected} elements, got {got}")
            }
            ModelError::PackedFormat { name, source } => write!(f, "{name}.qs: {source}"),
            ModelError::Glm52(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ModelError {}

impl From<ConfigError> for ModelError {
    fn from(e: ConfigError) -> Self {
        ModelError::Config(e)
    }
}

impl From<SafetensorsError> for ModelError {
    fn from(e: SafetensorsError) -> Self {
        ModelError::Safetensors(e)
    }
}

impl From<crate::glm52::model::ModelError> for ModelError {
    fn from(e: crate::glm52::model::ModelError) -> Self {
        ModelError::Glm52(e)
    }
}

fn ld(shards: &Shards, name: &str) -> Result<Vec<f32>, ModelError> {
    Ok(shards.read_f32(name, false)?)
}

fn qt_load(shards: &Shards, group_size: usize, name: &str, rows: usize, cols: usize, bits: u8) -> Result<QT, ModelError> {
    let qs_name = format!("{name}.qs");
    if shards.has(&qs_name) {
        let data = shards.read_raw(name, false)?;
        let scale = shards.read_f32(&qs_name, false)?;
        return QT::from_packed_grouped(rows, cols, bits, data, scale, group_size)
            .map_err(|source| ModelError::PackedFormat { name: name.to_string(), source });
    }

    let w = shards.read_f32(name, false)?;
    if w.len() != rows * cols {
        return Err(ModelError::ShapeMismatch { name: name.to_string(), expected: rows * cols, got: w.len() });
    }
    let mut t = QT::alloc(rows, cols, bits, false);
    t.fill(&w);
    Ok(t)
}

/// Builds a `glm52::config::Cfg` driving both `attention()` (MLA layers) and `glm52::moe`
/// (every layer's FFN) — same reasoning as `kimi_linear::model::to_glm_cfg`, with one real
/// difference: `q_lora` carries K3's REAL value (`cfg.base.q_lora`, e.g. `1536`), not hardcoded
/// `0` — K3's MLA layers genuinely use the LoRA q path, unlike Kimi Linear 48B's always-Direct.
pub fn to_glm_cfg(cfg: &Cfg) -> GlmCfg {
    let b = &cfg.base;
    GlmCfg {
        hidden: b.hidden,
        n_layers: b.n_layers,
        n_heads: b.n_heads,
        n_experts: b.n_experts,
        topk: b.topk,
        moe_inter: b.moe_inter,
        dense_inter: b.dense_inter,
        first_dense: 0,
        q_lora: b.q_lora,
        kv_lora: b.kv_lora,
        qk_nope: b.qk_nope,
        qk_rope: b.qk_rope,
        qk_head: b.qk_head,
        v_head: b.v_head,
        n_shared: b.n_shared,
        vocab: b.vocab,
        n_group: b.n_group,
        topk_group: b.topk_group,
        norm_topk: b.norm_topk,
        stop_ids: vec![],
        index_topk: 0,
        index_nh: 0,
        index_hd: 0,
        idx_type: vec![],
        eps: b.eps,
        theta: b.theta,
        attn_scale: b.attn_scale,
        routed_scale: b.routed_scale,
        group_size: b.group_size,
    }
}

/// Like `to_glm_cfg`, but with `.hidden` overridden to `cfg.routed_expert_hidden` when the
/// latent-MoE wrapper is active — the width `kimi_k3::moe::latent_moe`'s `cfg_expert` parameter
/// needs, since routed-expert tensors are genuinely shaped for that narrower width on disk (see
/// `kimi_k3::moe`'s doc). Identical to `to_glm_cfg` when latent-MoE is off.
pub fn to_glm_cfg_expert(cfg: &Cfg) -> GlmCfg {
    let mut g = to_glm_cfg(cfg);
    if cfg.routed_expert_hidden > 0 {
        g.hidden = cfg.routed_expert_hidden;
    }
    g
}

/// The `(glm_cfg, Dsa, Absorb, Rope, QProj)` combination every K3 MLA layer's `attention()` call
/// needs — `Dsa::Off`/`Rope::Off` always (no DSA indexer; `mla_use_nope: true` always, checked at
/// config-load time), `QProj` chosen per-checkpoint from `cfg.base.q_lora` (see this module's
/// doc point 1) rather than hardcoded either way like `kimi_linear::model::mla_call_args`.
pub fn mla_call_args(cfg: &Cfg) -> (GlmCfg, Dsa<'static>, Absorb, Rope, QProj) {
    let qproj = if cfg.base.q_lora > 0 { QProj::Lora } else { QProj::Direct };
    (to_glm_cfg(cfg), Dsa::Off, Absorb::Auto, Rope::Off, qproj)
}

/// The FFN gating activation every dense/shared/routed-expert MLP in this checkpoint uses (see
/// this module's doc point 2) — `Situ` when `cfg.use_situ_activation`, else `Silu`.
pub fn ffn_activation(cfg: &Cfg) -> Activation {
    if cfg.use_situ_activation {
        Activation::Situ { beta: cfg.situ_beta, linear_beta: cfg.situ_linear_beta }
    } else {
        Activation::Silu
    }
}

impl Model {
    pub fn load(snap_dir: &Path, dbits: u8, ebits: u8) -> Result<Model, ModelError> {
        Self::load_multi(std::slice::from_ref(&snap_dir.to_path_buf()), dbits, ebits)
    }

    /// Same as `load`, but reads the checkpoint's `.safetensors` shards from MULTIPLE
    /// directories — see `Shards::open_multi`'s doc.
    pub fn load_multi(dirs: &[PathBuf], dbits: u8, ebits: u8) -> Result<Model, ModelError> {
        // Diagnostic timing breakdown (2026-07-28): model load takes ~610s on the real 2.8T
        // checkpoint and nothing had ever measured WHERE that time actually goes -- see
        // PERFORMANCE_KIMI_K3.md for whatever this investigation found. Three phases: opening
        // shards (header parsing only, expected near-instant per `Shards::open_multi`'s own
        // doc), `embed_tokens`/`lm_head`/misc (two huge vocab-sized tensors, read+dequant+
        // requant, no `.qs` sidecar on the real checkpoint so this pays full BF16->f32->4bit
        // conversion cost), and the 93-layer loop itself (hundreds of small sequential
        // `ld`/`qt_load` calls, one blocking read each -- the same "many small syscalls" shape
        // `io_uring` batching fixed for routed experts, never applied here).
        let t_open = std::time::Instant::now();
        let cfg = Cfg::load(&dirs[0])?;
        let shards = Shards::open_multi(dirs)?;
        let open_s = t_open.elapsed().as_secs_f32();
        let t_head = std::time::Instant::now();
        let b = &cfg.base;
        let d = b.hidden as usize;
        let h = b.n_heads as usize;
        let qh = b.qk_head as usize;
        let qk_nope = b.qk_nope as usize;
        let qk_rope = b.qk_rope as usize;
        let v_head = b.v_head as usize;
        let kv_lora = b.kv_lora as usize;
        let q_lora = b.q_lora as usize;
        let d_inner = (b.kda_n_heads * b.kda_head_dim) as usize;
        let kda_head_dim = b.kda_head_dim as usize;
        let kda_n_heads = b.kda_n_heads as usize;
        let moe_hidden = cfg.routed_expert_hidden as usize;

        // K3's real checkpoint is the FULL multimodal `KimiK3ForConditionalGeneration`, which
        // wraps the text backbone as `self.language_model = KimiLinearForCausalLM(...)` --
        // EVERY text tensor lives under a `language_model.` prefix (confirmed against the real
        // `model.safetensors.index.json`, fetched 2026-07-27: e.g.
        // `language_model.model.layers.0.input_layernorm.weight`,
        // `language_model.lm_head.weight` -- note `lm_head` has NO `.model.` in between, unlike
        // everything else). The tiny oracle's `KimiLinearForCausalLM` (text-only, no vision
        // wrapper) has no such prefix, which is why this wasn't caught by the oracle test --
        // that's a genuinely different real top-level module, not a bug in the oracle.
        let io_bits = if dbits >= 8 { 16 } else { dbits };
        let t_embed = std::time::Instant::now();
        let embed = qt_load(&shards, b.group_size as usize, "language_model.model.embed_tokens.weight", b.vocab as usize, d, io_bits)?;
        let embed_s = t_embed.elapsed().as_secs_f32();
        let t_lm_head = std::time::Instant::now();
        let lm_head = qt_load(&shards, b.group_size as usize, "language_model.lm_head.weight", b.vocab as usize, d, io_bits)?;
        let lm_head_s = t_lm_head.elapsed().as_secs_f32();
        let final_norm = ld(&shards, "language_model.model.norm.weight")?;

        let output_attn_res = if cfg.attn_res_block > 0 {
            Some(AttnResWeights {
                norm: ld(&shards, "language_model.model.output_attn_res_norm.weight")?,
                proj: ld(&shards, "language_model.model.output_attn_res_proj.weight")?,
            })
        } else {
            None
        };
        let head_s = t_head.elapsed().as_secs_f32();

        let t_layers = std::time::Instant::now();
        // Parallelized across layers (2026-07-28): each layer's tensors are an independent,
        // read-only pread from `shards` (`Shards::read_f32`/`read_raw` take `&self`, explicit
        // offset, no shared cursor -- concurrent calls are safe), so this used to be ~600s of
        // hundreds of small sequential blocking reads with zero overlap, one at a time, the same
        // "many syscalls each paying full disk latency serially" shape `io_uring` batching fixed
        // for routed experts -- never applied here until now. `into_par_iter` lets rayon's
        // thread pool keep the drive's queue depth busy the same way `expert_cache.rs`'s
        // `io_uring` batching does, just via plain OS threads instead of one ring (simpler,
        // since load-time reads don't need the early-drain streaming trick expert loading does
        // -- nothing downstream can start using layer N until every layer is loaded anyway, so
        // there's no completion-order benefit to chase here). `collect::<Result<Vec<_>, _>>()`
        // preserves layer order regardless of which finishes first, and surfaces the first error
        // encountered (if any) same as the sequential loop's own `?` always did.
        let layers: Vec<Layer> = (0..b.n_layers as usize)
            .into_par_iter()
            .map(|i| -> Result<Layer, ModelError> {
            let p = |s: &str| format!("language_model.model.layers.{i}.{s}");
            let ap = |s: &str| format!("language_model.model.layers.{i}.self_attn.{s}");

            let in_ln = ld(&shards, &p("input_layernorm.weight"))?;
            let post_ln = ld(&shards, &p("post_attention_layernorm.weight"))?;

            let attn = if b.is_kda[i] {
                let output_gate = if cfg.kda_full_rank_gate {
                    KdaOutputGate::FullRank { g_proj: qt_load(&shards, b.group_size as usize, &ap("g_proj.weight"), d_inner, d, dbits)? }
                } else {
                    KdaOutputGate::LowRank {
                        g_a_proj: qt_load(&shards, b.group_size as usize, &ap("g_a_proj.weight"), kda_head_dim, d, dbits)?,
                        g_b_proj: qt_load(&shards, b.group_size as usize, &ap("g_b_proj.weight"), d_inner, kda_head_dim, dbits)?,
                    }
                };
                Attn::Kda(Box::new(KdaWeights {
                    q_proj: qt_load(&shards, b.group_size as usize, &ap("q_proj.weight"), d_inner, d, dbits)?,
                    k_proj: qt_load(&shards, b.group_size as usize, &ap("k_proj.weight"), d_inner, d, dbits)?,
                    v_proj: qt_load(&shards, b.group_size as usize, &ap("v_proj.weight"), d_inner, d, dbits)?,
                    q_conv: ld(&shards, &ap("q_conv1d.weight"))?,
                    k_conv: ld(&shards, &ap("k_conv1d.weight"))?,
                    v_conv: ld(&shards, &ap("v_conv1d.weight"))?,
                    f_a_proj: qt_load(&shards, b.group_size as usize, &ap("f_a_proj.weight"), kda_head_dim, d, dbits)?,
                    f_b_proj: qt_load(&shards, b.group_size as usize, &ap("f_b_proj.weight"), d_inner, kda_head_dim, dbits)?,
                    dt_bias: ld(&shards, &ap("dt_bias"))?,
                    a_log: ld(&shards, &ap("A_log"))?,
                    b_proj: qt_load(&shards, b.group_size as usize, &ap("b_proj.weight"), kda_n_heads, d, dbits)?,
                    output_gate,
                    o_norm: ld(&shards, &ap("o_norm.weight"))?,
                    o_proj: qt_load(&shards, b.group_size as usize, &ap("o_proj.weight"), d, d_inner, dbits)?,
                }))
            } else {
                let attn = if q_lora > 0 {
                    AttnWeights {
                        q_a: qt_load(&shards, b.group_size as usize, &ap("q_a_proj.weight"), q_lora, d, dbits)?,
                        q_a_ln: ld(&shards, &ap("q_a_layernorm.weight"))?,
                        q_b: qt_load(&shards, b.group_size as usize, &ap("q_b_proj.weight"), h * qh, q_lora, dbits)?,
                        kv_a: qt_load(&shards, b.group_size as usize, &ap("kv_a_proj_with_mqa.weight"), kv_lora + qk_rope, d, dbits)?,
                        kv_a_ln: ld(&shards, &ap("kv_a_layernorm.weight"))?,
                        kv_b: qt_load(&shards, b.group_size as usize, &ap("kv_b_proj.weight"), h * (qk_nope + v_head), kv_lora, dbits)?,
                        o: qt_load(&shards, b.group_size as usize, &ap("o_proj.weight"), d, h * v_head, dbits)?,
                    }
                } else {
                    AttnWeights {
                        q_a: qt_load(&shards, b.group_size as usize, &ap("q_proj.weight"), h * qh, d, dbits)?,
                        q_a_ln: vec![],
                        q_b: QT::alloc(1, 1, 32, false),
                        kv_a: qt_load(&shards, b.group_size as usize, &ap("kv_a_proj_with_mqa.weight"), kv_lora + qk_rope, d, dbits)?,
                        kv_a_ln: ld(&shards, &ap("kv_a_layernorm.weight"))?,
                        kv_b: qt_load(&shards, b.group_size as usize, &ap("kv_b_proj.weight"), h * (qk_nope + v_head), kv_lora, dbits)?,
                        o: qt_load(&shards, b.group_size as usize, &ap("o_proj.weight"), d, h * v_head, dbits)?,
                    }
                };
                let output_gate = if cfg.mla_output_gate {
                    Some(qt_load(&shards, b.group_size as usize, &ap("g_proj.weight"), h * v_head, d, dbits)?)
                } else {
                    None
                };
                Attn::Mla(Box::new(MlaWeights { attn, output_gate }))
            };

            let sparse = i >= b.first_dense as usize;
            let ffn = if !sparse {
                Ffn::Dense(DenseMlpWeights {
                    gate_proj: qt_load(&shards, b.group_size as usize, &p("mlp.gate_proj.weight"), b.dense_inter as usize, d, dbits)?,
                    up_proj: qt_load(&shards, b.group_size as usize, &p("mlp.up_proj.weight"), b.dense_inter as usize, d, dbits)?,
                    down_proj: qt_load(&shards, b.group_size as usize, &p("mlp.down_proj.weight"), d, b.dense_inter as usize, dbits)?,
                })
            } else {
                let mp = |s: &str| format!("language_model.model.layers.{i}.block_sparse_moe.{s}");
                let s_i = (b.moe_inter * b.n_shared) as usize;
                let moe = MoeWeights {
                    router: ld(&shards, &mp("gate.weight"))?,
                    router_bias: ld(&shards, &mp("gate.e_score_correction_bias"))?,
                    sh_gate: qt_load(&shards, b.group_size as usize, &mp("shared_experts.gate_proj.weight"), s_i, d, dbits)?,
                    sh_up: qt_load(&shards, b.group_size as usize, &mp("shared_experts.up_proj.weight"), s_i, d, dbits)?,
                    sh_down: qt_load(&shards, b.group_size as usize, &mp("shared_experts.down_proj.weight"), d, s_i, dbits)?,
                };
                let latent = if cfg.routed_expert_hidden > 0 {
                    let norm = if cfg.latent_moe_norm { Some(ld(&shards, &mp("routed_expert_norm.weight"))?) } else { None };
                    Some(LatentMoeWeights {
                        down_proj: qt_load(&shards, b.group_size as usize, &mp("routed_expert_down_proj.weight"), moe_hidden, d, dbits)?,
                        up_proj: qt_load(&shards, b.group_size as usize, &mp("routed_expert_up_proj.weight"), d, moe_hidden, dbits)?,
                        norm,
                    })
                } else {
                    None
                };
                Ffn::Moe(Box::new(MoeLayerWeights { moe, latent }))
            };

            let attn_res = if cfg.attn_res_block > 0 {
                Some(AttnResLayerWeights {
                    self_attn: AttnResWeights { norm: ld(&shards, &p("self_attention_res_norm.weight"))?, proj: ld(&shards, &p("self_attention_res_proj.weight"))? },
                    mlp: AttnResWeights { norm: ld(&shards, &p("mlp_res_norm.weight"))?, proj: ld(&shards, &p("mlp_res_proj.weight"))? },
                })
            } else {
                None
            };

            Ok(Layer { in_ln, post_ln, attn, ffn, attn_res })
            })
            .collect::<Result<Vec<Layer>, ModelError>>()?;
        let layers_s = t_layers.elapsed().as_secs_f32();

        eprintln!(
            "model load breakdown: shard-open {open_s:.1}s, embed {embed_s:.1}s, lm_head {lm_head_s:.1}s, \
             other head tensors {:.1}s, {} layers {layers_s:.1}s",
            (head_s - embed_s - lm_head_s).max(0.0),
            b.n_layers
        );

        Ok(Model { cfg, embed, lm_head, final_norm, layers, output_attn_res, ebits, route_cfg: RouteConfig::default() })
    }

    pub fn embed_row(&self, tok: usize) -> Vec<f32> {
        self.embed.row_f32(tok)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
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

    fn xorshift(seed: &mut u32) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 17;
        *seed ^= *seed << 5;
        ((*seed as f32 / u32::MAX as f32) - 0.5) * 2.0
    }

    fn random_row(n: usize, seed: &mut u32) -> Vec<f32> {
        (0..n).map(|_| xorshift(seed)).collect()
    }

    fn f32_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// A tiny 3-layer K3-shaped fixture exercising every branch `Model::load` has: layer 0 and 1
    /// KDA (`with_full_rank_gate()` picks which output-gate style both use — a single
    /// checkpoint-wide flag, not per-layer, see that method's doc) + dense/latent-MoE FFN
    /// respectively, layer 2 MLA (`q_lora>0`, output gate) + latent-MoE FFN — plus Attention
    /// Residuals weights on every layer and at the model level (`attn_res_block=2` so both a
    /// checkpoint layer,
    /// index 0, and a non-checkpoint layer, index 1, are covered).
    struct TinyFixture {
        dir: TempDir,
        tensors: Vec<(String, Vec<usize>, Vec<u8>)>,
        seed: u32,
        /// `kda_full_rank_gate` is one flag for the WHOLE checkpoint (`linear_attn_config.
        /// use_full_rank_gate`), not per-layer -- every KDA layer in a real checkpoint uses the
        /// same style, so this fixture applies it uniformly too.
        full_rank_gate: bool,
    }

    impl TinyFixture {
        fn new(name: &str) -> Self {
            TinyFixture { dir: TempDir::new(&format!("rabbit_test_k3_model_tiny_{name}")), tensors: Vec::new(), seed: 1, full_rank_gate: false }
        }

        fn with_full_rank_gate(mut self) -> Self {
            self.full_rank_gate = true;
            self
        }

        fn add(&mut self, name: &str, shape: Vec<usize>) {
            let n: usize = shape.iter().product::<usize>().max(1);
            let data = random_row(n, &mut self.seed);
            self.tensors.push((name.to_string(), shape, f32_bytes(&data)));
        }

        fn build(mut self) -> TempDir {
            let d = 8; // hidden
            let h = 2; // MLA heads
            let qk_nope = 3;
            let qk_rope = 2;
            let qh = qk_nope + qk_rope;
            let v_head = 4;
            let kv_lora = 5;
            let q_lora = 6;
            let vocab = 16;
            let dense_inter = 10;
            let n_experts = 4;
            let moe_inter = 6;
            let n_shared = 1;
            let kda_head_dim = 4;
            let kda_n_heads = 2;
            let d_inner = kda_head_dim * kda_n_heads;
            let kernel = 3;
            let moe_hidden = 4; // routed_expert_hidden_size, < d

            self.add("language_model.model.embed_tokens.weight", vec![vocab, d]);
            self.add("language_model.lm_head.weight", vec![vocab, d]);
            self.add("language_model.model.norm.weight", vec![d]);
            self.add("language_model.model.output_attn_res_norm.weight", vec![d]);
            self.add("language_model.model.output_attn_res_proj.weight", vec![1, d]);

            for i in 0..3usize {
                let p = |s: &str| format!("language_model.model.layers.{i}.{s}");
                self.add(&p("input_layernorm.weight"), vec![d]);
                self.add(&p("post_attention_layernorm.weight"), vec![d]);
                self.add(&p("self_attention_res_norm.weight"), vec![d]);
                self.add(&p("self_attention_res_proj.weight"), vec![1, d]);
                self.add(&p("mlp_res_norm.weight"), vec![d]);
                self.add(&p("mlp_res_proj.weight"), vec![1, d]);

                let is_kda = i < 2; // layers 0,1 KDA; layer 2 MLA
                if is_kda {
                    let ap = |s: &str| format!("language_model.model.layers.{i}.self_attn.{s}");
                    self.add(&ap("q_proj.weight"), vec![d_inner, d]);
                    self.add(&ap("k_proj.weight"), vec![d_inner, d]);
                    self.add(&ap("v_proj.weight"), vec![d_inner, d]);
                    self.add(&ap("q_conv1d.weight"), vec![d_inner, 1, kernel]);
                    self.add(&ap("k_conv1d.weight"), vec![d_inner, 1, kernel]);
                    self.add(&ap("v_conv1d.weight"), vec![d_inner, 1, kernel]);
                    self.add(&ap("f_a_proj.weight"), vec![kda_head_dim, d]);
                    self.add(&ap("f_b_proj.weight"), vec![d_inner, kda_head_dim]);
                    self.add(&ap("dt_bias"), vec![d_inner]);
                    self.add(&ap("A_log"), vec![1, 1, kda_n_heads, 1]);
                    self.add(&ap("b_proj.weight"), vec![kda_n_heads, d]);
                    if self.full_rank_gate {
                        self.add(&ap("g_proj.weight"), vec![d_inner, d]);
                    } else {
                        self.add(&ap("g_a_proj.weight"), vec![kda_head_dim, d]);
                        self.add(&ap("g_b_proj.weight"), vec![d_inner, kda_head_dim]);
                    }
                    self.add(&ap("o_norm.weight"), vec![kda_head_dim]);
                    self.add(&ap("o_proj.weight"), vec![d, d_inner]);
                } else {
                    let ap = |s: &str| format!("language_model.model.layers.{i}.self_attn.{s}");
                    self.add(&ap("q_a_proj.weight"), vec![q_lora, d]);
                    self.add(&ap("q_a_layernorm.weight"), vec![q_lora]);
                    self.add(&ap("q_b_proj.weight"), vec![h * qh, q_lora]);
                    self.add(&ap("kv_a_proj_with_mqa.weight"), vec![kv_lora + qk_rope, d]);
                    self.add(&ap("kv_a_layernorm.weight"), vec![kv_lora]);
                    self.add(&ap("kv_b_proj.weight"), vec![h * (qk_nope + v_head), kv_lora]);
                    self.add(&ap("o_proj.weight"), vec![d, h * v_head]);
                    self.add(&ap("g_proj.weight"), vec![h * v_head, d]); // mla_use_output_gate
                }

                if i == 0 {
                    self.add(&p("mlp.gate_proj.weight"), vec![dense_inter, d]);
                    self.add(&p("mlp.up_proj.weight"), vec![dense_inter, d]);
                    self.add(&p("mlp.down_proj.weight"), vec![d, dense_inter]);
                } else {
                    let mp = |s: &str| format!("language_model.model.layers.{i}.block_sparse_moe.{s}");
                    self.add(&mp("gate.weight"), vec![n_experts, d]);
                    self.add(&mp("gate.e_score_correction_bias"), vec![n_experts]);
                    let s_i = moe_inter * n_shared;
                    self.add(&mp("shared_experts.gate_proj.weight"), vec![s_i, d]);
                    self.add(&mp("shared_experts.up_proj.weight"), vec![s_i, d]);
                    self.add(&mp("shared_experts.down_proj.weight"), vec![d, s_i]);
                    self.add(&mp("routed_expert_down_proj.weight"), vec![moe_hidden, d]);
                    self.add(&mp("routed_expert_up_proj.weight"), vec![d, moe_hidden]);
                    self.add(&mp("routed_expert_norm.weight"), vec![moe_hidden]);
                    // routed expert tensors deliberately NOT written -- Model::load never
                    // touches them, matching glm52::model::Model::load's own streaming design.
                }
            }

            // Split into two `json!` calls -- one deeply nested literal hits serde_json's macro
            // recursion limit.
            let linear_attn_config = json!({
                "head_dim": kda_head_dim, "num_heads": kda_n_heads, "short_conv_kernel_size": kernel,
                "gate_lower_bound": -5.0, "use_full_rank_gate": self.full_rank_gate,
                "kda_layers": [1, 2], "full_attn_layers": [3]
            });
            let text_config = json!({
                "model_type": "kimi_linear",
                "hidden_size": d, "num_hidden_layers": 3, "num_attention_heads": h,
                "first_k_dense_replace": 1, "q_lora_rank": q_lora, "kv_lora_rank": kv_lora,
                "qk_nope_head_dim": qk_nope, "qk_rope_head_dim": qk_rope, "v_head_dim": v_head,
                "num_experts": n_experts, "num_experts_per_token": 2, "num_shared_experts": n_shared,
                "num_expert_group": 1, "topk_group": 1, "moe_intermediate_size": moe_inter,
                "intermediate_size": dense_inter, "vocab_size": vocab, "moe_renormalize": true,
                "rms_norm_eps": 1e-5, "routed_scaling_factor": 1.0, "mla_use_nope": true,
                "moe_router_activation_func": "sigmoid", "rope_theta": 10000.0,
                "hidden_act": "situ", "activation_situ_beta": 4.0, "activation_situ_linear_beta": 25.0,
                "routed_expert_hidden_size": moe_hidden, "latent_moe_use_norm": true,
                "attn_res_block_size": 2, "mla_use_output_gate": true,
                "linear_attn_config": linear_attn_config
            });
            let cfg_json = json!({ "model_type": "kimi_k3", "text_config": text_config });
            fs::write(self.dir.0.join("config.json"), cfg_json.to_string()).unwrap();

            let mut header = serde_json::Map::new();
            header.insert("__metadata__".to_string(), json!({"format": "rabbit-test"}));
            let mut data = Vec::new();
            for (name, shape, bytes) in &self.tensors {
                let start = data.len() as u64;
                data.extend_from_slice(bytes);
                let end = data.len() as u64;
                header.insert(name.clone(), json!({"dtype": "F32", "shape": shape, "data_offsets": [start, end]}));
            }
            let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
            let mut out = Vec::new();
            out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
            out.extend_from_slice(&header_bytes);
            out.extend_from_slice(&data);
            fs::write(self.dir.0.join("model.safetensors"), out).unwrap();

            self.dir
        }
    }

    #[test]
    fn loads_kda_and_mla_layers_with_every_k3_only_field_populated() {
        let fixture = TinyFixture::new("shapes").build();
        let m = Model::load(&fixture.0, 32, 32).unwrap();

        assert_eq!(m.layers.len(), 3);
        assert!(matches!(m.layers[0].attn, Attn::Kda(_)), "layer 0 must be KDA");
        assert!(matches!(m.layers[1].attn, Attn::Kda(_)), "layer 1 must be KDA");
        assert!(matches!(m.layers[2].attn, Attn::Mla(_)), "layer 2 must be MLA");

        assert!(matches!(m.layers[0].ffn, Ffn::Dense(_)), "layer 0 is before first_k_dense_replace=1");
        assert!(matches!(m.layers[1].ffn, Ffn::Moe(_)));
        assert!(matches!(m.layers[2].ffn, Ffn::Moe(_)));

        for li in [0, 1] {
            if let Attn::Kda(kda) = &m.layers[li].attn {
                assert!(matches!(kda.output_gate, KdaOutputGate::LowRank { .. }), "default fixture (no with_full_rank_gate()) uses the low-rank gate");
            } else {
                panic!("expected KDA");
            }
        }

        if let Attn::Mla(mla) = &m.layers[2].attn {
            assert_eq!(mla.attn.q_a.rows, 6); // q_lora
            assert_eq!(mla.attn.q_a.cols, 8); // hidden
            assert!(!mla.attn.q_a_ln.is_empty(), "q_a_ln must be real (QProj::Lora), not an unread placeholder");
            assert!(mla.output_gate.is_some(), "mla_use_output_gate=true in the fixture config");
        } else {
            panic!("expected MLA");
        }

        if let Ffn::Moe(m1) = &m.layers[1].ffn {
            assert_eq!(m1.moe.router.len(), 4 * 8); // n_experts * hidden
            let latent = m1.latent.as_ref().expect("routed_expert_hidden_size > 0 in the fixture config");
            assert_eq!(latent.down_proj.rows, 4); // moe_hidden
            assert_eq!(latent.down_proj.cols, 8); // hidden
            assert!(latent.norm.is_some(), "latent_moe_use_norm=true in the fixture config");
        } else {
            panic!("expected MoE");
        }

        for i in 0..3 {
            let ar = m.layers[i].attn_res.as_ref().expect("attn_res_block_size=2 in the fixture config -> every layer has these weights");
            assert_eq!(ar.self_attn.norm.len(), 8);
            assert_eq!(ar.self_attn.proj.len(), 8); // [1, hidden] flat-read is hidden-long
            assert_eq!(ar.mlp.norm.len(), 8);
        }
        let out_ar = m.output_attn_res.as_ref().expect("attn_res_block_size=2 -> model-level pooling weights present");
        assert_eq!(out_ar.norm.len(), 8);
    }

    #[test]
    fn to_glm_cfg_carries_the_real_nonzero_q_lora() {
        let fixture = TinyFixture::new("q_lora").build();
        let m = Model::load(&fixture.0, 32, 32).unwrap();
        let glm_cfg = to_glm_cfg(&m.cfg);
        assert_eq!(glm_cfg.q_lora, 6, "K3's q_lora_rank is non-null, unlike Kimi Linear 48B -- to_glm_cfg must NOT zero it");
    }

    #[test]
    fn to_glm_cfg_expert_overrides_hidden_to_the_latent_width() {
        let fixture = TinyFixture::new("latent_width").build();
        let m = Model::load(&fixture.0, 32, 32).unwrap();
        let full = to_glm_cfg(&m.cfg);
        let expert = to_glm_cfg_expert(&m.cfg);
        assert_eq!(full.hidden, 8);
        assert_eq!(expert.hidden, 4, "routed_expert_hidden_size from the fixture config");
    }

    #[test]
    fn mla_call_args_picks_lora_when_q_lora_is_nonzero() {
        let fixture = TinyFixture::new("mla_args").build();
        let m = Model::load(&fixture.0, 32, 32).unwrap();
        let (_, _, _, rope, qproj) = mla_call_args(&m.cfg);
        assert!(matches!(qproj, QProj::Lora));
        assert!(matches!(rope, Rope::Off), "mla_use_nope=true -> RoPE-free, same as Kimi Linear 48B");
    }

    #[test]
    fn ffn_activation_is_situ_with_the_real_checkpoint_betas() {
        let fixture = TinyFixture::new("activation").build();
        let m = Model::load(&fixture.0, 32, 32).unwrap();
        match ffn_activation(&m.cfg) {
            Activation::Situ { beta, linear_beta } => {
                assert!((beta - 4.0).abs() < 1e-6);
                assert_eq!(linear_beta, Some(25.0));
            }
            Activation::Silu => panic!("expected Situ, the fixture config sets hidden_act=situ"),
        }
    }

    #[test]
    fn embed_row_dequantizes_the_right_row() {
        let fixture = TinyFixture::new("embed_row").build();
        let m = Model::load(&fixture.0, 32, 32).unwrap();
        let row2 = m.embed_row(2);
        assert_eq!(row2.len(), 8); // hidden
    }

    #[test]
    fn loads_the_full_rank_kda_output_gate_when_configured() {
        let fixture = TinyFixture::new("full_rank_gate").with_full_rank_gate().build();
        let m = Model::load(&fixture.0, 32, 32).unwrap();
        for li in [0, 1] {
            if let Attn::Kda(kda) = &m.layers[li].attn {
                match &kda.output_gate {
                    KdaOutputGate::FullRank { g_proj } => {
                        assert_eq!(g_proj.rows, 8); // d_inner
                        assert_eq!(g_proj.cols, 8); // hidden
                    }
                    KdaOutputGate::LowRank { .. } => panic!("with_full_rank_gate() must produce FullRank"),
                }
            } else {
                panic!("expected KDA");
            }
        }
    }
}
