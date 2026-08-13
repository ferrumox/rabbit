//! Loads a real Qwen 3.8 checkpoint's non-expert weights into memory: embeddings, the 92 layers'
//! mixers (Gated DeltaNet or GQA attention, per `cfg.is_linear`), each layer's MoE router and shared
//! expert, and the final norm. The 512 routed experts per layer are NOT loaded here — they stay on
//! disk and stream through `expert_cache.rs` (see [`expert_naming`]), which is the whole point of
//! this engine.
//!
//! Every tensor name below was read from the real
//! `amd/Qwen3.8-2.4T-A95B-Quark-MXFP4/model.safetensors.index.json` (287,119 entries, downloaded
//! 2026-08-13), not guessed. Unlike Kimi K3's checkpoint there is no `language_model.` prefix — Qwen
//! 3.8-Max is a text-only `Qwen3_5MoeForCausalLM`, so the tree starts at `model.` (and `lm_head` sits
//! at the top level).
//!
//! **Sizing:** ~76 B non-expert parameters arrive as BF16 (49.6 GB on the wire) and get requantized
//! to `dbits` on load, the same BF16 -> f32 -> int4 path K3's loader pays; expect ~40 GB resident at
//! `--dbits 4` with 8-bit embeddings. The 92-layer loop is parallelized with `rayon` for the reason
//! K3's v0.26.0 entry documents (that change alone took its load from ~610 s to ~102 s): every
//! `Shards::read_*` takes `&self` with an explicit offset, so concurrent reads are safe and keep the
//! drive's queue busy instead of paying full latency one blocking read at a time.
//!
//! **`mtp.*` is skipped entirely.** The checkpoint's multi-token-prediction block is a FULL extra MoE
//! layer (its own 512 experts, ~14 GB of MXFP4) and rabbit does no speculative decoding; nothing here
//! reads those tensors.

use crate::expert_cache::ExpertNaming;
use crate::glm52::config::Cfg as GlmCfg;
use crate::quant::QT;
use crate::qwen38::config::{Cfg, ConfigError};
use crate::safetensors::{SafetensorsError, Shards};
use rayon::prelude::*;
use std::fmt;
use std::path::{Path, PathBuf};

/// One Gated DeltaNet layer's weights (`model.layers.{i}.linear_attn.*`).
pub struct GdnWeights {
    /// `[conv_dim, hidden]`, `conv_dim = 2 * key_dim + value_dim` — one projection for q|k|v
    /// together, split only after the convolution (see `gdn::activate_and_split`).
    pub in_proj_qkv: QT,
    /// `[value_dim, hidden]` — the output gate's input, `z`.
    pub in_proj_z: QT,
    /// `[n_v_heads, hidden]` — `b`, one scalar per V head, becomes `beta = sigmoid(b)`.
    pub in_proj_b: QT,
    /// `[n_v_heads, hidden]` — `a`, one scalar per V head, feeds the decay gate.
    pub in_proj_a: QT,
    /// `[conv_dim, kernel]` row-major, read flat from `conv1d.weight`'s `[conv_dim, 1, kernel]`
    /// (depthwise, `groups == conv_dim`, no bias) — exactly the layout
    /// `kimi_linear::short_conv::ShortConvState::step` expects.
    pub conv: Vec<f32>,
    /// `[n_v_heads]`, learned per head.
    pub a_log: Vec<f32>,
    /// `[n_v_heads]`, learned per head.
    pub dt_bias: Vec<f32>,
    /// `[head_v_dim]` — the gated RMSNorm's weight, shared across heads. Plain-`w` convention (it's
    /// an `Qwen3_5MoeRMSNormGated`, not a `Qwen3_5MoeRMSNorm`); see `qwen38::ops`.
    pub norm: Vec<f32>,
    /// `[hidden, value_dim]`.
    pub out_proj: QT,
}

/// One full-attention (GQA) layer's weights (`model.layers.{i}.self_attn.*`).
pub struct AttnWeights {
    /// `[2 * n_heads * head_dim, hidden]` — query AND gate, interleaved per head (see
    /// `qwen38::attention::split_query_gate`).
    pub q_proj: QT,
    /// `[n_kv_heads * head_dim, hidden]`.
    pub k_proj: QT,
    pub v_proj: QT,
    /// `[hidden, n_heads * head_dim]`.
    pub o_proj: QT,
    /// `[head_dim]` each, per-head RMSNorm weights with the `(1 + w)` convention.
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
}

/// Load-time-only, and the two variants are within ~2x of each other in field count; boxing keeps
/// `Layer` small enough that a 92-entry `Vec` stays cheap to move.
pub enum Mixer {
    Gdn(Box<GdnWeights>),
    Attn(Box<AttnWeights>),
}

impl Mixer {
    pub fn is_gdn(&self) -> bool {
        matches!(self, Mixer::Gdn(_))
    }
}

/// One layer's MoE weights, minus the routed experts (which stream from disk).
pub struct MoeWeights {
    /// `[n_experts, hidden]`, kept dense `f32`: 512x8192 is 16 MB per layer at f32 and the router's
    /// output decides WHICH experts get read from disk, so quantizing it would trade a rounding error
    /// on that decision for nothing worth having. Same call `glm52::model::MoeWeights` makes.
    pub router: Vec<f32>,
    /// `[hidden]` — `shared_expert_gate.weight`, a `Linear(hidden, 1)`, so one scalar logit per
    /// token: `sigmoid` of it scales the shared expert's whole output (`qwen38::moe`).
    pub shared_gate: Vec<f32>,
    /// The shared expert's own SwiGLU MLP at `shared_expert_intermediate_size`.
    pub sh_gate: QT,
    pub sh_up: QT,
    pub sh_down: QT,
}

pub struct Layer {
    /// `[hidden]`, `(1 + w)` convention.
    pub in_ln: Vec<f32>,
    pub post_ln: Vec<f32>,
    pub mixer: Mixer,
    pub moe: MoeWeights,
}

pub struct Model {
    pub cfg: Cfg,
    pub embed: QT,
    pub lm_head: QT,
    /// `[hidden]`, `(1 + w)` convention.
    pub final_norm: Vec<f32>,
    pub layers: Vec<Layer>,
    /// Carried for the expert cache, which quantizes routed experts at this width when they aren't
    /// already MXFP4 on disk.
    pub ebits: u8,
}

#[derive(Debug)]
pub enum ModelError {
    Config(ConfigError),
    /// An error out of the shared, architecture-agnostic expert-streaming machinery
    /// (`expert_cache.rs` / `glm52::moe`), which reports through GLM's error type.
    Glm52(crate::glm52::model::ModelError),
    Safetensors(SafetensorsError),
    ShapeMismatch { name: String, expected: usize, got: usize },
    PackedFormat { name: String, source: crate::quant::PackedFormatError },
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::Config(e) => write!(f, "{e}"),
            ModelError::Glm52(e) => write!(f, "{e}"),
            ModelError::Safetensors(e) => write!(f, "{e}"),
            ModelError::ShapeMismatch { name, expected, got } => {
                write!(f, "tensor {name}: expected {expected} elements, got {got}")
            }
            ModelError::PackedFormat { name, source } => write!(f, "tensor {name}: {source}"),
        }
    }
}

impl std::error::Error for ModelError {}

impl From<ConfigError> for ModelError {
    fn from(e: ConfigError) -> Self {
        ModelError::Config(e)
    }
}

impl From<crate::glm52::model::ModelError> for ModelError {
    fn from(e: crate::glm52::model::ModelError) -> Self {
        ModelError::Glm52(e)
    }
}

impl From<SafetensorsError> for ModelError {
    fn from(e: SafetensorsError) -> Self {
        ModelError::Safetensors(e)
    }
}

/// Which [`ExpertNaming`] this checkpoint's routed experts need. MXFP4 (every real Qwen 3.8
/// checkpoint rabbit targets, see `qwen38::config::Cfg::mxfp4_experts`) reads `{...}.weight` +
/// `{...}.weight_scale` U8 pairs straight into `QTKind::MxFp4`; anything else has tensor names
/// character-for-character identical to GLM-5.2's (`model.layers.{L}.mlp.experts.{E}.{gate,up,
/// down}_proj.weight`), so that variant is reused rather than duplicated.
pub fn expert_naming(cfg: &Cfg) -> ExpertNaming {
    if cfg.mxfp4_experts { ExpertNaming::Qwen38Mxfp4 } else { ExpertNaming::Glm52 }
}

/// A minimal `glm52::config::Cfg` for the expert-streaming machinery in `expert_cache.rs` /
/// `glm52::moe::apply_single_expert`, which is architecture-agnostic but takes GLM's `Cfg` as its
/// carrier. Only three fields are ever read from it on that path (`hidden`, `moe_inter`,
/// `group_size` — see `ExpertCache::submit_batch`/`load_expert`), so the rest are zeroed rather
/// than translated: Qwen has no GLM-shaped MLA/DSA/grouped-routing parameters to translate, and
/// filling them with plausible-looking values would invite someone to trust them.
pub fn to_glm_cfg_expert(cfg: &Cfg) -> GlmCfg {
    GlmCfg {
        hidden: cfg.hidden,
        moe_inter: cfg.moe_inter,
        group_size: cfg.group_size,
        n_experts: cfg.n_experts,
        topk: cfg.topk,
        vocab: cfg.vocab,
        n_layers: cfg.n_layers,
        eps: cfg.eps,
        // Not applicable to this architecture — see this function's doc.
        n_heads: 0,
        dense_inter: 0,
        first_dense: 0,
        q_lora: 0,
        kv_lora: 0,
        qk_nope: 0,
        qk_rope: 0,
        qk_head: 0,
        v_head: 0,
        n_shared: 0,
        n_group: 1,
        topk_group: 1,
        norm_topk: cfg.norm_topk,
        stop_ids: vec![],
        index_topk: 0,
        index_nh: 0,
        index_hd: 0,
        idx_type: vec![],
        theta: cfg.theta,
        attn_scale: cfg.attn_scale,
        routed_scale: 1.0,
    }
}

fn ld(shards: &Shards, name: &str) -> Result<Vec<f32>, ModelError> {
    Ok(shards.read_f32(name, false)?)
}

/// `ld` plus a length check — for the 1-D tensors whose width is a config-derived invariant worth
/// failing loudly on (`a_log`/`dt_bias` per head, the norms, the router).
fn ld_sized(shards: &Shards, name: &str, expected: usize) -> Result<Vec<f32>, ModelError> {
    let v = ld(shards, name)?;
    if v.len() != expected {
        return Err(ModelError::ShapeMismatch { name: name.to_string(), expected, got: v.len() });
    }
    Ok(v)
}

/// Reads one 2-D weight and returns it quantized at `bits`. A `{name}.qs` sidecar (rabbit's own
/// pre-quantized format) is wrapped as-is; otherwise the raw BF16/F32 tensor is decoded and
/// requantized, which is what every real Qwen checkpoint's non-expert block needs.
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

impl Model {
    pub fn load(snap_dir: &Path, dbits: u8, ebits: u8) -> Result<Model, ModelError> {
        Self::load_multi(std::slice::from_ref(&snap_dir.to_path_buf()), dbits, ebits)
    }

    /// Same as [`Model::load`] but reads `.safetensors` shards from several directories — how the
    /// real 1.37 TB checkpoint is stored here (split across both NVMe drives so they read in
    /// parallel). `config.json` comes from `dirs[0]` only.
    pub fn load_multi(dirs: &[PathBuf], dbits: u8, ebits: u8) -> Result<Model, ModelError> {
        let cfg = Cfg::load(&dirs[0])?;
        let shards = Shards::open_multi(dirs)?;

        let d = cfg.hidden as usize;
        let head_dim = cfg.head_dim as usize;
        let n_heads = cfg.n_heads as usize;
        let n_kv_heads = cfg.n_kv_heads as usize;
        let n_v_heads = cfg.lin_value_heads as usize;
        let key_dim = (cfg.lin_key_heads * cfg.lin_key_head_dim) as usize;
        let value_dim = (cfg.lin_value_heads * cfg.lin_value_head_dim) as usize;
        let head_v_dim = cfg.lin_value_head_dim as usize;
        let conv_dim = 2 * key_dim + value_dim;
        let kernel = cfg.conv_kernel as usize;
        let n_experts = cfg.n_experts as usize;
        let moe_inter = cfg.moe_inter as usize;
        let shared_inter = cfg.shared_inter as usize;
        let gs = cfg.group_size as usize;

        // Embeddings/lm_head at 16 bits when `dbits >= 8`, same ladder K3's loader uses: they're the
        // two vocab-sized tensors and the least tolerant of coarse quantization.
        let io_bits = if dbits >= 8 { 16 } else { dbits };
        let embed = qt_load(&shards, gs, "model.embed_tokens.weight", cfg.vocab as usize, d, io_bits)?;
        let lm_head = qt_load(&shards, gs, "lm_head.weight", cfg.vocab as usize, d, io_bits)?;
        let final_norm = ld_sized(&shards, "model.norm.weight", d)?;

        let layers: Vec<Layer> = (0..cfg.n_layers as usize)
            .into_par_iter()
            .map(|i| -> Result<Layer, ModelError> {
                let p = |s: &str| format!("model.layers.{i}.{s}");
                let lp = |s: &str| format!("model.layers.{i}.linear_attn.{s}");
                let ap = |s: &str| format!("model.layers.{i}.self_attn.{s}");
                let mp = |s: &str| format!("model.layers.{i}.mlp.{s}");

                let in_ln = ld_sized(&shards, &p("input_layernorm.weight"), d)?;
                let post_ln = ld_sized(&shards, &p("post_attention_layernorm.weight"), d)?;

                let mixer = if cfg.is_linear[i] {
                    Mixer::Gdn(Box::new(GdnWeights {
                        in_proj_qkv: qt_load(&shards, gs, &lp("in_proj_qkv.weight"), conv_dim, d, dbits)?,
                        in_proj_z: qt_load(&shards, gs, &lp("in_proj_z.weight"), value_dim, d, dbits)?,
                        in_proj_b: qt_load(&shards, gs, &lp("in_proj_b.weight"), n_v_heads, d, dbits)?,
                        in_proj_a: qt_load(&shards, gs, &lp("in_proj_a.weight"), n_v_heads, d, dbits)?,
                        conv: ld_sized(&shards, &lp("conv1d.weight"), conv_dim * kernel)?,
                        a_log: ld_sized(&shards, &lp("A_log"), n_v_heads)?,
                        dt_bias: ld_sized(&shards, &lp("dt_bias"), n_v_heads)?,
                        norm: ld_sized(&shards, &lp("norm.weight"), head_v_dim)?,
                        out_proj: qt_load(&shards, gs, &lp("out_proj.weight"), d, value_dim, dbits)?,
                    }))
                } else {
                    Mixer::Attn(Box::new(AttnWeights {
                        // 2x wide: query and gate together (`attn_output_gate`)
                        q_proj: qt_load(&shards, gs, &ap("q_proj.weight"), 2 * n_heads * head_dim, d, dbits)?,
                        k_proj: qt_load(&shards, gs, &ap("k_proj.weight"), n_kv_heads * head_dim, d, dbits)?,
                        v_proj: qt_load(&shards, gs, &ap("v_proj.weight"), n_kv_heads * head_dim, d, dbits)?,
                        o_proj: qt_load(&shards, gs, &ap("o_proj.weight"), d, n_heads * head_dim, dbits)?,
                        q_norm: ld_sized(&shards, &ap("q_norm.weight"), head_dim)?,
                        k_norm: ld_sized(&shards, &ap("k_norm.weight"), head_dim)?,
                    }))
                };

                let moe = MoeWeights {
                    router: ld_sized(&shards, &mp("gate.weight"), n_experts * d)?,
                    shared_gate: ld_sized(&shards, &mp("shared_expert_gate.weight"), d)?,
                    sh_gate: qt_load(&shards, gs, &mp("shared_expert.gate_proj.weight"), shared_inter, d, dbits)?,
                    sh_up: qt_load(&shards, gs, &mp("shared_expert.up_proj.weight"), shared_inter, d, dbits)?,
                    sh_down: qt_load(&shards, gs, &mp("shared_expert.down_proj.weight"), d, shared_inter, dbits)?,
                };

                Ok(Layer { in_ln, post_ln, mixer, moe })
            })
            .collect::<Result<Vec<_>, _>>()?;

        // `moe_inter` is only ever needed to size the streamed experts, so it isn't read here; the
        // assertion keeps a config whose experts couldn't be addressed from loading silently.
        debug_assert!(moe_inter > 0, "moe_intermediate_size must be set for a routed-MoE checkpoint");

        Ok(Model { cfg, embed, lm_head, final_norm, layers, ebits })
    }

    /// One token's embedding row, dequantized — `generate`'s entry point into a forward pass.
    pub fn embed_row(&self, tok: usize) -> Vec<f32> {
        self.embed.row_f32(tok)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;

    pub(crate) struct TempDir(pub(crate) PathBuf);
    impl TempDir {
        pub(crate) fn new(name: &str) -> TempDir {
            let dir = std::env::temp_dir().join(format!("{name}_{}", std::process::id()));
            fs::create_dir_all(&dir).unwrap();
            TempDir(dir)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).ok();
        }
    }

    /// A tiny Qwen-shaped config: 4 layers so layer 3 is the one full-attention layer under the real
    /// `full_attention_interval: 4`, and every width small enough to write by hand.
    const HIDDEN: usize = 8;
    const LAYERS: usize = 4;
    const N_HEADS: usize = 4;
    const N_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 8;
    const K_HEADS: usize = 1;
    const V_HEADS: usize = 2;
    const K_HEAD_DIM: usize = 4;
    const V_HEAD_DIM: usize = 4;
    const KERNEL: usize = 4;
    const N_EXPERTS: usize = 4;
    const MOE_INTER: usize = 6;
    const SHARED_INTER: usize = 6;
    const VOCAB: usize = 10;

    fn tiny_config() -> serde_json::Value {
        let layer_types: Vec<&str> =
            (0..LAYERS).map(|i| if (i + 1) % 4 == 0 { "full_attention" } else { "linear_attention" }).collect();
        json!({
            "model_type": "qwen3_5_moe_text",
            "hidden_size": HIDDEN,
            "num_hidden_layers": LAYERS,
            "num_attention_heads": N_HEADS,
            "num_key_value_heads": N_KV_HEADS,
            "head_dim": HEAD_DIM,
            "attn_output_gate": true,
            "output_gate_type": "swish",
            "partial_rotary_factor": 0.25,
            "layer_types": layer_types,
            "full_attention_interval": 4,
            "linear_num_key_heads": K_HEADS,
            "linear_num_value_heads": V_HEADS,
            "linear_key_head_dim": K_HEAD_DIM,
            "linear_value_head_dim": V_HEAD_DIM,
            "linear_conv_kernel_dim": KERNEL,
            "num_experts": N_EXPERTS,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": MOE_INTER,
            "shared_expert_intermediate_size": SHARED_INTER,
            "vocab_size": VOCAB,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25},
            "eos_token_id": 7,
            "mtp_num_hidden_layers": 1
        })
    }

    /// Writes a synthetic checkpoint whose every tensor is filled with a DISTINCT constant, so a test
    /// can tell not just "it loaded" but "this tensor landed in that field" — the failure mode that
    /// matters when a layer has four same-shaped input projections (`in_proj_a` vs `in_proj_b`) and
    /// three same-shaped MLP matrices.
    /// Appends one tensor filled with `value` plus a small deterministic per-element ripple.
    ///
    /// The ripple matters: with truly constant matrices every activation vector comes out with all
    /// components EQUAL, and RMSNorm maps any such vector to the same normalized value regardless of
    /// magnitude — so a forward pass would be blind to state changes, and a transposed matmul would
    /// be indistinguishable from a correct one. Element `[0][0]` is exactly `value`, which is what
    /// the field-placement assertions below read, so per-tensor identification still works.
    ///
    /// A free fn rather than a closure so the fixture can also write a row-varying tensor (the
    /// embeddings) against the same `data` buffer.
    fn push_const(
        header: &mut serde_json::Map<String, serde_json::Value>,
        data: &mut Vec<u8>,
        name: &str,
        shape: Vec<usize>,
        value: f32,
    ) {
        let n: usize = shape.iter().product();
        let cols = *shape.last().unwrap_or(&1);
        let start = data.len() as u64;
        for i in 0..n {
            let (row, col) = (i / cols.max(1), i % cols.max(1));
            let ripple = ((row * 7 + col * 3) % 5) as f32 * 0.05;
            data.extend_from_slice(&(value + ripple).to_le_bytes());
        }
        let end = data.len() as u64;
        header.insert(name.to_string(), json!({"dtype": "F32", "shape": shape, "data_offsets": [start, end]}));
    }

    pub(crate) fn write_tiny_checkpoint(dir: &Path) {
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_string(), json!({"format": "rabbit-test"}));
        let mut data: Vec<u8> = Vec::new();
        let mut tag = 0f32;
        let mut next = || {
            tag += 1.0;
            tag
        };

        // embeddings: row `t` filled with `t + 1` so `embed_row` is checkable per token
        {
            let start = data.len() as u64;
            for t in 0..VOCAB {
                for _ in 0..HIDDEN {
                    data.extend_from_slice(&((t + 1) as f32).to_le_bytes());
                }
            }
            let end = data.len() as u64;
            header.insert(
                "model.embed_tokens.weight".to_string(),
                json!({"dtype": "F32", "shape": [VOCAB, HIDDEN], "data_offsets": [start, end]}),
            );
        }
        push_const(&mut header, &mut data, "lm_head.weight", vec![VOCAB, HIDDEN], next());
        push_const(&mut header, &mut data, "model.norm.weight", vec![HIDDEN], 0.25);

        let key_dim = K_HEADS * K_HEAD_DIM;
        let value_dim = V_HEADS * V_HEAD_DIM;
        let conv_dim = 2 * key_dim + value_dim;
        for i in 0..LAYERS {
            push_const(&mut header, &mut data, &format!("model.layers.{i}.input_layernorm.weight"), vec![HIDDEN], 0.5 + i as f32);
            push_const(&mut header, &mut data, &format!("model.layers.{i}.post_attention_layernorm.weight"), vec![HIDDEN], 1.5 + i as f32);

            if (i + 1) % 4 == 0 {
                let ap = |s: &str| format!("model.layers.{i}.self_attn.{s}");
                push_const(&mut header, &mut data, &ap("q_proj.weight"), vec![2 * N_HEADS * HEAD_DIM, HIDDEN], next());
                push_const(&mut header, &mut data, &ap("k_proj.weight"), vec![N_KV_HEADS * HEAD_DIM, HIDDEN], next());
                push_const(&mut header, &mut data, &ap("v_proj.weight"), vec![N_KV_HEADS * HEAD_DIM, HIDDEN], next());
                push_const(&mut header, &mut data, &ap("o_proj.weight"), vec![HIDDEN, N_HEADS * HEAD_DIM], next());
                push_const(&mut header, &mut data, &ap("q_norm.weight"), vec![HEAD_DIM], next());
                push_const(&mut header, &mut data, &ap("k_norm.weight"), vec![HEAD_DIM], next());
            } else {
                let lp = |s: &str| format!("model.layers.{i}.linear_attn.{s}");
                push_const(&mut header, &mut data, &lp("in_proj_qkv.weight"), vec![conv_dim, HIDDEN], next());
                push_const(&mut header, &mut data, &lp("in_proj_z.weight"), vec![value_dim, HIDDEN], next());
                push_const(&mut header, &mut data, &lp("in_proj_b.weight"), vec![V_HEADS, HIDDEN], next());
                push_const(&mut header, &mut data, &lp("in_proj_a.weight"), vec![V_HEADS, HIDDEN], next());
                // conv1d.weight ships as [conv_dim, 1, kernel]; the loader reads it flat
                push_const(&mut header, &mut data, &lp("conv1d.weight"), vec![conv_dim, 1, KERNEL], next());
                push_const(&mut header, &mut data, &lp("A_log"), vec![V_HEADS], next());
                push_const(&mut header, &mut data, &lp("dt_bias"), vec![V_HEADS], next());
                push_const(&mut header, &mut data, &lp("norm.weight"), vec![V_HEAD_DIM], next());
                push_const(&mut header, &mut data, &lp("out_proj.weight"), vec![HIDDEN, value_dim], next());
            }

            let mp = |s: &str| format!("model.layers.{i}.mlp.{s}");
            push_const(&mut header, &mut data, &mp("gate.weight"), vec![N_EXPERTS, HIDDEN], next());
            push_const(&mut header, &mut data, &mp("shared_expert_gate.weight"), vec![1, HIDDEN], next());
            push_const(&mut header, &mut data, &mp("shared_expert.gate_proj.weight"), vec![SHARED_INTER, HIDDEN], next());
            push_const(&mut header, &mut data, &mp("shared_expert.up_proj.weight"), vec![SHARED_INTER, HIDDEN], next());
            push_const(&mut header, &mut data, &mp("shared_expert.down_proj.weight"), vec![HIDDEN, SHARED_INTER], next());

            // routed experts stay on disk; written here so the fixture matches a real checkpoint's
            // shape (and so a future expert-streaming test can use the same directory)
            for e in 0..N_EXPERTS {
                let ep = |s: &str| format!("model.layers.{i}.mlp.experts.{e}.{s}");
                push_const(&mut header, &mut data, &ep("gate_proj.weight"), vec![MOE_INTER, HIDDEN], next());
                push_const(&mut header, &mut data, &ep("up_proj.weight"), vec![MOE_INTER, HIDDEN], next());
                push_const(&mut header, &mut data, &ep("down_proj.weight"), vec![HIDDEN, MOE_INTER], next());
            }
        }
        // an MTP block, which the loader must ignore entirely
        push_const(&mut header, &mut data, "mtp.fc.weight", vec![HIDDEN, 2 * HIDDEN], 99.0);
        push_const(&mut header, &mut data, "mtp.norm.weight", vec![HIDDEN], 99.0);

        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_bytes);
        out.extend_from_slice(&data);
        fs::write(dir.join("model.safetensors"), out).unwrap();
        fs::write(dir.join("config.json"), serde_json::to_vec(&tiny_config()).unwrap()).unwrap();
    }

    /// `dbits = 16` keeps every weight in f32 (`QT::alloc`'s `>= 16` tier), so the constants written
    /// above come back exactly and field-placement assertions can use `assert_eq!`.
    fn load_tiny(dir: &Path) -> Model {
        Model::load(dir, 16, 4).expect("the tiny checkpoint must load")
    }

    #[test]
    fn loads_every_layer_with_the_right_mixer_kind() {
        let dir = TempDir::new("rabbit_test_qwen38_model_mixers");
        write_tiny_checkpoint(&dir.0);
        let m = load_tiny(&dir.0);

        assert_eq!(m.layers.len(), LAYERS);
        assert_eq!(m.cfg.n_layers as usize, LAYERS);
        let kinds: Vec<bool> = m.layers.iter().map(|l| l.mixer.is_gdn()).collect();
        assert_eq!(kinds, vec![true, true, true, false], "layers 0-2 are GDN, layer 3 is full attention");
    }

    #[test]
    fn every_tensor_lands_in_its_own_field_with_the_right_shape() {
        let dir = TempDir::new("rabbit_test_qwen38_model_fields");
        write_tiny_checkpoint(&dir.0);
        let m = load_tiny(&dir.0);

        let key_dim = K_HEADS * K_HEAD_DIM;
        let value_dim = V_HEADS * V_HEAD_DIM;
        let conv_dim = 2 * key_dim + value_dim;

        // GDN layer 0
        let Mixer::Gdn(g) = &m.layers[0].mixer else { panic!("layer 0 must be GDN") };
        assert_eq!((g.in_proj_qkv.rows, g.in_proj_qkv.cols), (conv_dim, HIDDEN));
        assert_eq!((g.in_proj_z.rows, g.in_proj_z.cols), (value_dim, HIDDEN));
        assert_eq!((g.in_proj_b.rows, g.in_proj_b.cols), (V_HEADS, HIDDEN));
        assert_eq!((g.in_proj_a.rows, g.in_proj_a.cols), (V_HEADS, HIDDEN));
        assert_eq!((g.out_proj.rows, g.out_proj.cols), (HIDDEN, value_dim));
        assert_eq!(g.conv.len(), conv_dim * KERNEL, "conv1d.weight is [conv_dim, 1, kernel] read flat");
        assert_eq!(g.a_log.len(), V_HEADS);
        assert_eq!(g.dt_bias.len(), V_HEADS);
        assert_eq!(g.norm.len(), V_HEAD_DIM, "the gated RMSNorm weight is head_v_dim wide, not value_dim");

        // The four same-shaped-ish input projections must not be swapped: each fixture tensor holds a
        // distinct constant, and `b` was written BEFORE `a` on purpose (the reference declares them
        // in the opposite order, so a copy-paste swap is the likely mistake).
        let val = |t: &QT| t.row_f32(0)[0];
        assert_ne!(val(&g.in_proj_a), val(&g.in_proj_b), "in_proj_a and in_proj_b must be distinct tensors");
        assert!(val(&g.in_proj_b) < val(&g.in_proj_a), "in_proj_b is written first in the fixture");
        assert_ne!(g.a_log[0], g.dt_bias[0], "A_log and dt_bias must not be swapped");
        assert!(g.a_log[0] < g.dt_bias[0], "A_log is written first in the fixture");

        // attention layer 3
        let Mixer::Attn(a) = &m.layers[3].mixer else { panic!("layer 3 must be full attention") };
        assert_eq!((a.q_proj.rows, a.q_proj.cols), (2 * N_HEADS * HEAD_DIM, HIDDEN), "q_proj carries query AND gate");
        assert_eq!((a.k_proj.rows, a.k_proj.cols), (N_KV_HEADS * HEAD_DIM, HIDDEN));
        assert_eq!((a.v_proj.rows, a.v_proj.cols), (N_KV_HEADS * HEAD_DIM, HIDDEN));
        assert_eq!((a.o_proj.rows, a.o_proj.cols), (HIDDEN, N_HEADS * HEAD_DIM));
        assert_eq!(a.q_norm.len(), HEAD_DIM);
        assert_eq!(a.k_norm.len(), HEAD_DIM);
        assert_ne!(a.q_norm[0], a.k_norm[0], "q_norm and k_norm must not be swapped");
        assert_ne!(val(&a.k_proj), val(&a.v_proj), "k_proj and v_proj must not be swapped");

        // MoE, on both layer kinds (every layer has one)
        for (i, layer) in m.layers.iter().enumerate() {
            assert_eq!(layer.moe.router.len(), N_EXPERTS * HIDDEN, "layer {i} router");
            assert_eq!(layer.moe.shared_gate.len(), HIDDEN, "layer {i} shared gate is Linear(hidden, 1)");
            assert_eq!((layer.moe.sh_gate.rows, layer.moe.sh_gate.cols), (SHARED_INTER, HIDDEN));
            assert_eq!((layer.moe.sh_down.rows, layer.moe.sh_down.cols), (HIDDEN, SHARED_INTER));
            assert_ne!(val(&layer.moe.sh_gate), val(&layer.moe.sh_up), "layer {i}: gate_proj vs up_proj");
        }

        // per-layer norms are distinct and in the right slots
        assert_eq!(m.layers[0].in_ln[0], 0.5);
        assert_eq!(m.layers[0].post_ln[0], 1.5);
        assert_eq!(m.layers[2].in_ln[0], 2.5);
        assert_eq!(m.final_norm[0], 0.25);
        assert_eq!(m.final_norm.len(), HIDDEN);
    }

    #[test]
    fn embed_row_returns_the_requested_token_row() {
        let dir = TempDir::new("rabbit_test_qwen38_model_embed");
        write_tiny_checkpoint(&dir.0);
        let m = load_tiny(&dir.0);

        for t in [0usize, 3, VOCAB - 1] {
            let row = m.embed_row(t);
            assert_eq!(row.len(), HIDDEN);
            assert!(row.iter().all(|&x| (x - (t + 1) as f32).abs() < 1e-6), "token {t}: {row:?}");
        }
    }

    /// The MTP block is a full extra MoE layer in the real checkpoint; loading must ignore it rather
    /// than trip over it or count it as a 93rd layer.
    #[test]
    fn skips_the_mtp_block_entirely() {
        let dir = TempDir::new("rabbit_test_qwen38_model_mtp");
        write_tiny_checkpoint(&dir.0);
        let m = load_tiny(&dir.0);
        assert_eq!(m.layers.len(), LAYERS, "mtp.* must not become a layer");
        assert_eq!(m.cfg.mtp_layers, 1, "...but the config still records that it exists");
        assert!(m.layers.iter().all(|l| l.in_ln[0] != 99.0), "no MTP tensor may leak into a layer");
    }

    #[test]
    fn a_missing_tensor_names_itself_in_the_error() {
        let dir = TempDir::new("rabbit_test_qwen38_model_missing");
        write_tiny_checkpoint(&dir.0);
        // drop `lm_head.weight` by rewriting the checkpoint without it
        let raw = fs::read(dir.0.join("model.safetensors")).unwrap();
        let hlen = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
        let mut header: serde_json::Value = serde_json::from_slice(&raw[8..8 + hlen]).unwrap();
        header.as_object_mut().unwrap().remove("lm_head.weight");
        let hb = serde_json::to_vec(&header).unwrap();
        let mut out = (hb.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&hb);
        out.extend_from_slice(&raw[8 + hlen..]);
        fs::write(dir.0.join("model.safetensors"), out).unwrap();

        let err = Model::load(&dir.0, 16, 4).map(|_| ()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("lm_head.weight"), "the error must name the missing tensor: {msg}");
    }

    /// A 1-D tensor whose width contradicts the config must fail with its own name, not be silently
    /// accepted at the wrong length (which would corrupt a later per-head loop).
    #[test]
    fn a_wrong_width_one_dimensional_tensor_is_rejected() {
        let dir = TempDir::new("rabbit_test_qwen38_model_badwidth");
        write_tiny_checkpoint(&dir.0);
        let raw = fs::read(dir.0.join("model.safetensors")).unwrap();
        let hlen = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
        let mut header: serde_json::Value = serde_json::from_slice(&raw[8..8 + hlen]).unwrap();
        // claim A_log covers one head instead of V_HEADS (offsets untouched: only the shape lies)
        let entry = header.get_mut("model.layers.0.linear_attn.A_log").unwrap();
        let offsets = entry["data_offsets"].clone();
        let start = offsets[0].as_u64().unwrap();
        *entry = json!({"dtype": "F32", "shape": [1], "data_offsets": [start, start + 4]});
        let hb = serde_json::to_vec(&header).unwrap();
        let mut out = (hb.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&hb);
        out.extend_from_slice(&raw[8 + hlen..]);
        fs::write(dir.0.join("model.safetensors"), out).unwrap();

        let err = Model::load(&dir.0, 16, 4).map(|_| ()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("A_log"), "the error must name the tensor: {msg}");
        assert!(msg.contains('2') && msg.contains('1'), "and both lengths: {msg}");
    }

    #[test]
    fn expert_naming_follows_the_checkpoints_quantization() {
        let dir = TempDir::new("rabbit_test_qwen38_model_naming");
        write_tiny_checkpoint(&dir.0);
        let m = load_tiny(&dir.0);
        assert!(!m.cfg.mxfp4_experts, "the fixture ships plain F32 experts");
        assert_eq!(expert_naming(&m.cfg), ExpertNaming::Glm52, "non-MXFP4 Qwen names are GLM's exactly");

        let mut mxfp4 = m.cfg.clone();
        mxfp4.mxfp4_experts = true;
        assert_eq!(expert_naming(&mxfp4), ExpertNaming::Qwen38Mxfp4);
    }
}
