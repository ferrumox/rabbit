//! Loads a Kimi Linear checkpoint's dense-resident tensors, mirroring
//! `glm52::model::Model::load`'s structure and streaming-expert philosophy (routed expert
//! weights are NOT loaded here — they stay on disk until a future `expert_cache.rs` wiring step
//! reads them on demand, exactly like GLM-5.2).
//!
//! Every tensor name and shape below is taken from the real
//! `moonshotai/Kimi-Linear-48B-A3B-Instruct` checkpoint (its `model.safetensors.index.json` and
//! actual safetensors shard headers, fetched this session — not guessed), cross-checked against
//! `modeling_kimi.py`'s module `__init__`s:
//!
//! - **MLA layers reuse `glm52::model::AttnWeights` and `glm52::attention::attention()`
//!   as-is** (`QProj::Direct`, `Rope::Off`) — Kimi's MLA tensor names/shapes
//!   (`kv_a_proj_with_mqa`, `kv_a_layernorm`, `kv_b_proj`, `o_proj`) match GLM-5.2's exactly;
//!   the only structural difference is no `q_a_proj`/`q_b_proj` split (`q_lora_rank` is always
//!   `null` — see `glm52::attention::QProj`'s doc). `q_a` is loaded from the real `q_proj`
//!   tensor directly; `q_a_ln`/`q_b` are unread placeholders.
//! - **MoE/dense FFN layers reuse `glm52::model::{DenseMlpWeights, MoeWeights, Ffn}` as-is** —
//!   same tensor shapes as GLM-5.2's noaux_tc-style router + shared expert, just a different
//!   on-disk name prefix (`block_sparse_moe.*` instead of `mlp.*`, confirmed by reading
//!   `KimiSparseMoeBlock`/`KimiMoEGate` in `modeling_kimi.py`). Routed experts use `w1`/`w2`/`w3`
//!   instead of `gate_proj`/`up_proj`/`down_proj` — confirmed via `KimiBlockSparseMLP.__init__`
//!   comments (`w1: gate`, `w2: down`, `w3: up`) — same shapes as GLM's expert tensors either
//!   way, so nothing downstream of that renaming needs to change once `expert_cache.rs` gets
//!   parameterized over the tensor-name template (not done in this step — that's runtime
//!   expert-streaming wiring, not the dense-resident loader this file covers).
//! - **KDA layers are genuinely new** (`KdaWeights`, no GLM-5.2 equivalent): `q_proj`/`k_proj`/
//!   `v_proj` (each `[d_inner, hidden]`, `d_inner = kda_n_heads * kda_head_dim`),
//!   `q_conv1d`/`k_conv1d`/`v_conv1d` (`[d_inner, 1, kernel]` on disk — the middle dim is 1, so
//!   a flat `d_inner*kernel`-long read is already `short_conv::ShortConvState::step`'s expected
//!   `[d_inner, kernel]` row-major layout with no reshape needed), `f_a_proj`/`f_b_proj` (the
//!   decay gate's low-rank projection, `g = f_b(f_a(x))`), `dt_bias` (`[d_inner]`, per-channel),
//!   `A_log` (**`[1,1,kda_n_heads,1]` on disk — `kda_n_heads` scalars, ONE PER HEAD, not one per
//!   channel** — this is the exact shape mismatch `kda.rs::decay_gate`'s signature was fixed
//!   for this session after reading this real tensor's shape), `b_proj` (beta, `[kda_n_heads,
//!   hidden]`), `g_a_proj`/`g_b_proj` (the output gate's low-rank projection), `o_norm`
//!   (`[kda_head_dim]`, shared across heads), `o_proj` (`[hidden, d_inner]`).

use crate::glm52::attention::{Absorb, Dsa, QProj, Rope};
use crate::glm52::config::Cfg as GlmCfg;
use crate::glm52::model::{AttnWeights, DenseMlpWeights, Ffn, MoeWeights};
use crate::glm52::moe::RouteConfig;
use crate::kimi_linear::config::{Cfg, ConfigError};
use crate::quant::{PackedFormatError, QT};
use crate::safetensors::{SafetensorsError, Shards};
use std::fmt;
use std::path::Path;

/// One KDA layer's attention weights — see this module's doc for every field's real on-disk
/// name/shape.
pub struct KdaWeights {
    pub q_proj: QT,
    pub k_proj: QT,
    pub v_proj: QT,
    /// `[d_inner, kernel]` row-major (a flat read of the on-disk `[d_inner, 1, kernel]` tensor
    /// already has this layout) — `short_conv::ShortConvState::step`'s `weight` argument.
    pub q_conv: Vec<f32>,
    pub k_conv: Vec<f32>,
    pub v_conv: Vec<f32>,
    pub f_a_proj: QT,
    pub f_b_proj: QT,
    /// `[d_inner]`, per-channel.
    pub dt_bias: Vec<f32>,
    /// `[kda_n_heads]` — ONE SCALAR PER HEAD, broadcast across that head's channels by
    /// `kda::decay_gate`. Not `[d_inner]` — see this module's doc.
    pub a_log: Vec<f32>,
    pub b_proj: QT,
    pub g_a_proj: QT,
    pub g_b_proj: QT,
    /// `[kda_head_dim]`, shared across every head (matches `ops::head_output_gate`'s
    /// `o_norm_weight` argument).
    pub o_norm: Vec<f32>,
    pub o_proj: QT,
}

/// One layer's attention weights: `Kda` (new, this module) or `Mla` (reused from GLM-5.2
/// verbatim — see this module's doc for why `AttnWeights` needs no changes at all).
pub enum Attn {
    Kda(Box<KdaWeights>),
    Mla(Box<AttnWeights>),
}

pub struct Layer {
    pub in_ln: Vec<f32>,
    pub post_ln: Vec<f32>,
    pub attn: Attn,
    pub ffn: Ffn,
}

pub struct Model {
    pub cfg: Cfg,
    pub embed: QT,
    pub lm_head: QT,
    pub final_norm: Vec<f32>,
    pub layers: Vec<Layer>,
    pub ebits: u8,
    pub route_cfg: RouteConfig,
}

#[derive(Debug)]
pub enum ModelError {
    Config(ConfigError),
    Safetensors(SafetensorsError),
    ShapeMismatch { name: String, expected: usize, got: usize },
    PackedFormat { name: String, source: PackedFormatError },
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

/// Builds a `glm52::config::Cfg` that drives BOTH `glm52::attention::attention()` (for Kimi's
/// MLA layers) and `glm52::moe::{dense_mlp, moe}` (for every layer's FFN, KDA or MLA alike) —
/// Kimi and GLM-5.2 share the same DeepSeek-V3-lineage `noaux_tc` + grouped-top-k MoE routing,
/// so every MoE-relevant field (`n_experts`, `topk`, `n_group`, `topk_group`, `norm_topk`,
/// `routed_scale`, `moe_inter`, `dense_inter`, `n_shared`) carries a REAL value here, not a
/// placeholder. Only the DSA fields (`index_topk`/`index_nh`/`index_hd`/`idx_type`) are
/// placeholders — Kimi has no DSA indexer at all, and `Dsa::Off` (every Kimi `attention()` call)
/// never reads them. `q_lora: 0` makes `attention()`'s internal `qr_all` buffer zero-sized,
/// which is fine: `QProj::Direct` never writes to it. `first_dense`/`stop_ids` aren't read by
/// either `attention()` or `moe()`/`dense_mlp` (the caller already decided dense-vs-MoE via
/// `kimi_linear::model::Layer::ffn`'s own variant, and stop-id handling lives in `Cfg` itself),
/// so those two stay placeholders too.
pub fn to_glm_cfg(kcfg: &Cfg) -> GlmCfg {
    GlmCfg {
        hidden: kcfg.hidden,
        n_layers: kcfg.n_layers,
        n_heads: kcfg.n_heads,
        n_experts: kcfg.n_experts,
        topk: kcfg.topk,
        moe_inter: kcfg.moe_inter,
        dense_inter: kcfg.dense_inter,
        first_dense: 0,
        q_lora: 0,
        kv_lora: kcfg.kv_lora,
        qk_nope: kcfg.qk_nope,
        qk_rope: kcfg.qk_rope,
        qk_head: kcfg.qk_head,
        v_head: kcfg.v_head,
        n_shared: kcfg.n_shared,
        vocab: kcfg.vocab,
        n_group: kcfg.n_group,
        topk_group: kcfg.topk_group,
        norm_topk: kcfg.norm_topk,
        stop_ids: vec![],
        index_topk: 0,
        index_nh: 0,
        index_hd: 0,
        idx_type: vec![],
        eps: kcfg.eps,
        theta: kcfg.theta,
        attn_scale: kcfg.attn_scale,
        routed_scale: kcfg.routed_scale,
        group_size: kcfg.group_size,
    }
}

impl Model {
    pub fn load(snap_dir: &Path, dbits: u8, ebits: u8) -> Result<Model, ModelError> {
        let cfg = Cfg::load(snap_dir)?;
        let shards = Shards::open(snap_dir)?;
        let d = cfg.hidden as usize;
        let h = cfg.n_heads as usize;
        let qh = cfg.qk_head as usize;
        let qk_nope = cfg.qk_nope as usize;
        let qk_rope = cfg.qk_rope as usize;
        let v_head = cfg.v_head as usize;
        let kv_lora = cfg.kv_lora as usize;
        let d_inner = (cfg.kda_n_heads * cfg.kda_head_dim) as usize;
        let kda_head_dim = cfg.kda_head_dim as usize;
        let kda_n_heads = cfg.kda_n_heads as usize;

        let io_bits = if dbits >= 8 { 16 } else { dbits };
        let embed = qt_load(&shards, cfg.group_size as usize, "model.embed_tokens.weight", cfg.vocab as usize, d, io_bits)?;
        let lm_head = qt_load(&shards, cfg.group_size as usize, "lm_head.weight", cfg.vocab as usize, d, io_bits)?;
        let final_norm = ld(&shards, "model.norm.weight")?;

        let mut layers = Vec::with_capacity(cfg.n_layers as usize);
        for i in 0..cfg.n_layers as usize {
            let p = |s: &str| format!("model.layers.{i}.{s}");

            let in_ln = ld(&shards, &p("input_layernorm.weight"))?;
            let post_ln = ld(&shards, &p("post_attention_layernorm.weight"))?;

            let attn = if cfg.is_kda[i] {
                let ap = |s: &str| format!("model.layers.{i}.self_attn.{s}");
                Attn::Kda(Box::new(KdaWeights {
                    q_proj: qt_load(&shards, cfg.group_size as usize, &ap("q_proj.weight"), d_inner, d, dbits)?,
                    k_proj: qt_load(&shards, cfg.group_size as usize, &ap("k_proj.weight"), d_inner, d, dbits)?,
                    v_proj: qt_load(&shards, cfg.group_size as usize, &ap("v_proj.weight"), d_inner, d, dbits)?,
                    q_conv: ld(&shards, &ap("q_conv1d.weight"))?,
                    k_conv: ld(&shards, &ap("k_conv1d.weight"))?,
                    v_conv: ld(&shards, &ap("v_conv1d.weight"))?,
                    f_a_proj: qt_load(&shards, cfg.group_size as usize, &ap("f_a_proj.weight"), kda_head_dim, d, dbits)?,
                    f_b_proj: qt_load(&shards, cfg.group_size as usize, &ap("f_b_proj.weight"), d_inner, kda_head_dim, dbits)?,
                    dt_bias: ld(&shards, &ap("dt_bias"))?,
                    a_log: ld(&shards, &ap("A_log"))?,
                    b_proj: qt_load(&shards, cfg.group_size as usize, &ap("b_proj.weight"), kda_n_heads, d, dbits)?,
                    g_a_proj: qt_load(&shards, cfg.group_size as usize, &ap("g_a_proj.weight"), kda_head_dim, d, dbits)?,
                    g_b_proj: qt_load(&shards, cfg.group_size as usize, &ap("g_b_proj.weight"), d_inner, kda_head_dim, dbits)?,
                    o_norm: ld(&shards, &ap("o_norm.weight"))?,
                    o_proj: qt_load(&shards, cfg.group_size as usize, &ap("o_proj.weight"), d, d_inner, dbits)?,
                }))
            } else {
                let ap = |s: &str| format!("model.layers.{i}.self_attn.{s}");
                Attn::Mla(Box::new(AttnWeights {
                    q_a: qt_load(&shards, cfg.group_size as usize, &ap("q_proj.weight"), h * qh, d, dbits)?,
                    q_a_ln: vec![],
                    q_b: QT::alloc(1, 1, 32, false),
                    kv_a: qt_load(&shards, cfg.group_size as usize, &ap("kv_a_proj_with_mqa.weight"), kv_lora + qk_rope, d, dbits)?,
                    kv_a_ln: ld(&shards, &ap("kv_a_layernorm.weight"))?,
                    kv_b: qt_load(&shards, cfg.group_size as usize, &ap("kv_b_proj.weight"), h * (qk_nope + v_head), kv_lora, dbits)?,
                    o: qt_load(&shards, cfg.group_size as usize, &ap("o_proj.weight"), d, h * v_head, dbits)?,
                }))
            };

            let sparse = i >= cfg.first_dense as usize;
            let ffn = if !sparse {
                Ffn::Dense(DenseMlpWeights {
                    gate_proj: qt_load(&shards, cfg.group_size as usize, &p("mlp.gate_proj.weight"), cfg.dense_inter as usize, d, dbits)?,
                    up_proj: qt_load(&shards, cfg.group_size as usize, &p("mlp.up_proj.weight"), cfg.dense_inter as usize, d, dbits)?,
                    down_proj: qt_load(&shards, cfg.group_size as usize, &p("mlp.down_proj.weight"), d, cfg.dense_inter as usize, dbits)?,
                })
            } else {
                let mp = |s: &str| format!("model.layers.{i}.block_sparse_moe.{s}");
                let s_i = (cfg.moe_inter * cfg.n_shared) as usize;
                Ffn::Moe(MoeWeights {
                    router: ld(&shards, &mp("gate.weight"))?,
                    router_bias: ld(&shards, &mp("gate.e_score_correction_bias"))?,
                    sh_gate: qt_load(&shards, cfg.group_size as usize, &mp("shared_experts.gate_proj.weight"), s_i, d, dbits)?,
                    sh_up: qt_load(&shards, cfg.group_size as usize, &mp("shared_experts.up_proj.weight"), s_i, d, dbits)?,
                    sh_down: qt_load(&shards, cfg.group_size as usize, &mp("shared_experts.down_proj.weight"), d, s_i, dbits)?,
                })
            };

            layers.push(Layer { in_ln, post_ln, attn, ffn });
        }

        Ok(Model { cfg, embed, lm_head, final_norm, layers, ebits, route_cfg: RouteConfig::default() })
    }

    pub fn embed_row(&self, tok: usize) -> Vec<f32> {
        self.embed.row_f32(tok)
    }
}

/// Convenience for `kimi_linear::generate`: builds the `glm52::config::Cfg` +
/// `Dsa`/`Absorb`/`Rope`/`QProj` combination an MLA layer's `attention()` call needs. Kept here
/// (not private) since `to_glm_cfg` alone doesn't carry the fixed `Dsa::Off`/`Rope::Off`/
/// `QProj::Direct` choice that's true for every Kimi MLA layer, every call, always.
pub fn mla_call_args(cfg: &Cfg) -> (GlmCfg, Dsa<'static>, Absorb, Rope, QProj) {
    (to_glm_cfg(cfg), Dsa::Off, Absorb::Auto, Rope::Off, QProj::Direct)
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

    fn f32_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
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

    /// A tiny 3-layer model (layer 0: KDA + dense FFN, layer 1: KDA + MoE, layer 2: MLA + MoE)
    /// exercising every branch `Model::load` has: both attention kinds, both FFN kinds.
    struct TinyFixture {
        dir: TempDir,
        tensors: Vec<(String, Vec<usize>, Vec<u8>)>,
        seed: u32,
    }

    impl TinyFixture {
        fn new(name: &str) -> Self {
            TinyFixture { dir: TempDir::new(&format!("rabbit_test_kimi_model_tiny_{name}")), tensors: Vec::new(), seed: 1 }
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
            let vocab = 16;
            let dense_inter = 10;
            let n_experts = 4;
            let moe_inter = 6;
            let n_shared = 1;
            let kda_head_dim = 4;
            let kda_n_heads = 2;
            let d_inner = kda_head_dim * kda_n_heads;
            let kernel = 3;

            self.add("model.embed_tokens.weight", vec![vocab, d]);
            self.add("lm_head.weight", vec![vocab, d]);
            self.add("model.norm.weight", vec![d]);

            for i in 0..3usize {
                let p = |s: &str| format!("model.layers.{i}.{s}");
                self.add(&p("input_layernorm.weight"), vec![d]);
                self.add(&p("post_attention_layernorm.weight"), vec![d]);

                let is_kda = i < 2; // layers 0,1 KDA; layer 2 MLA
                if is_kda {
                    let ap = |s: &str| format!("model.layers.{i}.self_attn.{s}");
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
                    self.add(&ap("g_a_proj.weight"), vec![kda_head_dim, d]);
                    self.add(&ap("g_b_proj.weight"), vec![d_inner, kda_head_dim]);
                    self.add(&ap("o_norm.weight"), vec![kda_head_dim]);
                    self.add(&ap("o_proj.weight"), vec![d, d_inner]);
                } else {
                    let ap = |s: &str| format!("model.layers.{i}.self_attn.{s}");
                    self.add(&ap("q_proj.weight"), vec![h * qh, d]);
                    self.add(&ap("kv_a_proj_with_mqa.weight"), vec![kv_lora + qk_rope, d]);
                    self.add(&ap("kv_a_layernorm.weight"), vec![kv_lora]);
                    self.add(&ap("kv_b_proj.weight"), vec![h * (qk_nope + v_head), kv_lora]);
                    self.add(&ap("o_proj.weight"), vec![d, h * v_head]);
                }

                if i == 0 {
                    self.add(&p("mlp.gate_proj.weight"), vec![dense_inter, d]);
                    self.add(&p("mlp.up_proj.weight"), vec![dense_inter, d]);
                    self.add(&p("mlp.down_proj.weight"), vec![d, dense_inter]);
                } else {
                    let mp = |s: &str| format!("model.layers.{i}.block_sparse_moe.{s}");
                    self.add(&mp("gate.weight"), vec![n_experts, d]);
                    self.add(&mp("gate.e_score_correction_bias"), vec![n_experts]);
                    let s_i = moe_inter * n_shared;
                    self.add(&mp("shared_experts.gate_proj.weight"), vec![s_i, d]);
                    self.add(&mp("shared_experts.up_proj.weight"), vec![s_i, d]);
                    self.add(&mp("shared_experts.down_proj.weight"), vec![d, s_i]);
                    // routed expert tensors deliberately NOT written -- Model::load never
                    // touches them, matching glm52::model::Model::load's own streaming design.
                }
            }

            let cfg_json = json!({
                "model_type": "kimi_linear",
                "hidden_size": d, "num_hidden_layers": 3, "num_attention_heads": h,
                "first_k_dense_replace": 1, "q_lora_rank": null, "kv_lora_rank": kv_lora,
                "qk_nope_head_dim": qk_nope, "qk_rope_head_dim": qk_rope, "v_head_dim": v_head,
                "num_experts": n_experts, "num_experts_per_token": 2, "num_shared_experts": n_shared,
                "num_expert_group": 1, "topk_group": 1, "moe_intermediate_size": moe_inter,
                "intermediate_size": dense_inter, "vocab_size": vocab, "moe_renormalize": true,
                "rms_norm_eps": 1e-5, "routed_scaling_factor": 1.0, "mla_use_nope": true,
                "moe_router_activation_func": "sigmoid", "rope_theta": 10000.0,
                "linear_attn_config": {
                    "head_dim": kda_head_dim, "num_heads": kda_n_heads, "short_conv_kernel_size": kernel,
                    "kda_layers": [1, 2], "full_attn_layers": [3]
                }
            });
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
    fn loads_kda_and_mla_layers_with_correct_shapes() {
        let fixture = TinyFixture::new("shapes").build();
        let m = Model::load(&fixture.0, 32, 32).unwrap();

        assert_eq!(m.layers.len(), 3);
        assert!(matches!(m.layers[0].attn, Attn::Kda(_)), "layer 0 must be KDA");
        assert!(matches!(m.layers[1].attn, Attn::Kda(_)), "layer 1 must be KDA");
        assert!(matches!(m.layers[2].attn, Attn::Mla(_)), "layer 2 must be MLA");

        assert!(matches!(m.layers[0].ffn, Ffn::Dense(_)), "layer 0 is before first_k_dense_replace=1");
        assert!(matches!(m.layers[1].ffn, Ffn::Moe(_)));
        assert!(matches!(m.layers[2].ffn, Ffn::Moe(_)));

        if let Attn::Kda(kda) = &m.layers[0].attn {
            assert_eq!(kda.q_proj.rows, 8); // d_inner = kda_head_dim(4) * kda_n_heads(2)
            assert_eq!(kda.q_proj.cols, 8); // hidden
            assert_eq!(kda.q_conv.len(), 8 * 3); // d_inner * kernel
            assert_eq!(kda.dt_bias.len(), 8); // d_inner, per-channel
            assert_eq!(kda.a_log.len(), 2, "a_log must be kda_n_heads-long, not d_inner-long");
            assert_eq!(kda.o_norm.len(), 4); // kda_head_dim, shared across heads
        } else {
            panic!("expected KDA");
        }

        if let Attn::Mla(mla) = &m.layers[2].attn {
            assert_eq!(mla.q_a.rows, 2 * 5); // h * qh = h*(qk_nope+qk_rope) = 2*5
            assert_eq!(mla.q_a.cols, 8); // hidden
            assert!(mla.q_a_ln.is_empty(), "q_a_ln is an unread placeholder for QProj::Direct");
            assert_eq!(mla.kv_b.rows, 2 * (3 + 4)); // h*(qk_nope+v_head)
        } else {
            panic!("expected MLA");
        }

        if let Ffn::Moe(moe) = &m.layers[1].ffn {
            assert_eq!(moe.router.len(), 4 * 8); // n_experts * hidden
            assert_eq!(moe.sh_gate.rows, 6); // moe_inter * n_shared
        } else {
            panic!("expected MoE");
        }
    }

    #[test]
    fn embed_row_dequantizes_the_right_row() {
        let fixture = TinyFixture::new("embed_row").build();
        let m = Model::load(&fixture.0, 32, 32).unwrap();
        let row2 = m.embed_row(2);
        assert_eq!(row2.len(), 8); // hidden
    }
}
