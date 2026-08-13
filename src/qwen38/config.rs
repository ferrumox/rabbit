//! Reads `config.json` into `Cfg` for Qwen 3.8 — the sibling of `glm52::config::Cfg` and
//! `kimi_linear::config::Cfg`, sharing neither's field set: there's no MLA (no `kv_lora_rank`,
//! no `qk_nope`/`qk_rope` split), no DSA indexer, no expert grouping (`n_group`/`topk_group`), and
//! the per-layer hybrid pattern arrives as an explicit, 0-indexed `layer_types` list of strings
//! instead of Kimi's 1-indexed integer layer lists.
//!
//! Every field name and value below comes from the REAL checkpoint's `config.json`
//! (`amd/Qwen3.8-2.4T-A95B-Quark-MXFP4`, byte-identical to `Qwen/Qwen3.8-2.4T-A95B`'s except for
//! `quantization_config`, read off disk 2026-08-13): `hidden_size: 8192`,
//! `num_hidden_layers: 92`, `num_attention_heads: 64`, `num_key_value_heads: 4`,
//! `head_dim: 256`, `partial_rotary_factor: 0.25`, `attn_output_gate: true`,
//! `output_gate_type: "swish"`, `full_attention_interval: 4`, `layer_types: [...92 entries...]`,
//! `linear_num_key_heads: 16`, `linear_num_value_heads: 128`, `linear_key_head_dim: 128`,
//! `linear_value_head_dim: 128`, `linear_conv_kernel_dim: 4`, `num_experts: 512`,
//! `num_experts_per_tok: 10`, `moe_intermediate_size: 2048`,
//! `shared_expert_intermediate_size: 2048`, `vocab_size: 248320`, `rms_norm_eps: 1e-6`,
//! `rope_parameters.rope_theta: 1e7`, `mtp_num_hidden_layers: 1`, `hidden_act: "silu"`,
//! `tie_word_embeddings: false`, `eos_token_id: 248044`.
//!
//! **`norm_topk_prob` is absent from that file, and the default it inherits is `true`** —
//! confirmed by reading `transformers`' own `configuration_qwen3_next.py` (Qwen3.5's config
//! subclasses it), where the attribute default is `norm_topk_prob: bool = True`. It is parsed
//! faithfully, but note that `Qwen3_5MoeTopKRouter` never reads the field: it renormalizes the
//! top-k probabilities unconditionally (see `qwen38::moe`), so this is recorded config rather than
//! a branch anything takes.
//!
//! **Stop tokens need BOTH files.** `config.json` says `eos_token_id: 248044`
//! (`<|endoftext|>`), but the chat template ends every assistant turn with `<|im_end|>`
//! (`248046`), and `generation_config.json` lists `[248046, 248044]` while
//! `tokenizer_config.json`'s `eos_token` is `<|im_end|>`. Reading only `config.json` — what every
//! other architecture in rabbit does, because for those two files agree — would give a model that
//! never stops in chat, the exact failure mode `chat.rs`'s GLM template doc already warns about.
//! So [`Cfg::load`] unions the two lists.

use serde_json::Value;
use std::fmt;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct Cfg {
    pub hidden: i32,
    pub n_layers: i32,
    /// Query heads (64). KV heads are `n_kv_heads` (4) — plain GQA, 16 query heads per KV head.
    pub n_heads: i32,
    pub n_kv_heads: i32,
    /// Per-head width for the full-attention layers (256) — NOT `hidden / n_heads` (which would
    /// be 128 here): Qwen 3.8 sets `head_dim` explicitly and it's twice that.
    pub head_dim: i32,
    /// How many of each head's `head_dim` dimensions RoPE actually rotates:
    /// `head_dim * partial_rotary_factor` = `256 * 0.25` = 64. The remaining 192 pass through
    /// unrotated (NoPE), so this is neither GLM's full-RoPE nor Kimi's `mla_use_nope`.
    pub rope_dim: i32,
    /// `attn_output_gate: true` — `q_proj` emits `2 * n_heads * head_dim` and the second half,
    /// through a swish (SiLU) gate, multiplies the attention output. `output_gate_type` is
    /// validated to be `"swish"` at load time (see [`ConfigError::UnsupportedOutputGate`]), so no
    /// separate field carries it.
    pub attn_output_gate: bool,
    /// Per layer (0-indexed): `true` = Gated DeltaNet (recurrent), `false` = full GQA attention.
    /// Derived from `layer_types`, or — when absent — from `full_attention_interval` exactly as
    /// `transformers`' `Qwen3NextConfig.__post_init__` does: layer `i` is full attention iff
    /// `(i + 1) % interval == 0`, i.e. layers 3, 7, 11, ... in a 4-interval model.
    pub is_linear: Vec<bool>,
    pub lin_key_heads: i32,
    pub lin_value_heads: i32,
    pub lin_key_head_dim: i32,
    pub lin_value_head_dim: i32,
    /// `linear_conv_kernel_dim` (4) — the causal depthwise conv window
    /// `kimi_linear::short_conv::ShortConvState` already implements at the same size.
    pub conv_kernel: i32,
    pub n_experts: i32,
    pub topk: i32,
    pub moe_inter: i32,
    /// `shared_expert_intermediate_size` (2048): ONE shared expert per layer, itself gated by
    /// `mlp.shared_expert_gate` (a single sigmoid-scored row), unlike GLM's/Kimi's ungated shared
    /// experts. `0` would mean no shared expert at all.
    pub shared_inter: i32,
    /// `norm_topk_prob` — renormalize the top-k router probabilities to sum to 1. Defaults to
    /// `true` when absent (see this module's doc for why that default is not a guess). Informational:
    /// the reference router always renormalizes regardless, and so does `qwen38::moe::route`.
    pub norm_topk: bool,
    pub vocab: i32,
    pub eps: f32,
    pub theta: f32,
    /// `1 / sqrt(head_dim)` = 0.0625, matching `Qwen3NextAttention`'s `scaling = head_dim**-0.5`.
    pub attn_scale: f32,
    /// Union of `config.json`'s and `generation_config.json`'s `eos_token_id`s — see this
    /// module's doc. Both `<|im_end|>` (248046) and `<|endoftext|>` (248044) on a real checkpoint.
    pub stop_ids: Vec<i32>,
    /// `mtp_num_hidden_layers` (1). Parsed only so the count is never silently lost: rabbit does
    /// no speculative decoding, and on this checkpoint the MTP block is a FULL extra MoE layer
    /// (its own 512 experts, ~14 GB of MXFP4), so every loader skips `mtp.*` entirely.
    pub mtp_layers: i32,
    /// `true` iff the routed experts are natively MXFP4 on disk (`{name}.weight` U8 +
    /// `{name}.weight_scale` E8M0, group 32) — picks `ExpertNaming::Qwen38Mxfp4` over the plain
    /// `.weight` float/int4 path. Detected from `quantization_config`, never assumed: see
    /// [`mxfp4_experts_from_quant`] for exactly which fields have to line up.
    pub mxfp4_experts: bool,
    /// `0` = per-row-scaled `.qs` sidecars (if any). `>0` = grouped int4 — same
    /// `rabbit_group_size` convention as `glm52::config::Cfg::group_size`, only ever present on a
    /// checkpoint rabbit's own converter produced.
    pub group_size: i32,
}

/// The flat, text-only form: Qwen3.8-Max's own `model_type`, every field at the top level.
const SUPPORTED_MODEL_TYPE: &str = "qwen3_5_moe_text";
/// The multimodal wrapper (Qwen3.6-35B-A3B and friends): same text fields, nested under
/// `text_config`, with a `vision_config` sibling this module ignores (rabbit is a text engine).
const SUPPORTED_WRAPPER_MODEL_TYPE: &str = "qwen3_5_moe";
const SUPPORTED_OUTPUT_GATE: &str = "swish";
const LINEAR_LAYER_TYPE: &str = "linear_attention";
const FULL_LAYER_TYPE: &str = "full_attention";

#[derive(Debug)]
pub enum ConfigError {
    Io(std::io::Error),
    Json(serde_json::Error),
    /// `model_type` was missing or matched neither supported spelling.
    UnsupportedArchitecture { found: Option<String> },
    /// `model_type` was the multimodal wrapper, but `text_config` is missing or not an object.
    MissingTextConfig,
    /// `layer_types` is present but its length doesn't match `num_hidden_layers`.
    LayerTypesLength { got: usize, want: usize },
    /// A `layer_types` entry that is neither `"linear_attention"` nor `"full_attention"` — a new
    /// third layer kind would need real code, so it can't be silently bucketed into either.
    UnknownLayerType { index: usize, found: String },
    /// `attn_output_gate` is on but `output_gate_type` isn't the swish gate this port implements.
    UnsupportedOutputGate { found: String },
    /// A `quantization_config` rabbit can't read the experts out of — better a clear error here
    /// than a `MissingTensor` 90 layers deep, or worse, silently reading packed bytes as floats.
    UnsupportedQuantization { detail: String },
    OutOfRange { name: &'static str, value: i64, lo: i64, hi: i64 },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::Io(e) => write!(f, "config.json: {e}"),
            ConfigError::Json(e) => write!(f, "config.json: {e}"),
            ConfigError::UnsupportedArchitecture { found: Some(m) } => write!(
                f,
                "config.json's model_type is {m:?}, but this loader only supports {SUPPORTED_MODEL_TYPE:?} / {SUPPORTED_WRAPPER_MODEL_TYPE:?} (Qwen 3.8)"
            ),
            ConfigError::UnsupportedArchitecture { found: None } => write!(f, "config.json has no model_type field"),
            ConfigError::MissingTextConfig => write!(
                f,
                "config.json's model_type is {SUPPORTED_WRAPPER_MODEL_TYPE:?} but it has no text_config object to read the text backbone's fields from"
            ),
            ConfigError::LayerTypesLength { got, want } => {
                write!(f, "config: layer_types has {got} entries but num_hidden_layers is {want}")
            }
            ConfigError::UnknownLayerType { index, found } => {
                write!(f, "config: layer_types[{index}] is {found:?}, expected {LINEAR_LAYER_TYPE:?} or {FULL_LAYER_TYPE:?}")
            }
            ConfigError::UnsupportedOutputGate { found } => {
                write!(f, "config: output_gate_type is {found:?}, but only {SUPPORTED_OUTPUT_GATE:?} is implemented")
            }
            ConfigError::UnsupportedQuantization { detail } => write!(f, "config: unsupported quantization_config ({detail})"),
            ConfigError::OutOfRange { name, value, lo, hi } => write!(f, "config: {name}={value} out of range [{lo},{hi}]"),
        }
    }
}

impl std::error::Error for ConfigError {}

impl From<std::io::Error> for ConfigError {
    fn from(e: std::io::Error) -> Self {
        ConfigError::Io(e)
    }
}

impl From<serde_json::Error> for ConfigError {
    fn from(e: serde_json::Error) -> Self {
        ConfigError::Json(e)
    }
}

fn gi(v: &Value, key: &str) -> i32 {
    v.get(key).and_then(Value::as_f64).unwrap_or(0.0) as i32
}

fn gf(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(Value::as_f64)
}

macro_rules! check_range {
    ($name:expr, $value:expr, $lo:expr, $hi:expr) => {
        // `!(lo..=hi).contains(...)` rather than `< || >`: identical for the integer bounds every
        // caller passes, and the form clippy's `manual_range_contains` accepts (the sibling
        // configs' older spelling predates that lint firing on these call shapes).
        if !($lo..=$hi).contains(&$value) {
            return Err(ConfigError::OutOfRange { name: $name, value: $value as i64, lo: $lo as i64, hi: $hi as i64 });
        }
    };
}

/// Every `eos_token_id` spelling HF uses: a bare int, or an array of them.
fn read_eos(v: &Value, out: &mut Vec<i32>) {
    match v.get("eos_token_id") {
        Some(Value::Array(arr)) => {
            for id in arr.iter().take(8).filter_map(Value::as_f64) {
                let id = id as i32;
                if !out.contains(&id) {
                    out.push(id);
                }
            }
        }
        Some(other) => {
            if let Some(id) = other.as_f64() {
                let id = id as i32;
                if !out.contains(&id) {
                    out.push(id);
                }
            }
        }
        None => {}
    }
}

/// Decides whether the routed experts are natively MXFP4, from `quantization_config` alone.
///
/// Recognized, in order:
/// * absent -> `false` (a plain bf16/f32 checkpoint, or one of rabbit's own int4 conversions).
/// * `quant_method: "quark"` with `global_quant_config.weight` = `{dtype: "fp4", qscheme:
///   "per_group", group_size: 32, scale_format: "e8m0"}` -> `true`. That is exactly the container
///   `quant.rs`'s `QTKind::MxFp4` reads, verified bit-for-bit against this checkpoint (see
///   `ExpertNaming::Qwen38Mxfp4`'s doc). Quark's `export.pack_method: "reorder"` does NOT change
///   the nibble order and is deliberately not checked.
/// * `format: "mxfp4-pack-quantized"` (`compressed-tensors`' spelling, what Kimi K3 ships) ->
///   `true`, so a Qwen checkpoint quantized by that tool instead would also work.
/// * `quant_method: "fp8"` -> `false`: FP8 tensors are handled by the expert cache's own
///   `_scale_inv` block-scale path (the format GLM-5.2's original checkpoint shipped), not here.
/// * anything else -> [`ConfigError::UnsupportedQuantization`].
fn mxfp4_experts_from_quant(quant: Option<&Value>) -> Result<bool, ConfigError> {
    let Some(q) = quant else { return Ok(false) };
    if q.get("format").and_then(Value::as_str) == Some("mxfp4-pack-quantized") {
        return Ok(true);
    }
    let method = q.get("quant_method").and_then(Value::as_str).unwrap_or("");
    match method {
        "quark" => {
            let w = q.get("global_quant_config").and_then(|g| g.get("weight"));
            let dtype = w.and_then(|w| w.get("dtype")).and_then(Value::as_str);
            let qscheme = w.and_then(|w| w.get("qscheme")).and_then(Value::as_str);
            let group = w.and_then(|w| w.get("group_size")).and_then(Value::as_i64);
            let scale = w.and_then(|w| w.get("scale_format")).and_then(Value::as_str);
            if (dtype, qscheme, group, scale) == (Some("fp4"), Some("per_group"), Some(32), Some("e8m0")) {
                Ok(true)
            } else {
                Err(ConfigError::UnsupportedQuantization {
                    detail: format!(
                        "quant_method \"quark\" but weight is dtype={dtype:?} qscheme={qscheme:?} group_size={group:?} scale_format={scale:?}, not the OCP MXFP4 container rabbit reads (fp4/per_group/32/e8m0)"
                    ),
                })
            }
        }
        "fp8" => Ok(false),
        other => Err(ConfigError::UnsupportedQuantization { detail: format!("quant_method {other:?} is not supported") }),
    }
}

impl Cfg {
    /// Reads `<snap_dir>/config.json`, then folds in `<snap_dir>/generation_config.json`'s own
    /// `eos_token_id`s if that file exists (see this module's doc — on a real Qwen checkpoint the
    /// chat-ending `<|im_end|>` is ONLY listed there). A missing or unparseable
    /// `generation_config.json` is not an error: it just leaves `stop_ids` as `config.json` had it.
    pub fn load(snap_dir: &Path) -> Result<Cfg, ConfigError> {
        let text = fs::read_to_string(snap_dir.join("config.json"))?;
        let root: Value = serde_json::from_str(&text)?;
        let mut cfg = Cfg::from_root(&root)?;
        if let Ok(gen_text) = fs::read_to_string(snap_dir.join("generation_config.json"))
            && let Ok(gen_cfg) = serde_json::from_str::<Value>(&gen_text)
        {
            read_eos(&gen_cfg, &mut cfg.stop_ids);
        }
        Ok(cfg)
    }

    /// Dispatches on `model_type` between the flat text-only layout and the multimodal wrapper's
    /// `text_config`, then parses whichever object holds the text fields. `quantization_config` is
    /// looked up in the text object FIRST and the root SECOND — Kimi K3 nests it inside
    /// `text_config`, while Qwen's own multimodal checkpoints keep it at the root next to
    /// `vision_config`.
    pub fn from_root(root: &Value) -> Result<Cfg, ConfigError> {
        let model_type = root.get("model_type").and_then(Value::as_str);
        let text = match model_type {
            Some(SUPPORTED_MODEL_TYPE) => root,
            Some(SUPPORTED_WRAPPER_MODEL_TYPE) => match root.get("text_config") {
                Some(t) if t.is_object() => t,
                _ => return Err(ConfigError::MissingTextConfig),
            },
            found => return Err(ConfigError::UnsupportedArchitecture { found: found.map(String::from) }),
        };
        let quant = text.get("quantization_config").or_else(|| root.get("quantization_config"));
        Cfg::from_fields(text, quant)
    }

    /// Parses `text` (the object holding the text backbone's fields — the root itself, or a
    /// `text_config`) into a `Cfg`. `quant` is the `quantization_config` object, if any.
    pub fn from_fields(text: &Value, quant: Option<&Value>) -> Result<Cfg, ConfigError> {
        let n_layers = gi(text, "num_hidden_layers");
        check_range!("num_hidden_layers", n_layers, 1, 1024);
        let hidden = gi(text, "hidden_size");
        check_range!("hidden_size", hidden, 1, 1 << 20);
        let n_heads = gi(text, "num_attention_heads");
        check_range!("num_attention_heads", n_heads, 1, 1 << 16);
        let n_kv_heads = gi(text, "num_key_value_heads").max(n_heads.min(1));
        check_range!("num_key_value_heads", n_kv_heads, 1, n_heads);
        if n_heads % n_kv_heads != 0 {
            return Err(ConfigError::OutOfRange { name: "num_key_value_heads", value: n_kv_heads as i64, lo: 1, hi: n_heads as i64 });
        }

        // `head_dim` is explicit (256) and deliberately NOT `hidden / n_heads` (128 here); fall
        // back to the derived value only for a checkpoint that omits it entirely.
        let head_dim = match gi(text, "head_dim") {
            0 => hidden / n_heads,
            d => d,
        };
        check_range!("head_dim", head_dim, 1, 1 << 16);

        // `partial_rotary_factor` appears both at the top level and inside `rope_parameters` on
        // the real checkpoint (same value); either spelling is accepted, the nested one wins.
        let rope_params = text.get("rope_parameters");
        let partial = rope_params
            .and_then(|r| r.get("partial_rotary_factor"))
            .and_then(Value::as_f64)
            .or_else(|| gf(text, "partial_rotary_factor"))
            .unwrap_or(1.0);
        if partial <= 0.0 || partial > 1.0 {
            return Err(ConfigError::OutOfRange { name: "partial_rotary_factor", value: (partial * 100.0) as i64, lo: 1, hi: 100 });
        }
        let rope_dim = (head_dim as f64 * partial) as i32;
        check_range!("rope_dim", rope_dim, 1, head_dim);

        let attn_output_gate = text.get("attn_output_gate").and_then(Value::as_bool).unwrap_or(false);
        let gate_type = text.get("output_gate_type").and_then(Value::as_str).unwrap_or(SUPPORTED_OUTPUT_GATE);
        if attn_output_gate && gate_type != SUPPORTED_OUTPUT_GATE {
            return Err(ConfigError::UnsupportedOutputGate { found: gate_type.to_string() });
        }

        let is_linear = layer_types(text, n_layers as usize)?;

        let theta = rope_params
            .and_then(|r| r.get("rope_theta"))
            .and_then(Value::as_f64)
            .or_else(|| gf(text, "rope_theta"))
            .unwrap_or(10_000.0) as f32;
        let eps = gf(text, "rms_norm_eps").unwrap_or(1e-6) as f32;

        let n_experts = gi(text, "num_experts");
        check_range!("num_experts", n_experts, 0, 1 << 20);
        let topk = gi(text, "num_experts_per_tok");
        check_range!("num_experts_per_tok", topk, 0, n_experts.max(1));

        let mut stop_ids = Vec::new();
        read_eos(text, &mut stop_ids);

        Ok(Cfg {
            hidden,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            rope_dim,
            attn_output_gate,
            is_linear,
            lin_key_heads: gi(text, "linear_num_key_heads"),
            lin_value_heads: gi(text, "linear_num_value_heads"),
            lin_key_head_dim: gi(text, "linear_key_head_dim"),
            lin_value_head_dim: gi(text, "linear_value_head_dim"),
            conv_kernel: gi(text, "linear_conv_kernel_dim"),
            n_experts,
            topk,
            moe_inter: gi(text, "moe_intermediate_size"),
            shared_inter: gi(text, "shared_expert_intermediate_size"),
            norm_topk: text.get("norm_topk_prob").and_then(Value::as_bool).unwrap_or(true),
            vocab: gi(text, "vocab_size"),
            eps,
            theta,
            attn_scale: 1.0 / (head_dim as f32).sqrt(),
            stop_ids,
            mtp_layers: gi(text, "mtp_num_hidden_layers"),
            mxfp4_experts: mxfp4_experts_from_quant(quant)?,
            group_size: text.get("rabbit_group_size").and_then(Value::as_i64).unwrap_or(0) as i32,
        })
    }

    /// How many full-attention (GQA) layers this config has — the count of ordinary KV caches a
    /// `KvState` will need, the rest being fixed-size recurrent state.
    pub fn n_full_attn_layers(&self) -> usize {
        self.is_linear.iter().filter(|&&lin| !lin).count()
    }

    /// Value heads per key head in the Gated DeltaNet layers (8 on the real checkpoint: 128 value
    /// heads over 16 key heads). `1` means no grouping, the shape `kimi_linear::kda` assumes.
    pub fn lin_heads_per_key(&self) -> i32 {
        if self.lin_key_heads > 0 { self.lin_value_heads / self.lin_key_heads } else { 0 }
    }
}

/// Builds the per-layer `is_linear` list from `layer_types`, or derives it from
/// `full_attention_interval` when that key is absent — the same rule
/// `transformers`' `Qwen3NextConfig.__post_init__` uses (`"linear_attention" if bool((i + 1) %
/// interval) else "full_attention"`), which on the real 92-layer/interval-4 checkpoint reproduces
/// its explicit list exactly (pinned by a test below).
fn layer_types(text: &Value, n_layers: usize) -> Result<Vec<bool>, ConfigError> {
    if let Some(list) = text.get("layer_types").and_then(Value::as_array) {
        if list.len() != n_layers {
            return Err(ConfigError::LayerTypesLength { got: list.len(), want: n_layers });
        }
        let mut out = Vec::with_capacity(n_layers);
        for (i, v) in list.iter().enumerate() {
            match v.as_str() {
                Some(LINEAR_LAYER_TYPE) => out.push(true),
                Some(FULL_LAYER_TYPE) => out.push(false),
                other => {
                    return Err(ConfigError::UnknownLayerType {
                        index: i,
                        found: other.map(String::from).unwrap_or_else(|| v.to_string()),
                    });
                }
            }
        }
        return Ok(out);
    }
    let interval = gi(text, "full_attention_interval").max(1) as usize;
    Ok((0..n_layers).map(|i| (i + 1) % interval != 0).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The real Qwen3.8-Max `config.json`'s fields, verbatim except that `layer_types` is built by
    /// the same formula the file spells out entry by entry — `real_layer_types_match_the_interval_
    /// formula` below is what proves those two are the same thing, so this helper can stay short.
    fn real_config() -> Value {
        let layer_types: Vec<&str> =
            (0..92).map(|i| if (i + 1) % 4 == 0 { FULL_LAYER_TYPE } else { LINEAR_LAYER_TYPE }).collect();
        json!({
            "architectures": ["Qwen3_5MoeForCausalLM"],
            "model_type": SUPPORTED_MODEL_TYPE,
            "attn_output_gate": true,
            "output_gate_type": "swish",
            "bos_token_id": 248044,
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 8192,
            "layer_types": layer_types,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 128,
            "linear_value_head_dim": 128,
            "max_position_embeddings": 262144,
            "moe_intermediate_size": 2048,
            "mtp_num_hidden_layers": 1,
            "num_attention_heads": 64,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "num_hidden_layers": 92,
            "num_key_value_heads": 4,
            "partial_rotary_factor": 0.25,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {"partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"},
            "shared_expert_intermediate_size": 2048,
            "tie_word_embeddings": false,
            "vocab_size": 248320
        })
    }

    /// The real AMD-Quark `quantization_config`, trimmed to the fields
    /// `mxfp4_experts_from_quant` actually reads.
    fn quark_quant() -> Value {
        json!({
            "quant_method": "quark",
            "quant_mode": "eager_mode",
            "version": "0.12.post1",
            "export": {"pack_method": "reorder", "weight_format": "real_quantized"},
            "global_quant_config": {
                "weight": {"dtype": "fp4", "is_dynamic": false, "qscheme": "per_group", "group_size": 32, "scale_format": "e8m0"},
                "input_tensors": {"dtype": "fp4", "qscheme": "per_group", "group_size": 32, "scale_format": "e8m0"}
            }
        })
    }

    struct TempDir(std::path::PathBuf);
    impl TempDir {
        fn new(name: &str) -> TempDir {
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

    #[test]
    fn parses_the_real_qwen38_max_config() {
        let c = Cfg::from_root(&real_config()).unwrap();
        assert_eq!((c.hidden, c.n_layers, c.vocab), (8192, 92, 248320));
        assert_eq!((c.n_heads, c.n_kv_heads, c.head_dim), (64, 4, 256));
        assert_eq!(c.rope_dim, 64, "partial_rotary_factor 0.25 of head_dim 256");
        assert_eq!(c.attn_scale, 1.0 / 16.0);
        assert!(c.attn_output_gate);
        assert_eq!((c.n_experts, c.topk, c.moe_inter, c.shared_inter), (512, 10, 2048, 2048));
        assert!(c.norm_topk, "absent norm_topk_prob must default to TRUE, not false");
        assert_eq!((c.lin_key_heads, c.lin_value_heads), (16, 128));
        assert_eq!((c.lin_key_head_dim, c.lin_value_head_dim, c.conv_kernel), (128, 128, 4));
        assert_eq!(c.lin_heads_per_key(), 8, "128 value heads over 16 key heads");
        assert_eq!(c.theta, 1e7);
        assert_eq!(c.eps, 1e-6);
        assert_eq!(c.mtp_layers, 1);
        assert_eq!(c.group_size, 0);
        assert!(!c.mxfp4_experts, "the bf16 original declares no quantization_config");
    }

    /// The 3:1 hybrid pattern: 69 Gated DeltaNet layers and 23 GQA layers, with full attention on
    /// layers 3, 7, 11, ... — checked by position, not just by count, since a swapped mapping
    /// would keep both counts identical.
    #[test]
    fn maps_the_hybrid_layer_pattern_by_position() {
        let c = Cfg::from_root(&real_config()).unwrap();
        assert_eq!(c.is_linear.len(), 92);
        assert_eq!(c.n_full_attn_layers(), 23);
        assert_eq!(c.is_linear.iter().filter(|&&l| l).count(), 69);
        for (i, &linear) in c.is_linear.iter().enumerate() {
            assert_eq!(linear, (i + 1) % 4 != 0, "layer {i}");
        }
        assert!(!c.is_linear[3] && !c.is_linear[91], "layers 3 and 91 are full attention");
        assert!(c.is_linear[0] && c.is_linear[2], "layers 0..2 are linear");
    }

    /// `layer_types` absent -> derived from `full_attention_interval`, and the result must equal
    /// what the real checkpoint lists explicitly. This is the test the `real_config` helper leans
    /// on to justify generating the list by formula.
    #[test]
    fn real_layer_types_match_the_interval_formula() {
        let mut explicit = real_config();
        let listed = Cfg::from_root(&explicit).unwrap().is_linear;

        explicit.as_object_mut().unwrap().remove("layer_types");
        let derived = Cfg::from_root(&explicit).unwrap().is_linear;
        assert_eq!(listed, derived);
    }

    /// Qwen3.6-35B-A3B-shaped: same text fields under `text_config`, a `vision_config` sibling to
    /// ignore, and `quantization_config` at the ROOT rather than nested.
    #[test]
    fn parses_the_multimodal_wrapper_form_identically() {
        let flat = real_config();
        let nested = json!({
            "model_type": SUPPORTED_WRAPPER_MODEL_TYPE,
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "text_config": flat,
            "vision_config": {"depth": 27, "hidden_size": 1152},
            "quantization_config": quark_quant()
        });
        let a = Cfg::from_root(&real_config()).unwrap();
        let b = Cfg::from_root(&nested).unwrap();
        assert!(b.mxfp4_experts, "root-level quantization_config must still be found");
        assert_eq!(Cfg { mxfp4_experts: false, ..b }, a, "every text field must parse the same either way");
    }

    #[test]
    fn detects_the_real_quark_mxfp4_experts() {
        let mut cfg = real_config();
        cfg.as_object_mut().unwrap().insert("quantization_config".to_string(), quark_quant());
        assert!(Cfg::from_root(&cfg).unwrap().mxfp4_experts);
    }

    /// `compressed-tensors`' spelling (what Kimi K3 ships) must also be recognized, and an FP8
    /// checkpoint must parse as "not MXFP4" rather than erroring — the expert cache reads FP8 via
    /// its own `_scale_inv` path.
    #[test]
    fn accepts_compressed_tensors_mxfp4_and_plain_fp8() {
        assert!(mxfp4_experts_from_quant(Some(&json!({"format": "mxfp4-pack-quantized"}))).unwrap());
        assert!(!mxfp4_experts_from_quant(Some(&json!({"quant_method": "fp8", "fmt": "e4m3"}))).unwrap());
        assert!(!mxfp4_experts_from_quant(None).unwrap());
    }

    /// A Quark export that isn't the OCP MXFP4 container (here: NVFP4's E4M3 scales over groups of
    /// 16) must be REJECTED, not silently read as MXFP4 — the bytes would decode to garbage.
    #[test]
    fn rejects_a_quark_export_that_is_not_ocp_mxfp4() {
        let nvfp4 = json!({
            "quant_method": "quark",
            "global_quant_config": {"weight": {"dtype": "fp4", "qscheme": "per_group", "group_size": 16, "scale_format": "e4m3"}}
        });
        let e = mxfp4_experts_from_quant(Some(&nvfp4)).unwrap_err();
        assert!(matches!(e, ConfigError::UnsupportedQuantization { .. }), "got {e}");
        assert!(e.to_string().contains("e4m3"), "the error must say what it actually found: {e}");
        let unknown = json!({"quant_method": "awq"});
        assert!(matches!(mxfp4_experts_from_quant(Some(&unknown)), Err(ConfigError::UnsupportedQuantization { .. })));
    }

    /// `<|im_end|>` (248046) lives ONLY in `generation_config.json`; a Cfg built from
    /// `config.json` alone would stop only on `<|endoftext|>` (248044) and babble past every turn
    /// end in chat.
    #[test]
    fn load_unions_stop_ids_from_generation_config() {
        let dir = TempDir::new("rabbit_test_qwen38_cfg_stops");
        fs::write(dir.0.join("config.json"), serde_json::to_vec(&real_config()).unwrap()).unwrap();

        let only_config = Cfg::load(&dir.0).unwrap();
        assert_eq!(only_config.stop_ids, vec![248044]);

        fs::write(
            dir.0.join("generation_config.json"),
            serde_json::to_vec(&json!({"bos_token_id": 248044, "eos_token_id": [248046, 248044], "top_k": 20})).unwrap(),
        )
        .unwrap();
        let both = Cfg::load(&dir.0).unwrap();
        assert_eq!(both.stop_ids, vec![248044, 248046], "union, config.json's first, no duplicates");
    }

    #[test]
    fn rejects_unknown_architectures_and_malformed_layer_types() {
        assert!(matches!(
            Cfg::from_root(&json!({"model_type": "glm_moe_dsa"})),
            Err(ConfigError::UnsupportedArchitecture { .. })
        ));
        assert!(matches!(Cfg::from_root(&json!({})), Err(ConfigError::UnsupportedArchitecture { found: None })));
        assert!(matches!(
            Cfg::from_root(&json!({"model_type": SUPPORTED_WRAPPER_MODEL_TYPE})),
            Err(ConfigError::MissingTextConfig)
        ));

        let mut short = real_config();
        short.as_object_mut().unwrap().insert("layer_types".to_string(), json!(vec![LINEAR_LAYER_TYPE; 91]));
        assert!(matches!(Cfg::from_root(&short), Err(ConfigError::LayerTypesLength { got: 91, want: 92 })));

        let mut weird = real_config();
        let mut types = vec![LINEAR_LAYER_TYPE; 92];
        types[7] = "sliding_attention";
        weird.as_object_mut().unwrap().insert("layer_types".to_string(), json!(types));
        match Cfg::from_root(&weird) {
            Err(ConfigError::UnknownLayerType { index: 7, found }) => assert_eq!(found, "sliding_attention"),
            other => panic!("expected UnknownLayerType at 7, got {other:?}"),
        }
    }

    #[test]
    fn rejects_an_output_gate_this_port_does_not_implement() {
        let mut cfg = real_config();
        cfg.as_object_mut().unwrap().insert("output_gate_type".to_string(), json!("gelu"));
        match Cfg::from_root(&cfg) {
            Err(ConfigError::UnsupportedOutputGate { found }) => assert_eq!(found, "gelu"),
            other => panic!("expected UnsupportedOutputGate, got {other:?}"),
        }
        // ...but only when the gate is actually ON.
        cfg.as_object_mut().unwrap().insert("attn_output_gate".to_string(), json!(false));
        assert!(Cfg::from_root(&cfg).is_ok());
    }

    #[test]
    fn rejects_gqa_head_counts_that_do_not_divide() {
        let mut cfg = real_config();
        cfg.as_object_mut().unwrap().insert("num_key_value_heads".to_string(), json!(5));
        assert!(matches!(Cfg::from_root(&cfg), Err(ConfigError::OutOfRange { name: "num_key_value_heads", .. })));
    }
}
