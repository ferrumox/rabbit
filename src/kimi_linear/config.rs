//! Reads `config.json` into `Cfg` for Kimi Linear (`model_type: "kimi_linear"`) — the sibling
//! of `glm52::config::Cfg`, not a variant of it: Kimi Linear has no DSA indexer at all, may have
//! no Q-LoRA (`q_lora_rank` is `null` on the real 48B/A3B checkpoint), and needs its own
//! per-layer KDA-vs-MLA type list that GLM-5.2 has no equivalent of.
//!
//! Every field name below and its exact value is taken from the real
//! `moonshotai/Kimi-Linear-48B-A3B-Instruct` checkpoint's `config.json` (fetched this session,
//! not guessed from the paper): `hidden_size: 2304`, `num_hidden_layers: 27`,
//! `num_attention_heads: 32`, `first_k_dense_replace: 1`, `q_lora_rank: null`,
//! `kv_lora_rank: 512`, `qk_nope_head_dim: 128`, `qk_rope_head_dim: 64`, `v_head_dim: 128`,
//! `num_experts: 256`, `num_experts_per_token: 8`, `num_shared_experts: 1`,
//! `num_expert_group: 1`, `topk_group: 1`, `moe_intermediate_size: 1024`,
//! `intermediate_size: 9216`, `vocab_size: 163840`, `moe_renormalize: true`,
//! `rms_norm_eps: 1e-5`, `routed_scaling_factor: 2.446`, `mla_use_nope: true`,
//! `moe_router_activation_func: "sigmoid"`, and a `linear_attn_config` object with
//! `head_dim: 128`, `num_heads: 32`, `short_conv_kernel_size: 4`,
//! `full_attn_layers: [4,8,12,16,20,24,27]`, `kda_layers: [1,2,3,5,6,7,9,...,25,26]`.
//!
//! **`full_attn_layers`/`kda_layers` are 1-indexed**, not 0-indexed: `full_attn_layers`' max
//! entry is `27` in a 27-layer model, only valid as a 1-indexed layer number (0-indexed layers
//! run `0..=26`). `Cfg::load` subtracts 1 from every entry before building `is_kda`.

use serde_json::Value;
use std::fmt;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct Cfg {
    pub hidden: i32,
    pub n_layers: i32,
    pub n_heads: i32,
    pub first_dense: i32,
    /// 0 means no Q-LoRA (the real 48B/A3B checkpoint's `q_lora_rank: null`) — the eventual
    /// model loader branches on this the same way llama.cpp's reference does (plain `wq` vs.
    /// `wq_a`/`wq_b` + a norm), not a hypothetical this config alone decides.
    pub q_lora: i32,
    pub kv_lora: i32,
    pub qk_nope: i32,
    pub qk_rope: i32,
    pub qk_head: i32,
    pub v_head: i32,
    pub n_experts: i32,
    pub topk: i32,
    pub n_shared: i32,
    pub n_group: i32,
    pub topk_group: i32,
    pub moe_inter: i32,
    pub dense_inter: i32,
    pub vocab: i32,
    pub norm_topk: bool,
    pub eps: f32,
    pub theta: f32,
    pub attn_scale: f32,
    pub routed_scale: f32,
    pub stop_ids: Vec<i32>,
    /// KDA's own head_dim (`linear_attn_config.head_dim`) — distinct from `qk_nope`/`qk_rope`,
    /// which are the MLA layers' head dimensions.
    pub kda_head_dim: i32,
    pub kda_n_heads: i32,
    pub short_conv_kernel: i32,
    /// Per layer (0-indexed): `true` = KDA (recurrent), `false` = MLA (global attention).
    /// Derived from `linear_attn_config`'s 1-indexed `kda_layers`/`full_attn_layers`.
    pub is_kda: Vec<bool>,
    /// `0` = every resident `.qs` sidecar (if any) is per-row-scaled. `>0` = the converter that
    /// produced this checkpoint used grouped int4 — see `glm52::config::Cfg::group_size`'s doc
    /// for the full reasoning (identical here, just mirrored for Kimi's own `Cfg`).
    pub group_size: i32,
}

const SUPPORTED_MODEL_TYPE: &str = "kimi_linear";
const SUPPORTED_ROUTER_ACTIVATION: &str = "sigmoid";

#[derive(Debug)]
pub enum ConfigError {
    Io(std::io::Error),
    Json(serde_json::Error),
    UnsupportedArchitecture { found: Option<String> },
    /// `linear_attn_config` is missing entirely, or malformed (not an object / not present).
    MissingLinearAttnConfig,
    /// A 0-indexed layer wasn't covered by exactly one of `kda_layers`/`full_attn_layers` — 0
    /// coverage means a config bug (or an unrecognized future variant) silently defaulting that
    /// layer's attention math to something unintended; 2+ coverage is an internally
    /// inconsistent config. Either way this must be a load error, not a silent guess.
    LayerTypeCoverage { layer: i32, times_covered: u32 },
    /// `mla_use_nope` was present and `false` — the global MLA layers would need real RoPE,
    /// which `glm52::attention::Rope::Off` (this port's whole MLA-reuse strategy) doesn't
    /// support. Not encountered on the real checkpoint; guarded against rather than guessed at.
    RopeNotSupported,
    /// `moe_router_activation_func` was something other than `"sigmoid"` — the routing math
    /// this port reuses from `glm52::moe` is sigmoid-based (`noaux_tc`-style), not generic.
    UnsupportedRouterActivation { found: String },
    /// `n_routed_experts`-equivalent doesn't divide evenly into `n_group` groups (same
    /// constraint as `glm52::config::ConfigError::InvalidGrouping`, kept as this crate's own
    /// variant rather than a cross-architecture `use` — see `rabbit-plan.md`'s Phase 1 notes).
    InvalidGrouping { n_experts: i32, n_group: i32 },
    OutOfRange { name: &'static str, value: i64, lo: i64, hi: i64 },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::Io(e) => write!(f, "config.json: {e}"),
            ConfigError::Json(e) => write!(f, "config.json: {e}"),
            ConfigError::UnsupportedArchitecture { found: Some(model_type) } => write!(
                f,
                "config.json's model_type is {model_type:?}, but rabbit's kimi_linear loader only supports {SUPPORTED_MODEL_TYPE:?}"
            ),
            ConfigError::UnsupportedArchitecture { found: None } => write!(
                f,
                "config.json has no model_type field; rabbit's kimi_linear loader only supports {SUPPORTED_MODEL_TYPE:?}"
            ),
            ConfigError::MissingLinearAttnConfig => {
                write!(f, "config.json has no (valid object) linear_attn_config -- required for kimi_linear")
            }
            ConfigError::LayerTypeCoverage { layer, times_covered: 0 } => {
                write!(f, "layer {layer} is in neither kda_layers nor full_attn_layers")
            }
            ConfigError::LayerTypeCoverage { layer, times_covered } => {
                write!(f, "layer {layer} is in both kda_layers and full_attn_layers ({times_covered}x)")
            }
            ConfigError::RopeNotSupported => {
                write!(f, "config.json's mla_use_nope is false -- MLA-with-RoPE isn't supported yet")
            }
            ConfigError::UnsupportedRouterActivation { found } => write!(
                f,
                "config.json's moe_router_activation_func is {found:?}, but rabbit only supports {SUPPORTED_ROUTER_ACTIVATION:?}"
            ),
            ConfigError::InvalidGrouping { n_experts, n_group } => {
                write!(f, "num_experts={n_experts} doesn't divide evenly into num_expert_group={n_group} groups")
            }
            ConfigError::OutOfRange { name, value, lo, hi } => {
                write!(f, "config: {name}={value} out of range [{lo},{hi}]")
            }
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

macro_rules! check_range {
    ($name:expr, $value:expr, $lo:expr, $hi:expr) => {
        if $value < $lo || $value > $hi {
            return Err(ConfigError::OutOfRange {
                name: $name,
                value: $value as i64,
                lo: $lo as i64,
                hi: $hi as i64,
            });
        }
    };
}

impl Cfg {
    pub fn load(snap_dir: &Path) -> Result<Cfg, ConfigError> {
        let text = fs::read_to_string(snap_dir.join("config.json"))?;
        let r: Value = serde_json::from_str(&text)?;

        match r.get("model_type").and_then(Value::as_str) {
            Some(model_type) if model_type == SUPPORTED_MODEL_TYPE => {}
            Some(model_type) => {
                return Err(ConfigError::UnsupportedArchitecture { found: Some(model_type.to_string()) });
            }
            None => return Err(ConfigError::UnsupportedArchitecture { found: None }),
        }

        Self::from_fields(&r)
    }

    /// The field-parsing/validation half of `load`, taking an already-parsed `config.json`
    /// object directly rather than reading a file — factored out so `kimi_k3::config::Cfg::load`
    /// can reuse it against `config.json`'s nested `text_config` object (K3's real checkpoint has
    /// `model_type: "kimi_k3"` at the top level and `text_config.model_type: "kimi_linear"`
    /// underneath, with every field below this point identical in shape/name to a standalone
    /// Kimi Linear checkpoint's top-level fields — confirmed against the real
    /// `moonshotai/Kimi-K3/config.json`, fetched 2026-07-27). Does NOT check `model_type` itself
    /// — callers are expected to have already confirmed they're looking at the right object
    /// (`load` above does so for the top-level case; `kimi_k3`'s loader does so for both levels).
    pub(crate) fn from_fields(r: &Value) -> Result<Cfg, ConfigError> {
        if r.get("mla_use_nope").and_then(Value::as_bool) == Some(false) {
            return Err(ConfigError::RopeNotSupported);
        }

        if let Some(activation) = r.get("moe_router_activation_func").and_then(Value::as_str)
            && activation != SUPPORTED_ROUTER_ACTIVATION
        {
            return Err(ConfigError::UnsupportedRouterActivation { found: activation.to_string() });
        }

        let n_layers = gi(r, "num_hidden_layers");
        let qk_nope = gi(r, "qk_nope_head_dim");
        let qk_rope = gi(r, "qk_rope_head_dim");
        let qk_head = qk_nope + qk_rope;

        let norm_topk = r.get("moe_renormalize").and_then(Value::as_bool).unwrap_or(false);
        let eps = r.get("rms_norm_eps").and_then(Value::as_f64).unwrap_or(1e-5) as f32;
        let routed_scale = r.get("routed_scaling_factor").and_then(Value::as_f64).unwrap_or(1.0) as f32;
        let theta = r.get("rope_theta").and_then(Value::as_f64).unwrap_or(10000.0) as f32;

        let mut stop_ids = Vec::new();
        if let Some(eo) = r.get("eos_token_id") {
            if let Some(n) = eo.as_f64() {
                stop_ids.push(n as i32);
            } else if let Some(arr) = eo.as_array() {
                for v in arr.iter().take(8) {
                    if let Some(n) = v.as_f64() {
                        stop_ids.push(n as i32);
                    }
                }
            }
        }

        let lac = r
            .get("linear_attn_config")
            .filter(|v| v.is_object())
            .ok_or(ConfigError::MissingLinearAttnConfig)?;
        let kda_head_dim = gi(lac, "head_dim");
        let kda_n_heads = gi(lac, "num_heads");
        let short_conv_kernel = gi(lac, "short_conv_kernel_size");

        // Both lists are 1-indexed in the real checkpoint (max full_attn_layers entry is 27 in
        // a 27-layer model) -- subtract 1 before using as a 0-indexed layer number.
        let mut times_covered = vec![0u32; n_layers.max(0) as usize];
        let mut mark = |list_key: &str| -> Result<(), ConfigError> {
            for v in lac.get(list_key).and_then(Value::as_array).into_iter().flatten() {
                if let Some(one_indexed) = v.as_f64() {
                    let idx = one_indexed as i32 - 1;
                    if idx >= 0 && (idx as usize) < times_covered.len() {
                        times_covered[idx as usize] += 1;
                    }
                }
            }
            Ok(())
        };
        mark("kda_layers")?;
        mark("full_attn_layers")?;

        let mut is_kda = Vec::with_capacity(n_layers.max(0) as usize);
        for (i, &covered) in times_covered.iter().enumerate() {
            if covered != 1 {
                return Err(ConfigError::LayerTypeCoverage { layer: i as i32, times_covered: covered });
            }
            let one_indexed = (i + 1) as f64;
            let is_full_attn = lac
                .get("full_attn_layers")
                .and_then(Value::as_array)
                .map(|arr| arr.iter().any(|v| v.as_f64() == Some(one_indexed)))
                .unwrap_or(false);
            is_kda.push(!is_full_attn);
        }

        let n_group = gi(r, "num_expert_group").max(1);
        let n_experts = gi(r, "num_experts");
        let topk_group = gi(r, "topk_group").max(1);
        if n_experts % n_group != 0 {
            return Err(ConfigError::InvalidGrouping { n_experts, n_group });
        }

        let c = Cfg {
            hidden: gi(r, "hidden_size"),
            n_layers,
            n_heads: gi(r, "num_attention_heads"),
            first_dense: gi(r, "first_k_dense_replace"),
            q_lora: gi(r, "q_lora_rank"),
            kv_lora: gi(r, "kv_lora_rank"),
            qk_nope,
            qk_rope,
            qk_head,
            v_head: gi(r, "v_head_dim"),
            n_experts,
            topk: gi(r, "num_experts_per_token"),
            n_shared: gi(r, "num_shared_experts"),
            n_group,
            topk_group,
            moe_inter: gi(r, "moe_intermediate_size"),
            dense_inter: gi(r, "intermediate_size"),
            vocab: gi(r, "vocab_size"),
            norm_topk,
            eps,
            theta,
            attn_scale: 1.0 / (qk_head as f32).sqrt(),
            routed_scale,
            stop_ids,
            kda_head_dim,
            kda_n_heads,
            short_conv_kernel,
            is_kda,
            group_size: r.get("rabbit_group_size").and_then(Value::as_i64).unwrap_or(0) as i32,
        };

        check_range!("hidden_size", c.hidden, 1, 1 << 20);
        check_range!("num_hidden_layers", c.n_layers, 1, 128);
        check_range!("num_attention_heads", c.n_heads, 1, 1024);
        check_range!("num_experts", c.n_experts, 1, 4096);
        check_range!("num_experts_per_token", c.topk, 1, 64);
        check_range!("num_expert_group", c.n_group, 1, c.n_experts);
        check_range!("topk_group", c.topk_group, 1, c.n_group);
        check_range!("moe_intermediate_size", c.moe_inter, 1, 1 << 20);
        check_range!("intermediate_size", c.dense_inter, 1, 1 << 24);
        check_range!("first_k_dense_replace", c.first_dense, 0, c.n_layers);
        check_range!("q_lora_rank", c.q_lora, 0, 1 << 20);
        check_range!("kv_lora_rank", c.kv_lora, 1, 1 << 20);
        check_range!("qk_nope_head_dim", c.qk_nope, 1, 1 << 16);
        check_range!("qk_rope_head_dim", c.qk_rope, 1, 1 << 16);
        check_range!("v_head_dim", c.v_head, 1, 1 << 16);
        check_range!("num_shared_experts", c.n_shared, 0, 64);
        check_range!("vocab_size", c.vocab, 1, 1 << 24);
        check_range!("linear_attn_config.head_dim", c.kda_head_dim, 1, 1 << 16);
        check_range!("linear_attn_config.num_heads", c.kda_n_heads, 1, 1024);
        check_range!("linear_attn_config.short_conv_kernel_size", c.short_conv_kernel, 1, 64);
        check_range!("rabbit_group_size", c.group_size, 0, 1 << 16);

        Ok(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_config(dir: &Path, json: &str) {
        fs::write(dir.join("config.json"), json).unwrap();
    }

    /// Condensed from the real moonshotai/Kimi-Linear-48B-A3B-Instruct config.json (fetched
    /// this session), scaled down to a tiny 4-layer model so the kda_layers/full_attn_layers
    /// lists stay hand-checkable: layers [1,2,4] KDA (1-indexed), layer [3] MLA.
    fn real_shaped_config_json(n_layers: i32) -> String {
        format!(
            r#"{{
                "model_type": "kimi_linear",
                "hidden_size": 2304, "num_hidden_layers": {n_layers}, "num_attention_heads": 32,
                "first_k_dense_replace": 1, "q_lora_rank": null, "kv_lora_rank": 512,
                "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
                "num_experts": 256, "num_experts_per_token": 8, "num_shared_experts": 1,
                "num_expert_group": 1, "topk_group": 1, "moe_intermediate_size": 1024,
                "intermediate_size": 9216, "vocab_size": 163840, "moe_renormalize": true,
                "rms_norm_eps": 1e-05, "routed_scaling_factor": 2.446, "mla_use_nope": true,
                "moe_router_activation_func": "sigmoid", "rope_theta": 10000.0,
                "eos_token_id": 163586,
                "linear_attn_config": {{
                    "head_dim": 128, "num_heads": 32, "short_conv_kernel_size": 4,
                    "kda_layers": [1, 2, 4], "full_attn_layers": [3]
                }}
            }}"#
        )
    }

    #[test]
    fn loads_the_real_checkpoints_shape() {
        let dir = std::env::temp_dir().join("rabbit_test_kimi_cfg_real_shape");
        fs::create_dir_all(&dir).unwrap();
        write_config(&dir, &real_shaped_config_json(4));

        let c = Cfg::load(&dir).unwrap();
        assert_eq!(c.hidden, 2304);
        assert_eq!(c.n_layers, 4);
        assert_eq!(c.n_heads, 32);
        assert_eq!(c.q_lora, 0, "q_lora_rank: null must load as 0 (no Q-LoRA)");
        assert_eq!(c.kv_lora, 512);
        assert_eq!(c.qk_head, 128 + 64);
        assert_eq!(c.v_head, 128);
        assert_eq!(c.n_experts, 256);
        assert_eq!(c.topk, 8);
        assert_eq!(c.n_shared, 1);
        assert_eq!(c.n_group, 1);
        assert_eq!(c.kda_head_dim, 128);
        assert_eq!(c.kda_n_heads, 32);
        assert_eq!(c.short_conv_kernel, 4);
        assert!((c.routed_scale - 2.446).abs() < 1e-6);
        assert_eq!(c.stop_ids, vec![163586]);
        // 1-indexed [1,2,4] KDA / [3] MLA -> 0-indexed [true,true,false,true].
        assert_eq!(c.is_kda, vec![true, true, false, true]);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_a_layer_covered_by_neither_list() {
        let dir = std::env::temp_dir().join("rabbit_test_kimi_cfg_uncovered_layer");
        fs::create_dir_all(&dir).unwrap();
        // 4 layers, but the lists only cover 1-indexed {1,2,3} -- layer 4 (0-indexed 3) is
        // covered by neither.
        let json = real_shaped_config_json(4).replace(r#""kda_layers": [1, 2, 4]"#, r#""kda_layers": [1, 2]"#);
        write_config(&dir, &json);

        let err = Cfg::load(&dir).unwrap_err();
        assert!(
            matches!(err, ConfigError::LayerTypeCoverage { layer: 3, times_covered: 0 }),
            "expected LayerTypeCoverage {{ layer: 3, times_covered: 0 }}, got {err:?}"
        );

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_a_layer_covered_by_both_lists() {
        let dir = std::env::temp_dir().join("rabbit_test_kimi_cfg_double_covered_layer");
        fs::create_dir_all(&dir).unwrap();
        // layer 3 (1-indexed) now appears in BOTH kda_layers and full_attn_layers.
        let json = real_shaped_config_json(4).replace(r#""kda_layers": [1, 2, 4]"#, r#""kda_layers": [1, 2, 3, 4]"#);
        write_config(&dir, &json);

        let err = Cfg::load(&dir).unwrap_err();
        assert!(
            matches!(err, ConfigError::LayerTypeCoverage { layer: 2, times_covered: 2 }),
            "expected LayerTypeCoverage {{ layer: 2, times_covered: 2 }}, got {err:?}"
        );

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_mla_use_nope_false() {
        let dir = std::env::temp_dir().join("rabbit_test_kimi_cfg_rope_needed");
        fs::create_dir_all(&dir).unwrap();
        let json = real_shaped_config_json(4).replace(r#""mla_use_nope": true"#, r#""mla_use_nope": false"#);
        write_config(&dir, &json);

        let err = Cfg::load(&dir).unwrap_err();
        assert!(matches!(err, ConfigError::RopeNotSupported), "expected RopeNotSupported, got {err:?}");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_a_non_sigmoid_router_activation() {
        let dir = std::env::temp_dir().join("rabbit_test_kimi_cfg_router_activation");
        fs::create_dir_all(&dir).unwrap();
        let json = real_shaped_config_json(4)
            .replace(r#""moe_router_activation_func": "sigmoid""#, r#""moe_router_activation_func": "softmax""#);
        write_config(&dir, &json);

        let err = Cfg::load(&dir).unwrap_err();
        assert!(
            matches!(err, ConfigError::UnsupportedRouterActivation { ref found } if found == "softmax"),
            "expected UnsupportedRouterActivation(\"softmax\"), got {err:?}"
        );

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_missing_linear_attn_config() {
        let dir = std::env::temp_dir().join("rabbit_test_kimi_cfg_missing_lac");
        fs::create_dir_all(&dir).unwrap();
        write_config(&dir, r#"{"model_type": "kimi_linear", "hidden_size": 2304}"#);

        let err = Cfg::load(&dir).unwrap_err();
        assert!(matches!(err, ConfigError::MissingLinearAttnConfig), "expected MissingLinearAttnConfig, got {err:?}");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_a_different_architectures_model_type() {
        let dir = std::env::temp_dir().join("rabbit_test_kimi_cfg_wrong_model_type");
        fs::create_dir_all(&dir).unwrap();
        write_config(&dir, r#"{"model_type": "glm_moe_dsa", "hidden_size": 128}"#);

        let err = Cfg::load(&dir).unwrap_err();
        assert!(
            matches!(err, ConfigError::UnsupportedArchitecture { found: Some(ref m) } if m == "glm_moe_dsa"),
            "expected UnsupportedArchitecture(\"glm_moe_dsa\"), got {err:?}"
        );

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_a_config_with_no_model_type_at_all() {
        let dir = std::env::temp_dir().join("rabbit_test_kimi_cfg_missing_model_type");
        fs::create_dir_all(&dir).unwrap();
        write_config(&dir, r#"{"hidden_size": 2304}"#);

        let err = Cfg::load(&dir).unwrap_err();
        assert!(
            matches!(err, ConfigError::UnsupportedArchitecture { found: None }),
            "expected UnsupportedArchitecture(None), got {err:?}"
        );

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_an_n_group_that_does_not_divide_n_experts_evenly() {
        let dir = std::env::temp_dir().join("rabbit_test_kimi_cfg_ngroup_uneven");
        fs::create_dir_all(&dir).unwrap();
        let json = real_shaped_config_json(4)
            .replace(r#""num_experts": 256"#, r#""num_experts": 100"#)
            .replace(r#""num_expert_group": 1"#, r#""num_expert_group": 3"#);
        write_config(&dir, &json);

        let err = Cfg::load(&dir).unwrap_err();
        assert!(
            matches!(err, ConfigError::InvalidGrouping { n_experts: 100, n_group: 3 }),
            "expected InvalidGrouping {{ n_experts: 100, n_group: 3 }}, got {err:?}"
        );

        fs::remove_dir_all(&dir).ok();
    }
}
