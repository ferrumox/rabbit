//! Reads `config.json` into `Cfg` for Kimi K3 (`model_type: "kimi_k3"`) — a thin wrapper around
//! `kimi_linear::config::Cfg`, not a from-scratch parser: K3's `config.json` nests
//! `model_type: "kimi_linear"` and every field a standalone Kimi Linear checkpoint has under
//! `text_config`, confirmed against the real `moonshotai/Kimi-K3/config.json` (fetched
//! 2026-07-27, not guessed). `base` is parsed via `kimi_linear::config::Cfg::from_fields` against
//! that nested object directly — zero duplicated field-parsing logic.
//!
//! On top of `base`, this module adds the handful of fields real only on K3-shaped checkpoints
//! (absent/defaulted on the 48B Kimi Linear checkpoint `kimi_linear::config` was written
//! against): `hidden_act`/`activation_situ_beta`/`activation_situ_linear_beta` (drives
//! `kimi_linear::ops::situ_and_mul`, already ported), `routed_expert_hidden_size`/
//! `latent_moe_use_norm` (the latent-MoE down/up-proj wrapper around routed experts — NOT YET
//! WIRED into any model/generate loader, this module only parses the config), `attn_res_block_size`
//! (Attention Residuals — also not yet wired), and two fields that gate real new attention math
//! **not yet ported at all**: `mla_use_output_gate` (an extra `g_proj` + output gate on K3's MLA
//! layers, confirmed via `modeling_kimi_linear.py`'s `KimiMLAAttention.__init__`) and KDA's
//! `use_full_rank_gate`/`gate_lower_bound` (changes the decay-gate projection from the existing
//! low-rank `f_a`/`f_b` pair `kda.rs::decay_gate` assumes to a full-rank `g_proj`, with an
//! optional floor on the gate value — confirmed via `KimiDeltaAttention.__init__`, real semantics
//! not yet fully traced into `KdaState::step`). All five are captured here as plain data so
//! nothing observed in the real config is lost, even though using them correctly is later work
//! (see `~/.claude/plans/giggly-greeting-bentley.md`'s Phase 4 section).

use crate::kimi_linear::config::{Cfg as KimiLinearCfg, ConfigError as KimiLinearConfigError};
use serde_json::Value;
use std::fmt;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct Cfg {
    pub base: KimiLinearCfg,
    /// `true` iff `text_config.hidden_act == "situ"` — when true, `activation_situ_beta` is
    /// guaranteed `Some` (checked at load time) and `kimi_linear::ops::situ_and_mul` is the MLP
    /// gating activation to use instead of plain `swish`-gated SwiGLU.
    pub use_situ_activation: bool,
    pub situ_beta: f32,
    /// `None` when `text_config.activation_situ_linear_beta` is absent — `situ_and_mul`'s own
    /// `linear_beta: Option<f32>` parameter maps directly onto this field.
    pub situ_linear_beta: Option<f32>,
    /// `0` = no latent-MoE wrapper (routed experts operate at `base.hidden` directly, same as
    /// Kimi Linear 48B). `>0` = down-project to this width before routing to experts, up-project
    /// back after (K3's real checkpoint: `3584`, half of `base.hidden`'s `7168`).
    pub routed_expert_hidden: i32,
    /// Only meaningful when `routed_expert_hidden > 0`.
    pub latent_moe_norm: bool,
    /// `0` = no Attention Residuals (plain per-layer residual stream, same as Kimi Linear 48B).
    /// `>0` = the block-pooling mechanism's checkpoint period (K3's real checkpoint: `12`).
    pub attn_res_block: i32,
    /// Gates an extra `g_proj` + output gate on K3's MLA (global attention) layers — real
    /// checkpoint has this `true`; absent on Kimi Linear 48B (defaults `false` there).
    pub mla_output_gate: bool,
    /// KDA's decay-gate projection style: `true` = full-rank `g_proj`, `false` = the existing
    /// low-rank `f_a`/`f_b` pair `kda.rs::decay_gate` already implements. Real K3 checkpoint has
    /// this `true` — **`kda.rs` does not yet support the full-rank path**, this field is parsed
    /// but not yet consumed by any loader.
    pub kda_full_rank_gate: bool,
    /// `None` = no floor on the decay gate (matches Kimi Linear 48B's behavior, which has no
    /// such concept at all). `Some(x)` = clip the gate at `x` (real checkpoint: `-5.0`) — exact
    /// application point not yet traced into `KdaState`/`decay_gate`.
    pub kda_gate_lower_bound: Option<f32>,
    /// `true` iff `text_config.quantization_config.format == "mxfp4-pack-quantized"` — the real
    /// published checkpoint's routed experts (`w1`/`w2`/`w3`) are natively OCP-MXFP4-quantized on
    /// disk (`{tensor}.weight_packed`/`{tensor}.weight_scale` U8 pairs, not a plain `.weight`
    /// float tensor), confirmed against the real `config.json`'s `quant_method: "compressed-tensors"`
    /// (fetched 2026-07-27). `ExpertCaches::new` uses this to pick `ExpertNaming::KimiK3Mxfp4`
    /// (real checkpoint) vs `ExpertNaming::KimiK3` (a plain-float/int4 checkpoint, e.g. every
    /// synthetic test fixture in this crate).
    pub mxfp4_experts: bool,
}

const SUPPORTED_MODEL_TYPE: &str = "kimi_k3";
const SUPPORTED_TEXT_MODEL_TYPE: &str = "kimi_linear";

#[derive(Debug)]
pub enum ConfigError {
    Io(std::io::Error),
    Json(serde_json::Error),
    UnsupportedArchitecture { found: Option<String> },
    /// `text_config` is missing entirely, or present but not a JSON object.
    MissingTextConfig,
    /// `text_config.model_type` was present and not `"kimi_linear"` (or absent — K3's real
    /// checkpoint always sets it, so treat a missing value as a real mismatch, not an
    /// unspecified default, unlike some other optional fields in this same object).
    UnsupportedTextModelType { found: Option<String> },
    /// Propagated from `kimi_linear::config::Cfg::from_fields` — any error parsing/validating
    /// the fields shared with a standalone Kimi Linear checkpoint.
    Base(KimiLinearConfigError),
    /// `hidden_act` is `"situ"` but `activation_situ_beta` is missing — `situ_and_mul` has no
    /// sensible default for its required `beta` parameter (unlike `linear_beta`, which is
    /// genuinely optional in the real formula).
    MissingSituBeta,
    OutOfRange { name: &'static str, value: i64, lo: i64, hi: i64 },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::Io(e) => write!(f, "config.json: {e}"),
            ConfigError::Json(e) => write!(f, "config.json: {e}"),
            ConfigError::UnsupportedArchitecture { found: Some(model_type) } => write!(
                f,
                "config.json's model_type is {model_type:?}, but rabbit's kimi_k3 loader only supports {SUPPORTED_MODEL_TYPE:?}"
            ),
            ConfigError::UnsupportedArchitecture { found: None } => write!(
                f,
                "config.json has no model_type field; rabbit's kimi_k3 loader only supports {SUPPORTED_MODEL_TYPE:?}"
            ),
            ConfigError::MissingTextConfig => {
                write!(f, "config.json has no (valid object) text_config -- required for kimi_k3")
            }
            ConfigError::UnsupportedTextModelType { found: Some(m) } => write!(
                f,
                "config.json's text_config.model_type is {m:?}, but rabbit's kimi_k3 loader only supports {SUPPORTED_TEXT_MODEL_TYPE:?}"
            ),
            ConfigError::UnsupportedTextModelType { found: None } => write!(
                f,
                "config.json's text_config has no model_type field; rabbit's kimi_k3 loader only supports {SUPPORTED_TEXT_MODEL_TYPE:?}"
            ),
            ConfigError::Base(e) => write!(f, "config.json's text_config: {e}"),
            ConfigError::MissingSituBeta => {
                write!(f, "config.json's text_config.hidden_act is \"situ\" but activation_situ_beta is missing")
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

impl From<KimiLinearConfigError> for ConfigError {
    fn from(e: KimiLinearConfigError) -> Self {
        ConfigError::Base(e)
    }
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

        let text_cfg = r.get("text_config").filter(|v| v.is_object()).ok_or(ConfigError::MissingTextConfig)?;

        match text_cfg.get("model_type").and_then(Value::as_str) {
            Some(model_type) if model_type == SUPPORTED_TEXT_MODEL_TYPE => {}
            Some(model_type) => {
                return Err(ConfigError::UnsupportedTextModelType { found: Some(model_type.to_string()) });
            }
            None => return Err(ConfigError::UnsupportedTextModelType { found: None }),
        }

        let base = KimiLinearCfg::from_fields(text_cfg)?;

        let hidden_act = text_cfg.get("hidden_act").and_then(Value::as_str).unwrap_or("");
        let use_situ_activation = hidden_act == "situ";
        let situ_beta = text_cfg.get("activation_situ_beta").and_then(Value::as_f64);
        if use_situ_activation && situ_beta.is_none() {
            return Err(ConfigError::MissingSituBeta);
        }
        let situ_linear_beta = text_cfg.get("activation_situ_linear_beta").and_then(Value::as_f64).map(|v| v as f32);

        let routed_expert_hidden =
            text_cfg.get("routed_expert_hidden_size").and_then(Value::as_f64).unwrap_or(0.0) as i32;
        let latent_moe_norm = text_cfg.get("latent_moe_use_norm").and_then(Value::as_bool).unwrap_or(false);

        let attn_res_block = text_cfg.get("attn_res_block_size").and_then(Value::as_f64).unwrap_or(0.0) as i32;

        let mla_output_gate = text_cfg.get("mla_use_output_gate").and_then(Value::as_bool).unwrap_or(false);

        let lac = text_cfg.get("linear_attn_config");
        let kda_full_rank_gate =
            lac.and_then(|v| v.get("use_full_rank_gate")).and_then(Value::as_bool).unwrap_or(false);
        let kda_gate_lower_bound =
            lac.and_then(|v| v.get("gate_lower_bound")).and_then(Value::as_f64).map(|v| v as f32);

        let mxfp4_experts = text_cfg
            .get("quantization_config")
            .and_then(|q| q.get("format"))
            .and_then(Value::as_str)
            == Some("mxfp4-pack-quantized");

        let c = Cfg {
            base,
            use_situ_activation,
            situ_beta: situ_beta.unwrap_or(0.0) as f32,
            situ_linear_beta,
            routed_expert_hidden,
            latent_moe_norm,
            attn_res_block,
            mla_output_gate,
            kda_full_rank_gate,
            kda_gate_lower_bound,
            mxfp4_experts,
        };

        check_range!("text_config.routed_expert_hidden_size", c.routed_expert_hidden, 0, 1 << 20);
        check_range!("text_config.attn_res_block_size", c.attn_res_block, 0, c.base.n_layers);

        Ok(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_config(dir: &Path, json: &str) {
        fs::write(dir.join("config.json"), json).unwrap();
    }

    /// Condensed from the real moonshotai/Kimi-K3/config.json (fetched 2026-07-27), scaled down
    /// to a tiny 4-layer text_config so kda_layers/full_attn_layers stay hand-checkable — same
    /// shrink kimi_linear::config's own tests apply, plus K3's five real new fields.
    fn real_shaped_config_json(n_layers: i32) -> String {
        format!(
            r#"{{
                "model_type": "kimi_k3",
                "text_config": {{
                    "model_type": "kimi_linear",
                    "hidden_size": 7168, "num_hidden_layers": {n_layers}, "num_attention_heads": 96,
                    "first_k_dense_replace": 1, "q_lora_rank": 1536, "kv_lora_rank": 512,
                    "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
                    "num_experts": 896, "num_experts_per_token": 16, "num_shared_experts": 2,
                    "num_expert_group": 1, "topk_group": 1, "moe_intermediate_size": 3072,
                    "intermediate_size": 33792, "vocab_size": 163840, "moe_renormalize": true,
                    "rms_norm_eps": 1e-05, "routed_scaling_factor": 1.0, "mla_use_nope": true,
                    "moe_router_activation_func": "sigmoid", "eos_token_id": 163586,
                    "hidden_act": "situ", "activation_situ_beta": 4.0,
                    "activation_situ_linear_beta": 25.0,
                    "routed_expert_hidden_size": 3584, "latent_moe_use_norm": true,
                    "attn_res_block_size": 2, "mla_use_output_gate": true,
                    "linear_attn_config": {{
                        "head_dim": 128, "num_heads": 96, "short_conv_kernel_size": 4,
                        "gate_lower_bound": -5.0, "use_full_rank_gate": true,
                        "kda_layers": [1, 2, 4], "full_attn_layers": [3]
                    }},
                    "quantization_config": {{
                        "quant_method": "compressed-tensors", "format": "mxfp4-pack-quantized"
                    }}
                }}
            }}"#
        )
    }

    #[test]
    fn loads_the_real_checkpoints_shape() {
        let dir = std::env::temp_dir().join("rabbit_test_k3_cfg_real_shape");
        fs::create_dir_all(&dir).unwrap();
        write_config(&dir, &real_shaped_config_json(4));

        let c = Cfg::load(&dir).unwrap();
        // Base fields delegate correctly to kimi_linear::config::Cfg::from_fields.
        assert_eq!(c.base.hidden, 7168);
        assert_eq!(c.base.n_layers, 4);
        assert_eq!(c.base.q_lora, 1536, "K3's q_lora_rank is non-null, unlike Kimi Linear 48B");
        assert_eq!(c.base.n_experts, 896);
        assert_eq!(c.base.topk, 16);
        assert_eq!(c.base.n_shared, 2);
        assert_eq!(c.base.is_kda, vec![true, true, false, true]);
        // K3-only fields.
        assert!(c.use_situ_activation);
        assert!((c.situ_beta - 4.0).abs() < 1e-6);
        assert_eq!(c.situ_linear_beta, Some(25.0));
        assert_eq!(c.routed_expert_hidden, 3584);
        assert!(c.latent_moe_norm);
        assert_eq!(c.attn_res_block, 2);
        assert!(c.mla_output_gate);
        assert!(c.kda_full_rank_gate);
        assert_eq!(c.kda_gate_lower_bound, Some(-5.0));
        assert!(c.mxfp4_experts, "real checkpoint's quantization_config.format is mxfp4-pack-quantized");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn defaults_the_five_k3_only_fields_when_absent() {
        // A text_config that omits every K3-only field must load like a plain Kimi Linear
        // checkpoint: no situ activation, no latent-MoE, no attention residuals, no output gate.
        let dir = std::env::temp_dir().join("rabbit_test_k3_cfg_defaults");
        fs::create_dir_all(&dir).unwrap();
        let json = r#"{
            "model_type": "kimi_k3",
            "text_config": {
                "model_type": "kimi_linear",
                "hidden_size": 2304, "num_hidden_layers": 4, "num_attention_heads": 32,
                "first_k_dense_replace": 1, "q_lora_rank": null, "kv_lora_rank": 512,
                "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
                "num_experts": 256, "num_experts_per_token": 8, "num_shared_experts": 1,
                "num_expert_group": 1, "topk_group": 1, "moe_intermediate_size": 1024,
                "intermediate_size": 9216, "vocab_size": 163840, "moe_renormalize": true,
                "rms_norm_eps": 1e-05, "routed_scaling_factor": 2.446, "mla_use_nope": true,
                "moe_router_activation_func": "sigmoid", "eos_token_id": 163586,
                "linear_attn_config": {
                    "head_dim": 128, "num_heads": 32, "short_conv_kernel_size": 4,
                    "kda_layers": [1, 2, 4], "full_attn_layers": [3]
                }
            }
        }"#;
        write_config(&dir, json);

        let c = Cfg::load(&dir).unwrap();
        assert!(!c.use_situ_activation);
        assert_eq!(c.situ_linear_beta, None);
        assert_eq!(c.routed_expert_hidden, 0);
        assert!(!c.latent_moe_norm);
        assert_eq!(c.attn_res_block, 0);
        assert!(!c.mla_output_gate);
        assert!(!c.kda_full_rank_gate);
        assert_eq!(c.kda_gate_lower_bound, None);
        assert!(!c.mxfp4_experts, "no quantization_config at all must not be misread as mxfp4");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_situ_activation_missing_its_beta() {
        let dir = std::env::temp_dir().join("rabbit_test_k3_cfg_missing_situ_beta");
        fs::create_dir_all(&dir).unwrap();
        let json = real_shaped_config_json(4).replace(r#""activation_situ_beta": 4.0,"#, "");
        write_config(&dir, &json);

        let err = Cfg::load(&dir).unwrap_err();
        assert!(matches!(err, ConfigError::MissingSituBeta), "expected MissingSituBeta, got {err:?}");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_a_top_level_model_type_that_is_not_kimi_k3() {
        let dir = std::env::temp_dir().join("rabbit_test_k3_cfg_wrong_top_model_type");
        fs::create_dir_all(&dir).unwrap();
        write_config(&dir, r#"{"model_type": "kimi_linear"}"#);

        let err = Cfg::load(&dir).unwrap_err();
        assert!(
            matches!(err, ConfigError::UnsupportedArchitecture { found: Some(ref m) } if m == "kimi_linear"),
            "expected UnsupportedArchitecture(\"kimi_linear\"), got {err:?}"
        );

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_a_missing_text_config() {
        let dir = std::env::temp_dir().join("rabbit_test_k3_cfg_missing_text_config");
        fs::create_dir_all(&dir).unwrap();
        write_config(&dir, r#"{"model_type": "kimi_k3"}"#);

        let err = Cfg::load(&dir).unwrap_err();
        assert!(matches!(err, ConfigError::MissingTextConfig), "expected MissingTextConfig, got {err:?}");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_a_text_config_with_the_wrong_nested_model_type() {
        let dir = std::env::temp_dir().join("rabbit_test_k3_cfg_wrong_nested_model_type");
        fs::create_dir_all(&dir).unwrap();
        write_config(&dir, r#"{"model_type": "kimi_k3", "text_config": {"model_type": "glm_moe_dsa"}}"#);

        let err = Cfg::load(&dir).unwrap_err();
        assert!(
            matches!(err, ConfigError::UnsupportedTextModelType { found: Some(ref m) } if m == "glm_moe_dsa"),
            "expected UnsupportedTextModelType(\"glm_moe_dsa\"), got {err:?}"
        );

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn propagates_a_base_config_error_from_kimi_linear() {
        // text_config missing linear_attn_config entirely -> kimi_linear::config's own
        // MissingLinearAttnConfig, wrapped in ConfigError::Base.
        let dir = std::env::temp_dir().join("rabbit_test_k3_cfg_propagates_base_error");
        fs::create_dir_all(&dir).unwrap();
        write_config(
            &dir,
            r#"{"model_type": "kimi_k3", "text_config": {"model_type": "kimi_linear", "hidden_size": 7168}}"#,
        );

        let err = Cfg::load(&dir).unwrap_err();
        assert!(
            matches!(err, ConfigError::Base(KimiLinearConfigError::MissingLinearAttnConfig)),
            "expected Base(MissingLinearAttnConfig), got {err:?}"
        );

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_an_attn_res_block_size_larger_than_n_layers() {
        let dir = std::env::temp_dir().join("rabbit_test_k3_cfg_attn_res_out_of_range");
        fs::create_dir_all(&dir).unwrap();
        let json = real_shaped_config_json(4).replace(r#""attn_res_block_size": 2"#, r#""attn_res_block_size": 999"#);
        write_config(&dir, &json);

        let err = Cfg::load(&dir).unwrap_err();
        assert!(
            matches!(err, ConfigError::OutOfRange { name: "text_config.attn_res_block_size", .. }),
            "expected OutOfRange for attn_res_block_size, got {err:?}"
        );

        fs::remove_dir_all(&dir).ok();
    }
}
