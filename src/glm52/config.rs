//! Port of `load_cfg` (`glm.c:636` in the reference implementation) — reads config.json into `Cfg`.

use serde_json::Value;
use std::fmt;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct Cfg {
    pub hidden: i32,
    pub n_layers: i32,
    pub n_heads: i32,
    pub n_experts: i32,
    pub topk: i32,
    pub moe_inter: i32,
    pub dense_inter: i32,
    pub first_dense: i32,
    pub q_lora: i32,
    pub kv_lora: i32,
    pub qk_nope: i32,
    pub qk_rope: i32,
    pub qk_head: i32,
    pub v_head: i32,
    pub n_shared: i32,
    pub vocab: i32,
    pub n_group: i32,
    pub topk_group: i32,
    pub norm_topk: bool,
    /// eos_token_id(s) from config — GLM-5.2 has three (endoftext, user, observation).
    pub stop_ids: Vec<i32>,
    pub index_topk: i32,
    pub index_nh: i32,
    pub index_hd: i32,
    /// per layer: true = full (computes the DSA indexer), false = shared (reuses it).
    pub idx_type: Vec<bool>,
    pub eps: f32,
    pub theta: f32,
    pub attn_scale: f32,
    pub routed_scale: f32,
    /// `0` = every resident/expert `.qs` sidecar (if any) is per-row-scaled, the original
    /// format. `>0` = the converter that produced this checkpoint used grouped int4
    /// (`quant_int4_grouped`/`--group-size`), one scale per this many columns instead of one
    /// per row — read once here and threaded to every `qt_load` call rather than guessed from
    /// a `.qs` sidecar's own length, which isn't always exactly invertible back to a group size
    /// (see `QTKind::I4Grouped`'s doc in `quant.rs`).
    pub group_size: i32,
}

/// The only `model_type` this loader recognizes today — rabbit's single supported
/// architecture family (see `src/glm52/mod.rs`'s module doc). Checked before any GLM-specific
/// field is read, so an unrelated checkpoint (a different HF architecture entirely) gets a
/// clear error here instead of silently misparsing into a `Cfg` full of zero-defaulted fields
/// (`gi()` below returns 0 for any missing key) that would only surface as a confusing panic
/// or garbage output much later, deep in the forward pass.
const SUPPORTED_MODEL_TYPE: &str = "glm_moe_dsa";

#[derive(Debug)]
pub enum ConfigError {
    Io(std::io::Error),
    Json(serde_json::Error),
    /// `config.json`'s `model_type` was missing or didn't match `SUPPORTED_MODEL_TYPE`.
    UnsupportedArchitecture { found: Option<String> },
    /// `n_routed_experts` doesn't divide evenly into `n_group` groups — grouped routing
    /// (DeepSeek-V3/GLM/Kimi Linear style: rank groups by their top-2 `choice` sum, keep only
    /// `topk_group` of them, then top-k within survivors) needs equal-size groups.
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
                "config.json's model_type is {model_type:?}, but rabbit only supports {SUPPORTED_MODEL_TYPE:?} (GLM-5.2) today"
            ),
            ConfigError::UnsupportedArchitecture { found: None } => write!(
                f,
                "config.json has no model_type field; rabbit only supports {SUPPORTED_MODEL_TYPE:?} (GLM-5.2) today"
            ),
            ConfigError::InvalidGrouping { n_experts, n_group } => {
                write!(f, "n_routed_experts={n_experts} doesn't divide evenly into n_group={n_group} groups")
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

        let n_layers = gi(&r, "num_hidden_layers");
        let qk_nope = gi(&r, "qk_nope_head_dim");
        let qk_rope = gi(&r, "qk_rope_head_dim");

        let norm_topk = r.get("norm_topk_prob").and_then(Value::as_bool).unwrap_or(false);
        let eps = r.get("rms_norm_eps").and_then(Value::as_f64).unwrap_or(1e-5) as f32;
        let routed_scale = r.get("routed_scaling_factor").and_then(Value::as_f64).unwrap_or(1.0) as f32;
        let theta = r
            .get("rope_parameters")
            .and_then(|rp| rp.get("rope_theta"))
            .and_then(Value::as_f64)
            .unwrap_or(10000.0) as f32;

        // GLM-5.2 has THREE stop tokens (endoftext, user, observation); stopping on only
        // the first means generating 5-10x wasted tokens of invisible junk after turn end.
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

        let index_topk = gi(&r, "index_topk");
        let index_nh = gi(&r, "index_n_heads");
        let index_hd = gi(&r, "index_head_dim");

        // DSA lightning indexer: per-layer type, either an explicit list or a freq/offset
        // formula.
        let indexer_types = r.get("indexer_types").and_then(Value::as_array);
        let freq = gi(&r, "index_topk_freq").max(1);
        let offset = r
            .get("index_skip_topk_offset")
            .and_then(Value::as_f64)
            .map(|v| v as i32)
            .unwrap_or(2);
        let mut idx_type = Vec::with_capacity(n_layers.max(0) as usize);
        for i in 0..n_layers {
            let full = if let Some(types) = indexer_types {
                types
                    .get(i as usize)
                    .and_then(Value::as_str)
                    .map(|s| s == "full")
                    .unwrap_or(false)
            } else {
                let v = (i - offset + 1).max(0);
                v % freq == 0
            };
            idx_type.push(full);
        }

        let qk_head = qk_nope + qk_rope;
        let attn_scale = 1.0 / (qk_head as f32).sqrt();

        // DeepSeek-V3/GLM/Kimi-Linear-style grouped routing: `n_experts` splits into `n_group`
        // equal-size groups, ranked by their own top-2 `choice` sum, before the true top-k runs
        // only within the `topk_group` highest-scoring groups (see `moe.rs::rank_top_grouped`).
        // `n_group` missing or 0 defaults to 1 (matches `index_topk_freq`'s own `.max(1)`
        // convention above) — every real GLM-5.2 checkpoint sets it to 1 explicitly anyway,
        // degenerating to plain top-k with no grouping at all.
        let n_group = gi(&r, "n_group").max(1);
        let n_experts = gi(&r, "n_routed_experts");
        let topk_group = gi(&r, "topk_group").max(1);
        if n_experts % n_group != 0 {
            return Err(ConfigError::InvalidGrouping { n_experts, n_group });
        }

        let c = Cfg {
            hidden: gi(&r, "hidden_size"),
            n_layers,
            n_heads: gi(&r, "num_attention_heads"),
            n_experts,
            topk: gi(&r, "num_experts_per_tok"),
            moe_inter: gi(&r, "moe_intermediate_size"),
            dense_inter: gi(&r, "intermediate_size"),
            first_dense: gi(&r, "first_k_dense_replace"),
            q_lora: gi(&r, "q_lora_rank"),
            kv_lora: gi(&r, "kv_lora_rank"),
            qk_nope,
            qk_rope,
            qk_head,
            v_head: gi(&r, "v_head_dim"),
            n_shared: gi(&r, "n_shared_experts"),
            vocab: gi(&r, "vocab_size"),
            n_group,
            topk_group,
            norm_topk,
            stop_ids,
            index_topk,
            index_nh,
            index_hd,
            idx_type,
            eps,
            theta,
            attn_scale,
            routed_scale,
            group_size: r.get("rabbit_group_size").and_then(Value::as_i64).unwrap_or(0) as i32,
        };

        // config.json comes from untrusted mirrors — hostile dimensions must not get past
        // this point. One choke point protects every downstream allocation.
        check_range!("hidden_size", c.hidden, 1, 1 << 20);
        check_range!("num_hidden_layers", c.n_layers, 1, 128);
        check_range!("num_attention_heads", c.n_heads, 1, 1024);
        check_range!("n_routed_experts", c.n_experts, 1, 4096);
        check_range!("num_experts_per_tok", c.topk, 1, 64);
        check_range!("n_group", c.n_group, 1, c.n_experts);
        check_range!("topk_group", c.topk_group, 1, c.n_group);
        check_range!("moe_intermediate_size", c.moe_inter, 1, 1 << 20);
        check_range!("intermediate_size", c.dense_inter, 1, 1 << 24);
        check_range!("first_k_dense_replace", c.first_dense, 0, c.n_layers);
        check_range!("q_lora_rank", c.q_lora, 0, 1 << 20);
        check_range!("kv_lora_rank", c.kv_lora, 1, 1 << 20);
        check_range!("qk_nope_head_dim", c.qk_nope, 1, 1 << 16);
        check_range!("qk_rope_head_dim", c.qk_rope, 1, 1 << 16);
        check_range!("v_head_dim", c.v_head, 1, 1 << 16);
        check_range!("n_shared_experts", c.n_shared, 0, 64);
        check_range!("vocab_size", c.vocab, 1, 1 << 24);
        check_range!("index_topk", c.index_topk, 0, 1 << 20);
        check_range!("index_n_heads", c.index_nh, 0, 1024);
        check_range!("index_head_dim", c.index_hd, 0, 1 << 16);
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

    #[test]
    fn loads_minimal_config_with_defaults() {
        let dir = std::env::temp_dir().join("rabbit_test_cfg_minimal");
        fs::create_dir_all(&dir).unwrap();
        write_config(
            &dir,
            r#"{
                "model_type": "glm_moe_dsa",
                "hidden_size": 128, "num_hidden_layers": 5, "num_attention_heads": 4,
                "n_routed_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
                "intermediate_size": 64, "first_k_dense_replace": 3, "q_lora_rank": 64,
                "kv_lora_rank": 32, "qk_nope_head_dim": 24, "qk_rope_head_dim": 8,
                "v_head_dim": 32, "n_shared_experts": 1, "vocab_size": 256, "n_group": 1,
                "topk_group": 1, "index_topk": 4096, "index_n_heads": 2, "index_head_dim": 16
            }"#,
        );

        let c = Cfg::load(&dir).unwrap();
        assert_eq!(c.hidden, 128);
        assert_eq!(c.n_layers, 5);
        assert_eq!(c.qk_head, 32);
        assert!((c.attn_scale - (1.0f32 / 32f32.sqrt())).abs() < 1e-6);
        // no rms_norm_eps/routed_scaling_factor/rope_parameters given -> defaults
        assert!((c.eps - 1e-5).abs() < 1e-9);
        assert!((c.routed_scale - 1.0).abs() < 1e-9);
        assert!((c.theta - 10000.0).abs() < 1e-6);
        assert!(c.stop_ids.is_empty());
        assert_eq!(c.idx_type.len(), 5);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn reads_eos_array_and_explicit_indexer_types() {
        let dir = std::env::temp_dir().join("rabbit_test_cfg_eos_idx");
        fs::create_dir_all(&dir).unwrap();
        write_config(
            &dir,
            r#"{
                "model_type": "glm_moe_dsa",
                "hidden_size": 128, "num_hidden_layers": 3, "num_attention_heads": 4,
                "n_routed_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
                "intermediate_size": 64, "first_k_dense_replace": 1, "q_lora_rank": 64,
                "kv_lora_rank": 32, "qk_nope_head_dim": 24, "qk_rope_head_dim": 8,
                "v_head_dim": 32, "n_shared_experts": 1, "vocab_size": 256, "n_group": 1,
                "topk_group": 1, "index_topk": 4096, "index_n_heads": 2, "index_head_dim": 16,
                "eos_token_id": [151329, 151336, 151338],
                "indexer_types": ["full", "shared", "full"],
                "rms_norm_eps": 1e-6, "routed_scaling_factor": 2.5,
                "rope_parameters": {"rope_type": "default", "rope_theta": 500000.0}
            }"#,
        );

        let c = Cfg::load(&dir).unwrap();
        assert_eq!(c.stop_ids, vec![151329, 151336, 151338]);
        assert_eq!(c.idx_type, vec![true, false, true]);
        assert!((c.eps - 1e-6).abs() < 1e-12);
        assert!((c.routed_scale - 2.5).abs() < 1e-9);
        assert!((c.theta - 500000.0).abs() < 1e-3);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_out_of_range_field() {
        let dir = std::env::temp_dir().join("rabbit_test_cfg_bad_range");
        fs::create_dir_all(&dir).unwrap();
        write_config(
            &dir,
            r#"{
                "model_type": "glm_moe_dsa",
                "hidden_size": 0, "num_hidden_layers": 5, "num_attention_heads": 4,
                "n_routed_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
                "intermediate_size": 64, "first_k_dense_replace": 3, "q_lora_rank": 64,
                "kv_lora_rank": 32, "qk_nope_head_dim": 24, "qk_rope_head_dim": 8,
                "v_head_dim": 32, "n_shared_experts": 1, "vocab_size": 256, "n_group": 1,
                "topk_group": 1, "index_topk": 4096, "index_n_heads": 2, "index_head_dim": 16
            }"#,
        );

        let err = Cfg::load(&dir).unwrap_err();
        matches!(err, ConfigError::OutOfRange { name: "hidden_size", .. });

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn loads_a_valid_n_group_greater_than_one() {
        // 8 experts split into 2 groups of 4 — a real grouped-routing config (Kimi Linear's
        // own shape, not just GLM-5.2's degenerate n_group=1), must load cleanly now.
        let dir = std::env::temp_dir().join("rabbit_test_cfg_ngroup_valid");
        fs::create_dir_all(&dir).unwrap();
        write_config(
            &dir,
            r#"{
                "model_type": "glm_moe_dsa",
                "hidden_size": 128, "num_hidden_layers": 5, "num_attention_heads": 4,
                "n_routed_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
                "intermediate_size": 64, "first_k_dense_replace": 3, "q_lora_rank": 64,
                "kv_lora_rank": 32, "qk_nope_head_dim": 24, "qk_rope_head_dim": 8,
                "v_head_dim": 32, "n_shared_experts": 1, "vocab_size": 256, "n_group": 2,
                "topk_group": 1, "index_topk": 4096, "index_n_heads": 2, "index_head_dim": 16
            }"#,
        );

        let c = Cfg::load(&dir).unwrap();
        assert_eq!(c.n_group, 2);
        assert_eq!(c.topk_group, 1);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_an_n_group_that_does_not_divide_n_experts_evenly() {
        let dir = std::env::temp_dir().join("rabbit_test_cfg_ngroup_uneven");
        fs::create_dir_all(&dir).unwrap();
        write_config(
            &dir,
            r#"{
                "model_type": "glm_moe_dsa",
                "hidden_size": 128, "num_hidden_layers": 5, "num_attention_heads": 4,
                "n_routed_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
                "intermediate_size": 64, "first_k_dense_replace": 3, "q_lora_rank": 64,
                "kv_lora_rank": 32, "qk_nope_head_dim": 24, "qk_rope_head_dim": 8,
                "v_head_dim": 32, "n_shared_experts": 1, "vocab_size": 256, "n_group": 3,
                "topk_group": 1, "index_topk": 4096, "index_n_heads": 2, "index_head_dim": 16
            }"#,
        );

        let err = Cfg::load(&dir).unwrap_err();
        assert!(
            matches!(err, ConfigError::InvalidGrouping { n_experts: 8, n_group: 3 }),
            "expected InvalidGrouping {{ n_experts: 8, n_group: 3 }}, got {err:?}"
        );

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_a_different_architectures_model_type() {
        let dir = std::env::temp_dir().join("rabbit_test_cfg_wrong_model_type");
        fs::create_dir_all(&dir).unwrap();
        write_config(&dir, r#"{"model_type": "kimi_k3", "hidden_size": 128}"#);

        let err = Cfg::load(&dir).unwrap_err();
        assert!(
            matches!(err, ConfigError::UnsupportedArchitecture { found: Some(ref m) } if m == "kimi_k3"),
            "expected UnsupportedArchitecture(\"kimi_k3\"), got {err:?}"
        );

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_a_config_with_no_model_type_at_all() {
        let dir = std::env::temp_dir().join("rabbit_test_cfg_missing_model_type");
        fs::create_dir_all(&dir).unwrap();
        write_config(&dir, r#"{"hidden_size": 128}"#);

        let err = Cfg::load(&dir).unwrap_err();
        assert!(
            matches!(err, ConfigError::UnsupportedArchitecture { found: None }),
            "expected UnsupportedArchitecture(None), got {err:?}"
        );

        fs::remove_dir_all(&dir).ok();
    }
}
