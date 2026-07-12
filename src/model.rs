//! Port of `model_init`/`qt_load`/`qt_from_disk`/`embed_row`/`ld` — loads every
//! dense-resident tensor (attention, DSA indexer, dense MLP or MoE router+shared-expert,
//! embed/lm_head/norms) into `QT`/`Vec<f32>`. Routed expert weights are NOT loaded here: this
//! stage only reaches expert *offsets* (Fase 5's `expert_load`/`expert_cache.rs` reads them on
//! demand from disk), matching the streaming architecture's whole reason to exist.
//!
//! `qt_load` mirrors `qt_from_disk`'s fast path: if `{name}.qs` exists, `name` already holds
//! our own packed int8/int4/int2 bytes (written by an external converter — `tools/convert.rs`,
//! or the community's pre-converted checkpoints for colibri itself) and `.qs` holds the
//! per-row F32 scale, so the tensor is wrapped via `QT::from_packed` with no quantization
//! work at load time. Otherwise it falls back to the plain path (read as f32/bf16/f16/FP8,
//! quantize at load) — the only branch the tiny oracle (no `.qs` siblings) ever exercises.

use crate::config::{Cfg, ConfigError};
use crate::quant::{PackedFormatError, QT};
use crate::safetensors::{SafetensorsError, Shards};
use std::fmt;
use std::path::Path;

pub struct AttnWeights {
    pub q_a: QT,
    pub q_a_ln: Vec<f32>,
    pub q_b: QT,
    pub kv_a: QT,
    pub kv_a_ln: Vec<f32>,
    pub kv_b: QT,
    pub o: QT,
}

/// DSA lightning indexer weights — present only on "full" indexer layers (`cfg.idx_type[i]`);
/// "shared" layers reuse a full layer's selection at forward time instead of having their own.
pub struct DsaWeights {
    pub wq_b: QT,
    pub wk: QT,
    pub weights_proj: QT,
    pub k_norm_w: Vec<f32>,
    pub k_norm_b: Vec<f32>,
}

pub struct DenseMlpWeights {
    pub gate_proj: QT,
    pub up_proj: QT,
    pub down_proj: QT,
}

/// Router + shared expert only. Routed experts live on disk until Fase 5's expert cache reads
/// them; the union of an inference batch's active experts is looked up by id at forward time.
pub struct MoeWeights {
    pub router: Vec<f32>,
    pub router_bias: Vec<f32>,
    pub sh_gate: QT,
    pub sh_up: QT,
    pub sh_down: QT,
}

pub enum Ffn {
    Dense(DenseMlpWeights),
    Moe(MoeWeights),
}

pub struct Layer {
    pub in_ln: Vec<f32>,
    pub post_ln: Vec<f32>,
    pub attn: AttnWeights,
    pub dsa: Option<DsaWeights>,
    pub ffn: Ffn,
}

pub struct Model {
    pub cfg: Cfg,
    pub embed: QT,
    pub lm_head: QT,
    pub final_norm: Vec<f32>,
    pub layers: Vec<Layer>,
    /// whether the DSA indexer is active model-wide — `false` unless every "full" indexer
    /// layer has its weights on disk, matching `model_init`'s all-or-nothing auto-detection
    /// (a partially `--indexer`-converted model must not silently mix sparse/dense layers).
    pub has_dsa: bool,
    /// quantization bits for routed experts, loaded on demand by `expert_cache.rs` — stored
    /// here (like `m->ebits` in the C) so it travels with the model instead of needing to be
    /// threaded through every call site that might load an expert.
    pub ebits: u8,
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

/// `pub(crate)`: `expert_cache.rs` reuses this to load a routed expert's gate/up/down tensors
/// the same way every other dense-resident tensor is loaded.
pub(crate) fn qt_load(shards: &Shards, name: &str, rows: usize, cols: usize, bits: u8) -> Result<QT, ModelError> {
    let qs_name = format!("{name}.qs");
    if shards.has(&qs_name) {
        let data = shards.read_raw(name, false)?;
        let scale = shards.read_f32(&qs_name, false)?;
        return QT::from_packed(rows, cols, bits, data, scale)
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

/// `model_init`'s `has_dsa` auto-detection: numeric preconditions from config, AND every
/// "full" indexer layer's weight tensors must actually be present on disk.
fn detect_has_dsa(cfg: &Cfg, shards: &Shards) -> bool {
    if !(cfg.index_topk > 0 && cfg.index_nh > 0 && cfg.index_hd > 0 && cfg.index_hd <= 256) {
        return false;
    }
    for i in 0..cfg.n_layers as usize {
        if !cfg.idx_type[i] {
            continue;
        }
        let name = format!("model.layers.{i}.self_attn.indexer.wq_b.weight");
        if !shards.has(&name) {
            return false;
        }
    }
    true
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
        let q_lora = cfg.q_lora as usize;
        let kv_lora = cfg.kv_lora as usize;

        // embed/lm_head are the I/O boundary: kept at high precision (like real dynamic
        // quant does) unless dbits is already aggressive, in which case there's no point
        // keeping them wider than the rest of the resident weights.
        let io_bits = if dbits >= 8 { 16 } else { dbits };
        let embed = qt_load(&shards, "model.embed_tokens.weight", cfg.vocab as usize, d, io_bits)?;
        let lm_head = qt_load(&shards, "lm_head.weight", cfg.vocab as usize, d, io_bits)?;
        let final_norm = ld(&shards, "model.norm.weight")?;

        let has_dsa = detect_has_dsa(&cfg, &shards);

        let mut layers = Vec::with_capacity(cfg.n_layers as usize);
        for i in 0..cfg.n_layers as usize {
            let p = |s: &str| format!("model.layers.{i}.{s}");

            let in_ln = ld(&shards, &p("input_layernorm.weight"))?;
            let post_ln = ld(&shards, &p("post_attention_layernorm.weight"))?;

            let attn = AttnWeights {
                q_a: qt_load(&shards, &p("self_attn.q_a_proj.weight"), q_lora, d, dbits)?,
                q_a_ln: ld(&shards, &p("self_attn.q_a_layernorm.weight"))?,
                q_b: qt_load(&shards, &p("self_attn.q_b_proj.weight"), h * qh, q_lora, dbits)?,
                kv_a: qt_load(&shards, &p("self_attn.kv_a_proj_with_mqa.weight"), kv_lora + qk_rope, d, dbits)?,
                kv_a_ln: ld(&shards, &p("self_attn.kv_a_layernorm.weight"))?,
                kv_b: qt_load(&shards, &p("self_attn.kv_b_proj.weight"), h * (qk_nope + v_head), kv_lora, dbits)?,
                o: qt_load(&shards, &p("self_attn.o_proj.weight"), d, h * v_head, dbits)?,
            };

            let sparse = i >= cfg.first_dense as usize;
            let ffn = if !sparse {
                Ffn::Dense(DenseMlpWeights {
                    gate_proj: qt_load(&shards, &p("mlp.gate_proj.weight"), cfg.dense_inter as usize, d, dbits)?,
                    up_proj: qt_load(&shards, &p("mlp.up_proj.weight"), cfg.dense_inter as usize, d, dbits)?,
                    down_proj: qt_load(&shards, &p("mlp.down_proj.weight"), d, cfg.dense_inter as usize, dbits)?,
                })
            } else {
                let s_i = (cfg.moe_inter * cfg.n_shared) as usize;
                Ffn::Moe(MoeWeights {
                    router: ld(&shards, &p("mlp.gate.weight"))?,
                    router_bias: ld(&shards, &p("mlp.gate.e_score_correction_bias"))?,
                    sh_gate: qt_load(&shards, &p("mlp.shared_experts.gate_proj.weight"), s_i, d, dbits)?,
                    sh_up: qt_load(&shards, &p("mlp.shared_experts.up_proj.weight"), s_i, d, dbits)?,
                    sh_down: qt_load(&shards, &p("mlp.shared_experts.down_proj.weight"), d, s_i, dbits)?,
                })
            };

            let dsa = if has_dsa && cfg.idx_type[i] {
                let ip = |s: &str| format!("model.layers.{i}.self_attn.indexer.{s}");
                Some(DsaWeights {
                    wq_b: qt_load(&shards, &ip("wq_b.weight"), (cfg.index_nh * cfg.index_hd) as usize, q_lora, dbits)?,
                    wk: qt_load(&shards, &ip("wk.weight"), cfg.index_hd as usize, d, dbits)?,
                    weights_proj: qt_load(&shards, &ip("weights_proj.weight"), cfg.index_nh as usize, d, dbits)?,
                    k_norm_w: ld(&shards, &ip("k_norm.weight"))?,
                    k_norm_b: ld(&shards, &ip("k_norm.bias"))?,
                })
            } else {
                None
            };

            layers.push(Layer { in_ln, post_ln, attn, dsa, ffn });
        }

        Ok(Model { cfg, embed, lm_head, final_norm, layers, has_dsa, ebits })
    }

    /// Dequantizes token `tok`'s embedding row into `x[hidden]`.
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

    /// A tiny, hand-built 2-layer model (layer 0 dense, layer 1 MoE+DSA) exercising every
    /// tensor `Model::load` needs to find, at shapes small enough to hand-check.
    struct TinyFixture {
        dir: TempDir,
        cfg_json: serde_json::Value,
        tensors: Vec<(String, String, Vec<usize>, Vec<u8>)>,
        seed: u32,
    }

    impl TinyFixture {
        // each fixture needs its own directory: tests run in parallel, and a shared fixed
        // name means one test's fs::write/Drop races another's mid-read (surfaced as
        // "short read" / "MissingTensor" failures that varied nondeterministically by run).
        fn new(name: &str) -> Self {
            TinyFixture {
                dir: TempDir::new(&format!("rabbit_test_model_tiny_{name}")),
                cfg_json: json!({}),
                tensors: Vec::new(),
                seed: 1,
            }
        }

        fn add(&mut self, name: &str, rows: usize, cols: usize) {
            // cols==0 means "1D tensor of length rows", not "0 columns".
            let n = if cols == 0 { rows } else { rows * cols };
            let data = random_row(n, &mut self.seed);
            let shape = if cols == 0 { vec![rows] } else { vec![rows, cols] };
            self.tensors.push((name.to_string(), "F32".to_string(), shape, f32_bytes(&data)));
        }

        fn build(mut self, has_dsa_layer1: bool) -> TempDir {
            let d = 8; // hidden
            let h = 2; // heads
            let qk_nope = 3;
            let qk_rope = 2;
            let qh = qk_nope + qk_rope;
            let v_head = 4;
            let q_lora = 6;
            let kv_lora = 5;
            let vocab = 16;
            let dense_inter = 10;
            let n_experts = 4;
            let moe_inter = 6;
            let n_shared = 1;
            let index_nh = 2;
            let index_hd = 3;

            self.add("model.embed_tokens.weight", vocab, d);
            self.add("lm_head.weight", vocab, d);
            self.add("model.norm.weight", d, 0);

            for i in 0..2 {
                let p = |s: &str| format!("model.layers.{i}.{s}");
                self.add(&p("input_layernorm.weight"), d, 0);
                self.add(&p("post_attention_layernorm.weight"), d, 0);
                self.add(&p("self_attn.q_a_proj.weight"), q_lora, d);
                self.add(&p("self_attn.q_a_layernorm.weight"), q_lora, 0);
                self.add(&p("self_attn.q_b_proj.weight"), h * qh, q_lora);
                self.add(&p("self_attn.kv_a_proj_with_mqa.weight"), kv_lora + qk_rope, d);
                self.add(&p("self_attn.kv_a_layernorm.weight"), kv_lora, 0);
                self.add(&p("self_attn.kv_b_proj.weight"), h * (qk_nope + v_head), kv_lora);
                self.add(&p("self_attn.o_proj.weight"), d, h * v_head);

                if i == 0 {
                    self.add(&p("mlp.gate_proj.weight"), dense_inter, d);
                    self.add(&p("mlp.up_proj.weight"), dense_inter, d);
                    self.add(&p("mlp.down_proj.weight"), d, dense_inter);
                } else {
                    self.add(&p("mlp.gate.weight"), n_experts, d);
                    self.add(&p("mlp.gate.e_score_correction_bias"), n_experts, 0);
                    let s_i = moe_inter * n_shared;
                    self.add(&p("mlp.shared_experts.gate_proj.weight"), s_i, d);
                    self.add(&p("mlp.shared_experts.up_proj.weight"), s_i, d);
                    self.add(&p("mlp.shared_experts.down_proj.weight"), d, s_i);

                    if has_dsa_layer1 {
                        let ip = |s: &str| format!("model.layers.{i}.self_attn.indexer.{s}");
                        self.add(&ip("wq_b.weight"), index_nh * index_hd, q_lora);
                        self.add(&ip("wk.weight"), index_hd, d);
                        self.add(&ip("weights_proj.weight"), index_nh, d);
                        self.add(&ip("k_norm.weight"), index_hd, 0);
                        self.add(&ip("k_norm.bias"), index_hd, 0);
                    }
                }
            }

            self.cfg_json = json!({
                "hidden_size": d, "num_hidden_layers": 2, "num_attention_heads": h,
                "n_routed_experts": n_experts, "num_experts_per_tok": 2,
                "moe_intermediate_size": moe_inter, "intermediate_size": dense_inter,
                "first_k_dense_replace": 1, "q_lora_rank": q_lora, "kv_lora_rank": kv_lora,
                "qk_nope_head_dim": qk_nope, "qk_rope_head_dim": qk_rope, "v_head_dim": v_head,
                "n_shared_experts": n_shared, "vocab_size": vocab, "n_group": 1, "topk_group": 1,
                "index_topk": if has_dsa_layer1 { 4096 } else { 0 },
                "index_n_heads": index_nh, "index_head_dim": index_hd,
                "indexer_types": ["shared", "full"],
            });
            fs::write(self.dir.0.join("config.json"), self.cfg_json.to_string()).unwrap();

            let mut header = serde_json::Map::new();
            header.insert("__metadata__".to_string(), json!({"format": "rabbit-test"}));
            let mut data = Vec::new();
            for (name, dtype, shape, bytes) in &self.tensors {
                let start = data.len() as u64;
                data.extend_from_slice(bytes);
                let end = data.len() as u64;
                header.insert(
                    name.clone(),
                    json!({"dtype": dtype, "shape": shape, "data_offsets": [start, end]}),
                );
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
    fn loads_dense_and_moe_layers_with_correct_shapes() {
        let fixture = TinyFixture::new("shapes").build(true);
        let m = Model::load(&fixture.0, 32, 32).unwrap();

        assert_eq!(m.layers.len(), 2);
        assert!(matches!(m.layers[0].ffn, Ffn::Dense(_)));
        assert!(matches!(m.layers[1].ffn, Ffn::Moe(_)));

        let attn = &m.layers[0].attn;
        assert_eq!(attn.q_a.rows, 6);
        assert_eq!(attn.q_a.cols, 8);
        assert_eq!(attn.q_b.rows, 2 * 5); // H * qh = 2*(qk_nope+qk_rope) = 2*5
        assert_eq!(attn.kv_b.rows, 2 * (3 + 4)); // H*(qk_nope+v_head)

        if let Ffn::Moe(moe) = &m.layers[1].ffn {
            assert_eq!(moe.router.len(), 4 * 8); // n_experts * hidden
            assert_eq!(moe.sh_gate.rows, 6); // moe_inter * n_shared
        } else {
            panic!("expected MoE layer");
        }
    }

    #[test]
    fn detects_dsa_only_when_all_full_layers_have_indexer_weights() {
        let fixture_with = TinyFixture::new("dsa_with").build(true);
        let m_with = Model::load(&fixture_with.0, 32, 32).unwrap();
        assert!(m_with.has_dsa);
        assert!(m_with.layers[0].dsa.is_none()); // layer 0 is "shared" type
        assert!(m_with.layers[1].dsa.is_some()); // layer 1 is "full" type

        let fixture_without = TinyFixture::new("dsa_without").build(false);
        let m_without = Model::load(&fixture_without.0, 32, 32).unwrap();
        assert!(!m_without.has_dsa);
        assert!(m_without.layers[1].dsa.is_none());
    }

    #[test]
    fn embed_row_dequantizes_the_right_row_at_full_precision() {
        let fixture = TinyFixture::new("embed_row").build(true);
        let m = Model::load(&fixture.0, 32, 32).unwrap(); // dbits=32 -> F32, exact passthrough

        let embed_tensor = &m.tensors_for_test_only_embed();
        let row2 = m.embed_row(2);
        assert_eq!(row2, embed_tensor[2 * 8..3 * 8]);
    }

    // test-only accessor: re-reads the raw embed tensor bytes to compare against embed_row's
    // output independently of QT's internals.
    impl Model {
        fn tensors_for_test_only_embed(&self) -> Vec<f32> {
            match &self.embed.kind {
                crate::quant::QTKind::F32(data) => data.clone(),
                _ => panic!("expected F32 embed for this test"),
            }
        }
    }

    #[test]
    fn quantized_dbits_still_loads_and_shapes_match() {
        let fixture = TinyFixture::new("quantized_dbits").build(true);
        let m = Model::load(&fixture.0, 8, 8).unwrap(); // int8 dense weights
        assert!(matches!(m.layers[0].attn.q_a.kind, crate::quant::QTKind::I8 { .. }));
        // io_bits = dbits>=8 ? 16 : dbits -> embed/lm_head stay F32 even at dbits=8.
        assert!(matches!(m.embed.kind, crate::quant::QTKind::F32(_)));
    }

    fn write_minimal_safetensors(dir: &std::path::Path, entries: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_string(), json!({"format": "rabbit-test"}));
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in entries {
            let start = data.len() as u64;
            data.extend_from_slice(bytes);
            let end = data.len() as u64;
            header.insert(name.to_string(), json!({"dtype": dtype, "shape": shape, "data_offsets": [start, end]}));
        }
        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_bytes);
        out.extend_from_slice(&data);
        fs::write(dir.join("model.safetensors"), out).unwrap();
    }

    /// `qt_load` must wrap a `.qs`-backed tensor as-is (bit-identical to what's on disk), not
    /// re-quantize it — the whole point of the fast path being *fast*. Hand-picks int8 bytes
    /// that a real `quantize_rows` pass on ANY f32 source would be astronomically unlikely to
    /// reproduce by coincidence, so an exact match is strong evidence no requantization ran.
    #[test]
    fn qt_load_wraps_a_qs_sidecar_without_requantizing() {
        let dir = TempDir::new("rabbit_test_qt_load_qs_int8");
        let rows = 2;
        let cols = 5;
        let packed: Vec<u8> = vec![7, 200, 3, 250, 9, 11, 13, 15, 17, 19]; // arbitrary, not derived from any f32 source
        let scale = [0.5f32, 2.25];
        write_minimal_safetensors(
            &dir.0,
            &[
                ("w.weight", "U8", vec![rows * cols], packed.clone()),
                ("w.weight.qs", "F32", vec![rows], f32_bytes(&scale)),
            ],
        );
        let shards = Shards::open(&dir.0).unwrap();

        let t = qt_load(&shards, "w.weight", rows, cols, 8).unwrap();
        match &t.kind {
            crate::quant::QTKind::I8 { data, scale: got_scale } => {
                let raw_i8: Vec<i8> = packed.iter().map(|&b| b as i8).collect();
                assert_eq!(data, &raw_i8, "packed bytes must pass through unchanged, not be requantized");
                assert_eq!(got_scale, &scale);
            }
            _ => panic!("expected I8 (byte count matches rows*cols)"),
        }
    }

    #[test]
    fn qt_load_falls_back_to_the_plain_path_without_a_qs_sidecar() {
        let dir = TempDir::new("rabbit_test_qt_load_no_qs");
        let rows = 2;
        let cols = 3;
        let w = [1.0f32, -2.0, 3.0, 4.0, -5.0, 6.0];
        write_minimal_safetensors(&dir.0, &[("w.weight", "F32", vec![rows, cols], f32_bytes(&w))]);
        let shards = Shards::open(&dir.0).unwrap();

        let t = qt_load(&shards, "w.weight", rows, cols, 32).unwrap(); // bits=32 -> exact F32 passthrough
        match &t.kind {
            crate::quant::QTKind::F32(data) => assert_eq!(data, &w),
            _ => panic!("expected F32"),
        }
    }

    /// The "native" claim: `qt_load` needs ZERO FP8-specific code of its own — an FP8-sourced
    /// tensor (`F8_E4M3` dtype + `{name}_scale_inv`, the real GLM-5.2-FP8 checkpoint's own
    /// format) loads through the exact same `read_f32` call as F32/BF16/F16 already did, and
    /// gets quantized to `bits` exactly like any other source. This is what actually lets
    /// rabbit read a real checkpoint without a separate conversion step first.
    #[test]
    fn qt_load_reads_a_real_fp8_checkpoint_tensor_natively() {
        let dir = TempDir::new("rabbit_test_qt_load_fp8_native");
        let rows = 2;
        let cols = 3;
        // 1.0, -1.0, 0.0, 1.0, 1.0, -1.0 (see f8e4m3_decodes_known_bit_patterns in
        // safetensors.rs), one block covers this whole tiny tensor.
        let fp8_bytes = vec![0x38u8, 0xB8, 0x00, 0x38, 0x38, 0xB8];
        let scale = 10.0f32;
        write_minimal_safetensors(
            &dir.0,
            &[
                ("w.weight", "F8_E4M3", vec![rows, cols], fp8_bytes),
                ("w.weight_scale_inv", "F32", vec![1, 1], f32_bytes(&[scale])),
            ],
        );
        let shards = Shards::open(&dir.0).unwrap();

        let t = qt_load(&shards, "w.weight", rows, cols, 32).unwrap(); // bits=32 -> F32, exact
        let expected = [1.0f32, -1.0, 0.0, 1.0, 1.0, -1.0].map(|v| v * scale);
        match &t.kind {
            crate::quant::QTKind::F32(data) => assert_eq!(data, &expected),
            _ => panic!("expected F32"),
        }
    }
}
