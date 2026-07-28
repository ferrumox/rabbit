//! Port of `convert_fp8_to_int4.py`'s `dequant`/`convert_shard` — per-tensor dequant dispatch
//! (NVFP4 / FP8 block-scale / plain BF16-F16-F32) plus the orchestration that classifies every
//! tensor in ONE shard file, dequantizes the kept ones, and quantizes them per `ConvertOpts`.
//!
//! Deliberately takes a `Shards` opened on a directory holding exactly one physical
//! `.safetensors` file (the disk-safe download-one-shard-at-a-time workflow's natural unit,
//! see `download.rs`) rather than rabbit's own runtime usage (`Shards::open` on a directory of
//! MANY shards at once) — this mirrors Python's `safe_open(one_shard_path)` exactly, including
//! `keys = set(f.keys())`'s role: NVFP4 detection (`(name+"_scale") in keys`) only ever looks
//! for the sidecar WITHIN the same shard, matching how real checkpoints co-locate a tensor with
//! its own scale sidecar (confirmed in this project's own colibrì diff-mining notes).

use crate::glm52::convert::classify::{classify, TensorKind};
use crate::glm52::convert::dequant::{dequant_nvfp4, NvFp4Error};
use crate::glm52::convert::writer::OutTensor;
use crate::quant::{quant_int4_grouped, QTKind, QT};
use crate::safetensors::{DType, SafetensorsError, Shards, Tensor};
use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug)]
pub enum ConvertError {
    Safetensors(SafetensorsError),
    NvFp4(NvFp4Error),
}

impl fmt::Display for ConvertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConvertError::Safetensors(e) => write!(f, "{e}"),
            ConvertError::NvFp4(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ConvertError {}

impl From<SafetensorsError> for ConvertError {
    fn from(e: SafetensorsError) -> Self {
        ConvertError::Safetensors(e)
    }
}

impl From<NvFp4Error> for ConvertError {
    fn from(e: NvFp4Error) -> Self {
        ConvertError::NvFp4(e)
    }
}

/// Per-tensor-class bit-width overrides — port of `--shared-bits`/`--o-bits`/`--kvb-bits`/
/// `--attn-bits`/`--dmlp-bits`. `None` means "fall back to `ConvertOpts::ebits`" (or
/// `io_bits`/`xbits` for the `Io`/`X` buckets, which have no override of their own — same as
/// the Python source, which never added `--io-bits`/`--xbits` overrides to `bits_map`).
#[derive(Default, Clone, Copy)]
pub struct BitsMap {
    pub sh: Option<u8>,
    pub o: Option<u8>,
    pub kvb: Option<u8>,
    pub attn: Option<u8>,
    pub dmlp: Option<u8>,
}

pub struct ConvertOpts {
    pub n_layers: usize,
    pub ebits: u8,
    pub io_bits: u8,
    pub xbits: u8,
    pub keep_mtp: bool,
    pub keep_idx: bool,
    /// 0 = per-row scaling (`quant_int4`/`quant_int2`/`quant_int8`'s existing behavior via
    /// `QT::alloc`/`fill`); >0 = grouped int4 (`quant_int4_grouped`), only when `bits<=4`.
    pub group_size: usize,
    pub bits_map: BitsMap,
    /// `--bits-override <pattern>:<bits>` escape hatch, checked BEFORE `bits_map`/bucket
    /// resolution — `(name_substring, bits)` pairs, first match (in list order) wins. Exists
    /// for exactly the case `resumen-conversor-rabbit.md`'s own design discipline predicted:
    /// don't design classifier buckets for hypothetical tensor classes, add an override once a
    /// REAL one shows up in a `--report` run (e.g. Kimi Linear's `self_attn.b_proj.weight` —
    /// tiny, but consistently the worst-quantized tensor family in a real `--report` run — see
    /// `project_rabbit_multiarch_kimi` memory). Applies regardless of which classifier
    /// (`classify`/`classify_generic`) or bucket a tensor landed in — a raw name match, nothing
    /// architecture-specific.
    pub overrides: Vec<(String, u8)>,
}

/// Resolves the bit-width for a kept (non-skip/consumed) tensor kind — port of `convert_shard`'s
/// bits-resolution logic. The Python source has a second, unreachable `if` after this that can
/// never change the result (every case it re-checks was already handled by the first
/// `if`/`else`) — not ported literally; this function implements the one EFFECTIVE behavior.
fn resolve_bits(kind: TensorKind, opts: &ConvertOpts) -> u8 {
    use TensorKind::*;
    let explicit = match kind {
        Sh => opts.bits_map.sh,
        O => opts.bits_map.o,
        Kvb => opts.bits_map.kvb,
        Attn => opts.bits_map.attn,
        Dmlp => opts.bits_map.dmlp,
        _ => None,
    };
    if let Some(b) = explicit {
        return b;
    }
    match kind {
        Io => opts.io_bits,
        X => opts.xbits,
        _ => opts.ebits,
    }
}

/// `resolve_bits`'s caller-facing entry point: checks `opts.overrides` first (a literal name
/// match beats any bucket-based resolution, first pattern in the list that matches wins), only
/// falling back to bucket/class resolution when nothing overrides this specific tensor.
fn resolve_bits_for(name: &str, kind: TensorKind, opts: &ConvertOpts) -> u8 {
    for (pattern, bits) in &opts.overrides {
        if name.contains(pattern.as_str()) {
            return *bits;
        }
    }
    resolve_bits(kind, opts)
}

/// `true` when `t` is an NVFP4 packed weight: a U8 tensor with a `{name}_scale` sidecar present
/// IN THE SAME `Shards` — port of Python's `dt in ("U8","uint8") and (name+"_scale") in keys`.
fn is_nvfp4(shards: &Shards, t: &Tensor) -> bool {
    t.dtype == DType::U8 && shards.has(&format!("{}_scale", t.name))
}

/// The tensor's REAL logical shape — `t.shape` as declared UNLESS it's an NVFP4 packed weight,
/// whose on-disk shape (`[O, I/2]`, two nibbles per byte) is half its logical `[O, I]` width.
fn logical_shape(shards: &Shards, t: &Tensor) -> Vec<u64> {
    if is_nvfp4(shards, t) {
        vec![t.shape[0], t.shape[1] * 2]
    } else {
        t.shape.clone()
    }
}

/// Port of `dequant(f, name, keys)`: NVFP4 (via `dequant_nvfp4`), FP8 block-scale (via
/// `Shards::read_f32`'s already-existing automatic handling, unchanged), or plain BF16/F16/F32
/// passthrough (same `read_f32` call — it only special-cases `F8E4M3`).
fn dequant_tensor(shards: &Shards, t: &Tensor) -> Result<Vec<f32>, ConvertError> {
    if is_nvfp4(shards, t) {
        let (o, half_i) = (t.shape[0] as usize, t.shape[1] as usize);
        let i = half_i * 2;
        let packed = shards.read_raw(&t.name, false)?;
        let bscale_raw = shards.read_raw(&format!("{}_scale", t.name), false)?;
        let gscale = shards.read_f32(&format!("{}_scale_2", t.name), false)?;
        let gscale = *gscale.first().ok_or_else(|| ConvertError::Safetensors(SafetensorsError::BadHeader(format!("{}_scale_2: empty tensor", t.name))))?;
        Ok(dequant_nvfp4(&packed, o, i, &bscale_raw, gscale)?)
    } else {
        Ok(shards.read_f32(&t.name, false)?)
    }
}

/// Quantizes `w` (`[rows, cols]`, row-major) at `bits`, dispatching to grouped or per-row
/// packing exactly like `convert_shard`'s own `if group_size > 0 and bits <= 4: ... else: ...`.
/// The per-row path reuses `QT::alloc`/`fill` (already a bit-exact port of `quant_int2`/
/// `quant_int4`/`quant_int8` — see `crate::quant`'s module doc) instead of re-implementing it.
fn quantize(w: &[f32], rows: usize, cols: usize, bits: u8, group_size: usize) -> (Vec<u8>, Vec<f32>) {
    if group_size > 0 && bits <= 4 {
        return quant_int4_grouped(w, rows, cols, bits, group_size);
    }
    let mut t = QT::alloc(rows, cols, bits, false);
    t.fill(w);
    match t.kind {
        QTKind::I2 { data, scale } => (data, scale),
        QTKind::I4 { data, scale } => (data, scale),
        // I8's container is `Vec<i8>` (rabbit's own runtime representation); the on-disk U8
        // container just wants the same bytes reinterpreted unsigned -- same two's-complement
        // `as u8` cast Python's `.view(np.uint8)` performs on its own int8 array.
        QTKind::I8 { data, scale } => (data.into_iter().map(|v| v as u8).collect(), scale),
        // QT::alloc only picks F32/MxFp4 for bits>=16 or a dedicated constructor -- unreachable
        // for any `bits` this function is ever called with (1..=15, `convert_shard`'s own range).
        _ => unreachable!("quantize: QT::alloc({rows},{cols},{bits},false) produced an unpacked kind"),
    }
}

/// Wraps this module's own GLM-5.2-specific `classify` (which needs `n_layers`/`keep_mtp`/
/// `keep_idx`, not just a tensor name) into the `Fn(&str, usize) -> TensorKind` shape
/// `convert_shard` wants (the `usize` is the tensor's rank — GLM's own classifier ignores it,
/// `classify_generic` doesn't) — the convenience every GLM-specific caller (this module's own
/// tests, `convert_fp8_to_int4.rs`) reaches for instead of writing the closure inline each time.
pub fn glm_classifier(opts: &ConvertOpts) -> impl Fn(&str, usize) -> TensorKind + '_ {
    move |name, _ndim| classify(name, opts.n_layers, opts.keep_mtp, opts.keep_idx)
}

/// Port of `convert_shard(path, out_dict, ...)`, minus the side-effecting `out_dict` parameter
/// (returns the built map instead — same content, more idiomatic Rust). `shard_dir` must be a
/// directory containing exactly the ONE `.safetensors` shard to convert (see this module's doc).
///
/// `classify_fn` is injected rather than hardcoded to this module's own GLM-5.2-specific
/// `classify` — everything else here (dequant dispatch, bits resolution, quantization) has no
/// GLM-5.2-specific logic at all, so this same function now serves BOTH converters: pass
/// `glm_classifier(opts)` for the literal colibrì port, or `crate::convert::classify::
/// classify_generic` directly (same `Fn(&str, usize) -> TensorKind` shape, no adapter needed)
/// for rabbit's own architecture-agnostic V1 — see `crate::convert`'s module doc.
///
/// `report`, when given, gets one `TensorError` appended per quantized (non-F32/non-skipped)
/// tensor — `--report`'s data source (see `crate::convert::quality`'s doc). `None` costs
/// nothing extra: the dequant/quantize work already happened for the real output either way,
/// `measure_error` is the only added cost, and it's skipped entirely when nobody wants it.
pub fn convert_shard(
    shard_dir: &std::path::Path,
    opts: &ConvertOpts,
    classify_fn: impl Fn(&str, usize) -> TensorKind,
    mut report: Option<&mut crate::convert::quality::QuantizationReport>,
) -> Result<BTreeMap<String, OutTensor>, ConvertError> {
    let shards = Shards::open(shard_dir)?;
    let mut out = BTreeMap::new();

    for t in shards.tensors() {
        let kind = classify_fn(&t.name, t.shape.len());
        if matches!(kind, TensorKind::Skip | TensorKind::Consumed) {
            continue;
        }
        let w = dequant_tensor(&shards, t)?;
        let shape = logical_shape(&shards, t);

        if kind == TensorKind::F32 || shape.len() != 2 {
            out.insert(t.name.clone(), OutTensor::F32 { shape, data: w });
            continue;
        }

        let bits = resolve_bits_for(&t.name, kind, opts);
        let (rows, cols) = (shape[0] as usize, shape[1] as usize);
        let (q, s) = quantize(&w, rows, cols, bits, opts.group_size);
        if let Some(report) = report.as_deref_mut() {
            let rel_error = crate::convert::quality::measure_error(&w, &q, &s, rows, cols, bits, opts.group_size);
            report.per_tensor.push(crate::convert::quality::TensorError { name: t.name.clone(), kind, bits, rel_error });
        }
        out.insert(t.name.clone(), OutTensor::U8 { shape: vec![q.len() as u64], data: q });
        out.insert(format!("{}.qs", t.name), OutTensor::F32 { shape: vec![s.len() as u64], data: s });
    }

    Ok(out)
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

    /// A tiny raw (BF16-free, plain F32-only source for simplicity) synthetic "FP8-checkpoint
    /// shaped" shard: one dense-MLP weight (quantizable), one norm (kept F32), one MTP-layer
    /// tensor (skipped by default), one indexer tensor (skipped by default).
    fn build_fixture(name: &str) -> TempDir {
        let dir = TempDir::new(name);
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        let mut add = |header: &mut serde_json::Map<String, serde_json::Value>, name: &str, dtype: &str, shape: Vec<u64>, bytes: Vec<u8>| {
            let start = data.len() as u64;
            data.extend_from_slice(&bytes);
            let end = data.len() as u64;
            header.insert(name.to_string(), json!({"dtype": dtype, "shape": shape, "data_offsets": [start, end]}));
        };

        // model.layers.0.mlp.gate_proj.weight: [2,4] f32 -> dense MLP (Dmlp bucket, layer 0 <
        // n_layers=1 so first_k_dense_replace doesn't matter here, classify() doesn't look at
        // it directly -- Dmlp is just "any mlp.{gate,up,down}_proj.weight").
        let w = [1.0f32, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0];
        add(&mut header, "model.layers.0.mlp.gate_proj.weight", "F32", vec![2, 4], f32_bytes(&w));
        add(&mut header, "model.layers.0.input_layernorm.weight", "F32", vec![4], f32_bytes(&[1.0, 1.0, 1.0, 1.0]));
        // n_layers=1 -> layer index 1 is the MTP head, skipped by default.
        add(&mut header, "model.layers.1.eh_proj.weight", "F32", vec![2, 2], f32_bytes(&[9.0, 9.0, 9.0, 9.0]));
        add(&mut header, "model.layers.0.self_attn.indexer.wk.weight", "F32", vec![2, 2], f32_bytes(&[5.0, 5.0, 5.0, 5.0]));

        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_bytes);
        out.extend_from_slice(&data);
        fs::write(dir.0.join("shard.safetensors"), out).unwrap();
        dir
    }

    fn base_opts() -> ConvertOpts {
        ConvertOpts { n_layers: 1, ebits: 4, io_bits: 8, xbits: 4, keep_mtp: false, keep_idx: false, group_size: 0, bits_map: BitsMap::default(), overrides: vec![] }
    }

    #[test]
    fn skips_mtp_and_indexer_tensors_by_default() {
        let fixture = build_fixture("rabbit_test_convert_shard_default_skip");
        let out = convert_shard(&fixture.0, &base_opts(), glm_classifier(&base_opts()), None).unwrap();
        assert!(!out.contains_key("model.layers.1.eh_proj.weight"));
        assert!(!out.contains_key("model.layers.0.self_attn.indexer.wk.weight"));
    }

    #[test]
    fn keeps_norm_as_plain_f32_never_quantized() {
        let fixture = build_fixture("rabbit_test_convert_shard_norm_f32");
        let out = convert_shard(&fixture.0, &base_opts(), glm_classifier(&base_opts()), None).unwrap();
        match out.get("model.layers.0.input_layernorm.weight").unwrap() {
            OutTensor::F32 { shape, data } => {
                assert_eq!(shape, &vec![4]);
                assert_eq!(data, &vec![1.0, 1.0, 1.0, 1.0]);
            }
            OutTensor::U8 { .. } => panic!("norm must stay F32"),
        }
    }

    #[test]
    fn quantizes_the_dense_mlp_weight_and_matches_qt_alloc_directly() {
        let fixture = build_fixture("rabbit_test_convert_shard_quantize");
        let out = convert_shard(&fixture.0, &base_opts(), glm_classifier(&base_opts()), None).unwrap();
        let (q, s) = match out.get("model.layers.0.mlp.gate_proj.weight").unwrap() {
            OutTensor::U8 { data, .. } => {
                let scale = match out.get("model.layers.0.mlp.gate_proj.weight.qs").unwrap() {
                    OutTensor::F32 { data, .. } => data.clone(),
                    _ => panic!("scale must be F32"),
                };
                (data.clone(), scale)
            }
            OutTensor::F32 { .. } => panic!("dense MLP weight must be quantized"),
        };

        let w = [1.0f32, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0];
        let mut t = QT::alloc(2, 4, 4, false); // Dmlp bucket has no override -> falls back to ebits=4
        t.fill(&w);
        match t.kind {
            QTKind::I4 { data, scale } => {
                assert_eq!(q, data);
                assert_eq!(s, scale);
            }
            _ => panic!("expected I4"),
        }
    }

    #[test]
    fn dmlp_bits_override_takes_priority_over_ebits() {
        let fixture = build_fixture("rabbit_test_convert_shard_dmlp_override");
        let mut opts = base_opts();
        opts.bits_map.dmlp = Some(8); // -> int8 (QTKind::I8), not the ebits=4 default
        let out = convert_shard(&fixture.0, &opts, glm_classifier(&opts), None).unwrap();
        match out.get("model.layers.0.mlp.gate_proj.weight").unwrap() {
            OutTensor::U8 { data, .. } => assert_eq!(data.len(), 8), // int8 = 1 byte/element, 8 elements
            _ => panic!("expected quantized"),
        }
    }

    #[test]
    fn name_override_wins_over_the_bucket_default() {
        let fixture = build_fixture("rabbit_test_convert_shard_name_override");
        let mut opts = base_opts();
        opts.overrides = vec![("mlp.gate_proj".to_string(), 8)];
        let out = convert_shard(&fixture.0, &opts, glm_classifier(&opts), None).unwrap();
        match out.get("model.layers.0.mlp.gate_proj.weight").unwrap() {
            OutTensor::U8 { data, .. } => assert_eq!(data.len(), 8), // int8 = 1 byte/element, 8 elements
            _ => panic!("expected quantized"),
        }
    }

    #[test]
    fn name_override_wins_even_over_an_explicit_bits_map_entry() {
        let fixture = build_fixture("rabbit_test_convert_shard_name_override_vs_bits_map");
        let mut opts = base_opts();
        opts.bits_map.dmlp = Some(8); // would normally pick int8...
        opts.overrides = vec![("mlp.gate_proj".to_string(), 2)]; // ...but the override wins: int2
        let out = convert_shard(&fixture.0, &opts, glm_classifier(&opts), None).unwrap();
        match out.get("model.layers.0.mlp.gate_proj.weight").unwrap() {
            OutTensor::U8 { data, .. } => assert_eq!(data.len(), 2), // int2 = 4 values/byte, 8 elements -> 2 bytes
            _ => panic!("expected quantized"),
        }
    }

    #[test]
    fn name_override_does_not_affect_a_non_matching_tensor() {
        let fixture = build_fixture("rabbit_test_convert_shard_name_override_scoped");
        let mut opts = base_opts(); // ebits=4 default
        opts.overrides = vec![("some_unrelated_pattern".to_string(), 8)];
        let out = convert_shard(&fixture.0, &opts, glm_classifier(&opts), None).unwrap();
        match out.get("model.layers.0.mlp.gate_proj.weight").unwrap() {
            OutTensor::U8 { data, .. } => assert_eq!(data.len(), 4), // int4 = 2 values/byte, 8 elements -> 4 bytes (ebits=4, unaffected)
            _ => panic!("expected quantized"),
        }
    }

    #[test]
    fn group_size_routes_through_the_grouped_quantizer() {
        let fixture = build_fixture("rabbit_test_convert_shard_grouped");
        let mut opts = base_opts();
        opts.group_size = 2; // cols=4 -> 2 groups of 2
        let out = convert_shard(&fixture.0, &opts, glm_classifier(&opts), None).unwrap();
        let scale = match out.get("model.layers.0.mlp.gate_proj.weight.qs").unwrap() {
            OutTensor::F32 { data, .. } => data.clone(),
            _ => panic!("expected F32 scale"),
        };
        assert_eq!(scale.len(), 4); // rows=2 * ngroups=2, vs. the per-row path's 2 scales
    }

    #[test]
    fn keep_mtp_mode_converts_only_the_mtp_layer() {
        let fixture = build_fixture("rabbit_test_convert_shard_mtp_mode");
        let mut opts = base_opts();
        opts.keep_mtp = true;
        let out = convert_shard(&fixture.0, &opts, glm_classifier(&opts), None).unwrap();
        assert!(out.contains_key("model.layers.1.eh_proj.weight"));
        assert!(!out.contains_key("model.layers.0.mlp.gate_proj.weight")); // a real layer, skipped in --mtp mode
    }

    #[test]
    fn keep_idx_mode_converts_only_indexer_tensors() {
        let fixture = build_fixture("rabbit_test_convert_shard_idx_mode");
        let mut opts = base_opts();
        opts.keep_idx = true;
        let out = convert_shard(&fixture.0, &opts, glm_classifier(&opts), None).unwrap();
        assert!(out.contains_key("model.layers.0.self_attn.indexer.wk.weight"));
        assert!(!out.contains_key("model.layers.0.mlp.gate_proj.weight"));
    }

    #[test]
    fn nvfp4_source_tensor_dequantizes_and_then_quantizes_like_any_other_weight() {
        let dir = TempDir::new("rabbit_test_convert_shard_nvfp4");
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        let mut add = |header: &mut serde_json::Map<String, serde_json::Value>, name: &str, dtype: &str, shape: Vec<u64>, bytes: Vec<u8>| {
            let start = data.len() as u64;
            data.extend_from_slice(&bytes);
            let end = data.len() as u64;
            header.insert(name.to_string(), json!({"dtype": dtype, "shape": shape, "data_offsets": [start, end]}));
        };
        // O=1, I=2 packed into 1 byte: low nibble=5(->3.0), high nibble=2(->1.0). Block scale
        // (1 block, GS=16>I) = 1.0 (0x38). gscale=0.1 (< 1.0, modelopt convention).
        let name = "model.layers.0.mlp.gate_proj.weight";
        add(&mut header, name, "U8", vec![1, 1], vec![(2u8 << 4) | 5u8]);
        add(&mut header, &format!("{name}_scale"), "F8_E4M3", vec![1, 1], vec![0x38]);
        add(&mut header, &format!("{name}_scale_2"), "F32", vec![1], f32_bytes(&[0.1]));

        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out_bytes = Vec::new();
        out_bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out_bytes.extend_from_slice(&header_bytes);
        out_bytes.extend_from_slice(&data);
        fs::write(dir.0.join("shard.safetensors"), out_bytes).unwrap();

        let mut opts = base_opts();
        opts.n_layers = 1;
        let out = convert_shard(&dir.0, &opts, glm_classifier(&opts), None).unwrap();
        // Dequants to [3.0, 1.0] * 1.0 * 0.1 = [0.3, 0.1] (logical shape [1,2]), then quantized
        // at ebits=4 (Dmlp bucket, no override) -> same as feeding that vector through QT
        // directly, cross-checked the same way as the plain-F32 quantize test above.
        let (q, s) = match out.get(name).unwrap() {
            OutTensor::U8 { data, .. } => {
                let scale = match out.get(&format!("{name}.qs")).unwrap() {
                    OutTensor::F32 { data, .. } => data.clone(),
                    _ => panic!("expected F32 scale"),
                };
                (data.clone(), scale)
            }
            OutTensor::F32 { .. } => panic!("expected quantized"),
        };
        let mut t = QT::alloc(1, 2, 4, false);
        t.fill(&[0.3, 0.1]);
        match t.kind {
            QTKind::I4 { data, scale } => {
                assert_eq!(q, data);
                assert_eq!(s, scale);
            }
            _ => panic!("expected I4"),
        }
    }
}
