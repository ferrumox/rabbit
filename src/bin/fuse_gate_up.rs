//! Repacks an EXISTING (already-quantized) GLM-5.2 checkpoint so every routed expert's
//! `gate_proj`+`up_proj` become one `gate_up_proj` tensor — ROADMAP.md's "Fuse gate_proj +
//! up_proj into one tensor at conversion time" idea. Turns 2 `io_uring` reads + 2 `matmul_qt`
//! calls per expert into 1 of each; `down_proj` is untouched (it depends on gate+up's SiLU'd
//! output via a genuinely separate compute stage, so it can't fuse the same way — see
//! `expert_cache.rs`'s `GateUp` enum and `moe.rs`'s `apply_single_expert` for the runtime side
//! that reads this format).
//!
//! Confirmed against the real checkpoint (`glm-5.2-colibri-int4`) before writing this: a given
//! expert's `gate_proj`/`up_proj` always live in the SAME physical `.safetensors` shard file,
//! byte-contiguous (`up_proj` starts exactly where `gate_proj` ends) — so this can process one
//! shard at a time, same "disk-safe, one file in flight" pattern as `bin/convert.rs`, rather
//! than needing a whole-checkpoint index held open at once.
//!
//! Pure byte-level concatenation, no dequant/requant: an int4/int8 `QT`'s packed `data` is
//! row-major (`[rows, cols_packed]`), and its `.qs` scale is one `f32` per row — concatenating
//! `gate`'s rows then `up`'s rows (both bytes and scale) is EXACTLY the packed representation
//! of the fused `[2*moe_inter, hidden]` tensor, bit-for-bit identical to quantizing that fused
//! matrix from scratch would produce. No accuracy change, ever — this is packaging, not math.
//!
//! Refuses (doesn't guess) rather than mis-handle a format this wasn't built for: gate/up must
//! either both have a `.qs` packed-scale sidecar (the real checkpoint's actual format) or both
//! be plain (no sidecar, same dtype) — an FP8 block-scale sidecar or a gate/up dtype mismatch
//! is an error, not a best-effort attempt.
//!
//! Usage: fuse_gate_up --indir <dir> [--shard-dirs <dir1,dir2,...>] --outdir <dir>

use rabbit::glm52::convert::writer::{write_safetensors, OutTensor};
use rabbit::safetensors::{DType, Shards};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

const USAGE: &str = "usage: fuse_gate_up --indir <dir> [--shard-dirs <dir1,dir2,...>] --outdir <dir>";

struct Args {
    indir: PathBuf,
    shard_dirs: Vec<PathBuf>,
    outdir: PathBuf,
}

fn parse_args() -> Result<Args, String> {
    let mut indir = None;
    let mut shard_dirs = Vec::new();
    let mut outdir = None;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut next = |flag: &str| args.next().ok_or_else(|| format!("{flag} needs a value"));
        match a.as_str() {
            "--indir" => indir = Some(PathBuf::from(next("--indir")?)),
            "--shard-dirs" => shard_dirs = next("--shard-dirs")?.split(',').map(PathBuf::from).collect(),
            "--outdir" => outdir = Some(PathBuf::from(next("--outdir")?)),
            "-h" | "--help" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown argument: {other}\n\n{USAGE}")),
        }
    }

    Ok(Args {
        indir: indir.ok_or_else(|| format!("--indir is required\n\n{USAGE}"))?,
        shard_dirs,
        outdir: outdir.ok_or_else(|| format!("--outdir is required\n\n{USAGE}"))?,
    })
}

/// `model.layers.N.mlp.experts.E.gate_proj.weight` -> `Some("model.layers.N.mlp.experts.E")`.
/// Deliberately strict (exact suffix match) rather than a generic "contains gate_proj" check —
/// a false positive here would silently mis-fuse an unrelated tensor.
fn expert_gate_prefix(name: &str) -> Option<&str> {
    name.strip_suffix(".gate_proj.weight").filter(|prefix| prefix.contains(".mlp.experts."))
}

/// `(packed_or_raw_bytes, qs_scale, dtype)` — `qs_scale` is `None` when there's no `.qs`
/// sidecar (the plain/unquantized case).
type WeightAndScale = (Vec<u8>, Option<Vec<f32>>, DType);

/// Reads `{prefix}.{suffix}.weight` (+ its `.qs` sidecar if present). Errors on an FP8
/// `_scale_inv` sidecar (block-scale, not row-scale — concatenating those isn't a simple byte
/// operation, and this tool was never validated against that case).
fn read_weight_and_scale(shards: &Shards, prefix: &str, suffix: &str) -> Result<WeightAndScale, String> {
    let name = format!("{prefix}.{suffix}.weight");
    let t = shards.find(&name).ok_or_else(|| format!("missing tensor: {name}"))?;
    let dtype = t.dtype;
    let qs_name = format!("{name}.qs");
    if shards.has(&qs_name) {
        let data = shards.read_raw(&name, false).map_err(|e| e.to_string())?;
        let scale = shards.read_f32(&qs_name, false).map_err(|e| e.to_string())?;
        return Ok((data, Some(scale), dtype));
    }
    if shards.has(&format!("{name}_scale_inv")) {
        return Err(format!("{name}: FP8 block-scale sidecar present -- refusing (this tool only handles per-row .qs or plain, see its own doc)"));
    }
    let data = shards.read_raw(&name, false).map_err(|e| e.to_string())?;
    Ok((data, None, dtype))
}

/// `model.layers.N.mlp.experts.E.up_proj.weight` — already folded into `gate_up_proj.weight`
/// when its `gate_proj` sibling is visited, so it must never be copied through separately.
fn is_expert_up_proj_weight(name: &str) -> bool {
    name.contains(".mlp.experts.") && name.ends_with(".up_proj.weight")
}

/// The `.qs` scale sidecar of EITHER `gate_proj` or `up_proj` for a routed expert — its values
/// are already folded into `gate_up_proj.weight.qs`, so (same reasoning as
/// `is_expert_up_proj_weight` above) it must never be copied through separately. Missing this
/// case would silently duplicate the scale data under its old name alongside the new fused one.
fn is_expert_gate_or_up_qs(name: &str) -> bool {
    if !name.contains(".mlp.experts.") {
        return false;
    }
    match name.strip_suffix(".qs") {
        Some(base) => base.ends_with(".gate_proj.weight") || base.ends_with(".up_proj.weight"),
        None => false,
    }
}

/// Roughly one output shard's worth of tensor data, in bytes — matches the real checkpoint's
/// own ~2.6GB shard size closely enough to be a reasonable default; not load-bearing for
/// correctness (any positive value works, this only affects how many output files result).
const SHARD_SIZE_BUDGET: u64 = 2 * 1024 * 1024 * 1024;

/// Approximate on-disk size this tensor will contribute to an output shard — used only to
/// decide when to flush, not for anything correctness-sensitive.
fn approx_bytes(t: &OutTensor) -> u64 {
    match t {
        OutTensor::F32 { data, .. } => (data.len() * 4) as u64,
        OutTensor::U8 { data, .. } => data.len() as u64,
    }
}

/// Fuses every `gate_proj`/`up_proj` pair across the WHOLE checkpoint (`shards` may span many
/// physical files — see this file's module doc for why a given expert's gate/up can straddle
/// an original shard boundary, confirmed on the real checkpoint: 47 of 19200 experts do),
/// passing every other tensor through byte-for-byte unchanged. Writes output shards
/// incrementally (flushing every `SHARD_SIZE_BUDGET` or so) rather than accumulating the whole
/// fused checkpoint in memory at once — output shard boundaries are NOT required to match the
/// input's own, unlike the per-shard-file approach this replaced.
fn fuse_all(shards: &Shards, outdir: &Path) -> Result<usize, String> {
    let mut out: BTreeMap<String, OutTensor> = BTreeMap::new();
    let mut shard_index = 0usize;

    for t in shards.tensors() {
        let is_merged_away = is_expert_up_proj_weight(&t.name) || is_expert_gate_or_up_qs(&t.name);

        if let Some(prefix) = expert_gate_prefix(&t.name) {
            let (gate_data, gate_scale, gate_dtype) = read_weight_and_scale(shards, prefix, "gate_proj")?;
            let (up_data, up_scale, up_dtype) = read_weight_and_scale(shards, prefix, "up_proj")?;
            if gate_dtype != up_dtype {
                return Err(format!("{prefix}: gate_proj dtype {gate_dtype:?} != up_proj dtype {up_dtype:?}, refusing"));
            }
            let fused_name = format!("{prefix}.gate_up_proj.weight");
            match (gate_scale, up_scale) {
                (Some(gs), Some(us)) => {
                    let mut data = gate_data;
                    data.extend(up_data);
                    let mut scale = gs;
                    scale.extend(us);
                    out.insert(fused_name.clone(), OutTensor::U8 { shape: vec![data.len() as u64], data });
                    out.insert(format!("{fused_name}.qs"), OutTensor::F32 { shape: vec![scale.len() as u64], data: scale });
                }
                (None, None) => {
                    // Plain (unquantized) gate/up: still row-major, still safe to concatenate
                    // raw bytes -- just re-declare the logical 2D shape (2x the row count)
                    // instead of the flat byte-count shape the packed branch above uses.
                    let rows = t.shape.first().copied().unwrap_or(0);
                    let cols = t.shape.get(1).copied().unwrap_or(0);
                    let mut data = gate_data;
                    data.extend(up_data);
                    match gate_dtype {
                        DType::F32 => {
                            let vals = Shards::decode_f32(&data, DType::F32);
                            out.insert(fused_name, OutTensor::F32 { shape: vec![2 * rows, cols], data: vals });
                        }
                        other => return Err(format!("{prefix}: unquantized gate/up in unsupported dtype {other:?}, refusing")),
                    }
                }
                _ => return Err(format!("{prefix}: gate_proj/up_proj .qs presence mismatch, refusing")),
            }
        } else if !is_merged_away {
            // Not a gate_proj tensor and not an up_proj/up_proj.qs already folded into a
            // gate_up_proj above: copy through unchanged.
            copy_tensor_unchanged(shards, t, &mut out)?;
        }

        let out_bytes: u64 = out.values().map(approx_bytes).sum();
        if out_bytes >= SHARD_SIZE_BUDGET {
            flush_shard(&mut out, outdir, shard_index)?;
            shard_index += 1;
        }
    }
    if !out.is_empty() {
        flush_shard(&mut out, outdir, shard_index)?;
        shard_index += 1;
    }
    Ok(shard_index)
}

fn flush_shard(out: &mut BTreeMap<String, OutTensor>, outdir: &Path, shard_index: usize) -> Result<(), String> {
    let path = outdir.join(format!("out-{shard_index:05}.safetensors"));
    write_safetensors(&path, out).map_err(|e| e.to_string())?;
    println!("[shard {shard_index}] wrote {} ({} tensors)", path.file_name().unwrap().to_string_lossy(), out.len());
    out.clear();
    Ok(())
}

fn copy_tensor_unchanged(shards: &Shards, t: &rabbit::safetensors::Tensor, out: &mut BTreeMap<String, OutTensor>) -> Result<(), String> {
    match t.dtype {
        DType::F32 => {
            let data = shards.read_f32(&t.name, false).map_err(|e| e.to_string())?;
            out.insert(t.name.clone(), OutTensor::F32 { shape: t.shape.clone(), data });
        }
        _ => {
            let data = shards.read_raw(&t.name, false).map_err(|e| e.to_string())?;
            out.insert(t.name.clone(), OutTensor::U8 { shape: t.shape.clone(), data });
        }
    }
    Ok(())
}

fn run(a: &Args) -> Result<(), String> {
    fs::create_dir_all(&a.outdir).map_err(|e| e.to_string())?;
    let mut dirs = vec![a.indir.clone()];
    dirs.extend(a.shard_dirs.iter().cloned());
    let shards = Shards::open_multi(&dirs).map_err(|e| e.to_string())?;

    let n_out_shards = fuse_all(&shards, &a.outdir)?;

    for extra in ["config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"] {
        let src = a.indir.join(extra);
        if src.is_file() {
            fs::copy(&src, a.outdir.join(extra)).map_err(|e| e.to_string())?;
        }
    }

    println!("DONE: {n_out_shards} output shards -> {}", a.outdir.display());
    Ok(())
}

fn main() -> std::process::ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    match run(&args) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rabbit::quant::{QT, QTKind};

    struct TempDir(PathBuf);
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

    /// Builds a real-shaped source shard: 2 experts' `gate_proj`/`up_proj`/`down_proj` (int4,
    /// per-row `.qs` scale — the real checkpoint's actual format) plus one unrelated F32 norm,
    /// to prove passthrough tensors survive untouched alongside the fused ones.
    fn build_source_shard(name: &str, n_experts: usize, moe_inter: usize, hidden: usize) -> TempDir {
        let dir = TempDir::new(name);
        let mut seed = 3u32;
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_string(), serde_json::json!({"format": "rabbit-test"}));
        let mut data = Vec::new();
        let mut push_qt = |header: &mut serde_json::Map<String, serde_json::Value>, name: String, rows: usize, cols: usize, seed: &mut u32| {
            let vals: Vec<f32> = (0..rows * cols).map(|_| xorshift(seed)).collect();
            let mut t = QT::alloc(rows, cols, 4, false);
            t.fill(&vals);
            let QTKind::I4 { data: packed, scale } = &t.kind else { unreachable!() };
            let start = data.len() as u64;
            data.extend_from_slice(packed);
            let end = data.len() as u64;
            header.insert(name.clone(), serde_json::json!({"dtype": "U8", "shape": [packed.len()], "data_offsets": [start, end]}));
            let sbytes: Vec<u8> = scale.iter().flat_map(|v| v.to_le_bytes()).collect();
            let sstart = data.len() as u64;
            data.extend_from_slice(&sbytes);
            let send = data.len() as u64;
            header.insert(format!("{name}.qs"), serde_json::json!({"dtype": "F32", "shape": [scale.len()], "data_offsets": [sstart, send]}));
        };
        for eid in 0..n_experts {
            push_qt(&mut header, format!("model.layers.5.mlp.experts.{eid}.gate_proj.weight"), moe_inter, hidden, &mut seed);
            push_qt(&mut header, format!("model.layers.5.mlp.experts.{eid}.up_proj.weight"), moe_inter, hidden, &mut seed);
            push_qt(&mut header, format!("model.layers.5.mlp.experts.{eid}.down_proj.weight"), hidden, moe_inter, &mut seed);
        }
        let norm_vals = [1.0f32, 2.0, 3.0, 4.0];
        let norm_bytes: Vec<u8> = norm_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let nstart = data.len() as u64;
        data.extend_from_slice(&norm_bytes);
        let nend = data.len() as u64;
        header.insert("model.layers.5.input_layernorm.weight".to_string(), serde_json::json!({"dtype": "F32", "shape": [4], "data_offsets": [nstart, nend]}));

        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_bytes);
        out.extend_from_slice(&data);
        fs::write(dir.0.join("model.safetensors"), out).unwrap();
        dir
    }

    fn qt_values(t: &QT) -> Vec<f32> {
        (0..t.rows).flat_map(|r| t.row_f32(r)).collect()
    }

    #[test]
    fn fuse_all_merges_gate_up_and_passes_everything_else_through() {
        let (n_experts, moe_inter, hidden) = (2, 4, 6);
        let src_dir = build_source_shard("rabbit_test_fuse_gate_up_source", n_experts, moe_inter, hidden);
        let src_shards = Shards::open(&src_dir.0).unwrap();
        let out_dir = TempDir::new("rabbit_test_fuse_gate_up_out");

        fuse_all(&src_shards, &out_dir.0).unwrap();
        let fused_shards = Shards::open(&out_dir.0).unwrap();

        // norm passes through byte-identical.
        assert_eq!(fused_shards.read_f32("model.layers.5.input_layernorm.weight", false).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

        for eid in 0..n_experts {
            let gate_name = format!("model.layers.5.mlp.experts.{eid}.gate_proj.weight");
            let up_name = format!("model.layers.5.mlp.experts.{eid}.up_proj.weight");
            let down_name = format!("model.layers.5.mlp.experts.{eid}.down_proj.weight");
            let fused_name = format!("model.layers.5.mlp.experts.{eid}.gate_up_proj.weight");

            // separate gate_proj/up_proj (AND their .qs sidecars) must be GONE from the
            // output, merged not duplicated -- the .qs check specifically catches a real bug
            // this test caught once already: the sidecar leaking through as a stray passthrough
            // tensor even after its data was already folded into gate_up_proj.weight.qs.
            assert!(!fused_shards.has(&gate_name), "expert {eid}: gate_proj must not survive separately");
            assert!(!fused_shards.has(&up_name), "expert {eid}: up_proj must not survive separately");
            assert!(!fused_shards.has(&format!("{gate_name}.qs")), "expert {eid}: gate_proj.qs must not survive separately");
            assert!(!fused_shards.has(&format!("{up_name}.qs")), "expert {eid}: up_proj.qs must not survive separately");

            // down_proj passes through byte-identical.
            let expected_down = src_shards.read_raw(&down_name, false).unwrap();
            assert_eq!(fused_shards.read_raw(&down_name, false).unwrap(), expected_down, "expert {eid}: down_proj must pass through unchanged");

            // gate_up_proj's decoded values must match manually concatenating the ORIGINAL
            // separate tensors' own values, row for row -- the actual correctness contract.
            let expected_gate = {
                let raw = src_shards.read_raw(&gate_name, false).unwrap();
                let scale = src_shards.read_f32(&format!("{gate_name}.qs"), false).unwrap();
                QT::from_packed(moe_inter, hidden, 4, raw, scale).unwrap()
            };
            let expected_up = {
                let raw = src_shards.read_raw(&up_name, false).unwrap();
                let scale = src_shards.read_f32(&format!("{up_name}.qs"), false).unwrap();
                QT::from_packed(moe_inter, hidden, 4, raw, scale).unwrap()
            };

            let got_raw = fused_shards.read_raw(&fused_name, false).unwrap();
            let got_scale = fused_shards.read_f32(&format!("{fused_name}.qs"), false).unwrap();
            let got = QT::from_packed(2 * moe_inter, hidden, 4, got_raw, got_scale).unwrap();
            let got_vals = qt_values(&got);
            let (got_gate, got_up) = got_vals.split_at(moe_inter * hidden);
            assert_eq!(got_gate, qt_values(&expected_gate).as_slice(), "expert {eid}: fused gate half");
            assert_eq!(got_up, qt_values(&expected_up).as_slice(), "expert {eid}: fused up half");
        }
    }
}
