//! Generates a **synthetic, loadable Kimi K3 checkpoint** at the real per-layer widths but only
//! a handful of layers — the Phase 1 test fixture from `K3_OPTIMIZE_BRIEF.md`. Real-checkpoint
//! iteration costs a 1.56 TB / ~10-minute load per experiment; K3 decode is per-layer
//! homogeneous, so a random-weight checkpoint with the SAME per-layer shapes but few layers
//! reproduces the kernel / parallelism / NUMA behaviour this brief tunes at ~5-10% of the
//! footprint (≈90 GB at the default 6 layers).
//!
//! The weights are random garbage: this fixture is a PERFORMANCE proxy, never a correctness one
//! (token-exact correctness is `tests/teacher_forcing_k3.rs`'s job, against the tiny oracle). The
//! routed experts are emitted as raw MXFP4 `.weight_packed`/`.weight_scale` byte pairs with the
//! exact byte counts `qt_load_mxfp4` checks (`rows*ceil(cols/2)` and `rows*ceil(cols/32)`); their
//! contents are random bytes, with E8M0 scale bytes pinned to [120,130] so dequantised magnitudes
//! stay finite (garbage logits are fine, NaN/inf poisoning is avoidable noise — see the brief).
//!
//! Layout: one `*.safetensors` shard per layer plus one model-level shard, all under the output
//! dir. `Shards::open` scans `*.safetensors` sorted by filename with no index needed
//! (`src/safetensors.rs`), so the `aa_model` / `layer_NN` names below load as one checkpoint.
//! Tensor names use K3's real `language_model.` prefix and `ExpertNaming::KimiK3Mxfp4`'s
//! `block_sparse_moe.experts.{eid}.w{1,3,2}` expert names (verified against `tensor_specs`).
//!
//! Usage:
//!   cargo run --release --bin gen_k3_synth -- --out <dir> [--layers 6] [--experts 896] \
//!       [--tokenizer-src /data/hf/hub/kimi-k3]
//!
//! `--tokenizer-src DIR` copies `tiktoken.model` + `tokenizer_config.json` from DIR so
//! `teacher_forced_decode_bench` (which needs a tokenizer) can run against the fixture; `k3_smoke`
//! needs no tokenizer.

use serde_json::{Map, Value, json};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

// ---- Real Kimi K3 per-layer dimensions (from config.rs / model.rs, verified against the real
// checkpoint's config.json). Everything a decoded token actually touches is at these widths. ----
const HIDDEN: usize = 7168; // hidden_size (d)
const MLA_HEADS: usize = 96; // num_attention_heads (h)
const QK_NOPE: usize = 128;
const QK_ROPE: usize = 64;
const V_HEAD: usize = 128;
const KV_LORA: usize = 512;
const Q_LORA: usize = 1536;
const VOCAB: usize = 163840;
const DENSE_INTER: usize = 33792; // intermediate_size (dense layer 0)
const MOE_INTER: usize = 3072; // moe_intermediate_size
const MOE_HIDDEN: usize = 3584; // routed_expert_hidden_size (latent width)
const N_SHARED: usize = 2;
const KDA_HEAD_DIM: usize = 128; // linear_attn_config.head_dim
const KDA_N_HEADS: usize = 96; // linear_attn_config.num_heads
const KDA_KERNEL: usize = 4; // short_conv_kernel_size
const FIRST_DENSE: usize = 1; // first_k_dense_replace

fn d_inner() -> usize {
    KDA_HEAD_DIM * KDA_N_HEADS
}

/// One tensor to emit into a shard: logical name, safetensors dtype string, shape, and the
/// generator that fills its bytes (fed a per-tensor deterministic seed).
struct Tensor {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
    nbytes: usize,
    fill: Fill,
}

#[derive(Clone, Copy)]
enum Fill {
    /// Random f32 in ~[-1,1], little-endian.
    F32,
    /// Random MXFP4 packed nibble bytes (any value is a valid pair of E2M1 codes).
    PackedU8,
    /// E8M0 scale bytes constrained to [120,130] so 2^(b-127) stays a small finite factor.
    ScaleU8,
}

fn xorshift(seed: &mut u32) -> u32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    *seed
}

fn f32t(name: String, shape: Vec<usize>) -> Tensor {
    let n: usize = shape.iter().product::<usize>().max(1);
    Tensor { name, dtype: "F32", shape, nbytes: n * 4, fill: Fill::F32 }
}

fn packed(name: String, rows: usize, cols: usize) -> Tensor {
    let cols_b = cols.div_ceil(2);
    Tensor { name, dtype: "U8", shape: vec![rows, cols_b], nbytes: rows * cols_b, fill: Fill::PackedU8 }
}

fn scale(name: String, rows: usize, cols: usize) -> Tensor {
    let cols_b = cols.div_ceil(32);
    Tensor { name, dtype: "U8", shape: vec![rows, cols_b], nbytes: rows * cols_b, fill: Fill::ScaleU8 }
}

/// Writes one shard file: 8-byte little-endian header length, JSON header, then each tensor's
/// bytes generated on the fly (never materialising the whole >10 GB blob in memory).
fn write_shard(path: &Path, tensors: &[Tensor]) -> std::io::Result<()> {
    let mut header = Map::new();
    header.insert("__metadata__".to_string(), json!({"format": "rabbit-k3-synth"}));
    let mut offset = 0u64;
    for t in tensors {
        let start = offset;
        offset += t.nbytes as u64;
        header.insert(
            t.name.clone(),
            json!({"dtype": t.dtype, "shape": t.shape, "data_offsets": [start, offset]}),
        );
    }
    let header_bytes = serde_json::to_vec(&Value::Object(header)).unwrap();
    let f = File::create(path)?;
    let mut w = BufWriter::with_capacity(1 << 22, f);
    w.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
    w.write_all(&header_bytes)?;

    // 4 MiB scratch, regenerated per tensor from a name-derived seed (deterministic output).
    let mut buf = vec![0u8; 1 << 22];
    for t in tensors {
        let mut seed = t.name.bytes().fold(0x9E3779B9u32, |a, b| a.wrapping_mul(31).wrapping_add(b as u32)) | 1;
        let mut remaining = t.nbytes;
        while remaining > 0 {
            let chunk = remaining.min(buf.len());
            match t.fill {
                Fill::F32 => {
                    // Fill whole f32 words; chunk is a multiple of 4 except possibly the tail,
                    // but nbytes is always a multiple of 4 for F32 tensors so alignment holds.
                    for word in buf[..chunk].chunks_mut(4) {
                        let r = xorshift(&mut seed);
                        let v = ((r as f32 / u32::MAX as f32) - 0.5) * 2.0;
                        word.copy_from_slice(&v.to_le_bytes()[..word.len()]);
                    }
                }
                Fill::PackedU8 => {
                    for b in &mut buf[..chunk] {
                        *b = (xorshift(&mut seed) & 0xFF) as u8;
                    }
                }
                Fill::ScaleU8 => {
                    for b in &mut buf[..chunk] {
                        *b = 120 + (xorshift(&mut seed) % 11) as u8; // [120,130]
                    }
                }
            }
            w.write_all(&buf[..chunk])?;
            remaining -= chunk;
        }
    }
    w.flush()
}

/// Builds the tensor list for one transformer layer (attention + FFN + residual-block norms).
fn layer_tensors(i: usize, is_kda: bool, is_moe: bool, n_experts: usize) -> Vec<Tensor> {
    let d = HIDDEN;
    let p = |s: &str| format!("language_model.model.layers.{i}.{s}");
    let ap = |s: &str| format!("language_model.model.layers.{i}.self_attn.{s}");
    let mut ts = Vec::new();
    ts.push(f32t(p("input_layernorm.weight"), vec![d]));
    ts.push(f32t(p("post_attention_layernorm.weight"), vec![d]));
    // attn_res_block_size = 2 -> per-layer residual norms/projs on every layer.
    ts.push(f32t(p("self_attention_res_norm.weight"), vec![d]));
    ts.push(f32t(p("self_attention_res_proj.weight"), vec![1, d]));
    ts.push(f32t(p("mlp_res_norm.weight"), vec![d]));
    ts.push(f32t(p("mlp_res_proj.weight"), vec![1, d]));

    if is_kda {
        let di = d_inner();
        ts.push(f32t(ap("q_proj.weight"), vec![di, d]));
        ts.push(f32t(ap("k_proj.weight"), vec![di, d]));
        ts.push(f32t(ap("v_proj.weight"), vec![di, d]));
        ts.push(f32t(ap("q_conv1d.weight"), vec![di, 1, KDA_KERNEL]));
        ts.push(f32t(ap("k_conv1d.weight"), vec![di, 1, KDA_KERNEL]));
        ts.push(f32t(ap("v_conv1d.weight"), vec![di, 1, KDA_KERNEL]));
        ts.push(f32t(ap("f_a_proj.weight"), vec![KDA_HEAD_DIM, d]));
        ts.push(f32t(ap("f_b_proj.weight"), vec![di, KDA_HEAD_DIM]));
        ts.push(f32t(ap("dt_bias"), vec![di]));
        ts.push(f32t(ap("A_log"), vec![1, 1, KDA_N_HEADS, 1]));
        ts.push(f32t(ap("b_proj.weight"), vec![KDA_N_HEADS, d]));
        // use_full_rank_gate = true -> single g_proj (see config.rs).
        ts.push(f32t(ap("g_proj.weight"), vec![di, d]));
        ts.push(f32t(ap("o_norm.weight"), vec![KDA_HEAD_DIM]));
        ts.push(f32t(ap("o_proj.weight"), vec![d, di]));
    } else {
        // MLA with q_lora>0 and mla_use_output_gate.
        ts.push(f32t(ap("q_a_proj.weight"), vec![Q_LORA, d]));
        ts.push(f32t(ap("q_a_layernorm.weight"), vec![Q_LORA]));
        ts.push(f32t(ap("q_b_proj.weight"), vec![MLA_HEADS * (QK_NOPE + QK_ROPE), Q_LORA]));
        ts.push(f32t(ap("kv_a_proj_with_mqa.weight"), vec![KV_LORA + QK_ROPE, d]));
        ts.push(f32t(ap("kv_a_layernorm.weight"), vec![KV_LORA]));
        ts.push(f32t(ap("kv_b_proj.weight"), vec![MLA_HEADS * (QK_NOPE + V_HEAD), KV_LORA]));
        ts.push(f32t(ap("o_proj.weight"), vec![d, MLA_HEADS * V_HEAD]));
        ts.push(f32t(ap("g_proj.weight"), vec![MLA_HEADS * V_HEAD, d]));
    }

    if !is_moe {
        ts.push(f32t(p("mlp.gate_proj.weight"), vec![DENSE_INTER, d]));
        ts.push(f32t(p("mlp.up_proj.weight"), vec![DENSE_INTER, d]));
        ts.push(f32t(p("mlp.down_proj.weight"), vec![d, DENSE_INTER]));
    } else {
        let mp = |s: &str| format!("language_model.model.layers.{i}.block_sparse_moe.{s}");
        let s_i = MOE_INTER * N_SHARED;
        ts.push(f32t(mp("gate.weight"), vec![n_experts, d]));
        ts.push(f32t(mp("gate.e_score_correction_bias"), vec![n_experts]));
        ts.push(f32t(mp("shared_experts.gate_proj.weight"), vec![s_i, d]));
        ts.push(f32t(mp("shared_experts.up_proj.weight"), vec![s_i, d]));
        ts.push(f32t(mp("shared_experts.down_proj.weight"), vec![d, s_i]));
        ts.push(f32t(mp("routed_expert_down_proj.weight"), vec![MOE_HIDDEN, d]));
        ts.push(f32t(mp("routed_expert_up_proj.weight"), vec![d, MOE_HIDDEN]));
        ts.push(f32t(mp("routed_expert_norm.weight"), vec![MOE_HIDDEN]));
        // Routed experts as MXFP4 pairs (KimiK3Mxfp4 naming: w1/w3 = [moe_inter, moe_hidden],
        // w2 = [moe_hidden, moe_inter]; qt_load_mxfp4 appends .weight_packed/.weight_scale).
        for eid in 0..n_experts {
            let ep = |suf: &str| format!("language_model.model.layers.{i}.block_sparse_moe.experts.{eid}.{suf}");
            for (w, rows, cols) in [("w1", MOE_INTER, MOE_HIDDEN), ("w3", MOE_INTER, MOE_HIDDEN), ("w2", MOE_HIDDEN, MOE_INTER)] {
                ts.push(packed(format!("{}.weight_packed", ep(w)), rows, cols));
                ts.push(scale(format!("{}.weight_scale", ep(w)), rows, cols));
            }
        }
    }
    ts
}

fn model_level_tensors() -> Vec<Tensor> {
    let d = HIDDEN;
    vec![
        f32t("language_model.model.embed_tokens.weight".into(), vec![VOCAB, d]),
        f32t("language_model.lm_head.weight".into(), vec![VOCAB, d]),
        f32t("language_model.model.norm.weight".into(), vec![d]),
        f32t("language_model.model.output_attn_res_norm.weight".into(), vec![d]),
        f32t("language_model.model.output_attn_res_proj.weight".into(), vec![1, d]),
    ]
}

/// Writes config.json for `n_layers`, with a 3:1 KDA:full-MLA attention pattern. `kda_layers`
/// and `full_attn_layers` are **1-indexed** and must together cover every layer exactly once
/// (config.rs errors otherwise); full-MLA sits at 1-based positions p where p%4==3 (matching the
/// real 4-layer template's `full_attn_layers:[3]`), KDA everywhere else.
fn write_config(dir: &Path, n_layers: usize) {
    let full: Vec<usize> = (1..=n_layers).filter(|p| p % 4 == 3).collect();
    let kda: Vec<usize> = (1..=n_layers).filter(|p| p % 4 != 3).collect();
    let linear_attn_config = json!({
        "head_dim": KDA_HEAD_DIM, "num_heads": KDA_N_HEADS, "short_conv_kernel_size": KDA_KERNEL,
        "gate_lower_bound": -5.0, "use_full_rank_gate": true,
        "kda_layers": kda, "full_attn_layers": full
    });
    let text_config = json!({
        "model_type": "kimi_linear",
        "hidden_size": HIDDEN, "num_hidden_layers": n_layers, "num_attention_heads": MLA_HEADS,
        "first_k_dense_replace": FIRST_DENSE, "q_lora_rank": Q_LORA, "kv_lora_rank": KV_LORA,
        "qk_nope_head_dim": QK_NOPE, "qk_rope_head_dim": QK_ROPE, "v_head_dim": V_HEAD,
        "num_experts": 896, "num_experts_per_token": 16, "num_shared_experts": N_SHARED,
        "num_expert_group": 1, "topk_group": 1, "moe_intermediate_size": MOE_INTER,
        "intermediate_size": DENSE_INTER, "vocab_size": VOCAB, "moe_renormalize": true,
        "rms_norm_eps": 1e-5, "routed_scaling_factor": 1.0, "mla_use_nope": true,
        "moe_router_activation_func": "sigmoid", "rope_theta": 10000.0, "eos_token_id": 163586,
        "hidden_act": "situ", "activation_situ_beta": 4.0, "activation_situ_linear_beta": 25.0,
        "routed_expert_hidden_size": MOE_HIDDEN, "latent_moe_use_norm": true,
        "attn_res_block_size": 2, "mla_use_output_gate": true,
        "linear_attn_config": linear_attn_config,
        // Read from INSIDE text_config (config.rs:211) -> sets mxfp4_experts -> KimiK3Mxfp4 naming.
        "quantization_config": { "quant_method": "compressed-tensors", "format": "mxfp4-pack-quantized" }
    });
    let cfg = json!({ "model_type": "kimi_k3", "text_config": text_config });
    std::fs::write(dir.join("config.json"), serde_json::to_vec_pretty(&cfg).unwrap()).unwrap();
}

fn arg(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned()
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let out = PathBuf::from(arg(&args, "--out").expect("--out <dir> is required"));
    let n_layers: usize = arg(&args, "--layers").and_then(|s| s.parse().ok()).unwrap_or(6);
    let n_experts: usize = arg(&args, "--experts").and_then(|s| s.parse().ok()).unwrap_or(896);
    assert!(n_layers >= 2, "need at least 2 layers (1 dense + >=1 MoE) to exercise the MoE path");
    std::fs::create_dir_all(&out)?;

    write_config(&out, n_layers);
    eprintln!("wrote config.json ({n_layers} layers, {n_experts} experts/MoE layer)");

    let t0 = std::time::Instant::now();
    write_shard(&out.join("aa_model.safetensors"), &model_level_tensors())?;
    eprintln!("wrote model-level shard");

    let full: Vec<usize> = (1..=n_layers).filter(|p| p % 4 == 3).collect();
    let mut total_bytes = 0u64;
    for i in 0..n_layers {
        let is_kda = !full.contains(&(i + 1)); // 1-indexed membership
        let is_moe = i >= FIRST_DENSE;
        let ts = layer_tensors(i, is_kda, is_moe, n_experts);
        let bytes: u64 = ts.iter().map(|t| t.nbytes as u64).sum();
        total_bytes += bytes;
        let path = out.join(format!("layer_{i:02}.safetensors"));
        write_shard(&path, &ts)?;
        eprintln!(
            "  layer {i}: {} attn, {} FFN, {:.1} GiB",
            if is_kda { "KDA" } else { "MLA" },
            if is_moe { "MoE" } else { "dense" },
            bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }

    if let Some(src) = arg(&args, "--tokenizer-src") {
        let src = PathBuf::from(src);
        for f in ["tiktoken.model", "tokenizer_config.json"] {
            if src.join(f).exists() {
                std::fs::copy(src.join(f), out.join(f))?;
                eprintln!("copied {f}");
            }
        }
    }

    eprintln!(
        "done in {:.1}s — {:.1} GiB total under {}",
        t0.elapsed().as_secs_f32(),
        total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        out.display()
    );
    Ok(())
}
