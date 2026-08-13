//! Qwen 3.8's architecture (`model_type: "qwen3_5_moe_text"`, `Qwen3_5MoeForCausalLM`) — named
//! after the MODEL, like `glm52` (whose own `model_type` is `glm_moe_dsa`), not after the
//! architecture id: "3.5" is the architecture family Qwen 3.8 is built on, and this one module is
//! meant to serve every checkpoint of it (Qwen3.8-Max/2.4T-A95B, the announced Qwen3.8-27B, and
//! the already-published Qwen3.6-35B-A3B, which nests the same fields under `text_config`
//! alongside a vision tower).
//!
//! Two pieces are genuinely new to rabbit, and neither is MLA:
//!
//! * **Full attention is plain GQA** (64 query heads, 4 KV heads, `head_dim: 256`) with per-head
//!   `q_norm`/`k_norm`, **partial RoPE** (`partial_rotary_factor: 0.25` — only the first 64 of
//!   each head's 256 dims are rotated) and an **output gate** (`attn_output_gate: true`,
//!   `output_gate_type: "swish"`, i.e. `q_proj` emits twice `num_heads * head_dim` and the second
//!   half gates the attention output). Simpler than GLM-5.2's/Kimi's MLA, but no absorbed-decode
//!   trick applies and the KV cache is an ordinary one.
//! * **Linear attention is Gated DeltaNet**, the mechanism Kimi Delta Attention refines: rabbit's
//!   `kimi_linear::kda`'s plain (unbounded) gate formula
//!   `alpha = exp(-exp(A_log) * softplus(g + dt_bias))` and its `short_conv` are already the right
//!   math, but GDN here is GQA-shaped too — **16 key heads feeding 128 value heads** (1:8) — which
//!   `kda.rs` currently doesn't model, and its projections are packed differently
//!   (`in_proj_qkv`/`in_proj_z`/`in_proj_a`/`in_proj_b`, per the real checkpoint's tensor names).
//!
//! Everything else is familiar shape: 3:1 hybrid layer pattern (`layer_types`, explicitly listed
//! and 0-indexed — no 1-indexed trap like `kimi_linear`'s `full_attn_layers`), 512 routed experts
//! with top-10 routing plus one shared expert with its own sigmoid gate, and routed experts that
//! ship natively MXFP4-quantized so `expert_cache.rs` streams them from disk unchanged (see
//! `ExpertNaming::Qwen38Mxfp4`).
//!
//! Real tensor names, read from `amd/Qwen3.8-2.4T-A95B-Quark-MXFP4`'s own
//! `model.safetensors.index.json` (287,119 tensors, downloaded 2026-08-13 — not guessed):
//!
//! ```text
//! lm_head.weight, model.embed_tokens.weight, model.norm.weight
//! model.layers.{L}.input_layernorm.weight, ...post_attention_layernorm.weight
//! linear_attention layer: linear_attn.{A_log, dt_bias, conv1d.weight, in_proj_qkv.weight,
//!                                     in_proj_z.weight, in_proj_a.weight, in_proj_b.weight,
//!                                     norm.weight, out_proj.weight}
//! full_attention layer:   self_attn.{q_proj, k_proj, v_proj, o_proj, q_norm, k_norm}.weight
//! MoE (every layer):      mlp.gate.weight, mlp.shared_expert.{gate,up,down}_proj.weight,
//!                         mlp.shared_expert_gate.weight,
//!                         mlp.experts.{E}.{gate,up,down}_proj.{weight, weight_scale}
//! MTP (skipped):          mtp.fc.weight, mtp.norm.weight, mtp.layers.0.* — a FULL extra MoE
//!                         layer with its own 512 experts, not just a small head
//! ```

pub mod attention;
pub mod chat_template;
pub mod config;
pub mod gdn;
pub mod generate;
pub mod kv_session;
pub mod model;
pub mod moe;
pub mod ops;
pub mod tokenizer;
