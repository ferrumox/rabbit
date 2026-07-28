//! GLM-5.2's architecture: MLA (+ DSA sparse-attention indexer) attention and `noaux_tc`
//! (sigmoid + bias-correction) MoE routing. The first of what's meant to become several
//! sibling architecture modules (see `rabbit-plan.md`) — everything GLM-5.2-shaped lives here;
//! everything architecture-agnostic (quantized tensor math, the expert disk-streaming cache,
//! safetensors loading, sampling/decode-loop orchestration) stays at the crate root.

pub mod attention;
pub mod config;
pub mod model;
pub mod moe;
