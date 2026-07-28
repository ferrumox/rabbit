//! Kimi K3's text backbone: `moonshotai/Kimi-K3`'s `config.json` has `model_type: "kimi_k3"` at
//! the top level, but nests `model_type: "kimi_linear"` and every field a standalone Kimi Linear
//! checkpoint has (hidden_size, linear_attn_config, MoE routing, ...) under `text_config` —
//! confirmed against the real checkpoint's `config.json`, fetched 2026-07-27. K3's own
//! `modeling_kimi_k3.py` (also fetched real, not guessed) contains ONLY the vision tower
//! (MoonViT) and a multimodal wrapper class — zero text-decoder code; the actual decoder classes
//! (`KimiDecoderLayer`, `KimiDeltaAttention`, `KimiMLAAttention`, `KimiSparseMoeBlock`, ...) live
//! in `modeling_kimi_linear.py`, the same file Phase 3's `crate::kimi_linear` module already
//! ports. Vision is out of scope (this project's CPU+disk-streaming credibility play has always
//! been the text engine, matching the GLM-5.2/rabbit precedent).
//!
//! This module therefore isn't a from-scratch third architecture: it wraps
//! `crate::kimi_linear::config::Cfg` (see `config.rs`) and, once the model/generate loaders land,
//! will reuse `crate::kimi_linear::model`/`generate` for everything KDA/MLA/MoE-routing-shaped,
//! adding only K3's three genuinely new pieces on top: `SituAndMul` activation
//! (`crate::kimi_linear::ops::situ_and_mul`, already ported), a latent-MoE down/up-proj wrapper
//! around the routed-expert block, and "Attention Residuals" (a block-pooling mechanism across
//! layers) — see `~/.claude/plans/giggly-greeting-bentley.md`'s Phase 4 section for the exact
//! formulas and real config field names.

pub mod attn_res;
pub mod chat_template;
pub mod config;
pub mod generate;
pub mod kv_session;
pub mod model;
pub mod moe;
pub mod tokenizer;
