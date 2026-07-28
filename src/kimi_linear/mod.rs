//! Kimi Linear's architecture: Kimi Delta Attention (KDA) — a gated linear attention that
//! refines Gated DeltaNet's delta rule with a channel-wise (not scalar) forget gate — hybridized
//! 3:1 with regular global MLA layers (no RoPE on those, per the reference implementation's
//! `mla_use_nope`).
//!
//! Built as a stepping stone toward Kimi K3 (see `rabbit-plan.md`'s Phase 3/4): K3 uses the same
//! KDA mechanism but a different, not-yet-published MoE routing scheme (Stable LatentMoE +
//! Quantile Balancing). Validating KDA here, against Kimi Linear's real 48B/A3B open weights,
//! de-risks the one genuinely novel piece before K3's own weights exist to test against.
//!
//! Source of truth for the math in this module: Kimi Linear's technical report
//! (arXiv:2510.26692, Eq. 1 for the core recurrence, §4 for the neural parameterization) cross-
//! checked against `moonshotai/Kimi-Linear-48B-A3B-Instruct`'s real `modeling_kimi.py` and the
//! `fla` (flash-linear-attention) library's `fused_kda_gate` — not guessed from the paper's prose
//! alone, per this project's own "read the reference before porting" discipline.

pub mod chat_template;
pub mod config;
pub mod generate;
pub mod kda;
pub mod kv_session;
pub mod model;
pub mod ops;
pub mod short_conv;
pub mod tokenizer;
