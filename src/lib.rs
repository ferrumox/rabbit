//! rabbit — Rust port of colibrì's GLM-5.2 inference engine. Work in progress.

pub mod chat;
pub mod convert;
pub mod expert_cache;
pub mod generate;
pub mod glm52;
pub mod kernels;
pub mod kimi_k3;
pub mod kimi_linear;
pub mod kv_session;
pub mod model;
pub mod quant;
pub mod safetensors;
pub mod server;
pub mod tokenizer;
mod unicode_tables;
pub mod usage_cache;
