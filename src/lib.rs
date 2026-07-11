//! rabbit — Rust port of colibrì's GLM-5.2 inference engine. Work in progress.

pub mod attention;
pub mod config;
pub mod kernels;
pub mod model;
pub mod quant;
pub mod safetensors;
pub mod tokenizer;
mod unicode_tables;
