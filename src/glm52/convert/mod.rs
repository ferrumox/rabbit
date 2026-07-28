//! Port of colibrì's `c/tools/convert_fp8_to_int4.py` (`STADIO B` FP8 → int4 checkpoint
//! converter, ~670 lines) — the tool that produced the `jlnsrk/GLM-5.2-colibri-int4` checkpoint
//! rabbit already runs. Ported to learn the real conversion pipeline end to end (tensor
//! classification, dequant, quantization, mixed precision, the disk-safe shard-by-shard
//! download strategy) before designing any of rabbit's own converter ideas
//! (`ROADMAP.md`'s gate/up-fusion and mixed-precision-converter items) — same "read the real
//! reference before porting" discipline as every other architecture piece in this crate.
//!
//! Scope: the full script, including pieces not (yet) exercised by rabbit's own checkpoint
//! (NVFP4 dequant, the `--mtp`/`--indexer` extraction modes, the multi-stream HF downloader) —
//! this is a faithful, whole-script port, not a cherry-picked subset. Submodule-by-submodule
//! correspondence to the Python source:
//!
//!   - `classify.rs` <- `layer_idx`/`classify`
//!   - `dequant.rs` <- `dequant`/`dequant_nvfp4` (FP8 block dequant itself already existed as
//!     `crate::safetensors::dequant_fp8_blockscale`, reused rather than reimplemented)
//!   - (more to come: `shard.rs`, `writer.rs`, `download.rs`)
//!
//! Grouped int4 quantization (`quant_int4_grouped`) lives in `crate::quant` instead, alongside
//! its ungrouped siblings (`pack_int4`/`pack_int2`/`quantize_rows`, already-existing ports of
//! this same Python file's `quant_int4`/`quant_int2`/`quant_int8` used by rabbit's own runtime
//! `QT` container) — it's architecture-agnostic quantization math, not converter-specific.

pub mod classify;
pub mod dequant;
pub mod download;
pub mod hf_api;
pub mod shard;
pub mod writer;
