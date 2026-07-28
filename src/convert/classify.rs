//! `TensorKind`, the precision-bucket enum shared by both converters (moved here from
//! `glm52::convert::classify`, which now imports it rather than defining its own copy), plus
//! `classify_generic` — the architecture-agnostic heuristic from `resumen-conversor-rabbit.md`'s
//! design discussion.
//!
//! `classify_generic` deliberately does NOT special-case any of `TensorKind`'s GLM-5.2-flavored
//! buckets (`Sh`/`O`/`Kvb`/`Attn`/`Dmlp` — shared-expert/o_proj/kv_b_proj/other-attention/dense-
//! MLP) — those stay real precision-control knobs for the literal colibrì port
//! (`glm52::convert::classify::classify`), which needs them to reproduce that script's exact
//! output. The generic heuristic only ever returns `Consumed`/`Skip`/`F32`/`Io`/`X`/`Q` — six
//! buckets inferred from NAME PATTERNS + tensor DIMENSIONALITY alone, with no assumption about
//! which architecture a checkpoint's tensors come from:
//!
//!   1. A known scale-sidecar name (`_scale_inv`, `.weight_scale`, `.weight_scale_2`,
//!      `.input_scale`) -> `Consumed` (read alongside its own weight during dequant, an
//!      HF/FP8/NVFP4 convention, not a GLM-5.2 one).
//!   2. Not a 2D tensor -> `F32` (biases, norms — always passthrough, checked before any name
//!      pattern so a stray 1D tensor can never get mis-bucketed by a name coincidence).
//!   3. Name contains `norm`/`layernorm`/`ln` -> `F32`.
//!   4. Name contains `embed`/`lm_head`/`wte`/`output` -> `Io`.
//!   5. Name matches `experts.\d+\.` -> `X` (routed expert).
//!   6. Name ends with `gate.weight` (not `gate_proj.weight` — `ends_with` already tells those
//!      apart, no extra exclusion needed) -> `F32` (a MoE router, tiny and precision-sensitive).
//!   7. Everything else 2D -> `Q` (the general/"dense" bucket).
//!
//! No `--skip-pattern`/`--io-pattern`/`--expert-pattern` escape hatches yet — per the design
//! doc's own stated discipline ("no diseñar para casos hipotéticos"), those get added the first
//! time a REAL model's tensor names don't fit this heuristic, not speculatively now.

/// Mirrors `convert_fp8_to_int4.py`'s classification string tags — see each variant's doc for
/// what it means. `Hash`/`Ord` are for `report.rs`'s bucket-counting map, not used by
/// classification itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TensorKind {
    /// Not emitted at all (MTP head when not in `--mtp` mode, DSA indexer weights when not in
    /// `--indexer` mode, etc — GLM-5.2-specific classifier only; `classify_generic` never
    /// returns this, since it has no architecture-specific "extra block" concept to skip).
    Skip,
    /// A scale sidecar (`*_scale_inv`, `*.weight_scale`, `*.weight_scale_2`, `*.input_scale`) —
    /// read together with its own weight tensor during dequant, never emitted on its own.
    Consumed,
    /// Kept as plain F32 (norms, the router itself, correction biases) — never quantized.
    F32,
    /// `embed_tokens`/`lm_head` — quantized at `io_bits` (typically higher than the general
    /// resident default) rather than the general dense bucket.
    Io,
    /// Routed expert weight — streamed from disk on demand at runtime, quantized at `xbits`.
    X,
    /// Shared expert (`shared_experts.*`) — fires on EVERY token, the resident (not streamed)
    /// tensor with the highest sensitivity to quantization error. GLM-5.2-specific classifier
    /// only.
    Sh,
    /// `o_proj.weight` — reconstructs attention output every step, the biggest attention tensor.
    /// GLM-5.2-specific classifier only.
    O,
    /// `kv_b_proj.weight` — reconstructs the KV cache every decode step. GLM-5.2-specific
    /// classifier only.
    Kvb,
    /// Other attention projections: `q_a_proj`/`q_b_proj`/`kv_a_proj_with_mqa`. GLM-5.2-specific
    /// classifier only.
    Attn,
    /// Dense MLP (`mlp.{gate,up,down}_proj.weight` on the first `first_k_dense_replace` layers,
    /// which have no MoE block at all). GLM-5.2-specific classifier only.
    Dmlp,
    /// Fallback: any other resident 2D weight tensor not covered by a more specific bucket —
    /// `classify_generic`'s "dense"/general bucket, quantized at the general resident default.
    Q,
}

const SCALE_SIDECAR_SUFFIXES: [&str; 4] = ["_scale_inv", ".weight_scale", ".weight_scale_2", ".input_scale"];
const NORM_MARKERS: [&str; 3] = ["norm", "layernorm", "ln"];
const IO_MARKERS: [&str; 4] = ["embed", "lm_head", "wte", "output"];

/// `true` if `name` contains an `experts.<digits>.` segment — the one place this heuristic needs
/// something regex-shaped, implemented by hand (a handful of lines) rather than pulling in a
/// pattern-matching dependency for one substring check.
fn contains_expert_index(name: &str) -> bool {
    let mut rest = name;
    while let Some(pos) = rest.find("experts.") {
        let after = &rest[pos + "experts.".len()..];
        let digits = after.bytes().take_while(u8::is_ascii_digit).count();
        if digits > 0 && after.as_bytes().get(digits) == Some(&b'.') {
            return true;
        }
        rest = &after[digits.min(after.len())..];
    }
    false
}

/// The architecture-agnostic heuristic — see this module's doc for the full rule list and why
/// each rule is ordered where it is. `ndim` is the tensor's own declared rank (`shape.len()`),
/// not inferred from the name.
pub fn classify_generic(name: &str, ndim: usize) -> TensorKind {
    if SCALE_SIDECAR_SUFFIXES.iter().any(|s| name.ends_with(s)) {
        return TensorKind::Consumed;
    }
    if ndim != 2 {
        return TensorKind::F32;
    }
    if NORM_MARKERS.iter().any(|m| name.contains(m)) {
        return TensorKind::F32;
    }
    if IO_MARKERS.iter().any(|m| name.contains(m)) {
        return TensorKind::Io;
    }
    if contains_expert_index(name) {
        return TensorKind::X;
    }
    if name.ends_with("gate.weight") {
        return TensorKind::F32;
    }
    TensorKind::Q
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_sidecars_are_consumed_regardless_of_dimensionality() {
        assert_eq!(classify_generic("model.layers.3.mlp.down_proj.weight_scale_inv", 2), TensorKind::Consumed);
        assert_eq!(classify_generic("model.layers.3.mlp.experts.0.gate_proj.weight_scale", 2), TensorKind::Consumed);
        assert_eq!(classify_generic("model.layers.3.self_attn.q_a_proj.weight_scale_2", 1), TensorKind::Consumed);
        assert_eq!(classify_generic("model.layers.3.self_attn.q_a_proj.input_scale", 1), TensorKind::Consumed);
    }

    #[test]
    fn any_non_2d_tensor_is_passthrough_f32_regardless_of_name() {
        assert_eq!(classify_generic("model.layers.0.self_attn.o_proj.bias", 1), TensorKind::F32);
        // even a name that WOULD match the embed/expert patterns must still passthrough if 1D.
        assert_eq!(classify_generic("model.embed_tokens.bias", 1), TensorKind::F32);
        assert_eq!(classify_generic("model.layers.0.mlp.experts.3.gate_proj.bias", 1), TensorKind::F32);
    }

    #[test]
    fn norm_layernorm_and_ln_markers_stay_f32() {
        assert_eq!(classify_generic("model.layers.0.input_layernorm.weight", 2), TensorKind::F32);
        assert_eq!(classify_generic("model.norm.weight", 2), TensorKind::F32);
        assert_eq!(classify_generic("transformer.h.0.ln_1.weight", 2), TensorKind::F32);
        assert_eq!(classify_generic("model.layers.0.pre_norm.weight", 2), TensorKind::F32);
    }

    #[test]
    fn embed_and_output_markers_get_the_io_bucket() {
        assert_eq!(classify_generic("model.embed_tokens.weight", 2), TensorKind::Io);
        assert_eq!(classify_generic("lm_head.weight", 2), TensorKind::Io);
        assert_eq!(classify_generic("transformer.wte.weight", 2), TensorKind::Io);
        assert_eq!(classify_generic("output.weight", 2), TensorKind::Io);
    }

    #[test]
    fn expert_index_pattern_matches_any_expert_naming_scheme() {
        assert_eq!(classify_generic("model.layers.5.mlp.experts.3.gate_proj.weight", 2), TensorKind::X);
        assert_eq!(classify_generic("model.layers.5.block_sparse_moe.experts.17.w1.weight", 2), TensorKind::X);
        // "experts." with no digit right after must NOT match -- avoids a false positive on an
        // unrelated tensor that merely contains the substring "experts." followed by a name.
        assert_eq!(classify_generic("model.layers.5.experts.gate.weight", 2), TensorKind::F32); // hits the gate.weight rule instead
    }

    #[test]
    fn router_gate_weight_is_f32_but_gate_proj_is_not() {
        assert_eq!(classify_generic("model.layers.5.mlp.gate.weight", 2), TensorKind::F32);
        assert_eq!(classify_generic("model.layers.5.mlp.gate_proj.weight", 2), TensorKind::Q);
    }

    #[test]
    fn unrecognized_2d_weight_falls_back_to_the_dense_bucket() {
        assert_eq!(classify_generic("model.layers.5.self_attn.some_new_proj.weight", 2), TensorKind::Q);
        assert_eq!(classify_generic("model.layers.5.self_attn.q_proj.weight", 2), TensorKind::Q);
    }
}
