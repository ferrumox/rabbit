//! Port of `convert_fp8_to_int4.py`'s `layer_idx`/`classify` — decides, purely from a tensor's
//! NAME (no shape/dtype inspection), whether the converter keeps it at all and, if so, which
//! precision-control bucket it falls into. This is what makes per-tensor-CLASS mixed precision
//! possible (`--shared-bits`/`--o-bits`/`--kvb-bits`/`--attn-bits`/`--dmlp-bits` in the real
//! script): shared experts fire on every token (highest sensitivity), `o_proj`/`kv_b_proj`
//! reconstruct attention output/KV cache every step, dense-MLP-only applies to the first
//! `first_k_dense_replace` layers, and so on — each gets its own bit-width knob instead of one
//! global `--ebits` for everything.
//!
//! `TensorKind` itself now lives in `crate::convert::classify` (re-exported here unchanged) —
//! it's shared with the architecture-agnostic `classify_generic` there, which uses the same
//! `Consumed`/`Skip`/`F32`/`Io`/`X`/`Q` buckets this GLM-5.2-specific classifier does, plus this
//! one's own extra `Sh`/`O`/`Kvb`/`Attn`/`Dmlp` fine-grained buckets the generic heuristic never
//! produces. See `crate::convert`'s module doc for the full reasoning.

pub use crate::convert::classify::TensorKind;

/// Extracts the 0-indexed transformer layer number from a tensor name shaped
/// `model.layers.<N>.*`, or `-1` for anything else (global tensors like `lm_head.weight`, or a
/// malformed/unexpected name) — `i64` (not `usize`) specifically so `-1` is representable
/// without a wrapping/sentinel hack, matching the Python source's own use of a real negative
/// sentinel.
fn layer_idx(name: &str) -> i64 {
    let p: Vec<&str> = name.split('.').collect();
    if p.len() > 2 && p[0] == "model" && p[1] == "layers" {
        p[2].parse::<i64>().unwrap_or(-1)
    } else {
        -1
    }
}

/// Port of `classify(name, n_layers, keep_mtp, keep_idx)` — see this module's doc and
/// `TensorKind`'s variant docs for what each bucket means. Branch ORDER matters (first match
/// wins, exactly like the Python `if/return` chain) — e.g. a `shared_experts` tensor that also
/// ends in `.weight` must hit the `Sh` check before it could ever fall through to the generic
/// `Q` fallback.
pub fn classify(name: &str, n_layers: usize, keep_mtp: bool, keep_idx: bool) -> TensorKind {
    if name.ends_with("_scale_inv") {
        return TensorKind::Consumed;
    }
    if name.ends_with(".weight_scale") || name.ends_with(".weight_scale_2") || name.ends_with(".input_scale") {
        return TensorKind::Consumed;
    }
    let li = layer_idx(name);
    let n_layers = n_layers as i64;
    if keep_idx {
        if li < 0 || li >= n_layers || !name.contains("indexer") {
            return TensorKind::Skip;
        }
        if name.ends_with("norm.weight") {
            return TensorKind::F32;
        }
        return TensorKind::Q;
    }
    if keep_mtp {
        if li != n_layers {
            return TensorKind::Skip;
        }
        if name.contains("indexer") {
            return TensorKind::Skip;
        }
    } else {
        if li >= n_layers {
            return TensorKind::Skip;
        }
        if ["indexer", "indexers_proj", "eh_proj", "enorm", "hnorm", "shared_head"].iter().any(|k| name.contains(k)) {
            return TensorKind::Skip;
        }
    }
    if name.ends_with("e_score_correction_bias") {
        return TensorKind::F32;
    }
    if name.ends_with("mlp.gate.weight") {
        return TensorKind::F32;
    }
    // The `name == "model.norm.weight"` arm is redundant with `ends_with("norm.weight")` above
    // it (every name matching the former also matches the latter) — kept anyway, faithfully
    // mirroring the Python source's own redundant `or` clause, since this port's whole point is
    // learning the real script's exact behavior, not silently "cleaning up" it while porting.
    if name.ends_with("norm.weight") || name == "model.norm.weight" {
        return TensorKind::F32;
    }
    if name == "model.embed_tokens.weight" || name == "lm_head.weight" {
        return TensorKind::Io;
    }
    if name.contains(".mlp.experts.") && name.ends_with(".weight") {
        return TensorKind::X;
    }
    if name.contains("shared_experts") {
        return TensorKind::Sh;
    }
    if name.ends_with("o_proj.weight") {
        return TensorKind::O;
    }
    if name.ends_with("kv_b_proj.weight") {
        return TensorKind::Kvb;
    }
    if ["q_a_proj.weight", "q_b_proj.weight", "kv_a_proj_with_mqa.weight"].iter().any(|k| name.ends_with(k)) {
        return TensorKind::Attn;
    }
    if ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"].iter().any(|k| name.ends_with(k)) {
        return TensorKind::Dmlp;
    }
    if name.ends_with(".weight") {
        return TensorKind::Q;
    }
    TensorKind::F32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_sidecars_are_always_consumed_regardless_of_layer_or_mode() {
        assert_eq!(classify("model.layers.3.mlp.down_proj.weight_scale_inv", 78, false, false), TensorKind::Consumed);
        assert_eq!(classify("model.layers.3.mlp.experts.0.gate_proj.weight_scale", 78, false, false), TensorKind::Consumed);
        assert_eq!(classify("model.layers.3.self_attn.q_a_proj.weight_scale_2", 78, false, false), TensorKind::Consumed);
        assert_eq!(classify("model.layers.3.self_attn.q_a_proj.input_scale", 78, false, false), TensorKind::Consumed);
    }

    #[test]
    fn default_mode_skips_the_mtp_layer_and_indexer_family_tensors() {
        // n_layers=78 -> the MTP head lives at layer index 78, one past the real layers.
        assert_eq!(classify("model.layers.78.eh_proj.weight", 78, false, false), TensorKind::Skip);
        assert_eq!(classify("model.layers.5.self_attn.indexer.wk.weight", 78, false, false), TensorKind::Skip);
        assert_eq!(classify("model.layers.5.self_attn.indexers_proj.weight", 78, false, false), TensorKind::Skip);
        assert_eq!(classify("model.layers.5.enorm.weight", 78, false, false), TensorKind::Skip);
        assert_eq!(classify("model.layers.5.hnorm.weight", 78, false, false), TensorKind::Skip);
        assert_eq!(classify("model.layers.5.shared_head.weight", 78, false, false), TensorKind::Skip);
    }

    #[test]
    fn keep_mtp_mode_keeps_only_the_mtp_layer_minus_its_indexer() {
        assert_eq!(classify("model.layers.78.eh_proj.weight", 78, true, false), TensorKind::Q);
        assert_eq!(classify("model.layers.77.mlp.gate_proj.weight", 78, true, false), TensorKind::Skip); // a real layer, not layer 78
        assert_eq!(classify("model.layers.78.self_attn.indexer.wk.weight", 78, true, false), TensorKind::Skip);
    }

    #[test]
    fn keep_idx_mode_keeps_only_indexer_tensors_within_real_layers() {
        assert_eq!(classify("model.layers.5.self_attn.indexer.wk.weight", 78, false, true), TensorKind::Q);
        assert_eq!(classify("model.layers.5.self_attn.indexer.q_norm.weight", 78, false, true), TensorKind::F32);
        assert_eq!(classify("model.layers.5.mlp.down_proj.weight", 78, false, true), TensorKind::Skip); // not an indexer tensor
        assert_eq!(classify("model.layers.78.self_attn.indexer.wk.weight", 78, false, true), TensorKind::Skip); // out of real-layer range
    }

    #[test]
    fn global_and_per_layer_f32_only_tensors_are_never_quantized() {
        assert_eq!(classify("model.layers.5.mlp.gate.weight", 78, false, false), TensorKind::F32); // router itself, NOT gate_proj
        assert_eq!(classify("model.layers.5.mlp.gate.e_score_correction_bias", 78, false, false), TensorKind::F32);
        assert_eq!(classify("model.layers.5.input_layernorm.weight", 78, false, false), TensorKind::F32);
        assert_eq!(classify("model.norm.weight", 78, false, false), TensorKind::F32);
    }

    #[test]
    fn embed_and_lm_head_get_the_io_bucket() {
        assert_eq!(classify("model.embed_tokens.weight", 78, false, false), TensorKind::Io);
        assert_eq!(classify("lm_head.weight", 78, false, false), TensorKind::Io);
    }

    #[test]
    fn per_tensor_class_buckets_resolve_in_priority_order() {
        assert_eq!(classify("model.layers.5.mlp.experts.3.gate_proj.weight", 78, false, false), TensorKind::X);
        assert_eq!(classify("model.layers.5.mlp.shared_experts.gate_proj.weight", 78, false, false), TensorKind::Sh);
        assert_eq!(classify("model.layers.5.self_attn.o_proj.weight", 78, false, false), TensorKind::O);
        assert_eq!(classify("model.layers.5.self_attn.kv_b_proj.weight", 78, false, false), TensorKind::Kvb);
        assert_eq!(classify("model.layers.5.self_attn.q_a_proj.weight", 78, false, false), TensorKind::Attn);
        assert_eq!(classify("model.layers.5.self_attn.q_b_proj.weight", 78, false, false), TensorKind::Attn);
        assert_eq!(classify("model.layers.5.self_attn.kv_a_proj_with_mqa.weight", 78, false, false), TensorKind::Attn);
        assert_eq!(classify("model.layers.0.mlp.gate_proj.weight", 78, false, false), TensorKind::Dmlp);
        assert_eq!(classify("model.layers.0.mlp.up_proj.weight", 78, false, false), TensorKind::Dmlp);
        assert_eq!(classify("model.layers.0.mlp.down_proj.weight", 78, false, false), TensorKind::Dmlp);
    }

    #[test]
    fn unrecognized_weight_falls_back_to_q_and_unrecognized_non_weight_falls_back_to_f32() {
        assert_eq!(classify("model.layers.5.self_attn.some_new_proj.weight", 78, false, false), TensorKind::Q);
        assert_eq!(classify("model.layers.5.self_attn.some_new_bias", 78, false, false), TensorKind::F32);
    }
}
