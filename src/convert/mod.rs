//! Rabbit's own architecture-agnostic checkpoint converter — the "V1 universal converter" from
//! `resumen-conversor-rabbit.md`'s design discussion, distinct from `glm52::convert` (a faithful
//! port of colibrì's GLM-5.2-specific `convert_fp8_to_int4.py`, built first to learn the real
//! conversion pipeline against a known-correct reference before generalizing it).
//!
//! What lives here is genuinely architecture-agnostic and shared by BOTH converters:
//! `classify::TensorKind` (the precision-bucket enum), dequant (`glm52::convert::dequant` — FP8/
//! NVFP4 detection depends only on tensor dtype + sidecar naming, never on any GLM-5.2-specific
//! tensor name, so it's reused from there as-is rather than duplicated), the safetensors writer,
//! and `glm52::convert::shard::convert_shard`'s orchestration (parameterized over an injectable
//! classifier — see that module's doc for why one function now serves both converters).
//!
//! `classify::classify_generic` is the new piece: a NAME-pattern heuristic that doesn't assume
//! GLM-5.2's specific tensor names (`kv_b_proj`, `shared_experts`, etc.), so it can classify a
//! checkpoint from ANY architecture rabbit hasn't even added inference support for yet — the
//! converter can run ahead of `model.rs`'s own per-architecture work, not behind it.

pub mod classify;
pub mod config;
pub mod quality;
pub mod report;
pub mod tempdir;

/// Parses one `--bits-override <pattern>:<bits>` CLI argument, shared by both converter
/// binaries. `pattern` is a plain name substring (matched via `str::contains`, the same
/// convention `classify_generic`'s own rules use — not a regex), `bits` an integer. Splits on
/// the LAST `:` so a hypothetical `:`-containing pattern stays unambiguous (safetensors tensor
/// names never contain one in practice, but nothing enforces that).
pub fn parse_bits_override(s: &str) -> Result<(String, u8), String> {
    let (pattern, bits) = s.rsplit_once(':').ok_or_else(|| format!("--bits-override {s:?}: expected <pattern>:<bits>"))?;
    let bits: u8 = bits.parse().map_err(|e| format!("--bits-override {s:?}: {e}"))?;
    if pattern.is_empty() {
        return Err(format!("--bits-override {s:?}: pattern must not be empty"));
    }
    Ok((pattern.to_string(), bits))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_well_formed_override() {
        assert_eq!(parse_bits_override("self_attn.b_proj:8").unwrap(), ("self_attn.b_proj".to_string(), 8));
    }

    #[test]
    fn splits_on_the_last_colon_so_a_colon_in_the_pattern_stays_intact() {
        assert_eq!(parse_bits_override("weird:pattern:8").unwrap(), ("weird:pattern".to_string(), 8));
    }

    #[test]
    fn rejects_missing_colon_empty_pattern_or_unparseable_bits() {
        assert!(parse_bits_override("no_colon_here").is_err());
        assert!(parse_bits_override(":8").is_err());
        assert!(parse_bits_override("pattern:not_a_number").is_err());
    }
}
