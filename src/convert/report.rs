//! The "log de clasificación" from `resumen-conversor-rabbit.md`: how many tensors fall into
//! each `TensorKind` bucket, plus a few example names per bucket — printed BEFORE any output is
//! written, so a converter run can be sanity-checked (does `Io` really only catch
//! `embed_tokens`/`lm_head`? did any tensor unexpectedly land in the generic `Q` bucket that
//! should have hit a more specific one?) without waiting for a full conversion to finish first.

use crate::convert::classify::TensorKind;
use crate::safetensors::Shards;
use std::collections::BTreeMap;

const MAX_EXAMPLES_PER_BUCKET: usize = 5;

#[derive(Default, Debug, Clone)]
pub struct BucketStats {
    pub count: usize,
    /// The first few tensor names that landed here — enough for a human to eyeball "does this
    /// bucket look right?" without dumping every one of possibly tens of thousands of names.
    pub examples: Vec<String>,
}

#[derive(Default, Debug, Clone)]
pub struct ClassificationReport {
    pub buckets: BTreeMap<TensorKind, BucketStats>,
    pub total: usize,
}

impl ClassificationReport {
    /// Human-readable summary — one line per non-empty bucket, sorted by `TensorKind`'s
    /// declaration order (`Skip`/`Consumed` first, since those are usually the least
    /// interesting to a human scanning for surprises; the "real" precision buckets `F32`/`Io`/
    /// `X`/`Q` and friends follow).
    pub fn summary(&self) -> String {
        let mut out = format!("{} tensors classified:\n", self.total);
        for (kind, stats) in &self.buckets {
            out.push_str(&format!("  {kind:?}: {}\n", stats.count));
            for ex in &stats.examples {
                out.push_str(&format!("    e.g. {ex}\n"));
            }
        }
        out
    }
}

/// Classifies every tensor in `shards` via `classify_fn`, building a per-bucket count + example
/// list — reusable for EITHER converter (`classify_fn` can be `glm52::convert::classify::classify`
/// wrapped in a closure, or `classify::classify_generic`), since it only depends on `TensorKind`
/// values, never on how they were produced.
pub fn classify_report(shards: &Shards, classify_fn: impl Fn(&str, usize) -> TensorKind) -> ClassificationReport {
    let mut report = ClassificationReport::default();
    for t in shards.tensors() {
        let kind = classify_fn(&t.name, t.shape.len());
        report.total += 1;
        let bucket = report.buckets.entry(kind).or_default();
        bucket.count += 1;
        if bucket.examples.len() < MAX_EXAMPLES_PER_BUCKET {
            bucket.examples.push(t.name.clone());
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::classify::classify_generic;
    use serde_json::json;
    use std::fs;

    fn build_fixture(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(name);
        fs::create_dir_all(&dir).unwrap();
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        let mut add = |header: &mut serde_json::Map<String, serde_json::Value>, name: &str, shape: Vec<u64>| {
            let n: usize = shape.iter().product::<u64>().max(1) as usize;
            let bytes: Vec<u8> = vec![0u8; n * 4];
            let start = data.len() as u64;
            data.extend_from_slice(&bytes);
            let end = data.len() as u64;
            header.insert(name.to_string(), json!({"dtype": "F32", "shape": shape, "data_offsets": [start, end]}));
        };
        add(&mut header, "model.embed_tokens.weight", vec![16, 8]);
        add(&mut header, "model.norm.weight", vec![8]);
        add(&mut header, "model.layers.0.input_layernorm.weight", vec![8]);
        add(&mut header, "model.layers.0.mlp.gate.weight", vec![4, 8]);
        add(&mut header, "model.layers.0.mlp.experts.0.gate_proj.weight", vec![4, 8]);
        add(&mut header, "model.layers.0.mlp.experts.1.gate_proj.weight", vec![4, 8]);
        add(&mut header, "model.layers.0.self_attn.q_proj.weight", vec![8, 8]);

        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_bytes);
        out.extend_from_slice(&data);
        fs::write(dir.join("shard.safetensors"), out).unwrap();
        dir
    }

    #[test]
    fn counts_and_examples_match_the_expected_buckets() {
        let dir = build_fixture("rabbit_test_convert_report_counts");
        let shards = Shards::open(&dir).unwrap();
        let report = classify_report(&shards, classify_generic);

        assert_eq!(report.total, 7);
        assert_eq!(report.buckets[&TensorKind::Io].count, 1);
        assert_eq!(report.buckets[&TensorKind::F32].count, 3); // norm, input_layernorm, mlp.gate.weight
        assert_eq!(report.buckets[&TensorKind::X].count, 2); // the two experts
        assert_eq!(report.buckets[&TensorKind::Q].count, 1); // q_proj

        assert_eq!(report.buckets[&TensorKind::Io].examples, vec!["model.embed_tokens.weight".to_string()]);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn summary_is_non_empty_and_mentions_every_populated_bucket() {
        let dir = build_fixture("rabbit_test_convert_report_summary");
        let shards = Shards::open(&dir).unwrap();
        let report = classify_report(&shards, classify_generic);
        let s = report.summary();
        assert!(s.contains("7 tensors classified"));
        assert!(s.contains("Io: 1"));
        assert!(s.contains("X: 2"));
        fs::remove_dir_all(&dir).ok();
    }
}
