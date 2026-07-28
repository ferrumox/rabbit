//! The capstone check for the V1 architecture-agnostic converter (`rabbit::convert`): runs the
//! SAME real tiny GLM-5.2 oracle checkpoint `convert_end_to_end.rs` uses through
//! `classify_generic` (not GLM-5.2's own hand-tuned classifier) and confirms the result still
//! loads and RUNS through rabbit's real engine, producing finite logits — proof the generic
//! heuristic classifies a real architecture's tensors well enough to stay usable, not just that
//! it runs without erroring. Skips (not fails) when the fixture is absent, same policy as
//! `teacher_forcing.rs`/`convert_end_to_end.rs`.

use rabbit::convert::classify::classify_generic;
use rabbit::convert::report::classify_report;
use rabbit::generate::{self, ExpertCaches, KvState};
use rabbit::glm52::convert::shard::{convert_shard, BitsMap, ConvertOpts};
use rabbit::glm52::convert::writer::write_safetensors;
use rabbit::glm52::model::Model;
use rabbit::safetensors::Shards;
use std::fs;
use std::path::Path;

struct TempDir(std::path::PathBuf);
impl Drop for TempDir {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).ok();
    }
}

#[test]
fn generic_heuristic_classifies_real_glm52_tensors_into_a_loadable_runnable_checkpoint() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracle/glm_tiny");
    if !src.is_dir() {
        eprintln!("skipping: tests/oracle/glm_tiny is absent (see tests/oracle/make_glm_oracle.py)");
        return;
    }

    let shards = Shards::open(&src).unwrap();
    let report = classify_report(&shards, classify_generic);
    assert!(report.total > 0, "must classify at least one tensor");
    // Sanity-check the heuristic actually separated the buckets that matter for a real GLM-5.2
    // checkpoint, not just dumped everything into one bucket -- io_bits/xbits only matter if
    // Io/X are non-empty, and F32 must never be zero (every real checkpoint has norms).
    assert!(report.buckets.get(&rabbit::convert::classify::TensorKind::Io).is_some_and(|b| b.count > 0), "embed/lm_head must land in Io");
    assert!(report.buckets.get(&rabbit::convert::classify::TensorKind::X).is_some_and(|b| b.count > 0), "routed experts must land in X");
    assert!(report.buckets.get(&rabbit::convert::classify::TensorKind::F32).is_some_and(|b| b.count > 0), "norms must land in F32");

    let out_dir = TempDir(std::env::temp_dir().join("rabbit_test_convert_generic_end_to_end"));
    fs::create_dir_all(&out_dir.0).unwrap();

    let opts = ConvertOpts { n_layers: 0, ebits: 4, io_bits: 8, xbits: 4, keep_mtp: false, keep_idx: false, group_size: 0, bits_map: BitsMap::default(), overrides: vec![] };
    let out = convert_shard(&src, &opts, classify_generic, None).expect("convert_shard failed with the generic classifier");
    assert!(!out.is_empty());
    write_safetensors(&out_dir.0.join("out-00000.safetensors"), &out).unwrap();
    fs::copy(src.join("config.json"), out_dir.0.join("config.json")).unwrap();

    let model = Model::load(&out_dir.0, 4, 4).expect("rabbit's own loader must accept the generically-converted output");
    let out_shards = Shards::open(&out_dir.0).unwrap();
    let mut caches = ExpertCaches::new(&model, 4);
    let mut kv = KvState::new(&model);
    let logits = generate::step(&model, &out_shards, &mut caches, &mut kv, &[1, 2, 3], 0).expect("a real forward step must succeed");
    assert_eq!(logits.len(), model.cfg.vocab as usize);
    assert!(logits.iter().all(|v| v.is_finite()), "must not produce NaN/inf logits");
}
