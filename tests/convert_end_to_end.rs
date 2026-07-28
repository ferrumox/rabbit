//! The capstone check for `glm52::convert`: converts the REAL tiny GLM-5.2 oracle checkpoint
//! (`tests/oracle/glm_tiny/`, the same fixture `teacher_forcing.rs` uses for inference
//! correctness) through rabbit's own converter, then loads and RUNS the converted output
//! through rabbit's normal `glm52::model::Model::load` + `generate::step` — proving the
//! converter's output isn't just "written without erroring" but an actually loadable, runnable
//! checkpoint in rabbit's own real format. Skips (not fails) when the fixture is absent, same
//! policy as `teacher_forcing.rs`.

use rabbit::generate::{self, ExpertCaches, KvState};
use rabbit::glm52::convert::shard::{convert_shard, glm_classifier, BitsMap, ConvertOpts};
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
fn converted_glm_tiny_checkpoint_loads_and_runs_through_rabbit_s_own_engine() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracle/glm_tiny");
    if !src.is_dir() {
        eprintln!("skipping: tests/oracle/glm_tiny is absent (see tests/oracle/make_glm_oracle.py)");
        return;
    }
    let n_layers: usize = {
        let cfg: serde_json::Value = serde_json::from_str(&fs::read_to_string(src.join("config.json")).unwrap()).unwrap();
        cfg["num_hidden_layers"].as_u64().unwrap() as usize
    };

    let out_dir = TempDir(std::env::temp_dir().join("rabbit_test_convert_end_to_end"));
    fs::create_dir_all(&out_dir.0).unwrap();

    let opts = ConvertOpts { n_layers, ebits: 4, io_bits: 8, xbits: 4, keep_mtp: false, keep_idx: false, group_size: 0, bits_map: BitsMap::default(), overrides: vec![] };
    let out = convert_shard(&src, &opts, glm_classifier(&opts), None).expect("convert_shard failed on the real glm_tiny fixture");
    assert!(!out.is_empty(), "conversion produced no tensors at all");
    write_safetensors(&out_dir.0.join("out-00000.safetensors"), &out).unwrap();
    fs::copy(src.join("config.json"), out_dir.0.join("config.json")).unwrap();

    // Load the CONVERTED checkpoint through rabbit's real loader and run one real step --
    // dbits=4 matches the ebits this conversion just quantized resident weights at.
    let model = Model::load(&out_dir.0, 4, 4).expect("rabbit's own loader must accept the converter's output");
    let shards = Shards::open(&out_dir.0).unwrap();
    let mut caches = ExpertCaches::new(&model, 4);
    let mut kv = KvState::new(&model);
    let logits = generate::step(&model, &shards, &mut caches, &mut kv, &[1, 2, 3], 0).expect("a real forward step must succeed on the converted checkpoint");
    assert_eq!(logits.len(), model.cfg.vocab as usize);
    assert!(logits.iter().all(|v| v.is_finite()), "converted checkpoint must not produce NaN/inf logits");
}
