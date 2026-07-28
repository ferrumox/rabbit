//! The grouped-scale counterpart of `convert_end_to_end.rs`: converts the real tiny GLM-5.2
//! oracle checkpoint with `--group-size` on (`QTKind::I4Grouped`, the format this session found
//! rabbit's runtime couldn't read at all until `Cfg::group_size`/`QT::from_packed_grouped`/
//! `matmul_i4_grouped` were added), stamps `rabbit_group_size` into the output `config.json` the
//! same way the real CLIs do, and confirms `Model::load` + `generate::step` actually run on it —
//! not just that `convert_shard` produces bytes, but that the FULL round trip (write grouped,
//! read grouped, matmul grouped) works. Skips (not fails) when the fixture is absent, same
//! policy as `convert_end_to_end.rs`.

use rabbit::convert::config::copy_config_with_group_size;
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
fn grouped_conversion_of_glm_tiny_loads_and_runs_through_rabbit_s_own_engine() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracle/glm_tiny");
    if !src.is_dir() {
        eprintln!("skipping: tests/oracle/glm_tiny is absent (see tests/oracle/make_glm_oracle.py)");
        return;
    }
    let n_layers: usize = {
        let cfg: serde_json::Value = serde_json::from_str(&fs::read_to_string(src.join("config.json")).unwrap()).unwrap();
        cfg["num_hidden_layers"].as_u64().unwrap() as usize
    };

    let out_dir = TempDir(std::env::temp_dir().join("rabbit_test_convert_grouped_end_to_end"));
    fs::create_dir_all(&out_dir.0).unwrap();

    // group_size=16 against glm_tiny's real dims (hidden=128, intermediate=64,
    // moe_intermediate=32) gives 8/4/2 groups on different tensors -- exercises more than one
    // group-count shape, not just a single lucky case.
    let group_size = 16usize;
    let opts = ConvertOpts { n_layers, ebits: 4, io_bits: 8, xbits: 4, keep_mtp: false, keep_idx: false, group_size, bits_map: BitsMap::default(), overrides: vec![] };
    let out = convert_shard(&src, &opts, glm_classifier(&opts), None).expect("convert_shard failed on the real glm_tiny fixture");
    assert!(!out.is_empty(), "conversion produced no tensors at all");
    write_safetensors(&out_dir.0.join("out-00000.safetensors"), &out).unwrap();
    copy_config_with_group_size(&src.join("config.json"), &out_dir.0, group_size).unwrap();

    let model = Model::load(&out_dir.0, 4, 4).expect("rabbit's own loader must accept a grouped-scale checkpoint");
    assert_eq!(model.cfg.group_size, group_size as i32, "Model::load must pick up rabbit_group_size from config.json");

    let shards = Shards::open(&out_dir.0).unwrap();
    let mut caches = ExpertCaches::new(&model, 4);
    let mut kv = KvState::new(&model);
    let logits = generate::step(&model, &shards, &mut caches, &mut kv, &[1, 2, 3], 0).expect("a real forward step must succeed on a grouped-scale checkpoint");
    assert_eq!(logits.len(), model.cfg.vocab as usize);
    assert!(logits.iter().all(|v| v.is_finite()), "grouped-scale checkpoint must not produce NaN/inf logits");
}
