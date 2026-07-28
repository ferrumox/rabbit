//! Copies a source checkpoint's `config.json` into a converter's output dir, stamping
//! `rabbit_group_size` onto it when the conversion used grouped int4 — the one piece of
//! information `glm52::config::Cfg`/`kimi_linear::config::Cfg` need at LOAD time to interpret a
//! `.qs` sidecar's scale layout correctly (see `quant.rs`'s `QTKind::I4Grouped` doc for why this
//! can't be inferred after the fact from the sidecar's own length). Shared by both converter
//! CLIs (`convert`, `convert_fp8_to_int4`) since neither's output format differs here.

use serde_json::Value;
use std::fs;
use std::path::Path;

/// Reads `src` (a source checkpoint's `config.json`), writes it to `dst_dir/config.json`, and —
/// only when `group_size > 0` — sets/overwrites its `rabbit_group_size` field. A no-op copy
/// (verbatim `fs::copy`) when `group_size == 0`, so an ungrouped conversion never touches bytes
/// it doesn't need to.
pub fn copy_config_with_group_size(src: &Path, dst_dir: &Path, group_size: usize) -> Result<(), String> {
    let dst = dst_dir.join("config.json");
    if group_size == 0 {
        fs::copy(src, &dst).map_err(|e| e.to_string())?;
        return Ok(());
    }
    let text = fs::read_to_string(src).map_err(|e| e.to_string())?;
    let mut v: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let obj = v.as_object_mut().ok_or_else(|| format!("{}: not a JSON object", src.display()))?;
    obj.insert("rabbit_group_size".to_string(), Value::from(group_size as u64));
    fs::write(&dst, serde_json::to_string_pretty(&v).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_size_zero_copies_verbatim() {
        let dir = std::env::temp_dir().join("rabbit_test_convert_config_verbatim");
        fs::create_dir_all(&dir).unwrap();
        let src = dir.join("in.json");
        fs::write(&src, r#"{"a":1}"#).unwrap();
        let outdir = dir.join("out");
        fs::create_dir_all(&outdir).unwrap();

        copy_config_with_group_size(&src, &outdir, 0).unwrap();
        let got: Value = serde_json::from_str(&fs::read_to_string(outdir.join("config.json")).unwrap()).unwrap();
        assert_eq!(got, serde_json::json!({"a": 1}));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn group_size_nonzero_stamps_the_field() {
        let dir = std::env::temp_dir().join("rabbit_test_convert_config_stamped");
        fs::create_dir_all(&dir).unwrap();
        let src = dir.join("in.json");
        fs::write(&src, r#"{"a":1,"rabbit_group_size":999}"#).unwrap();
        let outdir = dir.join("out");
        fs::create_dir_all(&outdir).unwrap();

        copy_config_with_group_size(&src, &outdir, 128).unwrap();
        let got: Value = serde_json::from_str(&fs::read_to_string(outdir.join("config.json")).unwrap()).unwrap();
        assert_eq!(got["a"], 1);
        assert_eq!(got["rabbit_group_size"], 128); // overwrites a stale value from the source, doesn't merge/keep it

        fs::remove_dir_all(&dir).ok();
    }
}
