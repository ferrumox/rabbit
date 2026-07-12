//! Persistent expert-usage histogram (`.rabbit_usage`, colibrì's `.coli_usage` equivalent) —
//! how many times each `(layer, expert)` pair got routed to, accumulated across process
//! restarts. `generate.rs`'s `ExpertCaches::warm_start` reads this at startup to decide which
//! experts are worth PIN CANDIDATES (`expert_cache.rs`'s `ExpertCache::mark_pin_candidates`) —
//! marked, not eagerly loaded; see that module's doc for why lazy/sticky promotion replaced
//! colibrì's eager `pin_load` here. `save_usage` writes this file back at turn/response
//! boundaries.
//!
//! Plain text, not a binary format like `kv_session.rs`: this file is small (at most
//! `moe_layers * n_experts` sparse rows), written once per turn (not per token), and
//! human-inspectable for free — none of the reasons `kv_session.rs` needed a binary format
//! (large per-turn tensor payloads, append-only perf) apply here. One line per nonzero
//! `(layer, expert)` pair: `"{layer} {eid} {count}\n"`, no header/magic/version, mirroring
//! colibrì's own `stats_dump_q` format exactly.
//!
//! Parsing is permissive by design (unlike `kv_session.rs`'s strict `LoadError`): losing this
//! file, or a corrupt line in it, just means some experts don't get pre-pinned this run — the
//! live counters rebuild it from scratch and it's re-saved at the next turn boundary. A missing
//! or garbled usage file must never block generation from starting.

use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

pub(crate) const FILE_NAME: &str = ".rabbit_usage";
pub(crate) const HIST_THRESHOLD: u64 = 5_000;
const CONFIDENCE_DENOM: f64 = 200_000.0;
const PIN_FRACTION: f64 = 0.5;

pub(crate) fn usage_path(model_dir: &Path) -> PathBuf {
    model_dir.join(FILE_NAME)
}

pub(crate) fn confidence(hist: u64) -> f32 {
    (hist as f64 / CONFIDENCE_DENOM).min(1.0) as f32
}

pub(crate) fn pin_budget(cache_capacity: usize, confidence: f32) -> usize {
    (cache_capacity as f64 * PIN_FRACTION * confidence as f64).floor() as usize
}

/// Reads `"{layer} {eid} {count}"` lines into a `(layer, eid) -> count` map. A missing file is
/// just an empty map (fresh checkpoint, or `--no-usage-cache` was used before); any other open
/// error or malformed line is logged to stderr and skipped — see the module doc for why this
/// never returns a `Result`.
pub(crate) fn load(path: &Path) -> HashMap<(usize, usize), u64> {
    let mut out = HashMap::new();
    let text = match fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return out,
        Err(e) => {
            eprintln!("usage cache: failed to read {}: {e}", path.display());
            return out;
        }
    };
    for (lineno, line) in text.lines().enumerate() {
        let mut parts = line.split_whitespace();
        let parsed = (|| {
            let layer: usize = parts.next()?.parse().ok()?;
            let eid: usize = parts.next()?.parse().ok()?;
            let count: u64 = parts.next()?.parse().ok()?;
            if parts.next().is_some() {
                return None;
            }
            Some((layer, eid, count))
        })();
        match parsed {
            Some((layer, eid, count)) => {
                out.insert((layer, eid), count);
            }
            None => eprintln!("usage cache: skipping malformed line {} in {}", lineno + 1, path.display()),
        }
    }
    out
}

/// Overwrites `path` with a full snapshot of `entries` (sorted by `(layer, eid)` for a
/// diffable file), atomically via a temp file + rename in the same directory — mirrors
/// colibrì's `stats_dump_q`. Always a full dump, not an incremental delta: the whole histogram
/// is small enough that this is cheap even at "once per HTTP response" cadence under `--serve`.
pub(crate) fn save(path: &Path, entries: impl Iterator<Item = (usize, usize, u64)>) -> io::Result<()> {
    let mut rows: Vec<(usize, usize, u64)> = entries.filter(|&(_, _, c)| c > 0).collect();
    rows.sort_unstable();

    let mut body = String::new();
    for (layer, eid, count) in rows {
        body.push_str(&layer.to_string());
        body.push(' ');
        body.push_str(&eid.to_string());
        body.push(' ');
        body.push_str(&count.to_string());
        body.push('\n');
    }

    // `.rabbit_usage`.with_extension("tmp") would yield `.tmp` (a leading-dot-only filename has
    // no stem as far as `Path::extension`/`with_extension` are concerned) -- build the sibling
    // path by hand instead.
    let tmp_name = format!("{}.tmp", path.file_name().and_then(|n| n.to_str()).unwrap_or(FILE_NAME));
    let tmp_path = path.with_file_name(tmp_name);

    let mut f = fs::File::create(&tmp_path)?;
    f.write_all(body.as_bytes())?;
    f.flush()?;
    drop(f);
    fs::rename(&tmp_path, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(name);
            fs::create_dir_all(&dir).unwrap();
            TempDir(dir)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).ok();
        }
    }

    #[test]
    fn save_then_load_round_trips_counts() {
        let dir = TempDir::new("rabbit_test_usage_cache_roundtrip");
        let path = usage_path(&dir.0);
        let entries = vec![(0usize, 1usize, 10u64), (0, 2, 20), (5, 7, 999)];
        save(&path, entries.clone().into_iter()).unwrap();

        let loaded = load(&path);
        assert_eq!(loaded.len(), 3);
        for (layer, eid, count) in entries {
            assert_eq!(loaded.get(&(layer, eid)), Some(&count));
        }
    }

    #[test]
    fn save_skips_zero_counts() {
        let dir = TempDir::new("rabbit_test_usage_cache_zero_skip");
        let path = usage_path(&dir.0);
        save(&path, vec![(0usize, 1usize, 0u64), (0, 2, 5)].into_iter()).unwrap();
        let loaded = load(&path);
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.get(&(0, 2)), Some(&5));
    }

    #[test]
    fn load_skips_corrupt_lines_and_keeps_the_rest() {
        let dir = TempDir::new("rabbit_test_usage_cache_corrupt");
        let path = usage_path(&dir.0);
        fs::write(&path, "0 1 10\ngarbage line\n0 2\n3 4 20\n\n").unwrap();
        let loaded = load(&path);
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.get(&(0, 1)), Some(&10));
        assert_eq!(loaded.get(&(3, 4)), Some(&20));
    }

    #[test]
    fn load_missing_file_returns_empty_map() {
        let dir = TempDir::new("rabbit_test_usage_cache_missing");
        let loaded = load(&usage_path(&dir.0));
        assert!(loaded.is_empty());
    }

    #[test]
    fn confidence_and_pin_budget_math() {
        assert_eq!(confidence(0), 0.0);
        assert!((confidence(100_000) - 0.5).abs() < 1e-6);
        assert_eq!(confidence(200_000), 1.0);
        assert_eq!(confidence(400_000), 1.0); // clamped

        assert_eq!(pin_budget(64, 1.0), 32);
        assert_eq!(pin_budget(64, 0.5), 16);
        assert_eq!(pin_budget(3, 1.0), 1); // floor(1.5)
        assert_eq!(pin_budget(64, 0.0), 0);
    }
}
