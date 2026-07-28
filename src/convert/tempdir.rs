//! A directory holding a symlink to exactly ONE shard file — `Shards::open`/`convert_shard`
//! want a directory (they glob every `*.safetensors` file in it), but the disk-safe conversion
//! workflow processes shards one at a time; this bridges the two without copying multi-GB
//! tensor data. Shared by every converter CLI (`glm52::convert`'s and the generic one) so the
//! symlink-target-must-be-absolute gotcha (see `TmpDir::new`'s doc) only needs fixing once.

use std::fs;
use std::path::{Path, PathBuf};

pub struct TmpDir(pub PathBuf);

impl Drop for TmpDir {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).ok();
    }
}

impl TmpDir {
    /// Creates a fresh temp directory containing a symlink to `shard`.
    ///
    /// Canonicalizes `shard` to an absolute path first — a symlink's target is resolved
    /// relative to the SYMLINK's own directory, not the caller's cwd, so a relative `shard`
    /// path (e.g. from walking a relative `--indir`) would otherwise silently point nowhere
    /// once placed inside this temp dir.
    pub fn with_one_shard(shard: &Path) -> Result<TmpDir, String> {
        let dir = std::env::temp_dir().join(format!("rabbit_convert_{}_{}", std::process::id(), next_unique_id()));
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let dest = dir.join(shard.file_name().ok_or("shard path has no file name")?);
        let abs_shard = fs::canonicalize(shard).map_err(|e| e.to_string())?;
        std::os::unix::fs::symlink(&abs_shard, &dest).map_err(|e| e.to_string())?;
        Ok(TmpDir(dir))
    }
}

/// A monotonic per-process counter, not a random number — `std::process::id()` alone collides
/// when the SAME process converts many shards in sequence (each call reused the same dir name,
/// only safe because the previous `TmpDir` had already been dropped/cleaned up by then); this
/// makes every call's directory name unique so nothing depends on that ordering assumption
/// holding forever (e.g. a future caller processing shards concurrently).
fn next_unique_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}
