//! Minimal Hugging Face Hub REST API client — just the two calls `convert_fp8_to_int4.py`'s
//! `--repo` mode needs (`HfApi().repo_info(...)` and a raw GET of `model.safetensors.index.json`
//! for `--mtp`/`--indexer`), reimplemented directly against HF's public HTTP API rather than
//! pulling in the Python `huggingface_hub` package's Rust equivalent (none of this crate's other
//! dependencies talk to any remote API at all — a whole SDK for two read-only calls would be a
//! lot of surface for what's needed here).
//!
//! **Not exercised against the real Hugging Face API in this session** (would need live network
//! access and, for a real repo, multi-GB transfers) — see `download.rs`'s own doc for the same
//! caveat on the transfer side. The endpoint shapes here match HF's documented public API
//! (`GET /api/models/{repo}?blobs=true` for sibling file sizes, `GET /{repo}/resolve/main/{path}`
//! for raw file content — the same `resolve/main` URL `download.rs`'s `download_retry` already
//! targets) as of this writing, not verified live.

use serde::Deserialize;
use std::collections::HashMap;
use std::fmt;

#[derive(Debug)]
pub enum HfApiError {
    Http(String),
    Json(String),
}

impl fmt::Display for HfApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HfApiError::Http(e) => write!(f, "huggingface api: {e}"),
            HfApiError::Json(e) => write!(f, "huggingface api: {e}"),
        }
    }
}

impl std::error::Error for HfApiError {}

#[derive(Deserialize)]
struct Sibling {
    rfilename: String,
    size: Option<u64>,
}

#[derive(Deserialize)]
struct RepoInfo {
    siblings: Vec<Sibling>,
}

/// `https://huggingface.co/{repo}/resolve/main/{path}` — the same raw-file URL scheme
/// `download.rs::download_retry` downloads from, factored out here since `hf_api.rs`'s own
/// `fetch_index_json` needs it too.
pub fn resolve_url(repo: &str, path: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/main/{path}")
}

/// Every `.safetensors` sibling file in `repo` and its size in bytes — port of
/// `HfApi().repo_info(repo, files_metadata=True)` + `SIZES.update({s.rfilename: s.size for s in
/// info.siblings if s.size})`, filtered to `.safetensors` files and sorted by name (matching the
/// Python source's own `sorted(s.rfilename for s in info.siblings if s.rfilename.endswith(...))`
/// at the call site) — returned already in the order shards get processed.
pub fn list_safetensors_shards(ag: &ureq::Agent, repo: &str) -> Result<Vec<(String, Option<u64>)>, HfApiError> {
    let url = format!("https://huggingface.co/api/models/{repo}?blobs=true");
    let mut resp = ag.get(&url).header("User-Agent", "rabbit-convert").call().map_err(|e| HfApiError::Http(e.to_string()))?;
    let info: RepoInfo = resp.body_mut().read_json().map_err(|e| HfApiError::Json(e.to_string()))?;
    let mut shards: Vec<(String, Option<u64>)> =
        info.siblings.into_iter().filter(|s| s.rfilename.ends_with(".safetensors")).map(|s| (s.rfilename, s.size)).collect();
    shards.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(shards)
}

/// `model.safetensors.index.json`'s `weight_map` (tensor name -> shard filename) — port of the
/// `--mtp`/`--indexer` modes' own raw `urllib.request.urlopen(.../resolve/main/model.safetensors
/// .index.json)` call (deliberately NOT going through the repo-info API, matching the source).
pub fn fetch_weight_map(ag: &ureq::Agent, repo: &str) -> Result<HashMap<String, String>, HfApiError> {
    let url = resolve_url(repo, "model.safetensors.index.json");
    let mut resp = ag.get(&url).header("User-Agent", "rabbit-convert").call().map_err(|e| HfApiError::Http(e.to_string()))?;
    #[derive(Deserialize)]
    struct Index {
        weight_map: HashMap<String, String>,
    }
    let idx: Index = resp.body_mut().read_json().map_err(|e| HfApiError::Json(e.to_string()))?;
    Ok(idx.weight_map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_url_matches_the_documented_hf_scheme() {
        assert_eq!(resolve_url("zai-org/GLM-5.2-FP8", "config.json"), "https://huggingface.co/zai-org/GLM-5.2-FP8/resolve/main/config.json");
    }
}
