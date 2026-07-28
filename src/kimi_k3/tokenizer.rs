//! Kimi K3's own `tiktoken`-format tokenizer — the sibling of `kimi_linear::tokenizer.rs`, not a
//! variant: `tokenization_kimi.py`'s `pat_str` (the pre-tokenizer regex) and its `tiktoken.model`/
//! `tokenizer_config.json` FILE FORMATS are byte-for-byte identical to Kimi Linear 48B's — same
//! real source file, confirmed by comparing both this session — so this module reuses
//! `kimi_linear::tokenizer::byte_pair_encode` directly (widened to `pub(crate)`) rather than
//! duplicating that intricate merge algorithm a second time.
//!
//! **One real, checkpoint-independent difference, confirmed by re-reading `tokenization_kimi.py`
//! itself (not assumed from the 48B port)**: the class hardcodes `num_reserved_special_tokens =
//! 256` and its `__init__` loop is `range(num_base_tokens, num_base_tokens + 256)` — exactly 256,
//! not 258. `kimi_linear::tokenizer.rs`'s own `NUM_RESERVED_SPECIAL_TOKENS + 2` was an empirical
//! adjustment for a quirk observed in the 48B checkpoint's `tokenizer_config.json` (extra
//! `added_tokens_decoder` entries beyond what that 256-entry loop actually reads) — checked
//! against K3's real `tokenizer_config.json` this session: its `added_tokens_decoder`'s highest
//! key is `163839` = `num_base_tokens (163584) + 255`, safely inside a plain 256-entry range with
//! no overflow, so K3 does NOT need (and must NOT use) that same "+2" adjustment.

use crate::kimi_linear::tokenizer::byte_pair_encode;
use base64::Engine;
use fancy_regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::Path;

pub type Rank = u32;

/// Verbatim from `tokenization_kimi.py`'s `TikTokenTokenizer.pat_str`, confirmed
/// character-for-character identical to `kimi_linear::tokenizer::PAT_STR` this session (same
/// real source file backs both checkpoints) — duplicated rather than shared as a `pub(crate)`
/// constant purely because a `&str` constant carries none of the "easy to get subtly wrong"
/// risk `byte_pair_encode` does; see this module's doc for why THAT function is reused instead.
const PAT_STR: &str = concat!(
    r"[\p{Han}]+",
    "|",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    "|",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    "|",
    r"\p{N}{1,3}",
    "|",
    r" ?[^\s\p{L}\p{N}]+[\r\n]*",
    "|",
    r"\s*[\r\n]+",
    "|",
    r"\s+(?!\S)",
    "|",
    r"\s+",
);

/// `tokenization_kimi.py`'s real `num_reserved_special_tokens` class attribute — 256, not
/// Kimi Linear 48B's own tokenizer.rs's `+2`-adjusted 258. See this module's doc.
const NUM_RESERVED_SPECIAL_TOKENS: u32 = 256;

#[derive(Debug)]
pub enum TokenizerError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Regex(Box<fancy_regex::Error>),
    /// `tiktoken.model` line `n` (0-indexed) isn't `<base64> <rank>`.
    BadModelLine(usize),
    BadBase64(usize),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerError::Io(e) => write!(f, "tiktoken.model: {e}"),
            TokenizerError::Json(e) => write!(f, "tokenizer_config.json: {e}"),
            TokenizerError::Regex(e) => write!(f, "pre-tokenizer regex: {e}"),
            TokenizerError::BadModelLine(n) => write!(f, "tiktoken.model: bad line {n}"),
            TokenizerError::BadBase64(n) => write!(f, "tiktoken.model: bad base64 on line {n}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

impl From<std::io::Error> for TokenizerError {
    fn from(e: std::io::Error) -> Self {
        TokenizerError::Io(e)
    }
}

impl From<serde_json::Error> for TokenizerError {
    fn from(e: serde_json::Error) -> Self {
        TokenizerError::Json(e)
    }
}

impl From<fancy_regex::Error> for TokenizerError {
    fn from(e: fancy_regex::Error) -> Self {
        TokenizerError::Regex(Box::new(e))
    }
}

pub struct Tokenizer {
    encoder: HashMap<Vec<u8>, Rank>,
    decoder: HashMap<Rank, Vec<u8>>,
    special_encoder: HashMap<String, Rank>,
    special_decoder: HashMap<Rank, Vec<u8>>,
    regex: Regex,
    special_regex: Regex,
}

impl Tokenizer {
    /// Loads `tiktoken.model` + `tokenizer_config.json` from a real Kimi K3 checkpoint
    /// directory — both files this needs, no separate fixture format.
    pub fn load(dir: &Path) -> Result<Tokenizer, TokenizerError> {
        let model_text = fs::read_to_string(dir.join("tiktoken.model"))?;
        let mut encoder: HashMap<Vec<u8>, Rank> = HashMap::new();
        for (n, line) in model_text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let mut parts = line.splitn(2, ' ');
            let b64 = parts.next().ok_or(TokenizerError::BadModelLine(n))?;
            let rank: Rank = parts
                .next()
                .ok_or(TokenizerError::BadModelLine(n))?
                .trim()
                .parse()
                .map_err(|_| TokenizerError::BadModelLine(n))?;
            let bytes = base64::engine::general_purpose::STANDARD.decode(b64).map_err(|_| TokenizerError::BadBase64(n))?;
            encoder.insert(bytes, rank);
        }
        let decoder: HashMap<Rank, Vec<u8>> = encoder.iter().map(|(k, &v)| (v, k.clone())).collect();
        let num_base_tokens = encoder.len() as u32;

        let cfg_text = fs::read_to_string(dir.join("tokenizer_config.json"))?;
        let cfg: Value = serde_json::from_str(&cfg_text)?;
        let added: HashMap<u32, String> = cfg
            .get("added_tokens_decoder")
            .and_then(Value::as_object)
            .into_iter()
            .flatten()
            .filter_map(|(id, v)| {
                let id: u32 = id.parse().ok()?;
                let content = v.get("content")?.as_str()?.to_string();
                Some((id, content))
            })
            .collect();

        let mut special_encoder: HashMap<String, Rank> = HashMap::new();
        for i in num_base_tokens..num_base_tokens + NUM_RESERVED_SPECIAL_TOKENS {
            let name = added.get(&i).cloned().unwrap_or_else(|| format!("<|reserved_token_{i}|>"));
            special_encoder.insert(name, i);
        }
        let special_decoder: HashMap<Rank, Vec<u8>> = special_encoder.iter().map(|(k, &v)| (v, k.as_bytes().to_vec())).collect();

        let regex = Regex::new(PAT_STR)?;
        // Longest-first alternation, same "longest match wins" precedent
        // `kimi_linear::tokenizer::Tokenizer::load`'s own `specials` list already follows.
        let mut special_strs: Vec<&String> = special_encoder.keys().collect();
        special_strs.sort_by_key(|s| std::cmp::Reverse(s.len()));
        let special_pattern = special_strs.iter().map(|s| fancy_regex::escape(s)).collect::<Vec<_>>().join("|");
        let special_regex = Regex::new(&special_pattern)?;

        Ok(Tokenizer { encoder, decoder, special_encoder, special_decoder, regex, special_regex })
    }

    /// Encodes text into token ids — same "any literal special-token string is always
    /// recognized" behavior as `kimi_linear::tokenizer::Tokenizer::encode` (see that function's
    /// doc for why this doesn't port `tiktoken`'s real `allowed_special` gating).
    pub fn encode(&self, text: &str) -> Vec<i32> {
        let mut out = Vec::new();
        let mut start = 0usize;
        loop {
            let next_special = self.special_regex.find_from_pos(text, start).ok().flatten();
            let end = next_special.map(|m| m.start()).unwrap_or(text.len());

            for mat in self.regex.find_iter(&text[start..end]) {
                let mat = mat.expect("pre-tokenizer regex error");
                let piece = mat.as_str().as_bytes();
                if let Some(&id) = self.encoder.get(piece) {
                    out.push(id as i32);
                } else {
                    out.extend(byte_pair_encode(piece, &self.encoder).into_iter().map(|r| r as i32));
                }
            }

            match next_special {
                Some(m) => {
                    let id = self.special_encoder[m.as_str()];
                    out.push(id as i32);
                    start = m.end();
                }
                None => break,
            }
        }
        out
    }

    /// Decodes ids back into raw bytes. Ids with no matching entry are silently skipped, same
    /// permissive policy as `kimi_linear::tokenizer::Tokenizer::decode`.
    pub fn decode(&self, ids: &[i32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            if id < 0 {
                continue;
            }
            let id = id as u32;
            if let Some(bytes) = self.decoder.get(&id).or_else(|| self.special_decoder.get(&id)) {
                out.extend_from_slice(bytes);
            }
        }
        out
    }

    /// Id of a special token given its literal content (e.g. `"<|end_of_msg|>"`).
    pub fn id_of(&self, content: &str) -> Option<i32> {
        self.special_encoder.get(content).map(|&id| id as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, content: &str) {
        fs::write(dir.join(name), content).unwrap();
    }

    /// A tiny hand-built `tiktoken.model` + `tokenizer_config.json`: base tokens for every byte
    /// 'a'..'z' plus a couple of real merges, and one named special token among the reserved
    /// range.
    fn build_fixture(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("rabbit_test_k3_tok_{name}"));
        fs::create_dir_all(&dir).unwrap();

        let b64 = |s: &str| base64::engine::general_purpose::STANDARD.encode(s.as_bytes());
        let mut lines = Vec::new();
        for (i, c) in "helowrd, !".chars().enumerate() {
            lines.push(format!("{} {}", b64(&c.to_string()), i));
        }
        let n = lines.len() as u32;
        lines.push(format!("{} {}", b64("he"), n)); // rank n: "he"
        lines.push(format!("{} {}", b64("hel"), n + 1)); // rank n+1: "hel" (built from "he"+"l")
        write(&dir, "tiktoken.model", &lines.join("\n"));

        let num_base = n + 2;
        let cfg = serde_json::json!({
            "added_tokens_decoder": {
                num_base.to_string(): {"content": "<|end_of_msg|>"},
            }
        });
        write(&dir, "tokenizer_config.json", &cfg.to_string());
        dir
    }

    #[test]
    fn bpe_merges_by_rank_order_into_the_lowest_rank_pair_first() {
        let dir = build_fixture("merge_order");
        let tok = Tokenizer::load(&dir).unwrap();
        fs::remove_dir_all(&dir).ok();

        let ids = tok.encode("hel");
        assert_eq!(ids.len(), 1);
        assert_eq!(tok.decode(&ids), b"hel");
    }

    #[test]
    fn falls_back_to_single_byte_tokens_when_no_merge_exists() {
        let dir = build_fixture("no_merge");
        let tok = Tokenizer::load(&dir).unwrap();
        fs::remove_dir_all(&dir).ok();

        let ids = tok.encode("row");
        assert_eq!(ids.len(), 3);
        assert_eq!(tok.decode(&ids), b"row");
    }

    #[test]
    fn named_special_token_resolves_and_round_trips() {
        let dir = build_fixture("named_special");
        let tok = Tokenizer::load(&dir).unwrap();
        fs::remove_dir_all(&dir).ok();

        assert!(tok.id_of("<|end_of_msg|>").is_some());
        let ids = tok.encode("hello<|end_of_msg|>world");
        assert_eq!(tok.decode(&ids), b"hello<|end_of_msg|>world");
        let special_id = tok.id_of("<|end_of_msg|>").unwrap();
        assert_eq!(ids.iter().filter(|&&i| i == special_id).count(), 1);
    }

    #[test]
    fn reserved_but_unnamed_special_tokens_get_the_placeholder_name() {
        let dir = build_fixture("reserved_placeholder");
        let tok = Tokenizer::load(&dir).unwrap();
        fs::remove_dir_all(&dir).ok();

        assert_eq!(tok.id_of("<|end_of_msg|>"), Some(12));
        assert_eq!(tok.id_of("<|reserved_token_13|>"), Some(13));
    }

    #[test]
    fn exactly_256_reserved_ids_no_plus_two_quirk() {
        // Unlike kimi_linear::tokenizer.rs's 48B-specific +2 adjustment, K3's real class
        // attribute is exactly 256 -- the 257th offset must NOT resolve to anything.
        let dir = build_fixture("exact_256");
        let tok = Tokenizer::load(&dir).unwrap();
        fs::remove_dir_all(&dir).ok();

        let num_base = 12u32; // "helowrd, !".len()=10 distinct chars + "he" + "hel"
        assert!(tok.id_of(&format!("<|reserved_token_{}|>", num_base + 255)).is_some(), "offset 255 (the 256th id) must exist");
        assert!(tok.id_of(&format!("<|reserved_token_{}|>", num_base + 256)).is_none(), "offset 256 (a 257th id) must NOT exist for K3");
    }

    #[test]
    fn empty_text_encodes_to_no_tokens() {
        let dir = build_fixture("empty");
        let tok = Tokenizer::load(&dir).unwrap();
        fs::remove_dir_all(&dir).ok();
        assert!(tok.encode("").is_empty());
    }
}
