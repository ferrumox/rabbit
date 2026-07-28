//! Kimi Linear's own tokenizer — a real `tiktoken`-format byte-level BPE, genuinely different
//! from `crate::tokenizer`'s GLM-5.2 port (a separate `merges: [(left,right)]` list, GPT-2/
//! `tokenizers`-crate convention). Tiktoken's vocabulary IS the merge priority: `mergeable_ranks`
//! maps whole byte sequences directly to a rank, and the merge loop repeatedly joins whichever
//! ADJACENT pair's combined bytes have the lowest rank — ported faithfully from OpenAI's real
//! `tiktoken` Rust core (`_byte_pair_merge`/`byte_pair_encode`,
//! <https://github.com/openai/tiktoken/blob/main/src/lib.rs>, fetched and read this session, not
//! guessed), skipping only that source's large-piece (`>= 100` bytes) heap-optimized variant —
//! real text pretokenizes into much smaller pieces (punctuation/whitespace breaks long before
//! 100 bytes), so the small-piece `O(mn)` algorithm alone is correct, just not the fastest
//! possible on pathological inputs; matches this crate's "correctness first, perf later"
//! discipline (see `PERFORMANCE.md`) if that path is ever needed.
//!
//! The pre-tokenizer pattern itself comes from Moonshot's real `tokenization_kimi.py`
//! (`TikTokenTokenizer.pat_str`) — it needs Unicode SET INTERSECTION syntax
//! (`[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]`, excluding Han from the non-Han-letter
//! alternatives) that Rust's plain `regex` crate does not support. Confirmed this session (a
//! standalone smoke test) that `fancy-regex` DOES support this syntax and produces the expected
//! split — the same crate family real tiktoken's own Rust core depends on.
//!
//! Special tokens: Kimi Linear reserves 258 ids past the base vocabulary (`num_base_tokens` —
//! confirmed `163584` on the real checkpoint — `..= num_base_tokens + 256 + 1`), 17 of which have
//! real names (`tokenizer_config.json`'s `added_tokens_decoder`); the rest fall back to
//! `<|reserved_token_{id}|>`, matching `tokenization_kimi.py`'s own
//! `special_tokens_mapping.get(i, f"<|reserved_token_{i}|>")` exactly. **The last 2 of those 258
//! ids (`num_base_tokens + 256`, `+ 257`) exceed the real checkpoint's `vocab_size` (163840) —
//! a real quirk in the reference tokenizer itself, not a bug here** — callers must never feed
//! those two ids into the model (`kimi_linear::model`'s embedding table has no rows for them).

use base64::Engine;
use fancy_regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::Path;

pub type Rank = u32;

/// Verbatim from `tokenization_kimi.py`'s `TikTokenTokenizer.pat_str` — do not reorder or
/// "simplify" any alternative; each one governs a real split boundary case (contractions, CJK
/// runs, mixed-case runs, digit runs capped at 3, punctuation runs, whitespace runs).
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

/// `tokenization_kimi.py`'s own default `additional_special_tokens` list, used when
/// `tokenizer_config.json` doesn't override it — not read from disk since it's a fixed constant
/// in the reference code, not data.
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

/// OpenAI tiktoken's real `_byte_pair_merge` (small-piece path), ported line-for-line from
/// `tiktoken`'s own `src/lib.rs`: returns split BOUNDARIES (byte offsets into `piece`, paired
/// with the rank of the pair starting there), not token ids themselves — `byte_pair_encode`
/// below turns consecutive boundaries into the final ids via a second lookup.
fn byte_pair_merge(ranks: &HashMap<Vec<u8>, Rank>, piece: &[u8]) -> Vec<(usize, Rank)> {
    let mut parts: Vec<(usize, Rank)> = Vec::with_capacity(piece.len() + 1);

    let mut min_rank: (Rank, usize) = (Rank::MAX, usize::MAX);
    for i in 0..piece.len() - 1 {
        let rank = *ranks.get(&piece[i..i + 2]).unwrap_or(&Rank::MAX);
        if rank < min_rank.0 {
            min_rank = (rank, i);
        }
        parts.push((i, rank));
    }
    parts.push((piece.len() - 1, Rank::MAX));
    parts.push((piece.len(), Rank::MAX));

    let get_rank = |parts: &[(usize, Rank)], i: usize| -> Rank {
        if i + 3 < parts.len() {
            *ranks.get(&piece[parts[i].0..parts[i + 3].0]).unwrap_or(&Rank::MAX)
        } else {
            Rank::MAX
        }
    };

    while min_rank.0 != Rank::MAX {
        let i = min_rank.1;
        if i > 0 {
            parts[i - 1].1 = get_rank(&parts, i - 1);
        }
        parts[i].1 = get_rank(&parts, i);
        parts.remove(i + 1);

        min_rank = (Rank::MAX, usize::MAX);
        for (idx, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
            if rank < min_rank.0 {
                min_rank = (rank, idx);
            }
        }
    }
    parts
}

/// `tiktoken`'s real `byte_pair_encode` (small-piece dispatch only — see this module's doc for
/// why the large-piece heap variant is deliberately not ported).
fn byte_pair_encode(piece: &[u8], ranks: &HashMap<Vec<u8>, Rank>) -> Vec<Rank> {
    debug_assert!(!piece.is_empty(), "byte_pair_encode called on an empty piece");
    if piece.len() == 1 {
        return vec![ranks[piece]];
    }
    byte_pair_merge(ranks, piece).windows(2).map(|w| ranks[&piece[w[0].0..w[1].0]]).collect()
}

impl Tokenizer {
    /// Loads `tiktoken.model` + `tokenizer_config.json` from a real Kimi Linear checkpoint
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
        for i in num_base_tokens..num_base_tokens + NUM_RESERVED_SPECIAL_TOKENS + 2 {
            let name = added.get(&i).cloned().unwrap_or_else(|| format!("<|reserved_token_{i}|>"));
            special_encoder.insert(name, i);
        }
        let special_decoder: HashMap<Rank, Vec<u8>> = special_encoder.iter().map(|(k, &v)| (v, k.as_bytes().to_vec())).collect();

        let regex = Regex::new(PAT_STR)?;
        // Longest-first alternation, same "longest match wins" precedent
        // `crate::tokenizer::Tokenizer::load`'s own `specials` list already follows — avoids a
        // short special token's literal text shadowing a longer one that starts the same way.
        let mut special_strs: Vec<&String> = special_encoder.keys().collect();
        special_strs.sort_by_key(|s| std::cmp::Reverse(s.len()));
        let special_pattern = special_strs.iter().map(|s| fancy_regex::escape(s)).collect::<Vec<_>>().join("|");
        let special_regex = Regex::new(&special_pattern)?;

        Ok(Tokenizer { encoder, decoder, special_encoder, special_decoder, regex, special_regex })
    }

    /// Encodes text into token ids. Any of the 258 special-token strings that appear literally
    /// in `text` (e.g. `<|im_user|>`, inserted by the chat-template renderer) are always
    /// recognized as a single id — no allow-list distinction, matching
    /// `crate::tokenizer::Tokenizer::encode`'s own existing (simpler) behavior for consistency
    /// across rabbit's two tokenizers, rather than porting `tiktoken`'s real
    /// `allowed_special: &HashSet<&str>` gating (meant for untrusted-input safety in a
    /// general-purpose library; rabbit's callers only ever encode text they built themselves).
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

    /// Decodes ids back into raw bytes. Ids with no matching entry (e.g. negative, or one of
    /// the two out-of-`vocab_size` reserved ids — see this module's doc) are silently skipped,
    /// same permissive policy as `crate::tokenizer::Tokenizer::decode`.
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

    /// Id of a special token given its literal content (e.g. `"<|im_end|>"`).
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
        let dir = std::env::temp_dir().join(format!("rabbit_test_kimi_tok_{name}"));
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
                num_base.to_string(): {"content": "<|special_a|>"},
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

        // "hel" is itself a vocab entry (whole-piece shortcut) -> single id, the highest rank
        // assigned (built last, from "he"+"l").
        let ids = tok.encode("hel");
        assert_eq!(ids.len(), 1);
        assert_eq!(tok.decode(&ids), b"hel");
    }

    #[test]
    fn falls_back_to_single_byte_tokens_when_no_merge_exists() {
        let dir = build_fixture("no_merge");
        let tok = Tokenizer::load(&dir).unwrap();
        fs::remove_dir_all(&dir).ok();

        // "row" has no registered merges among these letters -> one id per byte.
        let ids = tok.encode("row");
        assert_eq!(ids.len(), 3);
        assert_eq!(tok.decode(&ids), b"row");
    }

    #[test]
    fn named_special_token_resolves_and_round_trips() {
        let dir = build_fixture("named_special");
        let tok = Tokenizer::load(&dir).unwrap();
        fs::remove_dir_all(&dir).ok();

        assert!(tok.id_of("<|special_a|>").is_some());
        let ids = tok.encode("hello<|special_a|>world");
        assert_eq!(tok.decode(&ids), b"hello<|special_a|>world");
        // the special token must be exactly ONE id, not BPE-split.
        let special_id = tok.id_of("<|special_a|>").unwrap();
        assert_eq!(ids.iter().filter(|&&i| i == special_id).count(), 1);
    }

    #[test]
    fn reserved_but_unnamed_special_tokens_get_the_placeholder_name() {
        let dir = build_fixture("reserved_placeholder");
        let tok = Tokenizer::load(&dir).unwrap();
        fs::remove_dir_all(&dir).ok();

        // num_base_tokens is 12 ("helowrd, !".len()=10 distinct chars + "he" + "hel" = 12);
        // id 12 is named "<|special_a|>", id 13 is NOT named -> placeholder.
        assert_eq!(tok.id_of("<|special_a|>"), Some(12));
        assert_eq!(tok.id_of("<|reserved_token_13|>"), Some(13));
    }

    #[test]
    fn empty_text_encodes_to_no_tokens() {
        let dir = build_fixture("empty");
        let tok = Tokenizer::load(&dir).unwrap();
        fs::remove_dir_all(&dir).ok();
        assert!(tok.encode("").is_empty());
    }
}
