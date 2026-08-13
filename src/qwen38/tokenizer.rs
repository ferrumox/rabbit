//! Qwen 3.8's tokenizer: byte-level BPE from `tokenizer.json`, the SAME family as GLM-5.2's, so
//! this module adds no BPE code of its own — it reuses `crate::tokenizer::Tokenizer` wholesale (the
//! GPT-2 byte↔unicode map, the merge-rank loop, added-token longest-match splitting) and exists
//! only to pin down the two places where Qwen's file genuinely differs from GLM's, both of which
//! `crate::tokenizer` now reads from the file itself:
//!
//! | | GLM-5.2 | Qwen 3.8 |
//! |---|---|---|
//! | `model.ignore_merges` | `true` | **`false`** |
//! | `pre_tokenizer` Split regex | cl100k's | **[`QWEN38_PRETOK_PATTERN`]** |
//! | `normalizer` | `null` | `NFC` (see below) |
//!
//! The regex difference is not cosmetic. Against cl100k's, Qwen's pattern:
//! * groups combining marks with their base letter (`[\p{L}\p{M}]+` vs `\p{L}+`), so decomposed
//!   text like `cafe`+U+0301 is one piece here and two under cl100k's;
//! * splits **every digit into its own piece** (`\p{N}` vs `\p{N}{1,3}`), so `2024` is 4 pieces,
//!   not 2 — this alone would shift the id sequence of any prompt containing a number;
//! * also excludes `\p{M}` from the punctuation run.
//!
//! **The `NFC` normalizer is NOT implemented.** Text that is already NFC — everything typed on an
//! ordinary keyboard, and what HTTP/JSON clients send in practice — is unaffected, since NFC is
//! then the identity. Decomposed input would tokenize as its decomposed form: a different, still
//! round-trippable id sequence, i.e. slightly worse tokenization, not corruption (and Qwen's own
//! `[\p{L}\p{M}]+` alternative keeps such sequences in one piece rather than fragmenting them).
//! Implementing real NFC means shipping the canonical-composition tables, which this port defers
//! until something actually needs it.
//!
//! Everything here was read off the real `tokenizer.json` from
//! `amd/Qwen3.8-2.4T-A95B-Quark-MXFP4` (248,044 vocab entries, 247,587 merges, 33 added tokens
//! spanning ids 248,044-248,076), not from documentation.

use crate::tokenizer::{Tokenizer, TokenizerError};
use std::fmt;
use std::path::Path;

/// Qwen 3.8's `pre_tokenizer.Split.pattern.Regex`, verbatim from the real `tokenizer.json`.
pub const QWEN38_PRETOK_PATTERN: &str =
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

#[derive(Debug)]
pub enum LoadError {
    Tokenizer(TokenizerError),
    /// The file's declared `Split` pattern isn't [`QWEN38_PRETOK_PATTERN`]. Loading anyway would
    /// tokenize by whatever it does say, which may well be fine — but silently, and a
    /// pre-tokenizer change is exactly the kind of difference that shifts every id in a prompt
    /// without ever producing an error or garbled text, so it's surfaced instead.
    UnexpectedPattern { found: Option<String> },
    /// `model.ignore_merges` is on. On Qwen's real file it is off, and having it on lets any piece
    /// that happens to be a whole vocab entry bypass merge-rank ordering.
    IgnoreMergesOn,
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Tokenizer(e) => write!(f, "{e}"),
            LoadError::UnexpectedPattern { found: Some(p) } => write!(
                f,
                "tokenizer.json declares pre_tokenizer Split regex {p:?}, but this port was written against Qwen 3.8's {QWEN38_PRETOK_PATTERN:?}"
            ),
            LoadError::UnexpectedPattern { found: None } => {
                write!(f, "tokenizer.json declares no pre_tokenizer Split regex; Qwen 3.8's checkpoints all do")
            }
            LoadError::IgnoreMergesOn => {
                write!(f, "tokenizer.json has model.ignore_merges = true, but Qwen 3.8's real checkpoints set it false")
            }
        }
    }
}

impl std::error::Error for LoadError {}

impl From<TokenizerError> for LoadError {
    fn from(e: TokenizerError) -> Self {
        LoadError::Tokenizer(e)
    }
}

/// Loads `<snap_dir>/tokenizer.json` and verifies it is shaped the way this port assumes before
/// handing back the shared [`Tokenizer`].
pub fn load(snap_dir: &Path) -> Result<Tokenizer, LoadError> {
    let tok = Tokenizer::load(&snap_dir.join("tokenizer.json"))?;
    match tok.declared_pretok_pattern() {
        Some(p) if p == QWEN38_PRETOK_PATTERN => {}
        found => return Err(LoadError::UnexpectedPattern { found: found.map(str::to_string) }),
    }
    if tok.ignore_merges() {
        return Err(LoadError::IgnoreMergesOn);
    }
    Ok(tok)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    struct TempDir(std::path::PathBuf);
    impl TempDir {
        fn new(name: &str) -> TempDir {
            let dir = std::env::temp_dir().join(format!("{name}_{}", std::process::id()));
            fs::create_dir_all(&dir).unwrap();
            TempDir(dir)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).ok();
        }
    }

    /// A `tokenizer.json` with Qwen's real `pre_tokenizer`/`normalizer`/`ignore_merges` shape and a
    /// hand-built byte-level vocab big enough to encode the test strings below one byte at a time.
    fn write_qwen_tokenizer(dir: &Path, pattern: &str, ignore_merges: bool) {
        // byte-level (GPT-2) spellings for the bytes these tests use: printable ASCII maps to
        // itself, and space -> U+0120 "Ġ".
        let mut vocab: Vec<(String, i32)> = Vec::new();
        let mut next = 0i32;
        for b in 33u8..=126 {
            vocab.push(((b as char).to_string(), next));
            next += 1;
        }
        vocab.push(("Ġ".to_string(), next));
        next += 1;
        // The two multi-byte pieces the mark test needs, spelled byte-level: U+0301 is 0xCC 0x81,
        // which byte-level maps to U+0138 (Ĭ, for 0xCC) and U+0081's own mapping.
        for cp in [0x100u32, 0x101, 0x102, 0x103, 0x104, 0x105, 0x106, 0x107, 0x108, 0x109, 0x10A, 0x10B, 0x10C, 0x10D,
            0x10E, 0x10F, 0x110, 0x111, 0x112, 0x113, 0x114, 0x115, 0x116, 0x117, 0x118, 0x119, 0x11A, 0x11B, 0x11C,
            0x11D, 0x11E, 0x11F, 0x121, 0x122, 0x123, 0x124, 0x125, 0x126, 0x127, 0x128, 0x129, 0x12A, 0x12B, 0x12C,
            0x12D, 0x12E, 0x12F, 0x130, 0x131, 0x132, 0x133, 0x134, 0x135, 0x136, 0x137, 0x138, 0x139, 0x13A, 0x13B,
            0x13C, 0x13D, 0x13E, 0x13F, 0x140, 0x141, 0x142, 0x143]
        {
            vocab.push((char::from_u32(cp).unwrap().to_string(), next));
            next += 1;
        }
        let vocab_json: Vec<String> = vocab.iter().map(|(k, v)| format!("{k:?}:{v}")).collect();
        let json = format!(
            r#"{{"normalizer":{{"type":"NFC"}},
                 "pre_tokenizer":{{"type":"Sequence","pretokenizers":[
                     {{"type":"Split","pattern":{{"Regex":{pattern:?}}},"behavior":"Isolated","invert":false}},
                     {{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}}]}},
                 "model":{{"type":"BPE","ignore_merges":{ignore_merges},"byte_fallback":false,
                           "vocab":{{{}}},"merges":[]}},
                 "added_tokens":[{{"id":9000,"content":"<|im_start|>","special":true}},
                                 {{"id":9001,"content":"<|im_end|>","special":true}}]}}"#,
            vocab_json.join(",")
        );
        fs::write(dir.join("tokenizer.json"), json).unwrap();
    }

    #[test]
    fn loads_a_qwen_shaped_tokenizer_json() {
        let dir = TempDir::new("rabbit_test_qwen38_tok_ok");
        write_qwen_tokenizer(&dir.0, QWEN38_PRETOK_PATTERN, false);
        let tok = load(&dir.0).unwrap();
        assert!(!tok.ignore_merges());
        assert_eq!(tok.declared_pretok_pattern(), Some(QWEN38_PRETOK_PATTERN));
        // added tokens still split out of surrounding text, byte-level BPE handles the rest
        let ids = tok.encode("<|im_start|>hi<|im_end|>");
        assert_eq!(ids.first(), Some(&9000));
        assert_eq!(ids.last(), Some(&9001));
        assert_eq!(tok.decode(&ids), b"<|im_start|>hi<|im_end|>");
    }

    /// Qwen's `\p{N}` (one digit per piece) vs cl100k's `\p{N}{1,3}`: with no merges in the
    /// fixture, each piece emits its own id, so the ID COUNT reveals the split. "2024" must be four
    /// tokens; under GLM's pattern it would be two ("202" + "4") — and with a real vocab, entirely
    /// different ids.
    #[test]
    fn splits_every_digit_into_its_own_piece() {
        let dir = TempDir::new("rabbit_test_qwen38_tok_digits");
        write_qwen_tokenizer(&dir.0, QWEN38_PRETOK_PATTERN, false);
        let tok = load(&dir.0).unwrap();
        assert_eq!(tok.encode("2024").len(), 4);
        assert_eq!(tok.decode(&tok.encode("2024")), b"2024");
    }

    #[test]
    fn rejects_a_checkpoint_whose_pre_tokenizer_changed() {
        let dir = TempDir::new("rabbit_test_qwen38_tok_pattern");
        // GLM's cl100k pattern in a Qwen-shaped file: loads fine as a generic tokenizer, must be
        // refused here.
        let cl100k = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        write_qwen_tokenizer(&dir.0, cl100k, false);
        match load(&dir.0).map_err(|e| e.to_string()) {
            Err(msg) => {
                assert!(msg.contains("pre_tokenizer Split regex"), "wrong error: {msg}");
                assert!(msg.contains(r"\p{N}{1,3}"), "the error must quote the pattern it found: {msg}");
            }
            Ok(_) => panic!("a cl100k-patterned file must not load as a Qwen 3.8 tokenizer"),
        }
    }

    #[test]
    fn rejects_ignore_merges_on() {
        let dir = TempDir::new("rabbit_test_qwen38_tok_ignore_merges");
        write_qwen_tokenizer(&dir.0, QWEN38_PRETOK_PATTERN, true);
        assert!(matches!(load(&dir.0), Err(LoadError::IgnoreMergesOn)));
    }
}
