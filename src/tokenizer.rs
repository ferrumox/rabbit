//! Port of `tok.h` + `tok_unicode.h` — byte-level BPE tokenizer (cl100k/tiktoken style),
//! faithful to `tokenizer.json`'s `model.type = BPE, ignore_merges=true, byte_fallback=false`
//! and the cl100k pre-tokenizer regex (`Split` + `ByteLevel(add_prefix_space=false)`).
//!
//! The pre-tokenizer split (`pretok_pieces`) is pulled out as a pure function of bytes —
//! the original C inlines it into the BPE call, but separating "where do pieces start/end"
//! from "how a piece becomes ids" needs no vocab/merges state and is much easier to test in
//! isolation. Behavior is unchanged: `pretok_chunk` just feeds each piece to `bpe_piece`.

use crate::unicode_tables::{is_l, is_n, is_s};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::io;
use std::path::Path;

#[derive(Debug)]
pub enum TokenizerError {
    Io(io::Error),
    Json(serde_json::Error),
    MissingField(&'static str),
    BadVocabEntry(String),
    BadMergeEntry(usize),
    BadAddedToken,
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerError::Io(e) => write!(f, "tokenizer.json: {e}"),
            TokenizerError::Json(e) => write!(f, "tokenizer.json: {e}"),
            TokenizerError::MissingField(name) => write!(f, "tokenizer.json: missing {name}"),
            TokenizerError::BadVocabEntry(k) => write!(f, "tokenizer.json: bad vocab entry {k:?}"),
            TokenizerError::BadMergeEntry(i) => write!(f, "tokenizer.json: bad merges[{i}]"),
            TokenizerError::BadAddedToken => write!(f, "tokenizer.json: bad added_tokens entry"),
        }
    }
}

impl std::error::Error for TokenizerError {}

impl From<io::Error> for TokenizerError {
    fn from(e: io::Error) -> Self {
        TokenizerError::Io(e)
    }
}

impl From<serde_json::Error> for TokenizerError {
    fn from(e: serde_json::Error) -> Self {
        TokenizerError::Json(e)
    }
}

struct Special {
    text: String,
    id: i32,
}

pub struct Tokenizer {
    /// byte-level string -> id.
    vocab: HashMap<String, i32>,
    /// (left, right) byte-level strings -> merge rank (lower = merges first).
    merges: HashMap<(String, String), i32>,
    /// id -> stored string (byte-level for ordinary tokens, literal for added tokens).
    id2str: Vec<Option<String>>,
    /// id -> true if `id2str[id]` is a literal added-token string, not byte-level.
    id_added: Vec<bool>,
    /// added tokens (special or not), sorted by length descending for longest-match.
    specials: Vec<Special>,
    /// byte -> the (usually multi-byte) UTF-8 encoding of its byte-level codepoint.
    byte2str: Vec<Vec<u8>>,
    /// codepoint -> original byte, for codepoints < 1024 (covers the whole byte-level map).
    cp2byte: [Option<u8>; 1024],
}

/// GPT-2/ByteLevel byte<->unicode map: printable ASCII/Latin-1 bytes map to themselves;
/// the rest (control chars, space, DEL, etc.) map to codepoints starting at 256, so every
/// byte becomes a distinct, always-printable codepoint and byte-level strings round-trip.
fn build_bytemap() -> (Vec<Vec<u8>>, [Option<u8>; 1024]) {
    let mut isdir = [false; 256];
    isdir[33..=126].fill(true);
    isdir[161..=172].fill(true);
    isdir[174..=255].fill(true);

    let mut byte2str: Vec<Vec<u8>> = Vec::with_capacity(256);
    let mut cp2byte = [None; 1024];
    let mut n = 0u32;
    for (b, &dir) in isdir.iter().enumerate() {
        let cp = if dir {
            b as u32
        } else {
            let v = 256 + n;
            n += 1;
            v
        };
        let mut s = Vec::with_capacity(4);
        u8_put(cp, &mut s);
        if (cp as usize) < 1024 {
            cp2byte[cp as usize] = Some(b as u8);
        }
        byte2str.push(s);
    }
    (byte2str, cp2byte)
}

/// Decodes one UTF-8 scalar at `s[i]`, returning `(codepoint, byte_length)`. An invalid or
/// truncated leading byte falls back to treating it as a single raw byte — matches `u8_next`
/// in `tok.h`, which never assumes well-formed input since byte-level BPE must handle it.
fn u8_next(s: &[u8], i: usize) -> (u32, usize) {
    let len = s.len();
    let c = s[i];
    if c < 0x80 {
        return (c as u32, 1);
    }
    if (c >> 5) == 0x6 && i + 1 < len {
        return ((((c & 0x1F) as u32) << 6) | (s[i + 1] & 0x3F) as u32, 2);
    }
    if (c >> 4) == 0xE && i + 2 < len {
        return (
            (((c & 0x0F) as u32) << 12) | (((s[i + 1] & 0x3F) as u32) << 6) | (s[i + 2] & 0x3F) as u32,
            3,
        );
    }
    if (c >> 3) == 0x1E && i + 3 < len {
        return (
            (((c & 0x07) as u32) << 18)
                | (((s[i + 1] & 0x3F) as u32) << 12)
                | (((s[i + 2] & 0x3F) as u32) << 6)
                | (s[i + 3] & 0x3F) as u32,
            4,
        );
    }
    (c as u32, 1)
}

fn u8_put(cp: u32, out: &mut Vec<u8>) -> usize {
    if cp < 0x80 {
        out.push(cp as u8);
        1
    } else if cp < 0x800 {
        out.push(0xC0 | (cp >> 6) as u8);
        out.push(0x80 | (cp & 0x3F) as u8);
        2
    } else if cp < 0x10000 {
        out.push(0xE0 | (cp >> 12) as u8);
        out.push(0x80 | ((cp >> 6) & 0x3F) as u8);
        out.push(0x80 | (cp & 0x3F) as u8);
        3
    } else {
        out.push(0xF0 | (cp >> 18) as u8);
        out.push(0x80 | ((cp >> 12) & 0x3F) as u8);
        out.push(0x80 | ((cp >> 6) & 0x3F) as u8);
        out.push(0x80 | (cp & 0x3F) as u8);
        4
    }
}

fn byte_level_encode(byte2str: &[Vec<u8>], bytes: &[u8]) -> String {
    let mut s = Vec::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.extend_from_slice(&byte2str[b as usize]);
    }
    String::from_utf8(s).expect("byte2str only ever emits well-formed UTF-8")
}

fn parse_merge_entry(v: &Value) -> Option<(String, String)> {
    if let Some(arr) = v.as_array()
        && arr.len() == 2
    {
        return Some((arr[0].as_str()?.to_string(), arr[1].as_str()?.to_string()));
    }
    if let Some(s) = v.as_str() {
        let mut parts = s.splitn(2, ' ');
        return Some((parts.next()?.to_string(), parts.next()?.to_string()));
    }
    None
}

const ISNL: fn(u32) -> bool = |c| c == b'\r' as u32 || c == b'\n' as u32;
const LOW: fn(u32) -> u32 = |c| if (b'A' as u32..=b'Z' as u32).contains(&c) { c + 32 } else { c };

/// Splits `p[a..b)` into pre-tokenizer pieces (byte ranges), applying the cl100k pattern's
/// alternatives in order — literal port of `pretok_chunk`'s regex-by-hand state machine.
fn pretok_pieces(p: &[u8], a: usize, b: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    if b <= a {
        return out;
    }
    let mut cp: Vec<u32> = Vec::new();
    let mut off: Vec<usize> = Vec::new();
    let mut i = a;
    while i < b {
        let (c, k) = u8_next(p, i);
        off.push(i);
        cp.push(c);
        i += k;
    }
    off.push(b);
    let n = cp.len();

    let mut i = 0usize;
    while i < n {
        let start = i;
        let c = cp[i];

        // 1) (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if c == b'\'' as u32 && i + 1 < n {
            let d = LOW(cp[i + 1]);
            if i + 2 < n {
                let d2 = LOW(cp[i + 2]);
                if ((d == b'r' as u32 || d == b'v' as u32) && d2 == b'e' as u32)
                    || (d == b'l' as u32 && d2 == b'l' as u32)
                {
                    i += 3;
                    out.push((off[start], off[i]));
                    continue;
                }
            }
            if d == b's' as u32 || d == b't' as u32 || d == b'm' as u32 || d == b'd' as u32 {
                i += 2;
                out.push((off[start], off[i]));
                continue;
            }
        }

        // 2) [^\r\n\p{L}\p{N}]? \p{L}+
        {
            let mut j: i64 = i as i64;
            if !is_l(c) && !ISNL(c) && !is_n(c) {
                if (j as usize) + 1 < n && is_l(cp[j as usize + 1]) {
                    j += 1;
                } else {
                    j = -1;
                }
            }
            if j >= 0 {
                let mut jj = j as usize;
                if is_l(cp[jj]) {
                    while jj < n && is_l(cp[jj]) {
                        jj += 1;
                    }
                    i = jj;
                    out.push((off[start], off[i]));
                    continue;
                }
            }
        }

        // 3) \p{N}{1,3}
        if is_n(c) {
            let mut j = i;
            let mut k = 0;
            while j < n && is_n(cp[j]) && k < 3 {
                j += 1;
                k += 1;
            }
            i = j;
            out.push((off[start], off[i]));
            continue;
        }

        // 4) ' ?[^\s\p{L}\p{N}]+[\r\n]*'
        {
            let mut j = i;
            if c == b' ' as u32 && j + 1 < n && !is_s(cp[j + 1]) && !is_l(cp[j + 1]) && !is_n(cp[j + 1]) {
                j += 1;
            }
            if j < n && !is_s(cp[j]) && !is_l(cp[j]) && !is_n(cp[j]) {
                while j < n && !is_s(cp[j]) && !is_l(cp[j]) && !is_n(cp[j]) {
                    j += 1;
                }
                while j < n && ISNL(cp[j]) {
                    j += 1;
                }
                i = j;
                out.push((off[start], off[i]));
                continue;
            }
        }

        // 5/6) \s*[\r\n]+  |  \s+(?!\S)
        {
            let mut r = i;
            while r < n && is_s(cp[r]) {
                r += 1;
            }
            if r > i {
                let mut last: Option<usize> = None;
                for (j, &c) in cp.iter().enumerate().take(r).skip(i) {
                    if ISNL(c) {
                        last = Some(j);
                    }
                }
                if let Some(last) = last {
                    i = last + 1;
                    out.push((off[start], off[i]));
                    continue;
                }
                let mut end = if r < n { r - 1 } else { r };
                if end <= i {
                    end = i + 1;
                }
                i = end;
                out.push((off[start], off[i]));
                continue;
            }
        }

        // safety net: none of the alternatives matched (shouldn't happen).
        i += 1;
        out.push((off[start], off[i]));
    }
    out
}

impl Tokenizer {
    pub fn load(path: &Path) -> Result<Tokenizer, TokenizerError> {
        let text = fs::read_to_string(path)?;
        let root: Value = serde_json::from_str(&text)?;

        let model = root.get("model").ok_or(TokenizerError::MissingField("model"))?;
        let vocab_obj = model
            .get("vocab")
            .and_then(Value::as_object)
            .ok_or(TokenizerError::MissingField("model.vocab"))?;
        let merges_arr = model
            .get("merges")
            .and_then(Value::as_array)
            .ok_or(TokenizerError::MissingField("model.merges"))?;
        let added = root.get("added_tokens").and_then(Value::as_array);

        let mut maxid: i64 = -1;
        for v in vocab_obj.values() {
            if let Some(id) = v.as_i64() {
                maxid = maxid.max(id);
            }
        }
        if let Some(a) = added {
            for t in a {
                if let Some(id) = t.get("id").and_then(Value::as_i64) {
                    maxid = maxid.max(id);
                }
            }
        }
        let n_ids = (maxid + 1).max(0) as usize;
        let mut id2str: Vec<Option<String>> = vec![None; n_ids];
        let mut id_added = vec![false; n_ids];

        let mut vocab = HashMap::with_capacity(vocab_obj.len());
        for (k, v) in vocab_obj {
            let id = v
                .as_i64()
                .ok_or_else(|| TokenizerError::BadVocabEntry(k.clone()))? as i32;
            vocab.insert(k.clone(), id);
            if (id as usize) < n_ids {
                id2str[id as usize] = Some(k.clone());
            }
        }

        let mut merges = HashMap::with_capacity(merges_arr.len());
        for (rank, entry) in merges_arr.iter().enumerate() {
            let (l, r) = parse_merge_entry(entry).ok_or(TokenizerError::BadMergeEntry(rank))?;
            merges.insert((l, r), rank as i32);
        }

        let mut specials = Vec::new();
        if let Some(a) = added {
            for t in a {
                let content = t
                    .get("content")
                    .and_then(Value::as_str)
                    .ok_or(TokenizerError::BadAddedToken)?
                    .to_string();
                let id = t
                    .get("id")
                    .and_then(Value::as_i64)
                    .ok_or(TokenizerError::BadAddedToken)? as i32;
                if (id as usize) < n_ids {
                    id2str[id as usize] = Some(content.clone());
                    id_added[id as usize] = true;
                }
                specials.push(Special { text: content, id });
            }
            // longest match first, same as colibri's qsort(cmp_sp_len).
            specials.sort_by_key(|sp| std::cmp::Reverse(sp.text.len()));
        }

        let (byte2str, cp2byte) = build_bytemap();

        Ok(Tokenizer { vocab, merges, id2str, id_added, specials, byte2str, cp2byte })
    }

    /// BPE-encodes one pre-tokenized piece (raw bytes) and appends resulting ids to `out`.
    fn bpe_piece(&self, bytes: &[u8], out: &mut Vec<i32>) {
        let s = byte_level_encode(&self.byte2str, bytes);

        // ignore_merges: if the whole piece is itself a vocab entry, emit it directly.
        if let Some(&id) = self.vocab.get(s.as_str()) {
            out.push(id);
            return;
        }

        let mut symbols: Vec<(usize, usize)> =
            s.char_indices().map(|(i, c)| (i, c.len_utf8())).collect();

        loop {
            let mut best_rank = i32::MAX;
            let mut best_pos: Option<usize> = None;
            for i in 0..symbols.len().saturating_sub(1) {
                let (a_off, a_len) = symbols[i];
                let (b_off, b_len) = symbols[i + 1];
                let left = &s[a_off..a_off + a_len];
                let right = &s[b_off..b_off + b_len];
                if let Some(&rank) = self.merges.get(&(left.to_string(), right.to_string()))
                    && rank < best_rank
                {
                    best_rank = rank;
                    best_pos = Some(i);
                }
            }
            let Some(bp) = best_pos else { break };
            let (a_off, _) = symbols[bp];
            let (b_off, b_len) = symbols[bp + 1];
            symbols[bp] = (a_off, (b_off + b_len) - a_off);
            symbols.remove(bp + 1);
        }

        for (off, len) in symbols {
            if let Some(&id) = self.vocab.get(&s[off..off + len]) {
                out.push(id);
            }
        }
    }

    fn pretok_chunk(&self, p: &[u8], a: usize, b: usize, out: &mut Vec<i32>) {
        for (s, e) in pretok_pieces(p, a, b) {
            self.bpe_piece(&p[s..e], out);
        }
    }

    /// Encodes text into token ids: splits on added-token occurrences (longest match wins),
    /// then pre-tokenizes and BPE-encodes the text in between.
    pub fn encode(&self, text: &str) -> Vec<i32> {
        let p = text.as_bytes();
        let len = p.len();
        let mut out = Vec::new();
        let mut i = 0usize;
        while i < len {
            let mut hit: Option<(usize, usize, i32)> = None;
            'search: for j in i..len {
                for sp in &self.specials {
                    let sl = sp.text.len();
                    if sl > 0 && j + sl <= len && &p[j..j + sl] == sp.text.as_bytes() {
                        hit = Some((j, sl, sp.id));
                        break 'search;
                    }
                }
            }
            let chunk_end = hit.map(|(pos, _, _)| pos).unwrap_or(len);
            if chunk_end > i {
                self.pretok_chunk(p, i, chunk_end, &mut out);
            }
            let Some((pos, hitlen, hitid)) = hit else { break };
            out.push(hitid);
            i = pos + hitlen;
        }
        out
    }

    /// Decodes ids back into raw bytes: literal output for added tokens, byte-level inverse
    /// (via `cp2byte`) for ordinary tokens.
    pub fn decode(&self, ids: &[i32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            if id < 0 || id as usize >= self.id2str.len() {
                continue;
            }
            let Some(s) = &self.id2str[id as usize] else { continue };
            if self.id_added[id as usize] {
                out.extend_from_slice(s.as_bytes());
                continue;
            }
            let bytes = s.as_bytes();
            let mut j = 0;
            while j < bytes.len() {
                let (c, k) = u8_next(bytes, j);
                j += k;
                if (c as usize) < 1024
                    && let Some(b) = self.cp2byte[c as usize]
                {
                    out.push(b);
                }
            }
        }
        out
    }

    /// Id of an added token given its literal content (e.g. `"<|endoftext|>"`).
    pub fn id_of(&self, content: &str) -> Option<i32> {
        self.specials.iter().find(|sp| sp.text == content).map(|sp| sp.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytemap_round_trips_every_byte() {
        let (byte2str, cp2byte) = build_bytemap();
        for (b, s) in byte2str.iter().enumerate() {
            let (cp, k) = u8_next(s, 0);
            assert_eq!(k, s.len(), "byte2str[{b}] should decode as one codepoint");
            assert_eq!(cp2byte[cp as usize], Some(b as u8), "byte {b} did not round-trip");
        }
        // printable ASCII maps to itself; space (not in isdir range) maps elsewhere.
        assert_eq!(byte2str[b'A' as usize], vec![b'A']);
        assert_ne!(byte2str[b' ' as usize], vec![b' ']);
    }

    #[test]
    fn u8_next_handles_multibyte_and_invalid_leading_bytes() {
        for &cp in &[0x41u32, 0x1F9u32, 0x1F600u32, 0x10348u32] {
            let mut buf = Vec::new();
            let written = u8_put(cp, &mut buf);
            let (decoded, read) = u8_next(&buf, 0);
            assert_eq!(decoded, cp);
            assert_eq!(read, written);
        }
        // a lone continuation byte is not a valid leading byte -> falls back to itself.
        let (cp, k) = u8_next(&[0x80], 0);
        assert_eq!((cp, k), (0x80, 1));
    }

    #[test]
    fn pretok_splits_contractions_letters_numbers_and_whitespace() {
        let text = b"I'll say don't 123456 hi, world!\n\n  next";
        let pieces: Vec<&str> = pretok_pieces(text, 0, text.len())
            .into_iter()
            .map(|(a, b)| std::str::from_utf8(&text[a..b]).unwrap())
            .collect();
        // note: a lone space before a digit run does NOT attach to it (alt 4 requires the
        // char itself be non-space), and a punctuation run swallows a trailing \r\n run
        // (alt 4 ends in `[\r\n]*`) — both per the literal cl100k pattern, not a guess.
        assert_eq!(
            pieces,
            vec![
                "I", "'ll", " say", " don", "'t", " ", "123", "456", " hi", ",", " world",
                "!\n\n", " ", " next",
            ]
        );
    }

    #[test]
    fn pretok_trailing_whitespace_without_following_char_is_kept_whole() {
        let text = b"end   ";
        let pieces: Vec<&str> = pretok_pieces(text, 0, text.len())
            .into_iter()
            .map(|(a, b)| std::str::from_utf8(&text[a..b]).unwrap())
            .collect();
        assert_eq!(pieces, vec!["end", "   "]);
    }

    fn minimal_tokenizer_json(vocab: &[(String, i32)], merges: &[(String, String)], added: &[(&str, i32)]) -> String {
        let vocab_json: Vec<String> = vocab.iter().map(|(k, v)| format!("{:?}:{v}", k)).collect();
        let merges_json: Vec<String> = merges.iter().map(|(l, r)| format!("[{l:?},{r:?}]")).collect();
        let added_json: Vec<String> = added
            .iter()
            .map(|(s, id)| format!(r#"{{"id":{id},"content":{s:?},"special":true}}"#))
            .collect();
        format!(
            r#"{{"model":{{"type":"BPE","vocab":{{{}}},"merges":[{}]}},"added_tokens":[{}]}}"#,
            vocab_json.join(","),
            merges_json.join(","),
            added_json.join(","),
        )
    }

    fn write_fixture(name: &str, json: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("rabbit_test_tok_{name}.json"));
        fs::write(&path, json).unwrap();
        path
    }

    #[test]
    fn bpe_merges_by_rank_order_into_a_single_token() {
        let (byte2str, _) = build_bytemap();
        let bl = |s: &str| byte_level_encode(&byte2str, s.as_bytes());

        let vocab = [
            (bl("h"), 0),
            (bl("e"), 1),
            (bl("l"), 2),
            (bl("o"), 3),
            (bl("he"), 4),
            (bl("hel"), 5),
            (bl("hell"), 6),
            (bl("hello"), 7),
        ];
        let merges = [
            (bl("h"), bl("e")),
            (bl("he"), bl("l")),
            (bl("hel"), bl("l")),
            (bl("hell"), bl("o")),
        ];
        let json = minimal_tokenizer_json(&vocab, &merges, &[]);
        let path = write_fixture("merge_chain", &json);
        let tok = Tokenizer::load(&path).unwrap();
        fs::remove_file(&path).ok();

        assert_eq!(tok.encode("hello"), vec![7]);
        assert_eq!(tok.decode(&[7]), b"hello");
    }

    #[test]
    fn ignore_merges_shortcut_bypasses_rank_ordering() {
        let (byte2str, _) = build_bytemap();
        let bl = |s: &str| byte_level_encode(&byte2str, s.as_bytes());

        // "ab" is a whole vocab entry, but the merges list does NOT contain ("a","b") —
        // if the whole-piece shortcut didn't fire, the merge loop would find nothing and
        // emit the two single-char ids instead.
        let vocab = [(bl("a"), 0), (bl("b"), 1), (bl("ab"), 2)];
        let json = minimal_tokenizer_json(&vocab, &[], &[]);
        let path = write_fixture("ignore_merges", &json);
        let tok = Tokenizer::load(&path).unwrap();
        fs::remove_file(&path).ok();

        assert_eq!(tok.encode("ab"), vec![2]);
    }

    #[test]
    fn added_tokens_prefer_longest_match_and_round_trip() {
        let (byte2str, _) = build_bytemap();
        let bl = |s: &str| byte_level_encode(&byte2str, s.as_bytes());

        let vocab = [(bl("h"), 0), (bl("i"), 1)];
        let added = [("<|a|>", 100), ("<|a|>x", 101)];
        let json = minimal_tokenizer_json(&vocab, &[], &added);
        let path = write_fixture("added_tokens", &json);
        let tok = Tokenizer::load(&path).unwrap();
        fs::remove_file(&path).ok();

        assert_eq!(tok.id_of("<|a|>"), Some(100));
        assert_eq!(tok.id_of("<|a|>x"), Some(101));

        // at this position both "<|a|>" and "<|a|>x" match; the longer one must win.
        let ids = tok.encode("hi<|a|>x");
        assert_eq!(ids, vec![0, 1, 101]);
        assert_eq!(tok.decode(&ids), b"hi<|a|>x");
    }
}
