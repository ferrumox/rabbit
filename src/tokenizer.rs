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
    /// `pre_tokenizer` declared a `Split` regex the engine can't compile — better to fail loudly
    /// than to silently fall back to cl100k's pattern and mis-tokenize every prompt.
    UnsupportedPreTokenizer { pattern: String, detail: String },
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
            TokenizerError::UnsupportedPreTokenizer { pattern, detail } => {
                write!(f, "tokenizer.json: can't compile pre_tokenizer Split regex {pattern:?}: {detail}")
            }
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

/// How a checkpoint's `pre_tokenizer` splits text before BPE runs.
///
/// Two real patterns exist across the checkpoints rabbit reads, and they are NOT
/// interchangeable — see [`CL100K_PATTERN`] and [`Tokenizer::load`]'s selection logic.
enum Pretok {
    /// GLM-5.2's cl100k pattern, run by [`pretok_pieces`]'s hand-rolled state machine (a literal
    /// port of the reference C implementation). Kept as its own variant rather than routed through
    /// the regex engine so GLM's tokenization stays byte-for-byte what it has always been.
    Cl100k,
    /// Any other declared `Split` regex, compiled and run with `fancy_regex` (needed for the
    /// `(?!\S)` lookahead these patterns all use). Qwen 3.8's pattern lands here: it differs from
    /// cl100k in three ways that all change real output — `[\p{L}\p{M}]+` instead of `\p{L}+`
    /// (combining marks group WITH their base letter), `\p{N}` instead of `\p{N}{1,3}` (every
    /// digit is its own piece, so "2024" is four pieces, not one), and `\p{M}` also excluded from
    /// the punctuation run.
    Regex(Box<fancy_regex::Regex>),
}

/// GLM-5.2's `tokenizer.json`'s own `pre_tokenizer.Split.pattern.Regex`, verbatim (checked against
/// `tests/fixtures/tokenizer.json`, the real downloaded file). A checkpoint declaring exactly this
/// gets the hand-rolled [`pretok_pieces`] path; anything else gets compiled by `fancy_regex`.
const CL100K_PATTERN: &str =
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

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
    /// Which pre-tokenizer this checkpoint declared — see [`Pretok`].
    pretok: Pretok,
    /// The `Split` regex string exactly as the file declared it (`None` if it declared none) —
    /// kept so a per-architecture loader can VERIFY it is the pattern that architecture's port was
    /// written against, instead of trusting that whatever shipped is what it expects. See
    /// `qwen38::tokenizer::load`.
    declared_pattern: Option<String>,
    /// `model.ignore_merges`: when true, a piece that is itself a vocab entry is emitted as that
    /// single id without consulting the merge table at all. **`true` on GLM-5.2's real
    /// `tokenizer.json`, `false` on Qwen 3.8's** — with it wrongly on, any Qwen piece that happens
    /// to exist in the vocab skips merge-rank ordering and can produce a different (still
    /// decodable, so silently wrong) id sequence than the reference tokenizer.
    ///
    /// Defaults to `true` when the key is absent, which is what this port did unconditionally
    /// before this field existed — not HF's own default (`false`), but every checkpoint rabbit
    /// reads states the key explicitly, so the default only affects synthetic fixtures.
    ignore_merges: bool,
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

/// Digs the declared `Split` regex out of `pre_tokenizer`, whether it sits there directly or (as on
/// every real checkpoint) inside a `Sequence` alongside a `ByteLevel` step. `None` when there's no
/// `pre_tokenizer`, it isn't a `Split`/`Sequence`, or the `Split` carries a non-`Regex` pattern
/// (a plain string delimiter — no real checkpoint here uses one, and guessing a regex from it would
/// be worse than falling back).
fn split_pattern(root: &Value) -> Option<&str> {
    fn as_split_regex(v: &Value) -> Option<&str> {
        if v.get("type").and_then(Value::as_str) != Some("Split") {
            return None;
        }
        v.get("pattern").and_then(|p| p.get("Regex")).and_then(Value::as_str)
    }
    let pt = root.get("pre_tokenizer")?;
    as_split_regex(pt).or_else(|| pt.get("pretokenizers").and_then(Value::as_array)?.iter().find_map(as_split_regex))
}

const ISNL: fn(u32) -> bool = |c| c == b'\r' as u32 || c == b'\n' as u32;
const LOW: fn(u32) -> u32 = |c| if (b'A' as u32..=b'Z' as u32).contains(&c) { c + 32 } else { c };

/// Pre-tokenizer pieces for a declared `Split` regex, the counterpart of [`pretok_pieces`] for
/// [`Pretok::Regex`]: byte ranges into `s`, covering it completely.
///
/// Both the matches AND the gaps between them become pieces, matching HF `tokenizers`' `Split`
/// with `behavior: "Isolated"` (which keeps the delimiters as their own tokens rather than
/// dropping them). In practice these patterns match everything — the final `\s+` alternative plus
/// the `[^\s\p{L}\p{M}\p{N}]+` one leave no gaps for real text — but emitting the gaps anyway means
/// an unmatched byte is tokenized rather than silently deleted, which is the same choice
/// `pretok_pieces`' own "safety net" branch makes.
///
/// Empty matches are skipped (a zero-width match would otherwise emit an empty piece);
/// `fancy_regex`'s own iterator advances past them, so this can't loop.
fn regex_pieces(re: &fancy_regex::Regex, s: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut last = 0usize;
    for m in re.find_iter(s) {
        // A regex-engine error mid-iteration (backtrack limit) leaves the REST of the chunk
        // unsplit rather than dropped: `bpe_piece` still encodes it, just as one piece.
        let Ok(m) = m else { break };
        if m.start() > last {
            out.push((last, m.start()));
        }
        if m.end() > m.start() {
            out.push((m.start(), m.end()));
        }
        last = last.max(m.end());
    }
    if last < s.len() {
        out.push((last, s.len()));
    }
    out
}

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
            // longest match first, same as the reference implementation's qsort(cmp_sp_len).
            specials.sort_by_key(|sp| std::cmp::Reverse(sp.text.len()));
        }

        let (byte2str, cp2byte) = build_bytemap();

        // A checkpoint declaring exactly GLM's cl100k pattern (or declaring no `Split` at all,
        // which is what this crate's own minimal test fixtures do) keeps the hand-rolled path;
        // anything else is compiled and run as written, so a pattern rabbit has never seen still
        // tokenizes per the checkpoint rather than per GLM's assumptions.
        let declared = split_pattern(&root);
        let pretok = match declared {
            None => Pretok::Cl100k,
            Some(p) if p == CL100K_PATTERN => Pretok::Cl100k,
            Some(p) => Pretok::Regex(Box::new(fancy_regex::Regex::new(p).map_err(|e| {
                TokenizerError::UnsupportedPreTokenizer { pattern: p.to_string(), detail: e.to_string() }
            })?)),
        };
        let declared_pattern = declared.map(str::to_string);
        let ignore_merges = model.get("ignore_merges").and_then(Value::as_bool).unwrap_or(true);

        Ok(Tokenizer {
            vocab,
            merges,
            id2str,
            id_added,
            specials,
            byte2str,
            cp2byte,
            pretok,
            declared_pattern,
            ignore_merges,
        })
    }

    /// The `pre_tokenizer.Split` regex the loaded file declared, if any — for a per-architecture
    /// loader to check against the pattern it was written against (see `declared_pattern`'s doc).
    pub fn declared_pretok_pattern(&self) -> Option<&str> {
        self.declared_pattern.as_deref()
    }

    /// The loaded file's `model.ignore_merges` (defaulted per that field's doc).
    pub fn ignore_merges(&self) -> bool {
        self.ignore_merges
    }

    /// BPE-encodes one pre-tokenized piece (raw bytes) and appends resulting ids to `out`.
    fn bpe_piece(&self, bytes: &[u8], out: &mut Vec<i32>) {
        let s = byte_level_encode(&self.byte2str, bytes);

        // ignore_merges: if the whole piece is itself a vocab entry, emit it directly. Gated on the
        // checkpoint's own flag — see `Tokenizer::ignore_merges`; Qwen 3.8 sets it false.
        if self.ignore_merges
            && let Some(&id) = self.vocab.get(s.as_str())
        {
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
        match &self.pretok {
            Pretok::Cl100k => {
                for (s, e) in pretok_pieces(p, a, b) {
                    self.bpe_piece(&p[s..e], out);
                }
            }
            Pretok::Regex(re) => {
                // `encode` only ever hands over slices cut at added-token boundaries of a `&str`,
                // so this is valid UTF-8 in practice; a malformed slice is encoded whole rather
                // than dropped.
                let Ok(chunk) = std::str::from_utf8(&p[a..b]) else {
                    self.bpe_piece(&p[a..b], out);
                    return;
                };
                for (s, e) in regex_pieces(re, chunk) {
                    self.bpe_piece(&chunk.as_bytes()[s..e], out);
                }
            }
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

    /// Same fixture shape as `minimal_tokenizer_json`, plus the two fields that decide
    /// pre-tokenization: a declared `Split` regex and `model.ignore_merges`.
    fn tokenizer_json_with(vocab: &[(String, i32)], pattern: Option<&str>, ignore_merges: Option<bool>) -> String {
        let vocab_json: Vec<String> = vocab.iter().map(|(k, v)| format!("{:?}:{v}", k)).collect();
        let pre = pattern
            .map(|p| {
                format!(
                    r#""pre_tokenizer":{{"type":"Sequence","pretokenizers":[{{"type":"Split","pattern":{{"Regex":{p:?}}},"behavior":"Isolated"}},{{"type":"ByteLevel","use_regex":false}}]}},"#
                )
            })
            .unwrap_or_default();
        let im = ignore_merges.map(|v| format!(r#""ignore_merges":{v},"#)).unwrap_or_default();
        format!(r#"{{{pre}"model":{{"type":"BPE",{im}"vocab":{{{}}},"merges":[]}}}}"#, vocab_json.join(","))
    }

    fn ascii_vocab() -> Vec<(String, i32)> {
        let (byte2str, _) = build_bytemap();
        let mut v = Vec::new();
        for b in 32u8..=126 {
            v.push((byte_level_encode(&byte2str, &[b]), (b as i32) - 32));
        }
        // "é" decomposed is 'e' + U+0301 (0xCC 0x81); byte-level maps those two bytes to their own
        // codepoints, so give both an id to keep the mark test's pieces encodable.
        for (i, b) in [0xCCu8, 0x81].into_iter().enumerate() {
            v.push((byte_level_encode(&byte2str, &[b]), 200 + i as i32));
        }
        v
    }

    /// The whole point of `Pretok`: GLM's exact pattern keeps the hand-rolled state machine, and
    /// anything else (here Qwen 3.8's) is compiled and run as declared.
    #[test]
    fn declared_pattern_picks_the_hand_rolled_path_only_for_cl100k() {
        let v = ascii_vocab();

        let glm = write_fixture("flavor_cl100k", &tokenizer_json_with(&v, Some(CL100K_PATTERN), Some(true)));
        let tok = Tokenizer::load(&glm).unwrap();
        fs::remove_file(&glm).ok();
        assert!(matches!(tok.pretok, Pretok::Cl100k));
        assert_eq!(tok.declared_pretok_pattern(), Some(CL100K_PATTERN));

        let qwen_pattern = crate::qwen38::tokenizer::QWEN38_PRETOK_PATTERN;
        let qwen = write_fixture("flavor_qwen", &tokenizer_json_with(&v, Some(qwen_pattern), Some(false)));
        let tok = Tokenizer::load(&qwen).unwrap();
        fs::remove_file(&qwen).ok();
        assert!(matches!(tok.pretok, Pretok::Regex(_)));

        // no `pre_tokenizer` at all (this module's older fixtures) -> unchanged behavior
        let bare = write_fixture("flavor_none", &tokenizer_json_with(&v, None, None));
        let tok = Tokenizer::load(&bare).unwrap();
        fs::remove_file(&bare).ok();
        assert!(matches!(tok.pretok, Pretok::Cl100k));
        assert!(tok.ignore_merges(), "absent ignore_merges keeps this port's original behavior");
    }

    /// The three real differences between Qwen's declared pattern and cl100k's, checked as PIECES
    /// (not ids, which depend on a vocab): digits split one by one, and a combining mark stays with
    /// its base letter instead of being cut off into the punctuation run.
    #[test]
    fn qwen_pattern_isolates_digits_and_keeps_marks_with_letters() {
        let re = fancy_regex::Regex::new(crate::qwen38::tokenizer::QWEN38_PRETOK_PATTERN).unwrap();
        let piece_strs = |s: &str| -> Vec<String> {
            regex_pieces(&re, s).into_iter().map(|(a, b)| s[a..b].to_string()).collect()
        };

        // Neither pattern gives the digit run a ` ?` prefix (only the letter and punctuation
        // alternatives have one), so the space before a number is always its own piece; what
        // differs is how the digits themselves are grouped.
        assert_eq!(piece_strs("In 2024 we"), vec!["In", " ", "2", "0", "2", "4", " we"]);
        let cl100k: Vec<&str> = pretok_pieces(b"In 2024 we", 0, 10)
            .into_iter()
            .map(|(a, b)| std::str::from_utf8(&b"In 2024 we"[a..b]).unwrap())
            .collect();
        assert_eq!(cl100k, vec!["In", " ", "202", "4", " we"], "cl100k takes up to 3 digits at a time");

        // "cafe" + U+0301 (combining acute): one piece under Qwen's `[\p{L}\p{M}]+`...
        let decomposed = "cafe\u{301} x";
        assert_eq!(piece_strs(decomposed), vec!["cafe\u{301}", " x"]);
        // ...but two under cl100k's `\p{L}+`, where the mark falls into `[^\s\p{L}\p{N}]+`.
        let db = decomposed.as_bytes();
        let split: Vec<&str> = pretok_pieces(db, 0, db.len())
            .into_iter()
            .map(|(a, b)| std::str::from_utf8(&db[a..b]).unwrap())
            .collect();
        assert_eq!(split, vec!["cafe", "\u{301}", " x"]);
    }

    /// Pieces must tile the input exactly — no dropped bytes even for input the pattern's
    /// alternatives don't cover, which is what `regex_pieces`' gap handling is for.
    #[test]
    fn regex_pieces_cover_the_whole_input() {
        let re = fancy_regex::Regex::new(crate::qwen38::tokenizer::QWEN38_PRETOK_PATTERN).unwrap();
        for s in ["", "a", "  ", "hola, ¿qué tal?\n\n42 🎉", "\u{0}\u{1}raro"] {
            let pieces = regex_pieces(&re, s);
            let rebuilt: String = pieces.iter().map(|&(a, b)| &s[a..b]).collect();
            assert_eq!(rebuilt, s, "pieces must tile {s:?}");
            for w in pieces.windows(2) {
                assert_eq!(w[0].1, w[1].0, "no gaps/overlaps in {s:?}");
            }
        }
    }

    /// With `ignore_merges: false` (Qwen's setting) a piece that IS a vocab entry must still go
    /// through the merge table — the mirror image of `ignore_merges_shortcut_bypasses_rank_ordering`.
    #[test]
    fn ignore_merges_false_does_not_shortcut_whole_pieces() {
        let (byte2str, _) = build_bytemap();
        let bl = |s: &str| byte_level_encode(&byte2str, s.as_bytes());
        let vocab = [(bl("a"), 0), (bl("b"), 1), (bl("ab"), 2)];

        let on = write_fixture("im_on", &tokenizer_json_with(&vocab, None, Some(true)));
        let tok_on = Tokenizer::load(&on).unwrap();
        fs::remove_file(&on).ok();
        assert_eq!(tok_on.encode("ab"), vec![2], "shortcut ON: whole piece emitted directly");

        let off = write_fixture("im_off", &tokenizer_json_with(&vocab, None, Some(false)));
        let tok_off = Tokenizer::load(&off).unwrap();
        fs::remove_file(&off).ok();
        assert_eq!(tok_off.encode("ab"), vec![0, 1], "shortcut OFF: no merge exists, so two ids");
    }

    #[test]
    fn rejects_a_split_regex_the_engine_cannot_compile() {
        let v = ascii_vocab();
        let path = write_fixture("bad_regex", &tokenizer_json_with(&v, Some("(unclosed[a-"), None));
        let result = Tokenizer::load(&path);
        fs::remove_file(&path).ok();
        match result {
            Err(e @ TokenizerError::UnsupportedPreTokenizer { .. }) => {
                assert!(e.to_string().contains("unclosed"), "the error must quote the offending pattern: {e}");
            }
            Err(other) => panic!("wrong error: {other}"),
            Ok(_) => panic!("an uncompilable Split regex must not load"),
        }
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
