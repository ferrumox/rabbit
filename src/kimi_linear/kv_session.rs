//! Persists a `kimi_linear::generate::KvState` across process restarts for `--chat --session
//! <path>` — the sibling of `crate::kv_session`'s GLM-5.2 format, genuinely different rather
//! than reused, because KDA's per-layer state (`KdaState`'s `[d_k,d_v]` matrix, `ShortConvState`'s
//! FIFOs) is FIXED size and mutated in place, unlike MLA's `KvCache`, which only ever grows by
//! one row per token (see `kimi_linear::generate`'s own doc for why `KvState` is an enum over two
//! genuinely different shapes in the first place).
//!
//! **Two files, one persistence strategy each:**
//!
//! - `<path>` — an append-only log of new MLA rows per completed turn, format identical in
//!   spirit to `crate::kv_session`'s (own magic, no DSA — Kimi has none): appending is the right
//!   strategy here for the exact same reason GLM's format appends (total bytes written stays
//!   linear in conversation length, not quadratic).
//! - `<path>.kda` — the CURRENT KDA state (every KDA layer's head matrices + conv FIFOs),
//!   written via temp-file-then-`rename` so it's replaced atomically. Appending here instead
//!   would be wasteful, not just inelegant: KDA state doesn't grow with position (see above), so
//!   an "append the new rows" record doesn't exist for it — the only content a save could add is
//!   another full copy of the (already large — e.g. ~45MB for the real 48B checkpoint's 20 KDA
//!   layers) current state, which would make the file grow by that much every single turn for no
//!   reason. An atomic in-place replace keeps the file's size roughly constant instead.
//!
//! **Crash consistency across the two files**: `save` always appends the MLA record FIRST, then
//! atomically replaces the `.kda` file SECOND. A crash between the two steps (or mid-write of
//! either) can only ever leave the `.kda` file's `pos` <= the MLA log's furthest complete
//! `pos_after` (never the other way — the `.kda` rename can't happen before the MLA append it
//! follows). `load` exploits exactly that: it trusts the `.kda` file's `pos` as authoritative and
//! stops replaying MLA records once their `pos_after` would exceed it, discarding any newer
//! orphaned records the same way `crate::kv_session` discards a truncated trailing one. Either
//! file missing entirely (never yet saved) is treated as "no data of that kind" rather than an
//! error, so a model with zero KDA layers (or, hypothetically, zero MLA layers) degrades cleanly
//! to needing only the one file its shape actually has.
//!
//! All integers little-endian.

use crate::glm52::attention::KvCache;
use crate::kimi_linear::generate::{KdaLayerState, KvState, LayerState};
use crate::kimi_linear::kda::KdaState;
use crate::kimi_linear::model::{Attn, Model};
use crate::kimi_linear::short_conv::ShortConvState;
use std::ffi::OsString;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const MLA_MAGIC: &[u8; 8] = b"RBTKVKM1";
const MLA_VERSION: u32 = 1;
const KDA_MAGIC: &[u8; 8] = b"RBTKDAS1";
const KDA_VERSION: u32 = 1;

#[derive(Debug)]
pub enum LoadError {
    Io(io::Error),
    BadMagic,
    UnsupportedVersion(u32),
    LayerCountMismatch { file: usize, model: usize, what: &'static str },
    ConfigMismatch { layer: usize, field: &'static str, file: u32, model: u32 },
    Corrupt(String),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "session file: {e}"),
            LoadError::BadMagic => write!(f, "session file: bad magic (not a rabbit Kimi Linear KV session file)"),
            LoadError::UnsupportedVersion(v) => write!(f, "session file: unsupported version {v}"),
            LoadError::LayerCountMismatch { file, model, what } => {
                write!(f, "session file: {file} {what} layers saved, model has {model}")
            }
            LoadError::ConfigMismatch { layer, field, file, model } => {
                write!(f, "session file: layer {layer} {field}={file}, model expects {model}")
            }
            LoadError::Corrupt(m) => write!(f, "session file: corrupt: {m}"),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<io::Error> for LoadError {
    fn from(e: io::Error) -> Self {
        LoadError::Io(e)
    }
}

#[derive(Clone, Copy, PartialEq)]
struct MlaShape {
    kv_lora: u32,
    qk_rope: u32,
}

#[derive(Clone, Copy, PartialEq)]
struct KdaShape {
    n_heads: u32,
    head_dim: u32,
    kernel: u32,
}

fn kda_path(path: &Path) -> PathBuf {
    let mut s: OsString = path.as_os_str().to_owned();
    s.push(".kda");
    PathBuf::from(s)
}

fn kda_tmp_path(path: &Path) -> PathBuf {
    let mut s: OsString = path.as_os_str().to_owned();
    s.push(".kda.tmp");
    PathBuf::from(s)
}

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn decode_f32s(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// Reads up to `buf.len()` bytes, filling as much as possible before EOF — see
/// `crate::kv_session::read_partial`'s doc (duplicated rather than shared, matching this whole
/// module's "sibling, not reused" relationship to GLM's).
fn read_partial(r: &mut impl Read, buf: &mut [u8]) -> io::Result<usize> {
    let mut n = 0;
    while n < buf.len() {
        match r.read(&mut buf[n..])? {
            0 => break,
            k => n += k,
        }
    }
    Ok(n)
}

fn mla_caches(state: &KvState) -> impl Iterator<Item = &KvCache> {
    state.layers().iter().filter_map(|l| match l {
        LayerState::Mla(kv) => Some(kv),
        LayerState::Kda(_) => None,
    })
}

fn kda_layers(state: &KvState) -> impl Iterator<Item = &KdaLayerState> {
    state.layers().iter().filter_map(|l| match l {
        LayerState::Kda(k) => Some(k),
        LayerState::Mla(_) => None,
    })
}

fn mla_shape_of(state: &KvState) -> Vec<MlaShape> {
    mla_caches(state).map(|kv| MlaShape { kv_lora: kv.kv_lora() as u32, qk_rope: kv.qk_rope() as u32 }).collect()
}

fn kda_shape_of(state: &KvState) -> Vec<KdaShape> {
    kda_layers(state)
        .map(|kda| KdaShape {
            n_heads: kda.heads().len() as u32,
            head_dim: kda.heads().first().map(|h| h.d_k() as u32).unwrap_or(0),
            kernel: kda.q_conv().kernel() as u32,
        })
        .collect()
}

fn mla_shape_of_model(model: &Model) -> Vec<MlaShape> {
    let shape = MlaShape { kv_lora: model.cfg.kv_lora as u32, qk_rope: model.cfg.qk_rope as u32 };
    model.layers.iter().filter(|l| matches!(l.attn, Attn::Mla(_))).map(|_| shape).collect()
}

fn kda_shape_of_model(model: &Model) -> Vec<KdaShape> {
    let shape = KdaShape { n_heads: model.cfg.kda_n_heads as u32, head_dim: model.cfg.kda_head_dim as u32, kernel: model.cfg.short_conv_kernel as u32 };
    model.layers.iter().filter(|l| matches!(l.attn, Attn::Kda(_))).map(|_| shape).collect()
}

fn write_mla_header(shape: &[MlaShape], w: &mut impl Write) -> io::Result<()> {
    w.write_all(MLA_MAGIC)?;
    w.write_all(&MLA_VERSION.to_le_bytes())?;
    w.write_all(&(shape.len() as u32).to_le_bytes())?;
    for l in shape {
        w.write_all(&l.kv_lora.to_le_bytes())?;
        w.write_all(&l.qk_rope.to_le_bytes())?;
    }
    Ok(())
}

fn read_mla_header(r: &mut impl Read) -> Result<Vec<MlaShape>, LoadError> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MLA_MAGIC {
        return Err(LoadError::BadMagic);
    }
    let mut u32buf = [0u8; 4];
    r.read_exact(&mut u32buf)?;
    let version = u32::from_le_bytes(u32buf);
    if version != MLA_VERSION {
        return Err(LoadError::UnsupportedVersion(version));
    }
    r.read_exact(&mut u32buf)?;
    let n = u32::from_le_bytes(u32buf) as usize;
    let mut layers = Vec::with_capacity(n);
    for _ in 0..n {
        r.read_exact(&mut u32buf)?;
        let kv_lora = u32::from_le_bytes(u32buf);
        r.read_exact(&mut u32buf)?;
        let qk_rope = u32::from_le_bytes(u32buf);
        layers.push(MlaShape { kv_lora, qk_rope });
    }
    Ok(layers)
}

fn write_mla_record(state: &KvState, from_pos: usize, to_pos: usize, w: &mut impl Write) -> io::Result<()> {
    let n_new = (to_pos - from_pos) as u64;
    w.write_all(&(to_pos as u64).to_le_bytes())?;
    w.write_all(&n_new.to_le_bytes())?;
    for kv in mla_caches(state) {
        w.write_all(&f32_bytes(kv.l_range(from_pos, to_pos)))?;
        w.write_all(&f32_bytes(kv.r_range(from_pos, to_pos)))?;
    }
    Ok(())
}

fn save_mla_log(shape: &[MlaShape], state: &KvState, from_pos: usize, to_pos: usize, path: &Path) -> io::Result<()> {
    let exists = path.exists();
    if exists {
        if from_pos == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "kimi kv_session: from_pos=0 but session file already exists"));
        }
        let mut header_reader = BufReader::new(File::open(path)?);
        let file_shape = read_mla_header(&mut header_reader).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        if file_shape != shape {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "kimi kv_session: session file shape does not match the current session"));
        }
    } else if from_pos != 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "kimi kv_session: from_pos must be 0 when creating a new session file"));
    }

    let mut w = BufWriter::new(OpenOptions::new().create(true).append(true).open(path)?);
    if !exists {
        write_mla_header(shape, &mut w)?;
    }
    write_mla_record(state, from_pos, to_pos, &mut w)?;
    w.flush()
}

fn write_kda_header(shape: &[KdaShape], pos: u64, w: &mut impl Write) -> io::Result<()> {
    w.write_all(KDA_MAGIC)?;
    w.write_all(&KDA_VERSION.to_le_bytes())?;
    w.write_all(&pos.to_le_bytes())?;
    w.write_all(&(shape.len() as u32).to_le_bytes())?;
    for l in shape {
        w.write_all(&l.n_heads.to_le_bytes())?;
        w.write_all(&l.head_dim.to_le_bytes())?;
        w.write_all(&l.kernel.to_le_bytes())?;
    }
    Ok(())
}

fn read_kda_header(r: &mut impl Read) -> Result<(u64, Vec<KdaShape>), LoadError> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != KDA_MAGIC {
        return Err(LoadError::BadMagic);
    }
    let mut u32buf = [0u8; 4];
    r.read_exact(&mut u32buf)?;
    let version = u32::from_le_bytes(u32buf);
    if version != KDA_VERSION {
        return Err(LoadError::UnsupportedVersion(version));
    }
    let mut u64buf = [0u8; 8];
    r.read_exact(&mut u64buf)?;
    let pos = u64::from_le_bytes(u64buf);
    r.read_exact(&mut u32buf)?;
    let n = u32::from_le_bytes(u32buf) as usize;
    let mut shapes = Vec::with_capacity(n);
    for _ in 0..n {
        r.read_exact(&mut u32buf)?;
        let n_heads = u32::from_le_bytes(u32buf);
        r.read_exact(&mut u32buf)?;
        let head_dim = u32::from_le_bytes(u32buf);
        r.read_exact(&mut u32buf)?;
        let kernel = u32::from_le_bytes(u32buf);
        shapes.push(KdaShape { n_heads, head_dim, kernel });
    }
    Ok((pos, shapes))
}

fn save_kda_snapshot_atomic(shape: &[KdaShape], state: &KvState, pos: usize, path: &Path) -> io::Result<()> {
    let tmp_p = kda_tmp_path(path);
    {
        let mut w = BufWriter::new(File::create(&tmp_p)?);
        write_kda_header(shape, pos as u64, &mut w)?;
        for kda in kda_layers(state) {
            for head in kda.heads() {
                w.write_all(&f32_bytes(head.raw()))?;
            }
            w.write_all(&f32_bytes(kda.q_conv().history()))?;
            w.write_all(&f32_bytes(kda.k_conv().history()))?;
            w.write_all(&f32_bytes(kda.v_conv().history()))?;
        }
        w.flush()?;
    }
    fs::rename(&tmp_p, kda_path(path))
}

impl KvState {
    /// Appends everything new in `[from_pos, to_pos)` to `path` (MLA layers) and atomically
    /// replaces `path`'s `.kda` sidecar with the current KDA state — see this module's doc for
    /// why each half uses the strategy it does, and the write order (MLA log first, `.kda`
    /// second) that makes `load` crash-consistent.
    pub fn save(&self, from_pos: usize, to_pos: usize, path: &Path) -> io::Result<()> {
        if to_pos <= from_pos {
            return Ok(());
        }
        let mla_shape = mla_shape_of(self);
        let kda_shape = kda_shape_of(self);

        if !mla_shape.is_empty() {
            save_mla_log(&mla_shape, self, from_pos, to_pos, path)?;
        }
        if !kda_shape.is_empty() {
            save_kda_snapshot_atomic(&kda_shape, self, to_pos, path)?;
        }
        Ok(())
    }

    /// Loads a previously-saved session, validating both files (whichever the model's shape
    /// actually needs) against `model` — a mismatch is a hard error, never a silent fallback to
    /// an empty session. See this module's doc for the crash-recovery rule (trust the `.kda`
    /// file's `pos`, discard any MLA records past it).
    pub fn load(path: &Path, model: &Model) -> Result<(KvState, usize), LoadError> {
        let mla_shape_model = mla_shape_of_model(model);
        let kda_shape_model = kda_shape_of_model(model);

        let (kda_pos, kda_state_layers): (Option<u64>, Vec<KdaLayerState>) = if kda_shape_model.is_empty() {
            (None, Vec::new())
        } else {
            let mut f = BufReader::new(File::open(kda_path(path))?);
            let (pos, file_shape) = read_kda_header(&mut f)?;
            if file_shape.len() != kda_shape_model.len() {
                return Err(LoadError::LayerCountMismatch { file: file_shape.len(), model: kda_shape_model.len(), what: "kda" });
            }
            let mut layers = Vec::with_capacity(file_shape.len());
            for (i, (fs, ms)) in file_shape.iter().zip(&kda_shape_model).enumerate() {
                if fs.n_heads != ms.n_heads {
                    return Err(LoadError::ConfigMismatch { layer: i, field: "kda_n_heads", file: fs.n_heads, model: ms.n_heads });
                }
                if fs.head_dim != ms.head_dim {
                    return Err(LoadError::ConfigMismatch { layer: i, field: "kda_head_dim", file: fs.head_dim, model: ms.head_dim });
                }
                if fs.kernel != ms.kernel {
                    return Err(LoadError::ConfigMismatch { layer: i, field: "short_conv_kernel", file: fs.kernel, model: ms.kernel });
                }

                let head_dim = fs.head_dim as usize;
                let n_heads = fs.n_heads as usize;
                let d_inner = head_dim * n_heads;
                let hist_len = (fs.kernel as usize).saturating_sub(1);

                let mut heads = Vec::with_capacity(n_heads);
                for _ in 0..n_heads {
                    let mut buf = vec![0u8; head_dim * head_dim * 4];
                    f.read_exact(&mut buf)?;
                    heads.push(KdaState::from_raw(head_dim, head_dim, decode_f32s(&buf)));
                }
                let read_conv = |f: &mut BufReader<File>| -> io::Result<ShortConvState> {
                    let mut buf = vec![0u8; d_inner * hist_len * 4];
                    f.read_exact(&mut buf)?;
                    Ok(ShortConvState::from_raw(d_inner, fs.kernel as usize, decode_f32s(&buf)))
                };
                let q_conv = read_conv(&mut f)?;
                let k_conv = read_conv(&mut f)?;
                let v_conv = read_conv(&mut f)?;
                layers.push(KdaLayerState::from_raw(heads, q_conv, k_conv, v_conv));
            }
            (Some(pos), layers)
        };

        let (mla_pos, mla_state_caches): (Option<u64>, Vec<KvCache>) = if mla_shape_model.is_empty() {
            (None, Vec::new())
        } else {
            let mut f = BufReader::new(File::open(path)?);
            let file_shape = read_mla_header(&mut f)?;
            if file_shape.len() != mla_shape_model.len() {
                return Err(LoadError::LayerCountMismatch { file: file_shape.len(), model: mla_shape_model.len(), what: "mla" });
            }
            for (i, (fs, ms)) in file_shape.iter().zip(&mla_shape_model).enumerate() {
                if fs.kv_lora != ms.kv_lora {
                    return Err(LoadError::ConfigMismatch { layer: i, field: "kv_lora", file: fs.kv_lora, model: ms.kv_lora });
                }
                if fs.qk_rope != ms.qk_rope {
                    return Err(LoadError::ConfigMismatch { layer: i, field: "qk_rope", file: fs.qk_rope, model: ms.qk_rope });
                }
            }

            let n_layers = file_shape.len();
            let mut ls: Vec<Vec<f32>> = vec![Vec::new(); n_layers];
            let mut rs: Vec<Vec<f32>> = vec![Vec::new(); n_layers];
            let mut pos: u64 = 0;

            loop {
                if let Some(cap) = kda_pos
                    && pos >= cap
                {
                    break;
                }
                let mut rec_header = [0u8; 16];
                let got = read_partial(&mut f, &mut rec_header)?;
                if got < 16 {
                    break;
                }
                let pos_after = u64::from_le_bytes(rec_header[0..8].try_into().unwrap());
                let n_new = u64::from_le_bytes(rec_header[8..16].try_into().unwrap());
                if let Some(cap) = kda_pos
                    && pos_after > cap
                {
                    // Orphaned record from a save interrupted after the MLA append but before
                    // the .kda replace -- the KDA snapshot doesn't cover it, so neither should
                    // the state we reconstruct (see this module's doc).
                    break;
                }

                let mut payloads: Vec<(Vec<f32>, Vec<f32>)> = Vec::with_capacity(n_layers);
                let mut complete = true;
                for shape in &file_shape {
                    let l_len = n_new as usize * shape.kv_lora as usize * 4;
                    let r_len = n_new as usize * shape.qk_rope as usize * 4;
                    let mut l_buf = vec![0u8; l_len];
                    if read_partial(&mut f, &mut l_buf)? < l_len {
                        complete = false;
                        break;
                    }
                    let mut r_buf = vec![0u8; r_len];
                    if read_partial(&mut f, &mut r_buf)? < r_len {
                        complete = false;
                        break;
                    }
                    payloads.push((decode_f32s(&l_buf), decode_f32s(&r_buf)));
                }
                if !complete || payloads.len() != n_layers {
                    break;
                }
                if pos_after != pos + n_new {
                    return Err(LoadError::Corrupt(format!("mla record pos_after={pos_after} but expected {} (pos={pos}, n_new={n_new})", pos + n_new)));
                }
                for (i, (l, r)) in payloads.into_iter().enumerate() {
                    ls[i].extend(l);
                    rs[i].extend(r);
                }
                pos = pos_after;
            }

            let mut caches = Vec::with_capacity(n_layers);
            for (i, shape) in file_shape.iter().enumerate() {
                let l = std::mem::take(&mut ls[i]);
                let r = std::mem::take(&mut rs[i]);
                caches.push(KvCache::from_raw(shape.kv_lora as usize, shape.qk_rope as usize, l, r));
            }
            (Some(pos), caches)
        };

        let effective_pos = match (kda_pos, mla_pos) {
            (Some(k), Some(m)) => k.min(m),
            (Some(k), None) => k,
            (None, Some(m)) => m,
            (None, None) => 0,
        };

        let mut kda_iter = kda_state_layers.into_iter();
        let mut mla_iter = mla_state_caches.into_iter();
        let layers = model
            .layers
            .iter()
            .map(|l| match &l.attn {
                Attn::Kda(_) => LayerState::Kda(kda_iter.next().expect("kda layer count validated above")),
                Attn::Mla(_) => LayerState::Mla(mla_iter.next().expect("mla layer count validated above")),
            })
            .collect();

        Ok((KvState::from_raw(layers), effective_pos as usize))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kimi_linear::generate::{step, ExpertCaches};
    use crate::safetensors::Shards;
    use serde_json::json;
    use std::fs;

    struct TempDir(std::path::PathBuf);
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

    fn xorshift(seed: &mut u32) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 17;
        *seed ^= *seed << 5;
        ((*seed as f32 / u32::MAX as f32) - 0.5) * 2.0
    }

    fn random_vec(n: usize, seed: &mut u32) -> Vec<f32> {
        (0..n).map(|_| xorshift(seed)).collect()
    }

    fn f32b(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// One KDA layer + one MLA layer, dense FFN — same shape as `generate.rs`'s own
    /// `build_two_layer_fixture` (that module already covers KDA/MLA dispatch correctness in
    /// depth; this fixture exists only to give `save`/`load` real, non-trivial state to persist).
    fn build_fixture(name: &str) -> TempDir {
        let dir = TempDir::new(name);
        let mut seed = 9u32;
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_string(), json!({"format": "rabbit-test"}));
        let mut data = Vec::new();
        let mut add = |header: &mut serde_json::Map<String, serde_json::Value>, name: String, shape: Vec<usize>| {
            let n: usize = shape.iter().product::<usize>().max(1);
            let bytes = f32b(&random_vec(n, &mut seed));
            let start = data.len() as u64;
            data.extend_from_slice(&bytes);
            let end = data.len() as u64;
            header.insert(name, json!({"dtype": "F32", "shape": shape, "data_offsets": [start, end]}));
        };

        let d = 8;
        let h = 2;
        let qk_nope = 3;
        let qk_rope = 2;
        let qh = qk_nope + qk_rope;
        let v_head = 4;
        let kv_lora = 5;
        let vocab = 16;
        let dense_inter = 10;
        let kda_head_dim = 4;
        let kda_n_heads = 2;
        let d_inner = kda_head_dim * kda_n_heads;
        let kernel = 3;

        add(&mut header, "model.embed_tokens.weight".into(), vec![vocab, d]);
        add(&mut header, "lm_head.weight".into(), vec![vocab, d]);
        add(&mut header, "model.norm.weight".into(), vec![d]);

        for i in 0..2usize {
            let p = |s: &str| format!("model.layers.{i}.{s}");
            add(&mut header, p("input_layernorm.weight"), vec![d]);
            add(&mut header, p("post_attention_layernorm.weight"), vec![d]);
            add(&mut header, p("mlp.gate_proj.weight"), vec![dense_inter, d]);
            add(&mut header, p("mlp.up_proj.weight"), vec![dense_inter, d]);
            add(&mut header, p("mlp.down_proj.weight"), vec![d, dense_inter]);

            let ap = |s: &str| format!("model.layers.{i}.self_attn.{s}");
            if i == 0 {
                add(&mut header, ap("q_proj.weight"), vec![d_inner, d]);
                add(&mut header, ap("k_proj.weight"), vec![d_inner, d]);
                add(&mut header, ap("v_proj.weight"), vec![d_inner, d]);
                add(&mut header, ap("q_conv1d.weight"), vec![d_inner, 1, kernel]);
                add(&mut header, ap("k_conv1d.weight"), vec![d_inner, 1, kernel]);
                add(&mut header, ap("v_conv1d.weight"), vec![d_inner, 1, kernel]);
                add(&mut header, ap("f_a_proj.weight"), vec![kda_head_dim, d]);
                add(&mut header, ap("f_b_proj.weight"), vec![d_inner, kda_head_dim]);
                add(&mut header, ap("dt_bias"), vec![d_inner]);
                add(&mut header, ap("A_log"), vec![1, 1, kda_n_heads, 1]);
                add(&mut header, ap("b_proj.weight"), vec![kda_n_heads, d]);
                add(&mut header, ap("g_a_proj.weight"), vec![kda_head_dim, d]);
                add(&mut header, ap("g_b_proj.weight"), vec![d_inner, kda_head_dim]);
                add(&mut header, ap("o_norm.weight"), vec![kda_head_dim]);
                add(&mut header, ap("o_proj.weight"), vec![d, d_inner]);
            } else {
                add(&mut header, ap("q_proj.weight"), vec![h * qh, d]);
                add(&mut header, ap("kv_a_proj_with_mqa.weight"), vec![kv_lora + qk_rope, d]);
                add(&mut header, ap("kv_a_layernorm.weight"), vec![kv_lora]);
                add(&mut header, ap("kv_b_proj.weight"), vec![h * (qk_nope + v_head), kv_lora]);
                add(&mut header, ap("o_proj.weight"), vec![d, h * v_head]);
            }
        }

        let cfg_json = json!({
            "model_type": "kimi_linear",
            "hidden_size": d, "num_hidden_layers": 2, "num_attention_heads": h,
            "first_k_dense_replace": 2, "q_lora_rank": null, "kv_lora_rank": kv_lora,
            "qk_nope_head_dim": qk_nope, "qk_rope_head_dim": qk_rope, "v_head_dim": v_head,
            "num_experts": 1, "num_experts_per_token": 1, "num_shared_experts": 0,
            "num_expert_group": 1, "topk_group": 1, "moe_intermediate_size": 1,
            "intermediate_size": dense_inter, "vocab_size": vocab, "moe_renormalize": true,
            "rms_norm_eps": 1e-5, "routed_scaling_factor": 1.0, "mla_use_nope": true,
            "moe_router_activation_func": "sigmoid", "rope_theta": 10000.0,
            "linear_attn_config": {
                "head_dim": kda_head_dim, "num_heads": kda_n_heads, "short_conv_kernel_size": kernel,
                "kda_layers": [1], "full_attn_layers": [2]
            }
        });
        fs::write(dir.0.join("config.json"), cfg_json.to_string()).unwrap();

        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_bytes);
        out.extend_from_slice(&data);
        fs::write(dir.0.join("model.safetensors"), out).unwrap();

        dir
    }

    #[test]
    fn save_then_load_roundtrip_continues_identically() {
        let fixture = build_fixture("rabbit_test_kimi_kvsession_roundtrip");
        let model = Model::load(&fixture.0, 32, 32).unwrap();
        let shards = Shards::open(&fixture.0).unwrap();
        let session_path = fixture.0.join("session.bin");

        let mut caches_a = ExpertCaches::new(&model, 4);
        let mut kv_a = KvState::new(&model);
        let ids = vec![1usize, 2, 3];
        step(&model, &shards, &mut caches_a, &mut kv_a, &ids, 0).unwrap();
        let pos = ids.len();

        kv_a.save(0, pos, &session_path).unwrap();
        let (mut kv_b, loaded_pos) = KvState::load(&session_path, &model).unwrap();
        assert_eq!(loaded_pos, pos);

        let next_ids = vec![4usize];
        let mut caches_b = ExpertCaches::new(&model, 4);
        let logits_a = step(&model, &shards, &mut caches_a, &mut kv_a, &next_ids, pos).unwrap();
        let logits_b = step(&model, &shards, &mut caches_b, &mut kv_b, &next_ids, pos).unwrap();
        assert_eq!(logits_a, logits_b, "resumed session must continue bit-for-bit identically to the original");
    }

    #[test]
    fn two_turn_session_roundtrip_continues_identically() {
        let fixture = build_fixture("rabbit_test_kimi_kvsession_two_turn");
        let model = Model::load(&fixture.0, 32, 32).unwrap();
        let shards = Shards::open(&fixture.0).unwrap();
        let session_path = fixture.0.join("session.bin");

        let mut caches_a = ExpertCaches::new(&model, 4);
        let mut kv_a = KvState::new(&model);

        let turn1 = vec![1usize, 2];
        step(&model, &shards, &mut caches_a, &mut kv_a, &turn1, 0).unwrap();
        kv_a.save(0, turn1.len(), &session_path).unwrap();

        let turn2 = vec![3usize, 4, 5];
        step(&model, &shards, &mut caches_a, &mut kv_a, &turn2, turn1.len()).unwrap();
        let pos = turn1.len() + turn2.len();
        kv_a.save(turn1.len(), pos, &session_path).unwrap();

        let (mut kv_b, loaded_pos) = KvState::load(&session_path, &model).unwrap();
        assert_eq!(loaded_pos, pos);

        let next_ids = vec![6usize];
        let mut caches_b = ExpertCaches::new(&model, 4);
        let logits_a = step(&model, &shards, &mut caches_a, &mut kv_a, &next_ids, pos).unwrap();
        let logits_b = step(&model, &shards, &mut caches_b, &mut kv_b, &next_ids, pos).unwrap();
        assert_eq!(logits_a, logits_b);
    }

    #[test]
    fn crash_after_mla_append_before_kda_replace_recovers_to_previous_turn() {
        // Directly simulates a crash between save()'s two steps: the MLA log gets turn 2's
        // record, but the .kda sidecar is never updated past turn 1 -- load must recover to
        // turn 1's state (the last point both files agree on), not turn 2's.
        let fixture = build_fixture("rabbit_test_kimi_kvsession_crash_before_kda");
        let model = Model::load(&fixture.0, 32, 32).unwrap();
        let shards = Shards::open(&fixture.0).unwrap();
        let session_path = fixture.0.join("session.bin");

        let mut caches_ref = ExpertCaches::new(&model, 4);
        let mut kv_ref = KvState::new(&model); // ground truth: only turn 1 ever applied
        let turn1 = vec![1usize, 2];
        step(&model, &shards, &mut caches_ref, &mut kv_ref, &turn1, 0).unwrap();
        kv_ref.save(0, turn1.len(), &session_path).unwrap(); // both files land at pos=turn1.len()

        // Now advance a SEPARATE copy of the state through turn 2, but only append its MLA
        // record -- deliberately skip the .kda replace, reproducing the crash window.
        let mut caches_full = ExpertCaches::new(&model, 4);
        let mut kv_full = KvState::new(&model);
        step(&model, &shards, &mut caches_full, &mut kv_full, &turn1, 0).unwrap();
        let turn2 = vec![3usize, 4];
        step(&model, &shards, &mut caches_full, &mut kv_full, &turn2, turn1.len()).unwrap();
        let mla_shape = mla_shape_of(&kv_full);
        save_mla_log(&mla_shape, &kv_full, turn1.len(), turn1.len() + turn2.len(), &session_path).unwrap();

        let (mut kv_loaded, loaded_pos) = KvState::load(&session_path, &model).unwrap();
        assert_eq!(loaded_pos, turn1.len(), "must recover to the last position BOTH files agree on");

        let next_ids = vec![9usize];
        let mut caches_ref2 = ExpertCaches::new(&model, 4);
        let mut caches_loaded = ExpertCaches::new(&model, 4);
        let logits_ref = step(&model, &shards, &mut caches_ref2, &mut kv_ref, &next_ids, turn1.len()).unwrap();
        let logits_loaded = step(&model, &shards, &mut caches_loaded, &mut kv_loaded, &next_ids, turn1.len()).unwrap();
        assert_eq!(logits_ref, logits_loaded);
    }

    #[test]
    fn truncated_trailing_mla_record_recovers_to_previous_complete_turn() {
        let fixture = build_fixture("rabbit_test_kimi_kvsession_truncated");
        let model = Model::load(&fixture.0, 32, 32).unwrap();
        let shards = Shards::open(&fixture.0).unwrap();
        let session_path = fixture.0.join("session.bin");

        let mut caches = ExpertCaches::new(&model, 4);
        let mut kv = KvState::new(&model);
        let turn1 = vec![1usize, 2];
        step(&model, &shards, &mut caches, &mut kv, &turn1, 0).unwrap();
        kv.save(0, turn1.len(), &session_path).unwrap();
        let len_after_turn1 = fs::metadata(&session_path).unwrap().len();

        let turn2 = vec![3usize, 4, 5];
        step(&model, &shards, &mut caches, &mut kv, &turn2, turn1.len()).unwrap();
        kv.save(turn1.len(), turn1.len() + turn2.len(), &session_path).unwrap();

        // Simulate a crash mid-append of turn 2's record: cut the file back to just past where
        // turn 1's complete record ends.
        let f = fs::OpenOptions::new().write(true).open(&session_path).unwrap();
        f.set_len(len_after_turn1 + 4).unwrap();

        let (_kv_loaded, loaded_pos) = KvState::load(&session_path, &model).unwrap();
        assert_eq!(loaded_pos, turn1.len(), "a torn trailing record must be discarded, recovering to the last complete turn");
    }

    #[test]
    fn kda_shape_mismatch_on_load_is_a_hard_error() {
        let fixture = build_fixture("rabbit_test_kimi_kvsession_shape_mismatch");
        let model = Model::load(&fixture.0, 32, 32).unwrap();
        let shards = Shards::open(&fixture.0).unwrap();
        let session_path = fixture.0.join("session.bin");

        let mut caches = ExpertCaches::new(&model, 4);
        let mut kv = KvState::new(&model);
        let ids = vec![1usize, 2];
        step(&model, &shards, &mut caches, &mut kv, &ids, 0).unwrap();
        kv.save(0, ids.len(), &session_path).unwrap();

        // Loading against a model whose KDA head_dim differs from what was saved must be a hard
        // error, not a silent reinterpretation of the raw bytes.
        let mut model_mismatched = Model::load(&fixture.0, 32, 32).unwrap();
        model_mismatched.cfg.kda_head_dim = 8;
        match KvState::load(&session_path, &model_mismatched) {
            Err(LoadError::ConfigMismatch { field: "kda_head_dim", .. }) => {}
            Err(e) => panic!("expected a kda_head_dim ConfigMismatch, got a different error: {e}"),
            Ok(_) => panic!("expected a kda_head_dim ConfigMismatch, got Ok"),
        }
    }
}
