//! Persists a `KvState` across process restarts for `--chat --session <path>` (colibrì's
//! `.coli_kv` equivalent). `--serve` is deliberately stateless and never touches this module.
//!
//! Append-only format, not rewrite-on-every-turn: rewriting the whole state every turn makes
//! total bytes written grow quadratically with conversation length (see `rabbit-plan.md`).
//! One record per completed turn is appended instead, and a crash mid-write only ever corrupts
//! the LAST record, which `load` detects and discards (the conversation resumes from the last
//! complete turn instead of failing outright).
//!
//! All integers little-endian.
//!
//! Header (once, at file creation):
//!   magic: [u8; 8] = b"RBTKVSN1", version: u32, n_layers: u32,
//!   per layer: kv_lora: u32, qk_rope: u32, dsa_present: u8, [hd: u32 if dsa_present]
//!
//! Record (appended per completed turn):
//!   pos_after: u64, n_new: u64  (invariant: pos_after == pos_before + n_new)
//!   per layer: n_new*kv_lora f32 LE (new L rows), n_new*qk_rope f32 LE (new R rows), and if
//!              that layer has DSA: dsa_n_new: u64, then dsa_n_new*hd f32 LE (new DSA K rows).
//!
//! `dsa_n_new` is tracked separately from `n_new` (not assumed equal) because `generate.rs`'s
//! `layers_forward` only computes the DSA indexer on a session's very first forward call
//! (`pos_base == 0`); every later call — every turn after the first in `--chat` — leaves each
//! layer's `DsaCache` exactly as long as it was, even though the KV cache keeps growing. A
//! record's DSA segment is whatever overlap `[from_pos, to_pos)` has with `[0, dsa_cache.len())`
//! — full `n_new` rows while the cache is still growing, zero once it's done.

use crate::attention::{DsaCache, KvCache};
use crate::generate::KvState;
use crate::model::Model;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"RBTKVSN1";
const VERSION: u32 = 1;

#[derive(Debug)]
pub enum LoadError {
    Io(io::Error),
    BadMagic,
    UnsupportedVersion(u32),
    LayerCountMismatch { file: usize, model: usize },
    ConfigMismatch { layer: usize, field: &'static str, file: u32, model: u32 },
    DsaPresenceMismatch { layer: usize, file_has_dsa: bool, model_expects_dsa: bool },
    Corrupt(String),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "session file: {e}"),
            LoadError::BadMagic => write!(f, "session file: bad magic (not a rabbit KV session file)"),
            LoadError::UnsupportedVersion(v) => write!(f, "session file: unsupported version {v}"),
            LoadError::LayerCountMismatch { file, model } => {
                write!(f, "session file: {file} layers saved, model has {model}")
            }
            LoadError::ConfigMismatch { layer, field, file, model } => {
                write!(f, "session file: layer {layer} {field}={file}, model expects {model}")
            }
            LoadError::DsaPresenceMismatch { layer, file_has_dsa, model_expects_dsa } => {
                write!(
                    f,
                    "session file: layer {layer} has DSA data present={file_has_dsa}, model expects present={model_expects_dsa}"
                )
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

#[derive(Clone, PartialEq)]
struct LayerShape {
    kv_lora: u32,
    qk_rope: u32,
    hd: Option<u32>,
}

fn shape_of(state: &KvState) -> Vec<LayerShape> {
    state
        .layers()
        .iter()
        .zip(state.dsa())
        .map(|(kv, dsa)| LayerShape {
            kv_lora: kv.kv_lora() as u32,
            qk_rope: kv.qk_rope() as u32,
            hd: dsa.as_ref().map(|d| d.hd() as u32),
        })
        .collect()
}

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn decode_f32s(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn write_header(shape: &[LayerShape], w: &mut impl Write) -> io::Result<()> {
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&(shape.len() as u32).to_le_bytes())?;
    for l in shape {
        w.write_all(&l.kv_lora.to_le_bytes())?;
        w.write_all(&l.qk_rope.to_le_bytes())?;
        match l.hd {
            Some(hd) => {
                w.write_all(&[1u8])?;
                w.write_all(&hd.to_le_bytes())?;
            }
            None => w.write_all(&[0u8])?,
        }
    }
    Ok(())
}

fn read_header(r: &mut impl Read) -> Result<Vec<LayerShape>, LoadError> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(LoadError::BadMagic);
    }
    let mut u32buf = [0u8; 4];
    r.read_exact(&mut u32buf)?;
    let version = u32::from_le_bytes(u32buf);
    if version != VERSION {
        return Err(LoadError::UnsupportedVersion(version));
    }
    r.read_exact(&mut u32buf)?;
    let n_layers = u32::from_le_bytes(u32buf) as usize;
    let mut layers = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        r.read_exact(&mut u32buf)?;
        let kv_lora = u32::from_le_bytes(u32buf);
        r.read_exact(&mut u32buf)?;
        let qk_rope = u32::from_le_bytes(u32buf);
        let mut flag = [0u8; 1];
        r.read_exact(&mut flag)?;
        let hd = if flag[0] != 0 {
            r.read_exact(&mut u32buf)?;
            Some(u32::from_le_bytes(u32buf))
        } else {
            None
        };
        layers.push(LayerShape { kv_lora, qk_rope, hd });
    }
    Ok(layers)
}

fn write_record(state: &KvState, from_pos: usize, to_pos: usize, w: &mut impl Write) -> io::Result<()> {
    let n_new = (to_pos - from_pos) as u64;
    w.write_all(&(to_pos as u64).to_le_bytes())?;
    w.write_all(&n_new.to_le_bytes())?;
    for (kv, dsa) in state.layers().iter().zip(state.dsa()) {
        w.write_all(&f32_bytes(kv.l_range(from_pos, to_pos)))?;
        w.write_all(&f32_bytes(kv.r_range(from_pos, to_pos)))?;
        if let Some(d) = dsa {
            let cache_len = d.len();
            let seg_from = from_pos.min(cache_len);
            let seg_to = to_pos.min(cache_len);
            w.write_all(&((seg_to - seg_from) as u64).to_le_bytes())?;
            w.write_all(&f32_bytes(d.k_range(seg_from, seg_to)))?;
        }
    }
    Ok(())
}

/// Reads up to `buf.len()` bytes, filling as much as possible before EOF. Unlike `read_exact`,
/// a short read is not an error — the caller uses the returned count to detect a truncated
/// trailing record instead of getting an `UnexpectedEof`.
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

impl KvState {
    /// Appends everything new in `[from_pos, to_pos)` to `path` as one record, creating the
    /// file (and writing its header) if it doesn't exist yet. `from_pos` must be 0 exactly when
    /// creating a new file, and must NOT be 0 when appending to an existing one — both are
    /// caller bugs, not recoverable I/O conditions, so they're reported as `InvalidInput`.
    pub fn save(&self, from_pos: usize, to_pos: usize, path: &Path) -> io::Result<()> {
        if to_pos <= from_pos {
            return Ok(());
        }
        let exists = path.exists();
        if exists {
            if from_pos == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "kv_session: from_pos=0 but session file already exists",
                ));
            }
            let mut header_reader = BufReader::new(File::open(path)?);
            let file_shape = read_header(&mut header_reader)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            if file_shape != shape_of(self) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "kv_session: session file shape does not match the current session",
                ));
            }
        } else if from_pos != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "kv_session: from_pos must be 0 when creating a new session file",
            ));
        }

        let mut w = BufWriter::new(OpenOptions::new().create(true).append(true).open(path)?);
        if !exists {
            write_header(&shape_of(self), &mut w)?;
        }
        write_record(self, from_pos, to_pos, &mut w)?;
        w.flush()
    }

    /// Loads a session file, validating it against `model`'s shape (layer count, per-layer
    /// `kv_lora`/`qk_rope`/DSA presence) — a mismatch is a hard error, never a silent fallback
    /// to an empty session. Recovers from a crash mid-write of the final record by discarding
    /// it; returns the resulting `KvState` plus the absolute position it now holds.
    pub fn load(path: &Path, model: &Model) -> Result<(KvState, usize), LoadError> {
        let mut f = BufReader::new(File::open(path)?);
        let file_shape = read_header(&mut f)?;

        if file_shape.len() != model.layers.len() {
            return Err(LoadError::LayerCountMismatch { file: file_shape.len(), model: model.layers.len() });
        }
        let kv_lora = model.cfg.kv_lora as u32;
        let qk_rope = model.cfg.qk_rope as u32;
        let index_hd = model.cfg.index_hd as u32;
        for (i, (shape, layer)) in file_shape.iter().zip(model.layers.iter()).enumerate() {
            if shape.kv_lora != kv_lora {
                return Err(LoadError::ConfigMismatch { layer: i, field: "kv_lora", file: shape.kv_lora, model: kv_lora });
            }
            if shape.qk_rope != qk_rope {
                return Err(LoadError::ConfigMismatch { layer: i, field: "qk_rope", file: shape.qk_rope, model: qk_rope });
            }
            let model_expects_dsa = layer.dsa.is_some();
            let file_has_dsa = shape.hd.is_some();
            if file_has_dsa != model_expects_dsa {
                return Err(LoadError::DsaPresenceMismatch { layer: i, file_has_dsa, model_expects_dsa });
            }
            if let Some(hd) = shape.hd
                && hd != index_hd
            {
                return Err(LoadError::ConfigMismatch { layer: i, field: "index_hd", file: hd, model: index_hd });
            }
        }

        let n_layers = file_shape.len();
        let mut ls: Vec<Vec<f32>> = vec![Vec::new(); n_layers];
        let mut rs: Vec<Vec<f32>> = vec![Vec::new(); n_layers];
        let mut ks: Vec<Vec<f32>> = vec![Vec::new(); n_layers];
        let mut pos: u64 = 0;

        loop {
            let mut rec_header = [0u8; 16];
            let got = read_partial(&mut f, &mut rec_header)?;
            if got < 16 {
                break;
            }
            let pos_after = u64::from_le_bytes(rec_header[0..8].try_into().unwrap());
            let n_new = u64::from_le_bytes(rec_header[8..16].try_into().unwrap());

            let mut payloads: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::with_capacity(n_layers);
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
                let mut k_buf = Vec::new();
                if let Some(hd) = shape.hd {
                    let mut dsa_n_new_buf = [0u8; 8];
                    if read_partial(&mut f, &mut dsa_n_new_buf)? < 8 {
                        complete = false;
                        break;
                    }
                    let dsa_n_new = u64::from_le_bytes(dsa_n_new_buf);
                    let k_len = dsa_n_new as usize * hd as usize * 4;
                    k_buf = vec![0u8; k_len];
                    if read_partial(&mut f, &mut k_buf)? < k_len {
                        complete = false;
                        break;
                    }
                }
                payloads.push((decode_f32s(&l_buf), decode_f32s(&r_buf), decode_f32s(&k_buf)));
            }

            if !complete || payloads.len() != n_layers {
                break;
            }
            if pos_after != pos + n_new {
                return Err(LoadError::Corrupt(format!(
                    "record pos_after={pos_after} but expected {} (pos={pos}, n_new={n_new})",
                    pos + n_new
                )));
            }
            for (i, (l, r, k)) in payloads.into_iter().enumerate() {
                ls[i].extend(l);
                rs[i].extend(r);
                if file_shape[i].hd.is_some() {
                    ks[i].extend(k);
                }
            }
            pos = pos_after;
        }

        let mut layers = Vec::with_capacity(n_layers);
        let mut dsas = Vec::with_capacity(n_layers);
        for (i, shape) in file_shape.iter().enumerate() {
            let l = std::mem::take(&mut ls[i]);
            let r = std::mem::take(&mut rs[i]);
            layers.push(KvCache::from_raw(shape.kv_lora as usize, shape.qk_rope as usize, l, r));
            dsas.push(shape.hd.map(|hd| DsaCache::from_raw(hd as usize, std::mem::take(&mut ks[i]))));
        }
        Ok((KvState::from_raw(layers, dsas), pos as usize))
    }
}
