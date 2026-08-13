//! Persists a `qwen38::generate::KvState` across process restarts for `--chat --session <path>` —
//! the sibling of `crate::kv_session`'s (GLM-5.2) and `kimi_linear::kv_session`'s formats, with the
//! same two-file strategy and the same crash-consistency rule, adapted to Qwen's own two state
//! shapes.
//!
//! **Two files, one strategy each** (the split exists because the two state kinds grow differently):
//!
//! - `<path>` — an append-only log of the GQA layers' NEW rows per completed turn. A KV cache only
//!   ever grows by one K row and one V row per token per layer, so appending keeps total bytes
//!   written linear in conversation length instead of quadratic.
//! - `<path>.gdn` — the CURRENT Gated DeltaNet state (every GDN layer's 128 head matrices plus its
//!   conv FIFO), written to a temp file and `rename`d so it is replaced atomically. Appending would
//!   be pointless here: this state is FIXED size and mutated in place, so there is no "new rows"
//!   record to add — only another full copy. On the real checkpoint that's 69 layers x 128 heads x
//!   128x128 floats ≈ **578 MB**, which is exactly why it must not accumulate per turn.
//!
//! **Crash consistency:** `save` appends the KV record FIRST and replaces `.gdn` SECOND, so a crash
//! in between can only leave `.gdn`'s `pos` at or behind the KV log's furthest complete record —
//! never ahead. `load` therefore trusts `.gdn`'s `pos` as authoritative and stops replaying KV
//! records once their `pos_after` would exceed it, discarding newer orphans exactly as a truncated
//! trailing record is discarded.
//!
//! All integers little-endian, all floats f32 little-endian.

use crate::qwen38::attention::KvCache;
use crate::qwen38::generate::{GdnLayerState, KvState, LayerState};
use crate::qwen38::model::Model;
use crate::kimi_linear::kda::KdaState;
use crate::kimi_linear::short_conv::ShortConvState;
use std::ffi::OsString;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const KV_MAGIC: &[u8; 8] = b"RBTQWKV1";
const KV_VERSION: u32 = 1;
const GDN_MAGIC: &[u8; 8] = b"RBTQWGD1";
const GDN_VERSION: u32 = 1;

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
            LoadError::BadMagic => write!(f, "session file: not a rabbit Qwen 3.8 session file"),
            LoadError::UnsupportedVersion(v) => write!(f, "session file: unsupported format version {v}"),
            LoadError::LayerCountMismatch { file, model, what } => {
                write!(f, "session file has {file} {what} layers but this model has {model}")
            }
            LoadError::ConfigMismatch { layer, field, file, model } => {
                write!(f, "session file's {what} disagrees with the model at layer {layer}: {file} vs {model}", what = field)
            }
            LoadError::Corrupt(msg) => write!(f, "session file: {msg}"),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<io::Error> for LoadError {
    fn from(e: io::Error) -> Self {
        LoadError::Io(e)
    }
}

/// Per-GQA-layer shape recorded in the KV log's header, so a session can never be replayed into a
/// model whose attention geometry changed.
#[derive(Clone, Copy, PartialEq)]
struct KvShape {
    n_kv_heads: u32,
    head_dim: u32,
}

/// Per-GDN-layer shape recorded in the snapshot's header.
#[derive(Clone, Copy, PartialEq)]
struct GdnShape {
    n_heads: u32,
    d_k: u32,
    d_v: u32,
    kernel: u32,
    conv_dim: u32,
}

fn gdn_path(path: &Path) -> PathBuf {
    let mut name = OsString::from(path.file_name().unwrap_or_default());
    name.push(".gdn");
    path.with_file_name(name)
}

fn kv_shapes(state: &KvState) -> Vec<KvShape> {
    state
        .layers()
        .iter()
        .filter_map(|l| match l {
            LayerState::Attn(c) => Some(KvShape { n_kv_heads: c.n_kv_heads() as u32, head_dim: c.head_dim() as u32 }),
            LayerState::Gdn(_) => None,
        })
        .collect()
}

fn gdn_shapes(state: &KvState) -> Vec<GdnShape> {
    state
        .layers()
        .iter()
        .filter_map(|l| match l {
            LayerState::Gdn(g) => Some(GdnShape {
                n_heads: g.heads.len() as u32,
                d_k: g.heads.first().map(|h| h.d_k()).unwrap_or(0) as u32,
                d_v: g.heads.first().map(|h| h.d_v()).unwrap_or(0) as u32,
                kernel: g.conv.kernel() as u32,
                conv_dim: g.conv.d_inner() as u32,
            }),
            LayerState::Attn(_) => None,
        })
        .collect()
}

fn model_kv_shapes(model: &Model) -> Vec<KvShape> {
    let cfg = &model.cfg;
    model
        .layers
        .iter()
        .filter(|l| !l.mixer.is_gdn())
        .map(|_| KvShape { n_kv_heads: cfg.n_kv_heads as u32, head_dim: cfg.head_dim as u32 })
        .collect()
}

fn model_gdn_shapes(model: &Model) -> Vec<GdnShape> {
    let cfg = &model.cfg;
    let key_dim = (cfg.lin_key_heads * cfg.lin_key_head_dim) as u32;
    let value_dim = (cfg.lin_value_heads * cfg.lin_value_head_dim) as u32;
    model
        .layers
        .iter()
        .filter(|l| l.mixer.is_gdn())
        .map(|_| GdnShape {
            n_heads: cfg.lin_value_heads as u32,
            d_k: cfg.lin_key_head_dim as u32,
            d_v: cfg.lin_value_head_dim as u32,
            kernel: cfg.conv_kernel as u32,
            conv_dim: 2 * key_dim + value_dim,
        })
        .collect()
}

fn encode_f32s(v: &[f32], out: &mut Vec<u8>) {
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

fn decode_f32s(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

impl KvState {
    /// Appends the rows this turn added (`from_pos..to_pos`) to the KV log, then atomically replaces
    /// the GDN snapshot — in that order, per this module's crash-consistency rule.
    pub fn save(&self, from_pos: usize, to_pos: usize, path: &Path) -> io::Result<()> {
        if to_pos <= from_pos {
            return Ok(());
        }
        let kv = kv_shapes(self);
        if !kv.is_empty() {
            save_kv_log(&kv, self, from_pos, to_pos, path)?;
        }
        let gdn = gdn_shapes(self);
        if !gdn.is_empty() {
            save_gdn_snapshot_atomic(&gdn, self, to_pos, path)?;
        }
        Ok(())
    }

    /// Loads a previously-saved session, validating both files against `model`. A shape mismatch is a
    /// hard error, never a silent fall back to an empty session.
    pub fn load(path: &Path, model: &Model) -> Result<(KvState, usize), LoadError> {
        let kv_model = model_kv_shapes(model);
        let gdn_model = model_gdn_shapes(model);

        // GDN snapshot first: its `pos` is authoritative for how far the session really got.
        let (gdn_pos, mut gdn_layers): (Option<u64>, Vec<GdnLayerState>) = if gdn_model.is_empty() {
            (None, Vec::new())
        } else {
            let mut f = BufReader::new(File::open(gdn_path(path))?);
            let (pos, file_shapes) = read_gdn_header(&mut f)?;
            if file_shapes.len() != gdn_model.len() {
                return Err(LoadError::LayerCountMismatch { file: file_shapes.len(), model: gdn_model.len(), what: "GDN" });
            }
            let mut layers = Vec::with_capacity(file_shapes.len());
            for (i, (fs, ms)) in file_shapes.iter().zip(&gdn_model).enumerate() {
                check(i, "linear_num_value_heads", fs.n_heads, ms.n_heads)?;
                check(i, "linear_key_head_dim", fs.d_k, ms.d_k)?;
                check(i, "linear_value_head_dim", fs.d_v, ms.d_v)?;
                check(i, "linear_conv_kernel_dim", fs.kernel, ms.kernel)?;
                check(i, "conv_dim", fs.conv_dim, ms.conv_dim)?;

                let (d_k, d_v) = (fs.d_k as usize, fs.d_v as usize);
                let mut heads = Vec::with_capacity(fs.n_heads as usize);
                for _ in 0..fs.n_heads {
                    let mut buf = vec![0u8; d_k * d_v * 4];
                    f.read_exact(&mut buf)?;
                    heads.push(KdaState::from_raw(d_k, d_v, decode_f32s(&buf)));
                }
                let hist = (fs.kernel as usize).saturating_sub(1) * fs.conv_dim as usize;
                let mut buf = vec![0u8; hist * 4];
                f.read_exact(&mut buf)?;
                let conv = ShortConvState::from_raw(fs.conv_dim as usize, fs.kernel as usize, decode_f32s(&buf));
                layers.push(GdnLayerState { heads, conv });
            }
            (Some(pos), layers)
        };

        let (kv_pos, mut kv_caches): (u64, Vec<KvCache>) = if kv_model.is_empty() {
            (gdn_pos.unwrap_or(0), Vec::new())
        } else {
            read_kv_log(path, &kv_model, gdn_pos)?
        };

        // Rebuild in model order, drawing from the two per-kind lists.
        let mut gdn_iter = gdn_layers.drain(..);
        let mut kv_iter = kv_caches.drain(..);
        let layers: Vec<LayerState> = model
            .layers
            .iter()
            .map(|l| {
                if l.mixer.is_gdn() {
                    LayerState::Gdn(gdn_iter.next().expect("GDN layer count was validated above"))
                } else {
                    LayerState::Attn(kv_iter.next().expect("attention layer count was validated above"))
                }
            })
            .collect();
        Ok((KvState::from_raw(layers), kv_pos as usize))
    }
}

fn check(layer: usize, field: &'static str, file: u32, model: u32) -> Result<(), LoadError> {
    if file != model { Err(LoadError::ConfigMismatch { layer, field, file, model }) } else { Ok(()) }
}

fn save_kv_log(shapes: &[KvShape], state: &KvState, from_pos: usize, to_pos: usize, path: &Path) -> io::Result<()> {
    let fresh = !path.exists() || fs::metadata(path)?.len() == 0;
    let mut f = BufWriter::new(OpenOptions::new().create(true).append(true).open(path)?);
    if fresh {
        f.write_all(KV_MAGIC)?;
        f.write_all(&KV_VERSION.to_le_bytes())?;
        f.write_all(&(shapes.len() as u32).to_le_bytes())?;
        for s in shapes {
            f.write_all(&s.n_kv_heads.to_le_bytes())?;
            f.write_all(&s.head_dim.to_le_bytes())?;
        }
    }

    let mut rec = Vec::new();
    rec.extend_from_slice(&(from_pos as u64).to_le_bytes());
    rec.extend_from_slice(&(to_pos as u64).to_le_bytes());
    for l in state.layers() {
        if let LayerState::Attn(c) = l {
            let (k, v) = c.rows(from_pos, to_pos);
            encode_f32s(k, &mut rec);
            encode_f32s(v, &mut rec);
        }
    }
    f.write_all(&rec)?;
    f.flush()
}

fn save_gdn_snapshot_atomic(shapes: &[GdnShape], state: &KvState, pos: usize, path: &Path) -> io::Result<()> {
    let final_path = gdn_path(path);
    let tmp_path = {
        let mut name = OsString::from(final_path.file_name().unwrap_or_default());
        name.push(".tmp");
        final_path.with_file_name(name)
    };

    {
        let mut f = BufWriter::new(File::create(&tmp_path)?);
        f.write_all(GDN_MAGIC)?;
        f.write_all(&GDN_VERSION.to_le_bytes())?;
        f.write_all(&(pos as u64).to_le_bytes())?;
        f.write_all(&(shapes.len() as u32).to_le_bytes())?;
        for s in shapes {
            for field in [s.n_heads, s.d_k, s.d_v, s.kernel, s.conv_dim] {
                f.write_all(&field.to_le_bytes())?;
            }
        }
        let mut buf = Vec::new();
        for l in state.layers() {
            if let LayerState::Gdn(g) = l {
                buf.clear();
                for h in &g.heads {
                    encode_f32s(h.raw(), &mut buf);
                }
                encode_f32s(g.conv.history(), &mut buf);
                f.write_all(&buf)?;
            }
        }
        f.flush()?;
    }
    fs::rename(&tmp_path, &final_path)
}

fn read_kv_log(path: &Path, model_shapes: &[KvShape], gdn_pos: Option<u64>) -> Result<(u64, Vec<KvCache>), LoadError> {
    let mut f = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic != KV_MAGIC {
        return Err(LoadError::BadMagic);
    }
    let version = read_u32(&mut f)?;
    if version != KV_VERSION {
        return Err(LoadError::UnsupportedVersion(version));
    }
    let n_layers = read_u32(&mut f)? as usize;
    if n_layers != model_shapes.len() {
        return Err(LoadError::LayerCountMismatch { file: n_layers, model: model_shapes.len(), what: "attention" });
    }
    let mut file_shapes = Vec::with_capacity(n_layers);
    for (i, expected) in model_shapes.iter().enumerate() {
        let s = KvShape { n_kv_heads: read_u32(&mut f)?, head_dim: read_u32(&mut f)? };
        check(i, "num_key_value_heads", s.n_kv_heads, expected.n_kv_heads)?;
        check(i, "head_dim", s.head_dim, expected.head_dim)?;
        file_shapes.push(s);
    }

    let row_floats: usize = file_shapes.iter().map(|s| (s.n_kv_heads * s.head_dim) as usize).sum();
    let mut caches: Vec<Vec<f32>> = vec![Vec::new(); n_layers]; // K rows, per layer
    let mut values: Vec<Vec<f32>> = vec![Vec::new(); n_layers];
    let mut pos = 0u64;

    loop {
        let mut head = [0u8; 16];
        match f.read_exact(&mut head) {
            Ok(()) => {}
            // a clean end of file, or a torn record header from a crash mid-append: both mean
            // "nothing more to replay", same as `crate::kv_session`'s own truncation handling
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(LoadError::Io(e)),
        }
        let from = u64::from_le_bytes(head[..8].try_into().unwrap());
        let to = u64::from_le_bytes(head[8..].try_into().unwrap());
        if to <= from {
            return Err(LoadError::Corrupt(format!("record with to_pos {to} <= from_pos {from}")));
        }
        if from != pos {
            return Err(LoadError::Corrupt(format!("record starts at {from} but {pos} rows are loaded")));
        }
        // Orphan past the authoritative GDN position (crash between the two writes): stop here and
        // keep what's consistent, rather than replaying rows the recurrent state never saw.
        if let Some(gp) = gdn_pos
            && to > gp
        {
            break;
        }

        let n = (to - from) as usize;
        let mut buf = vec![0u8; n * row_floats * 2 * 4];
        match f.read_exact(&mut buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break, // truncated trailing record
            Err(e) => return Err(LoadError::Io(e)),
        }
        let mut off = 0usize;
        for (li, s) in file_shapes.iter().enumerate() {
            let per_layer = n * (s.n_kv_heads * s.head_dim) as usize * 4;
            caches[li].extend_from_slice(&decode_f32s(&buf[off..off + per_layer]));
            off += per_layer;
            values[li].extend_from_slice(&decode_f32s(&buf[off..off + per_layer]));
            off += per_layer;
        }
        pos = to;
    }

    let out = file_shapes
        .iter()
        .zip(caches.into_iter().zip(values))
        .map(|(s, (k, v))| KvCache::from_raw(s.n_kv_heads as usize, s.head_dim as usize, k, v))
        .collect();
    Ok((pos, out))
}

fn read_gdn_header(f: &mut BufReader<File>) -> Result<(u64, Vec<GdnShape>), LoadError> {
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic != GDN_MAGIC {
        return Err(LoadError::BadMagic);
    }
    let version = read_u32(f)?;
    if version != GDN_VERSION {
        return Err(LoadError::UnsupportedVersion(version));
    }
    let mut pos_bytes = [0u8; 8];
    f.read_exact(&mut pos_bytes)?;
    let pos = u64::from_le_bytes(pos_bytes);
    let n = read_u32(f)? as usize;
    let mut shapes = Vec::with_capacity(n);
    for _ in 0..n {
        shapes.push(GdnShape {
            n_heads: read_u32(f)?,
            d_k: read_u32(f)?,
            d_v: read_u32(f)?,
            kernel: read_u32(f)?,
            conv_dim: read_u32(f)?,
        });
    }
    Ok((pos, shapes))
}

fn read_u32(f: &mut BufReader<File>) -> Result<u32, LoadError> {
    let mut b = [0u8; 4];
    f.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen38::generate::{ExpertCaches, step};
    use crate::qwen38::model::tests::{TempDir, write_tiny_checkpoint};
    use crate::safetensors::Shards;

    fn setup(name: &str) -> (TempDir, Model, Shards) {
        let dir = TempDir::new(name);
        write_tiny_checkpoint(&dir.0);
        let model = Model::load(&dir.0, 16, 4).expect("fixture must load");
        let shards = Shards::open(&dir.0).expect("fixture shards must open");
        (dir, model, shards)
    }

    /// The property that actually matters: a session restored from disk must continue the
    /// conversation identically to one that never left memory — same next-token logits, bit for bit
    /// through both state kinds (GQA rows AND the GDN recurrence).
    #[test]
    fn a_restored_session_continues_identically_to_the_live_one() {
        let (dir, model, shards) = setup("rabbit_test_qwen38_session_roundtrip");
        let session = dir.0.join("session.bin");
        let mut caches = ExpertCaches::new(&model, 4);

        let mut live = KvState::new(&model);
        step(&model, &shards, &mut caches, &mut live, &[1, 2, 3], 0).unwrap();
        live.save(0, 3, &session).unwrap();

        let (mut restored, pos) = KvState::load(&session, &model).unwrap();
        assert_eq!(pos, 3, "three tokens were saved");

        let from_live = step(&model, &shards, &mut caches, &mut live, &[4], 3).unwrap();
        let from_restored = step(&model, &shards, &mut caches, &mut restored, &[4], 3).unwrap();
        assert_eq!(from_live, from_restored, "a restored session must produce identical logits");
    }

    /// Two turns: the KV log appends (it must not be rewritten from scratch) while the GDN snapshot is
    /// replaced in place, so its size stays constant.
    #[test]
    fn the_kv_log_grows_per_turn_while_the_gdn_snapshot_stays_the_same_size() {
        let (dir, model, shards) = setup("rabbit_test_qwen38_session_append");
        let session = dir.0.join("session.bin");
        let mut caches = ExpertCaches::new(&model, 4);
        let mut kv = KvState::new(&model);

        step(&model, &shards, &mut caches, &mut kv, &[1, 2], 0).unwrap();
        kv.save(0, 2, &session).unwrap();
        let kv_after_first = fs::metadata(&session).unwrap().len();
        let gdn_after_first = fs::metadata(gdn_path(&session)).unwrap().len();

        step(&model, &shards, &mut caches, &mut kv, &[3, 4], 2).unwrap();
        kv.save(2, 4, &session).unwrap();
        let kv_after_second = fs::metadata(&session).unwrap().len();
        let gdn_after_second = fs::metadata(gdn_path(&session)).unwrap().len();

        assert!(kv_after_second > kv_after_first, "the KV log must append the second turn's rows");
        assert_eq!(gdn_after_second, gdn_after_first, "the GDN snapshot must be replaced, not appended to");

        let (_, pos) = KvState::load(&session, &model).unwrap();
        assert_eq!(pos, 4, "both turns must replay");
    }

    /// A crash mid-append leaves a torn trailing record. Loading must keep the last complete turn
    /// instead of failing or replaying garbage.
    #[test]
    fn a_truncated_trailing_record_is_discarded() {
        let (dir, model, shards) = setup("rabbit_test_qwen38_session_truncated");
        let session = dir.0.join("session.bin");
        let mut caches = ExpertCaches::new(&model, 4);
        let mut kv = KvState::new(&model);

        step(&model, &shards, &mut caches, &mut kv, &[1, 2], 0).unwrap();
        kv.save(0, 2, &session).unwrap();
        let complete = fs::metadata(&session).unwrap().len();

        step(&model, &shards, &mut caches, &mut kv, &[3], 2).unwrap();
        kv.save(2, 3, &session).unwrap();
        // chop the second record in half, mid-payload
        let full = fs::metadata(&session).unwrap().len();
        let torn = complete + (full - complete) / 2;
        let f = OpenOptions::new().write(true).open(&session).unwrap();
        f.set_len(torn).unwrap();
        drop(f);

        let (_, pos) = KvState::load(&session, &model).unwrap();
        assert_eq!(pos, 2, "the torn turn is dropped, the complete one survives");
    }

    /// A GDN snapshot from a crash BEFORE the second turn's rename is behind the KV log; the log's
    /// newer rows are orphans and must not be replayed (they'd pair fresh KV rows with stale
    /// recurrent state).
    #[test]
    fn kv_rows_newer_than_the_gdn_snapshot_are_discarded() {
        let (dir, model, shards) = setup("rabbit_test_qwen38_session_orphan");
        let session = dir.0.join("session.bin");
        let mut caches = ExpertCaches::new(&model, 4);
        let mut kv = KvState::new(&model);

        step(&model, &shards, &mut caches, &mut kv, &[1, 2], 0).unwrap();
        kv.save(0, 2, &session).unwrap();
        let gdn_at_two = fs::read(gdn_path(&session)).unwrap();

        step(&model, &shards, &mut caches, &mut kv, &[3], 2).unwrap();
        kv.save(2, 3, &session).unwrap();
        // simulate the crash: restore the older snapshot, keeping the newer KV log
        fs::write(gdn_path(&session), &gdn_at_two).unwrap();

        let (_, pos) = KvState::load(&session, &model).unwrap();
        assert_eq!(pos, 2, "the KV log must be replayed only as far as the GDN snapshot reached");
    }

    #[test]
    fn a_session_from_a_different_shape_is_rejected() {
        let (dir, model, shards) = setup("rabbit_test_qwen38_session_mismatch");
        let session = dir.0.join("session.bin");
        let mut caches = ExpertCaches::new(&model, 4);
        let mut kv = KvState::new(&model);
        step(&model, &shards, &mut caches, &mut kv, &[1], 0).unwrap();
        kv.save(0, 1, &session).unwrap();

        // claim a different head_dim in the KV log's header (bytes 16.. are the per-layer shapes)
        let mut raw = fs::read(&session).unwrap();
        raw[20] = raw[20].wrapping_add(1);
        fs::write(&session, &raw).unwrap();
        match KvState::load(&session, &model) {
            Err(LoadError::ConfigMismatch { field, .. }) => assert_eq!(field, "head_dim"),
            other => panic!("expected ConfigMismatch, got {:?}", other.map(|_| ())),
        }

        // and a wrong magic is rejected outright
        fs::write(&session, b"NOTRABBIT-and-then-some").unwrap();
        assert!(matches!(KvState::load(&session, &model), Err(LoadError::BadMagic)));
    }

    #[test]
    fn saving_a_turn_that_added_no_tokens_writes_nothing() {
        let (dir, model, _shards) = setup("rabbit_test_qwen38_session_empty");
        let session = dir.0.join("session.bin");
        let kv = KvState::new(&model);
        kv.save(5, 5, &session).unwrap();
        assert!(!session.exists(), "an empty turn must not create a session file");
    }
}
