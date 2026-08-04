//! Port of `st.h` — indexes and reads tensors from one or more `.safetensors` shards.
//!
//! Uses `pread` (via `FileExt::read_at`), never `mmap`, BY DEFAULT: mmap leaves pages resident
//! in the process, which corrupts peak-RSS accounting — the streaming architecture needs to know
//! exactly how much RAM it holds so it never trips the OOM killer. See `rabbit-plan.md`.
//!
//! The one exception is the opt-in `--mmap-experts` experiment (`Shards::mmap_shard`/
//! `tensor_mmap_range`/`madvise_willneed` below, off by default): a scoped alternative load path
//! for `expert_cache.rs`'s ordinary LRU miss path, testing the hypothesis (see
//! `PERFORMANCE_KIMI_K3.md`) that this project's OWN resident `Vec<u8>` buffers double-cache
//! against the OS page cache for the same physical pages. Letting the kernel own the one copy
//! (via `mmap`) instead of also holding a second, rabbit-owned copy is exactly the tradeoff that
//! makes the RSS accounting above inexact under this flag — `expert_cache.rs`'s
//! `safe_mxfp4_capacity` doc explains why that's still safe (a conservative upper bound, not
//! wrong in the unsafe direction).

use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use std::fs::{self, File};
use std::io;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Bf16,
    F16,
    F32,
    /// raw bytes: our own quantized int4/int8/int2 containers.
    U8,
    /// 8-bit float, e4m3fn (1 sign, 4 exponent bits [bias 7], 3 mantissa bits, no infinities —
    /// `S.1111.111` is NaN instead). The real GLM-5.2-FP8 release stores weights this way,
    /// paired with a `{name}_scale_inv` block-scale tensor (128x128 blocks) — see
    /// `read_f32`'s doc for how that pairing is resolved.
    F8E4M3,
}

impl DType {
    fn from_str(s: &str) -> Result<DType, SafetensorsError> {
        match s {
            "BF16" => Ok(DType::Bf16),
            "F16" => Ok(DType::F16),
            "F32" => Ok(DType::F32),
            "U8" | "I8" => Ok(DType::U8),
            "F8_E4M3" | "float8_e4m3fn" => Ok(DType::F8E4M3),
            other => Err(SafetensorsError::UnknownDtype(other.to_string())),
        }
    }

    fn byte_size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::Bf16 | DType::F16 => 2,
            DType::U8 | DType::F8E4M3 => 1,
        }
    }
}

/// Decodes one e4m3fn byte to f32 — see `DType::F8E4M3`'s doc for the bit layout. Subnormals
/// (`exp==0`): `mantissa/8 * 2^-6`. Normals: `(1+mantissa/8) * 2^(exp-7)`, max finite `448.0`
/// at `exp=15,mantissa=6` (mantissa=7 there is reserved for NaN, the "fn" — finite, no
/// infinity — variant's one deviation from a "denser" plain e4m3 that would go up to 480).
pub(crate) fn f8e4m3_to_f32(b: u8) -> f32 {
    let sign = (b >> 7) & 1;
    let exp = (b >> 3) & 0xF;
    let mant = (b & 0x7) as f32;
    let magnitude = if exp == 0 {
        mant * 2f32.powi(-9)
    } else if exp == 0xF && b & 0x7 == 0x7 {
        return f32::NAN;
    } else {
        (1.0 + mant / 8.0) * 2f32.powi(exp as i32 - 7)
    };
    if sign == 1 { -magnitude } else { magnitude }
}

/// Applies FP8 block-scale dequant to already-read raw FP8 bytes and an already-read scale
/// vector — the math half of `Shards::read_f32`'s automatic FP8 handling, factored out for
/// callers that fetch the raw bytes themselves (e.g. `expert_cache.rs`'s `io_uring` batch
/// loader, which reads tensor bytes via its own ring instead of `Shards::read_f32`).
/// `scale_name` is only used to label a length-mismatch error.
pub fn dequant_fp8_blockscale(raw: &[u8], rows: u64, cols: u64, scale: &[f32], scale_name: &str) -> Result<Vec<f32>, SafetensorsError> {
    const BLOCK: u64 = 128;
    let sc_cols = cols.div_ceil(BLOCK);
    let expected_sc_len = rows.div_ceil(BLOCK) * sc_cols;
    if scale.len() as u64 != expected_sc_len {
        let msg = format!("{scale_name}: expected {expected_sc_len} block scales for a [{rows},{cols}] tensor, got {}", scale.len());
        return Err(SafetensorsError::BadHeader(msg));
    }
    let mut out = Vec::with_capacity(raw.len());
    for r in 0..rows {
        let block_row = r / BLOCK;
        for c in 0..cols {
            let block_col = c / BLOCK;
            let s = scale[(block_row * sc_cols + block_col) as usize];
            out.push(f8e4m3_to_f32(raw[(r * cols + c) as usize]) * s);
        }
    }
    Ok(out)
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    file_index: usize,
    /// absolute byte offset of the tensor's data within its shard file.
    offset: u64,
    pub nbytes: u64,
    pub dtype: DType,
    pub numel: u64,
    /// as declared in the safetensors header — needed (not just `numel`) to apply FP8 block
    /// scale, which depends on the 2D block structure, not just the flattened element count.
    pub shape: Vec<u64>,
}

#[derive(Debug)]
pub enum SafetensorsError {
    Io(io::Error),
    Json(serde_json::Error),
    MissingTensor(String),
    BadHeader(String),
    UnknownDtype(String),
}

impl fmt::Display for SafetensorsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SafetensorsError::Io(e) => write!(f, "io: {e}"),
            SafetensorsError::Json(e) => write!(f, "header json: {e}"),
            SafetensorsError::MissingTensor(n) => write!(f, "missing tensor: {n}"),
            SafetensorsError::BadHeader(m) => write!(f, "bad safetensors header: {m}"),
            SafetensorsError::UnknownDtype(d) => write!(f, "unhandled dtype: {d}"),
        }
    }
}

impl std::error::Error for SafetensorsError {}

impl From<io::Error> for SafetensorsError {
    fn from(e: io::Error) -> Self {
        SafetensorsError::Io(e)
    }
}

impl From<serde_json::Error> for SafetensorsError {
    fn from(e: serde_json::Error) -> Self {
        SafetensorsError::Json(e)
    }
}

pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let mut exp = ((h >> 10) & 0x1F) as i32;
    let mut man = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            exp = 127 - 15 + 1;
            while man & 0x400 == 0 {
                man <<= 1;
                exp -= 1;
            }
            man &= 0x3FF;
            sign | ((exp as u32) << 23) | (man << 13)
        }
    } else if exp == 0x1F {
        sign | 0x7F80_0000 | (man << 13)
    } else {
        sign | (((exp - 15 + 127) as u32) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

fn read_exact_at(file: &File, mut offset: u64, buf: &mut [u8]) -> io::Result<()> {
    let mut done = 0;
    while done < buf.len() {
        let n = file.read_at(&mut buf[done..], offset)?;
        if n == 0 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short read"));
        }
        done += n;
        offset += n as u64;
    }
    Ok(())
}

fn decode_floats(raw: &[u8], dtype: DType, out: &mut Vec<f32>) {
    match dtype {
        DType::F32 => {
            out.extend(raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())));
        }
        DType::Bf16 => {
            out.extend(raw.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes(c.try_into().unwrap()))));
        }
        DType::F16 => {
            out.extend(raw.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap()))));
        }
        // plain per-element decode, no block scale applied — `read_f32` special-cases
        // F8E4M3 separately (it needs the tensor's shape and a companion scale tensor,
        // neither of which this raw-bytes-in function has access to).
        DType::F8E4M3 => out.extend(raw.iter().map(|&b| f8e4m3_to_f32(b))),
        DType::U8 => unreachable!("decode_floats called on a raw U8 container"),
    }
}

#[derive(Debug)]
pub struct Shards {
    files: Vec<File>,
    tensors: Vec<Tensor>,
    index: HashMap<String, usize>,
    /// Lazily-created, cached mmap handle per shard file (same indexing as `files`) — only ever
    /// populated by the opt-in `--mmap-experts` path (`mmap_shard` below); the default `pread`
    /// path never touches this. `OnceLock` gives cheap, thread-safe, at-most-once mapping without
    /// needing `&mut self` (`Shards` is normally shared read-only across worker threads via
    /// `Arc<Shards>`).
    mmaps: Vec<OnceLock<Arc<memmap2::Mmap>>>,
}

impl Shards {
    /// indexes every `*.safetensors` file in `dir` (sorted by filename, matching how the C
    /// engine orders `model-0000N-of-...` shards).
    pub fn open(dir: &Path) -> Result<Shards, SafetensorsError> {
        Self::open_multi(std::slice::from_ref(&dir.to_path_buf()))
    }

    /// Same as `open`, but indexes `*.safetensors` files from MULTIPLE directories into one
    /// unified `Shards` — lets a checkpoint's shards be split across separate drives/mounts
    /// (e.g. a second NVMe added for capacity or read-bandwidth headroom, without needing an
    /// OS-level RAID array), matching the spirit of colibrì's `COLI_MODEL_DIRS` (see the
    /// project's colibri-backlog memory / `ROADMAP.md`'s hardware section for why this is safer
    /// than reshaping an in-use root partition into a striped array). `config.json`/tokenizer
    /// files are NOT looked for across these directories — those still come from the primary
    /// `--model` directory alone; this only widens where `.safetensors` shards may live. Sorted
    /// by filename (not full path) so the natural `model-0000N-of-...` shard order is preserved
    /// regardless of which directory a given shard physically lives in — a tensor name appearing
    /// in more than one directory is a checkpoint-authoring error, not something this function
    /// tries to resolve, so the later directory silently wins (matches `HashMap::insert`'s own
    /// last-write-wins semantics, same as if all shards had been dumped in one directory with a
    /// clashing name).
    pub fn open_multi(dirs: &[PathBuf]) -> Result<Shards, SafetensorsError> {
        let mut paths: Vec<PathBuf> = Vec::new();
        for dir in dirs {
            paths.extend(
                fs::read_dir(dir)?
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("safetensors")),
            );
        }
        paths.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

        let mut files = Vec::with_capacity(paths.len());
        let mut tensors = Vec::new();
        let mut index = HashMap::new();

        for path in paths {
            let file = File::open(&path)?;
            let file_len = file.metadata()?.len();
            let mut hlen_buf = [0u8; 8];
            read_exact_at(&file, 0, &mut hlen_buf)?;
            let hlen = u64::from_le_bytes(hlen_buf);
            // the header can't be bigger than the file it lives in — reject up front instead of
            // trusting a corrupt/malicious `hlen` into a multi-exabyte allocation attempt.
            if hlen > file_len {
                return Err(SafetensorsError::BadHeader(format!(
                    "{}: header length {hlen} exceeds file size {file_len}",
                    path.display()
                )));
            }
            let mut header_buf = vec![0u8; hlen as usize];
            read_exact_at(&file, 8, &mut header_buf)?;
            let header: Value = serde_json::from_slice(&header_buf)?;
            let data_start = 8 + hlen;

            let obj = header
                .as_object()
                .ok_or_else(|| SafetensorsError::BadHeader("top-level value is not an object".into()))?;

            let file_index = files.len();
            for (name, meta) in obj {
                if name == "__metadata__" {
                    continue;
                }
                let dtype = DType::from_str(
                    meta.get("dtype")
                        .and_then(Value::as_str)
                        .ok_or_else(|| SafetensorsError::BadHeader(format!("{name}: missing dtype")))?,
                )?;
                let offsets = meta
                    .get("data_offsets")
                    .and_then(Value::as_array)
                    .ok_or_else(|| SafetensorsError::BadHeader(format!("{name}: missing data_offsets")))?;
                if offsets.len() != 2 {
                    return Err(SafetensorsError::BadHeader(format!("{name}: data_offsets must have 2 elements")));
                }
                let a0 = offsets[0]
                    .as_u64()
                    .ok_or_else(|| SafetensorsError::BadHeader(format!("{name}: data_offsets[0] is not a u64")))?;
                let b0 = offsets[1]
                    .as_u64()
                    .ok_or_else(|| SafetensorsError::BadHeader(format!("{name}: data_offsets[1] is not a u64")))?;
                if b0 < a0 {
                    return Err(SafetensorsError::BadHeader(format!("{name}: data_offsets end {b0} precedes start {a0}")));
                }
                if data_start + b0 > file_len {
                    return Err(SafetensorsError::BadHeader(format!(
                        "{name}: data_offsets end {b0} extends past end of file ({} bytes available)",
                        file_len.saturating_sub(data_start)
                    )));
                }
                let shape = meta
                    .get("shape")
                    .and_then(Value::as_array)
                    .ok_or_else(|| SafetensorsError::BadHeader(format!("{name}: missing shape")))?;
                let shape: Vec<u64> = shape.iter().filter_map(Value::as_u64).collect();
                // product() of an empty iterator is 1 (multiplicative identity), which is
                // also the correct numel for a 0-dim (scalar) tensor.
                let numel = shape.iter().product::<u64>();

                tensors.push(Tensor {
                    name: name.clone(),
                    file_index,
                    offset: data_start + a0,
                    nbytes: b0 - a0,
                    dtype,
                    numel,
                    shape,
                });
                index.insert(name.clone(), tensors.len() - 1);
            }
            files.push(file);
        }

        let mmaps = files.iter().map(|_| OnceLock::new()).collect();
        Ok(Shards { files, tensors, index, mmaps })
    }

    /// Every indexed tensor across every shard in this `Shards` — the converter's
    /// (`glm52::convert`) equivalent of Python's `safe_open(path).keys()`/`.values()`, used to
    /// iterate a whole checkpoint (or, when `open` is pointed at a directory holding exactly one
    /// downloaded shard, that one file's own tensor set — the disk-safe conversion workflow's
    /// natural unit).
    pub fn tensors(&self) -> &[Tensor] {
        &self.tensors
    }

    pub fn find(&self, name: &str) -> Option<&Tensor> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// tells the kernel to start reading the tensor's pages in the background. No-op if the
    /// tensor doesn't exist.
    pub fn prefetch(&self, name: &str) {
        if let Some(t) = self.find(name) {
            let file = &self.files[t.file_index];
            unsafe {
                libc::posix_fadvise(
                    file.as_raw_fd(),
                    t.offset as libc::off_t,
                    t.nbytes as libc::off_t,
                    libc::POSIX_FADV_WILLNEED,
                );
            }
        }
    }

    fn fadvise_dontneed(&self, file_index: usize, offset: u64, nbytes: u64) {
        let file = &self.files[file_index];
        unsafe {
            libc::posix_fadvise(
                file.as_raw_fd(),
                offset as libc::off_t,
                nbytes as libc::off_t,
                libc::POSIX_FADV_DONTNEED,
            );
        }
    }

    /// reads a tensor and converts it to f32 regardless of its on-disk dtype. `F8E4M3`
    /// tensors are additionally dequantized with their companion `{name}_scale_inv`
    /// block-scale tensor (128x128 blocks) — the real GLM-5.2-FP8 release's own format, so
    /// every caller of `read_f32` (`model.rs`'s `qt_load`, `expert_cache.rs`'s sequential
    /// loader, ...) can load a raw FP8 checkpoint with zero changes of their own. `drop_cache
    /// = true` advises the kernel to discard the pages afterwards (streaming experts: we
    /// never want to reread stale ones from cache).
    pub fn read_f32(&self, name: &str, drop_cache: bool) -> Result<Vec<f32>, SafetensorsError> {
        let t = self.find(name).ok_or_else(|| SafetensorsError::MissingTensor(name.to_string()))?;
        if t.dtype == DType::F8E4M3 {
            return self.read_f32_fp8_blockscale(name, drop_cache);
        }
        let mut raw = vec![0u8; t.nbytes as usize];
        read_exact_at(&self.files[t.file_index], t.offset, &mut raw)?;
        let mut out = Vec::with_capacity(t.numel as usize);
        decode_floats(&raw, t.dtype, &mut out);
        if drop_cache {
            self.fadvise_dontneed(t.file_index, t.offset, t.nbytes);
        }
        Ok(out)
    }

    /// Dequantizes an FP8 (e4m3fn) weight tensor using its `{name}_scale_inv` sidecar: one
    /// scale per 128x128 block, broadcast to every element in that block (cropped at the
    /// tensor's actual edges when `rows`/`cols` isn't a multiple of 128) — `f32 =
    /// f8e4m3_to_f32(byte) * scale[row/128, col/128]`. Requires a 2D shape; every FP8 tensor
    /// in the real checkpoint is a 2D weight matrix (biases/norms are release in bf16/f32).
    fn read_f32_fp8_blockscale(&self, name: &str, drop_cache: bool) -> Result<Vec<f32>, SafetensorsError> {
        let t = self.find(name).ok_or_else(|| SafetensorsError::MissingTensor(name.to_string()))?;
        if t.shape.len() != 2 {
            return Err(SafetensorsError::BadHeader(format!("{name}: FP8 tensor must be 2D, got shape {:?}", t.shape)));
        }
        let (rows, cols) = (t.shape[0], t.shape[1]);

        let mut raw = vec![0u8; t.nbytes as usize];
        read_exact_at(&self.files[t.file_index], t.offset, &mut raw)?;
        if drop_cache {
            self.fadvise_dontneed(t.file_index, t.offset, t.nbytes);
        }

        let scale_name = format!("{name}_scale_inv");
        let sc = self.read_f32(&scale_name, drop_cache)?;
        dequant_fp8_blockscale(&raw, rows, cols, &sc, &scale_name)
    }

    /// reads a tensor's raw bytes with no dtype conversion — for our own pre-quantized U8
    /// containers (int4/int8/int2 packed + scales).
    pub fn read_raw(&self, name: &str, drop_cache: bool) -> Result<Vec<u8>, SafetensorsError> {
        let t = self.find(name).ok_or_else(|| SafetensorsError::MissingTensor(name.to_string()))?;
        let mut raw = vec![0u8; t.nbytes as usize];
        read_exact_at(&self.files[t.file_index], t.offset, &mut raw)?;
        if drop_cache {
            self.fadvise_dontneed(t.file_index, t.offset, t.nbytes);
        }
        Ok(raw)
    }

    /// reads a slice of `n_elems` elements starting at element `elem_off` and converts to
    /// f32. Used to read a single fused expert out of a `[E, ...]` block tensor without
    /// reading the whole block.
    pub fn read_slice_f32(
        &self,
        name: &str,
        elem_off: u64,
        n_elems: u64,
        drop_cache: bool,
    ) -> Result<Vec<f32>, SafetensorsError> {
        let t = self.find(name).ok_or_else(|| SafetensorsError::MissingTensor(name.to_string()))?;
        let esz: u64 = if t.dtype == DType::F32 { 4 } else { 2 };
        let byte_off = t.offset + elem_off * esz;
        let nbytes = n_elems * esz;
        let mut raw = vec![0u8; nbytes as usize];
        read_exact_at(&self.files[t.file_index], byte_off, &mut raw)?;
        let mut out = Vec::with_capacity(n_elems as usize);
        decode_floats(&raw, t.dtype, &mut out);
        if drop_cache {
            self.fadvise_dontneed(t.file_index, byte_off, nbytes);
        }
        Ok(out)
    }

    /// Raw location of a tensor — file descriptor + absolute byte range + on-disk dtype —
    /// for a caller doing its own I/O (Fase 8's io_uring batch expert loader) instead of
    /// going through the synchronous `read_at` calls `read_f32`/`read_raw` use internally.
    pub fn tensor_location(&self, name: &str) -> Option<TensorLocation> {
        let t = self.find(name)?;
        Some(TensorLocation {
            fd: self.files[t.file_index].as_raw_fd(),
            offset: t.offset,
            nbytes: t.nbytes,
            dtype: t.dtype,
        })
    }

    /// Decodes bytes read by external I/O (matching a `TensorLocation`) into `f32`s, using
    /// the exact same dtype conversion `read_f32` uses — so a caller with its own I/O
    /// mechanism still gets bf16/f16/f32 handled uniformly, not just f32.
    pub fn decode_f32(raw: &[u8], dtype: DType) -> Vec<f32> {
        let mut out = Vec::with_capacity(raw.len() / dtype.byte_size());
        decode_floats(raw, dtype, &mut out);
        out
    }

    /// Returns (creating on first use) a shared, read-only `mmap` of shard `file_index` — the
    /// `--mmap-experts` opt-in path's entry point (see this module's top doc). Cheap to call
    /// repeatedly: the actual `mmap(2)` syscall only happens once per shard per `Shards`
    /// instance (`OnceLock`), every later call just clones the `Arc`.
    ///
    /// # Safety
    /// `memmap2::Mmap::map` is unsafe because the memory it hands back can become invalid if the
    /// underlying file is truncated or mutated while mapped (the OS gives no protection against
    /// that in general). That precondition holds here: every `.safetensors` shard `Shards` opens
    /// is treated as read-only for the entire process lifetime — never written, truncated, or
    /// otherwise mutated after `open`/`open_multi` returns (see this module's top doc and
    /// `rabbit-plan.md`) — the exact same invariant `expert_cache.rs`'s `io_uring` usage already
    /// leans on for its own unsafe raw-pointer reads. The backing `File` is kept alive in
    /// `self.files` for as long as `Shards` (and so any `Arc<Mmap>` handed out here) lives.
    fn mmap_shard(&self, file_index: usize) -> Result<Arc<memmap2::Mmap>, SafetensorsError> {
        if let Some(m) = self.mmaps[file_index].get() {
            return Ok(Arc::clone(m));
        }
        // SAFETY: see the doc comment above — the shard file is read-only for the process's
        // entire lifetime, so no writer can ever race this mapping.
        let mmap = unsafe { memmap2::Mmap::map(&self.files[file_index]) }?;
        let arc = Arc::new(mmap);
        // Best-effort cache: if another thread raced us and already set it, just use theirs
        // (both mappings are equally valid views of the same read-only file, so there's no
        // correctness difference, only a wasted `mmap` call on the loser — acceptable for a
        // once-per-shard-per-process event).
        let _ = self.mmaps[file_index].set(Arc::clone(&arc));
        Ok(self.mmaps[file_index].get().cloned().unwrap_or(arc))
    }

    /// Given a tensor name, returns its shard's mmap handle plus the absolute byte range within
    /// that mapping holding the tensor's raw bytes — the mmap-path counterpart to
    /// `tensor_location` (which the default `io_uring`/`pread` path uses instead). `Ok(None)` if
    /// the tensor doesn't exist (matching `tensor_location`'s own `Option` return, not an error —
    /// callers that need a hard error, like `expert_cache.rs`'s mmap loader, map that themselves).
    /// `--mmap-experts` only; the default load path never calls this.
    pub fn tensor_mmap_range(&self, name: &str) -> Result<Option<MmapTensorRange>, SafetensorsError> {
        let Some(t) = self.find(name) else { return Ok(None) };
        let mmap = self.mmap_shard(t.file_index)?;
        let start = t.offset as usize;
        let end = start + t.nbytes as usize;
        Ok(Some(MmapTensorRange { mmap, range: start..end }))
    }

    /// Best-effort `MADV_WILLNEED` hint over `range` within `mmap` — tells the kernel to start
    /// pulling those pages in from disk in the background, the mmap path's analogue of this
    /// module's own `posix_fadvise(..., POSIX_FADV_WILLNEED)` (see `prefetch`) and of
    /// `expert_cache.rs`'s `io_uring` batch submission. A hint only: never blocks, and a failure
    /// (or the kernel just ignoring it) is silently fine — nothing here is a correctness
    /// requirement, only a possible latency win. `--mmap-experts` only.
    ///
    /// `madvise(2)` requires `addr` to be page-aligned; `range.start` is rounded DOWN to the
    /// nearest page boundary and the hinted length extended to match (never shrunk — rounding
    /// the END up too would risk covering unrelated data past a mapping that isn't itself a
    /// whole number of pages long, so instead `range.end` is clamped to `mmap.len()`).
    ///
    /// # Safety
    /// `addr`/`len` are derived from `range`, clamped to `0..mmap.len()` before the call, so the
    /// hinted region can never fall outside `mmap`'s own mapping — the one precondition
    /// `madvise` needs. `madvise` itself only touches kernel page-table bookkeeping for the
    /// given range; it never reads or writes the mapped bytes, so there is no data race to
    /// reason about even under concurrent access from other threads.
    pub fn madvise_willneed(mmap: &memmap2::Mmap, range: std::ops::Range<usize>) {
        let page = page_size();
        let end = range.end.min(mmap.len());
        let start_aligned = (range.start.min(end) / page) * page;
        if start_aligned >= end {
            return;
        }
        let len = end - start_aligned;
        // SAFETY: see the doc comment above.
        unsafe {
            let ptr = mmap.as_ptr().add(start_aligned) as *mut libc::c_void;
            libc::madvise(ptr, len, libc::MADV_WILLNEED);
        }
    }
}

/// Page size used to align `madvise_willneed`'s hinted range — this codebase has no existing
/// `PAGE_SIZE` constant to reuse (checked), so this reads the real value via `sysconf` (correct
/// even on the rare platform where it isn't 4096), falling back to the near-universal 4096 only
/// if the syscall itself fails, which it does not on any real Linux/Unix host — defensive only.
fn page_size() -> usize {
    // SAFETY: `sysconf` is a pure query with no pointer arguments; always safe to call.
    let sz = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if sz > 0 { sz as usize } else { 4096 }
}

/// File descriptor + absolute byte range + dtype for one tensor — see `Shards::tensor_location`.
#[derive(Debug, Clone, Copy)]
pub struct TensorLocation {
    pub fd: std::os::unix::io::RawFd,
    pub offset: u64,
    pub nbytes: u64,
    pub dtype: DType,
}

/// mmap handle + absolute byte range for one tensor — see `Shards::tensor_mmap_range`, the
/// `--mmap-experts` opt-in path's counterpart to `TensorLocation` above (which the default
/// `pread`/`io_uring` path uses instead).
pub struct MmapTensorRange {
    pub mmap: Arc<memmap2::Mmap>,
    pub range: std::ops::Range<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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

    /// hand-builds a valid `.safetensors` file: 8-byte LE header length, JSON header,
    /// then the concatenated raw tensor bytes in the same order.
    fn build_safetensors(entries: &[(&str, &str, Vec<usize>, Vec<u8>)]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_string(), json!({"format": "rabbit-test"}));
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in entries {
            let start = data.len() as u64;
            data.extend_from_slice(bytes);
            let end = data.len() as u64;
            header.insert(
                name.to_string(),
                json!({"dtype": dtype, "shape": shape, "data_offsets": [start, end]}),
            );
        }
        let header_bytes = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_bytes);
        out.extend_from_slice(&data);
        out
    }

    fn f32_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn bf16_and_f16_decode_known_constants() {
        // half-precision 1.0 = 0x3C00, -2.0 = 0xC000 (IEEE 754 binary16 reference values).
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        // bf16 is just the top 16 bits of f32: 1.5 -> 0x3FC0, 3.0 -> 0x4040.
        assert_eq!(bf16_to_f32(0x3FC0), 1.5);
        assert_eq!(bf16_to_f32(0x4040), 3.0);
    }

    #[test]
    fn indexes_and_reads_multiple_shards() {
        let dir = TempDir::new("rabbit_test_st_multishard");

        let shard1 = build_safetensors(&[(
            "dense.weight",
            "F32",
            vec![2, 3],
            f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )]);
        fs::write(dir.0.join("model-00001-of-00002.safetensors"), shard1).unwrap();

        let shard2 = build_safetensors(&[
            ("expert.0.raw", "U8", vec![4], vec![10, 20, 30, 40]),
            ("half.weight", "F16", vec![2], vec![0x00, 0x3C, 0x00, 0xC0]), // 1.0, -2.0 LE
            ("bf16.weight", "BF16", vec![2], vec![0xC0, 0x3F, 0x40, 0x40]), // 1.5, 3.0 LE
        ]);
        fs::write(dir.0.join("model-00002-of-00002.safetensors"), shard2).unwrap();

        let shards = Shards::open(&dir.0).unwrap();

        assert!(shards.has("dense.weight"));
        assert!(!shards.has("nope.weight"));
        assert_eq!(shards.find("dense.weight").unwrap().numel, 6);

        let dense = shards.read_f32("dense.weight", false).unwrap();
        assert_eq!(dense, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let tail = shards.read_slice_f32("dense.weight", 3, 3, false).unwrap();
        assert_eq!(tail, vec![4.0, 5.0, 6.0]);

        let raw = shards.read_raw("expert.0.raw", false).unwrap();
        assert_eq!(raw, vec![10, 20, 30, 40]);

        let half = shards.read_f32("half.weight", false).unwrap();
        assert_eq!(half, vec![1.0, -2.0]);

        let bf16 = shards.read_f32("bf16.weight", false).unwrap();
        assert_eq!(bf16, vec![1.5, 3.0]);
    }

    #[test]
    fn open_multi_indexes_shards_split_across_separate_directories() {
        // Simulates a checkpoint split across two drives/mounts (e.g. `--model` plus
        // `--shard-dirs`): two independent directories, each holding one shard, no directory
        // aware of the other's existence.
        let dir1 = TempDir::new("rabbit_test_st_multidir_1");
        let dir2 = TempDir::new("rabbit_test_st_multidir_2");

        let shard1 = build_safetensors(&[("dense.weight", "F32", vec![2, 3], f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))]);
        fs::write(dir1.0.join("out-00000.safetensors"), shard1).unwrap();

        let shard2 = build_safetensors(&[("expert.0.raw", "U8", vec![4], vec![10, 20, 30, 40])]);
        fs::write(dir2.0.join("out-00001.safetensors"), shard2).unwrap();

        let shards = Shards::open_multi(&[dir1.0.clone(), dir2.0.clone()]).unwrap();

        assert!(shards.has("dense.weight"));
        assert!(shards.has("expert.0.raw"));
        assert_eq!(shards.read_f32("dense.weight", false).unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(shards.read_raw("expert.0.raw", false).unwrap(), vec![10, 20, 30, 40]);
    }

    #[test]
    fn tensor_location_plus_decode_f32_matches_read_f32() {
        let dir = TempDir::new("rabbit_test_st_location");
        let shard = build_safetensors(&[
            ("dense.weight", "F32", vec![2, 3], f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            ("half.weight", "F16", vec![2], vec![0x00, 0x3C, 0x00, 0xC0]),
        ]);
        fs::write(dir.0.join("model.safetensors"), shard).unwrap();
        let shards = Shards::open(&dir.0).unwrap();

        for name in ["dense.weight", "half.weight"] {
            let loc = shards.tensor_location(name).unwrap();
            let mut raw = vec![0u8; loc.nbytes as usize];
            use std::os::unix::fs::FileExt;
            let file = unsafe { <std::fs::File as std::os::fd::FromRawFd>::from_raw_fd(loc.fd) };
            file.read_exact_at(&mut raw, loc.offset).unwrap();
            std::mem::forget(file); // borrowed fd: must not close it on drop

            let via_location = Shards::decode_f32(&raw, loc.dtype);
            let via_read_f32 = shards.read_f32(name, false).unwrap();
            assert_eq!(via_location, via_read_f32, "{name}");
        }

        assert!(shards.tensor_location("does.not.exist").is_none());
    }

    #[test]
    fn missing_tensor_is_an_error_not_a_panic() {
        let dir = TempDir::new("rabbit_test_st_missing");
        let shard = build_safetensors(&[("a", "F32", vec![1], f32_bytes(&[1.0]))]);
        fs::write(dir.0.join("model.safetensors"), shard).unwrap();

        let shards = Shards::open(&dir.0).unwrap();
        let err = shards.read_f32("does.not.exist", false).unwrap_err();
        assert!(matches!(err, SafetensorsError::MissingTensor(_)));
    }

    #[test]
    fn open_rejects_a_header_length_claiming_to_exceed_the_file() {
        let dir = TempDir::new("rabbit_test_st_hlen_overflow");
        // a real header length would be small; claim the header is bigger than the whole file.
        let mut out = Vec::new();
        out.extend_from_slice(&u64::MAX.to_le_bytes());
        out.extend_from_slice(b"{}");
        fs::write(dir.0.join("model.safetensors"), out).unwrap();

        let err = Shards::open(&dir.0).unwrap_err();
        assert!(matches!(err, SafetensorsError::BadHeader(_)), "{err}");
    }

    #[test]
    fn open_rejects_data_offsets_where_end_precedes_start() {
        let dir = TempDir::new("rabbit_test_st_offsets_inverted");
        let mut header = serde_json::Map::new();
        header.insert("a".to_string(), json!({"dtype": "F32", "shape": [1], "data_offsets": [4, 0]}));
        let header_bytes = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_bytes);
        out.extend_from_slice(&f32_bytes(&[1.0]));
        fs::write(dir.0.join("model.safetensors"), out).unwrap();

        let err = Shards::open(&dir.0).unwrap_err();
        assert!(matches!(err, SafetensorsError::BadHeader(_)), "{err}");
    }

    #[test]
    fn open_rejects_data_offsets_extending_past_end_of_file() {
        let dir = TempDir::new("rabbit_test_st_offsets_oob");
        let mut header = serde_json::Map::new();
        // claims 4096 bytes of tensor data but the file has none after the header.
        header.insert("a".to_string(), json!({"dtype": "F32", "shape": [1024], "data_offsets": [0, 4096]}));
        let header_bytes = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_bytes);
        fs::write(dir.0.join("model.safetensors"), out).unwrap();

        let err = Shards::open(&dir.0).unwrap_err();
        assert!(matches!(err, SafetensorsError::BadHeader(_)), "{err}");
    }

    #[test]
    fn drop_cache_flag_does_not_error_out() {
        let dir = TempDir::new("rabbit_test_st_dropcache");
        let shard = build_safetensors(&[("a", "F32", vec![2], f32_bytes(&[7.0, 8.0]))]);
        fs::write(dir.0.join("model.safetensors"), shard).unwrap();

        let shards = Shards::open(&dir.0).unwrap();
        let v = shards.read_f32("a", true).unwrap();
        assert_eq!(v, vec![7.0, 8.0]);
        shards.prefetch("a");
        shards.prefetch("does.not.exist"); // no-op, must not panic
    }

    #[test]
    fn f8e4m3_decodes_known_bit_patterns() {
        assert_eq!(f8e4m3_to_f32(0x00), 0.0); // +0 (subnormal, mantissa=0)
        assert_eq!(f8e4m3_to_f32(0x38), 1.0); // 0_0111_000: (1+0/8)*2^0
        assert_eq!(f8e4m3_to_f32(0xB8), -1.0); // sign bit set -> -1.0
        assert_eq!(f8e4m3_to_f32(0x7E), 448.0); // 0_1111_110: max finite e4m3fn value
        assert!(f8e4m3_to_f32(0x7F).is_nan()); // 0_1111_111: reserved for NaN
        // subnormal: mantissa=1, exp=0 -> 1 * 2^-9
        assert!((f8e4m3_to_f32(0x01) - 2f32.powi(-9)).abs() < 1e-12);
    }

    #[test]
    fn read_f32_dequantizes_fp8_with_a_single_block_scale() {
        let dir = TempDir::new("rabbit_test_st_fp8_single_block");
        // 2x3, well inside one 128x128 block -> one scale value covers the whole tensor.
        let fp8_bytes = vec![0x38, 0xB8, 0x00, 0x38, 0x38, 0xB8]; // 1.0,-1.0,0.0,1.0,1.0,-1.0
        let scale = 3.5f32;
        let shard = build_safetensors(&[
            ("w", "F8_E4M3", vec![2, 3], fp8_bytes),
            ("w_scale_inv", "F32", vec![1, 1], f32_bytes(&[scale])),
        ]);
        fs::write(dir.0.join("model.safetensors"), shard).unwrap();
        let shards = Shards::open(&dir.0).unwrap();

        let got = shards.read_f32("w", false).unwrap();
        let expected = [1.0, -1.0, 0.0, 1.0, 1.0, -1.0].map(|v: f32| v * scale);
        assert_eq!(got, expected);
    }

    #[test]
    fn read_f32_dequantizes_fp8_across_multiple_row_blocks() {
        let dir = TempDir::new("rabbit_test_st_fp8_multi_block");
        // 130 rows x 1 col -> spans row-block 0 (rows 0..128) and row-block 1 (rows 128..130),
        // each with its own scale; every FP8 byte is the same (0x38 = 1.0) so the ONLY thing
        // that can vary per row is which block's scale got applied.
        let rows = 130usize;
        let fp8_bytes = vec![0x38u8; rows]; // every element decodes to 1.0 before scaling
        let block0_scale = 2.0f32;
        let block1_scale = 100.0f32;
        let shard = build_safetensors(&[
            ("w", "F8_E4M3", vec![rows, 1], fp8_bytes),
            ("w_scale_inv", "F32", vec![2, 1], f32_bytes(&[block0_scale, block1_scale])),
        ]);
        fs::write(dir.0.join("model.safetensors"), shard).unwrap();
        let shards = Shards::open(&dir.0).unwrap();

        let got = shards.read_f32("w", false).unwrap();
        assert_eq!(got.len(), rows);
        for (r, &v) in got.iter().enumerate() {
            let expected = if r < 128 { block0_scale } else { block1_scale };
            assert_eq!(v, expected, "row {r}");
        }
    }

    #[test]
    fn tensor_mmap_range_bytes_match_read_raw() {
        let dir = TempDir::new("rabbit_test_st_mmap_range");
        let shard = build_safetensors(&[
            ("dense.weight", "F32", vec![2, 3], f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            ("expert.raw", "U8", vec![4], vec![10, 20, 30, 40]),
        ]);
        fs::write(dir.0.join("model.safetensors"), shard).unwrap();
        let shards = Shards::open(&dir.0).unwrap();

        for name in ["dense.weight", "expert.raw"] {
            let MmapTensorRange { mmap, range } = shards.tensor_mmap_range(name).unwrap().expect("tensor exists");
            let via_mmap = &mmap[range];
            let via_read_raw = shards.read_raw(name, false).unwrap();
            assert_eq!(via_mmap, via_read_raw.as_slice(), "{name}");
        }

        assert!(shards.tensor_mmap_range("does.not.exist").unwrap().is_none());
    }

    #[test]
    fn mmap_shard_is_cached_across_repeated_lookups() {
        let dir = TempDir::new("rabbit_test_st_mmap_cache");
        let shard = build_safetensors(&[("a", "F32", vec![1], f32_bytes(&[1.0]))]);
        fs::write(dir.0.join("model.safetensors"), shard).unwrap();
        let shards = Shards::open(&dir.0).unwrap();

        let mmap1 = shards.tensor_mmap_range("a").unwrap().unwrap().mmap;
        let mmap2 = shards.tensor_mmap_range("a").unwrap().unwrap().mmap;
        // same underlying allocation, not just equal bytes -- proves the `OnceLock` cache hit,
        // not a fresh `mmap(2)` syscall on the second lookup.
        assert!(Arc::ptr_eq(&mmap1, &mmap2));
    }

    #[test]
    fn madvise_willneed_does_not_panic_across_edge_case_ranges() {
        let dir = TempDir::new("rabbit_test_st_madvise");
        let shard = build_safetensors(&[("a", "F32", vec![8], f32_bytes(&[0.0; 8]))]);
        fs::write(dir.0.join("model.safetensors"), shard).unwrap();
        let shards = Shards::open(&dir.0).unwrap();
        let MmapTensorRange { mmap, range } = shards.tensor_mmap_range("a").unwrap().unwrap();

        Shards::madvise_willneed(&mmap, range.clone()); // ordinary, unaligned range
        Shards::madvise_willneed(&mmap, 0..mmap.len()); // whole mapping
        Shards::madvise_willneed(&mmap, 0..0); // empty range -- must be a no-op, not a panic
        Shards::madvise_willneed(&mmap, mmap.len()..mmap.len() + 4096); // past the end -- clamped away
    }

    #[test]
    fn read_f32_errors_when_fp8_scale_inv_is_missing() {
        let dir = TempDir::new("rabbit_test_st_fp8_no_scale");
        let shard = build_safetensors(&[("w", "F8_E4M3", vec![1, 2], vec![0x38, 0x38])]);
        fs::write(dir.0.join("model.safetensors"), shard).unwrap();
        let shards = Shards::open(&dir.0).unwrap();

        let err = shards.read_f32("w", false).unwrap_err();
        assert!(matches!(err, SafetensorsError::MissingTensor(_)));
    }
}
