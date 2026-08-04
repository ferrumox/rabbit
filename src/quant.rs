//! Port of the `QT` container + `qt_alloc`/`qt_fill`/`quantize_rows`/`pack_int4`/`pack_int2`
//! from `glm.c` — per-row symmetric quantization to int8 (1 byte/param), packed int4
//! (2 values/byte) or packed int2 (4 values/byte), plus the f32 passthrough format.
//!
//! `bits` and format are independent: `bits` sets the quantization range (`qmax`) and divides
//! the container into three storage tiers, but e.g. `bits=3` still lands in the int4 tier
//! (`>=3`) using less than its full nibble range — this is what the reference implementation's `NOPACK` env var
//! exploits to store sub-8-bit values in an unpacked int8 container for validating that the
//! packed and unpacked encodings agree bit-for-bit (see the tests below).
//!
//! Rounding: C's `lrintf` respects the default IEEE-754 rounding mode (round-to-nearest,
//! ties-to-even — "banker's rounding"), NOT `f32::round`'s round-half-away-from-zero. Using
//! `round_ties_even` here is required for token-exact parity with the C oracle, not a style
//! preference.

use std::sync::Arc;

/// Backing storage for one packed byte buffer inside `QTKind::MxFp4` — every other `QTKind`
/// variant still stores a plain owned `Vec<u8>` directly, unchanged. `Owned` behaves exactly
/// like those plain `Vec<u8>` fields always did; `Mapped` is the opt-in `--mmap-experts` load
/// path's alternative (see `expert_cache.rs`'s `mmap_load` module and
/// `safetensors.rs`'s `Shards::tensor_mmap_range`): a shared `mmap` handle plus the byte range
/// within it holding this buffer's bytes, avoiding the owned allocation + copy entirely.
/// `as_slice()` erases the distinction for every reader (`QT::row_f32`,
/// `kernels.rs::matmul_mxfp4`/`qt_addrow`/`qt_matvec_rows`, `expert_cache.rs::mlock_best_effort`,
/// ...): both variants deref to the exact same bytes an owned buffer would have held.
pub enum ByteStore {
    Owned(Vec<u8>),
    Mapped { mmap: Arc<memmap2::Mmap>, range: std::ops::Range<usize> },
}

impl ByteStore {
    pub fn as_slice(&self) -> &[u8] {
        match self {
            ByteStore::Owned(v) => v.as_slice(),
            ByteStore::Mapped { mmap, range } => &mmap[range.clone()],
        }
    }
}

#[inline]
pub(crate) fn lrint(x: f32) -> i32 {
    x.round_ties_even() as i32
}

/// `QT::from_packed`'s `data`/`scale` didn't match any of the three packed byte-count formulas
/// for the given `[rows,cols]` — a mismatched or corrupt `.qs` sidecar file.
#[derive(Debug)]
pub struct PackedFormatError {
    pub data_len: usize,
    pub scale_len: usize,
    pub rows: usize,
    pub cols: usize,
}

impl std::fmt::Display for PackedFormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "packed container for [{},{}]: {} data bytes / {} scales don't match int8 ({}), int4 ({}), or int2 ({}) row counts",
            self.rows,
            self.cols,
            self.data_len,
            self.scale_len,
            self.rows * self.cols,
            self.rows * self.cols.div_ceil(2),
            self.rows * self.cols.div_ceil(4),
        )
    }
}

impl std::error::Error for PackedFormatError {}

/// A quantized `[rows, cols]` weight matrix in one of five formats, chosen by `bits` (or, for
/// `MxFp4`, by the dedicated `alloc_mxfp4` constructor — see its doc for why it sits outside
/// the `bits`-threshold ladder the other four share).
pub struct QT {
    pub rows: usize,
    pub cols: usize,
    pub bits: u8,
    pub kind: QTKind,
}

pub enum QTKind {
    F32(Vec<f32>),
    /// 1 byte/value + per-row scale.
    I8 { data: Vec<i8>, scale: Vec<f32> },
    /// 2 values/byte (low nibble first), stored as `value + 8` in `0..16`.
    I4 { data: Vec<u8>, scale: Vec<f32> },
    /// 4 values/byte (2 bits each, low bits first), stored as `value + 2` in `0..4`.
    I2 { data: Vec<u8>, scale: Vec<f32> },
    /// Same packed layout as `I4`, but scaled per `group_size`-element GROUP along each row
    /// instead of once per whole row — `quant_int4_grouped`'s output format, see its own doc for
    /// why (matches the FP8 source's own 128×128 block-scale granularity, much better error than
    /// per-row for wide rows). `scale.len() == rows * cols.div_ceil(group_size)`, laid out
    /// `scale[row * ngroups + group]`. `group_size` is carried on the value itself (not
    /// inferred from `scale.len()` at load time) — inferring it back from `(cols, ngroups)` alone
    /// is only exact when `group_size` evenly divides `cols`; storing it explicitly avoids ever
    /// silently misassigning a tail column to the wrong group's scale.
    I4Grouped { data: Vec<u8>, scale: Vec<f32>, group_size: usize },
    /// OCP Microscaling (MX) FP4: 4-bit E2M1 floating-point elements, 2/byte (low nibble
    /// first — same nibble order as `I4`), with one shared E8M0 (8-bit, exponent-only) scale
    /// per 32-element block along the row — NOT a per-row `f32` scale like every other variant
    /// here. See `e2m1_decode`/`e8m0_decode`'s docs for the exact bit layouts (OCP Microscaling
    /// Formats (MX) v1.0 spec). `data.len() == rows * cols.div_ceil(2)`,
    /// `block_scale.len() == rows * cols.div_ceil(32)`. Stored as `ByteStore` (not a plain
    /// `Vec<u8>`, unlike every other variant here) so the `--mmap-experts` load path can hold
    /// mapped byte ranges instead — see `ByteStore`'s own doc.
    MxFp4 { data: ByteStore, block_scale: ByteStore },
}

impl QT {
    /// Chooses a format from `bits`, mirroring `qt_alloc`: `>=16` -> f32, `>=5` (or
    /// `nopack`, the reference's `NOPACK=1`) -> int8, `>=3` -> int4, else -> int2.
    pub fn alloc(rows: usize, cols: usize, bits: u8, nopack: bool) -> QT {
        let kind = if bits >= 16 {
            QTKind::F32(vec![0.0; rows * cols])
        } else if bits >= 5 || nopack {
            QTKind::I8 { data: vec![0; rows * cols], scale: vec![0.0; rows] }
        } else if bits >= 3 {
            QTKind::I4 { data: vec![0; rows * cols.div_ceil(2)], scale: vec![0.0; rows] }
        } else {
            QTKind::I2 { data: vec![0; rows * cols.div_ceil(4)], scale: vec![0.0; rows] }
        };
        QT { rows, cols, bits, kind }
    }

    /// Allocates the OCP-MX FP4 format — a dedicated constructor, not another `bits` threshold
    /// in `alloc`'s ladder, because MXFP4 isn't a point on that ladder's linear-quantization
    /// spectrum: it's a floating-point element format with a block-shared exponent-only scale,
    /// picked explicitly (by a checkpoint declaring itself MXFP4), never inferred from a bit
    /// count the way `alloc`'s other four tiers are.
    pub fn alloc_mxfp4(rows: usize, cols: usize) -> QT {
        QT {
            rows,
            cols,
            bits: 4,
            kind: QTKind::MxFp4 {
                data: ByteStore::Owned(vec![0; rows * cols.div_ceil(2)]),
                block_scale: ByteStore::Owned(vec![0; rows * cols.div_ceil(32)]),
            },
        }
    }

    /// Builds an MXFP4 `QT` directly from mmap'd byte ranges instead of allocating and filling
    /// owned buffers — the `--mmap-experts` load path's constructor (see
    /// `expert_cache.rs`'s `mmap_load` module), parallel to `alloc_mxfp4` + `fill()`'s
    /// owned-buffer path but skipping the allocation and copy entirely. Performs no validation
    /// of `data_range`/`scale_range`'s lengths against `rows`/`cols` — callers (`expert_cache.rs`)
    /// already validate that against the checkpoint's declared shape before calling this, the
    /// same way `mxfp4_from_raw` does for the owned path.
    pub fn from_mmap_mxfp4(
        rows: usize,
        cols: usize,
        data_mmap: Arc<memmap2::Mmap>,
        data_range: std::ops::Range<usize>,
        scale_mmap: Arc<memmap2::Mmap>,
        scale_range: std::ops::Range<usize>,
    ) -> QT {
        QT {
            rows,
            cols,
            bits: 4,
            kind: QTKind::MxFp4 {
                data: ByteStore::Mapped { mmap: data_mmap, range: data_range },
                block_scale: ByteStore::Mapped { mmap: scale_mmap, range: scale_range },
            },
        }
    }

    /// Allocates the grouped-scale int4 format — a dedicated constructor (like `alloc_mxfp4`),
    /// not another `bits`-ladder tier in `alloc`: grouping is an orthogonal choice from bits
    /// (any `bits<=4` value can be grouped or not), picked explicitly by a checkpoint declaring
    /// `rabbit_group_size` in its `config.json`, never inferred from `bits` alone.
    pub fn alloc_grouped(rows: usize, cols: usize, bits: u8, group_size: usize) -> QT {
        let ngroups = cols.div_ceil(group_size);
        QT {
            rows,
            cols,
            bits,
            kind: QTKind::I4Grouped { data: vec![0; rows * cols.div_ceil(2)], scale: vec![0.0; rows * ngroups], group_size },
        }
    }

    /// Fills from row-major f32 weights `w[rows*cols]`, quantizing per the chosen format.
    pub fn fill(&mut self, w: &[f32]) {
        assert_eq!(w.len(), self.rows * self.cols);
        match &mut self.kind {
            QTKind::F32(data) => data.copy_from_slice(w),
            QTKind::I8 { data, scale } => quantize_rows(w, data, scale, self.rows, self.cols, self.bits),
            QTKind::I4 { data, scale } => pack_int4(w, data, scale, self.rows, self.cols, self.bits),
            QTKind::I2 { data, scale } => pack_int2(w, data, scale, self.rows, self.cols, self.bits),
            QTKind::I4Grouped { data, scale, group_size } => {
                let (d, s) = quant_int4_grouped(w, self.rows, self.cols, self.bits, *group_size);
                *data = d;
                *scale = s;
            }
            QTKind::MxFp4 { data, block_scale } => {
                // `fill` quantizes fresh f32 weights into this `QT`'s OWN buffers -- only
                // meaningful for owned storage. A `Mapped` MXFP4 (the `--mmap-experts` load
                // path's `from_mmap_mxfp4`) is always constructed already-filled, straight from
                // the checkpoint's on-disk bytes, and is never mutated afterward -- reaching this
                // branch on one would be a caller bug, not a real runtime state.
                let (ByteStore::Owned(d), ByteStore::Owned(bs)) = (data, block_scale) else {
                    panic!("QT::fill called on a mmap-backed MxFp4 (Mapped storage is filled once at construction, never re-filled)");
                };
                pack_mxfp4(w, d, bs, self.rows, self.cols)
            }
        }
    }

    /// Resident bytes, mirroring `qt_bytes` — the RAM-accounting choke point the streaming
    /// architecture leans on (see `safetensors.rs`'s module doc for why this matters).
    pub fn resident_bytes(&self) -> usize {
        match &self.kind {
            QTKind::F32(_) => self.rows * self.cols * 4,
            QTKind::I8 { .. } => self.rows * self.cols + self.rows * 4,
            QTKind::I4 { .. } => self.rows * self.cols.div_ceil(2) + self.rows * 4,
            QTKind::I2 { .. } => self.rows * self.cols.div_ceil(4) + self.rows * 4,
            QTKind::I4Grouped { group_size, .. } => self.rows * self.cols.div_ceil(2) + self.rows * self.cols.div_ceil(*group_size) * 4,
            // E8M0 is 1 byte/block (vs. the other tiers' 4-byte f32 per-row scale) — one of
            // MX's actual design points: exponent-only scale metadata stays cheap even at a
            // much finer (32-element, not per-row) granularity.
            QTKind::MxFp4 { .. } => self.rows * self.cols.div_ceil(2) + self.rows * self.cols.div_ceil(32),
        }
    }

    /// Wraps already-quantized bytes — the reference's `name.qs` pre-quantized container (see
    /// `qt_from_disk` in `glm.c`): `data` is the packed int8/int4/int2 payload straight off
    /// disk (no `quantize_rows`/`pack_int4`/`pack_int2` work), `scale` is its per-row F32
    /// scale, also read straight off disk. Format is inferred from `data.len()` against the
    /// three possible byte counts for `[rows,cols]` — same probe as `qt_from_disk`'s `fmt`
    /// (`nb==O*I` -> int8, `nb==O*ceil(I/2)` -> int4, else int2), except this returns an
    /// error instead of silently assuming int2 when nothing matches (a mismatched/corrupt
    /// `.qs` file deserves a message, not a wrong guess).
    pub fn from_packed(rows: usize, cols: usize, bits: u8, data: Vec<u8>, scale: Vec<f32>) -> Result<QT, PackedFormatError> {
        if scale.len() != rows {
            return Err(PackedFormatError { data_len: data.len(), scale_len: scale.len(), rows, cols });
        }
        let kind = if data.len() == rows * cols {
            QTKind::I8 { data: data.into_iter().map(|b| b as i8).collect(), scale }
        } else if data.len() == rows * cols.div_ceil(2) {
            QTKind::I4 { data, scale }
        } else if data.len() == rows * cols.div_ceil(4) {
            QTKind::I2 { data, scale }
        } else {
            return Err(PackedFormatError { data_len: data.len(), scale_len: scale.len(), rows, cols });
        };
        Ok(QT { rows, cols, bits, kind })
    }

    /// `from_packed`'s grouped-scale counterpart: `group_size` is a caller-supplied fact (read
    /// from the checkpoint's `config.json`, see `glm52::model`/`kimi_linear::model`'s
    /// `qt_load`), not inferred from `scale.len()` alone — see `QTKind::I4Grouped`'s doc for why
    /// blindly recovering the group BOUNDARIES that way isn't always exact.
    ///
    /// `group_size` is a checkpoint-WIDE setting (one `--group-size` flag for the whole
    /// conversion), but not every tensor in a checkpoint is actually grouped: the converter only
    /// groups tensors quantized at `bits<=4` (`shard.rs::quantize`'s own condition) — `io_bits`
    /// (embed/lm_head, commonly 8) and any other `bits>4` tensor still get plain per-row
    /// scaling. So `group_size>0` alone must NOT force every `.qs` sidecar to be read as
    /// grouped: `scale.len()==rows` (this specific tensor's real, observed shape) is what
    /// decides "was THIS tensor grouped", with `group_size` only used to validate/interpret it
    /// once that's true — routes to plain `from_packed` whenever `group_size==0` OR the scale
    /// array is already exactly `rows` long (the per-row shape, unambiguous either way).
    pub fn from_packed_grouped(rows: usize, cols: usize, bits: u8, data: Vec<u8>, scale: Vec<f32>, group_size: usize) -> Result<QT, PackedFormatError> {
        if group_size == 0 || scale.len() == rows {
            return Self::from_packed(rows, cols, bits, data, scale);
        }
        let ngroups = cols.div_ceil(group_size);
        if data.len() != rows * cols.div_ceil(2) || scale.len() != rows * ngroups {
            return Err(PackedFormatError { data_len: data.len(), scale_len: scale.len(), rows, cols });
        }
        Ok(QT { rows, cols, bits, kind: QTKind::I4Grouped { data, scale, group_size } })
    }

    /// Dequantizes row `row` into a fresh `Vec<f32>` of length `cols` — port of `embed_row`
    /// (which is just this, specialized to the embedding table) generalized to any QT, since
    /// nothing about it is embedding-specific.
    pub fn row_f32(&self, row: usize) -> Vec<f32> {
        let cols = self.cols;
        match &self.kind {
            QTKind::F32(data) => data[row * cols..(row + 1) * cols].to_vec(),
            QTKind::I8 { data, scale } => {
                let s = scale[row];
                data[row * cols..(row + 1) * cols].iter().map(|&v| v as f32 * s).collect()
            }
            QTKind::I4 { data, scale } => {
                let s = scale[row];
                let rb = cols.div_ceil(2);
                let wr = &data[row * rb..(row + 1) * rb];
                (0..cols)
                    .map(|k| {
                        let byte = wr[k >> 1];
                        let nibble = if k & 1 == 0 { byte & 0xF } else { byte >> 4 };
                        (nibble as i32 - 8) as f32 * s
                    })
                    .collect()
            }
            QTKind::I2 { data, scale } => {
                let s = scale[row];
                let rb = cols.div_ceil(4);
                let wr = &data[row * rb..(row + 1) * rb];
                (0..cols)
                    .map(|k| {
                        let byte = wr[k >> 2];
                        let bits = (byte >> ((k & 3) * 2)) & 3;
                        (bits as i32 - 2) as f32 * s
                    })
                    .collect()
            }
            QTKind::I4Grouped { data, scale, group_size } => {
                let ngroups = cols.div_ceil(*group_size);
                let srow = &scale[row * ngroups..(row + 1) * ngroups];
                let rb = cols.div_ceil(2);
                let wr = &data[row * rb..(row + 1) * rb];
                (0..cols)
                    .map(|k| {
                        let byte = wr[k >> 1];
                        let nibble = if k & 1 == 0 { byte & 0xF } else { byte >> 4 };
                        (nibble as i32 - 8) as f32 * srow[k / group_size]
                    })
                    .collect()
            }
            QTKind::MxFp4 { data, block_scale } => {
                let data = data.as_slice();
                let block_scale = block_scale.as_slice();
                let rb = cols.div_ceil(2);
                let bpr = cols.div_ceil(32);
                let wr = &data[row * rb..(row + 1) * rb];
                let bsr = &block_scale[row * bpr..(row + 1) * bpr];
                (0..cols)
                    .map(|k| {
                        let byte = wr[k >> 1];
                        let nibble = if k & 1 == 0 { byte & 0xF } else { byte >> 4 };
                        e2m1_decode(nibble) * e8m0_decode(bsr[k / 32])
                    })
                    .collect()
            }
        }
    }
}

/// The 8 positive magnitudes E2M1 (OCP MX's 4-bit floating-point element format: 1 sign + 2
/// exponent + 1 mantissa bit, exponent bias 1) can represent, indexed by the low 3 bits
/// `(exp<<1)|mantissa`. Denormals (`exp==0`) give `{0, 0.5}`; normals (`exp>=1`) give
/// `(1 + mantissa*0.5) * 2^(exp-1)`.
const E2M1_MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Decodes one E2M1 nibble (bit 3 = sign, bits 2..0 index [`E2M1_MAGNITUDES`]). `pub(crate)`
/// (not private, unlike the linear formats' inline nibble unpacking in `kernels.rs`) so the
/// matmul kernel reuses this exact LUT instead of duplicating a subtle floating-point decode.
pub(crate) fn e2m1_decode(nibble: u8) -> f32 {
    let mag = E2M1_MAGNITUDES[(nibble & 0x7) as usize];
    if nibble & 0x8 != 0 { -mag } else { mag }
}

/// Encodes a value to the nearest E2M1 code — a plain nearest-magnitude search over the 8 fixed
/// points (ties keep the lower index, i.e. the smaller magnitude; with only 8 unevenly-spaced
/// points a closed-form round doesn't apply the way it does for the linear int formats above).
fn e2m1_encode(value: f32) -> u8 {
    let sign = if value.is_sign_negative() && value != 0.0 { 0x8u8 } else { 0 };
    let av = value.abs();
    let mut best = 0usize;
    let mut best_diff = f32::MAX;
    for (i, &m) in E2M1_MAGNITUDES.iter().enumerate() {
        let diff = (av - m).abs();
        if diff < best_diff {
            best_diff = diff;
            best = i;
        }
    }
    sign | best as u8
}

/// Decodes an E8M0 byte (OCP MX's shared block scale: 8 bits, exponent-only, no sign/mantissa)
/// to its `f32` power-of-two value, bias 127 — the same bias IEEE-754 `f32`'s own exponent
/// field uses. `0xFF` is reserved for NaN in the spec; unreached here since `choose_block_scale`
/// never emits it (see its doc), so this only ever computes `2^(byte-127)`.
///
/// Called on the order of a billion times per decode token (once per 32-element block in
/// `matmul_mxfp4`'s hot loop — see `kernels.rs`), so this is NOT `2f32.powi(byte as i32 - 127)`
/// despite computing the exact same value: a real `perf record` of the real checkpoint found
/// `__powisf2` (the generic compiler-rt power routine `powi` lowers to) alone responsible for
/// ~29% of total cycles during decode — a needless cost for something IEEE-754 already encodes
/// directly. Since E8M0's byte IS `f32`'s own biased exponent field (both bias 127), `2^(byte-127)`
/// is exactly the `f32` bit pattern with that exponent and a zero mantissa — one shift, no libm
/// call. The single edge case a plain shift gets wrong is `byte == 0` (`2^-127`): that exponent is
/// below `f32`'s normal range (min normal is `2^-126`), so it needs the one subnormal bit pattern
/// that represents it instead of the biased-exponent-0 pattern (which would just be `0.0`).
pub(crate) fn e8m0_decode(byte: u8) -> f32 {
    if byte == 0 {
        return f32::from_bits(1u32 << 22); // 2^-127 == 0.5 * 2^-126, the largest-mantissa-half subnormal
    }
    f32::from_bits((byte as u32) << 23)
}

/// Picks the E8M0 scale byte for one 32-element block from its max absolute value: the largest
/// power of two `s` such that `amax / s` still fits under E2M1's max representable magnitude
/// (6.0), i.e. `shared_exp = floor(log2(amax)) - 2`. Returns `(byte, s)`. `amax == 0.0` (an
/// all-zero block) gets an arbitrary `s = 1.0` — the encoded values are all zero regardless of
/// scale, so the choice is a no-op. Clamped to `[-127, 127]` (byte `0..=254`) to stay clear of
/// E8M0's reserved `0xFF` NaN code.
fn choose_block_scale(amax: f32) -> (u8, f32) {
    if amax <= 0.0 {
        return (127, 1.0);
    }
    let shared_exp = (amax.log2().floor() as i32 - 2).clamp(-127, 127);
    (u8::try_from(shared_exp + 127).expect("clamped to 0..=254"), 2f32.powi(shared_exp))
}

/// f32[rows,cols] -> MXFP4[rows,ceil(cols/2)] + one E8M0 byte per 32-element block per row.
fn pack_mxfp4(w: &[f32], q: &mut [u8], block_scale: &mut [u8], rows: usize, cols: usize) {
    let rb = cols.div_ceil(2);
    let bpr = cols.div_ceil(32);
    for o in 0..rows {
        let wr = &w[o * cols..(o + 1) * cols];
        let qr = &mut q[o * rb..(o + 1) * rb];
        for b in 0..bpr {
            let start = b * 32;
            let end = (start + 32).min(cols);
            let amax = wr[start..end].iter().fold(0f32, |m, &v| m.max(v.abs()));
            let (byte, scale) = choose_block_scale(amax);
            block_scale[o * bpr + b] = byte;
            for k in start..end {
                let code = e2m1_encode(wr[k] / scale);
                if k & 1 == 0 {
                    qr[k >> 1] = (qr[k >> 1] & 0xF0) | code;
                } else {
                    qr[k >> 1] = (qr[k >> 1] & 0x0F) | (code << 4);
                }
            }
        }
    }
}

/// f32[rows,cols] -> int8[rows,cols] + per-row scale, symmetric quantization.
fn quantize_rows(w: &[f32], q: &mut [i8], scale: &mut [f32], rows: usize, cols: usize, bits: u8) {
    let qmax = (1i32 << (bits - 1)) - 1;
    for o in 0..rows {
        let wr = &w[o * cols..(o + 1) * cols];
        let amax = wr.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let s = (amax / qmax as f32).max(1e-8);
        scale[o] = s;
        let qr = &mut q[o * cols..(o + 1) * cols];
        for (qi, &wi) in qr.iter_mut().zip(wr) {
            *qi = lrint(wi / s).clamp(-qmax - 1, qmax) as i8;
        }
    }
}

/// f32[rows,cols] -> int4-packed[rows,ceil(cols/2)] + per-row scale. Values are clamped to
/// the full nibble range `[-8, qmax]`, not `[-qmax-1, qmax]` — the container floor is fixed
/// by the 4-bit storage regardless of how tight `bits` makes the ceiling (see module docs).
fn pack_int4(w: &[f32], q4: &mut [u8], scale: &mut [f32], rows: usize, cols: usize, bits: u8) {
    let qmax = (1i32 << (bits - 1)) - 1;
    let rb = cols.div_ceil(2);
    for o in 0..rows {
        let wr = &w[o * cols..(o + 1) * cols];
        let amax = wr.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let s = (amax / qmax as f32).max(1e-8);
        scale[o] = s;
        let qr = &mut q4[o * rb..(o + 1) * rb];
        let mut i = 0;
        while i < cols {
            let v0 = lrint(wr[i] / s).clamp(-8, qmax);
            let v1 = if i + 1 < cols { lrint(wr[i + 1] / s).clamp(-8, qmax) } else { 0 };
            qr[i / 2] = ((v0 + 8) as u8) | (((v1 + 8) as u8) << 4);
            i += 2;
        }
    }
}

/// f32[rows,cols] -> int4-packed[rows,ceil(cols/2)] + one scale per `gs`-element GROUP along
/// each row's input dim, instead of `pack_int4`'s one scale for the whole row — port of
/// colibrì's `quant_int4_grouped` (`c/tools/convert_fp8_to_int4.py`), a checkpoint-conversion
/// format not (yet) produced by rabbit's own runtime, whose `QT::I4` stays per-row. Matches
/// the FP8 source's own 128×128 block-scale granularity, drastically reducing quantization
/// error vs. per-row scaling for wide rows — see `rabbit-plan.md`'s converter notes. Same
/// fixed `[-8, qmax]` clamp and even/odd nibble interleaving as `pack_int4`, just resolving
/// each element's scale from its own group (`col / gs`) instead of the whole row.
/// `scale`'s layout is `[rows * ngroups]`, `scale[o * ngroups + g]` = row `o`, group `g`,
/// `ngroups = ceil(cols/gs)`. A trailing partial group (when `cols` isn't a multiple of `gs`)
/// is sized to whatever real columns remain — never padded with fabricated zero elements that
/// could pull its `amax` (and so its scale) away from what the real data alone would produce.
pub fn quant_int4_grouped(w: &[f32], rows: usize, cols: usize, bits: u8, gs: usize) -> (Vec<u8>, Vec<f32>) {
    let qmax = (1i32 << (bits - 1)) - 1;
    let ngroups = cols.div_ceil(gs);
    let rb = cols.div_ceil(2);
    let mut out = vec![0u8; rows * rb];
    let mut scale = vec![0f32; rows * ngroups];

    for o in 0..rows {
        let wr = &w[o * cols..(o + 1) * cols];
        for g in 0..ngroups {
            let start = g * gs;
            let end = (start + gs).min(cols);
            let amax = wr[start..end].iter().fold(0f32, |m, &v| m.max(v.abs()));
            scale[o * ngroups + g] = (amax / qmax as f32).max(1e-8);
        }
        let qr = &mut out[o * rb..(o + 1) * rb];
        let mut i = 0;
        while i < cols {
            let s0 = scale[o * ngroups + i / gs];
            let v0 = lrint(wr[i] / s0).clamp(-8, qmax);
            let v1 = if i + 1 < cols {
                let s1 = scale[o * ngroups + (i + 1) / gs];
                lrint(wr[i + 1] / s1).clamp(-8, qmax)
            } else {
                0
            };
            qr[i / 2] = ((v0 + 8) as u8) | (((v1 + 8) as u8) << 4);
            i += 2;
        }
    }
    (out, scale)
}

/// f32[rows,cols] -> int2-packed[rows,ceil(cols/4)] + per-row scale. Same fixed-floor
/// clamping as `pack_int4`, but to `[-2, qmax]` (the 2-bit container's range).
fn pack_int2(w: &[f32], q2: &mut [u8], scale: &mut [f32], rows: usize, cols: usize, bits: u8) {
    let qmax = (1i32 << (bits - 1)) - 1;
    let rb = cols.div_ceil(4);
    for o in 0..rows {
        let wr = &w[o * cols..(o + 1) * cols];
        let amax = wr.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let s = (amax / qmax as f32).max(1e-8);
        scale[o] = s;
        let qr = &mut q2[o * rb..(o + 1) * rb];
        let mut i = 0;
        while i < cols {
            let mut byte = 0u8;
            for k in 0..4 {
                if i + k >= cols {
                    break;
                }
                let v = lrint(wr[i + k] / s).clamp(-2, qmax);
                byte |= ((v + 2) as u8) << (k * 2);
            }
            qr[i / 4] = byte;
            i += 4;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small xorshift PRNG — deterministic, no extra crate, good enough for test fixtures.
    fn xorshift(seed: &mut u32) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 17;
        *seed ^= *seed << 5;
        ((*seed as f32 / u32::MAX as f32) - 0.5) * 2.0
    }

    fn random_matrix(rows: usize, cols: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..rows * cols).map(|_| xorshift(&mut s)).collect()
    }

    #[test]
    fn alloc_picks_format_by_bits() {
        assert!(matches!(QT::alloc(4, 8, 32, false).kind, QTKind::F32(_)));
        assert!(matches!(QT::alloc(4, 8, 8, false).kind, QTKind::I8 { .. }));
        assert!(matches!(QT::alloc(4, 8, 4, false).kind, QTKind::I4 { .. }));
        assert!(matches!(QT::alloc(4, 8, 2, false).kind, QTKind::I2 { .. }));
        // NOPACK: even a sub-8-bit format lands in the (unpacked) int8 container.
        assert!(matches!(QT::alloc(4, 8, 3, true).kind, QTKind::I8 { .. }));
    }

    #[test]
    fn f32_format_is_an_exact_passthrough() {
        let w = random_matrix(3, 5, 1);
        let mut t = QT::alloc(3, 5, 32, false);
        t.fill(&w);
        match &t.kind {
            QTKind::F32(data) => assert_eq!(data, &w),
            _ => panic!("expected F32"),
        }
    }

    #[test]
    fn from_packed_round_trips_int8_and_int4_containers() {
        for bits in [8u8, 4] {
            let rows = 5;
            let cols = 11; // not a multiple of 2 -> exercises int4's tail nibble too
            let w = random_matrix(rows, cols, bits as u32 * 17 + 1);
            let mut original = QT::alloc(rows, cols, bits, false);
            original.fill(&w);

            let (data, scale) = match &original.kind {
                QTKind::I8 { data, scale } => (data.iter().map(|&v| v as u8).collect::<Vec<u8>>(), scale.clone()),
                QTKind::I4 { data, scale } => (data.clone(), scale.clone()),
                _ => panic!("expected I8 or I4"),
            };

            let rebuilt = QT::from_packed(rows, cols, bits, data, scale).unwrap();
            for r in 0..rows {
                assert_eq!(rebuilt.row_f32(r), original.row_f32(r), "bits={bits} row={r}");
            }
        }
    }

    #[test]
    fn from_packed_grouped_round_trips_including_a_non_multiple_tail_group() {
        let rows = 4;
        let cols = 37; // group_size=8 -> 5 groups/row, last group only 5 real columns
        let gs = 8;
        let w = random_matrix(rows, cols, 99);
        let mut original = QT::alloc_grouped(rows, cols, 4, gs);
        original.fill(&w);

        let (data, scale) = match &original.kind {
            QTKind::I4Grouped { data, scale, .. } => (data.clone(), scale.clone()),
            _ => panic!("expected I4Grouped"),
        };
        let rebuilt = QT::from_packed_grouped(rows, cols, 4, data, scale, gs).unwrap();
        for r in 0..rows {
            assert_eq!(rebuilt.row_f32(r), original.row_f32(r), "row {r}");
        }
    }

    #[test]
    fn from_packed_grouped_does_not_force_grouping_onto_a_genuinely_per_row_tensor() {
        // A checkpoint-wide group_size>0 must NOT force a DIFFERENT tensor's plain per-row .qs
        // sidecar (e.g. embed_tokens/lm_head at io_bits=8, never grouped by the converter) to be
        // misread as grouped -- scale.len()==rows is what actually decides this, not group_size
        // alone. Caught for real by tests/convert_grouped_end_to_end.rs before this fix existed.
        let rows = 256;
        let cols = 128;
        let w = random_matrix(rows, cols, 5);
        let mut t = QT::alloc(rows, cols, 8, false); // io_bits=8 -> plain I8, per-row scale
        t.fill(&w);
        let (data, scale) = match &t.kind {
            QTKind::I8 { data, scale } => (data.iter().map(|&v| v as u8).collect::<Vec<u8>>(), scale.clone()),
            _ => panic!("expected I8"),
        };
        assert_eq!(scale.len(), rows);

        let rebuilt = QT::from_packed_grouped(rows, cols, 8, data, scale, 16).unwrap();
        assert!(matches!(rebuilt.kind, QTKind::I8 { .. }), "a per-row .qs must stay I8, not get reinterpreted as I4Grouped");
        for r in 0..rows {
            assert_eq!(rebuilt.row_f32(r), t.row_f32(r));
        }
    }

    #[test]
    fn from_packed_grouped_with_group_size_zero_falls_back_to_plain_from_packed() {
        let rows = 3;
        let cols = 6;
        let w = random_matrix(rows, cols, 7);
        let mut t = QT::alloc(rows, cols, 4, false);
        t.fill(&w);
        let (data, scale) = match &t.kind {
            QTKind::I4 { data, scale } => (data.clone(), scale.clone()),
            _ => panic!("expected I4"),
        };
        let rebuilt = QT::from_packed_grouped(rows, cols, 4, data, scale, 0).unwrap();
        assert!(matches!(rebuilt.kind, QTKind::I4 { .. }), "group_size=0 must produce a plain I4, not I4Grouped");
    }

    #[test]
    fn from_packed_grouped_rejects_a_mismatched_scale_length() {
        let Err(err) = QT::from_packed_grouped(4, 20, 4, vec![0u8; 4 * 10], vec![0.0; 3], 8) else {
            panic!("expected an error");
        };
        assert_eq!(err.scale_len, 3);
    }

    #[test]
    fn from_packed_rejects_a_byte_count_matching_no_format() {
        let Err(err) = QT::from_packed(4, 9, 8, vec![0u8; 7], vec![0.0; 4]) else {
            panic!("expected an error");
        };
        assert_eq!(err.rows, 4);
        assert_eq!(err.cols, 9);
    }

    #[test]
    fn from_packed_rejects_a_mismatched_scale_length() {
        let Err(err) = QT::from_packed(4, 9, 8, vec![0u8; 4 * 9], vec![0.0; 3]) else {
            panic!("expected an error");
        };
        assert_eq!(err.scale_len, 3);
    }

    #[test]
    fn resident_bytes_matches_format() {
        assert_eq!(QT::alloc(10, 100, 32, false).resident_bytes(), 10 * 100 * 4);
        assert_eq!(QT::alloc(10, 100, 8, false).resident_bytes(), 10 * 100 + 10 * 4);
        assert_eq!(QT::alloc(10, 100, 4, false).resident_bytes(), 10 * 50 + 10 * 4);
        assert_eq!(QT::alloc(10, 100, 2, false).resident_bytes(), 10 * 25 + 10 * 4);
        // group_size=25 -> 4 groups/row exactly.
        assert_eq!(QT::alloc_grouped(10, 100, 4, 25).resident_bytes(), 10 * 50 + 10 * 4 * 4);
    }

    /// The whole point of NOPACK (per glm.c: "per validare il packing"): quantizing the same
    /// weights at the same `bits` through the int8 path and the packed int4 path must yield
    /// bit-identical per-element integers and identical per-row scales — packing is just a
    /// storage transform, not a different quantization.
    #[test]
    fn int4_packing_is_bit_identical_to_the_unpacked_int8_container() {
        let rows = 6;
        let cols = 13; // odd, exercises the int4 tail-nibble path
        let w = random_matrix(rows, cols, 42);
        let bits = 4;

        let mut wide = QT::alloc(rows, cols, bits, true); // NOPACK -> int8 container
        wide.fill(&w);
        let mut packed = QT::alloc(rows, cols, bits, false); // -> packed int4
        packed.fill(&w);

        let (wide_data, wide_scale) = match &wide.kind {
            QTKind::I8 { data, scale } => (data, scale),
            _ => panic!("expected I8"),
        };
        let (packed_data, packed_scale) = match &packed.kind {
            QTKind::I4 { data, scale } => (data, scale),
            _ => panic!("expected I4"),
        };
        assert_eq!(wide_scale, packed_scale);

        for o in 0..rows {
            for i in 0..cols {
                let byte = packed_data[o * cols.div_ceil(2) + i / 2];
                let nibble = if i % 2 == 0 { byte & 0xF } else { byte >> 4 };
                let unpacked = nibble as i32 - 8;
                assert_eq!(
                    unpacked, wide_data[o * cols + i] as i32,
                    "row {o} col {i}: int4 nibble disagrees with int8 container"
                );
            }
        }
    }

    #[test]
    fn int2_packing_is_bit_identical_to_the_unpacked_int8_container() {
        let rows = 4;
        let cols = 11; // not a multiple of 4, exercises the int2 tail path
        let w = random_matrix(rows, cols, 7);
        let bits = 2;

        let mut wide = QT::alloc(rows, cols, bits, true);
        wide.fill(&w);
        let mut packed = QT::alloc(rows, cols, bits, false);
        packed.fill(&w);

        let (wide_data, wide_scale) = match &wide.kind {
            QTKind::I8 { data, scale } => (data, scale),
            _ => panic!("expected I8"),
        };
        let (packed_data, packed_scale) = match &packed.kind {
            QTKind::I2 { data, scale } => (data, scale),
            _ => panic!("expected I2"),
        };
        assert_eq!(wide_scale, packed_scale);

        for o in 0..rows {
            for i in 0..cols {
                let byte = packed_data[o * cols.div_ceil(4) + i / 4];
                let sh = (i % 4) * 2;
                let unpacked = ((byte >> sh) & 3) as i32 - 2;
                assert_eq!(
                    unpacked, wide_data[o * cols + i] as i32,
                    "row {o} col {i}: int2 nibble disagrees with int8 container"
                );
            }
        }
    }

    #[test]
    fn quantize_rows_dequantizes_close_to_original() {
        let rows = 5;
        let cols = 32;
        let w = random_matrix(rows, cols, 99);
        let mut t = QT::alloc(rows, cols, 8, false);
        t.fill(&w);
        let (data, scale) = match &t.kind {
            QTKind::I8 { data, scale } => (data, scale),
            _ => panic!("expected I8"),
        };
        for o in 0..rows {
            for i in 0..cols {
                let dq = data[o * cols + i] as f32 * scale[o];
                // one quantization step of slack: values in [-1,1], 8-bit -> step ~1/127.
                assert!((dq - w[o * cols + i]).abs() < 0.02, "row {o} col {i}: {dq} vs {}", w[o * cols + i]);
            }
        }
    }

    #[test]
    fn pack_int4_byte_layout_is_low_nibble_first_with_exact_rounding() {
        // amax=8 -> scale=8/7; 3.0/s=2.625->3, -8.0/s=-7.0 exactly -> nibbles 3,-7.
        let mut t = QT::alloc(1, 2, 4, false);
        t.fill(&[3.0, -8.0]);
        match &t.kind {
            QTKind::I4 { data, .. } => assert_eq!(data[0], (3 + 8) | ((-7i32 + 8) << 4) as u8),
            _ => panic!("expected I4"),
        }
    }

    #[test]
    fn pack_int4_pads_the_odd_tail_nibble_with_zero() {
        let mut t = QT::alloc(1, 1, 4, false);
        t.fill(&[3.5]); // amax=3.5 -> scale=0.5; 3.5/s=7.0 -> low nibble=7+8=15, high padded to 0+8=8
        match &t.kind {
            QTKind::I4 { data, .. } => assert_eq!(data[0], 15 | (8 << 4)),
            _ => panic!("expected I4"),
        }
    }

    #[test]
    fn quant_int4_grouped_matches_pack_int4_when_gs_covers_the_whole_row() {
        // A single group spanning every column IS per-row scaling -- same output as pack_int4.
        let (rows, cols) = (3, 10);
        let w = random_matrix(rows, cols, 5);
        let mut t = QT::alloc(rows, cols, 4, false);
        t.fill(&w);
        let (per_row_data, per_row_scale) = match &t.kind {
            QTKind::I4 { data, scale } => (data.clone(), scale.clone()),
            _ => panic!("expected I4"),
        };
        let (grouped_data, grouped_scale) = quant_int4_grouped(&w, rows, cols, 4, cols);
        assert_eq!(grouped_data, per_row_data);
        assert_eq!(grouped_scale, per_row_scale); // ngroups=1 -> same [rows] layout
    }

    #[test]
    fn quant_int4_grouped_uses_a_tighter_scale_for_the_low_magnitude_group() {
        // Row = [100, 0, 3.0, -8.0], gs=2 -> group 0 = [100, 0] (huge scale), group 1 = [3.0,
        // -8.0] (the exact pack_int4_byte_layout... case above). The byte for group 1 must be
        // identical to that per-row test's expected nibbles -- proves group 0's huge magnitude
        // doesn't leak into group 1's own scale.
        let (rows, cols, gs) = (1, 4, 2);
        let w = [100.0f32, 0.0, 3.0, -8.0];
        let (data, scale) = quant_int4_grouped(&w, rows, cols, 4, gs);
        assert_eq!(scale.len(), 2); // ngroups=2
        assert_eq!(scale[1], (8.0f32 / 7.0)); // group 1's own amax=8, qmax=7
        assert_eq!(data[1], (3 + 8) | ((-7i32 + 8) << 4) as u8);
    }

    #[test]
    fn quant_int4_grouped_trailing_partial_group_is_not_padding_diluted() {
        // cols=3, gs=2 -> group 0 = cols[0,1], group 1 = cols[2] alone (no padding column).
        // If padding with a fabricated zero leaked into the amax it wouldn't change amax here
        // (abs(0)=0), so this test's real assertion is just that a partial trailing group
        // quantizes using ONLY its own real element's magnitude, not garbage from group 0.
        let w = [1.0f32, 1.0, -9.0];
        let (_, scale) = quant_int4_grouped(&w, 1, 3, 4, 2);
        assert_eq!(scale.len(), 2);
        assert_eq!(scale[1], 9.0f32 / 7.0); // group 1's sole element is -9.0
    }

    // ---- MXFP4 (OCP Microscaling Formats v1.0): E2M1 elements + E8M0 block scale ----

    #[test]
    fn e2m1_decode_matches_the_spec_s_8_positive_magnitudes() {
        // OCP MX v1.0: E2M1's representable positive values are exactly {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
        let expected = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (code, &want) in expected.iter().enumerate() {
            assert_eq!(e2m1_decode(code as u8), want, "code {code}");
            assert_eq!(e2m1_decode(code as u8 | 0x8), -want, "code {code} (signed)");
        }
    }

    #[test]
    fn e2m1_encode_round_trips_every_exact_representable_value() {
        for code in 0u8..16 {
            let v = e2m1_decode(code);
            let back = e2m1_decode(e2m1_encode(v));
            assert_eq!(back, v, "code {code} -> {v} -> re-encoded differently");
        }
    }

    #[test]
    fn e2m1_encode_rounds_an_inexact_value_to_the_nearest_representable_point() {
        // 2.9 sits between 3 (diff .1) and 2 (diff .9) -> nearest is 3.
        assert_eq!(e2m1_decode(e2m1_encode(2.9)), 3.0);
        // 5.0 sits between 4 (diff 1) and 6 (diff 1) -> exact tie, keeps the lower magnitude (4).
        assert_eq!(e2m1_decode(e2m1_encode(5.0)), 4.0);
        // -2.9 must round to -3, not lose its sign.
        assert_eq!(e2m1_decode(e2m1_encode(-2.9)), -3.0);
    }

    #[test]
    fn e8m0_decode_is_a_power_of_two_biased_by_127() {
        assert_eq!(e8m0_decode(127), 1.0);
        assert_eq!(e8m0_decode(128), 2.0);
        assert_eq!(e8m0_decode(126), 0.5);
        assert_eq!(e8m0_decode(127 + 4), 16.0);
    }

    #[test]
    fn e8m0_decode_matches_powi_exactly_across_the_whole_valid_byte_range() {
        // Exhaustive, not spot-checked: every byte 0..=254 (255 is the reserved NaN code,
        // never produced by `choose_block_scale`) against the reference `2f32.powi` formula the
        // bit-trick replaced, catching any off-by-one in the exponent-field shift or the one
        // subnormal special case (`byte == 0`).
        for byte in 0u8..255 {
            let reference = 2f32.powi(byte as i32 - 127);
            assert_eq!(e8m0_decode(byte), reference, "byte {byte}: bit-trick {} != powi reference {reference}", e8m0_decode(byte));
        }
    }

    #[test]
    fn choose_block_scale_keeps_amax_within_e2m1_s_representable_range() {
        // amax=6.0 -> shared_exp=0 (scale=1), amax/scale=6.0 exactly hits E2M1's max point.
        let (byte, scale) = choose_block_scale(6.0);
        assert_eq!(byte, 127);
        assert_eq!(scale, 1.0);
        // amax=12.0 -> shared_exp=1 (scale=2), amax/scale=6.0, same exact fit one octave up.
        let (byte, scale) = choose_block_scale(12.0);
        assert_eq!(byte, 128);
        assert_eq!(scale, 2.0);
        // an all-zero block gets an arbitrary but valid scale, never a NaN/inf byte.
        let (byte, scale) = choose_block_scale(0.0);
        assert_eq!(scale, 1.0);
        assert_ne!(byte, 0xFF);
    }

    #[test]
    fn mxfp4_fill_and_row_f32_round_trip_within_one_quantization_step() {
        let rows = 3;
        let cols = 40; // > 32 -> exercises 2 blocks per row, the second only 8 wide
        let w = random_matrix(rows, cols, 123);
        let mut t = QT::alloc_mxfp4(rows, cols);
        t.fill(&w);
        for o in 0..rows {
            let row = t.row_f32(o);
            for k in 0..cols {
                // random_matrix draws from (-1,1); E2M1's coarsest step at that scale is 0.5
                // (denormal spacing) — generous but format-appropriate slack, not a fudge.
                assert!((row[k] - w[o * cols + k]).abs() < 0.5, "row {o} col {k}: {} vs {}", row[k], w[o * cols + k]);
            }
        }
    }

    #[test]
    fn mxfp4_uses_a_separate_scale_per_32_element_block_not_per_row() {
        // Two blocks in one row: first all at magnitude ~1 (scale picks shared_exp so amax=1
        // fits), second all at magnitude ~100 (needs a much larger scale). A single per-row
        // scale (like every other format here) would butcher one of the two; MXFP4 must not.
        let cols = 64;
        let mut w = vec![1.0f32; cols];
        for v in w[32..].iter_mut() {
            *v = 100.0;
        }
        let mut t = QT::alloc_mxfp4(1, cols);
        t.fill(&w);
        let row = t.row_f32(0);
        for (k, &v) in row[0..32].iter().enumerate() {
            assert!((v - 1.0).abs() < 0.5, "block 0 col {k}: {v}");
        }
        for (k, &v) in row[32..cols].iter().enumerate() {
            assert!((v - 100.0).abs() < 8.0, "block 1 col {}: {v}", 32 + k);
        }
    }

    #[test]
    fn mxfp4_resident_bytes_matches_the_per_block_not_per_row_scale_formula() {
        // Same 4-bit payload as I4 (cols/2 bytes/row), but the scale metadata is 1 byte per
        // 32-element BLOCK instead of I4's flat 4-byte-per-ROW f32 scale — cheaper per scale
        // value (finer granularity), but NOT cheaper in total once a row has more than 4 blocks
        // (32 * 4 = 128 columns): at cols=320 (10 blocks/row) MXFP4 spends 10 bytes/row on scale
        // vs I4's flat 4, more metadata for finer-grained accuracy, not less. The formula itself,
        // not "always smaller", is the thing worth asserting here.
        let rows = 10;
        let cols = 320; // exactly 10 blocks/row, so the formula check below has no rounding slop
        let mxfp4 = QT::alloc_mxfp4(rows, cols).resident_bytes();
        assert_eq!(mxfp4, rows * (cols / 2) + rows * (cols / 32));

        // At a short row width (<=128 cols, <=4 blocks), the finer scale genuinely does cost
        // less total metadata than I4's flat 4 bytes/row.
        let short_cols = 64; // 2 blocks/row -> 2 bytes/row of scale, vs I4's flat 4
        let mxfp4_short = QT::alloc_mxfp4(rows, short_cols).resident_bytes();
        let i4_short = QT::alloc(rows, short_cols, 4, false).resident_bytes();
        assert!(mxfp4_short < i4_short, "mxfp4 {mxfp4_short} should be cheaper than i4 {i4_short} for a short row (few blocks)");
    }

    /// `--mmap-experts`'s whole premise (see `expert_cache.rs`'s `mmap_load` module and
    /// `PERFORMANCE_KIMI_K3.md`) rests on `ByteStore::Mapped` dequantizing identically to
    /// `ByteStore::Owned` holding the exact same bytes — this is that proof, at the `quant.rs`
    /// level (independent of `safetensors.rs`/`expert_cache.rs`'s own mmap wiring).
    #[test]
    fn mxfp4_mapped_byte_store_matches_owned_row_f32() {
        let rows = 2;
        let cols = 40; // > 32 -> exercises 2 blocks per row, same shape as the fill/round-trip test above
        let w = random_matrix(rows, cols, 77);
        let mut owned = QT::alloc_mxfp4(rows, cols);
        owned.fill(&w);
        let (data_bytes, scale_bytes) = match &owned.kind {
            QTKind::MxFp4 { data, block_scale } => (data.as_slice().to_vec(), block_scale.as_slice().to_vec()),
            _ => panic!("expected MxFp4"),
        };

        // Write the exact same bytes to a temp file and mmap it -- the same shape of mapping
        // `expert_cache.rs`'s `mmap_load` module builds via `Shards::tensor_mmap_range`.
        let mut path = std::env::temp_dir();
        path.push(format!("rabbit_test_quant_mxfp4_mmap_{}_{}", std::process::id(), cols));
        let mut all_bytes = data_bytes.clone();
        all_bytes.extend_from_slice(&scale_bytes);
        std::fs::write(&path, &all_bytes).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        // SAFETY: `path` is a private, just-written temp file this test alone owns; nothing else
        // mutates it while mapped (matching the real invariant `safetensors.rs`'s doc documents).
        let mmap = Arc::new(unsafe { memmap2::Mmap::map(&file) }.unwrap());
        std::fs::remove_file(&path).ok(); // unlinking doesn't invalidate the existing mapping

        let mapped = QT::from_mmap_mxfp4(rows, cols, Arc::clone(&mmap), 0..data_bytes.len(), mmap, data_bytes.len()..all_bytes.len());

        for o in 0..rows {
            assert_eq!(mapped.row_f32(o), owned.row_f32(o), "row {o}");
        }
    }
}
