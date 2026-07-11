//! Port of the `QT` container + `qt_alloc`/`qt_fill`/`quantize_rows`/`pack_int4`/`pack_int2`
//! from `glm.c` — per-row symmetric quantization to int8 (1 byte/param), packed int4
//! (2 values/byte) or packed int2 (4 values/byte), plus the f32 passthrough format.
//!
//! `bits` and format are independent: `bits` sets the quantization range (`qmax`) and divides
//! the container into three storage tiers, but e.g. `bits=3` still lands in the int4 tier
//! (`>=3`) using less than its full nibble range — this is what colibri's `NOPACK` env var
//! exploits to store sub-8-bit values in an unpacked int8 container for validating that the
//! packed and unpacked encodings agree bit-for-bit (see the tests below).
//!
//! Rounding: C's `lrintf` respects the default IEEE-754 rounding mode (round-to-nearest,
//! ties-to-even — "banker's rounding"), NOT `f32::round`'s round-half-away-from-zero. Using
//! `round_ties_even` here is required for token-exact parity with the C oracle, not a style
//! preference.

#[inline]
fn lrint(x: f32) -> i32 {
    x.round_ties_even() as i32
}

/// A quantized `[rows, cols]` weight matrix in one of four formats, chosen by `bits`.
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
}

impl QT {
    /// Chooses a format from `bits`, mirroring `qt_alloc`: `>=16` -> f32, `>=5` (or
    /// `nopack`, colibri's `NOPACK=1`) -> int8, `>=3` -> int4, else -> int2.
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

    /// Fills from row-major f32 weights `w[rows*cols]`, quantizing per the chosen format.
    pub fn fill(&mut self, w: &[f32]) {
        assert_eq!(w.len(), self.rows * self.cols);
        match &mut self.kind {
            QTKind::F32(data) => data.copy_from_slice(w),
            QTKind::I8 { data, scale } => quantize_rows(w, data, scale, self.rows, self.cols, self.bits),
            QTKind::I4 { data, scale } => pack_int4(w, data, scale, self.rows, self.cols, self.bits),
            QTKind::I2 { data, scale } => pack_int2(w, data, scale, self.rows, self.cols, self.bits),
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
        }
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
    fn resident_bytes_matches_format() {
        assert_eq!(QT::alloc(10, 100, 32, false).resident_bytes(), 10 * 100 * 4);
        assert_eq!(QT::alloc(10, 100, 8, false).resident_bytes(), 10 * 100 + 10 * 4);
        assert_eq!(QT::alloc(10, 100, 4, false).resident_bytes(), 10 * 50 + 10 * 4);
        assert_eq!(QT::alloc(10, 100, 2, false).resident_bytes(), 10 * 25 + 10 * 4);
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
}
