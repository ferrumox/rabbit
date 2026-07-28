//! Port of `convert_fp8_to_int4.py`'s `dequant`/`dequant_nvfp4`. The FP8 block-dequant branch
//! (`dt in ("F8_E4M3", ...)`, 128×128 block scale) is NOT re-implemented here — rabbit already
//! has that exact math (`crate::safetensors::dequant_fp8_blockscale`, used by `Shards::read_f32`
//! for the checkpoint rabbit already runs), so a source checkpoint's FP8 weights dequant through
//! that existing path unchanged. Only NVFP4 (NVIDIA `modelopt`'s 4-bit float format) is new: it
//! never appears in rabbit's own checkpoint, but a from-scratch converter needs to be able to
//! read it as a SOURCE format (the ROADMAP's `--repo` could point at an NVFP4 release, even
//! though `ROADMAP.md` ruled NVFP4 out as a rabbit-native TARGET format — reading one in is a
//! different, much smaller ask than emitting one).

use crate::safetensors::f8e4m3_to_f32;
use std::fmt;

/// FP4 e2m1: 1 sign + 2 exponent + 1 mantissa bit, 16 codes, magnitudes
/// `{0, .5, 1, 1.5, 2, 3, 4, 6}` — bit 3 is sign. Verified 1:1 against `ml_dtypes.float4_e2m1fn`
/// in the Python source's own `--selftest-nvfp4`.
const E2M1: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

#[derive(Debug)]
pub enum NvFp4Error {
    /// modelopt stores the small global scale and MULTIPLIES; compressed-tensors/llm-compressor
    /// stores its RECIPROCAL (large) and DIVIDES. A `weight_scale_2 >= 1.0` is almost certainly
    /// the latter convention fed into this modelopt-only decoder — refuse rather than silently
    /// corrupt every element by using the wrong operation (matches the Python source's own
    /// `assert`, an explicit design choice, not an incidental check).
    ScaleConventionMismatch { gscale: f32 },
    /// `weight_scale`'s column count didn't match `ceil(I/16)` — a swizzled/padded block-scale
    /// layout (e.g. CUTLASS/TensorRT's own internal layouts) this plain-modelopt decoder doesn't
    /// understand; refuse rather than silently misapply scales to the wrong elements.
    BlockScaleLayoutMismatch { got: usize, expected: usize },
    PackedLenMismatch { got: usize, expected: usize },
}

impl fmt::Display for NvFp4Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NvFp4Error::ScaleConventionMismatch { gscale } => write!(
                f,
                "nvfp4 weight_scale_2={gscale:.4e} >= 1.0 looks like the compressed-tensors reciprocal convention (which DIVIDES); this decoder assumes modelopt (which MULTIPLIES)"
            ),
            NvFp4Error::BlockScaleLayoutMismatch { got, expected } => {
                write!(f, "nvfp4 weight_scale has {got} columns, expected {expected} = ceil(I/16); unexpected (swizzled/padded?) layout")
            }
            NvFp4Error::PackedLenMismatch { got, expected } => write!(f, "nvfp4 packed weight has {got} bytes, expected {expected} = O*ceil(I/2)"),
        }
    }
}

impl std::error::Error for NvFp4Error {}

/// Dequantizes one NVIDIA-modelopt NVFP4 weight tensor to `f32[o, i]`, row-major.
///
/// - `packed`: `[o, i/2]` raw bytes, two e2m1 nibbles each — LOW nibble = even column, HIGH
///   nibble = odd column (`compressed_tensors`/vLLM packed order).
/// - `bscale_raw`: the `.weight_scale` sidecar's raw F8_E4M3 bytes, `[o, ceil(i/16)]` — one
///   per-16-element block scale, decoded here via the SAME `f8e4m3_to_f32` rabbit's FP8 path
///   already uses (this is a plain per-element decode, not `dequant_fp8_blockscale` — a block
///   scale has no block-scale of its own).
/// - `gscale`: the `.weight_scale_2` sidecar, a single per-tensor F32 scalar.
///
/// `W[row,col] = e2m1_lut[nibble] * f8_block_scale[row, col/16] * gscale` (modelopt's own
/// dequant convention — MULTIPLIES both scales, see `NvFp4Error::ScaleConventionMismatch`'s doc
/// for the footgun this guards against).
pub fn dequant_nvfp4(packed: &[u8], o: usize, i: usize, bscale_raw: &[u8], gscale: f32) -> Result<Vec<f32>, NvFp4Error> {
    const GS: usize = 16;
    if gscale >= 1.0 {
        return Err(NvFp4Error::ScaleConventionMismatch { gscale });
    }
    let half_i = i.div_ceil(2);
    if packed.len() != o * half_i {
        return Err(NvFp4Error::PackedLenMismatch { got: packed.len(), expected: o * half_i });
    }
    let nb = i.div_ceil(GS);
    if bscale_raw.len() != o * nb {
        return Err(NvFp4Error::BlockScaleLayoutMismatch { got: bscale_raw.len(), expected: o * nb });
    }
    let bscale: Vec<f32> = bscale_raw.iter().map(|&b| f8e4m3_to_f32(b)).collect();

    let mut out = vec![0f32; o * i];
    for row in 0..o {
        let prow = &packed[row * half_i..(row + 1) * half_i];
        let srow = &bscale[row * nb..(row + 1) * nb];
        for col in 0..i {
            let byte = prow[col / 2];
            let nib = if col % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F };
            out[row * i + col] = E2M1[nib as usize] * srow[col / GS] * gscale;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2m1_lut_matches_the_16_documented_codes() {
        let expect = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];
        assert_eq!(E2M1, expect);
    }

    #[test]
    fn nibble_order_is_low_equals_even_column_high_equals_odd() {
        // O=1, I=2, one packed byte. Low nibble (code 5 -> 3.0) is column 0, high nibble
        // (code 2 -> 1.0) is column 1. One 16-wide block (I<16), one trivial block scale = 1.0
        // (f8e4m3 byte 0x38 decodes to 1.0), gscale small and < 1.0.
        let packed = [(2u8 << 4) | 5u8]; // low nibble = 5 -> 3.0, high nibble = 2 -> 1.0
        let bscale_raw = [0x38u8]; // f8e4m3 for 1.0 (sign0 exp=7(0b0111) mant=0 -> 0011_1000)
        let gscale = 0.1f32;
        let out = dequant_nvfp4(&packed, 1, 2, &bscale_raw, gscale).unwrap();
        assert!((out[0] - 3.0 * 1.0 * gscale).abs() < 1e-6, "{out:?}");
        assert!((out[1] - 1.0 * 1.0 * gscale).abs() < 1e-6, "{out:?}");
    }

    #[test]
    fn each_16_wide_block_uses_its_own_scale() {
        // O=1, I=32 -> 2 blocks of 16. All nibbles code for 1.0 (code 2). Block 0 scale=2.0,
        // block 1 scale=4.0 (both exact f8e4m3 values) -> column 0 dequants to gscale*2.0,
        // column 16 dequants to gscale*4.0.
        let code = 2u8; // -> 1.0
        let byte = (code << 4) | code;
        let packed = vec![byte; 16]; // 32 columns / 2 per byte = 16 bytes
        let bscale_raw = [0x40u8, 0x48u8]; // f8e4m3: 0x40=2.0, 0x48=4.0
        let gscale = 0.01f32;
        let out = dequant_nvfp4(&packed, 1, 32, &bscale_raw, gscale).unwrap();
        assert!((out[0] - 1.0 * 2.0 * gscale).abs() < 1e-6, "{out:?}");
        assert!((out[16] - 1.0 * 4.0 * gscale).abs() < 1e-6, "{out:?}");
    }

    #[test]
    fn gscale_at_or_above_one_is_rejected_as_the_wrong_convention() {
        let packed = [0u8];
        let bscale_raw = [0x38u8];
        assert!(matches!(dequant_nvfp4(&packed, 1, 2, &bscale_raw, 1.0), Err(NvFp4Error::ScaleConventionMismatch { .. })));
        assert!(matches!(dequant_nvfp4(&packed, 1, 2, &bscale_raw, 5.0), Err(NvFp4Error::ScaleConventionMismatch { .. })));
    }

    #[test]
    fn mismatched_block_scale_column_count_is_rejected() {
        let packed = [0u8];
        let bscale_raw = [0x38u8, 0x38u8]; // 2 columns, but I=2 -> expected ceil(2/16)=1
        assert!(matches!(
            dequant_nvfp4(&packed, 1, 2, &bscale_raw, 0.1),
            Err(NvFp4Error::BlockScaleLayoutMismatch { got: 2, expected: 1 })
        ));
    }
}
