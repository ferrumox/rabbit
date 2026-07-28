//! `--report`: per-tensor quantization error, measured directly (dequant the just-written bytes,
//! compare against the original f32 values) rather than estimated — the cheap first step from
//! `resumen-conversor-rabbit.md`'s design discussion ("comparar cada tensor original vs. su
//! versión dequantizada... sacar un reporte tipo 'capa X, tensor Y: error relativo Z%'").
//!
//! Deliberately NOT a perplexity/benchmark measurement — this only says how much a SINGLE
//! tensor's own values moved, not how that error propagates through 27+ layers of matmuls and
//! attention/MoE routing to affect actual generated text (see
//! `examples/kimi_quality_check.rs` for a downstream, generation-level check). Still useful:
//! cheap (no model forward pass, just the tensor already in hand during conversion), and the
//! per-bucket breakdown is exactly the real data `resumen-conversor-rabbit.md` wanted before
//! trusting per-tensor-class bit-width choices to intuition alone — especially for MoE, where
//! thousands of routed-expert tensors can have very non-uniform error.

use crate::convert::classify::TensorKind;
use crate::quant::QT;
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct TensorError {
    pub name: String,
    pub kind: TensorKind,
    pub bits: u8,
    /// Mean absolute relative error: `sum(|orig-dequant|) / sum(|orig|)` over every element —
    /// one scalar per tensor, cheap to sort/threshold/aggregate. Not RMS/max — a single large
    /// outlier element skews RMS/max hard in a way this project doesn't have a documented
    /// tolerance for yet; mean relative error is the same statistic `--selftest`'s existing
    /// FP8/NVFP4 self-checks already use elsewhere in this codebase (see `glm52::convert`'s own
    /// doc comments), so a reader has a precedent to compare against.
    pub rel_error: f32,
}

#[derive(Default, Debug, Clone)]
pub struct QuantizationReport {
    pub per_tensor: Vec<TensorError>,
}

impl QuantizationReport {
    /// Mean relative error, grouped by bucket — the first thing worth reading: which tensor
    /// CLASSES are paying the most for their bit-width, not any one tensor in isolation.
    pub fn by_bucket(&self) -> BTreeMap<TensorKind, (usize, f32)> {
        let mut sums: BTreeMap<TensorKind, (usize, f32)> = BTreeMap::new();
        for t in &self.per_tensor {
            let e = sums.entry(t.kind).or_default();
            e.0 += 1;
            e.1 += t.rel_error;
        }
        sums.into_iter().map(|(k, (n, sum))| (k, (n, sum / n.max(1) as f32))).collect()
    }

    /// The `n` tensors with the highest relative error — the concrete follow-up question
    /// `resumen-conversor-rabbit.md` wanted an answer to ("¿qué experts concretos sufren más?"),
    /// not just a bucket-level average that could hide a handful of badly-hit tensors.
    pub fn worst(&self, n: usize) -> Vec<&TensorError> {
        let mut v: Vec<&TensorError> = self.per_tensor.iter().collect();
        v.sort_by(|a, b| b.rel_error.total_cmp(&a.rel_error));
        v.truncate(n);
        v
    }

    pub fn summary(&self) -> String {
        let mut out = format!("quantization error report: {} tensors measured\n", self.per_tensor.len());
        out.push_str("  by bucket (mean relative error):\n");
        for (kind, (n, mean)) in self.by_bucket() {
            out.push_str(&format!("    {kind:?}: {mean:.4} (n={n})\n"));
        }
        out.push_str("  worst 10 individual tensors:\n");
        for t in self.worst(10) {
            out.push_str(&format!("    {:.4}  {} ({:?}, {}bit)\n", t.rel_error, t.name, t.kind, t.bits));
        }
        out
    }
}

/// Dequantizes `data`/`scale` (the just-written packed bytes) and compares against `original`
/// (the pre-quantization f32 values) — mean absolute relative error. Reconstructs a throwaway
/// `QT` via `from_packed`/`from_packed_grouped` purely to reuse `row_f32`'s existing, already-
/// tested dequant logic rather than duplicating nibble-unpacking here a third time.
pub fn measure_error(original: &[f32], data: &[u8], scale: &[f32], rows: usize, cols: usize, bits: u8, group_size: usize) -> f32 {
    let qt = match QT::from_packed_grouped(rows, cols, bits, data.to_vec(), scale.to_vec(), group_size) {
        Ok(qt) => qt,
        // A shape this function's own caller (convert_shard) didn't produce correctly is a
        // logic bug worth seeing loudly during a --report run, not a silently-skipped tensor.
        Err(e) => panic!("measure_error: {e}"),
    };
    let mut num = 0f64;
    let mut den = 0f64;
    for r in 0..rows {
        let deq = qt.row_f32(r);
        let orig_row = &original[r * cols..(r + 1) * cols];
        for (&a, &b) in orig_row.iter().zip(&deq) {
            num += (a - b).abs() as f64;
            den += a.abs() as f64;
        }
    }
    (num / den.max(1e-12)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::QT;

    fn random_vec(n: usize, seed: &mut u32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed ^= *seed << 13;
                *seed ^= *seed >> 17;
                *seed ^= *seed << 5;
                ((*seed as f32 / u32::MAX as f32) - 0.5) * 2.0
            })
            .collect()
    }

    #[test]
    fn measure_error_is_zero_for_an_exact_f32_round_trip_shaped_input() {
        // int8 at a huge dynamic range (bits=8, values already near the quantization grid)
        // isn't exactly zero error in general, so instead check the DIRECTLY testable
        // invariant: measuring a tensor against ITS OWN dequantized self is exactly zero.
        let (rows, cols) = (4, 16);
        let mut seed = 7u32;
        let w = random_vec(rows * cols, &mut seed);
        let mut t = QT::alloc(rows, cols, 8, false);
        t.fill(&w);
        let (data, scale) = match &t.kind {
            crate::quant::QTKind::I8 { data, scale } => (data.iter().map(|&v| v as u8).collect::<Vec<u8>>(), scale.clone()),
            _ => unreachable!(),
        };
        let mut dequantized_self = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            dequantized_self.extend(t.row_f32(r));
        }
        let err = measure_error(&dequantized_self, &data, &scale, rows, cols, 8, 0);
        assert_eq!(err, 0.0);
    }

    #[test]
    fn measure_error_is_positive_and_small_for_a_real_int4_round_trip() {
        let (rows, cols) = (4, 32);
        let mut seed = 8u32;
        let w = random_vec(rows * cols, &mut seed);
        let mut t = QT::alloc(rows, cols, 4, false);
        t.fill(&w);
        let (data, scale) = match &t.kind {
            crate::quant::QTKind::I4 { data, scale } => (data.clone(), scale.clone()),
            _ => unreachable!(),
        };
        let err = measure_error(&w, &data, &scale, rows, cols, 4, 0);
        assert!(err > 0.0, "int4 quantization of random data must have SOME error");
        assert!(err < 0.5, "error={err} is implausibly large for int4 on unit-scale random data");
    }

    #[test]
    fn grouped_quantization_measures_lower_error_than_per_row_on_wide_rows() {
        // Rows with a wide dynamic-range spread within the row are exactly where grouped
        // scaling is supposed to help -- construct one: most of the row near zero, one
        // high-magnitude burst confined to a narrow span.
        let cols = 256;
        let mut w = vec![0.01f32; cols];
        for v in w.iter_mut().take(16) {
            *v = 50.0;
        }
        let rows = 1;

        let mut per_row = QT::alloc(rows, cols, 4, false);
        per_row.fill(&w);
        let (pr_data, pr_scale) = match &per_row.kind {
            crate::quant::QTKind::I4 { data, scale } => (data.clone(), scale.clone()),
            _ => unreachable!(),
        };
        let per_row_err = measure_error(&w, &pr_data, &pr_scale, rows, cols, 4, 0);

        let mut grouped = QT::alloc_grouped(rows, cols, 4, 32);
        grouped.fill(&w);
        let (g_data, g_scale) = match &grouped.kind {
            crate::quant::QTKind::I4Grouped { data, scale, .. } => (data.clone(), scale.clone()),
            _ => unreachable!(),
        };
        let grouped_err = measure_error(&w, &g_data, &g_scale, rows, cols, 4, 32);

        assert!(grouped_err < per_row_err, "grouped ({grouped_err}) should beat per-row ({per_row_err}) when one row mixes wildly different magnitudes");
    }

    #[test]
    fn by_bucket_averages_correctly_and_worst_sorts_descending() {
        let mut report = QuantizationReport::default();
        report.per_tensor.push(TensorError { name: "a".into(), kind: TensorKind::Q, bits: 4, rel_error: 0.1 });
        report.per_tensor.push(TensorError { name: "b".into(), kind: TensorKind::Q, bits: 4, rel_error: 0.3 });
        report.per_tensor.push(TensorError { name: "c".into(), kind: TensorKind::X, bits: 4, rel_error: 0.5 });

        let by_bucket = report.by_bucket();
        assert_eq!(by_bucket[&TensorKind::Q], (2, 0.2));
        assert_eq!(by_bucket[&TensorKind::X], (1, 0.5));

        let worst = report.worst(2);
        assert_eq!(worst[0].name, "c");
        assert_eq!(worst[1].name, "b");
    }
}
