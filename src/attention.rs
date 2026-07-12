//! Port of `attention()` (glm.c:1006) — MLA attention with a compressed KV cache, plus the
//! DSA lightning indexer. Two attention paths, chosen by `Absorb`:
//!
//! - **Absorbed** (DeepSeek-style, used for small `S` i.e. decode): folds
//!   `q_nope · k_nope_t = q_nope · (W_K L_t) = (W_K^T q_nope) · L_t` so scoring against every
//!   cached token costs `O(kv_lora)` instead of reconstructing `k_nope_t` per token.
//! - **Dense**: reconstructs `k_nope`/`v` for every token from the compressed latent in one
//!   batched matmul, then scores directly — used for prefill (large `S`), where the batched
//!   matmul amortizes better than per-token absorption.
//!
//! Both paths must be mathematically equivalent (same attention, two algebraic
//! rearrangements) — see the `absorbed_path_matches_dense_reconstruction_path` test.
//!
//! DSA selection is pulled out of `attention()` into `dsa_select()` and passed in via the
//! `Dsa` enum instead of being computed via mutable model-global state (`m->dsa_sel` in the
//! C): a "full" indexer layer computes a fresh `Selection` (`Dsa::Compute`, returned to the
//! caller for reuse), a "shared" layer reuses one computed earlier this forward pass
//! (`Dsa::Reuse`). The cross-layer decision of which is which belongs to the layer loop
//! (`layers_forward`, a later phase) — this module only needs the primitive.

use crate::config::Cfg;
use crate::kernels::{matmul_qt, qt_addrow, qt_matvec_rows};
use crate::model::{AttnWeights, DsaWeights};

pub fn rmsnorm(v: &mut [f32], w: &[f32], eps: f32) {
    let n = v.len() as f64;
    let ms: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum();
    let r = 1.0f32 / (((ms / n) as f32) + eps).sqrt();
    for (vi, &wi) in v.iter_mut().zip(w) {
        *vi *= r * wi;
    }
}

/// Classic mean/variance LayerNorm — used by the DSA indexer's `k_norm`.
pub fn layernorm(v: &mut [f32], w: &[f32], b: &[f32], eps: f32) {
    let n = v.len() as f64;
    let mu: f64 = v.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var: f64 = v.iter().map(|&x| { let d = x as f64 - mu; d * d }).sum::<f64>() / n;
    let r = 1.0f32 / (var as f32 + eps).sqrt();
    for i in 0..v.len() {
        v[i] = ((v[i] as f64 - mu) as f32) * r * w[i] + b[i];
    }
}

pub fn softmax(x: &mut [f32]) {
    let m = x.iter().fold(-1e30f32, |acc, &v| acc.max(v));
    let mut sum = 0f32;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// GLM's interleaved partial RoPE: reads adjacent pairs `(in[2j], in[2j+1])`, writes the
/// rotated components to the de-interleaved halves `v[j]`/`v[half+j]`. `v.len()` doubles as
/// `qk_rope` — every call site passes exactly a `qk_rope`-wide slice, so there's no separate
/// dimension parameter to keep in sync with the slice length.
pub fn rope_interleave(v: &mut [f32], pos: usize, theta: f32) {
    let qk_rope = v.len();
    let half = qk_rope / 2;
    let input = v.to_vec();
    for j in 0..half {
        let inv = theta.powf(-2.0 * j as f32 / qk_rope as f32);
        let ang = pos as f32 * inv;
        let (sn, cs) = ang.sin_cos();
        let a = input[2 * j];
        let b = input[2 * j + 1];
        v[j] = a * cs - b * sn;
        v[half + j] = b * cs + a * sn;
    }
}

/// Compressed MLA KV cache for one layer: `L` (kv-latent, rmsnorm'd) and `R` (rope key,
/// shared across heads), one row per absolute position, appended in order.
pub struct KvCache {
    kv_lora: usize,
    qk_rope: usize,
    l: Vec<f32>,
    r: Vec<f32>,
}

impl KvCache {
    pub fn new(kv_lora: usize, qk_rope: usize) -> KvCache {
        KvCache { kv_lora, qk_rope, l: Vec::new(), r: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.l.len().checked_div(self.kv_lora).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn push(&mut self, l_row: &[f32], r_row: &[f32]) {
        self.l.extend_from_slice(l_row);
        self.r.extend_from_slice(r_row);
    }

    fn l_row(&self, pos: usize) -> &[f32] {
        &self.l[pos * self.kv_lora..(pos + 1) * self.kv_lora]
    }

    fn r_row(&self, pos: usize) -> &[f32] {
        &self.r[pos * self.qk_rope..(pos + 1) * self.qk_rope]
    }

    /// contiguous `L` rows for positions `[from, to)` — one matmul input for reconstructing
    /// `k_nope`/`v` over a whole range at once (the dense path's whole point). `pub(crate)`
    /// (widened from private) so `kv_session.rs`'s save path can read out exactly the new rows
    /// a save needs without exposing `l` itself.
    pub(crate) fn l_range(&self, from: usize, to: usize) -> &[f32] {
        &self.l[from * self.kv_lora..to * self.kv_lora]
    }

    /// `l_range`'s counterpart for `R`.
    pub(crate) fn r_range(&self, from: usize, to: usize) -> &[f32] {
        &self.r[from * self.qk_rope..to * self.qk_rope]
    }

    pub(crate) fn kv_lora(&self) -> usize {
        self.kv_lora
    }

    pub(crate) fn qk_rope(&self) -> usize {
        self.qk_rope
    }

    /// Reconstructs a `KvCache` from raw saved rows — `kv_session.rs`'s load path, the mirror
    /// of `new` (which starts empty) for restoring one that already has history.
    pub(crate) fn from_raw(kv_lora: usize, qk_rope: usize, l: Vec<f32>, r: Vec<f32>) -> KvCache {
        KvCache { kv_lora, qk_rope, l, r }
    }
}

/// DSA lightning indexer's per-layer key cache (`Ic` in the C): one `index_hd`-wide row per
/// absolute position, same append-in-order shape as `KvCache`.
pub struct DsaCache {
    hd: usize,
    k: Vec<f32>,
}

impl DsaCache {
    pub fn new(hd: usize) -> DsaCache {
        DsaCache { hd, k: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.k.len().checked_div(self.hd).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn push(&mut self, row: &[f32]) {
        self.k.extend_from_slice(row);
    }

    fn row(&self, pos: usize) -> &[f32] {
        &self.k[pos * self.hd..(pos + 1) * self.hd]
    }

    pub(crate) fn hd(&self) -> usize {
        self.hd
    }

    /// `KvCache::l_range`'s counterpart for the DSA indexer's key cache.
    pub(crate) fn k_range(&self, from: usize, to: usize) -> &[f32] {
        &self.k[from * self.hd..to * self.hd]
    }

    /// Reconstructs a `DsaCache` from raw saved rows — mirrors `KvCache::from_raw`.
    pub(crate) fn from_raw(hd: usize, k: Vec<f32>) -> DsaCache {
        DsaCache { hd, k }
    }
}

/// Per-token selected absolute positions. An empty list means "no restriction" — the
/// consumer falls back to the full dense causal range, exactly like `dnsel[s]==0` in the C
/// (this is how the indexer is a no-op when `index_topk >= seq_len`).
#[derive(Clone, Debug, PartialEq)]
pub struct Selection {
    pub per_token: Vec<Vec<u32>>,
}

/// How this `attention()` call should handle the DSA indexer.
pub enum Dsa<'a> {
    /// Model has no DSA indexer, or this layer doesn't restrict attention.
    Off,
    /// A "full" indexer layer: compute k_idx for the new tokens and (if the context is long
    /// enough, or `force`) a fresh top-k selection. Returned to the caller for `Reuse` by
    /// later "shared" layers this same forward pass.
    Compute { weights: &'a DsaWeights, cache: &'a mut DsaCache, force: bool },
    /// A "shared" indexer layer: reuse a selection a "full" layer already computed.
    Reuse(&'a Selection),
}

/// Whether to use the weight-absorbed decode path or the dense reconstruction path.
/// `Auto` matches colibri's default heuristic (`g_absorb<0`): absorbed for `S<=4`.
pub enum Absorb {
    Auto,
    Always,
    Never,
}

/// Computes the DSA lightning indexer's key embeddings for the new tokens (always) and a
/// top-k selection (only when the context already exceeds `index_topk`, or `force`).
/// Mirrors the C's parameter list 1:1 — bundling these into a struct would just move the
/// same fields one level of indirection away, with only one real call site (`attention`).
#[allow(clippy::too_many_arguments)]
pub fn dsa_select(
    cfg: &Cfg,
    dsa: &DsaWeights,
    cache: &mut DsaCache,
    qr: &[f32],
    x: &[f32],
    s: usize,
    pos_base: usize,
    force: bool,
) -> Selection {
    let d = cfg.hidden as usize;
    let q_lora = cfg.q_lora as usize;
    let nh = cfg.index_nh as usize;
    let hd = cfg.index_hd as usize;
    let dtopk = cfg.index_topk as usize;

    for si in 0..s {
        let xs = &x[si * d..(si + 1) * d];
        let pos = pos_base + si;
        let mut kd = vec![0f32; hd];
        matmul_qt(&mut kd, xs, &dsa.wk, 1);
        layernorm(&mut kd, &dsa.k_norm_w, &dsa.k_norm_b, 1e-6);
        rope_interleave(&mut kd, pos, cfg.theta);
        cache.push(&kd);
    }

    let mut per_token = Vec::with_capacity(s);
    for si in 0..s {
        let pos = pos_base + si;
        let nk = pos + 1;
        if nk <= dtopk && !force {
            per_token.push(Vec::new());
            continue;
        }
        let keep = nk.min(dtopk);

        let qr_s = &qr[si * q_lora..(si + 1) * q_lora];
        let mut qi = vec![0f32; nh * hd];
        matmul_qt(&mut qi, qr_s, &dsa.wq_b, 1);
        for h in 0..nh {
            rope_interleave(&mut qi[h * hd..(h + 1) * hd], pos, cfg.theta);
        }

        let mut w32 = vec![0f32; nh];
        matmul_qt(&mut w32, &x[si * d..(si + 1) * d], &dsa.weights_proj, 1);

        let wsc = 1.0 / (nh as f32).sqrt();
        let rs = 1.0 / (hd as f32).sqrt();
        let mut isc = vec![0f32; nk];
        for (t, isc_t) in isc.iter_mut().enumerate() {
            let kt = cache.row(t);
            let mut a = 0f32;
            for h in 0..nh {
                let qh = &qi[h * hd..(h + 1) * hd];
                let mut d0: f32 = qh.iter().zip(kt).map(|(&a, &b)| a * b).sum();
                d0 *= rs;
                if d0 > 0.0 {
                    a += w32[h] * d0;
                }
            }
            *isc_t = a * wsc;
        }

        // threshold via a full descending sort, then a two-pass position-order scan so ties
        // don't need a stable sort: collect strictly-greater first, then exactly-equal to
        // fill the remainder. Both passes walk t ascending, so the result is always sorted.
        let mut sorted = isc.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let thr = sorted[keep - 1];

        let mut sel = Vec::with_capacity(keep);
        for (t, &v) in isc.iter().enumerate() {
            if v > thr {
                sel.push(t as u32);
                if sel.len() == keep {
                    break;
                }
            }
        }
        if sel.len() < keep {
            for (t, &v) in isc.iter().enumerate() {
                if v == thr {
                    sel.push(t as u32);
                    if sel.len() == keep {
                        break;
                    }
                }
            }
        }
        per_token.push(sel);
    }

    Selection { per_token }
}

/// MLA attention with a compressed KV cache, on new tokens `x[S,hidden]` starting at absolute
/// position `pos_base`. Writes `out[S,hidden]`. Returns the freshly computed `Selection` when
/// `dsa` was `Dsa::Compute` (for the caller to `Reuse` at later "shared" layers), else `None`.
/// Mirrors `attention()`'s C parameter list; see `dsa_select`'s doc for why it stays flat.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    cfg: &Cfg,
    w: &AttnWeights,
    kv: &mut KvCache,
    x: &[f32],
    s: usize,
    pos_base: usize,
    dsa: Dsa<'_>,
    absorb: Absorb,
    out: &mut [f32],
) -> Option<Selection> {
    let h = cfg.n_heads as usize;
    let d = cfg.hidden as usize;
    let qh = cfg.qk_head as usize;
    let vh = cfg.v_head as usize;
    let qk_nope = cfg.qk_nope as usize;
    let qk_rope = cfg.qk_rope as usize;
    let kv_lora = cfg.kv_lora as usize;
    let q_lora = cfg.q_lora as usize;
    // `st0`/`m->kv_start[layer]` in the C: always 0 here. It's only ever mutated for the MTP
    // KV row (`mtp_draft`/`mtp_absorb`, out of scope — MTP restarts its own decode-only
    // window there), so for every layer this port actually runs, the causal window always
    // starts at position 0. Earlier this derived `kv_start` from `kv.len()` instead (treating
    // it as "how many positions this call is continuing from"), which made a decode step
    // attend only to itself instead of the full history — caught by
    // `greedy_generation_from_prompt_reproduces_the_oracle_continuation` diverging right
    // after the first token.
    let kv_start = 0;

    let mut ctx = vec![0f32; s * h * vh];
    let mut q = vec![0f32; s * h * qh];
    let mut qr_all = vec![0f32; s * q_lora];

    // 1) per new token: roped query + normed latent + roped shared rope-key, into the cache.
    for si in 0..s {
        let xs = &x[si * d..(si + 1) * d];
        let pos = pos_base + si;

        let qresid = &mut qr_all[si * q_lora..(si + 1) * q_lora];
        matmul_qt(qresid, xs, &w.q_a, 1);
        rmsnorm(qresid, &w.q_a_ln, cfg.eps);

        let qfull = &mut q[si * h * qh..(si + 1) * h * qh];
        matmul_qt(qfull, qresid, &w.q_b, 1);
        for hh in 0..h {
            let start = hh * qh + qk_nope;
            rope_interleave(&mut qfull[start..start + qk_rope], pos, cfg.theta);
        }

        let mut comp = vec![0f32; kv_lora + qk_rope];
        matmul_qt(&mut comp, xs, &w.kv_a, 1);
        let mut l_row = comp[0..kv_lora].to_vec();
        rmsnorm(&mut l_row, &w.kv_a_ln, cfg.eps);
        let mut r_row = comp[kv_lora..kv_lora + qk_rope].to_vec();
        rope_interleave(&mut r_row, pos, cfg.theta);
        kv.push(&l_row, &r_row);
    }

    // 2) DSA selection (fresh, reused, or none).
    let mut computed_fresh = false;
    let selection: Option<Selection> = match dsa {
        Dsa::Off => None,
        Dsa::Compute { weights, cache, force } => {
            computed_fresh = true;
            Some(dsa_select(cfg, weights, cache, &qr_all, x, s, pos_base, force))
        }
        Dsa::Reuse(sel) => Some(sel.clone()),
    };
    let positions_for = |si: usize, pos: usize| -> Vec<usize> {
        match selection.as_ref().map(|sel| &sel.per_token[si]) {
            Some(list) if !list.is_empty() => list.iter().map(|&t| t as usize).collect(),
            _ => (kv_start..=pos).collect(),
        }
    };

    // 3) attention proper: weight-absorbed decode path, or dense reconstruction path.
    let use_absorb = matches!(absorb, Absorb::Always) || matches!(absorb, Absorb::Auto if s <= 4);
    if use_absorb && kv_lora <= 512 {
        for si in 0..s {
            let pos = pos_base + si;
            let positions = positions_for(si, pos);
            let nt = positions.len();
            for hh in 0..h {
                let qp = &q[si * h * qh + hh * qh..si * h * qh + (hh + 1) * qh];
                let qr = &qp[qk_nope..qk_nope + qk_rope];
                let rbase = hh * (qk_nope + vh);

                let mut qabs = vec![0f32; kv_lora];
                for (dd, &qpd) in qp.iter().enumerate().take(qk_nope) {
                    qt_addrow(&w.kv_b, rbase + dd, qpd, &mut qabs);
                }

                let mut sc = vec![0f32; nt];
                for (jj, &t) in positions.iter().enumerate() {
                    let lt = kv.l_row(t);
                    let kr = kv.r_row(t);
                    let mut a: f32 = qabs.iter().zip(lt).map(|(&x, &y)| x * y).sum();
                    a += qr.iter().zip(kr).map(|(&x, &y)| x * y).sum::<f32>();
                    sc[jj] = a * cfg.attn_scale;
                }
                softmax(&mut sc);

                let mut clat = vec![0f32; kv_lora];
                for (jj, &t) in positions.iter().enumerate() {
                    let lt = kv.l_row(t);
                    let a = sc[jj];
                    for i in 0..kv_lora {
                        clat[i] += a * lt[i];
                    }
                }
                let ctx_slot = &mut ctx[(si * h + hh) * vh..(si * h + hh + 1) * vh];
                qt_matvec_rows(&w.kv_b, rbase + qk_nope, vh, &clat, ctx_slot);
            }
        }
    } else {
        let kvb_dim = h * (qk_nope + vh);
        let tk = pos_base + s;
        let mut kvb_all = vec![0f32; tk * kvb_dim];
        matmul_qt(&mut kvb_all[kv_start * kvb_dim..], kv.l_range(kv_start, tk), &w.kv_b, tk - kv_start);

        for si in 0..s {
            let pos = pos_base + si;
            let positions = positions_for(si, pos);
            let nt = positions.len();
            for hh in 0..h {
                let qp = &q[si * h * qh + hh * qh..si * h * qh + (hh + 1) * qh];
                let qr = &qp[qk_nope..qk_nope + qk_rope];

                let mut sc = vec![0f32; nt];
                for (jj, &t) in positions.iter().enumerate() {
                    let base = t * kvb_dim + hh * (qk_nope + vh);
                    let kn = &kvb_all[base..base + qk_nope];
                    let kr = kv.r_row(t);
                    let mut a: f32 = qp[..qk_nope].iter().zip(kn).map(|(&x, &y)| x * y).sum();
                    a += qr.iter().zip(kr).map(|(&x, &y)| x * y).sum::<f32>();
                    sc[jj] = a * cfg.attn_scale;
                }
                softmax(&mut sc);

                let ctx_slot = &mut ctx[(si * h + hh) * vh..(si * h + hh + 1) * vh];
                for v in ctx_slot.iter_mut() {
                    *v = 0.0;
                }
                for (jj, &t) in positions.iter().enumerate() {
                    let base = t * kvb_dim + hh * (qk_nope + vh) + qk_nope;
                    let vv = &kvb_all[base..base + vh];
                    let a = sc[jj];
                    for dd in 0..vh {
                        ctx_slot[dd] += a * vv[dd];
                    }
                }
            }
        }
    }

    matmul_qt(out, &ctx, &w.o, s);
    if computed_fresh { selection } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::QT;

    fn xorshift(seed: &mut u32) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 17;
        *seed ^= *seed << 5;
        ((*seed as f32 / u32::MAX as f32) - 0.5) * 2.0
    }

    fn random_vec(n: usize, seed: &mut u32) -> Vec<f32> {
        (0..n).map(|_| xorshift(seed)).collect()
    }

    fn random_qt_f32(rows: usize, cols: usize, seed: &mut u32) -> QT {
        let mut t = QT::alloc(rows, cols, 32, false);
        t.fill(&random_vec(rows * cols, seed));
        t
    }

    fn tiny_cfg() -> Cfg {
        let qk_nope = 3;
        let qk_rope = 2;
        Cfg {
            hidden: 8,
            n_layers: 2,
            n_heads: 2,
            n_experts: 4,
            topk: 2,
            moe_inter: 6,
            dense_inter: 10,
            first_dense: 1,
            q_lora: 6,
            kv_lora: 5,
            qk_nope,
            qk_rope,
            qk_head: qk_nope + qk_rope,
            v_head: 4,
            n_shared: 1,
            vocab: 16,
            n_group: 1,
            topk_group: 1,
            norm_topk: false,
            stop_ids: vec![],
            index_topk: 4096,
            index_nh: 2,
            index_hd: 3,
            idx_type: vec![false, true],
            eps: 1e-5,
            theta: 10000.0,
            attn_scale: 1.0 / ((qk_nope + qk_rope) as f32).sqrt(),
            routed_scale: 1.0,
        }
    }

    fn tiny_attn_weights(cfg: &Cfg, seed: &mut u32) -> AttnWeights {
        let d = cfg.hidden as usize;
        let h = cfg.n_heads as usize;
        let qh = cfg.qk_head as usize;
        let qk_nope = cfg.qk_nope as usize;
        let v_head = cfg.v_head as usize;
        let q_lora = cfg.q_lora as usize;
        let kv_lora = cfg.kv_lora as usize;
        let qk_rope = cfg.qk_rope as usize;
        AttnWeights {
            q_a: random_qt_f32(q_lora, d, seed),
            q_a_ln: vec![1.0; q_lora],
            q_b: random_qt_f32(h * qh, q_lora, seed),
            kv_a: random_qt_f32(kv_lora + qk_rope, d, seed),
            kv_a_ln: vec![1.0; kv_lora],
            kv_b: random_qt_f32(h * (qk_nope + v_head), kv_lora, seed),
            o: random_qt_f32(d, h * v_head, seed),
        }
    }

    #[test]
    fn rope_interleave_at_pos_zero_is_a_pure_deinterleave() {
        // ang=0 for every j when pos=0 -> cos=1,sin=0 -> output is just (evens, then odds).
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        rope_interleave(&mut v, 0, 10000.0);
        assert_eq!(v, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn rmsnorm_matches_hand_computation() {
        let mut v = vec![3.0, 4.0]; // rms = sqrt((9+16)/2) = sqrt(12.5)
        let w = vec![1.0, 2.0];
        rmsnorm(&mut v, &w, 0.0);
        let r = 1.0 / 12.5f32.sqrt();
        assert!((v[0] - 3.0 * r * 1.0).abs() < 1e-5);
        assert!((v[1] - 4.0 * r * 2.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_sums_to_one_and_is_monotonic() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(x[0] < x[1] && x[1] < x[2]);
    }

    #[test]
    fn dsa_select_falls_back_to_dense_when_context_is_short() {
        let cfg = tiny_cfg(); // index_topk=4096 >> any tiny seq_len
        let mut seed = 7;
        let dsa = DsaWeights {
            wq_b: random_qt_f32((cfg.index_nh * cfg.index_hd) as usize, cfg.q_lora as usize, &mut seed),
            wk: random_qt_f32(cfg.index_hd as usize, cfg.hidden as usize, &mut seed),
            weights_proj: random_qt_f32(cfg.index_nh as usize, cfg.hidden as usize, &mut seed),
            k_norm_w: vec![1.0; cfg.index_hd as usize],
            k_norm_b: vec![0.0; cfg.index_hd as usize],
        };
        let mut cache = DsaCache::new(cfg.index_hd as usize);
        let s = 5;
        let qr = random_vec(s * cfg.q_lora as usize, &mut seed);
        let x = random_vec(s * cfg.hidden as usize, &mut seed);

        let sel = dsa_select(&cfg, &dsa, &mut cache, &qr, &x, s, 0, false);
        assert!(sel.per_token.iter().all(|l| l.is_empty()));
        assert_eq!(cache.len(), s); // k_idx is still computed even when selection is a no-op
    }

    #[test]
    fn dsa_select_forced_keeps_every_position_as_a_set() {
        // The two-pass >thr/==thr scan (ported verbatim from the C, see `dsa_select`) fills
        // ties in a second sweep, so with keep==nk the SET of selected positions is always
        // 0..=pos, but not necessarily emitted in ascending order — attention doesn't care
        // (softmax over a set of keys is order-invariant), so this checks membership, not order.
        let cfg = tiny_cfg();
        let mut seed = 11;
        let dsa = DsaWeights {
            wq_b: random_qt_f32((cfg.index_nh * cfg.index_hd) as usize, cfg.q_lora as usize, &mut seed),
            wk: random_qt_f32(cfg.index_hd as usize, cfg.hidden as usize, &mut seed),
            weights_proj: random_qt_f32(cfg.index_nh as usize, cfg.hidden as usize, &mut seed),
            k_norm_w: vec![1.0; cfg.index_hd as usize],
            k_norm_b: vec![0.0; cfg.index_hd as usize],
        };
        let mut cache = DsaCache::new(cfg.index_hd as usize);
        let s = 6;
        let qr = random_vec(s * cfg.q_lora as usize, &mut seed);
        let x = random_vec(s * cfg.hidden as usize, &mut seed);

        let sel = dsa_select(&cfg, &dsa, &mut cache, &qr, &x, s, 0, true);
        for (si, list) in sel.per_token.iter().enumerate() {
            let mut got = list.clone();
            got.sort_unstable();
            let expected: Vec<u32> = (0..=si as u32).collect();
            assert_eq!(got, expected, "token {si}: forced selection must cover {{0..=pos}} as a set");
        }
    }

    #[test]
    fn absorbed_path_matches_dense_reconstruction_path() {
        let cfg = tiny_cfg();
        let mut seed = 21;
        let w = tiny_attn_weights(&cfg, &mut seed);
        let s = 3;
        let d = cfg.hidden as usize;
        let x = random_vec(s * d, &mut seed);

        let mut kv_a = KvCache::new(cfg.kv_lora as usize, cfg.qk_rope as usize);
        let mut out_absorb = vec![0.0; s * d];
        attention(&cfg, &w, &mut kv_a, &x, s, 0, Dsa::Off, Absorb::Always, &mut out_absorb);

        let mut kv_d = KvCache::new(cfg.kv_lora as usize, cfg.qk_rope as usize);
        let mut out_dense = vec![0.0; s * d];
        attention(&cfg, &w, &mut kv_d, &x, s, 0, Dsa::Off, Absorb::Never, &mut out_dense);

        for (a, b) in out_absorb.iter().zip(&out_dense) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn dsa_force_select_all_matches_plain_dense_attention() {
        let cfg = tiny_cfg(); // index_topk=4096 >> seq_len -> forced selection covers everything
        let mut seed = 33;
        let w = tiny_attn_weights(&cfg, &mut seed);
        let dsa_w = DsaWeights {
            wq_b: random_qt_f32((cfg.index_nh * cfg.index_hd) as usize, cfg.q_lora as usize, &mut seed),
            wk: random_qt_f32(cfg.index_hd as usize, cfg.hidden as usize, &mut seed),
            weights_proj: random_qt_f32(cfg.index_nh as usize, cfg.hidden as usize, &mut seed),
            k_norm_w: vec![1.0; cfg.index_hd as usize],
            k_norm_b: vec![0.0; cfg.index_hd as usize],
        };
        let s = 4;
        let d = cfg.hidden as usize;
        let x = random_vec(s * d, &mut seed);

        let mut kv_plain = KvCache::new(cfg.kv_lora as usize, cfg.qk_rope as usize);
        let mut out_plain = vec![0.0; s * d];
        attention(&cfg, &w, &mut kv_plain, &x, s, 0, Dsa::Off, Absorb::Never, &mut out_plain);

        let mut kv_dsa = KvCache::new(cfg.kv_lora as usize, cfg.qk_rope as usize);
        let mut dsa_cache = DsaCache::new(cfg.index_hd as usize);
        let mut out_dsa = vec![0.0; s * d];
        let sel = attention(
            &cfg,
            &w,
            &mut kv_dsa,
            &x,
            s,
            0,
            Dsa::Compute { weights: &dsa_w, cache: &mut dsa_cache, force: true },
            Absorb::Never,
            &mut out_dsa,
        );
        assert!(sel.is_some(), "Dsa::Compute must return the freshly computed selection");

        for (a, b) in out_plain.iter().zip(&out_dsa) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }
}
