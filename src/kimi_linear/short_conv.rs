//! KDA's `ShortConv`: a causal depthwise 1-D convolution applied per-channel to the post-
//! projection Q/K/V streams before the KDA recurrence (arXiv:2510.26692 §4: `q,k =
//! L2Norm(Swish(ShortConv(Wx)))`, `v = Swish(ShortConv(Wx))`).
//!
//! Confirmed against a real, independent reference this session — llama.cpp's vendored
//! `kimi-linear.cpp` (`causal_conv1d()` + its `ggml_ssm_conv` call) — not just the paper's prose:
//! kernel size 4 (`ssm_d_conv`), one weight vector per channel of length `d_conv`, no bias, and a
//! causal window `[x[t-3], x[t-2], x[t-1], x[t]]` (oldest tap first, current token last) — the
//! same convention llama.cpp's GGUF conv-weight reshape and `ggml_ssm_conv` use.
//!
//! Decode is autoregressive one token at a time, so the only state needed per channel is the
//! trailing `d_conv - 1` history samples (a fixed-size FIFO) — no growth, unlike GLM-5.2's
//! `KvCache`, matching this crate's other KDA state (`kda.rs::KdaState`).

/// Per-channel causal-conv FIFO state for one Q, K, or V stream: `d_inner` channels, each
/// holding its trailing `kernel - 1` history samples (oldest first).
pub struct ShortConvState {
    d_inner: usize,
    kernel: usize,
    history: Vec<f32>,
}

impl ShortConvState {
    pub fn new(d_inner: usize, kernel: usize) -> ShortConvState {
        assert!(kernel >= 1, "a causal conv needs at least a 1-tap window");
        ShortConvState { d_inner, kernel, history: vec![0.0; d_inner * (kernel - 1)] }
    }

    pub fn d_inner(&self) -> usize {
        self.d_inner
    }

    pub fn kernel(&self) -> usize {
        self.kernel
    }

    /// The raw `d_inner * (kernel - 1)` history FIFO, oldest-first per channel — `kv_session.rs`'s
    /// save path, the read counterpart of `from_raw`.
    pub(crate) fn history(&self) -> &[f32] {
        &self.history
    }

    /// Reconstructs a `ShortConvState` from a previously-saved raw FIFO — `kv_session.rs`'s load
    /// path, the mirror of `new` (which starts all-zero) for restoring one with real history.
    pub(crate) fn from_raw(d_inner: usize, kernel: usize, history: Vec<f32>) -> ShortConvState {
        assert!(kernel >= 1, "a causal conv needs at least a 1-tap window");
        assert_eq!(history.len(), d_inner * (kernel - 1), "ShortConvState::from_raw: history length doesn't match d_inner*(kernel-1)");
        ShortConvState { d_inner, kernel, history }
    }

    /// Applies one token's causal depthwise conv in place: for each channel `c`,
    /// `out[c] = sum_{t=0}^{kernel-1} weight[c*kernel + t] * window[t]`, where `window` is that
    /// channel's `kernel - 1` history taps (oldest first) followed by the current `x[c]` — the
    /// same oldest-to-newest tap order `ggml_ssm_conv`'s reshaped GGUF weight expects. `weight`
    /// is `[d_inner, kernel]` row-major (one contiguous `kernel`-length row per channel, matching
    /// llama.cpp's `ggml_reshape_2d(conv_w, d_conv, d_inner)`). Updates the FIFO by dropping the
    /// oldest sample and pushing `x[c]` as the newest, so the next call sees the correct window.
    pub fn step(&mut self, x: &[f32], weight: &[f32], out: &mut [f32]) {
        let (d_inner, kernel) = (self.d_inner, self.kernel);
        assert_eq!(x.len(), d_inner);
        assert_eq!(out.len(), d_inner);
        assert_eq!(weight.len(), d_inner * kernel);
        let hist_len = kernel - 1;

        for c in 0..d_inner {
            let hist = &self.history[c * hist_len..(c + 1) * hist_len];
            let w = &weight[c * kernel..(c + 1) * kernel];
            let mut acc = 0f32;
            for t in 0..hist_len {
                acc += w[t] * hist[t];
            }
            acc += w[hist_len] * x[c];
            out[c] = acc;
        }

        for (c, &xc) in x.iter().enumerate() {
            let base = c * hist_len;
            for t in 0..hist_len.saturating_sub(1) {
                self.history[base + t] = self.history[base + t + 1];
            }
            if hist_len > 0 {
                self.history[base + hist_len - 1] = xc;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(seed: &mut u32) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 17;
        *seed ^= *seed << 5;
        ((*seed as f32 / u32::MAX as f32) - 0.5) * 2.0
    }

    fn random_vec(n: usize, seed: &mut u32) -> Vec<f32> {
        (0..n).map(|_| xorshift(seed)).collect()
    }

    /// Independent reference: run the whole causal conv over a full sequence at once, the
    /// textbook way (zero-padded on the left), rather than through the incremental FIFO.
    fn naive_full_sequence_conv(xs: &[Vec<f32>], weight: &[f32], d_inner: usize, kernel: usize) -> Vec<Vec<f32>> {
        xs.iter()
            .enumerate()
            .map(|(t, _)| {
                (0..d_inner)
                    .map(|c| {
                        let w = &weight[c * kernel..(c + 1) * kernel];
                        (0..kernel)
                            .map(|tap| {
                                // tap=kernel-1 is "current"; tap=0 is "kernel-1 steps back".
                                let src_t = t as isize - (kernel as isize - 1 - tap as isize);
                                if src_t < 0 { 0.0 } else { xs[src_t as usize][c] * w[tap] }
                            })
                            .sum::<f32>()
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn step_matches_naive_full_sequence_convolution() {
        let (d_inner, kernel) = (3, 4);
        let mut seed = 5u32;
        let weight = random_vec(d_inner * kernel, &mut seed);
        let xs: Vec<Vec<f32>> = (0..7).map(|_| random_vec(d_inner, &mut seed)).collect();

        let naive = naive_full_sequence_conv(&xs, &weight, d_inner, kernel);

        let mut st = ShortConvState::new(d_inner, kernel);
        for (t, x) in xs.iter().enumerate() {
            let mut out = vec![0f32; d_inner];
            st.step(x, &weight, &mut out);
            for (a, b) in out.iter().zip(&naive[t]) {
                assert!((a - b).abs() < 1e-5, "token {t}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn first_tokens_see_zero_padded_history() {
        // Before enough real tokens have flowed through, the missing history taps must act as
        // zeros (left-padding), not garbage or the current token repeated.
        let (d_inner, kernel) = (1, 4);
        let weight = [1.0, 1.0, 1.0, 1.0]; // sum of the window
        let mut st = ShortConvState::new(d_inner, kernel);

        let mut out = [0f32];
        st.step(&[5.0], &weight, &mut out);
        assert_eq!(out, [5.0], "only the current token, rest is zero-padded history");

        st.step(&[2.0], &weight, &mut out);
        assert_eq!(out, [7.0], "5.0 (now history) + 2.0 (current)");

        st.step(&[1.0], &weight, &mut out);
        assert_eq!(out, [8.0]);

        st.step(&[1.0], &weight, &mut out);
        assert_eq!(out, [9.0], "window now full: 5+2+1+1");

        st.step(&[1.0], &weight, &mut out);
        assert_eq!(out, [5.0], "oldest (5.0) fell out of the window: 2+1+1+1");
    }

    #[test]
    fn kernel_size_one_is_pure_pointwise_scaling_no_history() {
        let (d_inner, kernel) = (2, 1);
        let weight = [3.0, -2.0];
        let mut st = ShortConvState::new(d_inner, kernel);
        assert_eq!(st.kernel(), 1);

        let mut out = [0f32; 2];
        st.step(&[1.0, 1.0], &weight, &mut out);
        assert_eq!(out, [3.0, -2.0]);
        st.step(&[10.0, 10.0], &weight, &mut out);
        assert_eq!(out, [30.0, -20.0], "no cross-token leakage with a 1-tap kernel");
    }

    #[test]
    fn channels_are_independent() {
        // ch0's weight picks its oldest history tap (2 steps back); ch1's weight picks the
        // current token. A change in channel 0's weights/history must never leak into channel
        // 1's output, and vice versa.
        let (d_inner, kernel) = (2, 3);
        let weight = [1.0, 0.0, 0.0, /* ch0: oldest tap */ 0.0, 0.0, 1.0 /* ch1: current tap */];
        let mut st = ShortConvState::new(d_inner, kernel);
        let mut out = [0f32; 2];

        st.step(&[10.0, 100.0], &weight, &mut out);
        assert_eq!(out, [0.0, 100.0], "ch0's oldest tap is still zero-padded; ch1 sees its current token immediately");
        st.step(&[20.0, 200.0], &weight, &mut out);
        assert_eq!(out, [0.0, 200.0], "ch0's oldest tap is still zero-padded (only 1 real token so far)");
        st.step(&[30.0, 300.0], &weight, &mut out);
        // ch0 now has 2 real tokens of history, so its oldest tap is the first token (10.0);
        // ch1 keeps tracking the current token (300.0).
        assert_eq!(out, [10.0, 300.0]);
    }
}
