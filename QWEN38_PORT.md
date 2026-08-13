# Qwen 3.8 port — state as of 2026-08-13

What's built, how each piece was verified, and what's still missing. **No performance numbers here
yet**: nothing has run against the full checkpoint (the download is still in progress), and this
project's convention is that `PERFORMANCE*.md` files carry only measured facts. A
`PERFORMANCE_QWEN38.md` gets created the first time there's a real end-to-end run to report.

## The target and the checkpoint

**Qwen3.8-Max** (`Qwen/Qwen3.8-2.4T-A95B`, 2.446 T parameters, arch id `qwen3_5_moe_text`), read from
**`amd/Qwen3.8-2.4T-A95B-Quark-MXFP4`** — 1.37 TB instead of the bf16 original's 4.89 TB, and
**no conversion pass at all**: its routed experts are OCP MXFP4 with the exact byte layout
`quant.rs`'s `QTKind::MxFp4` already reads for Kimi K3.

That claim was verified before committing to the download, not assumed: decoding
`model.layers.3.mlp.experts.0.gate_proj` with rabbit's own nibble order scored **cosine 0.9933**
against the same rows of the bf16 original (74 KB of ranged HTTP reads), versus ~0.02 for both
alternative packings. Quark's `pack_method: "reorder"` does not mean a different nibble order.

Local layout (needs `--shard-dirs`, both drives read in parallel):

| | |
|---|---|
| `/mnt/data/qwen38-max-mxfp4` | config + tokenizer + 129 shards (nvme0n1) |
| `~/qwen38-max-mxfp4-shards2` | 84 shards (nvme1n1) |
| re-run / resume | `tools/download_qwen38_mxfp4.sh` |

Key numbers from the real config (printed by `examples/qwen38_config_dump.rs`): 92 layers, hidden
8192, 512 experts top-10 + 1 gated shared expert, **46.3 B routed-expert params per token = 24.6 GB
read per token**, 1.26 TB of experts on disk, ~76 B non-expert params arriving as bf16 (49.6 GB) to
requantize at load. Kimi K3, which this machine already ran, moves 25.8 GB/token — the same problem.

## Architecture, and the four places it bites

| Piece | Reused | Genuinely new |
|---|---|---|
| Expert streaming | `expert_cache.rs` + `glm52::moe::apply_single_expert` wholesale | tensor-naming variant `ExpertNaming::Qwen38Mxfp4` (`.weight` vs K3's `.weight_packed`) |
| Linear attention (69 layers) | `kimi_linear::kda::KdaState::step` **is** the gated delta rule | scalar-per-head decay, 16 K heads → 128 V heads, one conv over concatenated q\|k\|v, SiLU output gate |
| Full attention (23 layers) | — | GQA with partial RoPE (64 of 256 dims, `rotate_half`), per-head q/k norm, sigmoid output gate |
| MoE routing | expert application, chunking, early drain | plain softmax top-k (no sigmoid+bias, no groups, no scaling factor), shared expert with its own sigmoid gate |
| Normalization | — | `Qwen3_5MoeRMSNorm` scales by **`(1 + w)`**, weights initialized to zeros |

Four traps worth remembering, each pinned by a test that fails under the wrong reading:

1. **`q_proj` emits query and gate interleaved PER HEAD**, not as two contiguous halves. The wrong
   reading mixes head 32's query into head 0's gate and produces plausible garbage.
2. **`(1 + w)` RMSNorm.** Using the crate's usual `norm(x) * w` on a Qwen checkpoint (whose weights
   sit near 0) collapses activations toward zero, silently. Note the GDN output norm
   (`Qwen3_5MoeRMSNormGated`) is the *other* convention — plain `w`, initialized to ones.
3. **The output gates differ**: attention uses `sigmoid(gate)` despite `output_gate_type: "swish"`;
   GDN uses `silu(z)` — the exact opposite of Kimi's deliberate sigmoid there.
4. **Stop tokens need both files**: `config.json` lists only `<|endoftext|>` (248044); `<|im_end|>`
   (248046), which ends every assistant turn, appears only in `generation_config.json`. `Cfg::load`
   unions them.

**mRoPE is a no-op** for text-only input (all three position grids hold the same ids), even though the
reference rotary looks like it matters. **The MTP block is skipped**: it's a full extra MoE layer with
its own 512 experts (~14 GB), and rabbit does no speculative decoding.

## Modules

`src/qwen38/`: `config`, `tokenizer`, `chat_template`, `ops`, `attention`, `gdn`, `moe`, `model`,
`generate`, `kv_session`. Dispatch arms in `crate::model` and `crate::chat`, so `--prompt`, `--chat`,
`--serve` and `--session` all work for this family.

## How it's verified

- **Two independent oracles.** The tokenizer matches the Python `tokenizers` library on **23 of 24**
  cases, the 24th being the one deliberate known difference (the declared `NFC` normalizer isn't
  implemented; that case still round-trips byte-for-byte). The chat template matches the real
  `chat_template.jinja` rendered by Jinja2 on **9/9** cases.
- **A synthetic checkpoint** (4 layers, the real 3:1 pattern, experts and MTP block included) where
  every tensor holds a distinct constant plus a per-element ripple — so tests check *which field a
  tensor landed in*, not just that loading succeeded. The ripple matters: with pure constants every
  activation comes out with equal components and RMSNorm erases the difference, which made an earlier
  version of the fixture blind to state changes.
- **Property tests over the forward pass**: batched prefill equals token-by-token stepping; a restored
  session continues bit-identically to the live one; only attention layers grow with context.
- **Against the real checkpoint**: `config.json` parses, the tokenizer loads, and the loader walks
  layer by layer until it reaches a shard still downloading — i.e. real tensor names and shapes
  resolve.
- **A teacher-forcing oracle against `transformers` itself** (`tests/oracle/make_qwen38_oracle.py`,
  `tests/teacher_forcing_qwen38.rs`): a tiny random `Qwen3_5MoeForCausalLM` whose config turns on every
  Qwen-specific path at once. rabbit reproduces the reference's per-position argmax **at all 12
  positions** and its greedy continuation **exactly**, and a separate check confirms batched and
  incremental prefill agree on those weights too. The oracle's norm weights are randomized away from
  zero on purpose so the `(1 + w)` convention is actually exercised — reverting that one line to
  `norm(x) * w` makes 2 of the 3 tests fail, which is how we know the oracle bites.

436 crate tests + 17 integration tests, clippy clean.

## Still missing

1. **A full end-to-end run** — blocked on the download (~43% at the time of writing). The pieces that
   need it: real decode speed, real RSS, and whether the auto `--expert-cache` clamp picks a sane
   capacity at 92 layers. `examples/qwen38_smoke.rs` is ready for exactly that (it reports seconds per
   token split into compute vs pure disk wait).
2. **Performance work**: both the GDN per-head loop and the attention per-head loop are serial.
   `rayon` is the obvious next step, and GDN (128 heads × a 128×128 state per layer) is where the
   compute actually is — the 23 attention layers are the cheap ones here.
3. **`--reasoning-effort`**: the template supports `xhigh`/`medium`/`low` and the port hardcodes the
   template's own default (`xhigh`). Wiring a flag is a one-liner in `qwen38::chat_template`, whose
   private `Effort` enum already pins all three texts.
