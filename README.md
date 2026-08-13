<p align="center">
  <img src="assets/rabbit.svg" width="500" alt="rabbit — small hops, immense models">
</p>

**A 2.4-trillion-parameter model. One mini-PC. 46 GB of RAM.**

Frontier open-weight MoE inference in Rust. No GPU, no BLAS, no framework. The dense part stays in
RAM. The routed experts, 1.26 TB of them, stay on disk and stream in as each token needs them.

```
$ rabbit --model /mnt/data/qwen38-max-mxfp4 --shard-dirs ~/qwen38-max-mxfp4-shards2 \
         --prompt "¿Cuál es la capital de Francia? Respondé en una frase."
model loaded in 113.1s (92 layers)
prefill (14 tokens)...
  prefill done in 59.3s
  token 1/40 in 5.2s ...

<think>
The user is asking "What is the capital of France?" ... the capital of France is Paris.

phase breakdown: expert wait (disk) 143.8s, expert matmul 76.6s, attention 22.4s, lm_head 0.6s
```

Alibaba published Qwen3.8-Max (2.446 T parameters) in August 2026. rabbit runs it off AMD's MXFP4
release with no conversion step, reading the on-disk format byte for byte.

## Measured on a Slimbook ONE (Ryzen AI 9 HX 370, 12 cores, 123 GB RAM, 2x NVMe)

| Model | Params | On disk | RAM used | Decode | Version |
|---|---|---|---|---|---|
| **Qwen3.8-Max** | 2.446 T | 1.37 TB (MXFP4) | **45.7 GB** | **4.71 s/token, 0.212 tok/s** | v0.29.0, first working version |
| Kimi K3 | 2.8 T | 1.45 TB (MXFP4) | 46 GB | 5.93 s/token, 0.169 tok/s | v0.28.1, 7.3x faster than v0.23.0 |
| GLM-5.2 | 744 B | 378 GB (int4) | | 1.02 words/s, 3.5x faster than v0.14.0 | v0.22.0 |
| Kimi Linear 48B | 48 B | BF16 | | not characterized | |

Every number comes from a real run against the real checkpoint. Full chronological logs, including
what was tried and reverted: `PERFORMANCE_QWEN38.md`, `PERFORMANCE_KIMI_K3.md`, `PERFORMANCE.md`.

Where a Qwen3.8-Max decode step goes: 59% waiting for expert bytes, 31% multiplying them, 9% for all
92 layers of attention and Gated DeltaNet combined.

## The idea

A large MoE activates a small fraction of its parameters per token, and only the routed experts
change from token to token. So:

* the dense part (attention, shared experts, embeddings) stays resident in RAM, quantized. That is
  24.3 GB for a 2.4 T model.
* the routed experts, 47,104 of them on Qwen3.8-Max at 27 MB each, live on disk and stream on demand
  through a per-layer LRU cache, a persistent learned pin for the hottest ones, and the OS page cache
  as a free extra tier.

Per token that means about 24.6 GB read from disk. Splitting the checkpoint across two NVMe drives
(`--shard-dirs`) measures at 61%/39% of the reads, so both drives work in parallel.

## Architectures

| | Qwen 3.8 | Kimi K3 | Kimi Linear 48B | GLM-5.2 |
|---|---|---|---|---|
| params (total / routed per token) | 2.446 T / 46.3 B | 2.8 T / 48.6 B | 48 B / ~3 B | 744 B / ~40 B |
| attention | GQA + partial RoPE + output gate (23 of 92 layers) | KDA + Gated MLA | KDA + Gated MLA | MLA + DSA sparse indexer |
| linear attention | Gated DeltaNet (69 of 92 layers) | Kimi Delta Attention | Kimi Delta Attention | none |
| MoE routing | softmax top-10 of 512 + gated shared expert | Stable LatentMoE | grouped routing | `noaux_tc` sigmoid |
| native quantization read | OCP MXFP4 | OCP MXFP4 | BF16 | FP8 (E4M3, block-scale) |
| checkpoint | AMD's Quark MXFP4 release, as-is | Moonshot's release, as-is | Moonshot's release, as-is | pre-converted to int4 |

One family-dispatch enum (`src/model.rs`) picks the architecture from `config.json`'s `model_type`.
`--prompt`, `--chat`, `--serve` and `--session` work identically across all four.

## Quickstart

```bash
cargo build --release
cargo test                       # 461 tests, no checkpoint needed
```

```bash
# Qwen3.8-Max, split across two drives
rabbit --model /mnt/data/qwen38-max-mxfp4 --shard-dirs ~/qwen38-max-mxfp4-shards2 \
       --prompt "What is the capital of France?"

rabbit --model <dir> --chat --session ~/.rabbit_session      # multi-turn, resumes across restarts
rabbit --model <dir> --serve --port 8000                     # OpenAI-compatible HTTP
```

Checkpoints, no conversion needed:
[`amd/Qwen3.8-2.4T-A95B-Quark-MXFP4`](https://huggingface.co/amd/Qwen3.8-2.4T-A95B-Quark-MXFP4) (1.37 TB),
[`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3) (1.45 TB),
[`moonshotai/Kimi-Linear-48B-A3B-Instruct`](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct).
GLM-5.2 needs a pre-converted int4 checkpoint, which `bin/convert.rs` produces.

`--help` lists the rest: `--expert-cache`, `--io-batch-size`, `--dbits`/`--ebits`, `--temperature`,
`--nucleus`, `--think`, `--no-usage-cache`, `--mmap-experts`.

## What's implemented

* Faithful forward pass for four architectures, each validated token-exact against a tiny model built
  from that family's own real reference code.
* `io_uring`-batched expert streaming with per-expert early drain, so an expert's matmul starts the
  moment its own bytes land instead of when the whole batch finishes. Sequential `pread` fallback
  elsewhere.
* RAM-aware `--expert-cache` auto-clamp. On Qwen3.8-Max the flat default of 64 would have asked for
  about 236 GB, so it lowers itself to 9 and says so in the log.
* Persistent expert usage cache (`.rabbit_usage`) that learns which experts your prompts route to and
  pins them.
* int4 / int8 / int2 quantization, native FP8 and OCP MXFP4 reading, grouped-scale int4, and a `.qs`
  pre-quantized fast path.
* AVX2 and AVX-512/VNNI kernels, runtime-selected, with `rayon` across cores.
* KV-session persistence (`--session`) per architecture, including Qwen's hybrid state: an
  append-only log for the 23 attention layers, an atomically-replaced snapshot for the 69 recurrent
  ones.
* OpenAI-compatible server (`--serve`) with streaming `/v1/chat/completions`, `/v1/models`, and
  `/profile` for rolling per-turn phase timings.
* Architecture-agnostic checkpoint converter (`bin/convert.rs`) with per-bucket bit-depth control and
  a `--report` quality pass.

Not built: GPU, MTP speculative decoding, ARM NEON, grammar-constrained decoding, live re-pinning, a
web UI for `/profile`.

## How correctness is checked

Three independent layers, because fluent-looking output proves very little.

1. Teacher forcing against the reference implementation. A tiny random model built from the family's
   own PyTorch code (`tests/oracle/make_*_oracle.py`), then argmax compared at every position plus an
   incremental-decode replay. Qwen 3.8 matches at all 12 positions and reproduces the reference
   continuation exactly.
2. Oracles for the pieces around the model. The tokenizer against HuggingFace's own `tokenizers`
   (23 of 24 cases exact, the 24th a documented NFC difference that still round-trips), and the chat
   template against the real `chat_template.jinja` rendered by Jinja2 (9 of 9).
3. Property tests over the engine. Batched prefill equals token-by-token stepping, a restored session
   continues bit-identically to a live one, and every tensor is asserted into its own field from a
   synthetic checkpoint where each one holds a distinct value.

Some of these caught real bugs before any weights existed. Qwen's `Qwen3_5MoeRMSNorm` scales by
`(1 + w)` while its Gated DeltaNet norm scales by plain `w`; using the crate's usual RMSNorm would
have quietly collapsed every activation toward zero, with no error anywhere.

## Repo layout

```
src/
├── qwen38/           Qwen 3.8: GQA + partial RoPE, Gated DeltaNet, softmax MoE, (1+w) norms
├── kimi_k3/          Kimi K3: SituAndMul, LatentMoE, Attention Residuals, MXFP4
├── kimi_linear/      Kimi Linear 48B: KDA recurrence, short convs, tokenizer, chat template
├── glm52/            GLM-5.2: MLA + DSA attention, MoE router, checkpoint converter
├── model.rs          family dispatch: Model / KvState / ExpertCaches / Tokenizer
├── expert_cache.rs   LRU + pinned expert streaming, io_uring batching, MXFP4/FP8/int4 loading
├── kernels.rs        scalar / AVX2 / AVX-512-VNNI / MXFP4 matmuls
├── quant.rs, safetensors.rs, usage_cache.rs, generate.rs, kv_session.rs
└── chat.rs, server.rs, main.rs
tests/oracle/         per-architecture reference-model generators (vendored real code)
tools/                fixture downloaders, checkpoint download/convert scripts
benches/              criterion benchmarks (kernels, expert loading)
```

## Why this exists

Every algorithm here is implemented from the reference sources: each architecture's own published
modeling code, read and ported rather than approximated, then validated against it. On top of that
sit the pieces that make disk-resident inference work at all: `io_uring` expert streaming with
per-expert early drain, a RAM-aware cache clamp, persistent expert-usage learning, hybrid KV-session
persistence, and a family-dispatch design that serves four architectures from one engine.

Part of the [ferrumox](../) AI lab, alongside [fox](../fox), a production local-LLM server wrapping
llama.cpp. rabbit is the opposite kind of project: a research engine for models that don't fit in
memory even offloaded, built by hand instead of wrapping a runtime.

## Versioning

Pre-1.0, `0.MINOR.PATCH`, via git tags and `release/vX.Y.Z` branches. MINOR means a real measured
improvement or a new architecture, PATCH means kept-but-neutral work. Development phases are logged
in `rabbit-plan.md`, per-architecture port notes in `QWEN38_PORT.md`.

## License

Dual-licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.

Checkpoints are not covered by this license: each model's weights carry their own terms (Qwen3.8-Max
ships under Alibaba's own `qwen3.8-max` license, Kimi K3 and Kimi Linear under Moonshot's, GLM-5.2
under Zhipu's). Check them before redistributing anything you convert.
