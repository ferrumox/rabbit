<p align="center">
  <img src="assets/rabbit.svg" width="500" alt="rabbit — small hops, immense model">
</p>

**Small hops, immense model.** Run **GLM-5.2 (744B-parameter MoE)** on a single machine by keeping the dense part resident in RAM at int4 and streaming the 21,504 routed experts from disk on demand — a Rust reimplementation of [colibrì](https://github.com/JustVugg/colibri)'s C engine.

```
$ rabbit --model /nvme/glm52_i4 --prompt "The capital of France is"
prompt: 6 tokens
prefill (6 tokens)...
  prefill done in 4.2s (expert cache: 12 hits, 468 misses, 3.1s in disk I/O)
  token 1/64 in 2.1s (1.4s in disk I/O this step; expert cache totals: 20 hits, 476 misses)
  ...
 Paris. It is located in the north of the country...
```

## The idea

A 744B-parameter MoE model activates only ~40B parameters per token, and only a fraction of
those change token to token (the routed experts). So:

- the **dense part** (attention, shared expert, embeddings) stays **resident in RAM at int4**;
- the **21,504 routed experts** (75 MoE layers × 256 experts, ~19 MB each at int4) live **on
  disk** and are **streamed on demand**, via a per-layer LRU cache, a persistent learned pin for
  the hottest ones, and the OS page cache as a free extra tier.

## What's implemented

- **Faithful GLM-5.2 (`glm_moe_dsa`) forward** — validated token-exact against a synthetic
  oracle (32/32 teacher-forcing, 20/20 greedy decode) and against the real 744B checkpoint.
- **MLA attention** with **weight absorption** for decode (no per-token k/v reconstruction) and
  dense reconstruction for prefill — validated exact against each other, and parallelized with
  `rayon` across attention heads.
- **DSA sparse attention** — the lightning indexer, with selection sharing between "full" and
  "shared" layers across a forward pass.
- **`noaux_tc` sigmoid MoE router** with correction bias, shared expert, batch-union expert
  dispatch (each unique expert in a batch is read and applied once).
- **int4/int8/int2 quantization**, native **FP8** (E4M3, block-scale) checkpoint loading, and a
  **`.qs` pre-quantized fast path** — no external converter needed for any of the three.
- **AVX2 + AVX-512/VNNI kernels**, runtime-selected, plus `rayon` parallelization across CPU
  cores (not just SIMD) for every matmul and the absorbed-attention decode path.
- **`io_uring`-batched expert streaming**, with a sequential-`pread` fallback.
- **Persistent expert usage cache** (`.rabbit_usage`) — learns which experts your usage routes
  to and pins them, lazily: only once a candidate is actually loaded through normal use, unlike
  colibrì's own eager default (see `rabbit-plan.md` for the measured reason it was changed).
- **KV-cache persistence** (`--session`) — conversations reopen warm across restarts, via an
  append-only on-disk format.
- **OpenAI-compatible HTTP server** (`--serve`) — streaming and non-streaming
  `/v1/chat/completions`, `/v1/models`; plus `/profile`, a rolling per-turn phase-timing window
  (attention/expert-wait/expert-matmul/lm-head) as JSON.
- **Multi-turn chat** (`--chat`) with GLM-5.2's official template.

Not yet built: a standalone `.qs` converter (currently relies on checkpoints pre-converted by
colibrì's own tooling), live expert re-pinning, GPU/CUDA, MTP speculative decoding, ARM NEON,
grammar-constrained decoding, an expert-routing heatmap (colibrì's "Brain" view), and any web
UI (`/profile` above is a JSON endpoint, no page serves it yet — see `DASHBOARD_BRIEF.md` for
the planned direction). See `rabbit-plan.md` for the full phase-by-phase history.

## Honest numbers (Ryzen AI 9 HX 370, 12 cores/24 threads)

| metric | value |
|---|---|
| checkpoint | 378 GB (`jlnsrk/GLM-5.2-colibri-int4`) |
| `rayon` matmul parallelization | 128.9s → 36.3s for 5 tokens (3.5×), bit-exact output |
| `rayon` absorbed-attention parallelization | 224.3s → 158.4s for 70 decode tokens (~29% faster) |
| decode I/O share, steady state (warm cache) | ~30–35% disk I/O / 65–70% compute |
| prefill I/O share (cold cache) | ~75% disk I/O |
| expert-cache hit rate, steady-state decode | ~70–77% (miss floor ~23–30%) |
| usage-cache auto-pin | 150 experts (2/layer × 75 MoE layers) → prefill hits 0 → 136 |

All measured against the real checkpoint, not estimated — see the phase entries in
`rabbit-plan.md` for the commits and full methodology behind each number, or
[`PERFORMANCE.md`](PERFORMANCE.md) for the full chronological log, including techniques that
were tried and reverted.

## Building

```bash
cargo build --release
cargo test
```

```bash
./target/release/rabbit --model <checkpoint-dir> --prompt "The capital of France is"
./target/release/rabbit --model <checkpoint-dir> --chat --session ~/.rabbit_session
./target/release/rabbit --model <checkpoint-dir> --serve --port 8000
```

`--model` takes a directory with the same layout colibrì's converter produces — a pre-converted
checkpoint such as [`jlnsrk/GLM-5.2-colibri-int4`](https://huggingface.co/jlnsrk/GLM-5.2-colibri-int4)
works directly, no conversion step needed. See `--help` for the full flag list
(`--max-tokens`, `--temperature`, `--nucleus`, `--expert-cache`, `--no-usage-cache`, ...).

## Repo layout

```
src/
├── safetensors.rs, config.rs          shard index + config loading
├── tokenizer.rs, unicode_tables.rs    byte-level BPE tokenizer
├── quant.rs, kernels.rs               quantization + scalar/AVX2/AVX-512 kernels
├── model.rs, attention.rs, moe.rs     dense model, MLA/DSA attention, MoE router
├── expert_cache.rs, usage_cache.rs    LRU expert streaming + persistent usage learning
├── generate.rs, kv_session.rs         generation loop + KV-cache persistence
├── chat.rs, server.rs, main.rs        chat template, HTTP server, CLI entrypoint
tests/oracle/     synthetic GLM-5.2 oracle generator + teacher-forcing fixtures
tools/            real-tokenizer validation fixtures (dev-only, not a runtime dependency)
benches/          criterion benchmarks (kernels, expert loading)
```

## Why Rust, why colibrì

colibrì is C, hand-written, effectively zero-dependency. rabbit ports the same algorithms to
Rust with a short list of well-justified dependencies instead of a zero-dep stance — see
`rabbit-plan.md` for the reasoning on each one — plus its own performance work on top that
colibrì doesn't have in this exact form (the `rayon` parallelization above, KV-session and
expert-usage persistence). Validated the same way colibrì validates itself: token-exact
teacher-forcing against a tiny synthetic model with the real architecture.

Part of the [ferrumox](../) AI lab, alongside [fox](../fox) (a production local-LLM server
wrapping llama.cpp) — rabbit is the opposite kind of project: a research engine for a model that
doesn't fit in memory even offloaded, built by hand instead of wrapping an existing runtime.

The name is a nod to [RabbitLLM](https://github.com/ManuelSLemos/RabbitLLM), an earlier,
unrelated project of mine (a fork of AirLLM that streams full model layers through limited GPU
VRAM) — same interest in running large models on constrained hardware, different problem and a
completely different technique (CPU-side quantized MoE expert streaming here, vs. GPU layer
streaming there). Nothing in this codebase is derived from that one.

## Versioning

Pre-1.0, `0.MINOR.PATCH`: `MINOR` bumps at the end of each phase in `rabbit-plan.md` (`v0.1.0`
through `v0.15.0`), `PATCH` for fixes/polish within a phase.

## License

TBD.
