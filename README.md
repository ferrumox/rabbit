# rabbit

A Rust inference engine for GLM-5.2 (744B-parameter MoE): the dense part resident in RAM at
int4, the 21,504 routed experts streamed on demand from disk. Part of the [ferrumox](../) AI
lab, alongside [fox](../fox) (a production local-LLM server wrapping llama.cpp) — rabbit is the
opposite kind of project: a research engine for a model that doesn't fit in memory even
offloaded, built by hand instead of wrapping an existing runtime.

## What it does today

- `--prompt "<text>"` — single-shot completion.
- `--chat` — interactive multi-turn chat with GLM-5.2's official template, `:reset`/`:quit`,
  and `--session <path>` to persist the KV cache across process restarts.
- `--serve` — an OpenAI-compatible HTTP server (`POST /v1/chat/completions`, streaming and
  non-streaming; `GET /v1/models`).
- A persistent expert usage cache (`.rabbit_usage`, automatic, no flag needed) that learns which
  experts your usage routes to most and pins them in RAM across restarts.
- int4/int8/int2 quantization, native FP8 checkpoint loading, MLA attention with weight
  absorption for decode, the DSA sparse-attention indexer, `io_uring`-batched expert streaming,
  and `rayon`-parallel kernels across the hot paths.

See [`rabbit-plan.md`](rabbit-plan.md) for the full phase-by-phase history, and the `v0.*.0`
git tags for a working snapshot at the end of each phase.

## Why

This engine's design — MLA attention with weight absorption, the DSA sparse-attention indexer,
the `noaux_tc` MoE router, on-demand expert streaming from disk, the GLM-5.2 chat template — is
ported from [colibrì](https://github.com/JustVugg/colibri), a C, hand-written, effectively
zero-dependency inference engine for the same model. rabbit reimplements the same algorithms in
Rust, with a short list of well-justified dependencies instead of colibrì's zero-dep stance —
see `rabbit-plan.md` for the reasoning on each one — plus its own performance work on top (a
measured 3.5x from parallelizing the matmul kernels with `rayon`, ~29% from parallelizing the
absorbed-attention decode path, KV-session and expert-usage persistence colibrì doesn't have in
this exact form). Validated against a synthetic tiny GLM-5.2 oracle (token-exact teacher-forcing,
the same self-test colibrì uses) and against the real 744B checkpoint.

The name is a nod to [RabbitLLM](https://github.com/ManuelSLemos/RabbitLLM), an earlier,
unrelated project of mine (a fork of AirLLM that streams full model layers through limited GPU
VRAM) — same broad interest in running large models on constrained hardware, different problem
and a completely different technique (CPU-side quantized MoE expert streaming here, vs. GPU
layer streaming there). Nothing in this codebase is derived from that one.

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

## Status

All phases through Fase 15 are complete (see `rabbit-plan.md`). Not yet built: a standalone
`.qs` converter (task tracked as Fase 10 — currently relies on pre-converted checkpoints), live
expert re-pinning, GPU/CUDA, MTP speculative decoding, ARM NEON, and grammar-constrained decoding
— none of these are required for the engine to work, they're future scope.

## Versioning

Pre-1.0, `0.MINOR.PATCH`: `MINOR` bumps at the end of each phase in `rabbit-plan.md` (`v0.1.0`
through `v0.15.0`), `PATCH` for fixes/polish within a phase.

## License

TBD.
