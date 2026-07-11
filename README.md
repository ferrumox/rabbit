# rabbit

A from-scratch Rust port of [colibrì](https://github.com/JustVugg/colibri)'s inference engine
for GLM-5.2 (744B-parameter MoE): the dense part resident in RAM at int4, the 21,504 routed
experts streamed on demand from disk. Part of the [ferrumox](../) AI lab, alongside
[fox](../fox) (a production local-LLM server wrapping llama.cpp) — rabbit is the opposite kind
of project: a research engine for a model that doesn't fit in memory even offloaded, built by
hand instead of wrapping an existing runtime.

**Status: Phase 0 — scaffold only. There is no engine yet.** No tensors load, no tokens
generate. This is not a usable tool today; it's the first commit of a multi-phase port. See
[`rabbit-plan.md`](../rabbit-plan.md) for the full phase breakdown and what "done" looks like
for the current stage (teacher-forcing exact match against a synthetic tiny GLM-5.2 oracle, the
same self-test colibrì uses).

## Why

colibrì is C, hand-written, zero-dependency by philosophy. rabbit ports the same algorithms
(MLA attention, DSA sparse attention, MoE router with expert streaming, byte-level BPE
tokenizer, int4/int8/int2 quantization) to Rust with a short list of well-justified
dependencies instead of a religious zero-dep stance — see the plan for the reasoning on each
one. The port is validated the same way colibrì validates itself: token-exact teacher-forcing
against a tiny synthetic model with the real architecture, not against the 744B model (~370GB,
impractical for day-to-day development).

## Building

```bash
cargo build
cargo test
```

Right now this just builds an empty binary — there's nothing to run.

## Roadmap

See [`rabbit-plan.md`](../rabbit-plan.md): safetensors + config → tokenizer → quantization →
MLA/DSA attention → MoE + expert cache → generation → AVX2 → AVX-512/VNNI → io_uring expert
streaming. CLI, an OpenAI-compatible server, CUDA/GPU pinning, MTP speculative decoding, and a
web UI are explicitly out of scope until the base engine is validated.

## License

TBD.
