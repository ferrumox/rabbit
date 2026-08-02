# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

rabbit is a Rust CPU inference engine for frontier open-weight MoE models on a single machine.
The dense part of the model (attention, shared experts, embeddings) stays resident in RAM,
quantized; the thousands of routed experts live on disk and stream on demand. It began as a port
of **colibrì** (a C, GLM-5.2-only engine — referenced constantly in comments as "the reference
implementation", "the C", `glm.c`, `st.h`) and has since generalized into three architectures.

No GPU. No async runtime. `pread`, never `mmap` (mmap leaves pages resident and corrupts the
peak-RSS accounting the streaming design depends on — see `src/safetensors.rs`'s module doc).

## Build, test, run

```bash
cargo build --release
cargo test                     # unit + integration; fixture-dependent tests SKIP when absent
cargo test --test teacher_forcing        # one integration test file (GLM-5.2 oracle)
cargo test <name_substring>              # one unit test
cargo bench --bench kernels -- "qt_matvec_rows_i4"   # criterion, filtered
```

```bash
./target/release/rabbit --model <checkpoint-dir> --prompt "..." --max-tokens 40
./target/release/rabbit --model <dir> --chat --session ~/.rabbit_session
./target/release/rabbit --model <dir> --serve --port 8000
cargo run --release --bin convert -- --indir <dir> --outdir <dir> --classify-only
cargo run --release --example k3_smoke -- --model <dir> --max-tokens 5
cargo run --release --example teacher_forced_decode_bench -- --model <dir> --steps 30
```

`--model` auto-detects the architecture from `config.json`'s `model_type`. Kimi K3 (`kimi_k3`)
and Kimi Linear (`kimi_linear`) read Moonshot's published safetensors directly, no conversion.
GLM-5.2 (`glm_moe_dsa`) needs colibrì's converted layout.

There are no lint/format configs in the repo; the codebase is `rustfmt`-shaped but with a wide
line budget (~150 cols) and single-line `match` arms — match the surrounding file, don't
reformat.

### The oracle fixtures

Every architecture is validated **token-exact against a tiny random-weight model built from that
model family's own real reference code**, vendored under `tests/oracle/*_pkg/`. The fixtures are
gitignored and must be generated locally with Python (torch/transformers/safetensors, often
Docker — read each script's own docstring, they document the exact invocation and the patches
they apply):

```bash
cd tests/oracle && python3 make_glm_oracle.py    # -> glm_tiny/  + ref_glm.json
cd tests/oracle && python3 make_kimi_oracle.py   # -> kimi_tiny/ + ref_kimi.json
cd tests/oracle && python3 make_k3_oracle.py     # -> k3_tiny/   + ref_k3.json
cd tests/oracle && python3 make_convert_oracle.py  # -> convert_src/, convert_ref_{row,grouped}/
python3 tools/fetch_tokenizer_fixture.py         # -> tests/fixtures/ (real tokenizer, no weights)
python3 tools/gen_tokenizer_cases.py             # ground truth from the Python `tokenizers` lib
```

**Fixture-dependent tests skip rather than fail when the fixture is missing** — a green
`cargo test` does not mean the oracles ran. Check the output for `SKIP` lines before claiming
correctness. Affected: `teacher_forcing{,_kimi,_k3}.rs`, `convert_oracle.rs`,
`convert_*_end_to_end.rs`, `*tokenizer_fixture.rs`.

## Architecture

### Family dispatch

`src/model.rs` is the single dispatch point: `Model` / `KvState` / `ExpertCaches` / `Tokenizer`
are each a three-variant **enum** (not a trait object — the `KvState` shapes genuinely differ:
GLM's growing `KvCache`/`DsaCache` vs. Kimi's fixed-size `KdaLayerState` + conv FIFOs). `step` /
`step_all` / `step_profiled` match on the *triple* `(model, caches, kv)`; the `unreachable!` arms
only fire if a caller mixes state from two different `Model::load` calls.

`src/glm52/` is both GLM-5.2's implementation **and** the shared substrate. The two Kimi modules
reuse it unchanged:
- `glm52::moe::{moe, dense_mlp, RouteConfig}` — all three architectures route through this
- `glm52::attention::{attention, rmsnorm, OutputGate, Absorb}` — MLA for all three
- `crate::generate::{StepProfile, Phases, argmax, Rng, SamplingConfig}`

Kimi K3 wraps Kimi Linear rather than reimplementing it: `kimi_k3::config::Cfg` embeds
`kimi_linear::config::Cfg` as `.base` (K3's real `config.json` nests the whole Kimi Linear config
under `text_config`), and only K3's genuinely new pieces are new code — `ops::situ_and_mul`,
`moe::latent_moe` (routed experts compute in a narrower latent width), `attn_res.rs` (Attention
Residuals, transient per forward call, so nothing to persist in a session), MLA/KDA output gates.

So an architecture module is: `config.rs` (raw-JSON-first `Cfg::load`), `model.rs` (tensor
names + `ModelError`), `generate.rs` (`step`/`step_all`/`step_profiled`, `KvState`,
`ExpertCaches`), `kv_session.rs` (own on-disk format + magic bytes), `chat_template.rs`,
`tokenizer.rs`. Adding a fourth means those files plus five arms in `src/model.rs` plus an
`ExpertNaming` variant.

### The streaming path

`generate::step` → per-layer `layer_forward` → `glm52::moe::moe()` → `ExpertCache::get_or_load`.
On a miss, `src/expert_cache.rs` batches the reads: one `io_uring` submission on Linux, a
sequential-`pread` fallback elsewhere and for MXFP4 (which reads a packed+scale tensor *pair* the
ring doesn't model — deliberate, documented tradeoff). `ExpertNaming` (in `expert_cache.rs`) is
what maps a logical `(layer, expert)` to real on-disk tensor names per family, including the
fused-`gate_up` and `language_model.`-prefixed MXFP4 variants.

Tiers, cheapest first: per-layer LRU (`--expert-cache N` slots) → lazily-promoted pins from
`.rabbit_usage` (`src/usage_cache.rs`, a plain-text histogram learned across runs, permissively
parsed — a corrupt file must never block generation) → OS page cache → disk.

`moe.rs`'s **per-expert early drain** applies each miss's contribution the moment its read
completes, so float summation order depends on real disk timing. Free-running greedy decode is
therefore **not reproducible run-to-run** — this is why timing comparisons use
`examples/teacher_forced_decode_bench.rs` (fixed token sequence fed at each step) rather than
"generate N tokens and time it". Read that file's doc before benchmarking anything.

### Quantization and kernels

`src/quant.rs` is the `QT` container: f32 passthrough, int8, packed int4, packed int2, plus
grouped-scale int4, native FP8 E4M3 (block-scale, resolved via a `{name}_scale_inv` sibling
tensor) and OCP MXFP4 read straight off disk. `bits` and storage tier are independent (`bits=3`
lives in the int4 tier). Rounding uses `round_ties_even`, matching C's `lrintf` under the default
IEEE mode — required for token-exact parity, not a style choice.

`src/kernels.rs` dispatches at runtime via `is_x86_feature_detected!`. Parity expectations differ
per kernel and are load-bearing for tests: the integer IDOT path (`dot_i8i8`/`dot_i4i8`) must
agree **bit-for-bit** across scalar/AVX2/AVX-512-VNNI, while `matmul_i4`'s AVX-512 tier
reassociates and is only within-tolerance (and is in fact *more* accurate than scalar). MXFP4's
matmul is scalar-only today — the single biggest known K3 perf lever.

### CLI / server

`src/main.rs` is argument parsing only. `src/chat.rs` holds all session/generation/template logic
shared by `--chat` and `--serve`. `src/server.rs` is a single-threaded blocking `tiny_http`
accept loop with no server-side conversation state; every handler consumes its `Request` exactly
once. `/profile` exposes a rolling window of `PROFILE_TURNS` per-turn phase timings.

## Conventions that matter here

- **Module docs carry the reasoning.** Nearly every file opens with a long `//!` block explaining
  *why* — what colibrì did, what was deliberately not ported and why it doesn't affect
  correctness, what was measured, what was tried and reverted. This is the primary design record.
  Read the module doc before editing a module; extend it when you change a decision.
- **Provenance claims are explicit and literal.** Comments distinguish "confirmed against the
  real checkpoint's `config.json`, fetched 2026-07-27" from a guess, and name the reference source
  (`KimiBlockSparseMLP`'s own code comments, `fla.ops.kda.gate`'s installed source). Do not write
  such a claim unless you actually verified it, and say which artifact you read.
- **Correctness is validated against real reference code, never against intuition.** Reference
  Python is vendored verbatim (`tests/oracle/*_pkg/`, `colibri_convert_fp8_to_int4.py`), not
  reimplemented. New architecture math needs an oracle before it needs optimizing.
- Test names are full sentences: `load_rejects_an_unrecognized_model_type_without_touching_either_loader`.
- Errors are per-family enums with `Display`; the dispatch layer wraps rather than flattens them.
- Some prose (`rabbit-plan.md`, older comment fragments) is in Spanish. Code and new comments are
  English.

## Docs in the repo

- `README.md` — capabilities, measured numbers, what is explicitly *not* built yet.
- `PERFORMANCE.md` — chronological log of every perf technique, **including reverted ones**, plus
  the exact commands and hardware for reproducing each number. Append here after perf work;
  record failures too.
- `rabbit-plan.md` — phase-by-phase development history and the "out of scope" list (MTP /
  speculative decoding, prefetch instrumentation, GPU).
- `DASHBOARD_BRIEF.md` — build brief for a web UI over `/profile`; not implemented.
- `docs/k3-architecture.html` — browsable Kimi K3 architecture explorer (open it directly, no
  server). Layer stack → per-layer dataflow → KDA/MLA/LatentMoE internals, every tensor shape
  derived from a config preset so the real 2.8T checkpoint and the tiny oracle render the same
  diagrams. Its **Matmuls** tab enumerates every matmul in the model with `[N,K]` operands and
  MACs/token, self-checked against the parameter count (they reconcile exactly, so a missing or
  double-counted row shows up immediately). Model architecture only — no streaming or kernel
  material. Two things to know before editing:
  - **Its numbers are a second copy of the config.** If you change `src/kimi_k3/config.rs`'s
    real-shape fixture or the oracle generator, update the `PRESETS` object in its `<script>`.
  - It records the real `kda_layers`/`full_attn_layers` arrays, which **nothing in `src/` or the
    test fixtures does** — the fixture scales to 4 layers and so cannot express them. The real
    layout is every 4th layer plus layer 93, i.e. 69 KDA / 24 MLA, with 0-indexed layers 91 and
    92 both MLA. `K3_OPTIMIZE_BRIEF.md:79`'s "≈ 3 : 1" is that, rounded.

## Versioning

Pre-1.0 `0.MINOR.PATCH`, tracked by git tags and `release/vX.Y.Z` branches. `Cargo.toml`'s
`version` field is not the source of truth. MINOR bumps end a development phase; the release
commit subject describes the phase (`v0.23.0: Kimi K3 as a third architecture, validated against
the real 2.8T checkpoint`).
