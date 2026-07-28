# rabbit — Rust port of colibrì

## Context

[colibrì](../laboratory/colibri) is a pure C inference engine (2,574 lines in `c/glm.c` plus a
few small headers) that runs GLM-5.2 (a 744B-parameter MoE model) on a 25GB-RAM laptop by
streaming routed "experts" from disk on demand. The user wanted a Rust reimplementation, called
**rabbit**, living at `~/Documents/ferrumox/rabbit` (a sibling of their other Rust project,
`ferrumox/fox`), with its own git history from the initial commit.

Scope agreed for this first stage: **the complete engine** (safetensors, BPE tokenizer, config,
quantization, MLA+DSA attention, MoE with expert streaming, basic decoding) — **no CLI, no
`serve`, no CUDA backend yet**. Validated against the synthetic "tiny" model colibrì itself uses
for its self-test (not against the real 370GB GLM-5.2). Dependency philosophy: minimal and
idiomatic — a few well-chosen crates instead of a religious zero-dependency stance, but without
dragging in the whole Rust ML ecosystem.

This document is the plan for that first stage. CLI/serve/CUDA/web UI are left for later stages
(see "Stage 2" at the end).

## Status (updated 2026-07-12)

**Stage 1 (Phases 0-8, this plan as originally written): complete.** All 8 phases are ported,
tested, and committed — see `git log --oneline`. Validated against the synthetic oracle (32/32
teacher-forcing, 20/20 greedy decode) and the real tokenizer (18/18 cases).

**Stage 1.5 (post-plan, not originally scoped but done before this update): complete.** In
order:
- **Native FP8** (E4M3 + block-scale) and the **`.qs` fast path**: reading real checkpoints
  directly with no external converter, across all three loading paths (sequential, io_uring,
  direct `qt_load`).
- **Real CLI** (`src/main.rs`, previously a placeholder): `--model`/`--prompt`/`--max-tokens`/etc.,
  per-token progress on stderr, expert-cache hit/miss/I/O-time reporting.
- **End-to-end validation against colibrì's real checkpoint** (378GB,
  `jlnsrk/GLM-5.2-colibri-int4`, downloaded to `~/Documents/ferrumox/models/`): generates
  coherent, correct text ("The capital of France is" -> " Paris. It is located").
- Found and fixed a real bug the synthetic oracle never triggered: `moe()` assumed a forward
  pass's whole batch of unique experts always fit in the expert cache; it now dispatches in
  chunks sized to capacity (`b203fb6`).
- **`rayon` parallelization** of every matmul kernel (previously single-core SIMD only,
  ~10-20x slower than colibrì in practice) — **a measured 3.5x against the real checkpoint**,
  bit-exact same output (`ab8cec2`).
- Diagnosed where the time goes: prefill ~75% disk I/O (cold cache), steady-state decode
  ~65-70% compute / 30-35% I/O — for long generations compute already dominates, and it's
  already parallelized.

No further performance work was pursued this round (speculative cross-layer expert prefetch,
per-expert matmul fusion) — an explicit decision given the uncertain/diminishing returns against
the 3.5x already achieved. See the project's memory notes (`project_rabbit_glm52_port.md`) for
the full reasoning.

## Dependency mapping: philosophy revised after reading the code

Reading `c/st.h` closely: colibrì **deliberately avoids mmap**. It uses `pread` +
`posix_fadvise(DONTNEED)` because mmap leaves pages resident that corrupt peak-RSS measurement
(see the comment in `st.h`'s header — an "RSS bug"). This is central to the streaming
architecture (the engine needs to know exactly how much RAM it's using to avoid triggering the
OOM killer). So **rabbit must not use `memmap2`** for the safetensors reader — it replicates the
`pread`/`fadvise` pattern with `std::os::unix::fs::FileExt::read_at` plus the `libc` crate (for
`posix_fadvise`, `O_DIRECT`, `getrusage`).

Proposed dependencies (each justified 1:1 against a real need from the original):

| crate | replaces | why |
|---|---|---|
| `serde` + `serde_json` | `json.h` (149 lines, hand-rolled JSON parser) | The original admits it's incomplete ("no full \uXXXX unicode support"). For config/safetensors-header/tokenizer.json files, a robust, battle-tested parser removes a whole class of subtle bugs at no real cost — it's not a performance-critical component. |
| `libc` | `pread`, `posix_fadvise`, `O_DIRECT`, `getrusage` (RSS) | Without this there's no way to replicate the fine-grained page control that's the engine's whole point. |
| `rayon` (from Phase 4 on) | `#pragma omp parallel for` (attention, DSA top-k, expert application) | An idiomatic 1:1 mapping of OpenMP. Deferred until Phase 3 (single-thread correctness) is validated — the same order colibrì itself built in: correctness first, parallelism measured after. |
| `io-uring` (from Phase 8) | N threads blocked on `pread` for expert streaming | See Phase 8. Replaces the thread-per-read pattern with asynchronously queued O_DIRECT reads — more natural in Rust than in portable C. |

Everything else (BPE tokenizer, quantization containers, AVX2 kernels, MLA, DSA, MoE router,
expert LRU cache) is hand-written, same as the original — these are the project's algorithmic
core, and replicating the exact logic (not just the result) is what enables token-exact
validation against the oracle.

## Project structure

```
ferrumox/rabbit/
├── Cargo.toml                  # single-crate workspace for now (bin + lib)
├── .gitignore
├── README.md                   # vision + status, mirroring colibri's own
├── src/
│   ├── lib.rs
│   ├── json.rs                 # re-exports/wraps serde_json where its own types are needed
│   ├── safetensors.rs          # ~ st.h: shard index, pread, bf16/f16->f32, slice reads
│   ├── config.rs               # ~ load_cfg: Cfg struct from config.json
│   ├── unicode_tables.rs       # ~ tok_unicode.h: \p{L}/\p{N}/\s tables (ported, not generated)
│   ├── tokenizer.rs            # ~ tok.h: byte-level BPE, cl100k pretokenizer, added tokens
│   ├── quant.rs                # ~ QT container + quantize_rows/pack_int4/pack_int2
│   ├── kernels.rs              # ~ matmul*/dot_i8i8/dot_i4i8: scalar + AVX2 + AVX-512/VNNI
│   ├── model.rs                # ~ model_init/qt_load/expert_load: dense weights + expert index
│   ├── attention.rs            # ~ attention(): absorbed/dense MLA + DSA lightning indexer
│   ├── moe.rs                  # ~ moe(): sigmoid/noaux_tc router + batch union + expert cache
│   ├── expert_cache.rs         # ~ ESlot/tier.h: LRU + pin slots (no .coli_usage persistence yet)
│   └── generate.rs             # ~ step/step_all/layers_forward: forward loop, greedy/temp sampling
├── tests/
│   ├── oracle/
│   │   └── make_glm_oracle.py  # adapted from colibri/c/tools/make_glm_oracle.py (self-contained)
│   └── teacher_forcing.rs      # integration test: 32/32 positions against ref_glm.json
└── tools/
    └── fetch_tokenizer_fixture.py  # downloads ONLY tokenizer.json+config.json from the real HF repo (~MBs)
```

`kernels.rs` uses `std::arch::x86_64` with `unsafe` plus runtime
`is_x86_feature_detected!(...)` (`avx512vnni` > `avx2` > scalar) — no external SIMD crate, the
same hand-written approach as the original C. The AVX-512/VNNI/BF16 intrinsics
(`_mm512_dpbusd_epi32`, `_mm512_dpbf16_ps`) are already confirmed stable in Rust and detected on
this development machine (see Phase 7).

## C-to-Rust module map

| C source | Rust destination | Migration notes |
|---|---|---|
| `json.h` | `serde_json::Value` directly | No logic ported; uses the crate as-is. |
| `st.h` | `safetensors.rs` | `HashMap<String, usize>` replaces the hand-rolled hash map (`st_hash`/open addressing); same index scheme (name -> tensor, absolute offset, dtype, numel). `bf16_to_f32`/`f16_to_f32` are ported bit-for-bit (trivial, and no crate is worth pulling in for 10 lines of bit-twiddling). |
| `tok.h` + `tok_unicode.h` | `tokenizer.rs` + `unicode_tables.rs` | The pre-tokenizer's state machine (cl100k pattern rules 1-7) ported literally — the easiest place to introduce subtle bugs, so it's ported line by line, not "reinterpreted". |
| `load_cfg` in `glm.c:636` | `config.rs` | Same `Cfg` struct (matching field names makes it easier to diff against the C while debugging). |
| `qt_alloc/qt_fill/quantize_rows/pack_int4/pack_int2` | `quant.rs` | `QT { bits: u8, data: Vec<u8>, scale: Vec<f32>, rows: usize, cols: usize }` container. |
| `matmul/matmul_q/matmul_i4/matmul_i2/dot_i8i8/dot_i4i8/matmul_qt` | `kernels.rs` | Phase 3: scalar version first (correctness), then AVX2 intrinsics ported side by side with the C (`maddubs` -> `_mm256_maddubs_epi16` etc.), picking a kernel at runtime the same way `g_i4s` does. |
| `model_init/qt_load/qt_from_disk/embed_row/ld` | `model.rs` | Loads dense-resident tensors; for experts it only stores offsets (not read until `expert_load`). |
| `expert_load/expert_prefetch` | `model.rs` (or `expert_cache.rs`) | `st_read_raw`/`st_read_slice_f32` equivalents via `safetensors.rs`. |
| `attention()` (glm.c:1006) | `attention.rs` | Two paths: weight absorption (decode, S<=4) and dense reconstruction (prefill); DSA lightning indexer with top-k selection. The full logic is ported, including the `DSA_FORCE` path for the "full selection = dense attention" test. |
| `moe()` (glm.c:1163) | `moe.rs` + `expert_cache.rs` | Sigmoid router with correction bias, adaptive top-k/top-p, union of the batch's unique experts, application in blocks of 64, LRU promotion. The persistent learning cache (`.coli_usage`) is left for a later phase — not needed to pass the tiny oracle (every expert fits in RAM there). |
| `dense_mlp()` | `moe.rs` (free function) | GLM-5.2's first 3 dense layers. |
| `mtp_*`, `ngram_draft`, speculative decoding | **out of scope for this stage** | The base forward pass is validated against the oracle first (which doesn't exercise MTP); MTP is a separate module layered on top of an already-correct decode. |
| `step/step_all/layers_forward/kv_alloc` | `generate.rs` | Generation loop, compressed MLA KV-cache, temperature+nucleus sampling. |
| `backend_cuda.*` | out of scope for this stage | — |
| `coli` (Python CLI), `openai_server.py` | out of scope for this stage | — |

## Phases (each one is a compilable, verifiable increment)

**Phase 0 — Scaffold**
`cargo init` in `ferrumox/rabbit`, `git init`, `Cargo.toml` with 2021/2024 edition, `.gitignore`
(target/, test *.safetensors). README as candid about the project's state as the original's.

**Phase 1 — safetensors + config + JSON**
`safetensors.rs` (shard index + pread-based reading of F32/F16/BF16 tensors + slices) and
`config.rs`. Test: index and read a hand-generated toy `.safetensors` (via numpy) and compare
bytes.

**Phase 2 — Tokenizer**
`tokenizer.rs` + `unicode_tables.rs`, ported line by line from `tok.h`/`tok_unicode.h`.
Validation: `tools/fetch_tokenizer_fixture.py` downloads **only** `tokenizer.json` (a few MB, no
weights) from GLM-5.2's HF repo, and a Python script using HF's `tokenizers` crate generates
`text -> ids` cases (equivalent to `tests/test_tok.c`'s flow, which also doesn't run in
colibri's default `test-c` — it's manual validation against the real tokenizer). Encode/decode
round-trip is also checked.

**Phase 3 — Quantization + kernels (scalar first)**
`quant.rs` + `kernels.rs` with a pure scalar implementation (no AVX2 yet). Unit tests:
`pack_int4`/`pack_int2` bit-identical against their own dequantization, the same way colibri
validates it ("Packing validated bit-identical to the int8 container").

**Phase 4 — Dense model + MLA/DSA attention**
`model.rs` (dense tensor loading) + `attention.rs`. This is where `rayon` enters, for the
`collapse(2)` head/position loops. Verification: generate the tiny model with
`tests/oracle/make_glm_oracle.py` (requires `pip install torch transformers safetensors
huggingface_hub numpy` once — a dev-only dependency, never at runtime, same as colibri) and run
**teacher-forcing** over `full_ids` against `tf_pred` in `ref_glm.json` (already in colibri's
repo, 32 positions). Since the tiny oracle has `index_topk=4096 >> seq_len`, DSA selects the
entire context — this test validates MLA plus DSA-as-a-no-op without needing the real selection
path implemented yet.

**Phase 5 — MoE + expert cache**
`moe.rs` + `expert_cache.rs`. Since the tiny model has only 8 experts/layer, batch union and the
LRU get exercised but never forced into a real miss under RAM pressure — still enough to
validate routing logic (sigmoid router with bias, top-k) and accumulation arithmetic. Same
teacher-forcing test as Phase 4, now over the full forward pass (dense + MoE).

**Phase 6 — Generation**
`generate.rs`: a simple autoregressive loop (greedy + temperature/nucleus), KV-cache.
Verification: greedy generation from `prompt_ids` must reproduce `full_ids[len(prompt_ids):]`
from `ref_glm.json` (the same "32/32" criterion colibri's `setup.sh` uses).

**Phase 7 — Vectorized kernels (AVX2 + AVX-512/VNNI)**
Back to `kernels.rs`, in two steps over the Phase 3 scalar baseline:
- **AVX2** ported side by side with the original's `matmul_q_idot`/`matmul_i4_idot`/
  `dot_i8i8`/`dot_i4i8` — the portable floor, runs on any x86-64 since ~2013.
- **AVX-512/VNNI** as an additional runtime-selected tier. Validated on this development
  machine (Ryzen AI 9 HX 370): `_mm512_dpbusd_epi32` (int8 VNNI dot product) and
  `_mm512_dpbf16_ps` (BF16) **compile on stable Rust** (no nightly needed) and the CPU exposes
  them at runtime (`is_x86_feature_detected!("avx512vnni")` -> true). This is exactly the next
  performance jump colibrì identifies as unexploited in its own C engine (AVX2 only) — rabbit
  adds it from this phase instead of deferring it.

Runtime kernel selection: `avx512vnni` > `avx2` > scalar, the same idea as the original's
`g_i4s` (decided by measured shape/hardware, not at compile time).

Verification: the same teacher-forcing tests across all three tiers (must still give 32/32 on
all of them — integer quantization is exact, no floating-point tolerance to mask a kernel bug)
plus `cargo bench` comparing scalar vs AVX2 vs AVX-512/VNNI.

**Phase 8 — io_uring for expert streaming**
Phase 5's expert reader (`expert_load`/`expert_prefetch`) starts out with simple synchronous
`pread` (same as `st.h`), enough for correctness against the tiny oracle. This phase replaces it
in MoE's hot path with `io_uring` (the `io-uring` or `tokio-uring` crate): queuing the O_DIRECT
reads for a whole block of up to 64 unique experts with a handful of syscalls instead of N
threads blocked on `pread`, cutting context-switch overhead right where it dominates per-token
cost. This is an improvement a Rust rewrite enables naturally and that was outside the original's
"portable C, zero dependencies" scope. Verification: same result as Phase 5 (teacher-forcing
32/32) — it's an I/O mechanism change, not a logic change — plus a microbenchmark of reads/second
against the plain `pread` version.

## Performance ideas for future stages (backlog, not committed to in this plan)

Techniques evaluated but **not** turned into phases because their benefit isn't validated
(unlike AVX-512/VNNI and io_uring above, which were confirmed viable on this machine) — these
remain hypotheses to test later, once the base engine works:

- **Heat-adaptive quantization**: "hot" experts (already prioritized by the learning cache) at
  int8/int4, "cold" experts (most of the 21,504) compressed more aggressively to int2 — reduces
  the cost of a real miss without touching quality where it matters most. Depends on the
  persistent learning cache, already out of scope for this stage.
- **Co-activation-based expert reordering on disk**: profile which pairs/triples of experts
  route together often and repack them contiguously in the file — turns random reads into more
  sequential ones. An offline layout optimization, not a hot-path one.
- **Multiple usage profiles** (instead of a single global `.coli_usage`): pre-pin the right
  profile based on conversation type before the first pass. Also depends on the deferred
  persistent learning cache.
- **Small external draft model** as speculation complementary to native MTP (the standard
  *speculative decoding* technique with an auxiliary model). MTP itself is also out of scope for
  this stage.
- **System tuning** (dedicated `ionice`/cgroup, NVMe block `nr_requests`/readahead tuned to the
  19MB random-read pattern): not engine code, it's deployment guidance — document in rabbit's
  README once it exists.

## End-to-end verification for this stage

1. `cargo test` runs all unit tests (safetensors, tokenizer round-trip, packing).
2. `python3 tests/oracle/make_glm_oracle.py` generates `tests/oracle/glm_tiny/` and confirms
   `ref_glm.json`.
3. `cargo test --test teacher_forcing` loads `glm_tiny`, runs a teacher-forced forward pass, and
   requires an exact match across all 32 positions — the same threshold colibri's `setup.sh`
   uses as its architecture self-test.
4. Greedy generation from `prompt_ids` reproduces the rest of `full_ids`.
5. (Phase 2) Tokenizer: cases generated against the real tokenizer.json, not the tiny one.

## Stage 2 — next steps

Feature-by-feature against colibrì (2026-07-12): the inference engine is at parity (same
capabilities, now validated against the real checkpoint), but rabbit is still a single-turn,
single-prompt binary today — it's missing the whole "product" layer (chat, server, tooling)
colibrì already has. Proposed phases, ordered by practical value / effort:

**Phase 9 — Real chat (template + multi-turn loop): complete (2026-07-12, `e036737`).**
The downloaded checkpoint doesn't ship `chat_template.jinja` (only `tokenizer_config.json`, no
embedded template) — GLM-5.2's official template (`[gMASK]<sop><|user|>...<|assistant|>
<think></think>`, no newline between roles) was taken from colibrì (`c/glm.c`), which already
has it validated against this same checkpoint. `rabbit --chat` keeps `KvState`/`ExpertCaches` in
memory across turns (real continuation via `pos_base`, never reprocessing the whole conversation
each time), supports `:reset`/`:quit`, and `--think` for the reasoning block (nothink by
default, as colibrì warns: the wrong template makes the model never emit the stop token).
Validated with a real two-turn conversation where the second turn ("What is its population?")
depends on the first without repeating "Paris" — confirms real context continuation, not just
absence of a crash. **Not implemented in this pass** (system prompt was resolved in Phase 11, KV
persistence in Phase 13 — see below).

**Phase 10 — Own `.qs` converter**
Already tracked as task #20. Reads any source format (F32/BF16/F16/block-scale FP8, all already
supported by `Shards::read_f32`) and writes the `.qs` container using the same quantization math
as `quant.rs` (`QT::fill`/`pack_int4`/`pack_int2`) — equivalent to `convert_fp8_to_int4.py`,
removing the dependency on third-party pre-converted checkpoints.

**Phase 11 — OpenAI-compatible HTTP server: complete (2026-07-12, `c7a12bb`).** Detailed plan in
`~/.claude/plans/magical-wiggling-hennessy.md`. `rabbit --serve`: `POST /v1/chat/completions`
(streaming and non-streaming) plus `GET /v1/models`, via `tiny_http` (sync, no async runtime —
the engine is inherently blocking, a single `Model`/`ExpertCaches` in memory, no real benefit
from async). Single-threaded accept loop with no explicit queue (the TCP backlog already fills
that role). No conversation memory across requests (stateless, like the real OpenAI API) — every
request re-tokenizes the full `messages` array from scratch. `Session`/`generate_reply`/template
extracted into `src/chat.rs` (a library), shared with `--chat`. System prompt implemented
speculatively (`<|system|>{content}`) — validated with a real request against the checkpoint
(replied "Paris." and stopped cleanly, no rambling). Validated end to end: non-streaming,
streaming (coherent incremental text), system message, 401 auth, 400 errors — against the real
checkpoint and the synthetic oracle.

**Deliberately out of v1** (see the plan for the reasoning behind each): legacy
`/v1/completions`, an explicit admission queue (`--max-queue`), multi-session `--kv-slots` (no
session identity across requests in a stateless design).

**Phase 13 — KV-cache persistence across restarts (`--session`, equivalent to `.coli_kv`):
complete (2026-07-12).** `--chat` only (`--serve` stays stateless, a decision already made in
Phase 11 and untouched here). Append-only format in `src/kv_session.rs` (new module): a header
written once (magic + per-layer dims), then one record per completed turn with `KvCache`'s new
L/R rows and the DSA indexer's new K rows — never rewrites the whole file, so a long
conversation's total I/O grows linearly, not quadratically (an early design rewrote everything
every turn; a planning agent caught the problem before any code was written, see
`~/.claude/plans/magical-wiggling-hennessy.md`). `load()` validates hard against the `Model`
(layers/dims/DSA presence — never silently falls back to an empty session) and recovers from a
final record truncated by a mid-write crash by discarding it entirely. A real bug found during
testing (not in `kv_session.rs`, in the format's design): `layers_forward`'s DSA indexer is only
computed on a session's first call (`pos_base==0`) — on later turns each layer's `DsaCache`
stops growing even though the dense `KvCache` keeps growing. The format assumed lockstep growth;
fixed by storing a per-layer count of new DSA rows per record, independent of the turn's token
count. Verified: bit-identical round-trip against an uninterrupted continuation, append
semantics (two partial saves = one combined), validation errors (bad magic, mismatched
dimension), recovery from a record truncated mid-write — all five via `cargo test`, plus a
manual test against the real checkpoint (378GB): a two-turn conversation, `:quit` mid-way,
reopened with the same `--session`, the model recalls the first turn's information without
having reprocessed it.

**Phase 14 — Persistent expert usage learning cache (`.rabbit_usage`, equivalent to
`.coli_usage`): complete (2026-07-12).** The user explicitly requested this ahead of the `.qs`
converter, MTP, and GBNF. Automatic (no flag, like colibrì) across all 3 modes
(`--prompt`/`--chat`/`--serve`): `<model_dir>/.rabbit_usage`, plain text
`"{layer} {eid} {count}\n"` (unlike `kv_session.rs`'s binary format — here the file is small,
written once per turn, and plain text is trivially inspectable), atomic writes via
temp-file-plus-rename. Each `ExpertCache` layer gains a `pinned` tier separate from the LRU
(checked first in `get`/`get_or_load`/`begin_loading`, never touched by eviction) and a
per-expert `usage` counter, bumped in `moe.rs` right after the router's top-k (the same point as
colibrì's `eusage[layer][eid]++`). On startup (`ExpertCaches::warm_start`, called from
`chat::load_session`), if accumulated history exceeds 5000 selections, it marks up to
`floor(cache_capacity * 0.5 * confidence)` experts per layer as pin **candidates**
(`confidence = min(1, hist/200_000)`) — the same thresholds as colibrì, with the budget expressed
in expert count (the `--expert-cache` unit) instead of GB (rabbit has no `cap_for_ram`/`RAM_GB`,
and building one is out of scope). `--no-usage-cache` disables all of it. Out of scope (an
explicit decision): live re-pinning (`REPIN`/`eheat`) and VRAM/CUDA promotion.

**A deliberate deviation from colibrì, found by real measurement, not analysis**: colibrì's
`pin_load` loads its candidates EAGERLY (synchronously, at startup). A real A/B against the
checkpoint (`--prompt`, one process per invocation) showed that replicating this makes total
wall-clock worse, not better: 108.5s with 975 experts preloaded vs 104.3s with no cache (same OS
page-cache state in both) — every invocation pays the full cost of loading hundreds/thousands of
experts even if that particular prompt never uses them. The design was changed to **lazy/sticky
promotion**: `warm_start` only MARKS candidates (`ExpertCache::mark_pin_candidates`), loading
nothing; only once a candidate is actually loaded through the normal path (a real miss during
generation) does `insert_or_pin` promote it to the `pinned` tier. Zero wasted I/O, the same
eviction protection once an expert is genuinely used. Re-verified after the change: with the
cache (225 marked candidates) 106.6s vs without it 106.4s — the regression is gone (difference
within noise), and there's no more eager loading to log. Still unmeasured whether lazy pinning
gives a real gain in a long-running `--chat`/`--serve` session (where amortization would
actually apply) — the next step if this thread gets picked back up.

Verified: 18 tests (`usage_cache.rs`: round-trip, corrupted lines, threshold/confidence;
`expert_cache.rs`: a candidate loads nothing until first real use, gets promoted and survives
LRU pressure, marking late doesn't retroactively promote, marking twice doesn't duplicate;
`generate.rs`: `warm_start` seeds counters and marks candidates respecting the budget, without
loading anything; `moe.rs`: numeric output is identical whether or not a candidate was
pre-marked — pinning is purely bookkeeping). Manual test against the real checkpoint (378GB),
two rounds: the first confirmed the mechanism pins where it should (150 experts, exact formula)
but was slower in `--prompt` with the original eager design; the second, after the lazy
redesign, confirmed the regression was gone.

**Phase 15 — Parallelize the "absorbed" attention path (decode): complete (2026-07-12).**
Phase 12's original item ("parallelize `qt_addrow`/`qt_matvec_rows`") turned out to be
imprecise once investigated: those two functions do FIXED-cost work per call (`O(kv_lora)`,
`O(vh*kv_lora)`), not scaling with context length. What does scale with `nt` (KV-cache
positions) are the scoring and `clat` accumulation loops AROUND them, inside the same
`for hh in 0..h` in `attention.rs` — fully sequential until now. The WHOLE head loop was
parallelized (`par_chunks_mut` over `ctx`'s slice per `si`, each head writing a disjoint
slice — no cross-head reduction, so the result is bit-identical to the sequential version,
verified with `absorbed_path_matches_dense_reconstruction_path` and the real-oracle tests). No
`with_yt_scratch`/transpose needed: `ctx` is already contiguous and disjoint per head, direct
`par_chunks_mut` is enough.

**Measured, not estimated — and the first measurement attempt failed for a real reason**:
testing with a single large prefill of thousands of tokens (to cheaply simulate a long context)
doesn't work on this engine — a large batch touches nearly all ~21,504 of the model's experts at
once, triggering a read of close to the disk's full 370GB (a 401-token prompt didn't finish in 5
minutes). Long context has to be built through real decode instead (the LRU cache stays warm
between steps, each step only adding ~8 new misses). With that: a run of 70 decode tokens,
before/after the change, same prompt, same seed — **224.3s -> 158.4s (~29% faster)**,
bit-identical generated text in both runs. The per-step gain GROWS with position within the same
run (28% at token 1, ~nt=17; 42% at token 69, ~nt=86), confirming the analysis's prediction.
Context of thousands of tokens couldn't be measured (would take hours of sequential decode), but
the growing trend in this range already confirms the direction — left as future work if the
exact number at large `nt` is needed.

**Phase 16 — Per-turn profiling endpoint (`GET /profile`): complete (2026-07-18).** Prompted
by reviewing colibrì's own (unmerged `dev`-branch) profiling page,
which streams a per-turn phase-timing breakdown (disk service, I/O wait, expert matmul,
attention, lm_head) to a React dashboard. Rabbit had most of the raw data already:
`chat.rs`'s `generate_reply` was already reporting wall time, hits/misses, and `io_seconds`/
`io_wait_seconds` per step via `GenEvent` (the latter already isolating the genuine `io_uring`
stall from decode/copy overhead — see `expert_cache.rs`'s `io_wait_nanos` doc). Missing: a
breakdown of `attention_s`/`expert_matmul_s`/`lm_head_s`, and an HTTP surface.

Added `generate::Phases`/`StepProfile`, threaded as an `Option<&mut Phases>` through
`layer_forward`/`layers_forward` (zero-cost when `None`, so `step`/`step_all` and their ~15
existing test call sites are untouched); a new `step_profiled()` entry point times the
attention call, the dense/MoE FFN dispatch (splitting a MoE layer's FFN time into
`expert_wait_s`, from `ExpertCache::io_wait_nanos`'s before/after delta, and
`expert_matmul_s`, the remainder), and the final lm_head matmul — used only by
`chat.rs::generate_reply`, which now returns a `TurnProfile` (wall time, prompt/completion
tokens, hits/misses, the four phases, forward-pass count) for every caller. `Session` gained a
120-turn rolling `VecDeque<TurnProfile>`, pushed by `server.rs`'s two chat-completion handlers
(not by `generate_reply` itself — CLI callers just ignore the returned profile) and served at
`GET /profile` as JSON. No page serves it (yet): a hand-rolled `assets/dashboard.html`
(vanilla HTML/CSS/JS, no build step, styled after colibrì's real dashboard tokens/components)
went through several rounds and was ultimately pulled back out — verified working end to end
against the real checkpoint, but not judged good enough to keep, and the user decided a web UI
for this (and possibly other projects) deserves its own separate repo rather than living inside
rabbit's single-binary constraints. The reference-design research and a full handoff brief for
that future UI work (including this dashboard's postmortem) live in `DASHBOARD_BRIEF.md`.

Verified: `cargo test` (123 passing, including a new `step_profiled_reports_nonzero_...` test
confirming the timers actually fire on the 3-layer dense/MoE fixture), `cargo clippy` clean, and
the endpoint itself manually verified against the real checkpoint (curled `/profile`, sanity-
checked that the phase totals stay under the turn's wall time).

Deliberately out of scope here (see `DASHBOARD_BRIEF.md` for the full discussion): a
"Brain"-style expert routing heatmap and a `/health`-style runtime/hardware overview —
rabbit has no GPU/VRAM tier and no request scheduler/KV-slots the way colibrì does, so both
need real adaptation, not a straight port; left for whatever picks up `DASHBOARD_BRIEF.md`.

**Phase 12 — Performance, round 2 (the rest, not tackled)**
- Fusing a layer's active experts' matmuls into fewer parallel calls (today 8 experts x 3
  matmuls = 24 separate dispatches per layer) — estimated modest gain (~5-15%), requires being
  able to concatenate heterogeneous `QTKind`/scales across experts.
- Speculative cross-layer expert prefetch (colibrì's "pilot" thread, ~71.6% hit rate) — the
  hardest one: you don't know which experts the next layer will need until the current one
  finishes, so it needs prediction with a correct fallback on a miss.
- Live re-pinning (`REPIN`/`eheat`, `tier.h`'s decay+swap between turns) — the half of Phase 14
  NOT ported (that phase ported `eusage`/persistent auto-pin; `eheat` remains pending, and
  colibrì itself ships it off by default too).

**Undated backlog (evaluate if worth it when we get there)**
- Native MTP / speculative decoding (`DRAFT=n`/`MTP=1`) — **correction, 2026-07-17: the `jlnsrk`
  checkpoint DOES ship the MTP head** (confirmed directly against the checkpoint's own header:
  `model.layers.78`, ~5GB across the `out-mtp-*` shards, `eh_proj`/`enorm`/`hnorm` + its own
  experts — this line previously claimed otherwise). See `ROADMAP.md` for the current thinking on
  this idea.
- GPU/CUDA backend — colibrì itself notes there's no proven end-to-end speedup there yet, so
  it's not an obvious performance win, more a different deployment path.
- ARM NEON (rabbit is x86_64-only today).
- Grammar-constrained decoding (GBNF).
- Web UI.
