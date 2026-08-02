# Brief: Kimi K3 performance work (decode kernel, expert loading, parallelism, warmup)

This is an implementation brief for an AI (or engineer) with **no prior context** on this
project. It was written 2026-07-31 against commit `5853baa` (v0.23.0) after a live diagnosis
session on the target machine. Read it fully before writing any code. Companion documents:
`CLAUDE.md` (repo conventions — binding), `PERFORMANCE.md` (measurement history and method),
`README.md` (capabilities), `DASHBOARD_BRIEF.md` (unrelated scope, but the genre precedent for
this document).

Every `file:line` anchor below was verified at `5853baa`. Line numbers drift; anchors are given
as `path:line` plus a searchable symbol name. **If an anchor doesn't match what you find, stop
and re-read the surrounding module doc before proceeding — do not guess.**

---

## STATUS UPDATE (2026-08-01) — this brief has been EXECUTED; read this before anything else

All phases ran on the owner's instance (commits `a26e5bf`..`62238a1`, **local-only** — that box
cannot push to the GitHub remote). This document is now the historical record of what was
planned; `PERFORMANCE.md`'s "Kimi K3 on the target box" section is the record of what actually
happened. Active follow-on work lives in `NUMA_AMX_BRIEF.md` (rev 2), which supersedes §8's
Phase 6 and absorbs Phase 5's v2. What a new reader must know:

- **Result: 36.8 → 16.9 s/token (2.18×)** on the real checkpoint at default placement/threads,
  and **~1.42 s/token in live serving** once launched correctly (`numactl --interleave=all`,
  `--expert-cache 896 --preload-experts --threads 48`) — the serve-config change alone was
  worth ~4× over a default-cache launch.
- **Phase 3's key finding supersedes this brief's own framing:** the AVX-512 kernel is 3–5× at
  the kernel and ~0% end-to-end, because s=1 matmuls fork ~3072 one-row tasks and hit a
  ~4.8 ms rayon fork/join floor. Decode is **scheduling-overhead-bound first**, bandwidth-bound
  second. Phase 5 v1 only overlaps that floor across 16 experts; **v2 (D5) is now mandatory**
  and is fused into `NUMA_AMX_BRIEF.md` Phase N3.
- **Phase 6 ran as `numactl --interleave=all` only** (measured 1.4× + variance collapse on the
  synthetic fixture) — the zero-code outcome this brief preferred. Compute affinity, the larger
  remaining win, is `NUMA_AMX_BRIEF.md` N1–N3.
- **Errata:** K5's "`get` takes `&mut self`" is wrong — `get` is `&self`; LRU stamping happens
  in `begin_loading` (confirmed during Phase 5). 4d (`--drop-os-cache`) was deliberately
  deferred, not built.

---

## 1. What this is about

**rabbit** runs frontier open-weight MoE models on a single machine: the dense part stays
resident in RAM (quantized), and the routed experts stream from disk through a per-layer LRU
cache (`src/expert_cache.rs`). **Kimi K3** (`moonshotai/Kimi-K3`, 2.8T params, 896 experts/layer,
16 active) is the third and newest supported architecture. Its bring-up was correctness-first
(token-exact against Moonshot's own reference code, see `tests/teacher_forcing_k3.rs`); **zero
performance work has landed for it**. The precedent: GLM-5.2 started from a similarly slow floor
and got 3.5× faster across eight measured versions (`PERFORMANCE.md`).

### The two machines

| | dev laptop (PERFORMANCE.md's numbers) | **target box (this brief's goal)** |
|---|---|---|
| CPU | AMD Ryzen AI 9 HX 370, 12c/24t, AVX-512/VNNI | 2× Intel Xeon 6975P-C (Granite Rapids-AP, custom cloud SKU): 192 physical cores / 384 threads (SMT on), **6 NUMA nodes** (SNC3), full AVX-512 + **AMX** (tile/int8/bf16), KVM guest — details in Phase 0a |
| RAM | 123 GB | **3 TB** (6 NUMA nodes × ~507 GB) |
| K3 checkpoint | (validated, slow) | `/data/hf/hub/kimi-k3` (1.56 TB, 96 shards) |

The target box changes the economics completely: the **entire routed-expert set (~1.32 TiB) fits
in RAM**, so disk streaming — rabbit's whole reason to exist on the laptop — becomes a
warmup-only concern, and the bottleneck moves to compute and memory bandwidth.

### Observed baseline (target box, 2026-07-31, real checkpoint)

```
token 112/200 in 26.8s (5.3s in disk I/O [0.0s actual disk wait] this step;
                        expert cache totals: 95825 hits, 75004 misses)
```

Run: `rabbit --model ... --prompt ... --expert-cache 64` (the default capacity). Breakdown:

- **~21–22 s/token compute**, dominated by the scalar MXFP4 matmul kernel (§3, item K1).
- **~4–6 s/token "disk I/O"** — actually `load_nanos`: sequential, single-threaded expert
  loads (§3, item K4). The bracketed `[0.0s actual disk wait]` is **structurally always zero
  for K3** and means nothing (§3, item K3). Do not interpret it.
- Hit rate 56% at capacity 64. Hits+misses grow by **exactly 1472/token** = 16 experts × 92
  MoE layers (shared experts never touch the cache — they're resident dense weights).
- Effective compute rate: ~170 GFLOP/token ÷ 21 s ≈ **8 GFLOP/s aggregate across 192 cores**
  (~42 MFLOP/s/core). Plain scalar FMA should manage 1–3 GFLOP/s *per core*. The gap is the
  work of this brief.

Dev-box reference points (README): model load 610 s, prefill (7 tok) 412.8 s, decode 50–70 s/tok.

### Goal

Decode on the target box from ~27 s/token to **low single-digit seconds per token or better**,
with warmup measured in minutes instead of hours, without breaking token-exact correctness for
any architecture. Once compute is fixed, decode becomes memory-bandwidth-bound: ~26 GB of expert
weights must cross the memory bus per token no matter what. Phase 0 measures the real bandwidth;
until then the floor estimate (~0.025–0.06 s/token — Granite Rapids-AP carries 12 DDR5
channels/socket, ~1 TB/s-class aggregate) is an **estimate — do not present it as a promise**. Reaching it likely needs the follow-on items in §9 (not in scope).

---

## 2. Architecture numbers (verified)

Config constants asserted against the real checkpoint's `config.json` in
`src/kimi_k3/config.rs:245-307` (`loads_the_real_checkpoints_shape`):

| constant | value |
|---|---|
| `hidden_size` | 7168 |
| layers | 93 total; `first_k_dense_replace: 1` → 1 dense + 92 MoE |
| attention layout | KDA : full MLA ≈ 3 : 1 (`kda_layers`/`full_attn_layers` arrays) |
| `num_experts` / `num_experts_per_token` / `num_shared_experts` | 896 / 16 / 2 |
| `moe_intermediate_size` | 3072 |
| `routed_expert_hidden_size` (latent width) | 3584 |
| `vocab_size` | 163,840 |
| expert quantization | OCP MXFP4 (`quantization_config.format: "mxfp4-pack-quantized"`, group 32) |

Derived (arithmetic, not measured):

- **One routed expert** = gate `[3072,3584]` + up `[3072,3584]` + down `[3584,3072]`
  = 33,030,144 params. On disk/in cache as MXFP4: 16,515,072 B data + 1,032,192 B scale
  = **16.73 MiB/expert**.
- **Per decoded token**: 1472 expert applications, 48.6 G routed params ≈ 97 GFLOP routed,
  ≈ 170 GFLOP total (latent down/up, shared experts, attention, lm_head). **~25.8 GB** of
  expert weight bytes traversed.
- **Cache footprint**: capacity 64 (default, `src/main.rs:58`) ≈ 96 GiB; capacity 896 = full
  residency ≈ **1.32 TiB** experts + ~40 GB dense (at default `dbits=4`).

K3's routed experts run at the **latent width** (3584), not `hidden_size` — routing scores use
the full-width hidden states, expert compute happens after a down-projection. This is why K3 has
its own dispatch function instead of reusing `glm52::moe::moe()` (see K2 below). The two widths
are carried as two `Cfg` values (`cfg_full` / `cfg_expert`) differing in `.hidden`.

---

## 3. Verified code map — read these before touching anything

**K1 — the scalar kernel (the main cost).** `matmul_mxfp4`, `src/kernels.rs:222-242`. Inner
loop per element: byte load, `k & 1` parity branch, nibble mask/shift, `e2m1_decode(nibble)`
(already a table lookup + sign flip, `src/quant.rs:285-288`), and `e8m0_decode(bs[k / 32])`
**recomputed for every element instead of once per 32-block**. Scalar only — the module doc at
`src/kernels.rs:216-221` says so and why ("no real MXFP4 checkpoint to benchmark against yet"
— no longer true). Dispatched from `matmul_qt` at `src/kernels.rs:938`.

**K2 — K3's own dispatch loop (the restructure target).** `latent_moe`,
`src/kimi_k3/moe.rs:64-117`. Routing at full width → `record_selection` → down-proj →
chunked `ensure_loaded` → **sequential** `for &eid in chunk { apply_single_expert(...) }` →
optional RMSNorm → up-proj. Fully synchronous load-then-apply; **no early drain, no io_uring
here**. Its module doc explicitly says the overlap machinery was deliberately deferred. This
means: (a) the restructure never touches `glm52::moe::moe()` (GLM/Kimi-Linear's perf-tuned hot
path — **out of scope, do not modify it**); (b) warm-cache K3 decode is already deterministic,
so a bit-identical acceptance gate works (§8, Phase 5).

**K3 — the meaningless counter.** `io_wait_nanos` is incremented in exactly one place, the
io_uring completion path (`src/expert_cache.rs:754-755`). MXFP4 naming **never gets a ring**
(`ring: None`, `src/expert_cache.rs:610-615`, reason in `ExpertNaming::is_mxfp4`'s doc), so for
K3 the CLI's `[Xs actual disk wait]` (printed at `src/main.rs:151,156`) is always 0.0.

**K4 — sequential loads.** `sequential_fallback`, `src/expert_cache.rs:855-857`:
`misses.iter().map(|&eid| load_expert(...)).collect()` — one expert at a time, one thread.
`load_expert` → `qt_load_mxfp4` (`src/expert_cache.rs:884-900`) does two `read_raw(name,
false)` calls (packed + scale; `false` = no `POSIX_FADV_DONTNEED` afterward) with byte-count
shape checks, then moves the buffers into `QTKind::MxFp4` **with no transcode** — a K3 expert
"load" from page cache is nearly pure memcpy. `Shards::read_raw`/`read_at` are `&self` and
thread-safe (`pread` via `FileExt::read_at`; **never mmap** — see `src/safetensors.rs` module
doc for why that rule exists; it is binding).

**K5 — cache mechanics.** `ExpertCache` (`src/expert_cache.rs:522+`): per-layer, LRU at
`capacity`, plus a `pinned` tier fed lazily from the `.rabbit_usage` histogram
(`warm_start` → `mark_pin_candidates` → `insert_or_pin` promotes on first real load, `mlock`
best-effort). At `capacity == n_experts` eviction is impossible and pinning is pointless.
`get` stamps LRU recency (takes `&mut self`) — relevant borrow puzzle in Phase 5.

**K6 — thread pool.** `configure_thread_pool`, `src/main.rs:286-293`: default =
`num_cpus::get_physical()`. The SMT-off default was measured on the **12-core laptop**
(bandwidth-bound there); do not assume it transfers to 192 cores — Phase 0 sweeps it.

**K7 — build profile.** `Cargo.toml` has **no `[profile.release]` section at all**: defaults
(no LTO, 16 codegen units) and — unless `RUSTFLAGS` says otherwise — a generic x86-64 baseline
for everything outside the `is_x86_feature_detected!` hand-written kernels.

**K8 — existing MXFP4 test coverage.** `src/kernels.rs:1202`
(`matmul_mxfp4_matches_manual_e2m1_decode_across_two_blocks`) and `src/kernels.rs:1384`
(`matmul_qt_mxfp4_matches_manual_dequant_dot_product`); E2M1/E8M0 codec tests in
`src/quant.rs:777+`. **No criterion bench entry exists for MXFP4** in `benches/kernels.rs`.
Critically: **the tiny teacher-forcing oracle does NOT exercise `matmul_mxfp4`** — oracle
fixtures use plain float `.weight` tensors (naming `KimiK3`, not `KimiK3Mxfp4`). The kernel
unit tests + the real checkpoint are the only MXFP4 correctness coverage. Extend the unit
tests; do not lean on the oracle for kernel changes.

**K9 — measurement harnesses.**
- `examples/teacher_forced_decode_bench.rs` — THE decode timing tool. Feeds a fixed token
  sequence (free-running greedy is not run-to-run reproducible on the GLM path by documented
  design; use this harness for all decode timing). **Requires tokenizer files in the model dir**
  (`Tokenizer::load`, line 51).
- `examples/k3_smoke.rs` — load/prefill/decode structural probe on **raw token ids**
  (`--prompt-len`), no tokenizer needed.
- `benches/kernels.rs` — criterion, kernel-level.
- `tests/teacher_forcing_k3.rs` — token-exact architecture correctness vs the tiny oracle
  (fixtures gitignored; regenerate via `tests/oracle/make_k3_oracle.py`, Docker-based — read
  its docstring). **Skips, not fails, when fixtures are absent — a green `cargo test` does not
  mean it ran. Check for SKIP lines.**
- `Shards::open` scans `*.safetensors` in the directory, **sorted by filename, no
  `index.json` needed** (`src/safetensors.rs:211,235`) — the Phase 1 generator relies on this.

---

## 4. Invariants and house rules (binding)

1. **`cargo test` fully green at every phase**, including regenerated oracle fixtures where a
   phase could affect them (state in the PR/commit message whether oracles actually ran or
   skipped).
2. **Exactness contracts are per-path and load-bearing** (`src/kernels.rs` module doc):
   integer IDOT tiers must agree bit-for-bit; float paths with reassociation (AVX-512
   `matmul_i4` precedent) are within-tolerance only. New MXFP4 tiers follow the same split:
   Phase 2's scalar refactor must be **bit-exact**; Phase 3's AVX-512 tier is
   **within-tolerance** with a documented test tolerance.
3. **`pread`, never mmap** (`src/safetensors.rs` module doc — RSS accounting is a design
   pillar).
4. **Module docs carry the reasoning.** Every touched module's `//!` block gets updated when a
   decision it documents changes. Provenance discipline: never write "confirmed against X"
   unless you actually read X; name the artifact.
5. **No new dependencies** without the kind of justification comment `Cargo.toml` already
   models (`fancy-regex`, `base64`). NUMA work uses `libc` or external `numactl` — no `hwloc`,
   no `numa` crates.
6. **`PERFORMANCE.md` gets a dated section per phase** — the command, the hardware, the
   before/after numbers, and **techniques that didn't help** (the "Tried and didn't help"
   section is a deliberate part of the file's value).
7. **Scope discipline — do not touch:** `glm52::moe::moe()` and its early-drain/io_uring
   streaming path; MLA/KDA/attention math; the checkpoint converters; `kv_session` formats;
   versioning (no tags or `release/` branches — the owner cuts releases). One commit per phase,
   descriptive message.
8. Comments in English; test names are full sentences; match the existing wide-line rustfmt
   style of the file you're in.

---

## 5. Measurement protocol

- **Decode timing**: `teacher_forced_decode_bench` only. Before/after via `git worktree` at
  the pre-phase and post-phase commits, identical command lines — the exact method
  `PERFORMANCE.md`'s "Reproducing these numbers" documents.
- **Canonical target-box command** (after Phase 1, substitute the synthetic dir for iteration):

  ```
  cargo run --release --example teacher_forced_decode_bench -- \
      --model /data/hf/hub/kimi-k3 --steps 30 --expert-cache 896
  ```

- **Warm/cold discipline**: run everything twice; report both, label them. First run warms the
  page cache and the expert cache; steady-state claims come from run 2.
- **Kernel timing**: `cargo bench --bench kernels -- mxfp4` (entries added in Phase 2).
- Record `RUSTFLAGS`, thread count, `numactl` invocation, and THP state with every number.

---

## 6. Phase 0 — Baseline + build configuration (no logic changes)

**0a. Record the target box.** Topology was captured 2026-07-31 (owner-provided `lscpu` +
`numactl --hardware`); paste the raw output into the new `PERFORMANCE.md` section verbatim.
Summary of what it shows:

- 2× Intel Xeon 6975P-C (Granite Rapids-AP, custom cloud SKU), 96 cores/socket, SMT on →
  192 physical / 384 logical CPUs. KVM guest (cloud instance).
- **6 NUMA nodes** — SNC3, 3 sub-NUMA domains per socket; 64 logical CPUs + ~507 GB RAM per
  node. Distances: 10 local, 15–17 intra-socket, 21–28 cross-socket; socket 0 = nodes 0–2,
  socket 1 = nodes 3–5.
- Full AVX-512 (F/DQ/BW/VL/VNNI/VBMI2/BF16/FP16) and **AMX** (`amx_tile`/`amx_int8`/
  `amx_bf16`) — Phase 3 and §9's AMX follow-on are both hardware-supported.
- A live free-memory imbalance was captured while a run was resident (node 2: 19 GB free vs
  node 0: 467 GB) — first-touch concentration observed in the wild, exactly what Phase 6
  targets. Note that 1.32 TiB of expert slots cannot fit in one 507 GB node: without
  deliberate placement, allocations spill node-to-node in load order.

Still to record: a memory-bandwidth number (any STREAM-triad binary; record which — Granite
Rapids-AP has 12 DDR5 channels/socket, so ~1 TB/s-class aggregate is plausible), THP state
(`cat /sys/kernel/mm/transparent_hugepage/enabled`), and the guest kernel (`uname -r`).

**0b. Baseline runs** (real checkpoint, current `main`): the canonical command at
`--expert-cache 64` and `896`, plus `k3_smoke` for load/prefill numbers.

**0c. Experiments, zero code** — each one canonical-command before/after:
- `--threads` sweep: 48 / 96 / 192 / 384 (SMT is on; 192 = the physical-core default).
- `numactl --interleave=all` vs default.
- Single-socket control run: `numactl --cpunodebind=0-2 --membind=0-2` with `--threads 96` —
  halves peak bandwidth but removes all cross-socket traffic; tells you what the 21–28
  cross-socket distances actually cost.
- THP `always` vs `madvise` (system toggle).

**0d. Build profile.** Add to `Cargo.toml`:

```toml
[profile.release]
lto = "thin"
codegen-units = 1
```

Do **not** add `panic = "abort"` (the `--serve` accept loop's failure mode changes) and do
**not** commit `target-cpu=native` anywhere (open decision D1) — instead measure
`RUSTFLAGS="-C target-cpu=native"` as part of 0c and document the result.

**Gate:** numbers recorded in `PERFORMANCE.md`; `cargo test` green; no behavior change beyond
the profile.

---

## 7. Phase 1 — Synthetic K3-at-scale fixture (test infrastructure)

Real-checkpoint iteration costs a 1.56 TB load per experiment. Decode cost is per-layer
homogeneous, so a random-weight checkpoint at **real per-layer width** but few layers
reproduces kernel/parallelism/NUMA behavior at ~5–10% footprint.

**Build** `src/bin/gen_k3_synth.rs` (or an example — implementer's call): writes a loadable
K3-shaped checkpoint:

- `config.json`: adapt the template in `src/kimi_k3/config.rs:249-276` (it is condensed from
  the real checkpoint) — parameterized `num_hidden_layers` (default 6: layer 0 dense, then a
  3:1 KDA:full-MLA pattern), all K3 fields on, `quantization_config` present so
  `mxfp4_experts=true` selects `ExpertNaming::KimiK3Mxfp4`.
- All tensor names under the `language_model.` prefix (see `ExpertNaming::KimiK3Mxfp4`'s doc
  and the fixture in `src/model.rs`'s tests for the dense-name pattern).
- Routed experts: real shapes (`[3072,3584]`×2 + `[3584,3072]`) as U8
  `{...}.w?.weight_packed` + `.weight_scale` pairs with the exact byte counts
  `qt_load_mxfp4` checks (`rows*ceil(cols/2)`, `rows*ceil(cols/32)`). Keep E8M0 scale bytes in
  ~[120,130] so dequantized magnitudes stay sane — logits are garbage by design, but NaN/inf
  poisoning is avoidable noise.
- Dense/KDA/attention tensors at real dims, F32 or BF16 (D6). One shard file per layer;
  `Shards::open` needs no index (K9). At 6 layers expect ~90–100 GB.
- Reuse `glm52::convert::writer::write_safetensors` if its API fits raw-U8 output; if not,
  hand-roll the header+bytes (the format is trivial — see `write_safetensors` test helpers in
  `src/model.rs`). Don't force the reuse.
- For `teacher_forced_decode_bench` (needs a tokenizer, K9): fetch the real tokenizer files
  once via `python3 tools/fetch_k3_tokenizer_fixture.py` and copy
  `tiktoken.model`+`tokenizer_config.json` into the synthetic dir. `k3_smoke` needs nothing.

**Gate:** `rabbit --model <synth>` loads and decodes; `k3_smoke` and the bench run; per-token
decode time ≈ (6/92) × the real checkpoint's per-token time within reasonable noise — if the
extrapolation is badly off, that's a finding to record, not to hide.

---

## 8. The main work

### Phase 2 — Scalar MXFP4 kernel hygiene (bit-exact)

In `matmul_mxfp4` (K1): restructure the inner loop to iterate 32-element scale blocks —
decode `e8m0_decode(bs[b])` **once per block**; within a block iterate 16 packed bytes
decoding both nibbles (kills the `k & 1` branch). Keep the per-element multiply order
`x * e2m1 * scale` exactly as today: same inputs to the same decode functions in the same
per-element order → **bit-identical output**. Do not introduce per-block partial sums in this
phase (that reassociates — it belongs behind the tolerance-tested AVX-512 tier).

Also add criterion entries to `benches/kernels.rs` at real dims (3072×3584 and 3584×3072,
s=1 and s=8) — they're the before/after instrument for this phase and the next (K8: none
exist today).

**Gate:** K8's two kernel tests green unchanged; a new test pinning bit-exactness vs a
straight-port reference copy of the old loop (kept inside `#[cfg(test)]`); bench delta
recorded. Expected impact: 1.5–4× on the kernel — unknown until measured; record whatever it
is, including "less than hoped".

### Phase 3 — AVX-512 tier for `matmul_mxfp4`

The format is nearly designed for AVX-512: **one 32-element scale block = 16 packed bytes = two
16-lane f32 vectors**, and E2M1 has exactly 16 code points → the whole decode table fits one
`zmm` for `_mm512_permutexvar_ps`.

Per output row: iterate blocks; load 16 bytes; unpack low/high nibbles to two i32 vectors;
`permutexvar_ps` against the E2M1 table (sign handled by the table itself — 16 entries cover
sign×magnitude); FMA against `x`; multiply the block's E8M0 scale (scalar-decoded per block,
broadcast) into the block's contribution; two independent accumulator chains + tree reduction
(the exact pattern of `dot_i4_f32_avx512`, which the module doc documents as the precedent for
"more accurate but not bit-identical"). Handle the `cols % 32` tail scalar.

Runtime selection: same ladder as `matmul_i4` (`is_x86_feature_detected!` at dispatch,
`#[target_feature(enable = ...)]` unsafe inner fn, `pub use` the tier for benches). No AVX2
tier unless it falls out for free (D4).

**Gate:** within-tolerance parity tests vs scalar across random matrices including edge dims
(cols not divisible by 32/64, odd rows), tolerance chosen and documented the way the
`matmul_i4` tiers' tests do; kernel bench delta; **real-checkpoint sanity on the target box**
(same prompt, coherent answer — note in `PERFORMANCE.md` that reassociation may legitimately
flip a near-tie argmax; that is expected, not a bug); end-to-end decode delta via the canonical
command. Expected: this is the big one — kernel should go from ~8 GFLOP/s-class to
memory-bound; decode into low single digits of seconds. Measure, don't assume.

### Phase 4 — Expert loading: parallel misses, preload, honest logging

**4a. Parallelize `sequential_fallback`** (K4): `misses.par_iter().map(load_expert).collect()`
— rayon's ordered collect preserves input order, so cache insertion order (and pin promotion,
K5) is unchanged and results stay deterministic. `Shards` reads are `&self`/`pread`,
thread-safe (K4). This collapses K3's 4–6 s/token warmup cost and — as a side effect —
distributes NUMA first-touch of expert buffers across nodes.

**4b. `--preload-experts` flag.** New CLI flag (`src/main.rs` parse + `USAGE`), threaded
through `chat::LoadArgs` (`src/chat.rs`) into session setup: after `ExpertCaches::new` +
`warm_start`, load every MoE layer's experts up front — chunked per layer, parallel within
chunk (4a's machinery), one progress line per layer. Dispatch through a new method on
`crate::model::ExpertCaches` with per-family arms (all three families get it — it's generic).
Semantics at `capacity < n_experts`: fill to capacity in usage-histogram order when
`.rabbit_usage` exists, else expert-id order, and say so in the log (D2). Applies to `--serve`
too (same `Session` path). Expected: full 1.32 TiB preload in minutes (parallel reads) versus
hundreds of slow-warmup tokens; token 1 runs at steady-state speed.

**4c. Honest logging** (K3): when the expert path has no io_uring ring (MXFP4), stop printing
`[0.0s actual disk wait]` — print nothing or `[disk wait n/a]`. The counter misled the owner
within the first minutes of the first real run; that's a bug in communication if not in code.

**4d. (Flagged, default off — D3) `--drop-os-cache`:** pass `drop_cache: true` through the
expert-load `read_raw` calls when set, using the existing `fadvise_dontneed` machinery, to
avoid double-residency (1.32 TiB of cache slots + the page cache's copy of the same 1.56 TB
checkpoint on a 3 TB box). Tradeoff to document: with it, restarts re-read from disk; without
it, the kernel reclaims page cache under pressure anyway. Off by default.

**Gate:** oracle tests green (loads produce identical bytes; only concurrency changed); warmup
wall-time before/after on the synthetic fixture and, once, on the real checkpoint; preload
timing recorded; no `io_wait` regression lies in the log.

### Phase 5 — Across-expert parallelism in `latent_moe` (K2)

Today (post-Phase-3) each of the 16 experts' three matmuls runs one after another, each matmul
internally fanning out across all 192 cores (~16–19 rows/thread) and joining — ~48
fork/join regions per MoE layer, ~4,400 per token, each hammering one expert's
single-NUMA-node weights with every core.

**v1 (mandatory):** in `latent_moe`'s apply loop — after `ensure_loaded`, collect the chunk's
resident `&ExpertSlot`s, then `par_iter` over experts, each task computing its expert's full
gate→activation→down chain into a **per-expert output buffer** (`s × 3584` f32 — 14 KB at
decode; reuse `apply_single_expert`'s body refactored to write into a caller-supplied buffer
instead of `+=` into shared `routed`), then reduce the buffers into `routed` **sequentially, in
the exact order today's loop applies experts** (chunk iteration order). Per-expert math is
unchanged and reduction order is unchanged → **bit-identical to current output**. Nested rayon
(outer per-expert tasks, inner `par_chunks_mut` in the matmuls) shares one pool — no
oversubscription; work-stealing balances it.

Borrow note (K5): `cache.get(eid)` takes `&mut self` (LRU stamp). Two passes: stamp/collect
indices mutably, then take immutable slot refs via a `peek`-style accessor (add one if
needed) before the parallel region. Keep `record_selection`/usage semantics untouched.

**v2 (conditional — D5):** only if profiling on the target box shows v1 leaves cores idle:
flatten to (expert × row-block) tasks with row-range kernel entry points. Do not build this
speculatively.

**Gate:** `teacher_forcing_k3` (this path IS exercised by the tiny oracle — the fixtures'
plain-float naming still flows through `latent_moe`) plus a new determinism test on any
fixture: two identical warm runs produce bit-identical logits. Decode delta on the synthetic
fixture at 192 cores, then the real checkpoint. GLM-5.2/Kimi-Linear timings must be untouched
(their path wasn't modified — verify with one bench run anyway).

### Phase 6 — NUMA (conditional)

Only if Phase 0c's interleave experiment moved the number materially. The box is 6-node SNC3
(Phase 0a), and placement granularity is naturally per-expert: one expert (16.73 MiB) sits
entirely on one node, so a preload (4b) that round-robins experts across nodes yields an even
spread with no allocator tricks. Prefer, in order: documented `numactl` invocation (zero
code); first-touch distribution already gained from Phases 4a/4b (parallel loads spread
first-touch across the threads that load each expert); per-buffer `libc::madvise`/`mbind` as
a last resort with a measured justification. No new crates (rule 5).

---

## 9. Follow-on (documented for the owner — NOT in scope for this brief)

- **Prefill/batch integer path + AMX.** `apply_single_expert` already batches all tokens
  routed to an expert into one `nr`-row matmul (verified) — prefill's structure is right; it
  lacks a fast batched MXFP4 kernel. The codebase's own threshold (`I4_IDOT_MIN_S = 2`,
  `src/kernels.rs`) says int8-activation IDOT pays at S≥2; VNNI tiers exist for int4/int8. An
  MXFP4→int8 IDOT path, then an AMX tile kernel, is the natural sequel — it's the workload a
  192-core AMX box is shaped for. (AMX does not help batch-1 decode: one activation row leaves
  ~15/16 of the tile idle.)
- **Speculative decoding / MTP** — the only lever that beats the per-token weight-traffic
  floor. First step is a 5-minute check: grep the real checkpoint's
  `model.safetensors.index.json` for MTP-head tensors.
- **io_uring-batched MXFP4 loads** (teach the ring the packed+scale pair) — matters for
  disk-bound hosts, not the 3 TB box.
- **Converter MXFP4-input support** (`convert_shard` currently has none — its `read_f32`
  path doesn't know `.weight_packed`/`.weight_scale` pairs), enabling a one-time int4
  conversion as an alternative decode path, and a `.qs`-style pre-quantized dense sidecar to
  cut the 610 s model load.
- **Carving + determinism for `glm52::moe::moe()`** (the early-drain path) — same idea as
  Phase 5 but entangled with io_uring streaming; separate design needed.

---

## 10. Open decisions (resolve explicitly — with the owner if it changes behavior)

- **D1** Commit `target-cpu=native` (e.g. `.cargo/config.toml`)? Recommendation: no — document
  in `PERFORMANCE.md`; binaries stop being portable across the owner's machines otherwise.
- **D2** Preload semantics when `capacity < n_experts`. Recommendation: fill to capacity,
  usage-order when available, warn loudly.
- **D3** `--drop-os-cache` default. Recommendation: off.
- **D4** AVX2 MXFP4 tier. Recommendation: skip — both known machines have AVX-512; the module
  doc's own precedent is tiers-on-measured-need.
- **D5** Phase 5 v2 trigger. Recommendation: only on target-box profiling evidence of idle
  cores after v1.
- **D6** Synthetic fixture dense dtype (F32 simplicity vs BF16 size) and default layer count
  (recommendation: 6).

---

## 11. Definition of done

Per phase: code + tests + module-doc updates + a dated `PERFORMANCE.md` section (command,
hardware block, before/after, surprises, failures) + one commit. Overall: baseline vs final
canonical-command numbers on the target box at `--expert-cache 896` (warm), `cargo test` green
with oracle fixtures present and actually run (state it), GLM-5.2/Kimi-Linear timings
demonstrably unregressed, and §9 untouched.

## 12. Known traps (read twice)

1. Fixture-dependent tests **skip silently** when fixtures are absent (K9). Green ≠ ran.
2. The tiny K3 oracle does **not** exercise the MXFP4 kernel (K8) — kernel changes live and
   die by the `kernels.rs` unit tests and the real checkpoint.
3. Free-running greedy decode is **not** a valid timing comparison on this codebase — use the
   teacher-forced harness (K9; the harness's module doc explains why).
4. `[0.0s actual disk wait]` on K3 stdout is structurally zero, not evidence of a warm disk
   path (K3) — Phase 4c removes the trap.
5. Laptop-derived defaults (SMT-off thread count, `I4_IDOT_MIN_S`, io_uring-slower-than-pread
   bench notes) encode **12-core measurements**. Re-measure before trusting any of them on 192
   cores; don't silently change them either — measure, then change with the number in hand.
6. Line numbers in this brief were valid at `5853baa`; symbols are the durable anchors.
