# Brief: NUMA-affine execution + AMX follow-on for Kimi K3 on the target box

This is an implementation brief for an AI (or engineer) with **no prior context** on this
project, in the same genre as `K3_OPTIMIZE_BRIEF.md` (read that first — it is this document's
prerequisite, and its §1–§4 context is not repeated here). It was written 2026-08-01 against
commit `5853baa` (v0.23.0), from a design session with the owner on the dev Mac. Companion
documents: `CLAUDE.md` (repo conventions — binding), `PERFORMANCE.md` (measurement history),
`K3_OPTIMIZE_BRIEF.md` (the preceding perf brief; its rules §4 and measurement protocol §5 are
binding here too).

Every `file:line` anchor below was re-verified at `5853baa` while writing this document. If an
anchor doesn't match what you find, stop and re-read the surrounding module doc — do not guess.

**REV 2 (2026-08-01, later the same day):** `K3_OPTIMIZE_BRIEF.md`'s Phases 0–6 have since
been EXECUTED on the instance (commits `a26e5bf`..`62238a1`; results in `PERFORMANCE.md`'s K3
section — read it before this document). All prerequisites are met and **this brief is now the
active plan.** Its predecessor's Phase 6 ran as `numactl --interleave=all` only (measured 1.4×
on the synthetic fixture); the compute-affinity half (N1–N3 here) is unbuilt. One structural
change in rev 2: `K3_OPTIMIZE_BRIEF.md` Phase 5's deferred **v2 is fused into Phase N3** (§7)
— the two are one restructure, do not build them separately. A second: attention work is
promoted to its own **Phase N5** (§9), on the strength of the live serving profile; AMX moves
to N6.

---

## 1. The design principle (owner's directive)

> All tensors involved in one "computation group" live on one NUMA node, and the compute for
> that group runs on that node's cores.

Two clarifications agreed with the owner, binding on the implementation:

1. **This is bandwidth locality, not cache locality.** In decode every weight byte is read
   exactly once per token — there is no temporal reuse for caches to exploit, and adjacency
   between different tensors is irrelevant at these sizes (individual matrices are tens–hundreds
   of MB; L2 is 2 MB/core). What placement buys is *concurrent use of all 6 nodes' DRAM
   channels*: an expert whose 16.73 MiB lives on one node serves the whole 192-core machine at
   that one node's ~1/6 share of aggregate bandwidth, with 5/6 of the traffic additionally
   crossing SNC/UPI boundaries. Affinity (compute where the data is) is what converts placement
   into bandwidth.
2. **The schedulable granularity is 6 nodes** (or 2 sockets), not 96 heads. 96-head layers map
   to 16 heads per node; the 16 selected experts map to ~2.7 experts per node on average.

### What execution measured (instance, 2026-08-01) — binding context for every phase below

1. **Decode is scheduling-overhead-bound before it is bandwidth-bound.** The AVX-512 MXFP4
   kernel came out 3–5× faster at the kernel and ~0% faster end-to-end: an s=1 matmul forks
   ~3072 one-row tasks and hits a ~4.8 ms rayon fork/join floor that both tiers share
   (`PERFORMANCE.md` Phase 3). Removing that floor (v2's row-block tasks) and adding affinity
   (N3) are the same restructure — hence their fusion in §7.
2. **Bandwidth still matters underneath:** `numactl --interleave=all` alone was 1.4× with
   variance collapse (synthetic fixture, `PERFORMANCE.md` Phase 6) — which also proves the
   guest's NUMA topology has real teeth: a decorative topology cannot produce a placement
   effect.
3. **Live serving baseline** (real checkpoint, interleave + `--expert-cache 896
   --preload-experts --threads 48`; `/profile` misses frozen at 82,432 ⇒ the expert bucket is
   pure compute): **1.42 s/token wall** on a 128-token turn — expert matmul 0.83 s/token
   (59%), attention 0.53 s/token (37%), lm_head ~0.01 s. Two conclusions: the expert bucket
   moves 25.8 GB/token at only ~31 GB/s effective against a machine-class ~600 GB/s — **an
   order of magnitude for N3 to claim** — and attention costs 0.53 s/token at near-zero
   context (≤180 tokens; the MLA scan is trivial there), i.e. it is KDA-recurrence/scheduling
   cost, not KV traffic. **Once N3 lands, attention is ~80% of every token** — that is Phase
   N5's justification.

### The box (from `K3_OPTIMIZE_BRIEF.md` Phase 0a — owner-provided `lscpu`/`numactl`, not re-verified here)

2× Intel Xeon 6975P-C (Granite Rapids-AP), 192 physical cores / 384 threads, **6 NUMA nodes**
(SNC3: 3 sub-NUMA domains per socket, 64 logical CPUs + ~507 GB each, 3 TB total). Distances:
10 local, 15–17 intra-socket, 21–28 cross-socket. Full AVX-512 + AMX (`amx_tile`/`amx_int8`/
`amx_bf16`). **KVM guest** — see the go/no-go probe in Phase N0; if the guest's NUMA topology
is not backed by real host pinning, most of this brief is void.

**Provenance caveat (rev 2):** the owner is not certain every `PERFORMANCE.md` number came
from one and the same machine. Do not inherit this topology on trust: run `lscpu` and
`numactl --hardware` on the box you are actually on, record them with your numbers, and derive
every node/core count from `numa::topology()` at runtime — nothing in the code may hardcode 6
nodes or 192 cores.

### Where the traffic is (derived in `K3_OPTIMIZE_BRIEF.md` §2)

~26 GB of weight bytes cross the memory bus per decoded token; **~25.8 GB of that is routed
experts**. Dense weights (attention, shared experts, latent down/up, lm_head) are ~40 GB
resident but only a fraction is read per token; MLA KV caches and KDA states add hundreds of
MB/token at long context. Priority order follows the traffic: experts first (Phase N3), dense
row-sharding second (N4), state/cache placement last (N4c).

---

## 2. Verified code map

All line numbers checked at `5853baa` on 2026-08-01.

- **Thread pool.** One global rayon pool, no affinity anywhere: `configure_thread_pool`,
  `src/main.rs:286-293` (`num_threads(n).build_global()`, default = physical cores).
- **Matmul parallelism.** Every `matmul_*` fans output rows across the whole global pool:
  `yt.par_chunks_mut(s)` sites throughout `src/kernels.rs` (e.g. `:103`, `:653`, `:885`);
  layout rationale in the module doc at `src/kernels.rs:32-38`. Dispatch: `matmul_qt`,
  `src/kernels.rs:921`.
- **QT storage.** `src/quant.rs:52-83`: each weight is ONE contiguous allocation per tensor
  (`QTKind::{F32,I8,I4,I4Grouped,MxFp4}`, `Vec`-backed). MXFP4: `data.len() =
  rows*ceil(cols/2)`, `block_scale.len() = rows*ceil(cols/32)`, allocated in `alloc_mxfp4`
  (`src/quant.rs:106-113`). Row-major, so a contiguous **row range** of any QT is a contiguous
  byte range in each of its buffers — this is what makes per-node sharding clean (N4).
- **Experts.** `ExpertSlot { eid, gate_up, down, used }`, `src/expert_cache.rs:473-478`;
  `ExpertCache`, `src/expert_cache.rs:526-578` (per-layer LRU + pinned tier);
  `ensure_loaded(&mut self, ...)`, `src/expert_cache.rs:671`. **`get` is `&self` and
  read-only** (`src/expert_cache.rs:779`) — note this contradicts `K3_OPTIMIZE_BRIEF.md`'s K5
  ("`get` stamps LRU recency, takes `&mut self`"); the code is what I verified, and it means
  collecting `&ExpertSlot` refs for a parallel region needs no borrow gymnastics. MXFP4 loads:
  `qt_load_mxfp4`, `src/expert_cache.rs:884` — allocation and first write (= NUMA first touch)
  happen on the **calling thread**, inside `Shards::read_raw`. Whichever thread runs an
  expert's load decides where its pages land. This is the placement mechanism for N3.
- **K3 expert dispatch.** `latent_moe`, `src/kimi_k3/moe.rs:64-116`: route → down-proj →
  chunked `ensure_loaded` → sequential `for &eid in chunk { apply_single_expert(...) }` →
  optional RMSNorm → up-proj. (Post-`K3_OPTIMIZE_BRIEF.md`-Phase-5 this loop is a `par_iter`
  with per-expert output buffers and a fixed-order sequential reduction — N3 builds on that
  shape.) Expert math: `apply_single_expert`, `src/glm52/moe.rs:460`.
- **KDA.** Already 96-way head-parallel on the global pool:
  `state.heads.par_iter_mut()...`, `src/kimi_k3/generate.rs:258` (Kimi-Linear twin at
  `src/kimi_linear/generate.rs:281`). Per-head state = `KdaState` 128×128 f32 = 64 KB
  (`src/kimi_linear/kda.rs:92-96`). Heads are fully independent — the cleanest fit for
  16-heads-per-node partitioning.
- **MLA.** Absorbed decode is already 96-way head-parallel:
  `ctx_row.par_chunks_mut(vh)...`, `src/glm52/attention.rs:481`. But the compressed KV cache
  is **shared across all heads by construction** (`KvCache`, `src/glm52/attention.rs:86-91`:
  one `l`/`r` stream per layer, `kv_lora=512`; every head's score loop reads the same rows) —
  it cannot be partitioned by head. Weights partition per head; the KV stream must be
  interleaved or replicated instead (N4c).
- **Scope guard.** `glm52::moe::moe()` (GLM/Kimi-Linear's early-drain io_uring path) stays
  untouched, same as in the previous brief (its rule §4.7). All NUMA dispatch work happens in
  `latent_moe` and below `matmul_qt`.

### Per-block mapping of the principle (agreed with the owner)

| block | group unit | fit | scheme |
|---|---|---|---|
| routed experts (16 of 896/layer) | one expert (16.73 MiB, self-contained) | **clean — highest value** | home node per expert; compute on home node's pool (N3) |
| KDA (69 layers × 96 heads) | head (state + weight row-blocks) | clean | 16 heads/node static; weights row-sharded, states node-allocated (N4b) |
| MLA (24 layers × 96 heads) | head — **weights only** | partial by design | weight rows per head (16/node); shared KV stream interleaved (N4c) |
| dense matmuls (shared experts, latent down/up, dense MLP L0, lm_head) | row-block of each tensor | clean | shard each QT's rows 6 ways, node i computes rows on node i (N4a) |

Expected per-token routing skew (N3): which 16 experts a token picks is random, so per-node
load is 2.7 ± a few; some tokens will put 5–6 experts on one node. Do **not** rebalance by
stealing across nodes (that reintroduces the remote traffic this design removes) — it averages
out over 92 MoE layers per token. Log the skew; don't fight it.

---

## 3. Invariants (binding, additive to `K3_OPTIMIZE_BRIEF.md` §4)

1. **NUMA placement must never change math.** Placement and affinity are pure scheduling: same
   per-expert compute, same reduction order → decode output stays **bit-identical** with
   `--numa` on vs off. This is the acceptance gate for every phase except N6 (AMX, which is a
   tolerance-tier kernel like every reassociating SIMD tier before it).
2. **Everything is `#[cfg(target_os = "linux")]` + a `--numa` CLI flag, default off.** The dev
   laptop (12-core AMD, single node) and the owner's Mac must build and behave exactly as
   today. Non-Linux gets a no-op stub, not a compile error.
3. **No new runtime dependencies.** `libc` is already in `Cargo.toml`. `sched_setaffinity` is
   declared in the libc crate directly; `mbind`/`set_mempolicy` have **no glibc wrapper** (they
   live in libnuma, which is banned) — call them as raw syscalls:
   `libc::syscall(libc::SYS_mbind, ...)`. No `hwloc`, no `numa`/`numactl` crates. The one open
   exception is the AMX C shim's `cc` build-dependency (D4, owner sign-off required).
4. **Record the `numactl` invocation, `--numa` state, `--threads`, THP state, and `RUSTFLAGS`
   with every number** in `PERFORMANCE.md`. A NUMA measurement without its placement recorded
   is unreproducible noise.
5. Fixture-dependent tests skip silently when fixtures are absent — check for `SKIP` lines
   before claiming green (previous brief's trap #1; it still applies).

---

## 4. Phase N0 — Go/no-go probes (zero code, run on the instance first)

**N0a. Is the guest's NUMA topology real?** KVM guests only benefit from guest-side affinity
if the host pins vCPUs and backs each virtual node with matching host memory. Probe with any
STREAM-triad binary (record which):

```
numactl --cpunodebind=0 --membind=0 <stream>     # local
numactl --cpunodebind=0 --membind=2 <stream>     # intra-socket remote
numactl --cpunodebind=0 --membind=3 <stream>     # cross-socket remote
```

**Gate:** local vs cross-socket bandwidth should split ≥2×. If the three numbers are flat, the
guest topology is decorative — **stop here**, report to the owner, and fall back to
`K3_OPTIMIZE_BRIEF.md`'s Phase 6 minimum (documented `numactl --interleave=all` as the launch
default). Everything from N1 on assumes this gate passed. (Partially de-risked already: the
measured 1.4× interleave win implies real placement effects. Run the probe anyway — its
local:remote ratio sizes how much N3 can win and feeds decision D1.)

**N0b. Baseline sensitivity.** With the AVX-512 kernel + Phase 4/5 landed, canonical command
(`teacher_forced_decode_bench --expert-cache 896`, warm) under: default policy,
`--interleave=all`, and `--cpunodebind=0-2 --membind=0-2 --threads 96` (single socket). If
interleave barely moves the number, decode is not yet bandwidth-bound — investigate what still
dominates before building N2–N4. Record everything in `PERFORMANCE.md`.

**Recorded prior (owner, real checkpoint, scalar-kernel baseline, 2026-08-01):** `--threads
48` decodes much faster than the 192-physical-core default; all cores is *slower*. This is
consistent with the diagnosis in §1 — first-touch concentration means high thread counts add
contention on 1–2 nodes' channels + UPI, not bandwidth, and 192-way fork/joins thin the
per-thread grain to ~16 rows. Consequences: (a) re-run the thread sweep at every phase gate —
the optimum will move; (b) baseline numbers should be taken at both 48 and the default, both
recorded; (c) the NUMA phases' headline success criterion can be stated as: **after N3/N4,
high pinned thread counts finally beat 48 unpinned** — if the sweep still says 48 after N3/N4,
placement isn't working; stop and investigate rather than shipping the flag.

**Recorded baseline (rev 2 — live serve, post-Phases-2–6, 2026-08-01):** canonical serve launch:

```
numactl --interleave=all ./target/release/rabbit --model /data/hf/hub/kimi-k3 --serve \
    --port 8000 --expert-cache 896 --no-usage-cache --preload-experts --threads 48
```

Startup shape: ~15 min dense load (RSS → ~30 GiB, ~10 s/layer) then ~30 min expert preload at
~0.8 GB/s (RSS → ~1.44 TiB), then the port opens and `/profile`'s `misses` freezes at 82,432
(= 92 × 896, the preload itself). Baseline on a 128-token turn: **1.42 s/token** (expert 0.83 /
attention 0.53 / lm_head 0.01). N3's decode gate is measured against THIS number and this
launch configuration.

**N0c. Record** `uname -r`, THP state, and re-paste `numactl --hardware` from the live box.

---

## 5. Phase N1 — `src/numa.rs`: topology + placement primitives

New module, `libc`-only, fully `cfg`-gated, with a no-op twin for non-Linux. Public surface
(keep it this small):

- `topology() -> Option<Topology>` — parse `/sys/devices/system/node/node*/cpulist`. `None`
  (or 1 node) ⇒ everything downstream degrades to current behavior.
- `pin_current_thread(node)` — `sched_setaffinity` to the node's CPU list.
- `bind_region(ptr, len, node)` / `interleave_region(ptr, len)` — raw `SYS_mbind` on an
  address range. Note `mbind` sets a **VMA policy**: pages faulted *later* in the range follow
  it, which is exactly what makes it work for reserve-then-grow buffers (N4c) — but it does
  not migrate pages already touched (use first-touch discipline instead of `MPOL_MF_MOVE`;
  moving pages is a rescue, not a design).
- First-touch is the preferred placement mechanism everywhere a load/init already runs on a
  pinned thread (N3 preload); `bind_region` is for buffers whose faulting thread can't be
  controlled.

Unit tests: topology parser against fixture strings; a `cfg(target_os = "linux")` smoke test
that pins, allocates, touches, and reads back `move_pages`/`get_mempolicy` for the placement
(skip-not-fail when run on a 1-node machine, matching the repo's fixture-test convention).

**Gate:** `cargo test` green on Linux and macOS; `--numa` flag parsed (`src/main.rs` + USAGE)
but changing nothing yet.

---

## 6. Phase N2 — Pinned per-node pools

A `NodePools` singleton (built once in `configure_thread_pool`'s successor when `--numa` is
active): one `rayon::ThreadPool` per node, each worker pinned via
`ThreadPoolBuilder::start_handler(|_| numa::pin_current_thread(node))`. **Pool size derives
from the effective `--threads` total, not from the core count**: `total_threads / n_nodes`
(e.g. `--threads 48` → 8/node, default 192 → 32/node). The owner's observed 48-beats-192
result (N0b) means the thread sweep must stay meaningful with `--numa` on — hardcoding
32/node would silently override it. The global pool remains for non-NUMA paths and other
architectures — GLM/Kimi-Linear behavior must not change.

Two hazards to design around (they are the real content of this phase):

1. **Never call `pool_b.install(...)` from inside a `pool_a` worker** — it blocks a worker
   thread waiting on another pool. Cross-pool fan-out uses `spawn` + a latch/channel join from
   the *orchestrating* (non-pool) thread, or `rayon::scope` per pool from outside.
2. **Decide 6 pools vs 2 (per-socket) empirically.** SNC3 nodes share a socket's cache/UPI;
   per-socket pools (2×96) capture the expensive cross-socket hops with a third of the
   orchestration overhead. Build the pool count from `topology()`, benchmark both (D1).

**Gate:** a criterion bench of a sharded matvec (N4a's kernel) at both pool configs; no
user-visible behavior change with `--numa` off.

---

## 7. Phase N3 — Expert home nodes, FUSED with Phase 5 v2 (the main event)

Prerequisites (all landed): `K3_OPTIMIZE_BRIEF.md` Phase 4b (`--preload-experts`) and Phase 5
v1 (per-expert parallel apply with fixed-order reduction).

**Why fused:** v2 (flatten to expert × row-block tasks, one fork per MoE layer, each task
running the serial AVX-512 kernel over a row range) and home-node dispatch want the exact same
task shape. Building v2 on the global pool first and re-plumbing it onto node pools afterward
is the same surgery twice — do it once, here. This phase owns both the fork/join floor (~48
all-core matmul forks per MoE layer today) and locality.

**N3a. Assignment.** `home_node(layer, eid) = (layer * 896 + eid) % n_nodes` — static,
deterministic, no state to persist. ~2.2 GB/layer/node, ~220 GB/node total at full residency:
comfortably under 507 GB/node alongside interleaved dense weights (N4) and the OS page cache.

**N3b. Placement.** In the preload path, run each expert's `load_expert`/`qt_load_mxfp4` **on
its home node's pool** — allocation + `pread` fill happen on a pinned thread, so first touch
lands the pages correctly with zero `mbind` calls (see the code-map note on
`src/expert_cache.rs:884`). Misses loaded later during generation (capacity < 896, or cold
start without preload) go through the same routing. Store nothing new in `ExpertSlot` —
`home_node` is a pure function.

**N3c. Dispatch (the v2 task shape).** Add row-range entry points to the MXFP4 kernel:
compute rows `[r0, r1)` of one matmul with the serial AVX-512 tier inside — **no inner rayon**.
In `latent_moe`'s apply stage: group the chunk's selected experts by home node → submit each
expert's row-block tasks to its home node's pool (per-expert output buffer, `s × 3584`, ~14 KB
at decode, transient) → **one fork/join per MoE layer across all pools** instead of ~48
all-core forks → reduce the per-expert buffers into `routed` **sequentially in chunk iteration
order**. Keep each expert's gate→activation→down chain order and per-row math identical to v1
so the output stays bit-identical to v1 (which is itself bit-identical to the original
sequential loop). Grain check: ~48 matmuls × ~3–3.5k rows split across a node's threads =
hundreds of rows of real serial AVX-512 work per task — nothing like today's one-row
micro-tasks.

**N3d. Observability.** Extend `StepProfile`/`GET /profile`'s phase timings with per-node
busy time for the MoE phase, plus a per-token expert→node skew counter. This is how D1 (6 vs 2
pools) and the "don't rebalance" decision get validated with data instead of vibes.

**Gate:** `teacher_forcing_k3` green (fixtures present and RUN — say so); the Phase-5
determinism test (two warm runs → bit-identical logits) still green with `--numa` on; decode
delta on the canonical command, `--numa` on vs off, recorded. This phase is the one expected
to move the number materially — if it doesn't, N0b's sensitivity data was misread; stop and
re-profile before building N4. **Numeric success criteria against the rev-2 serve baseline:
expert bucket well under 0.3 s/token (from 0.83), and the thread sweep finally favoring high
pinned counts over 48 unpinned.**

---

## 8. Phase N4 — Dense weights, KDA/MLA weights, caches

**N4a. Row-sharded QT.** New `QTSharded` (or a `Vec<QT>` + row-offset table — implementer's
call): a weight's output rows split into `n_nodes` contiguous blocks, block *i* allocated on
node *i* (first-touch at load time by a pinned thread, same trick as N3b). A
`matmul_qt_sharded` issues one row-range task per node pool and joins. Per-row math and
per-row output slots are unchanged → bit-identical. Apply in descending traffic order:
lm_head (163,840×7,168 — the poster child), latent `down_proj`/`up_proj` (92 layers), shared
experts, dense MLP layer 0. **Do not convert every last tensor** — each conversion adds a
6-pool barrier per matmul (~4,400 fork/join regions/token already exist; measure the barrier
cost at decode dims in a criterion bench before and after).

**N4b. KDA/MLA per-head weights.** Head-aligned row blocks: 96 heads / 6 nodes = 16 heads ⇒
e.g. KDA q/k/v projections `[12288, 7168]` shard into 6 × 2048-row blocks on head boundaries;
MLA `q_b`/`kv_b` same treatment. (Rev 2: the recurrence loop's task grain and the `KdaState`
allocations moved to Phase N5 — the serve profile showed attention at 37% of the token, so
that work was promoted to its own phase. N4b is weights only.)

**N4c. The shared streams that can't be partitioned.** MLA's KV cache is one stream read by
all heads (code map): `reserve` it to a max-context capacity at session start and
`interleave_region` the reservation (VMA policy covers future faults — this is why N1's
`mbind` wrapper exists), so all nodes draw balanced bandwidth during the score scan. The
hidden-state activation vector (7168 f32 = 28 KB/token) is read by everyone but fits in L2 —
leave it alone. Replicating KV per node (6 × ~7 GB at 128k ctx across 24 MLA layers) is
affordable on this box but fans every append out 6× — only revisit if profiling shows the
interleaved scan bound on remote latency (D3).

**Gate:** bit-identity vs `--numa` off (N4a/b change nothing about order); decode + prefill
delta recorded; GLM-5.2/Kimi-Linear timings demonstrably unregressed (they share `matmul_qt` —
run one bench).

---

## 9. Phase N5 — Attention: KDA scheduling + state placement (new in rev 2)

Justification (measured, §1): attention costs **0.53 s/token at near-zero context** — that is
per-head scheduling overhead plus the scalar KDA recurrence, not KV traffic, and it becomes
~80% of every token once N3 lands. Scope guard unchanged: the *math* of MLA/KDA is
untouchable; task grain, scheduling, and memory placement around it are exactly this phase.

- **N5a. Measure first.** Split the profile's `attention_s` into KDA recurrence vs projections
  vs MLA, per layer kind (extend `StepProfile`). 69 KDA layers × (six projections + a
  96-head recurrence + short-conv) has several candidate costs — don't guess the split.
- **N5b. Task grain.** The per-head `par_iter` (96 tasks, each a ~64 KB state update) is
  plausibly fork/join-floor-bound exactly like the expert matmuls were. Batch heads: one task
  per node covering its 16 heads serially, states allocated on that node (absorbed from old
  N4b), one fork per KDA layer. Same bit-identity argument as the existing head loop —
  disjoint outputs, no cross-head reduction, only execution order changes.
- **N5c. Projections** are ordinary matmuls — they ride N4a's row-sharding (at head-aligned
  boundaries) if N4a's barrier-cost measurements came out favorable (D5).
- Floor check: KDA state traffic is 69 × 96 × 64 KB ≈ 425 MB/token read+write ⇒ ~1–2 ms/token
  at machine bandwidth. Attention's ceiling is nowhere near the current 0.53 s — there is real
  room here.

**Gate:** teacher-forced bit-identity (scheduling only); attention-bucket delta on the rev-2
serve baseline; GLM-5.2/Kimi-Linear unregressed with their oracle tests **actually running**
(KDA code is shared with Kimi Linear — this gate is not optional).

---

## 10. Phase N6 — AMX (follow-on; design-level here, own brief when reached)

AMX multiplies 16×64 tiles — it needs ≥16 activation rows to fill a tile. **Batch-1 decode
gets nothing from it** (one row ⇒ 15/16 idle, and post-N3/N4 decode is bandwidth-bound
anyway). The AMX targets are prefill and batched expert application — `apply_single_expert`
already batches all tokens routed to an expert into one multi-row matmul, so the structure is
ready today.

- **N6a. Transcode at preload, not on disk.** On this box experts are RAM-resident, so the
  cache-slot format need not match the disk format. MXFP4→int8-per-block is **exact**: E2M1's
  16 code points are {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6} — multiply by 2 and they're all
  integers in [−12, 12], so store `value×2` as int8 with the block scale halved (E8M0 is a
  pure power of two; fold the ×½ into the per-32-block f32 scale). Same 1.32 TiB footprint.
  This unlocks the *existing* VNNI IDOT tiers (`dot_i8i8_avx512vnni`, `src/kernels.rs:294`)
  for s≥2 before any AMX code exists — likely the best effort/payoff step of the whole phase.
- **N6b. AMX kernel.** Stable Rust has no AMX intrinsics (nightly-only as of this writing);
  the clean options are a small C file (`immintrin.h`, `-mamx-int8`) via `build.rs` + `cc`
  (needs D4 sign-off — first new build-dep) or inline asm. Runtime requirements: XFD
  permission via `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)` once per process before
  first tile op, `ldtilecfg` per kernel, `is_x86_feature_detected!("amx-int8")` in the dispatch
  ladder. Tolerance-tier parity tests (it reassociates), gated by an `s >= threshold` check
  measured the way `I4_IDOT_MIN_S` was.
- Scope note: MTP/speculative decoding (which would make AMX matter for *decode* by batching
  candidate tokens) remains out of scope, per `rabbit-plan.md`.

---

## 11. Measurement protocol

Identical to `K3_OPTIMIZE_BRIEF.md` §5 (teacher-forced bench only, worktree before/after,
warm/cold labeled, canonical command at `--expert-cache 896`) plus: every number carries its
`--numa` state and `numactl` prefix; every phase records per-node busy/skew from N3d once it
exists. Free-running greedy decode is still not a timing instrument.

## 12. Open decisions

- **D1** 6 pools (per-node) vs 2 (per-socket). Recommendation: build from `topology()` so
  both are one flag apart; decide on N2's bench + N3d's skew data.
- **D2** `--numa` default once proven. Recommendation: off until it has survived one full
  real-checkpoint session (`--chat`) without regression, then flip with the owner.
- **D3** KV cache interleave vs per-node replication. Recommendation: interleave; revisit
  only on profiling evidence.
- **D4** AMX shim route: C-via-`cc` vs inline asm vs nightly toolchain. Owner sign-off
  required either way (`cc` is the repo's first build-dep; inline asm avoids it at the cost of
  hand-written tile config). No recommendation until N6 is actually scheduled.
- **D5** Whether N4a extends to MLA/KDA projections at all if N4a's first conversions show
  the barrier overhead eating the locality win at decode dims (s=1).

## 13. Known traps

1. All of `K3_OPTIMIZE_BRIEF.md` §12 still applies (silent fixture skips, teacher-forced-only
   timing, stale laptop defaults).
2. **`pool.install` from another pool's worker deadlocks under load** — cross-pool fan-out
   only from an orchestrating thread (N2).
3. **`Vec` growth breaks placement**: `extend_from_slice` reallocs land wherever the growing
   thread runs. Any buffer with a placement policy must be exact-sized (expert QTs already
   are) or reserved-then-`mbind`ed (KV cache, N4c).
4. **THP vs sharding**: separate per-node allocations are THP-safe; do NOT try to `mbind`
   row-ranges *inside* one contiguous QT allocation — 2 MB huge pages straddle the boundaries
   and the policy silently loses. This is why N4a shards the allocation itself.
5. `ExpertCache::get` is `&self` (`src/expert_cache.rs:779`) — if you read
   `K3_OPTIMIZE_BRIEF.md`'s K5 first, its "`&mut self` borrow puzzle" note does not match the
   code at `5853baa`. Trust the code.
6. The KVM guest can lie (N0a). Also re-check after any instance resize/migration — host
   repinning silently invalidates every measurement after it.
7. `home_node` skew is per-token noise — do not add cross-node work stealing to "fix" it
   (§2); over 92 MoE layers it averages out, and stealing reintroduces remote traffic.

## 14. Definition of done

Per phase: code + tests + module-doc updates + dated `PERFORMANCE.md` section (command,
placement recorded per §11, before/after, failures included) + one commit. Overall: canonical
command on the target box, warm, `--numa` on vs off, bit-identical logits (N3/N4/N5) and the
speedup number; `cargo test` green on Linux **and** macOS with fixture status stated;
GLM-5.2/Kimi-Linear unregressed; N6 untouched unless separately scheduled with the owner.
