# Kimi K3 performance history

See `PERFORMANCE.md` for GLM-5.2's own performance history — a separate page, since the two
architectures shipped at different times against different checkpoints on different hardware
generations of this project. This page tracks Kimi K3 specifically: a 2.8-trillion-parameter
open-weight model whose routed experts (the vast majority of its size) ship natively OCP
MXFP4-quantized on disk, run here the same way GLM-5.2 is — most of the model kept on disk,
streamed into memory as each generated word needs it.

## In plain terms

K3 only just started working end-to-end (2026-07-28) — real weights, real tokenizer, real chat
template, coherent real generation — so unlike GLM-5.2's page, there isn't yet a long multi-version
history to show. What's here is the **first performance pass**, done the same day K3's correctness
work finished: the matrix-multiply routine used for every routed expert's math was still doing the
math value-by-value in a loop (the safe, simple way to get something correct first), even though
this exact kind of speed-up — using the CPU's wider math instructions to do many values at once —
had already paid off for GLM-5.2 (see that page's v0.17.0/v0.22.0 entries). Porting the same idea
to K3's own 4-bit number format made that one calculation **up to 25× faster in isolation**.

**End-to-end, on the real 2.8-trillion-parameter checkpoint, generating the same real answer to the
same real question: reading the prompt got ~1.9× faster, and the ongoing word-by-word generation
got roughly 1.3-1.9× faster** (not 25× — see "why the end-to-end win is smaller" below for exactly
why, the same story GLM-5.2's own page tells about its v0.22.0 entry). Both numbers are from a real
run against the actual downloaded checkpoint, not estimated.

The same day, once disk I/O (not compute) became the dominant cost, batched `io_uring` reads for
MXFP4 experts (mirroring GLM-5.2's own long-standing `io_uring` path, which MXFP4 had never used)
added a further, smaller real win on top — see "Speed by version" below. Finding that win took a
detour through a real, measured false alarm first (a background process eating the entire gain via
thousands of a doomed syscall's log lines, nothing to do with `io_uring` itself) — see "A red
herring" further down; worth reading if the numbers below ever look inconsistent between runs on
this same machine, since the ROOT CAUSE (accumulated `.rabbit_usage` history) is still there and
will recur on any long-lived checkpoint dir.

## Before there was a way to measure this properly

| Version | What existed | How solid is this |
|---|---|---|
| v0.20.0 - v0.22.0 | Kimi Linear 48B (K3's smaller sibling) working; K3 itself didn't exist as public weights yet | N/A — different, much smaller checkpoint, not comparable to the numbers below |
| v0.23.0, first pass | K3 loads and runs against the real checkpoint with made-up token ids, not a real question | **Real measurement, but not a real prompt**: `examples/k3_smoke.rs`, decode ~28-33s/token. Useful for confirming nothing crashes and memory stays bounded, not for judging real-world speed — no tokenizer or chat template involved yet at that point. |
| v0.23.0, real prompt | K3 answers a real question (`--prompt "What is the capital of France?" --max-tokens 40`) correctly, through the real tokenizer and chat template | **Real measurement**: model load 610.0s, prefill (7 tokens) 412.8s, decode steady-state ~50-70s/token (first generated token 122.3s — slower than steady-state, a one-time warmup cost). This is the "before" baseline the table below compares against. |

## Speed by version

Same test both times: `--model <checkpoint-dir> --prompt "What is the capital of France?"
--max-tokens 40`, same real 2.8T checkpoint, same machine, `--expert-cache 64` (the default).

| Version | Prefill (7 tokens) | Decode (steady-state) | Speed (tok/sec) | Change | What changed |
|---|---|---|---|---|---|
| v0.23.0 | 412.8s | ~50-70s/token | 0.014 | — (first working version) | K3 shipped: engine, native-format checkpoint loading, tokenizer, chat template, session persistence — the expert math itself still used the simple, unvectorized loop |
| v0.24.0, SIMD kernel only | 216.5s | ~37.5s/token avg (range 17.1-44.8s across 40 tokens) | ~0.027 | **Prefill ~1.9× faster; decode ~1.3-1.9× faster** vs v0.23.0 | Taught the expert math to use the CPU's wider instructions, the same idea GLM-5.2's own v0.17.0/v0.22.0 already used for its own number formats |
| v0.25.0, + `io_uring` for MXFP4 | **210.3s** | **35.0s/token avg** (1401.4s/40 tokens) | **~0.029** | **~3% faster prefill, ~7% faster decode** vs v0.24.0 — measured with `--no-usage-cache` to rule out the red-herring confound below | Batched `io_uring` reads for MXFP4's `{name}.weight_packed`+`{name}.weight_scale` pair, same mechanism GLM-5.2 already had for its own tensor formats — previously every MXFP4 load fell back to one-at-a-time synchronous reads |

Run-to-run noise on the very first generated token specifically was large enough (44.8s / 85.3s /
122.3s across three different runs on this same machine) that it isn't used as a headline number
here, unlike an earlier draft of this page — the steady-state averages above are the more reliable
comparison.

**Isolated kernel benchmark** (no disk I/O involved — just the math itself, a stand-in matrix size
of 4096×4096, same style as GLM-5.2's own kernel-only benchmark table):

| Kernel | Before (plain loop) | After (CPU's wide instructions) | Change |
|---|---|---|---|
| Routed-expert matmul, older/narrower instructions | 8.29 ms | 736 µs | **~11.3× faster** |
| Routed-expert matmul, newest/widest instructions | 8.29 ms | 331 µs | **~25× faster** |

## Why the end-to-end win is much smaller than the kernel win

This is the same story GLM-5.2's own page tells about its v0.22.0 entry, playing out even more
starkly here because the underlying kernel win is so much bigger. Before this change, most of each
generated word's time was the expert math itself (measured the day before at ~60-70% of each
word's time, vs ~30-40% waiting on the disk). Making that math ~25× faster didn't just speed
things up — **it flipped which part of the work is now the bottleneck**. A real excerpt from
today's run, after the fix:

```
token 6/40 in 37.1s (32.7s in disk I/O this step)
token 16/40 in 27.0s (22.0s in disk I/O this step)
token 33/40 in 41.3s (36.8s in disk I/O this step)
```

Disk I/O now accounts for **80-90% of each word's time**, not 30-40% — because the compute side
shrank so much it stopped being the limit. This is a genuinely good sign, not a disappointing one:
it means the next real speed-up has to come from the disk side, not from further speeding up math
that's no longer the thing anyone is waiting on. `io_uring` batching for MXFP4 (below) picked up
that exact thread the same day.

## A red herring: mlock spam looked like an io_uring regression

Worth recording in detail — this is exactly the kind of result that would otherwise get
mis-remembered as "`io_uring` for MXFP4 didn't help" the next time someone looks at this page.

The first `io_uring` measurement, run right after the SIMD-kernel one above (same real checkpoint,
same command, `--expert-cache 64`, the default — i.e. WITH the persisted `.rabbit_usage` history
from every prior real run on this checkpoint dir still active), came out SLOWER, not faster:
prefill 216.5s → 300.0s, decode 37.5s/token → 49.5s/token. That's a real regression against the
SIMD-only version, on the same real checkpoint.

The log explained why before any code needed re-reading: **5918 lines of `mlock failed ...
Cannot allocate memory (os error 12)`**, versus **zero** in the SIMD-only run right before it.
Nothing about `io_uring` causes this directly — it's `usage_cache`'s pin-candidate promotion
(`ExpertCache::insert_or_pin` → `mlock_best_effort`), which tries to `mlock` a expert's weight
buffers into RAM the first time a "worth keeping" expert (per persisted selection history) is
actually loaded. Each of this session's real runs on this SAME checkpoint dir had already been
appending to `.rabbit_usage`, so by the third real run, far more experts qualified as pin
candidates than the very first run ever did — and this unprivileged process's `RLIMIT_MEMLOCK` is
far too low to lock even one MXFP4 expert's multi-megabyte buffers, so EVERY one of those newly
promoted experts failed, each failure printing its own line (`eprintln!`, unconditionally, on
every single attempt — a real bug: the function's own doc comment already claimed "logged once,"
which the code never actually did).

Re-ran the exact same `io_uring` build with `--no-usage-cache` (no pin candidates possible at all)
to isolate the variable: **zero `mlock` failures, and the numbers above (210.3s / 35.0s/token)
came out — a real, if modest, win over the SIMD-only version, exactly as expected.** The regression
was never about `io_uring`'s own correctness or design; it was thousands of a doomed syscall's
worth of overhead (plus its logging) crowding out an unrelated real improvement in the same
measurement window.

**Fixed the same session**: `mlock_best_effort` now tracks a process-wide "already known to fail"
flag (`MLOCK_KNOWN_TO_FAIL`, checked before every attempt) — the first failure logs once and
disables every future attempt for the rest of the process, matching what the doc comment always
claimed. `RLIMIT_MEMLOCK` doesn't change over a process's lifetime, so a single failure reliably
predicts every later one; losing the (already best-effort, already not guaranteed) pin-to-RAM
benefit for the rest of a long session is a fine trade for not burning real wall-clock time on a
syscall that will never succeed. This means the confound above can't recur — but the underlying
`.rabbit_usage` growth pattern that triggered it is real and will keep happening on any
long-lived checkpoint dir; worth remembering if a future measurement on THIS SAME checkpoint looks
inexplicably slower than expected — check `.rabbit_usage`'s size / try `--no-usage-cache` before
assuming a code change caused a regression.

## A real crash, not a red herring: the mlock fix's own confirmation run got OOM-killed

Re-ran the `mlock` fix's confirmation test with the DEFAULT flags (usage cache ON — the realistic
case, not `--no-usage-cache`) to prove the fix holds under the exact conditions that broke before.
The fix itself worked exactly as designed: 2944 pin candidates marked this run (usage history kept
growing across the session), only **1** `mlock` failure line printed, not thousands. But the
process never finished — `journalctl -k` showed the real cause:

```
Out of memory: Killed process 506610 (rabbit) total-vm:182627624kB, anon-rss:122899668kB, ...
```

**~123GB resident** on a 123GB machine, killed by the kernel. Per-token times had been climbing
sharply right before the kill (39s → 273s across the last several tokens) — real memory pressure
building, not random noise. This has NOTHING to do with `mlock` failing or `io_uring` — it's a
structural sizing problem in `--expert-cache`'s flat `64`-per-layer default, exposed at K3's real
scale: each MXFP4 expert here is **~35MB** (`moe_intermediate_size=3072` × `hidden_size=7168` is
much bigger than whatever this default was last validated against), and K3 has **93** MoE layers.
Worst case, BEFORE even counting the separate never-evicted `pinned` tier on top:

```
93 layers × 64 slots/layer × 35MB/expert ≈ 209 GB   (ordinary LRU tier alone)
93 layers × 32 pins/layer  × 35MB/expert ≈ 104 GB   (pinned tier, once usage-cache confidence maxes out)
                                    combined ≈ 313 GB
```

...against 123GB of real RAM. The ordinary (evicting) LRU tier alone can already exceed this
machine's RAM if a long enough generation touches enough distinct experts per layer — the pinned
tier (permanent, never evicted, and this session's repeated real runs had already pushed
`.rabbit_usage`'s confidence to its 1.0 ceiling) just gets there faster.

**Fixed the same session**: `--expert-cache`'s CLI flag is now `Option<usize>`
(`LoadArgs::cache_capacity`) — an EXPLICIT `--expert-cache N` still means exactly `N`, unchanged,
never silently overridden (the user's own informed choice always wins). Only the AUTO default (no
flag passed) is now size-aware: `model::safe_default_expert_cache_capacity` computes each MXFP4
expert's real byte size from the checkpoint's own `moe_inter`/`hidden` (shape-only — an empty `QT`
asking its own `resident_bytes()`, no disk I/O) and clamps the flat `64` default so that
`n_moe_layers × capacity × 1.5 × per_expert_bytes` (the `1.5` accounting for the pinned tier's own
worst case) stays under a fixed, conservative 24GB budget — at K3's real scale that computes to
**auto-capacity 4** (worst case ≈19.6GB) instead of the old flat 64 (worst case ≈313GB). Verified
with fast unit tests against K3's exact real dimensions (`expert_cache::tests::safe_mxfp4_capacity_
clamps_the_flat_default_at_real_k3_scale`) — no real checkpoint needed to confirm the math, only to
confirm it holds under real disk/generation conditions too.

**Re-run against the real checkpoint right after, default flags (the exact scenario that OOM'd):
completed cleanly, no kernel kill.** The auto-clamp note printed exactly as designed:

```
expert cache: auto --expert-cache 64 would risk ~310GB peak memory on this checkpoint
(92 MoE layers x ~35.1MB/expert, LRU + pinned tiers combined) -- lowered the auto default to 4
```

Pin candidates dropped from the earlier crash's 2944 to **184** (`floor(4 × 0.5 × 1.0) × 92 ≈ 184`
— scales down with the smaller base capacity exactly as `usage_cache::pin_budget`'s formula
predicts). Full run: load 609.6s, prefill 44.0s, 40 tokens in 285.3s (**~7.1s/token, ~0.140
tok/sec**) — much faster than any earlier run on this page, almost certainly because this
session's many repeated real runs on the same checkpoint had already warmed the OS page cache for
its more commonly-touched shards; **not a clean apples-to-apples comparison with the earlier
cold-cache numbers above**, just confirmation that the fix holds under real conditions without a
crash.

## Reproducing these numbers

**End-to-end table**: `cargo build --release --bin rabbit`, then `./target/release/rabbit --model
<checkpoint-dir> --prompt "What is the capital of France?" --max-tokens 40 --expert-cache 64`. The
v0.23.0 row was measured on the commit tagged `v0.23.0`, the SIMD-only row on `v0.24.0`, the
`io_uring` row on `v0.25.0` — **add `--no-usage-cache`** for that last one (or any repeat run on a
checkpoint dir that already has real generation history in it), per "A red herring" above, or the
measurement will be contaminated by `mlock` retry overhead unrelated to the thing actually being
measured. The `--expert-cache 64` in the command above is explicit, so `v0.25.0`'s auto-clamp
safety fix (see "A real crash, not a red herring" below) doesn't kick in — that's intentional, to
keep every row in this table using the same cache capacity for a fair comparison.

**Isolated kernel benchmark**: `cargo bench --bench kernels -- matmul_mxfp4`.

Hardware: same machine as `PERFORMANCE.md` (AMD Ryzen AI 9 HX 370, 12 cores/24 threads, AVX2 +
AVX-512F/BW/VNNI), running the real `moonshotai/Kimi-K3` checkpoint (1.56TB, 96 shards, native OCP
MXFP4-quantized routed experts) from a local NVMe drive.
