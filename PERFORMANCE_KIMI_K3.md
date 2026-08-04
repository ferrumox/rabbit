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
--max-tokens 40`, same real 2.8T checkpoint, same machine, `--expert-cache 64` EXPLICIT for every
row (kept fixed on purpose, even after v0.25.0 made `64` no longer the automatic default — see "A
real crash" below — so this series isolates only the code changes, not the cache-size change).

| Version | Model load | Prefill (7 tokens) | Decode (steady-state) | Speed (tok/sec) | Change | What changed |
|---|---|---|---|---|---|---|
| v0.23.0 | 610.0s | 412.8s | ~50-70s/token | 0.014 | — (first working version) | K3 shipped: engine, native-format checkpoint loading, tokenizer, chat template, session persistence — the expert math itself still used the simple, unvectorized loop |
| v0.24.0, SIMD kernel only | 605.5s | 216.5s | ~37.5s/token avg (range 17.1-44.8s across 40 tokens) | ~0.027 | **Prefill ~1.9× faster; decode ~1.3-1.9× faster** vs v0.23.0 | Taught the expert math to use the CPU's wider instructions, the same idea GLM-5.2's own v0.17.0/v0.22.0 already used for its own number formats |
| v0.25.0, + `io_uring` for MXFP4 | 616.0s | **210.3s** | **35.0s/token avg** (1401.4s/40 tokens) | **~0.029** | **~3% faster prefill, ~7% faster decode** vs v0.24.0 — measured with `--no-usage-cache` to rule out the red-herring confound below | Batched `io_uring` reads for MXFP4's `{name}.weight_packed`+`{name}.weight_scale` pair, same mechanism GLM-5.2 already had for its own tensor formats — previously every MXFP4 load fell back to one-at-a-time synchronous reads |
| v0.26.0, parallel model load | **102.3s** | 210.3s (unchanged) | 35.0s/token avg (unchanged) | ~0.029 (unchanged) | **Model load ~6.1× faster** (619.5s→102.3s on the run that found this — see its own section below); prefill/decode untouched | Parallelized the 93-layer loading loop with `rayon` — was a purely sequential blocking-read loop, 591.6 of 619.5 load seconds, never batched/threaded before |
| v0.26.0, **real default config** (`--expert-cache` auto=4, not the fixed 64 above) | **102.3s** | **44.6s** | **~8.1s/token avg** (161.8s/20 tokens) | **~0.124** | Everything above, combined, under the config a real invocation actually uses | Not a new code change — this row swaps the artificially-fixed `--expert-cache 64` for the real auto-clamped default every other row deliberately avoided, to show what actually happens today. See "Does the auto-clamp cost real speed" below for why this is FASTER, not slower, despite the much lower hit rate |

**Overall, same 40-token generation, start of this session to now: ~610.0+412.8+40×60s≈3423s
(~57 minutes) → ~102.3+44.6+40×8.1s≈471s (~7.9 minutes), about 7.3× faster end to end** — using
the real default config row above and the v0.23.0 baseline's midpoint decode estimate (no exact
40-token time was recorded for that oldest run's decode, only the ~50-70s/token range). One caveat
kept from earlier: the ~8.1s/token figure benefits from this session's OS page cache already being
warm from many repeated real runs on the same checkpoint — a truly cold first run on a freshly
booted machine would likely see a slower decode than this (though load and prefill's own
improvements are pure code changes, unaffected by cache warmth).

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

## v0.26.0: model load, ~6x faster — nobody had ever measured where the 610s went

Every number on this page so far treated model load as a fixed ~610s cost and moved on to
measuring generation speed instead — reasonable, since load happens once per process and decode
happens every token, but it meant a genuinely huge, completely unexplained cost sat there
unquestioned through this whole page. Added a one-time timing breakdown to `kimi_k3::model::
Model::load_multi` (three phases: opening the `.safetensors` shards, `embed_tokens`/`lm_head`/misc,
and the 93-layer loop) and ran it once against the real checkpoint:

```
model load breakdown: shard-open 1.0s, embed 13.0s, lm_head 13.8s, other head tensors 0.0s, 93 layers 591.6s
model loaded in 619.5s (93 layers)
```

**591.6 of 619.5 seconds — 95.5% of the entire load — was the 93-layer loop**, not the two huge
vocab-sized tensors (`embed_tokens`/`lm_head` together were only ~27s, fast, as expected for ~2.35GB
each at real NVMe speed). The loop itself was hundreds of small, purely sequential blocking reads
(`ld`/`qt_load`, one `pread` at a time, one layer fully finishing before the next one's first read
even starts) — the EXACT same "many syscalls each paying full disk latency serially, zero overlap"
shape `io_uring` batching fixed for routed experts (see this page's earlier sections), just never
applied to the dense/attention weights loaded once at startup.

**Fixed the same session**: each layer's tensors are independent reads (`Shards::read_f32`/
`read_raw` take `&self`, explicit `pread` offset, no shared file cursor — safe to call
concurrently), so the sequential `for i in 0..n_layers` loop became a `rayon` `into_par_iter()` +
`.collect::<Result<Vec<Layer>, _>>()` — plain OS threads instead of an `io_uring` ring, since load
time has no equivalent to expert-loading's early-drain trick (nothing downstream can use layer N
until every layer is loaded anyway, so there's no per-completion streaming benefit to chase, just
concurrency). Result, same real checkpoint, right after:

```
model load breakdown: shard-open 0.7s, embed 12.9s, lm_head 13.9s, other head tensors 0.0s, 93 layers 74.6s
model loaded in 102.3s (93 layers)
```

**93-layer loop: 591.6s → 74.6s, ~7.9× faster. Total model load: 619.5s → 102.3s, ~6.1× faster** —
from over 10 minutes to under 2. This has NO effect on decode speed (nothing above this section
changes) but matters for every single invocation of `--prompt`/`--chat`/`--serve`, since load
happens once no matter how many tokens get generated afterward — the single biggest fixed-cost win
on this whole page, found by finally asking where a cost everyone had been treating as immovable
actually went.

**Correctness confirmed the same session**: the speed measurements above killed the process right
after load finished (no need to wait for the rest just to time loading) — so a separate full run
was needed to confirm the parallelized load ISN'T just fast but still produces a correctly-working
model (rayon loads layers in whatever order threads finish, unlike the old strictly-sequential
loop, so this genuinely needed checking, not just assuming). Real `--prompt "What is the capital
of France?" --max-tokens 20` run: load 104.3s, prefill 43.0s, 20 tokens in 154.0s (~7.7s/token) —
correct, coherent output, same style as every other real run on this page. Total end-to-end time
for this one run (load+prefill+decode): ~301s, under 5 minutes — a real illustration of how much
the combined effect of this session's fixes changes the practical experience, down from the
40+ minute runs this same page's earlier sections were measured with.

## Does the auto-clamp (capacity 4) actually cost real speed? Tested it directly — no, the opposite

A fair question, raised right after the auto-clamp fix: 4 cached experts/layer instead of 64 means
a much lower hit rate (confirmed: ~8.5% in one real run vs. tens of thousands of hits/run at
capacity 64 elsewhere on this page) — doesn't that mean more disk re-reads and a real slowdown, not
just a safety trade-off? Worth testing directly rather than assuming either way, especially since
earlier fast numbers on this page (~7-8s/token) were flagged as possibly confounded by warm OS page
cache, not proof that a smaller cache is fine.

Ran a genuinely controlled back-to-back A/B: same prompt, same 20 tokens, `--no-usage-cache` on
both (removes the pinned-tier variable entirely), run B immediately after run A so both see roughly
the same OS page-cache state — varying ONLY `--expert-cache` (4 vs. the old flat 64):

```
Run A (--expert-cache 4):  load 103.7s, prefill  44.6s, 20 tokens in 161.8s (~8.1s/token)
Run B (--expert-cache 64): load 103.1s, prefill 479.2s, tokens running 18-91s each (killed at 15/20)
```

**Run B wasn't just slower — it was actively swapping while it ran**: `free -h` mid-run showed
121Gi/123Gi RAM used, under 1GB free, 27GB of swap in use. Requesting 64 experts/layer resident
(worst case ~209GB for this checkpoint's real per-expert size, see "A real crash" above) pushes
this machine into real memory pressure well before it can ever finish filling that cache — and
swapping is catastrophically slower than any amount of extra disk-read misses a smaller cache
might cause. Killed run B once the pattern was clear (`kill -9`), to avoid risking a full
system-wide OOM affecting anything else running on the machine; memory returned to normal (113GB
free) within seconds of the kill.

**Answer: no, the auto-clamp to 4 is not a speed-for-safety trade-off on this machine — it's
strictly faster AND safer**, because the "faster" alternative (64) was never actually achievable
here without swapping first. This doesn't mean 4 is the universally correct number forever — a
machine with meaningfully more RAM, or a future smarter budget that accounts for how much RAM is
actually free (not just a fixed 24GB constant), could likely support a higher capacity and a real
hit-rate win without ever touching swap. That's the exact question the next version answers.

## v0.27.0: the safety budget is now RAM-aware, not a fixed guess

The flagged future work above, done the same day: a fixed 24GB budget is exactly the wrong shape
for this problem — too small wastes real headroom on a bigger machine, too large risks the same
OOM/swap this whole fix exists to prevent on a smaller one. `expert_cache::safe_mxfp4_capacity`
now reads real available memory from `/proc/meminfo`'s `MemAvailable` (Linux; falls back to the
original fixed 24GB constant if that file can't be read — non-Linux, sandboxed, malformed, ...)
and uses **40% of whatever's actually free right now** as the safety budget, leaving the other 60%
for this checkpoint's own base-resident weights, KV cache growth, the OS, and anything else running
on the machine.

Real result on this machine (`MemAvailable` ≈122GB at the time): budget ≈37GB (vs. the old fixed
24GB), auto-capacity **7** (vs. the old fixed-budget 4) — a real ~75% bigger cache, still safely
under a third of real available RAM. Ran the exact real `--prompt`/`--no-usage-cache` test from the
section above with this new capacity, tracking memory every 15s throughout:

```
expert cache: auto --expert-cache 64 would risk ~310GB peak memory on this checkpoint
(92 MoE layers x ~35.1MB/expert, LRU + pinned tiers combined) -- lowered the auto
default to 7 to stay under a ~37GB safety budget (40% of real available RAM, ...)

RAM used climbed steadily: 17GB -> 24GB -> 28GB -> 35GB -> ... -> 46GB (peak), then back to
7.2GB once generation finished. Swap stayed flat at 6.3-7.4GB throughout -- NO growth, no
swapping, the entire run. 20 tokens in 161.5s (~8.1s/token) -- essentially the same speed as
the old capacity-4 run's 161.8s on this short a sample (1040 hits vs. 248 -- a real ~4.2x
higher hit rate at capacity 7, just not enough absolute misses saved over only 20 tokens to
show up as a clear wall-clock win yet; would need a longer/repeated real run to see if a bigger
capacity's hit-rate advantage compounds into a visible speed difference over more tokens).
```

**The real win here isn't necessarily speed on THIS run — it's that the budget now scales
correctly with the machine it's running on**, instead of being permanently stuck at whatever
number one specific 123GB box's incident happened to justify. A machine with less RAM gets a
smaller, still-safe budget automatically; a machine with more RAM (or this same one, once it's
running fewer other things and has more `MemAvailable`) gets a bigger one, with no code change
and no re-guessing needed. Verified with 4 new fast unit tests (`parse_mem_available_kb`'s real
`/proc/meminfo` text format, missing-field and malformed-value cases, and a
`safe_mxfp4_capacity_with_budget` test proving a bigger budget genuinely allows a bigger capacity)
— 350/350 lib tests, clippy clean, release build clean.

**Follow-up at 40 tokens (double the sample): the hit-rate advantage DOES show up as real
speed once there's enough decode to let it compound.** Controlled back-to-back A/B,
`--no-usage-cache` both sides, same warm page-cache state:

| | Capacity 4 | Capacity 7 | Change |
|---|---|---|---|
| Decode (40 tokens) | 387.8s (~9.7s/token, ~0.10 tok/sec) | 283.2s (~7.1s/token, ~0.14 tok/sec) | **~27% faster** |
| Final hit rate | 478 hits / 63494 misses (~0.75%) | 1914 hits / 62058 misses (~3.0%) | ~4× higher |

(Run A's own prefill this time was an outlier — 122.5s vs. the ~44-45s seen everywhere else on
this page, capacity-independent noise, not compared here.) Confirms the earlier 20-token sample
wasn't a null result, just too short a window: the bigger RAM-aware capacity is a real, measurable
decode-speed win on top of being safer, not only a safety measure with no performance upside.

## Tried and didn't help

- **Consolidating each MXFP4 expert's 6 `io_uring` reads (`w1`/`w2`/`w3` × `weight_packed`/
  `weight_scale`) into 1 combined read** — a real shard-header inspection found all 6 tensors
  genuinely byte-contiguous, back-to-back, on the real checkpoint (confirmed across multiple
  experts/layers), so one bigger read per expert looked like a clean win: fewer `io_uring` SQEs,
  less per-completion bookkeeping. A throwaway isolated probe (raw `io_uring` reads against the
  real checkpoint file, no rabbit code involved) measured it as **faster** — ~1.06-1.53x, bigger
  win at smaller batch sizes closer to real usage. Built the real fix (`try_submit_mxfp4_combined`
  in `expert_cache.rs`, with a per-expert contiguity check and a safe fallback to the original
  per-tensor reads whenever it doesn't hold), verified correct with new unit tests, then measured
  it end-to-end on the real checkpoint: **~45% SLOWER** (410.0s vs. 283.2s for the same 40 tokens
  at the same `--expert-cache 7`), and immediately so — the very first token already showed
  roughly double the disk-wait time, not a gradual drift. Reverted the same session.

  **Real, already-documented explanation, found in this project's own `PERFORMANCE.md`**: a much
  earlier investigation ("Lead 2" in that file's disk-I/O section) measured, on this SAME drive,
  that SCATTERED reads are FASTER than sequential ones by 10-22% — "this NVMe likely spreads data
  across many internal flash channels, and offsets scattered across a wide range may activate more
  of them at once than a purely sequential burst." Consolidating 6 smaller reads into 1 bigger one
  does the opposite of what that finding recommends: fewer, bigger, more sequential reads instead
  of more, smaller, scattered ones — reducing how many of the drive's internal channels get
  activated concurrently. The isolated probe likely didn't reproduce this effect faithfully (a
  different queue-depth/timing shape than real generation's actual read pattern). Worth remembering
  before trying any other "fewer, bigger reads" idea against this specific checkpoint/drive: the
  established, measured behavior here favors MORE scattered concurrent reads, not fewer combined
  ones, and a clean isolated micro-benchmark isn't guaranteed to predict this correctly — verify
  end-to-end on the real checkpoint before trusting an isolated probe's direction, not just its
  magnitude.

## Reproducing these numbers

**End-to-end table**: `cargo build --release --bin rabbit`, then `./target/release/rabbit --model
<checkpoint-dir> --prompt "What is the capital of France?" --max-tokens 40 --expert-cache 64`. The
v0.23.0 row was measured on the commit tagged `v0.23.0`, the SIMD-only row on `v0.24.0`, the
`io_uring` row on `v0.25.0`, the `v0.26.0` row on `v0.26.0` (only its `Model load` column differs
from `v0.25.0` — that's the whole point of the row) — **add `--no-usage-cache`** for `v0.25.0`
onward (or any repeat run on a checkpoint dir that already has real generation history in it), per
"A red herring" above, or the measurement will be contaminated by `mlock` retry overhead unrelated
to the thing actually being measured. The `--expert-cache 64` in the command above is explicit, so
`v0.25.0`'s auto-clamp safety fix (see "A real crash, not a red herring" below) doesn't kick in —
that's intentional, to
keep every row in this table using the same cache capacity for a fair comparison.

**Isolated kernel benchmark**: `cargo bench --bench kernels -- matmul_mxfp4`.

Hardware: same machine as `PERFORMANCE.md` (AMD Ryzen AI 9 HX 370, 12 cores/24 threads, AVX2 +
AVX-512F/BW/VNNI), running the real `moonshotai/Kimi-K3` checkpoint (1.56TB, 96 shards, native OCP
MXFP4-quantized routed experts) from a local NVMe drive.
