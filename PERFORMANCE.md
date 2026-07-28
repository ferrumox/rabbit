# Performance history

## In plain terms

rabbit runs a 744-billion-parameter AI model on a single ordinary computer by keeping most of the
model on disk and pulling pieces of it into memory as needed, rather than requiring enough RAM to
hold the whole thing at once. That's what makes a model this size runnable at all here — but
pulling pieces from disk is inherently slower than if everything already lived in memory, so a
lot of the work on rabbit has been finding safe ways to claw that speed back.

**The result: generation speed is now very close to 3× the earliest version that could be
properly measured**, tested the same way, on the same prompt, on every version in the main table
below — not estimated. Three changes account for almost all of it: spreading the model's math
across every CPU core on the machine instead of just one core (first for the bulk of the
calculations, later for a step that had been missed the first time), teaching rabbit to use
newer, wider CPU instructions for its single most common calculation, and, most recently,
starting each newly-fetched expert's math the moment ITS OWN data arrives instead of waiting for
every other expert in the same batch too. Zoom out further, back to just before that multi-core
change landed, and the honest, if rougher, picture is closer to **roughly 17× faster** — real
numbers, just not all measured under the same controlled conditions (see below for exactly which
numbers are solid and which are estimates). A couple of other ideas were tried along the way and
made things worse or made no real difference — those are listed further down so nobody re-tries
them.

## Before there was a way to measure this properly

rabbit couldn't actually run a prompt from the command line until v0.10.0 — earlier versions were
still being built and had no user-facing way to generate text against the real model at all, so
nothing below this point is a fresh, controlled measurement like the table further down. Included
anyway, each clearly labeled with how solid it is, because "no number" reads as "nothing happened"
when real engineering did:

| Version | Speed (words/sec) | How solid is this |
|---|---|---|
| v0.1.0 – v0.5.0 | *(nothing to measure)* | The model couldn't generate anything yet at all — these versions built the tokenizer, quantization, and model-loading pieces first. |
| v0.6.0 / v0.7.0 (before the CPU got used efficiently) | **very roughly 0.003–0.006** | **Rough estimate, wide range on purpose.** Calculated by taking a real historical measurement from just after this point and working backward using the real speed-up factor the next change delivered (measured today, at that exact old version, via `cargo bench`) — not a guess, but a chain of real numbers with real uncertainty in it, not a direct measurement. |
| v0.8.0 (faster CPU instructions arrive) | **~0.05** | Reasoned, not directly measured: a later, isolated test showed the very next change (v0.9.0, below) made no real difference to generation speed by itself, so this version's speed is assumed equal to v0.9.0's. |
| v0.9.0 (just before the multi-core change) | **~0.05** | **Real measurement, taken at the time** — not re-verified today, but a genuine recorded number, not a guess. |

## Speed by version

Every row from here on is the *same* test — same prompt ("Write two sentences describing
France."), same 30-word generation limit, same real 744-billion-parameter checkpoint, same
machine — run fresh today against each version's actual code, not old notes. "Speed" is words
generated per second once the model is done reading the prompt (the slow, disk-heavy part
beforehand isn't counted, so this reflects the ongoing generation speed a user actually feels) —
the same unit as the historical table above, so the two are directly comparable.

| Version | Speed (words/sec) | Change | What changed |
|---|---|---|---|
| v0.10.0 | 0.29 | — (earliest runnable version) | Spread the model's math across every CPU core instead of one |
| v0.14.0 | 0.29 | no change | Added chat mode, the web server, and conversation memory — none of it touches the speed-critical math |
| v0.15.0 | 0.42 | **+44%** | Did the same all-cores trick for one more step used during generation, missed the first time |
| v0.16.0 | 0.40 | ~flat (run-to-run noise) | Added an opt-in "prefer nearby data" mode (off by default here — see below) |
| v0.17.0 | 0.60 | **+49%** | Taught rabbit to use newer, wider CPU instructions for its single most common calculation |
| v0.18.0 | 0.73 | **+21%** | Removed a redundant full copy of every piece of data fetched from disk (see below — this was found and measured as a *prefill* fix, but it turns out to speed up the ongoing generation too, since the same fetch code runs there as well) |
| v0.19.0 | 0.84 | **+15%** | Each newly-fetched expert now starts its matmul the moment its own data lands, instead of waiting for the whole batch of experts to finish loading first (see below) |

**Overall: 0.29 → 0.84 words/sec, about 2.9× faster, same test, same machine, across these seven
versions.** Zoomed out to the estimated/historical numbers above (~0.05 words/sec just before the
multi-core change), the full picture is roughly **0.05 → 0.84 words/sec, about 17× faster.**

**Opt-in, not shown above:** passing `--cache-route` on v0.16.0+ makes rabbit prefer expert data
it already has close at hand instead of always fetching fresh from disk — **16.5% faster** in its
own dedicated test, on top of whichever version it's paired with. It's off unless requested
because it hasn't been tested as widely as everything else on this page yet.

## v0.18.0: an efficiency win, not a speed win

`rayon` (the library rabbit uses to spread work across CPU cores) defaults to one thread per
*logical* core — on this machine, 24 (12 physical cores × 2 via hyperthreading/SMT). Rabbit now
sizes its thread pool to *physical* cores instead (12 here), matching a tuning colibrì already
does for the same reason. Measured on the real checkpoint, same prompt as everywhere else on this
page: wall-clock speed was **the same either way** (68.4s at 24 threads vs 67.1s at 12 — within
normal run-to-run noise, not a real difference), but total CPU time used was roughly **half**
(6m38s of CPU-seconds at 24 threads vs 3m23s at 12). This confirms the bottleneck here is memory
and disk bandwidth, not raw compute throughput — the extra logical threads from hyperthreading
were burning CPU cycles without producing any extra speed. Same speed, half the CPU: less heat,
more headroom for anything else running on the machine, no measured downside. Configurable via
`--threads N` if a different count works better on other hardware.

## v0.18.0: prefill I/O — a real speed win, found by measuring the gap between "achievable" and "achieved"

Prompted by the finding above, the next question was: if disk bandwidth is the limit, is rabbit
actually getting close to what this NVMe can deliver? A standalone probe (many concurrent ~6MB
reads spread across the real checkpoint's files, independent of rabbit's own code) measured this
drive at **~4.75 GB/s**. A real generation run's own reported "disk I/O" time implied only
**~1.15 GB/s** — a roughly 4x gap worth explaining before assuming the drive itself was the wall.

Splitting that reported "disk I/O" number into two separately-timed pieces (pure `io_uring` wait
vs. the decode/copy work that happens right after) found the real story: on a 133-token real
prompt, of a reported 61.7s "disk I/O" figure, only **36.2s was the drive actually being waited
on — the other 25.5s (~41%) was CPU work being mislabeled as disk time**, mostly an unnecessary
full copy of each expert tensor's bytes right after `io_uring` had already read them into an
owned buffer (`raw.to_vec()` cloning a buffer `QT::from_packed` could have taken directly).
Fixing that copy — plus a smaller, related fix (each expert's tiny `.qs` scale sidecar was being
read with its own *synchronous, sequential* `pread` call for every miss, one at a time, before
the batched `io_uring` submission even started, instead of joining the same batched round) —
measured, on the same 133-token prompt, repeated for a control:

| | Total prefill | "Disk I/O" bucket |
|---|---|---|
| Before | 114.8s | 61.1s |
| After (both fixes) | 96.2s / 98.3s (avg 97.25s) | 41.8s / 42.2s (avg 42.0s) |
| **Change** | **~15% faster** | **~31% faster** |

The scale-sidecar batching alone measured no real difference (a separate, controlled A/B showed
~61.1s vs ~61.7s — the sequential small reads mattered less than expected, not the dominant
cost). The buffer-copy elimination was the one that actually moved the number — and, worth
correcting here since an earlier draft of this section got it wrong: **this is NOT a prefill-only
win.** The same buffer-copy code path runs every time rabbit fetches a missing expert from disk,
prefill or not — decode routes to `topk` experts per generated word too, so the fix pays off
there as well. Confirmed directly with the standard "Speed by version" test above: decode-only
speed went from v0.17.0's 0.60 to v0.18.0's **0.73 words/sec, a genuine ~21% faster**, reproduced
twice (0.725 and 0.727). The prefill-specific numbers above are still real and still the bigger
percentage win on a prompt long enough for expert-loading to dominate, but it's not the *only*
place this fix helps. At the time, there was still a real-looking gap between the achieved ~2 GB/s
pure disk-wait rate and the probe's ~4.75 GB/s ceiling — see the section right below for what
chasing that gap further actually found.

## v0.18.0: chasing the rest of the disk-I/O gap — two leads investigated, both ruled out

Two candidate explanations for that remaining gap were tested directly against the real
checkpoint. Neither held up — worth recording so nobody re-investigates the same dead end:

**Lead 1: is the queue too shallow during decode to keep the drive busy?** Decode only routes
`topk` experts per generated word, so a decode round can never submit as many concurrent
`io_uring` reads as a prefill round does. A temporary diagnostic logged every round's size across
a real 107-token prefill + 10 decode steps (675 rounds): decode rounds really are tiny (median 3
missing experts/round = 18 reads) next to prefill's up to 45 misses/round (270 reads), out of a
512-deep ring. But bucketing ONLY the genuinely disk-bound rounds (filtering out ones fast enough
to be a page-cache hit) by size found **no relationship between queue depth and achieved
bandwidth**: rounds with just 1-3 misses averaged 3.47 GB/s, rounds with 26+ misses averaged the
same 3.47 GB/s. This same clean measurement also revealed the earlier "~2 GB/s" figure was a
rough blended average (mixing cache-hit and genuinely-cold rounds together) — the real disk-bound
rate is closer to **~3.5-4.2 GB/s**, meaningfully nearer the probe's ~4.75 GB/s ceiling than it
first looked. Ruled out; reverted the diagnostic (it had done its job).

**Lead 2: is rabbit's scattered per-tensor access pattern the reason, vs the original probe's
more sequential large reads?** Every real expert-weight read lands at an essentially arbitrary
offset within an 18GB-to-378GB shard file, picked by which experts the router selects — very
different from the original probe's more sequential large chunks. Built a focused, throwaway
`io_uring` probe isolating *only* this variable (same file, same 6MB chunk size, same queue depth
of 64, same total bytes): split one real shard in half, read one half sequentially and the other
at random offsets, then repeated with the halves swapped to rule out any position-within-file
bias. Result, on 3 different shards (6 runs total): **scattered reads were faster than sequential
in every single run, by 10-22%** — the opposite of the hypothesis. Best guess why, not chased
further: this NVMe likely spreads data across many internal flash channels, and offsets scattered
across a wide range may activate more of them at once than a purely sequential burst; buffered
sequential reads can also trigger kernel readahead that does work rabbit's own explicit reads
don't need. Ruled out.

**Where this leaves the disk-I/O investigation**: both leading candidate explanations for the raw
*bandwidth* gap are now eliminated, and the clean measurements from investigating them
(~2.5-4.7 GB/s depending on shard/condition) sit close enough to the original probe's
~4.75 GB/s ceiling that there may not be much real gap left to chase there. Raising the ceiling
itself is likely exhausted for now, barring a genuinely new lead — but Lead 1's own measurement
(a real overlap window inside each round) turned into a genuine win a different way: see
v0.19.0 below. One idea was scoped but not attempted: the real checkpoint's `gate_proj`/`up_proj`/
`down_proj` tensors (and their `.qs` scale sidecars) turned out to be byte-contiguous on disk per
expert — confirmed directly against the checkpoint's own header — which would let rabbit fetch an
expert in 1-2 `io_uring` reads instead of 3-6. Consolidating the tiny `.qs` sidecars this way was
tried and measured as no real difference (see below); consolidating the much bigger main tensors
the same way was reasoned through and *not* attempted, because reconstructing 3 owned buffers from
1 combined read isn't actually free in Rust — it would reintroduce most of the exact copy cost the
buffer-copy fix above just eliminated, unless `QT`'s internal storage first moves to something
that can be sliced without copying. A bigger, separate refactor, not done this round.

## v0.19.0: per-expert early drain — a real win found while investigating the disk-I/O gap

Lead 1 above (queue depth) left behind a more useful number than it first looked like: across
297 genuinely disk-bound rounds, **on average 33% of a round's total wait time still remained
even after half its reads had already completed** — a real, measured overlap window, not a
guess. Until this version, rabbit waited for an ENTIRE batch of expert reads to land before
computing ANY of their matmuls, even though many of them (especially the tiny `.qs` scale
sidecars, which tend to land first) were often ready far earlier than the batch as a whole.

Rabbit's expert loader now hands each expert to the compute step the moment ITS OWN reads land
(3 tensors + scale sidecar), while the disk keeps working on the rest of the batch — instead of
`io_uring`'s `submit_and_wait` blocking for the whole round before any matmul starts. This
required restructuring the completion path to wait one completion at a time and track, per
expert, how many of its reads have arrived so far, but changes nothing about the actual bytes
read from disk or the math performed on them — proven bit-identical against the same real-model
oracle tests every other change on this page has been checked against.

Measured, controlled (`git worktree` before/after, same "Speed by version" test, repeated —
2 runs before, 3 after):

| | Prefill | Decode-only |
|---|---|---|
| Before | 14.1s / 14.0s | 0.730 / 0.730 words/sec |
| After | 14.1s / 14.1s / 14.0s | 0.868 / 0.829 / 0.836 words/sec |
| **Change** | ~no change | **0.730 → 0.844 avg, ~+15.6% faster** |

Prefill didn't move — it already had its own, separate overlap trick (computing the shared
expert's contribution while the first batch of routed experts loads), which likely already
captured most of the slack available there. Decode has no equivalent trick of its own, so this
is where the new overlap actually shows up: real, reproducible, consistent across all 3 "after"
runs landing clearly above the tight 0.730/0.730 "before" control band.

## Tried and didn't help

- **Consolidating each expert's 3 tiny `.qs` scale-sidecar reads into 1** — the real checkpoint
  stores them byte-contiguous per expert (confirmed against its own header), so this replaced up
  to 3 small `io_uring` reads/expert with 1 larger one, with a dedicated test proving the merged
  path decodes identically to the old per-tensor one. Measured, controlled (2 runs each side via
  `git worktree`, same 107-token prompt): 81.9s/80.8s before vs 82.3s/81.2s after — **no real
  difference**. These sidecars (~40KB/expert) were already known to be too small for their read
  count to matter (see the prefill-I/O section above); consolidating them further didn't change
  that. Reverted — the added code was meaningfully more complex than what it replaced, for a null
  result.
- **Batching several experts' calculations together during generation** — looked like it should
  reduce overhead, actually made things ~25% slower (the cost of preparing the batched data
  outweighed the savings). Reverted, never shipped.
- **Guessing which data to pre-fetch based on the last few words generated** — no measurable
  benefit once properly tested, likely a very slight net negative. Reverted, never shipped.
- **Loading more data into memory at once** (raising the cache size well past its default) — on
  this machine, pushing past a certain point causes the computer to run out of RAM and start
  swapping to disk, which is far slower than the disk streaming rabbit already does on purpose.
  Confirms the current default is close to this machine's real ceiling, not overly conservative.
- **An earlier, "eager" version of the "prefer nearby data" cache** loaded a large batch of data
  upfront before generation even started — measured as a net loss for one-off prompts (nothing to
  amortize that upfront cost against). Redesigned to load lazily instead before ever being
  released; the eager version never shipped.
- **Batching three specific weight-reading steps during prompt processing** (ported from a real,
  measured colibrì win — colibrì saw -4.5% total prompt-processing time from this exact change).
  Built it, proved it bit-identical with dedicated tests, added a stopwatch just for this one
  step to measure it in isolation (the step is normally too small next to disk I/O to see at
  all) — and found a **reproducible ~16-17% regression** in rabbit specifically (4.95s → 5.75s
  average, repeated twice each side, stable both times), the opposite of colibrì's result. Root
  cause, best understanding: colibrì's version of this step re-reads the same weight data from
  scratch on every one of many small calls, so batching genuinely saves work there; rabbit's
  equivalent step already reads each piece of weight data once per call regardless of batch size
  (a structural difference in how the two engines are built), so batching only added the cost of
  a bigger temporary buffer and its cleanup step without saving anything to offset it. Reverted,
  never shipped — a clear example of "a technique that's a real, measured win in the code it was
  copied from isn't automatically a win once ported," worth remembering before porting another
  colibrì optimization without measuring it here first.

## Reproducing these numbers

**"Speed by version" table above**: `--model <checkpoint-dir> --prompt "Write two sentences
describing France." --max-tokens 30 --expert-cache 64 --no-usage-cache --temperature 0`
(temperature 0 for determinism). Checked out each version via `git worktree`, built fresh, ran
the exact same command.

**v0.18.0's two sections** used a longer, ~133-token prompt instead (a short prompt's prefill is
almost entirely disk I/O for the FIRST batch of misses, too little compute or wait time in either
direction to isolate a small effect from noise) and `--max-tokens 1` (only prefill timing
mattered for those two measurements, so decode was cut short on purpose): `--model
<checkpoint-dir> --prompt "France is a country in Western Europe known for its rich history,
diverse culture, and significant influence on art, philosophy, and cuisine. ..." --max-tokens 1
--expert-cache 64 --no-usage-cache`.

**The two disk-I/O leads above** used a shorter, deterministic 107-token version of that same
prompt (`--temperature 0`, `--max-tokens 10` for the queue-depth measurement so both prefill and
several decode rounds got captured) plus a throwaway `io_uring` micro-benchmark reading directly
from the checkpoint's `.safetensors` shard files, independent of rabbit's own code — neither
diagnostic was kept (both were reverted/deleted after use), so reproducing them means rebuilding
the same kind of probe rather than running an existing command.

Hardware throughout: AMD Ryzen AI 9 HX 370 (12 cores / 24 threads, AVX2 + AVX-512F/BW/VNNI),
123 GB RAM, NVMe SSD, running the real
[`jlnsrk/GLM-5.2-colibri-int4`](https://huggingface.co/jlnsrk/GLM-5.2-colibri-int4) checkpoint
(378 GB, 744B params, community int4 conversion via colibrì's own tooling). See `rabbit-plan.md`
for the full phase-by-phase development history behind each version.
