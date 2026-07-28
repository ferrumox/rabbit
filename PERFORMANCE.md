# Performance history

## In plain terms

rabbit runs a 744-billion-parameter AI model on a single ordinary computer by keeping most of the
model on disk and pulling pieces of it into memory as needed, rather than requiring enough RAM to
hold the whole thing at once. That's what makes a model this size runnable at all here — but
pulling pieces from disk is inherently slower than if everything already lived in memory, so a
lot of the work on rabbit has been finding safe ways to claw that speed back.

**The result: generation speed has roughly doubled since the earliest version that could be
properly measured**, tested the same way, on the same prompt, on every version in the main table
below — not estimated. Two changes account for almost all of it: spreading the model's math
across every CPU core on the machine instead of just one core (first for the bulk of the
calculations, later for a step that had been missed the first time), plus, most recently,
teaching rabbit to use newer, wider CPU instructions for its single most common calculation. Zoom
out further, back to just before that multi-core change landed, and the honest, if rougher,
picture is closer to **roughly 10× faster** — real numbers, just not all measured under the same
controlled conditions (see below for exactly which numbers are solid and which are estimates). A
couple of other ideas were tried along the way and made things worse or made no real difference —
those are listed further down so nobody re-tries them.

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

**Overall: 0.29 → 0.73 words/sec, about 2.5× faster, same test, same machine, across these six
versions.** Zoomed out to the estimated/historical numbers above (~0.05 words/sec just before the
multi-core change), the full picture is roughly **0.05 → 0.73 words/sec, about 13× faster.**

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
place this fix helps. There's still a real, unexplained gap left between the achieved ~2 GB/s
pure disk-wait rate and the probe's ~4.75 GB/s ceiling —
flagged as a lead for a future session, not chased further this round.

## Tried and didn't help

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

Hardware throughout: AMD Ryzen AI 9 HX 370 (12 cores / 24 threads, AVX2 + AVX-512F/BW/VNNI),
123 GB RAM, NVMe SSD, running the real
[`jlnsrk/GLM-5.2-colibri-int4`](https://huggingface.co/jlnsrk/GLM-5.2-colibri-int4) checkpoint
(378 GB, 744B params, community int4 conversion via colibrì's own tooling). See `rabbit-plan.md`
for the full phase-by-phase development history behind each version.
