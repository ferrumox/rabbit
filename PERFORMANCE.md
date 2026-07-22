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

**Overall: 0.29 → 0.60 words/sec, about 2× faster, same test, same machine, across these five
versions.** Zoomed out to the estimated/historical numbers above (~0.05 words/sec just before the
multi-core change), the full picture is roughly **0.05 → 0.60 words/sec, about 10× faster.**

**Opt-in, not shown above:** passing `--cache-route` on v0.16.0+ makes rabbit prefer expert data
it already has close at hand instead of always fetching fresh from disk — **16.5% faster** in its
own dedicated test, on top of whichever version it's paired with. It's off unless requested
because it hasn't been tested as widely as everything else on this page yet.

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

## Reproducing these numbers

`--model <checkpoint-dir> --prompt "Write two sentences describing France." --max-tokens 30
--expert-cache 64 --no-usage-cache --temperature 0` (temperature 0 for determinism). Checked out
each version via `git worktree`, built fresh, ran the exact same command. Hardware: AMD Ryzen AI
9 HX 370 (12 cores / 24 threads, AVX2 + AVX-512F/BW/VNNI), 123 GB RAM, NVMe SSD, running the real
[`jlnsrk/GLM-5.2-colibri-int4`](https://huggingface.co/jlnsrk/GLM-5.2-colibri-int4) checkpoint
(378 GB, 744B params, community int4 conversion via colibrì's own tooling). See `rabbit-plan.md`
for the full phase-by-phase development history behind each version.
