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
| v0.22.0† | 1.02 | **+7.6%** | Vectorized the attention math used while generating each word with newer, wider CPU instructions (see below) |

**Overall: 0.29 → 1.02 words/sec, about 3.5× faster, across these eight versions** (v0.22.0's own
step measured on the same machine, see the † note just below for why it isn't from quite the same
test as the rows above it). Zoomed out to the estimated/historical numbers above (~0.05 words/sec
just before the multi-core change), the full picture is roughly **0.05 → 1.02 words/sec, about
20× faster.**

**† v0.22.0 was measured differently on purpose, not with the "generate 30 words and time it"
method every row above uses.** That method turned out to not be reliably repeatable once tested
carefully (see the v0.22.0 section below for why) — so its 0.84→1.02 words/sec figure instead
comes from a new, deterministic test built specifically to make an honest before/after comparison
possible again: 0.951→1.023 words/sec, **~7.6% faster**, real checkpoint, real disk I/O, just not
the model's own free-running word choices. The 1.02 entered above is that same result, rounded to
match this table's other entries.

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

## v0.22.0: AVX-512/AVX2 for the MLA-absorb decode path

Ported two optimizations found in colibrì's v1.1.0 release (pulled 2026-07-22) onto rabbit's
absorbed-attention decode path — the path used for short decode sequences (the normal case while
generating word by word). Two spots were still doing this math the plain, unvectorized way even
though rabbit already had the AVX-512 building block sitting right next to them, unused, since
v0.17.0's dual-accumulator kernel: the int4 weight-dequantizing helpers (`qt_addrow`/
`qt_matvec_rows`) now use AVX-512 instead of a byte-by-byte unpacking loop, and the score/
value-mixing step right after now uses AVX2 instead of a plain sequential sum.

**Isolated kernel benchmarks** (no disk I/O involved — these measure only the math itself, at the
real dimension this actually runs at on the real checkpoint):

| Kernel | Before | After | Change |
|---|---|---|---|
| `qt_addrow` int4 | 473.9 ns | 22.3 ns | **~21× faster** |
| `qt_matvec_rows` int4 | 428.1 ns | 23.1 ns | **~18.6× faster** |
| MLA score dot-product | 277.1 ns | 224.3 ns | **~1.24× faster** |
| MLA value-mix step | 265.5 ns | 87.3 ns | **~3× faster** |

**Real, controlled, end-to-end result**: **0.951 → 1.023 words/sec, ~7.6% faster**, generating the
same 30 words, 2 runs each side, before/after via `git worktree`, on the real checkpoint. The gap
between the huge kernel-level numbers above and this smaller end-to-end number is the same story
as everywhere else on this page: disk I/O still dominates a big share of every generated word's
time on this machine, so even a dramatically faster attention calculation only speeds up the
smaller, non-disk part of each step. Still a real, reproducible win — both "before" runs and both
"after" runs landed within a fraction of a second of each other, with a clear, much bigger gap
between the two groups.

**A methodology problem discovered along the way, worth its own note**: the obvious way to measure
this — generate 30 words with the model's own predictions and time it — turned out to not be
reliably repeatable AT ALL, even running the exact same unmodified program twice in a row with the
same prompt and the same random seed. Traced to v0.19.0's per-expert early-drain (above): it starts
each expert's math the moment ITS OWN data lands from disk, so the order contributions get added
up in depends on real disk-timing jitter, which genuinely differs run to run. A resulting
razor-thin difference in the math can occasionally flip which word the model picks at a close
call, and once that happens the entire rest of the generated text goes a different way — nothing
wrong, just a real side effect of a change that was a genuine, worthwhile speed win on its own
terms. It means "generate N words and time it" — the exact method the table above uses — can't
reliably isolate ONE specific change's effect on its own via a single run. Built a different way to
measure instead: a new test program (`examples/teacher_forced_decode_bench.rs`) feeds the model a
fixed, predetermined sequence of words rather than letting it choose its own, so the exact same
calculation happens on every run regardless of which version is being tested — the real,
end-to-end number above comes from that, not from the usual generate-and-time method.

**Status**: implemented and measured, sitting on branch `release/v0.22.0`, not yet merged into
`develop`/`main`.

## v0.22.0: reading the checkpoint from two drives at once

A second NVMe drive was added to this machine for capacity and read-bandwidth headroom. A true
RAID0 setup (combining both drives into one striped array at the operating-system level) turned
out not to be practical: the original drive has no free space to shrink into an array without
risky surgery on a live system (resizing its filesystem, reinstalling the boot loader). Instead,
rabbit itself now knows how to read a checkpoint's files split across more than one directory —
`--shard-dirs <dir1,dir2,...>` alongside `--model <dir>` — an idea borrowed from colibrì's own
`COLI_MODEL_DIRS` feature. No operating-system array involved, and unlike RAID0, each drive keeps
holding a genuinely complete, independently-readable portion of the checkpoint.

Tested on the real checkpoint, split roughly in half by file count between the original drive and
the new one (verified byte-for-byte: nothing lost in the split). Loads and generates correctly
reading from both locations at once.

**Raw disk-reading speed, measured with a small standalone test program** (reads many chunks
concurrently, independent of rabbit's own code — same style as the very first probe used
earlier on this page): **one drive alone: 4.2-4.6 GB/s. Both drives read at the same time:
8.7-9.1 GB/s — essentially double**, close to the ideal outcome and better than the original
conservative ~1.7-1.9x guess for a RAID0 setup. Repeated 3 times, each time reading a different,
never-before-touched part of the files (an earlier, sloppier attempt at this same test re-read
data already sitting in memory from the previous run and measured an impossible 20+ GB/s — a
reminder that repeat runs of ANY disk-speed test need fresh data each time, not just a fresh
timer).

**Measured: this does NOT translate into a faster generation speed, at least not for a short
30-word test with the default expert-cache size.** Same test as the rest of this page: one drive
alone (measured earlier, before this feature existed at all) 1.0269 and 1.0196 words/sec; both
drives split (two separate test rounds) 1.0286, 1.0403, 1.0179, 1.0389 words/sec. Averaging out to
~1.02 either way — under 1% apart, well inside the normal run-to-run wobble already seen
throughout this page. The doubled raw disk-reading speed measured above is real, but for a short
run like this one, most of the words generated are already being served from the in-memory expert
cache (not the disk) by the time a handful of words have gone by, so there usually isn't enough
disk-bound work left in this particular test for faster reading to speed up much of anything.
**`--shard-dirs` is still worth having** for its other real benefits (fitting a bigger checkpoint
across two drives, surviving one drive failing instead of losing everything) — it's just not, on
its own, a speed feature for a test shaped like this one.

**Also tried the part of a run that reads the MOST from disk (processing a long prompt for the
first time, before any word has been generated) — expected that part to show the doubled disk
speed most clearly, since it's normally the most disk-bound moment in the whole process. Measured
the opposite: reading from one drive alone was ~9-10% FASTER, not slower.** Same long test prompt
used earlier on this page ("France is a country in Western Europe..."), read once (no words
generated yet): one drive alone 112.6s and 114.2s (consistent); split across both drives 123.8s
and 125.0s (also consistent) — a real, repeatable difference, not noise, and not the QLC-drive
slow-recovery issue from below (checked for that specifically this time and it wasn't a factor).
No confirmed explanation for why; best guess, unconfirmed: reading from two drives at once only
actually pays off if rabbit's own reading code asks for enough of both drives' data at the same
time to keep them both busy simultaneously, and it may not be doing that as fully as the earlier,
dedicated raw-speed test did. **Bottom line, revised: on this machine, with this checkpoint,
`--shard-dirs` doesn't make anything faster — reading a long prompt for the first time may even
be a little slower split across two drives than reading it from one.** Still worth having for the
capacity and reliability reasons above, just not for speed.

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

**The v0.22.0 section's numbers**: isolated kernel benchmarks via `cargo
bench --bench kernels -- "qt_addrow_i4|qt_matvec_rows_i4|mla_score_dot|mla_vmix_axpy"`. The
end-to-end number via the new deterministic harness: `cargo run --release --example
teacher_forced_decode_bench -- --model <checkpoint-dir> --steps 30 --expert-cache 64`, run twice
per side via `git worktree` (before at the commit right before this branch's changes, after at the
branch tip).

Hardware throughout: AMD Ryzen AI 9 HX 370 (12 cores / 24 threads, AVX2 + AVX-512F/BW/VNNI),
123 GB RAM, NVMe SSD, running the real
[`jlnsrk/GLM-5.2-colibri-int4`](https://huggingface.co/jlnsrk/GLM-5.2-colibri-int4) checkpoint
(378 GB, 744B params, community int4 conversion via colibrì's own tooling). See `rabbit-plan.md`
for the full phase-by-phase development history behind each version.

---

# Kimi K3 on the target box (Granite Rapids-AP) — performance work

Everything below this line is a **separate effort from the dev-laptop GLM numbers above**: the
real 2.8T-param Kimi K3 checkpoint (`moonshotai/Kimi-K3`, MXFP4 experts) on a 2-socket, 6-NUMA
Intel Xeon 6975P-C cloud instance. K3 shipped correctness-first (v0.23.0) with **zero** decode
performance work; this is that work, executed against `K3_OPTIMIZE_BRIEF.md`. Different hardware,
different checkpoint, different unit (seconds/token, not words/sec) — do not compare these numbers
to the table above.

## Phase 0 — baseline + build configuration (2026-07-31)

### 0a. The target box (owner-provided `lscpu` + `numactl --hardware`, captured 2026-07-31)

- **CPU:** 2× Intel Xeon 6975P-C (Granite Rapids-AP, custom cloud SKU), 96 cores/socket, SMT on →
  **192 physical / 384 logical CPUs**. `Model name: Intel(R) Xeon(R) 6975P-C`, family 6 model 173
  stepping 1. KVM guest (`Hypervisor vendor: KVM`, full virtualization).
- **ISA:** full AVX-512 (`avx512f/dq/bw/vl/vnni/vbmi/vbmi2/bf16/fp16/ifma/bitalg/vpopcntdq`) plus
  **AMX** (`amx_tile`, `amx_int8`, `amx_bf16`) and `avx_vnni`. Phase 3's AVX-512 tier and §9's AMX
  follow-on are both hardware-supported here.
- **NUMA:** 6 nodes (SNC3 — 3 sub-NUMA domains per socket). 64 logical CPUs + ~507 GB RAM per
  node; socket 0 = nodes 0–2, socket 1 = nodes 3–5. `node distances`: 10 local, 15–17
  intra-socket, 21–28 cross-socket.

  ```
  NUMA node0 CPU(s):   0-31,192-223      node 0 size: 507306 MB
  NUMA node1 CPU(s):   32-63,224-255     node 1 size: 507905 MB
  NUMA node2 CPU(s):   64-95,256-287     node 2 size: 507951 MB
  NUMA node3 CPU(s):   96-127,288-319    node 3 size: 507951 MB
  NUMA node4 CPU(s):   128-159,320-351   node 4 size: 507951 MB
  NUMA node5 CPU(s):   160-191,352-383   node 5 size: 507926 MB
  node distances:
  node     0    1    2    3    4    5
     0:   10   15   17   21   28   26
     1:   15   10   15   23   26   23
     2:   17   15   10   26   23   21
     3:   21   28   26   10   15   17
     4:   23   26   23   15   10   15
     5:   26   23   21   17   15   10
  ```

- **RAM:** 2.9 TiB total. A live free-memory imbalance was visible while a run was resident
  (node 2: 19 GB free vs node 0: 466 GB) — first-touch concentration observed in the wild, exactly
  what Phase 6 targets. 1.32 TiB of expert slots cannot fit in one 507 GB node, so without
  deliberate placement allocations spill node-to-node in load order.
- **Kernel:** `5.14.0-687.15.1.el9_8.x86_64` (Rocky Linux 9.8 guest). **THP:** `[always] madvise
  never` (system default is `always`).
- **Memory bandwidth** (hand-rolled OpenMP triad `a=b+scale*c` and a read-only sum, f64, 8 GiB
  arrays, `gcc -O3 -march=native -fopenmp`, run under `numactl`; not canonical McCalpin STREAM —
  a ballpark instrument, recorded as such):
  - read-only, `--interleave=all`, 384 threads: **~620 GB/s**
  - read-only, single socket (`--cpunodebind=0-2 --membind=0-2`), 192 threads: ~545 GB/s
  - triad (write-allocating), `--interleave=all`, 384 threads: ~342 GB/s (RFO write traffic caps it)

  Decode streams read-only expert weights, so the ~620 GB/s read figure sets the floor:
  25.8 GB/token ÷ 620 GB/s ≈ **0.042 s/token** — inside the brief's 0.025–0.06 s/token estimate.
  **This is a floor, not a promise** (reaching it needs the §9 follow-ons).

### 0b. Baseline (real checkpoint, `main` @ `5853baa`, before the build-profile change)

Command: `teacher_forced_decode_bench --model /data/hf/hub/kimi-k3 --steps 12 --expert-cache 896`
(the canonical command at a reduced 12 steps — one real-checkpoint run costs ~13 min to load +
~7 min to decode, so the exploratory sweeps below are deferred to the synthetic fixture; see 0c).
`RUSTFLAGS` unset, generic x86-64 baseline, default thread count (192 physical), THP `always`,
no `numactl`. Single process, page cache warm from prior owner runs.

| stage | number |
|---|---|
| model load | **773.0 s** (~13 min; dense-weight transcode-dominated) |
| prefill (6-token prompt) | 192.2 s |
| decode, 12 steps | **441.2 s = 36.8 s/token average** |
| decode, steady state (steps 7–12) | **≈ 34 s/token** |

Expert cache at cache-896 (no eviction): hits+misses grow by exactly 1472/step (16 experts × 92
MoE layers). By step 12: 13595 hits / 10250 misses — 80% hit rate, still ~289 single-threaded
expert loads/step, so 34 s/token is *compute + residual loads*, not pure compute. This ~34 s/token
is the "before" anchor Phases 2/3/5 must crush; per the brief, decode is dominated by the scalar
MXFP4 matmul (§3 K1), which at ~8 GFLOP/s aggregate across 192 cores is leaving ~40× on the table.

### 0c. Zero-code experiments — status

The brief's 0c sweep (threads 48/96/192/384, `numactl --interleave=all` vs default, single-socket
control, THP `always` vs `madvise`, `RUSTFLAGS=-C target-cpu=native`) is **deferred to run on the
Phase 1 synthetic fixture**, which the brief explicitly sanctions ("substitute the synthetic dir
for iteration"). Rationale recorded honestly: each real-checkpoint run is a ~20-minute round trip
dominated by a 13-minute load that tells us nothing about decode scaling, so running 8+ of them
back-to-back is ~2.5 hours of mostly-load. The synthetic fixture (real per-layer widths, 6 layers)
loads in a fraction of that and reproduces the thread-scaling and kernel behaviour these sweeps
probe. **The one experiment that genuinely needs real 1.32 TiB scale — `interleave=all` vs default,
which gates Phase 6 — is run once on the real checkpoint at the final milestone**, where its result
matters most. `target-cpu=native`'s effect on the kernels is measured cheaply via `cargo bench`
(Phases 2–3) rather than a full decode run. All results land in the phase sections that produce them.

### 0d. Build profile

Added `[profile.release] lto = "thin", codegen-units = 1` to `Cargo.toml` (was: no
`[profile.release]` at all → no LTO, 16 codegen units). Purpose: let the optimizer inline across
the `is_x86_feature_detected!` dispatchers into their `#[target_feature]` inner kernels. **Not**
added: `panic = "abort"` (the `--serve` accept loop unwinds panicked request threads) and
committed `target-cpu=native` (decision D1: no — keeps binaries portable; measured via `RUSTFLAGS`
instead). No behaviour change beyond codegen.

**Gate:** topology + bandwidth + baseline recorded above; `cargo test` green (oracle fixtures:
state whether they ran or skipped in the commit); build-profile is the only code change.

## Phase 1 — synthetic K3-at-scale fixture (2026-07-31)

`src/bin/gen_k3_synth.rs` writes a **loadable, random-weight K3 checkpoint at the real per-layer
widths** but few layers, so kernel/parallelism/NUMA iteration costs ~90 GB and a ~90 s load
instead of 1.56 TB and ~13 min. Weights are garbage (this is a performance proxy, never a
correctness one — that stays `tests/teacher_forcing_k3.rs`'s job); routed experts are emitted as
raw MXFP4 `.weight_packed`/`.weight_scale` byte pairs with the exact counts `qt_load_mxfp4` checks,
scale bytes pinned to [120,130] so magnitudes stay finite. One `*.safetensors` shard per layer,
K3's real `language_model.` tensor names, config with the real dims and a 3:1 KDA:full-MLA pattern.

Generated: `gen_k3_synth --out /data/k3-synth6 --layers 6 --experts 896` → **88.6 GiB in 185 s**
(layer 0 KDA/dense, layer 2 MLA/MoE, rest KDA/MoE).

**Gate (k3_smoke, `--expert-cache 896`, Phase-0 binary, real per-layer widths):**

| | synthetic (6 layers) | real (93 layers) | ratio |
|---|---|---|---|
| model load | **93.2 s** | 773 s | ~8× faster |
| decode, steady state | **~2.0 s/token** | ~34 s/token | 0.059 |

The brief's extrapolation `(6/92) × real ≈ 2.2 s/token` predicts ~2.2 s; measured ~2.0 s — within
noise, so the fixture faithfully reproduces per-token decode behaviour and is a valid stand-in for
the later phases' iteration (and the deferred 0c sweeps). RSS after decode: ~7 GiB (dense resident
+ warmed experts), confirming `pread` streaming, not mmap.

## Phase 2 — scalar MXFP4 kernel hygiene, bit-exact (2026-07-31)

`matmul_mxfp4`'s inner loop restructured to iterate the row's 32-element scale blocks:
`e8m0_decode` (a `2^(byte-127)` `powi`) is now evaluated **once per block** instead of once per
element (the old loop recomputed it for all 32 elements sharing a scale byte), and each packed
byte's two nibbles are decoded together, dropping the per-element `k & 1` branch. Per-element
arithmetic `x*e2m1*scale` and accumulation order are unchanged → **bit-identical** output, pinned
by `matmul_mxfp4_matches_the_pre_block_reference` (compares raw f32 bits against a straight port of
the old loop, across dims incl. `i` not a multiple of 32/64 and odd `o`).

Bench: `cargo bench --bench kernels -- matmul_mxfp4`, before via `git worktree` at the Phase-1
commit with the new bench file copied in, after at the Phase-2 tip. Real per-expert dims, `RUSTFLAGS`
unset, 384 threads (rayon default), on the target box.

| case (i×o) | before (old scalar) | after (new scalar) | speedup |
|---|---|---|---|
| gate/up 3584×3072, **s=1 (decode)** | 4.45 ms | 4.19 ms | 1.06× |
| gate/up 3584×3072, s=8 | 8.41 ms | 4.15 ms | **2.03×** |
| down 3072×3584, **s=1 (decode)** | 4.74 ms | 4.04 ms | 1.17× |
| down 3072×3584, s=8 | 8.36 ms | 4.18 ms | **2.00×** |

**Honest read (the brief asked for it, "including less than hoped"):** removing the redundant
`e8m0` `powi` is a clean **2× at s=8**, but only **~6–17% at s=1**, which is the batch-1 decode
case (each expert sees one token's row). At s=1 a single [3072,3584] matmul is only ~5.5 MB of
weight bytes touched in ~4 ms (~1.4 GB/s, far below the 620 GB/s floor) — so it is dominated by
rayon fork/join dispatch and memory latency across 384 threads on ~8 rows each, not by the arithmetic
Phase 2 trimmed. The decode kernel's real win is Phase 3's AVX-512 tier (and Phase 5 attacks the
fork/join count). Sanity check that the bench is representative: 92 MoE layers × 16 experts × 3
matmuls × ~4 ms ≈ 17.7 s/token, matching the ~21 s scalar-compute component of the Phase 0 baseline.

*Tried and didn't help / notably:* nothing reverted this phase. Note s=1 vs s=8 barely differing in
the NEW kernel (4.19 vs 4.15 ms) is the fork/join-dominated regime showing through — flagged for
Phase 5.

## Phase 3 — AVX-512 tier for matmul_mxfp4 (2026-07-31)

`matmul_mxfp4` is now a dispatcher (AVX-512F/BW > scalar, same ladder as `matmul_i4`).
`matmul_mxfp4_avx512` / `dot_mxfp4_f32_avx512`: one 32-element E8M0 block = 16 packed bytes = two
16-lane f32 vectors; the 16 E2M1 code points fit one `zmm` addressed by `_mm512_permutexvar_ps`
(low 4 bits of each index = the nibble, sign included). Per block: unpack low/high nibbles (same
interleave as `dot_i4_f32_avx512`), gather E2M1 via permute, fold the block's scalar-decoded E8M0
scale in, FMA into two accumulator chains reduced by one tree-sum; `cols % 32` tail scalar.
**Within-tolerance, not bit-exact** (scale-fold + reassociated reduction) — new parity test
`matmul_mxfp4_avx512_matches_scalar_within_tolerance` (relative-with-floor 2e-3 across edge dims).
The K8 `matmul_qt_mxfp4_matches_manual_dequant_dot_product` test was relaxed from `assert_eq!` to
the same within-tolerance form its **int4** sibling has always used (matmul_qt now dispatches the
MxFp4 path to the reassociating AVX-512 tier).

**The kernel is 3–5× faster — but end-to-end decode barely moves. This is the phase's key finding.**

Kernel, single-thread (`RAYON_NUM_THREADS=1`, isolates arithmetic), real per-expert dims:

| case | scalar | avx512 | speedup |
|---|---|---|---|
| gate/up 3584×3072, s=1 | 49.6 ms | 9.84 ms | **5.0×** |
| down 3072×3584, s=8 | 144.5 ms | 48.3 ms | **3.0×** |

Same matmul at the **default 384 threads** (rayon), the real decode configuration:

| case | scalar | avx512 | speedup |
|---|---|---|---|
| gate/up s=1 | 4.82 ms | 4.97 ms | **~1.0× (none)** |
| down s=8 | 5.03 ms | 4.88 ms | ~1.0× |

End-to-end synth decode (k3_smoke, `--expert-cache 896`): **~2.0 → ~1.9 s/token (~5%)**.

**Why (measured, not assumed):** at s=1 the matmul forks `par_chunks_mut(1)` = ~3072 one-row
micro-tasks across 384 threads. Single-thread scalar is 49.6 ms; at 384 threads it is 4.82 ms —
only ~10× from 384 cores (~2.6% parallel efficiency). The ~4.8 ms is a **rayon fork/join +
task-scheduling floor**, hit by both tiers: the AVX-512 kernel is so much faster that there is
almost no compute left for the 384 threads to divide, so its win is entirely masked. Decode's
MXFP4 matmul is **scheduling-overhead-bound, not compute-bound** — the 8 GFLOP/s-aggregate gap the
Phase 0 baseline noted is fork/join overhead from per-matmul 384-way parallelism over tiny work,
not the kernel's scalar-ness. **The AVX-512 tier is the necessary compute foundation; Phase 5
(across-expert parallelism, which collapses ~4400 fork/join regions/token to ~92) is the actual
decode unlock — where this 3–5× kernel finally shows.** Real-checkpoint coherence + decode delta
are folded into the post-Phase-5 milestone (synth weights are garbage, so coherence can't be
checked here, and Phase 3 alone doesn't move real decode either — same scheduling floor).

*Tried and didn't help (this phase, in isolation):* the AVX-512 matmul tier for end-to-end decode
— correct and 3–5× at the kernel, but ~0% end-to-end at 384-way per-matmul parallelism. Kept
because it is the foundation Phase 5 needs, not reverted. Recorded so nobody concludes "AVX-512
didn't help MXFP4" without the Phase 5 context.

## Phase 4 — expert loading: parallel misses, preload, honest logging (2026-08-01)

**4a — parallel `sequential_fallback`.** `misses.iter().map(load_expert)` → `misses.par_iter()...`.
rayon's *ordered* collect preserves insertion (miss) order, so cache insertion/pin order and
determinism are byte-for-byte unchanged; `Shards` reads are `&self` `pread` (thread-safe, never
mmap). This is the only load path K3's MXFP4 experts ever take (no io_uring ring).

**4b — `--preload-experts`.** New CLI flag threaded through `LoadArgs` into `load_session`; after
warm-start it loads every MoE layer's experts up front via a new `model::ExpertCaches::preload`
(per-family arms — all three families) over the shared `expert_cache::preload_layers`. At
`capacity >= n_experts` loads all (full residency); at `capacity < n_experts` fills to capacity in
usage-histogram order when `.rabbit_usage` exists else expert-id order (D2), logged. One progress
line per layer. Applies to `--serve` too (same `Session` path).

**4c — honest logging.** `ExpertCache::has_ring()` / `ExpertCaches::any_has_ring()`: when no ring
(MXFP4/K3), the CLI prints `[disk wait n/a]` instead of the structurally-always-`[0.0s actual disk
wait]` bracket that misled the owner in the first minutes of the first real K3 run.

**4d — `--drop-os-cache`: deferred.** Default-off, the least impactful sub-item (the OS reclaims
page cache under pressure anyway per the brief), and its `drop_cache` bool would have to thread
through the shared `begin_loading`/`sequential_fallback`/`load_expert` path used by all families
and the io_uring branch. Deliberately left out of this pass rather than churn that hot path for a
default-off flag; noted here so it's a known gap, not an oversight.

**Measured (synth fixture, `rabbit --model /data/k3-synth6 --preload-experts --expert-cache 896`):**

| | parallel (192 threads) | sequential-equiv (`RAYON_NUM_THREADS=1`) |
|---|---|---|
| preload 4480 experts (~73 GB) | **26.3 s** | 23.9 s |

**Honest 4a finding:** parallel ≈ sequential here — the preload is **disk-bandwidth-bound** (~73 GB
over NVMe at ~3 GB/s ≈ 24 s), not per-expert-CPU-bound, so 384-way parallelism has nothing to
divide. 4a is expected to pay off in the **warm** regime the target box actually runs in — the real
1.56 TB checkpoint is largely page-cache-resident (node 2 was full at baseline), so mid-decode
expert loads are memory-bandwidth memcpy where parallelism helps, not cold disk reads. Recorded as
"measured, no win in the disk-bound case." **4b clearly works regardless:** with `--preload-experts`
the expert cache reaches full residency (4480/4480, 0 further misses) and **token 1 runs at
steady-state 1.1 s with no warmup ramp** (vs ~2.3 s ramping down without it). **4c confirmed:** the
log now reads `... in disk I/O [disk wait n/a]`, not the misleading `[0.0s actual disk wait]`.

*Tried and didn't help:* parallelizing the expert loads (4a) on a cold, disk-bound fixture — no
speedup because the storage bandwidth, not the loader, is the ceiling there. Kept (correct, and a
win in the warm/page-cache regime), not reverted.

## Phase 5 — across-expert parallelism in latent_moe (2026-08-01)

`latent_moe`'s apply loop no longer runs the chunk's experts one after another (each expert's
three matmuls fanning out over all cores and joining — ~48 fork/join regions per MoE layer). It now
collects the chunk's resident `&ExpertSlot`s, runs each expert's full gate→activation→down chain
into its **own zeroed `s×moe_hidden` buffer in parallel** (`slots.par_iter()`), then reduces the
buffers into `routed` **sequentially in chunk order**. Per-expert math and the across-expert
accumulation order are byte-for-byte the old loop's (each buffer is nonzero only at its expert's
token rows, so adding a whole buffer only adds `+0.0` elsewhere), so the output is **bit-identical**
— pinned by the existing naive-reference tests plus a new `latent_moe_is_bit_identical_across_two_warm_runs`
determinism test (the reduction is fixed-order, not completion-order, so there is no run-to-run
variance).

**Borrow note:** the brief's K5 said `cache.get` takes `&mut self` (LRU stamp) and a `peek`
accessor might be needed. At this commit `get` is already `&self` — LRU stamping happens in
`begin_loading` (inside `ensure_loaded`), which runs before the parallel region — so the slots are
collected with plain immutable `get` and no new accessor.

**Required infra fix (`kernels.rs::with_yt_scratch`):** the matmuls' thread-local `yt` scratch
assumed single-level parallelism ("rayon workers never call back into `matmul_*`"). Phase 5 nests
an outer per-expert `par_iter` around the matmuls' inner `par_chunks_mut`, so a worker thread that
work-steals a second expert's matmul while holding its own borrow hit `RefCell already borrowed`.
Fixed additively: the reentrant (inner) call falls back to a one-shot heap buffer; the non-nested
GLM/Kimi path still hits the zero-alloc thread-local fast path unchanged (verified: `matmul_i4`
scalar 5.03 ms / avx512 4.23 ms, normal — GLM's hot path is structurally untouched, only
`kimi_k3::moe` changed).

**Measured (synth, k3_smoke `--expert-cache 896`, warm):**

| | Phase 4 (sequential experts) | Phase 5 (parallel experts) | speedup |
|---|---|---|---|
| decode, steady state | ~1.9 s/token | **~1.0 s/token** | **1.9×** |
| prefill (8 tokens) | 9.6 s | **4.77 s** | 2.0× |

**v2 not built (brief gates it on profiling evidence):** v1 still runs each expert's matmuls with
the inner per-matmul `par_chunks_mut`, so the fork/join floor Phase 3 identified is only *overlapped*
across experts, not removed — the ~1.9× comes from that overlap, not from the AVX-512 kernel's full
3–5× finally showing. The evidence that cores are still underused (Phase 3's fork/join finding +
this residual inner fork) points at v2 — flatten to (expert × row-block) tasks with row-range kernel
entry points, one fork per layer, each task doing real serial AVX-512 compute — as the next lever
toward the memory-bandwidth floor. Deliberately left for a measured follow-up rather than shipped
speculatively; this phase delivers the bit-identical overlap win first.

## Real-checkpoint milestone — Phases 2–5 combined (2026-08-01)

The exact Phase 0 baseline command on the real 2.8T checkpoint, all of Phases 2–5 in place:
`teacher_forced_decode_bench --model /data/hf/hub/kimi-k3 --steps 12 --expert-cache 896`,
`RUSTFLAGS` unset, 192 threads, no `numactl`, no preload (loads experts during decode, exactly
like the baseline).

| metric | Phase 0 baseline | Phases 2–5 | speedup |
|---|---|---|---|
| model load | 773.0 s | 774.6 s | unchanged (the kernel/MoE work doesn't touch load) |
| **prefill (6-token prompt)** | 192.2 s | **84.8 s** | **2.27×** |
| **decode, 12 steps** | 441.2 s = 36.8 s/token | **202.4 s = 16.9 s/token** | **2.18×** |
| decode, steady state | ~34 s/token | **~16.5 s/token** | ~2.1× |

**Apples-to-apples, pure compute win:** step-12 cache totals are 13608 hits / 10239 misses vs the
baseline's 13595 / 10250 — essentially identical routing and expert-loading (same misses ⇒ same
disk/memory traffic), so the ~2× is entirely the compute the kernel (Phase 3) and across-expert
parallelism (Phase 5) sped up, not a caching artifact. Correctness holds (Phase 2 bit-identical,
Phase 3 within-tolerance, Phase 5 bit-identical); the run produced its teacher-forced sequence with
no NaN/inf and the same routing as baseline.

**Where the remaining headroom is:** 16.5 s/token is still far above the ~0.042 s/token
memory-bandwidth floor. Two levers remain, both measured below: (1) **NUMA placement** — this
milestone ran with default first-touch, which Phase 6 shows leaves ~1.4× on the table; and (2)
**Phase 5 v2** (expert × row-block tasks, one fork/layer, serial AVX-512 per task) to remove the
inner per-matmul fork/join Phase 3 identified, so the kernel's full 3–5× shows end-to-end. This run
used neither `numactl` nor v2, so ~16.5 s/token is a conservative floor for what Phases 2–5 already
enable.

## Phase 6 — NUMA (conditional): TRIGGERED — `numactl --interleave=all` (2026-08-01)

The brief gates Phase 6 on the interleave experiment moving the number materially. **It does** — and
the intuition that "compute-bound ⇒ NUMA doesn't matter" was wrong, which is exactly why the brief
says measure. The compute *is* streaming 25.8 GB of expert weights from memory per token; default
first-touch concentrates those weights on whichever nodes the loading threads ran on (the live
imbalance captured at baseline — node 2 at 19 GB free vs node 0 at 466 GB), so 384 threads spread
across all 6 nodes contend on 1–2 memory controllers. Interleaving the pages across all nodes
removes that hotspot.

Measured on the synthetic fixture (k3_smoke, `--expert-cache 896 --prompt-len 8 --max-tokens 6`, warm):

| | decode per token | notes |
|---|---|---|
| default (first-touch) | ~1.5 s (1.22–2.00, high variance) | weights clustered on a few nodes |
| **`numactl --interleave=all`** | **~1.07 s (1.03–1.13, tight)** | **~1.4× faster, variance collapses** |

**Recommendation (zero code — the brief's preferred Phase 6 outcome): run with `numactl
--interleave=all`.** Real-scale (1.32 TiB spanning all 6 nodes) can only make the default hotspotting
worse, so the real checkpoint likely benefits at least as much — the milestone above (default
placement) would land lower under interleave. No `madvise`/`mbind` code was needed: interleaving via
`numactl` gets the win with none of the allocator surgery, exactly the "documented `numactl` first"
ordering the brief asks for. Per-expert round-robin placement during preload (Phase 4b) is the
natural in-process equivalent if a code-level default is ever wanted, but it isn't needed to capture
the win today.

## Phase 5 v2 — row-blocked expert dispatch, one fan-out per stage (2026-08-01/02)

Provenance, stated plainly: this restructure was found complete-but-uncommitted in the working
tree at the start of the session that executed `NUMA_AMX_BRIEF.md` (built in a prior working
session; its comments referenced measurements that had never been recorded here). It was
re-verified from scratch — full test suite including the oracle teacher-forcing suites, which now
RUN on this box (fixtures present under `tests/oracle/`), plus the controlled numbers below — and
is committed as part of Phase N3, which the brief fuses it into.

What it is:

- `kernels.rs::par_rows` — every `matmul_*` now hands rayon ~4 row-block tasks per pool thread
  instead of ONE TASK PER OUTPUT ROW (the fork/join pathology Phase 3 diagnosed: thousands of
  one-dot-product tasks). Task granularity never changes a row's own accumulation order —
  bit-identity unaffected, all existing kernel tests unchanged.
- `kernels.rs::matmul_qt_rows` + `RowActs` — row-range entry points that compute `[r0, r1)` of a
  matmul serially through the SAME per-row kernel bodies the whole-matrix entry points use
  (factored `row_dot_*` helpers), so reassembling row blocks reproduces `matmul_qt` bit for bit —
  pinned across all six `QTKind`s by `matmul_qt_rows_reassembled_is_bit_identical_to_matmul_qt`.
- `kimi_k3/moe.rs` — `latent_moe`'s apply stage flattened to (expert × row-block) tasks: TWO
  fan-outs per chunk (gate/up + activation, then down + accumulate), no nested rayon. Stage B
  parallelizes over OUTPUT ROWS, walking experts in chunk order per row — the float addition
  order is exactly the sequential loop's, so output is bit-identical (test:
  `latent_moe_row_blocked_dispatch_is_bit_identical_to_applying_experts_one_at_a_time`).
- `glm52/moe.rs` — visibility-only (`expert_rows` extracted; `Activation::{combine,apply}`
  `pub(crate)`). `glm52::moe::moe()` itself untouched (scope rule 4.7).

**Measured, worktree v1 (`ef8a40d`) vs v2, synth fixture, canonical command at `--steps 30
--expert-cache 896`, warm, RUSTFLAGS unset, THP `[always]`, default placement:**

| threads | v1 | v2 | change |
|---|---|---|---|
| 48 | 3.03 s (0.101 s/token) | 2.88–3.16 s (~0.10 s/token) | **~none** |
| 96 | 5.57 s | **2.48 s (0.083 s/token)** | **2.2×** |
| 384 | 27.50 s | 12.87 s | 2.1× |

**Honest read:** v2 does NOT speed up the 48-thread configuration the serve baseline runs at —
it removes the *negative* thread scaling. Under v1 every added thread past ~48 made decode
slower; under v2 the optimum moves to ~96 threads (0.083 s/token, 22% faster than any v1
configuration) and the 384-thread cliff drops from 9× to 4×. The remaining high-thread-count
degradation is NOT the expert path (see N3's finding below — it is attention-side, Phase N5's
target). GLM/Kimi-Linear shared kernels: `matmul_i4` bench after the coarsening runs 3.52–3.71 ms
vs the 4.23–5.03 ms recorded at Phase 5 — improved, not regressed.

## Phase N0 (NUMA_AMX_BRIEF) — go/no-go probes (2026-08-01)

Everything from here down executes `NUMA_AMX_BRIEF.md` rev 2 (which fuses `K3_OPTIMIZE_BRIEF.md`
Phase 5's deferred v2 into its Phase N3). Box re-verified per the brief's provenance caveat:
`lscpu` + `numactl --hardware` re-captured on the live instance and **identical to Phase 0a's
record** (6 nodes SNC3, 64 logical CPUs + ~507 GB each, distances 10 / 15–17 / 21–28, AMX
present, KVM). Kernel `5.14.0-687.15.1.el9_8.x86_64`, THP `[always]`, box idle, ~1.4 TiB of the
checkpoint warm in page cache.

**N0a — is the guest NUMA topology real? YES (gate passed).** Hand-rolled OpenMP triad + read
probe (`gcc -O3 -march=native -fopenmp`, 2 GiB f64 arrays — same ballpark-instrument genre as
Phase 0a, not canonical STREAM), `numactl --cpunodebind=0 --membind={0,2,3}`:

| placement | triad | read-only |
|---|---|---|
| local (mem 0) | 153.7 GB/s | 183.0 GB/s |
| intra-socket remote (mem 2) | 141.0 GB/s | 175.8 GB/s |
| cross-socket remote (mem 3) | 79.7 GB/s | 87.1 GB/s |

Local : cross-socket = **2.10× read** (1.93× triad) — ≥2×, the guest topology has real teeth; GO.
The surprise worth keeping: **intra-socket-remote is nearly free (−4%)** — SNC3 sub-node
boundaries barely cost anything; the socket boundary is the wall. (Feeds D1: per-node pools
already capture almost everything per-socket pools would.)

**N0b** is folded into the Phase 5 v2 / N3 sections below — the sensitivity matrix was run on
the v2 tree since that is what N3 ships on.

## Phase N1 — `src/numa.rs`: topology + placement primitives (2026-08-01)

`topology()` (parses `/sys/devices/system/node/*/cpulist`; pure parser, unit-tested),
`pin_current_thread` (`sched_setaffinity`), `bind_region`/`interleave_region` (raw `SYS_mbind` —
no glibc wrapper exists), `node_of_page` (`SYS_get_mempolicy`, the test instrument). libc-only,
no new dependencies, no-op stub twin for non-Linux. `--numa` parsed and deliberately inert (N1
gate). The placement smoke tests run REAL on this box and caught two genuine hazards:

- **`mbind` demands a page-aligned start** and heap pointers aren't (glibc's mmap'd chunks carry
  a 16-byte header) — the wrappers page-align internally, leaving sliver pages on the default
  policy (noise at multi-MB buffer sizes).
- **Memory policy is inherited per-thread from the creator** — under the canonical
  `numactl --interleave=all` launch every pool worker would start `MPOL_INTERLEAVE`, and first
  touch would spray each expert's pages across all six nodes, silently defeating the entire
  home-node scheme. `pin_current_thread` therefore also resets the calling thread's policy to
  default (first-touch-local); a dedicated test reproduces the inherited-interleave condition
  and asserts pages still land on the pinned node. Without this, `--numa` under the standard
  serve launch would have been a placement no-op that *looked* like "NUMA doesn't help".

## Phase N2 — pinned per-node pools (2026-08-01)

`numa::NodePools`: one pinned rayon pool per node, each sized `total/n_nodes` from the effective
`--threads` total (hardcoding cores/node would silently override the owner's thread sweep), built
once when `--numa` is active on a real multi-node box. `run_all(f)` is the single cross-pool
fan-out primitive — orchestrated from short-lived non-pool threads, which structurally rules out
the install-from-another-pool deadlock (brief trap #2).

Gate bench (`examples/numa_pool_bench.rs`: `[32768 × 7168]` int4 matvec at s=1 — N4a's
`matmul_qt_sharded` prototype; min / median of 30, RUSTFLAGS unset):

| total threads | global pool | per-node sharded (6) | per-socket sharded (2) |
|---|---|---|---|
| 48 | 0.739 / 0.941 ms | 0.904 / 0.998 ms | 0.931 / 1.197 ms |
| 192 | 1.391 / 1.955 ms | 0.847 / 0.973 ms | 0.604 / 1.136 ms |

At 48 threads the three configurations tie; at 192 the global pool **degrades ~2×** while the
pinned pools hold steady — pinning is what makes high thread counts usable at all. Per-socket ≈
per-node with more variance → **D1: per-node**, revisit only if N3d's skew data argues otherwise.

## Phase N3 — expert home nodes, fused with Phase 5 v2 (2026-08-01/02)

- **N3a** `numa::home_node(layer, eid) = (layer × n_experts + eid) mod n_nodes` — pure function,
  the single agreement point between placement and dispatch.
- **N3b placement**: `expert_cache::sequential_fallback` → `numa_homed_load` when `--numa` is
  active: each missing expert loads INSIDE its home node's pinned pool, so allocation + `pread`
  fill (= first touch of every page) happen on the node that will compute it. Zero `mbind`
  calls. Miss-order results preserved → cache insertion order, pin promotion, determinism all
  unchanged. Covers preload and mid-decode misses alike.
- **N3c dispatch**: `latent_moe` → `dispatch_numa`: ONE `run_all` fan-out per chunk; each node
  runs its own experts' gate→activation→down (same shared `gate_up_block` task body as the
  global path, row-block-parallel within the pool), and the routing-weighted accumulation moves
  to a sequential chunk-order scatter afterward — same additions per output element in the same
  order as the `--numa`-off path, so **bit-identical by construction** (unit test against real
  6-node pools: `numa_dispatch_is_bit_identical_to_the_global_pool_dispatch`).
- **N3d**: cumulative per-node busy/expert counters, served as `numa_moe` in `GET /profile`.
- `--numa-threads N` decouples the pinned-pool total from `--threads` — measured below, the two
  pools want very different widths.

**Bit-identity gate: PASS everywhere.** The bench now folds every decode step's full logits into
an FNV-1a fingerprint. Synth: `af9ee5068abb90b0` across all 13 runs (on/off × 48/96/192/384,
coupled and decoupled). Real checkpoint: `60c0f42c265f66a5` across all 4 runs (on/off × 48/96).
Same routing (identical hit/miss totals per step), same logits, only scheduling differs.

**Synth timings** (canonical command `--steps 30 --expert-cache 896`, warm, default policy):
coupled `--numa` LOSES at every width (48: 4.25 vs 3.03 off; 192: 5.78; 384: 13.35) — but
**numa@384 ≈ off@384 (13.35 vs 12.87)**, which acquits the expert dispatch: the high-thread
collapse is attention + dense matmuls degrading with GLOBAL pool width in both configurations
(measured ~10 s of the 13 s at 384). That reframes the brief's thread-sweep criterion — the
expert bucket and the attention bucket want different thread counts, hence `--numa-threads`.
Decoupled (global 48 for attention, pinned N for experts): p96 2.95 s, **p192 2.54 s**, p384
2.64 s — vs off@48 3.03 s and off@96 2.48 s. On synth, NUMA reaches parity with the best global
configuration, no more: synth decode moves ~1.3 GB/token at ~0.08 s/token ≈ 16 GB/s, nowhere
near bandwidth-bound, so home-node locality has nothing to win there and its costs (routing skew
— a node draws up to ~5 of 16 experts while others idle; cross-pool orchestration) show. Recorded
as expected behavior, not failure: the bandwidth case is the real checkpoint at 1.32 TiB.

**Real checkpoint** (`--steps 12 --expert-cache 896`, no preload, warm page cache, no `numactl`
prefix, steady state = steps 7–12):

| config | 12-step total | steady s/token |
|---|---|---|
| off, 48 threads | 31.06 s | 2.43 |
| off, 96 threads | 32.52 s | 2.55 |
| **--numa, global 48 / pinned 384** | **28.31 s** | **2.30** |
| --numa, global 96 / pinned 384 | 29.64 s | 2.35 |

NUMA-decoupled wins every pairing, but modestly (~6–9%) — this command mixes ~300 mid-decode
expert loads into every step. The full-residency serve measurement below is the clean gate.

**Serve gate** (canonical launch + `--numa --numa-threads 384`, i.e. `numactl --interleave=all
... --serve --expert-cache 896 --no-usage-cache --preload-experts --threads 48 --numa
--numa-threads 384`; 128-completion-token turn, 41-token prompt, `/profile`, misses frozen at
82,432 = pure compute):

| | rev-2 baseline (v1, 48 thr) | v2 + `--numa` g48/p384 |
|---|---|---|
| startup: preload | ~30 min | **687.6 s (~11.5 min, 2.6×)** |
| wall | 1.42 s/token | 1.51 s/token |
| expert bucket | 0.83 s/token | **0.90 s/token — GATE FAILED** (target < 0.3) |
| attention bucket | 0.53 s/token | 0.56 s/token |

**The failure is precisely diagnosed by N3d's own counters, and it is NOT the placement or the
compute.** Per-node busy time for the turn: 31.8–36.9 s over 128 tokens — cumulative max-node
busy is **0.288 s/token, already at the gate target**, and the expert→node spread is balanced
(±8%). The 0.6 s/token gap between that and the 0.90 wall decomposes as roughly:

- **~0.2 s/token routing skew** — per-LAYER max-node load (~4.5–5 of 16 experts) vs the 2.7
  mean. The brief's "skew averages out over 92 layers" (§2) is true of cumulative totals and
  false of wall time, which sums per-layer *maxima*. Measured, this argues for flipping **D1 to
  per-socket pools**: 16 experts over 2 sockets has a max/mean of ~1.19 vs ~1.7 over 6 nodes,
  and N0a measured intra-socket-remote reads at only −4% vs local — most of the skew cost
  bought back for almost no locality cost.
- **~0.18 s/token cross-pool fan-out latency** — measured directly (`numa_pool_bench`'s
  `run_all` micro-bench): a no-op `run_all` costs ~0.4 ms, but with real work at 64 threads/node
  the median is ~2.0 ms — a sleeping pool's wake cascade, paid once per MoE layer (92/token).
  Fix candidates: persistent per-pool feeder threads (kills the 6 OS-thread spawns/layer, ~0.4
  ms) and keeping pools from sleeping between layers (harder; rayon has no public spin config).
- remainder: latent down/up matmuls + routing on the (deliberately narrow) 48-thread global
  pool, sequential scatter, gather.

**Control run — plain v2, no `--numa`, `--threads 96` (the synth optimum), same launch
otherwise, same prompt:** wall 1.49 s/token, expert **0.73** (better than the 0.83 baseline),
attention **0.71** (much worse than 0.53 at 48 threads). Preload: 425.5 s (page cache warmer
than the numa run's 687.6 s — both crush the ~30 min baseline; the parallel-loading fix works
under either scheduler).

The three serve configurations side by side (per token, same 128-token turn shape):

| | v1 @48 (rev-2 baseline) | v2+numa g48/p384 | v2 @96 |
|---|---|---|---|
| wall | **1.42** | 1.51 | 1.49 |
| expert | 0.83 | 0.90 | **0.73** |
| attention | **0.53** | 0.56 | 0.71 |
| preload | ~30 min | 11.5 min | 7 min |

Bottom line for this phase, stated honestly: **placement works (balanced nodes, preload 3–4×
faster, bit-identical logits everywhere), the per-node compute hits the gate number (0.29
s/token max-node busy), but the dispatch orchestration re-introduces a fork/join floor at the
cross-pool level and the serve gate FAILS (expert 0.90 vs < 0.3 target).** No configuration
moved the serve WALL, because the expert and attention buckets want opposite thread counts:
experts scale past 48 (0.83→0.73@96, 0.29 achievable pinned), attention degrades past 48
(0.53→0.71@96) — per-token wall is now genuinely attention-gated as much as expert-gated. Per
the brief's own rule (stop-for-owner-review after N3's gate; "if it doesn't move the number,
stop and re-profile before building N4"), work STOPS here for owner review. Decisions on the
table: (a) flip D1 to per-socket pools (skew 1.7→1.19 at −4% locality) + persistent feeders
(−0.4 ms/layer) and re-gate N3; (b) run serving as plain v2 `--threads 48` (baseline wall,
none of the NUMA machinery) until N5 lands; (c) prioritize N5 — attention at 0.53–0.71 s/token
at near-zero context is scheduling cost, the same disease in the other bucket, and is now the
larger half of the token.

## D1 flip — per-socket pools, flattened domain dispatch (2026-08-02, owner-approved)

Owner decision after the N3 review: flip D1 to per-socket pools; expected expert ~0.6 s/token
(<0.3 needs the pool-wake residual, spun off as its own timeboxed investigation below); **gate
the flip on synth + the cheap real bench only — the serve re-gate is deliberately deferred into
one combined boot after N5** (deviation from re-gating each phase at the server, recorded here).

What changed:
- `Topology::socket_domains` groups nodes by the kernel distance matrix (<20 = same socket;
  identity fallback), and `NodePools::init` builds per-socket domains. `home_node`, placement
  and dispatch all follow automatically (they key off `pools.n()`).
- `run_all` rebuilt as nested `in_place_scope`s (no OS threads, no `'static`): a no-op fan-out
  fell from ~0.4 ms to **0.012–0.029 ms**.
- **A trap worth its own line:** the first per-socket build ran each domain's experts
  sequentially, splitting ONE expert's rows across the whole 96–192-thread pool —
  `block_rows(3072, 192×4)` is a 4-row task, v2's micro-task pathology reborn, and it measured
  as a straight regression (synth g48/p192: 3.16 s vs per-node's 2.54). Fixed by flattening
  each domain's fan-out across ALL its experts (the same `gate_up_blocked` shape the global
  path uses, scoped to the domain) — g48/p192 back to 2.60 s. Lesson: task grain must be sized
  against the pool that runs it, wherever the dispatch happens.

Gate results (fingerprints unchanged everywhere — synth `af9ee5068abb90b0`, real
`60c0f42c265f66a5`):

| config | synth (30 steps) | real steady (steps 7–12) |
|---|---|---|
| off, 48 threads | 3.04 s | 2.57 s/token |
| **--numa g48 / p192** | 2.60 s | **2.03 s/token (−21% vs off, −12% vs per-node p384's 2.30)** |
| --numa g48 / p384 | 3.79 s | 3.28 s/token |

**Operating point: `--numa-threads 192` (96/socket).** Wider pinned pools regress until the
wake/distribution cost is fixed.

**"Pools stay hot" (timeboxed investigation — findings only, no fix built):** the
`numa_pool_bench` `run_all` micro-bench with a token's-worth of work, back-to-back vs after a
5 ms idle gap (decode's real shape — attention runs between MoE layers, pools sleep):
24/96/192 threads-per-pool cost 0.10/0.37/1.84 ms back-to-back and 0.40/0.76/2.94 ms after the
gap. Both components — sleep-wake (~0.3–1.1 ms) and task distribution (the rest) — scale with
pool width. At p192 that is ~0.07 s/token of overhead (acceptable); at p384 ~0.27 s/token
(prohibitive) — which is exactly why p384 regresses. Fix candidates for the follow-up: a
keep-warm spinner per pool (kills the wake share), coarser minimum task grain at wide pools,
two-level fan-out (socket → node subgroups). Not built in this timebox.

## Phase N5a — attention split, measured (2026-08-02)

`attention_s` now splits into `attn_kda_proj_s` / `attn_kda_recur_s` / `attn_mla_s`
(instrumentation only — timing calls around unchanged code in `kimi_k3::generate`; served in
`/profile` and printed per-token by the bench, which now runs `step_profiled`).

Real checkpoint, g48/p192 run above: **attention 0.380 s/token = KDA projections 0.274 + KDA
recurrence 0.050 + MLA 0.055.** The brief's N5b hypothesis — that the 96-head recurrence
`par_iter` is fork/join-floor-bound like the expert matmuls were — is REFUTED by the
measurement: the recurrence is 13% of the bucket and its floor-check estimate (~1–2 ms/token)
was closer to right than the fear was. **72% of attention is the eight per-token projection
matmuls per KDA layer** (q/k/v/o are ~[12288×7168]-class int4 matvecs — 552 small matmuls/token
on the global pool). Consequence, owner-visible: **N5b (head-batching the recurrence, ceiling
0.05 s/token) is descoped on this evidence**; the attention lever is the projections, which is
N4b/N5c's row-sharding — the next phase in the approved order anyway. Also worth recording:
in the p384 run the same buckets bloat (kda_proj 0.637, attention 0.849) — oversubscribed
pinned pools actively hurt the global pool's attention, another reason p192 is the operating
point.

## Phase N4 — dense-weight row-sharding: lm_head kept, KDA projections built-measured-reverted (2026-08-02)

`QTSharded` (kernels.rs): a dense weight split into one contiguous row-block `QT` per NUMA
domain, each COPIED inside its domain's pinned pool at load (first touch = placement — N3b's
trick applied to dense weights; `QT::copy_rows` is the mechanism). `matvec`/`matvec_sharded_batch`
run s=1 matvecs as ONE cross-domain fan-out with every domain writing its disjoint output rows
directly (at s=1 the transposed layout IS the output layout, so assembly is free). `DenseQT`
(`Plain`/`Sharded`) is the storage type for converted weights; everything loads `Plain` and a
post-load `Model::distribute_dense` shards iff `--numa` pools exist. **Bit-identity pinned at
the unit level across all six `QTKind`s** (single + batched, real pools, uneven row split), and
the end-to-end fingerprints never moved (synth `af9ee5068abb90b0`, real `60c0f42c265f66a5`).

What the fan-out-frequency arithmetic ruled out up front (recorded so nobody re-tries it
blind): the latent down/up (184 matmuls/token, whole cost ~48 ms/token) and the small KDA
projections can never repay a ~0.7 ms fan-out each. Candidates were lm_head (1/token) and the
big KDA q/k/v/o (N5a's 72%-of-attention target, 2 fan-outs/layer with q/k/v batched into one).

**Measured — lm_head: WIN, kept.** Synth per-token lm_head bucket 0.008 → 0.003 s (real
0.009 → 0.008; one fan-out per token is easily repaid by 6-domain-local reads of the
[163840×7168] weight).

**Measured — KDA q/k/v/o: REGRESSION on synth AND real, reverted to plain.** Synth kda_proj
0.018 → 0.027 s/token; real 0.346 → 0.443 s/token (12-step averages, same-fingerprint runs).
At 69 KDA layers × 2 fan-outs = 138 fan-outs/token, the measured ~0.7 ms wake/distribution
cost per fan-out (~0.10 s/token) plus per-fan-out latency jitter exceeds what domain-local
reads win back. This is the same verdict the "pools stay hot" timebox predicted: **the
fan-out cost, not placement, is the binding constraint on every high-frequency NUMA dispatch.**
The `DenseQT` fields, the batched q/k/v fan-out in `kda_step`, and `distribute_dense` all stay
in place as the ready mechanism — re-enabling is four `shard_in_place` lines + a re-gate once
pools-stay-hot lands. Post-revert sanity: synth `--numa` g48/p192 = **2.40 s / 30 steps —
the best synth configuration measured to date** (off@48 2.90–3.16, off@96 2.48).

**N4c (KV-cache interleave): deferred, with justification.** The MLA bucket is 0.055 s/token
at short context (N5a) — KV traffic is not yet a cost; `KvCache` is shared with GLM-5.2; and
its buffers grow by `extend` (growth reallocs silently drop an mbind VMA policy — trap #3), so
doing this properly means a reserve-at-capacity change to shared code. Worth doing when
long-context serving is the workload; not worth the shared-code risk for 0 measured win today.

**N5b (recurrence head-batching): descoped** on N5a's measurement (ceiling 0.05 s/token) —
recorded in the N5a section above. N5c (projections riding N4a's sharding) is subsumed by the
KDA measurement above: same verdict, same re-enable path.

## Combined boot — one serve re-gate for the flip + N5a + N4 together (2026-08-02)

The owner-approved deviation resolved: instead of re-gating each phase at the server, ONE boot
with everything in. Launch = the canonical serve command plus the measured operating point:

```
numactl --interleave=all ./target/release/rabbit --model /data/hf/hub/kimi-k3 --serve \
    --port 8000 --expert-cache 896 --no-usage-cache --preload-experts \
    --threads 48 --numa --numa-threads 192
```

(48 global threads for attention, 2 pinned per-socket pools × 96 for experts + the sharded
lm_head.) Same 128-completion-token turn shape as every serve measurement on this page,
`/profile` buckets, misses frozen at 82,432. Turn 1 is first-touch-cold, turn 2 is the steady
number:

| per token | v1 @48 (rev-2 baseline) | combined boot, turn 1 | **combined boot, turn 2 (warm)** |
|---|---|---|---|
| wall | 1.42 s | 1.385 s | **1.264 s (−11%)** |
| expert | 0.83 s | 0.755 s | **0.656 s (−21%)** |
| attention | 0.53 s | 0.578 s | 0.556 s (+5%) |
| lm_head | 0.0084 s | 0.0034 s | **0.0031 s (2.7×)** |
| preload | ~30 min | **542 s (~9 min)** | — |

Per-socket MoE busy is balanced (37.0 / 36.2 s over the first turn — the D1 skew argument
holding at 2 domains). An accidental extra data point worth keeping: a stale-binary boot (old
per-node dispatch, pre-N4) at the same g48/p192 measured wall 1.362 / expert 0.784 — i.e.
per-node vs per-socket at the server is within run-to-run noise on wall; the flip's clear wins
were on the teacher-forced bench and in the skew counters.

**Where this leaves K3 serving.** Wall is now 1.26 s/token, attention-gated as much as
expert-gated (0.66 + 0.56 + 0.01 ≈ the wall). The two remaining levers, both already
diagnosed with data on this page: (1) **pools stay hot** — the ~0.4–0.8 ms per-fan-out
wake/distribution cost is the binding constraint on the expert bucket's remaining ~0.35
s/token of orchestration gap AND the thing that made the KDA-projection sharding (attention's
0.4 s/token kda_proj chunk) a measured regression; fixing it unlocks both at once, and the
re-enable path is four lines in `distribute_dense`. (2) The expert compute floor at current
pool widths is ~0.29 s/token (measured max-domain busy) — reaching it is (1) again. N6 (AMX)
remains unscheduled.

**Full-effort scoreboard** (real checkpoint): decode 36.8 s/token at the Phase 0 baseline →
16.9 (Phases 2–5) → **1.26 s/token in live serving** at this boot — with logits bit-identical
across every scheduling configuration tested, pinned by fingerprints at every gate.

## Tried and didn't help (target-box round)

- **`RUSTFLAGS="-C target-cpu=native"`** (2026-08-02, closing the measurement the original
  brief's 0c deferred): synth canonical command, default flags vs native — off@48 2.88–3.16 s
  (session range) vs 3.43 s native; numa g48/p192 2.40 s vs 2.32 s native. Within run-to-run
  noise both directions, logits fingerprint unchanged. Expected in hindsight: every hot matmul
  is hand-written `#[target_feature]` intrinsics behind runtime dispatch, so `native` only
  recompiles the glue around them, which the phase buckets show is not where the time goes.
  Confirms D1's "don't commit it" from the measurement side; binaries stay portable.

### Reproducing the K3 numbers

Decode timing via `teacher_forced_decode_bench` only (free-running greedy is not run-to-run
reproducible on this codebase — see the harness's module doc). Canonical command:
`cargo run --release --example teacher_forced_decode_bench -- --model /data/hf/hub/kimi-k3
--steps 30 --expert-cache 896` (Phase 0 baseline used `--steps 12` for cost; note the step count
with every number). Kernel timing via `cargo bench --bench kernels -- mxfp4`. Before/after per
phase via `git worktree` at the pre/post-phase commits, identical command lines. Record
`RUSTFLAGS`, thread count, `numactl` invocation, and THP state with every number — and since the
NUMA phases, the `--numa`/`--numa-threads` state too. The bench prints a per-run **logits
fingerprint** (FNV-1a over every step's logits bits): identical fingerprints = bit-identical
logits, the acceptance instrument for scheduling-only changes.
