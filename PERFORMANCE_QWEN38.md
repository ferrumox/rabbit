# Qwen 3.8 performance history

Companion to `PERFORMANCE.md` (GLM-5.2) and `PERFORMANCE_KIMI_K3.md` (Kimi K3). This page tracks
**Qwen3.8-Max** (`Qwen3.8-2.4T-A95B`, 2.446 T parameters, routed experts natively OCP MXFP4 on disk),
run the same way: most of the model stays on disk and streams in as each generated word needs it.

**First real run: 2026-08-13**, the day the 1.37 TB checkpoint finished downloading. The port's
correctness state lives in `QWEN38_PORT.md`; this page is only about speed and memory.

The measurement protocol below was written BEFORE any number was seen — because K3's own page documents
a measurement that was confounded twice (once by an accumulated usage cache, once by OS page-cache
warmth) before it could be trusted.

## What this model asks of this machine

Arithmetic from the real `config.json`, printed by `examples/qwen38_config_dump.rs` — not a
measurement, and the only numbers on this page that don't need a run:

| | Qwen3.8-Max | Kimi K3 (measured, for reference) |
|---|---|---|
| Total parameters | 2.446 T | 2.8 T |
| Routed experts per token | 10 of 512, x 92 layers | 16 of 896, x 92 layers |
| **Params read per token** | 46.3 B | 48.6 B |
| **Bytes read per token** (4-bit + block scales) | **24.6 GB** | 25.8 GB |
| Experts on disk | 1.26 TB | 1.45 TB |
| Non-expert block | ~76 B params (bf16 on disk, requantized at load) | ~77 B params |
| Layers with a growing KV cache | 23 of 92 (8 KB/token) | 92 of 93 (MLA latent) |

So Qwen3.8-Max asks for slightly LESS per token than the model this machine already runs at
**5.9 s/token** (K3, v0.28.1). That comparison is the point of this page: it's the honest baseline, and
if Qwen lands far off it, something is wrong rather than merely slow.

Two structural differences that could move it either way, both untested:

* **The checkpoint is split across both NVMe drives** (`--shard-dirs`), which K3 never was — it ran
  entirely off `/mnt/data`. The multi-drive split measured ~2x read bandwidth in isolation, so this is
  the first run where that should show up end to end.
* **Only 23 of 92 layers accumulate KV state.** The other 69 are Gated DeltaNet with fixed-size
  recurrent state, so context length costs almost nothing in memory — but each GDN layer touches
  128 heads x a 128x128 state per token, which is compute the MLA-heavy models don't pay.

## Protocol

Same prompt, same flags, one variable at a time. Every row must record which of these applied.

1. **`--no-usage-cache` for every comparison row.** K3's page has a whole section on a false alarm
   caused by an accumulated `.rabbit_usage` history changing the pin set between runs. The auto-pin
   behavior gets its own row, deliberately, rather than silently coloring all of them.
2. **`--expert-cache` stated explicitly per row.** Both forms get measured: a FIXED capacity (so code
   changes can be compared against each other) and the real RAM-aware auto default (so the page says
   what actually happens when someone just runs the thing). K3's page keeps these in separate tables
   for exactly this reason; this page will too.
3. **Page-cache warmth stated per row.** A first run after boot is cold; repeated runs against the same
   checkpoint are not. Both are legitimate numbers, but they are not the same number.
4. **`s/token` AND `tok/s` in every table**, plus model load and prefill separately.
5. **Disk wait split out from compute.** `examples/qwen38_smoke.rs` reports it per token from the expert
   cache's own `io_uring` accounting; the CLI reports the same via its phase breakdown. A decode step
   that is 80% disk wait and one that is 80% compute call for completely different next steps.
6. **Per-drive read counters** for the shard-split claim, since "both drives are being read" is an
   assumption until measured:
   ```
   awk '$3 ~ /^nvme[01]n1$/ {print $3, $6*512/1e9 " GB read"}' /proc/diskstats   # before and after
   ```
7. **A/B comparisons use `examples/teacher_forced_decode_bench.rs`, never free generation.** Free
   greedy decode is not reproducible run to run on this codebase (the per-expert early drain makes
   floating-point summation order depend on real disk timing, which can flip a near-tie argmax and send
   the continuation somewhere else entirely). That harness is already family-generic — it drives
   `rabbit::model`'s dispatch, so it works for Qwen with no changes.

Reference command for the first real run:

```
cargo run --release --example qwen38_smoke -- \
    --model /mnt/data/qwen38-max-mxfp4 \
    --shard-dirs ~/qwen38-max-mxfp4-shards2 \
    --prompt-len 8 --max-tokens 10
```

and, for coherence rather than timing (the smoke uses synthetic token ids, no tokenizer):

```
cargo run --release --bin rabbit -- \
    --model /mnt/data/qwen38-max-mxfp4 \
    --shard-dirs ~/qwen38-max-mxfp4-shards2 \
    --prompt "¿Cuál es la capital de Francia?" --max-tokens 40 --no-usage-cache
```

## First working version

| Version | Model load | Prefill (8 tokens) | Decode (10 tokens) | s/token | tok/s | Config | Page cache |
|---|---|---|---|---|---|---|---|
| v0.29.0 (first working version) | **101.8s** | **45.0s** | **52.4s** | **5.24** | **0.191** | auto `--expert-cache 9`, `--dbits 4 --ebits 4`, `--shard-dirs` across both NVMe | **cold** — first run ever against this checkpoint, minutes after its 1.37 TB download finished |

Per-token spread across the 10 decode steps: 4.37s to 6.33s, no trend (the first token is the slowest,
as on every other model here). Disk wait was 2.20-4.08s of each, i.e. **~50-64% of a decode step is
waiting for disk** and the rest is compute.

### Second run: a real prompt through the CLI

Same checkpoint, minutes later, so the OS page cache was no longer cold — that alone makes this row not
directly comparable to the one above, which is why both are here.

| | |
|---|---|
| Command | `--prompt "¿Cuál es la capital de Francia? Respondé en una frase." --max-tokens 40 --no-usage-cache` |
| Model load | 113.1s (11s more than the smoke: this path also reads the 12.8 MB `tokenizer.json`) |
| Prompt | 14 tokens |
| Prefill | 59.3s (5,978 expert misses, 39.6s of it disk wait) |
| **Decode, 39 timed steps** | **183.8s -> 4.71 s/token, 0.212 tok/s** |
| Per-step spread | min 3.00s, median 4.60s, max 5.80s |
| Disk wait | 104.4s = **57% of decode** |
| Expert cache at the end | 11,750 hits / 30,108 misses (~28% hit rate, up from the smoke's 17%) |

**Careful with the CLI's own summary line.** It prints `40 tokens in 243.5s (0.2 tok/s)`, but that
figure is the whole TURN — it includes the 59.3s prefill. Decode alone is the 4.71 s/token above,
computed by summing the per-step times the CLI reports. (The 40th "token" is the synthetic zero-cost
event `generate_reply` emits when `max_tokens` is hit exactly on selection, so 39 steps have real
times.)

Faster than the smoke's 5.24 s/token, and the likely reason is visible in the counters: a 28% cache hit
rate over 40 tokens versus 17% over 10, plus a page cache the smoke had already warmed.

### Does it actually say anything sensible?

Yes. Truncated at 40 tokens, still inside its reasoning block:

```
<think>
The user is asking "What is the capital of France?" and wants me to respond in a
single sentence... is straightforward - the capital of France is Paris. I'll
```

Note what this run does and does not exercise: the CLI's `--prompt` path encodes the raw text and does
NOT apply the chat template (same as for every other family here — the template is used by `--chat` and
`--serve`), so the `<think>` block above is the model recognizing its own training format unprompted,
not something the prompt gave it. Template correctness is covered separately, by the 9/9 comparison
against the real `chat_template.jinja` in `tests/qwen38_chat_template_fixture.rs`.

### Against Kimi K3, the only comparable thing this machine has run

| | Qwen3.8-Max (2.446 T) | Kimi K3 (2.8 T, v0.28.1) |
|---|---|---|
| Model load | 101.8s | 102.3s |
| Decode | **5.24 s/token — 0.191 tok/s** | 5.93 s/token — 0.169 tok/s |
| Bytes/token (arithmetic) | 24.6 GB | 25.8 GB |
| Checkpoint layout | split across BOTH NVMe (`--shard-dirs`) | entirely on one drive |

**~12% faster than K3**, in line with what the per-token byte arithmetic predicted. Two caveats before
reading anything more into it: K3's numbers came from a warm page cache after many runs, and Qwen's
checkpoint is split across both drives while K3's was not — so this is not a like-for-like
architecture comparison, it's what each model actually does on this machine as configured.

## Memory

| | Measured |
|---|---|
| Non-expert block resident, right after load | **24.3 GB** (arithmetic had predicted ~40 GB — it fits better than expected) |
| Expert cache capacity chosen by the auto clamp | **9 per layer** (down from the flat default of 64) |
| RSS after prefill / after 10 decode tokens | 45.6 GB / **45.7 GB** — flat, i.e. the LRU is holding its bound |

The clamp explained itself in the run's own output, which is the behavior K3's OOM crash produced:

```
auto --expert-cache 64 would risk ~236GB peak memory on this checkpoint
(92 MoE layers x ~26.7MB/expert, LRU + pinned tiers combined) -- lowered the auto
default to 9 to stay under a ~35GB safety budget (40% of real available RAM)
```

Without that clamp this run would have asked for ~236 GB on a 128 GB machine.

## Where the time goes

**~57-64% disk wait, the rest compute** (per-token, from the expert cache's own `io_uring` accounting).
That is a much healthier split than a naive reading of "24.6 GB per token" suggests, and the reason is
the two-drive layout — see below.

The CLI's phase breakdown over the whole 243.5s turn (prefill + 40 decode steps) puts numbers on every
piece, and it reorders the priorities badly enough to be worth stating plainly:

| Phase | Time | Share |
|---|---|---|
| Expert wait (disk) | 143.8s | **59%** |
| Expert matmul (compute) | 76.6s | 31% |
| Mixers — ALL 92 layers, the 69 GDN ones included | 22.4s | **9%** |
| `lm_head` | 0.6s | 0.2% |

**The 69 Gated DeltaNet layers plus the 23 attention ones together are 9% of the time.** An earlier
draft of this page listed "parallelize the GDN per-head loop" as the top optimization; that was wrong,
and the measurement is what says so — perfect parallelism there would buy at most a few percent. The
work is in the MoE: 59% waiting for expert bytes and 31% multiplying them.

### The shard split is real, and measured

`/proc/diskstats` deltas across the whole smoke run (prefill + 10 decode tokens):

| Drive | Read | Share |
|---|---|---|
| nvme0n1 (`/mnt/data`, 129 shards) | 233.6 GB | 61% |
| nvme1n1 (`/`, 84 shards) | 147.3 GB | 39% |
| **total** | **380.9 GB** | — |

61/39 against the 60/40 the download script aimed for when it distributed the shards. Both drives are
genuinely being read in parallel — this is the first checkpoint on this machine that does so.

**380.9 GB over 18 tokens = 21.2 GB/token measured**, against the 24.6 GB/token the config's arithmetic
predicts for a cold cache. The gap is the expert cache doing its job: 2,417 hits against 13,941 total
selections (~17%), and `24.6 x 0.83 = 20.4 GB`, which lands within noise of the measured figure. Two
independent numbers agreeing is the reason to trust either.

### What to try next, in order of expected value

Reordered after seeing the phase breakdown above, not before it.

1. **Read fewer bytes per token — the 59%.** Two candidates, both measurable before committing:
   *adaptive top-k* (drop routed experts whose cumulative router probability is already ~0.9; top-10
   effectively becomes ~6-7, cutting disk traffic proportionally), and *raising the cache hit rate*,
   which the counters show is already worth real time — 17% -> 28% between the two runs above tracked a
   5.24 -> 4.71 s/token improvement. The `.rabbit_usage` histogram this checkpoint is now accumulating
   feeds the auto-pin tier that exists for exactly this.
2. **Speed up the expert matmul — the 31%.** `perf` first, on the MXFP4 path specifically: K3's v0.28.0
   found a generic `powi()` call eating 29% of cycles in the block-scale decode, which no amount of
   parallelism would have fixed. Assume nothing here that a profile hasn't shown.
3. **More read bandwidth.** The 61/39 split across two drives is already working; the OCuLink port is
   free, and a third/fourth NVMe would attack the same 59% as item 1 without any quality trade-off.
4. **Parallelize the per-head loops — the 9%.** Last, not first. Worth doing eventually (it is easy and
   `glm52::attention` shows the shape), but it cannot move the headline number much.
5. **Re-measure settled.** Both runs above happened within minutes of writing 1.37 TB to QLC drives,
   which leaves them read-degraded for a while (`fstrim` needs root here, so it could not be forced).
   A re-run hours later is the honest steady-state number, and would also separate "warm page cache"
   from "settled drives", which these two runs conflate.

## Speed by version

Only one version exists so far (the row above). This table starts accumulating once there is a second
one to compare against — and per this project's convention, a minor version bump requires a real
measured improvement, not just a change that ought to help.
