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
| v0.23.0 | 412.8s | ~50-70s/token | ~0.014-0.02 | — (first working version) | K3 shipped: engine, native-format checkpoint loading, tokenizer, chat template, session persistence — the expert math itself still used the simple, unvectorized loop |
| v0.24.0 | 216.5s | ~37.5s/token avg (range 17.1-44.8s across 40 tokens) | **~0.027** | **Prefill ~1.9× faster; decode ~1.3-1.9× faster** (range because the "before" number was itself a range, not a single measurement) | Taught the expert math to use the CPU's wider instructions, the same idea GLM-5.2's own v0.17.0/v0.22.0 already used for its own number formats |

**First generated token specifically**: 122.3s (v0.23.0) → 44.8s (v0.24.0), **~2.7× faster** — the
single cleanest comparison point, since it's an exact number on both sides rather than a range.

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
it means the next real speed-up has to come from the disk side (K3's expert loader still reads
each missing expert with a plain, one-at-a-time synchronous read rather than GLM-5.2's batched
`io_uring` approach — a known, scoped, not-yet-done next step), not from further speeding up math
that's no longer the thing anyone is waiting on.

## Reproducing these numbers

**End-to-end table**: `cargo build --release --bin rabbit`, then `./target/release/rabbit --model
<checkpoint-dir> --prompt "What is the capital of France?" --max-tokens 40 --expert-cache 64`. The
"before" (v0.23.0) row was measured on the commit tagged `v0.23.0`; the "after" (v0.24.0) row on
this version.

**Isolated kernel benchmark**: `cargo bench --bench kernels -- matmul_mxfp4`.

Hardware: same machine as `PERFORMANCE.md` (AMD Ryzen AI 9 HX 370, 12 cores/24 threads, AVX2 +
AVX-512F/BW/VNNI), running the real `moonshotai/Kimi-K3` checkpoint (1.56TB, 96 shards, native OCP
MXFP4-quantized routed experts) from a local NVMe drive.
