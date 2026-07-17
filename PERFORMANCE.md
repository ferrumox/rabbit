# Performance history

Every entry below was measured against the real 744B-parameter checkpoint, never estimated from
architecture alone — see `rabbit-plan.md`'s phase entries for full methodology. Two lessons that
shaped how every number below was collected, learned the hard way early on and worth stating up
front:

1. **A controlled A/B beats a before/after against an older baseline.** Page-cache warmth, expert
   selection drift across processes, and general run-to-run noise on this machine are large
   enough to fake a "win" — every number below either reproduces on a repeated baseline run or
   says so explicitly when it doesn't.
2. **A sub-metric that moves while total wall-clock doesn't is a measurement bug, not a finding.**
   One entry below (the `load_nanos` fix) exists purely because an earlier "I/O time dropped"
   reading turned out to be a timer starting in the wrong place, not a real change in behavior.

Reverted or never-released experiments are included — a technique that didn't help is exactly as
worth recording as one that did, so nobody re-tries it blind.

## Test hardware

All numbers on this page, including the colibrì comparison, come from the same machine unless a
row says otherwise:

| | |
|---|---|
| CPU | AMD Ryzen AI 9 HX 370 — 12 cores / 24 threads, AVX2 + AVX-512F/BW/VNNI |
| RAM | 123 GB |
| Storage | NVMe SSD |
| OS | Linux |
| Checkpoint | [`jlnsrk/GLM-5.2-colibri-int4`](https://huggingface.co/jlnsrk/GLM-5.2-colibri-int4) — 378 GB, 744B params, int4, community conversion via colibrì's own `convert_fp8_to_int4.py` |

## Summary

Scan this table for the headline number; the full log below has the exact test conditions and
reasoning behind each row. **Speed values are absolute tok/s for that specific test run** (token
count varies row to row — a short prompt's early decode steps and a long run's steady-state
differ even with no code change at all — so treat each row as self-contained, not one continuous
series you can chain end to end). "Stage" is the release tag the change shipped in; a dash means
it never shipped (reverted, or a config-only experiment with no code to release).

| # | Stage | Technique | Speed (tok/s, before → after) | Δ | Verdict |
|---|---|---|---|---|---|
| 1 | v0.8.0 | AVX2 / AVX-512-VNNI SIMD kernels | n/a — bit-exactness check, not a speed comparison | — | Kept |
| 2 | v0.9.0 | `io_uring`-batched expert streaming | n/a — synthetic load-time bench only, not decode speed | ~0 | Kept — correctness, not speed |
| 3 | v0.10.0 | `rayon` matmul parallelization | 0.04 → 0.14 (5-token run) | **3.5×** | **Adopted** |
| 3b | v0.10.0 | ↳ same change, decode steady-state | 0.05 → 0.25 (per-token) | **3.8×** | **Adopted** |
| 4 | v0.10.0 | Overlap I/O with shared-expert compute | 0.18 → 0.18 (8-token run) | ~+2% | Neutral — kept |
| 5 | v0.10.0 | Reusable `yt` scratch buffer | 0.18 → 0.18 (8-token run) | ~0% | Neutral — kept |
| 6 | — (reverted) | Expert-matmul fusion (decode) | 0.18 → 0.15 (8-token run) | **−25%** | **Reverted** |
| 7 | — (reverted) | Cross-step expert prefetch hint | 0.20 → 0.19 (10-token run) | ~0% / slightly negative | **Reverted** |
| 8 | — (config only) | `--expert-cache` 64→128 | 0.21 → 0.13 (15-token run) | **−60%** (swap thrashing) | Don't raise past ~64 on this box |
| 9 | pre-v0.14.0 | Usage cache, eager pin (v1) | n/a — superseded before release, see row 10 | **−4%** | Redesigned |
| 10 | v0.14.0 | Usage cache, lazy/sticky pin (v2) | n/a — decode-only wall-clock only, token count not isolated | ~0% (regression fixed) | **Adopted** |
| 11 | v0.15.0 | Parallel absorbed-attention decode | 0.31 → 0.44 (70-token run) | **+29%** (grows with context) | **Adopted** |
| 12 | reference | rabbit vs colibrì, head-to-head | rabbit 0.20 vs colibrì 0.16 (10-token run, both) | rabbit **+17%** | Reference, not a rabbit change |
| 13 | v0.16.0 | `CACHE_ROUTE` (cache-aware MoE routing), overall | 0.32 → 0.36 (30-token run) | **+10%** | **Adopted**, opt-in |
| 13b | v0.16.0 | ↳ same change, decode-only | 0.40 → 0.48 (~29-token decode) | **+16.5%** | **Adopted**, opt-in |

## Full log

| Stage | Technique | Measured result | Verdict |
|---|---|---|---|
| v0.8.0 | AVX2 / AVX-512-VNNI SIMD kernels, runtime-selected | Bit-exact vs scalar across all tiers | Kept |
| v0.9.0 | `io_uring`-batched expert streaming | Synthetic bench: slower than sequential `pread` on a page-cache-hot fixture (~80ms vs ~21ms) — no blocking I/O left to collapse in that fixture; the real cold-disk target isn't reproducible on this 123 GB-RAM box | Kept — correctness verified separately (byte-identical vs sequential), architecture is the point, not this one benchmark |
| v0.10.0 | `rayon` matmul parallelization (cores, not just SIMD) | 128.9s → 36.3s for a 5-token prefill+decode run (**3.5×**, 0.04→0.14 tok/s); decode steady-state ~19s/token → ~4s/token (0.05→0.25 tok/s); prefill 52.0s → 19.7s; output bit-identical | **Adopted — the single largest win recorded here** |
| v0.10.0 | Overlap first `io_uring` chunk's read with the shared expert's compute | 43.6s vs 44.5s (8 tokens, 0.18 tok/s both sides) — noise-level | Kept (correct, zero risk, real architectural parity with colibrì) but **not a measured win** |
| v0.10.0 | *(measurement fix)* `load_nanos` timer started after `submit_batch`'s own synchronous sidecar reads | Fixed an artificial ~30% drop in reported I/O time; total wall-clock hadn't moved the whole time | Bug, not a technique — included because it's why the row above is trusted |
| v0.10.0 | Thread-local reusable `yt` scratch buffer (vs allocating per matmul call) | 44.6s vs the 43.6–44.5s baseline range (8 tokens, 0.18 tok/s both sides) — indistinguishable from noise | Kept (bit-identical output, zero risk) but **not a measured win** |
| — (reverted) | Expert-matmul fusion for decode (`s==1`): batch same-shaped experts into fewer, larger `matmul` calls | **~25% regression** (54.7s vs ~44s, 8 tokens, 0.18→0.15 tok/s) — `concat_rows`' copy cost (~7 GB/token at topk=8) outweighed the dispatch-overhead savings it was meant to buy | **Reverted**, never committed |
| — (reverted) | Cross-step expert prefetch hint (`posix_fadvise WILLNEED` on last turn's picks) | Controlled A/B: 50.6s without vs 51.3–51.8s with (10 tokens, 0.20→~0.19 tok/s) — no benefit, likely a slight net negative from the extra syscalls | **Reverted**, never committed |
| — (config only) | Raise `--expert-cache` capacity 64→128 | RSS climbed to 84 GB, 11+ GB swapped, decode steps blew up (22.0s/17.3s/15.7s/6.3s vs a normal ~3.5–4.5s) — 116.0s for 15 tokens vs an estimated ~72s at capacity 64 (**~60% regression**, 0.21→0.13 tok/s) | Confirms 64 is close to this 123 GB box's real safe ceiling, not an arbitrary conservative default — no code to revert, just don't raise it |
| pre-v0.14.0 | Persistent expert-usage cache (`.rabbit_usage`), first design: **eager** pin at session start | Decode-only 108.5s vs 104.3s no-cache; wall-clock incl. startup 124.5s vs 111.0s — **net regression** for `--prompt` (975 experts loaded synchronously before generation even begins, with no long session to amortize the cost over) | Redesigned before ever being released — v0.14.0 shipped only the fixed version below |
| v0.14.0 | Same feature, **lazy/sticky** pin (mark candidates, promote only on first real load through normal use) | 106.6s with-cache vs 106.4s no-cache — regression gone | **Adopted** — correct and low-risk; a real win in a long-lived `--chat`/`--serve` session is still unmeasured |
| v0.15.0 | Parallelize the absorbed-attention decode path with `rayon` (per-head loop, previously sequential) | 224.3s → 158.4s for 70 real decode tokens (**~29% faster**, 0.31→0.44 tok/s), output bit-identical; per-step speedup *grew* with context (28% at position ~17, 42% at position ~86), matching the theoretical prediction that the win scales with context length | **Adopted** |
| reference | Head-to-head vs colibrì (`--cap 64` both sides, colibrì's MTP disabled for a fair comparison) | rabbit 50.6s vs colibrì 60.91s for 10 tokens on the same checkpoint (**rabbit ~17% faster**, 0.20 vs 0.16 tok/s), despite colibrì prefilling more tokens and getting a 236-expert warm-start pin advantage rabbit didn't have; colibrì's own hit rate that run was 41.1% vs rabbit's typical 70–77% | Reference point, not a change to rabbit |
| v0.16.0 | `CACHE_ROUTE` (colibrì's max-rank cache-aware MoE routing, ported opt-in) — true top-J always kept, remaining slots prefer pin∪LRU-resident experts up to a wider rank window M | Baseline reproduced twice (93.0s/30 tokens both times, identical per-step hit/miss counts, 0.32 tok/s). With `--cache-route`: 83.6s/30 tokens (0.36 tok/s), hit rate 65.5%→73.3%. **~10% faster overall, ~16.5% faster decode-only** (72.9s→60.9s / 0.40→0.48 tok/s, excluding the cold-start prefill which sees no benefit since nothing is resident yet) | **Adopted, opt-in / off by default** |

**Evaluated but never attempted in rabbit at all:** EXPERT_BUDGET (capping distinct experts
loaded per layer). colibrì tried it, initially reported up to 1.8×, then quarantined the feature
themselves after reproducing on 3 hosts that it collapses accuracy to chance-floor and can
increase disk I/O rather than reduce it — capping during prefill corrupts the hidden state before
decode even starts. Read as a warning before implementing, not tried here.

## Composed effect

These aren't independent multipliers (the wins overlap — e.g. `rayon` parallelization changes the
compute/I/O balance that later decode-path and routing work then optimizes against), and the
absolute tok/s numbers above come from different prompts/token counts/context lengths, so treat
the tables as a log of individual, isolated measurements — not a chain you can multiply through
for a single "rabbit is N× faster than v0.1.0" headline. The two unambiguous structural wins to
date are the initial `rayon` matmul parallelization (3.5× on a short prefill+decode run) and the
absorbed-attention decode parallelization (~29% on a longer 70-token decode run, growing with
context) — both bit-identical to their pre-change output. `CACHE_ROUTE` is the newest addition
and the first specifically **routing-side** (rather than compute- or I/O-side) win.

## Reproducing these numbers

Every row above used the same real checkpoint and a controlled A/B (same prompt, same seed,
`--temperature 0` for determinism, baseline re-run at least once to confirm it reproduces before
trusting a delta). `--no-usage-cache` was used for routing/cache experiments specifically to keep
the persistent pin file from becoming a confound between the two sides of an A/B — see each row's
own phase entry in `rabbit-plan.md` for exact commands where it matters.
