# Brief: a web dashboard for rabbit (runtime health + execution + expert-usage visibility)

This is a build brief for an AI (or engineer) with no prior context on this project. It
describes the goal, the reference design, what backend data already exists, what doesn't yet,
and the open decisions that need to be made explicitly rather than assumed. Read it fully
before writing any code — the "Postmortem" section at the end exists specifically so you don't
repeat a multi-round trial-and-error cycle that already happened once.

## 1. What rabbit is

**rabbit** is a Rust reimplementation of **the reference implementation** (a C inference engine), both of which run
**GLM-5.2**, a 744-billion-parameter Mixture-of-Experts (MoE) language model, on a single
ordinary machine by keeping the dense part of the model resident in RAM and streaming the
21,504 routed "expert" sub-networks from disk on demand, as needed per token. rabbit is
CPU+disk only today — no GPU/VRAM tier (the reference implementation optionally supports GPU offload; rabbit does
not, at least not yet).

rabbit ships as `rabbit --serve`, an OpenAI-compatible HTTP server (`tiny_http`, single Rust
binary, no async runtime, single-threaded accept loop — the whole point is that the model is
tens of GB resident and only one forward pass can make CPU progress at a time regardless of
concurrency). Repo root: this directory. Key source files: `src/server.rs` (HTTP routes),
`src/chat.rs` (session + generation loop shared by `--chat` and `--serve`), `src/generate.rs`
(the per-layer forward pass), `src/expert_cache.rs` (the per-layer LRU+pin expert cache).

## 2. The goal

Build a **web dashboard**, served by `rabbit --serve` itself (or as a separate piece — see
§6's open decision), that helps a person understand a running rabbit process along three axes
— **not just two**. An earlier draft of this brief scoped this down to only the first two below
(that was an underscope, caught late — see §8's first entry); all three are in scope:

1. **Execution** — where does each chat turn's wall-clock time actually go? (attention vs.
   expert-matmul vs. waiting on disk vs. the final vocabulary projection)
2. **Usage** — which of the 21,504 experts are actually getting used, how "hot" is each one,
   and where does each one currently live (pinned in RAM, ordinarily cached in RAM, or only on
   disk)?
3. **Runtime / at-a-glance health** — is the engine alive and reachable, what hardware is it
   running on (CPU, RAM total/free — no GPU for rabbit, see §1), and in aggregate, across the
   whole model right now, how many of the 21,504 experts sit in each residency tier
   (pinned/cached/disk)? This is a *summary* view, distinct from #2's per-expert detail — it's
   the reference implementation's `/health` endpoint + its Chat sidebar's "Runtime" section, and it's the thing a
   person glances at *first*, before drilling into either #1 or #2. rabbit does not expose any
   of this today (see new §5b).

Also explicitly in scope to decide (not silently assume) is **whether the dashboard should let
you actually chat**, not just observe — the reference implementation's dashboard is, first and foremost, a chat
client (its Chat tab is the default view; Profiling/Brain are secondary tabs next to it).
rabbit's server already exposes `POST /v1/chat/completions` (streaming + non-streaming), so a
real chat UI is buildable, not blocked on new backend work — see §6.

This is explicitly modeled on **the reference implementation's own web dashboard** (a separate, more mature sibling
project — see §3), which already solves this same problem for the C engine. The ask is not to
copy the reference implementation pixel-for-pixel, but to build something that gives rabbit the same *category* of
visibility, executed well.

## 3. Reference design: the reference implementation's real dashboard

the reference implementation's web app lives in its own repo under `~/Documents/laboratory/` (React + Vite + Tailwind v4 +
`lucide-react` icons + shadcn-style components), a local clone on this same machine. It is
**not** part of rabbit and rabbit does not depend on it — it's reference material only.

- **Public screenshot of the shipped Chat view** (2-panel app-shell: sidebar + main panel):
  `https://github.com/JustVugg/<ref>/raw/main/docs/media/dashboard.png`
- **The "Profiling" tab** (execution timing — item 1 in §2) is real but **unmerged**, on
  the reference implementation's `dev` branch (commit `6afffbc`, "Profiling page: per-turn phase timings, live in
  the web dashboard"). It does not have a public screenshot. Its source:
  `web/src/Profiling.tsx` + the `/* ---- Profiling page ---- */` block appended to
  `web/src/index.css` in that same commit. Check that commit out (`git checkout 6afffbc`,
  detached HEAD — do not commit anything to that repo) to read it directly, or run it: it needs
  `node`/`npm` (not installed on this host — use `docker run --rm -v $(pwd):/app -w /app -p
  5173:5173 node:20 sh -c "npm install && npm run dev -- --host 0.0.0.0"` from `web/`, then it's
  reachable at `http://localhost:5173`, click the "Profiling" tab). It won't have live data
  without a running the reference implementation engine to connect to — either point it at nothing (empty state,
  shows layout/typography only) or temporarily hardcode a mock `turns` array into
  `Profiling.tsx`'s `useState` to see it fully populated. **Revert any edits to that checkout
  afterward** (`git checkout -- .` / `git checkout main`) — it's a real clone the user cares
  about, not a scratch copy.
- **The "Brain" tab** (expert usage/routing — item 2 in §2) is real, shipped, on `main`:
  `web/src/Brain.tsx` + the `/* ---- Brain page ---- */` block in `web/src/index.css`. This one
  you can actually run live against the reference implementation if a reference-implementation engine + checkpoint is available, or
  read directly — it's not gated behind an unmerged branch.

### Design tokens actually used (extracted directly from `web/src/index.css`, not guessed)

Dark-only (no light theme — `:root { color-scheme: dark; }`, a deliberate choice, not an
oversight):

```
--background: #080b0d       --primary (accent, mint): #4ed6a5
--foreground: #e9eff0        --primary-foreground: #052118
--card: #0d1215              --secondary: #151c20
--muted-foreground: #96a4a9  --border: #202a2f
--input: #10171a             --destructive: #ff766f
```

Phase palette used by the real Profiling.tsx (`I/O wait` #3987e5 blue, `Expert matmul` #199e70
teal, `Attention` #c98500 amber, `LM head` #008300 dark green, `Other` #9085e9 violet) — dark
text (`#06090b`, not white) on every stacked-bar segment; this is their real, working choice,
not a bug.

Typography: system sans stack (`Inter, ui-sans-serif, system-ui, ...` — Inter is referenced but
never actually loaded via `@font-face`/link in this codebase, so it silently falls back to the
OS sans; that's real, current behavior, not something to "fix" by pretending otherwise). One
deliberate exception: the brand wordmark ("the reference implementation") is set in italic Georgia serif — used
exactly once, nowhere else. All numeric/tabular data (stat tiles, table cells, badges) is
`ui-monospace`/`SFMono-Regular` with `font-variant-numeric: tabular-nums`.

Layout: a sticky ~240-290px sidebar (identity/brand mark, connection status, runtime stats as a
divided 2×2 box — see `.runtime-grid` — inference controls) beside a scrollable main panel with
a ~64-72px topbar (a static label + a row of live pill badges) and content below. Composition
bars use 2px gaps between stacked segments, rounded only at the two outer ends of the whole
bar, `transition: width .5s ease` so they animate when data refreshes. Column/trend charts are
hand-rolled inline SVG (`viewBox="0 0 100 H"`, `preserveAspectRatio="none"`, 3 horizontal
gridlines, one invisible wider hit-rect per column for hover, hover dims every other column to
~45% opacity and swaps a small "foot" readout line to that column's exact numbers).

## 4. Backend data that already exists in rabbit today

An HTTP server is already running and testable: `cargo build --release && ./target/release/
rabbit --model <checkpoint-dir> --serve --port 8000`. Existing routes (`src/server.rs`):

- `GET /v1/models`, `POST /v1/chat/completions` (OpenAI-compatible, streaming + non-streaming) —
  this is how you'd drive traffic to generate data to look at.
- `GET /profile` — **built this session, tested against the real 378GB checkpoint, works.**
  Returns a rolling window (last 120 turns) of per-chat-completion-turn phase timing:
  ```json
  { "seq": 2, "turns": [ { "wall_s": 5.5, "prompt_tokens": 13, "completion_tokens": 2,
      "hits": 918, "misses": 4373, "attention_s": 1.53, "expert_wait_s": 0.51,
      "expert_matmul_s": 4.53, "lm_head_s": 0.04, "forwards": 3 } ] }
  ```
  `hits`/`misses` are cross-layer expert-cache hit/miss counts; `expert_wait_s` is pure
  `io_uring` disk-wait time (already isolated from decode/copy overhead — see
  `ExpertCache::io_wait_nanos`'s doc in `src/expert_cache.rs`); `expert_matmul_s` is everything
  else in the FFN dispatch (dense layers' whole time counts here too, nothing to wait on
  there); `forwards` counts real forward passes in that turn (prefill + decode steps). This
  satisfies item 1 in §2 (**execution**) almost entirely — the instrumentation is `generate.rs`'s
  `Phases`/`StepProfile`/`step_profiled()`, aggregated per-turn in `chat.rs`'s `TurnProfile`,
  pushed into `Session.profile` (a `VecDeque`) by `server.rs`'s two chat-completion handlers.
  Existing, passing test: `generate::tests::step_profiled_reports_nonzero_attention_and_expert_matmul_time`.
- `GET /dashboard` (a hand-rolled `assets/dashboard.html` UI for the above) **existed briefly
  this session and was removed** — it went through ~5 design rounds and still wasn't judged good
  enough, and the user decided a real UI belongs in its own separate repo rather than inside
  rabbit's single-binary constraints (that decision is what this brief now exists to support —
  see §6's tech-stack note). It's gone from the working tree; if you want to see exactly what was
  tried, it's recoverable from this repo's git history (the commit(s) around the "Phase 16"
  entry in `rabbit-plan.md`) — worth a look for the *data-fetching pattern* (poll `/profile`
  every 2s) and the Postmortem (§8) already distills the concrete mistakes, but don't restart
  from that HTML file's actual CSS/layout.

**Nothing exists yet for item 2 in §2 (usage/routing/Brain equivalent)** — see §5.

## 5. Backend data that does NOT exist yet (needed for the Brain/usage view)

`src/expert_cache.rs`'s `ExpertCache` (one instance per MoE layer) already tracks, per expert
id: `is_pinned(eid) -> bool`, `usage_counts() -> impl Iterator<Item=(usize, u64)>` (a persistent
selection-count histogram, already saved/loaded via `src/usage_cache.rs`'s `.rabbit_usage`
file), `get(eid) -> Option<&ExpertSlot>` (residency check — `Some` means currently loaded
somewhere in RAM, `None` means only on disk right now), `pinned_len()`, `capacity()`. Model
scale: `model.cfg.n_experts` (experts per MoE layer) × `model.cfg.n_layers` (total layers, only
some of which are MoE — see `Ffn::Moe` vs `Ffn::Dense` in `src/model.rs`).

What's **missing** to build a reference-implementation-"Brain"-equivalent `/experts` endpoint:

- **A per-expert tier classification rabbit actually has**: since there's no GPU/VRAM here,
  the reference implementation's 3-tier VRAM/RAM/disk model doesn't map 1:1. rabbit's real 3 tiers are: **pinned**
  (`is_pinned`, promoted from historical usage — see `ExpertCaches::warm_start`'s doc),
  **LRU-resident** (`get(eid).is_some()` but not pinned — ordinarily cached, can be evicted),
  and **disk** (`get(eid).is_none()`). This is a legitimate adaptation, not a compromise — call
  it out explicitly in the UI rather than pretending it's the same 3 tiers the reference implementation has.
- **"Routed this turn" tracking** (the thing that drives the reference implementation's white pulse-flash animation
  on its heatmap): nothing in rabbit currently records *which* expert ids were selected on the
  most recent forward pass in a way that survives past that single call. `moe()`
  (`src/moe.rs`) computes a `Routing` per call and discards it. New instrumentation is needed:
  something like a `last_routed: Vec<bool>` (or a bitset) per layer, updated each `moe()` call,
  plus a sequence counter — mirroring the reference implementation's `hits_seq`/`HITS` protocol line
  (`m->hits`/`emap_emit`/`hits_emit` in the reference implementation's `c/glm.c`, if you want the reference
  implementation of that specific mechanic).
- **An actual `GET /experts` HTTP route** in `src/server.rs` serving something like: for each
  MoE layer, a packed grid of `(tier, heat)` per expert id (the reference implementation packs this as one byte per
  cell: top 2 bits = tier, low 6 bits = a log-scaled heat value — see `emap_emit` in `glm.c` for
  the exact packing if you want to mirror it) plus a routed-this-turn bitmap and sequence
  number for the pulse animation.
- Optional, not required for a first version: the reference implementation's "expert atlas" (`experts.json`, a
  measured topic-affinity table per (layer, expert) driving the specialist/generalist hover
  label in `Brain.tsx`) — rabbit has no equivalent measurement pipeline for this. Skip it unless
  explicitly asked for; the heatmap is useful without it (falls back to a generic
  "early/middle/late layer" depth-role heuristic, which `Brain.tsx` already has as a fallback —
  read its `depthRole()` function).

Decide and document, in whatever you build: is scale a problem? 21,504 experts total is a
20-30KB-ish flat byte array per snapshot at 1 byte/expert — cheap. Polling it every ~1.5s
(the reference implementation's interval) is fine.

## 5b. Backend data that does NOT exist yet (needed for the Runtime/health overview, item 3 in §2)

the reference implementation's engine serves this at `GET /health` (see `openai_server.py`'s `Engine`/`APIHandler` if
you want the reference shape): `{ status, hwinfo: { cores, ram_total_gb, ram_avail_gb, gpus,
vram_total_gb, cpu }, scheduler: { active, capacity, queued, max_queue, completed, rejected,
timed_out, cancelled }, kv_slots, tiers: { vram, ram, disk, vram_gb, ram_gb } }`. **rabbit has no
`/health` route at all today** — this needs to be built from scratch, and not everything in
the reference implementation's shape maps onto rabbit's actual architecture:

- **`hwinfo` maps over reasonably cleanly**, minus GPU fields (rabbit is CPU+disk only — report
  `gpus: 0` or omit the field, don't fabricate a value). CPU core count: rabbit already depends
  on the `num_cpus` crate (see `Cargo.toml` — it's already used for thread-pool sizing) and
  `main.rs` already prints `"using {n} threads ({n} physical cores detected)"` at startup, so
  the number is already computed at runtime, just not exposed over HTTP yet. RAM total/free and
  a CPU model string aren't currently read anywhere in rabbit; on Linux (rabbit's primary
  target — see the `io-uring` Linux-only dependency in `Cargo.toml`) both are plain reads of
  `/proc/meminfo` (`MemTotal`/`MemAvailable` lines) and `/proc/cpuinfo` (`model name` line) — no
  new dependency needed for a first version, just a small parser.
- **`scheduler` does NOT map onto rabbit's real architecture and should not be faked.** the reference implementation's
  scheduler numbers exist because its engine multiplexes concurrent requests across
  `KV_SLOTS>1` sessions. rabbit's `--serve` is explicitly single-threaded and stateless — one
  request fully blocks the next (see `src/server.rs`'s own module doc: "Single-threaded,
  always-serialized... accepting requests concurrently would buy nothing but complexity here").
  There is no queue, no concurrent "active" count above 1, no capacity concept beyond 1. If a
  runtime view is useful here at all, it's something much simpler and honest to rabbit's real
  model — e.g. "idle" vs. "generating" (a single boolean/enum), not a queue-depth panel. Do not
  build a fake scheduler panel just to visually match the reference implementation's — that would be actively
  misleading about how rabbit behaves.
- **`kv_slots` does NOT apply.** the reference implementation's `--kv-slots` multi-session concurrency was explicitly
  evaluated and ruled out of scope for rabbit's `--serve` (see `rabbit-plan.md`'s Phase 11 entry:
  "there is no per-request session identity to attach a slot to — every request is stateless").
  `--chat`'s `--session` flag (`src/kv_session.rs`) is a different, single-session,
  save/reload-across-restarts feature — not concurrent slots. Don't build a "KV session" picker
  UI control; it has nothing to attach to server-side.
- **`tiers` (aggregate, NOT the per-expert Brain heatmap from §5)** maps cleanly onto data
  that's almost already there: sum `ExpertCache::len()`, `pinned_len()`, and `capacity()` across
  every MoE layer's cache (`ExpertCaches` in `src/generate.rs` already has this exact aggregation
  pattern for other stats — see `hit_miss_totals()`/`io_wait_nanos_total()` for the pattern to
  follow) to get total-pinned / total-LRU-resident / total-on-disk (= total experts across all
  MoE layers minus the first two) counts. This is the natural summary counterpart to §5's
  per-expert detail — a bar or a few numbers, not a grid.

## 6. Open decisions — do not silently assume either way

**Tech stack.** rabbit is, today, a single Rust binary with zero Node/npm dependency — you
build it with `cargo build --release` alone. This session tried a hand-rolled vanilla
HTML/CSS/JS page (no build step, embedded via `include_str!`) specifically to preserve that
property, and iterated ~5 times without the result being judged good enough. Do **not** assume
vanilla is required — it was this session's choice, not a hard constraint anyone has actually
stated as non-negotiable in this brief. A real React+Tailwind+Vite build (mirroring the reference implementation's
own stack, `web/dist` embedded into the binary, or served as a separate static bundle) is a
completely legitimate alternative if it gets a better result — it does mean rabbit's build
process gains a Node/npm dependency (at least at *build* time; the shipped binary can still
embed the compiled output and stay dependency-free at *run* time, same as the reference implementation's own Tauri
desktop shell does with its `web/dist`). **Ask the user which they want before committing to
one**, or make a clearly-labeled default choice and say so plainly up front.

**Where the views live.** Three views are now in scope (runtime overview, execution/profiling,
usage/Brain — see §2). Whether they're separate routes, tabs in one page (the reference implementation's own
pattern), or something else is an implementation detail — just don't lose any of them.

**Whether a real Chat interface is in scope.** the reference implementation's dashboard is a chat client first,
observability tool second. rabbit's `/v1/chat/completions` (streaming + non-streaming) already
exists and works, so a chat tab isn't blocked on new backend work the way the Brain view is.
Decide explicitly whether this brief's dashboard should let a user actually converse with the
model (closer to the reference implementation's real scope) or stay observability-only (profiling + usage + runtime,
no message composer) — don't default to the narrower scope just because it's less work without
saying so.

**Whether to keep this session's `/profile` backend work.** It's tested, it's working, it's
independent of whatever frontend gets built (any frontend can just call `GET /profile`,
same-origin, no CORS setup needed since it'd be served by the same rabbit process). Recommend
keeping it and building the new UI to consume it, rather than redoing the Rust-side
instrumentation from scratch — but that's a recommendation, not a requirement; if the new
design needs a different data shape, `chat.rs`'s `TurnProfile` struct and `server.rs`'s
`handle_profile` are small and easy to reshape.

## 7. Deliverable scope (first version)

1. A **runtime overview**: consumes the new `/health`-equivalent from §5b. At minimum: engine
   reachability, CPU/RAM info, and the aggregate pinned/LRU-resident/disk expert tier counts.
   No fake scheduler/queue/KV-slot panel (§5b explains why those don't apply). This is the
   glance-first view — probably the natural default/landing tab.
2. An **execution/profiling view**: consumes `GET /profile` (existing). At minimum: current
   throughput, a phase-composition breakdown (attention / expert-wait / expert-matmul /
   lm-head / other) for the latest turn and for the whole window, and a per-turn trend over the
   recent window (not just a single latest-turn snapshot — this was the single biggest gap in
   this session's attempt: a page with only 4 stat tiles and one bar felt sparse; the reference implementation's real
   page also has per-turn column charts and a full per-phase table, which is what actually
   fills out a dashboard rather than feeling like an empty shell around 3 numbers).
3. A **usage/Brain-equivalent view**: needs the new `/experts` instrumentation from §5 first.
   At minimum: a per-layer × per-expert grid, color = tier (pinned/LRU/disk, adapted per §5),
   brightness or a separate signal = historical heat, and ideally a live "just routed" pulse.
   Hover a cell → show layer index, expert id, tier, heat, and (since no atlas exists) a
   generic depth-role guess.
4. A **chat view, if §6's chat-scope decision comes back yes** — a message composer against the
   existing `/v1/chat/completions` (streaming), same as the reference implementation's own primary tab.
5. All live views should **poll**, not require a manual refresh (`/profile`, `/experts`, and the
   new `/health` are all cheap, read-only GETs — matches the reference implementation's own "same trust level as
   `/health`" framing for this class of endpoint).

## 8. Postmortem — what went wrong in this session's attempt, so it isn't repeated blindly

- **This brief itself first shipped under-scoped**: it initially described only the Profiling
  and Brain views (§2's items 1-2) and treated the reference implementation's Chat-sidebar "Runtime" section (hwinfo,
  scheduler, tiers) as pure design reference for a "Session" box, not as its own data
  requirement — missing that it's a third, distinct thing worth its own view (§2 item 3, §5b),
  and missing the open question of whether chat itself belongs in scope at all. Caught only
  because the user pushed back with "but the reference implementation has a lot more than that" — read the reference implementation's
  *whole* dashboard (every tab, not just the two that seemed most relevant to "profiling") before
  scoping down, not after.
- Iterating on vibes ("make it look better") without a concrete reference produced 3+ rounds of
  guessing before actually fetching the reference implementation's real rendered UI (via the Docker+Vite approach in
  §3) settled most open questions immediately. **Get the real reference running or at least
  read the real source before designing**, don't reconstruct component CSS from memory of
  having read it once.
- A single flat page with only 4 stat tiles + 1-2 composition bars + a short table reads as
  "empty"/unfinished regardless of how clean the CSS is, because there just isn't enough real
  content filling the space. The fix that actually worked was adding real per-turn trend charts
  and a fuller table — i.e. **more real data density**, not more decoration (shadows, gradients,
  hero numbers) — solved the "this looks cheap" feedback better than any purely-visual change
  did.
- Copying a CSS rule verbatim without checking it against the actual HTML structure it'll apply
  to caused at least one real bug: a `.connection-state span { width: 6px; height: 6px; ... }`
  rule (correct in the reference implementation's source, where the dot is the only `<span>`) silently mangled a
  status-text `<span>` in the port, since the port used a `<span>` for the text too. Match
  selectors to your own markup, don't assume a copied rule is safe just because the source
  author validated it against *their* markup.
- Confirm real interaction end-to-end, not just static screenshots: hover states on the trend
  charts (dimming siblings, swapping a readout line) were only verified by actually hovering in
  a live browser session, which caught nothing wrong that time, but is exactly the kind of
  thing that silently regresses if only checked via a static screenshot diff.

## 9. How to verify whatever you build

Real checkpoint is at `/home/manuelslemos/Documents/ferrumox/models/glm-5.2-int4`
(378GB, GLM-5.2 int4). Loading takes ~5s (dense part only; experts stream on demand). Suggested
loop: `cargo build --release`, run `./target/release/rabbit --model <dir> --serve --port 8000
--no-usage-cache` (the `--no-usage-cache` flag avoids a slow/noisy `mlock`-retry path that
shows up after many repeated restarts against the same checkpoint during iterative UI dev —
harmless but noisy, unrelated to anything in this brief), fire a few `POST
/v1/chat/completions` requests with a short prompt (`max_tokens: 8` is enough to populate a
turn quickly), then open whatever URL serves the dashboard in a real browser and look at it —
don't just curl the JSON and assume the frontend renders it correctly.
