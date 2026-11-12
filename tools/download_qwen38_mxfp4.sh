#!/bin/sh
# Resumable download of `amd/Qwen3.8-2.4T-A95B-Quark-MXFP4` (1.37 TB, 213 shards), split across
# BOTH NVMe drives so rabbit's `--shard-dirs` can read them in parallel.
#
# Why this repo and not the 4.89 TB bf16 original: its routed experts ship in OCP MXFP4 with the
# exact on-disk layout rabbit already reads natively for Kimi K3 (`.weight` U8 packed, 2 values
# per byte low-nibble-first, + `.weight_scale` U8 E8M0 per 32 values). Verified this session
# against the bf16 original: decoding expert 0 of layer 3 with rabbit's own nibble order gives
# cosine 0.9933 vs the bf16 ground truth, while both alternative packings give ~0.02. So no
# conversion pass is needed at all -- only a tensor-NAMING variant in `expert_cache.rs`
# (`.weight`/`gate_proj` here vs `.weight_packed`/`w1` on K3).
#
# Split is 60/40 in favor of /mnt/data (nvme0n1, the faster Kioxia) matching its ~7.3 vs ~5.0
# GB/s rating, so both drives finish their share at roughly the same time under load.
#
# Safe to re-run: every file is size-checked against the remote and resumed with `curl -C -`,
# so an interrupted run picks up where it stopped instead of restarting.
#
# Plain POSIX sh, so it runs unchanged on Linux, FreeBSD and macOS: no bash features, no GNU-only
# tools (file sizes come from `wc -c`, header matching uses plain awk). Needs `curl` (on FreeBSD:
# `pkg install curl`). Every setting below can be overridden from the environment, e.g.
#
#   env PRIMARY=/models/qwen38-max-mxfp4 SECOND=/models/qwen38-max-mxfp4 sh tools/download_qwen38_mxfp4.sh
#
# Setting SECOND to the same directory as PRIMARY puts every shard in one place. That is the right
# choice when both drives are already one striped volume (gstripe, RAID 0): the stripe spreads every
# file across both drives by itself, so rabbit needs no `--shard-dirs` there.
set -u

REPO="${REPO:-amd/Qwen3.8-2.4T-A95B-Quark-MXFP4}"
BASE="${BASE:-https://huggingface.co/${REPO}/resolve/main}"
PRIMARY="${PRIMARY:-/mnt/data/qwen38-max-mxfp4}"
SECOND="${SECOND:-${HOME}/qwen38-max-mxfp4-shards2}"
LOG="${LOG:-${PRIMARY}/download.log}"
JOBS="${JOBS:-4}"
NSHARDS="${NSHARDS:-213}"

# Remote size via a HEAD that follows HF's CDN redirect; empty if the server won't say. The last
# Content-Length wins, since each redirect hop prints its own headers.
remote_size() {
    curl -sIL --retry 3 --retry-delay 2 "$1" \
        | awk 'tolower($0) ~ /^content-length:/ {n=$2} END {gsub(/\r/, "", n); print n}'
}

# Local size in bytes, 0 if the file doesn't exist yet. `wc -c` pads with spaces on BSD; `tr`
# strips them so the string comparisons below work on every system.
local_size() {
    if [ -f "$1" ]; then wc -c <"$1" | tr -d ' '; else echo 0; fi
}

fetch() { # fetch <url> <dest>
    f_url="$1"
    f_dest="$2"
    f_rsize=$(remote_size "$f_url")
    f_lsize=$(local_size "$f_dest")
    if [ -n "$f_rsize" ] && [ "$f_rsize" = "$f_lsize" ]; then
        echo "skip (complete): $(basename "$f_dest")"
        return 0
    fi
    curl -sL -C - --retry 20 --retry-delay 10 --retry-all-errors -o "$f_dest" "$f_url" || return 1
    f_lsize=$(local_size "$f_dest")
    if [ -n "$f_rsize" ] && [ "$f_rsize" != "$f_lsize" ]; then
        echo "SHORT: $(basename "$f_dest") got $f_lsize want $f_rsize"
        return 1
    fi
    echo "done: $(basename "$f_dest") ($f_lsize bytes)"
}

# `xargs` below runs this same script once per shard with `--fetch <url> <dest>` (sh has no
# `export -f`, so the script calls itself instead of a shell function).
if [ "${1:-}" = "--fetch" ]; then
    fetch "$2" "$3"
    exit $?
fi

mkdir -p "$PRIMARY" "$SECOND" || exit 1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# --- small files (config/tokenizer/template/index) all go to PRIMARY: `--model` points here,
# --- and only this directory is read for config.json + tokenizer files.
if [ "$SECOND" = "$PRIMARY" ]; then
    log "starting: $REPO -> $PRIMARY (all shards in one directory), $JOBS parallel streams"
else
    log "starting: $REPO -> $PRIMARY (60%) + $SECOND (40%), $JOBS parallel streams"
fi
for f in config.json generation_config.json chat_template.jinja model.safetensors.index.json \
         tokenizer.json tokenizer_config.json vocab.json merges.txt README.md LICENSE; do
    fetch "$BASE/$f" "$PRIMARY/$f" >>"$LOG" 2>&1 || log "note: $f unavailable or incomplete (skipping)"
done

# --- shards: deterministic 3-of-5 assignment so a re-run always targets the same drive.
PLAN="${LOG}.plan"
: >"$PLAN"
i=1
while [ "$i" -le "$NSHARDS" ]; do
    name=$(printf "model-%05d-of-%05d.safetensors" "$i" "$NSHARDS")
    # i%5 in {1,2,3} -> PRIMARY (60%), {4,0} -> SECOND (40%)
    r=$((i % 5))
    if [ "$r" -eq 4 ] || [ "$r" -eq 0 ]; then dir="$SECOND"; else dir="$PRIMARY"; fi
    echo "$BASE/$name $dir/$name" >>"$PLAN"
    i=$((i + 1))
done
if [ "$SECOND" != "$PRIMARY" ]; then
    log "$(grep -c " $PRIMARY/" "$PLAN") shards -> $PRIMARY, $(grep -c " $SECOND/" "$PLAN") shards -> $SECOND"
fi

xargs -P "$JOBS" -n 2 sh "$0" --fetch <"$PLAN" >>"$LOG" 2>&1
rc=$?

if [ "$SECOND" = "$PRIMARY" ]; then
    log "finished (exit $rc); $(du -sh "$PRIMARY" | cut -f1) in $PRIMARY"
else
    log "finished (exit $rc); primary $(du -sh "$PRIMARY" | cut -f1), second $(du -sh "$SECOND" | cut -f1)"
fi
exit $rc
