#!/usr/bin/env bash
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
set -uo pipefail

REPO="amd/Qwen3.8-2.4T-A95B-Quark-MXFP4"
BASE="https://huggingface.co/${REPO}/resolve/main"
PRIMARY="/mnt/data/qwen38-max-mxfp4"
SECOND="${HOME}/qwen38-max-mxfp4-shards2"
LOG="/mnt/data/qwen38-download.log"
JOBS="${JOBS:-4}"
NSHARDS=213

mkdir -p "$PRIMARY" "$SECOND"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# Remote size via a HEAD that follows HF's CDN redirect; empty if the server won't say.
remote_size() {
    curl -sIL --retry 3 --retry-delay 2 "$1" \
        | awk 'BEGIN{IGNORECASE=1} /^content-length:/ {n=$2} END{gsub(/\r/,"",n); print n}'
}

fetch() { # fetch <url> <dest>
    local url="$1" dest="$2" rsize lsize
    rsize=$(remote_size "$url")
    lsize=$(stat -c %s "$dest" 2>/dev/null || echo 0)
    if [[ -n "$rsize" && "$rsize" == "$lsize" ]]; then
        echo "skip (complete): $(basename "$dest")"
        return 0
    fi
    curl -sL -C - --retry 20 --retry-delay 10 --retry-all-errors -o "$dest" "$url" || return 1
    lsize=$(stat -c %s "$dest" 2>/dev/null || echo 0)
    if [[ -n "$rsize" && "$rsize" != "$lsize" ]]; then
        echo "SHORT: $(basename "$dest") got $lsize want $rsize"
        return 1
    fi
    echo "done: $(basename "$dest") ($lsize bytes)"
}
export -f fetch remote_size

# --- small files (config/tokenizer/template/index) all go to PRIMARY: `--model` points here,
# --- and only this directory is read for config.json + tokenizer files.
log "starting: $REPO -> $PRIMARY (60%) + $SECOND (40%), $JOBS parallel streams"
for f in config.json generation_config.json chat_template.jinja model.safetensors.index.json \
         tokenizer.json tokenizer_config.json vocab.json merges.txt README.md LICENSE; do
    fetch "$BASE/$f" "$PRIMARY/$f" >>"$LOG" 2>&1 || log "note: $f unavailable or incomplete (skipping)"
done

# --- shards: deterministic 3-of-5 assignment so a re-run always targets the same drive.
: >"${LOG}.plan"
for i in $(seq 1 $NSHARDS); do
    name=$(printf "model-%05d-of-%05d.safetensors" "$i" "$NSHARDS")
    # i%5 in {1,2,3} -> PRIMARY (60%), {4,0} -> SECOND (40%)
    if (( i % 5 == 4 || i % 5 == 0 )); then dir="$SECOND"; else dir="$PRIMARY"; fi
    echo "$BASE/$name $dir/$name" >>"${LOG}.plan"
done
log "$(grep -c "$PRIMARY" "${LOG}.plan") shards -> $PRIMARY, $(grep -c "$SECOND" "${LOG}.plan") shards -> $SECOND"

xargs -P "$JOBS" -n 2 bash -c 'fetch "$0" "$1"' <"${LOG}.plan" >>"$LOG" 2>&1
rc=$?

log "finished (exit $rc); primary $(du -sh "$PRIMARY" | cut -f1), second $(du -sh "$SECOND" | cut -f1)"
exit $rc
