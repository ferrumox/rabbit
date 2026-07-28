"""Builds a tiny synthetic FP8 checkpoint SHARD and runs it through colibrì's REAL
`convert_fp8_to_int4.py` (vendored verbatim as `colibri_convert_fp8_to_int4.py`, Apache-2.0,
same repo/author as this project) -- not a reimplementation, the actual script -- producing a
reference output directory. `tests/convert_oracle.rs` then runs the SAME synthetic source shard
through rabbit's own Rust port (`glm52::convert::shard::convert_shard`) and diffs every output
tensor byte-for-byte against this reference.

Covers the real-checkpoint-shaped path (FP8 e4m3 weight + `.weight_scale_inv` block-scale
sidecar, dequantized then re-quantized) in both the per-row (default) and grouped (`--group-size`)
int4 quantization modes. NVFP4 is intentionally NOT covered here -- it never appears in rabbit's
own checkpoint, and `dequant_nvfp4`/`quant_int4_grouped` already have dedicated hand-computed
unit tests in Rust (`src/glm52/convert/dequant.rs`, `src/quant.rs`) that don't need a Python
oracle to anchor known LUT/rounding values.

Run (needs Docker -- system pip/venv is broken on this host):
    docker run --rm -v $(pwd):/work -w /work python:3.11-slim bash -c "
        pip install --index-url https://download.pytorch.org/whl/cpu torch
        pip install safetensors numpy
        python3 tests/oracle/make_convert_oracle.py
    "
Then, from the repo root: cargo test --test convert_oracle
"""

import os
import sys

import torch
from safetensors.torch import save_file
from safetensors.numpy import save_file as save_file_np

sys.path.insert(0, os.path.dirname(__file__))
import colibri_convert_fp8_to_int4 as conv

HERE = os.path.dirname(__file__)
SRC_DIR = os.path.join(HERE, "convert_src")
REF_ROW_DIR = os.path.join(HERE, "convert_ref_row")
REF_GROUPED_DIR = os.path.join(HERE, "convert_ref_grouped")


def fp8_block_encode(w, block=128):
    """Real per-128x128-block FP8 e4m3 encode, same convention as the actual checkpoint
    (amax/448.0 per block) and as convert_fp8_to_int4.py's own `--selftest`."""
    O, I = w.shape
    nbr, nbc = (O + block - 1) // block, (I + block - 1) // block
    sc = torch.zeros(nbr, nbc)
    for bi in range(nbr):
        for bj in range(nbc):
            blk = w[bi * block:(bi + 1) * block, bj * block:(bj + 1) * block]
            sc[bi, bj] = blk.abs().max() / 448.0
    sc_full = sc.repeat_interleave(block, 0).repeat_interleave(block, 1)[:O, :I]
    q = (w / sc_full).to(torch.float8_e4m3fn)
    return q, sc


def main():
    os.makedirs(SRC_DIR, exist_ok=True)
    torch.manual_seed(7)

    # One FP8-source dense-MLP weight (Dmlp bucket -> quantized at ebits), one BF16 norm (kept
    # F32), one global norm (kept F32) -- n_layers=1, no MTP/indexer tensors (those code paths
    # are already covered by classify()'s own dedicated Rust unit tests, no oracle needed there).
    w = torch.randn(2, 4) * 3.0
    q, sc = fp8_block_encode(w)

    tensors = {
        "model.layers.0.mlp.gate_proj.weight": q,
        "model.layers.0.mlp.gate_proj.weight_scale_inv": sc,
        "model.layers.0.input_layernorm.weight": (torch.randn(4)).to(torch.bfloat16),
        "model.norm.weight": torch.randn(4),
    }
    shard_path = os.path.join(SRC_DIR, "shard.safetensors")
    save_file(tensors, shard_path)
    print(f"wrote synthetic FP8 shard: {shard_path}")

    for outdir, group_size in ((REF_ROW_DIR, 0), (REF_GROUPED_DIR, 2)):
        os.makedirs(outdir, exist_ok=True)
        out = {}
        conv.convert_shard(shard_path, out, n_layers=1, ebits=4, io_bits=8, xbits=4, group_size=group_size)
        out_np = {k: v.numpy() if torch.is_tensor(v) else v for k, v in out.items()}
        save_file_np(out_np, os.path.join(outdir, "out-00000.safetensors"))
        print(f"group_size={group_size}: wrote {len(out)} tensors -> {outdir}")


if __name__ == "__main__":
    main()
