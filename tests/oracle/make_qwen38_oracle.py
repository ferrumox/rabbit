"""Builds a MINISCULE Qwen 3.8 text model (`qwen3_5_moe_text`) with random weights, as an ORACLE for
`tests/teacher_forcing_qwen38.rs` — the sibling of `make_k3_oracle.py`/`make_kimi_oracle.py`.

Uses `transformers`' OWN `Qwen3_5MoeForCausalLM`, not a reimplementation, so the reference numbers
come from the same code the real checkpoint runs under. (That class IS the text-only causal LM despite
the un-suffixed name: it takes a `Qwen3_5MoeTextConfig` and declares
`_keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]`.) The tiny config deliberately turns on
every Qwen-specific path rabbit's `qwen38` module was built against, all at once:

  * the 3:1 hybrid layer pattern (3 Gated DeltaNet layers + 1 full-attention layer),
  * GQA (4 query heads over 2 KV heads) with `attn_output_gate` and PARTIAL RoPE,
  * GDN's own GQA shape (4 value heads over 2 key heads, i.e. `repeat_interleave` 1:2),
  * routed MoE (4 experts, top-2) plus a gated shared expert.

**Norm weights are randomized on purpose.** `Qwen3_5MoeRMSNorm` scales by `(1 + weight)` and
initializes `weight` to ZEROS, so an oracle built with default-initialized norms would produce
identical numbers under the wrong `norm(x) * weight` convention — the bug would pass unnoticed. Every
`*_layernorm`/`*_norm` weight here is drawn away from zero so the two conventions genuinely differ.

**`mtp_num_hidden_layers` is 0**: the MTP block is a full extra MoE layer that rabbit deliberately
never reads (see `qwen38::model`'s doc), so the oracle doesn't build one.

The routed experts are re-emitted PER EXPERT (`...experts.{e}.{gate,up,down}_proj.weight`) rather than
as `transformers`' fused `[num_experts, 2*inter, hidden]` parameter, because that's how the real
published checkpoint stores them and therefore what `expert_cache.rs` streams. Splitting rule
(verified against the real bf16 checkpoint, whose `gate_up_proj` is `[512, 4096, 8192]`): rows
`0..inter` are gate, rows `inter..2*inter` are up.

Usage (this host's system pip is unusable, so Docker — same as the sibling scripts):

    cd tests/oracle && mkdir -p qwen38_tiny
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/work" -w /work python:3.11-slim bash -c "
        pip install -q --target=/tmp/pylibs torch --index-url https://download.pytorch.org/whl/cpu &&
        pip install -q --target=/tmp/pylibs transformers safetensors numpy &&
        PYTHONPATH=/tmp/pylibs python make_qwen38_oracle.py"

`--user` matters: without it the container writes the fixture as root and `cargo test` then fails with
a bare `PermissionDenied` from `Shards::open`, which looks nothing like an ownership problem.
"""

import json

import torch
from safetensors.torch import save_file
from transformers.models.qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

torch.manual_seed(20260813)

HIDDEN = 32
LAYERS = 4
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 8
LIN_K_HEADS = 2
LIN_V_HEADS = 4
LIN_K_DIM = 8
LIN_V_DIM = 8
KERNEL = 4
EXPERTS = 4
TOPK = 2
MOE_INTER = 8
SHARED_INTER = 8
VOCAB = 64

cfg = Qwen3_5MoeTextConfig(
    hidden_size=HIDDEN,
    num_hidden_layers=LAYERS,
    num_attention_heads=HEADS,
    num_key_value_heads=KV_HEADS,
    head_dim=HEAD_DIM,
    attn_output_gate=True,
    output_gate_type="swish",
    full_attention_interval=4,
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    linear_num_key_heads=LIN_K_HEADS,
    linear_num_value_heads=LIN_V_HEADS,
    linear_key_head_dim=LIN_K_DIM,
    linear_value_head_dim=LIN_V_DIM,
    linear_conv_kernel_dim=KERNEL,
    num_experts=EXPERTS,
    num_experts_per_tok=TOPK,
    moe_intermediate_size=MOE_INTER,
    shared_expert_intermediate_size=SHARED_INTER,
    vocab_size=VOCAB,
    rms_norm_eps=1e-6,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25},
    mtp_num_hidden_layers=0,
    tie_word_embeddings=False,
    eos_token_id=VOCAB - 1,
    dtype=torch.float32,
)

model = Qwen3_5MoeForCausalLM(cfg).to(torch.float32).eval()

# Randomize EVERY parameter, norms included (see this script's doc for why norms matter), at a small
# scale so the tiny model's activations stay in a sane range.
with torch.no_grad():
    for name, p in model.named_parameters():
        if "A_log" in name:
            p.copy_(torch.rand_like(p) * 2.0)  # A = exp(A_log) in (1, ~7): a real decay, not ~0
        elif "dt_bias" in name:
            p.copy_(torch.rand_like(p) - 0.5)
        elif "norm" in name or "layernorm" in name:
            p.copy_(torch.randn_like(p) * 0.1)  # away from 0 -> (1 + w) differs from w
        else:
            p.copy_(torch.randn_like(p) * 0.1)

PROMPT = [3, 17, 42, 8, 25, 1]
GENERATE = 6

with torch.no_grad():
    ids = torch.tensor([PROMPT], dtype=torch.long)
    full = list(PROMPT)
    for _ in range(GENERATE):
        logits = model(torch.tensor([full], dtype=torch.long)).logits
        full.append(int(logits[0, -1].argmax()))

    # teacher-forced pass over the WHOLE sequence, one forward, no cache
    tf_logits = model(torch.tensor([full], dtype=torch.long), use_cache=False).logits
    tf_pred = [int(x) for x in tf_logits[0].argmax(dim=-1)]

# --- weights, renamed/reshaped to the real checkpoint's layout
sd = {}
for name, tensor in model.state_dict().items():
    t = tensor.detach().to(torch.float32).contiguous()
    if name.endswith("mlp.experts.gate_up_proj"):
        prefix = name[: -len("gate_up_proj")]
        for e in range(EXPERTS):
            sd[f"{prefix}{e}.gate_proj.weight"] = t[e, :MOE_INTER, :].contiguous()
            sd[f"{prefix}{e}.up_proj.weight"] = t[e, MOE_INTER:, :].contiguous()
    elif name.endswith("mlp.experts.down_proj"):
        prefix = name[: -len("down_proj")]
        for e in range(EXPERTS):
            sd[f"{prefix}{e}.down_proj.weight"] = t[e].contiguous()
    else:
        sd[name] = t

save_file(sd, "qwen38_tiny/model.safetensors", metadata={"format": "pt"})

config_json = cfg.to_dict()
config_json["model_type"] = "qwen3_5_moe_text"
json.dump(config_json, open("qwen38_tiny/config.json", "w"))
json.dump({"prompt_ids": PROMPT, "full_ids": full, "tf_pred": tf_pred}, open("ref_qwen38.json", "w"))
print(f"saved: qwen38_tiny/ ({len(sd)} tensors) and ref_qwen38.json")
print(f"  prompt {PROMPT} -> full {full}")
print(f"  tf_pred {tf_pred}")
