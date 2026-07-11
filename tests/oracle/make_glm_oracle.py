"""Builds a MINISCULE GLM-5.2 (`glm_moe_dsa`) with random weights, as an ORACLE.

Real architecture (MLA + DSA indexer + sigmoid/noaux_tc router + shared expert) at tiny
dimensions. Saves weights+config into `glm_tiny/` and a greedy reference into `ref_glm.json`.
Short sequence (<= index_topk) so DSA selects every key and attention coincides with dense
MLA: rabbit can validate token-exact output without needing the real sparse-selection path
to be exercised by this particular fixture (that's covered separately by
`attention.rs`'s `dsa_force_select_all_matches_plain_dense_attention` unit test).

Adapted from colibri's `c/tools/make_glm_oracle.py` — self-contained, no colibri checkout
needed. Run FROM this directory (`tests/oracle/`), since it writes `glm_tiny/`+`ref_glm.json`
relative to the current directory:

    pip install torch transformers safetensors huggingface_hub numpy
    cd tests/oracle && python3 make_glm_oracle.py

Then, from the repo root:

    cargo test --test teacher_forcing
"""

import json
import os

import torch
from safetensors.torch import save_file
from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

torch.manual_seed(1234)

cfg = GlmMoeDsaConfig(
    vocab_size=256,
    hidden_size=128,
    intermediate_size=64,  # dense MLP (first 3 layers)
    moe_intermediate_size=32,  # expert
    num_hidden_layers=5,  # 3 dense + 2 sparse
    first_k_dense_replace=3,
    num_attention_heads=4,
    num_key_value_heads=4,
    n_routed_experts=8,
    num_experts_per_tok=2,
    n_shared_experts=1,
    q_lora_rank=64,
    kv_lora_rank=32,
    qk_nope_head_dim=24,
    qk_rope_head_dim=8,  # even -> interleave is well-defined; head_dim becomes 8
    v_head_dim=32,
    index_topk=4096,  # >> seq_len -> DSA selects everything (no-op)
    index_head_dim=16,
    index_n_heads=2,
    n_group=1,
    topk_group=1,
    norm_topk_prob=True,
    routed_scaling_factor=2.5,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    tie_word_embeddings=False,
    rms_norm_eps=1e-5,
    attention_bias=False,
    max_position_embeddings=4096,
)
cfg._attn_implementation = "eager"

model = GlmMoeDsaForCausalLM(cfg).eval()
# make weights non-trivial (default init is very small): scale router/bias for varied topk.
with torch.no_grad():
    for n, p in model.named_parameters():
        if p.dim() >= 2:
            p.normal_(0, 0.05)
    # router correction bias: distinct values so selection is meaningful.
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, "gate"):
            layer.mlp.gate.e_score_correction_bias.copy_(torch.linspace(-0.1, 0.1, cfg.n_routed_experts))

print("=== state_dict tensors (names for the rabbit loader) ===")
for n, p in model.state_dict().items():
    print(f"  {n:60s} {tuple(p.shape)}")

prompt = [3, 14, 159, 26, 53, 58, 200, 11, 77, 240, 5, 99]  # arbitrary token ids, short seq
ids = torch.tensor([prompt])
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=20, do_sample=False, use_cache=True)
full = out[0].tolist()
print("\nprompt:", prompt)
print("full  :", full)

# teacher-forcing: a single forward over the whole sequence -> argmax per position.
# For greedy, tf_pred[i] == full[i+1] holds for i >= len(prompt)-1; this validates rabbit's
# PREFILL separately from decode.
with torch.no_grad():
    lg = model(torch.tensor([full]), use_cache=False).logits[0]  # [seq, vocab]
tf_pred = lg.argmax(-1).tolist()
print("tf_pred:", tf_pred)

# Unfuse routed-expert tensors before saving: `GlmMoeDsaExperts` batches all experts' gate+up
# into one `gate_up_proj[E, 2*I, D]` parameter for compute efficiency (its forward does
# `nn.functional.linear(x, gate_up_proj[e]).chunk(2, dim=-1)`, i.e. rows `[0:I]` are gate,
# rows `[I:2I]` are up — see `GlmMoeDsaExperts.forward` in `modeling_glm_moe_dsa.py`) and
# `down_proj[E, D, I]` similarly. colibri's loader (and rabbit's port) expect the real
# checkpoint's per-expert-file layout instead (`experts.{id}.{gate,up,down}_proj.weight`),
# so split them back out here rather than adding a "fused" special case to the Rust loader.
sd = model.state_dict()
out_sd = {}
for name, tensor in sd.items():
    if name.endswith(".mlp.experts.gate_up_proj"):
        prefix = name[: -len("gate_up_proj")]
        n_experts, two_i, _ = tensor.shape
        i = two_i // 2
        for e in range(n_experts):
            out_sd[f"{prefix}{e}.gate_proj.weight"] = tensor[e, :i, :].contiguous()
            out_sd[f"{prefix}{e}.up_proj.weight"] = tensor[e, i:, :].contiguous()
    elif name.endswith(".mlp.experts.down_proj"):
        prefix = name[: -len("down_proj")]
        for e in range(tensor.shape[0]):
            out_sd[f"{prefix}{e}.down_proj.weight"] = tensor[e].contiguous()
    else:
        out_sd[name] = tensor.contiguous()

os.makedirs("glm_tiny", exist_ok=True)
save_file(out_sd, "glm_tiny/model.safetensors", metadata={"format": "pt"})
json.dump(cfg.to_dict(), open("glm_tiny/config.json", "w"))
json.dump({"prompt_ids": prompt, "full_ids": full, "tf_pred": tf_pred}, open("ref_glm.json", "w"))
print("\nsaved: glm_tiny/ (weights+config) and ref_glm.json")
