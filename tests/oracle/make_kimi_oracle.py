"""Builds a MINISCULE Kimi Linear (`kimi_linear`) with random weights, as an ORACLE.

Real architecture (KDA + MLA hybrid + noaux_tc-style MoE router + shared expert) at tiny
dimensions, using the REAL `modeling_kimi.py`/`configuration_kimi.py` classes fetched from
moonshotai/Kimi-Linear-48B-A3B-Instruct -- not reimplemented. Saves weights+config into
`kimi_tiny/` and a greedy reference into `ref_kimi.json`.

Adapted from `tests/oracle/make_glm_oracle.py`'s own pattern (this repo's precedent for GLM-5.2),
with one necessary difference: Kimi Linear's real forward pass depends on `fla-core`'s
Triton-only kernels (ShortConvolution's causal conv1d, FusedRMSNormGated, chunk_kda/
fused_recurrent_kda, fused_kda_gate), which need a CUDA driver this machine doesn't have. This
script monkeypatches those FOUR kernel entry points onto plain-PyTorch equivalents BEFORE
constructing the model -- everything else (KimiMLAAttention, KimiDeltaAttention, KimiDecoderLayer,
KimiSparseMoeBlock, KimiLinearModel/ForCausalLM, the whole nn.Module wiring) is the REAL,
unmodified code. Three of the four patches wrap `fla`'s OWN documented CPU reference
implementations (`fla.ops.kda.naive.{naive_recurrent_kda,naive_chunk_kda}`,
`fla.ops.kda.gate.naive_kda_gate`) -- not reimplementations of the math, just compatibility
shims for `modeling_kimi.py`'s exact (and, for `fused_kda_gate`, installed-fla-core-version-
mismatched -- see PATCHES_NOTES below) calling convention. The fourth (`ShortConvolution`/
`FusedRMSNormGated`) are plain depthwise-conv1d and gated-RMSNorm reimplementations verified
against the real Triton kernels' documented formulas (see this session's research, captured in
rabbit-plan.md and this repo's own kda.rs/short_conv.rs/ops.rs doc comments).

PATCHES_NOTES: the installed `fla-core` (pip) release's `fused_kda_gate(g, A_log, dt_bias=None,
lower_bound=None, ...)` signature does NOT match modeling_kimi.py's actual call
(`fused_kda_gate(g, self.A_log, self.head_dim, g_bias=self.dt_bias)` -- a 3rd positional
`head_dim` and a `g_bias` kwarg neither exist in the installed fused_kda_gate) -- an API drift
between moonshotai's repo and the current fla-core release, not something specific to this
oracle. The compat shim below matches modeling_kimi.py's call, not the installed fused_kda_gate's.

Run (needs Docker -- system pip/venv is broken on this host):
    docker run --rm -v $(pwd):/work -w /work python:3.11-slim bash -c "
        pip install --index-url https://download.pytorch.org/whl/cpu torch
        pip install 'transformers==4.57.1' einops safetensors numpy packaging fla-core triton
        python3 make_kimi_oracle.py
    "
Then, from the repo root: cargo test --test teacher_forcing_kimi
"""

import json
import os

import torch
import transformers.utils as tu
from safetensors.torch import save_file


# auto_docstring's parameter-type-formatting helper crashes on `X | None` PEP604 annotations
# under this transformers/Python combination -- purely a docstring-generation decorator, no
# functional effect, so a no-op replacement is safe.
def _noop_auto_docstring(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]

    def deco(obj):
        return obj

    return deco


tu.auto_docstring = _noop_auto_docstring

from kimi_pkg.configuration_kimi import KimiLinearConfig  # noqa: E402
from kimi_pkg import modeling_kimi  # noqa: E402

# ---- patches: replace GPU/Triton-only kernels with plain-PyTorch equivalents ----
from fla.ops.kda.gate import naive_kda_gate  # noqa: E402
from fla.ops.kda.naive import naive_chunk_kda, naive_recurrent_kda  # noqa: E402


class NaiveShortConvolution(torch.nn.Conv1d):
    """Plain causal depthwise conv1d + activation -- same `nn.Conv1d` base class (so `self.weight`
    keeps the real checkpoint's `[d_inner, 1, kernel]` shape/name) as `fla.modules.ShortConvolution`,
    just without its Triton dispatch. No cache handling: this oracle only ever calls the model with
    `use_cache=False` (teacher-forcing and full-recompute greedy replay, see below), so `cache`/
    `cu_seqlens` are accepted for interface compatibility but never populated.
    """

    def __init__(self, hidden_size, kernel_size, bias=False, activation="silu", **kwargs):
        super().__init__(
            in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size,
            groups=hidden_size, bias=bias, padding=kernel_size - 1,
        )
        self.hidden_size = hidden_size
        self.activation = activation

    def forward(self, x, residual=None, mask=None, cache=None, output_final_state=False,
                cu_seqlens=None, chunk_indices=None, **kwargs):
        b, t, d = x.shape
        xt = x.transpose(1, 2)
        y = torch.nn.functional.conv1d(xt, self.weight, self.bias, padding=self.kernel_size[0] - 1, groups=d)
        y = y[..., :t].transpose(1, 2)
        if self.activation in ("silu", "swish"):
            y = torch.nn.functional.silu(y)
        return y, None


class NaiveRMSNormGated(torch.nn.Module):
    """`output = RMSNorm(x) * weight * sigmoid(g)` (for `activation="sigmoid"`) -- matches
    `fla.modules.fused_norm_gate`'s Triton kernel formula exactly (verified by reading its source
    this session: `b_y = b_x_hat * b_w; ... elif ACTIVATION == "sigmoid": b_y = b_y * sigmoid(b_g)`),
    just as plain PyTorch instead of a fused kernel.
    """

    def __init__(self, hidden_size, elementwise_affine=True, eps=1e-5, activation="swish", **kwargs):
        super().__init__()
        self.eps = eps
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, g, **kwargs):
        var = x.float().pow(2).mean(-1, keepdim=True)
        xhat = (x.float() * torch.rsqrt(var + self.eps)).to(x.dtype)
        y = xhat * self.weight
        if self.activation == "sigmoid":
            y = y * torch.sigmoid(g.float()).to(y.dtype)
        else:
            gf = g.float()
            y = y * (gf * torch.sigmoid(gf)).to(y.dtype)
        return y


def _apply_qk_l2norm(q, k, use_qk_l2norm_in_kernel):
    if use_qk_l2norm_in_kernel:
        q = torch.nn.functional.normalize(q, p=2, dim=-1, eps=1e-6)
        k = torch.nn.functional.normalize(k, p=2, dim=-1, eps=1e-6)
    return q, k


def patched_fused_recurrent_kda(q, k, v, g, beta, initial_state=None, output_final_state=False,
                                 use_qk_l2norm_in_kernel=False, cu_seqlens=None, **kwargs):
    q, k = _apply_qk_l2norm(q, k, use_qk_l2norm_in_kernel)
    return naive_recurrent_kda(q, k, v, g, beta, initial_state=initial_state, output_final_state=output_final_state)


def patched_chunk_kda(q, k, v, g, beta, initial_state=None, output_final_state=False,
                       use_qk_l2norm_in_kernel=False, cu_seqlens=None, **kwargs):
    q, k = _apply_qk_l2norm(q, k, use_qk_l2norm_in_kernel)
    return naive_chunk_kda(q, k, v, g, beta, initial_state=initial_state, output_final_state=output_final_state)


def patched_fused_kda_gate(g, A_log, head_dim=None, g_bias=None, lower_bound=None, output_dtype=torch.float32, **kwargs):
    # modeling_kimi.py passes g still FLAT ([..., H*K]) -- naive_kda_gate expects [..., H, K].
    # H comes from A_log's own element count (its shape is [1,1,H,1], not just "[H]").
    *lead, hk = g.shape
    h = A_log.numel()
    k = head_dim if head_dim is not None else hk // h
    g = g.view(*lead, h, k)
    return naive_kda_gate(g, A_log, dt_bias=g_bias, output_dtype=output_dtype)


modeling_kimi.ShortConvolution = NaiveShortConvolution
modeling_kimi.FusedRMSNormGated = NaiveRMSNormGated
modeling_kimi.chunk_kda = patched_chunk_kda
modeling_kimi.fused_recurrent_kda = patched_fused_recurrent_kda
modeling_kimi.fused_kda_gate = patched_fused_kda_gate
# ---- end patches ----

torch.manual_seed(1234)

cfg = KimiLinearConfig(
    vocab_size=64,
    hidden_size=16,
    num_hidden_layers=3,  # 0: KDA+dense, 1: KDA+MoE, 2: MLA+MoE
    num_attention_heads=2,
    num_key_value_heads=2,
    intermediate_size=10,  # dense MLP (layer 0)
    moe_intermediate_size=4,
    num_experts=4,
    num_experts_per_token=2,
    num_shared_experts=1,
    first_k_dense_replace=1,
    q_lora_rank=None,  # real checkpoint's own value -- Kimi Linear never uses Q-LoRA
    kv_lora_rank=8,
    qk_nope_head_dim=4,
    qk_rope_head_dim=2,
    v_head_dim=4,
    mla_use_nope=True,
    moe_renormalize=True,
    moe_router_activation_func="sigmoid",
    num_expert_group=1,
    topk_group=1,
    routed_scaling_factor=1.0,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
    tie_word_embeddings=False,
    linear_attn_config={
        "head_dim": 4, "num_heads": 2, "short_conv_kernel_size": 4,
        "kda_layers": [1, 2], "full_attn_layers": [3],
    },
)
cfg._attn_implementation = "eager"

model = modeling_kimi.KimiLinearForCausalLM(cfg).eval()
# KimiLinearModel.__init__ unconditionally forces flash_attention_2 (see modeling_kimi.py,
# "Ignoring the provided attention implementation") -- override AFTER construction; the MLA
# forward path reads config._attn_implementation live at call time, not a value cached at init.
model.config._attn_implementation = "eager"
model.model.config._attn_implementation = "eager"

with torch.no_grad():
    for n, p in model.named_parameters():
        if p.dim() >= 2:
            p.normal_(0, 0.05)
    for layer in model.model.layers:
        if hasattr(layer, "block_sparse_moe"):
            layer.block_sparse_moe.gate.e_score_correction_bias.copy_(
                torch.linspace(-0.1, 0.1, cfg.num_experts),
            )

print("=== state_dict tensors (names for the rabbit loader) ===")
for n, p in model.state_dict().items():
    print(f"  {n:60s} {tuple(p.shape)}")

prompt = [3, 14, 25, 6, 53, 8, 20, 11, 7, 40, 5, 9]  # arbitrary token ids, short seq
n_new = 20

# Greedy generation WITHOUT a KV cache: this oracle's patched ShortConvolution/kernels only
# implement the no-cache (full-recompute) path (see NaiveShortConvolution's doc) -- correct
# for teacher-forcing (a single forward over a fixed sequence) and, since KDA's recurrence and
# MLA's attention are both mathematically exact full recomputations here (not an approximation),
# also correct for greedy decode: re-running the whole growing prefix through an uncached forward
# every step is expensive but numerically IDENTICAL to real incremental decode with a cache
# (rabbit's own kda_layer_state_carries_across_sequential_single_token_steps test already proves
# rabbit's incremental decode matches its own prefill bit-for-bit, so this oracle only needs to
# validate PREFILL against the real model -- it doesn't need to also re-validate the cache path).
ids = list(prompt)
with torch.no_grad():
    for _ in range(n_new):
        lg = model(torch.tensor([ids]), use_cache=False).logits[0, -1]
        ids.append(int(lg.argmax(-1)))
full = ids
print("\nprompt:", prompt)
print("full  :", full)

with torch.no_grad():
    lg = model(torch.tensor([full]), use_cache=False).logits[0]  # [seq, vocab]
tf_pred = lg.argmax(-1).tolist()
print("tf_pred:", tf_pred)

os.makedirs("kimi_tiny", exist_ok=True)
sd = {name: tensor.contiguous() for name, tensor in model.state_dict().items()}
save_file(sd, "kimi_tiny/model.safetensors", metadata={"format": "pt"})
json.dump(cfg.to_dict(), open("kimi_tiny/config.json", "w"))
json.dump({"prompt_ids": prompt, "full_ids": full, "tf_pred": tf_pred}, open("ref_kimi.json", "w"))
print("\nsaved: kimi_tiny/ (weights+config) and ref_kimi.json")
