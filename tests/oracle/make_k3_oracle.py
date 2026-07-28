"""Builds a MINISCULE Kimi K3 text backbone (`kimi_linear`, K3-shaped config) with random
weights, as an ORACLE.

K3's own `modeling_kimi_k3.py` contains ONLY the vision tower + a multimodal wrapper class --
its real text decoder is `modeling_kimi_linear.py`'s `KimiLinearForCausalLM`, the SAME classes
`tests/oracle/make_kimi_oracle.py` already validates against the smaller Kimi-Linear-48B-A3B
checkpoint. This script reuses that exact pattern, vendoring the REAL `moonshotai/Kimi-K3`
checkpoint's `configuration_kimi_k3.py`/`modeling_kimi_linear.py` (fetched 2026-07-27, under
`k3_pkg/` -- not reimplemented) with a tiny config that turns on every K3-only field rabbit's
`kimi_k3` module (`ops::situ_and_mul`, `moe::latent_moe`, `attn_res.rs`) was built against:
`hidden_act="situ"`, `routed_expert_hidden_size` (Stable LatentMoE), `attn_res_block_size`
(Attention Residuals), `mla_use_output_gate`, and KDA's `use_full_rank_gate`/`gate_lower_bound`.

**Why this script's patches differ from `make_kimi_oracle.py`'s**: the installed `fla-core`
(0.5.2)'s `chunk_kda`/`fused_recurrent_kda` entry points have a NEWER calling convention than the
48B oracle's -- K3's `modeling_kimi_linear.py` passes raw `A_log`/`dt_bias`/`beta` directly and
sets `use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
safe_gate=(gate_lower_bound is not None), lower_bound=gate_lower_bound` -- the kernel is now
expected to resolve L2-norm/beta-sigmoid/the decay gate ITSELF, instead of the caller
pre-resolving them (`fused_kda_gate` as its own separate call, which the 48B oracle patched, is
now `# deprecated` per this file's own import comment). The real Triton kernels fail outright in
this Docker image regardless (`RuntimeError('0 active drivers')`, even under
`TRITON_INTERPRET=1` for `chunk_kda`/`ShortConvolution`/`FusedRMSNormGated` specifically, though
NOT for `fused_recurrent_kda` alone -- inconsistent enough not to rely on), so this script
resolves L2-norm/beta-sigmoid/the gate itself in plain PyTorch (reusing `fla.ops.kda.naive`'s own
`naive_recurrent_kda` for the actual recurrence -- including for the `mode == 'chunk'` call site,
since `naive_chunk_kda` asserts `seq_len % chunk_size == 0` and this oracle's tiny sequences don't
divide evenly; chunked vs. recurrent are just two computation strategies for the SAME recurrence,
so this is exact, not an approximation -- and `fla.ops.kda.gate`'s own `naive_kda_gate`/
`naive_kda_lowerbound_gate` for the two gate formulas -- both plain-torch reference
implementations `fla` itself ships, not reimplementations) before calling those, same spirit as
`make_kimi_oracle.py`'s patches, adapted to the wider call signature.

**Real discovery this session, worth recording** (not previously known -- confirmed by reading
`fla.ops.kda.gate`'s actual installed source, not guessed): K3's `gate_lower_bound`/`safe_gate`
is NOT a clip/floor bolted onto the existing decay formula. It's a DIFFERENT formula entirely:
    no lower bound (existing rabbit `kda.rs::decay_gate`, unchanged): g = -exp(A_log) * softplus(g + dt_bias)
    with lower bound (K3's real checkpoint):                         g = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))
The second form is bounded in `(lower_bound, 0)` (sigmoid saturates), which is presumably the
actual point of the name -- the first form has no such floor. This directly informs the still-TODO
Rust port of K3's new attention gates.

Run (needs Docker -- system pip/venv is broken on this host):
    docker run --rm -v $(pwd):/work -w /work python:3.11-slim bash -c "
        pip install --index-url https://download.pytorch.org/whl/cpu torch
        pip install 'transformers==4.57.1' einops safetensors numpy packaging fla-core triton
        python3 make_k3_oracle.py
    "
"""

import json
import os

import torch
import transformers.utils as tu
from safetensors.torch import save_file


# Same no-op shim make_kimi_oracle.py needed -- auto_docstring's parameter-type-formatting
# helper crashes on `X | None` PEP604 annotations under this transformers/Python combination.
def _noop_auto_docstring(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]

    def deco(obj):
        return obj

    return deco


tu.auto_docstring = _noop_auto_docstring

from k3_pkg.configuration_kimi_k3 import KimiLinearConfig  # noqa: E402
from k3_pkg import modeling_kimi_linear  # noqa: E402

# ---- patches: replace GPU/Triton-only kernels with plain-PyTorch equivalents ----
from fla.ops.kda.gate import naive_kda_gate, naive_kda_lowerbound_gate  # noqa: E402
from fla.ops.kda.naive import naive_recurrent_kda  # noqa: E402


class NaiveShortConvolution(torch.nn.Conv1d):
    """Same as `make_kimi_oracle.py`'s class of the same name -- plain causal depthwise conv1d +
    activation, no cache handling (this oracle only ever calls with `use_cache=False`)."""

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
    """Same as `make_kimi_oracle.py`'s class of the same name -- `RMSNorm(x) * weight *
    sigmoid(g)`, matching `fla.modules.fused_norm_gate`'s documented formula."""

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


def _apply_qk_l2norm(q, k):
    q = torch.nn.functional.normalize(q, p=2, dim=-1, eps=1e-6)
    k = torch.nn.functional.normalize(k, p=2, dim=-1, eps=1e-6)
    return q, k


def _resolve_gate_and_beta(g, beta, A_log, dt_bias, lower_bound):
    """Mirrors what `use_gate_in_kernel=True`/`use_beta_sigmoid_in_kernel=True` do INSIDE the
    real (Triton) kernel -- see this module's doc for the two real gate formulas, read from
    `fla.ops.kda.gate`'s actual installed source, not guessed."""
    if lower_bound is None:
        g = naive_kda_gate(g, A_log, dt_bias=dt_bias)
    else:
        g = naive_kda_lowerbound_gate(g, A_log, dt_bias=dt_bias, lower_bound=lower_bound)
    beta = torch.sigmoid(beta.float())
    return g, beta


def patched_chunk_kda(q, k, v, g, beta, A_log=None, dt_bias=None, scale=None, initial_state=None,
                       output_final_state=False, use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
                       use_beta_sigmoid_in_kernel=False, safe_gate=False, lower_bound=None, **kwargs):
    if use_qk_l2norm_in_kernel:
        q, k = _apply_qk_l2norm(q, k)
    if use_gate_in_kernel:
        g, _ = _resolve_gate_and_beta(g, beta, A_log, dt_bias, lower_bound if safe_gate else None)
    if use_beta_sigmoid_in_kernel:
        beta = torch.sigmoid(beta.float())
    # `naive_chunk_kda` asserts seq_len % chunk_size == 0 (default chunk_size=64) -- this tiny
    # oracle's sequences don't divide evenly, and chunked vs. recurrent are just two different
    # computation strategies for the SAME recurrence (rabbit's own step()-vs-prefill equivalence
    # tests already lean on this same mathematical fact), so route through the unconditionally
    # correct recurrent form instead of hunting for a compatible chunk_size.
    return naive_recurrent_kda(q, k, v, g, beta, scale=scale, initial_state=initial_state, output_final_state=output_final_state)


def patched_fused_recurrent_kda(q, k, v, g, beta, A_log=None, dt_bias=None, scale=None, initial_state=None,
                                 output_final_state=False, use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
                                 use_beta_sigmoid_in_kernel=False, lower_bound=None, **kwargs):
    if use_qk_l2norm_in_kernel:
        q, k = _apply_qk_l2norm(q, k)
    if use_gate_in_kernel:
        g, _ = _resolve_gate_and_beta(g, beta, A_log, dt_bias, lower_bound)
    if use_beta_sigmoid_in_kernel:
        beta = torch.sigmoid(beta.float())
    return naive_recurrent_kda(q, k, v, g, beta, scale=scale, initial_state=initial_state, output_final_state=output_final_state)


modeling_kimi_linear.ShortConvolution = NaiveShortConvolution
modeling_kimi_linear.FusedRMSNormGated = NaiveRMSNormGated
modeling_kimi_linear.chunk_kda = patched_chunk_kda
modeling_kimi_linear.fused_recurrent_kda = patched_fused_recurrent_kda
# ---- end patches ----

torch.manual_seed(1234)

cfg = KimiLinearConfig(
    vocab_size=64,
    hidden_size=16,
    num_hidden_layers=4,  # 0: KDA+dense, 1: KDA+MoE, 2: MLA+MoE, 3: KDA+MoE
    num_attention_heads=2,
    num_key_value_heads=2,
    intermediate_size=10,  # dense MLP (layer 0)
    hidden_act="situ",
    activation_situ_beta=4.0,
    activation_situ_linear_beta=25.0,
    moe_intermediate_size=4,
    num_experts=4,
    num_experts_per_token=2,
    num_shared_experts=1,
    first_k_dense_replace=1,
    q_lora_rank=6,  # K3's real checkpoint has this NON-null, unlike Kimi Linear 48B
    kv_lora_rank=8,
    qk_nope_head_dim=4,
    qk_rope_head_dim=2,
    v_head_dim=4,
    mla_use_nope=True,
    mla_use_output_gate=True,  # K3-only: extra g_proj + output gate on MLA layers
    moe_renormalize=True,
    moe_router_activation_func="sigmoid",
    num_expert_group=1,
    topk_group=1,
    routed_scaling_factor=1.0,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
    tie_word_embeddings=False,
    max_position_embeddings=64,
    attn_res_block_size=2,  # K3-only: Attention Residuals, checkpoints at layer_idx 0 and 2
    routed_expert_hidden_size=8,  # K3-only: Stable LatentMoE, half of hidden_size
    latent_moe_use_norm=True,
    linear_attn_config={
        "head_dim": 4, "num_heads": 2, "short_conv_kernel_size": 4,
        "kda_layers": [1, 2, 4], "full_attn_layers": [3],
        "use_full_rank_gate": True,  # K3-only: full-rank output gate instead of low-rank g_a/g_b
        "gate_lower_bound": -5.0,  # K3-only: the bounded decay-gate formula (see module doc)
    },
)
cfg._attn_implementation = "eager"

model = modeling_kimi_linear.KimiLinearForCausalLM(cfg).eval()
# KimiLinearModel.__init__ unconditionally forces flash_attention_2 -- override AFTER
# construction, same as make_kimi_oracle.py (the MLA forward path reads config live at call
# time, not a value cached at init).
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

# Greedy generation WITHOUT a KV cache -- same reasoning as make_kimi_oracle.py: this oracle's
# patched kernels only implement the no-cache (full-recompute) path, which is exact (not an
# approximation) for both teacher-forcing and greedy decode since KDA's recurrence and MLA's
# attention are both mathematically exact full recomputations here.
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

os.makedirs("k3_tiny", exist_ok=True)
# The REAL published checkpoint is the full multimodal `KimiK3ForConditionalGeneration`, which
# wraps this exact text backbone as `self.language_model = KimiLinearForCausalLM(...)` -- every
# real tensor name carries a `language_model.` prefix this standalone `KimiLinearForCausalLM`
# instance's own state_dict does NOT have (confirmed against the real checkpoint's
# `model.safetensors.index.json`, fetched 2026-07-27). Prepending it here keeps this fixture's
# naming convention identical to a real checkpoint's, so `kimi_k3::model::Model::load` (which
# rightfully expects the prefix unconditionally) doesn't need a test-only special case.
sd = {f"language_model.{name}": tensor.contiguous() for name, tensor in model.state_dict().items()}
save_file(sd, "k3_tiny/model.safetensors", metadata={"format": "pt"})
# `cfg` is a flat KimiLinearConfig -- the real moonshotai/Kimi-K3/config.json nests this exact
# shape under `text_config` with `model_type: "kimi_k3"` at the top level (confirmed against the
# real checkpoint's config.json, fetched 2026-07-27); rabbit's kimi_k3::config::Cfg::load expects
# that nested shape, not a bare KimiLinearConfig dict.
k3_config = {"model_type": "kimi_k3", "text_config": cfg.to_dict()}
json.dump(k3_config, open("k3_tiny/config.json", "w"))
json.dump({"prompt_ids": prompt, "full_ids": full, "tf_pred": tf_pred}, open("ref_k3.json", "w"))
print("\nsaved: k3_tiny/ (weights+config) and ref_k3.json")
