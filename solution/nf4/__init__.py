"""NF4 Linear4bit forward/backward on our Triton dequant kernel.

`patch()` replaces `bitsandbytes.nn.Linear4bit.forward` (class level, so peft's LoRA wrapper, which
calls `self.base_layer(x)`, picks it up). The new forward keeps bitsandbytes 0.50.2
`Linear4bit.forward` semantics -- x cast to `compute_dtype`, the bias cast the same way, the
output cast back to x's dtype -- and routes the matmul through `NF4Linear` (autograd.py) instead
of `bnb.matmul_4bit`. Anything it does not handle goes to the original forward unchanged.

Handled:
- `compute_dtype` fp16: as bitsandbytes, in fp16.
- `compute_dtype` fp32 (bitsandbytes switches a layer to fp32 on its first call when that call's
  input is fp32, which is what Qwen2's fp32 norms feed it after `prepare_model_for_kbit_training`)
  while CUDA autocast to fp16 is on (the Trainer's fp16 mode): bitsandbytes dequantizes to fp32 and
  autocast then casts x, W and bias to fp16 for the GEMM. Rounding the dequantized value straight
  to fp16 gives the same fp16 W, so we run that same fp16 GEMM without the fp32 round trip.

`linear4bit(x, module, keep_fp16=True)` (for fused callers): x fp16 (or fp32, cast to fp16 first);
returns the fp16 projection output bitsandbytes computes before its `.to(inp_dtype)` upcast, with
no upcast. Fallbacks return the original forward's result `.to(fp16)` (exact when bnb computed in
fp16). Autograd flows; grad_x comes back in x's dtype.

Idea sources: the forward reproduces bitsandbytes 0.50.2 `bitsandbytes.nn.Linear4bit.forward`
(bitsandbytes/nn/modules.py): x cast to `compute_dtype`, the in-place `bias.data` cast, the
output cast back with `.to(inp_dtype)`. Patching at class level relies on peft 0.20's LoRA
wrapper (peft/tuners/lora/bnb.py, `Linear4bit.forward`) calling `self.base_layer(x)`. Original
here: the fp32-compute-under-fp16-autocast shortcut, the fallback rules and the `keep_fp16`
contract. No code from bitsandbytes/peft/Unsloth/other projects is copied.

Fused projections sharing one input (q/k/v, gate/up): `linear4bit_multi(x, modules)` and
`dequant_multi(modules)`, see their docstrings. Idea sources: concatenating the weights of
projections that read the same input into one GEMM is the standard fused-QKV / fused gate-up
technique (e.g. Hugging Face transformers models with a fused `qkv_proj`/`gate_up_proj`, Megatron-LM's
fused QKV linear; Unsloth's fused LoRA MLP for doing it under 4-bit weights) -- taken as ideas
only. Original here: one dequant launch that decodes several NF4 weights straight into one
contiguous W_cat (no repack, no persistent memory), the re-dequantize-in-backward single dX GEMM
and the fallback rules. No code from those projects is copied.
"""

import torch

from .autograd import NF4Linear, NF4MultiLinear, nf4_linear
from .dequant import (MULTI_MAX, dequantize_nf4, dequantize_nf4_multi, multi_plan,
                      nf4_dequant_kernel, nf4_dequant_multi_kernel, supported, warmup)

__all__ = ["patch", "linear4bit", "linear4bit_multi", "dequant_multi", "dequantize_nf4",
           "dequantize_nf4_multi", "nf4_linear", "NF4Linear", "NF4MultiLinear",
           "nf4_dequant_kernel", "nf4_dequant_multi_kernel", "warmup"]

# Below this many rows bitsandbytes uses its own small-batch (gemv-like) kernels; leave them be.
MIN_ROWS = 8

_fp16 = torch.float16
_fp32 = torch.float32
_orig_forward = None


def _original(module, x):
    if _orig_forward is not None:
        return _orig_forward(module, x)
    from bitsandbytes.nn import Linear4bit
    return Linear4bit.forward(module, x)


def _fallback(module, x, keep_fp16):
    out = _original(module, x)
    return out.to(_fp16) if keep_fp16 else out


def _qs_ok(qs):
    ok = getattr(qs, "_soup_nf4_ok", None)
    if ok is None:
        ok = supported(qs)
        try:
            qs._soup_nf4_ok = ok
        except AttributeError:
            pass
    return ok


def linear4bit(x, module, keep_fp16=False):
    """bitsandbytes `Linear4bit.forward(module, x)` with the NF4 matmul on our kernels.

    keep_fp16=True: return the fp16 output (before bnb's `.to(x.dtype)` upcast) instead of x's
    dtype; x may be fp16 or fp32. Fallback paths return the original result `.to(fp16)`.
    """
    weight = module.weight
    qs = getattr(weight, "quant_state", None)
    if (qs is None or not module.compute_type_is_set or x.device.type != "cuda"
            or not _qs_ok(qs)):
        return _fallback(module, x, keep_fp16)
    K = x.shape[-1]
    if K != qs.shape[1] or x.numel() < MIN_ROWS * K:
        return _fallback(module, x, keep_fp16)

    cd = module.compute_dtype
    inp_dtype = x.dtype
    bias = module.bias
    if cd is _fp16:
        x = x.to(_fp16)
        if bias is not None and bias.dtype != _fp16:
            bias.data = bias.data.to(_fp16)  # bitsandbytes does this same in-place cast
    elif (cd is _fp32 and torch.is_autocast_enabled("cuda")
          and torch.get_autocast_dtype("cuda") is _fp16):
        x = x.to(_fp16)
        if bias is not None:
            if bias.dtype != _fp32:
                bias.data = bias.data.to(_fp32)  # as bitsandbytes
            bias = bias.to(_fp16)  # what autocast does inside F.linear
    else:
        return _fallback(module, x, keep_fp16)
    out = NF4Linear.apply(x, weight, bias, qs)
    return out if keep_fp16 else out.to(inp_dtype)


_GROUP_ATTR = "_soup_nf4_multi"


def _group_cache(modules):
    """Per-group cache dict, kept on the first module (keyed by the group's module ids)."""
    d = modules[0].__dict__.setdefault(_GROUP_ATTR, {})
    return d.setdefault(tuple(id(m) for m in modules), {})


def _plan(cache, states):
    key = tuple(id(qs) for qs in states)
    hit = cache.get("plan")
    if hit is None or hit[0] != key:
        hit = cache["plan"] = (key, multi_plan(states))
    return hit[1]


def _multi_states(modules, K=None):
    """The quant states when the one-launch dequant handles this group, else None."""
    if not 2 <= len(modules) <= MULTI_MAX:
        return None
    states = []
    for m in modules:
        qs = getattr(m.weight, "quant_state", None)
        if qs is None or not _qs_ok(qs):
            return None
        states.append(qs)
    K = states[0].shape[1] if K is None else K
    if any(qs.shape[1] != K for qs in states):
        return None
    return states


def dequant_multi(modules):
    """fp16 W_cat [sum(N_i), K] = cat of the modules' dequantized NF4 weights, row-major contiguous.

    One launch of `nf4_dequant_multi_kernel` for 2-3 Linear4bit modules with the same K whose
    states `dequantize_nf4` handles; each row block W_cat[N_0+..+N_{i-1} : ...] is bit-exact with
    `dequantize_nf4` of module i. Anything else: the same values from per-module dequants + cat
    (bitsandbytes' own dequant, cast to fp16, for states we do not handle). No autograd.
    """
    modules = tuple(modules)
    states = _multi_states(modules)
    if states is not None:
        return dequantize_nf4_multi([m.weight for m in modules], states,
                                    plan=_plan(_group_cache(modules), states))
    parts = []
    for m in modules:
        qs = m.weight.quant_state
        if _qs_ok(qs):
            parts.append(dequantize_nf4(m.weight, qs))
        else:
            import bitsandbytes.functional as BF
            parts.append(BF.dequantize_4bit(m.weight, qs).to(_fp16))
    return torch.cat(parts, 0)


def _multi_fallback(x, modules, split):
    outs = [linear4bit(x, m, keep_fp16=True) for m in modules]
    return tuple(outs) if split else torch.cat(outs, -1)


def linear4bit_multi(x, modules, split=True):
    """Fused `linear4bit(x, m, keep_fp16=True)` for 2-3 Linear4bit modules reading the same x.

    Values: `cat([linear4bit(x, m, keep_fp16=True) for m in modules], -1)` computed as ONE dequant
    launch (all weights into one W_cat) + ONE GEMM `F.linear(x16, W_cat, bias_cat)`, backward ONE
    re-dequant + ONE dX GEMM (`NF4MultiLinear`). x fp16 (or fp32, cast to fp16 first), fp16 out.
    split=True (default): a tuple of column views of Y_cat, one per module (autograd's split
    backward cats their grads into one dY_cat). split=False: Y_cat [..., sum(N_i)] itself.

    Fast-path conditions: per module those of `linear4bit` (our NF4 states, compute dtype fp16, or
    fp32 under fp16 CUDA autocast; at least MIN_ROWS rows), the same K, and either all or none of
    the modules with a bias. Bias semantics are linear4bit's (the same in-place `bias.data`
    casts); bias_cat is cached per group while the biases are frozen and unchanged (data pointer,
    version, dtype), and built per call when one needs a gradient. Anything else falls back to the
    separate linear4bit calls (a tuple of them, or their cat).
    """
    modules = tuple(modules)
    K = x.shape[-1]
    if (len(modules) < 2 or x.device.type != "cuda" or x.numel() < MIN_ROWS * K):
        return _multi_fallback(x, modules, split)
    states = _multi_states(modules, K)
    if states is None:
        return _multi_fallback(x, modules, split)
    autocast16 = (torch.is_autocast_enabled("cuda")
                  and torch.get_autocast_dtype("cuda") is _fp16)
    biases = []
    for m in modules:
        if not m.compute_type_is_set:
            return _multi_fallback(x, modules, split)
        cd = m.compute_dtype
        if not (cd is _fp16 or (cd is _fp32 and autocast16)):
            return _multi_fallback(x, modules, split)
        biases.append(m.bias)
    has_bias = [b is not None for b in biases]
    if any(has_bias) and not all(has_bias):
        return _multi_fallback(x, modules, split)

    cache = _group_cache(modules)
    bias_cat = None
    if biases[0] is not None:
        b16 = []
        for m, b in zip(modules, biases):
            if m.compute_dtype is _fp16:
                if b.dtype != _fp16:
                    b.data = b.data.to(_fp16)  # bitsandbytes does this same in-place cast
            elif b.dtype != _fp32:
                b.data = b.data.to(_fp32)  # as bitsandbytes (fp32 compute dtype)
            b16.append(b)
        if any(b.requires_grad for b in b16):
            bias_cat = torch.cat([b.to(_fp16) for b in b16])  # grads flow back to each bias
        else:
            key = tuple((b.data_ptr(), b._version, b.dtype) for b in b16)
            hit = cache.get("bias")
            if hit is None or hit[0] != key:
                with torch.no_grad():
                    hit = cache["bias"] = (key, torch.cat([b.to(_fp16) for b in b16]))
            bias_cat = hit[1]
    group = ([m.weight for m in modules], states, _plan(cache, states),
             None if bias_cat is None else bias_cat.dtype)
    y = NF4MultiLinear.apply(x.to(_fp16), bias_cat, group)
    if not split:
        return y
    return y.split([qs.shape[0] for qs in states], -1)


def _forward(self, x):
    return linear4bit(x, self)


_forward._soup_nf4 = True


def patch():
    """Route bitsandbytes Linear4bit through our NF4 kernels. Idempotent; False if no bnb."""
    global _orig_forward
    try:
        from bitsandbytes.nn import Linear4bit
    except Exception:
        return False
    current = Linear4bit.forward
    if getattr(current, "_soup_nf4", False):
        return True
    _orig_forward = current
    Linear4bit.forward = _forward
    return True
