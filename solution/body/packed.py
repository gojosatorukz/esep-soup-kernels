"""Pad removal inside one batch for the Qwen2 decoder (plan D1).

`Qwen2Model.forward` is class-patched. For a normal training batch (2D attention mask, no cache,
no captured outputs) the real tokens of all rows are packed into one `[1, T, H]` sequence, every
token-wise op of the 28 decoder layers (norms, q/k/v/o and MLP projections incl. NF4 + LoRA,
residual adds) runs on the T real tokens only, attention runs per row on that row's slice with a
causal mask, and the final hidden states are scattered back to `[B, L, H]` with zeros at the pad
positions. Anything unusual goes to the original forward.

Why this keeps the outputs: in the base, a real token attends to the real tokens before it in the
same row (causal mask AND padding mask), which is exactly causal attention on the packed slice of
that row; positions are the base positions gathered at the real tokens. Pad positions never reach
the loss, entropy, token accuracy or Soup's rewind row losses (all masked by labels == -100 or
attention_mask == 0), and zero hidden states give finite logits there.

The layer body is written out here (instead of calling `Qwen2DecoderLayer.forward`) so that each
piece is replaced by our own kernels in one place: `_add_norm` (K1 add_rmsnorm: residual add +
RMSNorm, fp32 residual, fp16 output for the projections), `_rope` (K2 rope from the packed
positions and the model's inv_freq; no cos/sin tensors), `_proj`, `_attention` (SDPA per row,
GQA by a 0-stride expand of k/v instead of repeat_interleave). Each falls back to the HF / torch
op when its inputs are not what the kernel expects (see `_norm_eps`, `_RopeInputs`);
in training, a guard that sends the norm kernel, the recompute MLP or the LoRA
path of a layer to the HF / torch op logs it once (`_fallback`), so partial fallbacks show in
the train log.
Projections (`_proj`, `_proj_q`): for bnb Linear4bit bases and peft LoRA Linear4bit wrappers the
base layer is called directly on the fp16 normed x (through the nf4 workstream's `linear4bit`
when that package is present) and the LoRA term runs on our kernels (solution/body/lora.py): v's
is added in place into the fresh base output, q's is passed to the RoPE kernel as `q_delta`.
Anything else is called as a module, as peft / bnb would run it (see `_route`).
When the nf4 package exports `linear4bit_multi`, the q/k/v base projections run as one fused call
on the shared normed x (`_qkv`: column views of one Y_cat; v's LoRA then added out of place), and
the recompute MLP's backward dequantizes gate and up in one `dequant_multi` launch.
Gradient fan-in: each norm returns one copy of its output per consumer (`_add_norm(n_out=...)`:
q/k/v base and q/v LoRA after the input norm; gate and up after the post-attention norm), so the
norm backward kernel sums the consumers' dx in fp32 instead of autograd adding them pairwise
with fp16 add kernels.
MLP (`_mlp`): with RECOMPUTE_MLP, a frozen NF4 Qwen2MLP runs as `_RecomputeMLP`, which saves no
[T, intermediate] tensor and recomputes gate/up (and its norm input x) in backward; otherwise
`layer.mlp(x)`. With NORM_SAVE_OUTPUT the decoder-layer norms save their fp16 output instead of
the fp32 residual (norm_rope.add_rmsnorm(save_output=True); idea: NVIDIA Apex PR #1715,
memory-efficient LayerNorm/RMSNorm), and the recompute MLP keeps that output as its input.
Idea source: activation recomputation / gradient checkpointing (Chen et al.
2016, "Training Deep Nets with Sublinear Memory Cost"), applied to one block by our own code.

Idea source: "padding-free" / unpadded training as in FlashAttention's variable-length mode and
HF's `DataCollatorWithFlattening`; the code here is our own.
"""

import contextlib
import logging

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

# Keyword arguments that may reach Qwen2Model.forward in a training step and that the packed
# path can ignore. Anything else -> original forward.
_IGNORED_KWARGS = {"num_items_in_batch", "return_dict", "output_attentions",
                   "output_hidden_states", "is_causal"}

_warned = set()


def _warn_once(reason):
    if reason not in _warned:
        _warned.add(reason)
        log.warning("solution.body.packed: original Qwen2Model.forward used (%s)", reason)


# Where the packed forward is: [training, layer index or "final"], set by packed_forward. The
# guards below report a fallback of one decoder piece through `_fallback` only in training.
_WHERE = [False, None]


def _fallback(piece, reason):
    """A per-module guard sent `piece` (norm kernel, recompute MLP, LoRA path) of the packed
    forward to the HF / torch op: one warning per (piece, reason) and process, in training
    only, through the same logger as the full fallback; the first layer seen is named. Returns
    None (guards do `return _fallback(...)`)."""
    if _WHERE[0]:
        key = f"{piece}: {reason}"
        if key not in _warned:
            _warned.add(key)
            log.warning("solution.body.packed: %s not used (%s; first at layer %s)", piece,
                        reason, _WHERE[1])
    return None


def _note_active():
    """One line per process on the first packed training forward, through the same logger and
    level as the fallback warning (INFO is not shown by default), so the train log shows that
    the packed path ran. Never per step."""
    if "active" not in _warned:
        _warned.add("active")
        log.warning("solution.body: packed decoder active (pad removal + own kernels)")


# ---------------------------------------------------------------------------------------------
# Swappable pieces: our Triton kernels (solution/body/norm_rope.py) where the inputs are what the
# kernels expect, HF / plain torch otherwise.
# ---------------------------------------------------------------------------------------------

# In-process switch for tests and benchmarks (packed + torch ops vs packed + our kernels). Not read
# from the environment; the training run always uses the default.
USE_KERNELS = True

_KERNEL_FLOATS = (torch.float16, torch.float32)


def _norm_eps(norm, residual, delta):
    """`variance_epsilon` if K1 add_rmsnorm can replace `norm(residual + delta)`, else None.

    The kernel computes in fp32 and gives the weight no gradient, so: an unmodified HF
    Qwen2RMSNorm without hooks and with a frozen [H] weight, an fp32 CUDA residual (the real
    run's residual stream; an fp16/bf16 residual keeps HF's rounding by falling back) and an
    fp16/fp32 delta of the same shape.
    """
    if not USE_KERNELS:
        return None
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    from .norm_rope import NORM_BLOCK

    if type(norm) is not Qwen2RMSNorm or _has_hooks(norm):
        return _fallback("norm kernel", "not a plain Qwen2RMSNorm or hooks")
    w = getattr(norm, "weight", None)
    eps = getattr(norm, "variance_epsilon", None)
    if not isinstance(w, torch.Tensor) or w.requires_grad or eps is None:
        return _fallback("norm kernel", "trainable or missing weight / eps")
    if w.dim() != 1 or w.dtype not in _KERNEL_FLOATS or not w.is_cuda:
        return _fallback("norm kernel", f"weight {w.dtype} {tuple(w.shape)}")
    if residual.dtype != torch.float32 or not residual.is_cuda or residual.shape[-1] != w.shape[0]:
        return _fallback("norm kernel", f"residual {residual.dtype} {residual.device}")
    if w.shape[0] > NORM_BLOCK:
        return _fallback("norm kernel", f"hidden {w.shape[0]} > {NORM_BLOCK}")
    if delta is not None and (delta.dtype not in _KERNEL_FLOATS or delta.shape != residual.shape
                              or delta.device != residual.device):
        return _fallback("norm kernel", f"delta {delta.dtype} {tuple(delta.shape)}")
    return float(eps)


# In-process switch for tests and benchmarks (not read from the environment; the run uses the
# default). True: the decoder-layer norms (a delta, fp16 output) save their fp16 output y for
# backward instead of the fp32 residual (norm_rope.add_rmsnorm(save_output=True): xhat = y / w,
# exact for the tiny-|w| channels), and the recompute MLP saves that y as its input instead of
# the residual. Frees two fp32 [T, H] residuals per layer (~11 MiB at T ~ 960). The final norm
# (fp32 output) and layer 0's input norm (no delta) are not affected.
NORM_SAVE_OUTPUT = True


def _add_norm(norm, residual, delta, final=False, with_eps=False, n_out=1, save_output=False):
    """`r = residual + delta` (delta may be None), returns (r, norm(r)); with_eps: also the eps
    the norm kernel used, or None when HF's norm ran (the MLP recompute needs to know).
    save_output: the kernel's save_output mode (see NORM_SAVE_OUTPUT); kernel path only.
    n_out > 1: norm(r) is returned as a tuple of n_out tensors, one per consumer. Kernel path:
    distinct outputs of the norm Function (views of one buffer), so the norm backward kernel
    sums the consumers' gradients; HF path: the same tensor n_out times (autograd sums).

    Kernel path: r stays fp32. The normed output of a decoder-layer norm only feeds projections,
    which compute in the autocast dtype (fp16 in the run), so it is stored in that dtype directly
    (the base computes it in fp32 and the projection casts it: same values). The final norm's
    output is returned in the dtype HF returns (promote(weight, residual) = fp32 in the run).
    """
    eps = _norm_eps(norm, residual, delta)
    if eps is not None:
        out_dtype = torch.promote_types(norm.weight.dtype, residual.dtype)
        if not final and torch.is_autocast_enabled("cuda"):
            out_dtype = torch.get_autocast_dtype("cuda")
        if out_dtype in _KERNEL_FLOATS:
            from .norm_rope import add_rmsnorm

            r, y = add_rmsnorm(residual, delta, norm.weight, eps, out_dtype, n_out,
                               save_output=save_output and not final)
            return (r, y, eps) if with_eps else (r, y)
    if delta is not None:
        residual = residual + delta
    y = norm(residual)
    if n_out > 1:
        y = (y,) * n_out
    return (residual, y, None) if with_eps else (residual, y)


# In-process switch for the LoRA path (our LoRA kernels around direct base-layer calls) vs peft's
# module forward; for tests and benchmarks only, like USE_KERNELS (which also turns it off).
USE_LORA_KERNELS = True

_NF4_LINEAR = []   # [solution.nf4.linear4bit or None], resolved on first use
_CLASSES = []      # [(bnb Linear4bit, peft lora.bnb.Linear4bit) or None]


def _linear4bit():
    """The nf4 workstream's `linear4bit(x, module, keep_fp16=True)` if that package is present."""
    if not _NF4_LINEAR:
        try:
            from ..nf4 import linear4bit
        except ImportError:
            linear4bit = None
        _NF4_LINEAR.append(linear4bit)
    return _NF4_LINEAR[0]


_NF4_MULTI = []    # [solution.nf4.linear4bit_multi or None], resolved on first use


def _linear4bit_multi():
    """The nf4 workstream's `linear4bit_multi(x, modules)` (fused q/k/v: one dequant launch + one
    GEMM, column views of one Y_cat) if that package is present."""
    if not _NF4_MULTI:
        try:
            from ..nf4 import linear4bit_multi
        except ImportError:
            linear4bit_multi = None
        _NF4_MULTI.append(linear4bit_multi)
    return _NF4_MULTI[0]


def _classes():
    if not _CLASSES:
        try:
            from bitsandbytes.nn import Linear4bit
            from peft.tuners.lora import bnb as peft_bnb
            pair = (Linear4bit, getattr(peft_bnb, "Linear4bit", None))
        except Exception:  # no bitsandbytes / peft, or a broken install: module path only
            pair = None
        _CLASSES.append(pair if pair is not None and pair[1] is not None else None)
    return _CLASSES[0]


def _base(base_layer, x2):
    """bnb Linear4bit `base_layer` on fp16 x2 [T, in] -> fresh fp16 [T, out]. 2D on purpose:
    F.linear with a bias on a 3D input returns a view, which the v LoRA add (in place) must not
    get. Frozen NF4 weight, nothing of x is saved for its backward (dx = dout @ W)."""
    linear4bit = _linear4bit()
    if linear4bit is not None:
        return linear4bit(x2, base_layer, keep_fp16=True)
    return base_layer(x2)


def _route(module, x):
    """How a q/k/v/o projection runs on the fp16 packed x [1, T, in]:
    ("lora", base_layer, (A, B, p, scaling))  peft LoRA Linear4bit wrapper, our LoRA path;
    ("base", base_layer, None)                bnb Linear4bit (plain, or a peft wrapper with no
                                              adapter of its own active), called directly;
    None                                      anything else: module(x) as before.

    Our path needs: kernels on, CUDA fp16 x under fp16 autocast (peft computes LoRA in the
    autocast dtype then; without autocast it casts to the LoRA weight dtype), exact classes
    without hooks, and for LoRA exactly the active 'default' adapter, not merged, adapters
    enabled, no DoRA / variants / bias, dropout nn.Dropout or nn.Identity, numeric scaling
    (see lora.lora_params)."""
    if not (USE_KERNELS and USE_LORA_KERNELS):
        return None
    if x.dtype != torch.float16 or not x.is_cuda:
        return _fallback("LoRA path", f"x {x.dtype} {x.device}")
    if not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != torch.float16:
        return _fallback("LoRA path", "no fp16 autocast")
    classes = _classes()
    if classes is None:
        return _fallback("LoRA path", "bitsandbytes / peft classes not found")
    if _has_hooks(module):
        return _fallback("LoRA path", f"hooks on {type(module).__name__}")
    bnb_cls, peft_cls = classes
    if type(module) is bnb_cls:
        return "base", module, None
    if type(module) is not peft_cls:
        return _fallback("LoRA path", f"projection is {type(module).__name__}")
    base = getattr(module, "base_layer", None)
    if type(base) is not bnb_cls or _has_hooks(base):
        return _fallback("LoRA path", f"base layer {type(base).__name__} or hooks")
    if getattr(module, "disable_adapters", False) or getattr(module, "merged", False):
        return _fallback("LoRA path", "adapters disabled or merged")
    if not any(a in module.lora_A for a in module.active_adapters):
        return "base", base, None  # what peft does then: base_layer(x) (+ a clone)
    from .lora import lora_params

    params = lora_params(module)
    if params is None:
        return _fallback("LoRA path", "adapter config (lora.lora_params)")
    return "lora", base, params


def _proj(module, x, x_lora=None):
    """A linear projection: x [1, T, in] -> [1, T, out]. v_proj (and any LoRA target) via our
    LoRA path: the LoRA term added in place into the fresh base output (peft's clone gone).
    x_lora: the same values as x as another output of the norm (a fan-out copy) for the LoRA
    input, so the LoRA dx and the base dx reach the norm backward separately (it sums them);
    unused (gets no gradient) when the LoRA path does not apply."""
    route = _route(module, x)
    if route is None:
        return module(x)
    kind, base, params = route
    x2 = x.reshape(-1, x.shape[-1])
    out = _base(base, x2)
    if kind == "lora":
        from .lora import LoRAFunction

        xl = x2 if x_lora is None else x_lora.reshape(-1, x_lora.shape[-1])
        out = LoRAFunction.apply(xl, out, *params)
    return out.view(1, x2.shape[0], -1)


def _proj_q(module, x, x_lora=None):
    """q_proj: (base q [1, T, out], LoRA delta [1, T, out] or None). The delta is not added here:
    the RoPE kernel adds it to q in fp16 before rotating (no extra pass), and its backward gives
    the same dq (pre-rotation) to the base projection and to the LoRA up/down backward. When the
    LoRA path does not apply, q is module(x) with LoRA included and the delta is None.
    x_lora: as in _proj (the LoRA input as its own fan-out copy of the norm output)."""
    route = _route(module, x)
    if route is None or route[0] != "lora":
        return _proj(module, x), None
    _, base, params = route
    from .lora import LoRADeltaFunction

    x2 = x.reshape(-1, x.shape[-1])
    q = _base(base, x2)
    xl = x2 if x_lora is None else x_lora.reshape(-1, x_lora.shape[-1])
    delta = LoRADeltaFunction.apply(xl, *params)
    return q.view(1, x2.shape[0], -1), delta.view(1, x2.shape[0], -1)


def _qkv(attn, x_q, x_ql, x_vl):
    """(q, q_delta, k, v) as `_proj_q` / `_proj` give them, with the three base projections
    fused into one nf4 `linear4bit_multi` call on x_q when it is present and the projections take
    our path today: q and v `_route` to "lora" or "base", k to "base" (a LoRA on k would need a
    fan-out copy of its own). Else None (the separate path runs).

    The fused base output is one Y_cat [T, Nq + Nk + Nv] (biases included); q, k, v are its
    column views. Its backward cats their grads into one dY_cat for one dX GEMM, so the norm
    backward gets ONE base gradient (through x_q) plus the q and v LoRA gradients (x_ql, x_vl)
    and sums them in its fp32 registers; the k and v base copies of the norm output get none.
    v's LoRA term is added out of place (LoRAFunction: base is a view, not a fresh tensor; an
    in-place add into a split view is not allowed and would not reach Y_cat's autograd)."""
    multi = _linear4bit_multi()
    if multi is None:
        return None
    rq = _route(attn.q_proj, x_q)
    rk = _route(attn.k_proj, x_q)
    rv = _route(attn.v_proj, x_q)
    if rq is None or rk is None or rv is None or rk[0] != "base":
        return None
    from .lora import LoRADeltaFunction, LoRAFunction

    x2 = x_q.reshape(-1, x_q.shape[-1])
    T = x2.shape[0]
    q, k, v = multi(x2, [rq[1], rk[1], rv[1]])
    q_delta = None
    if rq[0] == "lora":
        q_delta = LoRADeltaFunction.apply(x_ql.reshape(-1, x_ql.shape[-1]), *rq[2]).view(1, T, -1)
    if rv[0] == "lora":
        v = LoRAFunction.apply(x_vl.reshape(-1, x_vl.shape[-1]), v, *rv[2])
    return q.view(1, T, -1), q_delta, k.view(1, T, -1), v.view(1, T, -1)


class _RopeInputs:
    """What RoPE needs for one packed forward: our kernel's (positions [T] int64, inv_freq) when
    the rotary embedding is HF's default type with attention_scaling 1.0 (Qwen2.5), else HF's
    cos/sin, computed lazily (once) so the kernel path never builds them."""

    def __init__(self, model, hidden, pos):
        self.rotary, self.hidden, self.pos = model.rotary_emb, hidden, pos  # pos [1, T] int64
        self.positions = self.inv_freq = None
        self._cos_sin = None
        rot = self.rotary
        inv = getattr(rot, "inv_freq", None)
        hd = getattr(model.config, "head_dim", None) or \
            model.config.hidden_size // model.config.num_attention_heads
        if (USE_KERNELS and type(rot).__name__ == "Qwen2RotaryEmbedding" and not _has_hooks(rot)
                and getattr(rot, "rope_type", None) == "default"
                and getattr(rot, "attention_scaling", None) == 1.0
                and isinstance(inv, torch.Tensor) and inv.is_cuda and inv.dim() == 1
                and hd % 2 == 0 and inv.shape[0] == hd // 2 and (hd // 2) & (hd // 2 - 1) == 0):
            self.positions = pos.reshape(-1)  # one dtype (int64) for every call: one compile
            self.inv_freq = inv.detach().to(torch.float32).contiguous()

    def cos_sin(self):
        if self._cos_sin is None:
            self._cos_sin = self.rotary(self.hidden, self.pos)  # [1, T, D] each, hidden's dtype
        return self._cos_sin


def _rope(q, k, rope_in, q_delta=None):
    """RoPE on q [1, T, Hq, D] (+ q_delta [1, T, Hq * D], the q LoRA term) and k [1, T, Hk, D]
    -> same shapes. Kernel path: q + q_delta is formed in fp16 inside the RoPE kernel (as peft
    adds it in fp16), then rotated; otherwise it is added here first.
    """
    if (rope_in.positions is not None and q.dtype in _KERNEL_FLOATS and k.dtype == q.dtype
            and q.is_cuda and (q_delta is None or q_delta.dtype == q.dtype)):
        from .norm_rope import rope

        # squeeze(0), not [0]: a view whose backward is a view too (select's backward would
        # zero-fill a [1, T, ...] grad and copy into it).
        qo, ko = rope(q.squeeze(0), k.squeeze(0), rope_in.positions, rope_in.inv_freq,
                      q_delta=None if q_delta is None else q_delta.squeeze(0))
        return qo.unsqueeze(0), ko.unsqueeze(0)
    if q_delta is not None:
        q = q + q_delta.view_as(q)
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    cos, sin = rope_in.cos_sin()
    return apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)


def _attention(attn, q, k, v, lens):
    """Causal attention per row. q [1, T, Hq, D], k/v [1, T, Hk, D] -> [1, T, Hq * D].

    `lens` is the list of row lengths (host ints). GQA without materializing the kv repeat in
    the forward: per row the kv heads are the SDPA batch and the query group is the head dim,
    q [Hk, G, n, D] is a view of the packed q (query head h = kv_head * G + g, as in HF's
    repeat_kv) and k/v are expanded to [Hk, G, n, D] with a 0 stride (no copy; the attention
    backward makes them contiguous, the expand backward sums over the group). The outputs are
    concatenated straight from their strided views (one copy kernel for all rows).
    """
    if torch.is_autocast_enabled("cuda"):
        # Autocast would cast each row slice; casting once is the same values (no-op for fp16).
        dt = torch.get_autocast_dtype("cuda")
        q, k, v = q.to(dt), k.to(dt), v.to(dt)
    hq, hk, d = q.shape[2], k.shape[2], q.shape[3]
    groups = hq // hk
    dropout = attn.attention_dropout if attn.training else 0.0
    q0, k0, v0 = q.squeeze(0), k.squeeze(0), v.squeeze(0)  # views both ways (not select)
    if len(lens) == 1:
        rows = ((q0, k0, v0),)
    else:  # split: its backward is one cat per tensor (slicing would zero-fill + add per row)
        rows = zip(q0.split(lens), k0.split(lens), v0.split(lens))
    outs = []
    for qr, kr, vr in rows:
        n = qr.shape[0]
        qr = qr.reshape(n, hk, groups, d).permute(1, 2, 0, 3)          # [Hk, G, n, D]
        kr = kr.permute(1, 0, 2).unsqueeze(1).expand(hk, groups, n, d)  # stride 0 over G
        vr = vr.permute(1, 0, 2).unsqueeze(1).expand(hk, groups, n, d)
        o = F.scaled_dot_product_attention(qr, kr, vr, dropout_p=dropout, is_causal=True,
                                           scale=attn.scaling)
        outs.append(o.permute(2, 0, 1, 3))  # [n, Hk, G, D] view
    out = torch.cat(outs, dim=0) if len(outs) > 1 else outs[0].contiguous()
    return out.view(1, out.shape[0], hq * d)


# ---------------------------------------------------------------------------------------------
# MLP with gate/up recomputed in backward (activation memory)
# ---------------------------------------------------------------------------------------------

# Activation memory vs time. On by default: the T4 A/B (peak 3,693 -> 2,373 MiB, loop +7..10%) chose it.
# True: the MLP of each layer saves no [T, intermediate] tensor (the example SwiGLU saves gate
# and up: 2 * T * 8960 fp16, ~33 MiB per layer at T ~ 960) and recomputes gate/up in backward
# (2 more fp16 GEMMs per layer). Not read from the environment; tests and benchmarks flip it
# in-process.
RECOMPUTE_MLP = True

_FP16 = torch.float16
_NF4_DEQUANT = []  # [(solution.nf4.dequantize_nf4, supported) or None], resolved on first use


def _nf4_dequant():
    """The nf4 workstream's Triton dequant (bit-exact with bitsandbytes) if present."""
    if not _NF4_DEQUANT:
        try:
            from ..nf4 import dequantize_nf4, supported
            pair = (dequantize_nf4, supported)
        except ImportError:
            pair = None
        _NF4_DEQUANT.append(pair)
    return _NF4_DEQUANT[0]


_NF4_DEQUANT_MULTI = []  # [solution.nf4.dequant_multi or None], resolved on first use


def _nf4_dequant_multi():
    """The nf4 workstream's `dequant_multi(modules)` (one launch -> fp16 W_cat, bit-exact with
    the separate dequants) if present."""
    if not _NF4_DEQUANT_MULTI:
        try:
            from ..nf4 import dequant_multi
        except ImportError:
            dequant_multi = None
        _NF4_DEQUANT_MULTI.append(dequant_multi)
    return _NF4_DEQUANT_MULTI[0]


def _dequant(layer):
    """fp16 W [out, in] of a frozen bnb NF4 Linear4bit (a fresh tensor)."""
    weight = layer.weight
    qs = weight.quant_state
    nf4 = _nf4_dequant()
    if nf4 is not None and nf4[1](qs):
        return nf4[0](weight, qs)
    import bitsandbytes.functional as bnb_f

    return bnb_f.dequantize_4bit(weight.data, qs).to(_FP16)


def _mlp_layers(mlp, x):
    """(gate, up, down) when the recompute MLP applies to `mlp(x)`, else None: an unmodified
    Qwen2MLP with silu, exactly bnb NF4 Linear4bit projections without bias or hooks, no
    trainable parameter in the block, fp16 CUDA x under fp16 autocast (as `_route` requires;
    the projections then run in fp16) and grad mode on (otherwise there is nothing to save)."""
    if not RECOMPUTE_MLP or not torch.is_grad_enabled():
        return None
    why = "recompute MLP"
    if x.dtype != _FP16 or not x.is_cuda or x.dim() != 3:
        return _fallback(why, f"x {x.dtype} {x.device} {x.dim()}D")
    if not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != _FP16:
        return _fallback(why, "no fp16 autocast")
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    if type(mlp) is not Qwen2MLP or _has_hooks(mlp):
        return _fallback(why, f"{type(mlp).__name__} or hooks")
    if getattr(getattr(mlp, "config", None), "hidden_act", None) != "silu":
        return _fallback(why, "hidden_act is not silu")
    classes = _classes()
    if classes is None:
        return _fallback(why, "bitsandbytes / peft classes not found")
    layers = (getattr(mlp, "gate_proj", None), getattr(mlp, "up_proj", None),
              getattr(mlp, "down_proj", None))
    for lin in layers:
        if type(lin) is not classes[0] or _has_hooks(lin) or lin.bias is not None:
            return _fallback(why, f"projection {type(lin).__name__}, hooks or bias")
        qs = getattr(lin.weight, "quant_state", None)
        if qs is None or getattr(qs, "quant_type", None) != "nf4" or lin.weight.requires_grad:
            return _fallback(why, "weight not frozen NF4")
    if any(p.requires_grad for p in mlp.parameters()):
        return _fallback(why, "trainable parameter in the MLP")
    return layers


def _swiglu_fwd(g, u):
    """h = silu(g) * u in place into g (fp16 [T, I], contiguous). The organizers' example
    kernel (elementwise: each element is read before it is written, so in place is safe), or
    torch if the example module is gone. Not our kernel and not counted as one."""
    try:
        import triton

        from ..swiglu import BLOCK, example_swiglu_fwd_kernel
    except ImportError:
        return F.silu(g).mul_(u)
    n = g.numel()
    example_swiglu_fwd_kernel[(triton.cdiv(n, BLOCK),)](g, u, g, n, BLOCK=BLOCK)
    return g


def _swiglu_bwd(dh, g, u):
    """(dg, du) of h = silu(g) * u, written in place into g and u (same element-wise argument);
    the example kernel, or torch."""
    try:
        import triton

        from ..swiglu import BLOCK, example_swiglu_bwd_kernel
    except ImportError:
        sig = torch.sigmoid(g.float())
        gf, d = g.float(), dh.float()
        du = d * gf * sig
        dg = d * u.float() * sig * (1.0 + gf * (1.0 - sig))
        g.copy_(dg)
        u.copy_(du)
        return g, u
    n = g.numel()
    example_swiglu_bwd_kernel[(triton.cdiv(n, BLOCK),)](dh, g, u, g, u, n, BLOCK=BLOCK)
    return g, u


class _RecomputeMLP(torch.autograd.Function):
    """down(silu(gate(x)) * up(x)) for frozen NF4 projections, saving nothing of size
    [T, intermediate].

    forward: the projections run exactly as in the no-recompute path (`_base`: nf4's linear4bit
    or the bnb layer, fp16 out), the SwiGLU on the example kernel; g/u/h are freed on return.
    Saved: x itself (fp16), the default (with NORM_SAVE_OUTPUT the norm Function saves this
    very y, so it costs no memory and no recompute); or, when r fp32 is given (x is the output
    of our norm kernel in its r_out-saving mode), only r and the fp32 norm weight -- r is the
    tensor that norm Function saves -- and x is recomputed in backward with the same kernel
    (bit-exact).
    backward, with each 4-bit weight dequantized once and freed after its last use:
        dh = dout @ W_down
        g = x @ W_gate^T, u = x @ W_up^T            (recompute, the dequantized W reused below)
        dg, du = swiglu_bwd(dh, g, u)                (example kernel, in place into g, u)
        dx_gate = dg @ W_gate, dx_up = du @ W_up     (fp16, two GEMMs; returned separately as
                                                      the grads of x_gate and x_up)
    x_gate and x_up are two fan-out copies of the norm output (the same values, see
    _add_norm(n_out=2)): the norm backward kernel sums the two grads (no add kernel here or in
    autograd). Called with one tensor for both, autograd sums them as it would for mlp(x).
    The recompute equals the forward values bit-for-bit when the forward's projection is
    dequant + F.linear (nf4's linear4bit always; bitsandbytes above its custom-kernel M range).
    """

    @staticmethod
    def forward(ctx, x, x_up, r, w_norm, eps, gate, up, down):
        # x_up holds the same values as x (a second copy of the norm output): computing from x
        # alone is the same result; x_up only gives the up projection's dx its own slot.
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        g = _base(gate, x2).contiguous()
        u = _base(up, x2).contiguous()
        h = _swiglu_fwd(g, u)
        del u
        out = _base(down, h)
        del g, h
        if r is not None:
            ctx.save_for_backward(r, w_norm)
        else:
            ctx.save_for_backward(x2)
        ctx.recompute_x = r is not None
        ctx.eps, ctx.shape, ctx.layers = eps, shape, (gate, up, down)
        return out.view(*shape[:-1], out.shape[-1])

    @staticmethod
    def backward(ctx, dout):
        gate, up, down = ctx.layers
        if ctx.recompute_x:
            from .norm_rope import rmsnorm_recompute

            r, w_norm = ctx.saved_tensors
            x2 = rmsnorm_recompute(r.reshape(-1, r.shape[-1]), w_norm, ctx.eps, _FP16)
        else:
            (x2,) = ctx.saved_tensors
        dout2 = dout.reshape(-1, dout.shape[-1]).to(_FP16)
        w = _dequant(down)
        dh = torch.matmul(dout2, w)
        del w, dout2
        multi = _nf4_dequant_multi()
        if multi is not None:
            # gate and up in one dequant launch: W_cat [I_gate + I_up, H], row blocks bit-exact
            # with the separate dequants; dx = dg @ W_gate + du @ W_up as mm + addmm into one
            # buffer, returned as x's grad (x_up gets none: the norm backward gets one grad).
            w_gu = multi([gate, up])
            ng = gate.weight.quant_state.shape[0]
            w_gate, w_up = w_gu[:ng], w_gu[ng:]
            g = F.linear(x2, w_gate)
            u = F.linear(x2, w_up)
            del x2
            dg, du = _swiglu_bwd(dh, g, u)
            del dh
            dx = torch.mm(dg, w_gate)
            del dg, g
            dx.addmm_(du, w_up)
            del du, u, w_gate, w_up, w_gu
            return dx.view(ctx.shape), None, None, None, None, None, None, None
        w_gate = _dequant(gate)
        g = F.linear(x2, w_gate)
        w_up = _dequant(up)
        u = F.linear(x2, w_up)
        del x2
        dg, du = _swiglu_bwd(dh, g, u)
        del dh
        dx_gate = torch.matmul(dg, w_gate)
        del dg, g, w_gate
        dx_up = torch.matmul(du, w_up)
        del du, u, w_up
        return dx_gate.view(ctx.shape), dx_up.view(ctx.shape), None, None, None, None, None, None


def _fanout_forward(mlp, x):
    """f(x_gate, x_up) -> mlp(x) with the gate and up projections fed from two copies of x, when
    Qwen2MLP.forward is the organizers' example SwiGLU patch (solution/swiglu.py) or HF's own
    forward and would run its silu path; else None. Same modules and ops as that forward, only
    each of gate/up reads its own copy of the normed x, so their two dx reach the norm backward
    separately (summed there, no autograd add)."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    if type(mlp) is not Qwen2MLP or _has_hooks(mlp) or not x.is_cuda:
        return None
    if getattr(getattr(mlp, "config", None), "hidden_act", None) != "silu":
        return None
    fwd = type(mlp).forward
    name = (getattr(fwd, "__module__", None), getattr(fwd, "__qualname__", None))
    try:
        from .. import swiglu as example
    except ImportError:  # no example module: only HF's own forward is recognized
        example = None

    if example is not None and name == (example.__name__, "patch_qwen2_mlp.<locals>.forward"):
        return lambda xg, xu: mlp.down_proj(example.swiglu(mlp.gate_proj(xg), mlp.up_proj(xu)))
    if name == (Qwen2MLP.__module__, "Qwen2MLP.forward"):
        return lambda xg, xu: mlp.down_proj(mlp.act_fn(mlp.gate_proj(xg)) * mlp.up_proj(xu))
    return None


def _mlp(mlp, x, norm, residual, eps, norm_saves_y=False):
    """layer.mlp(x), through _RecomputeMLP when it applies. `eps` is not None when x came from
    our norm kernel as norm(residual): then x is recomputed from residual in backward, unless
    norm_saves_y (the norm ran in save_output mode and saves x itself: x is saved here too).
    x: a tensor, or a pair (x_gate, x_up) of fan-out copies of the norm output (same values),
    one per projection, whose grads the norm backward sums; mlp(x) ignores the second."""
    xg, xu = x if isinstance(x, tuple) else (x, x)
    layers = _mlp_layers(mlp, xg)
    if layers is None:
        fan = _fanout_forward(mlp, xg) if xu is not xg else None
        return mlp(xg) if fan is None else fan(xg, xu)
    r = w_norm = None
    if (eps is not None and not norm_saves_y and residual.dtype == torch.float32
            and residual.is_contiguous()):
        r, w_norm = residual, norm.weight.detach().to(torch.float32)
    return _RecomputeMLP.apply(xg, xu, r, w_norm, eps, *layers)


def _layer(layer, residual, delta, rope_in, lens):
    """One Qwen2DecoderLayer on packed tokens. Returns (residual, delta) where the layer output
    is residual + delta (the add is left pending so it can fuse with the next norm)."""
    attn = layer.self_attn
    save_y = NORM_SAVE_OUTPUT
    # The normed x feeds up to 5 consumers (q, k, v base projections, q and v LoRA inputs): one
    # fan-out copy each, so their dx are summed inside the norm backward kernel. A copy whose
    # consumer is not used (no LoRA path; with the fused q/k/v base, x_k and x_v) simply gets
    # no gradient (the kernel sums only the grads it gets: n_dy is a runtime value).
    residual, (x_q, x_ql, x_k, x_v, x_vl) = _add_norm(layer.input_layernorm, residual, delta,
                                                      n_out=5, save_output=save_y)
    n = x_q.shape[1]
    hd = attn.head_dim
    fused = _qkv(attn, x_q, x_ql, x_vl)
    if fused is not None:  # one fused base GEMM on x_q (x_k, x_v unused: no gradient)
        q, q_delta, k, v = fused
    else:
        q, q_delta = _proj_q(attn.q_proj, x_q, x_ql)
        k = _proj(attn.k_proj, x_k)
        v = _proj(attn.v_proj, x_v, x_vl)
    q = q.view(1, n, -1, hd)
    k = k.view(1, n, -1, hd)
    v = v.view(1, n, -1, hd)
    q, k = _rope(q, k, rope_in, q_delta)
    o = _proj(attn.o_proj, _attention(attn, q, k, v, lens))
    norm = layer.post_attention_layernorm
    residual, xs, eps = _add_norm(norm, residual, o, with_eps=True, n_out=2,  # gate, up
                                  save_output=save_y)
    return residual, _mlp(layer.mlp, xs, norm, residual, eps, norm_saves_y=save_y)


# ---------------------------------------------------------------------------------------------
# The patched forward
# ---------------------------------------------------------------------------------------------

def _has_hooks(module):
    return bool(module._forward_hooks or module._forward_pre_hooks
                or getattr(module, "_forward_hooks_with_kwargs", None))


def _unsupported(model, input_ids, attention_mask, past_key_values, inputs_embeds, use_cache,
                 kwargs):
    """None if the packed path applies, else a short reason."""
    if not isinstance(model, torch.nn.Module) or not hasattr(model, "layers"):
        return "not a real model"
    config = getattr(model, "config", None)
    if config is None:
        return "no config"
    if inputs_embeds is not None:
        return "inputs_embeds"
    if not isinstance(input_ids, torch.Tensor) or input_ids.dim() != 2 or not input_ids.is_cuda:
        return "input_ids not 2D CUDA"
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != input_ids.shape:
        return "attention_mask missing or not 2D"
    if past_key_values is not None:
        return "past_key_values"
    if use_cache is None:
        use_cache = getattr(config, "use_cache", False)
    if use_cache:
        return "use_cache"
    for key, value in kwargs.items():
        if key not in _IGNORED_KWARGS:
            return f"kwarg {key}"
    for key in ("output_attentions", "output_hidden_states"):
        if kwargs.get(key, getattr(config, key, False)):
            return key
    if getattr(config, "_attn_implementation", None) != "sdpa":
        return "attn_implementation is not sdpa"
    if any(t != "full_attention" for t in getattr(config, "layer_types", ["full_attention"])):
        return "sliding window layers"
    if getattr(model, "gradient_checkpointing", False) and model.training:
        return "gradient checkpointing"
    for layer in model.layers[: config.num_hidden_layers]:
        if _has_hooks(layer) or _has_hooks(layer.self_attn):
            return "hooks on decoder layers"
    return None


_LAYER_SCHED = []  # [(layer, boundary)], resolved on first use


def _no_offload_layer(i, n_layers):
    return contextlib.nullcontext()


def _no_offload_boundary(residual, delta, i):
    return residual, delta


def _layer_sched():
    """The head workstream's activation-offload front-ends (solution/layer_sched.py):
    `layer(i, n_layers)`, a context manager around decoder layer i, and
    `boundary(residual, delta, i) -> (residual, delta)` after it. Without that module: a no-op
    context and the identity."""
    if not _LAYER_SCHED:
        try:
            from .. import layer_sched
            pair = (layer_sched.layer, layer_sched.boundary)
        except ImportError:
            pair = (_no_offload_layer, _no_offload_boundary)
        _LAYER_SCHED.append(pair)
    return _LAYER_SCHED[0]


def packed_forward(model, input_ids, attention_mask, position_ids, kwargs):
    B, L = input_ids.shape
    mask = attention_mask.bool()
    lens = mask.sum(1).tolist()  # the one host sync of the forward
    if min(lens) == 0:
        return None
    T = sum(lens)
    flat_mask = mask.reshape(-1)
    idx = torch.nonzero_static(flat_mask, size=T).squeeze(1)  # row-major: rows stay in order

    if position_ids is None:
        pos = idx % L  # HF default: arange(L) for every row
    else:
        pos = position_ids.expand(B, L).reshape(-1).index_select(0, idx)
    pos = pos.to(torch.int64).unsqueeze(0)

    hidden = model.embed_tokens(input_ids.reshape(-1).index_select(0, idx).unsqueeze(0))
    _WHERE[0], _WHERE[1] = bool(model.training), "embedding"
    weight = getattr(model.embed_tokens, "weight", None)
    if (hidden.dtype != torch.float32 and isinstance(weight, torch.Tensor)
            and weight.dtype == torch.float32):
        _fallback("fp32 residual stream", f"embedding output is {hidden.dtype}")
    rope_in = _RopeInputs(model, hidden, pos)

    residual, delta = hidden, None
    offload_layer, offload_boundary = _layer_sched()
    layers = model.layers[: model.config.num_hidden_layers]
    for i, layer in enumerate(layers):
        _WHERE[1] = i
        with offload_layer(i, len(layers)):
            residual, delta = _layer(layer, residual, delta, rope_in, lens)
        residual, delta = offload_boundary(residual, delta, i)
    _WHERE[1] = "final norm"
    _, out = _add_norm(model.norm, residual, delta, final=True)
    _WHERE[0] = False

    # In place into the fresh zeros (a non-leaf without grad: autograd records the copy, the
    # source grad is grad.index_select(0, idx)); out of place would hold two [B*L, H] buffers.
    full = out.new_zeros(B * L, out.shape[-1])
    full.index_copy_(0, idx, out.squeeze(0))
    return full.view(B, L, -1)


def patch():
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2Model
    if getattr(cls.forward, "_solution_body_packed", False):
        return
    original = cls.forward

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
        reason = _unsupported(self, input_ids, attention_mask, past_key_values, inputs_embeds,
                              use_cache, kwargs)
        hidden = None
        if reason is None:
            hidden = packed_forward(self, input_ids, attention_mask, position_ids, kwargs)
            if hidden is None:
                reason = "a row without tokens"
        if hidden is not None and getattr(self, "training", False):
            _note_active()
        if hidden is None:
            if getattr(self, "training", False) and reason != "not a real model":
                _warn_once(reason)
            return original(self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        out = BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=None)
        return_dict = kwargs.get("return_dict", getattr(self.config, "return_dict", True))
        return out.to_tuple() if return_dict is False else out

    forward._solution_body_packed = True
    forward.__wrapped__ = original
    cls.forward = forward
