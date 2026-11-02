"""LoRA path of q_proj / v_proj: fused dropout with seed regeneration (Triton) around cuBLAS
matmuls, forward and backward, as one torch.autograd.Function.

What it computes (it must equal peft 0.20.0, `peft/tuners/lora/bnb.py` Linear4bit.forward,
lines 527-551, under torch.autocast fp16):

    result = base_layer(x).clone()
    result = result + lora_B(lora_A(dropout(x))) * scaling

with lora_A = A [r, in], lora_B = B [out, r] (no bias), scaling = lora_alpha / r = 2.0 and
dropout = nn.Dropout(p=lora_dropout) (= 0.05 for this task).

LoRA settings, dropout included -- the semantics are kept exactly:

* p is read from the peft module (`module.lora_dropout["default"].p`), never hard-coded.
  Dropout is applied only when that nn.Dropout module is in training mode and p > 0, exactly
  when peft applies it (eval mode or nn.Identity: no mask, plain x, no Triton dropout launch).
* Distribution: every element of the LoRA input is kept independently with probability
  1 - p and kept values are scaled by 1 / (1 - p); dropped values are 0. This is the same
  Bernoulli(1 - p) / (1 - p) law as torch.nn.functional.dropout.
* It is applied to the LoRA input only: the base (NF4) projection still receives the
  undropped x, as in peft, where `dropout(x)` feeds only lora_A.
* A fresh mask on every forward call: each call draws a new 63-bit Philox key from torch's
  CUDA default generator (`torch.randint(..., device="cuda")`, a one-element device tensor
  that the kernel reads with tl.load; no host sync). q_proj and v_proj therefore get
  independent masks, every layer and every step gets its own mask, and the stream follows the
  harness seed (torch.manual_seed seeds the CUDA generator), so runs are reproducible. The CPU
  generator is not touched, so the data order cannot change.
* The mask is a pure function of (key, element index): the backward pass regenerates the very
  same mask from the saved key. Neither the mask nor the dropped copy of x is stored between
  forward and backward. The tests check the keep rate, the scaling of kept values, that two
  forward calls give different masks and that forward and backward use the same mask.
* The individual random bits differ from torch's own dropout kernel (another Philox stream);
  only the distribution is the same, which is what dropout semantics require.

Math (dout [T, out] fp16; x_d = dropout(x)):
    xa    = x_d @ A^T                                 [T, r]      cuBLAS fp16
    out   = base + scaling * xa @ B^T                 [T, out]    cuBLAS addmm, in place on base
    dbase = dout
    dxa   = scaling * dout @ B                        [T, r]      cuBLAS
    dB    = scaling * dout^T @ xa                     [out, r]    cuBLAS
    dA    = dxa^T @ x_d                               [r, in]     cuBLAS, x_d regenerated
    dx    = (dxa @ A) * mask / (1 - p)                [T, in]     cuBLAS + Triton mask epilogue
The matmuls stay in cuBLAS: on sm_75 Triton 3.6 lowers tl.dot to FMA, not tensor cores.
The Triton kernels do the memory-bound parts: the dropout in forward, and in backward one pass that regenerates x_d (for dA) and applies the
mask and scale to dxa @ A (optionally accumulating into a given dx buffer, so a caller can
fuse the LoRA part of dx into the dx of the base projection). The fp32 -> fp16 cast of the
tiny A and B (r x in, out x r) is a plain torch .to(fp16), as peft's autocast does.
Weights are fp32 parameters in peft (cast_adapter_dtype, peft/tuners/tuners_utils.py:2243);
their grads are returned in the weight dtype.

Saved for backward: x (fp16, [T, in]; the RMSNorm output, the same storage for q_proj's and
v_proj's LoRA, so it is held once), xa (fp16, [T, r]), the fp16 copies of A and B (small)
and the 8-byte key.

Two autograd Functions: LoRAFunction (v_proj: the LoRA term added in place into the fresh base
output, which replaces peft's clone) and LoRADeltaFunction (q_proj: the LoRA term alone, added
to the base q inside the RoPE kernel, see norm_rope.rope(q_delta=...)).

Idea sources:
* Regenerating the dropout mask from a seed in backward instead of storing it: Triton tutorial
  "Low-Memory Dropout" (triton-lang.org, getting-started/tutorials/04-low-memory-dropout).
All kernels here are written from scratch.
"""

import torch
import triton
import triton.language as tl

# Fixed launch configs (no autotune: compile time counts). Element-wise kernels, chosen for
# the T4 (sm_75, 40 SMs): dropout 4096 elements per program = 4 quarters x 4 fp16 per
# thread with 8 warps (one 8-byte access per thread per quarter).
DROP_BLOCK, DROP_WARPS = 4096, 8

_KEY_HIGH = 2 ** 63 - 1


# Dropout mask layout (shared by forward and backward, which use the same BLOCK): the mask of
# element e is a pure function of (key, e). With Q = BLOCK // 4, element
# e = blk * BLOCK + j * Q + i (j in 0..3, i in 0..Q-1) uses uniform j of the one Philox-4x32
# evaluation at counter blk * Q + i: four uniforms per Philox call instead of one, and each
# quarter is a contiguous run of Q elements, so loads and stores stay vectorized.


@triton.jit
def _drop_quarter(x_ptr, xd_ptr, offs, n, keep, inv_keep):
    # Inlined into lora_dropout_fwd_kernel, never launched on its own.
    ok = offs < n
    xv = tl.load(x_ptr + offs, mask=ok, other=0.0).to(tl.float32)
    tl.store(xd_ptr + offs, tl.where(keep, xv * inv_keep, 0.0).to(xd_ptr.dtype.element_ty),
             mask=ok)


@triton.jit
def _drop_bwd_quarter(x_ptr, g_ptr, xd_ptr, dx_ptr, offs, n, keep, inv_keep, accumulate):
    # Inlined into lora_dropout_bwd_kernel, never launched on its own.
    ok = offs < n
    xv = tl.load(x_ptr + offs, mask=ok, other=0.0).to(tl.float32)
    tl.store(xd_ptr + offs, tl.where(keep, xv * inv_keep, 0.0).to(xd_ptr.dtype.element_ty),
             mask=ok)
    gv = tl.load(g_ptr + offs, mask=ok, other=0.0).to(tl.float32)
    dxv = tl.where(keep, gv * inv_keep, 0.0)
    if accumulate:
        dxv += tl.load(dx_ptr + offs, mask=ok, other=0.0).to(tl.float32)
    tl.store(dx_ptr + offs, dxv.to(dx_ptr.dtype.element_ty), mask=ok)


@triton.jit(do_not_specialize=["n", "p", "inv_keep"])
def lora_dropout_fwd_kernel(x_ptr, key_ptr, xd_ptr, n, p, inv_keep, BLOCK: tl.constexpr):
    """xd = x * keep / (1 - p), keep = (Philox uniform of the element) >= p."""
    Q: tl.constexpr = BLOCK // 4
    blk = tl.program_id(0)
    lane = tl.arange(0, Q)
    u0, u1, u2, u3 = tl.rand4x(tl.load(key_ptr), blk * Q + lane)
    offs = blk * BLOCK + lane
    _drop_quarter(x_ptr, xd_ptr, offs, n, u0 >= p, inv_keep)
    _drop_quarter(x_ptr, xd_ptr, offs + Q, n, u1 >= p, inv_keep)
    _drop_quarter(x_ptr, xd_ptr, offs + 2 * Q, n, u2 >= p, inv_keep)
    _drop_quarter(x_ptr, xd_ptr, offs + 3 * Q, n, u3 >= p, inv_keep)


@triton.jit(do_not_specialize=["n", "p", "inv_keep", "accumulate"])
def lora_dropout_bwd_kernel(x_ptr, g_ptr, key_ptr, xd_ptr, dx_ptr, n, p, inv_keep, accumulate,
                            BLOCK: tl.constexpr):
    """One pass over the same mask as the forward:
    xd = x * keep / (1 - p)                       (regenerated input of lora_A, for dA)
    dx = [dx +] g * keep / (1 - p),  g = dxa @ A  (LoRA part of the input gradient).
    dx_ptr may equal g_ptr (each element is read before it is written)."""
    Q: tl.constexpr = BLOCK // 4
    blk = tl.program_id(0)
    lane = tl.arange(0, Q)
    u0, u1, u2, u3 = tl.rand4x(tl.load(key_ptr), blk * Q + lane)
    offs = blk * BLOCK + lane
    _drop_bwd_quarter(x_ptr, g_ptr, xd_ptr, dx_ptr, offs, n, u0 >= p, inv_keep, accumulate)
    _drop_bwd_quarter(x_ptr, g_ptr, xd_ptr, dx_ptr, offs + Q, n, u1 >= p, inv_keep, accumulate)
    _drop_bwd_quarter(x_ptr, g_ptr, xd_ptr, dx_ptr, offs + 2 * Q, n, u2 >= p, inv_keep,
                      accumulate)
    _drop_bwd_quarter(x_ptr, g_ptr, xd_ptr, dx_ptr, offs + 3 * Q, n, u3 >= p, inv_keep,
                      accumulate)


# --------------------------------------------------------------------------------------------
# Python building blocks (also usable by a fused layer function later)
# --------------------------------------------------------------------------------------------

def new_dropout_key(device):
    """One fresh 63-bit Philox key from torch's CUDA generator, kept on the device (no sync)."""
    return torch.randint(0, _KEY_HIGH, (1,), device=device, dtype=torch.int64)


def cast_lora_weights(weight_a, weight_b):
    """fp16 copies of A [r, in] and B [out, r] (torch casts: the tensors are tiny)."""
    return weight_a.detach().to(torch.float16), weight_b.detach().to(torch.float16)


def _inv_keep(p):
    return 1.0 / (1.0 - p)


def lora_dropout(x, key, p):
    """dropout(x) for a contiguous fp16 x with the mask defined by key (p > 0)."""
    xd = torch.empty_like(x)
    n = x.numel()
    lora_dropout_fwd_kernel[(triton.cdiv(n, DROP_BLOCK),)](x, key, xd, n, float(p), _inv_keep(p),
                                                           BLOCK=DROP_BLOCK,
                                                           num_warps=DROP_WARPS)
    return xd


def lora_down_fwd(x, a16, key, p):
    """K3 forward: xa = dropout(x) @ A^T. x [T, in] fp16 contiguous -> xa [T, r] fp16."""
    if p > 0.0:
        return torch.mm(lora_dropout(x, key, p), a16.t())
    return torch.mm(x, a16.t())


def lora_down_bwd(x, a16, dxa, key, p, need_dx=True, dx_accum=None):
    """K3 backward. Returns (dx or None, dA fp32 [r, in]).

    dx_accum: optional contiguous [T, in] fp16 buffer (e.g. the dx of the base projection);
    when given, the LoRA part of dx is added into it in place and it is returned as dx."""
    if p > 0.0 and need_dx:
        n = x.numel()
        xd = torch.empty_like(x)
        g = torch.mm(dxa, a16)
        accumulate = dx_accum is not None
        dx = dx_accum if accumulate else g  # g is overwritten in place when not accumulating
        lora_dropout_bwd_kernel[(triton.cdiv(n, DROP_BLOCK),)](
            x, g, key, xd, dx, n, float(p), _inv_keep(p), int(accumulate),
            BLOCK=DROP_BLOCK, num_warps=DROP_WARPS)
    elif p > 0.0:
        xd = lora_dropout(x, key, p)  # only the regenerated x_d is needed (for dA)
        dx = None
    else:
        xd = x
        if need_dx:
            if dx_accum is not None:
                dx = dx_accum.addmm_(dxa, a16)
            else:
                dx = torch.mm(dxa, a16)
        else:
            dx = None
    da = _mm_f32(dxa.t(), xd)
    return dx, da


def lora_delta(xa, b16, scaling):
    """scaling * xa @ B^T without adding it (hook for fusing the add into the RoPE kernel)."""
    return torch.mm(xa, b16.t()).mul_(scaling)


def lora_up_bwd(dout, xa, b16, scaling):
    """K4 backward (dbase = dout is the caller's). Returns (dxa fp16 [T, r], dB fp32 [out, r])."""
    dxa = torch.mm(dout, b16).mul_(scaling)
    db = _mm_f32(dout.t(), xa).mul_(scaling)
    return dxa, db


_MM_OUT_DTYPE = [True]


def _mm_f32(a, b):
    """a @ b for fp16 a, b with an fp32 result: cuBLAS with fp32 output (torch.mm out_dtype,
    torch >= 2.9) when available, else an fp16 result cast to fp32 (what autocast gives)."""
    if _MM_OUT_DTYPE[0]:
        try:
            return torch.mm(a, b, out_dtype=torch.float32)
        except (RuntimeError, TypeError):
            _MM_OUT_DTYPE[0] = False
    return torch.mm(a, b).float()


class LoRAFunction(torch.autograd.Function):
    """out = base + scaling * (dropout(x) @ A^T) @ B^T, written into base's buffer.
    See the module docstring for the semantics and what is saved."""

    @staticmethod
    def forward(ctx, x, base, weight_a, weight_b, p, scaling):
        x2 = x.reshape(-1, x.shape[-1])
        if x2.dtype != torch.float16:
            x2 = x2.to(torch.float16)
        x2 = x2.contiguous()
        a16, b16 = cast_lora_weights(weight_a, weight_b)
        key = new_dropout_key(x2.device) if p > 0.0 else None
        xa = lora_down_fwd(x2, a16, key, p)
        # In place only on a plain fresh tensor: a view (e.g. F.linear's 3D-with-bias output
        # made inside another custom Function) must not be modified in place.
        if base.is_contiguous() and base.dtype == torch.float16 and not base._is_view():
            out = base
            out.view(-1, out.shape[-1]).addmm_(xa, b16.t(), alpha=scaling)
            ctx.mark_dirty(base)
        else:
            base2 = base.reshape(-1, base.shape[-1]).to(torch.float16)
            out = torch.addmm(base2, xa, b16.t(), alpha=scaling).view(base.shape)
        ctx.save_for_backward(x2, xa, a16, b16, key)
        ctx.p, ctx.scaling = p, scaling
        ctx.x_shape, ctx.x_dtype, ctx.base_dtype = x.shape, x.dtype, base.dtype
        ctx.w_dtypes = (weight_a.dtype, weight_b.dtype)
        return out

    @staticmethod
    def backward(ctx, dout):
        x2, xa, a16, b16, key = ctx.saved_tensors
        dout2 = dout.reshape(-1, dout.shape[-1])
        if dout2.dtype != torch.float16:
            dout2 = dout2.to(torch.float16)
        dout2 = dout2.contiguous()
        dxa, db = lora_up_bwd(dout2, xa, b16, ctx.scaling)
        need_dx = ctx.needs_input_grad[0]
        dx, da = lora_down_bwd(x2, a16, dxa, key, ctx.p, need_dx=need_dx)
        if dx is not None:
            dx = dx.view(ctx.x_shape).to(ctx.x_dtype)
        dbase = dout.to(ctx.base_dtype) if ctx.needs_input_grad[1] else None
        da = da.to(ctx.w_dtypes[0]) if ctx.needs_input_grad[2] else None
        db = db.to(ctx.w_dtypes[1]) if ctx.needs_input_grad[3] else None
        return dx, dbase, da, db, None, None


class LoRADeltaFunction(torch.autograd.Function):
    """delta = scaling * (dropout(x) @ A^T) @ B^T, the LoRA term alone (not added to anything).

    For q_proj: the caller adds it to the base projection inside the RoPE kernel
    (norm_rope.rope(q_delta=...)), whose backward hands the same gradient (dq before rotation)
    to the base projection and to this Function. Same math, dropout semantics and saved
    tensors as LoRAFunction (x, xa, the fp16 copies of A and B, the key); delta itself is not saved."""

    @staticmethod
    def forward(ctx, x, weight_a, weight_b, p, scaling):
        x2 = x.reshape(-1, x.shape[-1])
        if x2.dtype != torch.float16:
            x2 = x2.to(torch.float16)
        x2 = x2.contiguous()
        a16, b16 = cast_lora_weights(weight_a, weight_b)
        key = new_dropout_key(x2.device) if p > 0.0 else None
        xa = lora_down_fwd(x2, a16, key, p)
        delta = lora_delta(xa, b16, scaling)
        ctx.save_for_backward(x2, xa, a16, b16, key)
        ctx.p, ctx.scaling = p, scaling
        ctx.x_shape, ctx.x_dtype = x.shape, x.dtype
        ctx.w_dtypes = (weight_a.dtype, weight_b.dtype)
        return delta.view(*x.shape[:-1], delta.shape[-1])

    @staticmethod
    def backward(ctx, ddelta):
        x2, xa, a16, b16, key = ctx.saved_tensors
        d2 = ddelta.reshape(-1, ddelta.shape[-1])
        if d2.dtype != torch.float16:
            d2 = d2.to(torch.float16)
        d2 = d2.contiguous()
        dxa, db = lora_up_bwd(d2, xa, b16, ctx.scaling)
        dx, da = lora_down_bwd(x2, a16, dxa, key, ctx.p, need_dx=ctx.needs_input_grad[0])
        if dx is not None:
            dx = dx.view(ctx.x_shape).to(ctx.x_dtype)
        da = da.to(ctx.w_dtypes[0]) if ctx.needs_input_grad[1] else None
        db = db.to(ctx.w_dtypes[1]) if ctx.needs_input_grad[2] else None
        return dx, da, db, None, None


def lora_params(module, adapter="default"):
    """(A weight, B weight, dropout p in effect now, scaling) of a peft LoRA layer, or None if
    the layer uses anything this path does not implement (bias, DoRA/variants, several active
    adapters, merged or disabled adapters, a dropout that is not nn.Dropout/nn.Identity)."""
    if getattr(module, "disable_adapters", False) or getattr(module, "merged", False):
        return None
    if list(getattr(module, "active_adapters", [adapter])) != [adapter]:
        return None
    if adapter not in module.lora_A or adapter in getattr(module, "lora_variant", {}):
        return None
    lin_a, lin_b = module.lora_A[adapter], module.lora_B[adapter]
    if lin_a.bias is not None or lin_b.bias is not None:
        return None
    drop = module.lora_dropout[adapter]
    if isinstance(drop, torch.nn.Dropout):
        p = float(drop.p) if drop.training else 0.0
    elif isinstance(drop, torch.nn.Identity):
        p = 0.0
    else:
        return None
    if p >= 1.0:
        return None
    scaling = module.scaling[adapter]
    if isinstance(scaling, bool) or not isinstance(scaling, (int, float)):
        return None
    if getattr(module, "use_dora", {}).get(adapter, False):
        return None
    return lin_a.weight, lin_b.weight, p, float(module.scaling[adapter])


def lora_forward(module, x, base_out, adapter="default"):
    """base_out + the LoRA term of a peft LoRA layer on input x.

    x [..., in] (fp16 preferred), base_out [..., out] = base_layer(x) (a fresh tensor: it is
    updated in place, which replaces peft's defensive clone). Returns None when the layer is
    not supported; the caller then uses peft's own forward."""
    params = lora_params(module, adapter)
    if params is None or not x.is_cuda:
        return None
    weight_a, weight_b, p, scaling = params
    return LoRAFunction.apply(x, base_out, weight_a, weight_b, p, scaling)
