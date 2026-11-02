"""Our replacement for bitsandbytes' `MatMul4Bit` on the NF4 training path.

Same math as bitsandbytes 0.50.2 `MatMul4Bit` on its dequant + `F.linear` route (which is the
route its CUDA `gemm_4bit` dispatch takes at training batch sizes): forward
`F.linear(x, dequant(W), bias)`, backward `grad_x = grad_out @ dequant(W)`; the frozen 4-bit
weight gets no gradient. The weight is dequantized again in backward instead of being kept, so
peak memory is unchanged. The dequantization is our Triton kernel (`dequant.py`).

Idea sources: the math and route are bitsandbytes 0.50.2 `MatMul4Bit`
(bitsandbytes/autograd/_functions.py) when its `torch.ops.bitsandbytes.gemm_4bit` call lands on
the CUDA dequant + `F.linear` fallback (bitsandbytes/backends/cuda/ops.py,
`_dequant_linear_fallback`, always for M > 1536). Re-dequantizing in backward and saving only the
packed weight (no activations) is `MatMul4Bit.backward`'s own approach. A frozen 4-bit base under
trained LoRA adapters is QLoRA (Dettmers et al. 2023, https://arxiv.org/abs/2305.14314). Original
here: the single dequant kernel with the nested absmax fused in, and always taking that route. No
code from bitsandbytes/Unsloth/other projects is copied.
"""

import torch
import torch.nn.functional as F

from .dequant import dequantize_nf4, dequantize_nf4_multi


class NF4Linear(torch.autograd.Function):
    """y = x @ dequant(W).T + bias, with x and the dequantized W in fp16."""

    @staticmethod
    def forward(ctx, x, packed, bias, quant_state):
        w = dequantize_nf4(packed, quant_state)
        out = F.linear(x, w, bias)
        ctx.packed = packed
        ctx.quant_state = quant_state
        ctx.bias_dtype = None if bias is None else bias.dtype
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_x = grad_bias = None
        if ctx.needs_input_grad[2]:
            grad_bias = grad_out.reshape(-1, grad_out.shape[-1]).sum(0, dtype=ctx.bias_dtype)
        if ctx.needs_input_grad[0]:
            w = dequantize_nf4(ctx.packed, ctx.quant_state)
            if grad_out.dtype != w.dtype:
                w = w.to(grad_out.dtype)
            grad_x = torch.matmul(grad_out, w)
        return grad_x, None, grad_bias, None


def nf4_linear(x, packed, bias, quant_state):
    """fp16 x (any leading dims) -> fp16 y. bias is None or fp16."""
    return NF4Linear.apply(x, packed, bias, quant_state)


class NF4MultiLinear(torch.autograd.Function):
    """Y_cat = x @ W_cat.T + bias_cat for 2-3 NF4 weights sharing x (fused q/k/v or gate/up).

    W_cat = cat(dequant(W_i), 0) comes from one `dequantize_nf4_multi` launch, so forward is one
    dequant + one GEMM, and backward re-dequantizes W_cat (one launch again) for ONE dX GEMM
    grad_x = dY_cat @ W_cat, instead of n dequants + n GEMMs + (n-1) adds of the separate path.
    Saves only references to the packed weights and their states (as NF4Linear).
    `group` = (weights, quant_states, plan, bias_dtype) and is not differentiable.
    """

    @staticmethod
    def forward(ctx, x, bias_cat, group):
        weights, states, plan, _ = group
        w = dequantize_nf4_multi(weights, states, plan=plan)
        ctx.group = group
        return F.linear(x, w, bias_cat)

    @staticmethod
    def backward(ctx, grad_out):
        # grad_out may be any strided [..., sum(N)] (autograd's split backward gives a fresh
        # contiguous cat; other callers may not): matmul and reshape handle strides.
        weights, states, plan, bias_dtype = ctx.group
        grad_x = grad_bias = None
        if ctx.needs_input_grad[1]:
            grad_bias = grad_out.reshape(-1, grad_out.shape[-1]).sum(0, dtype=bias_dtype)
        if ctx.needs_input_grad[0]:
            w = dequantize_nf4_multi(weights, states, plan=plan)
            if grad_out.dtype != w.dtype:
                w = w.to(grad_out.dtype)
            grad_x = torch.matmul(grad_out, w)
        return grad_x, grad_bias, None
