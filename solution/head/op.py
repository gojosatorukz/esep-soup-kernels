"""Fused linear + cross-entropy op over the LM head, built on the kernels in kernels.py.

``loss, stats = fused_linear_ce(x, w16, target, ce_weight)``

- x:         [N, H] fp16 or fp32 hidden states (after the final norm; shifted / gathered by the
             caller). Cast to fp16 inside; the gradient comes back in x.dtype.
- w16:       [V, H] fp16 frozen LM head weight (gets no gradient)
- target:    [N] int64, -100 (or any id outside [0, V)) = no CE term for the row
- ce_weight: [N] fp32, weight of each row's CE term in the scalar loss (forced to 0 for ignored rows)
- loss:      0-dim fp32, sum_i ce_weight_i * (lse_i - z_i[target_i]); differentiable w.r.t. x
- stats:     dict of detached [N] tensors, every row gets lse / entropy / argmax (also ignored rows):
             "lse" fp32, "target_logit" fp32 (0 where ignored), "ce" fp32 (lse - target_logit,
             0 where ignored), "entropy" fp32, "argmax" int64

The dX precompute runs only when torch.is_grad_enabled() and x.requires_grad (need_grad=None);
dX is computed in the forward, saved as [N, H] fp32, and the backward only scales it.

Per chunk of rows: z = x_c @ W^T in cuBLAS (fp16 out) into one reused [Nc, V] buffer; our Triton
row kernel (kernels.py) computes the stats and overwrites z with g in place; dX_c = g @ W in
cuBLAS (fp32 out) for the weighted rows only, scaled by ce_weight / G_SCALE in fp32. Triton
3.6 has no tensor-core tl.dot on sm_75 (the T4), so the two GEMMs stay in cuBLAS.

Sources of ideas (code written from scratch): chunked linear + cross-entropy with the input
gradient precomputed in the forward pass, from Liger Kernel's FusedLinearCrossEntropy; never
materializing the full logits, from Apple's Cut Cross-Entropy (Wijmans et al. 2024); online
softmax (Milakov & Gimelshein 2018) in the row kernel. Original here: ordering the rows that
carry a loss first, so the gradient pass and the g @ W GEMM run on a prefix only, and the fixed
gradient scale that keeps fp16 g out of the subnormal range.
"""

import torch

from .kernels import head_row_softmax_kernel

# Rows per chunk. The chunk buffer is [CHUNK_ROWS, V] fp16: 256 x 151936 x 2 B = 74 MiB.
# Locally (sm_86) 256 was as fast as 1024 and faster than 512 (cuBLAS tile choice); recheck on T4.
CHUNK_ROWS = 256
# g = (softmax - onehot) * G_SCALE in fp16; undone in fp32 after the g @ W matmul.
G_SCALE = 1024.0
# Row kernel: BV vocab entries per loop step (8 per thread at 8 warps). No tl.dot, no smem
# staging; smem is only the cross-warp reduction scratch. Safe on sm_75.
ROW_BV, ROW_WARPS = 2048, 8
# Qwen2.5 vocabulary size: the V (and z row stride) the measured run launches the row kernel with.
VOCAB = 151936


def warmup(device=None, vocab=VOCAB):
    """Ahead-of-time compile of the row kernel variant the training run launches (WRITE_G=True).

    Compiles only (no launch). Reproduces the specialization of the launch in _run: same dtypes,
    16-byte aligned pointers (fresh buffers), stride_z = V = vocab, same constexprs and options.
    No-op without CUDA. Safe to call from a worker thread."""
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda" if device is None else device)
    with torch.cuda.device(device):
        z = torch.empty(1, vocab, device=device, dtype=torch.float16)
        tgt = torch.empty(1, device=device, dtype=torch.int64)
        f32 = [torch.empty(1, device=device, dtype=torch.float32) for _ in range(4)]
        cew, lse, tlogit, ent = f32
        amax = torch.empty(1, device=device, dtype=torch.int32)
        head_row_softmax_kernel.warmup(
            z, z.stride(0), tgt, cew, lse, tlogit, ent, amax, vocab,
            grid=(1,), BV=ROW_BV, WRITE_G=True, G_SCALE=G_SCALE, num_warps=ROW_WARPS,
        )


def _run(x16, w16, target, cew, n_live, need_grad, chunk_rows=CHUNK_ROWS):
    """Runs the chunks on rows as given; rows [0, n_live) are the weighted ones (cew != 0).

    target must be -1 for rows without a valid target and cew 0 there.
    Returns (lse, tlogit, ent, amax, dx)."""
    n, h = x16.shape
    v = w16.shape[0]
    dev = x16.device

    lse = torch.empty(n, device=dev, dtype=torch.float32)
    ent = torch.empty(n, device=dev, dtype=torch.float32)
    amax = torch.empty(n, device=dev, dtype=torch.int32)
    tlogit = torch.empty(n, device=dev, dtype=torch.float32)
    dx = torch.zeros(n, h, device=dev, dtype=torch.float32) if need_grad else None

    nc = max(1, min(chunk_rows, n))
    z = torch.empty(nc, v, device=dev, dtype=torch.float16)
    wt = w16.t()

    for r0 in range(0, n, nc):
        nr = min(nc, n - r0)
        zc = z[:nr]
        torch.mm(x16[r0:r0 + nr], wt, out=zc)  # cuBLAS, fp16 out, fp32 accumulation
        head_row_softmax_kernel[(nr,)](
            zc, zc.stride(0), target[r0:r0 + nr], cew[r0:r0 + nr],
            lse[r0:r0 + nr], tlogit[r0:r0 + nr], ent[r0:r0 + nr], amax[r0:r0 + nr], v,
            BV=ROW_BV, WRITE_G=need_grad, G_SCALE=G_SCALE, num_warps=ROW_WARPS,
        )
        nl = min(max(n_live - r0, 0), nr)
        if need_grad and nl:
            dxc = dx[r0:r0 + nl]
            torch.mm(zc[:nl], w16, out_dtype=torch.float32, out=dxc)  # cuBLAS, fp32 out
            dxc.mul_((cew[r0:r0 + nl] * (1.0 / G_SCALE))[:, None])
    return lse, tlogit, ent, amax, dx


def _forward(x, w16, target, ce_weight, need_grad, chunk_rows):
    v = w16.shape[0]
    valid = (target >= 0) & (target < v)
    cew = torch.where(valid, ce_weight.to(torch.float32), 0.0)
    target = torch.where(valid, target, -1)
    x16 = x.to(torch.float16)
    perm = None
    n_live = 0
    if need_grad:
        # Weighted rows first (stable): the g pass and the g @ W matmul then run on a prefix.
        live = cew != 0
        perm = torch.argsort((~live).to(torch.int8), stable=True)
        n_live = int(live.sum())  # one host sync per call
        x16, target_p, cew_p = x16[perm], target[perm], cew[perm]
    else:
        x16, target_p, cew_p = x16.contiguous(), target.contiguous(), cew
    lse, tlogit, ent, amax, dx = _run(x16, w16, target_p, cew_p, n_live, need_grad, chunk_rows)
    if perm is not None:
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel(), device=perm.device)
        lse, tlogit, ent, amax, dx = lse[inv], tlogit[inv], ent[inv], amax[inv], dx[inv]
    tlogit = torch.where(valid, tlogit, 0.0)
    ce = torch.where(valid, lse - tlogit, 0.0)
    loss = (cew * ce).sum()
    return loss, (lse, tlogit, ce, ent, amax.long()), dx


class FusedLinearCE(torch.autograd.Function):
    """Loss is differentiable w.r.t. x; dX is precomputed in the forward."""

    @staticmethod
    def forward(ctx, x, w16, target, ce_weight, chunk_rows):
        loss, stats, dx = _forward(x, w16, target, ce_weight, True, chunk_rows)
        ctx.save_for_backward(dx)
        ctx.x_dtype = x.dtype
        ctx.mark_non_differentiable(*stats)
        return (loss, *stats)

    @staticmethod
    def backward(ctx, grad_loss, *unused):
        (dx,) = ctx.saved_tensors
        return (dx * grad_loss).to(ctx.x_dtype), None, None, None, None


_KEYS = ("lse", "target_logit", "ce", "entropy", "argmax")


def fused_linear_ce(x, w16, target, ce_weight, need_grad=None, chunk_rows=CHUNK_ROWS):
    """loss = sum_i ce_weight_i * CE(x_i @ w16^T, target_i), plus per-token stats (see module doc).

    need_grad=None means: grad mode is on and x requires grad. Without a gradient the g pass and
    the g @ W matmul are skipped.
    """
    assert x.dim() == 2 and w16.dim() == 2 and x.shape[1] == w16.shape[1], "shape mismatch"
    assert x.dtype in (torch.float16, torch.float32), "x must be fp16 or fp32"
    assert w16.dtype == torch.float16, "w16 must be fp16"
    assert target.shape == (x.shape[0],) and ce_weight.shape == (x.shape[0],)
    assert x.is_cuda and w16.is_cuda
    if need_grad is None:
        need_grad = torch.is_grad_enabled() and x.requires_grad
    w16 = w16.contiguous()
    target = target.to(torch.int64)
    if not x.shape[0]:
        z = torch.zeros(0, device=x.device, dtype=torch.float32)
        loss = x.float().sum() * 0.0 if need_grad else z.sum()
        return loss, dict(zip(_KEYS, (z, z, z, z, z.long())))
    if need_grad:
        loss, *stats = FusedLinearCE.apply(x, w16, target, ce_weight, chunk_rows)
    else:
        with torch.no_grad():
            loss, stats, _ = _forward(x, w16, target, ce_weight, False, chunk_rows)
    return loss, dict(zip(_KEYS, stats))
