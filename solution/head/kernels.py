"""Triton kernel for the fused LM head: softmax statistics and dlogits over a logits chunk.

The op (op.py) never builds the full [N, V] logits. It walks over chunks of Nc token rows; per
chunk the two matmuls (z = x @ W^T and dX = g @ W) run in cuBLAS, because Triton 3.6 lowers
tl.dot to CUDA-core FMA on sm_75 (the T4), and this kernel does the memory-bound middle part:

``head_row_softmax_kernel``, one program per row of the fp16 chunk buffer z:
- pass 1 (reads the row once): per-lane online max / sum exp / sum exp * (z - max), the first
  index of the max and the target logit; lanes are merged at the end into lse, entropy and
  argmax (all fp32). Entropy = lse - sum p z = log S - T / S with S = sum exp(z - M),
  T = sum exp(z - M) (z - M), which avoids subtracting two large numbers.
- pass 2 (only when a gradient is needed and the row has a CE weight): overwrites the row in
  place with g = (softmax(z) - onehot(target)) * G_SCALE in fp16. G_SCALE = 2**10 keeps small
  probabilities out of the fp16 subnormal range (|g| <= 1024, far below the fp16 max); the
  caller undoes it in fp32 together with the CE weight after the g @ W matmul.
A weighted row costs 2 reads + 1 write of the row, an unweighted row 1 read.

Sources of ideas (all code here is written from scratch):
- Liger Kernel, FusedLinearCrossEntropy: chunking the tokens and computing the input gradient
  already in the forward pass, overwriting the logits chunk in place with the gradient.
- Apple Cut Cross-Entropy (Wijmans et al. 2024, "Cut Your Losses in Large-Vocabulary Language
  Models"): never materialize the full logits matrix.
- Online softmax (Milakov & Gimelshein 2018, "Online normalizer calculation for softmax"):
  one pass for max and normalizer, rescaling running sums by exp(m_old - m_new).

Fixed config (no autotune), set in op.py: BV vocab entries per step, no tl.dot, no shared-memory
staging; shared memory is only the cross-warp reduction scratch (tens of bytes), safe on sm_75.
"""

import triton
import triton.language as tl

# Finite "minus infinity": keeps 0 * (m_old - m_new) finite in the rescaling.
NEG_BIG = tl.constexpr(-1.0e30)


@triton.jit
def head_row_softmax_kernel(
    z_ptr, stride_z, tgt_ptr, cew_ptr,
    lse_ptr, tlogit_ptr, ent_ptr, amax_ptr, V,
    BV: tl.constexpr, WRITE_G: tl.constexpr, G_SCALE: tl.constexpr,
):
    row = tl.program_id(0)
    zrow = z_ptr + row.to(tl.int64) * stride_z
    tgt = tl.load(tgt_ptr + row)
    offs = tl.arange(0, BV)

    # Pass 1: per-lane online statistics.
    m_v = tl.full((BV,), NEG_BIG, tl.float32)  # running max
    s_v = tl.zeros((BV,), dtype=tl.float32)    # sum exp(z - m_v)
    t_v = tl.zeros((BV,), dtype=tl.float32)    # sum exp(z - m_v) * (z - m_v)
    i_v = tl.zeros((BV,), dtype=tl.int32)      # index of the first max seen by the lane
    tz_v = tl.zeros((BV,), dtype=tl.float32)   # target logit (one lane, one step)
    for v0 in range(0, V, BV):
        v = v0 + offs
        ok = v < V
        z = tl.load(zrow + v, mask=ok, other=0.0).to(tl.float32)
        z = tl.where(ok, z, NEG_BIG)
        m_new = tl.maximum(m_v, z)
        i_v = tl.where(z > m_v, v, i_v)
        dm = m_v - m_new  # <= 0
        sc = tl.exp(dm)
        d = z - m_new  # <= 0
        e = tl.where(ok, tl.exp(d), 0.0)
        t_v = sc * (t_v + s_v * dm) + e * d
        s_v = sc * s_v + e
        m_v = m_new
        tz_v += tl.where(v == tgt, z, 0.0)

    big_m = tl.max(m_v, axis=0)
    dl = m_v - big_m
    scl = tl.exp(dl)
    s_tot = tl.sum(s_v * scl, axis=0)
    t_tot = tl.sum(scl * (t_v + s_v * dl), axis=0)
    best = tl.min(tl.where(m_v == big_m, i_v, 2147483647), axis=0)
    log_s = tl.log(s_tot)
    lse = big_m + log_s
    tl.store(lse_ptr + row, lse)
    tl.store(ent_ptr + row, log_s - t_tot / s_tot)
    tl.store(amax_ptr + row, best)
    tl.store(tlogit_ptr + row, tl.sum(tz_v, axis=0))

    # Pass 2: g = (softmax - onehot) * G_SCALE, in place.
    if WRITE_G:
        cw = tl.load(cew_ptr + row)
        if cw != 0.0:
            for v0 in range(0, V, BV):
                v = v0 + offs
                ok = v < V
                z = tl.load(zrow + v, mask=ok, other=0.0).to(tl.float32)
                p = tl.exp(z - lse)
                g = tl.where(v == tgt, p - 1.0, p) * G_SCALE
                tl.store(zrow + v, g.to(z_ptr.dtype.element_ty), mask=ok)
