"""Fused residual-add + RMSNorm (K1) and rotary embedding (K2) for the packed Qwen2 decoder.

Both kernels work on packed tokens: a [T, ...] layout with no padding, T varies per step.
T only sets the launch grid (one program per token row), so it is never a kernel argument
and one compile serves every step.

K1 add_rmsnorm(r_in, delta, w, eps, out_dtype, save_output=False) -> (r_out, y)
    r_out = r_in + delta                         (fp32; delta fp16/fp32 or None)
    y     = w * (r_out * rsqrt(mean(r_out^2) + eps))   fp32 math, stored as out_dtype
    Same numerics as HF Qwen2RMSNorm on an fp32 hidden state (variance = mean(x^2),
    x * rsqrt(var + eps), then weight * x).
    Saved for backward, default: r_out (fp32 [T, H]) and rstd (fp32 [T]); w is only referenced
    (a frozen parameter, no copy). w gets no gradient: it is detached before the
    autograd Function sees it and backward returns None for it.
    Saved with save_output=True (decoder-layer norms with a delta and an fp16 y): the output y
    itself (fp16 [T, H], the tensor its consumers keep alive anyway: the q/v LoRA inputs, the
    recompute MLP's input) and rstd, NOT r_out, so the fp32 residual is freed as soon as the
    next norm has read it. Backward recovers xhat = y / w. That is ill-conditioned where |w| is
    tiny (y underflows into fp16 subnormals or to 0; y / w is 0/0 for w == 0), so those K
    channels (fixed per weight: the weights are frozen; found once per weight, cached) are
    exact: the forward kernel also writes their fp32 xhat into a [T, K] buffer (K may be 0;
    on Qwen2.5-1.5B K <= 17, mostly 0-2). Threshold, see SPECIAL_W_MIN: with |w| >= 2^-10,
    y = w * xhat is a normal fp16 number for every |xhat| >= 2^-4 (1/16 of the row RMS, which
    is 1), so the recovered xhat has the fp16 rounding error 2^-11 relative (what saving the
    residual in fp16 would give) and below that at most 2^-25 / 2^-10 = 2^-15 absolute (the
    fp16 error of an xhat of 1/16). |w| > 65504 / sqrt(H) (y could overflow; |xhat| <= sqrt(H))
    and non-finite w are exact channels too. Precision: about 5e-4 relative on the norm part of
    the gradient vs the fp32-residual backward (measured in solution/test_body_norm_rope.py).
    Backward, one kernel: dr = dr_out + rstd * (g - xhat * mean(g * xhat)), g = dy * w,
    xhat = r_out * rstd (or as above). d_r_in = dr (fp32), d_delta = dr stored in delta's dtype
    by the same kernel (no separate cast pass).
    Fan-out (add_rmsnorm(..., n_out=k)): y is returned k times (views of one buffer), one per
    consumer (q/k/v projections and the q/v LoRA inputs; MLP gate and up), so backward gets k
    gradients and the kernel sums them in fp32 registers (dy = dy_1 + ... + dy_k) instead of
    autograd summing them with k - 1 fp16 add kernels (each a full [T, H] read-read-write).
    Unused outputs (e.g. the final norm's r_out) arrive as None (set_materialize_grads(False)):
    no zero fill, no read.

K2 rope(q, k, positions, inv_freq) -> (q_rot, k_rot)
    q [T, HQ, D], k [T, HK, D] fp16 (any token/head strides, last dim must be contiguous;
    other layouts are copied with .contiguous()). positions [T] int32 or int64.
    angle = float(pos) * inv_freq[j] in fp32 (same single fp32 product as HF's K=1 matmul),
    cos/sin with libdevice (accurate cosf/sinf, as torch uses), rotate_half convention:
        out[:D/2] = x1 * c - x2 * s,   out[D/2:] = x2 * c + x1 * s     (fp32, stored fp16)
    Optional q_delta [T, HQ*D] fp16 (the q LoRA output) is added to q in fp16 inside the
    kernel before rotating, so the LoRA add costs no extra pass; its grad equals q's grad.
    q and k are handled by the same launch. Outputs are fresh contiguous [T, H*, D] tensors.
    Backward is the same kernel with the sine negated (rotation by -angle); saved: positions
    and inv_freq only. Forward and backward share one compiled kernel (the sign is a runtime
    argument).

Sources of ideas: the RMSNorm forward/backward formulas are the standard derivation
(Zhang & Sennrich 2019, "Root Mean Square Layer Normalization"); RoPE is from Su et al.
2021 (RoFormer) in the HF rotate_half form. Fusing the residual add into the norm and
saving only the per-row rstd is a common pattern in fused-norm kernels (e.g. the Triton
layer-norm tutorial and Apex fused layer norm); no code was taken from them. Saving the norm
output instead of its input and recovering the normalized input as y / w in backward
(save_output=True) is the memory-efficient LayerNorm/RMSNorm of NVIDIA Apex PR #1715; the
exact side buffer for the tiny-|w| channels is our own addition. All code here is written for
this project.
"""

import weakref

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# --- fixed launch configs (no autotune). Chosen for sm_75 (T4, 40 SMs, 64K regs/SM) and
# checked on sm_86: one row per program, a 2048-wide fp32 row over 4 warps is 16 values per
# thread; T up to ~2048 rows gives ~50 programs per SM, enough to hide memory latency.
NORM_BLOCK = 2048
NORM_WARPS_FWD = 4
NORM_WARPS_BWD = 8
ROPE_WARPS = 4


# ----------------------------------------------------------------------------------------
# K1: residual add + RMSNorm
# ----------------------------------------------------------------------------------------

@triton.jit(do_not_specialize=["n_special"])
def _add_rmsnorm_fwd(r_in_ptr, delta_ptr, w_ptr, r_out_ptr, y_ptr, rstd_ptr, slot_ptr, xs_ptr,
                     n_cols, eps, n_special,
                     HAS_DELTA: tl.constexpr, BLOCK: tl.constexpr):
    # n_special > 0 (save_output mode with K = n_special exact channels): also store xhat of
    # the channels whose slot (slot_ptr [n_cols] int32, -1 elsewhere) is >= 0 into
    # xs[row, slot] (fp32 [rows, n_special]). A runtime int, not specialized: one compile for
    # every K, incl. the launches without exact channels (n_special = 0, dummy pointers).
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    ok = cols < n_cols
    base = row * n_cols
    r = tl.load(r_in_ptr + base + cols, mask=ok, other=0.0).to(tl.float32)
    if HAS_DELTA:
        d = tl.load(delta_ptr + base + cols, mask=ok, other=0.0).to(tl.float32)
        r = r + d
        tl.store(r_out_ptr + base + cols, r, mask=ok)
    ms = tl.sum(r * r, axis=0) / n_cols
    inv = libdevice.rsqrt(ms + eps)
    w = tl.load(w_ptr + cols, mask=ok, other=0.0)
    xhat = r * inv
    y = w * xhat
    tl.store(y_ptr + base + cols, y.to(y_ptr.dtype.element_ty), mask=ok)
    tl.store(rstd_ptr + row, inv)
    if n_special > 0:
        slot = tl.load(slot_ptr + cols, mask=ok, other=-1)
        tl.store(xs_ptr + row * n_special + slot, xhat, mask=ok & (slot >= 0))


# Most gradients of a norm output y (one per consumer) that the backward kernel sums itself.
MAX_DY = 5


@triton.jit(do_not_specialize=["n_dy", "n_special"])
def _add_rmsnorm_bwd(dy0_ptr, dy1_ptr, dy2_ptr, dy3_ptr, dy4_ptr, n_dy, dr_out_ptr, x_ptr,
                     rstd_ptr, w_ptr, slot_ptr, xs_ptr, n_special, dr_ptr, ddelta_ptr, n_cols,
                     HAS_DR_OUT: tl.constexpr, WRITE_DDELTA: tl.constexpr,
                     FROM_Y: tl.constexpr, BLOCK: tl.constexpr):
    # dy = dy0 + ... + dy{n_dy-1}, summed in fp32 registers: the gradients of the n_dy copies
    # of y (one per consumer), which autograd would otherwise sum with one fp16 add kernel per
    # extra consumer. n_dy is a runtime value (0..MAX_DY, not specialized): one compile serves
    # every fan-out; the pointers past n_dy are dummies and never read.
    # x_ptr: FROM_Y = 0: the fp32 residual r_out, xhat = r * rstd. FROM_Y = 1: the saved output
    # y (fp16), xhat = y / w, except the n_special exact channels (slot >= 0), read from xs.
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    ok = cols < n_cols
    base = row * n_cols
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    if n_dy > 0:
        dy = tl.load(dy0_ptr + base + cols, mask=ok, other=0.0).to(tl.float32)
        if n_dy > 1:
            dy += tl.load(dy1_ptr + base + cols, mask=ok, other=0.0).to(tl.float32)
        if n_dy > 2:
            dy += tl.load(dy2_ptr + base + cols, mask=ok, other=0.0).to(tl.float32)
        if n_dy > 3:
            dy += tl.load(dy3_ptr + base + cols, mask=ok, other=0.0).to(tl.float32)
        if n_dy > 4:
            dy += tl.load(dy4_ptr + base + cols, mask=ok, other=0.0).to(tl.float32)
        inv = tl.load(rstd_ptr + row)
        if FROM_Y:
            # other=1.0: the masked tail gives xhat = 0 / 1 = 0 (and g = 0), never 0 / 0.
            w = tl.load(w_ptr + cols, mask=ok, other=1.0)
            xhat = tl.load(x_ptr + base + cols, mask=ok, other=0.0).to(tl.float32) / w
            if n_special > 0:
                slot = tl.load(slot_ptr + cols, mask=ok, other=-1)
                exact = slot >= 0
                xs = tl.load(xs_ptr + row * n_special + slot, mask=ok & exact, other=0.0)
                xhat = tl.where(exact, xs, xhat)  # drops the 0/0 or inf of w == 0 lanes
        else:
            w = tl.load(w_ptr + cols, mask=ok, other=0.0)
            xhat = tl.load(x_ptr + base + cols, mask=ok, other=0.0) * inv
        g = dy * w
        proj = tl.sum(g * xhat, axis=0) / n_cols
        acc = inv * (g - xhat * proj)
    if HAS_DR_OUT:
        acc += tl.load(dr_out_ptr + base + cols, mask=ok, other=0.0)
    tl.store(dr_ptr + base + cols, acc, mask=ok)
    if WRITE_DDELTA:
        tl.store(ddelta_ptr + base + cols, acc.to(ddelta_ptr.dtype.element_ty), mask=ok)


# save_output mode: channels with |w| < SPECIAL_W_MIN (or |w| > 65504 / sqrt(H), or w not
# finite) keep an exact fp32 xhat; the others recover xhat = y / w from the fp16 y. Why 2^-10:
# see the module docstring (y normal fp16 for |xhat| >= 1/16 of the row RMS; error floor 2^-15
# absolute below that). Absolute, not relative to max|w|: the fp16 underflow limit is absolute
# and xhat has a fixed scale (row RMS 1).
SPECIAL_W_MIN = 2.0 ** -10

_SPECIAL = {}  # (device, data_ptr, version, n, dtype) -> (weakref to the weight, slots, K)
_DUMMY_SLOTS = {}  # device -> int32 [1] (slot_ptr of launches without exact channels)


def _dummy_slots(device):
    d = _DUMMY_SLOTS.get(device)
    if d is None:
        d = _DUMMY_SLOTS[device] = torch.full((1,), -1, device=device, dtype=torch.int32)
    return d


def special_channels(weight):
    """(slots int32 [H], K) for a frozen norm weight [H]: slots[j] = 0..K-1 for the channels
    whose xhat must be saved exactly (|w| < SPECIAL_W_MIN, |w| > 65504 / sqrt(H), w not finite),
    -1 elsewhere. Computed once per weight (one host sync) and cached, keyed on the weight's
    storage, version counter and shape (an in-place change of the weight recomputes it); the
    cache keeps a weak reference, so a new tensor at a reused address is not mistaken for it."""
    key = (weight.device, weight.data_ptr(), weight._version, weight.shape[0], weight.dtype)
    hit = _SPECIAL.get(key)
    if hit is not None and hit[0]() is weight:
        return hit[1], hit[2]
    a = weight.detach().to(torch.float32).abs()
    big = 65504.0 / float(weight.shape[0]) ** 0.5
    special = (a < SPECIAL_W_MIN) | (a > big) | ~torch.isfinite(a)
    idx = special.nonzero().squeeze(1)
    k = int(idx.numel())
    slots = torch.full(a.shape, -1, device=a.device, dtype=torch.int32)
    if k:
        slots[idx] = torch.arange(k, device=a.device, dtype=torch.int32)
    try:
        _SPECIAL[key] = (weakref.ref(weight), slots, k)
    except TypeError:  # not weak-referenceable: no cache
        pass
    return slots, k


class AddRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r_in, delta, w, eps, out_dtype, n_out, slots, n_special):
        # slots not None: save_output mode (see the module docstring; slots/n_special from
        # special_channels(weight)).
        shape = r_in.shape
        n = shape[-1]
        assert n <= NORM_BLOCK, "row wider than NORM_BLOCK"
        assert 1 <= n_out <= MAX_DY
        r_in2 = r_in.reshape(-1, n).to(torch.float32).contiguous()
        rows = r_in2.shape[0]
        has_delta = delta is not None
        if has_delta:
            delta2 = delta.reshape(-1, n).contiguous()
            r_out = torch.empty_like(r_in2)
        else:
            delta2 = r_in2  # unused
            r_out = r_in2   # r_out is r_in itself: nothing to add
        y = torch.empty((rows, n), device=r_in.device, dtype=out_dtype)
        rstd = torch.empty((rows,), device=r_in.device, dtype=torch.float32)
        from_y = slots is not None
        k = n_special if from_y else 0
        xs = torch.empty((rows, k), device=r_in.device, dtype=torch.float32) if k else None
        if rows > 0:
            _add_rmsnorm_fwd[(rows,)](r_in2, delta2, w, r_out, y, rstd,
                                      slots if from_y else _dummy_slots(r_in.device),
                                      xs if k else rstd, n, eps, k,
                                      HAS_DELTA=has_delta, BLOCK=NORM_BLOCK,
                                      num_warps=NORM_WARPS_FWD)
        if from_y:
            ctx.save_for_backward(y, rstd, w, slots, xs)
        else:
            ctx.save_for_backward(r_out, rstd, w)
        ctx.from_y, ctx.n_special = from_y, k
        ctx.has_delta = has_delta
        ctx.delta_dtype = delta.dtype if has_delta else None
        ctx.shape = shape
        # An unused output (the final norm's r_out, a copy of y that nothing consumed) reaches
        # backward as None instead of a zero tensor that autograd would fill and the kernel read.
        ctx.set_materialize_grads(False)
        # Without delta, r_out aliases r_in (no copy); autograd still routes its gradient
        # (dr_out) through this Function's backward, which adds it to d_r_in.
        # y is returned n_out times as distinct views of one buffer (no copy). Each consumer of
        # the normed x takes its own copy, so autograd hands backward one gradient per consumer
        # (instead of summing them with add kernels) and the backward kernel sums them. Nothing
        # writes into y (autograd forbids in-place ops on views returned by a custom Function).
        ys = tuple(y.view(*shape[:-1], n) for _ in range(n_out))
        return (r_out.view(shape),) + ys

    @staticmethod
    def backward(ctx, dr_out, *dys):
        if ctx.from_y:
            x, rstd, w, slots, xs = ctx.saved_tensors
        else:
            (x, rstd, w), slots, xs = ctx.saved_tensors, None, None
        n = x.shape[-1]
        rows = x.shape[0]
        # autograd casts each gradient to its output's dtype (y's), so all dy share one dtype.
        dys = [d.reshape(-1, n).contiguous() for d in dys if d is not None]
        has_dr = dr_out is not None
        write_dd = ctx.has_delta and ctx.needs_input_grad[1]
        if not dys and not has_dr:
            return None, None, None, None, None, None, None, None
        dr = torch.empty((rows, n), device=x.device, dtype=torch.float32)
        slots_arg = slots if slots is not None else _dummy_slots(x.device)
        xs_arg = xs if xs is not None else rstd
        slots_dy = dys + [dys[0] if dys else dr] * (MAX_DY - len(dys))  # dummies: never read
        dr2 = dr_out.reshape(-1, n).to(torch.float32).contiguous() if has_dr else dr
        ddelta = torch.empty((rows, n), device=x.device, dtype=ctx.delta_dtype) \
            if write_dd else dr
        if rows > 0:
            _add_rmsnorm_bwd[(rows,)](*slots_dy, len(dys), dr2, x, rstd, w, slots_arg, xs_arg,
                                      ctx.n_special, dr, ddelta, n,
                                      HAS_DR_OUT=has_dr, WRITE_DDELTA=write_dd,
                                      FROM_Y=ctx.from_y, BLOCK=NORM_BLOCK,
                                      num_warps=NORM_WARPS_BWD)
        d_rin = dr.view(ctx.shape) if ctx.needs_input_grad[0] else None
        d_delta = ddelta.view(ctx.shape) if write_dd else None
        return d_rin, d_delta, None, None, None, None, None, None


def add_rmsnorm(r_in, delta, weight, eps=1e-6, out_dtype=torch.float16, n_out=1,
                save_output=False):
    """r_out = r_in + delta; y = weight * rmsnorm(r_out). Returns (r_out fp32, y out_dtype);
    with n_out > 1, (r_out, (y_1, ..., y_n_out)): n_out views of the same y, one for each
    consumer, whose gradients the backward kernel sums (no autograd add kernels).

    r_in: fp32 [..., H]; delta: fp16/fp32 [..., H] or None (then r_out is r_in);
    weight: fp32 [H], treated as frozen (detached, never gets a gradient).
    save_output: backward from the saved y (+ exact xhat of the tiny-|w| channels) instead of
    the fp32 r_out (see the module docstring). Applies only with a delta and an fp16 y (the
    decoder-layer norms); otherwise ignored (r_out is saved).
    """
    slots, k = None, 0
    if save_output and delta is not None and out_dtype == torch.float16:
        slots, k = special_channels(weight)
    out = AddRMSNorm.apply(r_in, delta, weight.detach().to(torch.float32), float(eps),
                           out_dtype, int(n_out), slots, k)
    return (out[0], out[1]) if n_out == 1 else (out[0], tuple(out[1:]))


def rmsnorm_recompute(r, w, eps, out_dtype=torch.float16):
    """y = w * rmsnorm(r) for fp32 r [T, H] (contiguous) and fp32 w [H], no autograd.

    The same K1 forward kernel (HAS_DELTA=False, the variant the first layer's norm already
    compiled) on the saved fp32 residual gives bit-for-bit the y that add_rmsnorm returned for
    it: the stored r_out is the fp32 value the forward normalized. Used by backward passes that
    recompute a norm output instead of saving it.
    """
    rows, n = r.shape
    y = torch.empty((rows, n), device=r.device, dtype=out_dtype)
    rstd = torch.empty((rows,), device=r.device, dtype=torch.float32)
    if rows > 0:
        _add_rmsnorm_fwd[(rows,)](r, r, w, r, y, rstd, _dummy_slots(r.device), rstd, n, eps, 0,
                                  HAS_DELTA=False, BLOCK=NORM_BLOCK, num_warps=NORM_WARPS_FWD)
    return y


# ----------------------------------------------------------------------------------------
# K2: rotary embedding for q and k in one launch
# ----------------------------------------------------------------------------------------

@triton.jit
def _rotate_heads(x_ptr, d_ptr, o_ptr, row, sx_t, sx_h, sd_t, sd_h, so_t, n_heads, c, s,
                  HAS_D: tl.constexpr, HP: tl.constexpr, HALF: tl.constexpr):
    # Inlined into _rope_kernel, never launched on its own.
    # x[row, h, :] (+ d[row, h, :]) -> o[row, h, :] for h < n_heads; c, s: [1, HALF] fp32.
    h = tl.arange(0, HP)[:, None]
    j = tl.arange(0, HALF)[None, :]
    ok = (h < n_heads) & (j < HALF)
    src = x_ptr + row * sx_t + h * sx_h + j
    x1 = tl.load(src, mask=ok, other=0.0)
    x2 = tl.load(src + HALF, mask=ok, other=0.0)
    if HAS_D:
        # The unfused model adds the LoRA output to the base projection in fp16 (one fp16
        # rounding), then rotates in fp32: do the same.
        dsrc = d_ptr + row * sd_t + h * sd_h + j
        x1 = x1 + tl.load(dsrc, mask=ok, other=0.0).to(x1.dtype)
        x2 = x2 + tl.load(dsrc + HALF, mask=ok, other=0.0).to(x2.dtype)
    x1 = x1.to(tl.float32)
    x2 = x2.to(tl.float32)
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s
    dst = o_ptr + row * so_t + h * (2 * HALF) + j
    tl.store(dst, o1.to(o_ptr.dtype.element_ty), mask=ok)
    tl.store(dst + HALF, o2.to(o_ptr.dtype.element_ty), mask=ok)


@triton.jit
def _rope_kernel(q_ptr, qd_ptr, k_ptr, qo_ptr, ko_ptr, pos_ptr, inv_freq_ptr,
                 sq_t, sq_h, sd_t, sd_h, sk_t, sk_h, n_q, n_k, sin_sign,
                 HAS_QD: tl.constexpr, HQP: tl.constexpr, HKP: tl.constexpr, HALF: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    p = tl.load(pos_ptr + row).to(tl.float32)
    f = tl.load(inv_freq_ptr + tl.arange(0, HALF))
    ang = (p * f)[None, :]
    c = libdevice.cos(ang)
    s = libdevice.sin(ang) * sin_sign
    _rotate_heads(q_ptr, qd_ptr, qo_ptr, row, sq_t, sq_h, sd_t, sd_h, n_q * (2 * HALF), n_q,
                  c, s, HAS_QD, HQP, HALF)
    _rotate_heads(k_ptr, k_ptr, ko_ptr, row, sk_t, sk_h, 0, 0, n_k * (2 * HALF), n_k,
                  c, s, False, HKP, HALF)


def _launch_rope(q, k, positions, inv_freq, sign, q_delta=None):
    if q.stride(-1) != 1:
        q = q.contiguous()
    has_qd = q_delta is not None
    if has_qd:
        q_delta = q_delta.reshape(q.shape)  # [T, HQ*D] -> [T, HQ, D] (a view when possible)
        if q_delta.stride(-1) != 1:
            q_delta = q_delta.contiguous()
    qd = q_delta if has_qd else q
    if k.stride(-1) != 1:
        k = k.contiguous()
    T, hq, d = q.shape
    hk = k.shape[1]
    qo = torch.empty((T, hq, d), device=q.device, dtype=q.dtype)
    ko = torch.empty((T, hk, d), device=k.device, dtype=k.dtype)
    if T > 0:
        _rope_kernel[(T,)](q, qd, k, qo, ko, positions, inv_freq,
                           q.stride(0), q.stride(1), qd.stride(0), qd.stride(1),
                           k.stride(0), k.stride(1), hq, hk, sign,
                           HAS_QD=has_qd, HQP=triton.next_power_of_2(hq), HKP=triton.next_power_of_2(hk),
                           HALF=d // 2, num_warps=ROPE_WARPS)
    return qo, ko


class Rope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, positions, inv_freq, q_delta):
        qo, ko = _launch_rope(q, k, positions, inv_freq, 1.0, q_delta)
        ctx.save_for_backward(positions, inv_freq)
        ctx.q_shape, ctx.q_dtype = qo.shape, qo.dtype
        ctx.k_shape, ctx.k_dtype = ko.shape, ko.dtype
        ctx.has_qd = q_delta is not None
        ctx.qd_shape = q_delta.shape if q_delta is not None else None
        return qo, ko

    @staticmethod
    def backward(ctx, dqo, dko):
        positions, inv_freq = ctx.saved_tensors
        T = positions.shape[0]
        # A None gradient (output unused) becomes zeros; never happens in the decoder.
        if dqo is None:
            dqo = torch.zeros(ctx.q_shape, device=positions.device, dtype=ctx.q_dtype)
        if dko is None:
            dko = torch.zeros(ctx.k_shape, device=positions.device, dtype=ctx.k_dtype)
        dq, dk = _launch_rope(dqo.view(T, *dqo.shape[-2:]), dko.view(T, *dko.shape[-2:]),
                              positions, inv_freq, -1.0)
        # d(q_delta) == d(q): the delta is added before the rotation.
        dqd = dq.view(ctx.qd_shape) if ctx.has_qd and ctx.needs_input_grad[4] else None
        return dq, dk, None, None, dqd


def default_inv_freq(head_dim=128, theta=1e6, device=None):
    """HF default-rope inverse frequencies, computed exactly as transformers does (fp32, CPU)."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
    return inv.to(device)


def rope(q, k, positions, inv_freq, q_delta=None):
    """Rotate q [T, HQ, D] and k [T, HK, D] (fp16) by per-token angles positions[t] * inv_freq.

    q_delta: optional fp16 [T, HQ*D] or [T, HQ, D] (the LoRA output); q + q_delta is formed in
    fp16 (as the unfused model does) and then rotated. Its gradient is the gradient of q
    (the same tensor is returned for both).

    positions: int32/int64 [T] on the GPU; inv_freq: fp32 [D/2] on the GPU (pass the model's
    rotary_emb.inv_freq buffer). Returns fresh contiguous (q_rot [T, HQ, D], k_rot [T, HK, D])
    in q's / k's dtype.
    """
    assert q.dim() == 3 and k.dim() == 3 and q.shape[0] == k.shape[0] == positions.shape[0]
    assert q.shape[-1] == k.shape[-1] and q.shape[-1] % 2 == 0
    assert (q.shape[-1] // 2) & (q.shape[-1] // 2 - 1) == 0, "D/2 must be a power of two"
    positions = positions.contiguous()
    inv_freq = inv_freq.detach().to(torch.float32).contiguous()
    return Rope.apply(q, k, positions, inv_freq, q_delta)
