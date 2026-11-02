"""Tests for solution/body/norm_rope.py (K1 add_rmsnorm, K2 rope) against HF-style PyTorch."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from solution.body.norm_rope import add_rmsnorm, default_inv_freq, rope  # noqa: E402

H, HQ, HK, D, EPS, THETA = 1536, 12, 2, 128, 1e-6, 1e6
TS = [1, 37, 1000, 2048]
DEV = "cuda"


def _err(a, b):
    a, b = a.double(), b.double()
    abs_err = (a - b).abs().max().item()
    rel = abs_err / max(b.abs().max().item(), 1e-30)
    return abs_err, rel


# ---------------------------------------------------------------- K1 reference (HF math)

def ref_add_rmsnorm(r_in, delta, w, out_dtype):
    r = r_in + delta if delta is not None else r_in  # fp32 + fp16 -> fp32, as in the decoder
    h = r.to(torch.float32)
    var = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(var + EPS)
    return r, (w * h.to(torch.float32)).to(out_dtype)


@pytest.mark.parametrize("T", TS)
@pytest.mark.parametrize("case", ["fp16_delta", "fp32_delta", "no_delta", "fp32_out"])
def test_add_rmsnorm(T, case):
    torch.manual_seed(T)
    delta_dtype = {"fp16_delta": torch.float16, "fp32_delta": torch.float32,
                   "no_delta": None, "fp32_out": torch.float16}[case]
    out_dtype = torch.float32 if case == "fp32_out" else torch.float16
    r_in = (torch.randn(1, T, H, device=DEV) * 3).requires_grad_()
    delta = None
    if delta_dtype is not None:
        delta = torch.randn(1, T, H, device=DEV).to(delta_dtype).requires_grad_()
    w = torch.nn.Parameter(1 + 0.1 * torch.randn(H, device=DEV))

    r_out, y = add_rmsnorm(r_in, delta, w, EPS, out_dtype)
    rr, yr = ref_add_rmsnorm(r_in, delta, w.detach(), out_dtype)
    assert r_out.dtype == torch.float32 and y.dtype == out_dtype
    assert r_out.shape == r_in.shape and y.shape == r_in.shape
    torch.testing.assert_close(r_out, rr, rtol=0, atol=0)
    tol = 2e-3 if out_dtype == torch.float16 else 1e-5
    torch.testing.assert_close(y.float(), yr.float(), rtol=tol, atol=tol)

    gy = torch.randn_like(y)
    gr = torch.randn_like(r_out)
    ins = [r_in] + ([delta] if delta is not None else [])
    g = torch.autograd.grad([r_out, y], ins, [gr, gy])
    g_ref = torch.autograd.grad([rr, yr], ins, [gr, gy])
    for a, b in zip(g, g_ref):
        assert a.dtype == b.dtype
        ae, re = _err(a, b)
        assert re < (5e-3 if a.dtype == torch.float16 else 1e-5), (ae, re)
    assert w.grad is None


def test_add_rmsnorm_dy_only():
    """Final norm: r_out is not used afterwards, so dr_out is None."""
    T = 37
    r_in = torch.randn(T, H, device=DEV, requires_grad=True)
    delta = torch.randn(T, H, device=DEV).half().requires_grad_()
    w = torch.rand(H, device=DEV) + 0.5
    _, y = add_rmsnorm(r_in, delta, w, EPS, torch.float32)
    _, yr = ref_add_rmsnorm(r_in, delta, w, torch.float32)
    gy = torch.randn_like(y)
    a = torch.autograd.grad(y, [r_in, delta], gy)
    b = torch.autograd.grad(yr, [r_in, delta], gy)
    torch.testing.assert_close(a[0], b[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(a[1], b[1], rtol=2e-3, atol=2e-3)


# ---------------------------------------------------------------- K2 reference (HF math)

def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def ref_rope(q, k, pos, inv_freq, q_delta=None):
    if q_delta is not None:
        q = q + q_delta.view(q.shape)  # fp16 add, as peft's base + lora
    freqs = (inv_freq[None, :, None].float() @ pos[None, None, :].float()).transpose(1, 2)[0]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos()[:, None, :], emb.sin()[:, None, :]  # fp32 (hidden dtype fp32)
    qe = (q * cos) + (_rotate_half(q) * sin)
    ke = (k * cos) + (_rotate_half(k) * sin)
    return qe.half(), ke.half()  # autocast casts to fp16 for SDPA


def _packed_positions(T, gen):
    out, left = [], T
    while left > 0:
        n = min(left, int(torch.randint(1, 600, (1,), generator=gen)))
        out.append(torch.arange(n))
        left -= n
    return torch.cat(out)


@pytest.mark.parametrize("T", TS)
@pytest.mark.parametrize("layout", ["contiguous", "strided_qkv"])
@pytest.mark.parametrize("with_delta", [False, True])
@pytest.mark.parametrize("pos_dtype", [torch.int64, torch.int32])
def test_rope(T, layout, with_delta, pos_dtype):
    gen = torch.Generator().manual_seed(T)
    torch.manual_seed(T)
    pos = _packed_positions(T, gen).to(DEV, pos_dtype)
    inv_freq = default_inv_freq(D, THETA, DEV)
    if layout == "contiguous":
        q = torch.randn(T, HQ * D, device=DEV).half().requires_grad_()
        k = torch.randn(T, HK * D, device=DEV).half().requires_grad_()
        qv, kv = q.view(T, HQ, D), k.view(T, HK, D)
        leaves = [q, k]
    else:
        qkv = torch.randn(T, (HQ + 2 * HK) * D, device=DEV).half().requires_grad_()
        qv = qkv[:, : HQ * D].view(T, HQ, D)
        kv = qkv[:, HQ * D:(HQ + HK) * D].view(T, HK, D)
        assert T == 1 or not qv.is_contiguous()
        leaves = [qkv]
    qd = None
    if with_delta:
        qd = (0.3 * torch.randn(T, HQ * D, device=DEV)).half().requires_grad_()
        leaves.append(qd)

    qo, ko = rope(qv, kv, pos, inv_freq, qd)
    qr, kr = ref_rope(qv, kv, pos, inv_freq, qd)
    assert qo.shape == (T, HQ, D) and ko.shape == (T, HK, D) and qo.is_contiguous()
    assert qo.dtype == torch.float16 and ko.dtype == torch.float16
    for a, b in ((qo, qr), (ko, kr)):
        ae, re = _err(a, b)
        assert ae <= 4e-3, (ae, re)  # fp16 output rounding (|x| up to ~6)

    gq, gk = torch.randn_like(qo), torch.randn_like(ko)
    g = torch.autograd.grad([qo, ko], leaves, [gq, gk])
    gr = torch.autograd.grad([qr, kr], leaves, [gq, gk])
    for a, b in zip(g, gr):
        ae, re = _err(a, b)
        assert ae <= 4e-3, (ae, re)
    if with_delta:
        # grad of the delta equals grad of q before rotation
        dq_view = g[0][:, : HQ * D] if layout == "strided_qkv" else g[0]
        torch.testing.assert_close(g[-1].view(T, HQ * D), dq_view.view(T, HQ * D),
                                   rtol=0, atol=0)


def test_rope_cos_sin_match_torch():
    """In-kernel angles/cos/sin against torch fp32 at large positions: rotate unit vectors."""
    T = 2048
    pos = torch.arange(T, device=DEV) * 16  # up to 32760, beyond the run's 512
    inv_freq = default_inv_freq(D, THETA, DEV)
    q = torch.zeros(T, 1, D, device=DEV, dtype=torch.float16)
    q[..., : D // 2] = 1.0  # x1 = 1, x2 = 0 -> out1 = cos, out2 = sin
    k = q.clone()
    qo, _ = rope(q, k, pos, inv_freq)
    ang = pos.float()[:, None] * inv_freq[None, :]
    torch.testing.assert_close(qo[:, 0, : D // 2].float(), ang.cos().half().float(),
                               rtol=0, atol=1e-3)
    torch.testing.assert_close(qo[:, 0, D // 2:].float(), ang.sin().half().float(),
                               rtol=0, atol=1e-3)


def test_inv_freq_matches_hf():
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    cfg = Qwen2Config(hidden_size=H, num_attention_heads=HQ, num_key_value_heads=HK,
                      rope_theta=THETA, max_position_embeddings=32768)
    emb = Qwen2RotaryEmbedding(cfg)
    torch.testing.assert_close(emb.inv_freq, default_inv_freq(D, THETA), rtol=0, atol=0)


@pytest.mark.parametrize("n_out", [1, 2, 3, 5])
@pytest.mark.parametrize("unused", [False, True], ids=["all_used", "some_unused"])
def test_add_rmsnorm_fanout(n_out, unused):
    """n_out copies of y (one per consumer): backward sums their grads in the kernel (one
    launch, no autograd adds); unused copies (None grads) are skipped; matches the reference
    where the one y feeds every consumer and autograd sums."""
    from triton.runtime.jit import JITFunction

    T = 300
    torch.manual_seed(n_out)
    r_in = (torch.randn(1, T, H, device=DEV) * 3).requires_grad_()
    delta = torch.randn(1, T, H, device=DEV).half().requires_grad_()
    w = 1 + 0.1 * torch.randn(H, device=DEV)
    r_out, ys = add_rmsnorm(r_in, delta, w, EPS, torch.float16, n_out=n_out)
    if n_out == 1:
        ys = (ys,)
    assert len(ys) == n_out and all(y.data_ptr() == ys[0].data_ptr() for y in ys)
    rr, yr = ref_add_rmsnorm(r_in, delta, w, torch.float16)
    used = [i for i in range(n_out) if not (unused and i % 2 == 1)]
    gys = [torch.randn_like(yr) for _ in used]
    gr = torch.randn_like(rr)
    runs = []
    run = JITFunction.run

    def counting_run(self, *args, **kwargs):
        runs.append(self.fn.__name__)
        return run(self, *args, **kwargs)

    JITFunction.run = counting_run
    try:
        g = torch.autograd.grad([r_out] + [ys[i] for i in used], [r_in, delta], [gr] + gys)
    finally:
        JITFunction.run = run
    assert runs == ["_add_rmsnorm_bwd"], runs
    # Reference: the consumers' grads summed in fp32 (as the base does, where x is fp32) and
    # an fp32 norm output, so only the kernel's own rounding is measured.
    rr, yr32 = ref_add_rmsnorm(r_in, delta, w, torch.float32)
    gsum = torch.stack([gy.float() for gy in gys]).sum(0) if gys else torch.zeros_like(yr32)
    g_ref = torch.autograd.grad([rr, yr32], [r_in, delta], [gr, gsum])
    for a, b in zip(g, g_ref):
        assert a.dtype == b.dtype
        ae, re = _err(a, b)
        assert re < (5e-3 if a.dtype == torch.float16 else 1e-5), (ae, re)


def test_add_rmsnorm_fanout_no_grad_outputs():
    """Only r_out used (no y grads): backward is dr_out passed through; nothing used: no grads."""
    r_in = torch.randn(1, 8, H, device=DEV, requires_grad=True)
    delta = torch.randn(1, 8, H, device=DEV).half().requires_grad_()
    w = torch.ones(H, device=DEV)
    r_out, ys = add_rmsnorm(r_in, delta, w, EPS, torch.float16, n_out=3)
    gr = torch.randn_like(r_out)
    a, b = torch.autograd.grad(r_out, [r_in, delta], gr)
    torch.testing.assert_close(a, gr, rtol=0, atol=0)
    torch.testing.assert_close(b, gr.half(), rtol=0, atol=0)


def test_add_rmsnorm_fanout_fp32_sum_no_overflow():
    """Large fp16 consumer grads whose fp16 sum overflows (2 x 40000 > 65504): the kernel loads
    each dy as fp32 and sums in registers, so the result is finite and equals the fp32 math."""
    T = 16
    torch.manual_seed(0)
    r_in = torch.randn(1, T, H, device=DEV).requires_grad_()
    delta = torch.randn(1, T, H, device=DEV).half().requires_grad_()
    w = torch.ones(H, device=DEV)
    r_out, (y1, y2) = add_rmsnorm(r_in, delta, w, EPS, torch.float16, n_out=2)
    big = torch.full((1, T, H), 40000.0, device=DEV)
    big[..., ::2] = -40000.0
    big[..., 0] = 40000.0  # not exactly zero-mean per row
    g1, g2 = big.half(), big.half()
    assert torch.isinf(g1 + g2).any()  # what an fp16 sum of the two would give
    d_rin, _ = torch.autograd.grad([y1, y2], [r_in, delta], [g1, g2])
    rr, yr = ref_add_rmsnorm(r_in, delta, w, torch.float32)
    ref, = torch.autograd.grad(yr, [r_in], [g1.float() + g2.float()])
    assert torch.isfinite(d_rin).all()
    torch.testing.assert_close(d_rin, ref, rtol=1e-4, atol=1e-2)


# ---------------------------------------------------------------- K1 save_output mode

def _weight(kind, gen):
    """Norm weights [H]: "regular" (no exact channel: K = 0), "zeros_tiny" (zeros, tiny values
    around and below SPECIAL_W_MIN, one huge), "qwen_like" (the Qwen2.5-1.5B shape: |w| ~ 0.3-3
    with a few channels at 1e-5..1e-3)."""
    w = 1 + 0.1 * torch.randn(H, generator=gen)
    if kind == "zeros_tiny":
        w[[0, 7, 100]] = 0.0
        w[[1, 2]] = torch.tensor([1e-6, -3e-5])
        w[[3, 4]] = torch.tensor([2.0 ** -10 * 0.999, -2.0 ** -10 * 0.5])  # just below
        w[5] = 2.0 ** -10 * 1.001  # just above: recovered from y
        w[H - 1] = 2000.0  # |w| * sqrt(H) could overflow fp16: exact channel
    elif kind == "qwen_like":
        w = (0.3 + 2.7 * torch.rand(H, generator=gen)) * torch.sign(torch.randn(H, generator=gen))
        w[[11, 500, 1400]] = torch.tensor([2e-5, 6.5e-6, 9e-4])   # exact
        w[[12, 501]] = torch.tensor([2e-3, 4e-3])                  # recovered
    return w.to(DEV)


def _rows(T, outliers, gen):
    """fp32 residual rows: N(0, 3); with outliers, a few "massive activation" channels up to
    1e3 (the tiny-|w| channels among them, as in Qwen) and rows scaled up to |r| ~ 1e3."""
    r = torch.randn(T, H, generator=gen) * 3
    if outliers:
        r[:, [0, 11, 500, 3, 12]] *= 300.0
        r[: T // 2] *= 50.0
        r.clamp_(-1e3, 1e3)
    return r.to(DEV)


def _norm_grads(r_in, delta, w, gys, gr, save_output, out_dtype=torch.float16):
    r_in = r_in.clone().requires_grad_()
    delta = delta.clone().requires_grad_()
    r_out, ys = add_rmsnorm(r_in, delta, w, EPS, out_dtype, n_out=len(gys),
                            save_output=save_output)
    ys = ys if isinstance(ys, tuple) else (ys,)
    g = torch.autograd.grad([r_out, *ys], [r_in, delta], [gr, *gys])
    return r_out.detach(), ys[0].detach(), g


def _ref_grads_f64(r_in, delta, w, gys, gr):
    r = (r_in.double() + delta.double()).requires_grad_()
    y = w.double() * (r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + EPS))
    gsum = torch.stack([g.double() for g in gys]).sum(0)
    (dr,) = torch.autograd.grad(y, [r], [gsum])
    return dr + gr.double()


@pytest.mark.parametrize("T", TS)
@pytest.mark.parametrize("kind", ["regular", "zeros_tiny", "qwen_like"])
def test_add_rmsnorm_save_output(T, kind):
    """save_output: same forward (r_out, y bit-identical); saved: y and rstd (+ [T, K] exact
    xhat), not r_out; grads match the fp64 reference with the accepted fp16-level error and
    are finite with w == 0 channels."""
    from solution.body.norm_rope import special_channels

    gen = torch.Generator().manual_seed(T)
    w = _weight(kind, gen)
    slots, k = special_channels(w)
    want_k = {"regular": 0, "zeros_tiny": 8, "qwen_like": 3}[kind]
    assert k == want_k and int((slots >= 0).sum()) == k
    if k:
        assert sorted(slots[slots >= 0].tolist()) == list(range(k))
    assert special_channels(w)[0] is slots  # cached
    r_in = _rows(T, False, gen).view(1, T, H)
    delta = torch.randn(1, T, H, generator=gen).half().to(DEV)
    gys = [torch.randn(1, T, H, generator=gen).half().to(DEV) for _ in range(2)]
    gr = torch.randn(1, T, H, generator=gen).to(DEV)

    saved = []

    def pack(x):
        saved.append((x.dtype, tuple(x.shape)))
        return x

    r_in_g = r_in.clone().requires_grad_()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        add_rmsnorm(r_in_g, delta.clone().requires_grad_(), w, EPS, torch.float16, n_out=2,
                    save_output=True)
    expect = [(torch.float16, (T, H)), (torch.float32, (T,)), (torch.float32, (H,)),
              (torch.int32, (H,))] + ([(torch.float32, (T, k))] if k else [])
    assert saved == expect, saved

    r0, y0, g_old = _norm_grads(r_in, delta, w, gys, gr, False)
    r1, y1, g_new = _norm_grads(r_in, delta, w, gys, gr, True)
    assert torch.equal(r0, r1) and torch.equal(y0, y1)
    ref = _ref_grads_f64(r_in, delta, w, gys, gr)
    for g in g_new:
        assert torch.isfinite(g).all()
    _, re_old = _err(g_old[0], ref)
    _, re_new = _err(g_new[0], ref)
    print(f"\n[save_output {kind} T={T} K={k}] d_r_in max rel err vs fp64: fp32-residual"
          f" {re_old:.2e}, from y {re_new:.2e}")
    assert re_new < 2e-3
    assert _err(g_new[1], ref)[1] < 5e-3  # fp16 d_delta


def test_add_rmsnorm_save_output_precision():
    """The gradient error of the y-based backward vs the fp32-residual one, realistic
    magnitudes: Qwen-like weights (tiny |w| at massive-activation channels), rows with
    |r| up to 1e3 and outlier channels. Measured on the norm part of d_r_in (dr_out = 0, the
    hard case: nothing larger to hide the error), relative to the max of the fp64 reference,
    per row (worst row) and over the whole tensor."""
    T = 1000
    gen = torch.Generator().manual_seed(1)
    w = _weight("qwen_like", gen)
    out = {}
    for outliers in (False, True):
        r_in = _rows(T, outliers, gen).view(1, T, H)
        delta = (torch.randn(1, T, H, generator=gen) * 0.5).half().to(DEV)
        gys = [torch.randn(1, T, H, generator=gen).half().to(DEV) * 1e-2 for _ in range(5)]
        gr = torch.zeros(1, T, H, device=DEV)
        ref = _ref_grads_f64(r_in, delta, w, gys, gr)[0]
        res = {}
        for mode in (False, True):
            g = _norm_grads(r_in, delta, w, gys, gr, mode)[2][0][0].double()
            row_rel = ((g - ref).abs().amax(1) / ref.abs().amax(1)).max().item()
            res[mode] = (_err(g, ref)[1], row_rel)
        out[outliers] = res
        print(f"\n[precision outliers={outliers}] d_r_in rel err vs fp64 (tensor / worst row):"
              f" fp32 residual {res[False][0]:.2e} / {res[False][1]:.2e},"
              f" from fp16 y {res[True][0]:.2e} / {res[True][1]:.2e}")
        assert res[True][0] < 2e-3 and res[True][1] < 5e-3


def test_add_rmsnorm_save_output_ignored():
    """save_output only applies with a delta and an fp16 y: no delta (layer 0) and fp32 y
    (the final norm) keep saving r_out."""
    for delta, dt in ((None, torch.float16), (torch.randn(4, H, device=DEV).half(), torch.float32)):
        saved = []
        r_in = torch.randn(4, H, device=DEV, requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(
                lambda x: (saved.append((x.dtype, tuple(x.shape))), x)[1], lambda x: x):
            add_rmsnorm(r_in, delta, torch.ones(H, device=DEV), EPS, dt, save_output=True)
        assert saved == [(torch.float32, (4, H)), (torch.float32, (4,)), (torch.float32, (H,))]


def test_special_channels_cache_invalidation():
    from solution.body.norm_rope import special_channels

    w = torch.ones(H, device=DEV)
    assert special_channels(w)[1] == 0
    w[3] = 0.0  # in place: version bump -> recomputed
    slots, k = special_channels(w)
    assert k == 1 and slots[3].item() == 0


def test_add_rmsnorm_save_output_needs_exact_channels(monkeypatch):
    """Negative control: without the exact channels (threshold 0), a w == 0 channel gives a
    non-finite gradient (0 / 0) and tiny |w| with small xhat a large error, which the default
    threshold removes."""
    from solution.body import norm_rope

    gen = torch.Generator().manual_seed(5)
    T = 64
    r_in = _rows(T, False, gen).view(1, T, H)
    delta = torch.randn(1, T, H, generator=gen).half().to(DEV)
    gys = [torch.randn(1, T, H, generator=gen).half().to(DEV)]
    gr = torch.zeros(1, T, H, device=DEV)
    for tiny in (0.0, 1e-6):
        w = _weight("regular", gen)
        w[[9, 99]] = tiny
        ref = _ref_grads_f64(r_in, delta, w, gys, gr)
        monkeypatch.setattr(norm_rope, "SPECIAL_W_MIN", 0.0)
        bad = _norm_grads(r_in, delta, w.clone(), gys, gr, True)[2][0]
        monkeypatch.setattr(norm_rope, "SPECIAL_W_MIN", 2.0 ** -10)
        good = _norm_grads(r_in, delta, w.clone(), gys, gr, True)[2][0]
        assert torch.isfinite(good).all() and _err(good, ref)[1] < 2e-3
        if tiny == 0.0:
            assert not torch.isfinite(bad).all()
        else:
            print(f"\n[no exact channels, |w| = 1e-6] rel err {_err(bad, ref)[1]:.2e}"
                  f" vs {_err(good, ref)[1]:.2e} with them")
            assert _err(bad, ref)[1] > 10 * _err(good, ref)[1]
