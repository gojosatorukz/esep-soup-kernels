"""Fused multi-projection NF4 linear (linear4bit_multi / dequant_multi) on the real Qwen2.5-1.5B
shapes: q/k/v (N = 1536, 256, 256, K = 1536, with biases) and gate/up (N = 8960, 8960, no bias)."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

bnb = pytest.importorskip("bitsandbytes")
import bitsandbytes.functional as BF  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from solution import nf4  # noqa: E402
from solution.nf4 import (dequant_multi, dequantize_nf4, dequantize_nf4_multi,  # noqa: E402
                          linear4bit, linear4bit_multi)

REPO = Path(__file__).resolve().parents[1]
K = 1536
GROUPS = {"qkv": ((1536, 256, 256), True), "gateup": ((8960, 8960), False)}


def _make(n, bias, seed):
    torch.manual_seed(seed)
    lin = bnb.nn.Linear4bit(K, n, bias=bias, compute_dtype=torch.float16, quant_type="nf4",
                            compress_statistics=True)
    w = torch.randn(n, K) * 0.02
    w[0, :64] *= 50  # an outlier block: exercises the nested absmax range
    lin.weight = bnb.nn.Params4bit(w.half(), requires_grad=False, quant_type="nf4",
                                   compress_statistics=True, blocksize=64)
    if bias:
        lin.bias = torch.nn.Parameter(torch.randn(n) * 0.1, requires_grad=False)
    lin = lin.cuda()
    lin(torch.randn(1, K, device="cuda", dtype=torch.float16))  # bnb fixes compute dtype (fp16)
    return lin


_cache = {}


def _group(name):
    if name not in _cache:
        ns, bias = GROUPS[name]
        _cache[name] = tuple(_make(n, bias, seed=10 + i) for i, n in enumerate(ns))
    return _cache[name]


def _ref_weights(mods):
    return [BF.dequantize_4bit(m.weight, m.weight.quant_state) for m in mods]


@pytest.mark.parametrize("name", list(GROUPS))
def test_multi_dequant_bit_exact(name):
    mods = _group(name)
    refs = _ref_weights(mods)
    for w in (dequant_multi(mods),
              dequantize_nf4_multi([m.weight for m in mods], [m.weight.quant_state for m in mods])):
        assert w.dtype == torch.float16 and w.is_contiguous()
        assert w.shape == (sum(GROUPS[name][0]), K)
        r0 = 0
        for m, ref in zip(mods, refs):
            n = ref.shape[0]
            assert torch.equal(w[r0:r0 + n], ref)
            assert torch.equal(w[r0:r0 + n], dequantize_nf4(m.weight, m.weight.quant_state))
            r0 += n
    # two of the three q/k/v weights (k, v): the 2-weight form of the launch
    if name == "qkv":
        w = dequant_multi(mods[1:])
        assert torch.equal(w, torch.cat(refs[1:]))


def _sep(x, mods):
    return [linear4bit(x, m, keep_fp16=True) for m in mods]


@pytest.mark.parametrize("name", list(GROUPS))
@pytest.mark.parametrize("lead", [(600,), (2, 413)])
def test_forward_backward(name, lead):
    mods = _group(name)
    ns = GROUPS[name][0]
    x = torch.randn(*lead, K, device="cuda", dtype=torch.float16)
    gs = [torch.randn(*lead, n, device="cuda", dtype=torch.float16) for n in ns]

    calls = []
    orig = nf4.NF4MultiLinear.forward
    nf4.NF4MultiLinear.forward = staticmethod(lambda *a: calls.append(1) or orig(*a))
    try:
        xm = x.clone().requires_grad_()
        ys = linear4bit_multi(xm, mods)
        torch.autograd.backward(ys, gs)
    finally:
        nf4.NF4MultiLinear.forward = staticmethod(orig)
    assert calls == [1], "fused path not taken"
    assert isinstance(ys, tuple) and len(ys) == len(mods)

    xs = x.clone().requires_grad_()
    yr = _sep(xs, mods)
    torch.autograd.backward(yr, gs)

    for y, r, n in zip(ys, yr, ns):
        assert y.dtype == torch.float16 and y.shape == (*lead, n)
        # one GEMM over W_cat vs a GEMM per weight: same fp32-accumulated dot products; cuBLAS
        # may tile them differently, so allow a last-bit fp16 difference
        torch.testing.assert_close(y, r, rtol=1e-3, atol=1e-3)
    # fp32 reference on the same dequantized weights
    w = torch.cat(_ref_weights(mods)).float()
    b = None if mods[0].bias is None else torch.cat([m.bias for m in mods]).float()
    ref = torch.nn.functional.linear(x.float(), w, b)
    torch.testing.assert_close(torch.cat(ys, -1).float(), ref, rtol=2e-2, atol=2e-2)
    # grad_x: one dX GEMM over sum(N) vs the sum of per-weight fp16 grads; fp32 reference
    assert xm.grad.dtype == torch.float16
    gref = torch.cat(gs, -1).float() @ w
    torch.testing.assert_close(xm.grad.float(), gref, rtol=2e-2, atol=5e-2)
    torch.testing.assert_close(xm.grad, xs.grad, rtol=2e-2, atol=5e-2)
    assert (xm.grad.float() - gref).abs().max() <= (xs.grad.float() - gref).abs().max() * 1.5


def test_forward_bitwise_report(capsys):
    """Reports (does not assert) whether fused forward columns are bit-identical to separate."""
    for name in GROUPS:
        mods = _group(name)
        x = torch.randn(600, K, device="cuda", dtype=torch.float16)
        ym = linear4bit_multi(x, mods, split=False)
        yr = torch.cat(_sep(x, mods), -1)
        with capsys.disabled():
            d = (ym.float() - yr.float()).abs().max().item()
            print(f"\n[nf4_multi] {name}: fwd bit-identical={torch.equal(ym, yr)} maxdiff={d:.3g}")


def test_split_false_and_strided_grad():
    mods = _group("qkv")
    x = torch.randn(2, 300, K, device="cuda", dtype=torch.float16)
    xa = x.clone().requires_grad_()
    y = linear4bit_multi(xa, mods, split=False)
    assert y.shape == (2, 300, 2048)
    g = torch.randn(300, 2, 2048, device="cuda", dtype=torch.float16).transpose(0, 1)
    assert not g.is_contiguous()
    y.backward(g)
    xb = x.clone().requires_grad_()
    torch.cat(linear4bit_multi(xb, mods), -1).backward(g.contiguous())
    # a strided dY takes another cuBLAS layout: same values up to fp16 rounding
    torch.testing.assert_close(xa.grad, xb.grad, rtol=2e-3, atol=2e-3)
    gref = g.float() @ torch.cat(_ref_weights(mods)).float()
    torch.testing.assert_close(xa.grad.float(), gref, rtol=2e-2, atol=5e-2)


def test_fp32_input_and_bias_cache():
    mods = _group("qkv")
    x32 = torch.randn(600, K, device="cuda")
    x = x32.clone().requires_grad_()
    ys = linear4bit_multi(x, mods)
    assert all(y.dtype == torch.float16 for y in ys)
    sum(y.float().sum() for y in ys).backward()
    assert x.grad.dtype == torch.float32
    yh = linear4bit_multi(x32.half(), mods)
    assert all(torch.equal(a, b) for a, b in zip(ys, yh))
    cache = mods[0].__dict__[nf4._GROUP_ATTR][tuple(id(m) for m in mods)]
    bc = cache["bias"][1]
    linear4bit_multi(x32.half(), mods)
    assert cache["bias"][1] is bc  # cached
    with torch.no_grad():
        mods[2].bias.add_(1.0)  # in-place change -> version bump -> rebuilt
    try:
        y = linear4bit_multi(x32.half(), mods)
        assert cache["bias"][1] is not bc
        torch.testing.assert_close(y[2], linear4bit(x32.half(), mods[2], keep_fp16=True),
                                   rtol=1e-3, atol=1e-3)
    finally:
        with torch.no_grad():
            mods[2].bias.sub_(1.0)


def test_trainable_bias_gets_grad():
    mods = tuple(_make(n, True, seed=30 + i) for i, n in enumerate((256, 256)))
    for m in mods:
        m.bias.requires_grad_(True)
    x = torch.randn(64, K, device="cuda", dtype=torch.float16)
    ys = linear4bit_multi(x, mods)
    sum(y.float().sum() for y in ys).backward()
    for m in mods:
        assert m.bias.grad is not None
        torch.testing.assert_close(m.bias.grad, torch.full_like(m.bias, 64.0))


def test_fallbacks():
    qkv = _group("qkv")
    calls = []
    orig = nf4.NF4MultiLinear.forward
    nf4.NF4MultiLinear.forward = staticmethod(lambda *a: calls.append(1) or orig(*a))
    try:
        # too few rows -> separate linear4bit calls (bnb's own path), still the same values
        x = torch.randn(2, K, device="cuda", dtype=torch.float16)
        ys = linear4bit_multi(x, qkv)
        assert len(ys) == 3 and all(torch.equal(a, b) for a, b in zip(ys, _sep(x, qkv)))
        # mixed bias / no bias
        nobias = _group("gateup")[0]
        x = torch.randn(64, K, device="cuda", dtype=torch.float16)
        ys = linear4bit_multi(x, (qkv[0], nobias))
        assert all(torch.equal(a, b) for a, b in zip(ys, _sep(x, (qkv[0], nobias))))
        y = linear4bit_multi(x, (qkv[0], nobias), split=False)
        assert y.shape == (64, 1536 + 8960)
        # one module
        (y1,) = linear4bit_multi(x, qkv[:1])
        assert torch.equal(y1, linear4bit(x, qkv[0], keep_fp16=True))
    finally:
        nf4.NF4MultiLinear.forward = staticmethod(orig)
    assert calls == []


_SCRIPT = textwrap.dedent("""
    import threading
    import torch, triton
    import bitsandbytes as bnb
    from solution import nf4

    compiles = []
    triton.knobs.runtime.jit_post_compile_hook = (
        lambda *, key, fn, **kw: compiles.append(fn.name))
    nf4.warmup(torch.device("cuda", torch.cuda.current_device()))
    before = list(compiles)
    mods = []
    for n, bias in [(1536, True), (256, True), (256, True), (8960, False), (8960, False)]:
        lin = bnb.nn.Linear4bit(1536, n, bias=bias, compute_dtype=torch.float16,
                                quant_type="nf4", compress_statistics=True)
        lin.weight = bnb.nn.Params4bit((torch.randn(n, 1536) * 0.02).half(), requires_grad=False,
                                       quant_type="nf4", compress_statistics=True, blocksize=64)
        mods.append(lin.cuda())
    n0 = len(compiles)  # bnb's quantize does not launch our kernels
    for T in (600, 826, 1427):
        x = torch.randn(1, T, 1536, device="cuda", dtype=torch.float16, requires_grad=True)
        for grp in (mods[:3], mods[3:], mods[1:3]):
            ys = nf4.linear4bit_multi(x, grp)
            torch.autograd.backward(ys, [torch.ones_like(y) for y in ys])
            nf4.dequant_multi(grp)
    torch.cuda.synchronize()
    after = [c for c in compiles[n0:] if c.startswith("nf4_")]
    print("BEFORE", before, "AFTER", after)
    assert sorted(before) == ["nf4_dequant_kernel", "nf4_dequant_multi_kernel"], before
    assert after == [], after
""")


def test_warmup_covers_multi_variant(tmp_path):
    env = dict(os.environ, TRITON_CACHE_DIR=str(tmp_path / "triton"), PYTHONPATH=str(REPO))
    r = subprocess.run([sys.executable, "-c", _SCRIPT], cwd=REPO, env=env, capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
