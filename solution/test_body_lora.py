"""Tests for solution/body/lora.py (LoRA dropout + down/up path), fp16 on CUDA.

Run: python -m pytest solution/test_body_lora.py -q   (add -s to see the error table)."""

import contextlib

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from solution.body import lora as L  # noqa: E402

IN = 1536
R = 16
SCALING = 2.0
P = 0.05


def _errs(got, ref):
    d = (got.float() - ref.float()).abs().max().item()
    scale = ref.float().abs().max().item()
    return d, d / max(scale, 1e-12)


def _check(name, got, ref, tol=1e-2):
    a, r = _errs(got, ref)
    print(f"{name:<34s} max_abs={a:.3e} max_rel={r:.3e}")
    assert r < tol, (name, a, r)


def _inputs(T, N, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(T, IN, device="cuda", generator=g).half()
    base = torch.randn(T, N, device="cuda", generator=g).half()
    a = (torch.randn(R, IN, device="cuda", generator=g) * 0.02)
    b = (torch.randn(N, R, device="cuda", generator=g) * 0.02)
    dout = torch.randn(T, N, device="cuda", generator=g).half()
    return x, base, a, b, dout


def _reference(x, base, a, b, dout, mask=None, p=0.0):
    """fp32 PyTorch reference; mask is the keep mask (bool) or None."""
    xr = x.float().requires_grad_(True)
    br = base.float().requires_grad_(True)
    ar = a.clone().requires_grad_(True)
    bb = b.clone().requires_grad_(True)
    xd = xr * mask / (1.0 - p) if mask is not None else xr
    out = br + SCALING * (xd @ ar.t()) @ bb.t()
    out.backward(dout.float())
    return out, xr.grad, br.grad, ar.grad, bb.grad


def _run_ours(x, base, a, b, dout, p):
    xo = x.clone().requires_grad_(True)
    bo = base.clone().requires_grad_(True)
    ao = a.clone().requires_grad_(True)
    bbo = b.clone().requires_grad_(True)
    out = L.LoRAFunction.apply(xo, bo * 1, ao, bbo, p, SCALING)  # bo * 1: a fresh non-leaf
    out.backward(dout)
    return out, xo.grad, bo.grad, ao.grad, bbo.grad


@pytest.mark.parametrize("N", [1536, 256])
@pytest.mark.parametrize("T", [1, 37, 1000, 2048])
def test_no_dropout_vs_reference(T, N):
    x, base, a, b, dout = _inputs(T, N)
    out, dx, dbase, da, db = _run_ours(x, base, a, b, dout, 0.0)
    r_out, r_dx, r_dbase, r_da, r_db = _reference(x, base, a, b, dout)
    assert out.dtype == torch.float16 and dx.dtype == torch.float16
    assert da.dtype == torch.float32 and db.dtype == torch.float32
    _check(f"T={T} N={N} out", out, r_out)
    _check(f"T={T} N={N} dx", dx, r_dx)
    _check(f"T={T} N={N} dbase", dbase, r_dbase, tol=1e-6)
    _check(f"T={T} N={N} dA", da, r_da)
    _check(f"T={T} N={N} dB", db, r_db)


def _capture_keys(monkeypatch):
    keys = []
    orig = L.new_dropout_key

    def spy(device):
        k = orig(device)
        keys.append(k.clone())
        return k

    monkeypatch.setattr(L, "new_dropout_key", spy)
    return keys


def _keep_mask(key, shape, p=P):
    ones = torch.ones(shape, device="cuda", dtype=torch.float16)
    return L.lora_dropout(ones, key, p) != 0


@pytest.mark.parametrize("N", [1536, 256])
@pytest.mark.parametrize("T", [1, 37, 1000, 2048])
def test_dropout_fwd_bwd_same_mask(T, N, monkeypatch):
    keys = _capture_keys(monkeypatch)
    x, base, a, b, dout = _inputs(T, N, seed=1)
    out, dx, dbase, da, db = _run_ours(x, base, a, b, dout, P)
    assert len(keys) == 1
    mask = _keep_mask(keys[0], x.shape)
    r_out, r_dx, r_dbase, r_da, r_db = _reference(x, base, a, b, dout, mask=mask, p=P)
    _check(f"drop T={T} N={N} out", out, r_out)   # forward used this mask
    _check(f"drop T={T} N={N} dx", dx, r_dx)      # backward regenerated the same mask
    _check(f"drop T={T} N={N} dA", da, r_da)
    _check(f"drop T={T} N={N} dB", db, r_db)
    # the LoRA part of dx is exactly zero where the forward dropped the input
    assert torch.all(dx[~mask] == 0)


def test_dropout_distribution_and_scaling():
    T = 2048
    key = L.new_dropout_key("cuda")
    x = torch.randn(T, IN, device="cuda").half()
    xd = L.lora_dropout(x, key, P)
    keep = xd != 0
    keep &= x != 0
    rate = keep.float().mean().item()
    print(f"keep rate over {x.numel()} elements: {rate:.5f} (expected {1 - P})")
    # std of the rate is sqrt(.95*.05/3.1e6) = 1.2e-4
    assert abs(rate - (1 - P)) < 1e-3
    # rows and columns are not degenerate
    row_rate = keep.float().mean(1)
    col_rate = keep.float().mean(0)
    assert row_rate.min() > 0.9 and row_rate.max() < 0.99
    assert col_rate.min() > 0.9 and col_rate.max() < 0.99
    # kept values are x / (1 - p) (one fp16 rounding)
    expect = (x.float() / (1 - P)).half()
    assert torch.equal(xd[keep], expect[keep])
    # same key -> same mask; deterministic
    assert torch.equal(L.lora_dropout(x, key, P), xd)


def test_fresh_mask_per_forward_call(monkeypatch):
    keys = _capture_keys(monkeypatch)
    x, base, a, b, _ = _inputs(256, 256, seed=2)
    L.LoRAFunction.apply(x, base.clone(), a, b, P, SCALING)
    L.LoRAFunction.apply(x, base.clone(), a, b, P, SCALING)
    assert len(keys) == 2 and keys[0].item() != keys[1].item()
    m0 = _keep_mask(keys[0], x.shape)
    m1 = _keep_mask(keys[1], x.shape)
    assert not torch.equal(m0, m1)
    # independent masks: overlap of drops close to p^2
    both = (~m0 & ~m1).float().mean().item()
    assert abs(both - P * P) < 2e-3


def test_keys_follow_torch_cuda_seed():
    torch.manual_seed(1234)
    k0 = L.new_dropout_key("cuda").item()
    torch.manual_seed(1234)
    k1 = L.new_dropout_key("cuda").item()
    assert k0 == k1


def test_dx_accumulate_into_buffer():
    T = 300
    x, _, a, _, _ = _inputs(T, 256, seed=3)
    a16 = a.half()
    dxa = torch.randn(T, R, device="cuda").half()
    key = L.new_dropout_key("cuda")
    for p in (P, 0.0):
        acc = torch.randn(T, IN, device="cuda").half()
        acc0 = acc.clone()
        dx_sep, da_sep = L.lora_down_bwd(x, a16, dxa, key, p)
        dx_acc, da_acc = L.lora_down_bwd(x, a16, dxa, key, p, dx_accum=acc)
        assert dx_acc.data_ptr() == acc.data_ptr()
        _check(f"accum p={p} dx", dx_acc, acc0.float() + dx_sep.float(), tol=2e-3)
        assert torch.equal(da_sep, da_acc)
        _, da_nodx = L.lora_down_bwd(x, a16, dxa, key, p, need_dx=False)
        assert torch.equal(da_sep, da_nodx)


def test_weight_cast():
    a = torch.randn(R, IN, device="cuda")
    b = torch.randn(256, R, device="cuda")
    a16, b16 = L.cast_lora_weights(a, b)
    assert torch.equal(a16, a.half()) and torch.equal(b16, b.half())


@contextlib.contextmanager
def _no_torchao_probe():
    # peft probes its torchao dispatcher for nn.Linear targets; with an old torchao (Kaggle has
    # 0.10) peft's is_torchao_available() raises ImportError. Switch the probe off around
    # get_peft_model, then restore it.
    import peft.tuners.lora.torchao as peft_torchao

    probe = peft_torchao.is_torchao_available
    peft_torchao.is_torchao_available = lambda: False
    try:
        yield
    finally:
        peft_torchao.is_torchao_available = probe


def _peft_model(N, dropout):
    peft = pytest.importorskip("peft")

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(IN, IN, bias=True)
            self.v_proj = torch.nn.Linear(IN, N, bias=True)

        def forward(self, x):
            return self.q_proj(x), self.v_proj(x)

    torch.manual_seed(0)
    model = Tiny().cuda()
    cfg = peft.LoraConfig(r=R, lora_alpha=32, lora_dropout=dropout,
                          target_modules=["q_proj", "v_proj"])
    with _no_torchao_probe():
        model = peft.get_peft_model(model, cfg)
    for mod in (model.base_model.model.q_proj, model.base_model.model.v_proj):
        torch.nn.init.normal_(mod.lora_B["default"].weight, std=0.02)
    return model


@pytest.mark.parametrize("T", [37, 1000])
def test_vs_peft_lora_dropout_zero(T):
    model = _peft_model(256, 0.0)
    model.train()
    x = torch.randn(T, IN, device="cuda").half()
    for name in ("q_proj", "v_proj"):
        mod = getattr(model.base_model.model, name)
        la, lb = mod.lora_A["default"].weight, mod.lora_B["default"].weight
        assert la.dtype == torch.float32
        assert L.lora_params(mod)[2:] == (0.0, SCALING)
        dout = torch.randn(T, mod.out_features, device="cuda").half()
        # peft's own path under autocast fp16
        xp = x.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.float16):
            ref = mod(xp)
        ref.backward(dout)
        ref_da, ref_db, ref_dx = la.grad.clone(), lb.grad.clone(), xp.grad.clone()
        la.grad = lb.grad = None
        # ours, with the same base layer
        xo = x.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.float16):
            base = mod.base_layer(xo)
            out = L.lora_forward(mod, xo, base)
        out.backward(dout)
        _check(f"peft T={T} {name} out", out, ref)
        _check(f"peft T={T} {name} dx", xo.grad, ref_dx)
        _check(f"peft T={T} {name} dA", la.grad, ref_da, tol=2e-2)
        _check(f"peft T={T} {name} dB", lb.grad, ref_db, tol=2e-2)
        assert la.grad.dtype == torch.float32
        la.grad = lb.grad = None


def test_peft_dropout_module_modes():
    model = _peft_model(256, P)
    mod = model.base_model.model.q_proj
    model.train()
    assert L.lora_params(mod)[2] == pytest.approx(P)
    model.eval()
    assert L.lora_params(mod)[2] == 0.0  # eval: no mask, like nn.Dropout in eval mode
