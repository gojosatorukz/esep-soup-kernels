"""NF4 dequant kernel and Linear4bit replacement against bitsandbytes itself."""

import pytest
import torch

bnb = pytest.importorskip("bitsandbytes")
import bitsandbytes.functional as BF  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from solution import nf4  # noqa: E402
from solution.nf4 import dequantize_nf4, nf4_linear, supported  # noqa: E402

SHAPES = [(1536, 1536), (256, 1536), (8960, 1536), (1536, 8960)]  # (N, K) of Qwen2.5-1.5B


def _quant(n, k, seed=0, dtype=torch.float16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.02).to(dtype)
    w[0, :64] *= 50  # one outlier block so the nested absmax range is exercised
    packed, qs = BF.quantize_4bit(w, quant_type="nf4", compress_statistics=True, blocksize=64)
    return packed, qs


@pytest.fixture
def unpatched():
    """Run a test with the original Linear4bit.forward, then restore the patch state."""
    from bitsandbytes.nn import Linear4bit
    saved = Linear4bit.forward
    if nf4._orig_forward is not None:
        Linear4bit.forward = nf4._orig_forward
    yield
    Linear4bit.forward = saved


@pytest.mark.parametrize("n,k", SHAPES)
def test_dequant_bit_exact(n, k):
    packed, qs = _quant(n, k)
    assert supported(qs)
    ref = BF.dequantize_4bit(packed, qs)
    assert ref.dtype == torch.float16
    assert torch.equal(dequantize_nf4(packed, qs), ref)
    # the transposed packed view Linear4bit passes around is the same bytes
    assert torch.equal(dequantize_nf4(packed.t(), qs), ref)


@pytest.mark.parametrize("n,k", SHAPES[:2])
def test_dequant_bit_exact_fp32_state(n, k):
    """fp32 quant state: bnb dequantizes to fp32; ours equals that value rounded to fp16."""
    packed, qs = _quant(n, k, seed=1, dtype=torch.float32)
    ref = BF.dequantize_4bit(packed, qs)
    assert ref.dtype == torch.float32
    assert torch.equal(dequantize_nf4(packed, qs), ref.half())


def _bnb_matmul(x, packed, qs, bias=None):
    return bnb.matmul_4bit(x, packed.t(), bias=bias, quant_state=qs)


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("lead", [(600,), (2048,), (4, 512)])
def test_autograd_matches_bnb(n, k, lead):
    packed, qs = _quant(n, k, seed=2)
    x = torch.randn(*lead, k, device="cuda", dtype=torch.float16)
    bias = torch.randn(n, device="cuda", dtype=torch.float16) * 0.1
    g = torch.randn(*lead, n, device="cuda", dtype=torch.float16)

    x1 = x.clone().requires_grad_()
    y1 = _bnb_matmul(x1, packed, qs, bias)
    y1.backward(g)
    x2 = x.clone().requires_grad_()
    y2 = nf4_linear(x2, packed, bias, qs)
    y2.backward(g)

    assert y2.dtype == y1.dtype == torch.float16 and y2.shape == y1.shape
    torch.testing.assert_close(y2, y1, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(x2.grad, x1.grad, rtol=2e-3, atol=2e-3)
    # against an fp32 reference on the same dequantized weight
    w = BF.dequantize_4bit(packed, qs).float()
    ref = torch.nn.functional.linear(x.float(), w, bias.float())
    torch.testing.assert_close(y2.float(), ref, rtol=2e-2, atol=2e-2)


def _make_linear4bit(k, n, bias, seed=3):
    torch.manual_seed(seed)
    lin = bnb.nn.Linear4bit(k, n, bias=bias, compute_dtype=torch.float16, quant_type="nf4",
                            compress_statistics=True)
    w = torch.randn(n, k) * 0.02
    lin.weight = bnb.nn.Params4bit(w.half(), requires_grad=False, quant_type="nf4",
                                   compress_statistics=True, blocksize=64)
    if bias:
        lin.bias = torch.nn.Parameter(torch.randn(n) * 0.1, requires_grad=False)
    return lin.cuda()


def _run(module, x, autocast):
    x = x.clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
        y = module(x)
    y.float().pow(2).mean().backward()
    return y.detach(), x.grad


@pytest.mark.parametrize("x_dtype,autocast", [(torch.float16, False), (torch.float32, True),
                                               (torch.float16, True)])
def test_linear4bit_module_patched_vs_bnb(unpatched, x_dtype, autocast):
    """Same module, same inputs: original bnb forward vs our patched forward (bit-exact)."""
    from bitsandbytes.nn import Linear4bit
    lin = _make_linear4bit(1536, 256, bias=True)
    x = torch.randn(4, 160, 1536, device="cuda").to(x_dtype)
    ref_y, ref_g = _run(lin, x, autocast)  # first call also fixes compute_dtype (bnb logic)
    ref_y, ref_g = _run(lin, x, autocast)
    assert nf4.patch() and nf4.patch()  # idempotent
    assert Linear4bit.forward is nf4._forward
    y, gx = _run(lin, x, autocast)
    assert y.dtype == ref_y.dtype == x_dtype
    assert torch.equal(y, ref_y)
    assert torch.equal(gx, ref_g)


def test_linear4bit_bit_exact_vs_bnb_function(unpatched):
    """linear4bit(x, module) equals bnb's forward exactly on the dequant+F.linear route."""
    lin = _make_linear4bit(1536, 8960, bias=False)
    x = torch.randn(2048, 1536, device="cuda", dtype=torch.float16)
    ref = lin(x)
    assert torch.equal(nf4.linear4bit(x, lin), ref)


def test_peft_lora_wrapper(unpatched):
    peft = pytest.importorskip("peft")

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = _make_linear4bit(1536, 1536, bias=True)
            self.is_loaded_in_4bit = True  # as a quantized HF model: peft then picks its bnb LoRA layer

        def forward(self, x):
            return self.q_proj(x)

    model = peft.get_peft_model(Tiny(), peft.LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                                        target_modules=["q_proj"]))
    for name, p in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=0.02)  # non-zero so the LoRA path matters
    x = torch.randn(4, 150, 1536, device="cuda", dtype=torch.float32)

    def step():
        model.zero_grad(set_to_none=True)
        xx = x.clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.float16):
            y = model(xx)
        y.float().pow(2).mean().backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        return y.detach(), xx.grad, grads

    step()
    ref_y, ref_gx, ref_grads = step()
    nf4.patch()
    calls = []
    orig = nf4.NF4Linear.forward
    nf4.NF4Linear.forward = staticmethod(lambda *a: calls.append(1) or orig(*a))
    try:
        y, gx, grads = step()
    finally:
        nf4.NF4Linear.forward = staticmethod(orig)
    assert calls, "patched path not taken inside the peft wrapper"
    assert torch.equal(y, ref_y)
    assert torch.equal(gx, ref_gx)
    assert ref_grads.keys() == grads.keys() and len(grads) == 2
    for n in grads:
        assert torch.equal(grads[n], ref_grads[n]), n


def test_fallbacks(unpatched):
    """Tiny batches (M < MIN_ROWS) go to the original bitsandbytes path."""
    nf4.patch()
    lin = _make_linear4bit(1536, 256, bias=False)
    x = torch.randn(1, 1536, device="cuda", dtype=torch.float16)
    lin(x)
    calls = []
    orig = nf4._orig_forward
    nf4._orig_forward = lambda m, xx: calls.append(1) or orig(m, xx)
    try:
        lin(x)
    finally:
        nf4._orig_forward = orig
    assert calls == [1]


def _fwd_bwd(fn, x, g):
    x = x.clone().requires_grad_()
    y = fn(x)
    y.backward(g)
    return y.detach(), x.grad


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("lead", [(600,), (4, 512)])
def test_linear4bit_keep_fp16(unpatched, n, k, lead):
    """keep_fp16=True returns bnb's fp16 output (before its upcast), bit-exact, grads included."""
    lin = _make_linear4bit(k, n, bias=True)
    x16 = torch.randn(*lead, k, device="cuda", dtype=torch.float16)
    x32 = x16.float() + torch.randn_like(x16, dtype=torch.float32) * 1e-3  # not fp16-exact
    g = torch.randn(*lead, n, device="cuda", dtype=torch.float16)
    lin(x16)  # warm-up: sets the quant state / compute dtype (fp16) as bnb does
    assert lin.compute_dtype is torch.float16

    # fp16 x: reference is the original bnb forward, which returns fp16 for fp16 input
    ref_y, ref_g = _fwd_bwd(lambda t: lin(t), x16, g)
    calls = []
    orig = nf4.NF4Linear.forward
    nf4.NF4Linear.forward = staticmethod(lambda *a: calls.append(1) or orig(*a))
    try:
        y, gx = _fwd_bwd(lambda t: nf4.linear4bit(t, lin, keep_fp16=True), x16, g)
    finally:
        nf4.NF4Linear.forward = staticmethod(orig)
    assert calls == [1], "our NF4 path was not taken"
    assert ref_y.dtype == y.dtype == torch.float16 and y.shape == ref_y.shape
    assert torch.equal(y, ref_y)
    assert gx.dtype == torch.float16 and torch.equal(gx, ref_g)

    # fp32 x: our fp16 result equals bnb's fp32 result rounded back (bnb's is an upcast fp16)
    ref_y32, ref_g32 = _fwd_bwd(lambda t: lin(t), x32, g.float())
    y, gx = _fwd_bwd(lambda t: nf4.linear4bit(t, lin, keep_fp16=True), x32, g)
    assert ref_y32.dtype == torch.float32 and y.dtype == torch.float16
    assert torch.equal(y, ref_y32.half())
    assert torch.equal(y.float(), ref_y32)
    assert gx.dtype == torch.float32 and torch.equal(gx, ref_g32)

    # default path unchanged: fp32 in, fp32 out, equal to bnb
    y_def = nf4.linear4bit(x32, lin)
    assert y_def.dtype == torch.float32 and torch.equal(y_def, ref_y32)


@pytest.mark.parametrize("x_dtype", [torch.float16, torch.float32])
def test_linear4bit_keep_fp16_fallback(unpatched, x_dtype):
    """Fallback (M < MIN_ROWS) with keep_fp16=True: bnb's result in fp16."""
    lin = _make_linear4bit(1536, 256, bias=True)
    lin(torch.randn(1, 1536, device="cuda", dtype=torch.float16))
    x = torch.randn(1, 1536, device="cuda", dtype=x_dtype)
    ref = lin(x)
    assert ref.dtype == x_dtype
    calls = []
    orig = nf4._orig_forward
    nf4._orig_forward = lambda m, xx: calls.append(1) or (orig or type(m).forward)(m, xx)
    try:
        y = nf4.linear4bit(x, lin, keep_fp16=True)
        y_def = nf4.linear4bit(x, lin)
    finally:
        nf4._orig_forward = orig
    assert calls == [1, 1]
    assert y.dtype == torch.float16 and torch.equal(y, ref.half())
    assert y_def.dtype == x_dtype and torch.equal(y_def, ref)
