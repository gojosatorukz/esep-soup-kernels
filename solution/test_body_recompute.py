"""The recompute MLP (packed.RECOMPUTE_MLP, packed._RecomputeMLP) against the no-recompute path.

The tiny NF4 + LoRA Qwen2 of test_body_packed (bnb NF4 Linear4bit everywhere, peft LoRA on q/v)
with the organizers' example SwiGLU patch on Qwen2MLP, as in the run. Compared per mode:
RECOMPUTE_MLP True vs False (both packed, our kernels) vs the original HF forward. The forward
is the same computation in both packed modes, so hidden states and loss must be bit-identical.
The backward recomputes gate/up with dequant + F.linear: bit-identical grads when the forward's
projections are dequant + F.linear too (shown with a stand-in for nf4's linear4bit); with plain
bitsandbytes at small M (its own GEMM kernel) the grads differ by fp16 noise only.
Saved tensors are counted with saved_tensors_hooks: with recompute no [T, intermediate] tensor
and no extra fp16 [T, hidden] x is saved.
"""

import contextlib

import pytest
import torch
import torch.nn.functional as F

from solution import test_body_packed as tp

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

INTER = 512  # intermediate_size of the tiny config


def _patch_all():
    from solution.body import patch
    from transformers.models.qwen2 import modeling_qwen2

    from solution.swiglu import patch_qwen2_mlp

    if not getattr(modeling_qwen2.Qwen2MLP, "_test_example_swiglu", False):
        patch_qwen2_mlp()
        modeling_qwen2.Qwen2MLP._test_example_swiglu = True
    patch()


@contextlib.contextmanager
def _recompute(on):
    from solution.body import packed

    old = packed.RECOMPUTE_MLP
    packed.RECOMPUTE_MLP = on
    try:
        yield
    finally:
        packed.RECOMPUTE_MLP = old


def _same(a, b):
    """(bitwise equal, max rel diff) over the run outputs."""
    eq = torch.equal(a["hidden"], b["hidden"]) and torch.equal(a["loss"], b["loss"])
    eq_g = all(torch.equal(a["grads"][n], b["grads"][n]) for n in a["grads"])
    g_rel = max(tp._rel(b["grads"][n], a["grads"][n]) for n in a["grads"])
    return eq, eq_g, g_rel


def _dequant_linear4bit(x, module, keep_fp16=False):
    """Stand-in for nf4's linear4bit: dequant + F.linear in fp16 (what nf4 computes)."""
    from solution.body import packed

    assert keep_fp16
    bias = None if module.bias is None else module.bias.to(torch.float16)
    return F.linear(x, packed._dequant(module), bias)


@pytest.mark.parametrize("hook", [False, True], ids=["bnb", "dequant_linear"])
@pytest.mark.parametrize("left", [False, True], ids=["right_pad", "left_pad"])
def test_recompute_matches(left, hook, monkeypatch):
    from solution.body import packed

    _patch_all()
    # Separate gate/up dequants in backward (bit-identical grads); the fused one: below.
    monkeypatch.setattr(packed, "_NF4_DEQUANT_MULTI", [None])
    if hook:
        monkeypatch.setattr(packed, "_NF4_LINEAR", [_dequant_linear4bit])
    model = tp._build("nf4_lora")
    ids, mask, labels = tp._batch(left)
    with tp._original_forward():
        ref = tp._run(model, ids, mask, labels)
    with _recompute(False):
        off = tp._run(model, ids, mask, labels)
    with _recompute(True), tp._count_launches() as counts:
        on = tp._run(model, ids, mask, labels)
    tag = f"{'left' if left else 'right'} {'dequant+linear' if hook else 'bnb'}"
    tp._compare("nf4_lora", ref, off, mask, f"recompute off, {tag}")
    tp._compare("nf4_lora", ref, on, mask, f"recompute on, {tag}")
    eq, eq_g, g_rel = _same(off, on)
    print(f"[recompute on vs off, {tag}] forward bitwise {eq} | grads bitwise {eq_g}"
          f" | grads max rel {g_rel:.3e}")
    assert eq  # same forward computation
    if hook:
        assert eq_g  # recompute == forward values: same backward bit for bit
    else:
        assert g_rel < 2e-2
    n_layers = model.config.num_hidden_layers
    # NORM_SAVE_OUTPUT: the MLP keeps the norm's saved y, no norm recompute in backward.
    recompute = 0 if packed.NORM_SAVE_OUTPUT else n_layers
    assert counts["_add_rmsnorm_fwd"] == 2 * (2 * n_layers + 1) + recompute, counts


def test_recompute_dequant_multi(monkeypatch):
    """nf4's dequant_multi present (a stand-in): the recompute backward dequantizes gate and up
    in one call and forms dx = dg @ W_gate + du @ W_up as mm + addmm (one fp16 rounding instead
    of two fp16 partials summed in fp32 by the norm): forward bit-identical, grads within fp16
    noise of the separate path and of the original."""
    from solution.body import packed

    _patch_all()
    monkeypatch.setattr(packed, "_NF4_LINEAR", [_dequant_linear4bit])
    model = tp._build("nf4_lora")
    ids, mask, labels = tp._batch()
    with tp._original_forward():
        ref = tp._run(model, ids, mask, labels)
    monkeypatch.setattr(packed, "_NF4_DEQUANT_MULTI", [None])
    with _recompute(True):
        sep = tp._run(model, ids, mask, labels)
    calls = []
    monkeypatch.setattr(packed, "_NF4_DEQUANT_MULTI", [tp._fake_dequant_multi(calls)])
    with _recompute(True):
        new = tp._run(model, ids, mask, labels)
    assert calls == [2] * model.config.num_hidden_layers, calls
    eq, _, g_rel = _same(sep, new)
    print(f"[dequant_multi vs separate] forward bitwise {eq} | grads max rel {g_rel:.3e}")
    assert eq and g_rel < 1e-2
    tp._compare("nf4_lora", ref, new, mask, "recompute + dequant_multi")


def _saved(model, ids, mask):
    """Saved tensors of one forward: list of (dtype, shape, storage ptr)."""
    saved = []

    def pack(x):
        saved.append((x.dtype, tuple(x.shape), x.untyped_storage().data_ptr()))
        return x

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x), \
            torch.autocast("cuda", dtype=torch.float16):
        out = model(input_ids=ids, attention_mask=mask, use_cache=False)
    del out
    return saved


def test_recompute_saves_no_intermediate():
    _patch_all()
    model = tp._build("nf4_lora")
    ids, mask, _ = tp._batch()
    T = int(mask.sum())
    with _recompute(False):
        off = _saved(model, ids, mask)
    with _recompute(True):
        on = _saved(model, ids, mask)
    n_layers = model.config.num_hidden_layers
    big_off = [s for s in off if s[1][-1:] == (INTER,)]
    big_on = [s for s in on if s[1][-1:] == (INTER,)]
    assert len(big_off) == 2 * n_layers, big_off  # the example SwiGLU's gate and up
    assert big_on == [], big_on
    from collections import Counter

    from solution.body import packed

    key = lambda s: (s[0], s[1])  # noqa: E731
    extra = Counter(map(key, on)) - Counter(map(key, off))
    missing = Counter(map(key, off)) - Counter(map(key, on))
    assert missing == Counter({(torch.float16, (1, T, INTER)): 2 * n_layers}), missing
    if packed.NORM_SAVE_OUTPUT:
        # Besides that, the recompute saves only its input x (fp16 [T, H]): the norm output
        # that the norm Function saves too (same storage, no memory); no residual.
        assert set(extra) == {(torch.float16, (T, 256))}, extra
        ptrs = lambda saved: {s[2] for s in saved if s[:2] == (torch.float16, (T, 256))}  # noqa
        assert len(ptrs(on)) == len(ptrs(off))  # no new storage: the norms' own y
        # fp32 [T, H]: only layer 0's input norm (no delta: r_out is the embedding output) and
        # the final norm (fp32 output) keep saving their residual, not the decoder-layer norms;
        # the third is the final norm's output, which the scatter back to [B, L, H]
        # (index_copy_, IndexCopyBackward0) saves.
        res = {s[2] for s in on if s[0] == torch.float32 and s[1][-1:] == (256,) and len(s[1]) > 1}
        assert len(res) == 3, res
    else:
        # Besides that, the recompute saves only the norm's fp32 residual [1, T, H] (the storage
        # the norm Function saves as r_out [T, H] in the same forward: no memory) and the fp32
        # norm weight (a parameter); in particular no fp16 x [T, H].
        assert set(extra) <= {(torch.float32, (1, T, 256)), (torch.float32, (256,))}, extra
        assert extra[(torch.float32, (1, T, 256))] == n_layers, extra
        r_out_ptrs = {s[2] for s in on if s[:2] == (torch.float32, (T, 256))}
        assert all(s[2] in r_out_ptrs for s in on if s[:2] == (torch.float32, (1, T, 256)))
    print(f"\n[saved] off: {len(off)} tensors, {len(big_off)} [T, {INTER}];"
          f" on: {len(on)} tensors, 0 [T, {INTER}]")


@pytest.mark.parametrize("multi", [False, True], ids=["separate", "dequant_multi"])
def test_recompute_saved_x_path(multi, monkeypatch):
    """x not from our norm kernel (eps None): x itself is saved, grads still match mlp(x)
    (bit for bit with separate gate/up dequants; with dequant_multi dx is mm + addmm, one
    rounding instead of autograd's fp16 add of two rounded partials: fp16 noise)."""
    from solution.body import packed

    _patch_all()
    monkeypatch.setattr(packed, "_NF4_DEQUANT_MULTI",
                        [tp._fake_dequant_multi() if multi else None])
    model = tp._build("nf4_lora")
    mlp = model.get_base_model().model.layers[0].mlp
    torch.manual_seed(3)
    x = torch.randn(1, 2000, 256, device="cuda", dtype=torch.float16)  # M > bnb custom range
    w = torch.randn(1, 2000, 256, device="cuda")
    res = []
    for on in (False, True):
        xr = x.clone().requires_grad_(True)
        with _recompute(on), torch.autocast("cuda", dtype=torch.float16):
            y = packed._mlp(mlp, xr, None, None, None)
        (y.float() * w).sum().backward()
        res.append((y.detach(), xr.grad))
    assert res[0][0].dtype == res[1][0].dtype == torch.float16
    assert res[1][1].dtype == torch.float16
    assert torch.equal(res[0][0], res[1][0])
    if multi:
        assert tp._rel(res[1][1].float(), res[0][1].float()) < 1e-2
    else:
        assert torch.equal(res[0][1], res[1][1])


def test_recompute_guards():
    """RECOMPUTE_MLP off, no grad mode, fp32 x, no autocast, a non-silu config or nn.Linear
    projections: the module's own forward."""
    from solution.body import packed

    _patch_all()
    model = tp._build("nf4_lora")
    mlp = model.get_base_model().model.layers[0].mlp
    x = torch.randn(1, 8, 256, device="cuda", dtype=torch.float16)
    plain = tp._build("plain").model.layers[0].mlp  # nn.Linear projections
    with _recompute(False), torch.autocast("cuda", dtype=torch.float16):
        assert packed._mlp_layers(mlp, x) is None
    with _recompute(True):
        with torch.autocast("cuda", dtype=torch.float16):
            assert packed._mlp_layers(mlp, x) is not None
            with torch.no_grad():
                assert packed._mlp_layers(mlp, x) is None
            assert packed._mlp_layers(mlp, x.float()) is None
            old = mlp.config.hidden_act
            mlp.config.hidden_act = "gelu"
            try:
                assert packed._mlp_layers(mlp, x) is None
            finally:
                mlp.config.hidden_act = old
            assert packed._mlp_layers(plain, x) is None
        assert packed._mlp_layers(mlp, x) is None  # no autocast
