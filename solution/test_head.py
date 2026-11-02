"""Fused LM head vs a plain PyTorch fp32 reference: loss, per-token stats and dX.

Run on a GPU: `python -m pytest solution/test_head.py -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _reference(x, w, target, ce_weight, rows_per_step=256):
    """fp32 logits = x.float() @ w.float().T, log_softmax in fp32, dX by autograd.

    Done in row slices so the [N, V] fp32 logits never exist at once (6 GB local GPU)."""
    w32 = w.float()
    v = w.shape[0]
    outs = {k: [] for k in ("lse", "tlogit", "ent", "maxlogit", "dx", "lse16", "tlogit16", "ent16")}
    loss = torch.zeros((), device=x.device, dtype=torch.float64)
    for r0 in range(0, x.shape[0], rows_per_step):
        xs = x[r0:r0 + rows_per_step].float().requires_grad_(True)
        ts = target[r0:r0 + rows_per_step]
        cs = ce_weight[r0:r0 + rows_per_step].float()
        logits = xs @ w32.T
        logp = torch.log_softmax(logits, dim=-1)
        valid = (ts >= 0) & (ts < v)
        safe_t = torch.where(valid, ts, torch.zeros_like(ts))
        nll = -logp.gather(1, safe_t[:, None])[:, 0]
        part = (torch.where(valid, cs, torch.zeros_like(cs)) * nll).sum()
        (dxs,) = torch.autograd.grad(part, xs)
        with torch.no_grad():
            lse = torch.logsumexp(logits, dim=-1)
            outs["lse"].append(lse)
            outs["tlogit"].append(torch.where(valid, logits.gather(1, safe_t[:, None])[:, 0],
                                              torch.zeros_like(lse)))
            outs["ent"].append(-(logp.exp() * logp).sum(-1))
            outs["maxlogit"].append(logits.max(-1).values)
            outs["dx"].append(dxs)
            # Same stats from fp16-rounded logits (cuBLAS fp16 out, as an fp16 lm_head gives).
            z16 = (x[r0:r0 + rows_per_step] @ w.T).float()
            lp16 = torch.log_softmax(z16, dim=-1)
            outs["lse16"].append(torch.logsumexp(z16, dim=-1))
            outs["tlogit16"].append(torch.where(valid, z16.gather(1, safe_t[:, None])[:, 0],
                                                torch.zeros_like(lse)))
            outs["ent16"].append(-(lp16.exp() * lp16).sum(-1))
            del z16, lp16
        loss += part.detach().double()
        del logits, logp
    res = {k: torch.cat(vs) for k, vs in outs.items()}
    res["loss"] = loss
    return res, w32


def _logit_at(x, w32, idx):
    return (x.float() * w32[idx]).sum(-1)


def _make(n, h, v, scale_w, peak, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(v, h, device="cuda", generator=g) * scale_w).half()
    target = torch.randint(0, v, (n,), device="cuda", generator=g)
    x = torch.randn(n, h, device="cuda", generator=g)
    if peak:
        # Push each row towards its target embedding with a random strength, so logits range
        # from flat to very peaked (target logit up to ~+30), as in a real LM head.
        wt = w[target].float()
        strength = torch.rand(n, 1, device="cuda", generator=g) * peak
        x = x + strength * wt / wt.pow(2).sum(-1, keepdim=True)
    x = x.half()
    ce_weight = torch.rand(n, device="cuda", generator=g) / n
    # some ignored rows and some zero-weight rows
    target[::7] = -100
    ce_weight[::7] = 0.0
    ce_weight[3::11] = 0.0
    return x, w, target, ce_weight


def _check_all(x, w, target, ce_weight, chunk_rows, dx_rtol):
    from solution.head import fused_linear_ce

    ref, w32 = _reference(x, w, target, ce_weight)
    xg = x.clone().requires_grad_(True)
    loss, pt = fused_linear_ce(xg, w, target, ce_weight, chunk_rows=chunk_rows)
    loss.backward()
    dx = xg.grad

    assert torch.isfinite(loss)
    # The logits are fp16 (cuBLAS fp16 out), so vs the fp32 reference the per-token errors are
    # bounded by fp16 rounding of |z| < 64 (half ulp <= 0.0156); vs the fp16-logit reference
    # they are fp32-tight.
    torch.testing.assert_close(loss.double(), ref["loss"], rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(pt["lse"], ref["lse"], rtol=0, atol=1.6e-2)
    torch.testing.assert_close(pt["target_logit"], ref["tlogit"], rtol=0, atol=1.6e-2)
    torch.testing.assert_close(pt["entropy"], ref["ent"], rtol=0, atol=3e-2)
    torch.testing.assert_close(pt["lse"], ref["lse16"], rtol=1e-6, atol=1e-4)
    torch.testing.assert_close(pt["target_logit"], ref["tlogit16"], rtol=0, atol=1e-6)
    torch.testing.assert_close(pt["entropy"], ref["ent16"], rtol=1e-4, atol=1e-4)
    # argmax: ties allowed, the logit at our index must equal the row max (up to fp16 rounding)
    ours_at = _logit_at(x, w32, pt["argmax"])
    torch.testing.assert_close(ours_at, ref["maxlogit"], rtol=0, atol=1.6e-2)
    assert pt["argmax"].dtype == torch.int64
    ref_ce = torch.where(target >= 0, ref["lse"] - ref["tlogit"], torch.zeros_like(ref["lse"]))
    torch.testing.assert_close(pt["ce"], ref_ce, rtol=0, atol=3e-2)
    ref_ce16 = torch.where(target >= 0, ref["lse16"] - ref["tlogit16"], torch.zeros_like(ref["lse"]))
    torch.testing.assert_close(pt["ce"], ref_ce16, rtol=1e-5, atol=1e-4)

    assert dx.dtype == x.dtype and dx.shape == x.shape
    ref_dx = ref["dx"]
    zero_rows = (ce_weight == 0) | (target < 0)
    assert dx[zero_rows].abs().max().item() == 0.0
    err = (dx.float() - ref_dx).norm() / ref_dx.norm()
    assert err.item() < dx_rtol, f"dx relative error {err.item():.3e}"
    scale = ref_dx.abs().max()
    assert ((dx.float() - ref_dx).abs().max() / scale).item() < 2 * dx_rtol
    return err.item()


@pytest.mark.parametrize("chunk_rows", [32, 256])
def test_small_odd_shapes(chunk_rows):
    x, w, target, ce_weight = _make(77, 64, 1001, scale_w=0.3, peak=20.0)
    _check_all(x, w, target, ce_weight, chunk_rows, dx_rtol=1e-2)


def test_realistic_lm_head():
    torch.cuda.empty_cache()
    x, w, target, ce_weight = _make(2048, 1536, 151936, scale_w=0.02, peak=30.0, seed=1)
    _check_all(x, w, target, ce_weight, chunk_rows=256, dx_rtol=1e-2)
    del x, w
    torch.cuda.empty_cache()


def test_no_grad_path_skips_dx():
    from solution.head import fused_linear_ce
    import solution.head.op as op

    x, w, target, ce_weight = _make(77, 64, 1001, scale_w=0.3, peak=20.0)
    ref, _ = _reference(x, w, target, ce_weight)
    calls = []  # WRITE_G of each row-kernel launch
    orig = op.head_row_softmax_kernel

    class Spy:
        def __getitem__(self, grid):
            def launch(*args, **kw):
                calls.append(kw["WRITE_G"])
                return orig[grid](*args, **kw)
            return launch

    op.head_row_softmax_kernel = Spy()
    try:
        xg = x.clone().requires_grad_(True)
        with torch.no_grad():
            loss, pt = fused_linear_ce(xg, w, target, ce_weight)
        assert calls == [False]
        assert not loss.requires_grad
        torch.testing.assert_close(loss.double(), ref["loss"], rtol=1e-3, atol=1e-4)
        torch.testing.assert_close(pt["lse"], ref["lse16"], rtol=1e-6, atol=1e-4)
        torch.testing.assert_close(pt["entropy"], ref["ent16"], rtol=1e-4, atol=1e-4)
        # with grad the dX kernel runs
        loss2, _ = fused_linear_ce(xg, w, target, ce_weight)
        assert calls == [False, True] and loss2.requires_grad
        torch.testing.assert_close(loss2.detach(), loss, rtol=1e-6, atol=1e-6)
    finally:
        op.head_row_softmax_kernel = orig


def test_fp32_input_and_half_ignored():
    """fp32 x is cast to fp16 inside, dX comes back in fp32; half the rows ignored (prompt)."""
    from solution.head import fused_linear_ce

    x, w, target, ce_weight = _make(300, 128, 5003, scale_w=0.2, peak=25.0, seed=3)
    target[:150] = -100  # contiguous ignored block, like prompt tokens
    ce_weight[:150] = 0.0
    ref, _ = _reference(x, w, target, ce_weight)
    xg = x.float().requires_grad_(True)
    loss, st = fused_linear_ce(xg, w, target, ce_weight, chunk_rows=128)
    loss.backward()
    assert xg.grad.dtype == torch.float32 and loss.dtype == torch.float32 and loss.dim() == 0
    torch.testing.assert_close(loss.double(), ref["loss"], rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(st["entropy"], ref["ent16"], rtol=1e-4, atol=1e-4)
    assert (st["target_logit"][:150] == 0).all() and (st["ce"][:150] == 0).all()
    err = (xg.grad - ref["dx"]).norm() / ref["dx"].norm()
    assert err.item() < 1e-2


def test_warmup_covers_training_launch():
    """warmup() compiles exactly the row kernel variant a real training forward launches."""
    import triton

    from solution.head import fused_linear_ce, warmup
    from solution.head.kernels import head_row_softmax_kernel
    from solution.head.op import VOCAB

    dev = torch.device("cuda", torch.cuda.current_device())
    # Detect compiles two ways: the per-device in-memory kernel cache (key -> compiled kernel,
    # filled by JITFunction._do_compile) and triton.knobs.runtime.jit_post_compile_hook.
    kernel_cache = head_row_softmax_kernel.device_caches[dev.index][0]
    saved = dict(kernel_cache)
    compiles = []
    old_hook = triton.knobs.runtime.jit_post_compile_hook

    def hook(*, key, fn, **kw):
        if fn.jit_function is head_row_softmax_kernel:
            compiles.append(key)
        return old_hook(key=key, fn=fn, **kw) if old_hook else None

    triton.knobs.runtime.jit_post_compile_hook = hook
    kernel_cache.clear()  # fresh in-memory cache (the on-disk cache keeps it cheap)
    try:
        warmup(dev)
        assert len(kernel_cache) == 1 and len(compiles) == 1, "warmup compiled nothing"
        compiles.clear()

        torch.manual_seed(0)
        n, h = 300, 64  # one full 256-row chunk + a partial one
        x = torch.randn(n, h, device=dev, dtype=torch.float16, requires_grad=True)
        w = (torch.randn(VOCAB, h, device=dev) * 0.05).half()
        target = torch.randint(0, VOCAB, (n,), device=dev)
        target[::7] = -100
        cew = torch.ones(n, device=dev)
        loss, _ = fused_linear_ce(x, w, target, cew)
        loss.backward()
        torch.cuda.synchronize()
        assert compiles == [], "real launch recompiled: warmup specialization differs"
        assert len(kernel_cache) == 1
    finally:
        triton.knobs.runtime.jit_post_compile_hook = old_hook
        kernel_cache.update(saved)
