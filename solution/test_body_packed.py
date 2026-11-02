"""Pad removal (solution/body/packed.py) against the original HF Qwen2 forward.

Tiny random Qwen2 (2 layers, hidden 256, 4 q heads, 2 kv heads, head_dim 64), fp32 weights on
CUDA under fp16 autocast like the real run; variants: plain, peft LoRA on q_proj/v_proj
(dropout 0 so it is deterministic), and NF4 Linear4bit base layers + LoRA (with input grads
enabled, as prepare_model_for_kbit_training does). Right- and left-padded
batches of 4 rows with different lengths. Compared: hidden states at real positions, the loss
(labels -100 at pads), entropy/accuracy-style masked metrics, and the trainable grads.
Both packed variants are checked: with our Triton kernels (add_rmsnorm, rope; the default) and
with the plain torch ops (packed.USE_KERNELS = False). Launches of our kernels are counted by
wrapping triton's JITFunction.run, so the kernel path is shown to have actually run (and the
fallbacks to not run them). Run with `-s` to see the error table.
"""

import contextlib

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

LENS = [37, 20, 9, 30]
L = 40
VOCAB = 1000


def _config():
    from transformers import Qwen2Config

    cfg = Qwen2Config(vocab_size=VOCAB, hidden_size=256, intermediate_size=512,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=64, max_position_embeddings=1024, rms_norm_eps=1e-6,
                      rope_theta=1e6, tie_word_embeddings=True, use_cache=False)
    cfg._attn_implementation = "sdpa"
    return cfg


def _to_nf4(model):
    """Replace every decoder linear with a bitsandbytes NF4 Linear4bit (compute fp16)."""
    import bitsandbytes as bnb

    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name, lin in list(parent.named_children()):
                if not isinstance(lin, torch.nn.Linear):
                    continue
                new = bnb.nn.Linear4bit(lin.in_features, lin.out_features,
                                        bias=lin.bias is not None, compute_dtype=torch.float16,
                                        quant_type="nf4", compress_statistics=True)
                new.weight = bnb.nn.Params4bit(lin.weight.data.half().cpu(), requires_grad=False,
                                               quant_type="nf4", compress_statistics=True)
                if lin.bias is not None:
                    new.bias = torch.nn.Parameter(lin.bias.data.float().cpu(),
                                                  requires_grad=False)
                setattr(parent, name, new.cuda())
    model.is_loaded_in_4bit = True
    return model


def _build(variant, lora_dropout=0.0):
    from transformers import Qwen2ForCausalLM

    torch.manual_seed(0)
    model = Qwen2ForCausalLM(_config())
    # Non-trivial norm weights and biases so the test sees them.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.add_(0.1 * torch.randn_like(p))
            elif name.endswith("bias"):
                p.normal_(0, 0.1)
    model = model.cuda().float()
    if variant == "nf4_lora":
        model = _to_nf4(model)
    if variant in ("lora", "nf4_lora"):
        from peft import LoraConfig, get_peft_model

        for p in model.parameters():
            p.requires_grad_(False)
        if variant == "nf4_lora":
            # As prepare_model_for_kbit_training does in the run: the embedding output requires
            # grad, so the first norm's backward runs too.
            model.enable_input_require_grads()
        # peft probes its torchao dispatcher for every target that its bnb dispatcher does not
        # take (here: the nn.Linear targets of "lora"); with an old torchao (Kaggle has 0.10)
        # peft's is_torchao_available() raises ImportError. The run's targets are Linear4bit
        # (taken by the bnb dispatcher first), so only the test needs the probe off.
        import peft.tuners.lora.torchao as peft_torchao

        probe = peft_torchao.is_torchao_available
        peft_torchao.is_torchao_available = lambda: False
        try:
            model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=lora_dropout,
                                                     target_modules=["q_proj", "v_proj"],
                                                     task_type="CAUSAL_LM"))
        finally:
            peft_torchao.is_torchao_available = probe
        with torch.no_grad():  # B starts at zero: make the LoRA path visible
            for name, p in model.named_parameters():
                if "lora_B" in name:
                    p.normal_(0, 0.05)
    model.train()
    return model


def _batch(left=False):
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, VOCAB, (len(LENS), L), generator=g)
    mask = torch.zeros(len(LENS), L, dtype=torch.long)
    for b, n in enumerate(LENS):
        if left:
            mask[b, L - n:] = 1
        else:
            mask[b, :n] = 1
    ids[mask == 0] = 0
    labels = ids.clone()
    labels[mask == 0] = -100
    labels[:, :3] = -100  # a "prompt" part
    if left:
        # The first real token of a left-padded row would be predicted from a pad position
        # (a pad-dependent target in the base too); TRL pads right, so drop that target.
        for b, n in enumerate(LENS):
            labels[b, L - n] = -100
    return ids.cuda(), mask.cuda(), labels.cuda()


@contextlib.contextmanager
def _original_forward():
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2Model
    patched = cls.forward
    cls.forward = patched.__wrapped__
    try:
        yield
    finally:
        cls.forward = patched


@contextlib.contextmanager
def _kernels(on):
    from solution.body import packed

    old = packed.USE_KERNELS
    packed.USE_KERNELS = on
    try:
        yield
    finally:
        packed.USE_KERNELS = old


OUR_KERNELS = ("_add_rmsnorm_fwd", "_add_rmsnorm_bwd", "_rope_kernel",
               "lora_dropout_fwd_kernel", "lora_dropout_bwd_kernel")


@contextlib.contextmanager
def _count_launches():
    """Counts launches of our body kernels by name (wraps triton JITFunction.run), and under
    "lora_forward" the forwards of our LoRA path (calls of lora.cast_lora_weights, once per
    LoRAFunction / LoRADeltaFunction forward)."""
    from triton.runtime.jit import JITFunction

    from solution.body import lora

    counts = dict.fromkeys(OUR_KERNELS + ("lora_forward",), 0)
    run = JITFunction.run
    cast = lora.cast_lora_weights

    def counting_cast(*args, **kwargs):
        counts["lora_forward"] += 1
        return cast(*args, **kwargs)

    def counting_run(self, *args, **kwargs):
        name = self.fn.__name__
        if name in counts:
            counts[name] += 1
        return run(self, *args, **kwargs)

    JITFunction.run = counting_run
    lora.cast_lora_weights = counting_cast
    try:
        yield counts
    finally:
        JITFunction.run = run
        lora.cast_lora_weights = cast


def _run(model, ids, mask, labels, autocast=True):
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
        out = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False,
                    output_hidden_states=False)
        hidden = model.get_base_model().model(input_ids=ids, attention_mask=mask,
                                              use_cache=False).last_hidden_state \
            if hasattr(model, "get_base_model") else \
            model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    out.loss.backward()
    logits = out.logits.detach().float()
    shift_labels = labels[:, 1:]
    m = shift_labels != -100
    acc = ((logits[:, :-1].argmax(-1) == shift_labels) & m).sum() / m.sum()
    logp = torch.log_softmax(logits, -1)
    ent = -(logp.exp() * logp).sum(-1)
    ent = (ent * mask).sum() / mask.sum()
    grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters()
             if p.requires_grad and p.grad is not None}
    return {"hidden": hidden.detach().float(), "loss": out.loss.detach().float(),
            "acc": acc, "ent": ent, "grads": grads}


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp(min=1e-12)).item()


def _compare(variant, ref, new, mask, tag, tol_h=5e-3, tol_loss=1e-3, tol_g=2e-2):
    real = mask.bool()
    h_abs = (new["hidden"][real] - ref["hidden"][real]).abs().max().item()
    h_rel = _rel(new["hidden"][real], ref["hidden"][real])
    loss_rel = abs((new["loss"] - ref["loss"]).item()) / abs(ref["loss"].item())
    assert set(new["grads"]) == set(ref["grads"]) and new["grads"]
    g_rel = max(_rel(new["grads"][n], ref["grads"][n]) for n in ref["grads"])
    print(f"\n[{variant} {tag}] hidden max abs {h_abs:.3e} rel {h_rel:.3e}"
          f" | loss {ref['loss'].item():.6f} vs {new['loss'].item():.6f} rel {loss_rel:.2e}"
          f" | acc {ref['acc'].item():.4f} vs {new['acc'].item():.4f}"
          f" | entropy rel {abs(new['ent'] - ref['ent']).item() / ref['ent'].item():.2e}"
          f" | grads ({len(ref['grads'])}) max rel {g_rel:.3e}")

    assert torch.all(new["hidden"][~real] == 0)
    assert h_rel < tol_h
    assert loss_rel < tol_loss
    assert g_rel < tol_g


@pytest.mark.parametrize("kernels", [True, False], ids=["kernels", "torch"])
@pytest.mark.parametrize("left", [False, True], ids=["right_pad", "left_pad"])
@pytest.mark.parametrize("variant", ["plain", "lora", "nf4_lora"])
def test_packed_matches_original(variant, left, kernels):
    from solution.body import patch

    patch()
    model = _build(variant)
    ids, mask, labels = _batch(left)

    with _original_forward():
        ref = _run(model, ids, mask, labels)
    with _kernels(kernels), _count_launches() as counts:
        new = _run(model, ids, mask, labels)
    _compare(variant, ref, new, mask,
             f"{'left' if left else 'right'} {'kernels' if kernels else 'torch'}")

    n_layers = model.config.num_hidden_layers
    if kernels:
        # _run does a loss forward + backward and one more forward for the hidden states:
        # 2 norms per layer + the final norm, one rope per layer, backward once each. "plain"
        # trains the norm weights: the norm kernel gives no weight grad, so HF's norm is used
        # (guard). Without input grads (lora) the first norm needs no backward.
        norms = 0 if variant == "plain" else 2 * n_layers + 1
        norm_bwd = norms - (1 if variant == "lora" else 0)
        # The recompute MLP (NF4 MLPs only) recomputes the post-attention norm output in
        # backward with the norm forward kernel: one more launch per layer.
        from solution.body import packed

        recompute = n_layers if (variant == "nf4_lora" and packed.RECOMPUTE_MLP
                                 and not packed.NORM_SAVE_OUTPUT) else 0
        assert counts["_add_rmsnorm_fwd"] == 2 * norms + recompute, counts
        assert counts["_add_rmsnorm_bwd"] == norm_bwd, counts
        assert counts["_rope_kernel"] == 2 * n_layers + n_layers, counts
        # Our LoRA path runs for the peft Linear4bit wrappers (nf4_lora): one LoRA forward per
        # q/v module per forward; dropout 0 -> no dropout kernels. nn.Linear targets (lora)
        # stay on peft's forward.
        fwds = 2 * n_layers * 2 if variant == "nf4_lora" else 0
        assert counts["lora_forward"] == fwds, counts
        assert counts["lora_dropout_fwd_kernel"] == counts["lora_dropout_bwd_kernel"] == 0, counts
    else:
        assert not any(counts.values()), counts


def test_packed_kernels_pure_fp32():
    """No autocast, fp32 model: the kernels run with fp32 outputs (out dtype follows HF)."""
    from solution.body import patch

    patch()
    model = _build("lora")
    ids, mask, labels = _batch()
    with _original_forward():
        ref = _run(model, ids, mask, labels, autocast=False)
    with _count_launches() as counts:
        new = _run(model, ids, mask, labels, autocast=False)
    # fp32 everywhere: only summation order differs (kernel reductions, SDPA per row).
    _compare("lora", ref, new, mask, "pure fp32", tol_h=1e-4, tol_loss=1e-5, tol_g=1e-3)
    assert counts["_add_rmsnorm_fwd"] > 0 and counts["_rope_kernel"] > 0, counts


def test_packed_fp16_residual_falls_back():
    """A pure fp16 model (fp16 residual stream) keeps HF's norm rounding: the norm kernel is
    not used there (guard), rope still is; outputs match the original."""
    from solution.body import patch

    patch()
    model = _build("lora").half()
    ids, mask, labels = _batch()
    with _original_forward():
        ref = _run(model, ids, mask, labels, autocast=False)
    with _count_launches() as counts:
        new = _run(model, ids, mask, labels, autocast=False)
    _compare("lora", ref, new, mask, "pure fp16")
    assert counts["_add_rmsnorm_fwd"] == 0 and counts["_add_rmsnorm_bwd"] == 0, counts
    assert counts["_rope_kernel"] > 0, counts


def test_fallbacks_match_original():
    """No mask, a row without tokens and output_hidden_states go to the original forward."""
    from solution.body import patch

    patch()
    model = _build("plain")
    ids, mask, _ = _batch()
    base = model.model
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        with _original_forward():
            ref = base(input_ids=ids, use_cache=False).last_hidden_state
        new = base(input_ids=ids, use_cache=False).last_hidden_state
        assert torch.equal(ref, new)

        empty = mask.clone()
        empty[2] = 0
        with _original_forward():
            ref = base(input_ids=ids, attention_mask=empty, use_cache=False).last_hidden_state
        new = base(input_ids=ids, attention_mask=empty, use_cache=False).last_hidden_state
        assert torch.equal(ref, new)

        out = base(input_ids=ids, attention_mask=mask, use_cache=False,
                   output_hidden_states=True)
        assert out.hidden_states is not None and len(out.hidden_states) == 3

        tup = base(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)
        assert isinstance(tup, tuple) and tup[0].shape == (len(LENS), L, 256)


def _lora_dropouts(model):
    return [m for n, m in model.named_modules()
            if n.endswith("lora_dropout.default") and isinstance(m, torch.nn.Dropout)]


def test_nf4_lora_dropout_runs_deterministic():
    """LoRA dropout 0.05 (the task's value) on the NF4 + LoRA model: our dropout kernels run
    (fwd per q/v module per forward, bwd once each), loss and LoRA grads are finite and nonzero,
    the same torch seed gives identical results, another seed gives other masks, and with the
    dropout modules in eval mode (no dropout, as peft) the result matches the original."""
    from solution.body import patch

    patch()
    model = _build("nf4_lora", lora_dropout=0.05)
    drops = _lora_dropouts(model)
    assert len(drops) == 2 * model.config.num_hidden_layers and all(d.p == 0.05 for d in drops)
    ids, mask, labels = _batch()
    runs = []
    for seed in (123, 123, 7):
        torch.manual_seed(seed)
        with _count_launches() as counts:
            runs.append(_run(model, ids, mask, labels))
        n_layers = model.config.num_hidden_layers
        assert counts["lora_dropout_fwd_kernel"] == 2 * n_layers * 2, counts
        assert counts["lora_dropout_bwd_kernel"] == 2 * n_layers, counts
        assert counts["lora_forward"] == 2 * n_layers * 2, counts
    a, b, c = runs
    assert torch.isfinite(a["loss"])
    lora_grads = {n: g for n, g in a["grads"].items() if "lora_" in n}
    assert len(lora_grads) == 4 * model.config.num_hidden_layers
    for n, g in lora_grads.items():
        assert torch.isfinite(g).all() and g.abs().max() > 0, n
    assert torch.equal(a["loss"], b["loss"]) and torch.equal(a["hidden"], b["hidden"])
    assert all(torch.equal(a["grads"][n], b["grads"][n]) for n in a["grads"])
    assert not torch.equal(a["hidden"], c["hidden"])  # other seed: other masks

    for d in drops:
        d.eval()
    with _original_forward():
        ref = _run(model, ids, mask, labels)
    with _count_launches() as counts:
        new = _run(model, ids, mask, labels)
    _compare("nf4_lora", ref, new, mask, "dropout modules in eval")
    assert counts["lora_dropout_fwd_kernel"] == counts["lora_dropout_bwd_kernel"] == 0, counts
    assert counts["lora_forward"] > 0, counts


def test_nf4_linear4bit_hook(monkeypatch):
    """When the nf4 workstream's `linear4bit` is importable, every NF4 q/k/v/o base projection
    goes through it as linear4bit(x16 [T, in], base_layer, keep_fp16=True); here a stand-in
    that calls the bnb layer and records the calls. Results match the original."""
    from solution.body import packed, patch

    patch()
    calls = []

    def fake_linear4bit(x, module, keep_fp16=False):
        assert keep_fp16 and x.dtype == torch.float16 and x.dim() == 2
        calls.append(type(module).__name__)
        return module(x)

    monkeypatch.setattr(packed, "_NF4_LINEAR", [fake_linear4bit])
    monkeypatch.setattr(packed, "_NF4_MULTI", [None])  # separate q/k/v (nf4 may be present)
    model = _build("nf4_lora")
    ids, mask, labels = _batch()
    with _original_forward():
        ref = _run(model, ids, mask, labels)
    new = _run(model, ids, mask, labels)
    _compare("nf4_lora", ref, new, mask, "linear4bit hook")
    # q, k, v, o per layer (+ gate, up, down in the recompute MLP, which calls the base
    # projections the same way), two forwards.
    per_layer = 7 if packed.RECOMPUTE_MLP else 4
    assert calls and len(calls) == per_layer * model.config.num_hidden_layers * 2, len(calls)
    assert set(calls) == {"Linear4bit"}


def _fake_multi(calls=None):
    """Stand-in for nf4's linear4bit_multi(x, modules): one F.linear of x on the cat of the
    dequantized weights (+ the cat of the fp16 biases), split into column views -- the same
    autograd shape as nf4's (one Function on x, split views out)."""
    from solution.body import packed

    def multi(x, modules):
        assert x.dtype == torch.float16 and x.dim() == 2 and len(modules) == 3
        if calls is not None:
            calls.append(tuple(type(m).__name__ for m in modules))
        w = torch.cat([packed._dequant(m) for m in modules])
        bias = None if modules[0].bias is None else \
            torch.cat([m.bias.to(torch.float16) for m in modules])
        y = F.linear(x, w, bias)
        return y.split([m.weight.quant_state.shape[0] for m in modules], -1)

    return multi


def _fake_dequant_multi(calls=None):
    from solution.body import packed

    def dequant_multi(modules):
        if calls is not None:
            calls.append(len(modules))
        return torch.cat([packed._dequant(m) for m in modules])

    return dequant_multi


@pytest.mark.parametrize("left", [False, True], ids=["right_pad", "left_pad"])
def test_nf4_qkv_multi_hook(left, monkeypatch):
    """When nf4's `linear4bit_multi` is importable, q/k/v base projections run as ONE call on
    the shared normed x (here a stand-in); q's LoRA delta still goes to RoPE and v's LoRA is
    added (out of place: v is a view of Y_cat). Results match the original and the separate
    path; the norm backward still sums the q/v LoRA grads and the one base grad itself."""
    from solution.body import packed, patch

    patch()
    calls = []
    model = _build("nf4_lora")
    ids, mask, labels = _batch(left)
    with _original_forward():
        ref = _run(model, ids, mask, labels)
    monkeypatch.setattr(packed, "_NF4_MULTI", [None])
    sep = _run(model, ids, mask, labels)
    monkeypatch.setattr(packed, "_NF4_MULTI", [_fake_multi(calls)])
    with _count_launches() as counts:
        new = _run(model, ids, mask, labels)
    _compare("nf4_lora", ref, new, mask, "qkv multi vs original")
    _compare("nf4_lora", sep, new, mask, "qkv multi vs separate")
    n_layers = model.config.num_hidden_layers
    assert calls == [("Linear4bit",) * 3] * (n_layers * 2), calls  # two forwards per _run
    assert counts["lora_forward"] == 2 * n_layers * 2, counts
    assert counts["_rope_kernel"] == 3 * n_layers, counts  # two forwards + one backward
    assert counts["_add_rmsnorm_bwd"] > 0, counts


def test_nf4_multi_absent(monkeypatch):
    """No solution.nf4 (ImportError): both multi hooks resolve to None and the separate path
    runs unchanged."""
    import sys

    from solution.body import packed, patch

    monkeypatch.setitem(sys.modules, "solution.nf4", None)
    monkeypatch.setattr(packed, "_NF4_MULTI", [])
    monkeypatch.setattr(packed, "_NF4_DEQUANT_MULTI", [])
    assert packed._linear4bit_multi() is None and packed._nf4_dequant_multi() is None
    patch()
    model = _build("nf4_lora")
    ids, mask, labels = _batch()
    with _original_forward():
        ref = _run(model, ids, mask, labels)
    new = _run(model, ids, mask, labels)
    _compare("nf4_lora", ref, new, mask, "no nf4 multi")


@pytest.mark.parametrize("multi", [False, True], ids=["separate", "multi"])
@pytest.mark.parametrize("recompute", [False, True], ids=["mlp", "recompute_mlp"])
def test_no_fp16_grad_adds(recompute, multi, monkeypatch):
    """NF4 + LoRA (dropout 0.05) with the example SwiGLU patch, as in the run: the grads of the
    normed x from its consumers (q/k/v base, q/v LoRA; gate/up) are summed inside the norm
    backward kernel, so one fwd+bwd launches no fp16 element-wise add kernel and no fp16 zero
    fill (pad-free path, squeeze views instead of select in RoPE / attention)."""
    from torch.profiler import ProfilerActivity, profile
    from transformers.models.qwen2 import modeling_qwen2

    from solution.body import packed, patch
    from solution.swiglu import patch_qwen2_mlp

    if not getattr(modeling_qwen2.Qwen2MLP, "_test_example_swiglu", False):
        patch_qwen2_mlp()
        modeling_qwen2.Qwen2MLP._test_example_swiglu = True
    patch()
    if multi:  # stand-ins for nf4's fused q/k/v and gate/up dequant
        monkeypatch.setattr(packed, "_NF4_MULTI", [_fake_multi()])
        monkeypatch.setattr(packed, "_NF4_DEQUANT_MULTI", [_fake_dequant_multi()])
    else:
        monkeypatch.setattr(packed, "_NF4_MULTI", [None])
        monkeypatch.setattr(packed, "_NF4_DEQUANT_MULTI", [None])
    model = _build("nf4_lora", lora_dropout=0.05)
    ids, mask, labels = _batch()
    old = packed.RECOMPUTE_MLP
    packed.RECOMPUTE_MLP = recompute
    try:
        _run(model, ids, mask, labels)  # warm (compiles)
        model.zero_grad(set_to_none=True)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)
            out.loss.backward()
            torch.cuda.synchronize()
    finally:
        packed.RECOMPUTE_MLP = old
    names = [e.name for e in prof.events() if e.device_type.name == "CUDA"]
    assert any("_add_rmsnorm_bwd" in n for n in names), names
    adds16 = [n for n in names if "CUDAFunctor_add<c10::Half>" in n]
    fills16 = [n for n in names if "FillFunctor<c10::Half>" in n]
    assert not adds16 and not fills16, (len(adds16), len(fills16))


def test_fanout_forward_without_example_module(monkeypatch):
    """solution/swiglu.py missing (ImportError): _fanout_forward still recognizes HF's own
    Qwen2MLP.forward and gives None (-> mlp(x)) for anything else, instead of raising."""
    import sys

    from transformers.models.qwen2 import modeling_qwen2

    from solution.body import packed

    mlp = _build("plain").model.layers[0].mlp
    x = torch.randn(1, 4, 256, device="cuda")
    monkeypatch.setitem(sys.modules, "solution.swiglu", None)  # import -> ImportError
    monkeypatch.delattr("solution.swiglu", raising=False)
    def hf_forward(self, x):  # stands for HF's own forward (whatever an earlier test patched)
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    hf_forward.__module__ = modeling_qwen2.Qwen2MLP.__module__
    hf_forward.__qualname__ = "Qwen2MLP.forward"
    monkeypatch.setattr(modeling_qwen2.Qwen2MLP, "forward", hf_forward)
    fan = packed._fanout_forward(mlp, x)
    assert fan is not None
    torch.testing.assert_close(fan(x, x), mlp(x))
    monkeypatch.setattr(modeling_qwen2.Qwen2MLP, "forward", lambda self, x: x)
    assert packed._fanout_forward(mlp, x) is None


def test_layer_sched_hooks(monkeypatch):
    """The head workstream's offload front-ends (solution/layer_sched.py, absent on this
    branch): a fake module in sys.modules is resolved by packed._layer_sched(); every decoder
    layer runs inside layer(i, n) and boundary(residual, delta, i) sees each layer's output
    (passed through unchanged). Without the module the hooks are no-ops; results match the
    original either way."""
    import sys
    import types

    import solution
    from solution.body import packed, patch

    patch()
    model = _build("nf4_lora")
    ids, mask, labels = _batch()
    with _original_forward():
        ref = _run(model, ids, mask, labels)

    # Absent: resolves to the no-op pair.
    monkeypatch.setattr(packed, "_LAYER_SCHED", [])
    monkeypatch.delitem(sys.modules, "solution.layer_sched", raising=False)
    if not hasattr(solution, "layer_sched"):
        assert packed._layer_sched() == (packed._no_offload_layer, packed._no_offload_boundary)
        _compare("nf4_lora", ref, _run(model, ids, mask, labels), mask, "no layer_sched")

    calls = []

    @contextlib.contextmanager
    def layer(i, n_layers):
        calls.append(("enter", i, n_layers))
        yield
        calls.append(("exit", i, n_layers))

    def boundary(residual, delta, i):
        assert isinstance(residual, torch.Tensor) and isinstance(delta, torch.Tensor)
        calls.append(("boundary", i))
        return residual, delta

    fake = types.ModuleType("solution.layer_sched")
    fake.layer, fake.boundary = layer, boundary
    monkeypatch.setitem(sys.modules, "solution.layer_sched", fake)
    monkeypatch.setattr(solution, "layer_sched", fake, raising=False)
    monkeypatch.setattr(packed, "_LAYER_SCHED", [])
    assert packed._layer_sched() == (layer, boundary)
    new = _run(model, ids, mask, labels)
    _compare("nf4_lora", ref, new, mask, "fake layer_sched")
    n = model.config.num_hidden_layers
    per_forward = [c for i in range(n)
                   for c in (("enter", i, n), ("exit", i, n), ("boundary", i))]
    assert calls == per_forward * 2, calls  # _run: loss forward + hidden-state forward


def _fallback_warnings(caplog):
    return [r.getMessage() for r in caplog.records if " not used (" in r.getMessage()]


def test_partial_fallback_warnings(monkeypatch, caplog):
    """A per-module guard that sends one decoder piece to the HF / torch op warns once per
    (piece, reason) in training, naming the first layer; the run's normal path (NF4 + LoRA
    under fp16 autocast) warns nothing; eval / no-grad forwards warn nothing."""
    import logging

    from solution.body import packed, patch

    patch()
    caplog.set_level(logging.WARNING, logger="solution.body.packed")
    monkeypatch.setattr(packed, "_warned", set())
    model = _build("nf4_lora")
    ids, mask, labels = _batch()
    _run(model, ids, mask, labels)
    assert _fallback_warnings(caplog) == []

    base = model.get_base_model().model
    layer = base.layers[1]
    hooks = [layer.post_attention_layernorm.register_forward_hook(lambda m, a, o: None),
             layer.mlp.register_forward_hook(lambda m, a, o: None),
             layer.self_attn.q_proj.register_forward_hook(lambda m, a, o: None)]
    try:
        _run(model, ids, mask, labels)
        msgs = _fallback_warnings(caplog)
        print("\n" + "\n".join(msgs))
        assert len(msgs) == 3, msgs
        assert all("first at layer 1" in m for m in msgs), msgs
        assert {m.split(" not used")[0].split(": ")[-1] for m in msgs} == \
            {"norm kernel", "recompute MLP", "LoRA path"}, msgs
        _run(model, ids, mask, labels)  # once per process: no repeat
        assert len(_fallback_warnings(caplog)) == 3
        monkeypatch.setattr(packed, "_warned", set())
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            base(input_ids=ids, attention_mask=mask, use_cache=False)
        assert len(_fallback_warnings(caplog)) == 3
        model.train()
    finally:
        for h in hooks:
            h.remove()

    # Embedding output not fp32 while the embedding weight is fp32.
    monkeypatch.setattr(packed, "_warned", set())
    h = base.embed_tokens.register_forward_hook(lambda m, a, o: o.half())
    try:
        _run(model, ids, mask, labels)
    finally:
        h.remove()
    assert any("fp32 residual stream not used (embedding output is torch.float16; first at "
               "layer embedding)" in m for m in _fallback_warnings(caplog)), caplog.text
