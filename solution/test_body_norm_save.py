"""packed.NORM_SAVE_OUTPUT (decoder-layer norms save their fp16 output, not the fp32 residual;
backward recovers xhat = y / w, exact for the tiny-|w| channels) in the packed decoder.

The tiny NF4 + LoRA Qwen2 of test_body_packed with the example SwiGLU patch, as in the run,
with norm weights that include zeros and tiny values (exact channels, K > 0) or not (K = 0).
For every combination of NORM_SAVE_OUTPUT and RECOMPUTE_MLP: outputs, loss and LoRA grads vs
the original HF forward; the forward is the same computation with the flag on or off (bitwise),
the LoRA grads differ by fp16 noise. Saved tensors per decoder layer (saved_tensors_hooks,
deduplicated by storage): no fp32 [T, H] residual with the flag on.
"""

import contextlib

import pytest
import torch

from solution import test_body_packed as tp
from solution import test_body_recompute as tr

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

HID = 256


@contextlib.contextmanager
def _flags(save, recompute):
    from solution.body import packed

    old = packed.NORM_SAVE_OUTPUT, packed.RECOMPUTE_MLP
    packed.NORM_SAVE_OUTPUT, packed.RECOMPUTE_MLP = save, recompute
    try:
        yield
    finally:
        packed.NORM_SAVE_OUTPUT, packed.RECOMPUTE_MLP = old


def _model(weights, dropout=0.0):
    tr._patch_all()
    model = tp._build("nf4_lora", lora_dropout=dropout)
    if weights == "zeros_tiny":
        with torch.no_grad():
            for i, layer in enumerate(model.get_base_model().model.layers):
                for norm in (layer.input_layernorm, layer.post_attention_layernorm):
                    w = norm.weight
                    w[[0, 17 + i]] = 0.0
                    w[[5, 99]] = torch.tensor([1e-5, -3e-4], device=w.device)
                    w[200] = 2e-3  # above the threshold: recovered from y
    return model


@pytest.mark.parametrize("weights", ["regular", "zeros_tiny"])
@pytest.mark.parametrize("recompute", [False, True], ids=["mlp", "recompute_mlp"])
def test_norm_save_output_matches(recompute, weights):
    from solution.body import norm_rope

    model = _model(weights)
    ids, mask, labels = tp._batch()
    with tp._original_forward():
        ref = tp._run(model, ids, mask, labels)
    with _flags(False, recompute):
        off = tp._run(model, ids, mask, labels)
    with _flags(True, recompute), tp._count_launches() as counts:
        on = tp._run(model, ids, mask, labels)
    tag = f"save_output, {'recompute' if recompute else 'mlp'}, {weights}"
    tp._compare("nf4_lora", ref, off, mask, f"off, {tag}")
    tp._compare("nf4_lora", ref, on, mask, f"on, {tag}")
    assert torch.equal(on["hidden"], off["hidden"]) and torch.equal(on["loss"], off["loss"])
    g_rel = max(tp._rel(on["grads"][n], off["grads"][n]) for n in off["grads"])
    ks = [norm_rope.special_channels(n.weight)[1] for layer in
          model.get_base_model().model.layers
          for n in (layer.input_layernorm, layer.post_attention_layernorm)]
    print(f"[{tag}] K per norm {ks} | LoRA grads on vs off max rel {g_rel:.3e}")
    assert g_rel < 1e-2
    assert (sum(ks) > 0) == (weights == "zeros_tiny")
    n_layers = model.config.num_hidden_layers
    # 2 forwards x (2 norms per layer + final); no norm recompute in the MLP backward.
    assert counts["_add_rmsnorm_fwd"] == 2 * (2 * n_layers + 1), counts
    assert counts["_add_rmsnorm_bwd"] == 2 * n_layers + 1, counts


def _saved_per_layer(model, ids, mask):
    """{layer: {storage ptr: (nbytes, dtype, shape)}} of one forward, parameters excluded."""
    from solution.body import packed

    params = {p.untyped_storage().data_ptr() for p in model.parameters()}
    per, cur = {}, [None]
    orig = packed._layer

    def wrap(*a, **k):
        cur[0] = len(per)
        per[cur[0]] = {}
        try:
            return orig(*a, **k)
        finally:
            cur[0] = None

    def pack(x):
        if cur[0] is not None and x.is_cuda:
            st = x.untyped_storage()
            if st.data_ptr() not in params:
                per[cur[0]][st.data_ptr()] = (st.nbytes(), x.dtype, tuple(x.shape))
        return x

    packed._layer = wrap
    try:
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x), \
                torch.autocast("cuda", dtype=torch.float16):
            out = model(input_ids=ids, attention_mask=mask, use_cache=False)
        del out
    finally:
        packed._layer = orig
    return per


@pytest.mark.parametrize("recompute", [False, True], ids=["mlp", "recompute_mlp"])
def test_norm_save_output_saved_tensors(recompute):
    """Per decoder layer: with the flag off two fp32 [T, H] residuals are saved;
    with it on none (layer 0 keeps the embedding output: its input norm has no delta), and the
    saved bytes drop by at least the two residuals minus the K exact columns."""
    model = _model("zeros_tiny")
    ids, mask, _ = tp._batch()
    T = int(mask.sum())
    res = {}
    for save in (False, True):
        with _flags(save, recompute):
            res[save] = _saved_per_layer(model, ids, mask)
    for save, per in res.items():
        for layer, st in per.items():
            fp32 = [v for v in st.values() if v[1] == torch.float32 and v[0] >= T * HID * 4]
            want = (1 if layer == 0 else 0) if save else 2
            assert len(fp32) == want, (save, layer, fp32)
    for layer in res[False]:
        b_off = sum(v[0] for v in res[False][layer].values())
        b_on = sum(v[0] for v in res[True][layer].values())
        print(f"[saved, {'recompute' if recompute else 'mlp'}] layer {layer}:"
              f" {b_off / 2**10:.1f} KiB -> {b_on / 2**10:.1f} KiB")
        if layer > 0:
            # Gone: two fp32 [T, H] residuals. New to the hooks: the post-attention y (fp16
            # [T, H]; without recompute bnb already holds it on ctx.tensors, invisible here, so
            # it costs nothing there; with recompute it is the MLP input instead of the
            # residual), xs [T, K] fp32 (K = 4 per norm) and the int32 [H] slot maps.
            # The exact byte count of the new small tensors depends on the environment (which
            # NF4 path holds y, the K of the weights), so require only a net saving of at least
            # one fp32 [T, H] residual; the residual count itself is asserted above.
            assert b_off - b_on >= T * HID * 4, (b_off, b_on)


def test_scatter_in_place_grads():
    """The scatter back to [B, L, H] (index_copy_ into fresh zeros): hidden grads reach the
    packed tokens exactly (embedding-input grads vs the original forward)."""
    model = _model("regular")
    ids, mask, _ = tp._batch()
    base = model.get_base_model().model
    # Pad outputs of the original forward depend on real tokens: weight them 0.
    w = torch.randn(*ids.shape, HID, device="cuda") * mask[..., None]
    grads = []
    for orig in (True, False):
        ctx = tp._original_forward() if orig else contextlib.nullcontext()
        emb = []
        hook = base.embed_tokens.register_forward_hook(lambda m, i, o: (o.retain_grad(),
                                                                       emb.append(o))[1])
        try:
            with ctx, torch.autocast("cuda", dtype=torch.float16):
                h = base(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            (h.float() * w).sum().backward()
        finally:
            hook.remove()
        g = emb[0].grad
        grads.append(g.reshape(-1, HID)[mask.reshape(-1).bool()] if orig else g.reshape(-1, HID))
        model.zero_grad(set_to_none=True)
    assert tp._rel(grads[1], grads[0]) < 2e-2
