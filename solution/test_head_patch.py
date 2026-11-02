"""Tests for solution/head/patch.py: the fused LM head wired into Qwen2ForCausalLM and TRL.

A tiny random Qwen2ForCausalLM (fp32 weights, tied and frozen embedding / lm_head, a few
trainable params) is run under fp16 autocast, with and without the patch, and the loss, grads
and the TRL / Soup metrics are compared. CUDA only.
"""

import copy
import importlib

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

VOCAB, HIDDEN = 1000, 64


def _tiny_model(seed=0, vocab=VOCAB):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=vocab, hidden_size=HIDDEN, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        tie_word_embeddings=True, attention_dropout=0.0, use_cache=False,
    )
    model = Qwen2ForCausalLM(cfg).cuda().float()
    for p in model.parameters():
        p.requires_grad_(False)
    for layer in model.model.layers:
        layer.self_attn.q_proj.weight.requires_grad_(True)
        layer.self_attn.v_proj.weight.requires_grad_(True)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    model.train()
    return model


def _batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    bsz, seq = 3, 24
    lengths = [24, 17, 9]
    prompts = [6, 5, 3]
    input_ids = torch.randint(0, VOCAB, (bsz, seq), generator=g)
    attention_mask = torch.zeros(bsz, seq, dtype=torch.long)
    labels = torch.full((bsz, seq), -100, dtype=torch.long)
    for b, (n, p) in enumerate(zip(lengths, prompts)):
        attention_mask[b, :n] = 1
        labels[b, p:n] = input_ids[b, p:n]
        input_ids[b, n:] = 0  # right padding
    return input_ids.cuda(), attention_mask.cuda(), labels.cuda()


def _run(model, input_ids, attention_mask, labels, num_items):
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", torch.float16):
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                    num_items_in_batch=num_items)
    out.loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    return out, grads


def _reference_stats(logits, labels, attention_mask):
    import torch.nn.functional as F

    logits = logits.detach().float()
    bsz, seq, vocab = logits.shape
    logps = F.log_softmax(logits, dim=-1)
    ent = -(logps.exp() * logps).sum(-1)
    entropy = (ent * attention_mask).sum() / attention_mask.sum()
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    pred = logits[:, :-1].argmax(-1)
    accuracy = ((pred == shift_labels) & mask).sum() / mask.sum()
    tl = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), shift_labels.reshape(-1),
                         ignore_index=-100, reduction="none").view(bsz, seq - 1)
    token_loss = torch.cat([torch.where(mask, tl, 0.0), tl.new_zeros(bsz, 1)], dim=1)
    return entropy, accuracy, token_loss, ent


@pytest.mark.parametrize("use_num_items", [True, False])
def test_patched_forward_matches_reference(use_num_items):
    from solution.head.patch import _MODEL_ATTR, arm_model, disarm_model

    model = _tiny_model()
    input_ids, attention_mask, labels = _batch()
    num_items = (labels[:, 1:] != -100).sum() if use_num_items else None

    ref_out, ref_grads = _run(model, input_ids, attention_mask, labels, num_items)
    assert ref_out.logits is not None

    assert arm_model(model)
    try:
        out, grads = _run(model, input_ids, attention_mask, labels, num_items)
        step = model.__dict__[_MODEL_ATTR].take()
    finally:
        disarm_model(model)

    assert out.logits is None
    assert step is not None
    torch.testing.assert_close(out.loss.float(), ref_out.loss.float(), rtol=1e-3, atol=1e-4)
    for name, g in ref_grads.items():
        torch.testing.assert_close(grads[name], g, rtol=2e-2, atol=1e-5 + 2e-2 * g.abs().max().item(),
                                   msg=lambda m, n=name: f"{n}: {m}")

    entropy, accuracy, token_loss, ent = _reference_stats(ref_out.logits, labels, attention_mask)
    got_entropy = (step.token_entropy * attention_mask).sum() / attention_mask.sum()
    torch.testing.assert_close(got_entropy, entropy, rtol=1e-3, atol=1e-4)
    m = attention_mask.bool()
    torch.testing.assert_close(step.token_entropy[m], ent[m], rtol=2e-3, atol=2e-3)
    got_acc = step.correct / step.labelled
    assert abs(got_acc.item() - accuracy.item()) <= 1.0 / step.labelled.item() + 1e-6
    assert step.labelled.item() == (labels[:, 1:] != -100).sum().item()
    assert step.token_loss.shape == labels.shape
    torch.testing.assert_close(step.token_loss, token_loss, rtol=2e-3, atol=2e-3)
    assert (step.token_loss[:, -1] == 0).all()

    # After disarming the original forward is back (logits returned).
    with torch.autocast("cuda", torch.float16):
        again = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert again.logits is not None


def test_untouched_without_labels_eval_or_autocast():
    from solution.head.patch import arm_model, disarm_model

    model = _tiny_model()
    input_ids, attention_mask, labels = _batch()
    assert arm_model(model)
    try:
        with torch.autocast("cuda", torch.float16):
            assert model(input_ids=input_ids, attention_mask=attention_mask).logits is not None
            model.eval()
            assert model(input_ids=input_ids, attention_mask=attention_mask,
                         labels=labels).logits is not None
            model.train()
        # no fp16 autocast: original path
        assert model(input_ids=input_ids, attention_mask=attention_mask,
                     labels=labels).logits is not None
    finally:
        disarm_model(model)


def test_arm_refuses_trainable_head():
    from solution.head.patch import arm_model

    model = _tiny_model()
    model.lm_head.weight.requires_grad_(True)
    assert not arm_model(model)


# --------------------------------------------------------------------------------------------
# Real TRL SFTTrainer + Soup's rewind recorder, 2 optimizer steps, patched vs unpatched.
# --------------------------------------------------------------------------------------------


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {f"t{i}": i for i in range(VOCAB)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="t1"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="t1", pad_token="t0",
                                   eos_token="t2")


def _dataset():
    from datasets import Dataset

    g = torch.Generator().manual_seed(3)
    rows = {"input_ids": [], "completion_mask": []}
    for i in range(8):
        n = int(torch.randint(8, 30, (1,), generator=g))
        p = int(torch.randint(2, n - 2, (1,), generator=g))
        ids = torch.randint(3, VOCAB, (n,), generator=g).tolist()
        rows["input_ids"].append(ids)
        rows["completion_mask"].append([0] * p + [1] * (n - p))
    return Dataset.from_dict(rows)


class _Sink:
    def __init__(self):
        self.records = []

    def record_batch(self, *, step, micro, rows, row_loss, row_tokens):
        self.records.append((step, micro, list(rows), list(row_loss), list(row_tokens)))


def _one_gpu(args):
    """Train on one GPU: with 2 visible GPUs (Kaggle) Trainer would wrap the model in
    nn.DataParallel, whose replica on cuda:1 is a different, unarmed module."""
    _ = args.n_gpu  # runs _setup_devices, which sets _n_gpu = device_count()
    args._n_gpu = min(args._n_gpu, 1)
    return args


def _train(tmp_path, armed):
    from soup_cli.trainer.rewind_hf import attach_rewind_state, make_rewind_trainer_class
    from trl import SFTConfig, SFTTrainer

    from solution.head.patch import arm_trainer, disarm_trainer

    model = _tiny_model()
    args = SFTConfig(
        output_dir=str(tmp_path / ("armed" if armed else "base")), max_steps=2,
        per_device_train_batch_size=2, gradient_accumulation_steps=2, learning_rate=1e-3,
        logging_steps=1, fp16=True, report_to=[], save_strategy="no", seed=0, data_seed=0,
        dataloader_num_workers=0, max_length=64, packing=False, padding_free=False,
    )
    _one_gpu(args)
    trainer = make_rewind_trainer_class(SFTTrainer)(
        model=model, args=args, train_dataset=_dataset(), processing_class=_tokenizer(),
    )
    sink = _Sink()
    state = attach_rewind_state(trainer, sink)
    if armed:
        assert arm_trainer(trainer)
    seen = []
    if armed:
        orig_forward = model.lm_head.forward

        def spy(*a, **k):
            seen.append(a[0].shape)
            return orig_forward(*a, **k)

        model.lm_head.forward = spy
    try:
        trainer.train()
    finally:
        if armed:
            disarm_trainer(trainer)
    assert not state.failed
    if armed:
        assert not seen, "lm_head ran over all positions on the armed path"
    logs = [h for h in trainer.state.log_history if "loss" in h]
    return logs, sink.records


def test_trainer_two_steps_patched_vs_unpatched(tmp_path):
    base_logs, base_rows = _train(tmp_path, armed=False)
    ours_logs, ours_rows = _train(tmp_path, armed=True)

    assert len(base_logs) == len(ours_logs) == 2
    for b, o in zip(base_logs, ours_logs):
        for key in ("loss", "entropy", "mean_token_accuracy", "num_tokens"):
            assert key in o, f"missing {key} in {o}"
        assert o["num_tokens"] == b["num_tokens"]
        assert o["loss"] == pytest.approx(b["loss"], rel=1e-3)
        assert o["entropy"] == pytest.approx(b["entropy"], rel=1e-3)
        assert o["mean_token_accuracy"] == pytest.approx(b["mean_token_accuracy"], abs=0.05)

    assert len(base_rows) == len(ours_rows) == 4
    for b, o in zip(base_rows, ours_rows):
        assert b[:3] == o[:3]
        assert b[4] == o[4]
        assert o[3] == pytest.approx(b[3], rel=2e-3)


# --------------------------------------------------------------------------------------------
# v2: one fp16 tied weight shared by the embedding lookup and the head.
# --------------------------------------------------------------------------------------------

# Rows of the tiny embedding that get values below fp16's normal range (they do not survive
# fp32 -> fp16 -> fp32), with up to 6 such elements per row, like the real checkpoint (<= 5).
_FIXUP_ROWS = (0, 3, 7, 100, 511, VOCAB - 1)


def _bf16_tiny_model(seed=0):
    """The tiny model with every weight bf16-representable (as the real checkpoint) and a few
    embedding elements that fp16 cannot hold exactly."""
    model = _tiny_model(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(p.to(torch.bfloat16).float())
        w = model.model.embed_tokens.weight
        g = torch.Generator().manual_seed(seed + 7)
        for i, row in enumerate(_FIXUP_ROWS):
            cols = torch.randperm(HIDDEN, generator=g)[: 1 + i % 6]
            vals = (torch.rand(cols.numel(), generator=g) * 2 - 1) * 3e-6
            w[row, cols.cuda()] = vals.to(torch.bfloat16).float().cuda()
    w = model.model.embed_tokens.weight
    assert (w.half().float().view(torch.int32) != w.view(torch.int32)).sum() > len(_FIXUP_ROWS)
    return model


def _ids_with_fixup_rows():
    input_ids, attention_mask, labels = _batch()
    input_ids[0, : len(_FIXUP_ROWS)] = torch.tensor(_FIXUP_ROWS, device=input_ids.device)
    input_ids[1, 2] = _FIXUP_ROWS[2]
    return input_ids, attention_mask, labels


def _bits(t):
    return t.contiguous().view(torch.int32)


def test_fixup_restores_exact_values():
    from solution.head.patch import _Fp16Fixup

    g = torch.Generator().manual_seed(5)
    w = torch.randn(300, 48, generator=g).to(torch.bfloat16).float()
    w[3, 5] = 1.9e-11
    w[3, 6] = -2.5e-7
    w[17, 0] = 1e5  # overflows fp16
    w[17, 47] = float("nan")
    w[200, 1] = -0.0
    w = w.cuda()
    w16 = w.half()
    fix = _Fp16Fixup(w, w16)
    assert fix.count == int((_bits(w16.float()) != _bits(w)).sum())
    assert fix.count >= 4 and fix.rows == 2
    assert torch.equal(_bits(fix.exact_weight(w16)), _bits(w))
    ids = torch.tensor([3, 17, 0, 3, 200, 299, 17], device="cuda")
    out = fix.apply_rows(w16[ids].float(), ids)
    assert torch.equal(_bits(out), _bits(w[ids]))


def test_shared_fp16_weight_embedding_exact_and_fp32_freed():
    import gc
    import weakref

    from solution.head.patch import arm_model, disarm_model

    model = _bf16_tiny_model()
    input_ids, attention_mask, labels = _ids_with_fixup_rows()
    embed = model.model.embed_tokens
    old = embed.weight
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    with torch.no_grad():
        ref_emb = embed(input_ids).clone()
        with torch.autocast("cuda", torch.float16):
            ref_emb_ac = embed(input_ids).clone()
        h = torch.randn(5, HIDDEN, device="cuda")
        ref_head = model.lm_head(h).clone()
    gc.collect()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    old_ref = weakref.ref(old)
    del old

    assert arm_model(model)
    try:
        gc.collect()
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated()
        fix = model.__dict__["_esep_fp16_fixup"]
        assert old_ref() is None, "the fp32 tied weight is still referenced"
        # fp32 (4 B/elem) freed, one fp16 copy (2 B/elem) and the small fix-up tables added.
        assert after - before <= -VOCAB * HIDDEN * 2 + fix.nbytes() + 4096, (before, after)
        assert model.lm_head.weight is embed.weight
        assert embed.weight.dtype == torch.float16 and not embed.weight.requires_grad
        assert {n for n, p in model.named_parameters() if p.requires_grad} == trainable
        assert len(list(model.parameters())) == len(set(map(id, model.parameters())))
        with torch.no_grad():
            emb = embed(input_ids)
            with torch.autocast("cuda", torch.float16):
                emb_ac = embed(input_ids)
            head = model.lm_head(h)
        assert emb.dtype == torch.float32 and torch.equal(_bits(emb), _bits(ref_emb))
        assert emb_ac.dtype == torch.float32 and torch.equal(_bits(emb_ac), _bits(ref_emb_ac))
        # No autocast: the head multiplies by the exact fp32 weight, as before.
        assert head.dtype == torch.float32 and torch.equal(head, ref_head)
    finally:
        disarm_model(model)


def _arm_and_run(share, use_hook=False):
    hp = importlib.import_module("solution.head.patch")  # "solution.head.patch" attr is the function

    model = _bf16_tiny_model()
    if use_hook:
        model.enable_input_require_grads()  # what prepare_model_for_kbit_training adds
    input_ids, attention_mask, labels = _ids_with_fixup_rows()
    num_items = (labels[:, 1:] != -100).sum()
    old_flag = hp._SHARE_FP16_WEIGHT
    hp._SHARE_FP16_WEIGHT = share
    try:
        assert hp.arm_model(model)
    finally:
        hp._SHARE_FP16_WEIGHT = old_flag
    assert (model.lm_head.weight.dtype == torch.float16) == share
    try:
        out, grads = _run(model, input_ids, attention_mask, labels, num_items)
        step = model.__dict__[hp._MODEL_ATTR].take()
        with torch.autocast("cuda", torch.float16), torch.no_grad():
            model.eval()
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            model.train()
    finally:
        hp.disarm_model(model)
    return out.loss.detach(), grads, step, logits


@pytest.mark.parametrize("use_hook", [False, True])
def test_shared_fp16_weight_matches_v1_bitwise(use_hook):
    loss1, grads1, step1, logits1 = _arm_and_run(share=False, use_hook=use_hook)
    loss2, grads2, step2, logits2 = _arm_and_run(share=True, use_hook=use_hook)
    assert torch.equal(loss1, loss2)
    assert grads1.keys() == grads2.keys()
    for name in grads1:
        assert torch.equal(grads1[name], grads2[name]), name
    for key in ("token_loss", "token_entropy", "correct", "labelled"):
        assert torch.equal(getattr(step1, key), getattr(step2, key)), key
    # The fallback lm_head path under fp16 autocast sees the same fp16 values as v1 / base.
    assert torch.equal(logits1, logits2)


def _peft_train(out_dir, share):
    import gc
    import weakref

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from soup_cli.trainer.rewind_hf import attach_rewind_state, make_rewind_trainer_class
    from trl import SFTConfig, SFTTrainer

    hp = importlib.import_module("solution.head.patch")  # "solution.head.patch" attr is the function

    model = _bf16_tiny_model()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05,
                                             target_modules=["q_proj", "v_proj"],
                                             task_type="CAUSAL_LM"))
    args = SFTConfig(
        output_dir=str(out_dir), max_steps=2, per_device_train_batch_size=2,
        gradient_accumulation_steps=2, learning_rate=1e-3, logging_steps=1, fp16=True,
        report_to=[], save_strategy="no", seed=0, data_seed=0, dataloader_num_workers=0,
        max_length=64, packing=False, padding_free=False,
    )
    _one_gpu(args)
    trainer = make_rewind_trainer_class(SFTTrainer)(
        model=model, args=args, train_dataset=_dataset(), processing_class=_tokenizer(),
    )
    sink = _Sink()
    state = attach_rewind_state(trainer, sink)
    causal_lm = hp._find_causal_lm(model)
    old_ref = weakref.ref(causal_lm.lm_head.weight)
    old_flag = hp._SHARE_FP16_WEIGHT
    hp._SHARE_FP16_WEIGHT = share
    try:
        assert hp.arm_trainer(trainer)
    finally:
        hp._SHARE_FP16_WEIGHT = old_flag
    gc.collect()
    assert (old_ref() is None) == share
    w = causal_lm.lm_head.weight
    try:
        trainer.train()
        opt_params = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
        assert id(w) not in opt_params
        assert opt_params == {id(p) for p in model.parameters() if p.requires_grad}
        trainer.save_model(str(out_dir))
    finally:
        hp.disarm_trainer(trainer)
    assert not state.failed
    logs = [h for h in trainer.state.log_history if "loss" in h]
    return logs, sink.records


def test_shared_fp16_weight_trainer_peft_save_identical(tmp_path, monkeypatch):
    from safetensors import safe_open
    # TEST-ONLY: this is peft's own module (not the torchao library); torchao is never imported or
    # used by solution/. The tiny model has plain nn.Linear layers, so peft's LoRA dispatch reaches
    # its torchao version probe, which raises on Kaggle's preinstalled torchao 0.10.0. The real run
    # dispatches to bitsandbytes and never gets there. We stub the probe to "not available".
    import peft.tuners.lora.torchao as peft_torchao

    monkeypatch.setattr(peft_torchao, "is_torchao_available", lambda: False)

    v1_logs, v1_rows = _peft_train(tmp_path / "v1", share=False)
    v2_logs, v2_rows = _peft_train(tmp_path / "v2", share=True)
    assert len(v1_logs) == len(v2_logs) == 2
    for a, b in zip(v1_logs, v2_logs):
        for key in ("loss", "entropy", "mean_token_accuracy", "num_tokens", "grad_norm"):
            assert a[key] == b[key], key
    assert v1_rows == v2_rows

    f1 = tmp_path / "v1" / "adapter_model.safetensors"
    f2 = tmp_path / "v2" / "adapter_model.safetensors"
    assert f1.read_bytes() == f2.read_bytes()
    with safe_open(str(f2), "pt") as f:
        keys = list(f.keys())
    assert keys and all("lora_" in k for k in keys), keys


# --------------------------------------------------------------------------------------------
# L1: prepare_model_for_kbit_training never builds the fp32 copy of an fp16 tied weight.
# --------------------------------------------------------------------------------------------

BIG_VOCAB = 32768  # makes the tied weight (2M elements) dominate the other params (~75k)


def _fp16_loaded(seed=0, dtype=torch.float16):
    """The tiny model as Soup loads the base on a T4: 16-bit frozen weights, flagged 4-bit (the
    NF4 layers are left out: only the flag matters to prepare_model_for_kbit_training)."""
    model = _tiny_model(seed, vocab=BIG_VOCAB).to(dtype)
    for p in model.parameters():
        p.requires_grad_(True)  # as from_pretrained leaves them
    model.is_loaded_in_4bit = True
    return model


def _hook_names(model):
    out = {}
    for name, m in model.named_modules():
        for kind in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks",
                     "_backward_pre_hooks", "_forward_hooks_with_kwargs"):
            hooks = getattr(m, kind, {})
            vals = [getattr(h, "__qualname__", repr(h)) for h in hooks.values()] \
                if isinstance(hooks, dict) else sorted(map(str, hooks))
            if vals:
                out[(name, kind)] = vals
        if hasattr(m, "gradient_checkpointing"):
            out[(name, "gradient_checkpointing")] = m.gradient_checkpointing
    return out


def _prepare_pair(use_gc, dtype=torch.float16):
    """(reference, ours, peaks): peft's own prepare vs ours on two identical models."""
    import gc

    from peft import prepare_model_for_kbit_training as prep

    from solution.head.patch import _prepare_keep_fp16_tied

    peaks = []
    models = []
    for ours in (False, True):
        model = _fp16_loaded(dtype=dtype)
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.memory_allocated()
        if ours:
            out = _prepare_keep_fp16_tied(prep, model, use_gradient_checkpointing=use_gc)
        else:
            out = prep(model, use_gradient_checkpointing=use_gc)
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated() - start)
        assert out is model
        models.append(model)
    return models[0], models[1], peaks


@pytest.mark.parametrize("use_gc", [False, True])
def test_prepare_keeps_fp16_tied_weight_same_side_effects(use_gc):
    from solution.head.patch import arm_model, disarm_model

    w_elems = BIG_VOCAB * HIDDEN
    ref, ours, (ref_peak, our_peak) = _prepare_pair(use_gc)

    # The reference really builds the fp32 copy; ours never allocates anything near it.
    assert ref_peak >= w_elems * 4, ref_peak
    assert our_peak < w_elems * 1, our_peak

    embed, head = ours.model.embed_tokens, ours.lm_head
    w = embed.weight
    assert head.weight is w and w.dtype == torch.float16 and w.is_contiguous()
    assert ours.__dict__["_esep_fp16_fixup"].count == 0
    assert ref.model.embed_tokens.weight.dtype == torch.float32
    assert torch.equal(_bits(ref.model.embed_tokens.weight), _bits(w.float()))

    # Every other parameter cast exactly as peft does; requires_grad identical everywhere.
    rp, op = dict(ref.named_parameters()), dict(ours.named_parameters())
    assert rp.keys() == op.keys()
    for name in rp:
        assert rp[name].requires_grad == op[name].requires_grad, name
        if op[name] is w:
            continue
        assert rp[name].dtype == op[name].dtype, name
        assert torch.equal(rp[name], op[name]), name
    del rp, op  # let the v2 swap below free the reference's fp32 weight
    assert not any(p.requires_grad for p in ours.parameters())

    # Same hooks on the same modules, same gradient-checkpointing state and config flags.
    assert _hook_names(ref) == _hook_names(ours)
    assert any(k[1] == "_forward_hooks" for k in _hook_names(ours)) == use_gc
    assert ref.is_gradient_checkpointing == ours.is_gradient_checkpointing == use_gc
    assert ref.config.to_dict() == ours.config.to_dict()
    assert ref.training == ours.training

    # Embedding output: fp32, bit-identical (with and without autocast), hooks still fire.
    g = torch.Generator().manual_seed(2)
    ids = torch.randint(0, BIG_VOCAB, (3, 17), generator=g).cuda()
    h = torch.randn(5, HIDDEN, device="cuda")
    for ac in (False, True):
        with torch.autocast("cuda", torch.float16, enabled=ac):
            a, b = ref.model.embed_tokens(ids), embed(ids)
            ha, hb = ref.lm_head(h), head(h)
        assert a.dtype == b.dtype == torch.float32
        assert torch.equal(_bits(a), _bits(b))
        assert a.requires_grad == b.requires_grad == use_gc
        assert ha.dtype == hb.dtype and torch.equal(ha, hb)

    # A training step (q/v trainable, as LoRA would be): identical loss and grads, then the same
    # after arming (v2 swap at arm time on the reference, a no-op on ours).
    input_ids, attention_mask, labels = _batch()
    num_items = (labels[:, 1:] != -100).sum()
    for m in (ref, ours):
        for layer in m.model.layers:
            layer.self_attn.q_proj.weight.requires_grad_(True)
            layer.self_attn.v_proj.weight.requires_grad_(True)
    out_r, grads_r = _run(ref, input_ids, attention_mask, labels, num_items)
    out_o, grads_o = _run(ours, input_ids, attention_mask, labels, num_items)
    assert torch.equal(out_r.loss, out_o.loss)
    assert torch.equal(out_r.logits, out_o.logits)
    for name in grads_r:
        assert torch.equal(grads_r[name], grads_o[name]), name
    assert arm_model(ref) and arm_model(ours)
    try:
        assert ours.model.embed_tokens.weight is w  # no swap, no copy
        assert ours.__dict__["_esep_head_state"].weight16(w).data_ptr() == w.data_ptr()
        assert ref.model.embed_tokens.weight.dtype == torch.float16  # the v2 swap
        out_r, grads_r = _run(ref, input_ids, attention_mask, labels, num_items)
        out_o, grads_o = _run(ours, input_ids, attention_mask, labels, num_items)
        assert out_r.logits is None and out_o.logits is None
        assert torch.equal(out_r.loss, out_o.loss)
        for name in grads_r:
            assert torch.equal(grads_r[name], grads_o[name]), name
    finally:
        disarm_model(ref)
        disarm_model(ours)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_prepare_non_fp16_tied_weight_untouched(dtype):
    ref, ours, _ = _prepare_pair(use_gc=False, dtype=dtype)
    assert "_esep_fp16_fixup" not in ours.__dict__
    assert "forward" not in ours.model.embed_tokens.__dict__
    assert "forward" not in ours.lm_head.__dict__
    assert ours.lm_head.weight is ours.model.embed_tokens.weight
    rp, op = dict(ref.named_parameters()), dict(ours.named_parameters())
    for name in rp:
        assert rp[name].dtype == op[name].dtype == torch.float32, name
        assert torch.equal(rp[name], op[name]) and rp[name].requires_grad == op[name].requires_grad
    assert _hook_names(ref) == _hook_names(ours)


def test_soup_setup_wrapper_installs_and_restores(monkeypatch):
    import sys
    import types

    import peft
    from soup_cli.trainer import sft as soup_sft

    hp = importlib.import_module("solution.head.patch")
    orig_prep = peft.prepare_model_for_kbit_training
    seen = []

    def fake_setup(self, cfg, tcfg):  # what Soup's _setup_transformers does with peft
        from peft import prepare_model_for_kbit_training

        seen.append(prepare_model_for_kbit_training)
        self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=False)

    monkeypatch.setattr(soup_sft.SFTTrainerWrapper, "_setup_transformers", fake_setup)
    monkeypatch.setitem(hp._patched, "setup", False)
    hp._patch_soup_setup()
    wrapped = soup_sft.SFTTrainerWrapper._setup_transformers
    assert wrapped._esep_original is fake_setup

    self = types.SimpleNamespace(model=_fp16_loaded())
    wrapped(self, None, None)
    assert seen[-1] is not orig_prep and seen[-1]._esep_original is orig_prep
    assert peft.prepare_model_for_kbit_training is orig_prep  # restored
    w = self.model.model.embed_tokens.weight
    assert w.dtype == torch.float16 and self.model.lm_head.weight is w and not w.requires_grad

    # Soup's tests: a SimpleNamespace peft. The wrapper must stay out of the way.
    fake = types.SimpleNamespace(prepare_model_for_kbit_training=lambda m, **k: ("fake", m))
    monkeypatch.setitem(sys.modules, "peft", fake)
    self = types.SimpleNamespace(model="not a module")
    wrapped(self, None, None)
    assert self.model == ("fake", "not a module")
    assert fake.prepare_model_for_kbit_training is seen[-1]


def _setup_and_save(out_dir, ours):
    """Soup's T4 setup on the tiny fp16 model: kbit prep (ours or peft's own), the task's LoRA,
    arming; then save the adapter. Returns (trainable {name: dtype}, model)."""
    from peft import LoraConfig, get_peft_model
    from peft import prepare_model_for_kbit_training as prep

    from solution.head.patch import _find_causal_lm, _prepare_keep_fp16_tied, arm_model

    model = _fp16_loaded()
    if ours:
        model = _prepare_keep_fp16_tied(prep, model, use_gradient_checkpointing=True)
    else:
        model = prep(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules=["q_proj", "v_proj"],
                                             task_type="CAUSAL_LM"))
    assert arm_model(_find_causal_lm(model))
    trainable = {n: p.dtype for n, p in model.named_parameters() if p.requires_grad}
    model.save_pretrained(str(out_dir))
    return trainable, model


def test_setup_trainable_fp32_and_adapter_f32(tmp_path, monkeypatch):
    """The fp16 tied weight is a storage change of one frozen tensor: after setup every trainable
    parameter is fp32 and the saved adapter holds only F32 tensors, the same names / dtypes /
    shapes as the unpatched path."""
    from safetensors import safe_open

    from solution.head.patch import _find_causal_lm, disarm_model
    # TEST-ONLY stub of peft's own torchao version probe, see
    # test_shared_fp16_weight_trainer_peft_save_identical for why.
    import peft.tuners.lora.torchao as peft_torchao

    monkeypatch.setattr(peft_torchao, "is_torchao_available", lambda: False)

    out = {}
    for ours in (False, True):
        trainable, model = _setup_and_save(tmp_path / str(ours), ours)
        try:
            assert trainable and all("lora_" in n for n in trainable), trainable
            assert all(d == torch.float32 for d in trainable.values()), trainable
            causal_lm = _find_causal_lm(model)
            w = causal_lm.lm_head.weight
            # Both paths end with the fp16 tied weight (ours from setup, the reference by the
            # arm-time swap), frozen and shared.
            assert w is causal_lm.model.embed_tokens.weight
            assert w.dtype == torch.float16 and not w.requires_grad
        finally:
            disarm_model(_find_causal_lm(model))
        with safe_open(str(tmp_path / str(ours) / "adapter_model.safetensors"), "pt") as f:
            out[ours] = {k: (f.get_slice(k).get_dtype(), tuple(f.get_slice(k).get_shape()))
                         for k in f.keys()}
        del model
    assert out[True] == out[False]
    assert out[True] and all(d == "F32" for d, _ in out[True].values()), out[True]
    assert len(out[True]) == 2 * 2 * 2  # 2 layers x {q_proj, v_proj} x {lora_A, lora_B}
    assert all(16 in shape for _, shape in out[True].values())
