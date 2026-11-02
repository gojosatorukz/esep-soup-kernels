"""Wire the fused LM head (``fused_linear_ce``) into the Qwen2 causal LM and TRL's SFTTrainer.

What changes, and only for a trainer we armed (see ``arm_trainer``):

1. ``Qwen2ForCausalLM.forward`` (class-level wrapper, active only on an armed model instance,
   in training mode, with labels, on CUDA, under fp16 autocast): the decoder runs as before, but
   ``lm_head`` is never applied to all positions. The final-norm hidden states of the positions
   that carry a token (attention_mask == 1) or a target are gathered, the next-token targets are
   built exactly as ``transformers.loss.loss_utils.ForCausalLMLoss`` does (labels[:, t + 1],
   -100 at the last position) and ``fused_linear_ce`` returns the summed cross-entropy plus
   per-token stats. The loss is divided by ``num_items_in_batch`` (or by the number of valid
   targets when it is None), the same reduction as ``fixed_cross_entropy``. The output carries
   ``loss`` and ``logits=None``; the per-token stats are kept on the model for the trainer.
2. ``trl.SFTTrainer.compute_loss`` (class-level wrapper, active only on an armed trainer in
   training mode): runs ``transformers.Trainer.compute_loss`` exactly as TRL does, then logs
   ``entropy``, ``num_tokens`` and ``mean_token_accuracy`` with TRL 0.29.1's formulas
   (sft_trainer.py:1239-1320) from the stats instead of the logits, and leaves the per-token
   losses on ``trainer._soup_chunked_step.token_loss`` (B, T), which Soup's rewind recorder
   reads when the model returns no logits (soup_cli/trainer/rewind_hf.py:347-357).
3. ``soup_cli.trainer.sft.SFTTrainerWrapper.train`` (wrapper): arms ``self.trainer`` once the
   trainer and model exist, before training starts; disarms afterwards.
4. The frozen tied embed_tokens / lm_head weight is stored as one fp16 Parameter shared by both
   modules instead of an fp32 one. On the T4 Soup loads the frozen base in fp16
   (soup_cli/utils/gpu.py:634-656), so on this path the weight is kept fp16 from setup: Soup's
   ``peft.prepare_model_for_kbit_training`` call is wrapped (``_patch_soup_setup``,
   ``_prepare_keep_fp16_tied``) so that its cast loop never upcasts it, and no fp32 copy ever
   exists. The fallback, for a base loaded in bf16 (whose tied weight peft turns into fp32), is
   the swap at arming: the fp32 weight is replaced by its fp16 rounding plus an exact fix-up table
   and the fp32 storage is freed (``_share_fp16_tied_weight``). Either way the embedding output
   stays fp32 and bit-identical to before (``_install_fp16_forwards``); the head sees the same
   fp16 values autocast gave it. Stays after disarm.

What item 4 is, and is not. It is a storage optimization of ONE frozen tensor: on the fp16-load
path the fp32 copy peft would build is an exact upcast of the fp16 values Soup loaded, so it
holds no information the fp16 tensor does not. Trainable parameters are untouched: the LoRA
parameters stay fp32 (peft creates them in fp32; Soup's align_trainable_dtype_for_fp16 casts any
bf16 ``lora_*`` by name and dtype, soup utils/mixed_precision.py:109-112), so the optimizer state
and the saved adapter are fp32 as in the base run (test_setup_trainable_fp32_and_adapter_f32).
``model.dtype`` (the dtype of the first floating-point parameter) and the embed / head weight
dtype now read fp16 where the base reads fp32; we do not fake them. The readers on the run's path
were checked, and none changes a decision or an output:
- transformers masking_utils.py:941 ``create_causal_mask`` takes its dtype from
  ``inputs_embeds``, which our embedding forward returns as fp32, so the mask min value and the
  first norm's input are unchanged;
- Soup utils/mixed_precision.py:109-112 ``align_trainable_dtype_for_fp16`` selects ``lora_*``
  parameters by name and dtype, not by ``model.dtype``;
- the check-5 fields (global_step, logged metrics, adapter tensor dtypes / shapes,
  adapter_config, trainer args, rewind rows) do not derive from ``model.dtype``;
- ``PeftModel.save_pretrained`` writes no dtype (``config.dtype`` is written only by a full-model
  save, which the run does not do).

Sources of ideas: dropping the logits from the model output and rebuilding the trainer's
metrics from per-token statistics follows the general approach of Liger Kernel's
FusedLinearCrossEntropy / Apple Cut Cross-Entropy integrations; the metric formulas are TRL
0.29.1's and Soup 0.77.0's own (cited above). The fp16 tied-weight sharing with an exact
fix-up table is our own. No code from these projects is used.

Nothing here changes trainer arguments, Soup settings, data, or trainable parameters. Every
check that does not hold (fake modules in Soup's tests, no CUDA, liger, padding-free, a trainable
or biased head, another precision, ...) leaves the original code path untouched.
"""

import functools
import warnings

_MODEL_ATTR = "_esep_head_state"  # on the Qwen2ForCausalLM instance: _HeadState
_TRAINER_ATTR = "_esep_head_causal_lm"  # on the trainer instance: the armed causal LM
_SOUP_STEP_ATTR = "_soup_chunked_step"  # Soup's rewind fallback slot (rewind_hf.py:349)

_patched = {"wrapper": False, "setup": False, "model": False, "trainer": False}


class HeadStep:
    """Per-token results of one fused head forward, all detached, shaped (B, T)."""

    __slots__ = ("token_loss", "token_entropy", "correct", "labelled")

    def __init__(self, token_loss, token_entropy, correct, labelled):
        self.token_loss = token_loss  # CE of labels[:, t+1]; 0.0 where ignored / last position
        self.token_entropy = token_entropy  # entropy of the prediction at t; 0.0 off the rows
        self.correct = correct  # 0-dim int64: argmax == target over valid targets
        self.labelled = labelled  # 0-dim int64: number of valid targets


class _HeadState:
    """Per-model state: the cached fp16 head weight and the last step's stats."""

    def __init__(self):
        self.w16 = None
        self.w16_key = None
        self.last_step = None

    def weight16(self, weight):
        import torch

        key = (weight.data_ptr(), tuple(weight.shape), weight.dtype, weight.device, weight._version)
        if self.w16 is None or self.w16_key != key:
            with torch.no_grad():
                self.w16 = weight.detach().to(torch.float16).contiguous()
            self.w16_key = key
        return self.w16

    def take(self):
        step, self.last_step = self.last_step, None
        return step


# --------------------------------------------------------------------------------------------
# One fp16 tied weight for the embedding lookup and the head
# --------------------------------------------------------------------------------------------
#
# After prepare_model_for_kbit_training the tied embed_tokens / lm_head weight is a frozen fp32
# Parameter (Qwen2.5-1.5B: [151936, 1536], 0.93 GB) and the head reads it only as fp16 (autocast
# casts it in the base run; the fused op takes fp16). The checkpoint is bf16, so almost every
# element survives fp32 -> fp16 -> fp32 unchanged. The exceptions are values below fp16's normal
# range (subnormal rounding; Qwen2.5-1.5B-Instruct: 43,373 of 233,373,696 elements, in 34,883
# rows, at most 5 per row, |error| <= 2.98e-8) and, in general, values that overflow fp16.
#
# When the trainer is armed, the fp32 Parameter is replaced by ONE fp16 Parameter shared by both
# modules (still tied), and the fp32 storage is freed. The head gets exactly the fp16 values the
# base run's autocast cast produces. The embedding lookup must keep the base run's fp32 values bit
# for bit, so it gathers fp16 rows, upcasts them to fp32 and writes back the exact fp32 value of
# every element that does not round-trip. Those elements are kept in a small padded per-row table
# (rows x max-per-row), so the fix-up has fixed shapes and needs no host sync. lm_head outside
# fp16 autocast (not used by the training run) rebuilds the exact fp32 weight on the fly.

_SHARE_FP16_WEIGHT = True  # tests turn it off to compare against v1 (fp32 weight + fp16 cache)
_MAX_FIXUP_FRACTION = 0.01  # refuse when more elements than this fail the fp16 round trip
_MAX_FIXUP_PER_ROW = 64
_FIXUP_CHUNK_ROWS = 8192


class _Fp16Fixup:
    """The elements of a fp32 weight that fp16 does not hold exactly, and how to restore them."""

    def __init__(self, w32, w16):
        import torch

        v, h = w32.shape
        flat_idx, flat_val = [], []
        for r0 in range(0, v, _FIXUP_CHUNK_ROWS):
            a = w32[r0:r0 + _FIXUP_CHUNK_ROWS]
            b = w16[r0:r0 + _FIXUP_CHUNK_ROWS].float()
            bad = (a.view(torch.int32) != b.view(torch.int32)).reshape(-1).nonzero().squeeze(1)
            flat_idx.append(bad + r0 * h)
            flat_val.append(a.reshape(-1).index_select(0, bad))
            del a, b
        self.flat_idx = torch.cat(flat_idx)  # int64 [K], sorted
        self.flat_val = torch.cat(flat_val)  # fp32 [K], exact base values
        self.count = int(self.flat_idx.numel())
        dev = w32.device
        rows = self.flat_idx // h
        cols = self.flat_idx % h
        uniq, counts = torch.unique_consecutive(rows, return_counts=True)
        n_rows = int(uniq.numel())
        width = int(counts.max()) if n_rows else 1
        self.rows, self.width = n_rows, width
        # slot[row] = index into the tables, n_rows (a dummy slot) for rows without fix-ups.
        self.slot = torch.full((v,), n_rows, device=dev, dtype=torch.int64)
        self.slot[uniq] = torch.arange(n_rows, device=dev)
        # Padded tables. Padding repeats the row's first entry, so a scatter writes the same value
        # twice; the dummy slot points at column 0 and is masked to keep the current value.
        self.cols = torch.zeros((n_rows + 1, width), device=dev, dtype=torch.int64)
        self.vals = torch.zeros((n_rows + 1, width), device=dev, dtype=torch.float32)
        if n_rows:
            starts = torch.cumsum(counts, 0) - counts
            slot_of = torch.repeat_interleave(torch.arange(n_rows, device=dev), counts)
            pos = torch.arange(self.count, device=dev) - starts[slot_of]
            first = starts[torch.arange(n_rows, device=dev)]
            self.cols[:n_rows] = cols[first][:, None]
            self.vals[:n_rows] = self.flat_val[first][:, None]
            self.cols[slot_of, pos] = cols
            self.vals[slot_of, pos] = self.flat_val

    @classmethod
    def exact(cls, w16):
        """The fix-up of a weight whose fp32 values are exactly ``w16.float()``: no entries.

        Builds no fp32 copy of the weight; ``apply_rows`` is a no-op and ``exact_weight`` is a
        plain upcast.
        """
        import torch

        self = cls.__new__(cls)
        dev = w16.device
        self.flat_idx = torch.empty((0,), device=dev, dtype=torch.int64)
        self.flat_val = torch.empty((0,), device=dev, dtype=torch.float32)
        self.count = 0
        self.rows, self.width = 0, 1
        self.slot = torch.empty((0,), device=dev, dtype=torch.int64)  # unused when rows == 0
        self.cols = torch.zeros((1, 1), device=dev, dtype=torch.int64)
        self.vals = torch.zeros((1, 1), device=dev, dtype=torch.float32)
        return self

    def nbytes(self):
        return sum(t.numel() * t.element_size()
                   for t in (self.flat_idx, self.flat_val, self.slot, self.cols, self.vals))

    def apply_rows(self, out, ids):
        """out [N, H] fp32 = upcast fp16 rows for token ids [N]; restores the exact values."""
        import torch

        if not self.rows:
            return out
        s = self.slot.index_select(0, ids)
        cols = self.cols.index_select(0, s)
        cur = out.gather(1, cols)
        keep = (s == self.rows)[:, None]
        out.scatter_(1, cols, torch.where(keep, cur, self.vals.index_select(0, s)))
        return out

    def exact_weight(self, w16):
        w = w16.float()
        w.view(-1).index_copy_(0, self.flat_idx, self.flat_val)
        return w


def _embedding_forward16(module, fixup, input):
    import torch
    import torch.nn.functional as F

    out = F.embedding(input, module.weight, module.padding_idx, None, 2.0,
                      module.scale_grad_by_freq, module.sparse).to(torch.float32)
    h = out.shape[-1]
    fixup.apply_rows(out.view(-1, h), input.reshape(-1))
    return out


def _lm_head_forward16(module, fixup, input):
    import torch
    import torch.nn.functional as F

    weight = module.weight
    if input.dtype != weight.dtype and not torch.is_autocast_enabled(input.device.type):
        # The base module would multiply by the fp32 weight: rebuild it exactly.
        weight = fixup.exact_weight(weight).to(input.dtype)
    return F.linear(input, weight, module.bias)


def _tied_pair(causal_lm):
    """(embed_tokens, lm_head) when they are plain modules sharing one 2-D CUDA Parameter."""
    import torch

    if not _SHARE_FP16_WEIGHT or causal_lm.__dict__.get("_esep_fp16_fixup") is not None:
        return None
    embed = causal_lm.get_input_embeddings()
    head = getattr(causal_lm, "lm_head", None)
    if type(embed) is not torch.nn.Embedding or type(head) is not torch.nn.Linear:
        return None
    if embed is not getattr(getattr(causal_lm, "model", None), "embed_tokens", None):
        return None
    w = embed.weight
    if head.weight is not w or head.bias is not None or embed.max_norm is not None:
        return None
    if type(w) is not torch.nn.Parameter or not w.is_cuda or not w.is_contiguous() or w.dim() != 2:
        return None
    if "forward" in embed.__dict__ or "forward" in head.__dict__:
        return None  # someone already replaced a forward; do not stack on it
    return embed, head


def _install_fp16_forwards(causal_lm, embed, head, fixup):
    """Give embed / head the forwards for an fp16 tied weight (outputs as with the fp32 weight).

    The embedding returns fp32, bit-identical to the fp32-weight embedding (``fixup`` restores
    the elements fp16 cannot hold; empty on the fp16-load path), so every reader of
    ``inputs_embeds.dtype`` downstream (e.g. create_causal_mask) sees fp32 as before.
    """
    embed.forward = functools.partial(_embedding_forward16, embed, fixup)
    head.forward = functools.partial(_lm_head_forward16, head, fixup)
    causal_lm.__dict__["_esep_fp16_fixup"] = fixup


def _share_fp16_tied_weight(causal_lm):
    """Replace the frozen fp32 tied weight by one fp16 Parameter. True when done.

    The arm-time fallback of module docstring item 4: it acts only when the tied weight is fp32
    at arming, i.e. a base loaded in bf16 (or fp32). On the T4 fp16-load path
    ``_prepare_keep_fp16_tied`` already left the weight fp16 and installed the fp16 forwards, so
    ``_tied_pair`` returns None here and nothing happens. Frozen weight only; trainable
    parameters are never touched.
    """
    import gc
    import weakref

    import torch

    pair = _tied_pair(causal_lm)
    if pair is None:
        return False
    embed, head = pair
    w32 = embed.weight
    if w32.dtype != torch.float32 or w32.requires_grad:
        return False
    with torch.no_grad():
        w16 = w32.detach().to(torch.float16)
        fixup = _Fp16Fixup(w32.detach(), w16)
    if fixup.count > _MAX_FIXUP_FRACTION * w32.numel() or fixup.width > _MAX_FIXUP_PER_ROW:
        warnings.warn(f"fused head: tied weight kept in fp32, {fixup.count} elements do not "
                      "round-trip through fp16", stacklevel=2)
        return False
    param = torch.nn.Parameter(w16, requires_grad=False)
    old_ref = weakref.ref(w32)
    embed.weight = param
    head.weight = param
    _install_fp16_forwards(causal_lm, embed, head, fixup)
    del w32
    if old_ref() is not None:
        gc.collect()
    if old_ref() is not None:
        warnings.warn("fused head: the old fp32 tied weight is still referenced; its memory is "
                      "not freed", stacklevel=2)
    return True


# On the T4 Soup loads the frozen base in fp16, so the tied weight is fp16 when Soup calls
# peft.prepare_model_for_kbit_training, whose cast loop (peft/utils/other.py:193-202) upcasts it
# to fp32 in place (param.data = param.data.to(fp32)). The fp32 values are then exactly
# w16.float(): the fix-up table is empty and the fp32 copy holds nothing the fp16 one does not.
# _prepare_keep_fp16_tied hides the fp16 Parameter from that loop and installs the fp16 state
# directly, so the 2 B/elem -> 4 B/elem upcast (and the later fp32 -> fp16 swap at arming, with
# fp32 and fp16 alive together) never happens.


def _fp16_tied_candidate(model):
    """(causal_lm, embed, head, w16) when ``model`` holds a Qwen2 LM with an fp16 tied weight."""
    import sys

    import torch

    if not isinstance(model, torch.nn.Module):
        return None
    mq = sys.modules.get("transformers.models.qwen2.modeling_qwen2")
    cls = getattr(mq, "Qwen2ForCausalLM", None)
    if not isinstance(cls, type):
        return None
    causal_lm = next((m for m in model.modules() if isinstance(m, cls)), None)
    if causal_lm is None:
        return None
    pair = _tied_pair(causal_lm)
    if pair is None:
        return None
    embed, head = pair
    w16 = embed.weight
    if w16.dtype != torch.float16:
        return None
    return causal_lm, embed, head, w16


def _prepare_keep_fp16_tied(original, model, *args, **kwargs):
    """peft's prepare_model_for_kbit_training without the fp32 copy of an fp16 tied weight.

    The setup-time path of module docstring item 4 (the T4 fp16 load). Only the one frozen
    tied weight is kept out of peft's cast; its fp32 copy would be the exact upcast
    ``w16.float()`` (so ``_Fp16Fixup.exact`` has an empty table). Every other parameter gets
    peft's own cast, and the LoRA parameters peft adds later are fp32 as without the patch.

    The fp16 Parameter is taken off embed_tokens / lm_head for the call (an empty fp32
    placeholder Parameter, shared by both, sits there so the cast loop skips it) and put back
    afterwards with the placeholder's requires_grad, i.e. what peft set. The module objects stay
    the same, so every hook, flag and config change peft makes is kept as is. Anything
    unexpected leaves the state peft would have produced.
    """
    import torch

    try:
        found = _fp16_tied_candidate(model)
    except Exception:  # noqa: BLE001 - the original call is always valid
        found = None
    if found is None:
        return original(model, *args, **kwargs)
    causal_lm, embed, head, w16 = found
    placeholder = torch.nn.Parameter(torch.empty((0,), device=w16.device, dtype=torch.float32),
                                     requires_grad=w16.requires_grad)
    embed.weight = placeholder
    head.weight = placeholder
    try:
        result = original(model, *args, **kwargs)
    finally:
        intact = embed.weight is placeholder and head.weight is placeholder
        embed.weight = w16
        head.weight = w16
    w16.requires_grad_(placeholder.requires_grad)
    if not intact or placeholder.dtype != torch.float32 or w16.dtype != torch.float16:
        # Not the peft we know: give the weight the cast peft gives every 16-bit parameter.
        warnings.warn("fused head: prepare_model_for_kbit_training changed the tied weight "
                      "placeholder; tied weight cast to fp32 as peft does", stacklevel=2)
        if w16.dtype == torch.float16:
            w16.data = w16.data.to(torch.float32)
        return result
    _install_fp16_forwards(causal_lm, embed, head, _Fp16Fixup.exact(w16))
    return result


def _stat(per_token, name):
    """Read a stat from the op's result (namedtuple or dict)."""
    if isinstance(per_token, dict):
        return per_token[name]
    return getattr(per_token, name)


# --------------------------------------------------------------------------------------------
# Model side
# --------------------------------------------------------------------------------------------


def _fused_applicable(model, kwargs, labels, logits_to_keep):
    import torch

    state = model.__dict__.get(_MODEL_ATTR)
    if state is None or labels is None or not model.training:
        return None
    if "shift_labels" in kwargs or kwargs.get("return_dict") is False:
        return None
    if isinstance(logits_to_keep, torch.Tensor) or logits_to_keep != 0:
        return None
    if not torch.is_tensor(labels) or not labels.is_cuda:
        return None
    if not (torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.float16):
        return None
    head = model.lm_head
    if head.weight.requires_grad or getattr(head, "bias", None) is not None:
        return None
    return state


def _fused_causal_lm_forward(model, state, input_ids, attention_mask, position_ids, past_key_values,
                             inputs_embeds, labels, use_cache, kwargs):
    import torch
    from transformers.modeling_outputs import CausalLMOutputWithPast

    from .op import fused_linear_ce

    outputs = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        **kwargs,
    )
    hidden = outputs.last_hidden_state  # [B, T, H], final norm output
    bsz, seq = hidden.shape[:2]
    labels = labels.to(hidden.device)
    # ForCausalLMLoss: pad labels with -100 on the right, then shift by one.
    target = torch.nn.functional.pad(labels, (0, 1), value=-100)[:, 1:]
    valid = target != -100
    if attention_mask is not None and attention_mask.dim() == 2:
        rows = (attention_mask.to(hidden.device) != 0) | valid
    else:
        rows = torch.ones_like(valid)
    flat_rows = rows.reshape(-1).nonzero().squeeze(1)
    x = hidden.reshape(bsz * seq, -1).index_select(0, flat_rows).to(torch.float16)
    tgt = target.reshape(-1).index_select(0, flat_rows).to(torch.int64)
    tgt_valid = tgt != -100
    ce_weight = tgt_valid.to(torch.float32)

    loss_sum, per_token = fused_linear_ce(x, state.weight16(model.lm_head.weight), tgt, ce_weight)

    # fixed_cross_entropy: sum / num_items_in_batch, or mean over valid targets.
    num_items = kwargs.get("num_items_in_batch")
    if num_items is None:
        loss = loss_sum / tgt_valid.sum()
    else:
        if torch.is_tensor(num_items):
            num_items = num_items.to(loss_sum.device)
        loss = loss_sum / num_items

    with torch.no_grad():
        if isinstance(per_token, dict) and "ce" in per_token:
            ce = per_token["ce"].float()
        else:
            ce = _stat(per_token, "lse").float() - _stat(per_token, "target_logit").float()
        ce = torch.where(tgt_valid, ce, 0.0)
        token_loss = hidden.new_zeros((bsz * seq,), dtype=torch.float32)
        token_loss.index_copy_(0, flat_rows, ce)
        token_entropy = hidden.new_zeros((bsz * seq,), dtype=torch.float32)
        token_entropy.index_copy_(0, flat_rows, _stat(per_token, "entropy").float())
        argmax = _stat(per_token, "argmax").to(torch.int64)
        correct = ((argmax == tgt) & tgt_valid).sum()
        state.last_step = HeadStep(
            token_loss=token_loss.view(bsz, seq),
            token_entropy=token_entropy.view(bsz, seq),
            correct=correct,
            labelled=tgt_valid.sum(),
        )

    return CausalLMOutputWithPast(
        loss=loss,
        logits=None,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def _patch_qwen2_forward():
    if _patched["model"]:
        return
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2ForCausalLM
    original = cls.forward

    @functools.wraps(original)  # keeps the signature (**kwargs) Trainer inspects
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, use_cache=None, logits_to_keep=0, **kwargs):
        state = _fused_applicable(self, kwargs, labels, logits_to_keep)
        if state is None:
            return original(self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                            logits_to_keep=logits_to_keep, **kwargs)
        return _fused_causal_lm_forward(self, state, input_ids, attention_mask, position_ids,
                                        past_key_values, inputs_embeds, labels, use_cache, kwargs)

    forward._esep_original = original
    cls.forward = forward
    _patched["model"] = True


def arm_model(causal_lm):
    """Turn the fused head on for one Qwen2ForCausalLM instance. Returns True when armed."""
    import torch

    _patch_qwen2_forward()
    from transformers.models.qwen2 import modeling_qwen2

    if not isinstance(causal_lm, modeling_qwen2.Qwen2ForCausalLM):
        return False
    head = getattr(causal_lm, "lm_head", None)
    if type(head) is not torch.nn.Linear or head.bias is not None:
        return False
    if head.weight.requires_grad or not head.weight.is_cuda:
        return False  # the fused op gives no gradient to the head weight
    if getattr(causal_lm.config, "final_logit_softcapping", None) is not None:
        return False
    if getattr(causal_lm.config, "loss_type", None) not in (None, "ForCausalLM"):
        return False
    if causal_lm.__dict__.get(_MODEL_ATTR) is None:
        causal_lm.__dict__[_MODEL_ATTR] = _HeadState()
    try:
        _share_fp16_tied_weight(causal_lm)  # nothing is changed unless it fully succeeds
    except Exception as exc:  # the fp32 weight (v1) is always a valid state
        warnings.warn(f"fused head: tied weight kept in fp32: {exc!r}", stacklevel=2)
    return True


def disarm_model(causal_lm):
    state = causal_lm.__dict__.pop(_MODEL_ATTR, None)
    if state is not None:
        state.w16 = None
        state.last_step = None


# --------------------------------------------------------------------------------------------
# Trainer side
# --------------------------------------------------------------------------------------------


def _step_from_logits(logits, labels, attention_mask):
    """The same stats from logits, for a step whose forward did not take the fused path."""
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        logits = logits.detach()
        bsz, seq, vocab = logits.shape
        target = F.pad(labels, (0, 1), value=-100)[:, 1:].to(logits.device)
        valid = target != -100
        token_loss = F.cross_entropy(logits.float().reshape(-1, vocab), target.reshape(-1),
                                     ignore_index=-100, reduction="none").view(bsz, seq)
        token_loss = torch.where(valid, token_loss, 0.0)
        from trl.trainer.utils import entropy_from_logits

        token_entropy = entropy_from_logits(logits)
        correct = ((logits.argmax(dim=-1) == target) & valid).sum()
    return HeadStep(token_loss, token_entropy, correct, valid.sum())


def _log_trl_metrics(trainer, mode, inputs, step):
    """TRL 0.29.1 SFTTrainer.compute_loss metrics (sft_trainer.py:1239-1320), from the stats."""
    import torch

    with torch.no_grad():
        if "attention_mask" in inputs:
            attention_mask = inputs["attention_mask"]
            entropy = torch.sum(step.token_entropy * attention_mask) / attention_mask.sum()
        elif "position_ids" in inputs:
            entropy = torch.mean(step.token_entropy)
        else:
            raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
        entropy = trainer.accelerator.gather_for_metrics(entropy).mean().item()
    trainer._metrics[mode]["entropy"].append(entropy)

    if mode == "train":
        if "attention_mask" in inputs:
            num_tokens_in_batch = trainer.accelerator.gather_for_metrics(
                inputs["attention_mask"].sum()).sum().item()
        elif "position_ids" in inputs:
            local_num_tokens = torch.tensor(inputs["position_ids"].size(1),
                                            device=inputs["position_ids"].device)
            num_tokens_in_batch = trainer.accelerator.gather_for_metrics(local_num_tokens).sum().item()
        else:
            raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
        trainer._total_train_tokens += num_tokens_in_batch
    trainer._metrics[mode]["num_tokens"] = [trainer._total_train_tokens]

    with torch.no_grad():
        correct_tokens = trainer.accelerator.gather_for_metrics(step.correct)
        total_tokens = trainer.accelerator.gather_for_metrics(step.labelled)
        total_sum = total_tokens.sum()
        accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
    trainer._metrics[mode]["mean_token_accuracy"].append(accuracy)


def _patch_sft_trainer():
    if _patched["trainer"]:
        return
    import trl

    sft_cls = trl.SFTTrainer
    original = sft_cls.compute_loss

    @functools.wraps(original)
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        causal_lm = self.__dict__.get(_TRAINER_ATTR)
        state = None if causal_lm is None else causal_lm.__dict__.get(_MODEL_ATTR)
        if (
            state is None
            or not self.model.training
            or "labels" not in inputs
            or "shift_labels" in inputs
        ):
            return original(self, model, inputs, return_outputs=return_outputs,
                            num_items_in_batch=num_items_in_batch)
        mode = "train"
        # As TRL does before it delegates to transformers.Trainer.compute_loss.
        inputs.pop("_prediction_loss_only", None)
        labels = inputs["labels"]
        inputs["use_cache"] = False
        setattr(self, _SOUP_STEP_ATTR, None)
        state.last_step = None
        loss, outputs = super(sft_cls, self).compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        step = state.take()
        if step is None:
            logits = getattr(outputs, "logits", None)
            if logits is None:
                raise RuntimeError("fused head: the model returned neither logits nor head stats")
            warnings.warn("fused head: forward took the original lm_head path", stacklevel=2)
            step = _step_from_logits(logits, labels, inputs.get("attention_mask"))
        setattr(self, _SOUP_STEP_ATTR, step)
        _log_trl_metrics(self, mode, inputs, step)
        return (loss, outputs) if return_outputs else loss

    compute_loss._esep_original = original
    sft_cls.compute_loss = compute_loss
    _patched["trainer"] = True


def _find_causal_lm(model):
    from transformers.models.qwen2 import modeling_qwen2

    for module in model.modules():
        if isinstance(module, modeling_qwen2.Qwen2ForCausalLM):
            return module
    return None


def arm_trainer(trainer):
    """Arm a TRL SFTTrainer (and its Qwen2 model) for the fused head. Returns True when armed.

    Any condition the fused path does not reproduce exactly leaves the trainer untouched.
    """
    import torch

    model = getattr(trainer, "model", None)
    if not isinstance(model, torch.nn.Module) or not torch.cuda.is_available():
        return False
    import trl

    if not isinstance(trainer, trl.SFTTrainer):
        return False
    args = getattr(trainer, "args", None)
    if args is None or getattr(args, "use_liger_kernel", False):
        return False
    if getattr(trainer, "padding_free", False) or getattr(trainer, "num_virtual_tokens", 0):
        return False
    if getattr(trainer, "aux_loss_enabled", False):
        return False
    if getattr(trainer, "compute_loss_func", None) is not None:
        return False
    if getattr(trainer, "label_smoother", None) is not None:
        return False
    if not hasattr(trainer, "_metrics") or not hasattr(trainer, "_total_train_tokens"):
        return False
    causal_lm = _find_causal_lm(model)
    if causal_lm is None or not arm_model(causal_lm):
        return False
    _patch_sft_trainer()
    trainer.__dict__[_TRAINER_ATTR] = causal_lm
    return True


def disarm_trainer(trainer):
    causal_lm = trainer.__dict__.pop(_TRAINER_ATTR, None)
    if causal_lm is not None:
        disarm_model(causal_lm)


# --------------------------------------------------------------------------------------------
# Soup side
# --------------------------------------------------------------------------------------------


def _patch_soup_train():
    if _patched["wrapper"]:
        return
    from soup_cli.trainer import sft as soup_sft

    cls = soup_sft.SFTTrainerWrapper
    original = cls.train

    @functools.wraps(original)
    def train(self, *args, **kwargs):
        trainer = getattr(self, "trainer", None)
        armed = False
        if trainer is not None:
            try:
                armed = arm_trainer(trainer)
            except Exception as exc:  # the original path is always a valid run
                import sys

                print(f"[solution.head] fused head not armed: {exc!r}", file=sys.stderr)
                armed = False
        try:
            return original(self, *args, **kwargs)
        finally:
            if armed:
                disarm_trainer(trainer)

    train._esep_original = original
    cls.train = train
    _patched["wrapper"] = True


def _real_peft():
    """The real peft module, or None (Soup's tests put SimpleNamespace fakes in sys.modules)."""
    import sys
    import types

    if "peft" not in sys.modules:
        try:
            import peft  # noqa: F401 - Soup's _setup_transformers imports it on its first line
        except Exception:  # noqa: BLE001 - let Soup's own import report it
            return None
    peft = sys.modules.get("peft")
    if not isinstance(peft, types.ModuleType):
        return None
    if not callable(getattr(peft, "prepare_model_for_kbit_training", None)):
        return None
    return peft


def _patch_soup_setup():
    if _patched["setup"]:
        return
    from soup_cli.trainer import sft as soup_sft

    cls = soup_sft.SFTTrainerWrapper
    original = cls._setup_transformers

    @functools.wraps(original)
    def _setup_transformers(self, *args, **kwargs):
        peft = _real_peft()
        prep = None if peft is None else peft.prepare_model_for_kbit_training
        if prep is None or getattr(prep, "_esep_original", None) is not None:
            return original(self, *args, **kwargs)

        @functools.wraps(prep)
        def prepare_model_for_kbit_training(model, *a, **k):
            return _prepare_keep_fp16_tied(prep, model, *a, **k)

        prepare_model_for_kbit_training._esep_original = prep
        peft.prepare_model_for_kbit_training = prepare_model_for_kbit_training
        try:
            return original(self, *args, **kwargs)
        finally:
            if peft.prepare_model_for_kbit_training is prepare_model_for_kbit_training:
                peft.prepare_model_for_kbit_training = prep

    _setup_transformers._esep_original = original
    cls._setup_transformers = _setup_transformers
    _patched["setup"] = True


def patch():
    """Install the Soup hooks. Model and trainer patches are installed when a trainer is armed."""
    try:
        _patch_soup_train()
        _patch_soup_setup()
    except ImportError:
        pass
