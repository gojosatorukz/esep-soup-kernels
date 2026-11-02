"""Ahead-of-time compile of every Triton kernel variant the body path launches.

`warmup(device)` calls `JITFunction.warmup(...)` (compile only, no launch) once per distinct
specialization that the training run hits, so step 0 finds every kernel in the in-process cache.
It is meant to run while the model loads (the nf4 workstream starts it from a thread in apply()).

How Triton 3.6 keys a compiled kernel (runtime/jit.py, create_function_from_signature, and
backends/compiler.py): per argument
  * tensor: element dtype and whether data_ptr % 16 == 0 (all our launches pass fresh tensors or
    whole-row views of 16-byte multiple rows, so always aligned; the dummies here are fresh);
  * int not in do_not_specialize: i32/i64 by magnitude, value 1 -> constexpr 1, else whether
    value % 16 == 0; in do_not_specialize: only i32/i64;
  * float: fp32, never specialized (eps, p, inv_keep, sin_sign need no care);
  * bool: u1; None: constexpr None;
  * constexpr values; and the launch options (num_warps, num_stages, ...).
No int argument of our kernels changes per step except the element counts of the LoRA kernels
(do_not_specialize there). n_cols (1536), the RoPE strides (1536/128/256 or the T*128 strides of
the backward's strided grads), head counts 12/2 and the example SwiGLU's n = T * 8960 are
multiples of 16 (never 1) for every T, so one compile per variant serves every step.

The variants (Qwen2.5-1.5B: H = 1536, 12 q heads, 2 kv heads, head_dim 128, intermediate 8960):
  add_rmsnorm fwd  HAS_DELTA=0, y fp16    layer 0 input norm (delta None; delta_ptr = r_in fp32)
                                          and (NORM_SAVE_OUTPUT off) the MLP backward's
                                          rmsnorm_recompute
  add_rmsnorm fwd  HAS_DELTA=1 fp16, y fp16   every other decoder norm
  add_rmsnorm fwd  HAS_DELTA=1 fp16, y fp32   the final norm (HF returns fp32 there)
    (the number of exact channels n_special of the save_output mode is a runtime int, not
    specialized, and its slot / xs pointers are always int32 / fp32 (dummies when unused): the
    same three variants serve both NORM_SAVE_OUTPUT settings and every K)
  add_rmsnorm bwd  dy fp16, HAS_DR_OUT=1, ddelta fp16   decoder norms with a delta; the number
                   of summed dy (5 input-norm consumers, 2 post-attention: gate/up, with or
                   without the recompute MLP) is a runtime int, not specialized: one variant.
                   NORM_SAVE_OUTPUT on: FROM_Y=1 with the saved y fp16 as x; off: FROM_Y=0 with
                   the fp32 residual (warmup compiles the one packed.NORM_SAVE_OUTPUT selects)
  add_rmsnorm bwd  dy fp32, HAS_DR_OUT=0, ddelta fp16, FROM_Y=0   final norm: its residual
                   output is unused and arrives as None (set_materialize_grads(False))
  add_rmsnorm bwd  dy fp16, HAS_DR_OUT=1, WRITE_DDELTA=0 (ddelta dummy fp32), FROM_Y=0
                   layer 0's input norm (no delta): it has a backward because the embedding
                   output requires grad (prepare_model_for_kbit_training's
                   enable_input_require_grads), so d_r_in flows to the embedding
  rope fwd  HAS_QD=1 (q LoRA delta fused);  rope bwd  HAS_QD=0
  lora_dropout fwd (forward, and backward when no dx is needed: layer 0);
  lora_dropout bwd (accumulate=0)
  example SwiGLU fwd/bwd (solution/swiglu.py; launched by the recompute MLP, not ours, but its
  compile time counts the same)

Idea source: original design (it only uses Triton's public JITFunction.warmup API); no kernel
code here.
"""

import logging

import torch

log = logging.getLogger(__name__)

# Real-model dims that end up in constexprs (head counts / head_dim) or in specialized ints.
_H, _HQ, _HK, _D, _INTER = 1536, 12, 2, 128, 8960

# Also build each kernel's launcher and load its cubin (see _load). In-process switch for
# measurements only; the run always uses the default.
LOAD_HANDLES = True


def _variants(device):
    """[(name, zero-arg callable)], one per specialization; tensors are made lazily per call."""
    import triton

    from ..swiglu import BLOCK as SWIGLU_BLOCK
    from ..swiglu import example_swiglu_bwd_kernel, example_swiglu_fwd_kernel
    from . import lora as L
    from . import norm_rope as N

    f16, f32, i32, i64 = torch.float16, torch.float32, torch.int32, torch.int64
    from .packed import NORM_SAVE_OUTPUT

    def t(dtype):
        return torch.empty(16, device=device, dtype=dtype)

    def norm_fwd(delta_dtype, has_delta, y_dtype):
        def job():
            r = t(f32)
            delta = t(delta_dtype) if has_delta else r
            return N._add_rmsnorm_fwd.warmup(
                r, delta, t(f32), t(f32), t(y_dtype), t(f32), t(i32), t(f32), _H, 1e-6, 2,
                HAS_DELTA=has_delta, BLOCK=N.NORM_BLOCK, num_warps=N.NORM_WARPS_FWD, grid=(1,))
        return job

    def norm_bwd(dy_dtype, has_dr_out, from_y, write_ddelta=True):
        def job():
            dys = [t(dy_dtype) for _ in range(N.MAX_DY)]
            return N._add_rmsnorm_bwd.warmup(
                *dys, 2, t(f32), t(f16 if from_y else f32), t(f32), t(f32), t(i32), t(f32), 2,
                t(f32), t(f16 if write_ddelta else f32), _H, HAS_DR_OUT=has_dr_out,
                WRITE_DDELTA=write_ddelta,
                FROM_Y=from_y, BLOCK=N.NORM_BLOCK, num_warps=N.NORM_WARPS_BWD, grid=(1,))
        return job

    def rope(has_qd, sign):
        def job():
            q = t(f16)
            qd = t(f16) if has_qd else q
            return N._rope_kernel.warmup(
                q, qd, t(f16), t(f16), t(f16), t(i64), t(f32),
                _H, _D, _H, _D, _HK * _D, _D, _HQ, _HK, sign,
                HAS_QD=has_qd, HQP=triton.next_power_of_2(_HQ),
                HKP=triton.next_power_of_2(_HK), HALF=_D // 2, num_warps=N.ROPE_WARPS, grid=(1,))
        return job

    def drop_fwd():
        return L.lora_dropout_fwd_kernel.warmup(
            t(f16), t(i64), t(f16), 16 * _H, 0.05, 1.0 / 0.95, BLOCK=L.DROP_BLOCK,
            num_warps=L.DROP_WARPS, grid=(1,))

    def drop_bwd():
        g = t(f16)
        return L.lora_dropout_bwd_kernel.warmup(
            t(f16), g, t(i64), t(f16), g, 16 * _H, 0.05, 1.0 / 0.95, 0, BLOCK=L.DROP_BLOCK,
            num_warps=L.DROP_WARPS, grid=(1,))

    def swiglu_fwd():
        g = t(f16)
        return example_swiglu_fwd_kernel.warmup(g, t(f16), g, 16 * _INTER, BLOCK=SWIGLU_BLOCK,
                                                grid=(1,))

    def swiglu_bwd():
        g, u = t(f16), t(f16)
        return example_swiglu_bwd_kernel.warmup(t(f16), g, u, g, u, 16 * _INTER,
                                                BLOCK=SWIGLU_BLOCK, grid=(1,))

    return [
        ("add_rmsnorm_fwd nodelta y16", norm_fwd(f32, False, f16)),
        ("add_rmsnorm_fwd d16 y16", norm_fwd(f16, True, f16)),
        ("add_rmsnorm_fwd d16 y32", norm_fwd(f16, True, f32)),
        ("add_rmsnorm_bwd dy16", norm_bwd(f16, True, NORM_SAVE_OUTPUT)),
        ("add_rmsnorm_bwd dy32 final", norm_bwd(f32, False, False)),
        ("add_rmsnorm_bwd dy16 nodelta", norm_bwd(f16, True, False, write_ddelta=False)),
        ("rope fwd qd", rope(True, 1.0)),
        ("rope bwd", rope(False, -1.0)),
        ("lora_dropout_fwd", drop_fwd),
        ("lora_dropout_bwd", drop_bwd),
        ("example_swiglu_fwd", swiglu_fwd),
        ("example_swiglu_bwd", swiglu_bwd),
    ]


def _load(kernel):
    """Finish what Triton would otherwise do on the first launch, without launching: build the
    kernel's C launcher (a gcc compile per signature, ~0.1-0.3 s each, cached in
    TRITON_CACHE_DIR) and load the cubin into the current CUDA context (cuModuleLoadData).
    CompiledKernel._init_handles is what `CompiledKernel.run` calls lazily on first use."""
    if hasattr(kernel, "result"):  # async compile mode returns a future
        kernel = kernel.result()
    init = getattr(kernel, "_init_handles", None)
    if init is not None:
        init()


def warmup(device):
    """Compile (not launch) every body kernel variant on `device`. Never raises."""
    try:
        device = torch.device(device)
        variants = _variants(device)
        ctx = torch.cuda.device(device)
        ctx.__enter__()
    except Exception as exc:  # noqa: BLE001 - warmup is best effort
        log.debug("solution.body.warmup: setup failed: %r", exc)
        return
    try:
        for name, job in variants:
            try:
                kernel = job()
                if LOAD_HANDLES:
                    _load(kernel)
            except Exception as exc:  # noqa: BLE001 - one failure must not skip the rest
                log.debug("solution.body.warmup: %s failed: %r", name, exc)
    finally:
        try:
            ctx.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass
