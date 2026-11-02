"""Workstream `body`: the Qwen2 decoder layers (pad removal, fused element-wise kernels).

Idea sources are credited in each module: packed.py (padding-free training as in
FlashAttention's variable-length mode and HF DataCollatorWithFlattening; activation
recomputation, Chen et al. 2016), norm_rope.py (RMSNorm, Zhang & Sennrich 2019; RoPE, Su et al. 2021), lora.py (seed-
regenerated dropout, Triton "Low-Memory Dropout" tutorial), warmup.py (original design).
"""


def patch():
    """Install the body patches. Safe to call more than once."""
    from .packed import patch as patch_packed

    patch_packed()


def warmup(device):
    """Compile every Triton kernel variant the body path launches (no launches; never raises).
    Meant to run during model load, e.g. from a background thread. See body/warmup.py."""
    from .warmup import warmup as _warmup

    _warmup(device)
