"""Fused LM head (lm_head matmul + cross-entropy + entropy + argmax) in Triton.

See kernels.py for the kernel design and the credited sources of ideas.
"""

from .op import fused_linear_ce, warmup
from .patch import patch

__all__ = ["fused_linear_ce", "patch", "warmup"]
