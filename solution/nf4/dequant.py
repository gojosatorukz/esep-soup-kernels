"""NF4 (double-quant) -> fp16 dequantization in one Triton kernel.

bitsandbytes stores a Linear4bit weight W[N, K] as (QLoRA, Dettmers et al. 2023,
https://arxiv.org/abs/2305.14314; layout read from bitsandbytes 0.50.2 `functional.quantize_4bit`
/ `dequantize_4bit` and its CUDA backend, not copied):

- `packed`: uint8, N*K/2 bytes over the row-major flattened weight. Byte i holds element 2i in
  its HIGH nibble and element 2i+1 in its LOW nibble; a nibble indexes the 16-entry NF4 table
  (`quant_state.code`).
- one fp32 scale per 64 consecutive elements (`blocksize` 64 -> one scale per 32 bytes). With
  double quant (`compress_statistics=True`) those scales are themselves stored as
  `quant_state.absmax` (uint8, an index into the 256-entry dynamic map `state2.code`), with one
  fp32 `state2.absmax` per 256 scales and a scalar fp32 `quant_state.offset` (the mean that was
  subtracted before the nested quantization):
      scale[b] = fp32(state2.code[absmax[b]] * state2.absmax[b // 256]) + offset
      W[e]     = fp16(fp32(nf4[nibble(e)] * scale[e // 64]))

bitsandbytes runs that as two kernels (nested 8-bit dequant to an fp32 scale tensor, plus a torch
add, then the 4-bit dequant); here it is one launch that rebuilds each scale in registers. The
rounding sequence is the same (fp32 mul, fp32 add, fp32 mul, round to fp16), and the kernel is
compiled with `enable_fp_fusion=False` so the mul+add is not contracted into an FMA, which keeps
the result bit-exact with `bitsandbytes.functional.dequantize_4bit`.
"""

import torch
import triton
import triton.language as tl

# 64-element scale blocks handled per program: 64 * 32 bytes = 2 KiB in, 8 KiB of fp16 out.
ROWS = 64
NUM_WARPS = 4


@triton.jit
def nf4_dequant_kernel(packed_ptr, qabs_ptr, abs2_ptr, code2_ptr, offset_ptr, nf4_ptr, out_ptr,
                       n_blocks, ROWS: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)            # scale-block index
    rmask = rows < n_blocks
    # nested scale: dynamic-map code * second-level absmax (one per 256 blocks) + offset
    q = tl.load(qabs_ptr + rows, mask=rmask, other=0).to(tl.int32)
    scale = tl.load(code2_ptr + q) * tl.load(abs2_ptr + rows // 256, mask=rmask, other=0.0)
    scale = scale + tl.load(offset_ptr)
    # 32 packed bytes per block
    cols = tl.arange(0, 32)
    boffs = rows[:, None] * 32 + cols[None, :]
    bmask = rmask[:, None]
    b = tl.load(packed_ptr + boffs, mask=bmask, other=0).to(tl.int32)
    hi = tl.load(nf4_ptr + (b >> 4)) * scale[:, None]               # element 2i
    lo = tl.load(nf4_ptr + (b & 15)) * scale[:, None]               # element 2i+1
    vals = tl.reshape(tl.join(hi, lo), (ROWS, 64))
    ooffs = rows[:, None] * 64 + tl.arange(0, 64)[None, :]
    tl.store(out_ptr + ooffs, vals.to(tl.float16), mask=bmask)


# n_blocks (= N*K/64) of the four NF4 weight shapes the measured run dequantizes (Qwen2.5-1.5B:
# q/o 1536x1536, k/v 256x1536, gate/up 8960x1536, down 1536x8960). All are multiples of 16, so the
# launches share one Triton specialization; warmup() compiles it with one of these values.
TRAIN_N_BLOCKS = (1536 * 1536 // 64, 256 * 1536 // 64, 8960 * 1536 // 64)


def warmup(device=None):
    """Ahead-of-time compile of the dequant kernel variants the training run launches (no launch):
    `nf4_dequant_kernel` and `nf4_dequant_multi_kernel`.

    Same dtypes, 16-byte aligned pointers, n_blocks class (multiple of 16), constexprs and compile
    options (`num_warps`, `enable_fp_fusion=False`) as `dequantize_nf4`. Uses dtype stand-ins
    (no allocation). No-op without CUDA. Safe to call from a worker thread: compiles for `device`.
    """
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda" if device is None else device)
    u8, f32 = torch.uint8, torch.float32
    with torch.cuda.device(device):
        nf4_dequant_kernel.warmup(
            u8, u8, f32, f32, f32, f32, torch.float16, TRAIN_N_BLOCKS[0],
            grid=(1,), ROWS=ROWS, num_warps=NUM_WARPS, enable_fp_fusion=False)
        # the one-launch multi-weight variant (linear4bit_multi / dequant_multi): its integer
        # arguments are not specialized, so these stand-in values cover every weight group
        w = (u8, u8, f32, f32, f32, f32)
        nf4_dequant_multi_kernel.warmup(
            *w, 1, *w, 1, *w, 1, 1, 2, 1, 2, torch.float16,
            grid=(1,), ROWS=ROWS, num_warps=NUM_WARPS, enable_fp_fusion=False)


def supported(quant_state) -> bool:
    """True when `dequantize_nf4` handles this quant state (NF4, double quant, blocksize 64/256)."""
    try:
        return (quant_state.quant_type == "nf4" and quant_state.blocksize == 64
                and quant_state.nested and quant_state.state2.blocksize == 256
                and len(quant_state.shape) == 2
                and (quant_state.shape[0] * quant_state.shape[1]) % 64 == 0
                and quant_state.absmax.is_cuda)
    except AttributeError:
        return False


def dequantize_nf4(packed: torch.Tensor, quant_state, out: torch.Tensor = None) -> torch.Tensor:
    """fp16 W[N, K] from a packed NF4 weight (any view of its N*K/2 bytes) and its quant state.

    Caller guarantees `supported(quant_state)`. Bit-exact with
    `bitsandbytes.functional.dequantize_4bit(...)` for an fp16 quant state, and with its fp32
    result cast to fp16 for an fp32 one.
    """
    shape = quant_state.shape
    if out is None:
        out = torch.empty(shape, dtype=torch.float16, device=packed.device)
    qabs = quant_state.absmax
    n_blocks = qabs.numel()
    state2 = quant_state.state2
    nf4_dequant_kernel[(triton.cdiv(n_blocks, ROWS),)](
        packed, qabs, state2.absmax, state2.code, quant_state.offset, quant_state.code, out,
        n_blocks, ROWS=ROWS, num_warps=NUM_WARPS, enable_fp_fusion=False)
    return out


# ---- several weights in one launch (fused q/k/v and gate/up projections, see linear4bit_multi) ----
# Up to MULTI_MAX weights that share K are dequantized by one launch into one contiguous fp16
# W_cat[sum(N_i), K]: weight i fills rows [N_0 + ... + N_{i-1}, ...). The grid is the per-weight
# grids laid end to end; each program picks its weight from its program id and then runs exactly
# nf4_dequant_kernel's decode on that weight's own buffers (same loads, same fp32 mul/add/mul, same
# fp16 rounding, no FMA contraction), so every weight comes out bit-identical to dequantize_nf4.
MULTI_MAX = 3
# Every integer argument varies between weight groups (block counts, program and output offsets):
# none is specialized, so all groups (and warmup()) share one compiled variant.
_MULTI_INTS = ("nb0", "nb1", "nb2", "P1", "P2", "OB1", "OB2")


@triton.jit(do_not_specialize=_MULTI_INTS)
def nf4_dequant_multi_kernel(pk0, qa0, a20, c20, of0, nf0, nb0,
                             pk1, qa1, a21, c21, of1, nf1, nb1,
                             pk2, qa2, a22, c22, of2, nf2, nb2,
                             P1, P2, OB1, OB2, out_ptr, ROWS: tl.constexpr):
    """Programs [0, P1) decode weight 0, [P1, P2) weight 1, [P2, grid) weight 2. Weight i has
    nb_i scale blocks and starts at scale block OB_i of out (OB_0 = 0)."""
    pid = tl.program_id(0)
    if pid < P1:
        packed_ptr, qabs_ptr, abs2_ptr, code2_ptr, offset_ptr, nf4_ptr = pk0, qa0, a20, c20, of0, nf0
        lp = pid
        n_blocks = nb0
        ob = 0
    elif pid < P2:
        packed_ptr, qabs_ptr, abs2_ptr, code2_ptr, offset_ptr, nf4_ptr = pk1, qa1, a21, c21, of1, nf1
        lp = pid - P1
        n_blocks = nb1
        ob = OB1
    else:
        packed_ptr, qabs_ptr, abs2_ptr, code2_ptr, offset_ptr, nf4_ptr = pk2, qa2, a22, c22, of2, nf2
        lp = pid - P2
        n_blocks = nb2
        ob = OB2
    # from here on: nf4_dequant_kernel, on this weight's block range
    rows = lp * ROWS + tl.arange(0, ROWS)                           # scale block within the weight
    rmask = rows < n_blocks
    q = tl.load(qabs_ptr + rows, mask=rmask, other=0).to(tl.int32)
    scale = tl.load(code2_ptr + q) * tl.load(abs2_ptr + rows // 256, mask=rmask, other=0.0)
    scale = scale + tl.load(offset_ptr)
    cols = tl.arange(0, 32)
    boffs = rows[:, None] * 32 + cols[None, :]
    bmask = rmask[:, None]
    b = tl.load(packed_ptr + boffs, mask=bmask, other=0).to(tl.int32)
    hi = tl.load(nf4_ptr + (b >> 4)) * scale[:, None]               # element 2i
    lo = tl.load(nf4_ptr + (b & 15)) * scale[:, None]               # element 2i+1
    vals = tl.reshape(tl.join(hi, lo), (ROWS, 64))
    ooffs = (ob + rows)[:, None] * 64 + tl.arange(0, 64)[None, :]
    tl.store(out_ptr + ooffs, vals.to(tl.float16), mask=bmask)


def multi_plan(quant_states):
    """Launch geometry for `dequantize_nf4_multi`: (shape of W_cat, grid, the 7 integer args).

    Caller guarantees 2..MULTI_MAX states, all `supported`, all with the same K. Pure Python on
    the states' shapes, so callers may cache it per weight group."""
    nbs = [qs.absmax.numel() for qs in quant_states]
    progs = [triton.cdiv(nb, ROWS) for nb in nbs]
    if len(nbs) == 2:
        nbs.append(0)
        progs.append(0)
    P1 = progs[0]
    P2 = P1 + progs[1]
    ints = (nbs[0], nbs[1], nbs[2], P1, P2, nbs[0], nbs[0] + nbs[1])
    n_total = sum(qs.shape[0] for qs in quant_states)
    return (n_total, quant_states[0].shape[1]), P2 + progs[2], ints


def dequantize_nf4_multi(weights, quant_states, out=None, plan=None):
    """fp16 W_cat = cat([dequantize_nf4(w, qs) for ...], 0) in ONE launch, no intermediate copies.

    weights: 2 or 3 packed NF4 weights (any view of their bytes), quant_states their states;
    caller guarantees each `supported(qs)` and equal K (`qs.shape[1]`). Bit-exact per weight with
    `dequantize_nf4`. `plan` is `multi_plan(quant_states)` (computed when not given)."""
    shape, grid, ints = plan if plan is not None else multi_plan(quant_states)
    if out is None:
        out = torch.empty(shape, dtype=torch.float16, device=weights[0].device)
    args = []
    for w, qs in zip(weights, quant_states):
        args += (w, qs.absmax, qs.state2.absmax, qs.state2.code, qs.offset, qs.code)
    if len(weights) == 2:
        args += args[6:]  # unused third slot (no program selects it): any valid pointers
    nf4_dequant_multi_kernel[(grid,)](
        *args[0:6], ints[0], *args[6:12], ints[1], *args[12:18], ints[2], *ints[3:], out,
        ROWS=ROWS, num_warps=NUM_WARPS, enable_fp_fusion=False)
    return out
