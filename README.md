# Esep — faster, leaner LLM fine-tuning on a single T4

Custom Triton kernels and memory-aware execution for **Soup 0.77.0 + Qwen2.5-1.5B-Instruct**. Built by a two-person team during Decentrathon 6.0, October 30–31, 2026.

**1.7118× end-to-end speedup · 85.7% lower peak GPU memory · 10/10 public harness checks passed**

[Watch the 2:51 demo](https://youtu.be/fWcVOOLVn7E) · [Original executed Kaggle notebook](https://www.kaggle.com/code/datamata2/esep-safe6-final?scriptVersionId=364278304) · [Portfolio reproduction notebook](notebooks/reproduce.ipynb)

## Measured results

| Metric | Baseline Soup | Esep safe-6 |
|---|---:|---:|
| Complete training process | 86.17 s | **50.34 s** |
| Peak GPU memory | 14,345 MiB | **2,053 MiB** |
| Evaluation loss | 2.04965 | 2.04953 |
| Training steps per run | 64 | 64 |

The unchanged harness measured **base → solution → solution → base** in one Kaggle T4 session. Full-process time includes model loading and JIT compilation. Each process uses one GPU. Baseline repeats: 85.88 / 86.47 s; solution repeats: 49.79 / 50.89 s. Memory is the maximum sampled peak; time is the median of two repeats (equal to their mean).

The loss difference (−0.006%) demonstrates equivalence within the allowed tolerance, not improved model quality. Eight original Triton entry points ran in both solution measurements, accounting for at least 10.7623% of training time from step 3 onward. Saved validation: **237 passed / 1 skipped** solution tests and **79 passed** Soup regression tests.

Evidence: [raw result](evidence/result.json), [raw checks](evidence/checks.json), [source manifest](evidence/source-manifest.json). Numbers are from the original saved run, not a new execution of this portfolio copy. No contest placement is claimed.

## Engineering approach

| Bottleneck | Implementation | Effect on execution |
|---|---|---|
| Full-vocabulary logits and metrics | [Chunked head](solution/head/op.py), [Triton row softmax](solution/head/kernels.py) | Reuses a 256-row, 74 MiB logits buffer while preserving loss, entropy, accuracy and per-row rewind outputs. |
| Padded tokens and retained decoder activations | [Within-batch unpadding](solution/body/packed.py), [RMSNorm/RoPE](solution/body/norm_rope.py), [LoRA dropout](solution/body/lora.py) | Retains original row/batch boundaries; fuses operations and recomputes selected activations in backward. |
| NF4 weight materialization | [Dequantization kernels](solution/nf4/dequant.py), [autograd integration](solution/nf4/autograd.py) | Decodes nested scales and weights together, groups supported projections and reconstructs weights for backward. |
| Process startup and compilation | [Warmup](solution/warmup.py), [startup patches](solution/startup.py) | Overlaps compilation with loading inside the measured process. No carried precompiled cache. |

Matrix multiplication remains in cuBLAS and attention in PyTorch SDPA. The reported gain belongs to the complete safe-6 implementation; individual patch contributions were not isolated. The starter SwiGLU example is retained and explicitly excluded from our original-kernel contribution.

## Reproduce

Requires **Linux + NVIDIA CUDA**, Internet, and sufficient GPU memory for the baseline. Official measurements use a **Kaggle Tesla T4**, Python 3.13, PyTorch 2.11 / CUDA 12.8, Triton 3.6, fp16. macOS CPU execution does not reproduce the benchmark.

### Kaggle

1. Download [notebooks/reproduce.ipynb](notebooks/reproduce.ipynb) and import it into Kaggle.
2. Select **GPU T4 x2**, enable **Internet**, and start a fresh session. Kaggle may require account phone verification; training uses only GPU 0.
3. Choose **Save Version → Save & Run All**. The public repository clone needs no GitHub token. Outputs are saved in `/kaggle/working/esep-soup-kernels/runs/`.

**Validation status:** the runtime source files match the measured safe-6 bytes. The portfolio notebook changes repository access and packaging; it has been syntax-checked but has not received a fresh GPU validation. For executed historical evidence, use the saved notebook linked above. That historical notebook still points at the original organizer repository.

### Linux / CUDA terminal

```bash
git clone https://github.com/gojosatorukz/esep-soup-kernels.git
cd esep-soup-kernels
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_source.py
python task/prepare_data.py
python -m pytest solution -q
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
python harness/run_task.py --repeats 2
python harness/check_rules.py --result runs/result.json
python harness/score.py runs/result.json
```

Keep other GPU workloads stopped during measurement. The harness downloads the public dataset and model. No dataset, model weights, trained adapter or compiled cache is bundled.

## Scope and limitations

- Fixed workload: Qwen2.5-1.5B-Instruct, NF4 double quantization, fp16, LoRA r=16 / alpha=32 / dropout=0.05 on q/v projections, batch 4, sequence length ≤512, 256 Kazakh instructions, 64 steps.
- Measurements cover one public training/evaluation split and seed 1, with two repetitions per side. Hidden-row results and other GPUs are unverified.
- Performance and memory depend on sequence lengths, versions, GPU clocks and numerical tolerances. Specialization and guarded fallbacks are intentional.
- Recomputing activations trades compute for memory. Separate ablation measurements were not completed.
- This is a post-hackathon portfolio snapshot. It does not change the frozen contest submission.

## Team and attribution

**Bekzat Rashiduly** — team captain; problem/solution framing, demo production, presentation and pitch.  
**Baltabay Zhanybek** — ML/AI engineering; kernel implementation, optimization and GPU validation.

Runtime implementation is preserved from safe-6 commit `389940d4710277e15444de1d89d1d35bbbe41403`. Portfolio documentation and public-access notebook were prepared after the event.

The [Soup Kernel Challenge starter kit](https://github.com/MakazhanAlpamys/soup-kernel-challenge) provides the harness, fixed task, Soup regression tests and example SwiGLU kernel. Source bytes in `solution/`, `harness/`, `tests/` and `task/` have not been edited during the portfolio migration. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for upstream attribution and data/model licensing.

Sources of implementation ideas and dependency boundaries are documented in [technical notes](docs/technical-notes.md).
