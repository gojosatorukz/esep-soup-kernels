"""Startup-time patches: work Soup does before and around training that does not change any output.

`patch()` installs four wrappers, each idempotent and each a plain pass-through when its
preconditions do not hold. Nothing here touches the model, the data, the batches or their order.

1. `datasets` fingerprints of in-memory datasets (no cache files) are random instead of a hash of
   the transform.
2. `gc.freeze()` at the start of `transformers.Trainer.train`.
3. bf16 -> fp16 weight casts at load run on the GPU instead of on the CPU.
4. `importlib.metadata.packages_distributions()` is computed once (transformers already did it
   before `apply()`), so trl's import does not walk every installed distribution again.

Idea sources (details in each wrapper's docstring): 1. datasets 5.1's own random-fingerprint
fallback (datasets/fingerprint.py, `update_fingerprint` -> `generate_random_fingerprint`).
2. CPython's `gc.freeze` (3.7+; documented as moving all tracked objects out of every future
collection), with the `gc.collect` that accelerate 1.14 already runs (accelerate/utils/memory.py,
`release_memory` -> `clear_device_cache`, via `Accelerator.free_memory`). 3. transformers 5.16's
`_materialize_copy` (transformers/core_model_loading.py). 4. `PACKAGE_DISTRIBUTION_MAPPING` in
transformers/utils/import_utils.py and trl/import_utils.py. Original here: each wrapper and the
choice of when it applies. No code from these libraries/Unsloth/other projects is copied.
"""

import functools
import gc
import sys

_MARK = "_soup_startup_patched"


def _is_patched(fn):
    return getattr(fn, _MARK, False)


def _mark(fn):
    setattr(fn, _MARK, True)
    return fn


# --------------------------------------------------------------------------------------------
# 1. datasets fingerprint
# --------------------------------------------------------------------------------------------

def _caller_dataset(frame):
    """The `datasets.Dataset` that the transform calling `update_fingerprint` works on, or None.

    `Dataset.map` calls it with the dataset as `self`; the `fingerprint_transform` decorator
    (filter, select, remove_columns, ...) with the dataset as the local `dataset`. Any other
    caller (e.g. `_concatenate_map_style_datasets`) gives None and keeps the original hashing.
    """
    try:
        from datasets import Dataset
    except Exception:
        return None
    if frame is None:
        return None
    loc = frame.f_locals
    for name in ("self", "dataset"):
        obj = loc.get(name)
        if isinstance(obj, Dataset):
            return obj
    return None


def _make_update_fingerprint(orig, random_fp):
    @functools.wraps(orig)
    def update_fingerprint(fingerprint, transform, transform_args):
        """Random fingerprint for a transform of a dataset that has no cache files.

        The fingerprint of an in-memory dataset only names it (and the in-memory result of the
        transform); with no cache files there is no cache file to find by it. Hashing the
        transform means pickling it with dill -- for Soup's tokenizing closures that is the
        whole tokenizer, and it is done for every map/filter call. `datasets` itself already
        falls back to `generate_random_fingerprint()` when hashing the transform fails
        (datasets/fingerprint.py, `update_fingerprint`), so a random fingerprint is a state the
        library supports. The data, the rows and their order are unchanged: only the name of
        the dataset differs. Soup's rewind log fingerprints the dataset by its own sha256 of
        the rows (soup_cli/monitoring/rewind_log.py, `dataset_fingerprint`), not by this value.

        A dataset with cache files (or a caller we do not recognize) keeps the original hash,
        so on-disk caching works as before.
        """
        try:
            ds = _caller_dataset(sys._getframe(1))
        except Exception:
            ds = None
        if ds is not None and not ds.cache_files:
            return random_fp()
        return orig(fingerprint, transform, transform_args)

    return _mark(update_fingerprint)


def _patch_fingerprint():
    try:
        import datasets.arrow_dataset as ad
        import datasets.fingerprint as fp
    except Exception:
        return
    orig = getattr(fp, "update_fingerprint", None)
    random_fp = getattr(fp, "generate_random_fingerprint", None)
    if not callable(orig) or not callable(random_fp):
        return
    if _is_patched(orig):
        wrapped = orig
    else:
        wrapped = _make_update_fingerprint(orig, random_fp)
        # `fingerprint_transform` looks the name up in datasets.fingerprint's globals;
        # `Dataset.map` and concatenation look it up in datasets.arrow_dataset's globals.
        fp.update_fingerprint = wrapped
    if getattr(ad, "update_fingerprint", None) is orig:
        ad.update_fingerprint = wrapped


# --------------------------------------------------------------------------------------------
# 2. gc.freeze at the start of Trainer.train
# --------------------------------------------------------------------------------------------

def _make_train(orig):
    @functools.wraps(orig)
    def train(self, *args, **kwargs):
        """Runtime GC tuning: long-lived model/tokenizer objects are moved to the permanent
        generation so cyclic GC does not rescan them during training.

        No `gc.collect()` before the freeze: accelerate already runs one right after this, at
        the start of the training loop (`Accelerator.free_memory` -> `clear_device_cache`,
        accelerate/utils/memory.py), and after the freeze that collection scans only the young
        objects. Measured locally (24 steps): an extra collect here costs 0.2-0.35 s of gen-2
        GC, freeze-only 0.03 s of GC in the whole run, same peak CUDA memory.
        """
        try:
            gc.freeze()
        except Exception:
            pass
        return orig(self, *args, **kwargs)

    return _mark(train)


def _patch_gc_freeze():
    try:
        import transformers
        trainer_cls = transformers.Trainer
    except Exception:
        return
    orig = trainer_cls.__dict__.get("train")
    if not callable(orig) or _is_patched(orig):
        return
    trainer_cls.train = _make_train(orig)


# --------------------------------------------------------------------------------------------
# 3. GPU-side dtype cast at load
# --------------------------------------------------------------------------------------------

def _make_materialize_copy(orig):
    import torch

    @functools.wraps(orig)
    def _materialize_copy(tensor, device=None, dtype=None):
        """Copy to the GPU in the checkpoint dtype, then cast there.

        `tensor.to(device=cuda, dtype=fp16)` on a CPU bf16 tensor casts on the CPU first;
        `.to(cuda).to(fp16)` moves the bf16 bytes and casts on the GPU. bf16 -> fp16 is one
        IEEE rounding either way, so the loaded values are bit-identical.
        """
        if dtype is None or device is None:
            return orig(tensor, device, dtype)
        try:
            is_cuda = torch.device(device).type == "cuda"
        except Exception:
            is_cuda = False
        if not is_cuda:
            return orig(tensor, device, dtype)
        # This slicing is what loads the tensor from the safetensors slice object.
        t = tensor[...]
        if not isinstance(t, torch.Tensor) or t.device.type != "cpu" or t.dtype == dtype:
            return t.to(device=device, dtype=dtype)
        return t.to(device=device).to(dtype)

    return _mark(_materialize_copy)


def _patch_materialize_copy():
    try:
        import transformers.core_model_loading as cml
    except Exception:
        return
    orig = getattr(cml, "_materialize_copy", None)
    if not callable(orig) or _is_patched(orig):
        return
    # spawn_materialize calls it by module-global name, so the module attribute is enough.
    cml._materialize_copy = _make_materialize_copy(orig)


# --------------------------------------------------------------------------------------------
# 4. memoized importlib.metadata.packages_distributions
# --------------------------------------------------------------------------------------------

def _copy_mapping(mapping):
    return {k: list(v) for k, v in mapping.items()}


def _transformers_mapping():
    """transformers' own `packages_distributions()` result, if it is already imported, else None."""
    mod = sys.modules.get("transformers.utils.import_utils")
    mapping = getattr(mod, "PACKAGE_DISTRIBUTION_MAPPING", None) if mod is not None else None
    return mapping if isinstance(mapping, dict) else None


def _make_packages_distributions(orig, seed=None):
    memo = [None if seed is None else _copy_mapping(seed)]

    @functools.wraps(orig)
    def packages_distributions():
        """Memoized `importlib.metadata.packages_distributions()`.

        `packages_distributions()` is a pure function of the installed distributions: it walks
        every distribution's files/top_level.txt (about 2.6 s for Kaggle's 924 distributions).
        No packages are installed during the run (a rule of the task, checked by
        harness/check_rules.py check 2), so the memoized mapping is the one a new call would
        compute. transformers computed it at import time, before `apply()`
        (transformers/utils/import_utils.py, `PACKAGE_DISTRIBUTION_MAPPING`), so the memo starts
        from that; trl recomputes it when it is imported (trl/import_utils.py), after `apply()`.

        Each call returns a new dict with new lists, so a caller that mutates its result
        cannot change what the next caller gets.
        """
        if memo[0] is None:
            memo[0] = _copy_mapping(orig())
        return _copy_mapping(memo[0])

    return _mark(packages_distributions)


def _patch_packages_distributions():
    import importlib.metadata as md
    orig = getattr(md, "packages_distributions", None)
    if not callable(orig) or _is_patched(orig):
        return
    # The harness puts the repo on sys.path after transformers' scan; memoize only when the repo
    # holds no distributions, so a fresh scan would give the same mapping.
    repo = __file__.rsplit("/solution/", 1)[0]
    if any(True for _ in md.distributions(path=[repo])):
        return
    # trl calls it through the module (`importlib.metadata.packages_distributions()`); a name
    # imported with `from importlib.metadata import ...` before apply() keeps the original.
    md.packages_distributions = _make_packages_distributions(orig, _transformers_mapping())


def patch():
    for fn in (_patch_fingerprint, _patch_gc_freeze, _patch_materialize_copy,
               _patch_packages_distributions):
        try:
            fn()
        except Exception:
            pass
