"""Startup patches (solution/startup.py): same outputs as the unpatched path."""

import gc
import importlib.metadata
import sys

import pytest
import torch

from solution import startup


def _unwrap(fn):
    while startup._is_patched(fn):
        fn = fn.__wrapped__
    return fn


@pytest.fixture(autouse=True)
def _unpatched():
    """Run every test against the unpatched libraries, then put back whatever was installed.

    With SOUP_APPLY_SOLUTION=1, tests/conftest.py calls `solution.apply()` when it is imported --
    during collection, before any test here runs (`pytest solution tests` or the other order) --
    so the startup patches may already be in place. The tests below compare against the
    original functions, so they must start from them.
    """
    targets = [(importlib.metadata, "packages_distributions",
                importlib.metadata.packages_distributions)]  # (owner, name, current value)
    try:
        import datasets.arrow_dataset as ad
        import datasets.fingerprint as fp
        for mod in (fp, ad):
            targets.append((mod, "update_fingerprint", mod.update_fingerprint))
    except Exception:
        pass
    try:
        import transformers.core_model_loading as cml
        targets.append((cml, "_materialize_copy", cml._materialize_copy))
    except Exception:
        pass
    try:
        import transformers
        targets.append((transformers.Trainer, "train", transformers.Trainer.__dict__["train"]))
    except Exception:
        pass
    for obj, name, cur in targets:
        setattr(obj, name, _unwrap(cur))
    try:
        yield
    finally:
        for obj, name, cur in targets:
            setattr(obj, name, cur)


def test_unpatched_fixture_restores_originals():
    import datasets.arrow_dataset as ad
    import datasets.fingerprint as fp
    import transformers
    assert not startup._is_patched(fp.update_fingerprint)
    assert not startup._is_patched(ad.update_fingerprint)
    assert not startup._is_patched(transformers.Trainer.__dict__["train"])
    assert not startup._is_patched(importlib.metadata.packages_distributions)


@pytest.fixture
def fp_patched():
    """Install the fingerprint patch, restore both module names afterwards."""
    ad = pytest.importorskip("datasets.arrow_dataset")
    import datasets.fingerprint as fp
    saved = fp.update_fingerprint, ad.update_fingerprint
    startup._patch_fingerprint()
    startup._patch_fingerprint()  # idempotent
    yield fp, ad, saved[0]
    fp.update_fingerprint, ad.update_fingerprint = saved


def _tok(batch):
    return {"ids": [[len(t), len(t) * 2] for t in batch["text"]]}


def _keep(r):
    return r["n"] % 3 != 0


def _rows():
    from datasets import Dataset
    return Dataset.from_dict({"text": ["a", "bb", "ccc", "dddd"] * 8, "n": list(range(32))})


def test_fingerprint_patch_installed_once(fp_patched):
    fp, ad, orig = fp_patched
    assert ad.update_fingerprint is fp.update_fingerprint
    assert startup._is_patched(fp.update_fingerprint)
    assert fp.update_fingerprint.__wrapped__ is orig


def test_in_memory_map_gets_random_fingerprint_and_same_data():
    pytest.importorskip("datasets")
    ref_ds = _rows()
    ref = ref_ds.map(_tok, batched=True).filter(_keep).remove_columns("n")
    ref2 = _rows().map(_tok, batched=True).filter(_keep).remove_columns("n")
    assert ref._fingerprint == ref2._fingerprint  # unpatched: deterministic hash

    import datasets.arrow_dataset as ad
    import datasets.fingerprint as fp
    saved = fp.update_fingerprint, ad.update_fingerprint
    calls = {"orig": 0, "random": 0}
    orig, orig_rand = fp.update_fingerprint, fp.generate_random_fingerprint

    def counting_orig(*a):
        calls["orig"] += 1
        return orig(*a)

    def counting_rand():
        calls["random"] += 1
        return orig_rand()

    w = startup._make_update_fingerprint(counting_orig, counting_rand)
    fp.update_fingerprint = ad.update_fingerprint = w
    try:
        ds = _rows()
        assert not ds.cache_files
        out = ds.map(_tok, batched=True).filter(_keep).remove_columns("n")
        out2 = _rows().map(_tok, batched=True).filter(_keep).remove_columns("n")
    finally:
        fp.update_fingerprint, ad.update_fingerprint = saved
    # map (arrow_dataset name), filter and remove_columns (fingerprint_transform, fingerprint name)
    assert calls["orig"] == 0 and calls["random"] >= 6
    assert out._fingerprint != out2._fingerprint != ref._fingerprint
    assert out.to_list() == ref.to_list()
    assert out.features == ref.features


def test_dataset_with_cache_files_keeps_original_hash(tmp_path):
    datasets = pytest.importorskip("datasets")
    _rows().save_to_disk(str(tmp_path / "d"))
    ds = datasets.load_from_disk(str(tmp_path / "d"))
    assert ds.cache_files
    calls = []
    import datasets.fingerprint as fp
    orig = fp.update_fingerprint

    def counting_orig(*a):
        calls.append(1)
        return orig(*a)

    w = startup._make_update_fingerprint(counting_orig, lambda: pytest.fail("random fingerprint used"))
    t = fp.format_transform_for_fingerprint(datasets.Dataset._map_single)
    # called from a frame whose `dataset` has cache files -> original
    dataset = ds  # noqa: F841 (read by the wrapper through the caller frame)
    got = w(ds._fingerprint, t, {"x": 1})
    assert calls and got == orig(ds._fingerprint, t, {"x": 1})
    # unknown caller (no dataset in the frame) -> original
    assert _call_without_dataset(w, ds._fingerprint, t) == got


def _call_without_dataset(w, f, t):
    return w(f, t, {"x": 1})


def test_gc_wrapper_freezes_and_returns_result():
    sentinel = object()

    class T:
        def train(self, *a, **k):
            return (sentinel, a, k, gc.get_freeze_count())

    T.train = startup._make_train(T.train)
    assert startup._is_patched(T.train)
    try:
        r = T().train(1, resume_from_checkpoint=None)
    finally:
        gc.unfreeze()
    assert r[0] is sentinel and r[1] == (1,) and r[2] == {"resume_from_checkpoint": None}
    assert r[3] > 0


def test_gc_patch_idempotent_on_trainer():
    transformers = pytest.importorskip("transformers")
    saved = transformers.Trainer.__dict__["train"]
    try:
        startup._patch_gc_freeze()
        once = transformers.Trainer.__dict__["train"]
        startup._patch_gc_freeze()
        assert transformers.Trainer.__dict__["train"] is once
        assert once.__wrapped__ is saved
    finally:
        transformers.Trainer.train = saved


def test_materialize_copy_cpu_paths_unchanged():
    cml = pytest.importorskip("transformers.core_model_loading")
    orig = cml._materialize_copy
    w = startup._make_materialize_copy(orig)
    x = torch.randn(64, 32).to(torch.bfloat16)
    for dev, dt in [(None, None), ("cpu", torch.float16), (None, torch.float16), ("cpu", None)]:
        a, b = orig(x, dev, dt), w(x, dev, dt)
        assert a.dtype == b.dtype and a.device == b.device and torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("src", [torch.bfloat16, torch.float32])
def test_materialize_copy_gpu_cast_bit_identical(src):
    cml = pytest.importorskip("transformers.core_model_loading")
    orig = cml._materialize_copy
    w = startup._make_materialize_copy(orig)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1536, 1536, generator=g) * 3
    x[0, :8] = torch.tensor([0.0, -0.0, 1e-8, -1e-8, 7e4, -7e4, 1e-5, 65504.0])  # subnormal, overflow
    x = x.to(src)
    for dt in (torch.float16, torch.float32, src):
        a, b = orig(x, "cuda", dt), w(x, torch.device("cuda:0"), dt)
        assert a.dtype == b.dtype == dt and b.device.type == "cuda"
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_materialize_copy_real_checkpoint_bit_identical():
    """Every tensor of the cached Qwen2.5-1.5B checkpoint, bf16 -> fp16 on CUDA."""
    cml = pytest.importorskip("transformers.core_model_loading")
    safetensors = pytest.importorskip("safetensors")
    try:
        from huggingface_hub import try_to_load_from_cache
        path = try_to_load_from_cache("Qwen/Qwen2.5-1.5B-Instruct", "model.safetensors")
    except Exception:
        path = None
    if not isinstance(path, str):
        pytest.skip("checkpoint not cached")
    orig = cml._materialize_copy
    w = startup._make_materialize_copy(orig)
    n = diff = 0
    with safetensors.safe_open(path, framework="pt") as f:
        for k in f.keys():
            a = orig(f.get_slice(k), "cuda", torch.float16)
            b = w(f.get_slice(k), "cuda", torch.float16)
            diff += int((a.view(torch.int16) != b.view(torch.int16)).sum())
            n += 1
    assert n > 300 and diff == 0


def test_packages_distributions_memo_equal_and_copied():
    orig = importlib.metadata.packages_distributions
    calls = []

    def counting():
        calls.append(1)
        return orig()

    w = startup._make_packages_distributions(counting)
    assert startup._is_patched(w) and w.__wrapped__ is counting
    ref = orig()
    first = w()
    assert first == ref and len(calls) == 1
    first["__soup_test__"] = ["x"]
    next(iter(v for v in first.values() if v)).append("__soup_test__")
    second = w()
    assert second == ref and second is not first and len(calls) == 1
    assert all(second[k] is not first[k] for k in second)


def test_packages_distributions_seeded_from_transformers():
    iu = pytest.importorskip("transformers.utils.import_utils")
    assert sys.modules["transformers.utils.import_utils"] is iu
    seed = startup._transformers_mapping()
    assert seed is iu.PACKAGE_DISTRIBUTION_MAPPING
    w = startup._make_packages_distributions(lambda: pytest.fail("original called"), seed)
    got = w()
    assert got == importlib.metadata.packages_distributions() == seed
    got.clear()
    assert w() == seed and iu.PACKAGE_DISTRIBUTION_MAPPING  # neither memo nor seed mutated


def test_packages_distributions_patch_idempotent(monkeypatch):
    orig = importlib.metadata.packages_distributions
    monkeypatch.setattr(startup, "_transformers_mapping", lambda: None)
    startup._patch_packages_distributions()
    once = importlib.metadata.packages_distributions
    startup._patch_packages_distributions()
    assert importlib.metadata.packages_distributions is once
    assert startup._is_patched(once) and once.__wrapped__ is orig
    assert once() == orig()  # the autouse fixture restores the module attribute


def test_patch_idempotent_and_quiet():
    import transformers
    import transformers.core_model_loading as cml
    import datasets.arrow_dataset as ad
    import datasets.fingerprint as fp
    md = importlib.metadata
    saved = (transformers.Trainer.__dict__["train"], cml._materialize_copy,
             fp.update_fingerprint, ad.update_fingerprint, md.packages_distributions)
    try:
        startup.patch()
        first = (transformers.Trainer.__dict__["train"], cml._materialize_copy,
                 fp.update_fingerprint, ad.update_fingerprint, md.packages_distributions)
        startup.patch()
        second = (transformers.Trainer.__dict__["train"], cml._materialize_copy,
                  fp.update_fingerprint, ad.update_fingerprint, md.packages_distributions)
        assert first == second and all(startup._is_patched(f) for f in first)
    finally:
        transformers.Trainer.train, cml._materialize_copy = saved[0], saved[1]
        fp.update_fingerprint, ad.update_fingerprint = saved[2], saved[3]
        md.packages_distributions = saved[4]
