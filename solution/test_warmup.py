"""Background Triton warmup: compiles exactly what training launches, never breaks training."""

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from solution import warmup as W

REPO = Path(__file__).resolve().parents[1]
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# Runs in a fresh process with an empty Triton cache: start() the thread as apply() does, join,
# check the variant is also loaded (launcher built, cubin loaded), then dequantize the four real
# weight shapes and count compiles of our kernel on each side.
_SCRIPT = textwrap.dedent("""
    import sys, threading
    import torch, triton
    torch.empty(1, device="cuda")  # the harness starts CUDA before apply()
    import bitsandbytes.functional as BF
    from solution import warmup
    from solution.nf4 import dequantize_nf4

    compiles = []
    def hook(*, key, fn, **kw):
        compiles.append((fn.name, threading.current_thread().name))
    triton.knobs.runtime.jit_post_compile_hook = hook

    warmup.start()
    warmup.start()  # idempotent
    assert warmup._thread is not None and warmup._thread.daemon
    assert warmup.join()
    before = list(compiles)
    from solution.nf4 import nf4_dequant_kernel
    dev = torch.cuda.current_device()
    ks = list(nf4_dequant_kernel.device_caches[dev][0].values())
    assert len(ks) == 1 and ks[0]._run is not None and ks[0].module is not None, "not loaded"
    for n, k in [(1536, 1536), (256, 1536), (8960, 1536), (1536, 8960)]:
        w = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.02
        packed, qs = BF.quantize_4bit(w, quant_type="nf4", compress_statistics=True, blocksize=64)
        assert torch.equal(dequantize_nf4(packed, qs), BF.dequantize_4bit(packed, qs))
    torch.cuda.synchronize()
    after = compiles[len(before):]
    print("BEFORE", before)
    print("AFTER", after)
    assert [c for c in before if c[0] == "nf4_dequant_kernel"] == [
        ("nf4_dequant_kernel", "solution-triton-warmup")], before
    assert after == [], after
""")


@needs_cuda
def test_nf4_warmup_covers_real_launches(tmp_path):
    pytest.importorskip("bitsandbytes")
    env = dict(os.environ, TRITON_CACHE_DIR=str(tmp_path / "triton"), PYTHONPATH=str(REPO))
    r = subprocess.run([sys.executable, "-c", _SCRIPT], cwd=REPO, env=env, capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[solution.warmup] nf4 " in r.stderr


def test_failing_warmup_is_logged_not_raised(monkeypatch, capsys):
    calls = []

    def bad(device):
        raise RuntimeError("boom")

    def good(device):
        calls.append(device)

    device = torch.device("cuda", 0) if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        pytest.skip("set_device needs CUDA")
    monkeypatch.setattr(W, "_notes", [])
    W._run(device, [("bad", bad), ("good", good)])  # must not raise
    assert capsys.readouterr().err == ""  # the thread prints nothing
    assert calls == [device]  # a failure does not stop the next workstream
    note = " ".join(W._notes)
    assert "bad failed" in note and "(RuntimeError)" in note and "good " in note
    assert "boom" not in note  # type names only, never messages
    W._flush()
    err = capsys.readouterr().err
    assert err.startswith("[solution.warmup] ") and "bad FAILED" in err and "RuntimeERR" in err
    assert not any(w in err for w in ("error", " failed", " passed")), err
    assert W._notes == []


def test_join_without_thread_is_noop(monkeypatch):
    monkeypatch.setattr(W, "_thread", None)
    assert W.join(0.01) is True


def test_join_hook_waits_then_calls_through(monkeypatch):
    transformers = pytest.importorskip("transformers")
    order = []

    class FakeThread:
        alive = True

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            order.append("join")
            self.alive = False

    def orig(self, *a, **k):
        order.append("step")
        return 1.0

    monkeypatch.setattr(transformers.Trainer, "training_step", orig)
    monkeypatch.setattr(W, "_thread", FakeThread())
    W._install_join_hook()
    W._install_join_hook()  # idempotent: no double wrap
    hooked = transformers.Trainer.training_step
    assert getattr(hooked, W._MARK) and hooked.__wrapped__ is orig
    assert hooked(None) == 1.0 and hooked(None) == 1.0
    assert order == ["join", "step", "step"]


# What tests/conftest.py does under SOUP_APPLY_SOLUTION=1: apply() (here twice) and nothing else,
# then exit while the thread may still be compiling: the atexit join must let it finish cleanly.
_APPLY_ONLY = textwrap.dedent("""
    import atexit, threading
    import torch
    if torch.cuda.is_available():
        torch.empty(1, device="cuda")
    import solution
    from solution import warmup
    registered = []
    _register = atexit.register
    def register(fn, *a, **k):
        registered.append(fn)
        return _register(fn, *a, **k)
    atexit.register = register
    solution.apply()
    t = warmup._thread
    solution.apply()
    assert warmup._thread is t, "started twice"
    if torch.cuda.is_available():
        assert t is not None and t.daemon and t.name == "solution-triton-warmup"
        assert registered.count(warmup._join_at_exit) == 1, registered
    print("APPLY_ONLY_OK", t is not None and t.is_alive())
""")


def test_apply_starts_thread_and_exits_cleanly(tmp_path):
    env = dict(os.environ, TRITON_CACHE_DIR=str(tmp_path / "triton"), PYTHONPATH=str(REPO))
    r = subprocess.run([sys.executable, "-c", _APPLY_ONLY], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=300)
    print("STDERR:", repr(r.stderr))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "APPLY_ONLY_OK" in r.stdout
    for bad in ("Traceback", "Fatal", "Exception", "error", " failed", " passed",
                "[solution.warmup]"):  # outside training, warmup prints nothing
        assert bad not in r.stderr, r.stderr


@needs_cuda
def test_start_registers_exit_join(monkeypatch):
    registered, ran = [], []
    monkeypatch.setattr(W, "_thread", None)
    monkeypatch.setattr(W, "_targets", lambda: [("x", lambda device: None)])
    monkeypatch.setattr(W, "_run", lambda device, targets: ran.append(targets[0][0]))
    monkeypatch.setattr(W, "_install_join_hook", lambda: None)
    monkeypatch.setattr(W.atexit, "register", registered.append)

    W.start()
    t = W._thread
    assert t is not None and t.name == "solution-triton-warmup" and t.daemon
    t.join(10)
    assert ran == ["x"] and registered == [W._join_at_exit]
    W.start()
    assert W._thread is t and registered == [W._join_at_exit]  # started once


def test_exit_join_is_silent_and_never_raises(monkeypatch, capsys):
    class Stuck:
        def is_alive(self):
            return True

        def join(self, timeout=None):
            self.timeout = timeout

    stuck = Stuck()
    monkeypatch.setattr(W, "_thread", stuck)
    W._join_at_exit()
    assert stuck.timeout == W.EXIT_JOIN_TIMEOUT_S == 120.0

    class Broken:
        def is_alive(self):
            raise KeyboardInterrupt

    monkeypatch.setattr(W, "_thread", Broken())
    W._join_at_exit()  # swallowed
    monkeypatch.setattr(W, "_thread", None)
    W._join_at_exit()  # no thread: no-op
    assert capsys.readouterr() == ("", "")


def _check10_last_line(stdout, stderr):
    """harness/check_rules.py run_tests(): the line check 10 reports."""
    lines = [line for line in (stdout + "\n" + stderr).splitlines() if line.strip()]
    summary = [line for line in lines if " passed" in line or " failed" in line or "error" in line]
    return (summary or lines or ["no output"])[-1]


@pytest.mark.skipif(os.environ.get("SOUP_APPLY_SOLUTION") == "1", reason="no recursion")
def test_soup_tests_check10_summary_is_pytests():
    """Check 10 as the harness runs it: our thread and exit join must not change its result."""
    env = dict(os.environ, SOUP_APPLY_SOLUTION="1")
    r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"],
                       cwd=REPO, env=env, capture_output=True, text=True, timeout=1800)
    last = _check10_last_line(r.stdout, r.stderr)
    print("LAST:", last, "STDERR:", repr(r.stderr))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr
    assert re.search(r"(?<!\d)79 passed\b", last) and " failed" not in last, last
    assert "[solution.warmup]" not in r.stderr, r.stderr
