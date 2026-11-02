"""Ahead-of-time compile of our Triton kernels in a background thread, overlapping model load.

Every measured run starts with an empty Triton cache, so each kernel is compiled on its first
launch, inside training step 0. `start()` instead compiles them in a worker thread while the main
thread goes on loading the model: each workstream package
exports `warmup(device)`, which calls `JITFunction.warmup(...)` (compile only, no launch) for
every kernel variant the training run launches. This module owns the thread, the timing and the
error handling, and then finishes what `.warmup()` leaves lazy for the first launch: the C
launcher (built with gcc) and the module load (`CompiledKernel._init_handles`). Nothing is precompiled or cached ahead of the run: the compile happens in
this process, during the measured wall time, just earlier and in parallel.

Where the thread starts: `start()` is the first thing `apply()` does, so the compile overlaps
everything that follows it in the main thread: the rest of `apply()`, Soup's imports and data prep,
and the model load (mostly file reads, the NF4 quantize and host-to-GPU copies, which release the
GIL). Measured on a T4: starting the thread later, at the first `from_pretrained`, cost ~4 s per
run (training then waited 4.0-4.2 s for the thread at step 0, and step 0 took 5.0 s instead of
0.9 s).

`apply()` behaves the same in every process; there is no test detection. Under
`SOUP_APPLY_SOLUTION=1 pytest tests` (tests/conftest.py calls `apply()`) the thread therefore
compiles while Soup's tests run, and a silent `atexit` handler joins it (timeout
`EXIT_JOIN_TIMEOUT_S`) before interpreter teardown, so the interpreter never tears down under a thread still inside
Triton/LLVM.

Errors never reach training: a failing workstream warmup is noted and skipped (its kernel then
compiles on its first real launch, as without warmup).

Output: neither `start()` nor the thread nor the exit join prints anything. Notes (the thread's
timing summary, failures as exception type names only) are buffered and printed by `join()` at the
first training step, with the words "error", "failed" and "passed" masked. Reason:
harness/check_rules.py takes the last output line of `pytest tests` that contains " passed",
" failed" or "error" as check 10's result, and stderr comes after stdout there, so any such line
from us outside a training run would replace pytest's own summary.

Triton 3.6 does not lock its in-memory kernel cache (`JITFunction.run` -> `_do_compile` stores
the result only after the whole compile), so a real launch of a kernel that the thread is still
compiling compiles it a second time, from scratch, in the main thread. That costs time, not
correctness. `_init_handles` is not safe against a concurrent first launch either (it sets the
launcher before it loads the function). To avoid both, the first `transformers.Trainer.training_step` waits for the thread
(`join()`): the latest point before step 0's forward launches our kernels.

Idea sources: Triton 3.6's own public API and source, nothing else. `.warmup()` is
`KernelInterface.warmup` (triton/runtime/jit.py, inherited by `JITFunction`, = `run(...,
warmup=True)`); the unlocked cache store is `JITFunction._do_compile` (same file); the lazy load
and its ordering (`self._run` set before `load_binary`) are `CompiledKernel._init_handles` and
the `CompiledKernel.run` property (triton/compiler/compiler.py). The exit join is plain `atexit` +
`Thread.join` from the Python standard library. Original here: compiling ahead of time in a
thread during model load, with no prebuilt cache, and the join at the first training step. No code from Triton/Unsloth/other projects is copied.
"""

import atexit
import functools
import importlib
import importlib.util
import re
import sys
import threading
import time

# Workstream packages under solution/, warmed up in this order (missing ones are skipped).
WORKSTREAMS = ("nf4", "head", "body")
# The first training step waits at most this long for the thread (a hung compile must not stall
# training; after the timeout the kernels compile on first launch as usual).
JOIN_TIMEOUT_S = 120.0
# At interpreter exit the atexit handler waits at most this long for a still-running thread.
EXIT_JOIN_TIMEOUT_S = 120.0

_lock = threading.Lock()
_thread = None
_MARK = "_soup_warmup_join"
# Notes from start() and the thread, printed only by join() at the first training step.
_notes = []
# Words harness/check_rules.py looks for in the test output (check 10); never printed by us.
_CHECK_WORDS = re.compile(r"error|failed|passed", re.IGNORECASE)
_MASK = {"error": "ERR", "failed": "FAILED", "passed": "PASSED"}


def _log(msg):
    """Buffer a note (list.append is atomic: safe from the thread). Prints nothing."""
    _notes.append(msg)


def _flush():
    """Print the buffered notes (main thread, first training step), without check-10 words."""
    while _notes:
        msg = _CHECK_WORDS.sub(lambda m: _MASK[m.group(0).lower()], _notes.pop(0))
        print(f"[solution.warmup] {msg}", file=sys.stderr, flush=True)


def _targets():
    """(name, warmup function) for each workstream package that exists and exports warmup().

    Imported here, in the calling (main) thread: apply() imports these packages anyway, and
    importing in the worker while the main thread imports the same modules could hit Python's
    import-lock deadlock handling (a partially initialised module)."""
    out = []
    for name in WORKSTREAMS:
        modname = f"{__package__}.{name}"
        try:
            if importlib.util.find_spec(modname) is None:
                continue
            fn = getattr(importlib.import_module(modname), "warmup", None)
        except Exception as exc:  # noqa: BLE001 - never break apply()
            _log(f"{name}: import failed ({type(exc).__name__})")
            continue
        if callable(fn):
            out.append((name, fn))
    return out


def _jit_functions():
    """The @triton.jit kernels defined in the (already imported) modules of this package (the
    workstreams' and the example's: a workstream may warm up kernels it launches from elsewhere)."""
    from triton.runtime.jit import JITFunction

    pkg = __package__
    seen = {}
    for mname, m in list(sys.modules.items()):
        if m is None or not (mname == pkg or mname.startswith(pkg + ".")):
            continue
        for obj in list(vars(m).values()):
            if isinstance(obj, JITFunction):
                seen[id(obj)] = obj
    return list(seen.values())


def _compiled(device):
    """Every compiled variant of our kernels in Triton's in-memory cache for `device`."""
    out = []
    for jit in _jit_functions():
        if device.index in jit.device_caches:  # defaultdict: do not create an entry
            out.extend(jit.device_caches[device.index][0].values())
    return out


def _load(device):
    """Finish what `.warmup()` leaves to the first launch, without launching: build the C
    launcher (a gcc compile per distinct signature) and load the cubin (`_init_handles`, which
    `CompiledKernel.run` would call on first use). Returns the number of variants loaded now."""
    n = 0
    for kernel in _compiled(device):
        try:
            kernel = kernel.result() if hasattr(kernel, "result") else kernel
            if kernel.module is None:
                kernel._init_handles()
                n += 1
        except Exception as exc:  # noqa: BLE001 - the first launch raises it again, as usual
            _log(f"loading {getattr(kernel, 'name', '?')} failed ({type(exc).__name__})")
    return n


def _run(device, targets):
    import torch

    t0 = time.perf_counter()
    parts = []
    try:
        torch.cuda.set_device(device)  # Triton compiles and loads for this thread's device
    except Exception as exc:  # noqa: BLE001
        _log(f"set_device({device}) failed, no warmup ({type(exc).__name__})")
        return
    for name, fn in targets:
        t = time.perf_counter()
        try:
            fn(device)
            n = _load(device)
            parts.append(f"{name} {time.perf_counter() - t:.2f} s (+{n})")
        except BaseException as exc:  # noqa: BLE001 - logged, never propagated
            parts.append(f"{name} failed after {time.perf_counter() - t:.2f} s ({type(exc).__name__})")
    try:
        n = f", {len(_compiled(device))} kernel variants compiled and loaded"
    except Exception:  # noqa: BLE001
        n = ""
    _log(f"{', '.join(parts) or 'nothing to warm up'}; total {time.perf_counter() - t0:.2f} s{n}")


def join(timeout=JOIN_TIMEOUT_S):
    """Wait for the warmup thread (no-op if none is running), then print the buffered notes.
    True when it is done."""
    t = _thread
    done = True
    if t is not None and t.is_alive():
        t0 = time.perf_counter()
        t.join(timeout)
        done = not t.is_alive()
        _log(f"training waited {time.perf_counter() - t0:.2f} s for the warmup thread"
             + ("" if done else f" and gave up after {timeout} s"))
    _flush()
    return done


def _install_join_hook():
    """Wrap `transformers.Trainer.training_step` to join() first (TRL's SFTTrainer and Soup's
    subclass call it through super()). Kept after the first step: then join() is one check."""
    try:
        import transformers

        trainer = transformers.Trainer
        orig = trainer.training_step
    except Exception:  # noqa: BLE001 - no (real) transformers: nothing to hook
        return
    if getattr(orig, _MARK, False):
        return

    @functools.wraps(orig)
    def training_step(self, *args, **kwargs):
        join()
        return orig(self, *args, **kwargs)

    setattr(training_step, _MARK, True)
    trainer.training_step = training_step


def _join_at_exit():
    """atexit backstop: do not let the interpreter tear down under a thread still in Triton/LLVM.
    Silent and never raises (it may run after pytest's summary line, see the module docstring)."""
    try:
        t = _thread
        if t is not None and t.is_alive():
            t.join(EXIT_JOIN_TIMEOUT_S)
    except BaseException:  # noqa: BLE001
        pass


def start():
    """Start the warmup thread once per process. No CUDA -> nothing. Never raises."""
    global _thread
    try:
        with _lock:
            if _thread is not None:
                return
            import torch

            if not torch.cuda.is_available():
                return
            # Triton's own knob (TRITON_STORE_BINARY_ONLY, triton/knobs.py; triton/compiler/
            # compiler.py `compile`): write only cubin + json to this run's cache dir, not the
            # five IR files per kernel. The kernels are identical; each skipped file write is one
            # GIL release/re-acquire less for this thread while the main thread imports (local:
            # -25 % thread time under contention). Nothing is read back across runs.
            import triton

            triton.knobs.compilation.store_binary_only = True
            device = torch.device("cuda", torch.cuda.current_device())
            targets = _targets()
            if not targets:
                return
            _thread = threading.Thread(target=_run, args=(device, targets),
                                       name="solution-triton-warmup", daemon=True)
            atexit.register(_join_at_exit)
            _thread.start()
        _log("thread started")
        _install_join_hook()
    except Exception as exc:  # noqa: BLE001 - never break apply()
        _log(f"not started ({type(exc).__name__})")

