"""Ownership of the AsyncToolExecutor's thread pool.

``ToolRegistry`` used to carry a ``__del__`` that called
``async_executor.shutdown()``. It was both redundant and harmful:

* ``clone_with_filter`` SHARES the parent's executor (unlike
  ``clone_for_subagent``, which sets it to None), so collecting a filtered
  clone shut down a pool the parent was still using — after which the parent
  raised ``RuntimeError: cannot schedule new futures after shutdown`` for the
  rest of the session. Deterministic, reproduced below.
* Running a blocking pool shutdown from a GC finalizer crashed the interpreter
  outright: ``tests/unit`` on 3.12 died with SIGSEGV, the faulthandler stack
  showing ``Garbage-collecting`` above ``tool_registry.__del__ ->
  AsyncToolExecutor.shutdown -> ThreadPoolExecutor.shutdown``, interrupting an
  unrelated ``symbol_search._walk_outline``. (A SIGBUS was also seen on 3.14,
  but its stack is GC during ``subprocess._close_pipe_fds`` with no repo
  finalizer visible — same class of hazard, deliberately NOT claimed as the
  same cause.)

CPython already does this cleanup, in a finalizer-safe way: ThreadPoolExecutor
registers ``weakref.ref(self, weakref_cb)`` whose callback puts ``None`` on the
work queue, so idle workers wake and exit once the executor becomes
unreachable. ``test_idle_workers_exit_when_executor_is_dropped`` pins that
mechanism, because it is the reason the ``__del__`` could be deleted rather
than merely made ownership-aware.
"""
from __future__ import annotations

import gc
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from external_llm.agent.tool_chain import ScopedToolFilter
from external_llm.agent.tool_registry import AgentConfig, ToolRegistry


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    (tmp_path / "a.py").write_text("x = 1\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return tmp_path


def _registry(repo: Path) -> ToolRegistry:
    return ToolRegistry(str(repo), AgentConfig(planning_enabled=False, rag_enabled=False))


def _needs_executor(reg: ToolRegistry):
    if reg.async_executor is None:
        pytest.skip("parallel execution disabled in this configuration")
    return reg.async_executor.executor


class TestSharedExecutorSurvivesCloneCollection:
    def test_filtered_clone_shares_the_parent_executor(self, repo: Path):
        """Pins the precondition: if this stops being shared the bug below
        cannot occur, and this test should be revisited rather than deleted."""
        parent = _registry(repo)
        _needs_executor(parent)
        clone = parent.clone_with_filter(ScopedToolFilter(allowed_write={"a.py"}))
        assert clone.async_executor is parent.async_executor

    def test_collecting_a_filtered_clone_leaves_the_parent_pool_usable(self, repo: Path):
        parent = _registry(repo)
        pool = _needs_executor(parent)

        clone = parent.clone_with_filter(ScopedToolFilter(allowed_write={"a.py"}))
        del clone
        gc.collect()

        assert pool._shutdown is False, (
            "a filtered clone's collection shut down the parent's live thread pool"
        )
        # The observable consequence, not just the flag: the parent must still
        # be able to schedule work.
        assert pool.submit(lambda: 21 * 2).result(timeout=10) == 42

    def test_subagent_clone_still_gets_no_executor(self, repo: Path):
        """Unchanged behaviour — subagent clones deliberately drop the pool."""
        parent = _registry(repo)
        _needs_executor(parent)
        sub = parent.clone_for_subagent(
            AgentConfig(planning_enabled=False, rag_enabled=False)
        )
        assert sub.async_executor is None

    def test_parent_pool_survives_many_clone_generations(self, repo: Path):
        parent = _registry(repo)
        pool = _needs_executor(parent)
        for _ in range(20):
            c = parent.clone_with_filter(ScopedToolFilter(allowed_write={"a.py"}))
            del c
        gc.collect()
        assert pool._shutdown is False
        assert pool.submit(lambda: "ok").result(timeout=10) == "ok"


def test_idle_workers_exit_when_executor_is_dropped():
    """CPython's weakref callback is what replaces the deleted ``__del__``.

    If this ever stops holding, dropping the destructor would start leaking
    idle worker threads in a long-lived process and the removal would need
    revisiting — so the mechanism is pinned here rather than assumed.
    """
    before = set(threading.enumerate())
    pool = ThreadPoolExecutor(max_workers=2)
    pool.submit(lambda: None).result(timeout=10)
    workers = [t for t in threading.enumerate() if t not in before]
    assert workers, "expected at least one worker thread to have been spawned"

    del pool
    gc.collect()

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if not any(t.is_alive() for t in workers):
            break
        time.sleep(0.05)

    assert not any(t.is_alive() for t in workers), (
        "idle workers did not exit after the executor became unreachable — "
        "CPython's weakref cleanup no longer covers what __del__ used to"
    )
