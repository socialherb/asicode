"""
Pytest fixtures for asicode agent tests.
"""

from __future__ import annotations

import contextlib
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest

# Heavy imports deferred to fixture bodies (saves ~300ms on collection).
# TYPE_CHECKING-only imports keep the annotation names (AgentConfig, ToolRegistry,
# AgentLoop, FileLockManager) resolvable for ruff's F821 without paying the
# collection-time import cost -- the block never executes at runtime.
if TYPE_CHECKING:
    from external_llm.agent.agent_loop import AgentLoop
    from external_llm.agent.orchestrator import FileLockManager
    from external_llm.agent.tool_registry import AgentConfig, ToolRegistry


@pytest.fixture
def restore_sys_modules():
    """Snapshot/restore ``sys.modules`` around a test that reloads a package.

    Purging a module tree without putting it back corrupts module IDENTITY for
    the rest of the session: a later ``import x.y`` builds a FRESH module
    object while every already-imported test module still holds classes whose
    ``__globals__`` point at the orphaned original. ``mock.patch`` then writes
    to one and the code under test reads the other, so the patch silently does
    nothing and the test it belongs to fails somewhere unrelated.

    That is not hypothetical twice over: ``test_planner_lane_facade`` hit it
    ("the patch silently misses and the real searcher runs against a bare Mock
    run_store") and grew a private copy of this fixture, then
    ``test_package_lazy_import`` reintroduced the same leak and cost four
    integration tests that passed alone and failed only in a combined
    ``tests/unit tests/integration`` run. Shared here so a third reloading test
    inherits the fix instead of rediscovering the symptom.

    Restores the exact pre-test table: modules imported during the test are
    removed, originals are put back.
    """
    import sys as _sys

    snapshot = dict(_sys.modules)
    try:
        yield
    finally:
        for k in list(_sys.modules):
            if k not in snapshot:
                del _sys.modules[k]
        _sys.modules.update(snapshot)


@pytest.fixture(autouse=True)
def _suppress_legacy_shadow_env(monkeypatch):
    """Suppress legacy shadow-logger env var for all tests.

    Prevents fixture-target pollution of ~/.asicode/*.jsonl.
    The shadow module was removed in a cleanup — the env var guard
    is kept for any residual references.
    """
    monkeypatch.setenv("ASICODE_DISABLE_SHADOW", "1")


@pytest.fixture(scope="session", autouse=True)
def _isolate_context_override_cache(tmp_path_factory):
    """Keep the test session off the user's real context-override cache.

    Sibling of :func:`_isolate_runs_dir`, for a leak that outlives any fixture.
    ``context_budget`` registers an atexit flush that force-writes the cache, and
    that fires at INTERPRETER exit — after pytest has torn down every
    monkeypatch, session-scoped ones included. Measured: planting
    ``{"MARKER": ...}`` in the real file and running the suite returned it as
    ``{}``, the user's persisted overrides erased. They exist so a model that hit
    a context-overflow 400 does not rediscover its real limit on every fresh
    process, so erasing them costs a real session one repeat 400. Writing junk is
    possible too — whatever ``_override_meta`` holds at exit gets serialised.

    Both levers are needed, and the env var is the load-bearing one:

    * ``ASICODE_CONTEXT_OVERRIDE_CACHE`` is read at import, so it is what
      redirects the 63 test files that spawn Python SUBPROCESSES. A child gets a
      fresh copy of the module constant and its own atexit handler, which no
      in-process patch in the parent can reach — verified: a bare ``python -c
      "import external_llm.agent.context_budget"`` rewrote the real file at exit,
      and a session fixture patching only the attribute did not stop the full
      suite from clobbering the marker.
    * the attribute assignment covers THIS process, where the module may already
      have been imported before the fixture ran.

    Neither is restored — nothing in a test process wants the real path, and
    leaving them in place is what makes the atexit flush safe.
    """
    import os

    import external_llm.agent.context_budget as cb

    target = str(tmp_path_factory.mktemp("ctx_override_cache") / "context_override_cache.json")
    os.environ["ASICODE_CONTEXT_OVERRIDE_CACHE"] = target
    cb._OVERRIDE_CACHE_FILE = target


@pytest.fixture(scope="session", autouse=True)
def _isolate_strategy_state(tmp_path_factory):
    """Keep the suite out of the user's real learning state.

    Third instance of the pattern :func:`_isolate_context_override_cache`
    documents. Audited 2026-07-28 by snapshotting ~/.asicode before and after a
    full run (read-only — size/mtime/md5, no marker writes):

    * ``learning/strategy_state.json`` was CREATED, carrying ``adaptive_hub``
      Q-values and counts fabricated by tests. The live agent loop writes this
      namespace every few seconds and reads it back, so test outcomes were being
      merged into the weights a real session acts on.
    * ``learning/experience_store.json`` was rewritten: the store is FIFO-capped
      at 200 records, so a run adds test entries and EVICTS real ones — 17 lost
      for 15 added in the audited run.

    One env var covers both: ``_path_for`` derives every sidecar from
    ``os.path.dirname(_STRATEGY_STATE_PATH)``. Set as an env var, and not
    restored, for the reasons in ``_isolate_context_override_cache`` — the
    leaking writers are in subprocesses that re-import the module.

    Audited and NOT leaking (no file created across a full run, kept here so the
    next audit does not redo them): ``routing_policy.json``, ``tool_state.json``,
    ``learning/exploration.json``, ``learning/run_history.jsonl``.
    """
    import os

    target = str(tmp_path_factory.mktemp("strategy_state") / "strategy_state.json")
    os.environ["ASICODE_STRATEGY_STATE"] = target
    try:
        import external_llm.editor.learning.strategy_state as ss

        ss._STRATEGY_STATE_PATH = target
    except Exception:  # pragma: no cover - import shape only
        pass


@pytest.fixture(scope="session")
def _isolate_session_base(tmp_path_factory) -> Path:
    """One numbered base dir for the whole session.

    pytest's ``make_numbered_dir`` scans the entire basetemp to find the highest
    existing number, so per-test ``mktemp`` calls are O(dir-count). The two
    autouse isolation fixtures below used to make ~16.5k of them per suite
    (8,237 tests x 2), which is O(n²) overall (~30-60s serial at 8k dirs).
    Per-test subdirs are now uuid-named below this base, which is O(1).
    """
    return tmp_path_factory.mktemp("asr_isolate")


@pytest.fixture(autouse=True)
def _isolate_runs_dir(_isolate_session_base, monkeypatch):
    """Isolate run artifacts and write-tool failure logs per test.

    Run artifacts: prevents test runs from leaking into the real .asicode/runs/.
    config.ASICODE_RUNS_DIR is resolved ONCE at import time, and consumers copy
    it by value via ``from config import ASICODE_RUNS_DIR``. Simply setting the
    env var is therefore insufficient (the value is already frozen). We (1)
    set the env var for any future reload, and (2) patch the attribute on
    ``config`` and every already-imported module that captured its own copy.
    Modules imported LATER during the test read config's (already-patched)
    value directly, so they are covered transitively.

    Write-tool failures: ``tool_failure_log`` reads
    ``ASICODE_WRITE_TOOL_FAILURE_LOG`` at call time (no module-attribute copy),
    so the env var alone is enough there. Keeps fabricated test failures out of
    the user's real ``~/.asicode/learning/write_tool_failures.jsonl`` — before
    this fixture existed, any test that tripped a write-tool failure appended to
    the real log; confirmed from the real log during a suite run: records
    carrying ``"repo": "/private/var/folders/.../T/e2e-repo-*"`` (pytest tmp
    dirs). The file is write-only for shipping code, so the cost is a skewed
    dataset rather than a wrong decision at runtime.

    Both env vars point under one uuid-named per-test subdir of the session
    base (O(1) to create, no basetemp scan).
    """
    import os
    import sys
    import uuid

    base = _isolate_session_base / f"t{uuid.uuid4().hex[:12]}"
    runs_dir = base / "runs"
    os.makedirs(runs_dir, exist_ok=True)
    target = str(runs_dir)
    monkeypatch.setenv("ASICODE_RUNS_DIR", target)
    monkeypatch.setenv(
        "ASICODE_WRITE_TOOL_FAILURE_LOG",
        str(base / "write_tool_failures.jsonl"),
    )
    # NOTE: use ``in mod.__dict__`` (membership) instead of ``hasattr``.
    # hasattr() triggers a module's lazy ``__getattr__`` — e.g. the
    # ``transformers`` package routes any ``ASICODE_*`` attribute through an
    # Aria-image-processing submodule, which imports torchvision and raises
    # ModuleNotFoundError. __dict__ membership sees only attributes that were
    # actually defined/imported by the module, with no side effects.
    for mod in list(sys.modules.values()):
        if mod is not None and "ASICODE_RUNS_DIR" in mod.__dict__:
            with contextlib.suppress(AttributeError, TypeError):
                monkeypatch.setattr(mod, "ASICODE_RUNS_DIR", target)


@pytest.fixture
def temp_repo_root() -> Generator[str, None, None]:
    """Create a temporary directory as a fake repository root."""
    tmpdir = tempfile.mkdtemp(prefix="asr-test-")
    try:
        # Initialize as a git repo for git operations. Five subprocess spawns
        # per test used to cost ~150ms each under 8-way parallelism; the user
        # identity is written straight into .git/config (git init always
        # creates that file) instead of two `git config` subprocesses.
        import subprocess

        subprocess.run(["git", "init", "-q"], cwd=tmpdir, capture_output=True, check=False)
        with (Path(tmpdir) / ".git" / "config").open("a") as cfg:
            cfg.write("[user]\n\temail = test@example.com\n\tname = Test User\n")

        # Create a sample Python file for testing
        sample_file = Path(tmpdir) / "sample.py"
        # NOTE: Avoid leading newline so patch hunks starting at line 1 match reliably.
        sample_file.write_text(
            "def hello() -> str:\n"
            '    return "world"\n'
            "\n"
            "class Calculator:\n"
            "    def add(self, a: int, b: int) -> int:\n"
            "        return a + b\n"
        )

        # Commit the file
        subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True, check=False)
        subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=tmpdir, capture_output=True, check=False)

        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def temp_repo_root_with_memory(temp_repo_root: str) -> Generator[str, None, None]:
    """Create temporary repo root with .asicode/memory.md file."""
    asicode_dir = Path(temp_repo_root) / ".asicode"
    asicode_dir.mkdir(exist_ok=True)
    memory_file = asicode_dir / "memory.md"
    memory_file.write_text("# Test Memory\n\nThis is a test memory file.")
    yield temp_repo_root


@pytest.fixture
def agent_config() -> AgentConfig:
    """Return a basic AgentConfig for testing."""
    from external_llm.agent.enums import Complexity, Scope
    from external_llm.agent.task_router import Lane, RouteDecision, TaskKind
    from external_llm.agent.tool_registry import AgentConfig

    config = AgentConfig(
        max_turns=5,
        run_tests=False,
        run_lint=False,
        auto_test_on_patch=False,
        self_review_enabled=False,
        rag_enabled=False,
        parallel_tool_execution_enabled=False,
    )
    # Without a route_decision, AgentLoop.run() hits the unhandled-lane
    # guard and returns error — every AgentLoop-based test would fail.
    config.route_decision = RouteDecision(
        task_kind=TaskKind.SINGLE_FILE_EDIT,
        complexity=Complexity.LOW,
        scope=Scope.SINGLE_FILE,
        lane=Lane.MAIN_AGENT,
        confidence=0.9,
        reasoning="Test fixture default",
    )
    return config


@pytest.fixture
def mock_llm_client() -> Mock:
    """Return a mock LLM client."""
    client = Mock()

    # Mock provider name - use "openai" to enable native tool calling
    client.get_provider_name.return_value = "openai"
    client.provider = "openai"

    # Mock chat_with_tools method
    mock_response = Mock()
    mock_response.content = "Test response"
    mock_response.tool_calls = []
    mock_response.prompt_tokens = 0
    mock_response.completion_tokens = 0
    mock_response.cache_read_input_tokens = 0
    mock_response.cache_creation_input_tokens = 0
    mock_response.raw_response = None  # For OpenAI format compatibility
    client.chat_with_tools.return_value = mock_response

    # Mock chat method for text mode
    client.chat.return_value = mock_response

    return client


@pytest.fixture
def tool_registry(temp_repo_root: str, agent_config: AgentConfig) -> ToolRegistry:
    """Return a ToolRegistry instance for testing."""
    from external_llm.agent.tool_registry import ToolRegistry

    return ToolRegistry(temp_repo_root, agent_config)


@pytest.fixture
def agent_loop(mock_llm_client: Mock, tool_registry: ToolRegistry, agent_config: AgentConfig) -> AgentLoop:
    """Return an AgentLoop instance with mocked LLM client."""
    from external_llm.agent.agent_loop import AgentLoop

    return AgentLoop(
        llm_client=mock_llm_client,
        registry=tool_registry,
        config=agent_config,
        model="test-model",
    )


@pytest.fixture
def sample_patch() -> str:
    """Return a sample valid patch for testing."""
    return """--- a/sample.py
+++ b/sample.py
@@ -1,6 +1,9 @@
 def hello() -> str:
     return "world"

 class Calculator:
+    def __init__(self):
+        self.memory = 0
+
     def add(self, a: int, b: int) -> int:
         return a + b
"""


@pytest.fixture
def invalid_patch() -> str:
    """Return an invalid patch for testing."""
    return """--- a/sample.py
+++ b/sample.py
@@ -100,6 +100,9 @@ def hello() -> str:
     return "world"

 class Calculator:
+    def __init__(self):
+        self.memory = 0
+
     def add(self, a: int, b: int) -> int:
         return a + b
"""


@pytest.fixture
def file_lock_manager() -> FileLockManager:
    """FileLockManager instance for multi-agent tests."""
    from external_llm.agent.orchestrator import FileLockManager

    return FileLockManager()


@pytest.fixture
def sample_plan_dict() -> dict[str, Any]:
    """Valid ASICODE_PLAN_V1 plan structure."""
    return {
        "version": "ASICODE_PLAN_V1",
        "operations": [
            {
                "type": "edit_blocks",
                "path": "sample.py",
                "blocks": [
                    {
                        "before": "    def add(self, a: int, b: int) -> int:\n        return a + b",
                        "after": "    def add(self, a: int, b: int) -> int:\n        return a + b\n\n    def subtract(self, a: int, b: int) -> int:\n        return a - b",
                    }
                ],
            }
        ],
    }


@pytest.fixture
def sample_simple_edit_plan_dict() -> dict[str, Any]:
    """Simple edit plan for testing."""
    return {
        "version": "ASICODE_PLAN_V1",
        "operations": [
            {
                "type": "edit_blocks",
                "path": "sample.py",
                "blocks": [
                    {
                        "before": "    def add(self, a: int, b: int) -> int:\n        return a + b",
                        "after": "    def add(self, a: int, b: int) -> int:\n        return a + b  # Fixed indentation",
                    }
                ],
            }
        ],
    }


@pytest.fixture
def sample_create_file_plan_dict() -> dict[str, Any]:
    """Create file plan for testing."""
    return {
        "version": "ASICODE_PLAN_V1",
        "operations": [
            {"type": "create_file", "path": "new_file.py", "content": 'def new_function():\n    return "new"'}
        ],
    }


@pytest.fixture
def sample_multi_file_plan_dict() -> dict[str, Any]:
    """Multi-file plan for testing."""
    return {
        "version": "ASICODE_PLAN_V1",
        "operations": [
            {
                "type": "edit_blocks",
                "path": "sample.py",
                "blocks": [
                    {
                        "before": "    def add(self, a: int, b: int) -> int:\n        return a + b",
                        "after": "    def add(self, a: int, b: int) -> int:\n        return a + b  # multi",
                    }
                ],
            },
            {"type": "create_file", "path": "utils.py", "content": 'def helper():\n    return "help"'},
        ],
    }
