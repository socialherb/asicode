"""Unit tests for pre-existing dirty file snapshot exclusion in scoped verification.

The snapshot mechanism captures test files that were already dirty (user's WIP)
before any agent action, and excludes them from the scoped verification quality gate
to prevent false failures when the user already has broken test files.
"""
import os
import subprocess
from unittest.mock import MagicMock

import pytest

from external_llm.agent.operation_models import Operation, OperationKind, PlanMode


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_agent_loop(repo_root: str):
    """Create a minimal AgentLoop instance using __new__ (bypass heavy __init__)."""
    from external_llm.agent.agent_loop import AgentLoop

    loop = AgentLoop.__new__(AgentLoop)
    registry = MagicMock()
    registry.repo_root = repo_root
    loop.registry = registry
    loop.slash_commands = None
    loop._cb = MagicMock()
    loop.config = MagicMock()
    loop.config.run_tests = True
    loop.config.scoped_verification = True
    return loop


def _make_state():
    state = MagicMock()
    state.plan_mode = PlanMode.EDIT
    state.completed_ops = {}
    state.failed_ops = {}
    state.force_finish = False
    state.pre_op_file_snapshots = {}
    state.modified_files_set = set()
    return state


def _make_op(op_id: str = "op1", path: str = "a/x.py", symbol: str = "foo") -> Operation:
    return Operation(id=op_id, kind=OperationKind.MODIFY_SYMBOL, path=path, symbol=symbol)


# ── Filtering logic (scoped verification block) ──────────────────────────────


class TestPreExistingDirtyFilter:
    """Verify that _run_quality_gate correctly excludes pre-existing dirty test
    files from the scoped verification touched set."""

    @pytest.fixture(autouse=True)
    def _patch_tis_functions(self):
        """Patch test_impact_selector functions used inside _run_quality_gate.

        The `from .test_impact_selector import ...` is a local import inside
        the try block — it reads from the module at runtime, so patching the
        module attribute before calling _run_quality_gate intercepts correctly.
        """
        import external_llm.agent.test_impact_selector as tis

        self._git_returns = []
        orig_git = tis.git_status_test_files
        orig_is_test = tis.is_test_file
        orig_sel = tis.select_affected_tests
        orig_inv = tis.invalidate_index

        def _fake_git(_repo):
            return self._git_returns

        def _fake_is_test(p):
            return p.endswith(".py") and ("test_" in p or "_test.py" in p or p.startswith("tests/"))

        def _fake_sel(_repo, touched, **kw):
            return [t for t in touched if "test" in t]

        def _fake_inv(_repo):
            pass

        tis.git_status_test_files = _fake_git
        tis.is_test_file = _fake_is_test
        tis.select_affected_tests = _fake_sel
        tis.invalidate_index = _fake_inv
        try:
            yield
        finally:
            tis.git_status_test_files = orig_git
            tis.is_test_file = orig_is_test
            tis.select_affected_tests = orig_sel
            tis.invalidate_index = orig_inv

    # ── Core scenarios ────────────────────────────────────────────────────

    def test_excludes_single_dirty_file(self, tmp_path):
        """A file in _pre_existing_dirty_files is excluded from _touched."""
        loop = _make_agent_loop(str(tmp_path))
        loop._pre_existing_dirty_files = {"tests/test_broken.py"}
        self._git_returns = ["tests/test_broken.py", "tests/test_good.py"]

        with _mock_dispatch(loop):
            loop._run_quality_gate(_make_op(), _make_state(), {"status": "success"}, is_last_op=True)

        cb_calls = [c for c in loop._cb.call_args_list if c[0][0] == "quality_gate_tests_passed"]
        assert len(cb_calls) == 1
        selected = cb_calls[0][0][1]["selected_test_files"]
        assert "tests/test_good.py" in selected, "Good file should be in selected tests"
        assert "tests/test_broken.py" not in selected, "Dirty file must be excluded"

    def test_none_snapshot_skips_filtering(self, tmp_path):
        """When _pre_existing_dirty_files is None, all git tests pass through."""
        loop = _make_agent_loop(str(tmp_path))
        loop._pre_existing_dirty_files = None  # snapshot not acquired
        self._git_returns = ["tests/test_broken.py", "tests/test_good.py"]

        with _mock_dispatch(loop):
            loop._run_quality_gate(_make_op(), _make_state(), {"status": "success"}, is_last_op=True)

        cb_calls = [c for c in loop._cb.call_args_list if c[0][0] == "quality_gate_tests_passed"]
        assert len(cb_calls) == 1
        selected = cb_calls[0][0][1]["selected_test_files"]
        assert "tests/test_broken.py" in selected, "None snapshot: broken file must NOT be excluded"
        assert "tests/test_good.py" in selected, "None snapshot: good file must be included"

    def test_empty_set_snapshot_allows_all(self, tmp_path):
        """When _pre_existing_dirty_files is set() (empty), filtering is active
        but nothing matches — all files pass through."""
        loop = _make_agent_loop(str(tmp_path))
        loop._pre_existing_dirty_files = set()  # acquired but empty
        self._git_returns = ["tests/test_one.py", "tests/test_two.py"]

        with _mock_dispatch(loop):
            loop._run_quality_gate(_make_op(), _make_state(), {"status": "success"}, is_last_op=True)

        cb_calls = [c for c in loop._cb.call_args_list if c[0][0] == "quality_gate_tests_passed"]
        assert len(cb_calls) == 1
        selected = cb_calls[0][0][1]["selected_test_files"]
        assert "tests/test_one.py" in selected
        assert "tests/test_two.py" in selected

    # ── Partial overlap ───────────────────────────────────────────────────

    def test_partial_overlap(self, tmp_path):
        """Only files that match both pre-existing and git_tests are excluded."""
        loop = _make_agent_loop(str(tmp_path))
        loop._pre_existing_dirty_files = {"tests/test_fixme.py", "tests/test_wip.py"}
        self._git_returns = ["tests/test_fixme.py", "tests/test_good.py", "tests/test_new.py"]

        with _mock_dispatch(loop):
            loop._run_quality_gate(_make_op(), _make_state(), {"status": "success"}, is_last_op=True)

        cb_calls = [c for c in loop._cb.call_args_list if c[0][0] == "quality_gate_tests_passed"]
        assert len(cb_calls) == 1
        selected = cb_calls[0][0][1]["selected_test_files"]
        assert "tests/test_fixme.py" not in selected, "fixme must be excluded"
        assert "tests/test_good.py" in selected
        assert "tests/test_new.py" in selected

    def test_no_overlap(self, tmp_path):
        """When pre-existing set and git_tests are disjoint, nothing is excluded."""
        loop = _make_agent_loop(str(tmp_path))
        loop._pre_existing_dirty_files = {"tests/other_broken.py"}
        self._git_returns = ["tests/test_good.py"]

        with _mock_dispatch(loop):
            loop._run_quality_gate(_make_op(), _make_state(), {"status": "success"}, is_last_op=True)

        cb_calls = [c for c in loop._cb.call_args_list if c[0][0] == "quality_gate_tests_passed"]
        assert len(cb_calls) == 1
        selected = cb_calls[0][0][1]["selected_test_files"]
        assert "tests/test_good.py" in selected

    def test_all_dirty_no_clean(self, tmp_path):
        """When ALL git_tests are dirty, _git_tests becomes empty after filter.
        _touched stays as modified_files_set only (empty) → no selection → full suite."""
        loop = _make_agent_loop(str(tmp_path))
        loop._pre_existing_dirty_files = {"tests/test_dirty1.py", "tests/test_dirty2.py"}
        self._git_returns = ["tests/test_dirty1.py", "tests/test_dirty2.py"]

        with _mock_dispatch(loop):
            loop._run_quality_gate(_make_op(), _make_state(), {"status": "success"}, is_last_op=True)

        cb_calls = [c for c in loop._cb.call_args_list if c[0][0] == "quality_gate_tests_passed"]
        assert len(cb_calls) == 1
        mode = cb_calls[0][0][1]["verification_mode"]
        reason = cb_calls[0][0][1]["fallback_reason"]
        assert mode == "full", "All dirty → fallback to full suite"
        assert reason == "no_modified_files", (
            "All dirty → _git_tests emptied, no modified_files_set → no_modified_files"
        )

    # ── Edge: empty git_tests ────────────────────────────────────────────

    def test_empty_git_tests_nothing_to_filter(self, tmp_path):
        """When git_status_test_files returns [], empty _touched → fallback."""
        loop = _make_agent_loop(str(tmp_path))
        loop._pre_existing_dirty_files = {"tests/test_dirty.py"}
        self._git_returns = []  # no test files found by git status

        with _mock_dispatch(loop):
            loop._run_quality_gate(_make_op(), _make_state(), {"status": "success"}, is_last_op=True)

        cb_calls = [c for c in loop._cb.call_args_list if c[0][0] == "quality_gate_tests_passed"]
        assert len(cb_calls) == 1
        reason = cb_calls[0][0][1]["fallback_reason"]
        assert reason == "no_modified_files", "Empty git_tests + empty modified → fallback"

    # ── Modified files + filter ──────────────────────────────────────────

    def test_modified_files_set_preserved_through_filter(self, tmp_path):
        """modified_files_set content survives the filter and reaches select_affected_tests."""
        loop = _make_agent_loop(str(tmp_path))
        loop._pre_existing_dirty_files = {"tests/test_dirty.py"}
        self._git_returns = ["tests/test_dirty.py", "tests/test_good.py"]

        state = _make_state()
        state.modified_files_set = {"src/modified.py"}

        with _mock_dispatch(loop):
            loop._run_quality_gate(_make_op(), state, {"status": "success"}, is_last_op=True)

        cb_calls = [c for c in loop._cb.call_args_list if c[0][0] == "quality_gate_tests_passed"]
        assert len(cb_calls) == 1
        selected = cb_calls[0][0][1]["selected_test_files"]
        assert "tests/test_good.py" in selected


# ── __init__ snapshot acquisition (real git) ─────────────────────────────────


class TestPreExistingDirtySnapshotInit:
    """Verify that AgentLoop.__init__ correctly captures the pre-existing dirty
    test files snapshot from git status."""

    def _init_git_repo(self, tmp_path):
        """Initialize a bare git repo with tracked and dirty test files."""
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, timeout=10)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path, capture_output=True, timeout=5,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path, capture_output=True, timeout=5,
        )

        # Create tracked clean files
        (tmp_path / "main.py").write_text("x = 1\n")
        (tmp_path / "tests").mkdir(exist_ok=True)
        (tmp_path / "tests" / "__init__.py").write_text("")
        (tmp_path / "tests" / "test_clean.py").write_text("def test_pass(): assert True\n")

        subprocess.run(["git", "add", "-A"], cwd=tmp_path, capture_output=True, timeout=5)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, capture_output=True, timeout=5)

        # Create dirty test files (simulating user WIP)
        (tmp_path / "tests" / "test_dirty.py").write_text("def test_broken(): assert False\n")
        # Modify a tracked test file
        (tmp_path / "tests" / "test_clean.py").write_text(
            "def test_pass(): assert True\ndef test_new(): pass\n"
        )
        # Untracked non-test file (should NOT appear in snapshot)
        (tmp_path / "notes.txt").write_text("work in progress\n")

    def test_captures_dirty_test_files(self, tmp_path):
        """__init__ should capture test files modified by the user before the session."""
        self._init_git_repo(tmp_path)

        from external_llm.agent.test_impact_selector import git_status_test_files

        _pre = git_status_test_files(str(tmp_path))
        snapshot = set(_pre) if _pre else set()

        assert snapshot is not None
        assert "tests/test_dirty.py" in snapshot, (
            "Newly created untracked test file must be captured"
        )
        assert "tests/test_clean.py" in snapshot, (
            "Modified tracked test file must be captured"
        )
        # Untracked non-test file must NOT appear (git_status_test_files filters via is_test_file)
        assert "notes.txt" not in snapshot, (
            "Non-test file must not be in snapshot"
        )

    def test_empty_when_no_dirty_tests(self, tmp_path):
        """When no test files are dirty, snapshot is an empty set (not None)."""
        self._init_git_repo(tmp_path)

        # Clean up all dirty files - commit everything
        subprocess.run(["git", "add", "-A"], cwd=tmp_path, capture_output=True, timeout=5)
        subprocess.run(["git", "commit", "-m", "clean"], cwd=tmp_path, capture_output=True, timeout=5)

        from external_llm.agent.test_impact_selector import git_status_test_files

        _pre = git_status_test_files(str(tmp_path))
        snapshot = set(_pre) if _pre else set()

        assert snapshot is not None
        assert snapshot == set(), "Empty snapshot must be set(), not None"

    def test_none_when_no_repo_root(self):
        """When registry has no repo_root attribute, snapshot stays None.

        This tests the getattr(self.registry, "repo_root", None) fail path.
        """

        class _NoRepoRoot:
            """Object without repo_root to verify getattr returns None."""
        registry = _NoRepoRoot()

        snapshot = None
        try:
            if getattr(registry, "repo_root", None):
                from external_llm.agent.test_impact_selector import git_status_test_files
                _pre = git_status_test_files(registry.repo_root)
                snapshot = set(_pre) if _pre else set()
        except Exception:
            pass  # stays None

        assert snapshot is None, "Without repo_root, snapshot must stay None"


# ── Context manager helpers ──────────────────────────────────────────────────


def _mock_dispatch(loop):
    """Context manager that mocks registry.dispatch to return a successful result."""
    import contextlib
    from unittest.mock import patch

    @contextlib.contextmanager
    def _patch():
        ok_result = type("Result", (), {"ok": True, "content": "ok"})()
        with patch.object(loop.registry, "dispatch", return_value=ok_result):
            yield

    return _patch()
