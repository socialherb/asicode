"""Dispatch write-safety parity tests — every disk-changing success path must
funnel through the central post-success point (``_after_write_success``).

Regression pin for the repair/soft-fail early returns in ``_dispatch_impl``
that used to skip cache invalidation + Undo checkpoint confirmation entirely:
  * edit_file indent auto-repair success
  * argument-mismatch repair success
  * soft-fail keep-changes success
Each of them wrote the repaired/kept file to disk and returned without
invalidating the file/walk/symbol/RAG/graph caches, the tool-result cache, or
confirming the write to the run checkpoint — the same stale-cache class as the
apply_patch incident the central post-success point was built to prevent, plus
run-created files without an Undo tombstone.

The rollback path (ok=False, disk restored to the pre-write state the caches
already hold) must NOT run it — pinned here so a refused write never leaves a
tombstone behind for Undo to act on.
"""

from __future__ import annotations

import pytest

from external_llm.agent.tool_registry import AgentConfig, ToolRegistry, ToolResult


class _SeqVerify:
    """Stateful stand-in for ``_verify_after_write``: (ok, detail) per call, last repeats."""

    def __init__(self, *results):
        self._results = list(results)
        self.calls = 0

    def __call__(self, snapshots, _post_contents=None):
        idx = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[idx]


@pytest.fixture
def registry(tmp_path):
    # A .git dir makes _detect_repo_language skip the subprocess git probe.
    (tmp_path / ".git").mkdir()
    cfg = AgentConfig(
        parallel_tool_execution_enabled=False,
        tool_result_cache_enabled=False,
    )
    reg = ToolRegistry(str(tmp_path), cfg)
    # Handler stub: dispatch's write-safety cycle is the unit under test, not
    # edit_file itself. It returns ok=True without touching the file system.
    reg._tool_edit_file = lambda args: ToolResult(ok=True, content="edited")
    # Snapshot stub: one existing target file with pre-write content.
    reg._snapshot_target_files = lambda tool_name, args: {str(tmp_path / "f.py"): "def foo():\n    pass\n"}
    # Avoid spawning ruff inside the semantic auto-repair phase.
    reg._safety_manager.auto_repair_semantic = lambda snapshots: 0
    return reg


def _instrument(reg):
    calls = []
    reg._invalidate_cache_after_write = lambda paths: calls.append(("invalidate", list(paths)))
    reg._checkpoint_after_write = lambda tool_name, args: calls.append(("checkpoint", tool_name, args))
    return calls


def _edit_file_args():
    return {"path": "f.py", "operations": [{"type": "replace", "anchor": "def", "content": "x"}]}


def _assert_post_success_ran(calls):
    assert any(c[0] == "invalidate" for c in calls), "_invalidate_cache_after_write must run on this success path"
    assert any(c[0] == "checkpoint" for c in calls), "_checkpoint_after_write must run on this success path"


class TestIndentRepairParity:
    def test_indent_repair_success_runs_post_write_success(self, registry):
        calls = _instrument(registry)
        registry._verify_after_write = _SeqVerify((False, "f.py:1:1: syntax error"), (True, ""))
        registry._auto_repair_indent = lambda orig, ops: "fixed\n"

        result = registry.dispatch("edit_file", _edit_file_args())

        assert result.ok
        assert "Auto-repaired indentation" in result.content
        _assert_post_success_ran(calls)


class TestArgRepairParity:
    def test_arg_repair_success_runs_post_write_success(self, registry):
        calls = _instrument(registry)
        registry._verify_after_write = _SeqVerify((False, "f.py:1:1: bad args"))
        registry._auto_repair_indent = lambda orig, ops: None
        registry._repair_verify_failure = lambda snapshots: True

        result = registry.dispatch("edit_file", _edit_file_args())

        assert result.ok
        assert result.metadata.get("repaired_args") is True
        _assert_post_success_ran(calls)


class TestSoftFailParity:
    def test_soft_fail_keep_runs_post_write_success(self, registry):
        calls = _instrument(registry)
        registry._verify_after_write = _SeqVerify((False, "f.py:1:1: type mismatch"))
        registry._auto_repair_indent = lambda orig, ops: None
        registry._repair_verify_failure = lambda snapshots: False
        registry._should_soft_fail_verify = lambda detail, snaps: True

        result = registry.dispatch("edit_file", _edit_file_args())

        assert result.ok
        assert result.metadata.get("verify_warning") == "f.py:1:1: type mismatch"
        _assert_post_success_ran(calls)


class TestRollbackParity:
    def test_rollback_skips_post_write_success(self, registry):
        """Refused write: disk restored to pre-write state, caches already hold it,
        and no tombstone may be left for Undo to delete a file the user made."""
        calls = _instrument(registry)
        registry._verify_after_write = _SeqVerify((False, "f.py:1:1: syntax error"))
        registry._auto_repair_indent = lambda orig, ops: None
        registry._repair_verify_failure = lambda snapshots: False
        registry._should_soft_fail_verify = lambda detail, snaps: False

        result = registry.dispatch("edit_file", _edit_file_args())

        assert not result.ok
        assert "ROLLBACK" in (result.error or "")
        assert calls == [], (
            "rollback must not run the post-success block: caches still hold the "
            "pre-write state and a refused write must not confirm tombstones"
        )


class TestNormalSuccessParity:
    def test_verify_ok_runs_post_write_success(self, registry):
        calls = _instrument(registry)
        registry._verify_after_write = _SeqVerify((True, ""))

        result = registry.dispatch("edit_file", _edit_file_args())

        assert result.ok
        assert result.content == "edited"
        _assert_post_success_ran(calls)


class _FakeGate:
    """RunCheckpointGate stand-in: enabled + stable checkpoint_id."""

    enabled = True
    checkpoint_id = "ck-42"

    def before_write(self, targets):
        pass

    def confirm_writes(self, targets):
        pass


class TestCheckpointIdMetadata:
    """F1: every disk-changing success path must expose the run's Undo
    checkpoint id in ``result.metadata`` — and the rollback path must not."""

    def test_success_metadata_exposes_run_checkpoint_id(self, registry):
        calls = _instrument(registry)
        registry._run_checkpoint_gate = _FakeGate()
        registry._verify_after_write = _SeqVerify((True, ""))
        registry._auto_repair_indent = lambda orig, ops: None

        result = registry.dispatch("edit_file", _edit_file_args())

        assert result.ok
        assert result.metadata.get("checkpoint_id") == "ck-42", (
            "a successful write must tell callers which Undo checkpoint to use"
        )
        _assert_post_success_ran(calls)

    def test_rollback_metadata_has_no_checkpoint_id(self, registry):
        calls = _instrument(registry)
        registry._run_checkpoint_gate = _FakeGate()
        registry._verify_after_write = _SeqVerify((False, "f.py:1:1: syntax error"))
        registry._auto_repair_indent = lambda orig, ops: None
        registry._repair_verify_failure = lambda snapshots: False
        registry._should_soft_fail_verify = lambda detail, snaps: False

        result = registry.dispatch("edit_file", _edit_file_args())

        assert not result.ok
        assert result.metadata.get("checkpoint_id") is None, (
            "a refused write (disk restored) must not advertise an Undo checkpoint"
        )
        assert not any(c[0] == "invalidate" for c in calls)
