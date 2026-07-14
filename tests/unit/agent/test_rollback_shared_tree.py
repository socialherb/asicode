"""Regression: write-tool rollback must not destroy a concurrent session's edits.

These tests guard two isolation holes that previously existed:

1. `_rollback_patches` (agent_loop) fell back to `git restore --source=HEAD` when
   the primary `git apply -R` failed. In a shared working tree (multi-agent
   orchestration / webapp thread pool) the primary fails *because another
   session edited the same file* — and the fallback then wiped that other
   session's change along with this session's. The fix removes the destructive
   fallback and surfaces a non-destructive `needs_manual_rollback` result.

2. The webapp single-agent path never injected a `FileLockManager`, so two
   concurrent webapp sessions editing the same file raced (the snapshot-based
   write-safety rollback could overwrite one session's change). The fix injects
   a `FileLockManager` into the webapp `AgentConfig`.
"""
import subprocess
from pathlib import Path
from unittest.mock import Mock

from external_llm.agent.agent_loop import AgentLoop
from external_llm.agent.tool_registry import AgentConfig, ToolRegistry


def _run(cmd, cwd, **kw):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, **kw)


def _make_loop(tmp_path: str) -> AgentLoop:
    """Build a minimal AgentLoop over a real git repo at tmp_path."""
    repo = Path(tmp_path)
    _run(["git", "init", "-q"], cwd=str(repo))
    _run(["git", "config", "user.email", "t@t.com"], cwd=str(repo))
    _run(["git", "config", "user.name", "t"], cwd=str(repo))
    # base file
    (repo / "f.txt").write_text("alpha=1\nbeta=2\ngamma=3\n")
    _run(["git", "add", "f.txt"], cwd=str(repo))
    _run(["git", "commit", "-qm", "base"], cwd=str(repo))

    client = Mock()
    client.get_provider_name.return_value = "openai"
    client.provider = "openai"
    cfg = AgentConfig(max_turns=1, planning_enabled=False, rag_enabled=False)
    reg = ToolRegistry(str(repo), cfg)
    loop = AgentLoop(llm_client=client, registry=reg, config=cfg, model="test")
    return loop


def test_rollback_preserves_concurrent_session_edit(tmp_path):
    """Primary reverse fails (file moved) → must NOT restore-to-HEAD.

    Scenario: session1 records a patch on `alpha`, then a concurrent session2
    edits `beta` (different region). Calling `_rollback_patches([session1_patch])`
    must leave session2's `beta` change intact, and must surface
    `needs_manual_rollback` rather than silently wiping the file.
    """
    loop = _make_loop(tmp_path)
    repo = Path(tmp_path)

    # --- session1: edit alpha, capture the recorded (forward) patch ---
    (repo / "f.txt").write_text("alpha=100\nbeta=2\ngamma=3\n")
    session1_patch = _run(["git", "diff"], cwd=str(repo)).stdout
    assert "alpha=100" in session1_patch

    # --- concurrent session2: edit a DIFFERENT region (beta) ---
    (repo / "f.txt").write_text("alpha=100\nbeta=200\ngamma=3\n")

    # --- session1 cancels / errors → rollback its patch ---
    result = loop._rollback_patches([session1_patch])

    # session2's beta change MUST survive (the whole point of the fix)
    assert "beta=200" in (repo / "f.txt").read_text(), \
        "rollback destroyed a concurrent session's edit on a shared file"

    # rollback must report it could NOT auto-rollback this patch
    assert result["success"] is False
    per_patch = result["results"][0]
    assert per_patch["success"] is False
    assert per_patch.get("needs_manual_rollback") is True
    assert "f.txt" in per_patch.get("affected_files", [])
    # the destructive fallback marker must be gone
    assert "used_fallback" not in per_patch


def test_rollback_clean_reverse_still_works(tmp_path):
    """When the file hasn't been touched by another session, the primary
    `git apply -R` must still succeed and fully revert the patch."""
    loop = _make_loop(tmp_path)
    repo = Path(tmp_path)

    (repo / "f.txt").write_text("alpha=100\nbeta=2\ngamma=3\n")
    session1_patch = _run(["git", "diff"], cwd=str(repo)).stdout

    # No concurrent edit — file is exactly at session1's state.
    result = loop._rollback_patches([session1_patch])

    assert result["success"] is True
    assert (repo / "f.txt").read_text() == "alpha=1\nbeta=2\ngamma=3\n"


def test_rollback_no_patches(tmp_path):
    """Empty patch list short-circuits with success."""
    loop = _make_loop(tmp_path)
    result = loop._rollback_patches([])
    assert result == {"success": True, "message": "No patches to rollback", "rolled_back": 0}


def test_rollback_result_metadata_reachable_by_handlers(tmp_path):
    """The cancel/error handlers consume rollback_result['success'] and
    ['rolled_back']/['total']; confirm the non-destructive path still emits
    these keys so the handlers' logging/reporting do not KeyError."""
    loop = _make_loop(tmp_path)
    repo = Path(tmp_path)

    (repo / "f.txt").write_text("alpha=100\nbeta=2\ngamma=3\n")
    session1_patch = _run(["git", "diff"], cwd=str(repo)).stdout
    (repo / "f.txt").write_text("alpha=100\nbeta=200\ngamma=3\n")  # concurrent

    r = loop._rollback_patches([session1_patch])
    for key in ("success", "rolled_back", "total", "results"):
        assert key in r, f"handler relies on rollback_result[{key!r}]"
    assert r["rolled_back"] == 0
    assert r["total"] == 1


# ─── Plan C: webapp injects FileLockManager ───────────────────────────────────

def test_webapp_injects_file_lock_manager():
    """The webapp route must inject a FileLockManager into AgentConfig so that
    concurrent user sessions editing the same file get per-file mutual exclusion.
    This is a source-level guard: if someone removes the injection, this test
    fails. (We import the module and statically verify the assignment exists.)"""
    import inspect

    import webapp.routes.agent_stream as mod
    src = inspect.getsource(mod)
    assert "FileLockManager" in src, \
        "webapp route lost its FileLockManager import / injection"
    assert "config.file_lock_manager = FileLockManager" in src, \
        "webapp route no longer injects FileLockManager into AgentConfig"


# ─── Plan 1-A: needs_manual_rollback signal surfaces to caller ─────────────────

from external_llm.agent.agent_turn_pipeline import _summarize_rollback


def test_summarize_rollback_no_result():
    """No rollback performed → 'No patches needed rollback' + performed=False."""
    msg, meta = _summarize_rollback(None)
    assert msg == "No patches needed rollback."
    assert meta["performed"] is False
    assert meta["result"] is None
    assert meta["needs_manual_rollback"] is False
    assert meta["affected_files"] == []


def test_summarize_rollback_all_success():
    """Successful full rollback → success message, no manual flag."""
    msg, meta = _summarize_rollback({
        "success": True,
        "rolled_back": 2,
        "total": 2,
        "results": [
            {"success": True, "patch_index": 1, "message": "ok"},
            {"success": True, "patch_index": 0, "message": "ok"},
        ],
    })
    assert "successfully rolled back" in msg
    assert "Manual" not in msg
    assert meta["performed"] is True
    assert meta["needs_manual_rollback"] is False
    assert meta["affected_files"] == []


def test_summarize_rollback_needs_manual_promotes_signal():
    """needs_manual_rollback in per-patch results → promoted to top-level meta
    AND surfaced in the human-readable message with affected file names."""
    msg, meta = _summarize_rollback({
        "success": False,
        "rolled_back": 0,
        "total": 1,
        "results": [{
            "success": False,
            "patch_index": 0,
            "needs_manual_rollback": True,
            "affected_files": ["src/a.py", "src/b.py"],
            "message": "git apply -R failed; aborted to protect concurrent edits",
        }],
    })
    # top-level promotion
    assert meta["needs_manual_rollback"] is True
    assert meta["affected_files"] == ["src/a.py", "src/b.py"]
    # human message must mention the files + manual action
    assert "partially failed" in msg
    assert "src/a.py" in msg and "src/b.py" in msg
    assert "Manual targeted rollback required" in msg


def test_summarize_rollback_dedupes_affected_files():
    """Multiple manual-rollback patches touching the same file → de-duplicated list."""
    _msg, meta = _summarize_rollback({
        "success": False,
        "rolled_back": 0,
        "total": 2,
        "results": [
            {"success": False, "needs_manual_rollback": True, "affected_files": ["x.py", "y.py"]},
            {"success": False, "needs_manual_rollback": True, "affected_files": ["y.py", "z.py"]},
        ],
    })
    assert meta["needs_manual_rollback"] is True
    assert meta["affected_files"] == ["x.py", "y.py", "z.py"]


def test_summarize_rollback_partial_without_manual_signal():
    """Partial failure WITHOUT needs_manual_rollback (e.g. transient git error) →
    manual flag stays False, message is the plain partial-failure text."""
    msg, meta = _summarize_rollback({
        "success": False,
        "rolled_back": 1,
        "total": 2,
        "results": [
            {"success": True, "patch_index": 1},
            {"success": False, "patch_index": 0, "message": "Exception during rollback: boom"},
        ],
    })
    assert meta["needs_manual_rollback"] is False
    assert meta["affected_files"] == []
    assert "1/2" in msg
    assert "Manual" not in msg
