"""Tests for scope_violation_policy (warn / revert / fail).

Verifies the genuine out-of-scope detection + policy application added to both
the IPC and in-process completion paths. Helper methods are exercised directly
against a real (tmp_path) git repository so tracked/untracked revert semantics
are verified end-to-end, plus the filtering (baseline / assigned / infra
subtraction) that makes the reverted set safe.
"""
from __future__ import annotations

import logging
import os
import subprocess
import types

from unittest.mock import Mock

import pytest

from external_llm.agent.orchestrator import OrchestratorAgent, OrchestratorConfig


def _git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True,
    )


@pytest.fixture
def repo(tmp_path):
    """A throwaway git repo with two committed tracked files."""
    r = str(tmp_path)
    _git(r, "init")
    _git(r, "config", "user.email", "t@t.com")
    _git(r, "config", "user.name", "tester")
    (tmp_path / "assigned.txt").write_text("assigned-original\n")
    (tmp_path / "tracked_other.txt").write_text("tracked-original\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-m", "init")
    return r


def _make_orch(policy="warn"):
    orch = OrchestratorAgent(
        Mock(), Mock(), OrchestratorConfig(scope_violation_policy=policy),
    )
    orch._baseline_dirty_paths = set()
    orch._global_assigned_paths = set()
    return orch


def _result(status="success", final_message="ok"):
    return types.SimpleNamespace(status=status, final_message=final_message)


# ── _revert_unassigned_changes ──────────────────────────────────────────


class TestRevertUnassignedChanges:
    def test_tracked_file_restored_to_head(self, repo):
        # worker edits a TRACKED file outside its assignment
        with open(os.path.join(repo, "tracked_other.txt"), "w") as f:
            f.write("worker-stray-edit\n")
        orch = _make_orch("revert")
        reverted = orch._revert_unassigned_changes(
            repo, [{"file": "tracked_other.txt"}],
        )
        assert reverted == ["tracked_other.txt"]
        with open(os.path.join(repo, "tracked_other.txt")) as f:
            assert f.read() == "tracked-original\n"

    def test_untracked_file_deleted(self, repo):
        # worker creates a NEW untracked file outside its assignment
        with open(os.path.join(repo, "stray_new.txt"), "w") as f:
            f.write("created-during-run\n")
        orch = _make_orch("revert")
        reverted = orch._revert_unassigned_changes(
            repo, [{"file": "stray_new.txt"}],
        )
        assert reverted == ["stray_new.txt"]
        assert not os.path.exists(os.path.join(repo, "stray_new.txt"))

    def test_untracked_directory_deleted(self, repo):
        d = os.path.join(repo, "stray_dir")
        os.makedirs(d)
        with open(os.path.join(d, "f.txt"), "w") as f:
            f.write("x\n")
        orch = _make_orch("revert")
        reverted = orch._revert_unassigned_changes(repo, [{"file": "stray_dir"}])
        assert reverted == ["stray_dir"]
        assert not os.path.exists(d)

    def test_accepts_plain_string_entries(self, repo):
        # entries may be plain strings, not just dicts
        with open(os.path.join(repo, "stray_s.txt"), "w") as f:
            f.write("y\n")
        orch = _make_orch("revert")
        reverted = orch._revert_unassigned_changes(repo, ["stray_s.txt"])
        assert reverted == ["stray_s.txt"]

    def test_failed_revert_skipped_not_raised(self, repo):
        # a path that is neither tracked nor on disk must not raise
        orch = _make_orch("revert")
        reverted = orch._revert_unassigned_changes(
            repo, [{"file": "nonexistent.xyz"}],
        )
        assert reverted == []


# ── _apply_scope_violation_policy ────────────────────────────────────────


class TestApplyScopeViolationPolicy:
    def test_empty_genuine_is_noop(self, repo):
        orch = _make_orch("revert")
        res = _result()
        assert orch._apply_scope_violation_policy(
            repo, [], res, agent_id="s1", mode="ipc",
        ) == []
        assert res.status == "success"

    def test_warn_leaves_files_and_status(self, repo):
        with open(os.path.join(repo, "stray.txt"), "w") as f:
            f.write("z\n")
        orch = _make_orch("warn")
        res = _result()
        out = orch._apply_scope_violation_policy(
            repo, [{"file": "stray.txt"}], res, agent_id="s1", mode="ipc",
        )
        assert out == []                                   # warn returns nothing
        assert res.status == "success"                     # unchanged
        assert os.path.exists(os.path.join(repo, "stray.txt"))  # file left in place

    def test_revert_restores_and_returns_list(self, repo):
        with open(os.path.join(repo, "stray.txt"), "w") as f:
            f.write("z\n")
        orch = _make_orch("revert")
        res = _result()
        out = orch._apply_scope_violation_policy(
            repo, [{"file": "stray.txt"}], res, agent_id="s1", mode="ipc",
        )
        assert out == ["stray.txt"]
        assert not os.path.exists(os.path.join(repo, "stray.txt"))

    def test_fail_promotes_to_error(self, repo):
        with open(os.path.join(repo, "stray.txt"), "w") as f:
            f.write("z\n")
        orch = _make_orch("fail")
        res = _result(final_message="did work")
        out = orch._apply_scope_violation_policy(
            repo, [{"file": "stray.txt"}], res, agent_id="s1", mode="ipc",
        )
        assert out == []                                  # fail does not revert
        assert res.status == "error"
        assert "scope_violation" in res.final_message

    def test_warn_logs_out_of_scope_signal(self, repo, caplog):
        with open(os.path.join(repo, "stray.txt"), "w") as f:
            f.write("z\n")
        orch = _make_orch("warn")
        with caplog.at_level(logging.WARNING):
            orch._apply_scope_violation_policy(
                repo, [{"file": "stray.txt"}], _result(),
                agent_id="s1", mode="ipc",
            )
        assert any("OUT-OF-SCOPE" in r.message for r in caplog.records)


# ── _detect_genuine_violations / filtering ──────────────────────────────


class TestDetectGenuineViolations:
    def test_detect_from_git_status_in_process(self, repo):
        # in-process path: raw_unassigned=None derives from `git status`
        with open(os.path.join(repo, "stray.txt"), "w") as f:
            f.write("new\n")
        # also dirty an assigned file — must be EXCLUDED
        with open(os.path.join(repo, "assigned.txt"), "w") as f:
            f.write("assigned-edited\n")
        orch = _make_orch("warn")
        genuine = orch._detect_genuine_violations(
            repo, ["assigned.txt"],   # assigned.txt is this worker's own file
        )
        files = {g["file"] for g in genuine}
        assert "stray.txt" in files
        assert "assigned.txt" not in files      # own assignment excluded

    def test_baseline_dirt_excluded(self, repo):
        # pre-run dirt captured in _baseline_dirty_paths is excluded
        with open(os.path.join(repo, "env_dirt.txt"), "w") as f:
            f.write("dirty-before-run\n")
        orch = _make_orch("warn")
        orch._baseline_dirty_paths = {os.path.normpath("env_dirt.txt")}
        genuine = orch._detect_genuine_violations(repo, [])
        files = {g["file"] for g in genuine}
        assert "env_dirt.txt" not in files

    def test_infra_paths_excluded(self, repo):
        os.makedirs(os.path.join(repo, ".asicode", "subagents"), exist_ok=True)
        with open(os.path.join(repo, ".asicode", "subagents", "worker.log"), "w") as f:
            f.write("log\n")
        orch = _make_orch("warn")
        genuine = orch._detect_genuine_violations(repo, [])
        files = {g["file"] for g in genuine}
        # nothing infra should appear as a genuine violation
        assert all(".asicode" not in fp for fp in files)

    def test_git_status_rename_no_ghost_path(self, repo):
        """A ``git mv`` rename (porcelain -z emits a second NUL field for the
        source) must yield ONLY the destination — never a phantom record carved
        from the source path's first 3 chars (the pre-fix ``[3:]`` bug)."""
        # tracked_other.txt is committed in the `repo` fixture; rename it.
        _git(repo, "mv", "tracked_other.txt", "renamed.txt")
        orch = _make_orch("warn")
        changed = orch._git_status_changed_paths(repo)
        files = {g["file"] for g in changed}
        # Destination is the genuine changed path.
        assert "renamed.txt" in files
        # The source must NOT appear, nor a 3-char-mangled ghost of it
        # ("cked_other.txt" from the old ``record[3:]`` mis-parse).
        assert "tracked_other.txt" not in files
        assert all(not f.endswith("cked_other.txt") for f in files), files


class TestRevertBatching:
    """Multiple strays must be handled with O(1) git subprocesses — ONE
    ``ls-files`` for the whole batch + ONE batched ``checkout`` for the tracked
    subset — not the O(n) per-file ``ls-files --error-unmatch`` + ``checkout``
    pair the previous implementation spawned."""

    def test_batch_tracked_and_untracked(self, repo, monkeypatch):
        import subprocess as _sp
        # tracked stray: overwrite the committed tracked file
        with open(os.path.join(repo, "tracked_other.txt"), "w") as f:
            f.write("stray\n")
        # untracked strays
        for name in ("new_a.txt", "new_b.txt", "new_c.txt"):
            with open(os.path.join(repo, name), "w") as f:
                f.write("created\n")

        git_calls = []
        real_run = _sp.run

        def _spy(args, *a, **k):
            if args[:1] == ["git"]:
                git_calls.append(args[1] if len(args) > 1 else "")
            return real_run(args, *a, **k)
        monkeypatch.setattr("subprocess.run", _spy)

        orch = _make_orch("revert")
        reverted = orch._revert_unassigned_changes(
            repo,
            [
                {"file": "tracked_other.txt"},
                {"file": "new_a.txt"},
                {"file": "new_b.txt"},
                {"file": "new_c.txt"},
            ],
        )

        assert set(reverted) == {"tracked_other.txt", "new_a.txt", "new_b.txt", "new_c.txt"}
        # tracked file restored to HEAD content
        assert open(os.path.join(repo, "tracked_other.txt")).read() == "tracked-original\n"
        # untracked files removed
        for name in ("new_a.txt", "new_b.txt", "new_c.txt"):
            assert not os.path.exists(os.path.join(repo, name))
        # ONE ls-files (batch tracked determination) + ONE checkout (batch revert).
        # The previous impl spawned 2 git subprocesses PER tracked file.
        assert git_calls.count("ls-files") == 1
        assert git_calls.count("checkout") == 1

    def test_batch_checkout_failure_falls_back_per_file(self, repo, monkeypatch):
        """If the batched checkout fails, revert falls back to per-file so one
        bad path doesn't abort the whole batch (correctness over batching)."""
        with open(os.path.join(repo, "tracked_other.txt"), "w") as f:
            f.write("stray\n")

        call_log = []
        real_run = subprocess.run

        def _spy(args, *a, **k):
            if args[:1] == ["git"] and args[1:2] == ["checkout"]:
                call_log.append(args)
                # Simulate batch checkout failure (e.g. one path missing) → still
                # produce a non-zero rc; per-file fallback must then run.
                return types.SimpleNamespace(returncode=1, stdout=b"", stderr=b"boom")
            return real_run(args, *a, **k)
        monkeypatch.setattr("subprocess.run", _spy)

        orch = _make_orch("revert")
        reverted = orch._revert_unassigned_changes(repo, [{"file": "tracked_other.txt"}])
        # batch attempted, then per-file attempted → at least 2 checkout calls
        assert sum(1 for a in call_log if a[1] == "checkout") >= 2
        # both failed (simulated) → nothing reverted, but no exception raised
        assert reverted == []

    def test_batch_nonascii_tracked_restored_not_deleted(self, repo):
        """Regression: ``git ls-files`` C-quotes non-ASCII paths by default
        (``"\\355\\225\\234..."`` for 한글), so a batch tracked determination
        WITHOUT ``-z`` misclassified non-ASCII tracked files as untracked →
        ``unlink``-ed (data loss) instead of ``git checkout``-restored, while
        still falsely reported in the "reverted" list. ``-z`` emits
        NUL-separated raw bytes with no quoting → exact membership match."""
        # Commit a tracked file with a non-ASCII (Korean) name.
        name = "한글파일.txt"
        with open(os.path.join(repo, name), "w") as f:
            f.write("tracked-original\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-m", "add non-ascii")
        # Worker overwrites it out-of-scope.
        with open(os.path.join(repo, name), "w") as f:
            f.write("stray\n")
        orch = _make_orch("revert")
        reverted = orch._revert_unassigned_changes(repo, [{"file": name}])
        # The file must be RESTORED to HEAD content, NOT deleted by unlink.
        assert name in reverted
        assert os.path.exists(os.path.join(repo, name))
        assert open(os.path.join(repo, name)).read() == "tracked-original\n"
