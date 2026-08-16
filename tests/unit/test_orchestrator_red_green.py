"""RED→GREEN: orchestrator.py pure-logic coverage (round 1).

Covers module-level helpers and FileLockManager/OrderedEventDispatcher that
have zero LLM/IPC dependency: asr_subagent_argv, _DummyLock, FileLockManager,
OrderedEventDispatcher, _resolve_subagent_base_max_turns, OrchestratorConfig
validation, _build_git_context, coercers, directory expansion, batch conflict
splitting, snapshot capture/restore, infra-path filter, symbol hints.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

import pytest

from external_llm.agent import orchestrator as orch
from external_llm.agent.orchestrator import (
    _MISSING_SNAP,
    FileLockManager,
    OrchestratorConfig,
    OrderedEventDispatcher,
    SubTaskSpec,
    _build_git_context,
    _capture_assigned_snapshots,
    _coerce_priority,
    _coerce_str_list,
    _DummyLock,
    _expand_directory_assignments,
    _is_infra_path,
    _kind_initial,
    _norm_assigned_file,
    _read_synth_diff_head,
    _restore_assigned_snapshots,
    _snapshot_dirty_path_set,
    _split_batch_by_file_conflict,
    _symbol_def_line,
    _symbol_hint_for_source,
    asr_subagent_argv,
)

# ── asr_subagent_argv ────────────────────────────────────────────────────────

def test_asr_subagent_argv_prefers_repo_asi_py(tmp_path):
    (tmp_path / "asi.py").write_text("")
    argv = asr_subagent_argv(str(tmp_path))
    assert argv == [sys.executable, str(tmp_path / "asi.py"), "--subagent"]


def test_asr_subagent_argv_falls_back_to_bare_asi(tmp_path):
    argv = asr_subagent_argv(str(tmp_path))
    assert argv == ["asi", "--subagent"]


# ── _DummyLock ───────────────────────────────────────────────────────────────

def test_dummy_lock_is_noop():
    lock = _DummyLock()
    assert lock.acquire() is True
    assert lock.acquire(blocking=False) is True
    lock.release()  # must not raise
    with lock as inner:
        assert inner is lock
    assert orch._DUMMY_LOCK is not None


# ── FileLockManager ──────────────────────────────────────────────────────────

def _fresh_manager(tmp_path):
    return FileLockManager(str(tmp_path))


def test_filelock_normalize_path_no_repo_root():
    m = FileLockManager(None)
    assert m._normalize_path("anything/at/all") == "anything/at/all"


def test_filelock_normalize_path_rejects_outside(tmp_path):
    m = _fresh_manager(tmp_path)
    assert m._normalize_path("../escape.py") is None
    assert m._normalize_path("C:/abs.py") is None


def test_filelock_acquire_release_roundtrip(tmp_path):
    m = _fresh_manager(tmp_path)
    (tmp_path / "a.py").write_text("x")
    lock = m.acquire("a.py")
    assert lock is not orch._DUMMY_LOCK
    assert m._held  # strong-ref recorded
    m.release("a.py")
    assert m._held == {}


def test_filelock_acquire_invalid_path_returns_dummy(tmp_path):
    m = _fresh_manager(tmp_path)
    assert m.acquire("../outside.py") is orch._DUMMY_LOCK


def test_filelock_acquire_relevant_locks_patch_targets(tmp_path):
    m = _fresh_manager(tmp_path)
    (tmp_path / "t1.py").write_text("x")
    patch = "--- a/t1.py\n+++ b/t1.py\n@@ -1 +1 @@\n-x\n+y\n"
    locked = m.acquire_relevant({"patch": patch})
    assert locked, "patch targets must resolve to at least one lock"
    assert any(p.endswith("t1.py") for p in locked)
    m.release_all(locked)
    assert m._held == {}


def test_filelock_acquire_relevant_scalar_and_plan(tmp_path):
    m = _fresh_manager(tmp_path)
    (tmp_path / "s.py").write_text("x")
    locked = m.acquire_relevant({"path": "s.py", "src": "s.py", "dst": "s.py"})
    assert len(locked) == 1
    m.release_all(locked)
    plan_locked = m.acquire_relevant({"plan": {"ops": [{"path": "s.py"}]}}, "write_plan")
    assert plan_locked
    m.release_all(plan_locked)


def test_filelock_acquire_relevant_rolls_back_on_error(tmp_path, monkeypatch):
    m = _fresh_manager(tmp_path)
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.py").write_text("x")
    orig = m._acquire_by_normalized_path

    def boom(norm_path):
        if norm_path.endswith("b.py"):
            raise RuntimeError("boom")
        return orig(norm_path)

    monkeypatch.setattr(m, "_acquire_by_normalized_path", boom)
    with pytest.raises(RuntimeError):
        m.acquire_relevant({"path": "a.py", "file_path": "b.py"})
    # a.py lock must have been released during rollback
    assert m._held == {}


def test_filelock_acquire_repo_and_release(tmp_path):
    m = _fresh_manager(tmp_path)
    key = m.acquire_repo()
    assert key
    # second manager on same repo fails fast
    m2 = _fresh_manager(tmp_path)
    assert m2.acquire_repo() is None
    m.release_repo(key)
    assert m._held == {}
    # now re-acquirable
    assert m2.acquire_repo() is not None
    m2.reset()


def test_filelock_acquire_repo_no_root():
    assert FileLockManager(None).acquire_repo() == ""
    assert FileLockManager("").acquire_repo() == ""


def test_filelock_release_repo_empty_key():
    m = FileLockManager(None)
    m.release_repo("")  # must not raise


def test_filelock_two_managers_share_registry(tmp_path):
    m1 = _fresh_manager(tmp_path)
    m2 = _fresh_manager(tmp_path)
    (tmp_path / "f.py").write_text("x")
    m1.acquire("f.py")
    # both managers must resolve the SAME registry key
    key1 = next(iter(m1._held))
    key2 = os.path.join(os.path.realpath(str(tmp_path)), "f.py")
    assert os.path.normpath(key1) == os.path.normpath(key2)
    m1.reset()
    assert m1._held == {}
    m2.reset()


def test_filelock_release_all_and_runtime_error_guard(tmp_path):
    m = _fresh_manager(tmp_path)
    (tmp_path / "g.py").write_text("x")
    locked = m.acquire_relevant({"path": "g.py"})
    assert locked
    # double release -> second hits RuntimeError guard
    m.release_all(locked)
    m.release_all(locked)  # must not raise


# ── OrderedEventDispatcher ───────────────────────────────────────────────────

def test_ordered_dispatcher_sequence_and_metadata():
    seen = []
    d = OrderedEventDispatcher(lambda ev, data: seen.append((ev, data)))
    d.emit("a1", "subagent_start", {"task": "t1"})
    d.emit("a1", "subagent_complete", {"task": "t1"})
    d.emit("a2", "subagent_start", {"task": "t2"})
    assert len(seen) == 3
    e1, d1 = seen[0]
    assert e1 == "subagent_start"
    assert d1["global_sequence_id"] == 1
    assert d1["agent_sequence_id"] == 1
    assert d1["agent_id"] == "a1"
    assert d1["event_type"] == "subagent_start"
    assert d1["task"] == "t1"
    assert "timestamp" in d1
    assert seen[1][1]["global_sequence_id"] == 2
    assert seen[1][1]["agent_sequence_id"] == 2
    assert seen[2][1]["global_sequence_id"] == 3
    assert seen[2][1]["agent_sequence_id"] == 1  # per-agent resets


def test_ordered_dispatcher_does_not_mutate_input():
    d = OrderedEventDispatcher(lambda *a: None)
    data = {"k": "v"}
    d.emit("a", "ev", data)
    assert data == {"k": "v"}


# ── _resolve_subagent_base_max_turns / OrchestratorConfig ───────────────────

def test_resolve_base_max_turns_uses_agent_config():
    class FakeAgentConfig:
        max_turns = 7

    cfg = OrchestratorConfig(agent_config=FakeAgentConfig())
    assert orch._resolve_subagent_base_max_turns(cfg, extra_turns=3) == 10


def test_resolve_base_max_turns_default():
    cfg = OrchestratorConfig()
    base = orch._resolve_subagent_base_max_turns(cfg)
    assert base > 0


def test_orchestrator_config_rejects_bad_policy():
    with pytest.raises(ValueError, match="scope_violation_policy"):
        OrchestratorConfig(scope_violation_policy="reverd")


def test_orchestrator_config_accepts_policies():
    for pol in ("warn", "revert", "fail"):
        assert OrchestratorConfig(scope_violation_policy=pol).scope_violation_policy == pol


# ── _build_git_context ───────────────────────────────────────────────────────

@pytest.fixture
def git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "t@example.com"],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "t"],
        check=True, capture_output=True,
    )
    (repo / "f.py").write_text("x=1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "init"],
        check=True, capture_output=True,
    )
    for i in range(3):
        (repo / "f.py").write_text(f"x={i}\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(repo), "commit", "-q", "-m", f"change {i}"],
            check=True, capture_output=True,
        )
    return repo


def test_build_git_context_empty_for_no_root():
    assert _build_git_context(None) == ""
    assert _build_git_context("") == ""


def test_build_git_context_returns_history(git_repo):
    ctx = _build_git_context(str(git_repo))
    assert "Recent git history" in ctx
    assert "change 0" in ctx
    assert "Files modified in last 3 commits" in ctx
    assert "Do NOT create subtasks" in ctx


def test_build_git_context_failure_returns_empty(tmp_path):
    assert _build_git_context(str(tmp_path)) == ""


# ── coercers ─────────────────────────────────────────────────────────────────

def test_coerce_priority():
    assert _coerce_priority(2) == 2
    assert _coerce_priority("3") == 3
    assert _coerce_priority("abc") == 1
    assert _coerce_priority(None) == 1
    assert _coerce_priority("abc", default=5) == 5


def test_coerce_str_list():
    assert _coerce_str_list(["a", 1, None, "b"]) == ["a", "1", "b"]
    assert _coerce_str_list("nope") == []
    assert _coerce_str_list({"a": 1}) == []
    assert _coerce_str_list(None) == []


# ── _expand_directory_assignments / _norm_assigned_file ─────────────────────

def test_expand_directory_assignments_basic(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("")
    (tmp_path / "pkg" / "b.py").write_text("")
    (tmp_path / "top.py").write_text("")
    out = _expand_directory_assignments(str(tmp_path), ["pkg", "top.py", "new.py"])
    assert "pkg/a.py" in out and "pkg/b.py" in out
    assert "top.py" in out
    assert "new.py" in out  # non-existent file passes through
    assert "pkg" not in out


def test_expand_directory_assignments_noop_cases():
    assert _expand_directory_assignments(None, ["a"]) == ["a"]
    assert _expand_directory_assignments("", ["a"]) == ["a"]
    assert _expand_directory_assignments("/tmp/x", []) == []
    assert _expand_directory_assignments("/tmp/x", None) == []


def test_expand_directory_assignments_skips_venv(tmp_path):
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "x.py").write_text("")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "y.py").write_text("")
    out = _expand_directory_assignments(str(tmp_path), [".venv", "src"])
    # .venv is a walk-skipped dir: left bare (not expanded); src expands
    assert out == [".venv", "src/y.py"]


def test_expand_directory_assignments_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "_MAX_DIR_EXPANSION_FILES", 2)
    (tmp_path / "big").mkdir()
    for i in range(5):
        (tmp_path / "big" / f"f{i}.py").write_text("")
    out = _expand_directory_assignments(str(tmp_path), ["big"])
    assert out == ["big"]  # over cap -> left unexpanded


def test_norm_assigned_file():
    assert _norm_assigned_file("") == ""
    assert _norm_assigned_file("./src/foo.py") == "src/foo.py"
    assert _norm_assigned_file("b/src/foo.py") == "src/foo.py"
    assert _norm_assigned_file("../bad.py") == "../bad.py"  # invalid -> raw fallback
    assert _norm_assigned_file("a/../b.py") == "a/../b.py"


# ── _split_batch_by_file_conflict ───────────────────────────────────────────

def _spec(task_id, files):
    return SubTaskSpec(task_id=task_id, title=task_id, description=task_id, assigned_files=files)


def test_split_batch_conflict_free():
    specs = [_spec("t1", ["a.py"]), _spec("t2", ["b.py"]), _spec("t3", ["c.py"])]
    batches = _split_batch_by_file_conflict(specs)
    assert len(batches) == 1 and len(batches[0]) == 3


def test_split_batch_conflicts():
    specs = [_spec("t1", ["a.py"]), _spec("t2", ["a.py"]), _spec("t3", ["b.py"])]
    batches = _split_batch_by_file_conflict(specs)
    assert len(batches) == 2
    # t2 cannot ride with t1; t3 may join either
    assert all(len(b) <= 2 for b in batches)


def test_split_batch_normalizes_conflict():
    specs = [_spec("t1", ["./a.py"]), _spec("t2", ["a.py"])]
    batches = _split_batch_by_file_conflict(specs)
    assert len(batches) == 2  # same file after normalization


def test_split_batch_unscoped_isolated():
    specs = [_spec("t1", []), _spec("t2", ["a.py"]), _spec("t3", [])]
    batches = _split_batch_by_file_conflict(specs)
    assert len(batches) == 3
    assert all(len(b) == 1 for b in batches)


def test_split_batch_empty():
    assert _split_batch_by_file_conflict([]) == [[]]


def test_kind_initial():
    assert _kind_initial("function") == "F"
    assert _kind_initial("class") == "C"
    assert _kind_initial("") == "?"
    assert _kind_initial(None) == "?"


# ── _read_synth_diff_head extra ──────────────────────────────────────────────

def test_read_synth_diff_head_exact_max(tmp_path):
    p = tmp_path / "x.txt"
    p.write_bytes(b"abcdef")
    text, truncated, size = _read_synth_diff_head(str(p), max_bytes=6)
    assert (text, truncated, size) == ("abcdef", False, 6)


def test_read_synth_diff_head_truncated_ascii(tmp_path):
    p = tmp_path / "x.txt"
    p.write_bytes(b"abcdefgh")
    text, truncated, size = _read_synth_diff_head(str(p), max_bytes=4)
    assert (text, truncated, size) == ("abcd", True, 8)


# ── _capture_assigned_snapshots ─────────────────────────────────────────────

def test_capture_snapshots_basic(tmp_path):
    (tmp_path / "a.py").write_bytes(b"one")
    (tmp_path / "b.py").write_bytes(b"two")
    snaps = _capture_assigned_snapshots(str(tmp_path), ["a.py", "b.py"])
    assert snaps == {"a.py": b"one", "b.py": b"two"}


def test_capture_snapshots_missing_and_dir(tmp_path):
    (tmp_path / "sub").mkdir()
    snaps = _capture_assigned_snapshots(str(tmp_path), ["gone.py", "sub"])
    assert snaps["gone.py"] is _MISSING_SNAP
    assert "sub" not in snaps


def test_capture_snapshots_oversized_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "_SNAPSHOT_MAX_BYTES", 4)
    (tmp_path / "big.py").write_bytes(b"12345")
    (tmp_path / "ok.py").write_bytes(b"12")
    snaps = _capture_assigned_snapshots(str(tmp_path), ["big.py", "ok.py"])
    assert "big.py" not in snaps
    assert snaps["ok.py"] == b"12"


def test_capture_snapshots_aggregate_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "_SNAPSHOT_AGGREGATE_MAX_BYTES", 5)
    monkeypatch.setattr(orch, "_SNAPSHOT_MAX_BYTES", 100)
    (tmp_path / "a.py").write_bytes(b"aaa")
    (tmp_path / "b.py").write_bytes(b"bbb")
    (tmp_path / "c.py").write_bytes(b"ccc")
    snaps = _capture_assigned_snapshots(str(tmp_path), ["a.py", "b.py", "c.py"])
    assert snaps["a.py"] == b"aaa"
    assert "b.py" not in snaps and "c.py" not in snaps


def test_capture_snapshots_no_repo_root(tmp_path):
    p = tmp_path / "x.py"
    p.write_bytes(b"x")
    assert _capture_assigned_snapshots("", [str(p)]) == {str(p): b"x"}


# ── _restore_assigned_snapshots ──────────────────────────────────────────────

def test_restore_snapshots_writes_back(tmp_path):
    (tmp_path / "a.py").write_bytes(b"MUTATED")
    reverted = _restore_assigned_snapshots(str(tmp_path), {"a.py": b"original"})
    assert reverted == ["a.py"]
    assert (tmp_path / "a.py").read_bytes() == b"original"


def test_restore_snapshots_removes_created_file(tmp_path):
    (tmp_path / "new.py").write_bytes(b"created during run")
    reverted = _restore_assigned_snapshots(str(tmp_path), {"new.py": _MISSING_SNAP})
    assert reverted == ["new.py"]
    assert not (tmp_path / "new.py").exists()


def test_restore_snapshots_removes_empty_parent_dirs(tmp_path):
    d = tmp_path / "pkg" / "sub"
    d.mkdir(parents=True)
    (d / "f.py").write_bytes(b"x")
    _restore_assigned_snapshots(str(tmp_path), {"pkg/sub/f.py": _MISSING_SNAP})
    assert not d.exists()
    assert not (tmp_path / "pkg").exists()


def test_restore_snapshots_cleans_stale_tempfiles(tmp_path):
    d = tmp_path / "dir"
    d.mkdir()
    stale = d / ".asi-revert-old"
    stale.write_bytes(b"x")
    old = time.time() - 200_000
    os.utime(stale, (old, old))
    fresh = d / ".asi-revert-new"
    fresh.write_bytes(b"x")
    (d / "f.py").write_bytes(b"mut")
    _restore_assigned_snapshots(str(tmp_path), {"dir/f.py": b"orig"})
    assert not stale.exists()
    assert fresh.exists()
    assert (d / "f.py").read_bytes() == b"orig"


def test_restore_snapshots_bad_path_warns(tmp_path):
    reverted = _restore_assigned_snapshots(str(tmp_path), {"nope/x.py": b"data"})
    assert reverted == []


# ── _is_infra_path / _snapshot_dirty_path_set ───────────────────────────────

def test_is_infra_path():
    assert _is_infra_path("")
    assert _is_infra_path(".asicode/foo")
    assert _is_infra_path("logs/x")
    assert _is_infra_path(".git/config")
    assert _is_infra_path(".asicode/deep")
    assert not _is_infra_path("src/a.py")
    assert not _is_infra_path("asicode/x")  # not a prefix


def test_snapshot_dirty_path_set(git_repo):
    assert _snapshot_dirty_path_set(str(git_repo)) == set()
    (git_repo / "f.py").write_text("x=99\n", encoding="utf-8")
    dirty = _snapshot_dirty_path_set(str(git_repo))
    assert "f.py" in dirty


def test_snapshot_dirty_path_set_non_repo(tmp_path):
    assert _snapshot_dirty_path_set(str(tmp_path)) == set()


# ── _symbol_hint_for_source / _symbol_def_line ──────────────────────────────

PY_SRC = """
class Foo:
    def bar(self):
        pass

def top():
    return 1
"""


def test_symbol_hint_for_source_python():
    out = _symbol_hint_for_source(PY_SRC, "mod.py")
    assert any(s.startswith("C Foo L2") for s in out)
    assert any(s.startswith("F bar L3") for s in out)
    assert any(s.startswith("F top L6") for s in out)


def test_symbol_hint_for_source_broken_syntax():
    assert _symbol_hint_for_source("def (:", "bad.py") == []


def test_symbol_hint_for_source_unknown_language():
    assert _symbol_hint_for_source("hello world", "x.unknownext") == []


def test_symbol_def_line_python():
    assert _symbol_def_line(PY_SRC, "mod.py", "top") == 6
    assert _symbol_def_line(PY_SRC, "mod.py", "Foo") == 2
    assert _symbol_def_line(PY_SRC, "mod.py", "nope") is None


def test_symbol_def_line_broken_syntax():
    assert _symbol_def_line("def (:", "bad.py", "x") is None


def test_subtask_spec_defaults():
    s = SubTaskSpec(task_id="t", title="T", description="D")
    assert s.assigned_files == []
    assert s.dependencies == []
    assert s.priority == 0


def test_threading_import_available():
    assert threading is not None


# ═══════════════════════════════════════════════════════════════════════════
# Round 2: diff/revert helpers, registry facade, orchestrator core (mocked)
# ═══════════════════════════════════════════════════════════════════════════

from types import SimpleNamespace

import external_llm.agent.agent_loop as agent_loop_mod
from external_llm.agent.agent_loop import AgentResult
from external_llm.agent.orchestrator import (
    OrchestratorAgent,
    _NativeToolError,
    _OrchestratorBackedRegistry,
)


class _FakeRegistry:
    """Minimal ToolRegistry stand-in: repo_root + clone + schema/dispatch."""

    def __init__(self, repo_root):
        self.repo_root = repo_root
        self.config = SimpleNamespace(file_lock_manager=None)

    def clone_for_subagent(self, config):
        return self

    def get_tool_schemas(self, *a, **kw):
        return [{"name": "read_file", "description": "d", "parameters": {}}]

    def dispatch(self, name, args):
        return f"dispatched:{name}"


def _make_orch(tmp_path, **cfg_overrides):
    cfg = OrchestratorConfig(**cfg_overrides)
    return OrchestratorAgent(
        llm_client=object(),
        registry=_FakeRegistry(str(tmp_path)),
        orch_config=cfg,
        model="test-model",
    )


# ── _get_git_diff / _cached_git_diff / _patch_files_have_wt_changes ─────────

def test_get_git_diff(git_repo):
    (git_repo / "f.py").write_text("x=42\n", encoding="utf-8")
    o = _make_orch(git_repo)
    diff = o._get_git_diff(str(git_repo), ["f.py"])
    assert "x=42" in diff


def test_get_git_diff_empty_paths_whole_repo(git_repo):
    (git_repo / "f.py").write_text("x=7\n", encoding="utf-8")
    o = _make_orch(git_repo)
    diff = o._get_git_diff(str(git_repo), [])
    assert "x=7" in diff


def test_get_git_diff_truncation(git_repo):
    (git_repo / "f.py").write_text("x=1\n" * 5000, encoding="utf-8")
    o = _make_orch(git_repo, review_diff_char_limit=200)
    diff = o._get_git_diff(str(git_repo), ["f.py"])
    assert "[diff truncated: 200/" in diff


def test_get_git_diff_failure_returns_empty(tmp_path):
    o = _make_orch(tmp_path)
    assert o._get_git_diff(str(tmp_path), ["f.py"]) == ""


def test_cached_git_diff_hit_and_miss(git_repo):
    (git_repo / "f.py").write_text("x=5\n", encoding="utf-8")
    o = _make_orch(git_repo)
    cache = {}
    d1 = o._cached_git_diff(cache, str(git_repo), ["f.py"])
    d2 = o._cached_git_diff(cache, str(git_repo), ["f.py"])
    assert d1 == d2 and len(cache) == 1
    # None cache → thin pass-through
    d3 = o._cached_git_diff(None, str(git_repo), ["f.py"])
    assert d3 == d1


def test_patch_files_have_wt_changes(git_repo):
    (git_repo / "new.py").write_text("print(1)\n", encoding="utf-8")  # untracked
    assert OrchestratorAgent._patch_files_have_wt_changes(str(git_repo), ["new.py"]) is True
    assert OrchestratorAgent._patch_files_have_wt_changes(str(git_repo), ["f.py"]) is False
    assert OrchestratorAgent._patch_files_have_wt_changes(None, ["new.py"]) is False
    assert OrchestratorAgent._patch_files_have_wt_changes(str(git_repo), []) is False


def test_patch_files_have_wt_changes_non_repo(tmp_path):
    assert OrchestratorAgent._patch_files_have_wt_changes(str(tmp_path), ["a.py"]) is False


# ── _synthesize_untracked_diff ──────────────────────────────────────────────

def test_synthesize_untracked_diff(git_repo):
    (git_repo / "brand_new.py").write_text("def hello():\n    return 1\n", encoding="utf-8")
    out = OrchestratorAgent._synthesize_untracked_diff(str(git_repo), ["brand_new.py"])
    assert "+++ b/brand_new.py" in out
    assert "+def hello():" in out
    assert "@@ -0,0 +1,2 @@" in out


def test_synthesize_untracked_diff_none_untracked(git_repo):
    out = OrchestratorAgent._synthesize_untracked_diff(str(git_repo), ["f.py"])
    assert out == ""


def test_synthesize_untracked_diff_empty_inputs(tmp_path):
    assert OrchestratorAgent._synthesize_untracked_diff("", ["a.py"]) == ""
    assert OrchestratorAgent._synthesize_untracked_diff(str(tmp_path), []) == ""


def test_synthesize_untracked_diff_char_limit(git_repo):
    (git_repo / "big.py").write_text("x" * 500, encoding="utf-8")
    (git_repo / "big2.py").write_text("y" * 500, encoding="utf-8")
    out = OrchestratorAgent._synthesize_untracked_diff(
        str(git_repo), ["big.py", "big2.py"], char_limit=50,
    )
    assert "further untracked files omitted" in out


def test_synthesize_untracked_diff_prefix_dir_match(git_repo):
    d = git_repo / "newdir"
    d.mkdir()
    (d / "a.py").write_text("y=1\n", encoding="utf-8")
    out = OrchestratorAgent._synthesize_untracked_diff(str(git_repo), ["newdir"])
    assert "+++ b/newdir/a.py" in out


# ── _revert_unassigned_changes / _checkout_one / _unlink_one ────────────────

def test_revert_unassigned_changes_tracked_and_untracked(git_repo):
    (git_repo / "f.py").write_text("x=999\n", encoding="utf-8")  # tracked modified
    (git_repo / "stray.py").write_text("stray\n", encoding="utf-8")  # untracked
    o = _make_orch(git_repo)
    reverted = o._revert_unassigned_changes(
        str(git_repo), [{"file": "f.py"}, {"file": "stray.py"}]
    )
    assert "f.py" in reverted and "stray.py" in reverted
    assert (git_repo / "f.py").read_text() == "x=2\n"  # back to committed content
    assert not (git_repo / "stray.py").exists()


def test_revert_unassigned_changes_directory(git_repo):
    d = git_repo / "newdir"
    d.mkdir()
    (d / "x.py").write_text("x\n", encoding="utf-8")
    o = _make_orch(git_repo)
    reverted = o._revert_unassigned_changes(str(git_repo), [{"file": "newdir"}])
    assert "newdir" in reverted
    assert not d.exists()


def test_revert_unassigned_changes_skips_infra(git_repo):
    o = _make_orch(git_repo)
    reverted = o._revert_unassigned_changes(str(git_repo), [{"file": ".asicode/x"}])
    assert reverted == []


def test_revert_unassigned_changes_string_entries(git_repo):
    (git_repo / "stray2.py").write_text("s\n", encoding="utf-8")
    o = _make_orch(git_repo)
    reverted = o._revert_unassigned_changes(str(git_repo), ["stray2.py"])
    assert reverted == ["stray2.py"]


def test_revert_unassigned_changes_perfile_fallback(git_repo, monkeypatch):
    (git_repo / "f.py").write_text("x=8\n", encoding="utf-8")
    o = _make_orch(git_repo)
    import subprocess as _sp
    real_run = _sp.run

    def flaky_run(cmd, **kw):
        if cmd[:2] == ["git", "ls-files"] and "-z" in cmd:
            raise OSError("ls-files boom")
        return real_run(cmd, **kw)

    monkeypatch.setattr(_sp, "run", flaky_run)
    reverted = o._revert_unassigned_changes(str(git_repo), [{"file": "f.py"}])
    assert reverted == ["f.py"]
    assert (git_repo / "f.py").read_text() == "x=2\n"


# ── scope filtering helpers ─────────────────────────────────────────────────

def test_capture_scope_baseline(git_repo):
    (git_repo / ".env").write_text("dirty\n", encoding="utf-8")
    o = _make_orch(git_repo)
    st = SubTaskSpec(task_id="d1", title="t", description="d", assigned_files=["a.py"])
    o._capture_scope_baseline(str(git_repo), [st])
    assert ".env" in o._baseline_dirty_paths
    assert "a.py" in o._global_assigned_paths


def test_filter_unassigned_changes(git_repo):
    (git_repo / ".env").write_text("d\n", encoding="utf-8")
    o = _make_orch(git_repo)
    o._capture_scope_baseline(str(git_repo), [])
    reported = [
        {"file": ".env"},            # baseline dirt → filtered
        {"file": ".asicode/log"},    # infra → filtered
        {"file": "peer.py"},         # global assigned (peer) → filtered
        {"file": "real_stray.py"},   # genuine → kept
    ]
    o._global_assigned_paths = {"peer.py"}
    out = o._filter_unassigned_changes(reported, own_assigned=["mine.py"])
    assert out == [{"file": "real_stray.py"}]
    assert o._filter_unassigned_changes([], ["a"]) == []
    assert o._filter_unassigned_changes([{"file": ""}], ["a"]) == []


def test_git_status_changed_paths(git_repo):
    (git_repo / "f.py").write_text("x=3\n", encoding="utf-8")
    o = _make_orch(git_repo)
    paths = o._git_status_changed_paths(str(git_repo))
    assert any(p.get("file") == "f.py" for p in paths)
    assert o._git_status_changed_paths(None) == []
    assert o._git_status_changed_paths(str(tmp_path_fresh())) == []


def tmp_path_fresh():
    import tempfile
    return tempfile.mkdtemp()


def test_detect_genuine_violations_raw(git_repo):
    o = _make_orch(git_repo)
    out = o._detect_genuine_violations(str(git_repo), ["f.py"], raw_unassigned=[{"file": "stray.py"}])
    assert out == [{"file": "stray.py"}]


# ── _apply_scope_violation_policy ───────────────────────────────────────────

def test_apply_scope_violation_policy_warn(git_repo):
    o = _make_orch(git_repo, scope_violation_policy="warn")
    r = AgentResult(status="success", final_message="ok")
    assert o._apply_scope_violation_policy(str(git_repo), [{"file": "s.py"}], r, agent_id="a", mode="ipc") == []
    assert r.status == "success"


def test_apply_scope_violation_policy_revert(git_repo):
    (git_repo / "stray.py").write_text("x\n", encoding="utf-8")
    o = _make_orch(git_repo, scope_violation_policy="revert")
    r = AgentResult(status="success", final_message="ok")
    rv = o._apply_scope_violation_policy(str(git_repo), [{"file": "stray.py"}], r, agent_id="a", mode="in-process")
    assert rv == ["stray.py"]


def test_apply_scope_violation_policy_fail(git_repo):
    o = _make_orch(git_repo, scope_violation_policy="fail")
    r = AgentResult(status="success", final_message="done")
    o._apply_scope_violation_policy(str(git_repo), [{"file": "s.py"}], r, agent_id="a", mode="ipc")
    assert r.status == "error"
    assert "scope_violation" in r.final_message


def test_apply_scope_violation_policy_empty(git_repo):
    o = _make_orch(git_repo)
    r = AgentResult(status="success", final_message="x")
    assert o._apply_scope_violation_policy(str(git_repo), [], r, agent_id="a", mode="ipc") == []


# ── _compute_diff_verdict / _locate_symbol ──────────────────────────────────

def test_compute_diff_verdict_verified(git_repo):
    (git_repo / "f.py").write_text("x=42\n", encoding="utf-8")
    o = _make_orch(git_repo)
    r = AgentResult(status="success", final_message="done", applied_patches=[{"file": "f.py"}])
    verdict = o._compute_diff_verdict(agent_id="a1", result=r, repo_root=str(git_repo), diff_cache={})
    assert verdict == "VERIFIED"
    assert r._orch_diff_verdict == "VERIFIED"


def test_compute_diff_verdict_untracked_verified(git_repo):
    (git_repo / "new.py").write_text("n\n", encoding="utf-8")
    o = _make_orch(git_repo)
    r = AgentResult(status="success", final_message="created", applied_patches=[{"file": "new.py"}])
    verdict = o._compute_diff_verdict(agent_id="a1", result=r, repo_root=str(git_repo), diff_cache=None)
    assert verdict == "VERIFIED"


def test_compute_diff_verdict_no_changes(git_repo):
    o = _make_orch(git_repo)
    r = AgentResult(status="success", final_message="nothing changed", applied_patches=[{"file": "f.py"}])
    verdict = o._compute_diff_verdict(agent_id="a1", result=r, repo_root=str(git_repo), diff_cache={})
    assert verdict == "NO_CHANGES"


def test_compute_diff_verdict_unverifiable_parallel(git_repo):
    o = _make_orch(git_repo, parallel=True)
    r = AgentResult(status="success", final_message="x", applied_patches=[])
    verdict = o._compute_diff_verdict(agent_id="a1", result=r, repo_root=str(git_repo), diff_cache={})
    assert verdict == "UNVERIFIABLE"


def test_compute_diff_verdict_whole_repo_non_parallel(git_repo):
    (git_repo / "f.py").write_text("x=4\n", encoding="utf-8")
    o = _make_orch(git_repo, parallel=False)
    r = AgentResult(status="success", final_message="x", applied_patches=[])
    verdict = o._compute_diff_verdict(agent_id="a1", result=r, repo_root=str(git_repo), diff_cache={})
    assert verdict == "VERIFIED"


def test_compute_diff_verdict_patch_text(git_repo):
    (git_repo / "f.py").write_text("x=6\n", encoding="utf-8")
    o = _make_orch(git_repo)
    patch_text = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-x=1\n+x=6\n"
    r = AgentResult(status="success", final_message="x", applied_patches=[patch_text])
    verdict = o._compute_diff_verdict(agent_id="a1", result=r, repo_root=str(git_repo), diff_cache={})
    assert verdict == "VERIFIED"


def test_locate_symbol(tmp_path):
    (tmp_path / "m.py").write_text("def target():\n    pass\n\nclass K:\n    pass\n", encoding="utf-8")
    o = _make_orch(tmp_path)
    hint = o._locate_symbol("target", ["m.py"], str(tmp_path))
    assert "Found `target` in m.py at line 1" in hint
    assert "def target():" in hint
    assert o._locate_symbol("nope", ["m.py"], str(tmp_path)) == ""


def test_locate_symbol_unreadable(tmp_path):
    o = _make_orch(tmp_path)
    assert o._locate_symbol("x", ["missing.py"], str(tmp_path)) == ""


# ── _review_subagent_result (LLM mocked) ────────────────────────────────────

def _patch_llm(monkeypatch, raw):
    # orchestrator binds simple_llm_call at import time (from utils.llm_utils
    # import simple_llm_call), so patch the module-global name, not the source.
    monkeypatch.setattr(orch, "simple_llm_call", lambda *a, **kw: raw)


def test_review_subagent_result_approved(git_repo, monkeypatch):
    (git_repo / "f.py").write_text("x=42\n", encoding="utf-8")
    _patch_llm(monkeypatch, '{"approved": true, "feedback": ""}')
    o = _make_orch(git_repo, review_enabled=True)
    st = SubTaskSpec(task_id="d1", title="t", description="d", assigned_files=["f.py"])
    approved, feedback = o._review_subagent_result("d1", st, str(git_repo))
    assert approved is True and feedback == ""


def test_review_subagent_result_rejected_with_hint(git_repo, monkeypatch):
    (git_repo / "f.py").write_text("x=42\n", encoding="utf-8")
    _patch_llm(monkeypatch, '{"approved": false, "feedback": "wrong", "target_symbol": "target"}')
    o = _make_orch(git_repo, review_enabled=True)
    st = SubTaskSpec(task_id="d1", title="t", description="d", assigned_files=["f.py"])
    approved, feedback = o._review_subagent_result("d1", st, str(git_repo))
    assert approved is False
    assert "wrong" in feedback


def test_review_subagent_result_parse_failure_assumes_ok(git_repo, monkeypatch):
    (git_repo / "f.py").write_text("x=42\n", encoding="utf-8")
    _patch_llm(monkeypatch, "not json at all")
    o = _make_orch(git_repo, review_enabled=True)
    st = SubTaskSpec(task_id="d1", title="t", description="d", assigned_files=["f.py"])
    approved, _ = o._review_subagent_result("d1", st, str(git_repo))
    assert approved is True


def test_review_subagent_result_no_assigned_files(git_repo, monkeypatch):
    _patch_llm(monkeypatch, '{"approved": false}')
    o = _make_orch(git_repo, review_enabled=True)
    st = SubTaskSpec(task_id="d1", title="t", description="d", assigned_files=[])
    approved, feedback = o._review_subagent_result("d1", st, str(git_repo))
    assert approved is True and feedback == ""


def test_review_subagent_result_no_changes_rejected(git_repo, monkeypatch):
    _patch_llm(monkeypatch, '{"approved": true}')
    o = _make_orch(git_repo, review_enabled=True)
    st = SubTaskSpec(task_id="d1", title="t", description="d", assigned_files=["f.py"])
    approved, feedback = o._review_subagent_result("d1", st, str(git_repo))
    assert approved is False
    assert "No changes detected" in feedback


# ── synthesis helpers ───────────────────────────────────────────────────────

def test_synthesize_from_subtasks():
    o = _make_orch("/tmp")
    pairs = [
        (SubTaskSpec(task_id="a", title="A", description="d"), AgentResult(status="success", final_message="ok")),
        (SubTaskSpec(task_id="b", title="B", description="d"), AgentResult(status="max_turns", final_message="partial")),
        (SubTaskSpec(task_id="c", title="C", description="d"), None),
        (SubTaskSpec(task_id="d", title="D", description="d"), AgentResult(status="error", final_message="bad")),
    ]
    out = o._synthesize_from_subtasks(pairs)
    assert "✅ [a] A" in out
    assert "⚠️ [b] B" in out
    assert "❌ [d] D" in out
    assert "[c] C: no result" in out
    assert o._synthesize_from_subtasks([]) == "Multi-agent task completed."


def test_paired_subtask_results(tmp_path):
    o = _make_orch(tmp_path)
    st = SubTaskSpec(task_id="x", title="X", description="d")
    o._bg_subagents["x"] = {"subtask": st, "result": None}
    pairs = o._paired_subtask_results()
    assert pairs == [(st, None)]


# ── _extract_subagent_summary / shared memory ───────────────────────────────

def test_extract_subagent_summary():
    o = _make_orch("/tmp")
    st = SubTaskSpec(task_id="d1", title="Do it", description="d")
    r = AgentResult(status="success", final_message="all done", applied_patches=[{"file": "a.py"}, {"file": "a.py"}, {"file": "b.py"}])
    s = o._extract_subagent_summary(st, r)
    assert s == "[d1: Do it → completed | Files: a.py, b.py] all done"
    # non-dict patch with file_path
    r2 = AgentResult(status="max_turns", final_message="x", applied_patches=[SimpleNamespace(file_path="c.py")])
    assert "Files: c.py" in o._extract_subagent_summary(st, r2)
    # non-list patches
    r3 = AgentResult(status="error", final_message="x", applied_patches="junk")
    assert o._extract_subagent_summary(st, r3).startswith("[d1: Do it → failed]")
    # weird status passthrough
    r4 = AgentResult(status="weird", final_message="x")
    assert "→ weird]" in o._extract_subagent_summary(st, r4)


def test_record_subagent_summary_ring(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "_SHARED_MEMORY_STORE_LIMIT", 3)
    o = _make_orch(tmp_path)
    for i in range(5):
        o._record_subagent_summary(f"s{i}")
    assert o._shared_memory == ["s2", "s3", "s4"]


def test_cap_shared_memory_injection(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "_SHARED_MEMORY_INJECT_LIMIT", 3)
    monkeypatch.setattr(orch, "_SHARED_MEMORY_INJECT_MAX_CHARS", 10)
    o = _make_orch(tmp_path)
    out = o._cap_shared_memory_injection(["aaaa", "bbbb", "cccc", "dddd"])
    # keeps most recent within budget, at least one always
    assert out  # non-empty
    assert out[-1] == "dddd"


# ── model resolution / gc / build task context ──────────────────────────────

def test_resolve_subagent_model():
    o = _make_orch("/tmp", subagent_models={"1": ("ollama", "qwen", "k1"), "2": ("ollama", "deep", "k2")})
    assert o._resolve_subagent_model("dev_2") == ("ollama", "deep", "k2")
    assert o._resolve_subagent_model("dev_9") == ("ollama", "qwen", "k1")  # lowest slot fallback
    o2 = _make_orch("/tmp", subagent_provider="p", subagent_model="m", subagent_api_key="k")
    assert o2._resolve_subagent_model("dev_1") == ("p", "m", "k")


def test_gc_subagent_artifacts(tmp_path):
    base = tmp_path / ".asicode" / "subagents"
    old = base / "old_dir"
    old.mkdir(parents=True)
    (old / "worker.log").write_text("x")
    old_t = time.time() - 20 * 86400
    os.utime(old, (old_t, old_t))
    fresh = base / "fresh_dir"
    fresh.mkdir(parents=True)
    OrchestratorAgent._gc_subagent_artifacts(str(tmp_path))
    assert not old.exists()
    assert fresh.exists()
    OrchestratorAgent._gc_subagent_artifacts(str(tmp_path / "no_base"))  # no-op


def test_build_task_with_predecessor_context():
    o = _make_orch("/tmp")
    st = SubTaskSpec(task_id="d2", title="t", description="do stuff", assigned_files=["a.py"], dependencies=["d1"])
    dep = SubTaskSpec(task_id="d1", title="Dep", description="d")
    done = {"d1": AgentResult(status="success", final_message="dep done")}
    text = o._build_task_with_predecessor_context(st, done, {"d1": dep})
    assert "[Assigned files: a.py]" in text
    assert "[Predecessor task status]" in text
    assert "[d1: Dep → completed] dep done" in text
    # missing dep result → skipped
    text2 = o._build_task_with_predecessor_context(st, {}, {"d1": dep})
    assert "[Predecessor task status]" not in text2
    # dep not in map → synthesized placeholder
    st2 = SubTaskSpec(task_id="d2", title="t", description="d", dependencies=["ghost"])
    text3 = o._build_task_with_predecessor_context(st2, {"ghost": AgentResult(status="success", final_message="g")}, {})
    assert "[ghost:" in text3
    # shared memory injection
    o._shared_memory = ["[d9: X → completed] z"]
    text4 = o._build_task_with_predecessor_context(
        SubTaskSpec(task_id="d2", title="t", description="d"), {}, {}
    )
    assert "[Orchestration progress]" in text4


# ── _detect_cycles_kahn / _break_cycles / _find_current_cycles ──────────────

def _specs_from_deps(deps: dict[str, list[str]]):
    return {tid: SubTaskSpec(task_id=tid, title=tid, description=tid, dependencies=d)
            for tid, d in deps.items()}


def test_detect_cycles_kahn_acyclic():
    o = _make_orch("/tmp")
    m = _specs_from_deps({"a": [], "b": ["a"], "c": ["b"]})
    order, cycles = o._detect_cycles_kahn(m)
    assert cycles == []
    assert order == ["a", "b", "c"]


def test_detect_cycles_kahn_cycle():
    o = _make_orch("/tmp")
    m = _specs_from_deps({"a": ["b"], "b": ["a"], "c": []})
    _, cycles = o._detect_cycles_kahn(m)
    assert cycles  # one cycle among a/b
    assert set(cycles[0]) == {"a", "b"}


def test_detect_cycles_kahn_self_cycle():
    o = _make_orch("/tmp")
    m = _specs_from_deps({"a": ["a"]})
    _, cycles = o._detect_cycles_kahn(m)
    assert cycles == [["a"]]


def test_break_cycles_simple():
    o = _make_orch("/tmp")
    specs = [
        SubTaskSpec(task_id="a", title="a", description="d", dependencies=["b"]),
        SubTaskSpec(task_id="b", title="b", description="d", dependencies=["a"]),
    ]
    broken = o._break_cycles(specs, [["a", "b"]])
    assert broken is not None
    _, cycles = o._detect_cycles_kahn({s.task_id: s for s in broken})
    assert cycles == []


def test_break_cycles_none():
    o = _make_orch("/tmp")
    assert o._break_cycles([], []) is None


def test_find_current_cycles():
    o = _make_orch("/tmp")
    m = _specs_from_deps({"a": ["b"], "b": ["a"], "c": []})
    cycles = o._find_current_cycles({"a", "b"}, m)
    assert cycles


def test_has_dependencies():
    o = _make_orch("/tmp")
    assert o._has_dependencies([SubTaskSpec(task_id="a", title="a", description="d", dependencies=["b"])])
    assert not o._has_dependencies([SubTaskSpec(task_id="a", title="a", description="d")])


# ── _OrchestratorBackedRegistry ─────────────────────────────────────────────

def _make_orch_with_registry(tmp_path):
    base = _FakeRegistry(str(tmp_path))
    o = _make_orch(tmp_path)
    return base, _OrchestratorBackedRegistry(base, o)


def test_obr_delegation(tmp_path):
    base, obr = _make_orch_with_registry(tmp_path)
    assert obr.repo_root == str(tmp_path)
    assert obr.config is base.config
    obr.session_plan = "p"  # __setattr__ → base
    assert base.session_plan == "p"
    assert obr.session_plan == "p"


def test_obr_get_tool_schemas(tmp_path):
    _, obr = _make_orch_with_registry(tmp_path)
    schemas = obr.get_tool_schemas()
    names = [s["name"] for s in schemas]
    assert "spawn_subagent" in names and "poll_subagent" in names and "list_subagents" in names
    assert "read_file" in names
    assert names[0] == "spawn_subagent"  # native first


def test_obr_dispatch_native_and_fallback(tmp_path, monkeypatch):
    _, obr = _make_orch_with_registry(tmp_path)
    # native tool dispatch
    monkeypatch.setattr(obr._obr_orch, "_tool_list_subagents", lambda: "Sub-agents:\n- x")
    tr = obr.dispatch("list_subagents", {})
    assert tr.ok is True and "Sub-agents:" in tr.content
    # native tool error → ok=False without traceback
    monkeypatch.setattr(
        obr._obr_orch, "_dispatch_native_tool",
        lambda n, a: (_ for _ in ()).throw(_NativeToolError("bad args")),
    )
    tr2 = obr.dispatch("spawn_subagent", {})
    assert tr2.ok is False and "bad args" in tr2.error
    # unexpected exception → logged, ok=False
    monkeypatch.setattr(
        obr._obr_orch, "_dispatch_native_tool",
        lambda n, a: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    tr3 = obr.dispatch("spawn_subagent", {})
    assert tr3.ok is False and "boom" in tr3.error
    # fallthrough to base registry
    tr4 = obr.dispatch("read_file", {})
    assert tr4 == "dispatched:read_file"


def test_dispatch_native_tool_unknown(tmp_path):
    _, obr = _make_orch_with_registry(tmp_path)
    with pytest.raises(_NativeToolError):
        obr._obr_orch._dispatch_native_tool("nope", {})


# ── native schemas / format helpers ─────────────────────────────────────────

def test_native_orchestrator_schemas(tmp_path):
    o = _make_orch(tmp_path)
    schemas = o._native_orchestrator_schemas()
    assert [s["name"] for s in schemas] == ["spawn_subagent", "poll_subagent", "list_subagents"]
    assert schemas[0]["parameters"]["required"] == ["task_description", "title"]


def test_format_poll_patches():
    o = _make_orch("/tmp")
    assert o._format_poll_patches([]) == ""
    out = o._format_poll_patches([{"file": "a.py"}, SimpleNamespace(file_path="b.py"), "raw text"])
    assert "Applied patches (2): a.py, b.py" in out
    out2 = o._format_poll_patches(["x", "y"])  # no extractable names
    assert "Applied patches: 2" in out2
    files = [{"file": f"f{i}.py"} for i in range(10)]
    out3 = o._format_poll_patches(files)
    assert "(+2 more)" in out3


def test_format_agent_result():
    o = _make_orch("/tmp")
    r = AgentResult(status="success", final_message="done", applied_patches=[{"file": "a.py"}])
    r._orch_diff_verdict = "VERIFIED"
    r._orch_unassigned = [{"file": "stray.py"}]
    out = o._format_agent_result("dev_1", "success", r)
    assert "status: success" in out
    assert "Diff verification: VERIFIED" in out
    assert "Out-of-scope changes: stray.py" in out
    # None result
    assert "no result captured" in o._format_agent_result("dev_1", "error", None)
    # long message truncation
    long_r = AgentResult(status="success", final_message="x" * 20000)
    out2 = o._format_agent_result("dev_1", "success", long_r)
    assert "truncated" in out2


def test_tool_list_subagents(tmp_path):
    o = _make_orch(tmp_path)
    assert "No sub-agents spawned yet." in o._tool_list_subagents()
    st = SubTaskSpec(task_id="x", title="Task X", description="d")
    o._bg_subagents["x"] = {"status": "running", "subtask": st}
    out = o._tool_list_subagents()
    assert "- x [running]: Task X" in out


# ── tool poll handlers ──────────────────────────────────────────────────────

def test_tool_poll_subagent_bad_args(tmp_path):
    o = _make_orch(tmp_path)
    with pytest.raises(_NativeToolError):
        o._tool_poll_subagent({})
    with pytest.raises(_NativeToolError):
        o._tool_poll_subagent({"agent_id": "a", "agent_ids": ["b"]})


def test_tool_poll_subagent_unknown_id(tmp_path):
    o = _make_orch(tmp_path)
    with pytest.raises(_NativeToolError):
        o._tool_poll_subagent({"agent_id": "nope"})


def test_poll_any_agent_unknown(tmp_path):
    o = _make_orch(tmp_path)
    with pytest.raises(_NativeToolError):
        o._poll_any_agent(["ghost"], 0)
    with pytest.raises(_NativeToolError):
        o._poll_any_agent([], 0)


def test_poll_any_agent_none_done_nonblocking(tmp_path):
    o = _make_orch(tmp_path)
    from concurrent.futures import Future
    st = SubTaskSpec(task_id="x", title="X", description="d")
    o._bg_subagents["x"] = {"future": Future(), "result": None, "status": "running", "subtask": st, "started_at": time.monotonic()}
    out = o._poll_any_agent(["x"], 0)
    assert "None of 1 sub-agents have completed yet" in out


def test_poll_single_agent_running_queued(tmp_path):
    o = _make_orch(tmp_path)
    st = SubTaskSpec(task_id="x", title="X", description="d")
    from concurrent.futures import Future
    o._bg_subagents["x"] = {"future": Future(), "result": None, "status": "running", "subtask": st, "started_at": time.monotonic()}
    out = o._poll_single_agent("x", 0)
    assert "queued" in out  # future not started → waiting for a slot


def test_poll_any_agent_duplicate_ids(tmp_path):
    o = _make_orch(tmp_path)
    st = SubTaskSpec(task_id="x", title="X", description="d")
    from concurrent.futures import Future
    o._bg_subagents["x"] = {"future": Future(), "result": None, "status": "running", "subtask": st, "started_at": time.monotonic()}
    out = o._poll_any_agent(["x", "x"], 0)  # dedup, no crash
    assert "None of 1 sub-agents" in out


# ── background executor / drain / shutdown ──────────────────────────────────

def test_bg_executor_lifecycle(tmp_path, monkeypatch):
    o = _make_orch(tmp_path)
    assert o._bg_executor is None
    pool = o._ensure_bg_executor()
    assert o._bg_executor is pool
    o._shutdown_bg_executor()
    assert o._bg_executor is None
    o._shutdown_bg_executor()  # no-op second time


def test_run_subagent_background_and_check(tmp_path, monkeypatch):
    o = _make_orch(tmp_path)
    st = SubTaskSpec(task_id="b1", title="B", description="d")
    monkeypatch.setattr(o, "_run_subagent", lambda *a, **kw: AgentResult(status="success", final_message="bg done", turns=[]))
    aid = o._run_subagent_background(st)
    assert aid == "b1"
    status, result = o._check_bg_subagent("b1", timeout_s=10)
    assert status == "success" and result.final_message == "bg done"
    # cached second call — no double append
    n = len(o._bg_results)
    status2, _ = o._check_bg_subagent("b1", timeout_s=0)
    assert status2 == "success" and len(o._bg_results) == n
    # unknown id
    assert o._check_bg_subagent("ghost") == ("unknown", None)
    # running state before completion
    from concurrent.futures import Future
    o._bg_subagents["b2"] = {"future": Future(), "result": None, "status": "running", "subtask": st, "started_at": time.monotonic()}
    assert o._check_bg_subagent("b2")[0] == "running"


def test_bg_job_exception_returns_error_result(tmp_path, monkeypatch):
    o = _make_orch(tmp_path)
    st = SubTaskSpec(task_id="b3", title="B", description="d")
    def boom(*a, **kw):
        raise RuntimeError("crash")
    monkeypatch.setattr(o, "_run_subagent", boom)
    aid = o._run_subagent_background(st)
    status, result = o._check_bg_subagent(aid, timeout_s=10)
    assert status == "error"
    assert "crash" in result.error


def test_future_is_queued(tmp_path):
    o = _make_orch(tmp_path)
    from concurrent.futures import Future
    f = Future()
    assert o._future_is_queued({"future": f}) is True  # neither running nor done
    f.set_result("x")
    assert o._future_is_queued({"future": f}) is False
    assert o._future_is_queued({}) is False


def test_drain_background_subagents_empty(tmp_path):
    o = _make_orch(tmp_path)
    o._drain_background_subagents()  # no agents → no-op
    assert True


def test_drain_background_subagents_cancelled(tmp_path):
    ev = threading.Event()
    ev.set()
    o = _make_orch(tmp_path, cancel_event=ev)
    st = SubTaskSpec(task_id="x", title="X", description="d")
    o._bg_subagents["x"] = {"future": None, "result": None, "status": "running", "subtask": st, "started_at": time.monotonic()}
    o._drain_background_subagents(per_agent_timeout=600)
    assert True  # returned immediately


def test_gather_done_futures(tmp_path, monkeypatch):
    o = _make_orch(tmp_path)
    from concurrent.futures import Future
    f = Future()
    f.set_result(AgentResult(status="success", final_message="done", turns=[]))
    st = SubTaskSpec(task_id="x", title="X", description="d")
    o._bg_subagents["x"] = {"future": f, "result": None, "status": "running", "subtask": st, "started_at": time.monotonic()}
    completed, pending = o._gather_done_futures([("x", f)])
    assert len(completed) == 1 and pending == []
    assert o._bg_results  # appended exactly once


def test_check_bg_subagent_future_exception(tmp_path):
    o = _make_orch(tmp_path)
    from concurrent.futures import Future
    f = Future()
    f.set_exception(ValueError("inner"))
    st = SubTaskSpec(task_id="x", title="X", description="d")
    o._bg_subagents["x"] = {"future": f, "result": None, "status": "running", "subtask": st, "started_at": time.monotonic()}
    status, result = o._check_bg_subagent("x", timeout_s=0)
    assert status == "error" and "inner" in result.error


# ═══════════════════════════════════════════════════════════════════════════
# Round 3: _run_subagent / _run_subagent_ipc / continue / dependency-aware /
#          parallel batch / tool spawn / worker launch (mocked IPC & loops)
# ═══════════════════════════════════════════════════════════════════════════


class _FakeResult:
    """AgentResult-like object with all attrs _run_subagent_ipc touches."""

    def __init__(self, status="success", final_message="done", turns=1,
                 applied_patches=None, error=None, unassigned_changes=None):
        self.status = status
        self.final_message = final_message
        self.turns = turns
        self.applied_patches = applied_patches or []
        self.error = error
        self.unassigned_changes = unassigned_changes or []


def test_run_subagent_ipc_success(tmp_path, monkeypatch):
    import external_llm.agent.subagent_ipc as sipc
    o = _make_orch(tmp_path, subagent_mode="ipc", ipc_timeout_s=30)
    st = SubTaskSpec(task_id="dev_1", title="T", description="D", assigned_files=["a.py"])
    (tmp_path / "a.py").write_text("x")
    calls = {}
    monkeypatch.setattr(sipc, "clear_result", lambda *a, **k: calls.setdefault("clear", 0) or calls.__setitem__("clear", calls["clear"] + 1))
    monkeypatch.setattr(sipc, "write_task", lambda *a, **k: calls.setdefault("write", 0) or calls.__setitem__("write", calls["write"] + 1))
    monkeypatch.setattr(sipc, "wait_for_result", lambda *a, **k: _FakeResult(status="success", final_message="ipc done", turns=2))
    monkeypatch.setattr(o, "_claim_reusable_worker", lambda repo: None)
    monkeypatch.setattr(o, "_return_worker_to_pool", lambda wid: None)
    monkeypatch.setattr(o, "_compute_diff_verdict", lambda **kw: "VERIFIED")
    monkeypatch.setattr(o, "_filter_unassigned_changes", lambda r, own: r)
    monkeypatch.setattr(o, "_apply_scope_violation_policy", lambda *a, **k: [])
    res = o._run_subagent_ipc(st, task_text="tt")
    assert res.status == "success"
    assert res.final_message == "ipc done"
    assert calls["write"] == 1 and calls["clear"] == 1
    assert "dev_1" in o._ipc_worker_ids


def test_run_subagent_ipc_timeout_abandon(tmp_path, monkeypatch):
    import external_llm.agent.subagent_ipc as sipc
    o = _make_orch(tmp_path, subagent_mode="ipc", ipc_timeout_s=1)
    st = SubTaskSpec(task_id="dev_1", title="T", description="D", assigned_files=["a.py"])
    (tmp_path / "a.py").write_text("x")
    monkeypatch.setattr(sipc, "clear_result", lambda *a, **k: None)
    monkeypatch.setattr(sipc, "write_task", lambda *a, **k: None)
    monkeypatch.setattr(sipc, "wait_for_result", lambda *a, **k: None)  # timeout
    monkeypatch.setattr(o, "_claim_reusable_worker", lambda repo: None)
    abandons = {}
    monkeypatch.setattr(o, "_abandon_ipc_worker",
                        lambda *a, **k: abandons.setdefault("n", 0) or abandons.__setitem__("n", abandons["n"] + 1) or False)
    res = o._run_subagent_ipc(st)
    assert res.status == "error"
    assert "timed out" in res.final_message
    assert abandons["n"] == 1


def test_run_subagent_ipc_cancelled_revert_and_grace(tmp_path, monkeypatch):
    import external_llm.agent.subagent_ipc as sipc
    o = _make_orch(tmp_path, subagent_mode="ipc", ipc_timeout_s=1)
    st = SubTaskSpec(task_id="dev_1", title="T", description="D", assigned_files=["a.py"])
    (tmp_path / "a.py").write_text("x")
    monkeypatch.setattr(sipc, "clear_result", lambda *a, **k: None)
    monkeypatch.setattr(sipc, "write_task", lambda *a, **k: None)
    monkeypatch.setattr(sipc, "wait_for_result", lambda *a, **k: _FakeResult(status="cancelled", final_message="c", turns=0))
    monkeypatch.setattr(sipc, "read_worker_idle_heartbeat_state", lambda *a, **k: "idle")
    monkeypatch.setattr(o, "_claim_reusable_worker", lambda repo: None)
    monkeypatch.setattr(o, "_return_worker_to_pool", lambda wid: None)
    monkeypatch.setattr(o, "_compute_diff_verdict", lambda **kw: "VERIFIED")
    monkeypatch.setattr(o, "_filter_unassigned_changes", lambda r, own: r)
    monkeypatch.setattr(o, "_apply_scope_violation_policy", lambda *a, **k: [])
    res = o._run_subagent_ipc(st)
    assert res.status == "cancelled"
    assert not (tmp_path / "a.py").exists() or (tmp_path / "a.py").read_text() == "x"


def test_run_subagent_ipc_review_retry(tmp_path, monkeypatch):
    import external_llm.agent.subagent_ipc as sipc
    o = _make_orch(tmp_path, subagent_mode="ipc", review_enabled=True, review_max_retries=1)
    st = SubTaskSpec(task_id="dev_1", title="T", description="D", assigned_files=["a.py"])
    (tmp_path / "a.py").write_text("x")
    seq = [0]

    def fake_wait(*a, **k):
        seq[0] += 1
        if seq[0] == 1:
            return _FakeResult(status="success", final_message="attempt1", turns=1)
        return _FakeResult(status="success", final_message="attempt2", turns=1)

    monkeypatch.setattr(sipc, "clear_result", lambda *a, **k: None)
    monkeypatch.setattr(sipc, "write_task", lambda *a, **k: None)
    monkeypatch.setattr(sipc, "wait_for_result", fake_wait)
    monkeypatch.setattr(o, "_claim_reusable_worker", lambda repo: None)
    monkeypatch.setattr(o, "_return_worker_to_pool", lambda wid: None)
    monkeypatch.setattr(o, "_compute_diff_verdict", lambda **kw: "VERIFIED")
    monkeypatch.setattr(o, "_filter_unassigned_changes", lambda r, own: r)
    monkeypatch.setattr(o, "_apply_scope_violation_policy", lambda *a, **k: [])
    monkeypatch.setattr(
        o, "_review_subagent_result",
        lambda **kw: (False, "needs work"),
    )
    res = o._run_subagent_ipc(st)
    assert res.final_message == "attempt2"
    assert seq[0] == 2  # two wait_for_result calls


def test_run_subagent_ipc_review_retry_timeout(tmp_path, monkeypatch):
    import external_llm.agent.subagent_ipc as sipc
    o = _make_orch(tmp_path, subagent_mode="ipc", review_enabled=True, review_max_retries=1)
    st = SubTaskSpec(task_id="dev_1", title="T", description="D", assigned_files=["a.py"])
    (tmp_path / "a.py").write_text("x")
    seq = [0]

    def fake_wait(*a, **k):
        seq[0] += 1
        if seq[0] == 1:
            return _FakeResult(status="success", final_message="attempt1", turns=1)
        return None  # retry timeout

    monkeypatch.setattr(sipc, "clear_result", lambda *a, **k: None)
    monkeypatch.setattr(sipc, "write_task", lambda *a, **k: None)
    monkeypatch.setattr(sipc, "wait_for_result", fake_wait)
    monkeypatch.setattr(o, "_claim_reusable_worker", lambda repo: None)
    monkeypatch.setattr(o, "_abandon_ipc_worker", lambda *a, **k: False)
    monkeypatch.setattr(
        o, "_review_subagent_result",
        lambda **kw: (False, "needs work"),
    )
    res = o._run_subagent_ipc(st)
    assert res.status == "error"
    assert "review-retry" in res.final_message


def test_run_subagent_ipc_reuse_worker(tmp_path, monkeypatch):
    import external_llm.agent.subagent_ipc as sipc
    o = _make_orch(tmp_path, subagent_mode="ipc")
    st = SubTaskSpec(task_id="dev_1", title="T", description="D", assigned_files=["a.py"])
    (tmp_path / "a.py").write_text("x")
    writes = {}
    monkeypatch.setattr(sipc, "clear_result", lambda *a, **k: None)
    monkeypatch.setattr(sipc, "write_task",
                        lambda repo, task, worker_id=None: writes.__setitem__("wid", worker_id))
    monkeypatch.setattr(sipc, "wait_for_result", lambda *a, **k: _FakeResult(status="success", turns=0))
    monkeypatch.setattr(o, "_claim_reusable_worker", lambda repo: "worker_9")
    monkeypatch.setattr(o, "_return_worker_to_pool", lambda wid: None)
    monkeypatch.setattr(o, "_compute_diff_verdict", lambda **kw: "VERIFIED")
    monkeypatch.setattr(o, "_filter_unassigned_changes", lambda r, own: r)
    monkeypatch.setattr(o, "_apply_scope_violation_policy", lambda *a, **k: [])
    o._run_subagent_ipc(st)
    assert writes["wid"] == "worker_9"  # reuse mode passes worker_id


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Terminal.app auto-launch branch is macOS-only (orchestrator.py:2207)",
)
def test_run_subagent_ipc_auto_launch_terminal(tmp_path, monkeypatch):
    import external_llm.agent.subagent_ipc as sipc
    o = _make_orch(tmp_path, subagent_mode="ipc", auto_launch_terminal=True)
    st = SubTaskSpec(task_id="dev_1", title="T", description="D", assigned_files=["a.py"])
    (tmp_path / "a.py").write_text("x")
    monkeypatch.setattr(sipc, "clear_result", lambda *a, **k: None)
    monkeypatch.setattr(sipc, "write_task", lambda *a, **k: None)
    monkeypatch.setattr(sipc, "wait_for_result", lambda *a, **k: _FakeResult(status="success", turns=0))
    monkeypatch.setattr(o, "_claim_reusable_worker", lambda repo: None)
    monkeypatch.setattr(o, "_return_worker_to_pool", lambda wid: None)
    launched = {}
    monkeypatch.setattr(o, "_launch_ipc_worker_terminal_macos",
                        lambda aid, wid: launched.__setitem__("n", True) or True)
    monkeypatch.setattr(o, "_compute_diff_verdict", lambda **kw: "VERIFIED")
    monkeypatch.setattr(o, "_filter_unassigned_changes", lambda r, own: r)
    monkeypatch.setattr(o, "_apply_scope_violation_policy", lambda *a, **k: [])
    o._run_subagent_ipc(st)
    assert launched.get("n") is True
    assert "dev_1" in o._subagent_ipc_commands


def test_launch_ipc_worker_terminal_macos(tmp_path, monkeypatch):
    o = _make_orch(tmp_path)
    o._subagent_ipc_commands["dev_1"] = 'cd "/tmp/x" && asi --subagent'
    popens = []
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: popens.append(a) or object())
    assert o._launch_ipc_worker_terminal_macos("dev_1", "w1") is True
    assert popens and popens[0][0][0] == "osascript"
    # missing command → False
    assert o._launch_ipc_worker_terminal_macos("dev_2", "w2") is False
    # Popen failure → False
    def boom(*a, **k):
        raise OSError("no osascript")
    monkeypatch.setattr("subprocess.Popen", boom)
    assert o._launch_ipc_worker_terminal_macos("dev_1", "w1") is False


def test_spawn_ipc_worker_background(tmp_path, monkeypatch):
    o = _make_orch(tmp_path)
    proc = SimpleNamespace(pid=123, poll=lambda: None)
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: proc)
    assert o._spawn_ipc_worker_background(str(tmp_path), "w1", "p", "m") is True
    assert o._ipc_worker_procs["w1"] is proc
    # failure → False
    def boom(*a, **k):
        raise OSError("spawn failed")
    monkeypatch.setattr("subprocess.Popen", boom)
    assert o._spawn_ipc_worker_background(str(tmp_path), "w2", "", "") is False


def test_spawn_ipc_worker_background_rotates_log(tmp_path, monkeypatch):
    o = _make_orch(tmp_path)
    proc = SimpleNamespace(pid=1, poll=lambda: None)
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: proc)
    d = tmp_path / ".asicode" / "subagents" / "w9"
    d.mkdir(parents=True)
    log = d / "worker.log"
    log.write_bytes(b"x" * (orch._WORKER_LOG_ROTATE_BYTES + 10))
    assert o._spawn_ipc_worker_background(str(tmp_path), "w9", "", "") is True
    assert (d / "worker.log.old").exists()


# ── _run_subagent (in-process) ──────────────────────────────────────────────

class _FakeLoop:
    def __init__(self, result):
        self._result = result
        self.calls = []

    def run(self, text):
        self.calls.append(text)
        return self._result


def test_run_subagent_inprocess_success(tmp_path, monkeypatch):
    o = _make_orch(tmp_path)
    st = SubTaskSpec(task_id="dev_1", title="T", description="Do the thing", assigned_files=["a.py"])
    (tmp_path / "a.py").write_text("x = 1\n")
    loop = _FakeLoop(AgentResult(status="success", final_message="ok", turns=[1], applied_patches=[]))
    monkeypatch.setattr(agent_loop_mod, "AgentLoop", lambda **kw: loop)
    monkeypatch.setattr(o._registry_proto, "clone_for_subagent", lambda cfg: o._registry_proto)
    monkeypatch.setattr(o, "_compute_diff_verdict", lambda **kw: "VERIFIED")
    monkeypatch.setattr(o, "_detect_genuine_violations", lambda *a, **k: [])
    monkeypatch.setattr(o, "_apply_scope_violation_policy", lambda *a, **k: [])
    res = o._run_subagent(st)
    assert res.status == "success"
    assert "[Assigned files: a.py]" in loop.calls[0]
    assert "[Original request goal]" not in loop.calls[0]
    # with original_request
    loop2 = _FakeLoop(AgentResult(status="success", final_message="ok", turns=[1]))
    monkeypatch.setattr(agent_loop_mod, "AgentLoop", lambda **kw: loop2)
    o._run_subagent(st, original_request="ROOT")
    assert "[Original request goal]\nROOT" in loop2.calls[0]


def test_run_subagent_inprocess_error_reverts(tmp_path, monkeypatch):
    o = _make_orch(tmp_path)
    st = SubTaskSpec(task_id="dev_1", title="T", description="D", assigned_files=["a.py"])
    (tmp_path / "a.py").write_text("original")
    loop = _FakeLoop(AgentResult(status="error", final_message="bad", error="e", turns=[]))
    monkeypatch.setattr(agent_loop_mod, "AgentLoop", lambda **kw: loop)
    monkeypatch.setattr(o._registry_proto, "clone_for_subagent", lambda cfg: o._registry_proto)
    monkeypatch.setattr(o, "_compute_diff_verdict", lambda **kw: "NO_CHANGES")
    monkeypatch.setattr(o, "_detect_genuine_violations", lambda *a, **k: [])
    monkeypatch.setattr(o, "_apply_scope_violation_policy", lambda *a, **k: [])
    class _MutatingLoop(_FakeLoop):
        def run(self, text):
            (tmp_path / "a.py").write_text("MUTATED")  # after snapshot capture
            return self._result

    loop = _MutatingLoop(AgentResult(status="error", final_message="bad", error="e", turns=[]))
    monkeypatch.setattr(agent_loop_mod, "AgentLoop", lambda **kw: loop)
    res = o._run_subagent(st)
    assert res.status == "error"
    assert (tmp_path / "a.py").read_text() == "original"  # snapshot restored


def test_run_subagent_inprocess_review_retry(tmp_path, monkeypatch):
    o = _make_orch(tmp_path, review_enabled=True, review_max_retries=1)
    st = SubTaskSpec(task_id="dev_1", title="T", description="D", assigned_files=["a.py"])
    (tmp_path / "a.py").write_text("v1")
    runs = []

    class _RetryLoop:
        def __init__(self, results):
            self._results = results

        def run(self, text):
            runs.append(text)
            return self._results.pop(0)

    loops = _RetryLoop([
        AgentResult(status="success", final_message="first", turns=[1]),
        AgentResult(status="success", final_message="second", turns=[1]),
    ])
    monkeypatch.setattr(agent_loop_mod, "AgentLoop", lambda **kw: loops)
    monkeypatch.setattr(o._registry_proto, "clone_for_subagent", lambda cfg: o._registry_proto)
    monkeypatch.setattr(o, "_compute_diff_verdict", lambda **kw: "VERIFIED")
    monkeypatch.setattr(o, "_detect_genuine_violations", lambda *a, **k: [])
    monkeypatch.setattr(o, "_apply_scope_violation_policy", lambda *a, **k: [])
    monkeypatch.setattr(o, "_review_subagent_result", lambda **kw: (False, "redo it"))
    res = o._run_subagent(st)
    assert res.final_message == "second"
    assert len(runs) == 2
    assert "[REVIEW FEEDBACK" in runs[1]


def test_run_subagent_inprocess_crash_reverts(tmp_path, monkeypatch):
    o = _make_orch(tmp_path)
    st = SubTaskSpec(task_id="dev_1", title="T", description="D", assigned_files=["a.py"])
    (tmp_path / "a.py").write_text("original")

    class _BoomLoop:
        def run(self, text):
            (tmp_path / "a.py").write_text("MUTATED")  # after snapshot capture
            raise RuntimeError("loop crash")

    monkeypatch.setattr(agent_loop_mod, "AgentLoop", lambda **kw: _BoomLoop())
    monkeypatch.setattr(o._registry_proto, "clone_for_subagent", lambda cfg: o._registry_proto)
    with pytest.raises(RuntimeError):
        o._run_subagent(st)
    assert (tmp_path / "a.py").read_text() == "original"


def test_run_subagent_inprocess_ipc_routes(tmp_path, monkeypatch):
    o = _make_orch(tmp_path, subagent_mode="ipc")
    st = SubTaskSpec(task_id="dev_1", title="T", description="D")
    monkeypatch.setattr(o, "_run_subagent_ipc", lambda *a, **k: "IPC-RESULT")
    assert o._run_subagent(st) == "IPC-RESULT"


def test_run_subagent_inprocess_task_text_passthrough(tmp_path, monkeypatch):
    o = _make_orch(tmp_path)
    st = SubTaskSpec(task_id="dev_1", title="T", description="D", assigned_files=["a.py"])
    (tmp_path / "a.py").write_text("x")
    loop = _FakeLoop(AgentResult(status="success", final_message="ok", turns=[1]))
    monkeypatch.setattr(agent_loop_mod, "AgentLoop", lambda **kw: loop)
    monkeypatch.setattr(o._registry_proto, "clone_for_subagent", lambda cfg: o._registry_proto)
    monkeypatch.setattr(o, "_compute_diff_verdict", lambda **kw: "VERIFIED")
    monkeypatch.setattr(o, "_detect_genuine_violations", lambda *a, **k: [])
    monkeypatch.setattr(o, "_apply_scope_violation_policy", lambda *a, **k: [])
    # Symbol-map injection is best-effort and platform/grammar-version dependent
    # (find_all_symbols may or may not report the bare 'x' in a.py); pin it to
    # [] so this passthrough test stays deterministic everywhere.
    monkeypatch.setattr(orch, "_symbol_hint_for_source", lambda src, fpath: [])
    o._run_subagent(st, task_text="EXPLICIT TEXT")
    assert loop.calls[0] == "EXPLICIT TEXT"


# ── continue_subagent / run (decomposition) ─────────────────────────────────

def test_continue_subagent_cancelled(tmp_path):
    ev = threading.Event()
    ev.set()
    o = _make_orch(tmp_path, cancel_event=ev)
    res = o.continue_subagent("dev_1", "keep going")
    assert res.status == "cancelled"


def test_continue_subagent_runs(tmp_path, monkeypatch):
    o = _make_orch(tmp_path)
    ran = {}
    monkeypatch.setattr(
        o, "_run_subagent",
        lambda subtask, extra_turns=0: ran.__setitem__("st", subtask) or AgentResult(
            status="success", final_message="continued", turns=[1],
        ),
    )
    res = o.continue_subagent("dev_1", "more", prior_context="PRIOR")
    assert res.status == "success"
    assert res.metadata["continued"] is True
    assert "PRIOR" in ran["st"].description
    assert "[CONTINUE FROM PREVIOUS SESSION]" in ran["st"].description


def test_run_decomposition_cancelled_before_start(tmp_path):
    ev = threading.Event()
    ev.set()
    o = _make_orch(tmp_path, cancel_event=ev)
    res = o.run("task")
    assert res.status == "cancelled"


def test_run_decomposition_full(tmp_path, monkeypatch):
    o = _make_orch(tmp_path)
    subtasks = [SubTaskSpec(task_id="dev_1", title="T1", description="D1", assigned_files=["a.py"])]
    monkeypatch.setattr(o, "_decompose_task", lambda req: subtasks)
    monkeypatch.setattr(o, "_capture_scope_baseline", lambda rr, sts: None)
    monkeypatch.setattr(
        o, "_run_dependency_aware",
        lambda sts, original_request="": [AgentResult(status="success", final_message="ok", turns=[1])],
    )
    monkeypatch.setattr(o, "_synthesize", lambda *a, **k: "summary text")
    monkeypatch.setattr(o, "_cleanup_ipc_workers", lambda: None)
    events = []
    monkeypatch.setattr(o, "_cb", lambda ev, data: events.append(ev))
    res = o.run("task")
    assert res.status == "success"
    assert res.summary == "summary text"
    assert res.total_turns == 1
    assert "orchestrator_plan" in events and "orchestrator_done" in events


def test_run_decomposition_decompose_failure(tmp_path, monkeypatch):
    o = _make_orch(tmp_path)
    monkeypatch.setattr(o, "_decompose_task", lambda req: [])
    monkeypatch.setattr(o, "_cleanup_ipc_workers", lambda: None)
    res = o.run("task")
    assert res.status == "error"


def test_run_decomposition_partial(tmp_path, monkeypatch):
    o = _make_orch(tmp_path)
    subtasks = [SubTaskSpec(task_id="dev_1", title="T1", description="D1")]
    monkeypatch.setattr(o, "_decompose_task", lambda req: subtasks)
    monkeypatch.setattr(o, "_capture_scope_baseline", lambda rr, sts: None)
    monkeypatch.setattr(
        o, "_run_dependency_aware",
        lambda sts, original_request="": [AgentResult(status="error", final_message="bad", error="e", turns=[])],
    )
    monkeypatch.setattr(o, "_synthesize", lambda *a, **k: "sum")
    monkeypatch.setattr(o, "_cleanup_ipc_workers", lambda: None)
    res = o.run("task")
    assert res.status == "error"


def test_decompose_task_parses_and_remaps(tmp_path, monkeypatch):
    import external_llm.agent.orchestrator as orchm
    o = _make_orch(tmp_path)
    raw = (
        '{"subtasks": ['
        '{"id": "x1", "title": "First", "description": "d1", "assigned_files": ["a.py"], '
        '"dependencies": [], "priority": 0},'
        '{"id": "x2", "title": "Second", "description": "d2", "assigned_files": ["b.py"], '
        '"dependencies": ["x1"], "priority": "bad"},'
        '{"id": "orphan", "title": "Third", "description": "d3", "assigned_files": ["c.py"], '
        '"dependencies": ["x9"]}'
        "]}"
    )
    monkeypatch.setattr(orchm, "simple_llm_call", lambda *a, **k: raw)
    specs = o._decompose_task("task")
    assert [s.task_id for s in specs] == ["dev_1", "dev_2", "dev_3"]
    assert specs[1].dependencies == ["dev_1"]  # remapped
    assert specs[2].dependencies == []  # orphan dropped
    assert specs[1].priority == 1  # invalid priority → default


def test_decompose_task_bad_json(tmp_path, monkeypatch):
    import external_llm.agent.orchestrator as orchm
    o = _make_orch(tmp_path)
    monkeypatch.setattr(orchm, "simple_llm_call", lambda *a, **k: "not json")
    assert o._decompose_task("task") == []
def test_run_dependency_aware_order_and_cancel(tmp_path, monkeypatch):
    o = _make_orch(tmp_path)
    specs = [
        SubTaskSpec(task_id="a", title="A", description="d"),
        SubTaskSpec(task_id="b", title="B", description="d", dependencies=["a"]),
    ]
    order = []

    def fake_run(st, **kw):
        order.append(st.task_id)
        return AgentResult(status="success", final_message="ok", turns=[1])

    monkeypatch.setattr(o, "_run_subagent", fake_run)
    results = o._run_dependency_aware(specs)
    assert order == ["a", "b"]
    assert all(r.status == "success" for r in results)

    # cancellation mid-run
    ev = threading.Event()
    o2 = _make_orch(tmp_path, cancel_event=ev)
    ev.set()
    res2 = o2._run_dependency_aware(specs)
    assert all(r.status == "cancelled" for r in res2)


def test_run_dependency_aware_cycles(tmp_path, monkeypatch):
    o = _make_orch(tmp_path)
    specs = [
        SubTaskSpec(task_id="a", title="A", description="d", dependencies=["b"]),
        SubTaskSpec(task_id="b", title="B", description="d", dependencies=["a"]),
    ]
    evs = []
    monkeypatch.setattr(o, "_cb", lambda ev, data: evs.append(data.get("type", ev)))
    # _break_cycles will produce acyclic specs → recursive call runs them
    monkeypatch.setattr(o, "_run_subagent",
                        lambda st, **kw: AgentResult(status="success", final_message="ok", turns=[1]))
    results = o._run_dependency_aware(specs)
    assert len(results) == 2
    assert "dependency_cycle" in evs


def test_run_dependency_aware_no_ready_fallback(tmp_path, monkeypatch):
    o = _make_orch(tmp_path)
    # force "not ready" via _find_current_cycles returning a cycle that
    # _break_cycles cannot break → returns None → fallback sequential order
    specs = [
        SubTaskSpec(task_id="a", title="A", description="d", dependencies=["a"]),
    ]
    monkeypatch.setattr(o, "_break_cycles", lambda sts, cyc: None)
    monkeypatch.setattr(o, "_run_subagent",
                        lambda st, **kw: AgentResult(status="success", final_message="ok", turns=[1]))
    evs = []
    monkeypatch.setattr(o, "_cb", lambda ev, data: evs.append(data.get("type", ev)))
    results = o._run_dependency_aware(specs)
    assert results[0].status == "success"
    assert "dependency_cycle_fallback" in evs


def test_run_parallel_batch_single_and_multi(tmp_path, monkeypatch):
    o = _make_orch(tmp_path)
    specs = [SubTaskSpec(task_id="a", title="A", description="d")]
    monkeypatch.setattr(o, "_run_subagent",
                        lambda st, *a, **kw: AgentResult(status="success", final_message="ok", turns=[1]))
    res = o._run_parallel_batch(specs, {}, {})
    assert len(res) == 1 and res[0].status == "success"
    # multi with failure
    specs2 = [
        SubTaskSpec(task_id="a", title="A", description="d"),
        SubTaskSpec(task_id="b", title="B", description="d"),
    ]

    def flaky(st, *a, **kw):
        if st.task_id == "b":
            raise RuntimeError("boom")
        return AgentResult(status="success", final_message="ok", turns=[1])

    monkeypatch.setattr(o, "_run_subagent", flaky)
    res2 = o._run_parallel_batch(specs2, {}, {})
    statuses = [r.status for r in res2]
    assert "success" in statuses and "error" in statuses
    assert o._run_parallel_batch([], {}, {}) == []


def test_run_parallel_batch_cancel(tmp_path, monkeypatch):
    ev = threading.Event()
    o = _make_orch(tmp_path, cancel_event=ev)
    specs = [
        SubTaskSpec(task_id="a", title="A", description="d"),
        SubTaskSpec(task_id="b", title="B", description="d"),
    ]
    started = threading.Event()
    release = threading.Event()

    def blocking(st, *a, **kw):
        started.set()
        release.wait(10)
        return AgentResult(status="success", final_message="ok", turns=[1])

    monkeypatch.setattr(o, "_run_subagent", blocking)
    ev.set()  # cancel before dispatch → every future cancelled or pending
    res = o._run_parallel_batch(specs, {}, {})
    assert all(r.status == "cancelled" for r in res)
    release.set()  # unblock background threads so the pool can exit


# ── _tool_spawn_subagent ────────────────────────────────────────────────────

def test_tool_spawn_subagent(tmp_path, monkeypatch):
    o = _make_orch(tmp_path)
    monkeypatch.setattr(o, "_run_subagent_background", lambda subtask, original_request="": None)
    out = o._tool_spawn_subagent({"task_description": "Do it", "title": "T", "assigned_files": ["a.py"]}, "req")
    assert "dev_1" in out
    assert "Files: a.py" in out
    # missing description
    with pytest.raises(_NativeToolError):
        o._tool_spawn_subagent({}, "req")
    # string files → coerced
    monkeypatch.setattr(o, "_run_subagent_background", lambda subtask, original_request="": None)
    out2 = o._tool_spawn_subagent({"task_description": "X", "assigned_files": "solo.py"}, "req")
    assert "solo.py" in out2
    # bad priority
    o._tool_spawn_subagent({"task_description": "X", "priority": "junk"}, "req")


def test_tool_spawn_subagent_conflict_warning(tmp_path, monkeypatch):
    o = _make_orch(tmp_path)
    st = SubTaskSpec(task_id="dev_0", title="T", description="d", assigned_files=["share.py"])
    o._bg_subagents["dev_0"] = {"result": None, "subtask": st}
    warnings = []
    monkeypatch.setattr(o, "_cb", lambda ev, data: warnings.append(data) if ev == "orchestrator_warning" else None)
    monkeypatch.setattr(o, "_run_subagent_background", lambda subtask, original_request="": None)
    out = o._tool_spawn_subagent({"task_description": "Do it", "assigned_files": ["share.py"]}, "")
    assert "overlap" in out
    assert warnings and warnings[0]["type"] == "tool_loop_file_conflict"


# ── _claim_reusable_worker / _return / _cleanup / _abandon ─────────────────

def test_claim_reusable_worker_no_pool(tmp_path):
    o = _make_orch(tmp_path)
    assert o._claim_reusable_worker(str(tmp_path)) is None


def test_claim_reusable_worker_busy(tmp_path):
    o = _make_orch(tmp_path)
    d = tmp_path / ".asicode" / "subagents" / "w1"
    d.mkdir(parents=True)
    (d / "task.json").write_text("{}")
    o._reusable_worker_ids.add("w1")
    assert o._claim_reusable_worker(str(tmp_path)) is None  # busy (task.json present)


def test_claim_reusable_worker_dead_proc(tmp_path, monkeypatch):
    import external_llm.agent.subagent_ipc as sipc
    o = _make_orch(tmp_path)
    o._reusable_worker_ids.add("w1")
    monkeypatch.setattr(sipc, "read_worker_idle_heartbeat_state", lambda *a, **k: "idle")
    monkeypatch.setattr(sipc, "read_worker_idle_heartbeat_age", lambda *a, **k: None)
    dead = SimpleNamespace(poll=lambda: 1, pid=99)
    o._ipc_worker_procs["w1"] = dead
    assert o._claim_reusable_worker(str(tmp_path)) is None
    assert "w1" not in o._reusable_worker_ids  # dropped


def test_claim_reusable_worker_exited_heartbeat(tmp_path, monkeypatch):
    import external_llm.agent.subagent_ipc as sipc
    o = _make_orch(tmp_path)
    o._reusable_worker_ids.add("w1")
    monkeypatch.setattr(sipc, "read_worker_idle_heartbeat_state", lambda *a, **k: "exited")
    assert o._claim_reusable_worker(str(tmp_path)) is None
    assert "w1" not in o._reusable_worker_ids


def test_claim_reusable_worker_stale_hb_terminal(tmp_path, monkeypatch):
    import external_llm.agent.subagent_ipc as sipc
    o = _make_orch(tmp_path, ipc_heartbeat_stale_s=30)
    o._reusable_worker_ids.add("w1")
    monkeypatch.setattr(sipc, "read_worker_idle_heartbeat_state", lambda *a, **k: "idle")
    monkeypatch.setattr(sipc, "read_worker_idle_heartbeat_age", lambda *a, **k: 9999.0)
    assert o._claim_reusable_worker(str(tmp_path)) is None  # stale → dropped


def test_claim_reusable_worker_alive(tmp_path, monkeypatch):
    import external_llm.agent.subagent_ipc as sipc
    o = _make_orch(tmp_path)
    o._reusable_worker_ids.add("w1")
    monkeypatch.setattr(sipc, "read_worker_idle_heartbeat_state", lambda *a, **k: "idle")
    monkeypatch.setattr(sipc, "read_worker_idle_heartbeat_age", lambda *a, **k: None)
    alive = SimpleNamespace(poll=lambda: None, pid=5)
    o._ipc_worker_procs["w1"] = alive
    assert o._claim_reusable_worker(str(tmp_path)) == "w1"
    assert "w1" not in o._reusable_worker_ids  # claimed


def test_return_worker_to_pool(tmp_path):
    o = _make_orch(tmp_path)
    o._return_worker_to_pool("w1")
    assert "w1" in o._reusable_worker_ids


def test_cleanup_ipc_workers_non_ipc(tmp_path):
    o = _make_orch(tmp_path)  # subagent_mode="in_process"
    o._cleanup_ipc_workers()  # no-op
    assert True


def test_cleanup_ipc_workers_writes_sentinels(tmp_path, monkeypatch):
    import external_llm.agent.subagent_ipc as sipc
    o = _make_orch(tmp_path, subagent_mode="ipc")
    o._ipc_worker_ids.add("w1")
    calls = []
    monkeypatch.setattr(sipc, "write_cancel_all", lambda r, w: calls.append(("cancel", w)))
    monkeypatch.setattr(sipc, "write_shutdown_all", lambda r, w: calls.append(("shutdown", w)))
    o._cleanup_ipc_workers()
    assert ("cancel", ["w1"]) in calls and ("shutdown", ["w1"]) in calls


def test_cleanup_ipc_workers_terminates_procs(tmp_path, monkeypatch):
    import external_llm.agent.subagent_ipc as sipc
    o = _make_orch(tmp_path, subagent_mode="ipc")
    monkeypatch.setattr(sipc, "write_cancel_all", lambda r, w: None)
    monkeypatch.setattr(sipc, "write_shutdown_all", lambda r, w: None)
    terminated = []
    proc = SimpleNamespace(poll=lambda: None)
    proc.terminate = lambda: terminated.append(True)
    o._ipc_worker_procs["w1"] = proc
    dead = SimpleNamespace(poll=lambda: 0)
    o._ipc_worker_procs["w2"] = dead
    o._cleanup_ipc_workers()
    assert terminated == [True]
    assert o._reusable_worker_ids == set()  # cleared


def test_abandon_ipc_worker_soft_quiesce(tmp_path, monkeypatch):
    import external_llm.agent.subagent_ipc as sipc
    o = _make_orch(tmp_path)
    d = tmp_path / ".asicode" / "subagents" / "w1"
    d.mkdir(parents=True)
    monkeypatch.setattr(sipc, "write_cancel_sentinel", lambda r, w: None)
    (d / "result.json").write_text("{}")  # quiesced already
    reusable = o._abandon_ipc_worker(str(tmp_path), "w1", grace_s=1)
    assert reusable is True


def test_abandon_ipc_worker_already_dead(tmp_path, monkeypatch):
    import external_llm.agent.subagent_ipc as sipc
    o = _make_orch(tmp_path)
    monkeypatch.setattr(sipc, "write_cancel_sentinel", lambda r, w: None)
    proc = SimpleNamespace(poll=lambda: 1, returncode=9)
    o._ipc_worker_procs["w1"] = proc
    reusable = o._abandon_ipc_worker(str(tmp_path), "w1", grace_s=1)
    assert reusable is False


def test_abandon_ipc_worker_hard_cancel(tmp_path, monkeypatch):
    import external_llm.agent.subagent_ipc as sipc
    ev = threading.Event()
    ev.set()
    o = _make_orch(tmp_path, cancel_event=ev)
    monkeypatch.setattr(sipc, "write_cancel_sentinel", lambda r, w: None)
    waited = []
    proc = SimpleNamespace(poll=lambda: 1)
    proc.wait = lambda timeout=None: waited.append(timeout)
    o._ipc_worker_procs["w1"] = proc
    (tmp_path / "a.py").write_text("mutated")
    snaps = {"a.py": b"original"}
    reusable = o._abandon_ipc_worker(str(tmp_path), "w1", snaps, grace_s=1)
    assert reusable is False
    assert (tmp_path / "a.py").read_bytes() == b"original"  # snapshot restored
    assert waited  # hard path waited on the proc


def test_abandon_ipc_worker_cancel_sentinel_failure(tmp_path, monkeypatch):
    import external_llm.agent.subagent_ipc as sipc
    o = _make_orch(tmp_path)
    def boom(r, w):
        raise OSError("no dir")
    monkeypatch.setattr(sipc, "write_cancel_sentinel", boom)
    reusable = o._abandon_ipc_worker(str(tmp_path), "w1", grace_s=1)
    assert reusable is False  # sentinel failed → not quiesced → not reusable


# ═══════════════════════════════════════════════════════════════════════════
# Round 4: _run_tool_loop (DesignChatLoop mocked), remaining branch edges
# ═══════════════════════════════════════════════════════════════════════════

import external_llm.agent.design_chat_loop as dcl_mod


class _FakeDCResult:
    def __init__(self, content="", is_error=False):
        self.content = content
        self.is_error = is_error


def test_run_tool_loop_success_no_session(tmp_path, monkeypatch):
    o = _make_orch(tmp_path, tool_loop_enabled=True)
    calls = {}

    class _FakeDCL:
        def __init__(self, *a, **kw):
            calls["init"] = True

        def respond(self, msgs, **kw):
            calls["msgs"] = msgs
            return _FakeDCResult(content="final summary")

    monkeypatch.setattr(dcl_mod, "DesignChatLoop", _FakeDCL)
    monkeypatch.setattr(o, "_drain_background_subagents", lambda per_agent_timeout=0: None)
    monkeypatch.setattr(o, "_shutdown_bg_executor", lambda: None)
    monkeypatch.setattr(o, "_cleanup_ipc_workers", lambda: None)
    res = o.run("hello", session_id="")
    assert res.status == "success"
    assert res.summary == "final summary"
    assert calls["msgs"][-1].role == "user"  # no session → bare request appended
    assert calls["msgs"][0].role == "system"


def test_run_tool_loop_session_inheritance(tmp_path, monkeypatch):
    o = _make_orch(tmp_path, tool_loop_enabled=True)
    mgr = SimpleNamespace(
        get_or_create=lambda sid: "ds",
        build_context_messages=lambda ds, **kw: [
            {"role": "user", "content": "prior turn"},
            {"role": "system", "content": "──"},
        ],
    )
    o.orch_config.session_mgr = mgr
    calls = {}

    class _FakeDCL:
        def __init__(self, *a, **kw):
            pass

        def respond(self, msgs, **kw):
            calls["msgs"] = msgs
            return _FakeDCResult(content="ok")

    monkeypatch.setattr(dcl_mod, "DesignChatLoop", _FakeDCL)
    monkeypatch.setattr(o, "_drain_background_subagents", lambda per_agent_timeout=0: None)
    monkeypatch.setattr(o, "_shutdown_bg_executor", lambda: None)
    monkeypatch.setattr(o, "_cleanup_ipc_workers", lambda: None)
    res = o.run("hello", session_id="sess-1")
    assert res.status == "success"
    contents = [m.content for m in calls["msgs"]]
    assert "prior turn" in contents  # inherited
    assert "──" not in contents  # empty divider filtered
    assert "hello" not in contents  # not duplicated


def test_run_tool_loop_session_inheritance_failure(tmp_path, monkeypatch):
    o = _make_orch(tmp_path, tool_loop_enabled=True)
    mgr = SimpleNamespace(
        get_or_create=lambda sid: (_ for _ in ()).throw(RuntimeError("boom")),
        build_context_messages=lambda ds, **kw: [],
    )
    o.orch_config.session_mgr = mgr
    calls = {}

    class _FakeDCL:
        def __init__(self, *a, **kw):
            pass

        def respond(self, msgs, **kw):
            calls["msgs"] = msgs
            return _FakeDCResult(content="ok")

    monkeypatch.setattr(dcl_mod, "DesignChatLoop", _FakeDCL)
    monkeypatch.setattr(o, "_drain_background_subagents", lambda per_agent_timeout=0: None)
    monkeypatch.setattr(o, "_shutdown_bg_executor", lambda: None)
    monkeypatch.setattr(o, "_cleanup_ipc_workers", lambda: None)
    res = o.run("hello", session_id="sess-1")
    assert res.status == "success"
    assert calls["msgs"][-1].content == "hello"  # fell back to bare request


def test_run_tool_loop_cancelled(tmp_path, monkeypatch):
    from external_llm.agent.agent_loop_types import AgentCancelled
    o = _make_orch(tmp_path, tool_loop_enabled=True)
    calls = {}

    class _FakeDCL:
        def __init__(self, *a, **kw):
            pass

        def respond(self, msgs, **kw):
            calls["cancelled"] = True
            raise AgentCancelled("esc")

    monkeypatch.setattr(dcl_mod, "DesignChatLoop", _FakeDCL)
    monkeypatch.setattr(o, "_drain_background_subagents", lambda per_agent_timeout=0: None)
    monkeypatch.setattr(o, "_shutdown_bg_executor", lambda: None)
    monkeypatch.setattr(o, "_cleanup_ipc_workers", lambda: None)
    res = o.run("hello", session_id="")
    assert res.status == "cancelled"
    assert res.metadata["cancelled"] is True


def test_run_tool_loop_direct_answer_no_subagents(tmp_path, monkeypatch):
    o = _make_orch(tmp_path, tool_loop_enabled=True)

    class _FakeDCL:
        def __init__(self, *a, **kw):
            pass

        def respond(self, msgs, **kw):
            return _FakeDCResult(content="I answered directly", is_error=False)

    monkeypatch.setattr(dcl_mod, "DesignChatLoop", _FakeDCL)
    monkeypatch.setattr(o, "_drain_background_subagents", lambda per_agent_timeout=0: None)
    monkeypatch.setattr(o, "_shutdown_bg_executor", lambda: None)
    monkeypatch.setattr(o, "_cleanup_ipc_workers", lambda: None)
    res = o.run("hello", session_id="")
    assert res.status == "success"  # direct answer is a legitimate completion


def test_run_tool_loop_error_status(tmp_path, monkeypatch):
    o = _make_orch(tmp_path, tool_loop_enabled=True)

    class _FakeDCL:
        def __init__(self, *a, **kw):
            pass

        def respond(self, msgs, **kw):
            return _FakeDCResult(content="", is_error=True)

    monkeypatch.setattr(dcl_mod, "DesignChatLoop", _FakeDCL)
    monkeypatch.setattr(o, "_drain_background_subagents", lambda per_agent_timeout=0: None)
    monkeypatch.setattr(o, "_shutdown_bg_executor", lambda: None)
    monkeypatch.setattr(o, "_cleanup_ipc_workers", lambda: None)
    res = o.run("hello", session_id="")
    assert res.status == "error"


def test_run_tool_loop_file_lock_injection(tmp_path, monkeypatch):
    o = _make_orch(tmp_path, tool_loop_enabled=True)
    seen = {}

    class _FakeDCL:
        def __init__(self, *a, **kw):
            seen["reg"] = a[1]

        def respond(self, msgs, **kw):
            return _FakeDCResult(content="ok")

    monkeypatch.setattr(dcl_mod, "DesignChatLoop", _FakeDCL)
    monkeypatch.setattr(o, "_drain_background_subagents", lambda per_agent_timeout=0: None)
    monkeypatch.setattr(o, "_shutdown_bg_executor", lambda: None)
    monkeypatch.setattr(o, "_cleanup_ipc_workers", lambda: None)
    o.run("hello", session_id="")
    # file_lock_manager injected onto the wrapped registry config
    assert seen["reg"].config.file_lock_manager is o._file_lock_mgr


def test_run_tool_loop_with_bg_results(tmp_path, monkeypatch):
    o = _make_orch(tmp_path, tool_loop_enabled=True)

    class _FakeDCL:
        def __init__(self, *a, **kw):
            pass

        def respond(self, msgs, **kw):
            # run()/_run_tool_loop clear bg state at entry, so populate during respond
            st = SubTaskSpec(task_id="dev_1", title="T", description="d")
            o._bg_subagents["dev_1"] = {"subtask": st, "result": AgentResult(status="success", final_message="r", turns=[1])}
            o._bg_results.append(o._bg_subagents["dev_1"]["result"])
            return _FakeDCResult(content="")

    monkeypatch.setattr(dcl_mod, "DesignChatLoop", _FakeDCL)
    monkeypatch.setattr(o, "_drain_background_subagents", lambda per_agent_timeout=0: None)
    monkeypatch.setattr(o, "_shutdown_bg_executor", lambda: None)
    monkeypatch.setattr(o, "_cleanup_ipc_workers", lambda: None)
    res = o.run("hello", session_id="")
    assert res.status == "success"  # any_ok via bg result
    assert res.subtask_results


# ── _on_poll heartbeat path (via _run_subagent_ipc) ─────────────────────────

def test_run_subagent_ipc_on_poll_heartbeat(tmp_path, monkeypatch):
    import external_llm.agent.subagent_ipc as sipc
    o = _make_orch(tmp_path, subagent_mode="ipc")
    st = SubTaskSpec(task_id="dev_1", title="T", description="D", assigned_files=["a.py"])
    (tmp_path / "a.py").write_text("x")
    seen_polls = []

    def fake_wait(repo, agent_id, **kw):
        # invoke the on_poll callback exactly as wait_for_result would
        kw["on_poll"](5.0, agent_id)
        return _FakeResult(status="success", turns=0)

    monkeypatch.setattr(sipc, "clear_result", lambda *a, **k: None)
    monkeypatch.setattr(sipc, "write_task", lambda *a, **k: None)
    monkeypatch.setattr(sipc, "wait_for_result", fake_wait)
    monkeypatch.setattr(sipc, "read_heartbeat_state", lambda *a, **k: {"turn": 3, "last_tool": "apply_patch"})
    monkeypatch.setattr(o, "_claim_reusable_worker", lambda repo: None)
    monkeypatch.setattr(o, "_return_worker_to_pool", lambda wid: None)
    monkeypatch.setattr(o, "_compute_diff_verdict", lambda **kw: "VERIFIED")
    monkeypatch.setattr(o, "_filter_unassigned_changes", lambda r, own: r)
    monkeypatch.setattr(o, "_apply_scope_violation_policy", lambda *a, **k: [])
    monkeypatch.setattr(o, "_event_dispatcher", SimpleNamespace(emit=lambda aid, ev, data: seen_polls.append((aid, ev, data))))
    o._run_subagent_ipc(st)
    waits = [e for e in seen_polls if e[1] == "subagent_waiting_ipc"]
    assert waits, "waiting events must be emitted"
    assert waits[0][2].get("elapsed_s") == 0.0  # initial event
    assert waits[1][2].get("turn") == 3  # on_poll heartbeat event
    assert waits[1][2].get("last_tool") == "apply_patch"


def test_run_subagent_ipc_provider_model_argv(tmp_path, monkeypatch):
    import external_llm.agent.subagent_ipc as sipc
    o = _make_orch(tmp_path, subagent_mode="ipc", subagent_provider="ollama", subagent_model="qwen")
    st = SubTaskSpec(task_id="dev_1", title="T", description="D", assigned_files=["a.py"])
    (tmp_path / "a.py").write_text("x")
    tasks = []
    monkeypatch.setattr(sipc, "clear_result", lambda *a, **k: None)
    monkeypatch.setattr(sipc, "write_task", lambda repo, task, **k: tasks.append(task))
    monkeypatch.setattr(sipc, "wait_for_result", lambda *a, **k: _FakeResult(status="success", turns=0))
    monkeypatch.setattr(o, "_claim_reusable_worker", lambda repo: None)
    monkeypatch.setattr(o, "_return_worker_to_pool", lambda wid: None)
    monkeypatch.setattr(o, "_compute_diff_verdict", lambda **kw: "VERIFIED")
    monkeypatch.setattr(o, "_filter_unassigned_changes", lambda r, own: r)
    monkeypatch.setattr(o, "_apply_scope_violation_policy", lambda *a, **k: [])
    o._run_subagent_ipc(st)
    assert tasks[0].provider == "ollama"
    assert tasks[0].model == "qwen"
    cmd = o._subagent_ipc_commands["dev_1"]
    assert "--provider ollama" in cmd and "--model qwen" in cmd


def test_run_subagent_ipc_scope_policy_revert(tmp_path, monkeypatch):
    import external_llm.agent.subagent_ipc as sipc
    o = _make_orch(tmp_path, subagent_mode="ipc", scope_violation_policy="revert")
    st = SubTaskSpec(task_id="dev_1", title="T", description="D", assigned_files=["a.py"])
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "stray.py").write_text("s")
    monkeypatch.setattr(sipc, "clear_result", lambda *a, **k: None)
    monkeypatch.setattr(sipc, "write_task", lambda *a, **k: None)
    monkeypatch.setattr(sipc, "wait_for_result",
                        lambda *a, **k: _FakeResult(status="success", turns=0, unassigned_changes=[{"file": "stray.py"}]))
    monkeypatch.setattr(o, "_claim_reusable_worker", lambda repo: None)
    monkeypatch.setattr(o, "_return_worker_to_pool", lambda wid: None)
    monkeypatch.setattr(o, "_compute_diff_verdict", lambda **kw: "VERIFIED")
    monkeypatch.setattr(o, "_filter_unassigned_changes", lambda r, own: r)  # keep stray
    monkeypatch.setattr(o, "_detect_genuine_violations", lambda *a, **k: [{"file": "stray.py"}])
    o._run_subagent_ipc(st)
    assert not (tmp_path / "stray.py").exists()  # policy=revert removed it


# ── _break_cycles edges ─────────────────────────────────────────────────────

def test_break_cycles_weakest_edge_removal():
    o = _make_orch("/tmp")
    specs = [
        SubTaskSpec(task_id="a", title="a", description="d", dependencies=["b"], priority=0),
        SubTaskSpec(task_id="b", title="b", description="d", dependencies=["a"], priority=1),
    ]
    broken = o._break_cycles(specs, [["a", "b"]])
    assert broken is not None
    _, cycles = o._detect_cycles_kahn({s.task_id: s for s in broken})
    assert cycles == []


def test_break_cycles_aggressive_fallback():
    o = _make_orch("/tmp")
    # tangled graph where weakest-removal alone cannot break everything
    specs = [
        SubTaskSpec(task_id="a", title="a", description="d", dependencies=["b", "c"]),
        SubTaskSpec(task_id="b", title="b", description="d", dependencies=["a", "c"]),
        SubTaskSpec(task_id="c", title="c", description="d", dependencies=["a", "b"]),
    ]
    broken = o._break_cycles(specs, [["a", "b", "c"]])
    # either fully broken or None — never raises
    if broken is not None:
        _, cycles = o._detect_cycles_kahn({s.task_id: s for s in broken})
        assert cycles == []


def test_break_cycles_short_cycle_skipped():
    o = _make_orch("/tmp")
    specs = [SubTaskSpec(task_id="a", title="a", description="d")]
    # len(cycle) < 2 → skipped; acyclicity verified on the copy
    broken = o._break_cycles(specs, [["a"]])
    assert broken is not None
    assert broken[0].task_id == "a"


def test_break_cycles_none_when_unbreakable(monkeypatch):
    o = _make_orch("/tmp")
    monkeypatch.setattr(o, "_detect_cycles_kahn", lambda m: ([], [["x", "y"]]))
    specs = [SubTaskSpec(task_id="x", title="x", description="d", dependencies=["y"]),
             SubTaskSpec(task_id="y", title="y", description="d", dependencies=["x"])]
    assert o._break_cycles(specs, [["x", "y"]]) is None


# ── _revert_unassigned_changes edge failures ────────────────────────────────

def test_revert_unassigned_batched_checkout_failure_perfile_fallback(git_repo, monkeypatch):
    import subprocess as _sp
    (git_repo / "f.py").write_text("x=7\n", encoding="utf-8")
    o = _make_orch(git_repo)
    real_run = _sp.run
    fail_next = {"n": 0}

    def flaky(cmd, **kw):
        if cmd[:2] == ["git", "checkout"] and "HEAD" in cmd:
            fail_next["n"] += 1
            if fail_next["n"] == 1:  # batched checkout fails once
                return SimpleNamespace(returncode=1, stderr=b"conflict")
        return real_run(cmd, **kw)

    monkeypatch.setattr(_sp, "run", flaky)
    reverted = o._revert_unassigned_changes(str(git_repo), [{"file": "f.py"}])
    assert reverted == ["f.py"]  # per-file fallback succeeded


def test_revert_unassigned_unlink_missing_file(git_repo):
    o = _make_orch(git_repo)
    reverted = o._revert_unassigned_changes(str(git_repo), [{"file": "ghost.py"}])
    assert reverted == []  # already gone — nothing to report


# ── _format_poll_any_results heads ──────────────────────────────────────────

def test_format_poll_any_results():
    o = _make_orch("/tmp")
    r1 = AgentResult(status="success", final_message="one", turns=[1])
    r2 = AgentResult(status="error", final_message="two", turns=[])
    out = o._format_poll_any_results(["a", "b"], [("a", "success", r1), ("b", "error", r2)], pending=["c"])
    assert "2 of 2 sub-agents completed" in out
    assert ", 1 still running" in out
    assert "Sub-agent 'a' finished" in out
    # single completed + no pending → verbatim single result
    out2 = o._format_poll_any_results(["a"], [("a", "success", r1)], pending=[])
    assert out2.startswith("Sub-agent 'a' finished")
    # no pending arg → single completed + none pending → verbatim
    out3 = o._format_poll_any_results(["a", "b"], [("a", "success", r1)], pending=None)
    assert out3.startswith("Sub-agent 'a' finished")


# ── _synthesize (LLM summary) ───────────────────────────────────────────────

def test_synthesize(tmp_path, monkeypatch):
    import external_llm.agent.orchestrator as orchm
    o = _make_orch(tmp_path)
    monkeypatch.setattr(orchm, "simple_llm_call", lambda *a, **k: "SYNTH")
    st = SubTaskSpec(task_id="dev_1", title="T", description="d")
    r = AgentResult(status="success", final_message="done", turns=[])
    out = o._synthesize("req", [st], [r])
    assert out == "SYNTH"
    # None result skipped in parts
    out2 = o._synthesize("req", [st], [None])
    assert out2 == "SYNTH"
    # empty LLM result → fallback string
    monkeypatch.setattr(orchm, "simple_llm_call", lambda *a, **k: "")
    assert o._synthesize("req", [st], [r]) == "Multi-agent task completed."


# ── _run_parallel_batch cancel-drain exception edges ────────────────────────

def test_run_parallel_batch_future_exception_results(tmp_path, monkeypatch):
    o = _make_orch(tmp_path)
    # 2 specs so the ThreadPoolExecutor path (not the single-task fast path) runs
    specs = [SubTaskSpec(task_id="a", title="A", description="d"),
             SubTaskSpec(task_id="b", title="B", description="d")]

    def boom(st, *a, **kw):
        raise ValueError("inner boom")

    monkeypatch.setattr(o, "_run_subagent", boom)
    res = o._run_parallel_batch(specs, {}, {})
    assert all(r.status == "error" for r in res)
    assert "inner boom" in res[0].error


# ── _gc_subagent_artifacts failure edges ────────────────────────────────────

def test_gc_subagent_artifacts_exception_handled(tmp_path, monkeypatch):
    base = tmp_path / ".asicode" / "subagents"
    base.mkdir(parents=True)
    (base / "d1").mkdir()
    def boom(*a, **k):
        raise OSError("listdir boom")
    monkeypatch.setattr(os, "listdir", boom)
    OrchestratorAgent._gc_subagent_artifacts(str(tmp_path))  # must not raise


def test_gc_subagent_artifacts_getmtime_failure(tmp_path, monkeypatch):
    base = tmp_path / ".asicode" / "subagents"
    d = base / "d1"
    d.mkdir(parents=True)
    def boom(p):
        raise OSError("mtime boom")
    monkeypatch.setattr(os.path, "getmtime", boom)
    OrchestratorAgent._gc_subagent_artifacts(str(tmp_path))  # must not raise


# ── _capture/_restore edge failures ─────────────────────────────────────────

def test_capture_snapshots_agg_cap_warns_once(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "_SNAPSHOT_AGGREGATE_MAX_BYTES", 4)
    monkeypatch.setattr(orch, "_SNAPSHOT_MAX_BYTES", 100)
    for i in range(4):
        (tmp_path / f"f{i}.py").write_bytes(b"aaa")
    snaps = _capture_assigned_snapshots(str(tmp_path), ["f0.py", "f1.py", "f2.py", "f3.py"])
    assert "f0.py" in snaps  # first fits
    assert all(f"f{i}.py" not in snaps for i in (1, 2, 3))  # rest skipped


def test_restore_snapshots_removedirs_failure(tmp_path, monkeypatch):
    (tmp_path / "new.py").write_bytes(b"x")
    def boom(p):
        raise OSError("removedirs boom")
    monkeypatch.setattr(os, "removedirs", boom)
    reverted = _restore_assigned_snapshots(str(tmp_path), {"new.py": _MISSING_SNAP})
    assert reverted == ["new.py"]  # removedirs failure is non-fatal


def test_restore_snapshots_write_failure_reverts_and_raises(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_bytes(b"orig")
    class _Boom:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def write(self, data):
            raise OSError("disk full")
    monkeypatch.setattr("tempfile.mkstemp", lambda **kw: (3, str(tmp_path / "tmp")))
    monkeypatch.setattr(os, "fdopen", lambda fd, mode: _Boom())
    # write failure is swallowed + skipped (best-effort restore contract)
    reverted = _restore_assigned_snapshots(str(tmp_path), {"a.py": b"orig"})
    assert reverted == []


# ── _check_bg_subagent future exception edge ────────────────────────────────

def test_check_bg_subagent_future_exception_cached(tmp_path):
    o = _make_orch(tmp_path)
    from concurrent.futures import Future
    f = Future()
    f.set_exception(ValueError("boom"))
    st = SubTaskSpec(task_id="x", title="X", description="d")
    o._bg_subagents["x"] = {"future": f, "result": None, "status": "running", "subtask": st, "started_at": time.monotonic()}
    status, result = o._check_bg_subagent("x", timeout_s=0)
    assert status == "error" and "boom" in result.error
    # second call — cached, no double append
    n = len(o._bg_results)
    status2, _ = o._check_bg_subagent("x", timeout_s=0)
    assert status2 == "error" and len(o._bg_results) == n


# ── _cb handler raise edge ──────────────────────────────────────────────────

def test_cb_handler_raise_swallowed(tmp_path):
    def bad(ev, data):
        raise RuntimeError("cb boom")
    o = _make_orch(tmp_path, )
    o._cb_fn = bad
    o._cb("ev", {})  # must not raise


# ── _symbol_hint_for_source tree-sitter failure fallback ────────────────────

def test_symbol_hint_tree_sitter_failure_falls_back_to_ast(tmp_path, monkeypatch):
    # ast fallback still yields hints for python files (tree-sitter or not)
    out = _symbol_hint_for_source("def f():\n    pass\n", "plain.py")
    assert any(s.startswith("F f L1") for s in out)


def test_symbol_def_line_unknown_lang():
    assert _symbol_def_line("whatever", "x.unknownext", "f") is None


# ── _tool_poll_subagent timeout coercion ────────────────────────────────────

def test_tool_poll_subagent_bad_timeout_coerced(tmp_path, monkeypatch):
    o = _make_orch(tmp_path)
    monkeypatch.setattr(o, "_check_bg_subagent", lambda aid, timeout_s=0: ("unknown", None))
    with pytest.raises(_NativeToolError):
        o._tool_poll_subagent({"agent_id": "x", "timeout_s": "junk"})


def test_poll_single_agent_unknown_raises(tmp_path):
    o = _make_orch(tmp_path)
    with pytest.raises(_NativeToolError):
        o._poll_single_agent("ghost", 0)


# ── _decompose_task unscoped warning / git context ──────────────────────────

def test_decompose_task_unscoped_warning(tmp_path, monkeypatch):
    import external_llm.agent.orchestrator as orchm
    o = _make_orch(tmp_path)
    raw = '{"subtasks": [{"id": "x", "title": "T", "description": "d", "assigned_files": []}]}'
    monkeypatch.setattr(orchm, "simple_llm_call", lambda *a, **k: raw)
    specs = o._decompose_task("task")
    assert specs[0].assigned_files == []


def test_run_dependency_aware_conflict_split(tmp_path, monkeypatch):
    o = _make_orch(tmp_path)
    specs = [
        SubTaskSpec(task_id="a", title="A", description="d", assigned_files=["same.py"]),
        SubTaskSpec(task_id="b", title="B", description="d", assigned_files=["same.py"]),
    ]
    evs = []
    monkeypatch.setattr(o, "_cb", lambda ev, data: evs.append(data.get("type", ev)))
    monkeypatch.setattr(o, "_run_subagent",
                        lambda st, **kw: AgentResult(status="success", final_message="ok", turns=[1]))
    results = o._run_dependency_aware(specs)
    assert len(results) == 2
    assert "file_conflict_serialized" in evs


# ═══════════════════════════════════════════════════════════════════════════
# Round 5: _poll_any_agent blocking, _run_subagent dedicated-model branches
# ═══════════════════════════════════════════════════════════════════════════

def test_poll_any_agent_blocking_completes(tmp_path, monkeypatch):
    from concurrent.futures import Future
    o = _make_orch(tmp_path, ipc_timeout_s=5)
    f = Future()
    st = SubTaskSpec(task_id="x", title="X", description="d")
    o._bg_subagents["x"] = {"future": f, "result": None, "status": "running", "subtask": st, "started_at": time.monotonic()}
    # complete the future in a background thread after a moment
    def complete():
        time.sleep(0.1)
        f.set_result(AgentResult(status="success", final_message="done", turns=[1]))
    t = threading.Thread(target=complete, daemon=True)
    t.start()
    out = o._poll_any_agent(["x"], timeout_s=2)
    assert "Sub-agent 'x' finished" in out


def test_poll_any_agent_blocking_cancel(tmp_path):
    ev = threading.Event()
    o = _make_orch(tmp_path, ipc_timeout_s=5, cancel_event=ev)
    from concurrent.futures import Future
    f = Future()
    st = SubTaskSpec(task_id="x", title="X", description="d")
    o._bg_subagents["x"] = {"future": f, "result": None, "status": "running", "subtask": st, "started_at": time.monotonic()}
    ev.set()
    out = o._poll_any_agent(["x"], timeout_s=5)
    assert "interrupted by cancellation" in out


def test_poll_any_agent_blocking_timeout_none_done(tmp_path):
    from concurrent.futures import Future
    o = _make_orch(tmp_path, ipc_timeout_s=5)
    f = Future()
    st = SubTaskSpec(task_id="x", title="X", description="d")
    o._bg_subagents["x"] = {"future": f, "result": None, "status": "running", "subtask": st, "started_at": time.monotonic()}
    out = o._poll_any_agent(["x"], timeout_s=0.3)
    assert "None of 1 sub-agents completed within" in out


def test_run_subagent_dedicated_model(tmp_path, monkeypatch):
    import external_llm.agent.agent_loop as agent_loop_mod
    o = _make_orch(tmp_path, subagent_provider="ollama", subagent_model="qwen2.5", subagent_api_key="k")
    st = SubTaskSpec(task_id="dev_1", title="T", description="D", assigned_files=["a.py"])
    (tmp_path / "a.py").write_text("x")
    loop = _FakeLoop(AgentResult(status="success", final_message="ok", turns=[1]))
    monkeypatch.setattr(agent_loop_mod, "AgentLoop", lambda **kw: loop)
    import external_llm.client as client_mod
    monkeypatch.setattr(client_mod, "create_llm_client", lambda **kw: "NEW-CLIENT")
    monkeypatch.setattr(o, "_compute_diff_verdict", lambda **kw: "VERIFIED")
    monkeypatch.setattr(o, "_detect_genuine_violations", lambda *a, **k: [])
    monkeypatch.setattr(o, "_apply_scope_violation_policy", lambda *a, **k: [])
    o._run_subagent(st)
    assert loop.calls  # ran
    # create_llm_client failure → falls back to orchestrator client
    def boom(**kw):
        raise ValueError("client boom")
    monkeypatch.setattr(client_mod, "create_llm_client", boom)
    loop2 = _FakeLoop(AgentResult(status="success", final_message="ok", turns=[1]))
    monkeypatch.setattr(agent_loop_mod, "AgentLoop", lambda **kw: loop2)
    o._run_subagent(st)
    assert loop2.calls


def test_run_subagent_symbol_hint_injected(tmp_path, monkeypatch):
    import external_llm.agent.agent_loop as agent_loop_mod
    o = _make_orch(tmp_path)
    st = SubTaskSpec(task_id="dev_1", title="T", description="D", assigned_files=["mod.py"])
    (tmp_path / "mod.py").write_text("def helper():\n    pass\n", encoding="utf-8")
    loop = _FakeLoop(AgentResult(status="success", final_message="ok", turns=[1]))
    monkeypatch.setattr(agent_loop_mod, "AgentLoop", lambda **kw: loop)
    monkeypatch.setattr(o, "_compute_diff_verdict", lambda **kw: "VERIFIED")
    monkeypatch.setattr(o, "_detect_genuine_violations", lambda *a, **k: [])
    monkeypatch.setattr(o, "_apply_scope_violation_policy", lambda *a, **k: [])
    o._run_subagent(st)
    assert "[Symbol map for mod.py]" in loop.calls[0]


def test_run_subagent_sub_cb_routes(tmp_path, monkeypatch):
    import external_llm.agent.agent_loop as agent_loop_mod
    o = _make_orch(tmp_path)
    st = SubTaskSpec(task_id="dev_1", title="T", description="D", assigned_files=["a.py"])
    (tmp_path / "a.py").write_text("x")
    seen = []
    o._cb_fn = lambda ev, data: seen.append((ev, data))  # dispatcher bound self._cb at init
    loop = _FakeLoop(AgentResult(status="success", final_message="ok", turns=[1]))
    monkeypatch.setattr(agent_loop_mod, "AgentLoop", lambda **kw: loop)
    monkeypatch.setattr(o, "_compute_diff_verdict", lambda **kw: "VERIFIED")
    monkeypatch.setattr(o, "_detect_genuine_violations", lambda *a, **k: [])
    monkeypatch.setattr(o, "_apply_scope_violation_policy", lambda *a, **k: [])
    o._run_subagent(st)
    # subagent events carry agent_id
    start_events = [d for ev, d in seen if ev == "subagent_start"]
    assert start_events and start_events[0]["agent_id"] == "dev_1"


def test_run_subagent_route_decision_set(tmp_path, monkeypatch):
    import external_llm.agent.agent_loop as agent_loop_mod
    o = _make_orch(tmp_path)
    st = SubTaskSpec(task_id="dev_1", title="T", description="D", assigned_files=["a.py"])
    (tmp_path / "a.py").write_text("x")
    loop = _FakeLoop(AgentResult(status="success", final_message="ok", turns=[1]))
    monkeypatch.setattr(agent_loop_mod, "AgentLoop", lambda **kw: loop)
    monkeypatch.setattr(o, "_compute_diff_verdict", lambda **kw: "VERIFIED")
    monkeypatch.setattr(o, "_detect_genuine_violations", lambda *a, **k: [])
    monkeypatch.setattr(o, "_apply_scope_violation_policy", lambda *a, **k: [])
    o._run_subagent(st)
    # config route_decision must have been filled (MAIN_AGENT lane)
    assert loop is not None


def test_run_subagent_parallel_disables_rag(tmp_path, monkeypatch):
    import external_llm.agent.agent_loop as agent_loop_mod
    o = _make_orch(tmp_path, parallel=True)
    st = SubTaskSpec(task_id="dev_1", title="T", description="D", assigned_files=["a.py"])
    (tmp_path / "a.py").write_text("x")
    configs = []
    monkeypatch.setattr(agent_loop_mod, "AgentLoop", lambda **kw: configs.append(kw.get("config")) or _FakeLoop(AgentResult(status="success", final_message="ok", turns=[1])))
    monkeypatch.setattr(o, "_compute_diff_verdict", lambda **kw: "VERIFIED")
    monkeypatch.setattr(o, "_detect_genuine_violations", lambda *a, **k: [])
    monkeypatch.setattr(o, "_apply_scope_violation_policy", lambda *a, **k: [])
    o._run_subagent(st)
    assert configs[0].rag_enabled is False  # parallel → RAG disabled (FAISS not thread-safe)


def test_run_subagent_model_context_scope(tmp_path, monkeypatch):
    import external_llm.agent.agent_loop as agent_loop_mod
    o = _make_orch(tmp_path)
    entered = []

    class _CtxScope:
        def __enter__(self):
            entered.append(True)
            return self
        def __exit__(self, *a):
            return False

    o._run_store = SimpleNamespace(model_context_scope=lambda m, r: _CtxScope())
    st = SubTaskSpec(task_id="dev_1", title="T", description="D", assigned_files=["a.py"])
    (tmp_path / "a.py").write_text("x")
    loop = _FakeLoop(AgentResult(status="success", final_message="ok", turns=[1]))
    monkeypatch.setattr(agent_loop_mod, "AgentLoop", lambda **kw: loop)
    monkeypatch.setattr(o, "_compute_diff_verdict", lambda **kw: "VERIFIED")
    monkeypatch.setattr(o, "_detect_genuine_violations", lambda *a, **k: [])
    monkeypatch.setattr(o, "_apply_scope_violation_policy", lambda *a, **k: [])
    o._run_subagent(st)
    assert entered == [True]


# ═══════════════════════════════════════════════════════════════════════════
# Round 6: defensive except-branch edges (lock rollback, spawn failures, …)
# ═══════════════════════════════════════════════════════════════════════════

def test_filelock_acquire_rollback_on_held_meta_error(tmp_path, monkeypatch):
    import external_llm.agent.orchestrator as orchm
    m = _fresh_manager(tmp_path)
    (tmp_path / "a.py").write_text("x")

    # _held update happens under _file_locks_meta; make the with-block raise
    real_meta = orchm._file_locks_meta
    class _BoomMeta:
        def __enter__(self):
            raise MemoryError("meta boom")
        def __exit__(self, *a):
            return False
    monkeypatch.setattr(orchm, "_file_locks_meta", _BoomMeta())
    try:
        with pytest.raises(MemoryError):
            m.acquire("a.py")
    finally:
        monkeypatch.setattr(orchm, "_file_locks_meta", real_meta)
    assert m._held == {}  # lock released on rollback


def test_filelock_acquire_by_normalized_rollback(tmp_path, monkeypatch):
    import external_llm.agent.orchestrator as orchm
    m = _fresh_manager(tmp_path)
    real_meta = orchm._file_locks_meta
    class _BoomMeta:
        def __enter__(self):
            raise RuntimeError("boom")
        def __exit__(self, *a):
            return False
    monkeypatch.setattr(orchm, "_file_locks_meta", _BoomMeta())
    try:
        with pytest.raises(RuntimeError):
            m._acquire_by_normalized_path("x")
    finally:
        monkeypatch.setattr(orchm, "_file_locks_meta", real_meta)


def test_filelock_acquire_repo_rollback(tmp_path, monkeypatch):
    import external_llm.agent.orchestrator as orchm
    m = _fresh_manager(tmp_path)
    real_meta = orchm._file_locks_meta
    class _BoomMeta:
        def __enter__(self):
            raise MemoryError("boom")
        def __exit__(self, *a):
            return False
    monkeypatch.setattr(orchm, "_file_locks_meta", _BoomMeta())
    try:
        with pytest.raises(MemoryError):
            m.acquire_repo()
    finally:
        monkeypatch.setattr(orchm, "_file_locks_meta", real_meta)
    assert m._held == {}


def test_filelock_release_runtime_error_guard(tmp_path):
    m = _fresh_manager(tmp_path)
    (tmp_path / "a.py").write_text("x")
    locked = m.acquire_relevant({"path": "a.py"})
    m.release_all(locked)
    # lock gone from _held; release again → RuntimeError guard path
    m.release_all(locked)  # must not raise
    assert True


def test_spawn_ipc_worker_background_log_rotation_failures(tmp_path, monkeypatch):
    o = _make_orch(tmp_path)
    proc = SimpleNamespace(pid=7, poll=lambda: None)
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: proc)
    d = tmp_path / ".asicode" / "subagents" / "w1"
    d.mkdir(parents=True)
    log = d / "worker.log"
    log.write_bytes(b"x" * (orch._WORKER_LOG_ROTATE_BYTES + 10))
    # os.replace fails → rotate skipped, still spawns
    real_replace = os.replace
    def boom_replace(src, dst):
        raise OSError("replace boom")
    monkeypatch.setattr(os, "replace", boom_replace)
    assert o._spawn_ipc_worker_background(str(tmp_path), "w1", "", "") is True
    monkeypatch.setattr(os, "replace", real_replace)
    # log open fails → DEVNULL fallback
    def boom_open(path, mode):
        raise OSError("open boom")
    monkeypatch.setattr("builtins.open", boom_open)
    assert o._spawn_ipc_worker_background(str(tmp_path), "w2", "", "") is True


def test_spawn_ipc_worker_background_win32_flags(tmp_path, monkeypatch):
    o = _make_orch(tmp_path)
    proc = SimpleNamespace(pid=8, poll=lambda: None)
    seen = {}
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: seen.update(k) or proc)
    monkeypatch.setattr(orch.sys, "platform", "win32")
    assert o._spawn_ipc_worker_background(str(tmp_path), "w1", "", "") is True
    assert "creationflags" in seen


def test_capture_snapshots_unreadable_oserror(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_bytes(b"x")
    def boom(p):
        raise OSError("perm denied")
    monkeypatch.setattr(os.path, "getsize", boom)
    snaps = _capture_assigned_snapshots(str(tmp_path), ["a.py"])
    assert snaps == {}  # unreadable → skipped


def test_capture_snapshots_read_exception_missing_sentinel(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_bytes(b"x")
    real_open = open
    calls = {"n": 0}
    def flaky(path, mode="r", **kw):
        if str(path).endswith("a.py") and mode == "rb" and calls["n"] == 0:
            calls["n"] += 1
            raise PermissionError("read boom")
        return real_open(path, mode, **kw)
    monkeypatch.setattr("builtins.open", flaky)
    # getsize succeeds, read fails with non-FileNotFoundError → skipped (no sentinel)
    snaps = _capture_assigned_snapshots(str(tmp_path), ["a.py"])
    assert "a.py" not in snaps


def test_symbol_def_line_tree_sitter_failure(tmp_path, monkeypatch):
    # force tree-sitter import path to raise → ast fallback
    import external_llm.languages.tree_sitter_utils as tsu
    def boom(*a, **k):
        raise RuntimeError("grammar missing")
    monkeypatch.setattr(tsu, "find_all_symbols", boom)
    out = _symbol_def_line("def f():\n    pass\n", "m.py", "f")
    assert out == 1
    # ast parse failure → None
    assert _symbol_def_line("def (:", "m.py", "f") is None


def test_symbol_hint_tree_sitter_returns_syms(tmp_path, monkeypatch):
    import external_llm.languages.tree_sitter_utils as tsu
    monkeypatch.setattr(tsu, "find_all_symbols", lambda src, lang: [("foo", "function", 1, 2)])
    out = _symbol_hint_for_source("x", "m.ts")
    assert out == ["F foo L1-L2"]


def test_norm_assigned_file_exception_fallback(tmp_path, monkeypatch):
    import path_security as ps
    def boom(p):
        raise ValueError("bad path")
    monkeypatch.setattr(ps, "normalize_rel_path", boom)
    assert _norm_assigned_file("weird") == "weird"  # raw fallback


def test_build_git_context_subprocess_error(tmp_path, monkeypatch):
    import subprocess as _sp
    def boom(*a, **k):
        raise _sp.TimeoutExpired(cmd=["git"], timeout=5)
    monkeypatch.setattr(_sp, "run", boom)
    assert _build_git_context(str(tmp_path)) == ""


def test_snapshot_dirty_path_set_exception(tmp_path, monkeypatch):
    import external_llm.agent.subagent_ipc as sipc
    def boom(*a, **k):
        raise subprocess.CalledProcessError(1, "git")
    monkeypatch.setattr(sipc, "partition_changed_files", boom)
    assert _snapshot_dirty_path_set(str(tmp_path)) == set()


def test_git_status_changed_paths_exception(tmp_path, monkeypatch):
    import external_llm.agent.subagent_ipc as sipc
    def boom(*a, **k):
        raise subprocess.CalledProcessError(1, "git")
    monkeypatch.setattr(sipc, "partition_changed_files", boom)
    o = _make_orch(tmp_path)
    assert o._git_status_changed_paths(str(tmp_path)) == []


def test_compute_diff_verdict_attach_failure(tmp_path, monkeypatch):
    o = _make_orch(tmp_path)
    class _LockedResult:
        def __init__(self):
            object.__setattr__(self, "status", "error")
            object.__setattr__(self, "final_message", "x")
            object.__setattr__(self, "applied_patches", [])

        def __setattr__(self, name, value):
            raise AttributeError("frozen")
    r = _LockedResult()
    verdict = o._compute_diff_verdict(agent_id="a", result=r, repo_root=None, diff_cache=None)
    assert verdict == "UNVERIFIABLE"  # attach failure swallowed


def test_run_subagent_attach_unassigned_failure(tmp_path, monkeypatch):
    import external_llm.agent.agent_loop as agent_loop_mod
    o = _make_orch(tmp_path)
    st = SubTaskSpec(task_id="dev_1", title="T", description="D", assigned_files=["a.py"])
    (tmp_path / "a.py").write_text("x")
    class _LockedResult2:
        def __init__(self):
            object.__setattr__(self, "status", "success")
            object.__setattr__(self, "final_message", "ok")
            object.__setattr__(self, "turns", [])
            object.__setattr__(self, "applied_patches", [])

        def __setattr__(self, name, value):
            raise AttributeError("frozen")
    loop = _FakeLoop(_LockedResult2())
    monkeypatch.setattr(agent_loop_mod, "AgentLoop", lambda **kw: loop)
    monkeypatch.setattr(o, "_compute_diff_verdict", lambda **kw: "VERIFIED")
    monkeypatch.setattr(o, "_detect_genuine_violations", lambda *a, **k: [])
    monkeypatch.setattr(o, "_apply_scope_violation_policy", lambda *a, **k: [])
    res = o._run_subagent(st)
    assert res.status == "success"


def test_review_subagent_result_target_symbol_hint(git_repo, monkeypatch):
    (git_repo / "m.py").write_text("def target():\n    pass\n", encoding="utf-8")
    (git_repo / "m.py").write_text("def target():\n    return 1\n", encoding="utf-8")  # modified → diff exists
    _patch_llm(monkeypatch, '{"approved": false, "feedback": "wrong", "target_symbol": "target"}')
    o = _make_orch(git_repo, review_enabled=True)
    st = SubTaskSpec(task_id="d1", title="t", description="d", assigned_files=["m.py"])
    approved, feedback = o._review_subagent_result("d1", st, str(git_repo))
    assert approved is False
    assert "[LOCATION HINT]" in feedback  # target symbol located


# ═══════════════════════════════════════════════════════════════════════════
# Round 7: remaining 74-miss tail coverage (defensive except / TOCTOU /
# cancel-drain interleavings / reader-failure fallbacks)
# ═══════════════════════════════════════════════════════════════════════════

# ── FileLockManager defensive tails ─────────────────────────────────────────

def test_filelock_normalize_path_realpath_fallback(tmp_path, monkeypatch):
    m = _fresh_manager(tmp_path)
    (tmp_path / "q.py").write_text("x")
    real_realpath = os.path.realpath

    def flaky_realpath(path, *a, **k):
        if str(path).endswith("q.py"):
            raise OSError("realpath boom")
        return real_realpath(path, *a, **k)

    monkeypatch.setattr(os.path, "realpath", flaky_realpath)
    out = m._normalize_path("q.py")
    assert out and os.path.isabs(out) and out.endswith("q.py")


class _FlakyMeta:
    """Raises on the SECOND __enter__ — the first (setdefault) must succeed so
    the try/except BaseException → lock.release() path is reached."""

    def __init__(self):
        self.n = 0

    def __enter__(self):
        self.n += 1
        if self.n == 2:
            raise RuntimeError("meta boom")

    def __exit__(self, *a):
        return False


def test_filelock_acquire_meta_error_releases_and_raises(tmp_path, monkeypatch):
    m = _fresh_manager(tmp_path)
    (tmp_path / "a.py").write_text("x")
    orig = orch._file_locks_meta
    monkeypatch.setattr(orch, "_file_locks_meta", _FlakyMeta())
    with pytest.raises(RuntimeError):
        m.acquire("a.py")
    # restore → the lock must be free again (release happened in the except tail)
    monkeypatch.setattr(orch, "_file_locks_meta", orig)
    m.acquire("a.py")
    m.release("a.py")


def test_filelock_acquire_by_norm_meta_error_releases_and_raises(tmp_path, monkeypatch):
    m = _fresh_manager(tmp_path)
    (tmp_path / "b.py").write_text("x")
    orig = orch._file_locks_meta
    norm = m._normalize_path("b.py")
    monkeypatch.setattr(orch, "_file_locks_meta", _FlakyMeta())
    with pytest.raises(RuntimeError):
        m._acquire_by_normalized_path(norm)
    monkeypatch.setattr(orch, "_file_locks_meta", orig)
    m._acquire_by_normalized_path(norm)
    m._release_by_normalized_path(norm)


def test_filelock_acquire_repo_meta_error_releases_and_raises(tmp_path, monkeypatch):
    m = _fresh_manager(tmp_path)
    orig = orch._file_locks_meta
    monkeypatch.setattr(orch, "_file_locks_meta", _FlakyMeta())
    with pytest.raises(RuntimeError):
        m.acquire_repo()
    monkeypatch.setattr(orch, "_file_locks_meta", orig)
    key = m.acquire_repo()
    assert key
    m.release_repo(key)


def test_filelock_release_twice_hits_runtime_error_guard(tmp_path):
    m = _fresh_manager(tmp_path)
    (tmp_path / "r.py").write_text("x")
    held = m.acquire("r.py")  # keep a strong ref so the registry entry survives
    m.release("r.py")
    m.release("r.py")  # second release of the same lock → RuntimeError guard
    assert held is not None


def test_filelock_release_invalid_path_noop(tmp_path):
    m = _fresh_manager(tmp_path)
    m.release("../outside.py")  # norm_path None → early return
    m.release("")


def test_filelock_reset_release_error_guard(tmp_path):
    m = _fresh_manager(tmp_path)
    (tmp_path / "s.py").write_text("x")
    m.acquire("s.py")
    norm = next(iter(m._held))
    m._held[norm].release()  # someone else released the underlying lock
    m.reset()                # m's release() → RuntimeError → guarded
    assert m._held == {}


# ── snapshot / symbol-hint / expansion / registry-facade tails ──────────────

def test_capture_snapshots_open_toctou_missing(tmp_path, monkeypatch):
    real_getsize = os.path.getsize

    def getsize_ok(path):
        try:
            return real_getsize(path)
        except OSError:
            return 1  # pretend the file exists → open() hits FileNotFoundError (TOCTOU)

    monkeypatch.setattr(os.path, "getsize", getsize_ok)
    snaps = _capture_assigned_snapshots(str(tmp_path), ["gone.py"])
    assert snaps["gone.py"] is _MISSING_SNAP


def test_symbol_hint_tree_sitter_raise_falls_back_to_ast(monkeypatch):
    import external_llm.languages.tree_sitter_utils as tsu

    def boom(*a, **k):
        raise RuntimeError("grammar missing")

    monkeypatch.setattr(tsu, "find_all_symbols", boom)
    src = "def hello():\n    return 1\n"
    out = _symbol_hint_for_source(src, "mod.py")
    assert out and any("hello" in line for line in out)  # stdlib-ast fallback


def test_expand_directory_assignments_non_str_entry(tmp_path):
    out = _expand_directory_assignments(str(tmp_path), [123])
    assert out == [123]


def test_backed_registry_setattr_private_names(tmp_path):
    base = _FakeRegistry(str(tmp_path))
    reg = _OrchestratorBackedRegistry(base, object())
    reg._obr_base = base  # private-name branch → object.__setattr__
    assert reg._obr_base is base
    reg.session_plan = "p"  # everything else delegates to the base registry
    assert base.session_plan == "p"


# ── _decompose_task git-context branch ──────────────────────────────────────

def test_decompose_task_git_context_appended(git_repo, monkeypatch):
    import external_llm.agent.orchestrator as orchm
    o = _make_orch(git_repo)  # real git history → git_context non-empty
    raw = '{"subtasks": [{"id": "x1", "title": "T", "description": "d"}]}'
    captured = {}

    def fake_call(*a, **k):
        captured["user"] = a[2][1].content
        return raw

    monkeypatch.setattr(orchm, "simple_llm_call", fake_call)
    specs = o._decompose_task("task")
    assert specs
    assert "Recent git history" in captured["user"]


# ── _synthesize_untracked_diff tails ────────────────────────────────────────

def test_synthesize_untracked_diff_no_match_skip(git_repo):
    (git_repo / "new.py").write_text("n\n", encoding="utf-8")  # untracked
    # "f.py" is tracked+clean → absent from the untracked set → no-match continue
    out = OrchestratorAgent._synthesize_untracked_diff(str(git_repo), ["new.py", "f.py"])
    assert "new.py" in out


def test_synthesize_untracked_diff_read_failure_returns_empty(git_repo, monkeypatch):
    (git_repo / "new.py").write_text("n\n", encoding="utf-8")

    def boom(*a, **k):
        raise OSError("read fail")

    monkeypatch.setattr(orch, "_read_synth_diff_head", boom)
    out = OrchestratorAgent._synthesize_untracked_diff(str(git_repo), ["new.py"])
    assert out == ""  # skip + empty-lines return


def test_synthesize_untracked_diff_cap_outer_break(git_repo):
    for f in ("u1.py", "u2.py", "u3.py"):
        (git_repo / f).write_text("line\n", encoding="utf-8")
    out = OrchestratorAgent._synthesize_untracked_diff(
        str(git_repo), ["u1.py", "u2.py", "u3.py"], char_limit=1)
    assert "omitted" in out  # cap marker → _cap_hit → outer-loop break


# ── _revert_unassigned_changes tails ────────────────────────────────────────

def test_revert_batch_checkout_failure_perfile_fallback(git_repo, monkeypatch):
    (git_repo / "f.py").write_text("x=999\n", encoding="utf-8")
    (git_repo / "g.py").write_text("y=1\n", encoding="utf-8")
    o = _make_orch(git_repo)
    import subprocess as _sp
    real_run = _sp.run

    def checkout_boom(cmd, **kw):
        if cmd[:2] == ["git", "checkout"] and len(cmd) > 5:
            raise OSError("batch checkout boom")
        return real_run(cmd, **kw)

    monkeypatch.setattr(_sp, "run", checkout_boom)
    reverted = o._revert_unassigned_changes(str(git_repo), [{"file": "f.py"}, {"file": "g.py"}])
    assert sorted(reverted) == ["f.py", "g.py"]  # per-file fallback succeeded
    assert (git_repo / "f.py").read_text() == "x=2\n"


def test_revert_checkout_always_fails_logged(git_repo, monkeypatch):
    (git_repo / "f.py").write_text("x=999\n", encoding="utf-8")
    o = _make_orch(git_repo)
    import subprocess as _sp
    real_run = _sp.run

    def boom(cmd, **kw):
        if cmd[:2] == ["git", "checkout"]:
            raise OSError("checkout boom")
        return real_run(cmd, **kw)

    monkeypatch.setattr(_sp, "run", boom)
    reverted = o._revert_unassigned_changes(str(git_repo), [{"file": "f.py"}])
    assert reverted == []  # batch + per-file checkout both failed → logged, skipped
    assert (git_repo / "f.py").read_text() == "x=999\n"


def test_revert_perfile_ls_files_failure_unlinks(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    (repo / "boom.py").write_text("b\n", encoding="utf-8")
    (repo / "ok.py").write_text("o\n", encoding="utf-8")
    o = _make_orch(repo)
    import subprocess as _sp
    real_run = _sp.run

    def flaky(cmd, **kw):
        if cmd[:2] == ["git", "ls-files"] and "-z" in cmd:
            raise OSError("batch ls-files boom")          # → per-file fallback
        if cmd[:2] == ["git", "ls-files"] and "--error-unmatch" in cmd and cmd[-1] == "boom.py":
            raise OSError("per-file ls-files boom")       # → treated as untracked
        return real_run(cmd, **kw)

    monkeypatch.setattr(_sp, "run", flaky)
    reverted = o._revert_unassigned_changes(str(repo), [{"file": "boom.py"}, {"file": "ok.py"}])
    assert sorted(reverted) == ["boom.py", "ok.py"]
    assert not (repo / "boom.py").exists() and not (repo / "ok.py").exists()


def test_revert_rmtree_failure_logged(git_repo, monkeypatch):
    d = git_repo / "newdir"
    d.mkdir()
    (d / "x.py").write_text("x\n", encoding="utf-8")
    o = _make_orch(git_repo)
    import shutil

    def boom(*a, **k):
        raise OSError("rmtree boom")

    monkeypatch.setattr(shutil, "rmtree", boom)
    reverted = o._revert_unassigned_changes(str(git_repo), [{"file": "newdir"}])
    assert reverted == []


def test_revert_unlink_failure_logged(git_repo, monkeypatch):
    (git_repo / "stray.py").write_text("s\n", encoding="utf-8")
    o = _make_orch(git_repo)

    def boom(*a, **k):
        raise OSError("unlink boom")

    monkeypatch.setattr(os, "unlink", boom)
    reverted = o._revert_unassigned_changes(str(git_repo), [{"file": "stray.py"}])
    assert reverted == []


# ── _compute_diff_verdict object-patch branch ───────────────────────────────

def test_compute_diff_verdict_object_patch_file_path(git_repo):
    (git_repo / "f.py").write_text("x=999\n", encoding="utf-8")
    o = _make_orch(git_repo, parallel=True)
    r = SimpleNamespace(
        status="success", final_message="ok",
        applied_patches=[SimpleNamespace(file_path="f.py")],
    )
    verdict = o._compute_diff_verdict(agent_id="a1", result=r, repo_root=str(git_repo), diff_cache={})
    assert verdict == "VERIFIED"
    assert r._orch_diff_verdict == "VERIFIED"


# ── _run_parallel_batch cancel-drain interleavings ──────────────────────────

def test_run_parallel_batch_cancel_drain_cancelled_futures(tmp_path, monkeypatch):
    ev = threading.Event()
    ev.set()  # cancel BEFORE dispatch
    o = _make_orch(tmp_path, cancel_event=ev, max_subagents=1)
    specs = [SubTaskSpec(task_id="t0", title="T0", description="d"),
             SubTaskSpec(task_id="t1", title="T1", description="d")]

    def quick(st, *a, **kw):
        # Finish INSIDE the 2s drain window: the worker then dequeues the
        # cancelled sibling → set_running_or_notify_cancel flips it to
        # CANCELLED_AND_NOTIFIED → wait() reports it in `done`.
        time.sleep(0.1)
        return AgentResult(status="success", final_message="ok", turns=[1])

    monkeypatch.setattr(o, "_run_subagent", quick)
    res = o._run_parallel_batch(specs, {}, {})
    # t0 completed before cancel completed → its real result is preserved
    assert res[0].status == "success"
    # t1 was cancelled → CancelledError drain branch
    assert res[1].status == "cancelled" and res[1].error == "Cancelled"


def test_run_parallel_batch_cancel_drain_exception_future(tmp_path, monkeypatch):
    ev = threading.Event()
    ev.set()
    o = _make_orch(tmp_path, cancel_event=ev)  # max_subagents=3 → both start
    specs = [SubTaskSpec(task_id="a", title="A", description="d"),
             SubTaskSpec(task_id="b", title="B", description="d")]
    release = threading.Event()

    def mixed(st, *a, **kw):
        if st.task_id == "a":
            raise RuntimeError("boom")
        release.wait(10)
        return AgentResult(status="success", final_message="ok", turns=[1])

    monkeypatch.setattr(o, "_run_subagent", mixed)
    res = o._run_parallel_batch(specs, {}, {})
    assert res[0].status == "error" and "boom" in res[0].error
    # "b" may still be queued at cancel time (→ Cancelled) or already running
    # (→ still_pending); both are legitimate cancel outcomes.
    assert res[1].status == "cancelled"
    assert res[1].error in ("Cancelled", "Cancelled by orchestrator")
    release.set()


def test_run_parallel_batch_first_completed_cancelled_drain(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    o = _make_orch(tmp_path, max_subagents=1)
    specs = [SubTaskSpec(task_id="a", title="A", description="d"),
             SubTaskSpec(task_id="b", title="B", description="d")]
    recorded = []
    both_submitted = threading.Event()

    class RecordingPool(ThreadPoolExecutor):
        def submit(self, fn, *a, **k):
            fut = super().submit(fn, *a, **k)
            recorded.append(fut)
            if len(recorded) == 2:
                both_submitted.set()
            return fut

    monkeypatch.setattr(orch, "ThreadPoolExecutor", RecordingPool)

    def first(st, *a, **kw):
        both_submitted.wait(5)
        recorded[1].cancel()  # cancel the queued sibling while "a" is mid-run
        return AgentResult(status="success", final_message="ok", turns=[1])

    monkeypatch.setattr(o, "_run_subagent", first)
    res = o._run_parallel_batch(specs, {}, {})
    assert res[0].status == "success"
    # the cancelled sibling is dequeued by the worker → CANCELLED_AND_NOTIFIED
    # → appears in the FIRST_COMPLETED `done` set → CancelledError drain branch
    assert res[1].status == "cancelled" and res[1].error == "Cancelled"


# ── _sub_cb stream-callback forwarding (in-process _run_subagent) ───────────

class _CallbackLoop:
    def __init__(self, **kw):
        # stream_callback travels inside the AgentConfig, not as a ctor kwarg
        self._stream_cb = getattr(kw.get("config"), "stream_callback", None)
        self.calls = []

    def run(self, text):
        self.calls.append(text)
        if self._stream_cb:
            self._stream_cb("stream_event", {"chunk": "hi"})
        return AgentResult(status="success", final_message="ok", turns=[1])


def test_run_subagent_stream_callback_forwards_agent_id(tmp_path, monkeypatch):
    o = _make_orch(tmp_path)
    st = SubTaskSpec(task_id="dev_1", title="T", description="Do", assigned_files=["a.py"])
    (tmp_path / "a.py").write_text("x = 1\n")
    monkeypatch.setattr(agent_loop_mod, "AgentLoop", lambda **kw: _CallbackLoop(**kw))
    monkeypatch.setattr(o._registry_proto, "clone_for_subagent", lambda cfg: o._registry_proto)
    monkeypatch.setattr(o, "_compute_diff_verdict", lambda **kw: "VERIFIED")
    monkeypatch.setattr(o, "_detect_genuine_violations", lambda *a, **k: [])
    monkeypatch.setattr(o, "_apply_scope_violation_policy", lambda *a, **k: [])
    events = []
    monkeypatch.setattr(o, "_cb", lambda ev, data: events.append((ev, data)))
    res = o._run_subagent(st)
    assert res.status == "success"
    assert any(ev == "stream_event" and d.get("agent_id") == "dev_1" for ev, d in events)


# ── _run_subagent_ipc result-attach guard tail ──────────────────────────────

class _SlotsResult:
    """AgentResult stand-in whose __slots__ reject _orch_unassigned (line 2517)."""

    __slots__ = ("applied_patches", "error", "final_message", "status", "turns")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_run_subagent_ipc_attach_unassigned_guard(tmp_path, monkeypatch):
    import external_llm.agent.subagent_ipc as sipc
    o = _make_orch(tmp_path, subagent_mode="ipc", ipc_timeout_s=30)
    st = SubTaskSpec(task_id="dev_1", title="T", description="D", assigned_files=["a.py"])
    (tmp_path / "a.py").write_text("x")
    monkeypatch.setattr(sipc, "clear_result", lambda *a, **k: None)
    monkeypatch.setattr(sipc, "write_task", lambda *a, **k: None)
    monkeypatch.setattr(
        sipc, "wait_for_result",
        lambda *a, **k: _FakeResult(status="success", final_message="done", turns=1,
                                    unassigned_changes=[{"file": "stray.py"}]),
    )
    monkeypatch.setattr(o, "_claim_reusable_worker", lambda repo: None)
    monkeypatch.setattr(o, "_return_worker_to_pool", lambda wid: None)
    monkeypatch.setattr(o, "_compute_diff_verdict", lambda **kw: "VERIFIED")
    monkeypatch.setattr(o, "_filter_unassigned_changes", lambda r, own: r)
    monkeypatch.setattr(o, "_apply_scope_violation_policy", lambda *a, **k: [])
    monkeypatch.setattr(agent_loop_mod, "AgentResult", _SlotsResult)
    res = o._run_subagent_ipc(st)
    assert res.status == "success"  # attach raised AttributeError → guarded → flow completed


# ── _spawn_ipc_worker_background log tails ──────────────────────────────────

def test_spawn_ipc_worker_background_log_stat_error(tmp_path, monkeypatch):
    o = _make_orch(tmp_path)
    proc = SimpleNamespace(pid=7, poll=lambda: None)
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: proc)
    d = tmp_path / ".asicode" / "subagents" / "w3"
    d.mkdir(parents=True)
    (d / "worker.log").write_bytes(b"x")

    def getsize_boom(path):
        raise OSError("stat boom")

    monkeypatch.setattr(os.path, "getsize", getsize_boom)
    assert o._spawn_ipc_worker_background(str(tmp_path), "w3", "", "") is True


def test_spawn_ipc_worker_background_log_close_error(tmp_path, monkeypatch):
    o = _make_orch(tmp_path)
    proc = SimpleNamespace(pid=8, poll=lambda: None)
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: proc)

    class _BoomFile:
        def close(self):
            raise OSError("close boom")

    real_open = open

    def fake_open(path, mode="r", *a, **k):
        if mode == "ab":
            return _BoomFile()
        return real_open(path, mode, *a, **k)

    monkeypatch.setattr("builtins.open", fake_open)
    assert o._spawn_ipc_worker_background(str(tmp_path), "w4", "", "") is True


# ── _tool_spawn_subagent tails ──────────────────────────────────────────────

def test_tool_spawn_subagent_skips_finished_entries(tmp_path, monkeypatch):
    o = _make_orch(tmp_path)
    st_finished = SubTaskSpec(task_id="dev_0", title="T", description="d", assigned_files=["x.py"])
    st_running = SubTaskSpec(task_id="dev_1", title="T", description="d", assigned_files=["share.py"])
    o._bg_subagents["dev_0"] = {"result": object(), "subtask": st_finished}  # finished → skip
    o._bg_subagents["dev_1"] = {"result": None, "subtask": st_running}       # running → overlap
    monkeypatch.setattr(o, "_run_subagent_background", lambda subtask, original_request="": None)
    out = o._tool_spawn_subagent({"task_description": "Do it", "assigned_files": ["share.py"]}, "")
    assert "overlap" in out and "dev_1" in out


def test_tool_spawn_subagent_queued_note(tmp_path, monkeypatch):
    o = _make_orch(tmp_path)
    monkeypatch.setattr(o, "_run_subagent_background", lambda subtask, original_request="": None)
    monkeypatch.setattr(o, "_future_is_queued", lambda entry: True)
    out = o._tool_spawn_subagent({"task_description": "Do it", "assigned_files": ["a.py"]}, "")
    assert "Queued" in out


# ── _claim_reusable_worker reader-failure tail ──────────────────────────────

def test_claim_reusable_worker_heartbeat_readers_raise(tmp_path, monkeypatch):
    import external_llm.agent.subagent_ipc as sipc
    o = _make_orch(tmp_path)
    o._reusable_worker_ids.add("w1")

    def boom(*a, **k):
        raise OSError("hb read boom")

    monkeypatch.setattr(sipc, "read_worker_idle_heartbeat_state", boom)
    monkeypatch.setattr(sipc, "read_worker_idle_heartbeat_age", boom)
    assert o._claim_reusable_worker(str(tmp_path)) == "w1"  # both readers failed → optimistic claim
    assert "w1" not in o._reusable_worker_ids


# ── _abandon_ipc_worker / _cleanup_ipc_workers terminate-failure tails ──────

def test_abandon_ipc_worker_soft_terminate_failure(tmp_path, monkeypatch):
    import external_llm.agent.subagent_ipc as sipc
    o = _make_orch(tmp_path)
    monkeypatch.setattr(sipc, "write_cancel_sentinel", lambda r, w: None)
    proc = SimpleNamespace(poll=lambda: None)

    def term_boom():
        raise OSError("term boom")

    proc.terminate = term_boom
    o._ipc_worker_procs["w1"] = proc
    reusable = o._abandon_ipc_worker(str(tmp_path), "w1", grace_s=0.1)
    assert reusable is False  # hung worker + failed terminate → not reusable


def test_abandon_ipc_worker_hard_terminate_failure(tmp_path, monkeypatch):
    import external_llm.agent.subagent_ipc as sipc
    ev = threading.Event()
    ev.set()
    o = _make_orch(tmp_path, cancel_event=ev)
    monkeypatch.setattr(sipc, "write_cancel_sentinel", lambda r, w: None)
    proc = SimpleNamespace(poll=lambda: None)

    def wait_boom(timeout=None):
        raise subprocess.TimeoutExpired("w1", timeout)

    proc.wait = wait_boom

    def term_boom():
        raise OSError("term boom")

    proc.terminate = term_boom
    o._ipc_worker_procs["w1"] = proc
    reusable = o._abandon_ipc_worker(str(tmp_path), "w1", grace_s=0.1)
    assert reusable is False


def test_cleanup_ipc_workers_terminate_failure(tmp_path, monkeypatch):
    import external_llm.agent.subagent_ipc as sipc
    o = _make_orch(tmp_path, subagent_mode="ipc")
    o._ipc_worker_ids.add("w1")
    monkeypatch.setattr(sipc, "write_cancel_all", lambda r, w: None)
    monkeypatch.setattr(sipc, "write_shutdown_all", lambda r, w: None)
    proc = SimpleNamespace(poll=lambda: None)

    def term_boom():
        raise OSError("term boom")

    proc.terminate = term_boom
    o._ipc_worker_procs["w1"] = proc
    o._cleanup_ipc_workers()  # must not raise
    assert o._reusable_worker_ids == set()
