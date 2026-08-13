"""B2: edit_file / anchor_edit main write paths must go through the atomic
funnel (``atomic_write_text``) — same as apply_patch/edit_text — so a
just-written file becomes visible to cached consumers (repo file index, walk
caches) within the same turn WITHOUT the dispatch-level
``_invalidate_cache_after_write`` having to run.

The handlers are invoked directly (bypassing dispatch), so any invalidation
observed here must have come from
``atomic_write_text -> invalidate_for_written_path`` — a plain
``Path.write_text`` would leave the pre-write listing cached until TTL.
"""
from __future__ import annotations

import pytest

from external_llm.agent.tool_handlers.write_tools import WriteToolsMixin
from external_llm.agent.tool_registry import ToolResult


class _Harness(WriteToolsMixin):
    """Minimal concrete host for the edit/patch mixins.

    Mirrors the per-mixin harnesses in test_write_tools_edit_mixin.py /
    test_write_tools_patch_mixin.py. Deliberately has NO
    ``_invalidate_cache_after_write`` hook — dispatch is bypassed in these
    tests, so the observed invalidation must come from the write itself.
    """

    def __init__(self, repo_root):
        self.repo_root = str(repo_root)
        self._repo_root_override = None
        self._applied_patches = []
        self._text_edited_files = set()

    @property
    def _effective_repo_root(self):
        return self.repo_root

    def _make_result(self, **kwargs):
        kwargs.setdefault("content", "")
        return ToolResult(**kwargs)

    def _run_syntax_check_for_file(self, path):
        return {"ok": True, "skipped": True, "reason": "test"}

    def _secure_path(self, path, *, confine=False):
        from pathlib import Path as _Path
        repo = _Path(self.repo_root).resolve()
        p = _Path(path)
        resolved = p.resolve() if p.is_absolute() else (repo / path).resolve()
        try:
            resolved.relative_to(repo)
        except ValueError:
            return None
        return resolved

    def _should_soft_fail_verify(self, verify_detail, snapshots):
        return False


@pytest.fixture
def harness(tmp_path):
    return _Harness(tmp_path)


@pytest.fixture
def cached_index(tmp_path, monkeypatch):
    """Populate the shared repo-file index for tmp_path and return its key."""
    import external_llm.common.repo_files as common_rf

    key = common_rf.canonical_repo_key(str(tmp_path))
    common_rf._FILE_INDEX_CACHE.pop(key, None)

    def fake_listing(root):
        return ["app.py"]

    # Patch at the SSOT (common.repo_files) — re-export bindings elsewhere
    # are not seen by the real call inside cached_repo_file_list.
    monkeypatch.setattr(common_rf, "git_list_repo_files", fake_listing)
    assert common_rf.cached_repo_file_list(str(tmp_path)) == ["app.py"]
    assert key in common_rf._FILE_INDEX_CACHE
    return key


def _assert_invalidated(key):
    import external_llm.common.repo_files as common_rf

    assert key not in common_rf._FILE_INDEX_CACHE, (
        "the write must invalidate the repo file index through the atomic "
        "funnel — no dispatch-level invalidation runs in these tests"
    )


# ── edit_file ───────────────────────────────────────────────────────────────

def test_edit_file_success_invalidates_repo_file_index(harness, tmp_path, cached_index):
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    r = harness._tool_edit_file({"path": "app.py", "operations": [
        {"type": "replace", "anchor": "x = 1", "content": "x = 2"}]})
    assert r.ok, r.error
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "x = 2\n"
    _assert_invalidated(cached_index)


def test_edit_file_rollback_write_is_atomic_and_invalidates(harness, tmp_path, cached_index, monkeypatch):
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")

    def bad_syntax(path):
        return {"ok": False, "skipped": False,
                "errors": [{"line": 1, "col": 1, "message": "boom"}]}

    monkeypatch.setattr(harness, "_run_syntax_check_for_file", bad_syntax)
    r = harness._tool_edit_file({"path": "app.py", "operations": [
        {"type": "replace", "anchor": "x = 1", "content": "x = 2"}]})
    assert not r.ok and "Syntax error" in r.error
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "x = 1\n", (
        "rollback must restore the original content"
    )
    # The rollback write itself must also go through the atomic funnel (a
    # crash between restore-write and return must not leave a truncated file).
    _assert_invalidated(cached_index)


# ── anchor_edit ─────────────────────────────────────────────────────────────

def test_anchor_edit_delete_write_invalidates_repo_file_index(harness, tmp_path, cached_index):
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    r = harness._tool_anchor_edit(
        {"file_path": "app.py", "edit_mode": "delete", "anchor_pattern": "x = 1"})
    assert r.ok, r.error
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == ""
    _assert_invalidated(cached_index)


def test_anchor_edit_insert_write_invalidates_repo_file_index(harness, tmp_path, cached_index):
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    r = harness._tool_anchor_edit(
        {"file_path": "app.py", "edit_mode": "insert_after",
         "anchor_pattern": "x = 1", "code_snippet": "y = 2"})
    assert r.ok, r.error
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "x = 1\ny = 2\n"
    _assert_invalidated(cached_index)
