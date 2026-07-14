"""Tests for apply_patch's session-edit guard (Opt D).

apply_patch / diff_apply reconstruct hunk context from HEAD, NOT the working
tree. On a freshly-edited target PatchEngine uses skip_3way=True, whose
_rollback() (diff_apply) reverts the working tree to HEAD — silently deleting
any working-tree edit. To prevent the agent from clobbering its OWN edits, the
4 text-editing tools (edit_text / modify_symbol / edit_ast / anchor_edit)
record each file they write in ``_text_edited_files``; apply_patch then
HARD-REJECTS (ok=false) any touched file found in that set, naming it in
metadata["refused_dirty_files"] + metadata["reason"] ==
"session_text_edit_overwrite_risk".

Unlike the prior git-dirty check (Opt A), this does NOT block user
pre-existing uncommitted edits or unrelated dirty files in a multi-file
patch — see test_non_session_dirty_file_is_allowed for that friction win.

These tests exercise both the real diff_apply main path and _tool_apply_patch
(the LLM-facing entry) against a real git repo.
"""
import subprocess

from external_llm.agent.tool_handlers.write_tools import WriteToolsMixin


class _Stub(WriteToolsMixin):
    """Minimal host providing only what _apply_patch_text / _tool_apply_patch need."""

    def __init__(self, repo_root):
        self.repo_root = str(repo_root)
        self._effective_repo_root = str(repo_root)
        self._applied_patches = []
        self._text_edited_files = set()

    def _make_result(self, **kw):
        from external_llm.agent.tool_registry import ToolResult
        kw.setdefault("content", "")
        return ToolResult(**kw)


_PATCH = """diff --git a/sample.py b/sample.py
--- a/sample.py
+++ b/sample.py
@@ -1,4 +1,4 @@
 def foo():
-    return 1
+    return 100

 def bar():
"""

_ORIG = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
# Session edit (matches the agent editing foo) — patch context for foo is now stale.
_DIRTY_FOO = "def foo():\n    return 999  # SESSION EDIT\n\ndef bar():\n    return 2\n"
# Unrelated user WIP (edits bar, leaves foo intact) — foo patch still applies cleanly.
_DIRTY_BAR = "def foo():\n    return 1\n\ndef bar():\n    return 77  # USER WIP\n"


def _init_repo(tmp_path, dirty_content=None):
    """Create a committed sample.py; optionally dirty the working tree with the
    given content (simulating either a session edit or unrelated user WIP)."""
    td = tmp_path
    subprocess.run(["git", "init", "-q"], cwd=td, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=td, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=td, check=True)
    (td / "sample.py").write_text(_ORIG, encoding="utf-8")
    subprocess.run(["git", "add", "sample.py"], cwd=td, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=td, check=True)
    if dirty_content is not None:
        (td / "sample.py").write_text(dirty_content, encoding="utf-8")
    return _Stub(td)


class TestApplyPatchSessionEditGuard:
    def test_session_edited_file_is_rejected_with_reason(self, tmp_path):
        """A touched file written this session by a text-editing tool must be
        HARD-REJECTED (ok=False) with the offending file named in error + metadata."""
        stub = _init_repo(tmp_path, dirty_content=_DIRTY_FOO)
        stub._record_text_edit("sample.py")  # simulate edit_text having written it
        res = stub._apply_patch_text(_PATCH, path_hint="sample.py")
        assert not res.ok
        assert "sample.py" in (res.error or "")
        meta = res.metadata or {}
        assert meta.get("reason") == "session_text_edit_overwrite_risk"
        assert meta.get("refused_dirty_files") == ["sample.py"]

    def test_session_edited_file_not_mutated_after_reject(self, tmp_path):
        """The reject must happen BEFORE any apply — the session-edited working-tree
        content is byte-for-byte preserved (never silently overwritten/reverted)."""
        stub = _init_repo(tmp_path, dirty_content=_DIRTY_FOO)
        stub._record_text_edit("sample.py")
        res = stub._apply_patch_text(_PATCH, path_hint="sample.py")
        assert not res.ok
        assert (tmp_path / "sample.py").read_text(encoding="utf-8") == _DIRTY_FOO

    def test_non_session_dirty_file_is_allowed(self, tmp_path):
        """Opt D friction win: a dirty file NOT written by a session text tool is
        allowed through the guard (here, user WIP on `bar` that doesn't conflict
        with the `foo` patch). The patch applies cleanly on top of the WIP —
        proving we no longer over-block unrelated dirty files (the prior Opt A
        git-dirty check would have refused this)."""
        stub = _init_repo(tmp_path, dirty_content=_DIRTY_BAR)
        # NOTE: no _record_text_edit — this is a user/external edit, not session.
        res = stub._apply_patch_text(_PATCH, path_hint="sample.py")
        assert res.ok, f"expected guard to allow non-session dirty file; got: {res.error}"
        assert (res.metadata or {}).get("refused_dirty_files") is None
        # foo patched (1->100), user's bar WIP (77) preserved.
        assert (tmp_path / "sample.py").read_text(encoding="utf-8") == (
            "def foo():\n    return 100\n\ndef bar():\n    return 77  # USER WIP\n"
        )

    def test_clean_file_applies_normally(self, tmp_path):
        """A committed (clean, never session-edited) file applies normally."""
        stub = _init_repo(tmp_path)
        res = stub._apply_patch_text(_PATCH, path_hint="sample.py")
        assert res.ok
        assert (res.metadata or {}).get("refused_dirty_files") is None
        assert "session" not in (res.content or "")

    def test_main_entry_tool_apply_patch_rejects_session_edited(self, tmp_path):
        """The MAIN entry point (_tool_apply_patch -> PatchEngine -> diff_apply) is
        what the LLM actually calls. It must reject a session-edited file BEFORE
        PatchEngine's diff_apply reverts the working tree to HEAD (skip_3way=True
        path) and silently DELETES the session edit while returning ok=False."""
        stub = _init_repo(tmp_path, dirty_content=_DIRTY_FOO)
        stub._record_text_edit("sample.py")
        res = stub._tool_apply_patch({"patch": _PATCH, "path": "sample.py"})
        assert not res.ok
        meta = res.metadata or {}
        assert meta.get("reason") == "session_text_edit_overwrite_risk"
        assert meta.get("refused_dirty_files") == ["sample.py"]
        # Critical: the session edit survives byte-for-byte — PatchEngine was
        # never reached, so no HEAD-revert occurred.
        assert (tmp_path / "sample.py").read_text(encoding="utf-8") == _DIRTY_FOO

    def test_record_text_edit_normalizes_path(self, tmp_path):
        """Absolute and relative paths are both normalized to repo-root-relative,
        so the guard (which compares against patch-extracted relative paths) matches."""
        stub = _init_repo(tmp_path)
        stub._record_text_edit(str(tmp_path / "sample.py"))  # absolute
        stub._record_text_edit("sample.py")                   # relative
        assert stub._text_edited_files == {"sample.py"}
