"""Hardening for ``_tool_create_file`` — it was the lone write handler without
repo confinement, a syntax gate, or an atomic write.

It is NOT dead code: apply_patch routes failed new-file / multi-symbol patches
to it (``_try_apply_patch_create_file_fallback`` /
``_try_apply_patch_multi_symbol_fallback``), so those fallback paths inherited
every gap. These tests pin the three safety nets every other write handler
(edit_text / modify_symbol / edit_ast / anchor_edit) already had.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from external_llm.agent.tool_registry import AgentConfig, ToolRegistry


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    (tmp_path / "existing.py").write_text("x = 1\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return tmp_path


def _registry(repo: Path) -> ToolRegistry:
    return ToolRegistry(str(repo), AgentConfig(rag_enabled=False))


class TestRepoConfinement:
    def test_absolute_path_outside_repo_is_refused(self, repo: Path):
        r = _registry(repo)._tool_create_file({"path": "/etc/evil_create_file.py", "content": "x=1\n"})
        assert r.ok is False
        assert "outside repo" in (r.error or "")

    def test_dotdot_traversal_is_refused(self, repo: Path):
        r = _registry(repo)._tool_create_file({"path": "../escape.py", "content": "x=1\n"})
        assert r.ok is False
        assert "outside repo" in (r.error or "")
        assert not (repo.parent / "escape.py").exists()

    def test_in_repo_relative_path_is_allowed(self, repo: Path):
        r = _registry(repo)._tool_create_file({"path": "new.py", "content": "x = 1\n"})
        assert r.ok is True
        assert (repo / "new.py").exists()


class TestSyntaxGate:
    def test_invalid_python_is_refused_and_not_written(self, repo: Path):
        broken = "def f(\n"  # unterminated, genuine SyntaxError
        r = _registry(repo)._tool_create_file({"path": "broken.py", "content": broken})
        assert r.ok is False
        assert "syntax error" in (r.error or "").lower()
        assert (r.metadata or {}).get("failure_class") == "syntax_invalid_after_edit"
        # The file must NOT exist — the gate runs before disk.
        assert not (repo / "broken.py").exists()

    def test_valid_python_is_written(self, repo: Path):
        r = _registry(repo)._tool_create_file({"path": "ok.py", "content": "def f():\n    return 1\n"})
        assert r.ok is True
        assert (repo / "ok.py").read_text() == "def f():\n    return 1\n"

    def test_unknown_language_skips_the_gate(self, repo: Path):
        """A .txt file has no validator — the gate must not block it."""
        r = _registry(repo)._tool_create_file({"path": "notes.txt", "content": "any (thing) at all\n"})
        assert r.ok is True
        assert (repo / "notes.txt").read_text() == "any (thing) at all\n"

    def test_overwrite_with_broken_python_is_refused(self, repo: Path):
        """overwrite=True must still pass the syntax gate."""
        r = _registry(repo)._tool_create_file({"path": "existing.py", "content": "def (:\n", "overwrite": True})
        assert r.ok is False
        assert "syntax error" in (r.error or "").lower()
        # Existing file untouched.
        assert (repo / "existing.py").read_text() == "x = 1\n"


class TestCreateOverwriteSemantics:
    def test_existing_without_overwrite_is_refused(self, repo: Path):
        r = _registry(repo)._tool_create_file({"path": "existing.py", "content": "y = 2\n"})
        assert r.ok is False
        assert "already exists" in (r.error or "")
        assert (repo / "existing.py").read_text() == "x = 1\n"

    def test_overwrite_replaces_content(self, repo: Path):
        r = _registry(repo)._tool_create_file({"path": "existing.py", "content": "y = 2\n", "overwrite": True})
        assert r.ok is True
        assert (repo / "existing.py").read_text() == "y = 2\n"
        assert (r.metadata or {}).get("file_path") == "existing.py"

    def test_parent_dirs_are_created(self, repo: Path):
        r = _registry(repo)._tool_create_file({"path": "a/b/c/deep.py", "content": "z = 3\n"})
        assert r.ok is True
        assert (repo / "a" / "b" / "c" / "deep.py").exists()


class TestAtomicWrite:
    def test_new_file_written_atomically(self, repo: Path):
        """Sanity: content lands exactly, no partial. The atomic helper leaves
        no sibling temp behind on success."""
        r = _registry(repo)._tool_create_file({"path": "atomic.py", "content": "v = 9\n"})
        assert r.ok is True
        assert (repo / "atomic.py").read_text() == "v = 9\n"
        leftover = [p for p in repo.iterdir() if p.name.startswith(".atomic_")]
        assert leftover == []


class TestApplyPatchFallbackReachesGate:
    """The fix's whole point: apply_patch's create_file fallback is the REACHABLE
    path that used to skip repo confinement + the syntax gate. A broken new-file
    patch routed through it must now be refused, not written."""

    def test_broken_new_file_patch_is_refused_via_fallback(self, repo: Path):
        import time as _time

        reg = _registry(repo)
        # A clean creation patch (--- /dev/null + new file mode + pure '+' body)
        # whose body is invalid Python. _extract_new_file_target routes this to
        # _tool_create_file, whose gate must now refuse it.
        patch = (
            "diff --git a/via_fallback.py b/via_fallback.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/via_fallback.py\n"
            "@@ -0,0 +1,2 @@\n"
            "+def broken(\n"
            "+\n"
        )
        r = reg._try_apply_patch_create_file_fallback(
            patch, None, "apply_patch failed: original error", _time.monotonic()
        )
        assert r is not None
        assert r.ok is False
        assert "syntax error" in (r.error or "").lower()
        assert not (repo / "via_fallback.py").exists()

    def test_valid_new_file_patch_succeeds_via_fallback(self, repo: Path):
        import time as _time

        reg = _registry(repo)
        patch = (
            "diff --git a/ok_fallback.py b/ok_fallback.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/ok_fallback.py\n"
            "@@ -0,0 +1,1 @@\n"
            "+x = 1\n"
        )
        r = reg._try_apply_patch_create_file_fallback(
            patch, None, "apply_patch failed: original error", _time.monotonic()
        )
        assert r is not None
        assert r.ok is True
        assert (repo / "ok_fallback.py").read_text().rstrip() == "x = 1"
