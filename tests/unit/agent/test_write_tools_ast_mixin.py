"""Unit tests for WriteToolsAstMixin._tool_edit_ast (the edit_ast handler).

Covers the AST edit end-to-end flow that existing write-tools tests do not
reach: validation errors, path/file/read failures, encoding fallback,
non-Python rejection, source syntax errors, field-alias normalization,
executor failure hinting, idempotent results, the post-apply compile gate,
dry-run preview, write errors, and text-edit recording.
"""
from __future__ import annotations

import pytest

from external_llm.agent.tool_handlers.write_tools import WriteToolsMixin
from external_llm.agent.tool_registry import ToolResult


class _Harness(WriteToolsMixin):
    """Minimal concrete host for the AST mixin (pattern shared with edit/patch tests)."""

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


def _write(tmp_path, name: str, content: str):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


# ── Validation errors ───────────────────────────────────────────────────────

class TestValidationErrors:
    def test_missing_file_path_no_raw(self, harness):
        r = harness._tool_edit_ast({"ops": [{"type": "replace_expr"}]})
        assert r.ok is False
        assert "'file_path' is required" in r.error
        assert "(raw args:" not in r.error

    def test_missing_file_path_short_raw_no_hint(self, harness):
        r = harness._tool_edit_ast({"ops": [], "__raw_arguments": "tiny"})
        assert r.ok is False
        assert "(raw args:" not in r.error

    def test_missing_file_path_long_raw_hint(self, harness):
        r = harness._tool_edit_ast({
            "ops": [],
            "__raw_arguments": "x" * 40,
        })
        assert r.ok is False
        assert "(raw args:" in r.error
        assert "x" * 40 in r.error

    def test_ops_not_a_list(self, harness):
        # [] is falsy → caught by the earlier "'ops' is required" guard; a
        # truthy non-list reaches the "non-empty list" guard instead.
        r = harness._tool_edit_ast({"file_path": "t.py", "ops": "replace"})
        assert r.ok is False
        assert "non-empty list" in r.error

    def test_empty_ops_list(self, harness):
        r = harness._tool_edit_ast({"file_path": "t.py", "ops": []})
        assert r.ok is False
        assert "'ops' is required" in r.error

    def test_path_blocked(self, harness):
        r = harness._tool_edit_ast({"file_path": "/etc/passwd", "ops": [{"type": "replace_expr"}]})
        assert r.ok is False
        assert "Path blocked" in r.error

    def test_file_not_found(self, harness, tmp_path):
        r = harness._tool_edit_ast({"file_path": "missing_file_xyz.py", "ops": [{"type": "replace_expr"}]})
        assert r.ok is False
        assert "File not found" in r.error


# ── Read failures ───────────────────────────────────────────────────────────

class TestReadFailures:
    def test_oserror_read(self, harness, tmp_path, monkeypatch):
        _write(tmp_path, "t.py", "x = 1\n")
        from external_llm.agent.tool_handlers import write_tools_ast_mixin as mod

        def _boom(_p):
            raise OSError("boom")
        monkeypatch.setattr(mod, "read_text_with_encoding_fallback", _boom)
        r = harness._tool_edit_ast({"file_path": "t.py", "ops": [{"type": "replace_expr"}]})
        assert r.ok is False
        assert "OSError" in r.error

    def test_unsupported_encoding(self, harness, tmp_path, monkeypatch):
        _write(tmp_path, "t.py", "x = 1\n")
        from external_llm.agent.tool_handlers import write_tools_ast_mixin as mod

        monkeypatch.setattr(mod, "read_text_with_encoding_fallback", lambda _p: (None, "utf-8"))
        r = harness._tool_edit_ast({"file_path": "t.py", "ops": [{"type": "replace_expr"}]})
        assert r.ok is False
        assert "unsupported encoding" in r.error

    def test_non_python_rejected(self, harness, tmp_path):
        _write(tmp_path, "t.ts", "const x = 1;\n")
        r = harness._tool_edit_ast({"file_path": "t.ts", "ops": [{"type": "replace_expr"}]})
        assert r.ok is False
        assert "only supported for Python files" in r.error

    def test_source_syntax_error(self, harness, tmp_path):
        _write(tmp_path, "broken.py", "def broken(:\n")
        r = harness._tool_edit_ast({"file_path": "broken.py", "ops": [{"type": "replace_expr"}]})
        assert r.ok is False
        assert "Syntax error" in r.error


# ── Happy path & op normalization ───────────────────────────────────────────

class TestHappyPath:
    def test_simple_replace_applies_and_records(self, harness, tmp_path):
        _write(tmp_path, "t.py", "x = 1\n")
        r = harness._tool_edit_ast({
            "file_path": "t.py",
            "ops": [{"type": "replace_expr", "old": "x = 1", "new": "x = 2"}],
        })
        assert r.ok is True
        assert "AST edit applied" in r.content
        assert "x = 2" in (tmp_path / "t.py").read_text()
        assert "t.py" in harness._text_edited_files
        assert r.metadata["changed"] is True
        assert r.metadata["ops_applied"] == 1
        assert r.metadata["ops_failed"] == []

    def test_field_alias_normalization(self, harness, tmp_path):
        _write(tmp_path, "t.py", "x = 1\n")
        r = harness._tool_edit_ast({
            "file_path": "t.py",
            "ops": [{"type": "replace_expr", "target": "x = 1", "new_expr": "x = 42"}],
        })
        assert r.ok is True
        assert "x = 42" in (tmp_path / "t.py").read_text()

    def test_add_import_alias(self, harness, tmp_path):
        _write(tmp_path, "t.py", "x = 1\n")
        r = harness._tool_edit_ast({
            "file_path": "t.py",
            "ops": [{"type": "add_import", "import_name": "import os"}],
        })
        assert r.ok is True
        assert "import os" in (tmp_path / "t.py").read_text()

    def test_op_key_fallback_to_op(self, harness, tmp_path):
        _write(tmp_path, "t.py", "x = 1\n")
        r = harness._tool_edit_ast({
            "file_path": "t.py",
            "ops": [{"op": "replace_expr", "old": "x = 1", "new": "x = 3"}],
        })
        assert r.ok is True
        assert "x = 3" in (tmp_path / "t.py").read_text()

    def test_action_key_fallback(self, harness, tmp_path):
        _write(tmp_path, "t.py", "x = 1\n")
        r = harness._tool_edit_ast({
            "file_path": "t.py",
            "ops": [{"action": "replace_expr", "old": "x = 1", "new": "x = 4"}],
        })
        assert r.ok is True
        assert "x = 4" in (tmp_path / "t.py").read_text()

    def test_non_dict_op_skipped(self, harness, tmp_path):
        _write(tmp_path, "t.py", "x = 1\n")
        r = harness._tool_edit_ast({
            "file_path": "t.py",
            "ops": ["not-a-dict", {"type": "replace_expr", "old": "x = 1", "new": "x = 5"}],
        })
        assert r.ok is True
        assert "x = 5" in (tmp_path / "t.py").read_text()

    def test_symbol_passed_to_executor(self, harness, tmp_path, monkeypatch):
        _write(tmp_path, "t.py", "class A:\n    def m(self):\n        return 1\n")
        seen = {}
        import external_llm.agent.tool_handlers.ast_op_executor as ex_mod
        orig_apply = ex_mod.ASTOpExecutor.apply

        def _spy(self, source, ops, symbol=""):
            seen["symbol"] = symbol
            return orig_apply(self, source, ops, symbol)
        monkeypatch.setattr(ex_mod.ASTOpExecutor, "apply", _spy)
        r = harness._tool_edit_ast({
            "file_path": "t.py",
            "symbol": "A.m",
            "ops": [{"type": "replace_expr", "old": "return 1", "new": "return 2"}],
        })
        assert r.ok is True
        assert seen["symbol"] == "A.m"


# ── Failure / idempotent / dry-run branches ─────────────────────────────────

class TestResultBranches:
    def test_executor_failure_with_hint(self, harness, tmp_path):
        _write(tmp_path, "t.py", "x = 1\n")
        r = harness._tool_edit_ast({
            "file_path": "t.py",
            "ops": [{"type": "replace_expr", "old": "zzz_not_there", "new": "q"}],
        })
        assert r.ok is False
        assert "AST edit failed" in r.error
        assert "no match found" in r.error

    def test_idempotent_no_change(self, harness, tmp_path):
        _write(tmp_path, "t.py", "x = 1\n")
        r = harness._tool_edit_ast({
            "file_path": "t.py",
            "ops": [{"type": "replace_expr", "old": "x = 1", "new": "x = 1"}],
        })
        assert r.ok is True
        assert "no changes needed" in r.content
        assert r.metadata["changed"] is False

    def test_post_apply_compile_gate(self, harness, tmp_path, monkeypatch):
        _write(tmp_path, "t.py", "x = 1\n")
        import external_llm.agent.tool_handlers.ast_op_executor as ex_mod

        class _FakeResult:
            success = True
            changed = True
            ops_applied = 1
            ops_failed = ()
            new_source = "def broken(:\n"
        monkeypatch.setattr(ex_mod.ASTOpExecutor, "apply",
                            lambda self, source, ops, symbol="": _FakeResult())
        r = harness._tool_edit_ast({
            "file_path": "t.py",
            "ops": [{"type": "replace_expr"}],
        })
        assert r.ok is False
        assert "invalid syntax" in r.error
        # File must not have been written
        assert (tmp_path / "t.py").read_text() == "x = 1\n"

    def test_dry_run_preview_no_write(self, harness, tmp_path):
        _write(tmp_path, "t.py", "x = 1\n")
        r = harness._tool_edit_ast({
            "file_path": "t.py",
            "dry_run": True,
            "ops": [{"type": "replace_expr", "old": "x = 1", "new": "x = 9"}],
        })
        assert r.ok is True
        assert "[DRY RUN]" in r.content
        assert r.metadata["dry_run"] is True
        assert r.metadata["changed"] is True
        assert (tmp_path / "t.py").read_text() == "x = 1\n"


# ── Write errors ────────────────────────────────────────────────────────────

class TestWriteErrors:
    def test_encode_error_on_write(self, harness, tmp_path, monkeypatch):
        _write(tmp_path, "t.py", "x = 1\n")
        from external_llm.agent.tool_handlers import write_tools_ast_mixin as mod
        monkeypatch.setattr(
            mod, "read_text_with_encoding_fallback",
            lambda _p: ("x = 'é'\n", "ascii"),
        )
        r = harness._tool_edit_ast({
            "file_path": "t.py",
            "ops": [{"type": "replace_expr", "old": "x = 'é'", "new": "x = 'ü'"}],
        })
        assert r.ok is False
        assert "Failed to write" in r.error

    def test_oserror_on_write(self, harness, tmp_path, monkeypatch):
        _write(tmp_path, "t.py", "x = 1\n")
        from external_llm.agent.tool_handlers import write_tools_ast_mixin as mod

        def _raise_oserror(_path, _data, **kw):
            raise OSError("denied")
        monkeypatch.setattr(mod, "atomic_write_bytes", _raise_oserror)
        r = harness._tool_edit_ast({
            "file_path": "t.py",
            "ops": [{"type": "replace_expr", "old": "x = 1", "new": "x = 7"}],
        })
        assert r.ok is False
        assert "Failed to write" in r.error
        # Read path must still work — only the write failed
        assert (tmp_path / "t.py").read_text() == "x = 1\n"


class TestAtomicWriteFunnel:
    """edit_ast must write through atomic_write_bytes (crash-safety + cache
    invalidation), never a raw open("wb") truncating write."""

    def test_write_routes_through_atomic_write_bytes(self, harness, tmp_path, monkeypatch):
        _write(tmp_path, "t.py", "x = 1\n")
        from external_llm.agent.tool_handlers import write_tools_ast_mixin as mod
        calls = []
        real = mod.atomic_write_bytes

        def _spy(path, data, **kw):
            calls.append((str(path), data))
            return real(path, data, **kw)
        monkeypatch.setattr(mod, "atomic_write_bytes", _spy)
        r = harness._tool_edit_ast({
            "file_path": "t.py",
            "ops": [{"type": "replace_expr", "old": "x = 1", "new": "x = 7"}],
        })
        assert r.ok is True
        assert len(calls) == 1, f"expected exactly one atomic write, got {len(calls)}"
        path, data = calls[0]
        assert path == str((tmp_path / "t.py").resolve())
        assert data == b"x = 7\n"
        assert (tmp_path / "t.py").read_text() == "x = 7\n"

    def test_handler_has_no_raw_truncating_open(self):
        """Structural guard (mirrors the write-safety repair-path audit): the
        handler body must not contain a direct open(..., 'w'|'wb') call."""
        import ast as _ast
        import inspect
        import textwrap

        from external_llm.agent.tool_handlers import write_tools_ast_mixin as mod
        tree = _ast.parse(textwrap.dedent(inspect.getsource(mod.WriteToolsAstMixin._tool_edit_ast)))
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Call) and getattr(node.func, "id", "") == "open":
                raise AssertionError(
                    "_tool_edit_ast must write via atomic_write_bytes, not open(...)"
                )
