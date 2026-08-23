"""RED→GREEN coverage tests for read_tools.py (83% → 100%).

Covers the remaining edge branches: glob error paths, read_file failures,
grep fallback/retry/timeout paths, find_symbol detail rendering,
find_references, get_file_outline, find_relevant_files, and read_image —
most of which had no behavioral test at all.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import types
from pathlib import Path

import pytest

from external_llm.agent.tool_handlers.read_tools import (
    ReadToolsMixin,
    _glob_to_regex,
)

# ── module-level helpers ────────────────────────────────────────────────────


def test_glob_to_regex_bare_double_star():
    """A bare `**` (not `**/`) matches any characters including separators."""
    rx = _glob_to_regex("**")
    assert rx.match("a/b/c.py")
    assert rx.match("x")


# ── glob tool edges ─────────────────────────────────────────────────────────


def test_glob_scope_value_error_reported_outside(tool_registry, monkeypatch, tmp_path):
    """The defensive relative_to ValueError (scope resolved outside the root
    between the two checks) degrades to the same outside-repo error."""
    monkeypatch.setattr(
        tool_registry,
        "_secure_path",
        lambda *a, **kw: tmp_path / "outside",
    )
    res = tool_registry.dispatch("glob", {"pattern": "*.py", "path": "sub"})
    assert not res.ok
    assert "outside the repository" in res.error


def test_glob_missing_file_in_index_degrades(monkeypatch, tool_registry, tmp_path):
    """A file that left the index (deleted between index and stat) must not
    fail the call — it sorts last and renders without an age."""
    (tmp_path / "real.py").write_text("x = 1\n")
    monkeypatch.setattr(
        "external_llm.agent.tool_handlers.write_tools._repo_file_index",
        lambda root: ["ghost.py", "real.py"],
    )
    res = tool_registry.dispatch("glob", {"pattern": "*.py"})
    assert res.ok, res.error
    assert "ghost.py" in res.content
    assert "real.py" in res.content


def test_read_file_empty_path_error(tool_registry):
    res = tool_registry.dispatch("read_file", {})
    assert not res.ok
    assert "'path' is required" in res.error


def test_read_file_open_failure_reported(tool_registry, tmp_path):
    secret = Path(tool_registry.repo_root) / "secret.txt"
    secret.write_text("hidden\n")
    secret.chmod(0)
    try:
        res = tool_registry.dispatch("read_file", {"path": "secret.txt"})
        assert not res.ok
        assert "Failed to read" in res.error
    finally:
        secret.chmod(0o644)


def test_over_cap_guidance_outline_failure_falls_back(tool_registry, monkeypatch):
    """A failing outline degrades to the plain count + range hint."""
    monkeypatch.setattr(
        tool_registry._symbol_searcher,
        "get_file_outline",
        lambda path: (_ for _ in ()).throw(RuntimeError("no grammar")),
    )
    out = tool_registry._over_cap_guidance("big.py", 5000)
    assert "5000 lines" in out
    assert "start_line and end_line" in out


# ── _run_search_bounded failure branches ────────────────────────────────────


def test_run_search_bounded_popen_oserror(tool_registry, monkeypatch):
    def _boom(*a, **kw):
        raise OSError("rg not installed")

    monkeypatch.setattr(subprocess, "Popen", _boom)
    with pytest.raises(RuntimeError, match="could not start"):
        ReadToolsMixin._run_search_bounded(["rg"], ".", 1, 5)


def test_run_search_bounded_drain_and_close_failures(tool_registry, monkeypatch):
    """Pipes closing under the drain threads degrade to partial results, and
    close() failures in the finally are logged, never raised."""

    class _Out:
        def __iter__(self):
            yield "line1\n"
            raise OSError("pipe closed")

        def close(self):
            raise ValueError("already closed")

    class _Err:
        def read(self, n):
            raise OSError("pipe closed")

        def close(self):
            raise ValueError("already closed")

    class _Proc:
        pid = 123
        returncode = 0
        stdout = _Out()
        stderr = _Err()

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: _Proc())
    rc, lines, total, stderr = ReadToolsMixin._run_search_bounded(["grep"], ".", 5, 5)
    assert rc == 0
    assert lines == ["line1"]
    assert total == 1
    assert stderr == ""


def test_run_search_bounded_timeout_killpg_missing(tool_registry, monkeypatch):
    """A hung search times out and tears the process group down; a killpg
    ProcessLookupError (leader already reaped) is tolerated."""

    def _gen():
        while True:
            yield "x\n"

    class _Out:
        def __iter__(self):
            return _gen()

        def close(self):
            pass

    class _Err:
        def read(self, n):
            return ""

        def close(self):
            pass

    class _Proc:
        pid = 999
        stdout = _Out()
        stderr = _Err()

        def wait(self, timeout=None):
            return None

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: _Proc())
    monkeypatch.setattr(os, "killpg", lambda *a, **kw: (_ for _ in ()).throw(ProcessLookupError()))
    with pytest.raises(subprocess.TimeoutExpired):
        ReadToolsMixin._run_search_bounded(["sleep", "5"], ".", 0.05, 5)


# ── grep tool edges ─────────────────────────────────────────────────────────


def test_grep_empty_pattern_error(tool_registry):
    res = tool_registry.dispatch("grep", {})
    assert not res.ok
    assert "'pattern' is required" in res.error


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
def test_grep_rg_ignore_case_and_include(tool_registry):
    (Path(tool_registry.repo_root) / "probe.py").write_text("NEEDLE = 1\n")
    res = tool_registry.dispatch("grep", {"pattern": "needle", "ignore_case": True, "include": "*.py"})
    assert res.ok, res.error
    assert "NEEDLE" in res.content


def test_grep_system_grep_fallback_flags(tool_registry, monkeypatch):
    """Without rg, the system-grep fallback builds -i/-C/--include/-E flags."""
    (Path(tool_registry.repo_root) / "probe.py").write_text("def probe():\n    return 1\n")
    monkeypatch.setattr(shutil, "which", lambda *a, **kw: None)
    res = tool_registry.dispatch(
        "grep",
        # regex special chars → use_fixed False → -E (extended regex)
        {"pattern": "def pro.*", "ignore_case": True, "context": 1, "include": "*.py"},
    )
    assert res.ok, res.error
    assert "probe" in res.content


def test_grep_timeout_reported(tool_registry, monkeypatch):
    def _hung(*a, **kw):
        raise subprocess.TimeoutExpired(["rg"], 120)

    monkeypatch.setattr(ReadToolsMixin, "_run_search_bounded", staticmethod(_hung))
    res = tool_registry.dispatch("grep", {"pattern": "x"})
    assert res.ok
    assert "timed out" in res.content


def test_grep_generic_failure_reported(tool_registry, monkeypatch):
    def _boom(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(ReadToolsMixin, "_run_search_bounded", staticmethod(_boom))
    res = tool_registry.dispatch("grep", {"pattern": "x"})
    assert not res.ok
    assert "grep failed: boom" in res.error


def test_grep_char_budget_cut(tool_registry, monkeypatch):
    """A result overflowing the char budget is cut mid-line and reported."""
    from external_llm.agent.config import thresholds as _th

    lines = "".join(f"line {i} " + "y" * 90 + "\n" for i in range(50))
    (Path(tool_registry.repo_root) / "wide.log").write_text(lines)
    tok = _th.config.tokens
    orig = tok.BASH_OUTPUT_MAX_CHARS
    object.__setattr__(tok, "BASH_OUTPUT_MAX_CHARS", 200)  # frozen dataclass
    try:
        res = tool_registry.dispatch("grep", {"pattern": "line", "max_results": 50})
    finally:
        object.__setattr__(tok, "BASH_OUTPUT_MAX_CHARS", orig)
    assert res.ok, res.error
    assert "truncated at 200 characters" in res.content


# ── read_symbol / find_symbol ───────────────────────────────────────────────


def test_read_symbol_not_found(tool_registry):
    res = tool_registry.dispatch("read_symbol", {"name": "zzz_absent"})
    assert res.ok
    assert "not found" in res.content


def test_read_symbol_multiple_definitions_header(tool_registry):
    (Path(tool_registry.repo_root) / "a.py").write_text("def shared():\n    return 1\n")
    (Path(tool_registry.repo_root) / "b.py").write_text("def shared():\n    return 2\n")
    res = tool_registry.dispatch("read_symbol", {"name": "shared"})
    assert res.ok, res.error
    assert "1st of 2 definitions" in res.content
    assert "others at" in res.content


def test_read_symbol_file_missing(tool_registry, monkeypatch):
    fake = types.SimpleNamespace(file="ghost.py", line=1, end_line=1, kind="function", name="g")
    monkeypatch.setattr(tool_registry._symbol_searcher, "find_symbol", lambda *a, **kw: [fake])
    res = tool_registry.dispatch("read_symbol", {"name": "g"})
    assert res.ok
    assert "File 'ghost.py' not found" in res.content


def test_find_symbol_empty_name_error(tool_registry):
    res = tool_registry.dispatch("find_symbol", {})
    assert not res.ok
    assert "'name' is required" in res.error


def test_find_symbol_truncated_index_note(tool_registry, monkeypatch):
    monkeypatch.setattr(tool_registry._symbol_searcher, "find_symbol", lambda *a, **kw: [])
    monkeypatch.setattr(tool_registry._symbol_searcher, "index_was_truncated", lambda *a, **kw: True)
    res = tool_registry.dispatch("find_symbol", {"name": "x"})
    assert res.ok
    assert "index was truncated" in res.content


def test_find_symbol_detail_fields_and_inheritance(tool_registry, monkeypatch):
    # The real indexer omits bases/decorators for same-file classes, so a rich
    # fake def drives the rendering branches (docstring/bases/methods/decorators).
    rich = types.SimpleNamespace(
        kind="class",
        name="C",
        file="detail.py",
        line=8,
        signature="",
        docstring="Docstring.",
        bases=["Base"],
        methods=[f"m{i}" for i in range(12)],
        decorators=["deco"],
    )
    monkeypatch.setattr(
        tool_registry._symbol_searcher,
        "find_symbol",
        lambda name, kind="any", search_path=None: [rich],
    )
    res = tool_registry.dispatch("find_symbol", {"name": "C"})
    assert res.ok, res.error
    assert "docstring" in res.content and "Docstring." in res.content
    assert "bases" in res.content and "Base" in res.content
    assert "methods" in res.content and "(+2 more)" in res.content  # 12 methods, first 10 shown
    assert "decorators" in res.content and "deco" in res.content

    info = {
        "subclasses": ["D"],
        "reference_count": 3,
        "referenced_in": ["b.py:5"],
        "sample_references": [{"file": "b.py", "line": 5, "context": "x = C()"}],
        "other_definitions": [{"kind": "class", "file": "c.py", "line": 2}],
    }
    monkeypatch.setattr(tool_registry._symbol_searcher, "get_symbol_info", lambda *a, **kw: info)
    res2 = tool_registry.dispatch("find_symbol", {"name": "C", "include_inheritance": True})
    assert res2.ok, res2.error
    assert "Subclasses : D" in res2.content
    assert "References : 3" in res2.content
    assert "Used in    : b.py:5" in res2.content
    assert "Sample references" in res2.content
    assert "Other definitions" in res2.content


# ── find_references ─────────────────────────────────────────────────────────


def test_find_references_all_paths(tool_registry):
    (Path(tool_registry.repo_root) / "refs.py").write_text("def f():\n    return 1\n\nx = f()\n")
    res = tool_registry.dispatch("find_references", {"name": "f"})
    assert res.ok, res.error
    assert "reference(s)" in res.content

    res_none = tool_registry.dispatch("find_references", {"name": "zzz_absent"})
    assert res_none.ok
    assert "No references found" in res_none.content

    res_missing = tool_registry.dispatch("find_references", {})
    assert not res_missing.ok
    assert "'name'" in res_missing.error


# ── get_file_outline ────────────────────────────────────────────────────────


def _sym(kind, name, line, end=None, signature="", bases=None, methods=None):
    return types.SimpleNamespace(
        kind=kind,
        name=name,
        line=line,
        end_line=end,
        signature=signature,
        bases=bases or [],
        methods=methods or [],
    )


def test_get_file_outline_errors(tool_registry):
    res = tool_registry.dispatch("get_file_outline", {})
    assert not res.ok
    assert "'path' is required" in res.error

    res2 = tool_registry.dispatch("get_file_outline", {"path": "../outside.py"})
    assert not res2.ok
    assert "outside repo" in res2.error

    (Path(tool_registry.repo_root) / "empty.py").write_text("")
    res3 = tool_registry.dispatch("get_file_outline", {"path": "empty.py"})
    assert res3.ok
    assert "No symbols found" in res3.content


def test_get_file_outline_kind_rendering(tool_registry, monkeypatch):
    (Path(tool_registry.repo_root) / "shapes.py").write_text("x = 1\n")
    mixed = [
        _sym("class", "C", 1, end=20, bases=["B"], methods=["m1", "m2"]),
        _sym("function", "f", 25, signature="(x)"),
        _sym("variable", "v", 30, signature="= 1"),
        _sym("macro", "M", 35, signature="..."),  # unknown kind → else branch
    ]
    monkeypatch.setattr(tool_registry._symbol_searcher, "get_file_outline", lambda path: mixed)
    res = tool_registry.dispatch("get_file_outline", {"path": "shapes.py"})
    assert res.ok, res.error
    assert "lines 1–20" in res.content  # noqa: RUF001 — get_file_outline emits an EN DASH in ranges
    assert "bases: B" in res.content
    assert "methods: m1, m2" in res.content
    assert "(line 25)" in res.content  # single-line extent
    assert "f((x)) (line 25)" in res.content  # function signature in parens
    assert "v (line 30) — = 1" in res.content
    assert "[macro] M (line 35)" in res.content
    assert "Use read_symbol" in res.content


# ── find_relevant_files / read_image ────────────────────────────────────────


def test_find_relevant_files_empty_query_error(tool_registry):
    res = tool_registry.dispatch("find_relevant_files", {})
    assert not res.ok
    assert "'query' is required" in res.error


def test_read_image_all_paths(tool_registry, monkeypatch):
    res = tool_registry.dispatch("read_image", {})
    assert not res.ok
    assert "'path' is required" in res.error

    res2 = tool_registry.dispatch("read_image", {"path": "../outside.png"})
    assert not res2.ok
    assert "outside repo" in res2.error

    (Path(tool_registry.repo_root) / "adir").mkdir()
    res3 = tool_registry.dispatch("read_image", {"path": "adir"})
    assert not res3.ok
    assert "Not a file" in res3.error

    img = Path(tool_registry.repo_root) / "pic.png"
    img.write_bytes(b"\x89PNG fake")
    monkeypatch.setattr(
        "external_llm.providers._try_ocr_base64",
        lambda data: "OCR TEXT",
    )
    res4 = tool_registry.dispatch("read_image", {"path": "pic.png"})
    assert res4.ok, res4.error
    assert "OCR TEXT" in res4.content

    (Path(tool_registry.repo_root) / "pic2.png").write_bytes(b"\x89PNG fake2")
    monkeypatch.setattr(
        "external_llm.providers._try_ocr_base64",
        lambda data: "",
    )
    res5 = tool_registry.dispatch("read_image", {"path": "pic2.png"})
    assert res5.ok
    assert "No text detected" in res5.content

    locked = Path(tool_registry.repo_root) / "locked.png"
    locked.write_bytes(b"data")
    locked.chmod(0)
    try:
        res6 = tool_registry.dispatch("read_image", {"path": "locked.png"})
        assert not res6.ok
        assert "Failed to read image file" in res6.error
    finally:
        locked.chmod(0o644)
