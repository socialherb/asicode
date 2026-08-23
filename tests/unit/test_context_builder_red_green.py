"""
RED→GREEN tests for external_llm/context_builder.py (56% → 100%).

Real defects found while writing these (fixed in the same change):
  CB-1 fallback target leak: _find_related_files' fallback compared the raw
      ``target_file``, while the preferred path compares
      normalize_rel_path(target_file) (SSOT).  A "./pkg/__init__.py"-style
      target that imports its own package never matched, so the TARGET
      leaked into its own Related Files — its content embedded twice in
      the prompt (defect A class, fallback half).
  CB-2 related-file output bound: related snippets embedded up to 1 MiB of
      raw content each with NO line cap, while the target block is capped
      at 5000 numbered lines (~70-90 KB) — the P21-3 "cap the OUTPUT
      expansion too" principle was applied to the target only.
  CB-3 marker honesty: the target-block footer claimed "file exceeds 1 MiB"
      even when truncation was caused solely by the 5000-line cap of a
      <1 MiB file — a false statement shipped to the model.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

import external_llm.context_builder as cb
from external_llm.context_builder import (
    ContextBuilder,
    EnhancedContextBuilder,
    _bounded_file_text,
    enhance_user_request,
)

# ── fixtures / helpers ───────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def isolated_caches():
    """Both process-wide TTL caches must not leak between tests."""
    hints = cb._structure_hints_cache
    logs = cb._git_log_cache
    hints.clear()
    logs.clear()
    try:
        yield
    finally:
        hints.clear()
        logs.clear()


@pytest.fixture
def builder(tmp_path):
    (tmp_path / "helper.py").write_text("H = 1\n", encoding="utf-8")
    (tmp_path / "targetmod.py").write_text("import helper\nfrom other import y\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("Y = 2\n", encoding="utf-8")
    return EnhancedContextBuilder(str(tmp_path))


def _patch_preferred(monkeypatch, selected=None, exc=False):
    """Control the context_collector preferred path of _find_related_files."""

    def fake(_root, _target):
        if exc:
            raise RuntimeError("context_collector forced failure")
        return (list(selected or []), {"reason": "forced"})

    monkeypatch.setattr("context_collector.collect_related_files_shallow", fake, raising=True)


# ── CB-1 (RED): fallback must exclude a "./"-prefixed target ────────────────


def test_fallback_excludes_prefixed_target(builder, tmp_path, monkeypatch):
    """'./pkg/__init__.py' importing its own package must NOT list itself.

    Old bug: fallback compared relp != target_file RAW — 'pkg/__init__.py'
    != './pkg/__init__.py' is True, so the target leaked into Related Files
    (duplicate content in the prompt).  The preferred path already compares
    the normalized form."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from pkg.core import x\n", encoding="utf-8")
    (pkg / "core.py").write_text("X = 1\n", encoding="utf-8")

    _patch_preferred(monkeypatch, selected=[])  # empty -> falls through

    assert builder._find_related_files("./pkg/__init__.py", 3) == []
    # already-normalized form behaves identically
    assert builder._find_related_files("pkg/__init__.py", 3) == []


# ── CB-2 (RED): related snippets need the same output line cap ──────────────


def test_related_snippet_line_capped(builder, tmp_path, monkeypatch):
    """A 5001-line related file (<1 MiB) must be line-capped like the target.

    Old bug: the target block is capped at 5000 numbered lines but related
    snippets embedded the raw head with no line cap — ~1 MiB of prompt per
    related file, dwarfing the target's own ~70-90 KB output bound."""
    long_rel = tmp_path / "bigrel.py"
    long_rel.write_text("\n".join(f"LINE_{i}" for i in range(1, 5002)) + "\n", encoding="utf-8")
    assert long_rel.stat().st_size < cb._FILE_CONTEXT_MAX_BYTES  # bytes-bound OK

    _patch_preferred(monkeypatch, selected=["bigrel.py"])
    ctx = builder._build_related_files_context("targetmod.py", max_files=3)

    assert "...[more lines omitted" in ctx
    assert f"showing first {cb._FILE_CONTEXT_MAX_LINES}" in ctx
    assert "LINE_5001" not in ctx  # beyond the cap: dropped
    assert "LINE_4999" in ctx  # inside the cap: kept


# ── CB-3 (RED): truncation footer must state the true cause ────────────────


def test_file_context_line_cap_marker_accurate(builder, tmp_path):
    """Line-cap-only truncation (file < 1 MiB) must NOT claim 'exceeds 1 MiB'."""
    big = tmp_path / "many.py"
    big.write_text("\n".join(f"x{i} = {i}" for i in range(1, 5002)) + "\n", encoding="utf-8")
    assert big.stat().st_size < cb._FILE_CONTEXT_MAX_BYTES

    ctx = builder._build_file_context("many.py")

    assert "more lines omitted" in ctx
    assert "exceeds 1 MiB" not in ctx  # the false claim (old behavior)
    assert "5000" in ctx  # the true cause: the line cap


# ── build_context assembly ──────────────────────────────────────────────────


def _seed_git(builder_root: Path, status=True, log=True) -> None:
    if status:
        cb._git_log_cache[(str(builder_root), 3)] = ("abc1234 fix thing", time.monotonic() + 999.0)


def test_build_context_full_assembly(builder, tmp_path, monkeypatch):
    monkeypatch.setattr(
        cb,
        "get_git_snapshot",
        lambda _r: {"status": "M targetmod.py", "branch": "main"},
    )
    _seed_git(builder.repo_root)
    _patch_preferred(monkeypatch, selected=["helper.py"])

    out = builder.build_context(
        user_request="fix the bug",
        target_file="targetmod.py",
    )
    assert isinstance(out, str)
    # ordered sections
    assert (
        out.index("# PROJECT CONTEXT FOR CODE EDITING")
        < out.index("## Git Status")
        < out.index("## Target File: `targetmod.py`")
        < out.index("## Related Files")
        < out.index("## Project Structure")
        < out.index("## User Request")
        < out.index("## Instructions")
    )
    assert "M targetmod.py" in out
    assert "abc1234 fix thing" in out
    assert "fix the bug" in out
    assert "### 1. `helper.py`" in out
    assert f"{tmp_path.name}/" in out


def test_build_context_without_git(builder, monkeypatch):
    _patch_preferred(monkeypatch, selected=["helper.py"])
    out = builder.build_context(user_request="r", target_file="targetmod.py", include_git_context=False)
    assert "## Git Status" not in out
    assert "## Target File" in out


def test_build_context_without_related(builder, monkeypatch):
    out = builder.build_context(user_request="r", target_file="targetmod.py", include_related_files=False)
    assert "## Related Files" not in out
    assert "## Target File" in out


def test_build_context_no_target(builder):
    out = builder.build_context(user_request="r")
    assert "## Target File" not in out
    assert "## Related Files" not in out
    assert "## User Request" in out
    assert "## Instructions" in out


def test_build_context_missing_target(builder):
    out = builder.build_context(user_request="r", target_file="ghost.py")
    assert "## Target File" not in out


def test_build_context_max_related_cap(builder, monkeypatch):
    _patch_preferred(monkeypatch, selected=["helper.py", "other.py"])
    out = builder.build_context(user_request="r", target_file="targetmod.py", max_related_files=1)
    assert "### 1. `helper.py`" in out
    assert "### 2." not in out


def test_context_builder_alias():
    assert ContextBuilder is EnhancedContextBuilder


# ── _build_git_context ──────────────────────────────────────────────────────


def test_git_context_status_and_commits(builder, monkeypatch):
    monkeypatch.setattr(cb, "get_git_snapshot", lambda _r: {"status": "M f"})
    cb._git_log_cache[(str(builder.repo_root), 3)] = ("deadbee commit", time.monotonic() + 999.0)
    ctx = builder._build_git_context()
    assert "M f" in ctx and "deadbee commit" in ctx and "**Recent Changes**:" in ctx


def test_git_context_status_only(builder, monkeypatch):
    monkeypatch.setattr(cb, "get_git_snapshot", lambda _r: {"status": "M f"})
    ctx = builder._build_git_context()  # no log entry, non-repo -> ""
    assert "M f" in ctx
    assert "**Recent Changes**:" not in ctx


def test_git_context_empty(builder, monkeypatch):
    monkeypatch.setattr(cb, "get_git_snapshot", lambda _r: {"status": ""})
    assert builder._build_git_context() == ""


# ── _fetch_recent_commits (real subprocess) ─────────────────────────────────


def test_fetch_recent_commits_real_repo(tmp_path):
    for args in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@t"],
        ["git", "config", "user.name", "t"],
    ):
        subprocess.run(args, cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "init commit"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    b = EnhancedContextBuilder(str(tmp_path))
    log = b._fetch_recent_commits(count=3)
    assert "init commit" in log


def test_fetch_recent_commits_non_repo(tmp_path):
    b = EnhancedContextBuilder(str(tmp_path))
    assert b._fetch_recent_commits(count=3) == ""


def test_fetch_recent_commits_subprocess_exception(builder, monkeypatch):
    def boom(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd="git", timeout=5)

    monkeypatch.setattr(cb.subprocess, "run", boom)
    assert builder._fetch_recent_commits(count=3) == ""


def test_cached_git_log_caches_empty_failure_sentinel(builder, monkeypatch):
    calls = {"n": 0}

    def fetch(_count):
        calls["n"] += 1
        return ""

    out1 = cb._cached_git_log(builder.repo_root, 3, fetch)
    out2 = cb._cached_git_log(builder.repo_root, 3, fetch)
    assert out1 == "" and out2 == ""
    assert calls["n"] == 1  # failure sentinel cached, no re-fetch within TTL


# ── _bounded_file_text ──────────────────────────────────────────────────────


def test_bounded_small_utf8(tmp_path):
    p = tmp_path / "s.py"
    p.write_text("héllo\n", encoding="utf-8")
    assert _bounded_file_text(p) == ("héllo\n", False)


def test_bounded_small_latin1_fallback(tmp_path):
    p = tmp_path / "s.bin"
    p.write_bytes(b"\xff\xfe raw bytes")
    assert _bounded_file_text(p) == ("ÿþ raw bytes", False)


def test_bounded_big_utf8_trims_incomplete_multibyte(tmp_path):
    p = tmp_path / "big.txt"
    body = "a" * (cb._FILE_CONTEXT_MAX_BYTES + 16)
    raw = body.encode("utf-8") + "가".encode()[:2]  # cut mid-char
    p.write_bytes(raw)
    text, truncated = _bounded_file_text(p)
    assert truncated is True
    assert text.endswith("a" * 16)
    assert len(text.encode("utf-8")) <= cb._FILE_CONTEXT_MAX_BYTES


def test_bounded_big_latin1_fallback(tmp_path):
    p = tmp_path / "big.bin"
    p.write_bytes(b"\xff" * (cb._FILE_CONTEXT_MAX_BYTES + 8))
    text, truncated = _bounded_file_text(p)
    assert truncated is True
    # The trailing 0xFF lead byte is dropped as an incomplete sequence; the
    # remaining invalid-utf-8 head falls back to latin-1.
    assert set(text) == {"\xff"}
    assert cb._FILE_CONTEXT_MAX_BYTES - 4 <= len(text) <= cb._FILE_CONTEXT_MAX_BYTES


# ── _build_file_context ─────────────────────────────────────────────────────


def test_file_context_numbered_lines(builder):
    ctx = builder._build_file_context("helper.py")
    assert ctx.startswith("```python\n   1 | H = 1\n")
    assert "**Total lines**: 2" in ctx  # "H = 1\n" -> 2 split-parts


def test_file_context_language_detection(builder, tmp_path):
    (tmp_path / "m.go").write_text("package main\n", encoding="utf-8")
    (tmp_path / "n.weird").write_text("???\n", encoding="utf-8")
    assert "```go\n" in builder._build_file_context("m.go")
    assert "```\n" in builder._build_file_context("n.weird")  # unknown -> no lang


def test_file_context_missing_file(builder):
    assert builder._build_file_context("ghost.py") == ""


def test_file_context_oserror(builder, monkeypatch):
    def boom(_p, _m=None):
        raise OSError("disk gone")

    monkeypatch.setattr(cb, "_bounded_file_text", boom)
    assert builder._build_file_context("helper.py") == ""


def test_file_context_broad_exception(builder, monkeypatch):
    def boom(_fn):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(builder, "_detect_language", boom)
    assert builder._build_file_context("helper.py") == ""


def test_file_context_byte_truncation_footer(builder, tmp_path):
    p = tmp_path / "huge.py"
    p.write_text("x" * (cb._FILE_CONTEXT_MAX_BYTES + 4096) + "\n", encoding="utf-8")
    ctx = builder._build_file_context("huge.py")
    assert "exceeds 1 MiB" in ctx
    assert "**Total lines**: >=" in ctx


# ── _build_related_files_context ────────────────────────────────────────────


def test_related_none(builder, monkeypatch):
    monkeypatch.setattr(builder, "_find_related_files", lambda _t, _m: [])
    assert builder._build_related_files_context("targetmod.py", max_files=3) == ""


def test_related_missing_files_skipped(builder, monkeypatch):
    _patch_preferred(monkeypatch, selected=["ghost.py"])
    assert builder._build_related_files_context("targetmod.py", max_files=3) == ""


def test_related_read_failure_skipped(builder, tmp_path, monkeypatch):
    real = cb._bounded_file_text

    def flaky(p, _m=None):
        if Path(p).name == "other.py":
            raise OSError("unreadable")
        return real(p)

    monkeypatch.setattr(cb, "_bounded_file_text", flaky)
    _patch_preferred(monkeypatch, selected=["helper.py", "other.py"])
    ctx = builder._build_related_files_context("targetmod.py", max_files=3)
    assert "`helper.py`" in ctx
    assert "`other.py`" not in ctx


def test_related_truncated_marker(builder, tmp_path, monkeypatch):
    (tmp_path / "bigrel2.py").write_text("z" * (cb._FILE_CONTEXT_MAX_BYTES + 100), encoding="utf-8")
    _patch_preferred(monkeypatch, selected=["bigrel2.py"])
    ctx = builder._build_related_files_context("targetmod.py", max_files=3)
    assert "...[TRUNCATED — head only]..." in ctx


def test_related_broad_exception(builder, monkeypatch):
    def boom(_t, _m):
        raise RuntimeError("finder broke")

    monkeypatch.setattr(builder, "_find_related_files", boom)
    assert builder._build_related_files_context("targetmod.py", max_files=3) == ""


# ── _find_related_files: fallback import parsing ────────────────────────────


def test_fallback_resolves_imports(builder, monkeypatch):
    _patch_preferred(monkeypatch, selected=[])
    assert builder._find_related_files("targetmod.py", 3) == ["helper.py", "other.py"]


def test_fallback_relative_imports_skipped(builder, tmp_path, monkeypatch):
    (tmp_path / "rel.py").write_text(
        "# plain comment line (no import -> skipped by the parser)\n"
        "x = 1\n"
        "from . import sibling\nfrom .pkg import thing\nimport helper\n",
        encoding="utf-8",
    )
    _patch_preferred(monkeypatch, selected=[])
    assert builder._find_related_files("rel.py", 3) == ["helper.py"]


def test_fallback_dotted_top_module(builder, tmp_path, monkeypatch):
    (tmp_path / "t2.py").write_text("import os.path\n", encoding="utf-8")
    (tmp_path / "os.py").write_text("stub\n", encoding="utf-8")
    _patch_preferred(monkeypatch, selected=[])
    assert builder._find_related_files("t2.py", 3) == ["os.py"]


def test_fallback_escape_returns_empty(builder, monkeypatch):
    _patch_preferred(monkeypatch, selected=[])
    assert builder._find_related_files("../outside.py", 3) == []


def test_fallback_missing_target(builder, monkeypatch):
    _patch_preferred(monkeypatch, selected=[])
    assert builder._find_related_files("ghost.py", 3) == []


def test_fallback_unsupported_extension(builder, tmp_path, monkeypatch):
    (tmp_path / "notes.txt").write_text("import helper\n", encoding="utf-8")
    _patch_preferred(monkeypatch, selected=[])
    assert builder._find_related_files("notes.txt", 3) == []


def test_fallback_unreadable_target(builder, monkeypatch):
    def boom(_p, _m=None):
        raise UnicodeDecodeError("utf-8", b"", 0, 1, "bad")

    monkeypatch.setattr(cb, "_bounded_file_text", boom)
    _patch_preferred(monkeypatch, selected=[])
    assert builder._find_related_files("targetmod.py", 3) == []


def test_fallback_max_files_cap(builder, monkeypatch):
    _patch_preferred(monkeypatch, selected=[])
    assert builder._find_related_files("targetmod.py", 1) == ["helper.py"]


def test_fallback_dedup(builder, tmp_path, monkeypatch):
    (tmp_path / "dup.py").write_text("import helper\nimport helper\nfrom helper import z\n", encoding="utf-8")
    _patch_preferred(monkeypatch, selected=[])
    assert builder._find_related_files("dup.py", 3) == ["helper.py"]


def test_fallback_after_preferred_exception(builder, monkeypatch):
    _patch_preferred(monkeypatch, exc=True)
    assert builder._find_related_files("targetmod.py", 3) == ["helper.py", "other.py"]


def test_fallback_broad_exception(builder, monkeypatch):
    def boom(_root, _rel):
        raise RuntimeError("resolve broke")

    monkeypatch.setattr(cb, "resolve_inside_repo", boom)
    _patch_preferred(monkeypatch, selected=[])
    assert builder._find_related_files("targetmod.py", 3) == []


# ── _get_project_structure_hints ────────────────────────────────────────────


def test_hints_skip_dotfiles(builder, tmp_path):
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "x.py").write_text("1\n", encoding="utf-8")
    (tmp_path / ".env").write_text("K=V\n", encoding="utf-8")
    hints = builder._get_project_structure_hints()
    assert ".hidden" not in hints
    assert ".env" not in hints
    assert "helper.py" in hints or "targetmod.py" in hints


def test_hints_iterdir_failure_not_cached(tmp_path):
    f = tmp_path / "notadir"
    f.write_text("x", encoding="utf-8")
    b = EnhancedContextBuilder(str(f))
    assert b._get_project_structure_hints() == ""
    assert str(b.repo_root) not in cb._structure_hints_cache  # failures uncached


def test_hints_stale_current_key_rebuilt(builder, tmp_path):
    key = str(builder.repo_root)
    cb._structure_hints_cache[key] = ("STALE", time.monotonic() - 1.0)
    out = builder._get_project_structure_hints()
    assert out != "STALE"
    _text, expiry = cb._structure_hints_cache[key]
    assert expiry > time.monotonic()  # re-cached fresh


# ── _detect_language / _get_llm_instructions / enhance_user_request ─────────


def test_detect_language_map(builder):
    m = builder._detect_language
    assert m("a.py") == "python"
    assert m("a.JS") == "javascript"  # suffix lower-cased
    assert m("a.tsx") == "tsx"
    assert m("a.go") == "go"
    assert m("a.rs") == "rust"
    assert m("a.yaml") == "yaml" and m("a.yml") == "yaml"
    assert m("a.md") == "markdown"
    assert m("a.unknown") == ""


def test_llm_instructions_with_target(builder):
    s = builder._get_llm_instructions("mod.py")
    assert "for `mod.py`" in s and "unified diff" in s


def test_llm_instructions_without_target(builder):
    s = builder._get_llm_instructions(None)
    assert "for `" not in s and "unified diff" in s


def test_enhance_no_hints_returns_original():
    assert enhance_user_request("hello") == "hello"
    assert enhance_user_request("hello", target_file=None, extra_hints=None) == "hello"


def test_enhance_target_and_hints():
    out = enhance_user_request("do it  \n", target_file="m.py", extra_hints=["a", "", "b"])
    assert out.startswith("do it\n\n[HINTS]\n")
    assert "- Target file: m.py" in out
    assert "- a" in out and "- b" in out
    assert "- \n" not in out  # empty hint filtered


def test_enhance_extra_hints_only():
    out = enhance_user_request("q", extra_hints=["only"])
    assert "[HINTS]\n- only" in out and "Target file" not in out
