"""RED→GREEN round: close remaining context_collector coverage gaps.

Baseline: 94/243 statements missed (61%) with the existing suites. The gaps
are the branch/edge machinery of the collector itself:

- _decode_bytes_best_effort: all-strict-decodes-fail fallback
- _truncate_utf8_bytes_safe: non-positive max_bytes + mid-multibyte cut
- _parse_python_imports: from-import lines, relative (./../) resolution,
  ascent-past-root guard
- _module_to_repo_paths: empty / dot-only module
- _parse_kotlin_import_symbols + _find_kotlin_files_for_symbol: whole Kotlin
  symbol machinery (alias, wildcard, lowercase, proximity scoring)
- collect_related_files_shallow: no_target / empty_target / target_missing /
  unsupported_ext / read_error / py+kt selection-cap break
- read_file_snippet_context: missing_args / read_error / invalid regex /
  regex-not-found fallback / byte-truncated body marker
"""
from __future__ import annotations

import context_collector as cc

# ---------- decode / truncate hardening ----------

def test_decode_all_strict_decodes_fail_falls_back_to_replace():
    # \xfe\xff is invalid in utf-8/utf-8-sig AND in cp949/euc-kr (bad trail).
    text, enc = cc._decode_bytes_best_effort(b"\xfe\xff")
    assert text == "\ufffd\ufffd"
    assert enc == "utf-8(replace)"


def test_truncate_utf8_nonpositive_max_bytes():
    assert cc._truncate_utf8_bytes_safe("abc", 0) == ("", True)
    assert cc._truncate_utf8_bytes_safe("abc", -5) == ("", True)


def test_truncate_utf8_mid_multibyte_cut_is_char_safe():
    # "a가" is b"a\xea\xb0\x80"; cutting at 3 bytes lands inside 가.
    assert cc._truncate_utf8_bytes_safe("a가", 3) == ("a", True)
    # exact fit is untouched
    assert cc._truncate_utf8_bytes_safe("abc", 3) == ("abc", False)


# ---------- python import parsing ----------

def test_parse_python_from_import_and_relative_resolution():
    mods = cc._parse_python_imports(
        "from os import path\n"
        "from .pkg import x\n"
        "from ..lib import y\n"
        "from . import z\n"
        "import a, b.c as cc2\n",
        "src/mods/app.py",
    )
    assert mods == ["os", "src.mods.pkg", "src.lib", "src.mods", "a", "b.c"]


def test_parse_python_deep_relative_ascent_stops_at_root():
    # rel_path with no parent dirs -> base_dir == Path(".") -> ascent guard.
    assert cc._parse_python_imports("from ......x import y", "app.py") == ["x"]


def test_parse_python_empty_import_lines_produce_nothing():
    assert cc._parse_python_imports("import  \nfrom  import x\n", "app.py") == []
    assert cc._parse_python_imports("", "app.py") == []


def test_module_to_repo_paths_empty_module():
    assert cc._module_to_repo_paths("/tmp", "") == []
    assert cc._module_to_repo_paths("/tmp", "...") == []


def test_module_to_repo_paths_resolves_real_files(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    assert cc._module_to_repo_paths(str(repo), "pkg.mod") == ["pkg/mod.py"]
    assert cc._module_to_repo_paths(str(repo), "pkg") == ["pkg/__init__.py"]


# ---------- kotlin symbol machinery ----------

def test_parse_kotlin_import_symbols():
    syms = cc._parse_kotlin_import_symbols(
        "package com.example\n"
        "// a comment line\n"
        "import com.example.Foo\n"
        "import com.example.Bar as B\n"
        "import com.example.baz\n"
        "import com.example.*\n"
        "import com.example.Foo\n"
    )
    assert syms == ["Foo", "Bar"]


def test_find_kotlin_files_for_symbol_prefers_nearby(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src" / "a").mkdir(parents=True)
    (repo / "src" / "b").mkdir(parents=True)
    (repo / "src" / "a" / "Widget.kt").write_text("class Widget\n", encoding="utf-8")
    (repo / "src" / "b" / "Widget.kt").write_text("class Widget\n", encoding="utf-8")

    found = cc._find_kotlin_files_for_symbol(str(repo), "src/a/main.kt", "Widget")
    assert found == ["src/a/Widget.kt", "src/b/Widget.kt"]


def test_find_kotlin_files_for_symbol_empty_and_no_match(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    assert cc._find_kotlin_files_for_symbol(str(repo), "main.kt", "") == []
    assert cc._find_kotlin_files_for_symbol(str(repo), "main.kt", "Nope") == []


# ---------- collect_related_files_shallow edge states ----------

def test_collect_no_target():
    sel, meta = cc.collect_related_files_shallow("", "app.py")
    assert sel == [] and meta["reason"] == "no_target"
    sel, meta = cc.collect_related_files_shallow("/tmp/repo", None)
    assert sel == [] and meta["reason"] == "no_target"


def test_collect_empty_target_after_normalize(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    sel, meta = cc.collect_related_files_shallow(str(repo), "/")
    assert sel == [] and meta["reason"] == "empty_target"


def test_collect_target_missing(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    sel, meta = cc.collect_related_files_shallow(str(repo), "nope.py")
    assert sel == [] and meta["reason"] == "target_missing"


def test_collect_unsupported_extension(tmp_path):
    repo = tmp_path / "repo"
    (repo).mkdir()
    (repo / "note.md").write_text("hi\n", encoding="utf-8")
    sel, meta = cc.collect_related_files_shallow(str(repo), "note.md")
    assert sel == ["note.md"]
    assert meta["kind"] == "other"
    assert meta["reason"] == "unsupported_target_ext"


def test_collect_read_error_degrades_gracefully(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo).mkdir()
    (repo / "app.py").write_text("x = 1\n", encoding="utf-8")

    def boom(*_a, **_k):
        raise OSError("disk read failed")

    monkeypatch.setattr(cc, "_read_text_best_effort", boom)
    sel, meta = cc.collect_related_files_shallow(str(repo), "app.py")
    assert sel == ["app.py"]
    assert meta["reason"] == "read_error"


def test_collect_py_selection_cap_break(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo).mkdir()
    (repo / "app.py").write_text("import helper\n", encoding="utf-8")
    (repo / "helper.py").write_text("h = 1\n", encoding="utf-8")
    monkeypatch.setattr(cc, "CTX_MAX_FILES", 1)
    sel, meta = cc.collect_related_files_shallow(str(repo), "app.py")
    # cap already reached with the target alone -> break before appending
    assert sel == ["app.py"]
    assert meta["reason"] == "ok"


def test_collect_kotlin_branch(tmp_path):
    repo = tmp_path / "repo"
    (repo).mkdir()
    (repo / "Main.kt").write_text("import com.example.Widget\n", encoding="utf-8")
    (repo / "Widget.kt").write_text("class Widget\n", encoding="utf-8")

    sel, meta = cc.collect_related_files_shallow(str(repo), "Main.kt")
    assert sel == ["Main.kt", "Widget.kt"]
    assert meta["kind"] == "kt"
    assert meta["candidates"] == ["Widget.kt"]
    assert meta["reason"] == "ok"


def test_collect_kotlin_selection_cap_break(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo).mkdir()
    (repo / "Main.kt").write_text("import com.example.Widget\n", encoding="utf-8")
    (repo / "Widget.kt").write_text("class Widget\n", encoding="utf-8")
    monkeypatch.setattr(cc, "CTX_MAX_FILES", 1)
    sel, meta = cc.collect_related_files_shallow(str(repo), "Main.kt")
    assert sel == ["Main.kt"]
    assert meta["reason"] == "ok"


# ---------- read_file_snippet_context edge states ----------

def test_snippet_missing_args(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    ctx, meta = cc.read_file_snippet_context("", "app.py", around_regex="^")
    assert ctx == "" and meta["reason"] == "missing_args"
    ctx, meta = cc.read_file_snippet_context(str(repo), "", around_regex="^")
    assert ctx == "" and meta["reason"] == "missing_args"


def test_snippet_read_error(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo).mkdir()
    (repo / "app.py").write_text("x = 1\n", encoding="utf-8")

    def boom(*_a, **_k):
        raise OSError("read failed")

    monkeypatch.setattr(cc, "_read_text_best_effort", boom)
    ctx, meta = cc.read_file_snippet_context(str(repo), "app.py", around_regex="^")
    assert ctx == "" and meta["reason"] == "read_error"


def test_snippet_invalid_regex_falls_back_to_start(tmp_path):
    repo = tmp_path / "repo"
    (repo).mkdir()
    (repo / "app.py").write_text("a = 1\nb = 2\n", encoding="utf-8")
    ctx, meta = cc.read_file_snippet_context(str(repo), "app.py", around_regex="[")
    assert meta["included"] is True
    assert meta["reason"] == "regex_not_found_fallback_to_start"
    assert "a = 1" in ctx


def test_snippet_regex_no_match_falls_back_to_start(tmp_path):
    repo = tmp_path / "repo"
    (repo).mkdir()
    (repo / "app.py").write_text("a = 1\nb = 2\n", encoding="utf-8")
    ctx, meta = cc.read_file_snippet_context(str(repo), "app.py", around_regex="zzz")
    assert meta["reason"] == "regex_not_found_fallback_to_start"
    assert "a = 1" in ctx


def test_snippet_body_byte_truncation_marker(tmp_path):
    repo = tmp_path / "repo"
    (repo).mkdir()
    lines = "\n".join('long_line = "%s"' % ("x" * 80) for _ in range(40))
    (repo / "app.py").write_text(lines + "\n", encoding="utf-8")
    ctx, meta = cc.read_file_snippet_context(
        str(repo), "app.py", around_regex="long_line", window_lines=120, max_bytes=64
    )
    assert meta["included"] is True
    assert "[...TRUNCATED...]" in ctx
    assert meta["truncated"] is True
    assert meta["bytes_total"] <= 128  # truncated body stays tiny
