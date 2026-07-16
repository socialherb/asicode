"""Tests for language classification helpers in external_llm.languages.models.

Covers the "callability family" abstraction shared by cross-file caller search
(ripgrep glob derivation) and the cross-language resolution guard.  These are
pure functions over the single source of truth (_LANGUAGE_EXTENSION_GROUPS),
so the tests pin the contract both consumers rely on.
"""
from __future__ import annotations

from external_llm.languages.models import (
    _LANGUAGE_EXTENSION_GROUPS,
    _get_language_group,
    caller_search_extensions,
)

# ── caller_search_extensions ───────────────────────────────────────────────

def test_known_language_returns_own_family():
    # A definition is only callable from its own family.
    assert caller_search_extensions("src/app.py") == [".py", ".pyi"]
    assert caller_search_extensions("svc.go") == [".go"]
    assert caller_search_extensions("Main.java") == [".java"]
    assert caller_search_extensions("util.kt") == [".kt", ".kts"]
    assert caller_search_extensions("lib.rs") == [".rs"]
    assert caller_search_extensions("helper.rb") == [".rb"]
    assert caller_search_extensions("page.php") == [".php"]
    assert caller_search_extensions("net.cs") == [".cs"]
    assert caller_search_extensions("app.swift") == [".swift"]
    assert caller_search_extensions("build.scala") == [".sc", ".scala"]
    assert caller_search_extensions("script.sc") == [".sc", ".scala"]
    assert caller_search_extensions("config.lua") == [".lua"]
    assert caller_search_extensions("run.sh") == [".bash", ".sh"]
    assert caller_search_extensions("run.bash") == [".bash", ".sh"]


def test_python_family_includes_type_stubs():
    # .pyi stubs declare the same symbols as .py implementations, so a symbol
    # defined in a stub is callable from .py (and vice-versa).  Without .pyi in
    # the Python group, caller search for a stub falls back to the broad union
    # of ALL code extensions — the exact broad-fallback bug the family mechanism
    # exists to prevent.
    py_family = {".py", ".pyi"}
    assert set(caller_search_extensions("pkg/__init__.pyi")) == py_family
    assert set(caller_search_extensions("pkg/module.py")) == py_family
    assert _get_language_group(".pyi") == _get_language_group(".py") == 1


def test_js_ts_family_is_mutually_callable():
    # JS and TS are different LanguageIds but one callability family —
    # a .ts function CAN be called from .js/.jsx/.tsx (the bug the old
    # hardcoded *.py glob silently missed).
    js_ts = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts"}
    for f in ("a.ts", "b.tsx", "c.js", "d.jsx", "e.mjs", "f.cjs", "g.mts", "h.cts"):
        assert set(caller_search_extensions(f)) == js_ts, f


def test_c_cpp_family_is_mutually_callable():
    # C and C++ are different LanguageIds but one callability family — a .cpp
    # function CAN be called from .c/.h (extern "C"), and headers (.h/.hpp/.hh)
    # declare symbols consumed by both .c and .cpp translation units.
    c_cpp = {".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hh"}
    for f in ("main.c", "lib.h", "core.cpp", "impl.cc", "view.cxx",
              "hdr.hpp", "raw.hh"):
        assert set(caller_search_extensions(f)) == c_cpp, f


def test_unknown_family_falls_back_to_all_code_extensions():
    # JSON/CSS/HTML are not in any family → broad fallback (union of all known
    # code-language extensions).  Strictly better than one hardcoded lang.
    expected = set()
    for g in _LANGUAGE_EXTENSION_GROUPS:
        expected |= g
    for f in ("config.json", "style.css", "page.html"):
        assert set(caller_search_extensions(f)) == expected, f


def test_none_def_file_uses_broad_fallback():
    expected = set()
    for g in _LANGUAGE_EXTENSION_GROUPS:
        expected |= g
    assert set(caller_search_extensions(None)) == expected


def test_case_insensitive_extension():
    assert caller_search_extensions("App.PY") == [".py", ".pyi"]
    assert ".ts" in caller_search_extensions("App.TS")


def test_result_is_sorted_for_determinism():
    exts = caller_search_extensions("a.ts")
    assert exts == sorted(exts)
    assert caller_search_extensions(None) == sorted(caller_search_extensions(None))


# ── _get_language_group (SpecGraphEnricher contract) ───────────────────────

def test_group_indices_are_stable():
    # spec_graph_enricher compares group INDICES to detect cross-language
    # resolution; the indices must stay stable after the move to models.py.
    assert _get_language_group(".ts") == _get_language_group(".js") == 0   # JS/TS
    assert _get_language_group(".py") == 1
    assert _get_language_group(".go") == 2
    assert _get_language_group(".java") == 3
    assert _get_language_group(".kt") == _get_language_group(".kts") == 4
    assert _get_language_group(".rs") == 5
    assert _get_language_group(".rb") == 6
    assert _get_language_group(".c") == _get_language_group(".cpp") == 7  # C/C++
    assert _get_language_group(".php") == 8                              # PHP
    assert _get_language_group(".cs") == 9                               # C#
    assert _get_language_group(".swift") == 10                           # Swift
    assert _get_language_group(".scala") == _get_language_group(".sc") == 11  # Scala
    assert _get_language_group(".lua") == 12                             # Lua
    assert _get_language_group(".sh") == _get_language_group(".bash") == 13  # Bash


def test_unknown_extension_returns_minus_one():
    for ext in (".md", ".txt", ".json", ".css", ".html", ""):
        assert _get_language_group(ext) == -1, ext
