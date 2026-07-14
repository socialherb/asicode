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
    assert caller_search_extensions("src/app.py") == [".py"]
    assert caller_search_extensions("svc.go") == [".go"]
    assert caller_search_extensions("Main.java") == [".java"]
    assert caller_search_extensions("util.kt") == [".kt", ".kts"]
    assert caller_search_extensions("lib.rs") == [".rs"]
    assert caller_search_extensions("helper.rb") == [".rb"]


def test_js_ts_family_is_mutually_callable():
    # JS and TS are different LanguageIds but one callability family —
    # a .ts function CAN be called from .js/.jsx/.tsx (the bug the old
    # hardcoded *.py glob silently missed).
    js_ts = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}
    for f in ("a.ts", "b.tsx", "c.js", "d.jsx", "e.mjs", "f.cjs"):
        assert set(caller_search_extensions(f)) == js_ts, f


def test_unknown_family_falls_back_to_all_code_extensions():
    # PHP/C#/Swift/C++ are not in any family yet → broad fallback (union of all
    # known code-language extensions).  Strictly better than one hardcoded lang.
    expected = set()
    for g in _LANGUAGE_EXTENSION_GROUPS:
        expected |= g
    for f in ("legacy.php", "Net.cs", "core.cpp", "app.swift"):
        assert set(caller_search_extensions(f)) == expected, f


def test_none_def_file_uses_broad_fallback():
    expected = set()
    for g in _LANGUAGE_EXTENSION_GROUPS:
        expected |= g
    assert set(caller_search_extensions(None)) == expected


def test_case_insensitive_extension():
    assert caller_search_extensions("App.PY") == [".py"]
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


def test_unknown_extension_returns_minus_one():
    for ext in (".php", ".cs", ".cpp", ".swift", ".md", ".txt", ""):
        assert _get_language_group(ext) == -1, ext
