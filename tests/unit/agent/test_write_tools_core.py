"""Unit tests for write_tools_core helper contracts.

Covers the P26-7 cleanup round: broad ``except Exception`` /
``contextlib.suppress(Exception)`` wrappers were removed (dead defensive
wrappers around first-party imports and internally-guarded calls).  These
tests pin the NEW fail-fast contract (internal errors propagate instead of
silently degrading) and the preserved graceful-degradation contracts
(parse failures, out-of-range anchors, unavailable tree-sitter).
"""
import pytest

from external_llm.agent.tool_handlers import write_tools_core as core
from external_llm.common.indent_utils import (
    _file_indent_unit_from_logical,
    detect_indent_char,
    reindent_to_match,
)

# ── tree-sitter direct import (was: try/except Exception stub block) ────────

def test_ts_names_are_direct_imports():
    """tree_sitter_utils imports only stdlib and guards the native library
    internally, so the old ImportError/Exception fallback stubs were dead —
    the module now imports directly with a constant flag."""
    assert core._HAS_TS is True
    assert callable(core._ts_find_all_symbols)
    assert callable(core._ts_language_available)
    assert isinstance(core._TS_LANG_MODULE_MAP, dict)
    # The availability probe itself never raises and returns a bool.
    assert isinstance(core._ts_language_available("python"), bool)


# ── _reindent_to_match (was: except Exception -> return input unchanged) ────

def test_reindent_to_match_identity_with_canonical():
    """Pure delegate: results are byte-identical to indent_utils.reindent_to_match."""
    cases = [
        ("    b = 1\n    c = 2", "a = 0"),
        ("def f():\n    pass", "def g():\n    pass"),
        ("        x = 1", "    y = 2"),
    ]
    for new, matched in cases:
        assert core._reindent_to_match(new, matched) == reindent_to_match(new, matched)


def test_reindent_to_match_fail_fast_on_internal_error(monkeypatch):
    """A bug in the canonical reindenter must PROPAGATE (fail-fast), not
    silently return the mis-indented input — the silent fallback produced
    exactly the syntax_invalid_after_edit class this helper exists to prevent."""

    def _boom(*args, **kwargs):
        raise RuntimeError("canonical reindenter bug")

    monkeypatch.setattr(core, "reindent_to_match", _boom)
    with pytest.raises(RuntimeError, match="canonical reindenter bug"):
        core._reindent_to_match("    x = 1", "a = 0")


# ── _detect_file_unit (was: suppress(Exception) around tokenizer path) ──────

def test_detect_file_unit_identity_with_canonical_helpers():
    """Garbled content still degrades gracefully — the tokenizer path guards
    TokenError/IndentationError/SyntaxError internally, so the outer
    suppress(Exception) was redundant."""
    samples = ["x = 1\n", "x = '\n", "   \n\t\n", "def f(:\n    pass\n"]
    for src in samples:
        expected = _file_indent_unit_from_logical(
            src, detect_indent_char(src.split("\n"))
        ) or None
        assert core._detect_file_unit(src) == expected


def test_detect_file_unit_fail_fast_on_internal_error(monkeypatch):
    """Errors OUTSIDE the tokenizer's known failure modes (TokenError &
    friends are handled inside indent_utils) must propagate."""

    def _boom(*args, **kwargs):
        raise RuntimeError("unit detection bug")

    monkeypatch.setattr(core, "_file_indent_unit_from_logical", _boom)
    with pytest.raises(RuntimeError, match="unit detection bug"):
        core._detect_file_unit("x = 1\n")


# ── _detect_fragment_duplication (was: suppress(Exception) around all) ──────

def test_detect_fragment_duplication_none_for_fresh_snippet():
    """Non-duplicating snippet -> None (the pre-guard must never block)."""
    file_lines = ["a = 1\n", "b = 2\n", "c = 3\n"]
    snippet = "def brand_new():\n    return 42\n"
    assert core._detect_fragment_duplication(file_lines, 1, snippet) is None


def test_detect_fragment_duplication_detects_overlap():
    """Overlapping snippet -> diagnostic dict (contract preserved)."""
    file_lines = ["a = 1\n", "b = 2\n", "c = 3\n"]
    snippet = "b = 2\nc = 3\nd = 4\n"
    out = core._detect_fragment_duplication(file_lines, 1, snippet)
    assert out is not None
    assert out["ratio"] > 0


# ── _detect_enclosing_scope (was: suppress(Exception) around all) ───────────

def test_detect_enclosing_scope_out_of_range_anchor_returns_defaults():
    """Out-of-range anchor degrades to the default dict (bounds check hoisted)."""
    out = core._detect_enclosing_scope(["a = 1"], 5)
    assert out == {
        "innermost": (None, None, None),
        "top_level": (None, None, None),
        "anchor_indent": 0,
    }


def test_detect_enclosing_scope_finds_innermost_and_top_level():
    file_lines = ["class A:", "    def m(self):", "        pass"]
    out = core._detect_enclosing_scope(file_lines, 1)
    assert out["innermost"] == ("function", "m", 4)
    assert out["top_level"] == ("class", "A", 0)
    assert out["anchor_indent"] == 4


# ── _check_block_introducer_nesting (was: except Exception -> return None) ───

def test_block_introducer_nesting_parse_failure_returns_none():
    """Unparseable content is a DOCUMENTED None path (SyntaxError caught at
    parse time) — separate from the removed walk-failure wrapper."""
    assert core._check_block_introducer_nesting("def f(:\n", 0, 1) is None


def test_block_introducer_nesting_catches_nested_in_function():
    new_content = "def outer():\n    def inner():\n        pass\n"
    err = core._check_block_introducer_nesting(new_content, 1, 2)
    assert err is not None
    assert "landed inside function" in err


def test_block_introducer_nesting_clean_insert_returns_none():
    new_content = "def f():\n    pass\nx = 1\n"
    assert core._check_block_introducer_nesting(new_content, 0, 2) is None


# ── _find_block_end_line: forced-fallback flag contract ─────────────────────

def test_find_block_end_line_ts_forced_off_uses_indent_fallback(monkeypatch):
    """_HAS_TS stays a patchable module flag so the non-TS strategies remain
    reachable (same pattern as symbol_search / code_structure_utils)."""
    monkeypatch.setattr(core, "_HAS_TS", False)
    lines = ["def f():", "    pass", "x = 1"]
    end = core._find_block_end_line("def f():\n    pass\nx = 1\n", "python", 0, lines)
    assert end == 1


# ── _find_block_end_line: header-shape x language matrix ──────────────────
# Regression matrix for the pre-fix order bug: the cheap per-language header
# filters ran BEFORE tree-sitter and returned None for common header shapes
# (Allman braces, multi-line signatures, trailing comments), silently nesting
# insert_after into the body. Every cell is asserted in BOTH modes:
# tree-sitter on (real grammars — Strategy 1) and forced off (_HAS_TS=False —
# the brace/indent fallbacks). Fallback cells whose shape only tree-sitter
# can recognise (Allman `{` on the next line, multi-line signatures,
# trailing comments) expect None — grammar-less fallbacks are best-effort.

_BRACE_KNR_CONTENTS = {
    "c": "void f() {\n    int x;\n}\nint y;\n",
    "cpp": "void f() {\n    int x;\n}\nint y;\n",
    "c_sharp": "void F() {\n    int x;\n}\nint y;\n",
    "java": "void f() {\n    int x;\n}\nint y;\n",
    "go": "func f() {\n    x := 1\n}\nvar y int\n",
    "rust": "fn f() {\n    let x = 1;\n}\nlet y = 1;\n",
    "kotlin": "fun f() {\n    val x = 1\n}\nval y = 1\n",
    "typescript": "function f() {\n    let x = 1;\n}\nlet y = 1;\n",
}

@pytest.mark.parametrize("lang", sorted(_BRACE_KNR_CONTENTS))
def test_find_block_end_line_kandr_matrix(lang, monkeypatch):
    """K&R ``header {`` — both strategies must agree on the block end (2)."""
    content = _BRACE_KNR_CONTENTS[lang]
    lines = content.splitlines(True)
    for ts in (True, False):
        monkeypatch.setattr(core, "_HAS_TS", ts)
        end = core._find_block_end_line(content, lang, 0, lines)
        assert end == 2, f"{lang} K&R, _HAS_TS={ts}: got {end}, want 2"


_GO_ALLMAN_QUIRK_EXPECTED_NONE = (
    "go: the tree-sitter-go function_declaration node excludes the Allman "
    "body ({...} parses as a sibling block), so Strategy 1 sees a one-line "
    "symbol and the brace fallback needs the opener on the anchor line. "
    "gofmt enforces K&R braces, so Allman Go is non-idiomatic — pinned as "
    "the documented limitation."
)

@pytest.mark.parametrize("lang", sorted(_BRACE_KNR_CONTENTS))
def test_find_block_end_line_allman_matrix(lang, monkeypatch):
    """Allman ``header`` / ``{`` on the NEXT line — only tree-sitter can
    resolve it (the brace fallback needs the opener on the anchor line)."""
    content = _BRACE_KNR_CONTENTS[lang].replace(" {\n", "\n{\n", 1)
    lines = content.splitlines(True)
    monkeypatch.setattr(core, "_HAS_TS", True)
    end = core._find_block_end_line(content, lang, 0, lines)
    if lang == "go":
        assert end is None, _GO_ALLMAN_QUIRK_EXPECTED_NONE
    else:
        assert end == 3, f"{lang} Allman: got {end}, want 3"
    monkeypatch.setattr(core, "_HAS_TS", False)
    assert core._find_block_end_line(content, lang, 0, lines) is None


def test_find_block_end_line_python_basic_both_modes(monkeypatch):
    content = "def f():\n    pass\nx = 1\n"
    lines = content.splitlines(True)
    for ts in (True, False):
        monkeypatch.setattr(core, "_HAS_TS", ts)
        assert core._find_block_end_line(content, "python", 0, lines) == 1


@pytest.mark.parametrize(
    ("label", "content", "lang", "ts_on", "ts_off"),
    [
        # (label, content, lang, expected with TS, expected without)
        ("py header + trailing comment",
         "def f():  # noqa: D103\n    pass\nx = 1\n", "python", 1, None),
        ("py multi-line signature",
         "def f(\n    a,\n) -> int:\n    pass\nx = 1\n", "python", 3, None),
        ("py decorator anchor",
         "@dec\ndef f():\n    pass\nx = 1\n", "python", 2, None),
        ("c multi-line signature",
         "void f(\n    int x) {\n    int y;\n}\nint z;\n", "c", 3, None),
        ("kotlin string brace",
         'fun a(): String {\n    val s = "}"\n    return s\n}\nval t = 1\n',
         "kotlin", 3, 3),
        ("kotlin comment brace",
         "fun a(): String {  // }\n    val x = 1\n}\nval t = 1\n",
         "kotlin", 2, 2),
        ("c closing-brace else opener",
         "} else {\n    int x;\n}\nint y;\n", "c", 2, 2),
        ("c single-line closed block",
         "if (x) { f(); }\nint y;\n", "c", None, None),
        ("c plain line (no brace)",
         "int y;\nvoid g() {\n}\n", "c", None, None),
    ],
)
def test_find_block_end_line_special_shapes(label, content, lang, ts_on, ts_off, monkeypatch):
    """Each shape must match its documented expectation in both modes —
    the cells that differ pin exactly which shapes the grammar-less
    fallbacks cannot resolve (trailing comments, multi-line signatures,
    Allman) and which they now CAN (literals containing braces)."""
    lines = content.splitlines(True)
    monkeypatch.setattr(core, "_HAS_TS", True)
    assert core._find_block_end_line(content, lang, 0, lines) == ts_on, f"{label} (TS on)"
    monkeypatch.setattr(core, "_HAS_TS", False)
    assert core._find_block_end_line(content, lang, 0, lines) == ts_off, f"{label} (TS off)"


def test_find_block_end_line_brace_fallback_delegates_to_base_ssot(monkeypatch):
    """Strategy 2 must delegate to languages.base's literal-aware scanner —
    a hand-rolled count() loop miscounts braces inside literals (the Kotlin
    ``val s = "}"`` case) and truncates the block early."""
    monkeypatch.setattr(core, "_HAS_TS", False)
    calls: list[int] = []
    real = core.find_brace_block_end
    monkeypatch.setattr(
        core, "find_brace_block_end",
        lambda c, o: (calls.append(o), real(c, o))[1],
    )
    content = 'fun a(): String {\n    val s = "}"\n    return s\n}\nval t = 1\n'
    lines = content.splitlines(True)
    end = core._find_block_end_line(content, "kotlin", 0, lines)
    assert end == 3
    assert calls, "brace fallback did not delegate to find_brace_block_end"
