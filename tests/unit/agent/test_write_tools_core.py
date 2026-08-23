"""Unit tests for write_tools_core helper contracts.

Covers the P26-7 cleanup round: broad ``except Exception`` /
``contextlib.suppress(Exception)`` wrappers were removed (dead defensive
wrappers around first-party imports and internally-guarded calls).  These
tests pin the NEW fail-fast contract (internal errors propagate instead of
silently degrading) and the preserved graceful-degradation contracts
(parse failures, out-of-range anchors, unavailable tree-sitter).
"""

import json

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
        expected = _file_indent_unit_from_logical(src, detect_indent_char(src.split("\n"))) or None
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
        ("py header + trailing comment", "def f():  # noqa: D103\n    pass\nx = 1\n", "python", 1, None),
        ("py multi-line signature", "def f(\n    a,\n) -> int:\n    pass\nx = 1\n", "python", 3, None),
        ("py decorator anchor", "@dec\ndef f():\n    pass\nx = 1\n", "python", 2, None),
        ("c multi-line signature", "void f(\n    int x) {\n    int y;\n}\nint z;\n", "c", 3, None),
        ("kotlin string brace", 'fun a(): String {\n    val s = "}"\n    return s\n}\nval t = 1\n', "kotlin", 3, 3),
        ("kotlin comment brace", "fun a(): String {  // }\n    val x = 1\n}\nval t = 1\n", "kotlin", 2, 2),
        ("c closing-brace else opener", "} else {\n    int x;\n}\nint y;\n", "c", 2, 2),
        ("c single-line closed block", "if (x) { f(); }\nint y;\n", "c", None, None),
        ("c plain line (no brace)", "int y;\nvoid g() {\n}\n", "c", None, None),
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
        core,
        "find_brace_block_end",
        lambda c, o: (calls.append(o), real(c, o))[1],
    )
    content = 'fun a(): String {\n    val s = "}"\n    return s\n}\nval t = 1\n'
    lines = content.splitlines(True)
    end = core._find_block_end_line(content, "kotlin", 0, lines)
    assert end == 3
    assert calls, "brace fallback did not delegate to find_brace_block_end"


# ── RED→GREEN gap coverage: edge branches not hit by the above ──────────────


def test_find_block_end_line_out_of_range_and_blank_anchor():
    """Out-of-range and blank anchors degrade to None (bounds/blank guards)."""
    assert core._find_block_end_line("x = 1\n", "python", -1, ["x = 1\n"]) is None
    assert core._find_block_end_line("x = 1\n", "python", 5, ["x = 1\n"]) is None
    assert core._find_block_end_line("", "python", 0, ["", "x = 1\n"]) is None
    assert core._find_block_end_line("", "python", 0, ["   \n"]) is None


def test_find_block_end_line_python_fallback_blank_line_and_eof(monkeypatch):
    """Indent fallback: blank interior lines are skipped; EOF-terminated
    blocks return the last line index."""
    monkeypatch.setattr(core, "_HAS_TS", False)
    content = "def f():\n    x = 1\n\n    y = 2\nz = 3\n"
    lines = content.splitlines(True)
    assert core._find_block_end_line(content, "python", 0, lines) == 3
    content2 = "def f():\n    x = 1\n"
    lines2 = content2.splitlines(True)
    assert core._find_block_end_line(content2, "python", 0, lines2) == 1


def test_detect_file_unit_empty_content_returns_none():
    """Empty content is undetectable → None (documented degrade)."""
    assert core._detect_file_unit("") is None


def test_fragment_dup_snippet_comment_and_trivial_lines():
    """Snippet lines that are code+comment (289), comment-only (288), blank
    (283) or trivial (291) are excluded from the duplication judgement."""
    file_lines = ["a = 1\n", "b = 2\n", "c = 3\n"]
    snippet = "x = 9  # trailing note\nreturn\n\n# pure comment\nz = 8\n"
    out = core._detect_fragment_duplication(file_lines, 1, snippet)
    # non-trivial unique lines: x = 9, z = 8 → below the 3-line minimum → None
    assert out is None


def test_fragment_dup_file_window_comment_blank_and_trailing():
    """Window lines that are blank (303), comment-only (306-307) or carry a
    trailing comment (305, 308) are normalised identically to snippet lines."""
    file_lines = ["a = 1\n", "\n", "# comment\n", "b = 2  # trailing\n", "c = 3\n"]
    snippet = "b = 2\nc = 3\nd = 4\n"
    out = core._detect_fragment_duplication(file_lines, 2, snippet)
    assert out is not None
    assert out["ratio"] >= 0.5


def test_detect_enclosing_scope_skips_blank_lines():
    """Blank lines between the anchor and the enclosing header are skipped."""
    file_lines = ["def f():", "", "    pass"]
    out = core._detect_enclosing_scope(file_lines, 2)
    assert out["innermost"] == ("function", "f", 0)
    assert out["top_level"] == ("function", "f", 0)


class TestExtractTruncatedOpPathGaps:
    """RED→GREEN: the truncation scanner's remaining edge branches."""

    def test_truncated_mid_path_value_marked_partial(self):
        raw = '{"ops":[{"op":"create_file","path":"src/fo'
        hint = core._extract_truncated_op_path(raw)
        assert hint == "op=create_file path=src/fo…"

    def test_whitespace_between_key_and_colon(self):
        raw = '{"op"    :"create_file"}'
        hint = core._extract_truncated_op_path(raw)
        assert hint == "op=create_file"

    def test_path_without_any_op(self):
        raw = '{"path":"foo.py"}'
        hint = core._extract_truncated_op_path(raw)
        assert hint == "path=foo.py"

    def test_op_with_empty_path_value(self):
        raw = '{"op":"create_file","path":""}'
        hint = core._extract_truncated_op_path(raw)
        assert hint == "op=create_file"


class TestRepairPlanJsonGaps:
    """RED→GREEN: remaining repair branches — markdown fences, raw newlines/
    CRs inside values, escape sequences, and single-quoted values."""

    def test_markdown_fence_extraction(self):
        text = '```json\n{"a": 1, "b": [1, 2]}\n```'
        out = core._repair_plan_json(text)
        assert json.loads(out) == {"a": 1, "b": [1, 2]}

    def test_raw_newline_inside_string_value_escaped(self):
        text = '{"msg": "line1\nline2", "n": 1}'
        out = core._repair_plan_json(text)
        assert json.loads(out) == {"msg": "line1\nline2", "n": 1}

    def test_raw_cr_inside_string_value_escaped(self):
        text = '{"msg": "a\rb", "n": 1}'
        out = core._repair_plan_json(text)
        assert json.loads(out) == {"msg": "a\rb", "n": 1}

    def test_escaped_sequence_inside_string_preserved(self):
        text = '{"a": "x\\ny", "b": 2}'
        out = core._repair_plan_json(text)
        assert json.loads(out) == {"a": "x\ny", "b": 2}

    def test_single_quoted_values_with_escapes(self):
        text = "{'a': 'x\\ny', 'b': 2}"
        out = core._repair_plan_json(text)
        assert json.loads(out) == {"a": "x\ny", "b": 2}

    def test_single_quoted_value_containing_double_quotes(self):
        text = "{'a': 'say \"hi\"', 'b': 1}"
        out = core._repair_plan_json(text)
        assert json.loads(out) == {"a": 'say "hi"', "b": 1}


# ── RED→GREEN gap coverage: final edge branches (round 32-7) ────────────────


def test_find_block_end_line_unknown_language_returns_none():
    """A language outside the python/brace families degrades to None
    (grammar-less fallback contract: nothing to scan)."""
    assert core._find_block_end_line("x = 1\n", "text", 0, ["x = 1\n"]) is None
    assert core._find_block_end_line("x = 1\n", "markdown", 0, ["x = 1\n"]) is None


def test_leading_indent_width_blank_text_returns_zero():
    """Empty / whitespace-only text has no first content line → width 0."""
    assert core._leading_indent_width("") == 0
    assert core._leading_indent_width("  \n\t\n") == 0


def test_leading_indent_width_first_content_line_wins():
    """Width is taken from the FIRST non-blank line; tabs count as width 1."""
    assert core._leading_indent_width("    x = 1\n") == 4
    assert core._leading_indent_width("\n\tx = 1\n") == 1


def test_fragment_dup_file_window_trivial_lines_skipped():
    """Trivial window lines (return / } / …) carry no structural identity and
    are excluded from the comparison set (L309-310)."""
    file_lines = ["a = 1\n", "return\n", "}\n", "b = 2\n"]
    snippet = "x = 9\ny = 8\nz = 7\n"
    out = core._detect_fragment_duplication(file_lines, 1, snippet)
    assert out is None  # nothing matched — no duplication reported


def test_fragment_dup_ratio_below_threshold_returns_none():
    """≥3 unique snippet lines but overlap below the 0.5 ratio → None
    (the dict-return guard at L326 is not reached)."""
    file_lines = ["a = 1\n", "b = 2\n", "c = 3\n"]
    snippet = "a = 1\nx = 9\ny = 8\n"  # 1/3 matched = 0.33 < 0.5
    assert core._detect_fragment_duplication(file_lines, 0, snippet) is None


def test_block_introducer_nesting_catches_swallowed_trailing_code():
    """A def/class whose body extends past the inserted range swallowed
    pre-existing trailing code — the second violation kind (L485)."""
    new_content = "def f():\n    x = 1\n    y = 2\n"
    err = core._check_block_introducer_nesting(new_content, 0, 1)
    assert err is not None
    assert "swallowed pre-existing code" in err


class TestExtractTruncatedOpPathEscapes:
    """RED→GREEN: escape handling inside string values of the truncation
    scanner — a backslash must not be skipped, and an escaped quote must not
    terminate the literal (L537-539)."""

    def test_escaped_quote_inside_path_value(self):
        raw = '{"op":"create_file","path":"src/fo\\"o'
        hint = core._extract_truncated_op_path(raw)
        assert hint is not None
        assert hint.startswith("op=create_file")
        # The escaped quote does NOT terminate the literal — the scanner keeps
        # the raw text (it only tracks string state, it does not unescape).
        assert 'src/fo\\"o' in hint

    def test_backslash_escape_roundtrip_in_value(self):
        raw = '{"op":"create_file","path":"a\\\\b'
        hint = core._extract_truncated_op_path(raw)
        assert hint is not None
        assert "a\\\\b" in hint.replace("…", "")

    def test_nothing_identifiable_returns_none(self):
        """No op/path key and no snapshot before the cut point → None."""
        assert core._extract_truncated_op_path('{"foo": 1}') is None
        assert core._extract_truncated_op_path("") is None
