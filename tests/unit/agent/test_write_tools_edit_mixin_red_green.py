"""RED→GREEN coverage tests for write_tools_edit_mixin.py (85% → 100%).

Covers the remaining edge branches: anchor-resolution fallbacks, edit_file
warnings/error paths, near-match/ast-fail/suggestion hints, scoped and
replace_all fallback splices, edit_text batch/scope validation + syntax-gate
diagnoses, create_file gates, and modify_symbol dry-run failures.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import ClassVar

import pytest

from external_llm.agent.tool_handlers.write_tools_edit_mixin import (
    WriteToolsEditMixin,
)

# ── _resolve_edit_anchor fallbacks ──────────────────────────────────────────


def test_resolve_anchor_single_stripped_match():
    _pos, actual, ratio = WriteToolsEditMixin._resolve_edit_anchor(
        None,
        "x = 1\ny = 2\n",
        "  x = 1",
        None,
    )
    assert actual == "x = 1"
    assert ratio == 0.0


def test_resolve_anchor_multiple_stripped_with_line_hint():
    content = "x = 1\ny = 2\nx = 1\n"
    # line=0 is out of range, so the line-hint shortcut (step 2) does NOT fire —
    # the stripped multi-match branch (with the out-of-range hint) resolves it.
    pos, actual, _ratio = WriteToolsEditMixin._resolve_edit_anchor(
        None,
        content,
        "  x = 1",
        0,
    )
    assert actual == "x = 1"
    assert content[pos : pos + len(actual)] == "x = 1"


def test_resolve_anchor_multiple_stripped_no_hint_fails():
    with pytest.raises(ValueError, match="anchor matches 2 lines"):
        WriteToolsEditMixin._resolve_edit_anchor(None, "x = 1\ny = 2\nx = 1\n", "  x = 1", None)


def test_resolve_anchor_duplicated_3line_block_progressive_scan():
    """A duplicated 3-line block exercises the progressive fallback loop (its
    shorter prefixes are duplicated too, so no return fires)."""
    content = "  a\n  b\n  c\n  a\n  b\n  c\n"
    with pytest.raises(ValueError, match="anchor text not found"):
        WriteToolsEditMixin._resolve_edit_anchor(None, content, "a\nb\nc", None)


def test_resolve_anchor_multiline_past_eof_single_line_return():
    """Multi-line anchor whose reconstruction runs past EOF → fall back to the
    matched line's byte offset."""
    content = "x = 1\n"
    pos, actual, _ = WriteToolsEditMixin._resolve_edit_anchor(None, content, "x = 1\ny = 2", None)
    assert actual == "x = 1"
    assert pos == 0


# ── edit_file branches ──────────────────────────────────────────────────────


def test_edit_file_ops_recovery_from_raw(tool_registry):
    p = Path(tool_registry.repo_root) / "f.py"
    p.write_text("a = 1\n")
    # Truncated JSON (no closing brace): the recovery layer cannot parse it, so
    # the operations array is extracted by the handler's own bracket matcher.
    res = tool_registry._tool_edit_file(
        {
            "path": "f.py",
            "__raw_arguments": '{"path": "f.py", "operations": [{"type": "replace", "anchor": "a = 1", "content": "b = 2"}]',
        }
    )
    assert res.ok, res
    assert "b = 2" in p.read_text()


def test_edit_file_no_path_raw_hint(tool_registry):
    res = tool_registry._tool_edit_file({"__raw_arguments": '{"operations": [{"type": "replace"'})
    assert not res.ok
    assert "raw args" in res.error


def test_edit_file_read_failure(tool_registry):
    p = Path(tool_registry.repo_root) / "locked.txt"
    p.write_text("x\n")
    p.chmod(0)
    try:
        res = tool_registry.dispatch(
            "edit_file", {"path": "locked.txt", "operations": [{"type": "replace", "anchor": "x", "content": "y"}]}
        )
        assert not res.ok
        assert "Failed to read" in res.error
    finally:
        p.chmod(0o644)


def test_edit_file_content_anchor_ratio_warning(tool_registry):
    p = Path(tool_registry.repo_root) / "w.py"
    p.write_text("x = 1\n")
    res = tool_registry.dispatch(
        "edit_file",
        {
            "path": "w.py",
            "operations": [{"type": "replace", "anchor": "x = 1", "content": "x = 1\n" + ("y = 2\n" * 100)}],
        },
    )
    assert res.ok, res.error
    assert any("much larger than" in w for w in res.metadata.get("edit_warnings", []))


def test_edit_file_insert_after_eof_no_newline(tool_registry):
    p = Path(tool_registry.repo_root) / "eof.txt"
    p.write_text("last line")
    res = tool_registry.dispatch(
        "edit_file",
        {"path": "eof.txt", "operations": [{"type": "insert_after", "anchor": "last line", "content": "new line"}]},
    )
    assert res.ok, res.error
    assert p.read_text() == "last line\nnew line\n"


def test_edit_file_insert_before_idempotent(tool_registry):
    p = Path(tool_registry.repo_root) / "ib.txt"
    p.write_text("b = 2\n")
    ops = [
        {"type": "insert_before", "anchor": "b = 2", "content": "a = 1"},
        {"type": "insert_before", "anchor": "b = 2", "content": "a = 1"},
    ]
    res = tool_registry.dispatch("edit_file", {"path": "ib.txt", "operations": ops})
    assert res.ok, res.error
    assert p.read_text() == "a = 1\nb = 2\n"


def test_edit_file_write_failure(tool_registry, monkeypatch):
    import external_llm.agent.tool_handlers.write_tools_edit_mixin as m

    p = Path(tool_registry.repo_root) / "wf.txt"
    p.write_text("x\n")
    monkeypatch.setattr(m, "atomic_write_text", lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full")))
    res = tool_registry.dispatch(
        "edit_file", {"path": "wf.txt", "operations": [{"type": "replace", "anchor": "x", "content": "y"}]}
    )
    assert not res.ok
    assert "Failed to write" in res.error


def test_edit_file_replace_op_type_mismatch_warning(tool_registry):
    p = Path(tool_registry.repo_root) / "rm.txt"
    p.write_text("x = 1\n")
    res = tool_registry.dispatch(
        "edit_file",
        {"path": "rm.txt", "operations": [{"type": "replace", "anchor": "x = 1", "content": "x = 1\ny = 2"}]},
    )
    assert res.ok, res.error
    assert any(
        "op-type mismatch" in w or "structurally always removes" in w for w in res.metadata.get("edit_warnings", [])
    )


# ── hint helpers ────────────────────────────────────────────────────────────


def test_patch_failure_snippet_malformed_patch(tool_registry, monkeypatch):
    import external_llm.agent.tool_handlers.write_tools_edit_mixin as m

    def _boom(patch_text):
        raise ValueError("no files")

    monkeypatch.setattr(m, "extract_files_from_patch", _boom)
    assert tool_registry._patch_failure_snippet("bad patch", None) == ""


def test_near_match_hint_blank_old(tool_registry):
    assert tool_registry._near_match_hint("x = 1\n", "   \n") == ""


def test_near_match_hint_window_skipped_anchors_single_line(tool_registry):
    big_old = "\n".join(f"line {i} content" for i in range(250))
    content = "line 1 content\n"
    out = tool_registry._near_match_hint(content, big_old)
    assert "Closest match" in out


def test_near_match_hint_diff_truncated(tool_registry):
    # old_string beyond max_window_lines (200) skips the quadratic window scan
    # (SequenceMatcher autojunk=False is O(n²) on repeated text) and anchors on
    # the single best line — the 250-line region still overflows the diff cap.
    content = "\n".join(f"file line {i}" for i in range(300))
    old = "\n".join(f"other line {i}" for i in range(250))
    out = tool_registry._near_match_hint(content, old)
    assert "diff truncated" in out


def test_near_match_hint_exception_degrades(tool_registry, monkeypatch):
    import difflib

    def _boom(*a, **kw):
        raise ValueError("boom")

    monkeypatch.setattr(difflib, "get_close_matches", _boom)
    assert tool_registry._near_match_hint("x = 1\n", "y = 2\n") == ""


def test_suggest_missing_paths_exception_degrades(tool_registry, monkeypatch):
    import external_llm.agent.tool_handlers.write_tools_edit_mixin as m

    monkeypatch.setattr(m, "_repo_file_index", lambda root: (_ for _ in ()).throw(OSError("walk failed")))
    assert tool_registry._suggest_missing_paths("ghost.py") == ""


def test_ast_fail_hint_class_symbol_and_qualified(tool_registry):
    src = "class A:\n    def m(self):\n        pass\n"
    out = tool_registry._ast_fail_hint(src, [], "zz")
    assert "Did you mean" in out or "Defined here" in out


def test_ast_fail_hint_non_dict_op(tool_registry):
    src = "def f():\n    pass\n"
    out = tool_registry._ast_fail_hint(src, ["not-a-dict"], "f")
    assert out == ""


def test_ast_fail_hint_exception_degrades(tool_registry, monkeypatch):
    monkeypatch.setattr(tool_registry, "_near_match_hint", lambda *a, **kw: (_ for _ in ()).throw(TypeError("boom")))
    assert tool_registry._ast_fail_hint("def f():\n    pass\n", [{"type": "replace_expr", "old": "zz"}], "f") == ""


def test_edited_line_regions_exception_safe(tool_registry, monkeypatch):
    import difflib

    class _Boom:
        def __init__(self, *a, **kw):
            raise RuntimeError("diff broken")

    monkeypatch.setattr(difflib, "SequenceMatcher", _Boom)
    assert WriteToolsEditMixin._edited_line_regions(None, "a\n", "b\n", 1) == (True, [])


def test_indentation_hint_messages(tool_registry):
    content = "def f():\n    a = 1\n    b = 2\n    c = 3\n    d = 4\n        e = 5\nx = 1\n"
    out = tool_registry._indentation_hint(content, 6, "unexpected indent")
    assert "Reduce this line" in out

    out2 = tool_registry._indentation_hint(
        "a = 1\n        b = 2\n", 2, "unindent does not match any outer indentation level"
    )
    assert "Valid outer" in out2

    out3 = tool_registry._indentation_hint("def f():\n", 2, "expected an indented block")
    assert "must be indented deeper" in out3


def test_indentation_hint_empty_candidates(tool_registry):
    # unexpected indent with no shallower neighbors → ""
    out = tool_registry._indentation_hint("                a = 1\n                b = 2\n", 2, "unexpected indent")
    assert out == ""
    # unindent with no outer level → ""
    out2 = tool_registry._indentation_hint(
        "        a = 1\n    b = 2\n", 1, "unindent does not match any outer indentation level"
    )
    assert out2 == ""
    # expected block with no ':' opener → ""
    out3 = tool_registry._indentation_hint("x = 1\n", 2, "expected an indented block")
    assert out3 == ""


# ── scoped / replace_all fallback splices ───────────────────────────────────


def test_apply_scoped_replacement_fallback_splice(tool_registry):
    content = "a = 1\nb = 2   \n"
    out = WriteToolsEditMixin._apply_scoped_replacement(
        tool_registry,
        content,
        "f.py",
        "b = 2",
        "c = 3",
        (2, 2),
    )
    assert out["ok"], out
    assert out["new_content"] == "a = 1\nc = 3   \n"  # trailing whitespace of the matched line is preserved


def test_apply_scoped_replacement_no_match_anywhere(tool_registry):
    out = WriteToolsEditMixin._apply_scoped_replacement(
        tool_registry,
        "a = 1\n",
        "f.py",
        "zzz",
        "c = 3",
        (1, 1),
    )
    assert not out["ok"]
    assert out["metadata"]["failure_class"] == "search_string_mismatch"


def test_apply_one_edit_fallback_contexts_and_replace_all(tool_registry):
    content = "x = 1  \nx = 1\n"
    # Single edit with >1 fallback matches → disambiguation contexts.
    out = WriteToolsEditMixin._apply_one_edit_text(
        tool_registry,
        content,
        "f.py",
        "x = 1",
        "y = 2",
        False,
    )
    assert not out["ok"]
    assert "Found 2 occurrences" in out["error"]
    # replace_all with fallback matches → position-based splice of both.
    out2 = WriteToolsEditMixin._apply_one_edit_text(
        tool_registry,
        content,
        "f.py",
        "x = 1",
        "y = 2",
        True,
    )
    assert out2["ok"], out2
    assert out2["occurrences"] == 2
    assert out2["new_content"] == "y = 2  \ny = 2\n"


def test_apply_one_edit_replace_all_zero(tool_registry):
    out = WriteToolsEditMixin._apply_one_edit_text(
        tool_registry,
        "a = 1\n",
        "f.py",
        "zzz",
        "y",
        True,
    )
    assert not out["ok"]
    assert out["metadata"]["failure_class"] == "search_string_mismatch"


# ── edit_text validation / gates ────────────────────────────────────────────


def test_edit_text_scope_validation_errors(tool_registry):
    p = Path(tool_registry.repo_root) / "sv.py"
    p.write_text("a = 1\n")
    base = {"file_path": "sv.py", "old_string": "a = 1", "new_string": "b = 1"}
    # Direct call: the dispatch argument repairer coerces non-int scopes away.
    res = tool_registry._tool_edit_text({**base, "scope_start_line": "x", "scope_end_line": "1"})
    assert not res.ok and "must be integers" in res.error
    res2 = tool_registry._tool_edit_text({**base, "scope_start_line": 0, "scope_end_line": 2})
    assert not res2.ok and ">= 1" in res2.error


def test_edit_text_batch_validation_errors(tool_registry):
    p = Path(tool_registry.repo_root) / "bv.py"
    p.write_text("a = 1\n")
    res = tool_registry.dispatch("edit_text", {"file_path": "bv.py", "edits": [123]})
    assert not res.ok and "must be an object" in res.error
    res2 = tool_registry.dispatch("edit_text", {"file_path": "bv.py", "edits": [{"new_string": "b"}]})
    assert not res2.ok and "missing old_string" in res2.error
    res3 = tool_registry.dispatch("edit_text", {"file_path": "bv.py", "edits": [{"old_string": "a = 1"}]})
    assert not res3.ok and "missing new_string" in res3.error
    res4 = tool_registry.dispatch(
        "edit_text",
        {
            "file_path": "bv.py",
            "edits": [{"old_string": "a", "new_string": "b", "scope_start_line": "z", "scope_end_line": 1}],
        },
    )
    assert not res4.ok and "must be integers" in res4.error


def test_edit_text_cascade_and_structure_diagnosis(tool_registry):
    """An indent shift whose cascade lands two lines past the edit (behind a
    comment, which the indentation scanner skips) surfaces the CASCADE
    diagnosis — the parser's reported line is OUTSIDE the edited region."""
    p = Path(tool_registry.repo_root) / "cascade.py"
    p.write_text("def f():\n    if True:\n        a = 1\n        b = 2\n    # comment\n    return 0\n")
    res = tool_registry.dispatch(
        "edit_text",
        {
            "file_path": "cascade.py",
            "old_string": "    if True:\n        a = 1\n        b = 2",
            "new_string": "        if True:\n            a = 1\n            b = 2",
        },
    )
    assert not res.ok
    assert "NOT directly edited" in res.error
    assert res.metadata["failure_class"] == "syntax_invalid_after_edit"
    assert res.metadata["error_in_edited_region"] is False


def test_edit_text_non_python_gate_refusal_and_soft_fail(tool_registry, monkeypatch):
    from external_llm.languages import LanguageRegistry

    p = Path(tool_registry.repo_root) / "gate.ts"
    p.write_text("let x = 1;\n")

    class _Err:
        def __init__(self, file, line, col, message):
            self.file, self.line, self.col, self.message = file, line, col, message

    class _FakeVal:
        ok = False
        errors: ClassVar[list] = [
            _Err("gate.ts", 1, 5, "syntax error: unexpected token"),
            _Err("gate.ts", 1, 6, "second error"),
            _Err("gate.ts", 1, 7, "third error"),
            _Err("gate.ts", 1, 8, "fourth error"),
        ]

    class _FakeProv:
        def capabilities(self):
            return types.SimpleNamespace(has_syntax_validator=True)

        def language_id(self):
            return types.SimpleNamespace(value="typescript")

        def validate_syntax(self, path, content):
            # Calls: 1 = ORIGINAL, 2 = NEW, 3 = origin re-check inside
            # _should_soft_fail_verify. Only the NEW content (call 2) is broken.
            calls[0] += 1
            return _FakeVal() if calls[0] == 2 else types.SimpleNamespace(ok=True, errors=[])

    calls = [0]
    real_get = LanguageRegistry.instance().get
    monkeypatch.setattr(LanguageRegistry.instance(), "get", lambda path: _FakeProv())
    try:
        res = tool_registry.dispatch(
            "edit_text",
            {"file_path": "gate.ts", "old_string": "let x = 1;", "new_string": "let x = ("},
        )
        assert not res.ok
        assert "refused" in res.error
        assert "+1 more syntax errors" in res.error
    finally:
        monkeypatch.setattr(LanguageRegistry.instance(), "get", real_get)


def test_edit_text_non_python_gate_provider_exceptions(tool_registry, monkeypatch):
    from external_llm.languages import LanguageRegistry

    p = Path(tool_registry.repo_root) / "gate2.ts"
    p.write_text("let x = 1;\n")

    class _BrokenProv:
        def capabilities(self):
            return types.SimpleNamespace(has_syntax_validator=True)

        def validate_syntax(self, path, content):
            raise RuntimeError("validator crashed")

    monkeypatch.setattr(LanguageRegistry.instance(), "get", lambda path: _BrokenProv())
    # orig validation raises → treated as OK; new raises → None → write proceeds.
    res = tool_registry.dispatch(
        "edit_text",
        {"file_path": "gate2.ts", "old_string": "let x = 1;", "new_string": "let y = 2;"},
    )
    assert res.ok, res.error


def test_edit_text_write_failure(tool_registry, monkeypatch):
    import external_llm.agent.tool_handlers.write_tools_edit_mixin as m

    p = Path(tool_registry.repo_root) / "wfail.py"
    p.write_text("a = 1\n")
    monkeypatch.setattr(m, "atomic_write_bytes", lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full")))
    res = tool_registry.dispatch("edit_text", {"file_path": "wfail.py", "old_string": "a = 1", "new_string": "b = 1"})
    assert not res.ok
    assert "Failed to write" in res.error


# ── create_file gates ───────────────────────────────────────────────────────


def test_create_file_non_str_content_and_raw_hint(tool_registry):
    res = tool_registry._tool_create_file({"path": "num.txt", "content": 123})
    assert res.ok, res.error
    assert (Path(tool_registry.repo_root) / "num.txt").read_text() == "123"
    res2 = tool_registry._tool_create_file({"__raw_arguments": '{"content": "truncated'})
    assert not res2.ok
    assert "raw args" in res2.error


def test_create_file_overwrite_unreadable_orig(tool_registry):
    p = Path(tool_registry.repo_root) / "locked2.txt"
    p.write_text("old\n")
    p.chmod(0)
    try:
        res = tool_registry._tool_create_file({"path": "locked2.txt", "content": "new\n", "overwrite": True})
        assert res.ok, res.error
    finally:
        p.chmod(0o644)


def test_create_file_non_python_gate_paths(tool_registry, monkeypatch):
    from external_llm.languages import LanguageRegistry

    class _Err:
        def __init__(self, file, line, col, message):
            self.file, self.line, self.col, self.message = file, line, col, message

    class _FakeProv:
        def __init__(self, errors, ok=False):
            self._errors = errors
            self._ok = ok

        def capabilities(self):
            return types.SimpleNamespace(has_syntax_validator=True)

        def language_id(self):
            return types.SimpleNamespace(value="typescript")

        def validate_syntax(self, path, content):
            return types.SimpleNamespace(ok=self._ok, errors=self._errors)

    monkeypatch.setattr(
        LanguageRegistry.instance(),
        "get",
        lambda path: _FakeProv(
            [
                _Err("a.ts", 1, 2, "syntax error: bad"),
                _Err("a.ts", 1, 3, "second"),
                _Err("a.ts", 1, 4, "third"),
                _Err("a.ts", 1, 5, "fourth"),
            ],
            ok=False,
        ),
    )
    res = tool_registry._tool_create_file({"path": "a.ts", "content": "let x = ("})
    assert not res.ok
    assert "refused" in res.error
    assert "+1 more syntax errors" in res.error

    # No-errors case → generic detail.
    monkeypatch.setattr(LanguageRegistry.instance(), "get", lambda path: _FakeProv([], ok=False))
    monkeypatch.setattr(tool_registry, "_should_soft_fail_verify", lambda *a, **kw: False)
    res2 = tool_registry._tool_create_file({"path": "b.ts", "content": "let x = ("})
    assert not res2.ok
    assert "syntax error in" in res2.error


def test_create_file_soft_fail_writes(tool_registry, monkeypatch):
    from external_llm.languages import LanguageRegistry

    class _Err:
        def __init__(self, file, line, col, message):
            self.file, self.line, self.col, self.message = file, line, col, message

    class _SoftProv:
        def capabilities(self):
            return types.SimpleNamespace(has_syntax_validator=True)

        def language_id(self):
            return types.SimpleNamespace(value="typescript")

        def validate_syntax(self, path, content):
            return types.SimpleNamespace(ok=False, errors=[_Err("s.ts", 1, 2, "undefined name 'zzz'")])

    monkeypatch.setattr(LanguageRegistry.instance(), "get", lambda path: _SoftProv())
    # The failure classifier has no "typescript" mapping, so the soft-fail
    # verdict is forced at the registry boundary.
    monkeypatch.setattr(tool_registry, "_should_soft_fail_verify", lambda *a, **kw: True)
    res = tool_registry._tool_create_file({"path": "s.ts", "content": "let x = zzz;"})
    assert res.ok, res.error
    assert res.metadata.get("syntax_gate") == "soft_fail"


def test_create_file_write_failure(tool_registry, monkeypatch):
    import external_llm.agent.tool_handlers.write_tools_edit_mixin as m

    monkeypatch.setattr(m, "atomic_write_text", lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full")))
    res = tool_registry._tool_create_file({"path": "cf.txt", "content": "x\n"})
    assert not res.ok
    assert "Failed to create" in res.error


def test_try_repair_truncated_json_delegates(tool_registry):
    out = WriteToolsEditMixin._try_repair_truncated_json('{"a": 1')
    assert out is None or out.get("a") == 1


# ── syntax check / modify_symbol ────────────────────────────────────────────


def test_run_syntax_check_no_provider(tool_registry):
    out = tool_registry._run_syntax_check_for_file("no_such_ext.zzz")
    assert out["skipped"] is True


def test_run_syntax_check_semantic_exception(tool_registry, monkeypatch):
    from external_llm.languages import LanguageRegistry

    p = Path(tool_registry.repo_root) / "sem.py"
    p.write_text("x = 1\n")

    class _FakeSem:
        ok = True
        errors: ClassVar[list] = []

    class _Prov:
        def capabilities(self):
            return types.SimpleNamespace(has_syntax_validator=True, has_semantic_validator=True)

        def validate_syntax(self, path, content):
            return types.SimpleNamespace(ok=True, errors=[], language=types.SimpleNamespace(value="python"))

        def validate_semantics(self, path):
            raise RuntimeError("checker crashed")

    monkeypatch.setattr(LanguageRegistry.instance(), "get", lambda path: _Prov())
    out = tool_registry._run_syntax_check_for_file("sem.py")
    assert out["ok"] is True
    assert "semantic_check_skipped" in out


def test_modify_symbol_dry_run_snapshot_failure(tool_registry):
    p = Path(tool_registry.repo_root) / "ms.py"
    p.write_text("def f():\n    pass\n")
    p.chmod(0)
    try:
        res = tool_registry.dispatch(
            "modify_symbol",
            {"file_path": "ms.py", "symbol": "f", "code": "def f():\n    return 1", "dry_run": True},
        )
        assert not res.ok
        assert "cannot snapshot" in res.error
    finally:
        p.chmod(0o644)


def test_modify_symbol_dry_run_restore_failure(tool_registry, monkeypatch):
    import external_llm.agent.tool_handlers.write_tools_edit_mixin as m
    from external_llm.agent import symbol_modify_tool

    p = Path(tool_registry.repo_root) / "ms2.py"
    p.write_text("def f():\n    pass\n")
    monkeypatch.setattr(
        symbol_modify_tool,
        "modify_symbol",
        lambda *a, **kw: (True, "+ def f():\n+     return 1", "def f():\n    return 1\n"),
    )
    monkeypatch.setattr(m, "atomic_write_text", lambda *a, **kw: (_ for _ in ()).throw(OSError("restore failed")))
    res = tool_registry.dispatch(
        "modify_symbol",
        {"file_path": "ms2.py", "symbol": "f", "code": "def f():\n    return 1", "dry_run": True},
    )
    assert not res.ok
    assert "restoring" in res.error
    assert res.metadata.get("restore_failed") is True


# ── P1: edit_text non-Python syntax-gate reuse ──────────────────────────────


def test_edit_text_non_python_gate_reuse_skips_post_spawn(tool_registry, monkeypatch):
    """A successful non-Python edit must NOT re-spawn validate_syntax after apply.

    The blocking gate runs validate_syntax TWICE in-memory: once on the
    ORIGINAL content (origin guard) and once on new_content. The pre-change
    code then ran a THIRD time via the post-apply check; now the post-check
    compares disk bytes against the gate-validated content and reuses the
    clean verdict, so validate_syntax runs exactly TWICE (gate only). The
    semantic check is a separate call and still runs.
    """
    from external_llm.languages import LanguageRegistry

    p = Path(tool_registry.repo_root) / "reuse.ts"
    p.write_text("let x = 1;\n")

    calls = {"syntax": 0, "semantic": 0}

    class _Prov:
        def capabilities(self):
            return types.SimpleNamespace(has_syntax_validator=True, has_semantic_validator=True)

        def language_id(self):
            return types.SimpleNamespace(value="typescript")

        def validate_syntax(self, path, content):
            calls["syntax"] += 1
            return types.SimpleNamespace(ok=True, errors=[], language=types.SimpleNamespace(value="typescript"))

        def validate_semantics(self, path):
            calls["semantic"] += 1
            return types.SimpleNamespace(ok=True, errors=[], checked=True)

    real_get = LanguageRegistry.instance().get
    monkeypatch.setattr(LanguageRegistry.instance(), "get", lambda path: _Prov())
    try:
        res = tool_registry.dispatch(
            "edit_text",
            {"file_path": "reuse.ts", "old_string": "let x = 1;", "new_string": "let y = 2;"},
        )
    finally:
        monkeypatch.setattr(LanguageRegistry.instance(), "get", real_get)

    assert res.ok, res.error
    assert calls["syntax"] == 2, f"validate_syntax should run twice (gate orig+new), got {calls['syntax']}"
    assert calls["semantic"] == 1, f"semantic check must still run, got {calls['semantic']}"
    assert res.metadata["syntax_check"]["ok"] is True
    assert (Path(tool_registry.repo_root) / "reuse.ts").read_text() == "let y = 2;\n"


def test_edit_text_non_python_gate_reuse_falls_back_on_disk_drift(tool_registry, monkeypatch):
    """When disk bytes drift from the gate-validated content, the post-apply
    check must NOT reuse the verdict — it re-runs validate_syntax (3 total:
    gate orig + gate new + post-apply).
    """
    from external_llm.languages import LanguageRegistry

    p = Path(tool_registry.repo_root) / "drift.ts"
    p.write_text("let x = 1;\n")

    calls = {"syntax": 0}

    class _Prov:
        def capabilities(self):
            return types.SimpleNamespace(has_syntax_validator=True, has_semantic_validator=False)

        def language_id(self):
            return types.SimpleNamespace(value="typescript")

        def validate_syntax(self, path, content):
            calls["syntax"] += 1
            return types.SimpleNamespace(ok=True, errors=[], language=types.SimpleNamespace(value="typescript"))

    real_get = LanguageRegistry.instance().get
    monkeypatch.setattr(LanguageRegistry.instance(), "get", lambda path: _Prov())
    import external_llm.agent.tool_handlers.write_tools_edit_mixin as m

    # Interpose after the gate validates but before the post-check reads disk:
    # make the on-disk content differ from new_content so the reuse condition
    # (gate_content == content) fails and a fresh syntax run is forced.
    orig_atomic = m.atomic_write_bytes

    def _write_with_drift(path, data):
        orig_atomic(path, data + b"\n// drift\n")

    monkeypatch.setattr(m, "atomic_write_bytes", _write_with_drift)
    try:
        res = tool_registry.dispatch(
            "edit_text",
            {"file_path": "drift.ts", "old_string": "let x = 1;", "new_string": "let y = 2;"},
        )
    finally:
        monkeypatch.setattr(m, "atomic_write_bytes", orig_atomic)
        monkeypatch.setattr(LanguageRegistry.instance(), "get", real_get)

    assert res.ok, res.error
    assert calls["syntax"] == 3, f"drift must force a fresh syntax run, got {calls['syntax']}"
