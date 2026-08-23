"""Unit tests for import_normalizer.py"""

import os
import tempfile
import textwrap

from external_llm.editor._editor_core.common.import_normalizer import (
    _collect_f821_protected_from_source,
    _f821_protected,
    collect_typing_usage,
    mark_f821_protected,
    normalize_typing_imports,
)

# ── collect_typing_usage ────────────────────────────────────────────────────


def test_collect_usage_basic():
    src = textwrap.dedent("""
        from typing import Dict, List
        def foo(x: Dict[str, int]) -> List[str]:
            return []
    """)
    used = collect_typing_usage(src)
    assert "Dict" in used
    assert "List" in used


def test_collect_usage_excludes_import_lines():
    # 'Optional' appears only in the import line, not in usage
    src = textwrap.dedent("""
        from typing import Optional
        def foo(x: int) -> int:
            return x
    """)
    used = collect_typing_usage(src)
    assert "Optional" not in used


def test_collect_usage_string_annotation():
    src = textwrap.dedent("""
        def foo(x: "Dict[str, Any]") -> "Optional[str]":
            return None
    """)
    used = collect_typing_usage(src)
    assert "Dict" in used
    assert "Any" in used
    assert "Optional" in used


def test_collect_usage_attribute_style():
    src = textwrap.dedent("""
        import typing
        def foo(x: typing.Dict[str, int]) -> typing.Optional[str]:
            return None
    """)
    used = collect_typing_usage(src)
    assert "Dict" in used
    assert "Optional" in used


def test_collect_usage_syntax_error_returns_empty():
    src = "def foo(: int:"
    used = collect_typing_usage(src)
    assert used == set()


def test_collect_usage_empty_file():
    used = collect_typing_usage("")
    assert used == set()


def test_collect_usage_no_typing():
    src = textwrap.dedent("""
        def foo(x: int) -> str:
            return str(x)
    """)
    used = collect_typing_usage(src)
    assert used == set()


# ── normalize_typing_imports ────────────────────────────────────────────────


def _write_temp(content: str, suffix=".py") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def test_normalize_adds_missing_import():
    src = textwrap.dedent("""\
        import os

        def foo(x: Dict[str, int]) -> List[str]:
            return []
    """)
    path = _write_temp(src)
    try:
        changed = normalize_typing_imports(path)
        assert changed is True
        with open(path) as f:
            result = f.read()
        assert "from typing import Dict, List" in result
    finally:
        os.unlink(path)


def test_normalize_removes_unused_imports():
    src = textwrap.dedent("""\
        from typing import Dict, List, Optional, Union
        import os

        def foo(x: Dict[str, int]) -> List[str]:
            return []
    """)
    path = _write_temp(src)
    try:
        changed = normalize_typing_imports(path)
        assert changed is True
        with open(path) as f:
            result = f.read()
        assert "Optional" not in result
        assert "Union" not in result
        assert "Dict" in result
        assert "List" in result
    finally:
        os.unlink(path)


def test_normalize_idempotent_when_correct():
    src = textwrap.dedent("""\
        from typing import Dict, List
        import os

        def foo(x: Dict[str, int]) -> List[str]:
            return []
    """)
    path = _write_temp(src)
    try:
        changed = normalize_typing_imports(path)
        assert changed is False
    finally:
        os.unlink(path)


def test_normalize_no_typing_usage_no_import():
    src = textwrap.dedent("""\
        import os

        def foo(x: int) -> str:
            return str(x)
    """)
    path = _write_temp(src)
    try:
        changed = normalize_typing_imports(path)
        assert changed is False
    finally:
        os.unlink(path)


def test_normalize_removes_all_when_no_usage():
    src = textwrap.dedent("""\
        from typing import Dict, Optional
        import os

        def foo(x: int) -> str:
            return str(x)
    """)
    path = _write_temp(src)
    try:
        changed = normalize_typing_imports(path)
        assert changed is True
        with open(path) as f:
            result = f.read()
        assert "from typing import" not in result
    finally:
        os.unlink(path)


def test_normalize_non_py_file_skipped():
    path = _write_temp("hello", suffix=".txt")
    try:
        changed = normalize_typing_imports(path)
        assert changed is False
    finally:
        os.unlink(path)


def test_normalize_multiline_import():
    src = textwrap.dedent("""\
        from typing import (
            Dict,
            List,
            Optional,
        )
        import os

        def foo(x: Dict[str, int]) -> List[str]:
            return []
    """)
    path = _write_temp(src)
    try:
        changed = normalize_typing_imports(path)
        assert changed is True
        with open(path) as f:
            result = f.read()
        # Optional should be removed, Dict+List should remain on one line
        assert "Optional" not in result
        assert "from typing import Dict, List" in result
    finally:
        os.unlink(path)


def test_normalize_preserves_ast_validity():
    src = textwrap.dedent("""\
        from typing import Any, Union
        import os

        def foo(x: Union[str, int]) -> Any:
            return x
    """)
    path = _write_temp(src)
    try:
        changed = normalize_typing_imports(path)
        assert changed is False  # already correct
    finally:
        os.unlink(path)


# ── f821-protected contract (tool_safety repair ↔ import_normalizer) ─────────
#
# Regression: the repair → normalizer contract was broken during the
# repair_core/repair_engine → tool_safety migration. These tests pin the
# reader/writer pair so it can never silently go inert again.


def test_collect_f821_protected_reads_marker_comment():
    """_collect_f821_protected_from_source reads markers from source text."""
    src = "from typing import Optional  # f821-protected\nx = 1\n"
    assert _collect_f821_protected_from_source(src) == {"Optional"}

    src2 = "from typing import Optional\nx = 1\n"  # no marker
    assert _collect_f821_protected_from_source(src2) == set()


def test_mark_f821_protected_writes_persistent_marker():
    """mark_f821_protected writes the marker into the file so it survives
    process restarts (the whole point of the on-disk marker design)."""
    src = "from typing import Optional\nx = Optional[int]\n"
    path = _write_temp(src)
    try:
        mark_f821_protected(path, "Optional")
        with open(path) as f:
            result = f.read()
        assert "# f821-protected" in result
        # Reader must see the marker we just wrote
        assert "Optional" in _collect_f821_protected_from_source(result)
    finally:
        os.unlink(path)


def test_normalize_preserves_f821_protected_unused_import():
    """The keystone contract: an F821-repaired typing import whose symbol is
    NOT visible to the AST pass must be preserved by the normalizer, not
    stripped. Without this, normalizer strips it -> F821 returns -> repair
    re-inserts -> infinite oscillation (the bug the migration introduced).
    """
    # Optional has no direct AST usage here, but the marker protects it.
    src = "from typing import Optional  # f821-protected\nx = 1\n"
    path = _write_temp(src)
    try:
        normalize_typing_imports(path)
        with open(path) as f:
            result = f.read()
        # Optional has no AST usage, but the marker protects it -> must survive
        assert "Optional" in result, (
            "f821-protected import must survive the normalizer; stripping it "
            "recreates the F821 the repair just fixed (oscillation)."
        )
        assert "# f821-protected" in result
    finally:
        os.unlink(path)


# ── RED→GREEN: uncovered branches ────────────────────────────────────────────


def test_collect_usage_vararg_kwarg_string_annotations():
    """*args/**kwargs string annotations are collected (L174/L176)."""
    src = textwrap.dedent("""
        def foo(*args: "Dict[str, int]", **kwargs: "List[int]") -> None:
            pass
    """)
    used = collect_typing_usage(src)
    assert "Dict" in used
    assert "List" in used


def test_collect_usage_annassign_string_annotation():
    """x: "Optional[int]" = None collects Optional (L182-183)."""
    src = 'x: "Optional[int]" = None\n'
    used = collect_typing_usage(src)
    assert "Optional" in used


def test_collect_usage_string_annotation_boundary_rescan():
    """A non-boundary match ("MyDict" contains "Dict") rescans and misses
    instead of false-positiving (L216)."""
    src = 'x: "MyDict" = None\n'
    used = collect_typing_usage(src)
    assert "Dict" not in used


def test_collect_f821_protected_ignores_other_comments():
    """Non-f821 markers on typing imports are skipped, not treated as
    protection (L60)."""
    src = "from typing import Optional  # noqa: F401\nx = 1\n"
    assert _collect_f821_protected_from_source(src) == set()


def test_mark_f821_protected_missing_file_is_noop():
    """mark_f821_protected on a missing file is a no-op: in-memory cache is
    still updated, nothing raises (L91-93)."""
    path = os.path.join(tempfile.mkdtemp(), "does_not_exist.py")
    mark_f821_protected(path, "Optional")
    assert "Optional" in _f821_protected.get(os.path.abspath(path), set())


def test_mark_f821_protected_write_failure_logs(monkeypatch, caplog):
    """Persistence failure is logged and the in-memory cache still wins
    (L107-108)."""
    import logging

    def _boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(
        "external_llm.editor._editor_core.common.import_normalizer.atomic_write_text",
        _boom,
    )
    src = "from typing import Optional\nx = Optional[int]\n"
    path = _write_temp(src)
    try:
        with caplog.at_level(
            logging.WARNING,
            logger="external_llm.editor._editor_core.common.import_normalizer",
        ):
            mark_f821_protected(path, "Optional")
        assert any("failed to persist marker" in r.message for r in caplog.records)
    finally:
        os.unlink(path)


def test_normalize_missing_file_returns_false():
    """A missing file is not an error, just a no-op (L237-238)."""
    path = os.path.join(tempfile.mkdtemp(), "missing.py")
    assert normalize_typing_imports(path) is False


def test_normalize_syntax_error_returns_false():
    """An unparseable file is skipped (L242-243)."""
    path = _write_temp("def foo(: int:")
    try:
        assert normalize_typing_imports(path) is False
    finally:
        os.unlink(path)


def test_normalize_non_typing_import_satisfies_usage():
    """Symbols already imported from a non-typing source (collections) are
    not re-added to the typing import (L261-262)."""
    src = textwrap.dedent("""\
        from collections import Mapping

        def f(x: Mapping[str, int]) -> None:
            pass
    """)
    path = _write_temp(src)
    try:
        assert normalize_typing_imports(path) is False
        with open(path) as fh:
            assert "from typing import" not in fh.read()
    finally:
        os.unlink(path)


def test_normalize_rewrite_parse_failure_skips_write(monkeypatch):
    """Post-rewrite validation failure (ast.parse #3) skips the write and
    returns False (L344-349)."""
    import ast as _ast

    real_parse = _ast.parse
    calls = {"n": 0}

    def _flaky_parse(source, *a, **k):
        calls["n"] += 1
        if calls["n"] == 3:  # normalize → collect_usage → rewrite validation
            raise SyntaxError("boom")
        return real_parse(source, *a, **k)

    monkeypatch.setattr(
        "external_llm.editor._editor_core.common.import_normalizer.ast.parse",
        _flaky_parse,
    )
    src = textwrap.dedent("""\
        from typing import Optional

        def f(x: int) -> int:
            return x
    """)
    path = _write_temp(src)
    try:
        assert normalize_typing_imports(path) is False
    finally:
        os.unlink(path)


def test_normalize_write_failure_returns_false(monkeypatch):
    """A failed atomic write returns False and keeps the original file
    (L360-362)."""

    def _boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(
        "external_llm.editor._editor_core.common.import_normalizer.atomic_write_text",
        _boom,
    )
    src = textwrap.dedent("""\
        def f(x: Dict[str, int]) -> None:
            pass
    """)
    path = _write_temp(src)
    try:
        assert normalize_typing_imports(path) is False
    finally:
        os.unlink(path)


def test_normalize_inserts_import_in_import_free_file():
    """A typing-using file with NO imports at all gets the import inserted at
    line 0 (_find_first_import_line's fallback, L372)."""
    src = textwrap.dedent("""\
        def f(x: Dict[str, int]) -> None:
            pass
    """)
    path = _write_temp(src)
    try:
        assert normalize_typing_imports(path) is True
        with open(path) as fh:
            result = fh.read()
        assert result.startswith("from typing import Dict")
    finally:
        os.unlink(path)
