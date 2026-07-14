"""Unit tests for import_normalizer.py"""
import os
import tempfile
import textwrap

from external_llm.editor._editor_core.common.import_normalizer import (
    _collect_f821_protected_from_source,
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
