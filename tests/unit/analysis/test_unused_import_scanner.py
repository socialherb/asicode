"""Tests for external_llm/analysis/unused_import_scanner.py."""
from __future__ import annotations

import tempfile
import textwrap
from pathlib import Path

from external_llm.analysis.unused_import_scanner import (
    _TYPING_MODULE_SYMBOLS,
    UnusedImportCandidate,
    _collect_load_names,
    _extract_all_names,
    scan_unused_imports,
)


def _make_py_file(source: str) -> str:
    """Write source to a temp .py file and return its absolute path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8",
    )
    tmp.write(textwrap.dedent(source))
    tmp.close()
    return tmp.name


# ── Basic detection ──────────────────────────────────────────────────────


def test_unused_import_basic():
    """A simple unused import is detected."""
    src = _make_py_file("""\
        import os
        import sys

        def greet() -> str:
            return "hello"
    """)
    candidates = scan_unused_imports(repo_root="", file_paths=[src])
    symbols = {c.symbol_name for c in candidates}
    assert "os" in symbols
    assert "sys" in symbols
    Path(src).unlink()


def test_used_import_not_flagged():
    """An import that is actually used is not flagged."""
    src = _make_py_file("""\
        import os
        import sys

        def greet() -> str:
            return os.getcwd()
    """)
    candidates = scan_unused_imports(repo_root="", file_paths=[src])
    symbols = {c.symbol_name for c in candidates}
    assert "os" not in symbols  # used
    assert "sys" in symbols     # unused
    Path(src).unlink()


def test_from_import():
    """from X import Y pattern."""
    src = _make_py_file("""\
        from pathlib import Path
        from collections import OrderedDict

        p = Path("/tmp")
    """)
    candidates = scan_unused_imports(repo_root="", file_paths=[src])
    symbols = {c.symbol_name for c in candidates}
    assert "Path" not in symbols  # used
    assert "OrderedDict" in symbols
    Path(src).unlink()


def test_aliased_import():
    """import X as Y pattern."""
    src = _make_py_file("""\
        import numpy as np
        import pandas as pd

        data = np.array([1, 2, 3])
    """)
    candidates = scan_unused_imports(repo_root="", file_paths=[src])
    symbols = {c.symbol_name for c in candidates}
    assert "np" not in symbols  # used via alias
    assert "pd" in symbols      # unused alias
    Path(src).unlink()


def test_from_import_alias():
    """from X import Y as Z pattern."""
    src = _make_py_file("""\
        from datetime import datetime as dt
        from datetime import timedelta as td

        now = dt.now()
    """)
    candidates = scan_unused_imports(repo_root="", file_paths=[src])
    symbols = {c.symbol_name for c in candidates}
    assert "dt" not in symbols  # used via alias
    assert "td" in symbols      # unused alias
    Path(src).unlink()


# ── Exclusion zones ──────────────────────────────────────────────────────


def test_future_import_skipped():
    """from __future__ import annotations is never flagged."""
    src = _make_py_file("""\
        from __future__ import annotations
        import os

        def greet() -> str:
            return "hello"
    """)
    candidates = scan_unused_imports(repo_root="", file_paths=[src])
    symbols = {c.symbol_name for c in candidates}
    assert "annotations" not in symbols  # __future__ is skipped
    assert "os" in symbols               # still detected as unused
    Path(src).unlink()


def test_typing_symbol_skipped():
    """Typing symbols are skipped (import_normalizer domain)."""
    for sym in list(_TYPING_MODULE_SYMBOLS)[:3]:
        src = _make_py_file(f"""\
            from typing import {sym}

            def greet() -> str:
                return "hello"
        """)
        candidates = scan_unused_imports(repo_root="", file_paths=[src])
        assert not candidates, f"{sym} should be skipped"
        Path(src).unlink()


def test_explicit_typing_import_still_flagged_if_not_in_set():
    """A non-typing import from typing (cast) that is never used is still flagged
    because 'cast' is in the exclusion set — actually verify cast IS excluded."""
    # cast is in _TYPING_MODULE_SYMBOLS
    src = _make_py_file("""\
        from typing import cast

        x = 42
    """)
    candidates = scan_unused_imports(repo_root="", file_paths=[src])
    assert not candidates  # cast is in the exclusion set
    Path(src).unlink()


def test_all_reexport_skipped():
    """Names in __all__ are considered public API re-exports."""
    src = _make_py_file("""\
        import os
        import sys

        __all__ = ["os", "sys"]
    """)
    candidates = scan_unused_imports(repo_root="", file_paths=[src])
    assert not candidates, "__all__ re-exports should be skipped"
    Path(src).unlink()


def test_type_checking_guarded_imports_skipped():
    """Imports inside a TYPE_CHECKING guard are skipped; imports outside it
    are still scanned normally."""
    src = _make_py_file("""\
        import os
        import sys

        if TYPE_CHECKING:
            from collections import OrderedDict

        def greet() -> str:
            return "hello"
    """)
    candidates = scan_unused_imports(repo_root="", file_paths=[src])
    flagged = {c.symbol_name for c in candidates}
    assert "OrderedDict" not in flagged, "guarded import must not be flagged"
    assert {"os", "sys"} <= flagged, "unguarded unused imports are still flagged"
    Path(src).unlink()


def test_all_dynamic_reexport_skipped():
    """Dynamic __all__ (with Name references) causes conservative skip."""
    src = _make_py_file("""\
        import os

        __all__ = [os]

        def greet() -> str:
            return "hello"
    """)
    candidates = scan_unused_imports(repo_root="", file_paths=[src])
    assert not candidates
    Path(src).unlink()


# ── Edge cases ───────────────────────────────────────────────────────────


def test_multiple_from_same_module():
    """Some imports from a module used, others not."""
    src = _make_py_file("""\
        from os.path import join, exists, isfile

        p = join("/tmp", "x.txt")
    """)
    candidates = scan_unused_imports(repo_root="", file_paths=[src])
    symbols = {c.symbol_name for c in candidates}
    assert "join" not in symbols   # used
    # "exists" or "isfile" may appear as unused
    assert symbols, "at least one import should be flagged as unused"
    Path(src).unlink()


def test_max_per_file_enforced():
    """Respect max_per_file limit."""
    src = _make_py_file("""\
        import a
        import b
        import c
        import d
        import e
        import f
        import g
        import h
        import i
        import j
        import k
        import l

        USED = 1
    """)
    candidates = scan_unused_imports(repo_root="", file_paths=[src], max_per_file=5)
    assert len(candidates) == 5
    Path(src).unlink()


def test_non_py_file_skipped():
    """Non-.py files are skipped (pre-filtered by ScannerRegistry)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w")
    tmp.write("import os\n")
    tmp.close()
    from external_llm.agent.scanner_registry import get_registry
    reg = get_registry()
    result = reg.run("unused_import_scanner", file_paths=[tmp.name])
    assert not result.candidates_raw, f"Expected 0 candidates, got {len(result.candidates_raw)}"
    Path(tmp.name).unlink()


def test_missing_file_skipped():
    """Non-existent files are skipped without error."""
    candidates = scan_unused_imports(repo_root="", file_paths=["/nonexistent/file.py"])
    assert not candidates


def test_syntax_error_file_skipped():
    """Files with syntax errors are skipped."""
    src = _make_py_file("""\
        import os
        this is not valid python
    """)
    candidates = scan_unused_imports(repo_root="", file_paths=[src])
    assert not candidates
    Path(src).unlink()


def test_import_in_function_not_confused():
    """Local import inside a function body is still correctly tracked."""
    src = _make_py_file("""\
        def use_tempfile():
            import tempfile
            return tempfile.gettempdir()

        UNUSED = 1
    """)
    candidates = scan_unused_imports(repo_root="", file_paths=[src])
    # tempfile IS used (inside the function), so it should NOT be flagged
    symbols = {c.symbol_name for c in candidates}
    assert "tempfile" not in symbols
    Path(src).unlink()


def test_attribute_access_counts_as_usage():
    """obj.attr usage counts as reference to obj."""
    src = _make_py_file("""\
        import os

        result = os.path.join("a", "b")
    """)
    candidates = scan_unused_imports(repo_root="", file_paths=[src])
    assert not candidates, "os is used via attribute access"
    Path(src).unlink()


# ── Candidate structure ──────────────────────────────────────────────────


def test_candidate_to_dict():
    """UnusedImportCandidate.to_dict() returns correct fields."""
    cand = UnusedImportCandidate(
        file="test.py",
        symbol_name="os",
        lineno=1,
        import_line_text="import os",
    )
    d = cand.to_dict()
    assert d["file"] == "test.py"
    assert d["symbol_name"] == "os"
    assert d["lineno"] == 1
    assert d["import_line_text"] == "import os"
    assert d["kind"] == "unused_import"


# ── Internal helpers ─────────────────────────────────────────────────────


def test_extract_all_names_empty():
    """_extract_all_names returns empty set when no __all__."""
    tree = __import__("ast").parse("import os\n")
    assert _extract_all_names(tree) == set()


def test_extract_all_names_literal():
    """_extract_all_names finds string names in __all__."""
    tree = __import__("ast").parse('__all__ = ["os", "sys"]\n')
    names = _extract_all_names(tree)
    assert "os" in names
    assert "sys" in names


def test_extract_all_names_dynamic():
    """_extract_all_names detects dynamic __all__."""
    tree = __import__("ast").parse("__all__ = [x for x in names]\n")
    names = _extract_all_names(tree)
    assert "*__dynamic__*" in names


def test_collect_load_names_ignores_import_lines():
    """_collect_load_names excludes names from import lines."""
    import ast
    tree = ast.parse("import os\nimport sys\n")
    used = _collect_load_names(tree)
    assert "os" not in used
    assert "sys" not in used


def test_collect_load_names_finds_usage():
    """_collect_load_names finds Name nodes in Load context."""
    import ast
    tree = ast.parse("import os\nx = os.getcwd()\n")
    used = _collect_load_names(tree)
    assert "os" in used


def test_string_annotation_forward_ref_not_flagged():
    """An import used only in an explicit string annotation is not flagged."""
    source = textwrap.dedent("""\
        from collections.abc import Callable
        from typing import Optional
        class Resolver:
            def resolve(self) -> "Callable":
                return lambda: 42
        x: "Optional[Resolver]" = None
    """)
    path = _make_py_file(source)
    try:
        results = scan_unused_imports(repo_root="", file_paths=[path])
        flagged = {c.symbol_name for c in results}
        assert "Callable" not in flagged, (
            f"Callable should not be flagged — it is used in -> 'Callable', got {flagged}"
        )
        assert "Optional" not in flagged, (
            f"Optional should not be flagged — it is used in x: 'Optional[Resolver]', got {flagged}"
        )
    finally:
        Path(path).unlink(missing_ok=True)


def test_string_annotation_nested_in_generic_not_flagged():
    """A forward reference nested inside a generic (list["X"], dict[str, "Y"])
    is not flagged — the annotation AST is ast.Subscript, not ast.Constant, so
    the scanner must walk the subtree to find the inner string literal."""
    source = textwrap.dedent("""\
        from typing import Operation, Handler
        def build() -> list["Operation"]:
            return []
        def lookup() -> dict[str, "Handler"]:
            return {}
    """)
    path = _make_py_file(source)
    try:
        results = scan_unused_imports(repo_root="", file_paths=[path])
        flagged = {c.symbol_name for c in results}
        assert "Operation" not in flagged, (
            f"Operation should not be flagged — used in -> list['Operation'], got {flagged}"
        )
        assert "Handler" not in flagged, (
            f"Handler should not be flagged — used in -> dict[str, 'Handler'], got {flagged}"
        )
    finally:
        Path(path).unlink(missing_ok=True)


def test_cast_string_arg_not_flagged():
    """An import used only as the type arg of typing.cast("Foo", ...) is not
    flagged — the quoted type never becomes an ast.Name node."""
    source = textwrap.dedent("""\
        from typing import cast
        from external_llm.alpha import Alpha

        x = cast("Alpha", None)
    """)
    path = _make_py_file(source)
    try:
        results = scan_unused_imports(repo_root="", file_paths=[path])
        flagged = {c.symbol_name for c in results}
        assert "Alpha" not in flagged, (
            f"Alpha should not be flagged — used in cast('Alpha', ...), got {flagged}"
        )
    finally:
        Path(path).unlink(missing_ok=True)


def test_typevar_bound_string_not_flagged():
    """An import used only as TypeVar(..., bound="Foo") (or a positional
    constraint "Foo") is not flagged."""
    source = textwrap.dedent("""\
        from typing import TypeVar
        from external_llm.beta import Beta
        from external_llm.gamma import Gamma

        U = TypeVar("U", bound="Beta")
        V = TypeVar("V", "Gamma")
    """)
    path = _make_py_file(source)
    try:
        results = scan_unused_imports(repo_root="", file_paths=[path])
        flagged = {c.symbol_name for c in results}
        assert "Beta" not in flagged, (
            f"Beta should not be flagged — used as TypeVar bound, got {flagged}"
        )
        assert "Gamma" not in flagged, (
            f"Gamma should not be flagged — used as TypeVar constraint, got {flagged}"
        )
    finally:
        Path(path).unlink(missing_ok=True)


def test_unrelated_string_does_not_mask_unused():
    """A genuinely-unused import is still flagged even when a *different* string
    constant happens to contain its name — only cast/TypeVar call sites mask,
    not arbitrary strings like print()."""
    source = textwrap.dedent("""\
        from external_llm.delta import Delta

        print("Delta is unused here")
    """)
    path = _make_py_file(source)
    try:
        results = scan_unused_imports(repo_root="", file_paths=[path])
        flagged = {c.symbol_name for c in results}
        assert "Delta" in flagged, (
            f"Delta SHOULD be flagged — print('Delta ...') is not a type use, got {flagged}"
        )
    finally:
        Path(path).unlink(missing_ok=True)


def test_typealias_value_string_not_flagged():
    """``X: TypeAlias = "Foo"`` — the aliased type is a forward-ref in the value
    position, not the annotation.  The annotation is ``TypeAlias`` itself, so the
    annotation walker must also inspect the *value* or the import is falsely
    flagged as unused."""
    source = textwrap.dedent("""\
        from typing import TypeAlias

        from external_llm.agent.failure_pattern_store import FailurePatternStore

        MyAlias: TypeAlias = "FailurePatternStore"
    """)
    path = _make_py_file(source)
    try:
        results = scan_unused_imports(repo_root="", file_paths=[path])
        flagged = {c.symbol_name for c in results}
        assert "FailurePatternStore" not in flagged, (
            f"FailurePatternStore used as TypeAlias forward-ref must NOT be flagged, got {flagged}"
        )
    finally:
        Path(path).unlink(missing_ok=True)


def test_typealias_value_does_not_mask_unrelated_string_var():
    """A plain ``x: str = "Foo"`` is an ordinary string variable, NOT a type
    alias — so ``Foo`` (if imported and otherwise unused) must still be flagged.
    Only an explicit ``TypeAlias`` annotation treats the value as type-bearing."""
    source = textwrap.dedent("""\
        from external_llm.agent.failure_pattern_store import FailurePatternStore

        note: str = "FailurePatternStore is a class"
    """)
    path = _make_py_file(source)
    try:
        results = scan_unused_imports(repo_root="", file_paths=[path])
        flagged = {c.symbol_name for c in results}
        assert "FailurePatternStore" in flagged, (
            f"plain str var must NOT mask unused import, got {flagged}"
        )
    finally:
        Path(path).unlink(missing_ok=True)


def test_pep695_type_statement_string_not_flagged():
    """``type X = "Foo"`` (PEP 695, 3.12+) — the ast.TypeAlias node's value is
    the aliased type expression; a string value is a forward reference that must
    not be flagged.  Skipped on Python < 3.12 where the ``type`` statement is a
    syntax error."""
    import sys
    if sys.version_info < (3, 12):
        import pytest
        pytest.skip("PEP 695 type statement requires Python 3.12+")
    source = textwrap.dedent("""\
        from external_llm.agent.failure_pattern_store import FailurePatternStore

        type MyAlias = "FailurePatternStore"
    """)
    path = _make_py_file(source)
    try:
        results = scan_unused_imports(repo_root="", file_paths=[path])
        flagged = {c.symbol_name for c in results}
        assert "FailurePatternStore" not in flagged, (
            f"PEP 695 type alias forward-ref must NOT be flagged, got {flagged}"
        )
    finally:
        Path(path).unlink(missing_ok=True)
