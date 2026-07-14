"""Tests for external_llm/analysis/public_dead_code_scanner.py."""
from __future__ import annotations

import tempfile
import textwrap
from pathlib import Path

from external_llm.analysis.public_dead_code_scanner import (
    DeadBlockCandidate,
    DeadBlockMember,
    _collect_all_defs,
    _collect_name_references,
    _has_framework_injection_decorator,
    _is_externally_referenced,
    scan_public_dead_blocks,
)


def _make_py_file(source: str) -> str:
    """Write source to a temp .py file and return its absolute path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8",
    )
    tmp.write(textwrap.dedent(source))
    tmp.close()
    return tmp.name


# ── Private symbol detection ─────────────────────────────────────────────


def test_private_functions_always_detected():
    """_-prefixed functions are always dead candidates (like dead_block_scanner)."""
    src = _make_py_file("""\
        def _helper_one():
            return 1

        def _helper_two():
            return 2

        def public_func():
            return _helper_one() + _helper_two()
    """)
    # Without cross_file_referenced_names, only private symbols are scanned.
    candidates = scan_public_dead_blocks(repo_root="", file_paths=[src])
    # _helper_one and _helper_two are dead (no reference outside their defs)
    # But they're used inside public_func... wait.
    # _helper_one is referenced inside public_func() at line 8, but let's check:
    # def _helper_one: lines 2-3, reference at line 8 is outside [2,3], so it IS referenced.
    # So it should NOT be dead.
    assert len(candidates) == 0
    Path(src).unlink()


def test_private_function_with_no_references():
    """_private function with zero references should be detected."""
    src = _make_py_file("""\
        def _helper_one():
            return 1

        def _helper_two():
            return 2

        def public_func():
            return 42
    """)
    candidates = scan_public_dead_blocks(repo_root="", file_paths=[src])
    assert len(candidates) == 1
    assert candidates[0].cluster_start == 1
    assert candidates[0].cluster_end == 5  # _helper_one (1-2), _helper_two (4-5)
    names = {m.name for m in candidates[0].members}
    assert names == {"_helper_one", "_helper_two"}
    Path(src).unlink()


def test_private_method_not_confused_with_dunder():
    """__init__ and similar dunders are NOT flagged."""
    src = _make_py_file("""\
        class MyClass:
            def __init__(self):
                pass

        def _unused():
            pass

        def _also_unused():
            pass
    """)
    candidates = scan_public_dead_blocks(repo_root="", file_paths=[src])
    # _unused + _also_unused should form a cluster
    assert len(candidates) >= 1
    names = [m.name for c in candidates for m in c.members]
    assert "__init__" not in names
    Path(src).unlink()


# ── Public symbol detection ──────────────────────────────────────────────


def test_public_symbols_not_detected_without_cross_file_refs():
    """Without cross_file_referenced_names, public symbols are conservatively skipped.

    Singleton private dead symbols are now emitted with lower confidence
    (0.55) instead of being silently dropped when they lack a cluster partner.
    """
    src = _make_py_file("""\
        def public_func_one():
            return 1

        def public_func_two():
            return 2

        def _private_func():
            return 3
    """)
    candidates = scan_public_dead_blocks(repo_root="", file_paths=[src])
    # _private_func emitted as singleton with lower confidence
    assert len(candidates) == 1
    assert candidates[0].is_singleton
    assert candidates[0].confidence == 0.55
    assert candidates[0].members[0].name == "_private_func"
    Path(src).unlink()


def test_public_symbol_detected_with_cross_file_refs():
    """With cross_file_referenced_names, public names NOT in the set are detected."""
    src = _make_py_file("""\
        def public_func_one():
            return 1

        def public_func_two():
            return 2

        def _private_func():
            return 3
    """)
    candidates = scan_public_dead_blocks(
        repo_root="", file_paths=[src],
        cross_file_referenced_names={"some_other_func"},
    )
    # public_func_one, public_func_two, _private_func should all be dead.
    # Need >=2 per cluster to be emitted.
    assert len(candidates) >= 1
    all_names = {m.name for c in candidates for m in c.members}
    assert "public_func_one" in all_names
    assert "public_func_two" in all_names
    assert "_private_func" in all_names
    Path(src).unlink()


def test_public_symbol_not_detected_if_in_cross_file_refs():
    """Public name present in cross_file_referenced_names is NOT flagged."""
    src = _make_py_file("""\
        def used_elsewhere():
            return 1

        def _helper():
            return 2
    """)
    candidates = scan_public_dead_blocks(
        repo_root="", file_paths=[src],
        cross_file_referenced_names={"used_elsewhere"},
    )
    # used_elsewhere has cross-file references → not dead
    # _helper is private with no references → dead singleton (emitted with 0.55 confidence)
    assert len(candidates) == 1
    assert candidates[0].is_singleton
    assert candidates[0].confidence == 0.55
    assert candidates[0].members[0].name == "_helper"
    Path(src).unlink()


# ── Clustering ───────────────────────────────────────────────────────────


def test_clustering_adjacent_defs():
    """Adjacent dead definitions are clustered together."""
    src = _make_py_file("""\
        def _dead_a():
            pass

        def _dead_b():
            pass

        def _dead_c():
            pass

        def alive():
            return 42
    """)
    candidates = scan_public_dead_blocks(repo_root="", file_paths=[src])
    assert len(candidates) == 1
    assert len(candidates[0].members) == 3
    member_names = [m.name for m in candidates[0].members]
    assert member_names == ["_dead_a", "_dead_b", "_dead_c"]
    Path(src).unlink()


def test_clustering_with_gap_tolerance():
    """Gap between defs larger than tolerance splits clusters."""
    src = _make_py_file("""\
        def _dead_a():
            pass



        def _dead_b():
            pass
    """)
    # _dead_a end_lineno=2, _dead_b lineno=6, gap >= 4 (2 to 6 = 4 lines of gap)
    # gap = 6 - 2 = 4, which is <= tolerance of 5, so they DO cluster
    # Let's use a larger gap
    lines = ["def _dead_a(): pass\n"] + ["\n"] * 10 + ["def _dead_b(): pass\n"]
    src = _make_py_file("".join(lines))
    candidates = scan_public_dead_blocks(repo_root="", file_paths=[src])
    # Gap > 5 → separate singletons, each with 1 member → 2 singleton candidates
    assert len(candidates) == 2
    for c in candidates:
        assert c.is_singleton
        assert c.confidence == 0.55
    Path(src).unlink()


# ── Framework injection decorators ──────────────────────────────────────


def test_pytest_fixture_not_flagged():
    """@pytest.fixture decorated functions are never dead — pytest discovers them by name."""
    src = _make_py_file("""\
        import pytest

        @pytest.fixture
        def agent_loop():
            return object()

        @pytest.fixture
        def tmp_workspace():
            return object()

        def _unused_helper():
            pass

        def _another_unused():
            pass
    """)
    candidates = scan_public_dead_blocks(
        repo_root="", file_paths=[src],
        cross_file_referenced_names=set(),
    )
    all_names = {m.name for c in candidates for m in c.members}
    assert "agent_loop" not in all_names
    assert "tmp_workspace" not in all_names
    Path(src).unlink()


def test_bare_fixture_decorator_not_flagged():
    """@fixture (imported directly) is also exempt."""
    src = _make_py_file("""\
        from pytest import fixture

        @fixture
        def my_fixture():
            return 42

        @fixture
        def another_fixture():
            return 99

        def _dead_helper():
            pass

        def _dead_other():
            pass
    """)
    candidates = scan_public_dead_blocks(
        repo_root="", file_paths=[src],
        cross_file_referenced_names=set(),
    )
    all_names = {m.name for c in candidates for m in c.members}
    assert "my_fixture" not in all_names
    assert "another_fixture" not in all_names
    Path(src).unlink()


def test_fixture_with_args_not_flagged():
    """@pytest.fixture(scope='session') call form is also exempt."""
    src = _make_py_file("""\
        import pytest

        @pytest.fixture(scope="session")
        def session_db():
            return object()

        @pytest.fixture(scope="module", autouse=True)
        def module_setup():
            pass

        def _unreferenced_a():
            pass

        def _unreferenced_b():
            pass
    """)
    candidates = scan_public_dead_blocks(
        repo_root="", file_paths=[src],
        cross_file_referenced_names=set(),
    )
    all_names = {m.name for c in candidates for m in c.members}
    assert "session_db" not in all_names
    assert "module_setup" not in all_names
    Path(src).unlink()


def test_non_fixture_public_symbol_still_detected():
    """Non-decorated public symbols without cross-file refs are still detected."""
    src = _make_py_file("""\
        import pytest

        @pytest.fixture
        def live_fixture():
            return object()

        def dead_helper_one():
            pass

        def dead_helper_two():
            pass
    """)
    candidates = scan_public_dead_blocks(
        repo_root="", file_paths=[src],
        cross_file_referenced_names=set(),
    )
    all_names = {m.name for c in candidates for m in c.members}
    assert "live_fixture" not in all_names
    assert "dead_helper_one" in all_names
    assert "dead_helper_two" in all_names
    Path(src).unlink()


def test_fixture_parameter_prevents_same_file_dead():
    """Fixture used as parameter in same file is counted as referenced (ast.arg)."""
    src = _make_py_file("""\
        def _helper_a():
            return 1

        def _helper_b():
            return 2

        def consumer(_helper_a, _helper_b):
            pass
    """)
    candidates = scan_public_dead_blocks(repo_root="", file_paths=[src])
    all_names = {m.name for c in candidates for m in c.members}
    assert "_helper_a" not in all_names
    assert "_helper_b" not in all_names
    Path(src).unlink()


def test_has_framework_injection_decorator_variants():
    """_has_framework_injection_decorator detects all four syntactic forms."""
    import ast as _ast

    forms = [
        "@fixture\ndef f(): pass",
        "@pytest.fixture\ndef f(): pass",
        "@fixture(scope='session')\ndef f(): pass",
        "@pytest.fixture(scope='session')\ndef f(): pass",
        "@hookimpl\ndef f(): pass",
        "@pytest.hookspec\ndef f(): pass",
    ]
    for src in forms:
        tree = _ast.parse(src)
        node = tree.body[0]
        assert _has_framework_injection_decorator(node), f"not detected: {src!r}"

    not_injection = [
        "@staticmethod\ndef f(): pass",
        "@classmethod\ndef f(): pass",
        "@property\ndef f(): pass",
        "@overload\ndef f(): pass",
        "@app.route('/')\ndef f(): pass",
    ]
    for src in not_injection:
        tree = _ast.parse(src)
        node = tree.body[0]
        assert not _has_framework_injection_decorator(node), f"false positive: {src!r}"


# ── Exclusion zones ──────────────────────────────────────────────────────


def test_all_reexport_skipped():
    """Names in __all__ are not dead code."""
    src = _make_py_file("""\
        def public_api():
            return 1

        def _internal():
            return 2

        __all__ = ["public_api"]
    """)
    candidates = scan_public_dead_blocks(
        repo_root="", file_paths=[src],
        cross_file_referenced_names=set(),
    )
    # public_api is in __all__ → not dead
    # _internal is private with no refs → dead singleton (emitted with 0.55 confidence)
    assert len(candidates) == 1
    assert candidates[0].is_singleton
    assert candidates[0].confidence == 0.55
    assert candidates[0].members[0].name == "_internal"
    Path(src).unlink()


def test_dunder_names_skipped():
    """__magic__ names are not flagged."""
    src = _make_py_file("""\
        __all__ = ["func"]

        def func():
            return 1

        def _helper():
            return 2

        def _also_helper():
            return 3
    """)
    candidates = scan_public_dead_blocks(
        repo_root="", file_paths=[src],
        cross_file_referenced_names=set(),
    )
    names = {m.name for c in candidates for m in c.members}
    assert "__all__" not in names  # dunder, and also assignment
    Path(src).unlink()


def test_overload_functions_skipped():
    """@overload decorated functions are excluded."""
    src = _make_py_file("""\
        from typing import overload

        @overload
        def func(x: int) -> int: ...
        @overload
        def func(x: str) -> str: ...

        def _dead():
            return 1

        def _also_dead():
            return 2
    """)
    candidates = scan_public_dead_blocks(repo_root="", file_paths=[src])
    # _dead + _also_dead should form cluster
    assert len(candidates) >= 1
    all_names = {m.name for c in candidates for m in c.members}
    assert "func" not in all_names  # overload excluded
    Path(src).unlink()


def test_same_file_reference_prevents_dead():
    """Reference outside its own def range but within same file prevents dead label."""
    src = _make_py_file("""\
        def _helper():
            return 42

        def caller():
            return _helper()
    """)
    candidates = scan_public_dead_blocks(repo_root="", file_paths=[src])
    # _helper IS referenced at line 6 (outside its def at lines 2-3)
    # So it should NOT be dead
    assert len(candidates) == 0
    Path(src).unlink()


# ── Edge cases ───────────────────────────────────────────────────────────


def test_max_per_file_enforced():
    """Respect max_per_file limit."""
    # Need enough gap between groups so they form separate clusters
    lines = []
    for i in range(7):
        lines.append(f"def _dead_{chr(ord('a') + i)}(): pass\n")
    # All adjacent, forms 1 cluster → can't test max_per_file this way
    # Instead generate 4 clusters of 2 defs each with gaps
    lines = []
    for cluster_idx in range(4):
        lines.append(f"def _dead_{cluster_idx}a(): pass\n")
        lines.append(f"def _dead_{cluster_idx}b(): pass\n")
        lines.extend(["\n"] * 8)  # large gap between clusters
    src = _make_py_file("".join(lines))
    candidates = scan_public_dead_blocks(repo_root="", file_paths=[src], max_per_file=2)
    assert len(candidates) == 2
    Path(src).unlink()


def test_non_py_file_skipped():
    """Non-.py files are skipped (pre-filtered by ScannerRegistry)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w")
    tmp.write("def _dead(): pass\n")
    tmp.close()
    from external_llm.agent.scanner_registry import get_registry
    reg = get_registry()
    result = reg.run("public_dead_code_scanner", file_paths=[tmp.name])
    assert not result.candidates_raw, f"Expected 0 candidates, got {len(result.candidates_raw)}"
    Path(tmp.name).unlink()


def test_missing_file_skipped():
    """Non-existent files are skipped without error."""
    candidates = scan_public_dead_blocks(repo_root="", file_paths=["/nonexistent/file.py"])
    assert not candidates


def test_syntax_error_file_skipped():
    """Files with syntax errors are skipped."""
    src = _make_py_file("""\
        def _dead_a():
            pass
        this is invalid python
    """)
    candidates = scan_public_dead_blocks(repo_root="", file_paths=[src])
    assert not candidates
    Path(src).unlink()


def test_dynamic_all_skipped():
    """Dynamic __all__ causes conservative skip."""
    src = _make_py_file("""\
        __all__ = [x for x in names]
        def _dead(): pass
    """)
    candidates = scan_public_dead_blocks(repo_root="", file_paths=[src])
    assert not candidates
    Path(src).unlink()


# ── Candidate structure ──────────────────────────────────────────────────


def test_dead_block_candidate_to_dict():
    """DeadBlockCandidate.to_dict() returns correct fields."""
    cand = DeadBlockCandidate(
        file="test.py",
        members=[
            DeadBlockMember(name="_dead", symbol_kind="function", lineno=2, end_lineno=4),
            DeadBlockMember(name="_also_dead", symbol_kind="function", lineno=6, end_lineno=8),
        ],
        cluster_start=2,
        cluster_end=8,
        includes_public=False,
    )
    d = cand.to_dict()
    assert d["file"] == "test.py"
    assert len(d["members"]) == 2
    assert d["members"][0]["name"] == "_dead"
    assert d["cluster_start"] == 2
    assert d["cluster_end"] == 8
    assert d["includes_public"] is False


# ── Internal helpers ─────────────────────────────────────────────────────


def test_collect_all_defs():
    """_collect_all_defs finds module-level and class-level functions, classes, assignments."""
    import ast
    tree = ast.parse("""\
def func(): pass
class Klass: pass
x = 1
y: int = 2
""")
    defs = _collect_all_defs(tree)
    names = [d[0] for d in defs]
    assert "func" in names
    assert "Klass" in names
    assert "x" in names
    assert "y" in names


def test_collect_all_defs_class_body():
    """_collect_all_defs skips class-level assignments (API-contract false positives)."""
    import ast
    tree = ast.parse("""\
class MyClass:
    _CONST = 1
    _ANOTHER: int = 2
    def method(self): pass
""")
    defs = _collect_all_defs(tree)
    by_name = {(d[0], d[4]): (d[1],) for d in defs}
    assert ("MyClass", None) in by_name  # class itself, no enclosing_class
    # Class-level assignments are NOT collected (false positive risk)
    assert ("_CONST", "MyClass") not in by_name
    assert ("_ANOTHER", "MyClass") not in by_name
    # method should NOT be collected
    assert not any(d[0] == "method" for d in defs)


def test_collect_name_references():
    """_collect_name_references finds Name nodes in Load context."""
    import ast
    tree = ast.parse("""\
x = os.getcwd()
y = os.path.join("a", "b")
""")
    refs = _collect_name_references(tree)
    assert "os" in refs


def test_is_externally_referenced_same_file():
    """_is_externally_referenced detects same-file references outside def range."""
    references = {"helper": [10, 15]}
    # def helper at lines 3-5, referenced at 10 and 15 → externally referenced
    assert _is_externally_referenced("helper", 3, 5, references) is True
    # Not referenced outside def range
    assert _is_externally_referenced("helper", 1, 20, references) is False


def test_is_externally_referenced_cross_file():
    """_is_externally_referenced detects cross-file references."""
    references = {}
    # No same-file refs, but name is in cross_file_referenced_names
    assert _is_externally_referenced(
        "helper", 1, 5, references,
        cross_file_referenced_names={"helper"},
    ) is True
    assert _is_externally_referenced(
        "helper", 1, 5, references,
        cross_file_referenced_names={"other"},
    ) is False


# ── Use-position references (for-iterable / assignment RHS / with) ───────


def test_for_iterable_and_rhs_references_are_not_dead():
    """If the definition-position check only looks at the parent node type,
    `_TABLE` in `for x in _TABLE:`, `_ORIG` in `alias = _ORIG`, and `_LOCK` in
    `with _LOCK:` fail to register as references and get misjudged as dead
    (real case: asi._SLASH_COMMANDS). All of these must count as in use."""
    src = _make_py_file("""\
        _TABLE = [("a", 1), ("b", 2)]
        _ORIG = object()
        _LOCK = __import__("threading").Lock()

        for _k, _v in _TABLE:
            print(_k, _v)

        alias = _ORIG

        def use_lock():
            with _LOCK:
                return alias
    """)
    candidates = scan_public_dead_blocks(repo_root="", file_paths=[src])
    dead_names = {m.name for c in candidates for m in c.members}
    assert "_TABLE" not in dead_names
    assert "_ORIG" not in dead_names
    assert "_LOCK" not in dead_names


def test_class_level_assignment_not_scanned_via_tree_sitter():
    """Class-level assignments can't be judged dead via single-file analysis,
    since mixin/instance attribute access (self._FOO) is cross-file — the ts
    path must exclude them just like the AST path does."""
    src = _make_py_file("""\
        class SomeMixin:
            _CLASS_TABLE: frozenset = frozenset({"a", "b"})

            def method(self):
                return 1
    """)
    candidates = scan_public_dead_blocks(repo_root="", file_paths=[src])
    dead_names = {m.name for c in candidates for m in c.members}
    assert "_CLASS_TABLE" not in dead_names
