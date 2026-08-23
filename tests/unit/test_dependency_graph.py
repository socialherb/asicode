"""Regression tests for DependencyGraphBuilder (external_llm/dependency_graph.py).

Primary target: DG-B1 — ``_track_internal_calls`` associated the FILE-LEVEL call
set (``analysis.calls``) with EVERY function in the file. A call made by one
function was attributed to all functions — including the callee itself
(false self-loop ``bar -> bar``) and unrelated functions. The false call edges
were fed to the LLM via ``super_context_builder``'s "relationships" block.
Coverage was 0/62 branches before this file.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from external_llm.dependency_graph import DependencyGraphBuilder


def _build(src: str, tmp_path: Path):
    (tmp_path / "pkg").mkdir(exist_ok=True)
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "mod.py").write_text(textwrap.dedent(src))
    builder = DependencyGraphBuilder(tmp_path)
    return builder.build_graph(tmp_path / "pkg" / "mod.py", max_depth=0)


# --- DG-B1: per-function call attribution ---


def test_only_actual_caller_gets_callee(tmp_path):
    """Only calls() calls bar(); bar and unrelated must NOT call bar."""
    graph = _build(
        """
        def bar():
            return 1

        def calls():
            return bar()

        def unrelated():
            return 2
        """,
        tmp_path,
    )
    calls = dict(graph.calls)  # snapshot before any defaultdict access
    assert calls.get("pkg/mod.py:calls") == ["pkg/mod.py:bar"]
    # bar itself must NOT have a self-loop (buggy: bar -> [bar])
    assert "pkg/mod.py:bar" not in calls
    # unrelated must NOT call bar (buggy: unrelated -> [bar])
    assert "pkg/mod.py:unrelated" not in calls


def test_called_by_reverse_correct(tmp_path):
    graph = _build(
        """
        def bar():
            return 1

        def calls():
            return bar()
        """,
        tmp_path,
    )
    called_by = dict(graph.called_by)
    assert called_by.get("pkg/mod.py:bar") == ["pkg/mod.py:calls"]


def test_function_with_no_calls_has_no_edges(tmp_path):
    graph = _build(
        """
        def lone():
            x = 1
            return x

        def helper():
            return lone()
        """,
        tmp_path,
    )
    calls = dict(graph.calls)
    assert "pkg/mod.py:lone" not in calls
    assert calls.get("pkg/mod.py:helper") == ["pkg/mod.py:lone"]


def test_nested_function_calls_not_attributed_to_outer(tmp_path):
    """inner() calls helper(), but inner is a nested scope — its calls must not
    be attributed to outer (buggy file-level set: outer -> [helper])."""
    graph = _build(
        """
        def helper():
            return 1

        def outer():
            def inner():
                return helper()
            return inner()
        """,
        tmp_path,
    )
    calls = dict(graph.calls)
    assert "pkg/mod.py:outer" not in calls


def test_two_independent_callers(tmp_path):
    graph = _build(
        """
        def base():
            return 0

        def left():
            return base()

        def right():
            return base()
        """,
        tmp_path,
    )
    calls = dict(graph.calls)
    assert calls.get("pkg/mod.py:left") == ["pkg/mod.py:base"]
    assert calls.get("pkg/mod.py:right") == ["pkg/mod.py:base"]
    called_by = dict(graph.called_by)
    assert set(called_by.get("pkg/mod.py:base", [])) == {"pkg/mod.py:left", "pkg/mod.py:right"}


# ===========================================================================
# DG-T1: _resolve_import + format_call_graph coverage (previously 0% covered)
# ===========================================================================
# During RED design a real defect was uncovered (DG-B2 below): CodeAnalyzer
# parsed ``ast.ImportFrom`` without ``node.level``, so a relative import like
# ``from .sibling import VAL`` was stored as ``module='sibling'`` (dots
# stripped). ``_resolve_import`` then took the ABSOLUTE branch and looked for
# ``repo_root/sibling.py`` — never finding the in-package sibling. The rest of
# the codebase already encodes ``node.level`` as leading dots
# (ast_op_executor.py:372/1018, tool_safety.py:634); code_analyzer.py was the
# sole outlier. Fixed in code_analyzer.py by mirroring that canonical pattern.


class TestResolveImport:
    """Unit-test ``_resolve_import`` directly across the import-space.

    Layout (current_file = root/pkg/sub/mod.py)::

        root/
            lib.py
            pkg/
                __init__.py
                util.py
                sub/
                    __init__.py
                    mod.py        <- current_file
                    local.py
                pkg2/
                    __init__.py
                    deep.py
    """

    @staticmethod
    def _layout(root: Path) -> Path:
        """Create the canonical layout and return current_file."""
        for rel in [
            "pkg/__init__.py",
            "pkg/util.py",
            "pkg/sub/__init__.py",
            "pkg/sub/mod.py",
            "pkg/sub/local.py",
            "pkg/pkg2/__init__.py",
            "pkg/pkg2/deep.py",
            "lib.py",
        ]:
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("X = 1\n")
        return root / "pkg" / "sub" / "mod.py"

    def test_empty_module_returns_none(self, tmp_path):
        cur = self._layout(tmp_path)
        b = DependencyGraphBuilder(tmp_path)
        assert b._resolve_import(cur, "") is None

    def test_nonexistent_absolute_returns_none(self, tmp_path):
        cur = self._layout(tmp_path)
        b = DependencyGraphBuilder(tmp_path)
        assert b._resolve_import(cur, "does.not.exist") is None

    def test_absolute_single_part_file(self, tmp_path):
        cur = self._layout(tmp_path)
        b = DependencyGraphBuilder(tmp_path)
        assert b._resolve_import(cur, "lib") == tmp_path / "lib.py"

    def test_absolute_dotted_file(self, tmp_path):
        """Nested absolute import ``pkg.pkg2.deep`` resolves to the .py file."""
        cur = self._layout(tmp_path)
        b = DependencyGraphBuilder(tmp_path)
        assert b._resolve_import(cur, "pkg.pkg2.deep") == tmp_path / "pkg" / "pkg2" / "deep.py"

    def test_absolute_package_resolves_to_init(self, tmp_path):
        """``pkg.sub`` (a package dir) resolves to its __init__.py."""
        cur = self._layout(tmp_path)
        b = DependencyGraphBuilder(tmp_path)
        assert b._resolve_import(cur, "pkg.sub") == tmp_path / "pkg" / "sub" / "__init__.py"

    def test_relative_single_dot_sibling_file(self, tmp_path):
        """``.local`` (level 1) resolves to a file in the SAME directory."""
        cur = self._layout(tmp_path)
        b = DependencyGraphBuilder(tmp_path)
        assert b._resolve_import(cur, ".local") == tmp_path / "pkg" / "sub" / "local.py"

    def test_relative_single_dot_missing_returns_none(self, tmp_path):
        """``.util`` from pkg/sub has no pkg/sub/util.py -> None."""
        cur = self._layout(tmp_path)
        b = DependencyGraphBuilder(tmp_path)
        assert b._resolve_import(cur, ".util") is None

    def test_relative_double_dot_parent_file(self, tmp_path):
        """``..util`` (level 2) walks up one dir to pkg/util.py."""
        cur = self._layout(tmp_path)
        b = DependencyGraphBuilder(tmp_path)
        assert b._resolve_import(cur, "..util") == tmp_path / "pkg" / "util.py"

    def test_relative_triple_dot_grandparent_file(self, tmp_path):
        """``...lib`` (level 3) walks up two dirs to root/lib.py."""
        cur = self._layout(tmp_path)
        b = DependencyGraphBuilder(tmp_path)
        assert b._resolve_import(cur, "...lib") == tmp_path / "lib.py"

    def test_relative_double_dot_package_init(self, tmp_path):
        """``..pkg2.deep``: pkg2 is a package dir -> its __init__.py wins."""
        cur = self._layout(tmp_path)
        b = DependencyGraphBuilder(tmp_path)
        assert b._resolve_import(cur, "..pkg2.deep") == tmp_path / "pkg" / "pkg2" / "__init__.py"


class TestFormatCallGraph:
    """Cover all branches of ``format_call_graph`` (calls / called_by / both /
    neither / cap). Previously 0% covered (lines 262-280)."""

    @staticmethod
    def _builder(tmp_path):
        return DependencyGraphBuilder(tmp_path)

    def test_no_information(self, tmp_path):
        b = self._builder(tmp_path)
        assert b.format_call_graph(_empty_graph(), "anything") == "No call information available"

    def test_calls_only(self, tmp_path):
        g = _empty_graph()
        g.calls["mod:f"] = ["mod:g", "mod:h"]
        b = self._builder(tmp_path)
        out = b.format_call_graph(g, "mod:f")
        assert "mod:f calls:" in out
        assert "├─ mod:g" in out
        assert "├─ mod:h" in out
        assert "Called by:" not in out

    def test_called_by_only(self, tmp_path):
        g = _empty_graph()
        g.called_by["mod:f"] = ["mod:c1", "mod:c2"]
        b = self._builder(tmp_path)
        out = b.format_call_graph(g, "mod:f")
        assert out.startswith("Called by:")
        assert "├─ mod:c1" in out
        assert "├─ mod:c2" in out
        assert "calls:" not in out

    def test_both_sections_separated_by_blank_line(self, tmp_path):
        g = _empty_graph()
        g.calls["mod:f"] = ["mod:g"]
        g.called_by["mod:f"] = ["mod:c1"]
        b = self._builder(tmp_path)
        out = b.format_call_graph(g, "mod:f")
        # blank line separates the two sections
        assert "\n\nCalled by:" in out
        assert out.index("mod:f calls:") < out.index("Called by:")

    def test_max_items_caps_list(self, tmp_path):
        g = _empty_graph()
        g.calls["mod:f"] = ["a", "b", "c", "d"]
        b = self._builder(tmp_path)
        out = b.format_call_graph(g, "mod:f", max_items=2)
        assert "├─ a" in out and "├─ b" in out
        assert "├─ c" not in out and "├─ d" not in out


def _empty_graph():
    """A DependencyGraph with no pre-existing keys (avoids defaultdict noise)."""
    from external_llm.dependency_graph import DependencyGraph

    return DependencyGraph()


# ===========================================================================
# DG-B2: relative imports must be tracked in file_imports / exports
# ===========================================================================
# RED before the fix: CodeAnalyzer dropped the leading dots, so ``from .sibling
# import VAL`` was parsed as module='sibling' and ``_resolve_import`` took the
# ABSOLUTE branch -> looked for repo_root/sibling.py -> None -> the import was
# silently dropped from the graph. After the fix module='.sibling' reaches
# _resolve_import's relative branch and resolves correctly.


def _pkg(tmp_path: Path):
    (tmp_path / "pkg").mkdir(exist_ok=True)
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "sibling.py").write_text("VAL = 1\n")
    return tmp_path / "pkg" / "mod.py"


def test_relative_single_dot_import_resolved(tmp_path):
    """``from .sibling import VAL`` must appear in file_imports + exports."""
    mod = _pkg(tmp_path)
    mod.write_text("from .sibling import VAL\n")
    b = DependencyGraphBuilder(tmp_path)
    graph = b.build_graph(mod, max_depth=2)
    imports = dict(graph.file_imports)
    assert imports.get("pkg/mod.py") == ["pkg/sibling.py"], imports
    exports = {k: list(v) for k, v in graph.exports.items()}
    assert exports.get("pkg/sibling.py") == ["VAL"], exports


def test_relative_double_dot_import_resolved(tmp_path):
    """``from ..other import x`` (level 2) walks up to the parent package."""
    (tmp_path / "pkg" / "sub").mkdir(parents=True)
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "sub" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "other.py").write_text("X = 1\n")
    mod = tmp_path / "pkg" / "sub" / "mod.py"
    mod.write_text("from ..other import X\n")
    b = DependencyGraphBuilder(tmp_path)
    graph = b.build_graph(mod, max_depth=2)
    imports = dict(graph.file_imports)
    assert imports.get("pkg/sub/mod.py") == ["pkg/other.py"], imports


def test_relative_import_not_misresolved_as_absolute(tmp_path):
    """Regression guard: ``.sibling`` must NOT resolve to repo_root/sibling.py.

    Before the fix, the stripped module name 'sibling' took the absolute branch
    and a stray repo_root/sibling.py (if it existed) would be wrongly matched.
    """
    # Plant a DECOY absolute sibling that the buggy path would have matched.
    (tmp_path / "pkg").mkdir(exist_ok=True)
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "sibling.py").write_text("VAL = 1\n")  # the real target
    (tmp_path / "sibling.py").write_text("DECOY = 999\n")  # must NOT match
    mod = tmp_path / "pkg" / "mod.py"
    mod.write_text("from .sibling import VAL\n")
    b = DependencyGraphBuilder(tmp_path)
    graph = b.build_graph(mod, max_depth=2)
    imports = dict(graph.file_imports)
    assert imports.get("pkg/mod.py") == ["pkg/sibling.py"], imports
