"""Unit tests for call_graph.CallGraphIndexer."""
import ast
import shutil
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

from external_llm.agent._shared_utils import invalidate_walk_caches
from external_llm.agent.call_graph import (
    CallGraphIndexer,
    _iter_calls,
    _walk_py_files,
    _walk_ts_js_files,
)


def _make_repo(files: dict) -> str:
    """Create a temp directory with given filename->source mapping."""
    d = tempfile.mkdtemp(prefix="test_cg_")
    for rel_path, source in files.items():
        full = Path(d) / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(textwrap.dedent(source))
    return d


# ─── Case 1: top-level function a() calls b() ────────────────────────────────

def test_simple_caller_callee():
    repo = _make_repo({
        "foo.py": """
            def b():
                pass

            def a():
                b()
        """
    })
    try:
        idx = CallGraphIndexer(repo)
        # callee of a -> b
        callees = idx.get_callees("a")
        callee_syms = [e.callee_symbol for e in callees]
        assert "b" in callee_syms, f"Expected 'b' in callees of 'a', got {callee_syms}"

        # caller of b -> a
        callers = idx.get_callers("b")
        caller_syms = [e.caller_symbol for e in callers]
        assert "a" in caller_syms, f"Expected 'a' in callers of 'b', got {caller_syms}"
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ─── Case 2: class method self.b() ───────────────────────────────────────────

def test_class_method_self_call():
    repo = _make_repo({
        "bar.py": """
            class X:
                def a(self):
                    self.b()

                def b(self):
                    pass
        """
    })
    try:
        idx = CallGraphIndexer(repo)
        callees = idx.get_callees("X.a")
        callee_syms = [e.callee_symbol for e in callees]
        assert "X.b" in callee_syms, f"Expected 'X.b' in callees of 'X.a', got {callee_syms}"

        callers = idx.get_callers("X.b")
        caller_syms = [e.caller_symbol for e in callers]
        assert "X.a" in caller_syms, f"Expected 'X.a' in callers of 'X.b', got {caller_syms}"
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ─── Case 2b: nested function calls belong to the nested caller only (B1) ───

def test_nested_function_calls_not_misattributed():
    """B1: calls inside a nested def are NOT attributed to the enclosing fn."""
    repo = _make_repo({
        "nested.py": """
            def helper():
                pass

            def outer():
                def inner():
                    helper()
                inner()

            def sibling():
                helper()
        """
    })
    try:
        idx = CallGraphIndexer(repo)
        # outer calls only inner() — helper() lives inside inner's body
        outer_callees = idx.get_callees("outer")
        outer_syms = [e.callee_symbol for e in outer_callees]
        assert outer_syms == ["inner"], (
            f"Expected outer -> [inner] only, got {outer_syms}"
        )
        # B1-2: caller_line points at the actual call site, not the def line
        assert outer_callees[0].caller_line == 8, (
            f"Expected caller_line=8 (inner() call site), "
            f"got {outer_callees[0].caller_line}"
        )
        # inner (nested) is its own caller and calls helper()
        inner_callees = idx.get_callees("inner")
        inner_syms = [e.callee_symbol for e in inner_callees]
        assert inner_syms == ["helper"], (
            f"Expected inner -> [helper] only, got {inner_syms}"
        )
        assert inner_callees[0].caller_line == 7
        # helper's callers: inner and sibling — never outer
        helper_callers = sorted(e.caller_symbol for e in idx.get_callers("helper"))
        assert helper_callers == ["inner", "sibling"], (
            f"Expected helper callers = [inner, sibling], got {helper_callers}"
        )
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ─── Case 3: get_related_symbols structure ───────────────────────────────────

def test_get_related_symbols_structure():
    repo = _make_repo({
        "svc.py": """
            def helper():
                pass

            def main():
                helper()
        """
    })
    try:
        idx = CallGraphIndexer(repo)
        result = idx.get_related_symbols("main")
        assert "symbol" in result
        assert "callees" in result
        assert "callers" in result
        assert "related_symbols" in result
        assert "next_read_candidates" in result
        assert isinstance(result["callees"], list)
        assert isinstance(result["next_read_candidates"], list)
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ─── Case 4: callee_file resolved cross-file ─────────────────────────────────

def test_cross_file_resolution():
    repo = _make_repo({
        "utils.py": """
            def util_fn():
                pass
        """,
        "main.py": """
            from utils import util_fn

            def caller():
                util_fn()
        """,
    })
    try:
        idx = CallGraphIndexer(repo)
        callees = idx.get_callees("caller")
        callee_syms = [e.callee_symbol for e in callees]
        assert "util_fn" in callee_syms

        # callee_file should be resolved to utils.py
        for e in callees:
            if e.callee_symbol == "util_fn":
                assert e.callee_file == "utils.py", (
                    f"Expected callee_file='utils.py', got {e.callee_file!r}"
                )
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ─── Case 5: invalidate clears index ─────────────────────────────────────────

def test_invalidate():
    repo = _make_repo({
        "a.py": """
            def foo():
                bar()
            def bar():
                pass
        """
    })
    try:
        idx = CallGraphIndexer(repo)
        assert len(idx.get_callees("foo")) > 0
        idx.invalidate()
        assert not idx._built
        # After invalidate, next call rebuilds
        result = idx.get_related_symbols("foo")
        assert idx._built
        assert result["symbol"] == "foo"
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ─── Case 7: _walk_py_files skips vendored/hidden dirs ──────────────────────

def test_walk_py_files_skips_hidden_dirs():
    repo = _make_repo({
        ".venv/lib/site-packages/pkg.py": "x = 1",
        "app/module.py": "y = 2",
    })
    try:
        files = _walk_py_files(Path(repo))
        names = [f.name for f in files]
        assert "module.py" in names, f"Expected module.py, got {names}"
        assert "pkg.py" not in names, "pkg.py under .venv should be skipped"
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ─── Case 8: _walk_ts_js_files skips vendored/hidden dirs ───────────────────

def test_walk_ts_js_files_skips_hidden_dirs():
    repo = _make_repo({
        "node_modules/lib/index.js": "var x = 1;",
        "src/app.ts": "const y = 2;",
    })
    try:
        files = _walk_ts_js_files(Path(repo))
        names = [f.name for f in files]
        assert "app.ts" in names, f"Expected app.ts, got {names}"
        assert "index.js" not in names, "index.js under node_modules should be skipped"
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ─── Case 9: SyntaxError in source file is skipped silently ─────────────────

def test_syntax_error_skipped():
    repo = _make_repo({
        "bad.py": "def foo( bar",  # SyntaxError
        "good.py": "def ok(): pass",
    })
    try:
        idx = CallGraphIndexer(repo)
        # build() should not crash
        callees = idx.get_callees("ok")
        assert callees == []
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ─── Case 10: suffix fallback in get_callees ────────────────────────────────

def test_get_callees_suffix_fallback():
    """Bare caller name matches qualified index key (e.g. 'a' → 'X.a')."""
    repo = _make_repo({
        "m.py": """
            class X:
                def b(self):
                    pass
                def a(self):
                    self.b()
        """,
    })
    try:
        idx = CallGraphIndexer(repo)
        # Direct key "a" doesn't exist in _forward; suffix fallback matches "X.a"
        callees = idx.get_callees("a")
        assert len(callees) > 0, "Expected callees via suffix fallback for 'a'"
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ─── Case 11: suffix fallback in get_callers ────────────────────────────────

def test_get_callers_suffix_fallback():
    """Bare callee name matches qualified reverse index key (e.g. 'helper' → 'X.helper')."""
    repo = _make_repo({
        "m.py": """
            class X:
                def helper(self):
                    pass
                def caller(self):
                    self.helper()
        """,
    })
    try:
        idx = CallGraphIndexer(repo)
        # "X.helper" is in _reverse (as callee); "helper" bare should match via suffix
        callers = idx.get_callers("helper")
        assert len(callers) > 0, "Expected callers via suffix fallback for 'helper'"
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ─── Case 11b: M1 — bare-name reverse index dedups shared symbols ──────────

def test_bare_index_dedups_shared_symbols():
    """M1: a symbol that is BOTH caller and callee appears once in _bare_index.

    Regression: the build-time bare-name index iterated (*_forward, *_reverse)
    without dedup, so a method that both calls and is called (X.a below) was
    registered twice -> suffix-fallback lookups returned doubled edges.
    """
    repo = _make_repo({
        "m.py": """
            class X:
                def a(self):      # caller of b, callee of c
                    self.b()

                def b(self):
                    pass

                def c(self):
                    self.a()
        """,
    })
    try:
        idx = CallGraphIndexer(repo)
        idx.build()
        assert idx._bare_index["a"] == ["X.a"], (
            f"Expected bare index ['X.a'] (deduped), got {idx._bare_index['a']}"
        )
        # suffix fallback on "a" must not double edges
        callees = idx.get_callees("a")
        assert [(e.callee_symbol, e.caller_symbol) for e in callees] == [("X.b", "X.a")], (
            f"Expected single callee edge, got "
            f"{[(e.callee_symbol, e.caller_symbol) for e in callees]}"
        )
        callers = idx.get_callers("a")
        assert [(e.caller_symbol, e.callee_symbol) for e in callers] == [("X.c", "X.a")], (
            f"Expected single caller edge, got "
            f"{[(e.caller_symbol, e.callee_symbol) for e in callers]}"
        )
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ─── Case 12: get_callees with file_path filter ─────────────────────────────

def test_get_callees_file_path_filter():
    """file_path parameter in get_callees filters by caller_file."""
    repo = _make_repo({
        "a.py": """
            def helper():
                pass
            def caller_a():
                helper()
        """,
    })
    try:
        idx = CallGraphIndexer(repo)
        # callees of "caller_a" have caller_file="a.py"
        callees = idx.get_callees("caller_a", file_path="a.py")
        assert len(callees) >= 1
        assert all(e.caller_file == "a.py" for e in callees)
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ─── Case 13: get_callers with file_path filter ─────────────────────────────

def test_get_callers_file_path_filter():
    """file_path parameter in get_callers filters by callee_file."""
    repo = _make_repo({
        "helper.py": """
            def helper():
                pass
        """,
        "main.py": """
            from helper import helper
            def caller():
                helper()
        """,
    })
    try:
        idx = CallGraphIndexer(repo)
        # callers of "helper" have callee_file="helper.py" (after _resolve_callees)
        callers = idx.get_callers("helper", file_path="helper.py")
        assert len(callers) >= 1
        assert all(e.callee_file == "helper.py" for e in callers)
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ─── Case 14: get_related_symbols with callers + depth=2 ────────────────────

def test_get_related_symbols_depth2():
    """depth=2 covers extra_callee expansion and caller/callee file scoring."""
    repo = _make_repo({
        "svc.py": """
            def deep_leaf():
                pass
            def leaf():
                deep_leaf()
            def middle():
                leaf()
            def top():
                middle()
        """,
    })
    try:
        idx = CallGraphIndexer(repo)
        # "middle" has both callers (top) and callees (leaf → deep_leaf)
        result = idx.get_related_symbols("middle", depth=2)
        assert result["symbol"] == "middle"
        callee_syms = {c["symbol"] for c in result["callees"]}
        assert "leaf" in callee_syms, f"Expected 'leaf' in callees, got {callee_syms}"
        caller_syms = {c["symbol"] for c in result["callers"]}
        assert "top" in caller_syms, f"Expected 'top' in callers, got {caller_syms}"
        assert len(result["next_read_candidates"]) > 0
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ─── Case 15: non-standard call forms are silently skipped ──────────────────

def test_non_standard_call_forms():
    """Calls with non-Name/non-Attribute func should not crash the indexer."""
    repo = _make_repo({
        "expr.py": """
            def f():
                (lambda: 42)()
        """,
    })
    try:
        idx = CallGraphIndexer(repo)
        result = idx.get_related_symbols("f")
        assert result["symbol"] == "f"
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ─── Case 16: attribute call where base of chain is not Name ────────────────

def test_attribute_chain_non_name_base():
    """foo().bar() has an Attribute chain ending in Call, not Name → L387."""
    repo = _make_repo({
        "expr.py": """
            def factory():
                return {}
            def f():
                factory().get("key")
        """,
    })
    try:
        idx = CallGraphIndexer(repo)
        result = idx.get_related_symbols("f")
        assert result["symbol"] == "f"
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ─── Case 17: obj.method() call (non-self attribute base) → L403 ────────────

def test_obj_method_call():
    """obj.method() should create a CallEdge with lower confidence (0.5)."""
    repo = _make_repo({
        "m.py": """
            class Helper:
                def do_it(self):
                    pass
            def f():
                h = Helper()
                h.do_it()
        """,
    })
    try:
        idx = CallGraphIndexer(repo)
        callees = idx.get_callees("f")
        assert len(callees) > 0
        # h.do_it() → confidence 0.5 (not self.)
        assert any(e.confidence == 0.5 for e in callees)
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ─── Case 18: Exception during file indexing (non-SyntaxError) → L114-115 ───

def test_index_file_os_error(monkeypatch):
    """An OSError during _index_file is caught and logged."""
    import external_llm.agent.call_graph as cg
    original_index = cg.CallGraphIndexer._index_file

    def broken_index(self, path):
        if "broken" in str(path):
            raise OSError("Permission denied")
        return original_index(self, path)

    monkeypatch.setattr(cg.CallGraphIndexer, "_index_file", broken_index)
    repo = _make_repo({
        "broken.py": "x = 1",
        "good.py": "def ok(): pass",
    })
    try:
        idx = cg.CallGraphIndexer(repo)
        idx.build()  # Should not raise — OSError is caught by generic except
        assert idx._built
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ─── Case 19: MULTILANG_CALLGRAPH import failure → L121-122 ─────────────────

def test_ml_cg_import_failure():
    """When config.MULTILANG_CALLGRAPH is absent, _ML_CG defaults to False."""
    import importlib

    import config as top_config

    saved = getattr(top_config, "MULTILANG_CALLGRAPH", None)
    try:
        if hasattr(top_config, "MULTILANG_CALLGRAPH"):
            del top_config.MULTILANG_CALLGRAPH
        repo = _make_repo({"empty.py": "x = 1"})
        try:
            idx = CallGraphIndexer(repo)
            idx.build()
            assert idx._built
        finally:
            shutil.rmtree(repo, ignore_errors=True)
    finally:
        if saved is not None:
            top_config.MULTILANG_CALLGRAPH = saved
        else:
            # Re-import config to restore state
            importlib.reload(top_config)


# ─── Case 16: unknown symbol returns empty gracefully ───────────────────────

def test_unknown_symbol_empty():
    repo = _make_repo({"empty.py": "x = 1\n"})
    try:
        idx = CallGraphIndexer(repo)
        callees = idx.get_callees("nonexistent_func")
        assert callees == []
        result = idx.get_related_symbols("nonexistent_func")
        assert result["node"] is None
        assert result["callees"] == []
        assert result["callers"] == []
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ─── Case 20: TS file indexing via build() ─────────────────────────────────

def test_ts_file_indexing():
    """build() indexes .ts files when MULTILANG_CALLGRAPH is True."""
    repo = _make_repo({
        "util.ts": """
            function greet(name: string): string {
                return "Hello " + name;
            }
        """,
        "main.py": """
            def f():
                pass
        """,
    })
    try:
        idx = CallGraphIndexer(repo)
        idx.build()
        assert "greet" in idx._nodes, f"Expected 'greet' in nodes, got {list(idx._nodes.keys())}"
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ─── Case 21: TS file with class methods ────────────────────────────────────

def test_ts_class_method_indexing():
    """TS class methods are registered as ClassName.method in _nodes."""
    repo = _make_repo({
        "app.ts": """
            class Calculator {
                add(x: number, y: number): number {
                    return x + y;
                }
                multiply(x: number, y: number): number {
                    return x * y;
                }
            }
        """,
        "main.py": "x = 1",
    })
    try:
        idx = CallGraphIndexer(repo)
        idx.build()
        assert "Calculator.add" in idx._nodes
        assert "Calculator.multiply" in idx._nodes
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ─── Case 22: TS call edges ────────────────────────────────────────────────

def test_ts_call_edge_indexing():
    """TS call sites create forward/reverse edges."""
    repo = _make_repo({
        "app.ts": """
            function helper(): void {
                // nothing
            }
            function caller(): void {
                helper();
            }
        """,
        "main.py": "x = 1",
    })
    try:
        idx = CallGraphIndexer(repo)
        idx.build()
        assert "caller" in idx._nodes
        assert "helper" in idx._nodes
        callees = idx.get_callees("caller")
        assert len(callees) > 0
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ─── Case 23: _rel ValueError path ──────────────────────────────────────────

def test_rel_value_error():
    """When relative_to raises ValueError, _rel returns the raw path str."""
    import unittest

    original_relative_to = Path.relative_to

    def broken_relative_to(self, *args, **kwargs):
        if "badpath" in str(self):
            raise ValueError("path is not relative")
        return original_relative_to(self, *args, **kwargs)

    with unittest.mock.patch.object(Path, "relative_to", broken_relative_to):
        repo = _make_repo({"badpath/good.py": "def f(): pass"})
        try:
            idx = CallGraphIndexer(repo)
            path = Path(repo) / "badpath" / "good.py"
            result = idx._rel(path)
            assert isinstance(result, str)
            assert "badpath" in result
        finally:
            shutil.rmtree(repo, ignore_errors=True)


# ─── Case 24: file walk limit break (patched) ───────────────────────────────

def test_walk_py_files_limit():
    """When MAX_PY_FILES is reached, walker breaks and returns."""
    import external_llm.agent.call_graph as cg
    old_limit = cg._MAX_PY_FILES
    try:
        cg._MAX_PY_FILES = 2
        repo = _make_repo({
            "a.py": "x = 1",
            "b.py": "y = 2",
            "c.py": "z = 3",
        })
        try:
            files = cg._walk_py_files(Path(repo))
            assert len(files) == 2, f"Expected 2 files, got {len(files)}"
        finally:
            shutil.rmtree(repo, ignore_errors=True)
    finally:
        cg._MAX_PY_FILES = old_limit


# ─── Case 25: TS file walk limit ────────────────────────────────────────────

def test_walk_ts_js_files_limit():
    """When MAX_TS_FILES is reached, walker returns early."""
    import external_llm.agent.call_graph as cg
    old_limit = cg._MAX_TS_FILES
    try:
        cg._MAX_TS_FILES = 2
        repo = _make_repo({
            "a.ts": "let x = 1;",
            "b.ts": "let y = 2;",
            "c.ts": "let z = 3;",
        })
        try:
            files = cg._walk_ts_js_files(Path(repo))
            assert len(files) == 2, f"Expected 2 files, got {len(files)}"
        finally:
            shutil.rmtree(repo, ignore_errors=True)
    finally:
        cg._MAX_TS_FILES = old_limit


# ─── Case 26: TS file indexing exception → L129-130 ────────────────────────

def test_ts_file_indexing_exception():
    """Exception during TS file indexing is caught without crashing build()."""
    import unittest

    import external_llm.agent.call_graph as cg

    def broken_index_ts(self, path):
        if "good" not in str(path):
            raise RuntimeError("TS parse failed")
        return original_index_ts(self, path)

    original_index_ts = cg.CallGraphIndexer._index_ts_file
    with unittest.mock.patch.object(
        cg.CallGraphIndexer, "_index_ts_file", broken_index_ts
    ):
        repo = _make_repo({
            "broken.ts": "let x = 1;",
            "good.ts": "function ok(): void { return; }",
            "main.py": "def f(): pass",
        })
        try:
            idx = cg.CallGraphIndexer(repo)
            idx.build()  # Should not raise — exception is caught
            assert idx._built
            # "ok" from good.ts should still be indexed
            assert "ok" in idx._nodes
        finally:
            shutil.rmtree(repo, ignore_errors=True)


# ─── Case 27: TS call_site with empty caller/callee → L449 ──────────────────

def test_ts_call_site_empty_caller():
    """Call sites with empty caller or callee are skipped (continue)."""
    import unittest
    from unittest.mock import PropertyMock

    from external_llm.agent.call_graph import CallGraphIndexer

    ts_module = MagicMock()
    ts_module.functions = []
    ts_module.classes = []
    # Add a call_site with empty caller (falsy)
    empty_caller = MagicMock()
    type(empty_caller).caller = PropertyMock(return_value="")
    type(empty_caller).callee = PropertyMock(return_value="target_func")
    type(empty_caller).receiver = PropertyMock(return_value=None)
    type(empty_caller).line = PropertyMock(return_value=1)
    type(empty_caller).is_method_call = PropertyMock(return_value=False)
    # Add a normal call_site
    normal_cs = MagicMock()
    type(normal_cs).caller = PropertyMock(return_value="caller_fn")
    type(normal_cs).callee = PropertyMock(return_value="target_func")
    type(normal_cs).receiver = PropertyMock(return_value=None)
    type(normal_cs).line = PropertyMock(return_value=2)
    type(normal_cs).is_method_call = PropertyMock(return_value=False)
    ts_module.call_sites = [empty_caller, normal_cs]

    from external_llm.editor.semantic.ts_semantic_tracer import TSSemanticTracer
    with unittest.mock.patch.object(
        TSSemanticTracer, "analyze_core", return_value=ts_module
    ):
        repo = _make_repo({
            "app.ts": "function caller_fn(): void { target_func(); }",
            "main.py": "x = 1",
        })
        try:
            idx = CallGraphIndexer(repo)
            idx.build()
            # The empty-caller call site should be skipped (continue)
            # The normal call site should create an edge
            callees = idx.get_callees("caller_fn")
            assert len(callees) == 1
        finally:
            shutil.rmtree(repo, ignore_errors=True)


# ─── Case 28: bare-name reverse index (M1) ───────────────────────────────────

def test_bare_name_index_supports_suffix_lookup():
    """M1: suffix fallback resolves via the build-time bare-name index."""
    repo = _make_repo({
        "m.py": """
            class X:
                def b(self):
                    pass
                def a(self):
                    self.b()
        """,
    })
    try:
        idx = CallGraphIndexer(repo)
        idx.build()  # explicit — the index is lazy (first access triggers build)
        # build() indexes both forward ("X.a") and reverse ("X.b") keys
        assert "a" in idx._bare_index and "X.a" in idx._bare_index["a"]
        assert "b" in idx._bare_index and "X.b" in idx._bare_index["b"]
        # suffix fallback still resolves in both directions
        assert len(idx.get_callees("a")) > 0
        assert len(idx.get_callers("b")) > 0
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_bare_name_index_rebuilt_after_invalidate():
    """M1: invalidate clears the bare index; the next access rebuilds it."""
    repo = _make_repo({
        "m.py": """
            class X:
                def b(self):
                    pass
                def a(self):
                    self.b()
        """,
    })
    try:
        idx = CallGraphIndexer(repo)
        assert len(idx.get_callees("a")) > 0  # builds + populates bare index
        idx.invalidate()
        assert idx._bare_index == {}
        # rebuild on next access restores the bare index and suffix fallback
        assert len(idx.get_callees("a")) > 0
        assert "X.a" in idx._bare_index["a"]
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ─── invalidate_files(): incremental re-indexing after writes ───────────────

def test_invalidate_files_reindexes_changed_file():
    repo = _make_repo({
        "a.py": "def foo(): pass\n",
        "b.py": "def bar(): foo()\n",
    })
    try:
        idx = CallGraphIndexer(repo)
        assert "foo" in {e.callee_symbol for e in idx.get_callees("bar")}
        # Edit b.py: bar() now also calls a NEW function
        (Path(repo) / "b.py").write_text(textwrap.dedent("""
            def bar():
                foo()
                baz()
        """))
        idx.invalidate_files(["b.py"])
        callees = {e.callee_symbol for e in idx.get_callees("bar")}
        assert "baz" in callees
        assert "foo" in callees
        # a.py untouched — its node is intact
        assert idx._nodes["foo"].file == "a.py"
        # bare index rebuilt — suffix lookup resolves the new symbol
        assert "baz" in idx._bare_index.get("baz", [])
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_invalidate_files_removes_deleted_file():
    repo = _make_repo({
        "a.py": "def foo(): pass\n",
        "b.py": "def bar(): foo()\n",
    })
    try:
        idx = CallGraphIndexer(repo)
        assert idx.get_callers("foo")
        (Path(repo) / "a.py").unlink()
        idx.invalidate_files(["a.py"])
        # a.py's edges are gone; b.py's bar->foo edge legitimately remains
        callers = idx.get_callers("foo")
        assert len(callers) == 1
        assert callers[0].caller_symbol == "bar"
        assert "foo" not in idx._nodes        # node gone
        assert "bar" in idx._nodes            # b.py untouched
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_invalidate_files_reassigns_shadowed_node():
    # Two files define the same symbol; the lexicographically first file
    # (a.py) owns the node.  Deleting a.py must move ownership to b.py —
    # matching what a full rebuild would produce.
    repo = _make_repo({
        "a.py": "def dup(): pass\n",
        "b.py": "def dup(): pass\n",
    })
    try:
        idx = CallGraphIndexer(repo)
        idx.get_callers("dup")  # lazy build
        assert idx._nodes["dup"].file == "a.py"
        (Path(repo) / "a.py").unlink()
        idx.invalidate_files(["a.py"])
        assert idx._nodes["dup"].file == "b.py"
        # and removing b.py afterwards removes the node entirely
        (Path(repo) / "b.py").unlink()
        idx.invalidate_files(["b.py"])
        assert "dup" not in idx._nodes
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_invalidate_files_reassignment_follows_walk_order_not_lexicographic():
    # Regression: reassignment used min(srcs) — lexicographic — but build()
    # visits ROOT files before subdirectory files (os.walk yields the root
    # dir's filenames first, then descends). b.py and c.py are therefore
    # visited before a/x.py despite "a/x.py" < "b.py". After deleting the
    # owner (b.py) the next full-rebuild winner is c.py, not a/x.py.
    repo = _make_repo({
        "a/x.py": "def dup(): pass\n",
        "b.py": "def dup(): pass\n",
        "c.py": "def dup(): pass\n",
    })
    try:
        idx = CallGraphIndexer(repo)
        idx.get_callers("dup")  # lazy build
        assert idx._nodes["dup"].file == "b.py"  # root files first
        (Path(repo) / "b.py").unlink()
        idx.invalidate_files(["b.py"])
        assert idx._nodes["dup"].file == "c.py", (
            "reassignment must mirror walk order (root c.py before a/x.py), "
            f"got {idx._nodes['dup'].file!r}"
        )
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_invalidate_files_owner_reclaims_when_redefined():
    # Regression: _register_node claimed ownership only when the symbol was
    # absent, so once a.py's definition was edited away (ownership moved to
    # b.py), editing a.py BACK never reclaimed — incremental ownership
    # diverged from a full rebuild forever after. The rank-based claim must
    # restore a.py (earlier in walk order) on re-index.
    repo = _make_repo({
        "a.py": "def dup(): pass\n",
        "b.py": "def dup(): pass\n",
    })
    try:
        idx = CallGraphIndexer(repo)
        idx.get_callers("dup")  # lazy build
        assert idx._nodes["dup"].file == "a.py"

        # Edit the owner to drop the definition -> ownership moves to b.py.
        (Path(repo) / "a.py").write_text("x = 1\n")
        idx.invalidate_files(["a.py"])
        assert idx._nodes["dup"].file == "b.py"

        # Edit a.py back -> it must reclaim (a full rebuild would agree).
        (Path(repo) / "a.py").write_text("def dup(): pass\n")
        idx.invalidate_files(["a.py"])
        assert idx._nodes["dup"].file == "a.py"

        # The old owner's ownership record is gone: removing b.py must leave
        # a.py as owner, not resurrect a ghost node.
        (Path(repo) / "b.py").unlink()
        idx.invalidate_files(["b.py"])
        assert idx._nodes["dup"].file == "a.py"
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_invalidate_files_keeps_unrelated_edges_intact():
    repo = _make_repo({
        "a.py": "def foo(): pass\ndef alpha(): foo()\n",
        "b.py": "def bar(): foo()\n",
    })
    try:
        idx = CallGraphIndexer(repo)
        idx.get_callers("foo")  # lazy build
        alpha_before = idx._forward["alpha"][0]
        foo_node_before = idx._nodes["foo"]
        (Path(repo) / "b.py").write_text(textwrap.dedent("""
            def bar():
                foo()
                extra()
        """))
        idx.invalidate_files(["b.py"])
        # a.py's edge and node are the SAME objects — no rebuild touched them
        assert idx._forward["alpha"][0] is alpha_before
        assert idx._nodes["foo"] is foo_node_before
        assert "extra" in {e.callee_symbol for e in idx.get_callees("bar")}
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_invalidate_files_identity_removal_keeps_equal_edges():
    # Both files call foo() — value-equal but distinct edge objects; removing
    # one file must not remove the other's edge (identity-based removal).
    repo = _make_repo({
        "a.py": "def foo(): pass\ndef alpha(): foo()\n",
        "b.py": "def beta(): foo()\n",
    })
    try:
        idx = CallGraphIndexer(repo)
        idx.get_callers("foo")  # lazy build
        assert len(idx._reverse["foo"]) == 2
        (Path(repo) / "a.py").unlink()
        idx.invalidate_files(["a.py"])
        remaining = idx._reverse["foo"]
        assert len(remaining) == 1
        assert remaining[0].caller_symbol == "beta"
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_invalidate_files_noop_when_not_built():
    repo = _make_repo({"a.py": "def foo(): pass\n"})
    try:
        idx = CallGraphIndexer(repo)
        assert not idx._built
        idx.invalidate_files(["a.py"])  # must not raise, must not build
        assert not idx._built
        # the next real access still builds correctly
        assert idx.get_callers("foo") == []
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_invalidate_files_handles_leading_slash_and_absolute_root():
    repo = _make_repo({
        "a.py": "def foo(): pass\ndef alpha(): foo()\n",
    })
    try:
        idx = CallGraphIndexer(repo)
        idx.get_callers("foo")  # lazy build
        (Path(repo) / "a.py").write_text(textwrap.dedent("""
            def foo(): pass
            def alpha(): foo()
            def gamma(): pass
        """))
        idx.invalidate_files(["/a.py"])  # facade-style leading slash
        assert "gamma" in idx._nodes
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ─── Property: incremental == full rebuild ───────────────────────────────────

def _cg_snapshot(idx):
    """Field-level snapshot of everything ownership/invalidation touches."""
    nodes = {s: (n.file, n.line, n.kind) for s, n in sorted(idx._nodes.items())}
    fwd = {s: {_edge_key(e) for e in es} for s, es in idx._forward.items()}
    rev = {s: {_edge_key(e) for e in es} for s, es in idx._reverse.items()}
    file_nodes = {rel: sorted(syms) for rel, syms in sorted(idx._file_nodes.items())}
    def_sources = {
        s: dict(sorted(srcs.items())) for s, srcs in sorted(idx._def_sources.items())
    }
    return nodes, fwd, rev, file_nodes, def_sources


def _edge_key(e):
    """Hashable full-field key (call_args is a list — astuple would not do)."""
    return (
        e.caller_symbol, e.caller_file, e.caller_line,
        e.callee_symbol, e.callee_display, e.callee_file, e.callee_line,
        e.confidence, str(e.edge_kind), tuple(e.call_args),
    )


def test_incremental_index_matches_full_rebuild_under_random_edits():
    """Property test: after an arbitrary edit sequence (create/modify/delete,
    root + nested dirs, overlapping definitions), the incrementally-updated
    index must be IDENTICAL to a fresh full rebuild — node ownership
    (file/line/kind), edge sets, per-file contribution maps.  Catches any
    divergence between invalidate_files' ownership rules and build()'s walk
    order: lexicographic-vs-walk-order reassignment and once-only-vs-
    reclaimable claims were both found by this shape of test."""
    import random

    rng = random.Random(20260811)  # fixed seed — deterministic, never flaky
    repo = _make_repo({})
    try:
        DEF_POOL = ["alpha", "beta", "gamma", "delta", "epsilon"]
        CALL_POOL = [("a_call", "alpha"), ("b_call", "beta"), ("c_call", "gamma")]
        RELS = ["a.py", "b.py", "sub/x.py", "sub/y.py", "deep/z.py"]

        def src(defs, calls=()):
            body = [f"def {d}(): pass" for d in defs]
            body += [f"def {c}(): {t}()" for c, t in calls]
            return "\n".join(body) + "\n"

        def random_src():
            defs = [d for d in DEF_POOL if rng.random() < 0.4]
            calls = [c for c in CALL_POOL if rng.random() < 0.3]
            return src(defs, calls) if defs or calls else "x = 1\n"

        files = {
            "a.py": src(["alpha", "beta"], [("a_call", "alpha")]),
            "b.py": src(["beta"]),
            "sub/x.py": src(["gamma", "alpha"]),
        }
        for rel, content in files.items():
            p = Path(repo) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)

        idx = CallGraphIndexer(repo)
        idx.get_callers("alpha")  # lazy full build — the baseline
        for _ in range(30):
            rel = rng.choice(RELS)
            if rng.random() < 0.2 and rel in files:
                del files[rel]
                (Path(repo) / rel).unlink()
            else:
                files[rel] = random_src()
                p = Path(repo) / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(files[rel])
            idx.invalidate_files([rel])

        # Fresh indexer, full rebuild — must agree field-for-field.  The
        # shared per-root walk cache (TTL 30s) would hand the fresh build the
        # t0 file list; real writes go through ToolRegistry which calls
        # invalidate_walk_caches() post-write, so mirror that here.
        from external_llm.agent._shared_utils import invalidate_walk_caches

        invalidate_walk_caches()
        fresh = CallGraphIndexer(repo)
        fresh.build()
        assert _cg_snapshot(idx) == _cg_snapshot(fresh), (
            "incremental index diverged from a full rebuild after random edits"
        )
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_invalidate_files_skips_walk_pruned_directory():
    """invalidate_files must not index paths under walk-pruned dirs (B1).

    build() walks via _walk_should_skip_dir (node_modules/.venv/...), so the
    incremental path must re-apply the SAME pruning or it indexes files a
    fresh build() would drop -- the graph diverges from a clean rebuild.
    """
    repo = _make_repo({
        "a.py": "def a():\n    b()\n",
        "b.py": "def b():\n    pass\n",
        "node_modules/pkg/v.py": "def vendored():\n    pass\n",
    })
    try:
        idx = CallGraphIndexer(repo)
        idx.build()
        before = sorted(idx._nodes)
        assert before == ["a", "b"], before

        idx.invalidate_files(["node_modules/pkg/v.py"])
        assert sorted(idx._nodes) == before, (
            "incremental path injected a pruned-dir symbol"
        )

        idx.build()
        assert sorted(idx._nodes) == before
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ─── A2 (2026-08-12): single-walk traversal parity ──────────────────────────

def test_single_walk_parity_with_reference_traversal():
    """The merged single-BFS-pass ``_index_file`` (A2) must produce exactly
    what the old three-walk traversal produced: a class-name pass, a def pass
    and a call pass, each a full ``ast.walk``.  The reference below re-
    implements the old traversal; the source exercises the risky cases:
    plain class methods, a nested class, a decorated method, a nested
    function, and module-level functions.
    """
    src = """
        class Outer:
            def method(self):
                self.helper()
                inner_top()

            def helper(self):
                pass

            class Inner:
                def nested(self):
                    plain_call()

            @staticmethod
            def decorated():
                top_level_call()

        def module_fn():
            def local():
                local_call()
            local()
            Outer().method()
    """
    repo = _make_repo({"sample.py": src})
    try:
        idx = CallGraphIndexer(repo)
        idx.build()

        # Reference: the OLD three-walk traversal (HEAD before A2).
        tree = ast.parse(textwrap.dedent(src))
        class_names: dict[int, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        class_names[id(child)] = node.name

        ref_nodes: dict[str, tuple[str, int, str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                cn = class_names.get(id(node))
                symbol = f"{cn}.{node.name}" if cn else node.name
                kind = (
                    "method"
                    if cn
                    else (
                        "async_function"
                        if isinstance(node, ast.AsyncFunctionDef)
                        else "function"
                    )
                )
                ref_nodes[symbol] = ("sample.py", node.lineno, kind)

        ref_edges: dict[str, list[str]] = {}
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            cn = class_names.get(id(node))
            caller = f"{cn}.{node.name}" if cn else node.name
            seen: set[str] = set()
            for child in _iter_calls(node):
                edge = idx._parse_call(
                    child, caller, "sample.py", child.lineno, cn
                )
                if edge and edge.callee_display not in seen:
                    seen.add(edge.callee_display)
                    ref_edges.setdefault(caller, []).append(edge.callee_display)

        got_nodes = {
            sym: (n.file, n.line, n.kind)
            for sym, n in idx._nodes.items()
        }
        got_edges = {
            sym: [e.callee_display for e in edges]
            for sym, edges in idx._forward.items()
        }
        assert got_nodes == ref_nodes, (got_nodes, ref_nodes)
        assert got_edges == ref_edges, (got_edges, ref_edges)
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ─── A4 (2026-08-12): incremental invalidation parity ────────────────────────

def test_invalidate_files_parity_with_full_rebuild():
    """A4: after a mixed edit/delete/add batch, the incrementally updated
    graph must equal a fresh build() over the same tree — node ownership,
    forward/reverse edges (including resolved callee_file/callee_line) and
    the bare-name suffix index.
    """
    repo = _make_repo({
        "pkg/a.py": """
            def shared():
                pass

            def from_a():
                shared()
                other()
        """,
        "pkg/b.py": """
            def other():
                pass

            def from_b():
                other()
        """,
    })
    try:
        idx = CallGraphIndexer(repo)
        idx.build()

        # Edit a.py: from_a disappears, new_a appears (calls shared + from_b).
        (Path(repo) / "pkg/a.py").write_text(textwrap.dedent("""
            def shared():
                pass

            def new_a():
                shared()
                from_b()
        """))
        # Delete b.py: other/from_b vanish; new_a -> from_b must clear.
        (Path(repo) / "pkg/b.py").unlink()
        # Add c.py.
        (Path(repo) / "pkg/c.py").write_text(textwrap.dedent("""
            def from_c():
                shared()
        """))
        # The shared walkers cache per-root for _WALK_CACHE_TTL; a fresh
        # indexer must re-walk to see c.py (otherwise ref misses it).
        invalidate_walk_caches()

        idx.invalidate_files(["pkg/a.py", "pkg/b.py", "pkg/c.py"])

        # Reference: full rebuild in a fresh indexer over the same tree.
        ref = CallGraphIndexer(repo)
        ref.build()

        assert idx._nodes == ref._nodes, (idx._nodes, ref._nodes)
        assert idx._forward == ref._forward, (idx._forward, ref._forward)
        assert idx._reverse == ref._reverse, (idx._reverse, ref._reverse)
        assert idx._bare_index == ref._bare_index, (
            idx._bare_index, ref._bare_index,
        )
        # Spot-check the resolved callee pointers the incremental path is
        # responsible for updating: shared's edges point at pkg/a.py; the
        # vanished from_b callee is cleared on new_a's edge to it.
        by_display = {e.callee_display: e for e in idx._forward["new_a"]}
        assert by_display["from_b"].callee_file is None  # from_b vanished
        assert by_display["shared"].callee_file == "pkg/a.py"
        for e in idx._forward["from_c"]:
            assert e.callee_file == "pkg/a.py", e  # shared lives in a.py
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_invalidate_files_new_dir_recomputes_ranks_only_when_needed():
    """A4: invalidating a file inside an already-known directory must not
    recompute dir ranks, while a file in a brand-new directory must.
    """
    repo = _make_repo({
        "a.py": "def a():\n    pass\n",
        "sub/b.py": "def b():\n    pass\n",
    })
    try:
        idx = CallGraphIndexer(repo)
        idx.build()
        ranks_before = idx._dir_ranks

        # Edit inside a known directory -> ranks object untouched.
        (Path(repo) / "a.py").write_text("def a():\n    return 1\n")
        idx.invalidate_files(["a.py"])
        assert idx._dir_ranks is ranks_before

        # Create a file in a brand-new directory -> ranks recomputed.
        (Path(repo) / "newdir").mkdir()
        (Path(repo) / "newdir/c.py").write_text("def c():\n    pass\n")
        idx.invalidate_files(["newdir/c.py"])
        assert idx._dir_ranks is not ranks_before
        assert "newdir" in idx._dir_ranks
    finally:
        shutil.rmtree(repo, ignore_errors=True)
