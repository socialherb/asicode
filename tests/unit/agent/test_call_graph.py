"""Unit tests for call_graph.CallGraphIndexer."""
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

from external_llm.agent.call_graph import (
    CallGraphIndexer,
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
        import shutil; shutil.rmtree(repo, ignore_errors=True)


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
        import shutil; shutil.rmtree(repo, ignore_errors=True)


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
        import shutil; shutil.rmtree(repo, ignore_errors=True)


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
        import shutil; shutil.rmtree(repo, ignore_errors=True)


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
        import shutil; shutil.rmtree(repo, ignore_errors=True)


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
        import shutil; shutil.rmtree(repo, ignore_errors=True)


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
        import shutil; shutil.rmtree(repo, ignore_errors=True)


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
        import shutil; shutil.rmtree(repo, ignore_errors=True)


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
        import shutil; shutil.rmtree(repo, ignore_errors=True)


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
        import shutil; shutil.rmtree(repo, ignore_errors=True)


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
        import shutil; shutil.rmtree(repo, ignore_errors=True)


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
        import shutil; shutil.rmtree(repo, ignore_errors=True)


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
        import shutil; shutil.rmtree(repo, ignore_errors=True)


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
        import shutil; shutil.rmtree(repo, ignore_errors=True)


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
        import shutil; shutil.rmtree(repo, ignore_errors=True)


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
        import shutil; shutil.rmtree(repo, ignore_errors=True)


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
        import shutil; shutil.rmtree(repo, ignore_errors=True)


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
            import shutil; shutil.rmtree(repo, ignore_errors=True)
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
        import shutil; shutil.rmtree(repo, ignore_errors=True)


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
        import shutil; shutil.rmtree(repo, ignore_errors=True)


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
        import shutil; shutil.rmtree(repo, ignore_errors=True)


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
        import shutil; shutil.rmtree(repo, ignore_errors=True)


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
            import shutil; shutil.rmtree(repo, ignore_errors=True)


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
            import shutil; shutil.rmtree(repo, ignore_errors=True)
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
            import shutil; shutil.rmtree(repo, ignore_errors=True)
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
            import shutil; shutil.rmtree(repo, ignore_errors=True)


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
            import shutil; shutil.rmtree(repo, ignore_errors=True)
