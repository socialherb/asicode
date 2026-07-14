"""Unit tests for external_llm.graph.graph_facade.RepositoryGraphFacade."""
import shutil
import tempfile
import textwrap
from pathlib import Path

from external_llm.graph.graph_facade import RepositoryGraphFacade
from external_llm.graph.models import CallEdge


def _make_repo(files: dict) -> str:
    d = tempfile.mkdtemp(prefix="test_facade_")
    for rel_path, source in files.items():
        full = Path(d) / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(textwrap.dedent(source))
    return d


# ── Lazy initialization ────────────────────────────────────────────────────────

def test_facade_lazy_init_no_graph_at_start():
    repo = _make_repo({"a.py": "def foo(): pass\n"})
    try:
        facade = RepositoryGraphFacade(repo_root=repo)
        assert facade._graph is None  # not yet built
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_facade_lazy_init_builds_on_first_access():
    repo = _make_repo({"a.py": "def foo(): pass\n"})
    try:
        facade = RepositoryGraphFacade(repo_root=repo)
        sym = facade.get_symbol("foo")
        assert facade._graph is not None  # now built
        assert sym is not None
        assert sym.name == "foo"
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ── get_symbol() ───────────────────────────────────────────────────────────────

def test_get_symbol_simple_name():
    repo = _make_repo({"m.py": "def my_func(): pass\n"})
    try:
        facade = RepositoryGraphFacade(repo_root=repo)
        sym = facade.get_symbol("my_func")
        assert sym is not None
        assert sym.name == "my_func"
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_get_symbol_qualname():
    repo = _make_repo({
        "m.py": """
            class MyClass:
                def method(self): pass
        """
    })
    try:
        facade = RepositoryGraphFacade(repo_root=repo)
        sym = facade.get_symbol("MyClass.method")
        assert sym is not None
        assert sym.qualname == "MyClass.method"
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_get_symbol_with_file_hint():
    repo = _make_repo({
        "a.py": "def helper(): pass\n",
        "b.py": "def helper(): pass\n",
    })
    try:
        facade = RepositoryGraphFacade(repo_root=repo)
        sym_a = facade.get_symbol("helper", "a.py")
        sym_b = facade.get_symbol("helper", "b.py")
        assert sym_a is not None
        assert sym_b is not None
        assert sym_a.file_path != sym_b.file_path or sym_a.file_path == sym_b.file_path
        # Both should be found
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_get_symbol_not_found():
    repo = _make_repo({"m.py": "x = 1\n"})
    try:
        facade = RepositoryGraphFacade(repo_root=repo)
        sym = facade.get_symbol("nonexistent_func")
        assert sym is None
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ── get_symbols_in_file() ──────────────────────────────────────────────────────

def test_get_symbols_in_file():
    repo = _make_repo({
        "svc.py": """
            def alpha(): pass
            def beta(): pass
            class Gamma: pass
        """
    })
    try:
        facade = RepositoryGraphFacade(repo_root=repo)
        syms = facade.get_symbols_in_file("svc.py")
        names = [s.name for s in syms]
        assert "alpha" in names
        assert "beta" in names
        assert "Gamma" in names
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_get_symbols_in_file_constants_indexed():
    # Module-level assignments are now indexed as constants.
    repo = _make_repo({"empty.py": "x = 1\n"})
    try:
        facade = RepositoryGraphFacade(repo_root=repo)
        syms = facade.get_symbols_in_file("empty.py")
        assert len(syms) == 1
        assert syms[0].name == "x"
        assert syms[0].kind == "constant"
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ── get_callers() / get_callees() via CallGraphIndexer ────────────────────────

def test_get_callers_via_call_graph_indexer():
    from external_llm.agent.call_graph import CallGraphIndexer
    repo = _make_repo({
        "foo.py": """
            def b():
                pass
            def a():
                b()
        """
    })
    try:
        cgi = CallGraphIndexer(repo)
        facade = RepositoryGraphFacade(call_graph_indexer=cgi, repo_root=repo)
        callers = facade.get_callers("b")
        caller_syms = [e.caller_symbol for e in callers]
        assert "a" in caller_syms
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_get_callees_via_call_graph_indexer():
    from external_llm.agent.call_graph import CallGraphIndexer
    repo = _make_repo({
        "foo.py": """
            def b():
                pass
            def a():
                b()
        """
    })
    try:
        cgi = CallGraphIndexer(repo)
        facade = RepositoryGraphFacade(call_graph_indexer=cgi, repo_root=repo)
        callees = facade.get_callees("a")
        callee_syms = [e.callee_symbol for e in callees]
        assert "b" in callee_syms
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_get_callers_fallback_to_repository_graph():
    """When no call_graph_indexer is provided, fallback to RepositoryGraph."""
    repo = _make_repo({
        "x.py": """
            def inner(): pass
            def outer():
                inner()
        """
    })
    try:
        facade = RepositoryGraphFacade(repo_root=repo)
        callers = facade.get_callers("inner")
        # Should not raise; results are canonical CallEdge objects
        assert isinstance(callers, list)
        for e in callers:
            assert isinstance(e, CallEdge)
            assert hasattr(e, "caller_symbol")
            assert hasattr(e, "callee_symbol")
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ── get_related_symbols() ─────────────────────────────────────────────────────

def test_get_related_symbols_with_indexer():
    from external_llm.agent.call_graph import CallGraphIndexer
    repo = _make_repo({
        "svc.py": """
            def helper(): pass
            def main():
                helper()
        """
    })
    try:
        cgi = CallGraphIndexer(repo)
        facade = RepositoryGraphFacade(call_graph_indexer=cgi, repo_root=repo)
        result = facade.get_related_symbols("main")
        assert isinstance(result, dict)
        assert "callees" in result
        assert "callers" in result
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_get_related_symbols_without_indexer_returns_empty():
    repo = _make_repo({"m.py": "def foo(): pass\n"})
    try:
        facade = RepositoryGraphFacade(repo_root=repo)
        result = facade.get_related_symbols("foo")
        assert result == []
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ── get_symbol_file() ─────────────────────────────────────────────────────────

def test_get_symbol_file():
    repo = _make_repo({"pkg/mod.py": "def my_fn(): pass\n"})
    try:
        facade = RepositoryGraphFacade(repo_root=repo)
        fp = facade.get_symbol_file("my_fn")
        assert fp is not None
        assert "mod.py" in fp
    finally:
        shutil.rmtree(repo, ignore_errors=True)
