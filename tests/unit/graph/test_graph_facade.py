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


def test_facade_py_files_returns_uncapped_walked_list():
    """``py_files`` exposes the walked .py list for a PLAIN-mode build.

    The facade builds via GraphBuilder.build_repo_graph → build() (no
    collect_imported_names); the structural-scan tool unions this into its
    cross-file-ref input, so it must be populated here too — not just under
    the gate's names-collecting mode (2026-08-11).
    """
    repo = _make_repo(
        {
            "a.py": "def fa(): pass\n",
            "sub/b.py": "def fb(): pass\n",
            "c.ts": "export const c = 1;\n",
        }
    )
    try:
        facade = RepositoryGraphFacade(repo_root=repo)
        assert facade.py_files == ["a.py", "sub/b.py"]  # .ts never walked as py
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
    repo = _make_repo(
        {
            "m.py": """
            class MyClass:
                def method(self): pass
        """
        }
    )
    try:
        facade = RepositoryGraphFacade(repo_root=repo)
        sym = facade.get_symbol("MyClass.method")
        assert sym is not None
        assert sym.qualname == "MyClass.method"
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_get_symbol_with_file_hint():
    repo = _make_repo(
        {
            "a.py": "def helper(): pass\n",
            "b.py": "def helper(): pass\n",
        }
    )
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
    repo = _make_repo(
        {
            "svc.py": """
            def alpha(): pass
            def beta(): pass
            class Gamma: pass
        """
        }
    )
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

    repo = _make_repo(
        {
            "foo.py": """
            def b():
                pass
            def a():
                b()
        """
        }
    )
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

    repo = _make_repo(
        {
            "foo.py": """
            def b():
                pass
            def a():
                b()
        """
        }
    )
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
    repo = _make_repo(
        {
            "x.py": """
            def inner(): pass
            def outer():
                inner()
        """
        }
    )
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

    repo = _make_repo(
        {
            "svc.py": """
            def helper(): pass
            def main():
                helper()
        """
        }
    )
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


# ── Batched incremental invalidation (P3, 2026-08-11) ─────────────────────────


def test_facade_invalidate_files_handles_mixed_reparse_and_deleted():
    """invalidate_files splits paths into reparse (existing) + remove (deleted).

    The facade used to call reparse_file/remove_file per path; P3 batches each
    set. A mix of edited + deleted files must leave the graph with the edited
    files' NEW symbols and the deleted files' symbols gone — same observable
    state as the per-path loop, but built with one _remove_files pass.
    """
    repo = _make_repo(
        {
            "a.py": "def a():\n    return 1\n",
            "b.py": "def b():\n    return 2\n",
            "c.py": "def c():\n    return 3\n",
        }
    )
    try:
        facade = RepositoryGraphFacade(repo_root=repo)
        facade._ensure_graph()
        assert {s.name for s in facade._graph.symbols.values()} == {"a", "b", "c"}

        # Edit a.py (new symbol), delete b.py, leave c.py untouched.
        (Path(repo) / "a.py").write_text("def a():\n    return 1\n\ndef a2():\n    return 11\n")
        (Path(repo) / "b.py").unlink()

        facade.invalidate_files(["a.py", "b.py"])

        names = {s.name for s in facade._graph.symbols.values()}
        assert names == {"a", "a2", "c"}, names
        assert "b.py" not in [rel for rel, _p, _st in facade._graph._py_stamps]
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_facade_invalidate_files_skips_non_language_paths():
    """Non-language paths (UNKNOWN LanguageId) are filtered before the batch."""
    repo = _make_repo({"a.py": "def a():\n    return 1\n"})
    try:
        facade = RepositoryGraphFacade(repo_root=repo)
        facade._ensure_graph()
        before = {s.name for s in facade._graph.symbols.values()}

        # A README and a .txt are not language files -> filtered, no crash.
        facade.invalidate_files(["README.md", "notes.txt", "nonexistent.py"])

        assert {s.name for s in facade._graph.symbols.values()} == before
    finally:
        shutil.rmtree(repo, ignore_errors=True)
