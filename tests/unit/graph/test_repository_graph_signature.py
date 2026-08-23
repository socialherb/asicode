"""Signature extraction/hash in RepositoryGraph — positive shapes + fail-fast.

B1/B2: ``_extract_signature`` (8 wrappers) and ``_compute_signature_hash``
previously ran inside ``suppress(Exception)`` — a bug silently indexed the
symbol WITHOUT a signature (search/read output degraded) or degraded the hash
(missing change detection).  They now fail fast: the error propagates to
``RepositoryGraph.build()``'s per-file guard, which skips the file with a
loud debug log instead of partially indexing it.  These tests pin (1) the
signature text for the exotic-args shapes that used to carry per-call
suppress wrappers, (2) hash determinism, (3) the fail-fast contract.
"""

import ast
import hashlib
import shutil
import tempfile
from pathlib import Path

import pytest

from external_llm.graph.repository_graph import GraphVisitor, RepositoryGraph


def _build_graph(source: str) -> RepositoryGraph:
    d = tempfile.mkdtemp(prefix="test_rsig_")
    fp = Path(d) / "mod.py"
    fp.write_text(source)
    try:
        g = RepositoryGraph(str(d))
        g.build()
        return g
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _only_symbol(g: RepositoryGraph):
    syms = list(g.symbols.values())
    assert len(syms) == 1, syms
    return syms[0]


# --- positive: the exotic-args shapes that used to carry per-call suppress ----
def test_signature_extraction_full_shapes():
    src = "async def f(a: int = 1, *args: str, b: float = 2.0, **kw: bool) -> None:\n    return\n"
    sym = _only_symbol(_build_graph(src))
    assert sym.signature == ("async def f(a: int = 1, *args: str, b: float = 2.0, **kw: bool) -> None"), sym.signature
    # Hash = sha1 of name + arg names (vararg/kwarg prefixed), first 8 hex chars.
    expected = hashlib.sha1(b"f,a,*args,b,**kw", usedforsecurity=False).hexdigest()[:8]
    assert sym.signature_hash == expected


def test_signature_extraction_plain_def():
    sym = _only_symbol(_build_graph("def g(x, *args, **kw):\n    return\n"))
    assert sym.signature == "def g(x, *args, **kw)"


def test_signature_extraction_kwonly_only():
    sym = _only_symbol(_build_graph("def h(*, y):\n    return\n"))
    assert sym.signature == "def h(*, y)"


def test_signature_extraction_defaults_offset():
    # defaults_offset: only the trailing args get " = <default>" (unparse
    # spacing: "b = 2", not "b=2").
    sym = _only_symbol(_build_graph("def k(a, b=2, c='x'):\n    return\n"))
    assert sym.signature == "def k(a, b = 2, c = 'x')"


# --- fail-fast contract -------------------------------------------------------
def test_extract_signature_fails_fast(monkeypatch):
    """ast.unparse cannot fail on a parser-produced AST; if it ever does, the
    error must propagate out of ``_extract_signature`` (was: silently swallowed
    → symbol indexed without a signature)."""
    visitor = GraphVisitor("mod.py", "/repo")
    node = ast.parse("def f(x: int = 1) -> None:\n    pass\n").body[0]

    def _boom(*a, **k):
        raise RuntimeError("unparse bug")

    monkeypatch.setattr(ast, "unparse", _boom)
    with pytest.raises(RuntimeError, match="unparse bug"):
        visitor._extract_signature(node)


def test_build_skips_broken_file_without_corrupting_others(tmp_path, monkeypatch):
    """End-to-end: a signature-extraction failure now skips the WHOLE file via
    build()'s per-file guard (debug log + build_exception_types) instead of
    partially indexing a symbol without its signature."""
    (tmp_path / "good.py").write_text("def ok():\n    pass\n")
    (tmp_path / "bad.py").write_text("def f(x: int) -> int:\n    return x\n")

    def _boom(*a, **k):
        raise RuntimeError("unparse bug")

    monkeypatch.setattr(ast, "unparse", _boom)
    g = RepositoryGraph(str(tmp_path))
    g.build()
    assert "good.py" in g.file_symbols
    assert "bad.py" not in g.file_symbols
    assert any("RuntimeError" in tag for tag in g.build_exception_types)


# --- RG-B1: positional-only args (def f(a, /, b)) must survive ----------------
def test_signature_extraction_posonly_preserved():
    """RG-B1: ``def f(a, /, b)`` previously dropped ``a`` and the ``/`` separator
    → rendered ``def f(b)`` (LLM-facing signature corrupted, a parameter lost)."""
    sym = _only_symbol(_build_graph("def f(a, /, b):\n    return\n"))
    assert sym.signature == "def f(a, /, b)", sym.signature


def test_signature_extraction_posonly_with_kwonly():
    sym = _only_symbol(_build_graph("def f(a, /, b, *, c):\n    return\n"))
    assert sym.signature == "def f(a, /, b, *, c)", sym.signature


def test_signature_extraction_posonly_with_vararg():
    # both '/' and *args coexist (def f(a, /, *args, b)) — the '/' is always
    # rendered when posonlyargs exist, independent of vararg.
    sym = _only_symbol(_build_graph("def f(a, /, *args, b):\n    return\n"))
    assert sym.signature == "def f(a, /, *args, b)", sym.signature


def test_signature_extraction_posonly_defaults_merged_offset():
    """RG-B1: ``defaults`` spans BOTH posonlyargs and regular args. The offset
    must be computed over the combined list, else a posonlyarg default
    misaligns with the wrong regular arg."""
    sym = _only_symbol(_build_graph("def f(a=1, /, b=2):\n    return\n"))
    assert sym.signature == "def f(a = 1, /, b = 2)", sym.signature


def test_signature_hash_posonly_no_collision():
    """RG-B1: ``def f(b)`` and ``def f(a, /, b)`` previously collided to the
    same hash — a posonly arg added/removed was invisible to change detection.
    A posonly↔regular conversion (API-breaking) must also change the hash."""
    v = GraphVisitor("mod.py", "/repo")
    h_plain = v._compute_signature_hash(ast.parse("def f(a, b):\n    pass\n").body[0])
    h_posonly = v._compute_signature_hash(ast.parse("def f(a, /, b):\n    pass\n").body[0])
    h_drop = v._compute_signature_hash(ast.parse("def f(b):\n    pass\n").body[0])
    assert h_plain != h_posonly, "posonly<->regular conversion not detected"
    assert h_drop != h_posonly, "posonly arg removal not detected (collision with f(b))"


def test_signature_hash_encodes_posonly_boundary_marker():
    """The '/' marker is encoded so the hash reflects posonly boundaries."""
    src = "def f(a, /, b):\n    return\n"
    sym = _only_symbol(_build_graph(src))
    expected = hashlib.sha1(b"f,a,/,b", usedforsecurity=False).hexdigest()[:8]
    assert sym.signature_hash == expected
