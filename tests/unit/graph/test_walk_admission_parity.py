"""All repo walkers must admit the SAME file set (B2' parity contract).

CGI / symbol_search / vulture / RAG share external_llm.agent._shared_utils
walkers (_walk_py_files / _walk_ts_js_files / rag_searcher._walk_files);
RepositoryGraph walks independently but
admits via the SAME predicates (repository_graph._file_is_walk_admissible →
external_llm.common.walk_policy._path_is_walk_admissible — the walk-policy
SSOT (F5)).  Any divergence makes get_callers(CGI)
and get_symbol(RG) answer different universes.

This test locks the two walker families equal — it FAILS on the pre-B2' code
at all three known divergence points (vendor/, .egg-info/, .min.js) and must
be committed together with the B2' fix (test-first).  The ts/js comparison
filters RepositoryGraph's non-py stamps by _TS_JS_EXTENSIONS: RG stamps every
non-Python language (go/java/...), the shared ts/js walker only ts/js — the
filter (and the .go/.java fixtures below) is what makes the two sets
comparable instead of relying on fixture luck.
"""

from pathlib import Path

from external_llm.agent._shared_utils import (
    _TS_JS_EXTENSIONS,
    _walk_py_files,
    _walk_ts_js_files,
)
from external_llm.agent.rag_searcher import RAGSearcher
from external_llm.common.walk_policy import (
    _WALK_SKIP_FILE_SUFFIXES,
    _walk_should_skip_dir,
)
from external_llm.graph.repository_graph import RepositoryGraph

FIXTURE = {
    "src/main.py": "def main():\n    pass\n",  # include (py)
    "app.js": "export function app() {}\n",  # include (ts/js)
    "vendor/dep.py": "def dep():\n    pass\n",  # divergence 1: shared indexes, RG skips
    "vendor/dep.js": "export function dep() {}\n",  # divergence 1
    "pkg.egg-info/x.py": "def egg():\n    pass\n",  # divergence 2: RG indexes, shared skips
    "lib.min.js": "function f() {}\n",  # divergence 3: shared indexes, RG skips
    "node_modules/lib/m.py": "def m():\n    pass\n",  # control: both skip
    ".venv/x.py": "def x():\n    pass\n",  # control: both skip
    "Main.java": "class Main {}\n",  # non-py non-ts/js: RG stamps, tsjs walker must not
    "svc/handler.go": "package svc\n",  # — locks the _TS_JS_EXTENSIONS filter in _rg_sets
}


def _make_fixture(tmp_path: Path) -> None:
    for rel, text in FIXTURE.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


def _shared_sets(root: Path) -> tuple[set[str], set[str]]:
    return (
        {str(p.relative_to(root)) for p in _walk_py_files(root, 100_000)},
        {str(p.relative_to(root)) for p in _walk_ts_js_files(root, 100_000)},
    )


def _rg_sets(g: RepositoryGraph) -> tuple[set[str], set[str]]:
    return (
        set(g.py_files),
        {rel for rel, _p, _st in g._nonpy_stamps if rel.endswith(_TS_JS_EXTENSIONS)},
    )


def test_py_walk_admission_parity(tmp_path):
    _make_fixture(tmp_path)
    g = RepositoryGraph(str(tmp_path))
    g.build()
    assert _shared_sets(tmp_path)[0] == _rg_sets(g)[0]
    # non-vacuous: shared py walk must actually contain the include file
    assert "src/main.py" in _shared_sets(tmp_path)[0]
    # and must NOT contain either divergence point
    assert not _shared_sets(tmp_path)[0] & {"vendor/dep.py", "pkg.egg-info/x.py"}


def test_tsjs_walk_admission_parity(tmp_path):
    _make_fixture(tmp_path)
    g = RepositoryGraph(str(tmp_path))
    g.build()
    assert _shared_sets(tmp_path)[1] == _rg_sets(g)[1]
    assert "app.js" in _shared_sets(tmp_path)[1]
    assert not _shared_sets(tmp_path)[1] & {"vendor/dep.js", "lib.min.js"}


def test_rag_walk_admission_parity(tmp_path):
    """4th walker: RAG must never admit a path the shared policy rejects.
    RAG's extension set is legitimately wider (md/toml/yaml/yml + every
    language), so the assertion is *zero shared-policy violations*, not set
    equality — the opposite direction (RAG skipping what others admit, e.g.
    migrations/) is RAG's own policy and allowed."""
    _make_fixture(tmp_path)
    searcher = RAGSearcher(str(tmp_path), vector_cache_enabled=False)
    rels = [str(p.relative_to(tmp_path)) for p in searcher._walk_files()]
    violations = [
        rel
        for rel in rels
        if any(_walk_should_skip_dir(d) for d in Path(rel).parts[:-1])
        or Path(rel).name.endswith(_WALK_SKIP_FILE_SUFFIXES)
    ]
    assert not violations, f"RAG admitted paths the shared policy skips: {violations}"
    # non-vacuous: real source is admitted, divergence fixtures are not
    assert "src/main.py" in rels and "app.js" in rels
    assert not any(rel.startswith(("vendor/", "node_modules/", ".venv/", "pkg.egg-info/")) for rel in rels)
