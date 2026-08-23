"""CallGraphIndexer.build() parsed every walked file at any size.

It was the last unbounded parse loop in the agent: the shared walkers filter on
extension and file COUNT, nothing looked at how big one file is, and build()
reads + ``ast.parse``s each one. Reachable from the shipping analyze_impact /
trace_call_path tools.

Measured on a scratch repo holding one 3.7 MB generated module plus one
four-line module:

    before the gate:  build 10.12 s, 762 MB peak RSS, 120,002 symbols
    after the gate:   build  0.00 s,  26 MB peak RSS,      2 symbols

``ast.parse`` alone holds ~155x the source size in transient memory, so the cap
is what bounds the peak. Verified against this repo: 796 walked .py and 41
walked .ts/.js, zero of them over either cap — the gate costs nothing here and
only excludes the generated class it was sized for.
"""

import pathlib

import pytest

from external_llm.agent.call_graph import (
    _MAX_PY_BYTES,
    _MAX_TS_BYTES,
    CallGraphIndexer,
    _too_big_to_index,
)


def _generated_module(path: pathlib.Path, byte_target: int) -> pathlib.Path:
    """A syntactically valid .py of roughly *byte_target* bytes."""
    unit = "def generated_{}():\n    return {}\n"
    body = []
    size = 0
    i = 0
    while size < byte_target:
        chunk = unit.format(i, i)
        body.append(chunk)
        size += len(chunk)
        i += 1
    path.write_text("".join(body))
    return path


@pytest.fixture
def repo_with_one_huge_module(tmp_path):
    """One ordinary module beside one module past the Python cap."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "ok.py").write_text("def alpha():\n    beta()\ndef beta():\n    pass\n")
    _generated_module(src / "big_generated.py", _MAX_PY_BYTES + 200_000)
    return tmp_path


class TestOversizedFilesAreSkipped:
    def test_the_oversized_module_contributes_no_symbols(self, repo_with_one_huge_module):
        idx = CallGraphIndexer(str(repo_with_one_huge_module))
        idx.build()

        assert not any(n.startswith("generated_") for n in idx._nodes), (
            "the gate did not stop the oversized module from being parsed"
        )

    def test_its_neighbour_is_still_indexed(self, repo_with_one_huge_module):
        """Degradation must be per-file, not a build that gives up."""
        idx = CallGraphIndexer(str(repo_with_one_huge_module))
        idx.build()

        assert "alpha" in idx._nodes
        assert idx.get_callees("alpha"), "the ordinary file lost its edges too"

    def test_the_file_is_never_READ(self, repo_with_one_huge_module, monkeypatch):
        """The gate has to precede the read, not just the parse.

        ast.parse is the expensive half but the read materialises the source
        first; gating between them would still pay the whole file in memory.
        Since P1 (2026-08-11) ``_index_file`` reads via ``parse_cache.parse_ast``
        (the shared parse layer) rather than ``Path.read_text``, the spy wraps
        ``parse_ast``: the oversized file must never be requested, the ordinary
        file must. The gate still precedes the read.
        """
        from external_llm.analysis import parse_cache as pc

        seen: list[str] = []
        real_parse_ast = pc.parse_ast

        def _spy(abs_path, *a, **k):
            seen.append(pathlib.Path(abs_path).name)
            return real_parse_ast(abs_path, *a, **k)

        monkeypatch.setattr(pc, "parse_ast", _spy)
        CallGraphIndexer(str(repo_with_one_huge_module)).build()

        assert "big_generated.py" not in seen, "oversized source was read into memory"
        assert "ok.py" in seen, "the spy never saw the file that SHOULD be read"


class TestTheGateIsSizeShaped:
    def test_a_file_just_under_the_cap_is_indexed(self, tmp_path):
        """A cap that trimmed real modules would be worse than the bug.

        The largest first-party file in this repo is 358 KB, so everything real
        sits far below the boundary — but the boundary itself has to hold.
        """
        src = tmp_path / "src"
        src.mkdir()
        _generated_module(src / "just_under.py", _MAX_PY_BYTES - 100_000)

        idx = CallGraphIndexer(str(tmp_path))
        idx.build()

        assert any(n.startswith("generated_") for n in idx._nodes)

    def test_the_predicate_answers_on_size(self, tmp_path):
        small = tmp_path / "small.py"
        small.write_text("x = 1\n")
        assert not _too_big_to_index(small, _MAX_PY_BYTES)
        assert _too_big_to_index(small, 2)

    def test_a_missing_file_is_not_reported_as_oversized(self, tmp_path):
        """A stat failure belongs to build()'s per-file except, not here.

        Answering True would silently convert every transient stat error into a
        permanent hole in the graph.
        """
        assert not _too_big_to_index(tmp_path / "gone.py", _MAX_PY_BYTES)

    def test_the_ts_cap_matches_the_tree_sitter_trade(self):
        """8 MiB is symbol_search's _NONPY_INPROC_MAX_BYTES, not a new number."""
        from external_llm.agent.symbol_search import _NONPY_INPROC_MAX_BYTES

        assert _MAX_TS_BYTES == _NONPY_INPROC_MAX_BYTES

    def test_the_python_cap_is_tighter_than_the_tree_sitter_one(self):
        """Different parsers, different amplification — deliberately not shared.

        ast.parse is ~155x the source in transient memory, so the 8 MiB that
        tree-sitter tolerates would let a single module cost over a gigabyte.
        """
        assert _MAX_PY_BYTES < _MAX_TS_BYTES


def test_the_ts_path_gates_before_its_read(tmp_path, monkeypatch):
    """_index_ts_file is the tree-sitter twin and had the identical hole."""
    big = tmp_path / "vendor.bundle.js"
    big.write_text("var x=1;\n" * ((_MAX_TS_BYTES // 9) + 5_000))
    assert big.stat().st_size > _MAX_TS_BYTES

    def _explode(self, *a, **kw):
        raise AssertionError(f"oversized {self.name} was read")

    monkeypatch.setattr(pathlib.Path, "read_text", _explode)
    # Returns without reading; no tracer import, no parse.
    CallGraphIndexer(str(tmp_path))._index_ts_file(big)
