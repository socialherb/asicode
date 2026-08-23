"""The `find_tests_for_symbol` tool and the finder behind it.

The tool answers "what covers this?", which the agent otherwise guesses at with
grep. Two properties are load-bearing and are pinned here:

* the ranking REASON survives to the model. A test that names the symbol is
  evidence of a different order than one that merely imports the module, and the
  old handler collapsed both to a bare path list.
* a miss says "no match was found", not "there are no tests". Those read the
  same to a model and lead to opposite actions.
"""

from __future__ import annotations

import pytest

from external_llm.testing.symbol_aware_test_finder import SymbolAwareTestFinder


@pytest.fixture
def repo(tmp_path):
    """A repo with one source module and three test files of different strength."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "widget.py").write_text(
        "class Widget:\n    def spin(self):\n        return 1\n", encoding="utf-8"
    )
    tests = tmp_path / "tests"
    tests.mkdir()
    # direct_symbol — names the symbol
    (tests / "test_widget.py").write_text(
        "from pkg.widget import Widget\n\ndef test_spin():\n    assert Widget().spin() == 1\n",
        encoding="utf-8",
    )
    # module_import — imports the module, never names the symbol
    (tests / "test_integration.py").write_text(
        "import pkg.widget\n\ndef test_smoke():\n    assert pkg.widget\n",
        encoding="utf-8",
    )
    # unrelated
    (tests / "test_other.py").write_text("def test_nothing():\n    assert True\n", encoding="utf-8")
    return tmp_path


def test_symbol_match_outranks_module_import(repo):
    finder = SymbolAwareTestFinder(str(repo))
    targets = finder.discover_test_targets(target_symbols=["Widget"], target_files=["pkg/widget.py"])
    paths = [t.test_path for t in targets]
    assert paths[0].endswith("test_widget.py"), paths
    assert targets[0].match_type == "direct_symbol"
    assert "Widget" in targets[0].matched_symbols
    by_path = {t.test_path: t for t in targets}
    integration = next(p for p in by_path if p.endswith("test_integration.py"))
    assert by_path[integration].match_type == "module_import"
    assert not any(p.endswith("test_other.py") for p in paths)


def test_symbol_beyond_the_old_read_cap_is_found(repo):
    """A large test file's second half used to be invisible.

    `_file_references_symbol` read only the first 50 000 characters, so a symbol
    exercised late in a big file reported "no related tests" — the answer that
    reads as "none exist". This repo's own test_shell_danger_policy.py is 93 KB.
    """
    padding = "# filler comment line to push the reference past the old cap\n" * 1200
    assert len(padding) > 50_000
    (repo / "tests" / "test_late.py").write_text(
        f"import pkg.widget\n{padding}\ndef test_late():\n    assert pkg.widget.Widget\n",
        encoding="utf-8",
    )
    finder = SymbolAwareTestFinder(str(repo))
    targets = finder.discover_test_targets(target_symbols=["Widget"])
    late = [t for t in targets if t.test_path.endswith("test_late.py")]
    assert late, f"symbol past the old 50 KB cap was missed: {[t.test_path for t in targets]}"
    assert late[0].match_type == "direct_symbol"


def test_file_contents_are_read_once_per_file(repo, monkeypatch):
    """Ten symbols must not mean ten reads of every test file."""
    finder = SymbolAwareTestFinder(str(repo))
    reads: list[str] = []
    _orig = finder._read_test_file.__func__

    def _counting(self, test_file):
        if test_file not in finder._content_cache:
            reads.append(test_file)
        return _orig(self, test_file)

    monkeypatch.setattr(SymbolAwareTestFinder, "_read_test_file", _counting, raising=True)
    finder.discover_test_targets(
        target_symbols=["Widget", "spin", "Nope1", "Nope2"],
        target_files=["pkg/widget.py"],
    )
    assert len(reads) == len(set(reads)), f"a test file was re-read: {reads}"


# ── the dispatchable tool ──────────────────────────────────────────────────


def _dispatch(tmp_path, args):
    from external_llm.agent.tool_registry import AgentConfig, ToolRegistry

    reg = ToolRegistry(repo_root=str(tmp_path), config=AgentConfig(run_lint=False))
    return reg.dispatch("find_tests_for_symbol", args)


def test_tool_is_dispatchable_and_reports_the_reason(repo):
    result = _dispatch(repo, {"symbol": "Widget"})
    assert result.ok, result.error
    assert "test_widget.py" in result.content
    assert "direct_symbol" in result.content, "the match reason must reach the model, not just the path"
    assert result.metadata.get("top_match_type") == "direct_symbol"


def test_tool_accepts_name_as_an_alias(repo):
    """asi.py's preview table and models both reach for `name`."""
    assert _dispatch(repo, {"name": "Widget"}).content == (_dispatch(repo, {"symbol": "Widget"}).content)


def test_a_miss_does_not_read_as_proof_of_no_coverage(repo):
    result = _dispatch(repo, {"symbol": "NoSuchSymbol"})
    assert result.ok
    assert result.metadata.get("match_count") == 0
    assert "NOT that the symbol is" in result.content, "a miss must distinguish 'not found' from 'not tested'"


def test_no_target_is_a_usable_error(repo):
    result = _dispatch(repo, {})
    assert not result.ok
    assert "symbol" in (result.error or "") and "file_path" in (result.error or "")
    assert result.retryable
