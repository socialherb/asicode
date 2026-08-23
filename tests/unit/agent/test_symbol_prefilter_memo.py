"""The find_symbol Python prefilter memo must save spawns without going stale.

``_rg_list_py_files`` spawns ripgrep to narrow the file set before any
tree-sitter parsing. The narrowing query depends only on (root, matcher args),
and the agent issues the SAME query twice in its most common sequence —
``find_symbol(X)`` then ``read_symbol(X)`` — which the tool-result cache cannot
dedupe because the two arrive under different tool names.

Memoizing that is only safe because both post-write invalidation paths clear it.
Without the clear it reintroduces "find_symbol answers 'No definitions found'
for a function that is on disk" (commit 77008787) on the Python side, so the
staleness case is asserted here as explicitly as the hit case.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from external_llm.agent._shared_utils import _PY_WALK_CACHE
from external_llm.agent.symbol_search import (
    _RG_PY_FILTER_CACHE,
    _RG_PY_FILTER_MAX_ENTRIES,
    SymbolSearcher,
    _capped_put,
    _rg_py_files_defining,
    invalidate_py_prefilter_cache,
)

pytestmark = pytest.mark.skipif(
    __import__("shutil").which("rg") is None,
    reason="prefilter only runs when ripgrep is installed",
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repo with >1 .py file — the prefilter only engages when it can narrow."""
    for i in range(6):
        (tmp_path / f"mod_{i}.py").write_text(f"import os\nX_{i} = {i}\n", encoding="utf-8")
    (tmp_path / "mod_a.py").write_text("class AlphaWidget:\n    pass\n", encoding="utf-8")
    return tmp_path


@pytest.fixture(autouse=True)
def _clean_caches():
    """Both the memo and the walk cache are module-global and TTL-based."""
    _RG_PY_FILTER_CACHE.clear()
    _PY_WALK_CACHE.clear()
    yield
    _RG_PY_FILTER_CACHE.clear()
    _PY_WALK_CACHE.clear()


@pytest.fixture
def count_prefilter_spawns(monkeypatch):
    """Count only the prefilter's own rg invocations, not every subprocess."""
    spawns: list = []
    _orig = subprocess.run

    def _traced(cmd, *a, **kw):
        if isinstance(cmd, (list, tuple)) and "--files-with-matches" in [str(c) for c in cmd]:
            spawns.append(cmd)
        return _orig(cmd, *a, **kw)

    monkeypatch.setattr(subprocess, "run", _traced)
    return spawns


def test_repeat_lookup_is_served_from_the_memo(repo, count_prefilter_spawns):
    """The find_symbol -> read_symbol repeat must not respawn ripgrep."""
    s = SymbolSearcher(str(repo))

    assert s.find_symbol("AlphaWidget"), "first lookup must find the symbol"
    after_first = len(count_prefilter_spawns)
    assert after_first > 0, "prefilter must actually run for this fixture"

    assert s.find_symbol("AlphaWidget"), "second lookup must still find the symbol"
    assert len(count_prefilter_spawns) == after_first, "repeat lookup respawned rg"


def test_a_distinct_query_is_not_served_by_another_entry(repo, count_prefilter_spawns):
    """The key is the QUERY, so a different symbol must miss."""
    s = SymbolSearcher(str(repo))
    s.find_symbol("AlphaWidget")
    before = len(count_prefilter_spawns)
    s.find_symbol("SomethingElse")
    assert len(count_prefilter_spawns) > before


def test_handed_out_set_is_a_copy(repo):
    """find_symbol intersects the result in place; that must not poison the memo."""
    first = _rg_py_files_defining(repo, "AlphaWidget", "any")
    assert first is not None
    key = next(iter(_RG_PY_FILTER_CACHE))
    snapshot = set(_RG_PY_FILTER_CACHE[key][1])

    first.add("/tmp/not-a-real-file.py")
    assert _RG_PY_FILTER_CACHE[key][1] == snapshot, "caller mutation leaked into the memo"
    assert _rg_py_files_defining(repo, "AlphaWidget", "any") == snapshot


def test_warm_memo_hides_a_file_written_afterwards(repo):
    """The staleness the invalidation hook exists to prevent — asserted, not assumed."""
    s = SymbolSearcher(str(repo))
    assert s.find_symbol("AlphaWidget")  # warms the memo for this query

    (repo / "mod_new.py").write_text("class AlphaWidget:\n    pass\n", encoding="utf-8")
    _PY_WALK_CACHE.clear()  # isolate the memo: the walk must already see the new file

    stale = {d.file for d in s.find_symbol("AlphaWidget")}
    assert not any("mod_new" in f for f in stale), (
        "fixture is not exercising the memo — the new file was visible without invalidation"
    )

    invalidate_py_prefilter_cache()
    fresh = {d.file for d in s.find_symbol("AlphaWidget")}
    assert any("mod_new" in f for f in fresh), "invalidation did not restore visibility"


def test_a_warm_memo_never_masks_a_missing_rg(repo, monkeypatch):
    """ "rg is gone" must still answer None, not a stale set.

    The memo lookup has to sit AFTER the ``shutil.which`` check. Put before it,
    a warm entry answers on a machine where ripgrep has since disappeared, and
    the caller's contract — None means "prefilter untrustworthy, scan every
    file" — silently becomes "here is what was true 30 seconds ago".
    """
    import external_llm.agent.symbol_search as ss

    warm = _rg_py_files_defining(repo, "AlphaWidget", "any")
    assert warm, "memo must be warm for this test to mean anything"

    monkeypatch.setattr(ss.shutil, "which", lambda _name: None)
    assert _rg_py_files_defining(repo, "AlphaWidget", "any") is None


def test_invalidate_clears_every_entry(repo):
    s = SymbolSearcher(str(repo))
    s.find_symbol("AlphaWidget")
    s.find_symbol("X_1")
    assert len(_RG_PY_FILTER_CACHE) > 1
    invalidate_py_prefilter_cache()
    assert _RG_PY_FILTER_CACHE == {}


def test_memo_is_bounded():
    """A long session looks up many distinct symbols; the memo must not grow forever."""
    _RG_PY_FILTER_CACHE.clear()
    for i in range(_RG_PY_FILTER_MAX_ENTRIES * 2):
        _capped_put(
            _RG_PY_FILTER_CACHE,
            ("/repo", (f"tok{i}",)),
            (0.0, set()),
            _RG_PY_FILTER_MAX_ENTRIES,
        )
    assert len(_RG_PY_FILTER_CACHE) <= _RG_PY_FILTER_MAX_ENTRIES
