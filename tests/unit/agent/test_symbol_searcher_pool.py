"""Process-shared SymbolSearcher pool (get_symbol_searcher) tests.

Covers the pool contract: same-root identity, resolved-key normalization,
LRU cap/eviction, invalidation reaching the shared instance, and the
patch_engine salvage path that used to call a non-existent ``search`` API
(which raised AttributeError every time, aborting symbol-aware anchoring).
"""

import pytest

from external_llm.agent import symbol_search as ss


@pytest.fixture(autouse=True)
def _isolated_pool():
    """Each test sees a pristine pool; restore the previous contents after."""
    with ss._searcher_pool_lock:
        saved = dict(ss._searcher_pool)
        ss._searcher_pool.clear()
    yield
    with ss._searcher_pool_lock:
        ss._searcher_pool.clear()
        ss._searcher_pool.update(saved)


def test_same_root_returns_same_instance(tmp_path):
    a = ss.get_symbol_searcher(str(tmp_path))
    b = ss.get_symbol_searcher(str(tmp_path))
    assert a is b


def test_resolved_root_key_normalization(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    direct = ss.get_symbol_searcher(str(proj))
    dotted = ss.get_symbol_searcher(str(proj / "."))
    assert direct is dotted


def test_distinct_roots_get_distinct_instances(tmp_path):
    r1 = tmp_path / "one"
    r2 = tmp_path / "two"
    r1.mkdir()
    r2.mkdir()
    assert ss.get_symbol_searcher(str(r1)) is not ss.get_symbol_searcher(str(r2))


def test_pool_caps_at_max_entries_with_lru_eviction(tmp_path):
    roots = [tmp_path / f"r{i}" for i in range(5)]
    for r in roots:
        r.mkdir()
    # Fill to cap (4); keep the first instance to detect eviction.
    first = ss.get_symbol_searcher(str(roots[0]))
    for r in roots[1:4]:
        ss.get_symbol_searcher(str(r))
    # Refresh roots[1] so it becomes most-recent, then insert the 5th root.
    refreshed = ss.get_symbol_searcher(str(roots[1]))
    assert refreshed is ss.get_symbol_searcher(str(roots[1]))
    ss.get_symbol_searcher(str(roots[4]))
    # roots[0] (oldest, not refreshed) was evicted -> fresh instance now.
    assert ss.get_symbol_searcher(str(roots[0])) is not first
    # roots[1] (refreshed) survived the eviction.
    assert ss.get_symbol_searcher(str(roots[1])) is refreshed


def test_invalidate_reaches_pooled_instance(tmp_path):
    searcher = ss.get_symbol_searcher(str(tmp_path))
    searcher._nonpy_index_cache["root"] = ("x", {"name": []})
    ss._NONPY_FILES_CACHE["probe"] = ("x", [], 0)
    ss._NONPY_BLOB_CACHE["blob"] = ("x", {}, "")
    searcher.invalidate_nonpy_caches()
    assert searcher._nonpy_index_cache == {}
    assert ss._NONPY_FILES_CACHE == {}
    assert ss._NONPY_BLOB_CACHE == {}


def test_salvage_anchor_uses_symbol_search(tmp_path):
    """Added-lines patch anchors BEFORE the existing def via find_symbol.

    Regression for the dead ``search()`` call: it raised AttributeError,
    ``except Exception`` swallowed it, and the salvage returned None.
    """
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "mod.py").write_text("import os\n\n\ndef foo(x):\n    return x\n")
    from external_llm.patch_engine import PatchEngine

    engine = PatchEngine(str(proj))
    out = engine._salvage_small_model_output(
        "+def foo():\n+    return 3\n", "mod.py"
    )
    assert out is not None
    assert "+def foo():" in out
    # Symbol anchor at line 4: the "import os" context line precedes the
    # insertion (a fallback anchor 0 would put the insertion first).
    assert " import os" in out
    assert out.index(" import os") < out.index("+def foo():")


def test_salvage_anchor_falls_back_to_top_when_symbol_missing(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "mod.py").write_text("import os\n\n\ndef foo():\n    return 1\n")
    from external_llm.patch_engine import PatchEngine

    engine = PatchEngine(str(proj))
    out = engine._salvage_small_model_output(
        "+def nosuch():\n+    return 3\n", "mod.py"
    )
    assert out is not None
    assert "+def nosuch():" in out
    # Unknown symbol -> fallback anchor 0: the insertion precedes everything.
    assert out.index("+def nosuch():") < out.index(" import os")
