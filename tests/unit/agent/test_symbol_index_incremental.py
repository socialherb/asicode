"""Tests for symbol_index incremental rebuild + FIFO cache cap.

The incremental rebuild must produce results identical to a full re-parse for
added / changed / removed files, and the cache must be bounded by an entry cap.
"""
import os
import textwrap

import pytest

from external_llm.agent import symbol_index as si


def _write(path, body):
    with open(path, "w") as f:
        f.write(textwrap.dedent(body))


@pytest.fixture()
def repo(tmp_path):
    """Minimal repo with three Python files."""
    _write(str(tmp_path / "a.py"), """\
        class Aa:
            pass


        def aa():
            pass
    """)
    _write(str(tmp_path / "b.py"), """\
        class Bb:
            pass


        CONST_B = 1
    """)
    (tmp_path / "sub").mkdir()
    _write(str(tmp_path / "sub" / "c.py"), """\
        async def cc():
            pass


        X: int = 5
    """)
    return tmp_path


@pytest.fixture(autouse=True)
def _isolate_cache():
    """Each test gets a clean _INDEX_CACHE so they don't interfere."""
    saved = dict(si._INDEX_CACHE)
    si._INDEX_CACHE.clear()
    yield
    si._INDEX_CACHE.clear()
    si._INDEX_CACHE.update(saved)


def _force_ttl_expiry(root):
    """Rewrite the cache entry's timestamp to 0 so the next call re-walks."""
    file_idx, name_idx, mtimes, _ = si._INDEX_CACHE[str(root)]
    si._INDEX_CACHE[str(root)] = (file_idx, name_idx, mtimes, 0.0)


def _full_reparse_name_index(root):
    """Ground truth: full re-parse from scratch, derived into name index."""
    mtimes = si._collect_mtimes(str(root))
    full = si._rebuild_file_index(str(root), mtimes)
    return si._name_index_from_file_index(full)


def test_cold_cache_indexes_all_top_level_symbols(repo):
    idx = si.build_repo_symbol_index(str(repo))
    assert set(idx) == {"Aa", "aa", "Bb", "CONST_B", "cc", "X"}
    # Kinds are preserved.
    assert {loc.kind for loc in idx["Aa"]} == {"class"}
    assert {loc.kind for loc in idx["aa"]} == {"function"}
    assert {loc.kind for loc in idx["cc"]} == {"async_function"}
    assert {loc.kind for loc in idx["CONST_B"]} == {"constant"}


def test_no_change_reuses_index(repo):
    first = si.build_repo_symbol_index(str(repo))
    _force_ttl_expiry(repo)
    second = si.build_repo_symbol_index(str(repo))
    assert first == second


def test_changed_file_incremental_equals_full_reparse(repo):
    si.build_repo_symbol_index(str(repo))
    # Edit b.py: drop CONST_B, add bb_new.
    _write(str(repo / "b.py"), """\
        class Bb:
            pass


        def bb_new():
            pass
    """)
    _force_ttl_expiry(repo)
    idx = si.build_repo_symbol_index(str(repo))
    assert "CONST_B" not in idx
    assert "bb_new" in idx
    assert idx == _full_reparse_name_index(repo)


def test_added_file_incremental_equals_full_reparse(repo):
    si.build_repo_symbol_index(str(repo))
    _write(str(repo / "d.py"), """\
        class Dd:
            pass
    """)
    _force_ttl_expiry(repo)
    idx = si.build_repo_symbol_index(str(repo))
    assert "Dd" in idx
    assert idx == _full_reparse_name_index(repo)


def test_removed_file_incremental_equals_full_reparse(repo):
    si.build_repo_symbol_index(str(repo))
    os.remove(str(repo / "a.py"))
    _force_ttl_expiry(repo)
    idx = si.build_repo_symbol_index(str(repo))
    assert "Aa" not in idx
    assert "aa" not in idx
    assert idx == _full_reparse_name_index(repo)


def test_incremental_reuses_untouched_entries(repo):
    """Only the changed file should be re-scanned; untouched files keep their
    parsed SymbolLocation objects verbatim (identity preserved)."""
    si.build_repo_symbol_index(str(repo))
    file_idx, name_idx, mtimes, _ = si._INDEX_CACHE[str(repo)]
    a_locs_before = file_idx["a.py"]
    # Touch b.py only.
    _write(str(repo / "b.py"), """\
        class Bb:
            pass
    """)
    _force_ttl_expiry(repo)
    si.build_repo_symbol_index(str(repo))
    file_idx2, _, _, _ = si._INDEX_CACHE[str(repo)]
    # a.py was untouched -> its list object is reused as-is (same identity).
    assert file_idx2["a.py"] is a_locs_before


def test_determinism_locations_sorted_by_file_then_kind(repo):
    """Same-named symbol in multiple files must be sorted (file_path, kind)."""
    _write(str(repo / "z.py"), """\
        def Aa():
            pass
    """)
    si.build_repo_symbol_index(str(repo))  # populate
    _force_ttl_expiry(repo)
    idx = si.build_repo_symbol_index(str(repo))
    aa_locs = idx["Aa"]
    assert [(l.file_path, l.kind) for l in aa_locs] == sorted(
        (l.file_path, l.kind) for l in aa_locs
    )


def test_cache_is_bounded_by_entry_cap(repo, tmp_path_factory):
    """Inserting more repos than the cap evicts the oldest (FIFO)."""
    orig_cap = si._INDEX_CACHE_MAX_ENTRIES
    si._INDEX_CACHE_MAX_ENTRIES = 2
    try:
        si._INDEX_CACHE.clear()
        roots = []
        for i in range(4):
            r = tmp_path_factory.mktemp(f"cap_{i}_")
            _write(str(r / "f.py"), f"class F{i}:\n    pass\n")
            si.build_repo_symbol_index(str(r))
            roots.append(str(r))
        assert len(si._INDEX_CACHE) == 2
        # oldest two evicted, newest two kept
        assert roots[0] not in si._INDEX_CACHE
        assert roots[1] not in si._INDEX_CACHE
        assert roots[2] in si._INDEX_CACHE
        assert roots[3] in si._INDEX_CACHE
    finally:
        si._INDEX_CACHE_MAX_ENTRIES = orig_cap


def test_apply_incremental_does_not_poison_cached_file_index(repo):
    """_apply_incremental must mutate a copy, leaving the cached alias intact."""
    si.build_repo_symbol_index(str(repo))
    file_idx, _name_idx, mtimes, _ = si._INDEX_CACHE[str(repo)]
    # Simulate a removed file by dropping it from cur_mtimes.
    cur = dict(mtimes)
    cur.pop("a.py")
    new_file_idx, _ = si._apply_incremental(str(repo), file_idx, mtimes, cur)
    # The cached (old) file_index still has a.py -- not mutated in place.
    assert "a.py" in file_idx
    assert "a.py" not in new_file_idx
