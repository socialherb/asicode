"""Tests for external_llm/analysis/parse_cache.py."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from external_llm.analysis import parse_cache


@pytest.fixture(autouse=True)
def _reset_cache():
    """Each test starts from the default cache state."""
    parse_cache.clear()
    parse_cache._max_entries = parse_cache._DEFAULT_CACHE_SIZE
    yield
    parse_cache.clear()


def _make_py(tmpdir: str, name: str, src: str) -> str:
    p = Path(tmpdir) / name
    p.write_text(src, encoding="utf-8")
    return str(p)


def test_parse_ast_caches_module():
    with tempfile.TemporaryDirectory() as d:
        path = _make_py(d, "a.py", "x = 1\n")
        first = parse_cache.parse_ast(path)
        second = parse_cache.parse_ast(path)
        assert first is not None
        # Same object returned from cache (not re-parsed).
        assert first is second


def test_parse_ast_invalidates_on_edit():
    with tempfile.TemporaryDirectory() as d:
        path = _make_py(d, "a.py", "x = 1\n")
        first = parse_cache.parse_ast(path)
        # Rewrite with different size -> stat key changes -> re-parse.
        Path(path).write_text("x = 1\ny = 2\n", encoding="utf-8")
        second = parse_cache.parse_ast(path)
        assert first is not second


def test_parse_ast_returns_none_on_syntax_error():
    with tempfile.TemporaryDirectory() as d:
        path = _make_py(d, "bad.py", "def (:\n")
        assert parse_cache.parse_ast(path) is None


def test_parse_ast_returns_none_for_missing_file():
    assert parse_cache.parse_ast("/no/such/file/at/all.py") is None


def test_ensure_capacity_grows_cache():
    parse_cache.ensure_capacity(1000)
    info = parse_cache.cache_info()
    # 1000 + headroom, capped at _MAX_CACHE_SIZE.
    assert info.maxsize == min(1000 + parse_cache._CAPACITY_HEADROOM,
                               parse_cache._MAX_CACHE_SIZE)


def test_ensure_capacity_is_capped():
    parse_cache.ensure_capacity(10_000_000)
    assert parse_cache.cache_info().maxsize == parse_cache._MAX_CACHE_SIZE


def test_max_cache_size_covers_capped_scan_list():
    """A capped scanner list must always fit: SCAN_FILE_CAP + headroom <= ceiling.

    parse_cache stays stdlib-only (it must never import scan_walk), so the
    invariant is pinned HERE at the test layer: raising SCAN_FILE_CAP without
    raising the ceiling fails loudly.  The UNCAPPED graph/cross-ref sets are
    best-effort above the ceiling (P2 policy, 2026-08-11 — see the module
    docstring's Cache sizing section).
    """
    from external_llm.analysis.scan_walk import SCAN_FILE_CAP

    assert parse_cache._MAX_CACHE_SIZE >= SCAN_FILE_CAP + parse_cache._CAPACITY_HEADROOM


def test_ensure_capacity_never_shrinks():
    parse_cache.ensure_capacity(500)
    big = parse_cache.cache_info().maxsize
    parse_cache.ensure_capacity(10)  # smaller request
    assert parse_cache.cache_info().maxsize == big


def test_grown_cache_holds_more_than_default_file_set():
    """The bug: a working set larger than the default size must survive a full
    pass so a second scanner over the same set hits the cache."""
    n = parse_cache._DEFAULT_CACHE_SIZE + 50
    with tempfile.TemporaryDirectory() as d:
        paths = [_make_py(d, f"m{i}.py", f"v{i} = {i}\n") for i in range(n)]
        parse_cache.ensure_capacity(n)
        trees = {p: parse_cache.parse_ast(p) for p in paths}
        # Second pass (simulating the next scanner) must return the SAME objects
        # for every file — i.e. nothing was evicted mid-pass.
        for p in paths:
            assert parse_cache.parse_ast(p) is trees[p]


def test_grown_capacity_preserves_populated_entries():
    """F2 (2026-08-11): regrowing the cache must NOT drop warm entries.

    The old lru_cache-wrapper rebuild discarded every populated entry on
    regrow, so a consumer sizing the cache AFTER another consumer warmed it
    forced a full re-parse.  The OrderedDict LRU raises the ceiling in place.
    """
    with tempfile.TemporaryDirectory() as d:
        paths = [_make_py(d, f"m{i}.py", f"v{i} = {i}\n") for i in range(3)]
        trees = {p: parse_cache.parse_ast(p) for p in paths}
        parse_cache.ensure_capacity(1000)  # grow AFTER warming
        for p in paths:
            assert parse_cache.parse_ast(p) is trees[p]


def test_byte_budget_evicts_lru_first(monkeypatch):
    """F1 (2026-08-11): the byte budget bounds resident AST memory.

    Entry cost is len(src) * (1 + _AST_BYTES_PER_SOURCE_BYTE); when the total
    exceeds the budget the least-recently-used entries are evicted.
    """
    monkeypatch.setattr(parse_cache, "_MAX_CACHE_BYTES", 2000)
    src = "x = 1\ny = 2\n"  # 12 bytes -> cost 12 * 16 = 192
    n = 12  # 12 * 192 = 2304 > 2000: inserts 11 and 12 each evict one LRU entry
    with tempfile.TemporaryDirectory() as d:
        paths = [_make_py(d, f"m{i}.py", src) for i in range(n)]
        for p in paths:
            parse_cache.parse_ast(p)
        assert parse_cache.cache_info().currsize == n - 2
        # The LRU entries (first-inserted) were evicted; the rest still hit.
        assert parse_cache.parse_ast(paths[0]) is not None  # re-parse (miss)
        assert parse_cache.cache_info().currsize == n - 2


def test_single_entry_heavier_than_budget_not_cached(monkeypatch):
    monkeypatch.setattr(parse_cache, "_MAX_CACHE_BYTES", 100)
    with tempfile.TemporaryDirectory() as d:
        path = _make_py(d, "big.py", "x = 1\n" * 50)  # cost 400 * 16 > 100
        tree = parse_cache.parse_ast(path)
        assert tree is not None
        assert parse_cache.cache_info().currsize == 0


def test_read_and_parse_returns_consistent_pair():
    """F3 (2026-08-11): one stat key serves both values; the tree is parsed
    from the returned source string."""
    with tempfile.TemporaryDirectory() as d:
        path = _make_py(d, "a.py", "x = 1\n")
        src, tree = parse_cache.read_and_parse(path)
        assert src == "x = 1\n"
        assert tree is not None
        # Later single calls hit the same cached objects.
        assert parse_cache.read_source(path) is src
        assert parse_cache.parse_ast(path) is tree


def test_read_and_parse_none_for_missing_file():
    assert parse_cache.read_and_parse("/no/such/file/at/all.py") == (None, None)


def test_read_and_parse_none_on_syntax_error():
    with tempfile.TemporaryDirectory() as d:
        path = _make_py(d, "bad.py", "def (:\n")
        src, tree = parse_cache.read_and_parse(path)
        assert src is not None
        assert tree is None


# ── Disk-cache path guard (P-1, 2026-08-16) ─────────────────────────────────


def test_cache_file_path_returns_guarded_path(tmp_path):
    path = parse_cache.cache_file_path(str(tmp_path), "x_v1.json")
    assert path == str(Path(tmp_path) / ".cache" / "x_v1.json")


def test_cache_file_path_rejects_empty_repo_root():
    with pytest.raises(parse_cache.CachePathError):
        parse_cache.cache_file_path("", "x.json")


def test_cache_file_path_rejects_empty_filename(tmp_path):
    with pytest.raises(parse_cache.CachePathError):
        parse_cache.cache_file_path(str(tmp_path), "")


def test_cache_file_path_rejects_directory_components(tmp_path):
    for bad in ("sub/x.json", "../x.json", "a/../x.json", "/abs/x.json"):
        with pytest.raises(parse_cache.CachePathError):
            parse_cache.cache_file_path(str(tmp_path), bad)


def test_cache_file_path_rejects_non_json_name(tmp_path):
    for bad in ("x.txt", "x", "x.json.bak"):
        with pytest.raises(parse_cache.CachePathError):
            parse_cache.cache_file_path(str(tmp_path), bad)


def test_cache_file_path_rejects_file_as_repo_root(tmp_path):
    f = tmp_path / "not_a_dir"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(parse_cache.CachePathError):
        parse_cache.cache_file_path(str(f), "x.json")


def test_cache_file_path_rejects_escaping_symlink(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (tmp_path / ".cache").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink not permitted on this platform")
    with pytest.raises(parse_cache.CachePathError):
        parse_cache.cache_file_path(str(tmp_path), "x.json")


# ── Partial-update persistence policy (round 32-F2) ─────────────────────────


def test_should_persist_partial_update_thresholds():
    """Persist when dirty>0 AND (small corpus OR fraction exceeds 5%)."""
    from external_llm.analysis import parse_cache as pc

    sp = pc.should_persist_partial_update
    assert sp(0, 0) is False, "nothing dirty — never save"
    assert sp(1, 3) is True, "small corpus always persists (test fixtures)"
    assert sp(49, 49) is True, "corpus below SAVE_SKIP_MIN_ENTRIES persists"
    assert sp(1, 100) is False, "1% of a large corpus — recompute is cheaper"
    assert sp(5, 100) is False, "exactly 5% still skips (must EXCEED the fraction)"
    assert sp(6, 100) is True, "6% of a large corpus persists"
    assert sp(100, 100) is True, "cold scan (100% dirty) persists"


# ── Fingerprint order contract (B1) ──────────────────────────────────────────


def test_stat_fingerprint_missing_returns_none(tmp_path):
    assert parse_cache.stat_fingerprint(str(tmp_path / "nope.py")) is None


def test_stat_fingerprint_matches_os_stat(tmp_path):
    import os

    p = _make_py(str(tmp_path), "a.py", "x = 1\n")
    st = os.stat(p)
    assert parse_cache.stat_fingerprint(p) == (st.st_mtime_ns, st.st_size)


def test_read_with_fingerprint_consistent_pair(tmp_path):
    p = _make_py(str(tmp_path), "a.py", "x = 1\n")
    pair = parse_cache.read_with_fingerprint(p)
    assert pair is not None
    src, fp = pair
    assert src == "x = 1\n"
    assert fp == parse_cache.stat_fingerprint(p)


def test_read_with_fingerprint_missing_returns_none(tmp_path):
    assert parse_cache.read_with_fingerprint(str(tmp_path / "nope.py")) is None


def test_read_with_fingerprint_stamp_never_newer_than_content(tmp_path, monkeypatch):
    """B1 order contract: an interleaved write must leave the returned stamp
    describing the PRE-read state, so a torn cache entry is unreachable.

    The write is injected into ``_read_impl`` — i.e. it lands between the
    helper's stat and its read.  If the order were flipped (read, then stat),
    the returned fingerprint would equal the file's post-write stat and a
    cache entry keyed by it would HIT on every future run while holding
    pre-write analysis — this test fails on that regression.
    """
    import os

    p = _make_py(str(tmp_path), "a.py", "x = 1\n")
    real_read = parse_cache._read_impl

    def interleaving_write(abs_path, mtime_ns, size):
        with open(abs_path, "a", encoding="utf-8") as fh:
            fh.write("# concurrent write\n")
        return real_read(abs_path, mtime_ns, size)

    monkeypatch.setattr(parse_cache, "_read_impl", interleaving_write)
    pair = parse_cache.read_with_fingerprint(p)
    monkeypatch.undo()
    assert pair is not None
    src, fp = pair
    assert "# concurrent write" in src, "read must observe the post-write content"
    st = os.stat(p)
    assert fp != (st.st_mtime_ns, st.st_size), (
        "stamp must describe the PRE-read state — a post-write stamp paired "
        "with this content would silently serve stale cache entries"
    )


@pytest.mark.parametrize(
    "module_name,helper",
    [
        ("external_llm.analysis.vulture_scanner", "_stat_fingerprint"),
        ("external_llm.analysis.unused_import_scanner", "_uic_stat"),
        ("external_llm.analysis._dead_block_shared", "_dbx_stat"),
        ("external_llm.analysis.container_reachability_scanner", "_crx_stat"),
        ("external_llm.analysis.cross_file_refs", "_ie_stat"),
    ],
)
def test_scanner_stat_helpers_delegate_to_canonical(module_name, helper, tmp_path):
    """Every per-file disk cache stats through the ONE canonical helper."""
    import importlib

    mod = importlib.import_module(module_name)
    fn = getattr(mod, helper)
    assert fn(str(tmp_path / "nope.py")) is None
    p = _make_py(str(tmp_path), "a.py", "x = 1\n")
    assert fn(p) == parse_cache.stat_fingerprint(p)
