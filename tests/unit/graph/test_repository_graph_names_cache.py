"""Names-cache admission control and GC (C1, 2026-08-12).

``_names_cache`` (process-wide imported-name memo) previously cleared ITSELF
at the cap — the exact thrash pattern the extraction cache abandoned (see
``_gc_extract_cache``'s docstring): every cap-sized rebuild re-populated the
cache from zero and re-read the disk JSON each time.  Now it mirrors the
extraction cache's admission contract: the cap is enforced at the INSERT
sites by sweeping dead entries and refusing new ones when still full — never
by clearing.  These tests pin:

* a sweep removes only entries whose source file no longer exists;
* the sweep is rate-limited (one full sweep per ``cap`` calls);
* a full cache refuses NEW entries but still returns the computed names
  (fail-open: caching is a speed optimization, never a correctness path);
* the disk-tier insert is capped too (it previously skipped the check);
* previously-cached entries survive a full-cache build and keep serving
  without recompute.
"""
import os
import textwrap

import pytest

from external_llm.analysis import cross_file_refs
from external_llm.graph import repository_graph as rg_module
from external_llm.graph.repository_graph import (
    RepositoryGraph,
    _gc_names_cache,
    _names_cache,
)


@pytest.fixture(autouse=True)
def isolated_names_cache():
    """Save/restore the process-wide names cache and gc rate-limit."""
    saved = dict(_names_cache)
    saved_deficit = rg_module._names_cache_gc_deficit
    _names_cache.clear()
    rg_module._names_cache_gc_deficit = 0
    yield
    _names_cache.clear()
    _names_cache.update(saved)
    rg_module._names_cache_gc_deficit = saved_deficit


def _make_repo(tmp_path, files: dict) -> str:
    for rel, src in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(src), encoding="utf-8")
    return str(tmp_path)


def _fill_to_cap(tmp_path, cap: int) -> None:
    """Fill the cache with ``cap`` live (existing) entries."""
    for i in range(cap):
        p = tmp_path / f"filler{i}.py"
        p.write_text("x = 1\n", encoding="utf-8")
        st = os.stat(p)
        _names_cache[(str(tmp_path), str(p))] = (st.st_mtime_ns, st.st_size, {str(p)})


def test_gc_sweeps_only_dead_entries(tmp_path):
    live = tmp_path / "live.py"
    live.write_text("x = 1\n", encoding="utf-8")
    st = os.stat(live)
    dead = tmp_path / "dead.py"  # never created
    _names_cache[(str(tmp_path), str(live))] = (st.st_mtime_ns, st.st_size, {"live"})
    _names_cache[(str(tmp_path), str(dead))] = (1, 1, {"dead"})

    _gc_names_cache()

    assert (str(tmp_path), str(live)) in _names_cache, "live entry must survive the sweep"
    assert (str(tmp_path), str(dead)) not in _names_cache, "dead entry must be swept"


def test_gc_rate_limits_sweeps(tmp_path, monkeypatch):
    """One full sweep per cap calls: the next call while deficit > 0 skips."""
    monkeypatch.setattr(rg_module, "_EXTRACT_CACHE_MAX_ENTRIES", 8)
    dead = tmp_path / "dead.py"
    _names_cache[(str(tmp_path), str(dead))] = (1, 1, {"dead"})

    _gc_names_cache()  # deficit 0 -> full sweep
    assert (str(tmp_path), str(dead)) not in _names_cache
    assert rg_module._names_cache_gc_deficit == 8

    _names_cache[(str(tmp_path), str(dead))] = (1, 1, {"dead"})
    _gc_names_cache()  # deficit > 0 -> skipped (no stat work)
    assert (str(tmp_path), str(dead)) in _names_cache, "sweep must be skipped while deficit > 0"
    assert rg_module._names_cache_gc_deficit == 7


def test_compute_tier_insert_refused_when_full(tmp_path, monkeypatch):
    """A full cache refuses NEW entries but still returns computed names."""
    monkeypatch.setattr(rg_module._names_cache, "cap", 8)
    repo = _make_repo(tmp_path, {"b.py": "import os\n"})
    _fill_to_cap(tmp_path, 8)
    g = RepositoryGraph(repo)
    b = tmp_path / "b.py"
    st = os.stat(b)

    names = g._imported_names_for("b.py", str(b), st)

    assert names is not None, "fail-open: names are returned even when not cached"
    assert len(_names_cache) == 8, "cache must stay at the cap"
    assert (str(tmp_path), str(b)) not in _names_cache, "new entry must be refused at the cap"


def test_disk_tier_insert_refused_when_full(tmp_path, monkeypatch):
    """The disk-tier insert (previously unchecked) is capped too."""
    monkeypatch.setattr(rg_module._names_cache, "cap", 8)
    repo = _make_repo(tmp_path, {"a.py": "def fa():\n    return 1\n"})
    a = tmp_path / "a.py"
    st = os.stat(a)
    _fill_to_cap(tmp_path, 8)
    g = RepositoryGraph(repo)
    g._disk_cache = {  # pre-set so _ensure_disk_cache is a no-op
        "manifest": {"a.py": [st.st_mtime_ns, st.st_size]},
        "imported_names": {"a.py": ["fa"]},
    }

    names = g._imported_names_for("a.py", str(a), st)

    assert names == {"fa"}, "disk tier still serves the names (fail-open)"
    assert len(_names_cache) == 8, "disk-tier insert must be refused at the cap"
    assert (str(tmp_path), str(a)) not in _names_cache


def test_full_cache_keeps_serving_cached_entries_without_recompute(
    tmp_path, monkeypatch
):
    """The thrash fix: an at-cap build must not evict previously-cached names."""
    monkeypatch.setattr(rg_module._names_cache, "cap", 8)
    repo = _make_repo(tmp_path, {"a.py": "def fa():\n    return 1\n"})
    a = tmp_path / "a.py"
    st = os.stat(a)
    g = RepositoryGraph(repo)

    first = g._imported_names_for("a.py", str(a), st)
    assert (str(tmp_path), str(a)) in _names_cache, "warmup must cache a.py"

    calls = {"n": 0}
    orig = cross_file_refs.extract_imported_names_for_file

    def counting(path):
        calls["n"] += 1
        return orig(path)

    monkeypatch.setattr(cross_file_refs, "extract_imported_names_for_file", counting)
    calls["n"] = 0  # ignore the warmup compute

    _fill_to_cap(tmp_path, 7)  # 7 fillers + a.py = cap
    b = tmp_path / "b.py"
    b.write_text("import os\n", encoding="utf-8")
    st_b = os.stat(b)
    g._imported_names_for("b.py", str(b), st_b)  # new file: refused at cap

    assert (str(tmp_path), str(a)) in _names_cache, "previously-cached entry must survive"
    assert len(_names_cache) == 8

    again = g._imported_names_for("a.py", str(a), st)
    assert calls["n"] == 1, "cached entry must be served without recompute"
    assert again == first
