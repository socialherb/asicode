"""Root-partitioned fair cache admission — multi-repo starvation fix (2026-08-12).

The three process-wide caches (RepositoryGraph ``_extract_cache`` /
``_names_cache``, CallGraphIndexer ``_file_cache``) previously admitted the
first-N live files globally: in the webapp ONE process serves MANY repos, so
once earlier repos filled the cap every later repo was refused for the whole
process lifetime — each of its builds re-parsed everything and re-loaded its
multi-MB snapshot.  RootCache gives every active root a guaranteed quota
(cap // roots), shares overflow, and lets a starved root claim slots from the
most-over-quota hoarder.

The unit tests pin the policy contracts; the last test reproduces the
starvation regression end-to-end through RepositoryGraph.
"""

import pytest

from external_llm.graph.repository_graph import RepositoryGraph, _extract_cache
from external_llm.graph.root_cache import RootCache

R1 = "/repo/a"
R2 = "/repo/b"
R3 = "/repo/c"


@pytest.fixture(autouse=True)
def isolated_extract_cache():
    """Save/restore the process-wide extraction cache + cap."""
    saved = dict(_extract_cache)
    saved_cap = _extract_cache.cap
    _extract_cache.clear()
    yield
    _extract_cache.clear()
    _extract_cache.cap = saved_cap
    _extract_cache.update(saved)


# ── policy contracts (pure RootCache units) ────────────────────────────────


def test_single_root_invariant_matches_old_cap():
    """One root: quota == cap and no other root — refuse beyond cap, no eviction."""
    c = RootCache(cap=10)
    for i in range(10):
        assert c.admit((R1, f"f{i}"), i)
    assert len(c) == 10
    assert c.admit((R1, "f10"), 10) is False, "single root must refuse beyond cap"
    assert len(c) == 10
    assert c.count(R1) == 10, "no eviction in single-root mode"


def test_starved_repo_claims_slot_from_hoarder():
    """Regression core: root2 arriving after root1 filled the cap must get in."""
    c = RootCache(cap=10)
    for i in range(10):
        c.admit((R1, f"f{i}"), i)
    assert c.admit((R2, "g0"), "g0") is True, "starved root must claim a slot"
    assert c.count(R2) == 1
    assert c.count(R1) == 9, "the slot came from the hoarder"


def test_converges_to_fair_quota():
    """Two repos converge to quota (cap//2) each, not 0/10."""
    c = RootCache(cap=10)
    for i in range(10):
        c.admit((R1, f"f{i}"), i)  # R1 takes everything (single-root overflow)
    for i in range(5):
        assert c.admit((R2, f"g{i}"), i)  # R2 squeezes R1 down to quota
    assert c.admit((R2, "g5"), 5) is False, "equilibrium: both at quota, cache full"
    assert c.count(R1) == 5
    assert c.count(R2) == 5
    assert len(c) == 10


def test_refusal_at_equilibrium():
    """All roots at quota and the cache full: further admits refused (stable)."""
    c = RootCache(cap=10)
    for i in range(5):
        c.admit((R1, f"f{i}"), i)
    for i in range(5):
        c.admit((R2, f"g{i}"), i)
    assert c.admit((R1, "f5"), 5) is False
    assert c.admit((R2, "g5"), 5) is False
    assert len(c) == 10
    assert c.count(R1) == 5 and c.count(R2) == 5


def test_overflow_sharing_before_full():
    """While the cache has room, one root may exceed its quota."""
    c = RootCache(cap=10)
    for i in range(8):
        c.admit((R1, f"f{i}"), i)  # quota is 5, but room exists
    assert c.count(R1) == 8
    for i in range(2):
        c.admit((R2, f"g{i}"), i)
    assert c.count(R1) == 8 and c.count(R2) == 2
    assert len(c) == 10


def test_eviction_targets_most_over_quota():
    """Three roots: the hoarder (not the at-quota root) yields first."""
    c = RootCache(cap=12)  # quota 4 each
    for i in range(6):
        c.admit((R1, f"f{i}"), i)
    for i in range(4):
        c.admit((R2, f"g{i}"), i)
    for i in range(2):
        c.admit((R3, f"h{i}"), i)
    # R1=6 (hoarder, +2), R2=4 (at quota), R3=2 (starved), total 12 == cap
    assert c.admit((R3, "h2"), 2) is True
    assert c.count(R1) == 5, "most-over-quota root yields first"
    assert c.count(R2) == 4, "at-quota root is untouched"
    assert c.count(R3) == 3


def test_registry_bound_drops_coldest_root():
    """max_roots=2: a third root drops the coldest (least recently admitted)."""
    c = RootCache(cap=10, max_roots=2)
    for i in range(6):
        c.admit((R1, f"f{i}"), i)
    for i in range(4):
        c.admit((R2, f"g{i}"), i)
    assert c.admit((R3, "h0"), "h0") is True
    assert c.count(R1) == 0, "coldest root dropped entirely"
    assert c.count(R2) == 4
    assert c.count(R3) == 1


def test_sweep_dead_removes_dead_and_empty_roots(tmp_path):
    live = tmp_path / "live.py"
    live.write_text("x = 1\n", encoding="utf-8")
    dead = tmp_path / "dead.py"  # never created
    c = RootCache(cap=10)
    c[(str(tmp_path), str(live))] = (live.stat().st_mtime_ns, 1, {})
    c[(str(tmp_path), str(dead))] = (1, 1, {})
    c[("/gone", "x")] = (1, 1, {})
    assert c.sweep_dead() == 2
    assert (str(tmp_path), str(live)) in c
    assert len(c) == 1
    assert all(k[0] != "/gone" for k in c), "root left empty must be dropped"


def test_admit_refreshes_existing_entry():
    c = RootCache(cap=2)
    assert c.admit((R1, "f0"), 0)
    assert c.admit((R1, "f1"), 1)
    assert c.admit((R1, "f0"), 2) is True, "refreshing an existing key must not evict"
    assert c.get((R1, "f0")) == 2
    assert len(c) == 2


def test_flat_dict_surface_roundtrip():
    """The flat dict API (used by direct-access tests) still works."""
    c = RootCache(cap=10)
    c[(R1, "f0")] = "v0"
    c[(R1, "f1")] = "v1"
    c[(R2, "g0")] = "w0"
    assert len(c) == 3
    assert (R1, "f0") in c
    assert c.get((R1, "f0")) == "v0"
    assert c.get((R1, "nope")) is None
    assert c.pop((R2, "g0")) == "w0"
    assert (R2, "g0") not in c
    assert set(c) == {(R1, "f0"), (R1, "f1")}
    assert dict(c) == {(R1, "f0"): "v0", (R1, "f1"): "v1"}
    saved = dict(c)
    c.clear()
    assert len(c) == 0
    c.update(saved)
    assert list(c.items()) == [((R1, "f0"), "v0"), ((R1, "f1"), "v1")]


# ── end-to-end regression ──────────────────────────────────────────────────


def test_multi_repo_starvation_regression(tmp_path):
    """A repo arriving after another filled the cache must get in-cache hits.

    Pre-fix: root2's EVERY build re-parsed all 30 files (0 hits forever —
    the flat cap stayed full of root1's entries).  Post-fix: root2 claims
    its quota share and the next build serves it from the in-process tier.
    """
    repo1 = tmp_path / "r1"
    repo2 = tmp_path / "r2"
    repo1.mkdir()
    repo2.mkdir()
    for i in range(30):
        (repo1 / f"m{i}.py").write_text(f"def f{i}():\n    return {i}\n", encoding="utf-8")
        (repo2 / f"m{i}.py").write_text(f"def g{i}():\n    return {i}\n", encoding="utf-8")

    _extract_cache.cap = 40  # small enough that one repo alone fills it
    g1 = RepositoryGraph(str(repo1))
    g1.build(collect_imported_names=True)  # track=True -> cache_stats counted
    assert _extract_cache.count(str(repo1)) == 30, "root1 fills the cache"

    g2 = RepositoryGraph(str(repo2))
    g2.build(collect_imported_names=True)  # must claim slots (evicting root1's overflow)
    assert _extract_cache.count(str(repo2)) > 0

    g3 = RepositoryGraph(str(repo2))
    g3.build(collect_imported_names=True)
    assert g3.cache_stats["hit"] > 0, "repo2 must not be starved by repo1"
    assert _extract_cache.count(str(repo1)) > 0, "fair: repo1 keeps its quota share"
