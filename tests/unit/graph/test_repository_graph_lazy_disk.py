"""Tests for lazy on-disk snapshot loading (P0, 2026-08-11).

``build()`` no longer pre-loads ``.cache/structural_graph_v1.json``: a warm
process whose in-process caches cover every file never touches the JSON (on
asicode that was a 25MB load + ~170MB of transient allocations per rebuild,
for data nothing would read — 0.205s → 0.023s).  These tests pin:

* a warm rebuild NEVER calls ``_load_structural_cache`` (plain AND gate mode —
  the rewrite hint + save fast-skip decide from the ``_disk_manifest_lens``
  memo and an existence probe, without loading the payload);
* a cold gate build still loads once and serves both files from the disk
  tier; and
* a deleted snapshot self-heals on the next warm rebuild (the existence probe
  must not let the memo suppress the rewrite).
* a SECOND RepositoryGraph over the same repo in the same process (A5,
  2026-08-12) inherits the process-wide manifest-length memo, so a warm
  in-process rebuild never loads the JSON either.
"""
import json
import textwrap

import pytest

from external_llm.graph import repository_graph as rg_module
from external_llm.graph.repository_graph import (
    RepositoryGraph,
    _extract_cache,
    _names_cache,
)


@pytest.fixture(autouse=True)
def isolated_extract_cache():
    """Save/restore the process-wide caches and gc rate-limit around every test."""
    saved = dict(_extract_cache)
    saved_names = dict(_names_cache)
    saved_deficit = rg_module._extract_cache_gc_deficit
    _extract_cache.clear()
    _names_cache.clear()
    rg_module._extract_cache_gc_deficit = 0
    yield
    _extract_cache.clear()
    _extract_cache.update(saved)
    _names_cache.clear()
    _names_cache.update(saved_names)
    rg_module._extract_cache_gc_deficit = saved_deficit


def _make_repo(tmp_path, files: dict) -> str:
    for rel, src in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(src), encoding="utf-8")
    return str(tmp_path)


def _count_loads(monkeypatch):
    """Count calls to the JSON loader — the thing P0 makes unnecessary."""
    loads = {"n": 0}
    orig = rg_module._load_structural_cache

    def counting(path):
        loads["n"] += 1
        return orig(path)

    monkeypatch.setattr(rg_module, "_load_structural_cache", counting)
    return loads


def test_warm_gate_rebuild_never_loads_disk_snapshot(tmp_path, monkeypatch):
    """build #2 (gate mode, nothing changed) must not touch the JSON at all."""
    repo = _make_repo(
        tmp_path,
        {"a.py": "def fa():\n    return 1\n", "b.py": "def fb():\n    return 1\n"},
    )
    g = RepositoryGraph(repo)
    g.build(collect_imported_names=True)  # cold: parse everything, write snapshot
    cache_path = tmp_path / ".cache" / "structural_graph_v1.json"
    assert cache_path.exists()
    blob_before = cache_path.read_bytes()

    loads = _count_loads(monkeypatch)
    g.build(collect_imported_names=True)  # warm: nothing changed
    assert loads["n"] == 0, "warm rebuild must never load the JSON"
    assert g._disk_cache is None, "payload never materialized"
    assert g.cache_stats["hit"] == 2
    assert g.cache_stats["changed"] == 0
    assert cache_path.read_bytes() == blob_before  # byte-identical, no rewrite


def test_plain_warm_rebuild_never_loads_disk_snapshot(tmp_path, monkeypatch):
    """The app hot path (facade rebuild after every edit) never loads the JSON."""
    repo = _make_repo(tmp_path, {"a.py": "def fa():\n    return 1\n"})
    g = RepositoryGraph(repo)
    g.build(collect_imported_names=True)  # create the snapshot
    loads = _count_loads(monkeypatch)
    g.build()  # plain mode — read-only, no names pass, no save path
    assert loads["n"] == 0
    assert g._disk_cache is None


def test_cold_gate_build_still_loads_and_serves_disk_tier(tmp_path, monkeypatch):
    """A fresh process (empty in-process caches) still warms from the JSON."""
    repo = _make_repo(
        tmp_path,
        {"a.py": "def fa():\n    return 1\n", "b.py": "def fb():\n    return 1\n"},
    )
    RepositoryGraph(repo).build(collect_imported_names=True)  # writes snapshot
    _extract_cache.clear()  # simulate a fresh process: in-process tier empty

    loads = _count_loads(monkeypatch)
    g2 = RepositoryGraph(repo)  # NEW instance
    g2.build(collect_imported_names=True)
    assert loads["n"] == 1, "one lazy load serves both files"
    assert g2.cache_stats["hit"] == 2  # both served from the disk tier
    assert g2.cache_stats["changed"] == 0


def test_second_instance_warm_inprocess_never_loads_json(tmp_path, monkeypatch):
    """A5 (2026-08-12): the manifest-length memo is process-wide (keyed by
    cache path), so a SECOND instance over the same repo — with the same
    warm in-process extraction cache — must not re-load the JSON even in
    gate mode (previously its instance-local memo started at 0, firing the
    rewrite hint and forcing one 25MB-class load).
    """
    repo = _make_repo(
        tmp_path,
        {"a.py": "def fa():\n    return 1\n", "b.py": "def fb():\n    return 1\n"},
    )
    RepositoryGraph(repo).build(collect_imported_names=True)  # writes snapshot

    loads = _count_loads(monkeypatch)
    g2 = RepositoryGraph(repo)  # NEW instance — in-process caches stay warm
    g2.build(collect_imported_names=True)
    assert loads["n"] == 0, "warm in-process second instance must not load JSON"
    assert g2._disk_cache is None
    assert g2.cache_stats["hit"] == 2
    assert g2.cache_stats["changed"] == 0


def test_deleted_snapshot_self_heals_on_warm_rebuild(tmp_path):
    """A warm build must recreate a deleted JSON — the existence probe in the
    rewrite hint must not let the manifest memo suppress the rewrite."""
    repo = _make_repo(tmp_path, {"a.py": "def fa():\n    return 1\n"})
    g = RepositoryGraph(repo)
    g.build(collect_imported_names=True)
    cache_path = tmp_path / ".cache" / "structural_graph_v1.json"
    assert cache_path.exists()
    cache_path.unlink()

    g.build(collect_imported_names=True)  # warm in-process, but JSON gone
    assert cache_path.exists(), "deleted snapshot must be recreated"
    blob = json.loads(cache_path.read_text())
    assert set(blob["manifest"]) == {"a.py"}


def test_invalid_snapshot_loads_once_not_per_file(tmp_path, monkeypatch):
    """A corrupt / version-mismatched snapshot must load ONCE per build.

    Regression (2026-08-12): the old code pinned the mtime marker to 0 on a
    failed load, so the per-file disk tier (``_disk_file_data`` /
    ``_imported_names_for``) re-read + re-parsed the whole JSON for EVERY
    file — on asicode (818 py files, ~25MB snapshot) the schema-version bump
    from new dataclass fields (P3 Stage 1 CallEdge additions) turned the
    structural-scanner commit gate into a 300s+ hang.  The marker must pin
    the CURRENT mtime instead: only a REWRITTEN file (new mtime) or a fresh
    build (marker reset to 0 in build()) retries the load.
    """
    repo = _make_repo(
        tmp_path,
        {"a.py": "def fa():\n    return 1\n", "b.py": "def fb():\n    return 1\n"},
    )
    RepositoryGraph(repo).build(collect_imported_names=True)  # valid snapshot
    cache_path = tmp_path / ".cache" / "structural_graph_v1.json"

    # Version-mismatch — exactly what a dataclass field addition does to old
    # snapshots (schema version = crc32 over dataclass field signatures).
    blob = json.loads(cache_path.read_text())
    blob["version"] = blob["version"] + 1
    cache_path.write_text(json.dumps(blob))
    # Simulate a fresh process: the in-process tiers are process-wide, so a
    # warm instance would never touch the disk tier (A5) and the test would
    # not exercise the failed-load path at all.
    _extract_cache.clear()
    _names_cache.clear()

    loads = _count_loads(monkeypatch)
    g2 = RepositoryGraph(repo)  # fresh instance: in-process tier empty
    g2.build(collect_imported_names=True)
    assert loads["n"] == 1, "invalid snapshot must load once per build, not once per file"
    assert g2.cache_stats["hit"] == 0
    assert g2.cache_stats["changed"] == 2  # fail-open: both re-parsed

    # The build rewrote the snapshot with the current schema — a NEW instance
    # now serves both files from the disk tier again.
    _extract_cache.clear()
    _names_cache.clear()
    loads = _count_loads(monkeypatch)
    g3 = RepositoryGraph(repo)
    g3.build(collect_imported_names=True)
    assert loads["n"] == 1
    assert g3.cache_stats["hit"] == 2
    assert g3.cache_stats["changed"] == 0


def test_cap_overflow_snapshot_converges_to_full_walk(tmp_path, monkeypatch, caplog):
    """Merge-preserving snapshot (P0, 2026-08-12): beyond-cap files must not
    be dropped from the on-disk snapshot.

    Regression: the snapshot was rebuilt from the in-process cache alone, so
    on a repo with more files than ``_EXTRACT_CACHE_MAX_ENTRIES`` the tail
    files were re-parsed on EVERY build in EVERY process — a permanent
    full-parse cliff (on a 9k-file repo that is minutes per gate run).  The
    rewrite now carries over stamp-valid disk payloads verbatim and persists
    beyond-cap fresh parses (``_pending_snapshot``), so the manifest
    converges to the full walked set and the next build serves everything
    from disk.
    """
    repo = _make_repo(
        tmp_path,
        {f"m{i}.py": f"def f{i}():\n    return {i}\n" for i in range(5)},
    )
    monkeypatch.setattr(rg_module, "_EXTRACT_CACHE_MAX_ENTRIES", 2)  # simulate N > cap
    monkeypatch.setattr(rg_module._extract_cache, "cap", 2)

    g1 = RepositoryGraph(repo)
    with caplog.at_level("WARNING"):
        g1.build(collect_imported_names=True)
    assert g1.cache_stats["total"] == 5
    assert g1.cache_stats["changed"] == 5  # every fresh parse counts now
    assert g1.cache_stats["parsed_uncapped"] == 3  # 5 walked - 2 admitted
    assert g1.cache_stats["hit"] == 0
    assert any("re-parsed beyond" in r.message for r in caplog.records), (
        "cap-pressure WARNING expected"
    )

    cache_path = tmp_path / ".cache" / "structural_graph_v1.json"
    blob = json.loads(cache_path.read_text())
    assert len(blob["manifest"]) == 5, "snapshot must converge to the full walked set, not the cap"
    assert set(blob["manifest"]) == set(blob["files"])

    # Fresh process: EVERY file (admitted or not) now serves from disk.
    _extract_cache.clear()
    _names_cache.clear()
    g2 = RepositoryGraph(repo)
    g2.build(collect_imported_names=True)
    assert g2.cache_stats["hit"] == 5
    assert g2.cache_stats["changed"] == 0
    assert g2.cache_stats["parsed_uncapped"] == 0

    # A later partial change must not evict beyond-cap files (carry-over
    # branch): change one ADMITTED file, fresh process.
    (tmp_path / "m0.py").write_text("def f0():\n    return 10\n", encoding="utf-8")
    _extract_cache.clear()
    _names_cache.clear()
    g3 = RepositoryGraph(repo)
    g3.build(collect_imported_names=True)
    assert g3.cache_stats["changed"] == 1
    assert g3.cache_stats["hit"] == 4
    blob = json.loads(cache_path.read_text())
    assert len(blob["manifest"]) == 5, "carry-over must keep beyond-cap coverage"
    assert set(blob["manifest"]) == set(blob["files"])


def test_cache_stats_reset_keeps_parsed_uncapped_key(tmp_path):
    """Minor 1 (2026-08-12): build()'s stats reset must keep parsed_uncapped.

    __init__ seeds it; the per-build reset dict used to omit it, so a build
    interrupted by an exception left the key missing for callers.
    """
    repo = _make_repo(tmp_path, {"a.py": "def fa():\n    pass\n"})
    g = RepositoryGraph(repo)
    g.build()
    assert g.cache_stats["parsed_uncapped"] == 0
    assert g.cache_stats["total"] == 1


def test_failed_load_zeroes_manifest_len_memo(tmp_path):
    """Minor 3 (2026-08-12): a failed snapshot load must memo 0, not stale.

    The file exists but is corrupt/version-mismatched: ``_disk_cache`` is
    None but the stale manifest-length memo survived, so build()'s rewrite
    hint compared against the OLD length (the OSError path already memoed 0).
    """
    repo = _make_repo(tmp_path, {"a.py": "def fa():\n    pass\n"})
    RepositoryGraph(repo).build(collect_imported_names=True)  # valid snapshot
    cache_path = tmp_path / ".cache" / "structural_graph_v1.json"
    blob = json.loads(cache_path.read_text())
    blob["version"] = blob["version"] + 1  # version-mismatch -> load fails
    cache_path.write_text(json.dumps(blob))

    _extract_cache.clear()
    _names_cache.clear()
    g = RepositoryGraph(repo)  # fresh instance: in-process tier empty
    rg_module._disk_manifest_lens[cache_path] = 999  # stale memo
    g._load_disk_cache_snapshot()
    assert g._disk_cache is None
    assert rg_module._disk_manifest_lens.get(cache_path) == 0
