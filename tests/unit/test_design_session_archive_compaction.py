"""P28-3: bounded design-session archive via compaction.

The append-only archive (<sid>.archive.jsonl) grows without bound on
long-lived sessions — content-history search (load_archived_turns) reads the
whole file and the BM25 cache holds the tokenised copy.  Past
``_ARCHIVE_MAX_BYTES`` the OLDEST records are folded into compacted summary
records (tail preserved verbatim), with the absolute-index invariant
``_archive_last_index`` relies on kept intact.
"""
from __future__ import annotations

import json

import pytest

from external_llm.design_session import DesignSessionManager


@pytest.fixture
def mgr(tmp_path):
    return DesignSessionManager(repo_root=str(tmp_path))


def _write(mgr, sid, records):
    path = mgr._archive_path(sid)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    return path


def _records(n: int, content_len: int = 80):
    return [
        {
            "i": j,
            "role": "user" if j % 2 == 0 else "assistant",
            "timestamp": 1000.0 + j,
            "content": f"turn-{j} " + "x" * content_len,
        }
        for j in range(n)
    ]


def test_archive_below_cap_untouched(mgr):
    _write(mgr, "s1", _records(5))
    path = mgr._archive_path("s1")
    before = path.read_text(encoding="utf-8")
    assert mgr._compact_archive("s1") is False
    assert path.read_text(encoding="utf-8") == before


def test_archive_compaction_folds_oldest_keeps_tail(mgr, monkeypatch):
    monkeypatch.setattr(mgr, "_ARCHIVE_MAX_BYTES", 3000)
    records = _records(60)
    _write(mgr, "s1", records)
    path = mgr._archive_path("s1")
    assert path.stat().st_size > 3000  # over the cap — compaction must fire

    assert mgr._compact_archive("s1") is True
    assert path.stat().st_size <= 3000  # folded back under the cap

    lines = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Newest records preserved verbatim (tail is what history search needs).
    assert lines[-1]["i"] == 59
    assert lines[-1]["content"] == records[-1]["content"]
    assert not lines[-1].get("compacted")
    # Oldest records folded into summary records.
    assert any(r.get("compacted") for r in lines)
    for r in lines:
        if r.get("compacted"):
            assert len(r["content"]) <= 200  # excerpt stays BM25-searchable
            assert r["i"] < 59  # folded region is the oldest
    # Already under the cap → no-op.
    assert mgr._compact_archive("s1") is False


def test_compaction_preserves_archive_index_invariant(mgr, monkeypatch):
    """After compaction, _archive_last_index still returns the true last
    absolute index — the crash-recovery dedup guard (abs_i <= last_i) must
    not re-append already-archived turns."""
    monkeypatch.setattr(mgr, "_ARCHIVE_MAX_BYTES", 3000)
    _write(mgr, "s1", _records(60))
    mgr._compact_archive("s1")
    assert DesignSessionManager._archive_last_index(mgr._archive_path("s1")) == 59


def test_archive_append_after_compaction_no_duplicates(mgr, monkeypatch):
    """Archiving NEW turns after a compaction must continue from the last
    absolute index (monotonic), not re-append anything."""
    monkeypatch.setattr(mgr, "_ARCHIVE_MAX_BYTES", 3000)
    _write(mgr, "s1", _records(60))
    mgr._compact_archive("s1")
    path = mgr._archive_path("s1")
    last_i = DesignSessionManager._archive_last_index(path)

    session = mgr.get_or_create("s1")
    # 3 new compressed-but-unarchived turns (absolute 60..62).
    session.turns = [
        {"role": "user", "content": "new-60", "timestamp": 2000.0},
        {"role": "assistant", "content": "new-61", "timestamp": 2001.0},
        {"role": "user", "content": "new-62", "timestamp": 2002.0},
    ]
    session.compressed_up_to = 63
    session.archived_count = 60  # 60 turns already on disk in the archive
    mgr._archive_compressed_turns(session)

    # The 3 new records were appended with correct absolute indices.
    lines = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [r["i"] for r in lines if r["i"] > last_i] == [60, 61, 62]
    assert DesignSessionManager._archive_last_index(path) == 62
    # No duplicate of the pre-compaction region (indices stay strictly
    # increasing).
    idxs = [r["i"] for r in lines]
    assert idxs == sorted(idxs)


def test_startup_sweep_compacts_overgrown_archive(tmp_path, monkeypatch):
    """P27-4: an abandoned session's overgrown archive is compacted at manager
    startup — compaction previously ran only on the write path (_save), so a
    dead session stayed over _ARCHIVE_MAX_BYTES forever (measured: 25 MB)."""
    monkeypatch.setattr(DesignSessionManager, "_ARCHIVE_MAX_BYTES", 3000)
    records = _records(60)
    sessions_dir = tmp_path / ".asicode" / "design_sessions"
    sessions_dir.mkdir(parents=True)
    (sessions_dir / "dead.archive.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    assert (sessions_dir / "dead.archive.jsonl").stat().st_size > 3000

    mgr = DesignSessionManager(repo_root=str(tmp_path))
    archive = mgr._archive_path("dead")
    assert archive.stat().st_size <= 3000
    lines = [
        json.loads(line)
        for line in archive.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Tail preserved verbatim — the archive-index invariant survives the sweep.
    assert lines[-1]["i"] == 59
    assert lines[-1]["content"] == records[-1]["content"]
    assert any(r.get("compacted") for r in lines)


def test_startup_sweep_skips_under_cap_archives(tmp_path, monkeypatch):
    """Sweep must be a no-op for archives under the cap (one stat each)."""
    monkeypatch.setattr(DesignSessionManager, "_ARCHIVE_MAX_BYTES", 3000)
    sessions_dir = tmp_path / ".asicode" / "design_sessions"
    sessions_dir.mkdir(parents=True)
    (sessions_dir / "small.archive.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in _records(5)),
        encoding="utf-8",
    )
    before = (sessions_dir / "small.archive.jsonl").read_text(encoding="utf-8")
    DesignSessionManager(repo_root=str(tmp_path))
    assert (sessions_dir / "small.archive.jsonl").read_text(encoding="utf-8") == before
