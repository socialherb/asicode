"""RED→GREEN coverage for external_llm/design_session.py (76% → 100%).

Targets every line the four dedicated suites leave uncovered: the
get_or_create refresh crash on never-written sessions (DS-1), corrupt-load
fallbacks, adopt/merge branches, add_turn optional fields and guarded
failure paths, delegation wiring to SessionCompressionContext, list_sessions
ordering/fallbacks, _save, archive-cut behavior, compaction failure modes,
archive tail-index hardening, load_archived_turns, and delete_session.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from external_llm.design_session import DesignSession, DesignSessionManager

_requires_file_perms = pytest.mark.skipif(
    getattr(__import__("os"), "geteuid", lambda: -1)() == 0,
    reason="root ignores file permission bits",
)


@pytest.fixture
def mgr(tmp_path):
    return DesignSessionManager(repo_root=str(tmp_path))


def _turn(role="user", content="x", ts=1.0, **extra):
    t = {"role": role, "content": content, "timestamp": ts}
    t.update(extra)
    return t


def _write_disk(mgr, sid, **fields):
    """Write a raw session JSON as if another process had saved it."""
    data = {
        "session_id": sid,
        "created_at": 1.0,
        "updated_at": 1.0,
        "turns": [],
        "compressed_summary": "",
        "compressed_up_to": 0,
        "archived_count": 0,
        "decisions": [],
        "chat_mode": "code",
    }
    data.update(fields)
    path = mgr._session_path(sid)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ── DS-1: refresh crash on cached-but-never-written sessions ─────────────


class TestGetOrCreateRefresh:
    def test_second_get_or_create_before_first_write_does_not_crash(self, mgr):
        # A brand-new session has no disk file and no recorded mtime; the
        # second cache hit must not blow up while stat-ing the missing file.
        s1 = mgr.get_or_create("brand-new")
        s2 = mgr.get_or_create("brand-new")
        assert s1 is s2

    def test_get_or_create_after_corrupt_file_returns_fresh(self, mgr):
        mgr._session_path("bad").write_text("{not json", encoding="utf-8")
        session = mgr.get_or_create("bad")
        assert session.session_id == "bad"
        assert session.turns == []

    def test_load_raw_corrupt_json_returns_none(self, mgr):
        mgr._session_path("bad").write_text("{not json", encoding="utf-8")
        assert mgr._load_raw("bad") is None

    def test_refresh_if_stale_absorb_failure_is_swallowed(self, mgr, monkeypatch):
        session = mgr.get_or_create("s")
        monkeypatch.setattr(mgr, "_adopt_from_disk", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        _write_disk(mgr, "s", turns=[_turn(content="disk")])  # mtime now differs
        mgr._refresh_if_stale(session)  # must not raise
        assert "disk" not in [t["content"] for t in session.turns]


# ── _adopt_from_disk merge branches ──────────────────────────────────────


class TestAdoptFromDisk:
    def test_local_archive_progress_wins_as_base(self, mgr):
        session = mgr.get_or_create("s")
        session.turns = [_turn(content="l0", ts=1.0), _turn(content="l1", ts=2.0)]
        session.archived_count = 5  # local side ahead of disk
        _write_disk(mgr, "s", turns=[_turn(content="disk", ts=3.0)], archived_count=0)
        mgr._adopt_from_disk(session)
        assert [t["content"] for t in session.turns] == ["l0", "l1", "disk"]
        assert session.archived_count == 5

    def test_adopt_takes_disk_compressed_pointer_and_summary(self, mgr):
        session = mgr.get_or_create("s")
        _write_disk(mgr, "s", compressed_up_to=7, compressed_summary="DISK SUM")
        mgr._adopt_from_disk(session)
        assert session.compressed_up_to == 7
        assert session.compressed_summary == "DISK SUM"

    def test_adopt_pointer_with_empty_summary_keeps_local_summary(self, mgr):
        session = mgr.get_or_create("s")
        session.compressed_summary = "LOCAL"
        _write_disk(mgr, "s", compressed_up_to=7, compressed_summary="")
        mgr._adopt_from_disk(session)
        assert session.compressed_up_to == 7
        assert session.compressed_summary == "LOCAL"


# ── add_turn: optional fields + guarded failure paths ────────────────────


class TestAddTurn:
    def test_persists_optional_fields(self, mgr):
        mgr.add_turn(
            "s",
            "assistant",
            "answer",
            model="m1",
            digest="DIG: read a.py",
            exclude_from_compression=True,
            tool_results=[{"name": "bash"}],
        )
        t = mgr.get_or_create("s").turns[-1]
        assert t["model"] == "m1"
        assert t["digest"] == "DIG: read a.py"
        assert t["exclude_from_compression"] is True
        assert t["tool_results"] == [{"name": "bash"}]

    def test_next_assistant_turn_clears_prior_tool_results(self, mgr):
        # NOTE: re-query the turn after each add_turn — _adopt_from_disk
        # replaces turn dicts with disk-parsed copies, so holding a stale
        # reference would observe the pre-clear dict (identity, not logic).
        mgr.add_turn("s", "assistant", "aborted", tool_results=[{"name": "bash"}])
        first = mgr.get_or_create("s").turns[0]
        assert "tool_results" in first
        mgr.add_turn("s", "assistant", "resumed answer")
        first = mgr.get_or_create("s").turns[0]
        assert "tool_results" not in first

    def test_adopt_failure_does_not_abort_append(self, mgr, monkeypatch):
        monkeypatch.setattr(mgr, "_adopt_from_disk", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
        mgr.add_turn("s", "user", "hello")
        assert mgr.get_or_create("s").turns[-1]["content"] == "hello"

    def test_reap_failure_does_not_abort_append(self, mgr, monkeypatch):
        monkeypatch.setattr(mgr, "_reap_zombie_in_progress", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
        mgr.add_turn("s", "user", "hello")
        assert mgr.get_or_create("s").turns[-1]["content"] == "hello"


class TestIsProcessAlive:
    def test_permission_error_assumed_alive(self, mgr, monkeypatch):
        def _denied(pid, sig):
            raise PermissionError(sig)

        monkeypatch.setattr("os.kill", _denied)
        assert mgr._is_process_alive(424242) is True
        assert 424242 not in mgr._dead_pids  # conservative, not cached dead


# ── delegation wiring to SessionCompressionContext ───────────────────────


class _FakeCtx:
    def __init__(self):
        self.calls = []

    def load_project_context_md(self):
        self.calls.append(("load_md",))
        return "PROJECT MD"

    def build_context_messages(self, session, model, skip_core_prompt, mode, owner):
        self.calls.append(("build", model, skip_core_prompt, mode, owner))
        return [{"role": "system", "content": "ctx"}]

    def schedule_background_compress(self, session, model, client, force, notify, persist):
        self.calls.append(("sched", model, force, notify))
        persist()  # prove the persist callback is wired

    def compact_now(self, session, model, client, recent_keep, notify, persist):
        self.calls.append(("compact", model, recent_keep))
        persist()
        return "OK"


class TestDelegations:
    def test_load_project_context_md_delegates(self, mgr, monkeypatch):
        fake = _FakeCtx()
        monkeypatch.setattr(mgr, "_ctx", fake)
        assert mgr._load_project_context_md() == "PROJECT MD"
        assert fake.calls == [("load_md",)]

    def test_schedule_background_compress_wires_persist_to_save(self, mgr, monkeypatch):
        fake = _FakeCtx()
        monkeypatch.setattr(mgr, "_ctx", fake)
        saved = []
        monkeypatch.setattr(
            mgr, "_save", lambda session: saved.append(session.id if hasattr(session, "id") else session.session_id)
        )
        session = DesignSession(session_id="s")
        notify = object()
        mgr.schedule_background_compress(session, "m", "client", force=True, notify=notify)
        assert fake.calls == [("sched", "m", True, notify)]
        assert saved == ["s"]

    def test_compact_now_delegates_and_returns(self, mgr, monkeypatch):
        fake = _FakeCtx()
        monkeypatch.setattr(mgr, "_ctx", fake)
        saved = []
        monkeypatch.setattr(mgr, "_save", lambda session: saved.append(session))
        session = DesignSession(session_id="s")
        assert mgr.compact_now(session, "m", "client", recent_keep=0) == "OK"
        assert fake.calls == [("compact", "m", 0)]
        assert saved == [session]


# ── list_sessions ────────────────────────────────────────────────────────


def _session_file(mgr, sid, updated_at=None, turns=(), summary="", archived=0):
    data = {
        "session_id": sid,
        "created_at": 1.0,
        "turns": list(turns),
        "compressed_summary": summary,
        "compressed_up_to": 0,
        "archived_count": archived,
        "decisions": [],
        "chat_mode": "code",
    }
    if updated_at is not None:
        data["updated_at"] = updated_at
    mgr._session_path(sid).write_text(json.dumps(data), encoding="utf-8")


class TestListSessions:
    def test_empty_dir_returns_empty(self, mgr):
        assert mgr.list_sessions() == []

    def test_orders_newest_first(self, mgr):
        _session_file(mgr, "older", updated_at=100.0)
        _session_file(mgr, "newer", updated_at=200.0)
        assert [s["session_id"] for s in mgr.list_sessions()] == ["newer", "older"]

    def test_skips_corrupt_files(self, mgr):
        _session_file(mgr, "good", updated_at=100.0)
        mgr._session_path("bad").write_text("{nope", encoding="utf-8")
        assert [s["session_id"] for s in mgr.list_sessions()] == ["good"]

    def test_legacy_file_without_updated_at_uses_mtime(self, mgr):
        _session_file(mgr, "legacy")  # no updated_at → file mtime (≈now)
        _session_file(mgr, "explicit-old", updated_at=100.0)
        assert [s["session_id"] for s in mgr.list_sessions()] == ["legacy", "explicit-old"]

    def test_turn_count_and_summary_flag(self, mgr):
        _session_file(mgr, "s", updated_at=1.0, turns=[_turn(), _turn(), _turn()], summary="SUM", archived=4)
        entry = mgr.list_sessions()[0]
        assert entry["turn_count"] == 7
        assert entry["has_summary"] is True

    def test_caps_at_20(self, mgr):
        for i in range(25):
            _session_file(mgr, f"s{i:02d}", updated_at=float(i))
        sessions = mgr.list_sessions()
        assert len(sessions) == 20
        assert sessions[0]["session_id"] == "s24"  # newest first

    def test_glob_failure_returns_empty(self, mgr, monkeypatch):
        def _raise(pattern):
            raise TypeError("bad dir")

        monkeypatch.setattr(mgr, "sessions_dir", SimpleNamespace(glob=_raise))
        assert mgr.list_sessions() == []

    def test_legacy_mtime_stat_failure_falls_back_to_zero(self, mgr, monkeypatch):
        # A legacy file (no updated_at) whose stat fails after the read —
        # e.g. deleted between glob and stat — must sort as 0, not crash.
        class _Vanishing:
            name = "legacy.json"
            stem = "legacy"

            def read_text(self, encoding="utf-8"):
                return json.dumps({"session_id": "legacy"})

            def stat(self):
                raise OSError("vanished")

        monkeypatch.setattr(mgr, "sessions_dir", SimpleNamespace(glob=lambda pat: [_Vanishing()]))
        sessions = mgr.list_sessions()
        assert sessions[0]["updated_at"] == 0
        assert sessions[0]["turn_count"] == 0


# ── startup sweep failure paths ─────────────────────────────────────────


class TestSweepOvergrownArchives:
    def test_glob_failure_returns_silently(self, mgr, monkeypatch):
        def _raise(pattern):
            raise OSError("dir gone")

        monkeypatch.setattr(mgr, "sessions_dir", SimpleNamespace(glob=_raise))
        mgr._sweep_overgrown_archives()  # must not raise

    def test_per_candidate_failure_warns_and_continues(self, mgr, monkeypatch, caplog):
        monkeypatch.setattr(mgr, "_ARCHIVE_MAX_BYTES", 100)
        _write_archive(mgr, "s1", [_rec(0), _rec(1)])  # over the tiny cap

        def _boom(sid):
            raise RuntimeError("compact failed")

        monkeypatch.setattr(mgr, "_compact_archive", _boom)
        with caplog.at_level("WARNING"):
            mgr._sweep_overgrown_archives()  # must not raise
        assert "s1" in caplog.text


# ── _save ────────────────────────────────────────────────────────────────


class TestSave:
    def test_merges_disk_turns_then_writes(self, mgr):
        session = mgr.get_or_create("s")
        session.turns = [_turn(content="local", ts=1.0)]
        _write_disk(mgr, "s", turns=[_turn(content="disk", ts=2.0)], updated_at=9.0)
        mgr._save(session)
        data = json.loads(mgr._session_path("s").read_text(encoding="utf-8"))
        # disk side wins as merge base when archive progress ties → disk turn first
        assert [t["content"] for t in data["turns"]] == ["disk", "local"]

    def test_write_failure_swallowed(self, mgr, monkeypatch):
        session = mgr.get_or_create("s")
        monkeypatch.setattr(mgr, "_write_session", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")))
        mgr._save(session)  # must not raise


# ── archive cut + failure paths ─────────────────────────────────────────


class TestArchiveCut:
    def test_moves_all_cut_turns_including_excluded(self, mgr):
        # exclude_from_compression turns are ephemeral by contract — they
        # archive like any other turn below the cut (nothing re-inserts them).
        session = DesignSession(session_id="s")
        session.turns = [
            _turn(content="q0", ts=1.0, exclude_from_compression=True),
            _turn(role="assistant", content="a1", ts=2.0),
            _turn(content="q2", ts=3.0),
        ]
        session.compressed_up_to = 2
        mgr._write_session(session)
        lines = mgr._archive_path("s").read_text(encoding="utf-8").splitlines()
        assert [json.loads(ln)["i"] for ln in lines] == [0, 1]
        data = json.loads(mgr._session_path("s").read_text(encoding="utf-8"))
        assert [t["content"] for t in data["turns"]] == ["q2"]
        assert data["archived_count"] == 2

    def test_archive_append_failure_keeps_turns_active(self, mgr, monkeypatch):
        monkeypatch.setattr(mgr, "_archive_last_index", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("io")))
        session = DesignSession(session_id="s")
        session.turns = [_turn(content=f"t{i}", ts=float(i)) for i in range(3)]
        session.compressed_up_to = 2
        mgr._write_session(session)
        assert not mgr._archive_path("s").exists()
        data = json.loads(mgr._session_path("s").read_text(encoding="utf-8"))
        assert len(data["turns"]) == 3  # nothing trimmed on failure
        assert data["archived_count"] == 0


def _write_archive(mgr, sid, records):
    path = mgr._archive_path(sid)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    return path


def _rec(i, size=80):
    return {"i": i, "role": "user", "timestamp": 1000.0 + i, "content": "x" * size}


class TestCompactArchiveFailures:
    def test_stat_failure_returns_false(self, mgr, monkeypatch):
        monkeypatch.setattr(mgr, "_archive_path", lambda sid: mgr.repo_root / "missing" / "a.jsonl")
        assert mgr._compact_archive("s") is False

    @_requires_file_perms
    def test_read_failure_returns_false(self, mgr, monkeypatch):
        monkeypatch.setattr(mgr, "_ARCHIVE_MAX_BYTES", 100)
        path = _write_archive(mgr, "s", [_rec(0), _rec(1)])
        path.chmod(0o000)
        try:
            assert mgr._compact_archive("s") is False
        finally:
            path.chmod(0o644)

    def test_valid_total_under_cap_untouched(self, mgr, monkeypatch):
        # On-disk size exceeds the cap only because of a corrupt giant line —
        # the parsed-record total is under the cap, so compaction stands down.
        monkeypatch.setattr(mgr, "_ARCHIVE_MAX_BYTES", 1000)
        path = mgr._archive_path("s")
        path.write_text(
            "".join(json.dumps(_rec(i, 80)) + "\n" for i in range(3)) + "x" * 1200 + "\n",
            encoding="utf-8",
        )
        before = path.read_text(encoding="utf-8")
        assert mgr._compact_archive("s") is False
        assert path.read_text(encoding="utf-8") == before

    def test_skips_corrupt_records_during_fold(self, mgr, monkeypatch):
        monkeypatch.setattr(mgr, "_ARCHIVE_MAX_BYTES", 2000)
        records = [_rec(i, 60) for i in range(31)]  # ~2.1 KB total
        path = mgr._archive_path("s")
        path.write_text("".join(json.dumps(r) + "\n" for r in records) + "{corrupt\n", encoding="utf-8")
        assert mgr._compact_archive("s") is True
        out = path.read_text(encoding="utf-8")
        assert "{corrupt" not in out
        first = json.loads(out.splitlines()[0])
        assert first["compacted"] is True and first["count"] > 1

    def test_folds_oversized_single_record(self, mgr, monkeypatch):
        monkeypatch.setattr(mgr, "_ARCHIVE_MAX_BYTES", 3000)
        records = [_rec(0, 70_000)] + [_rec(i, 60) for i in range(1, 6)]
        path = _write_archive(mgr, "s", records)
        assert mgr._compact_archive("s") is True
        lines = path.read_text(encoding="utf-8").splitlines()
        first = json.loads(lines[0])
        assert first["compacted"] is True
        assert first["count"] == 1  # the >64 KiB record folded alone
        assert len(lines) == 6  # one summary + five verbatim tail records

    def test_flushes_trailing_block_at_end(self, mgr, monkeypatch):
        # Budget straddles the last record: the loop ends with a pending block
        # that must still be flushed as one compacted summary.
        monkeypatch.setattr(mgr, "_ARCHIVE_MAX_BYTES", 1000)
        records = [_rec(i, 90) for i in range(5)] + [_rec(5, 590)]
        path = _write_archive(mgr, "s", records)
        assert path.stat().st_size > 1000
        assert mgr._compact_archive("s") is True
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        summary = json.loads(lines[0])
        assert summary["count"] == 6

    def test_write_failure_returns_false(self, mgr, monkeypatch):
        monkeypatch.setattr(mgr, "_ARCHIVE_MAX_BYTES", 2000)
        _write_archive(mgr, "s", [_rec(i, 60) for i in range(31)])

        def _replace(src, dst):
            raise OSError("no space")

        monkeypatch.setattr("os.replace", _replace)
        assert mgr._compact_archive("s") is False


class TestArchiveLastIndex:
    def test_all_corrupt_small_file_returns_minus_one(self, mgr):
        path = mgr._archive_path("s")
        path.write_text("garbage\nmore garbage\n", encoding="utf-8")
        assert mgr._archive_last_index(path) == -1

    def test_unreadable_path_returns_minus_one(self, mgr):
        # Path exists but is a directory — open() raises inside the reader.
        mgr._archive_path("s").mkdir()
        assert mgr._archive_last_index(mgr._archive_path("s")) == -1


class TestLoadArchivedTurns:
    def test_missing_archive_returns_empty(self, mgr):
        assert mgr.load_archived_turns("s") == []

    def test_parses_valid_skips_corrupt_and_blank(self, mgr):
        path = mgr._archive_path("s")
        good = json.dumps(_rec(0)) + "\n\n" + "{corrupt\n" + json.dumps(_rec(1)) + "\n"
        path.write_text(good, encoding="utf-8")
        turns = mgr.load_archived_turns("s")
        assert [t["i"] for t in turns] == [0, 1]

    def test_read_failure_returns_empty(self, mgr):
        mgr._archive_path("s").mkdir()  # exists but unreadable as a file
        assert mgr.load_archived_turns("s") == []


class TestArchivePathAccessor:
    def test_public_accessor_matches_internal(self, mgr):
        assert mgr.archive_path("s") == mgr._archive_path("s")


class TestDeleteSession:
    def test_removes_archive_and_session(self, mgr):
        _write_disk(mgr, "s", turns=[_turn()])
        _write_archive(mgr, "s", [_rec(0)])
        assert mgr.delete_session("s") is True
        assert not mgr._session_path("s").exists()
        assert not mgr._archive_path("s").exists()

    def test_archive_unlink_failure_still_deletes_session(self, mgr):
        _write_disk(mgr, "s", turns=[_turn()])
        mgr._archive_path("s").mkdir()  # unlink on a dir fails
        assert mgr.delete_session("s") is True

    def test_session_unlink_failure_returns_false(self, mgr):
        mgr._session_path("s").mkdir()
        assert mgr.delete_session("s") is False

    def test_nothing_on_disk_returns_false(self, mgr):
        assert mgr.delete_session("ghost") is False
