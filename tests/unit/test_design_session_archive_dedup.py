"""Crash-recovery dedup of archived turns: `_archive_last_index` must survive

1. a last record LONGER than the 64 KiB tail window (giant tool_results after
   an ESC-interrupt) — previously the truncated fragment failed to parse and
   the function returned -1, silently disabling the `abs_i <= last_i` guard
   so the next recovery re-appended already-archived turns; and
2. a trailing partial line left by a crash mid-append — previously the first
   parse failure bailed out instead of walking back to the previous complete
   record.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from external_llm.design_session import DesignSessionManager


@pytest.fixture
def mgr(tmp_path):

    return DesignSessionManager(repo_root=str(tmp_path))


def _archive(mgr, sid):

    return mgr._archive_path(sid)


def _write(mgr, sid, records):

    path = _archive(mgr, sid)

    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")

    return path


def _session(turns, compressed_up_to, archived_count=0, sid="s1"):

    return SimpleNamespace(
        session_id=sid,
        turns=turns,
        compressed_up_to=compressed_up_to,
        archived_count=archived_count,
    )


class TestArchiveLastIndex:
    def test_missing_file_returns_minus_one(self, mgr):

        assert mgr._archive_last_index(_archive(mgr, "nope")) == -1

    def test_empty_file_returns_minus_one(self, mgr):

        p = _archive(mgr, "s1")

        p.write_text("", encoding="utf-8")

        assert mgr._archive_last_index(p) == -1

    def test_returns_last_index(self, mgr):

        p = _write(mgr, "s1", [{"i": 0, "role": "user"}, {"i": 1, "role": "assistant"}, {"i": 2, "role": "user"}])

        assert mgr._archive_last_index(p) == 2

    def test_last_record_longer_than_tail_window(self, mgr):

        # Regression: a record > 64 KiB (giant tool_results after ESC-abort)

        # used to leave the tail window mid-line -> parse failure -> -1.

        big = {"i": 7, "role": "assistant", "content": "x" * 200_000}

        p = _write(mgr, "s1", [{"i": 0, "role": "user"}, big])

        assert mgr._archive_last_index(p) == 7

    def test_trailing_partial_line_walks_back(self, mgr):

        # Crash mid-append leaves a truncated final line without newline.

        p = _archive(mgr, "s1")

        p.write_text(
            json.dumps({"i": 3, "role": "user"}, ensure_ascii=False)
            + "\n"
            + '{"i": 4, "role": "assistant", "content": "unterminated',
            encoding="utf-8",
        )

        assert mgr._archive_last_index(p) == 3

    def test_corrupt_interior_line_skipped(self, mgr):

        p = _archive(mgr, "s1")

        p.write_text(
            json.dumps({"i": 0, "role": "user"}, ensure_ascii=False)
            + "\n{corrupt\n"
            + json.dumps({"i": 2, "role": "user"}, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )

        assert mgr._archive_last_index(p) == 2

    def test_whole_file_is_one_line(self, mgr):

        p = _archive(mgr, "s1")

        p.write_text(json.dumps({"i": 5, "content": "y" * 100_000}, ensure_ascii=False), encoding="utf-8")

        assert mgr._archive_last_index(p) == 5


class TestArchiveCompressedTurnsDedup:
    def test_no_duplicate_after_crash_recovery(self, mgr):

        # Archive already holds i=0,1 (appended before the crash that lost the

        # active-file save). Recovery re-runs archiving with stale counters;

        # the abs_i <= last_i guard must skip both records.

        p = _write(mgr, "s1", [{"i": 0, "role": "user"}, {"i": 1, "role": "assistant"}])

        s = _session([{"role": "user"}, {"role": "assistant"}], compressed_up_to=2)

        mgr._archive_compressed_turns(s)

        assert len(p.read_text(encoding="utf-8").splitlines()) == 2

        assert s.archived_count == 2

        assert s.turns == []

    def test_dedup_survives_long_last_record(self, mgr):

        # The exact P4 hazard: the archive's last record exceeds the tail

        # window, so last_i used to degrade to -1 and the guard re-appended

        # the same turns.

        big = {"i": 0, "role": "assistant", "tool_results": [{"content": "z" * 200_000}]}

        p = _write(mgr, "s1", [big])

        s = _session([{"role": "assistant"}], compressed_up_to=1)

        mgr._archive_compressed_turns(s)

        assert len(p.read_text(encoding="utf-8").splitlines()) == 1

        assert s.archived_count == 1

    def test_legacy_preserve_turns_migrate_and_archive(self, mgr):
        """The stop-at-preserve scan was removed as unreachable: nothing

        produces preserve=True any more (add_turn never sets it and _load_raw

        migrates legacy disk files to exclude_from_compression at the single

        read choke point).  Every turn below the cut archives — including

        migrated exclude turns, which are ephemeral by contract."""

        p = _write(mgr, "s1", [])

        s = _session(
            [{"role": "user"}, {"role": "assistant", "preserve": True}, {"role": "user"}],
            compressed_up_to=3,
        )

        # Simulate what _load_raw does to a legacy disk file:

        s.turns[1].pop("preserve")

        s.turns[1]["exclude_from_compression"] = True

        mgr._archive_compressed_turns(s)

        assert len(p.read_text(encoding="utf-8").splitlines()) == 3

        assert s.archived_count == 3

        assert len(s.turns) == 0
