"""Verify multi-terminal (multi-process) shared session behavior.

When two asi processes share the same session file:
1. One side's add_turn does not overwrite the other side's turns
   (flock + reload-merge).
2. Another process's in-progress user turn is rendered in the context as a
   [IN-PROGRESS IN ANOTHER TERMINAL] system label.
3. Once that process records an assistant turn, the flag clears, and the
   other process absorbs this on its next sync.

The two DesignSessionManager instances each have independent in-memory
caches, so this follows the same path as two real processes (sync via disk).
"""
import pytest

from external_llm.design_session import DesignSessionManager

SID = "shared-session"
_LABEL = "[IN-PROGRESS IN ANOTHER TERMINAL]"


def _mgr(tmp_path, owner: str) -> DesignSessionManager:
    m = DesignSessionManager(str(tmp_path))
    m._owner = owner
    return m


@pytest.fixture
def mgr_a(tmp_path):
    return _mgr(tmp_path, "pid:A")


@pytest.fixture
def mgr_b(tmp_path):
    return _mgr(tmp_path, "pid:B")


def _turn_contents(mgr, sid=SID):
    return [t["content"] for t in mgr.get_or_create(sid).turns]


def _seed_session_file(tmp_path, sid: str, turns: list[dict]) -> None:
    """Write session JSON directly to disk to simulate state left behind by a dead process.

    Zombie-recovery tests must start from a state where an in_progress turn
    already exists on disk — if we tamper with the in-memory session and then
    call _save, _adopt_from_disk would revert to the on-disk original and undo
    the tampering. Writing directly to disk is the real scenario.
    """
    import json
    sessions_dir = tmp_path / ".asicode" / "design_sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c for c in sid if c.isalnum() or c in "-_")
    path = sessions_dir / f"{safe}.json"
    data = {
        "session_id": sid, "created_at": __import__("time").time(),
        "updated_at": __import__("time").time(), "turns": turns,
        "compressed_summary": "", "compressed_up_to": 0,
        "archived_count": 0, "decisions": [], "chat_mode": "code",
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


class TestInterleavedAppends:
    def test_no_turn_loss_across_processes(self, mgr_a, mgr_b):
        mgr_a.add_turn(SID, "user", "request A", in_progress=True)
        mgr_b.add_turn(SID, "user", "request B", in_progress=True)
        # B's cache has A's turn merged in
        assert _turn_contents(mgr_b) == ["request A", "request B"]
        # A recording a response doesn't make B's turn disappear
        mgr_a.add_turn(SID, "assistant", "answer A")
        assert _turn_contents(mgr_a) == ["request A", "request B", "answer A"]

    def test_stale_cache_refreshed_via_get_or_create(self, mgr_a, mgr_b):
        mgr_b.get_or_create(SID)  # B caches an empty session
        mgr_a.add_turn(SID, "user", "request A", in_progress=True)
        # Cache hit path detects mtime change and absorbs A's turn
        assert _turn_contents(mgr_b) == ["request A"]


class TestInProgressLabeling:
    def test_other_process_dangling_turn_is_labeled(self, mgr_a, mgr_b):
        mgr_a.add_turn(SID, "user", "request A", in_progress=True)
        mgr_b.add_turn(SID, "user", "request B", in_progress=True)
        msgs = mgr_b.build_context_messages(mgr_b.get_or_create(SID))

        labeled = [m for m in msgs if m["role"] == "system" and _LABEL in m["content"]
                   and m["content"].startswith("(turn ")]
        assert len(labeled) == 1
        assert "request A" in labeled[0]["content"]
        # A's request does not appear as a regular user message
        assert not any(m["role"] == "user" and "request A" in m["content"] for m in msgs)
        # B's own requests remain as regular user messages
        assert any(m["role"] == "user" and "request B" in m["content"] for m in msgs)

    def test_own_in_progress_turn_not_labeled(self, mgr_a):
        mgr_a.add_turn(SID, "user", "request A", in_progress=True)
        msgs = mgr_a.build_context_messages(mgr_a.get_or_create(SID))
        assert not any(m["content"].startswith("(turn ") and _LABEL in m["content"]
                       for m in msgs)
        assert any(m["role"] == "user" and "request A" in m["content"] for m in msgs)

    def test_label_cleared_after_assistant_recorded(self, mgr_a, mgr_b):
        mgr_a.add_turn(SID, "user", "request A", in_progress=True)
        mgr_b.add_turn(SID, "user", "request B", in_progress=True)
        mgr_a.add_turn(SID, "assistant", "answer A")
        # A's flag clearing is reflected to B via disk
        mgr_b.add_turn(SID, "assistant", "answer B")
        msgs = mgr_b.build_context_messages(mgr_b.get_or_create(SID))
        assert not any(m["content"].startswith("(turn ") and _LABEL in m["content"]
                       for m in msgs)
        assert any(m["role"] == "user" and "request A" in m["content"] for m in msgs)

    def test_assistant_clears_only_own_turn(self, mgr_a, mgr_b):
        mgr_a.add_turn(SID, "user", "request A", in_progress=True)
        mgr_b.add_turn(SID, "user", "request B", in_progress=True)
        # B's response only clears B's own turn — A's turn is still in progress
        mgr_b.add_turn(SID, "assistant", "answer B")
        turns = mgr_b.get_or_create(SID).turns
        flags = {t["content"]: t.get("in_progress", False) for t in turns}
        assert flags["request A"] is True
        assert flags["request B"] is False


class TestCurrentRequestAnchor:
    def test_anchor_warns_about_parallel_sessions(self, mgr_a):
        mgr_a.add_turn(SID, "user", "hello")
        msgs = mgr_a.build_context_messages(mgr_a.get_or_create(SID))
        anchor = [m for m in msgs if "[CURRENT REQUEST]" in m["content"]]
        assert len(anchor) == 1
        assert "parallel session" in anchor[0]["content"]


class TestZombieInProgressReap:
    """Self-recovery of zombie in_progress turns left by an abnormally terminated terminal.

    Scenario: process A records a user turn with in_progress=True, then dies
    to SIGKILL/OOM/terminal force-close before it can record the assistant
    turn. When A restarts, it gets a new PID, so
    _clear_in_progress(owner==self._owner) can't clear its own old turn.
    Another terminal B keeps rendering A's zombie turn as [IN-PROGRESS] forever.

    Recovery: right before add_turn/_save, _reap_zombie_in_progress clears
    the flag if the owner PID is dead (or age >= 1h).
    """

    def test_dead_owner_pid_is_reaped_on_next_add_turn(self, mgr_b, tmp_path):
        """A zombie turn left on disk by A (owner=dead PID) is cleared on B's add_turn."""
        import time
        _seed_session_file(tmp_path, SID, [
            {"role": "user", "content": "zombie request A", "timestamp": time.time(),
             "in_progress": True, "owner": "pid:999999"},  # almost certainly not alive
        ])
        mgr_b.add_turn(SID, "user", "request B")
        turns = mgr_b.get_or_create(SID).turns
        assert not any(t.get("in_progress") for t in turns), \
            "zombie turn of a dead owner was not reaped"

    def test_alive_owner_pid_is_not_reaped(self, mgr_b, tmp_path):
        """A turn owned by a live PID is not treated as a zombie."""
        import os
        import time
        _seed_session_file(tmp_path, SID, [
            {"role": "user", "content": "alive request A", "timestamp": time.time(),
             "in_progress": True, "owner": f"pid:{os.getpid()}"},  # current process = alive
        ])
        mgr_b.add_turn(SID, "user", "request B")
        turns = mgr_b.get_or_create(SID).turns
        assert any(t.get("in_progress") for t in turns), \
            "a live owner's turn was incorrectly reaped"

    def test_self_pid_excluded_from_reap(self, mgr_a):
        """A turn owned by our own PID may still be in progress, so it's excluded from reap."""
        import os
        mgr_a._owner = f"pid:{os.getpid()}"
        mgr_a.add_turn(SID, "user", "my in-progress request", in_progress=True)
        # Our own add_turn is not treated as a zombie (nor on the next add_turn)
        mgr_a.add_turn(SID, "assistant", "my answer")
        turns = mgr_a.get_or_create(SID).turns
        # Should have been cleared normally by _clear_in_progress when the assistant turn was recorded
        assert not any(t.get("in_progress") for t in turns)

    def test_legacy_arbitrary_owner_uses_age_fallback(self, mgr_b, tmp_path):
        """If owner isn't in pid: form, only age >= MAX_AGE triggers reap (arbitrary test owner)."""
        import time
        old_ts = time.time() - mgr_b._IN_PROGRESS_MAX_AGE - 1
        _seed_session_file(tmp_path, SID, [
            {"role": "user", "content": "legacy zombie A", "timestamp": old_ts,
             "in_progress": True, "owner": "pid:A"},  # non-standard owner → age fallback
        ])
        mgr_b.add_turn(SID, "user", "request B")
        turns = mgr_b.get_or_create(SID).turns
        assert not any(t.get("in_progress") for t in turns), \
            "age-based fallback failed to reap an old legacy zombie"

    def test_recent_legacy_owner_not_reaped(self, mgr_b, tmp_path):
        """If age < MAX_AGE, don't reap even with a non-standard owner (conservative)."""
        import time
        _seed_session_file(tmp_path, SID, [
            {"role": "user", "content": "recent legacy A", "timestamp": time.time(),
             "in_progress": True, "owner": "pid:A"},
        ])
        mgr_b.add_turn(SID, "user", "request B")
        turns = mgr_b.get_or_create(SID).turns
        assert any(t.get("in_progress") for t in turns), \
            "a recent turn was incorrectly reaped by the age fallback"

    def test_reaped_turns_persisted_to_disk(self, tmp_path):
        """A zombie reap is persisted to disk, so it's visible even after another process restarts."""
        import os
        import time
        _seed_session_file(tmp_path, SID, [
            {"role": "user", "content": "zombie A", "timestamp": time.time(),
             "in_progress": True, "owner": "pid:999998"},
        ])
        # B triggers add_turn → the zombie reap is persisted to disk
        b = DesignSessionManager(str(tmp_path))
        b._owner = f"pid:{os.getpid()}"
        b.add_turn(SID, "assistant", "answer B")
        # A brand-new manager (simulating a restart) reads from disk and sees the reap
        fresh = DesignSessionManager(str(tmp_path))
        fresh._owner = f"pid:{os.getpid()}"
        session = fresh.get_or_create(SID)
        assert not any(t.get("in_progress") for t in session.turns), \
            "the zombie reap was not persisted to disk"

    def test_dead_pid_cached_permanently(self, mgr_b, tmp_path):
        """Once a PID is determined dead, it's cached permanently (defense against PID reuse)."""
        import time
        _seed_session_file(tmp_path, SID, [
            {"role": "user", "content": "zombie A", "timestamp": time.time(),
             "in_progress": True, "owner": "pid:999997"},
        ])
        mgr_b.add_turn(SID, "user", "trigger reap")
        assert 999997 in mgr_b._dead_pids
        # A second zombie turn from the same PID is also reaped immediately via a cache hit
        _seed_session_file(tmp_path, SID, [
            {"role": "user", "content": "zombie A", "timestamp": time.time(),
             "in_progress": True, "owner": "pid:999997"},
            {"role": "assistant", "content": "x", "timestamp": time.time()},
            {"role": "user", "content": "zombie A2", "timestamp": time.time(),
             "in_progress": True, "owner": "pid:999997"},
        ])
        mgr_b.add_turn(SID, "user", "trigger reap 2")
        turns = mgr_b.get_or_create(SID).turns
        assert not any(t.get("in_progress") for t in turns)
