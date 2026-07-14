"""멀티 터미널(멀티 프로세스) 공유 세션 동작 검증.

두 asi 프로세스가 같은 세션 파일을 공유할 때:
1. 한쪽의 add_turn이 다른 쪽 턴을 덮어쓰지 않는다 (flock + reload-merge).
2. 처리 중(in_progress)인 다른 프로세스의 user 턴은 컨텍스트에서
   [IN-PROGRESS IN ANOTHER TERMINAL] system 라벨로 렌더된다.
3. 해당 프로세스가 assistant 턴을 기록하면 플래그가 해제되고,
   다른 프로세스도 다음 동기화에서 이를 흡수한다.

두 DesignSessionManager 인스턴스는 각각 독립 인메모리 캐시를 가지므로
실제 두 프로세스와 동일한 경로(디스크 경유 동기화)를 탄다.
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
    """디스크에 세션 JSON을 직접 기록하여 죽은 프로세스가 남긴 상태를 시뮬레이션.

    좀비 복구 테스트는 "이미 디스크에 in_progress 턴이 존재"하는 상태에서
    시작해야 한다 — 인메모리 변조 후 _save를 호출하면 _adopt_from_disk가 디스크
    원본으로 되돌리므로 변조가 사라진다. 디스크에 직접 쓰는 것이 진짜 시나리오.
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
        # A의 플래그 해제가 디스크를 거쳐 B에도 반영된다
        mgr_b.add_turn(SID, "assistant", "answer B")
        msgs = mgr_b.build_context_messages(mgr_b.get_or_create(SID))
        assert not any(m["content"].startswith("(turn ") and _LABEL in m["content"]
                       for m in msgs)
        assert any(m["role"] == "user" and "request A" in m["content"] for m in msgs)

    def test_assistant_clears_only_own_turn(self, mgr_a, mgr_b):
        mgr_a.add_turn(SID, "user", "request A", in_progress=True)
        mgr_b.add_turn(SID, "user", "request B", in_progress=True)
        # B의 응답은 B의 턴만 해제한다 — A의 턴은 여전히 처리 중
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
    """비정상 종료된 터미널이 남긴 좀비 in_progress 턴의 자가 회복.

    시나리오: 프로세스 A가 user 턴을 in_progress=True로 기록한 뒤 SIGKILL/OOM/
    터미널 강제종료로 assistant 턴을 기록하지 못하고 죽는다. A 재시작 시 새 PID가
    되어 _clear_in_progress(owner==self._owner)로는 자기 턴인데도 해제 불가.
    다른 터미널 B는 A의 좀비 턴을 영구적으로 [IN-PROGRESS]로 렌더한다.

    복구: add_turn/_save 직전 _reap_zombie_in_progress가 owner PID가 죽었으면
    (또는 age >= 1h 이면) 플래그를 해제한다.
    """

    def test_dead_owner_pid_is_reaped_on_next_add_turn(self, mgr_b, tmp_path):
        """A(owner=죽은 PID)가 디스크에 남긴 좀비 턴이 B의 add_turn에서 해제된다."""
        import time
        _seed_session_file(tmp_path, SID, [
            {"role": "user", "content": "zombie request A", "timestamp": time.time(),
             "in_progress": True, "owner": "pid:999999"},  # 살아있지 않을 것이 거의 확실
        ])
        mgr_b.add_turn(SID, "user", "request B")
        turns = mgr_b.get_or_create(SID).turns
        assert not any(t.get("in_progress") for t in turns), \
            "dead owner의 좀비 턴이 해제되지 않음"

    def test_alive_owner_pid_is_not_reaped(self, mgr_b, tmp_path):
        """살아있는 owner PID의 턴은 좀비로 취급하지 않는다."""
        import os
        import time
        _seed_session_file(tmp_path, SID, [
            {"role": "user", "content": "alive request A", "timestamp": time.time(),
             "in_progress": True, "owner": f"pid:{os.getpid()}"},  # 현재 프로세스 = 살아있음
        ])
        mgr_b.add_turn(SID, "user", "request B")
        turns = mgr_b.get_or_create(SID).turns
        assert any(t.get("in_progress") for t in turns), \
            "살아있는 owner의 턴이 잘못 해제됨"

    def test_self_pid_excluded_from_reap(self, mgr_a):
        """자기 PID 소유 턴은 현재 처리 중일 수 있으므로 reap에서 제외."""
        import os
        mgr_a._owner = f"pid:{os.getpid()}"
        mgr_a.add_turn(SID, "user", "my in-progress request", in_progress=True)
        # 자기 add_turn은 좀비로 취급하지 않는다 (다음 add_turn에서도)
        mgr_a.add_turn(SID, "assistant", "my answer")
        turns = mgr_a.get_or_create(SID).turns
        # assistant 턴 기록 시 _clear_in_progress로 정상 해제되었어야 함
        assert not any(t.get("in_progress") for t in turns)

    def test_legacy_arbitrary_owner_uses_age_fallback(self, mgr_b, tmp_path):
        """owner가 pid: 형태가 아니면 age >= MAX_AGE로만 해제 (테스트용 임의 owner)."""
        import time
        old_ts = time.time() - mgr_b._IN_PROGRESS_MAX_AGE - 1
        _seed_session_file(tmp_path, SID, [
            {"role": "user", "content": "legacy zombie A", "timestamp": old_ts,
             "in_progress": True, "owner": "pid:A"},  # 비표준 owner → age fallback
        ])
        mgr_b.add_turn(SID, "user", "request B")
        turns = mgr_b.get_or_create(SID).turns
        assert not any(t.get("in_progress") for t in turns), \
            "age 기반 fallback이 오래된 legacy 좀비를 해제하지 못함"

    def test_recent_legacy_owner_not_reaped(self, mgr_b, tmp_path):
        """age < MAX_AGE이면 비표준 owner여도 해제하지 않는다 (conservative)."""
        import time
        _seed_session_file(tmp_path, SID, [
            {"role": "user", "content": "recent legacy A", "timestamp": time.time(),
             "in_progress": True, "owner": "pid:A"},
        ])
        mgr_b.add_turn(SID, "user", "request B")
        turns = mgr_b.get_or_create(SID).turns
        assert any(t.get("in_progress") for t in turns), \
            "최근 턴이 age fallback으로 잘못 해제됨"

    def test_reaped_turns_persisted_to_disk(self, tmp_path):
        """좀비 해제가 디스크에 반영되어 다른 프로세스 재시작 시에도 보인다."""
        import os
        import time
        _seed_session_file(tmp_path, SID, [
            {"role": "user", "content": "zombie A", "timestamp": time.time(),
             "in_progress": True, "owner": "pid:999998"},
        ])
        # B가 add_turn 트리거 → 좀비 해제가 디스크에 반영됨
        b = DesignSessionManager(str(tmp_path))
        b._owner = f"pid:{os.getpid()}"
        b.add_turn(SID, "assistant", "answer B")
        # 완전히 새 매니저(재시작 시뮬레이션)가 디스크에서 읽었을 때 좀비 해제됨
        fresh = DesignSessionManager(str(tmp_path))
        fresh._owner = f"pid:{os.getpid()}"
        session = fresh.get_or_create(SID)
        assert not any(t.get("in_progress") for t in session.turns), \
            "좀비 해제가 디스크에 영속되지 않음"

    def test_dead_pid_cached_permanently(self, mgr_b, tmp_path):
        """한 번 죽었다고 판정된 PID는 영구 캐시된다 (PID 재사용 방어)."""
        import time
        _seed_session_file(tmp_path, SID, [
            {"role": "user", "content": "zombie A", "timestamp": time.time(),
             "in_progress": True, "owner": "pid:999997"},
        ])
        mgr_b.add_turn(SID, "user", "trigger reap")
        assert 999997 in mgr_b._dead_pids
        # 같은 PID의 두 번째 좀비 턴도 캐시 히트로 즉시 해제
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
