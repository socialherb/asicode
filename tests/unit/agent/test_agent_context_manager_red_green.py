"""RED→GREEN: agent_context_manager — git 스냅샷/세션 컨텍스트/메시지 빌더.

get_git_snapshot의 병렬 수집·예외 경로와 ContextManagerMixin의
trim/compress/tier/git/session/continuation 브랜치를 고정한다.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

import external_llm.agent.agent_context_manager as acm
from external_llm.agent.agent_context_manager import (
    ContextManagerMixin,
    ContextTier,
    get_git_snapshot,
)


@pytest.fixture(autouse=True)
def _clean_git_cache():
    acm._git_cache.clear()
    acm._git_dirty_since.clear()
    yield
    acm._git_cache.clear()
    acm._git_dirty_since.clear()


class _FakeFuture:
    def __init__(self, fn, args):
        self._fn, self._args = fn, args

    def result(self, timeout=0):
        return self._fn(*self._args)


class _BoomFuture:
    def result(self, timeout=0):
        raise RuntimeError("git subprocess crashed")


class _FakePool:
    """submit()가 즉시 실행하는 인메모리 풀 (스레드 없이 결정적)."""

    def __init__(self, boom=False, concurrent_seed=None):
        self._boom = boom
        self._concurrent_seed = concurrent_seed
        self.submitted = []

    def submit(self, fn, *args):
        self.submitted.append((fn, args))
        if self._concurrent_seed is not None:
            self._concurrent_seed()
        if self._boom:
            raise RuntimeError("pool exhausted")
        return _FakeFuture(fn, args)


def _fake_git_raw(repo_root, *args):
    """_run_git_raw 대체: 인자 첫 요소로 분기하는 결정적 페이크."""
    key = args[0] if args else ""
    if key == "rev-parse":
        return "main"
    if key == "status":
        return " M src/x.py"
    if key == "log":
        return "abc123def\tfix: thing"
    return ""


# ── get_git_snapshot ────────────────────────────────────────────────────


class TestGetGitSnapshot:
    def test_empty_repo_root_returns_empty(self):
        assert get_git_snapshot("") == {}
        assert get_git_snapshot(None) == {}

    def test_happy_path_parallel_collection(self, monkeypatch):
        pool = _FakePool()
        monkeypatch.setattr("external_llm.agent._thread_pool.shared_pool", pool)
        monkeypatch.setattr(acm, "_run_git_raw", _fake_git_raw)
        snap = get_git_snapshot("/repo/a")
        assert snap["branch"] == "main"
        assert snap["status"] == " M src/x.py"
        assert snap["head_hash"] == "abc123def"
        assert snap["last_commit"] == "fix: thing"
        assert len(pool.submitted) == 3

    def test_second_call_serves_ttl_cache(self, monkeypatch):
        pool = _FakePool()
        monkeypatch.setattr("external_llm.agent._thread_pool.shared_pool", pool)
        monkeypatch.setattr(acm, "_run_git_raw", _fake_git_raw)
        get_git_snapshot("/repo/a")
        get_git_snapshot("/repo/a")
        assert len(pool.submitted) == 3  # 두 번째는 캐시 히트 — 재수집 없음

    def test_log_without_tab_splits_to_empty_last_commit(self, monkeypatch):
        pool = _FakePool()
        monkeypatch.setattr("external_llm.agent._thread_pool.shared_pool", pool)
        monkeypatch.setattr(acm, "_run_git_raw", lambda root, *args: "deadbeef" if args[0] == "log" else "")
        snap = get_git_snapshot("/repo/b")
        assert snap["head_hash"] == "deadbeef"
        assert snap["last_commit"] == ""

    def test_future_failure_fills_empty_strings(self, monkeypatch):
        class _Pool:
            def submit(self, fn, *args):
                return _BoomFuture()

        monkeypatch.setattr("external_llm.agent._thread_pool.shared_pool", _Pool())
        snap = get_git_snapshot("/repo/c")
        assert snap == {"branch": "", "status": "", "head_hash": "", "last_commit": ""}

    def test_pool_submit_failure_sets_defaults(self, monkeypatch):
        monkeypatch.setattr("external_llm.agent._thread_pool.shared_pool", _FakePool(boom=True))
        snap = get_git_snapshot("/repo/d")
        assert snap == {"branch": "", "status": "", "head_hash": "", "last_commit": ""}

    def test_concurrent_populate_recheck_hit(self, monkeypatch):
        def seed():
            key = acm.canonical_repo_key("/repo/e")
            acm._git_cache[key] = (
                time.monotonic(),
                {"branch": "concurrent", "status": "", "head_hash": "h", "last_commit": ""},
            )

        pool = _FakePool(concurrent_seed=seed)
        monkeypatch.setattr("external_llm.agent._thread_pool.shared_pool", pool)
        monkeypatch.setattr(acm, "_run_git_raw", _fake_git_raw)
        snap = get_git_snapshot("/repo/e")
        assert snap["branch"] == "concurrent"  # 재검사 히트 — 수집본이 아닌 선점 엔트리

    def test_status_truncated_at_ssot_bound(self, monkeypatch):
        pool = _FakePool()
        monkeypatch.setattr("external_llm.agent._thread_pool.shared_pool", pool)
        long_status = "M" * (acm.GIT_STATUS_MAX_CHARS + 100)
        monkeypatch.setattr(acm, "_run_git_raw", lambda root, *args: long_status if args[0] == "status" else "")
        snap = get_git_snapshot("/repo/f")
        assert len(snap["status"]) == acm.GIT_STATUS_MAX_CHARS


# ── ContextManagerMixin ─────────────────────────────────────────────────


class _CtxHost(ContextManagerMixin):
    def __init__(self, *, config=None, registry=None, model="m", context_tier=None):
        self.config = config or SimpleNamespace(context_window_size=60, is_subagent=False, model_name="m")
        self.registry = registry or SimpleNamespace(repo_root="/repo/host")
        self.model = model
        self._context_tier = context_tier
        self.events = []

    def _cb(self, event, data):
        self.events.append((event, data))


class TestInitContextManager:
    def test_creates_sliding_context(self):
        h = _CtxHost()
        h._init_context_manager()
        assert hasattr(h, "_context_sliding")
        assert hasattr(h._context_sliding, "prepare_before_call")


class TestTrimAndCompress:
    def test_trim_without_manager_returns_unchanged(self):
        h = _CtxHost()
        msgs = ["a", "b"]
        assert h._trim_context(msgs) is msgs

    def test_trim_delegates_to_sliding(self):
        h = _CtxHost()
        h._context_sliding = SimpleNamespace(prepare_before_call=lambda m, budget=None: ["trimmed"])
        assert h._trim_context(["x"]) == ["trimmed"]

    def test_trim_with_budget_delegates_token_budget(self):
        h = _CtxHost()
        seen = {}
        h._context_sliding = SimpleNamespace(
            prepare_before_call=lambda m, budget=None: seen.update(budget=budget) or ["trimmed"]
        )
        assert h._trim_context(["x"], token_budget=1234) == ["trimmed"]
        assert seen["budget"] == 1234

    def test_compress_without_manager_returns_empty(self):
        h = _CtxHost()
        assert h._trajectory_compress(["t"]) == ""

    def test_compress_delegates_to_sliding(self):
        h = _CtxHost()
        h._context_sliding = SimpleNamespace(trajectory_summary=lambda t: "summary")
        assert h._trajectory_compress(["t"]) == "summary"


class TestResolveContextTier:
    def test_subagent_gets_compact(self):
        h = _CtxHost(config=SimpleNamespace(is_subagent=True, context_window_size=60, model_name="m"))
        assert h._resolve_context_tier() == ContextTier.COMPACT

    def test_main_agent_default(self):
        assert _CtxHost()._resolve_context_tier() == ContextTier.MAIN_AGENT


class TestRunGit:
    def test_success_trimmed(self, monkeypatch):
        monkeypatch.setattr(acm.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="  out\n", stderr=""))
        assert _CtxHost()._run_git("status") == "out"

    def test_max_lines_trims_output(self, monkeypatch):
        monkeypatch.setattr(acm.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="l1\nl2\nl3\n", stderr=""))
        assert _CtxHost()._run_git("log", max_lines=2) == "l1\nl2"

    def test_exception_returns_empty(self, monkeypatch):
        def boom(*a, **k):
            raise OSError("git missing")

        monkeypatch.setattr(acm.subprocess, "run", boom)
        assert _CtxHost()._run_git("status") == ""


class TestBuildSessionContext:
    def test_tier_none_defaults_main_agent(self, monkeypatch):
        # None 기본값 → MAIN_AGENT → status 블록 실행("Working tree: clean" 라인 존재)
        monkeypatch.setattr(acm, "get_git_snapshot", lambda root: {"branch": "main", "status": ""})
        h = _CtxHost()
        out = h._build_session_context()
        assert "Working tree: clean" in out

    def test_registry_without_repo_root(self, monkeypatch):
        monkeypatch.setattr(acm, "get_git_snapshot", lambda root: {})
        h = _CtxHost(registry=SimpleNamespace())
        assert h._build_session_context(ContextTier.COMPACT) == "(session context unavailable)"

    def test_branch_and_status_rendered(self, monkeypatch):
        monkeypatch.setattr(acm, "get_git_snapshot", lambda root: {"branch": "main", "status": " M a.py"})
        h = _CtxHost()
        out = h._build_session_context(ContextTier.MAIN_AGENT)
        assert "Working directory: /repo/host" in out
        assert "Branch: main" in out
        assert "Modified files (git status):\n M a.py" in out

    def test_compact_tier_omits_status(self, monkeypatch):
        monkeypatch.setattr(acm, "get_git_snapshot", lambda root: {"branch": "main", "status": " M a.py"})
        h = _CtxHost()
        out = h._build_session_context(ContextTier.COMPACT)
        assert "Branch: main" in out
        assert "status" not in out

    def test_clean_tree_message(self, monkeypatch):
        monkeypatch.setattr(acm, "get_git_snapshot", lambda root: {"branch": "main", "status": ""})
        h = _CtxHost()
        out = h._build_session_context(ContextTier.MAIN_AGENT)
        assert "Working tree: clean" in out


class TestBuildInitialMessages:
    def test_tier_none_uses_instance_tier(self, monkeypatch):
        monkeypatch.setattr(_CtxHost, "_build_session_context", lambda self, tier: "SESSION")
        h = _CtxHost(context_tier=ContextTier.COMPACT)
        msgs = h._build_initial_messages("do it", "PROJ")
        assert len(msgs) == 2
        assert msgs[0].role == "system"
        assert "SESSION" in msgs[0].content and "PROJ" in msgs[0].content
        assert msgs[1].role == "user" and msgs[1].content == "do it"


class TestBuildContinuationMessages:
    def test_conversation_roundtrip_skips_empty(self):
        h = _CtxHost()
        msgs = h._build_continuation_messages(
            {
                "system_prompt": "SYS",
                "conversation": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": ""},  # 빈 턴 스킵
                    {"role": "user", "content": "again"},
                    {"role": "user"},  # content 키 없음 → 빈 값
                ],
            },
            "impl now",
        )
        assert msgs[0].role == "system" and msgs[0].content == "SYS"
        assert [m.content for m in msgs[1:3]] == ["hi", "again"]
        assert "Transition to Implementation Mode" in msgs[3].content
        assert msgs[4].role == "user" and msgs[4].content == "impl now"

    def test_default_role_user(self):
        h = _CtxHost()
        msgs = h._build_continuation_messages({"system_prompt": "S", "conversation": [{"content": "x"}]}, "r")
        assert msgs[1].role == "user"


class TestGetGitSnapshotGenerationChanged:
    def test_generation_bump_discards_collection(self, monkeypatch):
        """수집 중 무효화(generation 증가)가 일어나면 캐시하지 않고 신선본 반환."""

        def seed():
            acm._git_cache_gen += 1  # 수집 중 다른 스레드가 무효화한 상황

        pool = _FakePool(concurrent_seed=seed)
        monkeypatch.setattr("external_llm.agent._thread_pool.shared_pool", pool)
        monkeypatch.setattr(acm, "_run_git_raw", _fake_git_raw)
        snap = get_git_snapshot("/repo/g")
        assert snap["branch"] == "main"  # 수집본 그대로 반환
        assert "/repo/g" not in acm._git_cache  # 캐시에는 저장 안 됨
