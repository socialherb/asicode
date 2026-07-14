"""
Tests for CollaborationOrchestrator.

These tests verify the orchestrator's configuration and digest generation
without requiring a live Claude Code Agent connection.
"""
from __future__ import annotations

import pytest

from external_llm.agent.tool_registry import AgentConfig, ToolRegistry
from external_llm.repl.collaborate import (
    CollaborationOrchestrator,
    CollaborationOrchestratorConfig,
)


class TestOrchestratorConfig:
    """Verify OrchestratorConfig defaults and overrides."""

    def test_default_config(self):
        config = CollaborationOrchestratorConfig()
        assert config.max_iterations == 1
        from config import CLAUDE_SDK_MAX_TURNS
        assert config.max_turns_per_iteration == CLAUDE_SDK_MAX_TURNS
        assert config.model == "sonnet"
        assert config.permission_mode == "bypassPermissions"
        assert config.digest_max_files == 8
        # Digest is basic slim config — agent calls git log/scan directly when needed
        assert config.include_git_history is False
        assert config.include_scanner_results is False
        # Analysis mode default: destructive tools hidden
        assert config.allow_write_tools is False

    def test_custom_config(self):
        config = CollaborationOrchestratorConfig(
            max_turns_per_iteration=20,
            digest_max_files=15,
            include_scanner_results=True,
            allow_write_tools=True,
            model="claude-sonnet-4-20250514",
        )
        assert config.max_turns_per_iteration == 20
        assert config.digest_max_files == 15
        assert config.include_scanner_results is True
        assert config.allow_write_tools is True
        assert config.model == "claude-sonnet-4-20250514"


class TestOrchestratorDigest:
    """Verify digest generation logic (synchronous test)."""

    def test_generate_digest_does_not_crash(self):
        """Digest should run without exceptions even in an empty repo."""
        registry = ToolRegistry(repo_root=".", config=AgentConfig())
        config = CollaborationOrchestratorConfig()

        # We can't test the async method directly without an event loop
        # But we can verify the _build_prompt logic
        orch = CollaborationOrchestrator(registry, config)
        prompt = orch._build_prompt(
            task="Find bugs",
            digest="## Digest\nSome context",
            context="Additional info",
        )
        assert "Find bugs" in prompt
        assert "## Digest" in prompt
        assert "Additional info" in prompt
        # Static directive is in system append, not user message (cache stability)
        assert "# Instructions" not in prompt

    def test_generate_digest_sync_callable_without_event_loop(self):
        # _generate_digest_sync is a sync function, called directly without event loop.
        # (Previously trapped in async wrapper: "We can't test the async method directly
        #  without an event loop" — extracted body made direct testing possible.)
        from types import SimpleNamespace

        registry = ToolRegistry(repo_root=".", config=AgentConfig())
        config = CollaborationOrchestratorConfig()
        orch = CollaborationOrchestrator(registry, config)
        # 실제 도구를 돌리지 않고 빈 결과를 반환 — 속도/부작용 격리.
        registry.dispatch = lambda name, args: SimpleNamespace(ok=True, content="")

        out = orch._generate_digest_sync("find bugs")
        assert isinstance(out, str)

    def test_generate_digest_offloads_to_worker_thread(self):
        # _generate_digest (async) 는 dispatch 호출을 asyncio.to_thread 로
        # worker thread 에 오프로드해야 한다 — 그래야 Phase 1 동안 이벤트 루프가
        # 막히지 않고 interrupt() 가 응답한다. dispatch 가 이벤트 루프 스레드가
        # 아닌 별도 스레드에서 실행됨을 검증한다.
        import asyncio
        import threading
        from types import SimpleNamespace

        main_thread = threading.get_ident()
        seen_threads: list[int] = []

        registry = ToolRegistry(repo_root=".", config=AgentConfig())
        config = CollaborationOrchestratorConfig()
        orch = CollaborationOrchestrator(registry, config)

        def spy_dispatch(name, args):
            seen_threads.append(threading.get_ident())
            return SimpleNamespace(ok=True, content="")

        registry.dispatch = spy_dispatch

        asyncio.run(orch._generate_digest("find bugs"))

        assert seen_threads, "dispatch was never invoked"
        assert any(t != main_thread for t in seen_threads), (
            "digest dispatch ran in the event-loop thread — "
            "event loop is NOT offloaded to a worker"
        )

    def test_build_prompt_no_digest(self):
        registry = ToolRegistry(repo_root=".", config=AgentConfig())
        config = CollaborationOrchestratorConfig()
        orch = CollaborationOrchestrator(registry, config)

        prompt = orch._build_prompt(
            task="Simple task",
            digest="",
            context=None,
        )
        assert "Simple task" in prompt
        assert "# Context" not in prompt  # No digest section when empty

    def test_static_instructions_in_system_prompt(self):
        """정적 협업 지시문이 system preset append에 캐시-안정 형태로 들어간다."""
        pytest.importorskip("claude_agent_sdk")  # get_restricted_options needs the SDK
        from external_llm.repl.collaborate.asi_mcp_adapter import (
            get_restricted_options,
        )
        options = get_restricted_options(mcp_server_config={"type": "sdk"})
        sp = options.system_prompt
        assert isinstance(sp, dict)
        assert sp["preset"] == "claude_code"
        assert sp["exclude_dynamic_sections"] is True
        assert "mcp__asr__" in sp["append"]
        assert "structured verdict" in sp["append"]

    def test_analysis_mode_excludes_destructive_tools(self):
        from external_llm.repl.collaborate.asi_mcp_adapter import (
            get_excluded_tools,
        )
        excluded = get_excluded_tools(allow_write=False)
        # bash is in _ANALYSIS_SAFE_TOOLS, so allowed in analysis mode too
        assert "bash" not in excluded
        # Destructive tools are excluded from analysis mode
        assert "apply_patch" in excluded
        assert "edit_text" in excluded
        # Read-only tools remain exposed
        assert "read_file" not in excluded
        assert "find_relevant_files" not in excluded
        # Write-enabled mode releases destructive tools
        assert "apply_patch" not in get_excluded_tools(allow_write=True)

    def test_build_prompt_with_context_only(self):
        registry = ToolRegistry(repo_root=".", config=AgentConfig())
        config = CollaborationOrchestratorConfig()
        orch = CollaborationOrchestrator(registry, config)

        prompt = orch._build_prompt(
            task="Task",
            digest="",
            context="Extra context",
        )
        assert "Extra context" in prompt


class TestSessionHandoff:
    """asicode 세션 → Claude Code 핸드오프 조립 검증."""

    def _fake_session(self, **kw):
        from types import SimpleNamespace
        defaults = dict(
            compressed_summary="", compressed_up_to=0, archived_count=0,
            turns=[],
        )
        defaults.update(kw)
        return SimpleNamespace(**defaults)

    def test_empty_session(self):
        from external_llm.repl.collaborate import build_session_handoff
        assert build_session_handoff(None) == ""
        assert build_session_handoff(self._fake_session()) == ""

    def test_summary_and_recent_turns(self):
        from external_llm.repl.collaborate import build_session_handoff
        session = self._fake_session(
            compressed_summary="과거 대화: UI 구조 논의",
            compressed_up_to=2,
            archived_count=0,
            turns=[
                {"role": "user", "content": "old1"},
                {"role": "assistant", "content": "old2"},
                {"role": "user", "content": "이 구조 어떻게 생각해?"},
                {"role": "assistant", "content": "A" * 2000},  # 비-최근 턴 → 절단 대상
                {"role": "user", "content": "최근 질문"},        # 가장 최근 턴
            ],
        )
        out = build_session_handoff(session, per_turn_chars=100)
        assert "과거 대화: UI 구조 논의" in out
        # compressed_up_to 이전 턴은 verbatim 영역에서 제외
        assert "old1" not in out
        assert "이 구조 어떻게 생각해?" in out
        assert "…(truncated)" in out  # 비-최근 긴 턴은 결정적 절단

    def test_recent_turn_not_truncated(self):
        """가장 최근 턴(분석 결론/소견)은 per_turn_chars로 잘리지 않는다."""
        from external_llm.repl.collaborate import build_session_handoff
        conclusion = "결론입니다 " + "Z" * 2000
        session = self._fake_session(
            turns=[
                {"role": "user", "content": "B" * 2000},   # 이전 턴 → 절단
                {"role": "assistant", "content": conclusion},  # 최근 턴 → 온전 보존
            ],
        )
        out = build_session_handoff(session, per_turn_chars=100)
        # 최근 턴은 통째로 들어가고 "…(truncated)" 마커가 붙지 않는다
        assert conclusion in out
        # 이전 긴 턴은 잘렸다
        assert "…(truncated)" in out

    def test_max_chars_cap_preserves_recent_tail(self):
        from external_llm.repl.collaborate import build_session_handoff
        # summary(오래됨)는 앞쪽에서 깎이고, 최근 턴의 끝(소견)은 남아야 한다.
        session = self._fake_session(
            compressed_summary="S" * 1500,
            turns=[{"role": "user", "content": "x" * 400} for _ in range(7)]
            + [{"role": "assistant", "content": "y" * 400 + "FINAL_VERDICT"}],
        )
        out = build_session_handoff(session, max_chars=1000)
        assert len(out) <= 1000
        # 앞쪽(오래된 summary)부터 깎으므로 가장 최근 턴 끝의 소견이 보존된다
        assert "FINAL_VERDICT" in out

    def test_summary_exceeds_1500_chars_keeps_tail(self):
        """Regression: [:1500] kept the HEAD (oldest), discarding recent content.
        Now [-1500:] keeps the TAIL — consistent with budget-aware truncation."""
        from external_llm.repl.collaborate import build_session_handoff
        # Build summary > 1500 chars: old prefix + recent marker at the end
        old_prefix = "OLD_" * 500  # 2000 chars of old content
        recent_marker = "RECENT_CONCLUSION"
        full_summary = old_prefix + recent_marker  # 2015 chars
        session = self._fake_session(compressed_summary=full_summary)
        out = build_session_handoff(session, max_chars=8000)
        # The tail (recent conclusion) must survive the 1500-char cap
        assert recent_marker in out
        # The very beginning of old_prefix should be truncated away
        # ([-1500:] keeps last 1500 chars, so first ~515 chars of old_prefix are gone)
        assert not out.startswith(old_prefix[:50])


class TestVerdictForSession:
    """Claude verdict → asicode 세션 주입 텍스트 검증."""

    def test_format_includes_provenance_label(self):
        from external_llm.repl.collaborate import CollaborationVerdict, format_verdict_for_session
        from external_llm.repl.collaborate.claude_session import SessionResult
        result = SessionResult(verdict=CollaborationVerdict(
            status="success", summary="UI는 ui/에 위치", details="상세 내용",
            confidence=0.97, suggestions=["라우터 확인"],
        ))
        out = format_verdict_for_session(result, "UI 파일 위치는?")
        assert "[Claude Code external analysis" in out  # source provenance label
        assert "status: completed" in out
        assert "97%" in out
        assert "UI는 ui/에 위치" in out
        assert "- 라우터 확인" in out

    def test_details_not_truncated(self):
        # 회귀 가드: 예전 char cap이 분석 본문(제안 목록)을 잘라 디자인 LLM이
        # "마지막 제안이 분량 초과로 잘렸다"고 오인했다. 본문은 온전히 실린다.
        from external_llm.repl.collaborate import CollaborationVerdict, format_verdict_for_session
        from external_llm.repl.collaborate.claude_session import SessionResult
        big = "D" * 10000
        result = SessionResult(verdict=CollaborationVerdict(
            status="success", summary="s", details=big,
        ))
        out = format_verdict_for_session(result, "task")
        assert big in out  # 본문 전체 보존 — 절단 없음

    def test_all_suggestions_preserved(self):
        # 회귀 가드: 예전 suggestions[:5]가 6개째부터 버려 "Imp N"을 누락시켰다.
        from external_llm.repl.collaborate import CollaborationVerdict, format_verdict_for_session
        from external_llm.repl.collaborate.claude_session import SessionResult
        sugg = [f"Imp {i}" for i in range(1, 9)]
        result = SessionResult(verdict=CollaborationVerdict(
            status="success", summary="s", details="d", suggestions=sugg,
        ))
        out = format_verdict_for_session(result, "task")
        for s in sugg:
            assert f"- {s}" in out  # 모든 제안 보존


class TestOrchestratorSdkGate:
    """_ensure_session owns the 'clear ImportError on missing SDK' contract.

    It checks SDK availability explicitly (not transitively via
    build_asr_mcp_server) and fails fast before constructing any
    SDK-dependent object — guard-contract: the gate function checks the
    semantic condition promised in its docstring.
    """

    def test_ensure_session_raises_with_install_hint(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
        registry = ToolRegistry(repo_root=".", config=AgentConfig())
        orch = CollaborationOrchestrator(registry, CollaborationOrchestratorConfig())
        with pytest.raises(ImportError) as exc_info:
            orch._ensure_session()
        msg = str(exc_info.value)
        assert "pip install" in msg
        assert "collaborate" in msg
        # Nothing half-constructed on the failure path
        assert orch._mcp_server is None
        assert orch._sdk_options is None
        assert orch._session is None
