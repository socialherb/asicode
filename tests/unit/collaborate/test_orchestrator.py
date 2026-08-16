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
        # Returns an empty result without running the real tool — isolates speed/side-effects.
        registry.dispatch = lambda name, args: SimpleNamespace(ok=True, content="")

        out = orch._generate_digest_sync("find bugs")
        assert isinstance(out, str)

    def test_generate_digest_offloads_to_worker_thread(self):
        # _generate_digest (async) must offload the dispatch call to a worker
        # thread via asyncio.to_thread — otherwise the event loop stays
        # blocked during Phase 1 and interrupt() can't respond. Verifies
        # dispatch runs on a separate thread, not the event-loop thread.
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
        """Static collaboration instructions land in the system preset append in a cache-stable form."""
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
    """Verify assembly of the asicode session → Claude Code handoff."""

    def _fake_session(self, **kw):
        from types import SimpleNamespace
        defaults = {
            "compressed_summary": "",
            "compressed_up_to": 0,
            "archived_count": 0,
            "turns": [],
        }
        defaults.update(kw)
        return SimpleNamespace(**defaults)

    def test_empty_session(self):
        from external_llm.repl.collaborate import build_session_handoff
        assert build_session_handoff(None) == ""
        assert build_session_handoff(self._fake_session()) == ""

    def test_summary_and_recent_turns(self):
        from external_llm.repl.collaborate import build_session_handoff
        session = self._fake_session(
            compressed_summary="past conversation: discussed the UI structure",
            compressed_up_to=2,
            archived_count=0,
            turns=[
                {"role": "user", "content": "old1"},
                {"role": "assistant", "content": "old2"},
                {"role": "user", "content": "what do you think of this structure?"},
                {"role": "assistant", "content": "A" * 2000},  # non-recent turn → truncation candidate
                {"role": "user", "content": "recent question"},  # the most recent turn
            ],
        )
        out = build_session_handoff(session, per_turn_chars=100)
        assert "past conversation: discussed the UI structure" in out
        # Turns before compressed_up_to are excluded from the verbatim region
        assert "old1" not in out
        assert "what do you think of this structure?" in out
        assert "…(truncated)" in out  # non-recent long turns are deterministically truncated

    def test_recent_turn_not_truncated(self):
        """The most recent turn (analysis conclusion/finding) is not truncated by per_turn_chars."""
        from external_llm.repl.collaborate import build_session_handoff
        conclusion = "here is the conclusion " + "Z" * 2000
        session = self._fake_session(
            turns=[
                {"role": "user", "content": "B" * 2000},   # previous turn → truncated
                {"role": "assistant", "content": conclusion},  # recent turn → preserved intact
            ],
        )
        out = build_session_handoff(session, per_turn_chars=100)
        # The recent turn goes in whole, with no "…(truncated)" marker attached
        assert conclusion in out
        # The previous long turn was truncated
        assert "…(truncated)" in out

    def test_max_chars_cap_preserves_recent_tail(self):
        from external_llm.repl.collaborate import build_session_handoff
        # The (old) summary should be trimmed from the front, and the end of the recent turn (the finding) must survive.
        session = self._fake_session(
            compressed_summary="S" * 1500,
            turns=[{"role": "user", "content": "x" * 400} for _ in range(7)]
            + [{"role": "assistant", "content": "y" * 400 + "FINAL_VERDICT"}],
        )
        out = build_session_handoff(session, max_chars=1000)
        assert len(out) <= 1000
        # Trimming happens from the front (old summary), so the finding at the end of the most recent turn is preserved
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
    """Verify the Claude verdict → asicode session injection text."""

    def test_format_includes_provenance_label(self):
        from external_llm.repl.collaborate import CollaborationVerdict, format_verdict_for_session
        from external_llm.repl.collaborate.claude_session import SessionResult
        result = SessionResult(verdict=CollaborationVerdict(
            status="success", summary="the UI lives in ui/", details="detailed content",
            confidence=0.97, suggestions=["check the router"],
        ))
        out = format_verdict_for_session(result, "where is the UI located?")
        assert "[Claude Code external analysis" in out  # source provenance label
        assert "status: completed" in out
        assert "97%" in out
        assert "the UI lives in ui/" in out
        assert "- check the router" in out

    def test_details_not_truncated(self):
        # Regression guard: the old char cap truncated the analysis body (suggestion
        # list), causing the design LLM to mistake it for "the last suggestion got
        # cut off due to length." The body is now carried in full.
        from external_llm.repl.collaborate import CollaborationVerdict, format_verdict_for_session
        from external_llm.repl.collaborate.claude_session import SessionResult
        big = "D" * 10000
        result = SessionResult(verdict=CollaborationVerdict(
            status="success", summary="s", details=big,
        ))
        out = format_verdict_for_session(result, "task")
        assert big in out  # entire body preserved — no truncation

    def test_all_suggestions_preserved(self):
        # Regression guard: the old suggestions[:5] dropped everything from the 6th onward, losing "Imp N".
        from external_llm.repl.collaborate import CollaborationVerdict, format_verdict_for_session
        from external_llm.repl.collaborate.claude_session import SessionResult
        sugg = [f"Imp {i}" for i in range(1, 9)]
        result = SessionResult(verdict=CollaborationVerdict(
            status="success", summary="s", details="d", suggestions=sugg,
        ))
        out = format_verdict_for_session(result, "task")
        for s in sugg:
            assert f"- {s}" in out  # every suggestion preserved

    def test_plan_included_when_present(self):
        # The structured plan (output_format schema: "structured plan for asicode
        # to execute") must reach the design session — previously parsed and
        # stored but never surfaced (dead contract).
        from external_llm.repl.collaborate import CollaborationVerdict, format_verdict_for_session
        from external_llm.repl.collaborate.claude_session import SessionResult
        plan = {"steps": [{"action": "read_file", "path": "ui/"}], "goal": "locate UI"}
        result = SessionResult(verdict=CollaborationVerdict(
            status="success", summary="s", details="d", plan=plan,
        ))
        out = format_verdict_for_session(result, "task")
        assert "plan:" in out
        assert '"goal": "locate UI"' in out
        assert "read_file" in out

    def test_metadata_included_when_present(self):
        from external_llm.repl.collaborate import CollaborationVerdict, format_verdict_for_session
        from external_llm.repl.collaborate.claude_session import SessionResult
        result = SessionResult(verdict=CollaborationVerdict(
            status="success", summary="s", details="d",
            metadata={"tokens": 1234, "tool_calls": 7},
        ))
        out = format_verdict_for_session(result, "task")
        assert "metadata:" in out
        assert '"tokens": 1234' in out

    def test_plan_metadata_omitted_when_absent(self):
        # Absent plan/metadata must not produce empty stub lines in the injection text.
        from external_llm.repl.collaborate import CollaborationVerdict, format_verdict_for_session
        from external_llm.repl.collaborate.claude_session import SessionResult
        result = SessionResult(verdict=CollaborationVerdict(
            status="success", summary="s", details="d",
        ))
        out = format_verdict_for_session(result, "task")
        assert "plan:" not in out
        assert "metadata:" not in out

    def test_plan_metadata_unserializable_values_safe(self):
        # Untrusted model output can carry non-JSON types — must not crash injection.
        from external_llm.repl.collaborate import CollaborationVerdict, format_verdict_for_session
        from external_llm.repl.collaborate.claude_session import SessionResult
        result = SessionResult(verdict=CollaborationVerdict(
            status="success", summary="s", details="d",
            plan={"when": object()},
            metadata={"blob": object()},
        ))
        out = format_verdict_for_session(result, "task")  # no exception
        assert "plan:" in out
        assert "metadata:" in out

    def test_plan_metadata_sdk_xml_tags_stripped(self):
        # Same leak-guard as details: SDK/verdict XML tags must not reach the design LLM.
        from external_llm.repl.collaborate import CollaborationVerdict, format_verdict_for_session
        from external_llm.repl.collaborate.claude_session import SessionResult
        result = SessionResult(verdict=CollaborationVerdict(
            status="success", summary="s", details="d",
            plan={"steps": ["run <status>check</status> now"]},
            metadata={"note": "<plan>leak</plan>"},
        ))
        out = format_verdict_for_session(result, "task")
        assert "<status>" not in out
        assert "<plan>" not in out
        assert "check" in out  # inner text preserved


class TestOrchestratorNonSdkBranches:
    """run()/interrupt()/digest branches reachable without the Claude SDK.

    _ensure_session() returns the cached session when already set, so a fake
    session injected via orch._session exercises the full run() flow without
    claude_agent_sdk installed.
    """

    def test_run_success_path(self):
        import asyncio

        from external_llm.repl.collaborate import CollaborationVerdict
        from external_llm.repl.collaborate.claude_session import SessionResult

        class FakeSession:
            def __init__(self):
                self.queries = []
            async def query(self, prompt):
                self.queries.append(prompt)
                return SessionResult(
                    verdict=CollaborationVerdict(status="success", summary="done"),
                    duration_seconds=1.5,
                    tool_calls_count=3,
                )

        registry = ToolRegistry(repo_root=".", config=AgentConfig())
        orch = CollaborationOrchestrator(registry, CollaborationOrchestratorConfig())
        fake = FakeSession()
        orch._session = fake  # bypass the SDK gate — _ensure_session returns cached

        result = asyncio.run(orch.run("review the change", enable_preprocessing=False))
        assert result.verdict.summary == "done"
        assert len(fake.queries) == 1
        assert "review the change" in fake.queries[0]

    def test_run_error_path(self):
        import asyncio

        from external_llm.repl.collaborate import CollaborationVerdict
        from external_llm.repl.collaborate.claude_session import SessionResult

        class FakeSession:
            async def query(self, prompt):
                return SessionResult(
                    verdict=CollaborationVerdict(), error="agent gave up",
                )

        registry = ToolRegistry(repo_root=".", config=AgentConfig())
        orch = CollaborationOrchestrator(registry, CollaborationOrchestratorConfig())
        orch._session = FakeSession()
        result = asyncio.run(orch.run("task", enable_preprocessing=False))
        assert result.error == "agent gave up"

    def test_run_with_preprocessing_digest_phase(self):
        # enable_preprocessing=True runs the digest phase (dispatch calls)
        # before querying the session.
        import asyncio
        from types import SimpleNamespace

        from external_llm.repl.collaborate import CollaborationVerdict
        from external_llm.repl.collaborate.claude_session import SessionResult

        class FakeSession:
            async def query(self, prompt):
                return SessionResult(
                    verdict=CollaborationVerdict(status="success", summary="ok"),
                )

        registry = ToolRegistry(repo_root=".", config=AgentConfig())
        registry.dispatch = lambda name, args: SimpleNamespace(ok=True, content="digest-piece")
        orch = CollaborationOrchestrator(registry, CollaborationOrchestratorConfig())
        orch._session = FakeSession()
        result = asyncio.run(orch.run("task"))
        assert result.verdict.summary == "ok"

    def test_digest_optional_collector_failures_guarded(self):
        # Git/scan collectors only run when enabled — their failure paths are
        # individually guarded like the base collectors.
        registry = ToolRegistry(repo_root=".", config=AgentConfig())
        def boom(name, args):
            raise RuntimeError("tool unavailable")
        registry.dispatch = boom
        config = CollaborationOrchestratorConfig(
            include_git_history=True,
            include_scanner_results=True,
        )
        orch = CollaborationOrchestrator(registry, config)
        assert orch._generate_digest_sync("find bugs") == ""

    def test_interrupt_and_session_property(self):
        import asyncio

        class FakeSession:
            def __init__(self):
                self.interrupted = False
            async def interrupt(self):
                self.interrupted = True

        registry = ToolRegistry(repo_root=".", config=AgentConfig())
        orch = CollaborationOrchestrator(registry, CollaborationOrchestratorConfig())
        assert orch.session is None
        fake = FakeSession()
        orch._session = fake
        assert orch.session is fake
        asyncio.run(orch.interrupt())
        assert fake.interrupted
        # No session → no-op, not an error
        orch._session = None
        asyncio.run(orch.interrupt())

    def test_write_tools_warning(self, caplog):
        # allow_write_tools=True with the default bypassPermissions mode must
        # warn that destructive tools will run without user approval.
        import logging
        registry = ToolRegistry(repo_root=".", config=AgentConfig())
        config = CollaborationOrchestratorConfig(allow_write_tools=True)
        with caplog.at_level(
            logging.WARNING,
            logger="external_llm.repl.collaborate.collaboration_orchestrator",
        ):
            CollaborationOrchestrator(registry, config)
        assert any("destructive tools" in r.message for r in caplog.records)

    def test_generate_digest_sync_survives_dispatch_failures(self):
        # Each collector is individually guarded — a failing tool must not
        # kill the whole digest.
        registry = ToolRegistry(repo_root=".", config=AgentConfig())
        def boom(name, args):
            raise RuntimeError("tool unavailable")
        registry.dispatch = boom
        orch = CollaborationOrchestrator(registry, CollaborationOrchestratorConfig())
        assert orch._generate_digest_sync("find bugs") == ""

    def test_generate_digest_optional_collectors(self):
        from types import SimpleNamespace
        registry = ToolRegistry(repo_root=".", config=AgentConfig())
        def fake_dispatch(name, args):
            return SimpleNamespace(ok=True, content="collector-content")
        registry.dispatch = fake_dispatch
        config = CollaborationOrchestratorConfig(
            include_git_history=True,
            include_scanner_results=True,
        )
        orch = CollaborationOrchestrator(registry, config)
        out = orch._generate_digest_sync("find bugs")
        assert "## Project Info" in out
        assert "## Relevant Files" in out
        assert "## Recent Git History" in out
        assert "collector-content" in out  # scan result included


class TestSessionHandoffTrimEdge:
    """build_session_handoff summary-trim edge (avail <= marker length)."""

    def test_summary_trimmed_to_avail_when_marker_does_not_fit(self):
        from types import SimpleNamespace

        from external_llm.repl.collaborate import build_session_handoff
        session = SimpleNamespace(
            compressed_summary="S" * 100,
            compressed_up_to=0,
            archived_count=0,
            turns=[],
        )
        # max_chars leaves avail=8 after the header — too small for the
        # "…(truncated) " marker, so the plain tail-cut branch applies.
        out = build_session_handoff(session, max_chars=60)
        assert len(out) == 60
        assert out.endswith("SSSSSSSS")  # summary[-8:]


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
