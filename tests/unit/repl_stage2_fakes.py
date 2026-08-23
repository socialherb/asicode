"""Shared fakes for repl_impl stage-2 coverage tests (worker + full REPL pty).

Used by BOTH the in-process ``run_subagent_worker`` tests and the spawned
REPL child driver (``repl_stage2_child.py``) so the child behaves byte-
identically to what the in-process tests validated. Every fake mirrors the
established A+B-phase contracts (``test_repl_ab_phase_red_green.py``).
"""

from __future__ import annotations

import contextlib
from datetime import datetime
from types import SimpleNamespace
from typing import ClassVar

# Recent timestamps: should_nudge's age gate (NUDGE_AGE_DAYS_THRESHOLD=21)
# would fire on 2026-01-xx headers and block on `input()` at session end.
_TODAY = datetime.now().strftime("%Y-%m-%d %H:%M +0900")

COMPACTED_INSIGHTS = (
    "## \u2550\u2550\u2550 DESIGN INSIGHTS \u2550\u2550\u2550\n"
    "\n"
    f"### [pattern] {_TODAY}\n"
    "Compacted durable insight line one.\n"
    "\n"
    f"### [design_decision] {_TODAY}\n"
    "Compacted durable insight line two.\n"
)


class FakeAgentConfig:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeToolRegistry:
    instances: ClassVar[list] = []

    def __init__(self, repo_root, config):
        self.repo_root = repo_root
        self.config = config
        FakeToolRegistry.instances.append(self)


class FakeSession:
    """DesignSessionManager session stand-in: turns, chat_mode, add_turn,
    build_context_messages, background compress + compact_now (no-op/True)."""

    def __init__(self):
        self.turns = ["seed"]  # truthy -> /clear runs the compact path
        self.chat_mode = "code"

    def add_turn(self, session_id, role, note, **kw):
        self.turns.append({"session_id": session_id, "role": role, "note": note, **kw})

    def build_context_messages(self, session, **kw):
        return [{"role": "user", "content": "fake context"}]

    def schedule_background_compress(self, session, model, client, **kw):
        return None

    def compact_now(self, session, model, client, **kw):
        return True


class FakeDSM:
    def __init__(self, repo_root):
        self.repo_root = repo_root
        self.sessions = {}

    def get_or_create(self, sid):
        return self.sessions.setdefault(sid, FakeSession())

    def add_turn(self, session_id, role, note, **kw):
        self.get_or_create(session_id).add_turn(session_id, role, note, **kw)

    def build_context_messages(self, session, **kw):
        return [{"role": "user", "content": "fake context"}]

    def schedule_background_compress(self, session, model, client, **kw):
        return None

    def compact_now(self, session, model, client, **kw):
        if hasattr(session, "compact_now"):
            return session.compact_now(session, model, client, **kw)
        return True


class FakeLLMMessage:
    def __init__(self, content, role="assistant"):
        self.role = role
        self.content = content
        self.finish_reason = None
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cache_read_input_tokens = 0


class FakeClient:
    """Routes chat() by prompt content so every in-REPL LLM consumer gets a
    deterministic reply: next-suggestion -> NONE (silent), insights compact ->
    a valid compacted file, everything else -> a generic one-liner."""

    base_url = ""

    def __init__(self):
        self.calls = []

    def chat(self, messages=None, **kw):
        self.calls.append((messages or [], kw))
        joined = "\n".join(getattr(m, "content", "") or "" for m in (messages or []))
        if "[user request]" in joined:
            return FakeLLMMessage("NONE")
        if "### [pattern]" in joined or "### [design_decision]" in joined:
            # Insights-compact input embeds the whole design_insights file,
            # whose entry headers are the only "### [" marker in any REPL
            # prompt (suggestion prompts carry no entries).
            return FakeLLMMessage(COMPACTED_INSIGHTS)
        return FakeLLMMessage("ok, done.")


class FakeSvc:
    def __init__(self, provider="anthropic", model="claude-sonnet-4-6"):
        self.provider = provider
        self.model = model
        self.llm_service = SimpleNamespace(
            provider=provider,
            model=model,
            client=FakeClient(),
            thinking_mode=None,
            reasoning_effort=None,
        )


class FakeDesignChatLoop:
    """respond() returns a REAL DesignChatResult (all fields present).
    ``fail_mode`` drives the worker's cancel/error branches."""

    def __init__(self, client, registry, model, fail_mode=None):
        self.client = client
        self.registry = registry
        self.model = model
        self.fail_mode = fail_mode

    def respond(self, messages, stream_callback=None, **kw):
        if stream_callback is not None:
            with contextlib.suppress(Exception):
                stream_callback("design_llm_call", None)
        if self.fail_mode == "cancel":
            from external_llm.agent.agent_loop_types import AgentCancelled

            raise AgentCancelled("cancelled by orchestrator")
        if self.fail_mode == "error":
            raise RuntimeError("fake task crash")
        from external_llm.agent.design_chat_loop import DesignChatResult

        if self.fail_mode == "error_result":
            # An is_error result (NOT a raised exception) — the REPL's
            # "turn ended with an error" auto-continue stop path keys off
            # chat_result.is_error (L5907), which a raise never reaches.
            return DesignChatResult(
                content="fake error result",
                tool_calls_made=[],
                tokens_used=0,
                prompt_tokens=0,
                completion_tokens=0,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                last_call_prompt_tokens=0,
                last_call_completion_tokens=0,
                last_call_cache_read_tokens=0,
                last_call_cache_creation_tokens=0,
                provider="anthropic",
                is_error=True,
                error_type="general",
                hit_max_iterations=False,
                total_llm_calls=0,
            )
        return DesignChatResult(
            content="Here is the plan: done.",
            tool_calls_made=[],
            tokens_used=12,
            prompt_tokens=6,
            completion_tokens=6,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            last_call_prompt_tokens=6,
            last_call_completion_tokens=6,
            last_call_cache_read_tokens=0,
            last_call_cache_creation_tokens=0,
            provider="anthropic",
            is_error=False,
            hit_max_iterations=False,
            total_llm_calls=1,
        )


class FakeOrchAgent:
    def __init__(
        self,
        llm_client=None,
        registry=None,
        orch_config=None,
        model=None,
        callback=None,
        design_stream_callback=None,
        **kw,
    ):
        self.llm_client = llm_client
        self.registry = registry
        self.orch_config = orch_config
        self.model = model

    def run(self, task, session_id=None, **kw):
        return SimpleNamespace(
            status="success",
            summary="Refactored the parser module.\n2 files touched",
            total_turns=2,
            subtask_results=[],
        )


def worker_args(repo_root: str, **over) -> SimpleNamespace:
    base = {
        "subagent_id": "w1",
        "repo": repo_root,
        "provider": "anthropic",
        "model": "m1",
        "api_key": None,
        "verbose": False,
        "max_turns": 5,
        "orch_pid": 0,
    }
    base.update(over)
    return SimpleNamespace(**base)


# ── /claude collaboration fakes ─────────────────────────────────────────────
# The real ``external_llm.repl.collaborate`` module (and its
# ``streaming_display`` submodule) is replaced wholesale via ``sys.modules``
# in the spawned child, so every ``from external_llm.repl.collaborate import
# ...`` site in repl_impl resolves to these fakes. ``install_state`` is a
# mutable holder so the install-then-run flow (y -> pip install -> SDK
# suddenly "installed") is representable.


class FakeCollabInstallState:
    def __init__(self, sdk_installed: bool = True):
        self.sdk_installed = sdk_installed


class FakeStreamingDisplay:
    """StreamingDisplay stand-in: records events, prints deterministic header/summary."""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.events = []

    def handle_event(self, event, payload):
        self.events.append((event, payload))

    def print_header(self, task, model=None):
        print(f"[collab header] {task} ({model})")

    def print_summary(self, result):
        print(f"[collab summary] error={getattr(result, 'error', None)!r}")

    def flush_log(self):
        print("[collab log flushed]")

    def stop(self):
        pass


class FakeCollabOrchestrator:
    """Async context manager matching CollaborationOrchestrator's __aenter__/run/__aexit__."""

    def __init__(self, registry, config, *, fail_mode=None, result_error=None):
        self.registry = registry
        self.config = config
        self.fail_mode = fail_mode
        self.result_error = result_error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def run(self, task, context=None, enable_preprocessing=True):
        if self.fail_mode == "raise":
            raise RuntimeError("fake collaboration crash")
        if self.fail_mode == "keyboard":
            raise KeyboardInterrupt()
        return SimpleNamespace(error=self.result_error, summary="fake collab result")


class FakeCollabModule:
    """Module stand-in installed at ``external_llm.repl.collaborate``.

    ``install_state`` is shared with the test so the install flow
    (missing -> y -> _pip_install ok -> is_claude_sdk_installed() True)
    is scriptable.
    """

    def __init__(self, install_state: FakeCollabInstallState, orch_fail_mode: str = "none", result_error=None):
        self._state = install_state
        self.orch_fail_mode = orch_fail_mode
        self.result_error = result_error
        self.DEFAULT_COLLAB_MODEL = "claude-sonnet-4-6"
        self.install_spec = ["claude-agent-sdk"]
        self.handoff_raise = False
        self.verdict_raise = False
        self.orch_instances = []

    def is_claude_sdk_installed(self):
        return self._state.sdk_installed

    def build_collaborate_install_spec(self):
        return list(self.install_spec)

    def build_session_handoff(self, session):
        if self.handoff_raise:
            raise RuntimeError("handoff failure")
        return "handoff-context"

    def format_verdict_for_session(self, result, task):
        if self.verdict_raise:
            raise RuntimeError("verdict record failure")
        return "claude verdict note"

    def CollaborationOrchestratorConfig(self, **kw):
        return SimpleNamespace(**kw)

    def make_orchestrator(self, registry, config):
        orch = FakeCollabOrchestrator(registry, config, fail_mode=self.orch_fail_mode, result_error=self.result_error)
        self.orch_instances.append(orch)
        return orch

    def CollaborationOrchestrator(self, registry, config):
        return self.make_orchestrator(registry, config)


class FakeStreamingDisplayModule:
    """Stand-in for ``external_llm.repl.collaborate.streaming_display``."""

    def __init__(self):
        self.instances = []

    def StreamingDisplay(self, verbose=False):
        disp = FakeStreamingDisplay(verbose=verbose)
        self.instances.append(disp)
        return disp
