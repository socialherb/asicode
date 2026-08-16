"""agent_loop.py RED->GREEN coverage.

Baseline: 65.35% (684 statements / 237 missing) over tests/unit/agent (6264 tests).

Covers the remaining untested surface of AgentLoop:
- git-state collection (no repo root / exception), patch-rollback failure paths
- __init__ branches: cancel_event wiring, helper backend init + failure
- _try_readonly_early_finish decision matrix
- _strip_thinking_text markers, _extract_known_file_path, _extract_target_keywords
- run(): profile apply, low-confidence route, read-only phase, FS-op EDIT phase,
  routing-intent stream event, checkpoint-id metadata stamping
- _save_session_log rotation + failure; _cb enrichment + failure
- _check_native_tool_support (non-native provider, ollama no-tools)
- _llm_call_with_tools pre-flight guards (cancel, repair-drop) + dict tool_calls
- _retry_on_rate_limit error matrix: connection exhaustion, rate-limit recovery,
  server-unavailable recovery, LLMCancelled, cancel-before/during, None result,
  NaN tokens, reasoning-token logging
- _auto_repair_apply_patch_args repair rules 1-3 + failure exits
- guidance appenders (write_plan / edit_warnings / syntax / semantic / patch retry)
- _append_native_tool_messages per provider (openai reasoning+filter, ollama, generic)
- _hunk_to_before_after
"""
from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from external_llm.agent.agent_loop import AgentLoop
from external_llm.agent.agent_loop_types import AgentCancelled, AgentResult, AgentTurn
from external_llm.agent.task_router import Lane
from external_llm.agent.tool_registry import AgentConfig, ToolRegistry, ToolResult
from external_llm.client import (
    LLMCancelled,
    LLMConnectionError,
    LLMMessage,
    LLMRateLimitError,
    LLMServerUnavailableError,
)

# ---------------------------------------------------------------------------
# Harness — AgentLoop via __new__ (no __init__ side effects) with mocked deps
# ---------------------------------------------------------------------------

def _harness(**cfg_over):
    """Minimal AgentLoop host with the attributes the covered methods read."""
    loop = AgentLoop.__new__(AgentLoop)
    cfg = {
        "max_turns": 5,
        "model_name": "claude-test",
        "stream_callback": None,
        "cancel_event": None,
        "thinking_mode": False,
        "reasoning_effort": None,
        "session_id": "sess-1",
        "continuation_data": None,
        "agent_profile": None,
        "design_chat_mode": False,
    }
    cfg.update(cfg_over)
    loop.config = SimpleNamespace(**cfg)
    loop.llm_client = mock.MagicMock()
    loop.llm_client.get_provider_name.return_value = "anthropic"
    loop.llm_client.base_url = ""
    loop.registry = mock.MagicMock()
    loop.registry.applied_patches = []
    loop.registry.repo_root = "/tmp"
    loop.registry.repo_language = None
    loop.registry.get_tool_schemas.return_value = []
    loop.registry.local_assistant = None
    loop.model = "claude-test"
    loop.agent_id = "main"
    loop.performance_collector = mock.MagicMock()
    loop.performance_collector.get_summary.return_value = {"tools": 0}
    loop._shared_run_store = mock.MagicMock()
    loop._tool_success_memory = {}
    loop._tool_fail_memory = {}
    loop._context_budget = None
    loop._cb = mock.MagicMock()
    loop._record_llm_call_both = mock.MagicMock()
    return loop


class _FastEvent(threading.Event):
    """Event whose wait() never blocks longer than 1 ms (retry-backoff tests)."""

    def wait(self, timeout=None):
        if timeout is not None:
            timeout = min(timeout, 0.001)
        return super().wait(timeout=timeout)


# ---------------------------------------------------------------------------
# _collect_git_info / _rollback_patches
# ---------------------------------------------------------------------------

def test_collect_git_info_no_repo_root():
    loop = _harness()
    loop.registry.repo_root = None
    assert loop._collect_git_info() == {}


def test_collect_git_info_snapshot_exception():
    loop = _harness()
    with mock.patch(
        "external_llm.agent.agent_loop.get_git_snapshot", side_effect=RuntimeError("git boom")
    ):
        assert loop._collect_git_info() == {}


def test_rollback_patches_apply_fails(tmp_path):
    """git apply -R --check passes but apply fails -> non-destructive manual-rollback result."""
    loop = _harness()
    loop.registry.repo_root = str(tmp_path)
    patch = "diff --git a/f.txt b/f.txt\n--- a/f.txt\n+++ b/f.txt\n@@ -1 +1 @@\n-old\n+new\n"

    def fake_run(cmd, **kw):
        if "--check" in cmd:
            return SimpleNamespace(returncode=0, stderr="")
        return SimpleNamespace(returncode=1, stderr="apply exploded")

    with mock.patch("subprocess.run", side_effect=fake_run):
        res = loop._rollback_patches([patch])

    assert res["success"] is False
    r = res["results"][0]
    assert r["needs_manual_rollback"] is True
    assert r["affected_files"] == ["f.txt"]
    assert "apply exploded" in r["primary_error"]


def test_rollback_patches_exception(tmp_path):
    loop = _harness()
    loop.registry.repo_root = str(tmp_path)
    with mock.patch("subprocess.run", side_effect=RuntimeError("git missing")):
        res = loop._rollback_patches(["some patch"])
    assert res["success"] is False
    assert "Exception during rollback" in res["results"][0]["message"]


# ---------------------------------------------------------------------------
# __init__ branches
# ---------------------------------------------------------------------------

def _real_loop(**cfg_over):
    """Construct a REAL AgentLoop (exercises __init__ branches)."""
    cfg = AgentConfig(max_turns=1, rag_enabled=False)
    for k, v in cfg_over.items():
        setattr(cfg, k, v)
    reg = SimpleNamespace(repo_root="/tmp", local_assistant=None)
    client = mock.MagicMock()
    client.get_provider_name.return_value = "openai"
    return AgentLoop(llm_client=client, registry=reg, config=cfg, model="test")


def test_init_wires_cancel_event():
    ev = threading.Event()
    loop = _real_loop(cancel_event=ev)
    assert loop.llm_client.cancel_event is ev


def test_init_helper_enabled_initializes_backend():
    fake = mock.MagicMock()
    fake.return_value = "assistant-instance"
    with mock.patch("external_llm.agent.local_assistant.LocalAssistant", fake):
        loop = _real_loop(helper_enabled=True, helper_model="llama3")
    assert loop.registry.local_assistant == "assistant-instance"
    assert fake.call_args.kwargs["local_model"] == "llama3"


def test_init_helper_init_failure_sets_none():
    def _boom(*a, **k):
        raise RuntimeError("ollama unreachable")

    with mock.patch("external_llm.agent.local_assistant.LocalAssistant", side_effect=_boom):
        loop = _real_loop(helper_enabled=True, helper_model="llama3")
    assert loop.registry.local_assistant is None


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------

def test_resolve_routing_intent_design_chat():
    loop = _harness(design_chat_mode=True)
    assert loop._resolve_routing_intent(None) == "read_only"


def test_record_tool_usage_store_failure_still_memorizes():
    loop = _harness()
    loop._shared_run_store.record_tool_usage.side_effect = RuntimeError("disk full")
    loop._record_tool_success("read_file", {"path": "a.py"})
    assert loop._tool_success_memory
    key = next(iter(loop._tool_success_memory))
    assert loop._tool_success_memory[key][0] == "read_file"


def test_repair_json_brackets_delegates():
    assert AgentLoop._repair_json_brackets('{"a": [1, 2') == '{"a": [1, 2]}'


def test_try_parse_json_delegates():
    loop = _harness()
    assert loop._try_parse_json('{"a": 1}') == {"a": 1}


# ---------------------------------------------------------------------------
# _try_readonly_early_finish
# ---------------------------------------------------------------------------

def _ro_result(ok=True, content="", tool="find_symbol"):
    return ToolResult(ok=ok, content=content, error=None if ok else "boom")


def test_readonly_early_finish_not_readonly():
    loop = _harness()
    assert loop._try_readonly_early_finish("find_symbol", _ro_result(), "x", False) is None


def test_readonly_early_finish_tool_failed():
    loop = _harness()
    assert loop._try_readonly_early_finish("find_symbol", _ro_result(ok=False), "x", True) is None


def test_readonly_early_finish_analysis_intent():
    loop = _harness()
    for req in ("explain this code", "what is going on?"):
        assert loop._try_readonly_early_finish("find_symbol", _ro_result(), req, True) is None


def test_readonly_early_finish_short_content():
    loop = _harness()
    assert loop._try_readonly_early_finish("find_symbol", _ro_result(content="short"), "x", True) is None


def test_readonly_early_finish_definitive():
    loop = _harness()
    content = "c" * 100
    res = loop._try_readonly_early_finish("find_symbol", _ro_result(content=content), "x", True)
    assert res is not None
    assert res.status == "success"
    assert res.metadata["readonly_early_finish"] is True
    assert res.metadata["tool"] == "find_symbol"
    assert res.final_message == content


def test_readonly_early_finish_long_preview_truncated():
    loop = _harness()
    content = "c" * 500
    res = loop._try_readonly_early_finish("get_project_info", _ro_result(content=content), "x", True)
    assert res.final_message == "c" * 400 + "…"


# ---------------------------------------------------------------------------
# _strip_thinking_text
# ---------------------------------------------------------------------------

def test_strip_thinking_empty():
    loop = _harness()
    assert loop._strip_thinking_text("") == ""


def test_strip_thinking_think_block():
    loop = _harness()
    assert loop._strip_thinking_text("<think>secret plan</think>\nanswer") == "answer"


def test_strip_thinking_stray_tags():
    loop = _harness()
    assert loop._strip_thinking_text("a</think>b<think>") == "ab"


def test_strip_thinking_suspicious_marker():
    # The slice keeps the marker text itself (idx points at the leading "\n"),
    # so the stripped result retains "Final answer:" — documented actual behavior.
    loop = _harness()
    assert loop._strip_thinking_text("let me think\nFinal answer: yes") == "Final answer: yes"


def test_strip_thinking_marker_without_suspicion_kept():
    loop = _harness()
    text = "intro\nFinal answer: yes"
    assert loop._strip_thinking_text(text) == text


def test_strip_thinking_clean_text():
    loop = _harness()
    assert loop._strip_thinking_text("  plain answer  ") == "plain answer"


# ---------------------------------------------------------------------------
# _extract_known_file_path / _extract_target_keywords
# ---------------------------------------------------------------------------

def test_extract_known_file_empty_request():
    loop = _harness()
    assert loop._extract_known_file_path("") == ""


def test_extract_known_file_no_candidates():
    loop = _harness()
    assert loop._extract_known_file_path("hello world") == ""


def test_extract_known_file_found(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "x.py").write_text("x = 1\n")
    loop = _harness()
    loop.registry.repo_root = str(tmp_path)
    assert loop._extract_known_file_path("fix src/x.py please") == "src/x.py"


def test_extract_known_file_oserror_skipped(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "x.py").write_text("x = 1\n")
    loop = _harness()
    loop.registry.repo_root = str(tmp_path)
    with mock.patch("pathlib.Path.is_file", side_effect=OSError("denied")):
        assert loop._extract_known_file_path("fix src/x.py please") == ""


def test_extract_target_keywords_quoted_cap_three():
    loop = _harness()
    out = loop._extract_target_keywords('fix "alpha" and "beta" and "gamma" and "delta"')
    assert out == ["alpha", "beta", "gamma"]


# ---------------------------------------------------------------------------
# run() branches (real AgentLoop in a tmp git repo)
# ---------------------------------------------------------------------------

def _run(cmd, cwd, **kw):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, **kw, check=False)


def _make_run_loop(tmp_path) -> tuple[AgentLoop, Path]:
    repo = Path(tmp_path)
    _run(["git", "init", "-q"], cwd=str(repo))
    _run(["git", "config", "user.email", "t@t.com"], cwd=str(repo))
    _run(["git", "config", "user.name", "t"], cwd=str(repo))
    (repo / "f.txt").write_text("alpha=1\n")
    _run(["git", "add", "f.txt"], cwd=str(repo))
    _run(["git", "commit", "-qm", "base"], cwd=str(repo))
    client = mock.MagicMock()
    client.get_provider_name.return_value = "openai"
    client.provider = "openai"
    cfg = AgentConfig(max_turns=1, rag_enabled=False)
    reg = ToolRegistry(str(repo), cfg)
    loop = AgentLoop(llm_client=client, registry=reg, config=cfg, model="test")
    loop._run_llm_loop = lambda ctx: AgentResult(
        status="success", final_message="ok", turns=[], applied_patches=[], metadata={}
    )
    return loop, repo


def _main_route(**over):
    route = {
        "lane": Lane.MAIN_AGENT, "confidence": 0.9, "task_kind": "general",
        "reasoning": "", "complexity": "low", "target_specificity_score": 0.5,
    }
    route.update(over)
    return SimpleNamespace(**route)


def test_run_applies_agent_profile(tmp_path):
    loop, _ = _make_run_loop(tmp_path)
    loop.config.route_decision = _main_route()
    loop.config.agent_profile = SimpleNamespace(name="prof", apply=lambda cfg: setattr(cfg, "_applied", True))
    result = loop.run("change alpha to 2")
    assert result.status == "success"
    assert loop.config._applied is True


def test_run_low_confidence_route_warns(tmp_path):
    loop, _ = _make_run_loop(tmp_path)
    loop.config.route_decision = _main_route(confidence=0.05)
    result = loop.run("change alpha to 2")
    assert result.status == "success"


def test_run_readonly_intent_sets_discover_phase(tmp_path):
    loop, _ = _make_run_loop(tmp_path)
    loop.config.design_chat_mode = True
    result = loop.run("explain f.txt")
    assert result.metadata.get("unhandled_lane") is True
    assert loop._agent_phase == "DISCOVER"


def test_run_filesystem_op_starts_in_edit_phase(tmp_path):
    loop, _ = _make_run_loop(tmp_path)
    loop.config.route_decision = _main_route(reasoning="Filesystem operation detected")
    result = loop.run("change alpha to 2")
    assert result.status == "success"
    assert loop._agent_phase == "EDIT"


def test_run_streams_routing_intent_event(tmp_path):
    loop, _ = _make_run_loop(tmp_path)
    events = []
    loop.config.stream_callback = lambda e, d: events.append((e, d))
    loop.config.route_decision = _main_route()
    loop.run("change alpha to 2")
    assert any(e == "routing_intent" for e, _ in events)


def test_run_stamps_checkpoint_id(tmp_path):
    loop, _ = _make_run_loop(tmp_path)
    loop.config.route_decision = _main_route()
    loop._run_llm_loop = lambda ctx: AgentResult(
        status="success", final_message="ok", turns=[], applied_patches=[], metadata=None
    )
    with mock.patch.object(
        ToolRegistry, "run_checkpoint_id", new_callable=mock.PropertyMock, return_value="cp-1"
    ):
        result = loop.run("change alpha to 2")
    assert result.metadata["checkpoint_id"] == "cp-1"


# ---------------------------------------------------------------------------
# _save_session_log / _cb
# ---------------------------------------------------------------------------

def _session_result():
    turn = AgentTurn(0, "read_file", {}, ToolResult(ok=True, content="c", metadata={"touched_files": ["a.py"]}))
    return AgentResult(status="success", final_message="ok", turns=[turn], applied_patches=[], metadata={})


def test_save_session_log_rotation(tmp_path):
    log_dir = Path(tmp_path) / ".asicode"
    log_dir.mkdir()
    (log_dir / "sessions.jsonl").write_text("x" * (10 * 1024 * 1024 + 50))
    loop = _harness()
    loop.registry = SimpleNamespace(repo_root=str(tmp_path))
    loop._save_session_log("s1", "req", _session_result(), 10, 5)
    assert (log_dir / "sessions.jsonl.1").exists()
    rec = json.loads((log_dir / "sessions.jsonl").read_text())
    assert rec["status"] == "success"
    assert rec["touched_files"] == ["a.py"]


def test_save_session_log_exception_swallowed(tmp_path):
    loop = _harness()
    loop.registry = SimpleNamespace(repo_root=str(tmp_path))
    with mock.patch.object(AgentLoop, "_ensure_asicode_gitignored", side_effect=RuntimeError("boom")):
        loop._save_session_log("s1", "req", AgentResult(status="success"), 1, 1)  # no raise


def test_cb_enriches_and_forwards():
    loop = _harness(stream_callback=None)
    events = []
    loop.config.stream_callback = lambda e, d: events.append((e, d))
    loop._cb = AgentLoop._cb.__get__(loop)
    loop.turns = [1, 2, 3]
    loop._orchestrator_phase = "phase-x"
    loop._cb("evt", {"agent_id": "custom"})
    e, d = events[0]
    assert e == "evt"
    assert d["agent_id"] == "custom"
    assert d["agent_turn_num"] == 3
    assert d["orchestrator_phase"] == "phase-x"
    assert d["session_id"] == "sess-1"
    assert "global_sequence_id" in d


def test_cb_adds_default_agent_id():
    loop = _harness()
    events = []
    loop.config.stream_callback = lambda e, d: events.append(d)
    loop._cb = AgentLoop._cb.__get__(loop)
    loop._cb("evt", {})
    assert events[0]["agent_id"] == "main"


def test_cb_callback_exception_swallowed():
    loop = _harness()

    def _boom(e, d):
        raise RuntimeError("cb boom")

    loop.config.stream_callback = _boom
    loop._cb = AgentLoop._cb.__get__(loop)
    loop._cb("evt", {})  # no raise


# ---------------------------------------------------------------------------
# _check_native_tool_support
# ---------------------------------------------------------------------------

def test_native_support_non_native_provider():
    loop = _harness()
    loop.llm_client.get_provider_name.return_value = "custom"
    assert loop._check_native_tool_support() is False


def test_native_support_ollama_no_tools():
    loop = _harness()
    loop.llm_client.get_provider_name.return_value = "ollama"
    loop.model = "llama3"
    with mock.patch("external_llm.model_registry.ollama_supports_tools", return_value=False):
        assert loop._check_native_tool_support() is False


def test_native_support_ollama_unknown_assumes_supported():
    loop = _harness()
    loop.llm_client.get_provider_name.return_value = "ollama"
    loop.model = "llama3"
    with mock.patch("external_llm.model_registry.ollama_supports_tools", return_value=None):
        assert loop._check_native_tool_support() is True


# ---------------------------------------------------------------------------
# _llm_call_with_tools
# ---------------------------------------------------------------------------

def _llm_response(**over):
    resp = mock.MagicMock()
    resp.finish_reason = "end_turn"
    resp.content = "ok"
    resp.tool_calls = []
    resp.prompt_tokens = 10
    resp.completion_tokens = 5
    resp.tokens_used = None
    resp.cache_read_input_tokens = None
    resp.cache_creation_input_tokens = None
    for k, v in over.items():
        setattr(resp, k, v)
    return resp


def test_llm_call_cancelled_before_call():
    ev = threading.Event()
    ev.set()
    loop = _harness(cancel_event=ev)
    with pytest.raises(AgentCancelled):
        loop._llm_call_with_tools([LLMMessage(role="user", content="hi")])


def test_llm_call_repair_drop_notifies():
    loop = _harness()
    loop.llm_client.chat_with_tools.return_value = _llm_response()
    with mock.patch(
        "external_llm.agent.context_budget.repair_tool_message_sequence",
        side_effect=lambda msgs: msgs[:-1],
    ):
        out = loop._llm_call_with_tools(
            [LLMMessage(role="user", content="a"), LLMMessage(role="user", content="b")]
        )
    assert out["content"] == "ok"
    loop._cb.assert_called_once()
    assert loop._cb.call_args.args[0] == "agent_working"
    assert loop._cb.call_args.args[1]["dropped"] == 1


def test_llm_call_normalizes_dict_tool_calls():
    loop = _harness()
    calls = [{"id": "c1", "name": "read_file", "args": {"path": "a.py"}}]
    loop.llm_client.chat_with_tools.return_value = _llm_response(tool_calls=calls)
    out = loop._llm_call_with_tools([LLMMessage(role="user", content="hi")])
    assert out["tool_calls"] == calls
    assert out["prompt_tokens"] == 10


# ---------------------------------------------------------------------------
# _retry_on_rate_limit
# ---------------------------------------------------------------------------

def test_retry_none_result_returns_empty():
    loop = _harness()
    assert loop._retry_on_rate_limit(lambda: None) == {}


def test_retry_connection_exhausted_raises_and_records():
    loop = _harness()
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise LLMConnectionError("conn fail")

    with mock.patch("external_llm.agent.agent_loop.time.sleep"), pytest.raises(LLMConnectionError):
        loop._retry_on_rate_limit(boom, "test mode")
    assert calls["n"] == 4  # initial + 3 backoff retries
    loop._record_llm_call_both.assert_called_once()
    assert loop._record_llm_call_both.call_args.kwargs["failed"] is True
    event, payload = loop._cb.call_args.args
    assert event == "error"
    assert payload["error_type"] == "connection"


def test_retry_rate_limit_recovers():
    loop = _harness()
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise LLMRateLimitError("slow down")
        return {"prompt_tokens": 1}

    with mock.patch("external_llm.agent.agent_loop.time.sleep"):
        out = loop._retry_on_rate_limit(flaky)
    assert calls["n"] == 2
    assert out["prompt_tokens"] == 1


def test_retry_server_unavailable_recovers():
    loop = _harness()
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise LLMServerUnavailableError("down")
        return {"prompt_tokens": 2}

    with mock.patch("external_llm.agent.agent_loop.time.sleep"):
        out = loop._retry_on_rate_limit(flaky)
    assert calls["n"] == 2
    assert out["prompt_tokens"] == 2


def test_retry_llm_cancelled_surfaces_as_agent_cancelled():
    loop = _harness()

    def boom():
        raise LLMCancelled("interrupted")

    with pytest.raises(AgentCancelled):
        loop._retry_on_rate_limit(boom)


def test_retry_cancel_event_set_before_loop():
    ev = threading.Event()
    ev.set()
    loop = _harness(cancel_event=ev)
    with pytest.raises(AgentCancelled):
        loop._retry_on_rate_limit(lambda: {"prompt_tokens": 1})


def test_retry_cancel_event_set_during_wait():
    ev = threading.Event()
    loop = _harness(cancel_event=ev)

    def boom():
        ev.set()
        raise LLMConnectionError("x")

    with pytest.raises(AgentCancelled):
        loop._retry_on_rate_limit(boom)


def test_retry_cancel_event_wait_path():
    ev = _FastEvent()
    loop = _harness(cancel_event=ev)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise LLMConnectionError("x")
        return {"prompt_tokens": 3}

    with mock.patch("external_llm.agent.agent_loop.time.sleep"):
        out = loop._retry_on_rate_limit(flaky)
    assert calls["n"] == 2
    assert out["prompt_tokens"] == 3


def test_retry_nan_token_values_tolerated():
    loop = _harness()
    out = loop._retry_on_rate_limit(lambda: {"prompt_tokens": float("nan"), "completion_tokens": 5})
    assert out["completion_tokens"] == 5


def test_retry_reasoning_tokens_logged():
    loop = _harness()
    out = loop._retry_on_rate_limit(
        lambda: {"prompt_tokens": 100, "completion_tokens": 50, "reasoning_tokens": 20}
    )
    assert out["prompt_tokens"] == 100
    assert out["completion_tokens"] == 50


# ---------------------------------------------------------------------------
# _auto_repair_apply_patch_args
# ---------------------------------------------------------------------------

def test_auto_repair_non_string_patch():
    loop = _harness()
    assert loop._auto_repair_apply_patch_args({"patch": 42}) is None
    assert loop._auto_repair_apply_patch_args({"patch": "   "}) is None


def test_auto_repair_hunk_only_without_path():
    loop = _harness()
    assert loop._auto_repair_apply_patch_args({"patch": "@@ -1 +1 @@\n+x\n"}) is None


def test_auto_repair_hunk_only_invalid_path():
    loop = _harness()
    assert loop._auto_repair_apply_patch_args({"patch": "@@ -1 +1 @@\n+x\n", "path": "../../evil"}) is None


def test_auto_repair_hunk_only_wraps_headers():
    loop = _harness()
    out = loop._auto_repair_apply_patch_args({"patch": "@@ -1 +1 @@\n+x\n", "path": "src/a.py"})
    assert out is not None
    assert out["patch"].startswith("diff --git a/src/a.py b/src/a.py\n--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n+x\n")


def test_auto_repair_missing_diffgit_header_added():
    loop = _harness()
    patch = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-x\n+y\n"
    out = loop._auto_repair_apply_patch_args({"patch": patch})
    assert out is not None
    assert out["patch"].startswith("diff --git a/f.py b/f.py\n")


def test_auto_repair_mismatched_paths_no_repair():
    loop = _harness()
    patch = "--- a/f.py\n+++ b/g.py\n@@ -1 +1 @@\n-x\n+y\n"
    assert loop._auto_repair_apply_patch_args({"patch": patch}) is None


def test_auto_repair_crlf_normalized():
    loop = _harness()
    patch = "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\r\n-x\r\n+y\r\n"
    out = loop._auto_repair_apply_patch_args({"patch": patch})
    assert out is not None
    assert "\r\n" not in out["patch"]


def test_auto_repair_nothing_applicable():
    loop = _harness()
    assert loop._auto_repair_apply_patch_args({"patch": "plain text"}) is None


# ---------------------------------------------------------------------------
# Guidance appenders
# ---------------------------------------------------------------------------

def test_write_plan_guidance_ok_result_unchanged():
    loop = _harness()
    content = loop._append_write_plan_guidance("base", "write_plan", ToolResult(ok=True, content="c"))
    assert content == "base"


def test_write_plan_guidance_block_not_found():
    loop = _harness()
    res = ToolResult(ok=False, error="BLOCK NOT FOUND near line 12")
    content = loop._append_write_plan_guidance("base", "write_plan", res)
    assert "BLOCK NOT FOUND: Use find_symbol" in content


def test_write_plan_guidance_anchor_error():
    loop = _harness()
    res = ToolResult(ok=False, error="anchor 'x' not found")
    content = loop._append_write_plan_guidance("base", "write_plan", res)
    assert "ANCHOR ERROR" in content


def test_write_plan_guidance_ambiguous_match():
    loop = _harness()
    res = ToolResult(ok=False, error="block match count is not 1")
    content = loop._append_write_plan_guidance("base", "write_plan", res)
    assert "AMBIGUOUS MATCH" in content


def test_write_plan_guidance_placeholder():
    loop = _harness()
    res = ToolResult(ok=False, error="contains placeholder text")
    content = loop._append_write_plan_guidance("base", "write_plan", res)
    assert "PLACEHOLDER" in content


def test_write_plan_guidance_generic_advice():
    loop = _harness()
    res = ToolResult(ok=False, error="some other failure")
    content = loop._append_write_plan_guidance("base", "write_plan", res)
    assert "[RECOVERY] Read the file first" in content


def test_edit_warnings_guidance_tip():
    loop = _harness()
    res = ToolResult(ok=False, error="e", metadata={"edit_warnings": ["op-type mismatch in hunk 2"]})
    content = loop._append_edit_warnings_guidance("base", "edit_text", res)
    assert "[EDIT FILE WARNINGS]" in content
    assert "insert_after" in content


def test_edit_warnings_guidance_syntax_check():
    loop = _harness()
    res = ToolResult(
        ok=False, error="e",
        metadata={"syntax_check": {"ok": False, "skipped": False, "errors": [{"line": 3, "col": 4, "message": "bad indent"}]}},
    )
    content = loop._append_edit_warnings_guidance("base", "edit_text", res)
    assert "[SYNTAX WARNING]" in content
    assert "line 3:4" in content


def test_edit_warnings_guidance_empty_metadata_unchanged():
    loop = _harness()
    content = loop._append_edit_warnings_guidance("base", "edit_text", ToolResult(ok=True, content="c"))
    assert content == "base"


def test_semantic_diagnostics_syntax_path():
    loop = _harness()
    res = ToolResult(
        ok=False, error="e",
        metadata={"syntax_check": {"semantic_diagnostics": [{"file_path": "a.py", "line": 1, "message": "undefined name 'x'", "severity": "error"}]}},
    )
    content = loop._append_semantic_diagnostics("base", res)
    assert "<file_diagnostics>" in content
    assert "undefined name 'x'" in content


def test_semantic_diagnostics_semantic_report_path():
    loop = _harness()
    res = ToolResult(
        ok=False, error="e",
        metadata={"semantic_report": {"diagnostics": [{"file_path": "b.py", "line": 2, "message": "type mismatch", "severity": "warning"}]}},
    )
    content = loop._append_semantic_diagnostics("base", res)
    assert "<file_diagnostics>" in content
    assert "type mismatch" in content


def test_semantic_diagnostics_none_returns_unchanged():
    loop = _harness()
    content = loop._append_semantic_diagnostics("base", ToolResult(ok=True, content="c"))
    assert content == "base"


def test_patch_retry_guidance():
    loop = _harness()
    res = ToolResult(
        ok=False, error="e",
        metadata={
            "retry_guidance": {
                "failure_type": "ctx", "target_file": "a.py", "hint": "h",
                "instruction": "i", "exact_existing_snippet": "code = 1",
            }
        },
    )
    content = loop._append_patch_retry_guidance("base", "apply_patch", res)
    assert "[PATCH RETRY GUIDANCE]" in content
    assert "Failure type: ctx" in content
    assert "Exact existing code/snippet:" in content
    assert "```\ncode = 1\n```" in content


def test_patch_retry_guidance_ignored_for_success():
    loop = _harness()
    res = ToolResult(ok=True, content="c", metadata={"retry_guidance": {"hint": "x"}})
    content = loop._append_patch_retry_guidance("base", "apply_patch", res)
    assert content == "base"


# ---------------------------------------------------------------------------
# _append_native_tool_messages
# ---------------------------------------------------------------------------

def test_native_openai_reasoning_content_and_tool_msgs():
    loop = _harness()
    loop.llm_client.get_provider_name.return_value = "openai"
    raw = SimpleNamespace(
        raw_response={
            "choices": [{"message": {"tool_calls": [{"id": "c1", "function": {"name": "read_file"}}], "reasoning_content": "thinking"}}]
        }
    )
    resp = {"content": "hi", "raw": raw}
    tool_msgs = [LLMMessage(role="tool", content="res", tool_call_id="c1")]
    msgs = loop._append_native_tool_messages([], resp, tool_msgs)
    assert msgs[0].role == "assistant"
    assert msgs[0].reasoning_content == "thinking"
    assert msgs[0].tool_calls == [{"id": "c1", "function": {"name": "read_file"}}]
    assert msgs[1] is tool_msgs[0]


def test_native_openai_all_tool_calls_filtered():
    loop = _harness()
    loop.llm_client.get_provider_name.return_value = "openai"
    raw = SimpleNamespace(
        raw_response={"choices": [{"message": {"tool_calls": [{"id": "c9"}], "reasoning_content": None}}]}
    )
    resp = {"content": "hi", "raw": raw}
    tool_msgs = [LLMMessage(role="tool", content="r", tool_call_id="c1")]
    extra = [LLMMessage(role="user", content="strategy warning")]
    orig = LLMMessage(role="user", content="orig")
    msgs = loop._append_native_tool_messages([orig], resp, tool_msgs + extra)
    assert msgs == [orig, *extra]  # assistant+tool block skipped entirely


def test_native_ollama_format():
    loop = _harness()
    loop.llm_client.get_provider_name.return_value = "ollama"
    resp = {"content": "hi", "tool_calls": [{"name": "read_file", "args": {"p": "a.py"}}]}
    tool_msgs = [LLMMessage(role="tool", content="r")]
    extra = [LLMMessage(role="user", content="warn")]
    msgs = loop._append_native_tool_messages([], resp, tool_msgs + extra)
    assert msgs[0].role == "assistant"
    assert msgs[0].tool_calls[0]["function"]["name"] == "read_file"
    assert msgs[1] is tool_msgs[0]
    assert msgs[2] is extra[0]


def test_native_generic_fallback_folds_text():
    loop = _harness()
    loop.llm_client.get_provider_name.return_value = "generic"
    tool_msgs = [LLMMessage(role="tool", content="r1"), LLMMessage(role="tool", content="r2")]
    extra = [LLMMessage(role="user", content="warn")]
    msgs = loop._append_native_tool_messages([], {"content": "hi"}, tool_msgs + extra)
    assert msgs[0].role == "assistant"
    assert msgs[1].role == "user"
    assert msgs[1].content.startswith("r1\n\nr2")
    assert "warn" in msgs[1].content
    assert "Continue with the task." in msgs[1].content


# ---------------------------------------------------------------------------
# _hunk_to_before_after
# ---------------------------------------------------------------------------

def test_hunk_to_before_after_splits():
    loop = _harness()
    lines = [" context", "-old", "+new", "", "\\ No newline at end of file\n"]
    before, after = loop._hunk_to_before_after(lines)
    assert before == "context\nold"
    assert after == "context\nnew"


def test_hunk_to_before_after_all_empty():
    loop = _harness()
    assert loop._hunk_to_before_after(["@@ -1 +1 @@\n", "\\ No newline\n"]) == (None, None)
