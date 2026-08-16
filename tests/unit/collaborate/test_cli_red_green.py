"""RED→GREEN: external_llm/repl/collaborate/cli.py — 0% → 100% coverage.

Covers:
- main(): argparse wiring, -v logging level, command routing, error exits
- _run_collaborate(): registry/display/orchestrator wiring, quiet verdict
  printing, KeyboardInterrupt → 130, generic error → 1, stop() in finally
- _run_async(): run() kwargs, CancelledError → interrupt() + re-raise,
  interrupt failure → debug log
- _run_mcp(): list rendering (params/no-params/description truncation),
  start mode kwargs + unrestricted_read trust
- __main__ guard (runpy, behavior only — excluded from coverage report)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import runpy
import sys
from typing import ClassVar

import pytest

import external_llm.agent.tool_registry as tool_registry_mod
import external_llm.editor.agent.mcp as mcp_mod
import external_llm.repl.collaborate as collaborate_mod
import external_llm.repl.collaborate.streaming_display as display_mod
from external_llm.repl.collaborate import cli

# ── fakes ───────────────────────────────────────────────────────────────

class _FakeAgentConfig:
    """Records kwargs (unrestricted_read=True) — no real tool policy init."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeToolRegistry:
    """Records construction args — real ToolRegistry scans the repo graph."""

    last_instance: "_FakeToolRegistry | None" = None

    def __init__(self, repo_root=None, config=None):
        self.repo_root = repo_root
        self.config = config
        _FakeToolRegistry.last_instance = self


class _FakeDisplay:
    """Records StreamingDisplay wiring — real one prints ANSI/uses threads."""

    last_instance: "_FakeDisplay | None" = None

    def __init__(self, verbose: bool = False, output_file=None):
        self.verbose = verbose
        self.output_file = output_file
        self.print_header_calls: list[tuple] = []
        self.stop_calls = 0
        self.summary_calls = 0
        self.flush_calls = 0
        _FakeDisplay.last_instance = self

    def print_header(self, task, model=None):
        self.print_header_calls.append((task, model))

    def handle_event(self, event):  # pragma: no cover - wiring only
        pass

    def print_summary(self):
        self.summary_calls += 1

    def flush_log(self):
        self.flush_calls += 1

    def stop(self):
        self.stop_calls += 1


class _FakeVerdict:
    def __init__(self, status="success", summary="done"):
        self.status = status
        self.summary = summary

    def to_dict(self):
        return {"status": self.status, "summary": self.summary}


class _FakeSessionResult:
    def __init__(self, verdict=None):
        self.verdict = verdict or _FakeVerdict()


class _FakeOrchestrator:
    """Async context-manager fake for CollaborationOrchestrator."""

    instances: ClassVar[list["_FakeOrchestrator"]] = []
    result = None
    run_error: BaseException | None = None
    interrupt_error: Exception | None = None

    def __init__(self, registry, config):
        self.registry = registry
        self.config = config
        self.interrupt_calls = 0
        self.run_kwargs = None
        _FakeOrchestrator.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return None

    async def run(self, **kwargs):
        self.run_kwargs = kwargs
        if _FakeOrchestrator.run_error is not None:
            raise _FakeOrchestrator.run_error
        return _FakeOrchestrator.result

    async def interrupt(self):
        self.interrupt_calls += 1
        if _FakeOrchestrator.interrupt_error is not None:
            raise _FakeOrchestrator.interrupt_error


@pytest.fixture(autouse=True)
def _patch_external_deps(monkeypatch):
    """Route all function-level imports of cli.py to fakes."""
    monkeypatch.setattr(tool_registry_mod, "AgentConfig", _FakeAgentConfig)
    monkeypatch.setattr(tool_registry_mod, "ToolRegistry", _FakeToolRegistry)
    monkeypatch.setattr(display_mod, "StreamingDisplay", _FakeDisplay)
    monkeypatch.setattr(collaborate_mod, "CollaborationOrchestrator", _FakeOrchestrator)
    _FakeToolRegistry.last_instance = None
    _FakeDisplay.last_instance = None
    _FakeOrchestrator.instances = []
    _FakeOrchestrator.result = None
    _FakeOrchestrator.run_error = None
    _FakeOrchestrator.interrupt_error = None
    yield


def _collab_args(**overrides):
    """Namespace matching what argparse produces for the collaborate subcommand."""
    base = {
        "command": "collaborate",
        "task": "t",
        "context": None,
        "verbose": False,
        "file": None,
        "quiet": False,
        "max_turns": 100,
        "model": None,
        "no_digest": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


# ── main() ──────────────────────────────────────────────────────────────

def test_main_collaborate_routes_all_flags(monkeypatch):
    captured = []
    monkeypatch.setattr(cli, "_run_collaborate", lambda args: captured.append(args))
    monkeypatch.setattr(
        sys, "argv",
        ["cli", "collaborate", "--task", "hello", "--context", "ctx",
         "--model", "claude-sonnet-5", "--max-turns", "7", "--no-digest",
         "--file", "out.log", "--quiet"],
    )
    cli.main()
    assert len(captured) == 1
    a = captured[0]
    assert a.task == "hello"
    assert a.context == "ctx"
    assert a.model == "claude-sonnet-5"
    assert a.max_turns == 7
    assert a.no_digest is True
    assert a.file == "out.log"
    assert a.quiet is True
    assert a.verbose is False


def test_main_collaborate_short_flags(monkeypatch):
    captured = []
    monkeypatch.setattr(cli, "_run_collaborate", lambda args: captured.append(args))
    monkeypatch.setattr(
        sys, "argv",
        ["cli", "-v", "collaborate", "-t", "task", "-c", "ctx", "-m", "m", "-f", "f"],
    )
    cli.main()
    a = captured[0]
    assert a.verbose is True
    assert a.task == "task" and a.context == "ctx" and a.model == "m" and a.file == "f"


def test_main_verbose_sets_debug_logging(monkeypatch):
    calls = []
    monkeypatch.setattr(logging, "basicConfig", lambda **kw: calls.append(kw))
    monkeypatch.setattr(cli, "_run_collaborate", lambda args: None)
    monkeypatch.setattr(sys, "argv", ["cli", "-v", "collaborate", "--task", "t"])
    cli.main()
    assert calls and calls[0]["level"] == logging.DEBUG


def test_main_non_verbose_sets_info_logging(monkeypatch):
    calls = []
    monkeypatch.setattr(logging, "basicConfig", lambda **kw: calls.append(kw))
    monkeypatch.setattr(cli, "_run_collaborate", lambda args: None)
    monkeypatch.setattr(sys, "argv", ["cli", "collaborate", "--task", "t"])
    cli.main()
    assert calls and calls[0]["level"] == logging.INFO


def test_main_mcp_list_routes(monkeypatch):
    captured = []
    monkeypatch.setattr(cli, "_run_mcp", lambda args: captured.append(args))
    monkeypatch.setattr(sys, "argv", ["cli", "mcp", "list"])
    cli.main()
    assert captured[0].mcp_command == "list"


def test_main_mcp_start_routes_flags(monkeypatch):
    captured = []
    monkeypatch.setattr(cli, "_run_mcp", lambda args: captured.append(args))
    monkeypatch.setattr(
        sys, "argv",
        ["cli", "mcp", "start", "--mode", "http", "--port", "9000", "--host", "0.0.0.0"],
    )
    cli.main()
    a = captured[0]
    assert a.mcp_command == "start"
    assert a.mode == "http" and a.port == 9000 and a.host == "0.0.0.0"


def test_main_mcp_verbose_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(logging, "basicConfig", lambda **kw: calls.append(kw))
    monkeypatch.setattr(cli, "_run_mcp", lambda args: None)
    monkeypatch.setattr(sys, "argv", ["cli", "mcp", "-v", "list"])
    cli.main()
    assert calls and calls[0]["level"] == logging.DEBUG


def test_main_requires_subcommand(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["cli"])
    with pytest.raises(SystemExit) as ei:
        cli.main()
    assert ei.value.code == 2


def test_main_requires_task_for_collaborate(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["cli", "collaborate"])
    with pytest.raises(SystemExit) as ei:
        cli.main()
    assert ei.value.code == 2


def test_main_rejects_unknown_subcommand(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["cli", "bogus"])
    with pytest.raises(SystemExit) as ei:
        cli.main()
    assert ei.value.code == 2


# ── _run_collaborate ────────────────────────────────────────────────────

def test_run_collaborate_success_quiet(capsys):
    _FakeOrchestrator.result = _FakeSessionResult(_FakeVerdict(status="success", summary="done"))
    cli._run_collaborate(_collab_args(quiet=True))

    reg = _FakeToolRegistry.last_instance
    assert reg.repo_root == os.getcwd()
    assert reg.config.kwargs == {"unrestricted_read": True}

    display = _FakeDisplay.last_instance
    assert display.verbose is False
    assert display.output_file is None
    # DEFAULT_COLLAB_MODEL fallback when --model is absent
    assert display.print_header_calls == [("t", "sonnet")]
    assert display.summary_calls == 1
    assert display.flush_calls == 1
    assert display.stop_calls == 1

    orch = _FakeOrchestrator.instances[-1]
    assert orch.config.max_turns_per_iteration == 100
    assert orch.config.model == "sonnet"
    assert orch.config.repo_root == os.getcwd()
    assert orch.config.event_callback is None  # quiet → no streaming callback
    assert orch.run_kwargs == {
        "task": "t",
        "context": None,
        "enable_preprocessing": True,
    }

    out = capsys.readouterr().out
    assert "{'status': 'success', 'summary': 'done'}" in out  # verdict for scripting


def test_run_collaborate_success_streaming(capsys):
    _FakeOrchestrator.result = _FakeSessionResult()
    cli._run_collaborate(_collab_args(
        quiet=False, model="claude-opus-4-5", max_turns=3,
        no_digest=True, context="ctx", file="session.log",
    ))
    display = _FakeDisplay.last_instance
    assert display.verbose is False
    assert display.output_file == "session.log"
    assert display.print_header_calls == [("t", "claude-opus-4-5")]
    orch = _FakeOrchestrator.instances[-1]
    assert orch.config.max_turns_per_iteration == 3
    assert orch.config.model == "claude-opus-4-5"
    assert orch.config.event_callback is not None
    assert orch.config.event_callback.__self__ is display  # display.handle_event bound
    assert orch.config.event_callback.__func__ is _FakeDisplay.handle_event
    assert orch.run_kwargs == {
        "task": "t", "context": "ctx", "enable_preprocessing": False,
    }
    assert capsys.readouterr().out == ""  # non-quiet: no verdict dict on stdout


def test_run_collaborate_verbose_display(monkeypatch):
    _FakeOrchestrator.result = _FakeSessionResult()
    cli._run_collaborate(_collab_args(verbose=True))
    assert _FakeDisplay.last_instance.verbose is True


def test_run_collaborate_keyboard_interrupt_exits_130(monkeypatch, capsys):
    def _raise_ki(*args, **kwargs):
        raise KeyboardInterrupt

    # Patch the coroutine factory, not asyncio.run: a real `_run_async(...)`
    # coroutine created as the argument to a patched-async run() would be
    # abandoned un-awaited (RuntimeWarning at gc).
    monkeypatch.setattr(cli, "_run_async", _raise_ki)
    with pytest.raises(SystemExit) as ei:
        cli._run_collaborate(_collab_args())
    assert ei.value.code == 130
    assert "Interrupted by user" in capsys.readouterr().err
    assert _FakeDisplay.last_instance.stop_calls == 1  # finally cleanup


def test_run_collaborate_generic_error_exits_1(monkeypatch, capsys, caplog):
    def _raise_boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "_run_async", _raise_boom)
    with caplog.at_level(logging.ERROR, logger="external_llm.repl.collaborate.cli"), \
            pytest.raises(SystemExit) as ei:
        cli._run_collaborate(_collab_args())
    assert ei.value.code == 1
    assert "Error: boom" in capsys.readouterr().err
    assert "Collaboration failed" in caplog.text
    assert _FakeDisplay.last_instance.stop_calls == 1


# ── _run_async ──────────────────────────────────────────────────────────

def test_run_async_forwards_kwargs():
    _FakeOrchestrator.result = _FakeSessionResult()
    out = asyncio.run(cli._run_async(None, None, _collab_args(no_digest=True)))
    assert out is _FakeOrchestrator.result
    orch = _FakeOrchestrator.instances[-1]
    assert orch.run_kwargs == {
        "task": "t", "context": None, "enable_preprocessing": False,
    }


def test_run_async_cancelled_interrupts_orchestrator():
    _FakeOrchestrator.run_error = asyncio.CancelledError()
    try:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(cli._run_async(None, None, _collab_args()))
        orch = _FakeOrchestrator.instances[-1]
        assert orch.interrupt_calls == 1
    finally:
        _FakeOrchestrator.run_error = None


def test_run_async_cancel_interrupt_failure_logged(caplog):
    _FakeOrchestrator.run_error = asyncio.CancelledError()
    _FakeOrchestrator.interrupt_error = RuntimeError("interrupt failed")
    try:
        with caplog.at_level(logging.DEBUG, logger="external_llm.repl.collaborate.cli"), \
                pytest.raises(asyncio.CancelledError):
            asyncio.run(cli._run_async(None, None, _collab_args()))
        assert "Interrupt on cancel failed" in caplog.text
        assert _FakeOrchestrator.instances[-1].interrupt_calls == 1
    finally:
        _FakeOrchestrator.run_error = None
        _FakeOrchestrator.interrupt_error = None


# ── _run_mcp ────────────────────────────────────────────────────────────

def test_run_mcp_list_renders_tools(monkeypatch, capsys):
    tools = [
        {"name": "read_file", "description": "Read a file.",
         "parameters": {"properties": {"path": {"type": "string"}}}},
        {"name": "no_params", "description": "Takes nothing.",
         "parameters": {"properties": {}}},
        {"name": "long_desc", "description": "x" * 200,
         "parameters": {"properties": {}}},
    ]
    monkeypatch.setattr(mcp_mod, "list_mcp_tools", lambda: tools)
    cli._run_mcp(argparse.Namespace(mcp_command="list"))
    out = capsys.readouterr().out
    assert "asicode MCP Tools (3 total):" in out
    assert "read_file(path)" in out
    assert "no_params(no params)" in out
    assert ("x" * 100) in out
    assert ("x" * 101) not in out


def test_run_mcp_start_wires_registry_and_mode(monkeypatch, capsys):
    captured = {}

    def _fake_server(registry, mode, host, port):
        captured.update(registry=registry, mode=mode, host=host, port=port)

    monkeypatch.setattr(mcp_mod, "run_mcp_server", _fake_server)
    cli._run_mcp(argparse.Namespace(mcp_command="start", mode="sse", host="0.0.0.0", port=9999))
    assert captured["mode"] == "sse"
    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 9999
    reg = captured["registry"]
    assert reg is _FakeToolRegistry.last_instance
    assert reg.repo_root == os.getcwd()
    assert reg.config.kwargs == {"unrestricted_read": True}
    assert "Starting asicode MCP server (sse mode)..." in capsys.readouterr().out


# ── __main__ guard (behavior — excluded from coverage report) ───────────

def test_main_guard_runs_help(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["cli", "--help"])
    with pytest.raises(SystemExit) as ei:
        runpy.run_path(cli.__file__, run_name="__main__")
    assert ei.value.code == 0
    assert "collaborate" in capsys.readouterr().out
