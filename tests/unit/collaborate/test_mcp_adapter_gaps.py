"""Targeted gap tests for asi_mcp_adapter.

Covers the previously-missed branches:
  - build_collaborate_install_spec: malformed direct_url.json / non-file URL /
    scan-failure fallbacks (lines 105-107, 120-121)
  - _get_tool_annotations: SDK-absent fallback + neutral-tool None (310-313, 329)
  - build_asr_mcp_server: read-only skip of unclassified tools (393-397) and
    schema-without-handler skip (404-408)
  - _make_async_handler: dispatch error path (497), timeout path (511-512)
  - get_restricted_options: options assembly, model override, mode appends (553-567, 607-678)

These run against the real SDK when installed (skipif guards the SDK-only ones).
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from types import SimpleNamespace

import pytest

from external_llm.agent.tool_registry import AgentConfig, ToolRegistry
from external_llm.repl.collaborate.asi_mcp_adapter import (
    _get_tool_annotations,
    _make_async_handler,
    build_asr_mcp_server,
    build_collaborate_install_spec,
    get_restricted_options,
)

HAVE_SDK = importlib.util.find_spec("claude_agent_sdk") is not None


def _registry() -> ToolRegistry:
    return ToolRegistry(repo_root=".", config=AgentConfig())


class TestInstallSpecFallbacks:
    """malformed/non-editable/scan-failure fall back to the PyPI name."""

    def test_malformed_direct_url_json_falls_back(self, monkeypatch):
        import importlib.metadata as md

        class _BadDist:
            @property
            def metadata(self):
                return SimpleNamespace(get=lambda k, d="": "asicode" if k == "name" else d)

            def read_text(self, filename):
                return "{not valid json"

        monkeypatch.setattr(md, "distributions", lambda: [_BadDist()])
        assert build_collaborate_install_spec() == ["asicode[collaborate]"]

    def test_metadata_read_raises_falls_back(self, monkeypatch):
        import importlib.metadata as md

        class _RaisingDist:
            @property
            def metadata(self):
                return SimpleNamespace(get=lambda k, d="": "asicode" if k == "name" else d)

            def read_text(self, filename):
                raise OSError("unreadable metadata")

        monkeypatch.setattr(md, "distributions", lambda: [_RaisingDist()])
        assert build_collaborate_install_spec() == ["asicode[collaborate]"]

    def test_scan_exception_falls_back(self, monkeypatch):
        import importlib.metadata as md

        def _boom():
            raise RuntimeError("scan failed")

        monkeypatch.setattr(md, "distributions", _boom)
        assert build_collaborate_install_spec() == ["asicode[collaborate]"]

    def test_non_editable_url_falls_back(self, monkeypatch):
        import importlib.metadata as md

        class _WheelDist:
            @property
            def metadata(self):
                return SimpleNamespace(get=lambda k, d="": "asicode" if k == "name" else d)

            def read_text(self, filename):
                return '{"url": "https://files.pythonhosted.org/asicode.whl", "dir_info": {"editable": false}}'

        monkeypatch.setattr(md, "distributions", lambda: [_WheelDist()])
        assert build_collaborate_install_spec() == ["asicode[collaborate]"]


@pytest.mark.skipif(not HAVE_SDK, reason="claude_agent_sdk not installed")
class TestToolAnnotations:
    """_get_tool_annotations returns None when SDK absent; neutral tools return None."""

    def test_sdk_absent_returns_none(self, monkeypatch):
        # Force the import inside _get_tool_annotations to fail.
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
        assert _get_tool_annotations("some_tool") is None

    def test_neutral_tool_returns_annotations_object(self):
        # With the SDK installed, every tool gets a ToolAnnotations instance;
        # the None branch (miss 329) is the SDK-absent fallback covered above.
        ann = _get_tool_annotations("__no_such_annotation_tool__")
        assert ann is not None
        assert ann.readOnlyHint is False
        assert ann.destructiveHint is False
        assert ann.openWorldHint is False


@pytest.mark.skipif(not HAVE_SDK, reason="claude_agent_sdk not installed")
class TestBuildAsrMcpServerSkips:
    """Read-only builds skip unclassified tools and schema-without-handler tools."""

    def test_read_only_skips_unclassified_tool(self, monkeypatch):
        # read_only=True: a tool that is neither read-only nor analysis-safe
        # must be skipped (not exposed to the agent). We simulate one by
        # removing a real tool from the analysis-safe set — the branch that
        # logs "skipping unclassified tool" then fires for it. Capture the
        # tools passed to create_sdk_mcp_server to observe the exposure.
        import claude_agent_sdk

        captured = {}

        def _fake_create(name, version, tools):
            captured["tools"] = tools
            return {"name": name, "version": version, "tools": tools, "type": "sdk"}

        monkeypatch.setattr(claude_agent_sdk, "create_sdk_mcp_server", _fake_create)

        victim = "bash"  # analysis-safe by default
        import external_llm.repl.collaborate.asi_mcp_adapter as mod

        monkeypatch.setattr(mod, "_ANALYSIS_SAFE_TOOLS", set())
        build_asr_mcp_server(_registry(), read_only=True)
        names = {t.name for t in captured["tools"]}
        assert victim not in names
        # The read-only tools themselves still pass through.
        assert "read_file" in names

    def test_schema_without_handler_warns_and_skips(self, monkeypatch):
        import claude_agent_sdk

        captured = {}

        def _fake_create(name, version, tools):
            captured["tools"] = tools
            return {"name": name, "version": version, "tools": tools, "type": "sdk"}

        monkeypatch.setattr(claude_agent_sdk, "create_sdk_mcp_server", _fake_create)

        registry = _registry()
        # A schema with no handler must be skipped with a warning. Mock
        # has_tool_handler to False for one tool to exercise the branch.
        real = registry.has_tool_handler
        calls = []

        def fake_has_handler(name):
            calls.append(name)
            # Pretend the first checked tool has no handler.
            if len(calls) == 1:
                return False
            return real(name)

        monkeypatch.setattr(registry, "has_tool_handler", fake_has_handler)
        build_asr_mcp_server(registry, read_only=True)
        # The fake-removed tool is not exposed.
        first = calls[0]
        names = {t.name for t in captured["tools"]}
        assert first not in names


class TestAsyncHandlerErrorPaths:
    """_make_async_handler dispatch error and timeout paths."""

    def test_dispatch_error_returns_error_content(self):
        from external_llm.agent.tool_registry import ToolResult

        registry = _registry()
        result_obj = ToolResult(ok=False, error="simulated failure")
        registry.dispatch = lambda name, args: result_obj

        handler = _make_async_handler(registry, "get_project_info")
        out = asyncio.run(handler({}))
        assert out["isError"] is True
        assert "simulated failure" in out["content"][0]["text"]

    def test_timeout_returns_timeout_content(self, monkeypatch):
        import external_llm.repl.collaborate.asi_mcp_adapter as mod

        registry = _registry()
        # Force a sub-second timeout by resolving to a tiny ceiling.
        monkeypatch.setattr(
            mod,
            "_resolve_mcp_timeout",
            lambda tool, args: 0.01,
        )

        # A handler that blocks longer than the ceiling.
        def slow_dispatch(name, args):
            import time

            time.sleep(0.2)
            return SimpleNamespace(ok=True, content="too late")

        registry.dispatch = slow_dispatch
        handler = _make_async_handler(registry, "get_project_info")
        out = asyncio.run(handler({}))
        assert out["isError"] is True
        assert "TOOL_TIMEOUT" in out["content"][0]["text"]

    def test_dispatch_exception_returns_exception_content(self):
        # A dispatch that raises must surface 'EXCEPTION: ...' (the
        # except Exception branch in the handler) and never crash the
        # MCP tool call.
        registry = _registry()

        def raising_dispatch(name, args):
            raise RuntimeError("boom")

        registry.dispatch = raising_dispatch
        handler = _make_async_handler(registry, "get_project_info")
        out = asyncio.run(handler({}))
        assert out["isError"] is True
        assert "EXCEPTION: boom" in out["content"][0]["text"]

    def test_dispatch_ok_returns_success_content(self):
        # Success path: content passes through without isError.
        from external_llm.agent.tool_registry import ToolResult

        registry = _registry()
        registry.dispatch = lambda name, args: ToolResult(ok=True, content="hello")
        handler = _make_async_handler(registry, "get_project_info")
        out = asyncio.run(handler({}))
        assert out.get("isError") is None
        assert out["content"][0]["text"] == "hello"


@pytest.mark.skipif(not HAVE_SDK, reason="claude_agent_sdk not installed")
class TestRestrictedOptionsSdk:
    """get_restricted_options assembles real ClaudeAgentOptions."""

    def _opts(self, **kw):
        return get_restricted_options({"name": "asi"}, **kw)

    def test_defaults_restrict_native_tools(self):
        opts = self._opts()
        assert opts.allowed_tools == ["mcp__asr__*"]
        assert "Read" in opts.disallowed_tools
        assert "Write" in opts.disallowed_tools
        assert "Bash" in opts.disallowed_tools
        assert opts.setting_sources == []
        assert opts.system_prompt["exclude_dynamic_sections"] is True

    def test_model_override(self):
        assert self._opts(model="opus").model == "opus"

    def test_write_mode_appends_analysis_only(self):
        read_opts = self._opts()
        write_opts = self._opts(allow_write=True)
        assert read_opts.system_prompt["append"] != write_opts.system_prompt["append"]
        assert "read-only" in read_opts.system_prompt["append"].lower()

    def test_custom_system_prompt_appended(self):
        opts = self._opts(system_prompt="CUSTOM PROMPT 42")
        assert "CUSTOM PROMPT 42" in opts.system_prompt["append"]

    def test_output_format_json_schema(self):
        opts = self._opts()
        assert opts.output_format["type"] == "json_schema"
        assert "schema" in opts.output_format
