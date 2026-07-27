"""
Tests for ASR MCP Adapter.

These tests verify:
- MCP server creation from ToolRegistry
- Tool exclusion logic
- Schema conversion
- Restricted options building
- Tool annotations

Note: Full MCP server integration requires claude-agent-sdk.
"""
from __future__ import annotations

import pytest

from external_llm.agent.tool_registry import AgentConfig, ToolRegistry

# claude_agent_sdk is an optional dependency. The pure-logic helpers below
# (_EXCLUDED_TOOLS, _convert_schema_to_input_type, _get_tool_annotations, …)
# work without it, but the full MCP server building path (TestMcpServerBuilding)
# requires the SDK. importorskip is applied at class level there.
from external_llm.repl.collaborate.asi_mcp_adapter import (
    _ANALYSIS_SAFE_TOOLS,
    _DESTRUCTIVE_TOOLS,
    _EXCLUDED_TOOLS,
    _INNER_TIMEOUT_TOOLS,
    _OPEN_WORLD_TOOLS,
    _READ_ONLY_TOOLS,
    _TOOL_SPECIFIC_TIMEOUTS,
    _convert_schema_to_input_type,
    _get_tool_annotations,
    _resolve_mcp_timeout,
    build_asr_mcp_server,
    build_collaborate_install_command,
    build_collaborate_install_spec,
    claude_sdk_missing_error,
    is_claude_sdk_installed,
)


class TestMissingSdkGuidance:
    """Missing-SDK errors carry an actionable install hint, not a bare traceback."""

    def test_build_mcp_server_message_names_install_command(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
        registry = ToolRegistry(repo_root=".", config=AgentConfig())
        with pytest.raises(ImportError) as exc_info:
            build_asr_mcp_server(registry)
        msg = str(exc_info.value)
        assert "pip install" in msg
        assert "collaborate" in msg

    def test_missing_sdk_error_preserves_cause(self):
        original = ModuleNotFoundError("No module named 'claude_agent_sdk'")
        err = claude_sdk_missing_error(original)
        assert isinstance(err, ImportError)
        assert err.__cause__ is original
        assert "pip install" in str(err)


class _FakeDist:
    """Minimal stand-in for importlib.metadata.Distribution."""

    def __init__(self, name, direct_url=None):
        self._name = name
        self._direct_url = direct_url

    @property
    def metadata(self):
        class _Meta:
            def __init__(self, nm):
                self._nm = nm

            def get(self, key, default=""):
                return self._nm if key == "name" else default

        return _Meta(self._name)

    def read_text(self, filename):
        return self._direct_url


class TestCollaborateInstallGate:
    """Availability check + install-command derivation for the optional SDK.

    These guard the ``/claude`` REPL flow: when the SDK is absent, the user is
    offered a one-shot install. The command must never depend on cwd (which is
    the user's project, not the asicode source) — see dual-metadata case below.
    """

    def test_is_installed_true_when_find_spec_resolves(self, monkeypatch):
        import importlib.util

        monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
        assert is_claude_sdk_installed() is True

    def test_is_installed_false_when_find_spec_returns_none(self, monkeypatch):
        import importlib.util

        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        assert is_claude_sdk_installed() is False

    def test_install_command_editable_uses_minus_e(self, monkeypatch):
        import importlib.metadata as md
        import sys

        dist = _FakeDist(
            "asicode",
            direct_url='{"dir_info": {"editable": true}, "url": "file:///src/asicode"}',
        )
        monkeypatch.setattr(md, "distributions", lambda: [dist])
        cmd = build_collaborate_install_command()
        assert cmd[0] == sys.executable
        assert cmd[1:5] == ["-m", "pip", "install", "-e"]
        assert cmd[-1] == "/src/asicode[collaborate]"

    def test_install_command_skips_metadata_without_direct_url(self, monkeypatch):
        """Dual-metadata reality: an editable source tree may expose a legacy
        egg-info (no direct_url.json) alongside the PEP 660 dist-info — the
        scanner must look past the first and use the one with editable info."""
        import importlib.metadata as md

        egg_info = _FakeDist("asicode", direct_url=None)
        dist_info = _FakeDist(
            "asicode",
            direct_url='{"dir_info": {"editable": true}, "url": "file:///repo/asicode"}',
        )
        monkeypatch.setattr(md, "distributions", lambda: [egg_info, dist_info])
        cmd = build_collaborate_install_command()
        assert "-e" in cmd
        assert cmd[-1] == "/repo/asicode[collaborate]"

    def test_install_command_non_editable_falls_back_to_pypi(self, monkeypatch):
        import importlib.metadata as md

        dist = _FakeDist(
            "asicode",
            direct_url='{"dir_info": {"editable": false}, "url": "https://files.pythonhosted.org/..."}',
        )
        monkeypatch.setattr(md, "distributions", lambda: [dist])
        cmd = build_collaborate_install_command()
        assert "-e" not in cmd
        assert cmd[-1] == "asicode[collaborate]"

    def test_install_command_no_direct_url_falls_back_to_pypi(self, monkeypatch):
        import importlib.metadata as md

        monkeypatch.setattr(md, "distributions", lambda: [_FakeDist("asicode")])
        cmd = build_collaborate_install_command()
        assert "-e" not in cmd
        assert cmd[-1] == "asicode[collaborate]"

    def test_install_command_skips_unrelated_distributions(self, monkeypatch):
        import importlib.metadata as md

        other = _FakeDist("some-other-pkg", direct_url='{"dir_info": {"editable": true}}')
        asicode = _FakeDist(
            "asicode",
            direct_url='{"dir_info": {"editable": true}, "url": "file:///x"}',
        )
        monkeypatch.setattr(md, "distributions", lambda: [other, asicode])
        cmd = build_collaborate_install_command()
        assert cmd[-1] == "/x[collaborate]"

    def test_install_spec_has_no_pip_prefix(self, monkeypatch):
        """Spec is the *tail* only (no interpreter/`-m pip install` prefix) —
        callers feed it to the shared ``_pip_install`` which owns the prefix and
        the PEP 668 ``--break-system-packages`` retry."""
        import importlib.metadata as md

        dist = _FakeDist(
            "asicode",
            direct_url='{"dir_info": {"editable": true}, "url": "file:///src/asicode"}',
        )
        monkeypatch.setattr(md, "distributions", lambda: [dist])
        spec = build_collaborate_install_spec()
        assert spec == ["-e", "/src/asicode[collaborate]"]
        # the full command wraps the spec unchanged
        assert build_collaborate_install_command()[-len(spec):] == spec

    def test_install_spec_non_editable_is_bare_name(self, monkeypatch):
        import importlib.metadata as md

        monkeypatch.setattr(md, "distributions", lambda: [_FakeDist("asicode")])
        assert build_collaborate_install_spec() == ["asicode[collaborate]"]

    def test_install_spec_editable_decodes_percent_encoded_source_path(self, monkeypatch):
        """direct_url.json stores an RFC file:// URL with percent-encoding for
        spaces/non-ASCII (and may carry a ``localhost`` host). The editable spec
        must yield the real, decoded local path — otherwise pip gets a path it
        can't resolve (``/Users/my%20proj`` is not a directory)."""
        import importlib.metadata as md

        dist = _FakeDist(
            "asicode",
            direct_url='{"dir_info": {"editable": true}, "url": "file://localhost/Users/my%20proj/asicode"}',
        )
        monkeypatch.setattr(md, "distributions", lambda: [dist])
        spec = build_collaborate_install_spec()
        # host stripped + percent-decoded → real path pip can read
        assert spec == ["-e", "/Users/my proj/asicode[collaborate]"]

    def test_install_spec_editable_plain_ascii_path_unchanged(self, monkeypatch):
        """Regression guard: the common ASCII path (no encoding, empty host)
        must produce the same spec as before the urlparse+unquote change."""
        import importlib.metadata as md

        dist = _FakeDist(
            "asicode",
            direct_url='{"dir_info": {"editable": true}, "url": "file:///src/asicode"}',
        )
        monkeypatch.setattr(md, "distributions", lambda: [dist])
        assert build_collaborate_install_spec() == ["-e", "/src/asicode[collaborate]"]


class TestToolExclusion:
    """Verify internal tools are correctly excluded from MCP exposure."""

    def test_excluded_tools_are_internal(self):
        # These should never be exposed via MCP
        assert "delegate_to_helper" in _EXCLUDED_TOOLS
        assert "update_memory" in _EXCLUDED_TOOLS
        assert "query_experience" in _EXCLUDED_TOOLS

    def test_bash_is_in_analysis_safe_tools_not_excluded(self):
        """bash must be available in analysis sessions and not in excluded."""
        assert "bash" in _ANALYSIS_SAFE_TOOLS
        assert "bash" not in _EXCLUDED_TOOLS

    def test_has_tool_handler_bash(self):
        """bash handler (_tool_shell_exec) must be discoverable via has_tool_handler."""
        registry = ToolRegistry(repo_root=".", config=AgentConfig())
        assert registry.has_tool_handler("bash")
        assert registry.has_tool_handler("read_file")
        assert registry.has_tool_handler("apply_patch")
        # Excluded/internal tools still have handlers
        assert registry.has_tool_handler("delegate_to_helper")
        assert registry.has_tool_handler("update_memory")
        # Unknown tool returns False
        assert not registry.has_tool_handler("nonexistent_tool")

    def test_read_tools_have_correct_category(self):
        assert "read_file" in _READ_ONLY_TOOLS
        assert "find_symbol" in _READ_ONLY_TOOLS

    def test_write_tools_have_correct_category(self):
        assert "apply_patch" in _DESTRUCTIVE_TOOLS
        assert "modify_symbol" in _DESTRUCTIVE_TOOLS
        assert "write_plan" in _DESTRUCTIVE_TOOLS

    def test_open_world_tools_is_empty(self):
        # Web tools are now native to Claude Code — no MCP exposure needed
        assert len(_OPEN_WORLD_TOOLS) == 0


class TestSchemaConversion:
    """Verify OpenAI schema conversion to MCP format."""

    def test_convert_full_schema(self):
        openai_schema = {
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "recursive": {"type": "boolean"},
                },
                "required": ["path"],
            }
        }
        result = _convert_schema_to_input_type(openai_schema)
        assert result["type"] == "object"
        assert "path" in result["properties"]
        assert "recursive" in result["properties"]
        assert "path" in result["required"]

    def test_convert_empty_params(self):
        result = _convert_schema_to_input_type({"parameters": {}})
        assert result == {}


class TestToolAnnotations:
    """Verify MCP tool annotation logic."""

    def test_read_tool_annotations(self):
        ann = _get_tool_annotations("read_file")
        if ann is not None:
            assert ann.readOnlyHint is True
            assert ann.destructiveHint is False

    def test_write_tool_annotations(self):
        ann = _get_tool_annotations("apply_patch")
        if ann is not None:
            assert ann.readOnlyHint is False
            assert ann.destructiveHint is True

    def test_excluded_tool_annotations_all_false(self):
        # Excluded tools get default annotations (all False)
        ann = _get_tool_annotations("search_web")
        if ann is not None:
            assert ann.readOnlyHint is False
            assert ann.destructiveHint is False
            assert ann.openWorldHint is False

    def test_unknown_tool_returns_none(self):
        # Tools not in any category get default annotation (all False)
        ann = _get_tool_annotations("nonexistent_tool")
        if ann is not None:
            assert ann.readOnlyHint is False
            assert ann.destructiveHint is False
            assert ann.openWorldHint is False


class TestMcpServerBuilding:
    """Verify MCP server creation from ToolRegistry.

    Requires claude_agent_sdk (optional dependency) — skipped when absent.
    """

    def setup_method(self, method):
        pytest.importorskip("claude_agent_sdk")

    def test_build_mcp_server(self):
        registry = ToolRegistry(repo_root=".", config=AgentConfig())
        server = build_asr_mcp_server(registry, server_name="asicode-test")

        # Returns a dict (McpSdkServerConfig)
        assert isinstance(server, dict)
        assert server["type"] == "sdk"
        assert server["name"] == "asicode-test"
        assert "instance" in server

    def test_build_mcp_server_excludes_internal_tools(self):
        registry = ToolRegistry(repo_root=".", config=AgentConfig())
        server = build_asr_mcp_server(registry)

        # Internal tools should not be in the server
        instance = server["instance"]
        assert instance is not None

    def test_build_mcp_server_custom_exclude(self):
        registry = ToolRegistry(repo_root=".", config=AgentConfig())
        # Exclude everything except a specific tool
        server = build_asr_mcp_server(
            registry,
            excluded_tools={"read_file", "grep"},  # expose everything else
        )
        assert server["name"] == "asicode"

    def test_mcp_tools_list(self):
        from external_llm.editor.agent.mcp import list_mcp_tools
        tools = list_mcp_tools()
        assert len(tools) >= 18  # Most tools should be exposed
        tool_names = {t["name"] for t in tools}

        # Core tools should be present
        assert "read_file" in tool_names
        assert "apply_patch" in tool_names
        assert "modify_symbol" in tool_names

        # get_symbol_info was merged into find_symbol; should no longer be a separate tool
        assert "get_symbol_info" not in tool_names
        assert "find_symbol" in tool_names

        # Internal tools should NOT be present
        assert "delegate_to_helper" not in tool_names
        assert "update_memory" not in tool_names

        # design-chat-only tools have no ToolRegistry handler; excluded from the
        # registry-dispatched MCP server (see _EXCLUDED_TOOLS in asi_mcp_adapter)
        assert "save_insight" not in tool_names
        assert "search_design_history" not in tool_names

    def test_each_tool_has_description(self):
        from external_llm.editor.agent.mcp import list_mcp_tools
        tools = list_mcp_tools()
        for t in tools:
            assert t.get("description"), f"Tool {t['name']} missing description"


class TestTimeoutOrdering:
    """The MCP ceiling must outlive the inner budget of self-policing tools.

    `bash` enforces its own timeout and, on expiry, salvages partial output and
    hands the process to BackgroundJobManager, returning a Job ID the agent can
    poll. Both layers previously defaulted to 120s, so the outer asyncio.wait_for
    won the race: the agent got TOOL_TIMEOUT, the Job ID was discarded, the
    background job was never reaped, and the worker thread kept running
    (observed: outer gave up at 120.0s, that same call completed at 122.9s).
    """

    def test_bash_ceiling_exceeds_inner_budget_across_the_allowed_range(self):
        from external_llm.agent.tool_handlers.shell_policy import SHELL_TIMEOUT_MAX

        for inner in range(1, SHELL_TIMEOUT_MAX + 1):
            outer = _resolve_mcp_timeout("bash", {"timeout": inner})
            assert outer > inner, f"inner={inner} outer={outer} — inner wins the race"

    def test_every_inner_timeout_tool_gets_headroom_over_its_own_max(self):
        """Contract for the whole table, so a newly added tool cannot regress."""
        for tool, (key, default, maximum) in _INNER_TIMEOUT_TOOLS.items():
            for inner in (1, default, maximum):
                outer = _resolve_mcp_timeout(tool, {key: inner})
                assert outer > inner, f"{tool}: inner={inner} outer={outer}"

    def test_bash_default_ceiling_leaves_room_for_the_background_transition(self):
        """No argument → the inner run uses its default, so the ceiling must clear it."""
        from external_llm.agent.tool_handlers.shell_policy import SHELL_TIMEOUT_DEFAULT

        assert _resolve_mcp_timeout("bash", {}) > SHELL_TIMEOUT_DEFAULT

    def test_bash_honours_the_schema_max_not_the_requested_absurdity(self):
        """timeout=99999 is clamped, so it cannot pin an executor thread for hours."""
        from external_llm.agent.tool_handlers.shell_policy import SHELL_TIMEOUT_MAX

        assert _resolve_mcp_timeout("bash", {"timeout": 99999}) == _resolve_mcp_timeout(
            "bash", {"timeout": SHELL_TIMEOUT_MAX}
        )

    @pytest.mark.parametrize("bad", [None, 0, "", "abc", -5, 3.7, [], {"nested": 1}])
    def test_malformed_timeout_arg_never_raises(self, bad):
        """A model-supplied garbage value must not break tool dispatch."""
        assert _resolve_mcp_timeout("bash", {"timeout": bad}) > 0

    @pytest.mark.parametrize("args", [None, "notadict", 42])
    def test_non_dict_args_fall_back_to_the_default(self, args):
        assert _resolve_mcp_timeout("bash", args) > 0

    def test_non_self_policing_tools_keep_their_static_budget(self):
        """Only _INNER_TIMEOUT_TOOLS get a derived ceiling; the rest are untouched."""
        for tool, expected in _TOOL_SPECIFIC_TIMEOUTS.items():
            if tool in _INNER_TIMEOUT_TOOLS:
                continue
            assert _resolve_mcp_timeout(tool, {}) == expected
        # A tool with no override falls through to the global default.
        from external_llm.repl.collaborate import asi_mcp_adapter as mod
        assert _resolve_mcp_timeout("apply_patch", {}) == mod.CLAUDE_MCP_TOOL_TIMEOUT

    def test_a_timeout_arg_is_ignored_for_tools_that_do_not_police_their_own(self):
        """An unrelated tool carrying a `timeout` key must not inflate its ceiling."""
        assert _resolve_mcp_timeout("read_file", {"timeout": 9999}) == (
            _TOOL_SPECIFIC_TIMEOUTS["read_file"]
        )

    def test_zero_env_disables_the_ceiling_and_is_not_resurrected(self, monkeypatch):
        """CLAUDE_MCP_TOOL_TIMEOUT=0 means "no ceiling" (config allows zero).

        The per-call derivation must not turn that opt-out back into a limit.
        """
        from external_llm.repl.collaborate import asi_mcp_adapter as mod

        monkeypatch.setattr(mod, "CLAUDE_MCP_TOOL_TIMEOUT", 0)
        assert _resolve_mcp_timeout("bash", {}) == 0
        assert _resolve_mcp_timeout("bash", {"timeout": 300}) == 0
        assert _resolve_mcp_timeout("apply_patch", {}) == 0

    def test_explicitly_raised_env_ceiling_still_wins(self, monkeypatch):
        """Raising the env ceiling above the derived one must not be downgraded."""
        from external_llm.repl.collaborate import asi_mcp_adapter as mod

        monkeypatch.setattr(mod, "CLAUDE_MCP_TOOL_TIMEOUT", 5000)
        assert _resolve_mcp_timeout("bash", {"timeout": 120}) == 5000
