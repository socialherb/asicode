"""Read-only session act gate: a whitelisted tool NAME must not carry a write ACT.

The exposure whitelist in the MCP adapter gates tool names, but the shell is a
channel rather than a tool: `bash` has to stay exposed for inspection (`ls`,
`grep`, `git status`) while its SAME handler writes files (`cp`/`mv`/`tee`/`>`),
rewrites git state (`git stash`, `git commit`) and runs arbitrary code
(`python3 -c "open(f,'w')…"`). These tests pin the second layer that judges the
ACT, end to end through the built MCP surface:

  - a mutating command is refused with isError, is never dispatched, and leaves
    the workspace byte-identical (the property "read-only" actually promises),
  - read-only inspection commands still run, so the gate is not a shell ban,
  - a write session dispatches the identical call unchanged (the gate is scoped),
  - the refusal names a recovery path, because a silent refusal invites a retry,
  - the name whitelist can never admit a declared write tool or a dead name.

The classifier the gate consults is the registry's mutation SSOT, so the command
fixtures below are asserted to classify as mutating BEFORE they are used as
exploit cases — if that SSOT changes, these tests report it instead of passing
vacuously.
"""

from __future__ import annotations

import asyncio

import pytest

from external_llm.agent.agent_loop_types import WRITE_TOOL_NAMES
from external_llm.agent.tool_registry import AgentConfig, ToolRegistry
from external_llm.repl.collaborate.asi_mcp_adapter import (
    _ANALYSIS_SAFE_TOOLS,
    _DESTRUCTIVE_TOOLS,
    _READ_ONLY_TOOLS,
    _read_only_refusal,
    build_asr_mcp_server,
)

#: Commands that change file / git state, each observed to succeed on the server
#: before the gate existed. Fixture data, not a table the gate consults.
MUTATING_COMMANDS = [
    "git stash",  # silently reverts the working tree — no upload/download witness
    "cp notes.txt notes_copy.txt",
    "mv notes.txt moved_notes.txt",
    "tee tee_out.txt",
    "echo added > brand_new.txt",
    "printf 'x' >> notes.txt",
    "sed -i '' 's/original/edited/' notes.txt",
    "python3 -c \"open('hello.py','w').write('PWNED')\"",
    "python3 -c \"import os; os.remove('hello.py')\"",
    "node -e \"require('fs').writeFileSync('hello.py', 'x')\"",
]

#: Inspection commands an analysis session actually needs — must keep running.
READ_ONLY_COMMANDS = [
    "ls -la",
    "cat notes.txt",
    "grep -n original notes.txt",
    "wc -l notes.txt",
]


def _registry(repo_root: str = ".") -> ToolRegistry:
    return ToolRegistry(repo_root=repo_root, config=AgentConfig())


class TestClassifierIsTheGateOracle:
    """The gate must rest on the registry classifier, never on its own ladder."""

    @pytest.mark.parametrize("command", MUTATING_COMMANDS)
    def test_exploit_fixtures_really_classify_as_mutating(self, command):
        # Guards the fixtures: a gate test whose command is invisible to the
        # classifier would prove nothing about the gate.
        assert _registry().is_read_only_call("bash", {"command": command}) is False

    @pytest.mark.parametrize("command", READ_ONLY_COMMANDS)
    def test_inspection_fixtures_classify_as_read_only(self, command):
        assert _registry().is_read_only_call("bash", {"command": command}) is True

    def test_unknown_command_shape_fails_closed(self):
        # Not on any whitelist, so it must not be treated as inspectable.
        assert _registry().is_read_only_call("bash", {"command": "frobnicate --write x"}) is False


class TestReadOnlyRefusalUnit:
    """`_read_only_refusal`: the decision function, independent of the SDK."""

    def test_mutating_bash_is_refused_and_quotes_the_command(self):
        text = _read_only_refusal(_registry(), "bash", {"command": "git stash"})
        assert text is not None
        assert "READ_ONLY_DENIED" in text
        assert "'git stash'" in text  # the transcript names what was blocked

    def test_read_only_bash_is_allowed(self):
        assert _read_only_refusal(_registry(), "bash", {"command": "ls -la"}) is None

    def test_read_tool_is_allowed(self):
        assert _read_only_refusal(_registry(), "read_file", {"path": "x.py"}) is None

    @pytest.mark.parametrize("malformed", [[1], "x", 123])
    def test_malformed_args_defer_to_the_tool(self, malformed):
        # dispatch rejects a non-dict argument before any handler runs (bash →
        # ok=False, error='command is required'), so the gate must not guess —
        # inventing its own error here would change the tool's error surface.
        assert _read_only_refusal(_registry(), "bash", malformed) is None

    def test_classifier_failure_fails_closed(self, monkeypatch):
        registry = _registry()

        def _boom(*_args, **_kwargs):
            raise RuntimeError("classifier exploded")

        monkeypatch.setattr(registry, "is_read_only_call", _boom)
        text = _read_only_refusal(registry, "bash", {"command": "ls"})
        assert text is not None
        assert "READ_ONLY_DENIED" in text


class TestReadOnlySessionSurface:
    """End to end through the built MCP server: gated acts, kept capability."""

    def setup_method(self, method):
        pytest.importorskip("claude_agent_sdk")

    @pytest.fixture
    def workspace(self, tmp_path):
        (tmp_path / "notes.txt").write_text("original notes\n")
        (tmp_path / "hello.py").write_text("ORIGINAL = 1\n")
        return tmp_path

    def _built_session(self, workspace, monkeypatch, *, read_only: bool):
        """Build the session; return (registry, {tool name: sdk tool}).

        Commands run with cwd=registry.repo_root, so the workspace IS the repo
        root here — no chdir, hence no leakage into the checkout.
        """
        import claude_agent_sdk

        captured: dict = {}

        def _fake_create(name, version, tools):
            captured["tools"] = tools
            return {"name": name, "version": version, "tools": tools, "type": "sdk"}

        monkeypatch.setattr(claude_agent_sdk, "create_sdk_mcp_server", _fake_create)
        registry = ToolRegistry(repo_root=str(workspace), config=AgentConfig())
        build_asr_mcp_server(registry, read_only=read_only)
        return registry, {t.name: t for t in captured["tools"]}

    @staticmethod
    def _dispatched_calls(registry, monkeypatch) -> list:
        """Record every dispatch that reaches the registry."""
        seen: list = []
        real = registry.dispatch

        def _spy(tool_name, args):
            seen.append((tool_name, dict(args or {})))
            return real(tool_name, args)

        monkeypatch.setattr(registry, "dispatch", _spy)
        return seen

    @staticmethod
    def _workspace_snapshot(workspace) -> dict:
        """content of every workspace file, recursively, keyed by relative path.

        ``.asicode/`` is excluded: it is the registry's own metadata area, created
        at construction and written by its caching machinery — not a shell act.
        The property under test is that REFUSED shell acts change no content, so
        excluding registry bookkeeping keeps that property exact instead of
        turning it into a flaky comparison against the registry's own writes.
        """
        return {
            str(p.relative_to(workspace)): p.read_bytes()
            for p in workspace.rglob("*")
            if p.is_file() and ".asicode" not in p.relative_to(workspace).parts
        }

    def test_bash_is_still_exposed_in_a_read_only_session(self, workspace, monkeypatch):
        # The design is "expose bash, gate the act" — a fix that removed bash
        # would silently take read-only inspection away from the analysis.
        _registry_obj, tools = self._built_session(workspace, monkeypatch, read_only=True)
        assert "bash" in tools
        assert "read_file" in tools

    @pytest.mark.parametrize("command", MUTATING_COMMANDS)
    def test_mutating_command_is_refused_before_dispatch(self, workspace, monkeypatch, command):
        registry, tools = self._built_session(workspace, monkeypatch, read_only=True)
        seen = self._dispatched_calls(registry, monkeypatch)

        out = asyncio.run(tools["bash"].handler({"command": command}))

        assert out["isError"] is True
        assert "READ_ONLY_DENIED" in out["content"][0]["text"]
        assert seen == [], f"the gate must refuse BEFORE dispatch: {seen}"

    def test_refused_commands_leave_the_workspace_byte_identical(self, workspace, monkeypatch):
        # The property "read-only" actually promises, checked on the whole
        # exploit battery at once — including files created out of nothing.
        _registry_obj, tools = self._built_session(workspace, monkeypatch, read_only=True)
        before = self._workspace_snapshot(workspace)
        assert before, "empty snapshot would make the comparison vacuous"

        for command in MUTATING_COMMANDS:
            asyncio.run(tools["bash"].handler({"command": command}))

        assert self._workspace_snapshot(workspace) == before

    @pytest.mark.parametrize("command", READ_ONLY_COMMANDS)
    def test_read_only_inspection_still_runs(self, workspace, monkeypatch, command):
        registry, tools = self._built_session(workspace, monkeypatch, read_only=True)
        seen = self._dispatched_calls(registry, monkeypatch)

        out = asyncio.run(tools["bash"].handler({"command": command}))

        assert "READ_ONLY_DENIED" not in out["content"][0]["text"]
        assert [name for name, _args in seen] == ["bash"]

    def test_write_session_dispatches_the_identical_mutating_call(self, workspace, monkeypatch):
        registry, tools = self._built_session(workspace, monkeypatch, read_only=False)
        seen = self._dispatched_calls(registry, monkeypatch)

        asyncio.run(tools["bash"].handler({"command": "cp notes.txt notes_copy.txt"}))

        assert [name for name, _args in seen] == ["bash"]
        assert (workspace / "notes_copy.txt").read_text() == "original notes\n"

    def test_refusal_names_a_recovery_path(self, workspace, monkeypatch):
        _registry_obj, tools = self._built_session(workspace, monkeypatch, read_only=True)

        text = asyncio.run(tools["bash"].handler({"command": "git stash"}))["content"][0]["text"]

        assert "READ_ONLY_DENIED" in text
        assert "suggestion in the verdict" in text  # the sanctioned output channel
        assert "git status" in text  # names commands that DO work
        assert "do not retry" in text  # stops the retry loop a bare refusal invites


class TestExposureWhitelistInvariants:
    """Layer one must not drift: no write tool, no dead name in the whitelist."""

    def test_whitelist_never_admits_a_write_tool(self):
        whitelist = _READ_ONLY_TOOLS | _ANALYSIS_SAFE_TOOLS
        # Both sides derived: the write set is the registry's SSOT, the
        # destructive set the adapter excludes. A future edit that adds a write
        # tool to the read-only surface fails here, not in production.
        assert whitelist & set(WRITE_TOOL_NAMES) == set()
        assert whitelist & _DESTRUCTIVE_TOOLS == set()

    def test_every_whitelisted_name_is_a_real_exposed_tool(self):
        registry = _registry()
        exposed = {s["name"] for s in registry.get_tool_schemas(lang_filter=registry.repo_language)}
        whitelist = _READ_ONLY_TOOLS | _ANALYSIS_SAFE_TOOLS

        assert whitelist, "an empty whitelist would satisfy the subset checks vacuously"
        assert whitelist <= exposed
        for name in sorted(whitelist):
            assert registry.has_tool_handler(name), name

    def test_declared_read_only_tools_are_classified_read_only(self):
        registry = _registry()
        for name in sorted(_READ_ONLY_TOOLS):
            assert registry.is_read_only_call(name, {}) is True, name
