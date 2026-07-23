"""Tests for the trust-scoped cross-repo read boundary (AgentConfig.unrestricted_read).

The read tools (read_file / get_file_outline / read_image) resolve paths through
ToolRegistry._secure_path, which confines them to repo_root by default. A trusted
local CLI opts out via ``unrestricted_read=True`` so the agent can read sibling
repos / arbitrary host paths — matching what the always-available ``bash`` tool
can already do. The webapp (attacker-controlled repo_root) MUST keep the default.

These tests pin:
  1. default (restricted) still blocks reads outside repo_root,
  2. unrestricted allows them,
  3. in-repo reads work in both modes,
  4. the flag survives dataclasses.replace (how sub-agent/orchestrator configs
     inherit it),
  5. write-path confinement: modify_symbol/edit_ast stay confined to repo_root
     regardless of the flag (via ``_secure_path(confine=True)``).
"""
from __future__ import annotations

import dataclasses

from external_llm.agent.tool_registry import AgentConfig, ToolRegistry


def _make_repo_and_outside(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "inside.py").write_text("x = 1\n", encoding="utf-8")

    outside = tmp_path / "sibling"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("cross-repo content\n", encoding="utf-8")
    return repo, secret


class TestRestrictedByDefault:
    def test_secure_path_blocks_absolute_outside(self, tmp_path):
        repo, secret = _make_repo_and_outside(tmp_path)
        reg = ToolRegistry(str(repo), AgentConfig())
        assert reg._secure_path(str(secret)) is None

    def test_secure_path_blocks_relative_traversal(self, tmp_path):
        repo, _ = _make_repo_and_outside(tmp_path)
        reg = ToolRegistry(str(repo), AgentConfig())
        assert reg._secure_path("../sibling/secret.txt") is None

    def test_read_file_reports_outside_repo(self, tmp_path):
        repo, secret = _make_repo_and_outside(tmp_path)
        reg = ToolRegistry(str(repo), AgentConfig())
        result = reg._tool_read_file({"path": str(secret)})
        assert result.ok is False
        assert "outside repo" in (result.error or "")

    def test_in_repo_read_still_works(self, tmp_path):
        repo, _ = _make_repo_and_outside(tmp_path)
        reg = ToolRegistry(str(repo), AgentConfig())
        result = reg._tool_read_file({"path": "inside.py"})
        assert result.ok is True
        assert "x = 1" in result.content


class TestUnrestrictedRead:
    def test_secure_path_allows_absolute_outside(self, tmp_path):
        repo, secret = _make_repo_and_outside(tmp_path)
        reg = ToolRegistry(str(repo), AgentConfig(unrestricted_read=True))
        resolved = reg._secure_path(str(secret))
        assert resolved is not None
        assert resolved == secret.resolve()

    def test_secure_path_allows_relative_traversal(self, tmp_path):
        repo, secret = _make_repo_and_outside(tmp_path)
        reg = ToolRegistry(str(repo), AgentConfig(unrestricted_read=True))
        resolved = reg._secure_path("../sibling/secret.txt")
        assert resolved == secret.resolve()

    def test_read_file_reads_cross_repo(self, tmp_path):
        repo, secret = _make_repo_and_outside(tmp_path)
        reg = ToolRegistry(str(repo), AgentConfig(unrestricted_read=True))
        result = reg._tool_read_file({"path": str(secret)})
        assert result.ok is True
        assert "cross-repo content" in result.content

    def test_in_repo_read_still_works(self, tmp_path):
        repo, _ = _make_repo_and_outside(tmp_path)
        reg = ToolRegistry(str(repo), AgentConfig(unrestricted_read=True))
        result = reg._tool_read_file({"path": "inside.py"})
        assert result.ok is True
        assert "x = 1" in result.content


class TestInheritance:
    def test_default_is_false(self):
        assert AgentConfig().unrestricted_read is False

    def test_replace_preserves_flag(self):
        """Orchestrator sub-agent configs are built with dataclasses.replace(base,
        ...); the flag must propagate so sub-agents share the parent's trust."""
        base = AgentConfig(unrestricted_read=True)
        sub = dataclasses.replace(base, max_turns=3)
        assert sub.unrestricted_read is True


class TestWritePathAlwaysConfined:
    """Writes MUST stay confined to repo_root regardless of unrestricted_read.

    Regression guard for a real defect: ``modify_symbol`` / ``edit_ast`` resolve
    their target path through ``_secure_path`` (not ``resolve_inside_repo``), and
    ``_do_modify`` performs no boundary check of its own. Before the fix, an
    ``unrestricted_read=True`` trusted CLI could write outside repo_root via these
    tools — the flag was a read capability that silently became a write
    capability. ``_secure_path(path, confine=True)`` closes that; these tests
    exercise the ACTUAL write tool entry points (not resolve_inside_repo, which
    only the apply_patch/write_plan backends use).
    """

    def _make_target(self, tmp_path):
        """A sibling .py file outside repo with one top-level symbol to mutate."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "inside.py").write_text("x = 1\n", encoding="utf-8")

        outside = tmp_path / "sibling"
        outside.mkdir()
        victim = outside / "victim.py"
        victim.write_text("def f():\n    return 1\n", encoding="utf-8")
        return repo, victim

    def test_modify_symbol_blocked_outside_unrestricted(self, tmp_path):
        repo, victim = self._make_target(tmp_path)
        reg = ToolRegistry(str(repo), AgentConfig(unrestricted_read=True))
        result = reg._tool_modify_symbol(
            {"file_path": str(victim), "symbol": "f", "code": "def f():\n    return 999\n"}
        )
        assert result.ok is False
        assert "Path traversal blocked" in (result.error or "")
        # Crucially, the outside file is untouched.
        assert "return 1" in victim.read_text(encoding="utf-8")

    def test_modify_symbol_blocked_outside_restricted(self, tmp_path):
        repo, victim = self._make_target(tmp_path)
        reg = ToolRegistry(str(repo), AgentConfig())  # default restricted
        result = reg._tool_modify_symbol(
            {"file_path": str(victim), "symbol": "f", "code": "def f():\n    return 999\n"}
        )
        assert result.ok is False
        assert "return 1" in victim.read_text(encoding="utf-8")

    def test_edit_ast_blocked_outside_unrestricted(self, tmp_path):
        repo, victim = self._make_target(tmp_path)
        reg = ToolRegistry(str(repo), AgentConfig(unrestricted_read=True))
        result = reg._tool_edit_ast(
            {
                "file_path": str(victim),
                "symbol": "f",
                "ops": [{"type": "replace_expr", "old": "1", "new": "999"}],
            }
        )
        assert result.ok is False
        assert "outside repo" in (result.error or "") or "Path" in (result.error or "")
        assert "return 1" in victim.read_text(encoding="utf-8")

    def test_edit_ast_blocked_outside_restricted(self, tmp_path):
        repo, victim = self._make_target(tmp_path)
        reg = ToolRegistry(str(repo), AgentConfig())  # default restricted
        result = reg._tool_edit_ast(
            {
                "file_path": str(victim),
                "symbol": "f",
                "ops": [{"type": "replace_expr", "old": "1", "new": "999"}],
            }
        )
        assert result.ok is False
        assert "return 1" in victim.read_text(encoding="utf-8")

    def test_modify_symbol_in_repo_still_works(self, tmp_path):
        """Confinement must not break legitimate in-repo writes (unrestricted)."""
        repo, _ = self._make_target(tmp_path)
        (repo / "mod.py").write_text("def f():\n    return 1\n", encoding="utf-8")
        reg = ToolRegistry(str(repo), AgentConfig(unrestricted_read=True))
        result = reg._tool_modify_symbol(
            {"file_path": "mod.py", "symbol": "f", "code": "def f():\n    return 7\n"}
        )
        assert result.ok is True
        assert "return 7" in (repo / "mod.py").read_text(encoding="utf-8")
