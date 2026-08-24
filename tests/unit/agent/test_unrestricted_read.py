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
from pathlib import Path
from unittest.mock import patch

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

    def test_grep_blocks_absolute_outside(self, tmp_path):
        """grep must share the repo-boundary gate — the only read tool that lacked it."""
        repo, secret = _make_repo_and_outside(tmp_path)
        reg = ToolRegistry(str(repo), AgentConfig())
        result = reg._tool_grep({"pattern": "root", "path": str(secret.parent)})
        assert result.ok is False
        assert "outside" in (result.error or "")

    def test_grep_in_repo_still_works(self, tmp_path):
        repo, _ = _make_repo_and_outside(tmp_path)
        # repo/inside.py contains "x = 1"
        reg = ToolRegistry(str(repo), AgentConfig())
        result = reg._tool_grep({"pattern": "x", "path": "."})
        assert result.ok is True
        assert "inside.py" in result.content


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

    def test_grep_reads_cross_repo_unrestricted(self, tmp_path):
        """unrestricted_read=True: grep must reach outside repo (parity with read_file)."""
        repo, secret = _make_repo_and_outside(tmp_path)
        reg = ToolRegistry(str(repo), AgentConfig(unrestricted_read=True))
        result = reg._tool_grep({"pattern": "cross-repo", "path": str(secret.parent)})
        assert result.ok is True
        assert "cross-repo" in result.content


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
        result = reg._tool_modify_symbol({"file_path": "mod.py", "symbol": "f", "code": "def f():\n    return 7\n"})
        assert result.ok is True
        assert "return 7" in (repo / "mod.py").read_text(encoding="utf-8")


class TestSecurePathRootResolveCache:
    """The repo-root resolution inside ``_secure_path`` is memoized per
    effective-root string.

    The root is a session constant (``repo_root`` frozen at construction,
    ``_repo_root_override`` set at most once), so re-resolving it on every
    read/write tool call was pure filesystem I/O on the hottest tool path.
    Only the ROOT resolve is cached — the candidate path must keep resolving
    fresh on every call, because that resolution IS the symlink boundary check.
    """

    def test_root_resolved_once_candidate_every_call(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "inside.py").write_text("x = 1\n", encoding="utf-8")
        reg = ToolRegistry(str(repo), AgentConfig())
        # NOTE: a wraps= mock does NOT bind self for Path.resolve (TypeError →
        # _secure_path's blanket except returns None); a plain function as the
        # patched class attribute binds via the descriptor protocol and works.
        real_resolve = Path.resolve
        resolve_calls: list = []

        def counting_resolve(p, *args, **kwargs):
            resolve_calls.append(p)
            return real_resolve(p, *args, **kwargs)

        with patch.object(Path, "resolve", counting_resolve):
            assert reg._secure_path("inside.py") is not None
            first_call_count = len(resolve_calls)
            assert first_call_count == 2, f"first call: root + candidate resolves expected 2, got {first_call_count}"
            assert reg._secure_path("inside.py") is not None
        assert len(resolve_calls) == first_call_count + 1, (
            "second call must re-resolve ONLY the candidate (root memoized)"
        )

    def test_override_change_rekeys_cache(self, tmp_path):
        repo = tmp_path / "repo"
        repo2 = tmp_path / "repo2"
        repo.mkdir()
        repo2.mkdir()
        (repo / "a.py").write_text("a = 1\n", encoding="utf-8")
        (repo2 / "b.py").write_text("b = 1\n", encoding="utf-8")
        reg = ToolRegistry(str(repo), AgentConfig())
        # Seed the cache under the original root.
        assert reg._secure_path("a.py") is not None
        # A relative traversal that escapes the original root is blocked.
        assert reg._secure_path("../repo2/b.py") is None
        # Switching the override re-keys: the same traversal now lands INSIDE
        # the new effective root and must resolve.
        reg._repo_root_override = str(repo2)
        got = reg._secure_path("../repo2/b.py")
        assert got is not None
        assert got == (repo2 / "b.py").resolve()
        assert reg._secure_path("b.py") is not None
        # Both roots are cached independently (no cross-contamination).
        assert set(reg._secure_root_resolve_cache) == {
            str(Path(repo).resolve()),
            str(repo2),
        }

    def test_override_cycles_keep_cache_bounded_by_distinct_roots(self, tmp_path):
        # Regression seal: _secure_root_resolve_cache must stay bounded by the
        # number of DISTINCT effective roots ever used — never by the number of
        # override assignments or by per-path lookups (the key is the root
        # string, so re-assigning an override only ever ADDS a key for a root
        # not seen before; a second assignment of the same root reuses the
        # existing entry).
        repo_a = tmp_path / "a"
        repo_b = tmp_path / "b"
        repo_c = tmp_path / "c"
        for r in (repo_a, repo_b, repo_c):
            r.mkdir()
            (r / "f.py").write_text("x = 1\n", encoding="utf-8")
        reg = ToolRegistry(str(repo_a), AgentConfig())
        # Seed the cache under the initial root so repo_a's key exists BEFORE
        # the cycle (the invariant below counts 3 distinct roots, and counting
        # the seed would otherwise mask a growing-cache regression).
        assert reg._secure_path("f.py") is not None

        def cycle_roots() -> None:
            roots = [str(repo_b), str(repo_c), str(repo_b), str(repo_a)]
            for r in roots:
                reg._repo_root_override = r
                # Resolve a relative path that only exists inside the current
                # effective root, forcing root resolution under that override.
                assert reg._secure_path("f.py") is not None

        cycle_roots()
        # The cycle touched b, c, b, a — only 3 DISTINCT roots (a was already
        # seeded), so the cache must hold exactly one entry per distinct root.
        assert len(reg._secure_root_resolve_cache) == 3, (
            f"cache must hold one entry per DISTINCT root, got {len(reg._secure_root_resolve_cache)}"
            f" keys: {sorted(reg._secure_root_resolve_cache)}"
        )
        # Re-running the same cycle must NOT grow the cache further (same keys
        # hit again — no new entries).
        cycle_roots()
        assert len(reg._secure_root_resolve_cache) == 3

    def test_clone_for_subagent_gets_own_cache(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "a.py").write_text("a = 1\n", encoding="utf-8")
        reg = ToolRegistry(str(repo), AgentConfig())
        assert reg._secure_path("a.py") is not None
        clone = reg.clone_for_subagent(AgentConfig())
        # clone_for_subagent bypasses __init__ (object.__new__); the clone must
        # still resolve paths (own fresh memo, NOT the parent's object).
        assert clone._secure_path("a.py") is not None
        assert clone._secure_root_resolve_cache is not reg._secure_root_resolve_cache
