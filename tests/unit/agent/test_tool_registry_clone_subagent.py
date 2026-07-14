"""Regression tests for ToolRegistry.clone_for_subagent().

Bug: clone_for_subagent() built the clone via object.__new__ (bypassing
__init__) and set _applied_patches/_search_cache as "fresh mutable state,"
but never set _text_edited_files or _agent_profile — unlike its sibling
clone_with_filter(), which sets both (tool_registry.py:568/562).

Impact:
  * _text_edited_files: write_tools._tool_apply_patch unconditionally reads
    self._text_edited_files (write_tools.py:2059) on every apply_patch call.
    Without it, every subagent apply_patch (orchestrator.py's parallel/
    sequential subagent path, via clone_for_subagent) raised AttributeError,
    swallowed by dispatch()'s broad `except Exception` (tool_registry.py)
    into an opaque ok=False tool failure — subagents could never apply a
    patch.
  * _agent_profile: dispatch()'s tool-restriction gate uses
    hasattr(self, '_agent_profile') to decide whether to enforce
    allowed_tools/blocked_tools. Without the attribute, the gate is always
    skipped — subagent tool restrictions were silently unenforced.
"""
import subprocess

from external_llm.agent.tool_registry import AgentConfig, ToolRegistry


def _init_git_repo(tmp_path):
    repo_root = str(tmp_path)
    subprocess.run(["git", "init", "-q"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo_root, check=True)
    (tmp_path / "a.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "-A"], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo_root, check=True)
    return repo_root


def test_clone_for_subagent_sets_text_edited_files(tmp_path):
    repo_root = _init_git_repo(tmp_path)
    registry = ToolRegistry(repo_root, AgentConfig())
    clone = registry.clone_for_subagent(AgentConfig())

    assert hasattr(clone, "_text_edited_files")
    assert clone._text_edited_files == set()
    # Fresh per-subagent state — must NOT be the same object as the parent's.
    assert clone._text_edited_files is not registry._text_edited_files


def test_clone_for_subagent_inherits_agent_profile(tmp_path):
    repo_root = _init_git_repo(tmp_path)
    registry = ToolRegistry(repo_root, AgentConfig())
    registry._agent_profile = object()  # sentinel standing in for a real AgentProfile

    clone = registry.clone_for_subagent(AgentConfig())

    assert hasattr(clone, "_agent_profile")
    assert clone._agent_profile is registry._agent_profile


def test_clone_for_subagent_apply_patch_dispatch_succeeds(tmp_path):
    """End-to-end regression guard: a subagent registry must be able to
    dispatch apply_patch without crashing on missing _text_edited_files."""
    repo_root = _init_git_repo(tmp_path)
    registry = ToolRegistry(repo_root, AgentConfig())
    clone = registry.clone_for_subagent(AgentConfig())

    patch = (
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1 +1 @@\n"
        "-x = 1\n"
        "+x = 2\n"
    )
    result = clone.dispatch("apply_patch", {"patch": patch, "path": "a.py"})

    assert result.ok, result.error
    assert (tmp_path / "a.py").read_text() == "x = 2\n"
