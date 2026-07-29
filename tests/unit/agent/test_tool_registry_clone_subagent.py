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


# ---------------------------------------------------------------------------
# Clone-parity guard: every attribute set in __init__ must be present in
# BOTH clone paths.  object.__new__() bypasses __init__, so each new field
# must be hand-copied — and there is nothing forcing that to happen.  This
# test is the forcing function: add a field to __init__, the test fails
# until both clones mirror it (or it's added to the intentional-skip list
# below WITH a reason).
# ---------------------------------------------------------------------------

# The expected set is read off a LIVE instance rather than written out here.
# A literal list cannot be the forcing function this test is for: it drifts in
# exactly the way the clone methods drift, so a field added to __init__ and
# mirrored nowhere leaves the list — and therefore the test — untouched.
# Measured: with a hand-written list, adding `_brand_new_field` to __init__
# alone kept all seven tests green, which is precisely how _semantic_pending
# got in. Deriving it means a new field is in the expected set the moment it
# exists, and the assertion fails until both clones carry it.
def _init_attrs(registry) -> frozenset:
    return frozenset(vars(registry))


# Per-clone intentional omissions. An entry here needs a comment saying why the
# clone is correct without the attribute; both are empty because both clones
# currently mirror every field.
_CLONE_SUBAGENT_MISSING_OK: frozenset[str] = frozenset()
_CLONE_FILTER_MISSING_OK: frozenset[str] = frozenset()


def test_clone_for_subagent_has_all_init_attrs(tmp_path):
    """clone_for_subagent() must set every attribute that __init__ sets."""
    repo_root = _init_git_repo(tmp_path)
    registry = ToolRegistry(repo_root, AgentConfig())
    clone = registry.clone_for_subagent(AgentConfig())

    clone_attrs = frozenset(vars(clone).keys())
    missing = (_init_attrs(registry) - _CLONE_SUBAGENT_MISSING_OK) - clone_attrs
    assert missing == set(), (
        f"clone_for_subagent missing attrs: {sorted(missing)}. "
        f"Add them to clone_for_subagent(), or add to _CLONE_SUBAGENT_MISSING_OK "
        f"with a comment."
    )


def test_clone_with_filter_has_all_init_attrs(tmp_path):
    """clone_with_filter() must set every attribute that __init__ sets."""
    from external_llm.agent.tool_chain import ScopedToolFilter

    repo_root = _init_git_repo(tmp_path)
    registry = ToolRegistry(repo_root, AgentConfig())
    flt = ScopedToolFilter(allowed_write={repo_root})
    clone = registry.clone_with_filter(flt)

    clone_attrs = frozenset(vars(clone).keys())
    missing = (_init_attrs(registry) - _CLONE_FILTER_MISSING_OK) - clone_attrs
    assert missing == set(), (
        f"clone_with_filter missing attrs: {sorted(missing)}. "
        f"Add them to clone_with_filter(), or add to _CLONE_FILTER_MISSING_OK "
        f"with a comment."
    )


def test_clone_for_subagent_sets_semantic_coalesce_fields(tmp_path):
    """Regression: semantic-lint coalescing fields were missing from subagent
    clones, causing silent AttributeError → syntax/semantic diagnostics dropped
    for ALL subagent edits."""
    repo_root = _init_git_repo(tmp_path)
    registry = ToolRegistry(repo_root, AgentConfig())
    clone = registry.clone_for_subagent(AgentConfig())

    assert hasattr(clone, "_semantic_pending"), (
        "_semantic_pending missing — defer_semantic_check() raises AttributeError"
    )
    assert clone._semantic_pending == {}
    assert hasattr(clone, "_semantic_turn_active"), (
        "_semantic_turn_active missing — defer_semantic_check() raises AttributeError"
    )
    assert clone._semantic_turn_active is False
    # ISOLATED: mutations on the clone must NOT leak to parent.
    clone._semantic_pending["test.txt"] = "content"
    registry._semantic_pending["parent.txt"] = "pc"
    assert "test.txt" not in registry._semantic_pending
    assert "parent.txt" not in clone._semantic_pending


def test_clone_with_filter_has_repo_language(tmp_path):
    """Regression: clone_with_filter() was missing _repo_language, so
    agent_loop.py:2770 (self.registry.repo_language) raised AttributeError
    for scoped-delegation / filtered-clone dispatch."""
    from external_llm.agent.tool_chain import ScopedToolFilter
    from external_llm.languages.models import LanguageId

    repo_root = _init_git_repo(tmp_path)
    registry = ToolRegistry(repo_root, AgentConfig())
    flt = ScopedToolFilter(allowed_write={repo_root})
    clone = registry.clone_with_filter(flt)

    assert hasattr(clone, "_repo_language"), (
        "_repo_language missing — repo_language property raises AttributeError"
    )
    # Reading the property is the actual failure the delegate loop hit
    # (agent_loop builds tool schemas with registry.repo_language, on a line
    # no try covers), so exercise it rather than only the underlying field.
    assert clone.repo_language == registry.repo_language

    # Comparing the two straight out of a Python-only fixture repo asserts
    # None == None, which stays true even if the copy is deleted. Give the
    # parent a real value first, so the assertion is about the copy.
    registry._repo_language = LanguageId.TYPESCRIPT
    clone2 = registry.clone_with_filter(ScopedToolFilter(allowed_write={repo_root}))
    assert clone2._repo_language is LanguageId.TYPESCRIPT, (
        "clone_with_filter dropped a non-default _repo_language"
    )
