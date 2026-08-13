"""A file written this turn must be visible to the tools that list files.

`_repo_file_index` (git ls-files, 60 s TTL) backs the `glob` tool and the
"Did you mean:" path suggester. It was the one per-repo cache missing from both
post-write invalidation routines, so `glob` could not see a file the agent had
just created while `find_symbol` — fixed for exactly this — already could.

Two write shapes must both clear it: the write TOOLS go through
`_invalidate_cache_after_write` (known target paths), and a mutating `bash`
goes through `_invalidate_caches_unknown_scope` (targets unknowable).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from external_llm.agent.tool_handlers import write_tools as wt
from external_llm.agent.tool_handlers.write_tools import (
    _repo_file_index,
    canonical_repo_key,
    invalidate_repo_file_index,
)
from external_llm.agent.tool_registry import AgentConfig, ToolRegistry

NEW = "brand_new.py"


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    (tmp_path / "existing.py").write_text("x = 1\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return tmp_path


def _registry(repo: Path) -> ToolRegistry:
    return ToolRegistry(str(repo), AgentConfig(rag_enabled=False))


def _globbed(reg: ToolRegistry) -> set[str]:
    r = reg.dispatch("glob", {"pattern": "*.py"})
    assert r.ok
    return {
        line.strip().split("  (")[0]
        for line in (r.content or "").splitlines()[1:] if line.strip()
    }


class TestGlobSeesFreshWrites:
    """Each case warms the index first — an unwarmed cache would pass
    vacuously, since the miss path rebuilds from git anyway."""

    def test_bash_created_file(self, repo: Path):
        reg = _registry(repo)
        assert NEW not in _globbed(reg)
        assert reg.dispatch("bash", {"command": f"printf 'x=1\\n' > {NEW}"}).ok
        assert NEW in _globbed(reg)

    def test_write_plan_created_file(self, repo: Path):
        reg = _registry(repo)
        assert NEW not in _globbed(reg)
        assert reg.dispatch("write_plan", {"plan": {
            "version": "ASICODE_PLAN_V1",
            "operations": [{"op": "create_file", "path": NEW, "content": "def hello():\n    return 1\n"}],
        }}).ok
        assert NEW in _globbed(reg)

    def test_apply_patch_created_file(self, repo: Path):
        reg = _registry(repo)
        assert NEW not in _globbed(reg)
        assert reg.dispatch("apply_patch", {"patch":
            f"--- /dev/null\n+++ b/{NEW}\n@@ -0,0 +1,2 @@\n+def hello():\n+    return 1\n"}).ok
        assert NEW in _globbed(reg)

    def test_renamed_file_moves(self, repo: Path):
        """The stale-index symptom cuts both ways — a path that no longer
        exists must stop being listed, not just a new one start being listed.
        (`rm` is deliberately not used: it needs approval, which is correct.)"""
        reg = _registry(repo)
        assert "existing.py" in _globbed(reg)
        assert reg.dispatch("bash", {"command": "mv existing.py renamed.py"}).ok
        after = _globbed(reg)
        assert "renamed.py" in after
        assert "existing.py" not in after

    def test_read_only_bash_does_not_invalidate(self, repo: Path):
        """Read-only bash must leave the cache alone — the hit rate that makes
        the index worth having depends on it surviving interleaved reads."""
        reg = _registry(repo)
        _globbed(reg)
        key = canonical_repo_key(str(repo))
        before = wt._FILE_INDEX_CACHE.get(key)
        assert before is not None
        assert reg.dispatch("bash", {"command": "ls -la"}).ok
        assert wt._FILE_INDEX_CACHE.get(key) is before, "read-only bash evicted the index"


class TestFileIndexKey:
    def test_key_is_canonical_across_spellings(self, tmp_path: Path):
        """On macOS /var is a symlink to /private/var, so an unresolved and a
        resolved spelling of one repo occupied two entries of an 8-entry cache
        — and the invalidator could clear a key the reader never used."""
        resolved = tmp_path.resolve()
        assert canonical_repo_key(str(tmp_path)) == canonical_repo_key(str(resolved))

    def test_one_repo_occupies_one_entry(self, repo: Path):
        wt._FILE_INDEX_CACHE.clear()
        _repo_file_index(str(repo))
        _repo_file_index(str(repo.resolve()))
        _repo_file_index(str(repo) + "/")
        mine = [k for k in wt._FILE_INDEX_CACHE if repo.name in k]
        assert len(mine) == 1, f"one repo took {len(mine)} cache slots: {mine}"

    def test_invalidate_is_a_noop_for_unknown_root(self, tmp_path: Path):
        invalidate_repo_file_index(str(tmp_path / "never-cached"))  # must not raise

    def test_invalidate_drops_the_entry(self, repo: Path):
        _repo_file_index(str(repo))
        assert canonical_repo_key(str(repo)) in wt._FILE_INDEX_CACHE
        invalidate_repo_file_index(str(repo))
        assert canonical_repo_key(str(repo)) not in wt._FILE_INDEX_CACHE


def test_suggester_and_glob_share_one_index(repo: Path):
    """Both read `_repo_file_index`, so the invalidation covers both. Pinned
    because a future split would silently reintroduce the stale suggester."""
    import inspect

    from external_llm.agent.tool_handlers.read_tools import ReadToolsMixin
    assert "_repo_file_index" in inspect.getsource(ReadToolsMixin._tool_glob)
    assert "_repo_file_index" in inspect.getsource(wt.WriteToolsMixin._suggest_missing_paths)

