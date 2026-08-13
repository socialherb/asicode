"""get_git_snapshot must canonicalize repo_root before keying _git_cache (P1-1).

Callers spell the same repo differently (resolved registry.repo_root vs raw
request strings from service.py; macOS /var vs /private/var), and an
uncanonalized key let one repo occupy 2+ entries of the 8-entry cache — the
same bug the file-index cache was fixed for, now shared via canonical_repo_key.
"""
from __future__ import annotations

from pathlib import Path

import external_llm.agent.agent_context_manager as acm


def test_git_snapshot_same_repo_different_spelling_hits_one_cache_entry(monkeypatch, tmp_path):
    # Hard reset: _clear_git_cache is coalesced (P3) and no longer empties the
    # dict, so cross-test isolation needs explicit clears.
    acm._git_cache.clear()
    acm._git_dirty_since.clear()
    runs = []

    def _fake_git(root, *args):
        runs.append((root, args))
        return "main" if args and args[0] == "rev-parse" else ""

    monkeypatch.setattr(acm, "_run_git_raw", _fake_git)

    spelling_a = str(tmp_path)
    spelling_b = str(tmp_path) + "/."  # unresolved spelling; resolve() normalizes

    first = acm.get_git_snapshot(spelling_a)
    runs_after_first = len(runs)
    second = acm.get_git_snapshot(spelling_b)

    assert len(runs) == runs_after_first, "second spelling must hit the cache"
    assert first == second
    assert len(acm._git_cache) == 1, "one repo must occupy exactly one entry"
    resolved = str(Path(str(tmp_path)).resolve())
    assert all(k == resolved for k in acm._git_cache), "cache key must be canonical"
