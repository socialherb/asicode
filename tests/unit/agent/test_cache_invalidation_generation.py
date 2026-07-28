"""Post-write cache invalidation must survive landing MID-collection.

All three per-root caches refill as "read → slow collect → store". A `pop()`
that runs while the collect is in flight is lost: the store that follows
re-inserts the pre-write result, so the write stays invisible for a full TTL —
the "cannot find what it just wrote" symptom the invalidation exists to kill.

Each cache guards this with a generation counter read before the collect and
re-checked before the store. These tests drive that window directly, because
the two ways to get the counter wrong are both SILENT no-ops that read as
working code:

  * ``+= 1`` on an int imported into another module rebinds the importer's
    local and leaves the source module at 0.
  * passing ``[_GEN]`` into the walk engine boxes a COPY of the value, so the
    re-check compares a snapshot against itself and can never fire.

Both shipped once (the walk cache carried them simultaneously, making its fix
entirely inert while looking correct), hence the explicit assertions on the
counter object itself rather than only on end-to-end behaviour.
"""
from __future__ import annotations

import threading
import time

import external_llm.agent._shared_utils as su
import external_llm.agent.agent_context_manager as acm
import external_llm.agent.tool_handlers.write_tools as wt


# ── The counters must be mutable shared objects, not ints ──────────────────

def test_walk_generation_counters_are_shared_mutable_boxes():
    """A bare int cannot carry a bump across a ``from ... import``."""
    for gen in (su._PY_WALK_GEN, su._TS_WALK_GEN):
        assert isinstance(gen, list) and len(gen) == 1
        assert isinstance(gen[0], int)


def test_invalidate_walk_caches_bump_is_visible_to_importers():
    """Mirrors how tool_registry reaches the counters: import, then observe."""
    from external_llm.agent._shared_utils import _PY_WALK_GEN, _TS_WALK_GEN
    before = (_PY_WALK_GEN[0], _TS_WALK_GEN[0])

    su.invalidate_walk_caches()

    # The imported names must see the bump — they are the same objects.
    assert (_PY_WALK_GEN[0], _TS_WALK_GEN[0]) == (before[0] + 1, before[1] + 1)
    assert (su._PY_WALK_GEN[0], su._TS_WALK_GEN[0]) == (before[0] + 1, before[1] + 1)


# ── Mid-collection invalidation must not be resurrected ────────────────────

def _bump_after(delay: float, fn) -> threading.Thread:
    """Run *fn* (the post-write invalidation) *delay* seconds from now."""
    t = threading.Thread(target=lambda: (time.sleep(delay), fn()))
    t.start()
    return t


def test_walk_cache_not_repopulated_when_invalidated_mid_walk(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("x = 1\n")
    key = str(tmp_path)
    su._PY_WALK_CACHE.pop(key, None)

    real_walk = su.os.walk

    def slow_walk(root, *a, **kw):
        for item in real_walk(root, *a, **kw):
            time.sleep(0.3)          # the write lands during this window
            yield item

    monkeypatch.setattr(su.os, "walk", slow_walk)
    t = _bump_after(0.1, su.invalidate_walk_caches)
    files = su._walk_py_files(tmp_path, 5000)
    t.join()

    assert files, "the walk itself must still return its result to THIS caller"
    assert key not in su._PY_WALK_CACHE, (
        "a walk invalidated mid-collection re-populated the cache — the next "
        "caller will read a pre-write file list for a full TTL"
    )


def test_repo_file_index_not_repopulated_when_invalidated_mid_listing(tmp_path, monkeypatch):
    key = wt._file_index_key(str(tmp_path))
    wt._FILE_INDEX_CACHE.pop(key, None)

    def slow_listing(root):
        time.sleep(0.3)
        return ["stale.py"]

    monkeypatch.setattr(wt, "_git_list_tracked_files", slow_listing)
    t = _bump_after(0.1, lambda: wt.invalidate_repo_file_index(str(tmp_path)))
    paths = wt._repo_file_index(str(tmp_path))
    t.join()

    assert paths == ["stale.py"], "this caller still gets what it collected"
    assert key not in wt._FILE_INDEX_CACHE, (
        "a listing invalidated mid-collection re-populated the index — glob "
        "and the path suggester stay blind to the new file for a full TTL"
    )


def test_git_snapshot_not_repopulated_when_invalidated_mid_fetch(monkeypatch):
    acm._clear_git_cache()

    def slow_git(repo_root, *args):
        time.sleep(0.3)
        return "" if args[0] == "status" else "stale"

    monkeypatch.setattr(acm, "_run_git_raw", slow_git)
    t = _bump_after(0.1, acm._clear_git_cache)
    snap = acm.get_git_snapshot("/nonexistent-repo")
    t.join()

    assert snap.get("status") == "", "this caller still gets what it fetched"
    assert not acm._git_cache, (
        "a snapshot fetched across a write was cached — every later caller "
        "reads a clean tree for a full TTL after an edit"
    )


# ── A collect that starts AFTER invalidation is fresh and MUST cache ────────
# The guard has to distinguish "invalidation landed mid-collect" from "landed
# before it". Refusing to cache in the second case would disable the cache
# outright after every write, giving back the walk this exists to avoid.

def test_walk_started_after_invalidation_still_caches(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    key = str(tmp_path)

    su.invalidate_walk_caches()          # write finishes first...
    su._walk_py_files(tmp_path, 5000)     # ...then the walk starts

    assert key in su._PY_WALK_CACHE, (
        "a post-write walk refused to cache — the counter is being compared "
        "against the wrong baseline, so every write disables the cache"
    )


# ── Invalidation must reach SCOPED entries, not just the repo root ──────────

def test_invalidation_clears_subdirectory_walk_entries(tmp_path):
    """`find_symbol(..., search_path=sub)` caches under the SUBDIRECTORY.

    ``_resolve_search_root`` turns a ``search_path`` into a subtree Path, and
    ``_walk_repo_files`` keys the cache by whatever root it walked. An
    invalidator that popped one repo-root key therefore left every scoped entry
    behind — measured 1 of 2 live keys surviving — and that entry then answered a
    scoped find_symbol from a pre-write file list for the whole TTL. The
    generation counter cannot cover this: it only blocks an in-flight walk from
    storing, never an entry already stored.
    """
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "a.py").write_text("x = 1\n")

    su._PY_WALK_CACHE.clear()
    su._walk_py_files(tmp_path, 5000)   # whole-tree walk
    su._walk_py_files(sub, 5000)        # scoped walk -> subdirectory key
    assert {str(tmp_path), str(sub)} <= set(su._PY_WALK_CACHE)

    su.invalidate_walk_caches()

    assert not su._PY_WALK_CACHE, (
        f"scoped entries survived invalidation: {list(su._PY_WALK_CACHE)}"
    )


def test_invalidate_walk_caches_takes_no_root():
    """The signature is the fix: no caller can pick the wrong root again.

    The two call sites disagreed (``repo_root`` vs ``_effective_repo_root``) and
    neither reliably matched the searcher's own root.
    """
    import inspect
    assert list(inspect.signature(su.invalidate_walk_caches).parameters) == []


# ── Per-root git snapshot isolation ─────────────────────────────────────────

def test_git_snapshot_isolated_per_root(monkeypatch):
    """``get_git_snapshot(repo_root)`` must not leak repo A's snapshot to repo B.

    The webapp is a long-lived server process: ToolRegistry is created per-request,
    but the module-level ``_git_cache`` spans requests across different repos.
    Without per-root keying, the second request receives the first repo's branch
    name, commit hash, and modified-file list — injected into the system prompt
    and used as the rollback snapshot head_hash.
    """
    acm._clear_git_cache()

    def fake_git(repo_root, *args):
        # Return the repo_root in the output so each root gets a distinct snapshot.
        subcmd = args[0] if args else ""
        return f"{subcmd}-{repo_root}"

    monkeypatch.setattr(acm, "_run_git_raw", fake_git)

    snap_a = acm.get_git_snapshot("/repo-a")
    snap_b = acm.get_git_snapshot("/repo-b")

    # Same key would reuse the first snapshot — without per-root isolation
    # snap_b would be identical to snap_a.
    assert snap_a != snap_b, (
        "get_git_snapshot returned the SAME snapshot for two different roots — "
        "the cache is NOT keyed by repo_root"
    )

    # Both entries must survive in the cache independently.
    assert "/repo-a" in acm._git_cache
    assert "/repo-b" in acm._git_cache

    # A warm hit must return the SAME value, not re-fetch.
    acm._git_cache["/repo-b"] = (
        acm._git_cache["/repo-b"][0],
        {"branch": "warm-hit-b", "status": "", "head_hash": "b", "last_commit": ""},
    )
    snap_b2 = acm.get_git_snapshot("/repo-b")
    assert snap_b2["branch"] == "warm-hit-b", (
        "warm hit returned fresh data instead of cached — the entry was overwritten"
    )


def test_git_snapshot_cache_capped(tmp_path, monkeypatch):
    """FIFO cap (8 entries) must bound the per-root cache."""
    acm._clear_git_cache()

    def fake_git(repo_root, *args):
        subcmd = args[0] if args else ""
        return f"{subcmd}-{repo_root}"

    monkeypatch.setattr(acm, "_run_git_raw", fake_git)

    for i in range(12):
        acm.get_git_snapshot(f"/repo-{i}")

    assert len(acm._git_cache) == 8, (
        f"_git_cache grew to {len(acm._git_cache)} (cap=8) — _capped_put not working"
    )
    # The oldest 4 entries (/repo-0..3) must be evicted under FIFO.
    for i in range(4):
        assert f"/repo-{i}" not in acm._git_cache, (
            f"/repo-{i} survived FIFO eviction — oldest not evicted first"
        )


# ── Invalidation root alignment ──────────────────────────────────────────────

def test_invalidation_uses_effective_repo_root_consistently():
    """Both invalidation sites in ToolRegistry must use ``_effective_repo_root``.

    ``invalidate_walk_caches`` was fixed by removing the root argument entirely
    (verified by ``test_invalidate_walk_caches_takes_no_root`` above).
    ``invalidate_repo_file_index`` takes a root, which must be
    ``self._effective_repo_root`` at both call sites — the two disagreed
    (``repo_root`` vs ``_effective_repo_root``) and even the write tools'
    own documentation noted the latent mismatch.
    """
    import inspect
    import ast
    import textwrap
    from external_llm.agent.tool_registry import ToolRegistry

    for method_name in (
        "_invalidate_cache_after_write",
        "_invalidate_caches_unknown_scope",
    ):
        src = textwrap.dedent(inspect.getsource(getattr(ToolRegistry, method_name)))
        tree = ast.parse(src)
        # Collect every `invalidate_repo_file_index(...)` call argument.
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "invalidate_repo_file_index":
                    arg = ast.get_source_segment(src, node.args[0] if node.args else None)
                    assert arg and "self._effective_repo_root" in arg, (
                        f"{method_name}: invalidate_repo_file_index called with "
                        f"{arg!r} — must be self._effective_repo_root"
                    )
