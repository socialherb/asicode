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
            time.sleep(0.3)  # the write lands during this window
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
    key = wt.canonical_repo_key(str(tmp_path))
    wt._FILE_INDEX_CACHE.pop(key, None)

    def slow_listing(root):
        time.sleep(0.3)
        return ["stale.py"]

    # P5-2: the cache machinery now lives in common.repo_files (SSOT), so the
    # listing hook must be patched there (the write_tools_core names are
    # re-exports of the same objects, but patching the re-export binding would
    # not reach the real call inside cached_repo_file_list).
    import external_llm.common.repo_files as common_rf

    monkeypatch.setattr(common_rf, "git_list_repo_files", slow_listing)
    t = _bump_after(0.1, lambda: wt.invalidate_repo_file_index(str(tmp_path)))
    paths = wt._repo_file_index(str(tmp_path))
    t.join()

    assert paths == ["stale.py"], "this caller still gets what it collected"
    assert key not in wt._FILE_INDEX_CACHE, (
        "a listing invalidated mid-collection re-populated the index — glob "
        "and the path suggester stay blind to the new file for a full TTL"
    )


def test_git_snapshot_not_repopulated_when_invalidated_mid_fetch(monkeypatch):
    acm._git_cache.clear()
    acm._git_dirty_since.clear()
    key = acm.canonical_repo_key("/nonexistent-repo")

    def slow_git(repo_root, *args):
        time.sleep(0.3)
        return "" if args[0] == "status" else "stale"

    monkeypatch.setattr(acm, "_run_git_raw", slow_git)
    t = _bump_after(0.1, acm._clear_git_cache)
    snap = acm.get_git_snapshot("/nonexistent-repo")
    t.join()

    assert snap.get("status") == "", "this caller still gets what it fetched"
    assert key not in acm._git_cache, (
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

    su.invalidate_walk_caches()  # write finishes first...
    su._walk_py_files(tmp_path, 5000)  # ...then the walk starts

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
    su._walk_py_files(tmp_path, 5000)  # whole-tree walk
    su._walk_py_files(sub, 5000)  # scoped walk -> subdirectory key
    assert {str(tmp_path), str(sub)} <= set(su._PY_WALK_CACHE)

    su.invalidate_walk_caches()

    assert not su._PY_WALK_CACHE, f"scoped entries survived invalidation: {list(su._PY_WALK_CACHE)}"


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
    acm._git_cache.clear()
    acm._git_dirty_since.clear()

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
        "get_git_snapshot returned the SAME snapshot for two different roots — the cache is NOT keyed by repo_root"
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


def test_git_snapshot_status_bounded_at_ssot(monkeypatch):
    """The SSOT must truncate the display status for every prompt consumer.

    _build_session_context (system prompt), EnhancedContextBuilder and
    SuperContextBuilder all inject the snapshot's status verbatim; the only
    guard they share is the truncation HERE. A consumer-side slice would
    silently miss the next injection site, so the bound is tested at the SSOT.
    """
    acm._git_cache.clear()
    acm._git_dirty_since.clear()
    long_status = "\n".join(f" M file{i:04d}.py" for i in range(2000))  # ~34 KB
    assert len(long_status) > acm.GIT_STATUS_MAX_CHARS * 3

    def fake_git(repo_root, *args):
        subcmd = args[0] if args else ""
        if subcmd == "status":
            return long_status
        return f"{subcmd}-ok"

    monkeypatch.setattr(acm, "_run_git_raw", fake_git)

    snap = acm.get_git_snapshot("/repo-trunc")
    assert len(snap["status"]) == acm.GIT_STATUS_MAX_CHARS, (
        "a >5000-char worktree status leaked past the SSOT into prompt consumers"
    )
    assert snap["status"].startswith(" M file0000.py")
    # Truthiness (has_changes / `if status:`) must survive truncation.
    assert bool(snap["status"])

    # A warm hit serves the same bounded value, not an unbound re-fetch.
    snap2 = acm.get_git_snapshot("/repo-trunc")
    assert snap2["status"] == snap["status"]
    # The cache entry itself stores the bounded value.
    assert len(acm._git_cache["/repo-trunc"][1]["status"]) == acm.GIT_STATUS_MAX_CHARS

    # Clean tree (empty status) stays empty — no marker injected.
    def fake_git_clean(repo_root, *args):
        return "" if args[0] == "status" else f"{args[0]}-ok"

    monkeypatch.setattr(acm, "_run_git_raw", fake_git_clean)
    assert acm.get_git_snapshot("/repo-clean")["status"] == ""


def test_git_snapshot_cache_capped(tmp_path, monkeypatch):
    """FIFO cap (8 entries) must bound the per-root cache."""
    acm._git_cache.clear()
    acm._git_dirty_since.clear()

    def fake_git(repo_root, *args):
        subcmd = args[0] if args else ""
        return f"{subcmd}-{repo_root}"

    monkeypatch.setattr(acm, "_run_git_raw", fake_git)

    for i in range(12):
        acm.get_git_snapshot(f"/repo-{i}")

    assert len(acm._git_cache) == 8, f"_git_cache grew to {len(acm._git_cache)} (cap=8) — _capped_put not working"
    # The oldest 4 entries (/repo-0..3) must be evicted under FIFO.
    for i in range(4):
        assert f"/repo-{i}" not in acm._git_cache, f"/repo-{i} survived FIFO eviction — oldest not evicted first"


# ── Invalidation root alignment ──────────────────────────────────────────────


def test_invalidation_uses_effective_repo_root_consistently():
    """The unknown-scope full drop must use ``_effective_repo_root``; the
    known-scope path must invalidate per touched path instead.

    ``invalidate_walk_caches`` was fixed by removing the root argument entirely
    (verified by ``test_invalidate_walk_caches_takes_no_root`` above).
    ``invalidate_repo_file_index`` takes a root, which must be
    ``self._effective_repo_root`` at the unknown-scope call site — the two
    disagreed (``repo_root`` vs ``_effective_repo_root``) and even the write
    tools' own documentation noted the latent mismatch.

    The known-scope site (``_invalidate_cache_after_write``) must NOT call the
    wholesale drop at all: atomic writers already pop + gen-bump per path via
    the atomic funnel (``atomic_io -> invalidate_for_written_path``), and the
    per-path call covers the one non-atomic writer (apply_patch's git-apply
    subprocess). It must normalize relative touched paths against
    ``self._effective_repo_root`` so the resolve inside
    ``invalidate_for_written_path`` matches the canonical cache key.
    """
    import ast
    import inspect
    import textwrap

    from external_llm.agent.tool_registry import ToolRegistry

    # Unknown-scope: the wholesale drop is required (bash writes bypass the
    # atomic funnel), and its root must be the effective repo root.
    src = textwrap.dedent(inspect.getsource(ToolRegistry._invalidate_caches_unknown_scope))
    tree = ast.parse(src)
    _full_drops = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "invalidate_repo_file_index"
    ]
    assert _full_drops, "_invalidate_caches_unknown_scope lost its wholesale repo file index drop"
    for node in _full_drops:
        arg = ast.get_source_segment(src, node.args[0] if node.args else None)
        assert arg and "self._effective_repo_root" in arg, (
            f"_invalidate_caches_unknown_scope: invalidate_repo_file_index called with "
            f"{arg!r} — must be self._effective_repo_root"
        )

    # Known-scope: no wholesale drop; per-path invalidation only.
    src2 = textwrap.dedent(inspect.getsource(ToolRegistry._invalidate_cache_after_write))
    tree2 = ast.parse(src2)
    assert not [
        n
        for n in ast.walk(tree2)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "invalidate_repo_file_index"
    ], (
        "_invalidate_cache_after_write must not wholesale-drop the repo file "
        "index — the atomic funnel + per-path invalidate_for_written_path cover it"
    )
    _per_path = [
        n
        for n in ast.walk(tree2)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "invalidate_for_written_path"
    ]
    assert _per_path, "_invalidate_cache_after_write must invalidate the repo file index per touched path"


# ── The git snapshot must be invalidated by a real write, not only by tests ──


def _init_git_repo(root):
    import subprocess

    def _run(*a):
        return subprocess.run(a, cwd=root, capture_output=True, text=True, check=False)

    _run("git", "init", "-q")
    _run("git", "config", "user.email", "t@example.com")
    _run("git", "config", "user.name", "t")
    (root / "app.py").write_text("original\n", encoding="utf-8")
    _run("git", "add", "-A")
    _run("git", "commit", "-qm", "init")


def test_write_tool_invalidates_the_git_snapshot(tmp_path, monkeypatch):
    """A successful write must make the next get_git_snapshot see its own edit.

    ``_clear_git_cache`` existed with a full generation-counter protocol and a
    docstring claiming it was "registered as a write-success callback" — and no
    shipping code ever registered it. Every caller was a test, so the snapshot
    kept serving pre-write state for the whole 10 s TTL. Three consumers read
    it, including the "Modified files (git status)" block put in front of the
    model, which therefore described the repo as it was BEFORE the agent's own
    writes.

    This asserts the end-to-end behaviour rather than the wiring, because the
    wiring is exactly what moved: the correct carrier is the central mutation
    point in ``_dispatch_impl``, not the callback list (both clone paths reset
    it, while this cache is module-global and shared across clones).
    """
    from external_llm.agent.tool_registry import AgentConfig, ToolRegistry

    _init_git_repo(tmp_path)
    acm._git_cache.clear()
    acm._git_dirty_since.clear()
    # P3 coalescing: the post-write read below lands MILLISECONDS after the
    # write, i.e. inside the default 1 s window where the pre-write entry is
    # served by design. This test asserts the invalidation WIRING (a write
    # must make the next read rebuild), so shrink the window to zero — the
    # window semantics themselves are covered by the dedicated tests below.
    monkeypatch.setattr(acm, "_GIT_REBUILD_COALESCE_S", 0.0)

    # Populate the cache with the clean pre-write state.
    assert acm.get_git_snapshot(str(tmp_path))["status"] == ""

    reg = ToolRegistry(repo_root=str(tmp_path), config=AgentConfig())
    result = reg.dispatch(
        "edit_text",
        {"file_path": "app.py", "old_text": "original", "new_text": "edited"},
    )
    assert result.ok, result.error
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "edited\n"

    # Within the TTL: only invalidation can make this non-empty.
    status = acm.get_git_snapshot(str(tmp_path))["status"]
    assert "app.py" in status, f"git snapshot still pre-write after a successful edit: {status!r}"


def test_mutating_bash_invalidates_the_git_snapshot(tmp_path, monkeypatch):
    """The same guarantee for a mutating NON-write tool.

    bash changes git state (``git commit``, ``rm``) without going through any
    member of ``_WRITE_TOOLS``, so an invalidation hung off the write tools
    alone would leave this path stale. It shares the ``_should_invalidate``
    branch precisely so both are covered by one call.
    """
    from external_llm.agent.tool_registry import AgentConfig, ToolRegistry

    _init_git_repo(tmp_path)
    acm._git_cache.clear()
    acm._git_dirty_since.clear()
    # Same zero-window rationale as test_write_tool_invalidates_the_git_snapshot.
    monkeypatch.setattr(acm, "_GIT_REBUILD_COALESCE_S", 0.0)
    assert acm.get_git_snapshot(str(tmp_path))["status"] == ""

    reg = ToolRegistry(repo_root=str(tmp_path), config=AgentConfig())
    # `cp`, not a `>` redirect — the latter is refused by the destructive-op
    # gate, which would make this test assert nothing about invalidation.
    result = reg.dispatch("bash", {"command": "cp app.py copy.py"})
    assert result.ok, result.error
    assert reg._tool_call_mutates("bash", {"command": "cp app.py copy.py"})

    status = acm.get_git_snapshot(str(tmp_path))["status"]
    assert "copy.py" in status, f"git snapshot still pre-write after a mutating bash call: {status!r}"


# ── P3 coalesced invalidation: dirty-stamp window semantics ──────────────────
# _clear_git_cache no longer empties the dict; it stamps the root dirty and
# get_git_snapshot serves the pre-write entry inside the window, rebuilds
# past it. Window control via _GIT_REBUILD_COALESCE_S monkeypatch (read at
# call time): 60.0 = "every read is inside the window", 0.0 = "always past".


def _tagged_git(monkeypatch, tag: str) -> dict:
    """Fake _run_git_raw returning deterministic per-command values tagged by
    snapshot generation (parallel pool threads make call-ORDER nondeterministic,
    so sequence-number labels would be flaky). Swap the tag to distinguish
    populate vs post-write rebuilds."""
    calls = {"n": 0}

    def fake_git(repo_root, *args):
        calls["n"] += 1
        subcmd = args[0] if args else ""
        return f"{subcmd}-{tag}"

    monkeypatch.setattr(acm, "_run_git_raw", fake_git)
    return calls


def test_git_snapshot_within_coalesce_window_serves_prewrite_entry(monkeypatch):
    """A read <1 s after a write must NOT pay a rebuild — it gets the entry
    the cache already had. This is the whole point of P3: write→read gaps
    inside the window are the burst case (parallel subagents, webapp
    requests), not the freshness case (LLM-call-separated turn boundaries).
    """
    acm._git_cache.clear()
    acm._git_dirty_since.clear()
    _tagged_git(monkeypatch, "pre")
    monkeypatch.setattr(acm, "_GIT_REBUILD_COALESCE_S", 60.0)

    prewrite = acm.get_git_snapshot("/repo-x")  # 1 rebuild -> cached
    assert prewrite["status"] == "status-pre"
    rebuild_calls = _tagged_git(monkeypatch, "post")  # would label a rebuild

    acm._clear_git_cache("/repo-x")  # a write lands
    after = acm.get_git_snapshot("/repo-x")

    assert after == prewrite, "within-window read must serve the pre-write entry, not rebuild"
    assert rebuild_calls["n"] == 0, "within-window read must not spawn git subprocesses"
    assert acm._git_dirty_since.get("/repo-x") is not None, (
        "the dirty stamp must survive until a past-window read rebuilds"
    )


def test_git_snapshot_rebuilds_fresh_past_coalesce_window(monkeypatch):
    """A read past the window rebuilds fresh — the pre-write entry must not
    outlive _GIT_REBUILD_COALESCE_S, and the stamp is popped by the store so
    later reads are plain hits again."""
    acm._git_cache.clear()
    acm._git_dirty_since.clear()
    _tagged_git(monkeypatch, "pre")
    monkeypatch.setattr(acm, "_GIT_REBUILD_COALESCE_S", 0.0)

    acm.get_git_snapshot("/repo-y")  # populate
    post_calls = _tagged_git(monkeypatch, "post")
    acm._clear_git_cache("/repo-y")

    snap = acm.get_git_snapshot("/repo-y")  # past window -> rebuild
    assert snap["status"] == "status-post", "past-window read must rebuild fresh, not serve the pre-write entry"
    assert post_calls["n"] == 3, "one fresh snapshot = 3 git commands"
    assert "/repo-y" not in acm._git_dirty_since, (
        "the post-write rebuild stores fresh data, so the stamp must be popped"
    )

    snap2 = acm.get_git_snapshot("/repo-y")  # must be a hit now
    assert snap2 == snap, "the hit must return the same post-write entry"
    assert post_calls["n"] == 3, (
        "a clean entry must not rebuild on every read — the popped stamp must not leave the root permanently dirty"
    )


def test_clear_git_cache_root_scoped_stamp(monkeypatch):
    """A write to repo A stamps only A: repo B's entry keeps serving hits."""
    acm._git_cache.clear()
    acm._git_dirty_since.clear()
    calls = _tagged_git(monkeypatch, "x")
    monkeypatch.setattr(acm, "_GIT_REBUILD_COALESCE_S", 0.0)

    acm.get_git_snapshot("/repo-a")
    acm.get_git_snapshot("/repo-b")
    assert calls["n"] == 6, "2 snapshots = 3 git commands each"

    acm._clear_git_cache("/repo-a")  # write to repo A only
    acm.get_git_snapshot("/repo-a")  # rebuild (+3 -> 9)
    acm.get_git_snapshot("/repo-b")  # must HIT — untouched by A's write
    assert calls["n"] == 9, "a write to repo A must not force repo B's entry to rebuild"


def test_clear_git_cache_no_arg_stamps_all_roots(monkeypatch):
    """The no-arg (legacy/global) call site must invalidate every root."""
    acm._git_cache.clear()
    acm._git_dirty_since.clear()
    calls = _tagged_git(monkeypatch, "x")
    monkeypatch.setattr(acm, "_GIT_REBUILD_COALESCE_S", 0.0)

    acm.get_git_snapshot("/repo-a")
    acm.get_git_snapshot("/repo-b")
    acm._clear_git_cache()  # caller cannot name a root
    acm.get_git_snapshot("/repo-a")  # rebuild (+3 -> 9)
    acm.get_git_snapshot("/repo-b")  # rebuild (+3 -> 12)
    assert calls["n"] == 12, "no-arg clear must stamp every cached root, not leave some serving stale"


def test_clear_git_cache_stamps_only_cached_roots():
    """A root with no cache entry needs no stamp — a read misses and rebuilds
    anyway, and stamps for never-read roots would just accumulate."""
    acm._git_cache.clear()
    acm._git_dirty_since.clear()

    acm._clear_git_cache("/never-cached-root")
    assert not acm._git_dirty_since, "clearing a root that has no entry must not leave a stamp behind"


# ── Known-scope invalidation: per-path (not wholesale) ───────────────────────


class _PostWriteRegistry:
    """Minimal ToolRegistry stand-in for _invalidate_cache_after_write."""

    def __init__(self, root):
        self.root = str(root)

    @property
    def _effective_repo_root(self):
        return self.root


def test_invalidate_cache_after_write_covers_non_atomic_writer(tmp_path, monkeypatch):
    """A write that bypasses the atomic funnel must still invalidate per path.

    apply_patch's git-apply subprocess writes without going through
    ``atomic_write_text``, so ``invalidate_for_written_path`` never fires for
    it. ``_invalidate_cache_after_write`` is that writer's only invalidator —
    it must pop the listing for the touched repo-relative path.
    """
    import external_llm.common.repo_files as common_rf
    from external_llm.agent.tool_registry import ToolRegistry

    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    key = common_rf.canonical_repo_key(str(tmp_path))
    common_rf._FILE_INDEX_CACHE.pop(key, None)

    def fake_listing(root):
        return ["app.py"]

    # The listing hook must be patched at the SSOT (common.repo_files) — the
    # write_tools names are re-exports that the real call does not see.
    monkeypatch.setattr(common_rf, "git_list_repo_files", fake_listing)

    assert common_rf.cached_repo_file_list(str(tmp_path)) == ["app.py"]
    assert key in common_rf._FILE_INDEX_CACHE

    # Simulate the git-apply write: plain open(), no atomic funnel.
    (tmp_path / "app.py").write_text("x = 2\n", encoding="utf-8")
    assert key in common_rf._FILE_INDEX_CACHE, "sanity: a plain write must NOT auto-invalidate (funnel bypassed)"

    reg = _PostWriteRegistry(tmp_path)
    ToolRegistry._invalidate_cache_after_write(reg, ["app.py"])

    assert key not in common_rf._FILE_INDEX_CACHE, "per-path invalidation must cover the non-atomic (git-apply) writer"


def test_invalidate_cache_after_write_no_extra_bump_after_atomic_write(tmp_path, monkeypatch):
    """Atomic writers must not pay a second gen bump in _invalidate_cache_after_write.

    The atomic funnel already popped the listing and bumped the generation;
    the per-path call that follows must be a no-op (nothing cached -> no bump).
    A second unconditional bump would be the double invalidation this refactor
    removed.
    """
    import external_llm.common.repo_files as common_rf
    from external_llm.agent.tool_registry import ToolRegistry
    from external_llm.common.atomic_io import atomic_write_text

    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    key = common_rf.canonical_repo_key(str(tmp_path))
    common_rf._FILE_INDEX_CACHE.pop(key, None)

    def fake_listing(root):
        return ["app.py"]

    monkeypatch.setattr(common_rf, "git_list_repo_files", fake_listing)
    assert common_rf.cached_repo_file_list(str(tmp_path)) == ["app.py"]
    assert key in common_rf._FILE_INDEX_CACHE

    gen0 = common_rf._FILE_INDEX_GEN
    atomic_write_text(str(tmp_path / "app.py"), "x = 2\n")
    assert key not in common_rf._FILE_INDEX_CACHE
    gen1 = common_rf._FILE_INDEX_GEN
    assert gen1 == gen0 + 1, "atomic funnel must bump the generation exactly once"

    reg = _PostWriteRegistry(tmp_path)
    ToolRegistry._invalidate_cache_after_write(reg, ["app.py"])
    assert gen1 == common_rf._FILE_INDEX_GEN, (
        "per-path invalidation after an atomic write must NOT bump the generation again (double invalidation)"
    )


def test_unknown_scope_invalidation_drops_the_facade_graph():
    """B1: ``_invalidate_caches_unknown_scope`` must invalidate the RG facade
    graph, not only the CallGraphIndexer.

    ``cgi.invalidate()`` drops the indexer, but the RepositoryGraph inside the
    facade (``_graph``) is a separate build serving get_symbol / get_importers /
    get_file_dependencies / get_symbols_in_file. The wholesale unknown-scope
    drop covered only the first, so a bash-created file stayed invisible to
    RG-backed queries until the next lazy rebuild — the same stale-read class
    this method exists to prevent. (The known-scope path has always been
    symmetric: ``_call_graph.invalidate_files``.)
    """
    import ast
    import inspect
    import textwrap

    from external_llm.agent.tool_registry import ToolRegistry

    src = textwrap.dedent(inspect.getsource(ToolRegistry._invalidate_caches_unknown_scope))
    tree = ast.parse(src)

    facade_drops = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "invalidate"
    ]
    assert facade_drops, (
        "_invalidate_caches_unknown_scope lost its facade graph drop — mutating bash left RG-backed queries stale (B1)"
    )
    assert any(ast.get_source_segment(src, n.func.value) == "self._call_graph" for n in facade_drops), (
        "_invalidate_caches_unknown_scope must call self._call_graph.invalidate() "
        "— cgi.invalidate() alone leaves the facade's RG graph warm (B1)"
    )
