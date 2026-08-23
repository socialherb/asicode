"""Unit tests for RepositoryGraph.build()'s process-wide extraction cache.

The cache (``_extract_cache``) makes repeated builds in one process re-parse
only files whose (mtime_ns, size) changed; everything else is injected from
the cached ``extract_file`` payload.  These tests pin the two contracts:
bit-for-bit identical graphs, and re-extraction exactly on (and only on)
mtime/size changes.
"""

import os
import shutil
import tempfile
import textwrap
from pathlib import Path

import pytest

from external_llm.graph import repository_graph as rg_module
from external_llm.graph.repository_graph import (
    _EXTRACT_CACHE_MAX_ENTRIES,
    RepositoryGraph,
    _extract_cache,
    _extract_cache_key,
    _gc_extract_cache,
)


def _set_cap(monkeypatch, cap: int) -> None:
    """Patch BOTH the module constant (GC deficit/trigger) and the live
    RootCache instance cap (admission) — decoupled since 2026-08-12."""
    monkeypatch.setattr(rg_module, "_EXTRACT_CACHE_MAX_ENTRIES", cap)
    monkeypatch.setattr(rg_module._extract_cache, "cap", cap)


@pytest.fixture(autouse=True)
def isolated_extract_cache():
    """Save/restore the process-wide cache and gc rate-limit around every test."""
    saved = dict(_extract_cache)
    saved_deficit = _extract_cache._gc_deficit
    _extract_cache.clear()
    _extract_cache._gc_deficit = 0
    yield
    _extract_cache.clear()
    _extract_cache.update(saved)
    _extract_cache._gc_deficit = saved_deficit


def _make_repo(files: dict) -> str:
    d = tempfile.mkdtemp(prefix="test_rg_cache_")
    for rel_path, source in files.items():
        full = Path(d) / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(textwrap.dedent(source))
    return d


def _graph_snapshot(graph):
    """Field-level snapshot — SymbolNode is a plain class (no __eq__)."""
    symbols = {}
    for uid, s in sorted(graph.symbols.items()):
        symbols[uid] = (
            s.name,
            s.qualname,
            s.module,
            s.file_path,
            s.kind,
            s.start_line,
            s.end_line,
            s.language,
            s.signature_hash,
            s.docstring,
            s.signature,
            tuple(s.bases or ()),
        )
    edges = [(e.caller, e.callee, e.file_path, e.line) for e in graph.call_edges]
    imports = [(e.importer, e.imported, e.import_type) for e in graph.import_edges]
    return symbols, edges, imports


def _count_extract_calls(monkeypatch):
    """Return a counter object: counts RepositoryGraph.extract_file calls."""
    calls = {"n": 0}
    orig = RepositoryGraph.extract_file

    def counting(self, path):
        calls["n"] += 1
        return orig(self, path)

    monkeypatch.setattr(RepositoryGraph, "extract_file", counting)
    return calls


# ── Cache hits ───────────────────────────────────────────────────────────────


def test_py_files_accessor_lists_all_walked_py_files():
    """``py_files`` exposes the build's UNCAPPED walked list (repo-relative).

    ``build()`` walks without a file cap (unlike the structural scanners'
    SCAN_FILE_CAP) — the structural gate unions ``graph.py_files`` into its
    cross-file-ref input precisely so references beyond the cap still
    suppress dead-code candidates.  Pin: every walked .py file, in walk
    order, and nothing else (no non-py, no unwalked).
    """
    repo = _make_repo(
        {
            "a.py": "def fa():\n    return 1\n",
            "sub/b.py": "def fb():\n    return 1\n",
            "c.ts": "export const c = 1;\n",
        }
    )
    graph = RepositoryGraph(repo)
    graph.build(collect_imported_names=True)
    assert graph.py_files == ["a.py", "sub/b.py"]  # .ts never walked as py
    # Exactly the _py_stamps rel list — the accessor is its public face.
    assert graph.py_files == [rel for rel, _path, _st in graph._py_stamps]


def test_py_files_populated_in_plain_build_mode():
    """``py_files`` is populated for EVERY build mode, not just the gate's.

    The structural-scan TOOL path (analysis_tools.py) unions the graph's
    uncapped py list into its cross-file-ref input, and its graph is built
    in plain mode (facade → GraphBuilder.build_repo_graph → build()) — a
    plain build leaving ``py_files`` empty would make the tool-side union
    inert (2026-08-11).
    """
    repo = _make_repo(
        {
            "a.py": "def fa():\n    return 1\n",
            "b.py": "def fb():\n    return 1\n",
        }
    )
    graph = RepositoryGraph(repo)
    graph.build()  # plain mode — no collect_imported_names
    assert graph.py_files == ["a.py", "b.py"]
    assert graph.imported_names == set()  # names sweep still off
    assert graph.cache_stats["total"] == 2  # walk stats are real now


def _record_parse_cache_sizing(monkeypatch):
    """Wrap parse_cache.ensure_capacity, returning the recorded call args."""
    import external_llm.analysis.parse_cache as pc

    calls: list = []
    _orig = pc.ensure_capacity

    def recording(n):
        calls.append(n)
        return _orig(n)

    monkeypatch.setattr(pc, "ensure_capacity", recording)
    return calls


def test_build_with_names_sizes_parse_cache_to_walked_py_count(monkeypatch):
    """``build(collect_imported_names=True)`` sizes the shared parse cache to
    the UNCAPPED walked py count before its name pass (P2 2026-08-11).

    The graph walk never truncates (unlike the scanner lists' SCAN_FILE_CAP),
    and the name pass parses every walked py file through the cache — a
    default-sized cache would thrash on any repo bigger than it, and the
    same process's cross-file-ref pass re-parses importers right after.
    """
    repo = _make_repo(
        {
            "a.py": "def fa():\n    return 1\n",
            "b.py": "def fb():\n    return 1\n",
            "c.ts": "export const c = 1;\n",  # non-py never counts
        }
    )
    calls = _record_parse_cache_sizing(monkeypatch)
    graph = RepositoryGraph(repo)
    graph.build(collect_imported_names=True)
    assert calls == [2], "sized to the 2 walked py files, excluding c.ts"


def test_plain_build_does_not_size_parse_cache(monkeypatch):
    """Plain ``build()`` never parses via the shared cache (tree-sitter +
    its own extract cache) — sizing would be work the app does not consume."""
    repo = _make_repo({"a.py": "def fa():\n    return 1\n"})
    calls = _record_parse_cache_sizing(monkeypatch)
    graph = RepositoryGraph(repo)
    graph.build()
    assert calls == []


def test_py_files_stays_live_across_incremental_edits():
    """``remove_file``/``reparse_file`` maintain the walk stamp.

    The facade's incremental path (invalidate_files) must not leave
    ``py_files`` stale: a removed file would linger, and a re-parsed file
    would be stamped twice, until the next full build (2026-08-11).
    """
    repo = _make_repo(
        {
            "a.py": "def fa():\n    return 1\n",
            "b.py": "def fb():\n    return 1\n",
        }
    )
    graph = RepositoryGraph(repo)
    graph.build()
    graph.remove_file("a.py")
    assert graph.py_files == ["b.py"]
    graph.reparse_file(os.path.join(repo, "a.py"))
    assert graph.py_files == ["b.py", "a.py"]  # re-added once, no duplicate


def test_rebuild_serves_unchanged_files_from_cache(monkeypatch):
    repo = _make_repo(
        {
            "a.py": "def foo(): pass\n",
            "b.py": "def bar(): pass\n",
        }
    )
    try:
        RepositoryGraph(repo).build()  # cold build — parses everything
        calls = _count_extract_calls(monkeypatch)
        g2 = RepositoryGraph(repo)
        g2.build()
        assert calls["n"] == 0  # every file served from cache
        assert "a.py:foo" in g2.symbols
        assert "b.py:bar" in g2.symbols
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_cached_rebuild_bit_for_bit_identical():
    repo = _make_repo(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            import os
            from pkg import other

            def helper(x):
                return x

            class C:
                def m(self):
                    helper(1)
        """,
            "pkg/other.py": "def other(): pass\n",
        }
    )
    try:
        g1 = RepositoryGraph(repo)
        g1.build()
        g2 = RepositoryGraph(repo)
        g2.build()  # fully served from cache
        assert _graph_snapshot(g2) == _graph_snapshot(g1)
        # Edge ORDER identical too — injection follows the same sorted walk.
        assert [e.callee for e in g2.call_edges] == [e.callee for e in g1.call_edges]
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ── Invalidation (re-extraction) ─────────────────────────────────────────────


def test_mtime_change_triggers_reextract(monkeypatch):
    repo = _make_repo(
        {
            "a.py": "def foo(): pass\n",
            "b.py": "def bar(): pass\n",
        }
    )
    try:
        RepositoryGraph(repo).build()
        a = Path(repo) / "a.py"
        st = a.stat()
        os.utime(a, ns=(st.st_atime_ns, st.st_mtime_ns + 1000))
        calls = _count_extract_calls(monkeypatch)
        g = RepositoryGraph(repo)
        g.build()
        assert calls["n"] == 1  # only the touched file re-parsed
        assert "a.py:foo" in g.symbols
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_content_change_triggers_reextract(monkeypatch):
    repo = _make_repo({"a.py": "def foo(): pass\n"})
    try:
        RepositoryGraph(repo).build()
        (Path(repo) / "a.py").write_text("def foo(): pass\ndef new_fn(): pass\n")
        calls = _count_extract_calls(monkeypatch)
        g = RepositoryGraph(repo)
        g.build()
        assert calls["n"] == 1
        assert "a.py:foo" in g.symbols
        assert "a.py:new_fn" in g.symbols  # fresh extraction visible
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_deleted_file_absent_on_cached_rebuild():
    repo = _make_repo(
        {
            "a.py": "def foo(): pass\n",
            "b.py": "def bar(): pass\n",
        }
    )
    try:
        RepositoryGraph(repo).build()
        (Path(repo) / "a.py").unlink()
        g = RepositoryGraph(repo)
        g.build()
        assert "a.py:foo" not in g.symbols
        assert "b.py:bar" in g.symbols
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ── reparse_file / GC ────────────────────────────────────────────────────────


def test_reparse_file_refreshes_cache(monkeypatch):
    repo = _make_repo({"a.py": "def foo(): pass\n"})
    try:
        g1 = RepositoryGraph(repo)
        g1.build()
        (Path(repo) / "a.py").write_text("def foo(): pass\ndef extra(): pass\n")
        g1.reparse_file(str(Path(repo) / "a.py"))
        assert "a.py:extra" in g1.symbols

        calls = _count_extract_calls(monkeypatch)
        g2 = RepositoryGraph(repo)
        g2.build()
        assert calls["n"] == 0  # reparse refreshed the cache — rebuild is free
        assert "a.py:extra" in g2.symbols
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_gc_evicts_dead_entries_but_keeps_live_when_over_cap(monkeypatch):
    """Admission control: gc reclaims deleted sources, never clears live entries.

    The cap is enforced by the INSERT site refusing new entries (admission),
    not by gc dropping the whole cache.  gc's only job is to free slots held by
    deleted files so live files can be admitted.  Pinned because an earlier
    version cleared the whole cache when ``len >= cap``, thrashing to 0% hits
    on every rebuild of a repo larger than the cap.  (The pre-fix version of
    THIS test keyed entries by a bare string, so ``key[1]`` was a single char
    and ``os.stat`` failed for every entry — it passed for the wrong reason.)
    """
    monkeypatch.setattr(
        rg_module,
        "_EXTRACT_CACHE_MAX_ENTRIES",
        2,
    )
    tmp = Path(tempfile.mkdtemp(prefix="test_rg_gc_"))
    try:
        repo_root = str(tmp)
        alive = tmp / "alive.py"
        alive.write_text("x = 1\n")
        dead = tmp / "dead.py"
        dead.write_text("x = 1\n")
        extra = tmp / "extra.py"
        extra.write_text("x = 1\n")
        for p in (alive, dead, extra):
            key = _extract_cache_key(repo_root, str(p))
            _extract_cache[key] = (p.stat().st_mtime_ns, 5, {})
        dead.unlink()
        _gc_extract_cache()
        assert _extract_cache_key(repo_root, str(dead)) not in _extract_cache
        # Both live survivors stay — admission control, NOT a whole-cache clear.
        assert len(_extract_cache) == 2
        assert _extract_cache_key(repo_root, str(alive)) in _extract_cache
        assert _extract_cache_key(repo_root, str(extra)) in _extract_cache
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_admission_control_keeps_stable_cap_subset_across_rebuilds(monkeypatch):
    """Beyond the cap, the cache holds a stable first ``cap`` entries that rebuilds hit.

    Regression: the old whole-cache clear thrashed to 0% hits every rebuild
    because the walk revisits every file each build (clear -> refill -> clear).
    Admission control keeps the first ``cap`` live files instead, so a rebuild
    re-parses ONLY the files beyond the cap (hit rate cap/N, stable).
    """
    _set_cap(monkeypatch, 3)
    files = {f"mod_{i}.py": f"v{i} = {i}\n" for i in range(6)}  # 6 files, cap 3
    repo = _make_repo(files)
    try:
        parse_calls = {"n": 0}
        orig_extract = RepositoryGraph.extract_file

        def counting_extract(self, file_path):
            parse_calls["n"] += 1
            return orig_extract(self, file_path)

        monkeypatch.setattr(RepositoryGraph, "extract_file", counting_extract)
        # build #1: parse all 6, admit first 3
        RepositoryGraph(repo).build()
        assert len(_extract_cache) == 3
        assert parse_calls["n"] == 6
        cached_keys = set(_extract_cache)
        # build #2: first 3 served from cache (0 re-parse), last 3 re-parsed.
        # Old clear behavior would have re-parsed all 6 (0 cache hits).
        parse_calls["n"] = 0
        RepositoryGraph(repo).build()
        assert len(_extract_cache) == 3
        assert set(_extract_cache) == cached_keys  # stable subset, not churned
        assert parse_calls["n"] == 3  # only beyond-cap files re-parsed
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_cache_snapshot_manifest_mirrors_files_under_admission_cap(monkeypatch, tmp_path):
    """On a capped repo the saved manifest/files/names sections stay in sync.

    Merge-preserving (P0, 2026-08-12): the manifest CONVERGES to the full
    walked set (beyond-cap fresh parses are persisted via ``_pending_snapshot``),
    so 6 files with a cap of 3 still land all 6 in the snapshot.  The
    manifest/files/names sets must mirror each other exactly.
    """
    import json

    _set_cap(monkeypatch, 3)
    repo = tmp_path / "repo"
    repo.mkdir()
    for i in range(6):  # 6 files, cap 3
        (repo / f"mod_{i}.py").write_text(f"v{i} = {i}\n")
    g = RepositoryGraph(str(repo))
    g.build(collect_imported_names=True)  # writes the cache snapshot
    blob = json.loads((repo / ".cache" / "structural_graph_v1.json").read_text())
    assert len(blob["manifest"]) == 6, "merge-preserving snapshot must cover ALL walked files"
    assert len(blob["files"]) == 6
    assert len(blob["imported_names"]) == 6
    assert set(blob["manifest"]) == set(blob["files"]) == set(blob["imported_names"])


def test_overflow_reparse_does_not_rewrite_snapshot_or_count_as_changed(monkeypatch, tmp_path):
    """P0: on a capped repo a no-change rebuild must not rewrite the JSON.

    Files beyond the admission cap re-parse on EVERY build by design (they are
    never cached in-process) but the merge-preserving snapshot PERSISTS their
    payloads (``_pending_snapshot``), so build #1 counts all 10 re-parses as
    "changed" and converges the manifest to 10.  Build #2 (nothing changed) is
    then fully served: hit == 10, changed == 0, and the snapshot is NOT
    rewritten (byte-identical).
    """
    import json

    _set_cap(monkeypatch, 5)
    repo = tmp_path / "repo"
    repo.mkdir()
    for i in range(10):  # 10 files, cap 5 — 5 overflow files re-parse every build
        (repo / f"mod_{i}.py").write_text(f"v{i} = {i}\n")
    g1 = RepositoryGraph(str(repo))
    g1.build(collect_imported_names=True)
    blob_path = repo / ".cache" / "structural_graph_v1.json"
    first = blob_path.read_bytes()
    assert g1.cache_stats["changed"] == 10  # every fresh parse counts now
    assert g1.cache_stats["parsed_uncapped"] == 5
    assert len(json.loads(first)["manifest"]) == 10

    g2 = RepositoryGraph(str(repo))
    g2.build(collect_imported_names=True)
    assert g2.cache_stats["hit"] == 10  # 5 in-process + 5 from the snapshot
    assert g2.cache_stats["changed"] == 0  # no-op rebuild reports no changes
    assert blob_path.read_bytes() == first  # snapshot NOT rewritten


def test_delete_of_admitted_file_still_rewrites_snapshot(monkeypatch, tmp_path):
    """P0 companion: deletions of ADMITTED files still trigger a rewrite.

    The manifest-equality skip must not hide real changes: when a file in
    the persisted snapshot disappears, the freshly-built manifest has fewer
    keys and the dead entry must be cleaned from the JSON.
    """
    import json

    _set_cap(monkeypatch, 5)
    repo = tmp_path / "repo"
    repo.mkdir()
    for i in range(4):  # 4 < cap — every file admitted
        (repo / f"mod_{i}.py").write_text(f"v{i} = {i}\n")
    RepositoryGraph(str(repo)).build(collect_imported_names=True)
    blob_path = repo / ".cache" / "structural_graph_v1.json"
    (repo / "mod_2.py").unlink()

    RepositoryGraph(str(repo)).build(collect_imported_names=True)
    blob = json.loads(blob_path.read_text())
    assert "mod_2.py" not in blob["manifest"]
    assert len(blob["manifest"]) == 3


def test_edit_of_admitted_file_still_counts_as_changed_and_rewrites(monkeypatch, tmp_path):
    """P0 companion: editing an ADMITTED file still reports changed + rewrites.

    The admitted-only tracking must not hide genuine changes: a re-parse
    that lands in the cache (a new payload for the snapshot) is exactly
    what "changed" is for.
    """
    _set_cap(monkeypatch, 5)
    repo = tmp_path / "repo"
    repo.mkdir()
    for i in range(10):  # 10 files, cap 5 — mod_0..mod_4 admitted
        (repo / f"mod_{i}.py").write_text(f"v{i} = {i}\n")
    RepositoryGraph(str(repo)).build(collect_imported_names=True)
    blob_path = repo / ".cache" / "structural_graph_v1.json"
    first = blob_path.read_bytes()
    (repo / "mod_0.py").write_text("v0 = 999\n")  # admitted file, new size

    g = RepositoryGraph(str(repo))
    g.build(collect_imported_names=True)
    assert g.cache_stats["changed"] == 1
    assert blob_path.read_bytes() != first  # snapshot rewritten with new payload


def test_cap_is_defined_sane_default():
    # The cap must comfortably exceed the asicode repo's own file count so
    # the default never degrades the primary repo to admission-controlled caching.
    assert _EXTRACT_CACHE_MAX_ENTRIES >= 1024


# ── P0: cap-overflow re-parses must not skew "changed" nor force rewrites ────


def _cache_json_path(repo: str) -> Path:
    return Path(repo) / ".cache" / "structural_graph_v1.json"


def test_capped_repo_noop_rebuild_skips_snapshot_rewrite(monkeypatch):
    """A no-op rebuild on a capped repo reports changed=0 and skips the rewrite.

    Regression (P0): because admission control never persists the N>cap overflow
    files, they are re-parsed EVERY build.  Before the fix they were counted in
    ``_fresh_parsed`` → ``cache_stats["changed"] == N-cap`` and the byte-identical
    snapshot was re-serialized every build (the build's file-count hint is
    permanently true when N>cap).  Now only ADMITTED re-parses count, and
    ``_save_cache_snapshot`` skips the atomic rewrite when its freshly-built
    manifest equals the persisted one.
    """
    import time

    _set_cap(monkeypatch, 3)
    repo = _make_repo({f"mod_{i}.py": f"v{i} = {i}\n" for i in range(6)})  # 6 files, cap 3
    try:
        RepositoryGraph(repo).build(collect_imported_names=True)  # build #1: admit {0,1,2}
        # Drain the gc-deferral so a steady admitted set is on disk (avoids a
        # deferred dead-file sweep landing in the no-op build under test).
        RepositoryGraph(repo).build(collect_imported_names=True)
        blob = __import__("json").loads(_cache_json_path(repo).read_text())
        # Merge-preserving (P0, 2026-08-12): the snapshot converged to the full
        # walked set — beyond-cap files are persisted, not dropped.
        assert sorted(blob["manifest"]) == [f"mod_{i}.py" for i in range(6)]

        mtime_before = _cache_json_path(repo).stat().st_mtime_ns
        time.sleep(0.01)
        g = RepositoryGraph(repo)
        g.build(collect_imported_names=True)  # build #N: NOTHING changed
        mtime_after = _cache_json_path(repo).stat().st_mtime_ns

        assert g.cache_stats["changed"] == 0
        assert g._fresh_parsed == []
        assert mtime_after == mtime_before  # the snapshot was NOT rewritten
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_capped_repo_admitted_change_rewrites_and_counts_exactly_one(monkeypatch):
    """Mutating an ADMITTED file rewrites the snapshot and counts changed=1.

    Counterpart to the no-op test: a real content change to a file the snapshot
    contains must still rewrite (the manifest stamp changes) and count as one
    "changed" file — not N-cap, and not zero either.
    """
    import time

    _set_cap(monkeypatch, 3)
    repo = _make_repo({f"mod_{i}.py": f"v{i} = {i}\n" for i in range(6)})
    try:
        RepositoryGraph(repo).build(collect_imported_names=True)
        RepositoryGraph(repo).build(collect_imported_names=True)  # steady state

        time.sleep(0.01)
        (Path(repo) / "mod_0.py").write_text("v0 = 999\n")  # ADMITTED file changed
        mtime_before = _cache_json_path(repo).stat().st_mtime_ns
        g = RepositoryGraph(repo)
        g.build(collect_imported_names=True)
        mtime_after = _cache_json_path(repo).stat().st_mtime_ns

        assert g.cache_stats["changed"] == 1
        assert g._fresh_parsed == ["mod_0.py"]
        assert mtime_after != mtime_before  # snapshot rewritten
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_capped_repo_beyond_cap_add_rewrites_to_converge(monkeypatch):
    """Adding a file beyond the cap still converges the snapshot (merge-preserving).

    The new file is parsed and injected; although not admitted to the
    in-process cache (cap full of live files), its payload is persisted via
    ``_pending_snapshot`` so the NEXT build serves it from disk instead of
    re-parsing forever (P0, 2026-08-12).  It counts as one "changed" file.
    """
    import time

    _set_cap(monkeypatch, 3)
    repo = _make_repo({f"mod_{i}.py": f"v{i} = {i}\n" for i in range(6)})
    try:
        RepositoryGraph(repo).build(collect_imported_names=True)
        RepositoryGraph(repo).build(collect_imported_names=True)  # steady: admit {0,1,2}

        time.sleep(0.01)
        (Path(repo) / "mod_extra.py").write_text("extra = 1\n")  # beyond cap (7th file)
        mtime_before = _cache_json_path(repo).stat().st_mtime_ns
        g = RepositoryGraph(repo)
        g.build(collect_imported_names=True)
        mtime_after = _cache_json_path(repo).stat().st_mtime_ns

        assert g.cache_stats["changed"] == 1
        assert g._fresh_parsed == []
        assert g._fresh_parsed_uncapped == ["mod_extra.py"]
        assert mtime_after != mtime_before  # snapshot rewritten to converge
        blob = __import__("json").loads(_cache_json_path(repo).read_text())
        assert "mod_extra.py" in blob["manifest"]
        assert len(blob["manifest"]) == 7
        # The new file IS in the live graph (parsed+injected).
        assert "mod_extra.py" in g.py_files
    finally:
        shutil.rmtree(repo, ignore_errors=True)
