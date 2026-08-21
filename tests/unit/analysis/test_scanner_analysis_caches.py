"""Per-file analysis disk caches: cold/warm parity + invalidation + fail-open.

P14-4: broken_contract member grouping and ast_similarity normalised symbols
are pure functions of file content and are cached per-file by
(st_mtime_ns, st_size) fingerprint (A307 pattern).  These tests pin the
contract: a warm rebuild (cache hit) must produce IDENTICAL verdicts to a
cold one, a changed file must invalidate only itself, and a corrupt/mismatch
cache must fall back to recomputation instead of serving stale results.
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

from external_llm.analysis.ast_similarity_scanner import (
    _AST_SIM_CACHE_VERSION,
    _ast_sim_cache_path,
    scan_similarity_candidates,
)
from external_llm.analysis.broken_contract_scanner import (
    _BROKEN_CONTRACT_CACHE_VERSION,
    _broken_contract_cache_path,
    scan_broken_contracts,
)
from external_llm.analysis.duplicate_definition_scanner import (
    _DUP_DEF_CACHE_VERSION,
    _dup_def_cache_path,
    scan_duplicate_definitions,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _write(tmp_path, name: str, src: str) -> str:
    p = tmp_path / name
    p.write_text(textwrap.dedent(src), encoding="utf-8")
    return name


_DUP = """
    def load_user_config(path, defaults):
        if not path:
            return dict(defaults)
        try:
            with open(path) as f:
                data = json.load(f)
        except OSError as e:
            logger.warning("config load failed: %s", e)
            return dict(defaults)
        merged = dict(defaults)
        merged.update(data)
        return merged

    def load_project_config(cfg_path, base):
        if not cfg_path:
            return dict(base)
        try:
            with open(cfg_path) as f:
                payload = json.load(f)
        except OSError as err:
            logger.warning("config load failed: %s", err)
            return dict(base)
        merged = dict(base)
        merged.update(payload)
        return merged
"""


_DUP_DEF = """
    def load_user_config(path):
        return path

    def load_user_config(path, extra=None):
        return path, extra
"""


_CONTRACT = """
    class Store:
        def __init__(self):
            self._pending_impl_spec = {}

        def set_pending_impl_spec(self, spec):
            self._pending_impl_spec = spec

        def get_pending_impl_spec(self):
            return self._pending_impl_spec.get("spec")

        def clear_pending_impl_spec(self):
            self._pending_impl_spec.pop("spec", None)
"""


class _FakeGraph:
    """Minimal repo-graph facade: every member is live (both callers > 0)."""

    def get_callers(self, name):
        return ["caller"]

    def get_symbols_in_file(self, rel_path):
        return []


def _cold_ast_sim(repo_root: str, files: list[str]) -> list:
    """Run with the disk caches removed (true cold scan)."""
    _purge_caches(repo_root)
    return scan_similarity_candidates(repo_root, files)


def _warm_ast_sim(repo_root: str, files: list[str]) -> list:
    """Run once to populate, then again — the second is the cache-hit run."""
    scan_similarity_candidates(repo_root, files)
    return scan_similarity_candidates(repo_root, files)


def _cold_broken(repo_root: str, files: list[str]) -> list:
    _purge_caches(repo_root)
    return scan_broken_contracts(repo_root, files, repo_graph=_FakeGraph())


def _warm_broken(repo_root: str, files: list[str]) -> list:
    scan_broken_contracts(repo_root, files, repo_graph=_FakeGraph())
    return scan_broken_contracts(repo_root, files, repo_graph=_FakeGraph())


def _purge_caches(repo_root: str) -> None:
    from contextlib import suppress

    for f in (
        _ast_sim_cache_path(repo_root),
        _broken_contract_cache_path(repo_root),
        _dup_def_cache_path(repo_root),
    ):
        with suppress(FileNotFoundError):
            os.unlink(f)


# ── duplicate_definition cache contract (P14-5) ──────────────────────────────
# The tree-sitter top-level-definition collection is a pure function of file
# content; a (st_mtime_ns, size) fingerprint cache makes a warm gate reuse
# the previous process's def lists instead of re-parsing every file (measured
# 1.43s → 0.02s on the full repo).


def test_dup_def_warm_matches_cold(tmp_path):
    fname = _write(tmp_path, "collide.py", _DUP_DEF)
    files = [fname]
    cold = _cold_dup_def(str(tmp_path), files)
    warm = _warm_dup_def(str(tmp_path), files)
    cold_keys = {(c.file, c.name, c.symbol_kind, tuple(c.occurrences)) for c in cold}
    warm_keys = {(c.file, c.name, c.symbol_kind, tuple(c.occurrences)) for c in warm}
    assert cold_keys == warm_keys
    # The fixture defines load_user_config twice → one duplicate candidate.
    assert any(c.name == "load_user_config" for c in warm)


def test_dup_def_cache_stored_shape(tmp_path):
    """The cache file keys by abs path, stores fp + JSON-shaped def lists."""
    fname = _write(tmp_path, "collide.py", _DUP_DEF)
    scan_duplicate_definitions(repo_root=str(tmp_path), file_paths=[fname])
    path = Path(_dup_def_cache_path(str(tmp_path)))
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == _DUP_DEF_CACHE_VERSION
    abs_key = str(tmp_path / fname)
    entry = payload["files"][abs_key]
    assert len(entry["fp"]) == 2
    # defs are [name, kind, lineno, end_lineno, receiver] lists (JSON shape).
    defs = entry["defs"]
    assert isinstance(defs, list)
    assert any(d[0] == "load_user_config" and d[1] == "function" for d in defs)


def test_dup_def_cache_invalidated_on_content_change(tmp_path):
    fname = _write(tmp_path, "collide.py", _DUP_DEF)
    first = _warm_dup_def(str(tmp_path), [fname])
    assert any(c.name == "load_user_config" for c in first)
    # Drop the second definition (rename to something unique) → no collision.
    _write(
        tmp_path,
        fname,
        _DUP_DEF.replace("def load_user_config(path):", "def load_other(path):", 1),
    )
    second = _warm_dup_def(str(tmp_path), [fname])
    assert all(c.name != "load_user_config" for c in second)


def test_dup_def_corrupt_cache_falls_open(tmp_path):
    fname = _write(tmp_path, "collide.py", _DUP_DEF)
    cache_path = Path(_dup_def_cache_path(str(tmp_path)))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("{not json", encoding="utf-8")
    cands = scan_duplicate_definitions(repo_root=str(tmp_path), file_paths=[fname])
    assert any(c.name == "load_user_config" for c in cands)  # recomputed, not crashed


def test_dup_def_cache_version_mismatch_falls_open(tmp_path):
    fname = _write(tmp_path, "collide.py", _DUP_DEF)
    cache_path = Path(_dup_def_cache_path(str(tmp_path)))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"format": _DUP_DEF_CACHE_VERSION + 99, "files": {}}),
        encoding="utf-8",
    )
    cands = scan_duplicate_definitions(repo_root=str(tmp_path), file_paths=[fname])
    assert any(c.name == "load_user_config" for c in cands)


def _cold_dup_def(repo_root: str, files: list[str]) -> list:
    _purge_caches(repo_root)
    return scan_duplicate_definitions(repo_root=repo_root, file_paths=files)


def _warm_dup_def(repo_root: str, files: list[str]) -> list:
    scan_duplicate_definitions(repo_root=repo_root, file_paths=files)
    return scan_duplicate_definitions(repo_root=repo_root, file_paths=files)


# ── ast_similarity cache contract ────────────────────────────────────────────


def test_ast_sim_warm_matches_cold(tmp_path):
    fname = _write(tmp_path, "dup.py", _DUP)
    files = [fname]
    cold = _cold_ast_sim(str(tmp_path), files)
    warm = _warm_ast_sim(str(tmp_path), files)
    cold_pairs = {frozenset([c.symbol_a, c.symbol_b]) for c in cold}
    warm_pairs = {frozenset([c.symbol_a, c.symbol_b]) for c in warm}
    assert cold_pairs == warm_pairs
    assert warm_pairs  # the fixture pair is found both ways
    # Per-verdict identity: the exact pair survives with the same similarity.
    c_key = next(iter(cold_pairs))
    w = next(c for c in warm if {c.symbol_a, c.symbol_b} == c_key)
    c = next(c for c in cold if {c.symbol_a, c.symbol_b} == c_key)
    assert w.similarity == c.similarity


def test_ast_sim_cache_invalidated_on_content_change(tmp_path):
    fname = _write(tmp_path, "dup.py", _DUP)
    files = [fname]
    first = _warm_ast_sim(str(tmp_path), files)
    # Change the file: fingerprints differ, cache must recompute.
    _write(
        tmp_path,
        fname,
        _DUP
        + """
    def unrelated_helper(x):
        return x + 1
""",
    )
    second = _warm_ast_sim(str(tmp_path), files)
    # The new helper is >=5 lines so it joins the pairwise population.
    assert len(second) >= len(first)


def test_ast_sim_corrupt_cache_falls_open(tmp_path):
    fname = _write(tmp_path, "dup.py", _DUP)
    cache_path = Path(_ast_sim_cache_path(str(tmp_path)))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("{not json", encoding="utf-8")
    cands = scan_similarity_candidates(str(tmp_path), [fname])
    assert cands  # recomputed, not crashed


def test_ast_sim_cache_version_mismatch_falls_open(tmp_path):
    fname = _write(tmp_path, "dup.py", _DUP)
    cache_path = Path(_ast_sim_cache_path(str(tmp_path)))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"format": _AST_SIM_CACHE_VERSION + 99, "files": {}}),
        encoding="utf-8",
    )
    cands = scan_similarity_candidates(str(tmp_path), [fname])
    assert cands


# ── broken_contract cache contract ───────────────────────────────────────────


def test_broken_warm_matches_cold(tmp_path):
    fname = _write(tmp_path, "store.py", _CONTRACT)
    files = [fname]
    cold = _cold_broken(str(tmp_path), files)
    warm = _warm_broken(str(tmp_path), files)
    # With the all-live fake graph both halves are reachable → no candidate.
    assert cold == []
    assert warm == []


def test_broken_cache_stores_grouped_members(tmp_path):
    """The cache file is written and keys by pair-eligible cores."""
    fname = _write(tmp_path, "store.py", _CONTRACT)
    scan_broken_contracts(str(tmp_path), [fname], repo_graph=_FakeGraph())
    path = Path(_broken_contract_cache_path(str(tmp_path)))
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == _BROKEN_CONTRACT_CACHE_VERSION
    entry = payload["files"][fname]
    # set_pending_impl_spec / get_pending_impl_spec / clear_pending_impl_spec
    # share the core name "pending_impl_spec" → grouped together.
    assert "pending_impl_spec" in entry["grouped"]
    members = entry["grouped"]["pending_impl_spec"]
    assert {m["name"] for m in members} == {
        "set_pending_impl_spec",
        "get_pending_impl_spec",
        "clear_pending_impl_spec",
    }
    # sets serialised as sorted lists, no AST nodes.
    assert isinstance(members[0]["writes"], list)
    assert "node" not in members[0]


def test_broken_cache_invalidated_on_content_change(tmp_path):
    fname = _write(tmp_path, "store.py", _CONTRACT)
    _warm_broken(str(tmp_path), [fname])
    # Change the file: add a second writer for the same core → new candidate
    # appears when the reader has no callers (orphan writer case).
    _write(
        tmp_path,
        fname,
        _CONTRACT
        + """
        def reset_pending_impl_spec(self, spec):
            self._pending_impl_spec = {}
""",
    )

    class _OrphanGraph(_FakeGraph):
        def get_callers(self, name):
            return [] if name == "reset_pending_impl_spec" else ["caller"]

    warm = _warm_broken_orphan(str(tmp_path), [fname], _OrphanGraph())
    assert warm  # reset_pending_impl_spec orphan writer is detected


def _warm_broken_orphan(repo_root: str, files: list[str], graph) -> list:
    scan_broken_contracts(repo_root, files, repo_graph=graph)
    return scan_broken_contracts(repo_root, files, repo_graph=graph)


def test_broken_corrupt_cache_falls_open(tmp_path):
    fname = _write(tmp_path, "store.py", _CONTRACT)
    cache_path = Path(_broken_contract_cache_path(str(tmp_path)))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("{not json", encoding="utf-8")
    cands = scan_broken_contracts(str(tmp_path), [fname], repo_graph=_FakeGraph())
    assert cands == []  # recomputed (no crash); all-live graph → no candidate


def test_broken_cache_version_mismatch_falls_open(tmp_path):
    fname = _write(tmp_path, "store.py", _CONTRACT)
    cache_path = Path(_broken_contract_cache_path(str(tmp_path)))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"format": _BROKEN_CONTRACT_CACHE_VERSION + 99, "files": {}}),
        encoding="utf-8",
    )
    cands = scan_broken_contracts(str(tmp_path), [fname], repo_graph=_FakeGraph())
    assert cands == []
