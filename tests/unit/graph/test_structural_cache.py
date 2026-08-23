"""Unit tests for external_llm.graph.structural_cache — the shared on-disk
graph cache format (gate writes, RepositoryGraph reads).

Pins the format contract: four mandatory sections (version/manifest/files/
imported_names), fail-open load (None on any corruption or version mismatch),
atomic best-effort save, and lossless data_to_json/data_from_json roundtrip.
"""

import json
import os
import shutil
import tempfile
import textwrap
from pathlib import Path

import pytest

import external_llm.graph.structural_cache as sc  # module import — release-gate tracked
from external_llm.graph.repository_graph import RepositoryGraph


@pytest.fixture
def cache_dir():
    d = Path(tempfile.mkdtemp(prefix="test_sc_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _sample_cache():
    return {
        "manifest": {"a.py": [1000, 10]},
        "files": {"a.py": {"symbols": [], "calls": [], "imports": []}},
        "imported_names": {"a.py": ["os"]},
    }


def test_default_cache_path(cache_dir):
    assert sc.default_cache_path(cache_dir) == cache_dir / ".cache" / sc.CACHE_FILENAME


def test_save_load_roundtrip(cache_dir):
    path = sc.default_cache_path(cache_dir)
    data = _sample_cache()
    sc.save(path, data["manifest"], data["files"], data["imported_names"])
    assert sc.load(path) == {
        "version": sc._current_schema_version(),
        **data,
    }


def test_load_missing_file_returns_none(cache_dir):
    assert sc.load(cache_dir / "nope.json") is None


def test_load_corrupt_json_returns_none(cache_dir):
    path = cache_dir / "bad.json"
    path.write_text("{not json!!")
    assert sc.load(path) is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update({"version": 999}),
        lambda d: d.pop("manifest"),
        lambda d: d.update({"manifest": []}),  # wrong section type
        lambda d: d.pop("files"),
        lambda d: d.update({"files": []}),  # wrong section type
        lambda d: d.pop("imported_names"),  # mandatory — empty cross-refs
        lambda d: d.update({"imported_names": []}),  # would silently corrupt suppression
    ],
)
def test_load_rejects_bad_shape(cache_dir, mutate):
    path = cache_dir / "cache.json"
    data = {"version": sc._current_schema_version(), **_sample_cache()}
    mutate(data)
    path.write_text(json.dumps(data))
    assert sc.load(path) is None


def test_schema_folding_detects_payload_field_changes(cache_dir, monkeypatch):
    """The cache version folds in the payload dataclass field signatures:
    SymbolNode(**d) silently fills a missing key with the field default, so
    adding a defaulted field without bumping the version would mix stale
    cached files (default) with fresh parses (real value) inside one graph.
    A changed field set must invalidate the cache structurally."""
    path = cache_dir / "cache.json"
    data = {"version": sc._current_schema_version(), **_sample_cache()}
    path.write_text(json.dumps(data))
    assert sc.load(path) is not None

    from dataclasses import fields

    from external_llm.graph.models import SymbolNode

    real_fields = fields(SymbolNode)
    _saved_version = sc._schema_version
    sc._schema_version = None  # drop the memo so the patch is observed
    try:
        # Simulate a future payload-field addition (defaulted, so the silent
        # injection failure mode would be real).
        monkeypatch.setattr(
            SymbolNode,
            "__dataclass_fields__",
            {**SymbolNode.__dataclass_fields__, "zz_new_field": real_fields[0]},
        )
        assert sc._current_schema_version() != data["version"]
        assert sc.load(path) is None  # old cache discarded — fail-open
    finally:
        sc._schema_version = _saved_version
        monkeypatch.undo()


def test_save_fail_open_when_dir_unwritable(cache_dir):
    blocker = cache_dir / "blocker"
    blocker.write_text("x")
    # parent is a FILE — mkdir fails; save must not raise.
    sc.save(blocker / "sub" / "cache.json", {}, {}, {})
    assert not (blocker / "sub").exists()


def test_save_has_no_tmp_leftover(cache_dir):
    path = sc.default_cache_path(cache_dir)
    sc.save(path, *_sample_cache().values())
    assert path.exists()
    # PID-suffixed tmp is renamed away by os.replace — nothing left behind.
    assert not list(path.parent.glob(f"{path.name}.tmp.*"))
    # Valid JSON, atomically replaced (no partial write).
    json.loads(path.read_text(encoding="utf-8"))


def test_save_tmp_name_is_pid_scoped(cache_dir, monkeypatch):
    """Concurrent gate processes (pre-commit + manual) must never interleave
    into the same temp file: the tmp name carries the pid."""
    path = sc.default_cache_path(cache_dir)
    written: list[str] = []

    import external_llm.graph.structural_cache as _sc_mod

    orig = _sc_mod.os.replace

    def capturing(src, dst):
        written.append(str(src))
        return orig(src, dst)

    monkeypatch.setattr(_sc_mod.os, "replace", capturing)
    sc.save(path, *_sample_cache().values())
    assert len(written) == 1
    assert written[0].endswith(f".tmp.{os.getpid()}")


def test_data_roundtrip_lossless(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "mod.py").write_text(
        textwrap.dedent("""
        import os
        from pkg import other

        def helper(x):
            return x

        class C:
            def m(self):
                helper(1)
    """)
    )
    (repo / "pkg" / "other.py").write_text("def other(): pass\n")
    from dataclasses import asdict

    graph = RepositoryGraph(str(repo))
    for rel in ("pkg/mod.py", "pkg/other.py"):
        data = graph.extract_file(str(repo / rel))
        assert data is not None
        rebuilt = sc.data_from_json(sc.data_to_json(data))
        assert asdict(rebuilt["symbols"][0]) == asdict(data["symbols"][0])
        assert len(rebuilt["calls"]) == len(data["calls"])
        assert [asdict(e) for e in rebuilt["calls"]] == [asdict(e) for e in data["calls"]]
        assert [asdict(e) for e in rebuilt["imports"]] == [asdict(e) for e in data["imports"]]


def test_save_output_byte_identical_to_plain_dumps(cache_dir):
    """P0-2: the streaming writer must emit the exact bytes of the old single
    ``json.dumps(payload, ensure_ascii=False)`` — key order, default
    separators and non-ascii escaping all preserved (existing load-roundtrip
    tests cannot catch whitespace/order drift)."""
    path = sc.default_cache_path(cache_dir)
    n = 60
    manifest = {f"m{i}.py": [i * 1000, i] for i in range(n)}
    files = {
        f"m{i}.py": {
            "symbols": [{"name": f"s{i}", "qualname": f"m{i}.s{i}", "kind": "함수"}],
            "calls": [{"caller": f"m{i}.s{i}", "callee": f"m{i - 1}.s{i - 1}", "caller_line": i}],
            "imports": [{"module": f"m{i}", "name": "한글", "alias": None}],
        }
        for i in range(n)
    }
    imported_names = {f"m{i}.py": [f"n{i}"] for i in range(n)}
    sc.save(path, manifest, files, imported_names)
    expected = json.dumps(
        {
            "version": sc._current_schema_version(),
            "manifest": manifest,
            "files": files,
            "imported_names": imported_names,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    assert path.read_bytes() == expected


def test_save_streaming_peak_memory_bounded(cache_dir):
    """P0-2: save() must not materialize the whole payload as one string.

    The old dumps+write_text peaked at ~8x the serialized size (335.7MB for
    the 42MB snapshot); the entry-wise streaming writer peaks at a small
    multiple of ONE entry.  With 1500 entries the full-dumps path would blow
    past total//2 — the streaming writer stays far under it."""
    import tracemalloc

    path = sc.default_cache_path(cache_dir)
    n = 1500
    manifest = {f"m{i}.py": [i * 1000, i] for i in range(n)}
    files = {}
    for i in range(n):
        files[f"m{i}.py"] = {
            "symbols": [{"name": f"s{i}", "qualname": f"m{i}.s{i}", "kind": "function"}],
            "calls": [{"caller": f"m{i}.s{i}", "callee": f"m{i - 1}.s{i - 1}", "caller_line": i}],
            "imports": [{"module": f"m{i}", "name": f"s{i}", "alias": None}],
        }
    imported_names = {f"m{i}.py": [f"n{i}"] for i in range(n)}
    total_bytes = len(
        json.dumps(
            {
                "version": sc._current_schema_version(),
                "manifest": manifest,
                "files": files,
                "imported_names": imported_names,
            },
            ensure_ascii=False,
        ).encode("utf-8")
    )
    tracemalloc.start()
    try:
        sc.save(path, manifest, files, imported_names)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < total_bytes // 2, f"save peak {peak} vs serialized size {total_bytes}"


def test_save_swallows_serialization_failure_and_cleans_tmp(cache_dir):
    """C4 (2026-08-12): a non-serializable entry must not fail the caller
    (best-effort contract) and must not leave the PID-suffixed tmp behind.

    Pre-fix the entry-wise streaming writer opened the tmp BEFORE dumping
    entries, so json.dumps raising TypeError on an unserializable value (a)
    propagated to the caller, violating the documented fail-open contract,
    and (b) stranded ``.tmp.<pid>`` (the pid suffix means a different
    process's stale tmp is never reclaimed).  Post-fix TypeError/ValueError
    join OSError in the catch, and finally unlinks the uncommitted tmp.
    """
    import external_llm.graph.structural_cache as sc  # module import — release-gate tracked

    path = sc.default_cache_path(cache_dir)
    manifest = {"a.py": [1000, 10]}
    files = {"a.py": {"symbols": [{"name": "x", "bad": object()}], "calls": [], "imports": []}}
    imported_names = {"a.py": ["os"]}
    sc.save(path, manifest, files, imported_names)  # must NOT raise
    assert not path.exists(), "failed save must not leave a committed cache"
    leftovers = list(cache_dir.rglob("*.tmp.*"))
    assert leftovers == [], f"uncommitted tmp not cleaned: {leftovers}"
    # And the cache must still be writable afterwards (no stuck state).
    files2 = {"a.py": {"symbols": [], "calls": [], "imports": []}}
    sc.save(path, manifest, files2, imported_names)
    assert path.exists(), "save must work after a failed attempt"
    assert sc.load(path) is not None  # valid payload
