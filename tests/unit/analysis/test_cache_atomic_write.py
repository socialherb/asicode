"""B2 — disk-cache write atomicity (crash + concurrency policy, 2026-08-16).

The five gate/scanner disk caches under ``<repo_root>/.cache/`` are shared by
concurrent processes (parallel sessions' pre-commit gates, manual scans, the
REPL).  Their WRITE contract (see the policy block in ``parse_cache``) is:

* atomic whole-file replace via ``common.atomic_io.atomic_write_json`` —
  a crash mid-save must leave the PREVIOUS cache intact (never truncated),
* lock-free, last-writer-wins — concurrent savers never byte-interleave.

These tests pin that contract on every save site: the mid-dump crash test is
RED under a truncating ``open(path, "w")`` write (the target is destroyed the
moment the write begins).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from external_llm.analysis import _dead_block_shared as dbx
from external_llm.analysis import container_reachability_scanner as crx
from external_llm.analysis import cross_file_refs as cfr
from external_llm.analysis import unused_import_scanner as uic
from external_llm.analysis import vulture_scanner as vs

# (save callable, cache-path callable) — one per disk-cache save site
_SAVE_SITES = [
    pytest.param(
        lambda root: dbx._dbx_save(root, {"/abs/f.py": ((11, 2), ("py", {"a"}, (), {}))}),
        lambda root: dbx._dbx_cache_path(root),
        id="dead_block_extract",
    ),
    pytest.param(
        lambda root: uic._uic_save(root, {"/abs/f.py": ((11, 2), None)}),
        lambda root: uic._uic_cache_path(root),
        id="unused_import",
    ),
    pytest.param(
        lambda root: crx._crx_save(root, {"/abs/f.py": ((11, 2), {"reachable": []})}),
        lambda root: crx._crx_cache_path(root),
        id="container_reachability",
    ),
    pytest.param(
        lambda root: vs._save_vulture_scan_cache(root, {"/abs/f.py": {"fn": "f"}}, "1.0"),
        lambda root: vs._vulture_cache_path(root),
        id="vulture_scan",
    ),
    pytest.param(
        lambda root: cfr._save_importer_export_cache(root, {"/abs/f.py": {"fp": [11, 2]}}),
        lambda root: cfr._importer_export_cache_path(root),
        id="importer_export",
    ),
]


def _cache_path(root: Path, path_of) -> Path:
    return Path(path_of(str(root)))


@pytest.mark.parametrize(("save", "path_of"), _SAVE_SITES)
def test_crash_mid_dump_preserves_previous_cache(tmp_path, monkeypatch, save, path_of):
    """A save that dies mid-serialization must leave the old cache loadable.

    Under the old truncating ``open(path, "w")`` write the target is clobbered
    the moment the dump starts, so this is the RED pin for the atomic replace.
    """
    save(str(tmp_path))  # prime a valid previous cache
    path = _cache_path(tmp_path, path_of)
    old_bytes = path.read_bytes()
    json.loads(old_bytes)  # sanity: the primed file is complete

    real_dump = json.dump

    def crashing_dump(data, fh, *a, **k):
        fh.write('{"format": 99, "files": {')  # partial bytes land somewhere
        raise RuntimeError("simulated crash mid-dump")

    monkeypatch.setattr(json, "dump", crashing_dump)
    with pytest.raises(RuntimeError):
        save(str(tmp_path))
    monkeypatch.setattr(json, "dump", real_dump)

    # GREEN: the previous complete cache is untouched (write went to a tmp).
    assert path.read_bytes() == old_bytes
    json.loads(path.read_bytes())


@pytest.mark.parametrize(("save", "path_of"), _SAVE_SITES)
def test_concurrent_writers_never_interleave(tmp_path, save, path_of):
    """Two savers racing on one cache file must produce one complete payload.

    Lock-free last-writer-wins: every observable state of the file parses as
    one of the writers' whole payloads — never a byte-level mix.
    """
    import threading

    save(str(tmp_path))  # file exists from the start (reader never sees ENOENT)
    path = _cache_path(tmp_path, path_of)
    stop = threading.Event()
    parse_failures: list[str] = []

    def reader() -> None:
        while not stop.is_set():
            try:
                payload = json.loads(path.read_bytes())
                if not isinstance(payload.get("files"), dict):
                    parse_failures.append("shape")
            except ValueError as exc:  # JSONDecodeError — torn/partial file
                parse_failures.append(str(exc))

    def writer() -> None:
        for _ in range(25):
            save(str(tmp_path))

    rt = threading.Thread(target=reader)
    wt = threading.Thread(target=writer)
    rt.start(), wt.start()
    wt.join()
    stop.set()
    rt.join()
    assert not parse_failures, parse_failures[:3]


def test_atomic_write_json_matches_legacy_dump_bytes(tmp_path):
    """The primitive must reproduce the caches' legacy ``json.dump`` byte format.

    ``indent=None, ensure_ascii=True`` must equal a bare ``json.dump`` (no
    kwargs): switching save sites to the shared writer must NOT rewrite the
    on-disk format (that would churn 25MB+ files for zero reason).
    """
    from external_llm.common.atomic_io import atomic_write_json

    payload = {"format": 1, "files": {"/abs/테스트.py": {"fp": [11, 2], "x": [1, 2, 3]}}}
    legacy = tmp_path / "legacy.json"
    with legacy.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    via_primitive = tmp_path / "atomic.json"
    atomic_write_json(via_primitive, payload, indent=None, ensure_ascii=True)
    assert via_primitive.read_bytes() == legacy.read_bytes()
