"""Shared on-disk structural-graph cache — RepositoryGraph writes, all read.

Single source of truth for the ``.cache/structural_graph_v1.json`` format:

    {
      "version": 1,
      "manifest": {rel: [mtime_ns, size]},       # per-file staleness stamps
      "files":    {rel: {symbols/calls/imports}}, # asdict extraction payloads
      "imported_names": {rel: [name, ...]},       # py-only cross-ref names
    }

Ownership (layering decision, 2026-08-11 — updated by the pipeline
integration; originally gate-only write):

* WRITE path is ``RepositoryGraph.build(collect_imported_names=True)`` —
  the structural-scanner gate's mode (``scripts/check_structural_scanners.py``
  delegates its whole build to it).  That mode computes the per-file
  ``imported_names`` sets (``extract_imported_names_for_file``) the dead-code
  scanners consume and rewrites the cache with the COMPLETE payload only when
  at least one file was re-parsed — a fully cache-served build has nothing to
  add.  A graph-written cache can therefore never carry STALE names (every
  stamp-valid entry was produced against the same tree state) or an empty
  section (``load`` rejects a missing one as corrupt → full rebuild).

* READ path is shared: the plain ``RepositoryGraph.build()`` (no names, no
  write) loads the same JSON to warm its first build in a fresh process
  (cold start drops from a full re-parse to a stat + JSON lookup per file).
  The per-file manifest stamp is validated against the CURRENT stat before
  reuse, so a cache written from any earlier tree state is safe — changed
  files simply fall through to ``extract_file``.

Fail-open everywhere: any load problem (missing file, corrupt JSON, version
mismatch, wrong section types) yields None and callers fall back to a full
parse.  The cache is a speed optimization, never a correctness input.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, fields
from pathlib import Path

_logger = logging.getLogger(__name__)

CACHE_VERSION = 1
"""Manual version — bump when graph extraction changes (parser/visitor/
end-line logic).  A stale cache is silently discarded (fail-open).  Note the
``imported_names`` section is py-only by construction (the build records it
for ``_py_stamps`` files only — the cross_file_refs py-only contract,
2026-08-11), so a bump is NOT needed for that semantic."""

_schema_version: int | None = None


def _current_schema_version() -> int:
    """``CACHE_VERSION`` folded with the payload dataclass field signatures.

    ``SymbolNode(**d)`` (in :func:`data_from_json`) silently fills a missing
    key with the field's default, so adding a DEFAULTED field to a payload
    dataclass without bumping ``CACHE_VERSION`` would mix stale cached files
    (default value) with fresh parses (real value) inside one graph — no
    error anywhere, just silently wrong.  Folding the field-name signatures
    into the version makes that structurally impossible: any field addition
    or rename changes the version and discards the cache.

    Uses ``zlib.crc32`` — deliberately NOT ``hash()``: string hashing is
    randomized per process (``PYTHONHASHSEED``), so a hash()-based version
    would differ across processes and defeat the cache entirely.

    Lazy: importing ``CallEdge`` at module level would be circular
    (``repository_graph`` imports this module).
    """
    global _schema_version
    if _schema_version is None:
        import zlib

        from .models import ImportEdge, SymbolNode
        from .repository_graph import CallEdge

        _acc = CACHE_VERSION
        for _cls in (SymbolNode, CallEdge, ImportEdge):
            _sig = "|".join(f.name for f in fields(_cls)).encode("utf-8")
            _acc = (_acc * 31 + zlib.crc32(_sig)) & 0xFFFFFFFF
        _schema_version = _acc
    return _schema_version

CACHE_FILENAME = "structural_graph_v1.json"


def default_cache_path(repo_root: str | Path) -> Path:
    """The gate's cache location for a repository root."""
    return Path(repo_root) / ".cache" / CACHE_FILENAME


def load(cache_path: str | Path) -> dict | None:
    """Load the cache; None on any corruption/version mismatch.

    Mirrors the gate's historical ``_load_graph_cache`` validation exactly:
    all four sections must be present with the right shapes, or the cache is
    treated as corrupt (fail-open).  In particular ``imported_names`` is
    mandatory — an older v1 cache without it would load as a full graph hit
    but an EMPTY cross-ref name set, silently wrong dead-code suppression.
    """
    try:
        cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _logger.debug("structural graph cache unreadable (%s): %s", cache_path, exc)
        return None
    if (
        cache.get("version") != _current_schema_version()
        or not isinstance(cache.get("manifest"), dict)
        or not isinstance(cache.get("files"), dict)
        or not isinstance(cache.get("imported_names"), dict)
    ):
        _logger.debug("structural graph cache discarded (version/sections): %s", cache_path)
        return None
    return cache


def save(
    cache_path: str | Path,
    manifest: dict[str, list[int]],
    files: dict[str, dict],
    imported_names: dict[str, list[str]],
) -> None:
    """Atomically persist the cache (tmp + os.replace), streaming per entry.

    Best-effort: a failed write must never fail the caller — the next run
    just rebuilds (fail-open).

    Entry-wise streaming (P0-2, 2026-08-12): the old ``json.dumps(payload)``
    + ``write_text`` built the whole payload as one str AND re-encoded it to
    bytes — a 42MB snapshot peaked at ~335MB transient.  Writing each key and
    value with a small ``json.dumps`` produces a byte-identical file (same
    key order, same default ``separators``, same ``ensure_ascii=False``) while
    the peak drops to ~one entry (measured 335.7MB -> 7.3MB, and slightly
    faster: 1.87s -> 1.77s).
    """
    path = Path(cache_path)
    tmp: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # PID-suffixed tmp: two gate processes (pre-commit + manual) writing
        # concurrently must not interleave into the same temp file before the
        # atomic os.replace.  A stale tmp after a crash is harmless (never
        # read) and gets replaced/ignored on the next save.
        tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write('{"version": ')
            fh.write(json.dumps(_current_schema_version()))
            for key, section in (
                ("manifest", manifest),
                ("files", files),
                ("imported_names", imported_names),
            ):
                fh.write(f', "{key}": {{')
                first = True
                for rk, rv in section.items():
                    fh.write("" if first else ", ")
                    first = False
                    fh.write(json.dumps(rk, ensure_ascii=False))
                    fh.write(": ")
                    fh.write(json.dumps(rv, ensure_ascii=False))
                fh.write("}")  # close section
            fh.write("}")  # close payload
        os.replace(tmp, path)
        tmp = None  # committed — nothing to clean up
    except (OSError, TypeError, ValueError) as exc:
        # OSError: filesystem; TypeError/ValueError: an entry that
        # json.dumps cannot serialize (C4, 2026-08-12) — pre-streaming the
        # whole payload was dumped BEFORE the tmp existed, so a serialization
        # failure left nothing behind; entry-wise streaming can raise mid-
        # file, so both the exception class and the tmp cleanup are new.
        _logger.debug("structural graph cache write failed (%s): %s", cache_path, exc)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def data_to_json(data: dict) -> dict:
    """``extract_file`` result (dataclass objects) → JSON-ready dicts."""
    return {
        "symbols": [asdict(s) for s in data["symbols"]],
        "calls": [asdict(c) for c in data["calls"]],
        "imports": [asdict(i) for i in data["imports"]],
    }


def data_from_json(fdata: dict) -> dict:
    """Cached JSON dicts → ``extract_file``-shaped dataclass objects.

    Imports are function-local: ``repository_graph`` (CallEdge) imports this
    module, so a module-level import here would be circular.
    """
    from .models import ImportEdge, SymbolNode
    from .repository_graph import CallEdge

    return {
        "symbols": [SymbolNode(**d) for d in fdata["symbols"]],
        "calls": [CallEdge(**d) for d in fdata["calls"]],
        "imports": [ImportEdge(**d) for d in fdata["imports"]],
    }
