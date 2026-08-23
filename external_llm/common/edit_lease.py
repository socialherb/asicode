"""Cross-process edit leases — advisory WIP-file ownership for parallel sessions.

Problem (documented incidents in this repo's design insights): two asicode
sessions running in parallel terminals on the same repo silently clobber each
other's uncommitted WIP — apply_patch AUTO-REPAIR resurrecting symbols a
parallel session deleted, ``git checkout`` permanently losing a session's
edits, ``ruff format`` reformatting a mid-edit file. The working tree carries
no per-session ownership signal, so a session cannot distinguish "file nobody
is touching" from "file another session is mid-edit on".

Design — content-addressed lease records, fail-open everywhere:

* Every successful write-tool edit records
  ``<repo_root>/.asicode/edit_leases/<sha256(relpath)[:20]>.json`` containing
  ``{"v": 1, "path", "pid", "host", "token", "ts"}``, atomic-replaced via
  :func:`common.atomic_io.atomic_write_json` (the shared write contract).
* Before a write tool mutates a file it checks for a LIVE FOREIGN lease:

  - own identity (host + pid + token)           -> no conflict (self re-edit)
  - same host, pid dead                         -> stale, no conflict
  - same host, pid == ours but token differs    -> recycled pid, no conflict
  - same host, pid alive, age <= TTL_LONG       -> CONFLICT
  - other host, age <= TTL_CROSS_HOST           -> CONFLICT (pid not probeable)
  - anything older / unreadable / malformed     -> no conflict

* Fail-open contract: an unreadable/corrupt/absent lease, an empty
  ``repo_root`` (test harness bypass), or ``ASICODE_EDIT_LEASES=0`` means "no
  conflict" and "no acquire". A lease must never make a write tool LESS
  available — same philosophy as :mod:`common.file_lock` degrading to a no-op
  lock. Stale leases are reclaimed implicitly: acquire atomically replaces the
  record, and a 7-day mtime sweep keeps the directory bounded.

Known V1 limitation: lease keys are normalized against the session's
``repo_root``; a scoped subagent that passes paths relative to its effective
(sub)root produces different keys than the main session for the same file —
a missed-conflict false negative only (never a false block), and the
orchestrator's scope partitioning keeps subagent file sets disjoint anyway.

``pid_is_alive`` duplicates the probe in
``external_llm.agent.subagent_ipc._is_process_alive`` (kept private there);
this copy keeps ``common`` free of ``agent``-layer imports. If a third
consumer appears, promote this one to the canonical home and delegate.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import time
import uuid
from pathlib import Path

from external_llm.common.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)

# Same-host leases: the pid probe is authoritative, so the age cap only guards
# against PID reuse keeping a dead session's lease alive forever. 12h is far
# beyond any single agent turn while staying under typical PID-wraparound
# horizons on interactive systems.
LEASE_TTL_SAME_HOST_S = 12 * 3600
# Cross-host leases (shared/network filesystem): the pid probe is meaningless,
# so recency is the only liveness signal. 30min covers an agent's
# think-then-edit gap without letting a long-dead session block edits.
LEASE_TTL_CROSS_HOST_S = 30 * 60
# Lease files whose mtime predates this are swept on acquire (mirrors
# common.file_lock.sweep_stale_lock_files' 7-day age gate).
SWEEP_MAX_AGE_S = 7 * 24 * 3600

_IDENTITY: dict | None = None


def pid_is_alive(pid: int) -> bool:
    """Cross-platform liveness probe for *pid* (never raises for pid > 0).

    POSIX: ``os.kill(pid, 0)`` probes existence without a real signal.
    Windows: ``os.kill(pid, 0)`` calls ``TerminateProcess`` — killing the
    target — so OpenProcess/GetExitCodeProcess via ctypes is used instead.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        _process_query_limited_information = 0x1000
        _still_active = 259
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(_process_query_limited_information, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == _still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just owned by someone else
    except OSError:
        return False
    else:
        return True


def session_identity() -> dict:
    """Stable per-process identity: ``{"pid", "host", "token"}``.

    Cached at first use. A forked child would inherit the cache; asicode
    subagents spawn via fresh interpreters (subprocess), so each gets its own
    token — which is exactly the desired granularity: one lease owner per
    OS process.
    """
    global _IDENTITY
    if _IDENTITY is None:
        _IDENTITY = {
            "pid": os.getpid(),
            "host": platform.node() or "unknown-host",
            "token": uuid.uuid4().hex[:12],
        }
    return _IDENTITY


def leases_disabled() -> bool:
    """Kill switch: ``ASICODE_EDIT_LEASES`` in {"0","false","off"} disables both
    the pre-write check and lease acquisition (fail-open both ways)."""
    return os.environ.get("ASICODE_EDIT_LEASES", "").strip().lower() in {"0", "false", "off"}


def normalize_lease_key(repo_root: str, file_path: str) -> str:
    """Repo-root-relative POSIX key for a file path (abs or rel, either slash)."""
    p = (file_path or "").strip()
    if not p:
        return ""
    rr = (repo_root or "").rstrip("/")
    if rr and (p == rr or p.startswith(rr + "/")):
        p = p[len(rr) :]
    return p.lstrip("/")


def _lease_dir(repo_root: str) -> Path:
    return Path(repo_root) / ".asicode" / "edit_leases"


def _lease_file(repo_root: str, key: str) -> Path:
    stem = hashlib.sha256(key.encode("utf-8", "surrogatepass")).hexdigest()[:20]
    return _lease_dir(repo_root) / f"{stem}.json"


def _sweep_stale_leases(directory: Path, *, now: float | None = None) -> None:
    """Remove lease files untouched for ``SWEEP_MAX_AGE_S`` (best-effort).

    A live lease is rewritten (mtime bumped) on every acquire, so a 7-day-old
    mtime proves no session touched the file in a week. Unlink races with a
    concurrent acquire are benign: content-addressed names + atomic replace
    mean the worst case is deleting a lease nobody relies on any more.
    """
    now = time.time() if now is None else now
    try:
        entries = list(directory.glob("*.json"))
    except OSError:
        logger.debug("edit-lease sweep could not list %s", directory, exc_info=True)
        return
    for path in entries:
        try:
            if now - path.stat().st_mtime > SWEEP_MAX_AGE_S:
                path.unlink()
        except OSError as err:
            logger.debug("edit-lease sweep skipped %s (%s)", path, err)


def acquire_edit_lease(repo_root: str, file_path: str, *, tool: str = "", now: float | None = None) -> None:
    """Record/refresh this process's lease on ``file_path``. Never raises.

    Called after a successful write; overwriting a stale/dead owner's record
    IS the reclaim. ``repo_root`` empty (harness) or the kill switch set →
    no-op.
    """
    rr = (repo_root or "").strip()
    if not rr or leases_disabled():
        return
    key = normalize_lease_key(rr, file_path)
    if not key:
        return
    ident = session_identity()
    lease = dict(ident)
    lease.update(
        {
            "v": 1,
            "path": key,
            "ts": time.time() if now is None else float(now),
        }
    )
    if tool:
        lease["tool"] = tool
    directory = _lease_dir(rr)
    try:
        atomic_write_json(_lease_file(rr, key), lease)
        _sweep_stale_leases(directory)
    except (OSError, TypeError, ValueError) as err:
        logger.debug("edit-lease acquire failed for %s (%s)", key, err)


def read_edit_lease(repo_root: str, file_path: str) -> dict | None:
    """Load the lease record for ``file_path``; None when absent/unreadable."""
    rr = (repo_root or "").strip()
    if not rr:
        return None
    key = normalize_lease_key(rr, file_path)
    if not key:
        return None
    try:
        with open(_lease_file(rr, key), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        logger.debug("edit-lease read failed for %s; treating as absent", key, exc_info=True)
        return None
    return data if isinstance(data, dict) else None


def _conflict_from_lease(lease: dict, key: str, *, now: float) -> dict | None:
    """Live-foreign classification for one lease record (None = no conflict)."""
    ident = session_identity()
    try:
        pid = int(lease.get("pid", 0))
        host = str(lease.get("host", ""))
        token = str(lease.get("token", ""))
        ts = float(lease.get("ts", 0.0))
    except (TypeError, ValueError):
        logger.debug("edit-lease record has non-numeric fields; treating as stale", exc_info=True)
        return None
    age = max(0.0, now - ts)  # tolerate small clock skew
    if host == ident["host"] and pid == ident["pid"] and token == ident["token"]:
        return None  # our own lease
    if host == ident["host"]:
        if pid <= 0:
            return None  # malformed record — fail-open
        if pid == ident["pid"]:
            # Our pid with a foreign token: the previous holder died and the
            # pid was recycled into THIS process. Not a live foreign owner.
            return None
        if not pid_is_alive(pid):
            return None  # owner process is gone — lease is stale
        ttl = LEASE_TTL_SAME_HOST_S
    else:
        ttl = LEASE_TTL_CROSS_HOST_S  # cannot probe a pid on another host
    if age > ttl:
        return None
    return {"path": key, "pid": pid, "host": host, "age_s": round(age, 1)}


def find_live_foreign_leases(repo_root: str, paths, *, now: float | None = None) -> list:
    """Return conflict records for every path carrying a live foreign lease.

    Each record: ``{"path", "pid", "host", "age_s", "lease_file"}``. Empty list
    on no conflict, disabled, empty root, or any read failure (fail-open).
    """
    rr = (repo_root or "").strip()
    if not rr or leases_disabled():
        return []
    now = time.time() if now is None else float(now)
    seen = set()
    conflicts = []
    for raw in paths or []:
        key = normalize_lease_key(rr, str(raw))
        if not key or key in seen:
            continue
        seen.add(key)
        lease = read_edit_lease(rr, key)
        if lease is None:
            continue
        conflict = _conflict_from_lease(lease, key, now=now)
        if conflict is None:
            continue
        conflict["lease_file"] = str(_lease_file(rr, key))
        conflicts.append(conflict)
    return conflicts
