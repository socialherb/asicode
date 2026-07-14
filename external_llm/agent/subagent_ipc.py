"""
Sub-agent IPC — file-based task dispatch for multi-terminal orchestration.

Protocol: the orchestrator writes a ``task.json`` into a per-subagent directory
(``.asicode/subagents/<agent_id>/``).  An ``asi --subagent`` process polls
that directory, runs the AgentLoop, and writes ``result.json`` back.

┌─ Orchestrator (Terminal 1) ─────────────────┐
│  write task.json  →  .asicode/subagents/   │
│                         worker-1/task.json   │
│  poll  result.json ←  worker-1/result.json  │
└──────────────────────────────────────────────┘
           │   atomic acquisition via os.rename
           │   (task.json → task.json.claimed.<pid>)
           │   only one worker can win the rename →
           │   prevents double-dispatch across processes
           │
           │   epoch/nonce echoed in result.json →
           │   wait_for_result rejects stale results
┌─ Sub-agent   (Terminal 2) ───────────────────┐
│  asi --subagent --id worker-1             │
│    rename-claim task.json (atomic)            │
│    AgentLoop.run(task)                        │
│    write result.json (echoes task.epoch)      │
└───────────────────────────────────────────────┘
"""
from __future__ import annotations

import glob
import json
import logging
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass, field
from collections.abc import Callable
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Data models ────────────────────────────────────────────────────────────


@dataclass
class SubagentTask:
    """A task dispatched by the orchestrator to a sub-agent."""
    task_id: str
    title: str
    description: str
    assigned_files: list[str] = field(default_factory=list)
    original_request: str = ""
    priority: int = 0
    dependencies: list[str] = field(default_factory=list)
    max_turns: int = 12
    # Provider/model override for THIS sub-agent (overrides CLI defaults)
    provider: str = ""
    model: str = ""
    api_key: str = ""

    # When a predecessor hits max_turns, its partial diff/context is passed.
    predecessor_context: str = ""

    # Monotonic nonce assigned by the orchestrator (write_task) to prevent
    # wait_for_result from accepting a result left over from a previous task
    # on the same agent_id.  The worker echoes this into SubagentResult.epoch.
    epoch: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SubagentTask":
        # Filter out unknown keys so forward compat works
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class SubagentResult:
    """A result written back by the sub-agent after execution."""
    task_id: str
    status: str  # "success" | "error" | "max_turns" | "cancelled"
    final_message: str = ""
    diff: str = ""
    turns: int = 0
    applied_patches: list[dict] = field(default_factory=list)
    error: str = ""
    # Echoed from SubagentTask.epoch by the worker so wait_for_result can
    # reject stale results (epoch mismatch) on agent_id reuse.
    epoch: int = 0
    # Out-of-scope working-tree changes the worker made to files OUTSIDE its
    # ``assigned_files`` (B5). The orchestrator previously scoped
    # ``derive_applied_patches`` to ``assigned_files``, leaving such writes
    # permanently invisible to the review / diff cross-verification path. This
    # field surfaces them so the orchestrator can flag a scope violation or
    # include the files in a rollback set. Empty when the worker stayed in scope
    # (or when ``assigned_files`` was empty — then everything is in scope).
    unassigned_changes: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SubagentResult":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


def build_subagent_prompt(task: "SubagentTask") -> str:
    """Build the IPC worker's initial user-message text.

    Mirrors the in-process path (``orchestrator._run_subagent``):

    * ``task.predecessor_context`` carries the richly-built task_text — the
      subtask ``description`` plus an ``[Assigned files: …]`` hint, a
      ``[Predecessor task status]`` block (results of completed dependencies)
      and an ``[Orchestration progress]`` shared-memory block.
    * ``task.original_request`` is the overall user goal, wrapped around the
      task body when not already present.

    Using only ``task.description`` (as the worker previously did) drops the
    predecessor/shared-memory context, so dependent IPC subtasks ran "blind"
    while their in-process counterparts received full context. ``predecessor_context``
    is a superset of ``description`` (it starts from it), so preferring it
    loses nothing on independent tasks and restores parity on dependent ones.
    """
    prompt = task.predecessor_context or task.description
    if task.original_request and task.original_request not in prompt:
        prompt = (
            f"[Original request goal]\n{task.original_request}\n\n"
            f"[This sub-agent's task]\n{prompt}"
        )
    return prompt


def _path_matches_scope(changed_path: str, scope: set[str]) -> bool:
    """Check whether *changed_path* falls within *scope*, with directory-prefix matching.

    When a scope entry is a bare directory (e.g. ``src/utils`` — happened because
    the directory had more than ``_MAX_DIR_EXPANSION_FILES`` files and was left
    unexpanded), exact-path comparison would miss every file under it.  This
    helper also matches when a changed path *starts with* a scope entry followed
    by ``os.sep``, so cap-exceeded directories are still correctly recognised as
    in-scope (B6 / turn 13116).

    The ``+ os.sep`` guard ensures a scope entry ``a.txt`` does NOT match a
    changed path ``a.txt.bak`` — only genuine directory containment triggers.
    """
    _np = os.path.normpath(changed_path)
    if _np in scope:
        return True
    _sep = os.sep
    for _s in scope:
        if _np.startswith(_s + _sep):
            return True
    return False


def partition_changed_files(
    repo_root: str, assigned_files: list[str], timeout_s: float = 30.0,
) -> tuple[list[dict], list[dict]]:
    """Partition the working-tree changes into (in-scope, out-of-scope).

    A single UNSCOPED ``git status -z --porcelain --untracked-files=all`` pass
    captures every changed file (modified, added, untracked). The result is then
    split against *assigned_files*:

    * **in-scope** — files within the worker's assignment. These become
      ``applied_patches`` (the review / diff cross-verification path keys on
      them) and mirror ``_get_git_diff(repo_root, assigned_files)``.
    * **out-of-scope** — files the worker touched OUTSIDE its assignment.
      Previously these were invisible: ``derive_applied_patches`` scoped the
    ``git status`` call itself (``-- path1 path2``), so a worker that wrote to a
    scope-violating file (e.g. ``verify-repo/stats.py``) left the orchestrator's
    review permanently blind to it. Surfacing them lets the orchestrator flag a
    scope violation or include the files in a rollback set.

    When *assigned_files* is empty, EVERY changed file is treated as in-scope
    (preserving the legacy "no assignment = report everything" semantics); the
    out-of-scope list is then empty.

    The ``-z`` (NUL-delimited) mode is essential: even ``core.quotePath`` wraps
    space-containing paths in double quotes, but ``-z`` always emits raw
    (unquoted) paths. ``git diff --name-only`` is insufficient because it only
    captures *unstaged* modifications to *tracked* files — it misses untracked
    new files and staged changes.

    Any git failure yields ``([], [])`` (the worker degrades gracefully).
    """
    try:
        import subprocess as _sp
        cmd = ["git", "status", "-z", "--porcelain", "--untracked-files=all"]
        out = _sp.run(
            cmd, cwd=repo_root, capture_output=True, text=True, timeout=timeout_s,
        ).stdout
        # porcelain -z format (NUL-delimited):
        #   Non-rename: "XY PATH\0"
        #   Rename:     "XY NEW_PATH\0OLD_PATH\0"  (two consecutive fields)
        # NOTE: with -z, paths are ALWAYS raw/unquoted.  Must NOT use
        # strip() on the line — the status codes occupy positions 0-2
        # ('XY ') and strip() would destroy position-based parsing.
        all_paths: list[str] = []
        parts = out.split("\x00")
        i = 0
        while i < len(parts):
            line = parts[i]
            if not line or len(line) < 4:
                i += 1
                continue
            # Rename/copy can appear in EITHER status column: X position (staged,
            # e.g. "R  NEW\0OLD\0") OR Y position (worktree-detected, e.g. " R
            # NEW\0OLD\0", "DR", " C"). Checking only line[0] misses the Y-position
            # case and parses OLD as a phantom status line — its first 3 chars
            # (where the status codes live) get chopped, producing a ghost path
            # like " file.py" from "old file.py". Both columns must trigger the
            # two-field consume (skip OLD).
            if line[0] in ("R", "C") or line[1] in ("R", "C"):
                path = line[3:]
                i += 2
            else:
                path = line[3:]
                i += 1
            if path:
                all_paths.append(path)
        assigned_norm = {os.path.normpath(f) for f in (assigned_files or [])}
        if not assigned_norm:
            # No assignment → everything is in scope (legacy "report all" semantics).
            return [{"file": p} for p in all_paths], []
        # Use prefix-aware matching so unexpanded directory entries (those that
        # exceeded _MAX_DIR_EXPANSION_FILES) still match files under them.
        in_scope = [{"file": p} for p in all_paths if _path_matches_scope(p, assigned_norm)]
        out_scope = [{"file": p} for p in all_paths if not _path_matches_scope(p, assigned_norm)]
        return in_scope, out_scope
    except Exception:
        return [], []


def derive_applied_patches(
    repo_root: str, assigned_files: list[str], timeout_s: float = 30.0,
) -> list[dict]:
    """Derive the in-scope applied-patch file list (``git status --porcelain``).

    Thin wrapper over :func:`partition_changed_files` returning the in-scope
    half. Kept for backward compatibility — existing callers (and tests) see
    only the assigned-file subset, exactly as before.
    """
    return partition_changed_files(repo_root, assigned_files, timeout_s)[0]


def derive_unassigned_changes(
    repo_root: str, assigned_files: list[str], timeout_s: float = 30.0,
) -> list[dict]:
    """Derive the OUT-of-scope working-tree changes a worker made.

    Returns ``[{"file": <path>}, …]`` for files changed OUTSIDE
    *assigned_files*. Empty when the worker stayed in scope, or when
    *assigned_files* is empty (then everything is in scope). Any git failure
    yields ``[]``.
    """
    return partition_changed_files(repo_root, assigned_files, timeout_s)[1]


# ── Helpers ─────────────────────────────────────────────────────────────────


def _subagent_dir_path(repo_root: str, agent_id: str) -> str:
    """Return the IPC directory path for a given sub-agent (no side effects)."""
    return os.path.join(repo_root, ".asicode", "subagents", agent_id)


def _subagent_dir(repo_root: str, agent_id: str) -> str:
    """Return the IPC directory for a given sub-agent, creating it if needed."""
    d = _subagent_dir_path(repo_root, agent_id)
    os.makedirs(d, exist_ok=True)
    return d


def _atomic_write(path: str, content: str) -> None:
    """Write content to path atomically (unique tmpfile + rename).

    Uses a per-process-unique temp name (``mkstemp``) rather than a fixed
    ``path + ".tmp"`` so that concurrent writers to the same directory cannot
    truncate each other's buffer.  ``os.replace`` is atomic on POSIX.
    """
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".ipc-tmp-", suffix=os.path.basename(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, path)  # atomic on POSIX
    except BaseException:
        # Best-effort cleanup of the orphaned temp file on any failure.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ── Polling backoff & epoch sidecar ────────────────────────────────────────
# Exponential backoff caps idle filesystem polling during long sub-agent runs.
_POLL_BACKOFF_FACTOR = 1.5
_POLL_BACKOFF_CAP_S = 3.0


def _next_backoff(interval: float) -> float:
    """Grow the poll interval toward the cap to reduce idle FS load."""
    # Add jitter to decorrelate concurrent workers that start polling
    # simultaneously, avoiding thundering-herd sync on result.json rename.
    jitter = random.uniform(0.0, interval * 0.2)
    return min(interval * _POLL_BACKOFF_FACTOR + jitter, _POLL_BACKOFF_CAP_S)


def _expected_epoch_path(d: str) -> str:
    return os.path.join(d, "expected.json")


def _write_expected_epoch(d: str, epoch: int) -> None:
    """Write the epoch nonce the orchestrator expects the worker to echo."""
    _atomic_write(_expected_epoch_path(d), json.dumps({"epoch": epoch}))


def _read_expected_epoch(d: str) -> int:
    """Read the expected epoch sidecar (0 if absent/invalid → no validation)."""
    try:
        with open(_expected_epoch_path(d), "r", encoding="utf-8") as f:
            return int(json.load(f).get("epoch", 0))
    except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError):
        return 0


# ── Worker heartbeat (cross-process liveness) ─────────────────────────────
# The worker writes heartbeat.json periodically while processing a task so the
# orchestrator's wait_for_result can distinguish a BUSY worker (slow LLM turn,
# long tool run) from a DEAD one (OOM/segfault). Without this, a crashed worker
# burns the full ipc_timeout_s because result.json simply never appears.
#
# heartbeat.json uses wall-clock ``time.time()`` (NOT ``time.monotonic()``): it
# is written by the worker PROCESS and read by the orchestrator PROCESS, and
# monotonic clocks are per-process and not comparable across processes.
HEARTBEAT_INTERVAL_S = 15.0   # worker writes a heartbeat this often
HEARTBEAT_STALE_S = 120.0     # orchestrator presumes dead past this age

# Idle heartbeat: written by the worker into ITS OWN poll directory while
# waiting for a task (not the task's dir, which does not exist yet). This lets
# ``_claim_reusable_worker`` judge liveness of a Terminal-launched worker (no
# PID handle) by a heartbeat that exists + is fresh — closing the gap that the
# B6 quiesce-gate only half-fixed (a hung Terminal worker was still optimistically
# reusable and burned the full ``ipc_timeout_s`` on the next claim).
_IDLE_HEARTBEAT_INTERVAL_S = 30.0  # idle (between tasks) is less frequent than busy
_IDLE_HEARTBEAT_FILENAME = "worker.heartbeat.json"


def _heartbeat_path(d: str) -> str:
    return os.path.join(d, "heartbeat.json")


def _load_heartbeat_json(path: str) -> Optional[dict]:
    """Read & parse a heartbeat JSON file, returning ``None`` on any error.

    Centralizes the open/parse/except skeleton shared by EVERY heartbeat reader
    (``read_heartbeat_state``, ``read_heartbeat_age_s``, and the two idle
    readers). The exception tuple is identical across all callers — keeping it
    in ONE place prevents drift where one reader silently swallows an exception
    type another does not, which would make liveness judgements asymmetric
    (e.g. ``_claim_reusable_worker`` treating a worker as alive via one reader
    and dead via another). ``None`` means "missing or unreadable" — callers
    must treat it as inconclusive, never as alive or dead.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError,
            PermissionError, OSError):
        return None


def _heartbeat_age_from(data: Optional[dict]) -> Optional[float]:
    """Compute heartbeat age (seconds) from a parsed payload, or ``None``.

    Shared by both age readers so the ts-extraction + invalid/stale handling
    stays in ONE place. Historically each age reader ran ``float(data.get("ts",
    0))`` INSIDE the broad ``except`` block, so a non-numeric/``null`` ts
    (``{"ts": "abc"}``, ``{"ts": null}``) collapsed to ``None``. Hoisting the
    load into :func:`_load_heartbeat_json` would otherwise leak that
    ``ValueError``/``TypeError`` out — a behavior change. This helper restores
    the original ``None``-collapsing semantics for any falsy payload or bad ts.
    """
    if not data:
        return None
    try:
        ts = float(data.get("ts", 0))
    except (ValueError, TypeError):
        return None
    if ts <= 0:
        return None
    return time.time() - ts


def _write_idle_heartbeat(repo_root: str, worker_id: str, payload: dict) -> Optional[str]:
    """Atomically write the worker's idle heartbeat (``worker.heartbeat.json``).

    Centralizes the path computation (``_subagent_dir`` — which CREATES the
    directory; the reader/claim path uses ``_subagent_dir_path`` and must NOT
    create it, so this asymmetry is preserved by NOT sharing this helper with
    readers), the JSON serialization + atomic write, and the advisory
    ``except OSError -> None`` semantics shared by both idle-heartbeat writers.
    Returns the path written, or ``None`` on failure (advisory — never fatal).
    """
    d = _subagent_dir(repo_root, worker_id)
    path = os.path.join(d, _IDLE_HEARTBEAT_FILENAME)
    try:
        _atomic_write(path, json.dumps(payload))
    except OSError:
        return None
    return path


def write_heartbeat(
    repo_root: str, agent_id: str, *, pid: int = 0,
    turn: int = 0, last_tool: str = "",
) -> str:
    """Write a liveness heartbeat for *agent_id*'s current task.

    Called by the worker (``asi --subagent``) from a background thread
    while a task runs. Writes to the task's OWN directory (the same dir
    ``write_result`` deposits ``result.json`` in) so ``wait_for_result`` can
    read it back. Atomic (mkstemp + rename) so readers never see a partial file.

    *turn* and *last_tool* are advisory progress hints (current turn number and
    the name of the last tool invoked) so the orchestrator UI can show
    "turn 5, run_tests" instead of only an elapsed-time counter. Both default to
    empty/0 for backward compatibility (legacy callers / older heartbeats).
    """
    d = _subagent_dir(repo_root, agent_id)
    path = _heartbeat_path(d)
    _atomic_write(path, json.dumps({
        "ts": time.time(), "pid": pid, "agent_id": agent_id,
        "turn": int(turn) if turn else 0,
        "last_tool": last_tool or "",
    }))
    return path


def read_heartbeat_state(repo_root: str, agent_id: str) -> Optional[dict]:
    """Return the full heartbeat payload for *agent_id*, or ``None``.

    Exposes the progress hints (``turn``/``last_tool``) alongside the liveness
    timestamp. ``None`` means no heartbeat exists yet (worker just starting, or a
    legacy worker that never writes one).
    """
    d = _subagent_dir_path(repo_root, agent_id)
    return _load_heartbeat_json(_heartbeat_path(d))


def write_worker_idle_heartbeat(
    repo_root: str, worker_id: str, *, pid: int = 0,
    tasks_served: int = 0, last_task_id: str = "", uptime_s: float = 0.0,
) -> Optional[str]:
    """Write an IDLE liveness heartbeat into the worker's OWN poll directory.

    Distinct from :func:`write_heartbeat` (which targets the *task's* dir while a
    task runs): this is written by the worker's idle poll loop (between tasks),
    into ``.asicode/subagents/<worker_id>/worker.heartbeat.json`` — a path that
    EXISTS independent of any task. ``_claim_reusable_worker`` reads it back to
    judge liveness of a Terminal-launched worker (no PID handle) before reuse: a
    heartbeat that EXISTS and is fresher than ``HEARTBEAT_STALE_S`` proves the
    worker is alive and polling. A missing/stale heartbeat is NOT conclusive
    death (the worker may be a legacy build, or the FS clock skewed), so the
    claim treats "no heartbeat" as optimistic-reuse (same as before) and only
    DROPS a worker whose heartbeat EXISTS but is stale.

    ``tasks_served``/``last_task_id``/``uptime_s`` are advisory observability
    fields (not consumed by ``_claim_reusable_worker``) so ``--json-stream``
    consumers and diagnostics (e.g. tenet-diagnose) can see reused-worker pool
    health — how long a worker has been alive and how many tasks it has
    served — without a separate channel. All default to "unknown" values for
    backward compatibility with callers that do not track them.

    Returns the path written, or ``None`` on failure (advisory — never fatal).
    """
    return _write_idle_heartbeat(repo_root, worker_id, {
        "ts": time.time(), "pid": pid, "worker_id": worker_id,
        "state": "idle",
        "tasks_served": int(tasks_served),
        "last_task_id": last_task_id or "",
        "uptime_s": round(float(uptime_s), 1),
    })


def write_worker_exited_heartbeat(repo_root: str, worker_id: str, *, pid: int = 0) -> Optional[str]:
    """Mark the worker's idle heartbeat ``state`` as ``"exited"`` right before exit.

    Written once, synchronously, from the worker's main loop after
    ``shutdown_event`` breaks the poll loop (SIGINT or the ``shutdown.json``
    sentinel) — the idle-heartbeat daemon thread stops emitting at the same
    point. Without this, the LAST "idle"-state heartbeat (up to
    ``_IDLE_HEARTBEAT_INTERVAL_S`` seconds stale) still looks fresh to
    ``_claim_reusable_worker`` for up to ``ipc_heartbeat_stale_s`` (120s
    default) after the process has actually terminated — a dead worker gets
    re-claimed and the next dispatch burns the full ``ipc_timeout_s`` before
    failing. ``state="exited"`` lets the claim drop it immediately regardless
    of age.

    Returns the path written, or ``None`` on failure (advisory — never fatal).
    """
    return _write_idle_heartbeat(repo_root, worker_id, {
        "ts": time.time(), "pid": pid, "worker_id": worker_id,
        "state": "exited",
    })


def read_worker_idle_heartbeat_state(repo_root: str, worker_id: str) -> Optional[str]:
    """Return the ``state`` field of the worker's idle heartbeat, or ``None``.

    ``"exited"`` means the worker process has terminated cleanly and must never
    be reused regardless of heartbeat age. ``None`` covers both "no heartbeat
    file yet" and any read failure — callers must NOT treat ``None`` as either
    alive or dead, only as inconclusive (mirrors :func:`read_worker_idle_heartbeat_age`).
    """
    d = _subagent_dir_path(repo_root, worker_id)
    path = os.path.join(d, _IDLE_HEARTBEAT_FILENAME)
    data = _load_heartbeat_json(path)
    return data.get("state") if data is not None else None


def read_worker_idle_heartbeat_age(repo_root: str, worker_id: str) -> Optional[float]:
    """Return the age (seconds) of the worker's idle heartbeat, or ``None``.

    ``None`` means no idle heartbeat exists (worker is a legacy build that does
    not write one, or never started). Callers must NOT treat ``None`` as "dead"
    — only a heartbeat that EXISTS and exceeds the stale threshold counts. This
    preserves backward compatibility: a pre-idle-heartbeat worker is still
    optimistically reusable (the claim falls back to the prior behavior).
    """
    d = _subagent_dir_path(repo_root, worker_id)
    path = os.path.join(d, _IDLE_HEARTBEAT_FILENAME)
    return _heartbeat_age_from(_load_heartbeat_json(path))


def read_heartbeat_age_s(repo_root: str, agent_id: str) -> Optional[float]:
    """Return the age (seconds) of the worker's last heartbeat, or None.

    ``None`` means no heartbeat exists yet (worker just starting, or a legacy
    worker that never writes one). Callers must NOT treat ``None`` as "dead" —
    only a heartbeat that EXISTS and exceeds the stale threshold counts.
    """
    d = _subagent_dir_path(repo_root, agent_id)
    return _heartbeat_age_from(_load_heartbeat_json(_heartbeat_path(d)))


def _quarantine_and_recheck(result_path, expected_epoch, agent_id):
    """Atomically quarantine a stale ``result.json``, adopting a fresh result if
    one raced in between the caller's read and this rename.

    Moves the current on-disk ``result.json`` to a sidecar quarantine path via
    ``os.replace`` (atomic on POSIX). If the worker wrote a *fresh* result
    between the caller's read (which saw an old epoch) and this rename, the
    quarantined file holds the NEW content; when its epoch matches
    *expected_epoch*, that result is returned rather than dropped. Otherwise the
    stale file is discarded and the caller keeps polling.

    Returns the ``SubagentResult`` to adopt, or ``None`` if the file was
    genuinely stale (or already absent — worker cleaned up / never existed).
    """
    qpath = f"{result_path}.stale.{os.getpid()}.{time.monotonic_ns()}"
    try:
        os.replace(result_path, qpath)  # atomic — captures whatever is on disk now
    except FileNotFoundError:
        return None
    try:
        with open(qpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        candidate = SubagentResult.from_dict(data)
        if getattr(candidate, "epoch", 0) == expected_epoch:
            logger.info(
                "IPC: adopted fresh result for %s recovered from quarantine "
                "(raced write detected between read and rename)", agent_id,
            )
            return candidate
    except (json.JSONDecodeError, ValueError, TypeError, OSError):
        pass
    finally:
        try:
            os.unlink(qpath)
        except OSError:
            pass
    return None
# ── Orchestrator-side API ──────────────────────────────────────────────────


def write_task(repo_root: str, task: SubagentTask, worker_id: str = "") -> str:
    """Write a task for a sub-agent.

    If *worker_id* is provided, the task.json is written to *that* worker's
    polling directory rather than the task's own directory.  The sub-agent
    (which polls ``--subagent-id <worker_id>``) will pick it up.  The result
    is still written back to the task's own directory (via ``task.task_id``)
    by the sub-agent, so ``wait_for_result(repo_root, task.task_id)`` works
    regardless of whether a worker was reused.

    Returns the task directory path (where task.json was written).
    """
    if worker_id:
        d = _subagent_dir(repo_root, worker_id)
    else:
        d = _subagent_dir(repo_root, task.task_id)
    path = os.path.join(d, "task.json")
    # Clear any stale shutdown sentinel so a relaunched/reused worker doesn't
    # exit immediately.  Writing a task means "this worker is wanted alive";
    # a leftover shutdown.json from a previous run would poison the poll loop.
    # Also clear a stale cancel sentinel: a cancel.json left from a previous
    # (cancelled) task would make the worker's cancel watcher abort the NEW
    # task the instant its watcher thread starts.  Both sentinels are
    # fire-once and must not survive into a fresh dispatch.
    for _sentinel in ("shutdown.json", "cancel.json"):
        try:
            os.unlink(os.path.join(d, _sentinel))
        except FileNotFoundError:
            pass
    # Clean up any orphaned rename-claim files left by a worker that crashed
    # mid-acquisition (task.json.claimed.<pid>) or a quarantined malformed task
    # (task.json.claimed.<pid>.bad).  Also clean up orphaned .ipc-tmp-* files
    # left by a crash inside _atomic_write before os.replace completes.
    # These are harmless but can accumulate; a fresh task must start clean.
    for _stale in glob.glob(os.path.join(d, "task.json.claimed.*")):
        try:
            os.unlink(_stale)
        except OSError:
            pass
    for _stale in glob.glob(os.path.join(d, ".ipc-tmp-*")):
        try:
            os.unlink(_stale)
        except OSError:
            pass
    # Assign a monotonic epoch nonce so wait_for_result can reject a result
    # left over from a previous run on this agent_id (defense-in-depth: the
    # worker echoes task.epoch into result.epoch).
    if not task.epoch:
        task.epoch = time.monotonic_ns()
    _atomic_write(path, json.dumps(task.to_dict(), ensure_ascii=False, indent=2))
    # The expected-epoch sidecar must live in the task's OWN directory — the
    # same place write_result deposits result.json and wait_for_result reads it
    # back.  In reuse mode (worker_id set) task.json is written to the worker's
    # poll directory `d`, but result.json is keyed by task.task_id, so the nonce
    # must be written there to stay paired with the result the worker echoes it
    # into.  Writing it to `d` instead would leave the task dir without an
    # expected.json, causing wait_for_result to read epoch=0 and silently skip
    # validation (the reuse-path gap this closes).
    task_dir = _subagent_dir(repo_root, task.task_id)
    _write_expected_epoch(task_dir, task.epoch)
    logger.info("IPC: wrote task %s → %s (epoch=%d)", task.task_id, path, task.epoch)
    return d


def wait_for_result(
    repo_root: str,
    agent_id: str,
    *,
    poll_interval_s: float = 0.5,
    timeout_s: float = 600.0,
    cancel_event: Optional[Any] = None,
    on_poll: Optional[Callable[[float, str], None]] = None,
    heartbeat_stale_s: float = HEARTBEAT_STALE_S,
    startup_timeout_s: float = 0.0,
    max_timeout_s: float = 0.0,
) -> Optional[SubagentResult]:
    """Poll for a result file.  Returns the deserialized result or None on timeout/cancel.

    Blocks up to *timeout_s* seconds, polling every *poll_interval_s*.
    If *cancel_event* (a ``threading.Event``) is set, returns None immediately.
    If *on_poll* is provided, it is called every ~5 seconds with (elapsed_s, agent_id).

    On the sub-agent side, ``asi --subagent`` writes ``result.json``
    atomically via `write_result`, so partial reads are impossible.

    Worker-liveness guard: if *heartbeat_stale_s* > 0 (default), the loop also
    checks ``heartbeat.json`` written by the worker's background thread. If a
    heartbeat EXISTS but is older than *heartbeat_stale_s*, the worker is
    presumed dead (OOM/segfault) and the call returns None immediately instead
    of burning the remaining *timeout_s*. A non-existent heartbeat (worker not
    started, or a legacy build) never trips this — the normal timeout governs.

    Startup-failure guard: if *startup_timeout_s* > 0 (default 0 = disabled) and
    NO heartbeat has EVER been seen within that many seconds of dispatch, the
    worker is presumed to have never started (launch failed, import error,
    Terminal window closed) and the call returns None. Disabled by default so it
    cannot false-positive on a worker that simply does not write heartbeats; opt
    in once every worker in the deployment is heartbeat-capable. Once any
    heartbeat has been seen this guard is inert for the rest of the call. The
    guard probes heartbeat independently of *heartbeat_stale_s*, so it stays
    correct when liveness checking is disabled (heartbeat_stale_s <= 0) while the
    startup guard is enabled.

    Soft-timeout extension: if *max_timeout_s* > *timeout_s* (default 0 =
    disabled), a FRESH heartbeat (worker alive and progressing) extends the
    deadline to *max_timeout_s* once. A legitimately-slow task (long LLM turn,
    slow tool) is then not killed the instant it crosses *timeout_s*; only a
    truly dead worker (stale heartbeat) is abandoned early. The hard cap is
    *max_timeout_s*, so a busy-but-alive worker can never run forever.
    """
    d = _subagent_dir(repo_root, agent_id)
    result_path = os.path.join(d, "result.json")
    expected_epoch = _read_expected_epoch(d)

    start_ts = time.monotonic()
    deadline = start_ts + timeout_s
    last_poll_ts = start_ts
    interval = poll_interval_s
    heartbeat_ever_seen = False
    heartbeat_extended = False
    while time.monotonic() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            logger.info("IPC: wait_for_result(%s) cancelled", agent_id)
            return None
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                raw = f.read()
            data = json.loads(raw)
            result = SubagentResult.from_dict(data)
            # Reject a stale result whose epoch does not match the task we
            # dispatched.  Backward-compatible: if either side lacks an epoch
            # (0), skip validation so legacy workers/results still resolve.
            if (
                expected_epoch
                and getattr(result, "epoch", 0)
                and result.epoch != expected_epoch
            ):
                logger.warning(
                    "IPC: ignoring stale result for %s (epoch %d != expected %d)",
                    agent_id, result.epoch, expected_epoch,
                )
                # TOCTOU-safe stale-result removal. A plain ``os.unlink`` would
                # *lose* a fresh result that the worker renamed in between our
                # read (old epoch) and the delete — the loop would then burn the
                # rest of the timeout waiting for a result that already arrived.
                # Atomically swap the current on-disk file to a quarantine path
                # via ``os.replace`` (atomic on POSIX): whatever lands at rename
                # time is captured. If the quarantined content now carries the
                # expected epoch, a fresh write won the race — adopt it instead
                # of discarding it.
                _adopted = _quarantine_and_recheck(
                    result_path, expected_epoch, agent_id,
                )
                if _adopted is not None:
                    return _adopted
                raise ValueError("stale result")
            return result
        except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError, PermissionError, OSError):
            # Result not ready — check worker liveness via heartbeat so a
            # CRASHED worker (OOM/segfault) does not burn the full timeout.
            # Only trips when a heartbeat EXISTS and is stale: no heartbeat
            # means the worker is just-starting or a legacy build that never
            # writes one, in which case the normal timeout governs.
            if heartbeat_stale_s > 0:
                _hb_age = read_heartbeat_age_s(repo_root, agent_id)
                if _hb_age is not None:
                    heartbeat_ever_seen = True
                    if _hb_age > heartbeat_stale_s:
                        logger.warning(
                            "IPC: wait_for_result(%s) worker heartbeat stale "
                            "(%.0fs > %.0fs threshold); presuming worker dead.",
                            agent_id, _hb_age, heartbeat_stale_s,
                        )
                        return None
                    # Soft-timeout extension: heartbeat is fresh (worker alive and
                    # progressing). Extend the deadline toward the hard cap ONCE,
                    # so a legitimately-slow task isn't abandoned the instant it
                    # crosses timeout_s. Only a fresh heartbeat earns the
                    # extension; a subsequently-stale heartbeat is caught above.
                    if (
                        max_timeout_s > timeout_s
                        and not heartbeat_extended
                    ):
                        deadline = start_ts + max_timeout_s
                        heartbeat_extended = True
                        logger.info(
                            "IPC: wait_for_result(%s) heartbeat fresh (%.0fs) — "
                            "extending deadline %.0fs → %.0fs (worker progressing)",
                            agent_id, _hb_age, timeout_s, max_timeout_s,
                        )
            # Startup-failure guard: no heartbeat has EVER appeared past the
            # startup deadline → the worker likely never engaged the task
            # (launch failed, import error). Inert once a heartbeat was seen.
            #
            # heartbeat_ever_seen is normally maintained by the liveness block
            # above, but only while heartbeat_stale_s > 0. When liveness checking
            # is disabled (heartbeat_stale_s <= 0) yet the startup guard is
            # enabled, probe heartbeat directly here — otherwise a worker that
            # IS actively writing heartbeats would never flip
            # heartbeat_ever_seen, and a live worker would be falsely abandoned
            # once the startup deadline elapses. The two thresholds are
            # independent params; this keeps all their combinations consistent.
            if startup_timeout_s > 0 and not heartbeat_ever_seen:
                if heartbeat_stale_s <= 0:
                    _hb_age = read_heartbeat_age_s(repo_root, agent_id)
                    if _hb_age is not None:
                        heartbeat_ever_seen = True
                if (time.monotonic() - start_ts) > startup_timeout_s and not heartbeat_ever_seen:
                    logger.warning(
                        "IPC: wait_for_result(%s) startup timeout — no heartbeat "
                        "within %.0fs; presuming worker failed to start.",
                        agent_id, startup_timeout_s,
                    )
                    return None
            now = time.monotonic()
            if on_poll and (now - last_poll_ts) >= 5.0:
                on_poll(now - start_ts, agent_id)
                last_poll_ts = now
            time.sleep(interval)  # Still waiting
            interval = _next_backoff(interval)  # ease idle FS load
    _eff = max_timeout_s if (max_timeout_s > timeout_s and heartbeat_extended) else timeout_s
    logger.warning("IPC: wait_for_result(%s) timed out after %.0fs", agent_id, _eff)
    return None


def clear_result(repo_root: str, agent_id: str) -> None:
    """Remove any stale result.json so wait_for_result waits for a fresh one.

    Must be called by the orchestrator *before* writing a new task.json,
    otherwise wait_for_result may immediately read a result left over from
    a previous run and return before the sub-agent has even started on the
    new task.
    """
    d = _subagent_dir(repo_root, agent_id)
    result_path = os.path.join(d, "result.json")
    try:
        os.unlink(result_path)
        logger.debug("IPC: cleared stale result.json for %s", agent_id)
    except FileNotFoundError:
        pass
    # Also clear the expected-epoch sidecar so a fresh write_task starts clean.
    # (write_task overwrites it regardless, but this keeps the dir tidy and
    # guarantees no stale epoch can match a pre-existing result.)
    try:
        os.unlink(_expected_epoch_path(d))
    except FileNotFoundError:
        pass
    # Also clear a stale heartbeat: the previous task's heartbeat.json would
    # otherwise be seen as stale by the next wait_for_result and falsely flag
    # the (re)launched worker as dead before it has written a fresh one.
    try:
        os.unlink(_heartbeat_path(d))
    except FileNotFoundError:
        pass


# ── Shutdown sentinel ────────────────────────────────────────────────────
# The orchestrator writes a shutdown.json to tell workers to stop polling.
# This prevents worker process/terminal leaks when orchestration ends.


def _write_lifecycle_sentinel(repo_root: str, agent_id: str, *, kind: str) -> str:
    """Write a single lifecycle sentinel — shared by shutdown & cancel.

    ``kind`` is ``"shutdown"`` (exit the poll loop between tasks) or
    ``"cancel"`` (abort the in-flight task now). The two differ only in the
    filename (``{kind}.json``) and JSON flag key, so the write is unified here;
    the public wrappers below carry the distinct docstrings callers rely on.
    """
    d = _subagent_dir(repo_root, agent_id)
    path = os.path.join(d, f"{kind}.json")
    _atomic_write(path, json.dumps({kind: True, "agent_id": agent_id}))
    logger.info("IPC: wrote %s sentinel for %s → %s", kind, agent_id, path)
    return path


def _write_lifecycle_sentinels(
    repo_root: str, agent_ids: list[str], *, kind: str,
) -> list[str]:
    """Write lifecycle sentinels for all given agent IDs (best-effort)."""
    paths: list[str] = []
    for aid in agent_ids:
        try:
            paths.append(_write_lifecycle_sentinel(repo_root, aid, kind=kind))
        except Exception as e:
            logger.warning(
                "IPC: failed to write %s sentinel for %s: %s", kind, aid, e,
            )
    return paths


def write_shutdown_sentinel(repo_root: str, agent_id: str) -> str:
    """Write a shutdown sentinel so the worker stops polling and exits.

    Called by the orchestrator during cleanup after all sub-agents have
    completed or been cancelled.  The worker's ``poll_for_task`` checks
    for this file between polls.
    """
    return _write_lifecycle_sentinel(repo_root, agent_id, kind="shutdown")


def write_shutdown_all(repo_root: str, agent_ids: list[str]) -> list[str]:
    """Write shutdown sentinels for all given agent IDs."""
    return _write_lifecycle_sentinels(repo_root, agent_ids, kind="shutdown")


def _check_lifecycle_sentinel(repo_root: str, agent_id: str, *, kind: str) -> bool:
    """Check for a lifecycle sentinel — shared by shutdown & cancel.

    ``kind`` is ``"shutdown"`` or ``"cancel"``; the existence check differs only
    in the filename (``{kind}.json``). Pure read — no filesystem mutation and no
    directory creation (uses _subagent_dir_path, not _subagent_dir), so probing
    a never-spawned agent is side-effect-free.
    """
    d = _subagent_dir_path(repo_root, agent_id)
    return os.path.isfile(os.path.join(d, f"{kind}.json"))


def check_shutdown_sentinel(repo_root: str, agent_id: str) -> bool:
    """Check if a shutdown sentinel exists for the given agent.

    Returns True if the worker should exit.
    Pure check — no filesystem mutation.
    """
    return _check_lifecycle_sentinel(repo_root, agent_id, kind="shutdown")

# ── Cancel sentinel (mid-task abort) ─────────────────────────────────────
# Distinct from the shutdown sentinel: shutdown = "exit the poll loop after
# your current task" (honored only between tasks — see poll_for_task).  cancel
# = "abort the task you are running RIGHT NOW".  The orchestrator writes a
# cancel.json when its own cancel_event fires (Ctrl-C); a watcher thread in the
# worker polls it during task execution and sets the worker's local cancel_event
# so DesignChatLoop aborts at its next turn boundary (DCL checks cancel_event at
# the top of every iteration).  Without this a cancelled orchestration leaves
# the worker running its task to completion — burning tokens in an orphaned
# terminal the orchestrator can no longer reach (it has no PID handle for an
# osascript-launched Terminal.app window).


def write_cancel_sentinel(repo_root: str, agent_id: str) -> str:
    """Write a cancel sentinel so the worker aborts its IN-FLIGHT task.

    Called by the orchestrator when its ``cancel_event`` fires.  Distinct from
    ``write_shutdown_sentinel`` (which only takes effect between tasks): the
    worker's cancel watcher thread polls this DURING task execution and sets the
    worker's local ``cancel_event``, so ``DesignChatLoop`` aborts at its next
    turn boundary rather than running the cancelled task to completion.
    """
    return _write_lifecycle_sentinel(repo_root, agent_id, kind="cancel")


def write_cancel_all(repo_root: str, agent_ids: list[str]) -> list[str]:
    """Write cancel sentinels for all given agent IDs (best-effort)."""
    return _write_lifecycle_sentinels(repo_root, agent_ids, kind="cancel")


def check_cancel_sentinel(repo_root: str, agent_id: str) -> bool:
    """Return True if a cancel sentinel exists for the given agent.

    The worker's cancel watcher (asi ``_cancel_watcher``) polls this during
    task execution. This function is a PURE existence check — it performs no
    filesystem mutation. On True, the *caller* (the watcher) consumes the
    sentinel (``os.unlink`` — fire-once) and sets its local ``cancel_event`` so
    ``DesignChatLoop`` aborts at the next turn boundary.
    """
    return _check_lifecycle_sentinel(repo_root, agent_id, kind="cancel")


# ── Sub-agent-side API ─────────────────────────────────────────────────────


def _is_process_alive(pid: int) -> bool:
    """Cross-platform liveness probe for *pid*.

    POSIX: ``os.kill(pid, 0)`` probes existence without sending a real signal.
    Windows has no such convention — ``os.kill(pid, 0)`` there calls
    ``TerminateProcess`` with exit code 0, which would kill the target — so we
    use ``OpenProcess``/``GetExitCodeProcess`` via ctypes instead.
    """
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just owned by someone else
    except OSError:
        return False


def poll_for_task(
    repo_root: str,
    agent_id: str,
    *,
    poll_interval_s: float = 0.5,
    timeout_s: Optional[float] = None,
    cancel_event: Optional[Any] = None,
    expected_parent_pid: Optional[int] = None,
    orchestrator_pid: Optional[int] = None,
    idle_warn_s: float = 3600.0,
    max_poll_s: Optional[float] = 86400.0,
) -> Optional[SubagentTask]:
    """Poll for a new task.  Blocks until a task appears or timeout expires.

    If *cancel_event* (a ``threading.Event``) is set, returns None immediately.
    Returns None on timeout, otherwise the deserialized SubagentTask.
    The task file is deleted on read so the orchestrator knows it was picked up.

    If *expected_parent_pid* is set, the loop self-exits once that pid is no
    longer alive — i.e. the spawning process died.
    This closes the orphan leak when the orchestrator is SIGKILL'd:
    ``_cleanup_ipc_workers`` (which writes the shutdown sentinel) never runs,
    so without this check a headless detached worker polls forever. Check
    strategy is platform-dependent:

    * **POSIX**: ``os.getppid() != expected_parent_pid`` — on parent death
      the kernel reparents orphans to init/subreaper, so ``getppid()`` changes
      instantly.  This is immune to the PID‑reuse race (a recycled PID will
      not match the new init PID of 1).
    * **Windows**: ``_is_process_alive(expected_parent_pid)`` — no reparenting
      occurs, so ``getppid()`` would keep returning the dead pid forever.
      A direct liveness probe is used instead.  This has a theoretical PID‑
      reuse race (the dead PID could be recycled to a live process), but no
      better alternative exists without ``psutil``.

    ``expected_parent_pid`` is captured ONCE at worker process start (not
    per-call) by the caller. It is only correct as an orphan signal on a
    DIRECT-CHILD launch path (headless background spawn) — on the macOS
    Terminal.app path (osascript → Terminal.app → login shell → worker) the
    worker's real parent is the login shell, not the orchestrator, so
    ``getppid()`` never reflects the orchestrator dying and this check never
    fires there.

    ``orchestrator_pid`` closes that gap: when given, it is probed DIRECTLY
    via :func:`_is_process_alive` (cross-platform, no reparenting semantics
    involved) every poll iteration, in addition to (not instead of) the
    ``expected_parent_pid`` check — either one firing self-exits the worker.
    Callers should pass the orchestrator's actual PID here whenever it is
    known (e.g. via a ``--orch-pid`` launch flag); ``expected_parent_pid``
    remains as the legacy fallback for callers that do not.

    Infinite-poll safety net: when *timeout_s* is None (the worker reuse path),
    the call is bounded by *max_poll_s* (default 24h) so a worker whose
    orchestrator is ALIVE but broken (never queues a task, never writes the
    shutdown sentinel) cannot poll forever — the orphan check only fires when
    the orchestrator process is DEAD. A finite *timeout_s* already bounds the
    call, so *max_poll_s* is inert there. *idle_warn_s* (default 1h) emits a
    WARNING each time that much continuous idle elapses with no task, surfacing
    pathological idling well before the cap. Set *max_poll_s* to None to disable
    the cap (legacy unbounded behaviour).
    """
    d = _subagent_dir(repo_root, agent_id)
    task_path = os.path.join(d, "task.json")

    # When the caller asked for an unbounded poll, apply max_poll_s as a safety
    # cap (default 24h). A finite timeout_s already provides a deadline.
    if timeout_s is None:
        effective_timeout = max_poll_s if max_poll_s is not None else float("inf")
    else:
        effective_timeout = timeout_s
    start_ts = time.monotonic()
    deadline = start_ts + effective_timeout
    last_warn_ts = start_ts
    interval = poll_interval_s
    while time.monotonic() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            logger.info("IPC: sub-agent %s poll cancelled", agent_id)
            return None
        # Orphan self-exit: if our spawning process is gone we have no one to
        # serve and no shutdown.json will ever arrive (the writer is dead).
        # Checked at idle (top of the loop), never mid-task.
        if expected_parent_pid is not None:
            if os.name != "nt":
                # POSIX reparents orphans to init (PID 1) or a subreaper on
                # parent death, so getppid() changes immediately.  Immune to
                # PID-reuse race — a recycled PID won't match.
                if os.getppid() != expected_parent_pid:
                    logger.warning(
                        "IPC: sub-agent %s orphaned — getppid()=%d != "
                        "expected_parent_pid=%d; self-exit.",
                        agent_id, os.getppid(), expected_parent_pid,
                    )
                    return None
            else:
                # Windows: getppid() stays at the dead pid forever (no
                # reparenting), so use kernel-level liveness probe instead.
                # Has a theoretical PID-reuse race (dead PID recycled to a
                # live process) but no better option without psutil.
                if not _is_process_alive(expected_parent_pid):
                    logger.warning(
                        "IPC: sub-agent %s orphaned — originator pid=%s gone; "
                        "self-exit to avoid an infinite poll.",
                        agent_id, expected_parent_pid,
                    )
                    return None
        # Direct orchestrator liveness probe (cross-platform, no reparenting
        # semantics involved): fires independently of the getppid() check
        # above, so it also catches the macOS Terminal.app launch path where
        # getppid() reflects the login shell, not the orchestrator.
        if orchestrator_pid is not None and not _is_process_alive(orchestrator_pid):
            logger.warning(
                "IPC: sub-agent %s orphaned — orchestrator pid=%s gone; "
                "self-exit to avoid an infinite poll.",
                agent_id, orchestrator_pid,
            )
            return None
        # 1) A pending task ALWAYS wins over the shutdown sentinel — never
        #    silently drop work the orchestrator explicitly queued.  This closes
        #    the race where a cancel/exception triggers ``_cleanup_ipc_workers``
        #    (writing shutdown.json) while a reusable worker still has an
        #    unflushed task.json: the old shutdown-first order dropped that task
        #    and exited.  The picked-up task's DesignChatLoop honors cancel_event
        #    itself, so picking it up on cancel is safe (it aborts fast), then
        #    the worker loops back and honors the shutdown below.
        #
        #    Atomic acquisition: rename task.json → task.json.claimed.<pid>.
        #    os.rename removes the source atomically, so across multiple worker
        #    processes polling the SAME directory only ONE rename succeeds — the
        #    losers get FileNotFoundError on the (now-gone) source.  This is the
        #    only correct cross-process exclusion primitive here (thread locks
        #    are useless across processes; the prior read-then-unlink had a TOCTOU
        #    window that allowed double-dispatch / lost tasks).  Reading happens
        #    AFTER the claim so a never-acquired task cannot be partially read.
        claimed_path = f"{task_path}.claimed.{os.getpid()}"
        acquired = False
        try:
            os.rename(task_path, claimed_path)  # atomic acquisition
            acquired = True
        except FileNotFoundError:
            pass  # No task yet (or another worker already claimed it).
        except OSError as e:
            # rename failed (e.g. cross-filesystem on an unusual mount). Fall
            # back to the legacy read-then-unlink path so deployment degrades
            # gracefully rather than hard-failing. (Same-dir rename is always
            # single-filesystem, so this branch is effectively unreachable.)
            logger.warning("IPC: rename-claim failed (%s); using legacy read path", e)
            claimed_path = task_path
            acquired = os.path.isfile(claimed_path)
        if acquired:
            # We won the claim (or fell back) — parse the claimed file. Reading
            # happens AFTER the atomic rename so a never-acquired task cannot be
            # partially read by a losing worker.
            try:
                with open(claimed_path, "r", encoding="utf-8") as f:
                    raw = f.read()
                data = json.loads(raw)
                task = SubagentTask.from_dict(data)
                # Clean up the claimed file (delete on read so the orchestrator
                # knows it was picked up; in the legacy fallback this removes
                # task.json itself).
                os.unlink(claimed_path)
                logger.info("IPC: sub-agent %s picked up task %s", agent_id, task.task_id)
                return task
            except (json.JSONDecodeError, ValueError, TypeError):
                # Malformed — quarantine so we do NOT spin forever re-reading and
                # log-spamming every poll.  Move it aside for diagnosis, then the
                # orchestrator's next write_task overwrites task.json cleanly.
                _bad = f"{claimed_path}.bad"
                try:
                    os.replace(claimed_path, _bad)
                except FileNotFoundError:
                    pass
                logger.warning(
                    "IPC: sub-agent %s found malformed task.json, quarantined → %s",
                    agent_id, _bad,
                )
            except FileNotFoundError:
                # claimed_path vanished between rename and read (e.g. an external
                # cleanup) — treat as "no task" and keep polling.
                pass
        # 2) Only honor the shutdown sentinel when idle (no pending task).  The
        #    orchestrator writes this to signal workers to exit when orchestration
        #    ends (prevents terminal/process leaks).
        if check_shutdown_sentinel(repo_root, agent_id):
            logger.info("IPC: sub-agent %s received shutdown sentinel", agent_id)
            # Consume (delete) the sentinel so it is fire-once — a future worker
            # launched into this directory must not see a stale shutdown.json.
            try:
                os.unlink(os.path.join(d, "shutdown.json"))
            except FileNotFoundError:
                pass
            return None
        # Idle diagnostic: surface pathological idling (orchestrator alive but
        # not dispatching) long before the max_poll_s cap kills the poll. The
        # warning repeats every idle_warn_s so a hung worker stays visible.
        _now = time.monotonic()
        if idle_warn_s > 0 and (_now - last_warn_ts) >= idle_warn_s:
            logger.warning(
                "IPC: sub-agent %s idle %.0fs with no task; still polling "
                "(orchestrator alive but not dispatching?).",
                agent_id, _now - start_ts,
            )
            last_warn_ts = _now
        time.sleep(interval)
        interval = _next_backoff(interval)  # ease idle FS load

    if timeout_s is None and max_poll_s is not None:
        logger.warning(
            "IPC: sub-agent %s hit the max_poll_s safety cap (%.0fs idle) — "
            "self-exiting instead of polling forever.",
            agent_id, max_poll_s,
        )
    else:
        logger.info("IPC: sub-agent %s poll timeout", agent_id)
    return None


def write_result(repo_root: str, result: SubagentResult) -> str:
    """Write a result back to the orchestrator.  Returns the file path."""
    d = _subagent_dir(repo_root, result.task_id)
    path = os.path.join(d, "result.json")
    _atomic_write(path, json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    logger.info("IPC: wrote result %s (%s) → %s", result.task_id, result.status, path)
    return path
