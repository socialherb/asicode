"""
OrchestratorAgent for asicode — Scheduler

Role: SCHEDULER only.
  - Executes subtasks with priority-aware parallel scheduling.
  - Manages sub-agent lifecycle, file locking, and review/retry gates.

  Two execution modes via run(request):
  - tool-loop mode (tool_loop_enabled=True, the primary mode used by CLI
    /orchestrate): the orchestrator drives an LLM-native tool loop that spawns
   sub-agents dynamically.
  - decomposition mode (tool_loop_enabled=False): run() self-decomposes the
   request via _decompose_task() and schedules subtasks with dependency
   awareness. Legacy path retained for backward compatibility.

  Execution flow (decomposition mode):
  request  (str)
       ↓
  OrchestratorAgent.run(request)
       ↓
  _run_dependency_aware(subtasks)   ← priority + dependency aware
       ↓
  _run_parallel_batch(batch)        ← ThreadPoolExecutor for parallel tasks
       ↓
  _run_subagent(subtask)            ← single AgentLoop instance per subtask

File-level locking prevents concurrent writes to the same file.
When a subagent hits max_turns, its partial results are forwarded as context
to dependent subtasks so work continues gracefully.
"""
from __future__ import annotations

import ast
import concurrent.futures
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Optional

from utils.llm_utils import simple_llm_call
from utils.string_helper import parse_json

# Imported at call sites (originally inline to avoid any chance of a circular
# import). agent_loop.py does NOT import orchestrator, so this is safe at module
# level — deduplicates 5 identical inline imports.
from .agent_loop import AgentResult

logger = logging.getLogger(__name__)

# ── File-lock registry (module-level, shared across all orchestrator instances) ──
# WeakValueDictionary so that a Lock for a path is GC'd once no live
# FileLockManager references it (via acquire()/_held). A plain dict would grow
# unboundedly in long-running server processes as unique file paths accumulate.
import weakref

_file_locks: "weakref.WeakValueDictionary[str, threading.Lock]" = weakref.WeakValueDictionary()
_file_locks_meta = threading.Lock()


def asr_subagent_argv(repo_root: str) -> list[str]:
    """Return the argv that invokes the ``asi --subagent`` worker.

    SINGLE SOURCE OF TRUTH for "how do we launch the worker": prefer
    ``<sys.executable> <repo>/asi.py --subagent`` (works when the CLI is run
    directly as ``python asi.py`` and the bare command is not on PATH) over a
    bare ``asi --subagent`` (pip-installed entry point).

    Returns an argv LIST, never a shell string — so callers never round-trip
    through ``shlex.split``.  That is critical on Windows, where
    ``sys.executable`` contains backslashes that POSIX ``shlex.split`` would eat
    (``C:\\Python\\python.exe`` → ``C:Pythonpython.exe`` → FileNotFoundError on
    Popen).  Callers that need a shell string (the macOS Terminal.app ``do
    script``) derive it with ``shlex.join`` from this same list, so the launch
    decision lives in exactly one place.
    """
    _script = os.path.join(repo_root, "asi.py")
    if os.path.isfile(_script):
        return [sys.executable, _script, "--subagent"]
    # Fallback: assume pip-installed entry point in PATH.
    return ["asi", "--subagent"]


class _DummyLock:
    """No-op lock for paths outside the repo or invalid paths.

    Module-level singleton avoids re-creating the class on every acquire() call
    when locking is skipped (common for non-repo paths).
    """

    def acquire(self, blocking=True, timeout=-1):
        return True

    def release(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


_DUMMY_LOCK = _DummyLock()


class FileLockManager:
    """Acquire per-file threading locks before write operations."""

    def __init__(self, repo_root: Optional[str] = None):
        self.repo_root = repo_root
        # Normalised path → Lock object this manager instance has acquired but
        # not yet released. Two jobs:
        #   1. reset() uses it to release ONLY our own locks — never locks held
        #      by other Orchestrator instances sharing _file_locks.
        #   2. It holds a STRONG reference to each Lock for the duration of the
        #      hold. _file_locks is a WeakValueDictionary, so without this strong
        #      ref the Lock would be GC'd mid-critical-section, the registry
        #      entry would vanish, and a concurrent acquirer of the same path
        #      would mint a fresh Lock — silently breaking mutual exclusion.
        self._held: dict[str, threading.Lock] = {}

    def _normalize_path(self, path: str) -> Optional[str]:
        """Convert a path to a canonical absolute realpath for use as lock key.

        Returns None if the path is outside repo_root or invalid.
        """
        if self.repo_root is None:
            # No repo root, use path as-is (backward compatibility)
            return path

        from path_security import normalize_rel_path

        # First normalize as repo-relative path
        rel_path = normalize_rel_path(path)
        if not rel_path:
            # Path is invalid or outside repo, skip locking
            return None

        # Convert to absolute path and resolve symlinks
        abs_path = os.path.join(self.repo_root, rel_path)
        try:
            real_path = os.path.realpath(abs_path)
            return real_path
        except Exception:
            # If realpath fails, fall back to absolute path
            return os.path.abspath(abs_path)

    def acquire(self, path: str) -> threading.Lock:
        norm_path = self._normalize_path(path)
        if norm_path is None:
            # Path is invalid or outside repo - skip locking
            return _DUMMY_LOCK

        with _file_locks_meta:
            lock = _file_locks.setdefault(norm_path, threading.Lock())
        lock.acquire()
        try:
            with _file_locks_meta:
                # Strong-ref the Lock while held (see __init__ note) so the
                # WeakValueDictionary cannot GC it out from under us. NOTE: this
                # guard is NOT re-entry protection — threading.Lock is
                # non-reentrant, so re-acquiring an already-held path would block
                # forever inside lock.acquire() above, never reaching here. The
                # guard only avoids a redundant _held entry. Re-acquisition never
                # happens in practice because acquire_relevant() dedups its path
                # set and the sole caller (ToolRegistry) releases in a try/finally.
                if norm_path not in self._held:
                    self._held[norm_path] = lock
        except BaseException:
            lock.release()
            raise
        return lock

    def release(self, path: str) -> None:
        norm_path = self._normalize_path(path)
        if norm_path is None:
            return
        self._release_by_normalized_path(norm_path)

    def _release_by_normalized_path(self, norm_path: str) -> None:
        """Release a lock keyed by an ALREADY-normalized path.

        Symmetric with _acquire_by_normalized_path. Must NOT re-normalize:
        _normalize_path is not idempotent (feeding it an absolute key yields
        ``repo_root + repo_root + …``), so re-normalizing acquire_relevant()'s
        returned keys would look up the wrong entry and leak the lock forever.
        """
        with _file_locks_meta:
            # Prefer the strongly-held Lock; fall back to the registry only if
            # this manager never recorded it (defensive).
            lock = self._held.pop(norm_path, None) or _file_locks.get(norm_path)
        if lock:
            try:
                lock.release()
            except RuntimeError:
                pass  # already released

    @staticmethod
    def _patch_target_paths(patch: str) -> list[str]:
        """Extract target file paths from a unified-diff ``patch`` string.

        apply_patch carries its file targets INSIDE the diff body (``diff --git``
        / ``+++ b/`` headers), not in a scalar ``path`` arg — so without parsing
        the patch, ``acquire_relevant`` would lock ZERO files for the primary
        write tool, letting two parallel sub-agents patch the same file with no
        mutual exclusion. Mirrors ``WriteSafetyManager.snapshot_target_files``.
        """
        targets: list[str] = []
        for _line in patch.splitlines():
            if _line.startswith("diff --git "):
                _parts = _line.split()
                if len(_parts) >= 4 and _parts[3].startswith("b/"):
                    targets.append(_parts[3][2:])
            elif _line.startswith("--- a/") or _line.startswith("+++ b/"):
                _p = _line[6:].strip()
                if _p and _p != "/dev/null":
                    targets.append(_p)
        return targets

    @staticmethod
    def _plan_target_paths(plan: Any) -> list[str]:
        """Extract target file paths from a ``write_plan`` ``plan`` arg.

        ``plan`` may be a unified-diff string (delegated to :meth:`_patch_target_paths`)
        or a structured dict ``{"ops": [{"path": ...}, ...]}`` (the common form for
        write_plan, which is the primary multi-file write tool). Without this,
        ``acquire_relevant`` checked only the ``patch`` key and locked ZERO files for
        write_plan, so two concurrent sessions editing the same file via write_plan got
        no mutual exclusion — the exact snapshot-rollback-overwrite race the FileLockManager
        exists to prevent. Mirrors the dict-plan branch of
        ``WriteSafetyManager.snapshot_target_files`` and ``_extract_write_target_paths``.
        """
        if isinstance(plan, str):
            return FileLockManager._patch_target_paths(plan)
        targets: list[str] = []
        if isinstance(plan, dict):
            plan_ops = plan.get("ops") or plan.get("operations") or []
            # A bare {"path": ...} plan (no ops list) targets that single file.
            if not plan_ops and "path" in plan:
                plan_ops = [plan]
            for _op in plan_ops:
                if isinstance(_op, dict):
                    _p = _op.get("path")
                    if _p:
                        targets.append(str(_p))
        return targets

    def acquire_relevant(self, tool_args: dict[str, Any]) -> list[str]:
        """Acquire locks for all file paths referenced in tool_args. Returns locked paths."""
        # Collect and normalize all paths
        norm_paths_set = set()
        for key in ("path", "file_path", "src", "dst"):
            val = tool_args.get(key)
            if val and isinstance(val, str):
                norm_path = self._normalize_path(val)
                if norm_path is not None:
                    norm_paths_set.add(norm_path)
        # apply_patch: pull targets out of the unified-diff body (see helper).
        _patch = tool_args.get("patch")
        if _patch and isinstance(_patch, str):
            for _rel in self._patch_target_paths(_patch):
                _np = self._normalize_path(_rel)
                if _np is not None:
                    norm_paths_set.add(_np)
        # write_plan: pull targets out of the ``plan`` arg (string diff OR dict with
        # ops). write_plan is the primary multi-file write tool (create_file / patch
        # ops), and its targets live inside ``plan`` — NOT in ``path``/``file_path``
        # — so the scalar-key scan above finds nothing for it. Without this branch
        # write_plan locks ZERO files (see _plan_target_paths). Mirrors
        # snapshot_target_files / _extract_write_target_paths target resolution.
        _plan = tool_args.get("plan")
        if _plan is not None:
            for _rel in self._plan_target_paths(_plan):
                _np = self._normalize_path(_rel)
                if _np is not None:
                    norm_paths_set.add(_np)

        # Sort to ensure consistent lock acquisition order (deadlock prevention)
        norm_paths = sorted(norm_paths_set)

        # Acquire locks transactionally. The sole production caller
        # (ToolRegistry) assigns this method's *return value* to locked_paths
        # and releases exactly those paths in a try/finally — so a mid-loop
        # raise (e.g. MemoryError during the _held metadata update) would
        # leave locked_paths empty and orphan the already-acquired locks in
        # self._held until session reset(). On failure we release whatever we
        # acquired so far before re-raising; the failed path's own lock is
        # already released by _acquire_by_normalized_path's inner handler.
        acquired: list[str] = []
        try:
            for p in norm_paths:
                self._acquire_by_normalized_path(p)
                acquired.append(p)
        except BaseException:
            if acquired:
                self.release_all(acquired)
            raise
        return norm_paths

    def _acquire_by_normalized_path(self, norm_path: str) -> threading.Lock:
        """Acquire lock by already normalized path (internal use)."""
        with _file_locks_meta:
            lock = _file_locks.setdefault(norm_path, threading.Lock())
        lock.acquire()
        try:
            with _file_locks_meta:
                # Strong-ref the Lock while held (see __init__ note). Same
                # non-reentry caveat as acquire() — this guard only avoids a
                # redundant _held entry, it does not prevent re-acquire deadlock.
                if norm_path not in self._held:
                    self._held[norm_path] = lock
        except BaseException:
            lock.release()
            raise
        return lock

    def release_all(self, paths: list[str]) -> None:
        # paths are the ALREADY-normalized keys returned by acquire_relevant();
        # release by normalized path (no re-normalization — see method note).
        for p in paths:
            self._release_by_normalized_path(p)

    def reset(self) -> None:
        """Release locks held by THIS manager instance only.

        Called when an orchestration session ends (or starts, defensively) to
        release any locks this instance still holds. It must NOT release locks
        held by other Orchestrator instances, and must NOT clear the shared
        _file_locks registry — doing so would destroy the shared lock identity
        and break mutual exclusion across concurrently running orchestrators.

        (Note: Python's threading.Lock.release() does NOT verify ownership, so
        the previous implementation silently released other threads' locks.)
        """
        with _file_locks_meta:
            locks_to_release = list(self._held.items())
            self._held.clear()
        for _path, lock in locks_to_release:
            if lock is not None:
                try:
                    lock.release()
                except RuntimeError:
                    pass  # already released (e.g. caller called lock.release() directly)


class OrderedEventDispatcher:
    """
    Dispatcher that guarantees event ordering in a multi-agent environment.

    Features:
    - Per-agent event ordering guarantee
    - Global sequence number assignment
    - Automatic timestamp addition
    """

    def __init__(self, callback: Callable[[str, dict[str, Any]], None]):
        """
        Args:
            callback: Callback function that sends the actual event (existing _cb_fn)
        """
        self._callback = callback
        self._lock = threading.Lock()
        self._agent_last_seq: dict[str, int] = defaultdict(int)
        self._global_seq_counter = 0  # Monotonic counter for deterministic testing

    def emit(self, agent_id: str, event: str, data: dict[str, Any]) -> None:
        """
        Emit an event with per-agent ordering guarantee.

        Args:
            agent_id: ID of the agent that triggered the event
            event: Event type (subagent_start, subagent_complete, etc.)
            data: Event data
        """
        with self._lock:
            # Deterministic global sequence counter for testing consistency
            self._global_seq_counter += 1
            global_seq = self._global_seq_counter

            # Update per-agent sequence
            agent_seq = self._agent_last_seq[agent_id] + 1
            self._agent_last_seq[agent_id] = agent_seq

            # Add metadata
            enriched_data = dict(data)
            enriched_data.update({
                "global_sequence_id": global_seq,
                "agent_sequence_id": agent_seq,
                "agent_id": agent_id,
                "timestamp": time.monotonic(),
                "event_type": event
            })

            # Invoke callback
            self._callback(event, enriched_data)



# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SubTaskSpec:
    task_id: str          # "dev_1", "dev_2" …
    title: str
    description: str      # the actual task text for the sub-agent
    assigned_files: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)  # task_id of dependencies
    priority: int = 0     # 0=highest (run first), 1=normal, 2=lowest
    # "planner" → AST-only files; enables internal planning phase in AgentLoop.
    # "main_agent" → non-AST files or mixed; keeps planning disabled (more flexible).
    preferred_lane: str = "main_agent"


@dataclass
class OrchestratorConfig:
    max_subagents: int = 3
    parallel: bool = True
    agent_config: Optional[Any] = None  # AgentConfig forwarded to each SubAgent
    cancel_event: Optional[threading.Event] = None  # For server-side cancellation
    subagent_provider: Optional[str] = None   # e.g. "ollama"
    subagent_model: Optional[str] = None      # e.g. "qwen2.5-coder:3b"
    subagent_api_key: Optional[str] = None
    subagent_ollama_url: Optional[str] = None  # custom Ollama endpoint
    # Per-subagent model overrides keyed by 1-based slot number as string
    # ("1" → first spawned sub-agent).  Value: (provider, model, api_key).
    # Resolution (see OrchestratorAgent._resolve_subagent_model):
    #   explicit slot config → lowest configured dev slot (fallback) →
    #   orchestrator model.  Set via the REPL ``/model dev_N <model>`` command.
    subagent_models: dict = field(default_factory=dict)
    # Orchestrator review: after each subagent completes, the orchestrator LLM
    # checks the git diff against the original subtask and may request a retry.
    # Default is False so the Orchestrator stays LLM-free by default.
    # Set to True to enable the optional quality-gate review loop.
    review_enabled: bool = False
    review_max_retries: int = 1
    # Char cap for the git diff fed to the reviewer LLM.  The old hard-coded
    # 3000 was too small for meaningful review of real patches; 8000 (≈2K
    # tokens) gives the reviewer enough context without overflowing it.
    review_diff_char_limit: int = 8000
    # Wall-clock timeout (seconds) for a single IPC sub-agent run — applies to
    # both the initial dispatch and each review-driven retry.
    ipc_timeout_s: float = 600.0
    # Startup-failure deadline (seconds) for an IPC sub-agent: if the worker has
    # not produced its FIRST heartbeat within this window, wait_for_result
    # presumes the worker never started (launch failed, import error) and bails
    # instead of burning the full ipc_timeout_s. Default 0 = disabled (safe for
    # deployments that may run heartbeat-incapable worker binaries); opt in once
    # every worker is heartbeat-capable. Inert after the first heartbeat.
    ipc_startup_timeout_s: float = 0.0
    # Soft-timeout hard cap (seconds) for an IPC sub-agent. If > ipc_timeout_s, a
    # FRESH heartbeat (worker alive and progressing) extends the run deadline to
    # this value once — so a legitimately-slow task (long LLM turn / slow tool)
    # is not abandoned the instant it crosses ipc_timeout_s. Only a stale
    # heartbeat still kills a dead worker early. Default 0 = disabled (hard
    # ipc_timeout_s governs). Pair with heartbeat-capable workers.
    ipc_max_timeout_s: float = 0.0
    # Worker-liveness guard (seconds): a heartbeat older than this presumes the
    # worker is dead and bails early. ``wait_for_result`` already accepts it as a
    # param, but it was hard-wired to the HEARTBEAT_STALE_S constant (120s) with
    # no config exposure. Slow local models can legitimately pause long between
    # heartbeats; raising this avoids false "dead" verdicts. Default matches the
    # constant so behaviour is unchanged unless configured.
    ipc_heartbeat_stale_s: float = 120.0
    # Sub-agent execution mode:
    #   "in_process" (default) — AgentLoop runs in this process (blocking).
    #   "ipc" — write task.json, an external `asi --subagent` process picks
    #           it up and writes result.json back.  Enables true parallelism and
    #           lets each sub-agent run in its own terminal window.
    subagent_mode: str = "in_process"
    # Auto-launch Terminal.app (macOS only): when True and subagent_mode == "ipc",
    # the orchestrator runs an AppleScript that opens a new Terminal window
    # running `asi --subagent --subagent-id <id>`.  No-op on non-macOS.
    auto_launch_terminal: bool = False
    # Auto-spawn the worker as a HEADLESS background subprocess (cross-platform):
    # when True and subagent_mode == "ipc", the orchestrator launches the same
    # `asi --subagent` worker without a terminal window.  This is what makes
    # IPC work on Linux/Windows (no Terminal.app), and also serves as the macOS
    # fallback when the visible Terminal.app launch fails.  Output is redirected
    # to `.asicode/subagents/<id>/worker.log`.
    auto_spawn_worker: bool = False
    # ── Tool-loop mode ────────────────────────────────────────────────────
    # When True, the orchestrator drives an LLM-native tool loop itself
    # (DesignChatLoop style): it calls `spawn_subagent` / `poll_subagent` to
    # launch sub-agents in the BACKGROUND and may freely interleave read tools
    # (read_file, get_file_outline, bash, …) while sub-agents run.  This is the
    # inverse of the legacy decomposition path, where _run_dependency_aware()
    # BLOCKS the orchestrator thread until every sub-agent finishes.
    tool_loop_enabled: bool = False
    # Max LLM iterations in the orchestrator tool loop before forcing a final
    # synthesis (each iteration = one chat_with_tools call + tool dispatch).
    tool_loop_max_iterations: int = 40
    # Thinking-mode overrides forwarded to the DesignChatLoop the orchestrator
    # drives in tool-loop mode (mirrors the interactive REPL's thinking state).
    thinking_mode: Optional[bool] = None
    reasoning_effort: Optional[str] = None
    # DesignSessionManager forwarded to the DesignChatLoop in tool-loop mode, so the
    # orchestrator can call search_design_history (recall past decisions) exactly like
    # the interactive design-chat.  None disables that one tool.
    session_mgr: Optional[Any] = None
    # ── Scope-violation policy ─────────────────────────────────────────────
    # What to do when a sub-agent makes genuine out-of-scope writes (files outside
    # its assignment that are NOT pre-run baseline dirt, NOT a parallel peer's own
    # assigned file, and NOT infra — see ``_filter_unassigned_changes``).
    #   "warn"   (default, current behaviour) — log + surface in the completion event.
    #   "revert" — restore tracked via ``git checkout HEAD --``, unlink untracked
    #              (created-during-run). Applied (a) before a review retry so the
    #              retry starts from a clean baseline, and (b) at completion so the
    #              strays do not linger in the final tree.
    #   "fail"   — promote the sub-agent's result status to "error".
    #
    # CAVEAT (shared-worktree attribution): ``revert`` derives the candidate set
    # from a GLOBAL ``git status`` of the one shared working tree (in-process via
    # ``_git_status_changed_paths``; IPC via ``partition_changed_files``). The
    # filter subtracts baseline / every worker's assigned files / infra — but a
    # STILL-RUNNING peer's OWN stray (a file that peer wrote outside ITS
    # assignment) is none of those, so it is NOT subtracted and can be
    # mis-attributed to the completing worker and reverted, destroying the peer's
    # in-progress work. This is the documented shared-worktree limitation (see
    # ``_filter_unassigned_changes``) — a true fix needs per-worker git worktrees.
    # ``warn`` / ``fail`` are non-destructive and unaffected. Under parallel
    # execution prefer ``warn`` (the default) unless you accept this risk.
    scope_violation_policy: str = "warn"

    _SCOPE_VIOLATION_POLICIES = ("warn", "revert", "fail")

    def __post_init__(self) -> None:
        # Fail fast on a typo'd policy (e.g. "reverd") instead of silently
        # degrading to "warn" inside _apply_scope_violation_policy — a mis-set
        # policy is a config error the operator should hear about immediately.
        if self.scope_violation_policy not in self._SCOPE_VIOLATION_POLICIES:
            raise ValueError(
                f"scope_violation_policy={self.scope_violation_policy!r} "
                f"is invalid; expected one of {self._SCOPE_VIOLATION_POLICIES}"
            )


@dataclass
class OrchestratorResult:
    status: str           # "success" | "partial" | "error"
    summary: str
    subtask_results: list[Any] = field(default_factory=list)  # List[AgentResult]
    total_turns: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Decompose prompt ──────────────────────────────────────────────────────────

def _build_git_context(repo_root: Optional[str]) -> str:
    """Return a compact git-history block for the planner to avoid regression.

    Includes:
    - Last 10 commit messages (intent of recent changes)
    - Files modified in the last 3 commits (what is "in flight")

    Returns empty string on any failure so callers never crash.
    """
    if not repo_root:
        return ""
    import subprocess as _sp
    try:
        log = _sp.run(
            ["git", "log", "--oneline", "-10"],
            cwd=repo_root, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        changed = _sp.run(
            ["git", "diff", "--name-only", "HEAD~3..HEAD"],
            cwd=repo_root, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, _sp.SubprocessError):
        return ""
    if not log:
        return ""
    parts = ["## Recent git history (last 10 commits)"]
    parts.append(log)
    if changed:
        parts.append("\n## Files modified in last 3 commits (treat carefully)")
        parts.append(changed)
    parts.append(
        "\nIMPORTANT: Do NOT create subtasks that undo or re-introduce changes "
        "from the commits above. If a recent commit removed something, "
        "do not add it back."
    )
    return "\n".join(parts)


# Max chars of a sub-agent's ``final_message`` reported back to the
# orchestrator LLM via ``poll_subagent``.  Generous (≈4K tokens): normal
# LLM-generated summaries pass through entirely; the cap is only a safety
# net against pathological output bloating the orchestrator's context.
_POLL_RESULT_CAP = 16000

# Worker log rotation cap: a reused worker is ONE long-lived process serving
# many tasks, appending to a single ``worker.log``. Without a bound a long
# orchestration could grow it to hundreds of MB. When the existing log exceeds
# this size at spawn time it is rotated to ``worker.log.old`` (single
# generation). 10 MB is generous for diagnostics while staying negligible on
# disk.
_WORKER_LOG_ROTATE_BYTES = 10 * 1024 * 1024


def _coerce_priority(value, default: int = 1) -> int:
    """Coerce an untrusted LLM ``priority`` value to int (default=1=normal).

    Defaults to 1 (NOT the task index) so independent tasks share one
    priority and are scheduled in parallel — index-based defaults would make
    every task sequential even when no dependencies exist.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_str_list(value) -> list[str]:
    """Coerce an untrusted LLM field into ``list[str]``; non-lists → empty.

    Guards ``assigned_files`` / ``dependencies``: a stray string or dict
    would otherwise break the downstream ``set()`` / ``zip()`` calls.
    """
    if not isinstance(value, list):
        return []
    return [str(x) for x in value if x is not None]


# Cap on how many files a single directory entry may expand to. A planner
# that assigns "." or a large top-level directory would otherwise expand to
# every file in the repo — _capture_assigned_snapshots then loads ALL of
# their bytes into memory (worse with binaries in the tree) and the
# resulting assigned_files list bloats task.json. Above the cap we abandon
# the expansion and fall back to leaving the entry as the bare directory
# string, which reproduces the PRE-existing (already-known, bounded) bug —
# that one directory's snapshot/revert/scope-matching is silently skipped —
# rather than risking an OOM/multi-MB task.json for the whole run.
_MAX_DIR_EXPANSION_FILES = 500


def _expand_directory_assignments(repo_root: Optional[str], files: list[str]) -> list[str]:
    """Expand any directory entries in *files* into the files they contain.

    The planner LLM's ``assigned_files`` field is validated as ``list[str]``
    (``_coerce_str_list``) but never checked for "is this actually a file". If
    the LLM assigns a directory (e.g. ``"src/utils/"``),
    ``_capture_assigned_snapshots`` silently drops it (``open()`` on a
    directory raises ``IsADirectoryError``, caught by a broad ``except``), so
    NONE of that directory's edits get a pre-run snapshot and every revert
    path (review rejection, cancelled/error, abandon) leaves partial edits
    behind. Downstream scope matching (``partition_changed_files``,
    ``_filter_unassigned_changes``) also does exact-path comparison, so any
    legitimate edit under that directory reads as an out-of-scope violation.
    Expanding here — once, at assignment time — keeps every downstream
    consumer's exact-path comparison correct with no further changes needed.

    A path that does not exist yet (a file the sub-agent is expected to
    CREATE) passes through unchanged — only EXISTING directories are
    expanded. ``.git`` is skipped during the walk. A directory that would
    expand past ``_MAX_DIR_EXPANSION_FILES`` is left UNEXPANDED (see the
    constant's docstring) instead of risking a memory blowup.
    """
    if not isinstance(repo_root, str) or not repo_root or not files:
        return list(files or [])
    expanded: list[str] = []
    for f in files:
        try:
            _abs = os.path.join(repo_root, f)
            _is_dir = os.path.isdir(_abs)
        except (TypeError, ValueError, OSError):
            expanded.append(f)
            continue
        if _is_dir:
            _dir_files: list[str] = []
            _over_cap = False
            for _dirpath, _dirnames, _filenames in os.walk(_abs):
                _dirnames[:] = [d for d in _dirnames if d != ".git"]
                for _fn in _filenames:
                    _full = os.path.join(_dirpath, _fn)
                    _dir_files.append(os.path.relpath(_full, repo_root))
                    if len(_dir_files) > _MAX_DIR_EXPANSION_FILES:
                        _over_cap = True
                        break
                if _over_cap:
                    break
            if _over_cap:
                logger.warning(
                    "assigned_files entry %r is a directory with more than "
                    "%d files; NOT expanding (would risk a memory blowup on "
                    "snapshot capture) — left as a bare directory path, so "
                    "its snapshot/revert/scope-matching will be skipped. "
                    "Assign specific files instead.",
                    f, _MAX_DIR_EXPANSION_FILES,
                )
                expanded.append(f)
            else:
                logger.warning(
                    "assigned_files entry %r is a directory; expanded to %d "
                    "file(s) so snapshot/revert/scope-matching stay correct.",
                    f, len(_dir_files),
                )
                expanded.extend(_dir_files)
        else:
            expanded.append(f)
    return expanded


def _norm_assigned_file(path: str) -> str:
    """Canonicalize a file path for conflict comparison.

    ``normalize_rel_path`` strips ``./`` prefixes, normalizes separators and
    git ``a/`` / ``b/`` prefixes, so ``src/foo.py`` and ``./src/foo.py``
    compare equal (otherwise two tasks editing the same file slip through the
    conflict guard and run in parallel).  Falls back to the raw path if
    normalization rejects it, keeping distinct invalid paths distinct instead
    of collapsing them all to ``""``.
    """
    if not path:
        return ""
    try:
        from path_security import normalize_rel_path

        norm = normalize_rel_path(path)
        return norm if norm else str(path)
    except Exception:
        return str(path)


def _split_batch_by_file_conflict(specs: list["SubTaskSpec"]) -> list[list["SubTaskSpec"]]:
    """Partition specs into sequential conflict-free sub-batches.

    Two tasks conflict when their assigned_files overlap.  Greedy bin-packing:
    place each task in the first existing sub-batch that has no file overlap;
    otherwise open a new sub-batch.  Tasks within each sub-batch are safe to
    run in parallel; sub-batches must run sequentially.

    Performance: each sub-batch keeps a cumulative normalized file-set
    (``sub_batch_files``, parallel to ``sub_batches``) so placement is O(files)
    per spec instead of re-scanning every member of every sub-batch (O(N²·M)).

    Unscoped tasks (empty ``assigned_files``) are handled CONSERVATIVELY: their
    real write set is unknown, so two of them may well edit the same file, and
    EVERY downstream guard keys off ``assigned_files`` — a task with none gets
    NO pre-run snapshot (``_capture_assigned_snapshots`` returns ``{}``), NO
    scope-violation detection (``_detect_genuine_violations`` is skipped when
    ``not assigned_files``), and NO review (``_review_subagent_diff`` short-
    circuits on ``not subtask.assigned_files`` because the diff cannot be
    attributed). That means an unscoped task racing a sibling in parallel could
    silently clobber its writes with no guard left to catch it. To stay safe we
    therefore isolate every unscoped spec in its OWN sub-batch (serialized both
    from other unscoped specs and from every file-bearing spec), instead of the
    old behavior where ``not spec_files`` let it ride in the first sub-batch.
    """
    sub_batches: list[list["SubTaskSpec"]] = []
    sub_batch_files: list[set[str]] = []  # cumulative file-set, parallel to sub_batches
    for spec in specs:
        spec_files = {_norm_assigned_file(f) for f in (spec.assigned_files or [])}
        # An unscoped spec (empty assigned_files) cannot be conflict-checked —
        # its real write set is unknown — so never co-locate it with anything.
        # Each gets its own sub-batch (the new-sub-batch path below), which
        # serializes it against every other spec.
        placed = False
        if spec_files:
            for i, sbf in enumerate(sub_batch_files):
                # ``not sbf`` means this sub-batch was seeded by an unscoped
                # spec — skip it so a file-bearing spec never rides along in
                # an unscoped spec's batch (and vice-versa).
                if sbf and not sbf.intersection(spec_files):
                    sub_batches[i].append(spec)
                    sbf.update(spec_files)
                    placed = True
                    break
        if not placed:
            sub_batches.append([spec])
            sub_batch_files.append(set(spec_files))
    return sub_batches if sub_batches else [[]]


def _kind_initial(kind: str) -> str:
    """One-letter kind tag for symbol hints (e.g. "function" -> "F")."""
    return (kind or "?")[:1].upper()


class _NativeToolError(Exception):
    """Raised by orchestrator-native tool handlers for expected/handler-level
    errors (bad args, unknown id, unknown tool name).

    Distinct from an unexpected exception so that ``_OrchestratorBackedRegistry
    .dispatch`` can map it to ``ToolResult(ok=False, content=str(exc))`` —
    WITHOUT a noisy ``logger.exception`` traceback — while a bare ``Exception``
    (genuine crash) still gets full logging. Before this, handlers returned
    ``"Error: ..."`` strings that dispatch unconditionally wrapped in
    ``ToolResult(ok=True, ...)``, corrupting ok/failure semantics.
    """


# Sentinel marking "file did not exist before this sub-agent ran" in a snapshot,
# so the restore step can ``os.remove`` files the sub-agent created from scratch.
_MISSING_SNAP = object()

# Maximum bytes a single file snapshot may consume in RAM. Files above this
# threshold are skipped (logged) and left unrestorable for that file — the same
# degraded-but-safe semantics as unreadable files (B6 / turn 13116).
_SNAPSHOT_MAX_BYTES = 5 * 1024 * 1024  # 5 MiB

# Aggregate RAM cap across ALL snapshotted files in one batch. The per-file cap
# alone (5 MiB) still lets a 200-file assignment park ~1 GiB of pre-run bytes in
# RAM (revert snapshots are held for the whole sub-agent run). This budget caps
# the whole dict; files that don't fit are skipped with a warning — same
# degraded-revert semantics as oversized/unreadable files (no revert for them).
# Missing files (_MISSING_SNAP sentinel) are exempt: they're tiny and needed so
# a created-during-run file can be reverted by deletion.
_SNAPSHOT_AGGREGATE_MAX_BYTES = 64 * 1024 * 1024  # 64 MiB

# Per-file read cap for _synthesize_untracked_diff: the reviewer prompt is
# capped at review_diff_char_limit (~8000 chars) anyway, so reading more than
# this per file only burns RAM before truncation. 64 KiB comfortably covers
# the whole prompt budget while bounding the read of a huge generated file.
_SYNTH_DIFF_FILE_BYTES = 64 * 1024


def _capture_assigned_snapshots(repo_root: str, file_paths: list[str]) -> dict:
    """Capture pre-run bytes of assigned files so a rejected attempt can be
    reverted without clobbering the user's uncommitted edits or cross-batch
    shared-file changes (``git checkout HEAD --`` would destroy both).

    Files larger than ``_SNAPSHOT_MAX_BYTES`` (5 MiB) are skipped with a
    warning — their bytes are NOT loaded into RAM, and they are omitted from
    the snapshot so no revert is attempted for them.

    The TOTAL bytes loaded is bounded by ``_SNAPSHOT_AGGREGATE_MAX_BYTES``
    (64 MiB): the per-file cap alone still lets a large assignment park ~1 GiB
    in RAM (snapshots are held for the whole sub-agent run). Once the aggregate
    budget is exhausted, remaining files are skipped with a single warning —
    same degraded-revert semantics (no revert for them). Missing files keep the
    ``_MISSING_SNAP`` sentinel regardless of the cap (they're tiny and needed
    to revert a file created during the run by deletion).

    Returns ``{rel_path: bytes | _MISSING_SNAP}``.  Unreadable or oversized
    files are omitted (no revert attempted for them).
    """
    snaps: dict[str, object] = {}
    _cum = 0
    _agg_cap_hit = False
    _loaded = 0  # files actually read into the byte budget (excludes _MISSING_SNAP sentinels)
    for _fp in (file_paths or []):
        _abs = os.path.join(repo_root, _fp) if repo_root else _fp
        if os.path.isdir(_abs):
            # Cap-exceeded directory (not expanded to individual files).
            # No per-file snapshot is possible — the subtractive filter and
            # scope-violation checks still work (they're prefix-aware), but
            # revert for files inside this directory is DEGRADED.
            logger.warning(
                "Snapshot degraded for cap-exceeded directory %r — "
                "individual files within cannot be snapshotted for revert.",
                _fp,
            )
            continue
        # Resolve size first so the missing-file sentinel is recorded even when
        # the aggregate cap is already exhausted (a created-during-run file must
        # still be revertible by deletion).
        try:
            _size = os.path.getsize(_abs)
        except FileNotFoundError:
            snaps[_fp] = _MISSING_SNAP
            continue
        except OSError:
            continue  # unreadable: leave out (no revert for this file)
        # Check size BEFORE reading to avoid loading huge files into RAM
        if _size > _SNAPSHOT_MAX_BYTES:
            logger.warning(
                "Snapshot skip %r — file size %d bytes exceeds %d byte cap; "
                "no revert available for this file.",
                _fp, _size, _SNAPSHOT_MAX_BYTES,
            )
            continue  # same semantics as unreadable: no revert for this file
        if _cum + _size > _SNAPSHOT_AGGREGATE_MAX_BYTES:
            # Aggregate budget exhausted — skip this and subsequent files.
            # Warn ONCE (not per-file) to avoid log spam on a 200-file batch.
            if not _agg_cap_hit:
                logger.warning(
                    "Snapshot aggregate cap %d bytes exceeded after %d "
                    "file(s); remaining assigned files skip revert snapshot "
                    "(per-file revert degraded for them).",
                    _SNAPSHOT_AGGREGATE_MAX_BYTES, _loaded,
                )
                _agg_cap_hit = True
            continue
        try:
            with open(_abs, "rb") as _f:
                snaps[_fp] = _f.read()
            _cum += _size
            _loaded += 1
        except FileNotFoundError:
            snaps[_fp] = _MISSING_SNAP
        except Exception:
            pass  # unreadable: leave out (no revert for this file)
    return snaps


def _restore_assigned_snapshots(repo_root: str, snaps: dict) -> list[str]:
    """Inverse of :func:`_capture_assigned_snapshots`.

    Restores each file to its pre-run bytes (or removes it if it did not
    exist before).  Returns the list of rel-paths actually reverted.
    """
    reverted: list[str] = []
    # Best-effort cleanup of stale .asi-revert-* tempfiles orphaned by a
    # crash/SIGKILL during a prior restore. Scan each unique directory once.
    _stale_deadline = time.time() - 86400  # 1 day
    _seen_dirs: set[str] = set()
    for _fp, _data in snaps.items():
        _abs = os.path.join(repo_root, _fp) if repo_root else _fp
        _dir = os.path.dirname(_abs) or "."
        if _dir not in _seen_dirs:
            _seen_dirs.add(_dir)
            try:
                for _tf in os.listdir(_dir):
                    if _tf.startswith(".asi-revert-"):
                        _tp = os.path.join(_dir, _tf)
                        if os.path.isfile(_tp) and os.path.getmtime(_tp) < _stale_deadline:
                            os.unlink(_tp)
            except OSError:
                pass  # directory gone, permission denied, etc. — non-critical
        try:
            if _data is _MISSING_SNAP:
                if os.path.exists(_abs):
                    os.remove(_abs)
                    # Clean up empty parent directories left over when a
                    # freshly-created file tree is reverted. os.removedirs()
                    # walks leaf→root, stopping at the first non-empty dir.
                    try:
                        os.removedirs(os.path.dirname(_abs))
                    except OSError:
                        pass  # directory not empty (or already gone) — expected
            else:
                # Atomic write (mkstemp + os.replace) so a crash/SIGKILL/disk-full
                # mid-restore never leaves the user's file truncated/partial.
                # Mirrors the proven pattern in subagent_ipc._atomic_write and
                # common/atomic_io.atomic_write_json; the temp file lives in the
                # same directory so os.replace stays on one filesystem (atomic).
                _fd, _tmp = tempfile.mkstemp(dir=_dir, prefix=".asi-revert-")
                try:
                    with os.fdopen(_fd, "wb") as _f:
                        _f.write(_data)  # type: ignore[arg-type]
                    os.replace(_tmp, _abs)
                except BaseException:
                    try:
                        os.unlink(_tmp)
                    except OSError:
                        pass
                    raise
            reverted.append(_fp)
        except Exception as _e:
            logger.warning("Could not restore %s before retry: %s", _fp, _e)
    return reverted


# ── Scope-violation signal hygiene (B5 / findings 2+3) ──────────────────────
# Infrastructure paths that must NEVER count as a worker's scope violation: they
# are the orchestration framework's OWN artifacts (IPC dirs, run logs), not user
# edits. The canonical repo gitignores ``.asicode/`` and ``logs/``, but a repo
# WITHOUT a matching .gitignore (scratch repos, freshly-cloned foreign repos)
# would otherwise flood ``unassigned_changes`` with framework noise on every run.
_UNASSIGNED_INFRA_PREFIXES: tuple[str, ...] = (
    ".asicode/", "logs/", ".git/",
)


def _is_infra_path(rel: str) -> bool:
    """True for framework-artifact paths that are never a real scope violation."""
    if not rel:
        return True
    _norm = rel.replace("\\", "/")
    return any(_norm.startswith(_p) for _p in _UNASSIGNED_INFRA_PREFIXES)


def _snapshot_dirty_path_set(repo_root: str) -> set:
    """Return the set of currently-dirty rel-paths via one ``git status`` pass.

    Reuses :func:`partition_changed_files` (empty assignment ⇒ every changed file
    is "in scope") so the parsing logic — NUL-delimited porcelain, rename handling
    — stays in one place. Any failure yields an empty set (filter becomes a no-op,
    the worker's raw report passes through unchanged).
    """
    try:
        from .subagent_ipc import partition_changed_files
        _in_scope, _ = partition_changed_files(repo_root, [])
        return {os.path.normpath(_p["file"]) for _p in _in_scope if _p.get("file")}
    except Exception:
        return set()


def _symbol_hint_for_source(src: str, fpath: str) -> list[str]:
    """Build human-readable ``[Symbol map]`` lines for a source file.

    Uses the multi-language tree-sitter extractor (``find_all_symbols``), the
    single source of truth for symbol extraction — so TS/JS/Go/Java/… assigned
    files get real symbol names instead of being silently skipped. Falls back
    to the stdlib ``ast`` module for Python files when no tree-sitter grammar
    is installed, preserving the previous Python-only behaviour. Returns ``[]``
    for unsupported/undetectable languages.
    """
    try:
        from ..languages.models import LanguageId
        from ..languages.tree_sitter_utils import find_all_symbols
        lang_id = LanguageId.from_path(fpath)
        if lang_id is not LanguageId.UNKNOWN:
            syms = find_all_symbols(src, lang_id.value)
            if syms:
                return [f"{_kind_initial(k)} {n} L{s}-L{e}" for (n, k, s, e) in syms]
    except Exception:
        pass
    # Fallback: stdlib ast for Python files (no grammar dependency).
    try:
        tree = ast.parse(src)
        out: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                end = getattr(node, "end_lineno", node.lineno)
                out.append(f"{node.__class__.__name__[0]} {node.name} L{node.lineno}-L{end}")
        return out
    except Exception:
        return []
def _symbol_def_line(src: str, fpath: str, symbol: str) -> Optional[int]:
    """Return the 1-indexed definition line of *symbol* in *src*, or ``None``.

    Multi-language via the tree-sitter ``find_all_symbols`` single source of
    truth (the same extractor ``_symbol_hint_for_source`` uses), with a stdlib
    ``ast`` fallback for Python. Unlike a ``^\\s*(class|def)`` regex this finds
    TS/JS/Go/Java/… symbols and does not false-match inside strings/comments.
    Used by ``OrchestratorAgent._locate_symbol`` to build the review-retry hint.
    """
    try:
        from ..languages.models import LanguageId
        from ..languages.tree_sitter_utils import find_all_symbols
        lang_id = LanguageId.from_path(fpath)
        if lang_id is not LanguageId.UNKNOWN:
            for (_n, _k, s, _e) in find_all_symbols(src, lang_id.value):
                if _n == symbol:
                    return s
    except Exception:
        pass
    # Fallback: stdlib ast for Python files (no grammar dependency).
    try:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name == symbol:
                return node.lineno
    except Exception:
        pass
    return None
_DECOMPOSE_SYSTEM = """\
You are a software engineering task orchestrator. Your job is to decompose a
complex coding task into subtasks that can be executed by separate coding agents.

Rules:
- Each subtask must be self-contained and as independent as possible.
- Minimise file overlap between subtasks (each file should be owned by <= 1 subtask).
- 2-4 subtasks is ideal; never more than {max_subagents}.
- Descriptions must be detailed enough for an agent to work without further context.
- If subtasks have dependencies (e.g., bug fix must be applied before writing unit tests),
  specify dependencies using the 'dependencies' field listing task_ids that must complete
  before this subtask starts. Keep dependencies minimal to allow parallel execution.
- Assign priority to each subtask: 0 = highest priority (run first), 1 = normal, 2 = lowest.
  Subtasks at the same priority level with no mutual dependencies run in parallel.
  Critical path tasks (e.g., core implementation before tests) should have priority 0.

**System Anomaly Detection:** Flag any abnormality (missing info, contradictory
instructions, impossible requirements, system bugs) in your analysis. Do not silently
produce incorrect output.

Return ONLY a valid JSON object -- no prose, no markdown fences:
{{
  "analysis": "one-sentence summary of the overall task",
  "subtasks": [
    {{
      "id": "dev_1",
      "title": "Short action title",
      "description": "Full task description for this subtask",
      "assigned_files": ["path/to/file.py"],
      "dependencies": [],
      "priority": 0
    }}
  ]
}}"""

_SYNTHESISE_SYSTEM = """\
You are a senior software engineer summarising the results of a parallel
multi-agent coding session. Given the original request and each sub-agent's
outcome, write a concise summary of:
1. What was accomplished overall
2. Which subtasks succeeded / failed
3. Any important notes or caveats

Keep the summary under 300 words. Respond in the same language as the user's request.

**System Anomaly Detection:** Flag any abnormality (contradictory statuses, missing
outputs, system bugs) in your summary. Do not silently produce misleading conclusions."""


class _OrchestratorBackedRegistry:
    """Registry facade that lets a ``DesignChatLoop`` run AS the orchestrator.

    Wraps a real ``ToolRegistry`` (the orchestrator's prototype registry) so the
    orchestrator gains the **full** design-chat tool set (read + write + edit +
    analysis + web …) PLUS the three orchestrator-native tools
    (``spawn_subagent`` / ``poll_subagent`` / ``list_subagents``).

    Every attribute access (``config``, ``repo_root``, ``_WRITE_TOOLS``,
    ``_SERIAL_TOOLS``, ``_safety_manager``, ``repo_language``, ``session_plan``,
    ``normalize_args_for_display``, …) transparently delegates to the wrapped
    registry — only :meth:`get_tool_schemas` and :meth:`dispatch` are overridden:

    * ``get_tool_schemas`` → base schemas + the 3 native orchestrator schemas.
    * ``dispatch`` → routes the 3 native tools to the orchestrator's handlers;
      everything else falls through to the base registry unchanged.

    The native tools are intentionally NOT added to ``_WRITE_TOOLS`` /
    ``_SERIAL_TOOLS`` (they are non-blocking / pure-read), so DesignChatLoop's
    parallel-read phase handles them like any other read tool.
    """

    def __init__(self, base_registry: Any, orchestrator: "OrchestratorAgent") -> None:
        # Store via object.__setattr__ to avoid tripping our own __setattr__
        # override (which delegates everything to the base registry).
        object.__setattr__(self, "_obr_base", base_registry)
        object.__setattr__(self, "_obr_orch", orchestrator)

    # ── delegation ───────────────────────────────────────────────────────
    def __getattr__(self, name: str) -> Any:
        # Called only when normal attribute lookup fails — forward everything
        # (config, repo_root, _WRITE_TOOLS, _safety_manager, session_plan, …)
        # to the wrapped registry.
        return getattr(self._obr_base, name)

    def __setattr__(self, name: str, value: Any) -> None:
        # session_plan and similar per-run state set by DesignChatLoop must land
        # on the base registry so downstream readers (plan gate) see them.
        if name in ("_obr_base", "_obr_orch"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._obr_base, name, value)

    # ── overridden interface ─────────────────────────────────────────────
    def get_tool_schemas(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        base = list(self._obr_base.get_tool_schemas(*args, **kwargs))
        native = self._obr_orch._native_orchestrator_schemas()
        # Native tools first so the orchestrator's delegation tools are prominent;
        # de-dup by name in case the base already exposes one (it should not).
        _seen = {s["name"] for s in native}
        return native + [s for s in base if s.get("name") not in _seen]

    def dispatch(self, tool_name: str, args: dict[str, Any]) -> Any:
        if tool_name in ("spawn_subagent", "poll_subagent", "list_subagents"):
            from .tool_registry import ToolResult
            try:
                _content = self._obr_orch._dispatch_native_tool(tool_name, args)
                return ToolResult(ok=True, content=_content, execution_time=0.0)
            except _NativeToolError as exc:
                # Handler-level error (bad args, unknown id, ...): report ok=False
                # WITHOUT a traceback — the LLM still sees the message via
                # ``tr.content or tr.error`` rendering, but the ok/fail flag is
                # now correct for downstream success/failure aggregation.
                return ToolResult(ok=False, content="", error=str(exc), execution_time=0.0)
            except Exception as exc:
                logger.exception("Orchestrator-backed dispatch of %s failed", tool_name)
                return ToolResult(ok=False, content="", error=str(exc), execution_time=0.0)
        return self._obr_base.dispatch(tool_name, args)
class OrchestratorAgent:
    """
    Scheduler: executes subtasks across multiple DeveloperAgents (SubAgents).

    Primary entry point (tool-loop mode, used by CLI /orchestrate):
      run(request: str) -> OrchestratorResult   # tool_loop_enabled=True
      The orchestrator drives an LLM-native tool loop that spawns sub-agents.

    Legacy entry point (decomposition mode — backward compat):
      run(request: str) -> OrchestratorResult   # tool_loop_enabled=False
      Uses internal _decompose_task() (LLM call) and _synthesize() (LLM call).

    Execution model:
      - Tasks are grouped by effective priority (considering dependencies).
      - Within each group, independent tasks run in parallel (ThreadPoolExecutor).
      - When a predecessor hits max_turns, its partial results are forwarded
        as context to dependent tasks so work continues without hard failure.
      - A per-subagent review loop (LLM quality gate) can be enabled via
        OrchestratorConfig.review_enabled.
    """

    def __init__(
        self,
        llm_client: Any,
        registry: Any,         # ToolRegistry — cloned per sub-agent
        orch_config: OrchestratorConfig,
        model: str = "",
        callback: Optional[Callable[[str, dict[str, Any]], None]] = None,
        run_store: Optional[Any] = None,  # InMemoryRunStore — shared across sub-agents
        design_stream_callback: Optional[Callable[[str, dict[str, Any]], None]] = None,
    ):
        self.llm_client = llm_client
        self._registry_proto = registry  # used only for repo_root / config access
        self.orch_config = orch_config
        self.model = model
        self._cb_fn = callback
        self._run_store = run_store
        # The DesignChatLoop the orchestrator drives in tool-loop mode emits
        # design_* stream events (design_tool_call / design_thinking / …).  This
        # callback renders them — typically the SAME _ProgressPrinter the
        # interactive REPL uses, so orchestrator tool calls show up as ordinary
        # ✓ lines indistinguishable from a normal design-chat turn.
        self._design_cb = design_stream_callback
        # The original user request for the current tool-loop run; native tool
        # handlers (spawn_subagent) need it to seed sub-agent context.
        self._current_request: str = ""
        self._file_lock_mgr = FileLockManager(repo_root=getattr(registry, 'repo_root', None))
        self._event_dispatcher = OrderedEventDispatcher(self._cb)
        self.subagent_provider = orch_config.subagent_provider or ""
        self.subagent_model = orch_config.subagent_model or ""
        self.subagent_api_key = orch_config.subagent_api_key or ""
        self.subagent_models = dict(orch_config.subagent_models or {})
        self.subagent_ollama_url = orch_config.subagent_ollama_url or ""
        self.review_enabled = orch_config.review_enabled
        self.review_max_retries = orch_config.review_max_retries
        self.auto_launch_terminal = orch_config.auto_launch_terminal
        self.auto_spawn_worker = orch_config.auto_spawn_worker
        # IPC reusable worker POOL: when a sub-agent Terminal stays alive after
        # its task (polling for the next one), the next subtask reuses it instead
        # of opening a new window. Works in BOTH sequential and parallel modes
        # (P3): after a parallel batch completes its workers go idle, so the NEXT
        # batch reuses them rather than paying the ~5-8s spawn cost per worker.
        # A worker is claimed (removed) on dispatch and returned after its task
        # completes; the liveness check drops dead workers. Thread-safe via
        # _bg_lock (parallel batches share the pool via ThreadPoolExecutor).
        self._reusable_worker_ids: set[str] = set()
        # IPC command registry: agent_id → shell command string (for display /
        # auto-launch).  Populated by _run_subagent_ipc.
        self._subagent_ipc_commands: dict[str, str] = {}
        # IPC worker poll directories actually launched this run.  A worker polls
        # ``_subagent_dir(repo_root, <worker_id>)`` (its --subagent-id); in reuse
        # mode that differs from the task_id, so we track worker_ids explicitly to
        # know where to write cancel/shutdown sentinels that the worker will see.
        self._ipc_worker_ids: set[str] = set()
        # Headless background worker processes (cross-platform spawn): worker_id →
        # Popen handle.  The macOS Terminal.app path has no PID handle, but our
        # background spawns do — tracked so _cleanup_ipc_workers can terminate a
        # worker that missed the shutdown.json sentinel (orphan prevention).
        self._ipc_worker_procs: dict[str, Any] = {}
        # Shared memory: compact summaries of completed SubAgents, accumulated
        # across batches and injected into all subsequent SubAgents' context.
        self._shared_memory: list[str] = []
        # ── Tool-loop mode state ──────────────────────────────────────────
        # Background sub-agent bookkeeping for tool-loop mode. spawn_subagent
        # submits into _bg_executor and records a handle here; poll_subagent /
        # list_subagents read it back. The executor is created lazily on first
        # spawn so a non-tool-loop Orchestrator pays no thread-pool cost.
        self._bg_subagents: dict[str, dict[str, Any]] = {}
        self._bg_lock = threading.Lock()
        self._bg_executor: Optional[ThreadPoolExecutor] = None
        # Monotonic counter for auto-generated agent_ids (sub_1, sub_2, …).
        self._bg_counter = 0
        # Ordered results accumulated as background sub-agents complete, so the
        # final OrchestratorResult.subtask_results mirrors completion order.
        self._bg_results: list[Any] = []
        # B5 scope-violation signal hygiene (findings 2+3): the worker's
        # ``unassigned_changes`` are derived from a GLOBAL ``git status``, so
        # without correction they include (a) pre-existing dirt already present
        # before the run (.env, etc.) and (b) in PARALLEL mode the in-flight edits
        # of peer workers to their OWN assigned files. ``_baseline_dirty_paths``
        # (captured once at run start, before any sub-agent edits) lets the
        # completion path subtract the pre-run dirt; ``_global_assigned_paths``
        # (the union of every subtask's assignment) lets it subtract peer writes.
        # What remains is the worker's genuine NEW out-of-scope writes — the real
        # scope-violation signal B5 was meant to surface.
        self._baseline_dirty_paths: set = set()
        self._global_assigned_paths: set = set()

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self, request: str, session_id: str = "") -> OrchestratorResult:
        logger.info("OrchestratorAgent.run() — %s",
                    "tool-loop mode" if self.orch_config.tool_loop_enabled else "decomposition mode")

        # The shared design-chat session id — enables context inheritance in
        # tool-loop mode (the orchestrator reads prior turns / insights via the
        # forwarded session_mgr).  Set here so _run_tool_loop can pick it up;
        # empty string falls back to a context-less bare request (prior behavior).
        self._session_id = session_id or ""

        # Clean up any leftover locks and shared memory from previous sessions
        self._file_lock_mgr.reset()
        self._shared_memory = []
        self._bg_subagents.clear()
        self._bg_results = []
        self._ipc_worker_ids.clear()
        self._ipc_worker_procs.clear()
        # Invalidate the reusable-worker pool: _cleanup_ipc_workers() of the
        # PREVIOUS run already terminated those workers, so the ids are stale
        # references. Keeping them would make the next subtask enter the reuse
        # branch and — since _ipc_worker_procs is also cleared here — bypass the
        # Popen liveness guard, reusing a DEAD worker.
        self._reusable_worker_ids.clear()
        # Reset the B5 scope-violation baseline so a stale baseline from a prior
        # run never corrupts the current run's unassigned-changes filtering.
        self._baseline_dirty_paths = set()
        self._global_assigned_paths = set()

        # Best-effort GC of stale .asicode/subagents/ artifacts from prior runs
        _rr = getattr(self._registry_proto, "repo_root", None) or "."
        if isinstance(_rr, str):
            self._gc_subagent_artifacts(_rr)

        # ── Tool-loop mode: orchestrator drives an LLM-native tool loop ──
        if self.orch_config.tool_loop_enabled:
            return self._run_tool_loop(request)

        try:
            # Check for cancellation before starting
            if self.orch_config.cancel_event and self.orch_config.cancel_event.is_set():
                logger.info("OrchestratorAgent.run() cancelled before decomposition")
                return OrchestratorResult(
                    status="cancelled",
                    summary="Orchestration was cancelled.",
                )

            subtasks = self._decompose_task(request)
            if not subtasks:
                return OrchestratorResult(
                    status="error",
                    summary="Task decomposition failed. Please retry with a single agent.",
                )

            self._cb("orchestrator_plan", {
                "subtasks": [
                    {
                        "id": st.task_id,
                        "title": st.title,
                        "assigned_files": st.assigned_files,
                        "dependencies": st.dependencies,
                        "priority": st.priority,
                    }
                    for st in subtasks
                ],
            })

            # Capture the scope-violation baseline (B5 / findings 2+3) BEFORE any
            # sub-agent runs: the pre-run dirty set + the union of all assignments.
            # ``_filter_unassigned_changes`` subtracts both from each worker's raw
            # global-``git status`` report so only genuine NEW out-of-scope writes
            # survive as the scope-violation signal.
            _rr = getattr(self._registry_proto, "repo_root", None) or "."
            self._capture_scope_baseline(_rr, subtasks)

            # Always use priority + dependency aware execution
            # Forward the original goal so each sub-agent sees the WHOLE task (not
            # just its own subtask description) — the full plumbing
            # (_run_dependency_aware → _run_parallel_batch → _run_subagent → IPC
            # SubagentTask.original_request / in-process [Original request goal]
            # block) already exists, but the lone call site omitted it, leaving
            # every decomposition-mode sub-agent context-blind.
            results = self._run_dependency_aware(subtasks, original_request=request)

            total_turns = sum(len(r.turns) for r in results if r is not None)
            summary = self._synthesize(request, subtasks, results)

            all_ok = all(r.status in ("success", "already_satisfied", "max_turns") for r in results if r)
            any_ok = any(r.status in ("success", "already_satisfied", "max_turns") for r in results if r)
            status = "success" if all_ok else ("partial" if any_ok else "error")

            self._cb("orchestrator_done", {
                "status": status,
                "summary": summary,
                "total_turns": total_turns,
            })

            return OrchestratorResult(
                status=status,
                summary=summary,
                subtask_results=results,
                total_turns=total_turns,
                metadata={
                    "subtasks": len(subtasks),
                    "parallel": self.orch_config.parallel,
                },
            )
        finally:
            # Ensure all locks are released when the orchestration ends
            self._file_lock_mgr.reset()
            # Clean up any IPC workers (shutdown sentinel → worker poll loop exits)
            self._cleanup_ipc_workers()

    def continue_subagent(
        self,
        agent_id: str,
        request_text: str,
        prior_context: str = "",
        extra_turns: int = 5,
    ) -> OrchestratorResult:
        """
        Continue an existing subagent with additional turns.
        Used when a subagent hits max_turns and the user clicks "Continue".
        """
        logger.info("OrchestratorAgent.continue_subagent(%s)", agent_id)
        # Clean up any leftover locks from previous sessions
        self._file_lock_mgr.reset()

        try:
            # Check for cancellation before starting
            if self.orch_config.cancel_event and self.orch_config.cancel_event.is_set():
                logger.info("OrchestratorAgent.continue_subagent() cancelled")
                return OrchestratorResult(
                    status="cancelled",
                    summary="Sub-agent continuation was cancelled.",
                )

            # Create a dummy subtask spec for the continued agent
            combined_description = request_text
            if prior_context:
                combined_description = f"{prior_context}\n\n[CONTINUE FROM PREVIOUS SESSION]\n\n{request_text}"
            subtask = SubTaskSpec(
                task_id=agent_id,
                title="Continued subagent",
                description=combined_description,
                assigned_files=[],
                dependencies=[],
                priority=0,
            )
            # Run subagent using the same logic as _run_subagent
            result = self._run_subagent(subtask, extra_turns=extra_turns)
            # Wrap single AgentResult in OrchestratorResult
            return OrchestratorResult(
                status=result.status,
                summary=result.final_message or "Sub-agent continuation completed",
                subtask_results=[result],
                total_turns=len(result.turns),
                metadata={
                    "continued": True,
                    "agent_id": agent_id,
                },
            )
        finally:
            self._file_lock_mgr.reset()

    # ── Task decomposition (legacy — used by run() when tool_loop_enabled=False) ──

    def _decompose_task(self, request: str) -> list[SubTaskSpec]:
        from ..client import LLMMessage

        system = _DECOMPOSE_SYSTEM.format(max_subagents=self.orch_config.max_subagents)

        # Inject recent git history so the planner knows what was recently changed
        # and avoids re-introducing just-fixed bugs or conflicting with recent work.
        repo_root = getattr(self._registry_proto, 'repo_root', None)
        git_context = _build_git_context(repo_root)

        user_content = f"Task: {request}"
        if git_context:
            user_content = f"{user_content}\n\n{git_context}"

        messages = [
            LLMMessage(role="system", content=system),
            LLMMessage(role="user", content=user_content),
        ]
        # thinking_mode=False: decomposition is structured-JSON generation, not
        # open-ended reasoning.  GLM-5.2+ thinks by default (30s+ of wasted
        # reasoning tokens for a JSON list); turning it off makes this ~5s.
        raw = simple_llm_call(self.llm_client, self.model, messages, thinking_mode=False)
        obj = parse_json(raw)
        if not obj or "subtasks" not in obj:
            logger.warning("OrchestratorAgent: could not parse subtask JSON")
            return []

        # Normalize task_ids to ``dev_<1-based-index>`` so the per-slot model
        # override (``/model dev_N``) maps deterministically to spawn order,
        # regardless of what ids the planner emitted.  Dependency references
        # (which point at the planner's ids) are remapped to match.
        specs: list[SubTaskSpec] = []
        _id_remap: dict[str, str] = {}
        for i, item in enumerate(obj["subtasks"][: self.orch_config.max_subagents]):
            _planner_id = item.get("id") or f"sub_{i+1}"
            _new_id = f"dev_{i+1}"
            _id_remap[_planner_id] = _new_id
            specs.append(SubTaskSpec(
                task_id=_new_id,
                title=str(item.get("title") or f"Subtask {i+1}"),
                description=str(item.get("description") or ""),
                assigned_files=_expand_directory_assignments(
                    repo_root, _coerce_str_list(item.get("assigned_files")),
                ),
                dependencies=_coerce_str_list(item.get("dependencies")),
                priority=_coerce_priority(item.get("priority")),
            ))
        # Drop dependencies that reference truncated (unmapped) task ids.
        # An orphan planner id is absent from task_map, which would make the
        # ``dep not in remaining`` readiness check always True — silently
        # breaking execution ordering (_run_dependency_aware).
        for st in specs:
            st.dependencies = [_id_remap[d] for d in st.dependencies if d in _id_remap]
        # Warn about unscoped tasks: an empty assigned_files disables snapshot
        # capture, scope-violation detection, and review, and
        # _split_batch_by_file_conflict then isolates each in its own sub-batch
        # (serializing it). This usually means the planner LLM omitted the field
        # by mistake — surface it so the decomposition can be improved.
        _unscoped = [st.task_id for st in specs if not st.assigned_files]
        if _unscoped:
            logger.warning(
                "OrchestratorAgent: %d subtask(s) have empty assigned_files "
                "(%s) — no pre-run snapshot / scope-detection / review can "
                "apply, and each will run serialized; the planner likely "
                "omitted the field by mistake.",
                len(_unscoped), _unscoped,
            )
        return specs

    # ── Priority + dependency aware execution ─────────────────────────────

    def _run_dependency_aware(self, subtasks: list[SubTaskSpec], original_request: str = "") -> list[Any]:
        """
        Execute subtasks respecting both dependencies and priorities.
        """
        task_map = {st.task_id: st for st in subtasks}

        # Early cycle detection
        _topological_order, cycles = self._detect_cycles_kahn(task_map)

        if cycles:
            # Report cycles via SSE event
            self._cb("orchestrator_warning", {
                "type": "dependency_cycle",
                "cycles": cycles,
                "message": f"Found {len(cycles)} dependency cycle(s). Attempting to break cycles."
            })
            logger.warning("Dependency cycles detected: %s", cycles)

            # Try to break cycles by removing minimal dependencies.
            # ``_break_cycles`` returns a NEW acyclic list of SubTaskSpec (with
            # the offending edges removed) — re-enter ``_run_dependency_aware``
            # on it so the broken-cycle path gets the SAME capabilities as the
            # normal path: same-priority parallel batching, predecessor-context
            # injection (_build_task_with_predecessor_context), and shared-memory
            # accumulation. (Previously a separate sequential loop silently
            # dropped all three whenever the LLM produced cyclic dependencies.)
            broken_subtasks = self._break_cycles(subtasks, cycles)
            if broken_subtasks is not None:
                return self._run_dependency_aware(broken_subtasks, original_request=original_request)
            else:
                # Fall back to original order with warning
                logger.warning("Could not break cycles, falling back to original order")
                self._cb("orchestrator_warning", {
                    "type": "dependency_cycle_fallback",
                    "message": "Could not automatically break cycles, using original task order."
                })

        idx_map = {st.task_id: i for i, st in enumerate(subtasks)}
        results: list[Any] = [None] * len(subtasks)
        results_map: dict[str, Any] = {}   # task_id -> AgentResult (completed)
        remaining: set = set(task_map.keys())
        cancel_event = self.orch_config.cancel_event


        while remaining:
            # Check cancellation
            if cancel_event and cancel_event.is_set():
                logger.info("OrchestratorAgent._run_dependency_aware() cancelled")
                for tid in remaining:
                    results[idx_map[tid]] = AgentResult(
                        status="cancelled",
                        final_message="Sub-agent was cancelled.",
                        error="Cancelled by orchestrator",
                    )
                break

            # Find ready tasks (all dependencies completed)
            ready = [
                tid for tid in remaining
                if all(dep not in remaining for dep in task_map[tid].dependencies)
            ]

            if not ready:
                # This shouldn't happen if we used topological order, but handle gracefully
                logger.error(
                    "No ready tasks found despite topological ordering. "
                    "Possible bug in cycle detection or dynamic dependency changes. "
                    "Forcing remaining: %s", remaining
                )
                # Try to find and report the actual cycle
                current_cycles = self._find_current_cycles(remaining, task_map)
                if current_cycles:
                    self._cb("orchestrator_error", {
                        "type": "execution_deadlock",
                        "cycles": current_cycles,
                        "message": f"Execution deadlock detected with cycles: {current_cycles}"
                    })

                # Force execution as fallback
                ready = list(remaining)

            # Sort ready tasks by priority (lower number = higher priority first)
            ready.sort(key=lambda tid: task_map[tid].priority)

            # Group by priority — run the highest-priority batch in parallel
            top_priority = task_map[ready[0]].priority
            batch = [tid for tid in ready if task_map[tid].priority == top_priority]

            # File-conflict guard: tasks that share assigned_files must NOT run in
            # parallel — the second one would overwrite the first's changes without
            # seeing them.  Split into sequential conflict-free sub-batches.
            sub_batches = _split_batch_by_file_conflict([task_map[tid] for tid in batch])
            if len(sub_batches) > 1:
                logger.info(
                    "OrchestratorAgent: priority=%d batch=%s split into %d "
                    "sequential sub-batches due to file conflicts",
                    top_priority, batch, len(sub_batches),
                )
                self._cb("orchestrator_warning", {
                    "type": "file_conflict_serialized",
                    "batch": batch,
                    "sub_batch_count": len(sub_batches),
                    "message": (
                        f"{len(sub_batches)} tasks targeting the same file(s) were "
                        "serialized to prevent parallel overwrite."
                    ),
                })
            else:
                logger.info(
                    "OrchestratorAgent: running batch priority=%d tasks=%s",
                    top_priority, batch,
                )

            all_batch_tids: list[str] = []
            all_batch_results: list[Any] = []
            for sub_batch_specs in sub_batches:
                sub_batch_tids = [s.task_id for s in sub_batch_specs]
                sub_results = self._run_parallel_batch(
                    sub_batch_specs, results_map, task_map, original_request=original_request,
                )
                all_batch_tids.extend(sub_batch_tids)
                all_batch_results.extend(sub_results)

            # strict=True: all_batch_tids/results are built together (extend per
            # sub-batch), so they are provably length-aligned — fail fast if a
            # future change to _run_parallel_batch breaks that invariant instead
            # of silently dropping results.
            for tid, result in zip(all_batch_tids, all_batch_results, strict=True):
                results[idx_map[tid]] = result
                results_map[tid] = result
                remaining.discard(tid)
                # Accumulate compact summary into shared memory for subsequent batches
                if result is not None:
                    summary = self._extract_subagent_summary(task_map[tid], result)
                    if summary:
                        self._shared_memory.append(summary)

        return results

    def _run_parallel_batch(
        self,
        batch: list[SubTaskSpec],
        completed_results: dict[str, Any],
        task_map: dict[str, SubTaskSpec],
        original_request: str = "",
    ) -> list[Any]:
        """Run a batch of independent subtasks in parallel."""
        if not batch:
            return []

        # emit batch start event
        batch_id = str(uuid.uuid4())[:8]
        self._event_dispatcher.emit("orchestrator", "batch_start", {
            "batch_id": batch_id,
            "task_count": len(batch),
            "task_ids": [st.task_id for st in batch],
            "priority": batch[0].priority if batch else 0,
        })

        if len(batch) == 1:
            # Single task — no need for thread pool
            st = batch[0]
            task_text = self._build_task_with_predecessor_context(
                st, completed_results, task_map
            )
            result = [self._run_subagent(st, task_text=task_text, original_request=original_request)]

            # emit batch complete event
            self._event_dispatcher.emit("orchestrator", "batch_complete", {
                "batch_id": batch_id,
                "success_count": sum(1 for r in result if r.status in ("success", "already_satisfied")),
                "total_count": len(result),
            })
            return result

        cancel_event = self.orch_config.cancel_event
        results: list[Any] = [None] * len(batch)
        max_workers = min(len(batch), self.orch_config.max_subagents)

        pool = ThreadPoolExecutor(max_workers=max_workers)
        cancelled_batch = False
        try:
            futures: dict[concurrent.futures.Future, int] = {}
            for idx, st in enumerate(batch):
                task_text = self._build_task_with_predecessor_context(
                    st, completed_results, task_map
                )
                futures[pool.submit(self._run_subagent, st, 0, task_text, original_request)] = idx

            pending = set(futures.keys())
            while pending:
                if cancel_event and cancel_event.is_set():
                    for f in pending:
                        f.cancel()
                    done, still_pending = concurrent.futures.wait(
                        pending, timeout=2.0, return_when=concurrent.futures.ALL_COMPLETED
                    )
                    # Drain `done` BEFORE synthesising cancelled results. A cancelled
                    # future is reported in `done` too (wait() treats a cancelled
                    # future as completed), and so is any sub-agent that observed
                    # cancel_event at a turn boundary and self-terminated within the
                    # 2s window. Without this drain those futures' real results
                    # (status / final_message / partial diff) are dropped and their
                    # slots stay None — silently losing completed work on the very
                    # cancellation path that is supposed to preserve it. Mirrors the
                    # normal FIRST_COMPLETED drain below.
                    for future in done:
                        idx = futures[future]
                        try:
                            results[idx] = future.result()
                        except concurrent.futures.CancelledError:
                            results[idx] = AgentResult(
                                status="cancelled",
                                final_message="Sub-agent was cancelled.",
                                error="Cancelled",
                            )
                        except Exception as exc:
                            logger.exception(
                                "SubAgent %s raised an exception", batch[idx].task_id
                            )
                            results[idx] = AgentResult(
                                status="error",
                                error=str(exc),
                                final_message=f"SubAgent failed: {exc}",
                            )
                    for f in still_pending:
                        results[futures[f]] = AgentResult(
                            status="cancelled",
                            final_message="Sub-agent was cancelled.",
                            error="Cancelled by orchestrator",
                        )
                    cancelled_batch = True
                    pending = set()
                    break

                done, pending = concurrent.futures.wait(
                    pending, timeout=0.5, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    idx = futures[future]
                    try:
                        results[idx] = future.result()
                    except concurrent.futures.CancelledError:
                        results[idx] = AgentResult(
                            status="cancelled",
                            final_message="Sub-agent was cancelled.",
                            error="Cancelled",
                        )
                    except Exception as exc:
                        logger.exception(
                            "SubAgent %s raised an exception", batch[idx].task_id
                        )
                        results[idx] = AgentResult(
                            status="error",
                            error=str(exc),
                            final_message=f"SubAgent failed: {exc}",
                        )
        finally:
            # On cancellation, do NOT block on in-flight sub-agents here: they
            # observe cancel_event at turn boundaries (agent_loop) and
            # self-terminate. cancel_futures drops not-yet-started tasks; the
            # non-blocking shutdown lets the orchestrator bail fast, consistent
            # with the cancel-aware drain path. On normal completion every
            # future is already done, so wait=True is effectively a no-op.
            if cancelled_batch:
                pool.shutdown(wait=False, cancel_futures=True)
            else:
                pool.shutdown(wait=True)

        # emit batch complete event
        success_count = sum(1 for r in results if r is not None and r.status in ("success", "already_satisfied"))
        self._event_dispatcher.emit("orchestrator", "batch_complete", {
            "batch_id": batch_id,
            "success_count": success_count,
            "total_count": len(results),
        })
        return results

    def _build_task_with_predecessor_context(
        self,
        subtask: SubTaskSpec,
        completed_results: dict[str, Any],
        task_map: dict[str, SubTaskSpec],
    ) -> str:
        """Build task description injecting context from completed predecessor tasks.

        When a predecessor hit max_turns (partial completion), its final_message
        and applied patches are forwarded so the dependent agent can continue
        from where the predecessor left off rather than starting from scratch.
        """
        task_text = subtask.description
        if subtask.assigned_files:
            files_hint = ", ".join(subtask.assigned_files)
            task_text = f"{task_text}\n\n[Assigned files: {files_hint}]"

        predecessor_parts: list[str] = []
        for dep_id in subtask.dependencies:
            dep_result = completed_results.get(dep_id)
            if dep_result is None:
                continue
            dep_task = task_map.get(dep_id)
            if dep_task is None:
                dep_task = SubTaskSpec(task_id=dep_id, title=dep_id, description="")
            one_liner = self._extract_subagent_summary(dep_task, dep_result)
            predecessor_parts.append(f"[Predecessor] {one_liner}")

        if predecessor_parts:
            context_block = "\n".join(predecessor_parts)
            task_text = (
                f"[Predecessor task status]\n{context_block}\n\n"
                f"[Current task]\n{task_text}"
            )

        # Inject shared memory: summaries of ALL previously completed SubAgents
        # (not just direct dependencies). Excludes tasks already covered above.
        dep_ids = set(subtask.dependencies)
        shared = [
            s for s in self._shared_memory
            if not any(s.startswith(f"[{dep_id}:") for dep_id in dep_ids)
        ]
        if shared:
            shared_block = "\n".join(shared)
            task_text = (
                f"[Orchestration progress]\n{shared_block}\n\n"
                + task_text
            )

        return task_text

    def _extract_subagent_summary(self, subtask: SubTaskSpec, result: Any) -> str:
        """Extract a compact one-line summary from a completed SubAgent result.

        This summary is stored in _shared_memory and injected into all subsequent
        SubAgents' context so they know what has already been done and which files
        were modified — without sharing the full conversation history.
        """
        status_en = {
            "success": "completed",
            "max_turns": "partial",
            "error": "failed",
            "cancelled": "cancelled",
        }.get(result.status, result.status)

        # Collect modified files from applied patches
        patches = getattr(result, "applied_patches", []) or []
        if not isinstance(patches, (list, tuple)):
            patches = []
        files: list[str] = []
        for p in patches[:5]:
            if isinstance(p, dict):
                f = p.get("file") or p.get("file_path", "")
            elif hasattr(p, "file_path"):
                f = getattr(p, "file_path", "")
            else:
                f = ""
            if f and f not in files:
                files.append(f)

        files_note = f" | Files: {', '.join(files)}" if files else ""
        msg = (result.final_message or "")[:200].replace("\n", " ")
        return f"[{subtask.task_id}: {subtask.title} → {status_en}{files_note}] {msg}"

    # ── Single subagent runner ─────────────────────────────────────────────
    def _resolve_subagent_model(self, agent_id: str) -> tuple[str, str, str]:
        """Resolve ``(provider, model, api_key)`` for a sub-agent by its slot.

        Priority:
          1. Explicit per-slot config (``/model dev_N``) for this agent's slot.
          2. Lowest-numbered configured dev slot (so ``dev_1`` acts as the
             default for unconfigured slots — matches the user contract:
             "if only dev_1 is set, the rest fall back to dev_1").
          3. The orchestrator's own model (``subagent_provider``/``_model``/
             ``_api_key``) — i.e. the model selected via the global ``/model``.

        The slot is the trailing digits of ``agent_id`` (``dev_1``/``sub_1``
        → slot ``"1"``), so both the new ``dev_N`` IDs and legacy ``sub_N``
        IDs resolve identically.
        """
        models = self.subagent_models
        _m = re.search(r'(\d+)$', agent_id)
        _slot = _m.group(1) if _m else ""
        if _slot and _slot in models:
            return tuple(models[_slot])  # type: ignore[return-value]
        if models:
            _nums = [k for k in models if str(k).isdigit()]
            if _nums:
                return tuple(models[min(_nums, key=int)])  # type: ignore[return-value]
        return self.subagent_provider, self.subagent_model, self.subagent_api_key

    @staticmethod
    def _gc_subagent_artifacts(repo_root: str, max_age_days: int = 7) -> None:
        """Best-effort GC of stale subagent directories (worker logs, quarantined
        results, expired heartbeats, ``.bad`` task files). Runs at orchestrator
        start to prevent unbounded accumulation across runs. Non-critical — never
        raises.
        """
        _base = os.path.join(repo_root, ".asicode", "subagents")
        if not os.path.isdir(_base):
            return
        _deadline = time.time() - max_age_days * 86400
        try:
            for _entry in os.listdir(_base):
                _path = os.path.join(_base, _entry)
                if not os.path.isdir(_path):
                    continue
                try:
                    _mtime = os.path.getmtime(_path)
                except OSError:
                    continue
                if _mtime < _deadline:
                    shutil.rmtree(_path, ignore_errors=True)
                    logger.info(
                        "GC: removed stale subagent dir %r (age %.1f days)",
                        _entry, (time.time() - _mtime) / 86400,
                    )
        except Exception:
            logger.warning("GC of subagent artifacts failed (non-critical)", exc_info=True)

    def _run_subagent_ipc(
            self,
            subtask: SubTaskSpec,
            extra_turns: int = 0,
            task_text: Optional[str] = None,
            original_request: str = "",
        ) -> Any:
            """IPC mode: write task.json, wait for an external sub-agent to write result.json.

            The external sub-agent is an ``asi --subagent`` process that polls
            ``.asicode/subagents/<agent_id>/task.json``.  When it finishes it writes
            ``result.json`` which this method reads back and wraps in an AgentResult
            so downstream code (review loop, shared memory, synthesis) works the same
            as the in-process path.
            """
            from .subagent_ipc import SubagentTask, write_task, wait_for_result, clear_result
            from .agent_loop import AgentResult, AgentTurn

            agent_id = subtask.task_id
            logger.info("SubAgent %s (IPC) dispatching: %s", agent_id, subtask.title)

            # Per-subagent model: explicit dev_<N> config → lowest dev slot →
            # orchestrator model (see _resolve_subagent_model).
            _provider, _model, _api_key = self._resolve_subagent_model(agent_id)

            _base_max_turns = (
                self.orch_config.agent_config.max_turns
                if self.orch_config.agent_config else 12
            ) + extra_turns
            ipc_task = SubagentTask(
                task_id=subtask.task_id,
                title=subtask.title,
                description=subtask.description,
                assigned_files=list(subtask.assigned_files),
                original_request=original_request,
                priority=subtask.priority,
                dependencies=list(subtask.dependencies),
                max_turns=_base_max_turns,
                provider=_provider,
                model=_model,
                api_key=_api_key or "",
                predecessor_context=task_text or "",
            )

            repo_root = getattr(self._registry_proto, "repo_root", None) or "."
            # Clear stale result.json so wait_for_result does not read the previous
            # run's result before the sub-agent has started this new task.
            clear_result(repo_root, agent_id)

            # Snapshot assigned files' pre-run bytes so a rejected IPC attempt can
            # be reverted to a clean baseline — mirroring the in-process path.
            # Without this the external sub-agent would retry ON TOP of its own
            # rejected edits (diverging semantics vs in-process, where the retry
            # starts fresh).
            _revert_snapshots = _capture_assigned_snapshots(repo_root, subtask.assigned_files)

            # ── Reusable worker POOL: claim an idle worker from the pool instead
            # of spawning a fresh one. Works in BOTH sequential and parallel modes
            # (P3): after a batch completes its workers go idle, so the NEXT batch
            # reuses them (saving ~5-8s spawn per worker). _claim_reusable_worker
            # atomically removes the worker from the pool under _bg_lock (so two
            # parallel threads never claim the same worker) and verifies the
            # tracked Popen is still alive before returning it.
            _reuse_worker_id = self._claim_reusable_worker(repo_root)

            if _reuse_worker_id is not None:
                _worker_id: str = _reuse_worker_id
                write_task(repo_root, ipc_task, worker_id=_worker_id)
                logger.info(
                    "SubAgent %s (IPC) reusing worker %s — no new terminal",
                    agent_id, _worker_id,
                )
            else:
                _worker_id = agent_id
                write_task(repo_root, ipc_task)

            # Record the worker's actual poll directory so _cleanup_ipc_workers
            # (and the cancel path) can write sentinels where this worker will
            # see them.  In reuse mode worker_id != task_id.
            self._ipc_worker_ids.add(_worker_id)

            # Emit start event.
            self._event_dispatcher.emit(agent_id, "subagent_start", {
                "task_id": subtask.task_id,
                "title": subtask.title,
                "description": subtask.description,
                "assigned_files": list(subtask.assigned_files),
                "dependencies": list(subtask.dependencies),
                "priority": subtask.priority,
                "mode": "ipc",
            })

            # Store the launch command (for the macOS Terminal.app `do script`
            # and the manual-launch hint).  Derived from the SAME argv the
            # background spawn uses (asr_subagent_argv) via shlex.join, so the
            # worker invocation is defined in exactly one place.
            _shell_argv = asr_subagent_argv(repo_root) + [
                "--subagent-id", _worker_id, "--orch-pid", str(os.getpid()),
            ]
            if _provider:
                _shell_argv += ["--provider", _provider]
            if _model:
                _shell_argv += ["--model", _model]
            self._subagent_ipc_commands[agent_id] = (
                f"cd {shlex.quote(repo_root)} && " + shlex.join(_shell_argv)
            )

            # ── Auto-launch the worker — only for NEW workers (a reused worker is
            #    already a live process polling this directory).  Two strategies:
            #      * macOS + auto_launch_terminal → a VISIBLE Terminal.app window
            #        (osascript) so the user can watch the worker run.
            #      * otherwise (any OS) + auto_spawn_worker → a HEADLESS background
            #        subprocess (cross-platform; no console window).  This is what
            #        makes IPC work on Linux/Windows, and the macOS fallback when
            #        the Terminal.app launch fails.
            if _reuse_worker_id is None:
                _launched = False
                if self.auto_launch_terminal and sys.platform == "darwin":
                    _launched = self._launch_ipc_worker_terminal_macos(agent_id, _worker_id)
                if not _launched and (self.auto_spawn_worker or self.auto_launch_terminal):
                    _launched = self._spawn_ipc_worker_background(
                        repo_root, _worker_id, _provider, _model,
                    )
                if not _launched:
                    logger.warning(
                        "Sub-agent %s: no auto-launch path succeeded — the worker "
                        "must be started manually: %s",
                        agent_id, self._subagent_ipc_commands.get(agent_id, ""),
                    )

            # ── Wait for the external sub-agent to finish.
            def _on_poll(elapsed_s: float, _aid: str) -> None:
                # Surface heartbeat progress hints (F3) so the UI shows
                # "turn N, <tool>" rather than only an elapsed counter.
                _hb_turn, _hb_tool = 0, ""
                try:
                    from .subagent_ipc import read_heartbeat_state
                    _st = read_heartbeat_state(repo_root, _aid)
                    if isinstance(_st, dict):
                        _hb_turn = int(_st.get("turn", 0) or 0)
                        _hb_tool = str(_st.get("last_tool", "") or "")
                except Exception:
                    pass
                self._event_dispatcher.emit(_aid, "subagent_waiting_ipc", {
                    "agent_id": _aid,
                    "elapsed_s": elapsed_s,
                    "turn": _hb_turn,
                    "last_tool": _hb_tool,
                })
            # Emit an initial "waiting" event so the UI can show the launch command.
            self._event_dispatcher.emit(agent_id, "subagent_waiting_ipc", {
                "agent_id": agent_id,
                "elapsed_s": 0.0,
            })

            result = wait_for_result(
                repo_root,
                agent_id,
                timeout_s=self.orch_config.ipc_timeout_s,
                startup_timeout_s=self.orch_config.ipc_startup_timeout_s,
                max_timeout_s=self.orch_config.ipc_max_timeout_s,
                heartbeat_stale_s=self.orch_config.ipc_heartbeat_stale_s,
                cancel_event=self.orch_config.cancel_event,
                on_poll=_on_poll,
            )

            if result is None:
                logger.warning("SubAgent %s (IPC) did not return a result", agent_id)
                # If we bailed because the orchestrator was cancelled (Ctrl-C),
                # tell the worker to abort its IN-FLIGHT task: it has no PID
                # handle for an osascript-launched terminal, so a cancel.json
                # sentinel is the only signal that reaches a mid-task worker.
                # Without this the worker keeps burning tokens until its task
                # finishes or max_turns elapses.
                _reusable = self._abandon_ipc_worker(
                    repo_root, _worker_id, _revert_snapshots, task_id=agent_id,
                )
                # Only a worker that soft-quiesced (alive & idle, B6) is reusable.
                # One that failed to quiesce was terminated (tracked Popen) or is
                # hung (Terminal-launched, no PID handle — invisible to the claim's
                # liveness check). Returning a hung slot re-dispatches a dead
                # worker that then burns the full ipc_timeout_s on the next claim.
                # A hard cancel ends the run, so the pool is cleared regardless.
                if _reusable:
                    self._return_worker_to_pool(_worker_id)
                return AgentResult(
                    status="error",
                    final_message=f"Sub-agent '{agent_id}' timed out or was cancelled.",
                    turns=[],
                    error="no result from sub-agent",
                )

            # Reconstruct an AgentResult from the IPC result.
            agent_result = AgentResult(
                status=result.status,
                final_message=result.final_message,
                turns=[AgentTurn(
                    turn_num=i + 1,
                    tool_name="ipc_subagent",
                    tool_args={"agent_id": agent_id},
                    tool_result=result.final_message if i == 0 else "",
                ) for i in range(result.turns)] if result.turns > 0 else [],
                applied_patches=result.applied_patches,
                error=result.error if result.status == "error" else None,
            )
            # Track the latest IPC result's out-of-scope changes (B5) so the
            # completion event can surface a scope violation to the review UI /
            # downstream consumers. AgentResult itself has no such field (it is a
            # shared type); the IPC SubagentResult carries it.
            _ipc_unassigned: list[dict] = list(getattr(result, "unassigned_changes", []) or [])

            # ── Orchestrator review loop (reuses the in-process review helper).
            retry_count = 0
            while (
                self.review_enabled
                and agent_result.status in ("success", "already_satisfied")
                and retry_count < self.review_max_retries
            ):
                approved, feedback = self._review_subagent_result(
                    agent_id=agent_id,
                    subtask=subtask,
                    result=agent_result,
                    repo_root=repo_root,
                )
                if approved:
                    break
                retry_count += 1
                logger.info(
                    "SubAgent %s (IPC) review: not approved (%d/%d). Feedback: %s",
                    agent_id, retry_count, self.review_max_retries,
                    feedback[:200],
                )
                # Restore the pre-run snapshot so the external sub-agent retries
                # from a clean baseline (same semantics as the in-process path),
                # rather than layering a revision on top of its own rejected edits.
                reverted = _restore_assigned_snapshots(repo_root, _revert_snapshots)
                if reverted:
                    logger.info("SubAgent %s (IPC) restored %s before retry", agent_id, reverted)
                # policy=revert: also roll back genuine out-of-scope writes so the
                # retry starts from a clean baseline rather than layering on top of
                # the rejected attempt's strays.
                if self.orch_config.scope_violation_policy == "revert":
                    _retry_genuine = self._detect_genuine_violations(
                        repo_root, subtask.assigned_files,
                        raw_unassigned=_ipc_unassigned,
                    )
                    _retry_rv = self._revert_unassigned_changes(repo_root, _retry_genuine)
                    if _retry_rv:
                        reverted = list(reverted) + _retry_rv
                        logger.info(
                            "SubAgent %s (IPC) reverted %d out-of-scope file(s) before retry: %s",
                            agent_id, len(_retry_rv), _retry_rv,
                        )
                self._event_dispatcher.emit(agent_id, "subagent_retry", {
                    "task_id": subtask.task_id,
                    "retry": retry_count,
                    "max_retries": self.review_max_retries,
                    "feedback": feedback,
                    "reverted_files": reverted,
                    "mode": "ipc",
                })
                ipc_task.description = (
                    f"[REVISION REQUESTED — attempt {retry_count}/{self.review_max_retries}]\n"
                    f"The previous attempt was reverted. Start fresh and apply the correct change.\n"
                    f"Reviewer feedback: {feedback}\n\n"
                    f"Original task: {subtask.description}"
                )
                ipc_task.max_turns = _base_max_turns
                # write_task only mints a fresh epoch nonce when task.epoch is
                # falsy; the retry loop reuses the SAME ipc_task object, so
                # every attempt would otherwise share attempt-1's epoch and
                # the epoch check could not distinguish a stale (attempt-1)
                # result.json from the current attempt's. Sequential retries
                # make an actual mix-up unlikely, but resetting here restores
                # the defense-in-depth the epoch nonce is meant to provide.
                ipc_task.epoch = 0
                clear_result(repo_root, agent_id)
                # Reuse mode: the worker polls ITS OWN directory (_worker_id,
                # which differs from agent_id), not agent_id's dir. The initial
                # dispatch above passed worker_id correctly; this review-retry
                # must too, or the reused worker never sees the revision task and
                # idles until ipc_timeout_s (the original omission — a silent
                # hang on every retry under worker reuse + review). Safe in the
                # non-reuse branch too: there _worker_id == agent_id == task_id,
                # so the destination directory is unchanged.
                write_task(repo_root, ipc_task, worker_id=_worker_id)
                self._event_dispatcher.emit(agent_id, "subagent_waiting_ipc", {
                    "agent_id": agent_id,
                    "elapsed_s": 0.0,
                })
                result2 = wait_for_result(
                    repo_root, agent_id,
                    timeout_s=self.orch_config.ipc_timeout_s,
                    startup_timeout_s=self.orch_config.ipc_startup_timeout_s,
                max_timeout_s=self.orch_config.ipc_max_timeout_s,
                heartbeat_stale_s=self.orch_config.ipc_heartbeat_stale_s,
                    cancel_event=self.orch_config.cancel_event,
                    on_poll=_on_poll,
                )
                if result2 is None:
                    # Cancelled (or timed out) during the review-retry wait.
                    # The tree was already reverted to the pre-run snapshot above
                    # (so it is clean), but ``agent_result`` still holds the FIRST
                    # attempt's success — the review rejected it and we reverted
                    # it, yet leaving it would report a stale "success" that masks
                    # both the failure and the rollback. Mirror the initial-
                    # timeout path (above): abandon the worker and return an
                    # explicit error so the consumer never sees the stale result.
                    _reusable = self._abandon_ipc_worker(
                        repo_root, _worker_id, _revert_snapshots, task_id=agent_id,
                    )
                    if _reusable:
                        self._return_worker_to_pool(_worker_id)
                    return AgentResult(
                        status="error",
                        final_message=f"Sub-agent '{agent_id}' review-retry timed out or was cancelled.",
                        turns=[],
                        error="no result from sub-agent review-retry",
                    )
                agent_result = AgentResult(
                    status=result2.status,
                    final_message=result2.final_message,
                    turns=[AgentTurn(
                        turn_num=i + 1,
                        tool_name="ipc_subagent",
                        tool_args={"agent_id": agent_id},
                        tool_result=result2.final_message if i == 0 else "",
                    ) for i in range(result2.turns)] if result2.turns > 0 else [],
                    applied_patches=result2.applied_patches,
                    error=result2.error if result2.status == "error" else None,
                )
                # Reflect the retry attempt's out-of-scope changes (B5).
                _ipc_unassigned = list(getattr(result2, "unassigned_changes", []) or [])

            # ── Cancelled/error-by-worker revert ────────────────────────────────
            # The orchestrator-level cancel (cancel_event set) makes
            # wait_for_result return None → _abandon_ipc_worker reverts. But the
            # WORKER can ALSO cancel mid-task (Ctrl-C in its terminal, or a
            # late-arriving cancel.json it honors at the next turn boundary) and
            # still write a status="cancelled" result.json. Likewise a task that
            # raises mid-run writes status="error" with whatever it had already
            # written to disk. Either way the review loop above (success-only)
            # never reverts, so the aborted/crashed task's partial edits would
            # linger in the working tree. Revert to the pre-run snapshot so
            # neither leaves half-applied edits behind, mirroring the
            # timeout/abandon path. The result still carries what the worker
            # reported (applied_patches / unassigned_changes) for the completion
            # event below.
            _worker_confirmed_exited = False
            if agent_result.status in ("cancelled", "error"):
                _cancelled_revert = _restore_assigned_snapshots(
                    repo_root, _revert_snapshots,
                )
                if _cancelled_revert:
                    logger.info(
                        "SubAgent %s (IPC) %s; reverted partial edits: %s",
                        agent_id, agent_result.status, _cancelled_revert,
                    )
            if agent_result.status == "cancelled":
                # A "cancelled" result is ambiguous: a pure task-scope cancel.json
                # (B6) leaves the worker alive and reusable, but a Ctrl-C in the
                # worker's own terminal sets BOTH the task-scope cancel_event
                # (this result) AND the process-scope shutdown_event — the worker
                # exits its poll loop right after writing this result. Give it a
                # brief grace window to land its "exited" idle-heartbeat marker
                # before deciding the slot is safe to return to the pool;
                # otherwise a dead worker gets reused and the next dispatch burns
                # the full ipc_timeout_s.
                from .subagent_ipc import read_worker_idle_heartbeat_state
                _grace_deadline = time.time() + 2.0
                while time.time() < _grace_deadline:
                    try:
                        if read_worker_idle_heartbeat_state(repo_root, _worker_id) == "exited":
                            _worker_confirmed_exited = True
                            break
                    except Exception:
                        break
                    time.sleep(0.2)

            # Diff cross-verification (shared with the in-process path).
            # IPC is the DEFAULT mode on macOS (the terminal auto-launch
            # target), so without this the parallel poll loop would never see a
            # verdict here.  IPC applied_patches are clean {"file": ...} dicts,
            # so patch-file extraction is exact (no diff-text parsing needed).
            _ipc_diff_cache: dict = {}
            _diff_verdict = self._compute_diff_verdict(
                agent_id=agent_id, result=agent_result,
                repo_root=repo_root, diff_cache=_ipc_diff_cache,
            )
            # B5 signal hygiene (findings 2+3): the worker's raw ``_ipc_unassigned``
            # is a GLOBAL ``git status`` view, so subtract pre-run baseline dirt,
            # parallel peers' in-flight edits (global assignment union), and
            # framework-artifact paths (``.asicode/``, ``logs/``). What remains is
            # the worker's genuine NEW out-of-scope writes — the real signal.
            _ipc_unassigned = self._filter_unassigned_changes(
                _ipc_unassigned, subtask.assigned_files,
            )
            # Apply scope_violation_policy (warn / revert / fail) to the genuine
            # out-of-scope strays BEFORE the completion event, so the event reports
            # the post-policy state (reverted list / promoted error status).
            _ipc_scope_reverted = self._apply_scope_violation_policy(
                repo_root, _ipc_unassigned, agent_result,
                agent_id=agent_id, mode="ipc",
            )
            # Attach the filtered out-of-scope signal onto the result (same
            # mutation pattern as _orch_diff_verdict above) so poll_subagent —
            # the orchestrator LLM's only window into a sub-agent in tool-loop
            # mode — can SEE the scope violation at decision time and roll back /
            # re-direct, instead of the signal reaching only the event log.
            try:
                setattr(agent_result, "_orch_unassigned", _ipc_unassigned)
            except (AttributeError, TypeError):
                pass
            self._event_dispatcher.emit(agent_id, "subagent_complete", {
                "task_id": subtask.task_id,
                "status": agent_result.status,
                "final_message": (agent_result.final_message or "")[:300],
                "max_turns_reached": agent_result.status == "max_turns",
                "turns": len(agent_result.turns) if hasattr(agent_result, 'turns') else 0,
                "applied_patches": agent_result.applied_patches if hasattr(agent_result, 'applied_patches') else [],
                # Out-of-scope writes the worker made (B5), after filtering. Non-empty
                # ⇒ a genuine scope violation: the review/diff cross-verification was
                # previously blind to these. Surface them so the UI/consumer can flag
                # or roll back.
                "unassigned_changes": _ipc_unassigned,
                "reverted_out_of_scope": _ipc_scope_reverted,
                "diff_verdict": _diff_verdict,
                "mode": "ipc",
            })
            # (The OUT-OF-SCOPE warning is emitted by _apply_scope_violation_policy
            #  above in ALL policies, so no separate log block is needed here.)
            logger.info(
                "SubAgent %s (IPC) done: status=%s diff_verdict=%s",
                agent_id, agent_result.status, _diff_verdict,
            )
            # The worker finished its task (initial + any review retries) and is
            # now idle — return it to the pool so a subsequent subtask reuses it
            # instead of spawning a fresh worker (P3). Skip this for a cancelled
            # result confirmed exited above (Ctrl-C in the worker's terminal):
            # the process is gone, so returning it would let a later claim
            # re-dispatch to a dead worker.
            if not _worker_confirmed_exited:
                self._return_worker_to_pool(_worker_id)
            else:
                logger.info(
                    "SubAgent %s (IPC) worker %s exited after cancel; not "
                    "returning to reuse pool.", agent_id, _worker_id,
                )
            return agent_result

    def _launch_ipc_worker_terminal_macos(self, agent_id: str, worker_id: str) -> bool:
        """Open a VISIBLE Terminal.app window running the sub-agent worker (macOS).

        Returns True on a successful osascript launch.  AppleScript only accepts
        double-quoted strings (not shlex single quotes), so backslash and double
        quote are escaped.
        """
        _shell_cmd = self._subagent_ipc_commands.get(agent_id, "")
        if not _shell_cmd:
            return False
        _as_escaped = _shell_cmd.replace("\\", "\\\\").replace('"', '\\"')
        _osa = (
            'tell application "Terminal"\n'
            f'    do script "{_as_escaped}"\n'
            "    activate\n"
            "end tell"
        )
        try:
            subprocess.Popen(
                ["osascript", "-e", _osa],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info(
                "Auto-launched Terminal.app for sub-agent %s (worker %s)",
                agent_id, worker_id,
            )
            return True
        except Exception as e:
            logger.warning(
                "Failed to auto-launch terminal for sub-agent %s: %s", agent_id, e,
            )
            return False

    def _spawn_ipc_worker_background(
        self, repo_root: str, worker_id: str, provider: str, model: str,
    ) -> bool:
        """Launch the sub-agent worker as a HEADLESS background subprocess.

        Cross-platform (no Terminal.app / osascript): the worker is the same
        ``asi --subagent`` entry the macOS terminal path runs, just without a
        window.  stdout/stderr are redirected to ``worker.log`` in the worker's
        IPC dir so output is preserved.  The process is detached from the
        orchestrator's signal/process group (POSIX ``start_new_session`` /
        Windows ``CREATE_NEW_PROCESS_GROUP``) so a Ctrl-C in the REPL does NOT
        kill it mid-task — cancellation is signalled via ``cancel.json`` instead.
        On orchestration end the worker exits on the ``shutdown.json`` sentinel,
        with PID-terminate (see ``_cleanup_ipc_workers``) as a belt-and-braces
        orphan guard.  Returns True on a successful spawn.
        """
        # argv from the single source of truth (asr_subagent_argv returns a
        # LIST, so there is no shlex round-trip to mangle Windows backslashes).
        argv = asr_subagent_argv(repo_root) + [
            "--subagent-id", worker_id, "--orch-pid", str(os.getpid()),
        ]
        if provider:
            argv += ["--provider", provider]
        if model:
            argv += ["--model", model]

        _dir = os.path.join(repo_root, ".asicode", "subagents", worker_id)
        _logf: Any = subprocess.DEVNULL
        try:
            os.makedirs(_dir, exist_ok=True)
            _log_path = os.path.join(_dir, "worker.log")
            # Rotate the worker log if it has grown past the cap. A reused worker
            # is one long-lived process serving many tasks, so its log is opened
            # in append ("ab") mode with no bound — without rotation a long
            # orchestration could grow it to hundreds of MB. Move the existing
            # log aside (single .old generation, like logrotate's "1") before
            # opening a fresh append handle. Best-effort: any OSError is ignored
            # (the log is advisory, not load-bearing).
            try:
                if os.path.isfile(_log_path) and os.path.getsize(_log_path) > _WORKER_LOG_ROTATE_BYTES:
                    try:
                        os.replace(_log_path, _log_path + ".old")
                    except OSError:
                        pass
            except OSError:
                pass
            _logf = open(_log_path, "ab")
        except OSError:
            _logf = subprocess.DEVNULL

        _kwargs: dict[str, Any] = dict(
            cwd=repo_root, stdin=subprocess.DEVNULL,
            stdout=_logf, stderr=subprocess.STDOUT,
        )
        if sys.platform == "win32":
            # New process group + no console window: Ctrl-C in the REPL must not
            # propagate to the worker, and no terminal should flash.
            _kwargs["creationflags"] = (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "CREATE_NO_WINDOW", 0)
            )
        else:
            # Own session so the worker leaves the orchestrator's process group
            # (REPL SIGINT won't reach it; cancel.json is the cancel channel).
            _kwargs["start_new_session"] = True

        try:
            _proc = subprocess.Popen(argv, **_kwargs)
        except Exception as e:
            logger.warning(
                "Failed to background-spawn sub-agent worker %s: %s", worker_id, e,
            )
            return False
        finally:
            # Popen duplicated the fd; close our handle (not the DEVNULL sentinel).
            if _logf is not subprocess.DEVNULL:
                try:
                    _logf.close()
                except OSError:
                    pass

        with self._bg_lock:
            self._ipc_worker_procs[worker_id] = _proc
        logger.info(
            "Background-spawned sub-agent worker %s (pid=%s)", worker_id, _proc.pid,
        )
        return True

    def _run_subagent(
        self,
        subtask: SubTaskSpec,
        extra_turns: int = 0,
        task_text: Optional[str] = None,
        original_request: str = "",
    ) -> Any:
        """Instantiate and run one AgentLoop for a single subtask.

        Args:
            subtask: The subtask specification.
            extra_turns: Additional turns on top of base max_turns.
            task_text: Pre-built task description (with predecessor context).
                       If None, builds from subtask.description + assigned_files.
        """
        # ── IPC mode: write task.json, dispatch to an external
        # `asi --subagent` process, wait for result.json. ──────────────
        if self.orch_config.subagent_mode == "ipc":
            return self._run_subagent_ipc(subtask, extra_turns, task_text, original_request)

        from .agent_loop import AgentLoop
        from .tool_registry import AgentConfig

        agent_id = subtask.task_id
        logger.info("SubAgent %s starting: %s", agent_id, subtask.title)

        # emit start event (order guaranteed)
        self._event_dispatcher.emit(agent_id, "subagent_start", {
            "task_id": subtask.task_id,
            "title": subtask.title,
            "description": subtask.description,
            "assigned_files": list(subtask.assigned_files),
            "dependencies": list(subtask.dependencies),
            "priority": subtask.priority,
        })

        # Build per-subagent config (copy base config, override relevant fields)
        base: AgentConfig = self.orch_config.agent_config or AgentConfig()

        # Build a stream callback that forwards events with agent_id tag
        def _sub_cb(event: str, data: dict[str, Any]) -> None:
            data = dict(data)
            data["agent_id"] = agent_id
            self._cb(event, data)

        base_max_turns = base.max_turns + extra_turns
        # Resolve THIS sub-agent's model (per-slot dev_N → lowest dev → orchestrator).
        # Done once here so both the RAG decision and client creation below agree.
        _sub_provider, _sub_model, _sub_api_key = self._resolve_subagent_model(agent_id)
        # Use dataclasses.replace to copy all fields from base, overriding only necessary ones
        # When using a dedicated small subagent model, disable RAG injection to keep
        # the system prompt short (small models OOM/truncate when prompt >8k tokens).
        sub_rag = base.rag_enabled
        if _sub_provider and _sub_model:
            sub_rag = False  # subagent task is already scoped; RAG bloat hurts more than helps
        if self.orch_config.parallel:
            sub_rag = False  # FAISS C extension is not thread-safe; concurrent access causes heap corruption

        # AST-only subtasks benefit from an internal planning phase (DISCOVER→PLAN→EDIT).
        # Non-AST subtasks keep planning disabled — AgentLoop is more flexible without it.
        _sub_planning = (subtask.preferred_lane == "planner")

        sub_config = replace(
            base,
            max_turns=base_max_turns,
            stream_callback=_sub_cb,
            planning_enabled=_sub_planning,   # True for AST-only subtasks
            agent_id=agent_id,
            file_lock_manager=self._file_lock_mgr,
            is_subagent=True,              # skip small-model complexity gating
            rag_enabled=sub_rag,
            context_window_size=base.context_window_size,
            self_planning_enabled=False,   # subagent task is already scoped; self-critique adds ~50s with no benefit
            candidate_selection_enabled=False,  # likewise pre-scoped; candidate selection wastes LLM calls
        )

        # Ensure the subagent has a valid route_decision. Subagents are pre-scoped
        # tasks that run the direct LLM tool loop (MAIN_AGENT lane). The base config
        # from the webapp already carries a route_decision, but callers that pass a
        # bare AgentConfig() (e.g. /orchestrate REPL) leave it None — which makes
        # AgentLoop.run() hit the unhandled-lane guard and return partial_success.
        if getattr(sub_config, 'route_decision', None) is None:
            from .task_router import RouteDecision, Lane, TaskKind
            from .enums import Complexity, Scope
            sub_config.route_decision = RouteDecision(
                task_kind=TaskKind.SINGLE_FILE_EDIT,
                complexity=Complexity.MEDIUM,
                scope=Scope.SINGLE_FILE,
                lane=Lane.MAIN_AGENT,
                confidence=0.9,
                reasoning="orchestrator subagent — pre-scoped task",
            )

        repo_root = self._registry_proto.repo_root
        registry = self._registry_proto.clone_for_subagent(sub_config)

        # subagent-specific client (if a dedicated model is resolved for this slot)
        sub_llm_client = self.llm_client
        sub_model = self.model
        if _sub_provider and _sub_model:
            from ..client import create_llm_client
            try:
                sub_llm_client = create_llm_client(
                    provider=_sub_provider,
                    api_key=_sub_api_key or "",
                    base_url=self.subagent_ollama_url or None,
                )
                sub_model = _sub_model
                logger.info("SubAgent %s → provider=%s model=%s", agent_id, _sub_provider, _sub_model)
            except Exception as e:
                logger.warning("Subagent client creation failed (%s), using orchestrator client", e)

        loop = AgentLoop(
            llm_client=sub_llm_client,
            registry=registry,
            config=sub_config,
            model=sub_model,
            agent_id=agent_id,
            run_store=self._run_store,
        )

        # Build task description — use provided task_text or construct from subtask
        if task_text is None:
            task_text = subtask.description
            if subtask.assigned_files:
                files_hint = ", ".join(subtask.assigned_files)
                task_text = f"{task_text}\n\n[Assigned files: {files_hint}]"
        if original_request and original_request not in task_text:
            task_text = f"[Original request goal]\n{original_request}\n\n[This sub-agent's task]\n{task_text}"

        # Always inject symbol map for assigned files so the sub-agent's internal
        # PlannerAgent uses real symbol names instead of hallucinating them.
        # (Previously this was only done for dedicated small-model sub-agents.)
        hint_lines = []
        for fpath in (subtask.assigned_files or []):
            try:
                abs_path = os.path.join(repo_root, fpath) if repo_root else fpath
                with open(abs_path, encoding="utf-8", errors="replace") as _f:
                    src = _f.read()
                lines_out = _symbol_hint_for_source(src, fpath)
                if lines_out:
                    hint_lines.append(f"[Symbol map for {fpath}]\n" + "\n".join(lines_out))
            except Exception:
                pass
        if hint_lines:
            task_text = task_text + "\n\n" + "\n".join(hint_lines)

        # Snapshot the assigned files' current bytes so that on review rejection
        # we restore the EXACT pre-run content — preserving the user's
        # uncommitted edits and any cross-batch shared-file changes. Reverting
        # via ``git checkout HEAD --`` would clobber all of those.
        _revert_snapshots = _capture_assigned_snapshots(repo_root, subtask.assigned_files)

        # Bind this sub-agent's model to the current thread's run_store context so
        # run-completion telemetry (add_run → _write_to_unified_store) and planner
        # bias reads attribute to the sub-agent's model, not the parent session's
        # planner model (the singleton run_store is shared across sub-agents).
        # Thread-local (per-thread): parallel sub-agents on distinct worker threads
        # are naturally isolated; model_context_scope restores the parent's context
        # on exit so sequential mode (which reuses the parent thread) does not leak
        # the sub-agent model into later parent work. __enter__ before the try +
        # __exit__ in finally avoids re-indenting the run/review block.
        _model_ctx_scope = (
            self._run_store.model_context_scope(sub_model or "", sub_model or "")
            if self._run_store is not None else None
        )
        if _model_ctx_scope is not None:
            _model_ctx_scope.__enter__()

        try:
            result = loop.run(task_text)

            # ── Orchestrator review loop ────────────────────────────────────
            retry_count = 0
            # Per-subagent git-diff memo shared between the review loop and the
            # diff cross-verification below: on the common success path the
            # reviewed file set equals the reported patch set, so the same diff
            # is reused instead of recomputed.  Cleared before each retry (the
            # revert + re-run mutates the worktree, invalidating any cached diff).
            _diff_cache: dict = {}
            while (
                self.review_enabled
                and result.status in ("success", "already_satisfied")
                and retry_count < self.review_max_retries
            ):
                approved, feedback = self._review_subagent_result(
                    agent_id=agent_id,
                    subtask=subtask,
                    result=result,
                    repo_root=repo_root,
                    diff_cache=_diff_cache,
                )
                if approved:
                    break
                # Rejected — revert assigned files, then re-run with feedback
                retry_count += 1
                logger.info("SubAgent %s review rejected (retry %d/%d): %s",
                            agent_id, retry_count, self.review_max_retries, feedback[:120])

                # Restore the pre-run snapshot so the retry starts from a clean
                # baseline WITHOUT destroying uncommitted work. (Snapshot was
                # captured before the first loop.run above.)
                reverted = _restore_assigned_snapshots(repo_root, _revert_snapshots)
                if reverted:
                    logger.info("Restored %s before retry", reverted)
                # policy=revert: also roll back genuine out-of-scope writes so the
                # retry starts from a clean baseline.
                if self.orch_config.scope_violation_policy == "revert":
                    _retry_genuine = self._detect_genuine_violations(
                        repo_root, subtask.assigned_files,
                    )
                    _retry_rv = self._revert_unassigned_changes(repo_root, _retry_genuine)
                    if _retry_rv:
                        logger.info(
                            "Reverted %d out-of-scope file(s) before retry: %s",
                            len(_retry_rv), _retry_rv,
                        )

                # The revert + upcoming re-run mutate the worktree, so any diff
                # cached this attempt is now stale — drop it before re-running.
                _diff_cache.clear()

                self._event_dispatcher.emit(agent_id, "subagent_retry", {
                    "task_id": subtask.task_id,
                    "retry": retry_count,
                    "max_retries": self.review_max_retries,
                    "feedback": feedback,
                    "reverted_files": reverted,
                })
                retry_text = (
                    f"{task_text}\n\n"
                    f"[REVIEW FEEDBACK — attempt {retry_count}]\n"
                    f"{feedback}\n\n"
                    "The previous attempt was reverted. Start fresh and apply the correct change."
                )
                # Fresh loop for the retry (share expensive resources)
                retry_registry = self._registry_proto.clone_for_subagent(sub_config)
                retry_loop = AgentLoop(
                    llm_client=sub_llm_client,
                    registry=retry_registry,
                    config=sub_config,
                    model=sub_model,
                    agent_id=agent_id,
                    run_store=self._run_store,
                )
                result = retry_loop.run(retry_text)
            # ── end review loop ─────────────────────────────────────────────

            # ── Diff cross-verification ────────────────────────────────────────
            # Verify that what the subagent claims matches what actually changed.
            # Diff cross-verification (shared with the IPC path).
            # _compute_diff_verdict parses claimed patch files (dict for IPC,
            # structured objects, or unified-diff TEXT for in-process -- NOT raw
            # file names), scopes `git diff` to them, and attaches the verdict
            # onto `result` so poll_subagent surfaces it at decision time.
            _diff_verdict = self._compute_diff_verdict(
                agent_id=agent_id, result=result,
                repo_root=repo_root, diff_cache=_diff_cache,
            )

            # Genuine out-of-scope detection (parity with the IPC path, which
            # receives the worker-reported unassigned_changes). The in-process path
            # has no such report, so derive the raw changed set from a fresh
            # ``git status --porcelain`` and apply the same filtering + policy.
            _proc_unassigned = self._detect_genuine_violations(
                repo_root, subtask.assigned_files,
            )
            _proc_scope_reverted = self._apply_scope_violation_policy(
                repo_root, _proc_unassigned, result,
                agent_id=agent_id, mode="in-process",
            )

            # emit complete event (order guaranteed)
            self._event_dispatcher.emit(agent_id, "subagent_complete", {
                "task_id": subtask.task_id,
                "status": result.status,
                "final_message": (result.final_message or "")[:300],
                "max_turns_reached": result.status == "max_turns",
                "turns": len(result.turns) if hasattr(result, 'turns') else 0,
                "applied_patches": result.applied_patches if hasattr(result, 'applied_patches') else [],
                "unassigned_changes": _proc_unassigned,
                "reverted_out_of_scope": _proc_scope_reverted,
                "diff_verdict": _diff_verdict,
            })
            logger.info(
                "SubAgent %s (in-process) done: status=%s diff_verdict=%s",
                agent_id, result.status, _diff_verdict,
            )
            return result
        except Exception as e:
            # emit error event
            self._event_dispatcher.emit(agent_id, "subagent_error", {
                "task_id": subtask.task_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
            raise
        finally:
            # Restore the parent thread's model context (see _model_ctx_scope above).
            if _model_ctx_scope is not None:
                _model_ctx_scope.__exit__(None, None, None)

    # ── Orchestrator review ─────────────────────────────────────────────────

    def _get_git_diff(self, repo_root: str, paths: list[str]) -> str:
        """Run ``git diff HEAD -- <paths>`` and return the output.

        The diff is capped at ``OrchestratorConfig.review_diff_char_limit``
        chars (default 8000, ≈2K tokens) so the reviewer LLM gets enough
        context for a meaningful judgement without overflowing its window.
        When ``paths`` is empty the call returns the WHOLE-repo diff — callers
        that must attribute changes to one sub-agent are responsible for
        scoping first (see ``_review_subagent_result``).
        """
        import subprocess
        try:
            cmd = ["git", "diff", "HEAD", "--"] + (paths if paths else [])
            out = subprocess.check_output(cmd, cwd=repo_root, stderr=subprocess.DEVNULL, timeout=10)
            diff = out.decode("utf-8", errors="replace")
            _cap = self.orch_config.review_diff_char_limit
            if len(diff) > _cap:
                # Signal truncation to the reviewer so it knows the diff is
                # incomplete and can adjust its judgement accordingly (B6).
                return f"{diff[:_cap]}\n[diff truncated: {_cap}/{len(diff)} chars]\n"
            return diff
        except Exception as e:
            logger.warning("git diff failed: %s", e)
            return ""

    def _cached_git_diff(
        self, cache: Optional[dict], repo_root: str, paths: list[str],
    ) -> str:
        """``_get_git_diff`` with an optional per-subagent memo.

        Within a single ``_run_subagent`` call the review loop and the diff
        cross-verification both scope git diff to (near-)identical file sets —
        review uses ``assigned_files``, cross-verification uses the files the
        subagent *reported* (``applied_patches``).  In the common success path
        those sets match, so the same ``git diff`` was computed twice.  This
        memo dedups that: keyed by the sorted paths tuple, it returns the cached
        diff on a hit.

        Correctness under retries: the caller clears the cache before each
        retry's re-run (the worktree mutates on revert + re-run), so a stale
        diff can never be served across a mutation.  Between the final approved
        review and cross-verification the worktree is stable, so reuse there is
        safe.  When ``cache`` is None (e.g. direct ``_review_subagent_result``
        calls in tests) this is a thin pass-through to ``_get_git_diff``.
        """
        if cache is None:
            return self._get_git_diff(repo_root, paths)
        key = tuple(sorted(paths))
        if key not in cache:
            cache[key] = self._get_git_diff(repo_root, paths)
        return cache[key]

    @staticmethod
    def _patch_files_have_wt_changes(repo_root: Optional[str], patch_files: list[str]) -> bool:
        """Return True if any of *patch_files* has a working-tree change in ``git status``.

        Complements :meth:`_get_git_diff`: ``git diff HEAD -- <path>`` is EMPTY for
        untracked (newly-created) files, so a sub-agent that only creates files
        would otherwise get a false ``NO_CHANGES`` verdict. ``git status -z
        --porcelain --untracked-files=all -- <paths>`` reports ``??`` for untracked
        and ``M``/``A`` for modified/added entries, so a non-empty result means at
        least one real working-tree change exists among the claimed patch files.

        Any git failure returns ``False`` (degrades to the diff-only verdict).
        """
        if not repo_root or not patch_files:
            return False
        try:
            import subprocess as _sp
            out = _sp.check_output(
                ["git", "status", "-z", "--porcelain", "--untracked-files=all", "--"] + patch_files,
                cwd=repo_root, stderr=_sp.DEVNULL, timeout=10,
            )
            return bool(out.strip())
        except Exception:
            return False

    @staticmethod
    def _synthesize_untracked_diff(
        repo_root: str, assigned_files: list[str], char_limit: int = 0,
    ) -> str:
        """Build a synthetic unified-diff for untracked (newly-created) files.

        ``git diff HEAD`` returns EMPTY for files that have never been committed,
        so a sub-agent whose task is purely file-creation would get a "no changes"
        rejection from the review pipeline. This method reads the assigned files
        and formats them as ``--- /dev/null`` / ``+++ b/path`` additions so the
        reviewer LLM can judge the actual content.

        Only files that exist on disk AND are untracked are included. Returns
        empty string if no files qualify.

        Per-file inlining is capped at ``_SYNTH_DIFF_FILE_BYTES``: only that
        many bytes are read from disk (a huge generated/binary file must not be
        slurped into RAM just to be truncated later), and a marker line notes
        the omission. The caller applies the overall ``review_diff_char_limit``
        cap, same as the ``git diff`` path.

        When ``char_limit > 0`` is supplied, the build STOPS as soon as the
        accumulated output reaches that cap — without this, a batch of dozens of
        new files is read (64 KiB each) and formatted even though the caller
        truncates the result back down to the cap anyway. A marker line notes
        that further untracked files were omitted.
        """
        import subprocess as _sp
        if not repo_root or not assigned_files:
            return ""
        # Check which assigned files are actually untracked (git reports "??" for them).
        try:
            out = _sp.check_output(
                ["git", "status", "-z", "--porcelain", "--untracked-files=all", "--"] + assigned_files,
                cwd=repo_root, stderr=_sp.DEVNULL, timeout=10,
            )
        except Exception:
            return ""
        untracked: set[str] = set()
        parts = out.decode("utf-8", errors="replace").split("\x00")
        for p in parts:
            if p.startswith("?? "):
                untracked.add(os.path.normpath(p[3:]))
            # "A " (staged new file) is also opaque to git diff HEAD but tracked;
            # skip those — they show up in the normal diff.
        if not untracked:
            return ""
        lines: list[str] = []
        _sep = os.sep
        _cum_chars = 0
        _cap_hit = False
        for _f in assigned_files:
            if _cap_hit:
                break
            _nf = os.path.normpath(_f)
            # Use prefix-aware matching: an unexpanded directory assignment (cap-exceeded)
            # won't be in the untracked set, but untracked files UNDER it should still
            # be discovered (B6 / turn 13118).
            _matching = {
                u for u in untracked
                if u == _nf or u.startswith(_nf + _sep)
            }
            if not _matching:
                continue
            for _untracked_file in sorted(_matching):
                # Early termination: once we've already produced enough text to
                # fill the reviewer's char budget, stop reading/formatting more
                # files — the caller truncates to char_limit anyway, so any
                # further work is discarded.
                if char_limit and _cum_chars >= char_limit:
                    lines.append(
                        f"+[... further untracked files omitted: synthetic diff "
                        f"reached the {char_limit}-char cap before reading all "
                        f"files ...]"
                    )
                    _cap_hit = True
                    break
                _abs = os.path.join(repo_root, _untracked_file) if repo_root else _untracked_file
                try:
                    _size = os.path.getsize(_abs)
                    with open(_abs, "rb") as _fh:
                        content = _fh.read(_SYNTH_DIFF_FILE_BYTES)
                except Exception:
                    continue
                _added: list[str] = []
                _added.append("--- /dev/null")
                _added.append(f"+++ b/{_untracked_file}")
                _body = content.decode("utf-8", errors="replace")
                _body_lines = _body.splitlines(keepends=False)
                _added.append(f"@@ -0,0 +1,{len(_body_lines)} @@")
                for _bl in _body_lines:
                    _added.append(f"+{_bl}")
                if _size > _SYNTH_DIFF_FILE_BYTES:
                    _added.append(
                        f"+[file content truncated: showing first "
                        f"{_SYNTH_DIFF_FILE_BYTES} of {_size} bytes]"
                    )
                _added.append("")
                lines.extend(_added)
                _cum_chars += sum(len(_s) for _s in _added)
        if not lines:
            return ""
        return "\n".join(lines)

    def _capture_scope_baseline(self, repo_root: str, subtasks: list) -> None:
        """Snapshot the pre-run dirty set + global assignment union (B5 / findings 2+3).

        Called once at ``run()`` start in DECOMPOSITION mode, BEFORE any sub-agent
        edits, so :meth:`_filter_unassigned_changes` can subtract both from each
        worker's raw global-``git status`` report. ``_baseline_dirty_paths`` removes
        pre-existing dirt (e.g. ``.env`` already modified before the run);
        ``_global_assigned_paths`` removes parallel peers' in-flight edits to their
        OWN assigned files.

        Tool-loop mode (the primary mode for ``--orchestrate`` and the REPL) does
        NOT call this — subtasks spawn dynamically, so there is no upfront
        decomposition to build the union from. Instead it uses two hooks of
        equivalent effect: (a) ``_run_tool_loop`` snapshots the pre-run dirty set
        at loop start, (b) ``_tool_spawn_subagent`` accumulates each subtask's
        ``assigned_files`` into ``_global_assigned_paths`` as it spawns.
        """
        self._baseline_dirty_paths = _snapshot_dirty_path_set(repo_root)
        self._global_assigned_paths = {
            os.path.normpath(_f)
            for _st in (subtasks or [])
            for _f in (getattr(_st, "assigned_files", None) or [])
        }

    def _filter_unassigned_changes(
        self, reported: list, own_assigned: Optional[list[str]] = None,
    ) -> list:
        """Filter a worker's raw out-of-scope report to its genuine NEW violations.

        Subtracts three classes of false positives so the scope-violation signal
        (B5) is trustworthy instead of drowned in noise (turn 13106 verification
        observed all three):

        1. **Pre-run baseline dirt** (``.env`` already dirty before the run) — the
           worker didn't create it. Subtracted via ``_baseline_dirty_paths``.
        2. **Parallel peers' in-flight edits** — ``git status`` is global, so in a
           parallel batch worker-2 sees worker-1's edit to worker-1's OWN assigned
           file as "out of scope for worker-2". Subtracted via
           ``_global_assigned_paths`` (the union of ALL subtask assignments).
        3. **Framework artifacts** (``.asicode/``, ``logs/``) — the orchestration
           infra's own files (worker.log, run logs, IPC sentinels), never a user
           edit. Subtracted via :func:`_is_infra_path`.

        What remains is the worker's actual NEW writes outside its assignment —
        the real signal. Both baselines are captured in BOTH modes now:
        decomposition mode via :meth:`_capture_scope_baseline` (baseline + full
        union once), tool-loop mode via the loop-start dirty snapshot (hook a)
        plus incremental per-spawn union accumulation (hook b). The only residual
        degrade path is a direct ``_run_subagent_ipc`` test call that bypasses
        ``run()``/``_run_tool_loop`` — there items 1+2 silently no-op and only
        infra + this worker's own assignment are subtracted.

        Known structural limitation (turn 13110 bug 3): because ``git status`` is
        a GLOBAL view of ONE shared working tree, item 2 subtracts the FULL peer
        union unconditionally. A worker that VIOLATES a peer's assigned file is
        therefore indistinguishable from the peer's own legitimate in-flight edit
        and gets filtered out (false negative on the violation). This is
        irreducible under the shared-worktree IPC design — true per-worker
        attribution needs either per-worker git worktrees or filesystem-level
        process tracking, both out of scope here. The subtraction above is
        deliberately broad: it favours a trustworthy signal (no noise from peers'
        own edits — the common case) over catching the rare cross-peer write
        conflict, which surfaces instead as a merge/content conflict downstream.
        """
        if not reported:
            return []
        from .subagent_ipc import _path_matches_scope
        _own = {os.path.normpath(_f) for _f in (own_assigned or [])}
        # _baseline_dirty_paths is captured once at run/loop start and never
        # mutated after, so a bare read is race-free.
        _baseline = self._baseline_dirty_paths or set()
        # _global_assigned_paths is mutated incrementally in tool-loop mode
        # (spawn_subagent runs in the orchestrator's main thread) while
        # IPC-completion threads read it here. Snapshot under _bg_lock to avoid
        # "set changed size during iteration". Decomposition mode sets it once
        # before any sub-agent runs, so the lock is a cheap no-op there.
        with self._bg_lock:
            _global = set(self._global_assigned_paths or set())
        out: list = []
        for _e in reported:
            _raw = (_e.get("file") if isinstance(_e, dict) else _e) or ""
            _f = os.path.normpath(_raw)
            if not _f:
                continue
            if _f in _baseline:
                continue
            # Use prefix-aware matching so unexpanded directory entries (those
            # that exceeded _MAX_DIR_EXPANSION_FILES) still match files under
            # them for both peer-union and own-assignment (B6 / turn 13116).
            if _path_matches_scope(_f, _global) or _path_matches_scope(_f, _own):
                continue
            if _is_infra_path(_f):
                continue
            out.append(_e)
        return out

    def _git_status_changed_paths(self, repo_root: Optional[str]) -> list:
        """Return raw changed paths from ``git status -z --porcelain``.

        Same shape as the worker's IPC ``unassigned_changes`` report
        (``[{"file": ...}]``), so the same :meth:`_filter_unassigned_changes`
        pipeline (baseline / peer / infra subtraction) applies. The in-process
        path has no worker-reported list, so it derives the raw changed set from a
        fresh git status here — giving it the same genuine-violation detection the
        IPC path already has.

        Delegates to :func:`partition_changed_files` (empty assignment ⇒ every
        changed file is "in-scope") rather than re-implementing the porcelain
        parser: the ``-z`` format emits a rename/copy's source path as a SECOND
        NUL-delimited field that a naive ``split("\\x00")`` + ``[3:]`` would
        mis-parse as a phantom mangled path. ``partition_changed_files`` already
        handles both status columns and the two-field consume correctly.
        """
        if not repo_root:
            return []
        from .subagent_ipc import partition_changed_files
        # Empty assignment → all changed paths returned in the in-scope half.
        try:
            return partition_changed_files(repo_root, [])[0]
        except Exception:
            return []

    def _detect_genuine_violations(
        self, repo_root: Optional[str], assigned_files: Optional[list[str]],
        *, raw_unassigned: Optional[list] = None,
    ) -> list:
        """Return this worker's filtered genuine out-of-scope writes.

        The IPC path passes the worker-reported ``raw_unassigned``; the in-process
        path passes ``None``, which derives the raw set from a fresh
        :meth:`_git_status_changed_paths`. Both feed
        :meth:`_filter_unassigned_changes` so the identical baseline / peer / infra
        subtractions apply — the result is directly safe to revert.
        """
        if raw_unassigned is None:
            raw_unassigned = self._git_status_changed_paths(repo_root)
        return self._filter_unassigned_changes(raw_unassigned, assigned_files)

    def _revert_unassigned_changes(
        self, repo_root: Optional[str], unassigned: list,
    ) -> list[str]:
        """Revert genuine out-of-scope writes to their pre-run state.

        Every entry has passed :meth:`_filter_unassigned_changes`, so it is
        neither pre-run baseline dirt nor any worker's assigned file nor an
        infra path — it did not exist (in that state) before the run. Therefore:

        * **tracked** (``git ls-files`` reports it) →
          ``git checkout HEAD -- <file>`` restores the clean pre-run content.
        * **untracked** (created during the run) → ``unlink`` (file) or
          ``shutil.rmtree`` (directory).

        Returns the list of rel-paths actually reverted. Failures (git error,
        permission, …) are logged and skipped, never raised — revert is
        best-effort and must not crash the completion path.

        Subprocess batching: a stray-heavy run previously spawned TWO git
        subprocesses per file (``ls-files --error-unmatch`` + ``checkout``).
        Tracked determination is now ONE ``git ls-files -- <all files>`` (git
        prints exactly the tracked subset, omitting untracked ones), and tracked
        reverts are ONE batched ``git checkout HEAD -- <all tracked>``. Untracked
        files are ``unlink``-ed directly (no git subprocess). Any batch git call
        failing falls back to the original per-file path so one bad path doesn't
        abort the rest.
        """
        import shutil
        import subprocess as _sp
        reverted: list[str] = []

        # Normalize entries and split file vs directory targets in one pass.
        # Directories are always untracked (git tracks files) → rmtree path.
        file_entries: list[tuple[str, str]] = []  # (rel_path, abs_path)
        dir_entries: list[tuple[str, str]] = []
        for _e in (unassigned or []):
            _raw = (_e.get("file") if isinstance(_e, dict) else _e) or ""
            _f = os.path.normpath(_raw)
            if not _f or _is_infra_path(_f):
                continue
            _abs = os.path.join(repo_root, _f) if repo_root else _f
            if os.path.isdir(_abs):
                dir_entries.append((_f, _abs))
            else:
                file_entries.append((_f, _abs))

        def _checkout_one(rel: str) -> None:
            try:
                _ck = _sp.run(
                    ["git", "checkout", "HEAD", "--", rel],
                    cwd=repo_root, capture_output=True, timeout=10,
                )
                if _ck.returncode == 0:
                    reverted.append(rel)
                else:
                    logger.warning(
                        "scope_violation revert: git checkout failed for %r: %s",
                        rel, _ck.stderr.decode("utf-8", "replace").strip(),
                    )
            except Exception as _err:
                logger.warning("scope_violation revert: failed for %r: %s", rel, _err)

        def _unlink_one(rel: str, abs_path: str) -> None:
            try:
                if os.path.isfile(abs_path):
                    os.unlink(abs_path)
                    reverted.append(rel)
                # else: file already gone — nothing to revert, do NOT report
            except Exception as _err:
                logger.warning("scope_violation revert: failed for %r: %s", rel, _err)

        # Batch tracked determination: one `git ls-files -z -- <files>` prints
        # exactly the tracked subset. `-z` is REQUIRED — without it git C-quotes
 # non-ASCII paths ("\355\225\234..." for Korean characters), which then fail the
        # `in tracked` membership test and get misclassified as untracked →
        # deleted by unlink instead of restored by checkout (data loss + a
        # falsely-"reverted" report). `-z` emits NUL-separated raw bytes with no
        # quoting at all, so every entry matches its unnormalized rel-path.
        # None means the batch call failed → per-file fallback below.
        tracked: Optional[set[str]] = None
        if file_entries:
            try:
                _r = _sp.run(
                    ["git", "ls-files", "-z", "--"] + [_f for _f, _ in file_entries],
                    cwd=repo_root, capture_output=True, timeout=20,
                )
                if _r.returncode == 0:
                    tracked = {
                        _p for _p in _r.stdout.decode("utf-8", "replace").split("\0") if _p
                    }
            except Exception:
                tracked = None

        if tracked is not None:
            # Batched checkout for the tracked subset; per-file fallback on any
            # failure so one bad path doesn't abort the whole batch.
            _tracked_files = [_f for _f, _ in file_entries if _f in tracked]
            if _tracked_files:
                _batch_ok = False
                try:
                    _ck = _sp.run(
                        ["git", "checkout", "HEAD", "--"] + _tracked_files,
                        cwd=repo_root, capture_output=True, timeout=30,
                    )
                    if _ck.returncode == 0:
                        reverted.extend(_tracked_files)
                        _batch_ok = True
                    else:
                        logger.warning(
                            "scope_violation revert: batched git checkout failed: %s",
                            _ck.stderr.decode("utf-8", "replace").strip(),
                        )
                except Exception as _err:
                    logger.warning(
                        "scope_violation revert: batched git checkout failed: %s", _err,
                    )
                if not _batch_ok:
                    for _f, _ in file_entries:
                        if _f in tracked:
                            _checkout_one(_f)
            # Untracked files: unlink directly (no git subprocess needed).
            for _f, _abs in file_entries:
                if _f not in tracked:
                    _unlink_one(_f, _abs)
        else:
            # Batch ls-files failed → per-file tracked determination, then
            # checkout (tracked) or unlink (untracked). This is the original
            # one-subprocess-pair-per-file behaviour, preserved as a fallback.
            for _f, _abs in file_entries:
                _is_tracked = False
                try:
                    _tr = _sp.run(
                        ["git", "ls-files", "--error-unmatch", "--", _f],
                        cwd=repo_root, capture_output=True, timeout=10,
                    )
                    _is_tracked = (_tr.returncode == 0)
                except Exception:
                    _is_tracked = False
                if _is_tracked:
                    _checkout_one(_f)
                else:
                    _unlink_one(_f, _abs)

        # Directories (always untracked): best-effort rmtree.
        for _f, _abs in dir_entries:
            try:
                shutil.rmtree(_abs, ignore_errors=True)
                reverted.append(_f)
            except Exception as _err:
                logger.warning("scope_violation revert: failed for %r: %s", _f, _err)

        return reverted

    def _apply_scope_violation_policy(
        self, repo_root: Optional[str], genuine: list, result: Any,
        *, agent_id: str, mode: str,
    ) -> list[str]:
        """Apply ``scope_violation_policy`` to genuine out-of-scope writes.

        ``genuine`` is the filtered list (output of
        :meth:`_detect_genuine_violations` / :meth:`_filter_unassigned_changes`) —
        only genuine NEW strays.

        * ``warn``  — log + leave (current default behaviour). No-op return.
        * ``revert``— :meth:`_revert_unassigned_changes`; returns reverted paths.
        * ``fail``  — promote ``result.status`` to ``"error"`` (best-effort: result
          may be an immutable/foreign object).

        Returns the list of reverted rel-paths (empty unless policy == "revert").
        The warning log is emitted in ALL policies (including warn) so the signal is
        never silently dropped.
        """
        if not genuine:
            return []
        _policy = (getattr(self.orch_config, "scope_violation_policy", "warn")
                   or "warn")
        _files = [
            (_e.get("file") if isinstance(_e, dict) else _e) for _e in genuine
        ]
        logger.warning(
            "SubAgent %s (%s) made OUT-OF-SCOPE changes (policy=%s): %s",
            agent_id, mode, _policy, _files,
        )
        if _policy == "revert":
            _rv = self._revert_unassigned_changes(repo_root, genuine)
            if _rv:
                logger.info(
                    "SubAgent %s (%s) reverted %d out-of-scope file(s): %s",
                    agent_id, mode, len(_rv), _rv,
                )
            return _rv
        if _policy == "fail":
            _stamp = "[scope_violation: out-of-scope writes detected — policy=fail]"
            if hasattr(result, "status") and result.status != "error":
                result.status = "error"
            if hasattr(result, "final_message"):
                _prev = (result.final_message or "").strip()
                result.final_message = (f"{_prev}\n{_stamp}".strip()
                                        if _prev else _stamp)
            logger.warning(
                "SubAgent %s (%s) scope_violation_policy=fail: promoted to error",
                agent_id, mode,
            )
        # "warn" (or any unknown value): log only, already emitted above.
        return []

    def _compute_diff_verdict(
        self, *, agent_id: str, result: Any,
        repo_root: Optional[str], diff_cache: Optional[dict],
    ) -> str:
        """Cross-verify a sub-agent's reported edits against ``git diff`` and
        attach the verdict (``_orch_diff_verdict``) onto ``result``.

        Shared by the in-process (``_run_subagent``) and IPC
        (``_run_subagent_ipc``) paths so the parallel-mode ``poll_subagent``
        loop sees VERIFIED / NO_CHANGES / UNVERIFIABLE at decision time in BOTH
        modes -- not just the non-macOS in-process fallback.

        Patch extraction is mode-aware:
          * dict (IPC ``{"file": ...}``)        -> take the file name
          * object with ``file_path``            -> take the attr
          * str (in-process)                     -> the value is a full unified
            diff TEXT (``write_tools`` appends ``str(patch)``), NOT a file name.
            Feeding that text to ``git diff HEAD -- <text>`` matches nothing and
            yields a false NO_CHANGES, so the real paths are parsed out via
            ``extract_files_from_patch``.

        Returns the verdict string; ``result`` is mutated in place.
        """
        from ._shared_utils import extract_files_from_patch
        _raw_patches = getattr(result, 'applied_patches', None) or []
        _patch_files: list[str] = []
        for _p in _raw_patches:
            if isinstance(_p, dict):
                _f = _p.get("file") or _p.get("file_path", "")
                if _f and _f not in _patch_files:
                    _patch_files.append(_f)
            elif isinstance(_p, str) and _p.strip():
                for _f in extract_files_from_patch(_p):
                    if _f and _f not in _patch_files:
                        _patch_files.append(_f)
            elif hasattr(_p, "file_path"):
                _f = getattr(_p, "file_path", "")
                if _f and _f not in _patch_files:
                    _patch_files.append(_f)
        _actual_diff = ""
        _has_wt_change = False  # NEW/untracked or modified files seen by `git status`
        _can_attribute = bool(_patch_files) or not self.orch_config.parallel
        if repo_root and _patch_files:
            _actual_diff = self._cached_git_diff(diff_cache, repo_root, _patch_files)
            # `git diff HEAD -- <path>` returns EMPTY for untracked (newly-created)
            # files, so a worker whose only output is a brand-new file would get a
            # false NO_CHANGES (turn 13106 verification: dev_2 created notes.txt but
            # got diff_verdict=NO_CHANGES). `git status --porcelain -- <paths>`
            # shows '??' for untracked and 'M'/'A' for modified/added, so it catches
            # both — a non-empty status means a real working-tree change exists.
            _has_wt_change = self._patch_files_have_wt_changes(repo_root, _patch_files)
        elif repo_root and not _patch_files and not self.orch_config.parallel:
            _actual_diff = self._cached_git_diff(diff_cache, repo_root, [])
        _claims_changes = result.status in ("success", "already_satisfied", "max_turns", "partial_success")
        if _actual_diff:
            _diff_verdict = "VERIFIED"
            logger.debug("[DIFF_VERIFY] %s VERIFIED (%d chars diff)", agent_id, len(_actual_diff))
        elif _has_wt_change and _claims_changes and _can_attribute:
            # New/untracked file: present in `git status` but absent from
            # `git diff HEAD`. This IS a real change, so VERIFIED — not the false
            # NO_CHANGES the diff-only check produced before.
            _diff_verdict = "VERIFIED"
            logger.debug(
                "[DIFF_VERIFY] %s VERIFIED via new/untracked file (git status), "
                "diff HEAD empty (patches=%s)", agent_id, _patch_files,
            )
        elif _claims_changes and _can_attribute:
            _diff_verdict = "NO_CHANGES"
            logger.warning(
                "[DIFF_VERIFY] %s status=%s but git diff AND git status are empty "
                "— final_message may be inaccurate (patches=%s)",
                agent_id, result.status, _patch_files or "none",
            )
        else:
            _diff_verdict = "UNVERIFIABLE"
        try:
            setattr(result, "_orch_diff_verdict", _diff_verdict)
        except (AttributeError, TypeError):
            pass
        return _diff_verdict

    def _locate_symbol(self, symbol: str, file_paths: list[str], repo_root: str) -> str:
        """Find a class/function definition in assigned files.

        Returns a ready-to-use hint string for the review-retry prompt: the
        exact file path, line number, and a few lines of surrounding code.
        Multi-language via ``_symbol_def_line`` (tree-sitter ``find_all_symbols``
        + stdlib-ast fallback) — the same single source of truth as
        ``_symbol_hint_for_source``. This replaces a Python-only
        ``^\\s*(class|def)`` regex that silently missed TS/JS/Go/Java symbols
        and could false-match inside strings/comments.
        """
        results = []
        for rel_path in file_paths:
            abs_path = os.path.join(repo_root, rel_path)
            try:
                with open(abs_path, encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
            except OSError:
                continue
            line_no = _symbol_def_line("".join(lines), rel_path, symbol)
            if line_no:
                # Definition line + up to 5 following lines as context.
                snippet = "".join(lines[line_no - 1: line_no + 5])
                results.append(
                    f"Found `{symbol}` in {rel_path} at line {line_no}:\n"
                    f"```\n{snippet}```\n"
                    f"→ Use bash (cat -n) on '{rel_path}' starting at line {line_no} "
                    f"to read it, then write_plan with exact text from that result."
                )
        return "\n".join(results) if results else ""

    _REVIEW_SYSTEM = """\
You are a senior software engineer reviewing a subagent's code change.
Given the original subtask description and the resulting git diff, decide if the change is correct.

Return ONLY a JSON object — no prose, no markdown:
{
  "approved": true/false,
  "feedback": "concise explanation in the user's language (≤200 chars). Empty string if approved.",
  "target_symbol": "ExactClassName or function_name the subagent should have modified. Empty string if approved."
}

Reject if ANY of the following:
- Existing code was deleted instead of extended (e.g. __init__ replaced instead of __str__ added)
- The wrong class or function was modified
- The diff is empty but changes were expected
- The change introduces obvious syntax errors or logical bugs
- The task asked to ADD but the agent REPLACED
Approve if the change correctly implements what was asked without breaking existing code."""

    def _review_subagent_result(
        self,
        agent_id: str,
        subtask: "SubTaskSpec",
        result: Any,
        repo_root: str,
        *,
        diff_cache: Optional[dict] = None,
    ) -> tuple[bool, str]:
        """Call orchestrator LLM to review the subagent's diff. Returns (approved, feedback).

        ``diff_cache`` (optional) memoizes the git diff so the cross-verification
        pass in ``_run_subagent`` can reuse the exact diff this review judged —
        avoiding a redundant ``git diff`` on the common success path where the
        reviewed file set equals the reported patch set.  See ``_cached_git_diff``.
        """
        from ..client import LLMMessage

        # If no files are assigned we CANNOT attribute a diff to this sub-agent.
        # ``_get_git_diff(repo_root, [])`` would return the ENTIRE repo diff —
        # and in parallel mode sibling sub-agents are editing the worktree
        # concurrently, so the reviewer would judge a diff contaminated with
        # other agents' changes.  Skip review (assume approved) instead of
        # acting on a misleading diff.
        if not subtask.assigned_files:
            _why = ("parallel siblings" if self.orch_config.parallel
                    else "no assigned_files to scope the diff")
            logger.info(
                "SubAgent %s review skipped — %s; cannot attribute diff",
                agent_id, _why,
            )
            self._event_dispatcher.emit(agent_id, "review_skipped", {
                "task_id": subtask.task_id,
                "reason": "no_assigned_files" if not self.orch_config.parallel
                          else "no_assigned_files_parallel",
            })
            return True, ""  # unverifiable → do not block on a misleading diff

        self._event_dispatcher.emit(agent_id, "review_start", {
            "task_id": subtask.task_id,
            "title": subtask.title,
        })

        diff = self._cached_git_diff(diff_cache, repo_root, subtask.assigned_files)
        if not diff:
            # git diff HEAD is EMPTY for untracked (newly-created) files, so a
            # sub-agent that only creates files gets a false "no changes" verdict.
            # Check via git status before rejecting (B6 / turn 13116).
            if self._patch_files_have_wt_changes(repo_root, subtask.assigned_files):
                _cap = self.orch_config.review_diff_char_limit
                # Pass the char cap into the builder so it STOPS reading/formatting
                # once the output already fills the reviewer's budget — without this,
                # dozens of new files (64 KiB read each) are processed only to be
                # truncated back down to _cap below.
                _untracked_diff = self._synthesize_untracked_diff(
                    repo_root, subtask.assigned_files, char_limit=_cap,
                )
                if _untracked_diff:
                    # Apply the SAME overall cap as _get_git_diff: without it a
                    # large untracked file flows uncapped into the reviewer
                    # prompt (the git-diff path is capped, this one was not).
                    if len(_untracked_diff) > _cap:
                        _untracked_diff = (
                            f"{_untracked_diff[:_cap]}\n[diff truncated: "
                            f"{_cap}/{len(_untracked_diff)} chars]\n"
                        )
                    diff = _untracked_diff
            if not diff:
                # Genuinely no changes — reject
                self._event_dispatcher.emit(agent_id, "review_rejected", {
                    "task_id": subtask.task_id,
                    "feedback": "No changes detected (git diff is empty). The subagent must modify the assigned files.",
                })
                return False, "No changes detected in git diff. The subagent must modify the assigned files."

        user_content = (
            f"Subtask: {subtask.title}\n"
            f"Description: {subtask.description}\n\n"
            f"Git diff:\n```diff\n{diff}\n```"
        )
        messages = [
            LLMMessage(role="system", content=self._REVIEW_SYSTEM),
            LLMMessage(role="user", content=user_content),
        ]

        raw = simple_llm_call(self.llm_client, self.model, messages, thinking_mode=False)
        parsed = parse_json(raw)

        if not parsed:
            # Parse failure → assume approved to avoid blocking
            logger.warning("Review parse failed for %s, assuming approved. Raw (%d chars): %s", agent_id, len(raw), raw[:2000])
            self._event_dispatcher.emit(agent_id, "review_approved", {
                "task_id": subtask.task_id, "note": "parse_failed_assumed_ok",
            })
            return True, ""

        approved = bool(parsed.get("approved", True))
        feedback = parsed.get("feedback", "")
        target_symbol = parsed.get("target_symbol", "")

        # When rejected, locate the correct target and inject precise location into feedback
        if not approved and target_symbol:
            location_hint = self._locate_symbol(target_symbol, subtask.assigned_files, repo_root)
            if location_hint:
                feedback = f"{feedback}\n\n[LOCATION HINT]\n{location_hint}"

        event = "review_approved" if approved else "review_rejected"
        self._event_dispatcher.emit(agent_id, event, {
            "task_id": subtask.task_id,
            "approved": approved,
            "feedback": feedback,
            "target_symbol": target_symbol,
        })
        return approved, feedback

    # ── Result synthesis (legacy — used by run() for backward compat) ──────
    # New code should use PlannerAgent.summarize_results() instead.

    def _synthesize(
        self,
        request: str,
        subtasks: list[SubTaskSpec],
        results: list[Any],
    ) -> str:
        from ..client import LLMMessage

        parts = [f"Original request: {request}\n"]
        for st, res in zip(subtasks, results, strict=False):
            if res is None:
                continue
            parts.append(
                f"[{st.task_id}] {st.title}\n"
                f"  Status: {res.status}\n"
                f"  Summary: {(res.final_message or '')[:200]}\n"
            )
        user_content = "\n".join(parts)

        messages = [
            LLMMessage(role="system", content=_SYNTHESISE_SYSTEM),
            LLMMessage(role="user", content=user_content),
        ]
        return simple_llm_call(self.llm_client, self.model, messages, thinking_mode=False) or "Multi-agent task completed."

    def _paired_subtask_results(self) -> list[tuple[SubTaskSpec, Any]]:
        """Build (subtask, result) pairs keyed by agent_id from _bg_subagents.

        Single source of truth for tool-loop summary pairing: each subtask is
        paired with ITS OWN result (held in its bg-subagents entry), not
        positionally against the completion-ordered _bg_results list. A
        sub-agent that never produced a result (timeout / crash before append)
        pairs with None, which _synthesize_from_subtasks surfaces as "no
        result" so the gap is visible rather than silently truncated.
        """
        with self._bg_lock:
            return [
                (e["subtask"], e.get("result"))
                for e in self._bg_subagents.values()
                if e.get("subtask")
            ]

    def _synthesize_from_subtasks(
        self,
        pairs: list[tuple[SubTaskSpec, Any]],
    ) -> str:
        """
        No-LLM fallback summary for tool-loop-mode orchestration.

        Each (subtask, result) pair is keyed by agent_id at the call site (built
        from _bg_subagents via _paired_subtask_results), so out-of-order
        completion can't cross-wire a subtask's title with another's status /
        message -- _bg_results is completion-ordered (threads append as they
        finish) while a spawn-ordered subtask list would mis-pair under a
        positional zip. A sub-agent that never produced a result (timeout /
        crash before its result was appended) pairs with None and is surfaced
        here as "no result" instead of being silently dropped (a strict=False
        zip truncated at the shorter list and hid the gap).
        """
        lines = []
        for st, res in pairs:
            if res is None:
                lines.append(f"- [{st.task_id}] {st.title}: no result")
                continue
            icon = "✅" if res.status in ("success", "already_satisfied") else ("⚠️" if res.status == "max_turns" else "❌")
            msg = (res.final_message or "")[:120]
            lines.append(f"- {icon} [{st.task_id}] {st.title} ({res.status}): {msg}")
        return "Multi-agent task results:\n" + "\n".join(lines) if lines else "Multi-agent task completed."

    # ═══════════════════════════════════════════════════════════════════════
    # ── Tool-loop mode ─────────────────────────────────────────────────────
    # The orchestrator drives a REAL DesignChatLoop (the same loop the
    # interactive REPL uses) via _OrchestratorBackedRegistry, gaining the FULL
    # design-chat tool set plus 3 delegation tools:
    #   spawn_subagent  — launch a sub-agent in the BACKGROUND (non-blocking).
    #   poll_subagent   — check / collect a background sub-agent's result.
    #   list_subagents  — enumerate active + completed sub-agents.
    #   <full registry tools> — read_file, get_file_outline, grep, bash,
    #                           edit_text, apply_patch, modify_symbol, …
    # Sub-agents run concurrently while the orchestrator keeps calling tools —
    # the orchestrator is never blocked waiting on a single sub-agent.
    # ═══════════════════════════════════════════════════════════════════════
    _ORCH_TOOL_LOOP_SYSTEM = (
        "You are the ORCHESTRATOR agent. You drive a multi-agent coding task "
        "by delegating work to sub-agents AND by doing work yourself. You have "
        "the FULL design-chat tool set, so you can read, edit, search, run "
        "commands, and delegate — all in the same loop.\n\n"
        "## Your tools\n"
        "- spawn_subagent: delegate a self-contained coding task to a NEW "
        "sub-agent. The sub-agent runs in the background (non-blocking) and "
        "you get back its agent_id immediately. Give it a precise, complete "
        "task description and list the files it should touch.\n"
        "- poll_subagent: check whether a spawned sub-agent has finished. "
        "Returns its status and result (or a 'still running' message). Poll "
        "periodically — do NOT block on a single sub-agent.\n"
        "- list_subagents: see all spawned sub-agents and their current status.\n"
        "- All design-chat tools are available to you: read_file / read_symbol / "
        "get_file_outline / grep / find_relevant_files / find_symbol / bash / "
        "search_web / web_fetch to inspect code or docs; AND edit_text / "
        "apply_patch / modify_symbol / anchor_edit / write_plan to make changes "
        "yourself.\n\n"
        "## How to work (iterative loop — do NOT just wait)\n"
        "You are in a TOOL LOOP. Each turn you may call ONE OR MORE tools. "
        "The loop continues until you produce a plain-text final answer.\n\n"
        "1. **Understand** — use read tools to learn the codebase before "
        "planning sub-tasks.\n"
        "2. **Decompose & delegate** — split the work. Spawn sub-agents for "
        "substantial or independent pieces. You may spawn several concurrently.\n"
        "3. **Work in parallel** — CRITICAL: while sub-agents run in the "
        "background, do NOT sit idle polling. KEEP WORKING yourself: read other "
        "areas, make small edits, verify earlier results, or prepare the next "
        "sub-task. The orchestrator is never blocked.\n"
        "4. **Collect & verify** — when poll_subagent returns a completed "
        "result, EXAMINE it. Read the files the sub-agent changed and confirm "
        "the change landed correctly. If the result is wrong or incomplete, "
        "re-delegate with refined instructions or fix small issues yourself.\n"
        "5. **Iterate** — repeat steps 3-4 until the overall goal is truly met. "
        "You may spawn more sub-agents for follow-up work at any time.\n"
        "6. **Summarize** — ONLY when you are confident the user's request is "
        "fully addressed AND you have verified the results, write a "
        "comprehensive final summary of what each sub-agent did, what you did "
        "yourself, and the overall outcome. This is your LAST turn — do not end "
        "earlier just because a sub-agent finished.\n"
        "7. Respond in the SAME LANGUAGE as the user's request.\n"
    )
    def _native_orchestrator_schemas(self) -> list[dict[str, Any]]:
        """The 3 orchestrator-native tool schemas (spawn/poll/list_subagent).

        Flat shape (``{"name","description","parameters"}``) — same as
        ``AGENT_TOOL_SCHEMAS`` — because provider clients expect that flat shape
        (DesignChatLoop passes registry schemas straight to ``chat_with_tools``).
        """
        return [
            {
                "name": "spawn_subagent",
                "description": (
                    "Delegate a self-contained coding task to a new sub-agent. "
                    "The sub-agent runs in the background and you receive its "
                    "agent_id immediately (non-blocking). Use this to parallelize "
                    "work across independent sub-tasks."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_description": {
                            "type": "string",
                            "description": "Complete, precise description of what the sub-agent must do — goal, constraints, and which files to modify.",
                        },
                        "title": {
                            "type": "string",
                            "description": "Short title (<=60 chars) summarizing the sub-task.",
                        },
                        "assigned_files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Repo-relative file paths the sub-agent should focus on / modify.",
                        },
                        "priority": {
                            "type": "integer",
                            "description": "0=highest (run first), 1=normal, 2=lowest. Default 1.",
                            "default": 1,
                        },
                    },
                    "required": ["task_description", "title"],
                },
            },
            {
                "name": "poll_subagent",
                "description": (
                    "Check whether a previously-spawned sub-agent has finished and "
                    "retrieve its result. If still running, returns a status update — "
                    "poll again later."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "description": "The agent_id returned by spawn_subagent. Mutually exclusive with agent_ids.",
                        },
                        "agent_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of agent_ids to wait for. Returns when ANY ONE completes (\"wait-any\"). Mutually exclusive with agent_id.",
                        },
                        "timeout_s": {
                            "type": "number",
                            "description": "How long to wait before returning 'still running'. Default 0 (non-blocking).",
                            "default": 0,
                        },
                    },
                },
            },
            {
                "name": "list_subagents",
                "description": "List all sub-agents spawned in this orchestration with their current status.",
                "parameters": {"type": "object", "properties": {}},
            },
        ]

    def _dispatch_native_tool(self, name: str, args: Any) -> str:
        """Dispatch one of the 3 orchestrator-native tools; return its result.

        Used by ``_OrchestratorBackedRegistry.dispatch`` so that a DesignChatLoop
        can execute spawn/poll/list_subagent through the normal registry
        ``dispatch(name, args)`` interface.

        Raises ``_NativeToolError`` for expected/handler-level errors (which
        ``dispatch`` maps to ``ok=False``); unexpected exceptions propagate to
        ``dispatch``'s generic handler (which logs + maps to ``ok=False``).
        """
        _args = args if isinstance(args, dict) else {}
        if name == "spawn_subagent":
            return self._tool_spawn_subagent(_args, self._current_request)
        if name == "poll_subagent":
            return self._tool_poll_subagent(_args)
        if name == "list_subagents":
            return self._tool_list_subagents()
        raise _NativeToolError(f"unknown orchestrator tool '{name}'.")
    # ── Background sub-agent runner ────────────────────────────────────────
    def _ensure_bg_executor(self) -> ThreadPoolExecutor:
        """Lazily create the background sub-agent thread pool."""
        if self._bg_executor is None:
            _n = max(1, self.orch_config.max_subagents)
            self._bg_executor = ThreadPoolExecutor(
                max_workers=_n, thread_name_prefix="orch-bg",
            )
        return self._bg_executor
    def _run_subagent_background(
        self,
        subtask: SubTaskSpec,
        task_text: Optional[str] = None,
        original_request: str = "",
    ) -> str:
        """Launch a sub-agent in the background; return its agent_id immediately."""
        pool = self._ensure_bg_executor()
        agent_id = subtask.task_id
        def _bg_job() -> Any:
            try:
                return self._run_subagent(
                    subtask, task_text=task_text, original_request=original_request,
                )
            except Exception as exc:
                logger.exception("Background SubAgent %s failed", agent_id)
                from .agent_loop import AgentResult
                return AgentResult(
                    status="error",
                    final_message=f"Sub-agent crashed: {exc}",
                    error=str(exc),
                )
        future = pool.submit(_bg_job)
        # NOTE: "queued vs running" is NOT stored here — a flag captured at
        # submit time goes stale the moment a slot frees up and the task starts
        # (poll would keep reporting "queued" for an actively-running task).
        # Readers compute it live from the Future instead: see _future_is_queued.
        with self._bg_lock:
            self._bg_subagents[agent_id] = {
                "subtask": subtask,
                "future": future,
                "result": None,
                "status": "running",
                "started_at": time.monotonic(),
            }
        # NOTE: do NOT emit subagent_start here — _run_subagent() (in-process)
        # or _run_subagent_ipc() emits it from inside the background thread.
        # Emitting here too would double-count in the spinner progress UI.
        return agent_id
    @staticmethod
    def _future_is_queued(entry: dict) -> bool:
        """Live check: is this sub-agent still waiting for a thread-pool slot?

        Computed from the Future at READ time, never stored — a snapshot taken
        at submit goes stale the moment a slot frees up and the task starts
        (B6 / turn 13118). ``running() is False and done() is False`` is
        exactly "sitting in the executor queue".
        """
        _fut = entry.get("future")
        if _fut is None:
            return False
        return not _fut.running() and not _fut.done()

    def _check_bg_subagent(
        self, agent_id: str, timeout_s: float = 0.0,
    ) -> tuple[str, Any]:
        """Check a background sub-agent; return (status, result_or_None).

        The returned ``status`` is the sub-agent's real AgentResult status once
        it has resolved (``"success"``/``"already_satisfied"``/``"max_turns"``/
        ``"error"``), or ``"running"``/``"unknown"`` before that.

        On FIRST completion the result is cached and appended to ``_bg_results``.
        Subsequent calls return the cached result WITHOUT re-appending — this is
        what prevents a successfully-completed sub-agent from being counted twice
        (once when ``poll_subagent`` collects it, again when
        ``_drain_background_subagents`` runs in the ``finally`` block).  The guard
        keys off ``entry["result"] is not None`` (set exactly once at completion,
        never cleared) rather than the status string, because the stored status is
        the AgentResult's real value (e.g. ``"success"``) — an earlier guard that
        checked ``status in ("completed", "error")`` never matched the success
        family and let every successful sub-agent be double-counted.  The guard
        is checked TWICE: once unlocked as a fast path (line ~2317), and again
        INSIDE ``_bg_lock`` at the append site — the unlocked check alone was a
        TOCTOU (two concurrent callers could both pass it before either set the
        result), so the lock-internal re-check is what actually guarantees the
        "appended exactly once" invariant under concurrent access.
        """
        with self._bg_lock:
            entry = self._bg_subagents.get(agent_id)
        if entry is None:
            return "unknown", None
        future: concurrent.futures.Future = entry["future"]
        if entry["result"] is not None:
            return entry["status"], entry["result"]
        if timeout_s and timeout_s > 0:
            try:
                future.result(timeout=timeout_s)
            except concurrent.futures.TimeoutError:
                return "running", None
            except Exception:
                pass  # handled below
        if future.done():
            try:
                result = future.result()
            except Exception as exc:
                logger.exception("Background SubAgent %s raised", agent_id)
                from .agent_loop import AgentResult
                result = AgentResult(
                    status="error",
                    final_message=f"Sub-agent failed: {exc}",
                    error=str(exc),
                )
            status = ("error" if not result or result.status == "error"
                      else getattr(result, "status", "completed"))
            with self._bg_lock:
                # Re-check inside the lock: the unlocked guard above (L2317) is a
                # fast path, but two callers can both observe result is None,
                # both see future.done(), and both reach here — a TOCTOU that
                # would append the SAME result twice into _bg_results (double-
                # counting in final synthesis/status).  Once set the entry is
                # frozen, so the losing caller just returns the cached value.
                # This is what actually makes the docstring's "counted once"
                # invariant hold under concurrent poll_subagent / drain callers.
                if entry["result"] is not None:
                    return entry["status"], entry["result"]
                entry["status"] = status
                entry["result"] = result
                self._bg_results.append(result)
            # NOTE: do NOT emit subagent_complete here — _run_subagent()
            # (in-process) already emitted it from the background thread.
            # Re-emitting would double-increment the spinner's done-counter.
            return status, result
        return "running", None
    # ── Orchestrator-native tool handlers ──────────────────────────────────
    def _tool_spawn_subagent(self, args: dict[str, Any], original_request: str) -> str:
        """spawn_subagent tool handler.

        Raises ``_NativeToolError`` on bad args (mapped to ``ok=False`` by
        ``dispatch``); returns a status string on success.
        """
        _desc = (args.get("task_description") or "").strip()
        if not _desc:
            raise _NativeToolError("'task_description' is required.")
        _title = (args.get("title") or "").strip() or (_desc[:55] + ("…" if len(_desc) > 55 else ""))
        _files = args.get("assigned_files") or []
        if isinstance(_files, str):
            _files = [_files]
        _files = _expand_directory_assignments(
            getattr(self._registry_proto, "repo_root", None), list(_files),
        )
        try:
            _priority = int(args.get("priority", 1))
        except (TypeError, ValueError):
            _priority = 1
        with self._bg_lock:
            self._bg_counter += 1
            _idx = self._bg_counter
        agent_id = f"dev_{_idx}"
        subtask = SubTaskSpec(
            task_id=agent_id,
            title=_title,
            description=_desc,
            assigned_files=list(_files),
            priority=_priority,
            preferred_lane="main_agent",
        )

        # File-conflict guard for tool-loop spawns. The dependency-aware path
        # serializes same-file tasks via _split_batch_by_file_conflict, but
        # tool-loop mode relies on the LLM to assign disjoint files. Warn
        # (don't block) when this sub-agent's assigned_files overlap a
        # *currently running* bg sub-agent: FileLockManager serializes the
        # writes but does NOT force the second agent to re-read, so a lost
        # update is otherwise silent. Surfacing the overlap lets the LLM
        # sequence via dependencies or reassign files.
        _new_files = {_norm_assigned_file(f) for f in _files if f}
        _conflict_note = ""
        if _new_files:
            _conflict_ids: list[str] = []
            with self._bg_lock:
                for _aid, _entry in self._bg_subagents.items():
                    if _entry.get("result") is not None:
                        continue  # finished — no longer concurrent
                    _their = {
                        _norm_assigned_file(f)
                        for f in getattr(_entry.get("subtask"), "assigned_files", []) or []
                        if f
                    }
                    if _their & _new_files:
                        _conflict_ids.append(_aid)
            if _conflict_ids:
                self._cb("orchestrator_warning", {
                    "type": "tool_loop_file_conflict",
                    "agent_id": agent_id,
                    "conflicting_agents": _conflict_ids,
                    "files": sorted(_new_files),
                    "message": (
                        f"Sub-agent '{agent_id}' shares files with still-running "
                        f"sub-agent(s) {_conflict_ids}; sequence via dependencies "
                        f"to avoid a lost update."
                    ),
                })
                _conflict_note = (
                    f"\n⚠ File overlap with running sub-agent(s) "
                    f"{', '.join(_conflict_ids)} — to avoid one overwriting the "
                    f"other's changes, poll them first or assign disjoint files."
                )

        # Hook (b) for B5 scope-violation signal hygiene (findings 2+3):
        # accumulate this subtask's assigned_files into the global assignment
        # union so a PARALLEL peer's completion can subtract THIS subtask's
        # in-flight edits (git status is global — peer-2 sees peer-1's edit to
        # peer-1's OWN file as "out of scope for peer-2"). In decomposition mode
        # the union is known upfront (set once via _capture_scope_baseline);
        # tool-loop spawns dynamically, so each spawn adds its files here.
        #
        # Normalization MUST be os.path.normpath to match
        # _filter_unassigned_changes' comparison key — NOT _norm_assigned_file
        # (used by the conflict check above for its own internally-consistent
        # comparison). Guarded by _bg_lock: spawn_subagent runs in the
        # orchestrator's main thread while IPC-completion threads read this set
        # concurrently from _filter_unassigned_changes.
        _scope_union_files = {os.path.normpath(_f) for _f in _files if _f}
        if _scope_union_files:
            with self._bg_lock:
                self._global_assigned_paths |= _scope_union_files
        self._run_subagent_background(subtask, original_request=original_request)
        _queued_note = ""
        with self._bg_lock:
            _entry = self._bg_subagents.get(agent_id, {})
        if self._future_is_queued(_entry):
            _queued_note = "\n⚠ Queued (waiting for a free worker slot). It will start automatically when one becomes available."
        _mode_hint = (
            "A new Terminal window will open shortly for this sub-agent — you "
            "can watch it work there."
            if self.orch_config.subagent_mode == "ipc" and self.auto_launch_terminal
            else "It runs in the background."
        )
        return (
            f"Sub-agent '{agent_id}' spawned: {_title}\n"
            f"Files: {', '.join(_files) if _files else '(none specified)'}\n"
            f"{_mode_hint}{_conflict_note}{_queued_note}\n"
            f"Use poll_subagent(agent_id=\"{agent_id}\") to collect its result. "
            f"Non-blocking — spawn more sub-agents or inspect the repo meanwhile."
        )
    @staticmethod
    def _format_poll_patches(patches: list) -> str:
        """Render applied patches for the poll_subagent tool output.

        Surfaces actual file names (not just a count) so the orchestrator can
        reason about file overlap with siblings and check the narration against
        the reported scope at decision time.  Handles the patch shapes produced
        across modes: ``{"file": ...}`` dicts (IPC), objects with ``file_path``
        (in-process structured), and raw patch-text strings (legacy in-process).
        Names that cannot be extracted fall back to a bare count.
        """
        if not patches:
            return ""
        _cap = 8
        files: list[str] = []
        for _p in patches:
            if isinstance(_p, dict):
                _f = _p.get("file") or _p.get("file_path", "")
            elif hasattr(_p, "file_path"):
                _f = getattr(_p, "file_path", "")
            else:
                _f = ""
            if _f and _f not in files:
                files.append(_f)
        if files:
            _shown = files[:_cap]
            _more = len(files) - len(_shown)
            _tail = f" (+{_more} more)" if _more > 0 else ""
            return f"\nApplied patches ({len(files)}): {', '.join(_shown)}{_tail}"
        return f"\nApplied patches: {len(patches)}"

    def _tool_poll_subagent(self, args: dict[str, Any]) -> str:
        """poll_subagent tool handler.

        Supports two modes:
        * ``agent_id`` (str) — poll a single sub-agent (existing behavior).
        * ``agent_ids`` (list[str]) — wait for ANY ONE of the listed agents to
          complete (``concurrent.futures.wait(FIRST_COMPLETED)``). Returns the
          first completed agent's result.

        ``timeout_s`` semantics are identical in both modes (so behavior is
        predictable from the arguments): ``timeout_s=0`` (or omitted) is a
        NON-BLOCKING poll — already-finished agents are returned, else a "still
        running / none done" status comes back immediately; ``timeout_s>0``
        blocks up to that many seconds (capped at ``ipc_timeout_s``) for the
        first completion. A blocking poll can therefore never hang the tool loop
        on a worker whose liveness probe is missed.

        Raises ``_NativeToolError`` on bad args / unknown id (mapped to
        ``ok=False`` by ``dispatch``); returns a status string otherwise.
        """
        agent_id = (args.get("agent_id") or "").strip()
        agent_ids_raw = args.get("agent_ids")
        if not agent_id and not agent_ids_raw:
            raise _NativeToolError("One of 'agent_id' or 'agent_ids' is required.")
        if agent_id and agent_ids_raw:
            raise _NativeToolError("Use either 'agent_id' (single) or 'agent_ids' (multi), not both.")
        try:
            timeout_s = float(args.get("timeout_s", 0) or 0)
        except (TypeError, ValueError):
            timeout_s = 0.0

        if agent_id:
            return self._poll_single_agent(agent_id, timeout_s)
        return self._poll_any_agent(list(agent_ids_raw), timeout_s)

    def _poll_single_agent(self, agent_id: str, timeout_s: float) -> str:
        """Poll a single sub-agent and return a status/result string."""
        status, result = self._check_bg_subagent(agent_id, timeout_s=timeout_s)
        if status == "unknown":
            raise _NativeToolError(
                f"no sub-agent with id '{agent_id}'. Use list_subagents to see active ids."
            )
        if status == "running":
            with self._bg_lock:
                entry = self._bg_subagents.get(agent_id, {})
            _el = time.monotonic() - entry.get("started_at", time.monotonic())
            # Distinguish queued (waiting for a thread-pool slot) from actively
            # running, so the orchestrator LLM doesn't misinterpret "still
            # running" as "actively computing" when it's merely waiting in the
            # queue (B6 / turn 13118). Computed LIVE from the Future — a flag
            # stored at submit time would go stale once the task starts.
            if self._future_is_queued(entry):
                return (
                    f"Sub-agent '{agent_id}' is queued (waiting for a free slot, "
                    f"{_el:.0f}s since submit). Poll again later."
                )
            return (
                f"Sub-agent '{agent_id}' is still running ({_el:.0f}s elapsed). "
                f"Poll again later."
            )
        return self._format_agent_result(agent_id, status, result)

    def _gather_done_futures(
        self,
        known_futures: list[tuple[str, Any]],
    ) -> tuple[list[tuple[str, str, Any]], list[str]]:
        """Collect results for every already-resolved future WITHOUT waiting.

        Returns ``(completed, still_pending)``. ``completed`` holds
        ``(aid, status, result)`` for each future whose ``done()`` is True,
        gathered via ``_check_bg_subagent(timeout_s=0)`` — the single path that
        caches ``entry["result"]`` and appends to ``_bg_results`` (so each
        completion is counted exactly once). ``still_pending`` lists the ids
        whose futures have NOT resolved yet.

        This exists because ``entry["result"]`` is filled LAZILY at collection
        time, NOT by the background thread: a resolved-but-never-polled future
        still reports ``entry["result"] is None``. Callers that branch on that
        field alone therefore misclassify finished agents as still running.
        """
        completed: list[tuple[str, str, Any]] = []
        still_pending: list[str] = []
        for _aid, _fut in known_futures:
            if _fut.done():
                _status, _result = self._check_bg_subagent(_aid, timeout_s=0)
                completed.append((_aid, _status, _result))
            else:
                still_pending.append(_aid)
        return completed, still_pending

    def _poll_any_agent(self, agent_ids: list[str], timeout_s: float) -> str:
        """Wait for ANY of the listed agents, returning ALL that have completed.

        Timeout semantics are deliberately SYMMETRIC with ``_poll_single_agent``
        so the orchestrator LLM can predict the tool's behavior from its
        arguments alone — the same ``timeout_s=0`` means "non-blocking poll" in
        both modes:

        * ``timeout_s <= 0`` — NON-BLOCKING. Gather every already-finished
          agent and return immediately; if none are done, return a "none done"
          status. A hung worker (one whose liveness probe is missed) can never
          lock the tool loop: there is no wait to get stuck in.
        * ``timeout_s > 0`` — Block up to ``min(timeout_s, ipc_timeout_s)`` in
          cancel-responsive 2s slices (``FIRST_COMPLETED``), then return every
          agent that finished within the window. The ``ipc_timeout_s`` cap is
          the safety net: even an explicit long ``timeout_s`` (or a future
          caller that forgets cancel) cannot hold the loop indefinitely.

        BOTH branches first gather any future that has already resolved via
        ``_gather_done_futures``. This is essential because ``entry["result"]``
        is set lazily by ``_check_bg_subagent`` at collection time, NOT by the
        background thread — a finished-but-uncollected agent would otherwise be
        invisible to a pure ``entry["result"]`` check, so the non-blocking
        branch would forever report "none done" for an agent that is complete.

        Raises ``_NativeToolError`` on an unknown id (mapped to ``ok=False``).
        """
        import concurrent.futures as _cf

        if not agent_ids:
            raise _NativeToolError("'agent_ids' must be a non-empty list.")

        # Deduplicate agent_ids to prevent double-counting if the LLM passes
        # the same id twice (e.g. agent_ids=["dev_1","dev_1"]). Order preserved.
        if len(agent_ids) != len(set(agent_ids)):
            agent_ids = list(dict.fromkeys(agent_ids))

        # Resolve entries; unknown ids raise immediately. Separate agents whose
        # result is ALREADY collected (cached) from still-open futures.
        known_futures: list[tuple[str, _cf.Future]] = []
        done: list[tuple[str, str, Any]] = []  # (aid, status, result)
        for _aid in agent_ids:
            with self._bg_lock:
                _entry = self._bg_subagents.get(_aid)
            if _entry is None:
                raise _NativeToolError(
                    f"no sub-agent with id '{_aid}'. Use list_subagents to see active ids."
                )
            if _entry["result"] is not None:
                done.append((_aid, _entry["status"], _entry["result"]))
            else:
                known_futures.append((_aid, _entry["future"]))

        # Gather any future that has already resolved but has not yet been
        # polled/drained (entry["result"] still None). Without this, the
        # non-blocking branch below would return "none done" for an agent that
        # is in fact finished, and the cached fast-path return would list a
        # finished agent as "still running".
        _completed, still_running = self._gather_done_futures(known_futures)
        done.extend(_completed)

        # Anything completed (cached OR just-gathered) returns immediately, with
        # any genuinely-still-running agents noted. No wait. (agent_ids is
        # non-empty here, so if `done` is empty `known_futures` cannot be.)
        if done:
            return self._format_poll_any_results(
                agent_ids, done, pending=still_running,
            )

        # Nothing has finished yet.
        if timeout_s is None or timeout_s <= 0:
            # Non-blocking: report none done immediately. There is no wait here,
            # so a hung worker can never lock the tool loop.
            _first_aid = known_futures[0][0]
            with self._bg_lock:
                _entry = self._bg_subagents.get(_first_aid, {})
            _el = time.monotonic() - _entry.get("started_at", time.monotonic())
            return (
                f"None of {len(agent_ids)} sub-agents have completed yet "
                f"(timeout_s=0 → non-blocking). {len(still_running)} still "
                f"running (first {_first_aid}: {_el:.0f}s elapsed). "
                f"Poll again later."
            )

        # Blocking mode: cap the wait at ipc_timeout_s so a silently-hung worker
        # cannot hold the tool loop forever even without an explicit cancel.
        _ce = self.orch_config.cancel_event
        _cap = self.orch_config.ipc_timeout_s or timeout_s
        _deadline = time.monotonic() + min(timeout_s, _cap)
        _waited = min(timeout_s, _cap)
        _futures = [f for _, f in known_futures]
        while True:
            if _ce is not None and _ce.is_set():
                return (
                    f"Wait on {len(agent_ids)} sub-agents interrupted by "
                    f"cancellation; none collected."
                )
            _left = _deadline - time.monotonic()
            if _left <= 0:
                break
            _slice = min(2.0, _left)
            _done_set, _ = _cf.wait(
                _futures, timeout=_slice, return_when=_cf.FIRST_COMPLETED,
            )
            if _done_set:
                break

        # Re-gather after the wait — futures may have resolved during the window.
        completed, _still_pending = self._gather_done_futures(known_futures)

        if not completed:
            # Window elapsed with nothing done — report the first as still running.
            _first_aid = known_futures[0][0]
            with self._bg_lock:
                _entry = self._bg_subagents.get(_first_aid, {})
            _el = time.monotonic() - _entry.get("started_at", time.monotonic())
            return (
                f"None of {len(agent_ids)} sub-agents completed within "
                f"{_waited:.0f}s. First ({_first_aid}): still running "
                f"({_el:.0f}s elapsed). Poll again later."
            )

        return self._format_poll_any_results(
            agent_ids, completed, pending=_still_pending,
        )

    def _format_poll_any_results(
        self,
        agent_ids: list[str],
        completed: list[tuple[str, str, Any]],
        pending: list[str] | None = None,
    ) -> str:
        """Format one-or-more completed sub-agent results for the poll response.

        When exactly one agent finished and nothing else is still running, this
        returns that agent's result verbatim (identical to the single-completion
        contract the orchestrator already expects). Otherwise it prefixes a
        short summary line so simultaneous completions (#6) and pending agents
        are visible in a single response.
        """
        pending = pending or []
        _total = len(agent_ids)
        if len(completed) == 1 and not pending:
            _aid, _status, _result = completed[0]
            return self._format_agent_result(_aid, _status, _result)
        parts = [
            self._format_agent_result(_aid, _status, _result)
            for _aid, _status, _result in completed
        ]
        _head = f"{len(completed)} of {_total} sub-agents completed"
        if pending:
            _head += f", {len(pending)} still running"
        _head += ":\n\n"
        return _head + "\n\n".join(parts)

    def _format_agent_result(self, agent_id: str, status: str, result: Any) -> str:
        """Format a completed sub-agent result for the poll response."""
        if result is None:
            return f"Sub-agent '{agent_id}' status={status} (no result captured)."
        _msg = (getattr(result, "final_message", "") or "").strip()
        _st = getattr(result, "status", status)
        _patches = getattr(result, "applied_patches", None) or []
        _patch_note = self._format_poll_patches(_patches)
        _verdict = (getattr(result, "_orch_diff_verdict", "") or "").strip()
        _verdict_note = f"\nDiff verification: {_verdict}" if _verdict else ""
        _unassigned = getattr(result, "_orch_unassigned", None) or []
        _scope_note = ""
        if _unassigned:
            _files = [p.get("file") for p in _unassigned if isinstance(p, dict)]
            _scope_note = (
                "\n⚠ Out-of-scope changes: "
                + ", ".join(f for f in _files if f)
                + " — these were NOT in the assigned files; consider reverting."
            )
        if len(_msg) > _POLL_RESULT_CAP:
            _msg = f"{_msg[:_POLL_RESULT_CAP]}\n[…truncated {len(_msg) - _POLL_RESULT_CAP} chars]"
        return (
            f"Sub-agent '{agent_id}' finished — status: {_st}.\n"
            f"Result:\n{_msg}{_patch_note}{_verdict_note}{_scope_note}"
        )
    def _tool_list_subagents(self) -> str:
        """list_subagents tool handler."""
        with self._bg_lock:
            items = [
                (aid, e.get("status", "?"), e.get("subtask"))
                for aid, e in self._bg_subagents.items()
            ]
        if not items:
            return "No sub-agents spawned yet."
        lines = []
        for aid, status, st in items:
            _title = getattr(st, "title", "") if st else ""
            lines.append(f"- {aid} [{status}]: {_title}")
        return "Sub-agents:\n" + "\n".join(lines)
    # ── Main tool loop ─────────────────────────────────────────────────────
    def _run_tool_loop(self, request: str) -> OrchestratorResult:
        """Drive a real ``DesignChatLoop`` AS the orchestrator.

        Instead of a bespoke chat_with_tools loop, the orchestrator reuses the
        exact same battle-tested DesignChatLoop the interactive REPL uses — with
        the FULL design-chat tool set (read + write + edit + analysis + web) PLUS
        the three orchestrator-native tools (spawn/poll/list_subagent) injected
        via :class:`_OrchestratorBackedRegistry`.

        Consequences (this is what makes the orchestrator no longer "just wait"):
        * The orchestrator can do real work itself (read/edit/grep/bash) while
          sub-agents run in the background — it is never blocked on a single
          sub-agent.
        * DesignChatLoop's plan-completion gate prevents premature termination,
          so the orchestrator verifies results and iterates instead of
          summarising the instant a sub-agent finishes.
        * The orchestrator's own tool calls render through the normal design-chat
          UI (✓ lines) via the ``design_stream_callback``.
        """
        from .design_chat_loop import DesignChatLoop
        from .agent_loop_types import AgentCancelled
        from ..client import LLMMessage

        self._current_request = request
        self._file_lock_mgr.reset()
        self._shared_memory = []
        self._bg_subagents.clear()
        self._bg_results = []
        self._ipc_worker_ids.clear()
        self._ipc_worker_procs.clear()
        # Same rationale as run(): the previous run's _cleanup_ipc_workers()
        # terminated the workers backing these ids; keep them cleared so the next
        # subtask cannot optimistically reuse a dead worker whose Popen handle
        # was dropped above (defeats the liveness guard).
        self._reusable_worker_ids.clear()
        # Hook (a) for B5 scope-violation signal hygiene (findings 2+3): snapshot
        # the pre-run dirty set BEFORE the orchestrator (or any sub-agent) edits.
        # ``_filter_unassigned_changes`` subtracts this from each worker's raw
        # GLOBAL ``git status`` report so pre-existing dirt (.env, etc.) the worker
        # did NOT create isn't mistaken for a scope violation. The peer-assignment
        # union (hook b) is accumulated incrementally in ``_tool_spawn_subagent``
        # because tool-loop spawns dynamically — there is no upfront decomposition.
        # (run() reset both to empty at entry; decomposition mode captures the full
        # baseline + union once via _capture_scope_baseline instead.) Without this,
        # the tool-loop path — the PRIMARY mode for --orchestrate and the REPL —
        # never captured the baseline, so the filter degraded to infra-only and
        # .env / peer in-flight edits leaked into unassigned_changes as false
        # positives (turn 13108 verification).
        _rr = getattr(self._registry_proto, "repo_root", None) or "."
        self._baseline_dirty_paths = _snapshot_dirty_path_set(_rr)
        self._cb("orchestrator_plan", {"mode": "tool_loop", "subtasks": []})

        wrapped_registry = _OrchestratorBackedRegistry(self._registry_proto, self)
        # Inject the orchestrator's process-wide FileLockManager so the orchestrator's
        # OWN writes (edit_text / apply_patch / write_plan / …) get per-file locking —
        # the SAME instance the sub-agent registries are given (see file_lock_manager=
        # self._file_lock_mgr in the sub-agent clone path), so orchestrator + sub-agents
        # coordinate on locks instead of racing on shared files.  Without this the
        # tool-loop config's file_lock_manager was None and dispatch() skipped locking
        # entirely (flm is None guard).
        if getattr(wrapped_registry.config, "file_lock_manager", None) is None:
            wrapped_registry.config.file_lock_manager = self._file_lock_mgr
        loop = DesignChatLoop(
            self.llm_client, wrapped_registry, self.model,
            run_store=self._run_store,
        )
        # ── Session context inheritance ──────────────────────────────────────
        # The orchestrator now inherits the SAME design-chat session context the
        # interactive REPL builds: repo root, project.md, design insights, the
        # compressed conversation summary, and prior turns (orchestrator tasks +
        # design-chat exchanges alike — they share one session).  This makes
        # orchestrator mode continuous with design-chat: a task's result and any
        # earlier design-chat decisions are visible to the orchestrator and to
        # the next design-chat turn.
        #
        # build_context_messages(skip_core_prompt=True) yields the context blocks
        # WITHOUT the core identity prompt, so _ORCH_TOOL_LOOP_SYSTEM fills that
        # slot without duplication.  The REPL adds the user task turn BEFORE
        # calling run(), so build_context_messages already ends with that turn —
        # we must NOT append request again (would duplicate it).  Only when there
        # is no session_mgr (legacy/webapp fallback, no pre-added user turn) do we
        # append request as the sole user message.
        msgs: list[LLMMessage] = [LLMMessage(role="system", content=self._ORCH_TOOL_LOOP_SYSTEM)]
        _session_mgr = getattr(self.orch_config, "session_mgr", None)
        _session_id = getattr(self, "_session_id", "") or ""
        _ctx_inherited = False
        if _session_mgr is not None and _session_id:
            try:
                _ds = _session_mgr.get_or_create(_session_id)
                _ctx = _session_mgr.build_context_messages(_ds, current_model=self.model or "", skip_core_prompt=True, mode="code")
                # Filter out empty system dividers that add no value; keep the rest
                # (repo / project / insights / summary / turns, incl. the request).
                for _m in _ctx:
                    _mc = _m.get("content", "")
                    if _mc and _mc.strip() != "──":
                        msgs.append(LLMMessage(role=_m["role"], content=_mc))
                _ctx_inherited = True
            except Exception:
                # Context inheritance is best-effort — never block the orchestrator.
                # Fall back to a bare request (prior behavior).
                pass
        if not _ctx_inherited:
            # No session context (no session_mgr / session_id, or the build failed):
            # append request as the sole user message — preserves the legacy contract.
            msgs.append(LLMMessage(role="user", content=request))
        dc_result: Any = None
        _cancelled = False
        try:
            dc_result = loop.respond(
                msgs,
                stream_callback=self._design_cb,
                max_tool_iterations=max(1, int(self.orch_config.tool_loop_max_iterations)),
                mode="code",
                thinking_mode=self.orch_config.thinking_mode,
                reasoning_effort=self.orch_config.reasoning_effort,
                # Forward the DesignSessionManager so search_design_history works in
                # orchestrate mode — same capability the interactive design-chat has.
                session_mgr=getattr(self.orch_config, "session_mgr", None),
            )
        except AgentCancelled:
            # ESC / Ctrl-C — DesignChatLoop honored the cancel_event between
            # iterations. Collect partial results and return a cancelled result.
            _cancelled = True
        finally:
            # Collect any still-running background sub-agents before returning so
            # their results are reflected in the final OrchestratorResult.
            # (Cancel-aware: bails fast when cancel_event is set.)
            # Derive drain timeout from config: sub-task runs + retries + buffer.
            # Hard-coded 600s would kill legitimate long-running tasks when the user
            # raised ipc_timeout_s or uses ipc_max_timeout_s + review_max_retries (B6).
            _ipc_base = max(self.orch_config.ipc_timeout_s, self.orch_config.ipc_max_timeout_s or 0)
            _drain_timeout = max(600.0, _ipc_base * (1 + self.review_max_retries) + 60.0)
            self._drain_background_subagents(per_agent_timeout=_drain_timeout)
            self._shutdown_bg_executor()
            # Write shutdown sentinels for any remaining IPC workers
            # (the workers poll for this file and exit their loop).
            self._cleanup_ipc_workers()

        if _cancelled:
            _sum = self._synthesize_from_subtasks(self._paired_subtask_results())
            self._cb("orchestrator_done", {
                "status": "cancelled", "summary": _sum, "total_turns": 0,
            })
            return OrchestratorResult(
                status="cancelled",
                # Mirror the synthesised summary emitted in the event above (it
                # folds in each sub-agent's partial result). Fall back to the
                # generic string only when no sub-agent produced a summary — so
                # the event consumers and the return-value consumers see the SAME
                # content instead of diverging.
                summary=_sum or "Orchestration was cancelled.",
                subtask_results=list(self._bg_results),
                metadata={"mode": "tool_loop", "cancelled": True},
            )

        results = list(self._bg_results)
        subtasks = [
            self._bg_subagents[a]["subtask"]
            for a in self._bg_subagents
            if self._bg_subagents[a].get("subtask")
        ]
        total_turns = sum(
            len(getattr(r, "turns", []) or []) for r in results if r is not None
        )
        any_ok = any(
            getattr(r, "status", "") in ("success", "already_satisfied", "max_turns")
            for r in results if r is not None
        )
        _dc_content = (getattr(dc_result, "content", "") or "").strip() if dc_result else ""
        _dc_is_error = bool(getattr(dc_result, "is_error", False)) if dc_result else False
        if any_ok:
            status = "success"
        elif results:
            # Sub-agents ran but none succeeded.
            status = "partial"
        elif _dc_content and not _dc_is_error:
            # Orchestrator answered directly without spawning sub-agents — e.g.,
            # asked the user a clarifying question, or handled the request itself
            # via its own read/edit tools.  This is a legitimate turn completion,
            # NOT an error (a bare "no sub-agents" used to wrongly map to error,
            # surfacing a misleading `status: error` for a perfectly good answer).
            status = "success"
        else:
            status = "error"
        summary = _dc_content or self._synthesize_from_subtasks(self._paired_subtask_results())
        self._cb("orchestrator_done", {
            "status": status, "summary": summary, "total_turns": total_turns,
        })
        return OrchestratorResult(
            status=status, summary=summary, subtask_results=results,
            total_turns=total_turns,
            metadata={"mode": "tool_loop", "subagents": len(subtasks)},
        )
    def _drain_background_subagents(self, per_agent_timeout: float = 600.0) -> None:
        """Block until every background sub-agent resolves (or its timeout).

        Cancel-aware: if ``orch_config.cancel_event`` is set, stops waiting
        immediately and returns whatever results have arrived — so a Ctrl-C
        doesn't hang for the full per-agent timeout on each running sub-agent.

        Polls round-robin (not sequentially per-agent): each sweep advances
        every still-running agent by one poll step, so a fast-finishing agent
        is observed promptly even when an earlier agent is slow.  Per-agent
        timeout budget is tracked individually, preserving the original
        semantics.  Correctness is unaffected (results are cached); only
        wall-clock latency improves.
        """
        _ce = self.orch_config.cancel_event
        _poll_step = 2.0  # poll interval — keeps cancel responsive
        with self._bg_lock:
            aids = list(self._bg_subagents.keys())
        if not aids:
            return
        _remaining = {aid: per_agent_timeout for aid in aids}
        while True:
            if _ce is not None and _ce.is_set():
                return  # cancelled — don't block further
            # Agents with budget left that we still need to wait on.
            active = [aid for aid in aids if _remaining[aid] > 0]
            if not active:
                return
            for aid in active:
                _step = min(_poll_step, _remaining[aid])
                _status, _ = self._check_bg_subagent(aid, timeout_s=_step)
                _remaining[aid] -= _step
                if _status != "running":
                    _remaining[aid] = 0  # resolved — stop polling this agent
    def _shutdown_bg_executor(self) -> None:
        """Tear down the background sub-agent thread pool."""
        ex = self._bg_executor
        if ex is None:
            return
        self._bg_executor = None
        try:
            ex.shutdown(wait=False)
        except Exception:
            pass
    def _cleanup_ipc_workers(self) -> None:
        """Write cancel + shutdown sentinels for all IPC workers spawned this run.

        Two sentinels, both fire-once:

        * ``cancel.json``  — abort a task the worker is running RIGHT NOW.  The
          worker's cancel watcher thread polls this DURING execution and sets
          the local ``cancel_event``; DesignChatLoop then aborts at its next
          turn boundary.  This is what actually stops a mid-task worker —
          ``shutdown.json`` alone is only honored BETWEEN tasks.
        * ``shutdown.json`` — exit the poll loop (honored after the in-flight
          task aborts).  Prevents terminal/process leaks once orchestration ends.

        Sentinels are written to ``_ipc_worker_ids`` (the worker poll dirs —
        ``--subagent-id``), which in reuse mode differ from the task_ids.  The
        prior code wrote to ``_subagent_ipc_commands`` keys (task_ids) and so
        never reached a reused worker.
        """
        if self.orch_config.subagent_mode != "ipc":
            return
        from .subagent_ipc import write_cancel_all, write_shutdown_all
        worker_ids = list(self._ipc_worker_ids)
        if worker_ids:
            repo_root = getattr(self._registry_proto, "repo_root", None) or "."
            # Cancel first so a mid-task worker aborts before we tell it to exit.
            write_cancel_all(repo_root, worker_ids)
            write_shutdown_all(repo_root, worker_ids)
            logger.info(
                "Orchestrator: wrote cancel+shutdown sentinels for %d IPC workers",
                len(worker_ids),
            )
        # Belt-and-braces for HEADLESS background workers (the macOS Terminal path
        # has no PID handle, but our background spawns do): after the shutdown
        # sentinel, terminate any still-alive tracked process so a missed/late
        # sentinel can't leave an orphan worker polling forever (timeout_s=None).
        with self._bg_lock:
            _procs = list(self._ipc_worker_procs.values())
            self._ipc_worker_procs.clear()
        for _p in _procs:
            try:
                if _p.poll() is None:
                    _p.terminate()
            except Exception:
                pass
        # All tracked workers are now terminated (or were already dead) — the
        # reusable-worker pool they backed is gone. Reset it here (the single
        # source of truth for worker death) so the next run() cannot reuse a
        # dead worker whose Popen handle was just dropped above.
        self._reusable_worker_ids.clear()

    def _claim_reusable_worker(self, repo_root: str) -> Optional[str]:
        """Atomically claim an idle, alive worker from the reuse pool (P3).

        Returns the worker_id, or ``None`` if no reusable worker is available.
        Thread-safe: parallel batches share the pool via ThreadPoolExecutor, so
        the claim removes the worker from the pool under ``_bg_lock`` — two
        threads can never dispatch to the same worker (the prior single-slot
        design was sequential-only and could not express this).

        A worker is reusable when its ``task.json`` is absent (between tasks) and
        its tracked ``Popen`` is still alive. A dead background worker is dropped
        from the pool; a Terminal-launched worker (no PID handle) is verified via
        its idle heartbeat (``worker.heartbeat.json`` in its poll dir) — a
        heartbeat that EXISTS but is stale (older than
        ``ipc_heartbeat_stale_s``) proves the worker is hung, so it is dropped
        rather than optimistically reused (which would burn the full
        ``ipc_timeout_s`` on the next dispatch). A missing heartbeat is NOT
        conclusive (legacy worker, clock skew) and falls back to optimistic reuse.
        """
        from .subagent_ipc import (
            _IDLE_HEARTBEAT_INTERVAL_S,
            read_worker_idle_heartbeat_age,
            read_worker_idle_heartbeat_state,
        )
        # Clamp the effective staleness threshold to at least 2x the idle
        # heartbeat write interval. An idle worker writes a fresh heartbeat
        # every _IDLE_HEARTBEAT_INTERVAL_S (30s); if ipc_heartbeat_stale_s is
        # misconfigured below that, EVERY idle worker looks "stale" the moment
        # it is between writes, and the pool churns/drops workers that are
        # perfectly alive.
        _effective_stale_s = max(
            self.orch_config.ipc_heartbeat_stale_s, 2 * _IDLE_HEARTBEAT_INTERVAL_S,
        )
        with self._bg_lock:
            candidates = list(self._reusable_worker_ids)
        for wid in candidates:
            _task_path = os.path.join(
                repo_root, ".asicode", "subagents", wid, "task.json",
            )
            if os.path.exists(_task_path):
                continue  # still busy (task.json present)
            # An explicit "exited" heartbeat means the worker process wrote it
            # right before terminating (SIGINT / shutdown.json) — a definitive,
            # age-independent death signal that closes the gap a plain staleness
            # check leaves open: the last "idle" heartbeat can be up to
            # ipc_heartbeat_stale_s fresh-looking for a while after the process
            # is actually gone. Drop it unconditionally, before either liveness
            # check below.
            try:
                if read_worker_idle_heartbeat_state(repo_root, wid) == "exited":
                    with self._bg_lock:
                        self._reusable_worker_ids.discard(wid)
                    logger.warning(
                        "Reusable worker %s reported exited; dropping from pool.",
                        wid,
                    )
                    continue
            except Exception:
                pass
            # task.json absent → between tasks. Verify liveness before reusing.
            # Read the tracked Popen under ``_bg_lock`` for protocol consistency:
            # every other access (spawn / cleanup / abandon) holds ``_bg_lock``, so
            # a bare dict.get here was the lone exception. GIL makes the single-key
            # read atomic, but keeping the access invariant uniform avoids a future
            # refactor trap (a non-atomic compound read would silently race).
            with self._bg_lock:
                _proc = self._ipc_worker_procs.get(wid)
            if _proc is not None and _proc.poll() is not None:
                with self._bg_lock:
                    self._reusable_worker_ids.discard(wid)
                logger.warning(
                    "Reusable worker %s exited (pid=%s); dropping from pool.",
                    wid, _proc.pid,
                )
                continue
            # Heartbeat liveness for Terminal-launched workers (no PID handle):
            # a heartbeat that EXISTS but is stale ⇒ the worker is hung (not
            # merely slow — it has not polled in >> ipc_heartbeat_stale_s). Drop
            # it so the next dispatch does not burn the full ipc_timeout_s. A
            # missing heartbeat (None) is inconclusive — fall back to the prior
            # optimistic reuse so legacy / pre-heartbeat workers keep working.
            if _proc is None:
                try:
                    _hb_age = read_worker_idle_heartbeat_age(repo_root, wid)
                except Exception:
                    _hb_age = None
                if _hb_age is not None and _hb_age > _effective_stale_s:
                    with self._bg_lock:
                        self._reusable_worker_ids.discard(wid)
                    logger.warning(
                        "Reusable worker %s idle heartbeat stale (%.0fs > "
                        "%.0fs); dropping (hung Terminal worker).",
                        wid, _hb_age, _effective_stale_s,
                    )
                    continue
            # Alive (or terminal-launched with no PID handle) → claim under lock.
            with self._bg_lock:
                if wid in self._reusable_worker_ids:
                    self._reusable_worker_ids.discard(wid)
                    return wid
                # Another thread claimed it between our snapshot and the lock.
        return None

    def _return_worker_to_pool(self, worker_id: str) -> None:
        """Return a worker to the reuse pool after its task completes (P3).

        The worker has finished (or aborted but stayed alive per B6) and is now
        idle, polling for the next task — making it eligible for reuse by a
        subsequent subtask. Thread-safe. Re-adding a dead worker is harmless:
        the liveness check in :meth:`_claim_reusable_worker` drops it, and the
        pool is cleared at run end regardless.
        """
        with self._bg_lock:
            self._reusable_worker_ids.add(worker_id)

    def _abandon_ipc_worker(
        self, repo_root: str, worker_id: str,
        revert_snapshots: Optional[dict] = None, *, grace_s: float = 12.0,
        task_id: str = "",
    ) -> bool:
        """Stop an IPC worker we have given up on (timeout OR cancel) and revert
        its partial edits — split-brain prevention.

        Once ``wait_for_result`` returns ``None`` we treat the subtask as failed,
        but the worker process may still be mid-task and writing to the very
        files whose pre-run snapshot we hold. We MUST, in order:

        1. **Always** tell it to abort. The worker is unreachable by PID on the
           macOS Terminal launch path, so a ``cancel.json`` sentinel is the only
           signal that reaches a mid-task worker (its cancel-watcher thread
           honors it at the next turn boundary).
        2. **Quiesce** the worker so in-flight writes finish before we touch the
           tree. Two regimes (B6):

           * **Hard** (orchestrator-level cancel — ``cancel_event`` is set): the
             whole run is ending, so wait for a tracked ``Popen`` to exit and
             ``terminate()`` it if it lingers past *grace_s*. ``_cleanup_ipc_workers``
             also writes ``shutdown.json``, so the worker exits cleanly.

           * **Soft** (pure timeout — ``cancel_event`` NOT set): the worker is
             merely slow, not cancelled. Poll for a fresh ``result.json`` in the
             task's dir (the worker writes one immediately after it observes the
             cancel sentinel and aborts) as the *quiescence* signal — its
             appearance means all writes are done and the worker is idle. We then
             restore the snapshot but do **NOT** terminate the worker, so it stays
             alive and reusable for the next subtask (saving the ~5-8s respawn
             cost). If ``result.json`` never appears within *grace_s* the worker is
             truly hung (did not honor the sentinel) — terminate as a last resort.
        3. Restore the pre-run snapshot (if provided) to undo any partial edits
           the worker made before it observed the sentinel.

        *task_id* is the directory holding ``result.json`` for THIS task (the
        worker writes it to ``_subagent_dir(repo_root, task.task_id)``). In
        non-reuse mode it equals *worker_id*; in reuse mode they differ, so the
        soft-path quiescence poll needs the task dir explicitly. Defaults to
        *worker_id* for backward compatibility.

        ``revert_snapshots`` may be ``None`` when the caller has nothing to
        revert; step 3 is then skipped.

        Returns ``True`` ONLY when the worker was kept alive in a reusable
        (soft-quiesced, idle) state — the caller may then return it to the reuse
        pool. Returns ``False`` when the worker was terminated or is hung (hard
        cancel, already-dead, or quiesce-failure); such a worker must NOT be
        reused — re-dispatching a hung Terminal-launched worker (no PID handle,
        invisible to the claim's liveness check) burns the full ``ipc_timeout_s``
        on the next claim. ``False`` on a hard cancel is moot: the pool is
        cleared at run end regardless.
        """
        _reusable = False  # set True only on soft-quiesce success below
        # (1) Always signal abort. Harmlessly redundant with the run-end
        # _cleanup_ipc_workers on a real Ctrl-C; the essential fix on a timeout.
        try:
            from .subagent_ipc import write_cancel_sentinel
            write_cancel_sentinel(repo_root, worker_id)
        except Exception as e:
            logger.warning("IPC: failed to signal cancel for worker %s: %s", worker_id, e)

        # Hard vs soft: an orchestrator-level cancel (cancel_event set) ends the
        # whole run → terminate. A pure timeout (cancel_event clear) → quiesce and
        # keep the worker alive for reuse (B6).
        _hard = bool(
            self.orch_config.cancel_event and self.orch_config.cancel_event.is_set()
        )
        _proc = None
        with self._bg_lock:
            _proc = self._ipc_worker_procs.get(worker_id)

        # (2a) Soft quiescence: poll for a fresh result.json (the worker writes one
        # the instant it aborts on the cancel sentinel). Its appearance is a strong
        # guarantee — the worker's result-write is its LAST filesystem action, so a
        # present result.json means all in-flight edits have landed and the worker
        # is idle. We restore on top of that quiesced state WITHOUT killing the
        # worker (it stays reusable). If it never appears the worker is hung →
        # terminate as the last resort.
        if not _hard:
            _result_dir_id = task_id or worker_id
            _result_path = os.path.join(
                repo_root, ".asicode", "subagents", _result_dir_id, "result.json",
            )
            _quiesce_deadline = time.monotonic() + grace_s
            _quiesced = False
            # Perf #8: a worker that is ALREADY dead (poll() != None) — e.g. its
            # heartbeat went stale during wait_for_result because it crashed — will
            # NEVER write result.json, so polling up to `grace_s` (~12s) is pure
            # waste. Skip straight to the snapshot restore. A dead worker has no
            # pending writes, so restoring on top of whatever partial state it left
            # is safe and correct.
            _already_dead = _proc is not None and _proc.poll() is not None
            if _already_dead:
                logger.info(
                    "IPC: worker %s already exited (code %s) — skipping %.0fs "
                    "result.json quiesce poll (a dead worker never writes it).",
                    worker_id, _proc.returncode, grace_s,
                )
            else:
                while time.monotonic() < _quiesce_deadline:
                    if os.path.isfile(_result_path):
                        _quiesced = True
                        break
                    time.sleep(0.3)
                if _quiesced:
                    logger.info(
                        "IPC: worker %s quiesced (result.json appeared) after soft "
                        "timeout — keeping it alive for reuse.", worker_id,
                    )
                    _reusable = True
                else:
                    # Worker did not honor the sentinel within grace — terminate.
                    logger.warning(
                        "IPC: worker %s did not quiesce within %.0fs after cancel; "
                        "terminating (hung worker).", worker_id, grace_s,
                    )
                    if _proc is not None:
                        try:
                            _proc.terminate()
                        except Exception:
                            pass
        else:
            # (2b) Hard cancel: wait for a tracked Popen to exit, terminate if it
            # lingers. _cleanup_ipc_workers is the belt-and-braces at run end.
            if _proc is not None:
                try:
                    _proc.wait(timeout=grace_s)
                except Exception:
                    try:
                        _proc.terminate()
                    except Exception:
                        pass

        # (3) Revert partial edits to the captured pre-run baseline.
        if revert_snapshots:
            reverted = _restore_assigned_snapshots(repo_root, revert_snapshots)
            if reverted:
                logger.info(
                    "SubAgent IPC worker %s abandoned (%s); restored %s",
                    worker_id, "cancel" if _hard else "timeout", reverted,
                )
        return _reusable
    # ── Helpers ────────────────────────────────────────────────────────────

    def _has_dependencies(self, subtasks: list[SubTaskSpec]) -> bool:
        """Return True if any subtask has dependencies."""
        return any(st.dependencies for st in subtasks)

    @staticmethod
    def _detect_cycles_kahn(task_map: dict[str, SubTaskSpec]) -> tuple[list[str], list[list[str]]]:
        """
        Detect cycles in dependency graph using Kahn's algorithm.

        Returns:
            Tuple of (topological_order, cycles_found)
            - topological_order: List of task_ids in valid execution order (empty if cycles)
            - cycles_found: List of cycles, each cycle as list of task_ids
        """

        # Build adjacency list and indegree map
        adj = {tid: [] for tid in task_map}
        indegree = {tid: 0 for tid in task_map}

        for st in task_map.values():
            for dep in st.dependencies:
                if dep in task_map:
                    adj[dep].append(st.task_id)
                    indegree[st.task_id] = indegree.get(st.task_id, 0) + 1

        # Kahn's algorithm
        q = deque([tid for tid in indegree if indegree[tid] == 0])
        order = []

        while q:
            tid = q.popleft()
            order.append(tid)
            for nxt in adj[tid]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    q.append(nxt)

        # Check for cycles
        if len(order) == len(task_map):
            return order, []  # No cycles

        # Find cycles using DFS on remaining nodes
        cycles = []
        visited = set()
        on_stack = set()
        stack = []

        def dfs(v: str) -> None:
            visited.add(v)
            on_stack.add(v)
            stack.append(v)

            for w in adj[v]:
                if w not in task_map:
                    continue  # Skip dependencies not in current task set
                if w not in visited:
                    dfs(w)
                elif w in on_stack:
                    # Found a cycle — record it but KEEP SCANNING other branches.
                    # An early ``return`` here would skip the on_stack.remove /
                    # stack.pop below, corrupting state for the next root the
                    # ``remaining`` loop visits (extra/false cycles missed or
                    # invented).  Backtracking naturally keeps stack/on_stack
                    # consistent across the whole traversal.
                    cycle_start = stack.index(w)
                    cycles.append(stack[cycle_start:])

            on_stack.remove(v)
            stack.pop()

        # Deterministic traversal order: ``remaining`` is a set whose iteration
        # order is hash-seed dependent, which would make the recorded cycle
        # ROTATION (and downstream _break_cycles edge selection under score
        # ties) non-reproducible across processes — same input, different
        # scheduling. Sorting normalizes the rotation without changing the SET
        # of cycles found (DFS discovers the same SCCs regardless of root).
        remaining = sorted(set(task_map.keys()) - set(order))
        for tid in remaining:
            if tid not in visited:
                dfs(tid)

        return order, cycles

    def _break_cycles(self, subtasks: list[SubTaskSpec], cycles: list[list[str]]) -> Optional[list[SubTaskSpec]]:
        """
        Attempt to break cycles by removing minimal dependencies.

        Strategy:
        1. For each cycle, find the dependency with lowest priority difference
        2. Remove that dependency to break the cycle
        3. Return a NEW acyclic list of SubTaskSpec (with edges removed) if
           successful, else None. The caller re-enters ``_run_dependency_aware``
           on the returned list so the broken-cycle path keeps parallelism,
           predecessor-context injection, and shared-memory accumulation.

        Returning the modified subtasks (rather than a bare execution order)
        is what lets the normal dependency/priority-aware runner take over:
        it rebuilds ``task_map`` and re-runs cycle detection itself, so the
        order computed here is only used as an acyclicity gate.
        """
        if not cycles:
            return None

        # Create a copy of dependencies for modification
        modified_tasks = []
        for st in subtasks:
            modified_tasks.append(SubTaskSpec(
                task_id=st.task_id,
                title=st.title,
                description=st.description,
                assigned_files=list(st.assigned_files),
                dependencies=list(st.dependencies),
                priority=st.priority
            ))

        modified_map = {st.task_id: st for st in modified_tasks}

        # Break each cycle
        for cycle in cycles:
            if len(cycle) < 2:
                continue

            # Find the weakest dependency in the cycle
            weakest = None
            weakest_score = float('inf')

            for i, tid in enumerate(cycle):
                next_tid = cycle[(i + 1) % len(cycle)]
                # Edge direction: the recorded cycle is in DFS-adjacency order
                # (``adj[dep].append(task_id)`` in _detect_cycles_kahn), so an
                # edge ``tid -> next_tid`` here means ``next_tid`` DEPENDS ON
                # ``tid`` — the live edge therefore lives in
                # ``modified_map[next_tid].dependencies``, NOT deps[tid]. The
                # prior reversed check matched nothing along the recorded
                # direction, so this heuristic NEVER removed an edge and every
                # cycle fell through to the blunt fallback below — which on
                # tangled graphs ALSO failed, returning None and forcing the
                # caller into sequential execution (lost parallelism).
                if tid in modified_map[next_tid].dependencies:
                    # Score based on priority difference and task importance
                    prio_diff = abs(modified_map[tid].priority - modified_map[next_tid].priority)
                    score = prio_diff * 10 + len(modified_map[tid].assigned_files)

                    # Deterministic tie-break: when scores are equal, pick the
                    # lexicographically smallest (tid, next_tid) edge so the
                    # SAME edge is removed regardless of the recorded cycle's
                    # rotation. Combined with the sorted ``remaining`` in
                    # _detect_cycles_kahn this makes cycle-breaking fully
                    # reproducible (same input → same edges removed). Without
                    # it, a tied cycle removes whichever minimum edge happened
                    # to be enumerated first — process-dependent.
                    if (
                        score < weakest_score
                        or (score == weakest_score and weakest is not None and (tid, next_tid) < weakest)
                    ):
                        weakest_score = score
                        weakest = (tid, next_tid)

            # Remove the weakest dependency
            if weakest:
                # ``weakest`` is (tid, next_tid) where next_tid DEPENDS ON tid,
                # so drop tid from next_tid's dependency list.
                source_tid, target_tid = weakest  # target depends on source
                if source_tid in modified_map[target_tid].dependencies:
                    modified_map[target_tid].dependencies.remove(source_tid)
                    logger.info(
                        "Broke cycle by removing dependency %s → %s (score: %s)",
                        target_tid, source_tid, weakest_score
                    )

        # Verify acyclicity (the order itself is discarded — the caller
        # rebuilds task_map and re-runs detection).
        _order, new_cycles = self._detect_cycles_kahn(modified_map)

        if not new_cycles:
            return modified_tasks
        else:
            logger.warning("Could not break all cycles with minimal removal, remaining: %s", new_cycles)
            # Fallback: break cycles by removing all intra-cycle outgoing
            # edges that POINT TO the first node — i.e. cycle members that
            # DEPEND ON it. Same direction correction as the weakest step: an
            # edge _break_node -> other means other depends on _break_node, so
            # the live edge lives in modified_map[other].dependencies. (The
            # prior reversed check removed _break_node's OWN deps, which often
            # included no cycle member, leaving the cycle intact and returning
            # None on tangled graphs.)
            for cycle in new_cycles:
                _break_node = cycle[0]
                _removed = 0
                for other in cycle:
                    if other != _break_node and _break_node in modified_map[other].dependencies:
                        modified_map[other].dependencies.remove(_break_node)
                        _removed += 1
                if _removed:
                    logger.info(
                        "Force-removed %d intra-cycle deps on %s to break cycle %s",
                        _removed, _break_node, cycle,
                    )
            # Check again
            _order2, new_cycles2 = self._detect_cycles_kahn(modified_map)
            if not new_cycles2:
                return modified_tasks
            else:
                logger.warning("Still could not break cycles after aggressive removal: %s", new_cycles2)
                return None

    def _find_current_cycles(self, remaining: set, task_map: dict[str, SubTaskSpec]) -> list[list[str]]:
        """
        Find cycles in the currently remaining tasks.
        """
        # Build subgraph of remaining tasks
        subgraph = {tid: task_map[tid] for tid in remaining}
        _, cycles = self._detect_cycles_kahn(subgraph)
        return cycles

    def _cb(self, event: str, data: dict[str, Any]) -> None:
        if self._cb_fn:
            try:
                self._cb_fn(event, data)
            except Exception:
                pass  # non-critical — never block execution
