"""Sub-agent worker mode for the REPL (P6-2 decomposition — R1).

Extracted from ``external_llm/repl/repl_impl.py``: the ``--subagent`` worker
loop (poll task.json → run via DesignChatLoop → write result.json) and its
turn-budget resolver.

Import contract: this module does ``import asi`` to reach the shared REPL
scaffolding (``_C``/``_print``/``_REPO_ROOT``) and the repl_impl helpers it
depends on (``_resolve_repo_root``/``_resolve_subagent_max_turns``/
``_ProgressPrinter``).  ``asi`` is fully initialized whenever this module is
imported — ``repl_impl`` imports this module ONLY at its very bottom (after
``asi`` has imported ``repl_impl``), so no import cycle is possible.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time

# The turn-budget SSOT lives in thresholds (same module repl_impl uses).
from external_llm.agent.config.thresholds import config as _cfg


def _asi():
    """Import asi lazily (function-local) so importing this module NEVER
    triggers the asi ↔ repl_impl symmetric cycle (see repl_impl bottom-of-file
    import comment).  ``asi`` is guaranteed fully initialized at call time:
    ``_resolve_subagent_max_turns``/``run_subagent_worker`` only run after the
    REPL or orchestrator has imported asi, which completes repl_impl's
    bottom-of-file ``import asi`` first.
    """
    import asi

    return asi


def _resolve_subagent_max_turns(task, args: argparse.Namespace) -> int:
    """Resolve a sub-agent worker's turn budget: task → CLI args → SSOT.

    The orchestrator-written task.json carries the authoritative budget; the
    worker's own ``--max-turns`` (manual launch) is the override; the SSOT
    ``AGENT_MAX_TURNS_DEFAULT`` is the last-resort fallback so a task written
    without the field never silently shrinks to a magic number. ``0`` is
    treated as unset (matches the historical ``or`` chain).
    """
    return task.max_turns or getattr(args, "max_turns", None) or _cfg.counts.AGENT_MAX_TURNS_DEFAULT


def run_subagent_worker(args: argparse.Namespace) -> None:
    """Sub-agent worker mode: poll task.json → run → write result.json.

    Watches ``.asicode/subagents/<subagent_id>/task.json``; when a task arrives,
    runs it via DesignChatLoop and writes ``result.json`` back.  Stays alive in a
    loop so the orchestrator can reuse the worker for subsequent tasks (Ctrl-C
    to exit).

    Launched automatically by ``/orchestrate`` (auto_launch_terminal on macOS)
    or manually: ``asi --subagent --subagent-id <id> --provider ... --model ...``
    """
    # Lazy-bind shared REPL scaffolding from asi (fully initialized at call
    # time — see _asi docstring). Local names keep the extracted body
    # byte-identical to the original repl_impl source.
    global _REPO_ROOT
    _asi_mod = _asi()
    _colors = _asi_mod._C
    _print = _asi_mod._print
    _REPO_ROOT = _asi_mod._REPO_ROOT
    _resolve_repo_root = _asi_mod._resolve_repo_root
    _progress_printer = _asi_mod._ProgressPrinter

    repo_root = _resolve_repo_root(args.repo)
    _REPO_ROOT = repo_root

    agent_id = args.subagent_id
    if not agent_id:
        _print("--subagent-id is required with --subagent", _colors["red"])
        sys.exit(1)

    # Capture the spawning parent PID ONCE at worker start. If the orchestrator
    # (headless path: this worker is its direct child) is SIGKILL'd,
    # poll_for_task's orphan check (_is_process_alive, cross-platform) detects
    # that this pid is dead and self-exits. Captured here (not per-call)
    # because the pid to watch never changes.
    #
    # getppid() is only correct on the direct-child launch path (headless
    # background spawn). On the macOS Terminal.app path (osascript → Terminal
    # → login shell → this worker) the parent is the login shell, NOT the
    # orchestrator — getppid() never changes when the orchestrator itself
    # dies, so orphan self-exit silently never fires and the worker idles up
    # to max_poll_s (24h). --orch-pid carries the orchestrator's actual PID
    # explicitly (set by asr_subagent_argv callers) so poll_for_task can
    # probe it directly instead of trusting getppid(). Falls back to the old
    # getppid()-based check when absent (manual launch, older orchestrator).
    _origin_ppid = os.getppid()
    _orch_pid = getattr(args, "orch_pid", 0) or None

    # Idle heartbeat writer: proves the WORKER PROCESS is alive while it is
    # between tasks (polling), independent of any task dir. The orchestrator's
    # _claim_reusable_worker reads this to judge a Terminal-launched worker's
    # (no PID handle) liveness BEFORE reuse — closing the gap where a hung
    # Terminal worker was optimistically reused and burned ipc_timeout_s. See
    # write_worker_idle_heartbeat / read_worker_idle_heartbeat_age.
    from external_llm.agent.subagent_ipc import (
        _IDLE_HEARTBEAT_INTERVAL_S,
        HEARTBEAT_INTERVAL_S,
        SubagentResult,
        build_subagent_prompt,
        partition_changed_files,
        poll_for_task,
        write_heartbeat,
        write_result,
        write_worker_exited_heartbeat,
        write_worker_idle_heartbeat,
    )

    _print(
        f"Sub-agent worker [{agent_id}] started, watching {repo_root}/.asicode/subagents/{agent_id}/task.json",
        _colors["teal"],
    )

    cancel_event = threading.Event()  # TASK-scope: abort the current task only
    shutdown_event = threading.Event()  # PROCESS-scope: exit the worker loop

    def _sigint_handler(sig, frame):
        # Ctrl-C is a process-level intent: abort the in-flight task AND exit.
        _print(f"\nSub-agent [{agent_id}] shutting down…", _colors["yellow"])
        cancel_event.set()  # abort in-flight task immediately
        shutdown_event.set()  # exit the poll loop after the task unwinds

    signal.signal(signal.SIGINT, _sigint_handler)

    # ── Idle heartbeat: a long-lived daemon that writes worker.heartbeat.json
    # into the worker's OWN poll directory every _IDLE_HEARTBEAT_INTERVAL_S for
    # the ENTIRE process lifetime. This covers BOTH the between-tasks idle window
    # AND task execution (redundant with the per-task heartbeat, but harmless —
    # the idle heartbeat proves process liveness regardless of task state).
    # _claim_reusable_worker reads it to reject a hung Terminal-launched worker
    # (no PID handle) before reuse, so a dead worker is never re-dispatched.
    _idle_hb_stop = threading.Event()
    # Observability state for the idle heartbeat (Bug 4): mutated only by the
    # main poll loop below (single-threaded), read by this daemon thread. Not
    # lock-protected — advisory display data, not a correctness signal, and
    # simple int/str assignment is atomic enough under the GIL.
    _worker_start_ts = time.monotonic()
    _worker_stats = {"tasks_served": 0, "last_task_id": ""}

    def _write_idle_hb() -> None:
        write_worker_idle_heartbeat(
            repo_root,
            agent_id,
            pid=os.getpid(),
            tasks_served=_worker_stats["tasks_served"],
            last_task_id=_worker_stats["last_task_id"],
            uptime_s=time.monotonic() - _worker_start_ts,
        )

    def _idle_heartbeat_writer() -> None:
        while not _idle_hb_stop.wait(_IDLE_HEARTBEAT_INTERVAL_S):
            try:
                _write_idle_hb()
            except Exception:
                logging.getLogger(__name__).debug("idle heartbeat write failed", exc_info=True)

    # Write one immediately so a fresh heartbeat exists before the first poll.
    try:
        _write_idle_hb()
    except Exception:
        logging.getLogger(__name__).debug("initial idle heartbeat write failed", exc_info=True)
    threading.Thread(
        target=_idle_heartbeat_writer,
        name=f"ipc-idle-heartbeat-{agent_id}",
        daemon=True,
    ).start()

    # ── External cancel watcher ────────────────────────────────────────────
    # The orchestrator signals mid-task cancellation by writing a ``cancel.json``
    # sentinel into this worker's poll directory.  This PER-TASK daemon thread
    # polls for that sentinel WHILE a task runs and sets the task-scope
    # ``cancel_event`` — DesignChatLoop already checks ``cancel_event`` at the top
    # of every iteration (turn boundary) and aborts via ``AgentCancelled``.
    #
    # Lifecycle (B6): ``cancel.json`` is TASK-scoped, NOT process-scoped.  After
    # the task aborts and writes its error result, ``cancel_event`` is cleared and
    # the worker loops back to poll for the NEXT task — a single cancel no longer
    # kills a reusable worker (saving the ~5-8s respawn cost per cancelled task).
    # Only SIGINT, the ``shutdown.json`` sentinel, or orphan-detection exit the
    # worker.  The watcher is started PER-TASK (not as one long-lived thread) so a
    # stale sentinel cannot fire during idle polling between tasks (write_task also
    # clears a stale cancel.json as a belt-and-braces measure).
    from external_llm.agent.subagent_ipc import check_cancel_sentinel

    def _cancel_watcher(stop_flag: threading.Event) -> None:
        _cancel_path = os.path.join(
            repo_root,
            ".asicode",
            "subagents",
            agent_id,
            "cancel.json",
        )
        while not stop_flag.is_set() and not cancel_event.is_set():
            try:
                if check_cancel_sentinel(repo_root, agent_id):
                    _print(
                        f"\nSub-agent [{agent_id}] cancel signal received "
                        f"— aborting current task at the next turn boundary.",
                        _colors["yellow"],
                    )
                    try:
                        os.unlink(_cancel_path)  # fire-once
                    except OSError:
                        logging.getLogger(__name__).debug("cancel sentinel unlink failed", exc_info=True)
                    cancel_event.set()
                    return
            except Exception:
                logging.getLogger(__name__).debug("cancel watcher poll failed", exc_info=True)
            # P27-2: stop_flag.wait(0.5) — wake immediately on shutdown instead of
            # sleeping the full poll interval (cancelable-sleep pattern; keeps the
            # 0.5s cancel-sentinel cadence while a task runs).
            stop_flag.wait(0.5)

    # ── Per-process LLM service cache (worker reuse optimization).
    # create_intelligent_service_from_env builds a client (auth handshake, model
    # resolution) every call — a few seconds each. A reused worker serves MANY
    # tasks, often with the SAME (provider, model, api_key), so re-creating it per
    # task defeats the worker-reuse (P3) goal of saving the ~5-8s respawn cost.
    # Cache the service keyed by (provider, model, api_key); a task that changes
    # any of these misses the cache and re-initializes. The cache holds AT MOST
    # one entry (the common case: all tasks use the same provider) — a different
    # key evicts the prior entry. api_key is normalized to "" so None/"" collide.
    _svc_cache: dict = {}  # {(provider, model, api_key): svc} — single-slot cache

    # Run tasks in a loop.  The worker stays alive across tasks so the orchestrator
    # can reuse it (no new Terminal/process per task).  Exit ONLY on: SIGINT
    # (shutdown_event), the shutdown.json sentinel, or orphan-detection — NOT on a
    # task-level cancel.json (B6: that aborts only the current task, then the
    # worker loops back to serve the next one).
    while not shutdown_event.is_set():
        # Reset the task-scope cancel before polling so a previous task's cancel
        # does not leak into the next poll (poll_for_task checks cancel_event and
        # returns None if it is set).
        cancel_event.clear()
        _print(f"[{agent_id}] Polling for task… (Ctrl-C to exit)", _colors["muted"])
        task = poll_for_task(
            repo_root=repo_root,
            agent_id=agent_id,
            poll_interval_s=1.0,
            timeout_s=None,  # infinite — worker stays alive until killed
            cancel_event=cancel_event,
            expected_parent_pid=_origin_ppid,
            orchestrator_pid=_orch_pid,
        )
        if task is None or shutdown_event.is_set():
            break

        # Start THIS task's cancel watcher (per-task; stopped after the result is
        # written so it cannot fire during the next idle poll).
        _watcher_stop = threading.Event()
        threading.Thread(
            target=_cancel_watcher,
            args=(_watcher_stop,),
            name=f"ipc-cancel-{agent_id}-{task.task_id}",
            daemon=True,
        ).start()

        _print(f"[{agent_id}] Received task: {task.title}", _colors["green"])
        _print(f"  files: {task.assigned_files}", _colors["muted"])
        if task.description:
            _print(f"  description: {task.description[:200]}", _colors["muted"])

        # Task payload can override provider/model/api_key; else use CLI args.
        provider = task.provider or getattr(args, "provider", "") or ""
        model = task.model or getattr(args, "model", "") or ""
        api_key = task.api_key or getattr(args, "api_key", "") or None
        max_turns = _resolve_subagent_max_turns(task, args)

        printer = _progress_printer(verbose=getattr(args, "verbose", False))
        ipc_result = None

        # ── Heartbeat: prove liveness so the orchestrator's wait_for_result
        # can distinguish a BUSY worker (long LLM/tool turn) from a DEAD one
        # (OOM/segfault) instead of burning the full ipc_timeout_s. A daemon
        # thread writes heartbeat.json (wall-clock ts) every HEARTBEAT_INTERVAL_S
        # into the task's OWN dir (same as result.json) so the orchestrator can
        # read it back. The thread is stopped once the result is written below.
        # Uses pid so a diagnostic can identify the heartbeating process.
        _hb_stop = threading.Event()
        # Shared progress state, updated by the wrapped stream_callback below and
        # read by the heartbeat thread so heartbeats carry "turn N, <tool>" hints
        # (F3). Plain dict writes/reads are GIL-atomic for independent keys; the
        # heartbeat is advisory, so a torn read across keys is harmless.
        _hb_state = {"turn": 0, "last_tool": ""}

        # Bind the task id NOW: the worker loop reassigns ``task`` for the next
        # task, and the heartbeat thread may still be mid-write after
        # _hb_stop.set() (wait→False race). Late-binding task.task_id would then
        # stamp a heartbeat with the NEXT task's id into its dir. A plain local
        # rebinding cannot pin the value (the closures read the CELL, rebound on
        # the next iteration) — so _heartbeat_writer/_hb_stream_cb bind
        # _hb_stop/_task_id/_hb_state/_printer as def-time default args (B023):
        # each iteration's thread/callback is pinned to THIS iteration's
        # objects, and a straggler thread heartbeats its own task dir instead
        # of stamping the next task's id/state.
        _task_id = task.task_id

        def _heartbeat_writer(_hb_stop=_hb_stop, _task_id=_task_id, _hb_state=_hb_state) -> None:
            while not _hb_stop.wait(HEARTBEAT_INTERVAL_S):
                try:
                    write_heartbeat(
                        repo_root,
                        _task_id,
                        pid=os.getpid(),
                        turn=_hb_state.get("turn", 0),
                        last_tool=_hb_state.get("last_tool", ""),
                    )
                except Exception:
                    logging.getLogger(__name__).debug("task heartbeat write failed", exc_info=True)

        # Write one immediately so a heartbeat exists before the first poll gap.
        try:
            write_heartbeat(repo_root, task.task_id, pid=os.getpid())
        except Exception:
            logging.getLogger(__name__).debug("initial task heartbeat write failed", exc_info=True)
        threading.Thread(
            target=_heartbeat_writer,
            name=f"ipc-heartbeat-{agent_id}-{task.task_id}",
            daemon=True,
        ).start()

        # Imported OUTSIDE the try: the `except AgentCancelled:` clause below
        # must resolve even when an earlier failure (svc is None ->
        # RuntimeError) raised before the in-try import ran. A function-level
        # `from ... import` makes the name a local for the WHOLE function, so
        # an unexecuted import = UnboundLocalError inside the except clause,
        # which would kill the worker instead of writing an error result.
        from external_llm.agent.agent_loop_types import AgentCancelled

        try:
            # ── DesignChatLoop-based execution (lighter than AgentLoop/router) ──
            from external_llm.intelligent_service import (
                create_intelligent_service_from_env,
            )

            # Reuse the LLM service across tasks when (provider, model, api_key)
            # is unchanged (see _svc_cache decl above). A reused worker serves
            # many tasks with the same provider, so re-initializing the service
            # per task would burn seconds of auth/handshake each time — defeating
            # the P3 reuse goal. Cache key is the resolved triple; api_key is
            # normalized so None/"" collide.
            _svc_key = (provider or "", model or "", api_key or "")
            _cached = _svc_cache.get(_svc_key)
            if _cached is not None:
                svc = _cached
            else:
                svc = create_intelligent_service_from_env(
                    provider or None,
                    model or None,
                    api_key=api_key or None,
                )
                if svc is not None:
                    # Evict any prior entry (single-slot cache: the common case
                    # is one provider per worker; a provider switch is rare).
                    _svc_cache.clear()
                    _svc_cache[_svc_key] = svc
            if svc is None:
                raise RuntimeError(
                    f"failed to initialize LLM service for sub-agent {agent_id}\n"
                    f"  --provider {provider or '(unset)'} --model {model or '(unset)'}"
                )

            from external_llm.agent.design_chat_loop import DesignChatLoop
            from external_llm.agent.tool_registry import AgentConfig, ToolRegistry
            from external_llm.client import LLMMessage

            config = AgentConfig(
                model_name=svc.model or "",
                max_turns=max_turns,
                stream_callback=printer,
                consume_content_events=False,
                run_lint=True,
                run_tests=True,
                cancel_event=cancel_event,
                unrestricted_read=True,  # trusted local CLI
            )

            registry = ToolRegistry(repo_root, config)
            design_loop = DesignChatLoop(svc.llm_service.client, registry, svc.model)
            printer._start_spinner("")

            t0 = time.perf_counter()

            # Build the initial user message — mirroring the in-process path
            # (orchestrator._run_subagent). ``build_subagent_prompt`` prefers
            # ``task.predecessor_context`` (the richly-built task_text with
            # predecessor results + shared memory) over bare ``task.description``
            # and wraps it with ``task.original_request`` (the overall goal), so
            # dependent IPC subtasks no longer run "blind".
            messages = [LLMMessage(role="user", content=build_subagent_prompt(task))]

            # Wrap the printer so design-loop stream events also feed the heartbeat
            # progress state (F3): each LLM call bumps the turn counter, and a
            # tool_call start records the tool name. The orchestrator's wait_for_result
            # then surfaces "turn N, <tool>" instead of only elapsed time.
            _printer = printer

            def _hb_stream_cb(event: str, payload, _hb_state=_hb_state, _printer=_printer):
                try:
                    if event == "design_llm_call":
                        _hb_state["turn"] = int(_hb_state.get("turn", 0)) + 1
                    elif (
                        event == "design_tool_call" and isinstance(payload, dict) and payload.get("status") == "running"
                    ):
                        _hb_state["last_tool"] = str(payload.get("tool") or "")
                except Exception:  # pragma: no cover - dict int/str assignment cannot raise here
                    logging.getLogger(__name__).debug("heartbeat stream state update failed", exc_info=True)
                if _printer is not None:
                    try:
                        _printer(event, payload)
                    except Exception:
                        logging.getLogger(__name__).debug("subagent printer callback failed", exc_info=True)

            dc_result = design_loop.respond(
                messages,
                stream_callback=_hb_stream_cb,
                max_tool_iterations=max_turns,
            )

            elapsed = time.perf_counter() - t0

            # Collect diff via git for the result payload.
            diff = ""
            try:
                import subprocess as _sp

                diff = _sp.run(
                    ["git", "diff", "--stat", "HEAD", "--", *task.assigned_files],
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                ).stdout.strip()
            except Exception:
                logging.getLogger(__name__).debug("subagent git diff failed", exc_info=True)

            # Map DesignChatResult → SubagentResult fields.
            # is_error takes PRECEDENCE over hit_max_iterations: a task that both
            # exhausted its turn budget AND then failed to generate its final
            # response must surface as an error (not "max_turns"), otherwise the
            # orchestrator misreads a generation failure as "budget exhausted /
            # partial progress" and picks the wrong retry strategy. When
            # DesignChatLoop reaches the max-iterations tail it leaves is_error
            # False unless the final-response call itself failed (in which case
            # it now sets is_error=True) — so a clean budget-exhaustion still
            # reports "max_turns".
            if dc_result.is_error:
                status = "error"
            elif dc_result.hit_max_iterations:
                status = "max_turns"
            else:
                status = "success"
            final_message = dc_result.content or ""
            turns = dc_result.total_llm_calls or len(dc_result.tool_calls_made) or 1
            # DesignChatLoop does not itself track applied patches — derive the
            # file list from an UNSCOPED ``git status`` and partition into
            # in-scope (applied_patches) vs out-of-scope (unassigned_changes)
            # so the orchestrator's diff cross-verification AND scope-violation
            # review both have full visibility (B5). Previously the call was
            # scoped to assigned_files, hiding any out-of-scope write.
            patches, unassigned = partition_changed_files(repo_root, task.assigned_files)
            error_msg = dc_result.content if dc_result.is_error else ""

            ipc_result = SubagentResult(
                task_id=task.task_id,
                status=status,
                final_message=final_message,
                diff=diff,
                turns=turns,
                applied_patches=patches,
                error=error_msg,
                epoch=task.epoch,
                unassigned_changes=unassigned,
            )

            printer._stop_spinner()
            _print(
                f"[{agent_id}] Task complete: {status} ({turns} turns, {elapsed:.1f}s)",
                _colors["green"] if status == "success" else _colors["yellow"],
            )

        except AgentCancelled:
            # Task-level cancel (cancel.json sentinel, B6): abort ONLY this task,
            # write a cancelled result, then loop back to poll for the next one.
            # The worker process stays alive (shutdown_event is NOT set by a
            # task-scope cancel) so the orchestrator can reuse it.
            _cancel_msg = f"Sub-agent task '{task.task_id}' cancelled by orchestrator"
            logging.getLogger(__name__).info(
                "Sub-agent %s task %s cancelled (task-scope; worker stays alive)",
                agent_id,
                task.task_id,
            )
            # Report partial edits even when cancelled: a mid-task abort can
            # leave half-applied changes on disk. Without partition_changed_files
            # here the result carries applied_patches=[]/unassigned_changes=[],
            # leaving the orchestrator's diff cross-verification AND the B5 scope
            # signal blind to whatever the worker wrote before it stopped — and
            # hiding the exact set the orchestrator must revert (turn 13114 bug 1).
            try:
                patches, unassigned = partition_changed_files(repo_root, task.assigned_files)
            except Exception:
                patches, unassigned = [], []
            ipc_result = SubagentResult(
                task_id=task.task_id,
                status="cancelled",
                final_message=_cancel_msg,
                applied_patches=patches,
                error=_cancel_msg,
                epoch=task.epoch,
                unassigned_changes=unassigned,
            )
            try:
                printer._stop_spinner()
            except Exception:
                logging.getLogger(__name__).debug("subagent cancel spinner stop failed", exc_info=True)
            _print(f"[{agent_id}] Task cancelled (worker staying alive).", _colors["yellow"])
        except Exception as e:
            logging.getLogger(__name__).exception(
                "Sub-agent %s execution failed",
                agent_id,
            )
            # Drop the cached LLM service: an exception may mean it is broken
            # (expired auth, dropped connection after a long idle period) rather
            # than the task itself being bad. Without this, every subsequent
            # reused-worker task would hit the same dead client and fail
            # immediately instead of reinitializing. Cheap and safe — the next
            # task just re-creates it (a few seconds, only once).
            _svc_cache.clear()
            # Same rationale as the cancelled branch: a task that crashed mid-run
            # may have partial edits on disk. Report them so the orchestrator can
            # attribute, cross-verify, and revert as needed.
            try:
                patches, unassigned = partition_changed_files(repo_root, task.assigned_files)
            except Exception:
                patches, unassigned = [], []
            ipc_result = SubagentResult(
                task_id=task.task_id,
                status="error",
                applied_patches=patches,
                error=str(e),
                epoch=task.epoch,
                unassigned_changes=unassigned,
            )
            try:
                printer._stop_spinner()
            except Exception:
                logging.getLogger(__name__).debug("subagent error spinner stop failed", exc_info=True)
            _print(f"[{agent_id}] Task failed: {e}", _colors["red"])

        # Always write a result so the orchestrator's wait_for_result unblocks.
        # write_result itself can raise (I/O error, serialization failure); if it
        # does, the exception would propagate out of the poll loop and KILL the
        # worker — result.json would never appear and the orchestrator would burn
        # the full ipc_timeout_s before failing. Guard it so the "always writes a
        # result" contract holds even then: retry once with a minimal error result.
        if ipc_result is not None:
            try:
                write_result(repo_root, ipc_result)
                _print(f"[{agent_id}] Result written.", _colors["muted"])
            except Exception as _wr_exc:
                logging.getLogger(__name__).exception(
                    "Sub-agent %s: result write failed (%s); retrying with a minimal error result",
                    agent_id,
                    _wr_exc,
                )
                try:
                    write_result(
                        repo_root,
                        SubagentResult(
                            task_id=ipc_result.task_id,
                            status="error",
                            final_message=f"result write failed: {_wr_exc}",
                            error=f"result write failed: {_wr_exc}",
                            epoch=ipc_result.epoch,
                        ),
                    )
                    _print(f"[{agent_id}] Minimal error result written.", _colors["muted"])
                except Exception:
                    logging.getLogger(__name__).exception(
                        "Sub-agent %s: minimal error-result write also failed; orchestrator will time out.",
                        agent_id,
                    )

        # Record this task for the idle heartbeat's observability fields (Bug
        # 4) before the worker goes back to idle polling — reflects in the
        # NEXT periodic write (or the following task's completion), same as
        # the pre-existing idle-heartbeat cadence.
        _worker_stats["tasks_served"] += 1
        _worker_stats["last_task_id"] = task.task_id

        # Stop this task's heartbeat thread now that a result has been written
        # (or the write is irrecoverably failed). Prevents the daemon thread
        # from carrying a stale heartbeat into the next idle poll cycle.
        _hb_stop.set()
        # Stop this task's cancel watcher (B6): the task is done (or aborted), so
        # there is nothing to cancel. The worker loops back to poll; a fresh
        # watcher is started for the next task.
        try:
            _watcher_stop.set()
        except Exception:  # pragma: no cover - threading.Event.set cannot raise
            logging.getLogger(__name__).debug("subagent watcher stop failed", exc_info=True)

    # Stop the idle-heartbeat daemon and mark the heartbeat "exited" BEFORE the
    # process actually terminates. Without this, the last "idle" heartbeat (up
    # to _IDLE_HEARTBEAT_INTERVAL_S stale) still reads as fresh to
    # _claim_reusable_worker for up to ipc_heartbeat_stale_s (120s default)
    # after this process is gone, so a dead worker gets re-claimed and burns
    # the orchestrator's full ipc_timeout_s on the next dispatch.
    _idle_hb_stop.set()
    try:
        write_worker_exited_heartbeat(repo_root, agent_id, pid=os.getpid())
    except Exception:
        logging.getLogger(__name__).debug("subagent exited-heartbeat write failed", exc_info=True)

    _print(f"Sub-agent [{agent_id}] stopped.", _colors["teal"])
