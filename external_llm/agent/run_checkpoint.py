"""Pre-write checkpoint gate for the MAIN_AGENT loop.

Replaces the PLANNER lane's ``_checkpoint_plan_files`` gate, which created one
scoped snapshot of a plan's target files before the plan executed. MAIN_AGENT
has no plan to read targets from, so the snapshot is built *incrementally*: the
first write of a run creates the checkpoint, and every later write extends it
with files the run has not touched yet. Because
:meth:`CheckpointStore.extend` never overwrites an already-captured path, the
result is the same guarantee the planner gate gave — every file the run wrote
is stored at its pre-run state, and one ``restore()`` undoes the whole run.

A path the run is about to CREATE is captured too, as an absent tombstone that
``restore()`` unlinks. Dropping those (there is no content to snapshot) is what
made created files outlive an Undo, and the accumulating gate made it worse
than a missing deletion: a file created by one write and edited by the next
exists by the time the second write is gated, so it was snapshotted at its
half-written content and ``restore()`` left a tree the run never passed
through, reporting success.

``ASICODE_CHECKPOINT_ON_WRITE`` keeps the semantics it had under the planner
(case-insensitive; unset/empty means "scoped"):

    "0" / "off" / "false" / "no"  → disabled
    "full"                        → full-repo snapshot, taken once
    anything else                 → scoped snapshot of touched files
"""
from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODE_OFF = "off"
MODE_FULL = "full"
MODE_SCOPED = "scoped"

_DISABLED_VALUES = frozenset({"0", "off", "false", "no"})


def resolve_checkpoint_mode(raw: Optional[str]) -> str:
    """Map ``ASICODE_CHECKPOINT_ON_WRITE`` to one of the three MODE_* values."""
    value = (raw or MODE_SCOPED).strip().lower() or MODE_SCOPED
    if value in _DISABLED_VALUES:
        return MODE_OFF
    return MODE_FULL if value == MODE_FULL else MODE_SCOPED


def resolve_in_repo_paths(paths: Iterable[str], repo_root: str) -> list[str]:
    """Absolute, deduped, in-repo paths — existing or not.

    ``../`` escapes are rejected outright; everything surviving that is handed
    to the store, which decides per path whether it becomes a content snapshot
    or an absent tombstone. Returning nothing is meaningful: the caller must
    skip rather than write an empty checkpoint, whose ``restore()`` is a silent
    no-op reported as success.
    """
    root = Path(repo_root).resolve()
    out: list[str] = []
    for raw in sorted({p for p in paths if p}):
        cand = Path(raw)
        if not cand.is_absolute():
            cand = root / raw
        try:
            cand = cand.resolve()
            cand.relative_to(root)
        except (ValueError, OSError) as exc:
            logger.debug("checkpoint: skipping path outside repo root %r (%s)", raw, exc)
            continue
        out.append(str(cand))
    return out


class RunCheckpointGate:
    """Owns the single accumulating checkpoint for one agent run.

    Shared by reference across ToolRegistry clones so a multi-agent run produces
    one undoable checkpoint rather than one per subagent. All mutation happens
    under ``_lock`` because subagents write concurrently.
    """

    def __init__(self, repo_root: str, mode: Optional[str] = None) -> None:
        self.repo_root = repo_root
        self.mode = resolve_checkpoint_mode(
            mode if mode is not None else os.environ.get("ASICODE_CHECKPOINT_ON_WRITE")
        )
        self.checkpoint_id: Optional[str] = None
        self._store = None
        self._lock = threading.Lock()
        # Paths seen absent by before_write and NOT yet written to the store as
        # tombstones. The gate necessarily fires before the handler, so at that
        # moment "does not exist" does not yet mean "the run created it" — the
        # write may be refused (a syntax gate, a scoped write filter, a bad
        # argument), leaving a tombstone for a file that never appeared. Undo
        # would then DELETE that path if the user later created it by hand:
        # data loss caused by a write that never happened.
        #
        # So absence is remembered here and only persisted by confirm_writes(),
        # once the path actually exists. Nothing is lost by waiting — a run that
        # never creates the file needs no tombstone for it.
        self._pending_absent: set[str] = set()

    @property
    def enabled(self) -> bool:
        return self.mode != MODE_OFF

    def _get_store(self):
        if self._store is None:
            from external_llm.agent.checkpoint_store import CheckpointStore
            self._store = CheckpointStore(self.repo_root)
        return self._store

    def before_write(self, paths: Optional[Iterable[str]]) -> Optional[str]:
        """Capture pre-write state for ``paths``; returns the checkpoint id.

        Called on the write path, so it must never raise: a checkpoint is a
        convenience, and failing to take one must not fail the user's edit.
        """
        if not self.enabled:
            return None
        try:
            with self._lock:
                if self.mode == MODE_FULL:
                    if self.checkpoint_id is None:
                        self.checkpoint_id = self._get_store().create(
                            "Pre-run snapshot (full repo)"
                        )
                    return self.checkpoint_id

                in_repo = resolve_in_repo_paths(paths or (), self.repo_root)
                if not in_repo:
                    # No usable target at all (unknown args, or every path
                    # escaped the repo root). A later write still creates or
                    # extends the checkpoint.
                    return self.checkpoint_id

                # A path that does not exist yet has no content to snapshot; it
                # is remembered for confirm_writes() instead of being dropped,
                # which is what made a run's created files outlive its Undo.
                existing = [p for p in in_repo if Path(p).is_file()]
                self._pending_absent.update(p for p in in_repo if not Path(p).exists())
                if not existing:
                    return self.checkpoint_id

                store = self._get_store()
                if self.checkpoint_id is None:
                    self.checkpoint_id = store.create(
                        "Pre-run snapshot (scoped)", files=existing
                    )
                else:
                    store.extend(self.checkpoint_id, existing)
                return self.checkpoint_id
        except Exception:
            # Logged, not swallowed: a gate that fails on every write would
            # otherwise silently leave every run without an Undo point.
            logger.warning("run checkpoint gate failed", exc_info=True)
            return self.checkpoint_id

    def confirm_writes(self, paths: Optional[Iterable[str]]) -> Optional[str]:
        """Tombstone the pending absences that *this* successful write created.

        Called after a write tool succeeded, with the same targets the matching
        :meth:`before_write` received. A pending path is confirmed only if it is
        among those targets AND now exists — both halves are needed:

        * existence alone is not enough. A path can go from missing to present
          because the USER created it between two agent writes; confirming on
          existence made Undo delete their file. (A gate driven only by
          "does it exist now?" cannot tell the two apart, which is why the
          write's own targets are the evidence.)
        * being a target alone is not enough either — the write may have been
          refused before creating anything, and this method is reached on the
          success branch of one write while other targets stay pending.

        A target that still does not exist stays pending, because a later write
        in the same run may yet create it; if none does, it correctly never
        becomes a tombstone.

        Same contract as :meth:`before_write`: never raises, and returns the
        checkpoint id (which this call may be the first to create, for a run
        whose only effect so far is creating files).
        """
        if not self.enabled or self.mode == MODE_FULL:
            return self.checkpoint_id
        try:
            with self._lock:
                if not self._pending_absent:
                    return self.checkpoint_id
                targets = set(resolve_in_repo_paths(paths or (), self.repo_root))
                created = sorted(
                    p for p in self._pending_absent & targets if Path(p).exists()
                )
                if not created:
                    return self.checkpoint_id
                self._pending_absent.difference_update(created)

                store = self._get_store()
                if self.checkpoint_id is None:
                    self.checkpoint_id = store.create(
                        "Pre-run snapshot (scoped)", files=[], absent=created
                    )
                else:
                    store.extend(self.checkpoint_id, [], absent=created)
                return self.checkpoint_id
        except Exception:
            logger.warning("run checkpoint confirm failed", exc_info=True)
            return self.checkpoint_id
