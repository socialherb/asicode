"""Work-plan state for the design chat tool loop.

Pure helpers shared by the ``update_plan`` tool handler (tool_handlers/
agent_tools.py) and the completion gate in design_chat_loop.py. Kept in a
separate module so design_chat_loop can import it without touching
tool_registry (avoids circular imports).

A plan is a plain dict::

    {"goal": str, "items": [{"title": str, "status": str, "note": str}]}

Statuses:
  - pending / in_progress  — open (work remains)
  - done / skipped / blocked — terminal (item has an explicit disposition)

``skipped`` and ``blocked`` require a non-empty ``note`` explaining why —
the gate accepts an unfinished plan only when every remaining item carries
an honest reason.
"""
from __future__ import annotations

from typing import Any, Optional

OPEN_STATUSES = ("pending", "in_progress")
TERMINAL_STATUSES = ("done", "skipped", "blocked")
ALL_STATUSES = OPEN_STATUSES + TERMINAL_STATUSES

_STATUS_MARKS = {
    "pending": "[ ]",
    "in_progress": "[~]",
    "done": "[x]",
    "skipped": "[-]",
    "blocked": "[!]",
}

MAX_PLAN_ITEMS = 50
_MAX_TITLE_CHARS = 300
_MAX_NOTE_CHARS = 500


def validate_plan(goal: Any, items: Any) -> tuple[Optional[dict[str, Any]], str]:
    """Validate and normalize an update_plan payload.

    Returns (plan_dict, "") on success or (None, error_message) on failure.
    """
    if not isinstance(items, list) or not items:
        return None, "'items' must be a non-empty array"
    if len(items) > MAX_PLAN_ITEMS:
        return None, f"'items' exceeds the {MAX_PLAN_ITEMS}-item limit — merge related steps"

    norm_items: list[dict[str, str]] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            return None, f"items[{i}] must be an object"
        title = str(item.get("title", "")).strip()
        if not title:
            return None, f"items[{i}].title is required"
        status = str(item.get("status", "pending")).strip() or "pending"
        if status not in ALL_STATUSES:
            return None, (
                f"items[{i}].status '{status}' is invalid — "
                f"use one of: {', '.join(ALL_STATUSES)}"
            )
        note = str(item.get("note", "")).strip()
        if status in ("skipped", "blocked") and not note:
            return None, (
                f"items[{i}] ('{title[:60]}') is '{status}' but has no 'note' — "
                "a reason is required when skipping or blocking an item"
            )
        norm_items.append({
            "title": title[:_MAX_TITLE_CHARS],
            "status": status,
            "note": note[:_MAX_NOTE_CHARS],
        })

    return {"goal": str(goal or "").strip(), "items": norm_items}, ""


def open_items(plan: Optional[dict[str, Any]]) -> list[dict[str, str]]:
    """Return items still in an open (non-terminal) status."""
    if not plan:
        return []
    return [it for it in plan.get("items", []) if it.get("status") in OPEN_STATUSES]


def _short_title(title: str, limit: int = 40) -> str:
    return title if len(title) <= limit else title[: limit - 1] + "…"


def diff_plans(prev: Optional[dict[str, Any]], new: dict[str, Any]) -> str:
    """One-line summary of what changed between two plans.

    Emitted as the FIRST line of the update_plan tool result so the CLI's
    short tool-output preview shows WHY each call happened (created items,
    status transitions, additions/removals) instead of a generic header.
    Items are matched by title (plans are full replacements).
    """
    items = new.get("items", [])
    done = sum(1 for it in items if it.get("status") == "done")
    n_open = len(open_items(new))
    tail = f"({done}/{len(items)} done, {n_open} open)"

    if not prev:
        return f"Plan created: {len(items)} item(s) {tail}"

    prev_status = {it["title"]: it.get("status") for it in prev.get("items", [])}
    changes: list[str] = []
    for it in items:
        old = prev_status.pop(it["title"], None)
        status = it.get("status")
        if old is None:
            changes.append(f"+'{_short_title(it['title'])}'")
        elif old != status:
            changes.append(f"'{_short_title(it['title'])}' {old}→{status}")
    for title in prev_status:
        changes.append(f"-'{_short_title(title)}'")

    head = "; ".join(changes) if changes else "no status changes"
    return f"Plan updated — {head} {tail}"


def render_plan(plan: dict[str, Any]) -> str:
    """Render the plan as a compact checklist for prompt injection."""
    lines: list[str] = []
    goal = plan.get("goal", "")
    if goal:
        lines.append(f"Goal: {goal}")
    for it in plan.get("items", []):
        mark = _STATUS_MARKS.get(it.get("status", "pending"), "[ ]")
        line = f"{mark} {it.get('title', '')}"
        if it.get("note"):
            line += f" — {it['note']}"
        lines.append(line)
    done = sum(1 for it in plan.get("items", []) if it.get("status") == "done")
    lines.append(f"({done}/{len(plan.get('items', []))} done)")
    return "\n".join(lines)
