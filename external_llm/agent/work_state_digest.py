"""Deterministic work-state digest for design chat turns.

The design chat tool loop discards all tool call/result messages when a turn
ends — only the final assistant text survives into the next turn's context.
That makes follow-up requests ("now fix that function") re-explore from scratch.

This module extracts a compact, deterministic digest from the turn's tool
call records (DesignChatResult.tool_results) WITHOUT any LLM call. The digest
is stored on the session turn and re-injected into the next turn's context,
so the model knows which files it read/changed and which commands ran,
at ~1-2% of the token cost of keeping the raw tool outputs.

Determinism matters: the digest is rendered into the context verbatim every
turn, so it must be byte-identical across rebuilds to keep the prompt-cache
prefix stable.
"""
from __future__ import annotations

from typing import Any

# Tool taxonomy (names from tool_schemas.py). Unknown tools fall into "other"
# and are reported by name only — new tools degrade gracefully, never crash.
_READ_TOOLS = {"read_file", "read_symbol", "get_file_outline", "read_image"}
_WRITE_TOOLS = {
    "create_file", "apply_patch", "write_plan", "edit_ast", "edit_file",
    "edit_text", "modify_symbol", "anchor_edit",
}
_COMMAND_TOOLS = {"bash", "run_tests", "run_lint"}
_SEARCH_TOOLS = {
    "grep", "find_symbol", "find_references", "find_relevant_files",
    "explore_codebase", "query_dependency_graph", "analyze_change_impact",
    "find_import_source",
    "estimate_change_scope", "run_structural_scan",
    "analyze_semantic_gap", "suggest_edit_location", "find_tests_for_symbol",
    "search_web", "web_fetch", "search_design_history",
}
# Tools with no cross-turn state worth carrying (pure UI / meta)
_IGNORED_TOOLS = {"ask_user", "save_insight"}

_MAX_ITEMS_PER_SECTION = 10
_MAX_FAILURES = 5
_ERR_EXCERPT_CHARS = 120
_QUERY_EXCERPT_CHARS = 40
_CMD_EXCERPT_CHARS = 60

_PATH_ARG_KEYS = ("path", "file_path", "filepath", "file", "filename", "target_file")
_QUERY_ARG_KEYS = ("pattern", "query", "name", "symbol", "keywords", "description", "url")


def _arg_path(args: dict[str, Any]) -> str:
    for key in _PATH_ARG_KEYS:
        v = args.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _arg_query(args: dict[str, Any]) -> str:
    for key in _QUERY_ARG_KEYS:
        v = args.get(key)
        if isinstance(v, str) and v.strip():
            q = v.strip()
            if len(q) > _QUERY_EXCERPT_CHARS:
                q = q[:_QUERY_EXCERPT_CHARS] + "…"
            return q
    return ""


def _dedup_capped(items: list[str], cap: int) -> list[str]:
    """Order-preserving dedup, capped at `cap` with an overflow marker."""
    seen = list(dict.fromkeys(i for i in items if i))
    if len(seen) > cap:
        return [*seen[:cap], f"(+{len(seen) - cap} more)"]
    return seen


def _first_line(text: str, limit: int) -> str:
    line = (text or "").strip().split("\n", 1)[0]
    if len(line) > limit:
        line = line[:limit] + "…"
    return line


def build_work_state_digest(tool_results: list[dict[str, Any]]) -> str:
    """Build a compact digest from a turn's tool call records.

    Each entry: {"tool": str, "args": dict, "content": str, "ok": bool}
    (see DesignChatResult.tool_results). Returns "" when there is nothing
    worth carrying (no tool calls, or only ignored tools).
    """
    if not tool_results:
        return ""

    reads: list[str] = []
    writes: list[str] = []
    commands: list[str] = []
    searches: list[str] = []
    others: list[str] = []
    failures: list[str] = []

    for entry in tool_results:
        if not isinstance(entry, dict):
            continue
        tool = entry.get("tool") or ""
        args = entry.get("args") if isinstance(entry.get("args"), dict) else {}
        ok = bool(entry.get("ok", True))

        if tool in _IGNORED_TOOLS or not tool:
            continue

        if not ok:
            target = _arg_path(args) or _arg_query(args)
            err = _first_line(entry.get("content") or "", _ERR_EXCERPT_CHARS)
            failures.append(
                f"{tool} {target}".strip() + (f" — {err}" if err else "")
            )

        if tool in _READ_TOOLS:
            path = _arg_path(args)
            if path:
                sym = args.get("symbol") or args.get("name") or ""
                reads.append(f"{path}:{sym}" if isinstance(sym, str) and sym else path)
        elif tool in _WRITE_TOOLS:
            path = _arg_path(args)
            if path:
                status = "" if ok else " FAILED"
                writes.append(f"{path} ({tool}{status})")
        elif tool in _COMMAND_TOOLS:
            cmd = args.get("command") or args.get("cmd") or ""
            label = tool
            if isinstance(cmd, str) and cmd.strip():
                c = cmd.strip().replace("\n", " ")
                if len(c) > _CMD_EXCERPT_CHARS:
                    c = c[:_CMD_EXCERPT_CHARS] + "…"
                label = f"{tool}: {c}"
            commands.append(f"{label} → {'ok' if ok else 'FAILED'}")
        elif tool in _SEARCH_TOOLS:
            q = _arg_query(args) or _arg_path(args)
            searches.append(f"{tool} {q}".strip())
        else:
            others.append(tool)

    sections: list[str] = []
    if reads:
        sections.append("read: " + ", ".join(_dedup_capped(reads, _MAX_ITEMS_PER_SECTION)))
    if writes:
        sections.append("modified: " + ", ".join(_dedup_capped(writes, _MAX_ITEMS_PER_SECTION)))
    if commands:
        sections.append("ran: " + "; ".join(_dedup_capped(commands, _MAX_ITEMS_PER_SECTION)))
    if searches:
        sections.append("searched: " + "; ".join(_dedup_capped(searches, _MAX_ITEMS_PER_SECTION)))
    if others:
        sections.append("other tools: " + ", ".join(_dedup_capped(others, _MAX_ITEMS_PER_SECTION)))
    if failures:
        sections.append("failed: " + "; ".join(_dedup_capped(failures, _MAX_FAILURES)))

    return "\n".join(sections)
