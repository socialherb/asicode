"""Full-detail renderer for interrupted tool-loop results (Option B).

When a design chat turn is interrupted by ESC mid-tool-loop, the partial
DesignChatResult.tool_results carry the exact code the model had just
read/modified — detail that the deterministic work-state digest
(work_state_digest.py) intentionally omits (it keeps only file/tool names).

Option B persists those tool_results on the turn (design_session.add_turn's
``tool_results`` param) and renders them into the *next* turn's context via
this module, so the resumed model picks up the interrupted work without
re-reading files. A byte-budget cap keeps context from inflating unboundedly,
and as soon as the turn scrolls out of the verbatim window the normal
compression path replaces it with the compact digest — "full right after
resume, summarized later".

Determinism contract: identical input list → identical output string
(fixed iteration order, fixed headers/separators). build_context_messages
re-renders every turn, so the prompt-cache prefix must not churn.

Result-item shape (design_chat_loop.py):
    {"tool": str, "args": Any, "content": str, "ok": bool}
"""
from __future__ import annotations

import json
from typing import Any, Dict

# Budget caps for full render. per-tool 8KB / total 50KB — large enough to keep
# a readable code body, small enough to bound the context blowup on resume.
PER_RESULT_CHARS = 8000
TOTAL_CHARS = 50000
# args are summarized, not dumped raw — a few large read_file args would
# otherwise exhaust the budget instantly.
MAX_ARGS_CHARS = 400

_HEADER = "[Interrupted tool-loop results — full detail preserved]"


def render_interrupt_tool_results(
    tool_results: list[Dict[str, Any]],
    *,
    per_result_chars: int = PER_RESULT_CHARS,
    total_chars: int = TOTAL_CHARS,
) -> str:
    """Render interrupted tool_results within a byte budget.

    Returns "" for empty/None input so the caller can falsy-skip rendering.
    """
    if not tool_results:
        return ""
    parts: list[str] = []
    spent = 0
    for i, tr in enumerate(tool_results, 1):
        if spent >= total_chars:
            break
        name = tr.get("tool", "?")
        args = tr.get("args")
        ok = tr.get("ok")
        content = tr.get("content") or ""

        remaining = total_chars - spent
        cap = min(per_result_chars, remaining)
        if len(content) > cap:
            content = content[:cap] + f"\n…[truncated {len(content) - cap} chars]"

        # Summarize args: serialize dict/list compactly, str-cap everything.
        try:
            if isinstance(args, (dict, list)):
                arg_str = json.dumps(args, ensure_ascii=False, default=str)
            elif args is None:
                arg_str = ""
            else:
                arg_str = str(args)
        except Exception:
            arg_str = repr(args)
        if len(arg_str) > MAX_ARGS_CHARS:
            arg_str = arg_str[:MAX_ARGS_CHARS] + "…"

        status = "ok" if ok else "FAIL"
        block = f"[{i}] {name} ({status})"
        if arg_str:
            block += f"\n  args: {arg_str}"
        if content:
            block += f"\n  result:\n{content}"
        parts.append(block)
        spent += len(block)

    if not parts:
        return ""
    head = (
        f"{_HEADER}\n"
        f"[{len(parts)} of {len(tool_results)} tool call(s) shown; "
        f"budget {spent}/{total_chars} chars]"
    )
    return head + "\n" + "\n".join(parts)
