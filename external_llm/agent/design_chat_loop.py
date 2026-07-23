"""
Lightweight Tool-Use Loop for Design Chat

Handles a single design chat turn:
  LLM call → if tool_calls, execute tools → loop → final text response

Includes write-capable tools and bash with danger gates.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import random
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

from external_llm.client import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMClientError,
    LLMConnectionError,
    LLMMessage,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMServerUnavailableError,
    ToolCallResponse,
)

from ._shared_utils import context_message_cap, estimate_tokens_from_msgs, preemptive_trim
from .agent_loop_types import AgentCancelled
from .agent_turn_pipeline import _cache_hit_ratio, _evict_for_loop
from .performance_metrics import get_global_collector
from .config.thresholds import config as _cfg
from .context_budget import (
    _is_context_length_error,
    _record_context_overflow,
    _resolve_context_limit,
    repair_tool_message_sequence,
)
from .insights_manager import (
    atomic_write_text,
    build_archive_index,
    drop_entry,
    insights_write_lock,
    load_active_insights_cached,
    load_insights_file,
    parse_insights,
    select_promotable_entries,
    serialize_insights,
)
from .insights_manager import (
    insights_path as _insights_manager_path,
    _active_invalidate,
)
from .plan_state import open_items as _plan_open_items
from .rag_searcher import _TOKENIZER, _bm25_score

_HAS_VECTOR_CACHE = False
try:
    from .vector_cache import HAS_FAISS, HAS_NUMPY, HAS_SENTENCE_TRANSFORMERS, VectorCacheManager
    _HAS_VECTOR_CACHE = HAS_SENTENCE_TRANSFORMERS and HAS_NUMPY and HAS_FAISS
except ImportError:
    pass


logger = logging.getLogger(__name__)

# Design-chat sampling temperature: randomized ONCE per process (asi launch)
# and held until the process exits. Per-call randomization was unintended —
# every LLM call in a session should sample at the same temperature.
_PROCESS_TEMPERATURE = random.uniform(0.0, 0.3)

# z.ai endpoint failover (connection-error retry).  z.ai exposes the same GLM
# backend behind two protocol facades — an Anthropic-compatible endpoint
# (ZAIAnthropicClient) and an OpenAI-compatible one (ZAIClient).  When one
# facade is unreachable (timeout / cannot-connect) the other may still answer,
# so a connection-level retry flips between them.  The failover client gets a
# shorter timeout so alternating attempts can't stack N × the (long) primary
# timeout into a multi-minute hang; the retry fires almost immediately because
# switching facades — not waiting — is the mitigation.
_ZAI_FAILOVER_TIMEOUT = 60      # seconds; cap for the flipped sibling client
_ZAI_FAILOVER_RETRY_DELAY = 1   # seconds; near-zero backoff after an endpoint flip

# Plan completion gate: when the model ends a turn while its update_plan
# checklist still has open items, it is nudged ONCE to resume or to mark
# the items skipped/blocked with reasons. The nudge explicitly forbids the
# "continue" narrative preamble — the model must act directly.
# After one nudge without resolution, the turn ends with the open items
# surfaced honestly (no further nagging).
_PLAN_GATE_MAX_NUDGES = 1

# ── Shared fallback utilities (module-level for reuse by routes/design_chat.py) ──


def _strip_tool_messages(msgs: list[LLMMessage]) -> list[LLMMessage]:
    """
    Strip tool-related messages for plain chat() fallback.

    - Remove role="tool" messages entirely
    - Convert assistant messages with tool_calls to plain text
    """
    plain = []
    for m in msgs:
        if m.role == "tool":
            continue  # skip tool result messages
        if m.role == "assistant" and getattr(m, "tool_calls", None):
            # Convert to plain assistant message with tool call summary
            tool_names = []
            for tc in m.tool_calls:
                fn = tc.get("function", {})
                tool_names.append(fn.get("name", "?"))
            summary = m.content or ""
            if tool_names:
                summary += f"\n[Code analysis performed: {', '.join(tool_names)}]"
            plain.append(LLMMessage(
                role="assistant",
                content=summary.strip(),
                reasoning_content=getattr(m, "reasoning_content", None),
            ))
        else:
            plain.append(m)
    return plain


def _apply_context_hard_cap(
    messages: list[LLMMessage],
    model: str,
    tool_schemas: Any = None,
) -> list[LLMMessage]:
    """
    Context hard cap guard: trim messages if they exceed the model's context limit.

    Pre-allocates room for output tokens (and tool schemas if provided) to prevent
    HTTP 400 errors on small-window models.

    Returns the (possibly trimmed) message list.
    """
    _ctx_limit = _resolve_context_limit(model)
    _safety_margin = _cfg.tokens.CONTEXT_HARD_CAP_SAFETY_MARGIN
    _cap = context_message_cap(_ctx_limit, _safety_margin, tool_schemas)
    _est = estimate_tokens_from_msgs(messages)
    if _est > _cap:
        _before = len(messages)
        messages = preemptive_trim(messages, max_tokens=_cap, preserve_last=2, tag="DESIGN_CHAT_PREEMPTIVE_TRIM")
        # preemptive_trim is a count-based slice and can split an
        # assistant(tool_calls) ↔ role="tool" pair, leaving orphaned tool messages
        # whose preceding assistant was trimmed. Such orphans cause HTTP 400
        # ("orphaned tool_result" / "messages must alternate") on OpenAI/DeepSeek.
        # Repair the sequence — mirrors the guard AgentLoop applies after its trim.
        messages = repair_tool_message_sequence(messages)
        _after = len(messages)
        logger.warning(
            "[DESIGN_CHAT_CONTEXT_HARD_CAP] estimated %d tokens > cap %d "
            "(limit=%d, reserved %d for output/tool-schemas); preemptive_trim: %d->%d messages",
            _est, _cap, _ctx_limit, _ctx_limit - _cap,
            _before, _after,
        )
    return messages



def _extract_provider_message(raw: str) -> str:
    """Best-effort pull of the provider's ``{"error":{"message": ...}}`` text
    out of a raw error string.  Returns "" when nothing readable is found."""
    start = raw.find("{")
    if start < 0:
        return ""
    try:
        obj = json.loads(raw[start:])
    except (ValueError, TypeError):
        return ""
    err = obj.get("error") if isinstance(obj, dict) else None
    if isinstance(err, dict) and err.get("message"):
        return str(err["message"]).strip()
    if isinstance(err, str) and err.strip():
        return err.strip()
    return ""


def _upstream_gateway_name(provider_msg: str) -> Optional[str]:
    """Detect a gateway *upstream* failure and return the gateway's name.

    Aggregating gateways (e.g. opencode/zen "Console Go") that proxy to a
    third-party model provider wrap every upstream fault in a fixed envelope:
    ``Error from provider (<gateway>): <reason>`` — e.g. "Error from provider
    (Console Go): Upstream request failed".  That prefix is machine-generated by
    the gateway (not free-form user prose), so matching it reliably tells an
    upstream/provider-side fault apart from a fault in *our* request — letting us
    show a distinct, non-alarming line instead of the catch-all "An error
    occurred".  Returns the gateway name (may be "" if unparenthesised), or
    ``None`` when *provider_msg* is not a gateway upstream envelope.
    """
    if not provider_msg.lstrip().lower().startswith("error from provider"):
        return None
    start = provider_msg.find("(")
    end = provider_msg.find(")", start + 1)
    if start >= 0 and end > start:
        return provider_msg[start + 1:end].strip()
    return ""


def _user_facing_llm_error(e: Exception) -> str:
    """Map an LLMClientError to a concise, user-facing message.

    These errors are caused by the LLM service (rate limit, auth, quota, server
    overload), not by a bug in our code — so we surface a short, actionable line
    instead of a raw provider JSON blob or a stack trace.
    """
    provider_msg = _extract_provider_message(str(e))
    if isinstance(e, LLMRateLimitError):
        if e.error_code == 1305:
            base = "⚠️ The GLM server is currently busy (code 1305). Please try again in a moment."
        elif e.error_code == 1302:
            base = "⚠️ API request limit exceeded (code 1302). Please wait a moment and try again."
        else:
            base = "⚠️ The LLM server is temporarily busy (rate limit). Please try again in a moment."
    elif isinstance(e, (LLMServerUnavailableError, LLMConnectionError)):
        base = "⚠️ Cannot connect to the LLM server. Check your network or the server status and try again."
    elif isinstance(e, LLMAuthenticationError):
        base = "⚠️ LLM API authentication failed. Please check your API key."
    elif isinstance(e, LLMQuotaExceededError):
        base = "⚠️ LLM API credit/quota exhausted. Please check your billing status."
    elif (_gw := _upstream_gateway_name(provider_msg)) is not None:
        # A gateway upstream failure (distinct from a 429 rate limit, which is
        # matched above): the fault is on the provider/upstream side, not the
        # user's setup. Surfaced as its own line so it does not read as a
        # generic "something went wrong" or a config problem on our end.
        _where = f" ({_gw})" if _gw else ""
        base = (
            f"⚠️ The upstream model provider{_where} returned an error — this is a "
            "provider-side issue, not your setup. Please try again shortly or switch models."
        )
    else:
        base = "⚠️ An error occurred while processing the LLM request."
    return f"{base}\n(server message: {provider_msg})" if provider_msg else base


def _fallback_plain_chat(
    messages: list[LLMMessage],
    llm_client: Any,
    model: str,
    max_tokens: Optional[int] = None,
    token_callback: Optional[Callable] = None,
) -> dict[str, Any]:
    """
    Fallback plain chat without tool calling.

    Strips tool messages, applies context hard cap guard, calls plain chat.
    Shared by design_chat_loop._process_tool_call and routes/design_chat.py.

    Returns dict with: content, reasoning, error (bool), tokens_used,
    prompt_tokens, completion_tokens, cache_read_tokens, provider.
    """
    plain_msgs = _strip_tool_messages(messages)
    plain_msgs = _apply_context_hard_cap(plain_msgs, model)
    try:
        _fb_t0 = time.monotonic()
        response = llm_client.chat(
            messages=plain_msgs, model=model,
            temperature=_PROCESS_TEMPERATURE,
            max_tokens=max_tokens,
            token_callback=token_callback,
        )
        _fb_elapsed = time.monotonic() - _fb_t0
        _content = response.content or ""
        _reasoning = ""
        try:
            raw = response.raw_response or {}
            msg_obj = raw.get("choices", [{}])[0].get("message", {})
            _reasoning = msg_obj.get("reasoning_content", "") or ""
        except Exception:
            pass
        return {
            "content": _content,
            "reasoning": _reasoning,
            "error": False,
            "tokens_used": response.tokens_used or 0,
            "prompt_tokens": getattr(response, "prompt_tokens", 0) or getattr(response, "tokens_used", 0) or 0,
            "completion_tokens": getattr(response, "completion_tokens", 0) or 0,
            "cache_read_tokens": getattr(response, "cache_read_input_tokens", 0) or 0,
            "cache_creation_tokens": getattr(response, "cache_creation_input_tokens", 0) or 0,
            "provider": getattr(response, "provider", "") or "",
            "execution_time_ms": round(_fb_elapsed * 1000),
        }
    except Exception as e:
        # Re-raise ALL service-side LLM errors (auth/quota/rate-limit/server/
        # connection) so the caller's canonical error→message mapping applies
        # the proper actionable message and error_type. Swallowing any of these
        # degrades them into a generic "An error occurred" — e.g. a quota error
        # (LLMQuotaExceededError) became indistinguishable from a bug. Only truly
        # unexpected (non-LLMClientError) exceptions fall back to plain chat.
        # (Insight A38: defense-depth parity — match _respond_impl's
        # `except LLMClientError` handler, not a stale 2-name tuple.)
        from external_llm.client import LLMClientError
        if isinstance(e, LLMClientError):
            raise
        return {
            "content": f"An error occurred during LLM call: {e}",
            "reasoning": "",
            "error": True,
            "tokens_used": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "provider": "",
        }


# Tool names to keep in /general (chat) mode — excludes all code-related tools
_GENERAL_MODE_TOOLS: frozenset = frozenset({
    "save_insight",
    "delete_insight",
    "edit_insight",
    "search_web",
    "browser_action",
    "ask_user",
    "web_fetch",
    "search_design_history",
    "bash",
    "read_image",
})


def _parse_text_tool_calls(content: str) -> list[dict[str, Any]]:
    """Parse tool calls from LLM text output (text-mode models like qwen2.5-coder).

    Supported formats:
      1. {"name": "...", "arguments": {...}}          ← qwen2.5-coder, gemma
      2. {"tool": "...", "args": {...}}               ← internal format
      3. {"type": "function", "function": {...}}      ← OpenAI-style
      4. Arrays of any of the above
      5. ```json blocks``` or JSON embedded in free text
    """

    tool_calls: list[dict[str, Any]] = []

    def _repair_brackets(text: str) -> str:
        result, stack, in_str, esc = [], [], False, False
        for ch in text:
            if esc:
                result.append(ch); esc = False; continue
            if ch == "\\" and in_str:
                result.append(ch); esc = True; continue
            if ch == '"':
                in_str = not in_str; result.append(ch); continue
            if in_str:
                result.append(ch); continue
            if ch in ("{", "["):
                stack.append(ch); result.append(ch)
            elif ch == "}":
                if stack and stack[-1] == "{":
                    stack.pop(); result.append(ch)
            elif ch == "]":
                if stack and stack[-1] == "[":
                    stack.pop(); result.append(ch)
            else:
                result.append(ch)
        for opener in reversed(stack):
            result.append("}" if opener == "{" else "]")
        return "".join(result)

    def _try_json(text: str):
        try:
            return json.loads(text)
        except (TypeError, ValueError):
            pass
        repaired = _repair_brackets(text)
        try:
            return json.loads(repaired)
        except (TypeError, ValueError):
            return None

    def _normalize(data: dict) -> Optional[dict[str, Any]]:
        if not isinstance(data, dict):
            return None
        idx = len(tool_calls)
        # {"name": ..., "arguments": {...}}
        if "name" in data and "arguments" in data:
            args = data["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (TypeError, ValueError):
                    args = {}
            return {"id": f"text_{idx}", "name": data["name"], "args": args if isinstance(args, dict) else {}}
        # {"tool": ..., "args": {...}}
        if "tool" in data:
            _args = data.get("args", {})
            return {"id": f"text_{idx}", "name": data["tool"], "args": _args if isinstance(_args, dict) else {}}
        # {"tool_name": ..., "params": {...}}  ← some Ollama models (e.g. gemma)
        if "tool_name" in data:
            args = data.get("params", data.get("arguments", data.get("args", {})))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (TypeError, ValueError):
                    args = {}
            return {"id": f"text_{idx}", "name": data["tool_name"], "args": args if isinstance(args, dict) else {}}
        # {"type": "function", "function": {"name": ..., "arguments": ...}}
        if data.get("type") == "function" and "function" in data:
            func = data["function"]
            if isinstance(func, dict) and "name" in func:
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (TypeError, ValueError):
                        args = {}
                return {"id": f"text_{idx}", "name": func["name"], "args": args if isinstance(args, dict) else {}}
        return None

    content = (content or "").strip()
    if not content:
        return tool_calls

    # Try whole content as JSON first
    if content[:1] in ("{", "["):
        parsed = _try_json(content)
        if parsed is not None:
            if isinstance(parsed, dict):
                n = _normalize(parsed)
                if n:
                    tool_calls.append(n)
                    return tool_calls
            elif isinstance(parsed, list):
                for item in parsed:
                    n = _normalize(item)
                    if n:
                        tool_calls.append(n)
                if tool_calls:
                    return tool_calls

    # Try fenced ```json blocks. ``content.split('```')`` yields alternating
    # OUTSIDE (even index) / INSIDE (odd index) segments — only odd-indexed
    # segments are actual fenced code blocks. Restricting the scan to them avoids
    # misexecuting a JSON-shaped example written in free text (e.g. a text-mode
 # model that writes 'example: {"name": "edit_text", ...}' outside any fence would
    # otherwise be parsed and run as a real call here). Free-text JSON is still
    # recovered by the stage-3 fallback below, but ONLY when no fenced call was
    # found — so a real fenced call takes precedence over a free-text lookalike.
    for _seg_i, block in enumerate(content.split('```')):
        if _seg_i % 2 == 0:
            continue  # outside a fence — leave to the stage-3 free-text fallback
        block = block.strip()
        idx = block.find('{')
        if idx >= 0:
            parsed = _try_json(block[idx:].strip())
            if parsed is not None:
                n = _normalize(parsed)
                if n:
                    tool_calls.append(n)
    if tool_calls:
        return tool_calls

    # Scan for JSON objects in free text
    start = 0
    while start < len(content):
        ob = content.find("{", start)
        if ob == -1:
            break
        depth, i, in_str, esc = 1, ob + 1, False, False
        while i < len(content) and depth > 0:
            ch = content[i]
            if esc:
                esc = False
            elif ch == "\\" and in_str:
                esc = True
            elif ch == '"':
                in_str = not in_str
            elif not in_str:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
            i += 1
        if depth == 0:
            parsed = _try_json(content[ob:i])
            if parsed is not None:
                n = _normalize(parsed)
                if n:
                    tool_calls.append(n)
                    # Recognized tool call fully consumed at [ob, i): skip past
                    # it. Re-scanning inside only re-parses its own arg braces
                    # (which _normalize rejects), so advancing by 1 here is pure
                    # O(n^2) waste on large pastes. Unrecognized wrappers still
                    # fall through to ob+1 so nested tool calls stay reachable.
                    start = i
                    continue
        start = ob + 1

    return tool_calls



def build_handoff_context(
    *,
    decisions: Optional[list[dict[str, Any]]] = None,
    implementation_spec: Optional[dict[str, Any]] = None,
) -> tuple:
    """Assemble the prior_context string for design-chat → implementation handoff.

    Returns: (prior_context: str, preview_decisions: list[str], preview_target_files: list[str])

    Implementation spec JSON is used as structured data to generate prebuilt_spec_for_planner.

    Note: ``preview_target_files`` (3rd tuple element) is always ``[]`` and
    retained only for backward-compatible 3-tuple unpacking at the call site
    (``prior_context, _, _ = build_handoff_context(...)``).
    Called identically from both asi and routes/design_chat.py.
    """
    _ctx_parts: list[str] = []

    # Implementation Spec — JSON transport, for generating prebuilt_spec_for_planner
    if implementation_spec:
        _ctx_parts.append(
            "=== IMPLEMENTATION SPEC ===\n"
            + json.dumps(implementation_spec, ensure_ascii=False)
            + "\n=== END IMPLEMENTATION SPEC ==="
        )

    _handoff_context = "\n\n".join(_ctx_parts)
    _preview_decisions = [
        _d.get("decision", "") for _d in (decisions or [])
        if isinstance(_d, dict) and _d.get("decision")
    ]
    return _handoff_context, _preview_decisions, []


# _PLANNER_SWITCH_SIGNAL — removed (planner lane deactivated)
# _PLANNER_SWITCH_SIGNAL = "__PLANNER_SWITCH__"


def _save_insight_to_file(repo_root: str, insight: str, category: str) -> str:
    """Append a timestamped insight to .asicode/design_insights.md.

    The append is serialized by :func:`insights_write_lock` so it cannot be lost
    to a concurrent compactor's read-modify-write — without the lock, a
    compactor that read the file just before this append would rewrite it
    without the new entry, silently dropping it ("0 durable loss" violated).
    """
    insights_path = _insights_manager_path(repo_root)
    os.makedirs(os.path.dirname(insights_path), exist_ok=True)

    timestamp = time.strftime("%Y-%m-%d %H:%M %z", time.localtime())

    with insights_write_lock(repo_root):
        # Create file with header if new
        if not os.path.exists(insights_path):
            with open(insights_path, "w", encoding="utf-8") as f:
                f.write("# Design Chat Insights\n\n"
                        "Discoveries and insights saved by the design chat LLM across sessions.\n\n")

        with open(insights_path, "a", encoding="utf-8") as f:
            f.write(f"### [{category}] {timestamp}\n{insight}\n\n")

    _active_invalidate(repo_root)  # drop 0c content cache — active file just grew
    return f"Insight saved to .asicode/design_insights.md [{category}]: {insight[:100]}"


def _find_entry_by_match(
    entries: list, match_str: str,
) -> tuple[Optional[int], Optional[str]]:
    """Find an insight entry whose ``header_line`` contains ``match_str``.

    Returns ``(index, header_line)`` of the single match, or
    ``(None, error_message)`` if zero or multiple matches.

    Disambiguation uses :attr:`InsightEntry.category` and :attr:`InsightEntry.body`
    (both populated by :func:`parse_insights`) rather than re-splitting the raw
    header line. The previous implementation split on ``"["`` to recover the
    category, which raised ``IndexError`` for hand-written headers that have no
    ``[category]`` bracket — returning a useless error to the LLM instead of the
    candidate list.
    """
    matches: list[tuple[int, Any]] = []
    for i, entry in enumerate(entries):
        if match_str in entry.header_line:
            matches.append((i, entry))
    if len(matches) == 0:
        return None, (
            f"No insight found matching \"{match_str}\". "
            f"Check the design-chat context for available entries."
        )
    if len(matches) > 1:
        return None, (
            f"Multiple insights match \"{match_str}\". "
            f"Be more specific — available matches: "
            + "; ".join(
                f"[{entry.category or 'uncategorized'}] {entry.body.strip()[:60]}"
                for _, entry in matches
            )
            + "."
        )
    return matches[0][0], None


def _delete_insight(repo_root: str, entry_match: str) -> str:
    """Delete an insight entry by matching its header line. Returns a status message.

    The whole read-modify-write is serialized by :func:`insights_write_lock` so a
    concurrent save/append cannot be lost when this rewrite lands.
    """
    path = _insights_manager_path(repo_root)
    with insights_write_lock(repo_root):
        if not os.path.exists(path):
            return "Error: No design insights file found."

        content = load_insights_file(repo_root)
        if not content:
            return "Error: Design insights file is empty."

        preamble, entries = parse_insights(content)
        idx, err = _find_entry_by_match(entries, entry_match)
        if err:
            return f"Error: {err}"

        removed_header = entries[idx].header_line
        new_entries = drop_entry(entries, idx + 1)  # drop_entry uses 1-based index
        new_content = serialize_insights(preamble, new_entries)
        atomic_write_text(path, new_content)

    _active_invalidate(repo_root)  # drop 0c content cache — active file just changed
    return f"✅ Deleted insight: {removed_header}"


def _edit_insight(
    repo_root: str, entry_match: str,
    new_insight: str, new_category: Optional[str] = None,
) -> str:
    """Edit (replace body of) an insight entry. Returns a status message.

    The whole read-modify-write is serialized by :func:`insights_write_lock` so a
    concurrent save/append cannot be lost when this rewrite lands.
    """
    path = _insights_manager_path(repo_root)
    with insights_write_lock(repo_root):
        if not os.path.exists(path):
            return "Error: No design insights file found."

        content = load_insights_file(repo_root)
        if not content:
            return "Error: Design insights file is empty."

        preamble, entries = parse_insights(content)
        idx, err = _find_entry_by_match(entries, entry_match)
        if err:
            return f"Error: {err}"

        entry = entries[idx]
        old_header = entry.header_line

        if new_category:
            # Rebuild the header line with the new category
            timestamp_part = old_header.split("] ", 1)[-1] if "] " in old_header else ""
            new_header = f"### [{new_category}] {timestamp_part}"
        else:
            new_header = old_header

        entry.lines = [new_header + "\n", new_insight.rstrip("\n") + "\n\n"]
        entry.header_line = new_header
        entry.category = new_category if new_category else entry.category

        new_content = serialize_insights(preamble, entries)
        atomic_write_text(path, new_content)

    _active_invalidate(repo_root)  # drop 0c content cache — active file just changed
    return f"✅ Edited insight: {old_header}"


def load_design_insights(repo_root: str, max_chars: int = 50000) -> str:
    """Load design insights for context injection. Returns empty string if none.

    Returns the **stable** insights prefix: the active file (full) plus the
    Layer 2 archive header index. This block is placed EARLY in the prompt
    prefix (block 0c) so it stays cached across turns — it only changes when the
    insights/archive files themselves change (save_insight, ``/insights compact``
    demotion, ``/insights archive restore|drop``).

    The **turn-volatile** Layer 3 (relevant archived entries promoted back for
    THIS task) is intentionally NOT included here — see
    :func:`load_promoted_insights`. Layer 3 depends on the recent user turns and
    changes every turn, so embedding it in 0c would invalidate the prompt cache
    from 0c onward (compressed summary + all verbatim turns). It is injected
    separately at a late, always-uncached position by
    ``build_context_messages``.

    The max_chars parameter is accepted for backward compatibility but no longer
    applied — discarding information early just wastes tokens.

    Two-tier context model recap:
    - **Layer 1 (active, always-on)**: full active ``design_insights.md``.
    - **Layer 2 (archive index, always-on)**: compact header-only index of the
      newest archived entries, so the agent knows what was demoted and can
      retrieve it (``/insights archive restore <n>``).
    - **Layer 3 (promotion, per-turn)**: see :func:`load_promoted_insights`.
    """

    # Layer 1: the ACTIVE design_insights.md content, signature-cached on
    # ``(mtime_ns, size, write_version)`` (same scheme as the archive cache in
    # insights_manager). This skips the per-turn open()+read() — mirroring the
    # mtime cache already used for project.md (context_manager) — and keeps block
    # 0c byte-stable. Every active-file writer (save_insight / edit / delete /
    # compact-demote) calls ``_active_invalidate``, so a change is detected
    # instantly; a plain mtime/size change also invalidates.
    content = load_active_insights_cached(repo_root).strip()
    if not content:
        return ""
    try:
        if len(content) > max_chars:
            logger.debug("[TRUNCATION_VIOLATION] design_insights=%d chars exceeds %d, but passing full content", len(content), max_chars)

        # Layer 2: always-on archive header index (bounded — ~tens of bytes/line).
        # Only changes when the archive file changes (demotion/restore/drop), so
        # it is safe to keep inside the cached 0c prefix. (Already signature-cached
        # inside build_archive_index.)
        try:
            _idx = build_archive_index(repo_root)
            if _idx:
                content += "\n" + _idx.strip()
        except Exception:
            pass  # non-critical

        return content
    except Exception:
        return ""  # non-critical — never block execution


def load_promoted_insights(repo_root: str, task_query: str) -> str:
    """Return the Layer 3 promoted-insights block for THIS turn, or ``""``.

    Relevant archived entries (token-set overlap with ``task_query``) are
    promoted back in full so an old-but-relevant invariant resurfaces exactly
    when needed.

    **Cache contract**: this is **turn-volatile** (depends on the recent user
    turns), so callers MUST inject it at a LATE position in the message list —
    after the cached verbatim-turns prefix, never in the early 0c insights
    block. Injecting it in 0c would invalidate the prompt cache from 0c onward
    on every turn.

    Returns ``""`` when there is nothing to promote (no archive, no query, no
    overlap) so irrelevant turns pay zero cost AND the message list is
    byte-identical to a no-promotion turn (cache fully preserved).
    """
    if not task_query or not task_query.strip():
        return ""
    try:
        _promoted = select_promotable_entries(repo_root, task_query)
        if not _promoted:
            return ""
        _blocks = [
            "",
            "=== PROMOTED FROM ARCHIVE (relevance to current task) ===",
        ]
        for _e in _promoted:
            _blocks.append("".join(_e.lines).rstrip())
        return "\n" + "\n".join(_blocks) + "\n"
    except Exception:
        return ""  # non-critical — never block injection

@dataclass
class DesignChatResult:
    """Result of a design chat turn."""
    content: str = ""
    reasoning_content: str = ""
    tool_calls_made: list[dict[str, Any]] = field(default_factory=list)
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    last_call_prompt_tokens: int = 0
    last_call_completion_tokens: int = 0
    last_call_cache_read_tokens: int = 0
    last_call_cache_creation_tokens: int = 0
    provider: str = ""
    # Each entry: {"tool": str, "args": dict, "content": str, "ok": bool}
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    is_error: bool = False
    # True when content is an error message (e.g. LLM timeout) — main.py sends as design_error event
    hit_max_iterations: bool = False
    # True when tool-call budget exhausted (max_tool_iterations reached).
    # IPC worker maps this to status="max_turns" so the orchestrator's
    # max_turns_reached signal propagates correctly (B6 / turn 13122).
    total_llm_calls: int = 0
    # Total LLM API calls made (tool-loop iterations + final text-only call).
    # Counts EVERY call (including max_iterations fallback and error-fallback
    # plain chat), whereas tool_calls_made only counts tool-executing turns.
    # Useful for accurate turn-counting in IPC worker reports and dashboards.
    error_type: str = ""
    # Set to "auth" when the error is LLMAuthenticationError — allows the REPL
    # to offer an interactive API key re-entry prompt instead of just showing a
    # static error message.

class _SessionSearcher:
    """
    Lightweight BM25 searcher for design session turns / decisions / summary.

    Uses the same CodeTokenizer and BM25 formula as RAGSearcher (file searcher),
    but works on in-memory session content rather than repo files.
    Optionally uses VectorCacheManager for semantic re-ranking.

    Usage:
        searcher = _SessionSearcher()
        searcher.index_docs([(id1, text1), (id2, text2), ...])
        results = searcher.search("query", top_k=10)
        # Each result: {id, text, score}
    """

    def __init__(self, session_prefix: str = "", vector_cache: Optional[Any] = None) -> None:
        self._session_prefix = session_prefix
        self._doc_ids: list[Any] = []
        self._doc_token_counts: list[dict[str, int]] = []
        self._doc_lengths: list[int] = []
        self._doc_texts: list[str] = []
        self._df: dict[str, int] = {}
        self._avgdl: float = 0.0
        self._n_docs: int = 0
        # Optional vector cache for semantic re-ranking.
        #
        # A caller may pass a SHARED VectorCacheManager so that multiple
        # searchers built for the same search_design_history() call reuse one
        # index load instead of each triggering a fresh ~77ms on-disk FAISS +
        # metadata load (3 searchers × 77ms = 231ms otherwise). When None (the
        # default, used by RAGSearcher-style standalone callers), fall back to
        # creating a private cache for backward compatibility.
        if vector_cache is not None:
            self._vector_cache = vector_cache
        else:
            self._vector_cache = None
            if _HAS_VECTOR_CACHE:
                # Use the process-wide memoised VCM so standalone callers
                # (those that construct _SessionSearcher without a shared cache)
                # also avoid the ~77ms on-disk reload per call, and so all paths
                # share the same in-memory index + dirty tail.
                try:
                    self._vector_cache = _get_session_vcm()
                except Exception:
                    pass

    def index_docs(
        self,
        docs: list[tuple[Any, str]],
        pre_tokenized: Optional[list[tuple[Any, dict, int, str]]] = None,
        archive_sig: Optional[tuple] = None,
    ) -> None:
        """Index a list of (id, text) documents for BM25 retrieval.

        ``pre_tokenized`` (optional): already-tokenised docs as
        ``(id, token_count_dict, length, text)`` tuples — e.g. from the archive
        BM25 cache — merged in WITHOUT re-tokenising.  The df/avgdl aggregation
        is always recomputed over the full set (cheap O(n_docs)), so passing a
        cached prefix keeps results identical to a fresh build while skipping the
        expensive ``_TOKENIZER.tokenize()`` pass (measured ~3.9s for a 14k-turn
        archive; the dominant cost of search_design_history on long sessions).

        ``archive_sig`` (optional): when set, pre-tokenized entries came from an
        archive BM25 cache hit.  The vector-cache insertion for those entries is
        skipped on repeat searches of the same (unchanged) archive via the
        module-level ``_VECTOR_CACHE_INDEXED_ARCHIVES`` set, avoiding redundant
        ``add_document`` calls (14k SHA-256 hashes + 14k model loads per repeat).
        """
        self._doc_ids.clear()
        self._doc_token_counts.clear()
        self._doc_lengths.clear()
        self._doc_texts.clear()
        self._df.clear()
        total_len = 0

        entries: list[tuple[Any, dict, int, str]] = []
        n_pre_tokenized = len(pre_tokenized) if pre_tokenized else 0
        if pre_tokenized:
            entries.extend(pre_tokenized)
        for _id, text in docs:
            tokens = _TOKENIZER.tokenize(text)
            if not tokens:
                continue
            tc: dict[str, int] = {}
            for t in tokens:
                tc[t] = tc.get(t, 0) + 1
            entries.append((_id, tc, len(tokens), text))

        # Determine whether to skip vector-cache insertion for pre-tokenised entries
        skip_vector_archive = False
        if archive_sig is not None and n_pre_tokenized > 0:
            with _VECTOR_CACHE_LOCK:
                if archive_sig in _VECTOR_CACHE_INDEXED_ARCHIVES:
                    skip_vector_archive = True

        for i, (_id, tc, length, text) in enumerate(entries):
            self._doc_ids.append(_id)
            self._doc_token_counts.append(tc)
            self._doc_lengths.append(length)
            self._doc_texts.append(text)
            total_len += length
            for t in tc:
                self._df[t] = self._df.get(t, 0) + 1

            # Also index into vector cache if available
            if self._vector_cache is not None:
                # Skip pre-tokenised (archive) entries when already indexed
                if skip_vector_archive and i < n_pre_tokenized:
                    continue
                try:
                    doc_key = f"{self._session_prefix}{_id}"
                    # Serialize FAISS mutation: the shared VCM's index is touched
                    # from multiple threads (search_design_history runs in the
                    # shared pool — see the parallel dispatch branch). Per-doc
                    # scope keeps contention minimal while preventing an
                    # interleaved add/search on the same IndexFlatIP.
                    with _SESSION_VCM_IO_LOCK:
                        self._vector_cache.add_document(doc_key, text)
                except Exception:
                    pass

        # Mark archive sig as vector-indexed for future skips (always record
        # after a full pass — the sig is now always passed so that the second
        # search benefits from the skip; the first search always does full work).
        if archive_sig is not None and n_pre_tokenized > 0 and self._vector_cache is not None:
            with _VECTOR_CACHE_LOCK:
                _VECTOR_CACHE_INDEXED_ARCHIVES[archive_sig] = None
                # Safety cap: prevent unbounded growth across process lifetime.
                # FIFO eviction (oldest-inserted first) is deterministic — unlike
                # the previous set.pop() which dropped an arbitrary sig, possibly
                # a still-active one, forcing a redundant re-index on the next
                # search. (LRU would need a move_to_end on every hit; insertion
                # order is a sufficient approximation here.)
                while len(_VECTOR_CACHE_INDEXED_ARCHIVES) > _VECTOR_CACHE_INDEXED_ARCHIVES_MAX:
                    _VECTOR_CACHE_INDEXED_ARCHIVES.popitem(last=False)

        self._n_docs = len(self._doc_ids)
        self._avgdl = total_len / self._n_docs if self._n_docs > 0 else 0.0

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Search indexed documents with BM25 (+ optional vector re-ranking)."""
        if self._n_docs == 0 or not query.strip():
            return []

        query_tokens = _TOKENIZER.tokenize(query)
        if not query_tokens:
            return []

        # ── BM25 ranking ──────────────────────────────────────────────────
        bm25_scored: list[tuple[float, int]] = []  # (score, doc_idx)
        for i in range(self._n_docs):
            score = _bm25_score(
                query_tokens,
                self._doc_token_counts[i],
                self._doc_lengths[i],
                self._df,
                self._n_docs,
                self._avgdl,
            )
            if score > 0.0:
                bm25_scored.append((score, i))
        bm25_scored.sort(key=lambda x: -x[0])
        bm25_rank = {doc_idx: rank for rank, (_, doc_idx) in enumerate(bm25_scored)}
        bm25_score_of = {doc_idx: s for s, doc_idx in bm25_scored}

        # ── Vector ranking (optional) ───────────────────────────────────────
        # index_docs() stored each doc under doc_key = f"{session_prefix}{_id}",
        # so the lookup must use the SAME prefixed key. A bare str(_id) never
        # matched (prefix defaults to "local"), which previously made the whole
        # semantic signal a silent no-op.
        vector_rank: dict[int, int] = {}
        if self._vector_cache is not None:
            try:
                # Serialize FAISS read: concurrent search_design_history calls
                # share this index (see _SESSION_VCM_IO_LOCK). Only the FAISS
                # call is held; the subsequent key_to_idx/ranking is unlocked.
                with _SESSION_VCM_IO_LOCK:
                    vec_results = self._vector_cache.search(
                        query, top_k=min(max(top_k * 2, 10), self._n_docs)
                    )
                key_to_idx = {
                    f"{self._session_prefix}{self._doc_ids[i]}": i
                    for i in range(self._n_docs)
                }
                rank = 0
                for vr in vec_results:
                    idx = key_to_idx.get(vr.get("file_path"))
                    if idx is not None and idx not in vector_rank:
                        vector_rank[idx] = rank
                        rank += 1
            except Exception:
                vector_rank = {}

        # ── Reciprocal Rank Fusion (re-rank only) ───────────────────────────
        # RRF is scale-free: the previous weighted sum (0.7*bm25 + 0.3*vec)
        # mixed unbounded BM25 (~0-15+) with cosine (0-1), so BM25 dominated by
        # ~20x and the vector term almost never reordered anything. RRF fuses on
        # rank, so the semantic signal meaningfully reorders the BM25 hits.
        #
        # Candidates stay the BM25-matching set: vector search re-ranks but does
        # not introduce zero-lexical-overlap docs (that would be semantic
        # *recall*, a separate behavior change). This matches the original
        # "re-rank top BM25 results" intent.
        if not vector_rank:
            ranked_idxs = [doc_idx for _, doc_idx in bm25_scored]
        else:
            RRF_K = 60.0
            rrf_scored: list[tuple[float, int]] = []
            for idx in bm25_rank:
                rrf = 1.0 / (RRF_K + bm25_rank[idx])
                if idx in vector_rank:
                    rrf += 1.0 / (RRF_K + vector_rank[idx])
                rrf_scored.append((rrf, idx))
            rrf_scored.sort(key=lambda x: -x[0])
            ranked_idxs = [idx for _, idx in rrf_scored]

        # Build results. The displayed "score" stays the interpretable BM25
        # value (0.0 for vector-only hits); RRF drives ordering only.
        results = []
        for doc_idx in ranked_idxs[:top_k]:
            results.append({
                "id": self._doc_ids[doc_idx],
                "text": self._doc_texts[doc_idx],
                "score": round(bm25_score_of.get(doc_idx, 0.0), 4),
            })

        return results





# ── Archive BM25 cache ──────────────────────────────────────────────────────
# search_design_history() re-tokenised the ENTIRE archived-turn set on every
# call (a long session's .archive.jsonl is ~14k turns / ~26MB → ~3.9s of
# `_TOKENIZER.tokenize()` per search, plus the 0.17s JSONL reparse).  The
# archive is append-only and changes only when the compressor flushes old turns,
# so a (session_id, size, mtime_ns) signature keys an LRU cache of the
# pre-tokenised per-doc term-frequency vectors.  On a hit the searcher is fed
# the cached vectors via ``index_docs(pre_tokenized=...)`` and only the small
# compressed-but-not-yet-archived active prefix is tokenised fresh.
#
# MEMORY TRADE-OFF: the cached value holds one tc-dict per non-empty archived
# turn.  For a 14k-turn archive this is a few hundred MB resident while the
# session is cached.  The cap bounds the worst case; tune
# ``_ARCHIVED_BM25_CACHE_MAX`` (default 2: the current + most-recent other
# session) if memory matters more than cross-session search latency.
_ARCHIVED_BM25_CACHE: "OrderedDict[tuple, list[tuple[Any, dict, int, str]]]" = OrderedDict()
_ARCHIVED_BM25_CACHE_LOCK = threading.Lock()
"""Serialises all access to ``_ARCHIVED_BM25_CACHE`` (read + write).

``search_design_history`` is a read-tool: it can run concurrently in the
shared pool.  Without a lock, concurrent ``.get()`` / ``.move_to_end()`` /
``.__setitem__()`` / ``.popitem()`` calls can corrupt the internal
``OrderedDict`` doubly-linked list.  The lock is held only for the brief
cache-lookup / cache-write sections — the expensive tokenisation happens
outside it.
"""
_ARCHIVE_SMALL_SKIP = 2000
"""Small archives (≤ this many turns) skip BM25 caching — tokenisation is fast enough."""


def _parse_cache_max(raw: str | None = None) -> int | float:
    """Parse *ASICODE_ARCHIVED_BM25_CACHE_MAX* into a cache-cap value.

    Returns ``>= 1`` |int| or ``float("inf")`` for unlimited (negative /
    ``"inf"`` / ``"unlimited"``).  Empty, missing, or non-numeric input
    degrades to the safe default ``2`` so that an invalid env var never
    crashes import.
    """
    if raw is None:
        raw = os.environ.get("ASICODE_ARCHIVED_BM25_CACHE_MAX")
    if raw is None:
        return 2
    raw = raw.strip()
    if not raw:
        return 2
    raw_lower = raw.lower()
    if raw_lower in ("inf", "unlimited", "-1"):
        return float("inf")
    try:
        n = int(raw)
    except ValueError:
        logger.warning("invalid ASICODE_ARCHIVED_BM25_CACHE_MAX=%r, using default 2", raw)
        return 2
    return float("inf") if n < 0 else max(1, n)


_ARCHIVED_BM25_CACHE_MAX = _parse_cache_max()

_VECTOR_CACHE_INDEXED_ARCHIVES: "OrderedDict[tuple, None]" = OrderedDict()
"""Archive signatures whose documents have already been indexed into the vector cache.

The BM25 cache avoids re-tokenising the archive, but ``index_docs`` was still calling
``vector_cache.add_document`` for every archived turn on every search_design_history()
call — 14k SHA-256 hashes + 14k ``_ensure_model_loaded`` on every repeat search.

This set gates that: once a particular archive sig has been vector-indexed once in
this process lifetime, repeat searches of the same (unchanged) archive skip the
vector-insertion loop for the archived portion entirely.  Resets on process restart
(ephemeral); archive change (different sig) triggers a fresh index.
"""
_VECTOR_CACHE_LOCK = threading.Lock()
_VECTOR_CACHE_INDEXED_ARCHIVES_MAX = 100
"""Safety cap on ``_VECTOR_CACHE_INDEXED_ARCHIVES`` growth (process-lifetime set)."""


_SHARED_SESSION_VCM: Optional[Any] = None
"""Process-wide memoised ``VectorCacheManager`` for ``.asicode/session_vector_cache``.

Previously each ``search_design_history()`` call constructed a fresh
``VectorCacheManager`` (loading ``faiss_index.bin`` + the ~23MB
``metadata.json`` from disk, ~77ms each) and relied on ``__del__`` to flush the
dirty (<100-doc) tail back to disk before the NEXT call reloaded it.  That flush
is unreliable: an exception traceback can pin the frame local (preventing
prompt refcount destruction) or a GC cycle can delay ``__del__``, so the next
call's reload would miss the tail — and the archive-sig skip gate
(``_VECTOR_CACHE_INDEXED_ARCHIVES``) would then never re-insert it, silently
dropping the most-recent archived turn from vector re-ranking for the rest of
the process.

A single shared instance sidesteps the whole disk round-trip: the dirty tail
stays in the SAME in-memory index for the process lifetime, so the skip gate
reads against live state and inter-call persistence is no longer a correctness
dependency (disk is still checkpointed every 100 docs + at process exit).
"""

_SHARED_SESSION_VCM_LOCK = threading.Lock()
"""Guards creation of ``_SHARED_SESSION_VCM`` (double-checked init)."""

_SESSION_VCM_IO_LOCK = threading.Lock()
"""Serialises FAISS index mutation/read on the shared session VCM.

``search_design_history`` is a read-tool and dispatches concurrently in the
shared pool (a multi-tool batch runs read-tools in parallel threads — see the
parallel dispatch branch in ``_process_tool_call``).  The shared VCM's
``IndexFlatIP`` is therefore mutated (``add_document``) and read (``search``)
from multiple threads; FAISS add/search are not safe to interleave on one
index, so both operations take this lock.  Held only around the FAISS call —
BM25 ranking (the primary signal) and the per-doc dedup map are untouched, so
contention stays minimal (and the archive-sig gate means each archive is
indexed once per process, so ``add_document`` is rare after warmup).
"""


def _get_session_vcm() -> Optional[Any]:
    """Return the process-wide shared session vector cache (created once).

    Returns ``None`` when the vector stack is unavailable
    (``_HAS_VECTOR_CACHE`` is False) or the first construction raises — callers
    treat ``None`` as "no vector re-ranking" and BM25 alone drives results.
    """
    global _SHARED_SESSION_VCM
    if not _HAS_VECTOR_CACHE:
        return None
    with _SHARED_SESSION_VCM_LOCK:
        if _SHARED_SESSION_VCM is None:
            try:
                _SHARED_SESSION_VCM = VectorCacheManager(".asicode/session_vector_cache")
            except Exception:
                return None
        return _SHARED_SESSION_VCM


def _reset_session_vcm_for_test() -> None:
    """Drop the memoised session VCM (test-only: isolation between tests)."""
    global _SHARED_SESSION_VCM
    with _SHARED_SESSION_VCM_LOCK:
        _SHARED_SESSION_VCM = None


def _archive_sig(session_mgr: Any, sid: str):
    """Return (sid, size, mtime_ns) for a session's archive, or None if absent."""
    try:
        p = session_mgr.archive_path(sid)
        if not p.exists():
            return None
        st = p.stat()
        return (sid, st.st_size, st.st_mtime_ns)
    except Exception:
        return None


def _archived_bm25_entries(
    session_mgr: Any, sid: str, archived_turns: list,
) -> tuple[list[tuple[Any, dict, int, str]], tuple | None, bool]:
    """Return pre-tokenised BM25 entries for a session's archived turns (cached).

    On a cache miss the archived turns are tokenised once and stored under their
    ``(sid, size, mtime_ns)`` signature; repeat searches reuse them.  The doc id
    is the 0-based index into ``archived_turns`` (matches ``_turns`` ordering,
    which is always ``archived + active``).  Any error degrades to returning a
    freshly-built list without caching (search still works, just slower).

    Returns:
        ``(entries, sig, from_cache)`` where ``entries`` is the pre-tokenised list,
        ``sig`` is the archive signature (``(sid, size, mtime_ns)`` or ``None``),
        and ``from_cache`` is ``True`` if the result came from the BM25 cache.
    """
    sig = _archive_sig(session_mgr, sid)
    _small = len(archived_turns) <= _ARCHIVE_SMALL_SKIP if archived_turns else True

    # Cache hit (only for non-small archives)
    if sig is not None and not _small:
        with _ARCHIVED_BM25_CACHE_LOCK:
            cached = _ARCHIVED_BM25_CACHE.get(sig)
            if cached is not None:
                _ARCHIVED_BM25_CACHE.move_to_end(sig)
                return cached, sig, True  # from_cache=True

    entries: list[tuple[Any, dict, int, str]] = []
    for idx, turn in enumerate(archived_turns):
        content = turn.get("content", "") if isinstance(turn, dict) else ""
        if not content:
            continue
        tokens = _TOKENIZER.tokenize(content)
        if not tokens:
            continue
        tc: dict[str, int] = {}
        for t in tokens:
            tc[t] = tc.get(t, 0) + 1
        entries.append((idx, tc, len(tokens), content))

    # Store in cache (skip for small archives — tokenisation is fast enough)
    if sig is not None and not _small:
        with _ARCHIVED_BM25_CACHE_LOCK:
            _ARCHIVED_BM25_CACHE[sig] = entries
            _ARCHIVED_BM25_CACHE.move_to_end(sig)
            while len(_ARCHIVED_BM25_CACHE) > _ARCHIVED_BM25_CACHE_MAX:
                _evicted_sig, _evicted_val = _ARCHIVED_BM25_CACHE.popitem(last=False)
                # Keep the two caches' lifetimes coupled: the evicted BM25 sig
                # is also dropped from the vector-indexed set. Guard under the
                # VECTOR lock so this structure uses a single lock (the previous
                # code mutated it under the BM25 lock → latent two-lock race on
                # the same set). Lock order is always BM25 → VECTOR here (no
                # reverse acquisition exists anywhere), so no deadlock risk.
                with _VECTOR_CACHE_LOCK:
                    _VECTOR_CACHE_INDEXED_ARCHIVES.pop(_evicted_sig, None)

    return entries, sig, False


class DesignChatLoop:
    """
    Lightweight tool-use loop for design chat.

    Usage:
        loop = DesignChatLoop(llm_client, registry, model)
        result = loop.respond(messages, stream_callback=cb)
    """

    def __init__(
        self,
        llm_client,
        registry,  # ToolRegistry instance
        model: str,
        run_store=None,  # InMemoryRunStore — enables adaptive tool-usage learning
    ):
        self.llm_client = llm_client
        self.registry = registry
        self.model = model
        self._run_store = run_store  # Optional: adaptive learner hub for tool usage
        # Phase 1 — semantic lint feature flag (env var, default on)
        self._asr_semantic_lint = os.environ.get("ASICODE_SEMANTIC_LINT", "1") == "1"
        # Serialize the paired append of (tool_calls_made, tool_results) so that
        # concurrent read-tool threads in _respond_impl can't interleave the two
        # lists independently and produce index-mismatched records. Each pair
        # must stay atomically aligned (calls[i] ⟷ results[i]).
        self._result_lock = threading.Lock()

    def respond(
        self,
        messages: list[LLMMessage],
        stream_callback: Optional[Callable] = None,
        reasoning_callback: Optional[Callable] = None,
        max_tool_iterations: Optional[int] = None,
        token_callback: Optional[Callable] = None,
        session_id: Optional[str] = None,
        session_mgr=None,  # DesignSessionManager — enables search_design_history tool
        mode: str = "code",  # chat mode: "code" or "general"
        thinking_mode: Optional[bool] = None,  # override thinking/reasoning
        reasoning_effort: Optional[str] = None,  # thinking depth ("high" | "max")
    ) -> DesignChatResult:
        """Process a single design chat turn with tool use."""
        result = DesignChatResult()
        msgs = list(messages)
        self.session_id = session_id or ""
        self._session_mgr = session_mgr
        try:
            return self._respond_impl(msgs, stream_callback, reasoning_callback, max_tool_iterations, token_callback, result, mode=mode, thinking_mode=thinking_mode, reasoning_effort=reasoning_effort)
        except AgentCancelled as _ac:
            # User pressed ESC — let the caller pause/cancel, not an error.
            # Attach the partial result so the caller can persist what the agent
 # was doing (keeps a referent for a following "let's do that" confirmation).
            _ac.partial_result = result
            raise
        except LLMClientError as e:
            # Expected, service-side LLM errors (rate limit / auth / quota /
            # server overload) — not a bug in our code.  Log a single concise
            # WARNING (no stack trace) and show the user a clean, actionable
            # message instead of the raw provider JSON blob.
            # The terminal handler (_RowSafeRichHandler) crops each log line to a
            # single line (no_wrap/ellipsis), so the soft-wrapped, indented
            # continuation no longer appears — we can log the full error here.
            # The file handler keeps the complete detail instead of a mid-word
            # truncation; the user-facing _user_facing_llm_error carries the
            # actionable message in Korean.
            logger.warning("Design chat LLM call failed (%s): %s", type(e).__name__, e)
            if isinstance(e, LLMAuthenticationError):
                result.error_type = "auth"
            result.content = _user_facing_llm_error(e)
            result.is_error = True
            return result
        except Exception as e:
            logger.error("Design chat respond failed unexpectedly: %s", e, exc_info=True)
            result.content = f"An unexpected error occurred while processing design chat: {e}"
            result.is_error = True
            return result
        finally:
            # ── Thinking-ticker teardown ──────────────────────────────────────
            # _respond_impl fires "design_thinking_start" at the top of each LLM
            # call (→ CLI starts the "… thinking · Ns" ticker). The matching stop
            # is "design_thinking" (→ CLI _stop_spinner), but several return paths
            # exit WITHOUT emitting it:
            #   • normal final answer (no tool_calls → early return at the
            #     "if not _effective_tool_calls" branch — skips the
            #     "design_thinking" emit that lives *after* it)
            #   • LLMClientError / generic Exception handlers above
            #   • AgentCancelled (re-raised — no chance to emit)
            #   • max-iterations tail (separate chat() call, no start/stop pair)
            # Left unguarded, the ticker keeps spinning while the caller renders
            # the final message → "thinking" overlaps the answer. Emitting an
            # explicit stop here in a finally covers EVERY exit path once, so
            # individual return sites don't each have to remember to.
            if stream_callback is not None:
                try:
                    stream_callback("design_thinking_stop", {})
                except Exception:
                    pass  # non-critical — never block teardown

    def _call_llm_with_retry(
        self,
        fn: Callable[[], Any],
        _estimated_prompt_tokens: int | None = None,
        overflow_retry_cb: Callable[[], bool] | None = None,
    ) -> Any:
        """Call an LLM tool function, retrying on *transient* service errors.

        The provider client already retries internally (a few attempts over
        ~12s); a sustained overload (e.g. provider code 1305) outlives that.
        This adds N outer attempts on top, with exponential backoff, so a
        design-chat turn survives a longer-lived blip instead of surfacing a
        one-off error.

        Only rate-limit and server-unavailable are retried — auth and quota are
        permanent and re-raise immediately.  The wait honors the cancel_event so
        ESC stays responsive mid-backoff, and a server Retry-After hint when the
        exception carries one.

        Context-length 400 is also retried once after re-trimming messages via
        ``overflow_retry_cb``, with ``_estimated_prompt_tokens`` passed for fast
        override convergence.
        """
        retries = max(0, _cfg.counts.DESIGN_CHAT_LLM_MAX_RETRIES)
        _auth_flipped = False  # zai auth-failover: flip endpoint at most once
        for _attempt in range(retries + 1):
            try:
                return fn()
            except (LLMRateLimitError, LLMServerUnavailableError) as e:
                if _attempt >= retries:
                    raise
                _hint = getattr(e, "retry_after", None)
                delay = _hint if isinstance(_hint, int) and _hint > 0 else min(2 ** (_attempt + 2), 30)
                # INFO, not WARNING: a transient auto-recovering retry is file-only
                # noise (asi's _TerminalInfoFilter keeps it off the prompt).
                # The final give-up surfaces as a raised exception either way.
                logger.info(
                    "Design chat transient LLM error (%s), outer retry %d/%d in %ds",
                    type(e).__name__, _attempt + 1, retries, delay,
                )
                self._retry_wait(delay)
            except LLMConnectionError as e:
                # A connection-level failure (timeout / cannot-connect) is endpoint-
                # specific, so retry ONLY by flipping the z.ai facade to its sibling
                # (Anthropic-compat <-> OpenAI-compat).  If no flip is possible
                # (non-z.ai provider, or a custom base_url with no known sibling),
                # preserve the prior behavior of surfacing the error immediately
                # rather than silently re-hitting the same dead endpoint.
                if _attempt >= retries or not self._flip_zai_endpoint():
                    raise
                logger.info(
                    "Design chat connection error (%s) — flipped z.ai endpoint, "
                    "outer retry %d/%d", type(e).__name__, _attempt + 1, retries,
                )
                self._retry_wait(_ZAI_FAILOVER_RETRY_DELAY)
            except LLMAuthenticationError as e:
                # zai serves two endpoints (Anthropic-compat + OpenAI-compat)
                # under the SAME key, but an account may be authorized for only
                # one of them — e.g. the Anthropic-compat endpoint returns HTTP
                # 401 "Authentication Failed" (code 1000) for a key the
                # OpenAI-compat endpoint accepts (where the *real* problem is
                # balance/quota, surfaced via LLMQuotaExceededError). Surfacing
                # that 401 as an "Invalid API key" prompt traps the user:
                # re-entering the SAME valid key cannot fix an endpoint-specific
                # rejection. Flip to the sibling ONCE; if it also auth-fails the
                # key is genuinely invalid and re-raises to the auth-prompt path.
                # (Insight: defense-depth parity — extend the existing zai-endpoint
                # flip mechanism rather than a new special-case code path.)
                if _attempt >= retries or _auth_flipped or not self._flip_zai_endpoint():
                    raise
                _auth_flipped = True
                logger.info(
                    "Design chat auth error (%s) on zai — flipped endpoint, "
                    "retry %d/%d (no backoff)", type(e).__name__, _attempt + 1, retries,
                )
            except LLMAPIError as e:
                # Context-length 400 → record overflow override + in-turn retry
                if _is_context_length_error(e):
                    _record_context_overflow(self.model, estimated_prompt_tokens=_estimated_prompt_tokens)
                    logger.warning(
                        "Context-length 400 for %s — recorded overflow override (est=%s)",
                        self.model, _estimated_prompt_tokens,
                    )
                    # In-turn recovery: re-trim messages and retry once in this loop,
                    # continuing rather than raising so the retry loop handles it.
                    if overflow_retry_cb is not None and _attempt < retries:
                        if overflow_retry_cb():  # True = trim made progress
                            continue
                raise
        # Unreachable: the loop either returns or re-raises on the last attempt.
        # Note: if P1 guard (above) is correct, context-length 400 on the last attempt
        # will `raise` rather than `continue` — so this RuntimeError is never hit.
        raise RuntimeError("unreachable: _call_llm_with_retry exhausted without return")

    def _retry_wait(self, delay: float) -> None:
        """Sleep ``delay`` seconds between retries, honoring the cancel_event so
        ESC stays responsive mid-backoff."""
        _ce = getattr(self.registry.config, "cancel_event", None)
        if _ce is not None:
            if _ce.is_set():
                raise AgentCancelled("cancelled by user during retry wait")
            if _ce.wait(timeout=delay):  # returns True when set mid-wait
                raise AgentCancelled("cancelled by user during retry wait")
        else:
            time.sleep(delay)

    def _flip_zai_endpoint(self) -> bool:
        """Swap ``self.llm_client`` between z.ai's Anthropic-compat and
        OpenAI-compat endpoints for a connection-error retry, returning True when
        a flip happened.

        No-op (returns False) unless the active client is a z.ai client using a
        DEFAULT base_url — a custom base_url has no known sibling URL, so flipping
        would point at the wrong host.  The flip is local to this DesignChatLoop
        instance (a fresh loop is built per REPL turn from the service's original
        client), so it does not mutate the shared service client.  The sibling is
        given a capped timeout (``_ZAI_FAILOVER_TIMEOUT``) so alternating attempts
        can't stack into a multi-minute hang.
        """
        cur = self.llm_client
        try:
            if cur.get_provider_name().lower() != "zai":
                return False
        except Exception:
            return False
        if getattr(cur, "base_url", None):  # custom endpoint — no known sibling
            return False
        from external_llm.anthropic_client import ZAIAnthropicClient
        from external_llm.openai_client import ZAIClient
        api_key = getattr(cur, "api_key", None)
        if isinstance(cur, ZAIAnthropicClient):
            new_client: Any = ZAIClient(api_key, None, _ZAI_FAILOVER_TIMEOUT)
        elif isinstance(cur, ZAIClient):
            new_client = ZAIAnthropicClient(api_key, None, _ZAI_FAILOVER_TIMEOUT)
        else:
            return False
        self.llm_client = new_client
        logger.info(
            "z.ai endpoint flipped to %s (timeout=%ss) for connection-error retry",
            type(new_client).__name__, _ZAI_FAILOVER_TIMEOUT,
        )
        return True

    def _respond_impl(
        self,
        msgs: list[LLMMessage],
        stream_callback: Optional[Callable],
        reasoning_callback: Optional[Callable],
        max_tool_iterations: Optional[int],
        token_callback: Optional[Callable],
        result: DesignChatResult,
        mode: str = "code",
        thinking_mode: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
    ) -> DesignChatResult:
        _max_iterations = max_tool_iterations if max_tool_iterations and max_tool_iterations > 0 else _cfg.counts.DESIGN_CHAT_MAX_TOOL_ITERATIONS

        all_schemas = self.registry.get_tool_schemas(lang_filter=self.registry.repo_language)
        if mode == "general":
            # /general mode: keep only non-code tools (web search, ask user, etc.)
            tool_schemas = [s for s in all_schemas if s["name"] in _GENERAL_MODE_TOOLS]
        else:
            tool_schemas = all_schemas
        _max_tokens = 65536

        # ── Main tool loop ───────────────────────────────────────────────
        _empty_retried = False  # allow one empty-response retry across all iterations
        _plan_nudges = 0  # plan completion gate nudge count (per turn)
        # The work plan is per-turn: a stale plan from a previous turn would
        # trigger the completion gate on unrelated requests. Cross-turn carry
        # happens through the final message (persistence contract, rule 8).
        self.registry.session_plan = None
        for iteration in range(_max_iterations):
            # ── Cancel check (ESC) — stop cleanly between iterations ──
            # The tool dispatch guard (ToolRegistry.dispatch) short-circuits an
            # in-flight tool; this catches the cancel on the next loop turn so we
            # don't fire another LLM call after the user pressed ESC.
            _ce = getattr(self.registry.config, "cancel_event", None)
            if _ce is not None and _ce.is_set():
                raise AgentCancelled("cancelled by user during design chat")
            # ── Tool-result eviction (occupancy-gated gentle context bound; only
            # fires as the prompt nears the model's cap — see _evict_for_loop) ──
            msgs = _evict_for_loop(msgs, model=self.model or "", tool_schemas=tool_schemas)
            # ── Context hard cap guard (prevents HTTP 400 on oversized context) ──
            # Reserve output room AND account for tool-schema tokens — otherwise a
            # full prompt fills small windows (Ollama 8192) leaving 0 to generate.
            msgs = _apply_context_hard_cap(msgs, self.model, tool_schemas=tool_schemas)
            # ── Work-plan state ──
            # The work plan is surfaced to the model via the update_plan tool
            # result itself (agent_tools._tool_update_plan returns render_plan),
            # so the model always sees the current checklist inline after each
            # status change. Per-iteration re-injection was removed — it
 # redundantly pressured the model into "continuing now..." narrative
            # preambles on every cycle. The completion gate (below) still catches
            # turns that end with open items.
            _llm_msgs = msgs
            # Pre-call token estimate for fast context-override convergence.
            _est_tokens = estimate_tokens_from_msgs(_llm_msgs)

            def _re_trim_design_context_overflow() -> bool:
                """Re-trim _llm_msgs after a context-override reduction (in-turn recovery).

                Returns True if trim actually reduced the estimated token count
                (caller should retry), False if no progress was made (caller should
                raise the original 400 error).
                """
                nonlocal _llm_msgs
                _before_est = estimate_tokens_from_msgs(_llm_msgs)
                _before_count = len(_llm_msgs)
                _new_limit = _resolve_context_limit(self.model)
                _new_cap = context_message_cap(_new_limit, _cfg.tokens.CONTEXT_HARD_CAP_SAFETY_MARGIN, tool_schemas)
                _llm_msgs = preemptive_trim(_llm_msgs, max_tokens=_new_cap, preserve_last=2, tag="DESIGN_CHAT_PREEMPTIVE_TRIM")
                _llm_msgs = repair_tool_message_sequence(_llm_msgs)
                _after_est = estimate_tokens_from_msgs(_llm_msgs)
                _reduced = _after_est < _before_est
                logger.info(
                    "[DESIGN_CHAT_OVERFLOW_RETRY] %s — re-trimmed %d→%d messages "
                    "(new cap=%d, before_est=%s, after_est=%s, reduced=%s)",
                    self.model, _before_count, len(_llm_msgs),
                    _new_cap, _before_est, _after_est, _reduced,
                )
                return _reduced

            try:
                _call_start = time.monotonic()
                if stream_callback:
                    try:
                        stream_callback("design_thinking_start", {})
                    except Exception:
                        pass  # non-critical — never block execution
                response: ToolCallResponse = self._call_llm_with_retry(
                    lambda: self.llm_client.chat_with_tools(
                        messages=_llm_msgs, tools=tool_schemas, model=self.model,
                        cache_breakpoint_offset=0,
                        temperature=_PROCESS_TEMPERATURE, max_tokens=_max_tokens,
                        reasoning_callback=reasoning_callback,
                        token_callback=token_callback,
                        **(dict(thinking_mode=thinking_mode) if thinking_mode is not None else {}),
                        **(dict(reasoning_effort=reasoning_effort) if reasoning_effort else {}),
                    ),
                    _estimated_prompt_tokens=_est_tokens,
                    overflow_retry_cb=_re_trim_design_context_overflow,
                )
                _call_elapsed = time.monotonic() - _call_start
                logger.debug(
                    "Design chat LLM call: iter=%d elapsed=%.1fs tokens=%d tools=%d",
                    iteration, _call_elapsed, response.tokens_used or 0,
                    len(response.tool_calls or []) if response.tool_calls else 0,
                )
                # Record LLM call to global collector for dashboard visibility.
                # Design-chat's own _call_llm_with_retry has no recording hook
                # (only agent_loop does it automatically). Without this, design-
                # chat LLM usage is invisible on the dashboard while tool calls
                # (dispatched via tool_registry.dispatch → global collector) are
                # visible — creating a counterintuitive gap.
                _dc_pt = getattr(response, "prompt_tokens", None)
                if _dc_pt is None:
                    _dc_pt = response.tokens_used or 0
                get_global_collector().record_llm_call(
                    prompt_tokens=_dc_pt,
                    completion_tokens=getattr(response, "completion_tokens", None) or 0,
                    execution_time_ms=_call_elapsed * 1000,
                    failed=False,
                )
            except Exception as e:
                from external_llm.client import LLMClientError
                # Record the failed main LLM call (this exception path means
                # _call_llm_with_retry failed after all retries). The fallback
                # below will be recorded as a separate successful LLM call.
                get_global_collector().record_llm_call(
                    prompt_tokens=0,
                    completion_tokens=0,
                    execution_time_ms=round((time.monotonic() - _call_start) * 1000),
                    failed=True,
                )
                # ALL service-side LLM errors (rate-limit / auth / quota / 5xx
                # server-overload / 4xx request-rejection / connection) propagate
                # to respond()'s LLMClientError handler for a clean, actionable
                # message. Previously only RateLimit/Auth re-raised, so a transient
                # 5xx (z.ai overload) or a 400 silently degraded the whole turn to
                # a tool-LESS plain chat — masking infra blips and request bugs as
                # "the model answered without using tools."
                if isinstance(e, (LLMClientError, AgentCancelled)):
                    raise
                # WARNING (not DEBUG): this path silently degrades the tool loop
                # to a tool-LESS plain chat, which manufactures the exact
                # "model just talks, never calls a tool" symptom. Surface the
                # exception type AND message — for an HTTP rejection the message
                # carries the provider's error body (e.g. an Anthropic/z.ai 400
                # naming the offending tool-schema field), which is the single
                # most useful signal for diagnosing why native tools were dropped.
                logger.warning(
                    "Design chat tool call failed (%s) — falling back to tool-less "
                    "plain chat (NO TOOLS this turn): %s",
                    type(e).__name__, e,
                )
                result.total_llm_calls += 1
                _fb = _fallback_plain_chat(msgs, self.llm_client, self.model, max_tokens=_max_tokens, token_callback=token_callback)
                result.content = _fb["content"]
                result.is_error = _fb["error"]
                # Mirror the split-token accumulation of the normal/final-summary
                # paths — the fallback performs a real LLM call that consumes
                # prompt/completion budget, so it must be reflected in the
                # per-bucket counters (defense-depth parity with L1186-1198 and
                # L1645-1661). Without this, fallback-path tokens were silently
                # dropped from cost/token accounting.
                result.tokens_used += _fb.get("tokens_used", 0) or 0
                result.prompt_tokens += _fb.get("prompt_tokens", 0) or 0
                result.completion_tokens += _fb.get("completion_tokens", 0) or 0
                result.cache_read_tokens += _fb.get("cache_read_tokens", 0) or 0
                result.cache_creation_tokens += _fb.get("cache_creation_tokens", 0) or 0
                result.last_call_prompt_tokens = _fb.get("prompt_tokens", 0) or 0
                result.last_call_completion_tokens = _fb.get("completion_tokens", 0) or 0
                result.last_call_cache_read_tokens = _fb.get("cache_read_tokens", 0) or 0
                result.last_call_cache_creation_tokens = _fb.get("cache_creation_tokens", 0) or 0
                # Record the fallback LLM call to the global collector too, so
                # design-chat fallback-path token consumption is not invisible
                # on the dashboard (parallel with L1598-1606).
                get_global_collector().record_llm_call(
                    prompt_tokens=_fb.get("prompt_tokens", 0) or 0,
                    completion_tokens=_fb.get("completion_tokens", 0) or 0,
                    execution_time_ms=_fb.get("execution_time_ms", 0),
                    failed=False,
                )
                if not result.provider:
                    result.provider = _fb.get("provider", "") or ""
                return result

            result.total_llm_calls += 1
            result.tokens_used += response.tokens_used or 0
            _dc_pt = getattr(response, "prompt_tokens", None)
            if _dc_pt is None:
                _dc_pt = getattr(response, "tokens_used", 0) or 0  # fallback: total when split unavailable
            result.prompt_tokens += _dc_pt
            result.completion_tokens += getattr(response, "completion_tokens", None) or 0
            result.cache_read_tokens += getattr(response, "cache_read_input_tokens", None) or 0
            result.cache_creation_tokens += getattr(response, "cache_creation_input_tokens", None) or 0
            result.last_call_prompt_tokens = getattr(response, "prompt_tokens", None) or 0
            result.last_call_completion_tokens = getattr(response, "completion_tokens", None) or 0
            result.last_call_cache_read_tokens = getattr(response, "cache_read_input_tokens", None) or 0
            result.last_call_cache_creation_tokens = getattr(response, "cache_creation_input_tokens", None) or 0
            if not result.provider:
                result.provider = getattr(response, "provider", "") or ""

            # Extract reasoning content (DeepSeek Reasoner)
            reasoning_for_msg = ""
            raw = response.raw_response or {}
            try:
                msg_obj = raw.get("choices", [{}])[0].get("message", {})
                reasoning_for_msg = msg_obj.get("reasoning_content", "") or ""
                if reasoning_for_msg and not result.reasoning_content:
                    result.reasoning_content = reasoning_for_msg
            except (AttributeError, TypeError):
                pass

            # ── Text-mode fallback ──────────────────────────────────────────────
            # Models that don't support native /api/chat tool_calls output tools
            # as plain text. Ollama's API returns them in response.content rather
            # than tool_calls. Parse and execute them the same way.
            _effective_tool_calls = list(response.tool_calls or [])
            if not _effective_tool_calls and response.content:
                _text_parsed = _parse_text_tool_calls(response.content)
                if _text_parsed:
                    from external_llm.client import ToolCallRequest
                    _effective_tool_calls = [
                        ToolCallRequest(
                            call_id=tc["id"],
                            name=tc["name"],
                            args=tc.get("args", {}),
                        )
                        for tc in _text_parsed
                        if tc.get("name")  # valid name required
                    ]
                    if _effective_tool_calls:
                        logger.debug(
                            "Text-mode tool call parsed: %s",
                            [tc.name for tc in _effective_tool_calls],
                        )

            if not _effective_tool_calls:
                result.content = response.content or ""
                # GLM-5.2 (thinking ON) / DeepSeek Reasoner may emit the final
                # answer in reasoning_content with an empty content field. The
                # intermediate path (L1400) and max-iterations path (L1651) already
                # fall back to reasoning_content; the normal termination path must
                # too, else the closing summary is silently swallowed and the REPL
                # returns to prompt with no answer. Without this, the once-retry
                # below fires, but GLM-5.2 keeps answering in reasoning_content so
                # the retry is also discarded — leaving the user with zero output.
                if not result.content.strip() and reasoning_for_msg:
                    result.content = reasoning_for_msg
                # Auto-retry once if the LLM returned empty content with no tool calls
                if not result.content.strip() and not _empty_retried:
                    _empty_retried = True
                    logger.info("Design chat: empty response on iter %d, retrying once", iteration)
                    msgs.append(LLMMessage(
                        role="user",
                        content=(
                            "[SYSTEM] You produced an empty response. "
                            "Please either use the appropriate tool to fulfill the user's request "
                            "or provide a meaningful text response."
                        ),
                    ))
                    continue

                # ── Plan completion gate ──
                # An active plan with open items means the model is ending
                # early. Nudge it to continue or to mark the items
                # skipped/blocked with reasons. The gate never forces
                # completion — it only blocks ending silently with
                # unresolved items.
                _open = _plan_open_items(getattr(self.registry, "session_plan", None))
                if _open:
                    _titles = "\n".join(f"- {it['title']}" for it in _open)
                    if _plan_nudges < _PLAN_GATE_MAX_NUDGES:
                        _plan_nudges += 1
                        logger.info(
                            "Design chat plan gate: %d open item(s), nudge %d/%d",
                            len(_open), _plan_nudges, _PLAN_GATE_MAX_NUDGES,
                        )
                        if stream_callback:
                            try:
                                stream_callback("design_plan_gate", {
                                    "open_items": [it["title"] for it in _open],
                                    "nudge": _plan_nudges,
                                    "max_nudges": _PLAN_GATE_MAX_NUDGES,
                                })
                            except Exception:
                                pass  # non-critical — never block execution
                        msgs.append(LLMMessage(role="assistant", content=result.content))
                        msgs.append(LLMMessage(
                            role="user",
                            content=(
                                "[SYSTEM] Your work plan still has unresolved items:\n"
                                f"{_titles}\n"
                                "Do NOT narrate intent (no '이어서 진행하겠습니다' or similar filler) — "
                                "act directly: either (a) call the next tool immediately to make progress on "
                                "an open item, or (b) mark each unactionable item skipped/blocked with a reason "
                                "via update_plan, then give your final answer explaining what was not done and why. "
                                "Do not end with items silently unresolved."
                            ),
                        ))
                        continue
                    # Nudges exhausted — accept the exit, but surface the
                    # unfinished items so the user always sees honest state.
                    result.content += (
                        "\n\n---\n⚠️ Unresolved plan items (auto-noted by the system):\n" + _titles
                    )
                return result

            # Not final — tool calls follow. If tokens were streamed, signal frontend to reset.
            if token_callback and response.content:
                try:
                    token_callback(None)  # None = reset sentinel
                except Exception:
                    pass  # non-critical — never block execution

            # Emit text content alongside tool calls for CLI display
            if stream_callback and response.content and response.content.strip():
                try:
                    stream_callback("design_thinking", {
                        "content": response.content.strip()[:6000],
                        "elapsed": _call_elapsed,
                    })
                except Exception:
                    pass  # non-critical — never block execution

            # Keep the latest intermediate assistant statement in result.content so
            # that if the user cancels (ESC) mid-loop, the caller can still persist
            # what the agent was about to do. The final-response path (line above)
            # overwrites this with the closing message, so the normal flow is
            # unaffected.
            if response.content and response.content.strip():
                result.content = response.content.strip()
            elif response.raw_response:
                # Some models (DeepSeek Reasoner) may put analysis in
                # raw_response.choices[0].message.reasoning_content with an
                # empty top-level content field. Fall back to that.
                try:
                    _choices = response.raw_response.get("choices", [])
                    if _choices:
                        _msg = _choices[0].get("message", {})
                        _rc = (_msg.get("reasoning_content", "") or "").strip()
                        if _rc:
                            result.content = _rc
                except (AttributeError, TypeError):
                    pass

            # Preserve provider-native content blocks so they echo back on the
            # next turn. Anthropic extended-thinking multi-turn requires the
            # `thinking` block (with signature) to be sent back, otherwise the
            # API rejects with HTTP 400. OpenAI/Google responses have no
            # top-level "content" list, so this stays None for them — no impact
            # on non-Anthropic providers.
            _raw = response.raw_response or {}
            _assistant_raw_blocks = (
                _raw.get("content") if isinstance(_raw.get("content"), list) else None
            )

            # Build assistant message
            assistant_msg = LLMMessage(
                role="assistant", content=response.content or "",
                tool_calls=[
                    {"id": tc.call_id, "type": "function",
                     "function": {"name": tc.name, "arguments": json.dumps(tc.args, ensure_ascii=False)}}
                    for tc in _effective_tool_calls
                ],
                reasoning_content=reasoning_for_msg or None,
                raw_content=_assistant_raw_blocks,
            )
            msgs.append(assistant_msg)

            # Notify UI of the token usage for the most recent LLM call — CLI displays
            # it as "the actual context size injected into the LLM this call" on the tool
            # completion (✓) line. last_call_* fields are overwritten with the latest
            # response value on each call (see :973-975 above), so this sends the single
            # call value, not accumulated result.prompt_tokens. Accumulated billing totals
            # are NOT sent via events — CLI reads chat_result (result.prompt_tokens etc.)
            # directly after the loop returns and displays them in one line
            # (asi.py ~L7092-L7137).
            # Note: design_llm_call is only emitted on tool-call paths — the final answer
            # (no tool call) returns at L1266 and never reaches here; that call's cache
            # efficiency is observed via the cumulative summary line
            # (cache {hit_pct}% → ${actual}) above.
            if stream_callback:
                try:
                    stream_callback("design_llm_call", {
                        "prompt_tokens": result.last_call_prompt_tokens,
                        "completion_tokens": result.last_call_completion_tokens,
                        "cache_read_tokens": result.last_call_cache_read_tokens,
                        "cache_creation_tokens": result.last_call_cache_creation_tokens,
                        "cache_hit_ratio": _cache_hit_ratio(
                            cache_read_tokens=result.last_call_cache_read_tokens,
                            cache_creation_tokens=result.last_call_cache_creation_tokens,
                            prompt_tokens=result.last_call_prompt_tokens,
                            provider=getattr(response, "provider", "") or "",
                        ),
                        "provider": getattr(response, "provider", "") or "",
                        "tool_call_count": len(_effective_tool_calls),
                    })
                except Exception:
                    pass

            # ── Process each tool call ───────────────────────────────────
            if len(_effective_tool_calls) == 1:
                # Single tool call — no thread overhead
                tc = _effective_tool_calls[0]
                try:
                    tool_result = self._process_tool_call_with_learning(
                        tc, stream_callback, result,
                    )
                except Exception as _err:
                    # Defensive: _process_tool_call should never raise, but if
                    # it does (dispatcher bug, unexpected tool shape) we must
                    # still produce a coherent tool message for the LLM rather
                    # than aborting the entire design-chat iteration.
                    logger.warning("Design chat: single tool call failed: %s", _err)
                    tool_result = f"Error: tool execution failed: {_err}"
                    with self._result_lock:
                        result.tool_calls_made.append({"tool": tc.name, "args": tc.args, "result_length": 0})
                        result.tool_results.append({"tool": tc.name, "args": tc.args, "content": tool_result, "ok": False})
                msgs.append(LLMMessage(
                    role="tool", content=tool_result,
                    tool_call_id=tc.call_id, name=tc.name,
                ))
            else:
                # Multiple tool calls — execute in parallel.
                #
                # Ordering invariant: ALL read tools run before ANY write tool.
                # This eliminates two races that existed when reads and writes
                # ran concurrently in the same batch:
                #   (1) a read tool could SET a cache entry while a concurrent
                #       write tool cleared the cache (lost SET);
                #   (2) a read tool could snapshot stale state AFTER a write tool
                #       changed the file, caching a value that no longer matches
                #       the post-write filesystem (correctness bug).
                # Writes serialize on _write_lock (shared file state). Reads run
                # in parallel with each other. Two sequential phases (reads,
                # then writes) cleanly separate cache-filling from cache-
                # invalidating operations.
                _write_lock = threading.Lock()
                # Serialize stream_callback invocations so concurrent tool
                # threads don't interleave events to the same sink (e.g.
                # websocket/SSE buffers).
                _cb_lock = threading.Lock()

                def _is_mutating(tc):
                    """True if the tool call changes filesystem / source / git state.

                    Delegates to ``registry._tool_call_mutates`` (single source of
                    truth shared with cache invalidation and dispatch_parallel), so a
                    mutating bash (rm, git commit, "> file", …) is treated as a write:
                    it runs AFTER all reads in the serialized write phase (acquires
                    _write_lock) instead of racing with concurrent reads in the
                    parallel phase. Read-only bash (ls, git status, grep) and all
                    pure read tools stay parallel.
                    """
                    return self.registry._tool_call_mutates(tc.name, tc.args)

                def _safe_process(tc):
                    """Thread-safe wrapper around _process_tool_call.
                    Write tools (and mutating bash) acquire a lock; read tools run
                    concurrently. The stream callback is serialized across all threads.
                    """
                    is_mutating = _is_mutating(tc)

                    def _safe_cb(*cb_args, **cb_kwargs):
                        if stream_callback is None:
                            return
                        with _cb_lock:
                            stream_callback(*cb_args, **cb_kwargs)

                    if is_mutating:
                        with _write_lock:
                            tool_result = self._process_tool_call_with_learning(
                                tc, _safe_cb, result,
                            )
                    else:
                        tool_result = self._process_tool_call_with_learning(
                            tc, _safe_cb, result,
                        )
                    return (tc, tool_result)

                from ._thread_pool import shared_pool
                # Partition into read phase (parallel) and write phase (serialized),
                # preserving original indices so tool messages map back to call_ids
                # in the order the LLM emitted them.
                _read_calls = [(i, tc) for i, tc in enumerate(_effective_tool_calls)
                               if not _is_mutating(tc)
                               and not self.registry._tool_call_is_serial(tc.name, tc.args)]
                _write_calls = [(i, tc) for i, tc in enumerate(_effective_tool_calls)
                                if _is_mutating(tc)
                                and not self.registry._tool_call_is_serial(tc.name, tc.args)]
                # Serial tools (ask_user; job only for action == "kill" — see
                # ToolRegistry._tool_call_is_serial) run strictly one-at-a-time, NOT
                # via shared_pool (which would parallelize >1 of them). ask_user
                # blocks on human input and must not race on question_id / the
                # question-count limit, nor run concurrently with reads (which would
                # stall the batch on the slowest — human — response). A serial call
                # takes priority over both other phases even when it's also
                # "mutating" (e.g. job kill), so it isn't double-placed.
                _serial_calls = [(i, tc) for i, tc in enumerate(_effective_tool_calls)
                                 if self.registry._tool_call_is_serial(tc.name, tc.args)]
                # Structural invariant: every call lands in EXACTLY one phase. The
                # three filters above are independent, so a tool mistakenly placed
                # in two phases (e.g. a SERIAL tool also classed mutating) would
                # execute twice and silently overwrite _results[idx]. Enforce a
                # disjoint cover explicitly so that latent invariant can't regress.
                _all_idx = ([i for i, _ in _read_calls]
                            + [i for i, _ in _write_calls]
                            + [i for i, _ in _serial_calls])
                if sorted(_all_idx) != list(range(len(_effective_tool_calls))):
                    raise AssertionError("tool phase partition is not a disjoint cover")

                _results: list[Optional[str]] = [None] * len(_effective_tool_calls)
                _tc_by_index = {i: tc for i, tc in enumerate(_effective_tool_calls)}

                def _collect_phase(phase_calls):
                    if not phase_calls:
                        return
                    futures = [
                        (shared_pool.submit(_safe_process, tc), idx)
                        for idx, tc in phase_calls
                    ]
                    for future, idx in futures:
                        tc = _tc_by_index[idx]
                        try:
                            _, tool_result = future.result()
                        except Exception as _ferr:
                            # _safe_process itself can raise (e.g. registry lookup,
                            # dispatcher bug) — never let one failing tool abort
                            # collection of the remaining futures, whose side-effects
                            # may already have executed. Synthesize a tool error so
                            # the LLM still gets a coherent tool message.
                            logger.warning("Design chat: tool future failed: %s", _ferr)
                            tool_result = f"Error: tool execution failed: {_ferr}"
                            with self._result_lock:
                                result.tool_calls_made.append({"tool": tc.name, "args": tc.args, "result_length": 0})
                                result.tool_results.append({"tool": tc.name, "args": tc.args, "content": tool_result, "ok": False})
                        _results[idx] = tool_result

                # Phase 1: reads run first (fills cache without interference).
                _collect_phase(_read_calls)
                # Phase 2: writes run after (invalidates cache exactly once at the end).
                _collect_phase(_write_calls)
                # Phase 3: serial tools run strictly sequentially (one at a time).
                for idx, tc in _serial_calls:
                    try:
                        _, tool_result = _safe_process(tc)
                    except Exception as _ferr:
                        logger.warning("Design chat: serial tool failed: %s", _ferr)
                        tool_result = f"Error: tool execution failed: {_ferr}"
                        with self._result_lock:
                            result.tool_calls_made.append({"tool": tc.name, "args": tc.args, "result_length": 0})
                            result.tool_results.append({"tool": tc.name, "args": tc.args, "content": tool_result, "ok": False})
                    _results[idx] = tool_result

                # Append tool messages in original call order for correct tool_call_id mapping.
                for idx in range(len(_effective_tool_calls)):
                    tc = _tc_by_index[idx]
                    tool_result = _results[idx]
                    if tool_result is None:
                        # Defensive: should be unreachable (every index covered above).
                        tool_result = "Error: tool produced no result"
                    msgs.append(LLMMessage(
                        role="tool", content=tool_result,
                        tool_call_id=tc.call_id, name=tc.name,
                    ))

        # ── Safety limit reached — request final response ───────────────
        result.hit_max_iterations = True
        logger.info("Design chat: tool iteration safety limit reached, requesting final response")
        msgs.append(LLMMessage(
            role="user",
            content=self._build_final_instruction(),
        ))

        try:
            plain_msgs = _strip_tool_messages(msgs)
            plain_msgs = _apply_context_hard_cap(plain_msgs, self.model)
            result.total_llm_calls += 1
            _final_t0 = time.monotonic()
            final_response = self.llm_client.chat(
                messages=plain_msgs, model=self.model,
                temperature=_PROCESS_TEMPERATURE, max_tokens=_max_tokens,
                token_callback=token_callback,
            )
            _final_content = final_response.content or ""
            # Record the final-response LLM call to global collector for
            # dashboard visibility (parallel with tool-loop L1598-1606).
            get_global_collector().record_llm_call(
                prompt_tokens=getattr(final_response, "prompt_tokens", None) or (final_response.tokens_used or 0),
                completion_tokens=getattr(final_response, "completion_tokens", None) or 0,
                execution_time_ms=round((time.monotonic() - _final_t0) * 1000),
                failed=False,
            )
            # DeepSeek Reasoner may put analysis in reasoning_content with empty content
            if not _final_content.strip():
                raw = final_response.raw_response or {}
                try:
                    msg_obj = raw.get("choices", [{}])[0].get("message", {})
                    _rc = msg_obj.get("reasoning_content", "") or ""
                    if _rc:
                        _final_content = _rc
                        if not result.reasoning_content:
                            result.reasoning_content = _rc
                except (AttributeError, TypeError):
                    pass

            # ── Retry with stronger instruction when LLM still returns empty ──
            # Some tool-strong models (Gemma 4 via Ollama) may return empty
            # content + tool_calls even when the schema has no tools.
            _retry_superseded = False  # set True if a retry response supersedes final_response
            if not _final_content.strip():
                logger.warning(
                    "Design chat: final response empty, retrying with stronger instruction"
                )
                _retry_msgs = list(plain_msgs)
                # Replace the last user message with a more forceful version
                _retry_msgs[-1] = LLMMessage(
                    role="user",
                    content=(
                        "[SYSTEM] CRITICAL: You MUST respond in PLAIN TEXT only. "
                        "Do NOT emit any tool calls, function_calls, XML tags, DSML, "
                        "invoke, or markup. Write your answer in natural language. "
                        "This is your final attempt."
                    ),
                )
                try:
                    _retry_t0 = time.monotonic()
                    retry_response = self.llm_client.chat(
                        messages=_retry_msgs, model=self.model,
                        temperature=_PROCESS_TEMPERATURE, max_tokens=_max_tokens,
                        token_callback=token_callback,
                    )
                    _retry_content = retry_response.content or ""
                    if _retry_content.strip():
                        _final_content = _retry_content.strip()
                    # Also check reasoning_content on retry
                    if not _final_content.strip() and retry_response.raw_response:
                        try:
                            _retry_raw = retry_response.raw_response
                            _rm = _retry_raw.get("choices", [{}])[0].get("message", {})
                            _rc2 = (_rm.get("reasoning_content", "") or "").strip()
                            if _rc2:
                                _final_content = _rc2
                        except (AttributeError, TypeError):
                            pass
                    result.tokens_used += retry_response.tokens_used or 0
                    # Mirror the split-token accumulation of the final-response
                    # path below — retry also consumes prompt/completion budget
                    # and should be reflected in the per-bucket counters.
                    _dc_rpt = getattr(retry_response, "prompt_tokens", None)
                    if _dc_rpt is None:
                        _dc_rpt = getattr(retry_response, "tokens_used", 0) or 0
                    result.prompt_tokens += _dc_rpt
                    result.completion_tokens += getattr(retry_response, "completion_tokens", None) or 0
                    result.cache_read_tokens += getattr(retry_response, "cache_read_input_tokens", None) or 0
                    result.cache_creation_tokens += getattr(retry_response, "cache_creation_input_tokens", None) or 0
                    # A successful retry supersedes the final_response — its token
                    # split becomes the authoritative "last call" reading.
                    result.last_call_prompt_tokens = getattr(retry_response, "prompt_tokens", None) or 0
                    result.last_call_completion_tokens = getattr(retry_response, "completion_tokens", None) or 0
                    result.last_call_cache_read_tokens = getattr(retry_response, "cache_read_input_tokens", None) or 0
                    result.last_call_cache_creation_tokens = getattr(retry_response, "cache_creation_input_tokens", None) or 0
                    _retry_superseded = True
                    # Record the retry-response LLM call to global collector
                    # (parallel with final_response recording at L2059-2066).
                    get_global_collector().record_llm_call(
                        prompt_tokens=getattr(retry_response, "prompt_tokens", None) or (retry_response.tokens_used or 0),
                        completion_tokens=getattr(retry_response, "completion_tokens", None) or 0,
                        execution_time_ms=round((time.monotonic() - _retry_t0) * 1000),
                        failed=False,
                    )
                except Exception as retry_e:
                    # Record the failed retry — the original final_response was
                    # already recorded above, and the retry is an additional
                    # failed LLM call.
                    get_global_collector().record_llm_call(
                        prompt_tokens=0,
                        completion_tokens=0,
                        execution_time_ms=round((time.monotonic() - _retry_t0) * 1000),
                        failed=True,
                    )
                    logger.warning("Design chat: retry also failed: %s", retry_e)

            result.content = _final_content
            result.tokens_used += final_response.tokens_used or 0
            _dc_pt2 = getattr(final_response, "prompt_tokens", None)
            if _dc_pt2 is None:
                _dc_pt2 = getattr(final_response, "tokens_used", 0) or 0  # fallback: total when split unavailable
            result.prompt_tokens += _dc_pt2
            result.completion_tokens += getattr(final_response, "completion_tokens", None) or 0
            result.cache_read_tokens += getattr(final_response, "cache_read_input_tokens", None) or 0
            result.cache_creation_tokens += getattr(final_response, "cache_creation_input_tokens", None) or 0
            # Only record final_response's split as "last call" if it was not
            # already superseded by a successful retry above. The retry supersedes
            # final_response because the LLM's final answer was actually drawn
            # from retry_response, so its token split is authoritative.
            if not _retry_superseded:
                result.last_call_prompt_tokens = getattr(final_response, "prompt_tokens", None) or 0
                result.last_call_completion_tokens = getattr(final_response, "completion_tokens", None) or 0
                result.last_call_cache_read_tokens = getattr(final_response, "cache_read_input_tokens", None) or 0
                result.last_call_cache_creation_tokens = getattr(final_response, "cache_creation_input_tokens", None) or 0
            if not result.provider:
                result.provider = getattr(final_response, "provider", "") or ""
        except Exception as e:
            # Record the failed final-response LLM call — no successful
            # recording was made (the success-path record_llm_call was
            # never reached).
            get_global_collector().record_llm_call(
                prompt_tokens=0,
                completion_tokens=0,
                execution_time_ms=round((time.monotonic() - _final_t0) * 1000),
                failed=True,
            )
            # hit_max_iterations is already True (set just before the try), but a
            # final-response GENERATION failure is a real error, not "budget
            # exhausted with partial progress". Mark is_error so downstream
            # status mapping (asi worker → "error", not "max_turns") surfaces
            # the failure correctly instead of masking it as a clean budget exit.
            result.content = f"Final response generation failed: {e}"
            result.is_error = True

        return result


    @staticmethod
    def _build_final_instruction() -> str:
        """Build the max-iterations exhaustion instruction."""
        return (
            "[SYSTEM] Tool-call budget exhausted. "
            "Answer using only the information you have already gathered. "
            "Do NOT emit XML tags, function_calls, DSML, invoke, or any other markup. "
            "Respond as plain text in the same language as the user's request."
        )

    def _search_design_history(
        self, query: str, max_results: int = 10,
        target_session_id: Optional[str] = None,
        search_field: Optional[str] = None,
    ) -> str:
        """Search design chat conversation history, optionally across sessions.

        By default searches the current session's compressed (old) turns.
        If target_session_id is provided, loads that session from disk and
        searches ALL its turns (sessions are persisted as JSON files).

        Supports:
          - P0: Session listing when query matches "list sessions"/"세션 목록".
          - P2: Field-specific search via search_field ("content", "decisions",
                "summary", "all").
        """
        if not self._session_mgr:
            return "Design session manager not available."

        # ── P0: Session listing ──────────────────────────────────────────────
        _list_indicators = {
            "list sessions", "session list", "list session",
            "세션 목록", "세션 리스트", "세션 리스팅", "모든 세션",
            "show sessions", "sessions list", "list all sessions",
        }
        _q_norm = query.lower().strip()
        if _q_norm in _list_indicators or _q_norm.startswith("list session"):
            sessions = self._session_mgr.list_sessions()
            if not sessions:
                return "No sessions found."
            lines = [f"Found {len(sessions)} session(s):\n"]
            for s in sessions:
                sid = s.get("session_id", "?")
                created = s.get("created_at", 0)
                updated = s.get("updated_at", 0)
                turns = s.get("turn_count", 0)
                has_summary = s.get("has_summary", False)
                if created:
                    created_str = datetime.datetime.fromtimestamp(created).strftime("%Y-%m-%d %H:%M")
                else:
                    created_str = "?"
                if updated:
                    updated_str = datetime.datetime.fromtimestamp(updated).strftime("%Y-%m-%d %H:%M")
                else:
                    updated_str = "?"
                summary_mark = " 📋" if has_summary else ""
                lines.append(
                    f"  session={sid}{summary_mark}\n"
                    f"    created={created_str}, updated={updated_str}, "
                    f"turns={turns}"
                )
            lines.append("")
            lines.append(
                "Hint: pass target_session_id=<id> to search a specific session. "
                "Sessions with 📋 have compressed summaries."
            )
            return "\n".join(lines)

        # ── Normalize search_field ───────────────────────────────────────────
        _field = (search_field or "content").lower().strip()
        if _field not in ("content", "decisions", "summary", "all"):
            _field = "content"

        # decisions/summary are session-level (not per-turn) — no all-sessions scan needed

        def _load_archived(sid: str) -> list:
            # Old compressed turns live in <sid>.archive.jsonl — include them so
            # history search still covers the full conversation.
            if hasattr(self._session_mgr, "load_archived_turns"):
                try:
                    return self._session_mgr.load_archived_turns(sid)
                except Exception:
                    return []
            return []

        # Session whose archive is searched (for the BM25 cache signature).
        _search_sid = target_session_id or self.session_id
        if target_session_id and target_session_id != self.session_id:
            # Cross-session search: load the target session from disk
            session = self._session_mgr.get_or_create(target_session_id)
            _archived = _load_archived(target_session_id)
            _active = session.turns
            label = f"session '{target_session_id}'"
            if not (_archived or _active) and _field in ("content", "all"):
                return f"Session '{target_session_id}' has no conversation history."
        else:
            # Current session: search only the compressed (old) portion —
            # archived turns + the compressed-but-not-yet-archived active prefix
            if not self.session_id:
                return "No active session to search."
            session = self._session_mgr.get_or_create(self.session_id)
            _local_cut = max(
                0, session.compressed_up_to - getattr(session, "archived_count", 0)
            )
            _archived = _load_archived(self.session_id)
            _active = session.turns[:_local_cut] if _local_cut > 0 else []
            label = "current session"
            if not (_archived or _active) and _field in ("content", "all"):
                return "No old conversation history to search (all turns are still in the recent window)."
        _turns = _archived + _active

        # Derive session key for vector cache isolation
        _session_key = target_session_id or self.session_id or "local"

        # ── Shared vector cache for semantic re-ranking ───────────────────────
        # A single search_design_history() call may build up to 3 _SessionSearcher
        # instances (content + decisions + summary). Each would otherwise load the
        # on-disk FAISS index + 23MB metadata.json from scratch (~77ms each,
        # ~231ms total for an "all" search). Share one VectorCacheManager across
        # all of them — the BM25 state is per-searcher, and each searcher's
        # key_to_idx map restricts vector results to its own indexed docs, so
        # field isolation is preserved. Lazily created: never loaded if no
        # searcher is actually built (e.g. an early-return path).
        _shared_vcm: Optional[Any] = None
        _shared_vcm_loaded = False

        def _get_shared_vcm() -> Any:
            nonlocal _shared_vcm, _shared_vcm_loaded
            if not _shared_vcm_loaded:
                _shared_vcm_loaded = True
                # Use the process-wide memoised VCM: the OLD per-call
                # construction reloaded faiss_index.bin + metadata.json (~77ms,
                # ~23MB) on EVERY search_design_history() invocation, and relied
                # on __del__ flushing the dirty (<100-doc) tail to disk between
                # calls — a flush that an exception traceback (holding the frame
                # local) or a GC cycle could skip, after which the archive-sig
                # gate would silently drop the recently-archived turn from vector
                # re-ranking. A single shared instance keeps the tail in memory
                # for the whole process lifetime, so correctness no longer
                # depends on inter-call disk persistence.
                if _HAS_VECTOR_CACHE:
                    _shared_vcm = _get_session_vcm()
            return _shared_vcm

        # ── P2: Field-specific search (decisions / summary) ──────────────────
        # These are session-level documents, scored by BM25
        if _field in ("decisions", "summary"):
            docs: list[tuple[Any, str]] = []
            if _field == "decisions":
                for d in session.decisions:
                    docs.append(("Decision", d))
            else:  # summary
                _summary = session.compressed_summary
                if not _summary:
                    return "No compressed summary available for this session."
                docs = [("Summary", _summary)]

            if not docs:
                return f"No matches found for '{query}' in {_field} of {label}."

            searcher = _SessionSearcher(
                session_prefix=_session_key, vector_cache=_get_shared_vcm()
            )
            searcher.index_docs(docs)
            results = searcher.search(query, top_k=max_results)

            if not results:
                return f"No matches found for '{query}' in {_field} of {label}."

            lines = [
                f"Found {len(results)} match(es) in {_field} of {label}"
                f" (showing top {len(results)}):\n"
            ]
            for r in results:
                item_type = r["id"]
                item_text = r["text"][:500].replace("\n", " ")
                lines.append(f"  [{item_type}] (score={r['score']}) {item_text}")
                if len(r["text"]) > 500:
                    lines[-1] += "..."
                lines.append("")

            return "\n".join(lines)

        # ── Per-turn + field search (content / all) ──────────────────────────
        searcher = _SessionSearcher(
            session_prefix=_session_key, vector_cache=_get_shared_vcm()
        )
        # Cached archived-turn BM25 vectors (skips ~3.9s re-tokenisation on
        # repeat searches of a long archive).  The small active prefix is
        # always tokenised fresh — it grows as the conversation progresses, and
        # df/avgdl are recomputed over the combined set so scores stay correct.
        _archived_tok, _archive_sig_val, _archive_from_cache = (
            _archived_bm25_entries(self._session_mgr, _search_sid, _archived)
        )
        _base = len(_archived)
        _active_docs: list[tuple[int, str]] = []
        for j, turn in enumerate(_active):
            content = turn.get("content", "") if isinstance(turn, dict) else ""
            if content:
                _active_docs.append((_base + j, content))
        searcher.index_docs(
            _active_docs, pre_tokenized=_archived_tok,
            archive_sig=_archive_sig_val,
        )
        results = searcher.search(query, top_k=max_results)

        # Build combined result lines
        all_lines: list[str] = []

        # Turn content results
        if results:
            scored_turns = [(r["id"], r["score"]) for r in results]
            all_lines.append(
                f"Found {len(scored_turns)} turn(s) matching '{query}' in {label}"
                f" (showing top {max_results}):\n"
            )
            for turn_idx, score in scored_turns:
                start = max(0, turn_idx - 1)
                end = min(len(_turns), turn_idx + 2)
                for t in range(start, end):
                    turn = _turns[t]
                    prefix = "▶ " if t == turn_idx else "  "
                    role_label = "user" if turn.get("role") == "user" else "assistant"
                    ts = turn.get("timestamp", "")
                    if ts:
                        ts_str = datetime.datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")
                    else:
                        ts_str = ""
                    score_str = f" (score={score})" if t == turn_idx else ""
                    content = turn.get("content", "")
                    excerpt = content[:1000].replace("\n", " ")
                    all_lines.append(f"{prefix}[Turn {t + 1}] {role_label} ({ts_str}){score_str}: {excerpt}")
                    if len(content) > 1000:
                        all_lines[-1] += "..."
                all_lines.append("")

        # If "all", also search decisions + summary
        if _field == "all":
            # Decisions
            if session.decisions:
                d_searcher = _SessionSearcher(
                    session_prefix=_session_key, vector_cache=_get_shared_vcm()
                )
                d_searcher.index_docs([("Decision", d) for d in session.decisions])
                d_results = d_searcher.search(query, top_k=min(5, max_results))
                if d_results:
                    all_lines.append(f"Found {len(d_results)} match(es) in decisions of {label}:\n")
                    for r in d_results:
                        item_text = r["text"][:500].replace("\n", " ")
                        all_lines.append(f"  [Decision] (score={r['score']}) {item_text}")
                        if len(r["text"]) > 500:
                            all_lines[-1] += "..."
                        all_lines.append("")
            # Summary
            _summary = session.compressed_summary
            if _summary:
                s_searcher = _SessionSearcher(
                    session_prefix=_session_key, vector_cache=_get_shared_vcm()
                )
                s_searcher.index_docs([("Summary", _summary)])
                s_results = s_searcher.search(query, top_k=min(3, max_results))
                if s_results:
                    all_lines.append(f"Found {len(s_results)} match(es) in summary of {label}:\n")
                    for r in s_results:
                        item_text = r["text"][:500].replace("\n", " ")
                        all_lines.append(f"  [Summary] (score={r['score']}) {item_text}")
                        if len(r["text"]) > 500:
                            all_lines[-1] += "..."
                        all_lines.append("")

        if not all_lines:
            extra = ""
            if not target_session_id:
                extra = " Try specifying target_session_id to search another session."
            return f"No matches found for '{query}' in {label}.{extra}"

        return "\n".join(all_lines)

    def _process_tool_call_with_learning(
        self,
        tc: Any,
        stream_callback: Optional[Callable],
        result: DesignChatResult,
    ) -> str:
        """Wrapper around _process_tool_call that records tool usage for adaptive learning."""
        tool_result = self._process_tool_call(tc, stream_callback, result)
        # Record tool usage for adaptive learning (if run_store is available)
        if self._run_store is not None:
            success = not tool_result.startswith("Error:")
            try:
                self._run_store.record_tool_usage("tool_loop", tc.name, success)
            except Exception:
                pass  # non-critical — never block execution
        return tool_result

    def _apply_no_effective_progress_gate(
        self, tool_name: str, ok: bool, pre_snapshots: dict, metadata: Any
    ) -> bool:
        """NO_EFFECTIVE_PROGRESS hard gate (apply_patch only).

        A patch that applied "successfully" but left every touched file
        byte-identical (its hunks matched already-present content) is not
        progress: :meth:`summarize_change` already appended a "⚠️ NO CHANGE"
        warning to the tool result; here we additionally downgrade ``ok`` so
        progress/retry heuristics and the stream status treat it as a failure,
        and tag ``failure_class="no_effective_change"`` for downstream
        classification. Returns the (possibly downgraded) ``ok``.

        ``anchor_edit``'s ``already_equal`` no-op is a *deliberate* success and
        is excluded by the ``tool_name == "apply_patch"`` guard — it does not
        represent a failed edit attempt.

        Fail-open: any error in ``all_files_unchanged`` leaves ``ok`` unchanged
        (a redundant notice is benign; a false downgrade is not).
        """
        if tool_name != "apply_patch":
            return ok
        try:
            _no_progress = self.registry._safety_manager.all_files_unchanged(pre_snapshots)
        except Exception:
            _no_progress = False
            logger.debug("all_files_unchanged check failed", exc_info=True)
        if _no_progress:
            try:
                metadata.setdefault("failure_class", "no_effective_change")
            except (AttributeError, TypeError):
                pass
            return False
        return ok

    def _process_tool_call(
        self,
        tc: Any,
        stream_callback: Optional[Callable],
        result: DesignChatResult,
    ) -> str:
        """Process a single tool call with budget and routing checks.

        Returns the tool result string.
        """

        # save_insight: persist a discovery to .asicode/design_insights.md
        if tc.name == "save_insight":
            _insight = (tc.args.get("insight") or "").strip()[:1000]
            _category = (tc.args.get("category") or "general").strip()
            if not _insight:
                # Record the failed call so history reflects reality, then return
                # a recoverable tool error (no empty ok:True entry).
                with self._result_lock:
                    result.tool_calls_made.append({"tool": tc.name, "args": tc.args, "result_length": 0})
                    result.tool_results.append({"tool": tc.name, "args": tc.args, "content": "", "ok": False})
                return "Error: 'insight' is required and must not be empty."
            # Emit "running" for CLI in-flight tracking/spinner parity with the
            # generic-tool path (L1425-1433). save_insight writes to disk, which
            # can briefly block under heavy I/O.
            if stream_callback:
                try:
                    stream_callback("design_tool_call", {
                        "call_id": tc.call_id,
                        "tool": "save_insight", "args": tc.args,
                        "status": "running",
                    })
                except Exception:
                    pass  # non-critical — never block execution
            try:
                _saved = _save_insight_to_file(
                    repo_root=self.registry.repo_root,
                    insight=_insight,
                    category=_category,
                )
                with self._result_lock:
                    result.tool_calls_made.append({"tool": tc.name, "args": tc.args, "result_length": len(_saved)})
                    result.tool_results.append({"tool": tc.name, "args": tc.args, "content": _saved, "ok": True})
                if stream_callback:
                    try:
                        stream_callback("design_tool_call", {
                            "call_id": tc.call_id,
                            "tool": "save_insight", "args": tc.args,
                            "status": "complete",
                            "preview": f"💡 Insight saved: {_insight[:80]}...",
                        })
                    except Exception:
                        pass  # non-critical — never block execution
                return _saved
            except Exception as e:
                logger.warning("save_insight failed: %s", e)
                with self._result_lock:
                    result.tool_calls_made.append({"tool": tc.name, "args": tc.args, "result_length": 0})
                    result.tool_results.append({"tool": tc.name, "args": tc.args, "content": "", "ok": False})
                return f"Error saving insight: {e}"

        # delete_insight: remove an entry from .asicode/design_insights.md
        if tc.name == "delete_insight":
            _entry_match = (tc.args.get("entry_match") or "").strip()
            if not _entry_match:
                with self._result_lock:
                    result.tool_calls_made.append({"tool": tc.name, "args": tc.args, "result_length": 0})
                    result.tool_results.append({"tool": tc.name, "args": tc.args, "content": "", "ok": False})
                return "Error: 'entry_match' is required."
            if stream_callback:
                try:
                    stream_callback("design_tool_call", {
                        "call_id": tc.call_id,
                        "tool": "delete_insight", "args": tc.args,
                        "status": "running",
                    })
                except Exception:
                    pass
            try:
                _result = _delete_insight(
                    repo_root=self.registry.repo_root,
                    entry_match=_entry_match,
                )
                _is_ok = not _result.startswith("Error")
                with self._result_lock:
                    result.tool_calls_made.append({"tool": tc.name, "args": tc.args, "result_length": len(_result)})
                    result.tool_results.append({"tool": tc.name, "args": tc.args, "content": _result, "ok": _is_ok})
                if stream_callback:
                    try:
                        stream_callback("design_tool_call", {
                            "call_id": tc.call_id,
                            "tool": "delete_insight", "args": tc.args,
                            "status": "complete" if _is_ok else "error",
                            "preview": _result[:120],
                        })
                    except Exception:
                        pass
                return _result
            except Exception as e:
                logger.warning("delete_insight failed: %s", e)
                with self._result_lock:
                    result.tool_calls_made.append({"tool": tc.name, "args": tc.args, "result_length": 0})
                    result.tool_results.append({"tool": tc.name, "args": tc.args, "content": "", "ok": False})
                return f"Error deleting insight: {e}"

        # edit_insight: replace an entry's body in .asicode/design_insights.md
        if tc.name == "edit_insight":
            _entry_match = (tc.args.get("entry_match") or "").strip()
            _new_insight = (tc.args.get("new_insight") or "").strip()
            if not _entry_match:
                with self._result_lock:
                    result.tool_calls_made.append({"tool": tc.name, "args": tc.args, "result_length": 0})
                    result.tool_results.append({"tool": tc.name, "args": tc.args, "content": "", "ok": False})
                return "Error: 'entry_match' is required."
            if not _new_insight:
                with self._result_lock:
                    result.tool_calls_made.append({"tool": tc.name, "args": tc.args, "result_length": 0})
                    result.tool_results.append({"tool": tc.name, "args": tc.args, "content": "", "ok": False})
                return "Error: 'new_insight' is required and must not be empty."
            _new_category = (tc.args.get("new_category") or "").strip() or None
            if stream_callback:
                try:
                    stream_callback("design_tool_call", {
                        "call_id": tc.call_id,
                        "tool": "edit_insight", "args": tc.args,
                        "status": "running",
                    })
                except Exception:
                    pass
            try:
                _result = _edit_insight(
                    repo_root=self.registry.repo_root,
                    entry_match=_entry_match,
                    new_insight=_new_insight,
                    new_category=_new_category,
                )
                _is_ok = not _result.startswith("Error")
                with self._result_lock:
                    result.tool_calls_made.append({"tool": tc.name, "args": tc.args, "result_length": len(_result)})
                    result.tool_results.append({"tool": tc.name, "args": tc.args, "content": _result, "ok": _is_ok})
                if stream_callback:
                    try:
                        stream_callback("design_tool_call", {
                            "call_id": tc.call_id,
                            "tool": "edit_insight", "args": tc.args,
                            "status": "complete" if _is_ok else "error",
                            "preview": _result[:120],
                        })
                    except Exception:
                        pass
                return _result
            except Exception as e:
                logger.warning("edit_insight failed: %s", e)
                with self._result_lock:
                    result.tool_calls_made.append({"tool": tc.name, "args": tc.args, "result_length": 0})
                    result.tool_results.append({"tool": tc.name, "args": tc.args, "content": "", "ok": False})
                return f"Error editing insight: {e}"

        # search_design_history: keyword search over old (compressed) turns
        if tc.name == "search_design_history":
            _query = (tc.args.get("query") or "").strip()
            if not _query:
                with self._result_lock:
                    result.tool_calls_made.append({"tool": tc.name, "args": tc.args, "result_length": 0})
                    result.tool_results.append({"tool": tc.name, "args": tc.args, "content": "", "ok": False})
                return "Error: 'query' is required."
            # Emit "running" so the CLI registers an in-flight entry and shows a
            # live spinner (search_design_history runs BM25 + vector re-ranking,
            # which can take >1s). Mirrors the generic-tool path at L1425-1433.
            if stream_callback:
                try:
                    stream_callback("design_tool_call", {
                        "call_id": tc.call_id,
                        "tool": "search_design_history", "args": tc.args,
                        "status": "running",
                    })
                except Exception:
                    pass  # non-critical — never block execution
            # Coerce max_results defensively: a text-mode model may emit a
            # non-numeric / non-scalar value. Clamp to a sane range and keep
            # this a recoverable tool error rather than aborting the turn.
            try:
                _max_results = int(tc.args.get("max_results", 3))
            except (TypeError, ValueError):
                _max_results = 3
            _max_results = max(1, min(_max_results, 50))
            try:
                _target_session_id = tc.args.get("target_session_id") or None
                _search_field = tc.args.get("search_field") or None
                _result = self._search_design_history(
                    _query, _max_results,
                    target_session_id=_target_session_id,
                    search_field=_search_field,
                )
                if stream_callback:
                    try:
                        stream_callback("design_tool_call", {
                            "call_id": tc.call_id,
                            "tool": "search_design_history", "args": tc.args,
                            "status": "complete",
                            "preview": _result.split("\n")[0] if _result and not _result.startswith("No") and not _result.startswith("Error") else _result[:120],
                        })
                    except Exception:
                        pass  # non-critical — never block execution
                with self._result_lock:
                    result.tool_calls_made.append({"tool": tc.name, "args": tc.args, "result_length": len(_result)})
                    result.tool_results.append({"tool": tc.name, "args": tc.args, "content": _result, "ok": True})
                return _result
            except Exception as e:
                logger.warning("search_design_history failed: %s", e)
                _err_msg = f"Error searching design history: {e}"
                if stream_callback:
                    try:
                        stream_callback("design_tool_call", {
                            "call_id": tc.call_id,
                            "tool": "search_design_history", "args": tc.args,
                            "status": "error",
                            "preview": _err_msg,
                        })
                    except Exception:
                        pass  # non-critical — never block execution
                with self._result_lock:
                    result.tool_calls_made.append({"tool": tc.name, "args": tc.args, "result_length": 0})
                    result.tool_results.append({"tool": tc.name, "args": tc.args, "content": _err_msg, "ok": False})
                return _err_msg

        # ── Execute tool ──
        _tool_start = time.monotonic()
        if stream_callback:
            try:
                _display_args = self.registry.normalize_args_for_display(tc.args)
            except (AttributeError, TypeError):
                _display_args = tc.args
            try:
                stream_callback("design_tool_call", {"call_id": tc.call_id, "tool": tc.name, "args": _display_args, "status": "running"})
            except Exception:
                pass  # non-critical — never block execution

        # Pre-write snapshot for a deterministic post-edit diff summary.
        # Directly targets NO_EFFECTIVE_PROGRESS: surface whether a write tool
        # actually changed anything and where the change landed — pure diff,
        # no intent inference. Snapshot covers every write tool (incl. the
        # self-validating ones the registry skips for its own verify gate).
        _pre_snapshots: dict = {}
        if tc.name in self.registry._WRITE_TOOLS:
            try:
                _pre_snapshots = self.registry._snapshot_target_files(tc.name, tc.args)
            except Exception:
                _pre_snapshots = {}

        _dispatch_exc: BaseException | None = None
        try:
            tr = self.registry.dispatch(tc.name, tc.args)
            tool_result = tr.content or tr.error or ""
            ok = tr.ok
        except Exception as e:
            tr = None
            tool_result = f"Error: {e}"
            ok = False
            _dispatch_exc = e

        # ── Persist failed write-tool invocations for post-hoc analysis ──
        # Captures (tool, failure_class, file_path, args_summary, error) to
        # ~/.asicode/learning/write_tool_failures.jsonl so we can answer
        # "which tools fail, in which failure_class, against what file/args".
        # Success of a complete write tool is a no-op inside the helper.
        if tc.name in self.registry._WRITE_TOOLS and not ok:
            try:
                from .tool_failure_log import (
                    record_write_tool_failure,
                    record_write_tool_failure_from_tr,
                )
                _repo_root = getattr(self.registry, "repo_root", None)
                if tr is not None:
                    record_write_tool_failure_from_tr(
                        tool=tc.name, tr=tr, args=tc.args,
                        model=self.model, repo_root=_repo_root,
                    )
                else:
                    record_write_tool_failure(
                        tool=tc.name, ok=False,
                        error=tool_result, metadata=None,
                        args=tc.args, model=self.model, repo_root=_repo_root,
                    )
            except Exception:
                logger.debug("tool_failure_log: record failed", exc_info=True)

        # Settle any recall hint fired on a PRIOR tool call this turn: if a
        # [RECALL] nudge fired on the previous failure, did the LLM recover on
        # this call?  Must run before recall_on_failure arms a new marker below
        # so the two never tangle.
        try:
            from .failure_pattern_store import record_recall_outcome
            record_recall_outcome(ok=ok, session_key=str(id(result)))
        except Exception:
            logger.debug("recall_outcome settle error", exc_info=True)

        # ── Per-repo failure recall (ALL tools) ──────────────────────────
        # Shared hook with the webapp turn pipeline (recall_on_failure):
        # classify → in-session dedup → record → recall.  When the failure is
        # a known-bad recurring pattern in this repo, append a [RECALL] nudge
        # to the tool result so the LLM's next turn sees it.  This is
        # orthogonal to the write-tool-only JSONL logging above (a separate
        # post-hoc analysis feed) and is intentionally not merged with it.
        # Applies to every tool (read/search/write) so the CLI matches the
        # webapp surface — the most valuable recall cases (recurring "file
        # missing", wrong-path habits) are on read-family tools.
        if not ok:
            try:
                from .failure_pattern_store import recall_on_failure
                _recall_hint = recall_on_failure(
                    tc.name, tc.args, tr,
                    getattr(self.registry, "repo_root", None),
                    exc=_dispatch_exc,
                    # result is a fresh DesignChatResult per respond() call →
                    # id() is a stable per-turn key: retries within one turn are
                    # deduped, a new turn re-records & re-hints.
                    session_key=str(id(result)),
                )
                if _recall_hint:
                    tool_result = f"{tool_result}\n\n{_recall_hint}"
            except Exception:
                logger.debug("recall_on_failure error", exc_info=True)

        # Attach deterministic post-edit diff summary on successful writes.
        # (registry already rolled back / soft-failed syntax breakage; this adds
        # the orthogonal "did the intended change actually land, and where" signal.)
        if ok and _pre_snapshots:
            try:
                _change_summary = self.registry._safety_manager.summarize_change(_pre_snapshots)
            except Exception:
                _change_summary = None
            if _change_summary:
                tool_result = f"{tool_result}\n\n{_change_summary}"

            # NO_EFFECTIVE_PROGRESS hard gate (apply_patch only): a patch that
            # applied "successfully" but left every touched file byte-identical
            # (its hunks matched already-present content) is not progress.
            # summarize_change already appended the "⚠️ NO CHANGE" warning above;
            # here we additionally downgrade `ok` so progress/retry heuristics
            # and the stream status treat it as a failure. anchor_edit's
            # already_equal no-op is a deliberate success and is NOT touched.
            ok = self._apply_no_effective_progress_gate(
                tc.name, ok, _pre_snapshots, tr.metadata
            )

            # Phase 0 — surface verify_warning (soft-fail from registry's repair cascade)
            try:
                _vw = tr.metadata.get("verify_warning")
                if _vw:
                    tool_result = f"{tool_result}\n\n[⚠️ VERIFY WARNING]\n{_vw}"
            except (AttributeError, TypeError):
                pass

            # Phase 1 — semantic lint F-code findings (soft signal, no rollback)
            if self._asr_semantic_lint:
                try:
                    _sem = self.registry._safety_manager.new_semantic_warnings(_pre_snapshots)
                    if _sem:
                        tool_result = f"{tool_result}\n\n{_sem}"
                except Exception:
                    logger.debug(
                        "semantic_lint error", exc_info=True
                    )

            # Phase 2 — auto-repair notification (inform LLM + CLI that semantic fixes were applied)
            try:
                _ar = tr.metadata.get("semantic_repaired", 0)
                if _ar > 0:
                    tool_result = f"{tool_result}\n\n[AUTO-REPAIR] {_ar} semantic finding(s) auto-fixed"
            except (AttributeError, TypeError):
                pass

        if stream_callback:
            # Tool-specific preview limits: test/lint results need more room
            _preview_limits = {
                "run_tests": 2000,
                "run_lint": 2000,
                "apply_patch": 1200,
                "bash": 1200,
            }
            _preview_max = _preview_limits.get(tc.name, 600)
            preview = tool_result[:_preview_max] + ("..." if len(tool_result) > _preview_max else "")
            # Write tools append [POST-EDIT DIFF] at the END of the result — the
            # most informative part of the preview. Re-attach it when the
            # front-truncation above cut it off.
            if "[POST-EDIT DIFF]" in tool_result and "[POST-EDIT DIFF]" not in preview:
                preview += "\n" + tool_result[tool_result.rindex("[POST-EDIT DIFF]"):][:400]
            extra: dict = {"args": _display_args}
            if tc.name == "update_plan" and ok:
                # Attach structured plan + previous status map to the event — allows
                # CLI to render checklist/changelog UI without parsing text previews.
                try:
                    _md = tr.metadata or {}
                    if _md.get("plan"):
                        extra["plan"] = _md["plan"]
                        extra["plan_prev"] = _md.get("prev_statuses")
                except (AttributeError, TypeError):
                    pass
            try:
                stream_callback("design_tool_call", {
                    "call_id": tc.call_id,
                    "tool": tc.name, "status": "complete" if ok else "error",
                    "preview": preview, **extra,
                })
            except Exception:
                pass  # non-critical — never block execution

        with self._result_lock:
            result.tool_calls_made.append({"tool": tc.name, "args": tc.args, "result_length": len(tool_result)})
            result.tool_results.append({"tool": tc.name, "args": tc.args, "content": tool_result, "ok": ok})
        logger.debug(
            "Design chat tool executed: tool=%s elapsed=%.1fs result_len=%d ok=%s",
            tc.name, time.monotonic() - _tool_start, len(tool_result), ok,
        )
        return tool_result



