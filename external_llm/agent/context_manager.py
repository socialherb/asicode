


"""Context management strategy pattern for AgentLoop and DesignChat.

Both systems need to keep LLM context within a token budget, but use
different strategies:

- ``SlidingWindowContext`` (AgentLoop): in-memory sliding window with
  algorithmic compression. Fast, no LLM calls, no persistence.

- ``SessionCompressionContext`` (DesignChat): disk-backed session with
  LLM-based background compression. Supports per-model dynamic window sizing.
"""
from __future__ import annotations

import json
import logging
import threading
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from external_llm.agent.agent_context_manager import _SYSTEM_PROMPT_TEMPLATE
from external_llm.agent.interrupt_tool_results import render_interrupt_tool_results
from external_llm.agent.message_shapes import (
    _is_anthropic_tool_result,
    _is_anthropic_tool_call,
)
from external_llm.client import (
    LLMAuthenticationError,
    LLMQuotaExceededError,
    LLMRateLimitError,
)
import dataclasses

logger = logging.getLogger(__name__)

# Per-session once-latch: an auth/quota failure is sticky (the key/quota won't
# change between turns), so re-notifying every compress cycle is pure noise.
# The latch clears only when a *different* failure class is seen — e.g. after an
# auth error, a later quota error still gets one notice, but repeated auth
# errors stay silent. A transient success does NOT clear the latch: the next
# same-class failure would re-notify, but that's the desired behavior (the user
# may have rotated the key and needs to know it's still bad).
#
# The latch is keyed by session_id and NEVER cleared on session end, so a
# long-lived server accumulates one entry per distinct session that hit a
# compress failure — unbounded growth. Capped with oldest-first eviction: a
# long-gone session's latch entry serves no purpose (session_ids are unique per
# run and never reused), so evicting it is safe — the only effect is that a
# resurrected session (impossible) would get re-notified. dict insertion order
# (Py3.7+) makes next(iter(...)) the oldest entry.
_COMPRESS_FAIL_LATCH_MAX = 512
_compress_fail_latch: dict[str, str] = {}
_compress_fail_latch_lock = threading.Lock()


class _SuppressInfoFilter(logging.Filter):
    """Filter that suppresses ALL provider log records during background compress.

    Background compression is best-effort: a failed compress is already absorbed
    by the caller's ``except Exception`` (logged at debug) and the turns are
    preserved uncompressed. Surfacing the provider's own ``logger.error`` (e.g.
    "DeepSeek authentication failed (401)") on top of that double-reports the
    same failure — and because compress re-triggers every turn, a persistent
    auth/quota problem would spam the terminal every turn. Suppress everything
    here; the caller routes a single user-facing notice via ``notify`` instead.

    Thread-safe design: each caller creates its own instance.  Adding/removing
    a unique Filter object to a logger is an atomic operation under Python's
    logging module lock, so concurrent compress threads don't interfere.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return False


def _compress_failure_notice(session_id: str, model: str, exc: Exception,
                             *, use_latch: bool = True,
                             context: str = "background") -> Optional[str]:
    """Build a one-shot user-facing notice for a compress failure.

    Returns a human-readable message the *first* time a given failure class is
    seen for ``session_id``, and ``None`` thereafter (so the caller's ``notify``
    is not spammed every turn). The latch is keyed by failure class, so a switch
    from e.g. auth → quota still gets one notice, but repeated auth errors stay
    silent until the user changes the helper model/key.

    When ``use_latch=False`` (interactive ``/insights compact`` path), the
    once-latch is bypassed and every failure is reported — the user is actively
    diagnosing the problem and needs feedback on each attempt.

    ``context`` controls the user-facing wording:
      - ``"background"``: messages reference background compression + auto-retry.
      - ``"interactive"``: messages reference the explicit command the user ran.
    """
    _is_bg = (context == "background")
    if isinstance(exc, LLMAuthenticationError):
        _cls = "auth"
        if _is_bg:
            _msg = (
                f"⚠ Helper model '{model}' authentication failed — background "
                f"compression is disabled until you fix the API key or switch the "
                f"helper model (/helper). Conversation is preserved uncompressed."
            )
        else:
            _msg = (
                f"⚠ Helper model '{model}' authentication failed — fix the API key "
                f"or switch the helper model (/helper) and try again."
            )
    elif isinstance(exc, LLMQuotaExceededError):
        _cls = "quota"
        if _is_bg:
            _msg = (
                f"⚠ Helper model '{model}' quota exceeded — background compression "
                f"is disabled until credits are restored or you switch the helper "
                f"model (/helper). Conversation is preserved uncompressed."
            )
        else:
            _msg = (
                f"⚠ Helper model '{model}' quota exceeded — restore credits or "
                f"switch the helper model (/helper) and try again."
            )
    elif isinstance(exc, LLMRateLimitError):
        _cls = "rate"
        if _is_bg:
            _msg = (
                f"⚠ Helper model '{model}' rate-limited — background compression "
                f"skipped this turn; it will retry automatically next turn."
            )
        else:
            _msg = (
                f"⚠ Helper model '{model}' rate-limited — wait a moment and "
                f"try again, or switch the helper model (/helper)."
            )
    else:
        # Connection errors, API errors, timeouts, generic exceptions, etc.
        # Background path: stay silent (logged at debug by caller).
        # Interactive path (use_latch=False): surface the error so the user
        # can diagnose the problem instead of seeing an opaque fallback.
        if use_latch:
            return None  # transient/generic errors stay silent (already at debug)
        _cls = "error"
        _msg = (
            f"⚠ Helper model '{model}' failed: {type(exc).__name__}: {exc}"
        )

    if use_latch:
        with _compress_fail_latch_lock:
            _prev = _compress_fail_latch.get(session_id)
            if _prev == _cls:
                return None  # already notified for this sticky failure class
            _compress_fail_latch[session_id] = _cls
            # Bound the latch: evict the oldest entry once over cap. A re-set of
            # an existing session_id (class change) does not grow the dict, so this
            # only trims genuine overflow from many distinct failed sessions.
            if len(_compress_fail_latch) > _COMPRESS_FAIL_LATCH_MAX:
                _oldest = next(iter(_compress_fail_latch))
                del _compress_fail_latch[_oldest]
    return _msg


# ── ABC ────────────────────────────────────────────────────────────────────


class ContextManager(ABC):
    """Common interface for context window management."""

    @abstractmethod
    def prepare_before_call(
        self,
        messages: list,
        budget: Optional[Any] = None,
    ) -> list:
        """Trim / compress *messages* to fit within the context budget.

        Called before every LLM API call.  May drop old messages, evict
        consumed tool results, or summarise compressed history.
        """

    @abstractmethod
    def trajectory_summary(self, turns: list) -> str:
        """Compress agent-turn trajectory into a compact string."""


# ── SlidingWindowContext (AgentLoop) ────────────────────────────────────────


@dataclass
class SlidingWindowConfig:
    """Configuration for SlidingWindowContext."""
    context_window_size: int = 60
    # Hysteresis: when trimming, reduce to ``window * hysteresis_factor``
    # so the count must regrow past ``window`` before the next trim.
    hysteresis_factor: float = 0.6
    # Carry-forward cap: old compressed summary content carried forward into
    # a new compressed context is capped at this many bytes.
    carry_forward_bytes: int = 2048


class SlidingWindowContext(ContextManager):
    """In-memory sliding window context manager for AgentLoop.

    Keeps the system prompt + the most recent N non-system messages.
    Older messages are either dropped or algorithmically compressed into
    a compact summary block.
    """

    def __init__(
        self,
        config: SlidingWindowConfig,
        stream_callback: Optional[Callable] = None,
    ):
        self._config = config
        self._cb = stream_callback or (lambda _evt, _data: None)

    def prepare_before_call(
        self,
        messages: list,
        budget: Optional[Any] = None,
    ) -> list:
        """Apply sliding window to keep token usage bounded.

        Always keeps the system prompt(s).
        Dropped messages are summarised into a compressed context block.
        """
        window = self._config.context_window_size
        if not window or window <= 0:
            return messages

        # Split system messages (always kept) from the rest
        system_msgs = [m for m in messages if getattr(m, "role", "") == "system"]
        other_msgs = [m for m in messages if getattr(m, "role", "") != "system"]

        if len(other_msgs) <= window:
            return messages

        # ── Hysteresis: trim below window so the count must regrow past
        #     ``window`` before triggering again — cache-stable across ~N turns.
        trim_target = max(1, int(window * self._config.hysteresis_factor))
        trimmed = len(other_msgs) - trim_target
        dropped = other_msgs[:trimmed]
        kept = other_msgs[-trim_target:]

        # (a) Orphaned tool results — start of kept has no preceding assistant.
        while kept and (
            getattr(kept[0], "role", "") == "tool"
            or _is_anthropic_tool_result(kept[0])
        ):
            m = kept[0]
            # Anthropic-native: a *user* message may mix text blocks with
            # tool_result blocks.  Dropping the entire message loses the
            # text (strategy warnings, user input).  Strip only the
            # tool_result blocks; keep the message if text remains.
            # Guard on _is_anthropic_tool_result (role=user) only: standard
            # role="tool" messages may also carry a raw_content list
            # (provider-native content[]/parts[]), but those are pure tool
            # results — preserving "text" blocks there would leave an orphan
            # tool message in `kept` and abort the loop early.
            if _is_anthropic_tool_result(m):
                rc = getattr(m, "raw_content", None)
                if isinstance(rc, list):
                    text_blocks = [
                        b for b in rc
                        if isinstance(b, dict) and b.get("type") != "tool_result"
                    ]
                    if text_blocks:
                        kept[0] = dataclasses.replace(m, raw_content=text_blocks)
                        break
                    # Pure tool_result — drop the whole message (standard path below).
            dropped.append(kept[0])
            kept = kept[1:]
            trimmed += 1

        # (b) Orphaned assistant whose tool responses were dropped.
        while (
            kept
            and (
                getattr(kept[0], "role", "") == "assistant"
                and (
                    getattr(kept[0], "tool_calls", None)
                    or _is_anthropic_tool_call(kept[0])
                )
                and (len(kept) < 2 or (
                    getattr(kept[1], "role", "") != "tool"
                    and not _is_anthropic_tool_result(kept[1])
                ))
            )
        ):
            dropped.append(kept[0])
            kept = kept[1:]
            trimmed += 1

        logger.info(
            "Sliding window: trimming %d older messages "
            "(window=%d, hysteresis_target=%d, total=%d)",
            trimmed, window, trim_target, len(other_msgs),
        )
        self._cb("context_trimmed", {"trimmed": trimmed, "kept": len(kept)})
        self._cb("agent_working", {"reason": "context_compressed", "kept": len(kept)})

        summary_msg = self._build_compressed_message(dropped)
        return [*system_msgs, summary_msg, *kept]

    def trajectory_summary(self, turns: list) -> str:
        """Compress previous agent trajectory."""
        if not turns:
            return ""
        lines = ["[TRAJECTORY SUMMARY]"]
        for t in turns[-10:]:
            status = "✓" if t.tool_result.ok else "✗"
            lines.append(f"{t.turn_num}. {t.tool_name} {status}")
        lines.append("[END TRAJECTORY]")
        return "\n".join(lines)

    # ── Internal helpers ────────────────────────────────────────────────

    def _build_compressed_message(self, dropped: list) -> Any:
        """Build a [COMPRESSED CONTEXT] user message from dropped messages.

        Categorises events (errors, writes, reads, discussion) so the LLM
        can quickly locate what matters without scanning a flat bullet list.

        Carries forward the content of any PREVIOUS [COMPRESSED CONTEXT]
        message found in ``dropped`` — without this, successive trims would
        silently lose the entire prior summary on each cycle.
        """
        from ..client import LLMMessage  # local import to avoid circular dep

        # ── Carry-forward: extract old compressed context ──────────────────
        _old_summary = ""
        _remaining: list = []
        for m in dropped:
            content = getattr(m, "content", "") or ""
            if (
                getattr(m, "role", "") == "user"
                and isinstance(content, str)
                and content.startswith("[COMPRESSED CONTEXT]")
            ):
                # Extract the body (skip the first line header)
                _lines = content.split("\n", 1)
                _old_summary = _lines[1].strip() if len(_lines) > 1 else ""
            else:
                _remaining.append(m)

        # ── Strip the structural END marker ───────────────────────────────
        # The previous compressed message always ends with
        # "[END COMPRESSED CONTEXT]" (appended unconditionally at build time).
        # Carrying that trailing line forward would duplicate the marker on
        # the next cycle (the new message appends its own). Strip it now, so
        # the carried-forward body is pure content and the byte-cap below
        # measures/trims real content rather than a stale structural marker.
        _END_MARKER = "[END COMPRESSED CONTEXT]"
        if _old_summary.endswith(_END_MARKER):
            _old_summary = _old_summary[: -len(_END_MARKER)].rstrip()

        # ── Strip nested carry-forward to prevent infinite growth ─────────
        # Each compression cycle's body already contains its own
        # "Previous summary (carried forward):" section. When re-compressing,
        # that line and everything after it is just the previous cycle's
        # already-carried-forward data — keep only the direct categorisation
        # content that precedes it, then cap at 2 KB so the summary stays
        # compact over an arbitrarily long session.
        # Match the marker anywhere in the body (not just after a newline):
        # the header is stripped, so the body may START with the marker when
        # the previous cycle had no fresh categories; requiring a leading "\n"
        # would silently miss it (strip would be a no-op and the byte cap
        # below would freeze the oldest content instead of the fresh one).
        _CF_MARKER = "Previous summary (carried forward):"
        _cf_idx = _old_summary.find(_CF_MARKER)
        if _cf_idx != -1:
            _old_summary = _old_summary[:_cf_idx].rstrip()
        # Byte-length cap (preserve whole lines so the summary is readable)
        _MAX_CF_BYTES = self._config.carry_forward_bytes
        if len(_old_summary.encode("utf-8")) > _MAX_CF_BYTES:
            # Truncate to the last whole line that fits within the cap.
            # Walk backward from the cap position so we never split a line.
            _encoded = _old_summary.encode("utf-8")
            _cut = _encoded.rfind(b"\n", 0, _MAX_CF_BYTES)
            if _cut > 0:
                _old_summary = _encoded[:_cut].decode("utf-8")
            else:
                # No line boundary found — just keep the raw cap
                _old_summary = _encoded[:_MAX_CF_BYTES].decode(
                    "utf-8", errors="replace"
                )

        categories: dict[str, list[str]] = {
            "errors": [],        # ✗ tool failures
            "files_read": [],    # file read results (via bash / read_symbol)
            "changes": [],       # apply_patch / write_plan results
            "search": [],        # find_symbol / grep results
            "other_tools": [],   # remaining tools
            "discussion": [],    # user / assistant text
        }

        # ── Build tool_use_id → tool-name map (Anthropic-native path) ──────
        # Anthropic tool_result blocks carry only ``tool_use_id`` — there is no
        # ``name`` field, so ``_tb.get("name", "?")`` always returns "?" and
        # every result would be mis-bucketed as "other_tools" regardless of the
        # actual tool. The preceding assistant message's ``tool_use`` blocks DO
        # carry ``id`` + ``name``; index them here so the result blocks below
        # resolve the same tool-name-based buckets the role="tool" path uses.
        _id_to_name: dict[str, str] = {}
        for m in _remaining:
            _rc = getattr(m, "raw_content", None)
            if isinstance(_rc, list):
                for _b in _rc:
                    if (
                        isinstance(_b, dict)
                        and _b.get("type") == "tool_use"
                        and _b.get("id")
                    ):
                        _id_to_name[_b["id"]] = _b.get("name") or "?"

        for m in _remaining:
            role = getattr(m, "role", "?")
            raw_content_str = _safe_content(m)
            # ── Anthropic-native tool result: role="user" with tool_result blocks ──
            _mrc = getattr(m, "raw_content", None)
            _tool_result_blocks: list = []
            if isinstance(_mrc, list):
                _tool_result_blocks = [
                    b for b in _mrc
                    if isinstance(b, dict) and b.get("type") == "tool_result"
                ]
            if _tool_result_blocks:
                # Classify each tool_result block like a standard role="tool" msg.
                for _tb in _tool_result_blocks:
                    name = _tb.get("name") or _id_to_name.get(
                        _tb.get("tool_use_id", ""), "?"
                    )
                    content = _tb.get("content", "")
                    # Default ok mirrors the role="tool" path (ok=False): a
                    # non-JSON result is treated as failure in BOTH paths. The
                    # payload's ``ok`` field overrides when present; the
                    # provider's ``is_error`` flag forces failure.
                    ok = False
                    try:
                        if isinstance(content, str):
                            payload = json.loads(content)
                            ok = payload.get("ok", False)
                            body = payload.get("content") or ""
                            if not ok:
                                body = payload.get("error") or body
                        else:
                            body = str(content)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        body = str(content) if content else ""
                    if _tb.get("is_error"):
                        ok = False
                    snippet = str(body)[:160].replace("\n", " ").strip()
                    if not snippet:
                        snippet = "[empty result]"
                    _entry = f"{name}: {snippet}"

                    if not ok:
                        categories["errors"].append(_entry)
                    elif name in ("apply_patch", "write_plan"):
                        categories["changes"].append(_entry)
                    elif name in ("find_symbol", "find_references"):
                        categories["search"].append(_entry)
                    else:
                        categories["other_tools"].append(_entry)
                # Do NOT fall through to "user" discussion — the tool_result
                # blocks are the payload, not conversational text.
                continue

            if role == "tool":
                name = getattr(m, "name", "?") or "?"
                ok = False
                try:
                    payload = json.loads(raw_content_str)
                    ok = payload.get("ok", False)
                    body = payload.get("content") or ""
                    if not ok:
                        body = payload.get("error") or body
                    snippet = body[:160].replace("\n", " ").strip()
                    _entry = f"{name}: {snippet}"
                except (TypeError, ValueError):
                    _entry = f"{name}: {raw_content_str[:160]}"

                if not ok:
                    categories["errors"].append(_entry)
                elif name in ("apply_patch", "write_plan"):
                    categories["changes"].append(_entry)
                elif name in ("find_symbol", "find_references"):
                    categories["search"].append(_entry)
                else:
                    categories["other_tools"].append(_entry)

            elif role == "assistant":
                snippet = raw_content_str[:160].replace("\n", " ").strip()
                if snippet:
                    categories["discussion"].append(f"[assistant] {snippet}")
            elif role == "user":
                snippet = raw_content_str[:160].replace("\n", " ").strip()
                if snippet and not snippet.startswith("[COMPRESSED"):
                    categories["discussion"].append(f"[user] {snippet}")

        sections: list[str] = [
            "[COMPRESSED CONTEXT] Earlier conversation has been summarised to save context.",
        ]

        # Priority order: errors first, then changes, then search, then files, then discussion.
        _labels = [
            ("errors", "Failed tool calls"),
            ("changes", "Applied changes"),
            ("search", "Symbol / search results"),
            ("files_read", "Files read"),
            ("other_tools", "Other tool calls"),
            ("discussion", "Discussion summary"),
        ]
        for _key, _label in _labels:
            _entries = categories.get(_key, [])
            if _entries:
                sections.append(f"\n{_label} (oldest → newest):")
                sections.extend(f"  • {e}" for e in _entries)

        # Recent topics (keyword extraction)
        topic_keywords = _extract_topics(_remaining)
        if topic_keywords:
            sections.append(f"Recent topics: {', '.join(topic_keywords)}")

        # Carry forward the previous summary when re-compression occurs.
        # Placed AFTER the fresh categories/topics so that, on re-compression,
        # the stripped _old_summary body starts with the fresh categorisation
        # and the carry-forward marker sits mid-body — which is exactly what
        # makes the strip logic above actually locate and excise it, and keeps
        # the byte-cap honest (it trims the oldest tail = carried-forward data,
        # not the fresh categories).
        if _old_summary:
            sections.append(
                f"\nPrevious summary (carried forward):\n{_old_summary}"
            )

        sections.append("[END COMPRESSED CONTEXT]")
        return LLMMessage(role="user", content="\n".join(sections))


# ── SessionCompressionContext (DesignChat) ───────────────────────────────


# ── Cross-instance compression dedup ────────────────────────────────────────
# Module-level (process-global) so every SessionCompressionContext instance —
# which DesignSessionManager creates per-request — shares one Lock per session.
# With per-instance dicts, overlapping/rapid successive requests for the same
# session each got a distinct Lock, defeating the non-blocking acquire in
# schedule_background_compress and spawning redundant LLM summary calls. The
# disk layer (_adopt_from_disk + monotonic pointer) still guarantees
# correctness; this only prevents duplicate cost/latency.
#
# WeakValueDictionary, not plain dict: a plain dict keyed by session_id would
# accumulate one Lock per distinct session for the process lifetime — an
# unbounded leak in a long-lived webapp/CLI, since the only explicit removal
# path (DesignSessionManager.delete_session → .pop) fires solely on an explicit
# DELETE request, and normal sessions are created, used, then abandoned. Locks
# are weakref-able, so idle entries are GC'd automatically once no caller holds
# a strong reference; active compressions keep entries alive (the caller binds
# the returned Lock to a local that outlives the operation — see
# schedule_background_compress / compact_now), preserving cross-instance Lock
# identity exactly when dedup matters. _get_compress_lock uses a strong local
# binding to survive the assign→return window (a naive assign-then-re-read races
# with GC under WeakValueDictionary). Sibling defense: _compress_fail_latch
# (above) caps the per-session failure-notice latch to bound the same class of
# per-session module-global growth.
_MODULE_COMPRESS_LOCKS: "weakref.WeakValueDictionary[str, threading.Lock]" = weakref.WeakValueDictionary()
_MODULE_COMPRESS_LOCKS_MUTEX = threading.Lock()


class SessionCompressionContext(ContextManager):
    """Disk-backed session context with LLM-based background compression.

    Used by DesignChat.  Builds context messages from scratch each turn
    using session data (compressed summary + verbatim recent turns).
    """

    def __init__(self, repo_root: str):
        from ._shared_utils import load_project_context_md as _lpm
        from .config.thresholds import config as _cfg

        self._repo_root_str = repo_root
        self._repo_root_path = Path(repo_root)
        self._load_project_context_md_fn = _lpm
        self._cfg = _cfg

        # project.md mtime cache
        self._project_md_cache: Optional[tuple[float, str]] = None
        # Per-session compression threading state. Reference the module-level
        # dicts so all per-request instances share one Lock per session (see
        # _MODULE_COMPRESS_LOCKS rationale above); cross-instance dedup of
        # overlapping background compressions relies on this shared identity.
        self._compress_locks = _MODULE_COMPRESS_LOCKS
        self._compress_locks_mutex = _MODULE_COMPRESS_LOCKS_MUTEX

    # ── ContextManager ABC ───────────────────────────────────────────

    def prepare_before_call(
        self,
        messages: list,
        budget: Optional[Any] = None,
    ) -> list:
        """DesignChat does not trim a pre-built message list."""
        return messages

    def trajectory_summary(self, turns: list) -> str:
        """Not applicable for design chat (no agent turns)."""
        return ""

    # ── Project context (mtime-cached) ───────────────────────────────

    def load_project_context_md(self) -> str:
        """Read .asicode/project.md with mtime caching (per-turn reload)."""
        path = self._repo_root_path / ".asicode" / "project.md"
        try:
            if not path.is_file():
                return ""
            stat = path.stat()
            current_mtime = stat.st_mtime
            if self._project_md_cache and self._project_md_cache[0] == current_mtime:
                return self._project_md_cache[1]
            result = self._load_project_context_md_fn(
                self._repo_root_str,
            )
            self._project_md_cache = (current_mtime, result)
            return result
        except Exception as e:
            logger.warning("Could not load project context: %s", e)
            return ""

    def needs_compression(
        self, session, recent_keep: Optional[int] = None,
        batch_min: Optional[int] = None,
    ) -> bool:
        """True if enough turns have accumulated since last compression.

        ``batch_min`` overrides the default ``COMPRESS_BATCH_MIN`` threshold — used
        by the force path (occupancy-gated /general compression) which applies a
        smaller minimum (``FORCE_COMPRESS_MIN_TURNS``) so compression fires sooner
        than the conservative turn-count gate, but still avoids summarizing a
        single turn every turn.
        """
        if recent_keep is None:
            recent_keep = self._cfg.compression.MIN_RECENT_TURNS_KEEP
        if batch_min is None:
            batch_min = self._cfg.compression.COMPRESS_BATCH_MIN
        if not session.turns:
            return False
        # Count only turns not yet covered by compressed_up_to (i.e. the verbatim window).
        # compressed_up_to is ABSOLUTE (archived + active); session.turns holds only
        # the active (non-archived) tail, so convert to a local index first.
        _local_cut = max(0, session.compressed_up_to - getattr(session, "archived_count", 0))
        verbatim_count = len(session.turns) - _local_cut
        old_beyond_recent = verbatim_count - recent_keep
        return old_beyond_recent >= batch_min

    # ── Tier 2: Compress (LLM summarize) ──────────────────────────────

    def compress_old_turns(
        self, session, llm_client, llm_model: str,
        recent_keep: Optional[int] = None,
        cancel_event: Optional[threading.Event] = None,
        notify: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Compress old user+AI turns into summary.

        Compresses ALL turns before the recent window into a single summary.
        Old turns are kept on disk (not discarded) — only the compressed_up_to
        pointer advances so build_context_messages() skips them.
        """
        if not session.turns:
            return

        if recent_keep is None:
            recent_keep = self._cfg.compression.MIN_RECENT_TURNS_KEEP

        # cutoff is a LOCAL index into session.turns (the active tail);
        # the absolute boundary is captured now so the pointer assignment below
        # stays correct even if _save archives turns during the LLM call.
        _archived = getattr(session, "archived_count", 0)
        _local_cut = max(0, session.compressed_up_to - _archived)
        cutoff = len(session.turns) - recent_keep
        if cutoff <= 0:
            return  # nothing old enough to compress
        abs_cutoff = _archived + cutoff

        # Only turns since the last compression are new — already-summarized
        # turns (turns[:local_cut]) are omitted from the LLM call to
        # avoid sending redundant raw text.  compressed_summary covers them.
        new_turns = session.turns[_local_cut:cutoff]
        # exclude_from_compression=True turns are ephemeral (e.g. tool call
        # output logs) — omit from both LLM summary and verbatim re-insertion.
        compressible = [
            t for t in new_turns
            if not t.get("preserve") and not t.get("exclude_from_compression")
        ]
        # preserve=True turns are excluded from the LLM summary; they stay
        # visible verbatim because build_context_messages re-inserts every
        # preserve turn before compressed_up_to into the context.
        preserved = [
            t for t in new_turns
            if t.get("preserve") and not t.get("exclude_from_compression")
        ]

        if not compressible and not preserved:
            return  # nothing to do

        # If there are turns to summarize but no LLM client, we cannot produce
        # a summary. Advancing compressed_up_to now would silently archive those
        # turns without a verbatim re-insertion path (preserve turns survive,
        # but compressible turns do not). This mirrors the LLM-failure guard
        # below (the `except: return`) and the compact_now(recent_keep=0) case
        # where the *entire* verbatim window would otherwise be lost.
        if compressible and not llm_client:
            logger.debug(
                "compress_old_turns: %d compressible turn(s) but no llm_client; "
                "skipping to preserve turns for session %s",
                len(compressible), session.session_id,
            )
            return

        # LLM compress the non-preserve turns
        if compressible and llm_client:
            conv_text = ""
            if session.compressed_summary:
                conv_text += f"Previous summary:\n{session.compressed_summary}\n\n"
            conv_text += "New conversation turns to incorporate:\n"
            for t in compressible:
                role_label = "User" if t["role"] == "user" else "AI"
                content = t["content"]
                conv_text += f"[{role_label}]: {content}\n\n"
                # Turn digests (tool work records) are intentionally NOT fed to
                # the summarizer — only conversation content is summarized.
                # The digest is dropped once the turn is compressed.

            # Suppress LLM provider logging during compress.
            # Uses a Filter instead of setLevel() to avoid thread-safety races:
            # two concurrent sessions' compress threads would race on the global
            # setLevel/restore (saved _prev_level per-thread ≠ atomic).
            #
            # The filter MUST attach to the "external_llm" logger (the one
            # providers.py emits on via ``logger = getLogger(__name__)`` where
            # __name__ == "external_llm"), NOT to this module's own logger
            # ("external_llm.agent.context_manager"). Python logging propagates
            # child→parent, so a filter on the child does NOT intercept records
            # the parent emits directly — and providers.py's ``logger.error``
            # ("DeepSeek authentication failed (401)") is emitted on the parent.
            # Attaching to the parent logger closes that gap so the per-turn
            # ERROR spam is suppressed here; the single user-facing notice is
            # routed via ``notify`` below instead.
            _llm_logger = logging.getLogger("external_llm")
            _suppress = _SuppressInfoFilter()
            try:
                if cancel_event is not None and cancel_event.is_set():
                    return
                _llm_logger.addFilter(_suppress)
                from ..client import LLMMessage, effective_content

                summary_messages = [
                    LLMMessage(
                        role="system",
                        content=(
                            "You are a conversation summarizer. Incorporate the new turns into the "
                            "existing summary. Keep the total summary under approximately 600 tokens. Preserve:\n"
                            "- Key decisions made\n- Important technical details discussed\n"
                            "- Open questions or unresolved items\n- Action items agreed upon\n"
                            "Respond with ONLY the summary, no preamble."
                        ),
                    ),
                    LLMMessage(role="user", content=conv_text),
                ]
                response = llm_client.chat(
                    messages=summary_messages,
                    model=llm_model,
                    temperature=0.1,
                    max_tokens=2048,
                )
                # GLM-5.2 (thinking ON) / DeepSeek Reasoner may emit the summary in
                # reasoning_content with empty content. effective_content() recovers
                # it — otherwise new_summary stays empty, compressed_summary is left
                # stale, yet compressed_up_to still advances below and the compressed
                # turns are silently lost (no summary, no verbatim path).
                new_summary = effective_content(response).strip()
                if new_summary:
                    session.compressed_summary = new_summary
            except Exception as e:
                logger.debug("Failed to compress conversation: %s", e)
                # A persistent helper-model auth/quota problem would otherwise be
                # invisible (provider logger.error is suppressed by _SuppressInfoFilter
                # during compress, and this except only logs at debug). Route ONE
                # user-facing notice per failure-class per session so the user learns
                # the helper model is misconfigured without per-turn spam.
                _notice = _compress_failure_notice(session.session_id, llm_model, e)
                if _notice and notify:
                    notify(_notice)
                return  # don't discard turns if compression failed
            finally:
                _llm_logger.removeFilter(_suppress)

        # Advance the ABSOLUTE compressed_up_to pointer so build_context_messages()
        # skips the summarized turns; the next _save() moves them to the archive
        # file (preserve=True turns are re-inserted into context regardless).
        session.compressed_up_to = abs_cutoff

        # Defensive fallback: add_turn already clears tool_results from the previous
        # assistant turn (_clear_prior_tool_results), so normally no tool_results reach
        # here. But this handles the extreme edge case where an interrupted turn is the
        # last assistant turn (no subsequent assistant turn) and becomes a compression
        # target. digest preserves file/tool-level metadata long-term.
        for t in compressible:
            if t.get("tool_results"):
                t.pop("tool_results", None)

        _msg = (
            "Compressed %d user/AI turns into summary (%d chars), "
            "compressed_up_to advanced to %d for session %s" % (
                len(compressible), len(session.compressed_summary or ""),
                abs_cutoff, session.session_id,
            )
        )
        if notify:
            notify(_msg)
        else:
            logger.info(_msg)

    # ── Per-session threading state ──────────────────────────────────

    def _get_compress_lock(self, session_id: str) -> threading.Lock:
        with self._compress_locks_mutex:
            # Strong local binding: under WeakValueDictionary, an inline
            # assign-then-re-read (dict[k] = Lock(); return dict[k]) races with
            # GC — the freshly created Lock has no strong reference between the
            # assign and the re-read, so the weak entry may vanish before the
            # return. Binding to `lock` first keeps it alive across the return;
            # the caller (schedule_background_compress / compact_now) then holds
            # that same strong reference for the operation's lifetime, which in
            # turn keeps the weak entry alive while dedup must hold.
            lock = self._compress_locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                self._compress_locks[session_id] = lock
            return lock

    def _make_compress_cancel_event(self) -> threading.Event:
        """Create a fresh, unset cancel Event for a single compress run.

        Each compress invocation gets its own Event (scoped to that run via the
        _compress_lock for the session). A per-session Event cached in
        self._compress_cancel_events was previously used, but it was never
        .clear()-ed anywhere — so once set it would permanently block all future
        compression for that session (context grows unboundedly). A fresh Event
        per run makes cancellation opt-in and scoped to exactly one invocation.
        """
        return threading.Event()

    # ── Unified entry points ─────────────────────────────────────────
    def schedule_background_compress(
        self, session, model: str, llm_client, system_chars: int = 0,
        force: bool = False,
        notify: Optional[Callable[[str], None]] = None,
        persist: Optional[Callable[[], None]] = None,
    ) -> None:
        """Run compression in background thread if needed."""
        if force:
            # Even on the force path (occupancy-gated /general compression), require
            # a minimum number of compressible turns. Without this, when the recent
            # window itself keeps occupancy high, every turn would trigger an LLM
            # summarize call for a single turn (the compress-lock blocks concurrency,
            # not re-firing). Fewer than FORCE_COMPRESS_MIN_TURNS → skip; the
            # hard-cap front-trim (_apply_context_hard_cap) still bounds the window.
            if not self.needs_compression(
                session, batch_min=self._cfg.compression.FORCE_COMPRESS_MIN_TURNS
            ):
                return
        elif not self.needs_compression(session):
            return
        _lock = self._get_compress_lock(session.session_id)
        if not _lock.acquire(blocking=False):
            return  # already compressing

        def _run():
            try:
                self.compress_old_turns(
                    session, llm_client, model,
                    recent_keep=self._cfg.compression.MIN_RECENT_TURNS_KEEP,
                    cancel_event=self._make_compress_cancel_event(),
                    notify=notify,
                )
                if persist:
                    persist()
            except Exception:
                # Daemon thread: an uncaught exception would be silently lost.
                # Log so background compression failures are observable.
                logger.exception(
                    "Background compress failed for session %s", session.session_id,
                )
            finally:
                _lock.release()

        _t = threading.Thread(target=_run, daemon=True)
        _t.start()

    def compact_now(
        self, session, model: str, llm_client,
        recent_keep: int = 0,
        notify: Optional[Callable[[str], None]] = None,
        persist: Optional[Callable[[], None]] = None,
    ) -> bool:
        """Synchronously fold turns down to ``recent_keep`` verbatim turns.

        Unlike :meth:`schedule_background_compress` (which always keeps
        ``MIN_RECENT_TURNS_KEEP`` and runs in a thread), this compresses the
        recent verbatim window into ``compressed_summary`` *right now* so the
        freed context applies to the very next turn. ``recent_keep=0`` clears the
        entire verbatim window. Used by the ``/clear`` command.

        Returns ``True`` if anything was actually compressed.
        """
        if not session.turns:
            return False
        _lock = self._get_compress_lock(session.session_id)
        if not _lock.acquire(blocking=False):
            return False  # a background compress is already running
        try:
            _before = session.compressed_up_to
            self.compress_old_turns(
                session, llm_client, model,
                recent_keep=recent_keep,
                cancel_event=self._make_compress_cancel_event(),
                notify=notify,
            )
            _changed = session.compressed_up_to > _before
            if _changed and persist:
                persist()
            return _changed
        finally:
            _lock.release()

    # ── Context message builder ──────────────────────────────────────

    def build_context_messages(
        self, session, current_model: str = "", system_chars: int = 0,
        skip_core_prompt: bool = False,
        mode: str = "code",
        owner: str = "",
    ) -> list[dict[str, str]]:
        """Build optimized message list for LLM call.

        Args:
            session: DesignSession instance.
            current_model: Current model name for model-switch annotations.
            system_chars: System prompt character count (unused).
            skip_core_prompt: If True, skip embedding the core system prompt.
            mode: "code" (full context) or "general" (light — no project.md /
                  insights; repo root, conversation summary and recent turns
                  are shared with code mode for cross-mode continuity).
            owner: Identifier for this process (DesignSessionManager._owner). Used to
                  render another process's in-progress user turn with the "being handled
                  in another terminal" system label. An empty string disables labeling
                  (existing behavior).

        Returns: [{role, content}] with all turns verbatim (compression disabled).
        """
        messages: list[dict[str, str]] = []
        _divider: dict[str, str] = {"role": "system", "content": "──"}

        is_general = (mode == "general")

        # 0. System prompt (identity + core rules) — identical to main agent lane
        # (Chunk 1 before "## Available Tools", without tool/session/context placeholders)
        _marker = "\n## Available Tools\n"
        _midx = _SYSTEM_PROMPT_TEMPLATE.find(_marker)
        if _midx != -1:
            core_prompt = _SYSTEM_PROMPT_TEMPLATE[:_midx].rstrip()
        else:
            core_prompt = _SYSTEM_PROMPT_TEMPLATE
        if not skip_core_prompt:
            messages.append({"role": "system", "content": core_prompt})
            messages.append(_divider)

        # 0a. Repository root path — BOTH modes. Prevents the LLM from
        #     hallucinating /repo, /workspace, /app, /project as the working
        #     dir. General mode still exposes `bash` (_GENERAL_MODE_TOOLS),
        #     so omitting this there caused hallucinated repo paths.
        #     Placed right after the static core prompt so the
        #     [core_prompt][repo root] prefix stays cached across
        #     /code ↔ /general mode switches.
        messages.append({
            "role": "system",
            "content": f"═══ CURRENT REPOSITORY: {self._repo_root_str} ═══",
        })
        messages.append(_divider)

        # 0b. Project context (from .asicode/project.md — code mode only)
        if not is_general:
            project_context = self.load_project_context_md()
            if project_context:
                messages.append({
                    "role": "system",
                    "content": (
                        "=== Project Context (Auto-RAG + Prior Session) ===\n"
                        f"{project_context}"
                    ),
                })
                messages.append(_divider)

            # 0c. Design insights (saved from previous sessions via save_insight
            #     tool). Placed BELOW the static blocks above on purpose: the
            #     insights file grows when save_insight runs mid-session, and any
            #     change invalidates the cached prompt prefix from that point on.
            #     Volatility-monotonic ordering keeps project.md / repo root
            #     cached across insight saves.
            try:
                from external_llm.agent.design_chat_loop import load_design_insights
                # NOTE: Layer 3 task_query promotion is deliberately NOT requested
                # here. This 0c block must stay byte-stable (it changes only on
                # save_insight / compact-demote / archive restore|drop) so it
                # remains inside the cached prompt prefix. Layer 3 is turn-volatile
                # (depends on recent user turns) and is injected LATE — after the
                # verbatim-turns prefix — in the "2c" block below, so it can never
                # invalidate the compressed summary + all turns.
                insights_text = load_design_insights(self._repo_root_str)
            except Exception:
                insights_text = ""

            if insights_text:
                messages.append({
                    "role": "system",
                    "content": (
                        "=== DESIGN INSIGHTS (saved from previous sessions) ===\n"
                        "These are discoveries and insights you saved in earlier sessions. "
                        "Use them as context.\n\n"
                        + insights_text
                    ),
                })
                messages.append(_divider)

        # 1. Compressed summary of older turns (if available)
        if session.compressed_summary:
            _summary_label = (
                f"(turns 1-{session.compressed_up_to})"
                if session.compressed_up_to > 0
                else "(compressed)"
            )
            messages.append({
                "role": "system",
                "content": f"=== CONVERSATION SUMMARY {_summary_label} ===\n{session.compressed_summary}",
            })
            messages.append(_divider)

        # 1b. Preserved old turns: preserve=True turns that compression has
        #     passed over (compressed_up_to advanced beyond them). They are
        #     excluded from the LLM summary, so without re-inserting them here
        #     they would silently vanish from context. Absolute turn labels —
        #     stable across rebuilds, cache-friendly.
        #     (archiving stops at the first preserve turn, so these are always
        #     still present in the active session.turns)
        _archived = getattr(session, "archived_count", 0)
        _local_cut = max(0, session.compressed_up_to - _archived)
        _preserved_old = [
            (_pi, _pt)
            for _pi, _pt in enumerate(session.turns[:_local_cut])
            if _pt.get("preserve") and not _pt.get("exclude_from_compression")
        ]
        for _pi, _pt in _preserved_old:
            _prole = _pt["role"] if _pt["role"] in ("user", "assistant") else "system"
            _pcontent = f"(turn {_archived + _pi + 1}) {_pt['content']}"
            _pdg = _pt.get("digest")
            if _pdg and _prole == "assistant":
                _pcontent += f"\n\n[WORK STATE — tools used in this turn]\n{_pdg}"
            messages.append({
                "role": _prole,
                "content": _pcontent,
            })
        if _preserved_old:
            messages.append(_divider)

        # 2. Remaining verbatim turns (from compressed_up_to onwards)
        #    Each history turn gets an absolute "(turn N)" label. The current
        #    request is identified by a trailing static system marker instead of
        #    a "[REQUEST]" content prefix: the prefix mutated the previous user
        #    message on every new turn ([REQUEST] X → (turn N) X), breaking the
        #    cache prefix there. With stable labels the entire history is
        #    byte-identical across turns.
        # actual_start: ABSOLUTE index of the first verbatim turn (for "(turn N)"
        # labels); the slice itself uses the local index into the active tail.
        actual_start = _archived + _local_cut
        verbatim_turns = session.turns[_local_cut:]

        prev_model: str = ""

        total = len(verbatim_turns)
        if total > 1:
            # No turn count in the header: the count changes every turn and would
            # break the prompt-cache prefix for every verbatim message after it.
            messages.append({
                "role": "system",
                "content": "=== RECENT CONVERSATION ===",
            })
            messages.append(_divider)

        # Pre-scan: find the last non-excluded assistant turn (= the most recent
        # normal AI response). Any exclude_from_compression turn before this
        # index is a stale tool result from an earlier cycle — useful only for
        # the immediately following turn, now safe to discard.
        _last_normal_asst_idx = -1
        for _i, _t in enumerate(verbatim_turns):
            if _t["role"] == "assistant" and not _t.get("exclude_from_compression"):
                _last_normal_asst_idx = _i

        for idx, turn in enumerate(verbatim_turns):
            # Skip stale tool-result turns (only the most recent batch survives)
            if turn.get("exclude_from_compression") and idx < _last_normal_asst_idx:
                continue
            # User turn still being processed by another process (terminal): render as
            # a system label so the model doesn't confuse it with the current request
            # and duplicate the same work. When that process records an assistant turn,
            # the flag clears and it returns to a normal user turn (tail-end cache miss).
            if (
                turn["role"] == "user"
                and turn.get("in_progress")
                and owner
                and turn.get("owner")
                and turn.get("owner") != owner
            ):
                _ip_abs = actual_start + idx + 1
                messages.append({
                    "role": "system",
                    "content": (
                        f"(turn {_ip_abs}) [IN-PROGRESS IN ANOTHER TERMINAL] "
                        "The following user request is already being handled by a "
                        "parallel session — do NOT act on it:\n" + turn["content"]
                    ),
                })
                continue
            role = turn["role"]
            turn_model = turn.get("model", "")
            content = turn["content"]

            if role == "assistant" and turn_model and prev_model and turn_model != prev_model:
                messages.append({
                    "role": "system",
                    "content": (
                        f"[Model switched: {prev_model} → {turn_model}] "
                        f"The following response was generated by a different model."
                    ),
                })
            if role == "assistant" and turn_model:
                prev_model = turn_model

            if role == "assistant":
                # Assistant turns are NOT prefixed with "(turn N)". The label is
                # only a reading aid for the model; when prior assistant turns
                # carry it, the model mimics the surface pattern and prepends
                # "(turn N)" to its own generated responses. User/tool turns keep
                # the label so the conversation still has stable turn anchors.
                prefixed = content
                # Work-state digest: deterministic record of the tool loop that
                # produced this assistant turn (tool messages themselves are
                # discarded at turn end). Stored on the turn, so this rendering
                # is byte-identical across rebuilds — cache-prefix safe.
                _dg = turn.get("digest")
                if _dg:
                    prefixed += f"\n\n[WORK STATE — tools used in this turn]\n{_dg}"
                # ESC-interrupted turns persist full tool_results, so deterministically
                # render the full content within budget cap to restore the code body
                # the turn read right after resumption. More detailed than digest, but
                # when pushed outside the verbatim window and becoming a compression
                # target (not exclude_from_compression), it is normally summarized
                # or removed.
                _tr = turn.get("tool_results")
                if _tr:
                    _rendered = render_interrupt_tool_results(_tr)
                    if _rendered:
                        prefixed += f"\n\n{_rendered}"
            elif role in ("user", "tool"):
                # Absolute turn number (stable across user turns). A relative
                # "turns-from-now" index (total - idx) re-labels every prior turn
                # each time a new turn is appended, breaking the cache prefix.
                abs_idx = actual_start + idx + 1
                prefixed = f"(turn {abs_idx}) {content}"
            else:
                prefixed = content

            messages.append({"role": role, "content": prefixed})

        if current_model and prev_model and current_model != prev_model:
            messages.append({
                "role": "system",
                "content": (
                    f"[Model switched: {prev_model} → {current_model}] "
                    f"You are now continuing this conversation. "
                    f"Previous responses above were generated by {prev_model}."
                ),
            })

        # 2c. Layer 3 promoted insights (turn-volatile). This MUST sit AFTER the
        #     cached verbatim-turns prefix: the promoted set depends on the last
        #     few user turns and changes every turn, so injecting it in the early
        #     0c insights block would invalidate the prompt cache from 0c onward
        #     (compressed summary + all turns). Placed here — right before the
        #     [CURRENT REQUEST] marker, which itself follows the always-new
        #     current user turn — so only the tail is uncached. Empty when
        #     nothing is relevant → message list byte-identical to a no-promotion
        #     turn → cache fully preserved on irrelevant turns.
        if not is_general:
            _task_q = ""
            try:
                _uturns = [
                    _t.get("content", "")
                    for _t in session.turns[-8:]
                    if _t.get("role") == "user" and _t.get("content")
                ]
                _task_q = "\n".join(_uturns)[-2000:]
            except Exception:
                _task_q = ""
            if _task_q:
                try:
                    from external_llm.agent.design_chat_loop import load_promoted_insights
                    _promoted = load_promoted_insights(self._repo_root_str, _task_q)
                    if _promoted:
                        messages.append({
                            "role": "system",
                            "content": _promoted.strip(),
                        })
                except Exception:
                    pass  # non-critical

        # Static current-request marker (no turn number — the text must be
        # byte-identical every turn; on Anthropic it is hoisted into the system
        # block, which must stay stable for the system-prompt cache breakpoints).
        if verbatim_turns and verbatim_turns[-1]["role"] == "user":
            messages.append({
                "role": "system",
                "content": (
                    "[CURRENT REQUEST] The most recent user message above is the "
                    "current request — respond to it. Earlier turns are context. "
                    "Any earlier user message marked [IN-PROGRESS IN ANOTHER "
                    "TERMINAL] or left without an assistant response is being "
                    "handled in a parallel session — do NOT act on it and do NOT "
                    "merge it into the current request. "
                    "The \"(turn N)\" labels on history messages are an internal "
                    "reading aid — do NOT prefix your own response with \"(turn N)\" "
                    "or otherwise echo these labels."
                ),
            })

        if is_general:
            messages.append({
                "role": "system",
                "content": (
                    "[MODE: General Chat] The user is in general chat mode. "
                    "Code context is not loaded. "
                    "You can answer general questions (news, weather, stocks, current events, opinions, etc.) via web search or conversation. "
                    "Do NOT use code tools unless explicitly asked."
                ),
            })

        return messages

# ── Module-level helpers (shared between strategies) ────────────────────────


def _safe_content(msg) -> str:
    """Extract string content from an LLMMessage, handling list/None types."""
    raw = msg.content
    if isinstance(raw, list):
        return " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in raw
        )
    return (raw or "").strip()


def _extract_topics(messages: list, max_keywords: int = 8) -> list[str]:
    """Extract frequent keywords from message content (no LLM)."""
    _STOP = {
        "a", "an", "the", "is", "it", "in", "on", "at", "to", "of",
        "and", "or", "but", "for", "with", "this", "that", "was",
        "are", "be", "has", "have", "had",
    }
    word_freq: dict[str, int] = {}
    for m in messages:
        raw = m.content
        if isinstance(raw, list):
            raw = " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in raw
            )
        raw = raw or ""
        for word in raw.lower().split():
            w = word.strip('.,;:!?\"\'()[]{}')
            if len(w) >= 4:
                if w not in _STOP:
                    word_freq[w] = word_freq.get(w, 0) + 1
    top_words = sorted(word_freq.items(), key=lambda x: -x[1])[:max_keywords]
    return [w for w, _ in top_words]
