"""
CollaborationOrchestrator — manages the asicode ↔ Claude Code Agent collaboration flow.

Phases:
  1. asicode Preprocessing: digest generation (cheap engine)
  2. Claude Code Agent Analysis: receives digest, uses asicode MCP tools
  3. asicode Execution: optionally executes the plan from Claude
  4. Review Loop: Claude reviews asicode's execution (configurable)
"""
from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

# Strip Claude Code SDK internal XML tool call tags (used in format_verdict_for_session etc.)
# SDK internal tool tags (<invoke>, <parameter>) + structured verdict XML (<confidence>, <suggestions>, <plan>, <status>)
# ※ <details>/<summary> are standard HTML5 tags, excluded from stripping
_STRIP_XML_RE = re.compile(
    r'</?(?:invoke|parameter|confidence|suggestions|plan|status)[^>]*>'
)

from config import CLAUDE_SDK_MAX_TURNS

from .asi_mcp_adapter import (
    build_asr_mcp_server,
    claude_sdk_missing_error,
    get_excluded_tools,
    get_restricted_options,
)
from .claude_session import ClaudeSession, SessionEvent, SessionResult

logger = logging.getLogger(__name__)

# Collaboration session default model — MUST be specified explicitly. None (SDK default)
# falls back to the user's Claude Code default model (usually Opus-class), burning $2+
# per single-shot analysis session (measured: sonnet same task $0.10~0.30).
DEFAULT_COLLAB_MODEL = "sonnet"


@dataclass
class CollaborationOrchestratorConfig:
    """Configuration for CollaborationOrchestrator."""
    max_iterations: int = 1  # analysis-only by default
    max_turns_per_iteration: int = CLAUDE_SDK_MAX_TURNS
    permission_mode: str = "bypassPermissions"
    system_prompt: Optional[str] = None
    model: Optional[str] = DEFAULT_COLLAB_MODEL
    digest_max_files: int = 8
    # digest includes only Project Info + Relevant Files by default.
    # git log/scan is often irrelevant to the task, and the agent can query
    # them directly via mcp__asr__ tools when needed.
    include_git_history: bool = False
    include_scanner_results: bool = False
    # False (default) = analysis mode: bash/apply_patch/edit_* etc. destructive tools hidden
    allow_write_tools: bool = False
    event_callback: Optional[Callable[[SessionEvent], None]] = None
    repo_root: str = "."


class CollaborationOrchestrator:
    """Manages the asicode ↔ Claude Code Agent collaboration lifecycle.

    Usage:
        async with CollaborationOrchestrator(registry, config) as orch:
            result = await orch.run("Review this code change")
            print(result.verdict.summary)
    """

    def __init__(
        self,
        registry: Any,  # ToolRegistry
        config: Optional[CollaborationOrchestratorConfig] = None,
    ):
        self._registry = registry
        self._config = config or CollaborationOrchestratorConfig()
        self._repo_root = getattr(registry, "repo_root", self._config.repo_root)

        if (
            self._config.allow_write_tools
            and self._config.permission_mode == "bypassPermissions"
        ):
            logger.warning(
                "allow_write_tools=True with bypassPermissions — "
                "destructive tools (bash, apply_patch, …) will run "
                "without user approval"
            )

        # SDK-dependent resources (mcp_server, sdk_options, session) are built
        # lazily in _ensure_session() so that __init__ stays side-effect-free
        # and pure-logic methods (_build_prompt, _generate_digest, …) remain
        # usable without claude_agent_sdk installed — the SDK is an optional
        # dependency. This mirrors ClaudeSession's own pattern (__init__ is
        # pure; the SDK client is created in __aenter__).
        self._mcp_server: Any = None
        self._sdk_options: Any = None
        self._session: Optional[ClaudeSession] = None

    def _ensure_session(self) -> ClaudeSession:
        """Lazily build the MCP server, SDK options and ClaudeSession.

        Bundles all claude_agent_sdk-dependent construction so __init__ stays
        pure. Raises ImportError with an actionable install hint if the SDK is
        missing — checked explicitly here (the contract owner) rather than
        relied upon transitively from build_asr_mcp_server().
        """
        if self._session is not None:
            return self._session

        # Explicit SDK-availability gate — owns the docstring contract directly
        # (guard-contract: the gate function checks its promised condition) and
        # fails fast before constructing any SDK-dependent object.
        try:
            import claude_agent_sdk  # noqa: F401
        except ImportError as _sdk_err:
            raise claude_sdk_missing_error(_sdk_err) from _sdk_err

        self._mcp_server = build_asr_mcp_server(
            registry=self._registry,
            server_name="asicode",
            excluded_tools=get_excluded_tools(
                allow_write=self._config.allow_write_tools,
            ),
            # Analysis mode is whitelist fail-closed — unclassified new handlers
            # are not exposed to read-only sessions
            read_only=not self._config.allow_write_tools,
        )
        self._sdk_options = get_restricted_options(
            mcp_server_config=self._mcp_server,
            system_prompt=self._config.system_prompt,
            max_turns=self._config.max_turns_per_iteration,
            permission_mode=self._config.permission_mode,
            model=self._config.model,
            allow_write=self._config.allow_write_tools,
        )
        self._session = ClaudeSession(
            options=self._sdk_options,
            event_callback=self._config.event_callback,
            include_partial=True,
        )
        return self._session

    async def __aenter__(self) -> "CollaborationOrchestrator":
        await self._ensure_session().__aenter__()
        return self

    async def __aexit__(self, *args) -> None:
        await self._session.__aexit__(*args)

    async def run(
        self,
        task: str,
        context: Optional[str] = None,
        enable_preprocessing: bool = True,
    ) -> SessionResult:
        """Execute a collaboration session end-to-end.

        Must be called within ``async with CollaborationOrchestrator(...) as orch:``
        — the context manager connects the underlying Claude SDK client.

        Args:
            task: The main task description for Claude Code Agent.
            context: Optional additional context.
            enable_preprocessing: If True, asicode generates a digest first.

        Returns:
            SessionResult with verdict and full event log.
        """
        # Phase 1: asicode Preprocessing
        digest = ""
        if enable_preprocessing:
            logger.info("Phase 1: asicode preprocessing...")
            digest = await self._generate_digest(task)

        # Phase 2: Claude Code Agent Analysis
        logger.info("Phase 2: Claude Code Agent analysis...")
        prompt = self._build_prompt(task, digest, context)
        result = await self._ensure_session().query(prompt)

        # Log summary
        if result.error:
            logger.error("Collaboration failed: %s", result.error)
        else:
            logger.info(
                "Collaboration complete: %s (%.1fs, %d tool calls)",
                result.verdict.summary,
                result.duration_seconds,
                result.tool_calls_count,
            )

        return result

    def _generate_digest_sync(self, task: str) -> str:
        """Synchronous digest body — runs asicode's cheap engine tools.

        Factored out so the async entry point can offload it via
        ``asyncio.to_thread`` (see ``_generate_digest``): these ``dispatch``
        calls are synchronous (``tool_registry.dispatch``) and individually
        slow on large repos (semantic search, tree-sitter AST scan, git
        subprocess). Running them inline would block the event loop for the
        whole of Phase 1 and starve ``interrupt()`` / other coroutines.

        Collects:
        - Project info (get_project_info)
        - Relevant files (find_relevant_files)
        - Recent git changes
        - Structural scan results

        Returns a condensed text digest for Claude Code Agent.
        """
        parts: list[str] = []

        try:
            # Project info
            proj_result = self._registry.dispatch("get_project_info", {})
            if proj_result.ok and proj_result.content:
                parts.append(f"## Project Info\n{proj_result.content[:2000]}\n")
        except Exception as ex:
            logger.debug("Digest: get_project_info failed: %s", ex)

        try:
            # Relevant files for the task
            search_result = self._registry.dispatch(
                "find_relevant_files",
                {"query": task, "top_k": self._config.digest_max_files},
            )
            if search_result.ok and search_result.content:
                parts.append(f"## Relevant Files\n{search_result.content[:2000]}\n")
        except Exception as ex:
            logger.debug("Digest: find_relevant_files failed: %s", ex)

        try:
            # Recent git history — only meaningful for review/change analysis (default off)
            if self._config.include_git_history:
                git_result = self._registry.dispatch(
                    "bash",
                    {"command": "git log --oneline -10 2>/dev/null || echo '(no git)'"},
                )
                if git_result.ok and git_result.content:
                    parts.append(f"## Recent Git History\n{git_result.content[:1000]}\n")
        except Exception as ex:
            logger.debug("Digest: git log failed: %s", ex)

        try:
            # Structural scan (if enabled)
            if self._config.include_scanner_results:
                scan_result = self._registry.dispatch(
                    "run_structural_scan",
                    {"scanner": "unused_import_scanner", "max_results": 10},
                )
                if scan_result.ok and scan_result.content:
                    parts.append(scan_result.content[:1500])
        except Exception as ex:
            logger.debug("Digest: scan failed: %s", ex)

        return "\n---\n".join(parts)

    async def _generate_digest(self, task: str) -> str:
        """Generate a concise digest using asicode's cheap engine.

        Thin async wrapper that offloads the synchronous ``dispatch`` calls to a
        worker thread (``asyncio.to_thread``) so the event loop stays free —
        keeping ``interrupt()`` responsive and other coroutines schedulable —
        during Phase 1. Collection logic lives in ``_generate_digest_sync``.
        """
        return await asyncio.to_thread(self._generate_digest_sync, task)

    def _build_prompt(self, task: str, digest: str, context: Optional[str]) -> str:
        """Build the user message for Claude Code Agent.

        Volatile content only (task + digest + extra context) — the static
        collaboration instructions (tool usage, verdict format) live in the
        system prompt append (see asi_mcp_adapter.get_restricted_options),
        keeping the system prefix byte-identical across sessions for prompt
        cache hits.
        """
        prompt_parts = [
            "# Task",
            task,
            "",
        ]

        if digest:
            prompt_parts += [
                "# Context (from asicode)",
                "Pre-computed context — use it to skip redundant exploration.",
                "",
                digest,
                "",
            ]

        if context:
            prompt_parts += [
                "# Additional Context",
                context,
                "",
            ]

        return "\n".join(prompt_parts)

    async def interrupt(self) -> None:
        """Interrupt the current collaboration session."""
        if self._session is not None:
            await self._session.interrupt()

    @property
    def session(self) -> Optional[ClaudeSession]:
        return self._session


def build_session_handoff(
    session: Any,
    max_turns: int = 8,
    per_turn_chars: int = 1500,
    max_chars: int = 8000,
) -> str:
    """Assemble asicode design session → Claude Code handoff context.

    Combines two pieces — both free (no LLM calls):
      1. compressed_summary: reuses the background compression's already-built past summary
      2. compressed_up_to onward verbatim recent turns: deterministic truncation (string ops) only

    Truncation always starts from "oldest" — the most recent turn (analysis conclusion/findings)
    is NOT truncated by per_turn_chars; if the budget (max_chars) is insufficient, the summary
    is trimmed from the front (oldest side). If still over, the final safety net cuts the
    front of the result string to preserve the recent turns' end (findings).

    Reassembled on every /claude call, but cache is unaffected — the handoff goes in the
    user message (volatile region), and the shared cache prefix across sessions ends at the
    system prompt (see asi_mcp_adapter.get_restricted_options).
    """
    if session is None:
        return ""

    # ── verbatim recent turns — most recent turn (conclusion/findings) is NOT truncated ──
    turns = getattr(session, "turns", None) or []
    # compressed_up_to (absolute index) → local index conversion
    local_cut = max(
        0,
        getattr(session, "compressed_up_to", 0)
        - getattr(session, "archived_count", 0),
    )
    recent = turns[local_cut:][-max_turns:]
    recent_block = ""
    if recent:
        lines = ["## Recent turns (verbatim, older turns truncated)"]
        last = len(recent) - 1
        for i, t in enumerate(recent):
            role = t.get("role", "?")
            content = (t.get("content", "") or "").strip()
            # Only the last (most recent) turn is fully preserved; others truncated by per_turn_chars
            if i != last and len(content) > per_turn_chars:
                content = content[:per_turn_chars] + " …(truncated)"
            lines.append(f"[{role}] {content}")
        recent_block = "\n".join(lines)

    # ── compressed summary — guarantee recent turn block first, use remaining budget ──
    # Keep the TAIL (most recent content), not the head — consistent with the
    # budget-aware truncation below and the final safety net.
    summary = (getattr(session, "compressed_summary", "") or "").strip()[-1500:]
    summary_block = ""
    if summary:
        header = "## Conversation summary (earlier turns, compressed)\n"
        sep = "\n\n" if recent_block else ""
        avail = max_chars - len(recent_block) - len(sep) - len(header)
        if avail > 0:
            if len(summary) > avail:
                # Trim from the oldest (front) to preserve the newest (tail) summary.
                marker = "…(truncated) "
                summary = (
                    marker + summary[-(avail - len(marker)):]
                    if avail > len(marker)
                    else summary[-avail:]
                )
            summary_block = header + summary

    parts = [p for p in (summary_block, recent_block) if p]
    result = "\n\n".join(parts).strip()
    # Safety net: if still over max_chars, cut from the front (older content) to keep the tail (findings)
    return result[-max_chars:] if len(result) > max_chars else result


def format_verdict_for_session(
    result: SessionResult, task: str,
) -> str:
    """Claude Code verdict -> text for injecting into asicode session.

    Labels the source clearly to prevent the design LLM from mistaking it
    for its own speech or the user's speech.

    Does NOT truncate final findings (details/suggestions) -- a previous char cap
    cut the analysis body (e.g. a list of suggestions like "Imp 4") in the middle,
    causing the design LLM to misinterpret "the last suggestion was cut due to
    length". Context growth is controlled by ephemeral injection, not a cap --
    the /claude handler records this text as an exclude_from_compression=True
    turn, so only the immediately following turn sees it verbatim; after passing
    the recent window, it drops from context (see design_session.add_turn
    docstring). Only the task label is kept short.
    """

    v = result.verdict
    _status_display = {"success": "completed", "failure": "failed"}.get(v.status, v.status)
    lines = [
        f"[Claude Code external analysis — task: {task[:120]}]",
        f"status: {_status_display} (confidence {v.confidence:.0%})",
    ]
    if v.summary:
        lines.append(f"summary: {v.summary}")
    if v.details:
        _cleaned_details = _STRIP_XML_RE.sub("", v.details)
        lines.append(f"details: {_cleaned_details}")
    if v.suggestions:
        lines.append(
            "suggestions:\n"
            + "\n".join(f"- {s}" for s in v.suggestions)
        )
    return "\n".join(lines)


