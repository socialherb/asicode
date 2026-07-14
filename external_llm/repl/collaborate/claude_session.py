"""
ClaudeSession — ClaudeSDKClient wrapper with streaming, hooks, and interrupts.

Provides a robust async context manager for Claude Code Agent communication.
Handles streaming events, permission hooks, and session lifecycle.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

from .verdict import CollaborationVerdict

logger = logging.getLogger(__name__)


@dataclass
class SessionEvent:
    """A single event from a Claude Code Agent session."""
    type: str = "unknown"  # "text", "tool_call", "tool_result", "error", "verdict", "status"
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class SessionResult:
    """Complete result of a Claude Code Agent session."""
    verdict: CollaborationVerdict
    events: list[SessionEvent] = field(default_factory=list)
    tool_calls_count: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    duration_seconds: float = 0.0
    error: Optional[str] = None


class ClaudeSession:
    """Wrapper around ClaudeSDKClient for asicode collaboration.

    Manages the full lifecycle of a Claude Code Agent conversation:
    - Connection and disconnection
    - Streaming message handling
    - Tool call event capture
    - Permission control via hooks
    - Interrupt support
    - Verdict extraction
    """

    def __init__(
        self,
        options: Any = None,  # ClaudeAgentOptions
        event_callback: Optional[Callable[[SessionEvent], None]] = None,
        include_partial: bool = True,
    ):
        self._options = options
        self._event_callback = event_callback
        self._include_partial = include_partial
        self._client: Any = None  # ClaudeSDKClient
        self._events: list[SessionEvent] = []
        self._start_time: float = 0.0
        self._tool_calls_count: int = 0
        self._tool_names_by_id: dict[str, str] = {}
        self._last_cost_usd: float = 0.0
        self._last_total_tokens: int = 0
        self._structured_candidate: Optional[dict[str, Any]] = None

    async def __aenter__(self) -> "ClaudeSession":
        """Enter context: create and connect the SDK client."""
        from claude_agent_sdk import ClaudeSDKClient

        # SDK INFO logs ("Using bundled Claude Code CLI" etc.) would intrude into the
        # terminal UI, so only pass through WARNING and above — unrelated to file handlers.
        logging.getLogger("claude_agent_sdk").setLevel(logging.WARNING)

        # Set include_partial_messages for streaming text
        if self._options is not None and self._include_partial:
            self._options.include_partial_messages = True

        self._client = ClaudeSDKClient(options=self._options)
        self._start_time = time.time()

        try:
            await self._client.connect()
            logger.debug("ClaudeSession connected")
        except Exception as ex:
            # connect() failed — clean up the client to avoid leaking
            # SDK resources (subprocess, file descriptors). disconnect()
            # on a not-yet-connected client is a no-op or raises; both
            # are safe to ignore.
            try:
                await self._client.disconnect()
            except Exception:
                pass
            self._client = None
            logger.error("Failed to connect ClaudeSession: %s", ex)
            raise

        return self

    async def __aexit__(self, *args) -> None:
        """Exit context: disconnect the SDK client."""
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception as ex:
                logger.debug("Disconnect error (ignored): %s", ex)
            self._client = None
            logger.debug("ClaudeSession disconnected")

    async def query(self, prompt: str) -> SessionResult:
        """Send a query and process all responses into a SessionResult.

        Args:
            prompt: The prompt to send to Claude Code Agent.

        Returns:
            SessionResult containing the verdict and all session events.
        """
        if self._client is None:
            raise RuntimeError("ClaudeSession not connected; use async with")

        self._events = []
        self._tool_calls_count = 0
        self._tool_names_by_id = {}
        self._last_cost_usd = 0.0
        self._last_total_tokens = 0
        self._structured_candidate = None
        start = time.monotonic()

        try:
            # Send the query
            await self._client.query(prompt)

            # Process streaming response
            verdict = await self._process_response_stream()

            duration = time.monotonic() - start

            return SessionResult(
                verdict=verdict,
                events=self._events,
                tool_calls_count=self._tool_calls_count,
                total_tokens=self._last_total_tokens,
                total_cost_usd=self._last_cost_usd,
                duration_seconds=duration,
            )

        except Exception as ex:
            duration = time.monotonic() - start
            logger.exception("ClaudeSession.query failed")
            return SessionResult(
                verdict=CollaborationVerdict(
                    status="failure",
                    summary="Session error",
                    details=str(ex),
                    confidence=0.0,
                ),
                events=self._events,
                tool_calls_count=self._tool_calls_count,
                duration_seconds=duration,
                error=str(ex),
            )

    async def _process_response_stream(self) -> CollaborationVerdict:
        """Process all messages from the streaming response.

        Handles AssistantMessage (text + tool calls), ResultMessage (verdict),
        StreamEvent (partial tokens), and error messages.

        Returns:
            CollaborationVerdict extracted from the final ResultMessage, or
            a fallback if no structured verdict is returned.
        """
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            StreamEvent,
            SystemMessage,
            UserMessage,
        )

        final_verdict: Optional[CollaborationVerdict] = None
        accumulated_text: list[str] = []
        has_error = False

        async for message in self._client.receive_response():
            if isinstance(message, StreamEvent):
                self._handle_stream_event(message)

            elif isinstance(message, AssistantMessage):
                # Check for protocol-level error on the assistant message
                error = getattr(message, "error", None)
                if error:
                    has_error = True
                self._handle_assistant_message(message, accumulated_text)

            elif isinstance(message, UserMessage):
                # Tool results arrive as ToolResultBlocks inside UserMessage —
                # NOT inside AssistantMessage. Without this branch the display
                # never sees tool completion (✓) events.
                self._handle_user_message(message)

            elif isinstance(message, ResultMessage):
                # Check for errors in the result
                msg_errors = getattr(message, "errors", None)
                api_error = getattr(message, "api_error_status", None)
                if (msg_errors or api_error) and not has_error:
                    has_error = True
                    self._emit_event(SessionEvent(
                        type="error",
                        content=f"Result error: {msg_errors or api_error}",
                    ))
                final_verdict = self._handle_result_message(
                    message, accumulated_text,
                )

            elif isinstance(message, SystemMessage):
                pass  # system messages are informational

            # Note: TaskUsage (TypedDict) is not yielded by receive_response()

        # Fallback if no structured verdict
        if final_verdict is None:
            full_text = "\n".join(accumulated_text)
            final_verdict = CollaborationVerdict(
                status="error" if has_error else ("success" if full_text else "insufficient_info"),
                summary="No structured verdict returned",
                details=full_text,
                confidence=0.5,
            )

        return final_verdict

    def _handle_stream_event(self, event: Any) -> None:
        """Process a StreamEvent (partial messages)."""
        # StreamEvent has an .event dict attribute; a raw dict is also
        # accepted (pass-through). Normalize to a dict ONCE: the body calls
        # .get() unconditionally, and an .event that is None or a typed object
        # (has .type but no .get) would otherwise raise AttributeError deep in
        # the stream loop and abort the whole query.
        raw = getattr(event, "event", None)
        ev: dict = raw if isinstance(raw, dict) else (event if isinstance(event, dict) else {})
        event_type = ev.get("type", "")

        if event_type == "content_block_delta":
            delta = ev.get("delta", {})
            delta_type = delta.get("type", "")
            if delta_type == "text_delta":
                text = delta.get("text", "")
                if text:
                    self._emit_event(SessionEvent(
                        type="text",
                        content=text,
                        metadata={"partial": True},
                    ))
            elif delta_type == "input_json_delta":
                partial_json = delta.get("partial_json", "")
                if partial_json:
                    self._emit_event(SessionEvent(
                        type="tool_call",
                        content=partial_json,
                        metadata={"partial": True, "delta_type": "input_json"},
                    ))

        elif event_type == "content_block_start":
            block = ev.get("content_block", {})
            if block.get("type") == "tool_use":
                # Count increments only on ToolUseBlock in AssistantMessage (dedup)
                tool_id = block.get("id", "")
                tool_name = block.get("name", "?")
                if tool_id:
                    self._tool_names_by_id[tool_id] = tool_name
                self._emit_event(SessionEvent(
                    type="tool_call",
                    content=f"Starting tool: {tool_name}",
                    metadata={
                        "tool_name": tool_name,
                        "tool_id": tool_id,
                        "event": "start",
                    },
                ))

        elif event_type == "content_block_stop":
            pass  # block completed

    def _handle_assistant_message(
        self, message: Any, accumulated_text: list[str]
    ) -> None:
        """Process an AssistantMessage containing content blocks."""
        from claude_agent_sdk import TextBlock, ToolResultBlock, ToolUseBlock

        for block in getattr(message, "content", []) or []:
            if isinstance(block, TextBlock):
                text = getattr(block, "text", str(block))
                if text:
                    accumulated_text.append(text)
                    self._emit_event(SessionEvent(
                        type="text",
                        content=text,
                    ))

            elif isinstance(block, ToolUseBlock):
                tool_name = getattr(block, "name", "?")
                tool_input = getattr(block, "input", {})
                tool_id = getattr(block, "id", "")
                # SDK internal tools (StructuredOutput from output_format) are
                # not user-facing tool calls — but their input IS the verdict.
                # Keep it: if the final ResultMessage arrives as an error
                # (e.g. budget exceeded), structured_output is dropped by the
                # SDK and this captured copy is the only surviving verdict.
                if tool_name in {"StructuredOutput", "output", "output_json"}:
                    if isinstance(tool_input, dict) and tool_input:
                        self._structured_candidate = tool_input
                    continue
                if tool_id:
                    self._tool_names_by_id[tool_id] = tool_name
                self._tool_calls_count += 1
                self._emit_event(SessionEvent(
                    type="tool_call",
                    content=f"Tool: {tool_name}",
                    metadata={
                        "tool_name": tool_name,
                        "tool_id": tool_id,
                        "input": tool_input,
                        "event": "complete",
                    },
                ))

            elif isinstance(block, ToolResultBlock):
                self._emit_tool_result(block)

    def _handle_user_message(self, message: Any) -> None:
        """Process a UserMessage — extracts ToolResultBlocks (tool completions)."""
        from claude_agent_sdk import ToolResultBlock

        content = getattr(message, "content", None)
        if not isinstance(content, list):
            return
        for block in content:
            if isinstance(block, ToolResultBlock):
                self._emit_tool_result(block)

    def _emit_tool_result(self, block: Any) -> None:
        """Emit a tool_result event from a ToolResultBlock."""
        tool_use_id = getattr(block, "tool_use_id", "?")
        content = getattr(block, "content", "")
        is_error = bool(getattr(block, "is_error", False))
        # content can be str | list[{"type": "text", "text": ...}] | None
        if isinstance(content, list):
            content = "\n".join(
                str(part.get("text", "")) for part in content
                if isinstance(part, dict)
            )
        self._emit_event(SessionEvent(
            type="tool_result",
            content=str(content or "")[:500],
            metadata={
                "tool_use_id": tool_use_id,
                "tool_name": self._tool_names_by_id.get(tool_use_id, "?"),
                "is_error": is_error,
            },
        ))

    def _handle_result_message(
        self, message: Any, accumulated_text: Optional[list[str]] = None,
    ) -> CollaborationVerdict:
        """Extract a CollaborationVerdict from a ResultMessage.

        Salvage order: structured_output → captured StructuredOutput tool
        input → result text → accumulated assistant text → failure.
        A late error (budget exceeded, max turns) arrives AFTER the
        analysis is complete — the completed work must survive it.
        """
        # Capture usage/cost for the session summary
        self._last_cost_usd = float(getattr(message, "total_cost_usd", 0.0) or 0.0)
        usage = getattr(message, "usage", None)
        if isinstance(usage, dict):
            # Claude Code Agent is Anthropic-backed: input_tokens EXCLUDES
            # cache_creation_input_tokens and cache_read_input_tokens
            # (separate accounting — see _shared_utils._CACHE_TOKENS_SEPARATE).
            # Summing only input+output silently underreports total consumption
            # for cache-heavy collaboration sessions.
            self._last_total_tokens = (
                int(usage.get("input_tokens", 0) or 0)
                + int(usage.get("cache_creation_input_tokens", 0) or 0)
                + int(usage.get("cache_read_input_tokens", 0) or 0)
                + int(usage.get("output_tokens", 0) or 0)
            )

        errors = getattr(message, "errors", []) or []
        error_note = "; ".join(str(e) for e in errors)

        def _emit(verdict: CollaborationVerdict) -> CollaborationVerdict:
            self._emit_event(SessionEvent(
                type="verdict",
                content=f"Status: {verdict.status} | {verdict.summary}",
                metadata={"verdict": verdict.to_dict()},
            ))
            return verdict

        # 1. Structured output from output_format option
        structured = getattr(message, "structured_output", None)
        if isinstance(structured, dict):
            verdict = CollaborationVerdict.from_result_message(structured)
            if error_note:
                # Preserve a late error (e.g. budget exceeded) for parity with
                # the structured_candidate salvage path below — otherwise the
                # error context is silently dropped on the success path.
                verdict.metadata["result_error"] = error_note
            return _emit(verdict)

        # 2. StructuredOutput tool input captured from the stream —
        #    the SDK drops structured_output on error ResultMessages
        #    (e.g. 'Reached maximum budget') even though the verdict
        #    was already produced.
        if self._structured_candidate:
            verdict = CollaborationVerdict.from_result_message(
                self._structured_candidate,
            )
            if error_note:
                verdict.metadata["result_error"] = error_note
            return _emit(verdict)

        # 3. Result text (from unstructured output)
        result_text = str(getattr(message, "result", "") or "")
        first_line = result_text.split("\n")[0][:80] if result_text else ""
        if result_text:
            return _emit(CollaborationVerdict(
                status="needs_review" if not getattr(message, 'is_error', False) else "failure",
                summary=first_line or "Unstructured result",
                details=result_text,
                confidence=0.5,
            ))

        # 4. No result, but the assistant already streamed its analysis —
        #    return it for review instead of discarding the whole session.
        full_text = "\n".join(accumulated_text or []).strip()
        if full_text:
            return _emit(CollaborationVerdict(
                status="needs_review",
                summary=(
                    f"Session ended with error after analysis: {error_note}"
                    if error_note else "Analysis text without structured verdict"
                ),
                details=full_text,
                confidence=0.5,
                metadata={"result_error": error_note} if error_note else {},
            ))

        # 5. Truly empty — report the error
        is_error = getattr(message, 'is_error', False)
        if is_error:
            verdict = CollaborationVerdict(
                status="failure",
                summary="Execution failed",
                details=error_note or "Unknown error",
                confidence=1.0,
            )
        else:
            verdict = CollaborationVerdict(
                status="insufficient_info",
                summary="No result returned",
                confidence=0.0,
            )
        return _emit(verdict)

    async def interrupt(self) -> None:
        """Interrupt the current Claude Code Agent task."""
        if self._client is not None:
            await self._client.interrupt()
            self._emit_event(SessionEvent(type="status", content="INTERRUPTED"))

    def _emit_event(self, event: SessionEvent) -> None:
        """Emit an event to internal list and optional callback."""
        self._events.append(event)
        if self._event_callback:
            try:
                self._event_callback(event)
            except Exception as ex:
                logger.debug("Event callback error: %s", ex)
