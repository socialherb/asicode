"""
Verdict dataclass for structured collaboration output.

Claude Code Agent returns a typed verdict via output_format JSON schema.
This is the structured contract between asicode and Claude Code Agent.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class CollaborationVerdict:
    """Structured result from a Claude Code Agent collaboration session.

    Attributes:
        status: Task-completion outcome (NOT a verdict about the reviewed code).
            "success" = the agent finished the task, even if the finding is
            negative (a bug found is still a successful analysis). "failure" =
            the task itself could not be completed. Also "needs_review",
            "insufficient_info".
        summary: One-line summary of what was accomplished.
        details: Full explanation with evidence.
        confidence: Confidence level 0.0–1.0.
        suggestions: Actionable follow-up suggestions.
        plan: Optional structured plan (dict) for asicode to execute.
        metadata: Arbitrary extra data (tool calls, tokens, timing).
    """
    status: str = "needs_review"
    summary: str = ""
    details: str = ""
    confidence: float = 0.5
    suggestions: list[str] = field(default_factory=list)
    plan: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
            # Unreliable model output can arrive mistyped even outside from_result_message
            # — the _structured_candidate salvage path (streamed StructuredOutput tool input)
            # bypasses output_format JSON-Schema validation, so confidence can be a str or
            # outside [0,1]. All consumers format confidence as :.0% (ValueError on str),
            # so normalize once here — a single canonical coerce+clamp point covering all
            # creation paths (literal-float sites in claude_session.py, from_result_message,
            # and future callers).
            try:
                self.confidence = min(1.0, max(0.0, float(self.confidence)))
            except (TypeError, ValueError):
                self.confidence = 0.5
    def is_success(self) -> bool:
        return self.status == "success"

    def is_failure(self) -> bool:
        return self.status == "failure"

    def needs_review(self) -> bool:
        return self.status == "needs_review"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_result_message(cls, result: Any) -> "CollaborationVerdict":
        """Parse a ResultMessage JSON result field into a Verdict.

        Untrusted model output is normalized at this parse boundary: status is
        whitelisted, suggestions/metadata are type-coerced. ``confidence`` is
        passed through raw and normalized once by ``__post_init__`` (the single
        coerce+clamp point) — this method no longer duplicates that logic, so
        the two cannot diverge.
        """
        if hasattr(result, "result"):
            data = result.result
        elif isinstance(result, dict):
            data = result
        else:
            data = {}
        status = data.get("status", "needs_review")
        if status not in {"success", "failure", "needs_review", "insufficient_info"}:
            status = "needs_review"
        raw_suggestions = data.get("suggestions", [])
        if not isinstance(raw_suggestions, list):
            raw_suggestions = [raw_suggestions]
        suggestions = [str(s) for s in raw_suggestions]
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        return cls(
            status=status,
            summary=str(data.get("summary", "")),
            details=str(data.get("details", "")),
            confidence=data.get("confidence", 0.5),
            suggestions=suggestions,
            plan=data.get("plan"),
            metadata=metadata,
        )

    @classmethod
    def output_format_schema(cls) -> dict[str, Any]:
        """Return JSON Schema for ClaudeAgentOptions.output_format."""
        return {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["success", "failure", "needs_review", "insufficient_info"],
                    "description": (
                        "Whether YOU completed the requested task — not a verdict "
                        "about the code under review. If you finished the analysis "
                        "(even when it found bugs, regressions, or recommends "
                        "against a change), use 'success'; put the negative finding "
                        "in summary/details/suggestions. Use 'failure' ONLY when "
                        "you could not complete the task itself (tools failed, "
                        "blocked, gave up). 'needs_review' when partially done and "
                        "human judgment is required; 'insufficient_info' when you "
                        "lacked the context to reach any conclusion."
                    ),
                },
                "summary": {
                    "type": "string",
                    "description": "One-line summary of what was accomplished",
                },
                "details": {
                    "type": "string",
                    "description": "Full analysis with evidence and reasoning",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence in the result (0.0–1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "suggestions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Actionable follow-up suggestions",
                },
                "plan": {
                    "type": "object",
                    "description": "Optional structured plan for asicode to execute",
                    "additionalProperties": True,
                },
            },
            "required": ["status", "summary", "details"],
        }
