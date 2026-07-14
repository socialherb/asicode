"""
Execution spec model for Phase 6.0 Spec-Driven Planning Core.

ResolvedExecutionSpec is the CANONICAL planner-side spec.
All planner / decomposition / candidate / ranking code consumes this type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from .config.thresholds import config as _cfg_es
from .enums import EstimatedScope

# ── Exploration signal types ──────────────────────────────────────────────────

@dataclass
class ExplorationHints:
    """Structured weak exploration signal.

    Populated by SpecResolver when confidence is below the "reliable" bar.
    Carries the exploration results as soft guidance rather than discarding them.
    Planner and executor use this to constrain scope without locking in noisy targets.
    """
    files: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    confidence: float = 0.0
    mode: str = "targeted"          # "open_ended" | "diagnostic" | "targeted"
    anomaly: Optional[str] = None   # primary anomaly type if detected
    top1_score: float = 0.0         # score of the best candidate
    top12_margin: float = 0.0       # gap between top-1 and top-2 scores

    def to_dict(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "symbols": self.symbols,
            "confidence": self.confidence,
            "mode": self.mode,
            "anomaly": self.anomaly,
            "top1_score": self.top1_score,
            "top12_margin": self.top12_margin,
        }


@dataclass
class ExplorationSignals:
    """Multi-dimensional exploration quality signals consumed by decide_scope_mode().

    Replaces the single ``confidence < threshold`` pattern with a richer signal
    that accounts for candidate distribution and anomaly presence.
    """
    confidence: float
    top1_score: float
    margin: float       # top1 - top2 score gap
    anomaly: bool       # any anomaly detected?
    convergence_turns: int


def decide_scope_mode(
    signals: ExplorationSignals,
    request_mode: str,
) -> Literal["strict", "guided", "free"]:
    """Determine planner/executor scope constraint from exploration signals.

    Args:
        signals: Multi-dimensional quality signals from exploration.
        request_mode: "open_ended" | "diagnostic" | "targeted"

    Returns:
        "strict"  — only target_files are allowed (high confidence, clear target)
        "guided"  — hints.files define investigation scope (moderate confidence)
        "free"    — no constraint; LLM may discover targets autonomously

    Thresholds use *relative* margin (margin / top1_score) so that a strong
    relative winner is recognised as strict even when absolute scores are low.
    A top1=0.3, margin=0.12 case → relative_margin=0.40 is a clear winner.
    Absolute margin thresholds (0.15/0.20) penalise this incorrectly.
    """
    if request_mode == "open_ended":
        # Open-ended requests are inherently uncertain; always guided so the
        # planner starts from the exploration candidates rather than the full
        # codebase — but is not locked to them.
        return "guided"

    # relative margin: how dominant is the top candidate?
    _relative_margin = signals.margin / max(signals.top1_score, 0.01)

    if request_mode == "diagnostic":
        # Diagnostic: strict when top-1 has a clear relative lead
        if signals.top1_score > _cfg_es.scores.STRICT_TOP1 and _relative_margin > _cfg_es.scores.STRICT_MARGIN and not signals.anomaly:
            return "strict"
        return "guided"

    # targeted
    if signals.top1_score > _cfg_es.scores.STRICT_TARGETED_TOP1 and _relative_margin > _cfg_es.scores.STRICT_TARGETED_MARGIN and not signals.anomaly:
        return "strict"
    if signals.top1_score > _cfg_es.scores.GUIDED_TOP1:
        return "guided"

    # Confidence too low to constrain — let LLM resolve from scratch
    return "free"


# Canonical planner/execution spec
@dataclass
class ResolvedExecutionSpec:
    """
    Canonical representation of a user request.

    Phase 6.0: planner consumes this instead of raw request.
    """

    original_request: str

    intent: str
    request_type: str

    target_files: list[str] = field(default_factory=list)
    new_files: list[str] = field(default_factory=list)
    target_symbols: list[str] = field(default_factory=list)

    # Files explicitly mentioned in the request as reference/examples — read-only context.
    # Populated by SpecResolver when a mentioned file is not a target or new file.
    reference_files: list[str] = field(default_factory=list)

    constraints: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)

    risk_level: str = "medium"
    estimated_scope: EstimatedScope = EstimatedScope.SMALL

    # Populated by proposal→ChangeSpec converter when adaptive exploration runs.
    # action_hint: what kind of change is needed ("modify_logic", "modify_scoring", etc.)
    # change_goal: one-line description of the change goal for this spec
    action_hint: str = ""
    change_goal: str = ""

    suggested_strategies: list[str] = field(default_factory=list)

    # Primary language detected from target files (e.g. "python", "typescript")
    language: Optional[str] = None

    metadata: dict[str, Any] = field(default_factory=dict)

    # Optional GSG reasoning pack context for planning (populated by design-chat handoff)
    # Pre-analyzed code snippets from Design Chat analysis.
    # Each entry: {"reason": str, "file": str, "snippet": str}
    # Injected into planner prompt directly — no file I/O.
    code_context: list[dict[str, str]] = field(default_factory=list)
    gsg_context: str = ""

    # Integration targets: entry points that MUST wire in the new modules.
    # Populated by SpecResolver._detect_entry_points() for new runtime modules.
    # Format: [{"file": str, "symbol": str, "reason": str}]
    integration_targets: list[dict[str, str]] = field(default_factory=list)

    # ── Exploration signal fields (Phase: structured weak signal) ─────────────
    # Structured weak signals from adaptive exploration.
    hints: Optional[ExplorationHints] = None

    # Scope constraint derived from exploration signals via decide_scope_mode().
    #   "strict"  — only target_files are allowed write targets
    #   "guided"  — hints.files define the investigation scope
    #   "free"    — no constraint (LLM may discover targets autonomously)
    scope_mode: str = "free"

    # Whether target_files are authoritative (high-confidence, locked targets) or
    # non-authoritative (soft guidance from low-confidence exploration).
    # When False: target_files == hints.files (investigation starting point, not authority).
    # Planner uses this to decide whether to constrain or expand investigation.
    authoritative: bool = True

    # ── Intent-layer targets (authoritative, pre-grounding) ───────────────────
    # Canonical file paths / symbol names from IntentResult, preserved separately
    intent_files: list[str] = field(default_factory=list)
    intent_symbols: list[str] = field(default_factory=list)

    # ── Target provenance ───────────────────────────────────────────────────────
    # Structured signal indicating how target_files/target_symbols were determined.
    target_provenance: str = "explicit"

    # ── Symbol role classification ─────────────────────────────────────────────
    # Derived from IntentResult.modify_symbols / reference_symbols.
    modify_symbols: list[str] = field(default_factory=list)
    reference_symbols: list[str] = field(default_factory=list)

    # ── EvidenceState: symbol resolution quality ─────────────────────────────
    # Mirrors the "clean_empty vs suspicious_empty" principle: when
    had_symbol_mentions: bool = False  # True if intent_symbols was non-empty
    unresolved_mentions: list[str] = field(default_factory=list)  # intent_symbols - target_symbols
    analysis_notes: list[dict[str, str]] = field(default_factory=list)

    def _recompute_symbol_evidence(self) -> None:
        """Recompute ``had_symbol_mentions`` and ``unresolved_mentions`` from the
        current ``intent_symbols`` and ``target_symbols``.  SpecResolver calls
        this after finishing grounding so the fields reflect the final state.
        """
        self.had_symbol_mentions = bool(self.intent_symbols)
        _resolved_set = set(self.target_symbols)
        self.unresolved_mentions = [
            s for s in self.intent_symbols if s not in _resolved_set
        ]

    @property
    def suspicious_empty_symbols(self) -> bool:
        """True when target_symbols is empty despite the intent mentioning symbols.

        Equivalent to ``suspicious_empty`` in the EvidenceState pattern:
          * ``False`` — no symbols were mentioned; broad exploration is correct
          * ``True``  — symbols were mentioned but couldn't be resolved;
                        consider typo detection, re-exploration, or CLARIFY
        """
        return not self.target_symbols and self.had_symbol_mentions

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_request": self.original_request,
            "intent": self.intent,
            "request_type": self.request_type,
            "target_files": self.target_files,
            "new_files": self.new_files,
            "target_symbols": self.target_symbols,
            "intent_files": self.intent_files,
            "intent_symbols": self.intent_symbols,
            "modify_symbols": self.modify_symbols,
            "reference_symbols": self.reference_symbols,
            "reference_files": self.reference_files,
            "constraints": self.constraints,
            "acceptance_criteria": self.acceptance_criteria,
            "risk_level": self.risk_level,
            "estimated_scope": self.estimated_scope,
            "action_hint": self.action_hint,
            "change_goal": self.change_goal,
            "suggested_strategies": self.suggested_strategies,
            "language": self.language,
            "metadata": self.metadata,
            "hints": self.hints.to_dict() if self.hints is not None else None,
            "code_context": self.code_context,
            "scope_mode": self.scope_mode,
            "authoritative": self.authoritative,
            "target_provenance": self.target_provenance,
            "had_symbol_mentions": self.had_symbol_mentions,
            "unresolved_mentions": self.unresolved_mentions,
        }

    def adjust_scope_from_graph(self) -> None:
        """Upgrade estimated_scope based on graph_context if available.

        Called after graph enrichment to correct underestimated scope.
        Rules:
        - impact_files >= 5 → at least "medium"
        - impact_files >= 10 → "large"
        - any target symbol with callers in 4+ files → at least "medium"
        """
        graph_ctx = self.metadata.get("graph_context", {})
        if not graph_ctx:
            return
        if not self.authoritative:
            # Graph fan-out is impact/read context for inferred scopes, not proof
            # that every impacted file is an edit target.  Keep estimated_scope as
            # the edit scope; callers can still inspect graph_context for verification.
            self.metadata["scope_upgrade_suppressed"] = "non_authoritative_targets"
            return

        impact_files = graph_ctx.get("impact_files", [])
        callers_map = graph_ctx.get("callers", {})

        # Check impact file count
        if len(impact_files) >= 10 and self.estimated_scope != EstimatedScope.LARGE:
            self.metadata["scope_upgraded_from"] = self.estimated_scope
            self.estimated_scope = EstimatedScope.LARGE
            return
        if len(impact_files) >= 5 and self.estimated_scope in (EstimatedScope.SMALL, EstimatedScope.TINY):
            self.metadata["scope_upgraded_from"] = self.estimated_scope
            self.estimated_scope = EstimatedScope.MEDIUM

        # Check per-symbol fan-in
        for sym_name in self.target_symbols:
            callers = callers_map.get(sym_name, [])
            unique_files = {c.get("file", "") for c in callers if c.get("file")}
            if len(unique_files) >= 4 and self.estimated_scope in (EstimatedScope.SMALL, EstimatedScope.TINY):
                self.metadata["scope_upgraded_from"] = self.estimated_scope
                self.estimated_scope = EstimatedScope.MEDIUM
                return

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResolvedExecutionSpec":
        """Construct from a serialized dict (e.g. from to_dict())."""
        raw_hints = data.get("hints")
        hints: Optional[ExplorationHints] = None
        if raw_hints is not None:
            hints = ExplorationHints(
                files=raw_hints.get("files", []),
                symbols=raw_hints.get("symbols", []),
                confidence=raw_hints.get("confidence", 0.0),
                mode=raw_hints.get("mode", "targeted"),
                anomaly=raw_hints.get("anomaly"),
                top1_score=raw_hints.get("top1_score", 0.0),
                top12_margin=raw_hints.get("top12_margin", 0.0),
            )
        return cls(
            original_request=data.get("original_request", ""),
            intent=data.get("intent", ""),
            request_type=data.get("request_type", ""),
            target_files=data.get("target_files", []),
            new_files=data.get("new_files", []),
            target_symbols=data.get("target_symbols", []),
            intent_files=data.get("intent_files", []),
            intent_symbols=data.get("intent_symbols", []),
            modify_symbols=data.get("modify_symbols", []),
            reference_symbols=data.get("reference_symbols", []),
            reference_files=data.get("reference_files", []),
            code_context=data.get("code_context", []),
            constraints=data.get("constraints", []),
            acceptance_criteria=data.get("acceptance_criteria", []),
            risk_level=data.get("risk_level", "medium"),
            estimated_scope=data.get("estimated_scope", EstimatedScope.SMALL),
            action_hint=data.get("action_hint", ""),
            change_goal=data.get("change_goal", ""),
            suggested_strategies=data.get("suggested_strategies", []),
            language=data.get("language"),
            metadata=data.get("metadata", {}),
            hints=hints,
            scope_mode=data.get("scope_mode", "free"),
            authoritative=data.get("authoritative", True),
            target_provenance=data.get("target_provenance", "explicit"),
            had_symbol_mentions=data.get("had_symbol_mentions", False),
            unresolved_mentions=data.get("unresolved_mentions", []),
        )


# ── User-pinned spec sources ──────────────────────────────────────────────────
# Sources that carry user authority for target_files/target_symbols.
# Migrated here from spec_resolver_dataclasses (SpecResolver subsystem removed);
# the prebuilt-bypass (Design Chat) path does not set this metadata key, so
# is_user_pinned_spec() returns False for prebuilt specs (preserving prior behavior).
USER_PINNED_SPEC_SOURCES: frozenset = frozenset({
    "intent_direct",      # IntentResult.target_files/symbols (LLM extracted from request)
    "router_hints",       # Router-extracted hints (explicit path / backtick)
    "create_fast_path",   # User named a new file to create
    "backtick_ripgrep",   # Identifier the user quoted with backticks
})


def is_user_pinned_spec(spec: Any) -> bool:
    """True iff spec's grounding source is in the user-pinned whitelist."""
    if spec is None:
        return False
    _meta = getattr(spec, "metadata", {}) or {}
    _src = str(_meta.get("spec_resolver_source") or "").strip()
    return _src in USER_PINNED_SPEC_SOURCES


estimated_scope: EstimatedScope = EstimatedScope.SMALL
