"""
Problem signature: structured representation of a task context for experience matching.

Used to find similar past experiences and extract applicable patterns.
"""
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from external_llm.languages import LanguageId

logger = logging.getLogger(__name__)


@dataclass
class ProblemSignature:
    """Structured representation of a problem context."""
    symbol: str = ""
    module: str = ""
    failure_type: str = ""
    risk_level: str = "unknown"
    impact_size: str = "unknown"  # "small" | "medium" | "large"
    operation_kind: str = ""
    request_type: str = ""

    @property
    def key(self) -> str:
        """Deterministic key for exact matching."""
        parts = [self.failure_type, self.module, self.symbol, self.operation_kind]
        return "|".join(p for p in parts if p)

    @property
    def fingerprint(self) -> str:
        """SHA256 fingerprint for storage."""
        raw = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "module": self.module,
            "failure_type": self.failure_type,
            "risk_level": self.risk_level,
            "impact_size": self.impact_size,
            "operation_kind": self.operation_kind,
            "request_type": self.request_type,
        }

    def to_text(self) -> str:
        """Natural-language serialization for embedding-based semantic matching.

        The field label is included alongside each value (not just the value)
        because it anchors the embedding in the right semantic neighborhood:
        two signatures that share ``failure_type`` are conceptually the same
        *kind* of problem even when the values differ (e.g. ``test_failure``
        vs ``apply_failed``), and including the label ``failure_type`` makes
        that structural correspondence legible to the embedding model. Empty
        fields are omitted so absent attributes do not dilute the signal.
        """
        parts: list[str] = []
        if self.failure_type:
            parts.append(f"failure_type {self.failure_type}")
        if self.module:
            parts.append(f"module {self.module}")
        if self.symbol:
            parts.append(f"symbol {self.symbol}")
        if self.operation_kind:
            parts.append(f"operation_kind {self.operation_kind}")
        if self.request_type:
            parts.append(f"request_type {self.request_type}")
        if self.risk_level and self.risk_level != "unknown":
            parts.append(f"risk_level {self.risk_level}")
        if self.impact_size and self.impact_size != "unknown":
            parts.append(f"impact_size {self.impact_size}")
        return "\n".join(parts)

    def similarity_score(self, other: "ProblemSignature") -> float:
        """
        Compute similarity between two signatures (0.0-1.0).

        Weighted by field importance:
        - failure_type: 0.3
        - module: 0.25
        - symbol: 0.2
        - risk_level: 0.1
        - operation_kind: 0.1
        - request_type: 0.05
        """
        score = 0.0
        if self.failure_type and self.failure_type == other.failure_type:
            score += 0.3
        if self.module and self.module == other.module:
            score += 0.25
        if self.symbol and self.symbol == other.symbol:
            score += 0.2
        if self.risk_level and self.risk_level == other.risk_level:
            score += 0.1
        if self.operation_kind and self.operation_kind == other.operation_kind:
            score += 0.1
        if self.request_type and self.request_type == other.request_type:
            score += 0.05
        return score


def build_problem_signature(
    changed_symbols: Optional[list[str]] = None,
    failure_analysis=None,
    composite_risk=None,
    graph_context: Optional[dict] = None,
    operation_kind: str = "",
    request_type: str = "",
) -> ProblemSignature:
    """
    Build a ProblemSignature from available execution context.

    Never raises — returns empty signature on error.
    """
    sig = ProblemSignature(request_type=request_type, operation_kind=operation_kind)

    try:
        # Primary symbol
        if changed_symbols:
            sig.symbol = changed_symbols[0]
            # Extract module from symbol (e.g., "PlannerAgent.create_plan" → dotted path)
            if '.' in sig.symbol:
                sig.module = sig.symbol.rsplit('.', 1)[0]

        # Module from graph context
        if graph_context and not sig.module:
            primary_files = graph_context.get("primary_files", [])
            if primary_files:
                # Convert file path to module
                f = primary_files[0]
                if LanguageId.from_path(f) is LanguageId.PYTHON:
                    sig.module = f[:-3].replace('/', '.').replace('\\', '.')

        # Failure type
        if failure_analysis:
            ft = getattr(failure_analysis, 'failure_type', None)
            sig.failure_type = ft.value if hasattr(ft, 'value') else str(ft or "")

        # Risk level
        if composite_risk:
            sig.risk_level = getattr(composite_risk, 'level', 'unknown')

        # Impact size from graph context
        if graph_context:
            impact_count = len(graph_context.get("impact_files", []))
            if impact_count >= 6:
                sig.impact_size = "large"
            elif impact_count >= 3:
                sig.impact_size = "medium"
            else:
                sig.impact_size = "small"

    except Exception as e:
        logger.debug("Problem signature build failed: %s", e)

    return sig
