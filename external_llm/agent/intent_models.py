"""
Intent resolution data models for universal user request understanding.

Key principles:
1. Language-neutral: handle any language (Korean, English, typos, mixed)
2. LLM-powered: rely on planner model's natural language understanding
3. No keyword mapping: let LLM extract appropriate search terms
4. Single source of truth: intent result reused across pipeline
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from .config.thresholds import config as _cfg
from .enums import Complexity, Scope

if TYPE_CHECKING:
    from .guard_ir import GuardIR


@dataclass
class IntentResult:
    """LLM-based intent understanding result.

    Produced by IntentResolver and consumed by TaskRouter, SpecResolver,
    and downstream components to avoid duplicate LLM calls.
    """

    # Original request
    original_request: str

    # Normalized/cleaned version (typo corrected, language normalized)
    normalized_query: str

    # Search terms for codebase search (extracted by LLM)
    search_terms: list[str] = field(default_factory=list)

    # Intent classification
    intent_type: str = "unknown"  # "bugfix", "feature", "refactor", "exploration", "question", "modify", "extend", "create"

    # Target inference (LLM's best guess)
    target_files: list[str] = field(default_factory=list)    # full relative paths
    target_symbols: list[str] = field(default_factory=list)  # function/class names

    # New symbols to be created (not yet in codebase)
    # Populated when intent_type is "extend", "feature", or "create".
    # Each entry is a dict: {"name": str, "kind": "method"|"function"|"class", "parent": str|None}
    # SpecResolver passes these through unchanged — they are owned by the intent layer.
    new_symbols: list[dict[str, Any]] = field(default_factory=list)

    # ── Symbol role classification (intent-aware) ─────────────────────────────
    # modify_symbols: existing symbols that the user explicitly wants to CHANGE.
    modify_symbols: list[str] = field(default_factory=list)
    reference_symbols: list[str] = field(default_factory=list)

    # ── Edit kind classification (Intent → Policy) ───────────────────────────
    # Enables downstream execution policy decisions without re-analyzing intent.
    edit_kind: str = ""

    # Guard statement to insert (only populated when edit_kind == "guard_add")
    # e.g. "if not candidates: return None"
    guard_statement: str = ""

    # Authoritative typed guard IR — populated by IntentResolver (parse_guard on
    # guard_statement) and backfilled/canonicalized by SpecResolver.  Consumers
    # MUST prefer this over guard_statement string to avoid re-parsing.
    # [GUARD_SPEC] source: "intent_resolver" or "spec_resolver"
    guard_spec: Optional["GuardIR"] = None

    # Legacy dict repr (compact + condition + control) — kept for backward compat.
    # New code should read guard_spec instead; this is only written by SpecResolver
    # for callers that still expect a dict (e.g. old tool_schemas telemetry paths).
    guard_ir: Optional[dict[str, Any]] = None

    # Target loop iterable for guard_add when the guard must go inside a specific loop.
    # e.g. "undefined_names" — the iterable expression (as it appears in source code)
    target_loop_iterable: str = ""

    # Lane suggestion
    lane_hint: str = ""  # "planner", "main_agent", "read_only", "clarify"

    # Scope of the change (inferred by LLM from request context)
    # "single_file": change confined to one file/symbol
    # "multi_file": touches two or more files
    # "project_wide": affects many/all files or the whole codebase
    scope_hint: Scope = Scope.SINGLE_FILE

    # Complexity of the change (inferred by LLM)
    # "trivial": single-line, comment, typo, rename, import, constant tweak
    # "normal": typical function/class modification
    # "complex": multi-file, multi-step, architectural change
    complexity_hint: Complexity = Complexity.LOW

    # True when the request asks to write or generate tests
    is_test_write: bool = False

    # True when the request is about code style/formatting (lint, prettier, black)
    # Distinct from refactor: style fix changes formatting only, not logic
    is_style_fix: bool = False

    # True when the request is a filesystem operation (move, rename, delete file/dir)
    # Subset of main_agent tasks — distinct from UI/style changes
    is_filesystem_op: bool = False

    # True when the request targets visual/UI appearance (CSS, colors, layout, icons)
    # Subset of main_agent tasks — distinct from filesystem operations
    is_ui_change: bool = False

    # True when the request explicitly asks to preserve the existing public API /
    # call signature / backward compatibility.  Drives ``intent_policy`` in
    is_interface_preserving: bool = False

    # Confidence in this interpretation
    confidence: float = 0.5  # 0.0-1.0

    # Metadata about the resolution process
    metadata: dict[str, Any] = field(default_factory=dict)  # language detection, typo corrections, transformations

    # Hints for SpecResolver (compatible with existing llm_hints format)
    spec_hints: dict[str, Any] = field(default_factory=dict)  # modify_files, new_files

    # Behavioral code concepts — drives dataflow-anchored symbol resolution.
    # Extracted by IntentResolver and consumed by SpecResolver to find the
    code_concepts: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and normalize fields."""
        # Ensure search_terms are unique
        if self.search_terms:
            unique_terms = []
            seen = set()
            for term in self.search_terms:
                if term not in seen:
                    seen.add(term)
                    unique_terms.append(term)
            self.search_terms = unique_terms

        # Ensure target_files are unique
        if self.target_files:
            self.target_files = list(dict.fromkeys(self.target_files))

        # Ensure target_symbols are unique
        if self.target_symbols:
            self.target_symbols = list(dict.fromkeys(self.target_symbols))

        # Ensure new_symbols entries are valid dicts with at least "name"
        if self.new_symbols:
            seen_names: set = set()
            valid: list = []
            for entry in self.new_symbols:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name", "")
                if not name or name in seen_names:
                    continue
                seen_names.add(name)
                valid.append({
                    "name": name,
                    "kind": entry.get("kind", "function"),
                    "parent": entry.get("parent") or None,
                })
            self.new_symbols = valid

        # Normalize lane_hint
        if self.lane_hint:
            self.lane_hint = self.lane_hint.lower().strip()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "original_request": self.original_request,
            "normalized_query": self.normalized_query,
            "search_terms": self.search_terms,
            "intent_type": self.intent_type,
            "target_files": self.target_files,
            "target_symbols": self.target_symbols,
            "new_symbols": self.new_symbols,
            "modify_symbols": self.modify_symbols,
            "reference_symbols": self.reference_symbols,
            "lane_hint": self.lane_hint,
            "scope_hint": self.scope_hint.value,
            "complexity_hint": self.complexity_hint.value,
            "is_test_write": self.is_test_write,
            "is_style_fix": self.is_style_fix,
            "is_filesystem_op": self.is_filesystem_op,
            "is_ui_change": self.is_ui_change,
            "is_interface_preserving": self.is_interface_preserving,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "spec_hints": self.spec_hints,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IntentResult":
        """Create from dictionary."""
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        if 'scope_hint' in data:
            data['scope_hint'] = Scope(data['scope_hint'])
        if 'complexity_hint' in data:
            data['complexity_hint'] = Complexity(data['complexity_hint'])
        return cls(**{k: v for k, v in data.items() if k in known})

    def is_read_only(self) -> bool:
        """Check if this appears to be a read-only request."""
        return (self.intent_type in ("exploration", "question") or
                self.lane_hint == "read_only")

    def has_edit_intent(self) -> bool:
        """Check if this appears to be an edit request."""
        return self.intent_type in ("bugfix", "feature", "refactor", "modify", "extend", "create")

    def get_spec_hints(self) -> dict[str, Any]:
        """Get SpecResolver hints in compatible format."""
        hints = {}
        if self.target_files:
            hints["modify_files"] = self.target_files
        if "new_files" in self.spec_hints:
            hints["new_files"] = self.spec_hints["new_files"]
        return hints


@dataclass
class IntentResolutionConfig:
    """Configuration for IntentResolver."""

    # LLM client and model (should use planner model)
    llm_client: Any = None
    model: str = ""

    # Cache settings
    enable_cache: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes

    # Resolution parameters
    max_tokens: int = _cfg.tokens.INTENT_RESOLVER_DEFAULT
    max_search_terms: int = 10
    max_target_files: int = 5

    def __post_init__(self):
        """Validate configuration."""
        if not self.model:
            raise ValueError("IntentResolver requires a model name")

    def get_cache_key(self, request: str) -> str:
        """Generate cache key for request."""
        import hashlib
        # Simple hash of request (language + content)
        return hashlib.md5(request.encode('utf-8'), usedforsecurity=False).hexdigest()[:16]
