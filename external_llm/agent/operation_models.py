"""
Canonical operation schema for asicode's hybrid architecture.

Defines the structured operations that a Planner can produce and an Executor can execute.
"""

from __future__ import annotations

import enum
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol

from .guard_ir import GuardIR
from .placement_contract import PlacementContract

# ---------------------------------------------------------------------------
# Satisfaction confidence / source constants (graded noop signal)


# ── Intent Assertion: Planner → Developer semantic constraint ───────────────


class AssertionTier(str, enum.Enum):
    """Semantic tier of an IntentAssertion — determines gate authority.

    EXISTENCE  — asserts that a target (symbol/file) exists or was not removed.
                 Trivially true for any existing symbol BEFORE an edit.
                 Cannot prove a modification happened → excluded from the
                 already_satisfied skip gate for MODIFY_SYMBOL ops.
                 Usage: target-validity guard, blast_radius checks.

    STATE      — asserts that the current code satisfies a structural condition
                 (call present, param exists, import found, guard in scope).
                 May or may not hold pre-execution; naturally fails when the
                 condition is not yet met → safe for gate use.
                 Default tier for most content-verifying assertion kinds.

    CHANGE     — explicitly asserts that the op introduced a NEW structural
                 element.  Planner should set this when a call/import/param
                 was NOT present before the edit.  Provides the strongest
                 completion proof — use as primary evidence in gate + final
                 verification.  Also enables planner validation: every write
                 op should carry at least one CHANGE-tier assertion.

    """
    EXISTENCE = "existence"
    STATE     = "state"
    CHANGE    = "change"


def infer_tier(kind: "IntentAssertionKind") -> AssertionTier:
    """Return the default AssertionTier for a given IntentAssertionKind.

    Explicit ``tier`` override on IntentAssertion always takes precedence.

    EXISTENCE — symbol_exists / symbol_not_removed (trivially true pre-edit).
    CHANGE    — structural-addition kinds: the assertion by default describes
                something NEW being introduced.  An explicit tier=STATE override
                is still respected when the planner means a precondition note.
    STATE     — everything else (behavioral, usage, presupposition checks).
    """
    kind_str = str(kind.value if hasattr(kind, "value") else kind)
    _EXISTENCE_KINDS = frozenset({
        "symbol_exists",
        "symbol_not_removed",
    })
    # Structural-addition kinds: when no explicit tier is set, the default
    # semantics are "this op introduces this new element" → CHANGE tier.
    # A planner that explicitly sets tier=STATE overrides this (e.g. to
    # document a pre-existing precondition rather than a new addition).
    _STRUCTURAL_ADDITION_KINDS = frozenset({
        "symbol_has_param",
        "symbol_returns_type",
        "dict_key_exists",
        "import_exists",
        "guard_in_scope",
        "enum_member_exists",
        "symbol_has_bounded_cache_miss",
        "symbol_contains",
    })
    if kind_str in _EXISTENCE_KINDS:
        return AssertionTier.EXISTENCE
    if kind_str in _STRUCTURAL_ADDITION_KINDS:
        return AssertionTier.CHANGE
    return AssertionTier.STATE


class IntentAssertionKind(str, enum.Enum):
    """Kinds of AST-verifiable assertions that a Planner can attach to an op.

    Each kind maps to a deterministic AST check in intent_verifier.py.
    """
    SYMBOL_EXISTS = "symbol_exists"            # target symbol must exist after edit
    SYMBOL_CALLS = "symbol_calls"              # symbol body must call a specific function (actual call expr)
    SYMBOL_REFERENCES = "symbol_references"    # symbol body must reference a name — broader than symbol_calls:
    #   covers isinstance(x, Foo), type annotations, constant usage, module attribute access.
    #   params: {"symbol": str}  — name or dotted name, e.g. "ast.Call"
    SYMBOL_HAS_PARAM = "symbol_has_param"      # function must have a given parameter
    #   params: {"param_name": str,             — required: parameter identifier
    #            "annotation":  str,            — optional: type annotation (e.g. "str")
    SYMBOL_USES_PARAM = "symbol_uses_param"    # parameter must be referenced in function body (not just signature)
    SYMBOL_RETURNS_TYPE = "symbol_returns_type" # function return annotation matches
    IMPORT_EXISTS = "import_exists"            # a specific import must be present
    SYMBOL_NOT_REMOVED = "symbol_not_removed"  # existing symbol must NOT be deleted
    # ── Structural IR assertions (Stage 3 — vocabulary-aligned with ChangeKind) ──
    DICT_KEY_EXISTS = "dict_key_exists"        # dict/mapping must contain specified key(s)
    #   params: {"keys": ["k1", "k2"]}  — empty list: just check symbol is a dict
    GUARD_IN_SCOPE = "guard_in_scope"          # guard statement exists at the first line of a specific scope
    #   params: {"guard_statement": str,
    #            "insert_scope": "function_body"|"for_loop"|"while_loop",
    LOCAL_GUARD_AFTER_ASSIGNMENT = "local_guard_after_assignment"
    #   Verifies that a guard referencing local variable(s) was inserted immediately
    #   after the nearest dominating effective assignment of those variables.
    ENUM_MEMBER_EXISTS = "enum_member_exists"
    # Call-argument replacement assertion — verifies a string literal was updated
    # inside a specific call expression within target_symbol's body.
    CALL_ARG_CHANGED = "call_arg_changed"
    # Code-presence assertions — check that a symbol body does / does not contain
    # a specific code fragment (normalized whitespace substring match).
    SYMBOL_CONTAINS = "symbol_contains"          # symbol body MUST contain a code fragment
    SYMBOL_NOT_CONTAINS = "symbol_not_contains"  # symbol body must NOT contain a code fragment
    # Structural pattern assertions — AST-level verification of composite patterns
    # that span multiple statements (e.g. cache-miss with eviction, retry loops).
    # Params: none (verifier reconstructs the pattern from AST directly).
    SYMBOL_HAS_BOUNDED_CACHE_MISS = "symbol_has_bounded_cache_miss"
    #   symbol body must contain complete bounded-cache miss pattern:
    #   miss guard + capacity guard + eviction action + store (all nested in same path)
    # Structural negative assertions — AST-level checks for absence (SL58 Phase 2)
    IMPORT_ABSENT = "import_absent"              # a specific import must NOT exist (AST parse)
    DICT_KEY_ABSENT = "dict_key_absent"          # a specific dict key must NOT exist (AST parse)
    SYMBOL_ABSENT = "symbol_absent"              # a top-level symbol must NOT exist (AST parse)
    # Behavioral contract assertion — verifies that a behavioral effect was achieved
    # regardless of the specific syntactic form used (break/return/threshold/enum transition).
    BEHAVIORAL_CONTRACT = "behavioral_contract"
    # Exit-fingerprint preservation assertion — verifies that control-flow exits
    # (return/raise/assert/guard-if) are preserved at a semantic fingerprint level
    EXITS_PRESERVED = "exits_preserved"
    # Scope confinement assertion — verifies that ONLY specified symbol(s) changed.
    # All other symbols in the same file must remain byte-identical to pre-edit state.
    # params: {"confinement_symbols": ["SymbolA", "SymbolB"]}  — the ONLY symbols
    #          that are allowed to change.  Everything else must be untouched.
    SCOPE_CONFINEMENT = "scope_confinement"

    # ── Semantic assertions — behavioral intent, not structural presence ──
    SELECTION_GATED_BY = "selection_gated_by"
    # A selection mutation (append/add/...) on selection_var is conditionally
    # gated by a filter on filter_var.  Verifies that the filter actually controls
    # which items enter the selection, not just that both names are present.
    # params: {"filter_var": str, "selection_var": str}
    SCORE_COMPOSITION = "score_composition"
    # A score/weight attribute (target) is modified via an arithmetic expression
    # (multiplication or addition), meaning a new term is being composed into
    # the scoring path rather than just referenced.
    # params: {"target": str}  — attribute name, e.g. "final_score"
    MEMBERSHIP_GUARD = "membership_guard"
    # A conditional checks (not) membership in a collection, acting as an
    # inclusion/exclusion filter.  Stronger than SELECTION_GATED_BY: verifies
    # the directional membership operator, not just that both names appear.
    # params: {"collection": str, "direction": "exclude"|"include"}
    SLOT_BOUNDED_SELECTION = "slot_bounded_selection"
    # Selection is bounded by a slot quota derived from k - len(collection),
    # ensuring the output respects a capacity limit.
    # params: {"collection": str}  — collection whose length bounds the quota

@dataclass
class IntentAssertion:
    """A single verifiable assertion about post-edit code state.

    Generated by Planner alongside Operation, verified by intent_verifier
    after Developer LLM execution.  Pure AST checks — no LLM calls.

    The ``tier`` field controls gate authority:
    - EXISTENCE: excluded from already_satisfied skip gate for MODIFY_SYMBOL
    - STATE: valid gate evidence (fails pre-edit when condition not yet met)
    - CHANGE: strongest gate/verification evidence; planner should set this
              for assertions that verify a newly introduced structure
    - SEMANTIC: final-verdict authority only (not used in pre-exec gate)
    """
    kind: IntentAssertionKind
    target_file: str                                         # file to check
    target_symbol: str = ""                                  # symbol to check (if applicable)
    params: dict[str, Any] = field(default_factory=dict)     # kind-specific params
    severity: Literal["blocking", "warning"] = "blocking"
    description: str = ""                                    # human-readable explanation
    tier: "AssertionTier" = field(default=None)              # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.tier is None:
            self.tier = infer_tier(self.kind)


class CanonicalAction(str, enum.Enum):
    """Decision output from the canonical loop's _decide_next_action."""
    ACCEPT = "ACCEPT"       # Execution meets acceptance criteria → return
    FAIL = "FAIL"           # Terminal failure → return
    REPAIR = "REPAIR"       # P7-lite: deterministic repair → re-verify within loop
    REPLAN = "REPLAN"       # Generate new plan → re-execute within loop


@dataclass
class CanonicalDecision:
    """Structured decision from the canonical control loop."""
    action: CanonicalAction
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Semantic-Fit Judge: planner's pre-planning verdict ──────────────────────

class SemanticFitAction(str, enum.Enum):
    """Planner's pre-planning verdict action.

    Emitted by PlannerAgent.judge_semantic_fit() before plan generation.
    Decides whether the candidate spec from SpecResolver actually matches
    the user's real intent.
    """
    ACCEPT         = "ACCEPT"          # spec good → proceed to plan directly
    ANALYZE_FIRST  = "ANALYZE_FIRST"   # plausible area but fix point unclear → read/analyze op first
    RE_EXPLORE     = "RE_EXPLORE"      # semantically misaligned → re-run SpecResolver with guidance
    CLARIFY        = "CLARIFY"         # evidence too weak, re-exploration unlikely to help → ask user


@dataclass
class SemanticFitVerdict:
    """Planner's semantic-fit judgement on a candidate ResolvedExecutionSpec.

    Produced as the FIRST output of the planner before any plan generation,
    so upstream (SpecResolver) can be re-invoked with guidance when the
    candidate spec does not match user intent.

    Fields:
        semantic_fit:        strong / weak / wrong / unknown — how well the
                             current spec matches the user request semantically.
        action:              ACCEPT / ANALYZE_FIRST / RE_EXPLORE / CLARIFY.
                             Decoupled from semantic_fit so the caller can
                             override (e.g. downgrade RE_EXPLORE → ANALYZE_FIRST
                             after max retries).
        confidence:          [0.0, 1.0] — planner's self-reported confidence
                             in this verdict.
        reason:              short human-readable reason (for logs/UI/learning).
        reexplore_guidance:  when action=RE_EXPLORE, guidance to pass back to
                             SpecResolver. Shape:
                               { "high_confidence": [str, ...],
                                 "low_confidence": [str, ...],
                                 "notes": str }
    """
    semantic_fit: str = "unknown"     # "strong" | "weak" | "wrong" | "unknown"
    action: SemanticFitAction = SemanticFitAction.ANALYZE_FIRST
    confidence: float = 0.0
    reason: str = ""
    reexplore_guidance: Optional[dict[str, Any]] = None
    # True when the verdict was re-routed from RE_EXPLORE → ANALYZE_FIRST
    # because the user supplied an explicit target file+symbol (routing heuristic,
    routing_downgrade: bool = False
    # True when the explicit-anchor fast-path fired: routing was validated
    # (file+symbol intersection confirmed) but in-symbol semantic content was NOT
    anchor_skip_semantic_check: bool = False
    # When action=CLARIFY and a better refactoring target was found by the structural
    # scanner, these fields carry the suggested redirect target so agent_loop can
    # retarget the spec when the user confirms.
    redirect_symbols: Optional[list] = None   # e.g. ['A._convert_callers', 'A._convert_callees']
    redirect_files: Optional[list] = None     # corresponding file paths

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_fit": self.semantic_fit,
            "action": self.action.value if hasattr(self.action, "value") else str(self.action),
            "confidence": round(float(self.confidence or 0.0), 3),
            "reason": self.reason or "",
            "reexplore_guidance": self.reexplore_guidance or {},
            "routing_downgrade": self.routing_downgrade,
            "anchor_skip_semantic_check": self.anchor_skip_semantic_check,
        }


class OpCategory(enum.Enum):
    """Semantic category for dependency-graph failure propagation.

    CONSTRUCTIVE: modifies file content — other ops may depend on its result.
    ANALYTICAL:   read-only inspection — no side effects, never blocks.

    """
    CONSTRUCTIVE = "constructive"
    ANALYTICAL = "analytical"


class OperationKind(str, enum.Enum):
    """Canonical operation kinds supported by the executor."""
    READ_SYMBOL = "read_symbol"
    MODIFY_SYMBOL = "modify_symbol"
    INSERT_AFTER_SYMBOL = "insert_after_symbol"
    INSERT_AFTER_LINE = "insert_after_line"    # text-based: anchor_pattern + code_snippet for non-AST files (HTML/CSS/JSON)
    UPDATE_CALLERS = "update_callers"
    UPDATE_TESTS = "update_tests"
    SUMMARIZE_ANALYSIS = "summarize_analysis"
    READ_FILE_SEGMENT = "read_file_segment"  # Read a file segment by anchor or line range (non-symbol)
    # Future extension points
    CREATE_FILE = "create_file"
    REPLACE_FILE = "replace_file"
    DELETE_FILE = "delete_file"
    MOVE_SYMBOL = "move_symbol"
    ANCHOR_EDIT = "anchor_edit"
    EXTRACT_FUNCTION = "extract_function"
    INSERT_IMPORT = "insert_import"   # deterministic: add import stmt, no LLM
    REMOVE_IMPORT = "remove_import"   # deterministic: remove import stmt, no LLM
    REMOVE_IMPORT_NAME = "remove_import_name"   # deterministic: remove single name from from X import (A, B) block, no LLM
    ADD_ASSIGN = "add_assign"         # deterministic: add self.field = value to function body, no LLM
    DELETE_SYMBOL_RANGE = "delete_symbol_range"  # deterministic line-range
                                                  # deletion (duplicate / dead block);
                                                  # symbol presence after = caller-declared
                                                  # via metadata.expected_symbol_present_after
    RUN_SCANNER = "run_scanner"                  # execute a registered analysis scanner;
                                                    # result stored in accumulated_context
    OVERWRITE_FILE = "overwrite_file"            # replace an existing file's entire content
                                                    # (file MUST exist; opposite guard from CREATE_FILE)

    @property
    def is_propagation_op(self) -> bool:
        """Deprecated: use ``OP_KIND_POLICY[self].is_propagation_kind()`` instead.

        Kept for backward compatibility; no production call sites remain.
        """
        import warnings
        warnings.warn(
            "OperationKind.is_propagation_op is deprecated; "
            "use OP_KIND_POLICY[kind].is_propagation_kind() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.value in {"update_callers", "update_tests"}


# ── Polarity-ambiguous assertion kinds ────────────────────────────────────
# These assertion kinds verify *presence* of a structural element. On an
POLARITY_AMBIGUOUS_ON_NON_ADDITIVE = frozenset({
    IntentAssertionKind.IMPORT_EXISTS,
    IntentAssertionKind.SYMBOL_EXISTS,
    IntentAssertionKind.SYMBOL_HAS_PARAM,
    IntentAssertionKind.SYMBOL_USES_PARAM,
    IntentAssertionKind.SYMBOL_CALLS,
    IntentAssertionKind.SYMBOL_REFERENCES,
    IntentAssertionKind.SYMBOL_CONTAINS,
    IntentAssertionKind.DICT_KEY_EXISTS,
    IntentAssertionKind.ENUM_MEMBER_EXISTS,
})

_ADDITIVE_OP_KINDS = frozenset({
    OperationKind.INSERT_AFTER_SYMBOL,
    OperationKind.INSERT_IMPORT,
    OperationKind.CREATE_FILE,
    OperationKind.OVERWRITE_FILE,
    OperationKind.EXTRACT_FUNCTION,
})
_ADDITIVE_ACTION_HINTS = frozenset({
    "add", "append", "extend", "create", "insert", "register", "attach",
})


def op_intent_is_clearly_additive(op) -> bool:
    """Return True iff the op's intent is unambiguously to add code.

    Why:
      Pre-edit idempotency gates short-circuit when assertions report
      post-state presence already exists. That is correct for ADD intents
      and wrong for REMOVE/REFACTOR/CLEANUP intents — where presence ==
      not-yet-done. SL58: refactor op with IMPORT_EXISTS assertion on
      `hashlib` skipped LLM and left the import in place (false success).

    How to apply:
      Caller passes the Operation. If this returns False, exclude
      ``POLARITY_AMBIGUOUS_ON_NON_ADDITIVE`` kinds from the
      already_satisfied gate before deciding to short-circuit.

    Signals are structural (no prose / regex):
      • op.kind ∈ additive set (INSERT_*, CREATE_FILE, EXTRACT_FUNCTION)
      • op.metadata.action_hint is an additive verb (add/append/...)
    Conservative default: False.
    """
    if op is None:
        return False
    return _shared_logic_helper(op, _ADDITIVE_OP_KINDS, _REMOVAL_OP_KINDS, _ADDITIVE_ACTION_HINTS)


def _shared_logic_helper(op, positive_kinds_set, negative_kinds_set, positive_hints_set) -> bool:
    """Shared logic for op_intent_is_clearly_additive and op_intent_is_clearly_removal.

    Checks the operation's intent based on its kind and metadata hint.
    Returns True if the op's intent matches the positive set,
    False if it matches the negative set,
    otherwise falls back to checking the action_hint metadata.
    """
    if op is None:
        return False
    if getattr(op, "kind", None) in negative_kinds_set:
        return False
    if getattr(op, "kind", None) in positive_kinds_set:
        return True
    _hint = _safe_get_metadata(op, "action_hint", "").strip().lower()
    return _hint in positive_hints_set


def _safe_get_metadata(op, key: str = "", default: Any = "") -> Any:
    """Safely extract a metadata value from an Operation object.
    Returns default if op is None, metadata is missing, or key is missing.
    """
    if op is None:
        return default
    md = getattr(op, "metadata", None) or {}
    if not key:
        return md
    return md.get(key, default)


class _HasKindAndSymbol(Protocol):
    """Structural protocol for Operation / EditInstruction duck typing."""
    kind: Any
    symbol: Any


def _require_symbol_for_kinds(obj: _HasKindAndSymbol, required_kinds: set) -> None:
    """Raise ValueError if obj.kind is in required_kinds but obj.symbol is empty.

    Module-level function shared by Operation.__post_init__ and
    EditInstruction.__post_init__.  Not a method — call as a free function:
        _require_symbol_for_kinds(op, {OperationKind.MODIFY_SYMBOL, ...})
    """
    if obj.kind in required_kinds and not obj.symbol:
        raise ValueError(f"Symbol is required for operation kind {obj.kind}")


_REMOVAL_OP_KINDS = frozenset({
    OperationKind.DELETE_FILE,
    OperationKind.DELETE_SYMBOL_RANGE,
    OperationKind.REMOVE_IMPORT,
    OperationKind.REMOVE_IMPORT_NAME,
})
_REMOVAL_ACTION_HINTS = frozenset({
    "remove", "delete", "drop", "cleanup",
    "dead_code_removal", "remove_dead_code", "remove_unused",
    # Korean
    "제거", "삭제", "없애", "제거해야", "삭제해야",})
_REMOVAL_SEMANTIC_FAMILIES = frozenset({
    "removal", "deletion", "cleanup", "dead_code_removal",
    "remove_dead_code", "import_cleanup", "unused_import_removal",
})

# Embedding-based backstop for free-text intent: catches removal phrasings the
# fixed _REMOVAL_ACTION_HINTS list misses — synonyms ("wipe out", "purge", "get
# rid of") and other languages — without growing the keyword list per locale.
# Contrastive "other" examples keep additive/refactor intents from matching.
# No-op when the embedding model is unavailable (see semantic_intent.py).
# Balanced contrastive sets — the matcher scores by mean cosine per label, so
# both lists should broadly cover their intent (not just a few keywords). The
# "other" list deliberately includes additive, refactor, rename, fix and
# optimize phrasings that share imperative surface form with removal, which is
# what the margin must separate against.
_REMOVAL_INTENT_EXAMPLES = {
    "removal": [
        "remove the unused import",
        "delete this function",
        "drop the deprecated parameter",
        "clean up the dead code",
        "get rid of the redundant variable",
        "wipe out the old helper",
        "purge unused dependencies",
        "사용하지 않는 import 제거",
        "이 함수를 삭제해줘",
        "필요 없는 코드를 없애줘",
        "데드 코드 정리",
        "중복 코드를 제거",
        "안 쓰는 변수 치워줘",
    ],
    "other": [
        "add a new function",
        "fix the bug in this method",
        "refactor this class for readability",
        "rename the variable",
        "update the documentation",
        "optimize the loop",
        "새로운 기능을 추가",
        "버그를 수정",
        "변수 이름을 변경",
        "코드를 최적화",
        "주석을 추가",
        "에러 처리를 추가",
        "새 모듈을 생성",
        "성능을 개선",
    ],
}

_removal_matcher = None
_removal_matcher_lock = threading.Lock()


def _get_removal_matcher():
    """Lazily build the removal-intent semantic matcher (singleton).

    Imports are deferred so this low-level schema module stays import-light and
    free of cycles when the embedding stack is unused.
    """
    global _removal_matcher
    if _removal_matcher is not None:
        return _removal_matcher
    with _removal_matcher_lock:
        if _removal_matcher is None:
            from .config.thresholds import config as _cfg
            from .semantic_intent import SemanticIntentMatcher
            _removal_matcher = SemanticIntentMatcher(
                _REMOVAL_INTENT_EXAMPLES,
                threshold=_cfg.scores.SEMANTIC_INTENT_MIN,
                margin=_cfg.scores.SEMANTIC_INTENT_MARGIN,
                name="removal-intent",
            )
        return _removal_matcher


def op_intent_is_clearly_removal(op) -> bool:
    """Return True iff the op's intent is unambiguously to remove code.

    Why:
      Generic structural gates (DIFF_REGRESSION_PURE_DELETION, churn-only
      diff purity) are intent-blind: they classify deletion-only or
      import-only diffs as defective. For a legitimate cleanup intent
      that is exactly the expected outcome — flagging it forces a useless
      replan that often produces churn or 0-edit success. SL58 op1
      ("Remove unused hashlib import") was rejected by both gates even
      though the patch was correct.

    How to apply:
      Caller passes the Operation. If this returns True, the gate should
      *not* reject deletion-only / import-only changes for that op.

    Signals are structural (no prose):
      • op.kind ∈ removal set (DELETE_FILE, DELETE_SYMBOL_RANGE)
      • op.action_class == "delete"
      • op.metadata.action_hint is a removal verb (remove/delete/cleanup/...)
      • op.metadata.semantic_change_family is a removal family
    Conservative default: False (planner-emitted free-text fields with
    unrecognized values fall through to existing gate behavior).
    """
    if _shared_logic_helper(op, _REMOVAL_OP_KINDS, set(), _REMOVAL_ACTION_HINTS):
        return True
    # action_class="delete" is a typed deletion signal — even for MODIFY_SYMBOL.
    _action_class = getattr(op, "action_class", None) or ""
    if _action_class.strip().lower() == "delete":
        return True
    # Fallback: scan free-text intent for removal keywords (catches intent
    # fields where the planner preserved removal language).
    _intent = (getattr(op, "intent", None) or "").strip().lower()
    if any(bool(re.search(rf'\b{re.escape(_kw)}\b', _intent)) for _kw in _REMOVAL_ACTION_HINTS):
        return True
    _family = _safe_get_metadata(op, "semantic_change_family", "").strip().lower()
    if _family in _REMOVAL_SEMANTIC_FAMILIES:
        return True
    # Last resort: embedding similarity catches removal synonyms / other-language
    # phrasings the keyword list above misses. Runs only after all structural and
    # keyword signals miss, so the common path pays no embedding cost; a no-op
    # (returns False) when the embedding model is unavailable.
    if _intent and _get_removal_matcher().matches(_intent, "removal"):
        return True
    return False




@dataclass(frozen=True)
class OperationKindPolicy:
    """Declarative policy flags for a single OperationKind.

    Single source of truth for kind-level behaviour — replaces scattered
    frozenset constants (FILE_MODIFYING_KINDS, _WRITE_OP_KINDS, …) that
    previously diverged whenever a new kind was added.
    """
    modifies_files: bool = False       # writes to ≥1 existing file
    creates_files: bool = False        # creates new files (not pre-existing)
    deletes_files: bool = False        # removes files from disk
    deletes_symbols: bool = False    # removes symbols from file (e.g. DELETE_SYMBOL_RANGE)
    requires_preflight: bool = False   # needs _preflight_check_file_issues (currently equivalent to modifies_files; reserved for future ops that modify files but skip preflight)
    allows_directory_target: bool = False  # path may be a dir, not a single file
    read_only: bool = False            # never produces a write op
    already_sat_eligible: bool = False # already_satisfied verdict is meaningful
    convergence_excluded: bool = False  # exclude from convergence check
    symbol_edit: bool = False          # edits a symbol in-place (surgical window)
    propagation_op: bool = False       # propagates a parent op's changes
    precheck_eligible: bool = False    # eligible for pre-LLM assertion precheck
    inplace: bool = False             # modifies existing file content in-place (no file creation)
    category: OpCategory = OpCategory.CONSTRUCTIVE  # default; override for read-only/cleanup


# Central policy table. Add a new OperationKind here and all derived
# frozensets (below) update automatically — no more parallel frozenset edits.
OP_KIND_POLICY: dict[OperationKind, OperationKindPolicy] = {
    OperationKind.READ_SYMBOL: OperationKindPolicy(
        read_only=True, convergence_excluded=True,
        category=OpCategory.ANALYTICAL,
    ),
    OperationKind.READ_FILE_SEGMENT: OperationKindPolicy(
        read_only=True, convergence_excluded=True,
        category=OpCategory.ANALYTICAL,
    ),
    OperationKind.SUMMARIZE_ANALYSIS: OperationKindPolicy(
        read_only=True, convergence_excluded=True,
        category=OpCategory.ANALYTICAL,
    ),
    OperationKind.MODIFY_SYMBOL: OperationKindPolicy(
        modifies_files=True, requires_preflight=True,
        already_sat_eligible=True,
        symbol_edit=True, precheck_eligible=True,
        inplace=True,
    ),
    OperationKind.INSERT_AFTER_SYMBOL: OperationKindPolicy(
        modifies_files=True, requires_preflight=True,
        already_sat_eligible=True,
        symbol_edit=True, inplace=True,
    ),
    OperationKind.INSERT_AFTER_LINE: OperationKindPolicy(
        modifies_files=True, requires_preflight=True,
        already_sat_eligible=True, inplace=True,
    ),
    OperationKind.ANCHOR_EDIT: OperationKindPolicy(
        modifies_files=True, requires_preflight=True,
        already_sat_eligible=True, symbol_edit=True,
        precheck_eligible=True, inplace=True,
    ),
    OperationKind.CREATE_FILE: OperationKindPolicy(
        creates_files=True, requires_preflight=True,
    ),
    OperationKind.REPLACE_FILE: OperationKindPolicy(
        modifies_files=True, requires_preflight=True, inplace=True,
    ),
    OperationKind.OVERWRITE_FILE: OperationKindPolicy(
        modifies_files=True, requires_preflight=True, inplace=True,
    ),
    OperationKind.DELETE_FILE: OperationKindPolicy(
        deletes_files=True, requires_preflight=True,
    ),
    OperationKind.MOVE_SYMBOL: OperationKindPolicy(
        modifies_files=True, requires_preflight=True,
        convergence_excluded=True,
    ),
    OperationKind.UPDATE_CALLERS: OperationKindPolicy(
        modifies_files=True, requires_preflight=True,
        allows_directory_target=True, propagation_op=True,
    ),
    OperationKind.UPDATE_TESTS: OperationKindPolicy(
        modifies_files=True, requires_preflight=True,
        allows_directory_target=True, propagation_op=True,
    ),
    OperationKind.EXTRACT_FUNCTION: OperationKindPolicy(
        modifies_files=True, requires_preflight=True,
        convergence_excluded=True,
    ),
    OperationKind.INSERT_IMPORT: OperationKindPolicy(
        modifies_files=True, requires_preflight=True,
    ),
    OperationKind.REMOVE_IMPORT: OperationKindPolicy(
        modifies_files=True, requires_preflight=True,
    ),
    OperationKind.REMOVE_IMPORT_NAME: OperationKindPolicy(
        modifies_files=True, requires_preflight=True,
    ),
    OperationKind.ADD_ASSIGN: OperationKindPolicy(
        modifies_files=True, requires_preflight=True,
        inplace=True,
    ),
    OperationKind.DELETE_SYMBOL_RANGE: OperationKindPolicy(
        modifies_files=True, requires_preflight=True,
        # symbol_edit=False so the symbol-edit invariants (selector ladder,
        # semantic-verifier "symbol must remain") don't fire.
        already_sat_eligible=True,
        deletes_symbols=True,
    ),
    OperationKind.RUN_SCANNER: OperationKindPolicy(
        read_only=True, convergence_excluded=True,
        category=OpCategory.ANALYTICAL,
    ),
}


def _kinds_where(flag: str) -> frozenset[OperationKind]:
    """Return frozenset of OperationKind members where the named flag is True."""
    return frozenset(k for k, p in OP_KIND_POLICY.items() if getattr(p, flag, False))


def is_propagation_kind(kind: OperationKind) -> bool:
    """True if *kind* is a propagation op (UPDATE_CALLERS / UPDATE_TESTS).

    Preferred over ``kind.is_propagation_op`` because it reads directly from
    OP_KIND_POLICY, keeping the property and policy in sync automatically.
    """
    return OP_KIND_POLICY.get(kind, OperationKindPolicy()).propagation_op


class PlanMode(str, enum.Enum):
    """High-level plan mode indicating whether the plan includes edits."""
    EDIT = "edit"
    READ_ONLY_ANALYSIS = "read_only_analysis"


class ExecutionPhase(str, enum.Enum):
    """Tracks which phase of a multi-phase plan we are in.

    SINGLE_PHASE   — normal one-shot plan (no deferred writes).
    PHASE1_ANALYSIS — first phase: read/analyze only; requires_phase2=True means
                     a second execution phase will be generated from Phase 1 outputs.
    PHASE2_EXECUTION — second phase: modification ops generated from Phase 1 outputs.
    """
    SINGLE_PHASE = "single_phase"
    PHASE1_ANALYSIS = "phase1_analysis"
    PHASE2_EXECUTION = "phase2_execution"


class DeferredWriteMode(str, enum.Enum):
    """Why writes were deferred to Phase 2.

    ANALYZE_THEN_MODIFY  — plan has SUMMARIZE_ANALYSIS that produces a FixSpec;
                           Phase 2 regenerates modify ops from FixSpec.
    UPSTREAM_UNVERIFIED  — upstream symbol was not yet read; inject upstream READ first.
    READ_THEN_FIXSPEC    — semantic-fit judge flagged presupposition mismatch;
                           read symbol body first to reconcile or abstain.
    """
    ANALYZE_THEN_MODIFY = "analyze_then_modify"
    UPSTREAM_UNVERIFIED = "upstream_unverified"
    READ_THEN_FIXSPEC = "read_then_fixspec"


class OpStatus(str, enum.Enum):
    """Canonical op/plan result status codes.

    Using ``str, enum.Enum`` means each member compares equal to its string
    value, so existing dict comparisons like ``result["status"] == "success"``
    continue to work unchanged when producers switch to ``OpStatus.SUCCESS``.
    """
    SUCCESS              = "success"
    PARTIAL_SUCCESS      = "partial_success"
    SUCCESS_WITH_WARNINGS = "success_with_warnings"
    COMPLETED            = "completed"
    COMPLETED_PARTIAL    = "completed_partial"
    ALREADY_SATISFIED    = "already_satisfied"
    SKIPPED              = "skipped"
    FAILED               = "failed"
    ERROR                = "error"
    NOT_FOUND            = "not_found"
    VERIFICATION_FAILED  = "verification_failed"
    EXECUTION_ERROR      = "execution_error"
    PREFLIGHT_FAILED     = "preflight_failed"
    ROLLBACK             = "rollback"
    UNFULFILLED          = "unfulfilled"
    INVALIDATED          = "invalidated"
    NO_DIFF              = "no_diff_generated"
    BLOCKED              = "blocked"


class FailureClass(str, enum.Enum):
    """Canonical failure classification codes.

    Using ``str, enum.Enum`` preserves JSON serialisation and dict key equality
    with legacy plain-string values.  ``normalize_failure_class()`` maps unknown
    or legacy strings to ``UNKNOWN`` so callers never receive a raw string.
    """
    # ── edit / patch failures ──────────────────────────────────────────────
    SEARCH_STRING_MISMATCH   = "search_string_mismatch"
    NO_DIFF_GENERATED        = "no_diff_generated"
    NO_EFFECT                = "no_effect"
    NO_EFFECTIVE_PROGRESS    = "no_effective_progress"
    NO_OP_EDIT               = "no_op_edit"
    MODIFY_FAILED            = "modify_failed"
    INSERT_FAILED            = "insert_failed"
    PATCH_APPLY_FAILED       = "patch_apply_failed"
    WRITE_ERROR              = "write_error"
    READ_ERROR               = "read_error"

    # ── structural / AST failures ─────────────────────────────────────────
    SYNTAX_ERROR             = "syntax_error"
    SYNTAX_ERROR_AFTER_PATCH = "syntax_error_after_patch"
    SYNTAX_INVALID_AFTER_EDIT = "syntax_invalid_after_edit"
    AST_OP_FAILED            = "ast_op_failed"
    STRUCTURAL_GATE_VIOLATION = "structural_gate_violation"
    BLAST_RADIUS_VIOLATION   = "blast_radius_violation"
    OVERBROAD_EDIT           = "overbroad_edit"
    DEAD_CODE_INTRODUCED     = "dead_code_introduced"
    DECORATOR_DELETION       = "decorator_deletion"
    REGRESSION_PURE_DELETION = "regression_pure_deletion"
    REGION_RELOCATION_FAILED = "region_relocation_failed"
    EXTRACTION_ISOMORPHISM_FAILED = "extraction_isomorphism_failed"
    EXTRACTION_EQUIVALENCE_FAILED = "extraction_equivalence_failed"

    # ── anchor / target resolution ────────────────────────────────────────
    ANCHOR_MISS              = "anchor_miss"
    ANCHOR_LOSS              = "anchor_loss"
    # The LLM passed a multiline (``\n``-joined) anchor_pattern. The exact
    # matcher (``pattern in line``) can never match a single file line, so the
    # call would fall through to a fuzzy fallback and fail opaquely. Rejected
    # up front by detect_multiline_anchor() with an actionable error. Distinct
    # from ANCHOR_MISS (pattern genuinely absent) so the repair ladder can
    # steer the caller to use the first line + context_after, not retry-blind.
    ANCHOR_MULTILINE_PATTERN = "anchor_multiline_pattern"
    # A multiline anchor_pattern was accepted (it has non-empty lines) but a
    # later pattern line does NOT match the corresponding file line, or the
    # pattern extends past EOF. Emitted by anchor_shared.resolve_multiline_anchor
    # AFTER the first line matched, so it is distinct from ANCHOR_MISS (first
    # line absent) and ANCHOR_MULTILINE_PATTERN (rejected up front): the caller
    # had the right first line but a wrong/extra follow-on line, so the repair
    # hint is "re-read and provide the exact block" rather than "first line
    # wrong" or "use a single line".
    MULTILINE_MISMATCH       = "multiline_mismatch"
    ANCHOR_NOT_UNIQUE        = "anchor_not_unique"
    # The LLM-provided code_snippet for insert_before/insert_after contains a
    # copy of code already present around the anchor (the "fragment duplication"
    # failure mode). Pre-detected BEFORE the insert so the caller gets an
    # immediate, actionable failure instead of an opaque post-write syntax
    # error. Distinct from SYNTAX_INVALID_AFTER_EDIT (a generic post-write
    # syntax break) so the repair ladder can tell the caller "re-read and
    # provide only the NEW lines".
    FRAGMENT_DUPLICATION     = "fragment_duplication"
    TARGET_NOT_FOUND         = "target_not_found"
    FILE_NOT_FOUND           = "file_not_found"
    MISSING_PATH             = "missing_path"
    INVALID_LINE_RANGE       = "invalid_line_range"

    # ── semantic / verification failures ─────────────────────────────────
    SEMANTIC_VERIFICATION_FAILED = "semantic_verification_failed"
    SEMANTIC_VERIFY_FAILED   = "semantic_verify_failed"   # legacy alias
    VERIFICATION_FAILED      = "verification_failed"
    INTENT_ASSERTION_FAILED  = "intent_assertion_failed"
    # Target file on disk does not parse — assertion verifier could not
    # evaluate any structural check.  Distinct from INTENT_ASSERTION_FAILED
    # so the strategy ladder routes "broken disk" → revert/replan with
    # disk-aware context, not "cause_anchor_not_found" misdiagnosis.
    TARGET_FILE_SYNTAX_BROKEN = "target_file_syntax_broken"
    PRESUPPOSITION_VIOLATED  = "presupposition_violated"
    PLACEMENT_VIOLATION      = "placement_violation"
    EXTRACTION_VERIFY_FAILED = "extraction_verify_failed"
    MODULE_IMPORT_GATE       = "module_import_gate"

    # ── name / import errors ──────────────────────────────────────────────
    STRUCTURAL_HALLUCINATION = "structural_hallucination"
    F821_UNDEFINED_NAME      = "f821_undefined_name"
    UNDEFINED_NAME           = "undefined_name"
    NAME_REFERENCE_ERROR     = "name_reference_error"
    INVALID_IMPORT_STMT      = "invalid_import_stmt"
    INVALID_IMPORT_MODULE_PATH = "invalid_import_module_path"

    # ── plan / op-level failures ──────────────────────────────────────────
    BAD_OP_SPEC              = "bad_op_spec"
    EXECUTION_ERROR          = "execution_error"
    EXECUTION_FAILED         = "execution_failed"
    DEPENDENCY_BLOCKED       = "dependency_blocked"
    MIXED_FAILURE            = "mixed_failure"
    SEMANTIC_GATE_FAILED     = "semantic_gate_failed"
    PLAN_ACCEPTANCE_FAILED   = "plan_acceptance_failed"
    SIGNATURE_CHANGED        = "signature_changed"
    LINT_ERROR               = "lint_error"
    TIMEOUT                  = "timeout"
    API_CONNECTION_ERROR     = "api_connection_error"
    MODIFY_INTENT_WITHOUT_EDIT_OP = "modify_intent_without_edit_op"
    NOT_EXECUTED             = "not_executed"
    PLANNER_ACTION_ERROR     = "planner_action_error"

    # ── acceptance / alignment ────────────────────────────────────────────
    ACCEPTANCE_FAILED        = "acceptance_failed"
    ALIGNMENT_REJECTED       = "alignment_rejected"

    # ── no-op / satisfied ────────────────────────────────────────────────
    ALREADY_SATISFIED        = "already_satisfied"
    ALREADY_EQUAL            = "already_equal"

    # ── sentinel ─────────────────────────────────────────────────────────
    UNKNOWN                  = "unknown"
    UNSPECIFIED              = "unspecified"
    NONE                     = "none"

    # ── structural / parse (legacy aliases) ──────────────────────────────
    AST_FAILED               = "ast_failed"
    INSERT_POSITION_UNKNOWN  = "insert_position_unknown"
    SYMBOL_NOT_FOUND         = "symbol_not_found"


# Pre-built lookup for O(1) normalisation.
_FAILURE_CLASS_BY_VALUE: dict[str, FailureClass] = {fc.value: fc for fc in FailureClass}


def normalize_failure_class(value: str | None) -> FailureClass:
    """Map a raw failure_class string (or None) to ``FailureClass``.

    Unknown / legacy strings and None both return ``FailureClass.UNKNOWN``
    so callers never have to handle raw strings defensively.
    """
    if not value:
        return FailureClass.UNKNOWN
    # Case-insensitive to tolerate NO_PROGRESS-style uppercase legacy values.
    return _FAILURE_CLASS_BY_VALUE.get(value) or _FAILURE_CLASS_BY_VALUE.get(value.lower(), FailureClass.UNKNOWN)


# ── ExecutionState.skipped_ops reason codes ─────────────────────────────────


class SkipReason(str, enum.Enum):
    """Canonical head codes for ``ExecutionState.skipped_ops`` reason strings.

    The reason string written by ``state.mark_skipped(op_id, reason)`` follows
    the convention ``f"<head>:<detail>"`` (or just ``<head>`` for parameterless
    reasons). The HEAD must be one of these enum values. Classification code
    must look up the head via ``parse_skip_reason`` / ``is_dependency_related_skip``
    rather than substring-matching the raw string — adding a new SkipReason
    forces a deliberate choice about whether it belongs to the dependency
    cascade category, instead of accidentally matching by substring overlap.

    Categories:
      * dependency cascade — ``_DEPENDENCY_RELATED_SKIP_REASONS``: this op
        never got the chance to execute because of another op's failure or
        a graph topology problem (cycle, unschedulable).
      * non-dependency — gate-already-satisfied, nested-inside-parent, and
        aborted-after-critical-failure (the abort is a single-failure decision,
        not a cascade — keeping it out of the dependency category prevents
        ``has_dependency_skips`` from misrouting force-finish/critical-edit
        aborts to dependency-targeted repair).
    """
    GATE_ALREADY_SATISFIED         = "gate_already_satisfied"
    NESTED_INSIDE_PARENT           = "nested_inside_parent_op"
    GROUP_ABORTED                  = "group_aborted"
    BLOCKED_BY_FAILED_GROUP        = "blocked_by_failed_group"
    BLOCKED_BY_PRESUPPOSITION_GATE = "blocked_by_presupposition_gate"
    BLOCKED_BY_PREFLIGHT_FAIL      = "blocked_by_preflight_fail"
    BLOCKED_BY_SYMBOL_NOT_FOUND    = "blocked_by_symbol_not_found"
    BLOCKED_BY_AUTO_CORRECT_SKIP   = "blocked_by_auto_correct_skip"
    BLOCKED_BY_FAILED_DEPENDENCY   = "blocked_by_failed_dependency"
    ABORTED_AFTER_CRITICAL_FAILURE = "aborted_after_critical_failure"
    UNSCHEDULABLE                  = "unschedulable"
    UNRESOLVED_CYCLE               = "unresolved_cycle"
    BLOCKED_BY_DCR_AUTO_REJECT       = "blocked_by_dcr_auto_reject"
    BLOCKED_BY_DCR_APPROVAL_REQUIRED = "blocked_by_dcr_approval_required"


_SKIP_REASON_BY_VALUE: dict[str, SkipReason] = {sr.value: sr for sr in SkipReason}


# Closed table of skip reasons that mean "this op never got to run because of
# another op's failure or a graph topology problem". Mutating the executor to
_DEPENDENCY_RELATED_SKIP_REASONS: frozenset[SkipReason] = frozenset({
    SkipReason.GROUP_ABORTED,
    SkipReason.BLOCKED_BY_FAILED_GROUP,
    SkipReason.BLOCKED_BY_PRESUPPOSITION_GATE,
    SkipReason.BLOCKED_BY_PREFLIGHT_FAIL,
    SkipReason.BLOCKED_BY_SYMBOL_NOT_FOUND,
    SkipReason.BLOCKED_BY_AUTO_CORRECT_SKIP,
    SkipReason.BLOCKED_BY_FAILED_DEPENDENCY,
    SkipReason.UNSCHEDULABLE,
    SkipReason.UNRESOLVED_CYCLE,
})


def parse_skip_reason(reason: str) -> Optional[SkipReason]:
    """Parse a ``skipped_ops`` reason string into its ``SkipReason`` head.

    Reason format: ``"<head>:<detail>"`` or ``"<head>"``. Returns ``None``
    when the head is unknown — callers fall through to legacy/default
    behaviour rather than crashing on legacy strings.
    """
    if not reason:
        return None
    head = reason.split(":", 1)[0].strip()
    return _SKIP_REASON_BY_VALUE.get(head)


def is_dependency_related_skip(reason: str) -> bool:
    """True when ``reason``'s head is in ``_DEPENDENCY_RELATED_SKIP_REASONS``.

    Replaces the legacy substring-match heuristic
    (``any(kw in reason.lower() for kw in {"blocked", "cycle", "missing",
    "unschedulable", "dependency"})``). The substring path conflated
    ``aborted_after_critical_failure:force_finish`` (single-failure abort,
    not a dependency cascade) with real dependency blocks because they
    happened to share the substring "ed" / "missing" / "blocked"; the
    closed-table lookup makes the categorisation deliberate.
    """
    parsed = parse_skip_reason(reason)
    return parsed is not None and parsed in _DEPENDENCY_RELATED_SKIP_REASONS


# ── final_blocking_reasons sentinel codes ───────────────────────────────────


# Sentinel strings that ``_collect_final_blocking_reasons`` writes into
# ``final_blocking_reasons`` when a dependency cascade is detected.
_DEPENDENCY_BLOCKING_SENTINELS: frozenset[str] = frozenset({
    "dependency_blocked_operations_present",
})


def is_dependency_blocking_sentinel(reason: str) -> bool:
    """True when ``reason`` is a canonical dependency-blocking sentinel.

    Replaces ``any(kw in reason.lower() for kw in {"dependency_blocked",
    "missing_dependency", "blocked_by_failed_dependency"})``. The substring
    path was matching three different conceptual sources (the sentinel
    written by ``_collect_final_blocking_reasons``, raw skip reasons that
    happened to leak into blocking lists, and a "missing_dependency" tag
    that no source actually emits) — consolidating to a single closed set
    of authoritative sentinels keeps the producer side responsible for
    declaring the signal.
    """
    return bool(reason) and reason.strip() in _DEPENDENCY_BLOCKING_SENTINELS


@dataclass
class LadderContext:
    """Executor-internal flags injected by the strategy ladder into an Operation.

    Stored as ``op.ladder_ctx`` rather than polluting ``op.metadata``.
    All fields default to False / None so handlers can safely read without
    ``getattr`` guards.
    """
    skip_python_precise: bool = False   # skip AST path, go to line-range edit
    re_resolved: bool = False           # op path was re-resolved via SymbolSearcher
    context_widen: bool = False         # full-file content was injected into intent


# ── Edit Group: coordinated multi-symbol edits ───────────────────────────────

@dataclass
class EditGroup:
    """A group of operations that must succeed or fail together.

    When multiple operations target related symbols across files (e.g. rename a
    method and update all callers), they can be grouped so the executor treats
    them as a single atomic unit:
      - All ops in the group are executed sequentially.
      - If any op fails, the entire group is rolled back.
      - Cross-op consistency checks are run after the group completes.
    """

    id: str
    operation_ids: list[str]           # ordered list of op IDs in this group
    reason: str = ""                   # why these ops are grouped
    consistency_checks: list[str] = field(default_factory=list)  # post-group checks
    rollback_scope: str = "group"      # "group" = rollback all on failure

    def __post_init__(self) -> None:
        if len(self.operation_ids) < 2:
            raise ValueError("EditGroup requires at least 2 operation IDs")


@dataclass
class ExtractFunctionSpec:
    """Semantic specification for an EXTRACT_FUNCTION operation.

    Captures everything the executor needs to atomically extract a region from
    source_symbol into a new standalone helper function.

    Stored in ``Operation.context_hints["extract_spec"]`` at plan-build time so
    that the executor can drive the two-phase (INSERT helper + MODIFY source)
    write without re-inferring intent from free text.
    """
    source_symbol: str                              # function to extract from
    helper_name: str                                # new function name
    helper_scope: str = "module"                    # "module" | "class" | "nested"
    extract_anchor: Optional[str] = None            # anchor name for INSERT_AFTER_SYMBOL
    extract_region_hint: Optional[str] = None       # textual description of the region (legacy)
    replacement_stmt: Optional[str] = None          # e.g. "x = _helper(arg)" or "self._helper(...)"
    required_inputs: list[str] = field(default_factory=list)   # computed: params helper needs
    return_values: list[str] = field(default_factory=list)     # computed: values helper must return
    region_start_lineno: Optional[int] = None       # resolved region start (1-based)
    region_end_lineno: Optional[int] = None         # resolved region end (1-based, inclusive)
    helper_kind: str = "module_function"            # "method" | "module_function"
    helper_parent: Optional[str] = None             # class name when helper_kind="method"
    call_form: str = ""                             # "self._helper_name" | "_helper_name"
    region_anchor_text: Optional[str] = None        # first distinctive line (legacy, kept for compat)
    region_first_line_text: Optional[str] = None    # actual first line of region (positional) — for Phase B "begins at" hint
    region_anchor_texts: list[str] = field(default_factory=list)  # 2-5 scored anchor lines for fingerprint (text)
    region_anchor_fingerprints: list[str] = field(default_factory=list)  # ast.dump of top region stmts (AST-canonical)
    moved_statement_count: int = 0                  # top-level stmt count in extracted region
    # Exits (return/raise) inside the extracted region that MOVE to the helper.
    # Phase B semantic verifier uses this to assert:
    region_exits: list[dict] = field(default_factory=list)

    # ── Shared scaffold extraction fields ──────────────────────────────────
    # Set only when extraction_mode == "shared_scaffold".
    # All other fields above remain single-source semantics for backward compat.
    extraction_mode: str = "single_source"           # "single_source" | "shared_scaffold"
    source_symbols: list[str] = field(default_factory=list)   # [A, B, ...] for shared mode
    primary_source_symbol: str = ""                  # helper body basis; defaults to source_symbol
    shared_region_hints: dict[str, str] = field(default_factory=dict)  # symbol → region hint (text, legacy fallback)
    # Structural region locator: symbol → (start_body_idx, end_body_idx_exclusive)
    # start_body_idx: fn.body[start_body_idx] is the first top-level stmt of the region.
    region_body_idx_range: dict[str, tuple[int, int]] = field(default_factory=dict)


@dataclass
class EditContract:
    """Structured constraints for an operation — narrows LLM output freedom.

    Attached to Operation.metadata["edit_contract"] or Operation.edit_contract.
    Rendered into the developer LLM prompt to enforce precise edits.

    See `docs/representation_selector.md` for the full authority chain
    (planner → selector → dispatcher → repair → telemetry) and the rules
    for which fields are load-bearing on which control_path.

    Field layers (planner → executor selector → developer).  The class body
    is laid out in the same order so callers reading the dataclass see the
    primary→secondary→legacy progression directly:

      1) Behavioral layer (what the LLM may/must do)
         goal / change_summary / allowed_edits / forbidden_edits /
         success_criteria

      2) Semantic layer (what KIND of change this is)
         semantic_change_family / semantic_family_evidence / rewrite_scope /
         control_flow_preservation_required / signature_stability_required /
         anchor_sensitivity / preserve / must_keep_return_paths /
         must_keep_raise_paths

      3) Representation layer (executor selector authority)
         representation_hint / preferred_first_representation /
         allowed_representations / forbidden_representations /
         fallback_representations

      4) Legacy fields (back-compat only — see comments below)
         preferred_output_mode / forbidden_output_modes / max_changed_lines

    Authority model:
      • The selector reads the SEMANTIC + REPRESENTATION layers as the
        source of truth.  preferred_first_representation pinned at index 0
        of the dispatch order; forbidden_representations is a hard ban.
      • Legacy fields (preferred_output_mode etc.) are NOT promoted to
        gating signals on the selector_native path.  They survive as
        prompt-rendering hints for ops that have no semantic labelling
        (deferred_legacy path) and as one extra signal the selector falls
        back to when nothing else is set.
    """
    # ── 1) Behavioral layer ────────────────────────────────────────────────
    goal: str = ""                                          # one-line goal
    change_summary: list[str] = field(default_factory=list) # what changed upstream
    allowed_edits: list[str] = field(default_factory=list)  # what the LLM may do
    forbidden_edits: list[str] = field(default_factory=list)  # what the LLM must NOT do
    success_criteria: list[str] = field(default_factory=list)  # how to verify success

    # ── 2) Semantic layer (what KIND of change) ──
    # Enum values (not strict): "signature_change" | "local_guard" |
    semantic_change_family: str = ""
    # Short tags explaining WHY semantic_change_family was assigned.
    # Examples: "edit_kind:guard_add", "shared_scaffold_bundle",
    # "paired_helper_insert", "protected_exits_present:3", "has_try".
    # Free-form strings — used for telemetry / debugging mis-labelling.
    semantic_family_evidence: list[str] = field(default_factory=list)
    # Enum values: "token" | "stmt" | "block" | "function_body" |
    # "symbol_multi_region" | "" (unknown).
    rewrite_scope: str = ""
    control_flow_preservation_required: bool = False
    signature_stability_required: bool = False
    # Enum values: "none" | "after_anchor" | "before_return" | "inside_block" | "entry".
    anchor_sensitivity: str = "none"
    # Free-form labels the executor / verifier can react to.  Common values:
    # "signature", "decorators", "protected_exits", "placement_order",
    # "docstring", "comments".
    preserve: list[str] = field(default_factory=list)
    must_keep_return_paths: bool = False
    must_keep_raise_paths: bool = False

    # ── 3) Representation layer (executor selector authority) ──
    # Enum values across the four representation fields:
    representation_hint: str = ""
    preferred_first_representation: str = ""
    allowed_representations: list[str] = field(default_factory=list)
    forbidden_representations: list[str] = field(default_factory=list)
    fallback_representations: list[str] = field(default_factory=list)

    # ── 4) Legacy fields (back-compat only — DO NOT add new uses) ──
    # `preferred_output_mode` predates the representation layer.  Today it
    preferred_output_mode: str = ""                         # "anchor_edit", "surgical_edit", etc.
    forbidden_output_modes: list[str] = field(default_factory=list)
    max_changed_lines: int = 0                              # 0 = no limit

    def render(self) -> str:
        """Render contract as LLM-readable text block."""
        parts = []
        if self.goal:
            parts.append(f"GOAL: {self.goal}")
        if self.change_summary:
            parts.append("CHANGE SUMMARY:\n" + "\n".join(f"  - {c}" for c in self.change_summary))
        if self.allowed_edits:
            parts.append("ALLOWED EDITS:\n" + "\n".join(f"  ✓ {e}" for e in self.allowed_edits))
        if self.forbidden_edits:
            parts.append("FORBIDDEN (do NOT do these):\n" + "\n".join(f"  ✗ {e}" for e in self.forbidden_edits))
        if self.success_criteria:
            parts.append("SUCCESS CRITERIA:\n" + "\n".join(f"  • {c}" for c in self.success_criteria))
        # Semantic layer — surfaced to the developer LLM so it understands
        # WHAT kind of change is being requested (not just the body to write).
        if self.semantic_change_family:
            parts.append(f"CHANGE FAMILY: {self.semantic_change_family}")
        if self.rewrite_scope:
            parts.append(f"REWRITE SCOPE: {self.rewrite_scope}")
        if self.preserve:
            parts.append("PRESERVE:\n" + "\n".join(f"  • {p}" for p in self.preserve))
        if self.control_flow_preservation_required:
            parts.append(
                "CONTROL FLOW: must preserve existing return/raise paths "
                "and branch structure"
            )
        if self.signature_stability_required:
            parts.append("SIGNATURE: must remain unchanged")
        # Representation hint — kept low-key in the prompt; the selector is
        # the source of truth.  We surface it so the LLM produces output in
        # the chosen shape when it's already decided.
        if self.preferred_first_representation:
            parts.append(f"PREFERRED REPRESENTATION: {self.preferred_first_representation}")
        elif self.preferred_output_mode:
            # legacy fallback
            parts.append(f"PREFERRED OUTPUT: {self.preferred_output_mode}")
        if self.forbidden_representations:
            parts.append(
                "FORBIDDEN REPRESENTATIONS:\n"
                + "\n".join(f"  ✗ {r}" for r in self.forbidden_representations)
            )
        if self.max_changed_lines:
            parts.append(f"MAX CHANGED LINES: {self.max_changed_lines}")
        return "\n".join(parts)


class SymbolKind(str, enum.Enum):
    """Structural role of a symbol — used in SymbolRef."""
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    CONSTANT = "constant"   # module-level constant or named variable — use this, not VARIABLE
    IMPORT = "import"
    FILE = "file"


@dataclass
class SymbolRef:
    """First-class symbol reference: name + optional structural context.

    Separates the *what* (name, kind) from the *where* (file, line) and
    the *role* (anchor / produces / scope / targets) it plays in an Operation.
    """
    name: str
    kind: Optional[SymbolKind] = None
    file: Optional[str] = None
    qualified: Optional[str] = None   # e.g. "OperationPlan.write_op_count"
    line: Optional[int] = None


@dataclass
class Operation:
    """A single atomic operation to be executed by the executor.

    Each operation targets a specific symbol, file, or abstract action.

    Semantic fields (placement / effect / scope):
      anchor   — the existing symbol used as a placement reference (INSERT ops)
      produces — symbols created or substantially changed by this op
      scope    — enclosing class / module context
      targets  — existing symbols being modified or deleted (MODIFY/DELETE ops)

    ``symbol`` is kept for backward compatibility; new code should prefer the
    semantic fields above.
    """

    id: str
    kind: OperationKind
    # Target identification
    path: Optional[str] = None          # file path (if applicable)
    symbol: Optional[str] = None        # symbol name (function, class, variable)
    intent: Optional[str] = None        # free-text actionable instruction (developer-facing)
    code_snippet: Optional[str] = None  # optional: exact code block (LLM-emitted); _extract_code_from_intent fallback when absent
    rationale: Optional[str] = None     # planner-only background context; never sent to developer LLM
    # Semantic fields: placement / effect / scope
    anchor: Optional[SymbolRef] = None                       # INSERT: placement reference
    produces: list[SymbolRef] = field(default_factory=list)  # symbols created by this op
    scope: Optional[SymbolRef] = None                        # enclosing class/module context
    targets: list[SymbolRef] = field(default_factory=list)   # MODIFY/DELETE subjects
    # Dependencies and acceptance
    depends_on: list[str] = field(default_factory=list)   # operation IDs that must complete first
    acceptance: list[str] = field(default_factory=list)   # criteria for success (optional)
    # ANCHOR_EDIT specific fields
    anchor_pattern: Optional[str] = None        # Position-specifying pattern (regex, re.search) — fallback only when anchor_ast_lineno absent
    anchor_occurrence: int = -1                 # Which match occurrence (1=first, -1=last)
    edit_mode: str = "insert_before"            # "insert_before"|"insert_after"|"replace_line"
    anchor_ast_lineno: Optional[int] = None     # AST-resolved 1-indexed line; bypasses string search when set
    anchor_context_before: Optional[str] = None # Regex: line before the anchor must match this pattern (optional, disambiguation)
    anchor_context_after: Optional[str] = None  # Regex: line after the anchor must match this pattern (optional, disambiguation)
    # Additional context for the executor
    context_hints: dict[str, Any] = field(default_factory=dict)
    # Optional metadata for the planner
    metadata: dict[str, Any] = field(default_factory=dict)
    # 1st-class bundle fields — static contract for coordinated multi-op bundles
    bundle_id: Optional[str] = None           # which bundle this op belongs to
    bundle_position: Optional[str] = None     # "helper_insert" | "source_rewrite" | "extract_spec"
    # True if this op was verified atomic (≤2 steps) at plan creation time.
    # Set by _atomize_plan() in planner_agent.py — executor must NOT re-decompose.
    atomic: bool = False
    # P6: Edit contract — structured constraints for LLM output
    edit_contract: Optional[EditContract] = None
    # Stage 4: Pre-resolved SymbolDef — attached by executor before execution begins.
    # Holds a SymbolDef instance from SymbolSearcher so handlers can use exact line
    # ranges without re-searching at execution time.  Not serialized; runtime-only.
    resolved_symbol: Optional[Any] = field(default=None, compare=False, repr=False)
    # Intent Assertions: Planner-generated, AST-verifiable post-conditions (Proposal 1)
    intent_assertions: list[IntentAssertion] = field(default_factory=list)
    # Dataflow Cross-Ref: upstream/downstream op linkage for context injection (Proposal 4)
    # upstream_ops: op IDs whose output this op consumes (e.g. callee's return type)
    upstream_ops: list[str] = field(default_factory=list)
    # downstream_ops: op IDs that consume this op's output
    downstream_ops: list[str] = field(default_factory=list)
    # cross_ref: structured context from related ops for Developer LLM injection
    # e.g. {"upstream_return_change": "func_a return type changed from int to List[int]"}
    cross_ref: dict[str, str] = field(default_factory=dict)
    # Step 1 (shadow-mode): LLM-emitted placement contract.  Sits alongside the
    # canonical metadata["placement_contract"] (deterministic builder output) so
    placement: Optional[PlacementContract] = None
    # Multiple selector candidates from the LLM — never collapse to "single
    # answer".  Selector resolution is a search problem, so scoring + ranking
    # operates on this list.  First element typically duplicates `placement`
    # for convenience.
    placement_candidates: list[PlacementContract] = field(default_factory=list)
    # Strategy-ladder execution context — set by _execute_with_strategy_ladder.
    # NOT planner-supplied; executor-internal only.  Handlers read from here
    # instead of polling individual keys in op.metadata.
    ladder_ctx: Optional[LadderContext] = field(default=None, compare=False, repr=False)
    # Authoritative typed guard IR — propagated from IntentResult.guard_spec through
    # spec.metadata["guard_spec"] → op.guard_spec by planner_agent propagation passes.
    guard_spec: Optional[GuardIR] = field(default=None, compare=False, repr=False)
    # Typed action class — planner LLM emits this to signal deletion intent instead of
    # relying on post-hoc keyword scanning of the free-text intent field.
    # "delete" | "modify" | "create" | "read" | None (absent = unknown, keyword fallback)
    action_class: Optional[str] = None
    # P1: Typed parameter change specs — LLM emits these instead of embedding param
    # names in free-text intent. Each entry: {"name": str, "required": bool}.
    # Empty list = not set (falls back to intent text heuristic for backward compat).
    param_assertions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def anchor_name(self) -> Optional[str]:
        """Anchor symbol name for INSERT_AFTER_SYMBOL (placement reference).

        Prefers ``anchor.name`` (structured SymbolRef) over the legacy ``symbol``
        string field.  All new code reading the anchor should use this property.
        """
        return self.anchor.name if self.anchor else self.symbol

    @property
    def produces_names(self) -> "list[str]":
        """Names of all symbols produced (created) by this op."""
        return [r.name for r in self.produces] if self.produces else []

    def __post_init__(self) -> None:
        """Validate required fields — delegate to validate() as single source of truth."""
        err = self.validate()
        if err is not None:
            raise ValueError(err)

    def validate(self) -> Optional[str]:
        """Single source of truth for Operation field validation.

        Returns None if valid, or an error string if invalid.
        Used by:
          - __post_init__ (converts to ValueError)
          - auto_correction.validate_operation_preflight
          - handler fail-safe guards

        Any new required-field check must be added HERE exactly once.
        """
        # ── Path checks ──────────────────────────────────────────────
        _PATH_EXEMPT = {
            OperationKind.SUMMARIZE_ANALYSIS,
            OperationKind.RUN_SCANNER,
        }
        if self.kind not in _PATH_EXEMPT:
            if not self.path:
                return f"{self.kind.value}: missing path"
        if self.kind == OperationKind.RUN_SCANNER:
            if not self.path and not (self.metadata or {}).get("paths"):
                return "run_scanner: missing path or metadata.paths"

        # ── Symbol checks ────────────────────────────────────────────
        # Superset of all symbol-requiring kinds:
        #   __post_init__ set:  READ_SYMBOL, MODIFY_SYMBOL, INSERT_AFTER_SYMBOL, UPDATE_CALLERS
        #   auto_correction set: MODIFY_SYMBOL, INSERT_AFTER_SYMBOL, READ_SYMBOL, MOVE_SYMBOL
        # Union = READ_SYMBOL, MODIFY_SYMBOL, INSERT_AFTER_SYMBOL, UPDATE_CALLERS, MOVE_SYMBOL
        _SYMBOL_OPS = {
            OperationKind.MODIFY_SYMBOL,
            OperationKind.INSERT_AFTER_SYMBOL,
            OperationKind.READ_SYMBOL,
            OperationKind.MOVE_SYMBOL,
            OperationKind.UPDATE_CALLERS,
        }
        if self.kind in _SYMBOL_OPS and not self.symbol:
            # Allow INSERT_AFTER_SYMBOL with symbol=None when P8-S2 EOF
            # fallback cleared the symbol intentionally.
            if self.kind == OperationKind.INSERT_AFTER_SYMBOL and (
                self.metadata or {}
            ).get("_symbol_hallucinated_eof_fallback"):
                pass
            else:
                return f"{self.kind.value}: missing symbol"

        # ── ANCHOR_EDIT checks ───────────────────────────────────────
        if self.kind == OperationKind.ANCHOR_EDIT:
            if not self.path:
                return "anchor_edit requires path"
            if not self.anchor_pattern and self.anchor_ast_lineno is None:
                return "anchor_edit requires anchor_pattern or anchor_ast_lineno"
            if self.edit_mode not in ("insert_before", "insert_after", "replace_line", "replace_block", "delete"):
                return f"anchor_edit: invalid edit_mode={self.edit_mode!r}"

        # ── INSERT_AFTER_LINE checks ─────────────────────────────────
        if self.kind == OperationKind.INSERT_AFTER_LINE:
            if not self.path:
                return "insert_after_line requires path"
            if not self.anchor_pattern and self.anchor_ast_lineno is None:
                return "insert_after_line requires anchor_pattern or anchor_ast_lineno"

        # ── READ_FILE_SEGMENT checks ─────────────────────────────────
        if self.kind == OperationKind.READ_FILE_SEGMENT:
            if not self.path:
                return "read_file_segment requires path"

        # ── RUN_SCANNER checks ───────────────────────────────────────
        if self.kind == OperationKind.RUN_SCANNER:
            if not self.metadata.get("scanner_name"):
                return "run_scanner requires metadata['scanner_name']"

        return None


@dataclass
class GroundingSummary:
    """Typed grounding evidence attached to an OperationPlan.

    Single source of truth for all grounding confidence data.
    Replaces the ad-hoc dict that was constructed in agent_loop and read
    with inconsistent defaults across planner_agent, spec_resolver, and
    operation_executor.
    """

    grounding_confidence: float = 0.0
    grounding_best_score: float = 0.0
    grounding_candidate_count: int = 0
    grounding_top_candidates: list[dict[str, Any]] = field(default_factory=list)
    intent_files: list[str] = field(default_factory=list)
    intent_symbols: list[str] = field(default_factory=list)
    exploration_confidence: float = 0.0
    exploration_mode: str = ""
    source: str = ""
    scope_mode: str = ""
    authoritative: bool = True
    target_provenance: str = "explicit"
    confidence_grade: str = ""

    @classmethod
    def from_spec_meta(
        cls,
        meta: dict[str, Any],
        spec: Any = None,
        top_candidates: Optional[list[dict[str, Any]]] = None,
    ) -> "GroundingSummary":
        """Build from spec.metadata + optional spec attributes."""
        return cls(
            grounding_confidence=round(float(meta.get("grounding_confidence") or 0.0), 3),
            grounding_best_score=round(float(meta.get("grounding_best_score") or 0.0), 3),
            grounding_candidate_count=int(meta.get("grounding_candidate_count") or 0),
            grounding_top_candidates=top_candidates or [],
            intent_files=list(getattr(spec, "intent_files", []) or []) if spec is not None else [],
            intent_symbols=list(getattr(spec, "intent_symbols", []) or []) if spec is not None else [],
            exploration_confidence=round(float(meta.get("exploration_confidence") or 0.0), 3),
            exploration_mode=meta.get("exploration_mode", ""),
            source=meta.get("spec_resolver_source", ""),
            scope_mode=getattr(spec, "scope_mode", "") if spec is not None else "",
            authoritative=getattr(spec, "authoritative", True) if spec is not None else True,
            target_provenance=getattr(spec, "target_provenance", "explicit") if spec is not None else "explicit",
            confidence_grade=meta.get("grounding_confidence_grade", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict for prompt serialization."""
        return {
            "grounding_confidence": self.grounding_confidence,
            "grounding_best_score": self.grounding_best_score,
            "grounding_candidate_count": self.grounding_candidate_count,
            "grounding_top_candidates": self.grounding_top_candidates,
            "intent_files": self.intent_files,
            "intent_symbols": self.intent_symbols,
            "exploration_confidence": self.exploration_confidence,
            "exploration_mode": self.exploration_mode,
            "source": self.source,
            "scope_mode": self.scope_mode,
            "authoritative": self.authoritative,
            "target_provenance": self.target_provenance,
        }


_PLAN_POLICY_VALID_KINDS = frozenset(
    {"analysis_only", "analyze_then_modify", "direct_modify", "create_file", "deterministic", "scaffold", "edit"}
)


@dataclass
class PlanPolicy:
    """Planner's self-declared execution strategy for a plan.

    Replaces the ad-hoc plan.metadata["plan_policy"] dict that was read with
    inconsistent defaults across planner_agent, agent_loop, and operation_executor.
    """

    kind: str  # "analysis_only" | "analyze_then_modify" | "direct_modify" | "create_file"
    requires_code_changes: bool = True
    confidence: float = 0.5
    why: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "requires_code_changes": self.requires_code_changes,
            "confidence": self.confidence,
            "why": self.why,
        }



# ---------------------------------------------------------------------------
# StageContext — typed pipeline stage boundary (replaces metadata dict)
# ---------------------------------------------------------------------------


@dataclass
class StageContext:
    """Typed pipeline stage boundary context.

    Replaces the ad-hoc plan.metadata["key"] dict communication
    between pipeline stages with typed, documented fields.

    Design rationale
    ----------------
    The pipeline has ~260 unique metadata keys (577 use sites) flowing through
    Dict[str, Any].  This caused hallucination cascades (N1-N8) because:
      - Every stage trusts the previous stage's dict output unconditionally
      - Typos in keys become silent None propagation, not compile-time errors
      - No schema means no validation at stage boundaries

    StageContext solves this by providing typed fields for each stage boundary.
    Fields are grouped by the pipeline stage that WRITES them:

        Stage 0 (Spec Resolution) -> graph_context, evaluator_verdicts, ...
        Stage 1 (Graph Enrich)   -> graph_impact, graph_safety_issues, ...
        Stage 2 (Plan Creation)  -> strategy, plan_source, contract_applied, ...
        Stage 3 (Task Drift)     -> task_drift, task_drift_block
        Stage 4 (Pre-exec Gate)  -> pre_exec_replan_done, execution_strategy
        Stage 5 (Execution)      -> exec_info, f821_*, acceptance_*, ...
        Stage 6 (Verification)   -> termination_decision, alignment_*, ...
        Stage 7 (Result)         -> tokens_used, failure_classification, ...

    Usage (progressive migration):
        # Pipeline creates ctx at entry, populates from plan.metadata
        ctx = StageContext()
        ctx.graph_context = plan.metadata.get("graph_context")

        # Readers switch to ctx.graph_context (typed!)
        if ctx.task_drift_block:
            ...

        # Export back to plan.metadata at stage exit
        # (only needed for fields that downstream expects in metadata)
        plan.metadata["task_drift_block"] = ctx.task_drift_block
    """

    # -- Stage 0: Spec Resolution ------------------------------------------
    # Written by: spec_resolver.py, planner_agent.py
    # Read by:    planner_agent.py, operation_executor.py, verification_passes.py

    graph_context: Optional[dict[str, Any]] = None
    """Graph-enriched symbol context (SpecGraphEnricher output).
    Replaces spec.metadata["graph_context"]."""

    graph_impact: Optional[dict[str, Any]] = None
    """Impact analysis result -- spec.metadata["graph_impact"]."""

    graph_safety_issues: list[str] = field(default_factory=list)
    """GSG safety violations -- plan.metadata["graph_safety_issues"]."""

    graph_verification_scope: Optional[dict[str, Any]] = None
    """Verification scope -- plan.metadata["graph_verification_scope"]."""

    evaluator_verdicts: Optional[dict[str, Any]] = None
    """Semantic judge verdicts -- spec.metadata["evaluator_verdicts"]."""

    helper_contracts: Optional[dict[str, Any]] = None
    """Helper function contracts -- plan.metadata["helper_contracts"]."""

    helper_intents: Optional[list[dict[str, Any]]] = None
    """Helper intents from spec -- spec.metadata["helper_intents"]."""

    grounding_summary: Optional[dict[str, Any]] = None
    """Grounding confidence data -- plan.metadata["grounding_summary"].
    Prefer the typed GroundingSummary class on OperationPlan when available."""

    # -- Stage 1: Graph Enrich / Strategy Selection ------------------------
    # Written by: planner_agent.py, strategy_router.py
    # Read by:    operation_executor.py, planner_agent.py

    strategy: str = ""
    """Selected CandidateStrategy -- plan.metadata["strategy"]."""

    strategy_result: Optional[dict[str, Any]] = None
    """Strategy router result -- plan.metadata["strategy_result"]."""

    execution_strategy: str = "direct"
    """Execution approach -- plan.metadata["execution_strategy"].
    Values: "already_satisfied", "direct", "replan_required"."""

    plan_source: str = ""
    """Origin of the plan -- plan.metadata["plan_source"].
    Values: "structural_seed", "llm_read_then_plan", "post_read_replan"."""

    contract_applied: bool = False
    """Whether contract-driven planning was applied -- plan.metadata["contract_applied"]."""

    change_spec: Optional[dict[str, Any]] = None
    """Change specification for coverage -- plan.metadata["change_spec"]."""

    # -- Stage 2: Task Drift -----------------------------------------------
    # Written by: operation_executor.py (pre-execution gate)
    # Read by:    agent_planner_pipeline.py

    task_drift: Optional[dict[str, Any]] = None
    """Task drift report -- plan.metadata["_task_drift"]."""

    task_drift_block: bool = False
    """True = execution blocked by task drift -- plan.metadata["_task_drift_block"]."""

    # -- Stage 3: Pre-execution Replan -------------------------------------
    # Written by: operation_executor.py
    # Read by:    agent_planner_pipeline.py

    pre_exec_replan_done: bool = False
    """Whether pre-execution replan was performed -- plan.metadata["pre_exec_replan_done"]."""

    pre_exec_replan_reason: str = ""
    """Why pre-execution replan was triggered -- plan.metadata["pre_exec_replan_reason"]."""

    pre_execution_strategy_selection: Optional[dict[str, Any]] = None
    """Strategy selection from pre-execution gate -- plan.metadata["pre_execution_strategy_selection"]."""

    # -- Stage 4: Operation Execution --------------------------------------
    # Written by: operation_executor.py
    # Read by:    agent_planner_pipeline.py, verification_passes.py

    exec_info: Optional[dict[str, Any]] = None
    """Execution info summary -- plan.metadata["exec_info"]."""

    phase_a_exception: Optional[str] = None
    """Phase A exception message -- plan.metadata["phase_a_exception"]."""

    # F821 (undefined name) tracking
    f821_undefined_names: list[str] = field(default_factory=list)
    """Undefined names detected -- plan.metadata["f821_undefined_names"]."""

    f821_covered_by_remaining: bool = False
    """F821 covered by remaining ops -- plan.metadata["f821_covered_by_remaining"]."""

    f821_stubs_generated: int = 0
    """Number of stubs generated for F821 -- plan.metadata["f821_stubs_generated"]."""

    has_f821_errors: bool = False
    """Whether F821 errors exist -- plan.metadata["has_f821_errors"]."""

    # Import cycles
    import_cycles: list[str] = field(default_factory=list)
    """Import cycles detected -- plan.metadata["import_cycles"]."""

    has_import_cycles: bool = False
    """Whether import cycles exist -- plan.metadata["has_import_cycles"]."""

    # Cross-file issues
    cross_file_issues: list[str] = field(default_factory=list)
    """Cross-file issues -- plan.metadata["cross_file_issues"]."""

    has_cross_file_issues: bool = False
    """Whether cross-file issues exist -- plan.metadata["has_cross_file_issues"]."""

    # Acceptance tracking
    acceptance_met: Optional[bool] = None
    """Whether acceptance criteria met -- plan.metadata["acceptance_met"]."""

    acceptance_unmet: list[str] = field(default_factory=list)
    """Unmet acceptance criteria -- plan.metadata["acceptance_unmet"]."""

    deferred_nl_acceptance: Optional[dict[str, Any]] = None
    """Deferred natural-language acceptance -- plan.metadata["deferred_nl_acceptance"]."""

    # Rollback state
    _churn_rollback: bool = False
    """Whether churn rollback occurred -- plan.metadata["_churn_rollback"]."""

    _reference_orphans: list[str] = field(default_factory=list)
    """Reference orphans after rollback -- plan.metadata["_reference_orphans"]."""

    # -- Stage 5: Verification / Final Decision ----------------------------
    # Written by: operation_executor.py, verification_passes.py
    # Read by:    agent_planner_pipeline.py

    termination_decision: Optional[dict[str, Any]] = None
    """Termination decision -- plan.metadata["termination_decision"]."""

    alignment_score: float = 0.0
    """Semantic alignment score -- plan.metadata["alignment_score"]."""

    alignment_breakdown: Optional[dict[str, Any]] = None
    """Alignment breakdown -- plan.metadata["alignment_breakdown"]."""

    # -- Stage 6: Result / Performance -------------------------------------
    # Written by: agent_turn_pipeline.py, agent_loop.py
    # Read by:    asi.py

    tokens_used: int = 0
    """Total tokens consumed -- result.metadata["tokens"]."""

    performance: Optional[dict[str, Any]] = None
    """Performance summary -- result.metadata["performance"]."""

    failure_classification: Optional[dict[str, Any]] = None
    """Failure classification -- result.metadata["failure_classification"]."""

    # -- Fallback: keys not yet migrated to typed fields -------------------
    extra: dict[str, Any] = field(default_factory=dict)
    """Metadata keys not yet assigned to typed fields.

    During migration, all plan.metadata keys that don't yet have a dedicated
    field land here.  Once a field is added above, readers switch from
    ctx.extra["key"] to ctx.field_name, and writers populate both.
    Eventually extra is removed.
    """
@dataclass
class OperationPlan:
    """A complete plan consisting of a sequence of operations.

    Produced by the Planner, consumed by the Executor.
    """

    operations: list[Operation]
    mode: PlanMode = PlanMode.EDIT
    # Overall description of the plan (optional)
    description: str = ""
    # Plan-level acceptance criteria (optional)
    acceptance: list[str] = field(default_factory=list)
    # Metadata for the planner (e.g., original request, analysis)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Edit groups: coordinated multi-symbol operations
    edit_groups: list[EditGroup] = field(default_factory=list)
    # Plan-level intent assertions (checked after entire plan execution)
    plan_assertions: list[IntentAssertion] = field(default_factory=list)
    # 1st-class bundle fields — static contract for coordinated multi-op bundles
    bundle_id: Optional[str] = None                        # unique bundle identifier
    bundle_kind: Optional[str] = None                      # "shared_scaffold_extract" (extensible)
    bundle_required_roles: list[str] = field(default_factory=list)    # role names that must be fulfilled
    bundle_source_symbols: list[str] = field(default_factory=list)    # source symbols [A, B, ...]
    bundle_helper_name: Optional[str] = None               # resolved helper function name
    # 1st-class phase fields — multi-phase execution state (replaces scattered metadata flags)
    execution_phase: ExecutionPhase = ExecutionPhase.SINGLE_PHASE
    requires_phase2: bool = False                          # True when Phase 2 replan is mandatory
    deferred_write_mode: Optional[DeferredWriteMode] = None  # why writes were deferred
    phase_parent_plan_id: Optional[str] = None            # Phase 2 → which Phase 1 plan spawned it
    phase_contract: dict[str, Any] = field(default_factory=dict)  # phase-scoped structured state
    # Symbol expectation contract — Phase 0 guess vs Phase 1 confirmed
    # tentative: Phase 0 IntentResult guess; NOT an execution contract
    # confirmed: Phase 1 FixSpec analysis result; authoritative for Phase 2 planning
    tentative_new_symbols: list[str] = field(default_factory=list)
    confirmed_new_symbols: list[str] = field(default_factory=list)
    confirmed_insert_targets: list[dict[str, Any]] = field(default_factory=list)
    symbol_expectation_source: str = ""  # "intent_result" | "spec_resolver" | "fix_spec_phase1"
    # Typed grounding evidence — replaces scattered plan.metadata["grounding_confidence"] reads
    grounding_summary: Optional["GroundingSummary"] = None
    # Planner's self-declared execution strategy — replaces plan.metadata["plan_policy"] dict reads
    plan_policy: Optional["PlanPolicy"] = None
    # Resolved execution spec — replaces plan.metadata["execution_spec"] dict reads
    # Type annotation uses string to avoid circular import with execution_spec.py
    execution_spec: Optional[Any] = None  # ResolvedExecutionSpec

    def validate(self) -> list[str]:
        """Check for logical errors and return list of warnings."""
        warnings = []
        # Check for duplicate IDs
        ids = [op.id for op in self.operations]
        if len(ids) != len(set(ids)):
            warnings.append("Duplicate operation IDs found")
        # Check dependency references
        all_ids = set(ids)
        for op in self.operations:
            for dep in op.depends_on:
                if dep not in all_ids:
                    warnings.append(f"Operation {op.id} depends on unknown operation {dep}")
        # Validate edit groups
        for eg in self.edit_groups:
            for op_id in eg.operation_ids:
                if op_id not in all_ids:
                    warnings.append(f"EditGroup {eg.id} references unknown operation {op_id}")
        return warnings

    def get_group_for_op(self, op_id: str) -> Optional["EditGroup"]:
        """Return the EditGroup containing this operation, or None."""
        for eg in self.edit_groups:
            if op_id in eg.operation_ids:
                return eg
        return None


def normalize_op_semantic_fields(op: "Operation") -> None:
    """Populate anchor/produces/scope from context_hints when missing.

    Backward-compat bridge: ops generated by the LLM planner or produced via
    MODIFY→INSERT downgrade carry ``context_hints["new_symbol_name"]`` /
    ``context_hints["parent_class"]`` but not the structured SymbolRef fields.
    Call this immediately after any op is created outside of DPB so that all
    downstream code can rely solely on ``anchor``/``produces``/``scope``.
    """
    # Extended to support all ADDITIVE kinds + MODIFY_SYMBOL
    if op.kind not in _ADDITIVE_OP_KINDS and op.kind != OperationKind.MODIFY_SYMBOL:
        return
    if (op.kind == OperationKind.INSERT_AFTER_SYMBOL
            and op.produces and op.anchor):
        return  # already populated (DPB or prior normalization)

    hints = op.context_hints or {}
    parent_cls = hints.get("parent_class")
    _singular: Optional[str] = hints.get("new_symbol_name")
    _plural: list = hints.get("new_symbol_names") or []

    if op.symbol:
        op.anchor = SymbolRef(name=op.symbol, file=op.path)

    # Collect all produced names, deduplicating while preserving order.
    _all_names: list = []
    if _singular and _singular not in _all_names:
        _all_names.append(_singular)
    for _n in _plural:
        if _n and _n not in _all_names:
            _all_names.append(_n)

    if _all_names:
        _kind = SymbolKind.METHOD if parent_cls else SymbolKind.FUNCTION
        op.produces = [
            SymbolRef(
                name=_n,
                kind=_kind,
                file=op.path,
                qualified=f"{parent_cls}.{_n}" if parent_cls else _n,
            )
            for _n in _all_names
        ]
    if parent_cls:
        op.scope = SymbolRef(name=parent_cls, kind=SymbolKind.CLASS, file=op.path)


@dataclass
class ExecutorState:
    """Mutable state maintained across operation execution."""

    # Completed operations (ID -> result)
    completed_ops: dict[str, Any] = field(default_factory=dict)
    # Failed operations (ID -> error)
    failed_ops: dict[str, str] = field(default_factory=dict)
    # Currently executing operation ID
    current_op: Optional[str] = None
    # Symbols already visited (to avoid repeated analysis)
    visited_symbols: dict[str, list[str]] = field(default_factory=dict)  # symbol -> file list
    # Files already read (to cache content)
    read_files: dict[str, str] = field(default_factory=dict)  # path -> content (may be truncated for LLM)
    # Full-file content cache — separate from read_files (which may hold truncated/LLM-formatted content).
    # Populated by add_read_file() with the full content passed by callers.
    # get_read_file_for_ast() reads from this dict to avoid AST operations on truncated content.
    read_files_full: dict[str, str] = field(default_factory=dict)  # path -> full content
    # History of tool calls made during execution
    tool_history: list[dict[str, Any]] = field(default_factory=list)
    # If true, executor should finish after current operation
    force_finish: bool = False
    # Additional context accumulated during execution
    accumulated_context: dict[str, Any] = field(default_factory=dict)
    # Plan mode (edit vs read-only analysis)
    plan_mode: Optional[PlanMode] = None
    # Typed slots promoted from accumulated_context — avoids per-handler .get() with inconsistent defaults
    grounding_confidence: Optional[float] = None   # set by executor from plan.grounding_summary
    plan_policy: Optional["PlanPolicy"] = None     # set by executor bridge from plan.plan_policy
    request_type: str = ""                         # from plan.execution_spec.request_type
    scope_mode: str = ""                           # from plan.execution_spec.scope_mode
    workset_scope_files: list[str] = field(default_factory=list)  # from execution_spec metadata structural_worksets
    # Execution progress flags — replaces accumulated_context bool sentinels
    stub_repair_ran: bool = False
    scope_mismatch_repair_ran: bool = False
    consistency_solver_ran: bool = False
    cross_file_repair_attempted: bool = False
    rename_propagated: bool = False
    # FixSpec-adjacent scalar slots
    fix_spec_tests_run: bool = False
    fix_spec_tests_failed: str = ""
    fix_spec_constraint_mode: str = ""
    fix_spec_verified_targets: list[dict[str, Any]] = field(default_factory=list)
    fix_spec_suppressed_targets: list[dict[str, Any]] = field(default_factory=list)
    # Presupposition slots
    presupposition_invalidated: bool = False
    presupposition_clarification: str = ""
    # Semantic check counter
    semantic_check_count: int = 0
    # Scheduler internals bridged from build/plan
    non_blocking_edges: set[tuple[str, str]] = field(default_factory=set)
    escalation_constraints: Optional[dict[str, Any]] = None
    # Execution data collections
    change_events: list[dict[str, Any]] = field(default_factory=list)
    no_diff_hard_failures: list[str] = field(default_factory=list)
    auto_corrections: list[dict[str, Any]] = field(default_factory=list)
    symbols_modified_so_far: set[str] = field(default_factory=set)
    # Misc single-use slots
    stub_repair_active: bool = False    # live flag: stub repair currently running
    rename_instructions_count: int = 0
    modified_files_set: set[str] = field(default_factory=set)
    already_satisfied_ops: list[str] = field(default_factory=list)
    # Typed LLM-output structures — replaces accumulated_context dict round-trips
    presupposition_check: Optional[Any] = None  # PresuppositionCheckResult
    fix_spec: Optional[dict[str, Any]] = None
    # causal_bundle_v1: from execution_spec.metadata, passed through for summarize_analysis
    causal_bundle_v1: Optional[dict[str, Any]] = None
    # no_effect_symbols: per-(path:symbol) no-diff retry tracking
    no_effect_symbols: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Symbols deleted by prior ops (set by DELETE_SYMBOL_RANGE, checked by read_symbol skip)
    deleted_symbols: set[str] = field(default_factory=set)
    # blocking_reasons: hallucination/wiring failures (force_finish triggers)
    blocking_reasons: list[str] = field(default_factory=list)
    # Skipped operations (ID -> skip reason)
    skipped_ops: dict[str, str] = field(default_factory=dict)
    # Execution order of actually executed operations (completed/failed)
    execution_order: list[str] = field(default_factory=list)
    # Reactive Planning Loop: checkpoints recorded after each operation
    checkpoint_log: list["ExecutionCheckpoint"] = field(default_factory=list)
    replan_count: int = 0
    # replan_history: trace of each replan's op set for circuit-breaker detection
    # Each entry: {symbols, op_kinds, op_count, read_only, summarize_count, file_count}
    replan_history: list[dict[str, Any]] = field(default_factory=list)
    delegation_count: int = 0
    # Files whose read_files entry came from code_context (Design Chat snippets).
    # Used by contract extraction to identify primary contract source.
    code_context_files: set[str] = field(default_factory=set)
    # Strict reference files: holds read_file_segment numbered output and auto-injected type-def refs.
    # get_read_file_for_ast() skips these (returns None) so AST consumers read from disk.
    # NOTE: spec.reference_files content is NO LONGER preloaded here (use code_context instead).
    strict_reference_files: dict[str, str] = field(default_factory=dict)
    # EditGroup tracking: file snapshots taken before group execution for rollback
    group_snapshots: dict[str, dict[str, str]] = field(default_factory=dict)  # group_id -> {path: content}
    # Pre-edit snapshots: file content before any write in this execution.
    # Populated on first read; NOT updated on post-write add_read_file calls.
    # Used by repair_runtime for deterministic AST-diff-based inserted_symbols detection.
    pre_edit_contents: dict[str, str] = field(default_factory=dict)
    # Symbol Lineage (P1.3): tracks post-edit signatures of successfully modified symbols.
    # Populated by _handle_modify_symbol after each successful write.
    # Used by _collect_cross_file_lineage (P1.4) to propagate context across files.
    # Format: symbol_name -> {"signature": str, "file": str, "updated_by": str}
    symbol_lineage: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Execution log: per-op determinism tracking for replay and failure analysis.
    # Each entry: {op_id, file, symbol, pre_hash, post_hash, status, strategy, duration_ms}
    execution_log: list[dict[str, Any]] = field(default_factory=list)
    # Per-op file snapshots taken BEFORE each op executes, keyed by op_id.
    # Used by quality gate to revert critical failures without losing good changes
    # from earlier ops on the same file.
    # Format: {op_id: {abs_file_path: file_content}}
    pre_op_file_snapshots: dict[str, dict[str, str]] = field(default_factory=dict)
    # Files that were written to disk during a failed op AND whose restore also failed.
    # POST_REPAIR_PROMO skips promotion for these files — the file is dirty (neither the
    # original nor the intended edit; restore left the intermediate content on disk).
    restore_failed_files: set[str] = field(default_factory=set)
    # Strategy-ladder supersession map: original op ID → list of replacement op IDs.
    # Populated when _execute_with_strategy_ladder fires a REPLAN action.
    # Used by canonical replan to avoid re-scheduling already-superseded ops.
    ladder_replaced: dict[str, list[str]] = field(default_factory=dict)
    # Files injected as type-definition references (Enum/dataclass/TypedDict sources).
    # Populated by _execute_operations alongside read_files; used by prior_read_files
    # mode "type_defs_only" to include only structural type references, not raw ref files.
    type_def_ref_files: set[str] = field(default_factory=set)
    # (path, symbol) pairs that already produced silent_zero_edit_success.
    # On the first occurrence the op is downgraded to error (surfaces the short-circuit).
    # On repeated occurrences the op is treated as already_satisfied so the
    # strategy-ladder / staged-retry loop does not keep replanning with the same plan.
    silent_zero_symbols: set[tuple] = field(default_factory=set)
    replan_attempts: int = 0
    ladder_replan_count: int = 0
    code_context_analysis: str = ""
    # Raw Design Chat code_context items (list of {reason, file, snippet}).
    # Stored separately from the concatenated code_context_analysis string so
    # _extract_prior_analysis can filter by op relevance.
    code_context_items: list[dict[str, str]] = field(default_factory=list)
    retry_pool: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Per-file accumulated line delta: tracks how many lines each file has gained
    # (positive) or lost (negative) due to prior ops in this execution. Used by
    # delete_symbol_range to adjust stale line_range when prior ops modified the same file.
    # Signed convention: positive = lines added, negative = lines removed.
    # Handlers MUST return line_delta (signed) OR lines_changed with correct sign.
    # Format: file_path -> total_delta (accumulated line_delta from prior ops)
    file_line_deltas: dict[str, int] = field(default_factory=dict)

    def mark_completed(self, op_id: str, result: Any) -> None:
        self.completed_ops[op_id] = result
        self.current_op = None
        self.execution_order.append(op_id)

    def mark_failed(self, op_id: str, error: str) -> None:
        self.failed_ops[op_id] = error
        self.current_op = None
        self.execution_order.append(op_id)

    def add_visited_symbol(self, symbol: str, file_path: str) -> None:
        self.visited_symbols.setdefault(symbol, []).append(file_path)

    def mark_skipped(self, op_id: str, reason: str) -> None:
        """Mark an operation as skipped with a descriptive reason."""
        self.skipped_ops[op_id] = reason

    def has_visited_symbol(self, symbol: str, file_path: Optional[str] = None) -> bool:
        if symbol not in self.visited_symbols:
            return False
        if file_path is None:
            return True
        return file_path in self.visited_symbols[symbol]

    def add_read_file(self, path: str, content: str) -> None:
        # pre_edit_contents: save only on first read (before any write overwrites it).
        # Subsequent calls (post-write updates) only update read_files.
        if path not in self.pre_edit_contents:
            self.pre_edit_contents[path] = content
        self.read_files[path] = content
        # Also store full content in read_files_full for AST-safe retrieval.
        # NOTE: direct assignments to state.read_files (e.g. preload paths in
        # operation_executor) bypass this method and store only truncated content
        # in read_files — they intentionally do NOT pollute read_files_full.
        self.read_files_full[path] = content

    def get_read_file(self, path: str) -> Optional[str]:
        return self.read_files.get(path)

    def get_read_file_for_ast(self, path: str) -> Optional[str]:
        # Read from read_files_full first (full content, AST-safe).
        # Fall back to read_files for backward compat with paths that were
        # only ever stored via direct assignment (truncated preload content).
        if path in self.strict_reference_files or path in self.code_context_files:
            # code_context files contain truncated/pre-analyzed snippets, not
            # full file content — force disk read so AST verifiers get real code.
            return None
        _full = self.read_files_full.get(path)
        if _full is not None:
            return _full
        return self.read_files.get(path)


# ── Reactive Planning Loop (Phase 1) ─────────────────────────────────────────

class PlanValidityScore(str, enum.Enum):
    """4-level validity score for post-operation checkpoint evaluation.

    Phase 1: only GREEN and RED trigger actions.
    YELLOW/ORANGE are logged for observation (future Phase 2 triggers).
    """
    GREEN = "green"      # Continue normally
    YELLOW = "yellow"    # Continue but risk signal detected (log only)
    ORANGE = "orange"    # Replan recommended (log only in Phase 1)
    RED = "red"          # Anchor lost → limited replan


@dataclass
class ExecutionCheckpoint:
    """Post-operation checkpoint capturing plan validity state.

    Created after each operation execution to detect plan drift.
    Rule-based evaluation — no LLM calls.
    """
    operation_id: str
    operation_kind: OperationKind
    result_status: str                          # "success", "error", "not_found"
    # Anchor validation
    target_symbol_exists: bool = True           # post-execution: does the target still exist?
    compile_ok: bool = True                     # syntax/compile check passed?
    remaining_anchors_valid: bool = True        # remaining ops' target symbols still exist?
    # Scoring
    validity_score: PlanValidityScore = PlanValidityScore.GREEN
    invalidated_op_ids: list[str] = field(default_factory=list)  # remaining ops with lost anchors
    # Observation
    failure_classification: Optional[str] = None   # from FailureClassifier (if failed)
    downstream_impacted_ops: list[str] = field(default_factory=list)  # ops referencing changed files/symbols
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Scoped Delegation (Phase 3) ───────────────────────────────────────────────

@dataclass
class ScopedDelegation:
    """Scoped work scope definition to delegate to MAIN_AGENT.

    Used to delegate tasks that cannot be resolved via Operation in the
    PLANNER lane to a scope-limited AgentLoop.
    """
    goal: str                                       # Single goal (intent of the failed op)
    files: list[str] = field(default_factory=list)  # Writable files
    readonly_files: list[str] = field(default_factory=list)  # Read-only files (files modified by completed ops)
    context: str = ""                               # Execution result summary
    max_turns: int = 5                              # Turn limit
    completed_ops_summary: list[dict[str, Any]] = field(default_factory=list)
    failed_op_id: str = ""                          # Failed operation ID
    failed_op_kind: str = ""                        # Failed operation kind
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DelegationResult:
    """Delegation execution result — normalized to operation form."""
    success: bool
    modified_files: list[str] = field(default_factory=list)
    created_files: list[str] = field(default_factory=list)
    deleted_files: list[str] = field(default_factory=list)
    touched_symbols: list[str] = field(default_factory=list)
    summary: str = ""
    remaining_goal_resolved: bool = False
    turns_used: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Edit Instructions ───────────────────────────────────────────────────────

class EditInstructionKind(str, enum.Enum):
    """Types of structured edit instructions."""
    REPLACE_SYMBOL_BODY = "replace_symbol_body"
    INSERT_AFTER_SYMBOL = "insert_after_symbol"
    REPLACE_FILE = "replace_file"
    CREATE_FILE = "create_file"
    OVERWRITE_FILE = "overwrite_file"
    DELETE_FILE = "delete_file"
    PATCH_SYMBOL = "patch_symbol"  # surgical line-range edits within a symbol
    SURGICAL_EDIT = "surgical_edit"  # search/replace block edits (line-number independent)
    INSERT_AFTER_LINE = "insert_after_line"  # text-based line anchor insert (HTML/CSS/JSON)
    AST_DIRECT_BODY = "ast_direct_body"  # direct AST-based body replacement (no re-generation)
    AST_OP = "ast_op"               # typed AST ops: replace_expr/add_import/add_guard/delete_stmt
    # Future extensions
    RENAME_SYMBOL = "rename_symbol"
    MOVE_SYMBOL = "move_symbol"


@dataclass
class EditInstruction:
    """Structured edit instruction for precise code modifications.

    Provides a machine-readable specification of a code change that can be
    deterministically applied without requiring LLM interpretation.
    """
    kind: EditInstructionKind
    # Target identification
    file_path: str
    symbol: Optional[str] = None  # for symbol-level edits
    # Edit data (kind-specific)
    data: dict[str, Any] = field(default_factory=dict)
    # Metadata for debugging and validation
    metadata: dict[str, Any] = field(default_factory=dict)
    # Free-text instruction surfaced to the Developer LLM. Mirrors the
    # ``intent`` field on ``Operation`` from which this instruction is
    intent: Optional[str] = None
    # Strategy-ladder context — copied from Operation.ladder_ctx during execution.
    # Handlers read from here instead of polling individual metadata flags.
    # None when the instruction was not issued through the strategy ladder.
    ladder_ctx: Optional[LadderContext] = field(default=None, compare=False, repr=False)
    action_class: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate required fields based on kind."""
        _require_symbol_for_kinds(self, {EditInstructionKind.REPLACE_SYMBOL_BODY,
                                       EditInstructionKind.INSERT_AFTER_SYMBOL,
                                       EditInstructionKind.PATCH_SYMBOL,
                                       EditInstructionKind.SURGICAL_EDIT,
                                       EditInstructionKind.RENAME_SYMBOL,
                                       EditInstructionKind.MOVE_SYMBOL})
        if self.kind in (EditInstructionKind.CREATE_FILE, EditInstructionKind.OVERWRITE_FILE, EditInstructionKind.DELETE_FILE):
            # These must have file_path but no symbol
            pass


@dataclass
class ExecutionResult:
    success: bool
    final_status: str

    final_failure_class: Optional[str] = None

    modified_files: list[str] = field(default_factory=list)
    touched_symbols: list[str] = field(default_factory=list)

    apply_strategies: list[str] = field(default_factory=list)

    warnings: list[str] = field(default_factory=list)
    blocking_reasons: list[str] = field(default_factory=list)
    advisory_warnings: list[str] = field(default_factory=list)

    verification: dict = field(default_factory=dict)
    repair: dict = field(default_factory=dict)
    proof: dict = field(default_factory=dict)

    run_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    # ── Verdict fields (canonical authority — set by execute_plan_canonical) ──
    # 3-layer status: execution → verification → final
    execution_status: Optional[str] = None      # "all_completed" | "partial" | "all_failed" | None
    verification_status: Optional[str] = None   # "passed" | "passed_partial" | "failed" | "skipped"

    # Semantic gate (raw + effective after coverage override)
    semantic_gate_passed_raw: Optional[bool] = None
    effective_semantic_gate_passed: Optional[bool] = None
    semantic_gate_failed_reasons: list[str] = field(default_factory=list)

    # Final outcome summary (from _build_final_outcome_summary)
    final_blocking_reasons: list[str] = field(default_factory=list)
    final_warning_reasons: list[str] = field(default_factory=list)
    final_summary: dict = field(default_factory=dict)

    # Semantic aggregation (carried through for execute_plan metadata)
    semantic_summary: dict = field(default_factory=dict)
    plan_acceptance_passed: Optional[bool] = None

    # Execution log: per-op determinism records for replay and analysis
    execution_log: list[dict[str, Any]] = field(default_factory=list)
    # Plan-level failure lineage: op_id → failure_chain from each op
    plan_failure_chains: dict[str, list[str]] = field(default_factory=dict)

    # ── Verdict chain flags (set by producers, read by _finalize_canonical_verdict) ──
    # These advisory flags capture WHICH verifier signaled pass/fail.  They do NOT
    semantic_pass: Optional[bool] = None     # semantic_verifier.verify_edit_v2()
    structural_pass: Optional[bool] = None   # rewrite_transaction / canonical repair
    intent_pass: Optional[bool] = None       # intent_verifier assertions
    placement_pass: Optional[bool] = None    # placement_contract.verify_placement_contract
    quality_pass: Optional[bool] = None      # quality gate

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "final_status": self.final_status,
            "final_failure_class": self.final_failure_class,
            "modified_files": self.modified_files,
            "touched_symbols": self.touched_symbols,
            "apply_strategies": self.apply_strategies,
            "warnings": self.warnings,
            "blocking_reasons": self.blocking_reasons,
            "advisory_warnings": self.advisory_warnings,
            "verification": self.verification,
            "repair": self.repair,
            "proof": self.proof,
            "run_id": self.run_id,
            "metadata": self.metadata,
            # Verdict fields
            "execution_status": self.execution_status,
            "verification_status": self.verification_status,
            "semantic_gate_passed_raw": self.semantic_gate_passed_raw,
            "effective_semantic_gate_passed": self.effective_semantic_gate_passed,
            "semantic_gate_failed_reasons": self.semantic_gate_failed_reasons,
            "final_blocking_reasons": self.final_blocking_reasons,
            "final_warning_reasons": self.final_warning_reasons,
            "final_summary": self.final_summary,
            "semantic_summary": self.semantic_summary,
            "plan_acceptance_passed": self.plan_acceptance_passed,
            "plan_failure_chains": self.plan_failure_chains,
            # Verdict chain flags
            "semantic_pass": self.semantic_pass,
            "structural_pass": self.structural_pass,
            "intent_pass": self.intent_pass,
            "placement_pass": self.placement_pass,
            "quality_pass": self.quality_pass,
        }


@dataclass
class FailureTrace:
    """Structured failure trace for a single instruction execution.

    Current system: max 1 repair cycle → root/transitions/terminal is sufficient.
    Future (multi-cycle repair): replace with steps: List[Tuple[str, Optional[str]]]
      where each step = (failure_class, repair_action_that_was_tried).
      transitions field is intentionally kept flat until then to avoid premature
      complexity — it records intermediate failure classes within the single cycle.

    Fields:
      root        — first failure_class before any repair attempt (root cause)
      transitions — intermediate failure states within the repair cycle
      terminal    — final failure_class immediately before return

    as_list() produces the legacy list[str] so all existing consumers
    (FallbackStrategyEngine, plan_failure_chains, etc.) see no change.
    """

    root: str = "unknown"
    transitions: list[str] = field(default_factory=list)
    terminal: str = "unknown"

    def append(self, failure_class: str) -> None:
        """Record the next failure class in the trace."""
        if self.root == "unknown":
            # First call: root cause established
            self.root = failure_class
            self.terminal = failure_class
        elif failure_class != self.terminal:
            # Subsequent: push old terminal into transitions (if not root duplicate)
            if self.terminal != self.root:
                self.transitions.append(self.terminal)
            self.terminal = failure_class

    def as_list(self) -> list[str]:
        """Backwards-compatible list[str] for consumers expecting old failure_chain format."""
        if self.root == "unknown":
            return []
        result = [self.root]
        for t in self.transitions:
            if t not in result:
                result.append(t)
        if (
            self.terminal != self.root
            and self.terminal != "unknown"
            and self.terminal not in result
        ):
            result.append(self.terminal)
        return result

    # Allow FailureTrace to be used transparently where list[str] is expected
    def __bool__(self) -> bool:
        return self.root != "unknown"

    def __len__(self) -> int:
        return len(self.as_list())

    def __iter__(self):
        return iter(self.as_list())


class InstructionApplyResult(dict):
    """Per-instruction apply result from _apply_edit_instruction_with_repair.

    Inherits dict for backwards compatibility:
      - isinstance(result, dict) → True  (all consumer guards still pass)
      - result.get("key")       → works  (no consumer changes needed)
      - result["key"]           → works

    Typed properties give IDE support and static-analysis benefits for new code.
    """

    # ── Typed accessors ───────────────────────────────────────────────────────

    @property
    def status(self) -> str:
        return self.get("status", "unknown")  # type: ignore[return-value]

    @property
    def ok(self) -> bool:
        return self.get("status") == "success"

    @property
    def failure_class(self) -> Optional[str]:
        return self.get("failure_class")

    @property
    def failure_trace(self) -> "FailureTrace":
        return self.get("failure_trace", FailureTrace())  # type: ignore[return-value]

    @property
    def apply_strategy(self) -> str:
        return self.get("apply_strategy", "unknown")  # type: ignore[return-value]

    @property
    def precision_mode(self) -> str:
        return self.get("precision_mode", "unknown")  # type: ignore[return-value]

    @property
    def retry_success(self) -> bool:
        return bool(self.get("retry_success", False))

    @property
    def repair_attempted(self) -> bool:
        return bool(self.get("repair_attempted", False))

    @property
    def repair_action(self) -> Optional[str]:
        return self.get("repair_action")

    @property
    def patch_applied(self) -> Optional[str]:
        return self.get("patch_applied")

    @property
    def tool_error(self) -> Optional[str]:
        return self.get("tool_error")

    @property
    def instruction(self) -> Optional["EditInstruction"]:
        return self.get("instruction")


@dataclass
class CandidateSelectionFeedback:
    """Structured artifact capturing candidate selection rationale and execution outcome."""

    # Request and candidate generation context
    request: str = ""
    candidate_count: int = 0
    selected_candidate_id: Optional[str] = None
    selected_strategy: Optional[str] = None
    selected_score: Optional[float] = None
    selected_source: Optional[str] = None

    # Rejected candidates (summaries)
    rejected_candidates: list[dict[str, Any]] = field(default_factory=list)

    # Ranking explanations (from selected candidate)
    ranking_explanations: list[str] = field(default_factory=list)

    # Strategy distribution across all candidates
    strategy_distribution: list[str] = field(default_factory=list)

    # Graph and simulation usage
    graph_used: Optional[bool] = None
    impact_simulation_enabled: Optional[bool] = None
    graph_repo_root: Optional[str] = None

    # Execution outcome (filled after execution)
    execution_run_id: Optional[str] = None
    execution_success: Optional[bool] = None
    final_status: Optional[str] = None
    final_failure_class: Optional[str] = None

    # Verification and proof summaries (extracted from execution result)
    verification_summary: dict[str, Any] = field(default_factory=dict)
    proof_summary: dict[str, Any] = field(default_factory=dict)

    # Rollback and repair attempts
    rolled_back: Optional[bool] = None
    repair_attempted: Optional[bool] = None

    # Strategy switching metadata
    switched_from_strategy: Optional[str] = None
    switched_to_strategy: Optional[str] = None
    switch_reason: Optional[str] = None
    switch_hop: Optional[int] = None
    previous_strategies_tried: list[str] = field(default_factory=list)
    switch_chain: list[str] = field(default_factory=list)
    switch_memory_score: Optional[float] = None
    switch_memory_explanations: list[str] = field(default_factory=list)

    # Multi-hop strategy chain metadata (Phase 5.5)
    strategy_chain: list[str] = field(default_factory=list)
    strategy_chain_score: Optional[float] = None
    strategy_chain_depth: Optional[int] = None
    strategy_chain_initial_strategy: Optional[str] = None
    strategy_chain_final_strategy: Optional[str] = None
    is_multi_hop_strategy_chain: Optional[bool] = None

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "request": self.request,
            "candidate_count": self.candidate_count,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_strategy": self.selected_strategy,
            "selected_score": self.selected_score,
            "selected_source": self.selected_source,
            "rejected_candidates": self.rejected_candidates,
            "ranking_explanations": self.ranking_explanations,
            "strategy_distribution": self.strategy_distribution,
            "graph_used": self.graph_used,
            "impact_simulation_enabled": self.impact_simulation_enabled,
            "graph_repo_root": self.graph_repo_root,
            "execution_run_id": self.execution_run_id,
            "execution_success": self.execution_success,
            "final_status": self.final_status,
            "final_failure_class": self.final_failure_class,
            "verification_summary": self.verification_summary,
            "proof_summary": self.proof_summary,
            "rolled_back": self.rolled_back,
            "repair_attempted": self.repair_attempted,
            "switched_from_strategy": self.switched_from_strategy,
            "switched_to_strategy": self.switched_to_strategy,
            "switch_reason": self.switch_reason,
            "switch_hop": self.switch_hop,
            "previous_strategies_tried": self.previous_strategies_tried,
            "switch_chain": self.switch_chain,
            "switch_memory_score": self.switch_memory_score,
            "switch_memory_explanations": self.switch_memory_explanations,
            "strategy_chain": self.strategy_chain,
            "strategy_chain_score": self.strategy_chain_score,
            "strategy_chain_depth": self.strategy_chain_depth,
            "strategy_chain_initial_strategy": self.strategy_chain_initial_strategy,
            "strategy_chain_final_strategy": self.strategy_chain_final_strategy,
            "is_multi_hop_strategy_chain": self.is_multi_hop_strategy_chain,
            "metadata": self.metadata,
        }


def build_candidate_feedback_from_plan(
    plan: OperationPlan,
    request: str = "",
) -> CandidateSelectionFeedback:
    """Build a feedback artifact from a plan's candidate selection metadata."""
    selection = plan.metadata.get("candidate_selection", {})
    if not selection:
        # No candidate selection metadata; return empty artifact
        return CandidateSelectionFeedback(request=request)

    # Extract selected strategy from candidate metadata
    selected_strategy = selection.get("selected_strategy")
    if selected_strategy is None:
        # Fallback to extracting from selected candidate metadata if available
        selected_candidate_metadata = selection.get("selected_candidate_metadata", {})
        selected_strategy = selected_candidate_metadata.get("strategy")

    feedback = CandidateSelectionFeedback(
        request=request,
        candidate_count=selection.get("candidate_count", 0),
        selected_candidate_id=selection.get("selected_candidate_id"),
        selected_strategy=selected_strategy,
        selected_score=selection.get("selected_candidate_score"),
        selected_source=selection.get("selected_candidate_source"),
        rejected_candidates=selection.get("rejected_candidates", []),
        ranking_explanations=selection.get("ranking_explanations", []),
        strategy_distribution=selection.get("strategy_distribution", []),
        graph_used=selection.get("graph_used"),
        impact_simulation_enabled=selection.get("impact_simulation_enabled"),
        graph_repo_root=selection.get("graph_repo_root"),
    )
    # Extract switch metadata if present
    switch_metadata = selection.get("switch_metadata")
    if isinstance(switch_metadata, dict):
        feedback.switched_from_strategy = switch_metadata.get("switched_from_strategy")
        feedback.switched_to_strategy = switch_metadata.get("switched_to_strategy")
        feedback.switch_reason = switch_metadata.get("switch_reason")
        feedback.switch_hop = switch_metadata.get("switch_hop")
        feedback.previous_strategies_tried = switch_metadata.get("previous_strategies_tried", [])
        feedback.switch_chain = switch_metadata.get("switch_chain", [])
        feedback.switch_memory_score = switch_metadata.get("switch_memory_score")
        feedback.switch_memory_explanations = switch_metadata.get("switch_memory_explanations", [])

    # Extract strategy chain metadata if present
    selected_candidate_metadata = selection.get("selected_candidate_metadata", {})
    if selected_candidate_metadata:
        feedback.strategy_chain = selected_candidate_metadata.get("strategy_chain", [])
        feedback.strategy_chain_score = selected_candidate_metadata.get("strategy_chain_score")
        feedback.strategy_chain_depth = selected_candidate_metadata.get("strategy_chain_depth")
        feedback.strategy_chain_initial_strategy = selected_candidate_metadata.get("strategy_chain_initial_strategy")
        feedback.strategy_chain_final_strategy = selected_candidate_metadata.get("strategy_chain_final_strategy")
        feedback.is_multi_hop_strategy_chain = selected_candidate_metadata.get("is_multi_hop_strategy_chain")
    return feedback


def enrich_candidate_feedback_with_execution(
    feedback: CandidateSelectionFeedback,
    execution_result: ExecutionResult,
) -> CandidateSelectionFeedback:
    """Enrich a feedback artifact with execution outcome."""
    feedback.execution_run_id = execution_result.run_id
    feedback.execution_success = execution_result.success
    feedback.final_status = execution_result.final_status
    feedback.final_failure_class = execution_result.final_failure_class

    # Extract verification summary (simplify)
    verification = execution_result.verification
    if isinstance(verification, dict):
        feedback.verification_summary = {
            "success": verification.get("success"),
            "blocking_reasons": verification.get("blocking_reasons", []),
            "warnings": verification.get("warnings", []),
        }
    else:
        feedback.verification_summary = {"raw": str(verification)}

    # Extract proof summary (key fields)
    proof = execution_result.proof
    if isinstance(proof, dict):
        feedback.proof_summary = {
            "modified_files": proof.get("modified_files", []),
            "rolled_back": proof.get("rolled_back", False),
        }
    else:
        feedback.proof_summary = {"raw": str(proof)}

    # Determine rolled_back from proof or final_status
    feedback.rolled_back = proof.get("rolled_back") if isinstance(proof, dict) else execution_result.final_status in ["verification_failed", "execution_error"]
    # repair_attempted: check if repair dict is non-empty
    feedback.repair_attempted = bool(execution_result.repair)

    return feedback


@dataclass
class StrategyOutcomeStats:
    """Aggregated statistics for a single strategy across multiple runs."""
    strategy: str
    selected_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    verification_failure_count: int = 0
    rollback_count: int = 0
    repair_attempt_count: int = 0
    avg_selected_score: float = 0.0
    recent_requests: list[str] = field(default_factory=list)
    request_type_counts: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "selected_count": self.selected_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "verification_failure_count": self.verification_failure_count,
            "rollback_count": self.rollback_count,
            "repair_attempt_count": self.repair_attempt_count,
            "avg_selected_score": self.avg_selected_score,
            "recent_requests": self.recent_requests,
            "request_type_counts": self.request_type_counts,
            "metadata": self.metadata,
        }


@dataclass
class StrategyOutcomeMemory:
    """Aggregated memory of strategy outcomes across runs."""
    strategies: dict[str, StrategyOutcomeStats] = field(default_factory=dict)
    total_runs_considered: int = 0
    generated_at: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategies": {k: v.to_dict() for k, v in self.strategies.items()},
            "total_runs_considered": self.total_runs_considered,
            "generated_at": self.generated_at,
            "metadata": self.metadata,
        }


@dataclass
class SwitchOutcomeStats:
    from_strategy: str
    to_strategy: str

    attempted_count: int = 0
    success_count: int = 0
    failure_count: int = 0

    verification_failure_count: int = 0
    rollback_count: int = 0
    repair_attempt_count: int = 0

    avg_selected_score: float = 0.0

    recent_requests: list[str] = field(default_factory=list)
    recent_failure_classes: list[str] = field(default_factory=list)

    request_type_counts: dict[str, int] = field(default_factory=dict)
    request_type_success_counts: dict[str, int] = field(default_factory=dict)
    request_type_failure_counts: dict[str, int] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_strategy": self.from_strategy,
            "to_strategy": self.to_strategy,
            "attempted_count": self.attempted_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "verification_failure_count": self.verification_failure_count,
            "rollback_count": self.rollback_count,
            "repair_attempt_count": self.repair_attempt_count,
            "avg_selected_score": self.avg_selected_score,
            "recent_requests": self.recent_requests,
            "recent_failure_classes": self.recent_failure_classes,
            "request_type_counts": self.request_type_counts,
            "request_type_success_counts": self.request_type_success_counts,
            "request_type_failure_counts": self.request_type_failure_counts,
            "metadata": self.metadata,
        }


@dataclass
class SwitchOutcomeMemory:
    transitions: dict[str, SwitchOutcomeStats] = field(default_factory=dict)
    total_switch_runs_considered: int = 0
    generated_at: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transitions": {k: v.to_dict() for k, v in self.transitions.items()},
            "total_switch_runs_considered": self.total_switch_runs_considered,
            "generated_at": self.generated_at,
            "metadata": self.metadata,
        }


def classify_request_type(request: str) -> str:
    """DEPRECATED: request_type is now set by SpecResolver LLM.

    Kept for backward compatibility with tests. Returns 'unknown' —
    callers should use spec.request_type instead.
    """
    return "unknown"
