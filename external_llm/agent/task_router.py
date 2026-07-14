"""
Task Router for asicode Agent

Two execution lanes:
- PLANNER: structured pipeline (spec → operation plan → execute with verification/repair)
- MAIN_AGENT: LLM tool-use loop (read, search, modify via tools, no structured planning)

Lane resolution is fixed to MAIN_AGENT (PLANNER lane permanently disabled,
Tier 3 consolidation). RouteFeatures are still extracted for complexity/scope
metadata, but no longer drive lane selection.
No keyword-based gating. See DeterministicClassifier.decide_flow() for details.

Execution modes:
- planner:     Structured operation pipeline for AST-supported language edits
- main_agent:  Tool-use loop for non-structured files, exploratory/ambiguous edits
- clarify:     Handled inside PLANNER via Semantic-Fit Judge (not a separate lane)
- read_only:   Handled inside PLANNER as READ_ONLY_ANALYSIS operations (not a lane)

FAST_PATH has been absorbed into PLANNER:
- Trivial edits → DeterministicPlanBuilder (LLM 0-call)
- CSS/HTML → ANCHOR_EDIT operations
- Low confidence → SpecResolver handles file discovery

Helper is NOT a lane. Helper is a tool capability (delegate_to_helper) that the
Developer can call from any lane when it needs a subordinate model for code generation.
Helper ON/OFF is controlled by AgentConfig.helper_enabled, not by routing.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ..languages import LanguageRegistry
from .config.thresholds import config as _cfg
from .enums import Complexity, Scope
from .intent_models import IntentResult
from .intent_resolver import create_intent_resolver
from .language_hint import is_cjk

logger = logging.getLogger(__name__)


# ── Enums ────────────────────────────────────────────────────────────────────

class TaskKind(str, Enum):
    MICRO_EDIT = "MICRO_EDIT"
    SINGLE_FILE_EDIT = "SINGLE_FILE_EDIT"
    MULTI_FILE_FEATURE = "MULTI_FILE_FEATURE"
    REFACTOR = "REFACTOR"
    EXPLORATION = "EXPLORATION"
    TEST_WRITE = "TEST_WRITE"
    STYLE_FIX = "STYLE_FIX"
    BOILERPLATE = "BOILERPLATE"


# Complexity and Scope moved to enums.py; imported above.


class Lane(str, Enum):
    PLANNER = "planner"              # Structured pipeline: spec → plan → execute
    MAIN_AGENT = "main_agent"        # Tool-use loop: _run_llm_loop directly


# File extensions recognized as AST-parseable (used for lane decisions).
AST_EXTENSIONS = frozenset({
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".kt",
    ".rb", ".cs", ".cpp", ".c", ".h",
})


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class RouteDecision:
    task_kind: TaskKind
    complexity: Complexity
    scope: Scope
    lane: Lane
    confidence: float  # 0.0-1.0
    reasoning: str = ""

    # Planner signal: True = complex task, run Planner before Developer.
    # False (default) = simple task, go directly to Developer.
    requires_planner: bool = False

    # Config overrides — None means keep existing value
    planning_enabled: Optional[bool] = None
    self_review_enabled: Optional[bool] = None
    auto_test_on_patch: Optional[bool] = None
    rag_enabled: Optional[bool] = None
    multi_agent: Optional[bool] = None
    max_turns_override: Optional[int] = None

    # LLM target hints: modify_files, new_files from LLM classification
    llm_target_hints: Optional[dict[str, Any]] = None

    # (pre_resolved_spec and explore_report removed — ExploreAgent pipeline removed)

    # Target specificity score (0.0 = maximally ambiguous, 1.0 = fully specified)
    target_specificity_score: float = 0.5

    # READ_ONLY: question type for readonly pipeline
    readonly_kind: Optional[str] = None

    # Intent understanding result from IntentResolver (language-neutral, LLM-powered)
    intent_result: Optional[IntentResult] = None


# ── Route Features ──────────────────────────────────────────────────────────

@dataclass
class RouteFeatures:
    """Structural features extracted from a request.

    decide_flow() reads: has_edit_intent, has_explicit_file, has_explicit_symbol,
    has_specific_change_object, is_multi_file, is_project_wide, has_conflicting_intent,
    all_targets_non_structured, any_target_structured, mentioned_files, requests_new_file,
    requests_test_work, target_specificity_score, task_kind, complexity, scope.

    Other fields are intermediate — used within extract_features() and
    _classify_task_meta() to derive the above.
    """

    # raw / lexical
    request: str = ""              # intermediate
    request_lower: str = ""
    word_count: int = 0

    # file/symbol extraction
    mentioned_files: list[str] = field(default_factory=list)
    existing_files: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    mentioned_symbols: list[str] = field(default_factory=list)  # intermediate

    # intent
    has_edit_intent: bool = False
    has_read_intent: bool = False   # intermediate
    has_explain_intent: bool = False  # intermediate
    has_locate_intent: bool = False  # intermediate
    has_question_form: bool = False  # intermediate

    # task shape
    requests_new_file: bool = False
    requests_filesystem_op: bool = False  # intermediate
    requests_refactor: bool = False  # intermediate
    requests_test_work: bool = False
    requests_boilerplate: bool = False  # intermediate
    requests_style_change: bool = False  # intermediate
    requests_ui_change: bool = False  # intermediate

    # targeting / specificity
    has_explicit_file: bool = False
    has_explicit_symbol: bool = False
    has_anchor_or_exact_target: bool = False  # intermediate
    has_specific_change_object: bool = False
    target_specificity_score: float = 0.0

    # scope
    file_count: int = 0            # intermediate
    symbol_count: int = 0          # intermediate
    is_single_file: bool = True    # intermediate
    is_multi_file: bool = False
    is_project_wide: bool = False
    has_cross_file_signal: bool = False  # intermediate
    has_propagation_signal: bool = False  # intermediate

    # ambiguity / conflict
    is_ambiguous_write: bool = False  # intermediate
    has_conflicting_intent: bool = False

    # triviality
    looks_trivial_edit: bool = False  # intermediate

    # language / capability
    all_targets_structured: Optional[bool] = None  # intermediate
    any_target_structured: Optional[bool] = None
    all_targets_non_structured: Optional[bool] = None

    # readonly analysis subtype
    readonly_kind: Optional[str] = None  # intermediate

    # task classification (derived by _classify_task_meta, passed to RouteDecision)
    task_kind: TaskKind = TaskKind.SINGLE_FILE_EDIT
    complexity: Complexity = Complexity.LOW
    scope: Scope = Scope.SINGLE_FILE


# ── Deterministic Classifier ──────────────────────────────────────────────────

class DeterministicClassifier:
    """
    Feature-based sequential gate classifier.
    Runs in ~0ms with no LLM calls.

    Flow: extract_features(request) → decide_flow(features) → RouteDecision
    Keywords are hints for feature extraction ONLY, never directly decide lanes.
    """

    # ── Structural extraction patterns (text structure, not semantic intent) ──
    # These extract file paths and symbol names from text — structural operations
    # that regex handles well. Semantic intent signals come from IntentResult.

    # File extension pattern is dynamically generated from LanguageRegistry
    # to stay in sync with registered language providers.
    _file_ext_pattern: Optional["re.Pattern"] = None

    @classmethod
    def _get_file_ext_pattern(cls) -> "re.Pattern":
        if cls._file_ext_pattern is None:
            cls._file_ext_pattern = re.compile(
                LanguageRegistry.instance().get_file_pattern(),
                re.IGNORECASE,
            )
        return cls._file_ext_pattern

    _SYMBOL_PATTERN = re.compile(
        r'(?:'
        r'[A-Z][a-z]+[A-Z]'              # PascalCase: "UserService", "TodoApp"
        r'|[a-z]+_[a-z]+'                 # snake_case: "create_user", "login_handler"
        r'|[a-z]+[A-Z][a-z]+'            # camelCase: "myFunction", "getUser"
        r'|`[^`]+`'                        # backtick quoted: `my_func`
        r'|class\s+\w+'                   # explicit: "class Foo"
        r'|def\s+\w+'                     # explicit: "def bar"
        r'|function\s+\w+'               # explicit: "function baz"
        r')',
    )

    # ── edit intent detection (structural fallback) ──

    @staticmethod
    def _has_edit_intent(text: str) -> bool:
        """Structural fallback for edit intent when IntentResult is absent.
        Uses language-neutral structural signals only.
        Defaults to True (presume edit) because PLANNER correctly handles
        non-edit requests as READ_ONLY_ANALYSIS at the plan level.
        """
        return True

    # ── IntentResult → RouteFeatures mapping ────────────────────────────────

    @staticmethod
    def _features_from_intent(
        intent_result: "IntentResult",
        f: RouteFeatures,
    ) -> None:
        """Populate RouteFeatures from LLM IntentResult (primary source).

        IntentResult is the LLM's structured understanding of the request.
        It provides higher-quality signals than regex pattern matching because
        the LLM understands context, language nuance, and domain meaning.

        Sets all fields that IntentResult provides signal for.
        Structural regex (file path extraction, symbol name extraction) still runs
        in extract_features() for features that are inherently textual.
        """
        it = intent_result.intent_type or ""
        lane = intent_result.lane_hint or ""

        # ── Intent → read/edit flags ──
        _read_intents = {"exploration", "question"}

        if it in _read_intents or lane == "read_only":
            f.has_read_intent = True
            f.has_explain_intent = (it == "question")
            f.has_locate_intent = (it == "question")  # questions often ask "where is X"
            f.has_question_form = (it == "question")
            f.has_edit_intent = False
        else:
            # Any non-read intent (known edit types + unknown future types) → edit
            f.has_edit_intent = True

        # ── Task shape from intent_type ──
        if it == "refactor":
            f.requests_refactor = True
        if it == "create":
            f.requests_new_file = True
            f.requests_boilerplate = True
        elif it in ("extend", "feature"):
            f.is_multi_file = True  # extend/feature → at least multi-file scope

        # ── Filesystem op / UI change — from dedicated IntentResult fields ──
        # These are distinct sub-kinds of main_agent tasks; read from precise fields
        # rather than the coarse lane_hint to avoid over-classification.
        if intent_result.is_filesystem_op:
            f.requests_filesystem_op = True
        if intent_result.is_ui_change:
            f.requests_ui_change = True

        # ── Style fix ──
        if intent_result.is_style_fix:
            f.requests_style_change = True
            f.looks_trivial_edit = True

        # ── Test write ──
        if intent_result.is_test_write:
            f.requests_test_work = True

        # ── Scope from scope_hint (LLM-derived, language-neutral) ──
        sh = intent_result.scope_hint
        if sh == Scope.PROJECT_WIDE:
            f.is_project_wide = True
            f.is_multi_file = True
            f.has_cross_file_signal = True
            f.has_propagation_signal = True
        elif sh == Scope.MULTI_FILE:
            f.is_multi_file = True
            f.has_cross_file_signal = True

        # ── Complexity → triviality ──
        if intent_result.complexity_hint == Complexity.LOW:
            f.looks_trivial_edit = True

        # ── Targeting from IntentResult ──
        if intent_result.target_symbols:
            f.has_specific_change_object = True
        if len(intent_result.target_files) >= 2:
            f.is_multi_file = True

    # ── Feature Extraction ───────────────────────────────────────────────────

    def extract_features(
        self,
        request: str,
        repo_root: Optional[str] = None,
        intent_result: Optional["IntentResult"] = None,
    ) -> RouteFeatures:
        """Extract structural features from request. Keywords → features, NOT lanes."""
        import os

        f = RouteFeatures()
        f.request_lower = request.lower().strip()
        f.word_count = len(request.split())

        # CJK normalisation: ~1.5 CJK chars ≈ 1 English word (empirically better than //2
 # for Korean-heavy requests like "3-5줄짜리 파이썬 function를 리팩터링해주세요").
        _cjk_count = sum(
            1 for c in request
            if is_cjk(c)
        )
        if _cjk_count > 5:
            f.word_count = max(f.word_count, _cjk_count * 2 // 3)

        f.request = request

        rl = f.request_lower  # shorthand

        # ── file / symbol extraction ──
        f.mentioned_files = self._get_file_ext_pattern().findall(request)
        f.file_count = len(set(f.mentioned_files))
        if repo_root and f.mentioned_files:
            for fp in f.mentioned_files:
                full = os.path.join(repo_root, fp)
                if os.path.exists(full):
                    f.existing_files.append(fp)
                else:
                    f.missing_files.append(fp)

        _sym_matches = self._SYMBOL_PATTERN.findall(request)
        # Strip backticks from backtick-quoted symbol names
        _sym_matches = [m.strip('`') for m in _sym_matches]
        f.mentioned_symbols = _sym_matches
        f.symbol_count = len(_sym_matches)

        # ── intent extraction ──
        # PRIMARY: IntentResult from LLM (if available and confident)
        _intent_is_primary = (
            intent_result is not None
            and intent_result.confidence >= 0.5
            and intent_result.intent_type != "unknown"
        )
        if _intent_is_primary:
            self._features_from_intent(intent_result, f)  # type: ignore[arg-type]

        # FALLBACK: regex-based edit intent detection when IntentResult is absent/low-confidence.
        # This is the only semantic regex fallback remaining — all other intent signals
        # come from IntentResult. OR-merge: can only ADD, never revoke IntentResult's signals.
        if not f.has_edit_intent and not f.has_read_intent:
            f.has_edit_intent = self._has_edit_intent(rl)

        # has_question_form: structural check for "?" form (supplement IntentResult)
        f.has_question_form = f.has_question_form or rl.rstrip().endswith("?")

        # new file detection: supplement IntentResult.create with actual missing file paths
        f.requests_new_file = f.requests_new_file or (
            bool(f.missing_files) and f.has_edit_intent
        )

        # ── targeting / specificity ──
        f.has_explicit_file = f.file_count > 0
        f.has_explicit_symbol = f.symbol_count > 0

        _has_line_or_block_anchor = bool(re.search(
            r'(?:line\s*\d+|this\s+block|import\s+section)',
            rl,
        ))
        _has_css_selector_anchor = bool(re.search(
            r'(?:\.[a-z][\w-]+|#[a-z][\w-]+|data-[\w-]+)',
            rl,
        ))
        _has_exact_string_anchor = bool(re.search(r'"[^"]{2,}"', request))

        f.has_anchor_or_exact_target = any([
            f.has_explicit_file,
            f.has_explicit_symbol,
            _has_line_or_block_anchor,
            _has_css_selector_anchor,
            _has_exact_string_anchor,
        ])

        # has_specific_change_object: TRUE if the request mentions a concrete
        # change target (not just a vague "fix this"). Uses IntentResult's
        # target_symbols (LLM judgment) as primary, with regex symbol detection
        # as fallback. Replaces former _DOMAIN_KEYWORDS list which was domain-biased.
        if not f.has_specific_change_object:  # may already be set by _features_from_intent
            f.has_specific_change_object = (
                f.has_explicit_symbol
                or f.has_explicit_file
                or (intent_result is not None and bool(intent_result.target_symbols))
            )

        # Weighted specificity score
        _spec_score = 0.0
        if f.has_explicit_file:
            _spec_score += 0.35
        if f.has_explicit_symbol:
            _spec_score += 0.30
        if f.has_anchor_or_exact_target and not f.has_explicit_file and not f.has_explicit_symbol:
            _spec_score += 0.20
        if f.has_specific_change_object:
            _spec_score += 0.15
        f.target_specificity_score = min(_spec_score, 1.0)

        # ── scope extraction ──
        # Primary: from IntentResult.scope_hint (already applied in _features_from_intent).
        # Supplement: file_count >= 2 is structural evidence of multi-file scope.
        f.is_multi_file = f.is_multi_file or f.file_count >= 2
        # P5: extend/feature intent with single file → downgrade to single_file
        # when there is no cross-file evidence. The extend intent type is a weak
        # signal that inflates scope — file_count + cross-file signal are stronger.
        if (f.is_multi_file
            and _intent_is_primary
            and intent_result is not None
            and intent_result.intent_type in ("extend", "feature")
            and f.file_count <= 1
            and not f.has_cross_file_signal
            and not f.has_propagation_signal):
            f.is_multi_file = False
            f.is_single_file = True
        else:
            f.is_single_file = not f.is_multi_file

        # ── language / capability ──
        if f.mentioned_files:
            _registry = LanguageRegistry.instance()
            _structured = [_registry.supports_structured_ops(fp) for fp in f.mentioned_files]
            f.all_targets_structured = all(_structured)
            f.any_target_structured = any(_structured)
            f.all_targets_non_structured = not any(_structured)

        # ── ambiguity extraction ──
        f.has_conflicting_intent = (
            f.has_edit_intent
            and (f.has_read_intent or f.has_explain_intent or f.has_locate_intent)
        )

        # locate+question form = asking WHERE, not requesting edit
        _is_locate_question = (
            f.has_locate_intent and f.has_question_form and not f.has_explicit_file
        )
        # conflicting intent (read+edit both) = has a task structure, not ambiguous
        f.is_ambiguous_write = (
            f.has_edit_intent
            and not f.has_explicit_file
            and not f.has_explicit_symbol
            and not f.has_specific_change_object
            and not f.has_propagation_signal
            and not f.has_cross_file_signal
            and not _is_locate_question
            and not (f.has_conflicting_intent and f.word_count > 5)
            and f.word_count <= 20
        )

        # ── readonly subtype ──
        if f.has_read_intent or f.has_explain_intent or f.has_locate_intent:
            if f.has_locate_intent and (f.requests_ui_change or f.requests_style_change):
                f.readonly_kind = "locate_ui_style"
            elif f.has_locate_intent:
                f.readonly_kind = "locate_code"
            elif f.has_explain_intent:
                f.readonly_kind = "code_explain"
            else:
                f.readonly_kind = "general_readonly"

        # ── task classification (for RouteDecision compatibility) ──
        f.task_kind, f.complexity, f.scope = self._classify_task_meta(f)

        return f

    def _classify_task_meta(
        self, f: RouteFeatures,
    ) -> tuple:
        """Derive task_kind, complexity, scope from features (for RouteDecision fields)."""
        # Task kind
        if f.looks_trivial_edit:
            task_kind = TaskKind.MICRO_EDIT
        elif f.requests_style_change and not f.requests_refactor:
            task_kind = TaskKind.STYLE_FIX
        elif f.requests_refactor:
            task_kind = TaskKind.REFACTOR
        elif f.requests_test_work:
            task_kind = TaskKind.TEST_WRITE
        elif f.requests_boilerplate:
            task_kind = TaskKind.BOILERPLATE
        elif not f.has_edit_intent and (f.has_read_intent or f.has_explain_intent or f.has_locate_intent):
            task_kind = TaskKind.EXPLORATION
        elif f.is_multi_file:
            task_kind = TaskKind.MULTI_FILE_FEATURE
        else:
            task_kind = TaskKind.SINGLE_FILE_EDIT

        # Complexity
        # NOTE: the "vague refactor" escalation (no files named + substantial
        # length) must be checked BEFORE the generic word/file ladders below,
        # otherwise the `word_count > 25` (MEDIUM) guard shadows it and the
        # branch is unreachable. Such requests are underspecified and riskier,
        # so they deserve HIGH.
        if f.requests_refactor and f.file_count == 0 and f.word_count >= 30:
            complexity = Complexity.HIGH
        elif f.file_count >= 3 or f.word_count > 60:
            complexity = Complexity.HIGH
        elif f.is_multi_file or f.file_count >= 2 or f.word_count > 25:
            complexity = Complexity.MEDIUM
        else:
            complexity = Complexity.LOW

        # Scope
        if f.is_project_wide:
            scope = Scope.PROJECT_WIDE
        elif f.is_multi_file:
            scope = Scope.MULTI_FILE
        else:
            scope = Scope.SINGLE_FILE

        return task_kind, complexity, scope

    def decide_flow(self, f: RouteFeatures, intent_result: Optional["IntentResult"] = None) -> RouteDecision:
        """Build a MAIN_AGENT routing decision for the request.

        PLANNER lane is permanently disabled (Tier 3 consolidation). All
        requests route to MAIN_AGENT (direct LLM tool-use loop).

        Args:
            f: Extracted structural features for the request.
            intent_result: LLM intent analysis. Unused now that PLANNER routing
                is disabled, but kept for API compatibility with callers that
                still pass it (e.g. TaskRouter.route via classify()).

        Returns:
            RouteDecision always configured for the MAIN_AGENT lane.
        """
        return self._build_main_agent_decision(f)

    def _build_main_agent_decision(self, f: RouteFeatures) -> RouteDecision:
        """Build RouteDecision for MAIN_AGENT lane (LLM tool-use loop).

        MAIN_AGENT skips the structured PLANNER pipeline. The LLM directly uses
        tools to read, search, and modify files. Best for:
        - Straightforward changes where pipeline overhead isn't justified
        - Exploratory/ambiguous edits where early spec resolution adds cost
        - Any language/file type — MAIN_AGENT is a capability lane, not a file-type lane

        Infra features (self-review, RAG) are enabled proportionally to complexity.
        Only the planning pipeline itself is skipped (planning_enabled=False).
        """
        _conf = self._compute_confidence(f)
        _reason = self._build_main_agent_reason(f)

        return RouteDecision(
            task_kind=f.task_kind,
            complexity=f.complexity,
            scope=f.scope,
            lane=Lane.MAIN_AGENT,
            requires_planner=False,
            confidence=_conf,
            reasoning=_reason,
            planning_enabled=False,                           # MAIN_AGENT = direct tool loop
            self_review_enabled=f.complexity != Complexity.LOW,  # Self-review is useful for complex changes
            auto_test_on_patch=f.requests_test_work,          # Only when the user explicitly requests it
            rag_enabled=False,                                 #RAG (topic) PLANNER tier dedicated (tier check)
            multi_agent=False,
            target_specificity_score=f.target_specificity_score,
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_confidence(f: RouteFeatures) -> float:
        score = 0.70
        if f.has_explicit_file:
            score += 0.08
        if f.has_explicit_symbol:
            score += 0.07
        if f.has_specific_change_object:
            score += 0.05
        if f.is_multi_file:
            score -= 0.05
        if f.is_project_wide:
            score -= 0.08
        if f.has_conflicting_intent:
            score -= 0.03
        if f.all_targets_non_structured:
            score -= 0.05
        return max(0.55, min(score, 0.92))

    @staticmethod
    def _build_main_agent_reason(f: RouteFeatures) -> str:
        parts = []
        if f.all_targets_non_structured:
            parts.append("non-structured targets")
        if f.has_edit_intent:
            parts.append("edit request")
        if not f.has_specific_change_object:
            parts.append("ambiguous target")
        if f.requests_filesystem_op:
            parts.append("filesystem operation")
        if f.requests_ui_change:
            parts.append("UI/style change")
        return "MAIN_AGENT: " + ", ".join(parts) if parts else "MAIN_AGENT default path"

    # ── Public classify() entry point ────────────────────────────────────────

    def classify(
        self,
        request: str,
        repo_root: Optional[str] = None,
        intent_result: Optional["IntentResult"] = None,
    ) -> RouteDecision:
        """Feature-based routing: extract_features() → decide_flow()."""
        features = self.extract_features(
            request, repo_root=repo_root, intent_result=intent_result,
        )
        return self.decide_flow(features, intent_result=intent_result)


# ── Task Router ────────────────────────────────────────────────────────────────

class TaskRouter:
    """
    Two-stage hybrid router: Deterministic + Explore-First.

    Stage 1: DeterministicClassifier (always runs, ~0ms, no LLM cost)
    Stage 2: Explore-First (when deterministic confidence < EXPLORE_FIRST_THRESHOLD
             AND lane is PLANNER or MAIN_AGENT — triggers file exploration for accurate
             routing, replacing the old LLM classification stage which always returned
             confidence=0.80 and masked real uncertainty)

    Usage:
        router = TaskRouter(llm_client=svc.llm_service.client, model=svc.model)
        decision = router.route(request_text, repo_root=repo_root)
        # decision.lane → which execution path to use
    """

    # Replaces old LLM_FALLBACK_THRESHOLD (0.85) + EXPLORE_CONFIDENCE_THRESHOLD (0.60) pair.
    # When deterministic confidence < this, go straight to Explore-First instead of LLM guessing.
    EXPLORE_FIRST_THRESHOLD = _cfg.scores.EXPLORE_FIRST_THRESHOLD

    # Lane-specific default config overrides (only fills None fields in RouteDecision)
    _LANE_DEFAULTS: dict[str, dict[str, Any]] = {
        Lane.PLANNER: {
            "planning_enabled": True,
            "self_review_enabled": True,
            "auto_test_on_patch": False,
            "rag_enabled": True,
            "multi_agent": None,
        },
        Lane.MAIN_AGENT: {
            "planning_enabled": False,
            "self_review_enabled": False,
            "auto_test_on_patch": False,
            "rag_enabled": False,
            "multi_agent": False,
        },
    }

    def __init__(
        self,
        llm_client: Any = None,
        model: str = "",
        repo_root: Optional[str] = None,
    ):
        self._deterministic = DeterministicClassifier()
        self._intent_resolver = create_intent_resolver(
            llm_client=llm_client,
            model=model,
            repo_root=repo_root,
            enable_cache=True,
        )
        self._repo_root = repo_root

    def route(self, request: str, repo_root: Optional[str] = None) -> RouteDecision:
        """
        Route a user request to an execution lane.
        Returns RouteDecision with lane, task_kind, complexity, scope, confidence.
        """
        root = repo_root or self._repo_root

        # Stage 0: Intent Resolution (language-neutral, LLM-powered)
        intent_result = self._intent_resolver.resolve(request)
        logger.info(
            "Router intent resolution: query=%r, intent=%s, lane_hint=%s, confidence=%.2f",
            intent_result.normalized_query[:100],
            intent_result.intent_type,
            intent_result.lane_hint,
            intent_result.confidence,
        )

        # Stage 1: Deterministic (with IntentResult as primary signal source)
        decision = self._deterministic.classify(
            request, repo_root=root, intent_result=intent_result,
        )
        logger.info(
            "Router stage-1: kind=%s lane=%s confidence=%.2f reason=%r",
            decision.task_kind.value,
            decision.lane.value,
            decision.confidence,
            decision.reasoning,
        )

        # Attach intent result for downstream reuse (SpecResolver, etc.)
        decision.intent_result = intent_result

        # decide_flow() in DeterministicClassifier always routes to MAIN_AGENT
        # (PLANNER lane permanently disabled, Tier 3 consolidation).
        # RouteFeatures are still used for complexity/scope metadata.
        # MAIN_AGENT runs the direct LLM tool-use loop.

        # Apply lane-specific config defaults (only fills None fields)
        decision = self._apply_lane_defaults(decision)

        return decision

    def _apply_lane_defaults(self, decision: RouteDecision) -> RouteDecision:
        """Apply default config overrides for the selected lane (only fills None fields)."""
        defaults = self._LANE_DEFAULTS.get(decision.lane, {})
        for key, value in defaults.items():
            if getattr(decision, key, None) is None:
                setattr(decision, key, value)
        return decision
