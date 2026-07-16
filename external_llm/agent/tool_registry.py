# REGRESSION_TEST_01
"""
Tool Registry for asicode Agent

Provides safe tool dispatch for the LLM agent loop.
Security: all file operations are bounded by repo_root.
"""
from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional

if TYPE_CHECKING:
    import threading

    from .agent_profile import AgentProfile

import subprocess

from external_llm.common.indent_utils import reindent_text

from ..graph.graph_facade import RepositoryGraphFacade

# ── Extracted modules ────────────────────────────────────────────────────
from ..languages import LanguageId
from ._thread_pool import shared_pool
from .argument_repairer import ArgumentRepairer
from .call_graph import CallGraphIndexer
from .config.thresholds import config as _cfg
from .file_cache import get_global_file_cache
from .lint_runner import LintRunner
from .performance_metrics import get_global_collector
from .rag_searcher import RAGSearcher
from .symbol_search import SymbolSearcher
from .tool_chain import ScopedToolFilter
from .tool_dependency_graph import ToolDependencyGraph
from .tool_handlers.agent_tools import AgentToolsMixin
from .tool_handlers.analysis_tools import AnalysisToolsMixin
from .tool_handlers.browser_tools import BrowserActionToolsMixin

# ask_user default timeout (seconds) — defined once in leaf module, then re-exported.
# If tool_registry defined this directly, agent_tools would back-reference tool_registry
# for this constant, causing a circular import (triggered on standalone submodule import).
# See #constants.
from .tool_handlers.constants import ASK_USER_DEFAULT_TIMEOUT
from .tool_handlers.git_tools import ShellToolsMixin, _literal_intervals, _match_in_quotes
from .tool_handlers.read_tools import ReadToolsMixin
from .tool_handlers.test_tools import TestToolsMixin
from .tool_handlers.web_search_tools import WebSearchToolsMixin
from .tool_handlers.write_tools import WriteToolsMixin
from .tool_safety import WriteSafetyManager
from .tool_schemas import AGENT_TOOL_NAMES, AGENT_TOOL_SCHEMAS

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    max_turns: int = _cfg.counts.AGENT_MAX_TURNS_DEFAULT
    max_apply_attempts: int = 3
    run_tests: bool = False
    run_lint: bool = True
    # Scoped verification: after edits, run only tests likely affected by the
    # changed files (naming-convention + call-graph mapping) instead of the full
    # suite. When selection is empty, the gate falls back to the full suite
    # (safe-by-construction). Default True enables verify-by-default.
    scoped_verification: bool = True
    context_variant: str = "v7"
    stream_callback: Optional[Callable[[str, dict[str, Any]], None]] = None
    # Whether the stream_callback handles "content" events (incremental token rendering).
    # False in CLI mode (_ProgressPrinter has no "content" handler — all SSE overhead
    # would be discarded). Skips token_callback lambda creation + SSE streaming overhead.
    consume_content_events: bool = True
    # Lint maximum issue count
    max_lint_issues: int = 50          # Maximum number of issues to return from lint results
    # TDD auto-feedback loop
    auto_test_on_patch: bool = False   # Automatically run pytest after patch apply
    max_tdd_cycles: int = 3            # Maximum retry count after consecutive failures
    test_paths: list[str] = field(default_factory=list)  # pytest paths/arguments
    # Planner-Executor
    planning_enabled: bool = False     # Enable pre-execution planning phase
    # Self-Review
    self_review_enabled: bool = False  # Enable post-execution self-review phase
    max_review_turns: int = 3          # Maximum review turns for self-review corrections
    #RAG: related file automatic provide
    rag_enabled: bool = True           # Auto-inject related file Top-K at session start
    rag_top_k: int = 5                 # Number of files to auto-provide
    rag_force_injection: bool = False  # Always force-inject RAG context (default False for Cursor/Windsurf-style lazy loading)
    # Human-in-the-Loop approval gate
    # Callable: (tool_name, args, preview_text) -> bool (True=proceed)
    approval_callback: Optional[Callable[["str", "dict[str,Any]", "str"], bool]] = None

    # User Checkpoint: LLM asks user for questions/confirmations
    # Callable: (question_data: dict) -> dict  ({"status": "answered"|"timeout", "answer": ...})
    user_checkpoint_enabled: bool = True
    user_checkpoint_max_questions: int = 3       # Max questions per session
    user_checkpoint_timeout: int = ASK_USER_DEFAULT_TIMEOUT            # Timeout (seconds); on expiry, proceeds autonomously with default
    user_checkpoint_callback: Optional[Callable[["dict[str,Any]"], "dict[str,Any]"]] = None
    _user_checkpoint_count: int = 0              # Runtime: current session question counter

    # Server-side cancellation support (set by /agent/cancel)
    cancel_event: Optional["threading.Event"] = None

    # Mid-task user message injection (set by /agent/message/{session_id})
    # Queue items: str (user message text)
    message_queue: Optional[Any] = None

    # Context sliding window: keep only this many recent non-system messages
    # (0 = disabled, keep all). Prevents token overflow on long runs.
    context_window_size: int = 60

    # Multi-agent fields
    agent_id: str = "main"
    file_lock_manager: Optional[Any] = None  # FileLockManager instance

    # Auto-observation: after a successful patch, optionally inject the diff as a user observation.
    # Default False to keep tool dispatch predictable in tests and avoid extra tool calls.
    auto_observation_enabled: bool = False

    # Parallel tool execution: run independent tools concurrently
    parallel_tool_execution_enabled: bool = True
    max_parallel_workers: int = 5

    model_name: str = ""
    # ── Helper configuration (canonical) ─────────────────────────────────────
    # Helper is NOT a lane. Helper is the delegate_to_helper tool capability.
    helper_enabled: bool = False
    helper_model: str = ""           # Any model identifier (API or Ollama)
    helper_max_calls: int = 5        # Max delegation calls per session
    helper_ollama_url: str = "http://127.0.0.1:11434"  # Used when helper_model is Ollama

    # Task Router configuration
    task_router_enabled: bool = True
    route_decision: Optional[Any] = None

    # ── Bench / Experiment ────────────────────────────────────────────────────
    # Force a specific patch strategy for all modify_symbol ops in this session.
    # Values: "" (auto), "surgical_edit", "replace_symbol_body", "ast_op"
    # Used by patch_strategy_bench for multi-strategy comparison runs.
    force_patch_mode: str = ""
    # Exploration rate for online bandit learning (0.0 = pure exploit, 1.0 = pure random).
    # When triggered, a random strategy is tried instead of the policy-recommended one.
    patch_exploration_rate: float = 0.0

    # Learning system for tool recommendation
    learning_enabled: bool = False

    # Vector cache for semantic search
    vector_cache_enabled: bool = True

    # Ollama reasoning / thinking toggle
    thinking_mode: bool = False
    # Reasoning depth for providers that support it ("high" | "max"); None = provider default
    reasoning_effort: Optional[str] = None

    # Turn reduction optimizations
    dynamic_turn_budget_enabled: bool = True  # Dynamically adjust turn budget based on progress

    prefer_fused_tools: bool = True  # Prefer fused tools over sequential tool calls

    # Tool result cache: reuse results of read-only tools (safe invalidation on writes)
    tool_result_cache_enabled: bool = True
    tool_result_cache_ttl: int = 120  # seconds
    tool_result_cache_max_entries: int = 256

    # Debug / observability flags
    debug_sse: bool = False
    debug_context: bool = False
    debug_messages: bool = False
    debug_route: bool = False
    debug_retries: bool = False

    # Action memory: prevent duplicate tool calls
    action_memory_enabled: bool = False
    action_memory_max_entries: int = 64

    # Planning progress tracking
    plan_tracking_enabled: bool = False

    # State-aware tool result delta context
    delta_observation_enabled: bool = False

    # Workspace state memory for recent context
    workspace_state_enabled: bool = False

    # Tolerant patch mode: relaxed patch application for small/local models
    # When True: try whitespace-insensitive apply, fuzzy context re-anchoring,
    # and automatic edit_blocks fallback on repeated failures.
    # Set to True automatically when model_name matches small-model patterns.
    tolerant_patch_mode: bool = False
    # Max apply_patch failures before attempting edit_blocks auto-conversion
    tolerant_patch_max_failures: int = 2
    # Set True when this agent is a sub-agent in multi-agent mode.
    # Disables small-model complexity gating (orchestrator already scoped the task).
    is_subagent: bool = False

    # Planner model: primary LLM client/model (merged UI: Planner IS the main model).
    # Always set to user's selected planner. Cross-provider: separate client per provider.
    planner_llm_client: Optional[Any] = None
    planner_model: str = ""

    # Post-read replan: max operations allowed in replan_from_phase1_results.
    # read_symbol ops are now allowed, so a slightly higher budget may be useful.
    post_read_max_ops: int = 6

    # Developer model: separate LLM for OperationExecutor edit instruction generation
    developer_llm_client: Optional[Any] = None
    developer_model: str = ""


    # Self-planning: PlannerAgent critique + refine loop (2 extra LLM calls)
    self_planning_enabled: bool = True

    # Candidate selection: generate multiple plan candidates and pick the best one
    candidate_selection_enabled: bool = True

    # Design chat mode: write tools disabled, no early-finish, LLM synthesizes full response
    design_chat_mode: bool = False

    # Pre-built spec from design-chat analysis (implementation_spec).
    # When set, _run_planner_lane skips SpecResolver and uses this spec directly.
    prebuilt_spec_for_planner: Optional[Any] = None

    # ── Token continuity (cross-phase) ──────────────────────────────────────
    # Token offset from prior phases (e.g., Design Chat, main agent loop).
    # Applied by _run_planner_lane to PlannerAgent so token_usage events
    # show cumulative totals across phase boundaries. Cleared after apply.
    _token_offset_prompt_tokens: int = 0
    _token_offset_completion_tokens: int = 0
    _token_offset_cache_read_tokens: int = 0

    # Phase 7.1: Conversation layer integration (opt-in)
    # When True, routes requests through ConversationRouter → DesignStateManager →
    # FreezeManager → HandoffManager before SpecResolver.
    conversation_layer_enabled: bool = False

    # ── Context budget management ─────────────────────────────────────────
    context_budget_enabled: bool = True
    context_budget_reserve_output: int = 4096
    # ── Agent profile (custom tool/model/turn constraints) ────────────────
    # Load via: AgentProfile.load(name, repo_root) or load_profile(name, repo_root)
    # None = no profile, all defaults apply.
    agent_profile: Optional[Any] = None  # AgentProfile instance
    semantic_fit_divergence_threshold: float = 0.70
    planner_fallthrough_enabled: bool = True
    # Phase 7.2: When True, clarification_needed from planner
    # is re-routed to Design Chat instead of returning raw AgentResult.
    # Design Chat refines implementation_spec with user help → re-runs planner.
    design_chat_reroute_enabled: bool = True

    def __post_init__(self) -> None:
        """Validate and clamp default value ranges. Applies the same constraints on both server and CLI."""
        self.max_turns = max(1, self.max_turns)
        self.max_apply_attempts = max(1, min(self.max_apply_attempts, 10))
        self.max_lint_issues = max(1, min(self.max_lint_issues, 500))
        self.max_tdd_cycles = max(1, min(self.max_tdd_cycles, 10))
        self.max_review_turns = max(1, min(self.max_review_turns, 5))
        self.rag_top_k = max(1, min(self.rag_top_k, 15))
        self.patch_exploration_rate = max(0.0, min(1.0, self.patch_exploration_rate))
        self.helper_max_calls = max(1, self.helper_max_calls)
        self.context_window_size = max(10, self.context_window_size)

    def make_token_callback(self) -> Optional[Callable[[Optional[str]], None]]:
        """Return a gated, None-safe token callback for content streaming.

        Returns ``None`` when streaming is disabled or ``consume_content_events``
        is ``False`` (CLI mode — ``_ProgressPrinter`` has no ``"content"``
        handler, so all SSE overhead would be discarded).

        The returned callable:
        - Forwards ``text`` as a ``"content"`` event via ``stream_callback``
          (or ``None`` to signal a stream reset sentinel).
        - Guards against ``None`` text to prevent sending ``{"text": None}``
          events to the frontend.
        """
        cb = self.stream_callback
        if cb is None or not self.consume_content_events:
            return None
        # Capture cb in closure; guard None text (reset sentinel)
        def _token_cb(text: Optional[str]) -> None:
            if text is not None:
                cb("content", {"text": text})
        return _token_cb


# ── [REMOVED] is_small_model / model prefix lists ─────────────────────────────
# Model-name-based restrictions have been removed. All models are treated equally.


@dataclass
class ToolResult:
    ok: bool
    content: str = ""
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    partial_failure: bool = False  # True if operation partially succeeded
    retryable: bool = True  # True if operation can be retried
    retry_count: int = 0  # Number of retry attempts made


def _ensure_asicode_gitignored(repo_root: str) -> None:
    """Add .asicode/ to .gitignore if not already present.

    Standalone module-level function — called by ToolRegistry instance method
    and AgentLoop static method to avoid code duplication.
    """
    gitignore_path = os.path.join(repo_root, ".gitignore")
    entry = ".asicode/"
    try:
        if os.path.isfile(gitignore_path):
            with open(gitignore_path, encoding="utf-8") as f:
                content = f.read()
            if entry in content:
                return
            with open(gitignore_path, "a", encoding="utf-8") as fh:
                if content and not content.endswith("\n"):
                    fh.write("\n")
                fh.write(f"{entry}\n")
        else:
            with open(gitignore_path, "w", encoding="utf-8") as fh:
                fh.write(f"{entry}\n")
        logger.debug("Added %s to .gitignore", entry)
    except Exception as e:
        logger.warning("Could not update .gitignore: %s", e)


class ToolRegistry(
    ReadToolsMixin,
    WriteToolsMixin,
    AnalysisToolsMixin,
    ShellToolsMixin,
    TestToolsMixin,
    AgentToolsMixin,
    WebSearchToolsMixin,
    BrowserActionToolsMixin,
):
    """
    Dispatches tool calls from the agent LLM.

    Security:
    - bash/shell_exec: only within repo_root (path validation)
    - apply_patch: git apply --check must pass first
    - write_plan: uses plan_compiler path validation

    Tool handler methods are organized into category mixins in
    external_llm/agent/tool_handlers/:
      ReadToolsMixin    — find_symbol, find_references, find_relevant_files, etc.
      WriteToolsMixin   — write_plan, apply_patch, edit_ast
      AnalysisToolsMixin — get_project_info, explore_and_edit, etc.
      ShellToolsMixin     — shell_exec (bash)
      TestToolsMixin    — run_tests, run_lint
      AgentToolsMixin   — update_memory, delegate_to_helper
      WebSearchToolsMixin — search_web (DuckDuckGo/Brave/SearXNG)
    """

    # Directories pruned when counting source files for language detection.
    # These never indicate the repo's primary language (deps, caches, build
    # output, VCS metadata) and walking them only distorts counts + wastes time.
    _COUNT_SKIP_DIRS = frozenset({
        ".git", ".hg", ".svn",
        "node_modules", "bower_components",
        "vendor", ".venv", "venv", "env", ".env",
        "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
        "dist", "build", "target", "out", ".next", ".gradle",
        ".idea", ".vscode",
    })

    # Module-level cache for _detect_repo_language. The repo's language
    # composition is immutable during a run, so caching by repo_root avoids
    # the full os.walk (~259-452ms/call, measured on this repo) on every
    # ToolRegistry construction (IPC worker creates one per task).
    _LANGUAGE_DETECTION_CACHE: ClassVar["dict[str, Optional[LanguageId]]"] = {}

    # Single source of truth: tool name → handler method name mapping.
    # Used by dispatch() to resolve handlers and by has_tool_handler() for
    # handler-existence checks (e.g. MCP adapter). Keeping method names as
    # strings avoids class-level self-capture issues.
    _TOOL_HANDLER_MAP: ClassVar[dict[str, str]] = {
        # internal only — dispatched via delegate_to_helper, no direct LLM exposure
        "edit_file": "_tool_edit_file",
        "edit_text": "_tool_edit_text",
        "write_plan": "_tool_write_plan",
        "apply_patch": "_tool_apply_patch",
        "modify_symbol": "_tool_modify_symbol",
        # removed from schemas (bash equivalents); kept for backward compat
        "run_tests": "_tool_run_tests",
        "run_lint": "_tool_run_lint",
        "get_project_info": "_tool_get_project_info",
        "bash": "_tool_shell_exec",           # handler method != tool name
        "job": "_tool_job",
        "find_symbol": "_tool_find_symbol",
        "grep": "_tool_grep",
        "read_file": "_tool_read_file",
        "read_symbol": "_tool_read_symbol",
        "find_references": "_tool_find_references",
        "find_relevant_files": "_tool_find_relevant_files",
        # internal only — not exposed to LLM via AGENT_TOOL_SCHEMAS
        "update_memory": "_tool_update_memory",
        "update_plan": "_tool_update_plan",
        "query_dependency_graph": "_tool_query_dependency_graph",
        "get_file_outline": "_tool_get_file_outline",
        "analyze_change_impact": "_tool_analyze_change_impact",
        "run_structural_scan": "_tool_run_structural_scan",
        "edit_ast": "_tool_edit_ast",
        "anchor_edit": "_tool_anchor_edit",
        # internal only — not exposed to LLM directly
        "delegate_to_helper": "_tool_delegate_to_helper",
        "delegate_to_local_model": "_tool_delegate_to_helper",
        "ask_user": "_tool_ask_user",
        "query_experience": "_tool_query_experience",
        "search_web": "_tool_search_web",
        "web_fetch": "_tool_web_fetch",
        "browser_action": "_tool_browser_action",
        "read_image": "_tool_read_image",
    }

    @staticmethod
    def _detect_repo_language(repo_root: str) -> Optional[LanguageId]:
        """Detect the dominant code language of a repo by counting source files.

        Returns ``None`` (all tools visible) when:

        * the repo contains **any** Python files — intentionally conservative to
          avoid self-masking Python-only tools (e.g. a Python repo that also
          carries a root ``package.json`` for tooling), or
        * no recognized code files are found.

        Otherwise returns the LanguageId of the dominant non-Python family,
        using ``_LANGUAGE_EXTENSION_GROUPS`` as the single source of truth so
        that Java (``.java``) and Kotlin (``.kt``/``.kts``) are disambiguated by
        file count rather than by ambiguous build files like ``build.gradle``.

        Results are cached in ``_LANGUAGE_DETECTION_CACHE`` (module-level, keyed
        by resolved path) — the repo's language composition is immutable during
        a run, and the os.walk dominates ToolRegistry construction cost.
        """
        _norm = os.path.normpath(repo_root)
        _cached = ToolRegistry._LANGUAGE_DETECTION_CACHE.get(_norm)
        if _cached is not None or _norm in ToolRegistry._LANGUAGE_DETECTION_CACHE:
            return _cached
        from ..languages.models import _EXT_MAP, _LANGUAGE_EXTENSION_GROUPS

        # Guard: if repo_root is not a git repository, skip language detection.
        # Walking a non-repo directory (e.g. user's home directory ~) would be
        # prohibitively slow and yield no useful signal.
        _dot_git = Path(repo_root) / ".git"
        if not _dot_git.is_dir():
            try:
                _result = subprocess.run(
                    ["git", "rev-parse", "--show-toplevel"],
                    capture_output=True, text=True, timeout=5,
                    cwd=repo_root,
                )
                if _result.returncode != 0:
                    ToolRegistry._LANGUAGE_DETECTION_CACHE[_norm] = None
                    return None
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                ToolRegistry._LANGUAGE_DETECTION_CACHE[_norm] = None
                return None

        # Restrict counting to the "callable family" extensions (excludes
        # JSON/CSS/HTML/data formats which would distort the dominant signal).
        family_exts: set[str] = set()
        for _g in _LANGUAGE_EXTENSION_GROUPS:
            family_exts |= _g

        counts: dict[str, int] = {}  # LanguageId name -> file count
        for _root, dirs, files in os.walk(repo_root):
            dirs[:] = [d for d in dirs if d not in ToolRegistry._COUNT_SKIP_DIRS]
            for _f in files:
                ext = os.path.splitext(_f)[1].lower()
                if ext not in family_exts:
                    continue
                lang_name = _EXT_MAP.get(ext)
                if lang_name:
                    counts[lang_name] = counts.get(lang_name, 0) + 1

        # Python present at all → treat as Python/mixed → show all tools
        # (this is the conservative guard against self-masking).
        if counts.get("PYTHON", 0) > 0:
            ToolRegistry._LANGUAGE_DETECTION_CACHE[_norm] = None
            return None

        # Pure non-Python: pick the dominant family. Ties resolve to first
        # insertion order, which is harmless — every non-Python language masks
        # the same Python-only tool set.
        if not counts:
            ToolRegistry._LANGUAGE_DETECTION_CACHE[_norm] = None
            return None  # no recognized code files → safe default
        _result = LanguageId[max(counts, key=counts.get)]
        ToolRegistry._LANGUAGE_DETECTION_CACHE[_norm] = _result
        return _result

    def _make_result(self, **kwargs) -> "ToolResult":
        """Create a ToolResult without importing at module level."""
        return ToolResult(**kwargs)

    def __init__(self, repo_root: str, config: AgentConfig, local_assistant: Optional[Any] = None, agent_profile: Optional['AgentProfile'] = None):
        self.repo_root = str(Path(repo_root).resolve())
        self._repo_root_override: Optional[str] = None
        # Detect dominant code language by counting source files
        # (_LANGUAGE_EXTENSION_GROUPS — single source of truth). Used by
        # get_tool_schemas() to mask Python-only tools in pure non-Python repos.
        self._repo_language: Optional[LanguageId] = self._detect_repo_language(self.repo_root)
        self.config = config
        self._lint_runner = LintRunner(repo_root)
        self._symbol_searcher = SymbolSearcher(repo_root)
        self._rag_searcher = RAGSearcher(repo_root, vector_cache_enabled=config.vector_cache_enabled)
        _cgi = CallGraphIndexer(repo_root)  # lazy build on first access
        self._call_graph = RepositoryGraphFacade(call_graph_indexer=_cgi, repo_root=repo_root)

        # Argument repair layer
        self._arg_repairer = ArgumentRepairer()

        # Write safety manager (snapshot/verify/rollback + approval gating)
        self._safety_manager = WriteSafetyManager(self.repo_root)

        # Parallel execution support
        self.async_executor = None
        if config.parallel_tool_execution_enabled:
            from .async_tool_executor import AsyncToolExecutor
            self.async_executor = AsyncToolExecutor(self, max_workers=config.max_parallel_workers)
        self.dependency_graph = ToolDependencyGraph() if config.parallel_tool_execution_enabled else None
        # Failure prediction database
        # Collected patches from apply_patch calls (for result tracking)
        self._applied_patches: list[str] = []
        # Files written this session by text-editing tools (edit_text / modify_symbol /
        # edit_ast / anchor_edit). apply_patch consults this (Opt D) to refuse clobbering
        # a working-tree edit it cannot safely merge — see _tool_apply_patch guard.
        self._text_edited_files: set[str] = set()
        # File content cache with mtime validation and LRU eviction (global singleton)
        self._file_cache = get_global_file_cache()
        # Callbacks invoked after any successful write tool (apply_patch, write_file, etc.)
        # Used to propagate invalidation to dependent caches (e.g. RepositoryGraph).
        self._write_success_callbacks: list = []
        # Scoped write filter (None = unrestricted); set via clone_with_filter()
        self._write_filter: Optional[ScopedToolFilter] = None
        # Tool result cache for read-only tools (TTL + LRU). Built via the shared
        # helper so __init__ and BOTH clone paths construct identical, ISOLATED
        # caches (never shared — concurrent in-process subagents would race on
        # the same LRU/TTL state).
        self._tool_result_cache = self._make_tool_result_cache(config)
        if self._tool_result_cache is not None:
            logger.info(
                f"Tool result cache initialized (max={config.tool_result_cache_max_entries}, "
                f"TTL={config.tool_result_cache_ttl}s)"
            )
        self._search_cache: dict[str, ToolResult] = {}

        # Local Assistant instance for delegating tasks to local LLMs
        self.local_assistant = local_assistant
        # Agent profile: explicit param takes precedence over config field
        self._agent_profile = agent_profile if agent_profile is not None else getattr(config, 'agent_profile', None)
        if self._agent_profile is not None:
            logger.debug("Active agent profile: %s", self._agent_profile.name)

    def add_write_success_callback(self, cb) -> None:
        """Register a callback to be invoked after any successful write tool."""
        self._write_success_callbacks.append(cb)

    @property
    def repo_language(self) -> Optional[LanguageId]:
        """Dominant repo language by source-file count, or None if Python/mixed/unknown.

        ``None`` means all tools are visible (Python-only tools not masked). A
        concrete non-Python LanguageId means the repo is a pure non-Python repo
        and Python-only tools (``edit_ast``, ``run_structural_scan``) are hidden.
        """
        return self._repo_language

    @staticmethod
    def _make_tool_result_cache(config: "AgentConfig") -> Optional[Any]:
        """Build a fresh, ISOLATED ToolResultCache from ``config`` (or None).

        Shared by ``__init__`` and both clone paths (``clone_for_subagent``,
        ``clone_with_filter``) so every registry instance that opts in gets its
        OWN cache. Sharing a single cache across the parent and concurrent
        in-process subagents would let their LRU/TTL state race (one subagent's
        read evicts another's entry); nulling it (the previous clone behavior)
        threw away the most common subagent win — repeated ``read_file`` of the
        same path. A fresh per-clone cache keeps isolation while restoring that
        caching, and stays compatible with path-scoped invalidation (each cache
        invalidates only against its own writes).

        Each cache registers with the global metrics collector, which aggregates
        hit/miss/size stats across every live cache (parent + clones) in
        ``performance_metrics.get_summary()`` — see
        ``PerformanceCollector.register_tool_result_cache``.
        """
        if not getattr(config, "tool_result_cache_enabled", False):
            return None
        try:
            from .tool_result_cache import ToolResultCache
            cache = ToolResultCache(
                max_entries=getattr(config, "tool_result_cache_max_entries", 256),
                default_ttl=getattr(config, "tool_result_cache_ttl", 120),
            )
            get_global_collector().register_tool_result_cache(cache)
            return cache
        except Exception as e:
            logger.warning(f"Failed to initialize tool result cache: {e}")
            return None

    def clone_for_subagent(self, sub_config: "AgentConfig") -> "ToolRegistry":
        """Create a lightweight clone sharing expensive resources.

        Shared (immutable/thread-safe): SymbolSearcher, RAGSearcher,
        CallGraphIndexer, file_cache, LintRunner.
        Fresh (per-subagent mutable state): _applied_patches,
        _search_cache, config, tool_chain/async/watcher (disabled for subagents).
        """
        clone = object.__new__(ToolRegistry)
        clone.repo_root = self.repo_root
        clone._repo_language = self._repo_language
        clone.config = sub_config

        # Share expensive, thread-safe resources
        clone._lint_runner = self._lint_runner
        clone._symbol_searcher = self._symbol_searcher
        clone._rag_searcher = self._rag_searcher
        clone._call_graph = self._call_graph
        clone._file_cache = self._file_cache
        clone._arg_repairer = self._arg_repairer
        clone._safety_manager = self._safety_manager

        # Fresh mutable state per subagent
        clone._applied_patches: list[str] = []
        clone._search_cache: dict[str, Any] = {}

        # Subagents don't need these expensive resources
        clone.async_executor = None
        clone.dependency_graph = None
        # Fresh, ISOLATED cache (NOT shared with the parent, NOT None). A null
        # cache threw away the most common subagent win — repeated read_file of
        # the same path. Each clone gets its own cache via the shared helper so
        # concurrent subagents don't race on LRU/TTL state.
        clone._tool_result_cache = self._make_tool_result_cache(sub_config)
        clone.local_assistant = None
        clone._write_filter = None

        # Fresh callback list — subagents should not inherit parent callbacks
        clone._write_success_callbacks: list = []

        # Copy override state (if any); __init__ is bypassed via object.__new__
        clone._repo_root_override = getattr(self, "_repo_root_override", None)

        # Fresh mutable state, ISOLATED from the parent (NOT shared). In-process
        # subagents run concurrently via ThreadPoolExecutor (_run_parallel_batch),
        # so sharing _text_edited_files would cross-contaminate each clone's
        # apply_patch session-edit gate (a file one subagent edited via edit_text
        # would make a *different* concurrent subagent's apply_patch refuse the
        # same path). The parent's own edits are tracked separately; subagents
        # operate on disjoint assigned_files with file-level locking. Verified by
        # test_clone_for_subagent_sets_text_edited_files (must be a fresh set,
        # not the parent's object).
        clone._text_edited_files = set()
        clone._agent_profile = getattr(self, "_agent_profile", None)

        return clone

    def clone_with_filter(self, write_filter: "ScopedToolFilter") -> "ToolRegistry":
        """Create a lightweight clone sharing expensive resources but with fresh
        mutable state (read cache, search cache) and a write filter applied.

        Used for scoped delegation — restricts file write access.
        """
        clone = object.__new__(ToolRegistry)
        # Share thread-safe resources
        clone.repo_root = self.repo_root
        clone.config = self.config
        clone._lint_runner = getattr(self, "_lint_runner", None)
        clone._symbol_searcher = getattr(self, "_symbol_searcher", None)
        clone._rag_searcher = getattr(self, "_rag_searcher", None)
        clone._call_graph = getattr(self, "_call_graph", None)
        clone._file_cache = getattr(self, "_file_cache", None)
        # Safety/repair: shared (thread-safe, stateless-per-call)
        clone._arg_repairer = getattr(self, "_arg_repairer", None)
        clone._safety_manager = getattr(self, "_safety_manager", None)
        # Staging override (usually None; share as-is)
        clone._repo_root_override = getattr(self, "_repo_root_override", None)
        # Agent profile — shared read-only
        clone._agent_profile = getattr(self, "_agent_profile", None)
        # Parallel execution graph not used in filtered clones
        clone.dependency_graph = None
        # Fresh mutable state — not shared with original
        clone._search_cache = {}
        clone._applied_patches = []
        clone._text_edited_files = set()
        clone._write_success_callbacks = []
        # Apply the write filter
        clone._write_filter = write_filter
        # Shared convenience — subagents don't need these
        clone.async_executor = getattr(self, "async_executor", None)
        clone.local_assistant = getattr(self, "local_assistant", None)
        # Fresh, ISOLATED cache (NOT shared with the parent, NOT None). A null
        # cache meant scoped delegation / sub-agent execution had zero read
        # caching; a fresh cache keeps isolation while caching repeated reads.
        # Filtering only changes the write whitelist, so there is no staleness
        # concern (the cache starts empty — parent-cached results are not
        # inherited). Stays compatible with path-scoped invalidation.
        clone._tool_result_cache = self._make_tool_result_cache(self.config)
        return clone

    @property
    def write_filter(self) -> Optional["ScopedToolFilter"]:
        """Current write filter (None if unrestricted)."""
        return self._write_filter

    def _invalidate_cache_after_write(self, touched_paths: list[str]) -> None:
        """Invalidate file cache, call graph, RAG, and graph caches for touched paths (called after patch apply)."""
        import os
        for p in touched_paths:
            norm = p.strip().lstrip("/")
            abs_path = os.path.join(self.repo_root, norm)
            self._file_cache.invalidate(abs_path)

        # Invalidate call graph index if any supported language file was touched
        from ..languages import LanguageId as _LId
        if any(_LId.from_path(p) != _LId.UNKNOWN for p in touched_paths) and hasattr(self, '_call_graph'):
            cgi = getattr(self._call_graph, 'call_graph_indexer', None)
            if cgi is not None:
                cgi.invalidate()

        # Incrementally update RAG index for touched files (much faster than full rebuild)
        if hasattr(self, '_rag_searcher') and self._rag_searcher:
            try:
                self._rag_searcher.invalidate_files(touched_paths)
            except Exception as e:
                logger.debug(f"Failed to incrementally update RAG index: {e}")

        # Incrementally update GSG graph for touched Python files
        if hasattr(self, '_call_graph') and self._call_graph:
            try:
                self._call_graph.invalidate_files(touched_paths)
            except (AttributeError, TypeError):
                pass

        # Invalidate per-root file-walk caches so newly created files are
        # immediately visible to find_symbol / call-graph rebuilds.
        try:
            from external_llm.agent._shared_utils import _PY_WALK_CACHE, _TS_WALK_CACHE
            for _walk_cache in (_PY_WALK_CACHE, _TS_WALK_CACHE):
                _walk_cache.pop(self.repo_root, None)
        except Exception:
            pass  # non-critical — never block execution

        # Invalidate run-scoped graph cache for touched files
        try:
            from external_llm.graph.run_scoped_graph_cache import get_global_graph_cache
            graph_cache = get_global_graph_cache()
            graph_cache.invalidate_for_files(touched_paths)
        except Exception:
            pass  # non-critical — never block execution

    def _ensure_asicode_gitignored(self) -> None:
        """Add .asicode/ to .gitignore if not already present.

        Delegates to the module-level :func:`_ensure_asicode_gitignored` to
        share implementation with :class:`AgentLoop`.
        """
        _ensure_asicode_gitignored(self.repo_root)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Approval gate (delegated to WriteSafetyManager)
    # ------------------------------------------------------------------

    _PATCH_FILE_THRESHOLD = 3

    def _gate_check(self, tool_name: str, args: dict) -> "Optional[ToolResult]":
        rejection = self._safety_manager.gate_check(
            tool_name, args, self.config.approval_callback
        )
        if rejection is None:
            return None
        return ToolResult(
            ok=False, content="",
            error=rejection["error"],
            metadata=rejection["metadata"],
        )

    # Tools that write to files — need per-file locking in multi-agent mode
    # All write tools get snapshot + syntax verify + rollback safety wrapper
    _WRITE_TOOLS: ClassVar[set[str]] = {"apply_patch", "write_plan", "edit_ast", "edit_file", "edit_text", "modify_symbol", "anchor_edit"}
    # Tools that must NEVER run concurrently. ask_user blocks on human input and
    # relies on one-question-at-a-time invariants — a unique question_id
    # (millisecond timestamp) and an atomic question-count limit. Running two in
    # parallel collides the id and races the counter, and pairing one with read
    # tools blocks the whole batch on the slowest (human) response. Any batch
    # containing a serial tool falls back to sequential execution. NOT added to
    # _WRITE_TOOLS: that set drives file-locking, failure-logging and cache
    # invalidation, none of which apply to a user-facing question.
    # ask_user blocks on human input; job (kill) races with concurrent job output
    # on the same job_id. Both fall back to sequential execution when in a batch.
    _SERIAL_TOOLS = frozenset({"ask_user", "job"})
    # Read-only tools safe for result caching (no side effects, deterministic output)
    _READ_ONLY_TOOLS: ClassVar[set[str]] = {
        "get_project_info",
        "find_symbol", "find_references",
        "find_relevant_files",
        "get_file_outline",
        "analyze_change_impact",
        "query_dependency_graph",
        "run_structural_scan",
        # Experience query
        "query_experience",
        # Symbol / file read / search
        "read_symbol",
        "read_file",
        "grep",
        "read_image",
        # Web search/fetch — read-only network lookups; cached under the same TTL/LRU
        # as the others. Scope is None (network result), so a write-tool success drops
        # them conservatively, but repeated identical queries within a turn still hit.
        "search_web", "web_fetch",
    }

    # ── bash cache-invalidation heuristic ──────────────────────────────
    # Read-only bash commands (ls, cat, git log, grep, find, ...) never mutate
    # the filesystem, so they should NOT wipe the read-tool result cache.
    # Only commands that actually write/move/remove files or change source state
    # require cache invalidation. This keeps the cache effective across the very
    # common pattern of the model running `git status` / `grep` between edits.
    #
    # Static prefixes that are unconditionally read-only (whitelist, fastest path).
    _BASH_READONLY_PREFIXES = (
        "ls ", "cat ", "head ", "tail ", "less ", "more ",
        "grep ", "rg ", "find ", "fd ", "locate ",
        "wc ", "du ", "df ", "file ", "stat ", "md5sum ", "sha256sum ",
        "diff ", "git log", "git status", "git diff", "git show",
        "git rev-parse", "git remote -v", "git config --get",
        "git blame", "git ls-files", "git ls-tree", "git count-objects",
        "pwd", "whoami", "hostname", "uname", "echo ", "printf ",
        "which ", "command -v", "type ", "env", "printenv",
        "python3 -c ", "python -c ",  # introspection only when via -c (no pip/install)
        "node -e ", "node --check ",
        "pytest --collect-only", "pytest -q --co",
        "ruff check", "ruff --version",
        "test ", "[ ",
    )
    # Tokens whose presence ANYWHERE in the command implies a filesystem mutation
    # or source-state change → cache must be invalidated. Conservative: a false
    # positive only costs a cache miss, while a false negative serves stale data.
    #
    # NOTE: output redirection (``>``, ``>>``, ``2>``, ``&>`` …) is NOT detected
    # here by a fixed substring — ``"> "`` would miss the no-space form
    # ``>out.txt`` and serve stale cached data. It is detected separately by
    # ``_has_redirect_outside_quotes``, a quote-aware character scan that also
    # ignores redirect chars inside string literals (``echo ">x"``).
    _BASH_WRITE_TOKENS = (
        "tee ",      # writes stdin to a file
        "rm ", "rmdir", "mv ", "cp ",  # remove/move/copy files
        "mkdir", "touch ",            # create files/dirs
        "chmod", "chown",             # metadata changes
        "sed -i", "perl -i",          # in-place file edits
        " -delete", " -exec",         # find ... -delete / -exec: mutate despite "find " read-only prefix
        "git add", "git rm", "git mv", "git commit", "git checkout", "git reset",
        "git pull", "git push", "git merge", "git rebase",
        "git restore", "git clean",
        "pip install", "pip uninstall", "pip3 install", "pip3 uninstall",
        "npm install", "npm uninstall", "npm ci",
        "yarn add", "yarn remove", "pnpm add", "pnpm remove",
        "apply_patch", "patch -p",
        "curl -o", "wget -o", "wget -O",  # download to file
        "tar -x", "unzip ", "gunzip ",    # extract (creates files)
    )

    # `git branch` argument forms that only *query* branch state (no create/
    # delete/rename). Any other suffix (bare create-a-new-branch, -D/-d, -m/-M,
    # …) mutates. Checked as a dedicated case because a blanket "git branch"
    # prefix would also whitelist `git branch -D x` (deletes a branch).
    _GIT_BRANCH_READONLY_ARGS = (
        "--list", "-l", "-a", "-r", "-v", "-vv",
        "--contains", "--no-color", "--color", "--sort",
    )

    @staticmethod
    def _has_redirect_outside_quotes(command: str) -> bool:
        """True if ``command`` contains an output redirection operator that is
        OUTSIDE any single/double-quoted region.

        Covers ``>``, ``>>``, ``2>``, ``2>>``, ``&>`` — any ``>`` that is not
        inside a string literal is treated as a redirection (digit/ampersand
        prefixes like ``2>``/``&>`` still carry a ``>`` char and are caught by
        the same scan). A redirection writes/appends/truncates a file, so it is
        ALWAYS a cache-invalidating mutation regardless of the surrounding
        command.

        Why a character scan instead of a fixed substring token: the token
        ``"> "`` only matches when a space follows ``>``, so the common
        no-space form ``echo x >out.txt`` (and ``cmd 2>err``) escapes detection,
        gets classified read-only by the ``"echo "`` prefix, and serves stale
        cached data — exactly the false negative this classifier's own contract
        calls "worse than a miss". Tracking quote state lets us skip a ``>``
        that is part of a string literal (``echo "a>b"``) while still catching
        every real redirect.

        Conservative direction: a ``>`` used as a shell comparison operator
        (``[[ a > b ]]``) or fd-merge (``2>&1``) also trips this, but those only
        cost a cache miss — never stale data.

        This is the FALLBACK path used by :meth:`_bash_command_mutates_files`
        when tree-sitter-bash is unavailable or the command does not parse. The
        tree-sitter path (:meth:`_has_file_redirect_via_ts`) resolves the
        ``2>&1`` false positive exactly (fd-dup vs file), so it is preferred
        whenever the grammar is present.
        """
        quote = None  # currently-open quote char, or None
        i = 0
        n = len(command)
        while i < n:
            c = command[i]
            if quote is not None:
                if c == quote:
                    quote = None
                i += 1
                continue
            # Outside any quote:
            if c == "'" or c == '"':
                quote = c
            elif c == "\\":
                i += 1  # skip the escaped char (loop's i += 1 handles the 2nd)
            elif c == ">":
                return True
            i += 1
        return False

    @staticmethod
    def _redirect_is_fd_dup(redirect_text: str) -> bool:
        """True if a tree-sitter-bash ``file_redirect`` node body is an fd
        duplication/closure (``n>&m``, ``>&m``, ``n>&-``) rather than a file
        write.

        tree-sitter-bash tags BOTH real file redirects (``> f``, ``2>err``,
        ``&>all``) and fd-dups (``2>&1``) as ``file_redirect`` nodes, so the node
        type alone cannot tell them apart. The distinguishing token is an ``&``
        immediately after the ``>`` whose target is a file-descriptor number (or
        ``-`` for close) — never a path. Applied to a PARSED node body, quoting
        is already resolved by the grammar, so no quote tracking is needed here
        (unlike the raw-command scanner).
        """
        gt = redirect_text.find(">")
        if gt < 0:
            return False
        after = redirect_text[gt + 1:].lstrip()
        if not after.startswith("&"):
            return False
        rest = after[1:]
        return bool(rest) and (rest[0].isdigit() or rest[0] == "-")

    @classmethod
    def _has_file_redirect_via_ts(cls, command: str):
        """Detect a real file-writing redirection via tree-sitter-bash.

        Returns True iff *command* redirects stdout/stderr to a FILE (``>``,
        ``>>``, ``n>f``, ``&>f``) — truncating/appending/creating a file, hence
        always a cache-invalidating mutation. Returns False when the only
        redirections are fd duplications/closures (``2>&1``, ``>&-``) which
        touch no file. Returns None when tree-sitter-bash is unavailable or
        *command* does not parse cleanly — the caller falls back to the
        conservative quote-aware scanner (:meth:`_has_redirect_outside_quotes`).

        Why structural: the raw text scanner treats every ``>`` (including
        ``2>&1``) as a redirect, forcing a cache miss on the extremely common
        read-only ``cmd 2>&1 | head``. tree-sitter exposes the redirect nodes
        directly and the fd-dup vs file distinction is decided on the parsed
        node body (:meth:`_redirect_is_fd_dup`), with no quote tracking.
        """
        try:
            from ..languages import tree_sitter_utils as _ts_utils

            if not _ts_utils.is_available():
                return None
            _parser = _ts_utils.get_parser("bash")
        except Exception:
            return None
        if _parser is None:
            return None
        try:
            _tree = _parser.parse(bytes(command, "utf8"))
        except Exception:
            return None
        if _tree is None or _tree.root_node.has_error:
            return None
        _stack = [_tree.root_node]
        while _stack:
            _node = _stack.pop()
            if _node.type == "file_redirect":
                if not cls._redirect_is_fd_dup(
                    command[_node.start_byte:_node.end_byte]
                ):
                    return True
            # Always descend — a redirect may be nested inside command
            # substitution / a subshell / a loop body.
            if _node.children:
                _stack.extend(reversed(_node.children))
        return False

    @classmethod
    def _bash_segment_is_readonly(cls, segment: str) -> bool:
        """Is this single command (no |/&&/;/|| — already split, no write
        token anywhere in the full command) read-only?

        Handles the `git stash` / `git branch` special cases (only specific
        query forms are read-only; everything else mutates) plus the generic
        prefix whitelist. Matches a whitelist prefix either as a proper
        "prefix + argument" (``"head foo"`` matches ``"head "``) or as the
        bare command with no arguments at all (``"head"`` matches ``"head "``
        too — the trailing space in the prefix table must not require a
        following argument to exist).
        """
        if segment == "git stash" or segment.startswith("git stash "):
            rest = segment[len("git stash"):].strip()
            return rest.startswith("list") or rest.startswith("show")
        if segment == "git branch" or segment.startswith("git branch "):
            rest = segment[len("git branch"):].strip()
            return rest == "" or any(rest.startswith(a) for a in cls._GIT_BRANCH_READONLY_ARGS)
        for prefix in cls._BASH_READONLY_PREFIXES:
            if segment.startswith(prefix) or segment == prefix.rstrip():
                return True
        return False

    @classmethod
    def _bash_command_segments_via_ts(cls, command: str):
        """Structurally split *command* into its constituent command segments via
        tree-sitter-bash.

        Returns the text of every ``command`` node in the parse tree — including
        those nested inside command substitution (``$(...)`` / backticks),
        pipelines (``|``), lists (``&&`` / ``;`` / ``||``) and ``for``/``while``/
        ``if`` bodies — or ``None`` when tree-sitter-bash is unavailable or
        *command* does not parse cleanly (caller falls back to the conservative
        text splitter). Returns ``None`` too when no ``command`` node is found
        (bare comment / assignment) so the fallback classifies it.

        Why structural instead of a regex split on ``|``/``&&``/``;``: a regex
        cannot tell a separator inside a quoted string (``grep "a|b"``) from a
        real pipeline, and ``$(...)`` hides an arbitrary inner command — so the
        text path must bail out (invalidate) on both. tree-sitter resolves them
        exactly:

        - ``grep "foo|bar" f | head`` → ``['grep "foo|bar" f', 'head']``;
        - ``ls $(git stash pop)``     → ``['ls $(git stash pop)', 'git stash pop']``;
        - ``echo `date` ``            → ``['echo `date`', 'date']``.

        Output redirection is detected separately (quote-aware) by
        :meth:`_has_redirect_outside_quotes`, so this method only decomposes
        commands — it does not interpret redirects.
        """
        try:
            from ..languages import tree_sitter_utils as _ts_utils

            if not _ts_utils.is_available():
                return None
            _parser = _ts_utils.get_parser("bash")
        except Exception:
            return None
        if _parser is None:
            return None
        try:
            _tree = _parser.parse(bytes(command, "utf8"))
        except Exception:
            return None
        if _tree is None or _tree.root_node.has_error:
            return None

        _segments: list[str] = []
        _stack = [_tree.root_node]
        while _stack:
            _node = _stack.pop()
            if _node.type == "command":
                _segments.append(command[_node.start_byte:_node.end_byte])
            # Always descend: a command's arguments may contain command
            # substitution (``ls $(...)``), and pipelines/lists/loops contain
            # further command nodes that must each be classified individually.
            if _node.children:
                _stack.extend(reversed(_node.children))
        # No command node at all (bare comment / env-only assignment) → defer to
        # the fallback rather than treating it as "all read-only".
        if not _segments:
            return None
        return _segments

    @classmethod
    def _bash_command_mutates_files(cls, command: str) -> bool:
        """Does this bash command change filesystem / source state?

        Used to decide whether a successful ``bash`` tool call should invalidate
        the read-tool result cache. Read-only commands (``ls``, ``git status``,
        ``grep``, …) return False; anything that writes/moves/removes a file or
        changes git/source state returns True. Conservative: ambiguous commands
        default to True (invalidate) since a stale cache is worse than a miss.

        Two-stage classification:

        1. **Redirect + write-token scan** (runs first, on the WHOLE command) — a
           ``>``/``>>``/``2>`` redirection or any write token (``rm``, ``git add``,
           `` -exec`` …) anywhere → mutate, before any read-only classification
           can mask a mutating suffix/chain/substitution.

        2. **Per-segment read-only classification** — split the command into its
           constituent commands and require EVERY segment to be individually
           read-only. Splitting is done structurally via tree-sitter-bash
           (:meth:`_bash_command_segments_via_ts`) so that command substitution
           (``$(...)`` / backticks), a ``|`` inside quotes, and ``&&``/``;``/``||``
           lists / loop bodies are decomposed exactly instead of bailed out
           wholesale. When tree-sitter-bash is unavailable or the command does
           not parse, falls back to the conservative regex splitter (which still
           bails out on ``$(...)``/backticks and quoted pipelines).
        """
        if not command:
            return False
        stripped = command.strip()

        # 1. Redirect / write-token scan on the WHOLE command — unconditionally
        #    first, so a mutating suffix/chain/redirect is never masked by a
        #    read-only-looking prefix or subcommand earlier in the string.
        _ts_redirect = cls._has_file_redirect_via_ts(stripped)
        if _ts_redirect is None:
            # tree-sitter-bash unavailable / parse failed → conservative
            # quote-aware scan (treats fd-dups like 2>&1 as redirects too —
            # only a cache miss, never stale data).
            if cls._has_redirect_outside_quotes(stripped):
                return True
        elif _ts_redirect:
            return True
        for tok in cls._BASH_WRITE_TOKENS:
            if tok in stripped:
                return True

        # 2. Per-segment read-only classification. tree-sitter-bash yields
        #    correct segments; otherwise the conservative regex fallback (which
        #    bails out on $(...)/backticks and on quoted pipelines).
        segments = cls._bash_command_segments_via_ts(command)
        if segments is None:
            if "$(" in stripped or "`" in stripped:
                return True
            if any(sep in stripped for sep in ("|", "&&", ";", "||")):
                if "'" in stripped or '"' in stripped:
                    return True
                segments = re.split(r"\|\||&&|\||;", stripped)
            else:
                segments = [stripped]

        for segment in segments:
            segment = segment.strip()
            if segment and not cls._bash_segment_is_readonly(segment):
                return True
        return False

    def _tool_call_mutates(self, tool_name: str, args: dict) -> bool:
        """Single source of truth: does executing this tool call change
        filesystem / source / git state?

        Consumed by three call sites that MUST agree on "is this a mutating
        call?":
          - read-tool result cache invalidation (``dispatch``)
          - ``dispatch_parallel``'s parallel-vs-sequential gate
          - DesignChatLoop's read/write phase partition (``_is_mutating``)

        Write tools always mutate. ``bash`` mutates when its command writes,
        removes, moves or creates files or changes git/source state (per
        ``_bash_command_mutates_files`` — the conservative classifier also used
        for cache invalidation). Read-only bash (``ls``, ``git status``, ``grep``
        …) and all pure read tools return False, so they still parallelize.
        """
        if tool_name in self._WRITE_TOOLS:
            return True
        if tool_name == "bash":
            return self._bash_command_mutates_files((args or {}).get("command", ""))
        if tool_name == "job" and (args or {}).get("action") == "kill":
            return True  # kill mutates process state; can race with concurrent job output
        return False

    def _tool_call_is_serial(self, tool_name: str, args: dict) -> bool:
        """Must this call run strictly alone, never batched with other calls?

        Single source of truth for the ``_SERIAL_TOOLS`` gate, consumed by
        ``dispatch_parallel`` and DesignChatLoop's read/write/serial phase
        partition — mirrors ``_tool_call_mutates``'s role for the mutation gate.

        ``ask_user`` is unconditionally serial (see ``_SERIAL_TOOLS`` docstring).
        ``job`` is only serial for ``action == "kill"`` (races with concurrent
        job output on the same job_id); ``job list`` / ``job output`` are pure
        reads and must stay eligible for the parallel phase — treating every
        ``job`` call as serial regardless of action needlessly serializes read
        batches (and, if a killing call is also treated as mutating, forced
        double-placement in both the write and serial phase).
        """
        if tool_name == "ask_user":
            return True
        if tool_name == "job":
            return (args or {}).get("action") == "kill"
        return False

    # ── Path-scoped cache invalidation ───────────────────────────────────
    # Read-only tools whose result depends on exactly one file/dir path named
    # in a single arg. Used to tag cache entries so a later write only drops
    # overlapping entries instead of a full clear() (see _extract_write_target_paths).
    _PATH_SCOPED_READ_TOOLS = frozenset({"read_file", "get_file_outline", "read_image"})

    def _extract_read_scope_paths(self, tool_name: str, args: dict) -> Optional[frozenset]:
        """Absolute path(s) a cached read-only result depends on, or None if
        unknown/repo-wide (e.g. a search with no path filter, or any tool not
        listed below) — such entries are always dropped by a later invalidation."""
        args = args or {}
        if tool_name in self._PATH_SCOPED_READ_TOOLS:
            p = args.get("path")
            if isinstance(p, str) and p.strip():
                full = p if os.path.isabs(p) else os.path.join(self.repo_root, p)
                return frozenset({os.path.normpath(full)})
            return None
        if tool_name == "grep":
            p = args.get("path")
            p = p.strip() if isinstance(p, str) else ""
            if p and p not in (".", self.repo_root):
                full = p if os.path.isabs(p) else os.path.join(self.repo_root, p)
                return frozenset({os.path.normpath(full)})
            return None  # no path filter → repo-wide search, unknown scope
        return None

    def _extract_write_target_paths(self, tool_name: str, args: dict) -> Optional[frozenset]:
        """Best-effort absolute target path(s) for a write-tool call, so cache
        invalidation can drop only overlapping entries. Returns None when the
        target can't be determined (caller should fall back to a full clear()).

        Path-only (no file I/O) mirror of SafetyManager.snapshot_target_files's
        target-resolution logic (patch headers → plan ops → explicit file_path/
        path arg) — cheap enough to run for every write-tool call, including
        edit_text/edit_ast/anchor_edit which skip the I/O snapshot entirely.
        """
        args = args or {}
        targets: list = []
        if tool_name in ("apply_patch", "write_plan"):
            raw_plan = args.get("patch") or args.get("plan") or ""
            patch = raw_plan if isinstance(raw_plan, str) else ""
            for line in patch.splitlines():
                if line.startswith("--- a/") or line.startswith("--- b/"):
                    p = line[6:].strip()
                    if p and p != "/dev/null":
                        targets.append(p)
                elif line.startswith("+++ b/"):
                    p = line[6:].strip()
                    if p and p != "/dev/null":
                        targets.append(p)
            if not targets and isinstance(raw_plan, dict):
                plan_ops = raw_plan.get("ops") or raw_plan.get("operations") or []
                if not plan_ops and "path" in raw_plan:
                    plan_ops = [raw_plan]
                targets = [str(op["path"]) for op in plan_ops if op.get("path")]
            if not targets:
                explicit = args.get("file_path") or args.get("path") or ""
                if explicit and isinstance(explicit, str):
                    targets = [explicit]
        else:
            target = args.get("file_path") or args.get("path") or ""
            if target and isinstance(target, str):
                targets = [target]

        if not targets:
            return None  # unknown scope → caller falls back to full clear()

        return frozenset(
            os.path.normpath(t if os.path.isabs(t) else os.path.join(self.repo_root, t))
            for t in targets
        )

    # ── Write safety: snapshot + verify + rollback (delegated) ──────────

    def _snapshot_target_files(self, tool_name: str, args: dict) -> dict:
        """Capture file contents before a write operation."""
        return self._safety_manager.snapshot_target_files(tool_name, args)

    def _verify_after_write(self, snapshots: dict, _post_contents: dict | None = None) -> tuple[bool, str]:
        """Basic syntax check on files that were modified.

        Returns (True, "") or (False, "error detail").
        """
        return self._safety_manager.verify_after_write(snapshots, _post_contents=_post_contents)

    def _restore_snapshots(self, snapshots: dict) -> list[str]:
        """Restore files from pre-write snapshot. Returns list of failed paths."""
        return self._safety_manager.restore_snapshots(snapshots)

    def _repair_verify_failure(self, snapshots: dict) -> bool:
        """Attempt to repair argument mismatch errors before rollback.

        Called when verify_after_write fails. Tries to fix "not enough
        arguments" / "too many arguments" errors by adding/removing args
        at call sites. Only modifies files in *snapshots*.

        Returns True iff repair succeeded AND **every** file in *snapshots*
        re-verifies clean. The all-files contract is load-bearing: this
        method is only invoked when ``verify_after_write`` already found a
        break, and the dispatch caller trusts a True return as a final
        green light (it returns ``result`` without re-checking). Returning
        True after repairing just one file would leave the other files'
        syntax errors silently on disk — so a full re-verify gate is
        mandatory before claiming success.
        """
        import os as _os

        # Lazy imports to avoid circular deps
        # (vm subtree lives at _editor_core/vm after the editor_core repackaging)
        from external_llm.editor._editor_core.vm.failure_classifier import (
            FailureType,
            create_failure_classifier,
        )
        from external_llm.editor._editor_core.vm.repair_registry import RepairRegistry

        from ..languages import LanguageRegistry

        _repaired_any = False
        for path in snapshots:
            if not _os.path.isfile(path):
                continue

            provider = LanguageRegistry.instance().get(path)
            if not provider or not provider.capabilities().has_syntax_validator:
                continue

            # Read current (patched) content
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    current_code = f.read()
            except OSError:
                continue

            # Re-validate to get structured errors
            val = provider.validate_syntax(path, current_code)
            if val.ok:
                continue

            # Classify errors
            lang = provider.language_id().value
            try:
                classifier = create_failure_classifier(lang)
            except ValueError:
                continue

            # Convert SyntaxError_ → VerifyError
            from external_llm.editor._editor_core.vm.models import VerifyError
            verify_errors = [
                VerifyError(
                    message=e.message,
                    line=e.line,
                    column=e.col,
                )
                for e in (val.errors or [])
            ]

            # Try repair for each ARGUMENT_MISMATCH error
            registry = RepairRegistry(lang)
            for verr in verify_errors:
                ftype = classifier.classify([verr])
                if ftype != FailureType.ARGUMENT_MISMATCH:
                    continue

                strategy = registry.get(ftype)
                if strategy is None:
                    continue

                ops = strategy(current_code, verr, classifier)
                if ops is None:
                    continue

                # Apply repair
                if len(ops) == 1 and "__raw_code__" in ops[0].payload:
                    repaired_code = ops[0].payload["__raw_code__"]
                else:
                    continue  # Only raw replacements supported for now

                try:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(repaired_code)
                except OSError:
                    continue

                # Re-verify
                re_val = provider.validate_syntax(path, repaired_code)
                if re_val.ok:
                    current_code = repaired_code
                    _repaired_any = True
                    logger.info(
                        "Write safety: repaired argument mismatch in %s "
                        "(error: %s)", path, verr.message[:80],
                    )
                    # Code is now clean — stop processing this file
                    break
                else:
                    # Restore current_code on failure
                    try:
                        with open(path, "w", encoding="utf-8") as f:
                            f.write(current_code)
                    except OSError:
                        pass

        # ── All-files contract gate ───────────────────────────────────────
        # Per-file repair success is necessary but NOT sufficient: a True return
        # is the dispatch caller's final green light (no further verify), so we
        # must guarantee every snapshot file is clean. If even one file still
        # fails, the caller must fall through to rollback rather than ship a
        # partially-repaired multi-file write with a lingering syntax error.
        if _repaired_any:
            _repaired_any = self._verify_after_write(snapshots)[0]

        return _repaired_any

    @staticmethod
    def _should_soft_fail_verify(verify_detail: str, snapshots: dict) -> bool:
        """Classify a verify error: SYNTAX_ERROR = hard fail, other = soft fail.

        Non-syntax compilation errors (ARGUMENT_MISMATCH, TYPE_MISMATCH, etc.)
        may be resolved by downstream ops in a multi-op plan — preserve the
        intermediate changes instead of rolling back.

        Origin-skip guard: when the PRE-EDIT content of the edited file also
        fails isolated-compile, the verify errors are environmental cascade
        noise (missing deps/SDK — e.g. an Android ViewModel compiled without
        the SDK, or Kotlin without coroutines), NOT caused by this edit. We
        soft-fail so a correct edit is not rolled back against a broken
        baseline. This mirrors edit_text's ``_et_orig_ok`` gate ("we never
        block an edit fixing a pre-existing error") and is the root-cause
        general guard for the whole cascade-noise class. Only existing files
        have an origin to check; new-file snapshots (``_MISSING_SNAP``) skip.
        """
        from external_llm.editor._editor_core.vm.failure_classifier import FailureType, create_failure_classifier
        from external_llm.editor._editor_core.vm.models import VerifyError

        from ..languages import LanguageRegistry

        if not snapshots:
            return False

        # ── Parse the file path from verify_detail ──
        # verify_detail format (from verify_after_write):
        #   "file_path:line:col: message"
        _detail_path_match = re.match(r'^([^:]+):(\d+):(\d+): ', verify_detail)
        _detail_path = _detail_path_match.group(1) if _detail_path_match else None

        # Find a language provider + its pre-edit origin content for the error's file.
        _lang = None
        _provider = None
        _origin = None  # (path, content_str) of the snapshot file
        if _detail_path and _detail_path in snapshots:
            # Use the file that actually produced the error (Bug #1 fix)
            _path_provider = LanguageRegistry.instance().get(_detail_path)
            if _path_provider and _path_provider.capabilities().has_syntax_validator:
                _lang = _path_provider.language_id().value
                _provider = _path_provider
                _origin_content = snapshots[_detail_path]
                if isinstance(_origin_content, str):
                    _origin = (_detail_path, _origin_content)

        if not _lang:
            # Fallback: first snapshot file with a validator (pre-3.9 behaviour)
            for path, orig_content in snapshots.items():
                provider = LanguageRegistry.instance().get(path)
                if provider and provider.capabilities().has_syntax_validator:
                    _lang = provider.language_id().value
                    _provider = provider
                    if isinstance(orig_content, str):
                        _origin = (path, orig_content)
                    break

        if not _lang:
            return False

        # ── Origin-skip guard (mirrors edit_text's _et_orig_ok gate) ──
        if _origin is not None:
            _o_path, _o_content = _origin
            try:
                _orig_ok = _provider.validate_syntax(_o_path, _o_content).ok
            except Exception:
                _orig_ok = True  # validator crash → don't block the edit
            if not _orig_ok:
                logger.warning(
                    "Write safety: pre-edit content of %s also failed isolated "
                    "compile — verify errors are environmental cascade noise, "
                    "keeping edit (origin-skip guard): %s",
                    _o_path, verify_detail,
                )
                return True

        try:
            classifier = create_failure_classifier(_lang)
        except ValueError:
            return False

        ftype = classifier.classify([
            VerifyError(message=verify_detail, line=0, column=0),
        ])

        # SYNTAX_ERROR and UNKNOWN are hard failures — always rollback
        if ftype in (FailureType.SYNTAX_ERROR, FailureType.UNKNOWN):
            return False

        # All other recognizable errors (ARGUMENT_MISMATCH, TYPE_MISMATCH,
        # MISSING_RETURN, MISSING_VARIABLE, etc.) are cross-op fixable
        return True

    # _reindent_text → imported from external_llm.common.indent_utils.reindent_text

    @staticmethod
    def _auto_repair_indent(original_content: str, operations: list) -> Optional[str]:
        """Try to fix indentation in edit_file operations and re-apply them.

        Handles replace (re-indent content to anchor's column) and insert_after
        (re-indent content to anchor line's indentation level).  Skips insert_before
        and mid-line anchors where line-level correction is not meaningful.
        Returns the repaired file content, or None if no repair was needed or possible.
        """
        fixed = original_content

        for op in operations:
            op_type = op.get("type")
            anchor = op.get("anchor", "")
            content = op.get("content", "")

            if not anchor or not content or op_type not in ("replace", "insert_after"):
                continue

            idx = fixed.find(anchor)
            if idx < 0:
                return None

            if op_type == "replace":
                # Anchor's line position and leading whitespace on that line.
                line_start = fixed.rfind('\n', 0, idx) + 1
                leading_ws = fixed[line_start:idx]

                # Skip mid-line anchors — line-level indent fix doesn't apply when
                # there is non-whitespace before the anchor on its line.
                if leading_ws.strip():
                    continue

                anchor_col = len(leading_ws)
                adjusted = reindent_text(content, anchor_col)
                if adjusted is None:
                    continue

                # Consume the line's leading whitespace along with the anchor so the
                # indent baked into adjusted's first line replaces the existing
                # prefix instead of stacking on top of it.
                fixed = fixed[:line_start] + adjusted + fixed[idx + len(anchor):]

            else:  # insert_after
                # Get anchor line's indentation to use as target.
                line_start = fixed.rfind('\n', 0, idx) + 1
                nl_pos = fixed.find('\n', idx)
                no_trailing_nl = nl_pos == -1
                eol = len(fixed) if no_trailing_nl else nl_pos
                full_line = fixed[line_start:eol]
                anchor_indent = len(full_line) - len(full_line.lstrip())

                adjusted = reindent_text(content, anchor_indent)
                if adjusted is None:
                    continue

                # When the anchor's line has no trailing newline, insert one
                # before the new content so it doesn't get appended to the line.
                if no_trailing_nl:
                    fixed = fixed + '\n' + adjusted.rstrip('\n') + '\n'
                else:
                    fixed = fixed[:eol+1] + adjusted.rstrip('\n') + '\n' + fixed[eol+1:]

        return fixed if fixed != original_content else None

    def dispatch(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
        """Dispatch a tool call and return the result."""
        # Robust args handling for small models (7B/3B)
        if not isinstance(args, dict):
            args = {"path": str(args)}

        # Telemetry: greppable per-dispatch marker for tool-usage analysis.
        # Logged at entry so it counts the model's tool *selection* regardless of
        # downstream cache/gate outcome. File-only (root logger's file handler) —
        # never enters model context. Count later with:
        #   grep -rhoE "tool_dispatch: \w+" logs/ | sort | uniq -c | sort -rn
        logger.info("tool_dispatch: %s", tool_name)

        # Check for cancellation before any work
        if self.config.cancel_event and self.config.cancel_event.is_set():
            return ToolResult(
                ok=False,
                content="",
                error="Operation cancelled",
                execution_time=0.0,
                retryable=False,
            )

        # Agent profile tool access validation
        if hasattr(self, '_agent_profile') and self._agent_profile is not None:
            profile = self._agent_profile
            # blocked_tools takes precedence over allowed_tools
            if hasattr(profile, 'blocked_tools') and tool_name in profile.blocked_tools:
                return ToolResult(
                    ok=False,
                    content="",
                    error=f"Tool '{tool_name}' is blocked by agent profile '{profile.name}'",
                    execution_time=0.0,
                    metadata={"blocked": "agent_profile", "profile": profile.name}
                )
            # allowed_tools: empty list means no restriction (all tools allowed)
            if hasattr(profile, 'allowed_tools') and profile.allowed_tools and tool_name not in profile.allowed_tools:
                return ToolResult(
                    ok=False,
                    content="",
                    error=f"Tool '{tool_name}' not in allowed_tools for profile '{profile.name}'",
                    execution_time=0.0,
                    metadata={"blocked": "agent_profile", "profile": profile.name}
                )

        # Argument repair
        repair = self._arg_repairer.repair(tool_name, args)
        if repair.repaired:
            args = repair.repaired_args

        gate_result = self._gate_check(tool_name, args)
        if gate_result is not None:
            return gate_result


        # Tool result cache lookup (read-only tools only)
        cache_hit = False
        if (self._tool_result_cache is not None and
            tool_name in self._READ_ONLY_TOOLS):
            cached = self._tool_result_cache.get(tool_name, args)
            if cached is not None:
                # Reconstruct ToolResult from cached dict
                result = ToolResult(
                    ok=cached.get("ok", False),
                    content=cached.get("content", ""),
                    error=cached.get("error"),
                    # Defensive copy: ``cached`` is the dict stored in the cache
                    # entry (get() returns it by reference). Without a copy,
                    # ``result.metadata["cache_hit"] = True`` below would mutate
                    # the cache entry's own dict, permanently baking cache_hit
                    # into it AND letting any caller-side metadata addition
                    # leak back into the cache and propagate to later hits.
                    metadata=dict(cached.get("metadata") or {}),
                    execution_time=0.0,  # will be overwritten
                )
                result.metadata["cache_hit"] = True
                logger.debug("Tool result cache HIT: %s (cached) (args: %s)", tool_name, args)
                # Record performance with zero execution time
                get_global_collector().record_tool_call(tool_name, 0.0, True)
                return result

        # File lock manager + locked-paths holder. Acquisition is deferred to
        # inside the try below so the finally always releases whatever was
        # acquired — previously the snapshot ran between acquire and try, so a
        # raise there (or an unknown-tool early return) orphaned the locks in
        # self._held until session reset().
        flm = self.config.file_lock_manager
        locked_paths: list[str] = []

        method_name = self._TOOL_HANDLER_MAP.get(tool_name)
        if method_name is None:
            available = ", ".join(sorted(self._TOOL_HANDLER_MAP.keys()))
            return ToolResult(
                ok=False,
                content="",
                error=f"Unknown tool: {tool_name}. Available tools: [{available}]",
            )
        handler = getattr(self, method_name)

        # Snapshot target files before write operations for rollback on syntax error.
        # edit_text skips syntax validation (Claude Code-style simple replace) and has no
        # rollback path, so snapshotting is pure wasted I/O — skip it.
        # edit_ast already does compile() validation BEFORE writing, making snapshot+verify
        # redundant I/O for this self-validating tool — skip it too.
        # anchor_edit also validates syntax before writing (deterministic, self-validating).
        _write_snapshots: dict = {}
        try:
            # start_time FIRST so the except handler's execution_time reference is
            # always bound even if acquire/snapshot raise below.
            start_time = time.monotonic()
            # Acquire file locks for write operations INSIDE try so the finally
            # always releases them, even if snapshotting below raises.
            if flm is not None and tool_name in self._WRITE_TOOLS:
                locked_paths = flm.acquire_relevant(args)
            # Snapshot under the lock so the captured state is consistent w.r.t.
            # concurrent writers (restore-on-rollback must reflect pre-write content).
            if tool_name in self._WRITE_TOOLS and tool_name not in ("edit_text", "edit_ast", "anchor_edit"):
                _write_snapshots = self._snapshot_target_files(tool_name, args)
            # Snapshot _text_edited_files BEFORE the handler runs: non-excluded
            # write tools (e.g. modify_symbol) record the edited path in
            # _text_edited_files from INSIDE the handler (write_tools.py), which
            # is BEFORE dispatch's verify below. On a genuine rollback we must
            # also undo that recording, else a later apply_patch to the file is
            # wrongly refused with "already edited this session". Excluded tools
            # (edit_text/edit_ast/anchor_edit) never reach the rollback path
            # (no _write_snapshots), so their rollback-free recordings are safe.
            _pre_text_edits = set(self._text_edited_files)
            result = handler(args)
            result.execution_time = time.monotonic() - start_time

            # Safety check: verify Python syntax after write; rollback on failure
            # edit_text intentionally skips syntax validation (Claude Code-style simple replace)
            if _write_snapshots and result.ok and tool_name != "edit_text":
                _verify_ok, _verify_detail = self._verify_after_write(_write_snapshots)
                if not _verify_ok:
                    # --- Try auto-repair: indentation correction before rollback ---
                    _repair_ok = False
                    if tool_name == "edit_file" and _write_snapshots:
                        _orig_content = next(iter(_write_snapshots.values()), "")
                        # A new-file snapshot holds the _MISSING_SNAP sentinel, not
                        # str — there is no original indentation to repair against.
                        if not isinstance(_orig_content, str):
                            _orig_content = ""
                        _edit_ops = args.get("operations", [])
                        if _orig_content and _edit_ops:
                            _repaired = self._auto_repair_indent(_orig_content, _edit_ops)
                            if _repaired is not None:
                                _repair_path = next(iter(_write_snapshots))
                                try:
                                    with open(_repair_path, "w", encoding="utf-8") as _f:
                                        _f.write(_repaired)
                                except OSError:
                                    _repaired = None

                            if _repaired is not None:
                                _reverify_ok, _reverify_detail = self._verify_after_write(
                                    _write_snapshots,
                                    _post_contents={_repair_path: _repaired},
                                )
                                if _reverify_ok:
                                    _repair_ok = True
                                    logger.info(
                                        "Write safety: auto-repaired indentation for %s (%s ops)",
                                        _repair_path, len(_edit_ops),
                                    )

                    if _repair_ok:
                        return ToolResult(
                            ok=True,
                            content="Auto-repaired indentation — edit applied successfully",
                            execution_time=result.execution_time,
                        )

                    # --- Try argument mismatch repair before rollback ---
                    # Handles "not enough arguments" / "too many arguments"
                    # when a function signature changes but callers haven't
                    # been updated yet (e.g. in multi-op plans).
                    # An internal failure here (e.g. a broken lazy import) must
                    # NOT escape: the rollback below is the last line of defense
                    # keeping a syntax-broken file off the disk.
                    try:
                        _arg_repaired = self._repair_verify_failure(_write_snapshots)
                    except Exception as _repair_exc:
                        logger.error(
                            "Write safety: repair path crashed — falling through "
                            "to rollback: %s", _repair_exc, exc_info=True,
                        )
                        _arg_repaired = False
                    if _arg_repaired:
                        # Repair succeeded — file is already written, re-verify
                        logger.info(
                            "Write safety: argument mismatch repaired, "
                            "edit applied successfully"
                        )
                        result.metadata["repaired_args"] = True
                        return result

                    # --- Cross-op dependency guard: non-syntax compilation errors may be ---
                    # resolved by downstream ops (e.g. op1 changes a signature and
                    # op2 updates callers). Classify the error: true syntax errors
                    # always rollback, but type/compilation errors are kept.
                    try:
                        _soft_fail = self._should_soft_fail_verify(_verify_detail, _write_snapshots)
                    except Exception as _soft_exc:
                        logger.error(
                            "Write safety: soft-fail classification crashed — "
                            "treating as hard fail (rollback): %s",
                            _soft_exc, exc_info=True,
                        )
                        _soft_fail = False
                    if not _soft_fail:
                        self._restore_snapshots(_write_snapshots)
                        # Undo the handler's _text_edited_files recording so the
                        # working tree and the session-edit ledger stay consistent
                        # after rollback (see _pre_text_edits snapshot above).
                        self._text_edited_files = _pre_text_edits
                    else:
                        logger.warning(
                            "Write safety: non-syntax error — keeping changes "
                            "(may be resolved by downstream ops): %s",
                            _verify_detail,
                        )
                        result.metadata["verify_warning"] = _verify_detail
                        return result

                    _detail_parts = [_verify_detail]

                    # 1. Attempted change context for edit_file (operations list)
                    if tool_name == "edit_file":
                        _ops = args.get("operations", [])
                        if _ops:
                            _parts = []
                            for _oi, _op in enumerate(_ops):
                                _t = _op.get("type", "?")
                                _a = _op.get("anchor", "")[:120]
                                _c = _op.get("content", "")[:300]
                                _parts.append(
                                    f"  [{_oi}] type={_t}\n"
                                    f"       anchor: {_a!r}\n"
                                    f"       content: {_c!r}"
                                )
                            _detail_parts.append(
                                "--- Attempted edit operations ---\n" + "\n".join(_parts)
                            )

                    # 2. Restored (original) file content with line numbers
                    _err_path, _err_line = None, None
                    _parts = _verify_detail.split(':', 3)
                    if len(_parts) >= 3 and _parts[1].isdigit() and _parts[2].isdigit():
                        _err_path, _err_line = _parts[0], int(_parts[1])
                    for _path, _orig in _write_snapshots.items():
                        if _err_path and _path != _err_path:
                            continue
                        # New-file snapshots hold the _MISSING_SNAP sentinel (no
                        # original content) — rollback deleted the file, so there
                        # is no "AFTER ROLLBACK" context to show for it.
                        if not isinstance(_orig, str):
                            _detail_parts.append(
                                f"--- {_path}: newly created file removed by rollback ---"
                            )
                            continue
                        _lines = _orig.splitlines()
                        if _err_line and 1 <= _err_line <= len(_lines):
                            _start = max(0, _err_line - 3)
                            _end = min(len(_lines), _err_line + 2)
                            _ctx = "\n".join(
                                f"{i+1:4d}|{_lines[i]}"
                                for i in range(_start, _end)
                            )
                            _detail_parts.append(
                                f"--- {_path} (lines {_start+1}-{_end}) AFTER ROLLBACK ---\n{_ctx}"
                            )

                    _full_detail = "\n\n".join(_detail_parts)
                    logger.warning(
                        "Write safety: %s %s", tool_name, _full_detail,
                    )
                    return ToolResult(
                        ok=False,
                        content="",
                        error=f"ROLLBACK: {tool_name}: {_full_detail}",
                        execution_time=result.execution_time,
                    )

            # Record performance metrics
            cache_hit = result.metadata.get("cache_hit", False) if result.metadata else False
            get_global_collector().record_tool_call(tool_name, result.execution_time, cache_hit)

            # ── Phase 2: deterministic semantic auto-repair (F401/F821) ──
            # Runs after syntax verify passes (or after soft-fail preserves changes).
            # Auto-fixes unused imports (F401 via ruff --fix) and undefined names
            # (F821 via project-wide import search). Non-fatal: any failure here
            # degrades gracefully to Phase 1 warning surfacing in design_chat_loop.py.
            #
            # Self-validating tools (edit_text/edit_ast/anchor_edit) skip the
            # syntax-verify snapshot above (redundant I/O), but they can still
            # *introduce* undefined names (F821) — e.g. edit_ast inserting a
            # reference to a symbol whose import is missing. Semantic auto-repair
            # is orthogonal to syntax verify, so build an on-demand snapshot here
            # when the verify-path snapshot was skipped. Without this, F821 import
            # insertion silently never runs for these tools.
            if result.ok and tool_name in self._WRITE_TOOLS:
                _sem_snapshots = _write_snapshots
                if not _sem_snapshots:
                    try:
                        _sem_snapshots = self._snapshot_target_files(tool_name, args)
                    except Exception:
                        _sem_snapshots = {}
                if _sem_snapshots:
                    try:
                        _sem_repaired = self._safety_manager.auto_repair_semantic(
                            _sem_snapshots
                        )
                        if _sem_repaired > 0:
                            logger.info(
                                "[AUTO-REPAIR] Write safety: auto-repaired %d semantic finding(s)",
                                _sem_repaired,
                            )
                            result.metadata["semantic_repaired"] = _sem_repaired
                    except Exception as _sem_exc:
                        logger.debug(
                            "Semantic auto-repair error: %s", _sem_exc, exc_info=True
                        )

            # Cache result for read-only tools (if not already a cache hit)
            if (self._tool_result_cache is not None and
                tool_name in self._READ_ONLY_TOOLS and
                result.ok and not cache_hit):
                # Convert ToolResult to serializable dict
                cached = {
                    "ok": result.ok,
                    "content": result.content,
                    "error": result.error,
                    "metadata": dict(result.metadata),
                }
                _cache_paths = self._extract_read_scope_paths(tool_name, args)
                self._tool_result_cache.set(tool_name, args, cached, paths=_cache_paths)
                logger.debug("Tool result cache SET: %s (args: %s, paths: %s)", tool_name, args, _cache_paths)

            # Invalidate tool result cache + notify listeners on successful write operations.
            # For bash, only invalidate when the command actually mutates files/git state —
            # read-only bash (ls, git status, grep, …) leaving the cache intact greatly
            # improves hit rate because the model interleaves such commands between edits.
            _should_invalidate = result.ok and self._tool_call_mutates(tool_name, args)
            if _should_invalidate:
                if self._tool_result_cache is not None:
                    # Write tools know their target file(s) → drop only overlapping
                    # cache entries. bash (and anything else with an unknown target)
                    # falls back to a full clear — safer than guessing scope.
                    _write_paths = (
                        self._extract_write_target_paths(tool_name, args)
                        if tool_name in self._WRITE_TOOLS else None
                    )
                    if _write_paths:
                        _n = self._tool_result_cache.invalidate_paths(_write_paths)
                        logger.debug(
                            "Tool result cache scoped-invalidated %d entr(y/ies) for %s -> %s",
                            _n, tool_name, _write_paths,
                        )
                    else:
                        self._tool_result_cache.clear()
                        logger.debug("Tool result cache cleared due to successful write tool: %s", tool_name)
                for cb in self._write_success_callbacks:
                    try:
                        cb()
                    except Exception:
                        pass  # non-critical — never block execution

            return result
        except Exception as e:
            logger.exception("Tool %s raised exception", tool_name)
            return ToolResult(ok=False, content="", error=f"{type(e).__name__}: {e}", execution_time=time.monotonic() - start_time)
        finally:
            if locked_paths and flm is not None:
                flm.release_all(locked_paths)

    def dispatch_parallel(self, tool_calls: list[dict[str, Any]]) -> list[ToolResult]:
        """
        Execute multiple tool calls in parallel using async executor.

        Args:
            tool_calls: List of dicts with 'tool' (name) and 'args' keys

        Returns:
            List of ToolResult in the same order as input
        """
        # Safety: never parallelize write tools (apply_patch, write_plan, edit_ast)
        # nor serial tools (ask_user) — see _SERIAL_TOOLS docstring.
        # Safety: never parallelize mutating tools (write tools, or a bash whose
        # command mutates files/git state) nor serial tools (ask_user) — see
        # _SERIAL_TOOLS / _tool_call_mutates. A mutating bash (rm, git commit,
        # "> file", …) races with concurrent reads/other bash, so the whole batch
        # falls back to sequential exactly as it does for an explicit write tool.
        # Read-only bash (ls, git status, grep) still parallelizes.
        has_write_tool = any(
            self._tool_call_mutates(call.get("tool", ""), call.get("args", {}))
            for call in tool_calls
        )
        has_serial_tool = any(
            self._tool_call_is_serial(call.get("tool", ""), call.get("args", {}))
            for call in tool_calls
        )
        if (not self.config.parallel_tool_execution_enabled
                or len(tool_calls) <= 1
                or has_write_tool
                or has_serial_tool):
            # Fall back to sequential execution
            logger.debug("Parallel execution disabled or unsafe: enabled=%s, count=%d, has_write=%s, has_serial=%s",
                         self.config.parallel_tool_execution_enabled, len(tool_calls), has_write_tool, has_serial_tool)
            results = []
            for call in tool_calls:
                results.append(self.dispatch(call.get("tool", ""), call.get("args", {})))
            return results

        logger.debug("Parallel tool execution activated for %d tools", len(tool_calls))
        # Use async executor if available
        if self.async_executor is not None:
            import asyncio
            loop = None
            # Capture the caller's current loop WITHOUT raising. The legacy
            # get_event_loop() can raise RuntimeError ("There is no current event
            # loop in thread '...'", e.g. in a worker thread that never set one),
            # in which case _prev_loop stays None and we skip restoration. Setting
            # the loop to None is itself valid, so we restore whatever we captured.
            _prev_loop = None
            _captured = True
            try:
                _prev_loop = asyncio.get_event_loop_policy().get_event_loop()
            except RuntimeError:
                _captured = False  # no loop in this thread — nothing to restore
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(
                    self.async_executor.execute_parallel(tool_calls)
                )
                return results
            except Exception:
                logger.exception("Async parallel tool execution failed")
                # Fall back to thread pool
                pass
            finally:
                if loop is not None:
                    loop.close()
                # Restore the caller's original loop, but only if we captured one.
                if _captured:
                    try:
                        asyncio.set_event_loop(_prev_loop)
                    except Exception:
                        pass

        # Fallback: shared thread pool (eliminates pool create/destroy overhead)
        futures = []
        for call in tool_calls:
            tool_name = call.get("tool", "")
            args = call.get("args", {})
            future = shared_pool.submit(self.dispatch, tool_name, args)
            futures.append((future, call))

        # Collect results in order
        results = []
        for future, _call in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.exception("Parallel tool execution failed")
                results.append(ToolResult(
                    ok=False, content="",
                    error=f"Parallel execution error: {type(e).__name__}: {e}"
                ))
        return results

    def get_tool_schemas(
        self,
        read_only_request: bool = False,
        role: str = "agent",
        lang_filter: Optional[LanguageId] = None,
    ) -> list[dict[str, Any]]:
        """Return tool schemas for the LLM API.

        All lanes share the same tool schemas defined in ``AGENT_TOOL_SCHEMAS``.
        The ``role`` parameter is kept for backward compatibility but no longer
        filters tools — design chat and agent lane expose the same tool set.

        Args:
            read_only_request: Kept for backward compatibility — no longer filters
                write tools.
            role: Kept for backward compatibility — no longer filters tools.
            lang_filter: When set to a non-Python LanguageId, schemas with
                ``"x_python_only": True`` are excluded.  Pass ``None`` (default)
                to include all tools (Python or mixed-language repos).

        Note:
            Returns the shared ``AGENT_TOOL_SCHEMAS`` list directly when no
            filtering is needed (no per-call copy). Callers must NOT mutate the
            returned list or its dicts. When ``lang_filter`` filters out tools,
            the filtered list is memoized per registry instance — stable object
            identity restores the ``id()``-keyed token cache in
            ``estimate_tokens_from_tool_schemas``. Callers must NOT mutate the
            returned list or its dicts.
        """
        if lang_filter is not None and lang_filter is not LanguageId.PYTHON:
            try:
                return self._filtered_tool_schemas
            except AttributeError:
                _filtered = [s for s in AGENT_TOOL_SCHEMAS if not s.get("x_python_only")]
                self._filtered_tool_schemas = _filtered
                return _filtered
        return AGENT_TOOL_SCHEMAS

    def get_tool_names(self, lang_filter: Optional[LanguageId] = None) -> frozenset:
        """Return the frozen set of known tool names for O(1) membership checks.

        Cheaper than calling :meth:`get_tool_schemas` and building a set each
        turn — useful for validating LLM-emitted tool-call names (see
        ``agent_turn_pipeline._build_and_filter_prepared_calls``).

        Memoized per registry instance when ``lang_filter`` is set — stable
        object identity avoids repeated frozenset construction every turn.

        Args:
            lang_filter: When set to a non-Python LanguageId, Python-only tools
                (``"x_python_only": True``) are excluded so that a masked tool
                is rejected at validation time, not only hidden from the schema.
                Pass ``None`` (default) for the full set.
        """
        if lang_filter is not None and lang_filter is not LanguageId.PYTHON:
            try:
                return self._filtered_tool_names
            except AttributeError:
                _filtered = frozenset(s["name"] for s in AGENT_TOOL_SCHEMAS if not s.get("x_python_only"))
                self._filtered_tool_names = _filtered
                return _filtered
        return AGENT_TOOL_NAMES

    def has_tool_handler(self, tool_name: str) -> bool:
        """Return True if ``tool_name`` has a registered handler in this registry.

        Unlike :meth:`get_tool_names` (which checks schema existence), this
        checks the actual handler mapping. Tools whose handler method name
        differs from the tool name (e.g. ``bash`` → ``_tool_shell_exec``)
        are correctly accepted.
        """
        return tool_name in self._TOOL_HANDLER_MAP

    def _correct_bias_path(self, text: str) -> str:
        """LLM training-data path bias correction — replaces bias paths in shell commands/paths with the actual repo_root.

        Converts virtual paths containing bias paths like /workspace, /app, /project,
        /code and repo basenames to the actual repo_root. Preserves subpaths (e.g. /tests)
        and works within shell commands as well.
        """
        if not text:
            return text
        _basename = Path(self.repo_root).name
        _BIAS_PATHS = frozenset({"/workspace", "/app", "/project", "/code", "/repo"})

        # Pass 1: Strict bias paths (/workspace, /app, /project, /code)
        # When the LLM uses both virtual root + project name like /workspace/asicode,
        # remove the repo basename prefix (/asicode) from the subpath to prevent double paths.
        #   /workspace/asicode        → repo_root
        #   /workspace/asicode/tests   → repo_root/tests
        #   /workspace/tests            → repo_root/tests
        for _bp in _BIAS_PATHS:
            if _bp not in text:
                continue

            # Recompute each iteration: a prior rewrite may have shifted every
            # protected interval's offsets (replacement != matched length).
            _iv = _literal_intervals(text)

            def _strict_repl(m, _b=_basename):
                # Never rewrite a bias path that lives inside a shell-quoted
                # literal or a heredoc body (grep '/workspace', a config written
                # via <<'EOF', etc.) — doing so corrupts the literal content.
                if _match_in_quotes(m.start(), _iv):
                    return m.group(0)
                prefix = "" if m.group(1) == "~" else m.group(1)
                cd = m.group(2) or ""
                subpath = m.group(3) or ""
                if subpath.startswith(f"/{_b}"):
                    subpath = subpath[len(_b) + 1:]
                return prefix + cd + self.repo_root + subpath

            new_text = re.sub(
                rf'(^|[\s~])(cd\s+)?{re.escape(_bp)}(/\S*)?(?=\s|[&;]|$)',
                _strict_repl,
                text,
            )
            if new_text != text:
                logger.info("bias_path: '%s' -> '%s': %.200s", _bp, self.repo_root, new_text)
                text = new_text

        # Pass 2: Repo basename correction
        # For embedded paths (e.g. /Users/admin/workspace/asicode/tests),
        # strip the repo basename (/asicode) prefix and replace with repo_root
        #
        # ⚠️  Do NOT use \S*? (non-whitespace) for prefix matching:
        #    In URL query params (?repo_root=/asicode&...), \S*? would consume the
        #    entire URL, destroying the command.
        #    Use [\w./~+@-] (only characters found in file paths) instead.
        _bp_basename = f"/{_basename}"
        if _bp_basename in text:
            _iv = _literal_intervals(text)

            def _basename_repl(m):
                if _match_in_quotes(m.start(), _iv):
                    return m.group(0)
                prefix = "" if m.group(1) == "~" else m.group(1)
                cd = m.group(2) or ""
                subpath = m.group(3) or ""
                return prefix + cd + self.repo_root + subpath

            _re_basename = re.compile(
                rf'(^|[\s~])(cd\s+)?[\w./~+@-]*?{re.escape(_bp_basename)}(/\S*)?(?=\s|[&;]|$)',
                re.ASCII,
            )
            new_text = _re_basename.sub(_basename_repl, text)
            if new_text != text:
                logger.info("bias_path: '%s' -> '%s': %.200s", _bp_basename, self.repo_root, new_text)
                text = new_text

        # Safety dedup: clean up double paths like repo_root/asicode
        _double = f"{self.repo_root}/{_basename}"
        if _double in text:
            text = text.replace(_double, self.repo_root)
            logger.info("bias_path: dedup '.../%s': %.200s", _basename, text)

        return text

    def normalize_args_for_display(self, args: dict) -> dict:
        """Return a copy of *args* with bias paths corrected in all string values.

        Used before emitting event payloads so the CLI display shows real paths
        instead of LLM training-data bias paths (e.g. /workspace, /home/ubuntu/…).
        """
        return {
            k: self._correct_bias_path(v) if isinstance(v, str) else v
            for k, v in args.items()
        }

    @property
    def _effective_repo_root(self) -> str:
        """Return the effective repo root, preferring staging override."""
        return self._repo_root_override or self.repo_root

    def _secure_path(self, path: str) -> Optional[Path]:
        """
        Resolve path within repo_root.
        Returns None if path is outside repo_root or doesn't exist.
        """
        path = self._correct_bias_path(path)
        try:
            repo = Path(self._effective_repo_root).resolve()
            p = Path(path)

            # If path is absolute, check if it's within repo
            if p.is_absolute():
                resolved = p.resolve()
                try:
                    resolved.relative_to(repo)
                except ValueError:
                    logger.warning("Path traversal attempt blocked: %r -> %s", path, resolved)
                    return None
                return resolved
            else:
                # Relative path - resolve relative to repo
                resolved = (repo / path).resolve()
                try:
                    resolved.relative_to(repo)
                except ValueError:
                    logger.warning("Path traversal attempt blocked: %r -> %s", path, resolved)
                    return None
                return resolved
        except Exception:
            return None  # non-critical — never block execution

    @property
    def applied_patches(self) -> list[str]:
        """
        Return list of successfully applied patch texts.
        """
        return list(self._applied_patches)

    def __del__(self):
        # Release the AsyncToolExecutor's ThreadPoolExecutor to avoid leaking
        # worker threads when the registry is garbage-collected. Best-effort:
        # __del__ must never raise, so swallow all errors.
        try:
            _exec = getattr(self, "async_executor", None)
            if _exec is not None and hasattr(_exec, "shutdown"):
                _exec.shutdown()
        except Exception:
            pass
