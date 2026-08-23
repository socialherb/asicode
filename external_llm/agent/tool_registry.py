# REGRESSION_TEST_01
"""
Tool Registry for asicode Agent

Provides safe tool dispatch for the LLM agent loop.
Security: all file operations are bounded by repo_root.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from .agent_profile import AgentProfile

import subprocess
from concurrent.futures import TimeoutError as _FutureTimeoutError

from external_llm.common.atomic_io import atomic_write_text
from external_llm.common.indent_utils import reindent_text
from external_llm.common.walk_policy import _walk_should_skip_dir

from ..graph.graph_facade import RepositoryGraphFacade

# ── Extracted modules ────────────────────────────────────────────────────
from ..languages import LanguageId, SyntaxValidationResult
from ._thread_pool import CANCEL_POLL_INTERVAL, shared_pool
from .agent_loop_types import WRITE_TOOL_NAMES, AgentCancelled
from .argument_repairer import ArgumentRepairer
from .call_graph import CallGraphIndexer
from .cancel_scope import call_cancel_scope, current_cancel_event
from .config.thresholds import config as _cfg
from .intent_models import IntentResult
from .lint_runner import LintRunner
from .performance_metrics import get_global_collector
from .rag_searcher import RAGSearcher
from .symbol_search import get_symbol_searcher
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
from .tool_result_cache import _path_sig
from .tool_safety import WriteSafetyManager
from .tool_schemas import TOOL_NAME_VARIANTS, TOOL_SCHEMA_VARIANTS
from .write_targets import write_target_paths

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
    stream_callback: Callable[[str, dict[str, Any]], None] | None = None
    # Whether the stream_callback handles "content" events (incremental token rendering).
    # False in CLI mode (_ProgressPrinter has no "content" handler — all SSE overhead
    # would be discarded). Skips token_callback lambda creation + SSE streaming overhead.
    consume_content_events: bool = True
    # Lint maximum issue count
    max_lint_issues: int = 50  # Maximum number of issues to return from lint results
    # TDD auto-feedback loop
    auto_test_on_patch: bool = False  # Automatically run pytest after patch apply
    max_tdd_cycles: int = 3  # Maximum retry count after consecutive failures
    test_paths: list[str] = field(default_factory=list)  # pytest paths/arguments
    # Timeout budget (seconds) for the run_tests tool. Default 300, not 120:
    # with an empty test_paths the TDD gate runs the FULL suite, and 120 s was
    # smaller than it (~150-180 s here) — the gate timed out on every green run.
    test_timeout_sec: int = 300
    # Self-Review
    self_review_enabled: bool = False  # Enable post-execution self-review phase
    max_review_turns: int = 3  # Maximum review turns for self-review corrections
    # RAG: related file automatic provide
    rag_enabled: bool = True  # Auto-inject related file Top-K at session start
    rag_top_k: int = 5  # Number of files to auto-provide
    # Human-in-the-Loop approval gate
    # Callable: (tool_name, args, preview_text) -> bool (True=proceed)
    approval_callback: Callable[[str, dict[str, Any], str], bool] | None = None

    # User Checkpoint: LLM asks user for questions/confirmations
    # Callable: (question_data: dict) -> dict  ({"status": "answered"|"timeout", "answer": ...})
    user_checkpoint_enabled: bool = True
    user_checkpoint_max_questions: int = 3  # Max questions per session
    user_checkpoint_timeout: int = (
        ASK_USER_DEFAULT_TIMEOUT  # Timeout (seconds); on expiry, proceeds autonomously with default
    )
    user_checkpoint_callback: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    _user_checkpoint_count: int = 0  # Runtime: current session question counter

    # Server-side cancellation support (set by /agent/cancel)
    cancel_event: threading.Event | None = None

    # Mid-task user message injection (set by /agent/message/{session_id})
    # Queue items: str (user message text)
    message_queue: Any | None = None

    # Context sliding window: keep only this many recent non-system messages
    # (0 = disabled, keep all). Prevents token overflow on long runs.
    context_window_size: int = 60

    # Multi-agent fields
    agent_id: str = "main"
    file_lock_manager: Any | None = None  # FileLockManager instance

    # Auto-observation: after a successful patch, optionally inject the diff as a user observation.
    # Default False to keep tool dispatch predictable in tests and avoid extra tool calls.
    auto_observation_enabled: bool = False

    # Parallel tool execution: run independent tools concurrently
    # (concurrency is governed by the process-wide _thread_pool.shared_pool —
    # there is no per-session worker-count knob)
    parallel_tool_execution_enabled: bool = True

    model_name: str = ""
    # ── Helper configuration (canonical) ─────────────────────────────────────
    # Helper is NOT a lane. Helper is the delegate_to_helper tool capability.
    helper_enabled: bool = False
    helper_model: str = ""  # Any model identifier (API or Ollama)
    helper_max_calls: int = 5  # Max delegation calls per session
    helper_ollama_url: str = "http://127.0.0.1:11434"  # Used when helper_model is Ollama

    route_decision: Any | None = None
    # Intent understanding result from IntentResolver (language-neutral).
    # Assigned post-construction by the REPL engine (`_build_engine`);
    # consumed by SpecResolver to avoid duplicate LLM calls.
    intent_result: IntentResult | None = None

    # ── Bench / Experiment ────────────────────────────────────────────────────
    # Force a specific patch strategy for all modify_symbol ops in this session.
    # Values: "" (auto), "surgical_edit", "replace_symbol_body", "ast_op"
    # Used by patch_strategy_bench for multi-strategy comparison runs.
    force_patch_mode: str = ""
    # Exploration rate for online bandit learning (0.0 = pure exploit, 1.0 = pure random).
    # When triggered, a random strategy is tried instead of the policy-recommended one.
    patch_exploration_rate: float = 0.0

    # Vector cache for semantic search
    vector_cache_enabled: bool = True

    # Ollama reasoning / thinking toggle (None = not explicitly set; provider decides)
    thinking_mode: bool | None = False
    # Reasoning depth for providers that support it ("high" | "max"); None = provider default
    reasoning_effort: str | None = None

    # Turn reduction optimizations
    dynamic_turn_budget_enabled: bool = True  # Dynamically adjust turn budget based on progress

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

    # NO planner_llm_client / planner_model here. They carried the separate LLM
    # client PlannerAgent used, and PlannerAgent went with the PLANNER lane on
    # 2026-08-03 — after which every reference in the tree was an assignment or
    # a guard around one, with zero readers. Confirmed by running an AgentLoop
    # with a tripwire object in each field: zero accesses across a
    # read/grep/write turn, while the same probe on a field that IS read
    # (helper_enabled) fired immediately. The webapp still accepts the matching
    # planner_* request params and warns that they are inert; see agent_stream.

    # Developer model: the client/model the AgentLoop itself runs on when the
    # caller overrides it (webapp resolves this into AgentLoop's llm_client).
    developer_llm_client: Any | None = None
    developer_model: str = ""

    # Design chat mode: write tools disabled, no early-finish, LLM synthesizes full response
    design_chat_mode: bool = False

    # ── Token continuity (cross-phase) ──────────────────────────────────────
    # Token offset from prior phases (e.g., Design Chat, main agent loop).
    # Applied to the agent loop so token_usage events
    # show cumulative totals across phase boundaries. Cleared after apply.
    _token_offset_prompt_tokens: int = 0
    _token_offset_completion_tokens: int = 0
    _token_offset_cache_read_tokens: int = 0

    # Phase 7.1: Conversation layer integration (opt-in)
    # When True, routes requests through ConversationRouter → DesignStateManager →
    # FreezeManager → HandoffManager before the planner lane.
    conversation_layer_enabled: bool = False

    # ── Context budget management ─────────────────────────────────────────
    context_budget_enabled: bool = True
    context_budget_reserve_output: int = 4096
    # ── Agent profile (custom tool/model/turn constraints) ────────────────
    # Load via: AgentProfile.load(name, repo_root) or load_profile(name, repo_root)
    # None = no profile, all defaults apply.
    agent_profile: Any | None = None  # AgentProfile instance

    # ── Cross-repo read boundary (trust-scoped) ───────────────────────────
    # When False (default), read tools (read_file / get_file_outline /
    # read_image) are confined to repo_root by _secure_path. When True they may
    # read any resolvable path on the host.
    #
    # This is a TRUST toggle, NOT a folder allowlist: the CLI already exposes an
    # unrestricted `bash` tool (cat/rg any absolute path), so in a trusted local
    # CLI the repo-only read boundary is pure friction, not a security control —
    # the agent trivially bypasses it via bash. The webapp, where repo_root is
    # attacker-controlled, MUST leave this False so _secure_path keeps closing the
    # arbitrary-file-read surface that path_security.py guards. Only trusted local
    # entry points (asi.py CLI, collaborate REPL) opt in; sub-agent configs
    # inherit it via dataclasses.replace. Write tools ignore this flag entirely —
    # they stay confined to repo_root regardless.
    unrestricted_read: bool = False

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

    def make_token_callback(self) -> Callable[[str | None], None] | None:
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
        def _token_cb(text: str | None) -> None:
            if text is not None:
                cb("content", {"text": text})

        return _token_cb


# ── [REMOVED] is_small_model / model prefix lists ─────────────────────────────
# Model-name-based restrictions have been removed. All models are treated equally.


@dataclass(frozen=True)
class SemanticOutcome:
    """One file's turn-end semantic verdict — or the reason there isn't one.

    ``diagnostics == []`` and "nothing checked this file" are different answers
    with opposite meanings for the model, and they used to share a
    representation: every skip path in every provider returned ``ok=True,
    errors=[]``, which downstream is exactly what a clean check produces. A user
    with no pyright installed therefore had every Python edit reported as
    semantically verified. Semantic checking skips for ordinary reasons — the
    toolchain is not installed, it timed out, the project has no config for that
    language — and none of them is evidence about the file.

    ``skip_reason`` non-empty means no verdict was reached, and it reaches the
    model verbatim so it can act on the difference (re-check another way, or
    say the check was unavailable) instead of trusting a check that never ran.
    """

    diagnostics: list[dict] = field(default_factory=list)
    skip_reason: str = ""

    @property
    def checked(self) -> bool:
        return not self.skip_reason


@dataclass
class ToolResult:
    ok: bool
    content: str = ""
    error: str | None = None
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


def _bias_matched_candidate(match) -> str:
    """The path token of a bias-correction match — group(0) minus the leading
    whitespace/tilde prefix, an optional ``cd `` verb, and the trailing subpath.
    """
    _candidate = match.group(0)[len(match.group(1)) :]
    _candidate = re.sub(r"^cd\s+", "", _candidate)
    _sub = match.group(3) or ""
    if _sub:
        _candidate = _candidate[: -len(_sub)]
    return _candidate


def _bias_matched_path_is_real(match) -> bool:
    """True if the path matched by a bias-correction regex (up to and
    including the bias token) actually exists on disk.

    Training-data bias paths are VIRTUAL roots — they never existed on this
    machine (/workspace, /home/ubuntu/..., ...). A matched path that EXISTS is
    real user data: rewriting it would silently redirect a live command into
    the repo (reading/writing the wrong file). Real paths are therefore never
    rewritten; only nonexistent (virtual) ones are corrected.
    """
    _candidate = _bias_matched_candidate(match)
    return bool(_candidate) and os.path.exists(os.path.expanduser(_candidate))


# Bias tokens are virtual roots from LLM training data — never real machine
# paths. /repo is included: some models emit /repo as the workdir root.
_BIAS_PATHS: frozenset[str] = frozenset({"/workspace", "/app", "/project", "/code", "/repo"})

# Scratch/temp roots are REAL machine roots — a path under one of them is a
# user-intended destination (often not yet created: ``mkdir -p /tmp/<name>``,
# ``tar -C /tmp/<name>``, ``git worktree add /tmp/<name>``), NEVER a
# training-data virtual root. ``os.path.exists()`` alone cannot tell the two
# apart because the scratch destination legitimately does not exist yet — pass
# 2's basename regex would otherwise rewrite ``/tmp/<basename>`` → repo_root
# and run the command against the real repository. Live bug class 2026-08-05:
# ``ls /tmp/asicode/files`` / ``rm -rf /tmp/asicode`` were rewritten, and
# ``tar -C /tmp/<basename>`` is a silent destructive overwrite because
# tar/cp/mv/rsync have no approval gate.
_SCRATCH_ROOTS: frozenset[str] = frozenset(
    {
        tempfile.gettempdir(),
        "/tmp",
        "/private/tmp",
        "/var/tmp",
        "/var/folders",
        "/private/var/folders",
    }
)


def _under_scratch_root(candidate: str) -> bool:
    """True if *candidate* (~-expanded, symlink-resolved) lies under a scratch
    root. Symlink resolution keeps macOS (/tmp → /private/tmp) and Linux (/tmp)
    consistent. An empty candidate is never scratch.
    """
    if not candidate:
        return False
    c = os.path.realpath(os.path.expanduser(candidate))
    return any(c == r or c.startswith(r.rstrip("/") + "/") for r in (os.path.realpath(x) for x in _SCRATCH_ROOTS))


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
      WebSearchToolsMixin — search_web (SearXNG/Startpage/Exa merged; Brave/DDG/Naver fallback)
    """

    # Directories pruned when counting source files for language detection.
    # These never indicate the repo's primary language (deps, caches, build
    # output, VCS metadata) and walking them only distorts counts + wastes time.
    _COUNT_SKIP_DIRS = frozenset(
        {
            ".git",
            ".hg",
            ".svn",
            "node_modules",
            "bower_components",
            "vendor",
            ".venv",
            "venv",
            "env",
            ".env",
            "__pycache__",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            "dist",
            "build",
            "target",
            "out",
            ".next",
            ".gradle",
            ".idea",
            ".vscode",
        }
    )

    # Module-level cache for _detect_repo_language. The repo's language
    # composition is immutable during a run, so caching by repo_root avoids
    # the full os.walk (~259-452ms/call, measured on this repo) on every
    # ToolRegistry construction (IPC worker creates one per task).
    _LANGUAGE_DETECTION_CACHE: ClassVar[dict[str, LanguageId | None]] = {}

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
        "bash": "_tool_shell_exec",  # handler method != tool name
        "job": "_tool_job",
        "find_symbol": "_tool_find_symbol",
        "grep": "_tool_grep",
        "glob": "_tool_glob",
        "read_file": "_tool_read_file",
        "read_symbol": "_tool_read_symbol",
        "find_references": "_tool_find_references",
        "find_tests_for_symbol": "_tool_find_tests_for_symbol",
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
        "search_web": "_tool_search_web",
        "web_fetch": "_tool_web_fetch",
        "browser_action": "_tool_browser_action",
        "read_image": "_tool_read_image",
    }

    @staticmethod
    def _detect_repo_language(repo_root: str) -> LanguageId | None:
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
                    capture_output=True,
                    text=True,
                    timeout=5,
                    cwd=repo_root,
                    check=False,
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
            # _COUNT_SKIP_DIRS is a language-detection superset (build-output /
            # IDE dirs that never indicate the primary language), used with EXACT
            # match — but that misses venv* prefixes, *.egg-info and
            # site-packages dirs (vendored deps that distort the count).  Union
            # the shared walk_policy predicate on top so vendored trees are
            # excluded here just as they are from every other walker (F7).
            dirs[:] = sorted(d for d in dirs if d not in ToolRegistry._COUNT_SKIP_DIRS and not _walk_should_skip_dir(d))
            for _f in sorted(files):
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
        _result = LanguageId[max(counts, key=lambda k: counts[k])]
        ToolRegistry._LANGUAGE_DETECTION_CACHE[_norm] = _result
        return _result

    def _make_result(self, **kwargs) -> ToolResult:
        """Create a ToolResult without importing at module level."""
        return ToolResult(**kwargs)

    def __init__(
        self,
        repo_root: str,
        config: AgentConfig,
        local_assistant: Any | None = None,
        agent_profile: AgentProfile | None = None,
    ):
        self.repo_root = str(Path(repo_root).resolve())
        self._repo_root_override: str | None = None
        # Resolved-root memo for _secure_path. The effective root is a session
        # constant (repo_root frozen above; override set at most once), so
        # re-resolving it on every read/write tool call was pure filesystem I/O
        # on the hottest tool path. Keyed by the effective-root STRING so an
        # override change simply misses and re-resolves. Clones get their own
        # dict (clone_for_subagent bypasses __init__ via object.__new__).
        self._secure_root_resolve_cache: dict[str, Path] = {}
        # Detect dominant code language by counting source files
        # (_LANGUAGE_EXTENSION_GROUPS — single source of truth). Used by
        # get_tool_schemas() to mask Python-only tools in pure non-Python repos.
        self._repo_language: LanguageId | None = self._detect_repo_language(self.repo_root)
        self.config = config
        self._lint_runner = LintRunner(repo_root)
        self._symbol_searcher = get_symbol_searcher(repo_root)
        # Pass the live config (NOT a captured cancel_event value) so these
        # indexers read config.cancel_event FRESH at build() time. The design-
        # chat REPL mutates config.cancel_event PER TURN (asi.py) AFTER this
        # registry is constructed; a captured value would freeze the
        # construction-time value (None in design-chat) and leave ESC inert on
        # find_relevant_files / analyze_change_impact — the exact interactive
        # path it must protect. Mirrors the call-time fresh read vulture uses.
        self._rag_searcher = RAGSearcher(
            repo_root,
            vector_cache_enabled=config.vector_cache_enabled,
            config=config,
        )
        _cgi = CallGraphIndexer(repo_root, config=config)  # lazy build on first access
        self._call_graph = RepositoryGraphFacade(call_graph_indexer=_cgi, repo_root=repo_root)

        # Argument repair layer
        self._arg_repairer = ArgumentRepairer()

        # Write safety manager (snapshot/verify/rollback + approval gating)
        self._safety_manager = WriteSafetyManager(self.repo_root)

        # Parallel execution support — read-only batches dispatch through the
        # process-wide ``_thread_pool.shared_pool`` inside ``dispatch_parallel``.
        # The former asyncio layer (AsyncToolExecutor + ToolDependencyGraph) was
        # deleted: its workflow edges serialized independent read-only tools
        # (find_symbol -> find_references) while 3/4 edges were unreachable
        # behind the write-tool gate, and the remaining asyncio dispatch was
        # semantically identical to the thread-pool fallback.
        # Failure prediction database
        # Collected patches from apply_patch calls (for result tracking)
        self._applied_patches: list[str] = []
        # Files written this session by text-editing tools (edit_text / modify_symbol /
        # edit_ast / anchor_edit). apply_patch consults this (Opt D) to refuse clobbering
        # a working-tree edit it cannot safely merge — see _tool_apply_patch guard.
        self._text_edited_files: set[str] = set()
        # Pre-write checkpoint gate (Undo), shared by reference with subagent
        # clones so one run yields one undoable checkpoint.
        #
        # Built EAGERLY, which is the whole point. It used to be built lazily on
        # the first write, and the clone paths — which copy this attribute by
        # value under a "SHARED (not reset)" comment — therefore copied None:
        # OrchestratorAgent clones ``_registry_proto``, a registry it only ever
        # reads repo_root/config from, so at clone time no write had happened
        # and every subagent built its OWN gate. A multi-agent run produced one
        # checkpoint per subagent instead of one per run, and since
        # ``agent_loop`` stamps the id from the PARENT registry, a run whose
        # writes all happened in subagents reported no checkpoint_id at all.
        #
        # The read-only-run guarantee the laziness existed for is preserved by
        # the gate itself: constructing it only resolves the env var, and
        # ``RunCheckpointGate._get_store`` still defers CheckpointStore (and so
        # the .asicode/checkpoints/ mkdir) to the first captured write.
        from external_llm.agent.run_checkpoint import RunCheckpointGate

        self._run_checkpoint_gate = RunCheckpointGate(self.repo_root)
        # Semantic-lint coalescing (see begin_semantic_turn). Files written
        # during the current turn, awaiting ONE validate_semantics run at turn
        # end. Empty + inactive means "run inline", which is what every caller
        # outside the agent turn loop (MCP server, design chat, direct dispatch)
        # gets.
        self._semantic_pending: dict[str, str] = {}
        self._semantic_turn_active: bool = False
        # Tool result cache for read-only tools (TTL + LRU). Built via the shared
        # helper so __init__ and BOTH clone paths construct identical, ISOLATED
        # caches (never shared — concurrent in-process subagents would race on
        # the same LRU/TTL state).
        self._tool_result_cache = self._make_tool_result_cache(config)
        if self._tool_result_cache is not None:
            logger.info(
                "Tool result cache initialized (max=%s, TTL=%ss)",
                config.tool_result_cache_max_entries,
                config.tool_result_cache_ttl,
            )
        self._search_cache: dict[str, ToolResult] = {}

        # Local Assistant instance for delegating tasks to local LLMs
        self.local_assistant = local_assistant
        # Agent profile: explicit param takes precedence over config field
        self._agent_profile = agent_profile if agent_profile is not None else getattr(config, "agent_profile", None)
        if self._agent_profile is not None:
            logger.debug("Active agent profile: %s", self._agent_profile.name)

    def _checkpoint_before_write(self, tool_name: str, args: dict) -> None:
        """Capture pre-write state of this call's targets into the run checkpoint."""
        gate = self._run_checkpoint_gate
        if gate is None:
            # Unreachable via __init__ (which builds it) or either clone path
            # (which copy it). Reaching here means a registry was constructed
            # some third way and this run's writes are about to land in a
            # checkpoint of their own, splitting the run's Undo point — the
            # exact failure the eager construction removed. Recovered from, but
            # loudly: silently building one here is what hid it last time.
            from external_llm.agent.run_checkpoint import RunCheckpointGate

            logger.warning(
                "checkpoint gate missing on %s — building a detached one; this "
                "run's Undo point may be split across agents",
                type(self).__name__,
            )
            gate = RunCheckpointGate(str(self.repo_root))
            self._run_checkpoint_gate = gate
        if not gate.enabled:
            return
        # Target resolution sits INSIDE the try, not just inside before_write's.
        # The gate documents that it must never raise — a checkpoint is a
        # convenience, and failing to take one must not fail the user's edit —
        # but the resolution ran before that guarantee started, so a plan whose
        # ops were not dicts raised AttributeError straight out of dispatch and
        # replaced the handler's "each op must be a JSON object" guidance with a
        # raw traceback. write_target_paths no longer raises for that input; the
        # try makes the contract structural rather than a property of one callee.
        try:
            targets = self._extract_write_target_paths(tool_name, args)
        except Exception:
            logger.warning("checkpoint target resolution failed", exc_info=True)
            return
        gate.before_write(targets)

    def _checkpoint_after_write(self, tool_name: str, args: dict) -> None:
        """Confirm which pre-write absences the run actually turned into files.

        The gate has to fire BEFORE the handler, so at capture time "this path
        does not exist" does not yet mean "the run created it" — the write may
        be refused by the post-edit syntax gate, a scoped write filter or a bad
        argument. Confirming here keeps a refused write from leaving a tombstone
        that Undo would later act on by DELETING a file the user created by hand.
        """
        gate = self._run_checkpoint_gate
        if gate is None or not gate.enabled:
            return
        try:
            targets = self._extract_write_target_paths(tool_name, args)
        except Exception:
            logger.warning("checkpoint target resolution failed", exc_info=True)
            return
        gate.confirm_writes(targets)

    @property
    def run_checkpoint_id(self):
        """Id of this run's Undo checkpoint, or None if nothing was captured."""
        gate = self._run_checkpoint_gate
        return gate.checkpoint_id if gate is not None else None

    def begin_semantic_turn(self) -> None:
        """Start coalescing per-file semantic checks for one agent turn.

        Within a single turn the same file is often written several times
        (``edit_text`` + ``apply_patch``, or several ``modify_symbol`` calls on
        one class). Semantic validation spawns a heavy toolchain process —
        pyright / tsc / go build / javac / kotlinc / gcc -fsyntax-only — and
        350-400 ms of that is pure Node/JVM cold start rather than analysis
        (measured: 0.35 s for pyright on a two-line file with no imports). Five
        edits to one file cost 1.84 s of pyright against 1.94 s of total wall.

        So the run is coalesced to once per (turn, file). It must coalesce to
        the LAST write, not the first: the diagnostics describe the file on
        disk, and the agent needs to know about the error its most recent edit
        introduced. A first-write-wins cache reports the state before the later
        edits and then reports ``[]`` for each of them, which is indistinguishable
        from "checked, clean" downstream — a broken edit reads as a passing one.
        Hence deferral plus :meth:`drain_pending_semantic_checks` at turn end,
        rather than a seen-set.

        Outside a turn (MCP server, design chat, a direct ``dispatch`` call)
        nothing would ever drain the queue, so no turn is active there and
        ``_run_syntax_check_for_file`` keeps running the check inline.
        """
        self._semantic_pending.clear()
        self._semantic_turn_active = True

    def end_semantic_turn(self) -> None:
        """Discard any pending checks and mark no turn active, WITHOUT running them.

        ``drain_pending_semantic_checks`` runs the coalesced checks; this is the
        counterpart for paths that leave the turn body early — an uncaught
        exception or cancellation aborts the loop before the normal drain at
        turn end. It restores the "no active turn" invariant so a later
        out-of-turn dispatch (MCP server, design chat, direct ``dispatch``)
        keeps running its check inline instead of deferring into a queue
        nothing will ever drain, which would silently drop that dispatch's
        diagnostics. A no-op when the turn already ended normally, since drain
        already cleared both fields.
        """
        self._semantic_pending = {}
        self._semantic_turn_active = False

    def defer_semantic_check(self, abs_path: str, rel_path: str = "") -> bool:
        """Queue *abs_path* for the turn-end semantic run; False if not coalescing.

        Returning False is the signal to run the check inline right now.
        """
        if not self._semantic_turn_active:
            return False
        self._semantic_pending[abs_path] = rel_path or abs_path
        return True

    def drain_pending_semantic_checks(self) -> dict[str, SemanticOutcome]:
        """Run one semantic check per file written this turn; end the turn.

        Returns ``{abs_path: SemanticOutcome}``. The diagnostics inside use the
        same shape ``_run_syntax_check_for_file`` produces inline, so the two
        paths are interchangeable downstream.

        Every pending path appears in the mapping — including the ones nothing
        examined. "No diagnostics" and "no check ran" are different answers and
        must not share a representation: a provider whose toolchain is missing
        (no pyright, no tsc, no go) returns the second, and reporting it as an
        empty diagnostic list tells the model the file was verified. See
        :class:`SemanticOutcome`.
        """
        pending, self._semantic_pending = self._semantic_pending, {}
        self._semantic_turn_active = False
        if not pending:
            return {}
        from ..languages.registry import LanguageRegistry

        # Group by provider so each toolchain starts ONCE for the whole turn
        # rather than once per file. Coalescing already collapsed repeat writes
        # to one check per file; without this, a turn touching N files still
        # paid N cold starts, and startup is most of a short check (pyright
        # over 4 files: 2.167 s one-at-a-time vs 0.391 s batched). Providers
        # that have not overridden validate_semantics_batch fall back to the
        # same per-file loop as before. Grouping stops at the provider —
        # splitting a batch across project roots needs provider-specific
        # markers, so each override does its own.
        out: dict[str, SemanticOutcome] = {
            abs_path: SemanticOutcome(skip_reason="no semantic checker for this file type") for abs_path in pending
        }
        by_provider: dict[int, tuple[Any, list[str]]] = {}
        for abs_path in pending:
            try:
                provider = LanguageRegistry.instance().get(abs_path)
                if provider is None or not provider.capabilities().has_semantic_validator:
                    continue
                by_provider.setdefault(id(provider), (provider, []))[1].append(abs_path)
            except Exception as exc:  # advisory, never blocks
                logger.debug("Provider lookup failed for %s: %s", abs_path, exc)
                out[abs_path] = SemanticOutcome(
                    skip_reason="the language provider could not be loaded",
                )

        # Each group is a separate toolchain process (pyright, npx tsc, go
        # build), so a turn touching two languages used to pay their SUM.
        # Measured on one .py + one .ts: 0.424 s + 0.804 s = 1.228 s serial
        # against a 0.804 s parallel bound — the second toolchain becomes free
        # up to the cost of the slowest one. They share nothing: distinct
        # processes, distinct working directories, and the one provider that
        # writes a temp file (typescript) already names it by pid + uuid.
        #
        # The FIRST group always runs inline on this thread, so the common
        # single-language turn never touches the pool at all, and a saturated
        # pool can still only delay the extra groups rather than the whole
        # drain. Order of results does not matter — everything lands in `out`
        # keyed by path.
        _groups = list(by_provider.values())
        _pending_futures: list = []
        for _provider, _paths in _groups[1:]:
            try:
                _pending_futures.append(
                    (
                        _provider,
                        _paths,
                        shared_pool.submit(
                            _provider.validate_semantics_batch,
                            _paths,
                        ),
                    ),
                )
            except RuntimeError as exc:
                # Pool already shut down (interpreter teardown). Fall back to
                # running it inline rather than losing the diagnostics.
                logger.debug("Semantic batch could not be scheduled: %s", exc)
                _pending_futures.append((_provider, _paths, None))
        _first = [(p, paths, None) for p, paths in _groups[:1]]

        # Cancel-aware collection: poll instead of blocking on a bare
        # future.result() so ESC (cancel_event) is honored while a slow
        # toolchain is still running — the drain is advisory and must never
        # block the turn end. On cancel, the still-pending groups are marked
        # skipped (same shape as every other "no verdict" reason) and the
        # drain returns early; raising here would convert an advisory
        # diagnostic into a turn-end failure.
        _ce = getattr(self.config, "cancel_event", None)
        _cancelled = False
        results: dict[str, SyntaxValidationResult] = {}
        for provider, paths, future in [*_first, *_pending_futures]:
            if _cancelled:
                for abs_path in paths:
                    out[abs_path] = SemanticOutcome(
                        skip_reason="cancelled before the semantic check ran",
                    )
                continue
            try:
                if future is None:
                    results = provider.validate_semantics_batch(paths)
                else:
                    while True:
                        try:
                            results = future.result(timeout=CANCEL_POLL_INTERVAL)
                            break
                        except _FutureTimeoutError:
                            if _ce is not None and _ce.is_set():
                                _cancelled = True
                                for abs_path in paths:
                                    out[abs_path] = SemanticOutcome(
                                        skip_reason="cancelled before the semantic check ran",
                                    )
                                break
                    if _cancelled:
                        continue
            except Exception as exc:  # advisory, never blocks
                # One provider's failure must not cost the others their
                # diagnostics, so this is caught per group, not per drain.
                logger.debug("Deferred semantic checks failed for %s: %s", paths, exc)
                for abs_path in paths:
                    out[abs_path] = SemanticOutcome(
                        skip_reason="the semantic checker raised before reporting",
                    )
                continue
            for abs_path in paths:
                sem = results.get(abs_path)
                if sem is None:
                    continue
                if not getattr(sem, "checked", True):
                    # No tool examined this file (not installed, timed out, no
                    # project config). An empty diagnostic list here would be
                    # rendered as "checked, clean" — the miscue this whole
                    # design exists to avoid — so the skip travels as a skip.
                    out[abs_path] = SemanticOutcome(
                        skip_reason=(getattr(sem, "skip_reason", "") or "the checker did not run"),
                    )
                    continue
                out[abs_path] = SemanticOutcome(
                    diagnostics=[
                        {
                            "file_path": abs_path,
                            "line": e.line,
                            "col": e.col,
                            "message": e.message,
                            "severity": getattr(e, "severity", "error"),
                            "code": getattr(e, "code", ""),
                        }
                        for e in (sem.errors or [])
                    ]
                )
        return out

    @property
    def repo_language(self) -> LanguageId | None:
        """Dominant repo language by source-file count, or None if Python/mixed/unknown.

        ``None`` means all tools are visible (Python-only tools not masked). A
        concrete non-Python LanguageId means the repo is a pure non-Python repo
        and Python-only tools (``edit_ast``, ``run_structural_scan``) are hidden.
        """
        return self._repo_language

    @staticmethod
    def _make_tool_result_cache(config: AgentConfig) -> Any | None:
        """Build a fresh, ISOLATED ToolResultCache from ``config`` (or None).

        Shared by ``__init__`` and ``clone_for_subagent`` so every registry
        instance that opts in gets its OWN cache. Sharing a single cache across the parent and concurrent
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
        # getattr default matches the `tool_result_cache_enabled` field default (True), NOT
        # False: a duck-typed config that omits the attribute must get the
        # DOCUMENTED behavior (cache enabled), not silently fail-closed to
        # disabled. Real AgentConfig instances always carry the field.
        if not getattr(config, "tool_result_cache_enabled", True):
            return None
        try:
            from .tool_result_cache import ToolResultCache

            cache = ToolResultCache(
                max_entries=getattr(config, "tool_result_cache_max_entries", 256),
                default_ttl=getattr(config, "tool_result_cache_ttl", 120),
            )
            get_global_collector().register_tool_result_cache(cache)
        except Exception as e:
            logger.warning("Failed to initialize tool result cache: %s", e)
            return None
        else:
            return cache

    def clone_for_subagent(self, sub_config: AgentConfig) -> ToolRegistry:
        """Create a lightweight clone sharing expensive resources.

        Shared (immutable/thread-safe): SymbolSearcher, RAGSearcher,
        CallGraphIndexer, LintRunner.
        Fresh (per-subagent mutable state): _applied_patches,
        _search_cache, config, async/watcher (disabled for subagents).
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
        clone._arg_repairer = self._arg_repairer
        clone._safety_manager = self._safety_manager

        # Fresh mutable state per subagent
        clone._applied_patches = []
        clone._search_cache = {}

        # Fresh, ISOLATED cache (NOT shared with the parent, NOT None). A null
        # cache threw away the most common subagent win — repeated read_file of
        # the same path. Each clone gets its own cache via the shared helper so
        # concurrent subagents don't race on LRU/TTL state.
        clone._tool_result_cache = self._make_tool_result_cache(sub_config)
        clone.local_assistant = None

        # SHARED (not reset): the checkpoint gate is per-run, not per-agent. A
        # subagent's writes belong to the same Undo point as the parent's.
        clone._run_checkpoint_gate = self._run_checkpoint_gate

        # Copy override state (if any); __init__ is bypassed via object.__new__
        clone._repo_root_override = getattr(self, "_repo_root_override", None)
        # Fresh resolved-root memo (per-instance, NOT shared with the parent)
        clone._secure_root_resolve_cache = {}

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
        # Agent profile: sub_config's explicit profile wins (mirrors __init__'s
        # "explicit param > config" contract). Orchestrator's replace() copies
        # base.agent_profile into sub_config, so an unmodified sub_config simply
        # re-copies the parent's — but a subagent-specific profile set on the
        # config was previously SILENTLY IGNORED and the parent's enforced.
        # `is not None` (not truthiness): matches the `__init__` `agent_profile` fallback.
        sub_profile = getattr(sub_config, "agent_profile", None)
        clone._agent_profile = sub_profile if sub_profile is not None else getattr(self, "_agent_profile", None)
        # Semantic-lint coalescing — FRESH per clone (never shared — concurrent
        # subagent writes must NOT co-accumulate into the parent's batch).
        clone._semantic_pending = {}
        clone._semantic_turn_active = False

        return clone

    def _invalidate_cache_after_write(self, touched_paths: list[str]) -> None:
        """Invalidate call graph, RAG, and graph caches for touched paths (called after patch apply)."""
        # Normalize every touched path to repo-relative form ONCE, up front.
        # Callers disagree on form: _snapshot_target_files (the semantic-write
        # snapshotter) builds ABSOLUTE paths via os.path.join(repo_root,
        # target), while the patch mixin's touched/written lists are
        # repo-relative. The three incremental invalidators below
        # (CallGraphIndexer, RAGSearcher, GraphFacade) all assume relative —
        # each does its own strip().lstrip("/") and then re-joins against the
        # repo root — so an absolute path survives that lstrip as
        # "Users/.../foo.py" and the re-join points at a path that does not
        # exist: the invalidation silently no-ops and the index keeps
        # answering with pre-write state (regression introduced when the
        # semantic-write path moved from the path-agnostic invalidate() to
        # invalidate_files(touched_paths)). Paths outside the repo are kept
        # as-is so consumers' isfile checks treat them as no-ops instead of
        # mis-resolving them into the repo.
        _root = os.path.realpath(self._effective_repo_root)
        _normalized: list[str] = []
        for _p in touched_paths:
            _p = str(_p).strip()
            if not _p:
                continue
            if os.path.isabs(_p):
                try:
                    # realpath BOTH sides: on macOS repo_root is resolved to
                    # /private/var/... while a caller-supplied path can still
                    # read /var/... — relpath across that alias yields a
                    # ".."-prefixed junk path, which would wrongly fall
                    # through to the keep-as-is branch below.
                    _rel = os.path.relpath(os.path.realpath(_p), _root)
                except ValueError:  # different drive (Windows)
                    _rel = _p
                _normalized.append(_rel if not _rel.startswith("..") else _p)
            else:
                _normalized.append(_p.lstrip("/"))
        touched_paths = _normalized

        # Incrementally update the call graph index for touched files (much
        # faster than full rebuild — only the changed files are re-parsed;
        # node ownership reassignment follows the same first-definition-wins
        # rule as a full build).  The unknown-scope bash-mutation path below
        # keeps the wholesale invalidate(): parsing arbitrary shell for
        # target paths is the classifier trap, so a full clear is the only
        # safe choice there.
        from ..languages import LanguageId as _LId

        if any(_LId.from_path(p) != _LId.UNKNOWN for p in touched_paths) and hasattr(self, "_call_graph"):
            cgi = getattr(self._call_graph, "call_graph_indexer", None)
            if cgi is not None:
                cgi.invalidate_files(touched_paths)

        # Incrementally update RAG index for touched files (much faster than full rebuild)
        if hasattr(self, "_rag_searcher") and self._rag_searcher:
            try:
                self._rag_searcher.invalidate_files(touched_paths)
            except Exception as e:
                logger.debug("Failed to incrementally update RAG index: %s", e)

        # Incrementally update GSG graph for touched Python files
        if hasattr(self, "_call_graph") and self._call_graph:
            try:
                self._call_graph.invalidate_files(touched_paths)
            except (AttributeError, TypeError) as e:
                logger.debug("post-write invalidation: GSG graph failed: %s", e)

        # Invalidate per-root file-walk caches so newly created files are
        # immediately visible to find_symbol / call-graph rebuilds.
        try:
            from external_llm.agent._shared_utils import invalidate_walk_caches

            invalidate_walk_caches()
        except Exception as e:
            # Non-critical — never block execution. Logged because a failure
            # here is the "cannot find the symbol it just wrote" class of bug,
            # which is exactly what a silent handler makes unattributable.
            logger.debug("post-write invalidation: walk caches failed: %s", e)

        # Same reason, for the repo file LISTING (git ls-files). It backs the
        # `glob` tool and the "Did you mean:" path suggester, and was TTL-only:
        # a file created this turn stayed invisible to glob for a full 60 s
        # while find_symbol — fixed above — already saw it. Scope IS known here
        # (unlike _invalidate_caches_unknown_scope), so invalidate per touched
        # path instead of dropping the whole index: atomic writers
        # (edit_text/edit_file/anchor_edit/write_plan/...) already popped and
        # gen-bumped via the atomic funnel (atomic_io -> invalidate_for_written_path),
        # making this a cheap no-op for them, while the per-path call still
        # covers the non-atomic writer (apply_patch's git-apply subprocess),
        # which bypasses the funnel entirely.
        try:
            from external_llm.common.repo_files import invalidate_for_written_path

            for _p in touched_paths:
                _abs = _p if os.path.isabs(_p) else os.path.join(self._effective_repo_root, _p)
                invalidate_for_written_path(_abs)
        except Exception as e:
            logger.debug("post-write invalidation: repo file index failed: %s", e)

        # Same reason, for the non-Python symbol caches. Without this the
        # "cannot find a symbol in code it just wrote" symptom above persisted
        # for every non-Python language (measured: a new Go func stayed invisible
        # to find_symbol for the full 30 s TTL while an equivalent Python edit was
        # visible immediately). Scoped to non-Python touches because those caches
        # only ever index non-Python files, so a .py write cannot affect them —
        # and clearing them would cost a needless re-walk on the common path.
        try:
            if any(not p.endswith((".py", ".pyi")) for p in touched_paths):
                self._symbol_searcher.invalidate_nonpy_caches()
        except Exception as e:
            logger.debug("post-write invalidation: non-Python caches failed: %s", e)

        # The mirror image, for the Python find_symbol prefilter memo. Scoped to
        # Python touches for the same reason the block above is scoped to
        # non-Python ones: the memo only ever holds `rg --type py` answers, so a
        # .go write cannot stale it. Unscoped it would drop a live memo on every
        # write and give back the spawn it exists to avoid.
        try:
            if any(p.endswith((".py", ".pyi")) for p in touched_paths):
                from external_llm.agent.symbol_search import (
                    invalidate_py_prefilter_cache,
                )

                invalidate_py_prefilter_cache()
        except Exception as e:
            logger.debug("post-write invalidation: Python prefilter memo failed: %s", e)

        # The per-file symbol maps (_py_file_cache / _ts_file_cache). Unlike the
        # two blocks above these are path-keyed, so they are dropped for exactly
        # the touched files and need no language scoping. They were absent from
        # this method entirely: both key on a (mtime_ns, size) signature, which
        # detects a change only where the filesystem records one — a coarse-mtime
        # mount or an mtime-preserving restore (tar, rsync -t, cp -p) collides and
        # find_symbol then reports pre-edit LINE NUMBERS for a file just written.
        try:
            self._symbol_searcher.invalidate_file_caches(touched_paths)
        except Exception as e:
            logger.debug("post-write invalidation: per-file symbol maps failed: %s", e)

    def _invalidate_caches_unknown_scope(self) -> None:
        """Post-write invalidation for a mutating call whose targets are unknown.

        ``_invalidate_cache_after_write`` needs the touched paths, so it is only
        reachable from the write TOOLS. ``bash`` mutates just as often — the
        agent's own no-tool nudge literally instructs it to create files with
        ``bash('cat > path << EOF ...')`` — but carried none of it: a successful
        mutating bash cleared the tool-result cache and nothing else. Measured:
        a .py AND a .go file created by bash were both invisible to find_symbol
        afterwards, which is the same "cannot find a symbol in code it just
        wrote" symptom the write-tool path was fixed for.

        Scope is unknown here (parsing arbitrary shell for target paths is the
        classifier trap all over again), so this clears wholesale — the same
        choice the tool-result cache already makes for bash one level up
        ("falls back to a full clear — safer than guessing scope").

        RAG caches are path-keyed with no clear-all and are deliberately left
        alone: they rank relevance rather than answer "does this symbol exist",
        so staleness there degrades ordering, not correctness.

        The facade's own RG graph is NOT covered by the CGI invalidate above:
        ``cgi.invalidate()`` drops the CallGraphIndexer, but the RepositoryGraph
        held inside the facade (``_graph``) is a separate build serving
        get_symbol / get_importers / get_file_dependencies / get_symbols_in_file.
        Without dropping it, a bash-created file is invisible to those queries
        until the next lazy rebuild — the same "cannot find a symbol in code it
        just wrote" class this method exists for. The write-tool path already
        covers the facade (``_call_graph.invalidate_files``), so this is the
        missing mirror for unknown scope.
        """
        try:
            if hasattr(self, "_call_graph") and self._call_graph:
                self._call_graph.invalidate()
        except Exception as e:
            logger.debug("unknown-scope invalidation: facade graph invalidate failed: %s", e)
        try:
            cgi = getattr(getattr(self, "_call_graph", None), "call_graph_indexer", None)
            if cgi is not None:
                cgi.invalidate()
        except Exception as e:
            logger.debug("unknown-scope invalidation: call-graph invalidate failed: %s", e)
        try:
            from external_llm.agent._shared_utils import invalidate_walk_caches

            invalidate_walk_caches()
        except Exception as e:
            logger.debug("unknown-scope invalidation: walk cache pop failed: %s", e)
        try:
            self._symbol_searcher.invalidate_nonpy_caches()
        except Exception as e:
            logger.debug("unknown-scope invalidation: non-Python caches failed: %s", e)
        try:
            # Unconditional here, unlike the write-tool path: scope is unknown,
            # and `bash` creating a .py file is the exact case this method was
            # added for.
            from external_llm.agent.symbol_search import invalidate_py_prefilter_cache

            invalidate_py_prefilter_cache()
        except Exception as e:
            logger.debug("unknown-scope invalidation: Python prefilter memo failed: %s", e)
        try:
            # No paths to scope by here, so this clears both per-file symbol
            # maps wholesale — the same wholesale choice the rest of this
            # method makes, and cheap because they refill per file on demand.
            self._symbol_searcher.invalidate_file_caches()
        except Exception as e:
            logger.debug("unknown-scope invalidation: per-file symbol maps failed: %s", e)
        try:
            from external_llm.agent.tool_handlers.write_tools import (
                invalidate_repo_file_index,
            )

            invalidate_repo_file_index(self._effective_repo_root)
        except Exception as e:
            logger.debug("unknown-scope invalidation: repo file index failed: %s", e)

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

    def _gate_check(self, tool_name: str, args: dict) -> ToolResult | None:
        rejection = self._safety_manager.gate_check(tool_name, args, self.config.approval_callback)
        if rejection is None:
            return None
        return ToolResult(
            ok=False,
            content="",
            error=rejection["error"],
            metadata=rejection["metadata"],
        )

    # Tools that write to files — need per-file locking in multi-agent mode.
    # All write tools get snapshot + syntax verify + rollback safety wrapper.
    # Derived from the WRITE_TOOL_NAMES SSOT (agent_loop_types) so this stays
    # in lockstep with TurnContext.write_tools across all six mechanisms
    # (locking, failure-logging, cache invalidation, reads_since_last_edit
    # reset, write_tool_used detection, test-impact invalidation).
    _WRITE_TOOLS: ClassVar[set[str]] = set(WRITE_TOOL_NAMES)
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
        "find_symbol",
        "find_references",
        "find_tests_for_symbol",
        "find_relevant_files",
        "get_file_outline",
        "analyze_change_impact",
        "query_dependency_graph",
        "run_structural_scan",
        # Symbol / file read / search
        "read_symbol",
        "read_file",
        "grep",
        "glob",
        "read_image",
        # Web search/fetch — read-only network lookups; cached under the same TTL/LRU
        # as the others. Scope is frozenset() (no repo-file dependency), so a
        # write-tool success does NOT drop them — only TTL/LRU/clear do. Repeated
        # identical queries within a turn still hit.
        "search_web",
        "web_fetch",
    }

    def is_result_cacheable(self, tool_name: str) -> bool:
        """Whether ``tool_name`` will actually be probed against the result cache.

        Mirrors the cache-lookup guard in :meth:`_dispatch_impl` EXACTLY:
        ``self._tool_result_cache is not None AND tool_name in _READ_ONLY_TOOLS``.
        Membership in ``_READ_ONLY_TOOLS`` alone is NOT sufficient — when the cache
        is disabled (``tool_result_cache_enabled=False`` → ``_tool_result_cache`` is
        None), read-only tools are never probed, so a recorder that sees only the
        set membership would classify them as "missed" and pin their per-tool
        ``cache_hit_rate`` at a fake 0% (the very contamination the 3-state
        ``cache_hit`` recording was built to prevent). Checking the live cache
        reference makes this the true "was it probed?" SSOT shared by
        :meth:`_dispatch_impl` and both recording sites (``dispatch`` here and
        ``agent_turn_pipeline``), so they can never disagree.

        Exposed as a method (not a private peek) so the recorders classify a
        tool's cache outcome as hit / miss / **non-probed** (3-state) without
        duplicating the guard: a non-probed tool emits ``None`` (neither hit nor
        miss), keeping its per-tool rate honest.
        """
        return self._tool_result_cache is not None and tool_name in self._READ_ONLY_TOOLS

    # ── bash cache-invalidation heuristic ──────────────────────────────
    # Read-only bash commands (ls, cat, git log, grep, find, ...) never mutate
    # the filesystem, so they should NOT wipe the read-tool result cache.
    # Only commands that actually write/move/remove files or change source state
    # require cache invalidation. This keeps the cache effective across the very
    # common pattern of the model running `git status` / `grep` between edits.
    #
    # Static prefixes that are unconditionally read-only (whitelist, fastest path).
    _BASH_READONLY_PREFIXES = (
        "ls ",
        "cat ",
        "head ",
        "tail ",
        "less ",
        "more ",
        "grep ",
        "rg ",
        "find ",
        "fd ",
        "locate ",
        "wc ",
        "du ",
        "df ",
        "file ",
        "stat ",
        "md5sum ",
        "sha256sum ",
        "diff ",
        "git log",
        "git status",
        "git diff",
        "git show",
        "git rev-parse",
        "git remote -v",
        "git config --get",
        "git blame",
        "git ls-files",
        "git ls-tree",
        "git count-objects",
        "pwd",
        "whoami",
        "hostname",
        "uname",
        "echo ",
        "printf ",
        "which ",
        "command -v",
        "type ",
        "printenv",
        # NOT here, deliberately: "python -c", "python3 -c", "node -e". They were
        # whitelisted as "introspection only when via -c (no pip/install)", but -c
        # and -e run ARBITRARY code — `python3 -c "open('f','w').write(...)"`
        # rewrites a file while classifying as read-only. That is the single most
        # ambiguous command shape there is, and this classifier's stated policy is
        # that ambiguous defaults to mutating. `node --check` stays: it only parses.
        "node --check ",
        "pytest --collect-only",
        "pytest -q --co",
        "ruff check",
        "ruff --version",
        "test ",
        "[ ",
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
        "tee ",  # writes stdin to a file
        "rm ",
        "rmdir",
        "mv ",
        "cp ",  # remove/move/copy files
        "mkdir",
        "touch ",  # create files/dirs
        "chmod",
        "chown",  # metadata changes
        "sed -i",
        "perl -i",  # in-place file edits
        " -delete",
        " -exec",  # find ... -delete / -exec: mutate despite "find " read-only prefix
        "git add",
        "git rm",
        "git mv",
        "git commit",
        "git checkout",
        "git reset",
        "git pull",
        "git push",
        "git merge",
        "git rebase",
        "git restore",
        "git clean",
        "pip install",
        "pip uninstall",
        "pip3 install",
        "pip3 uninstall",
        "npm install",
        "npm uninstall",
        "npm ci",
        "yarn add",
        "yarn remove",
        "pnpm add",
        "pnpm remove",
        "apply_patch",
        "patch -p",
        "curl -o",
        "wget -o",
        "wget -O",  # download to file
        "tar -x",
        "unzip ",
        "gunzip ",  # extract (creates files)
    )

    # `git branch` argument forms that only *query* branch state (no create/
    # delete/rename). Any other suffix (bare create-a-new-branch, -D/-d, -m/-M,
    # …) mutates. Checked as a dedicated case because a blanket "git branch"
    # prefix would also whitelist `git branch -D x` (deletes a branch).
    _GIT_BRANCH_READONLY_ARGS = (
        "--list",
        "-l",
        "-a",
        "-r",
        "-v",
        "-vv",
        "--contains",
        "--no-color",
        "--color",
        "--sort",
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
            if c in {"'", '"'}:
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
        duplication/closure (``n>&m``, ``>&m``, ``n>&-``) or a known null / fd
        sink (``2>/dev/null``, ``2>/dev/stdout``, ``2>/dev/stderr``,
        ``2>/dev/fd/N``) rather than a file write. Append forms of the sinks
        (``>>/dev/null``, ``2>>/dev/null``, ``&>>/dev/null``) count too.

        tree-sitter-bash tags BOTH real file redirects (``> f``, ``2>err``,
        ``&>all``) and fd-dups (``2>&1``) as ``file_redirect`` nodes, so the node
        type alone cannot tell them apart. The distinguishing token is an ``&``
        immediately after the ``>`` whose target is a file-descriptor number (or
        ``-`` for close) — never a path.  Additionally, ``/dev/null``,
        ``/dev/stdout``, ``/dev/stderr``, and ``/dev/fd/N`` are known sinks that
        touch no real file on disk. Applied to a PARSED node body, quoting is
        already resolved by the grammar, so no quote tracking is needed here
        (unlike the raw-command scanner).
        """
        gt = redirect_text.find(">")
        if gt < 0:
            return False
        after = redirect_text[gt + 1 :].lstrip()
        # ``find`` returns the FIRST ``>``, so on an APPEND redirect the second
        # one is still sitting in *after* (``>>/dev/null`` -> ``>/dev/null``) and
        # no sink below would match. Drop it, so the append forms (``>>``,
        # ``2>>``, ``&>>``) classify exactly like their truncating twins. This
        # cannot mask a real write: ``>>out.txt`` simply becomes ``out.txt``,
        # which is no sink, and ``>&2`` never enters here (no leading ``>``).
        if after.startswith(">"):
            after = after[1:].lstrip()
        if not after:
            return False
        # fd duplication/closure: n>&m, >&m, n>&-
        if after.startswith("&"):
            rest = after[1:]
            return bool(rest) and (rest[0].isdigit() or rest[0] == "-")
        # Known null/fd sinks — stderr discard, stdout/stderr redirection, fd
        # pass-through.  These touch no real file on disk.
        return bool(after in ("/dev/null", "/dev/stdout", "/dev/stderr") or after.startswith("/dev/fd/"))

    @classmethod
    def _parse_bash_tree(cls, command: str):
        """Parse *command* with tree-sitter-bash — shared bootstrap for the
        structural bash classifiers.

        Returns the parse tree, or None when tree-sitter-bash is unavailable or
        *command* does not parse cleanly (``root_node.has_error``). Callers treat
        None as "fall back to the conservative text heuristic", so keeping the
        availability/parse contract in ONE place guarantees both classifiers
        agree on when the structural path is usable.
        """
        try:
            from ..languages import tree_sitter_utils as _ts_utils

            if not _ts_utils.is_available():
                return None
            _parser = _ts_utils.get_parser("bash")
        except Exception as _e:
            logger.debug("tree-sitter-bash bootstrap failed: %s", _e)
            return None
        if _parser is None:
            return None
        try:
            _tree = _parser.parse(bytes(command, "utf8"))
        except Exception as _e:
            logger.debug("tree-sitter-bash parse failed (%.100r): %s", command, _e)
            return None
        if _tree is None or _tree.root_node.has_error:
            return None
        return _tree

    @classmethod
    def _walk_bash_nodes(cls, command: str, node_type: str, _tree=None) -> list | None:
        """Return the text of every tree-sitter-bash node of *node_type* in *command*.

        Shares the bootstrap/parse contract of :meth:`_parse_bash_tree` and the
        always-descend DFS traversal used by both structural classifiers:
        matching nodes may be nested inside command substitution / a subshell
        / a loop body, so every child is visited regardless of depth.

        Returns ``None`` when tree-sitter-bash is unavailable or *command*
        does not parse cleanly (callers fall back to their conservative text
        path), ``[]`` when the tree is fine but no node of *node_type* exists,
        and the list of node texts otherwise.

        ``_tree`` — optional pre-parsed tree from :meth:`_parse_bash_tree`,
        supplied by :meth:`_bash_command_mutates_files` so one command is
        parsed exactly once per classification. When supplied it MUST have
        been parsed from *command* itself (never a stripped copy) — node byte
        offsets slice into *command* below. ``None`` (default) re-parses,
        keeping standalone callers working.
        """
        if _tree is None:
            _tree = cls._parse_bash_tree(command)
        if _tree is None:
            return None
        _texts: list[str] = []
        _stack = [_tree.root_node]
        while _stack:
            _node = _stack.pop()
            if _node.type == node_type:
                _texts.append(command[_node.start_byte : _node.end_byte])
            if _node.children:
                _stack.extend(reversed(_node.children))
        return _texts

    @classmethod
    def _has_file_redirect_via_ts(cls, command: str, _tree=None):
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

        ``_tree`` — optional pre-parsed tree from :meth:`_parse_bash_tree`
        (see :meth:`_walk_bash_nodes` for the parse-once contract).
        """
        _nodes = cls._walk_bash_nodes(command, "file_redirect", _tree)
        if _nodes is None:
            return None
        return any(not cls._redirect_is_fd_dup(_text) for _text in _nodes)

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
            rest = segment[len("git stash") :].strip()
            return rest.startswith(("list", "show"))
        if segment == "git branch" or segment.startswith("git branch "):
            rest = segment[len("git branch") :].strip()
            return rest == "" or any(rest.startswith(a) for a in cls._GIT_BRANCH_READONLY_ARGS)
        if segment == "env" or segment.startswith("env "):
            # Bare `env` (optionally with VAR=VAL assignments) prints the
            # environment and is read-only. `env <cmd> ...` RUNS <cmd>, so a plain
            # prefix match would whitelist an arbitrary command — the same hole
            # `python -c` had. Options (-i, -u FOO) fall to mutating: conservative,
            # and only costs a cache miss.
            rest = segment[len("env") :].strip()
            return all("=" in tok for tok in rest.split()) if rest else True
        return any(segment.startswith(prefix) or segment == prefix.rstrip() for prefix in cls._BASH_READONLY_PREFIXES)

    @classmethod
    def _bash_command_segments_via_ts(cls, command: str, _tree=None):
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

        ``_tree`` — optional pre-parsed tree from :meth:`_parse_bash_tree`
        (see :meth:`_walk_bash_nodes` for the parse-once contract).
        """
        _segments = cls._walk_bash_nodes(command, "command", _tree)
        # No command node at all (bare comment / env-only assignment) → defer
        # to the fallback rather than treating it as "all read-only".
        if not _segments:
            return None
        return _segments

    @classmethod
    @functools.lru_cache(maxsize=256)
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

           Pure text classifier over immutable class constants — results are cached
           (``functools.lru_cache(maxsize=256)``): the same command string is
           re-classified on every dispatch (read-only bash like ``git status`` runs
           between edits), and tree-sitter parsing is the dominant cost. Tests that
           monkeypatch a sub-classifier must clear the cache first via
           ``ToolRegistry._bash_command_mutates_files.cache_clear()``.
        """
        if not command:
            return False
        stripped = command.strip()

        # 1. Redirect / write-token scan on the WHOLE command — unconditionally
        #    first, so a mutating suffix/chain/redirect is never masked by a
        #    read-only-looking prefix or subcommand earlier in the string.
        # Parse ONCE and share the tree between the two structural classifiers
        # (N1: the tree MUST be built from the ORIGINAL `command`, not `stripped`
        # — both classifiers slice node byte offsets into the string the tree
        # was parsed from; a stripped-based tree would mis-slice every node).
        _tree = cls._parse_bash_tree(command)
        _ts_redirect = cls._has_file_redirect_via_ts(command, _tree=_tree)
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
        segments = cls._bash_command_segments_via_ts(command, _tree=_tree)
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
        # kill mutates process state; can race with concurrent job output
        return tool_name == "job" and (args or {}).get("action") == "kill"

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

    def _resolve_repo_scope(self, path_arg: Any) -> frozenset | None:
        """Resolve a path arg to an absolute in-repo scope, or None.

        Returns None for a blank arg, a bare repo root (repo-wide), or a path
        that escapes ``self.repo_root``. The escape guard mirrors
        ``SymbolSearcher._resolve_search_root`` (which rejects out-of-repo
        paths) — important because ``find_references`` falls back to a
        repo-wide rg when its search_path is rejected
        (``_resolve_search_root(...) or self.repo_root``), so scoping such a
        call to the rejected path would let a later in-repo write leave a
        stale cache entry.
        """
        p = path_arg.strip() if isinstance(path_arg, str) else ""
        if not p or p in (".", self.repo_root):
            return None
        full = os.path.normpath(p if os.path.isabs(p) else os.path.join(self.repo_root, p))
        root = os.path.normpath(self.repo_root)
        if full != root and not full.startswith(root + os.sep):
            return None  # escaped the repo — not a usable scope
        return frozenset({full})

    def _extract_read_scope_paths(self, tool_name: str, args: dict) -> frozenset | None:
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
        # glob shares grep's shape: an optional `path` narrows the scope, its
        # absence means repo-wide. Both must report "unknown scope" when
        # unscoped so a write anywhere invalidates the cached listing — a glob
        # result goes stale the moment a matching file is created or deleted.
        if tool_name in ("grep", "glob"):
            p = args.get("path")
            p = p.strip() if isinstance(p, str) else ""
            if p and p not in (".", self.repo_root):
                full = p if os.path.isabs(p) else os.path.join(self.repo_root, p)
                return frozenset({os.path.normpath(full)})
            return None  # no path filter → repo-wide search, unknown scope

        # ── Symbol search tools ────────────────────────────────────────────
        # read_symbol(file_path=) and find_symbol/find_references(search_path=)
        # narrow their walk to one file/subtree, so a cached result depends
        # only on files under it — same property grep/glob enjoy above, and the
        # same reason a write elsewhere must NOT evict it. Three exclusions
        # keep this from going stale (see _resolve_repo_scope):
        #  • no path arg → repo-wide search → unknown scope (None)
        #  • find_symbol + include_inheritance → get_symbol_info enriches with
        #    subclasses/refs found ANYWHERE in the repo → repo-wide (None)
        #  • a path escaping the repo → find_references falls back to a
        #    repo-wide rg, so it can't be scoped to the rejected path.
        if tool_name == "read_symbol":
            return self._resolve_repo_scope(args.get("file_path"))
        if tool_name in ("find_symbol", "find_references"):
            if tool_name == "find_symbol" and args.get("include_inheritance"):
                return None
            return self._resolve_repo_scope(args.get("search_path"))
        # Network lookups (search_web/web_fetch) depend on NO repo file — a
        # file write cannot make them stale. Report an EMPTY scope (not None):
        # invalidate_paths() keeps empty-scope entries (any() over no paths is
        # False) while unknown-scope (None) entries are still conservatively
        # dropped on every write.
        if tool_name in ("search_web", "web_fetch"):
            return frozenset()
        return None

    def _extract_write_target_paths(self, tool_name: str, args: dict) -> frozenset | None:
        """Best-effort absolute target path(s) for a write-tool call, so cache
        invalidation can drop only overlapping entries. Returns None when the
        target can't be determined (caller should fall back to a full clear()).

        Path-only (no file I/O). Extraction itself is delegated to
        ``write_targets.write_target_paths``, the single source of truth shared
        with the rollback snapshot, the approval gate and the file-lock manager
        — see that module for why four private copies of this logic drifted and
        what each of them missed.
        """
        targets = write_target_paths(tool_name, args)
        if not targets:
            return None  # unknown scope → caller falls back to full clear()

        return frozenset(os.path.normpath(t if os.path.isabs(t) else os.path.join(self.repo_root, t)) for t in targets)

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
                logger.debug("write safety: could not read %s after patch", path)
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
                logger.debug("write safety: no failure classifier for %s", lang)
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

                # Strategies consume a Classification (typed symbol extraction),
                # NOT the classifier object itself — passing `classifier` here
                # would make every `.symbol` / `.fix_hint` access in the
                # strategy body fail with AttributeError.
                classification = classifier.classify_typed([verr])
                ops = strategy(current_code, verr, classification)
                if ops is None:
                    continue

                # Apply repair
                if len(ops) == 1 and "__raw_code__" in ops[0].payload:
                    repaired_code = ops[0].payload["__raw_code__"]
                else:
                    continue  # Only raw replacements supported for now

                try:
                    atomic_write_text(path, repaired_code)
                except OSError:
                    logger.debug("write safety: could not write repaired %s", path)
                    continue

                # Re-verify
                re_val = provider.validate_syntax(path, repaired_code)
                if re_val.ok:
                    current_code = repaired_code
                    _repaired_any = True
                    logger.info(
                        "Write safety: repaired argument mismatch in %s (error: %s)",
                        path,
                        verr.message[:80],
                    )
                    # Code is now clean — stop processing this file
                    break
                # Restore current_code on failure
                with contextlib.suppress(OSError):
                    atomic_write_text(path, current_code)

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
        _detail_path_match = re.match(r"^([^:]+):(\d+):(\d+): ", verify_detail)
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
            # _origin is only ever set together with a syntax-capable
            # _provider; assert to narrow for the call below.
            assert _provider is not None
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
                    _o_path,
                    verify_detail,
                )
                return True

        try:
            classifier = create_failure_classifier(_lang)
        except ValueError:
            return False

        ftype = classifier.classify(
            [
                VerifyError(message=verify_detail, line=0, column=0),
            ]
        )

        # SYNTAX_ERROR and UNKNOWN are hard failures — always rollback;
        # all other recognizable errors (ARGUMENT_MISMATCH, TYPE_MISMATCH,
        # MISSING_RETURN, MISSING_VARIABLE, etc.) are cross-op fixable
        return ftype not in (FailureType.SYNTAX_ERROR, FailureType.UNKNOWN)

    # _reindent_text → imported from external_llm.common.indent_utils.reindent_text

    @staticmethod
    def _auto_repair_indent(original_content: str, operations: list) -> str | None:
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
                line_start = fixed.rfind("\n", 0, idx) + 1
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
                fixed = fixed[:line_start] + adjusted + fixed[idx + len(anchor) :]

            else:  # insert_after
                # Get anchor line's indentation to use as target.
                line_start = fixed.rfind("\n", 0, idx) + 1
                nl_pos = fixed.find("\n", idx)
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
                    fixed = fixed + "\n" + adjusted.rstrip("\n") + "\n"
                else:
                    fixed = fixed[: eol + 1] + adjusted.rstrip("\n") + "\n" + fixed[eol + 1 :]

        return fixed if fixed != original_content else None

    def _after_write_success(self, tool_name: str, args: dict, result: ToolResult, snapshots: dict) -> None:
        """Central post-success processing for a call whose changes are on disk.

        Runs semantic auto-repair, post-write cache invalidation
        (file/walk/symbol/RAG/graph + tool-result), Undo-checkpoint
        confirmation and git-snapshot clearing in ONE place so every
        disk-changing success path in :meth:`_dispatch_impl` — the normal tail
        AND the repair/soft-fail early returns — obeys the same post-success
        contract. The early returns were the gap: the edit_file indent-repair
        success, the argument-mismatch repair success and the soft-fail
        keep-changes return all wrote to disk before returning while skipping
        this block entirely — the same stale-cache class as the apply_patch
        incident documented below, plus a skipped checkpoint confirmation that
        left run-created files without an Undo tombstone.

        The rollback path (``ok=False``, disk restored) must NOT call this:
        the restored state is the pre-write state the caches already hold, and
        a refused write must not leave a tombstone behind for Undo to act on.
        """
        # ── Phase 1: expose the run's Undo checkpoint id to callers ──
        # Lets MCP/webapp callers learn "this turn's writes can be Undone"
        # without reaching into the registry (run_checkpoint_id property).
        # Set on EVERY disk-changing success path — the normal tail AND the
        # repair/soft-fail early returns all funnel through here — while the
        # rollback path never reaches this function, so a refused write
        # carries no checkpoint metadata.
        try:
            result.metadata = dict(result.metadata or {})
            _cid = self.run_checkpoint_id
            if _cid is not None:
                result.metadata["checkpoint_id"] = _cid
        except Exception:
            logger.debug("checkpoint_id metadata failed", exc_info=True)

        # ── Phase 2: deterministic semantic auto-repair (F401/F821) ──
        # Runs after syntax verify passes (or after soft-fail preserves changes).
        # Auto-fixes undefined names (F821 via project-wide import search;
        # F401 unused-import auto-fix is intentionally disabled — surfaced as a
        # soft warning only). Non-fatal: any failure here degrades gracefully.
        # Self-validating tools (edit_text/edit_ast/anchor_edit) skip the
        # syntax-verify snapshot in dispatch (redundant I/O), but they can
        # still *introduce* undefined names (F821) — e.g. edit_ast inserting a
        # reference to a symbol whose import is missing. Semantic auto-repair
        # is orthogonal to syntax verify, so build an on-demand snapshot here
        # when the verify-path snapshot was skipped.
        if result.ok and tool_name in self._WRITE_TOOLS:
            _sem_snapshots = snapshots
            if not _sem_snapshots:
                try:
                    _sem_snapshots = self._snapshot_target_files(tool_name, args)
                except Exception:
                    _sem_snapshots = {}
            if _sem_snapshots:
                try:
                    _sem_repaired = self._safety_manager.auto_repair_semantic(_sem_snapshots)
                    if _sem_repaired > 0:
                        logger.info(
                            "[AUTO-REPAIR] Write safety: auto-repaired %d semantic finding(s)",
                            _sem_repaired,
                        )
                        result.metadata["semantic_repaired"] = _sem_repaired
                except Exception as _sem_exc:
                    logger.debug("Semantic auto-repair error: %s", _sem_exc, exc_info=True)

                # ── Post-write cache invalidation (PARITY across write tools) ──
                # _invalidate_cache_after_write was reachable from exactly TWO
                # handler-internal call sites (write_plan, and one apply_patch
                # branch guarded by `if touched:`), while _WRITE_TOOLS has seven
                # members. Measured: a successful apply_patch — new file AND
                # existing file — invoked it ZERO times, so the file cache, call
                # graph, RAG index and per-root walk caches all kept serving
                # pre-write state until their TTLs expired. The agent then could
                # not find a symbol in code it had just written (find_symbol
                # answered "No definitions found" for a function on disk).
                #
                # Invalidating HERE, at the same central post-success point the
                # semantic auto-repair above already uses for the same parity
                # reason, makes it structurally impossible for a write tool to
                # skip it — including the three self-validating tools
                # (edit_text/edit_ast/anchor_edit) that never had a call at all.
                # The two handler-internal calls remain and are harmless: every
                # step of _invalidate_cache_after_write is idempotent.
                try:
                    self._invalidate_cache_after_write(sorted(_sem_snapshots))
                except Exception as _inv_exc:
                    logger.debug(
                        "Post-write cache invalidation failed for %s: %s",
                        tool_name,
                        _inv_exc,
                        exc_info=True,
                    )

        # ── Tool-result cache + Undo + git-snapshot invalidation ──
        # For bash, only invalidate when the command actually mutates
        # files/git state — read-only bash (ls, git status, grep, …) leaving
        # the cache intact greatly improves hit rate because the model
        # interleaves such commands between edits.
        _should_invalidate = result.ok and self._tool_call_mutates(tool_name, args)
        if _should_invalidate:
            if self._tool_result_cache is not None:
                # Write tools know their target file(s) → drop only overlapping
                # cache entries. bash (and anything else with an unknown target)
                # falls back to a full clear — safer than guessing scope.
                _write_paths = (
                    self._extract_write_target_paths(tool_name, args) if tool_name in self._WRITE_TOOLS else None
                )
                if _write_paths:
                    _n = self._tool_result_cache.invalidate_paths(_write_paths)
                    logger.debug(
                        "Tool result cache scoped-invalidated %d entr(y/ies) for %s -> %s",
                        _n,
                        tool_name,
                        _write_paths,
                    )
                else:
                    self._tool_result_cache.clear()
                    logger.debug(
                        "Tool result cache cleared due to successful write tool: %s",
                        tool_name,
                    )
            # Write tools already ran _invalidate_cache_after_write with their
            # known target paths above. A mutating NON-write tool (bash) never
            # reaches that, so its file / walk / symbol caches kept serving
            # pre-write state — see _invalidate_caches_unknown_scope.
            if tool_name not in self._WRITE_TOOLS:
                self._invalidate_caches_unknown_scope()
            # Promote this run's confirmed file CREATIONS to tombstones.
            # Sits on the success branch, so a refused write leaves nothing
            # behind for Undo to delete.
            if tool_name in self._WRITE_TOOLS:
                self._checkpoint_after_write(tool_name, args)
            # The git snapshot is module-global and keyed by repo_root, so a
            # subagent writing to the same root must invalidate it too — which
            # is why this lives at the central mutation point rather than on a
            # callback list (both clone paths reset per-registry callbacks, and
            # the callback machinery was removed after this call site replaced
            # its only consumer). It also covers mutating bash (git commit, rm),
            # which changes git state without going through a write tool.
            try:
                from .agent_context_manager import _clear_git_cache

                # Pass the registry's own root so only that repo's entry is
                # stamped dirty (coalesced invalidation); the no-arg fallback
                # still covers any caller that cannot name a root.
                _clear_git_cache(self.repo_root)
            except Exception:
                logger.debug("git snapshot invalidation failed", exc_info=True)

    def dispatch(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
        """Public entry: dispatch a tool call and record metrics.

        Wraps :meth:`_dispatch_impl` so EVERY return path — cache hit, normal
        completion, write-safety rollback, gate/arg validation, unknown tool,
        exception — is recorded exactly once on the **global** collector
        (webapp dashboard aggregate).  The per-loop per-turn summary is served
        by a separate collector (``agent_turn_pipeline`` records tool calls
        there).  These are distinct sinks — no double-counting.

        Serial and parallel (dispatch_parallel submits to the shared thread
        pool, which calls self.dispatch) execution all pass through this
        single wrapper.
        """
        result = self._dispatch_impl(tool_name, args)
        # 3-state cache outcome (mirrors agent_turn_pipeline's per-loop recording).
        # ``is_result_cacheable`` is the "was it probed?" SSOT — it matches
        # _dispatch_impl's guard (cache exists AND read-only), so a write/serial
        # tool OR any tool when the cache is disabled emits ``None`` (N/A).
        # Counting a non-probed tool as a miss would pin its per-tool
        # cache_hit_rate at a fake 0%. _dispatch_impl stamps
        # ``metadata["cache_hit"]=True`` ONLY on a real hit (cacheable tools), so
        # absence of that key on a cacheable tool means "miss"; absence on a
        # non-cacheable tool means "N/A".
        if result.metadata and result.metadata.get("cache_hit"):
            _cache_outcome: bool | None = True
        elif self.is_result_cacheable(tool_name):
            _cache_outcome = False
        else:
            _cache_outcome = None
        get_global_collector().record_tool_call(tool_name, result.execution_time, _cache_outcome, failed=not result.ok)
        return result

    def _dispatch_impl(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
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

        # Check for cancellation before any work — the agent-loop ESC event
        # AND this call's per-call scope. Executor submit sites (MCP adapter,
        # dispatch_parallel) install the scope; when the caller abandoned the
        # call while this worker was still QUEUED the event is already set and
        # the pool slot must free here without running the handler at all.
        _scope_ce = current_cancel_event()
        if (self.config.cancel_event is not None and self.config.cancel_event.is_set()) or (
            _scope_ce is not None and _scope_ce.is_set()
        ):
            return ToolResult(
                ok=False,
                content="",
                error="Operation cancelled",
                execution_time=0.0,
                retryable=False,
            )

        # Agent profile tool access validation
        if hasattr(self, "_agent_profile") and self._agent_profile is not None:
            profile = self._agent_profile
            # blocked_tools takes precedence over allowed_tools
            if hasattr(profile, "blocked_tools") and tool_name in profile.blocked_tools:
                return ToolResult(
                    ok=False,
                    content="",
                    error=f"Tool '{tool_name}' is blocked by agent profile '{profile.name}'",
                    execution_time=0.0,
                    metadata={"blocked": "agent_profile", "profile": profile.name},
                )
            # allowed_tools: empty list means no restriction (all tools allowed)
            if hasattr(profile, "allowed_tools") and profile.allowed_tools and tool_name not in profile.allowed_tools:
                return ToolResult(
                    ok=False,
                    content="",
                    error=f"Tool '{tool_name}' not in allowed_tools for profile '{profile.name}'",
                    execution_time=0.0,
                    metadata={"blocked": "agent_profile", "profile": profile.name},
                )

        # Argument repair (names, then types)
        repair = self._arg_repairer.repair(tool_name, args)
        if repair.repaired:
            args = repair.repaired_args
        if repair.errors:
            # A type the schema cannot accept and this layer will not guess.
            # Refusing HERE rather than letting the handler raise is the whole
            # point: the handler's `args.get(x, "").strip()` produces
            # "AttributeError: 'list' object has no attribute 'strip'", which
            # names neither the argument nor the expected type, so the model
            # has nothing to correct and repeats the call.
            return ToolResult(
                ok=False,
                content="",
                error=(
                    f"{tool_name}: invalid argument type(s). "
                    + " ".join(repair.errors)
                    + " Re-send the call with the types the schema declares."
                ),
                execution_time=0.0,
                retryable=True,
                metadata={"blocked": "argument_type", "tool": tool_name},
            )

        gate_result = self._gate_check(tool_name, args)
        if gate_result is not None:
            return gate_result

        # Tool result cache lookup (read-only tools only). A hit returns
        # immediately below, so no `cache_hit` flag is needed downstream — the
        # cache-store condition at the tail just checks `result.ok`.
        if self._tool_result_cache is not None and tool_name in self._READ_ONLY_TOOLS:
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
                return result

        # ── Pre-capture read scope BEFORE the handler runs (cache-miss path
        # only — a hit returned above). The signature stored in the cache must
        # describe the file state the handler SAW, not the state at set() time:
        # a write landing mid-read (background job, user editor, parallel
        # session) would otherwise be baked in as "fresh" next to the stale
        # content it raced with (TOCTOU — see ToolResultCache.set(file_sigs=)).
        # Pre-capture adds one os.stat per scoped path per read-only dispatch;
        # the tail's set() then reuses these instead of re-stating.
        _cache_paths: frozenset[str] | None = None
        _cache_sigs: dict[str, tuple[int, int, int] | None] | None = None
        if self._tool_result_cache is not None and tool_name in self._READ_ONLY_TOOLS:
            _cache_paths = self._extract_read_scope_paths(tool_name, args)
            if _cache_paths is not None:
                _cache_sigs = {p: _path_sig(p) for p in _cache_paths}

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
        _repair_path = ""
        start_time = 0.0
        try:
            # start_time FIRST so the except handler's execution_time reference is
            # always bound even if acquire/snapshot raise below.
            start_time = time.monotonic()
            # Acquire file locks for write operations INSIDE try so the finally
            # always releases them, even if snapshotting below raises.
            if flm is not None and tool_name in self._WRITE_TOOLS:
                locked_paths = flm.acquire_relevant(args, tool_name)
            # Snapshot under the lock so the captured state is consistent w.r.t.
            # concurrent writers (restore-on-rollback must reflect pre-write content).
            # Pre-write Undo checkpoint, under the same lock as the rollback
            # snapshot below so the captured state is consistent w.r.t.
            # concurrent writers. Unlike _write_snapshots this covers ALL write
            # tools — edit_text/edit_ast/anchor_edit self-validate and so skip
            # the rollback snapshot, but the user still wants to undo them.
            if tool_name in self._WRITE_TOOLS:
                self._checkpoint_before_write(tool_name, args)
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
                                    atomic_write_text(_repair_path, _repaired)
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
                                        _repair_path,
                                        len(_edit_ops),
                                    )

                    if _repair_ok:
                        _repaired_result = ToolResult(
                            ok=True,
                            content="Auto-repaired indentation — edit applied successfully",
                            execution_time=result.execution_time,
                        )
                        # Same central post-success contract as the normal tail:
                        # the repaired file is on disk, so the caches must drop
                        # pre-write state and the checkpoint must confirm it.
                        self._after_write_success(tool_name, args, _repaired_result, _write_snapshots)
                        return _repaired_result

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
                        logger.exception(
                            "Write safety: repair path crashed — falling through to rollback: %s",
                            _repair_exc,
                        )
                        _arg_repaired = False
                    if _arg_repaired:
                        # Repair succeeded — file is already written, re-verify
                        logger.info("Write safety: argument mismatch repaired, edit applied successfully")
                        result.metadata["repaired_args"] = True
                        self._after_write_success(tool_name, args, result, _write_snapshots)
                        return result

                    # --- Cross-op dependency guard: non-syntax compilation errors may be ---
                    # resolved by downstream ops (e.g. op1 changes a signature and
                    # op2 updates callers). Classify the error: true syntax errors
                    # always rollback, but type/compilation errors are kept.
                    try:
                        _soft_fail = self._should_soft_fail_verify(_verify_detail, _write_snapshots)
                    except Exception as _soft_exc:
                        logger.exception(
                            "Write safety: soft-fail classification crashed — treating as hard fail (rollback): %s",
                            _soft_exc,
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
                            "Write safety: non-syntax error — keeping changes (may be resolved by downstream ops): %s",
                            _verify_detail,
                        )
                        result.metadata["verify_warning"] = _verify_detail
                        self._after_write_success(tool_name, args, result, _write_snapshots)
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
                                _parts.append(f"  [{_oi}] type={_t}\n       anchor: {_a!r}\n       content: {_c!r}")
                            _detail_parts.append("--- Attempted edit operations ---\n" + "\n".join(_parts))

                    # 2. Restored (original) file content with line numbers
                    _err_path, _err_line = None, None
                    _parts = _verify_detail.split(":", 3)
                    if len(_parts) >= 3 and _parts[1].isdigit() and _parts[2].isdigit():
                        _err_path, _err_line = _parts[0], int(_parts[1])
                    for _path, _orig in _write_snapshots.items():
                        if _err_path and _path != _err_path:
                            continue
                        # New-file snapshots hold the _MISSING_SNAP sentinel (no
                        # original content) — rollback deleted the file, so there
                        # is no "AFTER ROLLBACK" context to show for it.
                        if not isinstance(_orig, str):
                            _detail_parts.append(f"--- {_path}: newly created file removed by rollback ---")
                            continue
                        _lines = _orig.splitlines()
                        if _err_line and 1 <= _err_line <= len(_lines):
                            _start = max(0, _err_line - 3)
                            _end = min(len(_lines), _err_line + 2)
                            _ctx = "\n".join(f"{i + 1:4d}|{_lines[i]}" for i in range(_start, _end))
                            _detail_parts.append(f"--- {_path} (lines {_start + 1}-{_end}) AFTER ROLLBACK ---\n{_ctx}")

                    _full_detail = "\n\n".join(_detail_parts)
                    logger.warning(
                        "Write safety: %s %s",
                        tool_name,
                        _full_detail,
                    )
                    return ToolResult(
                        ok=False,
                        content="",
                        error=f"ROLLBACK: {tool_name}: {_full_detail}",
                        execution_time=result.execution_time,
                    )

            # Performance metrics are recorded at the single dispatch() exit
            # (the wrapper), not here — that covers this path AND every early
            # return (write-safety rollback, gate/arg validation, unknown tool).

            # ── Central post-success point ──
            # Semantic auto-repair, post-write cache invalidation (file/walk/
            # symbol/RAG/graph + tool-result), Undo-checkpoint confirmation and
            # git-snapshot clearing — one call shared with the repair and
            # soft-fail early returns above, so no disk-changing success path
            # can skip them (see _after_write_success).
            self._after_write_success(tool_name, args, result, _write_snapshots)

            # Cache result for read-only tools (a cache hit already returned
            # above, so `result.ok` alone is sufficient here)
            if self._tool_result_cache is not None and tool_name in self._READ_ONLY_TOOLS and result.ok:
                # Convert ToolResult to serializable dict.
                # ``metadata`` defaults to {} via default_factory, but a handler
                # that passes metadata=None explicitly overrides that default —
                # and `dict(None)` here would raise, killing the whole tool call
                # for a merely cosmetic slip.  Worse, it only fires when the
                # cache is on and the tool is read-only, so the same handler
                # looks fine in tests that disable the cache.
                cached = {
                    "ok": result.ok,
                    "content": result.content,
                    "error": result.error,
                    "metadata": dict(result.metadata or {}),
                }
                # paths/file_sigs were captured BEFORE the handler ran (see the
                # miss-path pre-capture above) — a mid-read rewrite now makes
                # the next get() drop this entry instead of serving stale
                # content as fresh for the whole TTL.
                self._tool_result_cache.set(
                    tool_name,
                    args,
                    cached,
                    paths=_cache_paths,
                    file_sigs=_cache_sigs,
                )
                logger.debug("Tool result cache SET: %s (args: %s, paths: %s)", tool_name, args, _cache_paths)

        except Exception as e:
            logger.exception("Tool %s raised exception", tool_name)
            return ToolResult(
                ok=False, content="", error=f"{type(e).__name__}: {e}", execution_time=time.monotonic() - start_time
            )
        else:
            return result
        finally:
            if locked_paths and flm is not None:
                flm.release_all(locked_paths)

    def dispatch_parallel(self, tool_calls: list[dict[str, Any]]) -> list[ToolResult]:
        """
        Execute multiple tool calls in parallel via the shared thread pool.

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
        has_write_tool = any(self._tool_call_mutates(call.get("tool", ""), call.get("args", {})) for call in tool_calls)
        has_serial_tool = any(
            self._tool_call_is_serial(call.get("tool", ""), call.get("args", {})) for call in tool_calls
        )
        if not self.config.parallel_tool_execution_enabled or len(tool_calls) <= 1 or has_write_tool or has_serial_tool:
            # Fall back to sequential execution
            logger.debug(
                "Parallel execution disabled or unsafe: enabled=%s, count=%d, has_write=%s, has_serial=%s",
                self.config.parallel_tool_execution_enabled,
                len(tool_calls),
                has_write_tool,
                has_serial_tool,
            )
            return [self.dispatch(call.get("tool", ""), call.get("args", {})) for call in tool_calls]

        logger.debug("Parallel tool execution activated for %d tools", len(tool_calls))
        # Shared thread pool — the single parallel dispatch path. The former
        # asyncio layer was deleted: with no dependency edges it was semantically
        # identical to this (same pool, same ordering, same error wrapping), and
        # it relied on asyncio.get_event_loop_policy(), deprecated since 3.12 and
        # removed in Python 3.16.
        # Each call runs inside a per-call cancel scope so an aborted batch can
        # cooperatively free its workers (see cancel_scope module docstring).
        futures = []
        call_events: list[threading.Event] = []
        for call in tool_calls:
            tool_name = call.get("tool", "")
            args = call.get("args", {})
            ev = threading.Event()
            future = shared_pool.submit(self._dispatch_in_scope, tool_name, args, ev)
            futures.append((future, call))
            call_events.append(ev)

        # Collect results in order. Cancel-aware: poll instead of blocking on a
        # bare future.result() so ESC (cancel_event) is honored while a long tool
        # is still running — a blocking wait would freeze the whole turn until
        # the tool's own timeout (seconds to minutes). The in-flight tool keeps
        # running in the pool (threads cannot be killed); its result is discarded.
        results = []
        _ce = self.config.cancel_event
        try:
            for future, _call in futures:
                try:
                    while True:
                        try:
                            result = future.result(timeout=CANCEL_POLL_INTERVAL)
                            break
                        except _FutureTimeoutError:
                            if _ce is not None and _ce.is_set():
                                raise AgentCancelled("cancelled by user during parallel tool phase") from None
                except AgentCancelled:
                    # A cancel decided by the tool / the poll above must abort the
                    # batch, NOT be wrapped into a ToolResult error that the caller
                    # would feed back to the LLM as a tool failure.
                    raise
                except Exception as e:
                    logger.exception("Parallel tool execution failed")
                    result = ToolResult(
                        ok=False, content="", error=f"Parallel execution error: {type(e).__name__}: {e}"
                    )
                results.append(result)
        finally:
            # Cooperative cancellation of whatever is still in flight (batch
            # aborted above, or this collection raised): set each unfinished
            # call's scope event so its worker stops at its next checkpoint
            # (dispatch entry when still queued, scanner boundaries when
            # running) instead of occupying its pool slot to completion.
            for (future, _call), ev in zip(futures, call_events, strict=True):
                if not future.done():
                    ev.set()
        return results

    def _dispatch_in_scope(self, tool_name: str, args: dict[str, Any], cancel_event: threading.Event) -> ToolResult:
        """``dispatch`` + per-call cancel scope (see ``cancel_scope`` docs).

        Executor submit sites that can abandon a call mid-flight route through
        here so the worker observes the abandonment and frees its slot.
        """
        with call_cancel_scope(cancel_event):
            return self.dispatch(tool_name, args)

    @staticmethod
    def _schema_variant_key(lang_filter: LanguageId | None, design_chat: bool) -> tuple[bool, bool]:
        """``(include_python_only, include_design_chat)`` for the variant tables."""
        include_python_only = lang_filter is None or lang_filter is LanguageId.PYTHON
        return include_python_only, design_chat

    def get_tool_schemas(
        self,
        lang_filter: LanguageId | None = None,
        design_chat: bool = False,
    ) -> list[dict[str, Any]]:
        """Return tool schemas for the LLM API.

        Args:
            lang_filter: When set to a non-Python LanguageId, schemas with
                ``"x_python_only": True`` are excluded.  Pass ``None`` (default)
                to include all tools (Python or mixed-language repos).
            design_chat: Include tools whose handler lives on ``DesignChatLoop``
                rather than on this registry (``"x_design_chat_only": True`` —
                the insight tools and ``search_design_history``). Default False,
                because dispatching one of them here returns "Unknown tool":
                only the design chat loop, which intercepts them by name before
                dispatch, may advertise them. See ``DESIGN_CHAT_ONLY_TOOL_NAMES``.

        Note:
            Returns one of four shared, import-time lists (see
            ``tool_schemas.TOOL_SCHEMA_VARIANTS``) — no per-call copy and no
            per-registry duplicate. Callers must NOT mutate the returned list or
            its dicts; extend a ``list(...)`` copy instead, as
            ``orchestrator._obr_base`` does.
        """
        return TOOL_SCHEMA_VARIANTS[self._schema_variant_key(lang_filter, design_chat)]

    def get_tool_names(self, lang_filter: LanguageId | None = None, design_chat: bool = False) -> frozenset:
        """Return the frozen set of known tool names for O(1) membership checks.

        Cheaper than calling :meth:`get_tool_schemas` and building a set each
        turn — useful for validating LLM-emitted tool-call names (see
        ``agent_turn_pipeline._build_and_filter_prepared_calls``).

        Args:
            lang_filter: When set to a non-Python LanguageId, Python-only tools
                (``"x_python_only": True``) are excluded so that a masked tool
                is rejected at validation time, not only hidden from the schema.
                Pass ``None`` (default) for the full set.
            design_chat: See :meth:`get_tool_schemas`. Must match what was passed
                there, or validation and advertisement disagree.
        """
        return TOOL_NAME_VARIANTS[self._schema_variant_key(lang_filter, design_chat)]

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
        /code, /repo and repo basenames to the actual repo_root. Preserves subpaths
        (e.g. /tests) and works within shell commands as well.
        """
        if not text:
            return text
        _basename = Path(self.repo_root).name

        # Pass 1: Strict bias paths (/workspace, /app, /project, /code, /repo)
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

            def _strict_repl(m, _b=_basename, _iv=_iv):
                # Never rewrite a bias path that lives inside a shell-quoted
                # literal or a heredoc body (grep '/workspace', a config written
                # via <<'EOF', etc.) — doing so corrupts the literal content.
                if _match_in_quotes(m.start(), _iv):
                    return m.group(0)
                # Never rewrite a path that really exists on this machine — it
                # is real user data, not a training-data bias path.
                if _bias_matched_path_is_real(m):
                    return m.group(0)
                if _under_scratch_root(_bias_matched_candidate(m)):
                    return m.group(0)
                prefix = "" if m.group(1) == "~" else m.group(1)
                cd = m.group(2) or ""
                subpath = m.group(3) or ""
                if subpath.startswith(f"/{_b}"):
                    subpath = subpath[len(_b) + 1 :]
                return prefix + cd + self.repo_root + subpath

            new_text = re.sub(
                rf"(^|[\s~])(cd\s+)?{re.escape(_bp)}(/\S*)?(?=\s|[&;]|$)",
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
        #
        # ⚠️  The token must be ABSOLUTE (start with / or ~). Training-data bias
        #    paths are virtual roots, always spelled absolute. A RELATIVE token
        #    ending in the basename is real data, not a bias: a branch literally
        #    named `rename/asicode` made `git rev-parse rename/asicode` rewrite
        #    to `git rev-parse <repo_root>`, which resolves to nothing and reads
        #    as ref corruption (live incident 2026-08-02).
        _bp_basename = f"/{_basename}"
        if _bp_basename in text:
            _iv = _literal_intervals(text)

            def _basename_repl(m):
                if _match_in_quotes(m.start(), _iv):
                    return m.group(0)
                if _bias_matched_path_is_real(m):
                    return m.group(0)
                if _under_scratch_root(_bias_matched_candidate(m)):
                    return m.group(0)
                prefix = "" if m.group(1) == "~" else m.group(1)
                cd = m.group(2) or ""
                subpath = m.group(3) or ""
                return prefix + cd + self.repo_root + subpath

            _re_basename = re.compile(
                rf"(^|[\s~])(cd\s+)?(?:[~/][\w./~+@-]*?)?{re.escape(_bp_basename)}(/\S*)?(?=\s|[&;]|$)",
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
        return {k: self._correct_bias_path(v) if isinstance(v, str) else v for k, v in args.items()}

    @property
    def _effective_repo_root(self) -> str:
        """Return the effective repo root, preferring staging override."""
        return self._repo_root_override or self.repo_root

    def _secure_path(self, path: str, *, confine: bool = False) -> Path | None:
        """
        Resolve a path against repo_root.

        Returns None if the path escapes repo_root (unless unrestricted_read is
        set) or cannot be resolved. Absolute paths resolve as-is; relative paths
        anchor at repo_root.

        When ``config.unrestricted_read`` is True (trusted local CLI — see
        ``AgentConfig.unrestricted_read``) the repo-boundary check is skipped and
        any resolvable path is allowed. That flag is never set on the webapp path,
        where repo_root is attacker-controlled.

        ``confine=True`` forces the repo-boundary check to run REGARDLESS of
        ``unrestricted_read``. Write tools that mutate a file via
        ``symbol_modify_tool.modify_symbol`` (modify_symbol / edit_ast, and the
        apply_patch→modify_symbol auto-fallback) pass ``confine=True`` so writes
        can never escape repo_root even on a trusted CLI — the unrestricted flag
        is a READ capability only, never a write capability. (Read-only analysis
        helpers such as ``_analyze_patch_symbol_change`` keep the default,
        flag-respecting mode.)

        The repo-root resolution is memoized per effective-root string
        (``_secure_root_resolve_cache``): the root is a session constant, so the
        boundary check stays correct while skipping repeated filesystem
        resolution on this hot path. Only the root resolve is cached — the
        candidate path is resolved fresh on every call because resolving it IS
        the symlink boundary check.
        """
        path = self._correct_bias_path(path)
        unrestricted = getattr(getattr(self, "config", None), "unrestricted_read", False)
        try:
            root_str = self._effective_repo_root
            repo = self._secure_root_resolve_cache.get(root_str)
            if repo is None:
                repo = Path(root_str).resolve()
                self._secure_root_resolve_cache[root_str] = repo
            p = Path(path)
            # Absolute paths resolve as-is; relative paths anchor at repo_root.
            resolved = p.resolve() if p.is_absolute() else (repo / path).resolve()
            if confine or not unrestricted:
                try:
                    resolved.relative_to(repo)
                except ValueError:
                    logger.warning("Path traversal attempt blocked: %r -> %s", path, resolved)
                    return None
        except Exception:
            logger.debug("_secure_path: resolution failed for %r", path, exc_info=True)
            return None  # non-critical — never block execution
        else:
            return resolved

    @property
    def applied_patches(self) -> list[str]:
        """
        Return list of successfully applied patch texts.
        """
        return list(self._applied_patches)

    # NO __del__. There used to be one that shut down the (then per-registry)
    # AsyncToolExecutor thread pool "to avoid leaking worker threads". It was
    # redundant and harmful in two distinct ways:
    #
    # 1. It shut down a pool this object did not own: a filtered clone SHARED
    #    the parent's executor, so collecting the clone shut down the parent's
    #    live pool. (That clone path is long gone, and the executor itself is
    #    now deleted — parallel dispatch goes straight to the process-wide
    #    ``_thread_pool.shared_pool``.)
    # 2. It did blocking pool shutdown from a GC finalizer, which can run at any
    #    allocation point: ``tests/unit`` on 3.12 crashed the interpreter
    #    (SIGSEGV) with ``Garbage-collecting`` atop
    #    ``ThreadPoolExecutor.shutdown``.
    #
    # Nothing replaces it because CPython does this cleanup finalizer-safely:
    # ThreadPoolExecutor holds ``weakref.ref(self, weakref_cb)`` whose callback
    # puts None on the work queue, so idle workers wake and exit once the
    # executor becomes unreachable (thread.py, 3.12 L191 / 3.14 L226). Pinned by
    # test_thread_pool.test_idle_workers_exit_when_executor_is_dropped so the
    # justification fails loudly if that ever stops being true.
    #
    # If deterministic teardown is ever needed, add an explicit close()/context
    # manager — the same conclusion design_chat_loop reached for its own
    # ``__del__``-based flush (see its note on GC delaying finalization).
