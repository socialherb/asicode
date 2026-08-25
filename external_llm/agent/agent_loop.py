"""
Agent Loop for asicode

LLM tool-use loop: LLM calls tools autonomously to accomplish a task.
Falls back to text-based tool simulation for providers that don't support function calling.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import subprocess
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from external_llm.client import (
    ContextWindowCollapseError,
    LLMCancelled,
    LLMClient,
    LLMConnectionError,
    LLMMessage,
    LLMRateLimitError,
    LLMServerUnavailableError,
)
from path_security import normalize_rel_path

from ..graph.graph_facade import RepositoryGraphFacade
from ._response_utils import _TRUNCATION_REASONS, replace_tool_calls
from ._shared_utils import (
    MIN_USABLE_MESSAGE_BUDGET,
    _capped_put,
    context_message_cap,
    estimate_cache_adjusted_cost,
    estimate_cost,
    estimate_tokens_from_msgs,
    make_tool_signature,
    render_file_diagnostics_block,
)
from .agent_context_manager import (
    ContextManagerMixin,
    get_git_snapshot,
)
from .agent_fast_path import FastPathMixin

# Re-export types from the extracted types module
from .agent_loop_types import (
    AgentCancelled,
    AgentResult,
    AgentTurn,
    TurnContext,
)
from .agent_phase_manager import PhaseManagerMixin
from .agent_turn_pipeline import TurnPipelineMixin
from .config.thresholds import config
from .context_budget import (
    ContextBudgetManager,
    _is_context_length_error,
    _record_context_overflow,
    _resolve_context_limit,
)
from .failure_classifier import FailureClassifier
from .json_repair import repair_json_brackets, try_parse_json
from .performance_metrics import PerformanceCollector, get_global_collector
from .reasoning_utils import reasoning_ab_kwargs
from .request_intent_classifier import (
    RoutingIntent,
    intent_is_undetermined,
    is_non_edit_intent,
    routing_intent_from_intent_result,
)
from .run_store import InMemoryRunStore
from .symbol_search import SymbolSearcher
from .task_router import Lane
from .tool_registry import AgentConfig, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


def _new_session_id() -> str:
    """Generate a short session identifier used across AgentLoop flows."""
    return uuid.uuid4().hex[:16]


# Providers known to support native tool calling.
# "zai" is included because the factory (client.py) routes it exclusively to
# ZAIAnthropicClient, which speaks the Anthropic Messages API and ships a real
# chat_with_tools(). Without this entry, _check_native_tool_support() returns
# False for zai, which (a) makes the zai branch of _append_native_tool_messages
# dead code and (b) degrades zai tool calls to text-mode simulation in the
# main agent loop and the PLANNER_FALLTHROUGH/clarification fallback paths.
_NATIVE_TOOL_PROVIDERS = {"openai", "anthropic", "google", "deepseek", "ollama", "zai", "opencode"}

# Bounded adaptive tool memory cap. Entries are keyed by a sha256 tool signature
# (one per distinct (tool, args) combination), and only the last 3 entries of
# each memory (most recently touched — see _remember_tool) are ever read by
# _build_tool_hint — so without a cap, a long
# session with ever-varying args (edit_file, bash, ...) grows these dicts
# without bound.
_TOOL_MEMORY_MAX_ENTRIES = 256


def _remember_tool(memory: dict[str, Any], key: str, value: Any) -> None:
    """Insert into a bounded tool memory, evicting the oldest entry over the cap.

    Module-level (not a method) so the host pattern in tests can drive
    _record_tool_success/_failure without subclassing AgentLoop.

    Move-to-end (true LRU): re-inserting an existing key removes it first, so
    touched entries are re-ranked as most recent. Without this, a frequently
    re-used tool stays pinned at its first-insertion position and can be
    pushed out of the hint window by one-off tools inserted after it. Python
    dicts preserve insertion order, so ``next(iter(memory))`` is the
    least-recently-touched entry. The consumers (_build_tool_hint) only read
    the last 3 entries of each memory, so eviction never removes anything that
    would be displayed.
    """
    # Delegate to the shared FIFO/LRU eviction SSOT (``_capped_put``). Its
    # pop-then-assign = move-to-end matches this helper's documented true-LRU
    # contract (re-touched keys rank as most recent); ``_capped_put`` adds
    # free-threaded resize safety around ``next(iter())`` the bare pop lacked.
    _capped_put(memory, key, value, cap=_TOOL_MEMORY_MAX_ENTRIES)


class AgentLoop(FastPathMixin, ContextManagerMixin, PhaseManagerMixin, TurnPipelineMixin):
    """
    Main orchestration loop for the agent system.

    Responsibilities:
    - Accept user request
    - Resolve routing intent
    - Build and execute operation plan
    - Manage session state
    - Handle retries / failures / learning signals

    This is the top-level entry point coordinating planner, executor,
    and supporting subsystems.
    """

    def _record_git_state(self) -> dict[str, Any]:
        """Record current git state for potential rollback."""
        git_info = self._collect_git_info()
        return {
            "head_hash": git_info.get("head_hash", "unknown"),
            "has_changes": git_info.get("has_changes", False),
            "recorded_at": time.time(),
        }

    def _collect_git_info(self) -> dict[str, Any]:
        """Return branch, status, last_commit, head_hash, has_changes.

        Shares a single TTL-cached git snapshot with _build_session_context so
        that branch/status/log are each fetched ONCE per run, rather than
        independently here (3 git calls) AND again by the session-context
        builder (2 git calls). 5 subprocess spawns per run -> 3.
        """
        try:
            repo_root = getattr(self.registry, "repo_root", None)
            if not repo_root:
                return {}
            snap = get_git_snapshot(repo_root)
            # status is already bounded to GIT_STATUS_MAX_CHARS at the SSOT
            # (get_git_snapshot) — every prompt/metadata consumer shares that
            # single truncation contract; do not re-slice here.
            status = snap.get("status", "")
            return {
                "branch": snap.get("branch", ""),
                "status": status,
                "last_commit": snap.get("last_commit", ""),
                "head_hash": snap.get("head_hash", ""),
                "has_changes": bool(status),
            }
        except Exception as e:
            logger.warning("Failed to collect git info: %s", e)
            return {}

    @staticmethod
    def _extract_files_from_patch(patch: str) -> list[str]:
        """Parse unified diff header to extract affected file paths (repo-root relative)."""
        paths: list[str] = []
        for line in patch.splitlines():
            if line.startswith("+++ "):
                p = line[4:].split("\t")[0]
                if p.startswith("b/"):
                    p = p.removeprefix("b/")
                p = p.strip()
                if p and p != "/dev/null":
                    paths.append(p)
        return paths

    def _rollback_patches(self, patches: list[str]) -> dict[str, Any]:
        """Rollback applied patches in reverse order.

        Primary: git apply -R.

        Safety note on the fallback strategy:
        When the primary ``git apply -R`` fails (the file has moved past the patched
        state), the PREVIOUS implementation fell back to ``git restore --source=HEAD``
        per affected file. That is destructive in a *shared working tree*:

        - In multi-agent orchestration several subagents write to the same checkout.
        - In the webapp thread pool several concurrent user sessions share one checkout.

        A primary failure there means *another session already edited the same file*,
        so the file now carries a mix of this session's change and the other session's
        change. Restoring the whole file to HEAD silently WIPES the other session's
        change. We therefore **do NOT** run the destructive restore. Instead we surface
        a clear, non-destructive "needs manual rollback" result so the operator/LLM
        can perform a targeted manual revert. (``git apply -R --3way`` was evaluated
        as a preserve-concurrent-edits alternative, but it requires the post-image
        blob in the git object store / index, which this architecture's unstaged
        working-tree writes never guarantee — so it fails identically and is unused.)
        """
        if not patches:
            return {"success": True, "message": "No patches to rollback", "rolled_back": 0}

        rollback_results = []

        for i, patch in enumerate(reversed(patches)):
            patch_index = len(patches) - i - 1
            try:
                import os
                import tempfile

                temp_file: str | None = None
                try:
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as _tmp:
                        temp_file = _tmp.name
                        _tmp.write(patch)

                    check_result = subprocess.run(
                        ["git", "apply", "-R", "--check", temp_file],
                        cwd=self.registry.repo_root,
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=30,
                    )

                    if check_result.returncode == 0:
                        apply_result = subprocess.run(
                            ["git", "apply", "-R", temp_file],
                            cwd=self.registry.repo_root,
                            capture_output=True,
                            text=True,
                            check=False,
                            timeout=30,
                        )

                        if apply_result.returncode == 0:
                            rollback_results.append(
                                {
                                    "patch_index": patch_index,
                                    "success": True,
                                    "message": "Successfully rolled back via git apply -R",
                                }
                            )
                            continue
                        _primary_err = f"git apply -R failed: {apply_result.stderr.strip()}"
                    else:
                        _primary_err = f"git apply -R --check failed: {check_result.stderr.strip()}"

                    # Primary reverse failed: the file has moved past the patched state.
                    # In a shared working tree (multi-agent orchestration / webapp thread
                    # pool) this means ANOTHER session edited the same file, so a full
                    # `git restore --source=HEAD` would wipe that other session's change.
                    # We deliberately do NOT run the destructive restore here; instead we
                    # surface a non-destructive "needs manual rollback" result so the
                    # operator can perform a targeted revert of just this session's hunk.
                    affected_files = self._extract_files_from_patch(patch)
                    affected_list = ", ".join(affected_files) if affected_files else "(unparseable)"
                    logger.warning(
                        "[ROLLBACK] Primary reverse failed for patch %d (%s): %s — "
                        "NOT running destructive git restore (shared-tree safety). "
                        "Manual targeted rollback required for: %s",
                        patch_index,
                        affected_list,
                        _primary_err,
                        affected_list,
                    )
                    rollback_results.append(
                        {
                            "patch_index": patch_index,
                            "success": False,
                            "message": (
                                f"{_primary_err}; automatic rollback aborted to protect "
                                f"concurrent edits on shared file(s): {affected_list}. "
                                f"Manual targeted rollback required."
                            ),
                            "primary_error": _primary_err,
                            "needs_manual_rollback": True,
                            "affected_files": affected_files,
                        }
                    )

                finally:
                    if temp_file is not None:
                        with contextlib.suppress(OSError):
                            os.unlink(temp_file)

            except Exception as e:
                rollback_results.append(
                    {
                        "patch_index": patch_index,
                        "success": False,
                        "message": f"Exception during rollback: {e}",
                    }
                )

        success_count = sum(1 for r in rollback_results if r["success"])
        total_count = len(rollback_results)

        return {
            "success": success_count == total_count,
            "rolled_back": success_count,
            "total": total_count,
            "results": rollback_results,
        }

    def __init__(
        self,
        llm_client: LLMClient,
        registry: ToolRegistry,
        config: AgentConfig,
        model: str = "",
        agent_id: str = "main",
        run_store: InMemoryRunStore | None = None,
        session_id: str | None = None,
    ):
        self.llm_client = llm_client
        self.registry = registry
        self.config = config
        self.model = model
        self.agent_id = agent_id
        self.session_id = session_id
        # Wire the caller's cancellation signal into the client so its
        # internal retry backoff sleeps (openai_client._request_with_retry)
        # abort on ESC instead of blocking up to ~36s per attempt.  The
        # attribute is a per-client hook; loops that share a client (subagents
        # reusing the orchestrator client) all observe the same signal, which
        # is exactly the orchestrator-level cancel semantics wanted.
        _ce = getattr(config, "cancel_event", None)
        if _ce is not None:
            llm_client.cancel_event = _ce

        # NOTE: the session-state RESTORE surface used to live here (and a second
        # copy in run(), see the note there). Both are gone — and the
        # session_state.py module itself has since been removed too: nothing in
        # the app ever WROTE a session file (`save_state` / `SessionState.save()`
        # had no shipping caller, only tests), so the restore could not fire —
        # and if a file had ever existed it would have crashed on arrival,
        # because it read `saved_state.context` while `SessionState` had no
        # `context` attribute (`SessionStateManager.load_state` built one from
        # session_id/edit_history/plan only). That AttributeError sat unguarded
        # in __init__.
        #
        # The three attributes it assigned — self.edit_history / self.plan /
        # self.context — were assigned NOWHERE else and read nowhere at all, so
        # removing the block leaves no reader without a value. Constructing the
        # manager also mkdir'd `repo_root/.asicode/sessions/` on every AgentLoop
        # for a directory nothing writes to; that stops too.
        #
        # Removed rather than implemented, for the same reason
        # _filter_prepared_calls was: making it work means designing what a
        # session save/restore should contain, which is a feature, not a repair.
        # Per-loop PerformanceCollector for session-isolated per-turn
        # summaries. The webapp dashboard reads the global collector
        # (get_global_collector(), which receives ALL sessions' data via
        # ''record_llm_call'' at the agent_loop level and ''record_tool_call''
        # from the dispatch wrapper), while each loop's own collector provides
        # correct isolated metrics for the per-turn ``metadata["performance"]``.
        # This decoupling closes the previous split-brain (dashboard blind to
        # LLM metrics, per-turn summary blind to cache/rag) WITHOUT the
        # concurrent-session isolation regression of a shared singleton.
        self.performance_collector = PerformanceCollector(
            session_id=self.session_id,
        )
        self._failure_classifier = FailureClassifier()

        self._tool_success_memory = {}
        self._tool_fail_memory = {}
        self.current_intent = "general"

        self._tool_retry_counter = defaultdict(int)

        self._patch_fail_count = 0

        # Agent phase state machine: DISCOVER -> READ -> EDIT -> VERIFY -> FINISH
        self._agent_phase = "DISCOVER"
        self._phase_target_symbol = ""
        self._phase_target_file = ""

        # Optional per-run turn list (attach point for run_store); only read
        # behind hasattr in telemetry, but declared so pyright sees the field.
        self.turns: list[Any] = []

        # Hybrid architecture components (lazy-initialized). The PlannerAgent /
        # OperationExecutor half went with the PLANNER lane; these two survive
        # because MAIN_AGENT tooling reads them.
        self._symbol_searcher: SymbolSearcher | None = None
        self._call_graph: RepositoryGraphFacade | None = None
        self._shared_run_store: InMemoryRunStore = run_store if run_store is not None else InMemoryRunStore()
        _helper_enabled = config.helper_enabled
        _helper_model = config.helper_model
        _helper_max_calls = config.helper_max_calls
        _helper_ollama_url = config.helper_ollama_url

        if _helper_enabled and _helper_model:
            try:
                from .local_assistant import LocalAssistant

                self.registry.local_assistant = LocalAssistant(
                    local_model=_helper_model,
                    repo_root=self.registry.repo_root,
                    callback=config.stream_callback,
                    ollama_base_url=_helper_ollama_url,
                    max_local_calls=_helper_max_calls,
                )
                logger.info("Helper enabled (delegate_to_helper): model=%s", _helper_model)
            except Exception as e:
                logger.warning("Failed to initialize helper backend: %s", e)
                self.registry.local_assistant = None
        else:
            self.registry.local_assistant = None

        self._context_budget: ContextBudgetManager | None = None
        if config.context_budget_enabled:
            _budget_model = model or config.model_name or ""
            self._context_budget = ContextBudgetManager(
                model_name=_budget_model,
                reserve_for_output=config.context_budget_reserve_output,
            )

        # Context manager (sliding window trim/compress/evict)
        self._init_context_manager()

    def _resolve_routing_intent(self, route) -> RoutingIntent:
        """Resolve routing intent from route / config in one place."""
        if getattr(self.config, "design_chat_mode", False):
            return "read_only"
        _ir = getattr(route, "intent_result", None) if route else None
        return routing_intent_from_intent_result(_ir)

    def _record_tool_usage(self, tool_name: str, tool_args: dict[str, Any], ok: bool) -> None:
        """Record a tool execution outcome for adaptive routing (shared core).

        ``ok=True`` records a success and drops any failure entry for the same
        key (a recovered tool disappears from the failure hints — F2 contract);
        ``ok=False`` records a failure and leaves the success memory untouched.
        The value is (tool_name, count): the sha256 key (make_tool_signature)
        is not reversible, so the name must be carried alongside the count for
        _build_tool_hint to display. Consumers must never print the key itself.
        """
        memory = self._tool_success_memory if ok else self._tool_fail_memory
        key = self._tool_key(tool_name, tool_args)
        _cur = memory.get(key)
        count = (_cur[1] if _cur else 0) + 1
        _remember_tool(memory, key, (tool_name, count))
        if ok and key in self._tool_fail_memory:
            del self._tool_fail_memory[key]
        # Adaptive tool-usage learning channel
        try:
            self._shared_run_store.record_tool_usage("MAIN_AGENT", tool_name, ok, "")
        except Exception as _exc:
            logger.debug("record_tool_usage failed for %s: %s", tool_name, _exc)
            # non-critical — never block execution

    def _record_tool_success(self, tool_name: str, tool_args: dict[str, Any]) -> None:
        """Record a successful tool execution (delegates to :meth:`_record_tool_usage`)."""
        self._record_tool_usage(tool_name, tool_args, True)

    def _record_tool_failure(self, tool_name: str, tool_args: dict[str, Any]) -> None:
        """Record a failed tool execution (delegates to :meth:`_record_tool_usage`)."""
        self._record_tool_usage(tool_name, tool_args, False)

    def _try_readonly_early_finish(self, tool_name: str, tool_result, original_request: str, read_only_request: bool):
        """Return AgentResult for definitive read-only answers, or None to continue."""
        if not read_only_request:
            return None

        if not tool_result.ok:
            return None

        req_lower = original_request.lower().strip()
        _has_analysis_intent = (
            "?" in req_lower  # interrogative sentences (includes _has_question_form case)
            or req_lower.startswith(("explain", "describe", "summarize", "analyze", "what", "how", "why"))
        )
        if _has_analysis_intent:
            return None

        content = tool_result.content or ""
        definitive = False

        if tool_name in ("find_symbol", "get_project_info") and content and len(content) > 20:
            definitive = True

        if definitive:
            # Use first 400 chars of tool result as the answer preview
            preview = (content[:400] + "…") if len(content) > 400 else content
            return AgentResult(
                status="success",
                turns=[],  # will be filled by caller
                final_message=preview,
                applied_patches=list(self.registry.applied_patches),
                metadata={
                    "readonly_early_finish": True,
                    "tool": tool_name,
                    "deterministic_answer": True,
                },
            )
        return None

    def _strip_thinking_text(self, text: str) -> str:
        """Remove model reasoning/thinking text that leaks into assistant content."""
        if not text:
            return text

        cleaned = text

        cleaned = re.sub(
            r"<think>.*?</think>\s*",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()
        cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE).strip()

        markers = [
            "\nFinal answer:",
            "\nAnswer:",
            "\n답변:",
            "\n결론:",
            "\n완료:",
            "\nHello",
            "\nHi",
        ]

        suspicious_prefix_terms = [
            "let me think",
            "the user asked",
            "i should",
            "i'll go with",
            "first,",
            "hmm,",
            "wait,",
            "solid response",
            "how to respond",
        ]

        for marker in markers:
            idx = cleaned.find(marker)
            if idx > 0:
                prefix = cleaned[:idx].lower()
                if any(term in prefix for term in suspicious_prefix_terms):
                    cleaned = cleaned[idx:].lstrip()
                    break

        return cleaned.strip()

    def _extract_known_file_path(self, request: str) -> str:
        """Extract an explicit file path from the request. Returns repo-relative path or ""."""
        req = str(request or "").strip()
        if not req:
            return ""

        candidates = re.findall(
            r"[\w\./\-]+\.(?:py|js|ts|html|css|md|json|yaml|yml|toml|txt|ini|cfg|conf|kt|java|xml)", req
        )
        if not candidates:
            return ""

        from pathlib import Path

        repo_root = Path(str(self.registry.repo_root))

        for candidate in candidates:
            path = normalize_rel_path(str(candidate))
            if (
                not path
            ):  # pragma: no cover — regex candidates are always non-empty, so normalization can never return "" here
                continue
            try:
                full_path = repo_root / path
                if full_path.exists() and full_path.is_file():
                    return path
            except (OSError, AttributeError):
                logger.debug("_extract_known_file_path: path probe failed for %s", path)
                continue

        return ""

    def _extract_target_keywords(self, request: str) -> list[str]:
        """Extract target text keywords for keyword-miss detection (nudge model to search elsewhere)."""
        targets: list[str] = []

        for q in re.findall(
            r'["\u201c\u201d\u2018\u2019\'`]([^\'"` \u201c\u201d\u2018\u2019]{2,60})["\u201c\u201d\u2018\u2019\'`]',
            request,
        ):
            if q not in targets:
                targets.append(q)
            if len(targets) >= 3:
                break

        en = re.search(
            r'(?:change|replace|rename|update|convert)\s+["\']?([A-Za-z_\-\s]{2,40}?)["\']?\s+'
            r"(?:to|with|into|→)",
            request,
            re.IGNORECASE,
        )
        if en:
            w = en.group(1).strip()
            if w and w not in targets:
                targets.append(w)

        return [t for t in targets if 2 <= len(t) <= 60]

    def _is_local_model(self) -> bool:
        """Return True for local/Ollama-backed runs."""
        provider = self._get_provider_name()
        return provider in {"ollama", "local_ollama"}

    def _build_turn_context(
        self,
        request: str,
        context: str,
        route: Any,
        git_state: Any,
        session_id: str,
        is_local_model: bool,
        has_native_tools: bool,
        read_only_request: bool,
        known_target_file: str,
        target_keywords: list[str],
        tier: Any,
        plan: dict[str, Any] | None,
        plan_subtasks: list[dict[str, Any]],
        turns: list,
    ) -> TurnContext:
        """Build a standard TurnContext with consistent defaults."""
        return TurnContext(
            request=request,
            context=context,
            route=route,
            git_state=git_state,
            session_id=session_id,
            is_local_model=is_local_model,
            has_native_tools=has_native_tools,
            read_only_request=read_only_request,
            # Derived from the same route the intent hint came from, so the
            # "was this actually classified?" answer cannot drift from the
            # read_only_request it qualifies.
            intent_undetermined=intent_is_undetermined(getattr(route, "intent_result", None) if route else None),
            known_target_file=known_target_file,
            target_keywords=target_keywords,
            tier=tier,
            plan=plan,
            plan_subtasks=plan_subtasks,
            turn_num=0,
            turns=turns,
        )

    def run(self, request: str, context: str = "", continuation_data: dict | None = None) -> AgentResult:
        self._continuation_data = continuation_data or getattr(self.config, "continuation_data", None)
        if not hasattr(self, "state") or self.state is None:
            self.state = {}

        # NOTE: the second session-restore block lived here. It was unreachable
        # by construction, not merely by circumstance: `SessionState.load_state()`
        # returned None (it mutated self and fell off the end; the module has
        # since been removed), so `loaded_state`
        # was ALWAYS None and the five `self.state[...]` assignments below it
        # never ran once. Nothing read those keys either — `state['agent_phase']`
        # in particular was unrelated to the `_agent_phase` machine, which is a
        # plain attribute. Doubly dead: the SessionState it called was built with
        # a FRESH `_new_session_id()` each construction, so even a working
        # load_state() would have looked for a file that cannot exist.
        # See the companion note in __init__ for why this is removed, not fixed.
        _session_id = _new_session_id()

        _profile = getattr(self.config, "agent_profile", None)
        if _profile is not None and hasattr(_profile, "apply"):
            _profile.apply(self.config)
            logger.info("Agent profile applied: %s", _profile.name)

        self.performance_collector.start_session()

        route = getattr(self.config, "route_decision", None)
        if route:
            _route_conf = float(getattr(route, "confidence", 0.0))
            _route_lane = str(getattr(route, "lane", "?"))
            _route_kind = str(getattr(route, "task_kind", "?"))
            if _route_conf <= 0.10:
                logger.warning(
                    "Route confidence is suspiciously low (%.2f): lane=%s kind=%s reasoning=%s "
                    "target_specificity=%.2f — consider using a non-zero default",
                    _route_conf,
                    _route_lane,
                    _route_kind,
                    getattr(route, "reasoning", "?"),
                    float(getattr(route, "target_specificity_score", 0.0)),
                )
            logger.info(
                "Route applied: kind=%s lane=%s complexity=%s conf=%.2f",
                _route_kind,
                _route_lane,
                str(getattr(route, "complexity", "?")),
                _route_conf,
            )
            self._cb(
                "route_applied",
                {
                    "task_kind": str(getattr(route, "task_kind", "")),
                    "lane": str(getattr(route, "lane", "")),
                    "confidence": float(getattr(route, "confidence", 0.0)),
                },
            )

        self.current_intent = "general"
        logger.debug("Session intent: %s", self.current_intent)

        self._routing_intent_hint = self._resolve_routing_intent(route)

        read_only_request = is_non_edit_intent(self._routing_intent_hint)
        known_target_file = self._extract_known_file_path(request)
        _target_keywords: list[str] = self._extract_target_keywords(request)

        if self.config.stream_callback:
            self.config.stream_callback(
                "routing_intent",
                {
                    "intent": self._routing_intent_hint,
                    "source": "intent_result",
                },
            )

        self._agent_phase = "DISCOVER"
        self._phase_target_symbol = ""
        self._phase_target_file = known_target_file or ""

        if read_only_request:
            self._agent_phase = "DISCOVER"

        # Filesystem operations start in EDIT so bash is available immediately
        if route and hasattr(route, "reasoning") and "Filesystem operation" in (route.reasoning or ""):
            self._agent_phase = "EDIT"
            logger.info("Filesystem operation detected — starting in EDIT phase")

        git_state = self._record_git_state()
        turns: list[AgentTurn] = []

        is_local_model = self._is_local_model()
        has_native_tools = self._check_native_tool_support()

        # Context pre-fetch: PLANNER loads RAG; MAIN_AGENT/COMPACT use tools.
        _route_for_tier = route if route is not None else getattr(self.config, "route_decision", None)
        tier = self._resolve_context_tier()
        self._context_tier = tier  # stored for _build_initial_messages
        logger.info("Context tier resolved: %s", tier)

        # ── MAIN_AGENT lane: direct LLM tool-use loop ──
        # The only lane there is: task_router returns MAIN_AGENT for everything
        # and the PLANNER lane it used to contrast with has been removed. Only
        # route=None or an unhandled lane falls through to the guard below.
        if route and route.lane == Lane.MAIN_AGENT:
            logger.info("MAIN_AGENT lane: running direct LLM tool-use loop")
            ctx = self._build_turn_context(
                request,
                context,
                route,
                git_state,
                _session_id,
                is_local_model,
                has_native_tools,
                read_only_request,
                known_target_file,
                _target_keywords,
                tier,
                None,
                [],
                turns,
            )
            # Batch adaptive-hub persistence for the whole loop. Without this
            # every tool call re-serialises and fsyncs the full ~94 KB hub
            # namespace (_record_tool_success/_failure -> record_tool_usage):
            # 11 writes / 1.3 MB / 41 ms on a 12-turn run, most of the loop's
            # own non-LLM wall clock. The batch still flushes on an interval so
            # a crash cannot discard the whole session's learning signals.
            with self._shared_run_store.batch_adaptive_signals():
                _result = self._run_llm_loop(ctx)
            # Stamp the run's Undo point. Done here rather than at each
            # AgentResult construction site because run() is the single funnel
            # every MAIN_AGENT result leaves through, and the id is only known
            # after the loop has actually written something.
            _cp_id = self.registry.run_checkpoint_id
            if _cp_id:
                if _result.metadata is None:
                    _result.metadata = {}
                _result.metadata["checkpoint_id"] = _cp_id
            return _result

        logger.warning(
            "run() reached end without handling route.lane=%s — returning error (fail-closed)",
            str(getattr(getattr(route, "lane", None), "value", getattr(route, "lane", None))),
        )
        return AgentResult(
            status="error",
            error="No active lane handled this request. Ensure AgentConfig.route_decision is set.",
            turns=turns or [],
            final_message="No active lane handled this request.",
            applied_patches=list(self.registry.applied_patches),
            metadata={
                "session_id": _session_id,
                "git_state": git_state,
                "unhandled_lane": True,
            },
        )

    # .asicode/ housekeeping

    @staticmethod
    def _ensure_asicode_gitignored(repo_root: str) -> None:
        """Add .asicode/ to .gitignore if it isn't already there.

        Delegates to the shared :func:`external_llm.agent.tool_registry._ensure_asicode_gitignored`
        to avoid code duplication with :class:`ToolRegistry`.
        """
        from external_llm.agent.tool_registry import (
            _ensure_asicode_gitignored as _shared,
        )

        _shared(repo_root)

    # Session history logging

    def _save_session_log(
        self,
        session_id: str,
        request: str,
        result: AgentResult,
        prompt_tokens: int,
        completion_tokens: int,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> None:
        """Append a one-line JSON record to .asicode/sessions.jsonl."""
        # Read cache tokens from result metadata if available (callers may not pass them)
        _meta_tokens = (result.metadata or {}).get("tokens", {}) if hasattr(result, "metadata") else {}
        if not cache_read_tokens and "cache_read_tokens" in _meta_tokens:
            cache_read_tokens = _meta_tokens["cache_read_tokens"]
            cache_creation_tokens = _meta_tokens.get("cache_creation_tokens", 0)

        log_dir = os.path.join(self.registry.repo_root, ".asicode")
        log_path = os.path.join(log_dir, "sessions.jsonl")
        try:
            os.makedirs(log_dir, exist_ok=True)
            self._ensure_asicode_gitignored(self.registry.repo_root)
            provider = self.llm_client.get_provider_name().lower()
            record = {
                "session_id": session_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model": self.model,
                "provider": provider,
                "request": request[:300],
                "status": result.status,
                "turns_used": len(result.turns),
                "patches_applied": len(result.applied_patches),
                "touched_files": list(
                    {
                        f
                        for t in result.turns
                        if t.tool_result
                        for f in ((t.tool_result.metadata or {}).get("touched_files") or [])
                    }
                ),
                "tokens": {
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                    "total": prompt_tokens + completion_tokens,
                    "cost_usd": round(estimate_cost(provider, prompt_tokens, completion_tokens, model=self.model), 6),
                    "cache_adjusted_cost_usd": round(
                        estimate_cache_adjusted_cost(
                            provider,
                            prompt_tokens,
                            completion_tokens,
                            cache_read_tokens,
                            cache_creation_tokens,
                            model=self.model,
                            base_url=getattr(self.llm_client, "base_url", ""),
                        ),
                        6,
                    ),
                    "cache_read_tokens": cache_read_tokens,
                    "cache_creation_tokens": cache_creation_tokens,
                },
                "error": result.error,
            }
            # Serialize appends across concurrent agents (the orchestrator runs
            # agents in a ThreadPoolExecutor) and across processes sharing the
            # same repo_root/.asicode/sessions.jsonl. Without this, multi-KB
            # records (large touched_files lists) can interleave at the write()
            # syscall boundary — Python's buffered append may flush in multiple
            # chunks — producing torn (unparseable) JSONL lines. Mirrors the
            # index-lock pattern in webapp/run_store.py.
            from pathlib import Path as _Path

            from external_llm.common.file_lock import cross_process_flock

            _session_log_rotate_bytes = 10 * 1024 * 1024
            with cross_process_flock(_Path(log_dir) / "sessions.lock"):
                # Rotate when the log exceeds 10 MB (single generation, like
                # worker.log rotation in orchestrator.py).  Best-effort: any
                # OSError is silently ignored — the session log is advisory.
                with contextlib.suppress(OSError):
                    if os.path.isfile(log_path) and os.path.getsize(log_path) > _session_log_rotate_bytes:
                        os.replace(log_path, log_path + ".1")
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.debug("Session log written: %s", log_path)
        except Exception as e:
            logger.warning("Failed to write session log: %s", e)

    # Stream callback helper

    def _cb(self, event: str, data: dict[str, Any]) -> None:
        """Add metadata to event callback."""
        if self.config.stream_callback:
            try:
                # Default metadata
                enriched_data = dict(data)
                if "agent_id" not in enriched_data:
                    enriched_data["agent_id"] = self.agent_id

                # Additional metadata
                enriched_data.update(
                    {
                        "agent_turn_num": len(self.turns) if hasattr(self, "turns") else 0,
                        "event_timestamp": time.time(),
                        "orchestrator_phase": getattr(self, "_orchestrator_phase", "standalone"),
                        "session_id": getattr(self.config, "session_id", "unknown"),
                    }
                )

                # Add global sequence if missing
                if "global_sequence_id" not in enriched_data:
                    # Use nanosecond timestamp — maintain consistency with orchestrator
                    enriched_data["global_sequence_id"] = int(time.time_ns())

                self.config.stream_callback(event, enriched_data)
            except Exception as e:
                logger.warning("Error in _cb callback: %s", e)

    # LLM call strategies

    def _check_native_tool_support(self) -> bool:
        """Check if the provider supports native tool calling."""
        provider = self._get_provider_name()
        if provider not in _NATIVE_TOOL_PROVIDERS or not hasattr(self.llm_client, "chat_with_tools"):
            return False
        if provider == "ollama":
            # Ollama support varies per MODEL, not per provider: only tags
            # whose /api/show reports the "tools" capability can use the
            # native tool-calling protocol. Unknown (server unreachable /
            # non-tag name / older Ollama) → assume supported (status quo);
            # only a definitive "no tools" downgrades to the text protocol.
            from external_llm.model_registry import ollama_supports_tools

            _base_url = getattr(self.llm_client, "base_url", None)
            if ollama_supports_tools(self.model, _base_url) is False:
                logger.info(
                    "Ollama model %s reports no 'tools' capability — using text tool protocol",
                    self.model,
                )
                return False
        return True

    def _llm_call_with_tools(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        token_callback: Callable | None = None,
    ) -> dict[str, Any]:
        """Call LLM with native tool support."""
        # Check for cancellation before starting LLM call
        if self.config.cancel_event and self.config.cancel_event.is_set():
            raise AgentCancelled("cancelled by user before LLM call")

        # Pre-flight: fit messages to budget. fit_messages never truncates —
        # it returns the ORIGINAL list (docstring contract) — so there is no
        # "messages were dropped" signal here; actual drops surface via the
        # repair_tool_message_sequence count below. A len() comparison would
        # be permanently dead code.
        if self._context_budget:
            messages = self._context_budget.fit_messages(
                messages,
                tool_schemas=None,
                warn_cb=self._cb,
                trim_cb=self._force_trim_context,
            )

        # Pre-flight: repair orphaned tool messages before sending to provider.
        # fit_messages may leave orphaned tool messages when dropping exchange groups;
        # this is a safety net that prevents HTTP 400 "insufficient tool messages" errors.
        # Run unconditionally for all providers — repair is a harmless consistency fix.
        # Coverage: detects tool messages in all formats (standard role="tool",
        # Anthropic tool_use/tool_result blocks, Gemini functionCall/functionResponse
        # parts) via message_shapes.is_tool_result / is_tool_call.
        from .context_budget import repair_tool_message_sequence

        _before_repair = len(messages)
        messages = repair_tool_message_sequence(messages)
        if len(messages) < _before_repair:
            _dropped_count = _before_repair - len(messages)
            logger.info(
                "_llm_call_with_tools: repair_tool_message_sequence removed %d orphaned messages", _dropped_count
            )
            self._cb(
                "agent_working",
                {
                    "reason": "tool_message_repair",
                    "dropped": _dropped_count,
                },
            )

        # Tool schemas are serialised into the prompt, so build them before the
        # token guard below so it can account for their size.
        tool_schemas = self.registry.get_tool_schemas(
            lang_filter=self.registry.repo_language,
        )

        _est_tokens: int | None = None

        # Pre-flight: structural-collapse guard — raises a clear error when the
        # output reserve + tool schemas ALONE exhaust the window (the call would
        # 400 even with zero chat history). preemptive_trim was REMOVED (2026-08):
        # oversized prompts are enforced by the provider's own limit, with the
        # 400 → _record_context_overflow override backstop lowering the effective
        # limit for subsequent calls. Normal context management is handled by the
        # sliding-window compressor in context_manager.py.
        if self._context_budget:
            _ctx_limit = _resolve_context_limit(self.model, base_url=getattr(self.llm_client, "base_url", None))
            _safety_margin = config.tokens.CONTEXT_HARD_CAP_SAFETY_MARGIN
            # Reserve output room AND account for tool-schema tokens (see helper):
            # a full prompt on small windows (Ollama 8192) leaves 0 to generate.
            _cap = context_message_cap(_ctx_limit, _safety_margin, tool_schemas)
            if _cap < MIN_USABLE_MESSAGE_BUDGET:
                # Structural collapse: output reserve + tool schemas ALONE already
                # exhaust the window, so no message trim can fit the call — it
                # would 400 even with zero chat history. context_message_cap logs
                # the diagnosis once per signature; raise a clear error instead of
                # sending a guaranteed-400 request and burning 3 retries per turn.
                raise ContextWindowCollapseError(
                    f"context window ({_ctx_limit}) is too small for the tool "
                    f"schemas + output reserve: only {_cap} tokens remain for "
                    f"messages (minimum {MIN_USABLE_MESSAGE_BUDGET}). Reduce the "
                    f"toolset or raise the model's context window (e.g. num_ctx)."
                )
            _est_tokens = estimate_tokens_from_msgs(messages)
        _max_tokens = max_tokens if max_tokens is not None else config.tokens.AGENT_TOOL_CALL

        def call_llm() -> dict[str, Any]:
            nonlocal _max_tokens
            _attempt = 0
            _base = _max_tokens
            while True:
                # ── Reasoning A/B control (developer-scoped) ────────────────
                # Default: model default (reasoning ON). Set ASICODE_DEVELOPER_REASONING=off
                # to inject a suppression fragment into the DeepSeek payload. Same
                # mechanism as ASICODE_PLANNER_REASONING for the Planner agent.
                _reasoning_kwargs = reasoning_ab_kwargs("ASICODE_DEVELOPER_REASONING")
                response = self.llm_client.chat_with_tools(
                    messages=messages,
                    tools=tool_schemas,
                    model=self.model,
                    max_tokens=_max_tokens,
                    thinking_mode=self.config.thinking_mode,
                    reasoning_effort=getattr(self.config, "reasoning_effort", None),
                    reasoning_callback=(
                        (lambda text: self.config.stream_callback("reasoning", {"text": text, "append": True}))
                        if self.config.stream_callback
                        else None
                    ),
                    token_callback=token_callback,
                    **_reasoning_kwargs,
                )
                _finish_reason = getattr(response, "finish_reason", None)
                # "truncated" is the provider-level silent-truncation signal
                # (anthropic_client/openai_client/providers detect a dropped
                # final delta and report it instead of end_turn). Without it in
                # this condition, a truncated tool call is silently skipped and
                # the model's intended tool use is lost; retrying mirrors the
                # "length" (max_tokens) recovery exactly.
                if _finish_reason in _TRUNCATION_REASONS and _attempt < 2:
                    _attempt += 1
                    _max_tokens = _base * (1 << _attempt)
                    logger.warning(
                        "[llm_retry] finish_reason=%s (max_tokens=%d), retrying (%d/3)",
                        _finish_reason,
                        _max_tokens,
                        _attempt + 1,
                    )
                    continue
                if _finish_reason in _TRUNCATION_REASONS:
                    logger.error(
                        "[llm_retry] finish_reason=%s after 3 attempts "
                        "(max_tokens=%d), truncated response — clearing tool calls",
                        _finish_reason,
                        _max_tokens,
                    )
                    # Clear tool calls: a truncated tool call executes stale/partial
                    # arguments (e.g. a bash command cut mid-way). The text
                    # content (even if partial) is preserved so the turn loop can
                    # continue naturally.
                    _tool_calls_cleared = len(getattr(response, "tool_calls", []) or [])
                    response = replace_tool_calls(response, [])
                    if _tool_calls_cleared:
                        logger.warning(
                            "Cleared %d truncated tool call(s) from finish_reason=length/truncated response",
                            _tool_calls_cleared,
                        )
                    # Turn-level outcome channel (record_agent_result): the ONLY
                    # place this surfaces. The LLM call itself "succeeded", so
                    # record_llm_call(failed=True) never fires and llm_metrics
                    # reads healthy — without this, a truncation storm (repeated
                    # 3x-exhaustion) is invisible in the metrics.
                    self.performance_collector.record_agent_result(truncated=True)
                    # Dual-sink to the dashboard (global collector), mirroring
                    # _record_llm_call_both's per-loop + global pattern.
                    get_global_collector().record_agent_result(truncated=True)
                break
            tool_calls = getattr(response, "tool_calls", []) or []
            # Normalize to list of dicts
            normalized_calls = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    normalized_calls.append(tc)
                else:
                    normalized_calls.append(
                        {
                            "id": getattr(tc, "call_id", ""),
                            "name": getattr(tc, "name", ""),
                            "args": getattr(tc, "args", {}),
                        }
                    )
            return {
                "content": response.content,
                "tool_calls": normalized_calls,
                "raw": response,
                "prompt_tokens": getattr(response, "prompt_tokens", None),
                "completion_tokens": getattr(response, "completion_tokens", None),
                "tokens_used": getattr(response, "tokens_used", None),
                "cache_read_input_tokens": getattr(response, "cache_read_input_tokens", None),
                "cache_creation_input_tokens": getattr(response, "cache_creation_input_tokens", None),
                "finish_reason": _finish_reason,
            }

        return self._retry_on_rate_limit(
            call_llm,
            "native tool calling",
            _estimated_prompt_tokens=_est_tokens,
        )

    @staticmethod
    def _repair_json_brackets(text: str) -> str:
        """Delegate to shared :func:`repair_json_brackets`."""
        return repair_json_brackets(text)

    def _try_parse_json(self, text: str) -> Any | None:
        """Try to parse JSON with 3-tier repair via shared :func:`try_parse_json`."""
        return try_parse_json(text)

    def _record_llm_call_both(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        execution_time_ms: float = 0,
        failed: bool = False,
    ) -> None:
        """Record LLM call to per-loop AND global collectors.

        Dual-sink design: per-loop (session-isolated, per-turn summary) +
        global (dashboard aggregation). The two are independent collectors
        — no double-counting.

        ``provider`` is resolved from the MAIN agent client (``self._get_provider_name``)
        and threaded into both record_llm_call() calls so LLM metrics are bucketed
        PER-PROVIDER — a failing fallback provider is then not diluted by a healthy
        primary's traffic within one shared deque. This site records the main agent's
        tool-calling LLM calls; planner/developer sub-agents record their own streams.
        """
        _provider = self._get_provider_name()
        self.performance_collector.record_llm_call(
            provider=_provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            execution_time_ms=execution_time_ms,
            failed=failed,
        )
        get_global_collector().record_llm_call(
            provider=_provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            execution_time_ms=execution_time_ms,
            failed=failed,
        )

    def _retry_on_rate_limit(
        self,
        callable_func: Callable[[], dict[str, Any]],
        mode: str = "",
        _estimated_prompt_tokens: int | None = None,
        overflow_retry_cb: Callable[[], int | None] | None = None,
    ) -> dict[str, Any]:
        """
        Retry logic for LLM calls with exponential backoff on rate limit errors.

        Args:
            callable_func: Function that makes the LLM call
            mode: Description of the call mode for logging (e.g., "native tool calling", "text mode")
            _estimated_prompt_tokens: Optional pre-call token estimate, passed to
                _record_context_overflow for fast convergence when a context-length 400 fires.
            overflow_retry_cb: When a context-length 400 is caught, this callback is invoked
                before retrying, allowing the caller to re-trim messages in its scope.
                Enables in-turn recovery instead of always re-raising.

        Returns:
            LLM response dictionary

        Raises:
            LLMConnectionError: After max retries exceeded for connection errors
            LLMRateLimitError: After max retries exceeded for rate limits
            Exception: Other exceptions are re-raised immediately
        """

        def _handle_retry_error(
            e: Exception,
            attempt: int,
            max_retries: int,
            delay: float,
            event_name: str,
            error_type: str,
            event_message: str,
            loop_t0: float,
            **extra: Any,
        ) -> None:
            """Common retry-&exhausted handler for LLMConnectionError / LLMRateLimitError."""
            mode_str = f" in {mode}" if mode else ""
            if attempt < max_retries:
                logger.warning(
                    "%s hit%s (attempt %d/%d), retrying in %d seconds",
                    type(e).__name__,
                    mode_str,
                    attempt + 1,
                    max_retries,
                    delay,
                )
                self._cb(
                    event_name,
                    {
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "delay": delay,
                        "message": event_message,
                    },
                )
                if self.config.cancel_event and self.config.cancel_event.is_set():
                    raise AgentCancelled("cancelled by user during retry wait")
                if self.config.cancel_event:
                    self.config.cancel_event.wait(timeout=delay)
                else:
                    time.sleep(delay)
            else:
                logger.error(
                    "%s after %d attempts%s, giving up",
                    type(e).__name__,
                    max_retries,
                    mode_str,
                )
                payload: dict[str, Any] = {
                    "message": f"{type(e).__name__} after {max_retries} attempts{mode_str}: {e}",
                    "error_type": error_type,
                }
                if extra:
                    payload.update(
                        extra
                    )  # pragma: no cover — loop_t0 is a named param, so **extra is always empty at the 3 internal call sites
                self._cb("error", payload)
                # Record the final failure after all retries are exhausted —
                # this is the single logical LLM call that ultimately failed.
                # execution_time_ms covers the WHOLE retry span (call attempts +
                # backoff waits) so the dashboard's avg_time_ms is not biased
                # toward 0 by failed calls. Mirrors design-chat's _call_start
                # pattern; loop_t0 was captured before the retry for-loop.
                self._record_llm_call_both(
                    failed=True,
                    execution_time_ms=round((time.monotonic() - loop_t0) * 1000),
                )
                raise e

        max_retries = 3
        retry_delays = [10, 20, 40]  # Exponential backoff: 10s, 20s, 40s

        # Capture the whole retry span's start so an exhausted retry records the
        # true wall-time (attempt calls + backoff waits) instead of 0 — keeps
        # avg_time_ms honest when failures occur.
        loop_t0 = time.monotonic()
        # start_time is set inside the try on every iteration, but pre-binding
        # it keeps the except's telemetry reference safe for type checkers
        # (and makes the 0-iteration edge case defined, not just unreachable).
        start_time = loop_t0

        for attempt in range(max_retries + 1):  # +1 for the initial attempt
            # Check for cancellation before each retry attempt
            if self.config.cancel_event and self.config.cancel_event.is_set():
                raise AgentCancelled("cancelled by user during retry loop")

            try:
                # monotonic: measures elapsed duration immune to wall-clock
                # jumps (NTP sync / DST) that could yield negative or wildly
                # skewed execution_time_ms in telemetry.
                start_time = time.monotonic()
                result = callable_func()
                if result is None:
                    return {}
                execution_time_ms = (time.monotonic() - start_time) * 1000

                # Extract token counts from result (ensure int to guard against Mock/non-int values)
                def _to_int(v):
                    try:
                        return int(v) if v is not None and isinstance(v, (int, float)) else 0
                    except (TypeError, ValueError):
                        return 0

                _pt = result.get("prompt_tokens")
                _ct = result.get("completion_tokens")
                _tu = result.get("tokens_used")
                prompt_tokens = _to_int(_pt) or _to_int(_tu)
                completion_tokens = _to_int(_ct)
                # NOTE: tokens_used is the TOTAL (prompt + completion), NOT just completion.
                # The line above already falls back to tokens_used when prompt_tokens is missing.
                # Do NOT set prompt_tokens=0 here — that would distort log metrics (pt=0 issue).
                # When the split is unavailable, we use total as prompt_tokens for accounting.

                self._record_llm_call_both(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    execution_time_ms=execution_time_ms,
                    failed=False,
                )

                # Diagnostic: log reasoning_tokens when available (DeepSeek)
                # so output-token bloat vs reasoning-token bloat is distinguishable.
                _rt = _to_int(result.get("reasoning_tokens"))
                if _rt:
                    logger.debug(
                        "[TOKEN_BREAKDOWN] completion=%d reasoning=%d visible=%d",
                        completion_tokens,
                        _rt,
                        max(0, completion_tokens - _rt),
                    )

            except LLMConnectionError as e:
                # Clamp index: loop runs max_retries+1 times but retry_delays
                # only has max_retries entries. Without this, the final
                # exhaustion iteration triggers retry_delays[attempt] IndexError,
                # which masks the original connection error.
                delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                _handle_retry_error(
                    e,
                    attempt,
                    max_retries,
                    delay,
                    "connection_retry",
                    "connection",
                    f"Connection error, retrying in {delay}s...",
                    loop_t0=loop_t0,
                )
            except LLMRateLimitError as e:
                # Honor the server's Retry-After hint when present (providers
                # that don't retry internally surface it on the exception);
                # otherwise fall back to fixed exponential backoff.
                _hint = getattr(e, "retry_after", None)
                # Bounded at construction (LLMRateLimitError clamps to
                # [1, RETRY_AFTER_MAX_WAIT]); accept int/float defensively.
                delay = (
                    _hint
                    if isinstance(_hint, (int, float)) and _hint > 0
                    else retry_delays[min(attempt, len(retry_delays) - 1)]
                )
                _handle_retry_error(
                    e,
                    attempt,
                    max_retries,
                    delay,
                    "rate_limit_retry",
                    "rate_limit",
                    f"Rate limit hit, retrying in {delay}s...",
                    loop_t0=loop_t0,
                )

            except LLMServerUnavailableError as e:
                delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                _handle_retry_error(
                    e,
                    attempt,
                    max_retries,
                    delay,
                    "server_retry",
                    "server_unavailable",
                    f"Server unavailable, retrying in {delay}s...",
                    loop_t0=loop_t0,
                )

            except LLMCancelled as e:
                # Client-internal retry backoff was interrupted by our
                # cancel_event (ESC) — surface as a user cancellation, not an
                # LLM error, so the run stops promptly.
                raise AgentCancelled(str(e)) from e

            except Exception as e:
                # Context-length 400 (HTTP 400, "context length exceeded") —
                # record overflow override so subsequent calls pre-trim at the
                # corrected limit instead of hitting the same error repeatedly.
                if _is_context_length_error(e):
                    _record_context_overflow(
                        self.model,
                        estimated_prompt_tokens=_estimated_prompt_tokens,
                        base_url=getattr(self.llm_client, "base_url", None),
                    )
                    logger.warning(
                        "[CONTEXT_OVERFLOW] %s — recorded overflow override (est=%s)",
                        self.model,
                        _estimated_prompt_tokens,
                    )
                    # In-turn recovery: re-trim messages and continue the retry loop
                    # so the same attempt is retried with the corrected limit instead
                    # of re-raising immediately. The callback returns the POST-trim
                    # estimate (int) when trim made progress, None when it did not.
                    # The returned estimate replaces _estimated_prompt_tokens so a
                    # SECOND 400 in the same turn is recorded against the size that
                    # was actually rejected — a stale pre-trim estimate would inflate
                    # the override clamp (min(reduced, est*0.85)) and stall
                    # convergence until the 3-step reduction cap is exhausted.
                    if overflow_retry_cb is not None and attempt < max_retries:
                        _new_est = overflow_retry_cb()
                        if _new_est is not None:
                            _estimated_prompt_tokens = _new_est
                            continue
                # For other exceptions, send SSE error event and re-raise immediately
                self._cb(
                    "error",
                    {
                        "message": f"LLM API error: {e}",
                        "error_type": "api",
                    },
                )
                # Record the non-retriable LLM failure before propagating.
                # execution_time_ms reflects this attempt's wall-time (start_time
                # was captured at the top of the try block) so failed non-retriable
                # calls don't bias avg_time_ms toward 0.
                self._record_llm_call_both(
                    failed=True,
                    execution_time_ms=round((time.monotonic() - start_time) * 1000),
                )
                raise
            else:
                return result

        # Unreachable: every retry path either returns or re-raises (exhaustion
        # branch of _handle_retry_error, non-retriable except handlers).
        raise AssertionError("invariant: _retry_on_rate_limit exited without a response")  # pragma: no cover

    # Message management
    # ------------------------------------------------------------------

    def _auto_repair_apply_patch_args(self, args: dict) -> dict | None:
        """Apply deterministic repair rules to apply_patch arguments.

        Rules (attempted in order, only one repair per failure):
        1. HUNK-ONLY WRAP: patch starts with '@@' but missing diff --git/--- a/+++ b/ headers
        2. MISSING diff --git HEADER: patch has --- a/ and +++ b/ but no diff --git line
        3. CRLF NORMALIZATION: patch contains \r\n line endings

        Returns new args dict for retry, or None if no repair applicable.
        """
        patch = args.get("patch", "")
        if not isinstance(patch, str) or not patch.strip():
            return None

        lines = patch.strip().splitlines()
        # Find first non-empty line
        first_non_empty = None
        for line in lines:
            if line.strip():
                first_non_empty = line
                break

        # 1. HUNK-ONLY WRAP
        if first_non_empty and first_non_empty.startswith("@@"):
            # Check if already has headers
            patch_lower = patch.lower()
            if "diff --git" not in patch_lower and "--- a/" not in patch_lower and "+++ b/" not in patch_lower:
                # Need path hint
                path_hint = args.get("path")
                if not path_hint or not isinstance(path_hint, str) or not path_hint.strip():
                    # Cannot repair without path
                    return None
                # Validate path security using existing validator
                norm = normalize_rel_path(path_hint.strip())
                if not norm:
                    # Path invalid or unsafe
                    return None
                # Construct headers
                header = f"diff --git a/{norm} b/{norm}\n--- a/{norm}\n+++ b/{norm}\n"
                new_patch = header + patch
                new_args = args.copy()
                new_args["patch"] = new_patch
                # Ensure path is included (already present)
                return new_args

        # 2. MISSING diff --git HEADER FIX
        # Check if patch contains --- a/ and +++ b/ but no diff --git line
        patch_lower = patch.lower()
        if (
            "--- a/" in patch_lower
            and "+++ b/" in patch_lower
            and "diff --git" not in patch_lower
            and patch_lower.count("--- a/") == 1
            and patch_lower.count("+++ b/") == 1
        ):
            # Extract path from --- a/ line (first occurrence)
            a_match = re.search(r"^--- a/(.+)$", patch, re.MULTILINE)
            b_match = re.search(r"^\+\+\+ b/(.+)$", patch, re.MULTILINE)
            if a_match and b_match:
                a_path = a_match.group(1).strip()
                b_path = b_match.group(1).strip()
                if a_path == b_path:
                    # Same path, safe to add diff --git header
                    norm = normalize_rel_path(a_path)
                    if norm:
                        header = f"diff --git a/{norm} b/{norm}\n"
                        new_patch = header + patch
                        new_args = args.copy()
                        new_args["patch"] = new_patch
                        return new_args

        # 3. CRLF NORMALIZATION
        if "\r\n" in patch:
            new_patch = patch.replace("\r\n", "\n")
            # Only retry if patch changed
            if new_patch != patch:
                new_args = args.copy()
                new_args["patch"] = new_patch
                return new_args

        # No repair applicable
        return None

    # ── Tool result message helpers ──────────────────────────────────────────────

    def _append_write_plan_guidance(self, content: str, tool_name: str, result: ToolResult) -> str:
        """Append write_plan failure recovery advice."""
        with contextlib.suppress(AttributeError, TypeError, KeyError):  # metadata-shape tolerance
            if not result.ok and tool_name == "write_plan":
                err = str(result.error or content or "")
                err_lower = err.lower()
                advice = []
                if "block not found" in err_lower:
                    advice.append(
                        "BLOCK NOT FOUND: Use find_symbol to locate the exact function/method, "
                        "then read_file with start_line/end_line around the returned line. "
                        "Copy the EXACT text into 'before' — read_file's │N│ gutter gives each "
                        "line's leading-whitespace count, which bash cat does not show."
                    )
                elif "anchor" in err_lower and ("not found" in err_lower or "match" in err_lower):
                    advice.append(
                        "ANCHOR ERROR: Use a unique line from the file as anchor. "
                        "Read the file first and pick a line that appears only once."
                    )
                elif "block match count is not 1" in err_lower:
                    advice.append("AMBIGUOUS MATCH: Add more surrounding lines to make 'before' unique.")
                elif "placeholder" in err_lower:
                    advice.append("PLACEHOLDER: You must use actual code from the file, not example text.")
                if not advice:
                    advice.append("Read the file first, then retry write_plan with exact text in 'before'.")
                content = (content or "") + "\n\n[RECOVERY] " + " ".join(advice)
        return content

    def _append_edit_warnings_guidance(self, content: str, tool_name: str, result: ToolResult) -> str:
        """Append edit_file warnings from result metadata as guidance text.

        Detects:
          - edit_warnings (anchor fuzzy-match, content/anchor inversion, op-type mismatch)
          - syntax_check (post-write syntax validation failures, pre-Fix 1 rollbacks)
        and appends actionable guidance so the LLM can correct its next invocation.
        """
        with contextlib.suppress(AttributeError, TypeError, KeyError):  # metadata-shape tolerance
            edit_warnings = (result.metadata or {}).get("edit_warnings")
            if edit_warnings and isinstance(edit_warnings, list) and edit_warnings:
                guidance_lines = ["\n\n[EDIT FILE WARNINGS]"]
                guidance_lines.extend(f"- {w}" for w in edit_warnings)
                # General tip for the most common warning type
                if any("replace op(s) resulted in" in w or "op-type mismatch" in w for w in edit_warnings):
                    guidance_lines.append(
                        "Tip: '+N, -0' on a replace op means the intended operation was likely "
                        "insert_after or insert_before, not replace. "
                        "Use insert_after (or insert_before) to add content without removing the anchor."
                    )
                content = (content or "") + "\n".join(guidance_lines)

            # Also surface syntax_check metadata (post-write syntax validation results).
            # This provides defense-in-depth: even if Fix 1 rollback somehow doesn't
            # trigger (edge case), the LLM still sees the syntax error info.
            _syn = (result.metadata or {}).get("syntax_check")
            if _syn and not _syn.get("skipped") and not _syn.get("ok"):
                _err_list = _syn.get("errors") or []
                if _err_list:
                    _syn_lines = ["\n\n[SYNTAX WARNING]"]
                    # Limit to first 5 errors to avoid bloat
                    _syn_lines.extend(
                        f"- line {e.get('line')}:{e.get('col')} \u2014 {e.get('message', '').strip()}"
                        for e in _err_list[:5]
                    )
                    _syn_lines.append(
                        "Tip: Multi-line replace content often has indentation mismatches. "
                        "Use the EXACT indentation from the source file. "
                        "For Python changes, consider edit_ast to avoid indentation issues."
                    )
                    content = (content or "") + "\n".join(_syn_lines)

            # Surface semantic diagnostics (undefined names, types, missing imports)
            # collected by validate_semantics (pyright/tsc/go build against the real
            # project). These are NON-BLOCKING — they inform the LLM so it can
            # self-heal on the next turn, mirroring how an LSP publishDiagnostics
            # notification informs without rejecting the edit. Rendered as a
            # <file_diagnostics> block (Crush-style) which the LLM parses reliably.
            content = self._append_semantic_diagnostics(content, result)
        return content

    def _append_semantic_diagnostics(self, content: str, result: ToolResult) -> str:
        """Render semantic diagnostics as a <file_diagnostics> guidance block.

        Collects diagnostics from two metadata shapes (both supported):
          1. lightweight path: result.metadata["syntax_check"]["semantic_diagnostics"]
             (set by _run_syntax_check_for_file for apply_patch/edit_text/etc.)
          2. heavyweight path: result.metadata["semantic_report"]["diagnostics"]
             (set by the write_plan verification pipeline via ctx.details)

        Only *errors* and *warnings* are surfaced; info-level diagnostics are
        dropped to keep the LLM context focused. Returns *content* unchanged if
        there are no diagnostics to report.

        The rendering itself lives in the shared
        :func:`render_file_diagnostics_block` so the turn-end deferred path
        (``_settle_deferred_semantics``) produces the identical block for a
        coalesced check, rather than leaving the model to find raw diagnostics
        buried only in the JSON metadata.
        """
        diags: list = []
        # Path 1: lightweight tools (apply_patch, edit_text, edit_file, ...)
        _syn = (result.metadata or {}).get("syntax_check")
        if isinstance(_syn, dict):
            _sd = _syn.get("semantic_diagnostics")
            if isinstance(_sd, list):
                diags.extend(_sd)
        # Path 2: heavyweight write_plan pipeline
        _sem = (result.metadata or {}).get("semantic_report")
        if isinstance(_sem, dict):
            _sd2 = _sem.get("diagnostics")
            if isinstance(_sd2, list):
                diags.extend(_sd2)
        _block = render_file_diagnostics_block(diags)
        if not _block:
            return content
        return (content or "") + _block

    def _append_patch_retry_guidance(self, content: str, tool_name: str, result: ToolResult) -> str:
        """Append apply_patch retry guidance from result metadata."""
        with contextlib.suppress(AttributeError, TypeError, KeyError):  # metadata-shape tolerance
            retry_guidance = (result.metadata or {}).get("retry_guidance")
            if retry_guidance and not result.ok and tool_name == "apply_patch":
                guidance_lines = ["\n\n[PATCH RETRY GUIDANCE]"]
                for key, label in [
                    ("failure_type", "Failure type"),
                    ("target_file", "Target file"),
                    ("hint", "Hint"),
                    ("instruction", "Instruction"),
                ]:
                    val = retry_guidance.get(key)
                    if val:
                        guidance_lines.append(f"{label}: {val}")
                snippet = retry_guidance.get("exact_existing_snippet")
                if snippet:
                    guidance_lines.append("Exact existing code/snippet:")
                    guidance_lines.append("```")
                    guidance_lines.append(str(snippet))
                    guidance_lines.append("```")
                content = (content or "") + "\n".join(guidance_lines)
        return content

    def _build_tool_result_message(
        self, call_id: str, tool_name: str, result: ToolResult, tool_args: dict[str, Any] | None = None
    ) -> LLMMessage:
        """Build a message representing a tool result.

        OpenAI-compatible providers (e.g., DeepSeek) require tool_call_id on tool messages.
        """
        # Add tool chain suggestions to content if available
        content = result.content

        # Delegate guidance/hint generation to dedicated helpers
        content = self._append_write_plan_guidance(content, tool_name, result)
        content = self._append_patch_retry_guidance(content, tool_name, result)
        content = self._append_edit_warnings_guidance(content, tool_name, result)

        # Keep content machine-readable.
        # Convert metadata to JSON-serializable dict.
        serializable_metadata = dict(result.metadata or {})

        payload = {
            "ok": bool(result.ok),
            "content": content,
            "error": result.error,
            "metadata": serializable_metadata,
        }
        return LLMMessage(
            role="tool",
            name=tool_name,
            tool_call_id=call_id or None,
            content=json.dumps(payload, ensure_ascii=False),
        )

    def _append_native_tool_messages(
        self,
        messages: list[LLMMessage],
        response: dict[str, Any],
        tool_result_messages: list[LLMMessage],
    ) -> list[LLMMessage]:
        """
        Append tool result messages for native tool-calling providers.
        Adds the assistant message (with tool_calls) followed by tool results.
        """
        provider = self.llm_client.get_provider_name().lower()
        assistant_content = response.get("content", "")

        raw_resp = response.get("raw")
        raw_response_data = raw_resp.raw_response if raw_resp and hasattr(raw_resp, "raw_response") else None

        # _process_tool_results may interleave role="user" strategy/exhaustion
        # warnings *before* the role="tool" results in this list.  A native
        # assistant(tool_calls) message MUST be immediately followed by its tool
        # responses with nothing in between, or OpenAI/DeepSeek reject the
        # request (HTTP 400) and Anthropic/Gemini receive a malformed
        # tool_result block (a user warning with an empty tool_use_id).  Keep
        # only the tool responses adjacent to the assistant message.
        tool_msgs = [m for m in tool_result_messages if getattr(m, "role", "") == "tool"]
        extra_msgs = [m for m in tool_result_messages if getattr(m, "role", "") != "tool"]
        # The warnings are re-emitted after the tool block.  For openai/deepseek/
        # ollama a trailing user message is valid; for providers that require
        # strictly-alternating user/assistant turns (Anthropic, Gemini) a second
        # user turn would 400, so there the text is folded into the single user
        # turn that carries the tool results.
        extra_text = "\n\n".join((m.content or "") for m in extra_msgs).strip()

        if provider in ("openai", "deepseek", "opencode"):
            # OpenAI/DeepSeek format: tool messages are only valid if they
            # directly follow an assistant message that actually contains tool_calls.
            assistant_tool_calls = None
            reasoning_content = None
            if raw_response_data:
                raw_msg = raw_response_data.get("choices", [{}])[0].get("message", {}) or {}
                raw_tool_calls = raw_msg.get("tool_calls")
                if isinstance(raw_tool_calls, list) and raw_tool_calls:
                    assistant_tool_calls = raw_tool_calls
                # DeepSeek Reasoner: reasoning_content must be echoed back
                rc = raw_msg.get("reasoning_content")
                if rc is not None:
                    reasoning_content = rc

            # Filter tool_calls to only those with matching tool result messages.
            # Phase/guard filtering may have reduced the executed set, so we
            # must not advertise tool_calls that lack a corresponding response.
            # Otherwise DeepSeek/OpenAI returns HTTP 400:
            #   "assistant with tool_calls must be followed by tool messages
            #    responding to each tool_call_id".
            _filtered_tool_calls = assistant_tool_calls
            if assistant_tool_calls and tool_msgs:
                _executed_ids = {getattr(m, "tool_call_id", None) for m in tool_msgs}
                _filtered_tool_calls = [tc for tc in assistant_tool_calls if tc.get("id") in _executed_ids]
                if not _filtered_tool_calls:
                    # All tool calls were filtered out; skip this turn's
                    # assistant+tool block entirely.  The caller already
                    # handles the should_continue case via phase_rule_messages,
                    # so this branch is a defensive safety net.  Still surface
                    # any strategy warnings so the model sees the feedback.
                    return messages + extra_msgs if extra_msgs else messages

            # Always append the assistant message first.
            messages = [
                *messages,
                LLMMessage(
                    role="assistant",
                    content=assistant_content,
                    tool_calls=_filtered_tool_calls,
                    reasoning_content=reasoning_content,
                ),
            ]

            # Tool results need a preceding tool_calls block — otherwise
            # DeepSeek rejects assistant(no tool_calls) → tool with HTTP 400.
            if _filtered_tool_calls and tool_msgs:
                messages = messages + tool_msgs

            # Strategy warnings as a trailing user turn (valid after tool msgs).
            if extra_msgs:
                messages = messages + extra_msgs

        elif provider in ("anthropic", "zai"):
            # Anthropic: assistant content blocks + user tool_result blocks (keyed by tool_use_id).
            raw_blocks: list[dict[str, Any]] | None = None
            if raw_response_data:
                raw_blocks = raw_response_data.get("content")

            tool_result_blocks = [
                {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id or "",
                    "content": m.content,
                }
                for m in tool_msgs
            ]
            # Fold warnings into the same user turn (alternation-safe).
            if extra_text:
                tool_result_blocks = [*tool_result_blocks, {"type": "text", "text": extra_text}]
            messages = [
                *messages,
                LLMMessage(role="assistant", content=assistant_content, raw_content=raw_blocks),
                LLMMessage(role="user", content="", raw_content=tool_result_blocks),
            ]

        elif provider == "google":
            # Gemini: model parts (with functionCall) + user functionResponse parts.
            raw_parts: list[dict[str, Any]] | None = None
            if raw_response_data:
                candidates = raw_response_data.get("candidates", [])
                if candidates:
                    raw_parts = candidates[0].get("content", {}).get("parts")

            function_response_parts = [
                {
                    "functionResponse": {
                        "name": m.name or "",
                        "response": {"content": m.content},
                    }
                }
                for m in tool_msgs
            ]
            # Fold warnings into the same user turn (alternation-safe).
            if extra_text:
                function_response_parts = [*function_response_parts, {"text": extra_text}]
            messages = [
                *messages,
                LLMMessage(role="assistant", content=assistant_content, raw_content=raw_parts),
                LLMMessage(role="user", content="", raw_content=function_response_parts),
            ]

        elif provider == "ollama":
            # Ollama: assistant tool_calls + role="tool" results (no tool_call_id).
            tool_calls_normalized = response.get("tool_calls") or []
            ollama_tool_calls = None
            if tool_calls_normalized:
                ollama_tool_calls = [
                    {
                        "function": {
                            "name": tc.get("name", "") if isinstance(tc, dict) else "",
                            "arguments": tc.get("args", {}) if isinstance(tc, dict) else {},
                        }
                    }
                    for tc in tool_calls_normalized
                ]
            messages = [
                *messages,
                LLMMessage(
                    role="assistant",
                    content=assistant_content,
                    tool_calls=ollama_tool_calls,  # Ollama format; chat_with_tools detects it
                ),
            ]
            if ollama_tool_calls and tool_msgs:
                messages = messages + tool_msgs
            # Strategy warnings as a trailing user turn (valid after tool msgs).
            if extra_msgs:
                messages = messages + extra_msgs

        else:
            # Generic fallback — fold warnings into the single user turn.
            tool_results_text = "\n\n".join(m.content for m in tool_msgs)
            if extra_text:
                tool_results_text = tool_results_text + "\n\n" + extra_text
            messages = [
                *messages,
                LLMMessage(role="assistant", content=assistant_content),
                LLMMessage(role="user", content=tool_results_text + "\n\nContinue with the task."),
            ]

        return messages

    def _hunk_to_before_after(self, hunk_lines: list) -> tuple:
        """Extract (before_text, after_text) from a hunk body (list of lines).

        Returns (None, None) if extraction fails.
        """
        before_lines = []
        after_lines = []
        for hl in hunk_lines:
            if not hl:
                continue
            stripped = hl.rstrip("\n")
            if stripped.startswith(" "):
                before_lines.append(stripped[1:])
                after_lines.append(stripped[1:])
            elif stripped.startswith("-"):
                before_lines.append(stripped[1:])
            elif stripped.startswith("+"):
                after_lines.append(stripped[1:])
            # skip \\ No newline at end of file, etc.

        before = "\n".join(before_lines)
        after = "\n".join(after_lines)
        if not before.strip() and not after.strip():
            return None, None
        return before, after

    @staticmethod
    def _tool_key(tool_name: str, tool_args: dict[str, Any]) -> str:
        # Stable, collision-resistant signature. See make_tool_signature() for
        # why the old `hash(json.dumps(...))` form was unsafe (collision +
        # PYTHONHASHSEED instability).
        return make_tool_signature(tool_name, tool_args)

    def _get_provider_name(self) -> str:
        return self.llm_client.get_provider_name().lower()
