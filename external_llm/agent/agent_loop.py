"""
Agent Loop for asicode

LLM tool-use loop: LLM calls tools autonomously to accomplish a task.
Falls back to text-based tool simulation for providers that don't support function calling.
"""
from __future__ import annotations

import ast
import json
import logging
import os
import os as _qg_os
import re
import subprocess
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Optional

from external_llm.client import LLMClient, LLMConnectionError, LLMMessage, LLMRateLimitError, LLMServerUnavailableError, effective_content
from path_security import normalize_rel_path

from ..graph.graph_facade import RepositoryGraphFacade
from ..languages import LanguageId
from ..common.atomic_io import atomic_write_text
from ._shared_utils import (
    context_message_cap,
    estimate_cache_adjusted_cost,
    estimate_cost,
    estimate_tokens_from_msgs,
    make_tool_signature,
    preemptive_trim,
)
from .agent_context_manager import (
    ContextManagerMixin,
    ContextTier,
    _clear_git_cache,
    get_git_snapshot,
)
from .agent_escalation_pipeline import EscalationPipelineMixin
from .agent_fast_path import FastPathMixin

# Re-export types from the extracted types module
from .agent_loop_types import (
    AgentCancelled,
    AgentResult,
    AgentTurn,
    TurnContext,
)
from .agent_phase_manager import PhaseManagerMixin
from .agent_planner_pipeline import PlannerPipelineMixin
from .agent_turn_pipeline import TurnPipelineMixin
from .call_graph import CallGraphIndexer
from .config.thresholds import config
from .context_budget import (
    ContextBudgetManager,
    _is_context_length_error,
    _record_context_overflow,
    _resolve_context_limit,
)
from .failure_classifier import FailureClassifier
from .json_repair import repair_json_brackets, try_parse_json
from .operation_models import OperationKind, PlanMode, StageContext
from .performance_metrics import PerformanceCollector

# NOTE: PlannerAgent / OperationExecutor live in the (permanently-disabled) PLANNER
# lane. They are imported through the single choke-point facade so that a future
# removal of the lane directory is a one-file edit there (or facade deletion) and
# does not break this import. See planner_lane_facade docstring for rationale.
from .planner_lane_facade import OperationExecutor, PlannerAgent
from .reasoning_utils import reasoning_ab_kwargs
from .request_intent_classifier import is_non_edit_intent, routing_intent_from_intent_result
from .run_store import InMemoryRunStore
from .symbol_search import SymbolSearcher
from .task_router import Lane
from .tool_registry import AgentConfig, ToolRegistry, ToolResult
from common import normalize_rel_path_fast
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
_NATIVE_TOOL_PROVIDERS = {"openai", "anthropic", "google", "deepseek", "ollama", "zai"}

# ── Auto-fix registry (pluggable syntax fixers) ─────────────────────

class AutoFix:
    """Pluggable auto-fix for common compiler/parser errors.

    Subclasses implement two methods:

    * ``matches(content, errors, file_path, language)`` — return True if
      this fixer should be applied to the current validation result.
    * ``apply(content, errors, file_path)`` — return fixed content (str)
      or ``None`` if no fix was possible.

    *``apply`` is only called when ``matches`` returned True.*
    """

    def matches(
        self,
        content: str,
        errors: list,
        file_path: str,
        language: LanguageId,
    ) -> bool:
        raise NotImplementedError

    def apply(
        self,
        content: str,
        errors: list,
        file_path: str,
    ) -> Optional[str]:
        raise NotImplementedError


class AutoFixGoUnusedImport(AutoFix):
    """Remove ``import "X"`` lines where ``X`` is reported as unused.

    Handles both single-line (``import "os"``) and multi-line block
    (``import (\n\t"os"\n)``) import forms.  Pure text processing —
    no external dependencies.
    """

    def matches(
        self,
        content: str,
        errors: list,
        file_path: str,
        language: LanguageId,
    ) -> bool:
        return language is LanguageId.GO

    def apply(
        self,
        content: str,
        errors: list,
        file_path: str,
    ) -> Optional[str]:
        # Collect unused import names from all errors
        _unused: set = set()
        for _e in errors:
            _msg = getattr(_e, "message", None) or str(_e)
            _m = re.search(r'"([^"]+)"\s+imported and not used', _msg)
            if _m:
                _unused.add(_m.group(1))
        if not _unused:
            return None

        lines = content.split("\n")
        result: list = []
        i = 0
        while i < len(lines):
            _line = lines[i]
            _stripped = _line.strip()

            # Single-line import: import "os"  or  import "os" // comment
            _sm = re.match(r'import\s+"([^"]+)"', _stripped)
            if _sm and _sm.group(1) in _unused:
                i += 1
                continue

            # Import block: import ( ... )
            if _stripped == "import (" or _stripped.startswith("import\t("):
                _block_lines: list = []
                i += 1
                while i < len(lines) and lines[i].strip() != ")":
                    _block_lines.append(i)
                    i += 1
                _closing = lines[i] if i < len(lines) else None
                _kept: list = []
                for _idx in _block_lines:
                    _ls = lines[_idx].strip().strip('"')
                    if _ls not in _unused:
                        _kept.append(lines[_idx])
                if _kept:
                    result.append(_line)
                    result.extend(_kept)
                    if _closing is not None:
                        result.append(_closing)
                    i += 1
                else:
                    if _closing is not None:
                        i += 1
                    while i < len(lines) and lines[i].strip() == "":
                        i += 1
                continue

            result.append(_line)
            i += 1

        _fixed = "\n".join(result)
        return _fixed if _fixed != content else None


# Registry of auto-fix instances (ordered — first match wins)
_AUTO_FIX_REGISTRY: list[AutoFix] = [
    AutoFixGoUnusedImport(),
]



class AgentLoop(FastPathMixin, ContextManagerMixin, PhaseManagerMixin, TurnPipelineMixin, EscalationPipelineMixin, PlannerPipelineMixin):
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
            status = snap.get("status", "")
            return {
                "branch": snap.get("branch", ""),
                "status": status[:5000],
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
            if line.startswith('+++ '):
                p = line[4:].split('\t')[0]
                if p.startswith('b/'):
                    p = p[2:]
                p = p.strip()
                if p and p != '/dev/null':
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
                temp_file: Optional[str] = None
                try:
                    _tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False)
                    temp_file = _tmp.name
                    _tmp.write(patch)
                    _tmp.close()

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
                            rollback_results.append({
                                "patch_index": patch_index,
                                "success": True,
                                "message": "Successfully rolled back via git apply -R",
                            })
                            continue
                        else:
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
                        patch_index, affected_list, _primary_err, affected_list,
                    )
                    rollback_results.append({
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
                    })

                finally:
                    if temp_file is not None:
                        try:
                            os.unlink(temp_file)
                        except OSError:
                            pass

            except Exception as e:
                rollback_results.append({
                    "patch_index": patch_index,
                    "success": False,
                    "message": f"Exception during rollback: {e}",
                })

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
        run_store: Optional[InMemoryRunStore] = None,
        session_id: Optional[str] = None,
    ):
        self.llm_client = llm_client
        self.registry = registry
        self.config = config
        self.model = model
        self.agent_id = agent_id
        self.session_id = session_id

        try:
            from .session_state import SessionStateManager
            self.session_state_manager = SessionStateManager(self.registry.repo_root)
        except ImportError:
            self.session_state_manager = None

        if session_id and self.session_state_manager:
            saved_state = self.session_state_manager.load_state(session_id)
            if saved_state:
                self.edit_history = saved_state.edit_history
                self.plan = saved_state.plan
                self.context = saved_state.context
                logger.info(f"Loaded saved state for session_id={session_id}")
        self.performance_collector = PerformanceCollector()
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

        # Hybrid architecture components (lazy-initialized)
        self._operation_executor: Optional[OperationExecutor] = None
        self._symbol_searcher: Optional[SymbolSearcher] = None
        self._call_graph: Optional[RepositoryGraphFacade] = None
        self._planner_agent: Optional[PlannerAgent] = None
        self._shared_run_store: InMemoryRunStore = run_store if run_store is not None else InMemoryRunStore()
        _helper_enabled = config.helper_enabled
        _helper_model = config.helper_model
        _helper_max_calls = config.helper_max_calls
        _helper_ollama_url = config.helper_ollama_url

        if _helper_enabled and _helper_model:
            try:
                from .local_assistant import LocalAssistant
                self.registry.local_assistant = LocalAssistant(
                    planner_llm_client=self.llm_client,
                    planner_model=self.model,
                    local_model=_helper_model,
                    repo_root=self.registry.repo_root,
                    callback=config.stream_callback,
                    ollama_base_url=_helper_ollama_url,
                    max_local_calls=_helper_max_calls,
                )
                logger.info(f"Helper enabled (delegate_to_helper): model={_helper_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize helper backend: {e}")
                self.registry.local_assistant = None
        else:
            self.registry.local_assistant = None

        self._context_budget: Optional[ContextBudgetManager] = None
        if config.context_budget_enabled:
            _budget_model = model or config.model_name or ""
            self._context_budget = ContextBudgetManager(
                model_name=_budget_model,
                reserve_for_output=config.context_budget_reserve_output,
            )
        try:
            from .session_state import SessionState
            self.session_state = SessionState(session_id=_new_session_id())
        except (ImportError, TypeError):
            self.session_state = None
        try:
            from .slash_commands import SlashCommandRegistry
            self.slash_commands = SlashCommandRegistry()
        except ImportError:
            self.slash_commands = None

        # Context manager (sliding window trim/compress/evict)
        self._init_context_manager()

        # ── Pre-existing dirty test files snapshot ──────────────────────────
        # Capture test files that were dirty BEFORE any agent action (user's
        # WIP changes).  Scoped verification uses this snapshot to exclude
        # pre-existing dirty files from the quality gate — user's broken WIP
        # tests must not pollute verification results.
        #
        # LIMITATION: one-shot snapshot (session-start only).  Files that become
        # dirty during the session (new user WIP) are NOT excluded — they will
        # be included in scoped verification.  Files that were dirty at session
        # start but are committed/stashed during the session remain excluded
        # from the snapshot (but are clean, so git_status_test_files won't
        # return them — no practical gap).
        self._pre_existing_dirty_files: set[str] | None = None
        try:
            if getattr(self.registry, "repo_root", None):
                from .test_impact_selector import git_status_test_files
                _pre = git_status_test_files(self.registry.repo_root)
                self._pre_existing_dirty_files = set(_pre) if _pre else set()
        except Exception:
            pass  # stays None → filter skipped (no exclusion)

    def _resolve_routing_intent(self, route) -> str:
        """Resolve routing intent from route / config in one place."""
        if getattr(self.config, 'design_chat_mode', False):
            return "read_only"
        _ir = getattr(route, "intent_result", None) if route else None
        return routing_intent_from_intent_result(_ir)

    def _build_planner_summary(
        self,
        request: str,
        op_plan,
        exec_result: dict,
        completed: int,
        failed: int,
    ) -> str:
        """Build a human-readable PLANNER execution summary via lightweight LLM call, with structured fallback."""
        # ANALYZE_FIRST abort: surface semantic-fit reason verbatim so the user
        # understands why no change was made.
        _plan_meta = getattr(op_plan, "metadata", None) or {}
        if isinstance(_plan_meta, dict) and _plan_meta.get("analyze_first_aborted"):
            _cause = _plan_meta.get("analyze_first_abort_cause") or "abstain"
            _abort_reason = (_plan_meta.get("analyze_first_abort_reason") or "").strip()
            _fit_reason = (_plan_meta.get("semantic_fit_reason") or "").strip()
            _parts = [
                "The requested modification could not be applied — the actual code structure "
                "differs from the request's assumptions; the planner has withheld the change.",
            ]
            if _fit_reason:
                _parts.append(f"• Reason: {_fit_reason[:280]}")
            if _abort_reason and _abort_reason not in _fit_reason:
                _parts.append(f"• Abort reason ({_cause}): {_abort_reason[:200]}")
            _parts.append(
                "Please refine the request to match the actual code structure "
                "(variable names, types, etc.) and try again."
            )
            logger.info(
                "[PLANNER_SUMMARY] analyze_first_aborted short-circuit — "
                "cause=%s reason=%r",
                _cause, _abort_reason[:120],
            )
            return "\n".join(_parts)

        _state = exec_result.get("state")
        _failed_ops = _state.failed_ops if _state else {}
        _completed_ops = _state.completed_ops if _state else {}
        op_details = []
        _write_kinds = ("modify_symbol", "insert_after_symbol", "delete_symbol", "anchor_edit", "create_file", "extract_function")
        _noop_strategies = {
            "create_file_skipped_exists", "no_op", "idempotent_skip",
            "already_satisfied", "read_soft_fail", "read_only",
        }
        _any_write_succeeded = False
        _noop_completions = []
        _write_failures = []
        for op in op_plan.operations:
            _is_failed = op.id in _failed_ops
            _is_done = op.id in _completed_ops
            _op_result = _completed_ops.get(op.id) or {}
            _op_strategy = _op_result.get("strategy", "")
            # already_satisfied is only a noop when confidence is HIGH (AST-verified).
            # LOW/MEDIUM (convergence rescue, LLM no-diff) do not suppress write success tracking.
            _sat_conf = _op_result.get("satisfaction_confidence")
            _is_noop = _op_strategy in _noop_strategies or bool(
                _op_result.get("already_satisfied")
                and _sat_conf == "high"
            )
            # Low/medium confidence: op passed but warrants further investigation
            _is_weak_noop = (
                not _is_noop
                and isinstance(_op_result, dict)
                and _op_result.get("already_satisfied")
                and _sat_conf in ("low", "medium")
            )
            _op_status = "✓" if _is_done else ("✗" if _is_failed else "?")
            _err_suffix = ""
            if _is_failed and op.id in _failed_ops:
                _err_raw = (_failed_ops[op.id] or "")
                if "placement_violation" in _err_raw or "placement contract violated" in _err_raw:
                    # Extract the variable name from the violation message if present.
                    import re as _re_ps
                    _var_match = _re_ps.search(r"'(\w+)' assigned after target", _err_raw)
                    _var_hint = f" after `{_var_match.group(1)}` was computed" if _var_match else " after the relevant variable was assigned"
                    _err_reason = (
                        f"guard insertion position must be moved{_var_hint} "
                        f"(insertion point is before the dependency variable assignment)"
                    )
                elif "ast_fallback rejected" in _err_raw:
                    _err_reason = _err_raw.split("ast_fallback rejected:")[-1].strip()[:120]
                    _err_reason = _err_reason.replace("\n", " ")
                else:
                    _err_reason = _err_raw[:120].replace("\n", " ")
                _err_suffix = f" — {_err_reason}"
            elif _is_noop:
                _err_suffix = " — no change needed — already satisfied"
            elif _is_weak_noop:
                _sat_src = (_op_result.get("satisfaction_source") or "unknown") if isinstance(_op_result, dict) else "unknown"
                _err_suffix = f" — change was not applied and requires additional verification (source={_sat_src})"
            op_details.append(
                f"- [{_op_status}] {op.kind.value}: {op.symbol or ''} ({op.path or ''}){_err_suffix}\n"
            )
            if op.kind.value in _write_kinds:
                if _is_done and not _is_noop and not _is_weak_noop:
                    _any_write_succeeded = True
                elif _is_done and _is_noop:
                    _noop_completions.append((op.kind.value, op.path or op.symbol or "", _op_strategy))
                elif _is_done and _is_weak_noop:
                    _noop_completions.append((op.kind.value, op.path or op.symbol or "", f"{_op_strategy}[weak]"))
                elif _is_failed:
                    _label = op.symbol or op.path or op.kind.value
                    _err_reason = (_failed_ops.get(op.id) or "unknown error")[:120]
                    _write_failures.append(f"{op.kind.value}({_label}): {_err_reason}")

        raw_output = exec_result.get("output", "")

        if failed == 0 and _any_write_succeeded:
          _status = "success"
        elif failed == 0 and _noop_completions:
          _status = "no change — already exists"
        elif failed == 0:
          _status = "success"
        elif _any_write_succeeded:
          _status = f"partial success ({completed} done, {failed} failed)"
        else:
          _status = f"failed — no code changes ({failed} failed)"
        _files = set(op.path for op in op_plan.operations if op.path)
        fallback_parts = [f"[{_status}]"]
        if _write_failures:
            for wf in _write_failures:
                fallback_parts.append(f"• Failed: {wf}")
        elif _noop_completions:
            for _nk, _np, _ns in _noop_completions:
                if _ns.endswith("[weak]"):
                    fallback_parts.append(f"• Change not applied (further verification needed): {_np}")
                else:
                    fallback_parts.append(f"• Already exists (no change): {_np}")
        elif _files:
            fallback_parts.append(f"• Target files: {', '.join(_files)}")
        fallback = "\n".join(fallback_parts)

        try:
            _planner_client = getattr(self, '_planner_agent', None)
            _client = _planner_client._client if _planner_client else self.llm_client
            _model = _planner_client._model if _planner_client else self.model

            _failure_note = ""
            if _write_failures:
                _changed_note = (
                    " (some files were created/read successfully)"
                    if _any_write_succeeded else " No code was actually changed."
                )
                _failure_note = (
                    f"\n\nIMPORTANT: The following operations FAILED.{_changed_note}\n"
                    f"Failed ops: {'; '.join(_write_failures)}\n"
                    "Only report failures listed above — do NOT invent additional failure reasons."
                )
            elif not _any_write_succeeded and _noop_completions:
                _noop_desc = "; ".join(
                    f"{kind}({path}) already existed" for kind, path, _ in _noop_completions
                )
                _failure_note = (
                    f"\n\nIMPORTANT: No files were created or modified. "
                    f"The target(s) already existed and were left unchanged: {_noop_desc}. "
                    "Report this as 'file already exists' — do NOT claim the file was newly created."
                )
            elif not _any_write_succeeded and failed > 0:
                _failure_note = (
                    "\n\nIMPORTANT: No write operations succeeded. The file was NOT modified."
                )
            elif not _any_write_succeeded and failed == 0 and getattr(op_plan, "mode", None) and op_plan.mode.value == "edit":
                # Edit-mode plan but all completed ops were read-only — no code modified.
                _failure_note = (
                    "\n\nIMPORTANT: No code modification was performed. "
                    "All executed operations were read-only (read_symbol / bash cat). "
                    "The file was NOT changed. Do NOT claim any modification was made."
                )

            _sys = (
                "You are a concise coding assistant. Summarize the operation result in 1-2 sentences. "
                "Be HONEST about failures — if an edit failed, say exactly which operation failed and why, "
                "using ONLY the information provided. Do NOT invent failure reasons not listed. "
                "Do NOT claim no changes were made when the execution facts show files were modified "
                "or operations completed successfully. "
                "Explain in the same language as the user's request. No markdown headers."
            )
            _exec_facts: list[str] = []
            _exec_facts.append(f"Mode: {(getattr(op_plan, 'mode', None) and op_plan.mode.value) or 'unknown'}")
            _noop_count = len(_noop_completions)
            _exec_facts.append(f"Operations: {completed} completed, {failed} failed, {_noop_count} no-op")
            _patch_count = len(exec_result.get("modified_files", []) or [])
            _exec_facts.append(f"Files modified: {_patch_count}")
            _exec_facts_str = "\n".join(f"  {f}" for f in _exec_facts)
            _user = (
                f"Request: {request}\n\n"
                f"Operations (each line shows [✓=success/✗=failed/?=skipped] kind: symbol (file) — error if failed):\n"
                f"{''.join(op_details)}\n"
                f"Execution facts:\n{_exec_facts_str}\n"
                f"Details: {raw_output[:2000]}"
                f"{_failure_note}"
            )
            _total_ops = len(_completed_ops) + len(_failed_ops)

            # Input-size-aware output budget (Korean ~3 chars/token, English ~4).
            # Reserve = min(context_limit - input - safety, hard_cap).
            _input_chars = len(_sys) + len(_user)
            _est_input_tokens = _input_chars // 3
            _ctx_limit = _resolve_context_limit(_model)
            # Floor at 1024: 200 tokens is too few for any meaningful summary.
            _output_budget = max(1024, _ctx_limit - _est_input_tokens - 128)
            _max_tokens = min(_output_budget, config.tokens.PLANNER_SUMMARY)
            logger.debug(
                "[PLANNER_SUMMARY] budget: input_chars=%d est_input_tok=%d "
                "ctx=%d output_budget=%d max_tokens=%d",
                _input_chars, _est_input_tokens, _ctx_limit, _output_budget, _max_tokens,
            )

            _summary_msgs = [
                LLMMessage(role="system", content=_sys),
                LLMMessage(role="user", content=_user),
            ]
            resp = _client.chat(
                messages=_summary_msgs,
                model=_model,
                temperature=0.0,
                max_tokens=_max_tokens,
            )
            _finish_reason = getattr(resp, "finish_reason", None) or ""
            summary = effective_content(resp).strip()

            # finish_reason=length → one retry with a stripped-down prompt
            if _finish_reason == "length":
                logger.warning(
                    "[PLANNER_SUMMARY] finish_reason=length (max_tokens=%d, "
                    "input_chars=%d) — retrying with minimal prompt",
                    _max_tokens, _input_chars,
                )
                _minimal_user = (
                    f"Request: {request[:120]}\n"
                    f"Result: {completed} completed, {failed} failed.\n"
                    + (f"Failed: {'; '.join(_write_failures[:5])}\n" if _write_failures else "")
                    + "Summarize the outcome in 1 sentence."
                )
                _retry_resp = _client.chat(
                    messages=[
                        LLMMessage(role="system", content=_sys),
                        LLMMessage(role="user", content=_minimal_user),
                    ],
                    model=_model,
                    temperature=0.0,
                    max_tokens=config.tokens.PLANNER_SUMMARY_REPAIR,
                )
                _retry_reason = getattr(_retry_resp, "finish_reason", None) or ""
                _retry_text = (getattr(_retry_resp, "content", "") or "").strip()
                if _retry_text and _retry_reason != "length":
                    summary = _retry_text
                    logger.info("[PLANNER_SUMMARY] retry succeeded (reason=%s)", _retry_reason)
                else:
                    # Both attempts truncated — keep best fragment with marker
                    summary = (summary or _retry_text or fallback) + "…"
                    logger.warning("[PLANNER_SUMMARY] retry also truncated — using fallback fragment")

            # LLM summary passed through as-is. No system override applied.

            if summary and len(summary) > 10:
                return summary
        except Exception as exc:
            logger.debug("PLANNER summary LLM call failed, using fallback: %s", exc)

        return fallback

    def _init_hybrid_components(self) -> None:
        """Lazy initialization of hybrid architecture components."""
        if self._operation_executor is not None or getattr(self, '_hybrid_init_failed', False):
            return

        try:
            repo_root = getattr(self.registry, 'repo_root', '.')
            self._symbol_searcher = SymbolSearcher(repo_root)
            # config=self.config (not a captured cancel_event value) so the
            # indexer reads config.cancel_event FRESH at build() time — the
            # orchestrator mutates config.cancel_event per task and a captured
            # value would go stale. Matches tool_registry's wiring.
            _cgi = CallGraphIndexer(repo_root, config=self.config)
            self._call_graph = RepositoryGraphFacade(
                call_graph_indexer=_cgi,
                repo_root=repo_root,
            )
            self.registry.add_write_success_callback(_clear_git_cache)

            _planner_client = self.config.planner_llm_client or self.llm_client
            _planner_model = self.config.planner_model or self.model
            self._planner_agent = PlannerAgent(
                llm_client=_planner_client,
                model=_planner_model,
                max_tasks=50,
                callback=self._cb,
                config=self.config,
                run_store=self._shared_run_store,
                repo_root=self.registry.repo_root,
                post_read_max_ops=getattr(self.config, 'post_read_max_ops', 6),
            )

            self._operation_executor = OperationExecutor(
                tool_registry=self.registry,
                symbol_searcher=self._symbol_searcher,
                call_graph=self._call_graph,
                before_op_hook=self._before_operation_hook,
                after_op_hook=self._after_operation_hook,
                on_error_hook=self._on_operation_error,
                llm_generator=lambda messages: self._llm_call_simple(
                    [LLMMessage(role=m["role"], content=m["content"]) if isinstance(m, dict) else m for m in messages]
                ),
                llm_json_generator=lambda messages: self._llm_call_simple(
                    [LLMMessage(role=m["role"], content=m["content"]) if isinstance(m, dict) else m for m in messages],
                    json_mode=True,
                ),
                run_store=self._shared_run_store,
                planner=self._planner_agent,
                agent_loop_factory=self._create_scoped_agent_loop,
                config=self.config,
            )

            logger.info("Hybrid architecture components initialized")
        except Exception as exc:
            self._hybrid_init_failed = True
            logger.warning("Failed to initialize hybrid components: %s", exc)

    def _build_planner_fallback_context(
        self, exc, _spec=None, op_plan=None, exec_result=None,
    ) -> str:
        """Build fallback context from PLANNER findings so MAIN_AGENT can skip redundant exploration."""
        parts = [
            f"\n\n[PLANNER FALLBACK] The structured planner could not complete this task "
            f"(reason: {exc}). You must use the traditional tool-use approach: "
            f"read relevant files with bash (cat), then create/modify files with apply_patch. "
            f"This is a CODE EDITING task — you must produce actual code changes."
        ]

        if _spec is not None:
            _target_files = getattr(_spec, 'target_files', None) or []
            _target_symbols = getattr(_spec, 'target_symbols', None) or []
            _new_files = getattr(_spec, 'new_files', None) or []
            if _target_files or _target_symbols:
                parts.append("\n[PLANNER CONTEXT — already resolved, skip find_symbol for these]")
                if _target_files:
                    parts.append(f"  Target files: {', '.join(str(f) for f in _target_files)}")
                if _target_symbols:
                    parts.append(f"  Target symbols: {', '.join(str(s) for s in _target_symbols)}")
                if _new_files:
                    parts.append(f"  New files to create: {', '.join(str(f) for f in _new_files)}")

        if op_plan is not None:
            ops = getattr(op_plan, 'operations', None) or []
            if ops:
                parts.append(f"\n[PLANNER INTENDED OPERATIONS — {len(ops)} total]")
                for i, op in enumerate(ops, 1):
                    _kind = getattr(op, 'kind', '?')
                    _kind_val = _kind.value if hasattr(_kind, 'value') else str(_kind)
                    _sym = getattr(op, 'symbol', '') or ''
                    _path = getattr(op, 'path', '') or ''
                    _intent = getattr(op, 'intent', '') or ''
                    _desc = f"{_kind_val}"
                    if _sym:
                        _desc += f" {_sym}"
                    if _path:
                        _desc += f" in {_path}"
                    if _intent:
                        _desc += f" — {_intent[:80]}"
                    parts.append(f"  {i}. {_desc}")

        if exec_result is not None and isinstance(exec_result, dict):
            _completed_ids = exec_result.get("completed_ids", [])
            _failed_ids = exec_result.get("failed_ids", [])
            if _completed_ids:
                parts.append("\n[COMPLETED OPERATIONS — do NOT redo these]")
                parts.append(f"  Completed: {', '.join(str(x) for x in _completed_ids)}")
            if _failed_ids:
                parts.append("\n[FAILED OPERATIONS — focus on these]")
                parts.append(f"  Failed: {', '.join(str(x) for x in _failed_ids)}")

        # ── Symbol-not-found details ───────────────────────────────────────
        # When the planner aborted due to unresolved symbols, surface the
        # missing symbol names and nearest candidates so MAIN_AGENT can
        # skip redundant find_symbol calls and use the correct identifiers.
        if isinstance(exec_result, dict):
            _snf = exec_result.get("symbol_not_found") or exec_result.get("metadata", {}).get("symbol_not_found")
            if _snf:
                parts.append("\n[SYMBOL NOT FOUND — unresolved symbols]")
                for _entry in (_snf if isinstance(_snf, list) else [_snf]):
                    _sym = _entry.get("symbol", _entry.get("name", "?")) if isinstance(_entry, dict) else str(_entry)
                    _candidates = _entry.get("candidates", []) if isinstance(_entry, dict) else []
                    _cand_str = f" (nearest: {', '.join(str(c) for c in _candidates[:3])})" if _candidates else ""
                    _tf = _entry.get("target_file", "") if isinstance(_entry, dict) else ""
                    _tf_str = f" in {_tf}" if _tf else ""
                    parts.append(f"  ❌ {_sym}{_tf_str}{_cand_str}")

        return "\n".join(parts)

    def _create_scoped_agent_loop(self, scope) -> Any:
        """Create and run a scope-restricted AgentLoop for delegation."""
        try:
            from .tool_registry import ScopedToolFilter
            write_filter = ScopedToolFilter(
                allowed_write=set(scope.files),
                readonly_files=set(scope.readonly_files),
            )
            scoped_registry = self.registry.clone_with_filter(write_filter)

            augmented_request = (
                f"{scope.goal}\n\n"
                f"## Already Completed (DO NOT redo or revert these changes)\n"
                f"{scope.context}\n\n"
                f"## Allowed files to modify: {', '.join(scope.files) or 'any'}\n"
                f"## Read-only files: {', '.join(scope.readonly_files) or 'none'}\n"
            )

            _del_llm_client = getattr(self.config, "developer_llm_client", None) or self.llm_client
            _del_model = getattr(self.config, "developer_model", "") or self.model
            _del_label = f"{_del_model}" + (" (developer)" if _del_llm_client is not self.llm_client else "")

            self._cb("pipeline_stage", {
                "stage": "scoped_delegation",
                "status": "running",
                "detail": f"Delegating to agent ({_del_label}): {scope.goal[:100]}",
            })

            scoped_config = AgentConfig(
                max_turns=scope.max_turns,
                planning_enabled=False,
                self_review_enabled=False,
                rag_enabled=False,
            )
            scoped_config.stream_callback = self.config.stream_callback
            scoped_config.cancel_event = self.config.cancel_event

            scoped_loop = AgentLoop(
                llm_client=_del_llm_client,
                registry=scoped_registry,
                config=scoped_config,
                model=_del_model,
                agent_id="scoped_delegate",
            )

            result = scoped_loop.run(augmented_request)

            self._cb("pipeline_stage", {
                "stage": "scoped_delegation",
                "status": "completed",
                "detail": f"Delegation finished: {result.status if result else 'no result'}",
            })

            return result
        except Exception as exc:
            logger.warning("Scoped delegation failed: %s", exc, exc_info=True)
            self._cb("pipeline_stage", {
                "stage": "scoped_delegation",
                "status": "error",
                "detail": str(exc),
            })
            return None

    def _before_operation_hook(self, operation, state):
        """Emit operation_start event and save pre-op file snapshot for quality gate rollback."""
        self._cb("operation_start", {
            "operation_id": operation.id,
            "kind": operation.kind.value,
            "symbol": operation.symbol or "",
            "file_path": operation.path or "",
            "path": operation.path or "",
            "decomposition_guard": bool((getattr(operation, "metadata", None) or {}).get("decomposition_guard", False)),
            "missing_callee": (getattr(operation, "metadata", None) or {}).get("missing_callee", ""),
        })
        if (operation.path and
                operation.kind in (
                    OperationKind.MODIFY_SYMBOL, OperationKind.INSERT_AFTER_SYMBOL,
                    OperationKind.EXTRACT_FUNCTION, OperationKind.INSERT_AFTER_LINE,
                    OperationKind.ANCHOR_EDIT, OperationKind.DELETE_SYMBOL_RANGE,
                    OperationKind.UPDATE_CALLERS, OperationKind.UPDATE_TESTS,
                    OperationKind.INSERT_IMPORT, OperationKind.REMOVE_IMPORT,
                    OperationKind.REMOVE_IMPORT_NAME, OperationKind.ADD_ASSIGN,
                    OperationKind.MOVE_SYMBOL, OperationKind.OVERWRITE_FILE,
                    OperationKind.REPLACE_FILE,
                )):
            try:
                _snap_root = str(self.registry.repo_root) if hasattr(self.registry, 'repo_root') else '.'
                _snap_abs = (
                    operation.path if os.path.isabs(operation.path)
                    else os.path.join(_snap_root, operation.path)
                )
                if os.path.isfile(_snap_abs):
                    with open(_snap_abs, encoding='utf-8', errors='replace') as _sf:
                        # Accumulate (not overwrite): recursive re-entry with the same op.id
                        # may have a different file path; overwriting loses the original snapshot.
                        if operation.id not in state.pre_op_file_snapshots:
                            state.pre_op_file_snapshots[operation.id] = {}
                        if _snap_abs not in state.pre_op_file_snapshots[operation.id]:
                            state.pre_op_file_snapshots[operation.id][_snap_abs] = _sf.read()
            except OSError:
                pass

    def _after_operation_hook(self, operation, state, result):
        """Emit operation_complete event and run quality gate for edit ops."""
        _already_satisfied = bool(result.get("already_satisfied", False))
        _is_noop = _already_satisfied or result.get("strategy", "") in {
            "create_file_skipped_exists", "no_op", "idempotent_skip",
            "read_only",
        }
        # ── patch_preview for incremental diff display during execution ──
        # Priority: compute only this op's actual changes from pre-op snapshot vs current file content
        # (always clean git-diff format). Falls back to the handler's patch_applied string when
        # no snapshot exists (e.g., create_file).
        _patch_preview = ""
        _diff_max_chars = config.display.INLINE_OP_DIFF_MAX_CHARS
        # Skip snapshot re-read (file I/O) when the toggle is off. (CLI rendering also disabled.)
        if config.display.INLINE_OP_DIFF:
            try:
                import difflib as _difflib
                _snap_root = str(self.registry.repo_root) if hasattr(self.registry, 'repo_root') else '.'
                _snaps = (getattr(state, "pre_op_file_snapshots", None) or {}).get(operation.id) or {}
                _diff_parts = []
                for _abs, _before in _snaps.items():
                    if not os.path.isfile(_abs):
                        continue
                    try:
                        with open(_abs, encoding='utf-8', errors='replace') as _cf:
                            _after = _cf.read()
                    except OSError:
                        continue
                    if _after == _before:
                        continue
                    try:
                        _rel = os.path.relpath(_abs, _snap_root)
                    except ValueError:
                        _rel = _abs
                    # splitlines (without keepends) + '\n'.join → cleanly separated header/@@/body
                    # in git-diff format, ready for CLI's _build_file_diff_renderable to parse directly.
                    _diff_parts.append("\n".join(_difflib.unified_diff(
                        _before.splitlines(),
                        _after.splitlines(),
                        fromfile=f"a/{_rel}", tofile=f"b/{_rel}", lineterm="",
                    )))
                _patch_preview = ("\n".join(p for p in _diff_parts if p))[:_diff_max_chars]
            except Exception:
                _patch_preview = ""
        if not _patch_preview.strip():
            _raw_patch = result.get("patch_applied")
            if isinstance(_raw_patch, str) and _raw_patch.strip():
                _patch_preview = _raw_patch[:_diff_max_chars]
        self._cb("operation_complete", {
            "operation_id": operation.id,
            "kind": operation.kind.value,
            "result_status": result.get("status", "unknown"),
            "file_path": operation.path or "",
            "symbol": operation.symbol or "",
            "patch_preview": _patch_preview,
            # Idempotency signals — visible in logs for post-hoc verification
            "already_satisfied": _already_satisfied,
            "noop": _is_noop,
            "error_detail": result.get("error") or result.get("failure_class") or "",
            "strategy": result.get("strategy", ""),
            "final_failure_class": result.get("final_failure_class", ""),
            "action_hint": (getattr(operation, "metadata", None) or {}).get("action_hint", ""),
            "decomposition_guard": bool((getattr(operation, "metadata", None) or {}).get("decomposition_guard", False)),
            "missing_callee": (getattr(operation, "metadata", None) or {}).get("missing_callee", ""),
        })

        if hasattr(state, 'checkpoint_log') and state.checkpoint_log:
            _latest_ckpt = state.checkpoint_log[-1]
            if _latest_ckpt.operation_id == operation.id:
                self._cb("execution_checkpoint", {
                    "operation_id": _latest_ckpt.operation_id,
                    "validity_score": _latest_ckpt.validity_score.value,
                    "remaining_anchors_valid": _latest_ckpt.remaining_anchors_valid,
                    "invalidated_op_ids": _latest_ckpt.invalidated_op_ids,
                    "downstream_impacted_ops": _latest_ckpt.downstream_impacted_ops,
                    "compile_ok": _latest_ckpt.compile_ok,
                    "replan_count": state.replan_count,
                })

        if (operation.kind in (
                OperationKind.MODIFY_SYMBOL,
                OperationKind.INSERT_AFTER_LINE,
                OperationKind.INSERT_AFTER_SYMBOL,
                OperationKind.ANCHOR_EDIT,
                OperationKind.EXTRACT_FUNCTION,
                # File-producing ops were previously skipped, so a created/
                # overwritten file (e.g. a JS→TS migration emitting server.ts)
                # passed with ZERO syntax/compile validation. The gate already
                # has a CREATE_FILE branch and a tsc/py_compile syntax check —
                # they were just never reached. Wire them in.
                OperationKind.CREATE_FILE,
                OperationKind.OVERWRITE_FILE,
            ) and state.plan_mode == PlanMode.EDIT):
            _is_last = isinstance(result, dict) and result.get("_is_last_edit_op", False)
            self._run_quality_gate(operation, state, result, is_last_op=_is_last)

    def _check_compatibility(
        self,
        file_path: str,
        symbol: Optional[str] = None,
        content: Optional[str] = None,
    ) -> list[tuple[str, str]]:
        """Check AST syntax, function signature consistency, and import validity. Returns (type, msg) list.

        content: if provided, check this instead of reading from disk (for baseline diff).
        """
        issues = []
        from ..languages import LanguageRegistry
        _provider = LanguageRegistry.instance().get(file_path)
        if not _provider or not _provider.capabilities().has_ast_parser:
            return issues

        try:
            import ast
            if content is None:
                with open(file_path, encoding='utf-8') as f:
                    content = f.read()

            try:
                tree = ast.parse(content, filename=file_path)
            except SyntaxError as e:
                issues.append(("syntax", f"AST parsing failed: {e}"))
                return issues

            if symbol:
                function_defs = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == symbol:
                        function_defs.append(node)
                    elif isinstance(node, ast.AsyncFunctionDef) and node.name == symbol:
                        function_defs.append(node)

                if function_defs:
                    for func_def in function_defs:
                        arg_names = [arg.arg for arg in func_def.args.args]
                        if len(arg_names) != len(set(arg_names)):
                            issues.append(("signature", f"Duplicate parameter names in function '{symbol}'"))

            # Only flag relative private imports (absolute private imports are fine in monorepos)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level > 0:
                    module = node.module or ""
                    for alias in node.names:
                        if alias.name.startswith('_'):
                            issues.append((
                                "import",
                                f"Relative import of private name '{alias.name}' "
                                f"from '{module or '.'}'"
                            ))

        except Exception as exc:
            logger.warning(f"Compatibility check failed for {file_path}: {exc}")
            issues.append(("compatibility_check_error", str(exc)))

        return issues

    def _execute_plan_sequence_direct(
        self,
        plan_sequence: list,
        original_request: str,
        context: str,
        stage_ctx: "Optional[StageContext]" = None,
    ) -> dict:
        """Execute per-file OperationPlans sequentially and return aggregated result dict."""
        _total_completed = _total_failed = _total_skipped = _total_noop = 0
        _all_completed_ids: list = []
        _all_failed_ids: list = []
        _all_skipped_ids: list = []
        _all_modified_files: list = []
        _all_blocking_reasons: list = []
        _all_warning_reasons: list = []
        _outputs: list = []
        _first_failure_class = ""
        _last_state = None
        _last_raw: dict = {}
        _file_results: list = []
        _n = len(plan_sequence)

        for _idx, _file_plan in enumerate(plan_sequence, 1):
            # ── Cancel check between file-plan executions ──
            if self.config.cancel_event and self.config.cancel_event.is_set():
                raise AgentCancelled("cancelled by user during plan execution")
            _file = (_file_plan.metadata or {}).get("workset_file", "") or ""
            _short = _file.rsplit("/", 1)[-1] if _file else f"plan-{_idx}"
            logger.info("[SEQ_EXEC] file-plan %d/%d: %s (%d ops)",
                        _idx, _n, _short, len(_file_plan.operations))
            self._cb("pipeline_stage", {
                "stage": "execution", "status": "running",
                "detail": f"[{_idx}/{_n}] {_short}: {len(_file_plan.operations)} ops",
            })
            _res = self._operation_executor.execute_plan(
                _file_plan,
                original_request=original_request,
                context=context,
                max_operations=50,
                stage_ctx=stage_ctx,
            )
            _total_completed += _res.get("completed", 0)
            _total_failed += _res.get("failed", 0)
            _total_skipped += _res.get("skipped", 0)
            _total_noop += _res.get("noop_ops", 0)
            _all_completed_ids.extend(_res.get("completed_ids") or [])
            _all_failed_ids.extend(_res.get("failed_ids") or [])
            _all_skipped_ids.extend(_res.get("skipped_ids") or [])
            _all_modified_files.extend(_res.get("modified_files") or [])
            _all_blocking_reasons.extend(_res.get("final_blocking_reasons") or [])
            _all_warning_reasons.extend(_res.get("final_warning_reasons") or [])
            if _res.get("output"):
                _outputs.append(_res["output"])
            if not _first_failure_class and _res.get("final_failure_class"):
                _first_failure_class = _res["final_failure_class"]
            _last_state = _res.get("state", _last_state)
            _last_raw = _res
            _file_results.append({"file": _file, "result": _res})

        if _total_failed == 0 and _total_completed > 0:
            _final_status = "completed"
        elif _total_completed > 0:
            _final_status = "completed_partial"
        elif _total_failed > 0:
            _final_status = "failed"
        else:
            _final_status = "completed"  # all skipped/noop
        logger.info("[SEQ_EXEC] done: %d plans, completed=%d failed=%d skipped=%d status=%s",
                    _n, _total_completed, _total_failed, _total_skipped, _final_status)

        return {
            "completed": _total_completed,
            "failed": _total_failed,
            "skipped": _total_skipped,
            "noop_ops": _total_noop,
            "completed_ids": _all_completed_ids,
            "failed_ids": _all_failed_ids,
            "skipped_ids": _all_skipped_ids,
            "modified_files": list(dict.fromkeys(_all_modified_files)),
            "output": "\n".join(o for o in _outputs if o),
            "final_status": _final_status,
            "final_failure_class": _first_failure_class,
            "final_blocking_reasons": _all_blocking_reasons,
            "final_warning_reasons": _all_warning_reasons,
            "state": _last_state,
            "raw": _last_raw,
            "sequential_execution": True,
            "file_results": _file_results,
        }

    def _run_quality_gate(self, operation, state, result, is_last_op: bool = False):
        """Run syntax/lint/compatibility/test checks after an edit op. Syntax errors are critical."""
        if self.slash_commands:
            _op_desc = getattr(operation, 'description', None) or getattr(operation, 'intent', '') or ''
            slash_cmd = self.slash_commands.detect_slash_command(_op_desc)
            if slash_cmd:
                _new_desc = self.slash_commands.generate_prompt(slash_cmd, _op_desc)
                if hasattr(operation, 'description'):
                    operation.description = _new_desc
                elif hasattr(operation, 'intent'):
                    operation.intent = _new_desc
        logger.info(f"Running quality gate after {operation.kind.value} operation {operation.id}")

        target_files = []
        if operation.path:
            target_files.append(operation.path)
        elif isinstance(result, dict) and result.get("instruction"):
            instr = result["instruction"]
            if hasattr(instr, 'file_path'):
                target_files.append(instr.file_path)

        repo_root = str(self.registry.repo_root) if hasattr(self.registry, 'repo_root') else '.'
        resolved_files = []
        for fp in target_files:
            abs_fp = os.path.join(repo_root, fp) if not os.path.isabs(fp) else fp
            if os.path.isfile(abs_fp):
                resolved_files.append(abs_fp)
            else:
                _qg_skip_dirs = {
                    "node_modules", "__pycache__", ".venv", "venv", "env",
                    "dist", "build", ".git", ".next", ".cache", "target",
                }
                for dirpath, dirnames, filenames in _qg_os.walk(repo_root, topdown=True):
                    rel = _qg_os.path.relpath(dirpath, repo_root)
                    if any(p.startswith('.') for p in rel.split(_qg_os.sep)):
                        continue
                    # Prune vendored/build dirs in-place so walk doesn't descend
                    # (avoids matching basenames like index.js inside node_modules).
                    dirnames[:] = [d for d in dirnames if d not in _qg_skip_dirs]
                    if _qg_os.path.basename(fp) in filenames:
                        full = _qg_os.path.join(dirpath, _qg_os.path.basename(fp))
                        resolved_files.append(full)
                        logger.info("Quality gate: resolved %r → %s", fp, full)
                        break
                else:
                    logger.warning("Quality gate: file not found: %s", fp)
        target_files = resolved_files

        critical_failure = False
        failures = []

        # 0. CREATE_FILE: verify the new file is in a valid Python package directory.
        # Python-only: the package/__init__.py checks below are meaningless for
        # .ts/.json/.html/etc. and would emit spurious "not a valid Python
        # package" warnings now that non-Python creates reach this gate.
        if (operation.kind == OperationKind.CREATE_FILE and operation.path
                and LanguageId.from_path(operation.path) is LanguageId.PYTHON):
            _cf_path = operation.path
            _has_dir = ('/' in _cf_path or _qg_os.sep in _cf_path)
            if not _has_dir:
                _warn = (
                    f"CREATE_FILE '{_cf_path}' has no directory component — "
                    f"new module will be created in repo root instead of its package directory. "
                    f"Expected path like 'external_llm/agent/{_cf_path}'"
                )
                logger.warning(_warn)
                failures.append(("create_file_wrong_path", _warn))
                self._cb("quality_gate_compatibility_issue", {
                    "operation_id": operation.id,
                    "file": _cf_path,
                    "issue_type": "create_file_wrong_path",
                    "issue_msg": _warn,
                })
            else:
                _abs_cf = _qg_os.path.join(repo_root, _cf_path) if not _qg_os.path.isabs(_cf_path) else _cf_path
                _parent_dir = _qg_os.path.dirname(_abs_cf)
                _init_py = _qg_os.path.join(_parent_dir, "__init__.py")
                if _qg_os.path.isdir(_parent_dir) and not _qg_os.path.isfile(_init_py):
                    _warn = (
                        f"CREATE_FILE '{_cf_path}': parent directory has no __init__.py "
                        f"— may not be a valid Python package"
                    )
                    logger.warning(_warn)
                    failures.append(("create_file_no_package", _warn))
                    self._cb("quality_gate_compatibility_issue", {
                        "operation_id": operation.id,
                        "file": _cf_path,
                        "issue_type": "create_file_no_package",
                        "issue_msg": _warn,
                    })

        from ..languages import LanguageRegistry as _LR
        for file_path in target_files:
            _qg_provider = _LR.instance().get(file_path)
            if _qg_provider and _qg_provider.capabilities().has_syntax_validator:
                try:
                    with open(file_path, encoding='utf-8', errors='replace') as _f:
                        _content = _f.read()
                    _val = _qg_provider.validate_syntax(file_path, _content)
                    if not _val.ok:
                        _err = _val.errors[0].message if _val.errors else "syntax error"
                        error_msg = f"Syntax error in {file_path}: {_err}"
                        logger.warning(error_msg)

                        # Check if this syntax error pre-existed before the op.
                        # If the op made 0 changes (e.g. ALREADY_SATISFIED / noop),
                        # a syntax error is NOT op-caused — don't flag as critical.
                        _pre_content = (
                            (getattr(state, "pre_op_file_snapshots", None) or {})
                            .get(operation.id, {})
                            .get(file_path)
                        )
                        _is_pre_existing = False
                        if _pre_content is not None:
                            try:
                                _pre_val = _qg_provider.validate_syntax(file_path, _pre_content)
                                _is_pre_existing = not _pre_val.ok
                            except Exception as _pre_exc:
                                # Infrastructure failure (file not found, permission error, etc.)
                                # is NOT an operation-caused error — treat as pre-existing
                                logger.debug(
                                    "[QG_PREEXISTING] validate_syntax infrastructure failure for %s: %s",
                                    file_path, _pre_exc,
                                )
                                _is_pre_existing = True
                        if _is_pre_existing:
                            logger.info(
                                "[QG_PREEXISTING] syntax error in %s pre-existed before op %s — "
                                "not treating as critical failure",
                                file_path, operation.id,
                            )
                        else:
                            # ── Auto-fix registry (pluggable) ───────────────────
                            # Iterate _AUTO_FIX_REGISTRY; first fixer whose
                            # .matches() returns True gets to .apply().  If the
                            # fix produces valid syntax, skip critical failure.
                            _auto_fixed = False
                            _lang = LanguageId.from_path(file_path)
                            for _fixer in _AUTO_FIX_REGISTRY:
                                if not _fixer.matches(_content, _val.errors, file_path, _lang):
                                    continue
                                try:
                                    _fixed = _fixer.apply(_content, _val.errors, file_path)
                                except Exception as _af_exc:
                                    logger.debug(
                                        "[QG_AUTO_FIX] %s.apply failed for %s: %s",
                                        type(_fixer).__name__, file_path, _af_exc,
                                    )
                                    continue
                                if _fixed is None:
                                    continue
                                try:
                                    atomic_write_text(file_path, _fixed)
                                    _re_val = _qg_provider.validate_syntax(file_path, _fixed)
                                    if _re_val.ok:
                                        logger.info(
                                            "[QG_AUTO_FIX] %s fixed %s — "
                                            "re-validated OK, skipping critical failure",
                                            type(_fixer).__name__, file_path,
                                        )
                                        state.add_read_file(file_path, _fixed)
                                        _auto_fixed = True
                                        break
                                except Exception as _af_exc:
                                    logger.debug(
                                        "[QG_AUTO_FIX] %s write/re-val failed for %s: %s",
                                        type(_fixer).__name__, file_path, _af_exc,
                                    )
                            if not _auto_fixed:
                                failures.append(("syntax", error_msg))
                                critical_failure = True
                        self._cb("quality_gate_syntax_error", {
                            "operation_id": operation.id,
                            "file": file_path,
                            "error": _err[:500],
                        })
                    else:
                        _label = "syntax-check"  # default label
                        if LanguageId.from_path(file_path) is LanguageId.PYTHON:
                            _label = "py_compile"
                        elif file_path.endswith((".ts", ".tsx")):
                            _label = "ts_syntax"
                        elif file_path.endswith((".js", ".jsx")):
                            _label = "js_syntax"
                        elif file_path.endswith(".go"):
                            _label = "go_build"
                        logger.info(f"{_label} passed for {file_path}")
                        # UnboundLocal risk check: detect variables read after
                        # conditional but not defined in all branches
                        if file_path and LanguageId.from_path(file_path) is LanguageId.PYTHON:
                            try:
                                with open(file_path, encoding="utf-8") as _ul_fh:
                                    _ul_source = _ul_fh.read()
                                from .fix_spec_claim_validator import check_edit_unbound_locals
                                _ul_warnings = check_edit_unbound_locals(
                                    _ul_source, file_path,
                                )
                                for _ul_w in _ul_warnings:
                                    logger.warning("[UNBOUND_LOCAL] %s", _ul_w)
                                    self._cb("quality_gate_unbound_local", {
                                        "operation_id": operation.id,
                                        "warning": _ul_w,
                                    })
                            except Exception as _ul_exc:
                                logger.debug(
                                    "UnboundLocal check skipped (graceful): %s", _ul_exc,
                                )
                except Exception as exc:
                    _err_label = "syntax-check"
                    if LanguageId.from_path(file_path) is LanguageId.PYTHON:
                        _err_label = "py_compile"
                    elif file_path.endswith((".ts", ".tsx")):
                        _err_label = "ts_syntax"
                    elif file_path.endswith((".js", ".jsx")):
                        _err_label = "js_syntax"
                    elif file_path.endswith(".go"):
                        _err_label = "go_build"
                    logger.warning(f"{_err_label} check failed: {exc}")
                    self._cb("quality_gate_check_error", {
                        "operation_id": operation.id,
                        "check": _err_label,
                        "error": str(exc),
                    })

        # 1b. Circular import check: only fail when the op *introduced* the cycle.
        # Pre-existing cycles are logged at INFO only.
        for _ci_file in target_files:
            if LanguageId.from_path(_ci_file) is not LanguageId.PYTHON:
                continue
            try:
                _ci_abs = (
                    _ci_file if os.path.isabs(_ci_file)
                    else os.path.join(repo_root, _ci_file)
                )
                if not os.path.isfile(_ci_abs):
                    continue
                with open(_ci_abs, encoding="utf-8", errors="replace") as _f:
                    _ci_src = _f.read()
                _ci_tree = ast.parse(_ci_src)

                _ci_imports: list = []
                for _ci_node in ast.walk(_ci_tree):
                    if isinstance(_ci_node, ast.ImportFrom) and _ci_node.module:
                        _ci_imports.append(_ci_node.module)
                    elif isinstance(_ci_node, ast.Import):
                        for _ci_alias in _ci_node.names:
                            _ci_imports.append(_ci_alias.name)

                _pre_snap_dict = getattr(state, 'pre_op_file_snapshots', {}).get(operation.id, {})
                _pre_snap_content = _pre_snap_dict.get(_ci_abs)
                _pre_imports: set = set()
                if _pre_snap_content is not None:
                    try:
                        _pre_tree = ast.parse(_pre_snap_content)
                        for _pn in ast.walk(_pre_tree):
                            if isinstance(_pn, ast.ImportFrom) and _pn.module:
                                _pre_imports.add(_pn.module)
                            elif isinstance(_pn, ast.Import):
                                for _pa in _pn.names:
                                    _pre_imports.add(_pa.name)
                    except Exception:
                        pass  # parse failure → treat all imports as new

                _ci_rel = (
                    os.path.relpath(_ci_abs, repo_root)
                    if os.path.isabs(_ci_abs) else _ci_file
                )
                _ci_mod = _ci_rel.replace(os.sep, ".").removesuffix(".py")

                for _imp_mod in set(_ci_imports):
                    if _imp_mod == _ci_mod or _ci_mod.startswith(_imp_mod + ".") or _imp_mod.startswith(_ci_mod + "."):
                        continue
                    _imp_rel = _imp_mod.replace(".", os.sep) + ".py"
                    _imp_abs = os.path.join(repo_root, _imp_rel)
                    if not os.path.isfile(_imp_abs):
                        continue
                    try:
                        with open(_imp_abs, encoding="utf-8", errors="replace") as _f2:
                            _imp_src = _f2.read()
                        _imp_tree = ast.parse(_imp_src)
                        for _imp_node in ast.walk(_imp_tree):
                            _back_mod = None
                            if isinstance(_imp_node, ast.ImportFrom) and _imp_node.module:
                                _back_mod = _imp_node.module
                            elif isinstance(_imp_node, ast.Import):
                                for _a in _imp_node.names:
                                    if _ci_mod.startswith(_a.name) or _a.name.startswith(_ci_mod):
                                        _back_mod = _a.name
                            if _back_mod and (
                                _back_mod == _ci_mod
                                or _ci_mod.startswith(_back_mod + ".")
                                or _back_mod.startswith(_ci_mod + ".")
                            ):
                                _ci_err = (
                                    f"Circular import detected: {_ci_rel} imports {_imp_mod}, "
                                    f"which imports back to {_ci_mod}. "
                                    f"Move the import inside the function body or use TYPE_CHECKING guard."
                                )
                                # Blame model: was this import already present before the op?
                                _is_preexisting = _imp_mod in _pre_imports
                                if _is_preexisting:
                                    logger.info(
                                        "[QUALITY_GATE] Pre-existing circular import "
                                        "(not introduced by op %s): %s ↔ %s — skipping critical failure",
                                        operation.id, _ci_rel, _imp_mod,
                                    )
                                else:
                                    logger.warning("[QUALITY_GATE] %s", _ci_err)
                                    failures.append(("circular_import", _ci_err))
                                    critical_failure = True
                                    self._cb("quality_gate_circular_import", {
                                        "operation_id": operation.id,
                                        "file": _ci_file,
                                        "imported_module": _imp_mod,
                                        "error": _ci_err,
                                    })
                                break
                    except Exception:
                        continue  # non-critical: outer loop continues regardless
            except Exception:
                pass  # non-critical: inner loop already continued

        # 1c. @dataclass field ordering: non-default after default is a runtime TypeError.
        for _dc_file in target_files:
            try:
                _dc_abs = (
                    _dc_file if os.path.isabs(_dc_file)
                    else os.path.join(repo_root, _dc_file)
                )
                if not os.path.isfile(_dc_abs):
                    continue
                if LanguageId.from_path(_dc_file) is not LanguageId.PYTHON:
                    continue
                with open(_dc_abs, encoding="utf-8", errors="replace") as _f:
                    _dc_src = _f.read()
                _dc_tree = ast.parse(_dc_src)

                for _dc_node in _dc_tree.body:
                    if not isinstance(_dc_node, ast.ClassDef):
                        continue
                    _is_dc = any(
                        (isinstance(d, ast.Name) and d.id == "dataclass")
                        or (isinstance(d, ast.Attribute) and d.attr == "dataclass")
                        or (isinstance(d, ast.Call) and (
                            (isinstance(d.func, ast.Name) and d.func.id == "dataclass")
                            or (isinstance(d.func, ast.Attribute) and d.func.attr == "dataclass")
                        ))
                        for d in _dc_node.decorator_list
                    )
                    if not _is_dc:
                        continue
                    _had_default = False
                    for _stmt in _dc_node.body:
                        if not isinstance(_stmt, ast.AnnAssign):
                            continue
                        _has_default = _stmt.value is not None
                        _field_name = (
                            _stmt.target.id
                            if isinstance(_stmt.target, ast.Name) else "?"
                        )
                        if _had_default and not _has_default:
                            _dc_err = (
                                f"@dataclass field ordering error in {_dc_file} class "
                                f"'{_dc_node.name}': non-default field '{_field_name}' "
                                f"follows a field with a default value. "
                                f"Move non-default fields before default fields."
                            )
                            logger.warning("[QUALITY_GATE] %s", _dc_err)
                            failures.append(("dataclass_field_order", _dc_err))
                            critical_failure = True
                            self._cb("quality_gate_dataclass_field_order", {
                                "operation_id": operation.id,
                                "file": _dc_file,
                                "class": _dc_node.name,
                                "error": _dc_err,
                            })
                            break
                        if _has_default:
                            _had_default = True
            except (AttributeError, TypeError, SyntaxError):
                # SyntaxError: file already flagged by the syntax check above —
                # can't AST-parse a broken file to inspect dataclass fields, so skip.
                # (The narrower AttributeError/TypeError were for _stmt.target.id access.)
                pass

        # 2. Lint check
        try:
            lint_args = {}
            if target_files:
                lint_args["paths"] = target_files
            lint_result = self.registry.dispatch("run_lint", lint_args)
            if not lint_result.ok:
                lint_issues = lint_result.content[:2000] if lint_result.content else "Lint check failed"
                logger.warning(f"Lint issues found: {lint_issues}")
                failures.append(("lint", lint_issues))
                self._cb("quality_gate_lint_issues", {
                    "operation_id": operation.id,
                    "lint_output": lint_issues,
                })
            else:
                logger.info(f"Lint passed for operation {operation.id}")
        except Exception as exc:
            logger.warning(f"Lint check failed: {exc}")
            self._cb("quality_gate_check_error", {
                "operation_id": operation.id,
                "check": "lint",
                "error": str(exc),
            })

        # 3. Compatibility checks — diff against pre-edit snapshot to exclude pre-existing issues.
        _snap_dict = getattr(state, 'pre_op_file_snapshots', {}).get(operation.id, {})
        for file_path in target_files:
            _compat_provider = _LR.instance().get(file_path)
            if _compat_provider and _compat_provider.capabilities().has_ast_parser:
                # Resolve absolute path used as snapshot key
                _compat_root = str(self.registry.repo_root) if hasattr(self.registry, 'repo_root') else '.'
                _compat_abs = (
                    file_path if os.path.isabs(file_path)
                    else os.path.join(_compat_root, file_path)
                )
                _pre_content = _snap_dict.get(_compat_abs) or _snap_dict.get(file_path)
                all_issues = self._check_compatibility(file_path, operation.symbol)
                if _pre_content:
                    from collections import Counter as _Counter
                    _baseline_issues = self._check_compatibility(
                        file_path, operation.symbol, content=_pre_content
                    )
                    _post_cnt = _Counter(all_issues)
                    _base_cnt = _Counter(_baseline_issues)
                    compatibility_issues = []
                    for issue, post_n in _post_cnt.items():
                        extra = post_n - _base_cnt.get(issue, 0)
                        compatibility_issues.extend([issue] * max(extra, 0))
                else:
                    compatibility_issues = all_issues
                for issue_type, issue_msg in compatibility_issues:
                    logger.warning(f"Compatibility issue in {file_path}: {issue_type} - {issue_msg}")
                    failures.append((f"compatibility_{issue_type}", issue_msg))
                    self._cb("quality_gate_compatibility_issue", {
                        "operation_id": operation.id,
                        "file": file_path,
                        "issue_type": issue_type,
                        "issue_msg": issue_msg,
                    })

        if not self.config.run_tests:
            logger.debug("Quality gate: skipping test execution (run_tests=False)")
        elif not is_last_op:
            # Full test suite is holistic & expensive — run only on the final edit op,
            # not after every file edit in a multi-op plan. Per-op fast-fail is already
            # covered by the per-file syntax + lint checks above.
            logger.debug("Quality gate: deferring test execution to last op (is_last_op=False)")
        else:
            try:
                # ── Scoped verification (opt-in via config.scoped_verification) ──
                # Run only tests likely affected by the changed files (naming-
                # convention + call-graph mapping) instead of the full suite.
                # Falls back to the full suite when selection is empty or the
                # selector is unavailable, so this can never skip verification.
                _test_dispatch_args: dict[str, Any] = {}
                _scoped: list[str] = []  # populated when scoped verification is active
                _scoped_fallback_reason: str | None = None  # why "full" was used instead of "scoped"
                if getattr(self.config, "scoped_verification", False):
                    _touched = sorted(getattr(state, "modified_files_set", set()) or [])
                    try:
                        from .test_impact_selector import (
                            git_status_test_files, select_affected_tests)

                        # Augment _touched with test files created via bash/heredoc/cp
                        # (invisible to modified_files_set — only editor handlers populate
                        #  that set).  git_status_test_files runs `git status
                        #  --porcelain -z --untracked-files=all` (~50ms, cheap relative
                        #  to tests) and returns only still-existing test files, so
                        #  deletions and rename-origin paths can't poison pytest args.
                        _repo_root = getattr(self.registry, "repo_root", None)
                        if not _repo_root:
                            raise RuntimeError("repo_root not available")  # caught by outer except
                        _git_tests = git_status_test_files(_repo_root)

                        if _git_tests and self._pre_existing_dirty_files is not None:
                            # Exclude test files that were dirty before this session
                            # started (user's WIP changes — not caused by the agent).
                            _git_tests = [
                                f for f in _git_tests
                                if f not in self._pre_existing_dirty_files
                            ]

                        if _git_tests:
                            _touched = sorted(set(_touched) | set(_git_tests))

                        # NOTE: gate-time invalidate_index was intentionally removed
                        # (P1/P4).  Write-time invalidation handles new test files.
                        # The 600s TTL suffices for bash-created files; new files
                        # are still directly selected via the naming-convention path
                        # in select_affected_tests (is_test_file check inside it),
                        # select_affected_tests call below.

                        if _touched:
                            _cg = None
                            _cgw = getattr(self.registry, "_call_graph", None)
                            if _cgw is not None:
                                _cg = getattr(_cgw, "call_graph_indexer", None)
                            _scoped = select_affected_tests(
                                _repo_root, _touched, call_graph=_cg
                            )
                            if _scoped:
                                _test_dispatch_args = {"args": _scoped}
                                logger.info(
                                    "Quality gate: scoped verification → %d test file(s) "
                                    "for %d touched file(s)",
                                    len(_scoped), len(_touched),
                                )
                            else:
                                _scoped_fallback_reason = "no_tests_selected"
                        else:
                            _scoped_fallback_reason = "no_modified_files"
                    except Exception as _se:  # noqa: BLE001 — selection must never block the gate
                        _scoped_fallback_reason = "selector_error"
                        logger.debug("Scoped verification unavailable, running full suite: %s", _se)
                else:
                    _scoped_fallback_reason = "scoped_verification_disabled"
                test_result = self.registry.dispatch("run_tests", _test_dispatch_args)
                # Include the list of selected test files in both pass/fail events so
                # stream callbacks (and downstream self-eval / dashboards) can audit
                # whether scoped verification is selecting relevant tests.
                _selected_test_files = list(_scoped)
                _verification_mode = "scoped" if _scoped else "full"
                if test_result.ok:
                    logger.info(f"Tests passed for operation {operation.id}")
                    self._cb("quality_gate_tests_passed", {
                        "operation_id": operation.id,
                        "test_output": test_result.content[:2000] if test_result.content else "",
                        "selected_test_files": _selected_test_files,
                        "verification_mode": _verification_mode,
                        "fallback_reason": _scoped_fallback_reason,
                    })
                else:
                    logger.warning(f"Tests failed for operation {operation.id}: {test_result.error}")
                    _test_output = test_result.content or ""
                    _test_error = test_result.error or "Test failure"
                    failures.append(("tests", _test_error))
                    self._cb("quality_gate_tests_failed", {
                        "operation_id": operation.id,
                        "test_error": _test_error,
                        "test_output": _test_output[:1000],
                        "is_last_op": is_last_op,
                        "selected_test_files": _selected_test_files,
                        "verification_mode": _verification_mode,
                        "fallback_reason": _scoped_fallback_reason,
                    })
            except Exception as exc:
                logger.warning(f"Test execution failed: {exc}")
                self._cb("quality_gate_check_error", {
                    "operation_id": operation.id,
                    "check": "tests",
                    "error": str(exc),
                })

        # ── Plan-level target_files integrity check (runs once on last op) ──
        if is_last_op:
            _spec = getattr(self.config, 'prebuilt_spec_for_planner', None)
            if _spec and hasattr(_spec, 'target_files') and _spec.target_files:
                _declared = set(str(f) for f in _spec.target_files)
                _actual = set(getattr(state, 'modified_files_set', set()))
                _missing = _declared - _actual
                if _missing:
                    _msg = (
                        f"Target file drop: declared target_files "
                        f"{sorted(_declared)} but no ops touched: {sorted(_missing)}"
                    )
                    logger.warning("[TARGET_FILE_DROP] %s", _msg)
                    failures.append(("target_file_drop", _msg))
                    # Declared target_files must have at least one edit op touching
                    # them. Zero-touch means the plan is incomplete — this is a
                    # critical failure that forces replan with proper context.
                    # (run_20260607_084243: server.js modify_symbol silently degraded
                    #  to read_file_segment — no write op ever touched server.js,
                    #  but final_status was 'success'.)
                    critical_failure = True

        if failures:
            failure_desc = "; ".join([f"{typ}: {msg[:500]}" for typ, msg in failures])
            if critical_failure:
                # Syntax errors: revert to pre-op snapshot and mark failed.
                # Non-critical (lint, tests) are annotated but do NOT call mark_failed.

                # For create_file / overwrite_file ops: no pre-op snapshot exists
                # (the file didn't exist before), so the normal revert loop below
                # can't clean up. Remove the broken file so a replan can recreate it.
                _is_create_or_overwrite = operation.kind in (
                    OperationKind.CREATE_FILE, OperationKind.OVERWRITE_FILE,
                )
                if _is_create_or_overwrite and operation.path:
                    _target = (
                        operation.path if os.path.isabs(operation.path)
                        else os.path.join(repo_root, operation.path)
                    )
                    if os.path.isfile(_target):
                        try:
                            os.remove(_target)
                            logger.info("Quality gate critical: removed broken create_file artifact %s", _target)
                        except OSError as _rm_e:
                            logger.warning("Quality gate critical: failed to remove %s: %s", _target, _rm_e)
                            # Surface so the orchestrator knows a broken artifact still
                            # lingers on disk (a replan CREATE_FILE would otherwise collide).
                            self._cb("quality_gate_cleanup_failed", {
                                "operation_id": operation.id,
                                "cleanup_type": "remove_artifact",
                                "file": _target,
                                "error": str(_rm_e),
                            })
                _snap = getattr(state, 'pre_op_file_snapshots', {}).get(operation.id, {})
                for _snap_path, _snap_content in _snap.items():
                    try:
                        atomic_write_text(_snap_path, _snap_content)
                        logger.info("Quality gate critical: reverted %s to pre-op state", _snap_path)
                        self._cb("quality_gate_file_reverted", {
                            "operation_id": operation.id,
                            "file": _snap_path,
                        })
                    except Exception as _snap_e:
                        logger.warning("Quality gate critical: revert failed for %s: %s", _snap_path, _snap_e)
                        self._cb("quality_gate_cleanup_failed", {
                            "operation_id": operation.id,
                            "cleanup_type": "revert_snapshot",
                            "file": _snap_path,
                            "error": str(_snap_e),
                        })
                state.mark_failed(operation.id, f"Quality gate critical failure: {failure_desc}")
                state.force_finish = True
                logger.warning(f"Critical quality gate failure for operation {operation.id}, stopping execution")
                self._cb("quality_gate_critical_failure", {
                    "operation_id": operation.id,
                    "failures": failures,
                })
            else:
                existing = state.completed_ops.get(operation.id)
                if isinstance(existing, dict):
                    existing["quality_gate_failures"] = failure_desc
                    existing["quality_gate_passed"] = False
                logger.warning(f"Quality gate non-critical failures for operation {operation.id}")
                self._cb("quality_gate_non_critical_failures", {
                    "operation_id": operation.id,
                    "failures": failures,
                })
        else:
            logger.info(f"Quality gate passed for operation {operation.id}")
            self._cb("quality_gate_passed", {
                "operation_id": operation.id,
            })

    def _on_operation_error(self, operation, state, error,
                            failure_class: str = "", initial_failure_class: str = ""):
        """Hook called when an operation fails."""
        self._cb("operation_error", {
            "operation_id": operation.id,
            "kind": operation.kind.value,
            "symbol": operation.symbol or "",
            "file_path": operation.path or "",
            "error": error,
            # Structured failure classification (for bench analytics / model routing)
            "failure_class": failure_class,
            "initial_failure_class": initial_failure_class,
        })

    def _record_tool_success(self, tool_name: str, tool_args: dict[str, Any]) -> None:
        """Record successful tool execution for adaptive routing."""
        key = self._tool_key(tool_name, tool_args)
        self._tool_success_memory[key] = self._tool_success_memory.get(key, 0) + 1
        if key in self._tool_fail_memory:
            del self._tool_fail_memory[key]
        # Adaptive tool-usage learning channel
        try:
            self._shared_run_store.record_tool_usage("MAIN_AGENT", tool_name, True, "")
        except Exception:
            pass  # non-critical — never block execution

    def _record_tool_failure(self, tool_name: str, tool_args: dict[str, Any]) -> None:
        key = self._tool_key(tool_name, tool_args)
        self._tool_fail_memory[key] = True
        # Adaptive tool-usage learning channel
        try:
            self._shared_run_store.record_tool_usage("MAIN_AGENT", tool_name, False, "")
        except Exception:
            pass  # non-critical — never block execution

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

        if tool_name in ("find_symbol", "get_project_info"):
            if content and len(content) > 20:
                definitive = True

        if definitive:
            # Use first 400 chars of tool result as the answer preview
            preview = (content[:400] + "…") if len(content) > 400 else content
            return AgentResult(
                status="success",
                turns=[],  # will be filled by caller
                final_message=preview,
                applied_patches=self.registry.applied_patches,
                metadata={
                    "readonly_early_finish": True,
                    "tool": tool_name,
                    "deterministic_answer": True,
                }
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
            r'[\w\./\-]+\.(?:py|js|ts|html|css|md|json|yaml|yml|toml|txt|ini|cfg|conf|kt|java|xml)',
            req
        )
        if not candidates:
            return ""

        from pathlib import Path

        repo_root = Path(str(self.registry.repo_root))

        for candidate in candidates:
            path = normalize_rel_path_fast(str(candidate))
            if not path:
                continue
            try:
                full_path = repo_root / path
                if full_path.exists() and full_path.is_file():
                    return path
            except (OSError, AttributeError):
                continue

        return ""

    def _extract_target_keywords(self, request: str) -> list[str]:
        """Extract target text keywords for keyword-miss detection (nudge model to search elsewhere)."""
        targets: list[str] = []

        for q in re.findall(r'["\u201c\u201d\u2018\u2019\'`]([^\'"` \u201c\u201d\u2018\u2019]{2,60})["\u201c\u201d\u2018\u2019\'`]', request):
            if q not in targets:
                targets.append(q)
            if len(targets) >= 3:
                break

        en = re.search(
            r'(?:change|replace|rename|update|convert)\s+["\']?([A-Za-z_\-\s]{2,40}?)["\']?\s+'
            r'(?:to|with|into|→)',
            request, re.IGNORECASE,
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
        self, request: str, context: str, route: Any,
        git_state: Any, session_id: str, is_local_model: bool,
        has_native_tools: bool, read_only_request: bool, known_target_file: str,
        target_keywords: list[str], tier: Any, plan: Optional[dict[str, Any]],
        plan_subtasks: list[dict[str, Any]], turns: list,
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
            known_target_file=known_target_file,
            target_keywords=target_keywords,
            tier=tier,
            plan=plan,
            plan_subtasks=plan_subtasks,
            turn_num=0,
            turns=turns,
        )

    def run(self, request: str, context: str = "", continuation_data: dict | None = None) -> AgentResult:
        self._continuation_data = continuation_data or getattr(self.config, 'continuation_data', None)
        if not hasattr(self, "state") or self.state is None:
            self.state = {}

        loaded_state = None
        if self.session_state is not None:
            loaded_state = self.session_state.load_state()

        if loaded_state:
            self.state['edit_history'] = loaded_state.get('edit_history', [])
            self.state['plan'] = loaded_state.get('plan', [])
            self.state['context'] = loaded_state.get('context', context)
            self.state['agent_phase'] = loaded_state.get('agent_phase', 'initial')
            self.state['tool_calls'] = loaded_state.get('tool_calls', [])
        _session_id = _new_session_id()

        _profile = getattr(self.config, 'agent_profile', None)
        if _profile is not None and hasattr(_profile, 'apply'):
            _profile.apply(self.config)
            logger.info("Agent profile applied: %s", _profile.name)

        self.performance_collector.session_id = _session_id
        self.performance_collector.start_session()

        route = getattr(self.config, 'route_decision', None)
        if route:
            _route_conf = float(getattr(route, 'confidence', 0.0))
            _route_lane = str(getattr(route, 'lane', '?'))
            _route_kind = str(getattr(route, 'task_kind', '?'))
            if _route_conf <= 0.10:
                logger.warning(
                    "Route confidence is suspiciously low (%.2f): lane=%s kind=%s reasoning=%s "
                    "target_specificity=%.2f — consider using a non-zero default",
                    _route_conf, _route_lane, _route_kind,
                    getattr(route, 'reasoning', '?'),
                    float(getattr(route, 'target_specificity_score', 0.0)),
                )
            logger.info(
                "Route applied: kind=%s lane=%s complexity=%s conf=%.2f",
                _route_kind, _route_lane,
                str(getattr(route, 'complexity', '?')),
                _route_conf,
            )
            self._cb("route_applied", {
                "task_kind": str(getattr(route, 'task_kind', '')),
                "lane": str(getattr(route, 'lane', '')),
                "confidence": float(getattr(route, 'confidence', 0.0)),
            })

        self.current_intent = "general"
        logger.debug(f"Session intent: {self.current_intent}")

        self._routing_intent_hint = self._resolve_routing_intent(route)

        read_only_request = is_non_edit_intent(self._routing_intent_hint)
        known_target_file = self._extract_known_file_path(request)
        _target_keywords: list[str] = self._extract_target_keywords(request)

        if self.config.stream_callback:
            self.config.stream_callback("routing_intent", {
                "intent": self._routing_intent_hint,
                "source": "intent_result",
            })

        self._agent_phase = "DISCOVER"
        self._phase_target_symbol = ""
        self._phase_target_file = known_target_file or ""

        if read_only_request:
            self._agent_phase = "DISCOVER"

        # Filesystem operations start in EDIT so bash is available immediately
        if route and hasattr(route, 'reasoning') and 'Filesystem operation' in (route.reasoning or ''):
            self._agent_phase = "EDIT"
            logger.info("Filesystem operation detected — starting in EDIT phase")

        git_state = self._record_git_state()
        turns: list[AgentTurn] = []

        is_local_model = self._is_local_model()
        has_native_tools = self._check_native_tool_support()

        # Context pre-fetch: PLANNER loads RAG; MAIN_AGENT/COMPACT use tools.
        _route_for_tier = route if route is not None else getattr(self.config, 'route_decision', None)
        tier = self._resolve_context_tier(_route_for_tier)
        self._context_tier = tier  # stored for _build_initial_messages
        logger.info("Context tier resolved: %s", tier)

        if tier == ContextTier.PLANNER and self.config.rag_enabled:
            rag_ctx = self._build_rag_context(request)
            if rag_ctx:
                context = (context + "\n\n" + rag_ctx) if context else rag_ctx

        # PLANNER lane: operation-based execution.
        # READ_ONLY uses the standard tool loop (write tools disabled).
        if route and route.lane == Lane.PLANNER:
            _planner_outcome = self._run_planner_lane(
                request=request,
                context=context,
                route=route,
                git_state=git_state,
                session_id=_session_id,
                turns=turns,
            )
            if _planner_outcome.result is not None:
                # NEW: clarification_needed → Design Chat reroute
                if (_planner_outcome.result.status == "clarification_needed"
                        and self.config.design_chat_reroute_enabled):
                    return self._handle_clarification_via_design_chat(
                        request=request,
                        context=context,
                        route=route,
                        git_state=git_state,
                        session_id=_session_id,
                        turns=turns,
                        planner_result=_planner_outcome.result,
                    )
                return _planner_outcome.result

            # Check for intentional fallback (e.g., hybrid init failed,
            # empty operation plan, targets not found).  These are signaled
            # via fallback_context — allow Design Chat to take over.
            _planner_fb = getattr(_planner_outcome, 'fallback_context', None)
            if _planner_fb and self.config.planner_fallthrough_enabled:
                logger.info(
                    "[PLANNER_FALLTHROUGH] PLANNER lane produced fallback_context — "
                    "running Design Chat tool loop: %s",
                    _planner_fb[:120],
                )
                context = (context + "\n\n" + _planner_fb) if context else _planner_fb
                # Run Design Chat tool loop for fallback exploration (lighter than MAIN_AGENT)
                from external_llm.agent.design_chat_loop import DesignChatLoop

                _dc_has_native = self._check_native_tool_support()
                _dc_msg_objs = self._build_initial_messages(
                    request, context, _dc_has_native, tier=ContextTier.MAIN_AGENT,
                )
                _dc_msgs = list(_dc_msg_objs)

                _dc_loop = DesignChatLoop(
                    self.llm_client, self.registry, self.model,
                    run_store=self._shared_run_store,
                )

                _stream_cb = getattr(self.config, 'stream_callback', None)
                _dc_result = _dc_loop.respond(
                    _dc_msgs,
                    stream_callback=_stream_cb,
                    max_tool_iterations=self.config.max_turns,
                    token_callback=self.config.make_token_callback(),
                )

                return AgentResult(
                    status="success" if not _dc_result.is_error else "partial_success",
                    final_message=_dc_result.content,
                    turns=turns or [],
                    applied_patches=self.registry.applied_patches,
                    metadata={
                        "session_id": _session_id,
                        "git_state": git_state,
                        "planner_fallthrough_to_design_chat": True,
                    },
                )
            elif _planner_fb and not self.config.planner_fallthrough_enabled:
                logger.info(
                    "[PLANNER_FALLTHROUGH_BLOCKED] planner_fallthrough_enabled=False — "
                    "returning partial_success with fallback_context suppressed: %s",
                    _planner_fb[:120],
                )
                # Block fallthrough — return partial_success with fallthrough_blocked signal.
                return AgentResult(
                    status="partial_success",
                    turns=turns or [],
                    final_message=(
                        "PLANNER lane produced a fallback_context but "
                        "planner_fallthrough_enabled=False.\n\n"
                        "The planner could not produce a valid operation plan.\n\n"
                        "Suggestions:\n"
                        "  - Rephrase your request with more specific file/symbol targets\n"
                        "  - Set planner_fallthrough_enabled=True to allow MAIN_AGENT fallback\n"
                        "  - Type 'continue' to retry with the current accumulated context"
                    ),
                    applied_patches=self.registry.applied_patches,
                    metadata={
                        "session_id": _session_id,
                        "git_state": git_state,
                        "planner_fallthrough_blocked": True,
                    },
                )
            else:
                # No result AND no fallback_context — this is unexpected.
                # Block MAIN_AGENT drift to prevent silent misbehavior.
                logger.warning(
                    "[PLANNER_DRIFT_GUARD] PLANNER lane completed without valid result "
                    "or fallback_context — blocking MAIN_AGENT drift fallthrough"
                )
                return AgentResult(
                    status="partial_success",
                    turns=turns or [],
                    final_message=(
                        "PLANNER lane reached end without a valid result.\n\n"
                        "The planner could not produce an operation plan — "
                        "possibly because target files or symbols were not found.\n\n"
                        "Suggestions:\n"
                        "  - Rephrase your request with more specific file/symbol targets\n"
                        "  - Type 'continue' to retry with the current accumulated context"
                    ),
                    applied_patches=self.registry.applied_patches,
                    metadata={
                        "session_id": _session_id,
                        "git_state": git_state,
                        "planner_drift_blocked": True,
                    },
                )

        # ── MAIN_AGENT lane: direct LLM tool-use loop ──
        # task_router always routes to MAIN_AGENT (PLANNER permanently disabled),
        # so this is the primary execution path. The PLANNER branch above is only
        # reached when route.lane == PLANNER, which never happens under the current
        # router. Only route=None or an unhandled lane falls through to the guard.
        if route and route.lane == Lane.MAIN_AGENT:
            logger.info("MAIN_AGENT lane: running direct LLM tool-use loop")
            ctx = self._build_turn_context(
                request, context, route, git_state, _session_id,
                is_local_model, has_native_tools,
                read_only_request, known_target_file, _target_keywords,
                tier, None, [], turns,
            )
            return self._run_llm_loop(ctx)

        logger.warning(
            "run() reached end without handling route.lane=%s — returning partial_success",
            str(getattr(getattr(route, 'lane', None), 'value', getattr(route, 'lane', None))),
        )
        return AgentResult(
            status="partial_success",
            turns=turns or [],
            final_message="No active lane handled this request.",
            applied_patches=self.registry.applied_patches,
            metadata={
                "session_id": _session_id,
                "git_state": git_state,
                "unhandled_lane": True,
            },
        )

    def _handle_clarification_via_design_chat(
        self,
        request: str,
        context: str,
        route: Any,
        git_state: Any,
        session_id: str,
        turns: Optional[list],
        planner_result: AgentResult,
    ) -> AgentResult:
        """Handle clarification_needed from planner by routing to Design Chat.

        Instead of returning raw AgentResult with LLM questions (which loses Design Chat
        context and falls through to CLI-only Q&A), this method routes the clarification
        through Design Chat which:
        1. Has the full conversation context
        2. Receives structured missing_spec_fields from planner
        3. Can ask targeted follow-up questions to the user
        4. Refines the implementation_spec
        5. Implements the refined spec directly using editing tools

        If design_chat_reroute does not resolve the ambiguity (user doesn't provide
        enough info), the original clarification_needed result is returned.
        """
        _llm_questions = planner_result.final_message or ""
        _missing_fields = planner_result.metadata.get("missing_spec_fields", [])
        _required_clarifications = planner_result.metadata.get("required_clarifications", [])

        # Build structured clarification message for Design Chat
        _spec_feedback_parts = []
        if _missing_fields:
            _spec_feedback_parts.append(
                "The planner could not produce a valid operation plan "
                "because the following spec fields need attention:"
            )
            for _rc in _required_clarifications:
                _rc_field = _rc.get('field', '?')
                _rc_reason = _rc.get('reason', '')
                _rc_suggestion = _rc.get('suggestion', '')
                _spec_feedback_parts.append(
                    f"  - {_rc_field}: {_rc_reason}. Suggestion: {_rc_suggestion}"
                )

        _feedback_text = "\n".join(_spec_feedback_parts) if _spec_feedback_parts else ""

        _clarification_context = (
            f"[PLANNER_CLARIFICATION_NEEDED]\n"
            f"The planner encountered ambiguity in your request and could not proceed.\n"
            f"The developer LLM asked the following: {_llm_questions}\n\n"
            f"{_feedback_text}\n\n"
            f"[YOUR TASK]\n"
            f"Analyze what went wrong. Review the original request and the conversation context.\n"
            f"If you can refine the implementation_spec yourself (e.g., infer missing target_files\n"
            f"from the conversation), implement it directly using the editing tools.\n"
            f"If you need the user to provide additional information, use your tools to ask the user.\n"
            f"Do NOT retry with the same spec — that will reproduce the same error."
        )

        # Build Design Chat messages with clarification context
        from external_llm.agent.design_chat_loop import DesignChatLoop

        _dc_has_native = self._check_native_tool_support()
        _dc_msg_objs = self._build_initial_messages(
            request, _clarification_context, _dc_has_native, tier=ContextTier.MAIN_AGENT,
        )
        _dc_msgs = list(_dc_msg_objs)

        _dc_loop = DesignChatLoop(
            self.llm_client, self.registry, self.model,
            run_store=self._shared_run_store,
        )

        _stream_cb2 = getattr(self.config, 'stream_callback', None)
        _dc_result = _dc_loop.respond(
            _dc_msgs,
            stream_callback=_stream_cb2,
            max_tool_iterations=self.config.max_turns,
            token_callback=self.config.make_token_callback(),
            mode='code',
        )

        if _dc_result.is_error:
            logger.warning(
                "[DC_CLARIFICATION_REROUTE] Design Chat returned error — "
                "falling back to original clarification_needed result"
            )
            return planner_result

        # Check if Design Chat called switch_to_planner (spec was refined)
        _dc_switch = getattr(_dc_result, 'agent_switch', None) or {}
        _dc_switch_req = _dc_switch.get('request', '')
        _dc_switch_spec = _dc_switch.get('implementation_spec')

        if _dc_switch_req and _dc_switch_spec:
            logger.info(
                "[DC_CLARIFICATION_REROUTE] Design Chat refined the spec — "
                "re-running planner with improved implementation_spec"
            )
            # Build prebuilt spec from refined implementation_spec
            from .agent_planner_pipeline import build_prebuilt_spec_from_impl_spec
            _refined_spec = build_prebuilt_spec_from_impl_spec(
                request_text=_dc_switch_req,
                implementation_spec=_dc_switch_spec,
                source="design_chat_clarification_reroute",
                repo_root=str(self.registry.repo_root) if hasattr(self.registry, 'repo_root') else '',
            )
            if _refined_spec is not None:
                # Set the refined spec on config so _run_planner_lane picks it up
                self.config.prebuilt_spec_for_planner = _refined_spec
                # Re-run planner lane with improved spec
                logger.info(
                    "[DC_CLARIFICATION_REROUTE] Re-running planner with refined spec: "
                    "target_files=%s target_symbols=%s",
                    _refined_spec.target_files, _refined_spec.target_symbols,
                )
                _rerun_outcome = self._run_planner_lane(
                    request=_dc_switch_req,
                    context=context,
                    route=route,
                    git_state=git_state,
                    session_id=session_id,
                    turns=turns,
                )
                if _rerun_outcome.result is not None:
                    # If still clarification_needed, return with updated metadata
                    if _rerun_outcome.result.status == "clarification_needed":
                        _rerun_outcome.result.metadata["reroute_attempted"] = True
                        _rerun_outcome.result.metadata["original_clarification"] = _llm_questions
                    return _rerun_outcome.result
                # Fallthrough from re-run planner → return DC result

            logger.warning(
                "[DC_CLARIFICATION_REROUTE] Refined spec produced None — "
                "returning DC result"
            )

        # Design Chat did not switch to planner (no switch_to_planner call)
        # Return combined result: DC's analysis + planner's original question
        return AgentResult(
            status="clarification_needed",
            turns=turns or [],
            final_message=(
                f"[Design Chat Clarification]\n"
                f"{_dc_result.content}\n\n"
                f"[Planner's Original Question]\n"
                f"{_llm_questions}"
            ),
            applied_patches=self.registry.applied_patches,
            metadata={
                **planner_result.metadata,
                "reroute_to_design_chat": True,
                "dc_reroute_completed": True,
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
        result: "AgentResult",
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
                "touched_files": list({
                    f for t in result.turns
                    if t.tool_result
                    for f in ((t.tool_result.metadata or {}).get("touched_files") or [])
                }),
                "tokens": {
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                    "total": prompt_tokens + completion_tokens,
                    "cost_usd": round(estimate_cost(provider, prompt_tokens, completion_tokens, model=self.model), 6),
                    "cache_adjusted_cost_usd": round(
                        estimate_cache_adjusted_cost(
                            provider, prompt_tokens, completion_tokens, cache_read_tokens,
                            cache_creation_tokens,
                            model=self.model, base_url=getattr(self.llm_client, 'base_url', ''),
                        ), 6,
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
            _SESSION_LOG_ROTATE_BYTES = 10 * 1024 * 1024
            with cross_process_flock(_Path(log_dir) / "sessions.lock"):
                # Rotate when the log exceeds 10 MB (single generation, like
                # worker.log rotation in orchestrator.py).  Best-effort: any
                # OSError is silently ignored — the session log is advisory.
                try:
                    if os.path.isfile(log_path) and os.path.getsize(log_path) > _SESSION_LOG_ROTATE_BYTES:
                        os.replace(log_path, log_path + ".1")
                except OSError:
                    pass
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
                enriched_data.update({
                    "agent_turn_num": len(self.turns) if hasattr(self, 'turns') else 0,
                    "event_timestamp": time.time(),
                    "orchestrator_phase": getattr(self, '_orchestrator_phase', 'standalone'),
                    "session_id": getattr(self.config, 'session_id', 'unknown'),
                })

                # Add global sequence if missing
                if "global_sequence_id" not in enriched_data:
                    # Use nanosecond timestamp — maintain consistency with orchestrator
                    enriched_data["global_sequence_id"] = int(time.time_ns())

                self.config.stream_callback(event, enriched_data)
            except Exception as e:
                logger.warning(f"Error in _cb callback: {e}")

    # LLM call strategies

    def _check_native_tool_support(self) -> bool:
        """Check if the provider supports native tool calling."""
        provider = self._get_provider_name()
        return provider in _NATIVE_TOOL_PROVIDERS and hasattr(self.llm_client, "chat_with_tools")

    def _llm_call_with_tools(
        self,
        messages: list[LLMMessage],
        read_only_request: bool = False,
        max_tokens: Optional[int] = None,
        token_callback: Optional[Callable] = None,
    ) -> dict[str, Any]:
        """Call LLM with native tool support."""
        # Check for cancellation before starting LLM call
        if self.config.cancel_event and self.config.cancel_event.is_set():
            raise AgentCancelled("cancelled by user before LLM call")

        # Pre-flight: fit messages to budget
        if self._context_budget:
            _before_fit = len(messages)
            messages = self._context_budget.fit_messages(messages)
            if len(messages) < _before_fit:
                self._cb("agent_working", {
                    "reason": "context_compressed",
                    "kept": len(messages),
                    "dropped": _before_fit - len(messages),
                })

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
                "_llm_call_with_tools: repair_tool_message_sequence removed %d orphaned "
                "messages", _dropped_count
            )
            self._cb("agent_working", {
                "reason": "tool_message_repair",
                "dropped": _dropped_count,
            })

        # Tool schemas are serialised into the prompt, so build them before the
        # token guard below so it can account for their size.
        tool_schemas = self.registry.get_tool_schemas(
            read_only_request=read_only_request,
            lang_filter=self.registry.repo_language,
        )

        _est_tokens: int | None = None

        # Pre-flight: hard input token guard — prevents HTTP 400 from API providers
        # when accumulated context exceeds the model's context window (e.g. DeepSeek 1M).
        # preemptive_trim is a last-resort safety net; normal context management is
        # handled by the sliding-window compressor in context_manager.py.
        if self._context_budget:
            _ctx_limit = _resolve_context_limit(self.model)
            _safety_margin = config.tokens.CONTEXT_HARD_CAP_SAFETY_MARGIN
            # Reserve output room AND account for tool-schema tokens (see helper):
            # a full prompt on small windows (Ollama 8192) leaves 0 to generate.
            _cap = context_message_cap(_ctx_limit, _safety_margin, tool_schemas)
            _est_tokens = estimate_tokens_from_msgs(messages)
            if _est_tokens > _cap:
                _before = len(messages)
                messages = preemptive_trim(messages, max_tokens=_cap, preserve_last=2)
                _after = len(messages)
                logger.warning(
                    "[CONTEXT_HARD_CAP] estimated %d tokens > cap %d "
                    "(limit=%d, reserved %d for output/tool-schemas); preemptive_trim: %d→%d messages",
                    _est_tokens, _cap, _ctx_limit, _ctx_limit - _cap,
                    _before, _after,
                )
                self._cb("agent_working", {
                    "reason": "context_hard_cap",
                    "estimated": _est_tokens,
                    "cap": _cap,
                    "kept": _after,
                    "dropped": _before - _after,
                })

                # Re-run tool message repair after trim — trim may have broken
                # assistant(tool_calls)↔tool pairings by slicing in the middle
                # of an exchange group, leaving orphaned tool messages.
                from .context_budget import repair_tool_message_sequence
                _before_repair2 = len(messages)
                messages = repair_tool_message_sequence(messages)
                _repair_dropped = _before_repair2 - len(messages)
                if _repair_dropped:
                    logger.info(
                        "_llm_call_with_tools: post-trim repair removed %d orphaned "
                        "messages", _repair_dropped
                    )

                # P2: Recalculate _est_tokens after trim — the pre-trim estimate
                # would cause _record_context_overflow to compute a stale (too high)
                # override, defeating 1-shot convergence.
                _est_tokens = estimate_tokens_from_msgs(messages)

        _max_tokens = max_tokens if max_tokens is not None else config.tokens.AGENT_TOOL_CALL

        def _replace_tool_calls(resp, calls: list):
            """Replace tool_calls on a response dict or object, return the response."""
            if isinstance(resp, dict):
                resp["tool_calls"] = calls
            else:
                try:
                    resp.tool_calls = calls
                except (AttributeError, TypeError):
                    pass
            return resp

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
                        if self.config.stream_callback else None
                    ),
                    token_callback=token_callback,
                    **_reasoning_kwargs,
                )
                _finish_reason = getattr(response, "finish_reason", None)
                if _finish_reason == "length" and _attempt < 2:
                    _attempt += 1
                    _max_tokens = _base * (1 << _attempt)
                    logger.warning(
                        "[llm_retry] finish_reason=length (max_tokens=%d), retrying (%d/3)",
                        _max_tokens, _attempt + 1,
                    )
                    continue
                if _finish_reason == "length":
                    logger.error(
                        "[llm_retry] finish_reason=length after 3 attempts "
                        "(max_tokens=%d), truncated response — clearing tool calls",
                        _max_tokens,
                    )
                    # Clear tool calls: a truncated tool call executes stale/partial
                    # arguments (e.g. a bash command cut mid-way). The text
                    # content (even if partial) is preserved so the turn loop can
                    # continue naturally.
                    _tool_calls_cleared = len(getattr(response, "tool_calls", []) or [])
                    response = _replace_tool_calls(response, [])
                    if _tool_calls_cleared:
                        logger.warning(
                            "Cleared %d truncated tool call(s) from finish_reason=length response",
                            _tool_calls_cleared,
                        )
                break
            tool_calls = getattr(response, "tool_calls", []) or []
            # Normalize to list of dicts
            normalized_calls = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    normalized_calls.append(tc)
                else:
                    normalized_calls.append({
                        "id": getattr(tc, "call_id", ""),
                        "name": getattr(tc, "name", ""),
                        "args": getattr(tc, "args", {}),
                    })
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

        def _re_trim_context_overflow() -> bool:
            """Re-trim messages after a context-override reduction (in-turn recovery).

            Called by _retry_on_rate_limit after recording a context-length 400 override.
            Reassigns ``messages`` in the enclosing scope so ``call_llm`` picks up the
            reduced message list on the next retry.

            Returns:
                True if trim actually reduced the estimated token count (caller should
                continue the retry loop).  False when no progress was made (reduction cap
                reached or last-2 messages already exceed the new cap) — caller should
                fall through and let the original 400 error propagate.
            """
            nonlocal messages
            _before_est = estimate_tokens_from_msgs(messages)
            _new_limit = _resolve_context_limit(self.model)
            _new_cap = context_message_cap(_new_limit, config.tokens.CONTEXT_HARD_CAP_SAFETY_MARGIN, tool_schemas)
            messages = preemptive_trim(messages, max_tokens=_new_cap, preserve_last=2)
            from .context_budget import repair_tool_message_sequence
            messages = repair_tool_message_sequence(messages)
            _after_est = estimate_tokens_from_msgs(messages)
            _reduced = _after_est < _before_est
            logger.info(
                "[CONTEXT_OVERFLOW_RETRY] %s — re-trimmed to %d messages "
                "(new cap=%d, before_est=%s, after_est=%s, reduced=%s)",
                self.model, len(messages), _new_cap,
                _before_est, _after_est, _reduced,
            )
            return _reduced

        return self._retry_on_rate_limit(call_llm, "native tool calling",
            _estimated_prompt_tokens=_est_tokens,
            overflow_retry_cb=_re_trim_context_overflow)

    @staticmethod
    def _repair_json_brackets(text: str) -> str:
        """Delegate to shared :func:`repair_json_brackets`."""
        return repair_json_brackets(text)

    def _try_parse_json(self, text: str) -> "Optional[Any]":
        """Try to parse JSON with 3-tier repair via shared :func:`try_parse_json`."""
        return try_parse_json(text)

    def _retry_on_rate_limit(
        self,
        callable_func: Callable[[], dict[str, Any]],
        mode: str = "",
        _estimated_prompt_tokens: int | None = None,
        overflow_retry_cb: Callable[[], bool] | None = None,
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
            **extra: Any,
        ) -> None:
            """Common retry-&exhausted handler for LLMConnectionError / LLMRateLimitError."""
            mode_str = f" in {mode}" if mode else ""
            if attempt < max_retries:
                logger.warning(
                    "%s hit%s (attempt %d/%d), retrying in %d seconds",
                    type(e).__name__, mode_str, attempt + 1, max_retries, delay,
                )
                self._cb(event_name, {
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "delay": delay,
                    "message": event_message,
                })
                if self.config.cancel_event and self.config.cancel_event.is_set():
                    raise AgentCancelled("cancelled by user during retry wait")
                if self.config.cancel_event:
                    self.config.cancel_event.wait(timeout=delay)
                else:
                    time.sleep(delay)
            else:
                logger.error(
                    "%s after %d attempts%s, giving up",
                    type(e).__name__, max_retries, mode_str,
                )
                payload: dict[str, Any] = {
                    "message": f"{type(e).__name__} after {max_retries} attempts{mode_str}: {e}",
                    "error_type": error_type,
                }
                if extra:
                    payload.update(extra)
                self._cb("error", payload)
                raise e

        max_retries = 3
        retry_delays = [10, 20, 40]  # Exponential backoff: 10s, 20s, 40s

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

                # Record LLM call metrics
                self.performance_collector.record_llm_call(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    execution_time_ms=execution_time_ms,
                    failed=False
                )

                # Diagnostic: log reasoning_tokens when available (DeepSeek)
                # so output-token bloat vs reasoning-token bloat is distinguishable.
                _rt = _to_int(result.get("reasoning_tokens"))
                if _rt:
                    logger.debug(
                        "[TOKEN_BREAKDOWN] completion=%d reasoning=%d visible=%d",
                        completion_tokens, _rt, max(0, completion_tokens - _rt),
                    )

                return result

            except LLMConnectionError as e:
                # Clamp index: loop runs max_retries+1 times but retry_delays
                # only has max_retries entries. Without this, the final
                # exhaustion iteration triggers retry_delays[attempt] IndexError,
                # which masks the original connection error.
                delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                _handle_retry_error(
                    e, attempt, max_retries, delay,
                    "connection_retry", "connection",
                    f"Connection error, retrying in {delay}s...",
                )
            except LLMRateLimitError as e:
                # Honor the server's Retry-After hint when present (providers
                # that don't retry internally surface it on the exception);
                # otherwise fall back to fixed exponential backoff.
                _hint = getattr(e, "retry_after", None)
                delay = _hint if isinstance(_hint, int) and _hint > 0 else retry_delays[min(attempt, len(retry_delays) - 1)]
                _handle_retry_error(
                    e, attempt, max_retries, delay,
                    "rate_limit_retry", "rate_limit",
                    f"Rate limit hit, retrying in {delay}s...",
                    retries_exhausted=True,
                )

            except LLMServerUnavailableError as e:
                delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                _handle_retry_error(
                    e, attempt, max_retries, delay,
                    "server_retry", "server_unavailable",
                    f"Server unavailable, retrying in {delay}s...",
                )

            except Exception as e:
                # Context-length 400 (HTTP 400, "context length exceeded") —
                # record overflow override so subsequent calls pre-trim at the
                # corrected limit instead of hitting the same error repeatedly.
                if _is_context_length_error(e):
                    _record_context_overflow(self.model, estimated_prompt_tokens=_estimated_prompt_tokens)
                    logger.warning(
                        "[CONTEXT_OVERFLOW] %s — recorded overflow override (est=%s)",
                        self.model, _estimated_prompt_tokens,
                    )
                    # In-turn recovery: re-trim messages and continue the retry loop
                    # so the same attempt is retried with the corrected limit instead
                    # of re-raising immediately.
                    if overflow_retry_cb is not None and attempt < max_retries:
                        if overflow_retry_cb():  # True = trim made progress
                            continue
                # For other exceptions, send SSE error event and re-raise immediately
                self._cb("error", {
                    "message": f"LLM API error: {e}",
                    "error_type": "api",
                })
                raise

    # Message management
    # ------------------------------------------------------------------

    def _auto_repair_apply_patch_args(self, args: dict, result: ToolResult) -> Optional[dict]:
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
        if ("--- a/" in patch_lower and "+++ b/" in patch_lower and "diff --git" not in patch_lower and
            patch_lower.count("--- a/") == 1 and patch_lower.count("+++ b/") == 1):
            # Extract path from --- a/ line (first occurrence)
            a_match = re.search(r'^--- a/(.+)$', patch, re.MULTILINE)
            b_match = re.search(r'^\+\+\+ b/(.+)$', patch, re.MULTILINE)
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
        try:
            if (
                not result.ok and tool_name == "write_plan"
            ):
                err = str(result.error or content or "")
                err_lower = err.lower()
                advice = []
                if "block not found" in err_lower:
                    advice.append(
                        "BLOCK NOT FOUND: Use find_symbol to locate the exact function/method, "
                        "then bash (cat) at the returned line. Copy the EXACT text into 'before'."
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
        except Exception:
            pass
        return content

    def _append_edit_warnings_guidance(self, content: str, tool_name: str, result: ToolResult) -> str:
        """Append edit_file warnings from result metadata as guidance text.

        Detects:
          - edit_warnings (anchor fuzzy-match, content/anchor inversion, op-type mismatch)
          - syntax_check (post-write syntax validation failures, pre-Fix 1 rollbacks)
        and appends actionable guidance so the LLM can correct its next invocation.
        """
        try:
            edit_warnings = (result.metadata or {}).get("edit_warnings")
            if edit_warnings and isinstance(edit_warnings, list) and edit_warnings:
                guidance_lines = ["\n\n[EDIT FILE WARNINGS]"]
                for w in edit_warnings:
                    guidance_lines.append(f"- {w}")
                # General tip for the most common warning type
                if any(
                    "replace op(s) resulted in" in w or "op-type mismatch" in w
                    for w in edit_warnings
                ):
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
                    for e in _err_list[:5]:  # Limit to first 5 errors to avoid bloat
                        _syn_lines.append(
                            f"- line {e.get('line')}:{e.get('col')} \u2014 {e.get('message', '').strip()}"
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
        except Exception:
            pass
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
        """
        diags = []
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

        # De-duplicate by (file, line, message) and keep error/warning only.
        seen = set()
        filtered = []
        _total_count = 0
        _n_err_total = 0
        _n_warn_total = 0
        for d in diags:
            if not isinstance(d, dict):
                continue
            _sev = (d.get("severity") or "error").lower()
            if _sev not in ("error", "warning"):
                continue
            _key = (d.get("file_path", ""), d.get("line"), d.get("message", ""))
            if _key in seen:
                continue
            seen.add(_key)
            _total_count += 1
            if _sev == "error":
                _n_err_total += 1
            else:
                _n_warn_total += 1
            if len(filtered) >= 15:  # cap to avoid context bloat
                continue  # still count, but don't add to filtered
            filtered.append(d)

        if not filtered:
            return content

        _n_err = _n_err_total
        _n_warn = _n_warn_total
        _suppressed = _total_count - len(filtered)
        lines = ["\n\n<file_diagnostics>"]
        lines.append(
            f"Semantic check found {_total_count} unique issue(s) "
            f"({_n_err} error, {_n_warn} warning), showing {len(filtered)} below. "
            f"The edit was applied, but these may cause runtime failures — "
            f"consider fixing them next."
        )
        if _suppressed > 0:
            lines.append(
                f"... {_suppressed} more {'issues' if _suppressed > 1 else 'issue'} "
                f"suppressed (run the validator directly for full output)"
            )
        for d in filtered:
            _sev = (d.get("severity") or "error").lower()
            _tag = "Error" if _sev == "error" else "Warn"
            _loc = ""
            if d.get("line") is not None:
                _col = d.get("column") or d.get("col")
                _loc = f":{d.get('line')}" + (f":{_col}" if _col else "")
            _file = d.get("file_path", "") or ""
            _file_short = _file.rsplit("/", 1)[-1] if _file else ""
            _code = d.get("code")
            _code_str = f" [{_code}]" if _code else ""
            lines.append(
                f"{_tag}: {_file_short}{_loc}{_code_str} {d.get('message', '').strip()}"
            )
        lines.append("</file_diagnostics>")
        return (content or "") + "\n".join(lines)

    def _append_patch_retry_guidance(self, content: str, tool_name: str, result: ToolResult) -> str:
        """Append apply_patch retry guidance from result metadata."""
        try:
            retry_guidance = (result.metadata or {}).get("retry_guidance")
            if (
                retry_guidance and not result.ok
                and tool_name == "apply_patch"
            ):
                guidance_lines = ["\n\n[PATCH RETRY GUIDANCE]"]
                for key, label in [("failure_type", "Failure type"), ("target_file", "Target file"),
                                   ("hint", "Hint"), ("instruction", "Instruction")]:
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
        except Exception:
            pass
        return content



    def _build_tool_result_message(
        self, call_id: str, tool_name: str, result: ToolResult, tool_args: Optional[dict[str, Any]] = None
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
        raw_response_data = (
            raw_resp.raw_response
            if raw_resp and hasattr(raw_resp, "raw_response")
            else None
        )

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

        if provider in ("openai", "deepseek"):
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
                _filtered_tool_calls = [
                    tc for tc in assistant_tool_calls
                    if tc.get("id") in _executed_ids
                ]
                if not _filtered_tool_calls:
                    # All tool calls were filtered out; skip this turn's
                    # assistant+tool block entirely.  The caller already
                    # handles the should_continue case via phase_rule_messages,
                    # so this branch is a defensive safety net.  Still surface
                    # any strategy warnings so the model sees the feedback.
                    return messages + extra_msgs if extra_msgs else messages

            # Always append the assistant message first.
            messages = [*messages, LLMMessage(role="assistant", content=assistant_content, tool_calls=_filtered_tool_calls, reasoning_content=reasoning_content)]

            # Tool results need a preceding tool_calls block — otherwise
            # DeepSeek rejects assistant(no tool_calls) → tool with HTTP 400.
            if _filtered_tool_calls and tool_msgs:
                messages = messages + tool_msgs

            # Strategy warnings as a trailing user turn (valid after tool msgs).
            if extra_msgs:
                messages = messages + extra_msgs

        elif provider in ("anthropic", "zai"):
            # Anthropic: assistant content blocks + user tool_result blocks (keyed by tool_use_id).
            raw_blocks: Optional[list[dict[str, Any]]] = None
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
            messages = [*messages, LLMMessage(role="assistant", content=assistant_content, raw_content=raw_blocks), LLMMessage(role="user", content="", raw_content=tool_result_blocks)]

        elif provider == "google":
            # Gemini: model parts (with functionCall) + user functionResponse parts.
            raw_parts: Optional[list[dict[str, Any]]] = None
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
            messages = [*messages, LLMMessage(role="assistant", content=assistant_content, raw_content=raw_parts), LLMMessage(role="user", content="", raw_content=function_response_parts)]

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
            messages = messages + [
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
            messages = [*messages, LLMMessage(role="assistant", content=assistant_content), LLMMessage(role="user", content=tool_results_text + "\n\nContinue with the task.")]

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
